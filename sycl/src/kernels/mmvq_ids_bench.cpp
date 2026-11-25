// Benchmark: native_mmvq_ids at a decode call's shape (8 picks of a big expert table, one activation column, or one a
// pick), the kernel that runs the routed experts of a token.  Reports the time of a call and the weight bandwidth it
// reaches, with the picks drawn anew for every call so the table is read from DRAM, not from L2.
//   mmvq_ids_bench [type] [n_in] [n_out] [x_every] [picks] [experts] [calls]
// Defaults: NVFP4 (40), a gate/up matrix of Inkling 2 Mini (n_in 2304, n_out 2304), x_every 8 (one column for all the
// picks; use 1 for a down matrix: one column a pick), 8 picks, 128 experts, 200 calls.  Weights are random bytes, so
// the time is the kernel's, the outputs are only a checksum (the same inputs give the same checksum, which is how two
// builds of the kernel are compared: it must not change).
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "strata/kernels/native_mmvq.hpp"

namespace k = strata::kernels;

struct Fmt { int type; const char* name; int qk; int bytes; };
static const Fmt kFmts[] = {
    {40, "NVFP4", 64, 36}, {23, "IQ4_XS", 256, 136}, {12, "Q4_K", 256, 144}, {11, "Q3_K", 256, 110},
    {14, "Q6_K", 256, 210}, {8, "Q8_0", 32, 34}, {21, "IQ3_S", 256, 110}, {18, "IQ3_XXS", 256, 98},
    {17, "IQ2_XS", 256, 74}, {22, "IQ2_S", 256, 82}, {16, "IQ2_XXS", 256, 66}, {20, "IQ4_NL", 32, 18},
};

int main(int argc, char** argv) {
    const int type = argc > 1 ? std::atoi(argv[1]) : 40;
    const int n_in = argc > 2 ? std::atoi(argv[2]) : 2304;
    const int n_out = argc > 3 ? std::atoi(argv[3]) : 2304;
    const int x_every = argc > 4 ? std::atoi(argv[4]) : 8;
    const int picks = argc > 5 ? std::atoi(argv[5]) : 8;
    const int E = argc > 6 ? std::atoi(argv[6]) : 128;
    const int calls = argc > 7 ? std::atoi(argv[7]) : 200;
    const Fmt* f = nullptr;
    for (const Fmt& c : kFmts) if (c.type == type) f = &c;
    if (!f || n_in % f->qk) { std::fprintf(stderr, "bad type or n_in\n"); return 1; }
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order()}};
    const size_t row_bytes = (size_t) (n_in / f->qk) * f->bytes, mat = row_bytes * n_out, offset = 0;
    const size_t stride = (mat + 255) / 256 * 256;
    std::mt19937 rng(12345);
    std::vector<uint8_t> blob(E * stride);
    for (auto& b : blob) b = (uint8_t) (rng() & 0xFF);
    if (type == 8 || type == 20)   // FP16 scale of 2^-4 so the outputs stay finite
        for (size_t blk = 0; blk < (size_t) E * stride / f->bytes; ++blk) {
            const size_t p = blk * f->bytes;
            if (p + 2 <= blob.size()) { blob[p] = 0x00; blob[p + 1] = 0x2C; }
        }
    const int ncols = (picks + x_every - 1) / x_every;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) ncols * n_in);
    for (auto& v : x) v = nd(rng);
    uint8_t* d_blob = sycl::malloc_device<uint8_t>(blob.size(), q);
    float* d_x = sycl::malloc_device<float>(x.size(), q);
    const size_t xq = k::native_q8_1_bytes(n_in, 1);
    uint8_t* d_xq = sycl::malloc_device<uint8_t>(xq * ncols, q);
    float* d_y = sycl::malloc_device<float>((size_t) picks * n_out, q);
    const void** d_table = sycl::malloc_device<const void*>(E, q);
    int32_t* d_ids = sycl::malloc_device<int32_t>((size_t) (calls + 1) * picks, q);
    q.memcpy(d_blob, blob.data(), blob.size());
    q.memcpy(d_x, x.data(), x.size() * 4);
    std::vector<const void*> table(E);
    for (int e = 0; e < E; ++e) table[e] = d_blob + (size_t) e * stride;
    q.memcpy(d_table, table.data(), E * sizeof(void*));
    std::vector<int32_t> ids((size_t) (calls + 1) * picks);
    for (auto& i : ids) i = (int32_t) (rng() % E);
    q.memcpy(d_ids, ids.data(), ids.size() * 4);
    q.wait();
    for (int c = 0; c < ncols; ++c) k::native_quantize_q8_1(d_x + (size_t) c * n_in, d_xq + c * xq, n_in, 1, (void*) &q);
    q.wait_and_throw();
    if (!k::native_mmvq_ids_supported(type)) { std::fprintf(stderr, "type %d has no ids kernel\n", type); return 1; }
    const auto call = [&](int n) {
        k::MmvqIds mi{};
        mi.table = d_table; mi.ids = d_ids + (size_t) n * picks; mi.offset = offset; mi.x = d_xq; mi.x_stride = xq;
        mi.x_every = x_every; mi.y = d_y; mi.y_stride = (size_t) n_out; mi.lanes = false;
        if (!k::native_mmvq_ids(type, mi, picks, n_in, n_out, (void*) &q)) { std::fprintf(stderr, "ids kernel refused\n"); std::exit(1); }
    };
    // the checksum: the first call's outputs, as bits
    call(0);
    q.wait_and_throw();
    std::vector<float> y((size_t) picks * n_out);
    q.memcpy(y.data(), d_y, y.size() * 4).wait();
    uint64_t sum = 0, nonfin = 0;
    for (float v : y) { uint32_t b; std::memcpy(&b, &v, 4); sum = sum * 1099511628211ull + b; if (!std::isfinite(v)) ++nonfin; }
    for (int w = 0; w < 20; ++w) call(1 + w % calls);
    q.wait_and_throw();
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int c = 0; c < calls; ++c) call(1 + c % calls);
        q.wait_and_throw();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / calls;
        best = std::min(best, us);
    }
    const double bytes = (double) picks * n_out * row_bytes;
    std::printf("%-8s n_in %-5d n_out %-5d x_every %-2d picks %d: %7.1f us a call, %6.1f GB/s of weights (%.1f MB a call), checksum %016llx (%llu non-finite)\n",
                f->name, n_in, n_out, x_every, picks, best, bytes / best / 1e3, bytes / 1e6, (unsigned long long) sum,
                (unsigned long long) nonfin);
    return 0;
}
