// Standalone test: native_mmvq_ids and native_mmvq_slots (several matrices, one launch) against
// native_mmvq(type, w, x, y, n_in, n_out, 1) for each matrix, which is their documented contract (native_mmvq.hpp).
// Formats: the ones Inkling 2 Mini's experts use (NVFP4 and the K / IQ quants that go to iq_mmvq) plus Q8_0.
// Weights are random bytes.  A case passes when every finite output is bit-equal, or (for formats whose single-column
// kernel adds the terms in another order than the slots kernels) the largest difference is under 1e-4 of the largest
// output; NaN / Inf outputs are compared by class, because a NaN's payload depends on the order of the operations.
// A case where more than half of the reference outputs are not finite (random FP16 scales overflow float) is SKIPPED:
// there a finite-vs-infinite flip at the overflow edge says nothing about the kernels.
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
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

// Finite values must match bit for bit.  Random weight bytes give NaN / infinite outputs (random FP16 scales), and a NaN's
// sign and payload depend on the order of the operations that made it, so those are compared by class (NaN, +inf, -inf).
static double g_maxdiff = 0, g_scale = 0;   // largest |a-b| over the not-bit-equal finite pairs; largest finite |a|
static long g_notbit = 0, g_badclass = 0;    // finite pairs not bit-equal; pairs whose class (finite / NaN / Inf) differs
static int g_skipped = 0;                    // cases skipped because the random weights overflow
static bool same_bits(const std::vector<float>& a, const std::vector<float>& b, long& nonfinite, long& finite) {
    bool ok = true;
    for (size_t i = 0; i < a.size(); ++i) {
        const bool fa = std::isfinite(a[i]), fb = std::isfinite(b[i]);
        if (fa && fb) {
            ++finite;
            g_scale = std::fmax(g_scale, std::fabs((double) a[i]));
            if (std::memcmp(&a[i], &b[i], 4) != 0) {
                ok = false; ++g_notbit;
                g_maxdiff = std::fmax(g_maxdiff, std::fabs((double) a[i] - (double) b[i]));
            }
        }
        else {
            ++nonfinite;
            const bool both_nan = std::isnan(a[i]) && std::isnan(b[i]);
            const bool same_inf = std::isinf(a[i]) && std::isinf(b[i]) && (a[i] > 0) == (b[i] > 0);
            if (!both_nan && !same_inf) { ok = false; ++g_badclass; }   // finite vs non-finite, or NaN vs Inf: a real difference
        }
    }
    return ok;
}

static bool run(sycl::queue& q, const Fmt& f, int n_in, int n_out, int count, int x_every, bool lanes, uint32_t seed) {
    const int E = 16;                                   // experts in the table
    const size_t row_bytes = (size_t) (n_in / f.qk) * f.bytes, mat = row_bytes * n_out, offset = 512;
    const size_t stride = (mat + offset + 256 + 255) / 256 * 256;   // experts start 256-byte aligned, as the engine's do
    std::mt19937 rng(seed);
    std::vector<uint8_t> blob(E * stride);
    for (auto& b : blob) b = (uint8_t) (rng() & 0xFF);
    // Q8_0 and IQ4_NL have one FP16 scale first in every 32-value block: random scales overflow almost every output, so
    // give those blocks a scale of 2^-4 (0x2C00) and keep the random quants.
    if (f.type == 8 || f.type == 20)
        for (int e = 0; e < E; ++e)
            for (size_t blk = 0; blk < (size_t) n_out * (n_in / f.qk); ++blk) {
                uint8_t* p = blob.data() + (size_t) e * stride + offset + blk * f.bytes;
                p[0] = 0x00; p[1] = 0x2C;
            }
    // realistic scales where the format has an FP16 scale at a known place would help, but random bytes are enough
    // for a bit-for-bit comparison of two launches of the same arithmetic
    const int ncols = (count + x_every - 1) / x_every;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) ncols * n_in);
    for (auto& v : x) v = nd(rng);

    uint8_t* d_blob = sycl::malloc_device<uint8_t>(blob.size(), q);
    float* d_x = sycl::malloc_device<float>(x.size(), q);
    const size_t xq = k::native_q8_1_bytes(n_in, 1);
    uint8_t* d_xq = sycl::malloc_device<uint8_t>(xq * ncols, q);
    float* d_y = sycl::malloc_device<float>((size_t) count * n_out, q);
    float* d_ref = sycl::malloc_device<float>((size_t) count * n_out, q);
    const void** d_table = sycl::malloc_device<const void*>(E, q);
    int32_t* d_ids = sycl::malloc_device<int32_t>(count, q);
    q.memcpy(d_blob, blob.data(), blob.size());
    q.memcpy(d_x, x.data(), x.size() * 4);
    std::vector<const void*> table(E);
    for (int e = 0; e < E; ++e) table[e] = d_blob + (size_t) e * stride;
    std::vector<int32_t> ids(count);
    for (auto& i : ids) i = (int32_t) (rng() % E);       // duplicates allowed, as the router may pick one twice
    q.memcpy(d_table, table.data(), E * sizeof(void*));
    q.memcpy(d_ids, ids.data(), count * 4);
    q.wait();
    for (int c = 0; c < ncols; ++c) k::native_quantize_q8_1(d_x + (size_t) c * n_in, d_xq + c * xq, n_in, 1, (void*) &q);
    q.wait_and_throw();

    // the reference: every matrix on its own
    for (int j = 0; j < count; ++j)
        k::native_mmvq(f.type, (const uint8_t*) table[ids[j]] + offset, d_xq + (size_t) (j / x_every) * xq,
                       d_ref + (size_t) j * n_out, n_in, n_out, 1, (void*) &q);
    q.wait_and_throw();
    std::vector<float> ref((size_t) count * n_out), got(ref.size());
    q.memcpy(ref.data(), d_ref, ref.size() * 4).wait();
    {
        size_t bad = 0;
        for (float v : ref) if (!std::isfinite(v)) ++bad;
        if (bad * 2 > ref.size()) {
            std::printf("  %-8s n_in=%-5d n_out=%-4d count=%-2d x_every=%d lanes=%d  SKIP (%zu of %zu reference outputs not finite)\n",
                        f.name, n_in, n_out, count, x_every, (int) lanes, bad, ref.size());
            sycl::free(d_blob, q); sycl::free(d_x, q); sycl::free(d_xq, q); sycl::free(d_y, q); sycl::free(d_ref, q);
            sycl::free(d_table, q); sycl::free(d_ids, q);
            ++g_skipped;
            return true;
        }
    }

    bool ok = true; long nf = 0, fin = 0;
    g_maxdiff = g_scale = 0; g_notbit = 0; g_badclass = 0;
    // ids
    k::MmvqIds mi{};
    mi.table = d_table; mi.ids = d_ids; mi.offset = offset; mi.x = d_xq; mi.x_stride = xq; mi.x_every = x_every;
    mi.y = d_y; mi.y_stride = (size_t) n_out; mi.lanes = lanes;
    q.memset(d_y, 0xFF, got.size() * 4).wait();
    const bool supported = k::native_mmvq_ids_supported(f.type);
    bool ran_ids = false;
    if (supported) ran_ids = k::native_mmvq_ids(f.type, mi, count, n_in, n_out, (void*) &q);
    q.wait_and_throw();
    if (supported && ran_ids) { q.memcpy(got.data(), d_y, got.size() * 4).wait(); ok &= same_bits(ref, got, nf, fin); }
    else ok = false;
    const bool ids_ok = ok;                    // bit-exact (finite values) and ran
    const bool ids_ran_ok = supported && ran_ids;
    const long ids_notbit = g_notbit;
    // slots
    k::MmvqSlots ms{};
    for (int j = 0; j < count; ++j) {
        ms.w[j] = (const uint8_t*) table[ids[j]] + offset; ms.x[j] = d_xq + (size_t) (j / x_every) * xq;
        ms.y[j] = d_y + (size_t) j * n_out;
    }
    ms.lanes = lanes;
    q.memset(d_y, 0xFF, got.size() * 4).wait();
    const bool ran_slots = k::native_mmvq_slots(f.type, ms, count, n_in, n_out, (void*) &q);
    q.wait_and_throw();
    if (ran_slots) { q.memcpy(got.data(), d_y, got.size() * 4).wait(); ok &= same_bits(ref, got, nf, fin); } else ok = false;
    // Not bit-equal is acceptable only as a reordering of the same float sum: the largest difference is a small fraction
    // of the largest output.  (The port's single-column kernels for some formats are SIMD16 / wide variants that add the
    // terms in another order than the slots kernels, which are CUDA's EXACT layout.)
    const double rel = g_scale > 0 ? g_maxdiff / g_scale : 0;
    const bool close = ids_ran_ok && ran_slots && rel < 1e-4 && g_badclass == 0;
    std::printf("  %-8s n_in=%-5d n_out=%-4d count=%-2d x_every=%d lanes=%d  ids %s (%ld finite pairs not bit-equal) slots %s  "
                "compared finite %ld NaN/Inf %ld; max diff / max output = %.2e  %s\n",
                f.name, n_in, n_out, count, x_every, (int) lanes, ids_ok ? "bitwise" : (supported ? "DIFFERS" : "unsupported"),
                ids_notbit, ran_slots ? "ran" : "NO KERNEL", fin, nf, rel,
                ok ? "PASS (bitwise)" : (close ? "PASS (same sum, other order)" : "FAIL"));
    ok = ok || close;
    sycl::free(d_blob, q); sycl::free(d_x, q); sycl::free(d_xq, q); sycl::free(d_y, q); sycl::free(d_ref, q);
    sycl::free(d_table, q); sycl::free(d_ids, q);
    return ok;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int fail = 0, n = 0;
    uint32_t seed = 1;
    for (const Fmt& f : kFmts)
        for (int lanes = 0; lanes < 2; ++lanes) {
            ++n; if (!run(q, f, 2304, 100, 8, 2, lanes != 0, seed++)) ++fail;
        }
    // another shape: many matrices, n_out not a multiple of 4, the narrow K of a down projection
    for (const Fmt& f : kFmts) { ++n; if (!run(q, f, 768, 257, 24, 1, false, seed++)) ++fail; }
    std::printf("%s: %d of %d cases passed (%d of them skipped: random weights overflow)\n", fail ? "FAIL" : "PASS", n - fail, n,
                g_skipped);
    return fail ? 1 : 0;
}
