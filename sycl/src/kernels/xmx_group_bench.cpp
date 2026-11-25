// xmx_group_bench: iq_xmx_grouped as Engine::moe_xmx() calls it for a prompt batch of Inkling 2 Mini: 128 experts in
// launches of 16, rows an expert drawn like a router's (gamma, mean `mean_rows`), one matrix of N x K a launch (the
// gate/up tensor as one matrix of 2 * n_ff_exp rows, or the down).  Prints the time of a whole visit (8 launches) and
// the rate in TFLOPS.  The kernel's own switches (STRATA_XG_KS, STRATA_XG_SPEC, STRATA_XG_DBG) are read from the
// environment, so one build compares variants.  Types with a dequantizer (not NVFP4) are also checked against
// dequantize + oneMKL on the first launch.
//   xmx_group_bench [mean_rows=64] [experts=128] [reps=5]
#include "strata/kernels/iq_kernels.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }

int main(int argc, char** argv) {
    const int mean = argc > 1 ? std::atoi(argv[1]) : 64;
    const int experts = argc > 2 ? std::atoi(argv[2]) : 128;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;
    const int G = 16;   // experts a launch
    sycl::queue* q = &dpct::get_in_order_queue();
    dpct::blas::descriptor_ptr hd = new dpct::blas::descriptor();
    hd->set_queue(q);
    std::mt19937 rng(7);
    std::gamma_distribution<double> gd(1.5, mean / 1.5);
    std::vector<int32_t> cnt(experts);
    int64_t tot = 0;
    for (auto& c : cnt) { c = std::max(1, (int) gd(rng)); tot += c; }
    struct Case { int ty; int64_t N, K; const char* what; };
    const Case cases[] = {{18, 2304, 2304, "gate/up IQ3_XXS"}, {40, 2304, 2304, "gate/up NVFP4"}, {17, 2304, 2304, "gate/up IQ2_XS"},
                          {11, 2304, 2304, "gate/up Q3_K"},    {40, 2304, 1152, "down NVFP4"},    {20, 2304, 1152, "down IQ4_NL"}};
    const char* ks = std::getenv("STRATA_XG_KS");
    const char* sp = std::getenv("STRATA_XG_SPEC");
    const char* db = std::getenv("STRATA_XG_DBG");
    std::printf("rows %lld over %d experts (mean %d), STRATA_XG_KS=%s SPEC=%s DBG=%s\n", (long long) tot, experts, mean,
                ks ? ks : "-", sp ? sp : "-", db ? db : "-");
    int fails = 0;
    for (const Case& cs : cases) {
        const int64_t N = cs.N, K = cs.K;
        size_t mat_bytes;
        size_t blk_bytes, blk_elems;
        if (cs.ty == 40) { blk_elems = 64; blk_bytes = 36; }
        else if (cs.ty == 20) { blk_elems = 32; blk_bytes = strata::kernels::iq_row_bytes(cs.ty, 32); }
        else { blk_elems = 256; blk_bytes = strata::kernels::iq_row_bytes(cs.ty, 256); }
        mat_bytes = (size_t) (K / blk_elems) * blk_bytes * N;
        std::vector<uint8_t*> dev(experts);
        std::vector<uint8_t> blob(mat_bytes);
        for (int i = 0; i < experts; ++i) {
            for (auto& x : blob) x = (uint8_t) rng();
            if (cs.ty == 11)
                for (size_t o = blk_bytes - 2; o + 1 < blob.size(); o += blk_bytes) { blob[o] = 0x00; blob[o + 1] = 0x2C; }
            else if (cs.ty == 40)
                for (size_t o = 0; o + 3 < blob.size(); o += blk_bytes) for (int j = 0; j < 4; ++j) blob[o + j] = 0x30 + (rng() & 7);
            else
                for (size_t o = 0; o + 1 < blob.size(); o += blk_bytes) { blob[o] = 0x00; blob[o + 1] = 0x2C; }
            dev[i] = sycl::malloc_device<uint8_t>(mat_bytes, *q);
            q->memcpy(dev[i], blob.data(), mat_bytes);
        }
        std::vector<uint16_t> hx((size_t) tot * K);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& x : hx) x = f2h(nd(rng) * 0.25f);
        uint16_t* dx = sycl::malloc_device<uint16_t>(hx.size(), *q);
        q->memcpy(dx, hx.data(), hx.size() * 2);
        float* dy = sycl::malloc_device<float>((size_t) tot * N, *q);
        q->wait();
        auto visit = [&]() {
            int64_t done = 0;
            for (int e = 0; e < experts; e += G) {
                const uint8_t* a[G];
                int32_t c[G];
                int m = 0;
                int64_t rows = 0;
                for (; m < G && e + m < experts; ++m) { a[m] = dev[e + m]; c[m] = cnt[e + m]; rows += cnt[e + m]; }
                strata::kernels::iq_xmx_grouped(cs.ty, a, nullptr, c, m, dx + (size_t) done * K, dy + (size_t) done * N, (int) N,
                                                (int) K, q);
                done += rows;
            }
        };
        visit();
        q->wait();
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            visit();
            q->wait();
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const double tflops = 2.0 * (double) tot * N * K / (best * 1e-3) / 1e12;
        // the first launch against dequantize + oneMKL
        const char* verdict = "";
        if (cs.ty != 40) {
            int64_t rows0 = 0;
            for (int i = 0; i < G; ++i) rows0 += cnt[i];
            std::vector<float> a((size_t) rows0 * N), b((size_t) rows0 * N);
            q->memcpy(b.data(), dy, b.size() * 4).wait();
            uint16_t* dw = sycl::malloc_device<uint16_t>((size_t) N * K, *q);
            float* yr = sycl::malloc_device<float>((size_t) rows0 * N, *q);
            int64_t o = 0;
            for (int i = 0; i < G; ++i) {
                strata::kernels::iq_dequant_f16(cs.ty, dev[i], N * K, dw, q);
                const float one = 1.f, zero = 0.f;
                dpct::blas::gemm(hd, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, (int) N, cnt[i], (int) K, &one,
                                 dw, dpct::library_data_t::real_half, (int) K, dx + o * K, dpct::library_data_t::real_half, (int) K,
                                 &zero, yr + o * N, dpct::library_data_t::real_float, (int) N, dpct::compute_type::f32);
                o += cnt[i];
            }
            q->memcpy(a.data(), yr, a.size() * 4).wait();
            double e = 0, mx = 0;
            size_t bad = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                if (!std::isfinite(a[i])) continue;
                if (!std::isfinite(b[i])) { ++bad; continue; }
                e = std::max(e, (double) std::fabs(a[i] - b[i]));
                mx = std::max(mx, (double) std::fabs(a[i]));
            }
            const bool ok = bad == 0 && mx > 0 && e <= 2e-3 * mx;
            fails += !ok;
            static char buf[64];
            std::snprintf(buf, sizeof buf, " relerr %.1e %s", e / (mx > 0 ? mx : 1), ok ? "OK" : "FAIL");
            verdict = buf;
            sycl::free(dw, *q);
            sycl::free(yr, *q);
        }
        std::printf("  %-16s N %lld K %lld: %7.2f ms a visit, %5.1f TFLOPS%s\n", cs.what, (long long) N, (long long) K, best, tflops, verdict);
        for (auto p : dev) sycl::free(p, *q);
        sycl::free(dx, *q);
        sycl::free(dy, *q);
    }
    std::printf(fails ? "FAILED\n" : "done\n");
    return fails ? 1 : 0;
}
