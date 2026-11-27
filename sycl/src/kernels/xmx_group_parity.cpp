// xmx_group_parity: iq_xmx_grouped (the A770 port's fused decode + grouped XMX) against the engine's dequant +
// oneMKL per expert, on random expert blobs of each format the native packs use.  Also times both.
//   xmx_group_parity [mean_rows=160] [n=16]
#include "strata/kernels/iq_kernels.hpp"
#include "strata/poly/xmx_situ.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
#ifdef STRATA_XMX_SG16
static bool situ_matches(sycl::queue& q, int ty, const std::vector<const uint8_t*>& weights,
                         const std::vector<int32_t>& counts, const uint16_t* x, const float* gu, int K, int ff) {
    const int experts = (int) counts.size();
    std::vector<int32_t> ids;
    std::vector<float> scales(2 * experts);
    for (int e = 0; e < experts; ++e) {
        ids.insert(ids.end(), counts[e], experts - 1 - e);
        scales[e] = 0.25f + 0.0625f * e;
        scales[experts + e] = 1.125f - 0.03125f * e;
    }
    const size_t size = ids.size() * ff;
    auto* di = sycl::malloc_device<int32_t>(ids.size(), q);
    auto* ds = sycl::malloc_device<float>(scales.size(), q);
    auto* reference = sycl::malloc_device<uint16_t>(size, q);
    auto* fused = sycl::malloc_device<uint16_t>(size, q);
    q.memcpy(di, ids.data(), ids.size() * sizeof(int32_t));
    q.memcpy(ds, scales.data(), scales.size() * sizeof(float));
    q.fill(fused, uint16_t(0x7e00), size);
    constexpr float beta = 4.0f, linear_beta = 25.0f;
    const strata::kernels::XmxSitu s{ds, ds + experts, di, beta, linear_beta};
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> id) {
        const size_t i = id[0], row = i / ff, col = i % ff;
        const float g = gu[row * 2 * ff + col] * ds[di[row]];
        const float u = gu[row * 2 * ff + ff + col] * ds[experts + di[row]];
        reference[i] = sycl::bit_cast<uint16_t>(sycl::half(strata::poly::sycl_situ(g, u, beta, linear_beta)));
    });
    strata::kernels::iq_xmx_grouped_situ(ty, weights.data(), counts.data(), experts, x, fused, ff, K, s, &q);
    std::vector<uint16_t> expected(size), actual(size);
    q.memcpy(expected.data(), reference, size * sizeof(uint16_t));
    q.memcpy(actual.data(), fused, size * sizeof(uint16_t)).wait();
    sycl::free(di, q);
    sycl::free(ds, q);
    sycl::free(reference, q);
    sycl::free(fused, q);
    size_t changed = 0;
    for (size_t i = 0; i < size; ++i) if (expected[i] != actual[i]) ++changed;
    std::printf("type %2d fused SITU with separate gate/up scales: %zu of %zu FP16 values differ %s\n",
                ty, changed, size, changed ? "FAIL" : "OK");
    return changed == 0;
}
#endif

int main(int argc, char** argv) {
    const int mean = argc > 1 ? std::atoi(argv[1]) : 160;
    const int n = argc > 2 ? std::atoi(argv[2]) : 16;
    sycl::queue* q = &dpct::get_in_order_queue();
#ifdef STRATA_NO_DG2_XMX
    std::printf("xmx_group_parity: SKIP (the sub-group 8 / 8x8x16 kernels are not built for this AOT device)");
    return 0;
#else
#ifndef STRATA_XMX_SG16
    if (strata::intel_gpu_gen(q->get_device()) == strata::IntelGpuGen::Battlemage) {
        std::printf("xmx_group_parity: SKIP (this build has the sub-group 8 / 8x8x16 kernels; Battlemage needs the AOT bmg build)");
        return 0;
    }
#endif
#endif
    dpct::blas::descriptor_ptr hd = new dpct::blas::descriptor();
    hd->set_queue(q);
    std::mt19937 rng(3);
    std::vector<int32_t> cnt(n);
    std::gamma_distribution<double> gd(1.5, mean / 1.5);
    int64_t tot = 0;
    for (auto& c : cnt) { c = std::max(1, (int) gd(rng)); tot += c; }
    int fails = 0;
    struct Case { int ty; bool gu; int64_t K = 0, N = 0; };
    // the formats of the native packs' experts that the prompt path's fused XMX takes (prefill.cpp xmx2_fmt)
    // The explicit Inkling cases use one matrix per expert. Gate/up is 2304 x 2304 and down is 2304 x 1152.
    for (Case cs : {Case{18, true}, Case{21, true}, Case{22, true}, Case{23, true}, Case{20, false}, Case{42, false},
                    Case{17, false, 2304, 2304}, Case{11, false, 2304, 2304},
                    Case{8, false, 2304, 2304}, Case{8, false, 1152, 2304},
                    Case{40, false, 2304, 2304}}) {
        const int64_t K = cs.K ? cs.K : cs.gu ? 2560 : 640, N = cs.N ? cs.N : cs.gu ? 1280 : 2560, n_ff = 640;
        const int blk_elems = (cs.ty == 8 || cs.ty == 20) ? 32 : (cs.ty == 40 || cs.ty == 42) ? 64 : 256;
        const size_t blk_bytes = strata::kernels::iq_row_bytes(cs.ty, blk_elems);
        const size_t mat_bytes = strata::kernels::iq_row_bytes(cs.ty, K) * (cs.gu ? n_ff : N);
        if (blk_bytes == 0 || mat_bytes == 0) { std::printf("type %d: not supported\n", cs.ty); continue; }
        auto blob = [&]() {
            std::vector<uint8_t> b(mat_bytes);
            for (auto& x : b) x = (uint8_t) rng();
            // d = 0.0625 at each block's start (IQ1_M keeps its scale in the scales field; Q3_K's d is the block's last
            // two bytes)
            if (cs.ty == 11)
                for (size_t o = blk_bytes - 2; o + 1 < b.size(); o += blk_bytes) { b[o] = 0x00; b[o + 1] = 0x2C; }
            else if (cs.ty == 40) {
                for (size_t o = 0; o + 3 < b.size(); o += blk_bytes)
                    for (int scale = 0; scale < 4; ++scale) b[o + scale] = 0x30 + (rng() & 7);
            } else if (cs.ty == 8) {
                for (size_t o = 0; o + 1 < b.size(); o += blk_bytes) {
                    const uint16_t d = f2h(0.03125f * (1 + (rng() & 7)));
                    b[o] = (uint8_t) d;
                    b[o + 1] = (uint8_t) (d >> 8);
                }
            } else if (cs.ty != 29)
                for (size_t o = 0; o + 1 < b.size(); o += blk_bytes) { b[o] = 0x00; b[o + 1] = 0x2C; }
            return b;
        };
        std::vector<uint8_t*> dg(n), du(n);
        std::vector<const uint8_t*> pg(n), pu(n);
        for (int i = 0; i < n; ++i) {
            auto g = blob();
            dg[i] = sycl::malloc_device<uint8_t>(mat_bytes, *q);
            q->memcpy(dg[i], g.data(), mat_bytes).wait();
            pg[i] = dg[i];
            if (cs.gu) {
                auto u = blob();
                du[i] = sycl::malloc_device<uint8_t>(mat_bytes, *q);
                q->memcpy(du[i], u.data(), mat_bytes).wait();
                pu[i] = du[i];
            }
        }
        std::vector<uint16_t> hx((size_t) tot * K);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& x : hx) x = f2h(nd(rng) * 0.25f);
        uint16_t* dx = sycl::malloc_device<uint16_t>(hx.size(), *q);
        q->memcpy(dx, hx.data(), hx.size() * 2);
        uint16_t* dw = sycl::malloc_device<uint16_t>((size_t) N * K, *q);
        float* yr = sycl::malloc_device<float>((size_t) tot * N, *q);
        float* yx = sycl::malloc_device<float>((size_t) tot * N, *q);
        q->wait();
        auto ref = [&]() {
            if (cs.ty == 40) {   // xmx_nvfp4_parity independently checks these FP32 accumulators
                strata::kernels::iq_xmx_grouped(cs.ty, pg.data(), nullptr, cnt.data(), n, dx, yr, (int) N, (int) K, q);
                return;
            }
            int64_t o = 0;
            for (int i = 0; i < n; ++i) {
                if (cs.gu) strata::kernels::iq_dequant_gu_f16(cs.ty, dg[i], du[i], n_ff, K, dw, q);
                else strata::kernels::iq_dequant_f16(cs.ty, dg[i], N * K, dw, q);
                const float a = 1.f, b = 0.f;
                dpct::blas::gemm(hd, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, (int) N, cnt[i],
                                 (int) K, &a, dw, dpct::library_data_t::real_half, (int) K, dx + o * K,
                                 dpct::library_data_t::real_half, (int) K, &b, yr + o * N,
                                 dpct::library_data_t::real_float, (int) N, dpct::compute_type::f32);
                o += cnt[i];
            }
        };
        auto mine = [&]() {
            strata::kernels::iq_xmx_grouped(cs.ty, pg.data(), cs.gu ? pu.data() : nullptr, cnt.data(), n, dx, yx,
                                            (int) N, (int) K, q);
        };
        auto tm = [&](auto f) {
            f(); q->wait();
            const auto a = std::chrono::steady_clock::now();
            for (int r = 0; r < 10; ++r) f();
            q->wait();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count() / 10;
        };
        const double t_ref = tm(ref), t_x = tm(mine);
        std::vector<float> a((size_t) tot * N), b((size_t) tot * N);
        q->memcpy(a.data(), yr, a.size() * 4);
        q->memcpy(b.data(), yx, b.size() * 4).wait();
        double e = 0, mx = 0;
        size_t bad = 0, bad_ref = 0, mism = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(a[i])) { ++bad_ref; if (std::isfinite(b[i])) ++mism; continue; }
            if (!std::isfinite(b[i])) { ++bad; continue; }
            e = std::max(e, (double) std::fabs(a[i] - b[i]));
            mx = std::max(mx, (double) std::fabs(a[i]));
        }
        if (bad_ref) std::printf("  (reference non-finite %zu of %zu, fused finite there %zu)\n", bad_ref, a.size(), mism);
        const bool ok = bad == 0 && mism == 0 && mx > 0 && e <= 2e-3 * mx;
        fails += !ok;
        std::printf("type %2d N %lld K %lld rows %5lld: %s %7.3f ms | grouped XMX %7.3f ms (%.2fx)  relerr %.1e  %s\n",
                    cs.ty, (long long) N, (long long) K, (long long) tot, cs.ty == 40 ? "grouped reference" : "dequant+oneMKL",
                    t_ref, t_x, t_ref / t_x, e / (mx > 0 ? mx : 1), ok ? "OK" : "FAIL");
#ifdef STRATA_XMX_SG16
        if ((cs.ty == 8 || cs.ty == 40) && K == 2304 && N == 2304)
            fails += !situ_matches(*q, cs.ty, pg, cnt, dx, yx, (int) K, (int) (N / 2));
#endif
        for (int i = 0; i < n; ++i) { sycl::free(dg[i], *q); if (cs.gu) sycl::free(du[i], *q); }
        sycl::free(dx, *q); sycl::free(dw, *q); sycl::free(yr, *q); sycl::free(yx, *q);
    }
    std::printf("%s\n", fails ? "FAILED" : "all OK");
    return fails ? 1 : 0;
}
