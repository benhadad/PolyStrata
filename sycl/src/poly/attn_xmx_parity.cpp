// attn_xmx_test.cpp - attn_xmx.hpp against an FP64 CPU reference (Inkling's unsplit attention for a prompt batch).
// Inputs are random; K and V are rounded to halves (or Q8_0 codes times fp16 scales), so the cache is exact in the
// reference.  Q, R and the table are floats that the kernel rounds to halves.  Tolerance is relative to the largest
// |reference output|.
#include "attn_xmx.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

struct Case {
    const char* name;
    int rows;
    long long pos;
    int kv_heads, window, ring, extent;
    bool q8k, q8v, sinks;
    bool timing;
};

static float h2f(uint16_t b) { return float(sycl::bit_cast<sycl::half>(b)); }
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const Case cases[] = {
        {"40 tokens, from 0, full", 40, 0, 8, 0, 0, 1024, false, false, true, false},
        {"100 tokens at 57, full, short table", 100, 57, 8, 0, 0, 64, false, false, true, false},
        {"100 tokens at 700, window 128, ring 512", 100, 700, 8, 128, 512, 1024, false, false, true, false},
        {"Q8_0 K and V, 70 tokens at 130", 70, 130, 8, 0, 0, 1024, true, true, true, false},
        {"Q8_0 K only, no sinks, 33 tokens", 33, 5, 4, 0, 0, 1024, true, false, false, false},
        {"17 tokens (a short tail), window 32", 17, 90, 8, 32, 0, 1024, false, true, true, false},
        {"1023 tokens from 0, full (timing)", 1023, 0, 8, 0, 0, 1024, false, false, true, true},
        {"1023 tokens from 0, window 512, ring 1024 (timing)", 1023, 0, 8, 512, 1024, 1024, false, false, true, true},
    };
    int bad = 0;
    for (const Case& c : cases) {
        std::mt19937 rng(12345);
        std::normal_distribution<float> nd(0.f, 1.f);
        const int dim = 128, per = 3, n_head = c.kv_heads * per;
        const long long npos = c.pos + c.rows;
        const long long slots = c.ring > 0 ? c.ring : npos;
        const size_t kvn = (size_t) slots * c.kv_heads * dim;
        std::vector<float> kf(kvn), vf(kvn);               // the cache as floats (exact)
        std::vector<sycl::half> kh(kvn), vh(kvn);
        std::vector<int8_t> kc(kvn), vc(kvn);
        std::vector<uint16_t> ks(kvn / 32), vs(kvn / 32);
        // positions in increasing order: a ring slot ends up holding the latest position
        auto fill = [&](std::vector<float>& f, std::vector<sycl::half>& h, std::vector<int8_t>& codes,
                        std::vector<uint16_t>& sc, bool q8) {
            for (long long p = 0; p < npos; ++p) {
                const long long slot = c.ring > 0 ? p % c.ring : p;
                for (int hd = 0; hd < c.kv_heads; ++hd) {
                    const size_t at = ((size_t) slot * c.kv_heads + hd) * dim;
                    if (q8) {
                        for (int b = 0; b < dim / 32; ++b) {
                            const float d = h2f(f2h(0.03f + 0.01f * std::fabs(nd(rng))));
                            sc[(at + b * 32) / 32] = f2h(d);
                            for (int i = 0; i < 32; ++i) {
                                const int code = std::max(-127, std::min(127, (int) std::lround(nd(rng) * 40.f)));
                                codes[at + b * 32 + i] = (int8_t) code;
                                f[at + b * 32 + i] = h2f(f2h((float) code * d)) == (float) code * d
                                                         ? (float) code * d : (float) code * d;
                            }
                        }
                    } else {
                        for (int i = 0; i < dim; ++i) {
                            const sycl::half x = sycl::half(nd(rng));
                            h[at + i] = x;
                            f[at + i] = float(x);
                        }
                    }
                }
            }
        };
        fill(kf, kh, kc, ks, c.q8k);
        fill(vf, vh, vc, vs, c.q8v);
        const size_t qn = (size_t) c.rows * n_head * dim;
        std::vector<float> qv(qn), rv((size_t) c.rows * n_head * 16), tab((size_t) c.extent * 16), sk((size_t) c.rows * n_head);
        for (auto& x : qv) x = nd(rng);
        for (auto& x : rv) x = 0.4f * nd(rng);
        for (auto& x : tab) x = 0.4f * nd(rng);
        for (auto& x : sk) x = nd(rng);
        const float scale = 1.0f / std::sqrt((float) dim);

        auto dev = [&](auto& v) { auto* p = sycl::malloc_device<std::remove_reference_t<decltype(v[0])>>(v.size(), q); q.memcpy(p, v.data(), v.size() * sizeof(v[0])).wait(); return p; };
        attn_xmx::Args a{};
        a.q = dev(qv);
        a.kc = c.q8k ? nullptr : (const sycl::half*) dev(kh);
        a.vc = c.q8v ? nullptr : (const sycl::half*) dev(vh);
        a.kcodes = c.q8k ? dev(kc) : nullptr;
        a.vcodes = c.q8v ? dev(vc) : nullptr;
        a.kscales = c.q8k ? dev(ks) : nullptr;
        a.vscales = c.q8v ? dev(vs) : nullptr;
        a.rel_r = dev(rv);
        const int ttld = attn_xmx::tt_ld(c.extent);
        std::vector<sycl::half> tt((size_t) 16 * ttld, sycl::half(0.f));
        for (int d = 0; d < c.extent; ++d)
            for (int j = 0; j < 16; ++j) tt[(size_t) j * ttld + attn_xmx::PADL + d] = sycl::half(tab[(size_t) d * 16 + j]);
        a.rel_tt = dev(tt);
        a.rel_extent = c.extent;
        a.sinks = c.sinks ? dev(sk) : nullptr;
        a.pos = c.pos;
        a.rows = c.rows;
        a.n_head = n_head;
        a.kv_heads = c.kv_heads;
        a.window = c.window;
        a.ring = c.ring;
        a.scale = scale;
        float* out = sycl::malloc_device<float>(qn, q);
        q.memset(out, 0xff, qn * sizeof(float)).wait();   // NaN fill: every row must be written
        a.out = out;
        attn_xmx::launch(a, q);
        q.wait_and_throw();
        std::vector<float> o(qn);
        q.memcpy(o.data(), out, qn * sizeof(float)).wait();

        // reference
        double worst = 0, big = 0;
        long long nan = 0;
        const int rows_check = c.timing ? 48 : c.rows;   // the timing cases check a spread of rows
        double refmax = 0;
        std::vector<double> ref;
        std::vector<int> idx;
        for (int ri = 0; ri < rows_check; ++ri) {
            const int t = c.timing ? (int) ((long long) ri * (c.rows - 1) / (rows_check - 1)) : ri;
            const long long own = c.pos + t;
            for (int h = 0; h < n_head; ++h) {
                const int g = h / per;
                const size_t th = (size_t) t * n_head + h;
                const long long kmin = c.window > 0 ? std::max(0LL, own - c.window + 1) : 0;
                std::vector<double> s;
                double mx = -1e300;
                for (long long k = kmin; k <= own; ++k) {
                    const long long slot = c.ring > 0 ? k % c.ring : k;
                    const float* kk = &kf[((size_t) slot * c.kv_heads + g) * dim];
                    double dot = 0;
                    for (int i = 0; i < dim; ++i) dot += (double) qv[th * dim + i] * kk[i];
                    double sc = dot * scale;
                    const long long d = own - k;
                    if (d < c.extent)
                        for (int j = 0; j < 16; ++j) sc += (double) rv[th * 16 + j] * tab[(size_t) d * 16 + j];
                    s.push_back(sc);
                    mx = std::max(mx, sc);
                }
                if (c.sinks) mx = std::max(mx, (double) sk[th]);
                double den = c.sinks ? std::exp((double) sk[th] - mx) : 0.0;
                std::vector<double> acc(dim, 0.0);
                for (long long k = kmin; k <= own; ++k) {
                    const double wgt = std::exp(s[k - kmin] - mx);
                    den += wgt;
                    const long long slot = c.ring > 0 ? k % c.ring : k;
                    const float* vv = &vf[((size_t) slot * c.kv_heads + g) * dim];
                    for (int i = 0; i < dim; ++i) acc[i] += wgt * vv[i];
                }
                for (int i = 0; i < dim; ++i) {
                    const double r = acc[i] / den;
                    ref.push_back(r);
                    idx.push_back((int) (th * dim + i));
                    refmax = std::max(refmax, std::fabs(r));
                }
            }
        }
        for (size_t i = 0; i < ref.size(); ++i) {
            const float got = o[idx[i]];
            if (!std::isfinite(got)) { ++nan; continue; }
            worst = std::max(worst, std::fabs((double) got - ref[i]));
        }
        const double rel = worst / refmax;
        const bool pass = nan == 0 && rel < 6e-3;
        std::printf("  %-52s worst %.2e of max |out| %.3f -> rel %.2e, %lld non-finite -> %s\n", c.name, worst, refmax, rel, nan,
                    pass ? "PASS" : "FAIL");
        if (!pass) ++bad;
        if (c.timing) {
            double best = 1e9;
            for (int it = 0; it < 6; ++it) {
                auto t0 = std::chrono::steady_clock::now();
                attn_xmx::launch(a, q);
                q.wait_and_throw();
                best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            std::printf("    time of the launch: best of 6 = %.3f ms (%d heads)\n", best, n_head);
        }
        for (const void* p : {(const void*) a.q, (const void*) a.kc, (const void*) a.vc, (const void*) a.kcodes, (const void*) a.vcodes,
                              (const void*) a.kscales, (const void*) a.vscales, (const void*) a.rel_r, (const void*) a.rel_tt,
                              (const void*) a.sinks, (const void*) out})
            if (p) sycl::free(const_cast<void*>(p), q);
    }
    std::printf(bad ? "FAILED\n" : "all OK\n");
    return bad ? 1 : 0;
}
