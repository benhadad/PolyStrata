// Standalone test: NVFP4 (ggml type 40) through the grouped XMX GEMM (iq_xmx_grouped, the prompt batch's expert
// product) on the GPU vs an FP64 CPU reference.
//
// Reference (independent of the kernel's decoder): a 64-value block = 4 UE4M3 scale bytes + 32 bytes of E2M1 nibbles;
// in sub-block s (16 values) value j (0..7) is the low nibble of qs[8 s + j] and value j + 8 the high nibble.
// Value = e2m1(code) * ue4m3(d[s]); UE4M3 has no sign bit (bit 7 ignored) and 0x7F reads as 0.
// Y[row][o] = sum_k X[row][k] * W_e[o][k] for the rows of expert e (rows of the experts back to back, as the kernel
// takes them), X in FP16. The kernel decodes the weights to FP16 (exact: a 3-bit scale mantissa times a 3-bit code
// mantissa) and accumulates in FP32, so only summation order differs: tolerance 1e-5 of sum|terms|.
// -DNEG_CONTROL swaps the nibbles in the reference: every case must then FAIL.
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <random>
#include <vector>

#include "strata/kernels/iq_kernels.hpp"

namespace k = strata::kernels;

static double e2m1(int code) {
    static const double v[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    const double m = v[code & 7];
    return (code & 8) ? -m : m;
}
static double ue4m3(uint8_t x) {
    x &= 0x7F;
    if (x == 0 || x == 0x7F) return 0.0;
    const int e = (x >> 3) & 0xF, m = x & 7;
    return e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
}
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
static float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }

struct Blk { uint8_t d[4]; uint8_t qs[32]; };
static_assert(sizeof(Blk) == 36, "block_nvfp4 is 36 bytes");

struct Case { const char* what; int N, K; std::vector<int> cnt; bool all_bytes; };

static bool run_case(sycl::queue& q, const Case& c, uint32_t seed) {
    std::mt19937 rng(seed);
    const int E = (int) c.cnt.size(), nb = c.K / 64;
    int rows = 0;
    for (int n : c.cnt) rows += n;
    std::vector<std::vector<Blk>> w(E, std::vector<Blk>((size_t) c.N * nb));
    for (auto& ex : w)
        for (auto& b : ex) {
            for (auto& s : b.d) s = c.all_bytes ? (uint8_t) (rng() & 0xFF) : (uint8_t) ((3 + rng() % 9) << 3 | (rng() & 7));
            for (auto& v : b.qs) v = (uint8_t) (rng() & 0xFF);
        }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> x((size_t) rows * c.K);
    for (auto& v : x) v = f2h(nd(rng) * (rng() % 97 == 0 ? 8.f : 1.f));

    std::vector<uint8_t*> dw(E);
    for (int e = 0; e < E; ++e) {
        dw[e] = sycl::malloc_device<uint8_t>(w[e].size() * sizeof(Blk), q);
        q.memcpy(dw[e], w[e].data(), w[e].size() * sizeof(Blk));
    }
    uint16_t* dx = sycl::malloc_device<uint16_t>(x.size(), q);
    float* dy = sycl::malloc_device<float>((size_t) rows * c.N, q);
    q.memcpy(dx, x.data(), x.size() * 2);
    q.memset(dy, 0xFF, (size_t) rows * c.N * 4);   // NaN pattern: an unwritten output cannot pass
    q.wait();

    std::vector<const uint8_t*> a;
    std::vector<int32_t> cn;
    for (int e = 0; e < E; ++e)
        if (c.cnt[e] > 0) { a.push_back(dw[e]); cn.push_back(c.cnt[e]); }
    std::vector<float> y((size_t) rows * c.N);
    try {
        k::iq_xmx_grouped(40, a.data(), nullptr, cn.data(), (int) a.size(), dx, dy, c.N, c.K, (void*) &q);
        q.wait_and_throw();
        q.memcpy(y.data(), dy, y.size() * 4).wait();
    } catch (const sycl::exception& e) {
        printf("  %-34s SYCL exception: %s\n", c.what, e.what());
        return false;
    }

    int bad = 0, nonfinite = 0, row0 = 0;
    double worst = 0;
    for (int e = 0, at = 0; e < E; ++e) {
        std::vector<double> wf((size_t) c.N * c.K);
        for (int o = 0; o < c.N; ++o)
            for (int b = 0; b < nb; ++b) {
                const Blk& blk = w[e][(size_t) o * nb + b];
                for (int s = 0; s < 4; ++s) {
                    const double sc = ue4m3(blk.d[s]);
                    for (int j = 0; j < 8; ++j) {
                        const uint8_t byte = blk.qs[8 * s + j];
#ifdef NEG_CONTROL
                        wf[(size_t) o * c.K + b * 64 + s * 16 + j] = e2m1(byte >> 4) * sc;
                        wf[(size_t) o * c.K + b * 64 + s * 16 + j + 8] = e2m1(byte & 0xF) * sc;
#else
                        wf[(size_t) o * c.K + b * 64 + s * 16 + j] = e2m1(byte & 0xF) * sc;
                        wf[(size_t) o * c.K + b * 64 + s * 16 + j + 8] = e2m1(byte >> 4) * sc;
#endif
                    }
                }
            }
        for (int r = 0; r < c.cnt[e]; ++r, ++at)
            for (int o = 0; o < c.N; ++o) {
                double ref = 0, mag = 0;
                for (int kk = 0; kk < c.K; ++kk) {
                    const double t = (double) h2f(x[(size_t) at * c.K + kk]) * wf[(size_t) o * c.K + kk];
                    ref += t;
                    mag += std::fabs(t);
                }
                const float got = y[(size_t) at * c.N + o];
                if (!std::isfinite(got)) { ++nonfinite; ++bad; continue; }
                const double err = std::fabs((double) got - ref) / (mag + 1e-30);
                worst = std::fmax(worst, err);
                if (err > 1e-5) ++bad;
            }
        row0 = at;
    }
    printf("  %-34s N=%-5d K=%-5d experts=%d rows=%-4d worst %.3g of sum|terms|, %d of %d outputs bad (%d non-finite) -> %s\n",
           c.what, c.N, c.K, (int) a.size(), rows, worst, bad, rows * c.N, nonfinite, bad == 0 ? "PASS" : "FAIL");
    for (auto p : dw) sycl::free(p, q);
    sycl::free(dx, q);
    sycl::free(dy, q);
    (void) row0;
    return bad == 0;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::vector<Case> cases = {
        {"gate/up shape, realistic scales", 256, 2304, {0, 1, 7, 33, 130, 250}, false},
        {"down shape (K = 1152)", 256, 1152, {5, 0, 64, 129, 3}, false},
        {"all scale bytes (0x7F, bit 7)", 256, 1152, {9, 40, 100}, true},
        {"one expert, one row", 128, 1152, {1}, false},
        {"tiles smaller than a block", 384, 1152, {3, 17, 31, 65}, false},
        {"many experts (16)", 256, 1152, {2, 4, 8, 16, 32, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21}, false},
        {"large mean rows (Big tiles)", 256, 1152, {200, 300, 150, 260}, false},
    };
    int fails = 0;
    uint32_t seed = 1;
    for (auto& c : cases) fails += run_case(q, c, seed++) ? 0 : 1;
    printf("%s\n", fails ? "FAILED" : "all OK");
    return fails ? 1 : 0;
}
