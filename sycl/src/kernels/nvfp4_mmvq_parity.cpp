// Standalone test: NVFP4 (ggml type 40) native_mmvq on the GPU vs a CPU reference.
//
// Reference (FP64, independent of the kernel's bit tricks):
//   block of 64 values = 4 UE4M3 scale bytes + 32 bytes of E2M1 nibbles; in sub-block s (16 values), value j (0..7) is the
//   low nibble of qs[8 s + j] and value j + 8 the high nibble (ggml's reordering). Value = e2m1(code) * ue4m3(d[s]);
//   UE4M3 has no sign bit (bit 7 ignored) and 0x7F reads as 0.
// Activations are quantized by the library's own native_quantize_q8_1 and read back, so the comparison is against the
// exact int8 data and FP16 scales the kernel sees. Only float summation order can differ: tolerance 1e-5 of sum|terms|.
// The error against the ORIGINAL float activations is printed for information (it is Q8_1 quantization error).
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <random>
#include <vector>

#include "strata/kernels/native_mmvq.hpp"

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

struct Blk { uint8_t d[4]; uint8_t qs[32]; };
static_assert(sizeof(Blk) == 36, "block_nvfp4 is 36 bytes");

enum Mode { REALISTIC = 0, ALL_BYTES = 1, EDGE = 2 };

static uint8_t pick_scale(std::mt19937& rng, Mode mode, size_t counter) {
    switch (mode) {
        case REALISTIC: return (uint8_t) ((3 + rng() % 9) << 3 | (rng() & 7));            // exp 3..11
        case ALL_BYTES: return (uint8_t) (rng() & 0xFF);                                  // incl. 0x7F, bit 7, subnormals
        default: {
            static const uint8_t e[] = {0x00, 0x01, 0x07, 0x08, 0x7E, 0x7F, 0xFF, 0x80};
            return e[counter % 8];
        }
    }
}

struct Case { int n_in, n_out, ncols; Mode mode; bool zero_col; const char* what; };

static bool run_case(sycl::queue& q, const Case& c, uint32_t seed) {
    std::mt19937 rng(seed);
    const int nb = c.n_in / 64;
    std::vector<Blk> w((size_t) c.n_out * nb);
    size_t counter = 0;
    for (auto& b : w) {
        for (auto& s : b.d) s = pick_scale(rng, c.mode, counter++);
        for (auto& v : b.qs) v = (uint8_t) (rng() & 0xFF);
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) c.ncols * c.n_in);
    for (auto& v : x) v = nd(rng) * (rng() % 97 == 0 ? 8.f : 1.f);   // a few outliers
    if (c.zero_col) std::fill(x.begin(), x.begin() + c.n_in, 0.f);    // column 0 all zero: amax = 0

    const size_t wbytes = w.size() * sizeof(Blk);
    uint8_t* dw = sycl::malloc_device<uint8_t>(wbytes, q);
    float* dx = sycl::malloc_device<float>(x.size(), q);
    float* dy = sycl::malloc_device<float>((size_t) c.ncols * c.n_out, q);
    const size_t qbytes = k::native_q8_1_bytes(c.n_in, c.ncols);
    uint8_t* dq = sycl::malloc_device<uint8_t>(qbytes, q);
    q.memcpy(dw, w.data(), wbytes);
    q.memcpy(dx, x.data(), x.size() * 4);
    q.memset(dy, 0xFF, (size_t) c.ncols * c.n_out * 4);   // NaN pattern: an unwritten output cannot pass
    q.wait();

    std::vector<float> y((size_t) c.ncols * c.n_out), y1((size_t) c.ncols * c.n_out);
    std::vector<uint8_t> hq(qbytes);
    try {
        k::native_quantize_q8_1(dx, dq, c.n_in, c.ncols, (void*) &q);
        k::native_mmvq(40, dw, dq, dy, c.n_in, c.n_out, c.ncols, (void*) &q);
        q.wait_and_throw();
        q.memcpy(y.data(), dy, y.size() * 4).wait();
        q.memcpy(hq.data(), dq, qbytes).wait();
        // every column through a one-column call: the documented default (multi_exact) says bitwise equal
        for (int col = 0; col < c.ncols; ++col) {
            k::native_quantize_q8_1(dx + (size_t) col * c.n_in, dq, c.n_in, 1, (void*) &q);
            k::native_mmvq(40, dw, dq, dy + (size_t) col * c.n_out, c.n_in, c.n_out, 1, (void*) &q);
        }
        q.wait_and_throw();
        q.memcpy(y1.data(), dy, y1.size() * 4).wait();
    } catch (const std::invalid_argument& e) {
        printf("  %-34s n_in=%-5d n_out=%-4d cols=%d  SKIP (unsupported: %s)\n", c.what, c.n_in, c.n_out, c.ncols, e.what());
        sycl::free(dw, q); sycl::free(dx, q); sycl::free(dy, q); sycl::free(dq, q);
        return true;
    } catch (const sycl::exception& e) {
        printf("  %-34s SYCL exception: %s\n", c.what, e.what());
        return false;
    }

    // decode the weights once
    std::vector<double> wf((size_t) c.n_out * c.n_in);
    for (int o = 0; o < c.n_out; ++o)
        for (int b = 0; b < nb; ++b) {
            const Blk& blk = w[(size_t) o * nb + b];
            for (int s = 0; s < 4; ++s) {
                const double sc = ue4m3(blk.d[s]);
                for (int j = 0; j < 8; ++j) {
                    const uint8_t byte = blk.qs[8 * s + j];
#ifdef NEG_CONTROL   // wrong on purpose (low/high nibbles swapped): every case must FAIL, or the test checks nothing
                    wf[(size_t) o * c.n_in + b * 64 + s * 16 + j] = e2m1(byte >> 4) * sc;
                    wf[(size_t) o * c.n_in + b * 64 + s * 16 + j + 8] = e2m1(byte & 0xF) * sc;
#else
                    wf[(size_t) o * c.n_in + b * 64 + s * 16 + j] = e2m1(byte & 0xF) * sc;
                    wf[(size_t) o * c.n_in + b * 64 + s * 16 + j + 8] = e2m1(byte >> 4) * sc;
#endif
                }
            }
        }

    double worst_q = 0, worst_x = 0;
    long bad = 0, nonfinite = 0, notbitwise = 0;
    for (int col = 0; col < c.ncols; ++col) {
        // the activations as the kernel sees them: int8 * FP16 scale, per block of 32
        std::vector<double> xq(c.n_in);
        for (int blk = 0; blk < c.n_in / 32; ++blk) {
            const uint8_t* p = hq.data() + ((size_t) col * (c.n_in / 32) + blk) * 36;
            uint16_t dh; std::memcpy(&dh, p, 2);
            const double d = (double) (float) sycl::bit_cast<sycl::half>(dh);
            for (int i = 0; i < 32; ++i) xq[blk * 32 + i] = d * (double) (int8_t) p[4 + i];
        }
        for (int o = 0; o < c.n_out; ++o) {
            double ref_q = 0, abs_q = 0, ref_x = 0, abs_x = 0;
            const double* wr = &wf[(size_t) o * c.n_in];
            for (int i = 0; i < c.n_in; ++i) {
                ref_q += wr[i] * xq[i]; abs_q += std::fabs(wr[i] * xq[i]);
                const double t = wr[i] * (double) x[(size_t) col * c.n_in + i];
                ref_x += t; abs_x += std::fabs(t);
            }
            const float got = y[(size_t) col * c.n_out + o];
            if (!std::isfinite(got)) { ++nonfinite; ++bad; continue; }
            const double err_q = std::fabs((double) got - ref_q) / (abs_q > 0 ? abs_q : 1.0);
            const double err_x = std::fabs((double) got - ref_x) / (abs_x > 0 ? abs_x : 1.0);
            worst_q = std::max(worst_q, err_q);
            worst_x = std::max(worst_x, err_x);
            if (err_q > 1e-5) ++bad;
            if (std::memcmp(&got, &y1[(size_t) col * c.n_out + o], 4) != 0) ++notbitwise;
        }
    }
    const bool ok = bad == 0 && notbitwise == 0;
    printf("  %-34s n_in=%-5d n_out=%-4d cols=%d  vs-Q8_1-ref max %.2e | vs-float-x max %.2e | bad %ld (nonfinite %ld) | "
           "multi!=single %ld  %s\n", c.what, c.n_in, c.n_out, c.ncols, worst_q, worst_x, bad, nonfinite, notbitwise,
           ok ? "PASS" : "FAIL");
    sycl::free(dw, q); sycl::free(dx, q); sycl::free(dy, q); sycl::free(dq, q);
    return ok;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const Case cases[] = {
        {256, 64, 1, REALISTIC, false, "one block-row group, 1 col"},
        {2304, 100, 1, REALISTIC, false, "Inkling-like width, 1 col"},
        {2304, 100, 8, REALISTIC, false, "8 cols"},
        {4096, 257, 3, ALL_BYTES, false, "every scale byte, odd rows"},
        {256, 33, 5, EDGE, true, "edge scales + zero column"},
        {1024, 512, 8, ALL_BYTES, false, "bigger"},
        {320, 40, 2, REALISTIC, false, "n_in = 5 blocks (not x256)"},
        {64, 17, 1, REALISTIC, false, "a single block"},
    };
    int fail = 0, n = 0;
    uint32_t seed = 1;
    for (const auto& c : cases) { ++n; if (!run_case(q, c, seed++)) ++fail; }
    printf("%s: %d of %d cases passed\n", fail ? "FAIL" : "PASS", n - fail, n);
    return fail ? 1 : 0;
}
