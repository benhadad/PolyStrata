// src/poly/inkling_kernels_test.cu - Inkling 2 Mini's kernels (inkling_kernels.hpp) against the same arithmetic on
// the host in double precision, on random inputs of the model's shapes: the clips and tau, the short convolutions
// (with and without the input-dependent taps, walked and parallel, and their state), the depth mix, the loop
// controller, SiTU, and the attention over fp16 and Q8_0 caches, full and sliding-window (a ring), split and
// unsplit, with and without a room past the call, for 1, 7 and more than 1,024 tokens.
//
// usage: poly_inkling_kernels_test [bench | decode]   (exit 0: every check within its tolerance; 77: no usable card;
//        bench: also time the attention and the decode shapes; decode: only the decode-shape timings)
#include "strata/poly/inkling_kernels.hpp"
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace strata::poly;

int failures = 0;
std::mt19937 rng(1234);

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

std::vector<float> randv(size_t n, float sd, float mean = 0.f) {
    std::normal_distribution<float> d(mean, sd);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}

template <typename T> T* up(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T)), "malloc");
    if (!v.empty()) ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> v(n);
    ck(cudaDeviceSynchronize(), "sync");
    ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

// max |a - b| and max |a - b| / max(|b|, floor); reports and counts a failure past tol (on the absolute error)
void compare(const char* what, const std::vector<float>& got, const std::vector<double>& ref, double tol,
             double floor_ = 1e-3) {
    double abs_e = 0, rel_e = 0;
    bool nan = false;
    size_t worst = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) nan = true;
        const double e = std::fabs(double(got[i]) - ref[i]);
        if (e > abs_e) abs_e = e, worst = i;
        rel_e = std::max(rel_e, e / std::max(std::fabs(ref[i]), floor_));
    }
    const bool ok = !nan && abs_e <= tol;
    if (!ok) ++failures;
    std::printf("%-58s n=%-9zu max abs %.3e  max rel %.3e  %s", what, ref.size(), abs_e, rel_e, ok ? "ok" : "FAIL");
    if (!ok) std::printf("  (at %zu: got %.6g ref %.6g%s)", worst, got[worst], ref[worst], nan ? ", NaN" : "");
    std::printf("\n");
}

// ---- clip -------------------------------------------------------------------------------------------------------

double host_tau(InklingTau t, long long p) {
    if (t.floor <= 0) return 1.0;
    float v = (float) p + 1.0f;
    v = v * (1.0f / (float) t.floor);
    v = std::max(v, 1.0f);
    return 1.0 + double(t.alpha) * std::log((double) v);
}

void test_clip(const char* what, int dim, int heads, int rows, bool scaled, float threshold, InklingTau tau,
               long long pos, bool wide) {
    const int stride = heads * dim + (wide ? 64 : 0);   // wide: a token's input starts past the one before's end
    std::vector<float> x(size_t(rows) * stride);
    for (int t = 0; t < rows; ++t) {   // some rows far inside the threshold, some far outside
        const float sd = (t % 3 == 0 ? 0.2f : 3.0f) * threshold / std::sqrt((float) dim);
        auto r = randv(stride, sd);
        std::copy(r.begin(), r.end(), x.begin() + size_t(t) * stride);
    }
    std::vector<float> s = scaled ? randv(size_t(heads) * dim, 0.3f, 1.0f) : std::vector<float>{};
    float *dx = up(x), *ds = scaled ? up(s) : nullptr, *dy = nullptr;
    ck(cudaMalloc(&dy, size_t(rows) * heads * dim * sizeof(float)), "malloc");
    inkling_clip(dx, stride, ds, dy, dim, heads, rows, threshold, tau, pos, nullptr);
    std::vector<double> ref(size_t(rows) * heads * dim);
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < heads; ++h) {
            std::vector<double> v(dim);
            double ss = 0;
            for (int i = 0; i < dim; ++i) {
                v[i] = double(x[size_t(t) * stride + h * dim + i]) * (scaled ? s[h * dim + i] : 1.0);
                ss += v[i] * v[i];
            }
            const double f = std::min(1.0, std::max(0.0, threshold / (std::sqrt(ss) + 1e-6)));
            const double ta = host_tau(tau, pos + t);
            for (int i = 0; i < dim; ++i) ref[(size_t(t) * heads + h) * dim + i] = v[i] * f * ta;
        }
    compare(what, down(dy, ref.size()), ref, 2e-5 * threshold);
    cudaFree(dx), cudaFree(ds), cudaFree(dy);
}

// ---- conv -------------------------------------------------------------------------------------------------------

void host_conv(const std::vector<float>& x, std::vector<double>& state, const std::vector<float>& w,
               const std::vector<float>* taps, int heads, std::vector<double>& out, int channels, int rows) {
    const double scale = 1.0 / std::sqrt(double((taps ? 2 : 1) * 4 + 1));
    for (int t = 0; t < rows; ++t) {
        for (int c = 0; c < channels; ++c) {
            double* st = state.data() + size_t(c) * 3;
            const double v = x[size_t(t) * channels + c];
            double y = st[0] * w[4 * c] + st[1] * w[4 * c + 1] + st[2] * w[4 * c + 2] + v * w[4 * c + 3];
            if (taps) {
                const int h = c / (channels / heads);
                const float* tp = taps->data() + (size_t(t) * heads + h) * 4;
                y += tp[0] * v + tp[1] * st[2] + tp[2] * st[1] + tp[3] * st[0];
            }
            out[size_t(t) * channels + c] = (v + y) * scale;
            st[0] = st[1], st[1] = st[2], st[2] = v;
        }
    }
}

void test_conv(const char* what, int channels, int heads, bool dynamic, std::vector<int> calls) {
    std::vector<float> w = randv(size_t(channels) * 4, 0.4f);
    std::vector<float> st0 = randv(size_t(channels) * 3, 1.0f);
    std::vector<double> hstate(st0.begin(), st0.end());
    float *dw = up(w), *dst = up(st0);
    int total = 0;
    for (int rows : calls) {
        std::vector<float> x = randv(size_t(rows) * channels, 1.0f);
        std::vector<float> taps = randv(size_t(rows) * heads * 4, 0.3f);
        float *dx = up(x), *dt = dynamic ? up(taps) : nullptr, *dout = nullptr;
        ck(cudaMalloc(&dout, x.size() * sizeof(float)), "malloc");
        inkling_conv(dx, dst, dw, dt, heads, dout, channels, rows, nullptr);
        std::vector<double> ref(x.size());
        host_conv(x, hstate, w, dynamic ? &taps : nullptr, heads, ref, channels, rows);
        const std::string name = std::string(what) + " rows " + std::to_string(rows) + " out";
        compare(name.c_str(), down(dout, ref.size()), ref, 1e-5);
        cudaFree(dx), cudaFree(dt), cudaFree(dout);
        total += rows;
    }
    compare((std::string(what) + " state after " + std::to_string(total)).c_str(), down(dst, hstate.size()), hstate,
            0.0);
    cudaFree(dw), cudaFree(dst);
}

// ---- mix --------------------------------------------------------------------------------------------------------

// offset: the states start that many floats into their buffers (1: not 16-byte aligned, the scalar kernel)
void test_mix(int n_state, int n_query, int rows, int n = 2304, int offset = 0) {
    InklingMix m;
    m.n_state = n_state;
    m.n_query = n_query;
    std::vector<std::vector<float>> s;
    std::vector<float*> bufs;
    for (int k = 0; k < n_state; ++k) {
        s.push_back(randv(size_t(rows) * n, 0.5f + k, 0.1f * k));
        std::vector<float> padded(offset, 0.f);
        padded.insert(padded.end(), s.back().begin(), s.back().end());
        bufs.push_back(up(padded));
        m.state[k] = bufs.back() + offset;
    }
    std::vector<float> q = randv(size_t(n_query) * n, 1.0f);
    m.query = up(q);
    for (int i = 0; i < n_query; ++i) ck(cudaMalloc(&m.out[i], size_t(rows) * n * sizeof(float)), "malloc");
    inkling_mix(m, n, rows, 1e-6f, nullptr);
    std::vector<double> ref(size_t(n_query) * rows * n);
    for (int t = 0; t < rows; ++t) {
        std::vector<std::vector<double>> hat(n_state, std::vector<double>(n));
        for (int k = 0; k < n_state; ++k) {
            double ss = 0;
            for (int i = 0; i < n; ++i) ss += double(s[k][size_t(t) * n + i]) * s[k][size_t(t) * n + i];
            const double inv = 1.0 / std::sqrt(ss / n + 1e-6);
            for (int i = 0; i < n; ++i) hat[k][i] = s[k][size_t(t) * n + i] * inv;
        }
        for (int j = 0; j < n_query; ++j) {
            std::vector<double> sc(n_state);
            double mx = -1e300;
            for (int k = 0; k < n_state; ++k) {
                double d = 0;
                for (int i = 0; i < n; ++i) d += hat[k][i] * q[size_t(j) * n + i];
                sc[k] = d / std::sqrt((double) n);
                mx = std::max(mx, sc[k]);
            }
            double sum = 0;
            for (double& v : sc) sum += (v = std::exp(v - mx));
            for (int i = 0; i < n; ++i) {
                double y = 0;
                for (int k = 0; k < n_state; ++k) y += sc[k] / sum * hat[k][i];
                ref[(size_t(j) * rows + t) * n + i] = y;
            }
        }
    }
    std::vector<float> got;
    for (int j = 0; j < n_query; ++j) {
        auto o = down(m.out[j], size_t(rows) * n);
        got.insert(got.end(), o.begin(), o.end());
    }
    const std::string name = "mix states " + std::to_string(n_state) + " queries " + std::to_string(n_query) +
                             " rows " + std::to_string(rows) + " n " + std::to_string(n) +
                             (offset ? " (unaligned)" : "");
    compare(name.c_str(), got, ref, 2e-4);
    for (float* b : bufs) cudaFree(b);
    for (int j = 0; j < n_query; ++j) cudaFree(m.out[j]);
    cudaFree((void*) m.query);
}

// ---- controller -------------------------------------------------------------------------------------------------

// the reference of one controller
void host_control(const std::vector<float>& cur, const std::vector<float>& msg, const std::vector<float>& wd,
                  const std::vector<float>& wg, const std::vector<float>& wf, int n, int rank, int rows, double* ref) {
    for (int t = 0; t < rows; ++t) {
        std::vector<double> z(2 * n), r(rank);
        for (int half = 0; half < 2; ++half) {
            const float* x = (half ? msg.data() : cur.data()) + size_t(t) * n;
            double ss = 0;
            for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
            const double inv = 1.0 / std::sqrt(ss / n + 1e-6);
            for (int i = 0; i < n; ++i) z[half * n + i] = x[i] * inv;
        }
        for (int o = 0; o < rank; ++o) {
            double a = 0;
            for (int i = 0; i < 2 * n; ++i) a += double(wd[size_t(o) * 2 * n + i]) * z[i];
            r[o] = a;
        }
        for (int o = 0; o < n; ++o) {
            double g = 0, f = 0;
            for (int i = 0; i < rank; ++i) {
                g += double(wg[size_t(o) * rank + i]) * r[i];
                f += double(wf[size_t(o) * rank + i]) * r[i];
            }
            const double c = cur[size_t(t) * n + o], m = msg[size_t(t) * n + o];
            ref[size_t(t) * n + o] = (c - std::tanh(f) * c) + std::tanh(g) * m;
        }
    }
}

void test_control(void* blas, int rows) {
    const int n = 2304, rank = 64;
    std::vector<float> cur = randv(size_t(rows) * n, 1.0f), msg = randv(size_t(rows) * n, 2.0f);
    std::vector<float> wd = randv(size_t(rank) * 2 * n, 0.02f), wg = randv(size_t(n) * rank, 0.2f),
                       wf = randv(size_t(n) * rank, 0.2f);
    float *dc = up(cur), *dm = up(msg), *dd = up(wd), *dg = up(wg), *df = up(wf), *work = nullptr, *out = nullptr;
    ck(cudaMalloc(&work, size_t(rows) * (4 * n + rank) * sizeof(float)), "malloc");
    ck(cudaMalloc(&out, size_t(rows) * n * sizeof(float)), "malloc");
    inkling_control(blas, dc, dm, dd, dg, df, work, out, n, rank, rows, 1e-6f, nullptr);
    std::vector<double> ref(size_t(rows) * n);
    host_control(cur, msg, wd, wg, wf, n, rank, rows, ref.data());
    compare(("control rows " + std::to_string(rows)).c_str(), down(out, ref.size()), ref, 1e-4);
    cudaFree(dc), cudaFree(dm), cudaFree(dd), cudaFree(dg), cudaFree(df), cudaFree(work), cudaFree(out);
}

// ---- SiTU -------------------------------------------------------------------------------------------------------

void test_situ() {
    const int n = 1152, rows = 9;
    const float beta = 4.0f, lb = 25.0f;
    std::vector<float> gu = randv(size_t(rows) * 2 * n, 6.0f);
    for (int i = 0; i < 32; ++i) gu[i] = 60.f * (i - 16), gu[n + i] = 90.f * (i - 16);   // the saturated ends
    float *dgu = up(gu), *dout = nullptr;
    ck(cudaMalloc(&dout, size_t(rows) * n * sizeof(float)), "malloc");
    situ_rows(dgu, dout, n, rows, beta, lb, nullptr);
    std::vector<double> ref(size_t(rows) * n);
    for (int t = 0; t < rows; ++t)
        for (int i = 0; i < n; ++i) {
            const double g = gu[size_t(t) * 2 * n + i], u = gu[size_t(t) * 2 * n + n + i];
            ref[size_t(t) * n + i] = beta * std::tanh(g / beta) / (1 + std::exp(-g)) * lb * std::tanh(u / lb);
        }
    compare("situ_rows (n 1152, rows 9, beta 4, linear beta 25)", down(dout, ref.size()), ref, 2e-4);
    // lb <= 0: the up values as they are
    situ_rows(dgu, dout, n, 1, beta, 0.f, nullptr);
    std::vector<double> ref0(n);
    for (int i = 0; i < n; ++i) ref0[i] = beta * std::tanh(gu[i] / beta) / (1 + std::exp(-gu[i])) * gu[n + i];
    compare("situ_rows (no linear beta, 1 row)", down(dout, n), ref0, 2e-3);
    cudaFree(dgu), cudaFree(dout);
}

// ---- attention --------------------------------------------------------------------------------------------------

constexpr int kDim = 128, kHeads = 24, kKv = 8, kRel = 16, kExtent = 1024;

struct Cache {
    int rows_cap = 0;   // rows the cache holds (max context, or the ring)
    bool q8 = false;
    uint16_t *k = nullptr, *v = nullptr;
    int8_t *kc = nullptr, *vc = nullptr;
    uint16_t *ks = nullptr, *vs = nullptr;
    std::vector<float> hk, hv;   // what the cache holds, as the kernels read it, by cache row
    void alloc(int cap, bool quant) {
        rows_cap = cap, q8 = quant;
        const size_t n = size_t(cap) * kKv * kDim;
        if (q8) {
            ck(cudaMalloc(&kc, n), "malloc"), ck(cudaMalloc(&vc, n), "malloc");
            ck(cudaMalloc(&ks, n / 32 * 2), "malloc"), ck(cudaMalloc(&vs, n / 32 * 2), "malloc");
            // garbage where nothing was stored: the kernels must not read it
            ck(cudaMemset(kc, 0x7f, n), "memset"), ck(cudaMemset(vc, 0x7f, n), "memset");
            ck(cudaMemset(ks, 0x7c, n / 16), "memset"), ck(cudaMemset(vs, 0x7c, n / 16), "memset");
        } else {
            ck(cudaMalloc(&k, n * 2), "malloc"), ck(cudaMalloc(&v, n * 2), "malloc");
            ck(cudaMemset(k, 0xff, n * 2), "memset"), ck(cudaMemset(v, 0xff, n * 2), "memset");   // NaN halves
        }
        hk.assign(n, NAN), hv.assign(n, NAN);
    }
    void free_() { cudaFree(k), cudaFree(v), cudaFree(kc), cudaFree(vc), cudaFree(ks), cudaFree(vs); }
    // the rows of positions pos.. into cache rows p % ring (ring 0: p), then the host copy refreshed
    void store(const float* dkx, const float* dvx, long long pos, int rows, int ring) {
        const int width = kKv * kDim;
        for (int done = 0; done < rows;) {
            const long long p = pos + done;
            const int at = ring ? int(p % ring) : int(p);
            const int m = ring ? std::min(rows - done, ring - at) : rows - done;
            if (q8) {
                store_q8(dkx + size_t(done) * width, kc, ks, width, at, m, nullptr);
                store_q8(dvx + size_t(done) * width, vc, vs, width, at, m, nullptr);
            } else {
                store_f16(dkx + size_t(done) * width, k, width, at, m, nullptr);
                store_f16(dvx + size_t(done) * width, v, width, at, m, nullptr);
            }
            done += m;
        }
        refresh();
    }
    void refresh() {
        const size_t n = size_t(rows_cap) * kKv * kDim;
        if (q8) {
            auto c1 = down(kc, n), c2 = down(vc, n);
            auto s1 = down(ks, n / 32), s2 = down(vs, n / 32);
            for (size_t i = 0; i < n; ++i) {
                hk[i] = float(c1[i]) * __half2float(__ushort_as_half(s1[i / 32]));
                hv[i] = float(c2[i]) * __half2float(__ushort_as_half(s2[i / 32]));
            }
        } else {
            auto a = down(k, n), b = down(v, n);
            for (size_t i = 0; i < n; ++i) {
                hk[i] = __half2float(__ushort_as_half(a[i]));
                hv[i] = __half2float(__ushort_as_half(b[i]));
            }
        }
    }
};

// the reference for token t (position p) of the call
void host_attend(const Cache& c, const std::vector<float>& q, const std::vector<float>& r,
                 const std::vector<float>& table, const std::vector<float>* sinks, long long pos, int t, int window,
                 int ring, double* out) {
    const long long p = pos + t;
    const long long lo = window > 0 ? std::max<long long>(0, p - window + 1) : 0;
    const double scale = 1.0 / std::sqrt((double) kDim);
    for (int h = 0; h < kHeads; ++h) {
        const int g = h / (kHeads / kKv);
        const float* qh = q.data() + (size_t(t) * kHeads + h) * kDim;
        const float* rh = r.data() + (size_t(t) * kHeads + h) * kRel;
        std::vector<double> lg;
        double mx = sinks ? (*sinks)[size_t(t) * kHeads + h] : -1e300;
        for (long long j = lo; j <= p; ++j) {
            const long long row = ring ? j % ring : j;
            const float* kj = c.hk.data() + (size_t(row) * kKv + g) * kDim;
            double d = 0;
            for (int i = 0; i < kDim; ++i) d += double(qh[i]) * kj[i];
            const long long dist = p - j;
            double b = 0;
            if (dist < kExtent)
                for (int i = 0; i < kRel; ++i) b += double(table[size_t(dist) * kRel + i]) * rh[i];
            lg.push_back(d * scale + b);
            mx = std::max(mx, lg.back());
        }
        double sum = sinks ? std::exp((*sinks)[size_t(t) * kHeads + h] - mx) : 0.0;
        for (double& v : lg) sum += (v = std::exp(v - mx));
        double* o = out + size_t(h) * kDim;
        std::fill(o, o + kDim, 0.0);
        for (long long j = lo; j <= p; ++j) {
            const long long row = ring ? j % ring : j;
            const float* vj = c.hv.data() + (size_t(row) * kKv + g) * kDim;
            const double w = lg[size_t(j - lo)] / sum;
            for (int i = 0; i < kDim; ++i) o[i] += w * vj[i];
        }
    }
}

struct AttnCase {
    const char* name;
    int window, ring;   // 0 0: a full layer
    bool q8;
    std::vector<int> calls;   // tokens of each call, one after the other from position 0
    bool split;               // give inkling_attend its work buffer
    bool room;                // size the launch for a room past the call's positions (a full visit's)
    bool sinks;
    int check_every;          // check every n-th token of a long call (and its last, and the window's edge)
};

void test_attend(const AttnCase& tc, const std::vector<float>& table, float* dtable) {
    int total = 0;
    for (int n : tc.calls) total += n;
    Cache c;
    c.alloc(tc.ring ? tc.ring : total, tc.q8);
    float* work = nullptr;
    const size_t work_n = inkling_attend_work_floats(kHeads, kDim);
    if (tc.split) ck(cudaMalloc(&work, work_n * sizeof(float)), "malloc");
    long long pos = 0;
    for (int rows : tc.calls) {
        // queries clipped to the model's range (|q| <= 22.6), keys too, values rms-normed: sizes as the model's
        std::vector<float> q = randv(size_t(rows) * kHeads * kDim, 1.2f), k = randv(size_t(rows) * kKv * kDim, 1.2f),
                           v = randv(size_t(rows) * kKv * kDim, 1.0f), r = randv(size_t(rows) * kHeads * kRel, 1.5f),
                           s = randv(size_t(rows) * kHeads, 2.0f, 1.0f);
        float *dq = up(q), *dk = up(k), *dv = up(v), *dr = up(r), *ds = up(s), *dout = nullptr;
        ck(cudaMalloc(&dout, q.size() * sizeof(float)), "malloc");
        c.store(dk, dv, pos, rows, tc.ring);
        InklingRel rel{dr, dtable, kRel, kExtent};
        const Kvq8 kq = tc.q8 ? Kvq8{c.kc, c.ks} : Kvq8{}, vq = tc.q8 ? Kvq8{c.vc, c.vs} : Kvq8{};
        const float scale = 1.f / std::sqrt((float) kDim);
        inkling_attend(dq, c.k, c.v, kq, vq, rel, tc.sinks ? ds : nullptr, pos, rows, dout, kDim, kHeads, kKv, scale,
                       tc.window, tc.ring, work, work_n, nullptr, tc.room ? pos + rows + 517 : 0);
        ck(cudaGetLastError(), "attend");
        auto got = down(dout, q.size());
        std::vector<int> toks;
        for (int t = 0; t < rows; ++t) {
            const long long p = pos + t;
            const bool edge = (p >= 1020 && p <= 1030) || (p >= 2045 && p <= 2050);
            if (t % tc.check_every == 0 || t == rows - 1 || edge) toks.push_back(t);
        }
        std::vector<double> ref(toks.size() * kHeads * kDim);
        std::vector<float> sel(ref.size());
        for (size_t i = 0; i < toks.size(); ++i) {
            host_attend(c, q, r, table, tc.sinks ? &s : nullptr, pos, toks[i], tc.window, tc.ring,
                        ref.data() + i * kHeads * kDim);
            std::copy(got.begin() + size_t(toks[i]) * kHeads * kDim, got.begin() + size_t(toks[i] + 1) * kHeads * kDim,
                      sel.begin() + i * kHeads * kDim);
        }
        const std::string name = std::string(tc.name) + ": pos " + std::to_string(pos) + " rows " +
                                 std::to_string(rows) + " (" + std::to_string(toks.size()) + " checked)";
        compare(name.c_str(), sel, ref, 2e-5);
        cudaFree(dq), cudaFree(dk), cudaFree(dv), cudaFree(dr), cudaFree(ds), cudaFree(dout);
        pos += rows;
    }
    c.free_();
    cudaFree(work);
}

// A token's result the same, to the bit, alone and beside others (the split path).
void test_attend_alone(const std::vector<float>& table, float* dtable) {
    const int ctx = 3000, rows = 7, window = 1024, ring = window + 2048;
    for (int w : {0, window}) {
        Cache c;
        c.alloc(w ? ring : ctx + rows, false);
        auto k = randv(size_t(ctx + rows) * kKv * kDim, 1.2f), v = randv(size_t(ctx + rows) * kKv * kDim, 1.0f);
        float *dk = up(k), *dv = up(v);
        for (int p = 0; p < ctx + rows; p += 1000) {
            const int m = std::min(1000, ctx + rows - p);
            c.store(dk + size_t(p) * kKv * kDim, dv + size_t(p) * kKv * kDim, p, m, w ? ring : 0);
        }
        auto q = randv(size_t(rows) * kHeads * kDim, 1.2f), r = randv(size_t(rows) * kHeads * kRel, 1.5f),
             s = randv(size_t(rows) * kHeads, 2.0f);
        float *dq = up(q), *dr = up(r), *ds = up(s), *o1 = nullptr, *o2 = nullptr, *work = nullptr;
        ck(cudaMalloc(&o1, q.size() * 4), "malloc"), ck(cudaMalloc(&o2, q.size() * 4), "malloc");
        const size_t wn = inkling_attend_work_floats(kHeads, kDim);
        ck(cudaMalloc(&work, wn * 4), "malloc");
        const float scale = 1.f / std::sqrt((float) kDim);
        inkling_attend(dq, c.k, c.v, {}, {}, {dr, dtable, kRel, kExtent}, ds, ctx, rows, o1, kDim, kHeads, kKv, scale,
                       w, w ? ring : 0, work, wn, nullptr, ctx + 64);
        for (int t = 0; t < rows; ++t) {
            inkling_attend(dq + size_t(t) * kHeads * kDim, c.k, c.v, {}, {},
                           {dr + size_t(t) * kHeads * kRel, dtable, kRel, kExtent}, ds + size_t(t) * kHeads, ctx + t,
                           1, o2 + size_t(t) * kHeads * kDim, kDim, kHeads, kKv, scale, w, w ? ring : 0, work, wn,
                           nullptr, ctx + 64);
            ck(cudaDeviceSynchronize(), "sync");
        }
        auto a = down(o1, q.size()), b = down(o2, q.size());
        size_t diff = 0;
        for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
        std::printf("%-58s %zu of %zu values differ  %s\n",
                    w ? "attend split, 7 tokens vs one at a time (window)"
                      : "attend split, 7 tokens vs one at a time (full)",
                    diff, a.size(), diff ? "FAIL" : "ok");
        if (diff) ++failures;
        cudaFree(dk), cudaFree(dv), cudaFree(dq), cudaFree(dr), cudaFree(ds), cudaFree(o1), cudaFree(o2),
            cudaFree(work);
        c.free_();
    }
}

// Timings (with `bench`): a prompt batch's attention and a token's, full and window layers, fp16 caches.
void bench_attend(const std::vector<float>& table, float* dtable) {
    const int ctx = 32768, window = 1024;
    auto k = randv(size_t(ctx) * kKv * kDim, 1.0f);
    float* dk = up(k);   // the same numbers for keys and values: only the time is looked at
    uint16_t *kc = nullptr, *vc = nullptr;
    ck(cudaMalloc(&kc, k.size() * 2), "malloc"), ck(cudaMalloc(&vc, k.size() * 2), "malloc");
    store_f16(dk, kc, kKv * kDim, 0, ctx, nullptr);
    store_f16(dk, vc, kKv * kDim, 0, ctx, nullptr);
    const int rows = 4096;
    auto q = randv(size_t(rows) * kHeads * kDim, 1.0f), r = randv(size_t(rows) * kHeads * kRel, 1.0f),
         s = randv(size_t(rows) * kHeads, 1.0f);
    float *dq = up(q), *dr = up(r), *ds = up(s), *out = nullptr, *work = nullptr;
    ck(cudaMalloc(&out, q.size() * 4), "malloc");
    const size_t wn = inkling_attend_work_floats(kHeads, kDim);
    ck(cudaMalloc(&work, wn * 4), "malloc");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    auto time = [&](const char* what, auto&& f) {
        f();
        cudaEventRecord(e0);
        const int reps = 5;
        for (int i = 0; i < reps; ++i) f();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        std::printf("bench %-52s %9.3f ms\n", what, ms / reps);
    };
    const float sc = 1.f / std::sqrt((float) kDim);
    InklingRel rel{dr, dtable, kRel, kExtent};
    time("prompt 4096 tokens from 0, full layer", [&] {
        inkling_attend(dq, kc, vc, {}, {}, rel, ds, 0, rows, out, kDim, kHeads, kKv, sc, 0, 0, work, wn, nullptr);
    });
    time("prompt 4096 tokens from 0, window 1024", [&] {
        inkling_attend(dq, kc, vc, {}, {}, rel, ds, 0, rows, out, kDim, kHeads, kKv, sc, window, 0, work, wn, nullptr);
    });
    for (int p : {1024, 4096, 32767}) {
        const std::string a = "1 token at " + std::to_string(p) + ", full layer";
        time(a.c_str(), [&] {
            inkling_attend(dq, kc, vc, {}, {}, rel, ds, p, 1, out, kDim, kHeads, kKv, sc, 0, 0, work, wn, nullptr);
        });
    }
    time("1 token at 32767, window 1024", [&] {
        inkling_attend(dq, kc, vc, {}, {}, rel, ds, 32767, 1, out, kDim, kHeads, kKv, sc, window, 0, work, wn,
                       nullptr);
    });
    time("7 tokens at 32760, window 1024", [&] {
        inkling_attend(dq, kc, vc, {}, {}, rel, ds, 32760, 7, out, kDim, kHeads, kKv, sc, window, 0, work, wn,
                       nullptr);
    });
    ck(cudaDeviceSynchronize(), "bench");
    cudaFree(dk), cudaFree(kc), cudaFree(vc), cudaFree(dq), cudaFree(dr), cudaFree(ds), cudaFree(out), cudaFree(work);
}

// ---- the fused decode steps ---------------------------------------------------------------------------------------

void test_control_fused(int count, int rows) {
    const int n = 2304, rank = 64;
    std::vector<InklingCtrl> c(count);
    std::vector<double> ref(size_t(count) * rows * n);
    std::vector<void*> frees;
    for (int k = 0; k < count; ++k) {
        auto cur = randv(size_t(rows) * n, 1.0f), msg = randv(size_t(rows) * n, 2.0f);
        auto wd = randv(size_t(rank) * 2 * n, 0.02f), wg = randv(size_t(n) * rank, 0.2f),
             wf = randv(size_t(n) * rank, 0.2f);
        host_control(cur, msg, wd, wg, wf, n, rank, rows, ref.data() + size_t(k) * rows * n);
        float *dc = up(cur), *dm = up(msg), *dd = up(wd), *dg = up(wg), *df = up(wf), *out = nullptr;
        ck(cudaMalloc(&out, cur.size() * 4), "malloc");
        c[k].cur = dc, c[k].msg = dm, c[k].out = k % 2 ? dc : out;   // in place for every other one
        c[k].down = dd, c[k].gain = dg, c[k].forget = df;
        frees.insert(frees.end(), {dc, dm, dd, dg, df, out});
    }
    float* work = nullptr;
    ck(cudaMalloc(&work, size_t(count) * rows * rank * 4), "malloc");
    inkling_control_fused(c.data(), count, n, rank, rows, 1e-6f, work, nullptr);
    std::vector<float> got;
    for (int k = 0; k < count; ++k) {
        auto o = down(c[k].out, size_t(rows) * n);
        got.insert(got.end(), o.begin(), o.end());
    }
    const std::string name = std::string("control fused ") + std::to_string(count) + " ctrl, rows " +
                             std::to_string(rows);
    compare(name.c_str(), got, ref, 1e-4);
    for (void* p : frees) cudaFree(p);
    cudaFree(work);
}

void test_gemv(int n_in, int n_out, int rows) {
    auto w = randv(size_t(n_in) * n_out, 0.05f), x = randv(size_t(rows) * n_in, 1.0f);
    float *dw = up(w), *dx = up(x), *dy = nullptr;
    ck(cudaMalloc(&dy, size_t(rows) * n_out * 4), "malloc");
    inkling_gemv(dw, dx, dy, n_in, n_out, rows, nullptr);
    std::vector<double> ref(size_t(rows) * n_out);
    for (int t = 0; t < rows; ++t)
        for (int o = 0; o < n_out; ++o) {
            double a = 0;
            for (int i = 0; i < n_in; ++i) a += double(w[size_t(o) * n_in + i]) * x[size_t(t) * n_in + i];
            ref[size_t(t) * n_out + o] = a;
        }
    const std::string name = "gemv " + std::to_string(n_in) + " -> " + std::to_string(n_out) + ", rows " +
                             std::to_string(rows);
    compare(name.c_str(), down(dy, ref.size()), ref, 1e-4);
    cudaFree(dw), cudaFree(dx), cudaFree(dy);
}

// the bits of two device buffers of floats
void same_bits(const char* what, const float* a, const float* b, size_t n) {
    auto x = down(a, n), y = down(b, n);
    size_t diff = 0;
    for (size_t i = 0; i < n; ++i) diff += std::memcmp(&x[i], &y[i], 4) != 0;
    std::printf("%-58s %zu of %zu values differ  %s\n", what, diff, n, diff ? "FAIL" : "ok");
    if (diff) ++failures;
}

void test_clip_many(int rows) {
    const InklingTau tau{0.1f, 8192};
    auto q = randv(size_t(rows) * 3072, 2.0f), r = randv(size_t(rows) * 384, 2.0f), qs = randv(3072, 0.3f, 1.f),
         rs = randv(384, 0.3f, 1.f);
    float *dq = up(q), *dr = up(r), *dqs = up(qs), *drs = up(rs), *y1, *y2;
    ck(cudaMalloc(&y1, (q.size() + r.size()) * 4), "malloc"), ck(cudaMalloc(&y2, (q.size() + r.size()) * 4), "malloc");
    inkling_clip(dq, 0, dqs, y1, 128, 24, rows, 22.6f, tau, 20000, nullptr);
    inkling_clip(dr, 0, drs, y1 + q.size(), 16, 24, rows, 8.f, tau, 20000, nullptr);
    InklingClipJob j[2];
    j[0].x = dq, j[0].scale = dqs, j[0].y = y2, j[0].dim = 128, j[0].heads = 24, j[0].threshold = 22.6f, j[0].tau = tau;
    j[1].x = dr, j[1].scale = drs, j[1].y = y2 + q.size(), j[1].dim = 16, j[1].heads = 24, j[1].threshold = 8.f,
    j[1].tau = tau;
    inkling_clip_many(j, 2, rows, 20000, nullptr);
    same_bits(("clip_many Q + R vs two inkling_clip, rows " + std::to_string(rows)).c_str(), y1, y2,
              q.size() + r.size());
    cudaFree(dq), cudaFree(dr), cudaFree(dqs), cudaFree(drs), cudaFree(y1), cudaFree(y2);
}

void test_conv_acc(int channels, int rows) {
    auto x = randv(size_t(rows) * channels, 1.f), w = randv(size_t(channels) * 4, 0.4f),
         st = randv(size_t(channels) * 3, 1.f), acc = randv(size_t(rows) * channels, 1.f);
    float *dx = up(x), *dw = up(w), *s1 = up(st), *s2 = up(st), *a1 = up(acc), *a2 = up(acc), *tmp;
    ck(cudaMalloc(&tmp, x.size() * 4), "malloc");
    inkling_conv(dx, s1, dw, nullptr, 0, tmp, channels, rows, nullptr);
    axpy(tmp, 1.f, a1, rows * channels, true, nullptr);
    inkling_conv_acc(dx, s2, dw, nullptr, 0, a2, channels, rows, nullptr);
    same_bits(("conv_acc vs conv + axpy, rows " + std::to_string(rows)).c_str(), a1, a2, acc.size());
    same_bits(("conv_acc state, rows " + std::to_string(rows)).c_str(), s1, s2, st.size());
    cudaFree(dx), cudaFree(dw), cudaFree(s1), cudaFree(s2), cudaFree(a1), cudaFree(a2), cudaFree(tmp);
}

// inkling_kv_store() against the separate calls: its K and V floats against the host's double reference, the
// state to the bit, and its cache rows to the bit against store_f16 / store_q8 of its own floats.
void test_kv_store(int rows, bool q8, bool taps, int ring, long long pos) {
    const int kvh = 8, dim = 128, width = kvh * dim, cap = ring ? ring : int(pos) + rows;
    auto kraw = randv(size_t(rows) * width, 1.f), vraw = randv(size_t(rows) * width, 1.f),
         kw = randv(size_t(width) * 4, 0.4f), vw = randv(size_t(width) * 4, 0.4f),
         kt = randv(size_t(rows) * kvh * 4, 0.3f), vt = randv(size_t(rows) * kvh * 4, 0.3f),
         st = randv(size_t(width) * 6, 1.f);
    for (size_t i = 0; i < kraw.size(); i += 7) kraw[i] *= 8.f;   // some heads past the clip
    float *dk = up(kraw), *dv = up(vraw), *dkw = up(kw), *dvw = up(vw), *dkt = up(kt), *dvt = up(vt), *dst = up(st);
    float *ko, *vo;
    ck(cudaMalloc(&ko, kraw.size() * 4), "malloc"), ck(cudaMalloc(&vo, vraw.size() * 4), "malloc");
    Cache c;
    c.alloc(cap, q8);
    InklingKvDst d;
    d.k = c.k, d.v = c.v, d.kc = c.kc, d.ks = c.ks, d.vc = c.vc, d.vs = c.vs, d.ring = ring;
    inkling_kv_store(dk, dv, dst, dst + size_t(width) * 3, dkw, dvw, taps ? dkt : nullptr, taps ? dvt : nullptr, kvh,
                     dim, rows, 22.6f, 1e-6f, d, pos, ko, vo, nullptr);
    // the host's reference of the floats and the state
    std::vector<double> hst(st.begin(), st.end()), kref(kraw.size()), vref(vraw.size());
    std::vector<double> ks(hst.begin(), hst.begin() + width * 3), vs(hst.begin() + width * 3, hst.end());
    host_conv(kraw, ks, kw, taps ? &kt : nullptr, kvh, kref, width, rows);
    host_conv(vraw, vs, vw, taps ? &vt : nullptr, kvh, vref, width, rows);
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < kvh; ++h) {
            double* kk = kref.data() + (size_t(t) * kvh + h) * dim;
            double* vv = vref.data() + (size_t(t) * kvh + h) * dim;
            double sk = 0, sv = 0;
            for (int i = 0; i < dim; ++i) sk += kk[i] * kk[i], sv += vv[i] * vv[i];
            const double f = std::min(1.0, 22.6 / (std::sqrt(sk) + 1e-6)), r = 1.0 / std::sqrt(sv / dim + 1e-6);
            for (int i = 0; i < dim; ++i) kk[i] *= f, vv[i] *= r;
        }
    const std::string name = std::string("kv_store rows ") + std::to_string(rows) + (q8 ? " q8_0" : " fp16") +
                             (taps ? " taps" : "") + (ring ? " ring" : "");
    compare((name + ": K").c_str(), down(ko, kraw.size()), kref, 2e-5);
    compare((name + ": V").c_str(), down(vo, vraw.size()), vref, 2e-5);
    std::vector<double> hst2(ks.begin(), ks.end());
    hst2.insert(hst2.end(), vs.begin(), vs.end());
    compare((name + ": state").c_str(), down(dst, st.size()), hst2, 0.0);
    // the cache rows against the stores of its own floats
    Cache c2;
    c2.alloc(cap, q8);
    for (int side = 0; side < 2; ++side) {
        const float* x = side ? vo : ko;
        for (int done = 0; done < rows;) {
            const long long p = pos + done, r = ring ? p % ring : p;
            const int m = ring ? int(std::min<long long>(rows - done, ring - r)) : rows - done;
            if (q8)
                store_q8(x + size_t(done) * width, side ? c2.vc : c2.kc, side ? c2.vs : c2.ks, width, r, m, nullptr);
            else store_f16(x + size_t(done) * width, side ? c2.v : c2.k, width, r, m, nullptr);
            done += m;
        }
    }
    size_t diff = 0, n = 0;
    auto cmp = [&](const void* a, const void* b, size_t bytes) {
        std::vector<uint8_t> x(bytes), y(bytes);
        ck(cudaDeviceSynchronize(), "sync");
        cudaMemcpy(x.data(), a, bytes, cudaMemcpyDeviceToHost), cudaMemcpy(y.data(), b, bytes, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < bytes; ++i) diff += x[i] != y[i];
        n += bytes;
    };
    const size_t vals = size_t(cap) * width;
    if (q8) cmp(c.kc, c2.kc, vals), cmp(c.vc, c2.vc, vals), cmp(c.ks, c2.ks, vals / 16), cmp(c.vs, c2.vs, vals / 16);
    else cmp(c.k, c2.k, vals * 2), cmp(c.v, c2.v, vals * 2);
    std::printf("%-58s %zu of %zu bytes differ  %s\n", (name + ": cache vs store of its floats").c_str(), diff, n,
                diff ? "FAIL" : "ok");
    if (diff) ++failures;
    c.free_(), c2.free_();
    cudaFree(dk), cudaFree(dv), cudaFree(dkw), cudaFree(dvw), cudaFree(dkt), cudaFree(dvt), cudaFree(dst), cudaFree(ko),
        cudaFree(vo);
}

// Timings at decode shapes (with `bench`): the per-token kernels as the engine calls them for one token.  Each
// case runs `reps` times as one CUDA graph (no host launch cost between the calls), replayed 5 times; the fastest
// replay is reported.  The functions take the stream the graph is captured on.
cudaStream_t bs = nullptr;
template <class F> void time_us(const char* what, F&& f, int reps = 200) {
    if (!bs) cudaStreamCreateWithFlags(&bs, cudaStreamNonBlocking);
    ck(cudaDeviceSynchronize(), "before the timing");   // the setup ran on the legacy stream
    f();   // warm-up
    ck(cudaStreamSynchronize(bs), "warm-up");
    cudaGraph_t gr;
    cudaGraphExec_t ex;
    ck(cudaStreamBeginCapture(bs, cudaStreamCaptureModeRelaxed), "capture");
    for (int i = 0; i < reps; ++i) f();
    ck(cudaStreamEndCapture(bs, &gr), "end capture");
    ck(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    float best = 1e30f;
    for (int k = 0; k < 6; ++k) {
        cudaEventRecord(e0, bs);
        cudaGraphLaunch(ex, bs);
        cudaEventRecord(e1, bs);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (k) best = std::min(best, ms);   // the first replay uploads the graph
    }
    std::printf("bench %-58s %9.2f us\n", what, 1000.f * best / reps);
    cudaEventDestroy(e0), cudaEventDestroy(e1);
    cudaGraphExecDestroy(ex), cudaGraphDestroy(gr);
}

void bench_decode(const std::vector<float>& table, float* dtable) {
    const int n = 2304;
    // mix: 5 states, 1 / 3 / 4 queries, one token; 5 states with 4 queries for 7 tokens
    {
        std::vector<float*> st;
        for (int k = 0; k < 5; ++k) st.push_back(up(randv(size_t(7) * n, 1.0f)));
        float* qd = up(randv(size_t(4) * n, 1.0f));
        float* out[4];
        for (auto& o : out) ck(cudaMalloc(&o, size_t(7) * n * 4), "malloc");
        for (int nq : {1, 3, 4})
            for (int rows : {1, 7}) {
                InklingMix m;
                m.n_state = 5, m.n_query = nq, m.query = qd;
                for (int k = 0; k < 5; ++k) m.state[k] = st[k];
                for (int q = 0; q < nq; ++q) m.out[q] = out[q];
                const std::string w = "mix 5 states, " + std::to_string(nq) + " queries, " + std::to_string(rows) +
                                      " token(s)";
                time_us(w.c_str(), [&] { inkling_mix(m, n, rows, 1e-6f, bs); });
            }
        for (float* p : st) cudaFree(p);
        for (float* o : out) cudaFree(o);
        cudaFree(qd);
    }
    // clip and conv, one token
    {
        float *x = up(randv(size_t(n) * 8, 1.0f)), *y = nullptr, *sc = up(randv(3072, 1.0f)),
              *state = up(randv(size_t(n) * 3, 1.0f)), *w = up(randv(size_t(n) * 4, 0.3f)),
              *taps = up(randv(64, 0.3f));
        ck(cudaMalloc(&y, size_t(n) * 8 * 4), "malloc");
        time_us("clip Q 128 x 24, scale, tau, 1 token", [&] {
            inkling_clip(x, 0, sc, y, 128, 24, 1, 22.6f, InklingTau{0.1f, 8192}, 100, bs);
        });
        time_us("clip K 128 x 8, 1 token", [&] { inkling_clip(x, 0, nullptr, y, 128, 8, 1, 22.6f, {}, 100, bs); });
        time_us("clip R 16 x 24, 1 token", [&] { inkling_clip(x, 0, sc, y, 16, 24, 1, 8.f, {}, 100, bs); });
        time_us("conv 1024, taps 8 heads, 1 token", [&] { inkling_conv(x, state, w, taps, 8, y, 1024, 1, bs); });
        time_us("conv 2304, 1 token", [&] { inkling_conv(x, state, w, nullptr, 0, y, 2304, 1, bs); });
        cudaFree(x), cudaFree(y), cudaFree(sc), cudaFree(state), cudaFree(w), cudaFree(taps);
    }
    // the fused steps against the launches they replace, one token
    {
        const int rank = 64;
        std::vector<float*> fr;
        auto mk = [&](size_t count, float sd) { float* p = up(randv(count, sd)); fr.push_back(p); return p; };
        InklingCtrl c[4];
        float* work = mk(size_t(4) * 4 * (4 * n + rank), 0.f);
        for (int k = 0; k < 4; ++k) {
            c[k].cur = mk(n, 1.f), c[k].msg = mk(n, 1.f), c[k].out = mk(n, 0.f);
            c[k].down = mk(size_t(rank) * 2 * n, 0.02f), c[k].gain = mk(size_t(n) * rank, 0.2f);
            c[k].forget = mk(size_t(n) * rank, 0.2f);
        }
        auto separate_ctrl = [&](int count) {
            for (int k = 0; k < count; ++k)
                inkling_control(nullptr, c[k].cur, c[k].msg, c[k].down, c[k].gain, c[k].forget, work, c[k].out, n,
                                rank, 1, 1e-6f, bs);
        };
        time_us("control x1: inkling_control (6 launches)", [&] { separate_ctrl(1); });
        time_us("control x1: inkling_control_fused", [&] { inkling_control_fused(c, 1, n, rank, 1, 1e-6f, work, bs); });
        time_us("control x3: 3 x inkling_control", [&] { separate_ctrl(3); });
        time_us("control x3: one inkling_control_fused", [&] {
            inkling_control_fused(c, 3, n, rank, 1, 1e-6f, work, bs);
        });
        // the router's product
        float *rw = mk(size_t(n) * 128, 0.05f), *x = mk(n * 4, 1.f), *y = mk(128 * 4, 0.f);
        time_us("router 2304 -> 128: gemv_f32", [&] { gemv_f32(rw, x, y, n, 128, 128, 1, bs); });
        time_us("router 2304 -> 128: inkling_gemv", [&] { inkling_gemv(rw, x, y, n, 128, 1, bs); });
        // Q and R clips
        float *q = mk(3072, 2.f), *qs = mk(3072, 0.3f), *r = mk(384, 2.f), *rs = mk(384, 0.3f);
        const InklingTau tau{0.1f, 8192};
        time_us("clip Q + R: 2 x inkling_clip", [&] {
            inkling_clip(q, 0, qs, q, 128, 24, 1, 22.6f, tau, 100, bs);
            inkling_clip(r, 0, rs, r, 16, 24, 1, 8.f, tau, 100, bs);
        });
        InklingClipJob j[2];
        j[0].x = q, j[0].scale = qs, j[0].y = q, j[0].dim = 128, j[0].heads = 24, j[0].threshold = 22.6f;
        j[0].tau = tau;
        j[1].x = r, j[1].scale = rs, j[1].y = r, j[1].dim = 16, j[1].heads = 24, j[1].threshold = 8.f;
        j[1].tau = tau;
        time_us("clip Q + R: inkling_clip_many", [&] { inkling_clip_many(j, 2, 1, 100, bs); });
        // K and V to the cache
        const int width = 1024, cap = 2048;
        float *kraw = mk(width, 1.f), *vraw = mk(width, 1.f), *kst = mk(width * 3, 1.f), *vst = mk(width * 3, 1.f),
              *kw = mk(width * 4, 0.3f), *vw = mk(width * 4, 0.3f), *kt = mk(32, 0.3f), *vt = mk(32, 0.3f),
              *ko = mk(width, 0.f), *vo = mk(width, 0.f);
        int8_t *kc, *vc;
        uint16_t *ks, *vs;
        ck(cudaMalloc(&kc, size_t(cap) * width), "malloc"), ck(cudaMalloc(&vc, size_t(cap) * width), "malloc");
        ck(cudaMalloc(&ks, size_t(cap) * width / 16), "malloc");
        ck(cudaMalloc(&vs, size_t(cap) * width / 16), "malloc");
        fr.insert(fr.end(), {(float*) kc, (float*) vc, (float*) ks, (float*) vs});
        time_us("K, V to the q8_0 cache: the separate calls (6 launches)", [&] {
            inkling_conv(kraw, kst, kw, kt, 8, ko, width, 1, bs);
            inkling_conv(vraw, vst, vw, vt, 8, vo, width, 1, bs);
            inkling_clip(ko, 0, nullptr, ko, 128, 8, 1, 22.6f, {}, 100, bs);
            rms_norm(vo, nullptr, vo, 128, 8, 1e-6f, bs);
            store_q8(ko, kc, ks, width, 100, 1, bs);
            store_q8(vo, vc, vs, width, 100, 1, bs);
        });
        InklingKvDst d;
        d.kc = kc, d.ks = ks, d.vc = vc, d.vs = vs;
        time_us("K, V to the q8_0 cache: inkling_kv_store", [&] {
            inkling_kv_store(kraw, vraw, kst, vst, kw, vw, kt, vt, 8, 128, 1, 22.6f, 1e-6f, d, 100, ko, vo, bs);
        });
        // a post-convolution and its residual add
        float *xo = mk(n, 1.f), *pst = mk(n * 3, 1.f), *pw = mk(n * 4, 0.3f), *tmp = mk(n, 0.f), *res = mk(n, 1.f);
        time_us("post-conv + residual: inkling_conv + axpy", [&] {
            inkling_conv(xo, pst, pw, nullptr, 0, tmp, n, 1, bs);
            axpy(tmp, 1.f, res, n, true, bs);
        });
        time_us("post-conv + residual: inkling_conv_acc", [&] {
            inkling_conv_acc(xo, pst, pw, nullptr, 0, res, n, 1, bs);
        });
        ck(cudaDeviceSynchronize(), "fused bench");
        for (float* p : fr) cudaFree(p);
    }
    // attention, one token (and 7), over q8_0 and fp16 caches
    {
        const int ctx = 32768, window = 1024;
        auto kv = randv(size_t(ctx) * kKv * kDim, 1.0f);
        float* dkv = up(kv);
        Cache cf, c8;
        cf.alloc(ctx, false), c8.alloc(ctx, true);
        for (Cache* c : {&cf, &c8}) {
            const int width = kKv * kDim;
            if (c->q8) {
                store_q8(dkv, c->kc, c->ks, width, 0, ctx, nullptr);
                store_q8(dkv, c->vc, c->vs, width, 0, ctx, nullptr);
            } else {
                store_f16(dkv, c->k, width, 0, ctx, nullptr);
                store_f16(dkv, c->v, width, 0, ctx, nullptr);
            }
        }
        const int rows = 7;
        float *q = up(randv(size_t(rows) * kHeads * kDim, 1.0f)), *r = up(randv(size_t(rows) * kHeads * kRel, 1.0f)),
              *s = up(randv(size_t(rows) * kHeads, 1.0f)), *out = nullptr, *work = nullptr;
        ck(cudaMalloc(&out, size_t(rows) * kHeads * kDim * 4), "malloc");
        const size_t wn = inkling_attend_work_floats(kHeads, kDim);
        ck(cudaMalloc(&work, wn * 4), "malloc");
        const float sc = 1.f / std::sqrt((float) kDim);
        const InklingRel rel{r, dtable, kRel, kExtent};
        struct Case {
            const char* name;
            bool q8;
            int pos, rows, window;
            long long room;   // 0: pos + rows
        };
        const Case cases[] = {
            {"q8_0 full, 1 token at 25", true, 25, 1, 0, 0},
            {"q8_0 full, 1 token at 150", true, 150, 1, 0, 0},
            {"q8_0 swa,  1 token at 150", true, 150, 1, window, 0},
            {"q8_0 full, 1 token at 150, room 1024", true, 150, 1, 0, 1024},
            {"q8_0 swa,  1 token at 150, room 1024", true, 150, 1, window, 1024},
            {"q8_0 full, 1 token at 150, room 131072", true, 150, 1, 0, 131072},
            {"q8_0 full, 1 token at 1024", true, 1024, 1, 0, 0},
            {"q8_0 full, 1 token at 4096", true, 4096, 1, 0, 0},
            {"q8_0 full, 1 token at 32767", true, 32767, 1, 0, 0},
            {"q8_0 swa,  1 token at 32767 (ring)", true, 32767, 1, window, 0},
            {"q8_0 swa,  7 tokens at 32760 (ring)", true, 32760, 7, window, 0},
            {"fp16 full, 1 token at 150", false, 150, 1, 0, 0},
            {"fp16 full, 1 token at 1024", false, 1024, 1, 0, 0},
            {"fp16 full, 1 token at 32767", false, 32767, 1, 0, 0},
            {"fp16 swa,  1 token at 32767 (ring)", false, 32767, 1, window, 0},
        };
        for (const Case& k : cases) {
            const Cache& c = k.q8 ? c8 : cf;
            const Kvq8 kq = k.q8 ? Kvq8{c.kc, c.ks} : Kvq8{}, vq = k.q8 ? Kvq8{c.vc, c.vs} : Kvq8{};
            // a window layer reads the full cache as its ring (any ring >= window + rows is the same work)
            const int rg = k.window ? ctx : 0;
            const std::string w = std::string("attend ") + k.name;
            time_us(w.c_str(), [&] {
                inkling_attend(q, c.k, c.v, kq, vq, rel, s, k.pos, k.rows, out, kDim, kHeads, kKv, sc, k.window, rg,
                               work, wn, bs, k.room);
            }, 50);
        }
        cf.free_(), c8.free_();
        cudaFree(dkv), cudaFree(q), cudaFree(r), cudaFree(s), cudaFree(out), cudaFree(work);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "bench";
    const bool only_bench = argc > 1 && std::string(argv[1]) == "decode";
    int devices = 0;
    if (const cudaError_t e = cudaGetDeviceCount(&devices); e != cudaSuccess || devices == 0) {
        std::printf("no usable CUDA device (%s): skipped\n", e != cudaSuccess ? cudaGetErrorString(e) : "none");
        return 77;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    if (only_bench) {   // decode: only the decode-shape timings
        std::vector<float> table = randv(size_t(kExtent) * kRel, 0.3f);
        float* dtable = up(table);
        bench_decode(table, dtable);
        cudaFree(dtable);
        return 0;
    }
    const InklingTau tau{0.1f, 8192};
    test_clip("clip Q (128 x 24, scale, tau, pos 30000, 7 rows)", 128, 24, 7, true, 22.627417f, tau, 30000, true);
    test_clip("clip Q (128 x 24, scale, tau, pos 123456, 1 row)", 128, 24, 1, true, 22.627417f, tau, 123456, false);
    test_clip("clip K (128 x 8, no scale, 1500 rows)", 128, 8, 1500, false, 22.627417f, {}, 0, true);
    test_clip("clip R (16 x 24, scale, tau, pos 8000, 1100 rows)", 16, 24, 1100, true, 8.0f, tau, 8000, true);
    test_clip("clip R (16 x 24, scale, no tau, 1 row)", 16, 24, 1, true, 8.0f, {}, 5, true);

    test_conv("conv K (1024, taps 8 heads)", 1024, 8, true, {1, 7, 1, 40, 3});
    test_conv("conv V (1024, taps 8 heads)", 1024, 8, true, {5, 7});
    test_conv("conv attn-out (2304, no taps)", 2304, 1, false, {1, 7, 1100, 1});
    test_conv("conv ffn-out (2304, no taps)", 2304, 1, false, {3, 4});

    test_mix(1, 1, 3);
    test_mix(2, 1, 1);
    test_mix(5, 3, 7);
    test_mix(4, 4, 33);
    test_mix(8, 1, 2);
    test_mix(8, 4, 3, 128);      // one warp a token: fewer threads than the 40 sums
    test_mix(3, 2, 2, 100);
    test_mix(4, 3, 5, 2303);     // n not a multiple of 4: the scalar kernel
    test_mix(5, 4, 3, 2304, 1);  // states not 16-byte aligned: the scalar kernel

    void* blas = blas_create(nullptr);
    test_control(blas, 1);
    test_control(blas, 7);
    test_control(blas, 20);
    blas_destroy(blas);

    test_situ();

    for (int count : {1, 3, 4})
        for (int rows : {1, 3, 7}) test_control_fused(count, rows);
    test_control_fused(1, 5);
    for (int rows : {1, 4, 7}) test_gemv(2304, 128, rows);
    test_gemv(4608, 64, 2);
    for (int rows : {1, 7, 40}) test_clip_many(rows);
    for (int rows : {1, 7, 40}) test_conv_acc(2304, rows);
    test_kv_store(1, true, true, 0, 150);
    test_kv_store(1, true, false, 1030, 2047);
    test_kv_store(7, true, true, 1030, 1026);   // wraps the ring
    test_kv_store(7, false, false, 1030, 1026);
    test_kv_store(3, false, true, 0, 5);
    test_kv_store(16, false, true, 0, 40);      // the most tokens of the one launch
    test_kv_store(17, false, true, 0, 40);      // the fewest of the separate calls
    test_kv_store(20, true, true, 1030, 1020);
    test_kv_store(20, false, false, 0, 0);

    std::vector<float> table = randv(size_t(kExtent) * kRel, 0.3f);
    float* dtable = up(table);
    const std::vector<AttnCase> cases = {
        {"attend full fp16 split", 0, 0, false, {7, 1, 1}, true, false, true, 1},
        {"attend full fp16 split, 1 token at 0", 0, 0, false, {1, 1, 16, 17}, true, false, true, 1},
        {"attend full fp16 split, room, 1 token at 0", 0, 0, true, {1, 16, 17}, true, true, true, 1},
        {"attend full fp16 unsplit", 0, 0, false, {7, 1}, false, false, true, 1},
        {"attend full fp16 unsplit, 1 token at 0", 0, 0, false, {1, 3, 2, 17}, false, false, true, 1},
        {"attend full fp16 prompt >1024", 0, 0, false, {1100, 7, 1}, true, false, true, 23},
        {"attend full fp16 prompt, room", 0, 0, false, {1300, 1, 7}, true, true, true, 41},
        {"attend full q8_0", 0, 0, true, {1050, 1, 7}, true, false, true, 37},
        {"attend full fp16 no sinks", 0, 0, false, {33, 2}, true, false, false, 1},
        {"attend swa fp16 ring", 1024, 1024 + 2048, false, {2048, 1, 7, 2048, 1}, true, false, true, 29},
        {"attend swa fp16 ring, room", 1024, 1024 + 1100, false, {1100, 1, 7, 1100, 1}, true, true, true, 31},
        {"attend swa q8_0 ring unsplit", 1024, 1024 + 1500, true, {1500, 1, 7, 1500}, false, false, true, 43},
        {"attend swa q8_0 ring", 1024, 1024 + 1500, true, {1030, 7, 1}, true, true, true, 43},
    };
    for (const AttnCase& tc : cases) test_attend(tc, table, dtable);
    test_attend_alone(table, dtable);
    if (bench) bench_attend(table, dtable), bench_decode(table, dtable);
    cudaFree(dtable);

    ck(cudaDeviceSynchronize(), "final");
    std::printf("%s: %d failure%s\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
