// Standalone test: strata::poly::gdn_step() (the gated delta net's step) on the GPU vs a CPU reference in FP64.
//
// gdn_step is four launches: gdn_norm_kernel (q and k normed), gdn_gates_kernel (decay and share), the state's
// update (gdn_state_reg_kernel<PER> for a head of 32, 64, 128 or 256 values, gdn_state_kernel for the other
// multiples of 32), and gdn_out_kernel (the output's norm and gate).  The reference is the recurrence written out in
// double, from the documentation in kernels.hpp and the kernels' formulas:
//   q = q / sqrt(sum q^2 + eps) / sqrt(dim),  k = k / sqrt(sum k^2 + eps)        (per token and key head)
//   decay = exp(softplus(alpha + dt) * a),  share = sigmoid(beta)                 (per token and value head)
//   for a value head h (reads key head h % k_heads) and a column j of its dim x dim state S (S[j][i], key dim i):
//     sk = sum_i S[j][i] * decay * k[i];  d = (v[j] - sk) * share;
//     S[j][i] = S[j][i] * decay + k[i] * d;  o[j] = sum_i S[j][i] * q[i]
//   out[j] = o[j] / sqrt(sum_j o[j]^2 / dim + eps) * o_norm[j] * silu(z[j])
// It exists because dpct turned __fmul_rn(__fsub_rn(va, sk), sa) into va - sk * sa (DPCT1013) in the register
// kernel: with sa = sigmoid(beta) that is a different recurrence, which this test fails by O(1).  The negative
// control is to build kernels.dp.cpp with that line reverted (`const float d = va - sk * sa;`): the cases with a
// head of 32, 64, 128 or 256 values then fail, the ones with the plain kernel (96, 160, 192) still pass.
// Also checked: the final state, and the snapshots (the state after the tokens `snaps` has a bit for).
// Tolerance: 1e-3 of the largest magnitude of the reference, per output (floats, native exp, up to 33 steps).
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <random>
#include <vector>

#include "strata/poly/kernels.hpp"

namespace {

struct Case {
    int dim, n_head, k_heads, rows;
    unsigned snaps;      // bit r: the state after token r is also wanted
    bool zero_state;
    const char* what;
};

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct Worst {
    double err = 0, scale = 0;
    long bad = 0;
};

void compare(Worst& w, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
    double mx = 0;
    for (double v : ref) mx = std::max(mx, std::fabs(v));
    w.scale = std::max(w.scale, mx);
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = std::fabs((double) got[i] - ref[i]) / std::max(mx, 1e-6);
        if (!std::isfinite((double) got[i]) || e > tol) ++w.bad;
        w.err = std::max(w.err, std::isfinite((double) got[i]) ? e : 1e30);
    }
}

bool run_case(sycl::queue& q, const Case& c, uint32_t seed) {
    const int dim = c.dim, nh = c.n_head, kh = c.k_heads, rows = c.rows;
    const size_t keys = (size_t) dim * kh, d_inner = (size_t) dim * nh, row_n = 2 * keys + d_inner;
    const double eps = 1e-6;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);

    std::vector<float> qkv(rows * row_n), alpha((size_t) rows * nh), beta((size_t) rows * nh), dt(nh), a(nh),
        s0((size_t) nh * dim * dim), z(rows * d_inner), o_norm(dim);
    for (auto& v : qkv) v = nd(rng);
    for (auto& v : alpha) v = ud(rng) < 0.03f ? 25.f + 5.f * ud(rng) : nd(rng);   // a few above the softplus cutoff
    for (auto& v : beta) v = nd(rng);
    for (auto& v : dt) v = 0.5f * nd(rng);
    for (auto& v : a) v = -(0.1f + 2.f * ud(rng));                                  // a decay below 1, as A_log gives
    for (auto& v : s0) v = c.zero_state ? 0.f : 0.1f * nd(rng);
    for (auto& v : z) v = nd(rng);
    for (auto& v : o_norm) v = 1.f + 0.1f * nd(rng);

    // ---- reference ------------------------------------------------------------------------------------------------
    std::vector<double> S(s0.begin(), s0.end()), out_ref(rows * d_inner);
    std::vector<std::vector<double>> snap_ref;
    for (int t = 0; t < rows; ++t) {
        const float* x = qkv.data() + t * row_n;
        std::vector<double> qn(keys), kn(keys);
        for (int g = 0; g < kh; ++g) {
            double qs = 0, ks = 0;
            for (int i = 0; i < dim; ++i) {
                qs += (double) x[g * dim + i] * x[g * dim + i];
                ks += (double) x[keys + g * dim + i] * x[keys + g * dim + i];
            }
            for (int i = 0; i < dim; ++i) {
                qn[g * dim + i] = x[g * dim + i] / std::sqrt(qs + eps) / std::sqrt((double) dim);
                kn[g * dim + i] = x[keys + g * dim + i] / std::sqrt(ks + eps);
            }
        }
        for (int h = 0; h < nh; ++h) {
            const double g = (double) alpha[t * nh + h] + dt[h];
            const double decay = std::exp((g > 20.0 ? g : std::log(1.0 + std::exp(g))) * a[h]);
            const double share = sigmoid(beta[t * nh + h]);
            const int hk = h % kh;
            std::vector<double> o(dim);
            for (int j = 0; j < dim; ++j) {
                double* col = S.data() + ((size_t) h * dim + j) * dim;
                double sk = 0;
                for (int i = 0; i < dim; ++i) sk += col[i] * decay * kn[hk * dim + i];
                const double d = ((double) x[2 * keys + (size_t) h * dim + j] - sk) * share;
                double oj = 0;
                for (int i = 0; i < dim; ++i) {
                    col[i] = col[i] * decay + kn[hk * dim + i] * d;
                    oj += col[i] * qn[hk * dim + i];
                }
                o[j] = oj;
            }
            double os = 0;
            for (int j = 0; j < dim; ++j) os += o[j] * o[j];
            for (int j = 0; j < dim; ++j) {
                const double zz = z[t * d_inner + (size_t) h * dim + j];
                out_ref[t * d_inner + (size_t) h * dim + j] = o[j] / std::sqrt(os / dim + eps) * o_norm[j] * (zz * sigmoid(zz));
            }
        }
        if (t < 32 && (c.snaps >> t & 1u)) snap_ref.push_back(S);
    }

    // ---- the GPU ----------------------------------------------------------------------------------------------------
    const size_t state_n = (size_t) nh * dim * dim;
    const int nsnap = (int) snap_ref.size();
    auto up = [&](const std::vector<float>& v) {
        float* p = sycl::malloc_device<float>(std::max<size_t>(v.size(), 1), q);
        q.memcpy(p, v.data(), v.size() * sizeof(float)).wait();
        return p;
    };
    float *dqkv = up(qkv), *dalpha = up(alpha), *dbeta = up(beta), *ddt = up(dt), *da = up(a), *ds = up(s0), *dz = up(z),
          *don = up(o_norm);
    float* dout = sycl::malloc_device<float>(rows * d_inner, q);
    float* dsnap = nsnap ? sycl::malloc_device<float>((size_t) nsnap * state_n, q) : nullptr;
    bool ok = true;
    Worst wo, ws, wn;
    try {
        strata::poly::gdn_step(dqkv, dalpha, dbeta, ddt, da, ds, dz, don, dout, dim, nh, kh, rows, (float) eps, &q, dsnap,
                               state_n, c.snaps);
        q.wait_and_throw();
        std::vector<float> out(rows * d_inner), st(state_n);
        q.memcpy(out.data(), dout, out.size() * sizeof(float)).wait();
        q.memcpy(st.data(), ds, st.size() * sizeof(float)).wait();
        const double tol = 1e-3;
        compare(wo, out, out_ref, tol);
        compare(ws, st, S, tol);
        for (int k = 0; k < nsnap; ++k) {
            std::vector<float> sn(state_n);
            q.memcpy(sn.data(), dsnap + (size_t) k * state_n, state_n * sizeof(float)).wait();
            compare(wn, sn, snap_ref[k], tol);
        }
        ok = wo.bad == 0 && ws.bad == 0 && wn.bad == 0;
    } catch (const std::exception& e) {
        std::printf("%-44s EXCEPTION %s\n", c.what, e.what());
        ok = false;
    }
    std::printf("%-44s dim=%-3d heads=%-2d/%-2d rows=%-2d snaps=%d  out err %.2e (%ld bad) | state err %.2e (%ld bad) | "
                "snaps err %.2e (%ld bad)  %s\n",
                c.what, dim, nh, kh, rows, nsnap, wo.err, wo.bad, ws.err, ws.bad, wn.err, wn.bad, ok ? "PASS" : "FAIL");
    for (float* p : {dqkv, dalpha, dbeta, ddt, da, ds, dz, don, dout, dsnap}) if (p) sycl::free(p, q);
    return ok;
}

}  // namespace

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const Case cases[] = {
        // the register kernel: dim / 32 = 1, 2, 4, 8 values a lane
        {32, 4, 2, 1, 0, false, "reg<1>, one token"},
        {32, 6, 2, 9, 0, false, "reg<1>, 9 tokens"},
        {64, 8, 4, 5, 0, false, "reg<2>, zero start state"},
        {64, 8, 4, 17, 0x00015u, false, "reg<2>, snapshots of tokens 0, 2, 4"},
        {128, 8, 2, 12, 0, false, "reg<4>"},
        {128, 4, 4, 33, 0x80000001u, false, "reg<4>, 33 tokens, snapshot of 0 and 31"},
        {256, 4, 2, 7, 0, false, "reg<8>"},
        {128, 6, 3, 8, 0, true, "reg<4>, state starts at zero"},
        // the plain kernel (default of the switch): 96, 160, 192, 224
        {96, 6, 2, 6, 0, false, "plain, dim 96"},
        {160, 4, 2, 5, 0x6u, false, "plain, dim 160, snapshots of 1 and 2"},
        {192, 4, 4, 4, 0, false, "plain, dim 192"},
    };
    int fail = 0, n = 0;
    uint32_t seed = 11;
    for (const auto& c : cases) { ++n; if (!run_case(q, c, seed++)) ++fail; }
    std::printf("%s: %d of %d cases passed\n", fail ? "FAIL" : "PASS", n - fail, n);
    return fail ? 1 : 0;
}
