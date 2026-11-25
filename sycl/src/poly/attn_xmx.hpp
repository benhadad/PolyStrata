// sycl/src/poly/attn_xmx.hpp - Inkling's prompt-batch attention on the matrix units (joint_matrix, Xe2).
//
// The unsplit path of kernels_inkling.dp.cpp's attn_mma_kernel (CUDA: m16n8k16 fragments), written for joint_matrix:
// a work-group is NSG sub-groups of 16 lanes, a sub-group 8 query rows (row r of the block: token r / 3, head r % 3
// of the kv group), so a group takes MT = NSG * 8 / 3 tokens.  A tile is KT keys, counted from position 0:
//   S = Q K^T            8 x KT      (Q in log2 units, rounded to halves; K tile transposed in local memory)
//   bias = R slice       8 x BN      (R: the row's 16 relative values; slice: the table's rows of the distances the
//                                      tile spans, transposed), read back at each score's distance
//   online softmax in floats, two lanes a row; P rounded to halves
//   O += P V             8 x 128     (accumulators stay in registers; rescaled through their coordinates)
// A key outside the tokens' reach is loaded as zeros (its cache row may be anything) and masked.  The sink is added
// at the end as one more weight of no value.  Same arithmetic order per row whatever the other rows of the block.
#pragma once
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace attn_xmx {

namespace mx = sycl::ext::oneapi::experimental::matrix;

constexpr int DIM = 128, PER = 3;
constexpr int NSG = 6, SGSZ = 16, RSG = 8;          // sub-groups a group, lanes a sub-group, query rows a sub-group
constexpr int MT = NSG * RSG / PER;                 // tokens a group (16)
static_assert(NSG * RSG == PER * MT, "rows must be whole tokens");
constexpr int KT = 32;                              // keys a tile (64: slower; 128 does not build)
constexpr int BN = ((KT + 18 + 15) / 16) * 16;      // bias columns a tile: the distances a sub-group's rows span (+3), from a 16-aligned start (+15)
constexpr int PADL = 128, PADR = 128;               // zero columns before and after the table's rows (distances < 0 and >= extent)
// the transposed table's row length: [16][tt_ld(extent)] halves, column PADL + d holding table[d][j]
inline int tt_ld(int extent) { return ((PADL + extent + PADR + 15) / 16) * 16; }
constexpr int KTP = KT + 8, VP = DIM + 8, QP = DIM + 8, SSP = KT + 8, PSP = KT + 8, BSP = BN + 8, RP = 24, OSP = DIM + 4;
constexpr int OSZ = (RSG * SSP + RSG * BSP) > RSG * OSP ? (RSG * SSP + RSG * BSP) : RSG * OSP;   // the scores and the bias, then O

struct Args {
    const float* q;                       // [rows][n_head][DIM]
    const sycl::half* kc;                 // [position or ring slot][kv_heads][DIM], when not Q8_0
    const sycl::half* vc;
    const int8_t *kcodes, *vcodes;        // Q8_0: codes, and an fp16 scale (bits) for every 32 of them
    const uint16_t *kscales, *vscales;
    const float* rel_r;                   // [rows][n_head][16]
    const sycl::half* rel_tt;             // the table transposed and padded, halves: [16][tt_ld(extent)]
    int rel_extent;
    const float* sinks;                   // [rows][n_head] or null
    long long pos;                        // the first row's position
    int rows, n_head, kv_heads, window, ring;
    float scale;
    float* out;                           // [rows][n_head][DIM]
};

inline uint32_t half2_bits(float lo, float hi) {
    sycl::half2 h = sycl::float2(lo, hi).convert<sycl::half, sycl::rounding_mode::rte>();
    return *reinterpret_cast<const uint32_t*>(&h);
}

template <bool KQ8, bool VQ8>
inline void kernel(const Args& a, sycl::nd_item<1> it) {
    constexpr float LOG2E = 1.44269504f;
    sycl::sub_group sg = it.get_sub_group();
    sycl::group<1> grp = it.get_group();
    auto& kt = *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::half[DIM][KTP]>(grp);
    auto& vs = *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::half[KT][VP]>(grp);
    auto& qs_all = *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::half[NSG][RSG][QP]>(grp);
    auto& ss_all = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[NSG][OSZ]>(grp);
    auto& ps_all = *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::half[NSG][RSG][PSP]>(grp);
    auto& rs_all = *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::half[NSG][RSG][RP]>(grp);
    auto& fs_all = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[NSG][RSG]>(grp);

    const int w = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
    const int gid = (int) it.get_group(0);
    const int ntb = (a.rows + MT - 1) / MT;
    const int g = gid % a.kv_heads;
    const int t0 = (ntb - 1 - gid / a.kv_heads) * MT;   // the longest blocks first
    const int nt = sycl::min(MT, a.rows - t0);
    const long long p0 = a.pos + t0;
    const int tid = (int) it.get_local_linear_id();

    sycl::half* qs = &qs_all[w][0][0];
    float* ss = &ss_all[w][0];
    float* bs = ss + RSG * SSP;
    float* os = ss;
    sycl::half* ps = &ps_all[w][0][0];
    sycl::half* rs = &rs_all[w][0][0];
    float* fs = &fs_all[w][0];

    auto lp = [](auto* p) {
        return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
    };

    // this lane's row (two lanes a row, each half of its columns)
    const int r = lane >> 1, hf = lane & 1;
    const int R = w * RSG + r;                 // the row of the block
    const int tok = R / PER;
    const int head = g * PER + R % PER;
    const bool ok = tok < nt;
    const size_t th = (size_t(t0) + tok) * a.n_head + head;
    const int t_lo = (w * RSG) / PER;          // the sub-group's first token

    // Q (scaled, in log2 units) and R as halves
    {
        const float* qp = a.q + th * DIM + hf * 64;
        for (int c = 0; c < 64; c += 2) {
            float x0 = 0.f, x1 = 0.f;
            if (ok) {
                sycl::float2 x = *reinterpret_cast<const sycl::float2*>(qp + c);
                x0 = x.x() * a.scale * LOG2E;
                x1 = x.y() * a.scale * LOG2E;
            }
            *reinterpret_cast<uint32_t*>(&qs[r * QP + hf * 64 + c]) = half2_bits(x0, x1);
        }
        for (int c = 0; c < 8; c += 2) {
            float x0 = 0.f, x1 = 0.f;
            if (ok) {
                x0 = a.rel_r[th * 16 + hf * 8 + c] * LOG2E;
                x1 = a.rel_r[th * 16 + hf * 8 + c + 1] * LOG2E;
            }
            *reinterpret_cast<uint32_t*>(&rs[r * RP + hf * 8 + c]) = half2_bits(x0, x1);
        }
    }
    sycl::group_barrier(sg);
    mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, RSG, 16, mx::layout::row_major> qa[DIM / 16];
    for (int k = 0; k < DIM / 16; ++k) mx::joint_matrix_load(sg, qa[k], lp(qs + k * 16), QP);
    mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, RSG, 16, mx::layout::row_major> ra;
    mx::joint_matrix_load(sg, ra, lp(rs), RP);

    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, RSG, 16> o[DIM / 16];
    for (int n = 0; n < DIM / 16; ++n) mx::joint_matrix_fill(sg, o[n], 0.0f);
    float m = -INFINITY, l = 0.0f;
    // the row of each accumulator element this lane holds (the same for all of o[]: 8 rows, 16 lanes)
    int erow[RSG];
    {
        auto wi = sycl::ext::oneapi::detail::get_wi_data(sg, o[0]);
        for (int i = 0; i < RSG; ++i) {
            auto e = wi[i];
            erow[i] = (int) std::get<0>(e.get_coord());
        }
    }
    const int ttld = tt_ld(a.rel_extent);
    auto gp = [](auto* p) {
        return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
    };

    const long long hi = p0 + nt;
    const long long lo = a.window > 0 ? sycl::max(0LL, p0 - a.window + 1) : 0LL;
    const long long own = p0 + tok;

    for (long long k0 = lo / KT * KT; k0 < hi; k0 += KT) {
        sycl::group_barrier(grp);   // the last tile's keys and values are read
        // the tile's keys (transposed) and values, 8 values a thread at a time; positions outside [lo, hi) are zeros
        for (int i = tid; i < KT * DIM / 8; i += NSG * SGSZ) {
            const int kr = i / (DIM / 8), c = 8 * (i % (DIM / 8));
            const long long key = k0 + kr;
            sycl::uint4 kx(0, 0, 0, 0), vx(0, 0, 0, 0);
            if (key >= lo && key < hi) {
                const size_t at = (size_t(a.ring > 0 ? key % a.ring : key) * a.kv_heads + g) * DIM + c;
                auto load = [&](const sycl::half* f16, const int8_t* codes, const uint16_t* scales, bool quant) {
                    if (!quant) return *reinterpret_cast<const sycl::uint4*>(f16 + at);
                    const sycl::uint2 cd = *reinterpret_cast<const sycl::uint2*>(codes + at);
                    const float d = float(sycl::bit_cast<sycl::half>(scales[at / 32]));
                    uint32_t h[4];
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t word = j < 2 ? cd.x() : cd.y(), sh = 16 * (j % 2);
                        h[j] = half2_bits((float) (signed char) (word >> sh) * d,
                                          (float) (signed char) (word >> (sh + 8)) * d);
                    }
                    return sycl::uint4(h[0], h[1], h[2], h[3]);
                };
                kx = load(a.kc, a.kcodes, a.kscales, KQ8);
                vx = load(a.vc, a.vcodes, a.vscales, VQ8);
            }
            const sycl::half* kh = reinterpret_cast<const sycl::half*>(&kx);
            for (int j = 0; j < 8; ++j) kt[c + j][kr] = kh[j];
            *reinterpret_cast<sycl::uint4*>(&vs[kr][c]) = vx;
        }
        sycl::group_barrier(grp);

        // S = Q K^T
        for (int nb = 0; nb < KT / 16; ++nb) {
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, RSG, 16> acc;
            mx::joint_matrix_fill(sg, acc, 0.0f);
            for (int k = 0; k < DIM / 16; ++k) {
                mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major> b;
                mx::joint_matrix_load(sg, b, lp(&kt[k * 16][nb * 16]), KTP);
                mx::joint_matrix_mad(sg, acc, qa[k], b, acc);
            }
            mx::joint_matrix_store(sg, acc, lp(ss + nb * 16), SSP, mx::layout::row_major);
        }
        // the bias: the sub-group's rows times the table's rows of the distances d = dmin + e
        const long long dmin = p0 + t_lo - (k0 + KT - 1);
        const bool near = dmin < a.rel_extent;
        // columns e of the bias are distances dal + e, dal being dmin rounded down to a multiple of 16 so the
        // table's fragments are read aligned, straight from the padded transposed table in global memory
        const long long dal = (dmin >> 4) << 4;
        const int doff = (int) (dmin - dal);
        if (near) {
            const sycl::half* tt = a.rel_tt + PADL + dal;
            for (int nb = 0; nb < BN / 16; ++nb) {
                mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major> b;
                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, RSG, 16> acc;
                mx::joint_matrix_fill(sg, acc, 0.0f);
                mx::joint_matrix_load(sg, b, gp(const_cast<sycl::half*>(tt + nb * 16)), ttld);
                mx::joint_matrix_mad(sg, acc, ra, b, acc);
                mx::joint_matrix_store(sg, acc, lp(bs + nb * 16), BSP, mx::layout::row_major);
            }
        }
        sycl::group_barrier(sg);
        // online softmax: this lane's half of the row's columns
        constexpr int CH = KT / 2;
        float sv[CH];
        float mx_ = -INFINITY;
        const int e0 = doff + (tok - t_lo) + KT - 1;
        for (int c = 0; c < CH; ++c) {
            const int cc = hf * CH + c;
            const long long key = k0 + cc;
            float s = ss[r * SSP + cc];
            if (!ok || key > own || (a.window > 0 && key <= own - a.window)) s = -INFINITY;
            else if (near) s += bs[r * BSP + (e0 - cc)];
            sv[c] = s;
            mx_ = sycl::fmax(mx_, s);
        }
        mx_ = sycl::fmax(mx_, sycl::permute_group_by_xor(sg, mx_, 1));
        const float now = sycl::fmax(m, mx_), base = now == -INFINITY ? 0.0f : now;
        const float f = sycl::exp2(m - base);
        m = now;
        l *= f;
        for (int c = 0; c < CH; ++c) {
            const float p = sycl::exp2(sv[c] - base);
            l += p;
            ps[r * PSP + hf * CH + c] = sycl::half(p);
        }
        if (hf == 0) fs[r] = f;
        sycl::group_barrier(sg);
        // O *= f (by row), then O += P V
        // a row's weights change only when its max did: when no row of the sub-group has a new max (f is exactly 1)
        // the accumulators are left alone
        if (sycl::any_of_group(sg, f != 1.0f)) {
            float fr[RSG];
            for (int i = 0; i < RSG; ++i) fr[i] = fs[erow[i]];
            for (int n = 0; n < DIM / 16; ++n) {
                auto wi = sycl::ext::oneapi::detail::get_wi_data(sg, o[n]);
                for (int i = 0; i < RSG; ++i) {
                    auto e = wi[i];
                    const float x = e;
                    e = x * fr[i];
                }
            }
        }
        for (int k = 0; k < KT / 16; ++k) {
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, RSG, 16, mx::layout::row_major> pa;
            mx::joint_matrix_load(sg, pa, lp(ps + k * 16), PSP);
            for (int n = 0; n < DIM / 16; ++n) {
                mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major> b;
                mx::joint_matrix_load(sg, b, lp(&vs[k * 16][n * 16]), VP);
                mx::joint_matrix_mad(sg, o[n], pa, b, o[n]);
            }
        }
    }
    // the row's sum from both lanes, the sink, and out = O / l
    l += sycl::permute_group_by_xor(sg, l, 1);
    for (int n = 0; n < DIM / 16; ++n) mx::joint_matrix_store(sg, o[n], lp(os + n * 16), OSP, mx::layout::row_major);
    sycl::group_barrier(sg);
    if (ok) {
        float f = 1.0f;
        if (a.sinks) {
            const float sink = a.sinks[th] * LOG2E, now = sycl::fmax(m, sink);
            f = sycl::exp2(m - now);
            l = l * f + sycl::exp2(sink - now);
        }
        float* y = a.out + th * DIM + hf * 64;
        const float inv = f / l;
        for (int c = 0; c < 64; c += 4) {
            sycl::float4 v(os[r * OSP + hf * 64 + c], os[r * OSP + hf * 64 + c + 1], os[r * OSP + hf * 64 + c + 2],
                           os[r * OSP + hf * 64 + c + 3]);
            *reinterpret_cast<sycl::float4*>(y + c) = v * inv;
        }
    }
}

// Launch for `rows` tokens.  Takes n_rel == 16 only.
inline void launch(const Args& a, sycl::queue& q) {
    const int ntb = (a.rows + MT - 1) / MT;
    const size_t groups = (size_t) ntb * a.kv_heads;
    const bool kq = a.kcodes != nullptr, vq = a.vcodes != nullptr;
    auto run = [&](auto kq_c, auto vq_c) {
        q.parallel_for(sycl::nd_range<1>(groups * NSG * SGSZ, NSG * SGSZ),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                           kernel<decltype(kq_c)::value, decltype(vq_c)::value>(a, it);
                       });
    };
    using T = std::true_type;
    using F = std::false_type;
    if (kq && vq) run(T{}, T{});
    else if (kq) run(T{}, F{});
    else if (vq) run(F{}, T{});
    else run(F{}, F{});
}

}  // namespace attn_xmx
