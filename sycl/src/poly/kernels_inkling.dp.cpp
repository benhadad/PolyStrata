// src/poly/kernels_inkling.cu - Inkling 2 Mini's kernels; see include/strata/poly/inkling_kernels.hpp for the math
// and where it comes from.  Plain float arithmetic throughout, the caches' halves (or Q8_0 codes) the only rounding,
// but for the prompt batches' attention on the tensor cores (attn_mma_kernel), whose products take halves.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
// SYCL port: the PTX address conversion only feeds paths that are off here
#define __cvta_generic_to_shared(p) ((size_t)(p))
#include "strata/sycl_queue.hpp"
#include "strata/poly/inkling_kernels.hpp"
#include "strata/kernels/dp4a.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

// SYCL port (ink_fixups.py): launches dpct left as `<<<>>>`. The kernels take no dynamic shared memory.
#define INK_LAUNCH(NAME, GRID, BLOCK, ST, ...)                                                          \
    (ST)->parallel_for<class NAME>(                                                                     \
        sycl::nd_range<3>((GRID) * sycl::range<3>(1, 1, (BLOCK)), sycl::range<3>(1, 1, (BLOCK))),       \
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { __VA_ARGS__; })

namespace strata::poly {
namespace {

constexpr int WARP = 32;

inline dpct::queue_ptr cs(void *stream) { return strata::q_of(stream); }
inline int grid(size_t n, int block) { return int((n + block - 1) / block); }

void check_launch(const char* what) {
    /*
    DPCT1010:234: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    /*
    DPCT1000:233: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (const dpct::err0 e = 0; e != 0)
        /*
        DPCT1009:235: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001:232: The statement could not be removed.
        */
        throw std::runtime_error(std::string(what) + ": " +
                                 dpct::get_error_string_dummy(e));
}

__dpct_inline__ float warp_sum(float x) {
#pragma unroll
    /*
    DPCT1108:121: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = WARP / 2; o > 0; o >>= 1) x +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            o);
    return x;
}

__dpct_inline__ float warp_max(float x) {
#pragma unroll
    /*
    DPCT1108:122: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = WARP / 2; o > 0; o >>= 1) x = sycl::fmax(
        x, dpct::experimental::permute_sub_group_by_xor(
               0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
               x, o));
    return x;
}

__dpct_inline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}

// upstream: cast(pos) -> scale_bias(1, 1) -> scale(1 / floor) -> clamp(1, inf) -> log -> scale_bias(alpha, 1)
__dpct_inline__ float tau_of(float alpha, int floor_, long long p) {
    float v = (float) p * 1.0f + 1.0f;
    v = v * (1.0f / (float) floor_);
    v = v < 1.0f ? 1.0f : v;
    return sycl::log(v) * alpha + 1.0f;
}

// ---- clip ------------------------------------------------------------------------------------------------------

// A warp a head of a token.
/*
DPCT1110:123: The total declared local variable size in device function
clip_group exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void clip_group(const float *__restrict__ x, int x_stride,
                                const float *__restrict__ scale, float *y,
                                int dim, int heads, int gi, float threshold,
                                float alpha, int floor_, long long pos) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) %
        WARP;
    const int t = gi / heads, h = gi % heads;
    const float* xr = x + size_t(t) * x_stride + size_t(h) * dim;
    float* yr = y + (size_t(t) * heads + h) * dim;
    const float* sr = scale ? scale + size_t(h) * dim : nullptr;
    constexpr int MAXV = 1024 / WARP;
    float v[MAXV];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXV; ++k) {
        const int i = lane + k * WARP;
        v[k] = 0.0f;
        if (i < dim) {
            v[k] = sr ? xr[i] * sr[i] : xr[i];
            ss += v[k] * v[k];
        }
    }
    ss = warp_sum(ss);
    const float norm = sycl::sqrt(ss) + 1e-6f;
    const float factor = sycl::fmin(sycl::fmax(threshold / norm, 0.0f), 1.0f);
    const float tau = floor_ > 0 ? tau_of(alpha, floor_, pos + t) : 1.0f;
#pragma unroll
    for (int k = 0; k < MAXV; ++k) {
        const int i = lane + k * WARP;
        if (i < dim) yr[i] = floor_ > 0 ? v[k] * factor * tau : v[k] * factor;
    }
}

__dpct_inline__ void clip_kernel(const float *__restrict__ x, int x_stride,
                                 const float *__restrict__ scale, float *y,
                                 int dim, int heads, int groups,
                                 float threshold, float alpha, int floor_,
                                 long long pos) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int gi =
        item_ct1.get_group(2) * (item_ct1.get_local_range(2) / WARP) +
        item_ct1.get_local_id(2) / WARP;
    if (gi >= groups) return;
    clip_group(x, x_stride, scale, y, dim, heads, gi, threshold, alpha, floor_, pos);
}

struct ClipJobs {
    InklingClipJob j[4];
    int first[5];   // the jobs' first groups, and the end
};

// Where a call's tokens are: one sequence's, from a position on (inkling_clip_many()), or each row a sequence's of
// its own at a position of its own (inkling_clip_many_seqs()).  pos(t) - t is handed to clip_group(), which adds t.
struct ClipFrom {
    long long pos;
    long long first(int) const { return pos; }
};
struct ClipSeqs {
    long long pos[kInklingSeqRows];
    long long first(int t) const { return pos[t] - t; }
};

template <class From>
__dpct_inline__ void clip_many_kernel(ClipJobs c, int count, From from) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int gi =
        item_ct1.get_group(2) * (item_ct1.get_local_range(2) / WARP) +
        item_ct1.get_local_id(2) / WARP;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (k >= count) return;
        if (gi < c.first[k + 1]) {
            const InklingClipJob& j = c.j[k];
            const int g = gi - c.first[k];
            clip_group(j.x, j.x_stride ? j.x_stride : j.heads * j.dim, j.scale, j.y, j.dim, j.heads, g, j.threshold,
                       j.tau.alpha, j.tau.floor, from.first(g / j.heads));
            return;
        }
    }
}

// ---- attention -------------------------------------------------------------------------------------------------

constexpr int AT_DIM = 128, AT_PER = 3, AT_KEYS = 32, AT_THREADS = 128, AT_MAX_REL = 32;
constexpr int AT_KPAD = AT_DIM + 1;   // a key row in shared memory, padded so that 32 keys are 32 banks
constexpr int AT_MIN_SPLIT = 32;      // the split path's smallest split: short contexts still fill the card

struct AttnArgs {
    const float* q;
    const sycl::half *kc;
    const sycl::half *vc;
    Kvq8 kq;
    Kvq8 vq;
    InklingRel rel;
    const float* sinks;
    long long pos;
    int rows, n_head, kv_heads, window, ring;
    float scale;
    int split;   // the split path's keys a split (counted from position 0)
    int nslot;   // the split path's splits a token's partials have room for
    int i8_from;  // the split path's rows of splits of this many keys or more take attn_dec_i8_kernel (0: none)
    float* out;
    float* work;
};

template <bool Q8>
__dpct_inline__ float cache_value(const sycl::half *c, const Kvq8 &q8,
                                  size_t i) {
    if constexpr (Q8) return (float)q8.codes[i] *
        sycl::vec<sycl::half, 1>(
            sycl::bit_cast<sycl::half, unsigned short>(q8.scales[i / 32]))
            .convert<float, sycl::rounding_mode::automatic>()[0];
    else return sycl::vec<sycl::half, 1>(c[i])
        .convert<float, sycl::rounding_mode::automatic>()[0];
}

constexpr size_t attn_smem(int tq) {
    const int r = tq * AT_PER;
    return sizeof(float) * (size_t(r) * AT_DIM + AT_KEYS * AT_KPAD + AT_KEYS * AT_DIM + size_t(r) * AT_KEYS +
                            size_t(r) * AT_MAX_REL + 3 * r);
}

// The unsplit path: a block takes TQ tokens from blockIdx.z * TQ, the PER heads of group blockIdx.y, and all of
// the tokens' keys.  The keys go in tiles of AT_KEYS counted from position 0; a row's online softmax only takes
// the keys the row attends, so a key the row does not see changes nothing (its weight is exactly 0 and the row's
// max stays).
template <int TQ, bool KQ8, bool VQ8>
/*
DPCT1110:124: The total declared local variable size in device function
attn_kernel exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void attn_kernel(AttnArgs a, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int R = TQ * AT_PER;
    constexpr int RPT = (R * AT_KEYS + AT_THREADS - 1) / AT_THREADS;   // scores a thread per tile
    auto smem = (float *)dpct_local;
    float* Qs = smem;                        // [R][DIM]
    float* Ks = Qs + R * AT_DIM;             // [KEYS][KPAD]
    float* Vs = Ks + AT_KEYS * AT_KPAD;      // [KEYS][DIM]
    float* Ps = Vs + AT_KEYS * AT_DIM;       // [R][KEYS]
    float* Rs = Ps + R * AT_KEYS;            // [R][MAX_REL]
    float* Ms = Rs + R * AT_MAX_REL;         // [R]
    float* Ls = Ms + R;                      // [R]
    float* As = Ls + R;                      // [R]

    const int tid = item_ct1.get_local_id(2), lane = tid % WARP,
              warp = tid / WARP;
    const int g = item_ct1.get_group(1), t0 = item_ct1.get_group(0) * TQ;
    const int nt = sycl::min(TQ, a.rows - t0);
    const int n_rel = a.rel.n_rel;
    const long long p0 = a.pos + t0, plast = p0 + nt - 1;
    const long long kb = a.window > 0 && p0 >= a.window ? p0 - a.window + 1 : 0, ke = plast + 1;

    // the queries, their relative-position projections; the rows' running max, sum and output
    for (int i = tid; i < R * AT_DIM; i += AT_THREADS) {
        const int r = i / AT_DIM, d = i % AT_DIM, t = r / AT_PER, h = g * AT_PER + r % AT_PER;
        Qs[i] = t < nt ? a.q[((size_t(t0) + t) * a.n_head + h) * AT_DIM + d] : 0.0f;
    }
    for (int i = tid; i < R * AT_MAX_REL; i += AT_THREADS) {
        const int r = i / AT_MAX_REL, j = i % AT_MAX_REL, t = r / AT_PER, h = g * AT_PER + r % AT_PER;
        Rs[i] = t < nt && j < n_rel ? a.rel.r[((size_t(t0) + t) * a.n_head + h) * n_rel + j] : 0.0f;
    }
    if (tid < R) {
        Ms[tid] = -INFINITY;
        Ls[tid] = 0.0f;
    }
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    /*
    DPCT1065:236: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (long long kt = kb < ke ? kb / AT_KEYS * AT_KEYS : ke; kt < ke; kt += AT_KEYS) {
        // the tile's keys and values; a key outside [kb, ke) is zeros (its cache row may be anything)
        for (int j = 0; j < AT_KEYS; ++j) {
            const long long kp = kt + j;
            const bool live = kp >= kb && kp < ke;
            const long long row = a.ring > 0 ? kp % a.ring : kp;
            const size_t base = (size_t(row) * a.kv_heads + g) * AT_DIM + tid;
            Ks[j * AT_KPAD + tid] = live ? cache_value<KQ8>(a.kc, a.kq, base) : 0.0f;
            Vs[j * AT_DIM + tid] = live ? cache_value<VQ8>(a.vc, a.vq, base) : 0.0f;
        }
        /*
        DPCT1118:125: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:237: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // scores: a thread has key `lane` of the tile and the rows warp, warp + 4, ...
        {
            float s[RPT];
#pragma unroll
            for (int i = 0; i < RPT; ++i) s[i] = 0.0f;
            const float* kr = Ks + lane * AT_KPAD;
#pragma unroll 4
            for (int d = 0; d < AT_DIM; ++d) {
                const float kv = kr[d];
#pragma unroll
                for (int i = 0; i < RPT; ++i) {
                    const int r = warp + i * (AT_THREADS / WARP);
                    if (r < R) s[i] += Qs[r * AT_DIM + d] * kv;
                }
            }
            const long long kp = kt + lane;
#pragma unroll
            for (int i = 0; i < RPT; ++i) {
                const int r = warp + i * (AT_THREADS / WARP);
                if (r >= R) continue;
                const int t = r / AT_PER;
                const long long p = p0 + t;
                const long long dist = p - kp;
                const bool seen = t < nt && kp >= kb && kp < ke && dist >= 0 && (a.window <= 0 || dist < a.window);
                float v = -INFINITY;
                if (seen) {
                    float bias = 0.0f;
                    if (dist < a.rel.extent) {
                        const float* tb = a.rel.table + size_t(dist) * n_rel;
                        const float* rr = Rs + r * AT_MAX_REL;
#pragma unroll
                        for (int j = 0; j < n_rel; ++j) bias += tb[j] * rr[j];
                    }
                    v = s[i] * a.scale + bias;
                }
                Ps[r * AT_KEYS + lane] = v;
            }
        }
        /*
        DPCT1118:126: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:238: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // the online softmax: a warp a row
        for (int r = warp; r < R; r += AT_THREADS / WARP) {
            const float s = Ps[r * AT_KEYS + lane];
            const float m_old = Ms[r];
            const float m_new = sycl::fmax(m_old, warp_max(s));
            const float p =
                s == -INFINITY ? 0.0f : sycl::native::exp(s - m_new);
            const float sum = warp_sum(p);
            Ps[r * AT_KEYS + lane] = p;
            if (lane == 0) {
                const float alpha = m_old == -INFINITY
                                        ? 0.0f
                                        : sycl::native::exp(m_old - m_new);
                As[r] = alpha;
                Ms[r] = m_new;
                Ls[r] = Ls[r] * alpha + sum;
            }
        }
        /*
        DPCT1118:127: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:239: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // the values: a thread a dimension, every row
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] *= As[r];
        for (int j = 0; j < AT_KEYS; ++j) {
            const float v = Vs[j * AT_DIM + tid];
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] += Ps[r * AT_KEYS + j] * v;
        }
        /*
        DPCT1118:128: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:240: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int t = r / AT_PER, h = g * AT_PER + r % AT_PER;
        if (t >= nt) continue;
        const size_t tok = size_t(t0) + t;
        const float m = Ms[r], l = Ls[r];
        const float sink = a.sinks ? a.sinks[tok * a.n_head + h] : -INFINITY;
        const float mx = sycl::fmax(m, sink);
        const float f = m == -INFINITY ? 0.0f : sycl::native::exp(m - mx);
        const float denom =
            l * f + (sink == -INFINITY ? 0.0f : sycl::native::exp(sink - mx));
        a.out[(tok * a.n_head + h) * AT_DIM + tid] = acc[r] * f / denom;
    }
}

// The split path's rows: a call's tokens one after the other in one sequence (inkling_attend()), or each row a
// sequence's of its own with its own caches, position and splits (inkling_attend_seqs()).  AttnArgs::nslot is then
// the most splits of a row, and what a token's partials are strided by.
struct AttnRow {
    const sycl::half *kc;
    const sycl::half *vc;
    Kvq8 kq, vq;
    long long pos;
    int split, nslot;
};
struct AttnOne {
    AttnRow row(const AttnArgs& a, int t) const {
        return {a.kc, a.vc, a.kq, a.vq, a.pos + t, a.split, a.nslot};
    }
};
struct AttnSeqs {
    AttnRow r[kInklingSeqRows];
    AttnRow row(const AttnArgs&, int t) const { return r[t]; }
};
// the rows of a call whose splits are of `from` keys or more
inline int rows_from(const AttnOne&, const AttnArgs& a, int ntok, int from) { return a.split >= from ? ntok : 0; }
inline int rows_from(const AttnSeqs& s, const AttnArgs&, int ntok, int from) {
    int n = 0;
    for (int t = 0; t < ntok; ++t) n += s.r[t].split >= from;
    return n;
}

// The splits of a token's head merged, in the splits' order, with its sink.  A block a head of a token, a thread a
// value dimension.  A row of one split has its output from attn_dec_kernel.
template <class Rows>
__dpct_inline__ void attn_join_kernel(AttnArgs a, Rows rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int th = item_ct1.get_group(2), d = item_ct1.get_local_id(2);
    const int nslot = rows.row(a, th / a.n_head).nslot;
    if (nslot <= 1) return;
    const float* sinks = a.sinks;
    const float* w = a.work + size_t(th) * a.nslot * (AT_DIM + 2);
    float mx = sinks ? sinks[th] : -INFINITY;
#pragma unroll
    for (int s = 0; s < nslot; ++s)
        mx = sycl::fmax(mx, w[size_t(s) * (AT_DIM + 2)]);
    float denom = sinks ? sycl::native::exp(sinks[th] - mx) : 0.0f, sum = 0.0f;
    for (int s = 0; s < nslot; ++s) {
        const float* ws = w + size_t(s) * (AT_DIM + 2);
        if (ws[0] == -INFINITY) continue;
        const float f = sycl::native::exp(ws[0] - mx);
        denom += ws[1] * f;
        sum += ws[2 + d] * f;
    }
    a.out[size_t(th) * AT_DIM + d] = sum / denom;
}

// The split path's kernel: one token of one group, the keys of one split.  No tiles in shared memory and no
// barriers while it walks: each warp takes DEC_KEYS keys at a time (the block's keys go to the warps in turn,
// counted from the split's first key).  A key's score is the sum of 32 shares, a share four of its values' dot with
// a head's query (values 4j .. 4j + 3) and bias weight j, added up pairwise as a butterfly over 32 lanes would (j
// with j + 16, then 8 apart, 4, 2, 1).  A lane takes the four shares of one key that sum first (j, j + 8, j + 16,
// j + 24), so the four keys' scores are made at once by 8 lanes each with three shuffles a head.  Its values: a
// lane four of each key's, as the output is kept.  Each warp keeps its own running max, sum and output for the
// group's heads; the warps are merged in their order at the end, and the block's result is the token's output when
// its call has one split, else a partial for attn_join_kernel.  Nothing depends on the other tokens of the call.
constexpr int DEC_WARPS = 4, DEC_KEYS = 4, DEC_LANES = WARP / DEC_KEYS;   // DEC_LANES: a key's lanes for its score
constexpr int DEC_SHARES = WARP / DEC_LANES;                              // a lane's shares of a score
static_assert(DEC_WARPS * WARP == AT_DIM, "attn_dec_kernel's merge takes a thread a value dimension");
static_assert(AT_MAX_REL <= WARP, "a share has one bias weight at most");

// Four values of a cache from value i: Q8_0 codes times their block's scale, or halves.
template <bool Q8>
__dpct_inline__ void load4(const sycl::half *c, const Kvq8 &q8, size_t i,
                           float *x) {
    if constexpr (Q8) {
        const int codes = *reinterpret_cast<const int*>(q8.codes + i);
        const float d =
            sycl::vec<sycl::half, 1>(
                sycl::bit_cast<sycl::half, unsigned short>(q8.scales[i / 32]))
                .convert<float, sycl::rounding_mode::automatic>()[0];
#pragma unroll
        for (int j = 0; j < 4; ++j) x[j] = (float) (signed char) (codes >> (8 * j)) * d;
    } else {
        const sycl::uint2 bits = *reinterpret_cast<const sycl::uint2 *>(c + i);
        const sycl::half2 a = *reinterpret_cast<const sycl::half2 *>(&bits.x()),
                          b = *reinterpret_cast<const sycl::half2 *>(&bits.y());
        x[0] = a[0], x[1] = a[1], x[2] = b[0], x[3] = b[1];
    }
}

template <bool KQ8, bool VQ8, class Rows>
/*
DPCT1110:129: The total declared local variable size in device function
attn_dec_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void attn_dec_kernel(AttnArgs a, Rows rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sm_m = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    float[DEC_WARPS][AT_PER]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sm_l = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[DEC_WARPS][AT_PER]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sm_o = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[DEC_WARPS][AT_PER][AT_DIM]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sm_q = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[AT_PER][AT_DIM]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int lane = item_ct1.get_local_id(2) % WARP,
              warp = item_ct1.get_local_id(2) / WARP;
    const int sub = lane % DEC_LANES, mine = lane / DEC_LANES;   // the score's lanes: key `mine` of a step
    const int g = item_ct1.get_group(1), t = item_ct1.get_group(0),
              n_rel = a.rel.n_rel;
    const AttnRow r = rows.row(a, t);
    if ((int)item_ct1.get_group(2) >=
        r.nslot) return; // a row with fewer splits than the call's most
    if (a.i8_from > 0 && r.split >= a.i8_from) return;   // attn_dec_i8_kernel's row
    const int p = int(r.pos);
    const int lo = a.window > 0 && p >= a.window ? p - a.window + 1 : 0;
    const int s = lo / r.split + item_ct1.get_group(2);
    const int kb = sycl::max(lo, s * r.split),
              ke = sycl::min(p + 1, (s + 1) * r.split);
    // the queries in shared memory; share j = sub + DEC_LANES * k of each head's bias weight
    float rr[AT_PER][DEC_SHARES], m[AT_PER], l[AT_PER], acc[AT_PER][4];
#pragma unroll
    for (int h = 0; h < AT_PER; ++h) {
        const size_t th = size_t(t) * a.n_head + g * AT_PER + h;
        sm_q[h][item_ct1.get_local_id(2)] =
            a.q[th * AT_DIM + item_ct1.get_local_id(2)];
#pragma unroll
        for (int k = 0; k < DEC_SHARES; ++k) {
            const int j = sub + DEC_LANES * k;
            rr[h][k] = j < n_rel ? a.rel.r[th * n_rel + j] : 0.0f;
        }
        m[h] = -INFINITY, l[h] = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[h][i] = 0.0f;
    }
    // A step of the warp's DEC_KEYS keys from k0, at cache rows from row0.  A full step (its keys all live and not
    // across a ring's end: all but a split's last) goes without those checks, and a far one (its keys all past the
    // bias's extent: most of a long context) loads no bias weights (theirs are 0, as the general step finds).
    const size_t key_stride = size_t(a.kv_heads) * AT_DIM;
    /*
    DPCT1110:130: The total declared local variable size in device function
    operator() exceeds 128 bytes and may cause high register pressure. Consult
    with your hardware vendor to find the total register size available and
    adjust the code, or use smaller sub-group size to avoid high register
    pressure.
    */
    auto step = [&](int k0, int row0, auto full, auto far) {
        constexpr bool FULL = decltype(full)::value, FAR = decltype(far)::value;
        size_t at[DEC_KEYS];   // each key's row
        bool live[DEC_KEYS];
#pragma unroll
        for (int c = 0; c < DEC_KEYS; ++c) {
            if constexpr (FULL) {
                at[c] = (size_t(row0) * a.kv_heads + g) * AT_DIM + c * key_stride;
                live[c] = true;
            } else {
                int row = row0 + c;
                if (a.ring > 0 && row >= a.ring) row -= a.ring;
                at[c] = (size_t(row) * a.kv_heads + g) * AT_DIM;
                live[c] = k0 + c < ke;
            }
        }
        // the values: a lane four of each key
        float vx[DEC_KEYS][4];
#pragma unroll
        for (int c = 0; c < DEC_KEYS; ++c) {
            if (live[c]) {
                load4<VQ8>(r.vc, r.vq, at[c] + 4 * lane, vx[c]);
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) vx[c][i] = 0.0f;
            }
        }
        // the score of key `mine`: this lane's four shares of it, as the butterfly's first two rounds add them
        float sc[AT_PER];
        {
            const int kp = k0 + mine, dist = p - kp;
            const bool here = FULL || kp < ke;
            const bool near = !FAR && here && dist < a.rel.extent;
            size_t base = at[0];
#pragma unroll
            for (int c = 1; c < DEC_KEYS; ++c)
                if (mine == c) base = at[c];
            float sh[AT_PER][DEC_SHARES];
#pragma unroll
            for (int k = 0; k < DEC_SHARES; ++k) {
                const int j = sub + DEC_LANES * k;
                float kx[4];
                if (here) {
                    load4<KQ8>(r.kc, r.kq, base + 4 * j, kx);
                } else {
#pragma unroll
                    for (int i = 0; i < 4; ++i) kx[i] = 0.0f;
                }
                const float tb = near && j < n_rel ? a.rel.table[size_t(dist) * n_rel + j] : 0.0f;
#pragma unroll
                for (int h = 0; h < AT_PER; ++h) {
                    const sycl::float4 qv =
                        *reinterpret_cast<const sycl::float4 *>(
                            &sm_q[h][4 * j]);
                    /*
                    DPCT1013:241: The rounding mode could not be specified and
                    the generated code may have different accuracy than the
                    original code. Verify the correctness. SYCL math built-in
                    function rounding mode is aligned with OpenCL C 1.2
                    standard.
                    */
                    float d = qv.y() * kx[1]; // (the first two products: the
                                              // old kernel's contraction)
                    /*
                    DPCT1013:242: The rounding mode could not be specified and
                    the generated code may have different accuracy than the
                    original code. Verify the correctness. SYCL math built-in
                    function rounding mode is aligned with OpenCL C 1.2
                    standard.
                    */
                    d = sycl::fma(qv.x(), kx[0], d);
                    /*
                    DPCT1013:243: The rounding mode could not be specified and
                    the generated code may have different accuracy than the
                    original code. Verify the correctness. SYCL math built-in
                    function rounding mode is aligned with OpenCL C 1.2
                    standard.
                    */
                    d = sycl::fma(qv.z(), kx[2], d);
                    /*
                    DPCT1013:244: The rounding mode could not be specified and
                    the generated code may have different accuracy than the
                    original code. Verify the correctness. SYCL math built-in
                    function rounding mode is aligned with OpenCL C 1.2
                    standard.
                    */
                    d = sycl::fma(qv.w(), kx[3], d);
                    /*
                    DPCT1013:245: The rounding mode could not be specified and
                    the generated code may have different accuracy than the
                    original code. Verify the correctness. SYCL math built-in
                    function rounding mode is aligned with OpenCL C 1.2
                    standard.
                    */
                    sh[h][k] = sycl::fma(d, a.scale, tb * rr[h][k]);
                }
            }
#pragma unroll
            for (int h = 0; h < AT_PER; ++h) {
                // shares j and j + 16, j + 8 and j + 24, then those two
                /*
                DPCT1013:246: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                float v = sh[h][0] + sh[h][2] + sh[h][1] + sh[h][3];
#pragma unroll
                /*
                DPCT1108:131: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                /*
                DPCT1013:247: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                for (int o = DEC_LANES / 2; o > 0; o >>= 1) v =
                    v + dpct::experimental::permute_sub_group_by_xor(
                            0xffffffffu,
                            sycl::ext::oneapi::this_work_item::get_sub_group(),
                            v, o);
                sc[h] = v;
            }
        }
        float s[AT_PER][DEC_KEYS];
#pragma unroll
        for (int h = 0; h < AT_PER; ++h)
#pragma unroll
            /*
            DPCT1108:132: '__shfl_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int c = 0; c < DEC_KEYS; ++c)
                s[h][c] = dpct::experimental::select_from_sub_group(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), sc[h],
                    c * DEC_LANES);
#pragma unroll
        for (int h = 0; h < AT_PER; ++h) {
            float mx = -INFINITY;
#pragma unroll
            for (int c = 0; c < DEC_KEYS; ++c)
                if (live[c]) mx = sycl::fmax(mx, s[h][c]);
            const float m_new = sycl::fmax(m[h], mx);
            if (m_new == -INFINITY) continue;   // no live key yet
            const float alpha = m[h] == -INFINITY ? 0.0f : expf(m[h] - m_new);
            float pc[DEC_KEYS], sum = 0.0f;
#pragma unroll
            for (int c = 0; c < DEC_KEYS; ++c) {
                pc[c] = live[c] ? expf(s[h][c] - m_new) : 0.0f;
                /*
                DPCT1013:248: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                sum = sum + pc[c];
            }
            m[h] = m_new;
            /*
            DPCT1013:249: The rounding mode could not be specified and the
            generated code may have different accuracy than the original code.
            Verify the correctness. SYCL math built-in function rounding mode is
            aligned with OpenCL C 1.2 standard.
            */
            l[h] = sycl::fma(l[h], alpha, sum);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                /*
                DPCT1013:250: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                float y = sycl::fma(acc[h][i], alpha, pc[0] * vx[0][i]);
#pragma unroll
                /*
                DPCT1013:251: The rounding mode could not be specified and the
                generated code may have different accuracy than the original
                code. Verify the correctness. SYCL math built-in function
                rounding mode is aligned with OpenCL C 1.2 standard.
                */
                for (int c = 1; c < DEC_KEYS; ++c)
                    y = sycl::fma(pc[c], vx[c][i], y);
                acc[h][i] = y;
            }
        }
    };
    /*
    DPCT1065:252: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(); // the queries
    // a ring's wraps are found by comparing, not dividing, as the keys go on
    int row0 = kb + warp * DEC_KEYS;
    if (a.ring > 0) row0 %= a.ring;
    for (int k0 = kb + warp * DEC_KEYS; k0 < ke; k0 += DEC_WARPS * DEC_KEYS) {
        const bool full = k0 + DEC_KEYS <= ke && (a.ring <= 0 || row0 + DEC_KEYS <= a.ring);
        if (full && p - (k0 + DEC_KEYS - 1) >= a.rel.extent) step(k0, row0, std::true_type{}, std::true_type{});
        else if (full) step(k0, row0, std::true_type{}, std::false_type{});
        else step(k0, row0, std::false_type{}, std::false_type{});
        row0 += DEC_WARPS * DEC_KEYS;
        if (a.ring > 0 && row0 >= a.ring) row0 %= a.ring;
    }
#pragma unroll
    for (int h = 0; h < AT_PER; ++h) {
        if (lane == 0) sm_m[warp][h] = m[h], sm_l[warp][h] = l[h];
#pragma unroll
        for (int i = 0; i < 4; ++i) sm_o[warp][h][4 * lane + i] = acc[h][i];
    }
    /*
    DPCT1065:253: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int d = item_ct1.get_local_id(2); // a thread a value dimension
#pragma unroll
    for (int h = 0; h < AT_PER; ++h) {
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < DEC_WARPS; ++w) M = sycl::fmax(M, sm_m[w][h]);
        float L = 0.0f, O = 0.0f;
#pragma unroll
        for (int w = 0; w < DEC_WARPS; ++w) {
            const float f = sm_m[w][h] == -INFINITY
                                ? 0.0f
                                : sycl::native::exp(sm_m[w][h] - M);
            L += sm_l[w][h] * f;
            O += sm_o[w][h][d] * f;
        }
        const size_t th = size_t(t) * a.n_head + g * AT_PER + h;
        if (r.nslot > 1) {
            float *wp =
                a.work + (th * a.nslot + item_ct1.get_group(2)) * (AT_DIM + 2);
            if (d == 0) wp[0] = M, wp[1] = L;
            wp[2 + d] = O;
        } else {
            const float sink = a.sinks ? a.sinks[th] : -INFINITY;
            const float mx = sycl::fmax(M, sink);
            const float f = M == -INFINITY ? 0.0f : sycl::native::exp(M - mx);
            const float denom =
                L * f +
                (sink == -INFINITY ? 0.0f : sycl::native::exp(sink - mx));
            a.out[th * AT_DIM + d] = O * f / denom;
        }
    }
}

// The split path over Q8_0 keys and values with the queries quantized too (inkling_attend()'s int8_dot) for a
// full layer's rows of splits of I8_MIN_SPLIT keys or more: the same splits, partials and join as attn_dec_kernel,
// other arithmetic.  A window layer's splits are 32 keys (its window's), so it never comes here: no window, no
// ring.  A key's 128 values go to 8 lanes, 16 each (one 16-byte load of its codes), so a warp reads 4 keys at once;
// each 8-lane group takes I8_KEYS keys a step (the block's keys go to the warps in turn, counted from the split's
// first key, and a warp's to its groups in turn) and keeps its own running max, sum and output for the group's
// heads.  A score is the sum over the key's 32-value blocks of (its query's codes . the key's codes) * the query
// block's delta * the key block's, the dot of 16 values four __dp4a; the query is quantized here about as
// llama.cpp's flash-attention kernels quantize theirs for a quantized K (delta = the block's largest |value| / 127,
// codes the values times 1 / delta rounded to the nearest, where llama.cpp divides).  The groups of a warp are
// merged in their order at the end (lanes 8 and 16 apart), then the warps in theirs through shared memory.
constexpr int I8_WARPS = 4, I8_LANES = 8, I8_KEYS = 2;
// the shortest split it takes (a full layer's room of 32K): attn_dec_kernel was the faster on an RTX 2070 Super for
// window layers (a token at 32767: 19.3 us against 26.3 here) and short rooms (at 4096: 52.3 against 55.0), this
// one from 32K (at 32767: 254 us against 286)
constexpr int I8_MIN_SPLIT = 512;
constexpr int I8_DIMS = AT_DIM / I8_LANES;                        // a lane's values of a key: half a block
constexpr int I8_WARP_KEYS = WARP / I8_LANES * I8_KEYS;           // a warp's keys a step
constexpr int I8_STEP = I8_WARPS * I8_WARP_KEYS;                  // the block's
constexpr int I8_RELS = (AT_MAX_REL + I8_LANES - 1) / I8_LANES;   // a lane's relative-position weights a head
static_assert(I8_WARPS * WARP == AT_DIM, "attn_dec_i8_kernel's merge takes a thread a value dimension");
static_assert(I8_DIMS * 2 == 32, "a lane's values are half of a Q8_0 block");

// Four codes, unscaled, as floats: a code's byte offset by 128 under the bits of 2^23, less 2^23 + 128 (exact).
__dpct_inline__ void codes4(int w, float *x) {
    const unsigned u = unsigned(w) ^ 0x80808080u;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        x[j] = sycl::bit_cast<float>(
                   dpct::byte_level_permute(u, 0x4b000000u, 0x7540 + j)) -
               8388736.0f;
}

template <class Rows>
/*
DPCT1110:133: The total declared local variable size in device function
attn_dec_i8_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void attn_dec_i8_kernel(AttnArgs a, Rows rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sm_m = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    float[I8_WARPS][AT_PER]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sm_l = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[I8_WARPS][AT_PER]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sm_o = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[I8_WARPS][AT_PER][AT_DIM]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int lane = item_ct1.get_local_id(2) % WARP,
              warp = item_ct1.get_local_id(2) / WARP, sub = lane % I8_LANES;
    const int g = item_ct1.get_group(1), t = item_ct1.get_group(0),
              n_rel = a.rel.n_rel;
    const AttnRow r = rows.row(a, t);
    if ((int)item_ct1.get_group(2) >= r.nslot ||
        r.split < a.i8_from) return; // fewer splits, or attn_dec_kernel's row
    const int p = int(r.pos);
    const int kb = item_ct1.get_group(2) * r.split,
              ke = sycl::min(p + 1, kb + r.split);
    // this lane's 16 query values of each head as codes, their block's delta times the score's scale
    int qc[AT_PER][I8_DIMS / 4];
    float qd[AT_PER], rr[AT_PER][I8_RELS], m[AT_PER], l[AT_PER], acc[AT_PER][I8_DIMS];
#pragma unroll
    for (int h = 0; h < AT_PER; ++h) {
        const size_t th = size_t(t) * a.n_head + g * AT_PER + h;
        float x[I8_DIMS], amax = 0.0f;
#pragma unroll
        for (int i = 0; i < I8_DIMS / 4; ++i) {
            const sycl::float4 v = *reinterpret_cast<const sycl::float4 *>(
                a.q + th * AT_DIM + I8_DIMS * sub + 4 * i);
            x[4 * i] = v.x(), x[4 * i + 1] = v.y(), x[4 * i + 2] = v.z(),
                  x[4 * i + 3] = v.w();
        }
#pragma unroll
        for (int i = 0; i < I8_DIMS; ++i)
            amax = sycl::fmax(amax, sycl::fabs(x[i]));
        /*
        DPCT1108:134: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        amax = sycl::fmax(
            amax,
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                amax, 1)); // the block's other half
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
#pragma unroll
        for (int i = 0; i < I8_DIMS / 4; ++i) {
            int w = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) w |=
                (int(sycl::round(x[4 * i + j] * id)) & 0xff) << (8 * j);
            qc[h][i] = w;
        }
        qd[h] = d * a.scale;
#pragma unroll
        for (int i = 0; i < I8_RELS; ++i) {
            const int j = sub + I8_LANES * i;
            rr[h][i] = j < n_rel ? a.rel.r[th * n_rel + j] : 0.0f;
        }
        m[h] = -INFINITY, l[h] = 0.0f;
#pragma unroll
        for (int i = 0; i < I8_DIMS; ++i) acc[h][i] = 0.0f;
    }
    int k0 = kb + warp * I8_WARP_KEYS + lane / I8_LANES * I8_KEYS;   // this lane's first key
    for (int w0 = kb + warp * I8_WARP_KEYS; w0 < ke; w0 += I8_STEP) {
        sycl::int4 kc[I8_KEYS], vc[I8_KEYS];
        float kd[I8_KEYS], vd[I8_KEYS];
        bool live[I8_KEYS];
#pragma unroll
        for (int c = 0; c < I8_KEYS; ++c) {
            live[c] = k0 + c < ke;
            const size_t base = (size_t(k0 + c) * a.kv_heads + g) * AT_DIM + I8_DIMS * sub;
            if (live[c]) {
                kc[c] =
                    *reinterpret_cast<const sycl::int4 *>(r.kq.codes + base);
                vc[c] =
                    *reinterpret_cast<const sycl::int4 *>(r.vq.codes + base);
                kd[c] =
                    sycl::vec<sycl::half, 1>(
                        sycl::bit_cast<sycl::half, unsigned short>(
                            r.kq.scales[base / 32]))
                        .convert<float, sycl::rounding_mode::automatic>()[0];
                vd[c] =
                    sycl::vec<sycl::half, 1>(
                        sycl::bit_cast<sycl::half, unsigned short>(
                            r.vq.scales[base / 32]))
                        .convert<float, sycl::rounding_mode::automatic>()[0];
            } else {
                kc[c] = vc[c] = sycl::int4(0, 0, 0, 0);
                kd[c] = vd[c] = 0.0f;
            }
        }
        // a lane's share of each score: its 16 values' dot, scaled, and its share of the bias for keys near enough
        float sc[AT_PER][I8_KEYS];
#pragma unroll
        for (int c = 0; c < I8_KEYS; ++c)
#pragma unroll
            for (int h = 0; h < AT_PER; ++h) {
                int dot = STRATA_DP4A(qc[h][0], kc[c].x(), 0);
                dot = STRATA_DP4A(qc[h][1], kc[c].y(), dot);
                dot = STRATA_DP4A(qc[h][2], kc[c].z(), dot);
                dot = STRATA_DP4A(qc[h][3], kc[c].w(), dot);
                sc[h][c] = (float) dot * (qd[h] * kd[c]);
            }
        if (p - (k0 + I8_KEYS - 1) < a.rel.extent) {   // no shuffles in here: the groups may differ
#pragma unroll
            for (int c = 0; c < I8_KEYS; ++c) {
                const int dist = p - (k0 + c);
                if (!live[c] || dist >= a.rel.extent) continue;
                const float* tb = a.rel.table + size_t(dist) * n_rel;
#pragma unroll
                for (int i = 0; i < I8_RELS; ++i) {
                    const int j = sub + I8_LANES * i;
                    if (j < n_rel) {
                        const float w = tb[j];
#pragma unroll
                        for (int h = 0; h < AT_PER; ++h) sc[h][c] += w * rr[h][i];
                    }
                }
            }
        }
#pragma unroll
        for (int o = I8_LANES / 2; o > 0; o >>= 1)
#pragma unroll
            for (int h = 0; h < AT_PER; ++h)
#pragma unroll
                /*
                DPCT1108:135: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                for (int c = 0; c < I8_KEYS; ++c) sc[h][c] +=
                    dpct::experimental::permute_sub_group_by_xor(
                        0xffffffffu,
                        sycl::ext::oneapi::this_work_item::get_sub_group(),
                        sc[h][c], o);
        // the online softmax, then the values: each key's 16 values converted once for the three heads
        float alpha[AT_PER], pv[AT_PER][I8_KEYS];
#pragma unroll
        for (int h = 0; h < AT_PER; ++h) {
            float mx = -INFINITY;
#pragma unroll
            for (int c = 0; c < I8_KEYS; ++c)
                if (live[c]) mx = sycl::fmax(mx, sc[h][c]);
            const float m_new = sycl::fmax(m[h], mx);
            alpha[h] =
                m[h] == -INFINITY ? 0.0f : sycl::native::exp(m[h] - m_new);
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < I8_KEYS; ++c) {
                const float pc = live[c]
                                     ? sycl::native::exp(sc[h][c] - m_new)
                                     : 0.0f; // m_new is finite with a live key
                sum += pc;
                pv[h][c] = pc * vd[c];
            }
            m[h] = m_new;
            l[h] = l[h] * alpha[h] + sum;
        }
#pragma unroll
        for (int h = 0; h < AT_PER; ++h)
#pragma unroll
            for (int i = 0; i < I8_DIMS; ++i) acc[h][i] *= alpha[h];
#pragma unroll
        for (int c = 0; c < I8_KEYS; ++c) {
            float x[I8_DIMS];
            codes4(vc[c].x(), x), codes4(vc[c].y(), x + 4),
                codes4(vc[c].z(), x + 8), codes4(vc[c].w(), x + 12);
#pragma unroll
            for (int h = 0; h < AT_PER; ++h)
#pragma unroll
                for (int i = 0; i < I8_DIMS; ++i) acc[h][i] += pv[h][c] * x[i];
        }
        k0 += I8_STEP;
    }
    // the warp's groups merged: group 0 with 1 and 2 with 3, then the pairs (the earlier group's terms first)
#pragma unroll
    for (int o = I8_LANES; o < WARP; o <<= 1)
#pragma unroll
        for (int h = 0; h < AT_PER; ++h) {
            /*
            DPCT1108:136: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float mo = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                m[h], o);
            /*
            DPCT1108:137: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float lo_ = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                l[h], o);
            const bool first = (lane & o) == 0;
            const float M = sycl::fmax(m[h], mo);
            const float fa =
                m[h] == -INFINITY ? 0.0f : sycl::native::exp(m[h] - M);
            const float fb = mo == -INFINITY ? 0.0f : sycl::native::exp(mo - M);
            const float f0 = first ? fa : fb, f1 = first ? fb : fa;
            const float l0 = first ? l[h] : lo_, l1 = first ? lo_ : l[h];
            l[h] = l0 * f0 + l1 * f1;
#pragma unroll
            for (int i = 0; i < I8_DIMS; ++i) {
                /*
                DPCT1108:138: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                const float ao = dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(),
                    acc[h][i], o);
                const float a0 = first ? acc[h][i] : ao, a1 = first ? ao : acc[h][i];
                acc[h][i] = a0 * f0 + a1 * f1;
            }
            m[h] = M;
        }
    if (lane < I8_LANES) {
#pragma unroll
        for (int h = 0; h < AT_PER; ++h) {
            if (lane == 0) sm_m[warp][h] = m[h], sm_l[warp][h] = l[h];
#pragma unroll
            for (int i = 0; i < I8_DIMS; ++i) sm_o[warp][h][I8_DIMS * sub + i] = acc[h][i];
        }
    }
    /*
    DPCT1065:254: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int d = item_ct1.get_local_id(2); // a thread a value dimension
#pragma unroll
    for (int h = 0; h < AT_PER; ++h) {
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < I8_WARPS; ++w) M = sycl::fmax(M, sm_m[w][h]);
        float L = 0.0f, O = 0.0f;
#pragma unroll
        for (int w = 0; w < I8_WARPS; ++w) {
            const float f = sm_m[w][h] == -INFINITY
                                ? 0.0f
                                : sycl::native::exp(sm_m[w][h] - M);
            L += sm_l[w][h] * f;
            O += sm_o[w][h][d] * f;
        }
        const size_t th = size_t(t) * a.n_head + g * AT_PER + h;
        if (r.nslot > 1) {
            float *wp =
                a.work + (th * a.nslot + item_ct1.get_group(2)) * (AT_DIM + 2);
            if (d == 0) wp[0] = M, wp[1] = L;
            wp[2 + d] = O;
        } else {
            const float sink = a.sinks ? a.sinks[th] : -INFINITY;
            const float mx = sycl::fmax(M, sink);
            const float f = M == -INFINITY ? 0.0f : sycl::native::exp(M - mx);
            const float denom =
                L * f +
                (sink == -INFINITY ? 0.0f : sycl::native::exp(sink - mx));
            a.out[th * AT_DIM + d] = O * f / denom;
        }
    }
}

// ---- attention on the tensor cores (a prompt batch) -------------------------------------------------------------

// The tensor cores' m16n8k16 step and ldmatrix (kernels.cu's): Turing runs the step as two of k = 8 on the same
// registers.  AMD builds and cards before Turing have neither; there the host never launches attn_mma_kernel
// (mma_attention()).
#if 1   // SYCL port: inline PTX mma/ldmatrix; the XMX joint_matrix version is follow-up work
#define INKLING_MMA 0
#else
#define INKLING_MMA 1
#endif

#if INKLING_MMA
__dpct_inline__ void mma16816(float *c, const uint32_t *a, const uint32_t *b) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#else
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                   "r"(b[1]));
#else
    {
        volatile void *d_mat_frag_ct1[4] = {&c[0], &c[1], &c[2], &c[3]};
        sycl::vec<uint32_t, 4> a_mat_frag_ct1(a[0], a[1], a[2], a[3]);
        sycl::vec<uint32_t, 2> b_mat_frag_ct1(b[0], b[1]);
        sycl::vec<float, 4> c_mat_frag_ct1(c[0], c[1], c[2], c[3]);
        dpct::experimental::matrix::mma<16, 8, 16, sycl::half, float>(
            reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1,
            &b_mat_frag_ct1, &c_mat_frag_ct1);
    }
#endif
#endif
}

__dpct_inline__ void ldmatrix_x4(uint32_t *r, const sycl::half *p, bool trans) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    if (trans)
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                     "{%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                     : "r"(a));
#else
        dpct::experimental::matrix::ldmatrix((uintptr_t)a, &r[0], &r[1], &r[2],
                                             &r[3], true);
#endif
    else
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
            : "r"(a));
#else
        dpct::experimental::matrix::ldmatrix((uintptr_t)a, &r[0], &r[1], &r[2],
                                             &r[3]);
#endif
}

__dpct_inline__ uint32_t half2_bits(float lo, float hi) {
    const sycl::half2 h =
        sycl::float2(lo, hi).convert<sycl::half, sycl::rounding_mode::rte>();
    return *reinterpret_cast<const uint32_t*>(&h);
}
#endif

// attn_mma_kernel's shape: MMA_WARPS warps of 16 query rows, row r of the block token r / 3 and head r % 3 of the
// group (MMA_TOKENS tokens; the last row of the last warp is unused), keys MMA_KEYS at a time.
constexpr int MMA_WARPS = 4, MMA_KEYS = 32, MMA_TOKENS = 16 * MMA_WARPS / AT_PER;
constexpr int MMA_KROW = AT_DIM + 8;                                // a key or value row in shared memory, halves
constexpr int MMA_BIAS = MMA_KEYS + 8;                              // a warp's distances a tile (5 tokens + keys)
constexpr int MMA_TROW = 16 + 8;                                    // a row of the bias table, halves
constexpr int MMA_SLICE = 16 * (MMA_WARPS - 1) / AT_PER + MMA_BIAS;  // the block's distances a tile
static_assert(16 * MMA_WARPS / AT_PER * AT_PER >= 16 * MMA_WARPS - 2, "attn_mma_kernel: at most 2 unused rows");

// The unsplit path on the tensor cores, the way kernels.cu's attn_flash128_kernel goes: the queries (scaled to log2
// units and rounded to halves) stay in registers, a tile's keys and values go to shared memory as halves (a Q8_0
// cache's codes times their scales, rounded), S = Q K^T and O += P V are m16n8k16 steps that sum floats, and each
// row keeps its online softmax in floats.  The relative-position bias is a product too: the warp's rows' R (16
// values, rounded to halves) times the table's rows of the distances the tile spans (halves), a warp's 16 rows by
// MMA_BIAS distances, put in shared memory as floats and read back at each score's distance.  Only the tiles
// that hold a distance below the extent compute it.  The tiles are counted from position 0, so a row's sums do
// not depend on the other rows of the block.  The sink is added at the end, as one more weight of no value.
template <bool KQ8, bool VQ8>
/*
DPCT1110:139: The total declared local variable size in device function
attn_mma_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void attn_mma_kernel(AttnArgs a) {
#if INKLING_MMA
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &ks = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    sycl::half[MMA_KEYS][MMA_KROW]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &vs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        sycl::half[MMA_KEYS][MMA_KROW]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &slice = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        sycl::half[MMA_SLICE][MMA_TROW]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &bias = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[MMA_WARPS][16][MMA_BIAS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    constexpr float LOG2E = 1.44269504f;
    const int lane = item_ct1.get_local_id(2) % WARP,
              w = item_ct1.get_local_id(2) / WARP, gid = lane / 4,
              tig = lane % 4;
    const int g = item_ct1.get_group(1),
              t0 = (item_ct1.get_group_range(2) - 1 - item_ct1.get_group(2)) *
                   MMA_TOKENS; // the longest blocks first
    const int nt = sycl::min(MMA_TOKENS, a.rows - t0), wt0 = 16 * w / AT_PER;
    const long long p0 = a.pos + t0;
    int tok[2], head[2];   // the thread's rows gid and gid + 8 of the warp's, as tokens of the block
    bool row_ok[2];
    for (int i = 0; i < 2; ++i) {
        const int r = 16 * w + gid + 8 * i;
        tok[i] = r / AT_PER;
        head[i] = g * AT_PER + r % AT_PER;
        row_ok[i] = tok[i] < nt;
    }
    uint32_t qa[8][4], ra[4];   // a block of 16 values a step, in log2 units; R the same
    for (int i = 0; i < 2; ++i) {
        const size_t th = (size_t(t0) + tok[i]) * a.n_head + head[i];
        for (int u = 0; u < 2; ++u) {   // values 2 tig.. and 8 + 2 tig.. of each block of 16
            for (int k = 0; k < 8; ++k) {
                sycl::float2 x = sycl::float2(0.0f, 0.0f);
                if (row_ok[i]) x = *reinterpret_cast<const sycl::float2 *>(
                    a.q + th * AT_DIM + 16 * k + 8 * u + 2 * tig);
                qa[k][i + 2 * u] = half2_bits(x.x() * a.scale * LOG2E,
                                              x.y() * a.scale * LOG2E);
            }
            const float* rr = a.rel.r + th * 16 + 8 * u + 2 * tig;
            ra[i + 2 * u] = row_ok[i] ? half2_bits(rr[0] * LOG2E, rr[1] * LOG2E) : 0u;
        }
    }
    float o[16][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    // the block's tokens attend positions [lo, hi): from the first one's window to the last one
    const long long hi = p0 + nt,
                    lo = a.window > 0 ? sycl::max(0LL, p0 - a.window + 1) : 0;
    for (long long k0 = lo / MMA_KEYS * MMA_KEYS; k0 < hi; k0 += MMA_KEYS) {
        /*
        DPCT1118:140: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:255: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // the last tile's keys, values and bias are read
        // the tile's keys and values, 8 values a thread at a time; positions outside [lo, hi) are zeros (their
        // cache rows may be anything), which no weight reads
        for (int i = item_ct1.get_local_id(2); i < MMA_KEYS * AT_DIM / 8;
             i += MMA_WARPS * WARP) {
            const int r = i / (AT_DIM / 8), c = 8 * (i % (AT_DIM / 8));
            const long long key = k0 + r;
            sycl::uint4 kx = sycl::uint4(0, 0, 0, 0), vx = kx;
            if (key >= lo && key < hi) {
                const size_t at = (size_t(a.ring > 0 ? key % a.ring : key) * a.kv_heads + g) * AT_DIM + c;
                const auto load = [&](const sycl::half *f16, const Kvq8 &q8,
                                      bool quant) {
                    if (!quant) return *reinterpret_cast<const sycl::uint4 *>(
                        f16 + at);
                    const sycl::uint2 codes =
                        *reinterpret_cast<const sycl::uint2 *>(q8.codes + at);
                    const float d =
                        sycl::vec<sycl::half, 1>(
                            sycl::bit_cast<sycl::half, unsigned short>(
                                q8.scales[at / 32]))
                            .convert<float,
                                     sycl::rounding_mode::automatic>()[0];
                    uint32_t h[4];
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t word = j < 2 ? codes.x() : codes.y(),
                                       sh = 16 * (j % 2);
                        h[j] = half2_bits((float) (signed char) (word >> sh) * d,
                                          (float) (signed char) (word >> (sh + 8)) * d);
                    }
                    return sycl::uint4(h[0], h[1], h[2], h[3]);
                };
                kx = load(a.kc, a.kq, KQ8);
                vx = load(a.vc, a.vq, VQ8);
            }
            *reinterpret_cast<sycl::uint4 *>(&ks[r][c]) = kx;
            *reinterpret_cast<sycl::uint4 *>(&vs[r][c]) = vx;
        }
        // the table's rows of the block's distances d = p0 - (k0 + MMA_KEYS - 1) + e, zeros past the extent and
        // before 0 (the padded row, and keys after the token, which the mask removes)
        const long long d0 = p0 - (k0 + MMA_KEYS - 1);
        const bool near = d0 < a.rel.extent;
        if (near)
            for (int i = item_ct1.get_local_id(2); i < MMA_SLICE * 8;
                 i += MMA_WARPS * WARP) {
                const int e = i / 8, j = 2 * (i % 8);
                const long long d = d0 + e;
                sycl::float2 x = sycl::float2(0.0f, 0.0f);
                if (d >= 0 && d < a.rel.extent) x =
                    *reinterpret_cast<const sycl::float2 *>(a.rel.table +
                                                            d * 16 + j);
                *reinterpret_cast<uint32_t *>(&slice[e][j]) =
                    half2_bits(x.x(), x.y());
            }
        /*
        DPCT1118:141: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:256: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        float s[MMA_KEYS / 8][4];
        for (int n = 0; n < MMA_KEYS / 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            for (int k = 0; k < 8; k += 2) {
                uint32_t b[4];   // keys 8 n.., values 16 k.. and 16 (k + 1)..
                ldmatrix_x4(b, &ks[8 * n + lane % 8][16 * k + 8 * (lane / 8)], false);
                mma16816(s[n], qa[k], b);
                mma16816(s[n], qa[k + 1], b + 2);
            }
        }
        if (near) {
            // the warp's rows times its distances: column e is distance p0 + wt0 - (k0 + MMA_KEYS - 1) + e
            for (int n = 0; n < MMA_BIAS / 8; ++n) {
                const sycl::half *tr = &slice[wt0 + 8 * n + gid][2 * tig];
                const uint32_t b[2] = {*reinterpret_cast<const uint32_t*>(tr),
                                       *reinterpret_cast<const uint32_t*>(tr + 8)};
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                mma16816(c, ra, b);
                *reinterpret_cast<sycl::float2 *>(
                    &bias[w][gid][8 * n + 2 * tig]) = sycl::float2(c[0], c[1]);
                *reinterpret_cast<sycl::float2 *>(
                    &bias[w][gid + 8][8 * n + 2 * tig]) =
                    sycl::float2(c[2], c[3]);
            }
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
        }
        for (int i = 0; i < 2; ++i) {
            const long long own = p0 + tok[i];   // the row's token attends positions (own - window, own]
            const int e0 = tok[i] - wt0 + MMA_KEYS - 1;
            float mx = -INFINITY;
#pragma unroll
            for (int n = 0; n < MMA_KEYS / 8; ++n)
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    const int c = 8 * n + 2 * tig + j % 2;
                    const long long key = k0 + c;
                    if (!row_ok[i] || key > own || (a.window > 0 && key <= own - a.window))
                        s[n][j] = -INFINITY;
                    else if (near)
                        s[n][j] += bias[w][gid + 8 * i][e0 - c];
                    mx = sycl::fmax(mx, s[n][j]);
                }
            /*
            DPCT1108:142: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            mx = sycl::fmax(
                mx,
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), mx, 1));
            /*
            DPCT1108:143: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            mx = sycl::fmax(
                mx,
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), mx, 2));
            const float now = sycl::fmax(m[i], mx),
                        base = now == -INFINITY ? 0.0f : now;
            const float f = sycl::exp2(m[i] - base);
            m[i] = now;
            l[i] *= f;
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                o[n][2 * i] *= f;
                o[n][2 * i + 1] *= f;
            }
#pragma unroll
            for (int n = 0; n < MMA_KEYS / 8; ++n)
#pragma unroll
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    s[n][j] = sycl::exp2(s[n][j] - base);
                    l[i] += s[n][j];
                }
        }
        for (int k = 0; k < MMA_KEYS / 16; ++k) {   // keys 16 k..: the weights as an A fragment
            const uint32_t pa[4] = {half2_bits(s[2 * k][0], s[2 * k][1]), half2_bits(s[2 * k][2], s[2 * k][3]),
                                    half2_bits(s[2 * k + 1][0], s[2 * k + 1][1]),
                                    half2_bits(s[2 * k + 1][2], s[2 * k + 1][3])};
            for (int n = 0; n < 16; n += 2) {
                uint32_t b[4];   // keys 16 k.., values 8 n.. and 8 (n + 1)..
                ldmatrix_x4(b, &vs[16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                mma16816(o[n], pa, b);
                mma16816(o[n + 1], pa, b + 2);
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        /*
        DPCT1108:144: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        l[i] += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            l[i], 1);
        /*
        DPCT1108:145: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        l[i] += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            l[i], 2);
        if (!row_ok[i]) continue;
        const size_t th = (size_t(t0) + tok[i]) * a.n_head + head[i];
        float f = 1.0f;
        if (a.sinks) {   // the sink, in log2 units like the scores
            const float sink = a.sinks[th] * LOG2E,
                        now = sycl::fmax(m[i], sink);
            f = sycl::exp2(m[i] - now);
            l[i] = l[i] * f + sycl::exp2(sink - now);
        }
        float* y = a.out + th * AT_DIM + 2 * tig;
#pragma unroll
        for (int n = 0; n < 16; ++n)
            *reinterpret_cast<sycl::float2 *>(y + 8 * n) = sycl::float2(
                o[n][2 * i] * f / l[i], o[n][2 * i + 1] * f / l[i]);
    }
#else
    (void) a;   // never launched without the tensor cores (mma_attention())
#endif
}

// Whether attn_mma_kernel runs as such here: built for Turing or newer, and not from an older card's PTX compiled
// for this one (whose instructions are that card's).
bool mma_attention() try {
#if 1   // SYCL port
    return false;
#else
    dpct::kernel_function_info fa{};
    return DPCT_CHECK_ERROR(dpct::get_kernel_function_info(
               &fa, (const void *)attn_mma_kernel<false, false>)) == 0 &&
           fa.ptxVersion >= 75;
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <int TQ, bool KQ8, bool VQ8>
void launch_attn(const AttnArgs &a, dpct::dim3 g, dpct::queue_ptr st) try {
    constexpr size_t smem = attn_smem(TQ);
    if constexpr (smem > 48 * 1024) {
        // the device's own setting: asked on whichever card the call is for
        const dpct::err0 e =
            /*
            DPCT1027:260: The call to cudaFuncSetAttribute was replaced with 0
            because SYCL currently does not support corresponding setting.
            */
            0;
        /*
        DPCT1000:258: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (e != 0)
            /*
            DPCT1009:259: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1001:257: The statement could not be removed.
            */
            throw std::runtime_error(
                std::string("inkling_attend: shared memory: ") +
                dpct::get_error_string_dummy(e));
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(smem), cgh);

            cgh.parallel_for<dpct_kernel_name<
                class attn_kernel_6805a8, dpct_kernel_scalar<TQ>,
                dpct_kernel_scalar<KQ8>, dpct_kernel_scalar<VQ8>>>(
                sycl::nd_range<3>(g * sycl::range(1, 1, AT_THREADS),
                                  sycl::range(1, 1, AT_THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attn_kernel<TQ, KQ8, VQ8>(
                            a, dpct_local_acc_ct1
                                   .get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
                    });
        });
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <int TQ>
void launch_attn_q(const AttnArgs &a, dpct::dim3 g, dpct::queue_ptr st,
                   bool kq8, bool vq8) {
    if (kq8 && vq8) launch_attn<TQ, true, true>(a, g, st);
    else if (kq8) launch_attn<TQ, true, false>(a, g, st);
    else if (vq8) launch_attn<TQ, false, true>(a, g, st);
    else launch_attn<TQ, false, false>(a, g, st);
}

int split_keys(long long room) {
    const long long s = (room + kInklingMaxSplits - 1) / kInklingMaxSplits;
    return (int) std::max<long long>(AT_MIN_SPLIT, (s + AT_KEYS - 1) / AT_KEYS * AT_KEYS);
}

// The split path's keys a split and splits of a token whose positions + 1 are at most span_room.
void splits_of(long long span_room, int window, int& split, int& nslot) {
    // a window layer's tokens see `window` keys whatever the room: its splits are sized for those
    split = split_keys(window > 0 ? std::min<long long>(span_room, window) : span_room);
    long long n = (span_room + split - 1) / split;
    if (window > 0) n = std::min<long long>(n, (window + split - 1) / split + 1);
    nslot = (int) std::max<long long>(n, 1);
}

template <class Rows>
void launch_dec(const AttnArgs &a, const Rows &rows, int ntok, bool kq, bool vq,
                bool int8_dot, dpct::queue_ptr st) {
    const dpct::dim3 gd((unsigned)a.nslot, a.kv_heads, ntok);
    if (int8_dot && kq && vq && a.window <= 0) {   // a full layer's rows of long splits by __dp4a, the others not
        AttnArgs b = a;
        b.i8_from = I8_MIN_SPLIT;
        const int big = rows_from(rows, a, ntok, I8_MIN_SPLIT);
        if (big > 0) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<
                dpct_kernel_name<class attn_dec_i8_kernel_ca200f, Rows>>(
                sycl::nd_range<3>(gd * sycl::range(1, 1, I8_WARPS * WARP),
                                  sycl::range(1, 1, I8_WARPS * WARP)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attn_dec_i8_kernel<Rows>(b, rows);
                    });
        }
        if (big < ntok) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(st->get_device(),
                                         {sycl::aspect::fp16});

            st->parallel_for<dpct_kernel_name<class attn_dec_kernel_ac24cb,
                                              dpct_kernel_scalar<true>,
                                              dpct_kernel_scalar<true>, Rows>>(
                sycl::nd_range<3>(gd * sycl::range(1, 1, DEC_WARPS * WARP),
                                  sycl::range(1, 1, DEC_WARPS * WARP)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attn_dec_kernel<true, true, Rows>(b, rows);
                    });
        }
    } else if (kq && vq) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_dec_kernel_1015f8,
                                          dpct_kernel_scalar<true>,
                                          dpct_kernel_scalar<true>, Rows>>(
            sycl::nd_range<3>(gd * sycl::range(1, 1, DEC_WARPS * WARP),
                              sycl::range(1, 1, DEC_WARPS * WARP)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_dec_kernel<true, true, Rows>(a, rows);
            });
    } else if (kq) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_dec_kernel_2dcf07,
                                          dpct_kernel_scalar<true>,
                                          dpct_kernel_scalar<false>, Rows>>(
            sycl::nd_range<3>(gd * sycl::range(1, 1, DEC_WARPS * WARP),
                              sycl::range(1, 1, DEC_WARPS * WARP)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_dec_kernel<true, false, Rows>(a, rows);
            });
    } else if (vq) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_dec_kernel_ce9d7d,
                                          dpct_kernel_scalar<false>,
                                          dpct_kernel_scalar<true>, Rows>>(
            sycl::nd_range<3>(gd * sycl::range(1, 1, DEC_WARPS * WARP),
                              sycl::range(1, 1, DEC_WARPS * WARP)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_dec_kernel<false, true, Rows>(a, rows);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_dec_kernel_97416c,
                                          dpct_kernel_scalar<false>,
                                          dpct_kernel_scalar<false>, Rows>>(
            sycl::nd_range<3>(gd * sycl::range(1, 1, DEC_WARPS * WARP),
                              sycl::range(1, 1, DEC_WARPS * WARP)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_dec_kernel<false, false, Rows>(a, rows);
            });
    }
    check_launch("inkling_attend");
    if (a.nslot > 1) {   // a row of one split: the block wrote the token's output itself
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<
                dpct_kernel_name<class attn_join_kernel_1f09f8, Rows>>(
                sycl::nd_range<3>(sycl::range(1, 1, ntok * a.n_head) *
                                      sycl::range(1, 1, AT_DIM),
                                  sycl::range(1, 1, AT_DIM)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    attn_join_kernel<Rows>(a, rows);
                });
        }
        check_launch("inkling_attend (join)");
    }
}

// ---- convolution -----------------------------------------------------------------------------------------------

// The convolution of one token of one channel from its inputs s0 (oldest) .. s2 and v (the token's), in
// upstream's order: ssm_conv's sum, then the four input-dependent taps lag 0..3, then the residual and the scale.
__dpct_inline__ float conv_one(float s0, float s1, float s2, float v,
                               const float *wc, const float *tp, float scale) {
    float y = 0.0f;
    y += s0 * wc[0];
    y += s1 * wc[1];
    y += s2 * wc[2];
    y += v * wc[3];
    if (tp) {
        y += v * tp[0];
        y += s2 * tp[1];
        y += s1 * tp[2];
        y += s0 * tp[3];
    }
    return (v + y) * scale;
}

// The convolution's scale: 1/3 with taps, 1/sqrt(5) without.
inline float conv_scale(const float* taps) { return 1.0f / sqrtf(float((taps ? 2 : 1) * 4 + 1)); }

// Where conv_walk_kernel's tokens are: the call's tokens one after the other in one sequence, with one state (a
// thread walks them all), or each row a sequence's of its own with its own state (row blockIdx.y).
struct ConvOne {
    float* state;
    int rows;
    int first() const { return 0; }
    int end() const { return rows; }
    float* at(int) const { return state; }
};
struct ConvSeqs {
    float* state[kInklingSeqRows];
    int first() const {
        return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(1);
    }
    int end() const {
        return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(
                   1) +
                   1;
    }
    float* at(int t) const { return state[t]; }
};

template <bool ACC, class Rows>
__dpct_inline__ void conv_walk_kernel(const float *__restrict__ x, Rows rows,
                                      const float *__restrict__ w,
                                      const float *__restrict__ taps, int heads,
                                      float *__restrict__ out, int channels,
                                      float scale) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= channels) return;
    const float* wc = w + 4 * size_t(c);
    const int h = taps ? c / (channels / heads) : 0;
    const int r0 = rows.first(), r1 = rows.end();
    float* st = rows.at(r0) + size_t(c) * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    for (int row = r0; row < r1; ++row) {
        const size_t i = size_t(row) * channels + c;
        const float v = x[i];
        const float y = conv_one(s0, s1, s2, v, wc, taps ? taps + (size_t(row) * heads + h) * 4 : nullptr, scale);
        /*
        DPCT1013:261: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        out[i] = ACC ? out[i] + y : y; // no fma with the scale: axpy's bits
        s0 = s1;
        s1 = s2;
        s2 = v;
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
}

template <bool ACC>
__dpct_inline__ void
conv_rows_kernel(const float *__restrict__ x, const float *__restrict__ state,
                 const float *__restrict__ w, const float *__restrict__ taps,
                 int heads, float *__restrict__ out, int channels, int row0,
                 float scale) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= channels) return;
    const int row = row0 + item_ct1.get_group(1);
    const float* st = state + size_t(c) * 3;
    auto in = [&](int r) { return r >= 0 ? x[size_t(r) * channels + c] : st[3 + r]; };
    const int h = taps ? c / (channels / heads) : 0;
    const float y = conv_one(in(row - 3), in(row - 2), in(row - 1), in(row), w + 4 * size_t(c),
                             taps ? taps + (size_t(row) * heads + h) * 4 : nullptr, scale);
    float* o = out + size_t(row) * channels + c;
    /*
    DPCT1013:262: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    *o = ACC ? *o + y : y;
}

__dpct_inline__ void conv_state_kernel(const float *__restrict__ x,
                                       float *__restrict__ state, int channels,
                                       int rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= channels) return;
    float* st = state + size_t(c) * 3;
    auto in = [&](int r) { return r >= 0 ? x[size_t(r) * channels + c] : st[3 + r]; };
    const float a = in(rows - 3), b = in(rows - 2), d = in(rows - 1);
    st[0] = a;
    st[1] = b;
    st[2] = d;
}

// The state as it is after token t of x (t < rows) for each of the call's snapshot tokens (Snaps), row blockIdx.y
// of them into snap + blockIdx.y * stride: what conv_state_kernel leaves after t + 1 tokens.  Before the call
// advances the state.
constexpr int SNAPS_MAX = 32;
struct Snaps {
    int t[SNAPS_MAX];
};
__dpct_inline__ void conv_snaps_kernel(const float *__restrict__ x,
                                       const float *__restrict__ state,
                                       int channels, Snaps snaps,
                                       float *__restrict__ snap,
                                       size_t stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= channels) return;
    const int t = snaps.t[item_ct1.get_group(1)];
    const float* st = state + size_t(c) * 3;
    auto in = [&](int r) { return r >= 0 ? x[size_t(r) * channels + c] : st[3 + r]; };
    float *to = snap + item_ct1.get_group(1) * stride + size_t(c) * 3;
    to[0] = in(t - 2);
    to[1] = in(t - 1);
    to[2] = in(t);
}

constexpr int CONV_ROWS = 16;

// ---- mix -------------------------------------------------------------------------------------------------------

constexpr int MIX_THREADS = 512;
constexpr int MIX_SUMS = kInklingMaxStates * 5;   // a state's sum of squares and its dot with each of 4 queries

// A block a token, in one pass over the states: every state's sum of squares and its dots with the raw queries
// together (s_hat . q = inv * (s . q)), the block's sums of all of them at once, the softmaxes, then the outputs.
/*
DPCT1110:146: The total declared local variable size in device function
mix_kernel exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void mix_kernel(InklingMix m, int n, float eps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    float[MIX_THREADS / WARP][MIX_SUMS]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &tot =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[MIX_SUMS]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &wts = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[4][kInklingMaxStates]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const size_t t = item_ct1.get_group(2);
    const int ns = m.n_state, nq = m.n_query,
              lane = item_ct1.get_local_id(2) % WARP,
              warp = item_ct1.get_local_id(2) / WARP;
    float acc[MIX_SUMS];
#pragma unroll
    for (int j = 0; j < MIX_SUMS; ++j) acc[j] = 0.0f;
    for (int i = item_ct1.get_local_id(2); i < n; i += MIX_THREADS) {
        float qv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) qv[q] = q < nq ? m.query[size_t(q) * n + i] : 0.0f;
#pragma unroll
        for (int k = 0; k < kInklingMaxStates; ++k) {
            if (k >= ns) break;
            const float x = m.state[k][t * n + i];
            acc[k * 5] += x * x;
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[k * 5 + 1 + q] += x * qv[q];
        }
    }
#pragma unroll
    for (int j = 0; j < MIX_SUMS; ++j) {
        const float v = warp_sum(acc[j]);
        if (lane == 0) part[warp][j] = v;
    }
    /*
    DPCT1065:263: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < MIX_SUMS) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < MIX_THREADS / WARP; ++w) v +=
            part[w][item_ct1.get_local_id(2)];
        tot[item_ct1.get_local_id(2)] = v;
    }
    /*
    DPCT1065:264: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < nq) {
        const int q = item_ct1.get_local_id(2);
        const float scale = 1.0f / sycl::sqrt((float)n);
        float sc[kInklingMaxStates], mx = -INFINITY;
        for (int k = 0; k < ns; ++k) {
            const float inv = 1.0f / sycl::sqrt(tot[k * 5] / float(n) + eps);
            sc[k] = tot[k * 5 + 1 + q] * inv * scale;
            mx = sycl::fmax(mx, sc[k]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < ns; ++k) sum +=
            (sc[k] = sycl::native::exp(sc[k] - mx));
        // the weights times the states' norms, so that the sum below reads the raw states
#pragma unroll
        for (int k = 0; k < ns; ++k)
            wts[q][k] = sc[k] / sum / sycl::sqrt(tot[k * 5] / float(n) + eps);
    }
    /*
    DPCT1065:265: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = item_ct1.get_local_id(2); i < n; i += MIX_THREADS) {
        float xs[kInklingMaxStates];
#pragma unroll
        for (int k = 0; k < kInklingMaxStates; ++k) xs[k] = k < ns ? m.state[k][t * n + i] : 0.0f;
        for (int q = 0; q < nq; ++q) {
            float y = 0.0f;
#pragma unroll
            for (int k = 0; k < kInklingMaxStates; ++k)
                if (k < ns) y += wts[q][k] * xs[k];
            m.out[q][t * n + i] = y;
        }
    }
}

constexpr int MIX4_MAX = 768;   // threads: a float4 each, so n <= 3072
// mix_kernel for n a multiple of 4 up to 4 * MIX4_MAX, NS states: a thread a float4 of every state, read once into
// registers and kept for the output, so that a token's whole read is on its way at once.  The same sums otherwise.
template <int NS>
/*
DPCT1110:147: The total declared local variable size in device function
mix4_kernel exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void mix4_kernel(InklingMix m, int n, float eps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int SUMS = NS * 5;
    auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[MIX4_MAX / WARP][SUMS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &tot =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[SUMS]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &wts =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[4][NS]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const size_t t = item_ct1.get_group(2);
    const int nq = m.n_query, lane = item_ct1.get_local_id(2) % WARP,
              warp = item_ct1.get_local_id(2) / WARP;
    const int i = item_ct1.get_local_id(2), n4 = n / 4;
    const bool own = i < n4;
    sycl::float4 xs[NS];
    float acc[SUMS];
#pragma unroll
    for (int j = 0; j < SUMS; ++j) acc[j] = 0.0f;
    if (own) {
        sycl::float4 qv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
            qv[q] = q < nq ? reinterpret_cast<const sycl::float4 *>(
                                 m.query + size_t(q) * n)[i]
                           : sycl::float4(0, 0, 0, 0);
#pragma unroll
        for (int k = 0; k < NS; ++k) xs[k] =
            reinterpret_cast<const sycl::float4 *>(m.state[k] + t * n)[i];
#pragma unroll
        for (int k = 0; k < NS; ++k) {
            const sycl::float4 x = xs[k];
            acc[k * 5] =
                x.x() * x.x() + x.y() * x.y() + x.z() * x.z() + x.w() * x.w();
#pragma unroll
            for (int q = 0; q < 4; ++q)
                acc[k * 5 + 1 + q] = x.x() * qv[q].x() + x.y() * qv[q].y() +
                                     x.z() * qv[q].z() + x.w() * qv[q].w();
        }
    }
#pragma unroll
    for (int j = 0; j < SUMS; ++j) {
        const float v = warp_sum(acc[j]);
        if (lane == 0) part[warp][j] = v;
    }
    /*
    DPCT1065:266: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int warps = item_ct1.get_local_range(2) / WARP;
    for (int j = item_ct1.get_local_id(2); j < SUMS;
         j += item_ct1.get_local_range(
             2)) { // a block of n <= 128 has fewer threads than sums
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < warps; ++w) v += part[w][j];
        tot[j] = v;
    }
    /*
    DPCT1065:267: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < nq) {
        const int q = item_ct1.get_local_id(2);
        const float scale = 1.0f / sycl::sqrt((float)n);
        float sc[NS], inv[NS], mx = -INFINITY;
#pragma unroll
        for (int k = 0; k < NS; ++k) {
            inv[k] = 1.0f / sycl::sqrt(tot[k * 5] / float(n) + eps);
            sc[k] = tot[k * 5 + 1 + q] * inv[k] * scale;
            mx = sycl::fmax(mx, sc[k]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < NS; ++k) sum +=
            (sc[k] = sycl::native::exp(sc[k] - mx));
        // the weights times the states' norms, so that the sum below reads the raw states
#pragma unroll
        for (int k = 0; k < NS; ++k) wts[q][k] = sc[k] / sum * inv[k];
    }
    /*
    DPCT1065:268: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (!own) return;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        if (q >= nq) break;
        sycl::float4 y = sycl::float4(0, 0, 0, 0);
#pragma unroll
        for (int k = 0; k < NS; ++k) {
            const float w = wts[q][k];
            y.x() += w * xs[k].x(), y.y() += w * xs[k].y(),
                y.z() += w * xs[k].z(), y.w() += w * xs[k].w();
        }
        reinterpret_cast<sycl::float4 *>(m.out[q] + t * n)[i] = y;
    }
}

void launch_mix4(const InklingMix &m, int n, int rows, float eps,
                 dpct::queue_ptr st) {
    const int threads = (n / 4 + WARP - 1) / WARP * WARP;
    switch (m.n_state) {
        /*
        DPCT1049:148: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 1: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class mix4_kernel_e2ca1e, dpct_kernel_scalar<1>>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) *
                                  sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mix4_kernel<1>(m, n, eps);
            });
    } break;
        /*
        DPCT1049:149: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 2: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class mix4_kernel_657c1a, dpct_kernel_scalar<2>>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) *
                                  sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mix4_kernel<2>(m, n, eps);
            });
    } break;
        /*
        DPCT1049:150: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 3: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class mix4_kernel_be89d6, dpct_kernel_scalar<3>>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) *
                                  sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mix4_kernel<3>(m, n, eps);
            });
    } break;
        /*
        DPCT1049:151: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 4: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class mix4_kernel_4d7b6e, dpct_kernel_scalar<4>>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) *
                                  sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mix4_kernel<4>(m, n, eps);
            });
    } break;
        /*
        DPCT1049:152: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 5: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class mix4_kernel_af5cb9, dpct_kernel_scalar<5>>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) *
                                  sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mix4_kernel<5>(m, n, eps);
            });
    } break;
        /*
        DPCT1049:153: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 6: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class mix4_kernel_29305b, dpct_kernel_scalar<6>>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) *
                                  sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mix4_kernel<6>(m, n, eps);
            });
    } break;
        /*
        DPCT1049:154: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    case 7: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class mix4_kernel_f0953c, dpct_kernel_scalar<7>>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) *
                                  sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mix4_kernel<7>(m, n, eps);
            });
    } break;
        /*
        DPCT1049:155: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    default: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class mix4_kernel_f3d715, dpct_kernel_scalar<8>>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) *
                                  sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mix4_kernel<8>(m, n, eps);
            });
    } break;
    }
}

// ---- controller, SiTU ------------------------------------------------------------------------------------------

__dpct_inline__ void ctrl_mix_kernel(const float *cur,
                                     const float *__restrict__ msg,
                                     const float *__restrict__ gain,
                                     const float *__restrict__ forget,
                                     float *out, size_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        size_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const float c = cur[i];
    out[i] = (c - sycl::tanh(forget[i]) * c) + sycl::tanh(gain[i]) * msg[i];
}

__dpct_inline__ float situ_one(float g, float u, float beta, float lb) {
    const float act = sycl::tanh(g * (1.0f / beta)) * beta * sigmoidf(g);
    const float up = lb > 0.0f ? sycl::tanh(u * (1.0f / lb)) * lb : u;
    return act * up;
}

__dpct_inline__ sycl::float2 row_scales(const ExpertScales &s, int row) {
    if (!s.gate && !s.up) return sycl::float2(1.0f, 1.0f);
    const int e = s.ids ? s.ids[row] : s.e[row];
    return sycl::float2(s.gate ? s.gate[e] : 1.0f, s.up ? s.up[e] : 1.0f);
}

__dpct_inline__ void situ_rows_kernel(const float *__restrict__ gate_up,
                                      float *__restrict__ out, int n,
                                      float beta, float lb, ExpertScales s) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    const float *row = gate_up + size_t(item_ct1.get_group(1)) * 2 * n;
    const sycl::float2 sc = row_scales(s, item_ct1.get_group(1));
    out[size_t(item_ct1.get_group(1)) * n + i] =
        situ_one(row[i] * sc.x(), row[n + i] * sc.y(), beta, lb);
}

// ---- fused decode steps ------------------------------------------------------------------------------------------

__dpct_inline__ sycl::float4 ld4(const float *p, size_t i) {
    return *reinterpret_cast<const sycl::float4 *>(p + i);
}
__dpct_inline__ float dot4(sycl::float4 a, sycl::float4 b) {
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
}

constexpr int CTRL_ROWS = 4, CTRL_THREADS = 256, CTRL_OUT = 32;   // a block's outputs in the second launch
static_assert(kInklingMaxRank == 2 * WARP, "ctrl_up_kernel: a lane two of the rank's values");

struct CtrlArgs {
    const float* cur[4];
    const float* msg[4];
    const float* down[4];
    const float* gain[4];
    const float* forget[4];
    float* out[4];
    int n, rank, rows;
    float eps;
    float* work;
};

// The norms of cur and msg and output o of the down projection, for every row: down[o] . [cur / rms, msg / rms] is
// (down_left[o] . cur) / rms(cur) + (down_right[o] . msg) / rms(msg).  A block an output of a controller.
/*
DPCT1110:156: The total declared local variable size in device function
ctrl_down_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void ctrl_down_kernel(CtrlArgs a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    float[CTRL_THREADS / WARP][4 * CTRL_ROWS]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int o = item_ct1.get_group(2), k = item_ct1.get_group(1), n = a.n,
              rows = a.rows;
    const float* cur = a.cur[k];
    const float* msg = a.msg[k];
    const float* wr = a.down[k] + size_t(o) * 2 * n;
    float acc[4 * CTRL_ROWS];   // per row: down_left . cur, down_right . msg, cur . cur, msg . msg
#pragma unroll
    for (int j = 0; j < 4 * CTRL_ROWS; ++j) acc[j] = 0.0f;
    for (int i = 4 * item_ct1.get_local_id(2); i < n; i += 4 * CTRL_THREADS) {
        const sycl::float4 wl = ld4(wr, i), wm = ld4(wr, size_t(n) + i);
#pragma unroll
        for (int t = 0; t < CTRL_ROWS; ++t) {
            if (t >= rows) break;
            const sycl::float4 c = ld4(cur, size_t(t) * n + i),
                               m = ld4(msg, size_t(t) * n + i);
            acc[4 * t] += dot4(wl, c);
            acc[4 * t + 1] += dot4(wm, m);
            acc[4 * t + 2] += dot4(c, c);
            acc[4 * t + 3] += dot4(m, m);
        }
    }
    const int lane = item_ct1.get_local_id(2) % WARP,
              warp = item_ct1.get_local_id(2) / WARP;
#pragma unroll
    for (int j = 0; j < 4 * CTRL_ROWS; ++j) {
        const float v = warp_sum(acc[j]);
        if (lane == 0) part[warp][j] = v;
    }
    /*
    DPCT1065:269: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < rows) {
        const int t = item_ct1.get_local_id(2);
        float v[4] = {0, 0, 0, 0};
#pragma unroll
        for (int w = 0; w < CTRL_THREADS / WARP; ++w)
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] += part[w][4 * t + j];
        const float ic = 1.0f / sycl::sqrt(v[2] / float(n) + a.eps),
                    im = 1.0f / sycl::sqrt(v[3] / float(n) + a.eps);
        a.work[(size_t(k) * rows + t) * a.rank + o] = v[0] * ic + v[1] * im;
    }
}

// gain and forget of CTRL_OUT outputs of a controller (a warp four of them, a lane two of the rank's values), and
// the combine: out = (cur - tanh(forget) * cur) + tanh(gain) * msg.
/*
DPCT1110:157: The total declared local variable size in device function
ctrl_up_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void ctrl_up_kernel(CtrlArgs a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &r = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    float[CTRL_ROWS][kInklingMaxRank]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int k = item_ct1.get_group(1), n = a.n, rows = a.rows, rank = a.rank;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < rows * rank; i += CTRL_THREADS)
        r[i / rank][i % rank] = a.work[size_t(k) * rows * rank + i];
    /*
    DPCT1065:270: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float* gw = a.gain[k];
    const float* fw = a.forget[k];
    const int lane = item_ct1.get_local_id(2) % WARP,
              warp = item_ct1.get_local_id(2) / WARP;
    constexpr int PER = CTRL_OUT / (CTRL_THREADS / WARP);
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int o = item_ct1.get_group(2) * CTRL_OUT + warp * PER + j;
        if (o >= n) break;
        const bool l0 = lane < rank, l1 = lane + WARP < rank;
        const float g0 = l0 ? gw[size_t(o) * rank + lane] : 0.0f;
        const float g1 = l1 ? gw[size_t(o) * rank + lane + WARP] : 0.0f;
        const float f0 = l0 ? fw[size_t(o) * rank + lane] : 0.0f;
        const float f1 = l1 ? fw[size_t(o) * rank + lane + WARP] : 0.0f;
        float gs[CTRL_ROWS], fs[CTRL_ROWS];
#pragma unroll
        for (int t = 0; t < CTRL_ROWS; ++t) {
            const float r0 = t < rows && l0 ? r[t][lane] : 0.0f, r1 = t < rows && l1 ? r[t][lane + WARP] : 0.0f;
            gs[t] = warp_sum(g0 * r0 + g1 * r1);
            fs[t] = warp_sum(f0 * r0 + f1 * r1);
        }
#pragma unroll
        for (int t = 0; t < CTRL_ROWS; ++t)
            if (lane == t && t < rows) {
                const size_t i = size_t(t) * n + o;
                const float c = a.cur[k][i];
                a.out[k][i] = (c - sycl::tanh(fs[t]) * c) +
                              sycl::tanh(gs[t]) * a.msg[k][i];
            }
    }
}

// y[t][o] = w[o] . x[t], a warp an output, for up to CTRL_ROWS tokens.
constexpr int GEMV_WARPS = 4;
__dpct_inline__ void gemv_rows_kernel(const float *__restrict__ w,
                                      const float *__restrict__ x,
                                      float *__restrict__ y, int n_in,
                                      int n_out, int rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) % WARP,
              o = item_ct1.get_group(2) * GEMV_WARPS +
                  item_ct1.get_local_id(2) / WARP;
    if (o >= n_out) return;
    const float* wr = w + size_t(o) * n_in;
    float acc[CTRL_ROWS] = {0, 0, 0, 0};
#pragma unroll 4
    for (int i = 4 * lane; i < n_in; i += 4 * WARP) {
        const sycl::float4 wv = ld4(wr, i);
#pragma unroll
        for (int t = 0; t < CTRL_ROWS; ++t)
            if (t < rows) acc[t] += dot4(wv, ld4(x, size_t(t) * n_in + i));
    }
#pragma unroll
    for (int t = 0; t < CTRL_ROWS; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0 && t < rows) y[size_t(t) * n_out + o] = v;
    }
}

// inkling_kv_store(): a block a head of K (blockIdx.y 0) or of V (1), a thread a value of it; the tokens in turn.
constexpr int KV_ROWS = 16;
struct KvArgs {
    const float* raw[2];
    float* state[2];
    const float* w[2];
    const float* taps[2];
    float* out[2];
    InklingKvDst dst;
    int kv_heads, rows;
    float k_clip, eps;
    long long pos;
};

// Where kv_store_kernel's tokens go: the call's tokens one after the other in one sequence (a block takes them all),
// or each row a sequence's of its own (a block a row, blockIdx.z).
struct KvOne {
    int first(const KvArgs&) const { return 0; }
    int end(const KvArgs& a) const { return a.rows; }
    float* state(const KvArgs& a, int which, int) const { return a.state[which]; }
    const InklingKvDst& dst(const KvArgs& a, int) const { return a.dst; }
    long long pos(const KvArgs& a, int t) const { return a.pos + t; }
};
struct KvSeqs {
    InklingKvSeq row[kInklingSeqRows];
    int first(const KvArgs &) const {
        return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(0);
    }
    int end(const KvArgs &) const {
        return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(
                   0) +
                   1;
    }
    float* state(const KvArgs&, int which, int t) const { return which ? row[t].v_state : row[t].k_state; }
    const InklingKvDst& dst(const KvArgs&, int t) const { return row[t].dst; }
    long long pos(const KvArgs&, int t) const { return (long long) row[t].pos; }
};

template <class Rows>
/*
DPCT1110:158: The total declared local variable size in device function
kv_store_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void kv_store_kernel(KvArgs a, Rows rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[AT_DIM / WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int g = item_ct1.get_group(2), which = item_ct1.get_group(1),
              d = item_ct1.get_local_id(2), lane = d % WARP;
    const int channels = a.kv_heads * AT_DIM, c = g * AT_DIM + d;
    const float* x = a.raw[which];
    const float* taps = a.taps[which];
    const float* wc = a.w[which] + 4 * size_t(c);
    const int t0 = rows.first(a), t1 = rows.end(a);
    float* st = rows.state(a, which, t0) + size_t(c) * 3;
    const float scale = 1.0f / sycl::sqrt(float((taps ? 2 : 1) * 4 + 1));
    float s0 = st[0], s1 = st[1], s2 = st[2];
    for (int t = t0; t < t1; ++t) {
        const InklingKvDst& dst = rows.dst(a, t);
        const bool q8 = which ? dst.vc != nullptr : dst.kc != nullptr;
        const float v = x[size_t(t) * channels + c];
        const float y = conv_one(s0, s1, s2, v, wc, taps ? taps + (size_t(t) * a.kv_heads + g) * 4 : nullptr, scale);
        s0 = s1, s1 = s2, s2 = v;
        // the head's sum of squares: the warps' sums, then the four in order
        float ss = warp_sum(y * y);
        if (lane == 0) part[d / WARP] = ss;
        /*
        DPCT1118:159: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:271: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        ss = part[0] + part[1] + part[2] + part[3];
        /*
        DPCT1118:160: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:272: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        float val;
        if (which == 0) {
            const float factor = sycl::fmin(
                sycl::fmax(a.k_clip / (sycl::sqrt(ss) + 1e-6f), 0.0f), 1.0f);
            val = y * factor;
        } else {
            val = y * (1.0f / sycl::sqrt(ss / float(AT_DIM) + a.eps));
        }
        if (a.out[which]) a.out[which][size_t(t) * channels + c] = val;
        const long long p = rows.pos(a, t);
        const long long row = dst.ring > 0 ? p % dst.ring : p;
        if (!q8) {
            sycl::half *cache =
                reinterpret_cast<sycl::half *>(which ? dst.v : dst.k);
            cache[size_t(row) * channels + c] =
                sycl::vec<float, 1>(val)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
        } else {
            // a warp is a Q8_0 block of 32 values: store_q8's rule
            float amax = sycl::fabs(val);
#pragma unroll
            /*
            DPCT1108:161: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = WARP / 2; o > 0; o >>= 1) amax = sycl::fmax(
                amax, dpct::experimental::permute_sub_group_by_xor(
                          0xffffffffu,
                          sycl::ext::oneapi::this_work_item::get_sub_group(),
                          amax, o));
            const float dl = amax / 127.0f;
            const float id = dl > 0.f ? 1.0f / dl : 0.0f;
            int8_t* codes = which ? dst.vc : dst.kc;
            sycl::half *scales =
                reinterpret_cast<sycl::half *>(which ? dst.vs : dst.ks);
            const size_t at = size_t(row) * channels + c;
            if (lane == 0) scales[at / 32] =
                sycl::vec<float, 1>(dl)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
            const float q =
                sycl::fmin(sycl::fmax(sycl::rint(val * id), -127.0f), 127.0f);
            codes[at] = (int8_t) q;
        }
    }
    st[0] = s0, st[1] = s1, st[2] = s2;
}

}  // namespace

// ================================================================================================================

void inkling_clip(const float* x, int x_stride, const float* scale, float* y, int dim, int heads, int rows,
                  float threshold, InklingTau tau, int64_t pos, void* stream) {
    if (rows <= 0) return;
    if (dim <= 0 || dim > 1024) throw std::invalid_argument("inkling_clip: dim must be 1 to 1024");
    const int groups = rows * heads, per_block = 256 / WARP;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->submit([&](sycl::handler &cgh) {
            auto x_stride_x_stride_heads_dim_ct1 =
                x_stride ? x_stride : heads * dim;

            cgh.parallel_for<dpct_kernel_name<class clip_kernel_94de7a>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(groups, per_block)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        clip_kernel(x, x_stride_x_stride_heads_dim_ct1, scale,
                                    y, dim, heads, groups, threshold, tau.alpha,
                                    tau.floor, (long long)pos);
                    });
        });
    }
    check_launch("inkling_clip");
}

size_t inkling_attend_work_floats(int n_head, int dim) {
    return size_t(kInklingSplitRows) * n_head * (kInklingMaxSplits + 2) * (dim + 2);
}

namespace {
// inkling_attend()'s and inkling_attend_seqs()'s checks, and the arguments every row shares
AttnArgs attn_args(const char* who, const float* q, const InklingRel& rel, const float* sinks, int rows, float* out,
                   int dim, int n_head, int kv_heads, float scale, int window, int ring, float* work) {
    if (dim != AT_DIM || kv_heads <= 0 || n_head != AT_PER * kv_heads)
        throw std::invalid_argument(std::string(who) + ": heads of 128 values in groups of 3 only");
    if (rel.n_rel < 0 || rel.n_rel > AT_MAX_REL || (rel.n_rel > 0 && !rel.r) || !rel.table)
        throw std::invalid_argument(std::string(who) + ": bad relative-position bias");
    AttnArgs a{};
    a.q = q;
    a.rel = rel;
    a.sinks = sinks;
    a.rows = rows;
    a.n_head = n_head;
    a.kv_heads = kv_heads;
    a.window = window;
    a.ring = ring;
    a.scale = scale;
    a.out = out;
    a.work = work;
    return a;
}
}  // namespace

void inkling_attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, Kvq8 kq8, Kvq8 vq8,
                    InklingRel rel, const float* sinks, int64_t pos, int rows, float* out, int dim, int n_head,
                    int kv_heads, float scale, int window, int ring, float* work, size_t work_floats, void* stream,
                    int64_t room, bool int8_dot) {
    if (rows <= 0) return;
    AttnArgs a =
        attn_args("inkling_attend", q, rel, sinks, rows, out, dim, n_head, kv_heads, scale, window, ring, work);
    a.kc = (const sycl::half *)kcache;
    a.vc = (const sycl::half *)vcache;
    a.kq = kq8;
    a.vq = vq8;
    a.pos = (long long) pos;
    const bool kq = kq8.codes != nullptr, vq = vq8.codes != nullptr;
    const dpct::queue_ptr st = cs(stream);
    // positions + 1, at most: the caller's room when it gives one, else pos + rows
    const long long span_room = room > 0 ? (long long) room : (long long) pos + rows;

    if (work && rows <= kInklingSplitRows && span_room > 0) {
        splits_of(span_room, window, a.split, a.nslot);
        const size_t need = size_t(rows) * n_head * a.nslot * (AT_DIM + 2);
        if (need <= work_floats) {
            launch_dec(a, AttnOne{}, rows, kq, vq, int8_dot, st);
            return;
        }
    }
    if (rows >= kInklingMmaRows && rel.n_rel == 16 && mma_attention()) {
        const dpct::dim3 gm(grid(rows, MMA_TOKENS), kv_heads);
        if (kq && vq) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(st->get_device(),
                                         {sycl::aspect::fp16});

            st->parallel_for<dpct_kernel_name<class attn_mma_kernel_470041,
                                              dpct_kernel_scalar<true>,
                                              dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(gm * sycl::range(1, 1, MMA_WARPS * WARP),
                                  sycl::range(1, 1, MMA_WARPS * WARP)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attn_mma_kernel<true, true>(a);
                    });
        } else if (kq) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(st->get_device(),
                                         {sycl::aspect::fp16});

            st->parallel_for<dpct_kernel_name<class attn_mma_kernel_9fba76,
                                              dpct_kernel_scalar<true>,
                                              dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(gm * sycl::range(1, 1, MMA_WARPS * WARP),
                                  sycl::range(1, 1, MMA_WARPS * WARP)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attn_mma_kernel<true, false>(a);
                    });
        } else if (vq) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(st->get_device(),
                                         {sycl::aspect::fp16});

            st->parallel_for<dpct_kernel_name<class attn_mma_kernel_e2c4a4,
                                              dpct_kernel_scalar<false>,
                                              dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(gm * sycl::range(1, 1, MMA_WARPS * WARP),
                                  sycl::range(1, 1, MMA_WARPS * WARP)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attn_mma_kernel<false, true>(a);
                    });
        } else {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(st->get_device(),
                                         {sycl::aspect::fp16});

            st->parallel_for<dpct_kernel_name<class attn_mma_kernel_2e2479,
                                              dpct_kernel_scalar<false>,
                                              dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(gm * sycl::range(1, 1, MMA_WARPS * WARP),
                                  sycl::range(1, 1, MMA_WARPS * WARP)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attn_mma_kernel<false, false>(a);
                    });
        }
        check_launch("inkling_attend");
        return;
    }
    const int tq = rows >= 8 ? 8 : rows >= 4 ? 4 : rows >= 2 ? 2 : 1;
    const dpct::dim3 g(1, kv_heads, (rows + tq - 1) / tq);
    switch (tq) {
        case 8: launch_attn_q<8>(a, g, st, kq, vq); break;
        case 4: launch_attn_q<4>(a, g, st, kq, vq); break;
        case 2: launch_attn_q<2>(a, g, st, kq, vq); break;
        default: launch_attn_q<1>(a, g, st, kq, vq); break;
    }
    check_launch("inkling_attend");
}

void inkling_attend_seqs(const float* q, const InklingAttnSeq* seqs, int rows, InklingRel rel, const float* sinks,
                         float* out, int dim, int n_head, int kv_heads, float scale, int window, int ring,
                         float* work, size_t work_floats, void* stream, bool int8_dot) {
    if (rows <= 0) return;
    if (rows > kInklingSeqRows || !work)
        throw std::invalid_argument("inkling_attend_seqs: 1 to " + std::to_string(kInklingSeqRows) +
                                    " rows, and work");
    AttnArgs a = attn_args("inkling_attend_seqs", q, rel, sinks, rows, out, dim, n_head, kv_heads, scale, window, ring,
                           work);
    a.nslot = 1;
    const bool kq = seqs[0].kq8.codes != nullptr, vq = seqs[0].vq8.codes != nullptr;
    AttnSeqs r{};
    for (int t = 0; t < rows; ++t) {
        const InklingAttnSeq& c = seqs[t];
        if ((c.kq8.codes != nullptr) != kq || (c.vq8.codes != nullptr) != vq)
            throw std::invalid_argument("inkling_attend_seqs: every row's caches of one kind");
        AttnRow& w = r.r[t];
        w.kc = (const sycl::half *)c.kcache,
        w.vc = (const sycl::half *)c.vcache, w.kq = c.kq8, w.vq = c.vq8;
        w.pos = (long long) c.pos;
        splits_of(c.room > 0 ? (long long) c.room : w.pos + 1, window, w.split, w.nslot);
        a.nslot = std::max(a.nslot, w.nslot);
    }
    if (size_t(rows) * n_head * a.nslot * (AT_DIM + 2) > work_floats)
        throw std::invalid_argument("inkling_attend_seqs: work too small");
    launch_dec(a, r, rows, kq, vq, int8_dot, cs(stream));
}

namespace {
template <bool ACC>
void conv_any(const float* x, float* state, const float* w, const float* taps, int heads, float* out, int channels,
              int rows, void* stream) {
    if (rows <= 0) return;
    if (taps && (heads <= 0 || channels % heads))
        throw std::invalid_argument("inkling_conv: channels must split evenly into the taps' heads");
    const float scale = conv_scale(taps);
    if (rows >= CONV_ROWS) {
        for (int r = 0; r < rows; r += 65535)   // a grid has at most 65535 rows
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<dpct_kernel_name<class conv_rows_kernel_8b7d71,
                                                dpct_kernel_scalar<ACC>>>(
                    sycl::nd_range<3>(sycl::range(1, std::min(65535, rows - r),
                                                  grid(channels, 256)) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        conv_rows_kernel<ACC>(x, state, w, taps, heads, out,
                                              channels, r, scale);
                    });
        }
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<
                    dpct_kernel_name<class conv_state_kernel_685591, dpct_kernel_scalar<ACC>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, grid(channels, 256)) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        conv_state_kernel(x, state, channels, rows);
                    });
        }
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class conv_walk_kernel_7c53ac,
                                            dpct_kernel_scalar<ACC>, ConvOne>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(channels, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    conv_walk_kernel<ACC>(x, ConvOne{state, rows}, w, taps,
                                          heads, out, channels, scale);
                });
    }
    check_launch("inkling_conv");
}
}  // namespace

void inkling_conv(const float* x, float* state, const float* w, const float* taps, int heads, float* out,
                  int channels, int rows, void* stream) {
    conv_any<false>(x, state, w, taps, heads, out, channels, rows, stream);
}

void inkling_conv_acc(const float* x, float* state, const float* w, const float* taps, int heads, float* acc,
                      int channels, int rows, void* stream) {
    conv_any<true>(x, state, w, taps, heads, acc, channels, rows, stream);
}

void inkling_conv_snaps(const float* x, const float* state, int channels, int rows, unsigned bits, float* snap,
                        size_t stride, void* stream) {
    Snaps snaps{};
    int n = 0;
    for (int t = 0; t < rows && t < SNAPS_MAX; ++t)
        if (bits >> t & 1u) snaps.t[n++] = t;
    if (n == 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class conv_snaps_kernel_8fec87>>(
                sycl::nd_range<3>(sycl::range(1, n, grid(channels, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    conv_snaps_kernel(x, state, channels, snaps, snap, stride);
                });
    }
    check_launch("inkling_conv_snaps");
}

namespace {
template <bool ACC>
void conv_seqs(const float* x, float* const* states, const float* w, float* out, int channels, int rows,
               void* stream) {
    if (rows <= 0) return;
    if (rows > kInklingSeqRows)
        throw std::invalid_argument("inkling_conv_seqs: at most " + std::to_string(kInklingSeqRows) + " rows");
    ConvSeqs s{};
    for (int t = 0; t < rows; ++t) s.state[t] = states[t];
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->submit([&](sycl::handler &cgh) {
            auto conv_scale_nullptr_ct7 = conv_scale(nullptr);

            cgh.parallel_for<
                dpct_kernel_name<class conv_walk_kernel_a95c6e,
                                 dpct_kernel_scalar<ACC>, ConvSeqs>>(
                sycl::nd_range<3>(sycl::range(1, rows, grid(channels, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    conv_walk_kernel<ACC>(x, s, w, nullptr, 0, out, channels,
                                          conv_scale_nullptr_ct7);
                });
        });
    }
    check_launch("inkling_conv_seqs");
}
}  // namespace

void inkling_conv_seqs(const float* x, float* const* states, const float* w, float* out, int channels, int rows,
                       void* stream) {
    conv_seqs<false>(x, states, w, out, channels, rows, stream);
}

void inkling_conv_acc_seqs(const float* x, float* const* states, const float* w, float* acc, int channels, int rows,
                           void* stream) {
    conv_seqs<true>(x, states, w, acc, channels, rows, stream);
}

void inkling_mix(const InklingMix& m, int n, int rows, float eps, void* stream) {
    if (rows <= 0) return;
    if (m.n_state < 1 || m.n_state > kInklingMaxStates || m.n_query < 1 || m.n_query > 4)
        throw std::invalid_argument("inkling_mix: 1 to 8 states and 1 to 4 queries");
    bool aligned = n % 4 == 0 && n / 4 <= MIX4_MAX && (uintptr_t(m.query) & 15) == 0;
    for (int k = 0; k < m.n_state; ++k) aligned = aligned && (uintptr_t(m.state[k]) & 15) == 0;
    for (int q = 0; q < m.n_query; ++q) aligned = aligned && (uintptr_t(m.out[q]) & 15) == 0;
    if (aligned)
        launch_mix4(m, n, rows, eps, cs(stream));
    else
    /*
    DPCT1049:162: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->parallel_for<dpct_kernel_name<class mix_kernel_1cd0bf>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) *
                                  sycl::range(1, 1, MIX_THREADS),
                              sycl::range(1, 1, MIX_THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mix_kernel(m, n, eps);
            });
    }
    check_launch("inkling_mix");
}

void inkling_control(void* blas, const float* cur, const float* msg, const float* down, const float* gain,
                     const float* forget, float* work, float* out, int n, int rank, int rows, float eps,
                     void* stream) {
    if (rows <= 0) return;
    float* z = work;
    float* r = z + size_t(rows) * 2 * n;
    float* g = r + size_t(rows) * rank;
    float* f = g + size_t(rows) * n;
    rms_norm(cur, nullptr, z, n, rows, eps, stream, 2 * n);
    rms_norm(msg, nullptr, z + n, n, rows, eps, stream, 2 * n);
    if (rows < 8 || !blas) {
        gemv_f32(down, z, r, 2 * n, rank, rank, rows, stream);
        gemv_f32(gain, r, g, rank, n, n, rows, stream);
        gemv_f32(forget, r, f, rank, n, n, rows, stream);
    } else {
        gemm_f32(blas, down, z, r, 2 * n, rank, rank, rows);
        gemm_f32(blas, gain, r, g, rank, n, n, rows);
        gemm_f32(blas, forget, r, f, rank, n, n, rows);
    }
    const size_t count = size_t(rows) * n;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class ctrl_mix_kernel_8ebc9a>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(count, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    ctrl_mix_kernel(cur, msg, g, f, out, count);
                });
    }
    check_launch("inkling_control");
}

void situ_rows(const float* gate_up, float* out, int n, int rows, float beta, float lb, void* stream,
               const ExpertScales* scales) {
    if (n <= 0 || rows <= 0) return;
    for (int r = 0; r < rows; r += 65535) {   // a grid has at most 65535 rows
        ExpertScales s = scales ? *scales : ExpertScales{};
        if (s.ids) s.ids += r;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)->submit([&](sycl::handler &cgh) {
                auto gate_up_size_t_r_n_ct0 = gate_up + size_t(r) * 2 * n;
                auto out_size_t_r_n_ct1 = out + size_t(r) * n;

                cgh.parallel_for<
                    dpct_kernel_name<class situ_rows_kernel_996515>>(
                    sycl::nd_range<3>(sycl::range(1, std::min(65535, rows - r),
                                                  grid(n, 256)) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        situ_rows_kernel(gate_up_size_t_r_n_ct0,
                                         out_size_t_r_n_ct1, n, beta, lb, s);
                    });
            });
        }
    }
    check_launch("situ_rows");
}

void inkling_control_fused(const InklingCtrl* ctrl, int count, int n, int rank, int rows, float eps, float* work,
                           void* stream) {
    if (rows <= 0 || count <= 0) return;
    if (count > 4 || rank <= 0 || rank > kInklingMaxRank || n % 4)
        throw std::invalid_argument("inkling_control_fused: 1 to 4 controllers, rank 1 to " +
                                    std::to_string(kInklingMaxRank) + ", n a multiple of 4");
    const dpct::queue_ptr st = cs(stream);
    for (int r0 = 0; r0 < rows; r0 += CTRL_ROWS) {
        CtrlArgs a{};
        a.n = n, a.rank = rank, a.rows = std::min(CTRL_ROWS, rows - r0), a.eps = eps, a.work = work;
        for (int k = 0; k < count; ++k) {
            const size_t off = size_t(r0) * n;
            a.cur[k] = ctrl[k].cur + off;
            a.msg[k] = ctrl[k].msg + off;
            a.out[k] = ctrl[k].out + off;
            a.down[k] = ctrl[k].down, a.gain[k] = ctrl[k].gain, a.forget[k] = ctrl[k].forget;
        }
        const dpct::dim3 g1(rank, count),
            g2((n + CTRL_OUT - 1) / CTRL_OUT, count);
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<class ctrl_down_kernel_a0ddc6>>(
                sycl::nd_range<3>(g1 * sycl::range(1, 1, CTRL_THREADS),
                                  sycl::range(1, 1, CTRL_THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        ctrl_down_kernel(a);
                    });
        }
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<class ctrl_up_kernel_d872eb>>(
                sycl::nd_range<3>(g2 * sycl::range(1, 1, CTRL_THREADS),
                                  sycl::range(1, 1, CTRL_THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        ctrl_up_kernel(a);
                    });
        }
        check_launch("inkling_control_fused");
    }
}

void inkling_gemv(const float* w, const float* x, float* y, int n_in, int n_out, int rows, void* stream) {
    if (rows <= 0 || n_out <= 0) return;
    if (n_in % 4) throw std::invalid_argument("inkling_gemv: n_in must be a multiple of 4");
    for (int r0 = 0; r0 < rows; r0 += CTRL_ROWS) {
        const int m = std::min(CTRL_ROWS, rows - r0);
        const float* xr = x + size_t(r0) * n_in;
        float* yr = y + size_t(r0) * n_out;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<dpct_kernel_name<class gemv_rows_kernel_607070>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 1, grid(n_out, GEMV_WARPS)) *
                            sycl::range(1, 1, GEMV_WARPS * WARP),
                        sycl::range(1, 1, GEMV_WARPS * WARP)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            gemv_rows_kernel(w, xr, yr, n_in, n_out, m);
                        });
        }
    }
    check_launch("inkling_gemv");
}

namespace {
template <class From>
void clip_many(const InklingClipJob* jobs, int count, int rows, const From& from, void* stream) {
    if (rows <= 0 || count <= 0) return;
    if (count > 4) throw std::invalid_argument("inkling_clip_many: at most 4 jobs");
    ClipJobs c{};
    c.first[0] = 0;
    for (int k = 0; k < count; ++k) {
        if (jobs[k].dim <= 0 || jobs[k].dim > 1024) throw std::invalid_argument("inkling_clip_many: dim 1 to 1024");
        c.j[k] = jobs[k];
        c.first[k + 1] = c.first[k] + rows * jobs[k].heads;
    }
    const int per_block = 256 / WARP;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<
                dpct_kernel_name<class clip_many_kernel_cba5b3, From>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, grid(c.first[count], per_block)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        clip_many_kernel<From>(c, count, from);
                    });
    }
    check_launch("inkling_clip_many");
}
}  // namespace

void inkling_clip_many(const InklingClipJob* jobs, int count, int rows, int64_t pos, void* stream) {
    clip_many(jobs, count, rows, ClipFrom{(long long) pos}, stream);
}

void inkling_clip_many_seqs(const InklingClipJob* jobs, int count, int rows, const int64_t* pos, void* stream) {
    if (rows > kInklingSeqRows)
        throw std::invalid_argument("inkling_clip_many_seqs: at most " + std::to_string(kInklingSeqRows) + " rows");
    ClipSeqs from{};
    for (int t = 0; t < rows; ++t) from.pos[t] = (long long) pos[t];
    clip_many(jobs, count, rows, from, stream);
}

void inkling_kv_store(const float* kraw, const float* vraw, float* k_state, float* v_state, const float* k_w,
                      const float* v_w, const float* k_taps, const float* v_taps, int kv_heads, int dim, int rows,
                      float k_clip, float eps, const InklingKvDst& dst, int64_t pos, float* k_out, float* v_out,
                      void* stream) {
    if (rows <= 0) return;
    if (dim != AT_DIM) throw std::invalid_argument("inkling_kv_store: heads of 128 values only");
    const int width = kv_heads * dim;
    if (rows > KV_ROWS) {   // a prompt batch: the separate calls
        if (!k_out || !v_out) throw std::invalid_argument("inkling_kv_store: more than 16 tokens need k_out and v_out");
        inkling_conv(kraw, k_state, k_w, k_taps, kv_heads, k_out, width, rows, stream);
        inkling_conv(vraw, v_state, v_w, v_taps, kv_heads, v_out, width, rows, stream);
        inkling_clip(k_out, 0, nullptr, k_out, dim, kv_heads, rows, k_clip, InklingTau{}, pos, stream);
        rms_norm(v_out, nullptr, v_out, dim, rows * kv_heads, eps, stream);
        for (int side = 0; side < 2; ++side) {
            const float* x = side ? v_out : k_out;
            int8_t* codes = side ? dst.vc : dst.kc;
            uint16_t* scales = side ? dst.vs : dst.ks;
            uint16_t* cache = side ? dst.v : dst.k;
            for (int done = 0; done < rows;) {
                const int64_t p = pos + done;
                const int64_t r = dst.ring > 0 ? p % dst.ring : p;
                const int m = dst.ring > 0 ? (int) std::min<int64_t>(rows - done, dst.ring - r) : rows - done;
                if (codes) store_q8(x + size_t(done) * width, codes, scales, width, r, m, stream);
                else store_f16(x + size_t(done) * width, cache, width, r, m, stream);
                done += m;
            }
        }
        check_launch("inkling_kv_store");
        return;
    }
    KvArgs a{};
    a.raw[0] = kraw, a.raw[1] = vraw;
    a.state[0] = k_state, a.state[1] = v_state;
    a.w[0] = k_w, a.w[1] = v_w;
    a.taps[0] = k_taps, a.taps[1] = v_taps;
    a.out[0] = k_out, a.out[1] = v_out;
    a.dst = dst;
    a.kv_heads = kv_heads, a.rows = rows, a.k_clip = k_clip, a.eps = eps;
    a.pos = (long long) pos;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(cs(stream)->get_device(),
                                     {sycl::aspect::fp16});

        cs(stream)
            ->parallel_for<
                dpct_kernel_name<class kv_store_kernel_1b9cb2, KvOne>>(
                sycl::nd_range<3>(sycl::range(1, 2, kv_heads) *
                                      sycl::range(1, 1, AT_DIM),
                                  sycl::range(1, 1, AT_DIM)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        kv_store_kernel(a, KvOne{});
                    });
    }
    check_launch("inkling_kv_store");
}

void inkling_kv_store_seqs(const float* kraw, const float* vraw, const InklingKvSeq* seqs, const float* k_w,
                           const float* v_w, const float* k_taps, const float* v_taps, int kv_heads, int dim,
                           int rows, float k_clip, float eps, float* k_out, float* v_out, void* stream) {
    if (rows <= 0) return;
    if (dim != AT_DIM) throw std::invalid_argument("inkling_kv_store_seqs: heads of 128 values only");
    if (rows > kInklingSeqRows)
        throw std::invalid_argument("inkling_kv_store_seqs: at most " + std::to_string(kInklingSeqRows) + " rows");
    KvArgs a{};
    a.raw[0] = kraw, a.raw[1] = vraw;
    a.w[0] = k_w, a.w[1] = v_w;
    a.taps[0] = k_taps, a.taps[1] = v_taps;
    a.out[0] = k_out, a.out[1] = v_out;
    a.kv_heads = kv_heads, a.rows = rows, a.k_clip = k_clip, a.eps = eps;
    KvSeqs r{};
    for (int t = 0; t < rows; ++t) r.row[t] = seqs[t];
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(cs(stream)->get_device(),
                                     {sycl::aspect::fp16});

        cs(stream)
            ->parallel_for<
                dpct_kernel_name<class kv_store_kernel_f4de1f,
                                 KvSeqs>>(
                sycl::nd_range<3>(sycl::range(rows, 2, kv_heads) *
                                      sycl::range(1, 1, AT_DIM),
                                  sycl::range(1, 1, AT_DIM)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        kv_store_kernel(a, r);
                    });
    }
    check_launch("inkling_kv_store_seqs");
}

}  // namespace strata::poly
