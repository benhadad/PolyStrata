// src/poly/kernels.cu - see the header.  The reference for every formula is llama.cpp's graph of the model and the
// ggml ops it builds.  Where a kernel takes several tokens, blockIdx.y is the token unless it says otherwise.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
// SYCL port: the PTX address conversion only feeds paths that are off here
#define __cvta_generic_to_shared(p) ((size_t)(p))
#include "strata/sycl_queue.hpp"
#include "strata/poly/kernels.hpp"
#include "norm_q8_1.hpp"
#include "strata/kernels/dp4a.hpp"
#include "norm_q8_1_device.hpp"
#include <dpct/blas_utils.hpp>

#ifdef STRATA_POLY_GDN_MMA
#include "gdn_mma.hpp"
#include "strata/kernels/gdn_mma_policy.hpp"
#endif

#if defined(STRATA_USE_HIP)
#include "hip_prefill_gemm.cuh"
#include "hip_split_attend.cuh"
#include "hip_value_split.cuh"
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <dpct/lib_common_utils.hpp>

namespace strata::poly {
namespace {

constexpr int WARP = 32;
constexpr int HC = 4;   // hyper-connection streams; the model's hc_mult, checked at load
constexpr int KEY_BLOCK = 16;   // cache rows one block of dsa_keys_kernel copies
// attn_score_kernel: a block's warps, a row and a group each; the values of a key a thread reads at a step; the
// queries a key's values are kept for; attn_score_reg_kernel's rows a warp
constexpr int SCORE_WARPS = 8, SCORE_VALUES = 4, SCORE_QUERIES = 4, SCORE_KEYS = 8;
// attend_batch(): the bytes of a chunk's results, which every block of rows reads and writes once more: in the
// card's L2 cache (32 MiB on the 4070 Laptop's AD106) that costs little, from its memory a lot
constexpr size_t kAttendResults = size_t(8) << 20;
// attn_value_kernel: a block's value dimensions, a sum's shares, a block's queries
constexpr int VALUE_DIMS = 32, VALUE_SHARES = 32, VALUE_QUERIES = 8;
constexpr int KDA_TERMS = 16, KDA_COLUMNS = 16;   // kda_state_kernel: a thread's values, a block's columns
// the indexer: a key's values; idx_score_kernel's keys a tile and tokens a block; idx_select_kernel's threads
constexpr int IDX_DIM = 128, IDX_TILE = 32, IDX_WARPS = 8, IDX_SELECT = 1024;

__dpct_inline__ float warp_sum(float x) {
#pragma unroll
    /*
    DPCT1108:20: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = WARP / 2; o > 0; o >>= 1) x +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            o);
    return x;
}

// One value per thread summed over the block.  blockDim.x is a multiple of 32, and `part` has a float a warp: 8
// where a block is 256 threads at most, 32 where it can be 1024.
__dpct_inline__ float block_sum(float x, float *part) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    x = warp_sum(x);
    if ((item_ct1.get_local_id(2) & 31) == 0)
        part[item_ct1.get_local_id(2) >> 5] = x;
    /*
    DPCT1065:181: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < int(item_ct1.get_local_range(2)) / WARP; ++k) s +=
        part[k];
    /*
    DPCT1065:182: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(); // part is free again
    return s;
}

// A value summed over each run of `lanes` threads of a warp; lanes is a power of two.
__dpct_inline__ float lanes_sum(float x, int lanes) {
    /*
DPCT1108:21: '__shfl_xor_sync' was migrated with the experimental feature masked
sub_group function which may not be supported by all compilers or runtimes. You
may need to adjust the code.
*/
#pragma unroll
    for (int o = lanes / 2; o > 0; o >>= 1) x +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            o);
    return x;
}

__dpct_inline__ float block_max(float x, float *part) {
auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
#pragma unroll
    /*
    DPCT1108:22: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = WARP / 2; o > 0; o >>= 1) x = sycl::fmax(
        x, dpct::experimental::permute_sub_group_by_xor(
               0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(),
               x, o));
    if ((item_ct1.get_local_id(2) & 31) == 0)
        part[item_ct1.get_local_id(2) >> 5] = x;
    /*
    DPCT1065:183: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float m = part[0];
#pragma unroll
    for (int k = 1; k < int(item_ct1.get_local_range(2)) / WARP; ++k)
        m = sycl::fmax(m, part[k]);
    /*
    DPCT1065:184: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return m;
}

__dpct_inline__ float sigmoid(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}

__dpct_inline__ float ldw(const float *w, size_t i) { return w[i]; }
__dpct_inline__ float ldw(const uint16_t *w, size_t i) {
    return sycl::bit_cast<float>(uint32_t(w[i]) << 16);
}

template <typename W>
__dpct_inline__ void
gemv_kernel(const W *__restrict__ w, const float *__restrict__ x,
            float *__restrict__ y, int n_in, int n_out, int rows_per_x) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int o = item_ct1.get_group(2);
    const size_t tok = item_ct1.get_group(1);
    const W* wr = w + size_t(o) * n_in;
    const float* xr = x + (tok * size_t(n_out / rows_per_x) + size_t(o / rows_per_x)) * n_in;
    float acc = 0.0f;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n_in;
         i += item_ct1.get_local_range(2)) acc += ldw(wr, i) * xr[i];
    acc = block_sum(acc, part);
    if (item_ct1.get_local_id(2) == 0) y[tok * n_out + o] = acc;
}

}  // namespace

// The threads of a block that sums over n values.  A thread reads its values one after the other, and a kernel
// this small takes as long as those reads in a row: four a thread at most, while a block has the threads.  With
// 256 threads for 4096 values, GLM 5.3 Flash's layer norms, small products and stream mixing were 2.7 of the 11
// ms a token its card worked before a layer's router had its scores.
int sum_threads(int n) {
    int threads = 128;
    while (threads < 1024 && threads * 4 < n) threads *= 2;
    return threads;
}

namespace {

template<typename W>
void gemv(const W* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream) {
    const int threads = n_in >= 128 ? sum_threads(n_in) : 32;
    /*
    DPCT1049:23: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class gemv_kernel_f83ae4, W>>(
                sycl::nd_range<3>(sycl::range(1, rows, n_out) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gemv_kernel<W>(w, x, y, n_in, n_out, rows_per_x);
                    });
    }
}

// A prompt batch's product with a matrix of few rows (a gate a head, the indexer's weights).  cuBLAS cuts such a
// product into a few blocks that each walk all of n_in: 0.44 ms for 1,024 tokens of 2,048 values and 32 outputs on
// an RTX 5090, a quarter of what Qwen3.6's recurrent layers took to read a prompt.  A block here has SMALL_TOK tokens
// and SMALL_OUT outputs, a warp a token and a lane an output, and takes the inputs SMALL_K values at a time.
constexpr int SMALL_TOK = 8, SMALL_OUT = 32, SMALL_K = 64;

/*
DPCT1110:24: The total declared local variable size in device function
gemm_small_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gemm_small_kernel(const float *__restrict__ w,
                                       const float *__restrict__ x,
                                       float *__restrict__ y, int n_in,
                                       int n_out, int rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int THREADS = SMALL_TOK * WARP,
                  XS = SMALL_TOK * SMALL_K / THREADS,
                  WS = SMALL_OUT * SMALL_K / THREADS;
    auto &sx = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[SMALL_TOK][SMALL_K + 1]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sw = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[SMALL_OUT][SMALL_K + 1]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t0 = item_ct1.get_group(2) * SMALL_TOK,
              o0 = item_ct1.get_group(1) * SMALL_OUT;
    const int tt = item_ct1.get_local_id(2) / WARP,
              oo = item_ct1.get_local_id(2) % WARP;
    // the next step's inputs and weights, read while this one is worked on
    float xr[XS], wr[WS];
    const auto fetch = [&](int k0) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        for (int j = 0; j < XS; ++j) {
            const int i = item_ct1.get_local_id(2) + j * THREADS,
                      t = t0 + i / SMALL_K, k = k0 + i % SMALL_K;
            xr[j] = t < rows && k < n_in ? x[size_t(t) * n_in + k] : 0.0f;
        }
        for (int j = 0; j < WS; ++j) {
            const int i = item_ct1.get_local_id(2) + j * THREADS,
                      o = o0 + i / SMALL_K, k = k0 + i % SMALL_K;
            wr[j] = o < n_out && k < n_in ? w[size_t(o) * n_in + k] : 0.0f;
        }
    };
    fetch(0);
    float acc = 0.0f;
    for (int k0 = 0; k0 < n_in; k0 += SMALL_K) {
        for (int j = 0; j < XS; ++j) {
            const int i = item_ct1.get_local_id(2) + j * THREADS;
            sx[i / SMALL_K][i % SMALL_K] = xr[j];
        }
        for (int j = 0; j < WS; ++j) {
            const int i = item_ct1.get_local_id(2) + j * THREADS;
            sw[i / SMALL_K][i % SMALL_K] = wr[j];
        }
        /*
        DPCT1118:25: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (k0 + SMALL_K < n_in) fetch(k0 + SMALL_K);
#pragma unroll 16
        for (int c = 0; c < SMALL_K; ++c)
            acc = sycl::fma(sx[tt][c], sw[oo][c], acc);
        /*
        DPCT1118:26: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    const int t = t0 + tt, o = o0 + oo;
    if (t < rows && o < n_out) y[size_t(t) * n_out + o] = acc;
}

// gemm_small_kernel for more outputs (a router's 128 or 256): a block TILE_TOK tokens and TILE_OUT outputs, a thread
// 4 of each, so a value read from shared memory serves 4 products where it served one.  For 2,048 tokens of 2,048
// values and 256 outputs gemm_small_kernel took 0.34 ms on an RTX 5090, 241 ms of the 3.7 s Flint read a prompt of
// 29,816 tokens in.  Each output is still one fmaf after the other in the inputs' order: the same bits.
constexpr int TILE_TOK = 32, TILE_OUT = 64, TILE_K = 32;

/*
DPCT1110:27: The total declared local variable size in device function
gemm_tile_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gemm_tile_kernel(const float *__restrict__ w,
                                      const float *__restrict__ x,
                                      float *__restrict__ y, int n_in,
                                      int n_out, int rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int THREADS = TILE_TOK * TILE_OUT / 16;
    // a step's inputs and weights by input, so a thread reads its 4 tokens' and 4 outputs' values as one float4
    auto &sx = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[TILE_K][TILE_TOK + 4]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sw = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[TILE_K][TILE_OUT + 4]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t0 = item_ct1.get_group(2) * TILE_TOK,
              o0 = item_ct1.get_group(1) * TILE_OUT;
    const int to = 4 * (item_ct1.get_local_id(2) / (TILE_OUT / 4)),
              oo = 4 * (item_ct1.get_local_id(2) % (TILE_OUT / 4));
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < n_in; k0 += TILE_K) {
        for (int i = item_ct1.get_local_id(2); i < TILE_TOK * TILE_K;
             i += THREADS) {
            const int t = t0 + i / TILE_K, k = k0 + i % TILE_K;
            sx[i % TILE_K][i / TILE_K] = t < rows && k < n_in ? x[size_t(t) * n_in + k] : 0.0f;
        }
        for (int i = item_ct1.get_local_id(2); i < TILE_OUT * TILE_K;
             i += THREADS) {
            const int o = o0 + i / TILE_K, k = k0 + i % TILE_K;
            sw[i % TILE_K][i / TILE_K] = o < n_out && k < n_in ? w[size_t(o) * n_in + k] : 0.0f;
        }
        /*
        DPCT1118:28: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll 8
        for (int c = 0; c < TILE_K; ++c) {
            const sycl::float4 a =
                *reinterpret_cast<const sycl::float4 *>(&sx[c][to]);
            const sycl::float4 b =
                *reinterpret_cast<const sycl::float4 *>(&sw[c][oo]);
            const float av[4] = {a.x(), a.y(), a.z(), a.w()},
                        bv[4] = {b.x(), b.y(), b.z(), b.w()};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] =
                    sycl::fma((float)(av[i]), (float)(bv[j]), acc[i][j]);
        }
        /*
        DPCT1118:29: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = t0 + to + i, o = o0 + oo + j;
            if (t < rows && o < n_out) y[size_t(t) * n_out + o] = acc[i][j];
        }
}

// Where a row's Q8_1 blocks would keep a sum past the halves' range: a power of two to divide the row by, 1 for
// most.  The test is on runs of 16 (Q2_K's blocks keep sums of 16, the others of 32) against a quarter of 65504,
// so a block of 32 stays under half of it, and on the largest value, whose block delta is a half too.
__dpct_inline__ void guard_half_sums_kernel(float *x, size_t ld, int width,
                                            float *__restrict__ factor) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    float *row = x + size_t(item_ct1.get_group(2)) * ld;
    float most = 0.0f;
    for (int g = item_ct1.get_local_id(2); g < width / 16;
         g += item_ct1.get_local_range(2)) {
        float s = 0.0f, big = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float v = row[g * 16 + j];
            s += v;
            big = sycl::fmax(big, sycl::fabs(v));
        }
        most = sycl::fmax(most, sycl::fmax(2.0f * sycl::fabs(s), big / 64.0f));
    }
    most = block_max(most, part);
    const int k =
        most > 32768.0f ? (int)sycl::ceil(sycl::log2(most / 32768.0f)) : 0;
    if (item_ct1.get_local_id(2) == 0) factor[item_ct1.get_group(2)] =
        sycl::ldexp(1.0f, k);
    if (k == 0) return;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < width;
         i += item_ct1.get_local_range(2)) row[i] = sycl::ldexp(row[i], -k);
}

// Rows of y (row r at place[r] when place is given) times their factor, or divided by it.
__dpct_inline__ void scale_rows_kernel(float *y, size_t ld, int width,
                                       const int32_t *__restrict__ place,
                                       const float *__restrict__ factor,
                                       bool divide) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const float f = factor[item_ct1.get_group(1)];
    if (f == 1.0f) return;
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= width) return;
    float *row = y + size_t(place ? place[item_ct1.get_group(1)]
                                  : item_ct1.get_group(1)) *
                         ld;
    row[i] = divide ? row[i] / f : row[i] * f;
}

__dpct_inline__ void bf16_to_f32_kernel(const uint16_t *__restrict__ w,
                                        float *__restrict__ out, size_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        size_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) out[i] = ldw(w, i);
}

__dpct_inline__ void rms_norm_kernel(const float *__restrict__ x,
                                     const float *__restrict__ w, float *y,
                                     int n, float eps, int y_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const float *xr = x + size_t(item_ct1.get_group(2)) * n;
    float *yr = y + size_t(item_ct1.get_group(2)) * y_stride;
    float ss = 0.0f;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float scale = 1.0f / sycl::sqrt(ss / float(n) + eps);
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) yr[i] = w ? xr[i] * scale * w[i]
                                                     : xr[i] * scale;
}

// Weighted RMSNorm and Q8_1 quantization of one row per work-group.
__dpct_inline__ void rms_norm_q8_1_kernel(const float* x, const float* w, float* y,
                                         uint8_t* q8, int n, float eps) {
    const auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const auto row = item.get_group(2);
    norm_q8_1_detail::write_row(x + row * n, w, y + row * n, q8 + row * (n / 32) * 36, n, eps, item);
}

__dpct_inline__ void layer_norm_kernel(const float *__restrict__ x,
                                       const float *__restrict__ w,
                                       const float *__restrict__ b, float *y,
                                       int n, float eps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const float *xr = x + size_t(item_ct1.get_group(2)) * n;
    float *yr = y + size_t(item_ct1.get_group(2)) * n;
    float s = 0.0f;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) s += xr[i];
    const float mean = block_sum(s, part) / float(n);
    float v = 0.0f;
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    const float scale = 1.0f / sycl::sqrt(block_sum(v, part) / float(n) + eps);
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2))
        yr[i] = (xr[i] - mean) * scale * w[i] + b[i];
}

// mix[o] = fn row o . r for o < n_mix; mix[n_mix] = sum of r^2 (the flat RMS norm's input)
__dpct_inline__ void hc_mix_kernel(const float *__restrict__ r,
                                   const float *__restrict__ fn,
                                   float *__restrict__ mix, int n, int n_mix) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int o = item_ct1.get_group(2);
    r += size_t(item_ct1.get_group(1)) * n;
    mix += size_t(item_ct1.get_group(1)) * (n_mix + 1);
    float acc = 0.0f;
    if (o < n_mix) {
        const float* fr = fn + size_t(o) * n;
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < n;
             i += item_ct1.get_local_range(2)) acc += fr[i] * r[i];
    } else {
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < n;
             i += item_ct1.get_local_range(2)) acc += r[i] * r[i];
    }
    acc = block_sum(acc, part);
    if (item_ct1.get_local_id(2) == 0) mix[o] = acc;
}

inline void comb_norm_cols(float *comb, float eps) {
    for (int dst = 0; dst < HC; ++dst) {
        float sum = eps;
#pragma unroll
        for (int src = 0; src < HC; ++src) sum += comb[dst + HC * src];
        const float inv = 1.0f / sum;
#pragma unroll
        for (int src = 0; src < HC; ++src) comb[dst + HC * src] *= inv;
    }
}

inline void comb_norm_rows(float *comb, float eps) {
    for (int src = 0; src < HC; ++src) {
        float sum = eps;
#pragma unroll
        for (int dst = 0; dst < HC; ++dst) sum += comb[dst + HC * src];
        const float inv = 1.0f / sum;
#pragma unroll
        for (int dst = 0; dst < HC; ++dst) comb[dst + HC * src] *= inv;
    }
}

/*
DPCT1110:30: The total declared local variable size in device function
hc_read_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
hc_read_kernel(const float *__restrict__ r, const float *__restrict__ mix,
               const float *__restrict__ scale, const float *__restrict__ base,
               float *__restrict__ hcw, float *__restrict__ cur, int n_embd,
               float norm_eps, float hc_eps, int iters) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &pre = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[HC]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    r += size_t(item_ct1.get_group(1)) * HC * n_embd;
    mix += size_t(item_ct1.get_group(1)) * (HC * (2 + HC) + 1);
    hcw += size_t(item_ct1.get_group(1)) * (HC + HC * HC);
    cur += size_t(item_ct1.get_group(1)) * n_embd;
    if (item_ct1.get_local_id(2) == 0) {
        const float rs =
            1.0f /
            sycl::sqrt(mix[HC * (2 + HC)] / float(HC * n_embd) + norm_eps);
#pragma unroll
        for (int h = 0; h < HC; ++h)
            pre[h] = sigmoid(mix[h] * rs * scale[0] + base[h]) + hc_eps;
        if (item_ct1.get_group(2) == 0) {
#pragma unroll
            for (int h = 0; h < HC; ++h) hcw[h] =
                2.0f * sigmoid(mix[HC + h] * rs * scale[1] + base[HC + h]);
            float comb[HC * HC];
            for (int src = 0; src < HC; ++src) {   // softmax over dst
                float mx = -INFINITY;
                for (int dst = 0; dst < HC; ++dst) {
                    const int k = dst + HC * src;
                    comb[k] = mix[2 * HC + k] * rs * scale[2] + base[2 * HC + k];
                    mx = sycl::fmax(mx, comb[k]);
                }
                float sum = 0.0f;
                for (int dst = 0; dst < HC; ++dst) {
                    const int k = dst + HC * src;
                    comb[k] = sycl::native::exp(comb[k] - mx);
                    sum += comb[k];
                }
                const float inv = 1.0f / sum;
#pragma unroll
                for (int dst = 0; dst < HC; ++dst)
                    comb[dst + HC * src] = comb[dst + HC * src] * inv + hc_eps;
            }
            comb_norm_cols(comb, hc_eps);
#pragma unroll
            for (int it = 1; it < iters; ++it) {
                comb_norm_rows(comb, hc_eps);
                comb_norm_cols(comb, hc_eps);
            }
#pragma unroll
            for (int k = 0; k < HC * HC; ++k) hcw[HC + k] = comb[k];
        }
    }
    /*
    DPCT1113:185: Consider replacing
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
    sycl::nd_item::barrier() if function "hc_read_kernel" is called in a
    multidimensional kernel.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n_embd) return;
    float s = 0.0f;
#pragma unroll
    for (int h = 0; h < HC; ++h) s += r[size_t(h) * n_embd + i] * pre[h];
    cur[i] = s;
}

__dpct_inline__ void hc_post_kernel(const float *__restrict__ x,
                                    const float *__restrict__ hcw, float *r,
                                    int n_embd) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n_embd) return;
    x += size_t(item_ct1.get_group(1)) * n_embd;
    hcw += size_t(item_ct1.get_group(1)) * (HC + HC * HC);
    r += size_t(item_ct1.get_group(1)) * HC * n_embd;
    float in[HC], out[HC];
#pragma unroll
    for (int s = 0; s < HC; ++s) in[s] = r[size_t(s) * n_embd + i];
    const float xi = x[i];
    for (int dst = 0; dst < HC; ++dst) {
        float v = xi * hcw[dst];
#pragma unroll
        for (int src = 0; src < HC; ++src) v +=
            in[src] * hcw[HC + dst + HC * src];
        out[dst] = v;
    }
#pragma unroll
    for (int s = 0; s < HC; ++s) r[size_t(s) * n_embd + i] = out[s];
}

// DeepSeek V4's output read: hc weights from mix (hc_mix_kernel's, n_mix = HC), then their sum of the streams.
__dpct_inline__ void hc_head_kernel(const float *__restrict__ r,
                                    const float *__restrict__ mix,
                                    const float *__restrict__ scale,
                                    const float *__restrict__ base,
                                    float *__restrict__ y, int n_embd,
                                    float norm_eps, float hc_eps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &pre = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[HC]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    r += size_t(item_ct1.get_group(1)) * HC * n_embd;
    mix += size_t(item_ct1.get_group(1)) * (HC + 1);
    if (item_ct1.get_local_id(2) == 0) {
        const float rs =
            1.0f / sycl::sqrt(mix[HC] / float(HC * n_embd) + norm_eps);
#pragma unroll
        for (int h = 0; h < HC; ++h)
            pre[h] = sigmoid(mix[h] * rs * scale[0] + base[h]) + hc_eps;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n_embd) return;
    float s = 0.0f;
#pragma unroll
    for (int h = 0; h < HC; ++h) s += r[size_t(h) * n_embd + i] * pre[h];
    y[size_t(item_ct1.get_group(1)) * n_embd + i] = s;
}

__dpct_inline__ void hc_mean_kernel(const float *__restrict__ r,
                                    float *__restrict__ y, int n_embd) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n_embd) return;
    r += size_t(item_ct1.get_group(1)) * HC * n_embd;
    float acc = r[i];
#pragma unroll
    for (int s = 1; s < HC; ++s) acc += r[size_t(s) * n_embd + i];
    y[size_t(item_ct1.get_group(1)) * n_embd + i] = acc * (1.0f / HC);
}

// d_conv is 4: three past inputs per channel, then the new one.  A thread is a channel and walks the tokens.
// The state as it is after each of the first n_snap tokens goes to snap, a state `stride` floats after the other.
__dpct_inline__ void
kda_conv_kernel(const float *__restrict__ proj, float *__restrict__ state,
                const float *__restrict__ wq, const float *__restrict__ wk,
                const float *__restrict__ wv, float *__restrict__ out,
                int d_inner, int rows, float *__restrict__ snap, size_t stride,
                int n_snap) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (t >= 3 * d_inner) return;
    const int m = t / d_inner, c = t % d_inner;
    const float* w = (m == 0 ? wq : m == 1 ? wk : wv) + 4 * size_t(c);
    float* st = state + size_t(t) * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    const size_t first = size_t(m) * rows * d_inner + c;
    for (int row = 0; row < rows; ++row) {
        const size_t at = first + size_t(row) * d_inner;
        const float x = proj[at];
        const float y = s0 * w[0] + s1 * w[1] + s2 * w[2] + x * w[3];
        s0 = s1;
        s1 = s2;
        s2 = x;
        out[at] = y * sigmoid(y);
        if (row < n_snap) {
            float* m = snap + size_t(row) * stride + size_t(t) * 3;
            m[0] = s0;
            m[1] = s1;
            m[2] = s2;
        }
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
}

// kda_conv_kernel for a call that leaves no snapshots: a thread a channel of a token, the inputs before the call's
// first token from the state, the same sum in the same order (conv4_silu_rows_kernel's reason).
__dpct_inline__ void kda_conv_rows_kernel(const float *__restrict__ proj,
                                          const float *__restrict__ state,
                                          const float *__restrict__ wq,
                                          const float *__restrict__ wk,
                                          const float *__restrict__ wv,
                                          float *__restrict__ out, int d_inner,
                                          int rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (t >= 3 * d_inner) return;
    const int m = t / d_inner, c = t % d_inner, row = item_ct1.get_group(1);
    const float* w = (m == 0 ? wq : m == 1 ? wk : wv) + 4 * size_t(c);
    const float* st = state + size_t(t) * 3;
    const size_t first = size_t(m) * rows * d_inner + c;
    auto in = [&](int r) { return r >= 0 ? proj[first + size_t(r) * d_inner] : st[3 + r]; };
    const float s0 = in(row - 3), s1 = in(row - 2), s2 = in(row - 1), x = proj[first + size_t(row) * d_inner];
    const float y = s0 * w[0] + s1 * w[1] + s2 * w[2] + x * w[3];
    out[first + size_t(row) * d_inner] = y * sigmoid(y);
}

// The state after the call: its last three inputs, the state's own where the call has fewer.
__dpct_inline__ void kda_conv_state_kernel(const float *__restrict__ proj,
                                           float *__restrict__ state,
                                           int d_inner, int rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (t >= 3 * d_inner) return;
    const int m = t / d_inner, c = t % d_inner;
    float* st = state + size_t(t) * 3;
    const size_t first = size_t(m) * rows * d_inner + c;
    auto in = [&](int r) { return r >= 0 ? proj[first + size_t(r) * d_inner] : st[3 + r]; };
    const float a = in(rows - 3), b = in(rows - 2), d = in(rows - 1);
    st[0] = a;
    st[1] = b;
    st[2] = d;
}

// One block per head, and a column of the state to every `lanes` threads of it: a thread holds every lanes-th
// value of its column, so a column's two sums are a few terms a thread and a sum over its threads.  A thread a
// column read the column's 128 values one after the other, twice, and that was 25 microseconds a token.  The
// block walks the tokens.  s[i + hd * j] holds key dimension i of column j.  blockDim.x is hd * lanes, up to 1024,
// so the bound: without it sm_70 gave the kernel 72 registers a thread and its launch failed.
__dpct_inline__ void
kda_step_kernel(const float *__restrict__ qkv, const float *__restrict__ g1,
                const float *__restrict__ a, const float *__restrict__ beta,
                float *__restrict__ s, const float *__restrict__ g2,
                const float *__restrict__ o_norm, float *__restrict__ out,
                int hd, int n_head, int rows, float lower_bound, float norm_eps,
                float *__restrict__ snap, size_t stride, int n_snap) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &qn = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &kn = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &decay =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int lanes = int(item_ct1.get_local_range(2)) / hd;
    const int h = item_ct1.get_group(2), j = item_ct1.get_local_id(2) / lanes,
              lane = item_ct1.get_local_id(2) % lanes;
    const bool first = lane == 0;   // the thread that stands for its column in the sums over the head
    const size_t d_inner = size_t(hd) * n_head, plane = d_inner * rows;
    float* col = s + (size_t(h) * hd + j) * hd;
    for (int row = 0; row < rows; ++row) {
        const size_t at = size_t(row) * d_inner + size_t(h) * hd + j;
        const float q = qkv[at], k = qkv[plane + at], v = qkv[2 * plane + at];
        const float qs = block_sum(first ? q * q : 0.0f, part);
        const float ks = block_sum(first ? k * k : 0.0f, part);
        if (first) {
            qn[j] = q / sycl::sqrt(qs + 1e-6f) * (1.0f / sycl::sqrt(float(hd)));
            kn[j] = k / sycl::sqrt(ks + 1e-6f);
            // the decay gate of key dimension j: exp(lower_bound * sigmoid(-(g1 * a)))
            decay[j] = sycl::native::exp(
                lower_bound / (1.0f + sycl::native::exp(g1[at] * a[h])));
        }
        /*
        DPCT1118:31: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:186: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        // Which product is fused into which sum is spelled out, here and in kda_state_kernel: left to the
        // compiler, the two came out with other last bits.
        float sk = 0.0f;
        /*
DPCT1013:187: The rounding mode could not be specified and the generated code
may have different accuracy than the original code. Verify the correctness. SYCL
math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
*/
#pragma unroll
        for (int i = lane; i < hd; i += lanes)
            sk = sycl::fma(col[i] * decay[i], kn[i], sk);
        sk = lanes_sum(sk, lanes);
        const float d = (v - sk) * sigmoid(beta[size_t(row) * n_head + h]);
        float o = 0.0f;
        float* m = row < n_snap ? snap + size_t(row) * stride + (size_t(h) * hd + j) * hd : nullptr;
        for (int i = lane; i < hd; i += lanes) {
            /*
            DPCT1013:188: The rounding mode could not be specified and the
            generated code may have different accuracy than the original code.
            Verify the correctness. SYCL math built-in function rounding mode is
            aligned with OpenCL C 1.2 standard.
            */
            const float n = sycl::fma(col[i], decay[i], kn[i] * d);
            col[i] = n;
            if (m) m[i] = n;
            o = sycl::fma((float)n, qn[i], o);
        }
        o = lanes_sum(o, lanes);
        // the sum's last barrier also ends this token's reads of qn, kn and decay
        const float os = block_sum(first ? o * o : 0.0f, part);
        if (first) out[at] = o / sycl::sqrt(os / float(hd) + norm_eps) *
                             o_norm[j] * sigmoid(g2[at]);
    }
}

// kda_step_kernel in three for a prompt batch.  There a head's block walks the tokens with every column's values
// in the card's memory, and waits for its warps three times a token, for sums that need no token before them.
// Here the norms of q and k come first for every token at once, the state's columns walk the tokens in registers,
// a few columns a block so that the blocks spread over the card, and the output's norm comes last.  The sums add
// in kda_step_kernel's order, so a token comes out with the same bits.

// The sum over a head's values that kda_step_kernel's block_sum makes of them, a thread a value here.  There they
// lie `lanes` threads apart, so a warp sums WARP / lanes of them (`group`) and the block adds the warps' sums in
// turn.  part has a float a group.
__dpct_inline__ float kda_head_sum(float x, float *part, int group) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    x = lanes_sum(x, group);
    if (item_ct1.get_local_id(2) % group == 0)
        part[item_ct1.get_local_id(2) / group] = x;
    /*
    DPCT1065:189: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < int(item_ct1.get_local_range(2)) / group; ++k) s +=
        part[k];
    /*
    DPCT1065:190: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(); // part is free again
    return s;
}

// One block a token and head, a thread a value: q and k normed in place, and g1 made the decay of its key
// dimension.
__dpct_inline__ void kda_gates_kernel(float *__restrict__ qkv,
                                      float *__restrict__ g1,
                                      const float *__restrict__ a, int n_head,
                                      int rows, float lower_bound, int group) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int hd = item_ct1.get_local_range(2);
    const size_t at = size_t(item_ct1.get_group(2)) * hd +
                      item_ct1.get_local_id(2),
                 plane = size_t(hd) * n_head * rows;
    const float q = qkv[at], k = qkv[plane + at];
    // the squares are rounded before they are added, as there
    /*
    DPCT1013:191: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    const float qs = kda_head_sum(q * q, part, group);
    /*
    DPCT1013:192: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    const float ks = kda_head_sum(k * k, part, group);
    qkv[at] = q / sycl::sqrt(qs + 1e-6f) * (1.0f / sycl::sqrt(float(hd)));
    qkv[plane + at] = k / sycl::sqrt(ks + 1e-6f);
    /*
    DPCT1064:193: Migrated expf call is used in a macro/template definition and
    may not be valid for all macro/template uses. Adjust the code.
    */
    g1[at] = sycl::native::exp(
        lower_bound /
        (1.0f + sycl::native::exp(g1[at] * a[item_ct1.get_group(2) % n_head])));
}

// The state's update with what kda_gates_kernel left.  A column has `lanes` threads as in kda_step_kernel, each
// with KDA_TERMS of its values, and a block has KDA_COLUMNS of a head's columns.  out gets the output before its
// norm.
/*
DPCT1110:32: The total declared local variable size in device function
kda_state_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
kda_state_kernel(const float *__restrict__ qkv, const float *__restrict__ decay,
                 const float *__restrict__ beta, float *__restrict__ s,
                 float *__restrict__ out, int n_head, int rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lanes = int(item_ct1.get_local_range(2)) / KDA_COLUMNS,
              hd = lanes * KDA_TERMS;
    const int per_head = hd / KDA_COLUMNS;   // blocks a head
    const int h = item_ct1.get_group(2) / per_head,
              j = (item_ct1.get_group(2) % per_head) * KDA_COLUMNS +
                  item_ct1.get_local_id(2) / lanes;
    const int lane = item_ct1.get_local_id(2) % lanes;
    const size_t d_inner = size_t(hd) * n_head, plane = d_inner * rows;
    float* col = s + (size_t(h) * hd + j) * hd + lane;
    float c[KDA_TERMS];
#pragma unroll
    for (int t = 0; t < KDA_TERMS; ++t) c[t] = col[t * lanes];
    for (int row = 0; row < rows; ++row) {
        const size_t head = size_t(row) * d_inner + size_t(h) * hd;
        const float* qn = qkv + head + lane;
        const float* kn = qkv + plane + head + lane;
        const float* dc = decay + head + lane;
        const float v = qkv[2 * plane + head + j];
        float sk = 0.0f;   // kda_step_kernel's fused products
#pragma unroll
        /*
        DPCT1013:194: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        for (int t = 0; t < KDA_TERMS; ++t)
            sk = sycl::fma(c[t] * dc[t * lanes], (float)(kn[t * lanes]), sk);
        sk = lanes_sum(sk, lanes);
        const float d = (v - sk) * sigmoid(beta[size_t(row) * n_head + h]);
        float o = 0.0f;
#pragma unroll
        for (int t = 0; t < KDA_TERMS; ++t) {
            /*
            DPCT1013:195: The rounding mode could not be specified and the
            generated code may have different accuracy than the original code.
            Verify the correctness. SYCL math built-in function rounding mode is
            aligned with OpenCL C 1.2 standard.
            */
            const float n =
                sycl::fma(c[t], (float)(dc[t * lanes]), kn[t * lanes] * d);
            c[t] = n;
            o = sycl::fma((float)n, (float)(qn[t * lanes]), o);
        }
        o = lanes_sum(o, lanes);
        if (lane == 0) out[head + j] = o;
    }
#pragma unroll
    for (int t = 0; t < KDA_TERMS; ++t) col[t * lanes] = c[t];
}

// One block a token and head, a thread a value: the output's norm, its weights and the gate.
__dpct_inline__ void kda_out_kernel(float *__restrict__ out,
                                    const float *__restrict__ g2,
                                    const float *__restrict__ o_norm,
                                    float norm_eps, int group) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int hd = item_ct1.get_local_range(2);
    const size_t at =
        size_t(item_ct1.get_group(2)) * hd + item_ct1.get_local_id(2);
    const float o = out[at];
    /*
    DPCT1013:196: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    const float os = kda_head_sum(o * o, part, group);
    out[at] = o / sycl::sqrt(os / float(hd) + norm_eps) *
              o_norm[item_ct1.get_local_id(2)] * sigmoid(g2[at]);
}

__dpct_inline__ void store_f16_kernel(const float *__restrict__ x,
                                      sycl::half *__restrict__ dst, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) dst[i] =
        sycl::vec<float, 1>(x[i])
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

// Rows from the position the card holds when the launch runs.
__dpct_inline__ void store_f16_at_kernel(const float *__restrict__ x,
                                         sycl::half *__restrict__ cache, int n,
                                         int rows,
                                         const long long *__restrict__ at,
                                         int ring) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n * rows) return;
    const long long row = ring > 0 ? (*at + i / n) % ring : *at + i / n;
    cache[size_t(row) * n + i % n] =
        sycl::vec<float, 1>(x[i])
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

// Q8_0: d = amax/127 a 32-value block; codes = clamp(rint(x/d), -127, 127); scale stored fp16.
__dpct_inline__ void store_q8_kernel(const float *__restrict__ x,
                                     int8_t *__restrict__ codes,
                                     sycl::half *__restrict__ scales, int n,
                                     int rows, int nb) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int gb = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                   item_ct1.get_local_id(2);
    if (gb >= nb * rows) return;
    const int row = gb / nb, blk = gb % nb;
    const float* xr = x + (size_t) row * n + blk * 32;
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) amax = sycl::fmax(amax, sycl::fabs(xr[j]));
    const float d = amax / 127.0f;
    const float id = d > 0.f ? 1.0f / d : 0.0f;
    scales[(size_t)row * nb + blk] =
        sycl::vec<float, 1>(d)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    int8_t* cr = codes + (size_t) row * n + blk * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        float q = sycl::rint(xr[j] * id);
        q = sycl::fmin(sycl::fmax(q, -127.0f), 127.0f);
        cr[j] = (int8_t) q;
    }
}

// Rows from the position the card holds when the launch runs.
__dpct_inline__ void
store_q8_at_kernel(const float *__restrict__ x, int8_t *__restrict__ codes,
                   sycl::half *__restrict__ scales, int n, int rows, int nb,
                   const long long *__restrict__ at, int ring) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int gb = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                   item_ct1.get_local_id(2);
    if (gb >= nb * rows) return;
    const int trow = gb / nb, blk = gb % nb;
    const long long row = ring > 0 ? (*at + trow) % ring : *at + trow;
    const float* xr = x + (size_t) trow * n + blk * 32;
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) amax = sycl::fmax(amax, sycl::fabs(xr[j]));
    const float d = amax / 127.0f;
    const float id = d > 0.f ? 1.0f / d : 0.0f;
    scales[(size_t)row * nb + blk] =
        sycl::vec<float, 1>(d)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    int8_t* cr = codes + (size_t) row * n + blk * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        float q = sycl::rint(xr[j] * id);
        q = sycl::fmin(sycl::fmax(q, -127.0f), 127.0f);
        cr[j] = (int8_t) q;
    }
}

// How many cache rows token `tok` of a call from position pos attends to: every row up to its own while it is
// dense, then its list.
inline int dsa_keys(int tok, long long pos, int dense_rows, int top_rows, int kpool) {
    const long long n = pos + tok + 1;
    return tok >= dense_rows ? top_rows + int(n % kpool) : int(n);
}

// The keys of token first + blockIdx.y as float rows, n_keys of them; the rows past its last key are zero.  A
// block copies KEY_BLOCK rows.
__dpct_inline__ void dsa_keys_kernel(const sycl::half *__restrict__ cache,
                                     const int32_t *__restrict__ sel_all,
                                     int sel_stride, int top_rows, int kpool,
                                     long long pos, int first, int dense_rows,
                                     float *__restrict__ keys, int rank,
                                     int n_keys) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tok = first + item_ct1.get_group(1);
    const int n_sel = dsa_keys(tok, pos, dense_rows, top_rows, kpool);
    const int32_t* sel = tok >= dense_rows ? sel_all + size_t(tok - dense_rows) * sel_stride : nullptr;
    float *dst = keys + size_t(item_ct1.get_group(1)) * n_keys * rank;
    const int j0 = item_ct1.get_group(2) * KEY_BLOCK;
    for (int i = item_ct1.get_local_id(2); i < KEY_BLOCK * rank;
         i += item_ct1.get_local_range(2)) {
        const int j = j0 + i / rank, r = i % rank;
        if (j >= n_keys) return;
        dst[size_t(j) * rank + r] =
            j < n_sel ? sycl::vec<sycl::half, 1>(
                            cache[size_t(sel ? sel[j] : j) * rank + r])
                            .convert<float, sycl::rounding_mode::automatic>()[0]
                      : 0.0f;
    }
}

// Softmax in place over a head's scores of its token's keys; the entries past the last key become zero.  One
// block per head and token.
__dpct_inline__ void dsa_softmax_kernel(float *score, int n_keys, int top_rows,
                                        int kpool, long long pos, int first,
                                        int dense_rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int n_sel = dsa_keys(first + item_ct1.get_group(1), pos, dense_rows,
                               top_rows, kpool);
    float *row =
        score + (size_t(item_ct1.get_group(1)) * item_ct1.get_group_range(2) +
                 item_ct1.get_group(2)) *
                    n_keys;
    float mx = -INFINITY;
#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < n_sel;
         j += item_ct1.get_local_range(2)) mx = sycl::fmax(mx, row[j]);
    mx = block_max(mx, part);
    float sum = 0.0f;
#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < n_sel;
         j += item_ct1.get_local_range(2)) {
        row[j] = sycl::native::exp(row[j] - mx);
        sum += row[j];
    }
    const float inv = 1.0f / block_sum(sum, part);
#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < n_keys;
         j += item_ct1.get_local_range(2)) row[j] = j < n_sel ? row[j] * inv
                                                              : 0.0f;
}

// One block per token from position pos.
__dpct_inline__ void idx_store_kernel(const float *__restrict__ key,
                                      const float *__restrict__ gate,
                                      float *__restrict__ ring, int ring_rows,
                                      int dim, long long pos) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int d = item_ct1.get_local_id(2);
    if (d >= dim) return;
    float *row =
        ring + size_t((pos + item_ct1.get_group(2)) % ring_rows) * 2 * dim;
    row[d] = key[size_t(item_ct1.get_group(2)) * dim + d];
    row[dim + d] = gate[size_t(item_ct1.get_group(2)) * dim + d];
}

// One block per token from position pos; the ones that do not end a pool have nothing to do.
__dpct_inline__ void idx_pool_kernel(const float *__restrict__ ring,
                                     int ring_rows, float *__restrict__ pool,
                                     const float *__restrict__ ape, int dim,
                                     int kpool, long long pos) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int d = item_ct1.get_local_id(2);
    const long long last = pos + item_ct1.get_group(2);
    if (d >= dim || (last + 1) % kpool != 0) return;
    const auto row = [&](int k) { return ring + size_t((last - kpool + 1 + k) % ring_rows) * 2 * dim; };
    float mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < kpool; ++k)
        mx = sycl::fmax(mx, row(k)[dim + d] + ape[d + dim * k]);
    float sum = 0.0f, acc = 0.0f;
    for (int k = 0; k < kpool; ++k) {
        const float e =
            sycl::native::exp(row(k)[dim + d] + ape[d + dim * k] - mx);
        sum += e;
        acc += e * row(k)[d];
    }
    pool[size_t(last / kpool) * dim + d] = acc / sum;
}

// A block's warps are tokens, one each, that score a run of pools together, IDX_TILE keys at a time through shared
// memory; lane h holds head h's query.  A score is summed in one order in any company: a head's product along the
// key, then the heads by warp_sum().
template <int DIM>
/*
DPCT1110:33: The total declared local variable size in device function
idx_score_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
idx_score_kernel(const float *__restrict__ q, const float *__restrict__ w,
                 const float *__restrict__ pool, float *__restrict__ score,
                 int score_stride, int n_head, int kpool, long long pos,
                 float wscale, int run, int rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &tile = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    sycl::float4[IDX_TILE * DIM / 4]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int warps = item_ct1.get_local_range(2) / WARP,
              h = item_ct1.get_local_id(2) % WARP;
    const int tok =
        item_ct1.get_group(1) * warps + item_ct1.get_local_id(2) / WARP;
    const int first = item_ct1.get_group(2) * run;
    const int last = sycl::min(
        first + run,
        (int)(int((pos + sycl::min(rows, (int)(int(item_ct1.get_group(1) + 1) *
                                               warps))) /
                  kpool)));
    if (first >= last) return;   // not a whole pool at the block's last token yet
    const bool live = tok < rows;
    const int n_pool = live ? int((pos + tok + 1) / kpool) : 0;
    float qh[DIM];
    float wh = 0.0f;
    if (live && h < n_head) {
        const sycl::float4 *src = reinterpret_cast<const sycl::float4 *>(
            q + (size_t(tok) * n_head + h) * DIM);
#pragma unroll
        for (int d = 0; d < DIM / 4; ++d) {
            const sycl::float4 v = src[d];
            qh[4 * d] = v.x();
            qh[4 * d + 1] = v.y();
            qh[4 * d + 2] = v.z();
            qh[4 * d + 3] = v.w();
        }
        wh = w[size_t(tok) * n_head + h] * wscale;
    } else {
#pragma unroll
        for (int d = 0; d < DIM; ++d) qh[d] = 0.0f;
    }
    float* out = score + size_t(tok) * score_stride;
    const sycl::float4 *keys = reinterpret_cast<const sycl::float4 *>(pool);
    for (int p0 = first; p0 < last; p0 += IDX_TILE) {
        const int n = sycl::min(IDX_TILE, last - p0);
#pragma unroll 4
        for (int i = item_ct1.get_local_id(2); i < n * DIM / 4;
             i += item_ct1.get_local_range(2))
            tile[i] = keys[size_t(p0) * (DIM / 4) + i];
        /*
        DPCT1118:34: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:197: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const int m =
            sycl::min(n, n_pool - p0);     // the token's own pools in the tile
        for (int j = 0; j < m; j += 2) {   // two at a time
            const int j1 = sycl::min(j + 1, m - 1);
            const sycl::float4 *k0 = tile + j * (DIM / 4),
                               *k1 = tile + j1 * (DIM / 4);
            float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
            for (int d = 0; d < DIM / 4; ++d) {
                const sycl::float4 a = k0[d], b = k1[d];
                d0 += qh[4 * d] * a.x();
                d1 += qh[4 * d] * b.x();
                d0 += qh[4 * d + 1] * a.y();
                d1 += qh[4 * d + 1] * b.y();
                d0 += qh[4 * d + 2] * a.z();
                d1 += qh[4 * d + 2] * b.z();
                d0 += qh[4 * d + 3] * a.w();
                d1 += qh[4 * d + 3] * b.w();
            }
            float s0 = 0.0f, s1 = 0.0f;
            if (h < n_head) {
                s0 += sycl::fmax(d0, 0.0f) * wh;
                s1 += sycl::fmax(d1, 0.0f) * wh;
            }
            s0 = warp_sum(s0);
            s1 = warp_sum(s1);
            if (h == 0) {
                out[p0 + j] = s0;
                if (j1 > j) out[p0 + j1] = s1;
            }
        }
        /*
        DPCT1118:35: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:198: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
}

// idx_score_kernel's sum over the heads for a prompt batch, whose products cuBLAS made: dot holds q[h] . pooled(p)
// for a token and head's n_pool pools together.  A thread a pool, blockIdx.y the token.
__dpct_inline__ void idx_weigh_kernel(const float *__restrict__ dot,
                                      const float *__restrict__ w,
                                      float *__restrict__ score,
                                      int score_stride, int n_pool, int n_head,
                                      int kpool, long long pos, float wscale) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int p = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    const size_t tok = item_ct1.get_group(1);
    if (p >= (pos + tok + 1) / kpool) return;   // not a whole pool at this token yet
    dot += tok * n_head * n_pool + p;
    w += tok * n_head;
    float acc = 0.0f;
#pragma unroll
    for (int h = 0; h < n_head; ++h) acc +=
        sycl::fmax(dot[size_t(h) * n_pool], 0.0f) * (w[h] * wscale);
    score[tok * score_stride + p] = acc;
}

// A score as an integer: a higher score is a larger one, and -0 is +0, as float comparisons have them.
__dpct_inline__ uint32_t idx_key(float s) {
    const uint32_t b = sycl::bit_cast<unsigned int>(s + 0.0f);
    return b & 0x80000000u ? ~b : b | 0x80000000u;
}

// x summed over the threads before this one, and over all of them in *all.  part has a slot a warp.
__dpct_inline__ int block_before(int x, int *part, int *all) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) % WARP,
              wid = item_ct1.get_local_id(2) / WARP,
              warps = item_ct1.get_local_range(2) / WARP;
    int in = x;
#pragma unroll
    for (int o = 1; o < WARP; o <<= 1) {
        /*
        DPCT1108:36: '__shfl_up_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        const int y = dpct::experimental::shift_sub_group_right(
            0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(), in,
            o);
        if (lane >= o) in += y;
    }
    if (lane == WARP - 1) part[wid] = in;
    /*
    DPCT1065:199: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (wid == 0) {
        int v = lane < warps ? part[lane] : 0;
#pragma unroll
        for (int o = 1; o < WARP; o <<= 1) {
            /*
            DPCT1108:37: '__shfl_up_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int y = dpct::experimental::shift_sub_group_right(
                0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(),
                v, o);
            if (lane >= o) v += y;
        }
        if (lane < warps) part[lane] = v;
    }
    /*
    DPCT1065:200: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int before = (wid ? part[wid - 1] : 0) + in - x;
    *all = part[warps - 1];
    /*
    DPCT1065:201: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(); // part is free again
    return before;
}

// A token's n_top pools with the highest scores, the lower pool first among equals, as rows of its list in that
// order, then the rows no pool has yet.  One block of IDX_SELECT a token.  The cut, the n_top-th highest key, comes
// a byte at a time: each warp counts the next byte of the keys that agree with it so far.  Then a thread has a run
// of the token's pools.  The block's shared memory holds the chosen pools and then their keys, n_top each.
/*
DPCT1110:38: The total declared local variable size in device function
idx_select_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void idx_select_kernel(const float *__restrict__ score,
                                       int score_stride,
                                       int32_t *__restrict__ sel,
                                       int sel_stride, int n_top, int kpool,
                                       long long pos, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto chosen = (uint32_t *)dpct_local;
    auto &hist = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int[IDX_SELECT / WARP][256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int[IDX_SELECT / WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pick = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[2]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const long long at = pos + item_ct1.get_group(2);
    const int n_pool = int((at + 1) / kpool);
    const float *s = score + size_t(item_ct1.get_group(2)) * score_stride;
    const int lane = item_ct1.get_local_id(2) % WARP,
              wid = item_ct1.get_local_id(2) / WARP;
    uint32_t cut = 0, mask = 0;
    int want = n_top;   // the cut's place among the keys that agree with it on mask
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int d = lane; d < 256; d += WARP) hist[wid][d] = 0;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        for (int i = item_ct1.get_local_id(2); i < n_pool;
             i += item_ct1.get_local_range(2)) {
            const uint32_t k = idx_key(s[i]);
            if ((k & mask) == cut) dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(
                &hist[wid][(k >> shift) & 255], 1);
        }
        /*
        DPCT1118:39: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (item_ct1.get_local_id(2) < 256) {
            int t = 0;
#pragma unroll
            for (int k = 0; k < IDX_SELECT / WARP; ++k) t +=
                hist[k][item_ct1.get_local_id(2)];
            hist[0][item_ct1.get_local_id(2)] =
                t; // a thread reads and writes its own column only
        }
        /*
        DPCT1118:40: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (item_ct1.get_local_id(2) == 0) {
            int d = 255, above = 0;
#pragma unroll
            for (; d > 0 && above + hist[0][d] < want; --d) above += hist[0][d];
            pick[0] = d;
            pick[1] = above;
        }
        /*
        DPCT1118:41: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        cut |= uint32_t(pick[0]) << shift;
        mask |= 255u << shift;
        want -= pick[1];
        /*
        DPCT1118:42: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    // the pools above the cut, and of those on it the lowest, in the order of the pools
    const int per = (n_pool + int(item_ct1.get_local_range(2)) - 1) /
                    int(item_ct1.get_local_range(2));
    const int lo = item_ct1.get_local_id(2) * per,
              hi = sycl::min(lo + per, n_pool);
    int g = 0, e = 0;
    for (int i = lo; i < hi; ++i) {
        const uint32_t k = idx_key(s[i]);
        g += k > cut;
        e += k == cut;
    }
    int g_all, e_all;   // above and on the cut in all the threads
    const int g0 = block_before(g, part, &g_all);   // in the threads before this one
    int e0 = block_before(e, part, &e_all);
    const int need = n_top - g_all;
    int out = g0 + sycl::min(e0, need);
    for (int i = lo; i < hi; ++i) {
        const uint32_t k = idx_key(s[i]);
        if (k > cut || (k == cut && e0++ < need)) {
            chosen[out] = i;
            chosen[n_top + out++] = k;
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // a chosen pool's place: behind those with a higher score and the lower pools with its own
    for (int c = item_ct1.get_local_id(2); c < n_top;
         c += item_ct1.get_local_range(2)) {
        const uint32_t mine = chosen[n_top + c];
        int rank = 0;
#pragma unroll
        for (int k = 0; k < n_top; ++k) rank +=
            chosen[n_top + k] > mine || (chosen[n_top + k] == mine && k < c);
        int32_t *row = sel + size_t(item_ct1.get_group(2)) * sel_stride +
                       size_t(rank) * kpool;
#pragma unroll
        for (int j = 0; j < kpool; ++j) row[j] = int32_t(chosen[c]) * kpool + j;
    }
    if (item_ct1.get_local_id(2) != 0) return;
    int32_t *tail = sel + size_t(item_ct1.get_group(2)) * sel_stride +
                    size_t(n_top) * kpool;
#pragma unroll
    for (long long p = (long long)n_pool * kpool; p <= at; ++p) *tail++ =
        int32_t(p);
}

__dpct_inline__ void swiglu_clamp_kernel(const float *__restrict__ gate,
                                         const float *__restrict__ up,
                                         float *__restrict__ out, int n,
                                         float limit) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    const float g = sycl::fmin(gate[i], limit);
    const float u = sycl::fmin(sycl::fmax(up[i], -limit), limit);
    out[i] = g / (1.0f + sycl::native::exp(-g)) * u;
}

__dpct_inline__ void axpy_kernel(const float *__restrict__ x, float a,
                                 float *__restrict__ acc, int n, bool add) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    acc[i] = add ? acc[i] + x[i] * a : x[i] * a;
}

__dpct_inline__ void copy2_kernel(const uint32_t *__restrict__ x0,
                                  uint32_t *__restrict__ y0, int n0,
                                  const uint32_t *__restrict__ x1,
                                  uint32_t *__restrict__ y1, int n1) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n0) y0[i] = x0[i];
    else if (i < n0 + n1) y1[i - n0] = x1[i - n0];
}

// The sums may be one and the same, so a thread adds its value of each vector in turn.
__dpct_inline__ void add_scaled_kernel(ScaledRows rows, int count, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
#pragma unroll
    for (int k = 0; k < count; ++k)
        rows.acc[k][i] = rows.acc[k][i] + rows.x[k][i] * rows.w[k];
}

// a row's gate and up scales; 1 (exact) where there is none
__dpct_inline__ sycl::float2 row_scales(const ExpertScales &s, int row) {
    if (!s.gate && !s.up) return sycl::float2(1.0f, 1.0f);
    const int e = s.ids ? s.ids[row] : s.e[row];
    return sycl::float2(s.gate ? s.gate[e] : 1.0f, s.up ? s.up[e] : 1.0f);
}

__dpct_inline__ void swiglu_clamp_rows_kernel(const float *__restrict__ gate_up,
                                              float *__restrict__ out, int n,
                                              float limit, ExpertScales s) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    const float *row = gate_up + size_t(item_ct1.get_group(1)) * 2 * n;
    const sycl::float2 sc = row_scales(s, item_ct1.get_group(1));
    const float g = sycl::fmin(row[i] * sc.x(), limit);
    const float u = sycl::fmin(sycl::fmax(row[n + i] * sc.y(), -limit), limit);
    out[size_t(item_ct1.get_group(1)) * n + i] =
        g / (1.0f + sycl::native::exp(-g)) * u;
}

__dpct_inline__ void expert_weights_kernel(float *__restrict__ w,
                                           const float *__restrict__ scale,
                                           const int32_t *__restrict__ ids,
                                           int rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (r < rows) w[r] *= scale[ids[r]];
}

__dpct_inline__ float gelu(float x) {
    return 0.5f * x *
           (1.0f + sycl::tanh(0.79788456080286535587989211986876f * x *
                              (1.0f + 0.044715f * x * x)));
}

__dpct_inline__ void geglu_kernel(const float *__restrict__ gate,
                                  const float *__restrict__ up,
                                  float *__restrict__ out, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) out[i] = gelu(gate[i]) * up[i];
}

__dpct_inline__ void geglu_rows_kernel(const float *__restrict__ gate_up,
                                       float *__restrict__ out, int n,
                                       ExpertScales s) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    const float *row = gate_up + size_t(item_ct1.get_group(1)) * 2 * n;
    const sycl::float2 sc = row_scales(s, item_ct1.get_group(1));
    out[size_t(item_ct1.get_group(1)) * n + i] =
        gelu(row[i] * sc.x()) * (row[n + i] * sc.y());
}

// A warp a token, a lane every 32nd expert.  The pick is the host's: the n_used largest of the probabilities plus
// the bias, the lower expert first among equals, one after the other.
constexpr int ROUTE_PER = 16;   // experts a lane: up to 512
// The kernel takes PER experts a lane, the fewest that hold the model's: a slot past them is a -inf that changes no
// pick and no sum, and walking all sixteen made a token's routing 3.5 us on an RTX 5090.
template <int PER>

/*
DPCT1110:47: The total declared local variable size in device function
route_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void route_kernel(const float *__restrict__ logits, int n_expert, int n_used,
                  int rows, int gate, float in_scale,
                  const float *__restrict__ bias,
                  const float *__restrict__ escale, bool norm, float w_scale,
                  int32_t *__restrict__ ids, float *__restrict__ w) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) /
                    WARP,
              lane = item_ct1.get_local_id(2) % WARP;
    if (row >= rows) return;
    const float* lg = logits + size_t(row) * n_expert;
    float p[PER], pick[PER];
    float mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int e = lane + k * WARP;
        p[k] = e < n_expert ? lg[e] * in_scale : -INFINITY;
        mx = sycl::fmax(mx, p[k]);
    }
    if (gate == 0) {   // softmax
#pragma unroll
        /*
        DPCT1108:43: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        for (int o = WARP / 2; o > 0; o >>= 1) mx = sycl::fmax(
            mx, dpct::experimental::permute_sub_group_by_xor(
                    0xffffffff,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), mx, o));
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            p[k] = lane + k * WARP < n_expert ? sycl::native::exp(p[k] - mx)
                                              : 0.0f;
            sum += p[k];
        }
        const float inv = 1.0f / warp_sum(sum);
#pragma unroll
        for (int k = 0; k < PER; ++k) p[k] *= inv;
    } else if (gate == 3) {   // sigmoid_logit_add (Kolibri 1): the weights are the sigmoid of the scores, but the
                              // pick is on the scores and the bias, so it is taken before the scores turn to weights
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const int e = lane + k * WARP;
            pick[k] = e < n_expert ? p[k] + (bias ? bias[e] : 0.0f) : -INFINITY;
            p[k] = e < n_expert ? 1.0f / (1.0f + sycl::native::exp(-p[k]))
                                : -INFINITY;
        }
    } else {
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const float x = p[k];
            p[k] =
                gate == 1
                    ? 1.0f / (1.0f + sycl::native::exp(-x))
                    : sycl::sqrt(x > 20.0f
                                     ? x
                                     : sycl::log(1.0f + sycl::native::exp(x)));
        }
    }
    if (gate != 3) {
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const int e = lane + k * WARP;
            pick[k] = e < n_expert ? p[k] + (bias ? bias[e] : 0.0f) : -INFINITY;
        }
    }
    float sum = 0.0f, mine = 0.0f;
    int my_id = 0;
    for (int j = 0; j < n_used; ++j) {
        float best = -INFINITY;
        int at = n_expert;
#pragma unroll
        for (int k = 0; k < PER; ++k)
            if (pick[k] > best) best = pick[k], at = lane + k * WARP;
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) {
            /*
            DPCT1108:44: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float b = dpct::experimental::permute_sub_group_by_xor(
                0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(),
                best, o);
            /*
            DPCT1108:45: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int a = dpct::experimental::permute_sub_group_by_xor(
                0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(),
                at, o);
            if (b > best || (b == best && a < at)) best = b, at = a;
        }
        float pw = 0.0f;
#pragma unroll
        for (int k = 0; k < PER; ++k)
            if (lane + k * WARP == at) pw = p[k], pick[k] = -INFINITY;
        /*
        DPCT1108:46: '__shfl_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        pw = dpct::experimental::select_from_sub_group(
            0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(), pw,
            at % WARP);
        sum += pw;
        if (lane == j) mine = pw, my_id = at;
    }
    if (lane >= n_used) return;
    float v = mine;
    if (norm) v /= sycl::fmax(
        sum,
        6.103515625e-5f); // the floor ggml's MoE graph puts under the gate sum
    if (escale) v *= escale[my_id];
    if (w_scale != 0.0f) v *= w_scale;
    ids[size_t(row) * n_used + lane] = my_id;
    w[size_t(row) * n_used + lane] = v;
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <int PER>
void route_kernel_wrapper(const float *__restrict logits, int n_expert,
                          int n_used, int rows, int gate, float in_scale,
                          const float *__restrict bias,
                          const float *__restrict escale, bool norm,
                          float w_scale, int32_t *__restrict ids,
                          float *__restrict w) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    queue.parallel_for<dpct_kernel_name<class route_kernel_, dpct_kernel_scalar<PER>>>(
        nr, exp_props,
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            route_kernel<PER>(logits, n_expert, n_used, rows, gate, in_scale,
                              bias, escale, norm, w_scale, ids, w);
        });
}

__dpct_inline__ void moe_sum_kernel(const float *__restrict__ x,
                                    const float *__restrict__ w,
                                    float *__restrict__ acc, int n,
                                    int n_used) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    const size_t first = size_t(item_ct1.get_group(1)) * n_used;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < n_used; ++k) s += w[first + k] * x[(first + k) * n + i];
    acc[size_t(item_ct1.get_group(1)) * n + i] += s;
}

__dpct_inline__ void add_bias_kernel(const float *__restrict__ b,
                                     float *__restrict__ acc, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) acc[size_t(item_ct1.get_group(1)) * n + i] += b[i];
}

// One block per head and token.  The pairs of the rotation are i and i + n_rot / 2 for i below n_rot / 2.  With
// axes, a token has a position on each of three axes, and pair j turns by the one on axis (j % sections) % 3.
// With `at`, the first token's position is what the card holds there when the launch runs.
__dpct_inline__ void
head_norm_rope_kernel(const float *__restrict__ x, int x_stride,
                      const float *__restrict__ w, float *__restrict__ y,
                      int dim, int heads, int n_rot, float theta_scale,
                      const float *__restrict__ freq, long long pos, float eps,
                      const int *__restrict__ axes, int sections,
                      const long long *__restrict__ at) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const float *xr = x + size_t(item_ct1.get_group(2)) * x_stride;
    float *yr = y + size_t(item_ct1.get_group(2)) * dim;
    float ss = 0.0f;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < dim;
         i += item_ct1.get_local_range(2)) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float scale = 1.0f / sycl::sqrt(ss / float(dim) + eps);
    const int tok = item_ct1.get_group(2) / heads;
    const float own = float((at ? *at : pos) + tok);
    const int half = n_rot / 2;
    for (int i = item_ct1.get_local_id(2); i < dim;
         i += item_ct1.get_local_range(2)) {
        if (i >= n_rot) {
            yr[i] = w ? xr[i] * scale * w[i] : xr[i] * scale;
            continue;
        }
        const int j = i < half ? i : i - half;
        const float p = axes ? float(axes[size_t(tok) * 3 + (j % sections) % 3]) : own;
        const float turn = p * dpct::pow(theta_scale, float(j));
        const float theta = freq ? turn / freq[j] : turn;
        const float x0 = w ? xr[j] * scale * w[j] : xr[j] * scale;
        const float x1 = w ? xr[j + half] * scale * w[j + half] : xr[j + half] * scale;
        yr[i] =
            i < half
                ? x0 * sycl::cos((float)theta) - x1 * sycl::sin((float)theta)
                : x0 * sycl::sin((float)theta) + x1 * sycl::cos((float)theta);
    }
}

// A launch's `last` as attn_rows() takes it: -2 is a whole call of one launch whose position is on the card (attend()),
// which ends at the launch's last token.  p0 is the launch's first token's position.
__dpct_inline__ long long attn_last(long long p0, int tokens, long long last) {
    return last == -2 ? p0 + tokens - 1 : last;
}

// The rows the token at position p attends: how many, and the first of them.  They end at its own row, or at row
// `last` when that is given: the tokens of the call then see each other.
__dpct_inline__ int attn_rows(long long p, int window, long long last,
                              long long *lo) {
    *lo = window > 0 && p >= window ? p - window + 1 : 0;
    return int((last < 0 ? p : last) - *lo) + 1;
}

// The rows the `tokens` tokens from position p attend between them: how many, and the first of them.  A launch's
// scores lie in that order, so that its tokens read a row of the caches once between them.
__dpct_inline__ int attn_span(long long p, int tokens, int window,
                              long long last, long long *lo) {
    long long lo_last;
    const int n = attn_rows(p + tokens - 1, window, last, &lo_last);
    attn_rows(p, window, last, lo);
    return int(lo_last - *lo) + n;
}

// The scores of the key k with N queries that lie one after the other from q, for the warp of attn_score_kernel.
// score: where the first of them has its score of the key's row; the others lie n_keys floats apart.
template <int N, bool Q8 = false>
__dpct_inline__ void
attn_scores(const float *__restrict__ q, const sycl::half *__restrict__ k,
            const int8_t *__restrict__ kcodes,
            const uint16_t *__restrict__ kscales, float *__restrict__ score,
            int n_keys, int dim, float scale) {
    static_assert(SCORE_VALUES == 4, "a step is four halves of the key in one read and a float4 of a query");
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) %
        WARP;
    const float* qi[N];
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
        qi[i] = q + size_t(i) * dim;
        acc[i] = 0.0f;
    }
    for (int d = SCORE_VALUES * lane; d < dim; d += SCORE_VALUES * WARP) {
        float x[SCORE_VALUES];
        if constexpr (Q8) {
            // kcodes/kscales point at the row's (group's) first code and scale: a 32-value block holds one scale,
            // so the step's block is (SCORE_VALUES * lane) / 32, one scale per 8 lanes of the warp.
            const float sc =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(kscales[d / 32]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
#pragma unroll
            for (int j = 0; j < SCORE_VALUES; ++j) x[j] = (float) kcodes[d + j] * sc;
        } else {
            const unsigned long long bits = *reinterpret_cast<const unsigned long long*>(k + d);
#pragma unroll
            for (int j = 0; j < SCORE_VALUES; ++j)
                x[j] = sycl::vec<sycl::half, 1>(
                           sycl::bit_cast<sycl::half, unsigned short>(
                               (unsigned short)(bits >> 16 * j)))
                           .convert<float, sycl::rounding_mode::automatic>()[0];
        }
#pragma unroll
        for (int i = 0; i < N; ++i) {
            const sycl::float4 y =
                *reinterpret_cast<const sycl::float4 *>(qi[i] + d);
            acc[i] += y.x() * x[0];
            acc[i] += y.y() * x[1];
            acc[i] += y.z() * x[2];
            acc[i] += y.w() * x[3];
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
        /*
DPCT1108:48: '__shfl_down_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
        for (int off = WARP / 2; off > 0; off >>= 1) acc[i] +=
            dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                acc[i], off);
        if (lane == 0) score[size_t(i) * n_keys] = acc[i] * scale;
    }
}

// attn_scores() for n queries: N at a time, then what is left by halves of N.
template <int N, bool Q8 = false>
__dpct_inline__ void attn_scores_by(int n, const float *__restrict__ q,
                                    const sycl::half *__restrict__ k,
                                    const int8_t *__restrict__ kcodes,
                                    const uint16_t *__restrict__ kscales,
                                    float *__restrict__ score, int n_keys,
                                    int dim, float scale) {
#pragma unroll
    for (; n >= N; n -= N, q += size_t(N) * dim, score += size_t(N) * n_keys)
        attn_scores<N, Q8>(q, k, kcodes, kscales, score, n_keys, dim, scale);
    if constexpr (N > 1) attn_scores_by<N / 2, Q8>(n, q, k, kcodes, kscales, score, n_keys, dim, scale);
}

// score[token][head][i] = scale * (q[token][head] . the key of the head's group in the i-th row of the launch's
// span).  A warp has a row and one of the row's groups of heads.  Its threads read the key side by side,
// SCORE_VALUES values each a step, so that a step reads one place of the cache, and the launch's tokens read it
// once between them: the cache of a long context is what the kernel waits for.  A thread sums its products with
// a query, for up to SCORE_QUERIES heads of a token at a time, and a query's 32 sums are added in pairs.  A
// token gets scores for the rows of the span it does not attend as well, which the softmax leaves out.
template <bool Q8 = false>
__dpct_inline__ void attn_score_kernel(
    const float *__restrict__ q, const sycl::half *__restrict__ kcache,
    float *__restrict__ score, int n_keys, int dim, int n_head, int kv_heads,
    long long pos, int first, int tokens, float scale, int window, int ring,
    long long last, const long long *__restrict__ at,
    const int8_t *__restrict__ kcodes = nullptr,
    const uint16_t *__restrict__ kscales = nullptr) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int unit =
        item_ct1.get_group(2) * (item_ct1.get_local_range(2) / WARP) +
        item_ct1.get_local_id(2) / WARP; // a row and a group
    const int key = unit / kv_heads, g = unit % kv_heads, per = n_head / kv_heads;
    last = attn_last((at ? *at : pos) + first, tokens, last);
    long long lo;
    if (key >= attn_span((at ? *at : pos) + first, tokens, window, last, &lo)) return;
    const long long row = ring > 0 ? (lo + key) % ring : lo + key;
    // the row's key of this group: the codes use the fp16 cache's index, the scales block a 32-value block
    const size_t base = (size_t(row) * kv_heads + g) * dim;
    const sycl::half *k = kcache + base;
    const int8_t* kc = kcodes + base;
    const uint16_t* ks = kscales + base / 32;
    q += (size_t(first) * n_head + g * per) * dim;   // the group's heads of the launch's first token
    score += size_t(g) * per * n_keys + key;         // and their scores of the row
#pragma unroll
    for (int t = 0; t < tokens;
         ++t, q += size_t(n_head) * dim, score += size_t(n_head) * n_keys)
        attn_scores_by<SCORE_QUERIES, Q8>(per, q, k, kc, ks, score, n_keys, dim, scale);
}

// The sums of N scores a lane holds parts of, added over the warp in the tree attn_scores() adds one score in: at
// each of its steps a lane adds the part a lane `off` away has.  While a lane holds more than one score it keeps
// half of them and gives the other half away, so one exchange serves two scores; then the four lanes of a score
// finish it.  A sum is the other tree's to the bit, since a + b is b + a.  Returns the lane's score, `which` of the N.
template <int N, int FROM = WARP / 2>
__dpct_inline__ float scores_sum(float *v, int lane, int *which) {
    int idx = 0;
#pragma unroll
    for (int off = FROM, m = N; off > 0; off >>= 1) {
        if (m > 1) {
            const int half = m / 2;
            const bool upper = lane & off;
#pragma unroll
            for (int i = 0; i < half; ++i) {
                const float send = upper ? v[i] : v[i + half], keep = upper ? v[i + half] : v[i];
                /*
                DPCT1108:49: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                v[i] = keep +
                       dpct::experimental::permute_sub_group_by_xor(
                           0xffffffffu,
                           sycl::ext::oneapi::this_work_item::get_sub_group(),
                           send, off);
            }
            idx = 2 * idx + (upper ? 1 : 0);
            m = half;
        } else {
            /*
            DPCT1108:50: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            v[0] += dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                v[0], off);
        }
    }
    *which = idx;
    return v[0];
}

// attn_score_kernel for heads of 128 * STEPS values, N of a group's heads at a time: the same products in the same
// order and the same sums, but a warp has SCORE_KEYS rows of a group, and a token's queries stay in its registers
// while it goes through them.  The other kernel read them again for every row, and passed every score's parts from lane to
// lane on its own: those reads and moves were what it waited for.
template <int N, int STEPS, bool Q8 = false>
/*
DPCT1110:51: The total declared local variable size in device function
attn_score_reg_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void attn_score_reg_kernel(const float *__restrict__ q,
                           const sycl::half *__restrict__ kcache,
                           float *__restrict__ score, int n_keys, int n_head,
                           int kv_heads, long long pos, int first, int tokens,
                           float scale, int window, int ring, long long last,
                           const long long *__restrict__ at,
                           const int8_t *__restrict__ kcodes = nullptr,
                           const uint16_t *__restrict__ kscales = nullptr) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    static_assert(
        SCORE_VALUES == 4,
        "a step is four halves of the key in one read and a float4 of a query");
    constexpr int dim = SCORE_VALUES * WARP * STEPS;
    const int unit = item_ct1.get_group(2) * SCORE_WARPS +
                     item_ct1.get_local_id(2) / WARP; // a group's block of rows
    const int g = unit % kv_heads, k0 = unit / kv_heads * SCORE_KEYS,
              lane = item_ct1.get_local_id(2) % WARP;
    last = attn_last((at ? *at : pos) + first, tokens, last);
    long long lo;
    const int span = attn_span((at ? *at : pos) + first, tokens, window, last, &lo);
    if (k0 >= span) return;
    const int rows = sycl::min(SCORE_KEYS, span - k0), per = n_head / kv_heads;
    for (int th = 0; th < tokens * per; th += N) {   // N heads of a token
        const int t = th / per, h = g * per + th % per;
        const float* qt = q + ((size_t(first) + t) * n_head + h) * dim + SCORE_VALUES * lane;
        sycl::float4 y[N][STEPS];
#pragma unroll
        for (int i = 0; i < N; ++i)
#pragma unroll
            for (int s = 0; s < STEPS; ++s) y[i][s] =
                *reinterpret_cast<const sycl::float4 *>(qt + i * dim + 128 * s);
        float* st = score + (size_t(t) * n_head + h) * n_keys + k0;
        long long row = ring > 0 ? (lo + k0) % ring : lo + k0;
        for (int r = 0; r < rows; ++r) {
            // the row's key of this group: the codes use the fp16 cache's index, the scales block a 32-value block
            const size_t base = (size_t(row) * kv_heads + g) * dim;
            const sycl::half *k = kcache + base + SCORE_VALUES * lane;
            float acc[N];
#pragma unroll
            for (int i = 0; i < N; ++i) acc[i] = 0.0f;
#pragma unroll
            for (int s = 0; s < STEPS; ++s) {
                float x[SCORE_VALUES];
                if constexpr (Q8) {
                    const int off = SCORE_VALUES * lane + 128 * s;
                    const float d =
                        sycl::vec<sycl::half, 1>(
                            sycl::bit_cast<sycl::half, unsigned short>(
                                kscales[(base + off) / 32]))
                            .convert<float,
                                     sycl::rounding_mode::automatic>()[0];
#pragma unroll
                    for (int j = 0; j < SCORE_VALUES; ++j) x[j] = (float) kcodes[base + off + j] * d;
                } else {
                    const unsigned long long bits = *reinterpret_cast<const unsigned long long*>(k + 128 * s);
#pragma unroll
                    for (int j = 0; j < SCORE_VALUES; ++j)
                        x[j] =
                            sycl::vec<sycl::half, 1>(
                                sycl::bit_cast<sycl::half, unsigned short>(
                                    (unsigned short)(bits >> 16 * j)))
                                .convert<float,
                                         sycl::rounding_mode::automatic>()[0];
                }
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    acc[i] += y[i][s].x() * x[0];
                    acc[i] += y[i][s].y() * x[1];
                    acc[i] += y[i][s].z() * x[2];
                    acc[i] += y[i][s].w() * x[3];
                }
            }
            int which;
            const float sum = scores_sum<N>(acc, lane, &which);
            if ((lane & (WARP / N - 1)) == 0) st[size_t(which) * n_keys + r] = sum * scale;
            if (++row == ring) row = 0;
        }
    }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <int N, int STEPS, bool Q8 = false>
void attn_score_reg_kernel_wrapper(
    const float *__restrict q, const sycl::half *__restrict kcache,
    float *__restrict score, int n_keys, int n_head, int kv_heads,
    long long pos, int first, int tokens, float scale, int window, int ring,
    long long last, const long long *__restrict at,
    const int8_t *__restrict kcodes = nullptr,
    const uint16_t *__restrict kscales = nullptr) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
    dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

    queue.parallel_for<dpct_kernel_name<class attn_score_reg_kernel_, dpct_kernel_scalar<N>, dpct_kernel_scalar<STEPS>, dpct_kernel_scalar<Q8>>>(
        nr, exp_props,
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            attn_score_reg_kernel<N, STEPS, Q8>(
                q, kcache, score, n_keys, n_head, kv_heads, pos, first, tokens,
                scale, window, ring, last, at, kcodes, kscales);
        });
}

// Softmax in place over a head's scores of the rows its token attends; the other rows of the launch's span get
// zero, which adds nothing to the token's values.  One block per head and token.  With `wt` the weights go there
// instead, in the order attn_value_t_kernel reads them: for each of the span's rows, the VALUE_QUERIES queries of
// a block of the group's queries side by side (per: the group's heads).
__dpct_inline__ void attn_softmax_kernel(float *score, int n_keys,
                                         long long pos, int first, int tokens,
                                         int window, long long last,
                                         const long long *__restrict__ at,
                                         float *__restrict__ wt, int per,
                                         const float *__restrict__ sinks) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const long long p = (at ? *at : pos) + first;
    last = attn_last(p, tokens, last);
    long long lo, lo0;
    const int span = attn_span(p, tokens, window, last, &lo0);
    const int n = attn_rows(p + item_ct1.get_group(1), window, last, &lo),
              own = int(lo - lo0);
    float *all =
        score + (size_t(item_ct1.get_group(1)) * item_ct1.get_group_range(2) +
                 item_ct1.get_group(2)) *
                    n_keys;
    float* row = all + own;   // the token's rows of the span
    float mx = -INFINITY;
#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < n;
         j += item_ct1.get_local_range(2)) mx = sycl::fmax(mx, row[j]);
    mx = block_max(mx, part);
    if (sinks) mx = sycl::fmax(mx, sinks[item_ct1.get_group(2)]);
    float sum = 0.0f;
#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < n;
         j += item_ct1.get_local_range(2)) {
        row[j] = sycl::native::exp(row[j] - mx);
        sum += row[j];
    }
    sum = block_sum(sum, part);
    if (sinks) sum += sycl::native::exp(sinks[item_ct1.get_group(2)] - mx);
    const float inv = 1.0f / sum;
    if (!wt) {
#pragma unroll
        for (int j = item_ct1.get_local_id(2); j < n;
             j += item_ct1.get_local_range(2)) row[j] *= inv;
#pragma unroll
        for (int j = item_ct1.get_local_id(2); j < own;
             j += item_ct1.get_local_range(2)) all[j] = 0.0f;
#pragma unroll
        for (int j = own + n + item_ct1.get_local_id(2); j < span;
             j += item_ct1.get_local_range(2)) all[j] = 0.0f;
        return;
    }
    const int h = item_ct1.get_group(2),
              query = item_ct1.get_group(1) * per +
                      h % per; // the query among its group's
    const int blocks = (tokens * per + VALUE_QUERIES - 1) / VALUE_QUERIES;
    float* o = wt + (size_t(h / per) * blocks + query / VALUE_QUERIES) * n_keys * VALUE_QUERIES + query % VALUE_QUERIES;
#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < own;
         j += item_ct1.get_local_range(2)) o[size_t(j) * VALUE_QUERIES] = 0.0f;
#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < n;
         j += item_ct1.get_local_range(2))
        o[size_t(own + j) * VALUE_QUERIES] = row[j] * inv;
#pragma unroll
    for (int j = own + n + item_ct1.get_local_id(2); j < span;
         j += item_ct1.get_local_range(2)) o[size_t(j) * VALUE_QUERIES] = 0.0f;
}

// A thread's sums of attn_value_kernel for N queries: over its share of the span's rows from `row` of the cache
// on, of a row's value with each query's weight there.  w: the weights of the lane's query; the warp's first N
// lanes have the N queries between them.
template <int N, bool Q8>
__dpct_inline__ void
attn_values(const float *__restrict__ w, const sycl::half *__restrict__ v,
            const int8_t *__restrict__ codes,
            const uint16_t *__restrict__ scales, size_t width, size_t width_sc,
            long long row, int ring, int span, float *sums) {
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) acc[i] = 0.0f;
    v += size_t(row) * width;
    if constexpr (Q8) {
        codes += size_t(row) * width;
        scales += size_t(row) * width_sc;
    }
    for (int j =
             sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(
                 2) /
             VALUE_DIMS;
         j < span; j += VALUE_SHARES) {
        float x;
        if constexpr (Q8) x =
            (float)(*codes) *
            sycl::vec<sycl::half, 1>(
                sycl::bit_cast<sycl::half, unsigned short>(*scales))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        else x = sycl::vec<sycl::half, 1>(*v)
                     .convert<float, sycl::rounding_mode::automatic>()[0];
        const float weight = w[j];
#pragma unroll
        /*
        DPCT1108:52: '__shfl_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        for (int i = 0; i < N; ++i) acc[i] +=
            dpct::experimental::select_from_sub_group(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                weight, i) *
            x;
        v += VALUE_SHARES * width;
        if constexpr (Q8) {
            codes += VALUE_SHARES * width;
            scales += VALUE_SHARES * width_sc;
        }
        row += VALUE_SHARES;
        if (ring > 0 && row >= ring) {
            row -= ring;
            v -= size_t(ring) * width;
            if constexpr (Q8) {
                codes -= size_t(ring) * width;
                scales -= size_t(ring) * width_sc;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) sums[i] = acc[i];
}

// attn_values() for n queries, with N if no smaller power of two holds them.
template <int N, bool Q8>
__dpct_inline__ void attn_values_for(int n, const float *__restrict__ w,
                                     const sycl::half *__restrict__ v,
                                     const int8_t *__restrict__ codes,
                                     const uint16_t *__restrict__ scales,
                                     size_t width, size_t width_sc,
                                     long long row, int ring, int span,
                                     float *sums) {
    if constexpr (N > 1)
        if (n <= N / 2)
            return attn_values_for<N / 2, Q8>(n, w, v, codes, scales, width, width_sc, row, ring, span, sums);
    attn_values<N, Q8>(w, v, codes, scales, width, width_sc, row, ring, span, sums);
}

// out[token][head][d] = sum over the rows of score[token][head][row] * the value of the head's group in that
// row.  A block has a group of heads, VALUE_DIMS value dimensions (blockIdx.x) and up to VALUE_QUERIES of the
// group's queries (blockIdx.y): the group's heads of the launch's first token, then of its second, and so on.
// A thread has one of the dimensions and one of VALUE_SHARES shares of the span's rows, every VALUE_SHARES-th
// row from its own, so that many reads of the cache are on their way at a time.  It reads a row's value once
// for the block's queries; their weights of the row come in one read as well, a lane a query, and the warp
// passes them around.  A token's shares are counted from its own first row and added in their order, so that it
// comes out the same whatever tokens the launch has beside it.
template <bool Q8 = false>
__dpct_inline__ void attn_value_kernel(
    const float *__restrict__ score, const sycl::half *__restrict__ vcache,
    float *__restrict__ out, int n_keys, int dim, int n_head, int kv_heads,
    long long pos, int first, int tokens, int window, int ring, long long last,
    const long long *__restrict__ at,
    const int8_t *__restrict__ vcodes = nullptr,
    const uint16_t *__restrict__ vscales = nullptr) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    static_assert(VALUE_DIMS == WARP && VALUE_QUERIES <= WARP,
                  "a warp is a share and reads a query's weight a lane");
    auto &share = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[VALUE_QUERIES][VALUE_SHARES][VALUE_DIMS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int per = n_head / kv_heads, dims = (dim + VALUE_DIMS - 1) / VALUE_DIMS;
    const int g = item_ct1.get_group(2) / dims,
              q0 = item_ct1.get_group(1) * VALUE_QUERIES;
    const int live =
        sycl::min(VALUE_QUERIES, tokens * per - q0); // the block's queries
    const int lane = item_ct1.get_local_id(2) % VALUE_DIMS,
              s = item_ct1.get_local_id(2) / VALUE_DIMS;
    const int d = (item_ct1.get_group(2) % dims) * VALUE_DIMS + lane;
    const long long p = (at ? *at : pos) + first;
    last = attn_last(p, tokens, last);
    long long lo;
    const int span = attn_span(p, tokens, window, last, &lo);
    // the lane's query, and for a lane past the head's last value a value to read all the same
    const int mine = q0 + sycl::min(lane % VALUE_QUERIES, live - 1);
    const int dd = sycl::min(
        d, dim - 1); // a lane past the last value reads one all the same
    float sums[VALUE_QUERIES];
    attn_values_for<VALUE_QUERIES, Q8>(
        live, score + (size_t(mine / per) * n_head + g * per + mine % per) * n_keys,
        vcache + size_t(g) * dim + dd, vcodes + size_t(g) * dim + dd,
        vscales + size_t(g) * (dim / 32) + dd / 32, size_t(kv_heads) * dim, size_t(kv_heads) * (dim / 32),
        ring > 0 ? (lo + s) % ring : lo + s, ring, span, sums);
    for (int i = 0; i < live; ++i) {
        long long own;   // the first row of the query's token: the thread's rows are this share of the token's
        attn_rows(p + (q0 + i) / per, window, last, &own);
        share[i][(s + VALUE_SHARES - int((own - lo) % VALUE_SHARES)) % VALUE_SHARES][lane] = sums[i];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (s != 0 || d >= dim) return;
    for (int i = 0; i < live; ++i) {
        float sum = share[i][0][lane];
#pragma unroll
        for (int k = 1; k < VALUE_SHARES; ++k) sum += share[i][k][lane];
        const int query = q0 + i;
        out[((size_t(first) + query / per) * n_head + g * per + query % per) * dim + d] = sum;
    }
}

// attend() for a few tokens over the whole conversation, for heads of 64 values in groups of 4 (LFM2's) or of 128
// in groups of 8 (Flint's): the keys in blocks of SPLIT_KEYS, a launch block each with a group and a token, which
// scores them (half a warp a row of 64 values, a warp one of 128), takes the softmax's exponentials of its block for
// each head (a warp a head) and their values; attn_join_kernel adds the blocks up.  For a token of these shapes the
// score, softmax and value kernels launch 16 to 32 blocks, each walking the whole conversation.  A block's rows are
// counted from the cache's first, so a token comes out the same whatever tokens the call has beside it.
constexpr int SPLIT_KEYS = 64;

template <int DIM, int G>
/*
DPCT1110:53: The total declared local variable size in device function
attn_split_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void attn_split_kernel(const float *__restrict__ q,
                                       const sycl::half *__restrict__ kcache,
                                       const sycl::half *__restrict__ vcache,
                                       float *__restrict__ part, int n_head,
                                       int kv_heads, long long pos, int first,
                                       int tokens, float scale, int blocks,
                                       const long long *__restrict__ at) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROW = DIM / SCORE_VALUES, ROWS = WARP / ROW, WARPS = G,
                  VALS = DIM / WARP;
    static_assert(ROW * ROWS == WARP && ROW % G == 0 && VALS % 2 == 0, "a row is half a warp or a warp, a warp a head");
    auto &e = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[G][SPLIT_KEYS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int g = item_ct1.get_group(1), b = item_ct1.get_group(2),
              t = item_ct1.get_group(0), lane = item_ct1.get_local_id(2) % WARP,
              w = item_ct1.get_local_id(2) / WARP;
    const int side = lane / ROW, rl = lane % ROW;
    const long long p = (at ? *at : pos) + first;
    const int k0 = b * SPLIT_KEYS, span = int(p) + tokens;   // the call's tokens attend rows [0, span)
    if (k0 >= span) return;
    const int nk = sycl::min(SPLIT_KEYS, span - k0);
    const int own = int(p) + t + 1;   // the token attends rows [0, own)
    const float* qt = q + ((size_t(first) + t) * n_head + g * G) * DIM + SCORE_VALUES * rl;
    sycl::float4 y[G];
#pragma unroll
    for (int i = 0; i < G; ++i)
        y[i] = *reinterpret_cast<const sycl::float4 *>(qt + i * DIM);
    for (int pair = w; ROWS * pair < nk; pair += WARPS) {   // every lane of the warp goes round alike
        const int r = ROWS * pair + side, key = k0 + r;
        float acc[G];
#pragma unroll
        for (int i = 0; i < G; ++i) acc[i] = 0.0f;
        if (r < nk && key < own) {
            const sycl::half *k =
                kcache + (size_t(key) * kv_heads + g) * DIM + SCORE_VALUES * rl;
            const unsigned long long bits = *reinterpret_cast<const unsigned long long*>(k);
            float x[SCORE_VALUES];
#pragma unroll
            for (int j = 0; j < SCORE_VALUES; ++j)
                x[j] = sycl::vec<sycl::half, 1>(
                           sycl::bit_cast<sycl::half, unsigned short>(
                               (unsigned short)(bits >> 16 * j)))
                           .convert<float, sycl::rounding_mode::automatic>()[0];
#pragma unroll
            for (int i = 0; i < G; ++i) {
                acc[i] += y[i].x() * x[0];
                acc[i] += y[i].y() * x[1];
                acc[i] += y[i].z() * x[2];
                acc[i] += y[i].w() * x[3];
            }
        }
        int which;
        const float sum = scores_sum<G, ROW / 2>(acc, lane, &which);
        if (r < nk && (rl & (ROW / G - 1)) == 0) e[which][r] = key < own ? sum * scale : -INFINITY;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    float m = -INFINITY;
#pragma unroll
    for (int j = lane; j < nk; j += WARP) m = sycl::fmax(m, e[w][j]);
#pragma unroll
    /*
    DPCT1108:54: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = WARP / 2; o > 0; o >>= 1) m = sycl::fmax(
        m, dpct::experimental::permute_sub_group_by_xor(
               0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
               m, o));
    float l = 0.0f;
    for (int j = lane; j < nk; j += WARP) {
        const float x =
            m == -INFINITY
                ? 0.0f
                : sycl::native::exp(e[w][j] -
                                    m); // a block the token sees none of: 0
        e[w][j] = x;
        l += x;
    }
    l = warp_sum(l);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // the values: warp w takes every WARPS-th row for all the heads, a lane VALS values of a row, so a row of the
    // cache is read once; then warp w adds the warps' sums of head w
    float o[G][VALS];
#pragma unroll
    for (int i = 0; i < G; ++i)
#pragma unroll
        for (int u = 0; u < VALS; ++u) o[i][u] = 0.0f;
    const sycl::half *v =
        vcache + (size_t(k0) * kv_heads + g) * DIM + VALS * lane;
    for (int j = w; j < nk; j += WARPS) {
#pragma unroll
        for (int u = 0; u < VALS; u += 2) {
            const sycl::float2 x =
                (*reinterpret_cast<const sycl::half2 *>(
                     v + size_t(j) * kv_heads * DIM + u))
                    .template convert<float, sycl::rounding_mode::automatic>();
#pragma unroll
            for (int i = 0; i < G; ++i) {
                o[i][u] += e[i][j] * x.x();
                o[i][u + 1] += e[i][j] * x.y();
            }
        }
    }
    auto &sums = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS][G][DIM]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
#pragma unroll
    for (int i = 0; i < G; ++i)
#pragma unroll
        for (int u = 0; u < VALS; ++u) sums[w][i][VALS * lane + u] = o[i][u];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    float r[VALS];
#pragma unroll
    for (int u = 0; u < VALS; ++u) r[u] = 0.0f;
#pragma unroll
    for (int k = 0; k < WARPS; ++k)
#pragma unroll
        for (int u = 0; u < VALS; ++u) r[u] += sums[k][w][VALS * lane + u];
    float* out = part + ((size_t(t) * n_head + g * G + w) * blocks + b) * (DIM + 2);
#pragma unroll
    for (int u = 0; u < VALS; ++u) out[VALS * lane + u] = r[u];
    if (lane == 0) {
        out[DIM] = m;
        out[DIM + 1] = l;
    }
}

// The blocks of attn_split_kernel added up for a head of a token: each block's sums scaled to the largest score.  A
// warp takes every JOIN_WARPS-th block, a lane DIM / 32 values, and the warps' sums are added in their order; a block
// past the token's own rows has none it attends and is skipped, so the sums are the same whatever tokens the call
// has.  A head's sink, with `sinks`, takes its share of the weight as one more score that weighs no value.
constexpr int JOIN_WARPS = 16;

template <int DIM>
/*
DPCT1110:55: The total declared local variable size in device function
attn_join_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
attn_join_kernel(const float *__restrict__ part, float *__restrict__ out,
                 int n_head, long long pos, int first, int tokens, int blocks,
                 const long long *__restrict__ at,
                 const float *__restrict__ sinks) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int VALS = DIM / WARP;
    auto &top =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[JOIN_WARPS]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sl =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[JOIN_WARPS]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &so = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[JOIN_WARPS][DIM]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int h = item_ct1.get_group(2), t = item_ct1.get_group(1),
              lane = item_ct1.get_local_id(2) % WARP,
              w = item_ct1.get_local_id(2) / WARP;
    const long long p = (at ? *at : pos) + first;
    const int used = int((p + tokens + SPLIT_KEYS - 1) / SPLIT_KEYS);   // the blocks the split kernel wrote
    const float* pb = part + (size_t(t) * n_head + h) * blocks * (DIM + 2);
    float mx = -INFINITY;
#pragma unroll
    for (int b = item_ct1.get_local_id(2); b < used; b += JOIN_WARPS * WARP)
        mx = sycl::fmax(mx, pb[size_t(b) * (DIM + 2) + DIM]);
    mx = block_max(mx, top);
    if (sinks) mx = sycl::fmax(mx, sinks[h]);
    float l = 0.0f, o[VALS];
#pragma unroll
    for (int u = 0; u < VALS; ++u) o[u] = 0.0f;
#pragma unroll 4
    for (int b = w; b < used; b += JOIN_WARPS) {
        const float* x = pb + size_t(b) * (DIM + 2);
        if (x[DIM] == -INFINITY) continue;
        const float f = sycl::native::exp(x[DIM] - mx);
        l += f * x[DIM + 1];
#pragma unroll
        for (int u = 0; u < VALS; ++u) o[u] += f * x[VALS * lane + u];
    }
#pragma unroll
    for (int u = 0; u < VALS; ++u) so[w][VALS * lane + u] = o[u];
    if (lane == 0) sl[w] = l;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (w != 0) return;
    float s[VALS], sum = 0.0f;
#pragma unroll
    for (int u = 0; u < VALS; ++u) s[u] = 0.0f;
#pragma unroll
    for (int k = 0; k < JOIN_WARPS; ++k) {
#pragma unroll
        for (int u = 0; u < VALS; ++u) s[u] += so[k][VALS * lane + u];
        sum += sl[k];
    }
    if (sinks) sum += sycl::native::exp(sinks[h] - mx);
    float* y = out + ((size_t(first) + t) * n_head + h) * DIM;
#pragma unroll
    for (int u = 0; u < VALS; ++u) y[VALS * lane + u] = s[u] / sum;
}

// attn_value_kernel with the weights as attn_softmax_kernel leaves them with `wt`: a row's queries side by side,
// so a warp reads a row's weights in two loads that every lane makes alike, where the other passes them from lane
// to lane, and so waited on the passing (Nsight Compute).  A block has TILES of the blocks of VALUE_QUERIES
// queries, which read the cache's values once between them.  The products go in the rows' order: the sums are
// the other kernel's to the bit.
template <int TILES, bool Q8 = false>
/*
DPCT1110:56: The total declared local variable size in device function
attn_value_t_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void attn_value_t_kernel(
    const float *__restrict__ wt, const sycl::half *__restrict__ vcache,
    float *__restrict__ out, int n_keys, int dim, int n_head, int kv_heads,
    long long pos, int first, int tokens, int window, int ring, long long last,
    const long long *__restrict__ at,
    const int8_t *__restrict__ vcodes = nullptr,
    const uint16_t *__restrict__ vscales = nullptr) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    static_assert(VALUE_DIMS == WARP && VALUE_QUERIES == 8,
                  "a warp is a share, and a row's weights are two float4");
    auto &share = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[VALUE_QUERIES][VALUE_SHARES][VALUE_DIMS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int per = n_head / kv_heads, dims = (dim + VALUE_DIMS - 1) / VALUE_DIMS;
    const int blocks = (tokens * per + VALUE_QUERIES - 1) / VALUE_QUERIES;   // the group's blocks of queries
    const int g = item_ct1.get_group(2) / dims,
              b0 = item_ct1.get_group(1) * TILES;
    const int lane = item_ct1.get_local_id(2) % VALUE_DIMS,
              s = item_ct1.get_local_id(2) / VALUE_DIMS;
    const int d = (item_ct1.get_group(2) % dims) * VALUE_DIMS + lane;
    const long long p = (at ? *at : pos) + first;
    last = attn_last(p, tokens, last);
    long long lo;
    const int span = attn_span(p, tokens, window, last, &lo);
    const size_t width = size_t(kv_heads) * dim;
    const sycl::half *v =
        vcache + size_t(g) * dim +
        sycl::min(d,
                  dim - 1); // a lane past the last value reads one all the same
    // the q8 read: the codes use the fp16 cache's index, the scales a 32-value block (width / 32, one a row's block)
    const int8_t *vc = vcodes + size_t(g) * dim + sycl::min(d, dim - 1);
    const uint16_t *vs =
        vscales + size_t(g) * (dim / 32) + sycl::min(d, dim - 1) / 32;
    const float* w[TILES];   // a tile past the group's last reads the block's first again, for nothing
#pragma unroll
    for (int t = 0; t < TILES; ++t)
        w[t] = wt + (size_t(g) * blocks + (b0 + t < blocks ? b0 + t : b0)) * n_keys * VALUE_QUERIES;
    float acc[TILES][VALUE_QUERIES];
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
        for (int i = 0; i < VALUE_QUERIES; ++i) acc[t][i] = 0.0f;
    long long row = ring > 0 ? (lo + s) % ring : lo + s;
    for (int j = s; j < span; j += VALUE_SHARES) {
        float x;
        if constexpr (Q8)
            x = (float)vc[size_t(row) * width] *
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        vs[size_t(row) * (width / 32)]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        else
            x = sycl::vec<sycl::half, 1>(v[size_t(row) * width])
                    .convert<float, sycl::rounding_mode::automatic>()[0];
#pragma unroll
        for (int t = 0; t < TILES; ++t) {
            const sycl::float4 a = *reinterpret_cast<const sycl::float4 *>(
                w[t] + size_t(j) * VALUE_QUERIES);
            const sycl::float4 b = *reinterpret_cast<const sycl::float4 *>(
                w[t] + size_t(j) * VALUE_QUERIES + 4);
            acc[t][0] += a.x() * x;
            acc[t][1] += a.y() * x;
            acc[t][2] += a.z() * x;
            acc[t][3] += a.w() * x;
            acc[t][4] += b.x() * x;
            acc[t][5] += b.y() * x;
            acc[t][6] += b.z() * x;
            acc[t][7] += b.w() * x;
        }
        row += VALUE_SHARES;
        if (ring > 0 && row >= ring) row -= ring;
    }
#pragma unroll
    for (int t = 0; t < TILES; ++t) {
        const int q0 = (b0 + t) * VALUE_QUERIES,
                  live = sycl::min(VALUE_QUERIES, tokens * per - q0);
        if (live <= 0) break;
        /*
        DPCT1118:58: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:203: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        if (t > 0)
            item_ct1.barrier(); // the tile before has its sums out of share
        for (int i = 0; i < live; ++i) {
            long long own;   // the first row of the query's token: the thread's rows are this share of the token's
            attn_rows(p + (q0 + i) / per, window, last, &own);
            share[i][(s + VALUE_SHARES - int((own - lo) % VALUE_SHARES)) % VALUE_SHARES][lane] = acc[t][i];
        }
        /*
        DPCT1118:57: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:202: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (s == 0 && d < dim)
            for (int i = 0; i < live; ++i) {
                float sum = share[i][0][lane];
#pragma unroll
                for (int k = 1; k < VALUE_SHARES; ++k) sum += share[i][k][lane];
                const int query = q0 + i;
                out[((size_t(first) + query / per) * n_head + g * per + query % per) * dim + d] = sum;
            }
    }
}

// A chunk's queries and its results lie in two orders: the engine's (token, head) and the one a product per
// group of heads takes (group, token, head of the group).  This is where the engine's `row` lies in the second.
__dpct_inline__ size_t attn_by_group(int row, int n_head, int kv_heads,
                                     int tokens) {
    const int per = n_head / kv_heads, t = row / n_head, h = row % n_head;
    return (size_t(h / per) * tokens + t) * per + h % per;
}

// The queries to the order of the groups, as the halves the products take.  blockIdx.y is the token and the head.
__dpct_inline__ void attn_group_kernel(const float *__restrict__ src,
                                       sycl::half *__restrict__ dst, int dim,
                                       int n_head, int kv_heads, int tokens) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= dim) return;
    const float x = src[size_t(item_ct1.get_group(1)) * dim + i];
    dst[attn_by_group(item_ct1.get_group(1), n_head, kv_heads, tokens) * dim +
        i] = sycl::vec<float, 1>(x)
                 .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

// The results back to the engine's order.
__dpct_inline__ void attn_ungroup_kernel(const float *__restrict__ src,
                                         float *__restrict__ dst, int dim,
                                         int n_head, int kv_heads, int tokens) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= dim) return;
    dst[size_t(item_ct1.get_group(1)) * dim + i] =
        src[attn_by_group(item_ct1.get_group(1), n_head, kv_heads, tokens) *
                dim +
            i];
}

// A softmax over a chunk's rows a block of rows at a time, for scores in the order of the groups: blockIdx.x
// counts (group, token, head of the group), and the block's rows start at row0 of the sequence.  A score's weight
// is its exp against the largest its token and head have had (top), zero for a row the token does not attend, as
// the half the values' product takes.  sum is the sum of the weights against that top, and shrink what the
// weights of the blocks before are worth against it.  With `fresh` there was no block before.
__dpct_inline__ void
attn_softmax_block_kernel(const float *__restrict__ score,
                          sycl::half *__restrict__ weight, int n_keys,
                          long long row0, long long pos, int first, int window,
                          int tokens, int per, long long last, float *top,
                          float *sum, float *shrink, bool fresh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int tok = first + (item_ct1.get_group(2) % (tokens * per)) / per;
    long long lo;
    const int n = attn_rows(pos + tok, window, last, &lo);
    const long long end = lo + n;   // the token attends the sequence's rows from lo to before end
    const int j0 = lo > row0 ? int(lo - row0) : 0, j1 = end - row0 < n_keys ? int(end - row0) : n_keys;
    const float *row = score + size_t(item_ct1.get_group(2)) * n_keys;
    sycl::half *w = weight + size_t(item_ct1.get_group(2)) * n_keys;
    float mx = -INFINITY;
#pragma unroll
    for (int j = j0 + item_ct1.get_local_id(2); j < j1;
         j += item_ct1.get_local_range(2)) mx = sycl::fmax(mx, row[j]);
    mx = block_max(mx, part);
    const float before = fresh ? -INFINITY : top[item_ct1.get_group(2)],
                now = sycl::fmax(before, mx);
    float s = 0.0f;
#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < n_keys;
         j += item_ct1.get_local_range(2)) {
        w[j] = sycl::vec<float, 1>(
                   j >= j0 && j < j1 ? sycl::native::exp(row[j] - now) : 0.0f)
                   .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
        s += sycl::vec<sycl::half, 1>(w[j])
                 .convert<float, sycl::rounding_mode::automatic>()
                     [0]; // the sum is of the weights as the values get them
    }
    s = block_sum(s, part);
    if (item_ct1.get_local_id(2) != 0) return;
    const float f =
        mx == -INFINITY
            ? 1.0f
            : sycl::native::exp(
                  before - now); // none of the block's rows: nothing changes
    top[item_ct1.get_group(2)] = now;
    sum[item_ct1.get_group(2)] =
        (fresh ? 0.0f : sum[item_ct1.get_group(2)] * f) + s;
    shrink[item_ct1.get_group(2)] = f;
}

// A head's sink in a token's softmax sum, against the top its rows reached: a logit with no value of its own, which
// only takes weight from the rows.  b counts as attn_softmax_block_kernel's blocks do, a group's columns together.
__dpct_inline__ void attn_sink_sum_kernel(float *sum,
                                          const float *__restrict__ top,
                                          const float *__restrict__ sinks,
                                          int n, int per, int cols) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int b = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (b >= n) return;
    sum[b] += sycl::native::exp(sinks[(b / cols) * per + b % per] - top[b]);
}

// A chunk's results in the order of the groups, `dim` values a token and head: times its factor, or divided by it.
__dpct_inline__ void attn_scale_kernel(float *out,
                                       const float *__restrict__ factor,
                                       int dim, bool divide) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= dim) return;
    const float f = factor[item_ct1.get_group(1)];
    // a row whose largest score this block of keys did not raise is times 1: it is not read or written
    if (!divide && f == 1.0f) return;
    out[size_t(item_ct1.get_group(1)) * dim + i] *= divide ? 1.0f / f : f;
}

#if defined(STRATA_USE_HIP)
// STRATA_HIP_PREFILL_FP32 (attend_batch()): the same products on FP32 inputs.  On gfx1010 hipBLAS's FP32 products
// run at about 4.5 to 5 TFLOPS where its FP16-input, FP32-sum ones reach 3 to 4 (there is no packed FP16 math with an
// FP32 sum on the card), so the queries, the dequantized keys and values and the softmax's weights stay FP32.
__global__ void attn_group_f32_kernel(const float* __restrict__ src, float* __restrict__ dst, int dim, int n_head,
                                      int kv_heads, int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    dst[attn_by_group(blockIdx.y, n_head, kv_heads, tokens) * dim + i] = src[size_t(blockIdx.y) * dim + i];
}

__global__ void dequant_q8_f32_kernel(const int8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                                      float* __restrict__ out, int width, long long row) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const size_t r = (size_t) (row + blockIdx.y), at = r * width;
    const float d = __half2float(__ushort_as_half(scales[r * (width >> 5) + (i >> 5)]));
    out[(size_t) blockIdx.y * width + i] = (float) codes[at + i] * d;
}

// attn_softmax_block_kernel with the weights as floats, and their sum the sum of those.
__global__ void attn_softmax_block_f32_kernel(const float* __restrict__ score, float* __restrict__ weight, int n_keys,
                                              int ld, long long row0, long long pos, int first, int window,
                                              int tokens, int per, long long last, float* top, float* sum,
                                              float* shrink, bool fresh) {
    __shared__ float part[8];
    const int tok = first + (blockIdx.x % (tokens * per)) / per;
    long long lo;
    const int n = attn_rows(pos + tok, window, last, &lo);
    const long long end = lo + n;
    const int j0 = lo > row0 ? int(lo - row0) : 0, j1 = end - row0 < n_keys ? int(end - row0) : n_keys;
    const float* row = score + size_t(blockIdx.x) * ld;   // a row of the block's columns lies `ld` floats on
    float* w = weight + size_t(blockIdx.x) * ld;
    float mx = -INFINITY;
    for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) mx = fmaxf(mx, row[j]);
    mx = block_max(mx, part);
    const float before = fresh ? -INFINITY : top[blockIdx.x], now = fmaxf(before, mx);
    float s = 0.0f;
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x) {
        const float e = j >= j0 && j < j1 ? expf(row[j] - now) : 0.0f;
        w[j] = e;
        s += e;
    }
    s = block_sum(s, part);
    if (threadIdx.x != 0) return;
    const float f = mx == -INFINITY ? 1.0f : expf(before - now);
    top[blockIdx.x] = now;
    sum[blockIdx.x] = (fresh ? 0.0f : sum[blockIdx.x] * f) + s;
    shrink[blockIdx.x] = f;
}
#endif

// One m16n8k32 on the int8 tensor cores: A is [16][32] row-major (a query's codes), B is [8][32] (a key's codes,
// and that is the column-major 32x8 the instruction asks for), C comes back as a 16x8 block of int32 in four
// registers.  Both caches already lie in exactly these shapes, so nothing has to be shuffled into fragments.
#if defined(__HIPCC__) ||                                                      \
    (defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800)
// ROCm and cards before Ampere have no mma.sync.aligned.m16n8k32.s8 (ptxas refuses it below sm_80): attend_int8
// never launches the tensor-core tier there (int8_tensor_cores()), so the body is never run.
__device__ __forceinline__ void mma_s8(int&, int&, int&, int&, unsigned, unsigned, unsigned,
                                       unsigned, unsigned, unsigned) {}
#else
__dpct_inline__ void mma_s8(int &c0, int &c1, int &c2, int &c3, unsigned a0,
                            unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                            unsigned b1) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    {
        volatile void *d_mat_frag_ct1[4] = {&c0, &c1, &c2, &c3};
        sycl::vec<uint32_t, 4> a_mat_frag_ct1(a0, a1, a2, a3);
        sycl::vec<uint32_t, 2> b_mat_frag_ct1(b0, b1);
        sycl::vec<int32_t, 4> c_mat_frag_ct1(c0, c1, c2, c3);
        dpct::experimental::matrix::mma<16, 8, 32, int8_t, int32_t>(
            reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1,
            &b_mat_frag_ct1, &c_mat_frag_ct1);
    }
#endif
}
#endif
// attend_batch(): the bytes of a chunk's results, which every block of rows reads and writes once more: in the

// The tensor cores' m16n8k16 step and ldmatrix need Turing; AMD builds have neither, and the host keeps the cuBLAS
// products there (tensor_cores()).  Turing runs the step as two of k = 8 on the same registers.
#if 1   // SYCL port: no tensor-core (mma/ldmatrix) path
#define STRATA_POLY_MMA 0
#else
#define STRATA_POLY_MMA 1
#endif

#if STRATA_POLY_MMA
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

// 16 bytes from global to shared memory, or zeros where `whole` is false; asynchronous from Ampere on (cp.async: a
// thread's copies are a group from flash_commit() on, waited for by flash_wait()), a plain load and store on Turing.
__dpct_inline__ void flash_copy(sycl::half *dst, const sycl::half *src,
                                bool whole) {
#if 0   // SYCL port: no cp.async
    /*
    DPCT1053:59: Migration of device assembly code is not supported.
    */
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(
                     (unsigned)__cvta_generic_to_shared(dst)),
                 "l"(src), "r"(whole ? 16 : 0));
#else
    *reinterpret_cast<uint4*>(dst) = whole ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
#endif
}

__dpct_inline__ void flash_commit() {
#if 0   // SYCL port: no cp.async
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.commit_group;\n" ::);
#else

#endif
#endif
}

// Wait until at most `ahead` of the thread's groups are still on their way.
template <int ahead> __dpct_inline__ void flash_wait() {
#if 0   // SYCL port: no cp.async
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(ahead));
#else

#endif
#endif
}
#endif

// attend_batch() for heads of 64 values in groups of 4 (LFM2's), the way FlashAttention-2 goes: a launch block takes
// FLASH_TOKENS tokens of a group, 64 rows of queries with a token's 4 heads side by side, a warp 16 of them, and walks
// the keys FLASH_KEYS at a time on the tensor cores, keeping each row's softmax as it goes, so no score leaves the
// registers.  The queries and the weights are rounded to halves and the products sum floats, as with the cuBLAS
// products.  Fragment layouts are PTX's for mma.m16n8k16: a thread holds rows gid and gid + 8, columns 2 tig and
// 2 tig + 1 of each block of 8.  The next block of keys and values is copied in while one is worked on, two buffers
// in turn.
constexpr int FLASH_TOKENS = 16;
/*
DPCT1110:62: The total declared local variable size in device function
attn_flash64_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void attn_flash64_kernel(const float *__restrict__ q,
                                         const sycl::half *__restrict__ kcache,
                                         const sycl::half *__restrict__ vcache,
                                         float *__restrict__ out, int rows,
                                         int n_head, int kv_heads,
                                         long long pos, float scale_log2) {
#if STRATA_POLY_MMA
    // FLASH_ROW: a cache row in shared memory, in halves; 16-byte aligned, and ldmatrix's eight rows on different
    // banks.
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int DIM = 64, G = 4, FLASH_KEYS = 64, FLASH_ROW = 64 + 8;
    auto &ks = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        sycl::half[2][FLASH_KEYS][FLASH_ROW]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &vs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        sycl::half[2][FLASH_KEYS][FLASH_ROW]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int g = item_ct1.get_group(1),
              t0 = item_ct1.get_group(2) * FLASH_TOKENS,
              lane = item_ct1.get_local_id(2) % WARP,
              w = item_ct1.get_local_id(2) / WARP;
    const int gid = lane / 4, tig = lane % 4;
    int tok[2], head[2];   // the thread's rows gid and gid + 8 of the warp's
    for (int i = 0; i < 2; ++i) {
        const int r = w * 16 + gid + 8 * i;
        tok[i] = t0 + r / G;
        head[i] = g * G + r % G;
    }
    uint32_t qa[4][4];   // a block of 16 values a step, in log2 units
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
        for (int i = 0; i < 2; ++i)
            for (int hi = 0; hi < 2; ++hi) {
                sycl::float2 x = sycl::float2(0.0f, 0.0f);
                if (tok[i] < rows)
                    x = *reinterpret_cast<const sycl::float2 *>(
                        q + (size_t(tok[i]) * n_head + head[i]) * DIM + 16 * k +
                        8 * hi + 2 * tig);
                qa[k][i + 2 * hi] =
                    half2_bits(x.x() * scale_log2, x.y() * scale_log2);
            }
    float o[8][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
#pragma unroll
    for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    const int n_keys =
        int(pos) +
        sycl::min(t0 + FLASH_TOKENS,
                  rows); // the block's last token attends rows [0, n_keys)
    const auto fetch = [&](int buf, int k0) {
        for (int i = sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                         .get_local_id(2);
             i < FLASH_KEYS * DIM / 8; i += 4 * WARP) {
            const int r = i / (DIM / 8), c = 8 * (i % (DIM / 8));
            const bool whole = k0 + r < n_keys;
            const size_t at = whole ? (size_t(k0 + r) * kv_heads + g) * DIM + c : 0;
            flash_copy(&ks[buf][r][c], kcache + at, whole);
            flash_copy(&vs[buf][r][c], vcache + at, whole);
        }
        flash_commit();
    };
    fetch(0, 0);
    for (int k0 = 0, buf = 0; k0 < n_keys; k0 += FLASH_KEYS, buf ^= 1) {
        /*
        DPCT1118:63: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(
            sycl::access::fence_space::local_space); // the buffer the last
                                                     // block was worked on in
                                                     // is free
        if (k0 + FLASH_KEYS < n_keys) {
            fetch(buf ^ 1, k0 + FLASH_KEYS);
            flash_wait<1>();
        } else {
            flash_wait<0>();
        }
        /*
        DPCT1118:64: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        float s[8][4];
        for (int n = 0; n < 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            for (int k = 0; k < 4; k += 2) {
                uint32_t b[4];   // keys 8 n.., values 16 k.. and 16 (k + 1)..
                ldmatrix_x4(b, &ks[buf][8 * n + lane % 8][16 * k + 8 * (lane / 8)], false);
                mma16816(s[n], qa[k], b);
                mma16816(s[n], qa[k + 1], b + 2);
            }
        }
        for (int i = 0; i < 2; ++i) {
            const int own = int(pos) + tok[i];   // the row's token attends rows [0, own]
            float mx = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n)
#pragma unroll
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    if (k0 + 8 * n + 2 * tig + j % 2 > own) s[n][j] = -INFINITY;
                    mx = sycl::fmax(mx, s[n][j]);
                }
            /*
            DPCT1108:65: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            mx = sycl::fmax(
                mx,
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), mx, 1));
            /*
            DPCT1108:66: '__shfl_xor_sync' was migrated with the experimental
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
            for (int n = 0; n < 8; ++n) {
                o[n][2 * i] *= f;
                o[n][2 * i + 1] *= f;
#pragma unroll
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    s[n][j] = sycl::exp2(s[n][j] - base);
                    l[i] += s[n][j];
                }
            }
        }
        for (int k = 0; k < 4; ++k) {   // keys 16 k..: the weights as an A fragment
            const uint32_t pa[4] = {half2_bits(s[2 * k][0], s[2 * k][1]), half2_bits(s[2 * k][2], s[2 * k][3]),
                                    half2_bits(s[2 * k + 1][0], s[2 * k + 1][1]),
                                    half2_bits(s[2 * k + 1][2], s[2 * k + 1][3])};
            for (int n = 0; n < 8; n += 2) {
                uint32_t b[4];   // keys 16 k.., values 8 n.. and 8 (n + 1)..
                ldmatrix_x4(b, &vs[buf][16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                mma16816(o[n], pa, b);
                mma16816(o[n + 1], pa, b + 2);
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        /*
        DPCT1108:67: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        l[i] += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            l[i], 1);
        /*
        DPCT1108:68: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        l[i] += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            l[i], 2);
        if (tok[i] >= rows) continue;
        float* y = out + (size_t(tok[i]) * n_head + head[i]) * DIM + 2 * tig;
#pragma unroll
        for (int n = 0; n < 8; ++n)
            *reinterpret_cast<sycl::float2 *>(y + 8 * n) =
                sycl::float2(o[n][2 * i] / l[i], o[n][2 * i + 1] / l[i]);
    }
#else
    (void) q, (void) kcache, (void) vcache, (void) out, (void) rows, (void) n_head, (void) kv_heads, (void) pos,
        (void) scale_log2;   // a card without the tensor cores' steps never launches it (tensor_cores())
#endif
}

// attend_batch() for heads of 128 values in groups of 8 (Flint's), the way attn_flash64_kernel goes, with a window
// over a ring of rows and a sink a head: a launch block takes FLASH128_TOKENS tokens of a group, 64 rows of queries
// with a token's 8 heads side by side, a warp two tokens, and walks the keys 64 at a time.  The walk starts at a
// multiple of 64, so a token's keys go in the same blocks whatever tokens the call has beside it: a block it attends
// none of changes none of its sums.  The two buffers of keys and values take 68 KiB, past the 48 a block has without
// asking: the launch asks for them (attend_batch()).
constexpr int FLASH128_TOKENS = 8, FLASH128_KEYS = 64, FLASH128_ROW = 128 + 8;
/*
DPCT1083:107: The size of local memory in the migrated code may be different
from the original code. Check that the allocated memory size in the migrated
code is correct.
*/
constexpr int FLASH128_SMEM =
    2 * 2 * FLASH128_KEYS * FLASH128_ROW * int(sizeof(sycl::half));

/*
DPCT1110:69: The total declared local variable size in device function
attn_flash128_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void attn_flash128_kernel(
    const float *__restrict__ q, const sycl::half *__restrict__ kcache,
    const sycl::half *__restrict__ vcache, float *__restrict__ out, int rows,
    int n_head, int kv_heads, long long pos, float scale_log2, int window,
    int ring, const float *__restrict__ sinks, uint8_t *dpct_local) {
#if STRATA_POLY_MMA
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int DIM = 128, G = 8, KEYS = FLASH128_KEYS, ROW = FLASH128_ROW;
    auto flash_sm = (sycl::half *)dpct_local;
    sycl::half(*ks)[KEYS][ROW] =
        reinterpret_cast<sycl::half(*)[KEYS][ROW]>(flash_sm);
    sycl::half(*vs)[KEYS][ROW] =
        reinterpret_cast<sycl::half(*)[KEYS][ROW]>(flash_sm + 2 * KEYS * ROW);
    const int g = item_ct1.get_group(1),
              t0 = item_ct1.get_group(2) * FLASH128_TOKENS,
              lane = item_ct1.get_local_id(2) % WARP,
              w = item_ct1.get_local_id(2) / WARP;
    const int gid = lane / 4, tig = lane % 4, head = g * G + gid;
    const int tok[2] = {t0 + 2 * w, t0 + 2 * w + 1};   // the thread's rows gid and gid + 8: a head of two tokens
    uint32_t qa[8][4];   // a block of 16 values a step, in log2 units
#pragma unroll
    for (int k = 0; k < 8; ++k)
#pragma unroll
        for (int i = 0; i < 2; ++i)
            for (int hi = 0; hi < 2; ++hi) {
                sycl::float2 x = sycl::float2(0.0f, 0.0f);
                if (tok[i] < rows)
                    x = *reinterpret_cast<const sycl::float2 *>(
                        q + (size_t(tok[i]) * n_head + head) * DIM + 16 * k +
                        8 * hi + 2 * tig);
                qa[k][i + 2 * hi] =
                    half2_bits(x.x() * scale_log2, x.y() * scale_log2);
            }
    float o[16][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    // the block's tokens attend positions [lo, hi): from the first one's window to the last one
    const long long hi = pos + sycl::min(t0 + FLASH128_TOKENS, rows);
    const long long lo = window > 0 ? sycl::max(0LL, pos + t0 - window + 1) : 0;
    const auto fetch = [&](int buf, long long k0) {
        for (int i = sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                         .get_local_id(2);
             i < KEYS * DIM / 8; i += 4 * WARP) {
            const int r = i / (DIM / 8), c = 8 * (i % (DIM / 8));
            const long long key = k0 + r;
            const bool whole = key >= lo && key < hi;   // the others are zeros, which no weight reads
            const size_t at = whole ? (size_t(ring > 0 ? key % ring : key) * kv_heads + g) * DIM + c : 0;
            flash_copy(&ks[buf][r][c], kcache + at, whole);
            flash_copy(&vs[buf][r][c], vcache + at, whole);
        }
        flash_commit();
    };
    const long long first = lo / KEYS * KEYS;
    fetch(0, first);
    int buf = 0;
    for (long long k0 = first; k0 < hi; k0 += KEYS, buf ^= 1) {
        /*
        DPCT1118:70: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(
            sycl::access::fence_space::local_space); // the buffer the last
                                                     // block was worked on in
                                                     // is free
        if (k0 + KEYS < hi) {
            fetch(buf ^ 1, k0 + KEYS);
            flash_wait<1>();
        } else {
            flash_wait<0>();
        }
        /*
        DPCT1118:71: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        float s[8][4];
        for (int n = 0; n < 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            for (int k = 0; k < 8; k += 2) {
                uint32_t b[4];   // keys 8 n.., values 16 k.. and 16 (k + 1)..
                ldmatrix_x4(b, &ks[buf][8 * n + lane % 8][16 * k + 8 * (lane / 8)], false);
                mma16816(s[n], qa[k], b);
                mma16816(s[n], qa[k + 1], b + 2);
            }
        }
        for (int i = 0; i < 2; ++i) {
            const long long own = pos + tok[i];   // the row's token attends positions (own - window, own]
            float mx = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n)
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    const long long key = k0 + 8 * n + 2 * tig + j % 2;
                    if (key > own || (window > 0 && key <= own - window)) s[n][j] = -INFINITY;
                    mx = sycl::fmax(mx, s[n][j]);
                }
            /*
            DPCT1108:72: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            mx = sycl::fmax(
                mx,
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), mx, 1));
            /*
            DPCT1108:73: '__shfl_xor_sync' was migrated with the experimental
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
            for (int n = 0; n < 8; ++n)
#pragma unroll
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    s[n][j] = sycl::exp2(s[n][j] - base);
                    l[i] += s[n][j];
                }
        }
        for (int k = 0; k < 4; ++k) {   // keys 16 k..: the weights as an A fragment
            const uint32_t pa[4] = {half2_bits(s[2 * k][0], s[2 * k][1]), half2_bits(s[2 * k][2], s[2 * k][3]),
                                    half2_bits(s[2 * k + 1][0], s[2 * k + 1][1]),
                                    half2_bits(s[2 * k + 1][2], s[2 * k + 1][3])};
            for (int n = 0; n < 16; n += 2) {
                uint32_t b[4];   // keys 16 k.., values 8 n.. and 8 (n + 1)..
                ldmatrix_x4(b, &vs[buf][16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                mma16816(o[n], pa, b);
                mma16816(o[n + 1], pa, b + 2);
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        /*
        DPCT1108:74: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        l[i] += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            l[i], 1);
        /*
        DPCT1108:75: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        l[i] += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            l[i], 2);
        float f = 1.0f;
        if (sinks) {   // the sink, in log2 units like the scores, as one more weight of no value
            const float sink = sinks[head] * 1.44269504f,
                        now = sycl::fmax(m[i], sink);
            f = sycl::exp2(m[i] - now);
            l[i] = l[i] * f + sycl::exp2(sink - now);
        }
        if (tok[i] >= rows) continue;
        float* y = out + (size_t(tok[i]) * n_head + head) * DIM + 2 * tig;
#pragma unroll
        for (int n = 0; n < 16; ++n)
            *reinterpret_cast<sycl::float2 *>(y + 8 * n) = sycl::float2(
                o[n][2 * i] * f / l[i], o[n][2 * i + 1] * f / l[i]);
    }
#else
    (void) q, (void) kcache, (void) vcache, (void) out, (void) rows, (void) n_head, (void) kv_heads, (void) pos,
        (void) scale_log2, (void) window, (void) ring, (void) sinks;   // never launched without the tensor cores
#endif
}

__dpct_inline__ void softcap_kernel(float *x, size_t n, float cap) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        size_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) x[i] = sycl::tanh(x[i] * (1.0f / cap)) * cap;
}

// One block of 1024; a thread has every 1024th value.
__dpct_inline__ void likeliest_kernel(const float *__restrict__ x, int n,
                                      Likeliest *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &first = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    float m = -INFINITY;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) m = sycl::fmax(m, x[i]);
    m = block_max(m, part);
    int at = n;
    float sum = 0.0f;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) {
        if (x[i] == m) at = sycl::min(at, i);
        if (x[i] > m - 20.0f) sum += sycl::native::exp(x[i] - m);
    }
#pragma unroll
    /*
    DPCT1108:76: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = WARP / 2; o > 0; o >>= 1) at = sycl::min(
        at, dpct::experimental::permute_sub_group_by_xor(
                0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(),
                at, o));
    if ((item_ct1.get_local_id(2) & 31) == 0)
        first[item_ct1.get_local_id(2) >> 5] = at;
    sum = block_sum(sum, part);   // and first is written
    if (item_ct1.get_local_id(2) != 0) return;
#pragma unroll
    for (int k = 1; k < int(item_ct1.get_local_range(2)) / WARP; ++k)
        at = sycl::min(at, first[k]);
    out->token = at < n ? at : 0;
    out->prob = 1.0f / sum;
}

__dpct_inline__ void mul_sigmoid_kernel(float *__restrict__ acc,
                                        const float *__restrict__ g, int n,
                                        int g_stride, bool one_gate) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    acc[size_t(item_ct1.get_group(1)) * n + i] *= sigmoid(
        g[size_t(item_ct1.get_group(1)) * g_stride + (one_gate ? 0 : i)]);
}

// Where a token leaves the recurrent state as it is after it: `snaps` has a bit for each of a call's tokens that
// does, and their states lie in the tokens' order, `stride` floats apart.  Null for a token that leaves none.
__dpct_inline__ float *snap_of(float *snap, size_t stride, unsigned snaps,
                               int row) {
    return row < 32 && (snaps >> row & 1u)
               ? snap +
                     size_t(sycl::popcount(snaps & ((1u << row) - 1u))) * stride
               : nullptr;
}

// Three past inputs per channel, then the new one.  A thread is a channel and walks the tokens.  The state as it
// is after a token of `snaps` goes to snap (snap_of()).
__dpct_inline__ void conv4_silu_kernel(const float *__restrict__ x,
                                       float *__restrict__ state,
                                       const float *__restrict__ w,
                                       float *__restrict__ out, int channels,
                                       int rows, float *__restrict__ snap,
                                       size_t stride, unsigned snaps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= channels) return;
    const float* wc = w + 4 * size_t(c);
    float* st = state + size_t(c) * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    for (int row = 0; row < rows; ++row) {
        const size_t at = size_t(row) * channels + c;
        const float v = x[at];
        const float y = s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + v * wc[3];
        s0 = s1;
        s1 = s2;
        s2 = v;
        out[at] = y * sigmoid(y);
        if (float* m = snap_of(snap, stride, snaps, row)) {
            m += size_t(c) * 3;
            m[0] = s0;
            m[1] = s1;
            m[2] = s2;
        }
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
}

// From this many tokens a causal convolution takes them all at once: a walk is one launch, and the two launches of
// the parallel way cost a token alone more than its walk.
constexpr int CONV_ROWS = 16;

// conv4_silu_kernel for a call that leaves no snapshots: a thread a channel of a token, the inputs before the call's
// first token from the state, the same sum in the same order.  A thread that walked a channel's tokens waited for
// each token's input in turn: most of the time Qwen3.6's recurrent layers took to read a prompt.
__dpct_inline__ void conv4_silu_rows_kernel(const float *__restrict__ x,
                                            const float *__restrict__ state,
                                            const float *__restrict__ w,
                                            float *__restrict__ out,
                                            int channels) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= channels) return;
    const int row = item_ct1.get_group(1);
    const float* wc = w + 4 * size_t(c);
    const float* st = state + size_t(c) * 3;
    auto in = [&](int r) { return r >= 0 ? x[size_t(r) * channels + c] : st[3 + r]; };
    const float s0 = in(row - 3), s1 = in(row - 2), s2 = in(row - 1), v = x[size_t(row) * channels + c];
    const float y = s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + v * wc[3];
    out[size_t(row) * channels + c] = y * sigmoid(y);
}

// LFM2's short convolution.  A token's in_proj row is B, C and x, `channels` values each; the convolution runs over
// B * x with three taps, and C scales its output.  The state is the two B * x before the call, oldest first.  A
// thread is a channel and walks the tokens; the state as it is after a token of `snaps` goes to snap.
__dpct_inline__ void shortconv_kernel(const float *__restrict__ bcx,
                                      float *__restrict__ state,
                                      const float *__restrict__ w,
                                      float *__restrict__ out, int channels,
                                      int rows, float *__restrict__ snap,
                                      size_t stride, unsigned snaps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= channels) return;
    const float* wc = w + 3 * size_t(c);
    float* st = state + size_t(c) * 2;
    float s0 = st[0], s1 = st[1];
    for (int row = 0; row < rows; ++row) {
        const float* in = bcx + size_t(row) * 3 * channels;
        const float bx = in[c] * in[2 * channels + c];
        const float y = s0 * wc[0] + s1 * wc[1] + bx * wc[2];
        s0 = s1;
        s1 = bx;
        out[size_t(row) * channels + c] = in[channels + c] * y;
        if (float* m = snap_of(snap, stride, snaps, row)) {
            m[size_t(c) * 2] = s0;
            m[size_t(c) * 2 + 1] = s1;
        }
    }
    st[0] = s0;
    st[1] = s1;
}

// shortconv_kernel for a call that leaves no snapshots: a thread a channel of a token, as conv4_silu_rows_kernel.
__dpct_inline__ void shortconv_rows_kernel(const float *__restrict__ bcx,
                                           const float *__restrict__ state,
                                           const float *__restrict__ w,
                                           float *__restrict__ out,
                                           int channels) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= channels) return;
    const int row = item_ct1.get_group(1);
    const float* wc = w + 3 * size_t(c);
    auto bx = [&](int r) {
        if (r < 0) return state[size_t(c) * 2 + 2 + r];
        const float* in = bcx + size_t(r) * 3 * channels;
        return in[c] * in[2 * channels + c];
    };
    const float y = bx(row - 2) * wc[0] + bx(row - 1) * wc[1] + bx(row) * wc[2];
    out[size_t(row) * channels + c] = bcx[size_t(row) * 3 * channels + channels + c] * y;
}

// The short convolution's state after the call: its last two B * x, the state's own where the call has fewer.
__dpct_inline__ void shortconv_state_kernel(const float *__restrict__ bcx,
                                            float *__restrict__ state,
                                            int channels, int rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= channels) return;
    float* st = state + size_t(c) * 2;
    auto bx = [&](int r) {
        if (r < 0) return st[2 + r];
        const float* in = bcx + size_t(r) * 3 * channels;
        return in[c] * in[2 * channels + c];
    };
    const float a = bx(rows - 2), b = bx(rows - 1);
    st[0] = a;
    st[1] = b;
}

// The state after the call: its last three inputs, the state's own where the call has fewer.
__dpct_inline__ void conv4_state_kernel(const float *__restrict__ x,
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

// The gated delta net's step, in four launches.  A token's queries and keys are normed without the state, its
// decay and its share of the error are a head's alone, and its output is normed without the tokens after it: all
// of that is done for every token at once, and only the state's update walks the tokens.
//
// One block a token and key head: the head's q and k, normed in place.
__dpct_inline__ void gdn_norm_kernel(float *__restrict__ qkv, int dim,
                                     int k_heads, size_t row_n,
                                     float norm_eps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int hk = item_ct1.get_group(2) % k_heads,
              j = item_ct1.get_local_id(2);
    float *x = qkv + size_t(item_ct1.get_group(2) / k_heads) * row_n;
    float* qp = x + size_t(hk) * dim + j;
    float* kp = x + size_t(dim) * k_heads + size_t(hk) * dim + j;
    const float q = *qp, k = *kp;
    const float qs = block_sum(q * q, part);
    const float ks = block_sum(k * k, part);
    *qp = q / sycl::sqrt(qs + norm_eps) * (1.0f / sycl::sqrt(float(dim)));
    *kp = k / sycl::sqrt(ks + norm_eps);
}

// A token's decay and its share of the error for each head, in place of alpha and beta.
__dpct_inline__ void gdn_gates_kernel(float *__restrict__ alpha,
                                      float *__restrict__ beta,
                                      const float *__restrict__ dt,
                                      const float *__restrict__ a, int n_head,
                                      int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int at = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                   item_ct1.get_local_id(2);
    if (at >= n) return;
    const int h = at % n_head;
    const float g = alpha[at] + dt[h];
    alpha[at] = sycl::native::exp(
        (g > 20.0f ? g : sycl::log(1.0f + sycl::native::exp(g))) * a[h]);
    beta[at] = sigmoid(beta[at]);
}

#ifdef STRATA_POLY_GDN_MMA
__global__ void gdn_log_gates_kernel(float* __restrict__ alpha, float* __restrict__ beta, const float* __restrict__ dt,
                                 const float* __restrict__ a, int n_head, int n) {
    const int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= n) return;
    const int h = at % n_head;
    const float g = alpha[at] + dt[h];
    alpha[at] = (g > 20.0f ? g : logf(1.0f + expf(g))) * a[h];
    beta[at] = sigmoid(beta[at]);
}

#endif

// The state's update, a warp a column: the blocks of a head share its columns among their warps, and a lane holds
// every 32nd value of its warp's column, so a column is read and written as whole lines and its two sums are a
// few terms a lane and a sum over the warp.  The block walks the tokens, and no warp waits for another.
// s[i + dim * j] holds key dimension i of column j.  out gets the output before its norm.  decay and share are
// what gdn_gates_kernel left.  dim is a multiple of 32.
__dpct_inline__ void
gdn_state_kernel(const float *__restrict__ qkv, const float *__restrict__ decay,
                 const float *__restrict__ share, float *__restrict__ s,
                 float *__restrict__ out, int dim, int n_head, int k_heads,
                 int rows, float *__restrict__ snap, size_t stride,
                 unsigned snaps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int per_head =
        dim / int(item_ct1.get_local_range(2) / WARP); // blocks a head
    const int h = item_ct1.get_group(2) / per_head, hk = h % k_heads;
    const int j = (item_ct1.get_group(2) % per_head) *
                      int(item_ct1.get_local_range(2) / WARP) +
                  int(item_ct1.get_local_id(2) / WARP);
    const int lane = item_ct1.get_local_id(2) & (WARP - 1);
    if (j >= dim) return;   // a whole warp: the last block of a head whose columns do not fill it
    const size_t keys = size_t(dim) * k_heads, d_inner = size_t(dim) * n_head, row_n = 2 * keys + d_inner;
    const size_t mine = (size_t(h) * dim + j) * dim;
    float* col = s + mine;
    for (int row = 0; row < rows; ++row) {
        const float* x = qkv + size_t(row) * row_n;
        const float* qn = x + size_t(hk) * dim;
        const float* kn = x + keys + size_t(hk) * dim;
        const float v = x[2 * keys + size_t(h) * dim + j];
        const float dc = decay[size_t(row) * n_head + h];
        float sk = 0.0f;
#pragma unroll
        for (int i = lane; i < dim; i += WARP) sk += col[i] * dc * kn[i];
        sk = warp_sum(sk);
        const float d = (v - sk) * share[size_t(row) * n_head + h];
        float o = 0.0f;
        float* m = snap_of(snap, stride, snaps, row);
        if (m) m += mine;
        for (int i = lane; i < dim; i += WARP) {
            const float n = col[i] * dc + kn[i] * d;
            col[i] = n;
            if (m) m[i] = n;
            o += n * qn[i];
        }
        o = warp_sum(o);
        if (lane == 0) out[size_t(row) * d_inner + size_t(h) * dim + j] = o;
    }
}

// gdn_state_kernel with the warp's column in registers, PER values a lane, for the whole walk over the tokens:
// a token reads and writes no state.  The sums go in the same order, so the result is the same to the bit.  A
// token's inputs are read while the token before is computed: a walk that waited for each token's reads took
// about a microsecond a token, the latency of those reads.  A block is 32 warps, so the bound: without it PER 8
// took 79 registers a thread on sm_70 and 71 on sm_75, more than a 1024-thread launch can have.
template <int PER>
/*
DPCT1110:77: The total declared local variable size in device function
gdn_state_reg_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gdn_state_reg_kernel(
    const float *__restrict__ qkv, const float *__restrict__ decay,
    const float *__restrict__ share, float *__restrict__ s,
    float *__restrict__ out, int dim, int n_head, int k_heads, int rows,
    float *__restrict__ snap, size_t stride, unsigned snaps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int per_head = dim / int(item_ct1.get_local_range(2) / WARP);
    const int h = item_ct1.get_group(2) / per_head, hk = h % k_heads;
    const int j = (item_ct1.get_group(2) % per_head) *
                      int(item_ct1.get_local_range(2) / WARP) +
                  int(item_ct1.get_local_id(2) / WARP);
    const int lane = item_ct1.get_local_id(2) & (WARP - 1);
    if (j >= dim) return;
    const size_t keys = size_t(dim) * k_heads, d_inner = size_t(dim) * n_head, row_n = 2 * keys + d_inner;
    const size_t mine = (size_t(h) * dim + j) * dim;
    float* col = s + mine;
    float c[PER];
#pragma unroll
    for (int k = 0; k < PER; ++k) c[k] = col[lane + k * WARP];
    // a token's inputs: its query and key values of this lane, its value of the column, its decay and share
    float qa[PER], ka[PER], va = 0.0f, da = 0.0f, sa = 0.0f;
    auto fetch = [&](int row, float* q, float* kk, float& v, float& dc, float& sh) {
        const float* x = qkv + size_t(row) * row_n;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            q[k] = x[size_t(hk) * dim + lane + k * WARP];
            kk[k] = x[keys + size_t(hk) * dim + lane + k * WARP];
        }
        v = x[2 * keys + size_t(h) * dim + j];
        dc = decay[size_t(row) * n_head + h];
        sh = share[size_t(row) * n_head + h];
    };
    if (rows > 0) fetch(0, qa, ka, va, da, sa);
    for (int row = 0; row < rows; ++row) {
        float qb[PER], kb[PER], vb = 0.0f, db = 0.0f, sb = 0.0f;
        if (row + 1 < rows) fetch(row + 1, qb, kb, vb, db, sb);
        // the products and sums as gdn_state_kernel's compiled code has them, so that the result is the same bits
        float sk = 0.0f;
#pragma unroll
        /*
        DPCT1013:205: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        for (int k = 0; k < PER; ++k) sk = sycl::fma(da * c[k], ka[k], sk);
        sk = warp_sum(sk);
        /*
        DPCT1013:204: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        // the CUDA's __fmul_rn(__fsub_rn(va, sk), sa): dpct dropped the parentheses and made it va - sk * sa
        const float d = (va - sk) * sa;
        float o = 0.0f;
        float* m = snap_of(snap, stride, snaps, row);
        if (m) m += mine;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const int i = lane + k * WARP;
            /*
            DPCT1013:206: The rounding mode could not be specified and the
            generated code may have different accuracy than the original code.
            Verify the correctness. SYCL math built-in function rounding mode is
            aligned with OpenCL C 1.2 standard.
            */
            const float n = sycl::fma(da, c[k], ka[k] * d);
            c[k] = n;
            if (m) m[i] = n;
            /*
            DPCT1013:207: The rounding mode could not be specified and the
            generated code may have different accuracy than the original code.
            Verify the correctness. SYCL math built-in function rounding mode is
            aligned with OpenCL C 1.2 standard.
            */
            o = sycl::fma(n, qa[k], o);
        }
        o = warp_sum(o);
        if (lane == 0) out[size_t(row) * d_inner + size_t(h) * dim + j] = o;
#pragma unroll
        for (int k = 0; k < PER; ++k) qa[k] = qb[k], ka[k] = kb[k];
        va = vb, da = db, sa = sb;
    }
#pragma unroll
    for (int k = 0; k < PER; ++k) col[lane + k * WARP] = c[k];
}

// One block a token and value head: the output's norm, its weights and the gate.
__dpct_inline__ void gdn_out_kernel(float *__restrict__ out,
                                    const float *__restrict__ z,
                                    const float *__restrict__ o_norm, int dim,
                                    float norm_eps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const size_t at =
        size_t(item_ct1.get_group(2)) * dim + item_ct1.get_local_id(2);
    const float o = out[at];
    const float os = block_sum(o * o, part);
    const float gate = z[at];
    out[at] = o / sycl::sqrt(os / float(dim) + norm_eps) *
              o_norm[item_ct1.get_local_id(2)] * (gate * sigmoid(gate));
}

// conv4_silu_kernel for a block of tokens that see each other (Clover): a token's sum adds its next three inputs
// in the block, input t + d times right[d - 1], to the causal one.  The state is read and left as it was.
__dpct_inline__ void conv4_block_silu_kernel(const float *__restrict__ x,
                                             const float *__restrict__ state,
                                             const float *__restrict__ w,
                                             const float *__restrict__ right,
                                             float *__restrict__ out,
                                             int channels, int rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= channels) return;
    const float* wc = w + 4 * size_t(c);
    const float* rc = right + 3 * size_t(c);
    const float* st = state + 3 * size_t(c);
    const auto in = [&](int row) { return row < 0 ? st[3 + row] : x[size_t(row) * channels + c]; };
    for (int row = 0; row < rows; ++row) {
        float y = in(row - 3) * wc[0] + in(row - 2) * wc[1] + in(row - 1) * wc[2] + in(row) * wc[3];
#pragma unroll
        for (int d = 1; d <= 3 && row + d < rows; ++d) y +=
            x[size_t(row + d) * channels + c] * rc[d - 1];
        out[size_t(row) * channels + c] = y * sigmoid(y);
    }
}

// A value head's weights in a block of tokens that see each other (Clover's gated delta net in a block): token
// i's softmax over the block's tokens j, by the unscaled dot product of their normed query and key times the
// head's logit scale, and over the head's recurrent state, by bias + c * log(rows / 32).  w gets rows + 1 a token
// a head, the state's last.  qkv as gdn_norm_kernel left it: its queries are scaled by 1 / sqrt(dim).  A block a
// head; the queries and keys in shared memory.
/*
DPCT1110:78: The total declared local variable size in device function
gdn_block_weights_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gdn_block_weights_kernel(
    const float *__restrict__ qkv, const float *__restrict__ logit_scale,
    const float *__restrict__ hist_b, const float *__restrict__ hist_c,
    float *__restrict__ w, int dim, int n_head, int k_heads, int rows,
    uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sh = (float *)dpct_local;
    float* q = sh;                      // rows * dim
    float* k = q + size_t(rows) * dim;  // rows * dim
    float* sc = k + size_t(rows) * dim; // rows * (rows + 1)
    const int h = item_ct1.get_group(2), hk = h % k_heads, wide = rows + 1;
    const size_t keys = size_t(dim) * k_heads, row_n = 2 * keys + size_t(dim) * n_head;
    for (int at = item_ct1.get_local_id(2); at < rows * dim;
         at += item_ct1.get_local_range(2)) {
        const int r = at / dim, i = at % dim;
        const float* x = qkv + size_t(r) * row_n + size_t(hk) * dim;
        q[at] = x[i];
        k[at] = x[keys + i];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const float scale =
        logit_scale[h] *
        sycl::sqrt(float(dim)); // the queries' 1 / sqrt(dim) undone
    for (int at = item_ct1.get_local_id(2); at < rows * rows;
         at += item_ct1.get_local_range(2)) {
        const int i = at / rows, j = at % rows;
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < dim; ++d) dot +=
            q[size_t(i) * dim + d] * k[size_t(j) * dim + d];
        sc[size_t(i) * wide + j] = dot * scale;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const float past = hist_b[h] + hist_c[h] * sycl::log(float(rows) / 32.0f);
    for (int i = item_ct1.get_local_id(2); i < rows;
         i += item_ct1.get_local_range(2)) {
        float* s = sc + size_t(i) * wide;
        s[rows] = past;
        float mx = s[0];
#pragma unroll
        for (int j = 1; j < wide; ++j) mx = sycl::fmax(mx, s[j]);
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < wide; ++j) sum +=
            s[j] = sycl::native::exp(s[j] - mx);
        float* o = w + (size_t(h) * rows + i) * wide;
#pragma unroll
        for (int j = 0; j < wide; ++j) o[j] = s[j] / sum;
    }
}

// The block's output before its norm, a warp a column as gdn_state_kernel: token i's read of the state, S q_i
// decayed by the token's decay, times its weight on the state, and the block's values by its weights on them.
// The state is read and left as it was.
__dpct_inline__ void gdn_block_out_kernel(const float *__restrict__ qkv,
                                          const float *__restrict__ decay,
                                          const float *__restrict__ s,
                                          const float *__restrict__ w,
                                          float *__restrict__ out, int dim,
                                          int n_head, int k_heads, int rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int per_head = dim / int(item_ct1.get_local_range(2) / WARP);
    const int h = item_ct1.get_group(2) / per_head, hk = h % k_heads;
    const int j = (item_ct1.get_group(2) % per_head) *
                      int(item_ct1.get_local_range(2) / WARP) +
                  int(item_ct1.get_local_id(2) / WARP);
    const int lane = item_ct1.get_local_id(2) & (WARP - 1);
    if (j >= dim) return;
    const size_t keys = size_t(dim) * k_heads, d_inner = size_t(dim) * n_head, row_n = 2 * keys + d_inner;
    const float* col = s + (size_t(h) * dim + j) * dim;
    const int wide = rows + 1;
    for (int row = 0; row < rows; ++row) {
        const float* qn = qkv + size_t(row) * row_n + size_t(hk) * dim;
        float o = 0.0f;
#pragma unroll
        for (int i = lane; i < dim; i += WARP) o += col[i] * qn[i];
        o = warp_sum(o);
        if (lane != 0) continue;
        const float* wt = w + (size_t(h) * rows + row) * wide;
        float local = 0.0f;
#pragma unroll
        for (int r = 0; r < rows; ++r) local +=
            wt[r] * qkv[size_t(r) * row_n + 2 * keys + size_t(h) * dim + j];
        out[size_t(row) * d_inner + size_t(h) * dim + j] = o * decay[size_t(row) * n_head + h] * wt[rows] + local;
    }
}

// Qwen35::route_block() on the card: a thread an expert, one block for the call's tokens.
__dpct_inline__ void coreset_mask_kernel(float *__restrict__ logits,
                                         int n_expert, int n_used, int rows,
                                         float share, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sh = (float *)dpct_local;
    float* p = sh;               // a token's probabilities before their sum divides them
    float* mass = sh + n_expert; // each expert's probability, added up over the tokens that took it
    auto &part =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int e = item_ct1.get_local_id(2);
    mass[e] = 0.0f;
    for (int t = 0; t < rows; ++t) {
        const float s = logits[size_t(t) * n_expert + e];
        const float x = sycl::native::exp(s - block_max(s, part));
        const float sum = block_sum(x, part);
        p[e] = x;
        /*
        DPCT1118:79: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:208: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        int ahead = 0;
#pragma unroll
        for (int j = 0; j < n_expert; ++j) ahead +=
            p[j] > x || (p[j] == x && j < e);
        if (ahead < n_used) mass[e] += x / sum;
        /*
        DPCT1118:80: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:209: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    const float m = mass[e], total = block_sum(m, part);
    int ahead = 0;
    float before = 0.0f;
    for (int j = 0; j < n_expert; ++j) {
        const bool a = mass[j] > m || (mass[j] == m && j < e);
        ahead += a;
        before += a ? mass[j] : 0.0f;
    }
    if (ahead > 0 && before >= share * total)
#pragma unroll
        for (int t = 0; t < rows; ++t)
            logits[size_t(t) * n_expert + e] = -INFINITY;
}

inline dpct::queue_ptr cs(void *stream) { return strata::q_of(stream); }
inline int grid(size_t n, int block) { return int((n + block - 1) / block); }

// The PTX version the kernels run from here, 0 under HIP: a card running an older card's PTX has only that card's
// instructions.
int kernels_ptx() try {
#if 1   // SYCL port
    return 0;
#else
    dpct::kernel_function_info a{};
    return DPCT_CHECK_ERROR(dpct::get_kernel_function_info(
               &a, (const void *)attn_flash64_kernel)) == 0
               ? a.ptxVersion
               : 0;
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Whether the tensor-core kernels run as such here: built for Turing or newer (STRATA_POLY_MMA), and not from a PTX
// of an older card compiled for this one.
bool tensor_cores() { return kernels_ptx() >= 75; }

void blas_ck(int s, const char *what) {
    if (s != 0)
        throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string((int) s));
}

}  // namespace

void *blas_create(void *stream) try {
    dpct::blas::descriptor_ptr h = nullptr;
    blas_ck(DPCT_CHECK_ERROR(h = new dpct::blas::descriptor()), "cublasCreate");
    if (const int s = DPCT_CHECK_ERROR(h->set_queue(cs(stream))); s != 0) {
        delete (h);
        blas_ck(s, "cublasSetStream");
    }
    // cuBLAS takes 75 ms over a process's first product of halves, whatever its size: a prompt batch's attention
    // would wait for it.  A product that fails here fails again where its status is read.
    const int n = 16;
    void* p = nullptr;
    if (DPCT_CHECK_ERROR(p = (void *)sycl::malloc_device(
                             n * n * (2 * sizeof(sycl::half) + sizeof(float)),
                             dpct::get_in_order_queue())) != 0) {
        /*
        DPCT1026:210: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
; // or the next launch's check reports it
        return h;
    }
    cs(stream)->memset(p, 0, n * n * (2 * sizeof(sycl::half) + sizeof(float)));
    const float one = 1.0f, zero = 0.0f;
    const sycl::half *a = (const sycl::half *)p;
    dpct::blas::gemm_batch(
        h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, n,
        n, n, &one, a, dpct::library_data_t::real_half, n, n * n, a + n * n,
        dpct::library_data_t::real_half, n, n * n, &zero,
        (float *)(a + 2 * n * n), dpct::library_data_t::real_float, n, n * n, 1,
        dpct::compute_type::f32);
    cs(stream)->wait();
    sycl::free(p, dpct::get_in_order_queue());
    return h;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void blas_destroy(void* blas) {
    if (blas) delete ((dpct::blas::descriptor_ptr)blas);
}

void gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream) {
    gemv(w, x, y, n_in, n_out, rows_per_x, rows, stream);
}

void gemv_bf16(const uint16_t* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows,
               void* stream) {
    gemv(w, x, y, n_in, n_out, rows_per_x, rows, stream);
}

// A buffer of rows is its transpose to cuBLAS, so y[row] = w x[row] is Y = W^T X.  A group's rows of w, its part
// of every token's x and of every token's y are each a matrix with a stride.
void gemm_f32(void* blas, const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows) {
    const dpct::blas::descriptor_ptr h = (dpct::blas::descriptor_ptr)blas;
    const float one = 1.0f, zero = 0.0f;
    const int groups = n_out / rows_per_x;
    if (groups == 1 && n_out <= 8 * SMALL_OUT) {
        dpct::queue_ptr stream = &dpct::get_in_order_queue();
        blas_ck(DPCT_CHECK_ERROR(stream = &(h->get_queue())),
                "cublasGetStream");
        if (n_out > 2 * SMALL_OUT)
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            stream
                ->parallel_for<dpct_kernel_name<class gemm_tile_kernel_c49a48>>(
                    sycl::nd_range<3>(
                        sycl::range(1, (n_out + TILE_OUT - 1) / TILE_OUT,
                                    (rows + TILE_TOK - 1) / TILE_TOK) *
                            sycl::range(1, 1, TILE_TOK * TILE_OUT / 16),
                        sycl::range(1, 1, TILE_TOK * TILE_OUT / 16)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        gemm_tile_kernel(w, x, y, n_in, n_out, rows);
                    });
        } else {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            stream->parallel_for<
                dpct_kernel_name<class gemm_small_kernel_3d4dde>>(
                sycl::nd_range<3>(
                    sycl::range(1, (n_out + SMALL_OUT - 1) / SMALL_OUT,
                                (rows + SMALL_TOK - 1) / SMALL_TOK) *
                        sycl::range(1, 1, SMALL_TOK * WARP),
                    sycl::range(1, 1, SMALL_TOK * WARP)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    gemm_small_kernel(w, x, y, n_in, n_out, rows);
                });
        }
        return;
    }
    if (groups == 1) {
        blas_ck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
                    h->get_queue(), oneapi::mkl::transpose::trans,
                    oneapi::mkl::transpose::nontrans, n_out, rows, n_in, one, w,
                    n_in, x, n_in, zero, y, n_out)),
                "a batch's projection");
        return;
    }
    blas_ck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
                h->get_queue(), oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, rows_per_x, rows, n_in, one,
                w, n_in, (long long)rows_per_x * n_in, x, groups * n_in, n_in,
                zero, y, n_out, rows_per_x, groups)),
            "a batch's projections");
}

void gemm_f16w(void* blas, const uint16_t* w, const uint16_t* x, float* y, int n_in, int n_out, int rows) {
    const dpct::blas::descriptor_ptr h = (dpct::blas::descriptor_ptr)blas;
    const float one = 1.0f, zero = 0.0f;
    blas_ck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
                h->get_queue(), oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, n_out, rows, n_in,
                one, reinterpret_cast<const sycl::half*>(w), n_in, reinterpret_cast<const sycl::half*>(x), n_in,
                zero, y, n_out)),
            "a batch's fp16 projection");
}

void guard_half_sums(float* x, int64_t ld, int width, int rows, float* factor, void* stream) {
    if (rows > 0) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<
                dpct_kernel_name<class guard_half_sums_kernel_14f939>>(
                sycl::nd_range<3>(sycl::range(1, 1, rows) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        guard_half_sums_kernel(x, size_t(ld), width, factor);
                    });
    }
}

void scale_rows(float* y, int64_t ld, int width, int rows, const int32_t* place, const float* factor, bool divide,
                void* stream) {
    for (int r = 0; r < rows; r += 65535)   // a grid has at most 65535 rows
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->submit([&](sycl::handler &cgh) {
            auto place_y_y_size_t_r_ld_ct0 = place ? y : y + size_t(r) * ld;
            auto place_place_r_nullptr_ct3 = place ? place + r : nullptr;
            auto factor_r_ct4 = factor + r;

            cgh.parallel_for<dpct_kernel_name<class scale_rows_kernel_8fb6d3>>(
                sycl::nd_range<3>(sycl::range(1, std::min(65535, rows - r),
                                              grid(width, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    scale_rows_kernel(place_y_y_size_t_r_ld_ct0, size_t(ld),
                                      width, place_place_r_nullptr_ct3,
                                      factor_r_ct4, divide);
                });
        });
    }
}

void bf16_to_f32(const uint16_t* w, float* out, size_t n, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->parallel_for<dpct_kernel_name<class bf16_to_f32_kernel_bea9c2>>(
        sycl::nd_range<3>(sycl::range(1, 1, grid(n, 256)) *
                              sycl::range(1, 1, 256),
                          sycl::range(1, 1, 256)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
            bf16_to_f32_kernel(w, out, n);
        });
}

void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* stream, int y_stride) {
    /*
    DPCT1049:81: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class rms_norm_kernel_7db833>>(
                sycl::nd_range<3>(sycl::range(1, 1, rows) *
                                      sycl::range(1, 1, sum_threads(n)),
                                  sycl::range(1, 1, sum_threads(n))),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        rms_norm_kernel(x, w, y, n, eps,
                                        y_stride ? y_stride : n);
                    });
    }
    /*
    DPCT1010:213: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    /*
    DPCT1000:212: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (const dpct::err0 e = 0; e != 0)
        /*
        DPCT1009:214: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001:211: The statement could not be removed.
        */
        throw std::runtime_error(std::string("rms_norm: ") +
                                 dpct::get_error_string_dummy(e));
}

// SYCL port: rms_norm() and native_quantize_q8_1() of its output (n * rows values as one vector) in one launch;
// y is the norm's output as for rms_norm().  Written to equal the two launches bit for bit, but NOT compared with
// them yet: poly_shims.cpp calls it only when STRATA_FUSE_NORM=1, so that it can be.
void rms_norm_q8_1(const float* x, const float* w, float* y, void* y_q8_1, int n, int rows, float eps, void* stream) {
    if (n <= 0 || n % 32 != 0 || rows < 1 || !y_q8_1)
        throw std::invalid_argument("rms_norm_q8_1: n must be a multiple of 32, rows >= 1, and the Q8_1 buffer given");
    uint8_t* const q8 = static_cast<uint8_t*>(y_q8_1);
    const int threads = sum_threads(n);
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
    cs(stream)->parallel_for<dpct_kernel_name<class rms_norm_q8_1_kernel_3b9a41>>(
        sycl::nd_range<3>(sycl::range(1, 1, rows) * sycl::range(1, 1, threads), sycl::range(1, 1, threads)),
        exp_props, [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            rms_norm_q8_1_kernel(x, w, y, q8, n, eps);
        });
}

void rms_norm_q8_1_qkv(std::array<NormQ81Row, 3> rows, int n, float eps, void* stream) {
    const int threads = sum_threads(n);
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
    cs(stream)->parallel_for<dpct_kernel_name<class rms_norm_q8_1_qkv_kernel>>(
        sycl::nd_range<3>(sycl::range<3>(3, 1, threads), sycl::range<3>(1, 1, threads)),
        exp_props, [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
            // Group 0 selects the Q/K/V job; group 2 is zero for each job's single row.
            const NormQ81Row& row = rows[item.get_group(0)];
            rms_norm_q8_1_kernel(row.input, row.weight, row.normalized,
                                 static_cast<uint8_t*>(row.quantized), n, eps);
        });
}

void layer_norm(const float* x, const float* w, const float* b, float* y, int n, int rows, float eps, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->parallel_for<dpct_kernel_name<class layer_norm_kernel_2b9064>>(
        sycl::nd_range<3>(sycl::range(1, 1, rows) * sycl::range(1, 1, 128),
                          sycl::range(1, 1, 128)),
        exp_props,
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            layer_norm_kernel(x, w, b, y, n, eps);
        });
}

void hc_pre(const float* r, const float* fn, const float* scale, const float* base, float* mix, float* hcw,
            float* cur, int n_embd, int rows, float norm_eps, float hc_eps, int sinkhorn_iters, void* stream) {
    const int n_mix = HC * (2 + HC);
    /*
    DPCT1049:82: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->submit([&](sycl::handler &cgh) {
            auto HC_n_embd_ct3 = HC * n_embd;

            cgh.parallel_for<dpct_kernel_name<class hc_mix_kernel_1ef5ed>>(
                sycl::nd_range<3>(
                    sycl::range(1, rows, n_mix + 1) *
                        sycl::range(1, 1, sum_threads(HC * n_embd)),
                    sycl::range(1, 1, sum_threads(HC * n_embd))),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        hc_mix_kernel(r, fn, mix, HC_n_embd_ct3, n_mix);
                    });
        });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->parallel_for<dpct_kernel_name<class hc_read_kernel_24275d>>(
            sycl::nd_range<3>(sycl::range(1, rows, grid(n_embd, 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                hc_read_kernel(r, mix, scale, base, hcw, cur, n_embd, norm_eps,
                               hc_eps, sinkhorn_iters);
            });
    }
}

void hc_post(const float* x, const float* hcw, float* r, int n_embd, int rows, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->parallel_for<dpct_kernel_name<class hc_post_kernel_d3b090>>(
        sycl::nd_range<3>(sycl::range(1, rows, grid(n_embd, 256)) *
                              sycl::range(1, 1, 256),
                          sycl::range(1, 1, 256)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
            hc_post_kernel(x, hcw, r, n_embd);
        });
}

void hc_mean(const float* r, float* y, int n_embd, int rows, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->parallel_for<dpct_kernel_name<class hc_mean_kernel_4553e2>>(
        sycl::nd_range<3>(sycl::range(1, rows, grid(n_embd, 256)) *
                              sycl::range(1, 1, 256),
                          sycl::range(1, 1, 256)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
            hc_mean_kernel(r, y, n_embd);
        });
}

void hc_head(const float* r, const float* fn, const float* scale, const float* base, float* mix, float* y,
             int n_embd, int rows, float norm_eps, float hc_eps, void* stream) {
    /*
    DPCT1049:83: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->submit([&](sycl::handler &cgh) {
            auto HC_n_embd_ct3 = HC * n_embd;
            auto HC_ct4 = HC;

            cgh.parallel_for<dpct_kernel_name<class hc_mix_kernel_b7ea98>>(
                sycl::nd_range<3>(
                    sycl::range(1, rows, HC + 1) *
                        sycl::range(1, 1, sum_threads(HC * n_embd)),
                    sycl::range(1, 1, sum_threads(HC * n_embd))),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        hc_mix_kernel(r, fn, mix, HC_n_embd_ct3, HC_ct4);
                    });
        });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->parallel_for<dpct_kernel_name<class hc_head_kernel_ad889e>>(
            sycl::nd_range<3>(sycl::range(1, rows, grid(n_embd, 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                hc_head_kernel(r, mix, scale, base, y, n_embd, norm_eps,
                               hc_eps);
            });
    }
}

void kda_conv(const float* proj, float* state, const float* wq, const float* wk, const float* wv, float* out,
              int d_inner, int rows, void* stream, float* snap, size_t stride, int n_snap) {
    if (rows <= 0) return;
    if (rows >= CONV_ROWS && (!snap || n_snap <= 0)) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<
                    dpct_kernel_name<class kda_conv_rows_kernel_817b53>>(
                    sycl::nd_range<3>(
                        sycl::range(1, rows, grid(3 * d_inner, 256)) *
                            sycl::range(1, 1, 256),
                        sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        kda_conv_rows_kernel(proj, state, wq, wk, wv, out,
                                             d_inner, rows);
                    });
        }
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<
                    dpct_kernel_name<class kda_conv_state_kernel_57017c>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 1, grid(3 * d_inner, 256)) *
                            sycl::range(1, 1, 256),
                        sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        kda_conv_state_kernel(proj, state, d_inner, rows);
                    });
        }
        return;
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class kda_conv_kernel_73d380>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(3 * d_inner, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    kda_conv_kernel(proj, state, wq, wk, wv, out, d_inner, rows,
                                    snap, stride, n_snap);
                });
    }
}

void kda_step(float* qkv, float* g1, const float* a, const float* beta, float* s, const float* g2,
              const float* o_norm, float* out, int head_dim, int n_head, int rows, float gate_lower_bound,
              float norm_eps, void* stream, float* snap, size_t stride, int n_snap) {
    if (head_dim % WARP != 0 || head_dim > 256)
        throw std::invalid_argument("kda_step: a head of 32 to 256 values, in 32s");
    int lanes = WARP;   // threads a column, while a block has them
    while (head_dim * lanes > 1024) lanes /= 2;
    // a few tokens, or a head that is not kda_state_kernel's size: one launch
    if (rows < 32 || snap || head_dim != KDA_TERMS * lanes) {
        /*
        DPCT1049:87: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<dpct_kernel_name<class kda_step_kernel_a4680a>>(
                    sycl::nd_range<3>(sycl::range(1, 1, n_head) *
                                          sycl::range(1, 1, head_dim * lanes),
                                      sycl::range(1, 1, head_dim * lanes)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            kda_step_kernel(qkv, g1, a, beta, s, g2, o_norm,
                                            out, head_dim, n_head, rows,
                                            gate_lower_bound, norm_eps, snap,
                                            stride, snap ? n_snap : 0);
                        });
        }
        return;
    }
    const int group = WARP / lanes;
    /*
    DPCT1049:84: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class kda_gates_kernel_d09d22>>(
                sycl::nd_range<3>(sycl::range(1, 1, rows * n_head) *
                                      sycl::range(1, 1, head_dim),
                                  sycl::range(1, 1, head_dim)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        kda_gates_kernel(qkv, g1, a, n_head, rows,
                                         gate_lower_bound, group);
                    });
    }
    /*
    DPCT1049:85: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class kda_state_kernel_f27d4e>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, n_head * (head_dim / KDA_COLUMNS)) *
                        sycl::range(1, 1, KDA_COLUMNS * lanes),
                    sycl::range(1, 1, KDA_COLUMNS * lanes)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        kda_state_kernel(qkv, g1, beta, s, out, n_head, rows);
                    });
    }
    /*
    DPCT1049:86: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->parallel_for<dpct_kernel_name<class kda_out_kernel_697a80>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows * n_head) *
                                  sycl::range(1, 1, head_dim),
                              sycl::range(1, 1, head_dim)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                kda_out_kernel(out, g2, o_norm, norm_eps, group);
            });
    }
}

void store_f16(const float* x, uint16_t* cache, int n, int64_t pos, int rows, void* stream) {
    const size_t all = size_t(n) * rows;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(cs(stream)->get_device(),
                                     {sycl::aspect::fp16});

        cs(stream)->submit([&](sycl::handler &cgh) {
            auto half_cache_size_t_pos_n_ct1 =
                (sycl::half *)cache + size_t(pos) * n;

            cgh.parallel_for<dpct_kernel_name<class store_f16_kernel_d3f026>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(all, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    store_f16_kernel(x, half_cache_size_t_pos_n_ct1, int(all));
                });
        });
    }
}

void store_f16_at(const float* x, uint16_t* cache, int n, const long long* at, int rows, int ring, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
    dpct::has_capability_or_fail(cs(stream)->get_device(),
                                 {sycl::aspect::fp16});

    cs(stream)
        ->parallel_for<dpct_kernel_name<class store_f16_at_kernel_5a34e8>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid(size_t(n) * rows, 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                store_f16_at_kernel(x, (sycl::half *)cache, n, rows, at, ring);
            });
}

// --------------------------------------------------- int8 attention caches (--kv int8 / int8k, see the header)

// One block of quantize_row_q8_0_ref's rule, one thread: the largest magnitude sets the delta, every value rounds
// to that delta's units.  roundf, not __float2int_rn - the rule the CPU reference uses, to the bit.  (An all-zero
// block leaves the delta at 0 and every code 0; llama.cpp writes id = 0 there for the same reason.)  The codes go
// to their own array, which is what keeps them 4-aligned for every reader.
__dpct_inline__ void quantize_q8_0_block(const float *__restrict__ v,
                                         int8_t *__restrict__ code,
                                         uint16_t *__restrict__ delta) {
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < kQ8_0Values; ++j)
        amax = sycl::fmax(amax, sycl::fabs(v[j]));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    *delta = sycl::bit_cast<unsigned short, sycl::half>(
        sycl::vec<float, 1>(d)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0]);
#pragma unroll
    for (int j = 0; j < kQ8_0Values; ++j)
        code[j] = (int8_t)sycl::round(v[j] * id);
}

// One thread, one block.  per_row is the blocks a cache row has (n_kv * head_dim / kQ8_0Values).
__dpct_inline__ void store_int8_kernel(const float *__restrict__ x,
                                       int8_t *__restrict__ code,
                                       uint16_t *__restrict__ delta,
                                       int64_t n_blocks, int per_row) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        item_ct1.get_group(2) * (int64_t)item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n_blocks) return;
    const float* v = x + (i / per_row) * ((int64_t) per_row * kQ8_0Values) + (i % per_row) * kQ8_0Values;
    quantize_q8_0_block(v, code + i * kQ8_0Values, delta + i);
}

// The row a block goes to comes from what the card holds at `at` (a recorded launch), as store_f16_at's.
__dpct_inline__ void
store_int8_at_kernel(const float *__restrict__ x, int8_t *__restrict__ code,
                     uint16_t *__restrict__ delta, int64_t n_blocks,
                     int per_row, const long long *__restrict__ at, int ring) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        item_ct1.get_group(2) * (int64_t)item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n_blocks) return;
    const int64_t row = i / per_row, b = i % per_row;
    const float* v = x + row * ((int64_t) per_row * kQ8_0Values) + b * kQ8_0Values;
    const int64_t dst_row = ring > 0 ? (*at + row) % ring : *at + row;
    quantize_q8_0_block(v, code + (dst_row * per_row + b) * kQ8_0Values, delta + dst_row * per_row + b);
}

// A Q8_0 cache back as halves: a thread a kQ8_0Values-block, its codes read four at a time (back to back inside a
// block, so every fourth one is 4-aligned) and its delta once.  ggml's own kernel for this
// (dequantize_block_q8_0_f16) stages 34-byte blocks through shared memory to line the codes up; here they already
// are, so it is eight half2 multiplies and no shared memory at all.
__dpct_inline__ void dequant_int8_f16_kernel(const int8_t *__restrict__ code,
                                             const uint16_t *__restrict__ delta,
                                             sycl::half *__restrict__ out,
                                             int64_t n_blocks) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t b =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (b >= n_blocks) return;
    const sycl::half2 d2 =
        sycl::half2(sycl::bit_cast<sycl::half, unsigned short>(delta[b]));
    const int8_t* c = code + b * kQ8_0Values;
    sycl::half2 *o = reinterpret_cast<sycl::half2 *>(out + b * kQ8_0Values);
#pragma unroll
    for (int j = 0; j < kQ8_0Values / 4; ++j) {
        const sycl::char4 q = *reinterpret_cast<const sycl::char4 *>(c + 4 * j);
        o[2 * j] = sycl::half2(q.x(), q.y()) * d2;
        o[2 * j + 1] = sycl::half2(q.z(), q.w()) * d2;
    }
}

void dequant_int8_to_f16(const int8_t* code, const uint16_t* delta, uint16_t* out, int64_t n_blocks, void* stream) {
    if (n_blocks <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(cs(stream)->get_device(),
                                     {sycl::aspect::fp16});

        cs(stream)
            ->parallel_for<
                dpct_kernel_name<class dequant_int8_f16_kernel_81309f>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, grid((size_t)n_blocks, 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    dequant_int8_f16_kernel(code, delta,
                                            reinterpret_cast<sycl::half *>(out),
                                            n_blocks);
                });
    }
}

void store_int8(const float* x, int8_t* code, uint16_t* delta, int n_kv, int head_dim, int64_t pos, int rows,
              void* stream) {
    const int per_row = n_kv * head_dim / kQ8_0Values;
    const int64_t all = (int64_t) per_row * rows;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->submit([&](sycl::handler &cgh) {
            auto code_size_t_pos_per_row_kQ8_0Values_ct1 =
                code + (size_t)pos * per_row * kQ8_0Values;
            auto delta_size_t_pos_per_row_ct2 = delta + (size_t)pos * per_row;

            cgh.parallel_for<dpct_kernel_name<class store_int8_kernel_b4204f>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid((size_t)all, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    store_int8_kernel(
                        x, code_size_t_pos_per_row_kQ8_0Values_ct1,
                        delta_size_t_pos_per_row_ct2, all, per_row);
                });
        });
    }
}

void store_int8_at(const float* x, int8_t* code, uint16_t* delta, int n_kv, int head_dim, int rows,
                 const long long* at, int ring, void* stream) {
    const int per_row = n_kv * head_dim / kQ8_0Values;
    const int64_t all = (int64_t) per_row * rows;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class store_int8_at_kernel_5c5fd4>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid((size_t)all, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    store_int8_at_kernel(x, code, delta, all, per_row, at,
                                         ring);
                });
    }
}

void store_q8(const float* x, int8_t* codes, uint16_t* scales, int n, int64_t pos, int rows, void* stream) {
    if ((n & 31) != 0) throw std::invalid_argument("store_q8: n is not a multiple of 32");
    const int nb = n >> 5;
    const size_t all = (size_t) nb * rows;
    if (!all) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(cs(stream)->get_device(),
                                     {sycl::aspect::fp16});

        cs(stream)->submit([&](sycl::handler &cgh) {
            auto codes_size_t_pos_n_ct1 = codes + (size_t)pos * n;
            auto half_scales_size_t_pos_nb_ct2 =
                (sycl::half *)scales + (size_t)pos * nb;

            cgh.parallel_for<dpct_kernel_name<class store_q8_kernel_e41725>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(all, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    store_q8_kernel(x, codes_size_t_pos_n_ct1,
                                    half_scales_size_t_pos_nb_ct2, n, rows, nb);
                });
        });
    }
}

void store_q8_at(const float* x, int8_t* codes, uint16_t* scales, int n, const long long* at, int rows, int ring,
                 void* stream) {
    if ((n & 31) != 0) throw std::invalid_argument("store_q8_at: n is not a multiple of 32");
    const int nb = n >> 5;
    const size_t all = (size_t) nb * rows;
    if (!all) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(cs(stream)->get_device(),
                                     {sycl::aspect::fp16});

        cs(stream)
            ->parallel_for<dpct_kernel_name<class store_q8_at_kernel_f4d7e6>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(all, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    store_q8_at_kernel(x, codes, (sycl::half *)scales, n, rows,
                                       nb, at, ring);
                });
    }
}

void dsa_attend(void* blas, const float* q, const uint16_t* cache, const int32_t* sel, int sel_stride, int top_rows,
                int kpool, int64_t pos, int rows, int dense_rows, float* keys, float* score, int chunk, float* out,
                int rank, int n_head, float scale, void* stream) {
    const dpct::blas::descriptor_ptr h = (dpct::blas::descriptor_ptr)blas;
    const float one = 1.0f, zero = 0.0f;
    const long long head = (long long) n_head * rank;
    for (int first = 0; first < rows; first += chunk) {
        const int c = std::min(chunk, rows - first);
        int n_keys = 0;
        for (int t = first; t < first + c; ++t)
            n_keys = std::max(n_keys, dsa_keys(t, (long long) pos, dense_rows, top_rows, kpool));
        const long long key_n = (long long) n_keys * rank, score_n = (long long) n_head * n_keys;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(cs(stream)->get_device(),
                                         {sycl::aspect::fp16});

            cs(stream)
                ->parallel_for<dpct_kernel_name<class dsa_keys_kernel_335dbc>>(
                    sycl::nd_range<3>(
                        sycl::range(1, c, grid(n_keys, KEY_BLOCK)) *
                            sycl::range(1, 1, 256),
                        sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        dsa_keys_kernel((const sycl::half *)cache, sel,
                                        sel_stride, top_rows, kpool,
                                        (long long)pos, first, dense_rows, keys,
                                        rank, n_keys);
                    });
        }
        // score[head][key] = scale * (q[head] . keys[key]).  cuBLAS reads a buffer of rows as the transpose.
        blas_ck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
                    h->get_queue(), oneapi::mkl::transpose::trans,
                    oneapi::mkl::transpose::nontrans, n_keys, n_head, rank,
                    scale, keys, rank, key_n, q + size_t(first) * head, rank,
                    head, zero, score, n_keys, score_n, c)),
                "attention scores");
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<
                    dpct_kernel_name<class dsa_softmax_kernel_c7a60f>>(
                    sycl::nd_range<3>(sycl::range(1, c, n_head) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            dsa_softmax_kernel(score, n_keys, top_rows, kpool,
                                               (long long)pos, first,
                                               dense_rows);
                        });
        }
        // out[head] = sum over the keys of score[head][key] * keys[key]
        blas_ck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
                    h->get_queue(), oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::transpose::nontrans, rank, n_head, n_keys, one,
                    keys, rank, key_n, score, n_keys, score_n, zero,
                    out + size_t(first) * head, rank, head, c)),
                "attention values");
    }
}

void idx_store(const float* key, const float* gate, float* ring, int ring_rows, int dim, int64_t pos, int rows,
               void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->parallel_for<dpct_kernel_name<class idx_store_kernel_9b0be9>>(
        sycl::nd_range<3>(sycl::range(1, 1, rows) * sycl::range(1, 1, 256),
                          sycl::range(1, 1, 256)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
            idx_store_kernel(key, gate, ring, ring_rows, dim, (long long)pos);
        });
}

void idx_pool(const float* ring, int ring_rows, float* pool, const float* ape, int dim, int kpool, int64_t pos,
              int rows, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->parallel_for<dpct_kernel_name<class idx_pool_kernel_4e340f>>(
        sycl::nd_range<3>(sycl::range(1, 1, rows) * sycl::range(1, 1, 256),
                          sycl::range(1, 1, 256)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
            idx_pool_kernel(ring, ring_rows, pool, ape, dim, kpool,
                            (long long)pos);
        });
}

void idx_score(const float* q, const float* w, const float* pool, float* score, int score_stride, int dim,
               int n_head, int kpool, int64_t pos, int rows, void* stream) {
    if (dim != IDX_DIM || n_head > WARP) throw std::invalid_argument("idx_score: 128 values a key, 32 heads at most");
    const int n_pool = int((pos + rows) / kpool);   // the last token sees the most
    if (n_pool == 0) return;
    // whole tiles, and some 500 blocks at a long context
    const int run = std::max(IDX_TILE, std::min(1024, n_pool / 512 / IDX_TILE * IDX_TILE));
    const int warps = std::min(rows, IDX_WARPS);
    /*
    DPCT1049:88: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->submit([&](sycl::handler &cgh) {
            auto std_sqrt_float_dim_float_n_head_ct8 =
                1.0f / std::sqrt(float(dim) * float(n_head));

            cgh.parallel_for<dpct_kernel_name<class idx_score_kernel_279860,
                                              dpct_kernel_scalar<IDX_DIM>>>(
                sycl::nd_range<3>(sycl::range(1, (rows + warps - 1) / warps,
                                              (n_pool + run - 1) / run) *
                                      sycl::range(1, 1, warps * WARP),
                                  sycl::range(1, 1, warps * WARP)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        idx_score_kernel<IDX_DIM>(
                            q, w, pool, score, score_stride, n_head, kpool,
                            (long long)pos, std_sqrt_float_dim_float_n_head_ct8,
                            run, rows);
                    });
        });
    }
}

void idx_select(const float* score, int score_stride, int32_t* sel, int sel_stride, int n_top, int kpool,
                int64_t pos, int rows, void* stream) {
    /*
    DPCT1049:89: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->submit([&](sycl::handler &cgh) {
        /*
        DPCT1083:371: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range(2 * size_t(n_top) * sizeof(uint32_t)), cgh);

        cgh.parallel_for<dpct_kernel_name<class idx_select_kernel_5b23d9>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows) *
                                  sycl::range(1, 1, IDX_SELECT),
                              sycl::range(1, 1, IDX_SELECT)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                idx_select_kernel(
                    score, score_stride, sel, sel_stride, n_top, kpool,
                    (long long)pos,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
}

void idx_score_batch(void* blas, const float* q, const float* w, const float* pool, float* score, int score_stride,
                     int dim, int n_head, int kpool, int64_t pos, int rows, float* work, size_t work_floats,
                     void* stream) {
    const float one = 1.0f, zero = 0.0f;
    const int n_pool = int((pos + rows) / kpool);   // the last token sees the most
    if (n_pool == 0) return;
    const int chunk = (int) std::min<size_t>(65535, work_floats / (size_t(n_pool) * n_head));
    if (chunk < 1) throw std::invalid_argument("idx_score_batch: no room for a token's products");
    for (int first = 0; first < rows; first += chunk) {
        const int c = std::min(chunk, rows - first);
        blas_ck(
            DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
                ((dpct::blas::descriptor_ptr)blas)->get_queue(),
                oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                n_pool, c * n_head, dim, one, pool, dim,
                q + size_t(first) * n_head * dim, dim, zero, work, n_pool)),
            "indexer scores");
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)->submit([&](sycl::handler &cgh) {
                auto w_size_t_first_n_head_ct1 = w + size_t(first) * n_head;
                auto score_size_t_first_score_stride_ct2 =
                    score + size_t(first) * score_stride;
                auto long_long_pos_first_ct7 = (long long)pos + first;
                auto std_sqrt_float_dim_float_n_head_ct8 =
                    1.0f / std::sqrt(float(dim) * float(n_head));

                cgh.parallel_for<
                    dpct_kernel_name<class idx_weigh_kernel_f41435>>(
                    sycl::nd_range<3>(sycl::range(1, c, grid(n_pool, 256)) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        idx_weigh_kernel(work, w_size_t_first_n_head_ct1,
                                         score_size_t_first_score_stride_ct2,
                                         score_stride, n_pool, n_head, kpool,
                                         long_long_pos_first_ct7,
                                         std_sqrt_float_dim_float_n_head_ct8);
                    });
            });
        }
    }
}

void swiglu_clamp(const float* gate, const float* up, float* out, int n, float limit, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)
        ->parallel_for<dpct_kernel_name<class swiglu_clamp_kernel_7796aa>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid(n, 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                swiglu_clamp_kernel(gate, up, out, n, limit);
            });
}

void axpy(const float* x, float a, float* acc, int n, bool add, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->parallel_for<dpct_kernel_name<class axpy_kernel_8785d3>>(
        sycl::nd_range<3>(sycl::range(1, 1, grid(n, 256)) *
                              sycl::range(1, 1, 256),
                          sycl::range(1, 1, 256)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
            axpy_kernel(x, a, acc, n, add);
        });
}

void copy2(const void* x0, void* y0, int n0, const void* x1, void* y1, int n1, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->parallel_for<dpct_kernel_name<class copy2_kernel_2d1bb9>>(
            sycl::nd_range<3>(sycl::range(1, 1, grid(n0 + n1, 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                copy2_kernel((const uint32_t *)x0, (uint32_t *)y0, n0,
                             (const uint32_t *)x1, (uint32_t *)y1, n1);
            });
    }
    /*
    DPCT1010:217: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    /*
    DPCT1000:216: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (const dpct::err0 e = 0; e != 0)
        /*
        DPCT1009:218: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001:215: The statement could not be removed.
        */
        throw std::runtime_error(std::string("copy2: ") +
                                 dpct::get_error_string_dummy(e));
}

void add_scaled(const ScaledRows& rows, int count, int n, void* stream) {
    if (count < 1 || count > 32) throw std::invalid_argument("add_scaled: 1 to 32 vectors");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class add_scaled_kernel_a17364>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(n, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    add_scaled_kernel(rows, count, n);
                });
    }
}

void swiglu_clamp_rows(const float* gate_up, float* out, int n, int rows, float limit, void* stream,
                       const ExpertScales* scales) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->submit([&](sycl::handler &cgh) {
        ExpertScales scales_scales_ExpertScales_ct4 =
            scales ? *scales : ExpertScales{};

        cgh.parallel_for<
            dpct_kernel_name<class swiglu_clamp_rows_kernel_e823ef>>(
            sycl::nd_range<3>(sycl::range(1, rows, grid(n, 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                swiglu_clamp_rows_kernel(gate_up, out, n, limit,
                                         scales_scales_ExpertScales_ct4);
            });
    });
}

void expert_weights(float* w, const float* scale, const int32_t* ids, int rows, void* stream) {
    if (rows > 0) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<
                dpct_kernel_name<class expert_weights_kernel_7d9b0d>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(rows, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    expert_weights_kernel(w, scale, ids, rows);
                });
    }
}

void geglu(const float* gate, const float* up, float* out, int n, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->parallel_for<dpct_kernel_name<class geglu_kernel_86b853>>(
        sycl::nd_range<3>(sycl::range(1, 1, grid(n, 256)) *
                              sycl::range(1, 1, 256),
                          sycl::range(1, 1, 256)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
            geglu_kernel(gate, up, out, n);
        });
}

void geglu_rows(const float* gate_up, float* out, int n, int rows, void* stream, const ExpertScales* scales) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->submit([&](sycl::handler &cgh) {
        ExpertScales scales_scales_ExpertScales_ct3 =
            scales ? *scales : ExpertScales{};

        cgh.parallel_for<dpct_kernel_name<class geglu_rows_kernel_2e101e>>(
            sycl::nd_range<3>(sycl::range(1, rows, grid(n, 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                geglu_rows_kernel(gate_up, out, n,
                                  scales_scales_ExpertScales_ct3);
            });
    });
}

void route_rows(const float* logits, int n_expert, int n_used, int rows, int gate, float in_scale, const float* bias,
                const float* escale, bool norm, float w_scale, int32_t* ids, float* w, void* stream) {
    if (rows <= 0) return;
    if (n_expert > ROUTE_PER * WARP || n_used > WARP)
        throw std::invalid_argument("route_rows: more than 512 experts, or more than 32 a token");
    const auto launch = [&](auto kernel) {
        dpct::kernel_launcher::launch(kernel, grid(rows * WARP, 256), 256, 0,
                                      cs(stream), logits, n_expert, n_used,
                                      rows, gate, in_scale, bias, escale, norm,
                                      w_scale, ids, w);
    };
    if (n_expert <= 2 * WARP) launch(route_kernel_wrapper<2>);
    else if (n_expert <= 4 * WARP) launch(route_kernel_wrapper<4>);
    else if (n_expert <= 8 * WARP) launch(route_kernel_wrapper<8>);
    else launch(route_kernel_wrapper<ROUTE_PER>);
}

void moe_sum(const float* x, const float* w, float* acc, int n, int n_used, int rows, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->parallel_for<dpct_kernel_name<class moe_sum_kernel_15e8bc>>(
        sycl::nd_range<3>(sycl::range(1, rows, grid(n, 256)) *
                              sycl::range(1, 1, 256),
                          sycl::range(1, 1, 256)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
            moe_sum_kernel(x, w, acc, n, n_used);
        });
}

void add_bias(const float* b, float* acc, int n, int rows, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->parallel_for<dpct_kernel_name<class add_bias_kernel_c713a5>>(
        sycl::nd_range<3>(sycl::range(1, rows, grid(n, 256)) *
                              sycl::range(1, 1, 256),
                          sycl::range(1, 1, 256)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
            add_bias_kernel(b, acc, n);
        });
}

void head_norm_rope(const float* x, int x_stride, const float* w, float* y, int dim, int heads, int n_rot,
                    float theta_scale, int64_t pos, int rows, float eps, void* stream, const float* freq,
                    const int32_t* axes, int sections, const long long* at) {
    /*
    DPCT1049:90: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)
        ->parallel_for<dpct_kernel_name<class head_norm_rope_kernel_3239d8>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows * heads) *
                                  sycl::range(1, 1, dim >= 256 ? 256 : 128),
                              sycl::range(1, 1, dim >= 256 ? 256 : 128)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                head_norm_rope_kernel(x, x_stride, w, y, dim, heads, n_rot,
                                      theta_scale, freq, (long long)pos, eps,
                                      axes, sections, at);
            });
}

void attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows, float* score,
            int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window,
            int ring, bool whole, const long long* at, int room, size_t score_floats, Kvq8 kq8, Kvq8 vq8,
            const float* sinks, size_t part_floats) {
    if (at && (room < 1 || (whole && rows > chunk)))
        throw std::invalid_argument("attend: a position on the card needs its room, and with `whole` one launch");
    if (dim % SCORE_VALUES || uintptr_t(q) % (SCORE_VALUES * sizeof(float)) || uintptr_t(kcache) % (SCORE_VALUES * 2))
        throw std::invalid_argument("attend: queries and keys are read four values at a time");
    const long long last = !whole ? -1 : at ? -2 : pos + rows - 1;   // -2: the launch's last token (attn_last())
    const int per = n_head / kv_heads;
    const bool kq = kq8.codes != nullptr, vq = vq8.codes != nullptr;
    // attn_split_kernel, a launch as many tokens as score has room for every block's sums of at the longest the call
    // can attend
    const int most = grid(size_t(at ? room : pos + rows), SPLIT_KEYS);
    const int fit = (int) std::min<size_t>(chunk, score_floats / (size_t(n_head) * most * (dim + 2)));
    const bool split64 = dim == 64 && per == 4, split128 = dim == 128 && per == 8;
#if defined(STRATA_USE_HIP)
    // STRATA_HIP_SPLIT_ATTEND=1 (opt-in): up to 4 tokens over a long q8_0 cache, heads of 256 values in groups of 8, in one flash-decoding pass that
    // reads each key and value once for all the tokens (hip_split_attend.cuh).  `part_floats` of `score` hold its partial sums.
    if (kq && vq && part_floats > 0 && hip_split_attend::requested(std::getenv("STRATA_HIP_SPLIT_ATTEND"))) {
        const int all = int(at ? room : pos + rows);
        if (all >= 4096 && hip_split_attend::fits(dim, n_head, kv_heads, all, std::min(rows, hip_split_attend::kMaxRows), window, ring, whole,
                                                  sinks != nullptr, part_floats)) {
            for (int first = 0, c = 0; first < rows; first += c) {
                c = std::min(rows - first, hip_split_attend::kMaxRows);
                const int keys = int(at ? room : pos + first + c);
                hip_split_attend::run(q, kq8.codes, kq8.scales, vq8.codes, vq8.scales, score, out, n_head, kv_heads, (long long) pos, first, c,
                                      scale, (keys + hip_split_attend::kKeys - 1) / hip_split_attend::kKeys, at, (hipStream_t) cs(stream),
                                      hip_split_attend::mode(std::getenv("STRATA_HIP_SPLIT_ATTEND")));
            }
            return;
        }
    }
#endif
    if ((split64 || split128) && window == 0 && ring == 0 && !whole && !kq && !vq && fit > 0) {
        for (int first = 0, c = 0; first < rows; first += c) {
            c = std::min(fit, rows - first);
            const int blocks = grid(size_t(at ? room : pos + first + c), SPLIT_KEYS);
            const dpct::dim3 split(blocks, kv_heads, c), join(n_head, c);
            if (split64) {
                {
                    auto exp_props =
                        sycl::ext::oneapi::experimental::properties{
                            sycl::ext::oneapi::experimental::use_root_sync};
                    dpct::has_capability_or_fail(cs(stream)->get_device(),
                                                 {sycl::aspect::fp16});

                    cs(stream)
                        ->parallel_for<dpct_kernel_name<
                            class attn_split_kernel_76ad53,
                            dpct_kernel_scalar<64>, dpct_kernel_scalar<4>>>(
                            sycl::nd_range<3>(split *
                                                  sycl::range(1, 1, 4 * WARP),
                                              sycl::range(1, 1, 4 * WARP)),
                            exp_props,
                            [=](sycl::nd_item<3> item_ct1)
                                [[sycl::reqd_sub_group_size(32)]] {
                                    attn_split_kernel<64, 4>(
                                        q, (const sycl::half *)kcache,
                                        (const sycl::half *)vcache, score,
                                        n_head, kv_heads, (long long)pos, first,
                                        c, scale, blocks, at);
                                });
                }
                /*
                DPCT1049:91: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                {
                    auto exp_props =
                        sycl::ext::oneapi::experimental::properties{
                            sycl::ext::oneapi::experimental::use_root_sync};

                    cs(stream)
                        ->parallel_for<
                            dpct_kernel_name<class attn_join_kernel_a2c887,
                                             dpct_kernel_scalar<64>>>(
                            sycl::nd_range<3>(
                                join * sycl::range(1, 1, JOIN_WARPS * WARP),
                                sycl::range(1, 1, JOIN_WARPS * WARP)),
                            exp_props,
                            [=](sycl::nd_item<3> item_ct1)
                                [[sycl::reqd_sub_group_size(32)]] {
                                    attn_join_kernel<64>(score, out, n_head,
                                                         (long long)pos, first,
                                                         c, blocks, at, sinks);
                                });
                }
            } else {
                {
                    auto exp_props =
                        sycl::ext::oneapi::experimental::properties{
                            sycl::ext::oneapi::experimental::use_root_sync};
                    dpct::has_capability_or_fail(cs(stream)->get_device(),
                                                 {sycl::aspect::fp16});

                    cs(stream)
                        ->parallel_for<dpct_kernel_name<
                            class attn_split_kernel_9bb26f,
                            dpct_kernel_scalar<128>, dpct_kernel_scalar<8>>>(
                            sycl::nd_range<3>(split *
                                                  sycl::range(1, 1, 8 * WARP),
                                              sycl::range(1, 1, 8 * WARP)),
                            exp_props,
                            [=](sycl::nd_item<3> item_ct1)
                                [[sycl::reqd_sub_group_size(32)]] {
                                    attn_split_kernel<128, 8>(
                                        q, (const sycl::half *)kcache,
                                        (const sycl::half *)vcache, score,
                                        n_head, kv_heads, (long long)pos, first,
                                        c, scale, blocks, at);
                                });
                }
                /*
                DPCT1049:92: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                {
                    auto exp_props =
                        sycl::ext::oneapi::experimental::properties{
                            sycl::ext::oneapi::experimental::use_root_sync};

                    cs(stream)
                        ->parallel_for<
                            dpct_kernel_name<class attn_join_kernel_58fabc,
                                             dpct_kernel_scalar<128>>>(
                            sycl::nd_range<3>(
                                join * sycl::range(1, 1, JOIN_WARPS * WARP),
                                sycl::range(1, 1, JOIN_WARPS * WARP)),
                            exp_props,
                            [=](sycl::nd_item<3> item_ct1)
                                [[sycl::reqd_sub_group_size(32)]] {
                                    attn_join_kernel<128>(score, out, n_head,
                                                          (long long)pos, first,
                                                          c, blocks, at, sinks);
                                });
                }
            }
        }
        return;
    }
    for (int first = 0; first < rows; first += chunk) {
        const int c = std::min(chunk, rows - first);
        // the launch's span: from the first row its first token attends to the last its last token does
        const long long p = pos + first, lo = window > 0 && p >= window ? p - window + 1 : 0;
        const int keys = at ? room : int((whole ? last : p + c - 1) - lo) + 1;
        const unsigned units = grid(size_t(keys), SCORE_KEYS) * kv_heads, threads = SCORE_WARPS * WARP;
        const auto reg = [&](auto kernel) {
            dpct::kernel_launcher::launch(
                kernel, grid(units, SCORE_WARPS), threads, 0, cs(stream), q,
                (const sycl::half *)kcache, score, keys, n_head, kv_heads,
                (long long)pos, first, c, scale, window, ring, last, at,
                kq8.codes, kq8.scales);
        };
        // The queries in registers where a head has 128, 256 or 512 values and a group 4 or 8 heads.  With fewer
        // heads a group the cache is the most of what is read and the other kernel is as fast.  A q8 key cache
        // takes the same kernel the fp16 cache does, which reads it as Q8_0.
        if (per % 4 == 0 && dim == 128) {
            if (kq) reg(attn_score_reg_kernel_wrapper<4, 1, true>);
            else reg(attn_score_reg_kernel_wrapper<4, 1, false>);
        } else if (per % 8 == 0 && dim == 256) {
            if (kq) reg(attn_score_reg_kernel_wrapper<8, 2, true>);
            else reg(attn_score_reg_kernel_wrapper<8, 2, false>);
        } else if (per % 4 == 0 && dim == 256) {
            if (kq) reg(attn_score_reg_kernel_wrapper<4, 2, true>);
            else reg(attn_score_reg_kernel_wrapper<4, 2, false>);
        } else if (per % 4 == 0 && dim == 512) {
            if (kq) reg(attn_score_reg_kernel_wrapper<4, 4, true>);
            else reg(attn_score_reg_kernel_wrapper<4, 4, false>);
        } else if (kq)
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(cs(stream)->get_device(),
                                         {sycl::aspect::fp16});

            cs(stream)
                ->parallel_for<dpct_kernel_name<class attn_score_kernel_46c748,
                                                dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(
                        sycl::range(
                            1, 1, grid(size_t(keys) * kv_heads, SCORE_WARPS)) *
                            sycl::range(1, 1, threads),
                        sycl::range(1, 1, threads)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            attn_score_kernel<true>(
                                q, (const sycl::half *)kcache, score, keys, dim,
                                n_head, kv_heads, (long long)pos, first, c,
                                scale, window, ring, last, at, kq8.codes,
                                kq8.scales);
                        });
        } else {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(cs(stream)->get_device(),
                                         {sycl::aspect::fp16});

            cs(stream)
                ->parallel_for<dpct_kernel_name<class attn_score_kernel_9cf923,
                                                dpct_kernel_scalar<false>>>(
                    sycl::nd_range<3>(
                        sycl::range(
                            1, 1, grid(size_t(keys) * kv_heads, SCORE_WARPS)) *
                            sycl::range(1, 1, threads),
                        sycl::range(1, 1, threads)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            attn_score_kernel<false>(
                                q, (const sycl::half *)kcache, score, keys, dim,
                                n_head, kv_heads, (long long)pos, first, c,
                                scale, window, ring, last, at, nullptr,
                                nullptr);
                        });
        }
        // the weights as attn_value_t_kernel reads them, behind the scores where the buffer has room for them.
        // A q8 value cache takes the same value kernel the fp16 cache does, which reads it as Q8_0.
        const int blocks = grid(c * per, VALUE_QUERIES);
        float* wt = per >= 4 && size_t(kv_heads) * blocks * VALUE_QUERIES <= size_t(chunk - c) * n_head
                        ? score + size_t(c) * n_head * keys : nullptr;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<
                    dpct_kernel_name<class attn_softmax_kernel_739a16>>(
                    sycl::nd_range<3>(sycl::range(1, c, n_head) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            attn_softmax_kernel(score, keys, (long long)pos,
                                                first, c, window, last, at, wt,
                                                per, sinks);
                        });
        }
        // Two blocks of queries a launch block read the values half as often, where the launch still has
        // blocks for most of the card's 36 multiprocessors: for Qwen3.6's two tokens they are half as many and
        // slower, for three or four tokens and for Gemma 4's full layers faster.
        // The scores are dead after softmax. Reuse them for the partial sums,
        // leaving the transposed weights behind them intact. Off by default.
#if defined(STRATA_USE_HIP)
        if (wt && hip_value_split::requested(std::getenv("STRATA_HIP_VALUE_SPLIT")) &&
            hip_value_split::fits(dim, n_head, kv_heads, keys, c, window, ring, whole, size_t(wt - score))) {
            hip_value_split::run(wt, vcache, vq8.codes, vq8.scales, score, out, keys, n_head, kv_heads, first, c,
                                 (long long) pos, at, (hipStream_t) stream);
            continue;
        }
#endif
        const int dims = grid(dim, VALUE_DIMS);
        if (wt && blocks > 1 && kv_heads * dims * grid(blocks, 2) >= 32) {
            if (vq)
                /*
                DPCT1049:93: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};
                dpct::has_capability_or_fail(cs(stream)->get_device(),
                                             {sycl::aspect::fp16});

                cs(stream)
                    ->parallel_for<dpct_kernel_name<
                        class attn_value_t_kernel_751a83, dpct_kernel_scalar<2>,
                        dpct_kernel_scalar<true>>>(
                        sycl::nd_range<3>(
                            sycl::range(1, grid(blocks, 2), kv_heads * dims) *
                                sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES),
                            sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES)),
                        exp_props, [=](sycl::nd_item<3> item_ct1) {
                            attn_value_t_kernel<2, true>(
                                wt, (const sycl::half *)vcache, out, keys, dim,
                                n_head, kv_heads, (long long)pos, first, c,
                                window, ring, last, at, vq8.codes, vq8.scales);
                        });
            } else
            /*
            DPCT1049:94: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};
                dpct::has_capability_or_fail(cs(stream)->get_device(),
                                             {sycl::aspect::fp16});

                cs(stream)
                    ->parallel_for<dpct_kernel_name<
                        class attn_value_t_kernel_d2d657, dpct_kernel_scalar<2>,
                        dpct_kernel_scalar<false>>>(
                        sycl::nd_range<3>(
                            sycl::range(1, grid(blocks, 2), kv_heads * dims) *
                                sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES),
                            sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES)),
                        exp_props, [=](sycl::nd_item<3> item_ct1) {
                            attn_value_t_kernel<2, false>(
                                wt, (const sycl::half *)vcache, out, keys, dim,
                                n_head, kv_heads, (long long)pos, first, c,
                                window, ring, last, at, nullptr, nullptr);
                        });
            }
        } else if (wt) {
            if (vq)
                /*
                DPCT1049:95: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};
                dpct::has_capability_or_fail(cs(stream)->get_device(),
                                             {sycl::aspect::fp16});

                cs(stream)
                    ->parallel_for<dpct_kernel_name<
                        class attn_value_t_kernel_24649f, dpct_kernel_scalar<1>,
                        dpct_kernel_scalar<true>>>(
                        sycl::nd_range<3>(
                            sycl::range(1, blocks, kv_heads * dims) *
                                sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES),
                            sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES)),
                        exp_props, [=](sycl::nd_item<3> item_ct1) {
                            attn_value_t_kernel<1, true>(
                                wt, (const sycl::half *)vcache, out, keys, dim,
                                n_head, kv_heads, (long long)pos, first, c,
                                window, ring, last, at, vq8.codes, vq8.scales);
                        });
            } else
            /*
            DPCT1049:96: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};
                dpct::has_capability_or_fail(cs(stream)->get_device(),
                                             {sycl::aspect::fp16});

                cs(stream)
                    ->parallel_for<dpct_kernel_name<
                        class attn_value_t_kernel_134dd4, dpct_kernel_scalar<1>,
                        dpct_kernel_scalar<false>>>(
                        sycl::nd_range<3>(
                            sycl::range(1, blocks, kv_heads * dims) *
                                sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES),
                            sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES)),
                        exp_props, [=](sycl::nd_item<3> item_ct1) {
                            attn_value_t_kernel<1, false>(
                                wt, (const sycl::half *)vcache, out, keys, dim,
                                n_head, kv_heads, (long long)pos, first, c,
                                window, ring, last, at, nullptr, nullptr);
                        });
            }
        } else if (vq)
            /*
            DPCT1049:97: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(cs(stream)->get_device(),
                                         {sycl::aspect::fp16});

            cs(stream)
                ->parallel_for<dpct_kernel_name<class attn_value_kernel_cec4b3,
                                                dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(
                        sycl::range(1, blocks, kv_heads * dims) *
                            sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES),
                        sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            attn_value_kernel<true>(
                                score, (const sycl::half *)vcache, out, keys,
                                dim, n_head, kv_heads, (long long)pos, first, c,
                                window, ring, last, at, vq8.codes, vq8.scales);
                        });
        } else
        /*
        DPCT1049:98: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(cs(stream)->get_device(),
                                         {sycl::aspect::fp16});

            cs(stream)
                ->parallel_for<dpct_kernel_name<class attn_value_kernel_4a86b6,
                                                dpct_kernel_scalar<false>>>(
                    sycl::nd_range<3>(
                        sycl::range(1, blocks, kv_heads * dims) *
                            sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES),
                        sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            attn_value_kernel<false>(
                                score, (const sycl::half *)vcache, out, keys,
                                dim, n_head, kv_heads, (long long)pos, first, c,
                                window, ring, last, at, nullptr, nullptr);
                        });
        }
    }
}

// The rows of a q8_0 cache a batch product is about to read, dequantized into the fp16 workspace it reads
// instead: out[r * width + i] = (float)codes[row + r][i] * __half2float(scales[row + r][i / 32]).  blockIdx.y is
// the row of the block (blockIdx.x the columns).  A block never wraps a ring: it ends where a ring starts over.
__dpct_inline__ void dequant_q8_kernel(const int8_t *__restrict__ codes,
                                       const uint16_t *__restrict__ scales,
                                       sycl::half *__restrict__ out, int width,
                                       long long row) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= width) return;
    const size_t r = (size_t)(row + item_ct1.get_group(1)), at = r * width;
    const float d =
        sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(
                                     scales[r * (width >> 5) + (i >> 5)]))
            .convert<float, sycl::rounding_mode::automatic>()[0];
    out[(size_t)item_ct1.get_group(1) * width + i] =
        sycl::vec<float, 1>((float)codes[at + i] * d)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

size_t attend_split_floats(int chunk, int n_head, int max_rows, int dim) {
    return size_t(chunk) * n_head * grid(size_t(max_rows), SPLIT_KEYS) * (dim + 2);
}

// ---------------------------------------------- the int8 attention caches' readers (--kv int8 / int8k)

namespace {

// A Q8_0 block's delta (its two leading bytes, fp16) and the code at `within` of the 32 that follow.  A block is
// 34 bytes - not a power of two - so a value's place in the cache is an arithmetic offset and not a pointer step,
// which is why every reader below computes its block instead of advancing a typed pointer.
__dpct_inline__ float q8_delta(const uint8_t *__restrict__ blk) {
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>(
                   *reinterpret_cast<const uint16_t *>(blk)))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}
__dpct_inline__ const int8_t *q8_codes(const uint8_t *__restrict__ blk,
                                       int within) {
    return reinterpret_cast<const int8_t*>(blk + 2 + within);
}

// A block's delta, and the four codes at `at` of a row as an int for __dp4a.  One load: the codes of a row are
// back to back, so every four of them is 4-aligned (kQ8_0Values is a multiple of four and a row is n_kv *
// head_dim codes).
__dpct_inline__ float q8_half(uint16_t h) {
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>(h))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}
__dpct_inline__ int q8_pack4(const int8_t *__restrict__ code, int at) {
    return *reinterpret_cast<const int*>(code + at);
}

// attn_scores() with both sides in --kv int8's form.  A lane owns a run of dim / WARP of the head's values, a
// whole number of dp4a steps and no more than one block of them (PER is a divisor of kQ8_0Values here), so the
// delta is multiplied in once a run.  That is the difference from the interleaved four-a-lane form of the fp16
// kernel: a step there crossed blocks and scaled every product.  llama.cpp's vec_dot_q8_0_q8_1 has this shape,
// and it is why a quantized cache need not be read more slowly than an fp16 one.
template <int N, int PER> // PER = dim / WARP, the values a lane owns
__dpct_inline__ void
attn_scores_q8(const int8_t *__restrict__ qc, const uint16_t *__restrict__ qd,
               const int8_t *__restrict__ kc, const uint16_t *__restrict__ kd,
               float *__restrict__ score, int n_keys, float scale) {
    static_assert(SCORE_VALUES == 4, "a dp4a takes four codes");
    static_assert(PER % SCORE_VALUES == 0, "a lane's run is a whole number of dp4a steps");
    static_assert(kQ8_0Values % PER == 0, "a lane's run lies inside one block, so one delta scales it");
    constexpr int dim = PER * WARP, pd = dim / kQ8_0Values;
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) %
        WARP;
    const int base = lane * PER, block = base / kQ8_0Values;
    const float dk = q8_half(kd[block]);
    const int* k4 = reinterpret_cast<const int*>(kc + base);
    int sumi[N];
#pragma unroll
    for (int i = 0; i < N; ++i) sumi[i] = 0;
#pragma unroll
    for (int j = 0; j < PER / SCORE_VALUES; ++j)
#pragma unroll
        for (int i = 0; i < N; ++i) sumi[i] = STRATA_DP4A(q8_pack4(qc + (size_t) i * dim, base + SCORE_VALUES * j),
                                                     k4[j], sumi[i]);
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) acc[i] = (float) sumi[i] * (dk * q8_half(qd[(size_t) i * pd + block]));
#pragma unroll
    for (int i = 0; i < N; ++i) {
        /*
DPCT1108:99: '__shfl_down_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
        for (int off = WARP / 2; off > 0; off >>= 1) acc[i] +=
            dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                acc[i], off);
        if (lane == 0) score[size_t(i) * n_keys] = acc[i] * scale;
    }
}

// attn_scores_by() with both sides in --kv int8's form.
template <int N, int PER>
__dpct_inline__ void attn_scores_by_q8(int n, const int8_t *__restrict__ qc,
                                       const uint16_t *__restrict__ qd,
                                       const int8_t *__restrict__ kc,
                                       const uint16_t *__restrict__ kd,
                                       float *__restrict__ score, int n_keys,
                                       float scale) {
    constexpr int dim = PER * WARP, pd = dim / kQ8_0Values;
#pragma unroll
    for (; n >= N; n -= N, qc += (size_t)N * dim, qd += (size_t)N * pd,
                   score += size_t(N) * n_keys)
        attn_scores_q8<N, PER>(qc, qd, kc, kd, score, n_keys, scale);
    if constexpr (N > 1) attn_scores_by_q8<N / 2, PER>(n, qc, qd, kc, kd, score, n_keys, scale);
}

// attn_scores_q8 for a head NARROWER than a warp: LANES of the warp's lanes own a head (dim = PER * LANES), so its
// lanes cover WARP / LANES of the group's heads at once and a call walks the queries a round of those at a time.
// The form above cannot describe such a head, and this is why: a lane must own a whole dp4a step - four codes -
// and no more than one 32-value block, so PER is 4, 8 or 16 and a head of 64 values needs 4, 8 or 16 lanes.
// LFM2.5's 24B-A2B has 32 heads of 64 values, 8 of them holding keys, so PER = 8 over 8 lanes - the group's four
// queries are exactly one round of a warp.  The products, the deltas and the sums are the form above's, and so is
// the order they are added in: a lane's run is one delta's, and the tree is the same with LANES for WARP (an XOR
// by less than LANES never leaves a head's own lanes).
template <int PER, int LANES>
__dpct_inline__ void attn_scores_q8_sub(const int8_t *__restrict__ qc,
                                        const uint16_t *__restrict__ qd,
                                        const int8_t *__restrict__ kc,
                                        const uint16_t *__restrict__ kd,
                                        float *__restrict__ score, int n_keys,
                                        float scale, int n_q) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    static_assert(SCORE_VALUES == 4, "a dp4a takes four codes");
    static_assert(PER % SCORE_VALUES == 0, "a lane's run is a whole number of dp4a steps");
    static_assert(kQ8_0Values % PER == 0, "a lane's run lies inside one block, so one delta scales it");
    static_assert(WARP % LANES == 0 && LANES < WARP, "a head is whole lanes and narrower than a warp");
    constexpr int dim = PER * LANES, pd = dim / kQ8_0Values, HPW = WARP / LANES;
    const int lane = item_ct1.get_local_id(2) % LANES,
              half = item_ct1.get_local_id(2) / LANES % HPW;
    const int base = lane * PER, block = base / kQ8_0Values;
    const float dk = q8_half(kd[block]);
    const int* k4 = reinterpret_cast<const int*>(kc + base);
    // The rounds are the warp's, not a lane's: a lane whose head is past the call's queries takes the last one and
    // only the first of that head's lanes writes it, so every lane keeps its place in the shuffles below.
    for (int h0 = 0; h0 < n_q; h0 += HPW) {
        const int h = h0 + half, at = h < n_q ? h : n_q - 1;
        const int8_t* qch = qc + (size_t) at * dim;
        const uint16_t* qdh = qd + (size_t) at * pd;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < PER / SCORE_VALUES; ++j)
            sumi = STRATA_DP4A(q8_pack4(qch, base + SCORE_VALUES * j), k4[j], sumi);
        float acc = (float) sumi * (dk * q8_half(qdh[block]));
#pragma unroll
        /*
        DPCT1108:100: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        for (int off = LANES / 2; off > 0; off >>= 1) acc +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                acc, off);
        if (lane == 0 && h < n_q) score[(size_t) at * n_keys] = acc * scale;
    }
}

// attn_scores_by_q8 for such a head: the same blocking of the call's queries, one round of a warp the block.
template <int PER, int LANES>
__dpct_inline__ void attn_scores_by_q8_sub(int n, const int8_t *__restrict__ qc,
                                           const uint16_t *__restrict__ qd,
                                           const int8_t *__restrict__ kc,
                                           const uint16_t *__restrict__ kd,
                                           float *__restrict__ score,
                                           int n_keys, float scale) {
    constexpr int dim = PER * LANES, pd = dim / kQ8_0Values;
#pragma unroll
    for (; n >= SCORE_QUERIES; n -= SCORE_QUERIES,
                               qc += (size_t)SCORE_QUERIES * dim,
                               qd += (size_t)SCORE_QUERIES * pd,
                               score += (size_t)SCORE_QUERIES * n_keys)
        attn_scores_q8_sub<PER, LANES>(qc, qd, kc, kd, score, n_keys, scale, SCORE_QUERIES);
    if (n > 0) attn_scores_q8_sub<PER, LANES>(qc, qd, kc, kd, score, n_keys, scale, n);
}

// attn_score_kernel with both sides in --kv int8's form: the same warps, the same rows, the same sums.  PER is
// the values a lane owns, so a head is PER * LANES wide: LANES = WARP for the 128, 256 and 512-value heads this
// engine's own models have, and fewer for a narrower one (attn_scores_q8_sub above; 24B-A2B is 64 over 8 lanes).
template <int PER, int LANES = WARP>
void attn_score_int8_kernel(const KvInt8 q8, const KvInt8 kq, float* __restrict__ score, int n_keys,
                                     int n_head, int kv_heads, long long pos, int first, int tokens, float scale,
                                     int window, int ring, long long last, const long long* __restrict__ at) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int dim = PER * LANES, pd = dim / kQ8_0Values;
    const int unit =
        item_ct1.get_group(2) * (item_ct1.get_local_range(2) / WARP) +
        item_ct1.get_local_id(2) / WARP;
    const int key = unit / kv_heads, g = unit % kv_heads, per = n_head / kv_heads;
    long long lo;
    if (key >= attn_span((at ? *at : pos) + first, tokens, window, last, &lo)) return;
    const long long row = ring > 0 ? (lo + key) % ring : lo + key;
    const int8_t* kc = kq.code + ((size_t) row * kv_heads + g) * dim;
    const uint16_t* kd = kq.delta + ((size_t) row * kv_heads + g) * pd;
    const int8_t* qc = q8.code + (size_t)(first * n_head + g * per) * dim;
    const uint16_t* qd = q8.delta + (size_t)(first * n_head + g * per) * pd;
    score += size_t(g) * per * n_keys + key;
#pragma unroll
    for (int t = 0; t < tokens; ++t, qc += (size_t)n_head * dim,
             qd += (size_t)n_head * pd, score += size_t(n_head) * n_keys) {
        if constexpr (LANES == WARP) attn_scores_by_q8<SCORE_QUERIES, PER>(per, qc, qd, kc, kd, score, n_keys, scale);
        else attn_scores_by_q8_sub<PER, LANES>(per, qc, qd, kc, kd, score, n_keys, scale);
    }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <int PER, int LANES = WARP>
void attn_score_int8_kernel_wrapper(const KvInt8 q8, const KvInt8 kq,
                                    float *__restrict score, int n_keys,
                                    int n_head, int kv_heads, long long pos,
                                    int first, int tokens, float scale,
                                    int window, int ring, long long last,
                                    const long long *__restrict at) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    queue.parallel_for<dpct_kernel_name<class attn_score_int8_kernel_, dpct_kernel_scalar<PER>, dpct_kernel_scalar<LANES>>>(
        nr, exp_props,
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            attn_score_int8_kernel<PER, LANES>(q8, kq, score, n_keys, n_head,
                                               kv_heads, pos, first, tokens,
                                               scale, window, ring, last, at);
        });
}

// attn_score_reg_kernel's counterpart for --kv int8: a warp takes SCORE_KEYS rows of a group and N of the group's
// heads at a time, and a token's queries - their codes and their deltas - stay in its registers while it walks the
// rows.  The plain kernel reads each query's codes again for every row; on a prompt batch (8 queries at once, not
// the 1 of a decode) that re-read is 8x the traffic, which is the case this exists for.
template <int N, int PER>
/*
DPCT1110:101: The total declared local variable size in device function
attn_score_reg_int8_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void attn_score_reg_int8_kernel(const KvInt8 q8, const KvInt8 kq,
                                float *__restrict__ score, int n_keys,
                                int n_head, int kv_heads, long long pos,
                                int first, int tokens, float scale, int window,
                                int ring, long long last,
                                const long long *__restrict__ at) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    static_assert(SCORE_VALUES == 4, "a dp4a takes four codes");
    static_assert(PER % SCORE_VALUES == 0, "a lane's run is a whole number of dp4a steps");
    static_assert(kQ8_0Values % PER == 0, "a lane's run lies inside one block, so one delta scales it");
    constexpr int dim = PER * WARP, STEPS = PER / SCORE_VALUES, pd = dim / kQ8_0Values;
    const int unit = item_ct1.get_group(2) * SCORE_WARPS +
                     item_ct1.get_local_id(2) / WARP; // a group's block of rows
    const int g = unit % kv_heads, k0 = unit / kv_heads * SCORE_KEYS,
              lane = item_ct1.get_local_id(2) % WARP;
    long long lo;
    const int span = attn_span((at ? *at : pos) + first, tokens, window, last, &lo);
    if (k0 >= span) return;
    const int rows = sycl::min(SCORE_KEYS, span - k0), per = n_head / kv_heads;
    const int base = lane * PER, blk = base / kQ8_0Values;
    for (int th = 0; th < tokens * per; th += N) {
        const int t = th / per, h = g * per + th % per;
        const int8_t* qc = q8.code + ((size_t(first) + t) * n_head + h) * dim + base;
        const uint16_t* qd = q8.delta + ((size_t(first) + t) * n_head + h) * pd + blk;
        int qcode[N][STEPS];
        float qdel[N];
#pragma unroll
        for (int i = 0; i < N; ++i) {
#pragma unroll
            for (int s = 0; s < STEPS; ++s) qcode[i][s] = q8_pack4(qc + (size_t) i * dim, SCORE_VALUES * s);
            qdel[i] = q8_half(qd[(size_t) i * pd]);
        }
        float* st = score + (size_t(t) * n_head + h) * n_keys + k0;
        long long row = ring > 0 ? (lo + k0) % ring : lo + k0;
        for (int r = 0; r < rows; ++r) {
            const int8_t* kc = kq.code + ((size_t) row * kv_heads + g) * dim + base;
            const float dk = q8_half(kq.delta[((size_t) row * kv_heads + g) * pd + blk]);
            int kcode[STEPS];   // the row's codes, read once for the N products they take part in
#pragma unroll
            for (int s = 0; s < STEPS; ++s) kcode[s] = q8_pack4(kc, SCORE_VALUES * s);
            float acc[N];
#pragma unroll
            for (int i = 0; i < N; ++i) {
                int sum = 0;
#pragma unroll
                for (int s = 0; s < STEPS; ++s) sum = STRATA_DP4A(qcode[i][s], kcode[s], sum);
                acc[i] = (float) sum * (dk * qdel[i]);
            }
            int which;
            const float sum = scores_sum<N>(acc, lane, &which);
            if ((lane & (WARP / N - 1)) == 0) st[size_t(which) * n_keys + r] = sum * scale;
            if (++row == ring) row = 0;
        }
    }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <int N, int PER>
void attn_score_reg_int8_kernel_wrapper(const KvInt8 q8, const KvInt8 kq,
                                        float *__restrict score, int n_keys,
                                        int n_head, int kv_heads, long long pos,
                                        int first, int tokens, float scale,
                                        int window, int ring, long long last,
                                        const long long *__restrict at) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    queue.parallel_for<dpct_kernel_name<class attn_score_reg_int8_kernel_, dpct_kernel_scalar<N>, dpct_kernel_scalar<PER>>>(
        nr, exp_props,
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            attn_score_reg_int8_kernel<N, PER>(q8, kq, score, n_keys, n_head,
                                               kv_heads, pos, first, tokens,
                                               scale, window, ring, last, at);
        });
}

// The int8 tensor-core path.  A warp takes 8 keys of a group against 16 of that group's queries, one m16n8k32 a
// block of 32 channels, so a 256-wide head is 8 instructions where dp4a needs 256 - 3.4x on the arithmetic
// (measured by probe_mma_speed).  A lane loads the B fragment for key g and the A fragments for query rows g and
// g+8; the C block comes back as (query g, key 2*tig), (query g, key 2*tig+1), (query g+8, key 2*tig), (query g+8,
// key 2*tig+1), which is why the deltas of the columns - not of the loaded key - are the ones that scale it.
// The last key block and the last query block can reach past the end: those lanes read the last valid row instead
// and their results are dropped, which is cheaper than a second kernel for the tail.
template <int PER>
/*
DPCT1110:102: The total declared local variable size in device function
attn_score_mma_int8_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void attn_score_mma_int8_kernel(const KvInt8 q8, const KvInt8 kq,
                                float *__restrict__ score, int n_keys,
                                int n_head, int kv_heads, long long pos,
                                int first, int tokens, float scale, int window,
                                int ring, long long last,
                                const long long *__restrict__ at) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    static_assert(PER == 8,
                  "this path is written for a head of SCORE_KEYS lanes' worth");
    constexpr int dim = PER * WARP, pd = dim / kQ8_0Values;
    const int lane = item_ct1.get_local_id(2) % WARP,
              warp = item_ct1.get_local_id(2) / WARP;
    const int g = lane >> 2, tig = lane & 3;
    long long lo;
    const int span = attn_span((at ? *at : pos) + first, tokens, window, last, &lo);
    const int per = n_head / kv_heads, nq = tokens * per;
    const int kw_all = (span + SCORE_KEYS - 1) / SCORE_KEYS, qw_all = (nq + 15) / 16;
    const int widx = item_ct1.get_group(2) * SCORE_WARPS + warp;
    if (widx >= kw_all * qw_all * kv_heads) return;
    // the group changes slowest, the key block next: the 16 queries of a block are what the warps around it reuse
    const int gk = widx % kv_heads, rest = widx / kv_heads;
    const int kw = rest % kw_all, qw = rest / kw_all;
    const int key0 = kw * SCORE_KEYS, q0 = qw * 16;
    const int keys_real = sycl::min(SCORE_KEYS, span - key0);
    const int qa_row = q0 + g, qb_row = q0 + g + 8;   // the rows of the C block this lane owns

    // A: this lane's share of 16 queries x 32 channels, per block
    const int qa = sycl::min(qa_row, nq - 1), qb = sycl::min(qb_row, nq - 1);
    const int8_t* qpa = q8.code + ((size_t) (first + qa / per) * n_head + gk * per + qa % per) * dim;
    const int8_t* qpb = q8.code + ((size_t) (first + qb / per) * n_head + gk * per + qb % per) * dim;
    const uint16_t* qda = q8.delta + ((size_t) (first + qa / per) * n_head + gk * per + qa % per) * pd;
    const uint16_t* qdb = q8.delta + ((size_t) (first + qb / per) * n_head + gk * per + qb % per) * pd;
    // B: key g of the block; the deltas of the columns (2*tig and 2*tig+1) are read separately
    const long long krow = lo + sycl::min(key0 + g, span - 1);
    const int8_t* kp = kq.code + ((size_t) (ring > 0 ? krow % ring : krow) * kv_heads + gk) * dim;
    const long long ka_row = lo + sycl::min(key0 + tig * 2, span - 1);
    const long long kb_row = lo + sycl::min(key0 + tig * 2 + 1, span - 1);
    const uint16_t* kda = kq.delta + ((size_t) (ring > 0 ? ka_row % ring : ka_row) * kv_heads + gk) * pd;
    const uint16_t* kdb = kq.delta + ((size_t) (ring > 0 ? kb_row % ring : kb_row) * kv_heads + gk) * pd;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
    for (int b = 0; b < pd; ++b) {
        const unsigned a0 = *reinterpret_cast<const unsigned*>(qpa + b * kQ8_0Values + tig * 4);
        const unsigned a1 = *reinterpret_cast<const unsigned*>(qpb + b * kQ8_0Values + tig * 4);
        const unsigned a2 = *reinterpret_cast<const unsigned*>(qpa + b * kQ8_0Values + 16 + tig * 4);
        const unsigned a3 = *reinterpret_cast<const unsigned*>(qpb + b * kQ8_0Values + 16 + tig * 4);
        const unsigned b0 = *reinterpret_cast<const unsigned*>(kp + b * kQ8_0Values + tig * 4);
        const unsigned b1 = *reinterpret_cast<const unsigned*>(kp + b * kQ8_0Values + 16 + tig * 4);
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
        const float dq0 = q8_half(qda[b]), dq1 = q8_half(qdb[b]);
        const float dk0 = q8_half(kda[b]), dk1 = q8_half(kdb[b]);
        acc0 += (float) c0 * (dq0 * dk0);
        acc1 += (float) c1 * (dq0 * dk1);
        acc2 += (float) c2 * (dq1 * dk0);
        acc3 += (float) c3 * (dq1 * dk1);
    }
    const int key_hi = key0 + keys_real;
    const auto put = [&](int qrow, int key, float v) {
        if (qrow >= nq || key >= key_hi) return;
        score[(size_t) ((size_t) (qrow / per) * n_head + gk * per + qrow % per) * n_keys + key] = v * scale;
    };
    put(qa_row, key0 + tig * 2, acc0);
    put(qa_row, key0 + tig * 2 + 1, acc1);
    put(qb_row, key0 + tig * 2, acc2);
    put(qb_row, key0 + tig * 2 + 1, acc3);
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <int PER>
void attn_score_mma_int8_kernel_wrapper(const KvInt8 q8, const KvInt8 kq,
                                        float *__restrict score, int n_keys,
                                        int n_head, int kv_heads, long long pos,
                                        int first, int tokens, float scale,
                                        int window, int ring, long long last,
                                        const long long *__restrict at) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    queue.parallel_for<dpct_kernel_name<class attn_score_mma_int8_kernel_, dpct_kernel_scalar<PER>>>(
        nr, exp_props,
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            attn_score_mma_int8_kernel<PER>(q8, kq, score, n_keys, n_head,
                                            kv_heads, pos, first, tokens, scale,
                                            window, ring, last, at);
        });
}

// attn_values() with the values in --kv int8's form: the thread's value is one code and its block's delta.  vc and
// vd are the cache's two arrays, head_c/head_d the head's start in each, d the value within the head.
template <int N>
__dpct_inline__ void
attn_values_q8(const float *__restrict__ w, const int8_t *__restrict__ vc,
               const uint16_t *__restrict__ vd, size_t row_c, size_t row_d,
               size_t head_c, size_t head_d, long long row, int ring, int d,
               int span, float *sums) {
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) acc[i] = 0.0f;
    const int8_t* c = vc + size_t(row) * row_c + head_c + d;
    const uint16_t* q = vd + size_t(row) * row_d + head_d + d / kQ8_0Values;
    for (int j =
             sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(
                 2) /
             VALUE_DIMS;
         j < span; j += VALUE_SHARES) {
        const float x = (float) (*c) * q8_half(*q);
        const float weight = w[j];
#pragma unroll
        /*
        DPCT1108:103: '__shfl_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        for (int i = 0; i < N; ++i) acc[i] +=
            dpct::experimental::select_from_sub_group(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                weight, i) *
            x;
        c += VALUE_SHARES * row_c;
        q += VALUE_SHARES * row_d;
        row += VALUE_SHARES;
        if (ring > 0 && row >= ring) row -= ring, c -= size_t(ring) * row_c, q -= size_t(ring) * row_d;
    }
#pragma unroll
    for (int i = 0; i < N; ++i) sums[i] = acc[i];
}

template <int N>
__dpct_inline__ void attn_values_for_int8(
    int n, const float *__restrict__ w, const int8_t *__restrict__ vc,
    const uint16_t *__restrict__ vd, size_t row_c, size_t row_d, size_t head_c,
    size_t head_d, long long row, int ring, int d, int span, float *sums) {
    if constexpr (N > 1)
        if (n <= N / 2)
            return attn_values_for_int8<N / 2>(n, w, vc, vd, row_c, row_d, head_c, head_d, row, ring, d, span, sums);
    attn_values_q8<N>(w, vc, vd, row_c, row_d, head_c, head_d, row, ring, d, span, sums);
}

// attn_value_kernel with the values in --kv int8's form: a head's row is dim codes and dim / kQ8_0Values deltas.
/*
DPCT1110:104: The total declared local variable size in device function
attn_value_int8_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
attn_value_int8_kernel(const float *__restrict__ score, const KvInt8 v,
                       float *__restrict__ out, int n_keys, int dim, int n_head,
                       int kv_heads, long long pos, int first, int tokens,
                       int window, int ring, long long last,
                       const long long *__restrict__ at) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    static_assert(VALUE_DIMS == WARP && VALUE_QUERIES <= WARP,
                  "a warp is a share and reads a query's weight a lane");
    auto &share = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[VALUE_QUERIES][VALUE_SHARES][VALUE_DIMS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int pd = dim / kQ8_0Values;
    const int per = n_head / kv_heads, dims = (dim + VALUE_DIMS - 1) / VALUE_DIMS;
    const int g = item_ct1.get_group(2) / dims,
              q0 = item_ct1.get_group(1) * VALUE_QUERIES;
    const int live = sycl::min(VALUE_QUERIES, tokens * per - q0);
    const int lane = item_ct1.get_local_id(2) % VALUE_DIMS,
              s = item_ct1.get_local_id(2) / VALUE_DIMS;
    const int d = (item_ct1.get_group(2) % dims) * VALUE_DIMS + lane;
    const long long p = (at ? *at : pos) + first;
    long long lo;
    const int span = attn_span(p, tokens, window, last, &lo);
    const int mine = q0 + sycl::min(lane % VALUE_QUERIES, live - 1);
    const size_t row_c = (size_t) kv_heads * dim, row_d = (size_t) kv_heads * pd;
    float sums[VALUE_QUERIES];
    attn_values_for_int8<VALUE_QUERIES>(
        live,
        score + (size_t(mine / per) * n_head + g * per + mine % per) * n_keys,
        v.code, v.delta, row_c, row_d, (size_t)g * dim, (size_t)g * pd,
        ring > 0 ? (lo + s) % ring : lo + s, ring, sycl::min(d, dim - 1), span,
        sums);
    for (int i = 0; i < live; ++i) {
        long long own;
        attn_rows(p + (q0 + i) / per, window, last, &own);
        share[i][(s + VALUE_SHARES - int((own - lo) % VALUE_SHARES)) % VALUE_SHARES][lane] = sums[i];
    }
    /*
    DPCT1065:219: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (s != 0 || d >= dim) return;
    for (int i = 0; i < live; ++i) {
        float sum = share[i][0][lane];
#pragma unroll
        for (int k = 1; k < VALUE_SHARES; ++k) sum += share[i][k][lane];
        const int query = q0 + i;
        out[((size_t(first) + query / per) * n_head + g * per + query % per) * dim + d] = sum;
    }
}

}  // namespace

void attend_int8(const float* q, KvInt8 q8, KvInt8 k, KvInt8 v, int64_t pos, int rows, float* score, int chunk,
               float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window, int ring,
               bool whole, const long long* at, int room) {
    if (at && (whole || room < 1)) throw std::invalid_argument("attend_int8: a position on the card needs its room");
    if (uintptr_t(q) % (SCORE_VALUES * sizeof(float)))
        throw std::invalid_argument("attend_int8: queries are read four values at a time");
    const long long last = whole ? pos + rows - 1 : -1;
    const int per = n_head / kv_heads;
    for (int first = 0; first < rows; first += chunk) {
        const int c = std::min(chunk, rows - first);
        // this chunk's queries in the cache's own form, at the rows the score kernel will look them up at
        store_int8(q + (size_t) first * n_head * dim, const_cast<int8_t*>(q8.code),
                   const_cast<uint16_t*>(q8.delta), n_head, dim, first, c, stream);
        const long long p = pos + first, lo = window > 0 && p >= window ? p - window + 1 : 0;
        const int keys = at ? room : int((whole ? last : p + c - 1) - lo) + 1;
        // One warp is a (key, group) pair, so the grid is the pairs over the warps a block holds - the plain
        // kernel's own sizing.  (The register-blocked fp16 kernel also wrote `grid(units, SCORE_WARPS)` there,
        // but its `units` was already a warp count; copying that line here sized the grid a quarter too small
        // and left every key past the fourth without a score at all.)
        const unsigned threads = SCORE_WARPS * WARP;
        const auto go = [&](auto kernel) {
            dpct::kernel_launcher::launch(
                kernel, grid(size_t(keys) * kv_heads, SCORE_WARPS), threads, 0,
                cs(stream), q8, k, score, keys, n_head, kv_heads,
                (long long)pos, first, c, scale, window, ring, last, at);
        };
        // The head's size is the template: a lane owns dim / WARP of the head's values, the run the dp4a walks.
        // go is the plain kernel, kept as the fallback for a group whose heads do not divide by 4.
        {
            const auto reg = [&](auto kernel) {   // this one's units are a warp count, not a block count
                const unsigned units = grid(size_t(keys), SCORE_KEYS) * kv_heads;
                dpct::kernel_launcher::launch(
                    kernel, grid(units, SCORE_WARPS), threads, 0, cs(stream),
                    q8, k, score, keys, n_head, kv_heads, (long long)pos, first,
                    c, scale, window, ring, last, at);
            };
            // The register-blocked kernel is the default: a warp walks SCORE_KEYS rows while a token's queries (and
            // their codes and deltas) stay in its registers, which on a prompt batch - 8 queries at once rather than
            // a decode's 1 - takes q8 from 1.37x of fp16 to 0.99x at 8k keys and from 1.24x to 0.88x at 32k.
            // (An earlier note here said the plain kernel won; that comparison ran the plain one with a grid a
            // quarter too small, so it was doing a quarter of the work.  Set STRATA_KV_REG=0 to get it back.)
            // The tensor-core path is the default up to 16k keys: the A/B has it 5~7% ahead of the register-blocked
            // one there (the gap is widest on a prompt batch of 8 queries) and level with it past that, where the
            // register-blocked one also uses fewer registers.  STRATA_KV_MMA=1 forces it everywhere, =0 turns it
            // off; unset means pick by size.  A card whose int8 mma is missing should answer probe_mma_layout()
            // instead, once that is wired in.
            static const char* mma_env = std::getenv("STRATA_KV_MMA");
            const int mma_choice = mma_env ? std::atoi(mma_env) : -1;
            // a card without the int8 mma steps (HIP, or a PTX version below 8.0) takes the dp4a tiers instead:
            // STRATA_KV_MMA cannot force it there, since the kernel would not load
            const bool use_mma = int8_tensor_cores() && dim == 256 && mma_choice != 0 &&
                                     (mma_choice > 0 || keys <= 16384);
            const auto mgo = [&](auto kernel) {
                const unsigned warps =
                    grid(size_t(keys), SCORE_KEYS) * grid(size_t(c * per), 16) * kv_heads;   // a warp: 8 keys x 16
                dpct::kernel_launcher::launch(
                    kernel, grid(warps, SCORE_WARPS), threads, 0, cs(stream),
                    q8, k, score, keys, n_head, kv_heads, (long long)pos, first,
                    c, scale, window, ring, last, at);
            };
            static const char* reg_env = std::getenv("STRATA_KV_REG");
            const bool use_reg = !reg_env || std::atoi(reg_env) != 0;
            if (use_mma) mgo(attn_score_mma_int8_kernel_wrapper<256 / WARP>);
            else if (use_reg && dim == 256 && per % 8 == 0)
                reg(attn_score_reg_int8_kernel_wrapper<8, 256 / WARP>);
            else if (use_reg && dim == 256 && per % 4 == 0)
                reg(attn_score_reg_int8_kernel_wrapper<4, 256 / WARP>);
            else if (use_reg && dim == 128 && per % 4 == 0)
                reg(attn_score_reg_int8_kernel_wrapper<4, 128 / WARP>);
            else if (use_reg && dim == 512 && per % 4 == 0)
                reg(attn_score_reg_int8_kernel_wrapper<4, 512 / WARP>);
            else if (dim == 128) go(attn_score_int8_kernel_wrapper<128 / WARP>);
            else if (dim == 256) go(attn_score_int8_kernel_wrapper<256 / WARP>);
            else if (dim == 512) go(attn_score_int8_kernel_wrapper<512 / WARP>);
            // a head narrower than a warp: 8 lanes own it, a lane's run is 8 codes (two dp4a steps, inside one
            // 32-value block), and the warp's 32 lanes cover the group's four queries in one round
            else if (dim == 64) go(attn_score_int8_kernel_wrapper<8, 8>);
            else throw std::invalid_argument("attend_int8: a head is 64, 128, 256 or 512 values");
        }
        // the weights in place: attn_value_int8_kernel reads them from the scores' buffer, so no transposed copy
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<
                    dpct_kernel_name<class attn_softmax_kernel_f1c2d0>>(
                    sycl::nd_range<3>(sycl::range(1, c, n_head) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            attn_softmax_kernel(score, keys, (long long)pos,
                                                first, c, window, last, at,
                                                nullptr, per, nullptr);
                        });
        }
        {
            const int dims = grid(dim, VALUE_DIMS), blocks = grid(c * per, VALUE_QUERIES);
            /*
            DPCT1049:105: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                cs(stream)
                    ->parallel_for<
                        dpct_kernel_name<class attn_value_int8_kernel_894345>>(
                        sycl::nd_range<3>(
                            sycl::range(1, blocks, kv_heads * dims) *
                                sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES),
                            sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                attn_value_int8_kernel(score, v, out, keys, dim,
                                                       n_head, kv_heads,
                                                       (long long)pos, first, c,
                                                       window, ring, last, at);
                            });
            }
        }
    }
}

// attend_int8 with the values' side left in fp16 (--kv int8k): everything is the one above but the weighted
// sums, which read the fp16 cache the way attend() does.  `v_h` is `n_kv * dim` halves a row, the ring's.
void attend_int8_mixed(const float* q, KvInt8 q8, KvInt8 k, const uint16_t* v_h, int64_t pos, int rows, float* score, int chunk,
               float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window, int ring,
               bool whole, const long long* at, int room) {
    if (at && (whole || room < 1)) throw std::invalid_argument("attend_int8: a position on the card needs its room");
    if (uintptr_t(q) % (SCORE_VALUES * sizeof(float)))
        throw std::invalid_argument("attend_int8: queries are read four values at a time");
    const long long last = whole ? pos + rows - 1 : -1;
    const int per = n_head / kv_heads;
    for (int first = 0; first < rows; first += chunk) {
        const int c = std::min(chunk, rows - first);
        // this chunk's queries in the cache's own form, at the rows the score kernel will look them up at
        store_int8(q + (size_t) first * n_head * dim, const_cast<int8_t*>(q8.code),
                   const_cast<uint16_t*>(q8.delta), n_head, dim, first, c, stream);
        const long long p = pos + first, lo = window > 0 && p >= window ? p - window + 1 : 0;
        const int keys = at ? room : int((whole ? last : p + c - 1) - lo) + 1;
        // One warp is a (key, group) pair, so the grid is the pairs over the warps a block holds - the plain
        // kernel's own sizing.  (The register-blocked fp16 kernel also wrote `grid(units, SCORE_WARPS)` there,
        // but its `units` was already a warp count; copying that line here sized the grid a quarter too small
        // and left every key past the fourth without a score at all.)
        const unsigned threads = SCORE_WARPS * WARP;
        const auto go = [&](auto kernel) {
            dpct::kernel_launcher::launch(
                kernel, grid(size_t(keys) * kv_heads, SCORE_WARPS), threads, 0,
                cs(stream), q8, k, score, keys, n_head, kv_heads,
                (long long)pos, first, c, scale, window, ring, last, at);
        };
        // The head's size is the template: a lane owns dim / WARP of the head's values, the run the dp4a walks.
        // go is the plain kernel, kept as the fallback for a group whose heads do not divide by 4.
        {
            const auto reg = [&](auto kernel) {   // this one's units are a warp count, not a block count
                const unsigned units = grid(size_t(keys), SCORE_KEYS) * kv_heads;
                dpct::kernel_launcher::launch(
                    kernel, grid(units, SCORE_WARPS), threads, 0, cs(stream),
                    q8, k, score, keys, n_head, kv_heads, (long long)pos, first,
                    c, scale, window, ring, last, at);
            };
            // The register-blocked kernel is the default: a warp walks SCORE_KEYS rows while a token's queries (and
            // their codes and deltas) stay in its registers, which on a prompt batch - 8 queries at once rather than
            // a decode's 1 - takes q8 from 1.37x of fp16 to 0.99x at 8k keys and from 1.24x to 0.88x at 32k.
            // (An earlier note here said the plain kernel won; that comparison ran the plain one with a grid a
            // quarter too small, so it was doing a quarter of the work.  Set STRATA_KV_REG=0 to get it back.)
            // The tensor-core path is the default up to 16k keys: the A/B has it 5~7% ahead of the register-blocked
            // one there (the gap is widest on a prompt batch of 8 queries) and level with it past that, where the
            // register-blocked one also uses fewer registers.  STRATA_KV_MMA=1 forces it everywhere, =0 turns it
            // off; unset means pick by size.  A card whose int8 mma is missing should answer probe_mma_layout()
            // instead, once that is wired in.
            static const char* mma_env = std::getenv("STRATA_KV_MMA");
            const int mma_choice = mma_env ? std::atoi(mma_env) : -1;
            // a card without the int8 mma steps (HIP, or a PTX version below 8.0) takes the dp4a tiers instead:
            // STRATA_KV_MMA cannot force it there, since the kernel would not load
            const bool use_mma = int8_tensor_cores() && dim == 256 && mma_choice != 0 &&
                                     (mma_choice > 0 || keys <= 16384);
            const auto mgo = [&](auto kernel) {
                const unsigned warps =
                    grid(size_t(keys), SCORE_KEYS) * grid(size_t(c * per), 16) * kv_heads;   // a warp: 8 keys x 16
                dpct::kernel_launcher::launch(
                    kernel, grid(warps, SCORE_WARPS), threads, 0, cs(stream),
                    q8, k, score, keys, n_head, kv_heads, (long long)pos, first,
                    c, scale, window, ring, last, at);
            };
            static const char* reg_env = std::getenv("STRATA_KV_REG");
            const bool use_reg = !reg_env || std::atoi(reg_env) != 0;
            if (use_mma) mgo(attn_score_mma_int8_kernel_wrapper<256 / WARP>);
            else if (use_reg && dim == 256 && per % 8 == 0)
                reg(attn_score_reg_int8_kernel_wrapper<8, 256 / WARP>);
            else if (use_reg && dim == 256 && per % 4 == 0)
                reg(attn_score_reg_int8_kernel_wrapper<4, 256 / WARP>);
            else if (use_reg && dim == 128 && per % 4 == 0)
                reg(attn_score_reg_int8_kernel_wrapper<4, 128 / WARP>);
            else if (use_reg && dim == 512 && per % 4 == 0)
                reg(attn_score_reg_int8_kernel_wrapper<4, 512 / WARP>);
            else if (dim == 128) go(attn_score_int8_kernel_wrapper<128 / WARP>);
            else if (dim == 256) go(attn_score_int8_kernel_wrapper<256 / WARP>);
            else if (dim == 512) go(attn_score_int8_kernel_wrapper<512 / WARP>);
            // a head narrower than a warp: 8 lanes own it, a lane's run is 8 codes (two dp4a steps, inside one
            // 32-value block), and the warp's 32 lanes cover the group's four queries in one round
            else if (dim == 64) go(attn_score_int8_kernel_wrapper<8, 8>);
            else throw std::invalid_argument("attend_int8: a head is 64, 128, 256 or 512 values");
        }
        // the weights in place: attn_value_kernel reads them from the scores' buffer, so no transposed copy
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<
                    dpct_kernel_name<class attn_softmax_kernel_61f2a4>>(
                    sycl::nd_range<3>(sycl::range(1, c, n_head) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            attn_softmax_kernel(score, keys, (long long)pos,
                                                first, c, window, last, at,
                                                nullptr, per, nullptr);
                        });
        }
        {
            const int dims = grid(dim, VALUE_DIMS), blocks = grid(c * per, VALUE_QUERIES);
            /*
            DPCT1049:106: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};
                dpct::has_capability_or_fail(cs(stream)->get_device(),
                                             {sycl::aspect::fp16});

                cs(stream)
                    ->/*
  DPCT1050:372: The template argument of the dpct_kernel_name could not be
  deduced. You need to update this code.
  */
                    parallel_for<dpct_kernel_name<
                        class attn_value_kernel_65b19f>>(
                        sycl::nd_range<3>(
                            sycl::range(1, blocks, kv_heads * dims) *
                                sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES),
                            sycl::range(1, 1, VALUE_DIMS * VALUE_SHARES)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                attn_value_kernel(
                                    score, (const sycl::half *)v_h, out, keys,
                                    dim, n_head, kv_heads, (long long)pos,
                                    first, c, window, ring, last, at, nullptr,
                                    nullptr);
                            });
            }
        }
    }
}

namespace {

__dpct_inline__ void mma_probe_kernel(const int8_t *__restrict__ A,
                                      const int8_t *__restrict__ B,
                                      int *__restrict__ C) {
    const int lane = sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                         .get_local_id(2) %
                     32,
              g = lane >> 2, tig = lane & 3;
    const unsigned a0 = *reinterpret_cast<const unsigned*>(A + g * 32 + tig * 4);
    const unsigned a1 = *reinterpret_cast<const unsigned*>(A + (g + 8) * 32 + tig * 4);
    const unsigned a2 = *reinterpret_cast<const unsigned*>(A + g * 32 + 16 + tig * 4);
    const unsigned a3 = *reinterpret_cast<const unsigned*>(A + (g + 8) * 32 + 16 + tig * 4);
    const unsigned b0 = *reinterpret_cast<const unsigned*>(B + g * 32 + tig * 4);
    const unsigned b1 = *reinterpret_cast<const unsigned*>(B + g * 32 + 16 + tig * 4);
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    C[g * 8 + tig * 2] = c0;
    C[g * 8 + tig * 2 + 1] = c1;
    C[(g + 8) * 8 + tig * 2] = c2;
    C[(g + 8) * 8 + tig * 2 + 1] = c3;
}

}  // namespace

bool int8_tensor_cores() { return kernels_ptx() >= 80; }

bool probe_mma_layout() {
    int8_t A[16 * 32], B[8 * 32];
    for (int i = 0; i < 16 * 32; ++i) A[i] = (int8_t) ((i % 17) - 8);
    for (int i = 0; i < 8 * 32; ++i) B[i] = (int8_t) ((i % 13) - 6);
    int want[16 * 8], got[16 * 8];
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
            int s = 0;
            for (int k = 0; k < 32; ++k) s += (int) A[m * 32 + k] * (int) B[n * 32 + k];
            want[m * 8 + n] = s;
        }
    int8_t *dA = nullptr, *dB = nullptr;
    int* dC = nullptr;
    dA = (int8_t *)sycl::malloc_device(sizeof A, dpct::get_in_order_queue());
    dB = (int8_t *)sycl::malloc_device(sizeof B, dpct::get_in_order_queue());
    dC = (int *)sycl::malloc_device(sizeof got, dpct::get_in_order_queue());
    /*
    DPCT1114:220: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    dpct::get_in_order_queue().memcpy(dA, A, sizeof A).wait();
    /*
    DPCT1114:221: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    dpct::get_in_order_queue().memcpy(dB, B, sizeof B).wait();
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        dpct::get_in_order_queue()
            .parallel_for<dpct_kernel_name<class mma_probe_kernel_7ba9dd>>(
                sycl::nd_range<3>(sycl::range(1, 1, 32), sycl::range(1, 1, 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mma_probe_kernel(dA, dB, dC);
                    });
    }
    dpct::get_in_order_queue()
        .memcpy(got, dC, sizeof got)
        .wait(); // an implicit sync
    sycl::free(dA, dpct::get_in_order_queue());
    sycl::free(dB, dpct::get_in_order_queue());
    sycl::free(dC, dpct::get_in_order_queue());
    int bad = 0;
    for (int i = 0; i < 128; ++i)
        if (got[i] != want[i]) ++bad;
    std::fprintf(stderr, "[mma probe] %d of 128 wrong\n", bad);
    return bad == 0;
}

// The same 16x8x256 product twice over: eight m16n8k32 on the tensor cores, against the 256 dp4a it takes to do
// the same number of multiply-accumulates on the CUDA cores.
__dpct_inline__ void mma_speed_kernel(const int8_t *__restrict__ A,
                                      const int8_t *__restrict__ B,
                                      int *__restrict__ C, int iters) {
    const int lane = sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                         .get_local_id(2) %
                     32,
              g = lane >> 2, tig = lane & 3;
    const unsigned a0 = *reinterpret_cast<const unsigned*>(A + g * 32 + tig * 4);
    const unsigned a1 = *reinterpret_cast<const unsigned*>(A + (g + 8) * 32 + tig * 4);
    const unsigned a2 = *reinterpret_cast<const unsigned*>(A + g * 32 + 16 + tig * 4);
    const unsigned a3 = *reinterpret_cast<const unsigned*>(A + (g + 8) * 32 + 16 + tig * 4);
    const unsigned b0 = *reinterpret_cast<const unsigned*>(B + g * 32 + tig * 4);
    const unsigned b1 = *reinterpret_cast<const unsigned*>(B + g * 32 + 16 + tig * 4);
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    }
    C[g * 8 + tig * 2] = c0 + c1 + c2 + c3;
}

__dpct_inline__ void dp4a_speed_kernel(const int8_t *__restrict__ A,
                                       const int8_t *__restrict__ B,
                                       int *__restrict__ C, int iters) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) %
        32;
    const int xa = *reinterpret_cast<const int*>(A + (lane % 16) * 32 + (lane % 8) * 4);
    const int xb = *reinterpret_cast<const int*>(B + (lane % 8) * 32 + (lane % 8) * 4);
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0;   // four chains, so the result is not one long dependency
#pragma unroll
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            a0 = STRATA_DP4A(xa, xb, a0);
            a1 = STRATA_DP4A(xa, xb, a1);
            a2 = STRATA_DP4A(xa, xb, a2);
            a3 = STRATA_DP4A(xa, xb, a3);
        }
    }
    C[lane] = (a0 + a1) + (a2 + a3);
}

void probe_mma_speed(double* mma_ms, double* dp4a_ms) {
    int8_t A[16 * 32], B[8 * 32];
    for (int i = 0; i < 16 * 32; ++i) A[i] = (int8_t) ((i % 17) - 8);
    for (int i = 0; i < 8 * 32; ++i) B[i] = (int8_t) ((i % 13) - 6);
    int8_t *dA = nullptr, *dB = nullptr;
    int* dC = nullptr;
    dA = (int8_t *)sycl::malloc_device(sizeof A, dpct::get_in_order_queue());
    dB = (int8_t *)sycl::malloc_device(sizeof B, dpct::get_in_order_queue());
    dC = sycl::malloc_device<int>(32, dpct::get_in_order_queue());
    /*
    DPCT1114:222: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    dpct::get_in_order_queue().memcpy(dA, A, sizeof A).wait();
    /*
    DPCT1114:223: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    dpct::get_in_order_queue().memcpy(dB, B, sizeof B).wait();
    const int iters = 20000, blocks = 384;   // 384 warps: enough to fill the card
    dpct::event_ptr e0, e1;
    e0 = new sycl::event();
    e1 = new sycl::event();
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        dpct::get_in_order_queue()
            .parallel_for<dpct_kernel_name<class mma_speed_kernel_fa9878>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, 32),
                                  sycl::range(1, 1, 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mma_speed_kernel(dA, dB, dC, 10);
                    });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        dpct::get_in_order_queue()
            .parallel_for<dpct_kernel_name<class dp4a_speed_kernel_e84d21>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, 32),
                                  sycl::range(1, 1, 32)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    dp4a_speed_kernel(dA, dB, dC, 10);
                });
    }
    float tm = 0.0f, td = 0.0f;
    dpct::sync_barrier(e0, &dpct::get_in_order_queue());
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        dpct::get_in_order_queue()
            .parallel_for<dpct_kernel_name<class mma_speed_kernel_1ea409>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, 32),
                                  sycl::range(1, 1, 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        mma_speed_kernel(dA, dB, dC, iters);
                    });
    }
    dpct::sync_barrier(e1, &dpct::get_in_order_queue());
    e1->wait_and_throw();
    tm =
        (e1->get_profiling_info<sycl::info::event_profiling::command_end>() -
         e0->get_profiling_info<sycl::info::event_profiling::command_start>()) /
        1000000.0f;
    dpct::sync_barrier(e0, &dpct::get_in_order_queue());
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        dpct::get_in_order_queue()
            .parallel_for<dpct_kernel_name<class dp4a_speed_kernel_191ccf>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, 32),
                                  sycl::range(1, 1, 32)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    dp4a_speed_kernel(dA, dB, dC, iters);
                });
    }
    dpct::sync_barrier(e1, &dpct::get_in_order_queue());
    e1->wait_and_throw();
    td =
        (e1->get_profiling_info<sycl::info::event_profiling::command_end>() -
         e0->get_profiling_info<sycl::info::event_profiling::command_start>()) /
        1000000.0f;
    *mma_ms = tm;
    *dp4a_ms = td;
    dpct::destroy_event(e0);
    dpct::destroy_event(e1);
    sycl::free(dA, dpct::get_in_order_queue());
    sycl::free(dB, dpct::get_in_order_queue());
    sycl::free(dC, dpct::get_in_order_queue());
}

void attend_batch(void *blas, const float *q, const uint16_t *kcache,
                  const uint16_t *vcache, int64_t pos, int rows, float *score,
                  size_t score_floats, float *work, float *out, int dim,
                  int n_head, int kv_heads, float scale, void *stream,
                  int window, int ring, bool whole, Kvq8 kq8, Kvq8 vq8,
                  uint16_t *kwork, uint16_t *vwork, size_t work_halves,
                  const float *sinks) try {
    const int per = n_head / kv_heads, width = kv_heads * dim;
    // The fp16-only dim==64 fast path reads the fp16 caches, which are null under q8 (a q8 cache is load-restricted
    // to qwen35moe's head_dim 128, so it never reaches here); guard explicitly so it is provably not taken for q8.
    if (dim == 64 && per == 4 && window == 0 && ring == 0 && !whole && !sinks && kq8.codes == nullptr &&
        vq8.codes == nullptr &&
        tensor_cores()) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(cs(stream)->get_device(),
                                         {sycl::aspect::fp16});

            cs(stream)->submit([&](sycl::handler &cgh) {
                auto scale_ct8 = scale * 1.44269504f;

                cgh.parallel_for<
                    dpct_kernel_name<class attn_flash64_kernel_4f06b0>>(
                    sycl::nd_range<3>(
                        sycl::range(1, kv_heads, grid(rows, FLASH_TOKENS)) *
                            sycl::range(1, 1, 4 * WARP),
                        sycl::range(1, 1, 4 * WARP)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            attn_flash64_kernel(q, (const sycl::half *)kcache,
                                                (const sycl::half *)vcache, out,
                                                rows, n_head, kv_heads,
                                                (long long)pos, scale_ct8);
                        });
            });
        }
        return;
    }
    if (dim == 128 && per == 8 && !whole && kq8.codes == nullptr && vq8.codes == nullptr && kernels_ptx() >= 80) {
        // the device's own setting: asked on whichever card the call is for
        const dpct::err0 e =
            /*
            DPCT1027:226: The call to cudaFuncSetAttribute was replaced with 0
            because SYCL currently does not support corresponding setting.
            */
            0;
        /*
        DPCT1000:225: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (e != 0)
            /*
            DPCT1009:227: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1001:224: The statement could not be removed.
            */
            throw std::runtime_error(
                std::string("attend_batch: shared memory: ") +
                dpct::get_error_string_dummy(e));
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(cs(stream)->get_device(),
                                         {sycl::aspect::fp16});

            cs(stream)->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(FLASH128_SMEM), cgh);

                auto scale_ct8 = scale * 1.44269504f;

                cgh.parallel_for<
                    dpct_kernel_name<class attn_flash128_kernel_9b140c>>(
                    sycl::nd_range<3>(
                        sycl::range(1, kv_heads, grid(rows, FLASH128_TOKENS)) *
                            sycl::range(1, 1, 4 * WARP),
                        sycl::range(1, 1, 4 * WARP)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        attn_flash128_kernel(
                            q, (const sycl::half *)kcache,
                            (const sycl::half *)vcache, out, rows, n_head,
                            kv_heads, (long long)pos, scale_ct8, window, ring,
                            sinks,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        }
        return;
    }
    const dpct::blas::descriptor_ptr h = (dpct::blas::descriptor_ptr)blas;
    const float one = 1.0f, zero = 0.0f;
    const long long last = whole ? pos + rows - 1 : -1;
    const bool kq = kq8.codes != nullptr, vq = vq8.codes != nullptr;
    // the rows a q8 workspace holds: a block of keys larger than this is split into more products
    const int work_rows = width > 0 ? (int) std::min<size_t>(work_halves / (size_t) width, (size_t) INT_MAX) : 0;
    if (kq && (kwork == nullptr || work_rows < 1))
        throw std::invalid_argument("attend_batch: a q8 key cache needs a row of its workspace");
    if (vq && (vwork == nullptr || work_rows < 1))
        throw std::invalid_argument("attend_batch: a q8 value cache needs a row of its workspace");
    // The products take halves and sum floats, on the tensor cores: the cache's rows as they lie there, the
    // queries and the softmax's weights rounded.  Two thirds of the scratch are for a block's scores, the rest
    // for their weights.
#if defined(STRATA_USE_HIP)
    // STRATA_HIP_PREFILL_FP32=1: the products on FP32 inputs (see attn_group_f32_kernel).  Needs a q8 cache whose
    // workspace holds a block of FP32 rows, and scratch for the grouped queries; the scores and the weights then
    // take half of what is left each.
    // =2: our own kernels (hip_prefill_gemm.cuh) read the q8 cache as it lies, with no FP32 copy of it and no workspace.
    const char* fp32_env = std::getenv("STRATA_HIP_PREFILL_FP32");
    const int level = fp32_env && !fp32_env[1] && (fp32_env[0] == '1' || fp32_env[0] == '2') ? fp32_env[0] - '0' : 0;
    const size_t qres = std::max<size_t>(kAttendResults / sizeof(float), size_t(16) * n_head * dim);
    const int work_rows_f = width > 0 ? (int) std::min<size_t>(work_halves / 2 / (size_t) width, (size_t) INT_MAX) : 0;
    const size_t chunk_max = std::max<size_t>(16, kAttendResults / (size_t(n_head) * dim * sizeof(float)));
    const size_t cells_f = ((score_floats > qres ? score_floats - qres : 0) / 2) & ~size_t(3);   // a multiple of 4: the floats are read 4 at a time
    // level 2: the head's values in whole blocks of 32 for the scales, and a block of at least 64 keys at the largest chunk
    const bool own = level == 2 && kq && vq && dim % 32 == 0 && width % 32 == 0 && score_floats >= 4 * qres &&
                     cells_f / n_head >= 64 * chunk_max;
    const bool f32 = own || (level >= 1 && kq && vq && work_rows_f >= 256 && score_floats >= 4 * qres);
    const size_t cells = f32 ? cells_f : score_floats / 3 * 2;
    float* weight32 = score + cells;
    float* qg32 = score + 2 * cells;   // the grouped queries as floats, in the scratch past the scores and weights
#else
    const bool own = false;
    const bool f32 = false;
    const int work_rows_f = 0;
    const size_t cells = score_floats / 3 * 2;
#endif
    sycl::half *weight = (sycl::half *)(score + cells);
    for (int first = 0, c = 0; first < rows; first += c) {
        // every token left, unless a grid's 65535 rows or the scratch for one row's scores is less, or the results
        // would outgrow kAttendResults: every block of rows reads and writes them all again (og)
        c = (int) std::min({size_t(rows - first), size_t(65535 / n_head), cells / n_head,
                            std::max<size_t>(16, kAttendResults / (size_t(n_head) * dim * sizeof(float)))});
        const long long p0 = pos + first;   // the chunk's first position, and the first row its tokens read
        const long long lo0 = window > 0 && p0 >= window ? p0 - window + 1 : 0;
        const int n_keys = int((whole ? last : p0 + c - 1) - lo0) + 1;
        const size_t heads = size_t(c) * n_head;
        sycl::half *qg = (sycl::half *)work;          // the queries by group
        float* og = work + (heads * dim + 1) / 2;     // the results by group
        float* top = og + heads * dim;       // a token and head's largest score so far, the sum of its weights
        float* sum = top + heads;            // against it, and what a block of rows makes the sums before it worth
        float* shrink = sum + heads;
#if defined(STRATA_USE_HIP)
        if (f32)
            attn_group_f32_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(
                q + size_t(first) * n_head * dim, qg32, dim, n_head, kv_heads, c);
        else
#endif
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(cs(stream)->get_device(),
                                         {sycl::aspect::fp16});

            cs(stream)->submit([&](sycl::handler &cgh) {
                auto q_size_t_first_n_head_dim_ct0 =
                    q + size_t(first) * n_head * dim;

                cgh.parallel_for<
                    dpct_kernel_name<class attn_group_kernel_c19ac8>>(
                    sycl::nd_range<3>(
                        sycl::range(1, c * n_head, grid(dim, 256)) *
                            sycl::range(1, 1, 256),
                        sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        attn_group_kernel(q_size_t_first_n_head_dim_ct0, qg,
                                          dim, n_head, kv_heads, c);
                    });
            });
        }
        // The rows go a block at a time, as many as have their scores in the scratch, so that a product has the
        // chunk's every token for its columns however long the context is: cuBLAS is several times slower on a
        // few columns.  A token's softmax is kept from block to block (top, sum), and its values so far shrink
        // to the new top.
        const long long cols = (long long) c * per;
        const int fit = (int) (cells / heads);
        const int block = fit > 16 ? fit & ~15 : fit;   // the tensor cores are half as fast on a ragged block
        for (int k0 = 0, m = 0; k0 < n_keys; k0 += m) {
            // the block's rows where the cache has them: a block ends where a ring starts over
            const long long row = ring > 0 ? (lo0 + k0) % ring : lo0 + k0;
            m = (int) std::min<long long>(std::min(block, n_keys - k0), ring > 0 ? ring - row : n_keys);
            if (!own && (kq || vq) && m > (f32 ? work_rows_f : work_rows)) m = f32 ? work_rows_f : work_rows;   // the workspace's rows cap the block
#if defined(STRATA_USE_HIP)
            // a product whose leading dimension is a multiple of 1024 runs the values' FP32 product at half its speed
            if (own) {
                // our kernels: scores S[n][m] and weights W[n][m] a column apart by mp floats (a multiple of 4), read the q8 cache's rows
                const int mp = (m + 3) & ~3;
                hip_prefill_gemm::Args as{};
                as.codes = kq8.codes; as.scales = kq8.scales; as.row0 = row; as.width = width; as.dim = dim;
                as.B = qg32; as.b_stride = cols * dim; as.ldb = dim; as.C = score; as.c_stride = cols * mp; as.ldc = mp;
                as.M = m; as.N = (int) cols; as.K = dim; as.alpha = scale; as.accumulate = 0;
                hip_prefill_gemm::gemm_q8<true><<<dim3(((int) cols + 127) / 128, (m + 127) / 128, kv_heads), 256, 0,
                                                  cs(stream)>>>(as);
                attn_softmax_block_f32_kernel<<<c * n_head, 128, 0, cs(stream)>>>(
                    score, weight32, m, mp, lo0 + k0, (long long) pos, first, window, c, per, last, top, sum, shrink,
                    k0 == 0);
                if (k0 > 0)
                    attn_scale_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(og, shrink, dim, false);
                hip_prefill_gemm::Args av{};
                av.codes = vq8.codes; av.scales = vq8.scales; av.row0 = row; av.width = width; av.dim = dim;
                av.B = weight32; av.b_stride = cols * mp; av.ldb = mp; av.C = og; av.c_stride = cols * dim; av.ldc = dim;
                av.M = dim; av.N = (int) cols; av.K = m; av.alpha = 1.0f; av.accumulate = k0 > 0 ? 1 : 0;
                hip_prefill_gemm::gemm_q8<false><<<dim3(((int) cols + 127) / 128, (dim + 127) / 128, kv_heads), 256, 0,
                                                   cs(stream)>>>(av);
                continue;
            }
            // a product whose leading dimension is a multiple of 1024 runs the values' FP32 product at half its speed
            if (f32 && m > 16 && m % 1024 == 0) m -= 16;
            if (f32) {
                dequant_q8_f32_kernel<<<dim3(grid(width, 256), m), 256, 0, cs(stream)>>>(kq8.codes, kq8.scales,
                                                                                       (float*) kwork, width, row);
                dequant_q8_f32_kernel<<<dim3(grid(width, 256), m), 256, 0, cs(stream)>>>(vq8.codes, vq8.scales,
                                                                                       (float*) vwork, width, row);
                blas_ck(cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, m, (int) cols, dim, &scale,
                                                  (const float*) kwork, width, dim, qg32, dim, cols * dim, &zero, score,
                                                  m, cols * m, kv_heads),
                        "attention scores");
                attn_softmax_block_f32_kernel<<<c * n_head, 128, 0, cs(stream)>>>(
                    score, weight32, m, m, lo0 + k0, (long long) pos, first, window, c, per, last, top, sum, shrink,
                    k0 == 0);
                if (k0 > 0)
                    attn_scale_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(og, shrink, dim, false);
                blas_ck(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, dim, (int) cols, m, &one,
                                                  (const float*) vwork, width, dim, weight32, m, cols * m,
                                                  k0 > 0 ? &one : &zero, og, dim, cols * dim, kv_heads),
                        "attention values");
                continue;
            }
#endif
            // a q8 cache's rows are dequantized into the workspace the products read as an fp16 cache
            if (kq)
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};
                dpct::has_capability_or_fail(cs(stream)->get_device(),
                                             {sycl::aspect::fp16});

                cs(stream)
                    ->parallel_for<
                        dpct_kernel_name<class dequant_q8_kernel_93495d>>(
                        sycl::nd_range<3>(sycl::range(1, m, grid(width, 256)) *
                                              sycl::range(1, 1, 256),
                                          sycl::range(1, 1, 256)),
                        exp_props, [=](sycl::nd_item<3> item_ct1) {
                            dequant_q8_kernel(kq8.codes, kq8.scales,
                                              (sycl::half *)kwork, width, row);
                        });
            }
            if (vq)
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};
                dpct::has_capability_or_fail(cs(stream)->get_device(),
                                             {sycl::aspect::fp16});

                cs(stream)
                    ->parallel_for<
                        dpct_kernel_name<class dequant_q8_kernel_2ebf46>>(
                        sycl::nd_range<3>(sycl::range(1, m, grid(width, 256)) *
                                              sycl::range(1, 1, 256),
                                          sycl::range(1, 1, 256)),
                        exp_props, [=](sycl::nd_item<3> item_ct1) {
                            dequant_q8_kernel(vq8.codes, vq8.scales,
                                              (sycl::half *)vwork, width, row);
                        });
            }
            const sycl::half *krow =
                kq ? (const sycl::half *)kwork
                   : (const sycl::half *)kcache + size_t(row) * width;
            const sycl::half *vrow =
                vq ? (const sycl::half *)vwork
                   : (const sycl::half *)vcache + size_t(row) * width;
            // A product a group: its heads' queries of every token of the chunk against the group's keys.  cuBLAS
            // reads the cache's rows as the transpose, and the groups' keys lie `dim` apart in a row.
            blas_ck(
                DPCT_CHECK_ERROR(dpct::blas::gemm_batch(
                    h, oneapi::mkl::transpose::trans,
                    oneapi::mkl::transpose::nontrans, m, (int)cols, dim, &scale,
                    krow, dpct::library_data_t::real_half, width, dim, qg,
                    dpct::library_data_t::real_half, dim, cols * dim, &zero,
                    score, dpct::library_data_t::real_float, m, cols * m,
                    kv_heads, dpct::compute_type::f32)),
                "attention scores");
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};
                dpct::has_capability_or_fail(cs(stream)->get_device(),
                                             {sycl::aspect::fp16});

                cs(stream)->submit([&](sycl::handler &cgh) {
                    auto lo0_k0_ct3 = lo0 + k0;
                    auto k0_ct13 = k0 == 0;

                    cgh.parallel_for<dpct_kernel_name<
                        class attn_softmax_block_kernel_adf349>>(
                        sycl::nd_range<3>(sycl::range(1, 1, c * n_head) *
                                              sycl::range(1, 1, 128),
                                          sycl::range(1, 1, 128)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                attn_softmax_block_kernel(
                                    score, weight, m, lo0_k0_ct3,
                                    (long long)pos, first, window, c, per, last,
                                    top, sum, shrink, k0_ct13);
                            });
                });
            }
            if (k0 > 0)
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                cs(stream)
                    ->parallel_for<
                        dpct_kernel_name<class attn_scale_kernel_87dcc8>>(
                        sycl::nd_range<3>(
                            sycl::range(1, c * n_head, grid(dim, 256)) *
                                sycl::range(1, 1, 256),
                            sycl::range(1, 1, 256)),
                        exp_props, [=](sycl::nd_item<3> item_ct1) {
                            attn_scale_kernel(og, shrink, dim, false);
                        });
            }
            blas_ck(
                DPCT_CHECK_ERROR(dpct::blas::gemm_batch(
                    h, oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::transpose::nontrans, dim, (int)cols, m, &one,
                    vrow, dpct::library_data_t::real_half, width, dim, weight,
                    dpct::library_data_t::real_half, m, cols * m,
                    k0 > 0 ? &one : &zero, og, dpct::library_data_t::real_float,
                    dim, cols * dim, kv_heads, dpct::compute_type::f32)),
                "attention values");
        }
        if (sinks)
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)->submit([&](sycl::handler &cgh) {
                auto c_n_head_ct3 = c * n_head;

                cgh.parallel_for<
                    dpct_kernel_name<class attn_sink_sum_kernel_fa082e>>(
                    sycl::nd_range<3>(sycl::range(1, 1, grid(c * n_head, 256)) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        attn_sink_sum_kernel(sum, top, sinks, c_n_head_ct3, per,
                                             int(cols));
                    });
            });
        }
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<
                    dpct_kernel_name<class attn_scale_kernel_fc4685>>(
                    sycl::nd_range<3>(
                        sycl::range(1, c * n_head, grid(dim, 256)) *
                            sycl::range(1, 1, 256),
                        sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        attn_scale_kernel(og, sum, dim, true);
                    });
        }
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)->submit([&](sycl::handler &cgh) {
                auto out_size_t_first_n_head_dim_ct1 =
                    out + size_t(first) * n_head * dim;

                cgh.parallel_for<
                    dpct_kernel_name<class attn_ungroup_kernel_40700a>>(
                    sycl::nd_range<3>(
                        sycl::range(1, c * n_head, grid(dim, 256)) *
                            sycl::range(1, 1, 256),
                        sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        attn_ungroup_kernel(og, out_size_t_first_n_head_dim_ct1,
                                            dim, n_head, kv_heads, c);
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void softcap(float* x, size_t n, float cap, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->parallel_for<dpct_kernel_name<class softcap_kernel_adf733>>(
        sycl::nd_range<3>(sycl::range(1, 1, grid(n, 256)) *
                              sycl::range(1, 1, 256),
                          sycl::range(1, 1, 256)),
        exp_props, [=](sycl::nd_item<3> item_ct1) {
            softcap_kernel(x, n, cap);
        });
}

void likeliest(const float* logits, int n, Likeliest* out, void* stream) {
    /*
    DPCT1049:108: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    cs(stream)->parallel_for<dpct_kernel_name<class likeliest_kernel_f8604c>>(
        sycl::nd_range<3>(sycl::range(1, 1, 1024), sycl::range(1, 1, 1024)),
        exp_props,
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            likeliest_kernel(logits, n, out);
        });
}

void mul_sigmoid(float* acc, const float* g, int n, int rows, int g_stride, bool one_gate, void* stream) {
    for (int r = 0; r < rows; r += 65535)   // a grid has at most 65535 rows, and 4,096 tokens have 16 heads each
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->submit([&](sycl::handler &cgh) {
            auto acc_size_t_r_n_ct0 = acc + size_t(r) * n;
            auto g_size_t_r_g_stride_ct1 = g + size_t(r) * g_stride;

            cgh.parallel_for<dpct_kernel_name<class mul_sigmoid_kernel_17c63b>>(
                sycl::nd_range<3>(
                    sycl::range(1, std::min(65535, rows - r), grid(n, 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    mul_sigmoid_kernel(acc_size_t_r_n_ct0,
                                       g_size_t_r_g_stride_ct1, n, g_stride,
                                       one_gate);
                });
        });
    }
}

void conv4_silu(const float* x, float* state, const float* w, float* out, int channels, int rows, void* stream,
                float* snap, size_t stride, unsigned snaps) {
    if (rows <= 0) return;
    if (rows >= CONV_ROWS && (!snap || !snaps)) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<
                    dpct_kernel_name<class conv4_silu_rows_kernel_7bcbd9>>(
                    sycl::nd_range<3>(
                        sycl::range(1, rows, grid(channels, 256)) *
                            sycl::range(1, 1, 256),
                        sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        conv4_silu_rows_kernel(x, state, w, out, channels);
                    });
        }
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<
                    dpct_kernel_name<class conv4_state_kernel_f14bf1>>(
                    sycl::nd_range<3>(sycl::range(1, 1, grid(channels, 256)) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        conv4_state_kernel(x, state, channels, rows);
                    });
        }
        return;
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class conv4_silu_kernel_dc8f2f>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(channels, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    conv4_silu_kernel(x, state, w, out, channels, rows, snap,
                                      stride, snaps);
                });
    }
}

void shortconv(const float* bcx, float* state, const float* w, float* out, int channels, int rows, void* stream,
               float* snap, size_t stride, unsigned snaps) {
    if (rows <= 0) return;
    if (rows >= CONV_ROWS && (!snap || !snaps)) {
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<
                    dpct_kernel_name<class shortconv_rows_kernel_8f543a>>(
                    sycl::nd_range<3>(
                        sycl::range(1, rows, grid(channels, 256)) *
                            sycl::range(1, 1, 256),
                        sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        shortconv_rows_kernel(bcx, state, w, out, channels);
                    });
        }
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            cs(stream)
                ->parallel_for<
                    dpct_kernel_name<class shortconv_state_kernel_fbb923>>(
                    sycl::nd_range<3>(sycl::range(1, 1, grid(channels, 256)) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        shortconv_state_kernel(bcx, state, channels, rows);
                    });
        }
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class shortconv_kernel_894674>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(channels, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    shortconv_kernel(bcx, state, w, out, channels, rows, snap,
                                     stride, snaps);
                });
    }
    /*
    DPCT1010:230: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    /*
    DPCT1000:229: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (const dpct::err0 e = 0; e != 0)
        /*
        DPCT1009:231: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001:228: The statement could not be removed.
        */
        throw std::runtime_error(std::string("shortconv: ") +
                                 dpct::get_error_string_dummy(e));
}

#ifdef STRATA_POLY_GDN_MMA
bool gdn_mma_step(float* qkv, float* alpha, float* beta, const float* dt, const float* a, float* s,
                  const float* z, const float* o_norm, float* out, int dim, int n_head, int k_heads, int rows,
                  float norm_eps, void* stream, bool snapshots) {
    if (!kernels::gdn_mma_shape(dim, n_head, k_heads, rows, snapshots)) return false;
    const size_t keys = size_t(k_heads) * dim, row = (2 * size_t(k_heads) + n_head) * dim;
    kernels::GdnMmaArgs args{};
    args.q = qkv;
    args.k = qkv + keys;
    args.v = qkv + 2 * keys;
    args.log_decay = alpha;
    args.beta = beta;
    args.state = args.state_out = s;
    args.output = out;
    args.dim = dim;
    args.heads = n_head;
    args.key_heads = k_heads;
    args.rows = rows;
    args.qk_row_stride = args.v_row_stride = row;
    args.gate_row_stride = n_head;
    // Poly's normalization already includes 1/sqrt(dim) in Q.
    args.query_scale = 1.f;
    if (!kernels::gdn_mma_supported(args) || !kernels::gdn_mma_available(dim, n_head, k_heads)) return false;
    gdn_norm_kernel<<<rows * k_heads, dim, 0, cs(stream)>>>(qkv, dim, k_heads, row, norm_eps);
    gdn_log_gates_kernel<<<grid(rows * n_head, 256), 256, 0, cs(stream)>>>(alpha, beta, dt, a, n_head, rows * n_head);
    // Preparation mutates Q/K and gates. Never run stock on these modified inputs
    // if device availability unexpectedly changes after the preflight above.
    if (!kernels::gdn_mma_try(args, stream))
        throw std::runtime_error("GDN MMA: prepared execution became unavailable");
    gdn_out_kernel<<<rows * n_head, dim, 0, cs(stream)>>>(out, z, o_norm, dim, norm_eps);
    return true;
}
#endif

void gdn_step(float* qkv, float* alpha, float* beta, const float* dt, const float* a, float* s, const float* z,
              const float* o_norm, float* out, int dim, int n_head, int k_heads, int rows, float norm_eps, void* stream,
              float* snap, size_t stride, unsigned snaps) {
    if (dim % WARP != 0 || dim > 256) throw std::invalid_argument("gdn_step: a head of 32 to 256 values, in 32s");
    const size_t row_n = (2 * size_t(k_heads) + n_head) * dim;
    const int gates = rows * n_head, columns = std::min(dim, 1024 / WARP);   // a block's columns: a warp each
    /*
    DPCT1049:109: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class gdn_norm_kernel_a81011>>(
                sycl::nd_range<3>(sycl::range(1, 1, rows * k_heads) *
                                      sycl::range(1, 1, dim),
                                  sycl::range(1, 1, dim)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_norm_kernel(qkv, dim, k_heads, row_n, norm_eps);
                    });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class gdn_gates_kernel_ded1c0>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(gates, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    gdn_gates_kernel(alpha, beta, dt, a, n_head, gates);
                });
    }
    const int blocks = n_head * ((dim + columns - 1) / columns), threads = columns * WARP;
    const unsigned sn = snap ? snaps : 0u;
    switch (dim / WARP) {
    /*
    DPCT1049:111: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    case 1: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class gdn_state_reg_kernel_8c9cbc,
                                            dpct_kernel_scalar<1>>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_state_reg_kernel<1>(qkv, alpha, beta, s, out, dim,
                                                n_head, k_heads, rows, snap,
                                                stride, sn);
                    });
    } break;
    /*
    DPCT1049:112: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    case 2: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class gdn_state_reg_kernel_d8f804,
                                            dpct_kernel_scalar<2>>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_state_reg_kernel<2>(qkv, alpha, beta, s, out, dim,
                                                n_head, k_heads, rows, snap,
                                                stride, sn);
                    });
    } break;
    /*
    DPCT1049:113: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    case 4: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class gdn_state_reg_kernel_b2ab81,
                                            dpct_kernel_scalar<4>>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_state_reg_kernel<4>(qkv, alpha, beta, s, out, dim,
                                                n_head, k_heads, rows, snap,
                                                stride, sn);
                    });
    } break;
    /*
    DPCT1049:114: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    case 8: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class gdn_state_reg_kernel_b61d30,
                                            dpct_kernel_scalar<8>>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_state_reg_kernel<8>(qkv, alpha, beta, s, out, dim,
                                                n_head, k_heads, rows, snap,
                                                stride, sn);
                    });
    } break;
    /*
    DPCT1049:115: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    default: {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class gdn_state_kernel_c76455>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_state_kernel(qkv, alpha, beta, s, out, dim, n_head,
                                         k_heads, rows, snap, stride, sn);
                    });
    }
    }
    /*
    DPCT1049:110: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->parallel_for<dpct_kernel_name<class gdn_out_kernel_c3449d>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows * n_head) *
                                  sycl::range(1, 1, dim),
                              sycl::range(1, 1, dim)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_out_kernel(out, z, o_norm, dim, norm_eps);
            });
    }
}

void conv4_block_silu(const float* x, const float* state, const float* w, const float* right, float* out,
                      int channels, int rows, void* stream) {
    if (rows <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<
                dpct_kernel_name<class conv4_block_silu_kernel_8a75a8>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(channels, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    conv4_block_silu_kernel(x, state, w, right, out, channels,
                                            rows);
                });
    }
}

void gdn_block(float* qkv, float* alpha, float* beta, const float* dt, const float* a, const float* s,
               const float* z, const float* o_norm, const float* logit_scale, const float* hist_b,
               const float* hist_c, float* out, float* w, int dim, int n_head, int k_heads, int rows,
               float norm_eps, void* stream) {
    if (dim % WARP != 0 || dim > 256) throw std::invalid_argument("gdn_block: a head of 32 to 256 values, in 32s");
    /*
    DPCT1083:117: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const size_t shared =
        (2 * size_t(rows) * dim + size_t(rows) * (rows + 1)) * sizeof(float);
    if (rows < 1 || rows > 32 || shared > 48 * 1024)
        throw std::invalid_argument("gdn_block: 1 to 32 tokens whose queries and keys fit 48 KiB");
    const size_t row_n = (2 * size_t(k_heads) + n_head) * dim;
    const int gates = rows * n_head, columns = std::min(dim, 1024 / WARP);
    /*
    DPCT1049:116: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class gdn_norm_kernel_cec53f>>(
                sycl::nd_range<3>(sycl::range(1, 1, rows * k_heads) *
                                      sycl::range(1, 1, dim),
                                  sycl::range(1, 1, dim)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_norm_kernel(qkv, dim, k_heads, row_n, norm_eps);
                    });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class gdn_gates_kernel_b063c4>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid(gates, 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    gdn_gates_kernel(alpha, beta, dt, a, n_head, gates);
                });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(shared), cgh);

            cgh.parallel_for<
                dpct_kernel_name<class gdn_block_weights_kernel_c0b17b>>(
                sycl::nd_range<3>(sycl::range(1, 1, n_head) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    gdn_block_weights_kernel(
                        qkv, logit_scale, hist_b, hist_c, w, dim, n_head,
                        k_heads, rows,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
    const int blocks = n_head * ((dim + columns - 1) / columns), threads = columns * WARP;
    /*
    DPCT1049:118: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)
            ->parallel_for<dpct_kernel_name<class gdn_block_out_kernel_3d5cbc>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_block_out_kernel(qkv, alpha, s, w, out, dim, n_head,
                                             k_heads, rows);
                    });
    }
    /*
    DPCT1049:119: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->parallel_for<dpct_kernel_name<class gdn_out_kernel_8cc9c4>>(
            sycl::nd_range<3>(sycl::range(1, 1, rows * n_head) *
                                  sycl::range(1, 1, dim),
                              sycl::range(1, 1, dim)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gdn_out_kernel(out, z, o_norm, dim, norm_eps);
            });
    }
}

void coreset_mask(float* logits, int n_expert, int n_used, int rows, float share, void* stream) {
    if (n_expert % WARP != 0 || n_expert > 1024) throw std::invalid_argument("coreset_mask: 32 to 1024 experts, in 32s");
    /*
    DPCT1049:120: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs(stream)->submit([&](sycl::handler &cgh) {
            /*
            DPCT1083:373: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(2 * size_t(n_expert) * sizeof(float)), cgh);

            cgh.parallel_for<
                dpct_kernel_name<class coreset_mask_kernel_38851b>>(
                sycl::nd_range<3>(sycl::range(1, 1, n_expert),
                                  sycl::range(1, 1, n_expert)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        coreset_mask_kernel(
                            logits, n_expert, n_used, rows, share,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
}

}  // namespace strata::poly
