#pragma once

#include <dpct/dpct.hpp>
#include "strata/kernels/q8_1_finite.hpp"

namespace strata::poly::norm_q8_1_detail {

constexpr int kLanes = 32;

template <bool Max>
__dpct_inline__ float warp_reduce(sycl::sub_group sg, float value) {
#pragma unroll
    for (int offset = kLanes / 2; offset > 0; offset >>= 1) {
        const float peer = dpct::experimental::permute_sub_group_by_xor(0xffffffffu, sg, value, offset);
        if constexpr (Max) value = sycl::fmax(value, peer);
        else value += peer;
    }
    return value;
}

// Each warp sums in the same order as the original PolyStrata RMSNorm.
// The caller chooses the same work-group size for both storage layouts.
__dpct_inline__ void write_row(const float* __restrict__ x, const float* __restrict__ w,
                               float* y, uint8_t* q8, int n, float eps, sycl::nd_item<3> item) {
    auto& part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(item.get_group());
    const auto sg = item.get_sub_group();
    const int tid = item.get_local_id(2), threads = item.get_local_range(2);
    float ss = 0.0f;
#pragma unroll
    for (int i = tid; i < n; i += threads) ss += x[i] * x[i];
    ss = warp_reduce<false>(sg, ss);
    if (tid % kLanes == 0) part[tid / kLanes] = ss;
    item.barrier(sycl::access::fence_space::local_space);
    ss = 0.0f;
#pragma unroll
    for (int warp = 0; warp < threads / kLanes; ++warp) ss += part[warp];
    item.barrier(sycl::access::fence_space::local_space);

    const float scale = 1.0f / sycl::sqrt(ss / float(n) + eps);
    for (int i = tid; i < n; i += threads) {
        const float yv = w ? x[i] * scale * w[i] : x[i] * scale;
        y[i] = yv;
        const float amax = warp_reduce<true>(sg, sycl::fabs(yv));
        const float sum = warp_reduce<false>(sg, yv);
        const float d = kernels::q8_1_finite(amax / 127.0f);
        uint8_t* blk = q8 + size_t(i / kLanes) * 36;
        reinterpret_cast<int8_t*>(blk + 4)[i % kLanes] = kernels::q8_1_quant(yv, d, amax);
        if (i % kLanes == 0) *reinterpret_cast<sycl::half2*>(blk) = kernels::q8_1_ds(d, sum);
    }
}

}  // namespace strata::poly::norm_q8_1_detail
