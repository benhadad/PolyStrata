#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::poly {

inline float sycl_situ(float gate, float up, float beta, float linear_beta) {
    const float sigmoid = 1.0f / (1.0f + sycl::native::exp(-gate));
    const float act = sycl::tanh(gate * (1.0f / beta)) * beta * sigmoid;
    const float linear = linear_beta > 0.0f ? sycl::tanh(up * (1.0f / linear_beta)) * linear_beta : up;
    return act * linear;
}

}  // namespace strata::poly

namespace strata::kernels {

struct XmxSitu {
    const float* gate_scales;
    const float* up_scales;
    const int32_t* expert_ids;   // one expert ID per sorted input row, needed only when scales are present
    float beta;
    float linear_beta;
};

// BMG/LNL: merged NVFP4 or Q8_0 gate/up weights [2 * n_ff][K], FP16 inputs and activations.
void iq_xmx_grouped_situ(int ty, const uint8_t* const* weights, const int32_t* counts, int experts,
                         const uint16_t* inputs, uint16_t* activations, int n_ff, int K, const XmxSitu& situ, void* stream);

}  // namespace strata::kernels
