#pragma once

#include <array>

namespace strata::poly {

struct NormQ81Row {
    const float* input;
    const float* weight;
    float* normalized;
    void* quantized;
};

// Independently normalize and quantize three decode rows in one launch.
void rms_norm_q8_1_qkv(std::array<NormQ81Row, 3> rows, int n, float eps, void* stream);

struct InklingMix;

struct MixNormOutput {
    const float* weight;
    float* normalized;
    void* quantized;
};

// Return false when the existing split mix and norm kernels are needed.
bool inkling_mix_norm(const InklingMix& mix, std::array<MixNormOutput, 3> outputs,
                      int n, float mix_eps, float norm_eps, void* stream);

}  // namespace strata::poly
