#pragma once

namespace strata::poly {

// The convolution of one token of one channel from its inputs s0 (oldest) .. s2 and v (the token's), in
// upstream's order: ssm_conv's sum, then the four input-dependent taps lag 0..3, then the residual and the scale.
inline float conv_one(float s0, float s1, float s2, float v,
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

}  // namespace strata::poly

namespace strata::kernels {

// Fuse the raw Q8_0 wide matvec and Inkling's four-tap output convolution.
// Return false before enqueuing work when the split projection path is required.
bool native_q8_0_postconv(const void* weights, const float* input, void* xq, int n_in, int n_out,
                         const float* conv, float* history, float* residual, bool fresh, void* stream);

}  // namespace strata::kernels
