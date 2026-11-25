// SYCL port: the small kernels around the grouped XMX expert GEMM of a prompt batch (Engine::moe_xmx()).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace strata::poly::xmx {
/// Whether one of the matrix-unit paths for a prompt batch is on (name: STRATA_MOE_XMX, STRATA_PROJ_XMX or
/// STRATA_ATTN_XMX): on unless its variable, or STRATA_XMX for all three, is 0.  The variable of the path wins.
inline bool enabled(const char* name) {
    if (const char* v = std::getenv(name)) return std::atoi(v) != 0;
    if (const char* v = std::getenv("STRATA_XMX")) return std::atoi(v) != 0;
    return true;
}
/// out[i][0..n) = half(x[rows[i]][0..n)) for i < count; x has ld floats a row.
void gather_f16(const float* x, int ld, const int32_t* rows, int count, int n, uint16_t* out, void* stream);
/// out[i] = half(in[i]), i < count.
void f32_to_f16(const float* in, uint16_t* out, size_t count, void* stream);
/// acc[t][i] += sum over k < n_used of w[t * n_used + k] * y[pos[t * n_used + k]][i], summed in k order as moe_sum() does.
void sum_perm(const float* y, const int32_t* pos, const float* w, float* acc, int n, int n_used, int tokens, void* stream);
}  // namespace strata::poly::xmx
