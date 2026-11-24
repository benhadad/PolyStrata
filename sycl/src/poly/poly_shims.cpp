// SYCL port: the poly engine calls kernels the port's native_mmvq has no copy of.
#include <stdexcept>
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"
#include <dpct/dpct.hpp>
namespace { const bool g_vram_track = (dpct::detail::vram_tracker().on = true); }   // WSL2: no driver free-memory query
namespace strata::kernels {
// An RMS norm and the Q8_1 quantization of its output: the two launches (poly::rms_norm, native_quantize_q8_1)
// this is defined to equal bit for bit, as two launches.  `threads` is rms_norm()'s own sum_threads(n).
void native_add_rms_norm_q8_1(float* x, const float* add, const float* w, float* y, void* y_q8_1, int n, int rows,
                              float eps, int threads, void* stream) {
    (void) threads;
    if (add) throw std::runtime_error("native_add_rms_norm_q8_1: the residual add is not in the SYCL shim");
    strata::poly::rms_norm(x, w, y, n, rows, eps, stream);
    // The rows are contiguous and n is a multiple of the block size, so `rows` rows quantize as one vector of n * rows
    // elements, laid out as native_q8_1_bytes(n, rows) expects.  Passing `rows` as ncols hit the 8-column limit of
    // native_quantize_q8_1(), so a prompt batch of more than 8 tokens threw "requires 1 <= ncols <= 8".
    native_quantize_q8_1(y, y_q8_1, n * rows, 1, stream);
}
}  // namespace strata::kernels
