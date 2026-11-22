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
    native_quantize_q8_1(y, y_q8_1, n, rows, stream);
}
// The multi-matrix fast paths are not ported yet: false, with nothing launched, is their contract, and the engine
// then launches the matrices one by one.
bool native_mmvq_slots(int, const MmvqSlots&, int, int, int, void*) { return false; }
bool native_mmvq_slots_wide() { return false; }
bool native_mmvq_ids_supported(int) { return false; }
bool native_mmvq_ids(int, const MmvqIds&, int, int, int, void*) { return false; }
bool native_mmvq_jobs(int, const MmvqJobs&, int, int, int, void*) { return false; }
}  // namespace strata::kernels
