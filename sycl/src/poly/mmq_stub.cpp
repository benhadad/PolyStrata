// SYCL port: no ggml MMQ tiles.  Mmq::fits() is false for everything, so the engine never builds an Mmq
// (Engine: `L.mmq` and `batched()` need mmq_) and prompt rows take its native MMVQ path instead.
#include "mmq.hpp"
#include <stdexcept>
namespace strata::poly {
Mmq::Mmq() {}
Mmq::~Mmq() {}
bool Mmq::fits(int, int64_t, int64_t, bool) { return false; }
size_t Mmq::q8_bytes(int64_t, int64_t) { return 0; }
bool Mmq::half_sums(int) { return false; }
bool Mmq::split_exact(int, int64_t) { return true; }
void Mmq::quantize(const float*, const int32_t*, void*, int, int64_t, int64_t, int64_t, void*) {
    throw std::runtime_error("strata-poly: MMQ is not available in the SYCL build");
}
void Mmq::reserve_fixup() {}
void Mmq::mul(const void*, int, int64_t, int64_t, size_t, int, const void*, const int32_t*, const int32_t*, int64_t,
              int64_t, float*, int64_t, void*) {
    throw std::runtime_error("strata-poly: MMQ is not available in the SYCL build");
}
}  // namespace strata::poly
