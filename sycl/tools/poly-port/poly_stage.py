#!/usr/bin/env python3
"""Stage the migrated poly engine into ~/PolyStrata/sycl/src/poly and add the CMake target (idempotent)."""
import os, re, shutil, sys

R = os.path.expanduser("~/PolyStrata")
O = os.path.expanduser("~/poly-spike/out/src/poly")
D = os.path.join(R, "sycl/src/poly")
os.makedirs(D, exist_ok=True)

for src, dst in [("kernels.dp.cpp", "kernels.dp.cpp"), ("kernels_dsv4.dp.cpp", "kernels_dsv4.dp.cpp"),
                 ("kernels_inkling.dp.cpp", "kernels_inkling.dp.cpp"), ("engine.cpp.dp.cpp", "engine.cpp"),
                 ("inkling.cpp.dp.cpp", "inkling.cpp"), ("spill.cpp", "spill.cpp"), ("main.cpp", "main.cpp")]:
    shutil.copy(os.path.join(O, src), os.path.join(D, dst))

# Inkling-only engine: the other architectures are not migrated (they need fattn / the Qwen3.8 kernels)
p = os.path.join(D, "engine.cpp")
s = open(p).read()
for h in ("deepseek4", "gemma4", "glm5next", "kolibri1", "lfm2moe", "flint", "qwen35"):
    s = s.replace('#include "%s.hpp"\n' % h, "")
s = re.sub(r"        if \(name == \"glm5-next\".*?Kolibri1>\(\);\n", "", s, flags=re.S)
s = s.replace("#include \"inkling.hpp\"", "#include \"inkling.hpp\"\n#include \"strata/poly/kernels.hpp\"", 1) if False else s
open(p, "w").write(s)

open(os.path.join(D, "mmq_stub.cpp"), "w").write(r'''// SYCL port: no ggml MMQ tiles.  Mmq::fits() is false for everything, so the engine never builds an Mmq
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
''')

open(os.path.join(D, "poly_shims.cpp"), "w").write(r'''// SYCL port: the poly engine calls kernels the port's native_mmvq has no copy of.
#include <stdexcept>
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"
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
}  // namespace strata::kernels
''')

# declaration in the port's header
h = os.path.join(R, "sycl/include/strata/kernels/native_mmvq.hpp")
t = open(h).read()
if "native_add_rms_norm_q8_1" not in t:
    decl = ("\n/// SYCL port (poly_shims.cpp): rms norm + Q8_1 quantize of its output; `add` must be null.\n"
            "void native_add_rms_norm_q8_1(float* x, const float* add, const float* w, float* y, void* y_q8_1, int n, int rows,\n"
            "                              float eps, int threads, void* stream);\n")
    i = t.index("void native_quantize_q8_1(")
    j = t.index(";", i) + 1
    t = t[:j] + decl + t[j:]
    open(h, "w").write(t)

c = os.path.join(R, "sycl/CMakeLists.txt")
t = open(c).read()
if "STRATA_SYCL_POLY" not in t:
    t += r'''
# --- strata-poly (Inkling 2 only): the poly engine, migrated with dpct (sycl/tools + poly_fixups) ------------------
option(STRATA_SYCL_POLY "Build strata-poly (the poly engine; Inkling 2 only in this port)" OFF)
if(STRATA_SYCL_POLY AND STRATA_NATIVE_EXPERTS)
  if(STRATA_GGML_DIR)
    set(_poly_gg ${STRATA_GGML_DIR}/ggml)
  else()
    set(_poly_gg ${strata_llamacpp_SOURCE_DIR}/ggml)
  endif()
  strata_resolve(_poly src/poly/kernels.cu src/poly/kernels_dsv4.cu src/poly/kernels_inkling.cu
                       src/poly/engine.cpp src/poly/inkling.cpp src/poly/spill.cpp src/poly/main.cpp)
  list(APPEND _poly ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/mmq_stub.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/poly/poly_shims.cpp
       ${STRATA_ROOT}/src/poly/cpu_experts.cpp ${STRATA_ROOT}/src/poly/cpu_lowbit.cpp)
  set_source_files_properties(${STRATA_ROOT}/src/poly/cpu_lowbit.cpp PROPERTIES INCLUDE_DIRECTORIES ${_poly_gg}/src)
  add_executable(strata-poly ${_poly})
  target_include_directories(strata-poly PRIVATE ${STRATA_ROOT}/src/poly ${STRATA_ROOT}/src ${_poly_gg}/include ${_poly_gg}/src)
  target_link_libraries(strata-poly PRIVATE strata_engine strata_kernels strata_spec ggml-cpu ggml-base Threads::Threads)
endif()
'''
    open(c, "w").write(t)
print("staged")
