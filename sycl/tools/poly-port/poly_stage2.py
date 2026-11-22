#!/usr/bin/env python3
"""Add the upstream slots/ids/jobs mmvq API (declarations from the root header, stub bodies) to the SYCL port."""
import os
R = os.path.expanduser("~/PolyStrata")
root = open(os.path.join(R, "include/strata/kernels/native_mmvq.hpp")).read()
a = root.index("// Up to kMmvqSlotsMax (32)")
b = root.index("bool native_mmvq_jobs(")
b = root.index("\n", b) + 1
block = root[a:b]
h = os.path.join(R, "sycl/include/strata/kernels/native_mmvq.hpp")
t = open(h).read()
if "kMmvqSlotsMax" not in t:
    end = t.rindex("} // namespace strata::kernels")
    t = t[:end] + "\n// SYCL port: declarations as upstream; poly_shims.cpp defines them as 'no kernel' (callers fall back).\n" + block + "\n" + t[end:]
    if "#include <cstdint>" not in t:
        t = t.replace("#pragma once", "#pragma once\n#include <cstddef>\n#include <cstdint>", 1)
    open(h, "w").write(t)

s = os.path.join(R, "sycl/src/poly/poly_shims.cpp")
t = open(s).read()
if "native_mmvq_slots(" not in t:
    t = t.replace("}  // namespace strata::kernels", r'''// The multi-matrix fast paths are not ported yet: false, with nothing launched, is their contract, and the engine
// then launches the matrices one by one.
bool native_mmvq_slots(int, const MmvqSlots&, int, int, int, void*) { return false; }
bool native_mmvq_slots_wide() { return false; }
bool native_mmvq_ids_supported(int) { return false; }
bool native_mmvq_ids(int, const MmvqIds&, int, int, int, void*) { return false; }
bool native_mmvq_jobs(int, const MmvqJobs&, int, int, int, void*) { return false; }
}  // namespace strata::kernels''')
    open(s, "w").write(t)
print("ok")
