#!/usr/bin/env python3
"""Move the tracker into its own header, included by device.hpp and memory_detail.hpp."""
import os, re
R = os.path.expanduser("~/PolyStrata/sycl/include/dpct")
m = os.path.join(R, "detail/memory_detail.hpp")
t = open(m).read()
a = t.index("// SYCL port (strata-poly, WSL2): a count of dpct")
b = t.index("static inline void *dpct_malloc(size_t size, sycl::queue &q) {")
block = t[a:b]
t = t[:a] + t[b:]
t = t.replace("#include <atomic>\n#include <mutex>\n#include <unordered_map>\n", "", 1)
if "vram_tracker.hpp" not in t:
    t = t.replace("#include", '#include "../vram_tracker.hpp"\n#include', 1)
open(m, "w").write(t)
hdr = "#pragma once\n#include <atomic>\n#include <cstddef>\n#include <mutex>\n#include <unordered_map>\nnamespace dpct {\nnamespace detail {\n" + block + "}  // namespace detail\n}  // namespace dpct\n"
open(os.path.join(R, "vram_tracker.hpp"), "w").write(hdr)
d = os.path.join(R, "device.hpp")
t = open(d).read()
if "vram_tracker.hpp" not in t:
    t = t.replace("#include", '#include "vram_tracker.hpp"\n#include', 1)
    open(d, "w").write(t)
print("ok")
