#!/usr/bin/env python3
"""strata-poly on WSL2: Level Zero reports no free VRAM and there is no DRM fdinfo, so count dpct's own device
allocations and let get_memory_info report total - tracked when a program switches that on (idempotent)."""
import os
R = os.path.expanduser("~/PolyStrata/sycl/include/dpct")
m = os.path.join(R, "detail/memory_detail.hpp")
t = open(m).read()
if "vram_tracker" not in t:
    pre = r'''
// SYCL port (strata-poly, WSL2): a count of dpct's device allocations, for get_memory_info() where the driver reports
// no free memory.  Off unless a program sets vram_tracker().on.
struct vram_tracker_t {
  std::atomic<bool> on{false};
  std::atomic<unsigned long long> bytes{0};
  std::mutex mu;
  std::unordered_map<void *, size_t> sizes;
};
inline vram_tracker_t &vram_tracker() {
  static vram_tracker_t t;
  return t;
}
inline void *vram_note(void *p, size_t n) {
  auto &t = vram_tracker();
  if (p && t.on.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> g(t.mu);
    t.sizes[p] = n;
    t.bytes += n;
  }
  return p;
}
inline void vram_drop(void *p) {
  auto &t = vram_tracker();
  if (p && t.on.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> g(t.mu);
    auto it = t.sizes.find(p);
    if (it != t.sizes.end()) {
      t.bytes -= it->second;
      t.sizes.erase(it);
    }
  }
}

'''
    t = t.replace("static inline void *dpct_malloc(size_t size, sycl::queue &q) {", pre + "static inline void *dpct_malloc(size_t size, sycl::queue &q) {", 1)
    t = t.replace("  return sycl::malloc_device(size, q.get_device(), q.get_context());\n#endif // DPCT_USM_LEVEL_NONE\n}",
                  "  return vram_note(sycl::malloc_device(size, q.get_device(), q.get_context()), size);\n#endif // DPCT_USM_LEVEL_NONE\n}", 1)
    t = t.replace("    sycl::free(ptr, q.get_context());\n#endif // DPCT_USM_LEVEL_NONE\n  }\n}",
                  "    vram_drop(ptr);\n    sycl::free(ptr, q.get_context());\n#endif // DPCT_USM_LEVEL_NONE\n  }\n}", 1)
    if "#include <unordered_map>" not in t:
        t = t.replace("#include", "#include <atomic>\n#include <mutex>\n#include <unordered_map>\n#include", 1)
    open(m, "w").write(t)

d = os.path.join(R, "device.hpp")
t = open(d).read()
if "vram_tracker" not in t:
    old = '''    if (!has(sycl::aspect::ext_intel_free_memory)) {
      std::cerr << "get_memory_info: ext_intel_free_memory is not supported." << std::endl;
      free_memory = 0;
    } else {'''
    new = '''    if (!has(sycl::aspect::ext_intel_free_memory)) {
      if (detail::vram_tracker().on.load()) {   // SYCL port (strata-poly): total less what dpct allocated
        const size_t tot = get_device_info().get_global_mem_size();
        const size_t used = detail::vram_tracker().bytes.load();
        free_memory = tot > used ? tot - used : 0;
      } else {
        std::cerr << "get_memory_info: ext_intel_free_memory is not supported." << std::endl;
        free_memory = 0;
      }
    } else {'''
    assert old in t
    t = t.replace(old, new, 1)
    open(d, "w").write(t)

s = os.path.expanduser("~/PolyStrata/sycl/src/poly/poly_shims.cpp")
t = open(s).read()
if "vram_tracker" not in t:
    t = t.replace('#include "strata/poly/kernels.hpp"\n', '#include "strata/poly/kernels.hpp"\n#include <dpct/dpct.hpp>\nnamespace { const bool g_vram_track = (dpct::detail::vram_tracker().on = true); }   // WSL2: no driver free-memory query\n', 1)
    open(s, "w").write(t)
print("ok")
