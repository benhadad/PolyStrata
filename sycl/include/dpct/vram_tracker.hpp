#pragma once
#include <atomic>
#include <cstddef>
#include <mutex>
#include <unordered_map>
namespace dpct {
namespace detail {
// SYCL port (strata-poly, WSL2): a count of dpct's device allocations, for get_memory_info() where the driver reports
// no free memory.  Off unless a program sets vram_tracker().on.
// One global counter, not one per device: right for a single GPU (the WSL2 case); a multi-device (--layer-split) user
// must key `bytes` by device before relying on get_memory_info().  Allocations made with sycl::malloc_device directly
// are only counted if the caller reports them with vram_note() / vram_drop().
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

}  // namespace detail
}  // namespace dpct
