#include "moe_xmx.hpp"

#include <sycl/sycl.hpp>
#include "strata/sycl_queue.hpp"

namespace strata::poly::xmx {

void gather_f16(const float* x, int ld, const int32_t* rows, int count, int n, uint16_t* out, void* stream) {
    if (count <= 0 || n <= 0) return;
    sycl::queue& q = *strata::q_of(stream);
    const int per = (n + 255) / 256;
    q.parallel_for(sycl::nd_range<1>((size_t) count * per * 256, 256), [=](sycl::nd_item<1> it) {
        const size_t g = it.get_group(0);
        const int r = (int) (g / per), i = (int) (g % per) * 256 + (int) it.get_local_id(0);
        if (i >= n) return;
        const sycl::half h = (sycl::half) x[(size_t) rows[r] * ld + i];
        out[(size_t) r * n + i] = sycl::bit_cast<uint16_t>(h);
    });
}

void f32_to_f16(const float* in, uint16_t* out, size_t count, void* stream) {
    if (count == 0) return;
    sycl::queue& q = *strata::q_of(stream);
    const size_t groups = (count + 255) / 256;
    q.parallel_for(sycl::nd_range<1>(groups * 256, 256), [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        if (i < count) out[i] = sycl::bit_cast<uint16_t>((sycl::half) in[i]);
    });
}

void sum_perm(const float* y, const int32_t* pos, const float* w, float* acc, int n, int n_used, int tokens, void* stream) {
    if (tokens <= 0 || n <= 0) return;
    sycl::queue& q = *strata::q_of(stream);
    const int per = (n + 255) / 256;
    q.parallel_for(sycl::nd_range<1>((size_t) tokens * per * 256, 256), [=](sycl::nd_item<1> it) {
        const size_t g = it.get_group(0);
        const int t = (int) (g / per), i = (int) (g % per) * 256 + (int) it.get_local_id(0);
        if (i >= n) return;
        const size_t first = (size_t) t * n_used;
        float s = 0.0f;
        for (int k = 0; k < n_used; ++k) s += w[first + k] * y[(size_t) pos[first + k] * n + i];
        acc[(size_t) t * n + i] += s;
    });
}

}  // namespace strata::poly::xmx
