#!/usr/bin/env python3
"""Hand fixes on top of dpct's output for src/poly/kernels_inkling.cu (modelled on PolyStrata sycl/tools/fixups.py).
Idempotent.  usage: ink_fixups.py <out-root>   (the dpct --out-root: holds src/poly/kernels_inkling.dp.cpp and include/)"""
import re, sys, pathlib
root = pathlib.Path(sys.argv[1])
f = root / "src/poly/kernels_inkling.dp.cpp"
s = f.read_text(); orig = s

# 1. launches dpct could not migrate (it wrote `<kernel>_wrapper<<<...>>>`, a name that does not exist). These kernels
#    take their local memory at kernel scope (group_local_memory_for_overwrite), so a launch is a bare parallel_for.
MACRO = '''
// SYCL port (ink_fixups.py): launches dpct left as `<<<>>>`. The kernels take no dynamic shared memory.
#define INK_LAUNCH(NAME, GRID, BLOCK, ST, ...)                                                          \\
    (ST)->parallel_for<class NAME>(                                                                     \\
        sycl::nd_range<3>((GRID) * sycl::range<3>(1, 1, (BLOCK)), sycl::range<3>(1, 1, (BLOCK))),       \\
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { __VA_ARGS__; })
'''
if "INK_LAUNCH" not in s:
    s = s.replace("#include <type_traits>\n", "#include <type_traits>\n" + MACRO, 1)

# mix4: eight explicit cases
def mix4(m):
    n = m.group(2) or "8"
    return f"{m.group(1)}INK_LAUNCH(ink_mix4_{n}, dpct::dim3(rows), threads, st, mix4_kernel<{n}>(m, n, eps)); break;"
s = re.sub(r"(case (\d): |default: )mix4_kernel_wrapper<(\d)><<<rows, threads, 0, st>>>\(m, n, eps\); break;",
           lambda m: f"{m.group(1)}INK_LAUNCH(ink_mix4_{m.group(3)}, dpct::dim3(rows), threads, st, mix4_kernel<{m.group(3)}>(m, n, eps)); break;", s)

# kv_store: two launches (one sequence / many)
s = re.sub(r"kv_store_kernel_wrapper<<<dim3\(kv_heads, 2\), AT_DIM, 0, cs\(stream\)>>>\(\s*a, KvOne\{\}\);",
           "INK_LAUNCH(ink_kv_store_one, dpct::dim3(kv_heads, 2), AT_DIM, cs(stream), kv_store_kernel(a, KvOne{}));", s)
s = re.sub(r"kv_store_kernel_wrapper<<<dim3\(kv_heads, 2, rows\), AT_DIM, 0, cs\(stream\)>>>\(\s*a, r\);",
           "INK_LAUNCH(ink_kv_store_seqs, dpct::dim3(kv_heads, 2, rows), AT_DIM, cs(stream), kv_store_kernel(a, r));", s)

# 2. the tensor-core attention: inline PTX (mma.sync, ldmatrix, __cvta_generic_to_shared). dpct's DPCT_COMPATIBILITY_TEMP
#    (900) selects it; switch it off and let mma_attention() refuse, so the caller takes the scalar kernels (as on HIP).
# dpct left a `\` line continuation after the `||`
s = re.sub(r"#if defined\(__HIPCC__\) \|\|\s*\\?\s*\n\s*\(defined\(DPCT_COMPATIBILITY_TEMP\) && DPCT_COMPATIBILITY_TEMP < 750\)\n#define INKLING_MMA 0",
           "#if 1   // SYCL port: inline PTX mma/ldmatrix; the XMX joint_matrix version is follow-up work\n#define INKLING_MMA 0", s)
s = re.sub(r"    dpct::kernel_function_info fa\{\};\n    return cudaFuncGetAttributes\(&fa, attn_mma_kernel_wrapper<false, false>\) ==\s*0 &&\s*fa\.ptxVersion >= 75;",
           "    return false;   // SYCL port: no tensor-core attention yet", s)
s = re.sub(r"if \(kq && vq\) attn_mma_kernel_wrapper<true, true>.*?attn_mma_kernel_wrapper<false, false>\s*<<<gm, MMA_WARPS \* WARP, 0, st>>>\(a\);",
           "(void) gm;   // SYCL port: mma_attention() is false; no tensor-core launch", s, flags=re.S)

# 2b. mix4_kernel<NS>: dpct converted `y.x()` but left `xs[k].x` (a dependent-type float4 element) as a member access.
s = re.sub(r"(\bxs\[\w+\])\.([xyzw])\b(?!\s*\()", r"\1.\2()", s)

# 2c. conv_state_kernel launches inside a template<ACC> function with a kernel name that lacks ACC: two instantiations
#     collide (`redefinition of KernelInfo<...conv_state_kernel_685591>`). Name it per ACC like its siblings.
s = s.replace("dpct_kernel_name<class conv_state_kernel_685591>>", "dpct_kernel_name<class conv_state_kernel_685591, dpct_kernel_scalar<ACC>>>")

# 3. cudaFuncSetAttribute(MaxDynamicSharedMemorySize): the local_accessor's size is the launch's, nothing to set.
s = re.sub(r"const dpct::err0 e = cudaFuncSetAttribute\(.*?cudaFuncAttributeMaxDynamicSharedMemorySize, \(int\)smem\);",
           "const dpct::err0 e = 0;   // SYCL port: the local_accessor sizes the launch", s, flags=re.S)
if s != orig: f.write_text(s); print("fixed", f.name)

# 4. dp4a.hpp: __nanosleep (same fix as the repo's fixups.py)
h = root / "include/strata/kernels/dp4a.hpp"
if h.exists():
    t = h.read_text(); u = t.replace("    __nanosleep(100);\n", "    // SYCL port: no __nanosleep; the bounded spin is the backoff\n")
    if u != t: h.write_text(u); print("fixed", h.name)
