#!/usr/bin/env python3
"""Hand fixes for the dpct output of src/poly (kernels.cu, kernels_inkling.cu, engine.cpp). Idempotent.
usage: poly_fixups.py <out-root>   (the dir that holds src/poly/*.dp.cpp)"""
import re, sys, pathlib

root = pathlib.Path(sys.argv[1])
P = root / "src" / "poly"
changed = []


def edit(name, fn):
    p = P / name
    if not p.exists():
        print("  (absent)", name)
        return
    s = p.read_text()
    t = fn(s)
    if t != s:
        p.write_text(t)
        changed.append(name)
        print("  fixed", name)


def stub_fn(s, name, ret, body):
    """Replace Engine::<name>'s definition (incl. dpct's function-try/catch) with `body`."""
    m = re.search(r"^(?P<sig>(?:void|bool|size_t)\s+Engine::%s\([^)]*\))(?P<try>\s*try)?\s*\{" % name, s, re.M)
    if not m:
        return s
    i = m.end() - 1
    depth = 0
    n = len(s)
    j = i
    while j < n:
        c = s[j]
        if s.startswith("//", j):
            j = s.index("\n", j)
            continue
        if s.startswith("/*", j):
            j = s.index("*/", j) + 2
            continue
        if c == '"':
            j += 1
            while s[j] != '"':
                j += 2 if s[j] == "\\" else 1
        elif c == "'":
            j += 1
            while s[j] != "'":
                j += 2 if s[j] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    end = j + 1
    cm = re.match(r"\s*catch\s*\(sycl::exception const &exc\)\s*\{", s[end:])
    if cm:
        k = end + cm.end() - 1
        d = 0
        while True:
            if s[k] == "{":
                d += 1
            elif s[k] == "}":
                d -= 1
                if d == 0:
                    break
            k += 1
        end = k + 1
    return s[: m.start()] + m.group("sig") + " {\n" + body + "\n}" + s[end:]


# ---- kernels.dp.cpp / kernels_inkling.dp.cpp: no tensor-core / inline-PTX paths --------------------------------------
PRELUDE = "// SYCL port: the PTX address conversion only feeds paths that are off here\n#define __cvta_generic_to_shared(p) ((size_t)(p))\n"


def wrapper_names(s):
    """dpct's `template <...> void X_wrapper(...)` launches one fixed kernel name for every instantiation: add the
    template parameters to the name (non-type ones through dpct_kernel_scalar)."""
    out, pos = [], 0
    pat = re.compile(r"template <([^>]*)>\s*\n(?:__dpct_inline__ )?void (\w+)_wrapper\(")
    for m in pat.finditer(s):
        cls = re.compile(r"dpct_kernel_name<class (%s_[0-9a-f]*)>>" % re.escape(m.group(2)))
        c = cls.search(s, m.end())
        if not c or c.start() - m.end() > 6000:
            continue
        args = []
        for p in m.group(1).split(","):
            p = p.split("=")[0].strip()   # a default (`bool Q8 = false`) is not the parameter's name
            if not p:
                continue
            kind, name = p.rsplit(None, 1)
            args.append(name if kind in ("typename", "class") else "dpct_kernel_scalar<%s>" % name)
        if not args:
            continue
        out.append((c.start(), c.end(), "dpct_kernel_name<class %s, %s>>" % (c.group(1), ", ".join(args))))
    for a, b, t in sorted(out, reverse=True):
        s = s[:a] + t + s[b:]
    return s


def kern(s):
    # STRATA_POLY_MMA: off, as on HIP (the host keeps the non-tensor-core products)
    s = re.sub(r"#if defined\(__HIPCC__\) \|\|\s*\\?\s*\(defined\(DPCT_COMPATIBILITY_TEMP\) && DPCT_COMPATIBILITY_TEMP < 750\)",
               "#if 1   // SYCL port: no tensor-core (mma/ldmatrix) path", s)
    # sm_80 cp.async helpers: the plain-copy branch
    s = s.replace("#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 800",
                  "#if 0   // SYCL port: no cp.async")
    # the host probes that read a kernel's PTX version: report 'no'
    s = re.sub(r"#if defined\(__HIPCC__\)\n(\s*return (?:0|false);\n)#else\n(\s*dpct::kernel_function_info)",
               r"#if 1   // SYCL port\n\1#else\n\2", s)
    # a kernel name with a type from an anonymous namespace
    s = s.replace("strata::poly::(anonymous namespace)::", "")
    # a template argument dpct could not deduce, for a kernel that is not a template
    s = re.sub(r"(class attn_value_kernel_[0-9a-f]+),\s*dpct_placeholder /\*Fix the type mannually\*/>>", r"\1>>", s)
    # dpct (DPCT1013) drops the parentheses of nested __fadd_rn/__fsub_rn/__fmul_rn calls: restore the CUDA's grouping
    s = s.replace("const float d = va - sk * sa;",
                  "// the CUDA's __fmul_rn(__fsub_rn(va, sk), sa): dpct dropped the parentheses and made it va - sk * sa\n"
                  "        const float d = (va - sk) * sa;")
    s = s.replace("float v = sh[h][0] + sh[h][2] + sh[h][1] + sh[h][3];",
                  "// the CUDA's __fadd_rn(__fadd_rn(sh0, sh2), __fadd_rn(sh1, sh3)): dpct dropped the parentheses\n"
                  "                float v = (sh[h][0] + sh[h][2]) + (sh[h][1] + sh[h][3]);")
    s = wrapper_names(s)
    if "__cvta_generic_to_shared(p)" not in s.split("#include", 3)[0] and "#define __cvta_generic_to_shared" not in s:
        s = re.sub(r"(#include <dpct/dpct\.hpp>\n)", r"\1" + PRELUDE.replace("\\", "\\\\"), s, count=1)
    return s


edit("kernels.dp.cpp", kern)
edit("kernels_inkling.dp.cpp", kern)
edit("kernels_dsv4.dp.cpp", kern)

# ---- engine.cpp: no driver VMM on Level Zero (the repo's own port stubs src/core/vmm.cpp the same way) -------------------
NOVMM = 'throw std::runtime_error("strata-poly: VMM is not available in the SYCL build");'


def eng(s):
    # dpct turns cudaFreeHost into plain free(), but the buffers came from sycl::malloc_host
    s = re.sub(r"DPCT_CHECK_ERROR\(free\((\w+)\)\)", r"DPCT_CHECK_ERROR(sycl::free(\1, dpct::get_in_order_queue()))", s)
    s = s.replace("pool_on_ = opt_.vram_pool && vmm_ && opt_.kv_grow;",
                  "vmm_ = false;   // SYCL port: no VMM; every cache is allocated whole\n    pool_on_ = false;")
    s = stub_fn(s, "vmm_reserve", "void", "    (void) m; (void) bytes;\n    " + NOVMM)
    s = stub_fn(s, "vpool_claim", "void", "    (void) bytes;")
    s = stub_fn(s, "vpool_claim_to_budget", "bool", "    (void) margin; (void) limit; claimed = 0;\n    return false;")
    s = stub_fn(s, "vmm_settle", "void", "    ")
    s = stub_fn(s, "vmm_map", "void", "    (void) m; (void) bytes; (void) piece; (void) what;\n    " + NOVMM)
    s = stub_fn(s, "vmm_unmap", "void", "    (void) m; (void) bytes;")
    s = stub_fn(s, "vmm_free", "void", "    (void) m;")
    s = re.sub(r"for \(unsigned long long h : vpool_\) DPCT_CHECK_ERROR\(\s*delete \(h\)\);[^\n]*\n", "", s)
    # dpct's destroy_queue takes an lvalue
    s = re.sub(r"dpct::get_current_device\(\)\.destroy_queue\(cs\((\w+)\)\)",
               r"{ dpct::queue_ptr q_ = cs(\1); dpct::get_current_device().destroy_queue(q_); }", s)
    # cudaGraphInstantiateWithFlags -> finalize()
    s = re.sub(r"const dpct::err0 made = cudaGraphInstantiateWithFlags\(&exec, g, 0\);",
               "dpct::err0 made = 0;\n"
               "    try {\n"
               "        exec = new sycl::ext::oneapi::experimental::command_graph<\n"
               "            sycl::ext::oneapi::experimental::graph_state::executable>(g->finalize());\n"
               "    } catch (sycl::exception const&) { made = 1; }", s)
    return s


def swap(s, old, new, what):
    """Replace `old` by `new` exactly once; idempotent (a no-op when `new` is already there)."""
    if new in s:
        return s
    if s.count(old) != 1:
        sys.exit("poly_fixups: anchor for '%s' found %d times (dpct output changed?)" % (what, s.count(old)))
    return s.replace(old, new, 1)


# The model families this port does not build: the engine factory is Inkling-only, so their headers are dropped.
DROPPED = ["qwen35", "gemma4", "glm5next", "deepseek4", "flint", "kolibri1", "lfm2moe"]


def eng_sycl(s):
    """The SYCL-port changes to engine.cpp that dpct does not produce."""
    # vram_tracker: WSL2 has no free-VRAM query, so every engine allocation is counted (dmalloc) and uncounted (frees)
    s = swap(s, "    for (void *p : dev_allocs_)\n        DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue()));\n",
             "    for (void *p : dev_allocs_) {\n        dpct::detail::vram_drop(p);\n"
             "        DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue()));\n    }\n",
             "destructor free loop")
    s = swap(s, "    }\n    dev_allocs_.push_back(p);\n    return p;",
             "    }\n    // SYCL port: count it for dpct::get_memory_info() (WSL2 has no driver free-memory query); see vram_tracker.hpp.\n"
             "    dpct::detail::vram_note(p, std::max<size_t>(bytes, 256));\n    dev_allocs_.push_back(p);\n    return p;",
             "dmalloc vram_note")
    s = swap(s, "(void*) exp_));\n        ck(DPCT_CHECK_ERROR(sycl::free(exp_,",
             "(void*) exp_));\n        dpct::detail::vram_drop(exp_);\n        ck(DPCT_CHECK_ERROR(sycl::free(exp_,",
             "exp_ resize vram_drop")
    # WSL2 GPU watchdog (TDR): never queue gigabytes of copies, never run one multi-GB memset
    s = swap(s, '    ck(DPCT_CHECK_ERROR(cs(st_)->memcpy(dst, src, bytes)), "a copy into VRAM");\n}',
             '    ck(DPCT_CHECK_ERROR(cs(st_)->memcpy(dst, src, bytes)), "a copy into VRAM");\n'
             "    // SYCL port: CUDA's cudaMemcpyAsync from pageable memory is staged by the driver, so the host never has\n"
             "    // gigabytes queued.  Here the whole expert cache fill (12+ GB from a file mapping) went into one in-order\n"
             "    // queue and the WSL2 GPU was lost at the finish (TDR).  Drain it every 64 MiB.\n"
             "    h2d_queued_ += bytes;\n"
             "    if (h2d_queued_ >= ((size_t) 64 << 20)) {\n"
             '        ck(DPCT_CHECK_ERROR(cs(st_)->wait_and_throw()), "a copy into VRAM");\n'
             "        h2d_queued_ = 0;\n"
             "    }\n}",
             "h2d drain")
    s = swap(s, '    ck(DPCT_CHECK_ERROR(cs(st_)->memset(dst, 0, bytes)), "zeroing VRAM");\n}',
             "    // SYCL port: a multi-GB memset is one long GPU job, and the WSL2 GPU watchdog (TDR) kills it (DEVICE_LOST).\n"
             "    // Zero in 256 MiB pieces, each waited for.\n"
             "    const size_t piece = (size_t) 256 << 20;\n"
             "    for (size_t at = 0; at < bytes; at += piece) {\n"
             "        const size_t n = std::min(piece, bytes - at);\n"
             '        ck(DPCT_CHECK_ERROR(cs(st_)->memset((uint8_t*) dst + at, 0, n)), "zeroing VRAM");\n'
             '        if (bytes > piece) ck(DPCT_CHECK_ERROR(cs(st_)->wait_and_throw()), "zeroing VRAM");\n'
             "    }\n}",
             "zero_vram pieces")
    # A host buffer that dies when the caller returns: CUDA's cudaMemcpyAsync stages pageable memory, a SYCL memcpy
    # reads it when the queue gets to it (behind the queued expert copies), after the buffer was freed and reused.
    s = swap(s, "void Engine::zero_vram(void* dst, size_t bytes) {",
             "// SYCL port: CUDA's cudaMemcpyAsync from pageable memory is staged by the driver, so the host buffer may be reused\n"
             "// the moment it returns.  A SYCL memcpy reads the host buffer when the queue gets to it, behind whatever expert\n"
             "// copies are queued before it, so a buffer that dies on return (a std::vector in upload(), the loop's queries)\n"
             "// was read after it was freed and reused.  Callers with such a buffer use this.\n"
             "void Engine::h2d_copy(void* dst, const void* src, size_t bytes) {\n"
             '    ck(DPCT_CHECK_ERROR(cs(st_)->memcpy(dst, src, bytes)), "a copy into VRAM");\n'
             '    ck(DPCT_CHECK_ERROR(cs(st_)->wait_and_throw()), "a copy into VRAM");\n'
             "    h2d_queued_ = 0;\n"
             "}\n\n"
             "void Engine::zero_vram(void* dst, size_t bytes) {",
             "h2d_copy")
    s = swap(s, "        h2d(d.p, q.data(), out);\n", "        h2d_copy(d.p, q.data(), out);\n", "upload: requantized Q4_0 copy")
    s = swap(s, "        h2d(d.p, f.data(), out);\n", "        h2d_copy(d.p, f.data(), out);\n", "upload: float copy")
    s = swap(s, "    h2d(d.p, h.data(), out);\n", "    h2d_copy(d.p, h.data(), out);\n", "upload: bf16 copy")
    # Routing on the card (card_moe / moe_card) needs d_ids_, d_wts_, d_slots_, which the CUDA source allocates only
    # with MMQ tiles; this port has none, so every visit waited for the host (router wait).
    s = swap(s, "    if (mmq_) {\n        const size_t rr = n * n_used_,",
             "    if (!mmq_ && opt_.gpu_experts) {\n"
             "        // This port has no MMQ tiles (Mmq is a stub), so the block below is skipped, but the routing on the card\n"
             "        // (card_moe() / moe_card(), native_mmvq_ids) needs only these three.  Without them every visit waited for the host.\n"
             "        const size_t rr = n * n_used_;\n"
             "        d_ids_ = (int32_t*) dmalloc(rr * sizeof(int32_t), \"scratch\");\n"
             "        d_wts_ = (float*) dmalloc(rr * sizeof(float), \"scratch\");\n"
             "        d_slots_ = (void**) dmalloc((size_t) n_layer_ * n_expert_ * sizeof(void*), \"scratch\");\n"
             "    }\n"
             "    if (mmq_) {\n        const size_t rr = n * n_used_,",
             "card routing buffers without MMQ")
    # Inkling only: no other family's header or factory branch
    for n in DROPPED:
        s = s.replace('#include "%s.hpp"\n' % n, "")
    s = re.sub(r"^[ \t]*if \((?:name == \"(?:glm5-next|gemma4|deepseek4|lfm2moe|flint|kolibri1)\"(?: \|\| name == \"glm5next\")?|qwen35)\)"
               r" return std::make_unique<\w+>\(\);[^\n]*\n", "", s, flags=re.M)
    return s


def eng_hpp(s):
    s = swap(s, "    void h2d(void* dst, const void* src, size_t bytes);\n",
             "    void h2d(void* dst, const void* src, size_t bytes);\n"
             "    /// h2d() for a host buffer that dies (or is reused) when the caller returns: the copy has run when this does.\n"
             "    void h2d_copy(void* dst, const void* src, size_t bytes);\n",
             "h2d_copy declaration")
    return swap(s, "    std::vector<void*> dev_allocs_, host_allocs_;\n",
                "    std::vector<void*> dev_allocs_, host_allocs_;\n"
                "    size_t h2d_queued_ = 0;   ///< bytes queued on st_ by h2d() since its last drain (SYCL port; see h2d())\n",
                "h2d_queued_ member")


def ink_cpp(s):
    """inkling.cpp: its three uploads from a host buffer that does not outlive the call (see Engine::h2d_copy())."""
    s = swap(s, "                h2d(d, q.data(), q.size() * sizeof(float));\n",
             "                h2d_copy(d, q.data(), q.size() * sizeof(float));\n", "loop queries upload")
    s = swap(s, "            h2d(B.plane[i] + (size_t) first * row, src + (size_t) (first - from) * row, (size_t) n * row);\n",
             "            h2d_copy(B.plane[i] + (size_t) first * row, src + (size_t) (first - from) * row, (size_t) n * row);\n",
             "load_rows upload")
    return swap(s, "                h2d(B.plane[i] + at * one_row, src + slot * one_row, m * one_row);\n",
                "                h2d_copy(B.plane[i] + at * one_row, src + slot * one_row, m * one_row);\n", "load_extra upload")


edit("engine.cpp.dp.cpp", eng)
edit("engine.cpp.dp.cpp", eng_sycl)
edit("engine.hpp", eng_hpp)
edit("inkling.cpp", ink_cpp)
for n in DROPPED:
    f = P / (n + ".hpp")
    if f.exists():
        f.unlink()
        print("  dropped", f.name)
print("changed:", changed)
