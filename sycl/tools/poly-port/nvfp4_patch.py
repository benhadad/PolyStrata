#!/usr/bin/env python3
"""SYCL port: NVFP4 (GGML type 40) on the native mmvq path (decode and prompt rows), mirrored from the CUDA
vec_dot_nvfp4_q8_1 in src/kernels/cuda/iq_kernels.cu.  Idempotent."""
import os
R = os.path.expanduser("~/PolyStrata/sycl/src/kernels/cuda/")
F = R + "iq_kernels.dp.cpp"
N = R + "native_mmvq.dp.cpp"

t = open(F).read()
if "vec_dot_nvfp4_q8_1" not in t:
    fn = r'''// NVFP4 (ModelOpt): 64 values = 4 sub-blocks of 16, one UE4M3 scale each, E2M1 codes split-half like Q4_0.
// kvalues_fp4 holds the E2M1 values doubled, so the scale is halved here (ggml's ue4m3_to_fp32, in software).
__dpct_inline__ float ue4m3_to_f32_half(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const uint32_t exp = (x >> 3) & 0xF, man = x & 0x7;
    return exp == 0 ? (float) man * 0.0009765625f : sycl::bit_cast<float>(((exp + 119u) << 23) | (man << 20));
}
// One call = two sub-blocks (32 values); `bq8_1` is the block's first q8_1 block (two per 64 values).
__dpct_inline__ float vec_dot_nvfp4_q8_1(const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs) {
    const block_nvfp4* bq4 = (const block_nvfp4*) vbq + kbx;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int iqs0 = iqs + 2 * i, iqs1 = iqs0 + 1, is = iqs0 >> 1;
        const sycl::int2 v0 = get_int_from_table_16(get_int_b4(bq4->qs, iqs0), kvalues_fp4);
        const sycl::int2 v1 = get_int_from_table_16(get_int_b4(bq4->qs, iqs1), kvalues_fp4);
        const block_q8_1* bq8 = bq8_1 + (is >> 1);
        const int i8 = (is & 1) << 2;
        int sumi = ggml_cuda_dp4a(v0.x(), get_int_b4(bq8->qs, i8 + 0), 0);
        sumi = ggml_cuda_dp4a(v0.y(), get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi = ggml_cuda_dp4a(v1.x(), get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi = ggml_cuda_dp4a(v1.y(), get_int_b4(bq8->qs, i8 + 3), sumi);
        sum += ue4m3_to_f32_half(bq4->d[is]) * bq8->ds[0] * (float) sumi;
    }
    return sum;
}

'''
    mark = "// ---------------------------------------------------------------- the formats\n"
    assert mark in t
    t = t.replace(mark, fn + mark, 1)
    f42 = "template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;"
    assert f42 in t
    t = t.replace(f42, "template<> struct Fmt<40> { static constexpr int qk = 64, ipb = 2, step = 4;   // NVFP4: 2 calls of 32 values\n"
                       "    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_nvfp4_q8_1(v, y, kbx, iqs); } };\n" + f42, 1)
    t = t.replace("#define STRATA_MMVQ_FMTS(X) X(16)", "#define STRATA_MMVQ_FMTS(X) X(40) X(16)", 1)
    t = t.replace("        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);",
                  "        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);\n        case 40: return (size_t) (n / 64) * sizeof(block_nvfp4);", 1)
    open(F, "w").write(t)

n = open(N).read()
if "ggml_type == 40" not in n:
    n = n.replace("ggml_type == 23 || ggml_type == 42 || ggml_type == 16", "ggml_type == 23 || ggml_type == 42 || ggml_type == 40 || ggml_type == 16", 1)
    n = n.replace("    case 42: block_elems = 64; block_bytes = 18; break;",
                  "    case 42: block_elems = 64; block_bytes = 18; break;\n    case 40: block_elems = 64; block_bytes = 36; break;", 1)
    n = n.replace("    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;",
                  "    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;\n    case 40: iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;", 1)
    open(N, "w").write(n)
print("ok")
