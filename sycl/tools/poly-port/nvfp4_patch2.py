#!/usr/bin/env python3
"""NVFP4: look the E2M1 codes up in registers instead of the dpct::global_memory kvalues_fp4."""
import os
F = os.path.expanduser("~/PolyStrata/sycl/src/kernels/cuda/iq_kernels.dp.cpp")
t = open(F).read()
if "fp4_pair" not in t:
    helper = r'''// = get_int_from_table_16(aux, kvalues_fp4) without the table in global memory: the low nibbles of aux's four bytes
// (x) and the high nibbles (y), each looked up in the E2M1 values doubled {0,1,2,3,4,6,8,12, then negated}.
__dpct_inline__ sycl::int2 fp4_pair(int aux) {
    const uint32_t a = (uint32_t) aux;
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const uint32_t byte = (a >> (8 * b)) & 0xFF;
        const uint32_t cl = byte & 0xF, ch = byte >> 4;
        const int ml = cl & 7, mh = ch & 7;
        const int vl = ml < 5 ? ml : (ml == 5 ? 6 : (ml == 6 ? 8 : 12));
        const int vh = mh < 5 ? mh : (mh == 5 ? 6 : (mh == 6 ? 8 : 12));
        const int sl = (cl & 8) ? -vl : vl, sh = (ch & 8) ? -vh : vh;
        lo |= ((uint32_t) (uint8_t) (int8_t) sl) << (8 * b);
        hi |= ((uint32_t) (uint8_t) (int8_t) sh) << (8 * b);
    }
    return sycl::int2((int) lo, (int) hi);
}
'''
    mark = "// NVFP4 (ModelOpt): 64 values"
    assert mark in t
    t = t.replace(mark, helper + mark, 1)
    t = t.replace("get_int_from_table_16(get_int_b4(bq4->qs, iqs0), kvalues_fp4)", "fp4_pair(get_int_b4(bq4->qs, iqs0))")
    t = t.replace("get_int_from_table_16(get_int_b4(bq4->qs, iqs1), kvalues_fp4)", "fp4_pair(get_int_b4(bq4->qs, iqs1))")
    open(F, "w").write(t)
print("ok")
