#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

// Native GGUF Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL
// and IQ4_XS / CUDA Q8_1 adapters, pinned to llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, sm_120 generic MMVQ.
// All pointers are device pointers, at least 4-byte aligned, with no overlap.
// All calls enqueue on the explicit non-null CUDA stream; no allocation or wait.
// The translation unit must use --use_fast_math, as the pinned CUDA oracle does.
//
// Shapes use GGUF order: n_in is the contiguous reduction dimension, n_out is
// the weight row count, and ncols is the activation column/token count, 1..8
// (plan v0.3 P3). Columns are contiguous: activation column j is x + j * n_in
// (its Q8_1 blocks at j * n_in / 32) and output column j is y + j * n_out. See
// native_mmvq_set_multi_exact for how ncols > 1 relates to ncols == 1.
// n_in must be a positive multiple of 32 for quantization/Q4_0/Q5_0/Q8_0/IQ4_NL,
// 64 for Q2_0, or 256 for the other formats. n_out must be positive. Weights remain
// unmodified row-major GGUF blocks: Q3_K=110, Q4_K=144, Q5_K=176, Q6_K=210 and
// IQ4_XS=136 bytes per 256 elements; Q2_0 is 18 bytes per 64 elements.
// Q4_0/IQ4_NL=18, Q5_0=22, Q8_0=34 bytes per 32 elements.
// Q8_1 scratch has 36 bytes per 32 elements, with no extra row padding here.
// Input floats must be finite, and their block scales/sums representable in FP16.
std::size_t native_q8_1_bytes(int n_in, int ncols = 1);

// Layout for ncols > 1. false: llama.cpp's generic multi-column table (upstream), equal to ncols == 1 to
// float rounding, speed not yet measured. true (default): the ncols == 1 layout, every column bitwise equal to a
// single-column call. Set before
// graph capture; captured graphs keep the kernels they captured.
void native_mmvq_set_multi_exact(bool exact);
void native_mmvq_set_q6k_wide(bool on);   // SYCL port: the 16-byte-load Q6_K kernel (default on)
void native_q6_k_mmvq_stride224(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                                void* stream);   // SYCL port: experiment, Q6_K blocks at a 224-byte stride
bool native_mmvq_multi_exact();

// One quantization may serve multiple weight matrices sharing the same input.
// Q8_1 stores FP16 scale and FP16 warp sum of the ORIGINAL float inputs; it does
// not reconstruct that sum from the quantized integers.
void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols,
                          void* stream);
/// SYCL port (poly_shims.cpp): rms norm + Q8_1 quantize of its output; `add` must be null.
void native_add_rms_norm_q8_1(float* x, const float* add, const float* w, float* y, void* y_q8_1, int n, int rows,
                              float eps, int threads, void* stream);


void native_swiglu_quantize_q8_1(const float* gate, const float* up, void* x_q8_1,
                                 int n_in, int ncols, void* stream);

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

// Convenience composition: caller owns scratch sized by native_q8_1_bytes.
void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

/// STRATA_Q8_PACKED=1 (opt-in): a lossless load-time repack of a Q8_0 matrix into a qs plane (n_out x n_in int8)
/// followed by a d plane (n_out x n_in / 32 fp16), the same bytes (34 per 32 values). A registered matrix's
/// native_q8_0_mmvq calls (keyed by its GGUF-layout device pointer, which stays valid for the prompt path) read the
/// packed copy instead; every output is bitwise equal to the GGUF-layout kernels.
bool native_q8_0_packed_enabled();
bool native_q8_0_packed_eligible(int n_in, int n_out);
void native_q8_0_pack_host(const void* gguf_blocks, void* out, int n_in, int n_out);
void native_q8_0_packed_register(const void* gguf_weights, const void* packed, int n_in, int n_out);
void native_q8_0_packed_unregister(const void* gguf_weights);

/// STRATA_Q6_PACKED=1 (opt-in): a packed copy of a Q6_K matrix (the output heads) - the same bytes as ql / qh /
/// scales / d planes - that native_q6_k_mmvq calls on `weights` read instead (bitwise equal outputs). The copy is
/// owned here: native_q6_k_pack builds it from the device matrix (false: not eligible or failed; nothing changes),
/// native_q6_k_unpack frees it. STRATA_Q6P_SELFTEST=1 checks it bitwise and times it at load.
bool native_q6_k_packed_enabled();
bool native_q6_k_pack(const void* weights, int n_in, int n_out, const char* what);
void native_q6_k_unpack(const void* weights);

/// STRATA_Q6_PLANAR (SYCL port; =0 turns it off): the Q6_K output heads stored as planes - ql (128 B), qh (64 B), the scales
/// (16 B, each lane's pair adjacent), d (2 B) of every 210-byte block - so the decode kernel's weight loads are single
/// aligned 16-byte loads (the GGUF layout is 2-byte aligned: two loads and a shift per run). Same bytes, ~14% faster, outputs
/// bitwise equal. native_q6_k_planar_adopt converts a GGUF-layout matrix (the old allocation is freed, the returned pointer
/// replaces it and is what native_q6_k_mmvq is called with; free it like the old one, after native_q6_k_unpack);
/// native_q6_k_planar_gather takes rows of a planar matrix as a new planar one (nullptr: `planar` is not one).
bool native_q6_k_planar_enabled();
bool native_q6_k_is_planar(const void* weights);
void* native_q6_k_planar_adopt(void* weights, int n_in, int n_out, const char* what);
void* native_q6_k_planar_gather(const void* planar, int n_in, const int32_t* rows_dev, int64_t n_sel, const char* what);

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

// Storage/dispatch helpers take stable GGML type IDs, avoiding a ggml runtime
// dependency in the engine: Q4_0=2, Q5_0=6, Q8_0=8, Q3_K=11, Q4_K=12, Q5_K=13,
// Q6_K=14, IQ4_NL=20, IQ4_XS=23, Q2_0=42. Unsupported IDs throw in the byte-count
// and launch helpers; only the
// capability query returns false.
bool native_mmvq_supported(int ggml_type) noexcept;
std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out);
void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream);
/// S26 STRATA_LFUSE: native_mmvq(w1 -> y1) and native_mmvq(w2 -> y2), same type / shape / input, in ONE launch,
/// bitwise the two calls. Returns false (nothing launched) where that is not the case (Q8_0, 2-8 columns only).
bool native_mmvq_pair(int ggml_type, const void* w1, const void* w2, const void* x_q8_1, float* y1, float* y2,
                      int n_in, int n_out, int ncols, void* stream);


// SYCL port: declarations as upstream; poly_shims.cpp defines them as 'no kernel' (callers fall back).
// Up to kMmvqSlotsMax (32) matrices of one format and shape in one launch: matrix i times its own Q8_1 column x[i]
// into y[i], as native_mmvq(type, w[i], x[i], y[i], n_in, n_out, 1, stream) would each.  A launch costs the calling thread more
// than a small product costs the card, which is what this saves.  False, with nothing launched: there is no
// kernel for it (a format native_mmvq() does not take, or a HIP build), and the caller launches them one by one.
constexpr int kMmvqSlotsMax = 32;   // 8 before: strata-poly's cached experts go 32 a launch where the wide path is on
struct MmvqSlots {
    const void* w[kMmvqSlotsMax];
    const void* x[kMmvqSlotsMax];
    float* y[kMmvqSlotsMax];
    // The same bits through other kernels: native_mmvq.cu's formats a warp a row instead of a block of WARPS warps
    // (lane l with the terms of the block's threads l, 32 + l, ..., added in the block's order), where the block's
    // threads had uneven work; the IQ formats whose grid iq_kernels.cu stages in shared memory (IQ1_S, IQ1_M, IQ2_XXS,
    // IQ2_XS, IQ2_S), 8 rows a warp with the grid staged once.  The other IQ formats' launches are as without it.
    // Faster for Inkling 2's experts on an RTX 2070 Super; measured on CUDA only.  It takes precedence over
    // native_mmvq_slots_wide() (on by default on HIP); Metal takes no notice.
    bool lanes = false;
};
bool native_mmvq_slots(int ggml_type, const MmvqSlots& slots, int count, int n_in, int n_out, void* stream);
// Whether the slots launches take a warp per row group where a row's blocks fit one warp (an expert's IQ4_XS down
// projection of 512 values), and strata-poly gives them its cached experts 32 at a time instead of 8.  The same bits
// either way.  On by default on HIP (measured on an RX 7800 XT), off by default on CUDA (not measured there);
// STRATA_SLOTS_WARP=0 turns it off, =1 on.
bool native_mmvq_slots_wide();
// native_mmvq_slots with the matrices the card picked, so that the launch needs nothing of the host (a decode
// step's routed experts when every one of them is in VRAM): matrix j is table[ids[j]] + offset bytes, times the
// Q8_1 column x + (j / x_every) * x_stride bytes, into y + j * y_stride floats.  count may pass 8.  False, with
// nothing launched: there is no kernel for the format (native_mmvq_ids_supported() says so beforehand).
struct MmvqIds {
    const void* const* table;
    const int32_t* ids;
    std::size_t offset;
    const void* x;
    std::size_t x_stride;
    int x_every;
    float* y;
    std::size_t y_stride;
    bool lanes = false;   // as MmvqSlots::lanes
};
bool native_mmvq_ids_supported(int ggml_type);
bool native_mmvq_ids(int ggml_type, const MmvqIds& ids, int count, int n_in, int n_out, void* stream);
// Several matrices of one format and input width in one launch (an attention's projections of the same input), each
// with its own rows, Q8_1 columns and output: native_mmvq(type, w[i], x[i], y[i], n_in, n_out[i], ncols, stream)
// for each, bit for bit.  Q8_0, Q4_K, Q5_K and Q6_K.  False, with nothing launched: another format, a repacked
// matrix, or a layout the switches above chose whose bits these are not; the caller launches them one by one.
constexpr int kMmvqJobsMax = 8;
struct MmvqJobs {
    const void* w[kMmvqJobsMax];
    const void* x[kMmvqJobsMax];
    float* y[kMmvqJobsMax];
    int n_out[kMmvqJobsMax];
};
bool native_mmvq_jobs(int ggml_type, const MmvqJobs& jobs, int count, int n_in, int ncols, void* stream);

} // namespace strata::kernels
