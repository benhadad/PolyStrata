# strata-poly on SYCL (Intel Arc): work in progress

Status: **work in progress, not a pull request.** A SYCL port of the poly engine, built to run Inkling 2 Mini on an Intel Arc Pro B70. It was built on the Inkling 2 Mini code of upstream PR #200. That PR was closed without merging on 2026-11-23; its closing comment says Inkling 2 Mini is in upstream `main` (v0.2.20-poly.1) and that the later work from the PR's branch came in as separate PRs. So upstream's Inkling code has moved on from the version this port migrated, and this branch has not been moved onto it (not yet checked how far the two differ). The branch's base is an older commit, not current `main`. Testers welcome: see "Test it yourself" below.

## Measured (one machine; speed is repeatable, correctness is not yet checked)
- Hardware: Intel Arc Pro B70 32 GB under WSL2 (Ubuntu 24.04, `/dev/dxg`), 64 GB RAM. oneAPI `icpx` 2026.1.1, level-zero 1.26.0, compute runtime `libze-intel-gpu1` 25.48.36300.8, IGC 2.24.8; AOT `intel_gpu_bmg_g31`.
- Model: `Inkling-2-Mini-UD-Q3_K_XL.gguf` (about 24 GB; 20 of its 24 MoE layers are NVFP4).
- Build: this branch at commit `8c9babfc1` (the binary was built from sources identical to it).
- Run: `--tokens 1,2,3,4,5,6,7,8 --max-new 64 --max-context 2048`, default settings (greedy, `--lookup` 3). The token IDs are arbitrary, not text.
- Result, two batches of runs on the same binary, all 3072 routed experts in VRAM (11.4 GiB), router wait 0.3 to 0.8 ms/token:
  - Batch 1: 15 runs started 5 s apart, 14 completed: decode 34.5 to 41.3 tok/s (mean 38.8), prompt 13.7 to 20.3 tok/s for the 8-token prompt, load 68 to 104 s.
  - Batch 2: 30 runs started 3 to 4 minutes apart (60 s idle between runs), all 30 completed: decode 31.3 to 38.6 tok/s (mean 35.9). Prompt speed and load time were not collected for this batch.
  - The two batches differ by about 3 tok/s in the mean and we do not know why (the gap between runs is one difference; GPU clocks and temperature were not recorded). Expect roughly 31 to 41 tok/s on this machine.
  - All 44 completed runs printed the same 64 tokens.
- One of the 15 runs in batch 1 exited at startup with `Level-Zero error:78000013` (`ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION`) in the oneMKL warm-up product (`kernels.dp.cpp`, `blas_create`). Not explained; it did not recur in batch 2's 30 runs. The one `DEVICE_LOST` we saw during evaluation (1 of 11 runs, before `h2d_copy`) has not recurred in the 62 runs since (8 + 6 + 3 + 15 + 30 on the fixed build).
- Earlier figures (17.2, 19.97 and 22.97 tok/s) were single runs on other days, before the expert-routing "card" path worked, so they are not comparable with the figures above.
- **Not validated:** no comparison of logits or tokens with llama.cpp or the CUDA build, so "the same every time" is not "correct". One prompt, one model, one machine, WSL2 only. No CUDA/HIP build of the touched shared files.

## How the commits are split

| # | Commit | Depends on #200 / Inkling? | Main-stream candidate? |
|---|--------|----------------------------|------------------------|
| 1 | SYCL NVFP4 (type 40) native_mmvq, with a CPU-reference test | No. Overlaps upstream: `main`'s SYCL `iq_kernels.dp.cpp` has NVFP4 (type 40) too (seen 2026-11-23, not compared with ours or run) | Only if it still adds something over upstream's |
| 2 | SYCL `vram_tracker` for runtimes with no free-memory query (WSL2) | No; off unless enabled | Yes |
| 3 | Shared poly engine, dpct output + fixups (`engine`, `kernels`, `kernels_dsv4`, `spill`, `main`, shims, headers) | Poly engine, not Inkling-specific | After 1-2; needs CUDA/HIP check of anything shared |
| 4 | `native_mmvq_slots/_ids/_jobs` and `iq_mmvq_slots/_ids`, with a parity test | No | After 3 |
| 5 | Allocate the card-routing buffers without the MMQ tiles | No | After 3 |
| 6 | `Engine::h2d_copy`: copy and wait, for host buffers that die on return | No | After 3 |
| 7 | Inkling 2 only (`kernels_inkling`, `inkling`) | Yes | Only after being moved onto upstream's current Inkling code |
| 8 | CMake target `strata-poly` (`-DSTRATA_SYCL_POLY=ON`) | Yes | With 3 and 7 |
| 9 | Migration scripts and these notes | No | Optional |

Commits 3 and 7 are source only and do not build alone; commit 8 wires them in.

## Fixes found on the way
- dpct turned `cudaFreeHost` into `free()` on `sycl::malloc_host` memory; now `sycl::free(p, queue)` (exit crash).
- WSL2 has no free-VRAM query (`ext_intel_free_memory` unsupported), so the expert cache was sized to 0: `vram_tracker`.
- WSL2 GPU watchdog: one multi-GB fill (the whole expert pool) and ~10,000 queued copies gave `UR_RESULT_ERROR_DEVICE_LOST`. `Engine::zero_vram` now fills in 256 MiB pieces and `Engine::h2d` drains the queue every 64 MiB. Found with `SYCL_UR_TRACE=2`.
- NVFP4 mmvq: 64 values per 36-byte block (4 UE4M3 scale bytes + 32 bytes of E2M1 nibbles). 20 of 24 MoE layers in this GGUF are NVFP4, so without it only 512 experts could sit on the card (2.7 tok/s).
- The card-routing path was dead: its three buffers were only allocated when the MMQ tiles exist, and `Mmq` is a stub here, so every layer visit took the host-routed path (a device-to-host router copy and a wait). Allocating them fixed it: router wait fell from 8.8-10.1 to 0.3-0.7 ms/token.
- Output differed from run to run. Cause: `Engine::h2d` is an asynchronous SYCL copy that reads the host buffer when the queue reaches it, and four callers passed buffers freed on return (CUDA stages such copies at call time). Those callers now use `Engine::h2d_copy`. Before: six runs gave three different token lists; after: every run gave one list. That the loop's message queries were the buffer read late is inferred from where the layer checksums first differed, not inspected.

## Known gaps
- Stubs: `Mmq` and the `native_add_rms_norm_q8_1` residual variant (throws). Tensor-core attention and VMM are off. Prompt speed is the weak spot.
- Prompts of more than 8 tokens fail with `native MMVQ requires 1 <= ncols <= 8` (the stubbed `Mmq` falls back to `native_mmvq`). Probably not new; not checked against the old binary.
- `native_mmvq_jobs` (the attention projections) has no test; the Inkling kernels have only `inkling_gemv`.
- The engine factory is trimmed to Inkling only.
- `sycl/include` here is older than main's, hence the shims above.
- `poly_fixups.py` reproduces the hand changes to `engine.cpp`/`engine.hpp`, the `inkling.cpp` calls and the card-buffer change (checked byte-for-byte, idempotent), but not the slots/ids/jobs kernels.
- Not rebased onto the `engine/` layout of current main, and not moved onto upstream's current Inkling code (see the status above).

## Test it yourself
You need an Intel Arc GPU, oneAPI `icpx` with the level-zero runtime, and the model file. We tested only a B70 (`bmg-g31`) on WSL2; other Arc cards need a different `STRATA_SYCL_AOT` value (an empty one compiles to SPIR-V and JITs at run time). These values are from our build cache; a fresh clone configured exactly like this has not been tried.

```
git clone -b sycl-inkling-wip https://github.com/benhadad/PolyStrata.git
cd PolyStrata
source /opt/intel/oneapi/setvars.sh
cmake -S sycl -B build-sycl-aot -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx \
      -DSTRATA_SYCL_POLY=ON -DSTRATA_SYCL_AOT=bmg-g31
cmake --build build-sycl-aot -j 5 --target strata-poly     # about 6-10 minutes to relink
export ONEAPI_DEVICE_SELECTOR=level_zero:0
./build-sycl-aot/strata-poly --model /path/to/Inkling-2-Mini-UD-Q3_K_XL.gguf \
      --tokens 1,2,3,4,5,6,7,8 --max-new 64 --max-context 2048
```

The run prints a line like `prompt 8 tokens 17.6 tok/s, generated 64 tokens 41.17 tok/s ...`. Our 14 completed runs all printed these 64 tokens:

```
9,10,11,12,13,14,77863,81999,97293,24,25,26,27,145028,30,31,146104,146842,146321,146511,42,146212,145914,47,146232,145866,146159,54,147151,58,59,60,61,62,63,66345,90,91,92,93,1,68014,2,220,16,13,220,142702,220,27788,27057,220,27364,78926,198,2,220,142702,149943,25849,220,143436,28511,27066
```

A different GPU or driver may legitimately change floating-point rounding and so the tokens; a mismatch is a result to report, not necessarily a bug. We have not checked this token list against llama.cpp.

Useful to report: GPU and driver (`apt list --installed 2>/dev/null | grep libze-intel-gpu1`), oneAPI version, native Linux or WSL2, the model file's SHA-256 (ours was not recorded), the decode and prompt tok/s over several runs, whether the 64 tokens match, and the first error lines if a run fails. For `DEVICE_LOST` or other Level Zero errors, add the tail of a run with `SYCL_UR_TRACE=2`. Please do not compare these numbers with other engines' numbers: one machine, one prompt.

## Reproduce the port
Scripts in this folder (paths are the author's: WSL home, `/mnt/h/dev/polystrata-work`). `poly_all.sh` runs dpct and the fixups, `poly_aot.sh` compiles the kernels AOT, `poly_run2.sh <ids> <n> "<args>"` runs the model, `poly_trace2.sh` runs with the Level Zero trace.
