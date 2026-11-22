# strata-poly on SYCL (Intel Arc): work in progress

Status: **work in progress, not a pull request.** A SYCL port of the poly engine, built to run Inkling 2 Mini on an Intel Arc Pro B70. It sits on top of PR #200 (Inkling 2 Mini), so it only makes sense with that code.

## Measured (one machine, one run each, no reference comparison yet)
- Hardware: Intel Arc Pro B70 32 GB under WSL2 (Ubuntu 24.04, `/dev/dxg`), 64 GB RAM, oneAPI icpx, AOT `intel_gpu_bmg_g31`.
- Model: `Inkling-2-Mini-UD-Q3_K_XL.gguf`, `--tokens 1,2,3,4,5 --max-new 32 --max-context 2048`.
- Result: all 3072 routed experts in VRAM (11.4 GiB), 17.2 tok/s decode, 10.5 tok/s prompt, load 90 s, clean exit.
- Check: the five prompt tokens are the ASCII characters `"#$%&`; the model continues `'()*+,-./012345678 9:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[`.
- **Not done:** no logit comparison with llama.cpp or the CUDA build, no kernel tests, no CUDA/HIP build of the touched shared files, no other model family, no drafter. A successful build and one plausible output are not hardware correctness.

## How the commits are split

| # | Commit | Depends on #200 / Inkling? | Main-stream candidate? |
|---|--------|----------------------------|------------------------|
| 1 | SYCL NVFP4 (type 40) native_mmvq | No (the plain `strata` SYCL build has no NVFP4 experts either) | Yes |
| 2 | SYCL `vram_tracker` for runtimes with no free-memory query (WSL2) | No; off unless enabled | Yes |
| 3 | Shared poly engine, dpct output + fixups (`engine`, `kernels`, `kernels_dsv4`, `spill`, `main`, shims, headers) | Poly engine, not Inkling-specific | After 1-2; needs CUDA/HIP check of anything shared |
| 4 | Inkling 2 only (`kernels_inkling`, `inkling`) | Yes | Only with #200 |
| 5 | CMake target `strata-poly` (`-DSTRATA_SYCL_POLY=ON`) | Yes | With 3 and 4 |
| 6 | Migration scripts and these notes | No | Optional |

Commits 3 and 4 are source only and do not build alone; commit 5 wires them in.

## Fixes found on the way
- dpct turned `cudaFreeHost` into `free()` on `sycl::malloc_host` memory; now `sycl::free(p, queue)` (exit crash).
- WSL2 has no free-VRAM query (`ext_intel_free_memory` unsupported), so the expert cache was sized to 0: `vram_tracker`.
- WSL2 GPU watchdog: one multi-GB fill (the whole expert pool) and ~10,000 queued copies gave `UR_RESULT_ERROR_DEVICE_LOST`. `Engine::zero_vram` now fills in 256 MiB pieces and `Engine::h2d` drains the queue every 64 MiB. Found with `SYCL_UR_TRACE=2`.
- NVFP4 mmvq: 64 values per 36-byte block (4 UE4M3 scale bytes + 32 bytes of E2M1 nibbles). 20 of 24 MoE layers in this GGUF are NVFP4, so without it only 512 experts could sit on the card (2.7 tok/s).

## Known gaps
- Stubs (the engine falls back to launching matrices one by one): `native_mmvq_slots/_ids/_jobs`, `Mmq`, `native_add_rms_norm_q8_1` residual variant (throws). Tensor-core attention and VMM are off. Prompt speed is the weak spot.
- The engine factory is trimmed to Inkling only.
- `sycl/include` here is older than main's, hence the shims above.
- `poly_fixups.py` reproduces every hand change to the dpct output of `engine.cpp`/`engine.hpp` (the `h2d`/`zero_vram` TDR guards, `vram_note`/`vram_drop`, `h2d_queued_`, the Inkling-only factory and the seven dropped model headers); checked byte-for-byte against the committed files, and idempotent.
- Not rebased onto the `engine/` layout of current main.

## Reproduce
Scripts in this folder (paths are the author's: WSL home, `/mnt/h/dev/polystrata-work`). `poly_all.sh` runs dpct and the fixups, `poly_aot.sh` compiles the kernels AOT, `poly_run2.sh <ids> <n> "<args>"` runs the model, `poly_trace2.sh` runs with the Level Zero trace.
Build: `cmake -DSTRATA_SYCL_POLY=ON ...`, then `cmake --build build-sycl-aot --target strata-poly` (about 6-10 minutes to relink).
