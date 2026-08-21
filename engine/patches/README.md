# engine/patches

`bf16-out-prod.patch` teaches ggml-cuda's `out_prod` to accept a **BF16 `src0`**, which is what lets the DiT trainer's `--mirror bf16` keep frozen trainable-layer weights in BF16 instead of promoting the whole mirror to F32. `engine/ggml/` is a git submodule, so a submodule update silently reverts it.

`mm-backward.patch` adds an **env-gated alternative formulation** for the `MUL_MAT` activation gradient in `ggml.c`'s `ggml_compute_backward`. Upstream emits `out_prod(src0, transpose(grad))`, and ggml-cuda's `OUT_PROD` is F32-only — which forces the frozen weight to F32 and drags the *forward* `mul_mat` onto TF32 tensor cores too. With `GGML_BACKWARD_MM=1` set, the backward becomes `mul_mat(cont(transpose(src0)), grad)` instead: mathematically and shape-wise identical, but dtype-agnostic, so a BF16 weight rides real BF16 tensor cores in both directions with no dequant. Measured on an RTX 5090 at ~1.7–1.8× per layer per step. With the env var unset the emitted graph is byte-identical to upstream.

`cpy-q-occupancy.patch` fixes the **launch geometry of ggml-cuda's quant→F32 copies**. Upstream launches them as `<<<ne, 1>>>` — one CUDA *block* per *element*, one thread inside it — while the kernel body indexes `i = tid*qk` and returns on `i >= ne`. With `qk = 32` that means 31 of every 32 blocks exist only to hit the guard, and each surviving thread runs alone in a 32-lane warp; residency caps around 32 blocks/SM. The patch changes the launch shape and nothing else (the index expression already reads `blockDim.x`, so each thread gets exactly the element it would have had as a lone block — identical work, identical output), matching how the F16/F32 scalar copies in the same file already launch.

This matters because MM3 LM training on a quantized base is QLoRA-style dequantize-per-matmul: `qwen3_f32()` emits an in-graph `ggml_cast`, which becomes a `CPY q8_0 → F32` node, which is this kernel. **The failure mode is silent** — without the patch the numbers are still correct, just far slower — so `verify-hooks.ps1` Hook 9 greps for the marker.

The sibling F32→quant launches are deliberately left alone: they share the 1-thread-per-block shape but already use the correct block *count* (`ne/qk`), so they waste occupancy without wasting blocks, and they are an inference path (quantized KV cache) rather than the training path this was measured against.

`ace-train`'s `--bwd <outprod|mm>` sets that env var; the Training Studio defaults both trainers to `mm`.

`cpy-q-occupancy.patch` fixes the **launch geometry of ggml-cuda's quant→F32 copies**. Upstream launches them as `<<<ne, 1>>>` — one CUDA *block* per *element*, one thread inside it — while the kernel body indexes `i = tid*qk` and returns on `i >= ne`. With `qk = 32` that means 31 of every 32 blocks exist only to hit the guard, and each surviving thread runs alone in a 32-lane warp; residency caps around 32 blocks/SM. The patch changes the launch shape and nothing else (the index expression already reads `blockDim.x`, so each thread gets exactly the element it would have had as a lone block — identical work, identical output), matching how the F16/F32 scalar copies in the same file already launch. Measured on a 5090, MM3 8.6B q8_0 training: **11.1 → 3.75 s/step**, i.e. parity with f16.

`quant-cpy-kquant.patch` teaches the same `CPY` dispatch to reach the **generic quant→F32 converter ggml already ships**. Upstream hand-writes one quant→F32 copy per type and has written five — `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`. Every K-quant, every IQ type, MXFP4 and NVFP4 have none, so `ggml_cast(w, F32)` on such a weight is refused by `supports_op` and the graph falls off the GPU. Those dequantizers are not missing: `convert.cu` exposes `ggml_get_to_fp32_cuda()` covering `Q2_K`–`Q6_K`, `IQ1`–`IQ4`, MXFP4 and NVFP4, and it is the same converter the `mul_mat` path uses on every token of every quantized inference. Only the CPY dispatch never learned to ask for it.

So that patch adds **one fallback branch rather than twenty kernels**, last in the else-if chain — every type upstream already handles keeps its existing kernel byte for byte, and only combinations that would have hit `GGML_ABORT` reach the new path. It requires contiguous src and dst of the same shape, which is exactly what `ggml_cast()` produces and the only form the converters accept. `cpy.cuh` exports `ggml_cuda_cpy_quant_to_f32_supported()` so the dispatch and `supports_op()` answer that question from one place instead of two lists that drift.

Together these two are what make LM training possible on a card smaller than 32 GB, and both matter for the same reason — **QLoRA-style dequantize-per-matmul**: the frozen weight enters the graph through `ggml_cast`, so the backward only ever sees the cast's F32 output. Measured, MM3 8.6B, rank 256 / 1500 frames, peak VRAM, all within ~5 % of the same step time:

| base | size | peak VRAM | 1st-step loss vs f16 |
|---|---|---|---|
| f16 | 16.0 GB | 31.4 GB | reference |
| q8_0 | 8.5 GB | 22.6 GB | +0.02 % |
| Q6_K | 6.6 GB | 20.7 GB | +0.3 % |
| Q4_K_M | 5.1 GB | 19.2 GB | +0.8 % |
| MXFP4 | 5.1 GB | 19.1 GB | +2.7 % |
| Q2_K | 3.4 GB | 17.5 GB | +14.3 % — too lossy to train against |

Base quant alone does not reach a 12 GB card; **LoRA rank is the other lever**. Q4_K_M peaks: r256/1500 19.2 GB, r64/1500 13.2 GB, r32/750 11.1 GB, r16/500 10.2 GB.

`cudagraph-log.patch` adds an **env-gated one-line-per-compute decision trace**
(`GGML_CUDA_GRAPH_LOG=1`) to `ggml_backend_cuda_graph_compute` — key, uid,
node count, compatibility, warmup state, and a line whenever node properties
change. Zero cost when unset. This is a *diagnostic*, not a fix: it is the
tool that established that CUDA graph capture is already active for every MM3
graph (LM decode −29 %, depth −14 % vs `GGML_CUDA_DISABLE_GRAPHS=1`) and that
an apparent post-model-swap "graph thrash" was actually the select-model API
resetting an omitted LM role to auto/f16. **Losing this patch is benign** —
no verify-hook guards it; reapply it when you next need the trace.

Reapply from the repo root — **apply all of them**. Two of them now touch `cpy.cu`, so they are no longer file-disjoint, but their hunks do not overlap and **both orders were verified to apply cleanly**; a full pristine→apply-all replay was checked to reproduce the tested bytes exactly. The glob below is what CI runs:

```sh
for p in engine/patches/*.patch; do git apply --verbose "$p"; done
```

Verify they are still in place: `powershell -File engine\verify-hooks.ps1` (Hook 7 greps `out-prod.cu`, Hook 8 `ggml.c`, Hook 9 `cpy.cu`, Hook 10 `cpy.cuh`, each for its HOT-Step marker comment). **Hooks 9 and 10 matter most, because both failures are silent**: losing 9 leaves the numbers right and only the clock wrong, and losing 10 makes every sub-`q8_0` base vanish from the trainer rather than error. CI reapplies them in the "Apply engine patches" step of every build job, using the same glob loop.
