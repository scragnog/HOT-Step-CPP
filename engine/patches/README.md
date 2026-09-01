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

`f16-f32-accumulate.patch` makes ggml-cuda's batched cuBLAS GEMM **accumulate and write F32 for an F16 `src0`**, which upstream only does on Volta, RDNA4 and CDNA. Everywhere else F16 lands on `CUBLAS_COMPUTE_16F`: the dot product accumulates in half precision *and* `dst` is written in half precision, so a partial sum past 65504 becomes `+inf` and everything downstream of it NaN. Nothing reports it — the GEMM succeeds and returns garbage. BF16 and F32 have always taken the F32 path; F16 was the odd one out.

This is not theoretical. The MiniMax-Music3 LM is Qwen3-8B, with K of 4096 and 12288, and it clears that ceiling as soon as an LM LoRA/LoKr shifts the residual stream up — **and only then**, which is why it presented for months as "adapters are broken on f16, use q8" rather than as a matmul bug. Measured on an RTX 5090 (cc 12.0), same prompt, same seed, same adapter, one variable:

| LM base | non-finite candidate logits |
|---|---|
| f16 | one whole CFG row dead on **every** step |
| bf16 | none — already took the F32 branch |
| q8_0 | none — never reaches cuBLAS |

`bf16` is the control that settles it: same value range as `f16`, same graph, and the only difference is which compute type it lands on. Cost of the fix, MM3 LM decode on a 5090: 11.03 → 11.59 ms/step (+5 %), landing exactly on bf16's 11.59 — which is what you would expect once they share a path. FP16 inputs with an FP32 accumulator run at full tensor-core rate from Ampere on.

**Losing this patch is silent and expensive** — no error, no crash, just noise instead of music on any f16 render with an adapter — so `verify-hooks.ps1` Hook 11 greps for the marker. The two guards in `mm3-ar-loop.h` (refuse a plan whose logits are wholesale non-finite) and `mm3-lm-adapter.h` (refuse a checkpoint whose factors would overflow the f16 store) are the second line of defence, and they live in the main repo rather than here.

`flash-attn-train.patch` adds two fused attention ops — `GGML_OP_FLASH_ATTN_TRAIN` and `GGML_OP_FLASH_ATTN_TRAIN_BACK`, appended at the tail of the op enum so every existing value stays put — with a CPU reference implementation, CUDA kernels in two new self-contained files (`ggml-cuda/fattn-train.{cu,cuh}`; no existing `fattn-*` file is touched), and an autodiff case so `ggml_build_backward_expand` emits the backward on its own.

What it is for is the DiT trainer's retained softmax. `dit_attn_f32` materialises a `[S_kv, S, Nh, B]` f32 softmax at every attention site and the backward keeps it alive, so attention memory is quadratic in the crop. At `S = 3000` across 32 layers that is 36.9 GB of self-attention softmax plus 9.4 GB of cross-attention softmax — the arithmetic behind `docs/TRAINING.md`'s "full-song training is impossible" line. The fused forward emits one packed tensor holding `O` and the log-sum-exp and nothing of size S²; the fused backward recomputes tiles from Q/K/LSE using the row identity `D_i = rowsum(dO∘O)`. Both trainer attention sites go through it, so both quadratic terms go.

The ops are deliberately generic rather than trainer-shaped — arbitrary additive F16 mask with `soft_max_ext`'s broadcast rule, GQA handled natively (`Nkv < Nh` with no pre-expansion), `S_kv != S`, `B >= 1`. That is what makes adoption by the other trainers call-site wiring rather than kernel work.

Three properties are load-bearing, and all three are deliberate:

- **A fully-masked query row is defined, not inherited.** `ggml_soft_max_ext` gives NaN there; these ops give `O = 0`, `LSE = 0`. It has to be that way: ggml forms the packed tensor's own gradient as `ggml_scale(packed, 0.0f)`, and `0 * NaN` is NaN, which would ride into `dO` and poison every parameter. Same reason the forward zeroes its alignment gap.
- **No floating-point atomics.** Every element of dQ/dK/dV is written once, by one thread, out of a register accumulator; GQA folds by looping query heads ascending. Two runs on the same inputs give bit-identical gradients, which is the only thing that makes an A/B against `--attn exact` mean anything.
- **Masked positions are bitwise zero**, not small. `exp(-INF - finite)` is exactly `0.0f`, so `dS = P * (dP - D_i)` is exactly zero wherever the mask is.

Nothing here is reachable unless a caller asks. `ace-train train-dit --attn flash` is the only switch, and with `--attn exact` — the default — the emitted graph is byte-identical to before: the self-test's T3 tap comparison reports `0.00e+00` on every named tensor and SC1/SC2/SC3 report a `0.000e+00` gradient delta.

Measured on an RTX 5090 with `fattn-train-test` (S in {64, 129, 198, 384, 1000} x B in {1, 2} x {no mask, window, window+pad, cross-attention dead column} x GQA 32/8 x D 128, forward and all three gradients against the autodiff'd `dit_attn_f32` chain, bar 1e-4):

| run | cases | worst rel err | determinism |
|---|---|---|---|
| CPU | 36/36 | 3.517e-06 | n/a |
| CUDA | 36/36 | 3.088e-06 | bitwise across two runs |
| CUDA `--extra` | 48/48 | 3.088e-06 | bitwise |
| CUDA `--large` (S = 3000) | 37/37 | 3.088e-06 | bitwise |

`--bench` on CUDA, per attention site, forward + backward, 50 timed iterations (B 1, Nh 32 / Nkv 8, D 128, window mask):

| S | manual chain | fused | manual VRAM | fused VRAM |
|---|---|---|---|---|
| 625 | 0.642 ms | 3.015 ms (4.7x) | 301.3 MB | 98.6 MB |
| 1250 | 2.374 ms | 6.906 ms (2.9x) | 985.6 MB | 198.8 MB |
| 3000 | 11.123 ms | 23.209 ms (2.1x) | 4939.0 MB | 487.0 MB |

So v1 buys memory, not time. The kernels are f32 with no tensor-core tile pass yet; the ratio narrows as S grows only because the manual chain's own quadratic is catching up. A TF32/f16 tile pass is the obvious next lever and has not been started.

What the memory buys is the crop. The trainer carries a second arena model (`dit_vram_arena_bytes_flash()` in `dit-vram.h`, selected by `DitVramModel::flash_attn`) built term by term rather than refitted, and in flash mode the auto-fit's `crop_max` default lifts from 1250 to the dataset's longest track. Over a 20-cell {crop, segments, dataset} grid on a 5090 the raw model reproduces measured arena high-water within -2.0 %/+1.3 %; with its one fitted headroom coefficient it over-predicts every cell by +9.2 % to +13.3 % and never under. On `fightstar_behinddevilsback` with no crop pins, exact auto-fits crop 1250 (its cap) and flash auto-fits **crop 3846 at full 32-layer depth** — 3.08x — estimated 24,660 MB against 22,912 MB measured, NVML tripwire silent. At `--ckpt 8` the same dataset walks to crop 6000, the whole song.

Drift sits in the same documented non-identity class as `--bwd mm`: same dataset, same seed, 2 layers, crop 375, 2 epochs, probe loss 1.677639 exact against 1.677924 flash, about 1.7e-4.

**Losing this patch is loud, not silent.** `ggml.h` loses the op declarations and `ace-train` stops compiling; lose only the CUDA half and the trainer's `supports_op` probe aborts at init with `attn-unsupported` rather than letting the scheduler quietly split attention onto the CPU. `verify-hooks.ps1` Hook 12 greps `ggml.c` for the marker anyway, because the autodiff registration is the one piece that cannot be reconstructed from the two new CUDA files if a submodule update takes it — and those two files are *new*, so a submodule re-checkout deletes them outright rather than reverting them.

Reapply from the repo root — **apply all of them**. They are no longer file-disjoint: two touch `cpy.cu`, two touch `ggml.c`, and five now touch `ggml-cuda.cu`. Their hunks still do not overlap, and a full pristine→apply-all replay in glob order was re-checked when `flash-attn-train.patch` landed: every file that patch touches comes back byte-identical to the tested tree. The glob below is what CI runs:

```sh
for p in engine/patches/*.patch; do git apply --verbose "$p"; done
```

Verify they are still in place: `powershell -File engine\verify-hooks.ps1` (Hook 7 greps `out-prod.cu`, Hook 8 `ggml.c`, Hook 9 `cpy.cu`, Hook 10 `cpy.cuh`, Hook 11 `ggml-cuda.cu`, Hook 12 `ggml.c` again for the flash-attn-train autodiff case, each for its HOT-Step marker comment). **Hooks 9 and 10 matter most, because both failures are silent**: losing 9 leaves the numbers right and only the clock wrong, and losing 10 makes every sub-`q8_0` base vanish from the trainer rather than error. CI reapplies them in the "Apply engine patches" step of every build job, using the same glob loop.
