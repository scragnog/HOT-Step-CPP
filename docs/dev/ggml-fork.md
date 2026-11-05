# The HOT-ggml fork

`engine/ggml` is a submodule pinned to the `hot-step-neutral` branch of **HOT-ggml**, a fork of
[ggml-org/ggml](https://github.com/ggml-org/ggml); `.gitmodules` holds the fork's URL and the
branch name. The branch is a ggml-org release plus HOT-Step's ggml changes as commits, with the
rationale in each commit message. Most changes are one commit; a few have follow-up fixes, and
the Metal training ops arrived as a merged pull request. `git submodule status engine/ggml`
shows the pinned commit, and `git -C engine/ggml describe --tags` the ggml-org release under
it. A normal checkout and build leave `engine/ggml` clean; there is no patch step. Until
2026-10 these changes lived in `engine/patches/*.patch` and were applied over a stock
submodule at configure time.

| Change | What it is for | Hook |
|---|---|---|
| CPU fused snake (from the acestep.cpp upstream's ggml fork) | One CPU kernel for the mul → sin → sqr → mul → add chain | — |
| alloc-free-blocks | ggml-alloc free-block table 256 → 1024 | 13 |
| bf16-out-prod | BF16 `src0` in ggml-cuda's `OUT_PROD` | 7 |
| cudagraph-log | `GGML_CUDA_GRAPH_LOG=1` capture decision trace | — |
| f16-f32-accumulate | F16 cuBLAS GEMMs accumulate and write F32 | 11 |
| flash-attn-train | `FLASH_ATTN_TRAIN` / `_BACK` fused training attention (CPU + CUDA) | 12 |
| metal-acc-set-cpy | Metal non-inplace `ACC`/`SET` copy every column | — |
| metal-bin-threads | Metal binary ops without the 256-thread cap | — |
| metal-im2col-ic | Metal im2col parallel over input channels for N == 1 1D convs | — |
| Metal training ops (a merged pull request) | Metal kernels for the ops YuE2 joint training runs: `FLASH_ATTN_TRAIN` / `_BACK` (where the shape allows, the backward writes dS once and reuses it; `GGML_METAL_FA_TRAIN_DSW=0` recomputes instead), `CONVROT8` / `_BACK`, `BF16_ROUND` with epilogues in the unary and binary kernels, `RMS_NORM_BACK` and `REPEAT_BACK`, plus `OUT_PROD` and `SILU_BACK` kernels. Also CPU `CONVROT8` / `_BACK` kernels, and three optional hints on the fused attention op: `kv_grad_start`, a causal hint and a saved forward result in `src[4]`. The fork's `docs/metal-training.md` lists the Metal switches | — |
| metal-kquant-cpy | Metal K-quant (Q2_K to Q6_K) → F32/F16 `CPY`, the dequant the `mm-backward` arm needs to train on a K-quant base | — |
| metal-out-prod-smem | Metal tiled `OUT_PROD` store scratch aliases its input tiles, so two threadgroups share a core; same arithmetic | — |
| metal-cpy-transpose | Same-type F32/BF16 transposed row views copy through a padded 32 by 32 integer tile, preserving all bits; other layouts keep the generic copy. `GGML_METAL_CPY_TRANSPOSE=0` keeps the generic path | — |
| metal-bf16-round-producer | Metal skips an in-place `BF16_ROUND` of an already-rounded value, and the simdgroup `mul_mm`, an unfused `RMS_NORM` and `SILU` store a following single-consumer `BF16_ROUND` themselves. Bit-identical; `GGML_METAL_NO_BF16_ROUND_PRODUCER=1` turns it off | — |
| metal-mul-mm-half-prepass | In training graphs on Apple8, a base-forward BF16 `mul_mm` runs on half copies of its operands for every 32-wide K step whose values are all exact in half, and on bfloat otherwise. Bit-identical; `GGML_METAL_HALF_PREPASS_DISABLE=1` turns it off | — |
| mm-backward | `GGML_BACKWARD_MM=1` mul_mat formulation of the activation gradient; on Windows the latch reads the process environment, so a static-CRT build sees the trainer's `_putenv` | 8 |
| quant-cpy-kquant | quant → F32 `CPY` for every type the dequantizers cover | 10 |
| sched-unplaced-log | name the node no backend accepts | — |
| bf16-mul-mat-grad | a BF16 `MUL_MAT` weight operand may carry a gradient | 18 |
| zz-yue2-convrot8 | `CONVROT8` / `_BACK`, CUDA int8 rotated-linear training op (the Metal training ops add CPU and Metal kernels) | 15 |
| zzz-yue2-bf16-round | `BF16_ROUND` unary op (CPU + CUDA) | 16 |
| cuda-bf16-round-fuse | CUDA runs an F32 `ADD`/`SUB`/`MUL` and its single-consumer `BF16_ROUND` as one launch; `GGML_CUDA_NO_BF16_ROUND_FUSION=1` turns it off | — |
| cuda-bf16-round-defer | CUDA skips an in-place `BF16_ROUND` whose only reader is an F32 `ADD`/`SUB`/`MUL`, and that reader rounds the input as it loads it (bit-identical); `GGML_CUDA_NO_BF16_ROUND_DEFER=1` turns it off | — |
| cuda-bf16-storage | BF16 sources with F32 results for `ADD`/`SUB`/`MUL`, `REPEAT`, `SCALE`, `SILU`/`NEG`, `SILU_BACK`, `RMS_NORM`(`_BACK`), `ACC`, `SET` and `OUT_PROD`, plus `BF16_ROUND` as an F32 → BF16 store, for the YuE2 trainer's BF16 activation storage. An F32 `ADD`/`SUB`/`MUL` and its single-consumer BF16 round store run as one launch (`GGML_CUDA_NO_BF16_STORE_FUSION=1` turns it off); a deferred round's reader may have a BF16 other input; the other fusions skip a 32-node window around BF16 activations. Bit-identical: each load widens exactly and the arithmetic is the F32 path's | — |
| cuda-gemm-bf16-store | a BF16-weight `MUL_MAT` whose only reader is the next node's F32 → BF16 `BF16_ROUND` store has cuBLAS write the BF16 result there (FP32 accumulate, BF16 C), for the YuE2 trainer's BF16 activation storage; the trainer stores such rounds instead of deferring them. Not bit-identical (cuBLAS may pick another kernel); YuE2 track 003 losses within 0.01%, about -50 ms/step on a 5090. `GGML_CUDA_NO_GEMM_BF16_STORE=1` turns both off | — |
| cuda-bf16-producer-store | an `RMS_NORM`, `RMS_NORM_BACK`, `SILU`, `SILU_BACK` or NeoX `ROPE` whose only reader is the next node's F32 → BF16 `BF16_ROUND` store writes the BF16 value itself (the kernels take a BF16 dst), also exactly over its own dead BF16 input. Bit-identical: same F32 result, same RNE cast; about -45 ms/step on YuE2 training. `GGML_CUDA_NO_BF16_PRODUCER_STORE=1` turns it off | — |
| bf16-round-elide | `BF16_ROUND`'s backward passes a gradient that is already a round output straight through; `GGML_NO_BF16_ROUND_ELIDE=1` turns it off | — |
| bf16-round-inplace | a `BF16_ROUND` of an already-rounded value is built in place and CUDA skips its launch (the backward rule stays); `GGML_NO_BF16_ROUND_INPLACE=1` turns it off | — |
| CUDA BF16 flash-attn-train | BF16 Q/K/V `FLASH_ATTN_TRAIN` / `_BACK` kernels on sm_80+ (`mma.sync` m16n8k16, F32 accumulate) for an explicit `GGML_PREC_BF16` request with D = 128 and 4-byte aligned views. Packed outputs stay F32. The CPU backend refuses BF16 operands, so the scheduler never hands it one, and the kernels also build under HIP | — |
| fa-train-cuda-hints | the BF16 `FLASH_ATTN_TRAIN` kernels use the causal hint instead of reading the mask, and skip dK/dV below `kv_grad_start`; `GGML_CUDA_FA_TRAIN_NO_HINTS=1` turns it off. With the causal hint and 16-byte aligned rows they pipeline their tile loads with `cp.async` (bit-identical); `GGML_CUDA_FA_TRAIN_NO_PIPE=1` keeps the scalar kernels | — |
| zzzz-vulkan-train-ops | Vulkan `BF16_ROUND` and fused training attention | 14 |
| strict-f32-hint | `GGML_HINT_STRICT_F32` on a `MUL_MAT`: ggml-cuda skips the TF32 mma kernels, runs the cuBLAS F32 GEMM with `CUBLAS_DEFAULT_MATH` for that one call and puts the handle back in TF32 mode after. Nothing without the hint changes; `GGML_PREC_F32` keeps its meaning (existing YuE2 callers unchanged). Other backends ignore it. Set only by the MM3 refiner's MERT-v1 conditioner, which misses its parity gate under TF32 | 19 |
| ggml-rpc patch-level check | refuse an RPC server that lacks the fork's ops | — |

The YuE2 trainer's own speed changes sit outside the fork:

| change | rollback |
|---|---|
| gradient clip runs on the device, its abs-max as two cascaded 64-wide pools | `YUE2_AITK_CLIP_HOST=1` keeps the host path (D2H, host norm, H2D) |
| a LoRA scale of exactly 1.0 (alpha == rank) is skipped | none: `x * 1.0f` is bit-exact, so there is nothing to roll back |
| device-only stack blocks launch async, and each stack call synchronizes once at its end | `YUE2_AITK_SYNC_BLOCKS=1` waits after every block |
| the block backward's loss seed points at a resident 1.0 instead of `ggml_graph_reset` uploading it per block | `YUE2_AITK_BLOCK_RESET=1` |
| the AR head rows and the NAR final read the stack's final hidden and write dx as device tensors | `YUE2_AITK_HANDOFF_HOST=1` hands them over through host vectors |
| on CUDA the gradient clip is four kernel launches on its own stream (abs-max, fixed-order sum of squares, factor, apply) instead of a ggml graph of about ten nodes per slot. Same maths; the norm can differ in the last bits (summation order), the applied factor is unchanged unless the clip engages. About -50 ms/step on a 5090 | `YUE2_AITK_CLIP_GRAPH=1` |
| the fused q\|k\|v and gate\|up linears over a fused GGUF weight run one GEMM per output over a row view of the weight (and of the companion and LoRA B), instead of one fused GEMM whose output is sliced apart by copies, whose gradients then accumulate into a zero canvas. Not bit-identical (the input gradient is a sum of per-output GEMMs); track 003 losses within 0.03%, about -55 ms/step on a 5090. LoKr, the dense companion, ConvRot bases and strict F32 keep the fused path | `YUE2_AITK_FUSED_SLICE=1` |
| the frozen teacher's final hidden is cached on the host per AR input-id sequence (it depends on nothing else), so a repeated item skips the 28 base blocks. Filled until `YUE2_AITK_TEACHER_CACHE_MB` (default 2048) and never evicted, since epochs revisit items in a cycle. Bit-identical; about -130 ms/step from the second epoch on a 5090 | `YUE2_AITK_TEACHER_CACHE_MB=0` |
| BF16 flash attention takes q and k as they are (one BF16 copy each into its layout) and applies 1/sqrt(D) to the F32 scores, instead of scaling q and k by D^-1/4 in F32 and rounding them again. Not bit-identical; track 003 losses within 0.04%, about -23 ms/step on a 5090. Math attention and strict F32 keep the prescale | `YUE2_AITK_QK_PRESCALE=1` |
| RoPE is one `ggml_rope_ext` NeoX op (F32 trig, theta 1e6, the op the YuE2 inference graph runs) with one BF16 round of its output, instead of the BF16 table path that rounds each product and joins the halves through a zero canvas. Not bit-identical; track 003 losses within 0.03%, about -85 ms/step on a 5090. Strict F32 keeps the table path | `YUE2_AITK_ROPE_TABLE=1` |
| on CUDA (the only backend with the BF16-source kernels) and outside strict F32, BF16 activation storage stores every BF16_ROUND output (and the views and conts that reread it) as a BF16 tensor; every other op keeps its F32 result, so the values are the same. A round whose only reader is an `ADD`/`SUB`/`MUL` and whose source is not one stays F32 for the CUDA deferral (`YUE2_AITK_BF16_DEFER=0` retypes it too). About -40 ms/step on a 5090; a GEMM whose only reader is a BF16_ROUND has the cuBLAS call write BF16 C directly (fork `c2a4fc04`), fusing the round into the store; not bit-identical for every shape, and `GGML_CUDA_NO_GEMM_BF16_STORE=1` restores the separate round. On CUDA the AR head stores its logits as BF16 too, and the loss kernel writes the logit gradient as BF16. A BF16 copy of a BF16 tensor that only flash attention reads becomes a strided view (`YUE2_AITK_ATTENTION_COPY=1` keeps the copies) | `YUE2_AITK_BF16_STORAGE=0` |

`cpy-q-occupancy` (the quant → F32 copy launch geometry) is upstream since ggml-org
`b64fb805`; Hook 9 checks the launch shape itself. Hook 17 checks that `engine/ggml` is clean
and at the pinned commit.

## Changing ggml

Make the change on HOT-ggml's `hot-step-neutral` branch as its own commit, with the rationale in the
message, then move the gitlink in this repo. Never leave edits in `engine/ggml`: they are not
what CI or users build, and the next submodule update discards them. To take a newer ggml-org
master, rebase or replay `hot-step-neutral` onto it, rebuild, and rerun the parity checks the
original changes were measured with (below).

If `engine/ggml` is a stock ggml-org checkout, the submodule URL is cached from before the
fork. From the repo root: `git submodule sync -- engine/ggml`, then
`git submodule update --init engine/ggml`. `engine/CMakeLists.txt` refuses to configure
without the fork's ops, and `engine/verify-hooks.ps1` names the missing piece.

### ConvRot8 graph-replay probe

`convrot-graph-probe` (`engine/tools/convrot-graph-probe.cpp`, CUDA builds only) runs
YuE2-shaped ConvRot8 forward and its autodiff gradient on the CUDA backend with CUDA graphs on.
Each cycle builds fresh graphs, and every phase runs a direct compute, a capture and then
replays, flipping the input pointer between phases to force a re-capture. One graph is ConvRot
alone; the other puts F32 cuBLAS GEMMs before and after ConvRot on the same stream's handle.
It fails on any non-finite, all-zero or non-bitwise-identical output or gradient, on a
"replay" that still calls cuBLAS (graphs not in use), and on any `cublasSetStream` outside
handle creation, counted from cuBLAS's own API log after a positive control. A workspace-loss
capture failure aborts inside ggml-cuda, so any run that ends without a `RESULT` line is a fail.

```bat
rem from engine\ (Visual Studio builds put the binaries in build\Release\)
build\Release\convrot-graph-probe.exe --device CUDA0
build\Release\convrot-graph-probe.exe --device CUDA0 --cycles 5 --phases 4 --computes 8 --rows 1024
```

Exit codes: 0 pass, 1 a check failed, 2 setup error. `--fp32` runs ConvRot without BF16, and
`GGML_CUDA_GRAPH_LOG=1` adds ggml-cuda's per-compute graph decision line. On the pre-fork
patch overlay the stream-reset check is expected to fail, because the old `StreamGuard` and
old ggml's cuBLAS mul_mat and out_prod each call `cublasSetStream` per op; the per-graph counts
separate the two. To build it there, cherry-pick the commit that adds the probe (the tool and
its `engine/CMakeLists.txt` block only) onto that checkout.

## Metal transposed copies

`metal-cpy-transpose` handles same-type F32 and BF16 `CPY`/`CONT` from a
transposed row view into a contiguous tensor of the same shape. Both matrix
dimensions must be at least 32. Source rows and batch planes may have padding,
and source and destination may start at an element-aligned view offset. The
dispatch checks strides, batch-plane separation and disjoint storage ranges.
Reshapes, casts, smaller matrices, other permutations and strided destinations
keep their existing copy path. A missing pipeline or a threadgroup limit below
256 also keeps that path.

Each 32 by 8 threadgroup loads a 32 by 32 tile along the contiguous source
dimension, synchronizes, and writes along the contiguous destination dimension.
The tile has a padded column to reduce bank conflicts. BF16 uses `ushort` storage
and F32 uses `uint`, so the kernel performs no floating-point conversion or
arithmetic. Tail guards cover both loads and stores. `GGML_METAL_CPY_TRANSPOSE=0`
selects the original dispatch for a control run; it is read once per process.

Production changes are confined to `src/ggml-metal/` in the fork. Shared ops,
tensor ABI, graph construction and CPU, CUDA and Vulkan code are unchanged.

Measured on an M2 Max: the 6144 by 2048 BF16 transpose took 0.525 ms versus
3.933 ms on the generic path, averaged over 100 synchronized copies after
warmup. The independent host-reference probe passes 126 cases with the feature
enabled and disabled, including partial tiles, batches, padding, offsets,
non-contiguous fallbacks, source preservation, destination canaries and raw
F32/BF16 bit patterns.

Three paired fresh-state warm-song updates took 33.779/33.271/32.781 s for the
predecessor and 31.980/32.304/32.395 s with the transpose enabled. Median paired
saving is 0.967 s, or 2.91%; the separate medians are 33.271 and 32.304 s.
This uses the approved exact-copy exception to the default 3% kernel-change bar.
All seven matched updates log bit-identical AR CE, NAR MSE, KL and gradient norm
against both the predecessor and original Metal baseline. C alone on the
original fork also matches all seven updates. The existing Metal CPY/CONT suite
passes 285/285 cases with the feature enabled and disabled. Raw evidence lives
under `.scratch/hs344/` on the development checkout and `.probe-logs/hs344/` on
the measurement mirror (HOTSTEP-344).

## Metal BF16_ROUND producer epilogues

The YuE2 trainer rounds most activations with `ggml_bf16_round()`. On Metal each
round ran as its own unary pass: about 9,000 per training update and about 370 GB
of extra memory traffic. `ADD`/`SUB`/`MUL` already store a following round
(`rnd=1`). `metal-bf16-round-producer` covers the rest of the large producers:

- A round that `ggml_bf16_round()` marked as a round of an already-rounded value
  (`op_params[1] == 1`, built in place) encodes nothing, as on CUDA.
- The simdgroup `mul_mm` kernel, an `RMS_NORM` that no table fusion took, and
  `SILU` write `bf16_round_cast(result)` into the round's buffer when the next
  node is a single-consumer F32 `BF16_ROUND` of their contiguous F32 result.
  `mul_mm` stages its accumulators through threadgroup memory and rounds on the
  copy out. Rounding the simdgroup matrices in registers made it about 4x slower.

The F32 result and the rounding function are the standalone pass's, so the output
is bit-identical. The tensor-API `mul_mm`, mat-vec paths, the K = 32 kernel and
table fusions keep the separate round. `GGML_METAL_NO_BF16_ROUND_PRODUCER=1` or
`GGML_METAL_FUSION_DISABLE=1` turns all of it off; `GGML_METAL_FUSION_DEBUG=2`
logs each fusion and skip. Rounds of graph inputs and those after `ROPE`,
`RMS_NORM_BACK`, `SILU_BACK` and the attention output still run standalone,
about 2,300 per update and 54 GB.

Measured on an M2 Max. A 2048 by 6144 BF16 GEMM over 7,668 rows took 26.8 ms
with the epilogue against 27.5 ms for GEMM plus round. A probe of 27 cases
(full and partial tiles, K tails, batches, LoRA rank, F32 weights, a second
consumer, views, in-place chains, mat-vec and K = 32 fallbacks, `RMS_NORM` and
`SILU` row shapes) matches the unfused graph bit for bit with the feature on, off
and with fusion disabled. Three paired fresh-state warm-song updates took
32.319/32.426/32.293 s for the predecessor and 30.903/30.953/30.750 s with it;
median paired saving 1.473 s (4.6%). All seven matched updates log bit-identical
AR CE, NAR MSE, KL and gradient norm against the predecessor and the original
Metal baseline, and so do the switched-off control and this change alone on the
original fork. Raw evidence lives under `.scratch/hs345/` on the development
checkout and `.probe-logs/hs345/` on the measurement mirror (HOTSTEP-345).

## Metal half pre-pass for base-forward GEMMs

On Apple8 the half MAC is about 22% faster than the bfloat one, and a half product
equals the bfloat product whenever both inputs are exact in half, also with the type
mixed per K step (HOTSTEP-347). Converting inside the GEMM cost as much as it saved
(HOTSTEP-348). `metal-mul-mm-half-prepass` converts first instead:

- `kernel_mul_mm_hp_cvt_w` writes a half copy of the BF16 weights, and
  `kernel_mul_mm_hp_cvt_a` writes bfloat and half copies of the BF16-rounded F32
  activations. Both set one flag per 8-wide K block of each tile (64 weight rows,
  32 activation rows) when every value in it is exact in half.
- `kernel_mul_mm_bf16_f32_hp` runs a 32-wide K step on the half MAC when all of
  the step's flags are set and on the bfloat MAC otherwise. Staging is a 16-bit
  copy from whichever copy the step picks; accumulation and the store, BF16_ROUND
  epilogue included, are `kernel_mul_mm`'s.

It applies only in graphs built with grads, on Apple8 without the tensor API, to
2-D contiguous BF16-by-F32 GEMMs with `K % 32 == 0` whose weight reaches a frozen
leaf without a transpose. dX GEMMs, BF16 casts of trainable LoRA factors, graphs
without grads and inference keep the existing kernel. The copies share one private
scratch buffer grown on demand; the op puts barriers around its pre-pass. Each use
holds a reference until its command buffer completes, and a replaced buffer is freed
once none remain, on the next scratch request or in `ggml_metal_free`.
`GGML_METAL_HALF_PREPASS_DISABLE=1` turns it off.

Measured on an M2 Max. Microbenchmark on the base-forward shapes with the audit's
fallback mix: 0.903 of the existing kernel's time with both pre-passes, FLOP-weighted
(F16 weights would be 0.78), bit-identical on every shape and mix. Three paired
fresh-state warm-song updates took 31.765/32.259/32.032 s for the predecessor and
31.146/31.173/31.402 s with it; median paired saving 0.629 s (1.97%), and 7.35 s over
the seven updates. Peak process footprint rose 0.53 GB at the median. All seven
matched updates log bit-identical AR CE, NAR MSE, KL and gradient norm against the
original Metal baseline, with equal exported adapter hashes, and so do the
switched-off control and this change alone on the original fork. `scratch-life.cpp`
checks that replaced scratch buffers outlive their command buffers and are all freed. Report: `tools/yue2-aitk-reference/hs353-half-prepass/REPORT.md`
(HOTSTEP-353).

## Why each change exists

The sections below are the measurements and reasoning behind each change, kept from the
old patch README.


`bf16-out-prod` teaches ggml-cuda's `out_prod` to accept a **BF16 `src0`**, which is what lets the DiT trainer's `--mirror bf16` keep frozen trainable-layer weights in BF16 instead of promoting the whole mirror to F32.

`mm-backward` adds an **env-gated alternative formulation** for the `MUL_MAT` activation gradient in `ggml.c`'s `ggml_compute_backward`. Upstream emits `out_prod(src0, transpose(grad))`, and ggml-cuda's `OUT_PROD` is F32-only — which forces the frozen weight to F32 and drags the *forward* `mul_mat` onto TF32 tensor cores too. With `GGML_BACKWARD_MM=1` set, the backward becomes `mul_mat(cont(transpose(src0)), grad)` instead: mathematically and shape-wise identical, but dtype-agnostic, so a BF16 weight rides real BF16 tensor cores in both directions with no dequant. Measured on an RTX 5090 at ~1.7–1.8× per layer per step. With the env var unset the emitted graph is byte-identical to upstream.

The same change carries a second, narrower rule added for the DiT trainer's third mirror mode, `--mirror bf16-f32` (2026-09-02). That mode stores frozen trainable-layer weights as BF16 and promotes each one to F32 with an in-graph `ggml_cast` at its `mul_mat` site, so the GEMM is f32 while residency stays bf16 — the fix for adapters that render coarse under plain `--mirror bf16`. It only works if the cast's F32 output dies with the forward matmul, and the `mm` arm above would have kept it alive: naming `src0` in a backward node makes ggml-alloc hold all 32 layers' F32 copies from the forward pass until each layer's backward runs, which is precisely the ~8 GB the mode exists not to spend. **The failure would have been silent** — correct gradients, correct loss, and a peak footprint back at the F32 mirror's.

So when `src0` is exactly that node — a `ggml_cast` (a `CPY` with the self-referencing `src[1]`) to F32 off a gradient-free, non-parameter BF16 *leaf* — the backward re-casts the leaf instead of referencing the forward's cast. Same value (BF16→F32 is exact), its own short-lived node, and the forward copy is reclaimed the moment its matmul is done. Nothing else in the engine matches that shape, and the `out_prod` fallback arm is deliberately left alone. Measured on one training dataset (32-layer XL base, LoKR dim 512 + MLP, `--attn flash --bwd mm --optimizer muon`, 12 same-seed epochs, crop pinned 552): mirror 8133 MB against f32's 15893 MB, trainer-owned peak 14370 MB against bf16's 14190 MB and f32's 21997 MB — i.e. the transient window costs 180 MB, not 7.8 GB — and `loss`/`ma5`/`raw`/`gnorm` came out **bit-identical to `--mirror f32` at every one of the twelve epochs** — equal to the last digit of the full-precision `dit_train_log.json`, and the two runs' exported `lokr_weights.safetensors` have the same MD5 — where `--mirror bf16` diverged from the first epoch and reached 7.8e-3 by the twelfth. Unpinned, the flash auto-fit picks crop 1542 against f32's 610 and bf16's 1632. The cast is not free: 3.21 s/epoch against f32's 2.56 and bf16's 2.17, i.e. +25% over f32 at equal crop — which the 2.5× crop repays.

`cpy-q-occupancy` fixed the **launch geometry of ggml-cuda's quant→F32 copies**: upstream launched them one CUDA block per element with one thread inside, so 31 of every 32 blocks only hit a bounds guard. Measured on a 5090, MM3 8.6B q8_0 training: **11.1 → 3.75 s/step**, parity with f16. The same fix landed upstream as ggml-org `b64fb805`, so the fork no longer carries it. The failure mode was silent (right numbers, wrong clock), which is why Hook 9 still checks the launch shape.

`ace-train`'s `--bwd <outprod|mm>` sets that env var. The Training Studio defaults `train-dit` to `mm` and always passes `mm` to `sa3-train`, but defaults `train-lm` to `outprod`: the LM's default `--weights bf16` reaches the same mul_mat backward by rewriting the graph's `out_prod` nodes, and under `mm` there are none to rewrite, so the server refuses that pair.

The `mm` arm also dequantizes a **block-quantized `src0`** to F32 before transposing it (#197). A quantized weight's blocks run along `ne0`, so `cont(transpose(W))` is a CPY that no backend implements: CUDA aborts in `ggml_cuda_cpy` ("q8_0 to q8_0"), and Vulkan and the CPU crash with an access violation. It only bites trainers that feed a quantized weight straight into `mul_mat`, which is `yue2-nar-train` (MM3 LM already casts in-graph, so its backward only ever sees F32). Before the fix, a YuE2 NAR run on any non-BF16 base died on its first backward, on every backend. After it, q8_0, Q4_K_M and IQ3_XXS train on CUDA at the bf16 step time (0.44 s/step, 5090) with first-step loss within 3 % of bf16, and on Vulkan at about 2.4 s/step. The cast is transient, like the `--mirror bf16-f32` re-cast above.

`zzzz-vulkan-train-ops` gives ggml-vulkan the two custom ops the YuE2 joint trainer cannot run without, so it trains on a Vulkan build (#197). The trainer computes each layer with `ggml_backend_graph_compute` straight on one backend, no scheduler, so an op the backend lacks is a refusal, not a CPU fallback.

- **`BF16_ROUND`**: one more entry in the generic `unary.comp`, round-to-nearest-even with NaN kept quiet, bit-identical to the CPU's `GGML_FP32_TO_BF16`.
- **`FLASH_ATTN_TRAIN` / `_BACK`**: three new kernels (`fa_train_fwd`, `fa_train_bwd_dq`, `fa_train_bwd_dkv`) implementing the CPU oracle's maths and layouts: online softmax with LSE, then dQ per query row and dK/dV per key row recomputing P from the LSE. One 32-lane cluster per row, reduced with `subgroupShuffleXor`, so they need 32-wide subgroups with shuffle; without them `supports_op` says no and the trainer uses exact attention. `fattn-train-test` passes 49/49 against the CPU with `--extra` on an RTX 5090 and a 4090 over Vulkan (measured when `--backend cuda` still took the first GPU; the same run is `--backend vulkan` now), including GQA, S ≠ S_kv, dead key columns and bitwise determinism. The kernels are plain f32 scalar code: correct, several times slower than CUDA's.
- **Dispatches are sliced** to about 2^25 (row, partner, head) pairs each (`GGML_VK_FA_TRAIN_SLICE_PAIRS` overrides, for tests). One dispatch over a whole song is seconds of GPU time, which freezes the desktop on a GPU that also drives a display and can trip the Windows TDR reset. The parity grid passes at the smallest slices too.

`bf16-mul-mat-grad` relaxes one assert in `ggml_build_backward_expand`, which required every source on a gradient path to be F32 or F16. The YuE2 trainer casts its F32 adapter factors to BF16 at each GEMM so the products run on BF16 tensor cores, and that cast is `MUL_MAT`'s `src0` on the parameter's gradient path. `MUL_MAT`'s backward writes the `src0` gradient in F32 (`OUT_PROD`), and the cast's `CPY` backward reshapes it onto the F32 parameter, so nothing downstream needs the restriction. The allowance is BF16, `MUL_MAT`, `src0` only; every graph that built before builds the same. Lost, YuE2 joint training aborts at its first backward unless `YUE2_AITK_PRECISION=strict-f32`.

`sched-unplaced-log` makes `ggml_backend_sched_split_graph` **name the node no backend accepts** before its `GGML_ASSERT(*cur_backend_id != -1)`: op, type, shape, and each source's type, strides and buffer. Without it, a user on Vulkan or Metal who reaches an op neither the GPU backend nor the CPU fallback supports sends a bare assert, and we have to guess which op it was.

`quant-cpy-kquant` teaches the same `CPY` dispatch to reach the **generic quant→F32 converter ggml already ships**. Upstream hand-writes one quant→F32 copy per type and has written five — `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`. Every K-quant, every IQ type, MXFP4 and NVFP4 have none, so `ggml_cast(w, F32)` on such a weight is refused by `supports_op` and the graph falls off the GPU. Those dequantizers are not missing: `convert.cu` exposes `ggml_get_to_fp32_cuda()` covering `Q2_K`–`Q6_K`, `IQ1`–`IQ4`, MXFP4 and NVFP4, and it is the same converter the `mul_mat` path uses on every token of every quantized inference. Only the CPY dispatch never learned to ask for it.

So that change adds **one fallback branch rather than twenty kernels**, last in the else-if chain — every type upstream already handles keeps its existing kernel byte for byte, and only combinations that would have hit `GGML_ABORT` reach the new path. It requires contiguous src and dst of the same shape, which is exactly what `ggml_cast()` produces and the only form the converters accept. `cpy.cuh` exports `ggml_cuda_cpy_quant_to_f32_supported()` so the dispatch and `supports_op()` answer that question from one place instead of two lists that drift.

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

`cudagraph-log` adds an **env-gated one-line-per-compute decision trace**
(`GGML_CUDA_GRAPH_LOG=1`) to `ggml_backend_cuda_graph_compute` — key, uid,
node count, compatibility, warmup state, and a line whenever node properties
change. Zero cost when unset. This is a *diagnostic*, not a fix: it is the
tool that established that CUDA graph capture is already active for every MM3
graph (LM decode −29 %, depth −14 % vs `GGML_CUDA_DISABLE_GRAPHS=1`) and that
an apparent post-model-swap "graph thrash" was actually the select-model API
resetting an omitted LM role to auto/f16. **Losing this change is benign** —
no verify-hook guards it; reapply it when you next need the trace.

`f16-f32-accumulate` makes ggml-cuda's batched cuBLAS GEMM **accumulate and write F32 for an F16 `src0`**, which upstream only does on Volta, RDNA4 and CDNA. Everywhere else F16 lands on `CUBLAS_COMPUTE_16F`: the dot product accumulates in half precision *and* `dst` is written in half precision, so a partial sum past 65504 becomes `+inf` and everything downstream of it NaN. Nothing reports it — the GEMM succeeds and returns garbage. BF16 and F32 have always taken the F32 path; F16 was the odd one out.

This is not theoretical. The MiniMax-Music3 LM is Qwen3-8B, with K of 4096 and 12288, and it clears that ceiling as soon as an LM LoRA/LoKr shifts the residual stream up — **and only then**, which is why it presented for months as "adapters are broken on f16, use q8" rather than as a matmul bug. Measured on an RTX 5090 (cc 12.0), same prompt, same seed, same adapter, one variable:

| LM base | non-finite candidate logits |
|---|---|
| f16 | one whole CFG row dead on **every** step |
| bf16 | none — already took the F32 branch |
| q8_0 | none — never reaches cuBLAS |

`bf16` is the control that settles it: same value range as `f16`, same graph, and the only difference is which compute type it lands on. Cost of the fix, MM3 LM decode on a 5090: 11.03 → 11.59 ms/step (+5 %), landing exactly on bf16's 11.59 — which is what you would expect once they share a path. FP16 inputs with an FP32 accumulator run at full tensor-core rate from Ampere on.

**Losing this change is silent and expensive** — no error, no crash, just noise instead of music on any f16 render with an adapter — so `verify-hooks.ps1` Hook 11 greps for the marker. The two guards in `mm3-ar-loop.h` (refuse a plan whose logits are wholesale non-finite) and `mm3-lm-adapter.h` (refuse a checkpoint whose factors would overflow the f16 store) are the second line of defence, and they live in the main repo rather than here.

`flash-attn-train` adds two fused attention ops — `GGML_OP_FLASH_ATTN_TRAIN` and `GGML_OP_FLASH_ATTN_TRAIN_BACK`, appended at the tail of the op enum so every existing value stays put — with a CPU reference implementation, CUDA kernels in two new self-contained files (`ggml-cuda/fattn-train.{cu,cuh}`; no existing `fattn-*` file is touched), and an autodiff case so `ggml_build_backward_expand` emits the backward on its own.

What it is for is the DiT trainer's retained softmax. `dit_attn_f32` materialises a `[S_kv, S, Nh, B]` f32 softmax at every attention site and the backward keeps it alive, so attention memory is quadratic in the crop. At `S = 3000` across 32 layers that is 36.9 GB of self-attention softmax plus 9.4 GB of cross-attention softmax — the arithmetic behind `docs/dev/training-internals.md`'s "full-song training is impossible" line. The fused forward emits one packed tensor holding `O` and the log-sum-exp and nothing of size S²; the fused backward recomputes tiles from Q/K/LSE using the row identity `D_i = rowsum(dO∘O)`. Both trainer attention sites go through it, so both quadratic terms go.

The ops are deliberately generic rather than trainer-shaped — arbitrary additive F16 mask with `soft_max_ext`'s broadcast rule, GQA handled natively (`Nkv < Nh` with no pre-expansion), `S_kv != S`, `B >= 1`. That is what makes adoption by the other trainers call-site wiring rather than kernel work.

Three properties are load-bearing, and all three are deliberate:

- **A fully-masked query row is defined, not inherited.** `ggml_soft_max_ext` gives NaN there; these ops give `O = 0`, `LSE = 0`. It has to be that way: ggml forms the packed tensor's own gradient as `ggml_scale(packed, 0.0f)`, and `0 * NaN` is NaN, which would ride into `dO` and poison every parameter. Same reason the forward zeroes its alignment gap.
- **No floating-point atomics.** Every element of dQ/dK/dV is written once, by one thread, out of a register accumulator; GQA folds by looping query heads ascending. Two runs on the same inputs give bit-identical gradients, which is the only thing that makes an A/B against `--attn exact` mean anything.
- **Masked positions are bitwise zero**, not small. `exp(-INF - finite)` is exactly `0.0f`, so `dS = P * (dP - D_i)` is exactly zero wherever the mask is.

Nothing here is reachable unless a caller asks. `ace-train train-dit --attn <exact|flash|flash-f32>` is the only switch, and with `--attn exact` — the default — the emitted graph is byte-identical to before: the self-test's T3 tap comparison reports `0.00e+00` on every named tensor and SC1/SC2/SC3 report a `0.000e+00` gradient delta.

Each op carries a precision request in `op_params` slot 3 (`ggml_flash_attn_train_set_prec`), resolved once per dispatch in `fa_train_resolve_prec()`. `GGML_PREC_F32` pins the original scalar FFMA kernels; the default selects `m16n8k8` TF32 tensor-core kernels for both directions, guarded on `AMPERE_MMA_AVAILABLE` and on the alignment contract, and falling back to the scalar path (with the reason in the resolved label) on anything else. A forward that rounds to TF32 can never be paired with a scalar backward: the resolver decides once and both directions read the same answer. `--attn flash` asks for TF32, `--attn flash-f32` pins the scalar kernels.

That distinction only matters against the CPU, not against what shipped: the exact-mode CUDA path has *always* run its attention `mul_mat`s on cuBLAS TF32 — the self-test measures that reference rounding at ~3e-3 on CUDA against ~2.5e-6 for the same case on CPU. A TF32 fused kernel is therefore precision-par with every adapter this project has ever trained in exact mode, and the strict-f32 kernels are *tighter* than the historical path. So the parity tool gates TF32 at 5e-3 and strict f32 at 1e-4. The looser bar also has a **floor**: a run that silently fell back to the scalar kernels would sail under 5e-3 and report as a TF32 pass, so the tool refuses any TF32 case whose error is f32-sized, and asserts through the backend registry (`ggml_backend_cuda_fattn_train_last_prec`) that the kernel it asked for is the kernel that ran, in both directions.

Measured on an RTX 5090 with `fattn-train-test` (S in {64, 129, 198, 384, 1000} x B in {1, 2} x {no mask, window, window+pad, cross-attention dead column} x GQA 32/8 x D 128, forward and all three gradients against the autodiff'd `dit_attn_f32` chain):

| run | bar | cases | worst rel err | determinism |
|---|---|---|---|---|
| CPU | 1e-4 | 36/36 | 3.517e-06 | n/a |
| CUDA f32 | 1e-4 | 36/36 | 3.088e-06 | bitwise across two runs |
| CUDA f32 `--extra` | 1e-4 | 51/51 | 3.088e-06 | bitwise |
| CUDA f32 `--large` (S = 3000) | 1e-4 | 37/37 | 3.088e-06 | bitwise |
| CUDA `--prec tf32` | 5e-3 | 36/36 | 4.732e-04 | bitwise |
| CUDA `--prec tf32 --extra` | 5e-3 | 51/51 | 4.732e-04 | bitwise |
| CUDA `--prec tf32 --large` | 5e-3 | 37/37 | 4.732e-04 | bitwise |

`--bench` on CUDA, per attention site, forward + backward, 50 timed iterations (B 1, Nh 32 / Nkv 8, D 128, window mask):

| S | manual chain | fused f32 | fused tf32 | manual VRAM | fused VRAM |
|---|---|---|---|---|---|
| 625 | 0.645 ms | 3.004 ms (4.66x) | **0.604 ms (0.94x)** | 301.3 MB | 98.6 MB |
| 1250 | 2.370 ms | 6.931 ms (2.92x) | **1.518 ms (0.64x)** | 985.6 MB | 198.8 MB |
| 3000 | 11.143 ms | 23.300 ms (2.09x) | **5.449 ms (0.49x)** | 4939.0 MB | 487.0 MB |

The TF32 pass is what turned the feature from a memory trade into a straight win. The scalar kernels bought 3x-10x less attention VRAM at 2.1x-4.7x the time; the tensor-core kernels keep the identical memory (the same tiling, only the two products moved onto `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`) and are now faster than the cuBLAS chain everywhere measured — level at S = 625, 1.6x at 1250, 2.0x at 3000. The margin grows with S because the manual chain still pays a dense S² softmax where the fused kernel skips whole tiles the window mask has killed; the bench prints live keys/row per S so a miss reads as a tile-skip problem or a throughput one rather than a mystery.

End to end that shows up as parity rather than a penalty. Same dataset, same seed, 32 layers, crop 1250, 20 epochs: exact 44,160 ms at 21,950 MB peak, `--attn flash` 46,493 ms at 18,938 MB, `--attn flash-f32` 89,845 ms at the same 18,938 MB. So TF32 flash costs about 5 % of wall clock at a crop the exact model can still afford, and buys 14 % of the peak back — and unlike exact it is not capped there.

What the memory buys is the crop. The trainer carries a second arena model (`dit_vram_arena_bytes_flash()` in `dit-vram.h`, selected by `DitVramModel::flash_attn`) built term by term rather than refitted, and in flash mode the auto-fit's `crop_max` default lifts from 1250 to the dataset's longest track. Over a 20-cell {crop, segments, dataset} grid on a 5090 the raw model reproduces measured arena high-water within -2.0 %/+1.3 %; with its one fitted headroom coefficient it over-predicts every cell by +9.2 % to +13.3 % and never under. On one dataset with no crop pins, exact auto-fits crop 1250 (its cap) and flash auto-fits **crop 3846 at full 32-layer depth** — 3.08x — estimated 24,660 MB against 22,912 MB measured, NVML tripwire silent. At `--ckpt 8` the same dataset walks to crop 6000, the whole song.

Flash mode also drops `dit_expand_heads` — the fused ops do GQA natively — and with it the `repeat`/`repeat_back` node pair per attention site per layer, the x(Nh/Nkv) K/V activation blow-up, and ggml's CUDA `REPEAT_BACK` cap of 32768 on `n_kv_heads * max(S, enc_S) * B`. That cap is a correctness constraint on the micro-batch, not a memory one, and in exact mode it is what refuses `--batch 4` on a dataset with `enc_S = 1877` (8 * 1877 * 4 = 60,064). In flash mode the clamp is disarmed and only VRAM decides.

Drift sits in the same documented non-identity class as `--bwd mm` and `--mirror bf16`, and it does not compound. Same dataset, same seed, 2 layers, crop 374: at 2 epochs exact 1.416741139 against flash 1.416809989 (6.9e-5); at 200 epochs exact 1.130100180 against flash 1.130353922 (2.5e-4) and flash-f32 1.130287604 (1.9e-4). Deeper, at 32 layers and crop 1250 over 20 epochs: exact 0.883770987, flash 0.888023981 (4.3e-3), flash-f32 0.885164537 (1.4e-3).

**Losing this change is loud, not silent.** `ggml.h` loses the op declarations and `ace-train` stops compiling; lose only the CUDA half and the trainer's `supports_op` probe aborts at init with `attn-unsupported` rather than letting the scheduler quietly split attention onto the CPU. `verify-hooks.ps1` Hook 12 greps `ggml.c` for the marker anyway, because the autodiff registration is the one piece that cannot be reconstructed from the two new CUDA files if a submodule update takes it.

`alloc-free-blocks` raises ggml-alloc's **per-chunk free-block table** from 256 to 1024 entries. `ggml_dyn_tallocr` tracks the holes in each buffer chunk in a fixed array and asserts when it fills; the DiT trainer's LoKR at dim 256 / factor 6 routes 224 matrices through factorized `w2`, which makes a ~19k-node fwd+bwd graph whose many small, short-lived tensors fragment a chunk past 256 holes. Seen 2026-09-02 during the overnight sweep — twice, same signature, epoch 8 at crop 936 and epoch 17 at crop 850 — as `ggml-alloc.c:135: GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS)`. Capacity only: the array is per chunk, so the cost is 768 × 16 B per chunk and no allocation decision changes; every other configuration allocates exactly as before. It is an inference-shared file, so a smoke generation after applying it is the honest check.

`metal-acc-set-cpy` fixes the **non-inplace `ACC` and `SET` on Metal** (#206). Both ops first copy `src0` into `dst` with `kernel_cpy_t_t`, which copies one element per thread, but upstream dispatched one threadgroup per row, so a row wider than the threadgroup size (1024) had only its first 1024 columns copied and the rest of `dst` kept whatever the reused buffer held. Autodiff builds the gradient of a `view` from exactly this op, so every YuE2 joint-training backward on Metal was affected: non-finite adapter gradients a few steps in, and silently polluted gradients on the steps that did not crash. The patch dispatches `ceil(ne00/nth)` threadgroups per row, as `ggml_metal_op_cpy` already does for the same kernel. Found, diagnosed and fixed by the issue's reporter on an M1 Max; nobody on the team has Metal hardware to re-run it. No other backend is touched and no verify-hook guards it.

`metal-out-prod-smem` (fork `bea8ded3`) shrinks the **tiled Metal `OUT_PROD`**'s threadgroup memory. `kernel_out_prod_mm_f32` kept a separate 32x64 F32 store scratch next to its src0 and src1 tiles, 22016 bytes, which filled a core, so one 128-thread threadgroup ran per core and its scalar strided loads had nothing to hide behind. The scratch is only written after the K loop, so it now aliases the src0 tile behind a barrier: 13824 bytes (9216 for the small-M kernel), two threadgroups per core, unchanged arithmetic and order. These are the LoRA weight gradients of YuE2 joint training, which reduce over the whole sequence. On an M2 Max at rank 64 and K = 7668, a 6144x64 gradient went from 9.7 to 5.3 ms and a 64x2048 one from 4.9 to 2.7 ms; seven matched joint-training updates logged bit-identical AR CE, NAR MSE, KL and gradient norm, and warm update 1 went from 36.01 to 34.67 s (median of three interleaved pairs). `yue2-out-prod-metal-tiled-test` covers the real orientation (`ar-*`, `nar-*`). Only `src/ggml-metal/` changes.

**Tried and rejected: split-K for the same kernel** (fork `ad683338`, reverted by `14fcaf20`, which is what the pin carries). Splitting K into chunks with F32 partials summed in a fixed order took the big shapes to 3.7 ms and repeated bit-identically, but it changes the summation order, and AdamW's first step (close to lr*sign(g)) turns that into training differences: AR CE up to 3.5e-4 on updates 2-7, outside the per-update scalar gates frozen for this work (`tools/yue2-aitk-reference/metal-baseline/gates.json`). Two different split counts differed from each other by the same amount, so no chunking passes. It gained 0.31 s per update over the aliasing. Logs: `.scratch/hs343/` on the development machine (HOTSTEP-343).

Verify they are present: `powershell -File engine\verify-hooks.ps1` (hooks 7-18, see the table above). **Hooks 9 and 10 matter most among the silent ones**: losing 9 leaves the numbers right and only the clock wrong, and losing 10 makes every sub-`q8_0` base vanish from the trainer rather than error.
