# The HOT-ggml fork

`engine/ggml` is a submodule pinned to the `hot-step` branch of **HOT-ggml**, a fork of
[ggml-org/ggml](https://github.com/ggml-org/ggml). The branch is upstream master plus
HOT-Step's ggml changes, one commit each, with the rationale in every commit message. A
normal checkout and build leave `engine/ggml` clean; there is no patch step. Until
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
| mm-backward | `GGML_BACKWARD_MM=1` mul_mat formulation of the activation gradient | 8 |
| quant-cpy-kquant | quant → F32 `CPY` for every type the dequantizers cover | 10 |
| sched-unplaced-log | name the node no backend accepts | — |
| zz-yue2-convrot8 | `CONVROT8` / `_BACK`, CUDA int8 rotated-linear training op | 15 |
| zzz-yue2-bf16-round | `BF16_ROUND` unary op (CPU + CUDA) | 16 |
| zzzz-vulkan-train-ops | Vulkan `BF16_ROUND` and fused training attention | 14 |
| ggml-rpc patch-level check | refuse an RPC server that lacks the fork's ops | — |

`cpy-q-occupancy` (the quant → F32 copy launch geometry) is upstream since ggml-org
`b64fb805`; Hook 9 checks the launch shape itself. Hook 17 checks that `engine/ggml` is clean
and at the pinned commit.

## Changing ggml

Make the change on HOT-ggml's `hot-step` branch as its own commit, with the rationale in the
message, then move the gitlink in this repo. Never leave edits in `engine/ggml`: they are not
what CI or users build, and the next submodule update discards them. To take a newer ggml-org
master, rebase or replay `hot-step` onto it, rebuild, and rerun the parity checks the
original changes were measured with (below).

If `engine/ggml` is a stock ggml-org checkout, the submodule URL is cached from before the
fork. From the repo root: `git submodule sync -- engine/ggml`, then
`git submodule update --init engine/ggml`. `engine/CMakeLists.txt` refuses to configure
without the fork's ops, and `engine/verify-hooks.ps1` names the missing piece.

## Why each change exists

The sections below are the measurements and reasoning behind each change, kept from the
old patch README.


`bf16-out-prod` teaches ggml-cuda's `out_prod` to accept a **BF16 `src0`**, which is what lets the DiT trainer's `--mirror bf16` keep frozen trainable-layer weights in BF16 instead of promoting the whole mirror to F32.

`mm-backward` adds an **env-gated alternative formulation** for the `MUL_MAT` activation gradient in `ggml.c`'s `ggml_compute_backward`. Upstream emits `out_prod(src0, transpose(grad))`, and ggml-cuda's `OUT_PROD` is F32-only — which forces the frozen weight to F32 and drags the *forward* `mul_mat` onto TF32 tensor cores too. With `GGML_BACKWARD_MM=1` set, the backward becomes `mul_mat(cont(transpose(src0)), grad)` instead: mathematically and shape-wise identical, but dtype-agnostic, so a BF16 weight rides real BF16 tensor cores in both directions with no dequant. Measured on an RTX 5090 at ~1.7–1.8× per layer per step. With the env var unset the emitted graph is byte-identical to upstream.

The same change carries a second, narrower rule added for the DiT trainer's third mirror mode, `--mirror bf16-f32` (2026-09-02). That mode stores frozen trainable-layer weights as BF16 and promotes each one to F32 with an in-graph `ggml_cast` at its `mul_mat` site, so the GEMM is f32 while residency stays bf16 — the fix for adapters that render coarse under plain `--mirror bf16`. It only works if the cast's F32 output dies with the forward matmul, and the `mm` arm above would have kept it alive: naming `src0` in a backward node makes ggml-alloc hold all 32 layers' F32 copies from the forward pass until each layer's backward runs, which is precisely the ~8 GB the mode exists not to spend. **The failure would have been silent** — correct gradients, correct loss, and a peak footprint back at the F32 mirror's.

So when `src0` is exactly that node — a `ggml_cast` (a `CPY` with the self-referencing `src[1]`) to F32 off a gradient-free, non-parameter BF16 *leaf* — the backward re-casts the leaf instead of referencing the forward's cast. Same value (BF16→F32 is exact), its own short-lived node, and the forward copy is reclaimed the moment its matmul is done. Nothing else in the engine matches that shape, and the `out_prod` fallback arm is deliberately left alone. Measured on one training dataset (32-layer XL base, LoKR dim 512 + MLP, `--attn flash --bwd mm --optimizer muon`, 12 same-seed epochs, crop pinned 552): mirror 8133 MB against f32's 15893 MB, trainer-owned peak 14370 MB against bf16's 14190 MB and f32's 21997 MB — i.e. the transient window costs 180 MB, not 7.8 GB — and `loss`/`ma5`/`raw`/`gnorm` came out **bit-identical to `--mirror f32` at every one of the twelve epochs** — equal to the last digit of the full-precision `dit_train_log.json`, and the two runs' exported `lokr_weights.safetensors` have the same MD5 — where `--mirror bf16` diverged from the first epoch and reached 7.8e-3 by the twelfth. Unpinned, the flash auto-fit picks crop 1542 against f32's 610 and bf16's 1632. The cast is not free: 3.21 s/epoch against f32's 2.56 and bf16's 2.17, i.e. +25% over f32 at equal crop — which the 2.5× crop repays.

`cpy-q-occupancy` fixed the **launch geometry of ggml-cuda's quant→F32 copies**: upstream launched them one CUDA block per element with one thread inside, so 31 of every 32 blocks only hit a bounds guard. Measured on a 5090, MM3 8.6B q8_0 training: **11.1 → 3.75 s/step**, parity with f16. The same fix landed upstream as ggml-org `b64fb805`, so the fork no longer carries it. The failure mode was silent (right numbers, wrong clock), which is why Hook 9 still checks the launch shape.

`ace-train`'s `--bwd <outprod|mm>` sets that env var; the Training Studio defaults both trainers to `mm`.

The `mm` arm also dequantizes a **block-quantized `src0`** to F32 before transposing it (#197). A quantized weight's blocks run along `ne0`, so `cont(transpose(W))` is a CPY that no backend implements: CUDA aborts in `ggml_cuda_cpy` ("q8_0 to q8_0"), and Vulkan and the CPU crash with an access violation. It only bites trainers that feed a quantized weight straight into `mul_mat`, which is `yue2-nar-train` (MM3 LM already casts in-graph, so its backward only ever sees F32). Before the fix, a YuE2 NAR run on any non-BF16 base died on its first backward, on every backend. After it, q8_0, Q4_K_M and IQ3_XXS train on CUDA at the bf16 step time (0.44 s/step, 5090) with first-step loss within 3 % of bf16, and on Vulkan at about 2.4 s/step. The cast is transient, like the `--mirror bf16-f32` re-cast above.

`zzzz-vulkan-train-ops` gives ggml-vulkan the two custom ops the YuE2 joint trainer cannot run without, so it trains on a Vulkan build (#197). The trainer computes each layer with `ggml_backend_graph_compute` straight on one backend, no scheduler, so an op the backend lacks is a refusal, not a CPU fallback.

- **`BF16_ROUND`**: one more entry in the generic `unary.comp`, round-to-nearest-even with NaN kept quiet, bit-identical to the CPU's `GGML_FP32_TO_BF16`.
- **`FLASH_ATTN_TRAIN` / `_BACK`**: three new kernels (`fa_train_fwd`, `fa_train_bwd_dq`, `fa_train_bwd_dkv`) implementing the CPU oracle's maths and layouts: online softmax with LSE, then dQ per query row and dK/dV per key row recomputing P from the LSE. One 32-lane cluster per row, reduced with `subgroupShuffleXor`, so they need 32-wide subgroups with shuffle; without them `supports_op` says no and the trainer uses exact attention. `fattn-train-test --backend cuda` (which just asks for the first GPU) passes 49/49 against the CPU with `--extra` on an RTX 5090 and a 4090 over Vulkan, including GQA, S ≠ S_kv, dead key columns and bitwise determinism. The kernels are plain f32 scalar code: correct, several times slower than CUDA's.
- **Dispatches are sliced** to about 2^25 (row, partner, head) pairs each (`GGML_VK_FA_TRAIN_SLICE_PAIRS` overrides, for tests). One dispatch over a whole song is seconds of GPU time, which freezes the desktop on a GPU that also drives a display and can trip the Windows TDR reset. The parity grid passes at the smallest slices too.

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

Verify they are present: `powershell -File engine\verify-hooks.ps1` (hooks 7-17, see the table above). **Hooks 9 and 10 matter most among the silent ones**: losing 9 leaves the numbers right and only the clock wrong, and losing 10 makes every sub-`q8_0` base vanish from the trainer rather than error.
