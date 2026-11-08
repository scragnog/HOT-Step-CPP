# Training and runtime adapters

Resolve runtime paths first using [Finding the data folder](../../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

> Reference for the `mm3-backend` skill. Read only when the task needs it.

## Runtime LM adapters (SHIPPED 2026-08-20 — engine + server + UI)

`engine/src/minimax/mm3-lm-adapter.h` loads PEFT LM LoRAs (SimpleTuner
`language_model.`-prefixed checkpoints, q/k/v/o + gate/up/down × 36 layers;
since 2026-09-09 also the PiSSA DELTA form, `hot_step.param_method = 4`, which
needs the residual file beside the resident base — the loader takes the base
path as its third argument and refuses with the file's name when it is
missing; see `engine/src/pissa-residual.h`)
and applies them as RUNTIME low-rank deltas in the AR stage — base weights
untouched, so per-group scales are live per generation. Wire fields on
`POST /mm3/synth`: `lm_adapter` (abs path), `lm_adapter_scale{,_attn,_mlp,
_early,_mid,_late}` (all default 1.0; range ±4). Scales bake into cached
graphs as constants — `mm3_lm_set_adapter` invalidates the slots on change
(once per generation, cheap). Single-slot cache keyed (path, mtime), guarded
by g_mm3_mutex; dropped on /mm3/unload, transient release, and the staged
after_ar handover (VRAM). Load failure FAILS THE JOB — never silently base.
Server: `backends/minimax/lmAdapter.ts` (containment-checked refs under
`<adapters root>/mm3-lm-adapters/`, sidecar metadata, `params.mm3LmAdapter*`),
defaults = the ablation-validated attention 1.0 / MLP 0.5. Catalogue route
`GET /api/mm3/lm-adapters` (adapters + sidecars + default scales). UI:
`ui/src/components/global-bar/Mm3LmAdapterDropdown.tsx` renders in the Adapters
cluster whenever `capabilities().features.lmAdapters` is true and
`features.adapters` is false — its own flag, because `adapters: true` gates
ACE's whole DiT-stack UI (merge/runtime modes, per-section masking, trigger
embedding). Picker prefills the scale dials from the picked adapter's sidecar
`recommendedScales`; depth thirds sit behind an advanced disclosure. Values
ride `backendParams` → `getGlobalParams()`, so a new dial needs no store field.
**Two application modes since 2026-08-21** (`lm_adapter_mode`, UI toggle,
`params.mm3LmAdapterMode`): `runtime` (default; live dials, measured +28 %
LM step on f16 and **+51 % on q8_0** — the fixed overhead looms larger over
halved streaming) and `merge` (mm3-lm-merge.h folds scale·B·A into the
resident weights: merged step == base step, 15 s once per (adapter, dials)
on q8_0 via dequant+delta+requant — IQ types refused, need imatrix).
`MM3Model::lm_merge_tag` tracks what is baked in; mismatch forces a pristine
LM reload (verified live under keep-loaded); staged mode reloads per gen
anyway. Merge failure part-way drops LM residency — never leaves mixed
weights serving.
**DoRA LM adapters** (PEFT `lora_magnitude_vector`) run through `mm3-lm-dora.h`, a one-time
column-norm pass for runtime mode; a file with a partial set of magnitudes is refused. The
dial renorm that lets DoRA follow the six scale dials is opt-in: request field
`adapter_runtime_dora: true` (the same key ACE reads, `mm3-request.h:1110-1121`), sent by the
server only when `params.adapterRuntimeDora === true`.
**Writing a checkpoint + a JSON sidecar into `<adapters>/mm3-lm-adapters/<run>/`
is the entire publish step** — that is the contract the future native trainer
targets (docs/plans/2026-08-20-mm3-training-server-design.md §5). r256 ≈ 1.4 GB
resident, ≈ +9 % AR cost (unmeasurable at short lengths). Smoke-validated:
252 modules load, base + adapter renders complete same wall time.

## Native LM LoRA training: `ace-train mm3-lm-train` (SHIPPED 2026-08-20)

MM3's LM is Qwen3-8B, and `train/lm-graph.h` was already a trainable
cache-free unfused Qwen3 forward — so the trainer is a RETARGET, not a build.
`train/mm3-lm-load.h` does the load side (llama.cpp tensor names, `qwen3.*` KV,
UNTIED head); `train/mm3-lm-train-run.h` does the data path, the frame
embedding, the output slice and the loop. AdamW, **Muon**, PEFT export,
checkpoint/resume all come from the existing ACE machinery.

Validation ladder, all runnable:
- `ace-train mm3-lm-probe` — the trainer's forward vs the engine's own prefill
  (cos 0.999999951, argmax identical).
- `ace-train mm3-lm-loss` — teacher-forced CE with falsification diagnostics
  (`--target-shift`, `--no-prompt`).
- `ace-train mm3-lm-train --fd-check N --f32-layers 2` — **the gradient gate,
  and the one to run after touching anything in the backward.** Exit code 0/1,
  two independent bars: checkpointed-vs-naive gradients < 2e-3 (measures
  3.78e-07) and finite differences < 2e-2 (measures 0.0023). Both verified by
  injected faults, not just by passing — see the header of
  `engine/src/train/mm3-lm-train-run.h`.
  **`--f32-layers` is not optional if you want an answer.** Without it, f16
  rounding across 36 layers is larger than the defect being looked for: an
  injected off-by-one moved the number from 7.70e-02 to 2.14e+00 and the
  command still exited 0. It truncates to 2 layers and mirrors them plus the
  scored head slice to F32 (~1.7 GB; a full 8.6B F32 mirror would be ~34 GB).
  Run it against **f16 even when training on q8_0** — isolating a quantized
  base would measure the quantizer, and `mm3_f32_isolate()` refuses it.

### Training bases: any installed quant, and the VRAM ladder

Since `quant-cpy-kquant.patch` the trainable base is **whatever is installed**,
not just f16/q8_0 — every K-quant, MXFP4, NVFP4 and IQ type. The mechanism is
QLoRA-style dequantize-per-matmul: `qwen3_f32()` emits `ggml_cast` on the frozen
weight, so the backward only ever sees the cast's F32 output.

Peak VRAM, MM3 8.6B, rank 256 / 1500 frames, **all within ~5 % of the same step
time** (~3.8 s/step on a 5090):

| base | size | peak | 1st-step loss vs f16 |
|---|---|---|---|
| f16 | 16.0 GB | 31.4 GB | reference |
| q8_0 | 8.5 GB | 22.6 GB | +0.02 % |
| Q6_K | 6.6 GB | 20.7 GB | +0.3 % |
| Q4_K_M | 5.1 GB | 19.2 GB | +0.8 % |
| MXFP4 | 5.1 GB | 19.1 GB | +2.7 % |
| Q2_K | 3.4 GB | 17.5 GB | **+14.3 % — do not train against this** |

**Rank is the bigger lever below 20 GB**, at 31.2 MB per unit: Q4_K_M peaks
19.2 GB at r256/1500, 13.2 at r64/1500, 11.1 at r32/750, 10.2 at r16/500.
~10 GB is the practical floor for MM3 LM training.

`estimateMm3PeakMb()` in `services/training/mm3Train.ts` predicts all of this to
<0.3 %, and `recommendMm3Config()` picks base + rank for the detected card
(read from the engine's `/vram`). **f16 is never recommended**: it measures
fidelity-equivalent to q8_0 at twice the VRAM, and at 1.5 GB free it pages over
WDDM — measured at 12–14 s/step against q8_0's 3.75 on the same 12 steps.

### Four things that cost real time here

1. **THE PROMPT DOMINATES THE SEQUENCE.** An MM3 prompt is ~1,125 tokens, so
   `--max-frames 128` still gives S=1253. Shrinking the crop is NOT the VRAM
   dial you expect, and per-layer gradient checkpointing is load-bearing rather
   than an optimisation. Naive fwd+bwd retains ~18 GB of activations on top of
   a 16 GB f16 base and spills into WDDM shared memory: 38 s/step where the
   same forward alone takes 644 ms.
2. **AT r256 ON A 32 GB CARD, MUON FITS AND ADAMW DOES NOT.** AdamW keeps two
   momentum buffers where Muon keeps one — +2.66 GB at rank 256 on a config
   already peaking at 31.7 GB. Lowering `--max-frames` does NOT rescue it (the
   crop only moves the ~0.6 GB of checkpoint buffers). Muon here is the
   optimizer that runs, not just the one that might train better.
3. **Grad clipping is nearly a no-op for Muon params.** Newton-Schulz
   Frobenius-normalises its input, so clipping only rescales a direction that
   is renormalised anyway. Do not read the clip figures as a tuning signal;
   `--muon-lr-scale` is the knob, and Muon's LR does not mean what AdamW's
   means.
4. **lm-ckpt.h was EXTENDED, not forked**, with two hooks that are inert by
   default (`LmCkptCfg::{head_w,head_row0,head_v}` for an untied scored head;
   `LmCkptRun::embed_build` for the frame-embedding entry). An ACE run emits
   byte-identical graphs. Keep it that way — that file is what makes ACE 4B
   training fit.

### Where a whole-song step goes (measured 2026-11-06)

The in-app recipe (q8_0, `--weights f32-window`, HOT-PiZZA r16, `--attn flash`,
whole-song crops, S up to 7,178), 20 steps on a 13-track album, RTX 5090:

- **Phase timer** (`HOTSTEP_LM_PHASE_TIMER=1`, prints a `[phase0]` line every
  20 micro-steps): 6.55 s per micro-step, 94.5 % GPU (`gpu_sync`), 4.2 % host
  graph build and input uploads, 1.2 % kernel submit, 0.05 % planning. About
  106 graphs per step. Since 2026-11-06 the clock starts before the mask and
  token uploads, so `build(host)` really includes them.
- **Kernels** (Nsight Systems, 8 steps, 51.3 s of kernel time): GEMM 63.7 %,
  flash attention 24.1 % (fwd 7.5, dQ 7.7, dK/dV 8.8), elementwise 9.0 % (mostly
  `add`/`sub` over `[out, S]`), q8_0 to F32 weight copies 3.1 %. **Every big
  GEMM is `cutlass_80_simt_sgemm`: plain FP32 SIMT, no tensor cores.** That is
  57.8 % of kernel time, and the place to look for a BF16 or TF32 win.
  Adapter-sized GEMMs (small tiles, all four LoRA/PiSSA products forward and
  backward) are 4.8 %.

**The causal mask was rebuilt twice per step.** Whole-song crops change S on
almost every step, and the MM3 loop and `lm_ckpt_upload_mask` each kept their
own last-S cache over the same buffer, so each one built the S x S mask in F32,
converted it to F16 one element at a time, and uploaded it. That took about
104 ms of host time per build at S = 7,178. Now it is built once, with
row fills straight in the buffer's type (`lm_mask_set_causal`, `lm-common.h`),
and the loop marks the checkpointed path's cache. A fill takes 10.5 ms. Bytes
are unchanged (`engine/tests/test-lm-causal-mask.cpp`). Two interleaved
20-step pairs gave identical losses, gradient norms and resume state; the patched
binary was faster on 20 of 20 steps in both pairs, by 1.4 % and 3.2 % in
total (about 90 ms a step, more on longer songs).

**Flash still needs the full mask.** The fused op's causal hint
(`ggml_flash_attn_train_set_causal`) is only read by the BF16 CUDA kernels and
Metal, and is ignored when no mask is attached. The TF32 kernels the recipe
runs read the mask, and CUDA requires it contiguous, so a strided view of one
S_max mask is not an option without a fork change.

**Stacking [A; A0] into one matmul was measured and not built.** HOT-PiZZA's
frozen branch reads x a second time per site, but all adapter-sized GEMMs
together are 4.8 % of kernel time, so merging the A-side read would save
under 1 %, below the 3 % bar for kernel churn. The larger cost of that branch
is the `[out, S]` output of `B0 (A0 x)` and the `sub` after it. Merging that
needs a differentiable concat, and ggml's `CONCAT` has no backward.

**Why the GEMMs were SIMT: `NVIDIA_TF32_OVERRIDE=0`.** ggml-cuda puts every
cuBLAS handle in TF32 mode, but `mm3_lm_train_main` sets the override before
the backend starts (TF32 was judged "a few percent"), so every F32 GEMM ran
`cutlass_80_simt_sgemm`. The override does not reach the custom `fa_train`
TF32 kernels or BF16 `GemmEx`.

**Three opt-in levers (2026-11-06), all off by default.** Same recipe, 20
steps, seed 42, app shut down, no Unreal. ms is the phase timer's
per-micro-step total; VRAM is device peak minus idle; loss deltas are per step
against the f32-window run:

| flags | ms/step | speedup | peak VRAM | max / mean loss delta |
|---|---|---|---|---|
| f32-window, flash (in-app today) | 6,643 | 1.00x | 19,105 MB | (losses identical to the pre-change binary) |
| `--tf32 on` | 4,791 | 1.39x | 19,054 MB | 0.29 % / 0.06 % |
| `--attn flash-bf16` | 5,554 | 1.19x | 18,959 MB | 0.84 % / 0.12 % |
| `--weights bf16` | 4,043 | 1.65x | 19,200 MB | 0.32 % / 0.07 % |
| `--weights bf16 --attn flash-bf16` | 2,861 | 2.32x | 19,163 MB | 0.57 % / 0.08 % |
| the two plus `--tf32 on` | 2,821 | 2.35x | 19,214 MB | 0.80 % / 0.11 % |

- `--weights bf16` now takes any base. `lm_linear` casts a q8_0 (or other
  quant, or F16) projection to BF16 in-graph (q8_0 goes through F32: the fork
  has no direct quant-to-BF16 copy), and Lever A's rewrite is keyed by the
  cast. The S18 tripwire (7 rewrites per segment) still holds.
- `--attn flash-bf16` is the YuE2 trainer's attention: Q/K/V cast to BF16,
  `GGML_PREC_BF16`, and causal hint 0 when there is no prefix, so the BF16
  kernels skip reading the mask.
- After both (Nsight, 8 steps): kernel time 51.3 s to 22.9 s. BF16
  tensor-core GEMMs (`cutlass_80_tensorop_s16816gemm_bf16`) 41.6 %,
  elementwise 17.9 %, FP32 SIMT left (adapter, head, depth) 9.9 %, F32 to BF16
  and other copies 9.7 %, q8_0 to F32 7.5 %, flash 10.7 %. The phase timer
  then read host build at 9.7 % of the step, but most of that was the
  acoustic-loss head's GPU time, which ran outside the timer (next block).
  Next candidate: a direct q8_0 to BF16 copy in HOT-ggml (most of the 17 % of
  casts).

**Later the same night: the host share was the depth head.** Per-phase
timers inside the micro-step (BF16 levers, 20 steps): forward 795 ms, CE head
85, acoustic-loss head 198, backward segments 1,736, and building all 36
backward graphs only 2.5 ms. The acoustic-loss head (`mm3_depth_train_head`)
called `ggml_backend_sched_graph_compute` directly, so its 161 ms of GPU work
landed in `build(host)`. It now counts as `gpu_sync`; real host time is about
85 ms a step (3 %), and an overlap that built graph i+1 while graph i ran
measured 0.3 %, so it was dropped. Graph reuse and Lever B are closed for MM3.
The head itself was slow for a fixable reason: its frozen-weight GEMMs took
`[in, 8, K]` activations, which ggml-cuda runs as K batched 8-column products
(`mul_mat_f` forward, cuBLAS `gemmSN_NN` backward, 5.3 % of kernel time
together), re-reading each F32 weight per frame. As one 2D GEMM the head takes
65 ms instead of 161: 2.85 to 2.70 s a step under the levers, 6.69 to 6.54 s
with the default flags. Not bit-identical (summation order), loss within
0.18 % over 20 steps, the head's fd tripwire passes. Also bit-identical and
small: HOT-PiZZA's frozen branch now adds `B0 (-s A0 x)` instead of
subtracting, which drops a NEG over `[out, S]` per site from the backward
(1.2 % of kernel time).

**q8_0 to BF16 without the F32 copy.** Under `--weights bf16` on q8_0, the
in-graph weight cast (q8_0 to F32 to BF16, 12.5 % of kernel time) is replaced
by a one-layer BF16 stage that an engine-side kernel (`q8_bf16_cuda.cu`,
compiled into `yue2-aitk-kernels`) fills before each layer graph, with the
same arithmetic as the two ggml copies. Bit-identical 20-step losses; 2.70 to
2.48 s a step; +378 MB peak VRAM (the stage itself). Together with the depth
head fix, the whole-song step under `--weights bf16 --attn flash-bf16` went
from 2.85 to 2.48 s (1.15x) tonight, 2.68x against the f32-window default.
A direct q8_0 to BF16 copy in HOT-ggml would give the same win in-graph, but
it is no longer needed for this path.

**`--pissa-fold on` (opt-in, not heard).** The stage kernel can also subtract
HOT-PiZZA's frozen term, writing `bf16(W - s B0 A0)`, and the graph then drops
the frozen branch (two adapter GEMMs, an `[out, S]` add per site and their
backward). 2.47 to 2.25 s a step (1.10x on top of everything above; removing
the branch outright measured 2.19 s, so the fold kernel costs about 30 ms).
Not bit-identical: against the f32-window run the 20-step loss is within
0.34 % (the unfolded BF16 run: 0.65 %). Needs an A/B by ear before any default.

After tonight, whole-song micro-step on the 5090: default 6.54 s, `--weights
bf16` 3.71 s, plus `--attn flash-bf16` 2.48 s, plus `--pissa-fold on` 2.25 s,
plus `--tf32 on` 2.20 s (3.0x the 6.64 s default at the start of the night;
loss within 0.25 % of the f32-window run over 20 steps).
Kernel time under the last (Nsight, 8 steps, 2.18 s a step): BF16
tensor-core GEMMs 47 %, elementwise, norms and RoPE 21 % (mostly the residual
and adapter adds and gradient accumulation, which cannot go without changing
the summation order), flash attention 15 %, FP32 GEMMs 10 % (the rank-16
adapter products and the depth head), copies and casts 7 % (activation casts
to BF16 inside ggml's BF16 matmul, Q/K/V casts, the transposed weights, the
stage). The adds run at about 0.8 TB/s against the card's 1.8
(`k_bin_bcast_unravel`, one element per thread), so a contiguous vectorised
add in HOT-ggml is the next lead: up to about 7 % of the step, a fork change.

Gates. `mm3-lm-probe` on f16 (it now takes `--weights` and `--attn`) passes
every arm; `--weights bf16 --attn flash-bf16` gives cos 0.999999898, rel
9.5e-4 against the 1e-3 bar (f32-window exact: 0.999999951, 3.5e-4). On q8_0
every arm misses the rel bar, the unchanged one included (4.5e-3), so probe on
f16. `--fd-check 8 --f32-layers 2`: the F32-isolated layers take the shipped
path, so it cannot test `--weights bf16`. With `--attn flash-bf16` the route
check passes (2.75e-4) but finite differences fail on the q_proj probes
(worst 0.64 at the default eps, 0.068 at `--fd-eps 0.2`, where TF32 flash
gives 0.002). The analytic gradients match TF32 flash within 0.15 % on all 8
probes, so the BF16-rounded forward is too coarse for central differences on
attention-side factors.

Matched 150-step adapters (same recipe and seed): f32-window 1,024 s,
`--weights bf16` 625 s (1.64x), plus `--attn flash-bf16` 440 s (2.34x).
Per-step losses drift apart over 150 steps (mean 2 %, single steps up to 20 %),
but held-out loss at step 150 is 3.026 / 3.033 / 3.044. Two 60 s renders per
adapter are staged under
`D:\Ace-Step-Latest\_experiments\_LISTENING\2026-11-06-mm3-speed\bf16-train\`
for the ear test. Heard 2026-11-06: no audible loss at either level, so the
app's `trainPrecision` default is `fastest` (`--weights bf16 --attn
flash-bf16`). The CLI flags themselves stay off by default. Since 2026-11-08 the
app's `fast` and `fastest` also pass `--tf32 on`; `full` does not.

## Native codes export: `ace-train mm3-codes` (SHIPPED 2026-08-20)

Audio -> RVQ codes, natively. `engine/src/minimax/mm3-rvq-encode.h` ports
PurpleOrc's V4Encoder (169M: conv stack + 3 dilated ResBlocks + frame pooling +
8 pre-LN transformer layers + causal depth decoder). The graph is
**weights-agnostic** — the whole community encoder lineage shares this
architecture, so a new checkpoint is a `engine/tools/convert-rvq-encoder.py`
run (arch `mm3rvq`, verbatim PyTorch tensor names), never a code change.
Adopted checkpoint: `<data root>/models/mm3/mm3-rvq-53kpooled-f32.gguf`.

```
ace-train mm3-codes --dataset <ds.json> --rvq <mm3-rvq-*.gguf> --enc <mm3-enc-*.gguf> --out <dir>
ace-train mm3-codes --rvq <mm3-rvq-*.gguf> --fixture <f.fix>     # standing gate, no audio needed
```

Four traps, all pinned in the header because each is SILENT if wrong:
1. **GELU must be `ggml_gelu_erf`**, not `ggml_gelu` — torch's default is
   erf-exact, ggml_gelu is the tanh approximation, ~1e-3 apart, against a
   semantic argmax margin whose p05 is 0.06.
2. **`GroupNorm(1, C)` reduces over channels AND positions jointly**, not per
   position — a [T,1,C,1] reshape into `ggml_group_norm(n_groups=1)`.
3. **The encoder stack is pre-LN with NO final norm**; `norm_out` is applied by
   V4Encoder outside `nn.TransformerEncoder`.
4. **Windowing is the model's own `frame_latent_starts`**, integer for integer,
   constants carried in the GGUF; windows are FIRST-WINS.

**Parity, and the lesson in how to gate an argmax port**: the fixture rung is
exact (feats/logits cos=1.000000000, 0/128 semantic codes differ). End-to-end
against the python exporter over 13 tracks: row counts identical, semantic
99.910% identical, acoustic 99.412%. The residual is f32 GEMM ordering, MEASURED
not assumed — the reference encoder run on OUR latents shows the same ~0.05%
(so it is not the DAV port), and every flipped frame sits at a reference argmax
margin of 0.0002-0.0028 against an all-frame median of 1.05. **"Bit-exact codes"
is the wrong unit gate for any argmax port**; split flips by margin, which the
fixture mode does.

## Training: DiT yes, LM ~~never~~ — SUPERSEDED (see below), assessed 2026-08-14

Full teardown + staged plan: `docs/plans/2026-08-14-mm3-training-feasibility.md` (local).

**MM3 ships decode-side only for the audio-token path — there is no audio → code encoder.**
Verified from the HF tensor inventory, not inferred: `qwen_7B/` (18.5 GB) is the **LM +
RVQ depth decoder in MiniMax's original `AbabForCausalLM` format** (`model.layers.*` +
`model.audio_decoder.*` with 7 audio heads, `embed_tokens [200000,4096]`), matching
`language_model/` + `rvq_depth_decoder/` to ~5 MB. It is **NOT a music tokenizer** — an earlier
note here said it was, which wrongly made LM training look possible. Consequences:

1. **LM / planner adapters are BLOCKED.** ← **SUPERSEDED 2026-08-17+**: the community
   audio→codes encoders (SimpleTuner v4 lineage → PurpleOrc 53k → Mothersuperior
   53k-pooled, best-by-ear 2026-08-20) provide the codes; LM code-SFT WORKS (our lm_sft
   line + SimpleTuner LM mode; the target vocalist's timbre was reached 2026-08-20 with r256 attn+MLP).
   The native-GGML LM trainer is the open build (docs/plans/2026-08-20-mm3-training-studio.md).
2. **DiT conditioning cannot be teacher-forced.** The only way to get `frame_hiddens` for a
   training clip is to AR-sample the LM from its caption/lyrics — so condition and target come
   from *different music*, matching only in caption/lyrics/duration. A timbre/production adapter
   can still work (low sigma is well-posed); the risk is the DiT learning to ignore
   `encoder_hidden_states`, i.e. prompt/lyric adherence. Measure it with the LRC probe above.

The encoder that IS needed and IS released: **`dav.pth` (492 MB) — the DAV encoder**, audio →
128-ch latents @ 86.13 Hz, L/R through shared weights, **mean only** (`logs_proj` unused). Our
local `comfy/vae/minimax_music3_dav.safetensors` is **decoder-only**, so it is a new download.

SimpleTuner ([#3074](https://github.com/bghira/SimpleTuner/pull/3074)/#3075, merged 2026-08-14)
trains the 2.4B flow DiT only, targets `to_q/to_k/to_v/to_out.0/ff_in/ff_out/proj_in/proj_out`,
plain rectified-flow loss, and defines the de-facto **ComfyUI MM3 LoRA format**. No published
evidence anyone has trained a *good* MM3 LoRA yet — treat it as a reference implementation, not
as validation. Two traps for loading those LoRAs: we **fuse q,k,v** into `dit.blk.N.attn_qkv`,
and our `mm3.dit.glu_order = value_gate` must be reconciled with their `swiglu_gate_first`
metadata. `convert-mm3.py:build_dit` already resolves both ComfyUI and diffusers names.

Recommended first step is **load, don't train**: add an `mm3` target to the adapter system,
train one LoRA in SimpleTuner, and hear whether it works before building a native trainer.

**Where that stands now (2026-10-23).** Both halves exist, neither is in the app's UI:
- **Load:** `engine/src/minimax/mm3-adapter.h` merges SimpleTuner flow-DiT LoRAs (ComfyUI
  fused-`to_qkv` form or diffusers q/k/v form, SwiGLU order read from the
  `swiglu_gate_first` metadata, DoRA magnitudes included) into the DiT at load time. The
  adapter set comes from the environment only, `MM3_ADAPTER` (paths separated by `;` or `:`)
  and `MM3_ADAPTER_SCALE` (`mm3_apply_adapters`, `mm3-model.h`), so it is fixed for the
  process lifetime; a request field is the planned next increment. LoKr, basin re-base and
  runtime mode are out of scope there.
- **Train:** `ace-train mm3-train-dit --cache <dir from mm3-condition> --models <dir> --out <dir>`
  trains a flow-DiT LoRA natively (DoRA since 68b81a30), fed by `mm3-preprocess` (dataset →
  target latents) and `mm3-condition` (AR rollout → conditioning cache). The Training Studio
  does not drive it.
