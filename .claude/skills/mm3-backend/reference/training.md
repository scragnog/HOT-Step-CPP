# Training and runtime adapters

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

## Native codes export: `ace-train mm3-codes` (SHIPPED 2026-08-20)

Audio -> RVQ codes, natively. `engine/src/minimax/mm3-rvq-encode.h` ports
PurpleOrc's V4Encoder (169M: conv stack + 3 dilated ResBlocks + frame pooling +
8 pre-LN transformer layers + causal depth decoder). The graph is
**weights-agnostic** — the whole community encoder lineage shares this
architecture, so a new checkpoint is a `engine/tools/convert-rvq-encoder.py`
run (arch `mm3rvq`, verbatim PyTorch tensor names), never a code change.
Adopted checkpoint: `models/mm3/mm3-rvq-53kpooled-f32.gguf`.

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
   line + SimpleTuner LM mode; Skiba timbre reached 2026-08-20 with r256 attn+MLP).
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
