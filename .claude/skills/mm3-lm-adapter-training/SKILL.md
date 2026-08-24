---
name: mm3-lm-adapter-training
description: The validated recipe for training MiniMax-Music3 planner-LM style adapters (artist/album clones) with ace-train mm3-lm-train and the Training Studio. Use when training an MM3 LM LoRA, choosing rank/optimizer/steps, picking which checkpoint to ship, diagnosing "the adapter barely works" or "it sounds overcooked", or setting Training Studio defaults.
---

# MM3 LM Adapter Training — the working recipe

Goal this recipe is tuned for: an **album clone**. You want a generation that
could be another track off that record. Memorisation is acceptable and in fact
desirable; style bleed is acceptable. This is NOT tuned for a surgical trigger
LoRA that leaves the base model otherwise untouched.

Established over a 6-album, 30,000-step sweep on 2026-08-23/24 (Green Day,
System Of A Down, Lagwagon, White Stripes, Outkast, Alkaline Trio), each album
laddered by ear across 20 checkpoints. Working log: `docs/plans/
2026-08-23-mm3-soad-style-adapter-findings.md` *(gitignored, local only)*.

## THE CURRENT BEST RECIPE

Keep this block up to date — it is the thing the in-app trainer should
eventually be dialled to, and `docs/plans/` is gitignored so nothing there
survives a fresh clone.

```
--rank 128 --alpha 128
--optimizer adamw --lr 8e-5 --lr-end-frac 0.005 --warmup 50
--max-frames 128 --crop-mode random --crop-anchor song
--rank-dropout 0.1
--steps 2500 --save-every 250
--caption-file <one shared caption for the whole album>
--trigger "<artist>" --trigger-prepend
--holdout 0.15 --eval-every 250 --eval-crop 128
```

**At render: MLP scale 0.63-0.75, attention 1.00, global 1.00. Ship ~ck2000.**

Measured: 17.5 GB VRAM, 924 ms/step, ~39 min/album, 1.4 GB per checkpoint.

**This LoRA configuration is FROZEN as of 2026-08-24** — settled by ear over a
rank sweep at 64/128/256 with a full MLP dial at each. Further LoRA tuning is
not planned; the open work is LoKr, Prodigy and the Muon step-size sweep.

Rank 128 was chosen over 256 because it matches by ear at **5.4 GB less VRAM,
7% less step time and half the checkpoint size** (1.4 GB vs 2.7 GB) — which is
what decides whether a six-album overnight sweep fits on disk.

Started from bghira's published SimpleTuner config at rank 64 and moved by ear
over a 6-album sweep plus a rank x optimizer 2x2 (2026-08-23/24).

### The three axes, which are separable

This is the load-bearing insight. Earlier work conflated them and went in
circles.

| axis | what it controls | setting |
|---|---|---|
| **rank** | separating the style from the LM's *language ability* | **128** |
| **MLP dial** (render) | vocal identity vs audio fidelity | **0.63–0.75** *(rank-dependent)* |
| **steps** | how much likeness | 750–2000 |

### The MLP dial is RANK-DEPENDENT — never carry it between adapters

Measured on the same album, same checkpoint, two seeds each:

| rank | 0.50 | 0.75 | 0.85 | 1.00 |
|---|---|---|---|---|
| 128 | coherent, **voice missing** | **best** | worse than 0.75 | voice, degraded |
| 256 | **best** | worse than 0.50 | — | degraded |

The optimum moves UP the dial as rank comes DOWN. That fits total delta
magnitude scaling with rank and the dial scaling it back: less rank needs less
scaling back. So **every shipped adapter needs its own `recommendedScales` in
its sidecar** — a house default of 0.50 would gut the voice on a rank-128
adapter, and 1.00 would degrade a rank-256 one.

- **Rank 64 produces gibberish lyrics.** Fantastic likeness, good audio, but the
  words stop being words. With only 64 directions the adapter commandeers ones
  that also carry linguistic competence. **128 is enough to separate them**;
  256 is not needed for it.
- **Rank 256 alone degrades audio quality** — and the MLP dial fixes it.
  `mm3-lm-adapter.h` already documented why: attention carries the plan/genre,
  the MLPs carry vocal identity AND the fidelity damage. At MLP 0.50 the
  degradation lifts and lyric coherence survives (confirmed on two independent
  seeds). MLP 0.75 was worse than 0.50 on both.
- **The loss cannot see any of this.** r64 and r256 have near-identical held-out
  curves (min 2.536 vs 2.569) and r256 ends *better*. "The lyrics became
  gibberish" is invisible to cross-entropy over codes.

### DO NOT USE MUON at the default scale

`--optimizer muon --muon-lr-scale 64` on MM3 produces an adapter whose B
matrices are **27x larger** than AdamW's, a 31x stronger delta, and at rank 64 it
renders **digital silence** (all 800,000 samples exactly 0, verified — not
quiet, zero). Rank 256 Muon survives but is audibly damaged.

Weights are finite; there is no NaN. The scale is simply wrong: 64 was measured
on the **ACE** planner LM, and Muon's update is normalised, so the scale IS the
step size and does not transfer across models. This is not a verdict on Muon —
it is an untuned hyperparameter. A sweep of {1, 4, 16} at rank 64 is pencilled
in. Until then AdamW is also 21% faster and needs no tuning.

The training log says so before you ever render: Muon's epoch loss ROSE before
falling, mean |grad| was 11.6 vs AdamW's 4.5, and peak |grad| hit 300 vs 14.

### Known confound

Every rank above ran at lr 8e-5, which bghira tuned at **rank 64**. Higher rank
generally wants a different LR, so rank and learning rate are confounded in
these results. Either sweep the LR per rank, or adopt an optimizer that sets its
own (Prodigy — not implemented; `lm-optim.h` branches on a single `want_muon`
bool and only knows adamw/muon).

### Dialling in the in-app trainer

`MM3_LM_DEFAULTS` in `server/src/services/training/mm3Train.ts` is the single
source of truth — the Jobs form is a VIEW of it and must never hold a second
copy. It still carries the pre-sweep recipe. When testing ends:

| field | current | target | why |
|---|---|---|---|
| `rank` / `alpha` | 64 / 64 | **128 / 128** | 64 eats the lyrics; 256 costs 5.4 GB for no audible gain |
| `lr` | 5e-5 | **8e-5** | bghira's published value; ours was stale |
| `steps` | 1000 | **2500** | ear picks land 750–2000; nothing improves after |
| `saveEvery` | 100 | **250** | the ladder is auditioned, so rungs must exist |
| `maxFrames` | 4096 | **128** | random short windows; 4096 is the old regime |
| `cropMode` | `beginning` | **`random`** | pairs with the 128-frame window |
| `evalEvery` | 50 | **250** | eval is only a divergence alarm now |
| *(new)* `evalCrop` | — | **128** | MUST be <= maxFrames or eval silently dies |
| `lrEndFrac` | 0.008 | **0.005** | matches the runs above |
| `optimizer` | adamw | adamw | keep; do not offer Muon until it is retuned |

Render-side default for a rank-128 adapter: **scaleMlp 0.75** (0.63 worth
trying), scaleAttn 1.00. Rank-dependent — see the dial table above.

Anything still under test and NOT ready to promote: LoKr, Prodigy, the Muon
step-size sweep.

## Do this first or nothing works

**Export RVQ codes.** The trainer reads codes, not audio. A dataset that has
only ever been captioned has none, and the failure is a bare "no usable
samples".

```
ace-train mm3-codes --jsonl --dataset <dataset.json> \
  --rvq models/mm3/mm3-rvq-53kpooled-f32.gguf \
  --enc models/mm3/mm3-enc-f16.gguf \
  --out server/data/training/datasets/<slug>/mm3-codes
```

~1 minute for 12 whole tracks. Writes `<out>/codes/<id>.codes`.

## Steps: 750–2000, and the loss will NOT tell you which

The single most important finding. Held-out cross-entropy bottoms out very
early and then rises for the rest of the run — but the checkpoint that actually
*sounds* right is **1–8x later than that minimum**:

| album | held-out min | ear pick | MLP |
|---|---|---|---|
| lagwagon_hoss | 750 | 750 | 1.00 |
| outkast_stankonia | 250 | 750 | 1.00 |
| alk3_thisaddiction | 500 | 2000 | 1.00 |
| whitestripes_elephant | 250 | 1250 | 0.50 |
| soad_toxicity | 250 | 1750 | 0.50 |
| greenday_warning | 250 | 2000 | 1.00 |

**Never pick a checkpoint by held-out loss.** It measures generalisation to
*unseen* songs by the artist; the goal is a clone of the seen ones. Keep the
eval on as a divergence alarm only. Save every 250 and audition the ladder.

Beyond ~2500 steps nothing improved in any album. 5000 steps is ~2x wasted time.

The table above is from the rank-64 sweep. Rank 256 was auditioned at ck1000
and ck2000 only; ck2000 won, and the finer rungs have not been walked at that
rank. Do not assume the rank-64 optimum transfers.

## The two axes: likeness vs coherence

They move in opposite directions with training length, and conflating them is
why "more steps" felt ambiguous for so long:

- **Likeness** (does it sound like the band) rises with steps.
- **Coherence** (does the song hold together, are the vocals articulate) falls
  with steps at MLP 1.0. Symptom: jumbled or "stroke-like" vocals.

The **MLP scale slider is a generation-time knob** and recovers coherence at a
late checkpoint without giving back likeness. It should therefore never cost a
training run — the only training decision is the step count. Default the slider
to 1.0 (4 of 6 albums preferred it); drop to 0.5 if vocals jumble.

## Rank and optimizer — measured, not estimated

All measured on a 32 GB RTX 5090 at `--max-frames 128`, 12-track album,
20-step probes, 2026-08-24. **Every one of these fits with room to spare:**

| rank | optimizer | VRAM | ms/step | 2500 steps |
|---|---|---|---|---|
| 64 | adamw | 14.7 GB | 828 | ~35 min |
| 64 | muon | 14.1 GB | 943 | ~39 min |
| 128 | adamw | 17.4 GB | 811 | ~34 min |
| 128 | muon | 16.1 GB | 988 | ~41 min |
| 256 | adamw | **22.7 GB** | 955 | ~40 min |
| 256 | muon | 20.1 GB | 1281 | ~53 min |

Two things this overturns:

1. **`ace-train --help` says Muon "FITS at r256 (AdamW second momentum buffer
   does not)". That is not true in this regime** — r256 AdamW runs in 22.7 GB.
   The AdamW-over-Muon delta is exactly the extra second-moment buffer
   (measured 2658 MB at r256, predicted 2664 MB), and at 128-frame windows
   there is ample headroom for it. The help text presumably reflects the old
   1500-frame default where activations were far larger; **it needs updating.**
   Consequence: rank and optimizer are *independent* choices here, so a clean
   2x2 comparison is available rather than two confounded packages.
2. **AdamW is faster than Muon at every rank** — by 12% at r64 and 25% at r256.
   Muon's normalised update also makes an AdamW-style LR meaningless, so the two
   are not swappable without re-tuning `--muon-lr-scale` (default 64, chosen as
   best of {1,4,16,64} by measurement).

Muon classifies all 504 LoRA tensors into 34 buckets at every rank tested — a
run that classifies ZERO tensors onto Muon is silently training as AdamW, so
check the `{"type":"optimizer",...}` line reports `"muon":504`.

Rank 64 is the validated default. Higher rank is **untested for quality** — more
capacity should memorise faster, so expect the sweet spot to move EARLIER, and
keep 250-granularity checkpoints rather than assuming 2500 transfers.

## Album-specific behaviour — do not apply a blanket rule

- **White Stripes had a severe rhythm defect at ck500** ("not 4/4"), on both
  seeds, gone by ck1250. That album needed MORE training. A blanket "stop early"
  rule would have shipped the broken one.
- **Outkast was the weakest clone** — likeness immediately, poor coherence at
  every rung. It is also the only album with ~2032-token prompts (4x the others,
  dense rap lyrics) and the most eclectic track list. Treat dense-lyric or
  stylistically scattered albums as harder, not as training failures.
- Corpus size (10–20 tracks) showed **no reliable effect**. An apparent
  "more tracks = later optimum" trend across five albums was flatly contradicted
  by the sixth. Do not plan around it.

## Captions: one shared caption for the whole album

Per-song MOSS captions were the original regime and it "hardly worked". Caption
*constancy* across rows is what binds the style to the prompt.

Write ONE caption, ~60–80 tokens, comma-separated descriptors, **opening with
the trigger** so `--trigger-prepend` is a no-op on the text and only the sidecar
records it. Shape:

```
<artist>, <album> album, <genre>, <guitar/instrument character>, <vocal
character>, <rhythm section>, <production character>, <tempo>, <structure>
```

At generation time the caption should look like the training caption — but
**do not type the trigger yourself.**

### The trigger duplicates at inference if you type it

`ace-train` stamps the trigger into the adapter's safetensors metadata
(`hot_step_trigger`, `hot_step_trigger_position=prepend`), `readAdapterTrigger`
in `server/src/services/adapters/stMetadata.ts` reads it, and
`translateParams.ts` applies it whenever an LM adapter is loaded — including the
MM3 path. **It is applied unconditionally**: `applyTriggers` in
`server/src/services/generation/triggerWords.ts` supports a `skipPresent` option
that drops words the caption already contains, and translateParams calls it
WITHOUT that option. So a caption that already opens with the artist name comes
out as `green day, green day, warning album, ...`, which is not what was trained.

Practical rule: the training caption opens with the trigger, so at generation
**start your caption at the SECOND item** and let the app prepend the first:

```
training caption : green day, warning album, pop punk, bright major-key ...
what you type    :            warning album, pop punk, bright major-key ...
what the LM sees : green day, warning album, pop punk, bright major-key ...
```

Passing `{skipPresent: true}` at the translateParams call site would make this
robust; it is not done today. See
[mm3-captioning](../mm3-captioning/SKILL.md) for the MM3 Structured Caption
format used elsewhere — note this recipe deliberately uses the short
comma-separated style, not the three-section format, for the *training* caption.

## Lyrics shape matters as much as the adapter

MM3 plans the whole track from the whole lyric, and `duration` only truncates.
Write a FULL song's worth of lyrics even for a 40 s render. Tidy four-line
stanzas produce mainstream rock and suppress the style; irregular line lengths
matching the artist's own writing work far better.

## RENDER ADAPTERS ON q8_0 ONLY — the f16 adapter path is broken

**Any LM adapter rendered on the `mm3-lm-f16` base produces garbled, incoherent
audio.** Not subtly worse — "like tuning a radio". Measured by ear, 2026-08-24:

| adapter | render base | result |
|---|---|---|
| none | f16 | fine |
| none | q8_0 | fine |
| trained on q8_0 | q8_0 | coherent |
| trained on q8_0 | **f16** | **garbled** |
| trained on f16 | **f16** | **garbled** (matched, and still broken) |
| trained on f16 | q8_0 | **excellent** |

The matched f16 case being broken is what identifies this. It is not a
train/render mismatch and not adapter portability: the base alone is fine on
both, so **the adapter-apply path on an f16 base is defective**. Root cause not
yet found.

**It is NOT caused by the LoKr work.** Reverting `mm3-lm-adapter.h`,
`mm3-lm-graph.h` and `mm3-lm-merge.h` to their pre-LoKr state and rebuilding
produced **bit-identical** renders on both bases. It is pre-existing, and was
almost certainly never exercised because the app pins q8_0.

### What this means in practice

- The app is unaffected — the Node server selects q8_0.
- Any script that starts its own `ace-server` MUST pin the base first, because a
  bare engine picks the best-quality variant it can find, which is f16:

```
POST /mm3/select-model {"lm": "q8_0"}      # BEFORE any /mm3/synth
```

  Eight comparison renders were silently void before this was noticed. The
  `[MM3] LM:` line in the engine log names the resident variant — check it
  before trusting a render.

### Training base is a QUALITY lever, not just a compatibility one

Training on f16 and rendering on q8_0 produced the best Green Day adapter of the
whole sweep — **at 750 steps, beating the 2000-step q8_0-trained one by ear.**
Unverified beyond one album and one listen, but if it holds it is worth the
extra VRAM: f16 training measured 26.7 GB against q8_0's 17.5 GB at rank 128,
and is slightly FASTER per step (842 ms vs 924) because it skips dequantisation.

### Sample-level identity is not a health check

Two renders differing 100% at sample level are NOT evidence of a bug. Renders
are bit-deterministic for a fixed binary, flags and seed (verified twice here,
and independently in-app), but any numeric perturbation flips one sampled token
in the autoregressive rollout and changes the whole song. Judge by RMS and by
ear. RMS is only a rough guide: `BASE_no_adapter_f16` at 0.1213 was fine while
`f16trained_on_f16` at 0.1187 was garbled.

## Traps

- **`--eval-crop` defaults to 400 and is pinned independently of `--max-frames`.**
  With 128-frame windows every eval crop exceeds `S_max`, all are silently
  skipped, and the run reports an eval plan at startup then never evaluates.
  Always pass `--eval-crop <= --max-frames`; that always fits, because
  `S_max = max_prompt + max_frames` and `max_prompt` already includes holdout.
- **`--trigger` alone only writes the sidecar.** It does not train the trigger.
  `--trigger-prepend` is what injects it into the captions.
- **`--crop-anchor song`** matters: without it every crop is taught as if it were
  the song's opening.
- The milestone `loss` field is a single windowed value, not a mean — it swings
  wildly between adjacent checkpoints. Use the epoch mean.
- Windows file-locks `ace-train.exe` while training; you cannot rebuild the
  engine mid-run.

## Verify a run started correctly

```
grep -E "songs \(|evaluation:|caption begins|VRAM after" <train.log>
```

Expect: sensible train/holdout split, a non-zero eval crop count, the caption
starting with the trigger exactly once, and VRAM inside the card.
