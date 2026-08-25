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
--lm mm3-lm-q8_0.gguf
--rank 128 --alpha 128
--optimizer adamw --lr 8e-5 --lr-end-frac 0.005 --warmup 50
--max-frames 4272 --crop-mode structured --crop-start-frac 0.85 --crop-end-frac 0.15
--crop-anchor song
--rank-dropout 0.1
--steps 2500 --save-every 250
--caption-file <one shared caption for the whole album>
--trigger "<artist>" --trigger-prepend
--holdout 0.15 --eval-every 250 --eval-crop 1024
```

**At render: MLP scale 0.63-0.75, attention 1.00, global 1.00. Ship ~ck2000.**

Est. 29.1 GB VRAM. **The crop settings changed on 2026-08-24 and everything
auditioned before that date was trained at `--max-frames 128 --crop-mode
random` — five seconds per step. Treat pre-2026-08-24 ear results about the
BASE and the OPTIMIZER as void** (see below); the rank and MLP-dial findings
stand, because they were measured against each other under the same broken
crop.

### The crop is the axis that decides whether it sounds like a song

Found 2026-08-24 after every checkpoint of an ADTR run rendered a track that
began part-way through a song and faded out mid-render without resolving.

The crop was **128 frames — 5.12 seconds — against a 204 s median track.** Two
separate defects came out of that, and the second is the one that bites:

**1. Under-coverage.** Measured on the reference dataset (10 tracks, median
5099 frames):

| | |
|---|---|
| crops reaching the track end, the ONLY place EOS is supervised | **2.6%** |
| crops starting at frame 0 | **0.02%** |

So the model was asked at render time to produce an opening and an ending it
had essentially never been shown.

**2. Random crops teach the wrong lesson.** A random crop presents the prompt
followed immediately by mid-song audio *with no history in front of it*, and
supervises it. Every one of those teaches "a song may legitimately begin at
position c0". A render that starts mid-flow, or stops and restarts, is the
model doing what it was trained to do. This framing is ScragBot's and it is
better than "under-coverage" — it explains the fades, which under-coverage
alone does not.

Hence `--crop-mode structured`: 85% of steps anchored at frame 0, so every
supervised position carries the song's real history exactly as it will at
generation time, and 15% flush to the track end, which is the only place EOS
is supervised. `beginning` cannot do the second and `random` cannot do either.

**Whole-track training is not reachable on a 32 GB card.** 9 of 10 reference
tracks exceed 4096 frames and the median would need ~17 GB of attention scores
alone.

### Crop length is QUADRATIC in VRAM, and that is why f16 lost

Peak is `loaded + perRank*rank + 0.2679*S + 0.00044765*S^2 + const`, with
`S = 1142 + frames`. The backward retains `[S, S, heads]` attention scores and
**ggml's flash-attn has no backward**, so there is no cheaper path. What fits
in 30 GB:

| config | max crop | covers a 204 s track |
|---|---|---|
| f16 r128 prodigy | 1650 fr / 66 s | 32% |
| f16 r128 adamw | 2496 fr / 100 s | 49% |
| f16 r64 adamw | 3190 fr / 128 s | 63% |
| **q8_0 r128 adamw** | **4272 fr / 171 s** | **84%** |
| q8_0 r64 adamw | 4771 fr / 191 s | 94% |

f16 and Prodigy together were holding 10.6 GB, which is ~2600 frames of crop.
Both were dropped for the crop.

**This reverses the f16-over-q8_0 ear result deliberately.** That test (a
750-step f16-trained adapter beating a 2000-step q8_0-trained one) was run at
the 128-frame crop, where *both* candidates had been trained on five-second
fragments and neither had learned how a song starts or ends. It compared two
structurally broken adapters. **Re-run it at this crop before spending 8 GB on
f16 again.**

### BF16 tensor cores (`--weights bf16`) — works, but not on a 32 GB card

The trainer runs base matmuls in **F32**: `ggml_out_prod` is F32-only, so
`lm_linear` dequantizes each weight in-graph and the GEMMs land on TF32. Lever A
(`engine/src/train/lm-bf16.h`) feeds the raw BF16 weight to mul_mat and rewrites
the backward's OUT_PROD nodes into MUL_MAT, reaching the tensor cores. Wired into
`mm3-lm-train` on 2026-08-24; needs a BF16 base from
`convert-mm3.py --components lm --quant bf16`.

Measured on a 5090, matched in every other respect:

| config | crop | track | step | ms/frame | peak | free |
|---|---|---|---|---|---|---|
| q8_0 + F32 window | 4272 | 84% | 15.50 s | 3.63 | 27.7 GB | 4.2 |
| q8_0 + F32 window | 2496 | 49% | 10.50 s | 4.21 | 20.1 GB | 11.8 |
| bf16 + Lever A | 2496 | 49% | **7.50 s** | 3.00 | 28.9 GB | 3.0 |
| bf16 + Lever A | 3100 | 61% | 9.75 s | 3.15 | 30.6 GB | 1.2 |

**1.4x faster at matched crop, for 9.0 GB** — 7.7 GB of which is simply the base
being 16-bit rather than q8_0. That ceilings bf16 at ~3100 frames where q8_0
reaches 4272, so the speed costs **23 points of track coverage**, and coverage is
what decides whether a render sounds like a song. Per supervised frame bf16 is
only 1.15x ahead there.

**Default stays q8_0.** Pick bf16 when coverage is not the binding constraint: a
bigger card, a shorter corpus, or a deliberate speed run.

One loose thread: identical step-1 loss (3.5930 vs 3.5932) at a **29% lower
gradient norm** (5.561 vs 7.870) is the quantizer's error appearing as gradient
noise. Whether that matters by ear is untested.

BF16 is also the SOURCE dtype of the MM3 weights — but it is not better for
inference than f16, which keeps all 7 of BF16's mantissa bits and adds 3 more.
Render on q8_0 as always.

### Supervising fewer positions does NOT buy VRAM

Worth writing down because it is an intuitive and wrong idea. Restricting the
*supervised* span while keeping the visible prefix saves **compute, not memory**:

- peak is driven by the **visible** span S, not by `s_tr`;
- the CE head is already chunked — `lm_ckpt_head_chunked` loops
  `for (i = 0; i < s_tr; i += CH)` allocating per chunk from an arena sized by
  `s_max`, so the supervised count drives iteration count;
- allocation happens upfront at `s_max` regardless of the actual S per step.

The version that WOULD buy the memory is a **no-grad frozen-KV prefix** —
condition on 0..E without retaining prefix activations, supervise only the
tail. That turns O(S^2) into O(S) plus a small window, and it is an engine
project (split forward: frozen prefix -> trained tail), not a config change.
It also costs exactness: the adapter stops getting gradient for how it shapes
the prefix. NOT BUILT.

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

## Every run keeps its own log (since 2026-08-24)

`<run dir>/train-log.jsonl` and `<run dir>/train-console.log`, beside the
checkpoints. Before this the trainer's JSONL was parsed into job events and
dropped and only a 30-line stderr tail survived, so any question asked after a
run finished — "step 750 came out as noise, what happened at 750?" — had no
loss curve, no Prodigy `d` and no warning left to read. Check these FIRST.

## The acoustic loss (2026-08-25): WHY adapters wrecked vocal timbre, and the fix

**Root cause of chipmunk/goblin renders, found and fixed.** The depth decoder
generates every acoustic codebook — the timbre — conditioned on the LM's
`last_hidden_state` (mm3-ar-loop.h: `depth_decode(last_hidden, sampled)`).
Semantic-only training leaves that hidden state unconstrained; the frozen depth
decoder then decodes states it never saw, and vocals come out formant-shifted.
Direction is unconstrained drift — ADTR came out chipmunk, Fightstar goblin,
base model always clean. The ear-validated "MLP 0.5" render dial was this fault
being managed empirically. It affects EVERY planner-only MM3 adapter, including
bghira's SimpleTuner recipe (worth reporting to the working group).

Fix (f50c0753): `--depth-loss-weight 1.0` (DEFAULT ON) supervises books 1..7
through the FROZEN depth decoder, teacher-forced (one causal 8-token pass per
sampled frame, 128 frames/step), gradient into the adapter via last_hidden.
Cost: ~nil step time, +0.4 GB measured at crop 2496. `depthLoss` rides the
step JSONL. 0 disables — A/B only.

**The fd tripwire runs at step 1 of every run and ABORTS on mismatch.** It
caught three real bugs in this very feature before any could train, including
ggml's accumulator contract: `ggml_build_backward_expand` with no accumulator
array leaves dL/dL unseeded and every gradient computes as exactly zero,
silently. Never bypass it.

Adapters trained BEFORE this fix carry the timbre fault baked in; retrain
rather than re-dial. Whether MLP-at-render can go back to 1.0 with the loss on
is an open ear question.

## The "sped up and higher pitched" renders: NOT a sample-rate error — measured and closed

Adapter renders of a drop-C# band read as the artist sped up and pitched up,
and a linked 0.9188 (=44.1/48) resample in a DAW "fixes" them. The obvious
conclusion — a 48kHz/44.1kHz clock error — is WRONG, and was excluded three
ways on 2026-08-24:

1. **Encode timing**: all 10 dataset songs' .codes run at 24.97 fps of true
   FLAC time (a rate mix-up would give 22.97 or 27.2).
2. **Pitch grid**: every render sits 0-3 cents ON the A440 semitone grid. A
   real 48/44.1 shift parks everything +47 cents off-grid — verified by
   simulating the error on a real track, which measured +45.
3. **Unison replay**: the engine accepts `forced_semantic` + `forced_acoustic`
   in /mm3/synth (mm3-request.h) — feed a song's stored .codes straight through
   cond→DiT→voc with the planner bypassed. The reconstruction came back at
   tempo x1.000, 0 cents, +0.00 semitones vs the FLAC (spectral corr 0.998).
   The codec loop is transparent end-to-end.

What remains: the PLANNER free-runs faster and higher-registered than the
band. Teacher-forced it is exact; sampled, it drifts to prior pacing (exposure
bias). Note the shared caption feeds that prior: tempo WORDS ("mid-to-fast
tempo", "double-kick bursts") are MM3's only real tempo control (bpm/key are
dead caption knobs), so an accelerant-stuffed caption is self-inflicted.
A linked DAW resample "fixing" it only proves the correction lands in the
right zone, not that a clock error exists — linked-vs-linked A/Bs cannot
separate the axes. Use the replay recipe above before ever re-opening this.

## Structured crops v2: the random share is load-bearing

85/15/0 (start/end/random) memorised: two DETERMINISTIC positions per song =
16 distinct samples on an 8-song album, train loss 0.0003 by step 800, audible
degradation from ~ck550. Defaults are now **40/15/45** with **steps 500,
saveEvery 50, warmup 25** — variety restored, checkpoint grid fine enough to
catch a 150-350 ear-optimum.

## Preview history: everything before 2026-08-24 evening rendered on f16

The preview renderer never pinned a base, so it rendered on the engine's
best-first pick — f16, the one base adapters are garbled on. Fixed 8e42f8ba:
adapter previews now pin q8_0 and record `renderBase` on the preview. A preview
record WITHOUT that field predates the fix and is not evidence about its
checkpoint — including "the early ones sounded fine" (small delta, little for
the broken path to act on) and especially "the late ones are all the same
garble" (the failure belongs to the path, so every large-delta checkpoint
collapses onto the same output; measured pairwise spectral cosine 0.9999 across
steps 311-1000, vs 0.896 for the same checkpoints on q8_0).

## Structured crops and dataset size are COUPLED — 85/15 with no random share memorises

The first structured-crop run (8 songs, crop 2496, start/end 0.85/0.15, 1000
steps, AdamW 8e-5) hit train loss 0.0003 by step 800. That is recitation, and
the arithmetic says why: frame-0 and flush-to-end are SINGLE positions, so
85/15/0 collapses ~20,000 distinct random crops into **16 distinct samples**
(2 per song), while the long crop multiplied exposure — 61 effective passes
over the album against the ear-validated recipe's 7.8.

Nested endpoints (vary E with c0=0) do NOT rescue this: a shorter prefix crop
is contained in the longer one, so it is the same content at different lengths,
not augmentation. When retuning, move at least one of: the random share (a
c0=0 ANCHOR SHARE plus random remainder, not 85/15/0), the step count (~130
steps matches the old exposure at crop 2496 on 8 songs — but warmup 50 then
eats 38% of the run), or the corpus size. Ear evidence for where the damage
starts: none yet; the q8_0 ladder from the memorised run is the first valid
listen.

## A single gibberish preview is not necessarily a training fault

Previews are ONE autoregressive sample at one seed. A 1-ULP logit change flips
a token and diverges the whole render, so an isolated bad checkpoint between
two good ones is as likely to be the sampling lottery as a real instability.
**Re-render that checkpoint at two other seeds before believing it.** The
checkpoint is on disk; it costs about a minute.

## Verify a run started correctly

```
grep -E "songs \(|evaluation:|caption begins|VRAM after" <train.log>
```

Expect: sensible train/holdout split, a non-zero eval crop count, the caption
starting with the trigger exactly once, and VRAM inside the card.
