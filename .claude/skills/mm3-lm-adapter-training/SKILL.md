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

## The recipe

```
--rank 64 --alpha 64
--optimizer adamw --lr 8e-5 --lr-end-frac 0.005 --warmup 50
--max-frames 128 --crop-mode random --crop-anchor song
--rank-dropout 0.1
--steps 2500 --save-every 250
--caption-file <one shared caption for the whole album>
--trigger "<artist>" --trigger-prepend
--holdout 0.15 --eval-every 250 --eval-crop 128
```

This is bghira's published SimpleTuner configuration, adapted. It measured
~800 ms/step and ~14.7 GB VRAM at rank 64 on a 32 GB card, so a 2500-step run
is roughly **35 minutes per album**.

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

At generation time the caption should look like the training caption. See
[mm3-captioning](../mm3-captioning/SKILL.md) for the MM3 Structured Caption
format used elsewhere — note this recipe deliberately uses the short
comma-separated style, not the three-section format, for the *training* caption.

## Lyrics shape matters as much as the adapter

MM3 plans the whole track from the whole lyric, and `duration` only truncates.
Write a FULL song's worth of lyrics even for a 40 s render. Tidy four-line
stanzas produce mainstream rock and suppress the style; irregular line lengths
matching the artist's own writing work far better.

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
