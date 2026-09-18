---
name: mm3-lm-adapter-training
description: The validated recipe for training MiniMax-Music3 planner-LM style adapters (artist/album clones) with ace-train mm3-lm-train and the Training Studio. Use when training an MM3 LM LoRA, choosing rank/optimizer/steps, picking which checkpoint to ship, diagnosing "the adapter barely works" or "it sounds overcooked", or setting Training Studio defaults.
---

# MM3 LM Adapter Training — the working recipe

Goal this recipe is tuned for: an **album clone**. You want a generation that
could be another track off that record. Memorisation is acceptable and in fact
desirable; style bleed is acceptable. This is NOT tuned for a surgical trigger
LoRA that leaves the base model otherwise untouched.

Established over a 6-album, 30,000-step sweep on 2026-08-23/24 (album B,
the album-C artist, album K, album M, album L, the album-A artist), each album
laddered by ear across 20 checkpoints. Working log: `docs/plans/
2026-08-23-mm3-albumC-style-adapter-findings.md` *(gitignored, local only)*.

## THE CURRENT BEST RECIPE

Keep this block up to date — it is the thing the in-app trainer should
eventually be dialled to, and `docs/plans/` is gitignored so nothing there
survives a fresh clone.

**WHOLE-SONG since 2026-09-11.** The crop-750 / history-2048 recipe below
this block's history taught adapters that lost the song's arc: one sung
passage, minutes of looped instrumental, no ending (vocal share 0.16 on album
P against the album's 0.68). Nothing in a 30 s window ever scores more than
30 s at once. On the 2026-09-10 overnight those adapters ended 12 of 72
renders across six fresh albums while the pristine base ended 36 of 36 on the
same prompts, with no dependence on album length or genre. Training every
track as ONE sequence (`--max-frames 9000`, no prefix, flash) ended 7/12 on
albums B and S and 6/12 on album P at the 360 s ceiling, and lifted album P's
vocal share to 0.42. Cost on an RTX 5090: 6-12 s/step, 29 GB peak at rank 128
(the engine clamps its buffers to the album's longest track since 2026-09-11,
so short albums cost less). Whole-song ending times follow the LYRIC SHEET's
length more than the album's. Not yet re-heard for likeness at the time of
writing; the ledger `docs/plans/mm3-endings-checklist.md` has the tables.

```
--lm mm3-lm-q8_0.gguf
--rank 128 --alpha 128 --adapter-type lora --hot-pizza          # HOT-PiZZA: PiSSA with principal-subspace dropout (2026-09-06)
--pissa-frozen-f16 --pissa-cache-dir <adapters>/mm3-lm-adapters/_pissa-init-cache
--optimizer adamw --lr 8e-5 --lr-end-frac 0.005 --warmup 25    # AdamW tied Prodigy by ear, 2.3 GB lighter (2x LR broke vocals)
--attn flash                                                   # no --prefix-frames: under the whole-song window the track is its own history
--max-frames 9000 --drop-over-frames 9000                      # WHOLE SONG (2026-09-11, Rob): every track as one sequence; tracks over 360 s left out
--crop-mode structured --crop-start-frac 0.2 --crop-end-frac 0.15   # only matters for a track longer than the window (none, with the drop)
--crop-start-tiles 3 --crop-anchor song
--rank-dropout 0.1                                             # the mask IS the method under --hot-pizza; never 0
--steps 600 --save-every 100                                   # Balanced = 600, Fast = 300, Thorough = 900 (2026-09-11); stop on STEPS
--depth-loss-weight 1.0 --depth-loss-frames 128
# captions: per-track <stem>.mm3.txt from MOSS/Gemini, and ONLY those. No --caption-file:
# the shared caption killed endings (0/6 vs 4/6) and was removed on 2026-09-09.
# NO --reg-* prior by default (it was a workaround for the shared caption; costs likeness)
--trigger "<artist>" --trigger-prepend
--holdout 0.15 --eval-every 250 --eval-crop 500
```

Previews: every 50 steps (= every checkpoint), 40 s, control + baseline off,
rendered on q8_0 at MLP 1.0 — the same dials generation uses.

**Rob, 2026-09-06, on the stacked HOT-PiZZA recipe above (blind-hotpissa-recipe
letter F): "sounds fantastic, this should be the default in-app."** Cost on
albumA: 4.7 s/step, 39 min for 500 steps, peak 25.9 GB — against the
2026-09-05 LoKr/Prodigy/exact/4096 line's 5.9 s, 49 min, 30.4 GB. Every
lever was first tied individually in a blind set, then stacked and heard.
The earlier LoKr line stays in the git history of this file.

**2026-09-07, the safe stack.** An overnight one-lever-at-a-time speed trial
(`overnight-speed/COSTS.md` + `blind-speed/RESULTS.md` in the album A hub) found
prefix 1024 (69 of 90) and crop 500 (67.5) tie the reference (68), while the
prefill chunk 256 → 1024 cuts 17% off the step with an identical step-1 loss.
Combined and heard blind (`blind-confirm/RESULTS.md`): safe stack 67 vs the
crop-750 recipe 70.5, inside the ~6 noise floor, at 3.1 s/step and 26 min per
500 steps (was 4.5 s, 37 min). That is the recipe block above. What did NOT
survive: doubling the LR (every 2x arm 3-4 under, and a 2x-LR/300-step stack
produced a vocal-free plan on 1 of 6 songs across two seeds) and turning the
acoustic loss off (lowest score). Held-out loss every 50 has the same shape in
every arm — it follows the seed's crop order, not the adapter — so it is not a
stopping signal.

**Presets (Rob, 2026-09-07, `MM3_LM_PRESETS` in mm3Train.ts, a Recipe row in
the Training Studio card; the route lays `preset` under the request's own
fields, so `{preset:'thorough'}` alone trains Thorough):**

| preset | steps | lr | window | history | min/album (5090) | record |
|---|---|---|---|---|---|---|
| Fast | 300 | 8e-5 | whole song (9000) | none | 30-60 | the 2026-09-10 overnight arm: album B 7/12, album S 7/12, album P 3/12 at 300 s and 6/12 at 360 s |
| **Balanced** (default) | 600 | 8e-5 | whole song (9000) | none | 60-120 | Fast at twice the depth; Rob's pick for the default (2026-09-11), not yet heard |
| Thorough | 900 | 8e-5 | whole song (9000) | none | 90-180 | three times Fast's depth; not yet heard |

(Superseded 2026-09-11: Fast 300 / Balanced 500 / Thorough 1000 at crop 750
with a 2048- or 4096-frame history. That regime's record — Fast's Simlish
vocals of 2026-09-07, Balanced's 4/6 on album B — is in this file's git
history.) All three share flash, AdamW, HOT-PiZZA r128, f16 factors and the
acoustic loss; tracks longer than the window are left out (`longTracks:
'exclude'`, engine `--drop-over-frames`). `MM3_LM_DEFAULTS` carries the
Balanced values, so an empty API body trains Balanced. Prior preservation is
OFF unless the request names a corpus (since 2026-09-09).

**Rob, 2026-08-25, on the LoKr configuration this replaced: "the closest we've ever gotten to
artist replication."** Crop 750 = 30 s = ~3 s/step; he set it by ear after
finding 15 s steps at crop 4272 unworkable and the shorter crop *better*, not
merely faster.

**At render: everything 1.0 for adapters trained WITH the acoustic loss
(2026-08-25 onward) — previews and generation now default there. The old
"MLP 0.63-0.75" dial was damage control for the timbre fault and applies only
to PRE-FIX adapters (their sidecar recommendedScales override the defaults).**

19.7 GB VRAM measured at crop 750 (the acoustic loss adds its frozen depth
decoder, ~1.2 GB). **The crop settings changed on 2026-08-24 and everything
auditioned before that date was trained at `--max-frames 128 --crop-mode
random` — five seconds per step. Treat pre-2026-08-24 ear results about the
BASE and the OPTIMIZER as void** (see below); the rank and MLP-dial findings
stand, because they were measured against each other under the same broken
crop.

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
| albumK | 750 | 750 | 1.00 |
| albumL | 250 | 750 | 1.00 |
| albumA3 | 500 | 2000 | 1.00 |
| albumM | 250 | 1250 | 0.50 |
| albumC | 250 | 1750 | 0.50 |
| albumB | 250 | 2000 | 1.00 |

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

- **album M had a severe rhythm defect at ck500** ("not 4/4"), on both
  seeds, gone by ck1250. That album needed MORE training. A blanket "stop early"
  rule would have shipped the broken one.
- **album L was the weakest clone** — likeness immediately, poor coherence at
  every rung. It is also the only album with ~2032-token prompts (4x the others,
  dense rap lyrics) and the most eclectic track list. Treat dense-lyric or
  stylistically scattered albums as harder, not as training failures.
- Corpus size (10–20 tracks) showed **no reliable effect**. An apparent
  "more tracks = later optimum" trend across five albums was flatly contradicted
  by the sixth. Do not plan around it.

## Traps

- **`--eval-crop` defaults to 400 and is pinned independently of `--max-frames`.**
  With 128-frame windows every eval crop exceeds `S_max`, all are silently
  skipped, and the run reports an eval plan at startup then never evaluates.
  Always pass `--eval-crop <= --max-frames`; that always fits, because
  `S_max = max_prompt + max_frames` and `max_prompt` already includes holdout.
- **`--trigger` alone only writes the sidecar.** It does not train the trigger.
  `--trigger-prepend` is what injects it into the captions — and the sidecar now
  records which of the two happened, because the render path auto-prepends and
  must not do that for a trigger the model never saw.
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

## Verify a run started correctly

```
grep -E "songs \(|evaluation:|caption begins|VRAM after" <train.log>
```

Expect: sensible train/holdout split, a non-zero eval crop count, the caption
starting with the trigger exactly once, and VRAM inside the card.

## Deeper reference (read on demand)

Everything below is out of this file to keep it cheap to load. Open the one you need:

- [`reference/captions-and-lyrics.md`](reference/captions-and-lyrics.md) — **Captions and lyrics**: Captions: per-track .mm3.txt ONLY; Lyrics shape matters as much as the adapter
- [`reference/crops-and-vram.md`](reference/crops-and-vram.md) — **Crops, VRAM and precision**: Structured crops v3: the CONTENT MIX decides whether songs open like songs; Structured crops v2: the random share is load-bearing; Structured crops and dataset size are COUPLED; The crop is the axis that decides whether it sounds like a song; Crop length is QUADRATIC in VRAM, and that is why f16 lost; BF16 tensor cores
- [`reference/method-and-knobs.md`](reference/method-and-knobs.md) — **Adapter method, knobs and axes**: HOT-PiZZA is the default method since 2026-09-06; Endings, honest status; Adapter files: the residual + delta form; New adapter knobs; Supervising fewer positions does NOT buy VRAM; `--prefix-frames N`: real history in front of the crop; Do NOT set `--crop-start-frac` to 0 once a prefix is on; `--crop-anchor song` did nothing until 2026-08-26; The three axes, which are separable; The MLP dial is RANK-DEPENDENT; DO NOT USE MUON at the default scale; Known confound; Dialling in the in-app trainer
- [`reference/rendering-and-losses.md`](reference/rendering-and-losses.md) — **Rendering, losses and closed investigations**: RENDER ADAPTERS ON q8_0 ONLY; The acoustic loss; The "sped up and higher pitched" renders: NOT a sample-rate error; SimpleTuner's nextlat is NOT our acoustic loss; Preview history: everything before 2026-08-24 evening rendered on f16; A single gibberish preview is not necessarily a training fault; Cover-laundered codes for dense-mix artists
- [`reference/stopping-and-resume.md`](reference/stopping-and-resume.md) — **Stopping rules and resuming runs**: Target loss as a stopping rule; Continuing a finished or halted run
