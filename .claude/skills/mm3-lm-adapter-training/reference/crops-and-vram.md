# Crops, VRAM and precision

> Reference for the `mm3-lm-adapter-training` skill. Read only when the task needs it.

## Structured crops v3: the CONTENT MIX decides whether songs open like songs

Found 2026-08-25 on the best-ever album D run (crop 750): renders jumped in
"like a cut" — no intro — despite `--crop-anchor song` and 40% of steps
anchored at frame 0. Mechanism, confirmed by Rob's render-MLP A/B (lowering
MLP restores intros and spends identity):

**Adapter weight deltas are position-independent.** A LoRA/LoKr cannot store
"intro at position 0, chug at position 2000" — it mostly encodes a CONTENT
prior that follows the overall supervised mix. At crop 750 under 40/15/45,
~60% of supervised audio was mid-flow material with no arc, so generations
lean mid-flow from bar one. The position anchor cannot carry this alone: the
base model was pretrained with frames starting right after the prompt, so
anchored mid-song positions are patterns it has no strong machinery for.

Fixes, in the shipped defaults:
- **Shares 55/15/30** (start/end/random — random is the implicit REMAINDER of
  the two UI fields, now displayed in the end-frac hint).
- **Tiled starts** (`--crop-start-tiles 3`): half the start share stays at
  frame 0 (openings must dominate), half lands on aligned tiles at K and 2K at
  their true positions — the intro→build→verse ARC taught in order at
  3-second-step prices. `1` = the old frame-0-only behaviour. Effective mix at
  defaults: ~27.5% true openings, ~27.5% arc tiles, 15% endings, 30% random.

## Structured crops v2: the random share is load-bearing

85/15/0 (start/end/random) memorised: two DETERMINISTIC positions per song =
16 distinct samples on an 8-song album, train loss 0.0003 by step 800, audible
degradation from ~ck550. Defaults are now **40/15/45** with **steps 500,
saveEvery 50, warmup 25** — variety restored, checkpoint grid fine enough to
catch a 150-350 ear-optimum.

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

## The crop is the axis that decides whether it sounds like a song

Found 2026-08-24 after every checkpoint of an album Q run rendered a track that
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

## Crop length is QUADRATIC in VRAM, and that is why f16 lost

Peak is `loaded + perRank*rank + 0.2679*S + 0.00044765*S^2 + const`, with
`S = 1142 + frames`. The backward retains `[S, S, heads]` attention scores —
that was true unconditionally until 2026-09-05. **`ggml's flash-attn has no
backward` is no longer correct**: HOT-Step's own fused
`GGML_OP_FLASH_ATTN_TRAIN`/`_BACK` closed that gap on 2026-09-01, and
`mm3-lm-train` gained `--attn flash|flash-f32` on 2026-09-05, making attention
memory linear in S instead of quadratic. Measured on this corpus (RTX 5090,
`mm3-lm-f16`/`mm3-lm-q8_0`, rank 256, checkpointed): flash moves the usable
crop ceiling from ~4300 frames (exact, table below) to at least 11,178
frames — this dataset's longest track, no OOM reached — before the same ~29 GB
practical spill ceiling. **Default is still `exact`, the table below is
unchanged, and none of this is in the recipe** — the shipped recipe trains at
crop 750, where flash measures no benefit (checkpointed segments already hide
the small softmax in allocator slack), and nothing trained under flash has
been heard. Full numbers: `docs/TRAINING.md` MM3 section,
`.claude/skills/flash-attn-training/SKILL.md` §3/§7. What fits in 30 GB
**at `--attn exact`**, the only mode any shipped adapter has trained under:

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

## BF16 tensor cores (`--weights bf16`) — works, but not on a 32 GB card

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
bigger card, a shorter corpus, or a deliberate speed run. Since 2026-09-03 the
base is downloadable: `mm3-lm-bf16` is on scragnog/MiniMax-Music3-GGUF and in
the registry, and picking it in the train form's base picker is all it takes
(the server maps a bf16 base to `--weights bf16`). Untested on Ampere (A40),
where the F32 fallback is slowest and the gain should be largest.

One loose thread: identical step-1 loss (3.5930 vs 3.5932) at a **29% lower
gradient norm** (5.561 vs 7.870) is the quantizer's error appearing as gradient
noise. Whether that matters by ear is untested.

BF16 is also the SOURCE dtype of the MM3 weights — but it is not better for
inference than f16, which keeps all 7 of BF16's mantissa bits and adds 3 more.
Render on q8_0 as always.
