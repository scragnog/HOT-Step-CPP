# Adapter method, knobs and axes

> Reference for the `mm3-lm-adapter-training` skill. Read only when the task needs it.

## HOT-PiZZA is the default method since 2026-09-06 (Rob)

`--hot-pizza` (implies `--pissa`): PiSSA with the rank-dropout mask on the
principal component itself, so every micro-step a random `--rank-dropout`
share of the base's top-128 subspace is deleted while the adapter fits the
album. Found as a masking bug on 2026-09-05 (lm-graph.h masked one branch of
the PiSSA fold), kept on purpose: blind on albumA it beat every other
method twice (68 and 66.5 of 90; LoRA 60–64, DoRA 63, LoKr 41.5 with a drone
failure; the CORRECTED PiSSA was worst at 39.5 with drone plans in half its
renders). Tables: `_experiments/_LISTENING/2026-09-06-mm3-method-ab/albumA/`.
Consequences for the recipe block above: `--adapter-type lora --rank 128
--alpha 128 --hot-pizza` replaces the LoKr line; **stop on steps (500 heard;
250/350 ladder pending), never on loss** — the perturbed forward keeps the
trailing train loss at 2.7–5, so a loss target never binds. Plain LoRA is
next in line; LoKr stays selectable. One album so far: the second-artist run
is the outstanding validation. Cost is LoRA's: ~5.9 s/step, peak 30.4 GB on
the 32 GB card at crop 750 + prefix 4096.

## Endings, honest status (2026-09-10)

Tool: `tools/vocal-end/vocal_end.py` (SuperSep vocal stem + energy trace + whisper hint) tells you where singing
starts and stops in any render or dataset track; use it before believing an LRC timestamp or guessing at a tail.


Per-track captions are necessary (the shared caption gave 0/6) but not sufficient: the same recipe's per-plan
natural-ending rate is 0.2-0.5, varies with the caption x lyrics pair and with the training draw (the trainer is
deterministic per seed; two draws gave 2/6 and 3/6 with equal likeness), and Rob's in-app test on three fresh albums
was 2 of 9. No training-side lever has raised it without losing likeness (score-last ends everything and sounds like
nothing). The natural-ending candidates feature (3 takes, up to 4 rounds) is what makes a render end. The full ledger
and the ordered list of what is left to try: `docs/plans/mm3-endings-checklist.md`. Pitch/tempo drift in adapter
renders (every arm with the acoustic loss; DL0 clean) has its own file: `docs/plans/mm3-pitch-tempo-drift.md`.

## Adapter files: the residual + delta form (2026-09-09)

A HOT-PiZZA / PiSSA export used to be a rank-2r F32 PEFT LoRA: 2.79 GB per
checkpoint at r128, half of it the frozen A0/B0 pair (the base weight's own
top-128 singular directions), which is the SAME bytes in every adapter
trained on the same base. Since 2026-09-09 that half ships once:

- `models/mm3/mm3-lm-q8_0.pissa-r128.safetensors` (0.7 GB, F16; registry id
  `mm3-lm-q8_0-pissa-r128`, in the Q8_0 / Balanced / Training packs; format in
  `engine/src/pissa-residual.h`). The trainer reads it at init when rank, SVD
  parameters and the base's byte size match, else falls back to the SVD cache
  or the SVD and WRITES it beside the base. Keyed by base file size: a
  re-quantized base misses rather than pairs wrongly.
- Adapters trained against it export the DELTA form: `lora_A = A - A0`,
  `lora_B = s(B - B0)` at rank r, F16, `hot_step.param_method = 4`,
  `hot_step.pissa.meta`, `hot_step_pissa_residual` in adapter_config.json.
  0.7 GB per checkpoint instead of 2.8. `minimax/mm3-lm-adapter.h` rebuilds
  the rank-2r pair from the two files (needs the resident base's path, so the
  residual is found beside it); everything downstream is unchanged, VRAM too.
  NOT a plain LoRA: PEFT/SimpleTuner would apply (B-B0)(A-A0), which is
  nothing; the AS1.5 LM loader refuses the marker by name.
- Standalone rank-2r exports (no residual beside the base, or the AS1.5 LM
  trainer) are still written, now F16 (1.4 GB). Every older adapter loads as
  before.
- Missing residual at load: the engine error names the file and the Models
  page; download it there. Its SVD is deterministic but not bit-reproducible
  across GPUs, so the file is the truth, never recomputed on a user's machine
  for the shipped base.

Also new that day: `saveEvery` default 100 (was 50) and the ~4.2 GB
`resume-state.bin` is deleted when a run reaches its end unless "Keep resume
state after completion" is ticked (`keepResumeState`); a run that stops short
keeps it. `verifyExport: true` on the request runs `--verify-export` (every
checkpoint round-trips the runtime loader; used for the first delta run).

## New adapter knobs (2026-09-04/05) — none of this is in the recipe above

`mm3-lm-train` gained the same six parameterizations `train-dit` has:
`--dora`, `--rslora`, `--hira`, `--loha`, `--pissa` (+ `--pissa-oversample`,
`--pissa-iters`) and `--hra`, mutually exclusive under the same rules as the
DiT trainer (HiRA excludes DoRA; LoHa excludes both; PiSSA excludes
DoRA/HiRA/LoHa; HRA excludes everything including rsLoRA and needs an even
`--rank`) — plus two soft-prompt flags that now reach MM3 generation as well
as training: `--artist-token`/`--artist-token-k`/`--artist-token-lr` and
`--prefix-n` (a **trained** KV prefix, distinct from `--prefix-frames` above,
which is frozen history with no gradient). All seven arms pass their
finite-difference gate at the default epsilon since 2026-09-05, when `--fd-eps`
became a floor on the step rather than the step (HiRA's gradients are 16-68x
smaller than a plain LoRA's, so a fixed step measured it at 50x worse
signal-to-noise — the estimator, not the backward). `--hra` is the one arm whose
graph and activations scale with `--rank`: its default-rank (64) crash is fixed
(graph budgets now come from the bank), but rank 64 still does not FIT at crop
750 on 32 GB and is refused with the arithmetic rather than crashing. Rank 8
peaks at 17.6 GiB and ~6.8 s/step, rank 32 at ~30.0 GiB and ~27 s/step, against
a plain rank-64 LoRA's 18.2 GiB and ~2.4 s/step. `--prefix-n` and prior
preservation (`--reg-*`) are mutually
exclusive — a trained prefix is non-zero from init, and prior capture needs
an inert model — and so are `--hra`/`--rslora`; the training routes 400 on
both pairs rather than letting the job fail after the model loads.

**None of this is in THE CURRENT BEST RECIPE block and none of it is
ear-validated.** The DiT trainer's own blind listening test found that no
parameterization beat plain LoRA — treat that as the prior for MM3 too until
MM3 has run its own test. Full gate numbers and the open HiRA/HRA bugs:
`docs/TRAINING.md` MM3 section.

## Supervising fewer positions does NOT buy VRAM

Worth writing down because it is an intuitive and wrong idea. Restricting the
*supervised* span while keeping the visible prefix saves **compute, not memory**:

- peak is driven by the **visible** span S, not by `s_tr`;
- the CE head is already chunked — `lm_ckpt_head_chunked` loops
  `for (i = 0; i < s_tr; i += CH)` allocating per chunk from an arena sized by
  `s_max`, so the supervised count drives iteration count;
- allocation happens upfront at `s_max` regardless of the actual S per step.

The version that buys the memory is a **no-grad frozen-KV prefix** — condition
on 0..E without retaining prefix activations, supervise only the tail. **BUILT
2026-08-26, `--prefix-frames N`** — see below.

## `--prefix-frames N`: real history in front of the crop

Built 2026-08-26 (`engine/src/train/lm-kvprefix.h`; working notes in
`docs/plans/2026-08-26-lm-frozen-kv-prefix.md`, gitignored). **Off by default,
never ear-tested, no UI** — a run has to be launched by hand.

The problem it solves is not coverage, it is CONTEXT. A crop at frame 3000
carries its true RoPE position with ~750 keys of evidence in front of it, so
the middle third of the stack — the layers doing long-range aggregation — is
trained to produce position-3000 behaviour from a position-750 view. That is
the band Rob switches off at render time with `scaleMid 0`.

A prefix needs no backward, so it escapes the quadratic term entirely: K and V
cost **0.28125 MB per column** across MM3's 36 layers. Measured at crop 750,
rank 64, AdamW: **+856 MB and about +60% step time for 750 frames (30 s) of
history**.

**How much history.** Measured over the corpus (202 tracks, 14 datasets, median
203 s), as the share of supervised steps whose crop sees as much history as it
will at render:

| prefix | | full-context steps | mean prefix used | prefill cost |
|---|---|---|---|---|
| 750 | 30 s | 48.9% | 505 fr | 1.0x |
| 2250 | 90 s | 73.5% | 1088 fr | 2.8x |
| **4096** | **164 s** | **87.7%** | 1453 fr | 4.8x |
| 5000 | 200 s | 94.4% | 1532 fr | 5.4x |
| 6000 | 240 s | 98.0% | 1566 fr | 5.7x |

**4096 is the default**: past it the curve flattens and you are spending
quadratically to chase the tail. The flag is a CEILING, not a fixed cost -- a
crop near the song's start has little history to load, which is why the mean at
4096 is only ~1450 frames. VRAM is linear and never the binding constraint (1.7
GB at 4096).

Two things it changes that are worth knowing:

* **The window takes one extra input frame.** With history present, the row that
  predicts frame c0 must be frame c0-1's, not the caption's last token. Without
  that shift, every crop still teaches "a song may begin at c0" — the exact
  lesson the crop work exists to remove — and the equivalence self-test catches
  it as a 0.083-nat gap.
* **`--prefix-selftest` is the gate.** Attention over `[prefix ; window]` is
  mathematically identical to one long crop covering both, so the supervised CE
  must not care which way it was produced. It found two real bugs before it
  passed. Run it before trusting a prefix run.

## Do NOT set `--crop-start-frac` to 0 once a prefix is on

Tempting, because the start bucket's ORIGINAL justification was history:
anchoring at frame 0 was the only way a supervised position got the song's real
past in front of it, and the prefix now does that everywhere. But the bucket has
a second job the prefix does not touch.

**Frame 0 is the only place the caption-to-first-frame transition is trained.**
With a prefix, `lead` makes the previous FRAME the predictor of every supervised
position -- correct for mid-song, and exactly what generation does after t=0. At
`c0 == 0` there is no previous frame, `lead` is 0, and the caption's last token
is the predictor. That is the one case generation faces at t=0, and a uniform
draw lands on it with probability 1/span, about 0.02%. Setting the share to 0
reintroduces the 2026-08-24 bug (renders that begin mid-flow) by a different
route.

What the prefix DOES retire is the tiles. `--crop-start-tiles 3` put half the
start share on aligned tiles to teach the intro-build-verse arc, because a crop
at 1500 otherwise had no past. It has one now, so a tile crop is just an
ordinary crop. **`--crop-start-frac 0.20 --crop-start-tiles 1`** is the
reallocation: the opening keeps a share comparable to the ending's 0.15 (both
are one event per song), and the freed 27% goes to random crops. Costs ~1.35x
the prefill, because more steps land late where the prefix is longest.
UNTESTED -- change it on its own run, not alongside a prefix change.

## `--crop-anchor song` did nothing until 2026-08-26

Found while building the above, and it applies to **every MM3 LM adapter trained
before that date**. The trainer computed song-anchored RoPE positions and
uploaded them; `lm_ckpt_micro_step` then overwrote `t_pos` with `0..S-1` before
building any graph. Only the non-checkpointed path honoured them, and nothing
uses that path because it does not fit in VRAM.

So the 2026-08-24 crop work fixed where crops were TAKEN but not where they were
PRESENTED: the model still saw every crop as the song's opening. Fixed via
`LmCkptRun::pos_external`. `--crop-anchor zero` reproduces the old behaviour
exactly, and pre-2026-08-26 runs are not comparable with later ones.

**This LoRA configuration is FROZEN as of 2026-08-24** — settled by ear over a
rank sweep at 64/128/256 with a full MLP dial at each. Further LoRA tuning is
not planned; the open work is LoKr, Prodigy and the Muon step-size sweep.

Rank 128 was chosen over 256 because it matches by ear at **5.4 GB less VRAM,
7% less step time and half the checkpoint size** (1.4 GB vs 2.7 GB) — which is
what decides whether a six-album overnight sweep fits on disk.

Started from bghira's published SimpleTuner config at rank 64 and moved by ear
over a 6-album sweep plus a rank x optimizer 2x2 (2026-08-23/24).

## The three axes, which are separable

This is the load-bearing insight. Earlier work conflated them and went in
circles.

| axis | what it controls | setting |
|---|---|---|
| **rank** | separating the style from the LM's *language ability* | **128** |
| **MLP dial** (render) | vocal identity vs audio fidelity | **0.63–0.75** *(rank-dependent)* |
| **steps** | how much likeness | 750–2000 |

## The MLP dial is RANK-DEPENDENT — never carry it between adapters

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

## DO NOT USE MUON at the default scale

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

## Known confound

Every rank above ran at lr 8e-5, which bghira tuned at **rank 64**. Higher rank
generally wants a different LR, so rank and learning rate are confounded in
these results. Either sweep the LR per rank, or adopt an optimizer that sets its
own (Prodigy — not implemented; `lm-optim.h` branches on a single `want_muon`
bool and only knows adamw/muon).

## Dialling in the in-app trainer

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
