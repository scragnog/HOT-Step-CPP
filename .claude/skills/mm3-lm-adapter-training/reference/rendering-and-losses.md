# Rendering, losses and closed investigations

> Reference for the `mm3-lm-adapter-training` skill. Read only when the task needs it.

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

Training on f16 and rendering on q8_0 produced the best album B adapter of the
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

## The acoustic loss (2026-08-25): WHY adapters wrecked vocal timbre, and the fix

> **NOT SUFFICIENT — re-opened 2026-08-25 evening.** Both album D adapters
> trained post-fix with `acoustic loss: ON — weight 1, 128 frames/step`
> (train-console.log verified) still render Charlie Simpson chipmunked at 1.0×,
> correct at ~0.9188×. The mechanism below is still the best-supported theory
> (formant shift with pitch on-grid cannot be a resample error), but the
> teacher-forced anchor did not stop the drift — likely because it constrains
> last_hidden on the real-codes manifold while inference free-runs (the same
> exposure-bias shape as the tempo drift). Do NOT tell the user this is fixed.

**Root cause of chipmunk/goblin renders, found and fixed.** The depth decoder
generates every acoustic codebook — the timbre — conditioned on the LM's
`last_hidden_state` (mm3-ar-loop.h: `depth_decode(last_hidden, sampled)`).
Semantic-only training leaves that hidden state unconstrained; the frozen depth
decoder then decodes states it never saw, and vocals come out formant-shifted.
Direction is unconstrained drift — album Q came out chipmunk, album D goblin,
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

What remained was then SPLIT by the 2026-08-25 findings: the formant half
("chipmunk"/"goblin" voices) was the depth-decoder hidden-state drift — fixed
by the acoustic loss (see its section) — and the residual pitch/register reads
made on pre-acoustic-loss adapters are void with the rest of their timbre.
The original framing for reference: the planner free-runs faster and
higher-registered than the band. Teacher-forced it is exact; sampled, it drifts to prior pacing (exposure
bias). Note the shared caption feeds that prior: tempo WORDS ("mid-to-fast
tempo", "double-kick bursts") are MM3's only real tempo control (bpm/key are
dead caption knobs), so an accelerant-stuffed caption is self-inflicted.
A linked DAW resample "fixing" it only proves the correction lands in the
right zone, not that a clock error exists — linked-vs-linked A/Bs cannot
separate the axes. Use the replay recipe above before ever re-opening this.

## SimpleTuner's nextlat is NOT our acoustic loss (read the source before agreeing)

The Discord thread concluded SimpleTuner's `xm`+`nextlat` was "the constraint
we're missing." Reading `origin/main`'s `helpers/training/nextlat.py`: nextlat
is **hidden-state SELF-prediction** — a small trained head predicts
`hidden[t+1]` from `hidden[t]`, target detached but still the model's own
(drifting) state. **No ground truth anywhere in the loss.** It is a trajectory
-smoothness prior (bghira's "belief system"), softens drift indirectly, and by
construction cannot anchor register or formants — if the manifold drifts, the
predictor drifts with it. bghira's own "nextlat needs xm to keep semantic
coherence" is what a blunt smoothness prior does. Our acoustic loss anchors
the depth interface to ground-truth codebooks — the targeted constraint. A
nextlat-style smoothness term MAY compose with it; that is an ablation for
later, not a rescue.

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

## A single gibberish preview is not necessarily a training fault

Previews are ONE autoregressive sample at one seed. A 1-ULP logit change flips
a token and diverges the whole render, so an isolated bad checkpoint between
two good ones is as likely to be the sampling lottery as a real instability.
**Re-render that checkpoint at two other seeds before believing it.** The
checkpoint is on disk; it costs about a minute.

## Cover-laundered codes for dense-mix artists (2026-08-31)

The champion code encoder is balance-sensitive: on real dense mixes the vocal
sinks out of the code targets, and the adapter clones the deficiency. The fix
is `ace-train mm3-launder` — real audio -> rec7 states -> the flow DiT's own
latents -> champion codes — which puts the vocal where the codes can carry it.
**Ear-validated (Deftones White Pony A/B, identical recipes, only the training
audio differed): the laundered arm won "on every metric I can hear".**

In the app: the codes card's "Cover-launder" checkbox exports into a SIBLING
cache (`mm3-codes-laundered/`), and the train form offers "Train on
cover-laundered codes" only when that cache exists. Off = the standard
pipeline, byte-identical. ~1.4x realtime per track, once per dataset, cached
forever. Needs `mm3-rec7-*.gguf` installed (converted with
`convert-rvq-encoder.py --head --m3` — the file carries the LM's semantic
table slices, so laundering never runs the 8B's forward).

When to use it: albums where the vocal or lead lines bury in the mix
(nu-metal, shoegaze, dense punk). Unmeasured on sparse/acoustic corpora — do
not assume it helps there; A/B before adopting it as a house default.
