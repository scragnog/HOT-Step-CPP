#pragma once
// train/mm3-lm-train-run.h — MiniMax-Music3 LM LoRA trainer.
//
// HOT-Step file. Included by tools/ace-train.cpp only.
//
// ── WHAT IS NEW HERE, AND WHAT IS NOT ───────────────────────────────────────
//
// Almost nothing about the OPTIMISATION is new. `train/lm-graph.h` already
// carries a trainable cache-free unfused Qwen3 forward, `lm-optim.h` already
// carries AdamW *and Muon* with per-parameter rule selection, `lm-export.h`
// already writes PEFT safetensors, and MM3's LM is Qwen3-8B. So this file is
// the MM3-shaped parts only:
//
//   1. the DATA (codes + captions + lyrics -> a teacher-forced sequence),
//   2. the INPUT EMBEDDING (two tables, two files, summed and scaled),
//   3. the OUTPUT SLICE (semantic codes + EOS, not the 200k vocab),
//   4. the training loop that hangs those on the existing machinery.
//
// Everything in 1-3 was built and falsified first, without an optimiser
// attached, by `ace-train mm3-lm-loss` — see its header. Attaching a backward
// pass to an unverified sequence is how the DiT-trainer delta fiasco happened.
//
// ── MUON COMES FREE, AND THAT IS THE POINT ──────────────────────────────────
//
// `lm_optim_init` classifies EVERY parameter itself: a genuine 2-D matrix whose
// short side clears `muon.min_dim` goes on Muon, everything else falls to
// AdamW. A rank-256 LoRA on q/k/v/o + gate/up/down is 252 matrices of
// [in, 256] / [256, out] — all comfortably Muon-eligible. So `--optimizer muon`
// works here for exactly the same reason it works for ACE, with the same knobs
// (`--muon-lr-scale/-momentum/-ns-steps/-nesterov/-min-dim/-bucket`), and
// `opt.n_muon` is logged because a run where Muon silently classified zero
// parameters and trained as AdamW is the failure mode to watch for.
//
// ── THE OUTPUT SLICE: SEMANTIC + EOS ────────────────────────────────────────
//
// The AR loop masks its logits to "semantic codes + EOS" — 16,385 live
// candidates out of a 200,000-row head. Training over the full vocabulary would
// spend 1.12 GiB on logits (and the same again on their gradient) to supervise
// 8% of it.
//
// `eos_audio` (151670) sits just BELOW `semantic_vocab_offset` (151675), so
// [eos_audio, semantic_offset + semantic_size) is ONE CONTIGUOUS ROW RANGE of
// 16,389 rows — a single ggml_view_2d, no gather, no copy. The four rows
// between them are the caption/lyric delimiters; they ride along in the softmax
// denominator, which is a deliberate and stated approximation:
//
//   * at inference they are masked out, so the training distribution is very
//     slightly wider than the sampling one;
//   * the base model already puts negligible mass on a caption delimiter at an
//     audio position, and training only pushes it lower;
//   * the exact alternative is a concatenated [H, 16385] head built once
//     outside the graph (134 MB at f16) — the right fix if this is ever
//     measured to matter. It has not been.
//
// ── EOS SUPERVISION IS A CROP PROPERTY ──────────────────────────────────────
//
// The lm2 run trained on intros only, because `max_frames` truncated from the
// START; the random-crop patch fixed it and, just as importantly, restored EOS
// supervision by marking `has_audio_end` ONLY when the crop actually reaches
// the track's end. Both live here from day one:
//
//   crop reaches the end -> inputs are all K frames, targets are
//                           f[1..K-1] then EOS          (K+1 supervised)
//   crop does not        -> inputs are f[0..K-2],
//                           targets are f[0..K-1]       (K supervised)
//
// ── OUTPUT LAYOUT ───────────────────────────────────────────────────────────
//
// Each checkpoint is a PEFT directory `<out>/ckpt-<step>/` (adapter_config.json
// + adapter_model.safetensors), plus a `.json` sidecar beside the safetensors.
// Two consumers, one file:
//   * the python side (lm_sft_infer, SimpleTuner) reads a PEFT dir;
//   * pointing `--out` at `<adapters>/mm3-lm-adapters/<run>` makes the shipped
//     server lister find it with NO changes — it scans two directory levels and
//     reads `<file>.json` as the sidecar — so a finished checkpoint appears in
//     the MM3 adapter picker with its trigger word and recommended scales.
// The exported PEFT key names (`base_model.model.model.layers.N.<mod>.lora_A`)
// are already exactly what `minimax/mm3-lm-adapter.h` parses. Verified, not
// assumed: that parser strips `language_model.` and `base_model.model.` and
// then matches the same seven module strings `lm_slot_peft_name` emits.

#include "train/lm-graph.h"
#include "train/lm-optim.h"
#include "train/lm-data.h"
#include "train/lm-export.h"
#include "train/lm-ckpt.h"
#include "train/mm3-lm-resume.h"
#include "train/mm3-lm-prior.h"
#include "train/mm3-f32-isolate.h"
#include "train/mm3-lm-load.h"
#include "train/mm3-depth-train.h"
#include "minimax/mm3-request.h"
#include "minimax/mm3-tokenizer.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// jl() / json_escape() are defined in tools/ace-train.cpp, which includes this
// header. Declared here so the emitter below can be written next to the code
// it describes rather than in the CLI file.
static void        jl(const char * fmt, ...);
static std::string json_escape(const std::string & s);

// ── THE DEFAULTS ARE bghira's SimpleTuner RECIPE (2026-08-23) ───────────────
//
// They used to be ours: r256 / alpha 256 / lr 8e-5 constant-ish / Muon at
// lr_scale 64 / random 1500-frame crops / 800 steps. That recipe produced
// adapters that carried a trace of the artist's timbre while making the backing
// track audibly simpler and cheaper — identity weak AND base damaged.
//
// bghira trains the same component with SimpleTuner and gets better results, and
// he publishes the configs. Two references:
//   * terminusresearch/minimax-music3-lm-lora-fiona-crapple  (9 tracks)
//   * RareConcepts/soad-mm3-vanilla-20260822 + the training-tournament card
//     (45 tracks, 1000 steps, simpletuner_config.json in every checkpoint)
//
// The SOAD config, verbatim, is what the numbers below now mirror:
//
//     lora_rank 64            lora_alpha = rank        lora_dropout 0.1
//     learning_rate 5e-5      lr_scheduler cosine      lr_warmup_steps 50
//     lr_end 4e-7             optimizer adamw_bf16     max_grad_norm 1.0
//     train_batch_size 1      grad_accum 1             max_train_steps 1000
//     minimax_music_lm_max_frames 0   (= whole tracks, truncated FROM THE START)
//
// The comparison that motivated the change: over 45 tracks he runs 42 epochs;
// we ran 61 over 13 songs, at four times the rank, with a normalised-update
// optimizer at an untuned scale, no warmup and no regularisation of any kind.
// We were training harder AND with more capacity on less data.
//
// TWO OF HIS SETTINGS ARE DELIBERATELY NOT COPIED:
//   * `lora_dropout 0.1` is not implemented here yet. It is not a default we
//     can flip: per-layer activation checkpointing recomputes the forward, so
//     any dropout mask must be REPRODUCIBLE at recompute time or the gradients
//     are computed against a different network than the loss was. That needs a
//     position-derived deterministic mask, not a stored random tensor.
//   * `caption_dropout_probability 0.1` is INERT in his LM runs — his
//     _lm_prompt_text raises on an empty caption, so the knob can never fire in
//     language_model mode. Copying it would be diverging, not replicating.
struct MM3LmTrainArgs {
    std::string lm_path, depth_path, manifest, captions_dir, codes_dir, out_dir;
    int         rank = 64, alpha = 64;

    // ── Adapter parameterization: "lora" (default) or "lokr" ───────────────
    //
    // The LoKr parameterization already lives in lm-graph.h (lm_lokr_init) and
    // lm-export.h (lm_export_lokr), shared with the ACE LM trainer. This
    // backend only chooses it. dW = kron(w1, w2) instead of a low-rank B@A.
    //
    // FACTOR, NOT DIM, IS THE SIZE KNOB, and train-lm's default of 6 is WRONG
    // HERE. Measured on MM3's dims (4096 / 1024 / 12288, 36 layers):
    //
    //     factor  6 -> 320.9M params (1283 MB f32)  <- BIGGER than LoRA r64
    //     factor  8 -> 108.5M        ( 434 MB)
    //     factor 16 ->  27.2M        ( 109 MB)      <- 6.4x smaller than r64
    //     (LoRA r64 = 174.6M / 698 MB; r256 = 698.4M / 2793 MB)
    //
    // ACE's 4B LM has smaller dims, so factor 6 splits into small blocks there
    // and into 1024x1024 blocks here. Copying that default would have shipped
    // adapters LARGER than the LoRA they replace.
    //
    // At factor 16 every slot's w2 goes monolithic, and lm_lokr_init then
    // forces a_eff = dim, making lokr_scale exactly 1.0 — so --lokr-dim and
    // --lokr-alpha are INERT at this factor. Kept for CLI parity with train-lm
    // and for lower factors, not because they do anything at 16.
    /** Prodigy's initial step-size estimate. It only ever grows, so a d0 that
     *  is too small costs a few warm-up steps; one that is too large cannot be
     *  taken back. 1e-6 is the reference default. */
    double      prodigy_d0          = 1e-6;
    std::string adapter_type        = "lora";
    int         lokr_dim            = 512;
    float       lokr_alpha          = 512.0f;
    int         lokr_factor         = 16;
    bool        lokr_decompose_both = true;

    bool        is_lokr() const { return adapter_type == "lokr"; }
    double      lr = 5e-5, weight_decay = 0.01, grad_clip = 1.0;
    int         steps = 1000, save_every = 100, warmup = 50;
    /** Cosine floor as a fraction of lr, i.e. SimpleTuner's `lr_end`: his
     *  4e-7 over a 5e-5 base is 0.008. Our shared schedule bottomed at 0.1,
     *  which ran the tail of the run twelve times hotter than his. */
    double      lr_end_frac = 0.008;
    /** 4096 frames = 164 s at 25 fps, and it pairs with crop_mode `beginning`
     *  to reproduce his `lm_max_frames: 0` — whole track where it fits, and
     *  otherwise truncated FROM THE START, which is the point: his option help
     *  says "taken from the start so lyrics stay aligned".
     *
     *  Our old random 1500-frame crop broke that alignment on every crop with
     *  a non-zero offset — the prompt carried the whole song's lyrics while the
     *  supervised audio came from a minute in. --crop-anchor at least tells the
     *  model WHERE it is now; start-truncation removes the mismatch instead. */
    int64_t     max_frames = 4096;
    std::string crop_mode = "beginning";  // random | beginning | structured
    /** `structured` only. The share of training steps whose crop is pinned to
     *  frame 0, and the share pinned flush to the track's end.
     *
     *  The end share is the one that buys EOS: `at_end` below is true only when
     *  c0 + K reaches n_frames, so a random crop supervises an ending at rate
     *  K/(n_frames-K+1) - 2.6% on a 204 s track at K=128. The start share buys
     *  the opening, which a random crop hits at rate 1/(n_frames-K+1), i.e.
     *  0.02%. Those two numbers are why a trained adapter renders a song that
     *  begins mid-flow and never resolves.
     *
     *  Both pinned crops are DETERMINISTIC per song (c0 = 0 and c0 = n-K are
     *  single positions, not ranges), so a large share of them is repeated
     *  exposure to the same few seconds. 0.2 each is deliberately short of the
     *  point where intros and outros start to memorise. */
    double      crop_start_frac = 0.2;
    double      crop_end_frac   = 0.2;
    /** Tiled starts (2026-08-25). The start share does not have to mean "frame
     *  0 only": with short crops that starves the model of the intro->build->
     *  verse ARC, and the adapter's content prior follows the mix — renders
     *  jump in mid-flow. With N tiles, HALF the start share stays at frame 0
     *  (openings must dominate) and the rest lands on aligned tiles K, 2K, ..
     *  (N-1)K at their true positions, teaching the arc in order. 1 = the old
     *  behaviour exactly. 3 at crop 750 covers what crop 2496's start share
     *  used to. */
    int         crop_start_tiles = 3;
    int         grad_accum = 1, seed = 42;
    // ADAMW BY DEFAULT as of 2026-08-23, because the recipe it belongs to is
    // now rank 64.
    //
    // Muon was never chosen on merit — it was chosen because it FIT. AdamW's
    // second momentum buffer costs +2.66 GB at rank 256 on a run that already
    // peaked at 31.7 GB of a 32 GB card. At rank 64 that buffer is 0.67 GB and
    // the constraint evaporates, which removes the only argument for Muon here.
    //
    // Muon also made the learning rate meaningless: its update is normalised by
    // Newton-Schulz, so `lr` had to be paired with an --muon-lr-scale that was
    // never tuned (a 50-step sweep gave 1 -> 3.3114, 4 -> 3.1842, 16 -> 2.8960,
    // 64 -> 2.5407 — monotonic, so 64 was the best of four values TESTED and the
    // top of the range was never found). A 5e-5 AdamW rate is a number with a
    // published reference behind it. Muon remains available via --optimizer.
    std::string optimizer = "adamw";      // adamw | muon
    float       muon_lr_scale = 64.0f, muon_momentum = 0.95f;
    int         muon_ns_steps = 5, muon_min_dim = 16, muon_bucket = 16;
    bool        muon_nesterov = true;
    /** The trigger word. Recorded in the adapter sidecar either way; whether it
     *  is actually TRAINED depends on --trigger-prepend below. */
    std::string trigger;
    /** Prepend `<trigger>, ` to every training caption at prompt assembly.
     *
     *  WITHOUT THIS THE TRIGGER IS NOT TRAINED AT ALL, and that is not a
     *  hypothetical: the first SOAD run recorded `soad_toxicity` in the sidecar,
     *  none of the 14 MOSS-written captions contained it, and rendering with
     *  `soad_toxicity, <caption>` therefore bolted an unseen token sequence onto
     *  an otherwise in-distribution prompt. It measurably HURT — the same
     *  checkpoint sounded better with the trigger removed, and supported full
     *  adapter strength instead of half.
     *
     *  The injected shape is `trigger, ` at the very front of the caption's
     *  first line, on the same line as `Global Metadata`, which is exactly what
     *  lm_apply_tag / applyTriggerTag emit at inference. A trigger on its own
     *  line is a different token sequence and dilutes to nothing.
     *
     *  Captions on disk are never modified — this happens in memory, so the
     *  dataset stays clean and the choice is recorded in the run rather than
     *  baked into files. */
    bool        trigger_prepend = false;
    /** Probability that a training step uses the TRIGGER WORD ALONE as the
     *  caption instead of the full descriptor caption. 0 = off.
     *
     *  The point is to make the trigger carry the style BY ITSELF, which is the
     *  only way a bare-trigger prompt works at inference — a model that has only
     *  ever seen the trigger alongside a full caption has no reason to have
     *  learned what it means on its own.
     *
     *  Deliberately not 1.0-by-another-name: training on trigger-only rows
     *  EXCLUSIVELY would leave the descriptor path untrained, so the adapter
     *  could produce the album but never be steered ("soad_toxicity, 140 BPM,
     *  acoustic" would mean nothing to it). Mixing keeps both.
     *
     *  This is SimpleTuner's caption_dropout_probability in spirit, but it drops
     *  to the trigger rather than to empty — his LM path raises on an empty
     *  caption, so that knob can never fire there, and MM3 conditions so heavily
     *  on the structured caption that an empty one is far out of distribution.
     *
     *  Drawn from the training RNG, so it is reproducible and survives a resume
     *  along with everything else. */
    double      caption_dropout = 0.0;
    /** Fraction of LoRA rank components zeroed each step (survivors rescaled by
     *  1/(1-p)). 0 = off. bghira runs lora_dropout 0.1.
     *
     *  NOT the same mechanism as PEFT's lora_dropout, which drops elements of
     *  the layer INPUT — that mask is [hidden, S] per module, 88 MB each at our
     *  sequence lengths. This drops rank components instead (LyCORIS calls it
     *  rank_dropout): r floats per step, and each step trains a random
     *  lower-rank subnetwork. Comparable in spirit, cheaper by four orders of
     *  magnitude, and worth stating plainly rather than filing under the same
     *  name. */
    double      rank_dropout = 0.0;
    /** Replace EVERY training caption with the contents of this file.
     *
     *  For a style adapter over one album, a shared caption is not a shortcut —
     *  it is the point. With the caption constant across rows, the only thing
     *  distinguishing them is the audio (and the lyrics), so the adapter has
     *  nowhere to put the style except into itself, and the caption becomes a
     *  handle that summons the album. It also raises the trigger's share of the
     *  prompt from 0.5% of a 1300-token MOSS caption to ~15% of a short one,
     *  which is the difference between a handle and a rounding error.
     *
     *  The .mm3.txt files on disk are never touched. Lyrics still come from the
     *  manifest per row. */
    std::string caption_file;
    std::string dataset_name;
    // Per-layer gradient checkpointing. ON by default and that is not a
    // preference: the MM3 prompt is ~1,100 tokens, so even a 128-frame crop
    // gives S > 1,200 and a naive fwd+bwd retains ~18 GB of activations on top
    // of a 16 GB f16 base. Measured: it spills into WDDM shared memory and a
    // step takes 38 s that should take under one.
    /** Lever A. `bf16` feeds the raw BF16 weight to mul_mat to reach the tensor
     *  cores; `f32-window` is the shipped path, which dequantizes each weight to
     *  F32 in-graph because ggml_out_prod is F32-only.
     *
     *  Requires CUDA (only ggml-cuda carries engine/patches/bf16-out-prod.patch)
     *  AND a BF16-native base (mm3-lm-bf16.gguf, from convert-mm3.py --quant
     *  bf16). Both are checked below and FALL BACK with a warning rather than
     *  failing, because a run that silently ran F32 under a bf16 label is worse
     *  than one that says so.
     *
     *  It CHANGES THE TRAINED WEIGHTS: activation gradients are BF16-rounded at
     *  every layer. That is the trade, and it is recorded in the run log. */
    std::string weights    = "f32-window";   // f32-window | bf16
    /** The acoustic loss (mm3-depth-train.h): teacher-forced CE through the
     *  FROZEN depth decoder, gradient into the adapter via last_hidden. This
     *  is the fix for adapters that shift vocal timbre ("chipmunk"/"goblin"):
     *  the depth decoder consumes the LM's hidden state at render, and a
     *  semantic-only objective leaves that state unconstrained. 0 disables —
     *  which restores the old, known-broken objective; do that only for A/B. */
    double      depth_loss_weight = 1.0;
    int         depth_loss_frames = 128;
    bool        ckpt       = true;
    int         ckpt_chunk = 128;

    // ── Held-out evaluation ────────────────────────────────────────────────
    //
    // The reason this exists: a training loss measured on a RANDOM CROP cannot
    // tell learning from memorising. lm2 bottomed at 0.0003 and was pure
    // sequence memorisation; lm3 ended at 0.031 and was the good run. Nothing
    // in the training curve distinguishes those two — only a fixed, held-out
    // set does, which is also what makes "which checkpoint is best" a decision
    // instead of a retrospective guess.
    float       holdout    = 0.15f;  // fraction of songs withheld; 0 disables
    int         eval_every = 50;     // steps between evaluations; 0 disables
    int64_t     eval_crop  = 400;    // frames per eval crop — SHORTER than a
                                     // training crop on purpose: eval only has
                                     // to be COMPARABLE with itself, and a
                                     // short crop keeps the cost off the run.
    int         eval_crops = 3;      // deterministic crops per held-out song

    // ── Target-loss stopping (2026-08-26) ──────────────────────────────────
    //
    // The second training strategy. `--steps` alone answers "how long do I
    // want to wait"; this answers "how far down do I want the loss", which is
    // the question a user actually has when they do not yet know how many
    // steps this album needs. `--steps` REMAINS the hard cap — a target that
    // is never reached must end the run rather than run forever — exactly as
    // --epochs caps --target-loss in the ACE trainers.
    //
    // WHICH LOSS. A single step's loss here is one crop of one song and swings
    // by more than the whole run's improvement, so stopping on it would stop on
    // noise. Two honest choices, both offered:
    //
    //   train — the mean of the last `target_loss_epochs` COMPLETED PASSES over
    //           the album (prior-preservation steps excluded: soft-target CE
    //           against the frozen base is a different quantity and averaging
    //           the two describes neither). Always available. Cannot tell
    //           learning from memorising, the standing caveat on this curve.
    //
    //           EPOCHS, NOT STEPS, and that is not a cosmetic choice. A fixed
    //           25-step window on an 11-song album spans two passes plus three
    //           songs, so three tracks weigh triple and eight weigh double —
    //           and tracks differ in loss by more than a run improves in fifty
    //           steps. The window would then rise and fall with WHICH songs it
    //           happened to catch, a sawtooth tied to the phase of the pass. A
    //           whole number of passes counts every song identically. It is
    //           also what the DiT trainer's ma5 already does with the same
    //           decision (dit-train-run.h).
    //   eval  — the held-out loss, checked only on the steps that produce a
    //           fresh one. This is the number that means "the adapter
    //           generalises to this artist", and it needs --holdout and
    //           --eval-every to be on.
    //
    // The window must be FULL before the target can fire, so a resume (which
    // starts the window empty) cannot stop on two lucky steps.
    //
    // NOTE THE SCHEDULE. The cosine decays over `--steps`, so a run that stops
    // early stops with the learning rate still part-way down its curve. That is
    // the same trade the ACE trainers make and it is recorded in the run log.
    float       target_loss        = 0.0f;   // 0 = off (run to --steps)
    int         target_loss_epochs = 5;      // completed passes averaged
    std::string target_loss_metric = "train";  // train | eval

    // ── Final resume state ─────────────────────────────────────────────────
    //
    // Pause/resume was built for previews, so state was only ever written when
    // a preview asked for it — which left a FINISHED run holding a state file
    // from its last preview point, tens of steps behind the weights it just
    // exported. "Continue this run for 250 more steps" then silently rewound.
    // Writing the state once more on a clean exit costs one file write of a few
    // seconds and is what makes a completed run resumable at the step it
    // actually reached.
    bool        final_state = true;          // --no-final-state to skip

    // ── Crop position anchoring ────────────────────────────────────────────
    //
    // "song": a crop taken at frame c0 is presented at RoPE positions
    // P + c0 + j, i.e. WHERE IT ACTUALLY IS in the track.
    // "zero": every crop is presented at P + j, as if it were the opening.
    //
    // "zero" was the original behaviour and it is a train/inference mismatch:
    // generation always starts at frame 0, so position P+5 at inference means
    // 0.2 s into the song, while under "zero" the trainer used those same
    // positions to teach material from 60 s in. The model learns that a song
    // can begin anywhere. bghira's 2026-08-22 SOAD campaign reports the two
    // symptoms this predicts — an instant-sound-at-0:00 artifact and tempo
    // drift mid-track — and reports that position-labelled windowed crops fix
    // the pacing. Kept switchable because it changes the recipe: a run trained
    // under "zero" is not comparable with one trained under "song".
    std::string crop_anchor = "song";     // song | zero

    // ── FROZEN KV PREFIX (train/lm-kvprefix.h) ─────────────────────────────
    //
    // Frames of REAL history placed in front of the crop, forward-only and
    // with no gradient: the crop then attends back over the song as it will at
    // generation time instead of starting from an empty context labelled with
    // a late position. 0 disables it and the trainer is unchanged.
    //
    // Costs Nkv*D floats per position per layer (288 KB/position across MM3's
    // 36 layers in F32) plus one forward pass over the prefix per micro-step.
    // No backward, so the O(S^2) attention-score retention that caps --max-
    // frames does not apply.
    int64_t     prefix_frames = 0;
    int         prefix_chunk  = 256;   // prefill positions per graph

    /** Prove the prefix before training on it.
     *
     *  Attention over [prefix ; window] is mathematically identical to
     *  attention over one long crop covering both, so the CE of a supervised
     *  span must not care which way it was produced. Any difference is an
     *  implementation error — a mask off by one column, a stale store, RoPE
     *  applied at the wrong positions. Same discipline as --fd-check: the
     *  runtime-LoKR audit is the standing reminder that "the math reads right"
     *  is not evidence. */
    bool        prefix_selftest = false;

    // ── Pause / resume (mm3-lm-resume.h) ───────────────────────────────────
    //
    // Empty pause_file disables the check entirely. The server uses this to
    // interleave audio previews: touch the sentinel, let the trainer save and
    // exit, render the checkpoint with the whole card, resume.
    std::string pause_file;               // default <out>/PAUSE
    std::string resume_path;              // --resume <state file>
    bool        no_pause = false;         // --no-pause: never look for a sentinel

    // ── Prior preservation (train/mm3-lm-prior.h) ──────────────────────────
    //
    // A second, UNRELATED corpus whose batches are scored against the frozen
    // base model's own next-token distribution instead of their ground-truth
    // codes. The adapter is thereby punished for changing its mind about
    // material that has nothing to do with the artist — which is the only term
    // in this objective that distinguishes "learned the voice" from "rewrote
    // the planner".
    //
    // OFF by default: it needs a corpus the user has to supply, and silently
    // training a different objective than the one asked for would be worse than
    // not offering it.
    std::string reg_manifest, reg_captions_dir, reg_codes_dir;
    /** Every Nth optimizer step is a regularisation step. 0 = off. 3 mirrors
     *  bghira's 1:2 ratio (one prior step for every two style steps).
     *
     *  NOTE THIS DILUTES STYLE EXPOSURE: at --reg-every 3, a 1000-step run
     *  spends only ~667 steps on the artist. His regularised variant raised the
     *  step count to keep style exposure constant, and so should you. */
    int         reg_every = 0;
    /** Classes kept per position. 64 is his; the producer logs the measured
     *  probability mass it covers so the choice is checkable. */
    int         reg_topk = 64;
    /** Where the captured base distributions live. Empty = <reg-codes>/../prior,
     *  so a second run over the same corpus reuses them. */
    std::string reg_prior_dir;
};

struct MM3LmSample {
    std::string          id;
    std::vector<int32_t> prompt;          // tokenised MM3 prompt
    /** The same row with the caption reduced to the TRIGGER WORD ALONE, for
     *  caption dropout. Empty when dropout is off or there is no trigger.
     *
     *  Tokenised up front rather than on demand: it costs one extra pass over a
     *  short string per song at load, and doing it per step would put the BPE
     *  tokenizer inside the training loop. */
    std::vector<int32_t> prompt_trigger_only;
    std::vector<int32_t> codes;           // [n_frames * 8], warm-up row already dropped
    int64_t              n_frames = 0;
};

// ── data ────────────────────────────────────────────────────────────────────

static bool mm3_lm_read_file(const std::string & path, std::string * out) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out->assign((size_t) n, '\0');
    const bool ok = n == 0 || fread(&(*out)[0], 1, (size_t) n, f) == (size_t) n;
    fclose(f);
    return ok;
}

static bool mm3_lm_load_samples(const MM3LmTrainArgs & a, const MM3TrainLm & t,
                                std::vector<MM3LmSample> * out, std::string * err);

// ── the MM3 input embedding ─────────────────────────────────────────────────
//
// (token_embd[semantic + offset] + SUM_c audio_embd[code_c + (c-1)*1024])
// * num_codebooks^-0.5 — the reference's _embed_audio_frame, verbatim, for a
// whole crop at once, with the prompt's ordinary token embeddings in front.
//
// The acoustic indices are uploaded BOOK-MAJOR so the gather comes back as
// [H, Fin, NC] and the sum over books is NC-1 whole-slab adds rather than a
// strided gather.
struct MM3EmbedCtx {
    const MM3TrainLm * t   = nullptr;
    ggml_tensor *      t_prompt = nullptr, * t_sem = nullptr, * t_ac = nullptr;
    int64_t            P = 0, Fin = 0;
};

static ggml_tensor * mm3_lm_build_embed(ggml_context * ctx, const MM3EmbedCtx & e) {
    const MM3TrainLm & t   = *e.t;
    const int64_t      H   = t.lm.cfg.hidden_size;
    const int64_t      NC  = (int64_t) t.num_codebooks - 1;

    // P == 0 is reachable only from the KV prefill, whose chunks can land
    // wholly inside the audio; every training crop carries its prompt.
    ggml_tensor * e_prompt =
        e.P > 0 ? ggml_get_rows(ctx, t.lm.embed_tokens, ggml_view_1d(ctx, e.t_prompt, e.P, 0)) : nullptr;
    if (e.Fin <= 0) {
        GGML_ASSERT(e_prompt && "an embedding of nothing was requested");
        return e_prompt;
    }
    ggml_tensor * e_sem = ggml_get_rows(ctx, t.lm.embed_tokens, ggml_view_1d(ctx, e.t_sem, e.Fin, 0));
    ggml_tensor * e_ac  = ggml_reshape_3d(
        ctx, ggml_get_rows(ctx, t.audio_embd, ggml_view_1d(ctx, e.t_ac, e.Fin * NC, 0)), H, e.Fin, NC);
    ggml_tensor * acc = ggml_view_2d(ctx, e_ac, H, e.Fin, e_ac->nb[1], 0);
    for (int64_t k = 1; k < NC; k++) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, e_ac, H, e.Fin, e_ac->nb[1], (size_t) k * e_ac->nb[2]));
    }
    ggml_tensor * frames = ggml_scale(ctx, ggml_add(ctx, e_sem, acc), t.embedding_scale);
    return e_prompt ? ggml_concat(ctx, e_prompt, frames, 1) : frames;
}

// The LmCkptRun hook. P1 calls this instead of get_rows on token ids.
static ggml_tensor * mm3_lm_ckpt_embed(ggml_context * ctx, LmCkptRun & r, int S) {
    const MM3EmbedCtx & e = *(const MM3EmbedCtx *) r.embed_user;
    GGML_ASSERT(e.P + e.Fin == (int64_t) S);
    return mm3_lm_build_embed(ctx, e);
}

// ── FROZEN KV PREFIX (train/lm-kvprefix.h) ──────────────────────────────────
//
// The prefill stream is [prompt ; frames pfx_lo .. c0-1] and lm_kvprefix_run
// walks it in chunks, so this builder has to answer for an ARBITRARY [i0, i0+n)
// slice of it: part prompt and part audio, or either alone. Stream position P
// is frame pfx_lo.
//
// Its own t_sem / t_ac, sized for one chunk, because the window's are live with
// the crop for the whole micro-step.
struct MM3PrefixCtx {
    ggml_tensor *        t_sem = nullptr, * t_ac = nullptr;
    const int32_t *      prompt = nullptr;   // P ids
    const int32_t *      codes  = nullptr;   // the song's [n_frames, 8]
    int64_t              P = 0, pfx_lo = 0;
    int64_t              NC = 0, AV = 0, sem_off = 0;
    std::vector<int32_t> sem, ac;
    MM3EmbedCtx          e;                  // t / t_prompt shared with the window
};

static ggml_tensor * mm3_lm_prefix_embed(ggml_context * ctx, void * user, int64_t i0, int64_t n) {
    MM3PrefixCtx & c   = *(MM3PrefixCtx *) user;
    const int64_t  p_n = std::max<int64_t>(0, std::min<int64_t>(c.P - i0, n));
    const int64_t  f_n = n - p_n;

    if (p_n > 0) {
        ggml_backend_tensor_set(c.e.t_prompt, c.prompt + i0, 0, (size_t) p_n * sizeof(int32_t));
    }
    if (f_n > 0) {
        const int64_t f0 = c.pfx_lo + (i0 + p_n - c.P);
        c.sem.resize((size_t) f_n);
        c.ac.resize((size_t) (f_n * c.NC));
        for (int64_t i = 0; i < f_n; i++) {
            const int32_t * f  = &c.codes[(size_t) ((f0 + i) * 8)];
            c.sem[(size_t) i]  = f[0] + (int32_t) c.sem_off;
            for (int64_t k = 0; k < c.NC; k++) {
                c.ac[(size_t) (k * f_n + i)] = f[1 + k] + (int32_t) (k * c.AV);
            }
        }
        ggml_backend_tensor_set(c.t_sem, c.sem.data(), 0, c.sem.size() * sizeof(int32_t));
        ggml_backend_tensor_set(c.t_ac, c.ac.data(), 0, c.ac.size() * sizeof(int32_t));
    }
    // The window's ctx names the window's t_sem/t_ac; swap in the prefill's for
    // this build and put them back, so one MM3EmbedCtx keeps describing the
    // frame layout in exactly one place.
    MM3EmbedCtx e = c.e;
    e.t_sem       = c.t_sem;
    e.t_ac        = c.t_ac;
    e.P           = p_n;
    e.Fin         = f_n;
    return mm3_lm_build_embed(ctx, e);
}

// Read the manifest into teacher-forced samples. Factored out because the
// FD-check entry point needs exactly the same data path as training — a
// gradient check against a DIFFERENT sequence than the trainer builds would
// verify nothing that matters.
static bool mm3_lm_load_samples_from(const std::string & manifest, const std::string & captions_dir,
                                     const std::string & codes_dir, const std::string & lm_path,
                                     const std::string & trigger_prefix,
                                     const std::string & caption_override,
                                     const MM3TrainLm & t,
                                     std::vector<MM3LmSample> * out, std::string * err) {
    struct { std::string manifest, captions_dir, codes_dir, lm_path; } a {
        manifest, captions_dir, codes_dir, lm_path };
    std::vector<MM3LmSample> & samples = *out;
        std::string jbuf;
        if (!mm3_lm_read_file(a.manifest, &jbuf)) {
            if (err) *err = "cannot read " + a.manifest;
            return false;
        }
        yyjson_doc * doc = yyjson_read(jbuf.c_str(), jbuf.size(), 0);
        yyjson_val * arr = doc ? yyjson_obj_get(yyjson_doc_get_root(doc), "samples") : nullptr;
        if (!arr || !yyjson_is_arr(arr)) {
            if (err) *err = a.manifest + " has no `samples` array";
            if (doc) yyjson_doc_free(doc);
            return false;
        }
        MM3Model stub = {};
        stub.lm_file.found = true;
        stub.lm_file.path  = a.lm_path;
        stub.lm_file.name  = a.lm_path;
        stub.lm_cfg.semantic_vocab_offset = t.semantic_vocab_offset;
        MM3Tokenizer tok = {};
        if (!mm3_tokenizer_load(stub, &tok, err)) {
            yyjson_doc_free(doc);
            return false;
        }

        yyjson_val *    s;
        yyjson_arr_iter it = yyjson_arr_iter_with(arr);
        while ((s = yyjson_arr_iter_next(&it))) {
            auto js = [&](const char * k) -> std::string {
                yyjson_val * v = yyjson_obj_get(s, k);
                return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v)) : std::string();
            };
            const std::string id = js("id"), filename = js("filename"), lyrics = js("lyrics");
            std::string       stem = filename;
            const size_t      d    = stem.find_last_of('.');
            if (d != std::string::npos) stem = stem.substr(0, d);

            // The MM3 caption lives beside the audio as <stem>.mm3.txt, exactly
            // as mm3-condition reads it. REFUSE rather than fall back to the ACE
            // caption in the manifest: an ACE caption trains the wrong genre and
            // the failure is invisible.
            std::string caption;
            if (!caption_override.empty()) {
                caption = caption_override;
            } else if (!mm3_lm_read_file(a.captions_dir + "/" + stem + ".mm3.txt", &caption)) {
                fprintf(stderr, "[mm3-lm-train] SKIP %s: no %s.mm3.txt\n", id.c_str(), stem.c_str());
                continue;
            }
            std::string cbuf;
            if (!mm3_lm_read_file(a.codes_dir + "/" + id + ".codes", &cbuf) || cbuf.size() < 16 * sizeof(int32_t)) {
                fprintf(stderr, "[mm3-lm-train] SKIP %s: no usable %s.codes\n", id.c_str(), id.c_str());
                continue;
            }
            const int64_t n_rows = (int64_t) (cbuf.size() / sizeof(int32_t)) / 8;
            MM3LmSample   sm;
            sm.id       = id;
            sm.n_frames = n_rows - 1;                        // drop the warm-up row
            sm.codes.resize((size_t) (sm.n_frames * 8));
            memcpy(sm.codes.data(), cbuf.data() + 8 * sizeof(int32_t), sm.codes.size() * sizeof(int32_t));
            if (!trigger_prefix.empty()) {
                // Front of the FIRST line, comma + space — the training-row shape.
                size_t lead = caption.find_first_not_of(" \t\r\n\xEF\xBB\xBF");
                std::string body = lead == std::string::npos ? caption : caption.substr(lead);
                // IDEMPOTENT. A shared caption written for a style adapter will
                // usually open with the trigger already, and prepending a second
                // copy trains "system of a down, system of a down, ..." — a token
                // sequence no render will ever reproduce.
                std::string lb = body.substr(0, trigger_prefix.size()), lp = trigger_prefix;
                for (auto & ch : lb) ch = (char) tolower((unsigned char) ch);
                for (auto & ch : lp) ch = (char) tolower((unsigned char) ch);
                caption = (lb == lp) ? body : trigger_prefix + body;
            }
            if (samples.empty()) {
                // The first assembled caption, once, so a trigger that is not
                // actually in the prompt is visible in the log instead of being
                // discovered weeks later by ear.
                const std::string head = caption.substr(0, caption.find('\n'));
                fprintf(stderr, "[mm3-lm-train] first training caption begins: %.120s\n", head.c_str());
            }
            mm3_tokenizer_encode(tok, mm3_assemble_prompt(caption, lyrics), &sm.prompt);
            if (!trigger_prefix.empty()) {
                // Caption dropout's alternative prompt: the trigger and nothing
                // else. Lyrics are KEPT — dropping those too would change what
                // the model is being asked to sing, not how it is described.
                std::string bare = trigger_prefix;
                while (!bare.empty() && (bare.back() == ' ' || bare.back() == ',')) bare.pop_back();
                mm3_tokenizer_encode(tok, mm3_assemble_prompt(bare, lyrics), &sm.prompt_trigger_only);
            }
            if (sm.prompt.empty() || sm.n_frames < 8) {
                fprintf(stderr, "[mm3-lm-train] SKIP %s: empty prompt or %lld frames\n", id.c_str(),
                        (long long) sm.n_frames);
                continue;
            }
            samples.push_back(std::move(sm));
        }
        yyjson_doc_free(doc);
    return true;
}

static bool mm3_lm_load_samples(const MM3LmTrainArgs & a, const MM3TrainLm & t,
                                std::vector<MM3LmSample> * out, std::string * err) {
    std::string shared;
    if (!a.caption_file.empty() && !mm3_lm_read_file(a.caption_file, &shared)) {
        if (err) *err = "cannot read --caption-file " + a.caption_file;
        return false;
    }
    while (!shared.empty() && (shared.back() == '\n' || shared.back() == '\r'
                               || shared.back() == ' ')) {
        shared.pop_back();
    }
    return mm3_lm_load_samples_from(a.manifest, a.captions_dir, a.codes_dir, a.lm_path,
                                    a.trigger_prepend && !a.trigger.empty() ? a.trigger + ", " : "",
                                    shared, t, out, err);
}

// ── finite-difference gradient check ────────────────────────────────────────
//
// The decisive correctness gate for the backward, and the one the DiT-trainer
// fiasco is a warning about: there, the graph trained, the loss fell, and the
// DELTAS were wrong. A falling loss is necessary and nowhere near sufficient.
//
// For a handful of individual LoRA weights this measures
//
//     numeric  = (L(w + eps) - L(w - eps)) / (2 eps)      central difference
//     analytic = the accumulated gradient the backward produced
//
// and reports the relative error. It runs the check TWICE — once through the
// naive fwd+bwd graph and once through the CHECKPOINTED path — because those
// are two different pieces of machinery that must agree with each other and
// with the numbers. That second comparison is what actually tests the hooks
// this program added to lm-ckpt.h (untied scored head, frame-embedding entry).
//
// Two things make this work at all, and both are borrowed from the ACE
// self-test rather than rediscovered:
//   * B MUST BE INITIALISED NON-ZERO. With PEFT's B = 0, dL/dA is identically
//     zero by construction, so a check on A would "pass" against a graph that
//     computes nothing. b_sigma is 1e-2 here for the same reason lm-selftest.h
//     uses it.
//   * a SMALL rank and a SHORT crop, so the naive path fits and the whole
//     check is seconds rather than minutes.
//
// ── WHAT THIS GATE CAN AND CANNOT DECIDE, MEASURED ─────────────────────────
//
// The finite-difference arm is INCONCLUSIVE against an f16 base, and that is a
// property of the model rather than a bug to fix. Three measurements pin it:
//   * the loss is perfectly deterministic (repeat delta exactly 0.00e+00), so
//     this is not run-to-run noise;
//   * relative error grows ~10x for every 10x DECREASE in eps — the signature
//     of catastrophic cancellation, i.e. the loss CHANGE is below the forward's
//     arithmetic resolution, the opposite of a truncation problem;
//   * moving the cross-entropy off the GPU and into host double changed the
//     numbers in the 5th significant figure only, so the floor is in the
//     LOGITS (36 layers of f16 matmul), not in the CE aggregation.
// Only the probe with the largest ||g|| clears the bar (0.2 %), and the error
// tracks 1/||g|| exactly as a fixed absolute floor predicts.
//
// ── --f32-layers 2: THE VERDICT ────────────────────────────────────────────
//
// Both problems above are the SAME problem — f16 rounding is larger than the
// defect being looked for — and one switch removes it. `--f32-layers N`
// (mm3-f32-isolate.h) truncates the trunk to N layers and mirrors those layers
// plus the scored head slice to F32, for ~1.7 GB instead of the ~34 GB a full
// 8.6B F32 mirror would need. Measured at N=2, both checks change character:
//
//                          f16, 36 layers        F32-isolated, 2 layers
//   ckpt vs naive          1.35e-02  (report)    3.78e-07  PASS (bar 2e-3)
//   finite differences     1 of 6 within 15%     6 of 6, worst 0.002 (bar 2e-2)
//
// The route comparison lands 5,300x inside ACE's own bar, and the FD arm goes
// from noise to three-decimal agreement. So the f16 numbers were arithmetic all
// along, not wiring — but that could only be ASSERTED before and is MEASURED
// now, which is the whole point.
//
// ── PROVEN BY NEGATIVE CONTROL, not by passing ──────────────────────────────
//
// A gate that has only ever passed is a green light, not a gate. Two faults
// were injected, built and measured, and the gate is kept honest by them:
//
//   1. CKPT ARM ONLY — the checkpointed supervised window shifted one position
//      (s2.n_masked = P-1). Route comparison 3.78e-07 -> 3.14e-01, GATE FAIL,
//      exit 1. The FD gate stayed PASS, correctly: the fault was not in the arm
//      FD probes. Under f16 the same fault moved the number (7.70e-02 ->
//      2.14e+00) but there was no bar, so it still exited 0 — which is exactly
//      the blind spot isolation removes.
//
//   2. WRONG dL/dloss — the loss-gradient seed set to 2.0. FD 0/6 probes, worst
//      0.5005, GATE FAIL: the signature of an analytic gradient exactly twice
//      the true one, |1-2|/2. (The route comparison flagged it too, so this
//      control does NOT demonstrate the both-arms-share-it case; it
//      demonstrates that FD detects a scale error, with the factor readable
//      straight off the number.)
//
// So the checks are complementary by CONSTRUCTION: the route comparison tests
// one backward against the other, FD tests a backward against the FORWARD.
// Control 1 shows the first catching what the second cannot. A defect shared by
// both backward routes is the case only FD could catch, and that remains
// reasoning rather than a measured result.
//
// Run the gate on the f16 base even when training on q8_0: isolating a
// quantized base would measure the quantizer rather than the wiring, so
// mm3_f32_isolate() refuses it outright.
//
// IT PERTURBS A DIRECTION, NOT A SINGLE WEIGHT, and that is not a detail.
// A per-entry difference was tried first and measured nothing but noise: with
// a per-entry gradient of order 1e-4 and eps 1e-2, the true loss change is
// ~2e-6, which is single digits of f32 ULP on a loss of ~4.1. The measured
// "numeric" column came out ~1e-2 with random signs — pure rounding.
//
// So each probe perturbs a WHOLE LoRA factor along the unit gradient direction
// v = g/||g||. The directional derivative is then exactly ||g||, and the loss
// change is ~2*eps*||g|| — thousands of times the noise floor, because every
// entry contributes with the same sign instead of cancelling. This is the
// standard way to finite-difference a low-precision model, and it still
// catches every failure that matters: a wrong sign flips the numeric value
// negative, a wrong scale shows up directly in the ratio, and a structurally
// zero gradient gives ||g|| = 0 with a non-zero measured change.
static int mm3_lm_fdcheck_main(const MM3LmTrainArgs & a, int n_probe, double eps, int64_t frames,
                               int64_t prompt_cap, int f32_layers) {
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
    std::string err;
    MM3TrainLm  t = {};
    if (!mm3_train_lm_load(&t, a.lm_path.c_str(), &err) ||
        !mm3_train_lm_load_audio_embd(&t, a.depth_path.c_str(), &err)) {
        fprintf(stderr, "[mm3-fd] load failed: %s\n", err.c_str());
        mm3_train_lm_free(&t);
        return 1;
    }
    // F32 ISOLATION, and the reason this command can return a verdict at all.
    // Must happen before ANYTHING reads the layer count or the head: it
    // truncates the trunk and swaps lm_head for a pre-sliced F32 copy.
    MM3F32Slice iso;
    if (f32_layers > 0 && !mm3_f32_isolate(&t, f32_layers, &iso, &err)) {
        fprintf(stderr, "[mm3-fd] %s\n", err.c_str());
        mm3_f32_isolate_free(&iso);
        mm3_train_lm_free(&t);
        return 1;
    }
    const bool isolated = f32_layers > 0;

    const Qwen3LMConfig & c  = t.lm.cfg;
    const int64_t         H  = c.hidden_size;
    const int64_t         NC = (int64_t) t.num_codebooks - 1;
    const int64_t         AV = t.acoustic_vocab_size;
    const int64_t         SL = mm3_lm_train_slice_size(t);

    std::vector<MM3LmSample> samples;
    if (!mm3_lm_load_samples(a, t, &samples, &err) || samples.empty()) {
        fprintf(stderr, "[mm3-fd] no samples: %s\n", err.c_str());
        mm3_train_lm_free(&t);
        return 1;
    }
    const MM3LmSample & smp = samples[0];
    // TRUNCATE THE PROMPT. A real MM3 prompt is ~1,125 tokens, which would put
    // the NAIVE arm of this check back over the card for exactly the reason
    // the trainer needed checkpointing — and the naive arm is half of what is
    // being compared. A gradient check needs the same graph STRUCTURE, not a
    // meaningful caption; cutting mid-BPE is acceptable here and nowhere else.
    const int64_t       P   = std::min<int64_t>(prompt_cap, (int64_t) smp.prompt.size());
    const int64_t       K   = std::min<int64_t>(frames, smp.n_frames);
    const int64_t       Fin = K - 1;          // never at_end: keep the case simple
    const int64_t       n_sup = K;
    const int64_t       S   = P + Fin;
    fprintf(stderr, "[mm3-fd] %s: prompt %lld (of %zu, truncated) + %lld frames = seq %lld, rank %d, eps %.3g\n",
            smp.id.c_str(), (long long) P, smp.prompt.size(), (long long) K, (long long) S, a.rank, eps);

    LmLora     lora;
    const bool fd_lokr = a.is_lokr();
    const bool fd_init_ok =
        fd_lokr ? lm_lokr_init(&lora, &t.lm, 0, c.n_layers, a.lokr_dim, a.lokr_alpha, a.lokr_factor,
                               a.lokr_decompose_both, (uint64_t) a.seed, &err)
                : lm_lora_init(&lora, &t.lm, 0, c.n_layers, a.rank, (float) a.alpha, (uint64_t) a.seed,
                               /*b_sigma=*/1e-2f, &err);
    if (!fd_init_ok) {
        fprintf(stderr, "[mm3-fd] %s init failed: %s\n", fd_lokr ? "LoKr" : "LoRA", err.c_str());
        mm3_train_lm_free(&t);
        return 1;
    }
    if (fd_lokr) {
        // The LoKr equivalent of the LoRA path's b_sigma. lm_lokr_init zeroes w2
        // (monolithic) or w2_b (factorized) so the delta starts at EXACTLY zero,
        // which is right for training and useless for a gradient check: with
        // dW = kron(w1, w2) and w2 == 0, dL/dw1 is identically zero, so a probe on
        // w1 compares 0 against 0 and reports a pass while measuring nothing.
        LmRng rng;
        lm_rng_seed(&rng, (uint64_t) a.seed ^ 0xD1B54A32D192ED03ull);
        std::vector<float> v;
        size_t             perturbed = 0;
        for (int l = 0; l < c.n_layers; l++) {
            for (int sl = 0; sl < QW_LORA_NSLOTS; sl++) {
                QwLoraPair & pr = lora.layers[l].p[sl];
                ggml_tensor * z = pr.w2 ? pr.w2 : pr.w2_b;
                if (!z) {
                    continue;
                }
                v.assign((size_t) ggml_nelements(z), 0.0f);
                lm_rng_fill_normal(&rng, v, 1e-2f);
                ggml_backend_tensor_set(z, v.data(), 0, v.size() * sizeof(float));
                perturbed++;
            }
        }
        fprintf(stderr, "[mm3-fd] LoKr: perturbed %zu w2 tensors to sigma 1e-2 so the w1 gradient is not identically zero\n", perturbed);
    }
    LmOptim opt;
    opt.optimizer = "adamw";
    if (!lm_optim_init(&opt, lora.params, t.lm.backend, &err)) {
        fprintf(stderr, "[mm3-fd] optimizer init failed: %s\n", err.c_str());
        lm_lora_detach(&lora, &t.lm);
        lm_lora_free(&lora);
        mm3_train_lm_free(&t);
        return 1;
    }

    // ── inputs, fixed for the whole check ──
    ggml_context * ctx_static = nullptr;
    {
        ggml_init_params ip = { 32 * ggml_tensor_overhead(), nullptr, true };
        ctx_static          = ggml_init(ip);
    }
    ggml_tensor * t_prompt = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, P);
    ggml_tensor * t_sem    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, Fin);
    ggml_tensor * t_ac     = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, Fin * NC);
    ggml_tensor * t_pos    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, S);
    ggml_tensor * t_msk    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, S * S);
    ggml_tensor * t_lab    = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, SL, n_sup);
    ggml_tensor * t_lg     = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_clip   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_epsT   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_gn2    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_adamw  = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 7);
    ggml_tensor * t_gs     = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_one    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_tok    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, S);
    for (ggml_tensor * x : { t_prompt, t_sem, t_ac, t_pos, t_msk, t_lab, t_tok }) ggml_set_input(x);
    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, t.lm.backend);
    ggml_backend_buffer_clear(buf_static, 0);
    {
        const float one = 1.0f, ep = 1e-6f;
        ggml_backend_tensor_set(t_lg, &one, 0, sizeof(float));
        ggml_backend_tensor_set(t_one, &one, 0, sizeof(float));
        ggml_backend_tensor_set(t_clip, &one, 0, sizeof(float));
        ggml_backend_tensor_set(t_epsT, &ep, 0, sizeof(float));
    }
    opt.t_adamw = t_adamw; opt.t_lossgrad = t_lg; opt.t_clip = t_clip;
    opt.t_eps = t_epsT; opt.t_gnorm2 = t_gn2;

    std::vector<int32_t> sem_in((size_t) Fin), ac_in((size_t) (Fin * NC)), tgt((size_t) n_sup), pos((size_t) S);
    for (int64_t i = 0; i < Fin; i++) {
        const int32_t * f = &smp.codes[(size_t) (i * 8)];
        sem_in[(size_t) i] = f[0] + (int32_t) t.semantic_vocab_offset;
        for (int64_t k = 0; k < NC; k++) ac_in[(size_t) (k * Fin + i)] = f[1 + k] + (int32_t) (k * AV);
    }
    for (int64_t j = 0; j < n_sup; j++) tgt[(size_t) j] = mm3_lm_train_slice_index(t, smp.codes[(size_t) (j * 8)]);
    for (int64_t i = 0; i < S; i++) pos[(size_t) i] = (int32_t) i;
    std::vector<float> msk;
    lm_causal_mask((int) S, &msk);
    ggml_backend_tensor_set(t_prompt, smp.prompt.data(), 0, (size_t) P * sizeof(int32_t));
    ggml_backend_tensor_set(t_sem, sem_in.data(), 0, sem_in.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_ac, ac_in.data(), 0, ac_in.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_msk, msk.data(), 0, msk.size() * sizeof(float));

    MM3EmbedCtx embed_ctx{ &t, t_prompt, t_sem, t_ac, P, Fin };

    std::vector<uint8_t> arena((size_t) 512 << 20);
    BackendPair          bp;
    bp.backend = t.lm.backend; bp.cpu_backend = t.lm.cpu_backend;
    bp.has_gpu = t.lm.backend != t.lm.cpu_backend;
    ggml_backend_sched_t sched = backend_sched_new(bp, 65536);

    // Forward-only loss for the numeric side.
    //
    // THE CE IS COMPUTED ON THE HOST IN DOUBLE, from downloaded logits, and
    // that is the whole reason this check works. ggml's in-graph
    // cross_entropy_loss is a 16,389-way logsumexp averaged over rows, all in
    // f32: perfectly deterministic (measured: repeat delta exactly 0) but only
    // ~1e-4 ACCURATE. Differencing two such values is catastrophic
    // cancellation — which showed up unmistakably as relative error that grew
    // 10x for every 10x DECREASE in eps, the opposite of truncation.
    // Aggregating in double removes that floor, and as a bonus makes the
    // numeric side an INDEPENDENT implementation of the loss rather than the
    // same kernel twice.
    std::vector<float> lg_host((size_t) (SL * n_sup));
    auto forward_loss = [&]() -> double {
        ggml_init_params gip = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(gip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, false);
        ggml_tensor *    h_in = mm3_lm_build_embed(ctx, embed_ctx);
        ggml_tensor *    hid  = lm_build_trunk_embeds(ctx, &t.lm, h_in, t_pos, t_msk, (int) S);
        ggml_tensor *    hd   = ggml_cont(
            ctx, ggml_view_2d(ctx, hid, H, n_sup, hid->nb[1], (size_t) (P - 1) * hid->nb[1]));
        ggml_tensor * lg = ggml_mul_mat(ctx, mm3_lm_train_out_slice(ctx, t), hd);   // [SL, n_sup]
        ggml_set_output(lg);
        ggml_build_forward_expand(gf, lg);
        double v = std::nan("");
        ggml_backend_sched_reset(sched);
        if (ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS) {
            ggml_backend_tensor_get(lg, lg_host.data(), 0, lg_host.size() * sizeof(float));
            double sum = 0.0;
            for (int64_t r = 0; r < n_sup; r++) {
                const float * row = lg_host.data() + (size_t) (r * SL);
                double        mx  = row[0];
                for (int64_t j = 1; j < SL; j++) if (row[j] > mx) mx = row[j];
                double se = 0.0;
                for (int64_t j = 0; j < SL; j++) se += std::exp((double) row[j] - mx);
                sum += mx + std::log(se) - (double) row[(size_t) tgt[(size_t) r]];
            }
            v = sum / (double) n_sup;
        }
        ggml_free(ctx);
        return v;
    };

    // One naive fwd+bwd; gradients land in opt.acc[].
    auto backward_naive = [&]() -> bool {
        ggml_backend_buffer_clear(opt.buf_grad, 0);
        ggml_init_params gip = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(gip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
        ggml_tensor *    h_in = mm3_lm_build_embed(ctx, embed_ctx);
        ggml_tensor *    hid  = lm_build_trunk_embeds(ctx, &t.lm, h_in, t_pos, t_msk, (int) S);
        ggml_tensor *    hd   = ggml_cont(
            ctx, ggml_view_2d(ctx, hid, H, n_sup, hid->nb[1], (size_t) (P - 1) * hid->nb[1]));
        ggml_tensor * lg   = ggml_mul_mat(ctx, mm3_lm_train_out_slice(ctx, t), hd);
        ggml_tensor * labv = ggml_view_2d(ctx, t_lab, SL, n_sup, t_lab->nb[1], 0);
        ggml_tensor * loss = ggml_cross_entropy_loss(ctx, lg, labv);
        ggml_set_loss(loss);
        ggml_set_output(loss);
        ggml_build_forward_expand(gf, loss);
        std::vector<ggml_tensor *> gacc;
        lm_optim_fill_gacc(&opt, gf, &gacc);
        ggml_build_backward_expand(ctx, gf, gacc.data());
        bool ok = false;
        {
            LmLabelGuard guard(t_lab, tgt.data(), (int) n_sup, (int) SL);
            ggml_backend_sched_reset(sched);
            ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
        }
        ggml_free(ctx);
        return ok;
    };

    // The same thing through the checkpointed path.
    LmCkptState ckpt_st;
    LmCkptRun   ckpt_run;
    auto backward_ckpt = [&]() -> bool {
        ggml_backend_buffer_clear(opt.buf_grad, 0);
        LmSample s2;
        s2.tokens.assign((size_t) S, 0);
        s2.targets  = tgt;
        s2.n_masked = (int) P;
        s2.s_tr     = (int) n_sup;
        double ce = 0.0;
        return lm_ckpt_micro_step(ckpt_run, s2, true, &ce);
    };

    auto grad_vec = [&](ggml_tensor * par) -> std::vector<float> {
        auto it = opt.param_slot.find(par);
        GGML_ASSERT(it != opt.param_slot.end());
        ggml_tensor *      acc = opt.acc[(size_t) it->second];
        std::vector<float> g((size_t) ggml_nelements(acc));
        ggml_backend_tensor_get(acc, g.data(), 0, g.size() * sizeof(float));
        return g;
    };

    // Probe a spread of layers, both factors and several module slots, so the
    // check also exercises the layer/slot indexing rather than one lucky spot.
    struct Probe { int layer, slot; bool is_a; };
    // `is_a` means FIRST factor, whichever parameterization is in play:
    //   LoRA -> A / B        LoKr -> w1 / (w2 | w2_a)
    // Without this the probes address q.A and q.B, which are null on a LoKr
    // pair, and grad_vec trips GGML_ASSERT on the param_slot lookup.
    auto probe_tensor = [&](const QwLoraPair & q, bool first) -> ggml_tensor * {
        if (q.has_lokr()) {
            return first ? q.w1 : (q.w2 ? q.w2 : q.w2_a);
        }
        return first ? q.A : q.B;
    };
    auto probe_suffix = [&](const QwLoraPair & q, bool first) -> const char * {
        if (q.has_lokr()) {
            return first ? "lokr_w1" : (q.w2 ? "lokr_w2" : "lokr_w2_a");
        }
        return first ? "A" : "B";
    };
    std::vector<Probe> probes;
    {
        const int layers[3] = { 0, c.n_layers / 2, c.n_layers - 1 };
        const int slots[3]  = { QW_LORA_Q, QW_LORA_GATE, QW_LORA_DOWN };
        for (int i = 0; i < n_probe; i++) {
            probes.push_back(Probe{ layers[i % 3], slots[(i / 3) % 3], (i % 2) == 0 });
        }
    }

    // The noise floor, measured rather than assumed: two evaluations of the
    // SAME configuration. Anything the difference test claims below has to be
    // large compared to this, and printing it is what turned a mystifying
    // per-entry result into an obvious one.
    const double l0 = forward_loss();
    const double l1 = forward_loss();
    fprintf(stderr, "[mm3-fd] base loss %.6f (repeat %.6f, |delta| %.2e)\n", l0, l1, std::abs(l1 - l0));

    if (!backward_naive()) {
        fprintf(stderr, "[mm3-fd] naive backward failed\n");
        return 1;
    }
    // Whole-tensor gradients, one vector per probe.
    std::vector<std::vector<float>> g_naive;
    for (const Probe & pr : probes) {
        const QwLoraPair & q = lora.layers[pr.layer].p[pr.slot];
        g_naive.push_back(grad_vec(probe_tensor(q, pr.is_a)));
    }

    // Checkpointed gradients for the same probes.
    std::vector<std::vector<float>> g_ckpt;
    {
        LmCkptCfg cc;
        cc.chunk     = 64;
        cc.s_max     = (int) S;
        cc.layer_lo  = 0;
        cc.layer_hi  = c.n_layers;
        cc.head_w    = t.lm_head;
        cc.head_row0 = t.head_slice_row0;
        cc.head_v    = (int) SL;
        if (!lm_ckpt_alloc(&ckpt_st, &t.lm, cc, &err) || !lm_ckpt_build_embed_t(&ckpt_st, &err)) {
            fprintf(stderr, "[mm3-fd] checkpoint setup failed: %s\n", err.c_str());
        } else {
            ckpt_run.lm = &t.lm; ckpt_run.opt = &opt; ckpt_run.st = &ckpt_st; ckpt_run.sched = sched;
            ckpt_run.t_tok = t_tok; ckpt_run.t_pos = t_pos; ckpt_run.t_msk = t_msk;
            ckpt_run.t_gs = t_gs;   ckpt_run.t_one = t_one; ckpt_run.grad_accum = 1;
            ckpt_run.embed_build = mm3_lm_ckpt_embed; ckpt_run.embed_user = &embed_ctx;
            if (backward_ckpt()) {
                for (const Probe & pr : probes) {
                    const QwLoraPair & q = lora.layers[pr.layer].p[pr.slot];
                    g_ckpt.push_back(grad_vec(probe_tensor(q, pr.is_a)));
                }
            } else {
                fprintf(stderr, "[mm3-fd] checkpointed backward failed\n");
            }
        }
    }

    // ── numeric: directional derivative along v = g/||g|| ──
    fprintf(stderr, "\n[mm3-fd] %-26s %10s %13s %13s %8s\n", "probe (whole tensor)", "n", "||g||",
            "numeric", "rel");
    int                 n_bad = 0;
    double              worst = 0.0;
    std::vector<double> fd_rel;
    bool                gate_fd_ran = false, gate_fd_pass = false;
    for (size_t i = 0; i < probes.size(); i++) {
        const Probe &      pr  = probes[i];
        const QwLoraPair & q   = lora.layers[pr.layer].p[pr.slot];
        ggml_tensor *      par = probe_tensor(q, pr.is_a);

        const std::vector<float> & g = g_naive[i];
        double norm2 = 0.0;
        for (float x : g) norm2 += (double) x * (double) x;
        const double gnorm = std::sqrt(norm2);

        std::vector<float> w0((size_t) ggml_nelements(par)), wtmp(w0.size());
        ggml_backend_tensor_get(par, w0.data(), 0, w0.size() * sizeof(float));

        double num = std::nan("");
        if (gnorm > 0.0) {
            for (size_t k = 0; k < w0.size(); k++) wtmp[k] = (float) (w0[k] + eps * g[k] / gnorm);
            ggml_backend_tensor_set(par, wtmp.data(), 0, wtmp.size() * sizeof(float));
            const double lp = forward_loss();
            for (size_t k = 0; k < w0.size(); k++) wtmp[k] = (float) (w0[k] - eps * g[k] / gnorm);
            ggml_backend_tensor_set(par, wtmp.data(), 0, wtmp.size() * sizeof(float));
            const double lmn = forward_loss();
            num = (lp - lmn) / (2.0 * eps);
        }
        ggml_backend_tensor_set(par, w0.data(), 0, w0.size() * sizeof(float));

        const double rel = std::abs(num - gnorm) / std::max(1e-12, gnorm);
        char         nm[64];
        snprintf(nm, sizeof(nm), "L%d.%s.%s", pr.layer, lm_slot_peft_name(pr.slot),
                 probe_suffix(q, pr.is_a));
        fprintf(stderr, "[mm3-fd] %-26s %10zu %13.6e %13.6e %8.3f\n", nm, g.size(), gnorm, num, rel);
        fd_rel.push_back(rel);
        if (!(rel < 0.15)) n_bad++;
        worst = std::max(worst, rel);
    }

    // Whether this is a VERDICT or a note depends entirely on isolation.
    //
    // Against the f16 base it is a note: the difference is below the forward own
    // resolution for every probe but the largest-gradient one, so a pass/fail
    // would be theatre. Under F32 isolation it is a verdict, and a valuable one
    // — it is the ONLY check here that tests the backward against the FORWARD
    // rather than against another backward, so it catches a defect that both
    // gradient routes could share.
    //
    // The bar is 2e-2, not tighter. This is a central difference with eps 1e-2,
    // so it carries a genuine O(eps^2 * third-derivative) truncation error that
    // no amount of precision removes; 2e-2 is ~10x the worst observed (2e-3 at
    // 2 layers), which leaves room for probe-to-probe variation without
    // admitting a real scale error — a wrong gradient scale misses by a FACTOR,
    // not by a percent.
    const double fd_bar = isolated ? 2e-2 : 0.15;
    n_bad = 0;
    for (double r : fd_rel) {
        if (!(r < fd_bar)) n_bad++;
    }
    if (isolated) {
        const bool fd_ok = n_bad == 0;
        gate_fd_ran  = true;
        gate_fd_pass = fd_ok;
        fprintf(stderr,
                "\n[mm3-fd] GATE %s: finite differences, F32-isolated, bar %.0e, %d/%zu probes, worst %.4f\n",
                fd_ok ? "PASS" : "FAIL", fd_bar, (int) probes.size() - n_bad, probes.size(), worst);
        if (!fd_ok) {
            fprintf(stderr,
                    "[mm3-fd]   The analytic gradient disagrees with the measured loss change. Unlike the\n"
                    "[mm3-fd]   route comparison below, this catches a defect BOTH backward routes share\n"
                    "[mm3-fd]   — a wrong scale on the chunked CE, or a missing term.\n");
        }
        jl("{\"type\":\"gate\",\"check\":\"finite-difference-f32\",\"pass\":%s,\"worst\":%.6e,"
           "\"bar\":%.6e,\"probes\":%d}",
           fd_ok ? "true" : "false", worst, fd_bar, (int) probes.size());
    } else {
        fprintf(stderr, "\n[mm3-fd] %d/%zu probes within 15%% (worst %.3f) — INDICATIVE ONLY\n",
                (int) probes.size() - n_bad, probes.size(), worst);
        fprintf(stderr, "[mm3-fd]   (FD cannot certify an f16 base: error tracks 1/||g||, and shrinking eps\n"
                        "[mm3-fd]    makes it WORSE. Add --f32-layers 2 to turn this into a verdict.)\n");
    }
    bool gate_ran = false, gate_pass = false;
    if (g_ckpt.size() == g_naive.size()) {
        // Two exact routes to the same gradient, so this compares whole vectors
        // by relative L2 rather than by a single entry (where 1e-6 values make
        // a relative error meaningless).
        double wc = 0.0;
        for (size_t i = 0; i < g_ckpt.size(); i++) {
            double dn = 0.0, rn2 = 0.0;
            for (size_t k = 0; k < g_naive[i].size(); k++) {
                const double d = (double) g_ckpt[i][k] - (double) g_naive[i][k];
                dn += d * d;
                rn2 += (double) g_naive[i][k] * (double) g_naive[i][k];
            }
            wc = std::max(wc, std::sqrt(dn) / std::max(1e-30, std::sqrt(rn2)));
        }
        // The load-bearing number. Two structurally different routes to the same
        // gradient — whole-graph autodiff vs segmented recompute with a
        // surrogate loss — so a wiring error in the lm-ckpt.h hooks shows up
        // here and nowhere else. Both run the f16 base, so the floor is f16
        // accumulation over 36 layers, not a design difference.
        fprintf(stderr, "[mm3-fd] checkpointed vs naive gradients: worst relative L2 %.2e\n", wc);
        if (!isolated) {
            fprintf(stderr, "[mm3-fd]   (ACE's own bar for this comparison is 2e-3, but measured under F32\n"
                            "[mm3-fd]    isolation; both routes here run the f16 base, so this REPORTS and\n"
                            "[mm3-fd]    does not certify. Add --f32-layers 2 for a verdict.)\n");
        } else {
            // THE GATE. Under isolation the only remaining difference between
            // the two routes is F32 reassociation, so anything above ACE's 2e-3
            // is a wiring defect rather than arithmetic - which is exactly the
            // distinction the f16 run could not make.
            gate_ran  = true;
            gate_pass = wc < 2e-3;
            fprintf(stderr, "[mm3-fd] GATE %s: F32-isolated (%d layers), bar 2e-3, worst %.2e\n",
                    gate_pass ? "PASS" : "FAIL", f32_layers, wc);
            if (!gate_pass) {
                fprintf(stderr,
                        "[mm3-fd]   The checkpointed path disagrees with whole-graph autodiff by more than\n"
                        "[mm3-fd]   F32 reassociation explains. Suspect the two hooks this trainer added to\n"
                        "[mm3-fd]   lm-ckpt.h: the untied scored head (head_w/head_row0/head_v) and the\n"
                        "[mm3-fd]   frame-embedding entry (embed_build/embed_user), or the chunked-CE scale.\n");
            }
            jl("{\"type\":\"gate\",\"check\":\"ckpt-vs-naive-f32\",\"pass\":%s,\"worst\":%.6e,"
               "\"bar\":2.0e-03,\"layers\":%d}",
               gate_pass ? "true" : "false", wc, f32_layers);
        }
    } else if (isolated) {
        // A verdict was asked for and one arm never produced gradients: that is
        // a failure, not a missing measurement.
        gate_ran  = true;
        gate_pass = false;
        fprintf(stderr, "[mm3-fd] GATE FAIL: one of the two gradient routes did not run\n");
    }

    ggml_backend_sched_free(sched);
    lm_ckpt_free(&ckpt_st);
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctx_static);
    lm_optim_free(&opt);
    lm_lora_detach(&lora, &t.lm);
    lm_lora_free(&lora);
    // The isolated tensors are what lm_lora_detach just restored the base
    // pointers around, so this has to come after it and before the model free.
    mm3_f32_isolate_free(&iso);
    mm3_train_lm_free(&t);
    // Non-zero ONLY when a verdict was asked for and lost — either gate.
    // Without --f32-layers this command reports and always succeeds, as before.
    const bool failed = (gate_ran && !gate_pass) || (gate_fd_ran && !gate_fd_pass);
    return failed ? 1 : 0;
}

// ── the run ─────────────────────────────────────────────────────────────────

static int mm3_lm_train_main(const MM3LmTrainArgs & a) {
    // TF32 off for the same reason every other MM3 training-data path turns it
    // off: this is gradient arithmetic against a frozen f16 base, and TF32's
    // ~1e-3 is not a trade worth taking for a few percent.
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif

    std::string err;
    MM3TrainLm  t = {};
    if (!mm3_train_lm_load(&t, a.lm_path.c_str(), &err)) {
        fprintf(stderr, "[mm3-lm-train] LM load failed: %s\n", err.c_str());
        jl("{\"type\":\"fatal\",\"message\":\"%s\"}", json_escape(err).c_str());
        return 1;
    }
    if (!mm3_train_lm_load_audio_embd(&t, a.depth_path.c_str(), &err)) {
        fprintf(stderr, "[mm3-lm-train] audio_embd load failed: %s\n", err.c_str());
        jl("{\"type\":\"fatal\",\"message\":\"%s\"}", json_escape(err).c_str());
        mm3_train_lm_free(&t);
        return 1;
    }
    const Qwen3LMConfig & c   = t.lm.cfg;
    const int64_t         H   = c.hidden_size;
    const int64_t         NC  = (int64_t) t.num_codebooks - 1;
    const int64_t         AV  = t.acoustic_vocab_size;
    const int64_t         SL  = mm3_lm_train_slice_size(t);

    // VRAM once the frozen base is resident, before the LoRA, the optimizer
    // state and the checkpoint buffers. NOT a pre-flight baseline — the query
    // needs the backend, which only exists after the load — so it is labelled
    // for what it is.
    //
    // It is still the early-warning number: the base is ~16 GB and everything
    // after it is ~13 GB more, so a `free` here below about 14 GB means this
    // run will end up over the card and spill into shared memory, where a 4 s
    // step becomes ~40. Reported before step 1 so the answer arrives before
    // the wait does.
    {
        size_t bfree = 0, btotal = 0;
        lm_vram_query(t.lm.backend, &bfree, &btotal);
        if (btotal > 0) {
            const long long used0 = (long long) ((btotal - bfree) / (1024 * 1024));
            jl("{\"type\":\"vram\",\"step\":0,\"usedMb\":%lld,\"freeMb\":%lld,\"totalMb\":%lld,"
               "\"phase\":\"after-model-load\"}",
               used0, (long long) (bfree / (1024 * 1024)), (long long) (btotal / (1024 * 1024)));
            fprintf(stderr, "[mm3-lm-train] VRAM after model load: %lld/%lld MB used\n",
                    used0, (long long) (btotal / (1024 * 1024)));
        }
    }

    // ── samples ──
    std::vector<MM3LmSample> samples;
    if (!mm3_lm_load_samples(a, t, &samples, &err)) {
        fprintf(stderr, "[mm3-lm-train] %s\n", err.c_str());
        mm3_train_lm_free(&t);
        return 1;
    }
    if (samples.empty()) {
        fprintf(stderr, "[mm3-lm-train] no usable samples\n");
        mm3_train_lm_free(&t);
        return 1;
    }

    // ── train / held-out split ─────────────────────────────────────────────
    //
    // Deterministic: the LAST ceil(holdout * n) songs by manifest order. Not
    // random, because a holdout that moves between runs makes two runs'
    // evaluation numbers incomparable, which is the entire point of having one.
    // Refused below 6 songs — withholding 1 of 5 costs 20 % of an already tiny
    // corpus, and an identity adapter needs every track more than it needs a
    // measurement.
    size_t n_hold = 0;
    if (a.holdout > 0.0f && samples.size() >= 6) {
        n_hold = (size_t) std::ceil((double) a.holdout * (double) samples.size());
        n_hold = std::min(n_hold, samples.size() / 4);   // never more than a quarter
        n_hold = std::max<size_t>(n_hold, 1);
    }
    std::vector<MM3LmSample> holdout;
    if (n_hold > 0) {
        holdout.assign(samples.end() - (long) n_hold, samples.end());
        samples.resize(samples.size() - n_hold);
    }
    if (n_hold == 0 && a.eval_every > 0) {
        fprintf(stderr, "[mm3-lm-train] no held-out songs (%zu total) — evaluation disabled, the training "
                        "loss is the only signal and it cannot distinguish learning from memorising\n",
                samples.size());
    }

    int64_t max_prompt = 0;
    for (const auto & s : samples) max_prompt = std::max(max_prompt, (int64_t) s.prompt.size());
    for (const auto & s : holdout) max_prompt = std::max(max_prompt, (int64_t) s.prompt.size());
    const int64_t K_max = a.max_frames > 0 ? a.max_frames : 4096;
    // A crop that reaches the track end uses all K frames as INPUT, and with a
    // prefix the window takes one more in front of them (see `lead`). So the
    // widest input span is K_max + 1, and every buffer sized off the sequence
    // has to know that or the last crop of a flush-to-the-end step writes one
    // frame past the end of t_sem.
    const int64_t F_max = K_max + (a.prefix_frames > 0 ? 1 : 0);
    const int64_t S_max = max_prompt + F_max;
    fprintf(stderr, "[mm3-lm-train] %zu training songs (+%zu held out), longest prompt %lld tok, "
                    "crop <= %lld frames, seq <= %lld\n",
            samples.size(), holdout.size(), (long long) max_prompt, (long long) K_max, (long long) S_max);
    // ── JSONL (--jsonl), the contract the server runner relays ──
    // Same vocabulary as `train-lm` so mm3TrainLmRunner is a relay clone and the
    // Monitor's loss chart works with no new event types:
    // init / step / milestone / progress / export / fatal / done.
    jl("{\"type\":\"init\",\"samples\":%zu,\"holdout\":%zu,\"stepsPerEpoch\":%d,\"totalSteps\":%d,"
       "\"maxPrompt\":%lld,\"maxFrames\":%lld,\"seqMax\":%lld,\"rank\":%d,\"alpha\":%d,"
       "\"optimizer\":\"%s\",\"lrScale\":%.4f}",
       samples.size(), holdout.size(), (int) samples.size(), a.steps, (long long) max_prompt,
       (long long) K_max, (long long) S_max, a.rank, a.alpha, a.optimizer.c_str(),
       a.optimizer == "muon" ? (double) a.muon_lr_scale : 1.0);

    // ── LoRA (attaches to the model) + optimizer ──
    LmLora     lora;
    const bool want_lokr = a.is_lokr();
    const bool init_ok =
        want_lokr
            ? lm_lokr_init(&lora, &t.lm, 0, c.n_layers, a.lokr_dim, a.lokr_alpha, a.lokr_factor,
                           a.lokr_decompose_both, (uint64_t) a.seed, &err)
            : lm_lora_init(&lora, &t.lm, 0, c.n_layers, a.rank, (float) a.alpha, (uint64_t) a.seed, 0.0f,
                           &err);
    if (!init_ok) {
        fprintf(stderr, "[mm3-lm-train] %s init failed: %s\n", want_lokr ? "LoKr" : "LoRA", err.c_str());
        mm3_train_lm_free(&t);
        return 1;
    }
    if (want_lokr) {
        fprintf(stderr, "[mm3-lm-train] LoKr: dim %d alpha %.0f factor %d, decompose %s\n", a.lokr_dim,
                (double) a.lokr_alpha, a.lokr_factor, a.lokr_decompose_both ? "both" : "w1-only");
        jl("{\"type\":\"adapter\",\"kind\":\"lokr\",\"dim\":%d,\"alpha\":%.0f,\"factor\":%d}", a.lokr_dim,
           (double) a.lokr_alpha, a.lokr_factor);
    }
    LmOptim opt;
    opt.optimizer     = a.optimizer;
    opt.muon.lr_scale = a.muon_lr_scale;
    opt.muon.momentum = a.muon_momentum;
    opt.muon.ns_steps = a.muon_ns_steps;
    opt.muon.nesterov = a.muon_nesterov;
    opt.muon.min_dim  = a.muon_min_dim;
    opt.muon.bucket   = a.muon_bucket;
    if (!lm_optim_init(&opt, lora.params, t.lm.backend, &err)) {
        fprintf(stderr, "[mm3-lm-train] optimizer init failed: %s\n", err.c_str());
        lm_lora_detach(&lora, &t.lm);
        lm_lora_free(&lora);
        mm3_train_lm_free(&t);
        return 1;
    }
    // A run where Muon classified ZERO parameters trains as AdamW and says
    // nothing about it. Print the split so that is visible.
    fprintf(stderr, "[mm3-lm-train] %zu LoRA tensors (rank %d, alpha %d) — %d on Muon in %zu buckets, %zu on AdamW\n",
            lora.params.size(), a.rank, a.alpha, opt.n_muon, opt.muon_buckets.size(),
            lora.params.size() - (size_t) opt.n_muon);
    // The split is an EVENT, not just a log line: a run that put zero
    // parameters on Muon trained as AdamW, and the UI should be able to say so.
    jl("{\"type\":\"optimizer\",\"name\":\"%s\",\"tensors\":%zu,\"muon\":%d,\"buckets\":%zu,\"lrScale\":%.4f}",
       a.optimizer.c_str(), lora.params.size(), opt.n_muon, opt.muon_buckets.size(),
       // 1.0 under AdamW: reporting the Muon scale on an AdamW run made the
       // event say lrScale 64 next to "504 on AdamW", and the UI multiplies the
       // displayed learning rate by it.
       a.optimizer == "muon" ? (double) a.muon_lr_scale : 1.0);

    // ── persistent tensors ──
    ggml_context * ctx_static = nullptr;
    {
        ggml_init_params p = { 32 * ggml_tensor_overhead(), nullptr, true };
        ctx_static         = ggml_init(p);
    }
    ggml_tensor * t_prompt = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, max_prompt);
    ggml_tensor * t_sem    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, F_max);
    ggml_tensor * t_ac     = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, F_max * NC);
    // With a frozen prefix the mask is rectangular, [n_pfx + S, S] — see
    // lm_causal_mask_prefix. Size for the worst case up front; the prefill's
    // own chunk masks live in its store, not here.
    const int64_t PFX_Q   = a.prefix_frames > 0 ? max_prompt + a.prefix_frames : 0;
    const int64_t MSK_CAP = (PFX_Q + S_max) * S_max;
    ggml_tensor * t_pos    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, S_max);
    ggml_tensor * t_msk    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, MSK_CAP);
    // Prefill inputs, one chunk wide. SEPARATE from the window's, and not an
    // optimisation to undo: the prefill embeds its chunks while the window's
    // own ids are already uploaded, so sharing a buffer means the prefill
    // overwrites the caption the window is about to embed. That cost 0.215
    // nats on the equivalence self-test.
    ggml_tensor * t_pfx_prompt = a.prefix_frames > 0
                                ? ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, a.prefix_chunk)
                                : nullptr;
    ggml_tensor * t_pfx_sem = a.prefix_frames > 0
                                ? ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, a.prefix_chunk)
                                : nullptr;
    ggml_tensor * t_pfx_ac  = a.prefix_frames > 0
                                ? ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, a.prefix_chunk * NC)
                                : nullptr;
    // The [SL, K_max+1] one-hot buffer is a NAIVE-path structure (98 MB at
    // K_max 1500). The checkpointed head chunks its own labels into a
    // [SL, chunk] buffer inside LmCkptState, so allocating this too would be
    // pure waste on the path that actually runs.
    ggml_tensor * t_lab    = a.ckpt ? nullptr
                                    : ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, SL, K_max + 1);
    ggml_tensor * t_adamw  = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 7);
    ggml_tensor * t_lg     = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_clip   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_gn2    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    // Checkpointed path only: the per-chunk upstream scalar and the segment
    // surrogate's loss gradient (which is exactly 1.0 — see lm-ckpt.h D9).
    ggml_tensor * t_gs     = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_one    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    // Rank-dropout mask, [r]. A PERSISTENT buffer, not a per-graph tensor,
    // precisely so the checkpoint's recompute pass reads the same bytes the
    // collect pass did (lm-ckpt.h D13). A freshly drawn mask per pass would
    // compute gradients for a different network than the loss.
    ggml_tensor * t_rankmask = a.rank_dropout > 0.0
                                 ? ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, a.rank, 1)
                                 : nullptr;
    // Sized but never read under checkpointing (the override supplies the
    // embedding); lm_ckpt_micro_step skips its upload.
    ggml_tensor * t_tok    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, S_max);
    for (ggml_tensor * x : { t_prompt, t_sem, t_ac, t_pos, t_msk, t_tok }) ggml_set_input(x);
    if (t_lab) ggml_set_input(t_lab);
    if (t_pfx_sem) { ggml_set_input(t_pfx_prompt); ggml_set_input(t_pfx_sem); ggml_set_input(t_pfx_ac); }

    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, t.lm.backend);
    if (!buf_static) {
        fprintf(stderr, "[mm3-lm-train] static buffer allocation failed (lower --max-frames)\n");
        lm_lora_detach(&lora, &t.lm);
        lm_lora_free(&lora);
        mm3_train_lm_free(&t);
        return 1;
    }
    ggml_backend_buffer_clear(buf_static, 0);
    {
        const float lg = 1.0f / (float) std::max(1, a.grad_accum);
        const float cl = (float) a.grad_clip;
        const float ep = 1e-6f;
        ggml_backend_tensor_set(t_lg, &lg, 0, sizeof(float));
        ggml_backend_tensor_set(t_clip, &cl, 0, sizeof(float));
        ggml_backend_tensor_set(t_eps, &ep, 0, sizeof(float));
        const float one = 1.0f;
        ggml_backend_tensor_set(t_one, &one, 0, sizeof(float));
    }
    opt.t_adamw      = t_adamw;
    opt.t_lossgrad   = t_lg;
    opt.t_clip       = t_clip;
    opt.t_eps        = t_eps;
    opt.t_gnorm2     = t_gn2;
    opt.base_lr      = (float) a.lr;
    opt.lr_floor     = (float) a.lr_end_frac;
    if (a.optimizer == "prodigy") {
        // MUST come after the two lines above, which would otherwise overwrite it.
        // Under Prodigy lr is GAMMA -- a schedule multiplier, not a step size --
        // and d carries the magnitude. Leaving a hand-tuned 8e-5 here scales every
        // step by 1e-4 and looks exactly like "Prodigy does not converge".
        if (a.lr != 1.0) {
            fprintf(stderr,
                    "[mm3-lm-train] prodigy: --lr %.3g is IGNORED. Prodigy sets its own step size; "
                    "lr is only a schedule multiplier and is forced to 1.0.\n",
                    a.lr);
        }
        opt.base_lr    = 1.0f;
        opt.prodigy_d0 = (float) a.prodigy_d0;
    }
    opt.weight_decay = (float) a.weight_decay;
    opt.grad_clip    = (float) a.grad_clip;
    opt.total_steps  = a.steps;
    opt.warmup_steps = a.warmup;

    // ── per-layer gradient checkpointing ──
    //
    // Not an optimisation here. A naive fwd+bwd retains every layer's
    // activations at once; with one segment per layer exactly ONE is live, and
    // the chunked CE head keeps the [16389, chunk] logits off the peak too.
    // The head override is what makes the second half work for MM3: an UNTIED
    // head, scored only over [eos_audio, semantic_offset + semantic_size).
    // ── Lever A (--weights bf16) ────────────────────────────────────────────
    //
    // Resolved BEFORE the graph is built, and it falls back rather than failing.
    // Neither gate has a graceful path further in: on a non-CUDA backend the
    // BF16 out_prod patch is absent and ggml would GGML_ABORT mid-backward, and
    // on a non-BF16 base the surgery has no weights to rewrite. `weights_used`
    // (not a.weights) is what gets reported, so a fallback is never mislabelled
    // as the bf16 run it was not.
    bool        weights_bf16 = (a.weights == "bf16");
    std::string weights_used = a.weights;
    if (weights_bf16) {
        std::string why;
        if (strncmp(ggml_backend_name(t.lm.backend), "CUDA", 4) != 0) {
            why = std::string("BF16 weights require CUDA (only ggml-cuda carries the BF16 out_prod patch) "
                              "— falling back to f32-window; this run is on ")
                + ggml_backend_name(t.lm.backend);
        } else if (!lm_bf16_base_is_bf16(t.lm)) {
            why = std::string("BF16 weights require a BF16-native base — falling back to f32-window; ")
                + a.lm_path + " loads its projections as " + lm_bf16_base_proj_type_name(t.lm)
                + ". Build one with: python engine/tools/convert-mm3.py --components lm --quant bf16";
        }
        if (!why.empty()) {
            fprintf(stderr, "[mm3-lm-train] WARNING: %s\n", why.c_str());
            weights_bf16 = false;
            weights_used = "f32-window";
        }
    }
    fprintf(stderr, "[mm3-lm-train] base matmuls: %s\n",
            weights_bf16 ? "bf16 (tensor cores, Lever A)"
                         : "f32-window (weights dequantized to F32 in-graph)");
    jl("{\"type\":\"weights\",\"mode\":\"%s\"}", weights_used.c_str());

    LmCkptState  ckpt_st;
    LmCkptRun    ckpt_run;
    MM3EmbedCtx  embed_ctx;
    LmKvPrefix   kvpfx;
    MM3PrefixCtx pfx_ctx;
    const bool   kv_on = a.prefix_frames > 0;

    // ── the acoustic loss (see MM3LmTrainArgs::depth_loss_weight) ───────────
    //
    // Loads the FULL depth decoder (~1.2 GB f16) beside the audio_embd slice
    // the trainer always loaded. A load failure is FATAL rather than a warning:
    // silently training without the term would reintroduce the timbre fault
    // this exists to fix, wearing a fixed-trainer label.
    MM3DepthTrain depth;
    bool          depth_fd_done = false;
    if (a.depth_loss_weight > 0.0 && a.ckpt) {
        if (!mm3_depth_train_load(&depth, a.depth_path.c_str(), t.lm.backend, &err)) {
            fprintf(stderr, "[mm3-lm-train] acoustic loss: %s\n", err.c_str());
            mm3_train_lm_free(&t);
            return 1;
        }
        depth.lm_embed    = t.lm.embed_tokens;
        depth.audio_embd  = t.audio_embd;
        depth.sem_offset  = (int64_t) t.semantic_vocab_offset;
        depth.audio_vocab = (int64_t) t.acoustic_vocab_size;
        depth.K           = a.depth_loss_frames;
        depth.lambda      = (float) a.depth_loss_weight;
        depth.seed        = (uint64_t) a.seed;
        fprintf(stderr,
                "[mm3-lm-train] acoustic loss: ON — weight %.3g, %d frames/step through the frozen "
                "depth decoder\n",
                a.depth_loss_weight, a.depth_loss_frames);
    } else if (a.ckpt) {
        fprintf(stderr,
                "[mm3-lm-train] acoustic loss: OFF — adapters trained this way shift vocal timbre "
                "at render (the depth decoder reads the LM hidden state); use only for A/B\n");
    }
    jl("{\"type\":\"depthLossCfg\",\"weight\":%.6g,\"frames\":%d}",
       a.ckpt ? a.depth_loss_weight : 0.0, a.depth_loss_frames);
    if (a.ckpt) {
        // Quantized bases are allowed ONLY on the f32-window path, where
        // lm_linear casts every weight to F32 in-graph so the backward never
        // sees the quantized tensor. Under Lever A the raw weight goes straight
        // to mul_mat, so a quantized one would hit the transpose path that
        // block-quantized types cannot take — hence the gate above, which has
        // already forced weights_bf16 off unless every projection is BF16.
        if (!lm_ckpt_check_base(&t.lm, &err, /*allow_quantized=*/!weights_bf16)) {
            fprintf(stderr, "[mm3-lm-train] %s\n", err.c_str());
            lm_lora_detach(&lora, &t.lm);
            lm_lora_free(&lora);
            mm3_train_lm_free(&t);
            return 1;
        }
        if (kv_on && !lm_kvprefix_alloc(&kvpfx, &t.lm, 0, c.n_layers, PFX_Q, S_max, a.prefix_chunk, &err)) {
            fprintf(stderr, "[mm3-lm-train] kv prefix setup failed: %s\n", err.c_str());
            lm_lora_detach(&lora, &t.lm);
            lm_lora_free(&lora);
            mm3_train_lm_free(&t);
            return 1;
        }
        LmCkptCfg cc;
        cc.kv           = kv_on ? &kvpfx : nullptr;
        cc.chunk        = a.ckpt_chunk;
        cc.weights_bf16 = weights_bf16;                 // Lever A
        cc.rank_mask = t_rankmask;
        cc.s_max     = (int) S_max;
        cc.layer_lo  = 0;
        cc.layer_hi  = c.n_layers;
        cc.head_w    = t.lm_head;                       // UNTIED
        cc.head_row0 = t.head_slice_row0;              // normally EOS; 0 once F32-isolated
        cc.head_v    = (int) SL;
        if (!lm_ckpt_alloc(&ckpt_st, &t.lm, cc, &err) || !lm_ckpt_build_embed_t(&ckpt_st, &err)) {
            fprintf(stderr, "[mm3-lm-train] checkpoint setup failed: %s\n", err.c_str());
            lm_ckpt_free(&ckpt_st);
            lm_lora_detach(&lora, &t.lm);
            lm_lora_free(&lora);
            mm3_train_lm_free(&t);
            return 1;
        }
        embed_ctx.t        = &t;
        embed_ctx.t_prompt = t_prompt;
        embed_ctx.t_sem    = t_sem;
        embed_ctx.t_ac     = t_ac;

        ckpt_run.lm          = &t.lm;
        ckpt_run.opt         = &opt;
        ckpt_run.st          = &ckpt_st;
        ckpt_run.t_tok       = t_tok;
        ckpt_run.t_pos       = t_pos;
        ckpt_run.t_msk       = t_msk;
        ckpt_run.t_gs        = t_gs;
        ckpt_run.t_one       = t_one;
        ckpt_run.grad_accum  = std::max(1, a.grad_accum);
        ckpt_run.embed_build = mm3_lm_ckpt_embed;
        ckpt_run.embed_user  = &embed_ctx;
        // MM3 uploads song-anchored positions itself; without this the
        // micro-step overwrites them with 0..S-1 and --crop-anchor is a no-op.
        ckpt_run.pos_external = true;

        pfx_ctx.e          = embed_ctx;
        pfx_ctx.e.t_prompt = t_pfx_prompt;   // never the window's — see above
        pfx_ctx.t_sem    = t_pfx_sem;
        pfx_ctx.t_ac     = t_pfx_ac;
        pfx_ctx.NC       = NC;
        pfx_ctx.AV       = AV;
        pfx_ctx.sem_off  = (int64_t) t.semantic_vocab_offset;
    }

    // ── graph sizing + scheduler ──
    // The scheduler is SHARED with the optimizer step, so it must be sized for
    // whichever graph is larger. This bit a real 4B Muon run: Muon's optimizer
    // graph is ~7-9k nodes while a segmented training graph was ~569, and ggml
    // asserts hash_set.size >= n_nodes + n_leafs mid-run. Do not "simplify".
    std::vector<uint8_t> arena((size_t) 512 << 20);
    int                  graph_nodes = 0;

    auto build_graph = [&](ggml_context * ctx, ggml_cgraph * gf, int64_t P, int64_t Fin, int64_t n_sup,
                           ggml_tensor ** out_loss) {
        const int64_t S = P + Fin;
        MM3EmbedCtx   ec{ &t, t_prompt, t_sem, t_ac, P, Fin };
        ggml_tensor * h_in = mm3_lm_build_embed(ctx, ec);

        ggml_tensor * hidden = lm_build_trunk_embeds(ctx, &t.lm, h_in, t_pos, t_msk, (int) S);
        // Supervised positions are a contiguous tail starting at P-1.
        ggml_tensor * hd = ggml_cont(
            ctx, ggml_view_2d(ctx, hidden, H, n_sup, hidden->nb[1], (size_t) (P - 1) * hidden->nb[1]));
        ggml_tensor * logits = ggml_mul_mat(ctx, mm3_lm_train_out_slice(ctx, t), hd);   // [SL, n_sup]
        ggml_tensor * labv   = ggml_view_2d(ctx, t_lab, SL, n_sup, t_lab->nb[1], 0);
        ggml_tensor * loss   = ggml_cross_entropy_loss(ctx, logits, labv);
        ggml_set_loss(loss);
        ggml_set_output(loss);
        ggml_build_forward_expand(gf, loss);
        *out_loss = loss;
    };

    if (a.ckpt) {
        // The worst checkpointed graph is ONE backward segment at S_max — the
        // trunk is never built whole, so sizing from it would over-allocate the
        // scheduler by ~L x. embed_ctx must describe a real crop first: the
        // probe builds P1, which calls the override.
        embed_ctx.P   = max_prompt;
        embed_ctx.Fin = F_max;
        ckpt_run.sched = nullptr;
        graph_nodes    = lm_ckpt_probe_segment_nodes(ckpt_run, (int) S_max);
    } else {
        ggml_init_params gip = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(gip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
        ggml_tensor *    loss = nullptr;
        build_graph(ctx, gf, max_prompt, F_max, K_max + 1, &loss);
        std::vector<ggml_tensor *> gacc;
        lm_optim_fill_gacc(&opt, gf, &gacc);
        ggml_build_backward_expand(ctx, gf, gacc.data());
        graph_nodes = ggml_graph_n_nodes(gf);
        ggml_free(ctx);
    }
    fprintf(stderr, "[mm3-lm-train] %s graph: %d nodes\n",
            a.ckpt ? "worst backward segment" : "fwd+bwd", graph_nodes);

    BackendPair bp;
    bp.backend     = t.lm.backend;
    bp.cpu_backend = t.lm.cpu_backend;
    bp.has_gpu     = t.lm.backend != t.lm.cpu_backend;
    const int sched_nodes = std::max(std::max(8192, graph_nodes + graph_nodes / 2 + 2048),
                                     opt.est_nodes + opt.est_nodes / 4 + 1024);
    ggml_backend_sched_t sched = backend_sched_new(bp, sched_nodes);
    if (a.ckpt) {
        ckpt_run.sched = sched;
    }

    // ── training loop ──
    LmRng rng;
    lm_rng_seed(&rng, (uint64_t) a.seed ^ 0x9E3779B97F4A7C15ull);

    // Rank dropout. `active=false` writes all ones, which is what eval and the
    // prior capture must see: dropout is a training-time perturbation, and a
    // measurement taken through a randomly crippled adapter measures nothing.
    std::vector<float> rankmask_host;
    auto set_rank_mask = [&](bool active) {
        if (!t_rankmask) {
            return;
        }
        rankmask_host.assign((size_t) a.rank, 1.0f);
        if (active) {
            const float keep = 1.0f - (float) a.rank_dropout;
            const float up   = keep > 0.0f ? 1.0f / keep : 1.0f;
            for (int i = 0; i < a.rank; i++) {
                rankmask_host[(size_t) i] =
                    lm_rng_uniform(&rng) < (float) a.rank_dropout ? 0.0f : up;
            }
        }
        ggml_backend_tensor_set(t_rankmask, rankmask_host.data(), 0,
                                rankmask_host.size() * sizeof(float));
    };

    // --crop-anchor. Read once: it is consulted per micro-step and per eval crop.
    const bool anchor_song = (a.crop_anchor != "zero");
    fprintf(stderr, "[mm3-lm-train] crop anchor: %s\n",
            anchor_song ? "song (crops carry their true position)"
                        : "zero (every crop presented as the opening — the legacy convention)");
    jl("{\"type\":\"cropAnchor\",\"mode\":\"%s\"}", anchor_song ? "song" : "zero");
    if (kv_on) {
        // Both are hard requirements, not preferences. The non-checkpointed
        // path builds one whole-trunk graph and has no place to splice a store
        // into; and a prefix under `zero` anchoring would sit at positions the
        // window then re-uses, which is not a history, it is a contradiction.
        if (!a.ckpt) {
            fprintf(stderr, "[mm3-lm-train] --prefix-frames needs the checkpointed path (drop --no-ckpt)\n");
            lm_kvprefix_free(&kvpfx);
            return 1;
        }
        if (!anchor_song) {
            fprintf(stderr, "[mm3-lm-train] --prefix-frames needs --crop-anchor song\n");
            lm_kvprefix_free(&kvpfx);
            return 1;
        }
        fprintf(stderr,
                "[mm3-lm-train] kv prefix: up to %lld frames (%.1f s) of no-grad history in front of each crop\n",
                (long long) a.prefix_frames, (double) a.prefix_frames / 25.0);
        jl("{\"type\":\"kvPrefix\",\"frames\":%lld,\"chunk\":%d}", (long long) a.prefix_frames,
           a.prefix_chunk);
    }
    if (a.crop_mode == "structured") {
        fprintf(stderr,
                "[mm3-lm-train] crop policy: structured - %.0f%% start share (half at frame 0, "
                "half over %d aligned tiles), %.0f%% flush to the end (EOS), %.0f%% random\n",
                a.crop_start_frac * 100.0, a.crop_start_tiles, a.crop_end_frac * 100.0,
                (1.0 - a.crop_start_frac - a.crop_end_frac) * 100.0);
    }
    jl("{\"type\":\"cropPolicy\",\"mode\":\"%s\",\"startFrac\":%.3f,\"endFrac\":%.3f}",
       a.crop_mode.c_str(), a.crop_start_frac, a.crop_end_frac);
    std::vector<int32_t> sem_in, ac_in, tgt, pos, pfx_pos;
    std::vector<float>   msk;
    int                  last_mask_S = 0;
    double               running = 0.0;
    int                  n_micro = 0, rc = 0;
    LmStepStats          stats;
    const int64_t        t_start = ggml_time_ms();
    int64_t              t_step0 = t_start;
    double               best_eval = -1.0;
    int                  best_eval_step = 0;
    // Completed epoch means, oldest first — the window the target-loss stop
    // averages. Declared up here with the other run-spanning statistics because
    // the resume block and the state snapshot below both touch it.
    std::vector<double>  ep_means;
    // Prior-preservation steps are accounted separately throughout: their loss
    // is soft-target CE against the frozen base, not CE against a code, and
    // averaging the two together produces a number that describes neither.
    double               reg_running = 0.0;
    int                  reg_n_micro = 0;
    // PER SEGMENT, not per run: n_micro is restored from the resume state and
    // these are not, so dividing one by the other after a pause under-reports
    // the rate. Reporting the segment is honest and needs no change to the
    // resume format — which matters, since bumping it would strand every state
    // file already on disk.
    int                  n_dropped   = 0;   // caption-dropout steps, THIS SEGMENT
    int                  n_style_seg = 0;   // style steps, THIS SEGMENT

    auto save_ckpt = [&](int step, double loss) -> std::string {
        char sub[64];
        snprintf(sub, sizeof(sub), "ckpt-%d", step);
        const std::string dir = a.out_dir + "/" + sub;
        LmExportMeta      meta;
        meta.producer = "ace-train mm3-lm-train";
        meta.lm_path  = a.lm_path;
        meta.rank     = a.rank;
        meta.alpha    = a.alpha;
        meta.lr       = a.lr;
        meta.seed     = a.seed;
        meta.samples  = (int) samples.size();
        meta.trigger  = a.trigger;
        meta.saved_loss = loss;
        LmExportResult res;
        std::string    xerr;
        // LoKr writes lokr_weights.safetensors; LoRA writes a PEFT directory
        // (adapter_config.json + adapter_model.safetensors). Same ckpt-<step>/
        // directory, different file name — the sidecar below has to follow, and
        // so does every consumer that hard-codes adapter_model.safetensors.
        const bool lokr_out = a.is_lokr();
        const bool exported = lokr_out ? lm_export_lokr(lora, meta, dir, &res, &xerr)
                                       : lm_export_peft(lora, c, meta, dir, &res, &xerr);
        if (!exported) {
            fprintf(stderr, "[mm3-lm-train] export failed: %s\n", xerr.c_str());
            return std::string();
        }
        // The sidecar the shipped MM3 adapter picker reads. Written beside the
        // safetensors so `<out>` pointed at <adapters>/mm3-lm-adapters/<run>
        // makes the checkpoint appear in the UI with no install step.
        const std::string side =
            dir + (lokr_out ? "/lokr_weights.safetensors.json" : "/adapter_model.safetensors.json");
        FILE *            sf   = hs_fopen(side, "wb");
        // The recommended MLP dial depends on the OBJECTIVE this run used: with
        // the acoustic loss the hidden state stays where the depth decoder
        // expects it and full strength is correct; without it the old 0.5
        // crutch remains the honest recommendation, because the timbre fault
        // is baked into the weights.
        if (sf) {
            fprintf(sf,
                    "{\"name\":\"%s ckpt-%d\",\"trigger\":\"%s\",\"rank\":%d,\"dataset\":\"%s\","
                    "\"trainedSteps\":%d,\"recommendedScales\":{\"scaleMlp\":%.1f},"
                    "\"notes\":\"ace-train mm3-lm-train, loss %.4f; render captions must carry the "
                    "artist's true bpm/tuning\"}\n",
                    a.dataset_name.empty() ? "MM3 LM" : a.dataset_name.c_str(), step, a.trigger.c_str(), a.rank,
                    a.dataset_name.c_str(), step, a.depth_loss_weight > 0.0 ? 1.0 : 0.5, loss);
            fclose(sf);
        }
        fprintf(stderr, "[mm3-lm-train] saved %s (loss %.4f)\n", dir.c_str(), loss);
        jl("{\"type\":\"milestone\",\"step\":%d,\"loss\":%.6f,\"path\":\"%s\"}", step, loss,
           json_escape(dir).c_str());
        jl("{\"type\":\"export\",\"tensors\":%zu,\"path\":\"%s\"}", lora.params.size(),
           json_escape(dir).c_str());
        return dir;
    };

    // ── held-out evaluation ────────────────────────────────────────────────
    //
    // A FIXED set of crops, chosen once, evaluated identically every time. The
    // point is comparability: across steps within a run, and across runs. Crops
    // are evenly spaced through each held-out song (never random), so the same
    // eval number always measures the same audio.
    struct EvalCrop { const MM3LmSample * s; int64_t c0; int64_t K; };
    std::vector<EvalCrop> eval_plan;
    if (!holdout.empty() && a.eval_every > 0) {
        for (const auto & hs : holdout) {
            const int64_t K = std::min<int64_t>(a.eval_crop, hs.n_frames);
            if (K < 8) continue;
            const int n = std::max(1, a.eval_crops);
            for (int i = 0; i < n; i++) {
                // Evenly spaced starts, with the last crop flush to the end so
                // the set always includes a real ending (where EOS lives).
                const int64_t span = hs.n_frames - K;
                const int64_t c0   = n == 1 ? span / 2 : (span * i) / (n - 1);
                eval_plan.push_back(EvalCrop{ &hs, std::max<int64_t>(0, c0), K });
            }
        }
        fprintf(stderr, "[mm3-lm-train] evaluation: %zu fixed crops from %zu held-out song(s), every %d steps\n",
                eval_plan.size(), holdout.size(), a.eval_every);
    }

    // One held-out pass. Runs through the SAME checkpointed machinery with
    // forward_only set, so it allocates nothing new — which matters when the
    // run already sits at ~30 GB of a 32 GB card.
    auto run_eval = [&]() -> double {
        if (eval_plan.empty() || !a.ckpt) {
            return -1.0;
        }
        ckpt_run.forward_only = true;
        set_rank_mask(false);      // measure the whole adapter, not a subnetwork
        double sum = 0.0;
        int    n   = 0;
        for (const EvalCrop & ec : eval_plan) {
            const MM3LmSample & es = *ec.s;
            const int64_t P = (int64_t) es.prompt.size();
            const bool    at_end = (ec.c0 + ec.K) >= es.n_frames;
            const int64_t Fin    = at_end ? ec.K : ec.K - 1;
            const int64_t n_sup  = at_end ? ec.K + 1 : ec.K;
            const int64_t S      = P + Fin;
            if (S > S_max || Fin < 1) continue;

            sem_in.resize((size_t) Fin);
            ac_in.resize((size_t) (Fin * NC));
            for (int64_t i = 0; i < Fin; i++) {
                const int32_t * f = &es.codes[(size_t) ((ec.c0 + i) * 8)];
                sem_in[(size_t) i] = f[0] + (int32_t) t.semantic_vocab_offset;
                for (int64_t k = 0; k < NC; k++) {
                    ac_in[(size_t) (k * Fin + i)] = f[1 + k] + (int32_t) (k * AV);
                }
            }
            tgt.resize((size_t) n_sup);
            for (int64_t j = 0; j < n_sup; j++) {
                tgt[(size_t) j] = (at_end && j == n_sup - 1)
                                    ? mm3_lm_train_slice_eos(t)
                                    : mm3_lm_train_slice_index(t, es.codes[(size_t) ((ec.c0 + j) * 8)]);
            }
            if ((int) S != last_mask_S) {
                lm_causal_mask((int) S, &msk);
                ggml_backend_tensor_set(t_msk, msk.data(), 0, msk.size() * sizeof(float));
                last_mask_S = (int) S;
            }
            // Same anchoring as training — an eval measured under a different
            // position convention is not measuring the thing being trained.
            pos.resize((size_t) S);
            for (int64_t i = 0; i < P; i++) pos[(size_t) i] = (int32_t) i;
            for (int64_t j = 0; j < Fin; j++) {
                pos[(size_t) (P + j)] = (int32_t) (P + (anchor_song ? ec.c0 : 0) + j);
            }
            ggml_backend_tensor_set(t_prompt, es.prompt.data(), 0, (size_t) P * sizeof(int32_t));
            ggml_backend_tensor_set(t_sem, sem_in.data(), 0, sem_in.size() * sizeof(int32_t));
            ggml_backend_tensor_set(t_ac, ac_in.data(), 0, ac_in.size() * sizeof(int32_t));
            ggml_backend_tensor_set(t_pos, pos.data(), 0, pos.size() * sizeof(int32_t));

            embed_ctx.P   = P;
            embed_ctx.Fin = Fin;
            LmSample smp;
            smp.tokens.assign((size_t) S, 0);
            smp.targets  = tgt;
            smp.n_masked = (int) P;
            smp.s_tr     = (int) n_sup;
            double ce = 0.0;
            if (lm_ckpt_micro_step(ckpt_run, smp, true, &ce)) {
                sum += ce;
                n++;
            }
        }
        ckpt_run.forward_only = false;
        return n > 0 ? sum / (double) n : -1.0;
    };


    // ── prefix equivalence self-test (see MM3LmTrainArgs::prefix_selftest) ──
    //
    //   A   one square crop  [prompt ; frames c0-N .. c0+K-2]
    //   B   prefix           [prompt ; frames c0-N .. c0-2  ]
    //       window           [prompt ; frames c0-1 .. c0+K-2]
    //
    // Both supervise the SAME K positions with the same targets, and in both
    // the row that predicts frame c0 is frame c0-1's — which is why B's window
    // starts one frame early (see `lead` in the training loop). If the store,
    // the rectangular mask and the position anchoring are right, the two
    // cross-entropies are the same number.
    if (kv_on && a.prefix_selftest) {
        const MM3LmSample * ss = nullptr;
        int64_t             P = 0, N = std::min<int64_t>(a.prefix_frames, 96), K = 96;
        for (const MM3LmSample & cand : samples) {
            const int64_t p = (int64_t) cand.prompt.size();
            if (cand.n_frames > N + K && p + N + K <= S_max) {
                ss = &cand;
                P  = p;
                break;
            }
        }
        if (!ss) {
            fprintf(stderr, "[mm3-lm-train] --prefix-selftest: no sample is long enough (need > %lld frames)\n",
                    (long long) (N + K));
            rc = 1;
        } else {
            const MM3LmSample & sx = *ss;
            const int64_t       c0 = N;

            // Targets are the same for both runs: frames c0 .. c0+K-1.
            tgt.resize((size_t) K);
            for (int64_t j = 0; j < K; j++) {
                tgt[(size_t) j] = mm3_lm_train_slice_index(t, sx.codes[(size_t) ((c0 + j) * 8)]);
            }
            ggml_backend_tensor_set(t_prompt, sx.prompt.data(), 0, (size_t) P * sizeof(int32_t));

            // f0 = first INPUT frame, Fin = how many, n_masked = unsupervised
            // lead, n_prompt = rows that stay blind to the prefix.
            auto run = [&](int64_t f0, int64_t Fin, int64_t n_masked, int64_t n_prompt) -> double {
                const int64_t S = P + Fin;
                sem_in.resize((size_t) Fin);
                ac_in.resize((size_t) (Fin * NC));
                for (int64_t i = 0; i < Fin; i++) {
                    const int32_t * f  = &sx.codes[(size_t) ((f0 + i) * 8)];
                    sem_in[(size_t) i] = f[0] + (int32_t) t.semantic_vocab_offset;
                    for (int64_t k = 0; k < NC; k++) {
                        ac_in[(size_t) (k * Fin + i)] = f[1 + k] + (int32_t) (k * AV);
                    }
                }
                pos.resize((size_t) S);
                for (int64_t i = 0; i < P; i++) {
                    pos[(size_t) i] = (int32_t) i;
                }
                for (int64_t j = 0; j < Fin; j++) {
                    pos[(size_t) (P + j)] = (int32_t) (P + f0 + j);
                }
                ggml_backend_tensor_set(t_sem, sem_in.data(), 0, sem_in.size() * sizeof(int32_t));
                ggml_backend_tensor_set(t_ac, ac_in.data(), 0, ac_in.size() * sizeof(int32_t));
                ggml_backend_tensor_set(t_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
                embed_ctx.P   = P;
                embed_ctx.Fin = Fin;
                LmSample smp;
                smp.tokens.assign((size_t) S, 0);
                smp.targets  = tgt;
                smp.n_masked = (int) n_masked;
                smp.n_prompt = (int) n_prompt;
                smp.s_tr     = (int) K;
                double ce    = 0.0;
                return lm_ckpt_micro_step(ckpt_run, smp, true, &ce) ? ce : NAN;
            };

            ckpt_run.forward_only = true;
            set_rank_mask(false);   // a random subnetwork would differ between runs
            std::string perr;

            // A: no prefix. Q == 0 also clears the store, which is what makes
            // this a genuine square run rather than one with stale history.
            double ce_a = NAN, ce_b = NAN;
            if (lm_kvprefix_run(&kvpfx, &t.lm, sched, arena, lm_ckpt_layer_opts(ckpt_st), mm3_lm_prefix_embed,
                                &pfx_ctx, nullptr, 0, &perr)) {
                ce_a = run(c0 - N, N + K - 1, P + N, P + N);
            }

            // B: prefix [prompt ; frames c0-N .. c0-1], window from c0.
            pfx_ctx.prompt = sx.prompt.data();
            pfx_ctx.codes  = sx.codes.data();
            pfx_ctx.P      = P;
            pfx_ctx.pfx_lo       = c0 - N;
            const int64_t Qb     = P + N - 1;   // history stops at frame c0-2
            pfx_pos.resize((size_t) Qb);
            for (int64_t i = 0; i < P; i++) {
                pfx_pos[(size_t) i] = (int32_t) i;
            }
            for (int64_t i = 0; i < N - 1; i++) {
                pfx_pos[(size_t) (P + i)] = (int32_t) (P + (c0 - N) + i);
            }
            if (lm_kvprefix_run(&kvpfx, &t.lm, sched, arena, lm_ckpt_layer_opts(ckpt_st), mm3_lm_prefix_embed,
                                &pfx_ctx, pfx_pos.data(), Qb, &perr)) {
                ce_b = run(c0 - 1, K, P + 1, P);
            }

            ckpt_run.forward_only = false;
            set_rank_mask(true);

            // The two runs differ in summation order across a 36-layer F32
            // trunk, so they will not be bit-equal. A real defect — a mask
            // column, a stale store, RoPE at the wrong offset — moves the CE
            // by whole nats, not by the sixth decimal.
            const double d   = std::fabs(ce_a - ce_b);
            const double tol = 2e-3 * std::max(1.0, std::fabs(ce_a));
            fprintf(stderr,
                    "[mm3-lm-train] prefix self-test: square %.6f vs prefix %.6f  (|d| %.2e, tol %.2e) -> %s\n",
                    ce_a, ce_b, d, tol, (std::isfinite(d) && d <= tol) ? "PASS" : "FAIL");
            jl("{\"type\":\"prefixSelftest\",\"square\":%.6f,\"prefix\":%.6f,\"delta\":%.3e,\"pass\":%s}",
               ce_a, ce_b, d, (std::isfinite(d) && d <= tol) ? "true" : "false");
            if (!(std::isfinite(d) && d <= tol)) {
                fprintf(stderr, "[mm3-lm-train] refusing to train on an unproven prefix\n");
                rc = 1;
            }
        }
        if (rc != 0) {
            lm_kvprefix_free(&kvpfx);
            lm_ckpt_free(&ckpt_st);
            lm_lora_detach(&lora, &t.lm);
            lm_lora_free(&lora);
            mm3_train_lm_free(&t);
            return rc;
        }
    }

    // ── epoch order ────────────────────────────────────────────────────────
    //
    // A SHUFFLED PASS, not sampling with replacement. The old sampler drew
    // `samples[rng % n]` every step, which over 800 steps and 13 songs gives a
    // typical least-seen/most-seen of 49/74 and a tail as wide as 30/98 — one
    // track carrying triple the weight of another by luck alone. For an
    // identity adapter over one album, uniform exposure is the point. A pass
    // also gives an honest EPOCH BOUNDARY for free, which is what the epoch
    // curve and the 5-epoch average are computed over.
    //
    // The crop is still fresh every time a song comes up, which is what the
    // reference's random-crop patch does per epoch.
    std::vector<int> order;
    size_t           order_pos = 0;
    int              epoch     = 0;
    double           epoch_loss_sum = 0.0;
    int              epoch_n = 0;
    int64_t          epoch_t0 = ggml_time_ms();

    // ── prior preservation ─────────────────────────────────────────────────
    //
    // The regularisation corpus and its cached base distributions. Loaded here,
    // captured below, and drawn from on every reg_every'th step. Its own pass
    // order is separate from the style order on purpose: a reg step must not
    // consume a style epoch, or the epoch curve would stop meaning "how many
    // times the model has seen the album".
    std::vector<MM3LmSample>   reg_samples;
    std::vector<MM3PriorCache> reg_priors;
    std::vector<int>           reg_order;
    int                        reg_epoch_cur = -1;
    const bool reg_on = a.reg_every > 0 && !a.reg_manifest.empty();

    auto next_sample = [&]() -> const MM3LmSample & {
        if (order_pos >= order.size()) {
            lm_epoch_order(&order, (int) samples.size(), true, (uint64_t) a.seed, epoch);
            order_pos = 0;
        }
        return samples[(size_t) order[order_pos++]];
    };

    // ── pause / resume (mm3-lm-resume.h) ───────────────────────────────────
    //
    // The whole point of this machinery is the audio preview loop: the trainer
    // cannot render a sample while it is resident, so the server pauses it,
    // renders the checkpoint with the whole card, and resumes. Everything that
    // makes the resumed run identical to an uninterrupted one lives in the
    // state file — momentum included; see the header for why that matters.
    const std::string pause_file = a.no_pause
                                     ? std::string()
                                     : (a.pause_file.empty() ? mm3_lm_pause_path(a.out_dir) : a.pause_file);
    const std::string state_path = a.out_dir + "/resume-state.bin";
    bool              paused     = false;
    std::string       fatal_msg;

    MM3LmResumeState rstate;
    rstate.rank           = a.rank;
    rstate.alpha          = a.alpha;
    rstate.seed           = a.seed;
    rstate.n_params       = (int32_t) lora.params.size();
    rstate.n_samples      = (int32_t) samples.size();
    rstate.n_holdout      = (int32_t) holdout.size();
    // An optimizer CODE, not a muon flag: adamw and prodigy disagree about what
    // m and v mean, so resuming one into the other must be refused rather than
    // silently accepted the way a 0/1 flag would.
    rstate.optimizer_muon = a.optimizer == "muon" ? 1 : (a.optimizer == "prodigy" ? 2 : 0);

    // ── Prodigy x0 ─────────────────────────────────────────────────────────
    //
    // lm_optim_init seeded x0 from the weights as they are RIGHT NOW, which is
    // correct for a fresh run and wrong for a resumed one: by then the weights
    // have moved, and re-seeding would re-base the <g, x0 - x> numerator so the
    // step-size estimate silently restarts from the pause point.
    if (a.optimizer == "prodigy") {
        const std::string x0p = mm3_prodigy_x0_path(a.out_dir);
        std::string       x0e;
        if (a.resume_path.empty()) {
            if (!mm3_prodigy_x0_save(x0p, opt, &x0e)) {
                fprintf(stderr, "[mm3-lm-train] cannot save prodigy x0: %s\n", x0e.c_str());
                mm3_train_lm_free(&t);
                return 1;
            }
        } else if (!mm3_prodigy_x0_load(x0p, opt, &x0e)) {
            fprintf(stderr, "[mm3-lm-train] cannot restore prodigy x0: %s\n", x0e.c_str());
            mm3_train_lm_free(&t);
            return 1;
        }
    }

    int start_step = 0;
    if (!a.resume_path.empty()) {
        std::string rerr;
        if (!mm3_lm_resume_load(a.resume_path, &rstate, lora, opt, &rerr)) {
            fprintf(stderr, "[mm3-lm-train] resume failed: %s\n", rerr.c_str());
            fatal_msg = "resume failed: " + rerr;
            rc        = 1;
        } else {
            start_step     = rstate.steps_done;
            n_micro        = rstate.n_micro;
            running        = rstate.running;
            epoch          = rstate.epoch;
            epoch_n        = rstate.epoch_n;
            epoch_loss_sum = rstate.epoch_loss_sum;
            order.assign(rstate.order.begin(), rstate.order.end());
            order_pos      = (size_t) rstate.order_pos;
            best_eval      = rstate.best_eval;
            best_eval_step = rstate.best_eval_step;
            // Empty from a v1/v2 file, which simply means the target window
            // refills over the next few epochs.
            ep_means.assign(rstate.epoch_means.begin(), rstate.epoch_means.end());
            opt.opt_step   = rstate.opt_step;
            if (a.optimizer == "prodigy") {
                // d only ever GROWS, so losing it does not perturb the run — it
                // discards every step of estimation done so far and restarts the
                // warm-up from d0, which reads as "the run got worse after a
                // preview" rather than as a bug.
                if (rstate.prodigy_d > 0.0) opt.prodigy_d = rstate.prodigy_d;
                opt.prodigy_r = rstate.prodigy_r;
                fprintf(stderr, "[mm3-lm-train] prodigy resumed at d %.6g\n", opt.prodigy_d);
            }
            opt.opt_iter   = rstate.opt_iter;
            for (int i = 0; i < 4; i++) rng.s[i] = rstate.rng[i];
            fprintf(stderr,
                    "[mm3-lm-train] resumed at step %d/%d (epoch %d, %zu/%zu through the pass, "
                    "optimizer iter %d)\n",
                    start_step, a.steps, epoch, order_pos, order.size(), opt.opt_iter);
            jl("{\"type\":\"resumed\",\"step\":%d,\"totalSteps\":%d,\"epoch\":%d,\"bestEvalStep\":%d}",
               start_step, a.steps, epoch, best_eval_step);
        }
    }
    if (start_step >= a.steps && rc == 0) {
        fprintf(stderr, "[mm3-lm-train] resume state is already at step %d of %d — nothing to do\n",
                start_step, a.steps);
    }

    // Everything that makes a resumed run identical to an uninterrupted one,
    // collected in one place. It was inline in the pause branch until the run's
    // clean exit needed the same snapshot (see a.final_state) — and two copies
    // of this list is exactly how a resume ends up restoring fifteen of sixteen
    // things and looking like a training bug.
    auto snapshot_state = [&](int step) {
        rstate.steps_done     = step;
        rstate.n_micro        = n_micro;
        rstate.running        = running;
        rstate.epoch          = epoch;
        rstate.epoch_n        = epoch_n;
        rstate.epoch_loss_sum = epoch_loss_sum;
        rstate.order.assign(order.begin(), order.end());
        rstate.order_pos      = (int32_t) order_pos;
        rstate.best_eval      = best_eval;
        rstate.best_eval_step = best_eval_step;
        rstate.opt_step       = opt.opt_step;
        rstate.prodigy_d      = opt.prodigy_d;
        rstate.prodigy_r      = opt.prodigy_r;
        rstate.opt_iter       = opt.opt_iter;
        for (int i = 0; i < 4; i++) rstate.rng[i] = rng.s[i];
        // Only the tail the stop can ask for. The whole history would grow
        // without bound in a file already measured in gigabytes, and nothing
        // reads the older entries.
        const size_t keep = (size_t) std::max(1, a.target_loss_epochs);
        rstate.epoch_means.assign(
            ep_means.size() > keep ? ep_means.end() - (long) keep : ep_means.begin(),
            ep_means.end());
    };

    // ── target-loss stopping state ─────────────────────────────────────────
    //
    // ep_means is declared with the other run-spanning statistics above (the
    // resume block restores it). What is left here is the last held-out loss
    // and where the run actually got to — the loop variable is out of scope by
    // the time the clean-exit state is written.
    double              last_eval      = -1.0;
    int                 last_step_done = start_step;
    double              last_win       = 0.0;
    const bool          target_on      = a.target_loss > 0.0f;
    const bool          target_on_eval = target_on && a.target_loss_metric == "eval";
    if (target_on) {
        const std::string how = target_on_eval
                                    ? std::string("held-out loss")
                                    : std::to_string(a.target_loss_epochs)
                                          + "-epoch mean training loss";
        fprintf(stderr, "[mm3-lm-train] stopping at %s <= %.4f, or at step %d, whichever comes first\n",
                how.c_str(), (double) a.target_loss, a.steps);
        jl("{\"type\":\"targetLoss\",\"target\":%.6g,\"metric\":\"%s\",\"epochs\":%d,\"capSteps\":%d}",
           (double) a.target_loss, target_on_eval ? "eval" : "train", a.target_loss_epochs, a.steps);
    }

    // ── prior preservation: load the corpus and capture the base's answers ──
    //
    // ORDER MATTERS. The capture has to happen here — after the checkpoint
    // machinery exists, and before the first optimizer step — because it relies
    // on the adapter being inert. PEFT initialises B to zero, so right now a
    // forward pass IS the frozen base; one step from now it is not, and the
    // teacher would quietly become the student.
    //
    // A resume therefore cannot regenerate: it must find the cache on disk, and
    // says so rather than capturing a contaminated distribution.
    if (reg_on && rc == 0) {
        std::string rerr;
        if (!mm3_lm_load_samples_from(a.reg_manifest, a.reg_captions_dir, a.reg_codes_dir, a.lm_path,
                                      /*trigger_prefix=*/"", /*caption_override=*/"", t,
                                      &reg_samples, &rerr)
            || reg_samples.empty()) {
            fatal_msg = "regularisation set has no usable samples"
                      + (rerr.empty() ? std::string(" — check --reg-manifest/--reg-captions/--reg-codes")
                                      : (": " + rerr));
            fprintf(stderr, "[mm3-lm-train] %s\n", fatal_msg.c_str());
            rc = 1;
        }
    }
    if (reg_on && rc == 0) {
        std::string prior_dir = a.reg_prior_dir;
        if (prior_dir.empty()) {
            prior_dir = a.reg_codes_dir + "/../prior";
        }
        pm_mkdir_p(prior_dir);
        const int W = (int) SL;   // scored width the capture spans
        reg_priors.resize(reg_samples.size());

        int    made = 0, loaded = 0;
        double cov_sum = 0.0;
        const int64_t cap_t0 = ggml_time_ms();
        for (size_t i = 0; i < reg_samples.size() && rc == 0; i++) {
            const MM3LmSample & rs = reg_samples[i];
            // The crop is the SAME deterministic one training will use, which is
            // only true because the recipe truncates from the start. A random
            // crop would need a cache per offset, or a teacher that disagrees
            // with the student about which audio it is looking at.
            const int64_t K      = std::min<int64_t>(K_max, rs.n_frames);
            const bool    at_end = K >= rs.n_frames;
            const int64_t Fin    = at_end ? K : K - 1;
            const int64_t n_sup  = at_end ? K + 1 : K;
            const int64_t P      = (int64_t) rs.prompt.size();
            const int64_t S      = P + Fin;
            if (S > S_max || Fin < 1) {
                fprintf(stderr, "[mm3-lm-train] SKIP reg %s: sequence %lld exceeds %lld\n",
                        rs.id.c_str(), (long long) S, (long long) S_max);
                continue;
            }
            const std::string path = mm3_prior_path(prior_dir, rs.id, a.lm_path, a.reg_topk);
            std::string       lerr;
            if (mm3_prior_load(path, a.reg_topk, (int) n_sup, W, &reg_priors[i], &lerr)) {
                loaded++;
                cov_sum += mm3_prior_coverage(reg_priors[i]);
                continue;
            }
            if (start_step > 0) {
                fatal_msg = "resuming, but the prior cache for " + rs.id + " is missing or stale ("
                          + lerr + "). It can only be captured before the first optimizer step, so "
                            "the adapter is no longer inert and a fresh capture would teach the "
                            "student its own output. Delete the run and start over, or point "
                            "--reg-prior at the original cache.";
                fprintf(stderr, "[mm3-lm-train] %s\n", fatal_msg.c_str());
                rc = 1;
                break;
            }

            // Forward-only through the machinery training already allocated, so
            // the capture costs compute and not VRAM.
            sem_in.resize((size_t) Fin);
            ac_in.resize((size_t) (Fin * NC));
            for (int64_t j = 0; j < Fin; j++) {
                const int32_t * f = &rs.codes[(size_t) (j * 8)];
                sem_in[(size_t) j] = f[0] + (int32_t) t.semantic_vocab_offset;
                for (int64_t k2 = 0; k2 < NC; k2++) {
                    ac_in[(size_t) (k2 * Fin + j)] = f[1 + k2] + (int32_t) (k2 * AV);
                }
            }
            tgt.resize((size_t) n_sup);
            for (int64_t j = 0; j < n_sup; j++) {
                tgt[(size_t) j] = (at_end && j == n_sup - 1)
                                    ? mm3_lm_train_slice_eos(t)
                                    : mm3_lm_train_slice_index(t, rs.codes[(size_t) (j * 8)]);
            }
            if ((int) S != last_mask_S) {
                lm_causal_mask((int) S, &msk);
                ggml_backend_tensor_set(t_msk, msk.data(), 0, msk.size() * sizeof(float));
                last_mask_S = (int) S;
            }
            pos.resize((size_t) S);
            for (int64_t j = 0; j < P; j++)   pos[(size_t) j] = (int32_t) j;
            for (int64_t j = 0; j < Fin; j++) pos[(size_t) (P + j)] = (int32_t) (P + j);
            ggml_backend_tensor_set(t_prompt, rs.prompt.data(), 0, (size_t) P * sizeof(int32_t));
            ggml_backend_tensor_set(t_sem, sem_in.data(), 0, sem_in.size() * sizeof(int32_t));
            ggml_backend_tensor_set(t_ac, ac_in.data(), 0, ac_in.size() * sizeof(int32_t));
            ggml_backend_tensor_set(t_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
            embed_ctx.P = P; embed_ctx.Fin = Fin;

            MM3PriorCache pc;
            pc.k = a.reg_topk; pc.n_pos = (int) n_sup; pc.width = W;
            pc.idx.reserve((size_t) n_sup * (size_t) a.reg_topk);
            pc.p.reserve((size_t) n_sup * (size_t) a.reg_topk);
            ckpt_run.forward_only = true;
            set_rank_mask(false);  // the teacher is the base, not a subnetwork
            ckpt_run.capture_k    = a.reg_topk;
            ckpt_run.capture_idx  = &pc.idx;
            ckpt_run.capture_p    = &pc.p;
            LmSample smp;
            smp.tokens.assign((size_t) S, 0);
            smp.targets  = tgt;
            smp.n_masked = (int) P;
            smp.s_tr     = (int) n_sup;
            const bool ok = lm_ckpt_micro_step(ckpt_run, smp, false, nullptr);
            ckpt_run.capture_k   = 0;
            ckpt_run.capture_idx = nullptr;
            ckpt_run.capture_p   = nullptr;
            ckpt_run.forward_only = false;
            if (!ok || (int) pc.idx.size() != pc.n_pos * pc.k) {
                fatal_msg = "prior capture failed for " + rs.id;
                fprintf(stderr, "[mm3-lm-train] %s\n", fatal_msg.c_str());
                rc = 1;
                break;
            }
            std::string serr;
            if (!mm3_prior_save(path, pc, &serr)) {
                fprintf(stderr, "[mm3-lm-train] prior cache not written (%s) — continuing in memory\n",
                        serr.c_str());
            }
            cov_sum += mm3_prior_coverage(pc);
            reg_priors[i] = std::move(pc);
            made++;
        }
        if (rc == 0) {
            const int n_ok = made + loaded;
            fprintf(stderr,
                    "[mm3-lm-train] prior preservation: %d regularisation song(s) (%d captured, %d cached), "
                    "top-%d covering %.2f%% of the base's probability mass, %lld s\n",
                    n_ok, made, loaded, a.reg_topk,
                    n_ok ? 100.0 * cov_sum / (double) n_ok : 0.0,
                    (long long) ((ggml_time_ms() - cap_t0) / 1000));
            fprintf(stderr,
                    "[mm3-lm-train] every %d%s step trains against the frozen base instead of the "
                    "artist — style exposure is %d of %d steps\n",
                    a.reg_every, a.reg_every == 2 ? "nd" : a.reg_every == 3 ? "rd" : "th",
                    a.steps - a.steps / a.reg_every, a.steps);
            jl("{\"type\":\"prior\",\"songs\":%d,\"captured\":%d,\"cached\":%d,\"topK\":%d,"
               "\"coverage\":%.6f,\"regEvery\":%d}",
               n_ok, made, loaded, a.reg_topk, n_ok ? cov_sum / (double) n_ok : 0.0, a.reg_every);
        }
    }

    for (int step = start_step + 1; step <= a.steps && rc == 0; step++) {
        double acc_loss = 0.0;
        // A REGULARISATION STEP, on the schedule rather than at random, so two
        // runs with the same seed see the same steps and the ratio is exactly
        // what was asked for rather than what a coin gave.
        const bool is_reg = reg_on && !reg_samples.empty() && (step % a.reg_every == 0);
        for (int micro = 0; micro < std::max(1, a.grad_accum) && rc == 0; micro++) {
            const MM3PriorCache * prior = nullptr;
            const MM3LmSample *   sp    = nullptr;
            if (is_reg) {
                // Its own pass, so a reg step never consumes a style epoch — and
                // its position in that pass is DERIVED FROM `step` rather than
                // carried in a cursor. That makes it survive a pause/resume for
                // free: a cursor would have to live in the resume state, and one
                // that silently reset would re-bias exposure in exactly the way
                // the shuffled pass exists to prevent.
                const int nreg      = (int) reg_samples.size();
                const int reg_index = step / a.reg_every - 1;   // 0-based reg step
                const int reg_epoch = reg_index / nreg;
                if (reg_epoch != reg_epoch_cur || reg_order.empty()) {
                    lm_epoch_order(&reg_order, nreg, true, (uint64_t) a.seed ^ 0x9E37ull, reg_epoch);
                    reg_epoch_cur = reg_epoch;
                }
                const int ri = reg_order[(size_t) (reg_index % nreg)];
                sp    = &reg_samples[(size_t) ri];
                prior = &reg_priors[(size_t) ri];
                if (prior->n_pos <= 0) {
                    continue;   // skipped at capture time (too long); no step
                }
            } else {
                sp = &next_sample();
            }
            const MM3LmSample & s = *sp;
            // Caption dropout: a reg step never drops (its cached teacher was
            // captured against the full caption and no other), and a row with no
            // trigger-only form has nothing to drop to.
            const bool drop_caption =
                !is_reg && a.caption_dropout > 0.0 && !s.prompt_trigger_only.empty()
                && lm_rng_uniform(&rng) < (float) a.caption_dropout;
            const std::vector<int32_t> & prompt_ids =
                drop_caption ? s.prompt_trigger_only : s.prompt;
            const int64_t       P = (int64_t) prompt_ids.size();

            // Fresh crop every time this song comes up. `beginning` exists only
            // to reproduce the intros-only failure lm2 hit. A reg step is ALWAYS
            // deterministic from the start — its cached teacher was captured on
            // exactly that crop and no other.
            int64_t K = std::min<int64_t>(K_max, s.n_frames);
            int64_t c0 = 0;
            if (!is_reg && a.crop_mode != "beginning" && s.n_frames > K) {
                const int64_t span = s.n_frames - K;          // largest legal c0
                if (a.crop_mode == "structured") {
                    // One draw picks the bucket, a second places a random crop,
                    // so the number of draws varies per step. That is safe for
                    // resume - the crop RNG rides in the resume state, so the
                    // sequence continues from wherever it actually got to - but
                    // it does mean a run is only bit-reproducible against the
                    // same crop mode, which was already true of the crop itself.
                    const double u = (double) (lm_rng_next(&rng) % 1000000u) / 1000000.0;
                    if (u < a.crop_start_frac) {
                        // Tiled starts: half the share at the true opening, the
                        // rest across the aligned tiles after it (see the field
                        // comment). Extra RNG draws are fine for resume — the
                        // crop RNG rides the resume state.
                        const int64_t max_tiles =
                            std::min<int64_t>((int64_t) a.crop_start_tiles, span / K + 1);
                        if (max_tiles <= 1) {
                            c0 = 0;
                        } else {
                            const double v = (double) (lm_rng_next(&rng) % 1000000u) / 1000000.0;
                            c0 = v < 0.5
                                     ? 0
                                     : K * (1 + (int64_t) (lm_rng_next(&rng) % (uint64_t) (max_tiles - 1)));
                        }
                    } else if (u < a.crop_start_frac + a.crop_end_frac) {
                        c0 = span;                            // flush to the end: EOS
                    } else {
                        c0 = (int64_t) (lm_rng_next(&rng) % (uint64_t) (span + 1));
                    }
                } else {
                    c0 = (int64_t) (lm_rng_next(&rng) % (uint64_t) (span + 1));
                }
            }
            const bool    at_end = (c0 + K) >= s.n_frames;
            const int64_t Fin    = at_end ? K : K - 1;      // frames used as INPUT
            const int64_t n_sup  = at_end ? K + 1 : K;      // supervised positions
            // ONE EXTRA INPUT FRAME WHEN THERE IS HISTORY.
            //
            // Without a prefix the row that predicts frame c0 is the caption's
            // last token — which is the "a song may begin at c0" lesson the
            // crop work exists to remove, and it is only the right predictor
            // at frame 0. With history in front, frame c0-1 comes in as an
            // input and becomes that row instead; the caption stays blind to
            // the prefix, so its own states are unchanged.
            const int64_t lead   = (kv_on && !is_reg && c0 > 0) ? 1 : 0;
            const int64_t f0     = c0 - lead;               // first INPUT frame
            const int64_t Finw   = Fin + lead;
            const int64_t S      = P + Finw;
            const int64_t anchor0 = anchor_song ? f0 : 0;

            // Acoustic loss inputs for this micro-step. Reg steps opt out: the
            // prior path scores soft targets from the base model and has no
            // aligned acoustic ground truth.
            if (depth.wctx && depth.lambda > 0.0f) {
                ckpt_run.aux_head = mm3_depth_train_head;
                ckpt_run.aux_user = &depth;
                depth.codes       = is_reg ? nullptr : s.codes.data();
                depth.c0          = c0;
                depth.n_sup       = n_sup;
                depth.at_end      = at_end;
                depth.opt_step    = step;
            }

            sem_in.resize((size_t) Finw);
            ac_in.resize((size_t) (Finw * NC));
            for (int64_t i = 0; i < Finw; i++) {
                const int32_t * f = &s.codes[(size_t) ((f0 + i) * 8)];
                sem_in[(size_t) i] = f[0] + (int32_t) t.semantic_vocab_offset;
                for (int64_t k = 0; k < NC; k++) {
                    ac_in[(size_t) (k * Finw + i)] = f[1 + k] + (int32_t) (k * AV);
                }
            }
            // Targets: position P-1+j predicts frame c0+j, and the last one is
            // EOS when the crop really reached the end.
            tgt.resize((size_t) n_sup);
            for (int64_t j = 0; j < n_sup; j++) {
                tgt[(size_t) j] = (at_end && j == n_sup - 1)
                                    ? mm3_lm_train_slice_eos(t)
                                    : mm3_lm_train_slice_index(t, s.codes[(size_t) ((c0 + j) * 8)]);
            }

            if ((int) S != last_mask_S) {
                lm_causal_mask((int) S, &msk);
                ggml_backend_tensor_set(t_msk, msk.data(), 0, msk.size() * sizeof(float));
                last_mask_S = (int) S;
            }
            // Prompt at 0..P-1; frames at their TRUE position in the track
            // under --crop-anchor song (see MM3LmTrainArgs). Under "zero" the
            // frames restart at P, which is what every crop used to claim.
            pos.resize((size_t) S);
            for (int64_t i = 0; i < P; i++) pos[(size_t) i] = (int32_t) i;
            for (int64_t j = 0; j < Finw; j++) pos[(size_t) (P + j)] = (int32_t) (P + anchor0 + j);
            ggml_backend_tensor_set(t_prompt, prompt_ids.data(), 0, (size_t) P * sizeof(int32_t));
            ggml_backend_tensor_set(t_sem, sem_in.data(), 0, sem_in.size() * sizeof(int32_t));
            ggml_backend_tensor_set(t_ac, ac_in.data(), 0, ac_in.size() * sizeof(int32_t));
            ggml_backend_tensor_set(t_pos, pos.data(), 0, pos.size() * sizeof(int32_t));

            // ── frozen KV prefix ──────────────────────────────────────
            //
            // A regularisation step gets NO prefix: its teacher distribution
            // was captured with none, and scoring against it through a
            // different context would be measuring drift that is not there.
            if (kv_on) {
                int64_t Q = 0;
                if (!is_reg) {
                    // History runs up to f0, because frame f0 itself is an
                    // input to the window (see `lead`).
                    const int64_t pfx_lo = std::max<int64_t>(0, f0 - a.prefix_frames);
                    const int64_t npfx   = f0 - pfx_lo;
                    pfx_ctx.prompt       = prompt_ids.data();
                    pfx_ctx.codes        = s.codes.data();
                    pfx_ctx.P            = P;
                    pfx_ctx.pfx_lo       = pfx_lo;
                    Q                    = P + npfx;
                    pfx_pos.resize((size_t) Q);
                    for (int64_t i = 0; i < P; i++) {
                        pfx_pos[(size_t) i] = (int32_t) i;
                    }
                    for (int64_t i = 0; i < npfx; i++) {
                        pfx_pos[(size_t) (P + i)] = (int32_t) (P + pfx_lo + i);
                    }
                }
                // Q == 0 still runs, because it is what CLEARS the store: the
                // window accs its own K/V over columns n..n+S and stale values
                // there from a longer previous prefix would be read as history.
                std::string perr;
                if (!lm_kvprefix_run(&kvpfx, &t.lm, sched, arena, lm_ckpt_layer_opts(ckpt_st),
                                     mm3_lm_prefix_embed, &pfx_ctx, pfx_pos.data(), Q, &perr)) {
                    fprintf(stderr, "[mm3-lm-train] %s\n", perr.c_str());
                    rc = 1;
                    break;
                }
            }

            if (drop_caption) n_dropped++;
            // A fresh subnetwork per micro-step, uploaded BEFORE the forward so
            // the checkpoint recompute sees the same one.
            set_rank_mask(true);
            double ce = 0.0;
            bool   ok = false;
            if (a.ckpt) {
                // lm_ckpt_micro_step reads S from tokens.size() and the trained
                // span from (n_masked, s_tr); the ids themselves are never read
                // because the embedding is overridden.
                embed_ctx.P   = P;
                embed_ctx.Fin = Finw;
                LmSample smp;
                smp.tokens.assign((size_t) S, 0);
                smp.targets  = tgt;
                smp.n_masked = (int) (P + lead);
                smp.n_prompt = (int) P;
                smp.s_tr     = (int) n_sup;
                if (prior) {
                    // Score against what the base model itself predicted here,
                    // not against this song's actual codes. `targets` goes
                    // unused; LmChunkLabelGuard switches on soft_k.
                    smp.soft_k   = prior->k;
                    smp.soft_idx = prior->idx;
                    smp.soft_p   = prior->p;
                    smp.s_tr     = std::min<int>(smp.s_tr, prior->n_pos);
                }
                ok = lm_ckpt_micro_step(ckpt_run, smp, true, &ce);
                // The gradient tripwire, once, on the first real step: central
                // finite differences against the exact scatter-added dL/dh.
                // The runtime-LoKR audit taught that unvalidated reimplemented
                // math ships confident and wrong; this one refuses to.
                if (ok && depth.wctx && depth.lambda > 0.0f && !depth_fd_done && !is_reg) {
                    depth_fd_done = true;
                    std::string fderr;
                    if (!mm3_depth_train_fdcheck(depth, ckpt_run, smp, &fderr)) {
                        fprintf(stderr, "[mm3-lm-train] %s\n", fderr.c_str());
                        mm3_train_lm_free(&t);
                        return 1;
                    }
                }
            } else {
                ggml_init_params gip = { arena.size(), arena.data(), true };
                ggml_context *   ctx = ggml_init(gip);
                ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
                ggml_tensor *    loss = nullptr;
                build_graph(ctx, gf, P, Finw, n_sup, &loss);
                std::vector<ggml_tensor *> gacc;
                lm_optim_fill_gacc(&opt, gf, &gacc);
                ggml_build_backward_expand(ctx, gf, gacc.data());
                {
                    LmLabelGuard guard(t_lab, tgt.data(), (int) n_sup, (int) SL);
                    ggml_backend_sched_reset(sched);
                    ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
                    if (ok) {
                        float lv = 0.0f;
                        ggml_backend_tensor_get(loss, &lv, 0, sizeof(float));
                        ce = (double) lv;
                    }
                }
                ggml_free(ctx);
            }
            if (!ok) {
                fprintf(stderr, "[mm3-lm-train] micro-step failed (lower --max-frames?)\n");
                rc = 1;
            } else {
                acc_loss += ce;
                if (!is_reg) {
                    running += ce;
                    n_micro++;
                    n_style_seg++;
                } else {
                    reg_running += ce;
                    reg_n_micro++;
                }
            }
        }
        if (rc) break;

        if (!lm_optim_step(&opt, sched, &stats)) {
            fprintf(stderr, "[mm3-lm-train] optimizer step failed\n");
            rc = 1;
            break;
        }
        const double win = acc_loss / std::max(1, a.grad_accum);
        last_step_done = step;
        last_win       = win;
        // Every step, unlike the human log: the chart wants them all, and one
        // JSON line per step is nothing next to a 3.9 s step.
        // stepMs is the PER-STEP time, not elapsed: it is the number that makes a
        // spill visible (3.9 s fitting vs ~40 s paging), so the UI gets it
        // directly rather than having to difference timestamps.
        const int64_t now_ms  = ggml_time_ms();
        const int64_t step_ms = now_ms - t_step0;
        t_step0 = now_ms;
        // `reg` marks a prior-preservation step. Its loss is a DIFFERENT
        // quantity — cross-entropy against a soft distribution rather than a
        // one-hot code — so plotting the two on one series would draw a curve
        // that means two things. Consumers filter on this flag.
        jl("{\"type\":\"step\",\"step\":%d,\"totalSteps\":%d,\"loss\":%.6f,\"lr\":%.9g,"
           "\"gradNorm\":%.6f,\"clipScale\":%.6f,\"ms\":%lld,\"stepMs\":%lld,\"reg\":%s,"
           "\"depthLoss\":%.6f}",
           step, a.steps, win, (double) stats.lr, (double) stats.grad_norm, (double) stats.clip,
           (long long) (now_ms - t_start), (long long) step_ms, is_reg ? "true" : "false",
           depth.last_loss);
        jl("{\"type\":\"progress\",\"completed\":%d,\"total\":%d,\"phase\":\"train\"}", step, a.steps);
        if (step == 1 || step % 10 == 0 || step == a.steps) {
            fprintf(stderr, "[mm3-lm-train] step %d/%d loss %.4f lr %.3g |g| %.3f clip %.3f %lld s\n", step,
                    a.steps, win, (double) stats.lr, (double) stats.grad_norm, (double) stats.clip,
                    (long long) ((ggml_time_ms() - t_start) / 1000));
        }
        // VRAM after step 1 (everything is allocated by then) and periodically.
        // Peak occupancy is the single thing that decides whether this run takes
        // 4 s or 40 s a step, and it is invisible from inside the app otherwise.
        if (step == 1 || step % 25 == 0) {
            size_t vfree = 0, vtotal = 0;
            lm_vram_query(t.lm.backend, &vfree, &vtotal);
            if (vtotal > 0) {
                const long long used_mb = (long long) ((vtotal - vfree) / (1024 * 1024));
                const long long tot_mb  = (long long) (vtotal / (1024 * 1024));
                jl("{\"type\":\"vram\",\"step\":%d,\"usedMb\":%lld,\"freeMb\":%lld,\"totalMb\":%lld}",
                   step, used_mb, (long long) (vfree / (1024 * 1024)), tot_mb);
                if (step == 1) {
                    fprintf(stderr, "[mm3-lm-train] VRAM after step 1: %lld/%lld MB used (%lld free)\n",
                            used_mb, tot_mb, (long long) (vfree / (1024 * 1024)));
                }
            }
        }
        // ── epoch boundary ────────────────────────────────────────────────
        // One pass over the training songs. With 13 songs that is 13 steps, so
        // the epoch mean is a 13-crop average — the smooth line the per-step
        // noise is drawn against, and what the 5-epoch average is taken over.
        if (!is_reg) {
            epoch_loss_sum += win;
            epoch_n++;
        }
        bool epoch_closed = false;
        if (!is_reg && order_pos >= order.size() && !order.empty()) {
            epoch++;
            const double emean = epoch_loss_sum / std::max(1, epoch_n);
            // The target-loss window, one entry per COMPLETE pass. Reg steps
            // never reach here: their loss is soft-target CE against the frozen
            // base, a different quantity, and they do not consume a style epoch.
            ep_means.push_back(emean);
            epoch_closed = true;
            jl("{\"type\":\"epoch\",\"epoch\":%d,\"loss\":%.6f,\"step\":%d,\"lr\":%.9g,\"ms\":%lld}",
               epoch, emean, step, (double) stats.lr, (long long) (ggml_time_ms() - epoch_t0));
            epoch_loss_sum = 0.0;
            epoch_n        = 0;
            epoch_t0       = ggml_time_ms();
        }

        // ── held-out evaluation ───────────────────────────────────────────
        bool eval_fresh = false;
        if (a.eval_every > 0 && !eval_plan.empty()
            && (step % a.eval_every == 0 || step == a.steps)) {
            const double ev = run_eval();
            if (ev >= 0.0) {
                last_eval  = ev;
                eval_fresh = true;
                jl("{\"type\":\"eval\",\"step\":%d,\"loss\":%.6f,\"crops\":%zu}", step, ev,
                   eval_plan.size());
                fprintf(stderr, "[mm3-lm-train] step %d: held-out loss %.4f (train %.4f)\n", step, ev, win);
                if (ev < best_eval || best_eval < 0.0) {
                    best_eval      = ev;
                    best_eval_step = step;
                }
            }
        }

        const bool saved_here = a.save_every > 0 && (step % a.save_every == 0 || step == a.steps);
        std::string ckpt_dir;
        if (saved_here) {
            ckpt_dir = save_ckpt(step, win);
        }

        // ── target loss reached? ──────────────────────────────────────────
        //
        // Checked AFTER the checkpoint so the step that hits the target has
        // one, and before the pause check so a target hit ends the run rather
        // than paying for a preview it will never resume from.
        //
        // The window has to be FULL — and it survives a pause, because the
        // history rides in the resume state. Without both, a run previewing
        // every 50 steps closes only ~4 epochs per segment, a 5-epoch window
        // would never fill, and the stop would silently never fire.
        if (target_on) {
            double     value = -1.0;
            // Each metric is only asked when it has just MOVED: a fresh
            // held-out number, or a pass that has just closed. Re-testing a
            // stale value every step would stop the run at an arbitrary point
            // after the real crossing rather than at it.
            const bool ready = target_on_eval
                                   ? (eval_fresh && last_eval >= 0.0)
                                   : (epoch_closed
                                      && (int) ep_means.size() >= a.target_loss_epochs);
            if (ready) {
                if (target_on_eval) {
                    value = last_eval;
                } else {
                    double sum = 0.0;
                    for (int i = 0; i < a.target_loss_epochs; i++) {
                        sum += ep_means[ep_means.size() - 1 - (size_t) i];
                    }
                    value = sum / a.target_loss_epochs;
                }
            }
            if (ready && value <= (double) a.target_loss) {
                if (!saved_here && a.save_every > 0) {
                    ckpt_dir = save_ckpt(step, win);
                }
                fprintf(stderr,
                        "[mm3-lm-train] target loss reached at step %d: %s %.4f <= %.4f — stopping\n",
                        step, target_on_eval ? "held-out" : "epoch mean", value,
                        (double) a.target_loss);
                jl("{\"type\":\"target_stop\",\"step\":%d,\"value\":%.6f,"
                   "\"targetLoss\":%.6g,\"metric\":\"%s\",\"ckpt\":\"%s\"}",
                   step, value, (double) a.target_loss, target_on_eval ? "eval" : "train",
                   json_escape(ckpt_dir).c_str());
                break;
            }
        }

        // ── pause for an audio preview ────────────────────────────────────
        //
        // One stat() per step, against a ~4 s step. The checkpoint is exported
        // FIRST (if this step did not already export one) so the `paused` event
        // can name something renderable; then the full optimizer state goes to
        // disk and this process exits, handing the card to the render.
        if (mm3_lm_pause_requested(pause_file)) {
            if (!saved_here) {
                ckpt_dir = save_ckpt(step, win);
            }
            snapshot_state(step);

            const int64_t t_save0 = ggml_time_ms();
            std::string   serr;
            if (!mm3_lm_resume_save(state_path, rstate, lora, opt, &serr)) {
                // A pause that cannot be resumed is worse than no pause: the
                // run would silently restart from zero. Fail loudly instead.
                fprintf(stderr, "[mm3-lm-train] cannot save resume state: %s\n", serr.c_str());
                fatal_msg = "cannot save resume state: " + serr;
                rc        = 1;
                break;
            }
            mm3_lm_resume_meta_write(a.out_dir, state_path, rstate, "pause", a.optimizer.c_str(),
                                     a.adapter_type.c_str(), a.steps, win);
            mm3_lm_pause_clear(pause_file);
            fprintf(stderr, "[mm3-lm-train] paused at step %d/%d — state saved in %lld ms, %s\n", step,
                    a.steps, (long long) (ggml_time_ms() - t_save0), state_path.c_str());
            jl("{\"type\":\"paused\",\"step\":%d,\"totalSteps\":%d,\"loss\":%.6f,\"state\":\"%s\","
               "\"ckpt\":\"%s\"}",
               step, a.steps, win, json_escape(state_path).c_str(), json_escape(ckpt_dir).c_str());
            paused = true;
            break;
        }
    }

    // ── the state a CONTINUATION resumes from ──────────────────────────────
    //
    // Until this existed, state was written only when a preview asked for it,
    // so a finished run left a state file from its last preview point — tens of
    // steps behind the checkpoint it had just exported. "Run this for 250 more
    // steps" then silently rewound to step 200 and retrained ground it had
    // already covered, which is indistinguishable from the run being worse the
    // second time.
    //
    // Only on a CLEAN exit. A run that died mid-step has an optimizer whose
    // momentum and weights disagree about which step they are on, and freezing
    // that would hand back a resume that trains on a lie; the last pause state
    // is behind but true, so it is left alone to be resumed from instead.
    if (rc == 0 && !paused && a.final_state && last_step_done > start_step) {
        snapshot_state(last_step_done);
        const int64_t t_fin0 = ggml_time_ms();
        std::string   ferr;
        if (mm3_lm_resume_save(state_path, rstate, lora, opt, &ferr)) {
            mm3_lm_resume_meta_write(a.out_dir, state_path, rstate, "final", a.optimizer.c_str(),
                                     a.adapter_type.c_str(), a.steps, last_win);
            fprintf(stderr, "[mm3-lm-train] resume state written at step %d in %lld ms — this run can be continued\n",
                    last_step_done, (long long) (ggml_time_ms() - t_fin0));
            jl("{\"type\":\"resumable\",\"step\":%d,\"state\":\"%s\"}",
               last_step_done, json_escape(state_path).c_str());
        } else {
            // Not fatal: the checkpoints are on disk and the run succeeded. The
            // user loses the ability to continue it, and should be told which.
            fprintf(stderr, "[mm3-lm-train] could not write the final resume state: %s\n",
                    ferr.c_str());
        }
    }
    if (n_dropped > 0) {
        fprintf(stderr, "[mm3-lm-train] caption dropout: %d of %d style steps THIS SEGMENT "
                        "used the trigger alone (%.1f%%, asked for %.1f%%)\n",
                n_dropped, n_style_seg,
                n_style_seg ? 100.0 * n_dropped / (double) n_style_seg : 0.0,
                100.0 * a.caption_dropout);
    }
    if (reg_n_micro > 0) {
        fprintf(stderr, "[mm3-lm-train] prior preservation: %d steps, mean soft CE %.4f\n",
                reg_n_micro, reg_running / (double) reg_n_micro);
    }
    fprintf(stderr, "[mm3-lm-train] %s after %d micro-steps, mean loss %.4f, %lld s\n",
            rc ? "STOPPED" : paused ? "paused" : "done", n_micro, n_micro ? running / n_micro : 0.0,
            (long long) ((ggml_time_ms() - t_start) / 1000));
    if (best_eval >= 0.0) {
        // The point of the holdout: which checkpoint to reach for FIRST, decided
        // by a number rather than in hindsight by ear. Not a claim that the
        // others are useless — the alk3 ladder taught us the ear can prefer a
        // more-degraded checkpoint that carries more identity.
        jl("{\"type\":\"best\",\"step\":%d,\"loss\":%.6f}", best_eval_step, best_eval);
        fprintf(stderr, "[mm3-lm-train] best held-out loss %.4f at step %d — start the ear test there\n",
                best_eval, best_eval_step);
    }
    if (rc) {
        jl("{\"type\":\"fatal\",\"message\":\"%s\"}",
           fatal_msg.empty() ? "training stopped early — see the engine log"
                             : json_escape(fatal_msg).c_str());
    } else if (!paused) {
        // A paused run has already said so, and `done` is what the server reads
        // as "this training is over" — emitting it here would end the run at the
        // first preview.
        jl("{\"type\":\"done\",\"steps\":%d,\"meanLoss\":%.6f,\"ms\":%lld}", n_micro,
           n_micro ? running / n_micro : 0.0, (long long) (ggml_time_ms() - t_start));
    }

    ggml_backend_sched_free(sched);
    if (a.ckpt) {
        lm_kvprefix_free(&kvpfx);
        lm_ckpt_free(&ckpt_st);
    }
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctx_static);
    lm_optim_free(&opt);
    lm_lora_detach(&lora, &t.lm);
    lm_lora_free(&lora);
    mm3_train_lm_free(&t);
    return rc;
}
