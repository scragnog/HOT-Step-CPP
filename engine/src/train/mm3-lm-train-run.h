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
#include "train/mm3-lm-verify-export.h"
#include "train/lm-pissa.h"
#include "train/lm-hra.h"
#include "train/lm-prefix.h"
#include "minimax/mm3-tokenizer.h"

#include <algorithm>
#include <cmath>
#include <map>
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
//   * RareConcepts/<his public MM3 dataset> + the training-tournament card
//     (45 tracks, 1000 steps, simpletuner_config.json in every checkpoint)
//
// The album C config, verbatim, is what the numbers below now mirror:
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
    /** --drop-over-frames N: leave out every style track longer than N frames
     *  at load (0 = keep all). The whole-song recipe (2026-09-11) asks for the
     *  engine's 9000-frame ceiling so that every track trains as one sequence;
     *  a track that does not fit could only be trained in pieces, at positions
     *  past the base's 10240-token range, so the presets drop it instead and
     *  the log names what was dropped. Reg corpora are never filtered. */
    int64_t     drop_over_frames = 0;
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
    /** Varied end supervision (lever 4a, 2026-09-08). The end share above is a
     *  single deterministic crop per song, so twelve songs teach twelve
     *  memorised frames and the adapter's own plans never reach a state it
     *  recognises as an ending (P(EOS) = 1.0 at the trained frame, 0 one frame
     *  earlier). With this on, every end step draws its crop length uniformly
     *  in [end_crop_min, K] and its frozen-prefix span uniformly in
     *  [0, prefix_frames], so the same real ending is seen from many distances
     *  and with many amounts of history: an ending as a STATE, not a frame.
     *  Off = the pinned crop exactly as before. */
    bool        end_crop_vary = false;
    int64_t     end_crop_min  = 128;
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
     *  hypothetical: the first album C run recorded `albumC` in the sidecar,
     *  none of the 14 MOSS-written captions contained it, and rendering with
     *  `albumC, <caption>` therefore bolted an unseen token sequence onto
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
     *  could produce the album but never be steered ("albumC, 140 BPM,
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
    /** Lyrics dropout (2026-09-08): share of style steps whose prompt carries
     *  NO lyrics (the assembler writes the instrumental marker). SimpleTuner's
     *  positioned continuation spans train without lyrics on most steps, and
     *  its adapters end songs where ours do not; the hypothesis is that always
     *  training with the full lyrics binds song structure to them so tightly
     *  that the base's "lyrics done, wrap up, stop" is overwritten. Reg steps
     *  never drop (their teacher was captured with the full prompt). */
    double      lyrics_dropout = 0.0;

    /** Artist token (textual inversion), V3. k learned vectors accumulated onto
     *  the first k prompt positions, whose ids are placeholder copies spliced at
     *  the front of the tokenised prompt.
     *
     *  Front insertion, not a string splice: MM3 tokenises the assembled prompt
     *  in one call, so putting the trigger in the caption text would give an
     *  unpredictable id count. Inserting ids post-tokenisation fixes both the
     *  count and the offset, and survives `prompt_cap` truncation, which keeps
     *  the FIRST P ids.
     *
     *  Empty name = off, and the whole feature is then unreachable. */
    std::string artist_token = "";
    int         artist_k     = 8;
    std::string artist_init  = "band";
    bool        artist_only  = false;
    float       artist_lr    = 0.0f;  // soft-prompt LR; 0 = same as --lr (P1b)
    float       lora_plus_ratio = 1.0f;  // LoRA+: B at ratio x A's LR (AdamW-rule tensors only)

    // ── Post-LoRA parameterizations (2026-09-05) ───────────────────────────
    //
    // Shared with train-lm through lm-graph.h, so this backend only chooses
    // them. All four are LoRA-type only and mutually exclusive where the DiT
    // says so; cmd_mm3_lm_train refuses every other combination at exit 2.
    //
    //   rslora — alpha/sqrt(r) instead of alpha/r. At r256 that is a 16x
    //            difference in delta strength, so it MUST also reach the file
    //            (use_rslora) or the runtime applies a different adapter than
    //            the one that trained.
    //   dora   — learned per-output magnitude; runtime and merge both honour it
    //   hira   — W (.) (s*BA): the delta is not low-rank, so MERGE MODE ONLY
    //   loha   — (A1B1) (.) (A2B2), LyCORIS hada_w* on disk; merge mode only
    bool        rslora   = false;
    bool        dora     = false;
    bool        hira     = false;
    bool        loha     = false;
    //   pissa  — SVD init for the plain LoRA slot. The residual is folded into a
    //            FROZEN A0/B0 pair rather than written into the base, because the
    //            shipped recipe trains against q8_0 and there is nowhere to put an
    //            F32 residual (lm-pissa.h). Exports a rank-2r plain LoRA, so no
    //            loader learns anything. Not resumable.
    //   hra    — `rank` Householder reflections on each site's INPUT; its own
    //            parameterization, not a LoRA modifier. Exports an exact rank-r
    //            plain LoRA plus a vector sidecar for resume.
    bool        pissa    = false;
    int         pissa_oversample = 8;
    int         pissa_iters      = 2;
    //   hot_pizza — PiSSA with the rank-dropout mask on the principal component
    //            itself (QwLoraPair::hot_pizza). Implies pissa. The recipe that
    //            won the 2026-09-06 MM3 blind tests; export identical to pissa.
    bool        hot_pizza        = false;
    //   pissa_cache_dir — directory for the SVD init cache (lm-pissa.h); '' = off.
    //   pissa_f16 — frozen A0/B0 in F16 (halves their VRAM; init cancels to f16).
    std::string pissa_cache_dir;
    bool        pissa_f16        = false;
    bool        hra      = false;
    /** After each checkpoint export, load it straight back with the RUNTIME
     *  loader and check that the scale, the tensors and the parameterization
     *  flags round-trip (train/mm3-lm-verify-export.h). Off by default: it
     *  costs one adapter reload per checkpoint. On for the gates. */
    bool        verify_export = false;
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

    // ── Attention formulation (--attn, R3 of the flash-attn roadmap) ───────
    //
    // Same three values, same meanings and the same shared code as
    // `train-lm` — lm-graph.h routes the whole-head attention site through
    // lm_attn_flash when LmLayerOpts::attn_flash is set, and lm-ckpt.h copies
    // LmCkptCfg::attn_flash/attn_prec into every segment's opts. This trainer
    // was the one caller of that machinery that never set the fields (D12);
    // this flag is that wiring, not a new primitive.
    //
    //   "exact"     = the shipped manual chain (mul_mat -> soft_max_ext ->
    //                 mul_mat), which retains an [S_kv,S,Nh] softmax per layer.
    //                 THE DEFAULT, and byte-identical to pre-flag runs.
    //   "flash"     = fused GGML_OP_FLASH_ATTN_TRAIN / _BACK, so the softmax is
    //                 never materialised and attention memory is linear in S.
    //                 MM3 crops are long (S = prompt + 1500..4096 frames), which
    //                 is exactly where the quadratic term dominates.
    //   "flash-f32" = the same fused ops pinned to GGML_PREC_F32, i.e. the v1
    //                 scalar kernels instead of the TF32 tensor-core ones
    //                 "flash" selects. Slower; it separates "did fusion move the
    //                 training" from "did TF32 move it", and it is the mode the
    //                 --fd-check gate should use.
    //
    // NOT combinable with --prefix-frames: the frozen KV prefix makes the mask
    // rectangular (S_kv = n_pfx + S), which is outside both the capability probe
    // below and lm_build_trunk_embeds' flash arm (it asserts). Refused at the
    // CLI, exit 2, rather than coerced.
    //
    // Flash gradients are NOT bit-identical to exact — same drift family as
    // --weights bf16 — so the default stays the shipped arithmetic.
    std::string attn       = "exact";

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
    // can begin anywhere. bghira's 2026-08-22 album C campaign reports the two
    // symptoms this predicts — an instant-sound-at-0:00 artifact and tempo
    // drift mid-track — and reports that position-labelled windowed crops fix
    // the pacing. Kept switchable because it changes the recipe: a run trained
    // under "zero" is not comparable with one trained under "song".
    std::string crop_anchor = "song";     // song | zero
    /** Stage A of the end-of-song work (2026-09-07). When set, the style corpus
     *  loader reads `<codes_dir>/trim.json` ({"<id>": keep_frames}) and drops
     *  every frame after keep_frames, so the EOS target follows the last
     *  MUSICAL frame instead of the 2-4 s of digital silence most rips carry.
     *  The file is produced from the AUDIO (tools/mm3-trim-silence) and is
     *  inspected before use; the trainer only applies it. Reg corpora are never
     *  trimmed. Off by default. */
    bool        trim_trailing_silence = false;

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

    // ── TRAINABLE KV PREFIX (train/lm-prefix.h) ────────────────────────────
    //
    // A DIFFERENT FEATURE that shares one mechanism with the block above, and
    // the two names are close enough to be worth spelling out. `prefix_frames`
    // is FROZEN audio history — real frames of the song, forward-only, no
    // gradient. `prefix_n` is prefix tuning: n learned K/V columns per layer,
    // parameters, with no position and no audio behind them. Both arrive at
    // attention through lm_kv_splice, and lm_train_layer asserts they are never
    // both set, so the CLI refuses the pair rather than letting that assert
    // decide.
    //
    // Checkpointed path only. Every per-layer opt and the rectangular mask come
    // from lm-ckpt.h (LmCkptCfg::pfx); the naive path would need its own copy of
    // both, for a route nobody trains on.
    int         prefix_n     = 0;
    float       prefix_sigma = 0.02f;

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
    /** Ending-targeted prior (room plan rev 7, 2026-09-08). A reg step scores
     *  its soft-target loss on the LAST N supervised rows of the excerpt only;
     *  inputs, prefix and teacher are unchanged, so the model still sees the
     *  whole context and only the trained span shrinks (n_masked moves up,
     *  s_tr = N). The base teacher puts its stopping decision on the final row
     *  and ~no EOS mass anywhere else, so scoring all ~500 rows is 500 rows of
     *  "be the base" to 1 of "stop": the likeness cost measured on 2026-09-08.
     *  0 = every row (today's behaviour, bit-identical). The loss is a mean
     *  over the scored rows, so each retained row's coefficient rises by
     *  n_sup/N; that number is logged. */
    int         reg_score_last = 0;
    /** Style-step counterpart (2026-09-08 evening): score only the last N
     *  supervised rows of every style crop. SimpleTuner's `continuation` mode
     *  scores the last 128 frames of a span whose earlier frames are context,
     *  and its adapters end songs ~30-67% of the time where ours never do; this
     *  ports that half of the objective. 0 = every row (today's behaviour). */
    int         score_last = 0;
    /** --score-last-end-only (2026-09-09): apply --score-last to END crops
     *  only. Interior crops keep every row scored, so the style supervision
     *  that FAITHSL threw away (6/6 endings, zero likeness) stays intact, and
     *  the ending still gets its concentrated share on the crops that hold
     *  one. */
    bool        score_last_end_only = false;
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
    std::vector<int32_t> prompt_no_lyrics;      // lyrics dropout: caption kept, lyrics replaced by the instrumental marker
    std::vector<int32_t> codes;           // [n_frames * 8], warm-up row already dropped
    int64_t              n_frames = 0;
    /** Absolute frame index of codes[0] in the track this sample was cut from.
     *  0 for ordinary corpora. A regularisation corpus of EXCERPTS (the last K
     *  frames of a base-model plan, so the excerpt ends at a real EOS) carries
     *  the excerpt's true start here, read from the manifest's `frame_offset`,
     *  so under --crop-anchor song the teacher capture and the reg step both
     *  place the frames at the positions the base saw. */
    int64_t              frame_offset = 0;
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

    // Artist token (textual inversion): [H, k], accumulated onto the FIRST k
    // prompt positions. The placeholder ids were spliced at the front of
    // sm.prompt, which is also what makes this truncation-safe — a prompt
    // clipped to `prompt_cap` keeps its first P ids, so the token survives a
    // caption long enough to be cut.
    ggml_tensor *      t_art = nullptr;
    int                art_k = 0;
    // Stream position of this embedding's first prompt id. 0 for the training
    // window and the FD check; the KV prefill sets it per chunk, because a
    // prompt longer than --prefix-chunk is embedded in pieces and the token
    // span [0, art_k) may fall in a later piece, be split across two, or be
    // absent from a chunk altogether.
    int64_t            art_off = 0;
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
    ggml_tensor * out    = e_prompt ? ggml_concat(ctx, e_prompt, frames, 1) : frames;

    // Artist token, applied AFTER the concat and deliberately so. ggml_concat
    // has no backward — it appears in ggml.c only as a constructor — so accing
    // onto e_prompt before the concat would put a parameter upstream of a
    // non-differentiable op and abort. Downstream of it, the concat's own
    // sources still need no gradient and it is never differentiated.
    if (e.t_art && e.art_k > 0 && e.P > 0) {
        GGML_ASSERT(e.t_art->ne[0] == H && e.t_art->ne[1] == e.art_k);
        // The part of the token span [0, art_k) that lies inside this
        // embedding's prompt ids [art_off, art_off + P). Whole span at
        // offset 0 is the common case (the window, the FD check, and a
        // prefill whose first chunk holds the whole prompt); the slices are
        // the prefill chunks of a long prompt.
        const int64_t lo = std::min<int64_t>(e.art_off, e.art_k);
        const int64_t hi = std::min<int64_t>(e.art_k, e.art_off + e.P);
        if (lo == 0 && hi == e.art_k) {
            out = ggml_acc(ctx, out, e.t_art, out->nb[1], out->nb[2], out->nb[3], 0);
        } else if (hi > lo) {
            ggml_tensor * span = ggml_view_2d(ctx, e.t_art, H, hi - lo, e.t_art->nb[1], (size_t) lo * e.t_art->nb[1]);
            out = ggml_acc(ctx, out, span, out->nb[1], out->nb[2], out->nb[3], 0);
        }
    }
    return out;
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
    e.art_off     = i0;   // this chunk's first prompt id sits at stream position i0
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
                                     bool trim_trailing,
                                     const MM3TrainLm & t,
                                     std::vector<MM3LmSample> * out, std::string * err,
                                     int64_t drop_over_frames = 0) {
    struct { std::string manifest, captions_dir, codes_dir, lm_path; } a {
        manifest, captions_dir, codes_dir, lm_path };
    int n_dropped_long = 0;
    // Stage A: per-track keep_frames from <codes_dir>/trim.json, applied below.
    std::map<std::string, int64_t> trim;
    if (trim_trailing) {
        std::string tbuf;
        if (!mm3_lm_read_file(a.codes_dir + "/trim.json", &tbuf)) {
            if (err) *err = "--trim-trailing-silence set but " + a.codes_dir + "/trim.json is missing";
            return false;
        }
        yyjson_doc * td = yyjson_read(tbuf.c_str(), tbuf.size(), 0);
        yyjson_val * tr = td ? yyjson_doc_get_root(td) : nullptr;
        if (!tr || !yyjson_is_obj(tr)) {
            if (td) yyjson_doc_free(td);
            if (err) *err = a.codes_dir + "/trim.json is not a JSON object";
            return false;
        }
        yyjson_obj_iter ti = yyjson_obj_iter_with(tr);
        yyjson_val *    tk;
        while ((tk = yyjson_obj_iter_next(&ti))) {
            yyjson_val * tv = yyjson_obj_iter_get_val(tk);
            if (tv && yyjson_is_num(tv)) trim[yyjson_get_str(tk)] = (int64_t) yyjson_get_num(tv);
        }
        yyjson_doc_free(td);
        fprintf(stderr, "[mm3-lm-train] trailing-silence trim: %zu entries from %s/trim.json\n", trim.size(),
                a.codes_dir.c_str());
    }
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
            {
                auto tit = trim.find(id);
                if (tit != trim.end() && tit->second > 0 && tit->second < sm.n_frames) {
                    fprintf(stderr, "[mm3-lm-train] trim %s: %lld -> %lld frames (-%.1f s of trailing silence)\n",
                            id.c_str(), (long long) sm.n_frames, (long long) tit->second,
                            (double) (sm.n_frames - tit->second) / 25.0);
                    sm.n_frames = tit->second;
                    sm.codes.resize((size_t) (sm.n_frames * 8));
                }
            }
            if (drop_over_frames > 0 && sm.n_frames > drop_over_frames) {
                // --drop-over-frames: the whole-song recipe. See MM3LmTrainArgs.
                fprintf(stderr, "[mm3-lm-train] DROP %s: %lld frames (%.0f s) exceeds the %lld-frame window (%.0f s)\n",
                        id.c_str(), (long long) sm.n_frames, (double) sm.n_frames / 25.0,
                        (long long) drop_over_frames, (double) drop_over_frames / 25.0);
                n_dropped_long++;
                continue;
            }
            {
                yyjson_val * ov = yyjson_obj_get(s, "frame_offset");
                if (ov && yyjson_is_num(ov)) sm.frame_offset = (int64_t) yyjson_get_num(ov);
            }
            if (!trigger_prefix.empty()) {
                // Front of the FIRST line, comma + space — the training-row shape.
                size_t lead = caption.find_first_not_of(" \t\r\n\xEF\xBB\xBF");
                std::string body = lead == std::string::npos ? caption : caption.substr(lead);
                // IDEMPOTENT. A shared caption written for a style adapter will
                // usually open with the trigger already, and prepending a second
                // copy trains "the album-C artist, the album-C artist, ..." — a token
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
            mm3_tokenizer_encode(tok, mm3_assemble_prompt(caption, std::string()), &sm.prompt_no_lyrics);
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
    if (n_dropped_long > 0) {
        fprintf(stderr, "[mm3-lm-train] %d track(s) longer than %lld frames left out (--drop-over-frames); %zu remain\n",
                n_dropped_long, (long long) drop_over_frames, samples.size());
    }
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
                                    shared, a.trim_trailing_silence, t, out, err, a.drop_over_frames);
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
//
// "Thousands of times the noise floor" is true of a plain LoRA and NOT of every
// parameterization — HiRA's gradients are 50x smaller on the same sites, which
// put its loss change ~30x above the floor and failed the gate on arithmetic.
// `--fd-eps` is therefore a FLOOR on the step, not the step: see the step-floor
// block by the probe loop for the measurement and the rule.
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

    // ── the soft-prompt halves, under the SAME gate as the adapter ─────────
    //
    // --artist-token and --prefix-n add parameters the FD arm never saw: the
    // token sits in the embedding stage (its gradient comes from lm-ckpt.h's
    // P1B backward, a route nothing else here exercises), and the prefix sits
    // in every layer's K/V. Both are gated exactly like a LoRA factor —
    // whole-tensor directional derivative against the measured loss change.
    //
    // The placeholder id here is ARBITRARY (0 works): this check never decodes
    // anything, and the trainer's own splice is the thing under test on the
    // real run. What matters is that k consecutive ids exist at the front of
    // the prompt so the acc lands where mm3_lm_build_embed asserts it does.
    const int fd_art_k = a.artist_token.empty() ? 0 : std::max(1, std::min(a.artist_k, 8));
    std::vector<int32_t> fd_prompt = smp.prompt;
    if (fd_art_k > 0) {
        fd_prompt.insert(fd_prompt.begin(), (size_t) fd_art_k, (int32_t) 0);
    }
    // TRUNCATE THE PROMPT. A real MM3 prompt is ~1,125 tokens, which would put
    // the NAIVE arm of this check back over the card for exactly the reason
    // the trainer needed checkpointing — and the naive arm is half of what is
    // being compared. A gradient check needs the same graph STRUCTURE, not a
    // meaningful caption; cutting mid-BPE is acceptable here and nowhere else.
    const int64_t       P   = std::min<int64_t>(prompt_cap, (int64_t) fd_prompt.size());
    const int64_t       K   = std::min<int64_t>(frames, smp.n_frames);
    const int64_t       Fin = K - 1;          // never at_end: keep the case simple
    const int64_t       n_sup = K;
    const int64_t       S   = P + Fin;
    fprintf(stderr, "[mm3-fd] %s: prompt %lld (of %zu, truncated) + %lld frames = seq %lld, rank %d, eps %.3g\n",
            smp.id.c_str(), (long long) P, fd_prompt.size(), (long long) K, (long long) S, a.rank, eps);
    if (fd_art_k > 0 && P < fd_art_k) {
        fprintf(stderr, "[mm3-fd] --fd-prompt %lld is shorter than the artist token span (k=%d)\n",
                (long long) P, fd_art_k);
        mm3_train_lm_free(&t);
        return 1;
    }

    // ── --attn (R3): the SAME gate, run against the fused ops ─────────────
    //
    // Both arms of this check honour the flag, which is the point: with
    // `--attn flash-f32` the numeric side measures the FUSED forward and the
    // analytic side is the FUSED backward, so the finite-difference rung
    // validates GGML_OP_FLASH_ATTN_TRAIN_BACK on its own terms rather than
    // diffing it against the exact arm. Under isolation that is a verdict.
    const bool      fd_flash    = (a.attn == "flash" || a.attn == "flash-f32");
    const ggml_prec fd_prec_req = (a.attn == "flash-f32") ? GGML_PREC_F32 : GGML_PREC_DEFAULT;
    LmLayerOpts     fd_opts;
    fd_opts.attn_flash = fd_flash;
    fd_opts.attn_prec  = fd_prec_req;
    if (fd_flash) {
        // Same refusal-not-fallback discipline as train-lm: a `false` here would
        // otherwise make the scheduler split the fused ops onto the CPU, which
        // is correct, unusably slow, and looks like a pass on every number this
        // command prints. Nkv is the NATIVE GQA width — nothing on this path
        // expands heads before lm_attn_flash.
        const float ascale = 1.0f / sqrtf((float) c.head_dim);
        bool        pf = false, pb = false;
        dit_flash_probe(t.lm.backend, c.head_dim, c.n_heads, c.n_kv_heads, (int) S, (int) S, /*B=*/1, ascale, &pf,
                        &pb);
        if (!(pf && pb)) {
            fprintf(stderr,
                    "[mm3-fd] --attn %s: backend %s does not support the fused attention ops at this geometry "
                    "(D %d, Nh %d, Nkv %d, S %lld, B 1) — fwd %s / bwd %s. Refusing: a CPU split would look "
                    "like a pass. Use --attn exact.\n",
                    a.attn.c_str(), ggml_backend_name(t.lm.backend), c.head_dim, c.n_heads, c.n_kv_heads,
                    (long long) S, pf ? "yes" : "NO", pb ? "yes" : "NO");
            mm3_f32_isolate_free(&iso);
            mm3_train_lm_free(&t);
            return 1;
        }
        fprintf(stderr, "[mm3-fd] --attn %s: %s runs FLASH_ATTN_TRAIN and _BACK at D %d, Nh %d, Nkv %d, S %lld "
                        "— no CPU split. Requested arithmetic: %s\n",
                a.attn.c_str(), ggml_backend_name(t.lm.backend), c.head_dim, c.n_heads, c.n_kv_heads,
                (long long) S, fd_prec_req == GGML_PREC_F32 ? "strict f32" : "tf32 where available");
    }

    LmLora     lora;
    const bool fd_lokr = a.is_lokr();
    const bool fd_init_ok =
        fd_lokr ? lm_lokr_init(&lora, &t.lm, 0, c.n_layers, a.lokr_dim, a.lokr_alpha, a.lokr_factor,
                               a.lokr_decompose_both, (uint64_t) a.seed, &err)
                : lm_lora_init(&lora, &t.lm, 0, c.n_layers, a.rank, (float) a.alpha, (uint64_t) a.seed,
                               /*b_sigma=*/1e-2f, &err,
                               LmLoraOpts{ a.dora, a.hira, a.loha, a.pissa, a.hra, a.hot_pizza, a.pissa_f16 });
    if (!fd_init_ok) {
        fprintf(stderr, "[mm3-fd] %s init failed: %s\n", fd_lokr ? "LoKr" : "LoRA", err.c_str());
        mm3_train_lm_free(&t);
        return 1;
    }
    // THE GATE MUST TRAIN THE SAME GRAPH THE RUN DOES. rsLoRA changes the
    // in-graph scale, so a gate that skipped it would validate a different
    // function than the one the main run builds (the fd-check/main-run
    // divergence this file's two init sites exist to make possible).
    if (!fd_lokr && a.rslora) {
        lm_lora_apply_rslora(&lora);
        fprintf(stderr, "[mm3-fd] rsLoRA: in-graph scale alpha/sqrt(r) = %.4f\n", (double) lora.scale);
    }
    // Same reasoning as rsLoRA above. PiSSA changes what A and B ARE and adds a
    // frozen -s*B0A0 term at every adapted site, so a gate that skipped the init
    // would validate a graph the run does not build. It also leaves B non-zero,
    // which is what makes dL/dA measurable at all.
    if (!fd_lokr && a.pissa) {
        LmPissaStats ps;
        if (!lm_pissa_init_standalone(&lora, a.pissa_oversample, a.pissa_iters, &ps, &err, a.pissa_cache_dir, a.lm_path)) {
            fprintf(stderr, "[mm3-fd] PiSSA init failed: %s\n", err.c_str());
            lm_lora_detach(&lora, &t.lm);
            lm_lora_free(&lora);
            mm3_train_lm_free(&t);
            return 1;
        }
        fprintf(stderr,
                "[mm3-fd] PiSSA: %d sites, captured energy mean %.2f%% min %.2f%% (a random rank-%d subspace "
                "would hold ~%.2f%%)\n",
                ps.sites, 100.0 * ps.energy_mean, 100.0 * ps.energy_min, a.rank,
                100.0 * (double) a.rank / (double) c.hidden_size);
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
    // ── the soft-prompt parameters ────────────────────────────────────────
    //
    // Both are PERTURBED away from their production init, for the reason the
    // LoKr block above states: a zero-initialised artist token gives dL/dt an
    // honest value but a zero WEIGHT, and the probe would then compare a
    // gradient against a loss curve that is flat in the only direction it can
    // move. The prefix already inits non-zero (lm-prefix.h), so it keeps its
    // real sigma; the token is given the same 1e-2 the LoRA's B gets.
    ggml_tensor *         fd_art     = nullptr;
    ggml_context *        fd_art_ctx = nullptr;
    ggml_backend_buffer_t fd_art_buf = nullptr;
    if (fd_art_k > 0) {
        ggml_init_params ap = { 2 * ggml_tensor_overhead(), nullptr, true };
        fd_art_ctx          = ggml_init(ap);
        fd_art              = ggml_new_tensor_2d(fd_art_ctx, GGML_TYPE_F32, H, fd_art_k);
        ggml_set_name(fd_art, "artist_token");
        ggml_set_param(fd_art);
        fd_art_buf = ggml_backend_alloc_ctx_tensors(fd_art_ctx, t.lm.backend);
        if (!fd_art_buf) {
            fprintf(stderr, "[mm3-fd] artist-token buffer allocation failed\n");
            mm3_train_lm_free(&t);
            return 1;
        }
        LmRng              arng;
        lm_rng_seed(&arng, (uint64_t) a.seed ^ 0xA5A5C0FFEEull);
        std::vector<float> av((size_t) ggml_nelements(fd_art), 0.0f);
        lm_rng_fill_normal(&arng, av, 1e-2f);
        ggml_backend_tensor_set(fd_art, av.data(), 0, av.size() * sizeof(float));
        fprintf(stderr, "[mm3-fd] artist token: k=%d at the front of the prompt, perturbed to sigma 1e-2 "
                        "(a zero token would make the numeric side flat)\n", fd_art_k);
    }
    LmPrefix fd_pfx;
    if (a.prefix_n > 0) {
        std::string perr;
        if (!lm_prefix_init(&fd_pfx, &t.lm, 0, c.n_layers, a.prefix_n, S, (uint64_t) a.seed, a.prefix_sigma,
                            &perr)) {
            fprintf(stderr, "[mm3-fd] trainable prefix init failed: %s\n", perr.c_str());
            mm3_train_lm_free(&t);
            return 1;
        }
        fprintf(stderr, "[mm3-fd] trainable prefix: n=%d over %d layers, %zu params\n", fd_pfx.n, c.n_layers,
                fd_pfx.n_params);
    }
    std::vector<ggml_tensor *> fd_params = lora.params;
    if (fd_art) {
        fd_params.push_back(fd_art);
    }
    for (size_t i = 0; i < fd_pfx.params.size(); i++) {
        fd_params.push_back(fd_pfx.params[i]);
    }

    LmOptim opt;
    opt.optimizer = "adamw";
    if (!lm_optim_init(&opt, fd_params, t.lm.backend, &err)) {
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
    // F32 under --attn exact (the shipped allocation, byte for byte), F16 under
    // flash: ggml_flash_attn_train asserts mask->type == GGML_TYPE_F16.
    ggml_tensor * t_msk    = lm_mask_alloc(ctx_static, (a.prefix_n + S) * S, fd_flash);
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
    if (fd_pfx.active()) {
        // Every row sees the whole prefix (pfx_lo 0, n_prompt 0) — lm-prefix.h
        // contract 2, and the same call lm_ckpt_upload_mask makes.
        lm_causal_mask_prefix(fd_pfx.n, /*pfx_lo=*/0, (int) S, /*n_prompt=*/0, &msk);
    } else {
        lm_causal_mask((int) S, &msk);
    }
    ggml_backend_tensor_set(t_prompt, fd_prompt.data(), 0, (size_t) P * sizeof(int32_t));
    ggml_backend_tensor_set(t_sem, sem_in.data(), 0, sem_in.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_ac, ac_in.data(), 0, ac_in.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    lm_mask_set(t_msk, msk);   // converts to F16 when the buffer is F16 (--attn flash)

    MM3EmbedCtx embed_ctx{ &t, t_prompt, t_sem, t_ac, P, Fin, fd_art, fd_art_k };
    // The naive arm resolves the prefix per layer from these tables, exactly as
    // lm_ckpt_layer_kv does for the checkpointed one — so both arms of the
    // comparison below run the same attention.
    if (fd_pfx.active()) {
        fd_opts.pfx_k_all = &fd_pfx.k;
        fd_opts.pfx_v_all = &fd_pfx.v;
        fd_opts.pfx_zero  = fd_pfx.zero;
        fd_opts.pfx_n     = fd_pfx.n;
    }

    // HRA's forward is `rank` reflections per site, so the trunk's node count
    // scales with --rank (lm-graph.h). Both graphs below build the WHOLE trunk,
    // so they and the arena behind them come from the bank rather than from a
    // constant that assumed a plain LoRA. With no such arm in play both are the
    // 65,536 and 512 MiB they always were.
    const int fd_trunk_extra = lm_trunk_extra_nodes(&t.lm, 0, c.n_layers);
    const int fd_fwd_nodes   = 65536 + fd_trunk_extra;
    const int fd_bwd_nodes   = 65536 + LM_GRAPH_BWD_NODE_MULT * fd_trunk_extra;
    std::vector<uint8_t> arena(std::max<size_t>((size_t) 512 << 20, lm_graph_arena_bytes(fd_bwd_nodes)));
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
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, fd_fwd_nodes, false);
        ggml_tensor *    h_in = mm3_lm_build_embed(ctx, embed_ctx);
        ggml_tensor *    hid =
            lm_build_trunk_embeds(ctx, &t.lm, h_in, t_pos, t_msk, (int) S, 0, c.n_layers, fd_opts);
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
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, fd_bwd_nodes, true);
        ggml_tensor *    h_in = mm3_lm_build_embed(ctx, embed_ctx);
        ggml_tensor *    hid =
            lm_build_trunk_embeds(ctx, &t.lm, h_in, t_pos, t_msk, (int) S, 0, c.n_layers, fd_opts);
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
    // `extra` selects the parameterization-specific tensors that are NOT the
    // plain first/second factor. Without it a DoRA run would gate its A and B
    // and say nothing at all about dL/dm, and a LoHa run would gate pair 1 and
    // never touch pair 2 — a backward can be wrong in exactly the half the
    // probes skip, which is the whole reason this gate exists.
    enum ProbeExtra { PROBE_FACTOR = 0, PROBE_A2, PROBE_B2, PROBE_M };
    // `direct` names a parameter that is not a LoRA factor at all — the artist
    // token, or one layer's prefix K/V. When it is set, layer/slot/is_a/extra
    // are ignored and `dname` is the label.
    struct Probe { int layer, slot; bool is_a; int extra; ggml_tensor * direct; const char * dname; };
    // `is_a` means FIRST factor, whichever parameterization is in play:
    //   LoRA -> A / B        LoKr -> w1 / (w2 | w2_a)
    // Without this the probes address q.A and q.B, which are null on a LoKr
    // pair, and grad_vec trips GGML_ASSERT on the param_slot lookup.
    auto probe_tensor = [&](const QwLoraPair & q, bool first, int extra) -> ggml_tensor * {
        if (q.hra) return q.A;  // reflection vectors are the only parameter
        if (extra == PROBE_A2) return q.A2;
        if (extra == PROBE_B2) return q.B2;
        if (extra == PROBE_M)  return q.m;
        if (q.has_lokr()) {
            return first ? q.w1 : (q.w2 ? q.w2 : q.w2_a);
        }
        return first ? q.A : q.B;
    };
    auto probe_suffix = [&](const QwLoraPair & q, bool first, int extra) -> const char * {
        if (q.hra) return "hra_v";
        if (extra == PROBE_A2) return "hada_A2";
        if (extra == PROBE_B2) return "hada_B2";
        if (extra == PROBE_M)  return "dora_m";
        if (q.has_lokr()) {
            return first ? "lokr_w1" : (q.w2 ? "lokr_w2" : "lokr_w2_a");
        }
        return first ? "A" : "B";
    };
    // Resolve a probe to the tensor it perturbs and the name it prints under.
    auto probe_par = [&](const Probe & pr) -> ggml_tensor * {
        if (pr.direct) {
            return pr.direct;
        }
        return probe_tensor(lora.layers[pr.layer].p[pr.slot], pr.is_a, pr.extra);
    };
    auto probe_label = [&](const Probe & pr, char * out, size_t n) {
        if (pr.direct) {
            snprintf(out, n, "%s", pr.dname);
            return;
        }
        const QwLoraPair & q = lora.layers[pr.layer].p[pr.slot];
        snprintf(out, n, "L%d.%s.%s", pr.layer, lm_slot_peft_name(pr.slot),
                 probe_suffix(q, pr.is_a, pr.extra));
    };
    std::vector<Probe> probes;
    {
        const int layers[3] = { 0, c.n_layers / 2, c.n_layers - 1 };
        const int slots[3]  = { QW_LORA_Q, QW_LORA_GATE, QW_LORA_DOWN };
        for (int i = 0; i < n_probe; i++) {
            // An HRA pair has ONE parameter, so `is_a` no longer separates two
            // probes: without swapping which axis varies fastest, consecutive
            // probes would address the identical tensor and the gate would
            // silently cover a third of what it claims.
            if (lora.hra) {
                probes.push_back(Probe{ layers[(i / 3) % 3], slots[i % 3], true, PROBE_FACTOR, nullptr, nullptr });
            } else {
                probes.push_back(
                    Probe{ layers[i % 3], slots[(i / 3) % 3], (i % 2) == 0, PROBE_FACTOR, nullptr, nullptr });
            }
        }
        // Appended, never substituted: the plain-LoRA probes above still run,
        // so a DoRA/LoHa gate is strictly a superset of the LoRA one.
        if (lora.loha) {
            probes.push_back(Probe{ 0, QW_LORA_Q, true, PROBE_A2, nullptr, nullptr });
            probes.push_back(Probe{ 1 % std::max(1, c.n_layers), QW_LORA_GATE, true, PROBE_B2, nullptr, nullptr });
        }
        if (lora.dora) {
            probes.push_back(Probe{ 0, QW_LORA_Q, true, PROBE_M, nullptr, nullptr });
            probes.push_back(Probe{ 1 % std::max(1, c.n_layers), QW_LORA_DOWN, true, PROBE_M, nullptr, nullptr });
        }
        // The soft-prompt parameters. One probe for the token; for the prefix,
        // the FIRST and LAST adapted layer's K and V — the prefix is resolved
        // per layer from a table, so a probe on layer 0 alone would say nothing
        // about the indexing.
        if (fd_art) {
            probes.push_back(Probe{ 0, 0, true, PROBE_FACTOR, fd_art, "artist_token" });
        }
        if (fd_pfx.active()) {
            const int lo = fd_pfx.layer_lo, hi = fd_pfx.layer_hi - 1;
            probes.push_back(Probe{ 0, 0, true, PROBE_FACTOR, fd_pfx.k[(size_t) lo], "prefix_k.first" });
            probes.push_back(Probe{ 0, 0, true, PROBE_FACTOR, fd_pfx.v[(size_t) lo], "prefix_v.first" });
            if (hi != lo) {
                probes.push_back(Probe{ 0, 0, true, PROBE_FACTOR, fd_pfx.k[(size_t) hi], "prefix_k.last" });
                probes.push_back(Probe{ 0, 0, true, PROBE_FACTOR, fd_pfx.v[(size_t) hi], "prefix_v.last" });
            }
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
        g_naive.push_back(grad_vec(probe_par(pr)));
    }

    // Checkpointed gradients for the same probes.
    std::vector<std::vector<float>> g_ckpt;
    {
        LmCkptCfg cc;
        cc.chunk     = 64;
        cc.pfx       = fd_pfx.active() ? &fd_pfx : nullptr;
        // lm_ckpt_layer_opts copies these into every P2/P3/P7 segment graph, so
        // both arms of the comparison below run the same attention formulation.
        cc.attn_flash = fd_flash;
        cc.attn_prec  = fd_prec_req;
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
            // P1B: without this the artist token's gradient is silently zero on
            // the checkpointed arm, which the route comparison WOULD catch (the
            // naive arm has it) — that is the point of running both.
            ckpt_run.embed_trainable = fd_art != nullptr;
            ckpt_run.embed_has_param = fd_art != nullptr;
            if (backward_ckpt()) {
                for (const Probe & pr : probes) {
                    g_ckpt.push_back(grad_vec(probe_par(pr)));
                }
            } else {
                fprintf(stderr, "[mm3-fd] checkpointed backward failed\n");
            }
        }
    }

    // ── THE STEP IS PER PROBE, and `eps` is only its floor ─────────────────
    //
    // What has to clear the forward's own resolution is not the step on the
    // PARAMETER, it is the LOSS CHANGE that step produces — and along the unit
    // gradient direction that is exactly `2 * eps * ||g||`. So a fixed eps
    // measures every probe at a different signal-to-noise ratio, in direct
    // proportion to its gradient norm.
    //
    // That is not a theoretical worry, it is what broke the HiRA gate. HiRA's
    // delta is `W (.) (s B A)`, i.e. a plain LoRA's delta scaled entrywise by
    // the frozen weight, so its factor gradients come out 16x to 68x smaller
    // than the same site's under plain LoRA (measured, this fixture, all eight
    // probes). Same graph, same data, same probe list — only the parameter
    // scale differs.
    //
    // The forward here is bit-deterministic (|l1 - l0| is exactly 0 above), so
    // its floor is not variance, it is QUANTISATION: the logits come back as
    // F32 and no amount of repeating moves them. Central-differencing divides
    // that fixed floor by 2*eps, which puts a fixed ABSOLUTE error on `num` —
    // and a fixed absolute error is a relative failure only where ||g|| is
    // small. Measured on the oasis_morningglory fixture, rank 4, 2 F32 layers,
    // eight probes per arm, at the CLI's default eps 1e-2:
    //
    //   |num - ||g|||   plain LoRA 2.6e-6 .. 2.9e-5   HiRA 1.2e-6 .. 1.8e-5
    //
    // The same band for both. Only `rel` differs, because HiRA divides it by a
    // gradient norm 50x smaller. Sweeping eps over 0.005 .. 2.0 on HiRA's worst
    // probe (||g|| 5.5e-4) settles it:
    //
    //   eps    0.005  0.01   0.02   0.05   0.1    0.2    0.5    1.0    2.0
    //   rel    0.059  0.033  0.007  0.000  0.003  0.003  0.000  0.000  0.000
    //
    // The error SHRINKS as the step grows and flips sign along the way. A wrong
    // backward does neither: it misses by a factor, the same factor at every
    // eps. Plain LoRA on the same probes runs the other way — flat to eps 0.2,
    // then 0.021 / 0.076 / 0.202 at 0.5 / 1.0 / 2.0, the eps^2 truncation the
    // 2e-2 bar was set against. Both arms collapse onto one curve in the loss
    // change: rel ~ 3.5e-7 / (2*eps*||g||) on the rounding side, for every
    // parameterization. So HiRA's 3.3e-2 was the ESTIMATOR, and the fix belongs
    // in the estimator.
    //
    // Hence a floor on the loss change rather than a bigger bar for HiRA. It is
    // the rule train-dit's own T4 gate already uses ("steps are chosen so the
    // loss moves by a target amount, not by a fixed h", dit-selftest.h) in its
    // cheapest form: raise eps only for a probe whose loss change would sit
    // under `dl_min`, and leave every other probe on exactly the step it had —
    // so the LoRA, DoRA, LoHa and PiSSA numbers are untouched, bit for bit.
    //
    // dl_min = 256 F32 ULPs of the loss. The measured floor is ~3.5e-7 on a
    // loss of 11.5, i.e. ~0.4 ULP, so 256 ULPs caps the rounding term at ~1.5e-3
    // — an order of magnitude inside the 2e-2 bar — while staying far below the
    // ~2e-2 loss change where truncation starts to bite. Expressed in ULPs
    // rather than as an absolute so it follows a fixture whose loss sits
    // somewhere else.
    const double l_ulp =
        (double) std::nextafterf((float) std::abs(l0), 3.4e38f) - (double) (float) std::abs(l0);
    const double dl_min = 256.0 * l_ulp;
    fprintf(stderr, "[mm3-fd] step floor: dL >= %.2e (256 F32 ULPs of the loss); a probe whose "
                    "2*eps*||g|| clears it keeps eps %.3g\n", dl_min, eps);

    // ── numeric: directional derivative along v = g/||g|| ──
    fprintf(stderr, "\n[mm3-fd] %-26s %10s %13s %9s %13s %8s\n", "probe (whole tensor)", "n", "||g||",
            "step", "numeric", "rel");
    int                 n_bad = 0;
    double              worst = 0.0;
    int                 n_raised = 0;
    std::vector<double> fd_rel;
    bool                gate_fd_ran = false, gate_fd_pass = false;
    for (size_t i = 0; i < probes.size(); i++) {
        const Probe & pr  = probes[i];
        ggml_tensor * par = probe_par(pr);

        const std::vector<float> & g = g_naive[i];
        double norm2 = 0.0;
        for (float x : g) norm2 += (double) x * (double) x;
        const double gnorm = std::sqrt(norm2);

        // The whole of the adaptive rule.
        double h = eps;
        if (gnorm > 0.0 && 2.0 * eps * gnorm < dl_min) {
            h = dl_min / (2.0 * gnorm);
            n_raised++;
        }

        std::vector<float> w0((size_t) ggml_nelements(par)), wtmp(w0.size());
        ggml_backend_tensor_get(par, w0.data(), 0, w0.size() * sizeof(float));

        double num = std::nan("");
        if (gnorm > 0.0) {
            for (size_t k = 0; k < w0.size(); k++) wtmp[k] = (float) (w0[k] + h * g[k] / gnorm);
            ggml_backend_tensor_set(par, wtmp.data(), 0, wtmp.size() * sizeof(float));
            const double lp = forward_loss();
            for (size_t k = 0; k < w0.size(); k++) wtmp[k] = (float) (w0[k] - h * g[k] / gnorm);
            ggml_backend_tensor_set(par, wtmp.data(), 0, wtmp.size() * sizeof(float));
            const double lmn = forward_loss();
            num = (lp - lmn) / (2.0 * h);
        }
        ggml_backend_tensor_set(par, w0.data(), 0, w0.size() * sizeof(float));

        const double rel = std::abs(num - gnorm) / std::max(1e-12, gnorm);
        char         nm[64];
        probe_label(pr, nm, sizeof(nm));
        fprintf(stderr, "[mm3-fd] %-26s %10zu %13.6e %9.3g %13.6e %8.3f\n", nm, g.size(), gnorm, h, num, rel);
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
    // The bar is 2e-2, not tighter, and it is the SAME bar for every
    // parameterization. This is a central difference, so it carries a genuine
    // O(h^2 * third-derivative) truncation error that no amount of precision
    // removes; 2e-2 is ~10x the worst observed (2e-3 at 2 layers), which leaves
    // room for probe-to-probe variation without admitting a real scale error —
    // a wrong gradient scale misses by a FACTOR, not by a percent. The step
    // floor above is what keeps that one bar honest across parameterizations
    // whose gradient norms differ by 50x; raising the bar for the awkward one
    // would have hidden exactly the defect the bar exists to catch.
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
                "\n[mm3-fd] GATE %s: finite differences, F32-isolated, --attn %s, bar %.0e, %d/%zu probes, "
                "worst %.4f (step raised on %d)\n",
                fd_ok ? "PASS" : "FAIL", a.attn.c_str(), fd_bar, (int) probes.size() - n_bad, probes.size(),
                worst, n_raised);
        if (!fd_ok) {
            fprintf(stderr,
                    "[mm3-fd]   The analytic gradient disagrees with the measured loss change. Unlike the\n"
                    "[mm3-fd]   route comparison below, this catches a defect BOTH backward routes share\n"
                    "[mm3-fd]   — a wrong scale on the chunked CE, or a missing term.\n");
        }
        jl("{\"type\":\"gate\",\"check\":\"finite-difference-f32\",\"pass\":%s,\"worst\":%.6e,"
           "\"bar\":%.6e,\"probes\":%d,\"dlMin\":%.6e,\"stepRaised\":%d,\"attn\":\"%s\",\"attnPrec\":\"%s\"}",
           fd_ok ? "true" : "false", worst, fd_bar, (int) probes.size(), dl_min, n_raised, a.attn.c_str(),
           fd_flash ? dit_flash_prec_label(t.lm.backend).c_str() : "n/a");
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
            fprintf(stderr, "[mm3-fd] GATE %s: F32-isolated (%d layers), --attn %s, bar 2e-3, worst %.2e\n",
                    gate_pass ? "PASS" : "FAIL", f32_layers, a.attn.c_str(), wc);
            if (!gate_pass) {
                fprintf(stderr,
                        "[mm3-fd]   The checkpointed path disagrees with whole-graph autodiff by more than\n"
                        "[mm3-fd]   F32 reassociation explains. Suspect the two hooks this trainer added to\n"
                        "[mm3-fd]   lm-ckpt.h: the untied scored head (head_w/head_row0/head_v) and the\n"
                        "[mm3-fd]   frame-embedding entry (embed_build/embed_user), or the chunked-CE scale.\n");
            }
            jl("{\"type\":\"gate\",\"check\":\"ckpt-vs-naive-f32\",\"pass\":%s,\"worst\":%.6e,"
               "\"bar\":2.0e-03,\"layers\":%d,\"attn\":\"%s\",\"attnPrec\":\"%s\"}",
               gate_pass ? "true" : "false", wc, f32_layers, a.attn.c_str(),
               fd_flash ? dit_flash_prec_label(t.lm.backend).c_str() : "n/a");
        }
    } else if (isolated) {
        // A verdict was asked for and one arm never produced gradients: that is
        // a failure, not a missing measurement.
        gate_ran  = true;
        gate_pass = false;
        fprintf(stderr, "[mm3-fd] GATE FAIL: one of the two gradient routes did not run\n");
    }

    // What the fused kernels ACTUALLY ran, read back from the backend rather
    // than restated from the flag: a tf32 request silently drops to the v1
    // scalar kernels at D != 128, on pre-Ampere, and on an unaligned view.
    if (fd_flash) {
        fprintf(stderr, "[mm3-fd] --attn %s resolved to %s\n", a.attn.c_str(),
                dit_flash_prec_label(t.lm.backend).c_str());
    }

    ggml_backend_sched_free(sched);
    lm_ckpt_free(&ckpt_st);
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctx_static);
    lm_optim_free(&opt);
    if (fd_art_buf) {
        ggml_backend_buffer_free(fd_art_buf);
    }
    if (fd_art_ctx) {
        ggml_free(fd_art_ctx);
    }
    lm_prefix_free(&fd_pfx);
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

    // ── Artist token: splice k placeholder ids at the front of every prompt ──
    //
    // Post-tokenisation, deliberately. MM3 encodes the assembled prompt in one
    // call, so putting a trigger in the caption TEXT would tokenise to an
    // unpredictable number of ids and leave the parameter's span unknowable.
    // Inserting ids fixes both count and offset, and the front position is what
    // makes it survive `prompt_cap` — truncation keeps the first P ids.
    int art_placeholder = -1;
    const int art_k     = a.artist_token.empty() ? 0 : a.artist_k;
    if (art_k > 0) {
        if (a.artist_k < 1 || a.artist_k > 64) {
            fprintf(stderr, "[mm3-lm-train] --artist-token-k must be 1-64\n");
            mm3_train_lm_free(&t);
            return 1;
        }
        // The tokenizer is loaded per use, the same way mm3_lm_load_samples_from
        // does it — MM3TrainLm does not carry one.
        MM3Model tok_stub = {};
        tok_stub.lm_file.found = true;
        tok_stub.lm_file.path  = a.lm_path;
        tok_stub.lm_file.name  = a.lm_path;
        tok_stub.lm_cfg.semantic_vocab_offset = t.semantic_vocab_offset;
        MM3Tokenizer art_tok = {};
        std::string  tok_err;
        if (!mm3_tokenizer_load(tok_stub, &art_tok, &tok_err)) {
            fprintf(stderr, "[mm3-lm-train] artist token: %s\n", tok_err.c_str());
            mm3_train_lm_free(&t);
            return 1;
        }
        std::vector<int32_t> seed_ids;
        mm3_tokenizer_encode(art_tok, a.artist_init, &seed_ids);
        if (seed_ids.empty()) {
            fprintf(stderr, "[mm3-lm-train] --artist-token-init \"%s\" encodes to no tokens\n",
                    a.artist_init.c_str());
            mm3_train_lm_free(&t);
            return 1;
        }
        art_placeholder = seed_ids[0];
        for (size_t i = 0; i < samples.size(); i++) {
            samples[i].prompt.insert(samples[i].prompt.begin(), (size_t) art_k, (int32_t) art_placeholder);
            if (!samples[i].prompt_trigger_only.empty()) {
                samples[i].prompt_trigger_only.insert(samples[i].prompt_trigger_only.begin(), (size_t) art_k,
                                                      (int32_t) art_placeholder);
            }
        }
        fprintf(stderr,
                "[mm3-lm-train] artist token \"%s\": k=%d placeholder=%d (from \"%s\", %zu ids, first used)%s\n",
                a.artist_token.c_str(), art_k, art_placeholder, a.artist_init.c_str(), seed_ids.size(),
                a.artist_only ? ", adapter frozen" : ", trained alongside the adapter");
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
    // The longest track in frames, across everything the run will crop from.
    // The sequence budget below is clamped to it: a whole-song recipe asks for
    // the engine's 9000-frame ceiling so that every track trains whole, and
    // sizing the checkpoints, arena and attention probe off 9000 frames on an
    // album whose longest track is 4000 frames allocated 29 GB for a run that
    // needed far less (three albums on 2026-09-10, all 29.0-29.1 GB peak
    // regardless of track length). No crop can exceed the track it is cut from,
    // so the clamp changes nothing about what is trained, only what is reserved.
    int64_t longest_track = 0;
    for (const auto & s : samples) max_prompt = std::max(max_prompt, (int64_t) s.prompt.size());
    for (const auto & s : holdout) max_prompt = std::max(max_prompt, (int64_t) s.prompt.size());
    for (const auto & s : samples) longest_track = std::max(longest_track, s.n_frames);
    for (const auto & s : holdout) longest_track = std::max(longest_track, s.n_frames);
    // A regularisation corpus carries its own prompts, and they are not bounded
    // by the style corpus: a Lyric Studio MM3 caption runs ~1,100-1,300 tokens
    // against ~600 for a captioned album track. Sizing the graph off the style
    // corpus alone made every such reg sample "sequence exceeds S_max" at
    // capture time, and a reg step with no prior is a silent no-op (2026-09-07:
    // a 450-step run with 150 reg steps that trained exactly like the 300-step
    // baseline). Pre-scan the corpus here so the graph fits it; the samples are
    // loaded again where the priors are captured.
    if (a.reg_every > 0 && !a.reg_manifest.empty()) {
        std::vector<MM3LmSample> reg_probe;
        std::string              perr;
        if (mm3_lm_load_samples_from(a.reg_manifest, a.reg_captions_dir, a.reg_codes_dir, a.lm_path,
                                     /*trigger_prefix=*/"", /*caption_override=*/"", /*trim_trailing=*/false, t,
                                     &reg_probe, &perr)) {
            int64_t reg_max = 0;
            for (const auto & s : reg_probe) reg_max = std::max(reg_max, (int64_t) s.prompt.size());
            for (const auto & s : reg_probe) longest_track = std::max(longest_track, s.n_frames);
            fprintf(stderr, "[mm3-lm-train] reg corpus: %zu samples, longest prompt %lld tok (style %lld) - graph sized to fit both\n",
                    reg_probe.size(), (long long) reg_max, (long long) max_prompt);
            max_prompt = std::max(max_prompt, reg_max);
        } else {
            fprintf(stderr, "[mm3-lm-train] reg corpus pre-scan failed (%s); sizing off the style corpus only\n", perr.c_str());
        }
    }
    const int64_t K_ask = a.max_frames > 0 ? a.max_frames : 4096;
    const int64_t K_max = (longest_track > 0 && longest_track < K_ask) ? longest_track : K_ask;
    if (K_max != K_ask) {
        fprintf(stderr, "[mm3-lm-train] crop budget %lld frames clamped to the longest track (%lld frames, %.0f s): "
                        "every track trains whole and the buffers are sized to the album, not the ask\n",
                (long long) K_ask, (long long) longest_track, (double) longest_track / 25.0);
    }
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

    // ── --attn: resolve the mode, then PROVE the backend can run it ────────
    //
    // Resolved here because everything downstream depends on it: the mask's
    // dtype (F16 vs F32), the LmCkptCfg the segment graphs read, and the naive
    // graph's opts. The probe is asked at the run's real geometry (S_max, the
    // native GQA width) and BEFORE the first buffer, because a `false` from
    // ggml_backend_supports_op is not something this trainer would otherwise
    // notice: backend_sched_new registers the CPU backend alongside CUDA, so the
    // scheduler would quietly split the fused ops onto the CPU — Q/K/V and the
    // F16 mask over PCIe, 36 layers deep, every micro-step. Correct, unusably
    // slow, LOW on VRAM, i.e. indistinguishable from a pass on every number
    // this run reports. So: refuse, never fall back.
    //
    // ONE attention shape, as in train-lm: no cross-attention, so S_kv == S ==
    // S_max. Nkv is the NATIVE GQA width — lm_train_layer never expands heads on
    // the way to lm_attn_flash, so probing Nh would ask about a geometry that is
    // never built.
    const bool      attn_flash    = (a.attn == "flash" || a.attn == "flash-f32");
    const ggml_prec attn_prec_req = (a.attn == "flash-f32") ? GGML_PREC_F32 : GGML_PREC_DEFAULT;
    // A frozen KV prefix makes the attention rectangular: S_kv = n_pfx + S,
    // with the prompt's stored columns in front of the history. Until
    // 2026-09-06 this pair was refused; the fused op always took the shape
    // (mask [S_kv, >= S]), what was missing was probing the backend at the
    // real key length and giving the prefill an F16 mask. This is where flash
    // pays on MM3: at crop 750 the shipped recipe attends over ~6000 columns,
    // and the exact path materialises every [S, S_kv] score matrix.
    const int64_t S_kv_max = (a.prefix_frames > 0 ? max_prompt + a.prefix_frames : 0) + S_max;
    if (attn_flash) {
        const float ascale = 1.0f / sqrtf((float) c.head_dim);
        bool        pf = false, pb = false;
        dit_flash_probe(t.lm.backend, c.head_dim, c.n_heads, c.n_kv_heads, (int) S_max, (int) S_kv_max, /*B=*/1,
                        ascale, &pf, &pb);
        if (!(pf && pb)) {
            fprintf(stderr,
                    "[mm3-lm-train] --attn %s: backend %s does not support the fused attention ops at this "
                    "geometry (D %d, Nh %d, Nkv %d, S %lld, S_kv %lld, B 1) — fwd %s / bwd %s. Refusing to "
                    "start: the scheduler would silently run them on the CPU instead, which is correct, "
                    "unusably slow, and looks like a pass on every number this run reports. Use --attn exact.\n",
                    a.attn.c_str(), ggml_backend_name(t.lm.backend), c.head_dim, c.n_heads, c.n_kv_heads,
                    (long long) S_max, (long long) S_kv_max, pf ? "yes" : "NO", pb ? "yes" : "NO");
            jl("{\"type\":\"fatal\",\"message\":\"--attn %s unsupported by %s at D %d Nh %d Nkv %d S %lld\"}",
               a.attn.c_str(), ggml_backend_name(t.lm.backend), c.head_dim, c.n_heads, c.n_kv_heads,
               (long long) S_max);
            mm3_train_lm_free(&t);
            return 1;
        }
        fprintf(stderr,
                "[mm3-lm-train] --attn %s: %s supports FLASH_ATTN_TRAIN and FLASH_ATTN_TRAIN_BACK at D %d, "
                "Nh %d, Nkv %d, S %lld, S_kv %lld, B 1 — no CPU split. Requested arithmetic: %s (the backend "
                "resolves it per launch; the attn event after step 1 records what actually ran)\n",
                a.attn.c_str(), ggml_backend_name(t.lm.backend), c.head_dim, c.n_heads, c.n_kv_heads,
                (long long) S_max, (long long) S_kv_max,
                attn_prec_req == GGML_PREC_F32 ? "strict f32" : "tf32 where available");
    }
    // The REQUESTED arithmetic. GGML_PREC_DEFAULT is 0, which is also what a
    // zero-initialised op_params gives, and on sm_80+ it resolves to the TF32
    // kernels — so "flash" already means TF32 unless asked otherwise. What ran
    // is a separate event, emitted after step 1.
    jl("{\"type\":\"attn\",\"mode\":\"%s\",\"prec\":\"%s\"}", a.attn.c_str(),
       attn_flash ? (attn_prec_req == GGML_PREC_F32 ? "f32" : "tf32-where-available") : "n/a");

    // ── LoRA (attaches to the model) + optimizer ──
    LmLora     lora;
    const bool want_lokr = a.is_lokr();
    const bool init_ok =
        want_lokr
            ? lm_lokr_init(&lora, &t.lm, 0, c.n_layers, a.lokr_dim, a.lokr_alpha, a.lokr_factor,
                           a.lokr_decompose_both, (uint64_t) a.seed, &err)
            : lm_lora_init(&lora, &t.lm, 0, c.n_layers, a.rank, (float) a.alpha, (uint64_t) a.seed, 0.0f,
                           &err, LmLoraOpts{ a.dora, a.hira, a.loha, a.pissa, a.hra, a.hot_pizza, a.pissa_f16 });
    if (!init_ok) {
        fprintf(stderr, "[mm3-lm-train] %s init failed: %s\n", want_lokr ? "LoKr" : "LoRA", err.c_str());
        mm3_train_lm_free(&t);
        return 1;
    }
    // Identical to the fd-check site above, deliberately: the two are the one
    // place this trainer can silently validate a graph it does not train.
    if (!want_lokr && a.rslora) {
        lm_lora_apply_rslora(&lora);
        fprintf(stderr, "[mm3-lm-train] rsLoRA: in-graph scale alpha/sqrt(r) = %.4f (alpha/r would be %.4f)\n",
                (double) lora.scale, (double) (lora.alpha / (float) lora.rank));
        jl("{\"type\":\"adapter\",\"kind\":\"rslora\",\"scale\":%.6f}", (double) lora.scale);
    }
    if (!want_lokr && (a.dora || a.hira || a.loha || a.hra)) {
        const char * pm = a.hra ? "HRA" : a.dora ? "DoRA" : a.hira ? "HiRA" : "LoHa";
        fprintf(stderr, "[mm3-lm-train] parameterization: %s\n", pm);
        jl("{\"type\":\"adapter\",\"kind\":\"%s\"}",
           a.hra ? "hra" : a.dora ? "dora" : a.hira ? "hira" : "loha");
    }
    // PiSSA runs its SVD here, AFTER any rsLoRA rescale (the factors are stored
    // so that s*B0A0 is W's rank-r truncation, so they depend on the final s)
    // and BEFORE the optimizer and training graphs exist — the init holds a
    // ~200 MB scratch and gives it back before anything competes for VRAM.
    if (!want_lokr && a.pissa) {
        LmPissaStats ps;
        if (!lm_pissa_init_standalone(&lora, a.pissa_oversample, a.pissa_iters, &ps, &err, a.pissa_cache_dir, a.lm_path)) {
            fprintf(stderr, "[mm3-lm-train] PiSSA init failed: %s\n", err.c_str());
            lm_lora_detach(&lora, &t.lm);
            lm_lora_free(&lora);
            mm3_train_lm_free(&t);
            return 1;
        }
        fprintf(stderr,
                "[mm3-lm-train] PiSSA: %d sites, captured energy mean %.2f%% min %.2f%% (a random rank-%d "
                "subspace would hold ~%.2f%%)\n",
                ps.sites, 100.0 * ps.energy_mean, 100.0 * ps.energy_min, a.rank,
                100.0 * (double) a.rank / (double) c.hidden_size);
        jl("{\"type\":\"adapter\",\"kind\":\"%s\",\"sites\":%d,\"energyMean\":%.6f,"
           "\"energyMin\":%.6f,\"residual\":\"%s\"}",
           a.hot_pizza ? "hot-pizza" : "pissa", ps.sites, ps.energy_mean, ps.energy_min,
           json_escape(lora.pissa_residual).c_str());
        if (!lora.pissa_residual.empty()) {
            fprintf(stderr, "[mm3-lm-train] PiSSA export: adapter-only (delta) files against residual %s\n",
                    lora.pissa_residual.c_str());
        } else {
            fprintf(stderr, "[mm3-lm-train] PiSSA export: standalone rank-%d files (no residual beside %s)\n",
                    2 * a.rank, a.lm_path.c_str());
        }
    }
    if (want_lokr) {
        fprintf(stderr, "[mm3-lm-train] LoKr: dim %d alpha %.0f factor %d, decompose %s\n", a.lokr_dim,
                (double) a.lokr_alpha, a.lokr_factor, a.lokr_decompose_both ? "both" : "w1-only");
        jl("{\"type\":\"adapter\",\"kind\":\"lokr\",\"dim\":%d,\"alpha\":%.0f,\"factor\":%d}", a.lokr_dim,
           (double) a.lokr_alpha, a.lokr_factor);
    }
    // ── Artist token parameter ────────────────────────────────────────────
    //
    // [H, k] F32, zero-init: zero means "behave exactly like the placeholder
    // token", so step 0 is a no-op and a run that never moved is visibly
    // distinct from one that moved wrongly.
    ggml_tensor *         t_art   = nullptr;
    ggml_context *        art_ctx = nullptr;
    ggml_backend_buffer_t art_buf = nullptr;
    if (art_k > 0) {
        ggml_init_params ap = { 2 * ggml_tensor_overhead(), nullptr, true };
        art_ctx             = ggml_init(ap);
        if (!art_ctx) {
            fprintf(stderr, "[mm3-lm-train] cannot create the artist-token context\n");
            lm_lora_detach(&lora, &t.lm);
            lm_lora_free(&lora);
            mm3_train_lm_free(&t);
            return 1;
        }
        t_art = ggml_new_tensor_2d(art_ctx, GGML_TYPE_F32, t.lm.cfg.hidden_size, art_k);
        ggml_set_name(t_art, "artist_token");
        ggml_set_param(t_art);
        art_buf = ggml_backend_alloc_ctx_tensors(art_ctx, t.lm.backend);
        if (!art_buf) {
            fprintf(stderr, "[mm3-lm-train] artist-token buffer allocation failed\n");
            ggml_free(art_ctx);
            lm_lora_detach(&lora, &t.lm);
            lm_lora_free(&lora);
            mm3_train_lm_free(&t);
            return 1;
        }
        std::vector<float> z((size_t) ggml_nelements(t_art), 0.0f);
        ggml_backend_tensor_set(t_art, z.data(), 0, z.size() * sizeof(float));
    }

    // ── Trainable KV prefix (train/lm-prefix.h) ───────────────────────────
    //
    // Allocated over EVERY layer, because lm_ckpt_layer_kv indexes the tables
    // by absolute layer and a null entry at a layer the segment builds would
    // dereference. Init sigma is on both K and V for the reason lm-prefix.h
    // states: a zero V pins dL/dK at exactly zero forever.
    LmPrefix pfx;
    if (a.prefix_n > 0) {
        std::string perr;
        if (!lm_prefix_init(&pfx, &t.lm, 0, c.n_layers, a.prefix_n, S_max, (uint64_t) a.seed, a.prefix_sigma,
                            &perr)) {
            fprintf(stderr, "[mm3-lm-train] trainable prefix init failed: %s\n", perr.c_str());
            lm_lora_detach(&lora, &t.lm);
            lm_lora_free(&lora);
            mm3_train_lm_free(&t);
            return 1;
        }
        fprintf(stderr,
                "[mm3-lm-train] trainable KV prefix: n=%d over %d layers, row %lld, %zu params (%.1f MB), "
                "sigma %.3g\n",
                pfx.n, c.n_layers, (long long) pfx.row, pfx.n_params, pfx.n_params * 4.0 / 1048576.0,
                (double) a.prefix_sigma);
        jl("{\"type\":\"prefixTune\",\"n\":%d,\"params\":%zu}", pfx.n, pfx.n_params);
    }

    // Freezing the adapter means CLEARING GGML_TENSOR_FLAG_PARAM, not merely
    // omitting it here: lm_ckpt_fill_gacc asserts every PARAM-flagged graph node
    // has an optimizer slot, so flagged-but-unoptimized aborts the run.
    std::vector<ggml_tensor *> train_params;
    if (art_k > 0 && a.artist_only) {
        for (size_t i = 0; i < lora.params.size(); i++) {
            lora.params[i]->flags &= ~(int32_t) GGML_TENSOR_FLAG_PARAM;
        }
    } else {
        train_params = lora.params;
    }
    // The trained parameters that are NOT LoRA factors. mm3-lm-resume.h saves
    // and restores `lora.params` by name, so without this list a paused run
    // came back with a zero artist token and a re-seeded prefix — carrying the
    // Adam momentum of the ones it had thrown away, because the momentum lives
    // in the optimizer and WAS being saved.
    std::vector<ggml_tensor *> soft_params;
    if (t_art) {
        train_params.push_back(t_art);
        soft_params.push_back(t_art);
    }
    for (size_t i = 0; i < pfx.params.size(); i++) {
        train_params.push_back(pfx.params[i]);
        soft_params.push_back(pfx.params[i]);
    }

    LmOptim opt;
    opt.optimizer     = a.optimizer;
    opt.muon.lr_scale = a.muon_lr_scale;
    opt.muon.momentum = a.muon_momentum;
    opt.muon.ns_steps = a.muon_ns_steps;
    opt.muon.nesterov = a.muon_nesterov;
    opt.muon.min_dim  = a.muon_min_dim;
    opt.muon.bucket   = a.muon_bucket;
    if (t_art) {
        opt.adamw_only.push_back(t_art);  // a [H, k] matrix is not a weight; keep it off Muon
    }
    for (size_t i = 0; i < pfx.params.size(); i++) {
        opt.adamw_only.push_back(pfx.params[i]);  // same reasoning: K/V columns are not a weight matrix
    }
    if (!lm_optim_init(&opt, train_params, t.lm.backend, &err)) {
        fprintf(stderr, "[mm3-lm-train] optimizer init failed: %s\n", err.c_str());
        lm_lora_detach(&lora, &t.lm);
        lm_lora_free(&lora);
        mm3_train_lm_free(&t);
        return 1;
    }
    // A run where Muon classified ZERO parameters trains as AdamW and says
    // nothing about it. Print the split so that is visible.
    if (a.artist_lr > 0.0f && a.artist_lr != a.lr) {
        // One dial for BOTH soft-prompt parameterizations, exactly as train-lm
        // does it: the token and the prefix are the same kind of thing (a
        // conditioning input, not a weight) and want the same larger LR.
        const float mul = a.artist_lr / a.lr;
        size_t      n   = 0;
        if (t_art) {
            lm_optim_set_lr_mul(&opt, t_art, mul);
            n++;
        }
        for (size_t i = 0; i < pfx.params.size(); i++) {
            lm_optim_set_lr_mul(&opt, pfx.params[i], mul);
            n++;
        }
        if (n) {
            fprintf(stderr, "[mm3-lm-train] soft-prompt LR %.3g (x%.1f the LoRA's %.3g) on %zu tensor(s)\n",
                    a.artist_lr, mul, a.lr, n);
        }
    }
    if (a.lora_plus_ratio != 1.0f && !lora.is_lokr) {
        int n_b = 0, n_muon_b = 0;
        for (int l = lora.layer_lo; l < lora.layer_hi; l++) {
            for (int sl = 0; sl < QW_LORA_NSLOTS; sl++) {
                ggml_tensor * B = lora.layers[l].p[sl].B;
                if (!B) {
                    continue;
                }
                auto it = opt.param_slot.find(B);
                if (it == opt.param_slot.end()) {
                    continue;
                }
                if (opt.rule[(size_t) it->second] == LM_RULE_MUON) {
                    n_muon_b++;
                    continue;
                }
                lm_optim_set_lr_mul(&opt, B, a.lora_plus_ratio);
                n_b++;
            }
        }
        fprintf(stderr, "[mm3-lm-train] LoRA+: B at x%.1f the base LR on %d tensors%s", a.lora_plus_ratio, n_b,
                n_muon_b ? " (Muon-rule B tensors ignore it; this trainer defaults to Muon)" : "");
        fputc(10, stderr);
    }
    // Counts come from the OPTIMIZER, not from lora.params. Those were the same
    // number until --artist-token-only made it possible to build the LoRA and
    // then not train it: reporting the tensor count would have said "504 on
    // AdamW" for a run whose optimizer holds exactly one parameter.
    fprintf(stderr,
            "[mm3-lm-train] %zu LoRA tensors (rank %d, alpha %d) — optimizing %zu: %d on Muon in %zu buckets, %zu on "
            "AdamW\n",
            lora.params.size(), a.rank, a.alpha, opt.params.size(), opt.n_muon, opt.muon_buckets.size(),
            opt.params.size() - (size_t) opt.n_muon);
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
    // A TRAINABLE prefix widens the same mask by its own n (lm_ckpt_n_kv), and
    // the two are mutually exclusive, so one max covers both.
    const int64_t MSK_CAP = (std::max<int64_t>(PFX_Q, a.prefix_n) + S_max) * S_max;
    ggml_tensor * t_pos    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, S_max);
    // F32 under --attn exact — the shipped allocation, byte for byte — and F16
    // under flash, where ggml_flash_attn_train asserts mask->type == F16. The
    // flag refuses --prefix-frames, so PFX_Q is 0 whenever this is F16 and
    // MSK_CAP collapses to the square S_max*S_max the probe above covered.
    ggml_tensor * t_msk    = lm_mask_alloc(ctx_static, MSK_CAP, attn_flash);
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
        if (kv_on && !lm_kvprefix_alloc(&kvpfx, &t.lm, 0, c.n_layers, PFX_Q, S_max, a.prefix_chunk, &err, attn_flash)) {
            fprintf(stderr, "[mm3-lm-train] kv prefix setup failed: %s\n", err.c_str());
            lm_lora_detach(&lora, &t.lm);
            lm_lora_free(&lora);
            mm3_train_lm_free(&t);
            return 1;
        }
        LmCkptCfg cc;
        cc.kv           = kv_on ? &kvpfx : nullptr;
        cc.pfx          = pfx.active() ? &pfx : nullptr;
        cc.chunk        = a.ckpt_chunk;
        cc.weights_bf16 = weights_bf16;                 // Lever A
        // lm_ckpt_layer_opts is the ONLY place the segment graphs get their
        // options, so P2 (forward collect), P3 (tail) and P7 (backward) cannot
        // disagree about the attention formulation — which is what D13's
        // "recompute must match collect" already depended on.
        cc.attn_flash   = attn_flash;
        cc.attn_prec    = attn_prec_req;
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
        embed_ctx.t_art    = t_art;
        embed_ctx.art_k    = art_k;

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
        // P1 is forward-only and grads=false unless told otherwise, and its
        // header states the embed_build override "must be frozen and
        // gradient-free". An artist token breaks exactly that contract, so the
        // driver has to run its P1B backward or the vectors get zero gradient
        // while the adapter's loss falls convincingly.
        ckpt_run.embed_trainable = (t_art != nullptr);
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

    // ── the artist token is a PER-SAMPLE property, not a run-wide one ───────
    //
    // A prior-preservation sample is loaded from a different manifest and never
    // goes through the placeholder splice above, so its prompt does not contain
    // the token's span. The trainer used to hand the SAME t_art / art_k into
    // every graph regardless, which accs an artist-specific vector onto the
    // first k positions of prompts that have nothing to do with the artist —
    // corrupting the very steps that exist to hold the base model still — and
    // ran a P1B backward into t_art on those steps too, so the token also
    // learned from them.
    //
    // train-lm has carried the right shape since the token landed there
    // (`run.embed_has_param = s.artist_off >= 0`). This is the same switch:
    // art_k == 0 removes the acc from the graph, and embed_has_param stops
    // lm-ckpt.h building a backward over a stage that now holds no parameter
    // (ggml_build_backward_expand asserts on a graph with none).
    int  art_k_step = art_k;
    auto set_art    = [&](bool on) {
        art_k_step               = on ? art_k : 0;
        embed_ctx.art_k          = art_k_step;
        pfx_ctx.e.art_k          = art_k_step;
        ckpt_run.embed_has_param = art_k_step > 0;
    };
    set_art(art_k > 0);

    // ── graph sizing + scheduler ──
    // The scheduler is SHARED with the optimizer step, so it must be sized for
    // whichever graph is larger. This bit a real 4B Muon run: Muon's optimizer
    // graph is ~7-9k nodes while a segmented training graph was ~569, and ggml
    // asserts hash_set.size >= n_nodes + n_leafs mid-run. Do not "simplify".
    // Same rank-proportional term as the FD arm above: the naive (non-`--ckpt`)
    // graphs build every layer, so their budget follows the parameterization.
    // The checkpointed path sizes its own segment graphs in lm-ckpt.h.
    const int naive_nodes = 65536 + LM_GRAPH_BWD_NODE_MULT * lm_trunk_extra_nodes(&t.lm, 0, c.n_layers);
    std::vector<uint8_t> arena(std::max<size_t>((size_t) 512 << 20, lm_graph_arena_bytes(naive_nodes)));
    int                  graph_nodes = 0;

    auto build_graph = [&](ggml_context * ctx, ggml_cgraph * gf, int64_t P, int64_t Fin, int64_t n_sup,
                           ggml_tensor ** out_loss) {
        const int64_t S = P + Fin;
        MM3EmbedCtx   ec{ &t, t_prompt, t_sem, t_ac, P, Fin, t_art, art_k_step };
        ggml_tensor * h_in = mm3_lm_build_embed(ctx, ec);

        LmLayerOpts   nopts;
        nopts.attn_flash = attn_flash;
        nopts.attn_prec  = attn_prec_req;
        ggml_tensor * hidden =
            lm_build_trunk_embeds(ctx, &t.lm, h_in, t_pos, t_msk, (int) S, 0, c.n_layers, nopts);
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
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, naive_nodes, true);
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

    // ── --hra: refuse a rank that cannot start, rather than crash on it ─────
    //
    // HRA is the one parameterization whose ACTIVATION cost scales with --rank,
    // and at the default rank of 64 it does not fit a 32 GB card at the shipped
    // crop: the allocator asks for a 30 GiB compute buffer, cudaMalloc says no,
    // and ggml_gallocr's failure path takes the process down with a segfault
    // three lines after saying so. There is nothing to diagnose in that, and
    // nothing the caller can catch, so the arithmetic happens here instead.
    //
    // Deliberately compared against the terms we KNOW — the reflection
    // activations, the checkpoint state and the adapter's own four buffers
    // (weights, two AdamW moments, gradients). The baseline compute buffer is
    // left out, which makes this an UNDER-estimate on purpose: it refuses only
    // what could not have started under any accounting, and never a
    // configuration that would have run.
    if (lora.hra) {
        size_t vfree = 0, vtotal = 0;
        lm_vram_query(t.lm.backend, &vfree, &vtotal);
        const size_t hra_b  = lm_hra_segment_bytes(&t.lm, 0, c.n_layers, S_max, /*whole_trunk=*/!a.ckpt);
        const size_t ckpt_b = a.ckpt ? ckpt_st.fixed_bytes() : 0;
        const size_t par_b  = 4 * lora.n_params * sizeof(float);
        const double gib    = 1024.0 * 1024.0 * 1024.0;
        if (vfree > 0 && (double) (hra_b + ckpt_b + par_b) > (double) vfree) {
            fprintf(stderr,
                    "[mm3-lm-train] --hra rank %d does not fit at --max-frames %lld (sequence %d).\n"
                    "  The %s retains ~%.1f GiB of reflection activations (r x sum(in) x S x 4,\n"
                    "  every reflection's running x, broadcast v and correction are backward inputs), and with\n"
                    "  %.1f GiB of checkpoint state and %.1f GiB of adapter + AdamW + gradient buffers that is\n"
                    "  %.1f GiB against %.1f GiB free — before the baseline compute buffer, which is not counted.\n"
                    "  Reference, 36-layer MM3 LM on a q8_0 base at crop 750: rank 8 peaks at 17.6 GiB, rank 32\n"
                    "  at 30.0 GiB, and rank 64 asks the allocator for 29.4 GiB of compute buffer alone. Lower\n"
                    "  --rank or --max-frames. Refusing here rather than letting the allocator fail, which\n"
                    "  segfaults inside ggml_gallocr with nothing to diagnose.\n",
                    a.rank, (long long) a.max_frames, (int) S_max,
                    a.ckpt ? "worst backward segment" : "whole-trunk graph", (double) hra_b / gib,
                    (double) ckpt_b / gib, (double) par_b / gib, (double) (hra_b + ckpt_b + par_b) / gib,
                    (double) vfree / gib);
            jl("{\"type\":\"fatal\",\"message\":\"--hra rank %d does not fit at max-frames %lld: needs at least "
               "%.1f GiB, %.1f GiB free\"}",
               a.rank, (long long) a.max_frames, (double) (hra_b + ckpt_b + par_b) / gib, (double) vfree / gib);
            lm_lora_detach(&lora, &t.lm);
            lm_lora_free(&lora);
            lm_ckpt_free(&ckpt_st);
            lm_optim_free(&opt);
            mm3_train_lm_free(&t);
            return 1;
        }
        fprintf(stderr, "[mm3-lm-train] --hra rank %d: ~%.1f GiB of reflection activations in the %s "
                        "at sequence %d, %.1f GiB free\n",
                a.rank, (double) hra_b / gib, a.ckpt ? "worst backward segment" : "whole-trunk graph",
                (int) S_max, (double) vfree / gib);
    }

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
        // The two hard requirements — the checkpointed path (the naive path has
        // nowhere to splice a store into) and `song` anchoring (a history at
        // positions the window then re-uses is a contradiction) — are refused by
        // cmd_mm3_lm_train from the flags alone, before the base load. Asserted
        // here so a future caller that skips the parser is loud rather than
        // subtly wrong.
        GGML_ASSERT(a.ckpt && anchor_song && "--prefix-frames needs --crop-anchor song on the ckpt path");
        fprintf(stderr,
                "[mm3-lm-train] kv prefix: up to %lld frames (%.1f s) of no-grad history in front of each crop\n",
                (long long) a.prefix_frames, (double) a.prefix_frames / 25.0);
        jl("{\"type\":\"kvPrefix\",\"frames\":%lld,\"chunk\":%d}", (long long) a.prefix_frames,
           a.prefix_chunk);
    }
    // Same as the frozen prefix above: refused in cmd_mm3_lm_train, asserted
    // here. The old refusal sat at this point in the run body, which meant an
    // 8.5 GB base load before the message and a `return 1` that freed the prefix
    // and nothing else.
    GGML_ASSERT(!(pfx.active() && !a.ckpt) && "--prefix-n needs the checkpointed path");
    if (a.crop_mode == "structured") {
        fprintf(stderr,
                "[mm3-lm-train] crop policy: structured - %.0f%% start share (half at frame 0, "
                "half over %d aligned tiles), %.0f%% flush to the end (EOS), %.0f%% random\n",
                a.crop_start_frac * 100.0, a.crop_start_tiles, a.crop_end_frac * 100.0,
                (1.0 - a.crop_start_frac - a.crop_end_frac) * 100.0);
        if (a.end_crop_vary) {
            fprintf(stderr, "[mm3-lm-train] varied end supervision: end crops draw their length in [%lld, K] and "
                            "their history in [0, %lld] frames (lever 4a)\n",
                    (long long) a.end_crop_min, (long long) a.prefix_frames);
        }
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
    int                  n_dropped   = 0;
   // caption-dropout steps, THIS SEGMENT
    int64_t n_end_vary = 0;   // lever 4a: end steps that drew a varied window/history
    bool    reg_span_logged = false;   // rev-7 ending-targeted prior: announce the scored span once
    int64_t n_lyrics_dropped = 0;      // lyrics dropout: style steps trained without lyrics
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
        // The three facts a loader cannot infer from the tensors: rsLoRA's
        // scale rule, DoRA's magnitude rescale, and which parameterization the
        // hada_w*/lora_* keys belong to. Dropping any of them here is the
        // turbo8 no-op trap in reverse — the file would load and mean something
        // else.
        meta.rslora       = a.rslora;
        // Provenance for the soft-prompt halves, so the config says what was
        // trained even though the tensors are self-describing.
        meta.artist_token = a.artist_token;
        meta.artist_k     = art_k;
        meta.artist_lr    = a.artist_lr;
        meta.prefix_n     = pfx.active() ? pfx.n : 0;
        meta.param_method = a.hra    ? "hra"
                            : a.pissa ? (a.hot_pizza ? "hot-pizza" : "pissa")
                            : a.hira  ? "hira"
                            : a.loha  ? "loha"
                            : a.dora  ? "dora"
                                      : "lora";
        meta.adapter_type = a.adapter_type;
        LmExportResult res;
        std::string    xerr;
        // LoKr writes lokr_weights.safetensors; LoRA writes a PEFT directory
        // (adapter_config.json + adapter_model.safetensors). Same ckpt-<step>/
        // directory, different file name — the sidecar below has to follow, and
        // so does every consumer that hard-codes adapter_model.safetensors.
        // Soft-prompt half, inside the same safetensors as the LoRA (lm-export.h).
        LmExtraExport mx;
        if (t_art) {
            mx.art_t           = t_art;
            mx.art_k           = art_k;
            mx.art_placeholder = art_placeholder;
            mx.site            = 2;  // mm3_lm
        }
        if (pfx.active()) {
            mx.pfx_k  = pfx.k;
            mx.pfx_v  = pfx.v;
            mx.pfx_n  = pfx.n;
            mx.pfx_lo = pfx.layer_lo;
            mx.pfx_hi = pfx.layer_hi;
        }
        const bool lokr_out = a.is_lokr();
        // PiSSA and HRA go out as ORDINARY PEFT LoRAs — rank 2r and rank r
        // respectively — through the same writer, with their factors
        // substituted. That is what lets the runtime and merge paths stay
        // completely unaware of either.
        // PiSSA (2026-09-09): the DELTA form when the init read or wrote the
        // residual beside the base (adapter-only, rank r), else the standalone
        // rank-2r form. Both F16 now.
        const bool exported = lokr_out  ? lm_export_lokr(lora, meta, dir, &res, &xerr)
                              : a.hra   ? lm_export_hra(lora, c, meta, dir, sched, &res, &xerr, &mx)
                              : a.pissa ? (lora.pissa_residual.empty()
                                               ? lm_export_pissa(lora, c, meta, dir, &res, &xerr, &mx)
                                               : lm_export_pissa_delta(lora, c, meta, dir, &res, &xerr, &mx))
                                        : lm_export_peft(lora, c, meta, dir, &res, &xerr, &mx);
        if (!exported) {
            fprintf(stderr, "[mm3-lm-train] export failed: %s\n", xerr.c_str());
            return std::string();
        }
        // ── the legacy artist_token.safetensors sidecar ────────────────────
        //
        // REDUNDANT SINCE 2026-09-05 and kept for ONE release. The token now
        // ships inside adapter_model.safetensors as hot_step.artist_token.vec /
        // .meta (site 2, lm-export.h) and minimax/mm3-lm-adapter.h reads it from
        // there — that is the only path generation uses. This pair of files is
        // written purely so a checkpoint produced by this build still loads in a
        // build that predates the unified reader. Delete the block, and
        // artist-token-io.h's mm3 caller, once no shipped release reads it.
        if (t_art) {
            ArtistTokenMeta am;
            am.name        = a.artist_token;
            am.site        = "mm3_lm";
            am.base_model  = a.lm_path;
            am.k           = art_k;
            am.placeholder = art_placeholder;
            std::string aerr;
            if (!artist_token_write(t_art, am, dir, &aerr)) {
                fprintf(stderr, "[mm3-lm-train] artist token export failed: %s\n", aerr.c_str());
                return std::string();
            }
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
            // `triggerPrepend` is not decoration: the server auto-prepends the
            // trigger to render captions, and it must not do that for a run
            // that recorded a trigger it never trained (see --trigger-prepend
            // above). Sidecars written before this field existed are read as
            // trained, which is right for every run the flag defaulted on for.
            fprintf(sf,
                    "{\"name\":\"%s ckpt-%d\",\"trigger\":\"%s\",\"triggerPrepend\":%s,"
                    "\"rank\":%d,\"dataset\":\"%s\","
                    "\"trainedSteps\":%d,\"recommendedScales\":{\"scaleMlp\":%.1f},"
                    "\"notes\":\"ace-train mm3-lm-train, loss %.4f; render captions must carry the "
                    "artist's true bpm/tuning\"}\n",
                    a.dataset_name.empty() ? "MM3 LM" : a.dataset_name.c_str(), step, a.trigger.c_str(),
                    a.trigger_prepend ? "true" : "false", a.rank,
                    a.dataset_name.c_str(), step, a.depth_loss_weight > 0.0 ? 1.0 : 0.5, loss);
            fclose(sf);
        }
        // Export -> load round trip against the RUNTIME loader. A file that
        // exports cleanly and loads as something else is the failure mode this
        // whole parameterization family invites (rsLoRA's scale, DoRA's
        // magnitudes, LoHa's key layout), and nothing else in the run would
        // catch it.
        if (a.verify_export && !lokr_out) {
            std::string verr;
            const bool  vok = (a.pissa || a.hra) ? mm3_lm_verify_export_delta(dir, lora, &verr, a.lm_path)
                                                 : mm3_lm_verify_export(dir, lora, &verr, a.lm_path);
            if (!vok) {
                fprintf(stderr, "[mm3-lm-train] --verify-export FAILED: %s\n", verr.c_str());
                return std::string();
            }
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
            // Skipped entirely under a trainable prefix: lm_ckpt_upload_mask
            // owns t_msk then (the mask is rectangular, [n + S, S]), and a
            // square upload here would be silently reinstated over it.
            if (!pfx.active() && (int) S != last_mask_S) {
                lm_causal_mask((int) S, &msk);
                lm_mask_set(t_msk, msk);   // F16 under --attn flash, same bytes under exact
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
            // Held-out crops come from the same manifest as the training ones,
            // so they DO carry the placeholder span. Set explicitly rather than
            // inherited: the last micro-step before an eval may have been a reg
            // step, which leaves the token off.
            set_art(art_k > 0);
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
    // Adapter identity. Same string the export writes as `param_method`, so a
    // state file and the checkpoint beside it always agree about what they are.
    rstate.param_method   = a.hra     ? "hra"
                            : a.pissa ? (a.hot_pizza ? "hot-pizza" : "pissa")
                            : a.hira  ? "hira"
                            : a.loha  ? "loha"
                            : a.dora  ? "dora"
                                      : "lora";
    rstate.rslora         = a.rslora ? 1 : 0;

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
        if (!mm3_lm_resume_load(a.resume_path, &rstate, lora, opt, &rerr, soft_params)) {
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
                                      /*trigger_prefix=*/"", /*caption_override=*/"", /*trim_trailing=*/false, t,
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
            // The window. An excerpt no longer than K is rehearsed whole from
            // its first frame. An excerpt LONGER than K carries history: its
            // last K frames are the window and the frames before them go into
            // the frozen KV prefix, exactly as a style crop's history does
            // (Phase 4 step 2b, 2026-09-07). Either way the window reaches the
            // excerpt's end, so the base's stopping decision is the last
            // supervised position. Same arithmetic as the reg step below.
            const int64_t K      = std::min<int64_t>(K_max, rs.n_frames);
            const int64_t c0     = rs.n_frames - K;              // 0 for a history-free excerpt
            const bool    at_end = true;                         // c0 + K == n_frames
            const int64_t lead   = (kv_on && c0 > 0) ? 1 : 0;   // frame c0-1 as input, as in a style crop
            const int64_t f0     = c0 - lead;
            const int64_t Fin    = K + lead;                     // input frames, lead included
            const int64_t n_sup  = K + 1;
            const int64_t P      = (int64_t) rs.prompt.size();
            const int64_t S      = P + Fin;
            const int64_t npfx   = kv_on ? f0 - std::max<int64_t>(0, f0 - a.prefix_frames) : 0;
            if (S > S_max || K < 1) {
                fprintf(stderr, "[mm3-lm-train] SKIP reg %s: sequence %lld exceeds %lld\n",
                        rs.id.c_str(), (long long) S, (long long) S_max);
                continue;
            }
            // An excerpt's teacher was captured at ITS positions; a cache for the
            // same id at another offset is a different teacher, so the offset is
            // in the name like the base model and K already are.
            std::string cache_id = rs.id;
            if (rs.frame_offset) cache_id += ".o" + std::to_string((long long) rs.frame_offset);
            // A windowed excerpt's teacher saw a prefix; a cache captured with
            // another window or history length is a different teacher.
            if (c0 > 0) cache_id += ".c" + std::to_string((long long) c0) + ".p" + std::to_string((long long) npfx);
            const std::string path = mm3_prior_path(prior_dir, cache_id, a.lm_path, a.reg_topk);
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
                const int32_t * f = &rs.codes[(size_t) ((f0 + j) * 8)];
                sem_in[(size_t) j] = f[0] + (int32_t) t.semantic_vocab_offset;
                for (int64_t k2 = 0; k2 < NC; k2++) {
                    ac_in[(size_t) (k2 * Fin + j)] = f[1 + k2] + (int32_t) (k2 * AV);
                }
            }
            tgt.resize((size_t) n_sup);
            for (int64_t j = 0; j < n_sup; j++) {
                tgt[(size_t) j] = (at_end && j == n_sup - 1)
                                    ? mm3_lm_train_slice_eos(t)
                                    : mm3_lm_train_slice_index(t, rs.codes[(size_t) ((c0 + j) * 8)]);
            }
            // Skipped entirely under a trainable prefix: lm_ckpt_upload_mask
            // owns t_msk then (the mask is rectangular, [n + S, S]), and a
            // square upload here would be silently reinstated over it.
            if (!pfx.active() && (int) S != last_mask_S) {
                lm_causal_mask((int) S, &msk);
                lm_mask_set(t_msk, msk);   // F16 under --attn flash, same bytes under exact
                last_mask_S = (int) S;
            }
            pos.resize((size_t) S);
            for (int64_t j = 0; j < P; j++)   pos[(size_t) j] = (int32_t) j;
            // Excerpt corpora carry their true start; under --crop-anchor song the
            // teacher sees the frames where the base saw them (frame_offset is 0
            // for ordinary corpora, so this is the old P + j there).
            const int64_t cap_off = anchor_song ? rs.frame_offset : 0;
            for (int64_t j = 0; j < Fin; j++) pos[(size_t) (P + j)] = (int32_t) (P + cap_off + f0 + j);
            ggml_backend_tensor_set(t_prompt, rs.prompt.data(), 0, (size_t) P * sizeof(int32_t));
            ggml_backend_tensor_set(t_sem, sem_in.data(), 0, sem_in.size() * sizeof(int32_t));
            ggml_backend_tensor_set(t_ac, ac_in.data(), 0, ac_in.size() * sizeof(int32_t));
            ggml_backend_tensor_set(t_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
            embed_ctx.P = P; embed_ctx.Fin = Fin;
            // A reg prompt carries no placeholder span, so the token is off for
            // the capture too — the teacher distribution has to be the one the
            // reg steps below will be scored against, and those run token-free.
            set_art(false);
            set_rank_mask(false);  // the teacher is the base, not a subnetwork (prefix included)
            // The frozen prefix for a history-bearing excerpt, built the way the
            // reg step will build it. Q == 0 still runs when the store is on:
            // it clears K/V a longer previous capture left behind.
            if (kv_on) {
                int64_t Q = 0;
                if (c0 > 0) {
                    const int64_t pfx_lo = f0 - npfx;
                    pfx_ctx.prompt       = rs.prompt.data();
                    pfx_ctx.codes        = rs.codes.data();
                    pfx_ctx.P            = P;
                    pfx_ctx.pfx_lo       = pfx_lo;
                    Q                    = P + npfx;
                    pfx_pos.resize((size_t) Q);
                    for (int64_t i2 = 0; i2 < P; i2++)    pfx_pos[(size_t) i2] = (int32_t) i2;
                    for (int64_t i2 = 0; i2 < npfx; i2++) pfx_pos[(size_t) (P + i2)] = (int32_t) (P + cap_off + pfx_lo + i2);
                }
                std::string perr;
                if (!lm_kvprefix_run(&kvpfx, &t.lm, sched, arena, lm_ckpt_layer_opts(ckpt_st),
                                     mm3_lm_prefix_embed, &pfx_ctx, pfx_pos.data(), Q, &perr)) {
                    fatal_msg = "prior capture prefix failed for " + rs.id + ": " + perr;
                    fprintf(stderr, "[mm3-lm-train] %s\n", fatal_msg.c_str());
                    rc = 1;
                    break;
                }
            }

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
            smp.n_masked = (int) (P + lead);
            smp.n_prompt = (int) P;
            smp.s_tr     = (int) n_sup;
            const bool ok = lm_ckpt_micro_step(ckpt_run, smp, false, nullptr);
            ckpt_run.capture_k   = 0;
            ckpt_run.capture_idx = nullptr;
            ckpt_run.capture_p   = nullptr;
            ckpt_run.forward_only = false;
            set_art(art_k > 0);
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
            const bool drop_lyrics =
                !is_reg && !drop_caption && a.lyrics_dropout > 0.0 && !s.prompt_no_lyrics.empty()
                && lm_rng_uniform(&rng) < (float) a.lyrics_dropout;
            const std::vector<int32_t> & prompt_ids =
                drop_caption ? s.prompt_trigger_only : (drop_lyrics ? s.prompt_no_lyrics : s.prompt);
            if (drop_lyrics) n_lyrics_dropped++;
            const int64_t       P = (int64_t) prompt_ids.size();

            // Fresh crop every time this song comes up. `beginning` exists only
            // to reproduce the intros-only failure lm2 hit. A reg step is ALWAYS
            // deterministic from the start — its cached teacher was captured on
            // exactly that crop and no other.
            int64_t K = std::min<int64_t>(K_max, s.n_frames);
            int64_t c0 = 0;
            int64_t pfx_span = a.prefix_frames;   // history in front of the window; lever 4a shortens it on end steps
            // A history-bearing reg excerpt (longer than K) is windowed flush to
            // its end and its history becomes the prefix — the capture above
            // used the same window, so the cached teacher matches.
            if (is_reg && s.n_frames > K) c0 = s.n_frames - K;
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
                        if (a.end_crop_vary) {
                            // Lever 4a: a shorter window still flush to the
                            // end, and a random slice of history in front of
                            // it. Extra RNG draws ride the resume state.
                            const int64_t k_lo = std::max<int64_t>(1, std::min<int64_t>(a.end_crop_min, K));
                            K = k_lo + (int64_t) (lm_rng_next(&rng) % (uint64_t) (K - k_lo + 1));
                            if (kv_on && a.prefix_frames > 0) {
                                pfx_span = (int64_t) (lm_rng_next(&rng) % (uint64_t) (a.prefix_frames + 1));
                            }
                            n_end_vary++;
                        }
                        c0 = s.n_frames - K;                  // flush to the end: EOS
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
            const int64_t lead   = (kv_on && c0 > 0) ? 1 : 0;   // a reg step: only with a windowed excerpt
            const int64_t f0     = c0 - lead;               // first INPUT frame
            const int64_t Finw   = Fin + lead;
            const int64_t S      = P + Finw;
            const int64_t anchor0 = anchor_song ? (s.frame_offset + f0) : 0;   // frame_offset: excerpt corpora only

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

            // Skipped entirely under a trainable prefix: lm_ckpt_upload_mask
            // owns t_msk then (the mask is rectangular, [n + S, S]), and a
            // square upload here would be silently reinstated over it.
            if (!pfx.active() && (int) S != last_mask_S) {
                lm_causal_mask((int) S, &msk);
                lm_mask_set(t_msk, msk);   // F16 under --attn flash, same bytes under exact
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
            // A regularisation step gets the prefix its teacher was captured
            // with: none for a history-free excerpt, the excerpt's own history
            // for a windowed one. Scoring through any other context would be
            // measuring drift that is not there.
            if (kv_on) {
                int64_t Q = 0;
                if (!is_reg || c0 > 0) {
                    // History runs up to f0, because frame f0 itself is an
                    // input to the window (see `lead`).
                    const int64_t pfx_lo = std::max<int64_t>(0, f0 - pfx_span);
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
                    // Excerpt corpora carry their true start (frame_offset is 0
                    // for ordinary corpora): the history sits where the base saw it.
                    const int64_t pfx_off = anchor_song ? s.frame_offset : 0;
                    for (int64_t i = 0; i < npfx; i++) {
                        pfx_pos[(size_t) (P + i)] = (int32_t) (P + pfx_off + pfx_lo + i);
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
            // The token is on for a style step and off for a reg step — see the
            // set_art definition. Set here, per micro-step, so the two kinds of
            // step can alternate inside one grad_accum group.
            set_art(art_k > 0 && !is_reg);
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
                if (!prior && a.score_last > 0 && (int) n_sup > a.score_last &&
                    (!a.score_last_end_only || at_end)) {
                    // --score-last: the earlier rows stay as input context and
                    // leave the loss; same column arithmetic as the head.
                    const int skip = (int) n_sup - a.score_last;
                    smp.n_masked += skip;
                    smp.col_skip  = skip;   // the depth loss indexes by crop frame
                    smp.s_tr      = a.score_last;
                    smp.targets.assign(tgt.begin() + skip, tgt.begin() + skip + a.score_last);
                }
                if (prior) {
                    // Score against what the base model itself predicted here,
                    // not against this song's actual codes. `targets` goes
                    // unused; LmChunkLabelGuard switches on soft_k.
                    smp.soft_k   = prior->k;
                    smp.s_tr     = std::min<int>(smp.s_tr, prior->n_pos);
                    const int skip = (a.reg_score_last > 0 && smp.s_tr > a.reg_score_last)
                                         ? smp.s_tr - a.reg_score_last : 0;
                    if (skip > 0) {
                        // Ending-targeted: the first `skip` supervised rows stay
                        // as INPUT context and drop out of the loss. Same
                        // column arithmetic as the head (n_masked - 1 + i).
                        smp.n_masked += skip;
                        smp.s_tr     -= skip;
                        smp.soft_idx.assign(prior->idx.begin() + (size_t) skip * (size_t) prior->k,
                                            prior->idx.begin() + (size_t) (skip + smp.s_tr) * (size_t) prior->k);
                        smp.soft_p.assign(prior->p.begin() + (size_t) skip * (size_t) prior->k,
                                          prior->p.begin() + (size_t) (skip + smp.s_tr) * (size_t) prior->k);
                    } else {
                        smp.soft_idx = prior->idx;
                        smp.soft_p   = prior->p;
                    }
                    if (!reg_span_logged) {
                        reg_span_logged = true;
                        fprintf(stderr, "[mm3-lm-train] prior preservation: scoring the last %d of %d supervised rows per reg step "
                                        "(per-row coefficient x%.3g vs the full window)\n",
                                smp.s_tr, (int) n_sup, (double) n_sup / (double) smp.s_tr);
                    }
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
                ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, naive_nodes, true);
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
        // DoRA: A/B moved, so ||W + s*BA||_col is stale for the window that
        // starts now. Nothing reads nrm between the step and the next
        // micro-batch, so refreshing here is exactly the DiT's preWindow.
        if (lora.dora) {
            std::string derr;
            if (!lm_lora_dora_refresh(&lora, sched, &derr)) {
                fprintf(stderr, "[mm3-lm-train] %s\n", derr.c_str());
                rc = 1;
                break;
            }
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
        // WHAT ACTUALLY RAN, once a real fused launch has happened. Not a
        // restatement of --attn: the CUDA dispatch drops to the v1 scalar
        // kernels on pre-Ampere devices, at D != 128 and on an 8-byte-unaligned
        // view, so two runs whose logs both say "flash" can differ in
        // arithmetic. Read from the backend registry, once.
        if (attn_flash && step == 1) {
            const std::string prec = dit_flash_prec_label(t.lm.backend);
            jl("{\"type\":\"attnResolved\",\"mode\":\"%s\",\"prec\":\"%s\"}", a.attn.c_str(),
               json_escape(prec).c_str());
            fprintf(stderr, "[mm3-lm-train] --attn %s resolved to %s\n", a.attn.c_str(), prec.c_str());
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

        // --save-every 0 means "no intermediate checkpoints", never "no
        // adapter": the final step always exports. A 2.4 h albumY run
        // (2026-09-13) ended with nothing on disk because this condition
        // gated the last step too, and the server then removed the resume
        // state as a finished run's leftovers.
        const bool saved_here = (a.save_every > 0 && step % a.save_every == 0) || step == a.steps;
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
                if (!saved_here) {
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
            if (!mm3_lm_resume_save(state_path, rstate, lora, opt, &serr, soft_params)) {
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
        if (mm3_lm_resume_save(state_path, rstate, lora, opt, &ferr, soft_params)) {
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
    if (a.lyrics_dropout > 0.0) {
        fprintf(stderr, "[mm3-lm-train] lyrics dropout: %lld style steps trained without lyrics (asked for %.1f%%)\n",
                (long long) n_lyrics_dropped, 100.0 * a.lyrics_dropout);
    }
    if (a.end_crop_vary) {
        fprintf(stderr, "[mm3-lm-train] varied end supervision: %lld end steps drew a varied window/history\n",
                (long long) n_end_vary);
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
        // others are useless — the album A ladder taught us the ear can prefer a
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
    if (art_buf) {
        ggml_backend_buffer_free(art_buf);
    }
    if (art_ctx) {
        ggml_free(art_ctx);
    }
    lm_prefix_free(&pfx);
    lm_lora_detach(&lora, &t.lm);
    lm_lora_free(&lora);
    mm3_train_lm_free(&t);
    return rc;
}
