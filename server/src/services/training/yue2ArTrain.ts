import { YUE2_OPTIM_DEFAULTS, yue2OptimArgs, type Yue2OptimOptions } from './yue2Optim.js';
// training/yue2ArTrain.ts — YuE2 AR LoRA training: defaults, the regulariser
// pack's location and the argv builder.
//
// The AR half is the COMPOSER, and it is the half that carries artist likeness;
// yue2Train.ts's NAR half is the renderer. They share a base GGUF and a
// manifest and nothing else, so they get separate files for the reason
// yue2Train.ts gives about mm3Train.ts: a merged options type would be mostly
// dead fields. Model resolution IS shared and is imported rather than retyped.
//
// THREE CACHE STAGES, NOT ONE. `yue2-nar-train` needs only the latents that
// `yue2-preprocess` writes. `yue2-ar-train` needs two more passes over the same
// dataset, and all three write into the SAME yue2_preprocess.json:
//
//   1. yue2-preprocess → latents/   (what the app already wires)
//   2. yue2-tokenize   → codes/     `codec_ids` per SOURCE, the whole-song code
//                                   array the next-token loss is scored on.
//                                   Without it the trainer has nothing to train.
//   3. yue2-align      → cursor/    `cursor_words` per source, the forced-
//                                   alignment spans --cursor-weight needs.
//
// So a caller that has only run stage 1 cannot build a working AR run, and the
// failure is at spawn time, not here.
//
// TWO ENGINE REFUSALS THE CALLER OWNS. Both exit 2 before any GPU work, and
// both are reachable from ordinary form values, so they are options here rather
// than surprises:
//
//   --steps > 1500          needs --allow-overtrain. Upstream's README is
//                           emphatic that past ~1500 steps the model memorises
//                           the songs instead of the style.
//   an empty --minted       needs --allow-no-minted. See yue2MintedManifest.
//
// NO VRAM OR TIME MODEL HERE. YUE2_VRAM_MODEL is fitted to the NAR half's site
// set and its fixed 250-frame clips; the AR half trains on whole songs up to
// --max-len 12288 and its buffers are one [H,S] F32 per layer, so nothing in
// that model transfers. Nothing has been measured for this stage, and a
// plausible-looking number would read as a measured one.
//
// LICENCE: YuE2 weights are CC BY-NC 4.0 and a trained adapter is a
// derivative. The notice is YUE2_LICENSE_NOTICE in
// services/backends/yue2/index.ts.

import fs from 'fs';
import path from 'path';

import { config } from '../../config.js';
import { resolveYue2TrainModels, yue2ModelDir } from './yue2Train.js';

// ── The regulariser pack ────────────────────────────────────────────────────

/** The two halves of the minted pack, as the model registry installs them
 *  (`yue2-minted-manifest` / `yue2-minted-codes`, both `subdir: "yue2"`). The
 *  trainer resolves the codes blob RELATIVE TO THE MANIFEST, so they are one
 *  unit on disk and are treated as one here. */
export const YUE2_MINTED_MANIFEST_FILE = 'minted_manifest.json';
export const YUE2_MINTED_CODES_FILE = 'minted_codes.i32';

/** `<models>/yue2/minted_manifest.json`, or '' when the pack is not installed.
 *
 *  Never throws — same discipline as yue2Train.ts's newestMatching: this feeds
 *  a polled status endpoint and a route guard, and a missing models directory
 *  must read as a missing file rather than 500.
 *
 *  BOTH halves are checked. A manifest whose codes blob is absent is a
 *  half-finished download, and reporting it as present would turn a plain
 *  "pack not installed" into a failure deep inside the trainer.
 *
 *  Empty is a legitimate state, not an error: the run then needs
 *  `allowNoMinted`, prints a banner and exports `minted: "absent"` in the
 *  adapter metadata. It is a bad trade — our encoder repeats adjacent codes
 *  3.3-6.6% of the time against YuE2's own 0.02-0.17%, which is out of
 *  distribution in the direction that produces LOOPING, and the 50/50 mix of
 *  true YuE2 tokens is the counterweight — but it is the caller's to make. */
export function yue2MintedManifest(): string {
  try {
    const dir = yue2ModelDir();
    const manifest = path.join(dir, YUE2_MINTED_MANIFEST_FILE);
    const codes = path.join(dir, YUE2_MINTED_CODES_FILE);
    return fs.existsSync(manifest) && fs.existsSync(codes) ? manifest : '';
  } catch {
    return '';
  }
}

// ── Defaults ────────────────────────────────────────────────────────────────

export type Yue2ArTarget = 'attn' | 'attn_mlp';
/** How the artist style string is built from trigger + caption. `upstream` is
 *  the training template (`"<trig>, in the style of <trig>. <caption>.
 *  <genre>, <bpm> BPM, key of <key>."`, the tail written by preprocess from the
 *  ACE sidecar); `bare` is `"<trig>, <caption>"`. GENERATION MUST COMPOSE THE
 *  SAME STRING, which is why the trainer stores the choice in the adapter's
 *  metadata as `style_template`. */
export type Yue2StyleTemplate = 'upstream' | 'bare';
export type Yue2ArAttn = 'exact' | 'flash' | 'flash-f32';
export type Yue2ArLrScheduler = 'cosine' | 'constant';

export function isYue2ArTarget(v: unknown): v is Yue2ArTarget {
  // SMALLER than the NAR target set on purpose. `attn_mlp_embed` is NOT
  // IMPLEMENTED engine-side and says so: the chunked head computes dL/dh by
  // hand, so a trainable lm_head needs a different design.
  return v === 'attn' || v === 'attn_mlp';
}
export function isYue2StyleTemplate(v: unknown): v is Yue2StyleTemplate {
  return v === 'upstream' || v === 'bare';
}
export function isYue2ArAttn(v: unknown): v is Yue2ArAttn {
  return v === 'exact' || v === 'flash' || v === 'flash-f32';
}
export function isYue2ArLrScheduler(v: unknown): v is Yue2ArLrScheduler {
  return v === 'cosine' || v === 'constant';
}

/** The step count above which the engine refuses without `allowOvertrain`. */
export const YUE2_AR_OVERTRAIN_STEPS = 1500;

// The engine's own Yue2ArTrainArgs defaults, with the settled recipe laid over
// them. The recipe is the one proven BY EAR on 2026-09-15; it is NOT the NAR
// recipe (256/256, 10 000 steps), which belongs to the other model half and has
// no bearing here.
//
// Four numbers differ from the engine's:
//
//   steps      1600 → 400   the ear picked step 300 off a 400-step ladder, and
//                           the engine's own 1600 is past the point upstream
//                           warns about memorisation.
//   saveEvery   200 → 50    a rung every 50 steps is what makes 300 pickable.
//   ckptFrom    200 → 50    likewise: the coherent region starts early, and a
//                           rung you never rendered is not a choice.
//   evalEvery   100 → 50    one eval per snapshot, so a rung and its held-out
//                           number line up.
//   captionDropout 0 → 0.5  the dropped form is the trigger phrase ALONE, the
//                           same string for every song, which is what gives the
//                           adapter a canonical prompt to be addressed by.
//
// The form is a VIEW of this object and it is shipped in the status payload, so
// no second copy of these numbers exists on the client. Same rule as
// YUE2_NAR_DEFAULTS and MM3_LM_DEFAULTS.
export const YUE2_AR_DEFAULTS = {
  ...YUE2_OPTIM_DEFAULTS,
  /** The base the recipe was proven on, and the only one an AR run has been
   *  measured against. Whether a quantized base trains here is not
   *  established. */
  lmType: 'bf16',

  // ── the LoRA ──
  /** Application default. The upstream reference and gradient gate use 64. */
  rank: 128,
  alpha: 128,
  /** qkvo + gate/up/down on all 28 layers — upstream's own group. */
  target: 'attn_mlp' as Yue2ArTarget,
  styleTemplate: 'upstream' as Yue2StyleTemplate,

  // ── the loop ──
  lr: 1e-4,
  lrScheduler: 'cosine' as Yue2ArLrScheduler,
  /** The cosine HORIZON (upstream's SCHED_STEPS), deliberately LONGER than the
   *  run: the schedule never reaches its floor of 0.2*lr, so a 400-step run is
   *  the head of a 3000-step curve rather than a complete decay. Moving `steps`
   *  without moving this changes where on the curve the run stops, which is
   *  part of the recipe. */
  schedSteps: 3000,
  warmup: 50,
  steps: 400,
  gradAccum: 2,
  maxGradNorm: 1.0,
  weightDecay: 0,
  /** Upstream's betas, NOT lm-optim.h's (0.9, 0.999). Side by side on the same
   *  twelve songs these descended from an identical step-0 loss to 0.032 by
   *  step 800 where the 0.999 run was still at 0.165: beta2 is how long AdamW's
   *  second moment remembers, so it sets how hard a small set is memorised. */
  adamBeta1: 0.9,
  adamBeta2: 0.95,
  /** One draw per micro-step: `random() < frac ? artist : minted`. */
  artistFrac: 0.5,
  captionDropout: 0.5,

  /** The preprocess caption mode this trainer needs, which is NOT the NAR
   *  default. `none` leaves every row style-less, and the AR prefix IS the
   *  style: an adapter trained that way has no caption to condition on and
   *  none to steer with at generation. `ace` parses the Option-A sidecar,
   *  taking `caption:` as the style and `lyrics:` as the lyrics. Measured
   *  2026-09-15 (_LISTENING/2026-09-14/RESULTS.md, arms 130-147): the
   *  conditioning is the whole difference between an adapter that caps and one
   *  that sounds like the artist. `yue2` is that same parse with the style
   *  taken from `<stem>.yue2.txt` where the labeling pass wrote one — the
   *  planner's own sentence order, so the trained prefix and the prompt Lyric
   *  Studio writes for a new song are the same shape. */
  captionMode: 'yue2' as const,
  /** Upstream's CUR_W. The lyric-cursor auxiliary loss, measured to matter:
   *  frame-to-lyric alignment loss 12.3 → 1.6 with it, 10.8 → 16.3 without, on
   *  the same songs. It needs `cursor_words` in the manifest, i.e. the
   *  `yue2-align` stage. */
  cursorWeight: 0.08,
  /** doc 19 decision 4: when a source's manifest carries `abc` (a SheetSage2
   *  lead sheet, written by `yue2-sheet`), the AR prefix is built cot=full
   *  w.p. (1 - abcDropout) instead of cot=off. A source with `abc_error` or
   *  neither field always trains cot=off, never entering this draw. 0.5 is
   *  upstream's own split. */
  abcDropout: 0.5,
  seed: 42,

  // ── shapes and cost ──
  /** Upstream's MAXLEN. A song over it truncates WITHOUT MUSIC_END, so a
   *  truncated song never teaches a fake ending. There are no crop flags by
   *  design — whole songs fit, and random crops teach that a song may begin
   *  mid-flow, which MM3 paid for twice. This also SIZES the buffers (one [H,S]
   *  F32 per layer), so lowering it is the VRAM lever. */
  /** 9000, not the engine's 12288: the AR sequence is prefix + whole song, and
   *  12288 sizes the per-layer F32 buffers for a length no song in a normal
   *  album reaches. 9000 frames is 6 minutes of audio and is what every run
   *  behind the settled recipe used; a longer song is skipped with its name,
   *  never cropped. */
  maxLen: 9000,
  /** exact | flash | flash-f32. Flash removes the retained [S,S,Nh] softmax —
   *  6.0 GiB at S=10,001 — and is probed at the run's real shapes with a HARD
   *  ERROR on an unsupported backend. Left at `exact` because the recipe was
   *  proven there. */
  /** `flash` because the difference is 40 s/it vs 8 s/it on the same card, not
   *  a quality trade: the fused path is gated against exact (docs/plans/yue2,
   *  the FD gate passes under flash at 128 and 2048 frames). `exact` is the
   *  engine's own default because it is the one that always exists; every run
   *  behind the settled recipe used flash. */
  attn: 'flash' as Yue2ArAttn,
  /** Supervised rows per CE chunk. */
  chunk: 256,

  // ── the ladder ──
  saveEvery: 50,
  ckptFrom: 50,
  evalEvery: 50,
  /** Every step. A 400-step run at ~5 s/it emits one line every five seconds,
   *  which is not a flood — and at 20 the step counter in the UI sat still for
   *  two minutes at a time, which reads as a hung run. The chart wants the
   *  points too: 20 samples across a run is not a curve. */
  logEvery: 1,
  /** Which rung the ear picked, and what a picker should preselect. It is a
   *  default, not a verdict on any particular dataset: upstream's instruction
   *  is to pick the checkpoint BY EAR, and the held-out loss does not identify
   *  it. */
  ckptPickStep: 300,

  /** Read HOT-Step's ACE dataset sidecar `<stem>.txt` beside the source audio
   *  when the manifest carries no lyrics (it does not yet). Off means the
   *  prefix falls back to --style/--lyrics, and a style-less, lyric-less prefix
   *  trains and is wrong. */
  sidecars: true,
} as const;

// ── Layout ──────────────────────────────────────────────────────────────────

/** Where every YuE2 AR run lives, one level above a run directory.
 *
 *  A SEPARATE ROOT from the NAR one, and not a cosmetic split: run discovery
 *  matches `<stem>_step<N>.safetensors` against ONE fixed stem per directory
 *  tree, so AR and NAR runs sharing a root would each be invisible to the
 *  other's scanner or, worse, half-visible. */
export function yue2ArAdapterRoot(): string {
  return path.join(config.aceServer.adapters, 'yue2-ar-adapters');
}

export function yue2ArAdapterRunDir(runName: string): string {
  return path.join(yue2ArAdapterRoot(), runName);
}

/** The export stem `yue2-ar-train --name` defaults to. Fixed rather than
 *  offered, for the reason YUE2_ADAPTER_STEM is: the snapshot ladder is found
 *  by matching this stem in the run directory, and a per-run stem would have to
 *  be stored somewhere before the directory could be read. */
export const YUE2_AR_ADAPTER_STEM = 'yue2_ar_lora';

/** `<dataset>-YYYY-MM-DD_HH-MM-SS`, the logs/ convention: name-sorted is
 *  time-sorted, and retraining never overwrites an earlier run. Same shape as
 *  the NAR runs' names — the roots keep the two apart, not the names. */
export { yue2RunName as yue2ArRunName } from './yue2Train.js';

// ── Arg building ────────────────────────────────────────────────────────────
//
// Every flag below is one cmd_yue2_ar_train actually parses
// (engine/tools/ace-train.cpp). An unknown option is a hard exit 2, so nothing
// speculative belongs here.

export interface ResolvedYue2ArTrainOptions extends Partial<Yue2OptimOptions> {
  /** `<latents>/yue2_preprocess.json`, read as SOURCES rather than clips: the
   *  source-level `codec_ids` file is the whole-song code array. It must
   *  already carry codes (`yue2-tokenize`), and `cursor_words`
   *  (`yue2-align`) when cursorWeight > 0. */
  manifest: string;
  /** The minted pack's manifest, or '' for an artist-only run. Empty REQUIRES
   *  allowNoMinted; see yue2MintedManifest. */
  minted: string;
  /** Start the run anyway with an empty `minted`. Without it the engine exits 2
   *  before touching the GPU. */
  allowNoMinted: boolean;
  outDir: string;
  /** `yue2-lm-<lmType>.gguf`, resolved at spawn time rather than stored, so a
   *  model moved between queueing and running fails with a named file. */
  lmType: string;
  /** Prepended to every caption and the word the trained style is addressed by
   *  at generation time. It is written into the exported metadata, so half a
   *  run under a different trigger is a different adapter. NOT applied to
   *  minted styles. */
  trigger: string;
  styleTemplate: Yue2StyleTemplate;
  /** Fallback conditioning for a clip that states none of its own. The loop
   *  takes captions from the manifest and, with sidecars on, lyrics from the
   *  ACE sidecar; these apply only when neither does. */
  style: string;
  lyrics: string;
  sidecars: boolean;
  target: Yue2ArTarget;
  rank: number;
  alpha: number;
  lr: number;
  lrScheduler: Yue2ArLrScheduler;
  /** The cosine horizon, not the run length. See YUE2_AR_DEFAULTS.schedSteps. */
  schedSteps: number;
  warmup: number;
  steps: number;
  /** Required once steps exceeds YUE2_AR_OVERTRAIN_STEPS, where upstream says
   *  the model starts memorising the songs rather than the style. */
  allowOvertrain: boolean;
  gradAccum: number;
  maxGradNorm: number;
  weightDecay: number;
  artistFrac: number;
  adamBeta1: number;
  adamBeta2: number;
  captionDropout: number;
  attn: Yue2ArAttn;
  maxLen: number;
  chunk: number;
  /** 0 switches the lyric-cursor term off explicitly. Above 0 the engine
   *  REFUSES a run where no artist song binds its `cursor_words`, which is the
   *  gate that would otherwise print "cursor nan" forever and still finish. */
  cursorWeight: number;
  /** See YUE2_AR_DEFAULTS.abcDropout. */
  abcDropout: number;
  seed: number;
  ckptFrom: number;
  saveEvery: number;
  evalEvery: number;
  logEvery: number;
  /** Continue `<out>/yue2_ar_ckpt.bin`. The engine REFUSES a resume across
   *  rank/alpha/target/grad-accum/seed and any change to manifest/minted/
   *  trigger/style/lyrics/sidecars/artist-frac/max-len/attn; it NOTES but
   *  allows the schedule knobs. Exactness holds within one machine and build
   *  only — a different GPU, driver, ggml build or base GGUF breaks it and
   *  cannot be detected. */
  resume: boolean;
  /** For the run manifest and the job's log lines. */
  datasetSlug: string;
  datasetName: string;
}

export function buildYue2ArTrainArgs(o: ResolvedYue2ArTrainOptions): string[] {
  const m = resolveYue2TrainModels(o.lmType);
  const args = [
    'yue2-ar-train',
    ...yue2OptimArgs(o),
    '--lm', m.lm,
    '--manifest', o.manifest,
    '--out', o.outDir,
    '--name', YUE2_AR_ADAPTER_STEM,
    '--style-template', o.styleTemplate,
    '--target', o.target,
    '--rank', String(o.rank),
    '--alpha', String(o.alpha),
    '--lr', String(o.lr),
    '--lr-scheduler', o.lrScheduler,
    '--sched-steps', String(o.schedSteps),
    '--warmup', String(o.warmup),
    '--steps', String(o.steps),
    '--grad-accum', String(o.gradAccum),
    '--max-grad-norm', String(o.maxGradNorm),
    '--weight-decay', String(o.weightDecay),
    '--artist-frac', String(o.artistFrac),
    '--attn', o.attn,
    '--max-len', String(o.maxLen),
    '--chunk', String(o.chunk),
    '--cursor-weight', String(o.cursorWeight),
    '--abc-dropout', String(o.abcDropout),
    '--adam-beta1', String(o.adamBeta1),
    '--adam-beta2', String(o.adamBeta2),
    '--caption-dropout', String(o.captionDropout),
    '--ckpt-from', String(o.ckptFrom),
    '--save-every', String(o.saveEvery),
    '--eval-every', String(o.evalEvery),
    '--log-every', String(o.logEvery),
    '--seed', String(o.seed),
    // `off` is the only spelling the engine tests for; anything else is on.
    '--sidecars', o.sidecars ? 'on' : 'off',
  ];
  if (o.minted) args.push('--minted', o.minted);
  // Emitted only when set, like --trigger below: the engine takes an empty
  // value for these and passing `--style ""` is the same run with a
  // different-looking command line.
  if (o.style) args.push('--style', o.style);
  if (o.lyrics) args.push('--lyrics', o.lyrics);
  if (o.trigger) args.push('--trigger', o.trigger);
  // The two permissions. NOT inferred from `steps`/`minted` here: if the caller
  // has not decided to overtrain or to train without the regulariser, the
  // engine's own refusal is the right place to find out, and quietly adding the
  // flag would turn a guard into a formality.
  if (o.allowOvertrain) args.push('--allow-overtrain');
  if (o.allowNoMinted) args.push('--allow-no-minted');
  // NOT suppressed when the state file is absent: `--resume` with nothing to
  // resume is the engine's own error to report, and dropping the flag would
  // restart from step 1 while the UI said "continuing".
  if (o.resume) args.push('--resume');
  // Deliberately NOT emitted: --weights (f32 is the default and mul_mat's
  // activation backward is F32-only on CUDA anyway), --ce-full/--ce-slice, and
  // the --fd-* / --forward-check gates, which exit instead of training.
  return args;
}
