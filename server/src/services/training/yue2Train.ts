// training/yue2Train.ts — YuE2 NAR LoRA training: model resolution, defaults,
// the VRAM model and the two argv builders.
//
// The YuE2 analogue of mm3Train.ts, and kept apart from it for the same reason
// that file is kept apart from aceTrain.ts: the CLI surfaces share nothing.
// `yue2-preprocess` takes a FOLDER of audio and writes cached VAE latents;
// `yue2-nar-train` takes that cache's manifest and trains a LoRA on the frozen
// NAR half. Neither takes a dataset manifest, a codes directory or a tensor
// cache, so folding them into either existing file would produce a type where
// most fields are always unused.
//
// Phase 5 of docs/plans/yue2/08-nar-lora-trainer.md; the survey this follows is
// docs/plans/yue2/11-training-studio-notes.md.
//
// THREE SHAPES THAT DIFFER FROM MM3, all of them consequences of the engine
// CLI rather than choices made here (11 §4.2, §7):
//
//   1. `yue2-preprocess` scans --audio FLAT and takes no manifest, so the
//      dataset's exclusions and its `recursive` flag cannot reach it. The
//      count the tool reports and the count in dataset.json can therefore
//      disagree, and the route says so rather than hiding it.
//   2. There is no `--captions <dir>` and no `.yue2.txt` convention.
//      `--caption-mode txt` reads the same-named `.txt` beside the audio RAW
//      AND WHOLE — which in a HOT-Step dataset is the ACE Option-A sidecar
//      (`caption: …`, `genre: …`, then lyrics), i.e. training the style
//      encoder on field syntax. So the default here is `none`: the clip
//      caption is empty and the style is the trigger word alone, which is
//      exactly what the adapter is addressed by at generation time
//      (yue2_nt_style, yue2-nar-train-run.h:1594). `default` and `txt` are
//      offered, and `txt` is warned about at the route.
//   3. The trainers emit NO JSONL. Progress is parsed out of their stderr —
//      see yue2TrainRunner.ts.
//
// LICENCE: YuE2 weights are CC BY-NC 4.0 and a trained adapter is a
// derivative, so it carries the same terms. The notice is
// YUE2_LICENSE_NOTICE in services/backends/yue2/index.ts and is re-exported
// through the status route rather than retyped — one place to read it from.

import fs from 'fs';
import path from 'path';

import { config, getFFmpegPath } from '../../config.js';
import { datasetDir } from './paths.js';

// ── Model files ─────────────────────────────────────────────────────────────

/** `<models>/yue2` — the same directory the engine's own discovery scans
 *  (yue2_discover tries `<models>/yue2` then `<models>`; we only ever write
 *  the first, and both CLI tools take `--models <models>` and do their own
 *  two-level search from there). */
export function yue2ModelDir(): string {
  return path.join(config.aceServer.models, 'yue2');
}

export type Yue2VaeVariant = 'standard' | 'legacy';

export interface Yue2TrainModels {
  /** `<models>/yue2/yue2-lm-<type>.gguf`. */
  lm: string;
  /** The quant token of `lm`, echoed back so a caller need not re-split it. */
  lmType: string;
  /** Newest `yue2-vae-<variant>-*.gguf`, or '' when none is installed. The
   *  type token is open (f32 today, quantized variants later), so this is a
   *  scan rather than a fixed filename. */
  vae: string;
  vaeVariant: Yue2VaeVariant;
}

/** Newest file matching a prefix, or ''. Same helper shape as mm3Train's:
 *  returns '' on ANY failure so a missing directory reads as a missing file
 *  rather than throwing out of a status endpoint. */
function newestMatching(dir: string, prefix: string): string {
  try {
    const hits = fs.readdirSync(dir)
      .filter(f => f.startsWith(prefix) && f.endsWith('.gguf'))
      .map(f => ({ f, m: fs.statSync(path.join(dir, f)).mtimeMs }))
      .sort((a, b) => b.m - a.m);
    return hits.length ? path.join(dir, hits[0].f) : '';
  } catch {
    return '';
  }
}

export function resolveYue2TrainModels(lmType: string = YUE2_NAR_DEFAULTS.lmType,
                                       vaeVariant: Yue2VaeVariant = 'standard'): Yue2TrainModels {
  const dir = yue2ModelDir();
  return {
    lm: path.join(dir, `yue2-lm-${lmType}.gguf`),
    lmType,
    vae: newestMatching(dir, `yue2-vae-${vaeVariant}-`),
    vaeVariant,
  };
}

/** Which required files are missing, as user-facing names. Empty = ready.
 *
 *  `need` narrows, and the two stages genuinely want different things:
 *
 *    'preprocess' → the VAE only. The encoder half is what runs, and the
 *                   trainer never sees a VAE at all.
 *    'train'      → the LM only. `yue2-nar-train` accepts `--vae-dir` and
 *                   IGNORES it (yue2-nar-train-run.h:121-123, warned at :389):
 *                   latents reach it through the manifest. Checking for a VAE
 *                   here would refuse a run that would have worked.
 *
 *  The encoder caveat: `has_encoder` is a GGUF key, not a filename fact
 *  (yue2-model.h:132). Both shipped checkpoints carry it, so the filename
 *  check is adequate today — but a decoder-only VAE fails as "preprocess died
 *  after loading the model", which is why the label below names the encoder. */
export function missingYue2TrainModels(need: 'preprocess' | 'train',
                                       opts: { lmType?: string; vaeVariant?: Yue2VaeVariant } = {}): string[] {
  const m = resolveYue2TrainModels(opts.lmType ?? YUE2_NAR_DEFAULTS.lmType,
                                   opts.vaeVariant ?? 'standard');
  const wanted: Array<[string, string]> = need === 'preprocess'
    ? [[m.vae, `a YuE2 ${m.vaeVariant} VAE with its encoder (yue2-vae-${m.vaeVariant}-*.gguf)`]]
    : [[m.lm, path.basename(m.lm)]];
  return wanted.filter(([p]) => !p || !fs.existsSync(p)).map(([, label]) => label);
}

export interface Yue2BaseInfo {
  /** The quant token: 'bf16', 'Q6_K', 'IQ4_XS-imat', … */
  id: string;
  file: string;
  bytes: number;
  /** Peak VRAM at the default rank, for the picker. */
  peakMb: number;
  /** Measured under training. Only bf16 has been. See YUE2_VRAM_MODEL. */
  proven: boolean;
}

/** Every installed `yue2-lm-*.gguf`, bf16 first and the rest by size.
 *
 *  DELIBERATELY NOT a quality ladder like availableMm3Bases(). MM3 publishes a
 *  measured first-step loss delta per base; nothing equivalent has been
 *  measured for YuE2, and MM3's QLoRA-style dequantize-per-matmul result is an
 *  MM3 fact about mm3-lm-train, not a YuE2 one. So the list reports what is on
 *  disk and marks the ONE base a run has actually been trained on. Inventing a
 *  ladder here would be inventing measurements. */
export function availableYue2Bases(rank: number = YUE2_NAR_DEFAULTS.rank): Yue2BaseInfo[] {
  const dir = yue2ModelDir();
  let files: string[] = [];
  try {
    files = fs.readdirSync(dir).filter(f => f.startsWith('yue2-lm-') && f.endsWith('.gguf'));
  } catch {
    return [];
  }
  return files
    .map(f => {
      const id = f.slice('yue2-lm-'.length, -'.gguf'.length);
      let bytes = 0;
      try { bytes = fs.statSync(path.join(dir, f)).size; } catch { /* raced */ }
      return {
        id,
        file: path.join(dir, f),
        bytes,
        peakMb: estimateYue2PeakMb(bytes, rank),
        proven: id === YUE2_NAR_DEFAULTS.lmType,
      };
    })
    .sort((a, b) => (Number(b.proven) - Number(a.proven)) || (b.bytes - a.bytes));
}

// ── Defaults: the measured recipe, in one place ─────────────────────────────
//
// MEASURED 2026-09-14 on real albums (RTX 5090, bf16 base). These are NOT the
// engine's own defaults — Yue2NarTrainArgs ships rank 16 / alpha 16 / 800
// steps, which is upstream's starting point from 08 §7 and was never the
// recipe anything was trained under here.
//
// The form is a VIEW of this object; it is shipped in the status payload so no
// second copy of these numbers exists on the client. Same rule as
// MM3_LM_DEFAULTS.
export const YUE2_NAR_DEFAULTS = {
  /** The only base a run has been measured on. 19.6 GB at rank 256, ~7 GB of
   *  which is the file itself. Whether a quantized base trains at all here is
   *  NOT established — see availableYue2Bases. */
  lmType: 'bf16',
  vaeVariant: 'standard' as Yue2VaeVariant,

  // ── the LoRA ──
  /** 256/256 measured. Upstream's 16/16 fits in a third of the VRAM and has
   *  not been heard here; the ladder between them is unexplored. */
  rank: 256,
  alpha: 256,
  /** nar_attn | nar_attn_mlp | nar_attn_mlp_proj. `nar_attn_mlp` is 196 LoRA
   *  sites (392 exported tensors) — attention q/k/v/output plus the FFN
   *  gate/up/down on all 28 NAR blocks, and none of the projection heads.
   *  `_proj` adds the four head sites (vae2llm, llm2vae, time_embd.{0,1}),
   *  which is 200 sites and 400 tensors.
   *
   *  `_proj` because that is what the campaign's renders were made under
   *  (18 §7, "NAR rank 256 codec-conditioned nar_attn_mlp_proj, 10k steps,
   *  merged at ~0.6"), and it is what the engine's own --fd-check promotes an
   *  unset preset to. The VRAM model below was fitted on nar_attn_mlp, so it
   *  now under-reads by the four head sites — 2% of the parameters, tens of
   *  MB at rank 256, inside the slack the constant term already carries. */
  target: 'nar_attn_mlp_proj' as Yue2NarTarget,

  // ── the loop ──
  lr: 1e-4,
  lrScheduler: 'cosine' as 'cosine' | 'constant',
  warmup: 50,
  /** 10 000 at 0.06-0.07 s/it is about 12 minutes. Not an ear-validated
   *  optimum: it is the depth the measured runs used. Ladder the snapshots. */
  steps: 10000,
  /** A snapshot every 2000 steps gives five rungs on a 10 000-step run, which
   *  is the ladder the ear needs. Each snapshot also rewrites the resume
   *  state, so this is the granularity a killed run comes back at. */
  saveEvery: 2000,
  gradAccum: 1,
  maxGradNorm: 1.0,
  weightDecay: 0.01,
  /** Chance per micro-step of swapping in the EMPTY-style prefix. It is what
   *  keeps the base style reachable at all after the adapter is merged, so
   *  0 is not "more likeness", it is "the trigger word stops meaning
   *  anything relative to no trigger word". */
  captionDropout: 0.1,
  tSampling: 'logit-normal' as 'logit-normal' | 'uniform',
  seed: 42,
  /** AR-prefix K/V canvases held at once, ~104 MB each at 10 s clips. A real
   *  VRAM knob; the engine default is 8 and that is what was measured. */
  kvCache: 8,
  /** Micro-steps one working set of clips is held for. With per-clip
   *  `codec_ids` every clip is its own conditioning, so a uniform draw hits
   *  the K/V cache only `cap/N` of the time — 8 of 249 on a thirteen-song
   *  album — and pays a full AR prefill on nearly every step (0.18 s/it
   *  against 0.065 warm, with the GPU at 41%). Sampling inside a set sized to
   *  the cache keeps it warm. 0 draws from the whole corpus, which is what the
   *  text-only regime wants; the engine engages it only when the clips carry
   *  codec ids, so this default is inert on a text-only cache. */
  clipBlock: 64,
  logEvery: 10,
  /** Keep `<out>/yue2_nar_ckpt.bin` after a clean finish? The engine deletes
   *  it on a successful export (yue2-nar-train-run.h, "the run's MIDDLE, not
   *  its result"), which is the right default — it is adapter + AdamW moments
   *  and it is large. There is no server-side way to keep it without changing
   *  the engine, so this is reported, not offered. */
  deletesResumeStateOnFinish: true,

  // ── preprocess ──
  /** 10 s = 250 frames at 25 fps. Contract §8 calls this structural: the
   *  trainer reads the manifest's own clip_frames unless --frames is typed. */
  clipSeconds: 10,
  /** `none` = the clip carries no caption and the style is the trigger alone.
   *  See the file header for why this, and not `txt`, is the default. */
  captionMode: 'none' as Yue2CaptionMode,
  defaultCaption: '',
  /** auto = the repo's own WAV/MP3 decoder for those two and ffmpeg for the
   *  rest; ffmpeg = one resampler across a mixed corpus. */
  decode: 'auto' as 'auto' | 'ffmpeg',
  /** Encoder tiling. The halo floor is the encoder's own receptive field and
   *  the engine refuses anything under it. */
  tileFrames: 750,
  haloFrames: 20,
} as const;

export type Yue2NarTarget = 'nar_attn' | 'nar_attn_mlp' | 'nar_attn_mlp_proj';
export type Yue2CaptionMode = 'ace' | 'txt' | 'default' | 'none';

export function isYue2Target(v: unknown): v is Yue2NarTarget {
  return v === 'nar_attn' || v === 'nar_attn_mlp' || v === 'nar_attn_mlp_proj';
}
export function isYue2CaptionMode(v: unknown): v is Yue2CaptionMode {
  return v === 'ace' || v === 'txt' || v === 'default' || v === 'none';
}
export function isYue2VaeVariant(v: unknown): v is Yue2VaeVariant {
  return v === 'standard' || v === 'legacy';
}

/** NAR blocks in the shipped checkpoint. The engine takes this from the model
 *  (`yue2_nt_adapter_tensor_count(n_layers, target)`,
 *  yue2-nar-train-graph.h:805); 28 is what the shipped LM reports — "392 LoRA
 *  tensors over 28 NAR layers" in the logs of the runs these defaults came
 *  from. It is here only so the form can show a count before anything is
 *  spawned; the trainer's own line is the authority. */
export const YUE2_NAR_BLOCKS = 28;

/** Exported SAFETENSORS per target — an A and a B for every LoRA site, which
 *  is the number the trainer prints ("exported 392 tensors"). `(blocks *
 *  per_layer + heads) * 2`, with per_layer 4 for attention-only and 7 with the
 *  FFN, plus 4 head sites (vae2llm, llm2vae, time_embd.{0,1}) for the proj
 *  variant. NOTE the two ways of counting: nar_attn_mlp is 196 SITES and 392
 *  TENSORS, and the measured recipe's "196" is the first. */
export function yue2TargetTensorCount(target: Yue2NarTarget, blocks = YUE2_NAR_BLOCKS): number {
  const perLayer = target === 'nar_attn' ? 4 : 7;
  const heads = target === 'nar_attn_mlp_proj' ? 4 : 0;
  return (blocks * perLayer + heads) * 2;
}

export const YUE2_TARGET_TENSORS: Record<Yue2NarTarget, number> = {
  nar_attn: yue2TargetTensorCount('nar_attn'),
  nar_attn_mlp: yue2TargetTensorCount('nar_attn_mlp'),
  nar_attn_mlp_proj: yue2TargetTensorCount('nar_attn_mlp_proj'),
};

// ── Presets ─────────────────────────────────────────────────────────────────
//
// Step count is the only lever measured here, so it is the only thing the
// presets move. 0.06-0.07 s/it means the wall-clock is close to linear and
// easy to state honestly.
export type Yue2PresetName = 'fast' | 'balanced' | 'thorough';
export const YUE2_DEFAULT_PRESET: Yue2PresetName = 'balanced';
export const YUE2_PRESETS: Record<Yue2PresetName, { steps: number; saveEvery: number }> = {
  fast:     { steps: 5000,  saveEvery: 1000 },
  balanced: { steps: 10000, saveEvery: 2000 },
  thorough: { steps: 20000, saveEvery: 2500 },
};
export function isYue2PresetName(v: unknown): v is Yue2PresetName {
  return v === 'fast' || v === 'balanced' || v === 'thorough';
}
type Yue2PresetFields = (typeof YUE2_PRESETS)[Yue2PresetName];
export type Yue2EffectiveDefaults = Omit<typeof YUE2_NAR_DEFAULTS, keyof Yue2PresetFields> & Yue2PresetFields;
/** Defaults with a named preset laid over them; an unknown or absent name
 *  returns the defaults untouched (which ARE the Balanced preset). */
export function applyYue2Preset(defaults: typeof YUE2_NAR_DEFAULTS, preset: unknown): Yue2EffectiveDefaults {
  return isYue2PresetName(preset) ? { ...defaults, ...YUE2_PRESETS[preset] } : defaults;
}

// ── VRAM ────────────────────────────────────────────────────────────────────
//
// TWO ANCHORS, MEASURED 2026-09-14 (RTX 5090, yue2-lm-bf16, AdamW, 10 s clips,
// target nar_attn_mlp, kv-cache 8):
//
//     rank 128  ->  17.0 GB peak
//     rank 256  ->  19.6 GB peak
//
// The bf16 base file is 6931 MB and is INCLUDED in both, so the two anchors
// fix a straight line in rank on top of it:
//
//     perRankMb = (19600 - 17000) / 128 = 20.3125
//     constMb   = 17000 - 6931 - 128 * 20.3125 = 7469
//
// Linear in rank, not quadratic in sequence length like MM3's: the clip length
// is fixed at 250 frames by the manifest and the AR prefix is cached, so
// nothing here scales with a crop the user can drag. If clip length ever
// becomes a form control this model needs a second pair of anchors before it
// can be trusted at another length.
export const YUE2_VRAM_MODEL = {
  /** 12 bytes per LoRA parameter (weight + grad + one momentum, F32) plus
   *  AdamW's second moment, over the nar_attn_mlp site set. Fitted, not
   *  derived — the two anchors are the authority. */
  perRankMb: 20.3125,
  /** Everything that is neither the base file nor the rank: activations, the
   *  checkpoint segments, the K/V canvases and the arena. */
  constMb: 7469,
  /** The bf16 base's own size, used only when the file cannot be stat'd. */
  fallbackBaseMb: 6931,
  /** The preprocess encode compute buffer, measured the same night. The VAE
   *  encoder is the only thing resident in that stage, so this plus the VAE
   *  file is the whole footprint. */
  encodeComputeMb: 3700,
  /** Measured seconds per optimizer step at the anchors above, in the
   *  TEXT-ONLY regime — one conditioning for the whole corpus, so the AR
   *  prefix is prefilled once and every step is NAR work alone. */
  secondsPerStep: 0.065,
  /** And with per-clip codec conditioning, where a step also pays its share of
   *  AR prefills. 0.10 measured on the Crimson cache with the working-set
   *  sampler at --clip-block 64 / --kv-cache 8 (88% cache hits); it was 0.18
   *  under the uniform draw that preceded it, at 3%. A cache built by the AR
   *  pipeline ALWAYS carries codec ids, so this is the number most in-app runs
   *  get and the form must not quote the other one. */
  secondsPerStepCodec: 0.10,
} as const;

/** Peak VRAM for a training configuration, in MB.
 *
 *  SHIPPED TO THE UI AS COEFFICIENTS, not just as an answer: the form
 *  re-estimates as rank moves, and a second copy of these numbers over there
 *  would drift from the measurements that produced them. */
export function estimateYue2PeakMb(baseBytes: number, rank: number): number {
  const M = YUE2_VRAM_MODEL;
  const loaded = baseBytes > 0 ? baseBytes / 1048576 : M.fallbackBaseMb;
  return Math.round(loaded + M.perRankMb * Math.max(0, rank) + M.constMb);
}

/** Peak VRAM for the preprocess stage, in MB. The VAE file plus the encode
 *  compute buffer; there is no rank and no optimizer. */
export function estimateYue2PreprocessMb(vaeBytes: number): number {
  return Math.round((vaeBytes > 0 ? vaeBytes / 1048576 : 0) + YUE2_VRAM_MODEL.encodeComputeMb);
}

/** Wall-clock estimate for a run, in ms.
 *
 *  Takes the regime, because it is worth a factor on a 10 000-step run:
 *  text-only is ~11 min, codec-conditioned ~17. Quoting the text-only number
 *  for a codec-conditioned cache is how the form came to promise 11 minutes
 *  for a run that took 30. */
export function estimateYue2RunMs(steps: number, codecConditioned = false): number {
  const perStep = codecConditioned
    ? YUE2_VRAM_MODEL.secondsPerStepCodec
    : YUE2_VRAM_MODEL.secondsPerStep;
  return Math.max(0, steps) * perStep * 1000;
}

// ── Layout ──────────────────────────────────────────────────────────────────

/** `<training>/datasets/<slug>/yue2-latents` — the latent cache and the
 *  manifest `yue2-nar-train --manifest` consumes.
 *
 *  Under the dataset, for the reason mm3CodesDir gives: deleting a dataset
 *  should take its derived data with it. The cache is keyed per SOURCE FILE
 *  and is clip-length independent, so re-cutting at a different --clip-seconds
 *  costs a manifest rewrite and no GPU time (--force). */
export function yue2LatentsDir(slug: string): string {
  return path.join(datasetDir(slug), 'yue2-latents');
}

/** The manifest yue2-preprocess writes into that directory. Its name is fixed
 *  by the engine (yue2-preprocess-run.h), not chosen here. */
export function yue2PreprocessManifest(slug: string): string {
  return path.join(yue2LatentsDir(slug), 'yue2_preprocess.json');
}

/** What the latent cache holds, as the manifest itself reports it. */
export interface Yue2CacheSummary {
  sources: number;
  clips: number;
  clipFrames: number;
  captionMode: string;
  totalAudioSec: number;
  skipped: number;
  failed: number;
}

/** Read the preprocess manifest's own summary block. Null when there is no
 *  cache, or when the file is mid-write. Never throws: this feeds a polled
 *  status endpoint and a route guard, and neither may 500 over a partial
 *  file. */
export function readYue2PreprocessSummary(manifestPath: string): Yue2CacheSummary | null {
  try {
    const j = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as Record<string, unknown>;
    // Keyed on the FIELD, not on `format`. The trainer itself only NOTEs a
    // format it does not recognise and reads the file anyway
    // (yue2-nar-train-run.h:1279), so refusing here on a future
    // `yue2-preprocess-v2` would block a run the engine would have accepted.
    if (!Number.isFinite(Number(j.n_clips))) return null;
    return {
      sources: Number(j.n_sources) || 0,
      clips: Number(j.n_clips) || 0,
      clipFrames: Number(j.clip_frames) || 0,
      captionMode: typeof j.caption_mode === 'string' ? j.caption_mode : '',
      totalAudioSec: Number(j.total_audio_sec) || 0,
      skipped: Number(j.n_skipped) || 0,
      failed: Number(j.n_failed) || 0,
    };
  } catch {
    return null;
  }
}

/** Where every YuE2 NAR run lives, one level above a run directory. */
export function yue2AdapterRoot(): string {
  return path.join(config.aceServer.adapters, 'yue2-nar-adapters');
}

export function yue2AdapterRunDir(runName: string): string {
  return path.join(yue2AdapterRoot(), runName);
}

/** `<dataset>-YYYY-MM-DD_HH-MM-SS`, the logs/ convention: name-sorted is
 *  time-sorted, and retraining never overwrites an earlier run. */
export function yue2RunName(slug: string): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, '0');
  return `${slug}-${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_`
       + `${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
}

/** The export stem `yue2-nar-train --name` defaults to. Fixed rather than
 *  offered: yue2Runs.ts matches `<name>_step<N>.safetensors` to find the
 *  snapshot ladder, and a per-run stem would mean storing the stem somewhere
 *  before the directory could be read. */
export const YUE2_ADAPTER_STEM = 'yue2_nar_lora';

// ── Arg building ────────────────────────────────────────────────────────────
//
// Every flag below is one the two subcommands actually parse
// (engine/tools/ace-train.cpp cmd_yue2_preprocess / cmd_yue2_nar_train). An
// unknown option is a hard exit 2, so nothing speculative belongs here.

export interface ResolvedYue2PreprocessOptions {
  /** The dataset's source folder. Scanned FLAT by the engine: subfolders and
   *  the dataset's exclusions do not reach it. */
  audioDir: string;
  outDir: string;
  vaeVariant: Yue2VaeVariant;
  clipSeconds: number;
  captionMode: Yue2CaptionMode;
  defaultCaption: string;
  decode: 'auto' | 'ffmpeg';
  tileFrames: number;
  haloFrames: number;
  /** Case-insensitive name filter, and a cap on how many files are taken.
   *  0/'' = no limit, no filter. */
  only: string;
  limit: number;
  /** Re-cut an existing manifest that was built at a different clip length.
   *  Cheap: the cached latents are clip-length independent. */
  force: boolean;
  /** Carried for the runner's own log lines and the run manifest. */
  datasetSlug: string;
}

export function buildYue2PreprocessArgs(o: ResolvedYue2PreprocessOptions): string[] {
  const args = [
    'yue2-preprocess',
    '--audio', o.audioDir,
    '--out', o.outDir,
    '--models', config.aceServer.models,
    '--vae', o.vaeVariant,
    '--clip-seconds', String(o.clipSeconds),
    '--caption-mode', o.captionMode,
    '--decode', o.decode,
    '--tile-frames', String(o.tileFrames),
    '--halo-frames', String(o.haloFrames),
  ];
  // Only meaningful in `default` mode, and the engine REFUSES that mode with
  // an empty one — so it is emitted exactly when it is required.
  if (o.captionMode === 'default') args.push('--default-caption', o.defaultCaption);
  // FLAC/OGG/M4A cannot be decoded without it, and the corpus here is mostly
  // FLAC. The engine's own default is the bare string "ffmpeg" (i.e. PATH),
  // which is not a safe assumption on a portable Windows install — so the
  // bundled/ffmpeg-static binary is passed explicitly when there is one, and
  // the route refuses `--decode ffmpeg` when there is not.
  const ff = getFFmpegPath();
  if (ff) args.push('--ffmpeg', ff);
  if (o.only) args.push('--only', o.only);
  if (o.limit > 0) args.push('--limit', String(o.limit));
  if (o.force) args.push('--force');
  return args;
}

export interface ResolvedYue2TrainOptions {
  /** `<latents>/yue2_preprocess.json`. */
  manifest: string;
  outDir: string;
  /** `yue2-lm-<lmType>.gguf`, resolved at spawn time rather than stored, so a
   *  model moved between queueing and running fails with a named file. */
  lmType: string;
  /** Prepended to every caption and the word the trained style is addressed
   *  by. Not optional in practice: without one the adapter has no handle and
   *  the engine only WARNS. */
  trigger: string;
  rank: number;
  alpha: number;
  target: Yue2NarTarget;
  lr: number;
  lrScheduler: 'cosine' | 'constant';
  steps: number;
  warmup: number;
  saveEvery: number;
  logEvery: number;
  gradAccum: number;
  maxGradNorm: number;
  weightDecay: number;
  captionDropout: number;
  tSampling: 'logit-normal' | 'uniform';
  seed: number;
  kvCache: number;
  clipBlock: number;
  /** Continue `<out>/yue2_nar_ckpt.bin`. The engine REFUSES a resume whose
   *  rank/alpha/target/grad-accum/frames/seed, or whose
   *  trigger/lyrics/t-sampling/caption-dropout, differ from the saved run; it
   *  NOTES but allows a changed steps/lr/warmup/lr-scheduler/weight-decay/
   *  max-grad-norm (ace-train.cpp usage). Exactness holds within one machine
   *  and build only — a different GPU, driver, ggml build, base GGUF or
   *  manifest breaks it and cannot be detected. */
  resume: boolean;
  /** For the run manifest and the job's log lines. */
  datasetSlug: string;
  datasetName: string;
}

export function buildYue2TrainArgs(o: ResolvedYue2TrainOptions): string[] {
  const m = resolveYue2TrainModels(o.lmType);
  const args = [
    'yue2-nar-train',
    '--lm', m.lm,
    '--manifest', o.manifest,
    '--out', o.outDir,
    '--name', YUE2_ADAPTER_STEM,
    '--rank', String(o.rank),
    '--alpha', String(o.alpha),
    '--target', o.target,
    '--lr', String(o.lr),
    '--lr-scheduler', o.lrScheduler,
    '--steps', String(o.steps),
    '--warmup', String(o.warmup),
    '--grad-accum', String(o.gradAccum),
    '--max-grad-norm', String(o.maxGradNorm),
    '--weight-decay', String(o.weightDecay),
    '--caption-dropout', String(o.captionDropout),
    '--t-sampling', o.tSampling,
    '--seed', String(o.seed),
    '--kv-cache', String(o.kvCache),
    '--clip-block', String(o.clipBlock),
    '--log-every', String(o.logEvery),
    '--save-every', String(o.saveEvery),
  ];
  // Emitted only when set. The engine takes an empty --trigger and warns, but
  // passing `--trigger ""` and passing nothing are the same run with a
  // different-looking command line, and the second is the honest one.
  if (o.trigger) args.push('--trigger', o.trigger);
  // NOT suppressed when the state file is absent: `--resume` with nothing to
  // resume is the engine's own error to report, and silently dropping the flag
  // would restart a run from step 1 while the UI said "continuing".
  if (o.resume) args.push('--resume');
  // Deliberately NOT emitted: --clip-seconds and --frames. An unset --frames
  // takes the manifest's own clip_frames, which is what the latents were cut
  // at; typing one here would override the cache with a length it was not cut
  // for. --vae-dir is also skipped: the trainer accepts and ignores it.
  return args;
}
