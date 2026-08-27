// trainingApi.ts — API client for the Training Studio (Dataset phase)
//
// Talks to the Node server's /api/training endpoints. The type block below is
// a verbatim copy of the frozen contract shared with
// server/src/services/training/types.ts — keep both sides in sync by hand.

const API_BASE = '/api/training';

// ── Core enums ────────────────────────────────────────────────────────────
export type TagPosition = 'prepend' | 'append' | 'replace';

export type MergePolicy =
  | 'fill_missing'        // default: only write fields currently empty
  | 'overwrite_caption'
  | 'overwrite_lyrics'
  | 'overwrite_all';

export type TrainingJobKind =
  | 'label' | 'enhance-genius' | 'enhance-caption' | 'build'
  | 'preprocess' | 'train-lm' | 'train-dit'
  | 'audition' | 'lm-calibrate' | 'dit-calibrate'
  // MiniMax-Music3. Both GPU-lane and engine-stopping — an MM3 training step
  // peaks at 31.7 GB of a 32 GB card.
  | 'mm3-codes' | 'mm3-train-lm';

export type TrainingJobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

// ─── MiniMax-Music3 training (mirrors server types.ts) ────────────────────

/** GET /api/training/datasets/:id/mm3 */
export interface Mm3Status {
  codesDir: string;
  /** How many .codes files the cache holds (0 = never exported). */
  codes: number;
  /** Which encoder produced them, from codes.json ('' = unknown). */
  encoder: string;
  /** The dataset-wide caption currently in force ('' = none, so the trainer
   *  falls back to per-song .mm3.txt and skips tracks that have none). */
  sharedCaption?: string;
  /** Per-stage, because the two need different files. Non-empty = disable. */
  missingForCodes: string[];
  missingForTrain: string[];
  /** Bases actually installed, best fidelity first. The picker offers only
   *  these — listing a file that is not there moves the failure to spawn time. */
  bases: Mm3BaseInfo[];
  /** The card, from the engine's own reading. 0 = UNKNOWN (engine down or
   *  CPU-only), which must render as a greyed-out estimate rather than as a
   *  "will not fit" warning on a machine that would fit it fine. */
  gpuTotalMb: number;
  /** Base AND rank that fit this card, best fidelity first. Rank is included
   *  because on a 16 GB card nothing fits at the default rank 256, so a base
   *  alone would be advice the user cannot act on. */
  recommended: Mm3Recommendation;
  /** Coefficients for estimateMm3PeakMb below, so the form can re-estimate as
   *  rank and crop length move without carrying its own copy of measurements
   *  taken server-side. */
  vramModel: Mm3VramModel;
  /** previewEverySteps rides along so the form can default the preview cadence
   *  to the checkpoint cadence without hardcoding a second copy of it. */
  defaults: Mm3TrainLmRequest & { maxFrames: number; cropMode: string;
                                  previewEverySteps?: number;
                                  previewEveryMinutes?: number };
  /** Datasets usable as a prior-preservation corpus: they have RVQ codes and
   *  are not this one. Absent on an older server. */
  regCandidates?: Array<{ id: string; name: string; songs: number }>;
}

/** One installed base. `quality` and `lossDelta` are MEASURED against f16 on an
 *  identical seed and crop — see MM3_BASE_FACTS in services/training/mm3Train.ts.
 *  `lossDelta` null means "not measured", not "no difference". */
export interface Mm3BaseInfo {
  id: string;
  file: string;
  bytes: number;
  lossDelta: number | null;
  quality: 'reference' | 'excellent' | 'good' | 'fair' | 'poor';
  /** Measured excess over the fitted model — add it to any local estimate. */
  extraMb: number;
  peakMb: number;
}

export interface Mm3Recommendation {
  base: string;
  rank: number;
  /** Nothing in the catalogue fits this card at any rank on the ladder, so the
   *  values above are a best effort rather than a promise. */
  overBudget: boolean;
}

export interface Mm3VramModel {
  loadedOverheadMb: number;
  perRankMb: number;
  /** AdamW's second momentum buffer. Optional: an older server does not send it. */
  adamwPerRankMb?: number;
  perTokenMb: number;
  /** Quadratic sequence term (attention scores). Optional for the same reason. */
  perTokenSqMb?: number;
  promptTokens: number;
  constMb: number;
  /** Frozen KV prefix, MB per stored column. Optional: an older server does not
   *  send it, and without it a prefix estimate is simply 0. */
  prefixMbPerColumn?: number;
}

/** Peak VRAM in MB. The coefficients come from the server, which fitted them to
 *  measured configurations; this is only the arithmetic. */
/** Extra peak from a frozen KV prefix, MB. Linear, which is the whole point:
 *  the quadratic term above is the backward retaining attention scores, and a
 *  prefix has no backward. Mirrors estimateMm3PrefixMb server-side. */
export function estimateMm3PrefixMb(prefixFrames: number, maxFrames: number, m: Mm3VramModel,
                                    chunk = 256): number {
  if (!(prefixFrames > 0) || !m.prefixMbPerColumn) return 0;
  const qMax = m.promptTokens + prefixFrames;
  const sMax = m.promptTokens + maxFrames + 1;   // +1: the window's lead frame
  const w    = Math.max(sMax, chunk);
  const mask = (qMax + w) * w * 4 / 1048576;
  return Math.round(m.prefixMbPerColumn * (qMax + sMax + chunk) + mask);
}

export function estimateMm3PeakMb(baseBytes: number, rank: number, maxFrames: number,
                                  m: Mm3VramModel,
                                  optimizer: 'muon' | 'adamw' | 'prodigy' = 'adamw',
                                  prefixFrames = 0, prefixChunk = 256): number {
  const loaded  = baseBytes / 1048576 + m.loadedOverheadMb;
  const S       = m.promptTokens + Math.max(0, maxFrames);
  // The S term is QUADRATIC — the backward retains [S, S, heads] attention
  // scores for the live checkpoint segment. Older coefficients had only the
  // linear part, which was exact around S ~ 2642 and 6 GB low at S ~ 5238.
  // Muon one momentum buffer, AdamW two, Prodigy four (m, v, s, x0). Each extra
  // buffer is one adamwPerRankMb. Measured at rank 128: 14.2 / 17.5 / 20.2 GB.
  const extraBuffers = optimizer === 'adamw' ? 1 : optimizer === 'prodigy' ? 3 : 0;
  const perRank = m.perRankMb + extraBuffers * (m.adamwPerRankMb ?? 0);
  const sq      = (m.perTokenSqMb ?? 0) * S * S;
  return Math.round(loaded + perRank * rank + m.perTokenMb * S + sq + m.constMb
                    + estimateMm3PrefixMb(prefixFrames, maxFrames, m, prefixChunk));
}

export interface Mm3CodesRequest {
  maxDuration?: number;
}

/** Every field optional: omitted means the validated recipe, which lives
 *  server-side in services/training/mm3Train.ts and nowhere else. */
export interface Mm3TrainLmRequest {
  rank?: number;
  alpha?: number;
  lr?: number;
  steps?: number;
  saveEvery?: number;
  warmup?: number;
  gradAccum?: number;
  seed?: number;
  maxFrames?: number;
  cropMode?: 'random' | 'beginning' | 'structured';
  cropStartFrac?: number;
  cropEndFrac?: number;
  cropStartTiles?: number;
  /** Acoustic loss through the frozen depth decoder; 0 = old objective (A/B only). */
  depthLossWeight?: number;
  depthLossFrames?: number;
  optimizer?: 'muon' | 'adamw' | 'prodigy';
  /** One caption used for EVERY track, persisted to
   *  <dataset>/_shared-caption.txt. Empty = per-song .mm3.txt files. */
  sharedCaption?: string;
  adapterType?: 'lora' | 'lokr';
  lokrFactor?: number;
  lokrDim?: number;
  lokrAlpha?: number;
  muonLrScale?: number;
  /** Any installed base id ('f16', 'q8_0', 'Q4_K_M', ...). All of them train:
   *  the frozen base is dequantized in-graph per matmul, so only its VRAM and
   *  its fidelity differ, not the mechanism. Validated server-side against what
   *  is on disk. */
  basePrecision?: string;
  /** Fraction of songs withheld for evaluation. 0 disables it, and then the
   *  training loss is the only signal — which cannot tell learning from
   *  memorising. */
  holdout?: number;
  evalEvery?: number;
  trigger?: string;
  /** Prepend `<trigger>, ` to every training caption at prompt assembly, which
   *  is what actually TRAINS the trigger. Default true.
   *
   *  It used to be impossible: `trigger` was recorded in the adapter sidecar and
   *  nothing put it in the prompt, so unless the captions already contained it
   *  the word was never learned. The first SOAD run shipped that way — 14 MOSS
   *  captions, none containing `soad_toxicity` — and rendering with the trigger
   *  then bolted an UNSEEN token sequence onto an in-distribution prompt. It
   *  measurably hurt: the same checkpoint sounded better with the trigger
   *  removed, and tolerated full adapter strength instead of half.
   *
   *  Captions on disk are untouched; the injection happens in memory. */
  triggerPrepend?: boolean;
  /** `song` (default) presents each crop at its TRUE position in the track.
   *  `zero` is the pre-2026-08-23 convention where every crop claimed to be the
   *  song's opening — a train/inference mismatch, since generation always
   *  starts at frame 0. Kept only to reproduce an older run. */
  cropAnchor?: 'song' | 'zero';
  /** Frames of no-grad history in front of each crop. 0 = off. Needs 'song'. */
  prefixFrames?: number;
  prefixChunk?: number;
  prefixSelftest?: boolean;
  /** Mid-run audio previews. Both cadence fields zero = off. */
  preview?: Mm3PreviewOptions;
  /** Prior preservation. Omitted, or no datasetId, = off. */
  regularisation?: Mm3RegularisationOptions;
  /** Which question the run answers. 'steps' runs a fixed number of them.
   *  'loss' runs until the loss reaches `targetLoss` — with `steps` demoted to
   *  a CAP, so a target that never arrives still ends the run. */
  stopMode?: 'steps' | 'loss';
  targetLoss?: number;
  /** 'train' is the mean of the last `targetLossEpochs` completed passes over
   *  the dataset and is always available. 'eval' is the held-out loss — the
   *  number that distinguishes learning from memorising — and needs holdout +
   *  evalEvery. */
  targetLossMetric?: 'train' | 'eval';
  targetLossEpochs?: number;
}

// -- previous runs, and continuing one ---------------------------------------

export interface Mm3RunCheckpoint {
  step: number;
  name: string;
  dir: string;
  loss?: number;
}

export interface Mm3RunSummary {
  runName: string;
  dir: string;
  datasetId?: string;
  datasetName?: string;
  startedAt?: number;
  updatedAt: number;
  /** How many times this directory has been trained into. >1 = resumed. */
  launches: number;
  /** The step cap the last launch was given. */
  configuredSteps: number;
  lastStep: number;
  lastLoss?: number;
  outcome: 'completed' | 'target-reached' | 'halted' | 'failed' | 'unknown';
  failure?: string;
  checkpoints: Mm3RunCheckpoint[];
  best?: { step: number; loss: number };
  targetLoss?: number;
  targetLossMetric?: string;
  /** Absent = this run cannot be continued (no saved optimizer state). */
  resume?: {
    step: number;
    /** 'final' = saved on a clean exit, so continuing loses nothing. 'pause' =
     *  saved at a preview point the run then carried on past. */
    reason: 'pause' | 'final';
    savedAt: number;
    statePath: string;
    /** Steps that would be retrained, because the state is behind the last one
     *  the run reached. */
    behindBy: number;
    rank: number;
    alpha: number;
    optimizer: string;
    adapterType: string;
    samples: number;
    holdout: number;
    bestEval?: number;
  };
  /** 'manifest' = the run's own recorded recipe. 'log' = reconstructed from its
   *  training log, with today's defaults filling what the log does not carry. */
  optionsSource: 'manifest' | 'log' | 'none';
  sizeBytes: number;
  running?: boolean;
}

export interface Mm3ResumeRequest {
  runName: string;
  /** Steps to add on top of where the state sits. Ignored if `steps` is set. */
  addSteps?: number;
  /** The new total cap, absolute. */
  steps?: number;
  saveEvery?: number;
  stopMode?: 'steps' | 'loss';
  targetLoss?: number;
  targetLossMetric?: 'train' | 'eval';
  targetLossEpochs?: number;
  preview?: Mm3PreviewOptions;
}

/** Prior preservation: some steps train against the FROZEN BASE MODEL'S OWN
 *  predictions on an unrelated corpus, so the adapter is punished for changing
 *  its mind about material that has nothing to do with the artist. */
export interface Mm3RegularisationOptions {
  datasetId: string;
  /** Every Nth step. 3 = one prior step per two style steps. */
  every?: number;
  /** Classes kept per position. Measured coverage of the base's probability
   *  mass: 64 -> 89.6%, 128 -> 94.1%, 256 -> 97.0%. */
  topK?: number;
}

/** Mid-run audio previews. Each preview point pauses training for roughly a
 *  minute: the trainer must leave the card entirely for the render (it holds
 *  ~22.6 GB of a 32.6 GB card and a warm MM3 stack is another 11.8 GB), so it
 *  saves its optimizer state, exits, and is relaunched with --resume. */
export interface Mm3PreviewOptions {
  everySteps?: number;
  everyMinutes?: number;
  seconds?: number;
  seed?: number;
  /** Blank = the first HELD-OUT song's caption, trigger prepended. */
  caption?: string;
  lyrics?: string;
  control?: boolean;
  controlCaption?: string;
  baseline?: boolean;
  /** Adapter MLP scale for the preview render, independent of the scale
   *  generation uses. Defaults to 0.65 server-side. */
  scaleMlp?: number;
  scaleAttn?: number;
}

/** One rendered preview; the audio is at
 *  GET /api/training/mm3/preview?run=<run>&file=<file>. */
export interface TrainingPreview {
  id: string;
  step: number;
  totalSteps: number;
  kind: 'artist' | 'control';
  base: boolean;
  file: string;
  seconds: number;
  seed: number;
  caption: string;
  loss?: number;
  bytes: number;
  ms: number;
  ts: number;
}

export type SampleLabelStatus =
  | 'unlabeled'   // no sidecar caption
  | 'labeled'     // sidecar caption present
  | 'pending'     // queued in an active job
  | 'processing'  // currently being labeled
  | 'error';      // last label attempt failed (see `error`)

export type FieldSource = 'sidecar' | 'understand' | 'essentia' | 'genius' | 'llm' | 'tags' | 'filename' | 'user';

// ── Dataset ──────────────────────────────────────────────────────────────
export interface TrainingDatasetSummary {
  id: string;                 // uuid v4
  slug: string;               // filesystem-safe, unique
  name: string;
  sourceDir: string;          // absolute path to the user's audio folder
  recursive: boolean;
  customTag: string;          // trigger word; '' = none
  tagPosition: TagPosition;
  genreRatio: number;         // 0-100
  defaultArtist: string;
  defaultAlbum: string;
  defaultGenre: string;
  defaultLanguage: string;   // language the user KNOWS the corpus is in; overrides understand's guess
  sampleCount: number;
  labeledCount: number;
  excludedCount: number;
  status: 'draft' | 'labeling' | 'labeled' | 'built' | 'error';
  builtAt: string;            // ISO or ''
  datasetJsonPath: string;    // absolute path or ''
  /** Friendly album name from the tracks' embedded tags — '' when unknown. */
  albumName: string;
  createdAt: string;          // ISO
  updatedAt: string;          // ISO
  /** What the dataset has on DISK beyond its row — attached by the list and
   *  detail endpoints, absent on the bare row a PATCH echoes back. */
  assets?: DatasetAssets;
}

/** One trained adapter directory found on disk. */
export interface TrainingAdapterHit {
  path: string;               // absolute adapter run dir
  kind: 'dit' | 'lm';
  detail: string;             // dit-<base> shorthand / LM size — display only
  trainedAt: string;          // ISO or ''
}

/** Per-dataset pipeline progress, read fresh off disk on every request. */
export interface DatasetAssets {
  labeled: boolean;           // at least one caption
  built: boolean;             // dataset.json written
  tensorVariants: number;     // preprocessed variant dirs
  tensorVariantKey: string;   // newest variant, '' when none
  tensorSamples: number;      // .safetensors files in that variant
  ditBase: string;            // base the newest variant was preprocessed against
  lm: TrainingAdapterHit | null;
  dit: TrainingAdapterHit | null;
}

export interface TrainingSample {
  sampleId: string;           // sha1(relPath).slice(0,16) — stable
  relPath: string;            // path relative to sourceDir, forward slashes
  audioPath: string;          // absolute, OS-native separators
  filename: string;           // basename with extension
  sidecarPath: string;        // absolute path of <stem>.txt
  sidecarExists: boolean;
  fileMissing: boolean;       // audio file no longer on disk

  // Sidecar-backed editable fields
  caption: string;
  genre: string;
  bpm: number | null;
  key: string;                // e.g. "C minor"  (sidecar key `key`)
  signature: string;          // e.g. "4/4"
  language: string;           // ISO code or ''
  isInstrumental: boolean;
  lyrics: string;
  customTag: string;          // per-sample override; '' inherits dataset tag
  repeat: number;             // >= 1
  promptOverride: string | null;

  // Studio-private (labels/<sampleId>.json)
  excluded: boolean;
  labelStatus: SampleLabelStatus;
  error: string | null;
  hasAudioCodes: boolean;
  sources: Partial<Record<'caption' | 'lyrics' | 'genre' | 'bpm' | 'key' | 'signature' | 'language', FieldSource>>;

  // Derived / read-only
  duration: number;           // seconds, 0 if unknown
  sizeBytes: number;
  tagArtist: string;          // from embedded audio tags
  tagTitle: string;
  tagAlbum: string;
  labeledAt: string;          // ISO or ''
}

export interface TrainingDatasetDetail extends TrainingDatasetSummary {
  samples: TrainingSample[];
  warnings: string[];         // e.g. "Only 6 samples — 10+ recommended"
  activeJobId: string | null;
  preprocessedVariants: number;   // subdirs of data/training/tensors/<slug> with a preprocess_meta.json
}

// ── Jobs ─────────────────────────────────────────────────────────────────
export interface TrainingJobSummary {
  id: string;
  datasetId: string;
  kind: TrainingJobKind;
  status: TrainingJobStatus;
  total: number;
  done: number;
  failed: number;
  currentSampleId: string | null;
  phase: string;              // free-form: 'essentia' | 'understand' | 'waiting-for-engine' | 'genius' | 'llm' | 'writing' | 'build'
                              // kind==='preprocess' adds: 'engine-stop' | 'loading-models'
                              //                          | 'preprocess' | 'stats' | 'engine-restart'
                              // kind==='train-lm'   adds: 'engine-stop' | 'loading-models' | 'extract'
                              //                          | 'train' | 'export' | 'engine-restart'
                              // kind==='train-dit'  adds: 'engine-stop' | 'loading-models'
                              //                          | 'train' | 'export' | 'engine-restart'
  engineQueueDepth: number;   // # of ace-server jobs ahead of ours; 0 if unknown
  error: string | null;
  createdAt: number;          // epoch ms
  startedAt: number | null;
  finishedAt: number | null;
}

// ── Preprocess ───────────────────────────────────────────────────────────
export type PreprocessDtype   = 'f32' | 'bf16';
export type PreprocessCompat  = 'hotstep' | 'sidestep';
export type PreprocessNormalize = 'none' | 'peak';

export interface PreprocessOptions {
  /** DiT base model name from capabilities.preprocess.ditModels.
   *  Omitted → server uses capabilities.preprocess.defaultDit; if that is
   *  empty the request is rejected with 400. */
  ditModel?: string;
  vaeModel?: string;          // default capabilities.preprocess.defaultVae
  textEncoder?: string;       // default capabilities.preprocess.defaultTextEnc
  sampleIds?: string[];       // omit = every non-excluded, non-missing sample
  maxDuration?: number;       // default 240; 0 = no truncation
  normalize?: PreprocessNormalize;  // default 'peak'
  targetDb?: number;          // default -1.0
  dtype?: PreprocessDtype;    // default 'f32'
  compat?: PreprocessCompat;  // default 'hotstep'
  maxCaptionTokens?: number;  // default 256
  maxLyricTokens?: number;    // default 512
  vaeChunk?: number;          // default 384    (latent frames; ~15.4s tiles — VRAM-bounded)
  vaeOverlap?: number;        // default 64     (latent frames)
  overwrite?: boolean;        // default false
  stopEngine?: boolean;       // default TRUE — stop ace-server for the job
  outputDir?: string;         // default data/training/tensors/<slug>/<variantKey>
}

export interface PreprocessVariantStatus {
  variantKey: string;      // directory name, e.g. 'acestep-v15-base-BF16'
  modelVariant: string;    // exact DiT file name from preprocess_meta.json
  outputDir: string;       // absolute
  createdAt: string;       // ISO, '' if unknown
  compat: string;          // 'hotstep' | 'sidestep' | '' (unknown/legacy)
  dtype: string;           // 'F32' | 'BF16' | ''
  total: number;           // songs the run attempted
  processed: number;
  failed: number;
  cachedCount: number;     // .safetensors files actually present on disk
  staleCount: number;      // cached entries whose source audio size/mtime changed
  missingCount: number;    // current dataset samples with no cache entry
  bytes: number;           // total size of the variant dir
  hasChannelStats: boolean;
}

export interface PreprocessStatus {
  tensorsRoot: string;                    // data/training/tensors/<slug>
  variants: PreprocessVariantStatus[];    // newest createdAt first
}

// ── LM training ──────────────────────────────────────────────────────────
export type LmSize = '0.6B' | '1.7B' | '4B';
export type TrainLmStage = 'extract' | 'train' | 'export';

export interface TrainLmOptions {
  /** Base LM size. 4B trains through the engine's low-VRAM path (per-layer
   *  gradient checkpointing + chunked CE); 0.6B/1.7B stay on the naive path. */
  lmSize?: LmSize;                 // default '0.6B'
  /** Explicit LM GGUF name from capabilities.trainLm.lmModels. Omit to let the
   *  server pick the BF16 model matching lmSize. */
  lmModel?: string;
  /** Which preprocess variant to train from. Omit = the newest variant. */
  variantKey?: string;
  /** Adapter directory stem; final dir is `<adapterName>-<lmSize>`.
   *  Omit = the dataset slug. */
  adapterName?: string;
  targetLoss?: number;             // default 0.1;  0 disables auto-stop
  epochs?: number;                 // default 16 (hard cap)
  rank?: number;                   // default 16
  alpha?: number;                  // default 32
  /** Adapter parameterization (2026-07-30). 'lora' is the shipped path and the
   *  default. 'lokr' writes lokr_weights.safetensors in the SAME LyCORIS layout
   *  the DiT trainer writes, gated by the LK1/LK2 rungs. */
  adapterType?: 'lora' | 'lokr';   // default 'lora'
  /** Optimizer (2026-07-30). 'adamw' is the default and the shipped path.
   *  'muon' puts every 2-D parameter whose short side is >= 16 on
   *  orthogonalized-momentum updates — FOR A LoRA THE SHORT SIDE IS THE RANK,
   *  so at rank 16 every matrix qualifies and at rank 8 none would. On the DiT
   *  Muon measured 1.41x fewer epochs to target; on the LM it is unproven. */
  optimizer?: 'adamw' | 'muon';    // default 'adamw'
  muonLrScale?: number;            // default 20 (the DiT's measured value)
  muonNsSteps?: number;            // default 5
  lokrDim?: number;                // default 512  (adapterType==='lokr' only)
  lokrAlpha?: number;              // default 512; 0 = dim
  lokrFactor?: number;             // default 6
  lokrDecomposeBoth?: boolean;     // default true
  learningRate?: number;           // default 0.0001
  gradAccum?: number;              // default 4
  gradClip?: number;               // default 1.0;  0 disables
  warmupRatio?: number;            // default 0.05
  weightDecay?: number;            // default 0.01
  maxLen?: number;                 // default 0 = auto-fit from free VRAM
  seed?: number;                   // default 42
  lossOnCot?: boolean;             // default true
  order?: 'shuffle' | 'fixed';     // default 'shuffle'
  milestoneStep?: number;          // default 0.1;  0 disables
  milestoneKeep?: number;          // default 6
  stages?: TrainLmStage[];         // default ['extract','train','export']
  overwrite?: boolean;             // default false — re-extract every song
  stopEngine?: boolean;            // default TRUE — stop ace-server for the job
  /** Resume: continue training from this exported adapter run dir
   *  (--init-adapter). Identity hyperparams are adopted from its
   *  lm_train_log.json by the engine; contradictions are refused.
   *  'latest' = newest non-calibrated run of this adapter name, or a scratch
   *  run when there is none. OMITTING THE FIELD MEANS 'latest' (2026-08-12);
   *  send '' to force training from scratch. */
  initAdapter?: string;            // default 'latest' = resume if there is one
  /** Run the post-training calibration job (eval candidates x scales, pick
   *  under guards, bake the winner, write hot_step_eval.json).
   *  Default FALSE (2026-08-12) — only an explicit `true` runs it. */
  calibrate?: boolean;
  /** Let calibration repoint this artist's album preset(s) at the served
   *  adapter. Default TRUE. Meaningless when calibrate is false. */
  calibrateRepoint?: boolean;
  /** Per-layer gradient checkpointing + chunked CE. 'auto' (default) turns it on
   *  for 4B, and for smaller bases only when the naive path would have to skip
   *  full-song samples. */
  lowVram?: 'auto' | 'on' | 'off';   // default 'auto'
  attnHeadBlock?: number;            // default 0 = engine picks
  chunk?: number;                    // default 0 = engine default (128)
  // ── LM speed levers (2026-07-28 plan §2.4). Verbatim contract text. ──────
  /** Projection GEMM dtype. 'f32-window' is the shipped per-segment F32 weight
   *  cast (still the CLI's own default, ace-train.cpp). 'bf16' runs the
   *  projections in BF16 and rewrites the backward activation-gradient nodes;
   *  ~1.65-1.78x on the projection mix, but it changes the trained weights
   *  (BF16 gradient rounding) and requires the CUDA backend + a BF16-native
   *  base + low-VRAM (the engine forces low-VRAM mode itself). Neither of the
   *  other two is a hard fatal any more (2026-07-29): a non-CUDA backend or a
   *  non-BF16-native base each warn and fall back to 'f32-window'
   *  (lm-train-run.h) — the same graceful-fallback treatment as the DiT
   *  mirror. The SERVER default is 'bf16' (training.ts train-lm handler). */
  weights?: 'f32-window' | 'bf16';   // default 'bf16'
  /** Micro-batch size 1..8, or 'auto' (largest of {1,2,4} that fits without
   *  dropping a song). >1 forces low-VRAM mode. */
  batch?: number | 'auto';           // default 1
  /** MUL_MAT activation-gradient formulation (engine/patches/mm-backward.patch).
   *  'outprod' is upstream ggml's out_prod(W, transpose(grad)), which ggml-cuda
   *  implements F32-only. 'mm' emits mul_mat(cont(transpose(W)), grad) — the
   *  same shape and the same maths, but dtype-agnostic, so a BF16 weight uses
   *  BF16 tensor cores (~1.7-1.8x per layer per step on an RTX 5090). Unlike
   *  `weights` it does NOT change which quantity is computed.
   *
   *  THE LM SERVER DEFAULT IS 'outprod', unlike train-dit's 'mm': `weights:
   *  'bf16'` already reaches the same mul_mat backward by rewriting ggml's
   *  out_prod nodes in place (lm-bf16.h), and that rewrite aborts when --bwd mm
   *  leaves it nothing to rewrite. The pair is refused with a 400. */
  bwd?: 'outprod' | 'mm';            // default 'outprod' (train-dit defaults to 'mm')
}

export interface TrainLmEpoch {
  epoch: number;
  loss: number;
  lr: number;
  gradNorm: number;
  ms: number;
}

export interface TrainLmStatus {
  /** '' when the dataset has no preprocessed variant at all. */
  variantKey: string;
  tensorsDir: string;              // absolute, '' if none
  /** lm_codes.jsonl */
  codesPath: string;               // absolute, '' if none
  codesExists: boolean;
  codesCount: number;              // rows in lm_codes.jsonl
  codesStale: number;              // rows whose tensor file changed since extract
  codesMissing: number;            // cached tensors with no codes row
  adapterName: string;             // stem, without the -<size> suffix
  adapterDir: string;              // absolute; the newest trained run, else the artist dir
  adapterExists: boolean;          // adapter_model.safetensors present
  adapterBytes: number;
  lmSize: string;                  // from lm_train_log.json, '' if unknown
  trainedAt: string;               // ISO, '' if unknown
  finalLoss: number;               // -1 if unknown
  bestLoss: number;                // -1 if unknown
  epochsRun: number;               // 0 if unknown
  targetLoss: number;              // -1 if unknown
  stoppedOnTarget: boolean;
  epochs: TrainLmEpoch[];          // [] if unknown
  milestones: Array<{ loss: number; epoch: number; path: string }>;
}

export interface TrainLmCapabilities {
  available: boolean;              // ace-train binary found
  lmModels: string[];              // from the cached /props snapshot (models.lm)
  /** Sizes this build can train: ['0.6B','1.7B','4B']. */
  sizes: LmSize[];
  /** size -> preferred BF16 model name; '' when none is installed. */
  defaultLmBySize: Record<string, string>;
  adaptersRoot: string;            // <adapters>/lm
}

// ── DiT training ─────────────────────────────────────────────────────────
export type DitAdapterType = 'lora' | 'lokr';
export type TrainDitStage  = 'train' | 'export';

export interface TrainDitOptions {
  /** Which preprocess variant to train from. Omit = the newest variant. */
  variantKey?: string;
  /** Adapter directory name under <adapters>/. Omit = the dataset slug. */
  adapterName?: string;
  /** Engine CLI default stays 'lora' for back-compat; the UI form defaults
   *  to 'lokr' (K1). */
  adapterType?: DitAdapterType;    // default 'lora'
  rank?: number;                   // default 128 (adapterType==='lora' only)
  alpha?: number;                  // default 256 (adapterType==='lora' only)
  /** LyCORIS LoKR factors, per Rob's Uber-LoKR-4 preset (K2). Ignored unless
   *  adapterType==='lokr'. */
  lokrDim?: number;                // default 512, range [4,4096]
  lokrAlpha?: number;              // default 512, range (0,8192]; 0 -> dim
  lokrFactor?: number;             // default 6, -1 or [2,64]
  lokrDecomposeBoth?: boolean;     // default true (parity knob; inert at dim 512)
  targetMlp?: boolean;             // default true
  layers?: number;                 // default 0 = auto (top-K depth)
  crop?: number;                   // default 0 = auto-fit
  cropMin?: number;                // default 375
  cropMax?: number;                // default 1250
  targetLoss?: number;             // default 0.1;  0 disables auto-stop
  epochs?: number;                 // default 400 (hard cap)
  learningRate?: number;           // default 0.0005 (lora) / 0.01 (lokr, K2)
  gradAccum?: number;              // default 4 (lora) / 20 (lokr — Side-Step's effective batch 20, which the lokr lr assumes)
  gradClip?: number;               // default 1.0;  0 disables
  warmupRatio?: number;            // default 0.05
  weightDecay?: number;            // default 0.01 (lora) / 0.001 (lokr, K2)
  lossWeighting?: 'none' | 'flow_snr';   // default 'flow_snr' (lora) / 'none' (lokr, K2)
  snrGamma?: number;               // default 5.0
  tBias?: number;                  // default 0.5
  channelBalance?: boolean;        // default true
  timestepMu?: number;             // default -0.4
  timestepSigma?: number;          // default 1.0
  tMin?: number;                   // default 0
  tMax?: number;                   // default 1
  cfgRatio?: number;               // default 0.15
  genreRatio?: number;             // default 30 (percent)
  seed?: number;                   // default 42
  order?: 'shuffle' | 'fixed';     // default 'shuffle'
  milestoneStep?: number;          // default 0.1;  0 disables
  milestoneKeep?: number;          // default 6
  vramReserveMb?: number;          // default 2048
  /** Resume: continue training from this exported adapter run dir
   *  (--init-adapter). Identity hyperparams are adopted from its
   *  dit_train_log.json by the engine; contradictions are refused.
   *  'latest' = newest non-calibrated run of this adapter name, or a scratch
   *  run when there is none. OMITTING THE FIELD MEANS 'latest' (2026-08-13);
   *  send '' to force training from scratch. */
  initAdapter?: string;            // default 'latest' = resume if there is one
  /** Run the post-training DiT calibration job (latent-Frechet eval of
   *  old-vs-new x scales, strict-win pick, bake, sidecar).
   *  Default FALSE (2026-08-13) — only an explicit `true` runs it. */
  calibrate?: boolean;
  /** Let calibration repoint this artist's album preset(s). Default TRUE. */
  calibrateRepoint?: boolean;
  /** Frozen-weight mirror precision. Default 'bf16' (2026-07-29) — keeps the
   *  trainable layers' matmul weights in the base's native BF16 instead of
   *  promoting them to F32, roughly halving the mirror's VRAM. Needs the
   *  patched out_prod in engine/patches/bf16-out-prod.patch and the CUDA
   *  backend; on CPU/Vulkan the engine warns and falls back to 'f32' itself
   *  (dit-train-run.h), so an explicit 'f32' request is the only way to opt
   *  out deliberately. */
  mirror?: 'f32' | 'bf16';         // default 'bf16'
  /** MUL_MAT activation-gradient formulation (engine/patches/mm-backward.patch).
   *  'outprod' is upstream ggml's out_prod(W, transpose(grad)), which ggml-cuda
   *  implements F32-only. 'mm' emits mul_mat(cont(transpose(W)), grad) — the
   *  same shape and the same maths, but dtype-agnostic, so a BF16 mirror uses
   *  BF16 tensor cores (~1.7-1.8x per layer per step on an RTX 5090). SERVER
   *  default is 'mm'. */
  bwd?: 'outprod' | 'mm';          // default 'mm'
  /** Optimizer for the trainable parameters (2026-07-30). 'adamw' is the
   *  shipped path and the default; 'muon' puts every 2-D parameter whose SHORT
   *  side is >= muonMinDim on orthogonalized-momentum (Newton-Schulz) updates
   *  and leaves the rest on AdamW — orthogonalizing a LoKR w1 of [4,5] is
   *  vacuous, so the hybrid is the design, not a compromise. Muon also carries
   *  ONE momentum buffer where AdamW carries two (~870 MB less at LoKR
   *  dim512). NOTE: dit-vram.h still charges for both, so the auto-fit is
   *  conservative rather than wrong. */
  optimizer?: 'adamw' | 'muon';    // default 'adamw'
  /** Multiplies the shared LR schedule, for MUON PARAMETERS ONLY. Muon's update
   *  is normalized by construction, so its LR does not mean AdamW's: measured
   *  parity with AdamW around 20 on a LoKR dim512 run (5 undershoots, 50
   *  overshoots). Inert unless optimizer === 'muon'. */
  muonLrScale?: number;            // default 1.0
  muonMomentum?: number;           // default 0.95
  muonNsSteps?: number;            // default 5, Newton-Schulz iterations
  /** Short-side floor for Muon eligibility; below it a parameter falls to
   *  AdamW. 16 keeps LoKR w1 ([4,5] at factor 6) on AdamW. */
  muonMinDim?: number;             // default 16
  /** Crops per micro-batch, from that many DIFFERENT songs (design §2.2 /
   *  C5). Effective samples/optimizer-step is batch x gradAccum.
   *  Default 1 = OFF (2026-07-29, measured): batching is ~2.5x SLOWER at full
   *  depth on a 32 GB card and ~2.4x faster on shallow/partial-depth runs, so
   *  raising it is a deliberate choice for the latter. */
  batch?: number;                  // default 1, range [1,16]
  /** Gradient-checkpointing segments (design §2.2 / C3). 0 = off (today's
   *  monolithic graph); 1 = auto (engine picks segment count from free
   *  VRAM); 2-32 = that many segments, fixed. */
  ckptSegments?: number;           // default 1 (auto); 0 = off; 2-32 fixed
  stages?: TrainDitStage[];        // default ['train','export']
  overwrite?: boolean;             // default false
  stopEngine?: boolean;            // default TRUE
}

/** Structurally identical to TrainLmEpoch so LossSparkline is reused unedited. */
export interface TrainDitEpoch {
  epoch: number;
  loss: number;
  lr: number;
  gradNorm: number;
  ms: number;
}

export interface TrainDitStatus {
  variantKey: string;              // '' when the dataset has no preprocessed variant
  tensorsDir: string;              // absolute, '' if none
  ditModel: string;                // the variant's base, from preprocess_meta.json
  sampleCount: number;             // usable cached songs in the variant
  channelStats: boolean;           // channel_stats.json present
  adapterName: string;
  adapterDir: string;              // absolute; the newest trained run, else the artist dir
  adapterExists: boolean;          // adapter_model.safetensors OR lokr_weights.safetensors present
  adapterBytes: number;
  /** From dit_train_log.json's config.adapter_type; '' when the log is
   *  missing/unreadable (adapter trained elsewhere, or not trained yet). */
  adapterType: string;
  trainedAt: string;               // ISO, '' if unknown
  finalLoss: number;               // -1 if unknown
  bestLoss: number;                // -1 if unknown
  epochsRun: number;               // 0 if unknown
  targetLoss: number;              // -1 if unknown
  stoppedOnTarget: boolean;
  crop: number;                    // 0 if unknown
  layers: number;                  // 0 if unknown
  partialDepth: boolean;
  epochs: TrainDitEpoch[];         // [] if unknown
  milestones: Array<{ loss: number; epoch: number; path: string }>;
}

export interface TrainDitCapabilities {
  available: boolean;              // ace-train binary found
  adapterTypes: DitAdapterType[];  // ['lora','lokr']
  adaptersRoot: string;            // <adapters>
  /** Minimum total VRAM this build will accept, MB. v1: 16384 (D9). */
  minVramMb: number;
}

// ── Capabilities ─────────────────────────────────────────────────────────
export interface TrainingCapabilities {
  engine: {
    up: boolean;
    ready: boolean;               // from /api/health engine.ready
    understandSupported: boolean; // lm && dit && vae registries all non-empty
    missingModels: string[];      // e.g. ['lm'] — which registries are empty
    queueDepth: number;           // ace-server GET /jobs length
    lmModels: string[];           // LM registry names, for the understand model picker
    defaultLmModel: string;       // server's pick (biggest LM, fast quant) when none is sent
  };
  essentia: { available: boolean; binPath: string };
  genius: { configured: boolean };
  /**
   * Local audio captioning via MOSS-Music-8B. Not a credential check — it is
   * "binary built + weights on disk", and `missing` names which one is absent
   * so the panel can say "download the GGUF" rather than "rebuild the engine".
   */
  moss: { available: boolean; missing: string };
  llm: {
    configured: boolean;
    defaultProvider: string;
    providers: Array<{ id: string; name: string; available: boolean; models: string[]; defaultModel: string }>;
  };
  preprocess: {
    available: boolean;      // ace-train binary found on disk
    binPath: string;         // '' when not found
    ditModels: string[];     // from the cached /props snapshot
    vaeModels: string[];
    textEncoders: string[];
    defaultDit: string;      // first /bf16/i match, else ''
    defaultVae: string;
    defaultTextEnc: string;
    modelsCachedAt: number;  // epoch ms of the snapshot; 0 = never probed
    engineSuspended: boolean;// true while a preprocess job owns the GPU
  };
  trainLm: TrainLmCapabilities;
  trainDit: TrainDitCapabilities;
}

// ── SSE stream ───────────────────────────────────────────────────────────

/** Structured training numbers (L21). Additive member of TrainingStreamEvent —
 *  consumers MUST ignore unknown `metric` values rather than throwing. */
export interface TrainingMetricEvent {
  type: 'metric';
  /** `eval` is held-out loss — the only series that can distinguish learning
   *  from memorising, and therefore the one worth watching. */
  /** `target` announces a target-loss run's stopping line once, at the top of
   *  the run, carrying `loss` (the target) and `totalSteps` (the cap the run
   *  still ends at if the target never arrives). */
  metric: 'vram' | 'data' | 'step' | 'epoch' | 'milestone' | 'eval' | 'target';
  ts: number;
  // epoch / step
  epoch?: number;
  epochs?: number;
  step?: number;
  totalSteps?: number;
  loss?: number;
  lr?: number;
  gradNorm?: number;
  clipScale?: number;
  /** Wall time of THIS step, not elapsed. The direct spill signal. */
  stepMs?: number;
  /** eval: how many fixed held-out crops the number averages. */
  crops?: number;
  etaMs?: number;
  ms?: number;
  best?: boolean;
  // vram
  freeMb?: number;
  totalMb?: number;
  /** VRAM in USE. The ACE path reports free/total; MM3 reports used directly,
   *  because "how close to the ceiling is this run" is the number that decides
   *  whether a step takes 4 s or spills to host memory and takes ten times
   *  that. */
  usedMb?: number;
  estMb?: number;
  vramMb?: number;
  maxLen?: number;
  // data
  samples?: number;
  skippedLong?: number;
  stepsPerEpoch?: number;
  loraParams?: number;
  // milestone
  path?: string;
  // train-dit additions (§2.6). Optional fields on the EXISTING interface —
  // deliberately not a new union member, so every consumer keeps working.
  crop?: number;      // latent frames in the active crop window
  layers?: number;    // trained decoder-layer count (top-K)
  t?: number;         // last micro-step timestep (telemetry)
  ma5?: number;       // 5-epoch moving average (the target-loss quantity)
  rawLoss?: number;   // unweighted MSE, for telemetry
  // train-lm low-VRAM additions (4B plan §2.2). All optional, additive.
  mode?: string;      // 'naive' | 'lowvram'
  baseMb?: number;    // resident base weights
  ckptMb?: number;    // per-layer checkpoint buffers
  segPeakMb?: number; // transient peak of one segment graph
  // train-lm speed levers (2026-07-28 plan §2.2/§2.4). All optional, additive.
  // §2.4's verbatim three:
  weights?: string; batch?: number; padPct?: number;
  // …plus the rest of §2.2's JSONL fields, which the server relay forwards by
  // name and therefore need a home on this interface.
  batchSource?: string;  // vram: 'user' | 'auto'
  batches?: number;      // data: micro-batches per epoch
  padTokens?: number;    // data: padding tokens added by batching
  // NB `samples` (declared above under `data`) is reused by `step` as the
  // per-optimizer-step sample count (micro * B_cur); no new field needed.
}

export type TrainingStreamEvent =
  | { type: 'job'; job: TrainingJobSummary }                       // first event on connect + on any status change
  | {
      type: 'progress'; done: number; total: number; failed: number;
      phase: string; currentSampleId: string | null; engineQueueDepth: number;
    }
  | {
      type: 'sample'; sampleId: string; status: SampleLabelStatus;
      sample?: TrainingSample; error?: string;                      // `sample` present on success
    }
  | { type: 'log'; level: 'info' | 'warn' | 'error'; message: string; ts: number }
  | { type: 'preview'; run: string; preview: TrainingPreview }
  | TrainingMetricEvent
  | { type: 'status'; status: TrainingJobStatus; error?: string };  // terminal; server closes after this

// ── Request / response shapes ────────────────────────────────────────────

export interface ScanPreview {
  root: string;
  audioFiles: number;
  withSidecar: number;
  withCaption: number;
  hasDatasetJson: boolean;
  extensions: Record<string, number>;   // { ".flac": 12, ".mp3": 3 }
  sampleNames: string[];                // first 10 basenames
}

export interface CreateDatasetInput {
  name: string;
  sourceDir: string;
  recursive?: boolean;
  customTag?: string;
  tagPosition?: TagPosition;
  genreRatio?: number;
  defaultArtist?: string;
  defaultAlbum?: string;
  defaultGenre?: string;
  defaultLanguage?: string;  // default 'english'
}

export type PatchDatasetInput = Partial<{
  name: string;
  customTag: string;
  tagPosition: TagPosition;
  genreRatio: number;
  defaultArtist: string;
  defaultAlbum: string;
  defaultGenre: string;
  defaultLanguage: string;
  recursive: boolean;
}>;

export type PatchSampleInput = Partial<{
  caption: string;
  genre: string;
  bpm: number | null;
  key: string;
  signature: string;
  language: string;
  isInstrumental: boolean;
  lyrics: string;
  customTag: string;
  repeat: number;
  promptOverride: string | null;
  excluded: boolean;
}>;

export type BulkSetInput = Partial<{
  excluded: boolean;
  isInstrumental: boolean;
  genre: string;
  customTag: string;
  language: string;
  repeat: number;
}>;

export interface BulkResult {
  updated: number;
  failed: Array<{ sampleId: string; error: string }>;
}

export interface LabelOptions {
  sampleIds?: string[];
  scope?: 'all' | 'unlabeled';
  useEssentia?: boolean;   // default true  — local BPM/key
  useGenius?: boolean;     // default false — canonical lyrics (needs token)
  useCaption?: boolean;    // default false — LLM caption+genre (audio-grounded on gemini)
  useUnderstand?: boolean; // default false — LEGACY /understand path
  mergePolicy?: MergePolicy;
  caption?: { provider?: string; model?: string };
  understand?: {
    lmModel?: string;
    synthModel?: string;
    lmTemperature?: number;
    lmTopP?: number;
    lmTopK?: number;
    seed?: number;
  };
}

export interface GeniusOptions {
  sampleIds?: string[];
  mergePolicy?: MergePolicy;
  artist?: string;
  album?: string;
  sanitizeHeaders?: boolean;
}

export interface CaptionOptions {
  sampleIds?: string[];
  provider?: string;
  model?: string;
  mergePolicy?: MergePolicy;
  includeLyricsExcerpt?: boolean;
  temperature?: number;
}

// ─── Lyric Studio export ──────────────────────────────────────────────────
//
// A labeled dataset already holds everything a Lyric Studio artist/album entry
// needs (Genius lyrics, LLM caption+genre, Essentia bpm/key/signature), so a
// dataset can be exported straight into the lireek tables. Artist/album are
// DETECTED (embedded-tag majority vote → dataset defaults → folder name) and
// user-overridable; an existing artist+album set is updated in place, never
// duplicated. Trained adapters for the dataset are offered as the album preset.

export type LyricStudioFieldSource = 'tags' | 'default' | 'filename' | 'folder' | 'dataset-name';

export interface LyricStudioExportSong {
  sampleId: string;
  title: string;
  album: string;              // per-song album (own tag first, set album else)
  hasCaption: boolean;
}

export interface LyricStudioAdapterHit {
  path: string;               // absolute adapter run dir (folder-only presets)
  kind: 'dit' | 'lm';
  detail: string;             // dit-<base> shorthand / LM size — display only
  trainedAt: string;          // ISO or ''
}

export interface LyricStudioExportPreview {
  artist: string;             // detected, pre-fills the override field
  album: string;
  artistSource: LyricStudioFieldSource;
  albumSource: LyricStudioFieldSource;
  songs: LyricStudioExportSong[];   // includable songs, dataset order
  skippedInstrumental: number;
  skippedNoLyrics: number;
  skippedExcluded: number;
  existingArtistId: number | null;    // COLLATE NOCASE match on the DETECTED names;
  existingLyricsSetId: number | null; // the commit re-resolves after overrides
  existingSongCount: number;
  ditAdapter: LyricStudioAdapterHit | null;
  lmAdapter: LyricStudioAdapterHit | null;
  geniusConfigured: boolean;  // album/artist art fetch will be attempted
}

export interface LyricStudioExportInput {
  artist?: string;            // override; omit/'' = detected value
  album?: string;
  linkAdapters?: boolean;     // default true — write the album preset from trained adapters
}

export interface LyricStudioExportResult {
  artistId: number;
  lyricsSetId: number;
  updatedExisting: boolean;
  songCount: number;
  presetUpdated: boolean;
  imageUrl: string;           // album art fetched during export, '' if none
}

// ── Batch pipeline (multi-folder import + auto-chained stages) ─────────────
//
// Verbatim copy of docs/plans/2026-07-28-training-batch-pipeline.md §2.1 /
// server/src/services/training/types.ts — keep both sides in sync by hand.

export type PipelineStage =
  'label' | 'build' | 'preprocess' | 'train-dit' | 'train-lm' | 'lyric-studio';
export type PipelineStatus = 'running' | 'paused' | 'done' | 'failed' | 'cancelled';
export type PipelineItemStatus = 'pending' | 'creating' | 'running' | 'done' | 'failed' | 'cancelled';

export interface PipelineFolderSpec {
  sourceDir: string;
  /** Dataset name; omit = folder basename. */
  name?: string;
  /** Trigger word; omit = slugified name. */
  customTag?: string;
}

/** Per-run label-stage overrides applied to every dataset in the pipeline, on
 *  top of the stored label defaults. Lets a bulk re-caption skip Genius and
 *  Essentia and force the merge policy it needs. Absent field = default. */
export interface PipelineLabelOptions {
  /** 'all' re-labels every sample; default 'unlabeled' only fills gaps. */
  scope?: 'all' | 'unlabeled';
  useEssentia?: boolean;
  useGenius?: boolean;
  useCaption?: boolean;
  mergePolicy?: MergePolicy;
}

export interface StartPipelineInput {
  folders: PipelineFolderSpec[];
  /** Which stages to run, in canonical order. Omit = all five. */
  stages?: PipelineStage[];
  labelOptions?: PipelineLabelOptions;
}

export interface PipelineStageResult {
  stage: PipelineStage;
  jobId: string;          // '' if the stage never started
  status: PipelineItemStatus;
  error: string | null;
  startedAt: number | null;
  finishedAt: number | null;
}

export interface PipelineItem {
  sourceDir: string;
  name: string;
  datasetId: string;      // '' until created/resolved
  reusedExisting: boolean;
  status: PipelineItemStatus;
  currentStage: PipelineStage | null;
  stages: PipelineStageResult[];
  error: string | null;
}

export interface PipelineSummary {
  id: string;
  status: PipelineStatus;
  stages: PipelineStage[];
  items: PipelineItem[];
  createdAt: number;
  finishedAt: number | null;
  /** Pause asked for but the in-flight stage is still finishing; status flips
   *  to 'paused' once the runner parks. Absent on pre-pause snapshots. */
  pauseRequested?: boolean;
}

export interface TrainingDefaults {
  label: Record<string, unknown>;
  preprocess: Record<string, unknown>;
  trainLm: Record<string, unknown>;
  trainDit: Record<string, unknown>;
}

/** Canonical stage order, mirrors server pipelineRunner.ts's PIPELINE_STAGES —
 *  not itself part of the frozen contract, but keeping one copy avoids the UI
 *  re-deriving order from whatever sequence a checkbox row happens to render. */
export const PIPELINE_STAGES: readonly PipelineStage[] =
  ['label', 'build', 'preprocess', 'train-dit', 'train-lm', 'lyric-studio'];

// ── fetch helpers ────────────────────────────────────────────────────────

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API error: ${res.status}`);
  }
  return res.json();
}

function jsonBody(body: unknown): RequestInit {
  return { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) };
}

// ── Capabilities ─────────────────────────────────────────────────────────

export async function getCapabilities(): Promise<TrainingCapabilities> {
  return request<TrainingCapabilities>('/capabilities');
}

// ── Scan preview ─────────────────────────────────────────────────────────

export async function scanPreview(path: string, recursive = true): Promise<ScanPreview> {
  return request<ScanPreview>(`/scan-preview?path=${encodeURIComponent(path)}&recursive=${recursive ? 1 : 0}`);
}

// ── Dataset CRUD ─────────────────────────────────────────────────────────

export async function listDatasets(): Promise<TrainingDatasetSummary[]> {
  const data = await request<{ datasets: TrainingDatasetSummary[] }>('/datasets');
  return data.datasets;
}

export async function createDataset(input: CreateDatasetInput): Promise<TrainingDatasetDetail> {
  const data = await request<{ dataset: TrainingDatasetDetail }>('/datasets', { method: 'POST', ...jsonBody(input) });
  return data.dataset;
}

export async function getDataset(id: string): Promise<TrainingDatasetDetail> {
  return request<TrainingDatasetDetail>(`/datasets/${encodeURIComponent(id)}`);
}

export async function patchDataset(id: string, patch: PatchDatasetInput): Promise<TrainingDatasetSummary> {
  const data = await request<{ dataset: TrainingDatasetSummary }>(
    `/datasets/${encodeURIComponent(id)}`, { method: 'PATCH', ...jsonBody(patch) },
  );
  return data.dataset;
}

export async function rescanDataset(id: string): Promise<TrainingDatasetDetail> {
  return request<TrainingDatasetDetail>(`/datasets/${encodeURIComponent(id)}/rescan`, { method: 'POST' });
}

export async function deleteDataset(id: string): Promise<void> {
  await request<{ ok: boolean }>(`/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

// ── Sample edits ─────────────────────────────────────────────────────────

export async function patchSample(id: string, sampleId: string, patch: PatchSampleInput): Promise<TrainingSample> {
  const data = await request<{ sample: TrainingSample }>(
    `/datasets/${encodeURIComponent(id)}/samples/${encodeURIComponent(sampleId)}`,
    { method: 'PATCH', ...jsonBody(patch) },
  );
  return data.sample;
}

export async function bulkSetSamples(id: string, sampleIds: string[], set: BulkSetInput): Promise<BulkResult> {
  return request<BulkResult>(
    `/datasets/${encodeURIComponent(id)}/samples/bulk`,
    { method: 'POST', ...jsonBody({ sampleIds, set }) },
  );
}

/** Direct audio URL for the preview player — never routed through the playback store. */
export function sampleAudioUrl(id: string, sampleId: string): string {
  return `${API_BASE}/datasets/${encodeURIComponent(id)}/samples/${encodeURIComponent(sampleId)}/audio`;
}

/** MM3 Structured Caption (`<stem>.mm3.txt`) — served on demand, not in the samples list. */
export async function getSampleMm3(id: string, sampleId: string): Promise<{ text: string }> {
  return request(`/datasets/${encodeURIComponent(id)}/samples/${encodeURIComponent(sampleId)}/mm3`);
}

export async function saveSampleMm3(id: string, sampleId: string, text: string): Promise<{ text: string }> {
  return request(
    `/datasets/${encodeURIComponent(id)}/samples/${encodeURIComponent(sampleId)}/mm3`,
    { method: 'PUT', ...jsonBody({ text }) },
  );
}

// ── Jobs ─────────────────────────────────────────────────────────────────

export async function startLabel(id: string, opts: LabelOptions): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(`/datasets/${encodeURIComponent(id)}/label`, { method: 'POST', ...jsonBody(opts) });
}

export async function startGenius(id: string, opts: GeniusOptions): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(`/datasets/${encodeURIComponent(id)}/enhance/genius`, { method: 'POST', ...jsonBody(opts) });
}

export async function startCaption(id: string, opts: CaptionOptions): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(`/datasets/${encodeURIComponent(id)}/enhance/caption`, { method: 'POST', ...jsonBody(opts) });
}

export async function startBuild(id: string, outputPath?: string): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(
    `/datasets/${encodeURIComponent(id)}/build`,
    { method: 'POST', ...jsonBody(outputPath ? { outputPath } : {}) },
  );
}

export async function startPreprocess(id: string, opts: PreprocessOptions): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(
    `/datasets/${encodeURIComponent(id)}/preprocess`,
    { method: 'POST', ...jsonBody(opts) },
  );
}

export async function getPreprocessStatus(id: string): Promise<PreprocessStatus> {
  return request<PreprocessStatus>(`/datasets/${encodeURIComponent(id)}/preprocess`);
}

export async function deletePreprocessVariant(id: string, variantKey: string): Promise<void> {
  await request<{ ok: boolean }>(
    `/datasets/${encodeURIComponent(id)}/preprocess/${encodeURIComponent(variantKey)}`,
    { method: 'DELETE' },
  );
}

// ── MiniMax-Music3 training ──────────────────────────────────────────────

export async function getMm3Status(id: string): Promise<Mm3Status> {
  return request<Mm3Status>(`/datasets/${encodeURIComponent(id)}/mm3`);
}

export async function startMm3Codes(id: string, opts: Mm3CodesRequest = {}): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(
    `/datasets/${encodeURIComponent(id)}/mm3-codes`,
    { method: 'POST', ...jsonBody(opts) },
  );
}

/** Checkpoints land in the MM3 adapter folder, so a finished run shows up in
 *  the generation panel's adapter picker with no install step. */
export async function startMm3TrainLm(
  id: string, opts: Mm3TrainLmRequest = {},
): Promise<{ jobId: string; runName: string; outDir: string }> {
  return request<{ jobId: string; runName: string; outDir: string }>(
    `/datasets/${encodeURIComponent(id)}/mm3-train-lm`,
    { method: 'POST', ...jsonBody(opts) },
  );
}

/** Previous MM3 LM runs for a dataset, newest first. Read off disk each time —
 *  a run directory is the source of truth, not a database row. */
export async function listMm3Runs(id: string): Promise<{ runs: Mm3RunSummary[]; busy: boolean }> {
  return request<{ runs: Mm3RunSummary[]; busy: boolean }>(
    `/datasets/${encodeURIComponent(id)}/mm3-runs`,
  );
}

/** Continue a previous run: same adapter directory, same recipe, more steps.
 *  Everything not named here comes from what that run was trained with. */
export async function resumeMm3TrainLm(
  id: string, opts: Mm3ResumeRequest,
): Promise<{ jobId: string; runName: string; outDir: string; from: number; steps: number;
             optionsSource: string }> {
  return request<{ jobId: string; runName: string; outDir: string; from: number; steps: number;
                   optionsSource: string }>(
    `/datasets/${encodeURIComponent(id)}/mm3-resume-lm`,
    { method: 'POST', ...jsonBody(opts) },
  );
}

export async function startTrainLm(id: string, opts: TrainLmOptions): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(
    `/datasets/${encodeURIComponent(id)}/train-lm`,
    { method: 'POST', ...jsonBody(opts) },
  );
}

export async function getTrainLmStatus(
  id: string,
  q?: { variantKey?: string; adapterName?: string; lmSize?: LmSize },
): Promise<TrainLmStatus> {
  const params = new URLSearchParams();
  if (q?.variantKey) params.set('variantKey', q.variantKey);
  if (q?.adapterName) params.set('adapterName', q.adapterName);
  if (q?.lmSize) params.set('lmSize', q.lmSize);
  const qs = params.toString();
  return request<TrainLmStatus>(
    `/datasets/${encodeURIComponent(id)}/train-lm${qs ? `?${qs}` : ''}`,
  );
}

export async function startTrainDit(id: string, opts: TrainDitOptions): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(
    `/datasets/${encodeURIComponent(id)}/train-dit`,
    { method: 'POST', ...jsonBody(opts) },
  );
}

export async function getTrainDitStatus(
  id: string,
  q?: { variantKey?: string; adapterName?: string },
): Promise<TrainDitStatus> {
  const params = new URLSearchParams();
  if (q?.variantKey) params.set('variantKey', q.variantKey);
  if (q?.adapterName) params.set('adapterName', q.adapterName);
  const qs = params.toString();
  return request<TrainDitStatus>(
    `/datasets/${encodeURIComponent(id)}/train-dit${qs ? `?${qs}` : ''}`,
  );
}

export async function getJob(jobId: string): Promise<TrainingJobSummary> {
  return request<TrainingJobSummary>(`/jobs/${encodeURIComponent(jobId)}`);
}

export async function listJobs(datasetId?: string): Promise<TrainingJobSummary[]> {
  const qs = datasetId ? `?datasetId=${encodeURIComponent(datasetId)}` : '';
  const data = await request<{ jobs: TrainingJobSummary[] }>(`/jobs${qs}`);
  return data.jobs;
}

export async function cancelJob(jobId: string): Promise<void> {
  await request<{ ok: boolean }>(`/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
}

export const jobStreamUrl = (jobId: string) => `${API_BASE}/jobs/${encodeURIComponent(jobId)}/stream`;

/** Built dataset.json, parsed. */
export async function getDatasetJson(id: string): Promise<{ path: string; builtAt: string; dataset: unknown }> {
  return request<{ path: string; builtAt: string; dataset: unknown }>(`/datasets/${encodeURIComponent(id)}/dataset-json`);
}

// ── Lyric Studio export ──────────────────────────────────────────────────

export async function getLyricStudioPreview(id: string): Promise<LyricStudioExportPreview> {
  return request<LyricStudioExportPreview>(`/datasets/${encodeURIComponent(id)}/lyric-studio`);
}

export async function exportToLyricStudio(
  id: string, input: LyricStudioExportInput,
): Promise<LyricStudioExportResult> {
  return request<LyricStudioExportResult>(
    `/datasets/${encodeURIComponent(id)}/lyric-studio`,
    { method: 'POST', ...jsonBody(input) },
  );
}

// ─── Codes audition (pure-LM preview) ─────────────────────────────────────

/** Which half of an A/B a preview file belongs to. A single-sided audition
 *  (sample codes, or a milestone with no baseline) always uses 'base'. */
export type AuditionSlot = 'base' | 'adapter';

export type AuditionKind = 'ab' | 'sample' | 'milestone';

/** One LM run to perform. Both sides of an A/B are identical except lmAdapter. */
export interface AuditionSideSpec {
  slot: AuditionSlot;
  label: string;            // free text shown on the player, e.g. 'Base 4B' / 'abba-4B'
  lmAdapter: string;        // '' = base LM; else a registry name or a path inside adapters/lm
  lmAdapterScale: number;   // default 1.0, clamp 0..2
}

export interface AuditionSideResult {
  slot: AuditionSlot;
  label: string;
  lmAdapter: string;
  lmAdapterScale: number;
  ok: boolean;
  error: string;            // '' when ok
  audioUrl: string;         // '' when !ok — /api/training/previews/<id>/<slot>
  codesCount: number;       // number of 5 Hz codes the LM emitted
  codesSha1: string;        // sha1 of the raw audio_codes string — the determinism receipt
  durationSec: number;      // codesCount / 5, rounded to 0.1
  caption: string;          // the plan the LM handed back (C13)
  lyrics: string;
  bpm: number;              // 0 = not reported
  keyscale: string;
  timesignature: string;
  lmMs: number;             // wall clock of the /lm job
  decodeMs: number;         // wall clock of the /codes-decode job
  /** Opt-in DiT render of this side's codes (same DiT/seed/steps both sides,
   *  NO sound adapter — the LM adapter stays the only variable). Absent on
   *  previews recorded before the feature and when renderDit was off. */
  renderUrl?: string;       // /api/training/previews/<id>/<slot>-render
  renderMs?: number;        // wall clock of the /synth render
  renderError?: string;     // render failed; the codes sketch above is still valid
  /** Opt-in SECOND render through the dataset's trained DiT adapter — with
   *  renderUrl this gives the 2×2 {base LM, LM adapter} × {bare DiT, DiT adapter}. */
  renderAdapterUrl?: string;   // /api/training/previews/<id>/<slot>-render-adapter
  renderAdapterMs?: number;
  renderAdapterError?: string;
}

export interface AuditionPreview {
  previewId: string;        // uuid v4
  datasetId: string;
  kind: AuditionKind;
  createdAt: string;        // ISO
  seed: number;             // the lm_seed actually used — never -1
  caption: string;          // the prompt that went IN
  lyrics: string;           // '' = the LM was asked to write its own
  durationSec: number;      // requested duration
  lmModel: string;
  ditModel: string;         // the DiT whose FSQ detokenizer decoded the codes
  vaeModel: string;
  sampleId: string;         // '' unless kind === 'sample' or the prompt came from a sample
  variantKey: string;       // '' when the prompt was free text
  sides: AuditionSideResult[];
  /** Set when the sides carry DiT renders — reproducibility receipt. */
  renderDitModel?: string;
  renderSteps?: number;
  /** The resolved DiT-adapter run dir the *-render-adapter files used. */
  renderDitAdapter?: string;
  /** Mirrored-generation receipts — what "Send to Custom-Gen" replays.
   *  Absent on previews recorded before the feature. */
  captionInput?: string;    // the caption BEFORE the trigger tag was applied
  bpm?: number;             // metadata pins (0/'' = LM predicted)
  keyscale?: string;
  timesignature?: string;   // numerator form ('4')
  lmTemperature?: number;
  lmTopP?: number;
  lmCfgScale?: number;
  lmRepPenalty?: number;
}

export interface AuditionOptions {
  sides: AuditionSideSpec[];      // 1..2 entries, slots must be distinct
  caption: string;                // required, 1..4000 chars
  lyrics?: string;                // default ''
  seed?: number;                  // >= 0; omitted or < 0 → the server picks one and records it
  durationSec?: number;           // default 30, clamp 10..300
  lmModel?: string;               // default: the newest variant's LM, else registry[0]
  ditModel?: string;              // default: variantDitModel(slug, variantKey)
  vaeModel?: string;              // default: engine's resolve_name default
  variantKey?: string;            // default: newestVariantKey(slug)
  sampleId?: string;              // when set, caption/lyrics default from its lm_codes.jsonl row
  /** Explicit metadata pins (Lyric Studio prompt source) — force_fields'd into
   *  the CoT like the sample-row pins; win over the row when both present. */
  bpm?: number;
  keyscale?: string;
  timesignature?: string;
  temperature?: number;           // default 0.85, clamp 0.1..2
  topP?: number;                  // default 0.9,  clamp 0.05..1
  cfgScale?: number;              // default 2.0,  clamp 0..10
  repPenalty?: number;            // default 1.0,  clamp 1..1.5
  format?: 'wav16' | 'mp3';       // default 'wav16' (C10)
  coResident?: boolean;           // default true (C15)
  kind?: 'ab' | 'milestone';      // default 'ab'
  /** Also render each side's codes through a DiT so the A/B is audible as
   *  MUSIC, not a 5 Hz sketch. Same DiT, same seed, same steps on both sides
   *  and never a sound adapter — the LM adapter stays the only variable. */
  renderDit?: boolean;            // default false
  renderSteps?: number;           // default 8, clamp 2..60
  renderDitModel?: string;        // default: newest installed xl-turbo, else the detok DiT
  /** With renderDit: ALSO render each side through the dataset's latest trained
   *  DiT adapter (2×2 matrix). Server resolves the adapter and pins every
   *  render to its training base; 409 when nothing is trained. */
  renderDitAdapter?: boolean;     // default false
  renderDitAdapterName?: string;  // default: the dataset slug
}

export interface AuditionListResponse {
  previews: AuditionPreview[];    // newest first, max 20
  engineReady: boolean;
  engineSuspended: boolean;       // true while a preprocess/train job owns the GPU
}

/** Sync response of the sample-codes audition. */
export interface SampleAuditionResponse {
  preview: AuditionPreview;       // kind 'sample', exactly one side, slot 'base'
  source: 'lm_codes' | 'label';   // which stored-codes source was used (C12 / §5.3)
}

/** POST the A/B (or milestone) audition. Returns the training job id — the two
 *  `/lm` runs are far too slow for a synchronous response (C6). */
export async function startAudition(id: string, opts: AuditionOptions): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(
    `/datasets/${encodeURIComponent(id)}/audition`,
    { method: 'POST', ...jsonBody(opts) },
  );
}

/** One Lyric Studio generation, offered as an audition prompt. */
export interface LsGeneration {
  id: number;
  title: string;
  caption: string;
  lyrics: string;
  bpm: number;          // 0 = unknown
  key: string;          // '' = unknown
  duration: number;     // seconds, 0 = unknown
  createdAt: string;
}

export interface LsGenerationsResponse {
  lyricsSetId: number;  // 0 = this dataset has no linked Lyric Studio album
  artist: string;
  album: string;
  generations: LsGeneration[];
}

/** Generated lyrics of the dataset's linked Lyric Studio album (persisted
 *  link, falling back to artist/album detection which then persists). */
export async function getLsGenerations(id: string): Promise<LsGenerationsResponse> {
  return request<LsGenerationsResponse>(
    `/datasets/${encodeURIComponent(id)}/ls-generations`,
  );
}

/** Preview history for one dataset, newest first. */
export async function getAuditions(id: string, limit = 20): Promise<AuditionListResponse> {
  return request<AuditionListResponse>(
    `/datasets/${encodeURIComponent(id)}/audition?limit=${encodeURIComponent(String(limit))}`,
  );
}

/** Decode a sample's STORED codes. Synchronous (C7) — no `/lm` call happens. */
export async function auditionSample(
  id: string,
  sampleId: string,
  format?: 'wav16' | 'mp3',
): Promise<SampleAuditionResponse> {
  return request<SampleAuditionResponse>(
    `/datasets/${encodeURIComponent(id)}/samples/${encodeURIComponent(sampleId)}/audition`,
    { method: 'POST', ...jsonBody(format ? { format } : {}) },
  );
}

/** Range-served preview audio. `AuditionSideResult.audioUrl` already carries
 *  this string; this helper exists for callers that only hold id + slot. */
export function previewAudioUrl(previewId: string, slot: AuditionSlot): string {
  return `${API_BASE}/previews/${encodeURIComponent(previewId)}/${encodeURIComponent(slot)}`;
}

// ─── Batch pipeline ────────────────────────────────────────────────────────

export async function startPipeline(input: StartPipelineInput): Promise<PipelineSummary> {
  const data = await request<{ pipeline: PipelineSummary }>('/pipeline', { method: 'POST', ...jsonBody(input) });
  return data.pipeline;
}

export async function listPipelines(): Promise<PipelineSummary[]> {
  const data = await request<{ pipelines: PipelineSummary[] }>('/pipeline');
  return data.pipelines;
}

export async function getPipeline(id: string): Promise<PipelineSummary> {
  return request<PipelineSummary>(`/pipeline/${encodeURIComponent(id)}`);
}

export async function pausePipeline(id: string): Promise<void> {
  await request<{ ok: boolean }>(`/pipeline/${encodeURIComponent(id)}/pause`, { method: 'POST' });
}

export async function resumePipeline(id: string): Promise<void> {
  await request<{ ok: boolean }>(`/pipeline/${encodeURIComponent(id)}/resume`, { method: 'POST' });
}

export async function cancelPipeline(id: string): Promise<void> {
  await request<{ ok: boolean }>(`/pipeline/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

export async function getTrainingDefaults(): Promise<TrainingDefaults> {
  return request<TrainingDefaults>('/defaults');
}

export async function putTrainingDefaults(patch: Partial<TrainingDefaults>): Promise<TrainingDefaults> {
  return request<TrainingDefaults>('/defaults', { method: 'PUT', ...jsonBody(patch) });
}
