// training/types.ts — Dataset Studio API contract (FROZEN)
//
// Single source of truth for every type crossing the /api/training boundary.
// ui/src/services/trainingApi.ts holds a verbatim copy of the same text — keep
// the two in sync by hand; they are deliberately not shared at build time.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §2

// ─── Core enums ────────────────────────────────────────────────────────────
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
  // MiniMax-Music3 (docs/plans/2026-08-20-mm3-training-server-design.md §2.1).
  // Both are GPU-lane and both stop the engine: an MM3 training step peaks at
  // 31.7 GB of a 32 GB card, so nothing else may be resident.
  | 'mm3-codes' | 'mm3-train-lm';

export type TrainingJobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

// ─── MiniMax-Music3 training ──────────────────────────────────────────────

/** POST /api/training/datasets/:id/mm3-codes — audio -> RVQ codes. */
export interface Mm3CodesRequest {
  /** Cap each track before encoding (seconds). 0 = whole track. */
  maxDuration?: number;
}

/** POST /api/training/datasets/:id/mm3-train-lm.
 *  Every field is optional: omitted means the validated recipe
 *  (MM3_LM_DEFAULTS in services/training/mm3Train.ts), which is the single
 *  place those numbers live. */
export interface Mm3TrainLmRequest {
  rank?: number;
  alpha?: number;
  lr?: number;
  steps?: number;
  saveEvery?: number;
  warmup?: number;
  gradAccum?: number;
  seed?: number;
  /** Crop length in FRAMES at 25 fps. Note this is not the VRAM dial it looks
   *  like: an MM3 prompt is ~1,100 tokens, so the sequence is prompt-dominated. */
  maxFrames?: number;
  /** `beginning` reproduces the intros-only failure on purpose; do not ship it. */
  cropMode?: 'random' | 'beginning' | 'structured';
  /** `structured` only: the share of steps pinned to frame 0 and the share
   *  pinned flush to the track's end. */
  cropStartFrac?: number;
  cropEndFrac?: number;
  optimizer?: 'muon' | 'adamw' | 'prodigy';
  /** 'lora' (default) or 'lokr'. LoKr writes lokr_weights.safetensors
   *  instead of a PEFT directory. */
  /** One caption used for EVERY track, persisted to
   *  <dataset>/_shared-caption.txt. Empty = per-song .mm3.txt files. */
  sharedCaption?: string;
  adapterType?: 'lora' | 'lokr';
  lokrFactor?: number;
  lokrDim?: number;
  lokrAlpha?: number;
  muonLrScale?: number;
  /** The trigger word. Recorded in the adapter sidecar either way; whether it is
   *  TRAINED depends on `triggerPrepend`. */
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
  /** `song` (default) presents each crop at its TRUE position in the track;
   *  `zero` is the pre-2026-08-23 convention where every crop claimed to be the
   *  opening, which is a train/inference mismatch (generation always starts at
   *  frame 0). Kept only to reproduce an older run. */
  cropAnchor?: 'song' | 'zero';
  /** Frames of no-grad history in front of each crop. 0 = off, and off is the
   *  default. Requires cropAnchor 'song'. See MM3_LM_DEFAULTS.prefixFrames. */
  prefixFrames?: number;
  prefixChunk?: number;
  prefixSelftest?: boolean;
  /** Mid-run audio previews. Omitted or all-zero cadence = off. */
  preview?: Mm3PreviewOptions;
  /** Prior preservation. Omitted or `every: 0` = off. */
  regularisation?: Mm3RegularisationOptions;
}

/** Prior preservation: train some steps against the FROZEN BASE MODEL'S OWN
 *  predictions on an unrelated corpus, so the adapter is penalised for changing
 *  its mind about material that has nothing to do with the artist.
 *
 *  This is the one term in the objective that distinguishes "learned the voice"
 *  from "rewrote the planner", and it is what bghira's regularised fiona variant
 *  added after finding that generic captions still summoned the style. */
export interface Mm3RegularisationOptions {
  /** Dataset id of the unrelated corpus. It needs MM3 captions and RVQ codes,
   *  exactly like a training dataset; lyrics may be empty. */
  datasetId: string;
  /** Every Nth step is a regularisation step. 0 = off, 3 = bghira's 1:2 ratio.
   *
   *  It DILUTES style exposure: at 3, a 1000-step run spends ~667 steps on the
   *  artist. Raise the step count to compensate, as his regularised run did. */
  every?: number;
  /** Classes kept per position when capturing the base's distribution. 64 is
   *  his. Measured coverage of the base's probability mass on MM3 semantic
   *  codes: 64 -> 89.6%, 128 -> 94.1%, 256 -> 97.0%. The per-step cost does not
   *  depend on K (the label block is uploaded dense), only the cache size does. */
  topK?: number;
}

/** Mid-run audio previews for an MM3 LM training run (services/training/
 *  mm3Preview.ts). The cadence fields are the switch: both zero means off.
 *
 *  A preview is not free — the trainer has to pause, save ~5.6 GB of optimizer
 *  state and exit so the render can have the card, then reload. Budget roughly
 *  a minute and a half per preview point, against ~4 s per training step. */
export interface Mm3PreviewOptions {
  /** Render every N optimizer steps. 0 = off. */
  everySteps?: number;
  /** …or every N minutes of training, whichever comes first. 0 = off. */
  everyMinutes?: number;
  /** Sample length in seconds (8–120). 24 s ≈ 16 s of GPU. */
  seconds?: number;
  /** Fixed across every preview in the run — that is what makes step-to-step
   *  comparison legible. */
  seed?: number;
  /** Blank = the first HELD-OUT song's own caption, with the trigger prepended
   *  the way the training rows carry it. */
  caption?: string;
  lyrics?: string;
  /** Also render a neutral off-genre caption with the adapter active, to catch
   *  collateral damage to the base planner. */
  control?: boolean;
  /** Override the built-in control caption. '-' disables the control render. */
  controlCaption?: string;
  /** Render both captions with NO adapter before step 1, as the reference. */
  baseline?: boolean;
  /** Adapter MLP scale for the preview render, INDEPENDENT of what generation
   *  uses. A preview is a progress read-out, not a release render: the MLP dial
   *  trades identity against fidelity and is rank-dependent, so the scale that
   *  best shows "is identity arriving yet" is not necessarily the one you would
   *  ship at. Pinning it here also keeps the step-to-step comparison honest — if
   *  the shipped generation default moved mid-campaign, previews from before and
   *  after would stop being comparable. */
  scaleMlp?: number;
  /** Companion to scaleMlp. Left at 1.0 by default: attention is where identity
   *  lives, and turning it down is a different experiment. */
  scaleAttn?: number;
}

/** One rendered preview. The audio is served by
 *  GET /api/training/mm3/preview?run=<run>&file=<file>. */
export interface TrainingPreview {
  id: string;
  step: number;
  totalSteps: number;
  kind: 'artist' | 'control';
  /** True for the no-adapter reference rendered before step 1. */
  base: boolean;
  /** Filename inside the run's previews/ directory. */
  file: string;
  seconds: number;
  seed: number;
  caption: string;
  /** Which LM base this was rendered on. Recorded because it is NOT the base
   *  the user has selected for generation: adapters are garbled on f16, so an
   *  adapter preview pins q8_0. Absent on previews made before that fix, every
   *  one of which rendered on f16 and should not be trusted. */
  renderBase?: string;
  /** Training loss at the checkpoint this was rendered from. */
  loss?: number;
  bytes: number;
  /** Wall time of the render itself. */
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

// ─── Dataset ──────────────────────────────────────────────────────────────
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
  defaultLanguage: string;    // language the user KNOWS the corpus is in; overrides understand's guess
  sampleCount: number;
  labeledCount: number;
  excludedCount: number;
  status: 'draft' | 'labeling' | 'labeled' | 'built' | 'error';
  builtAt: string;            // ISO or ''
  datasetJsonPath: string;    // absolute path or ''
  /** Friendly album name from the tracks' embedded tags — '' when unknown.
   *  Cached in the row; detected by datasetAssets.ts (majority vote). */
  albumName: string;
  /** Lyric Studio lyrics_sets.id this dataset exported to — 0 = never linked.
   *  Written by the export commit; lazily backfilled when the audition's
   *  Lyric Studio prompt source resolves the album by detection. */
  lyricsSetId?: number;
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

/**
 * Per-dataset pipeline progress, read fresh off disk on every request: which
 * stages have actually left an artefact behind. Never cached — deleting a
 * tensors folder or an adapter has to show up immediately.
 */
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
  /** Subdirs of data/training/tensors/<slug> holding a preprocess_meta.json. */
  preprocessedVariants: number;
}

// ─── Jobs ─────────────────────────────────────────────────────────────────
export interface TrainingJobSummary {
  id: string;
  datasetId: string;
  kind: TrainingJobKind;
  status: TrainingJobStatus;
  total: number;
  done: number;
  failed: number;
  currentSampleId: string | null;
  // 'essentia' | 'understand' | 'waiting-for-engine' | 'genius' | 'llm' | 'writing' | 'build'
  // kind==='preprocess' adds: 'engine-stop' | 'loading-models' | 'preprocess' | 'stats' | 'engine-restart'
  // kind==='train-lm'   adds: 'engine-stop' | 'loading-models' | 'extract' | 'train' | 'export' | 'engine-restart'
  // kind==='train-dit'  adds: 'engine-stop' | 'loading-models' | 'train' | 'export' | 'engine-restart'
  phase: string;
  engineQueueDepth: number;   // # of ace-server jobs ahead of ours; 0 if unknown
  error: string | null;
  createdAt: number;          // epoch ms
  startedAt: number | null;
  finishedAt: number | null;
}

// ─── Capabilities ─────────────────────────────────────────────────────────
export interface TrainingCapabilities {
  engine: {
    up: boolean;
    ready: boolean;               // from /api/health engine.ready
    understandSupported: boolean; // lm && dit && vae registries all non-empty
    missingModels: string[];      // e.g. ['lm'] — which registries are empty
    queueDepth: number;           // ace-server GET /jobs length
    lmModels: string[];           // LM registry names, for the understand model picker
    defaultLmModel: string;       // server's pick (biggest model, fast quant) when the UI sends none
  };
  essentia: { available: boolean; binPath: string };
  genius: { configured: boolean };
  /**
   * Local audio captioning via MOSS-Music-8B. Unlike `llm`, this is not a
   * credential check — it is "is the binary built and are the weights on disk",
   * and `missing` says which so the UI can tell the user the right thing.
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

// ─── SSE stream ───────────────────────────────────────────────────────────

/** Structured training numbers (L21). Additive member of TrainingStreamEvent.
 *  Consumers MUST ignore unknown `metric` values rather than throwing. */
export interface TrainingMetricEvent {
  type: 'metric';
  /** `eval` is held-out loss — the only series that can distinguish learning
   *  from memorising, and therefore the one worth watching. */
  /** `target` announces a target-loss run's stopping line ONCE, at the top
   *  of the run, so the chart can draw it. It carries `loss` (the target)
   *  and `totalSteps` (the cap the run still ends at if the target never
   *  arrives). */
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
  // train-dit additions (§2.6). No new union member — a DiT job reuses every
  // metric name above and adds these five.
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
  // …plus the rest of §2.2's JSONL fields. These are NOT in §2.4's block, but
  // §2.2 mandates them on the `vram` and `data` events and the relay whitelists
  // every field by name, so without a home on this interface they could never
  // reach a consumer. Same class of gap as the 4B plan's mode/baseMb/ckptMb.
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
      sample?: TrainingSample; error?: string;                     // `sample` present on success
    }
  | { type: 'log'; level: 'info' | 'warn' | 'error'; message: string; ts: number }
  // NB `preview`, not `sample` — `sample` is already this union's dataset-row
  // event and reusing it would collide in every consumer's switch.
  | { type: 'preview'; run: string; preview: TrainingPreview }
  | { type: 'status'; status: TrainingJobStatus; error?: string }   // terminal; server closes after this
  | TrainingMetricEvent;

// ─── Request payloads ─────────────────────────────────────────────────────

export interface ScanPreview {
  root: string;
  audioFiles: number;
  withSidecar: number;
  withCaption: number;
  hasDatasetJson: boolean;
  extensions: Record<string, number>;
  sampleNames: string[];
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
  defaultLanguage?: string;   // default 'english'
}

export type PatchDatasetInput = Partial<Pick<
  TrainingDatasetSummary,
  'name' | 'customTag' | 'tagPosition' | 'genreRatio' | 'defaultArtist' | 'defaultAlbum' | 'defaultGenre' | 'defaultLanguage' | 'recursive'
>>;

export type PatchSampleInput = Partial<Pick<
  TrainingSample,
  'caption' | 'genre' | 'bpm' | 'key' | 'signature' | 'language' | 'isInstrumental'
  | 'lyrics' | 'customTag' | 'repeat' | 'promptOverride' | 'excluded'
>>;

export type BulkSetInput = Partial<Pick<
  TrainingSample,
  'excluded' | 'isInstrumental' | 'genre' | 'customTag' | 'language' | 'repeat'
>>;

export interface BulkResult {
  updated: number;
  failed: Array<{ sampleId: string; error: string }>;
}

/** Optional `/understand` overrides forwarded into the engine `request` part. */
export interface UnderstandOverrides {
  lmModel?: string;
  synthModel?: string;
  lmTemperature?: number;
  lmTopP?: number;
  lmTopK?: number;
  seed?: number;
}

export interface LabelOptions {
  sampleIds?: string[];
  scope?: 'all' | 'unlabeled';
  useEssentia?: boolean;   // default true  — local BPM/key
  useGenius?: boolean;     // default false — canonical lyrics (needs token)
  useCaption?: boolean;    // default false — LLM caption+genre (audio-grounded on gemini)
  useUnderstand?: boolean; // default false — LEGACY /understand path (2026-07-27 pivot)
  mergePolicy?: MergePolicy;
  understand?: UnderstandOverrides;
  caption?: { provider?: string; model?: string };
  /** Answer 200 {jobId:null, skipped} instead of 400 when nothing needs
   *  labelling. Set by the bulk pipeline, for which an already-labelled dataset
   *  is a completed stage rather than a failure. */
  allowEmpty?: boolean;
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
  /**
   * `moss` provider only: also emit the MM3 Structured Caption to
   * `<stem>.mm3.txt`. Defaults to on — it is one extra decode off an encode
   * already paid for, and it is the file `ace-train mm3-condition` reads.
   * Ignored by the cloud providers, which have no MM3 mode.
   */
  wantMm3?: boolean;
}

export interface BuildOptions {
  outputPath?: string;
}

// ─── Preprocess (Training Studio phase 2) ─────────────────────────────────
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

// ─── LM LoRA training (Training Studio phase 3) ───────────────────────────
// Spec: docs/plans/2026-07-27-lm-trainer-implementation.md §2.7
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
  /** Explicit target-loss chain, one ace-train leg per entry, strictly
   *  descending (0 legal only last). Omit for the default: targetLoss becomes
   *  the FINAL stage of the 2.0 → 1.5 → final ladder (2026-08-29 — the staged
   *  chain is the recipe; a single-entry array forces the legacy one-shot). */
  targetLossStages?: number[];
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
  milestoneStep?: number;          // default 0 (milestones off)
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
   *  'outprod' is upstream ggml — out_prod(W, transpose(grad)) — which ggml-cuda
   *  implements F32-only, forcing an F32 weight and dragging the forward mul_mat
   *  onto TF32 too. 'mm' emits mul_mat(cont(transpose(W)), grad) instead:
   *  provably the same shape and the same maths, but dtype-agnostic, so a BF16
   *  weight rides real BF16 tensor cores in both directions. Measured ~1.7-1.8x
   *  per layer per step on an RTX 5090. Unlike `weights`, this does NOT change
   *  which quantity is computed.
   *
   *  THE LM SERVER DEFAULT IS 'outprod', unlike train-dit's 'mm': `weights:
   *  'bf16'` already reaches the same mul_mat backward by rewriting ggml's
   *  out_prod nodes in place (lm-bf16.h Lever A), and its S18 tripwire aborts
   *  when --bwd mm leaves it nothing to rewrite. The pair is refused with a 400.
   *  On the f32-window path 'mm' also buys nothing — the transposed weight is
   *  the F32 window, so the GEMM stays TF32 and the extra cont is pure cost. */
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
  adapterName: string;             // artist name (per-base layout: unsuffixed)
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

// ─── DiT LoRA/LoKR training (Training Studio phase 4 + D22 LoKR) ──────────
// Spec: docs/plans/2026-07-28-dit-trainer-implementation.md §2.6
//       docs/plans/2026-07-28-lokr-dit-training.md §3
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
  /** Crop regime (2026-08-29). Defaults ('song'/'structured') are the fix:
   *  RoPE positions carry the crop's true offset and crop draws weight the
   *  track's start/end. 'zero'/'random' reproduce the legacy behaviour. */
  cropAnchor?: 'song' | 'zero';
  cropMode?: 'structured' | 'random';
  cropStartFrac?: number;          // default 0.2
  cropEndFrac?: number;            // default 0.2
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
   *  'outprod' is upstream ggml — out_prod(W, transpose(grad)) — which ggml-cuda
   *  implements F32-only, forcing an F32 weight and dragging the forward mul_mat
   *  onto TF32 too. 'mm' emits mul_mat(cont(transpose(W)), grad) instead:
   *  provably the same shape and the same maths, but dtype-agnostic, so a BF16
   *  mirror rides real BF16 tensor cores in both directions. Measured ~1.7-1.8x
   *  per layer per step on an RTX 5090. ace-train's own default is 'outprod';
   *  the SERVER default is 'mm' (training.ts train-dit handler). */
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

/** camelCase mirror of a `training_datasets` row. Identical in shape to
 *  TrainingDatasetSummary — the summary IS the row (D3: no per-sample table). */
export type TrainingDatasetRow = TrainingDatasetSummary;

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
  /** Opt-in DiT render of this side's codes (same DiT/seed/steps both sides,
   *  NO sound adapter — the LM adapter stays the only variable). Absent on
   *  previews recorded before the feature and when renderDit was off. */
  renderUrl?: string;       // /api/training/previews/<id>/<slot>-render
  renderMs?: number;        // wall clock of the /synth render
  renderError?: string;     // render failed; the codes sketch above is still valid
  /** Opt-in SECOND render of the same codes through the dataset's trained DiT
   *  adapter (renderDitAdapter) — together with renderUrl this gives the 2×2
   *  matrix {base LM, LM adapter} × {bare DiT, DiT adapter}. */
  renderAdapterUrl?: string;   // /api/training/previews/<id>/<slot>-render-adapter
  renderAdapterMs?: number;    // wall clock of the adapter /synth render
  renderAdapterError?: string; // adapter render failed; bare render/sketch still valid
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
  /** Mirrored-generation receipts (2026-08-12 parity recipe): everything
   *  "Send to Custom-Gen" needs to reproduce this run bit-identically.
   *  Absent on previews recorded before the feature. */
  captionInput?: string;    // the caption BEFORE the trigger tag was applied
  bpm?: number;             // metadata pins the LM was forced to (0/'' = LM predicted)
  keyscale?: string;
  timesignature?: string;   // numerator form ('4'), as pinned
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
  /** Explicit metadata pins (Lyric Studio prompt source): force_fields'd into
   *  the CoT FSM exactly like the sample-row pins. Take precedence over the
   *  sample row when both are present; 0/'' = not pinned. */
  bpm?: number;
  keyscale?: string;
  timesignature?: string;         // '4' or '4/4' — normalized to the numerator
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
   *  DiT adapter, giving the 2×2 {base LM, LM adapter} × {bare DiT, DiT
   *  adapter}. The adapter is resolved server-side from the adapter layout;
   *  409 when none is trained. Pins every render to the adapter's training
   *  base (cross-base adapters are basin-sensitive). */
  renderDitAdapter?: boolean;     // default false; only meaningful with renderDit
  renderDitAdapterName?: string;  // default: the dataset slug (the trainer's default)
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

/** Legacy name for TrainingAdapterHit — same shape, kept so the Lyric Studio
 *  export contract reads the way its consumers already expect. */
export type LyricStudioAdapterHit = TrainingAdapterHit;

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

// ─── Batch pipeline (multi-folder import + auto-chained stages) ────────────
//
// Spec: docs/plans/2026-07-28-training-batch-pipeline.md §2.1

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

export interface StartPipelineInput {
  folders: PipelineFolderSpec[];
  /** Which stages to run, in canonical order. Omit = all five. */
  stages?: PipelineStage[];
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

/** Per-pipeline overrides for the label stage, applied to EVERY dataset in the
 *  run on top of the stored label defaults. Lets a bulk re-caption skip the
 *  Genius and Essentia passes it does not need, and force the merge policy a
 *  re-caption requires (fill_missing keeps the very captions it should
 *  replace). Absent field = the stored default decides. */
export interface PipelineLabelOptions {
  /** 'all' re-labels every sample; default 'unlabeled' only fills gaps. A
   *  re-caption run MUST use 'all' — the corpus is already labeled, so the
   *  default scope finds zero targets and the stage silently no-ops. */
  scope?: 'all' | 'unlabeled';
  useEssentia?: boolean;
  useGenius?: boolean;
  useCaption?: boolean;
  mergePolicy?: MergePolicy;
}

export interface PipelineSummary {
  id: string;
  status: PipelineStatus;
  stages: PipelineStage[];
  items: PipelineItem[];
  createdAt: number;
  finishedAt: number | null;
  /** Pause has been requested but the in-flight stage is still finishing.
   *  status flips to 'paused' once the runner parks at the next stage
   *  boundary. Absent on snapshots written before pause existed. */
  pauseRequested?: boolean;
  /** Label-stage overrides for this run — persisted with the state so a
   *  resumed pipeline keeps them. Absent on older snapshots. */
  labelOptions?: PipelineLabelOptions;
}

export interface TrainingDefaults {
  label: Record<string, unknown>;
  preprocess: Record<string, unknown>;
  trainLm: Record<string, unknown>;
  trainDit: Record<string, unknown>;
}
