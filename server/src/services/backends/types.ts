// backends/types.ts — EngineBackend interface + capability manifest shapes
//
// Backend capability and generation contracts.
// The request and outcome records are additive. Existing backend mappings stay
// private to their generation modules; these contracts do not translate or
// mutate the request.

import type { PluginParamSchema } from '../aceClient.js';
import type { GenerationJob, StageTiming } from '../generation/jobTypes.js';
import type { LaneLease } from '../generation/gpuLane.js';

/** Which job queue a backend's generations serialize against (plan §3.2/§4.4). */
export type ResourcePool = 'gpu' | 'remote';

export type BackendLifecycleStatus = 'down' | 'starting' | 'ready' | 'suspended' | 'crashed';

export interface EngineBackendLifecycle {
  start(): Promise<void>;
  /** Clean shutdown (frees VRAM/resources). */
  stop(): Promise<void>;
  status(): BackendLifecycleStatus;
}

/** Which core (model-agnostic) params the active backend honors, and their
 *  ranges. Mirrors the "~15 field" core inventory in plan §3.3. Left open
 *  (index signature) so a backend can report extra core-ish knobs without a
 *  type churn every time the set grows. */
export interface BackendCoreCapabilities {
  /** `max` is the hard ceiling. `auto` says the backend can decide the length
   *  itself when no duration is asked for — MM3's planner LM emits a stop
   *  token and the render ends there (engine mm3-ar-loop.h), so a requested
   *  duration is only a CEILING, never a target. ACE has no such stop: its
   *  metadata FSM is told a length and aims for it. */
  /** `editable` says whether the UI's duration control does anything on this
   *  backend. Optional and additive (plan §4.2 `duration.editable`): a
   *  backend that omits it is read as editable (ACE's own behavior, and the
   *  correct default for every backend written before this field existed).
   *  false means "model-ended" — the UI hides the control and `max` is a
   *  ceiling, never a target (see YuE2/MM3, both auto:true). */
  duration: { max: number; auto: boolean; editable?: boolean };
  bpm: boolean;
  keyscale: boolean;
  negativePrompt: boolean;
  batch: { max: number };
  seed: boolean;
  [key: string]: unknown;
}

/** Gates whole UI regions/studios. Mirrors plan §4.2's feature list. Left
 *  open (index signature) so new studios/features don't require editing this
 *  type in lockstep with every backend that gains or lacks them. */
export interface BackendFeatureCapabilities {
  /** The backend exposes a user-selectable model catalogue (models() returns
   *  non-empty buckets). Deliberately separate from `lm`: MM3 has selectable
   *  weights but no ACE-style LM/CoT stage, and gating the Models cluster on
   *  `lm` hid the model picker for it. */
  models: boolean;
  lm: boolean;
  plugins: boolean;
  /** The Lua sampler plugins (solvers, schedulers, guidance) actually RUN on
   *  this backend's denoiser.
   *
   *  Deliberately separate from `plugins`, which by now means something
   *  narrower than its name suggests: it selects WHICH Generation dropdown the
   *  UI renders (ACE's plugin-registry one vs the generic seed + declared
   *  extensions one) and rides herd on the ACE-VAE-coupled post stages. MM3
   *  needs the generic dropdown for its own steps/cfg knobs AND the plugin
   *  controls, so it wants `plugins: false` with `samplerPlugins: true` —
   *  a combination the single flag could not express.
   *
   *  Engine side: ACE runs them natively (hot-step-sampler.h); MM3 runs them
   *  through the convention bridge in minimax/mm3-plugins.h. Postprocess
   *  plugins are NOT covered — those replace ACE's tiled VAE decode and have no
   *  MM3 analogue. */
  samplerPlugins: boolean;
  adapters: boolean;
  /** The backend exposes RUNTIME LM LoRA adapters (a picker + strength dials
   *  applied to its language/planner stage), independent of `adapters`, which
   *  gates ACE's DiT adapter stack UI — merge/runtime modes, per-section
   *  masking, trigger embedding, DiT group scales. MiniMax-Music3 has the
   *  former and none of the latter, so one flag could not express it.
   *  Required like the rest of the manifest: "we don't have this" is a
   *  statement, not an omission. */
  lmAdapters: boolean;
  /** The LM adapter is ENGINE STATE, chosen through POST /api/backends/models
   *  (`lmAdapter` bucket), not a per-request parameter.
   *
   *  Both kinds exist and they need different UI. MiniMax-Music3 applies its
   *  LoRA per generation from the request, so its picker writes to the
   *  request bag and nothing is posted until Generate. YuE2 MERGES the delta
   *  into the resident weights at load (yue2-adapter.h) — there is no way to
   *  change it in place, so picking one is a POST that evicts the model and
   *  the next generation pays a reload. Reading only `lmAdapters` cannot tell
   *  the two apart, and the UI must not branch on a backend id. */
  lmAdapterSelectable: boolean;
  /** The model-agnostic post-processing stages run for this backend: the VST
   *  chain and reference mastering, both of which read the sample rate from
   *  the WAV rather than assuming one. Separate from `plugins`, which gates
   *  ACE's Lua solver/scheduler registry and the ACE-VAE-coupled stages
   *  (PP-VAE re-encode, Spectral Lifter). */
  postProcess: boolean;
  /** StableStep / SA3 refinement is available for this backend's output. SA3
   *  is natively 44.1 kHz and rate-transparent on its whole-mix path, so it is
   *  not tied to ACE's 48 kHz pipeline. */
  stableStep: boolean;
  /** Whisper transcription of the rendered audio. Backend-agnostic: whisper-cli
   *  takes a file path and resamples internally, so it depends on nothing but
   *  the output existing. */
  whisper: boolean;
  /** Lyric timestamps (LRC) derived from the model's own attention during
   *  generation. ACE reads its DiT's lyric cross-attention; MiniMax-Music3's
   *  DiT has no cross-attention and never sees lyrics, so this is false there
   *  until the LM-attention route (MM3_ALIGN_DUMP findings) is wired up. */
  lyricTimestamps: boolean;
  /** Lyric timestamps recovered AFTER the render by force-aligning the audio
   *  against the lyrics it was given, for backends whose model cannot supply
   *  them itself. YuE2 has this: the MMS_FA aligner it trains its own lyric
   *  cursor with, exposed as POST /yue2/align. Opt-in per render (it is a
   *  second model and a second forward), which is why it is a toggle in the
   *  post-processing dropdown and not simply always on. */
  forcedAlignment: boolean;
  cover: boolean;
  repaint: boolean;
  lego: boolean;
  extract: boolean;
  streaming: boolean;
  training: boolean;
  midi: boolean;
  stems: boolean;
  understand: boolean;
  conceptSteering: boolean;
  [key: string]: boolean;
}

/** GET /api/capabilities response shape (plan §4.2). Shared code must only
 *  branch on these flags, never on `backend` id, outside this backend's own
 *  module. */
/** Which top-bar cluster a declared knob belongs in.
 *
 *  The generic dropdowns render one flat list per group, so this is the ONLY
 *  thing keeping a backend's planner-LM knobs out of the Generation panel.
 *  Absent means 'generation' — the behaviour before groups existed, and the
 *  right default for a backend that never thinks about it. */
export type BackendExtensionGroup = 'generation' | 'lm';

/** A backend-declared knob. Same schema the Lua plugins use (so one renderer
 *  serves both), plus the cluster it belongs to. */
export interface BackendExtensionParam extends PluginParamSchema {
  group?: BackendExtensionGroup;
  /** Optional accordion inside the cluster. Knobs sharing a section render
   *  together under one collapsible header (collapsed by default), in the
   *  order the section first appears; knobs without one render flat above
   *  the sections. Keeps a backend with many knobs from showing them all at
   *  once. */
  section?: string;
  /** Hover text for the section header; the first knob in a section that
   *  carries one wins. */
  section_hint?: string;
}

export interface BackendCapabilities {
  backend: string;
  up: boolean;
  core: BackendCoreCapabilities;
  features: BackendFeatureCapabilities;
  /** Backend-specific knobs, rendered generically by the existing
   *  PluginControls schema renderer (reuses the Lua plugin param schema —
   *  plan §4.2, §3.6). `group` splits them across the top-bar clusters. */
  extensions: BackendExtensionParam[];
  /** Plain-text licensing notice the picker/Model Manager should show
   *  verbatim (e.g. YuE2's CC BY-NC 4.0 notice, the authors' individual-use
   *  carve-out, and the upstream contact for a company licence). The text is
   *  the backend module's own constant — never reword it here or in the UI,
   *  since it is a licensing claim. Optional and additive — most backends omit
   *  it. Not a substitute for MM3's own license-mandated `displayName`
   *  string (backends/minimax/index.ts), which stays as it is. */
  license?: string;
}

/** Backend-shaped model catalogue. ACE has an {lm,dit,vae,embedding} split;
 *  other backends may not (plan §4.5: "Music 3 has no lm/dit/vae split to
 *  show") — hence a generic bucket map rather than a fixed shape. */
export interface BackendModels {
  /** e.g. { lm: [...], dit: [...], vae: [...], embedding: [...] } for ACE,
   *  { lm: [...quants], synth: [...quants] } for MiniMax-Music3. */
  buckets: Record<string, string[]>;
  adapters?: string[];
  lmAdapters?: string[];
  /** Optional per-entry detail for `lmAdapters`, keyed by the SAME reference
   *  the list carries (YuE2: the absolute .safetensors path). Everything here
   *  is cosmetic — a picker with none of it still works — except `trigger`,
   *  which is load-bearing: a YuE2 NAR adapter does nothing unless its trigger
   *  word leads the style prompt, so the user has to be able to read it. */
  lmAdapterMeta?: Record<string, {
    label?: string;
    runName?: string;
    trigger?: string;
    /** Joint export fallback from the dataset setting, absent from training metadata. */
    triggerInferred?: boolean;
    rank?: number;
    steps?: number;
    bytes?: number;
    dataset?: string;
    /** The run's final export rather than a mid-run snapshot. */
    final?: boolean;
    loss?: number;
  }>;
  defaults?: Record<string, unknown>;
  /** Optional per-bucket display metadata (size on disk, filename), keyed
   *  bucket -> option value. Purely cosmetic; the UI renders the bare option
   *  when absent. */
  meta?: Record<string, Record<string, { label?: string; bytes?: number }>>;
}

/** Which model each bucket should use, e.g. { lm: 'q8_0', synth: 'Q4_K_M' }.
 *  An empty-string value means "auto / backend default". */
export type BackendModelSelection = Record<string, string>;

// ── Generation request and execution records ────────────────────────────────

export const GENERATION_ENVELOPE_VERSION = 1 as const;

export type GenerationOperation =
  | 'text2music' | 'cover' | 'repaint' | 'lego' | 'extract' | 'complete'
  | (string & {});

/** Descriptive common inputs. Optional values stay absent when the caller did
 * not supply them; backend defaults remain in the existing mappers. */
export interface CommonInputs {
  caption: string;
  lyrics: string;
  instrumental?: boolean;
  duration?: number;
  seed?: number;
  randomSeed?: boolean;
  batchSize?: number;
  title?: string;
}

/** Added to GenerationJob in step 5. */
export interface GenerationEnvelope {
  version: typeof GENERATION_ENVELOPE_VERSION;
  jobId: string;
  userId: string;
  backendId: string;
  operation: GenerationOperation;
  submission: Readonly<Record<string, unknown>>;
  common: CommonInputs;
  models: BackendModelSelection;
  options: Partial<Record<string, Readonly<Record<string, unknown>>>>;
  policy: GenerationPolicy;
  enqueuedAt: number;
  submittedBackendMismatch?: string;
}

export interface GenerationPolicy {
  retry: {
    maxAttempts: number;
    reseedOnRetry: boolean;
  };
  timeoutMinutes?: number;
}

export type GenerationEndReason =
  | 'completed'
  | 'cancelled'
  | 'engine_failed'
  | 'stalled'
  | 'timeout'
  | 'reset'
  | 'failed';

/** Added to GenerationJob in step 7. */
export interface GenerationAttempt {
  attempt: number;
  startedAt: number;
  endedAt?: number;
  effective: {
    seed?: number | string;
    /** ACE's planner seed, when the backend writes it back; MM3 may omit it. */
    lmSeed?: number | string;
    models: BackendModelSelection;
    [key: string]: unknown;
  };
  reseeded: boolean;
  engineJobIds: string[];
  endReason?: GenerationEndReason;
  error?: string;
  submittedBackendMismatch?: string;
}

export type CancelReason = 'user' | 'cancel-all' | 'reset' | 'stall' | 'timeout' | 'shutdown';

export interface CancelAck {
  /** Terminal engine status or verified exit of the owning process only. */
  acknowledged: boolean;
  engineStatus?: 'cancelled' | 'done' | 'failed' | 'process_exited';
  waitedMs: number;
}

export interface GenerationHooks {
  onEngineJob(engineJobId: string): void;
  onStage(stage: string, progress?: number): void;
  onArtifact(artifact: GenerationArtifact): void;
}

export type PollUntilDone = (
  engineJobId: string,
  job: GenerationJob,
  signal: AbortSignal,
  timeoutMinutes?: number,
) => Promise<void>;

/** Step 8 context. The lease is the shared lane ownership token for this run. */
export interface GenerationContext {
  envelope: Readonly<GenerationEnvelope>;
  attempt: GenerationAttempt;
  lease: LaneLease;
  signal: AbortSignal;
  pollUntilDone: PollUntilDone;
  hooks: GenerationHooks;
}

export interface ResolvedRequest {
  operation: GenerationOperation;
  common: CommonInputs;
  models: BackendModelSelection;
  options: Readonly<Record<string, unknown>>;
  policy: GenerationPolicy;
}

export type GenerationArtifactKind =
  | 'audio' | 'mastered' | 'latent' | 'lrc' | 'lyrics' | 'cover-art'
  | (string & {});

export interface GenerationArtifact {
  kind: GenerationArtifactKind;
  trackIndex: number;
  url?: string;
  path?: string;
}

export interface GenerationOutcome {
  endReason: GenerationEndReason;
  stages: StageTiming[];
  artifacts: GenerationArtifact[];
  songIds: string[];
  result?: GenerationJob['result'];
  error?: string;
}

/** A registered generation backend. Request resolution and one-attempt
 *  execution are behind the interface in step 4. Cancellation is added with
 *  complete engine acknowledgement and ID tracking in step 9. */
export interface EngineBackend {
  /** 'ace' | 'minimax-m3' | ... */
  id: string;
  displayName: string;
  /** Which job queue this backend's generations share (plan §3.2). */
  resourcePool: ResourcePool;
  lifecycle: EngineBackendLifecycle;
  /** Cached, cheap — safe to call on every UI render. */
  capabilities(): Promise<BackendCapabilities>;
  models(): Promise<BackendModels>;
  /** Choose which model each bucket runs. Optional: a backend whose model set
   *  is fixed, or which is selected per-request rather than as engine state
   *  (ACE passes model names on each generate call), simply omits it and the
   *  route answers 501. Implementations must be idempotent — the UI posts the
   *  whole selection on every change. */
  selectModel?(selection: BackendModelSelection): Promise<{ changed: boolean; [k: string]: unknown }>;

  /** Operations accepted by this backend. */
  operations: readonly GenerationOperation[];
  /** Pure request snapshot; does not call the engine or run a mapper. */
  resolveRequest(submission: Readonly<Record<string, unknown>>): ResolvedRequest;
  /** Run one attempt using the shared step 4 context. */
  generate(job: GenerationJob, ctx: GenerationContext): Promise<GenerationOutcome>;
  /** Optional until the residency redesign wires engine-side arbitration. */
  arbitratesResidencyInEngine?: boolean;
  /** Release this backend's GPU residency WITHOUT stopping it (plan §4.4:
   *  arbitration is model residency, not process switching). Called
   *  fire-and-forget on the OUTGOING backend when the active backend changes,
   *  so the two model families never sit in VRAM together.
   *
   *  Optional and best-effort: a backend with nothing to free (or no way to
   *  free it) simply omits it, and shared code must never block on or fail
   *  because of it. Declared here rather than branching on backend id in
   *  routes/backends.ts — see plan §2 principle 2. */
  releaseVram?(): Promise<void>;
}
