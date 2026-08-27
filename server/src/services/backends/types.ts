// backends/types.ts — EngineBackend interface + capability manifest shapes
//
// Phase 1 scaffolding (docs/plans/multi-backend-architecture.md §4.1/§4.2).
// ADDITIVE ONLY: nothing in this file is wired into the generation path yet.
// The ACE backend module (backends/ace/index.ts) wraps the existing
// aceClient/aceEngineProcess/engineState modules by delegation — zero
// behavior change to the live generation flow.
//
// `translate()` / `generate()` / `cancel()` are DELIBERATELY NOT part of this
// interface yet. Per the plan, Phase 1 wraps lifecycle + capabilities +
// models only; the generation path itself keeps flowing through the existing
// routes/generate.ts → aceClient chain until a later phase moves it behind
// this abstraction. Adding them here now would be a same-file no-op that
// invites someone to half-wire the hot path before the plan calls for it.

import type { PluginParamSchema } from '../aceClient.js';

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
  duration: { max: number; auto: boolean };
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
export interface BackendCapabilities {
  backend: string;
  up: boolean;
  core: BackendCoreCapabilities;
  features: BackendFeatureCapabilities;
  /** Backend-specific knobs, rendered generically by the existing
   *  PluginControls schema renderer (reuses the Lua plugin param schema —
   *  plan §4.2, §3.6). */
  extensions: PluginParamSchema[];
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
  defaults?: Record<string, unknown>;
  /** Optional per-bucket display metadata (size on disk, filename), keyed
   *  bucket -> option value. Purely cosmetic; the UI renders the bare option
   *  when absent. */
  meta?: Record<string, Record<string, { label?: string; bytes?: number }>>;
}

/** Which model each bucket should use, e.g. { lm: 'q8_0', synth: 'Q4_K_M' }.
 *  An empty-string value means "auto / backend default". */
export type BackendModelSelection = Record<string, string>;

/** A registered generation backend. See file header for what's intentionally
 *  missing (translate/generate/cancel — later phase). */
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
