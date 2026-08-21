// globalParamsStore.ts -- Zustand store replacing GlobalParamsContext
//
// Migrated from React Context + usePersistedState to Zustand with
// per-field localStorage keys.  Consumers use selectors:
//   const val = useGlobalParamsStore(s => s.fieldName);
//
// Uses the SAME hs-* localStorage keys -- zero migration needed.

import { create } from 'zustand';
import type { GenerationParams, LmRepMode } from '../types';
import { DEFAULT_SETTINGS, type AppSettings } from '../components/settings/SettingsPanel';
import { useBackendStore } from './backendStore';

// -- Types --

export interface AdapterGroupScales {
  self_attn: number;
  cross_attn: number;
  mlp: number;
  cond_embed: number;
  time_embed: number;
  proj_in: number;
}

// -- Per-key localStorage adapter --

function readKey<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return v !== null ? JSON.parse(v) : fallback;
  } catch { return fallback; }
}

function writeKey<T>(key: string, value: T): void {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* full */ }
}

// -- Per-backend settings scope --
//
// A handful of fields are NOT global: they are remembered per generation
// backend. MiniMax-Music3 runs the SAME Lua solver/scheduler/guidance plugins
// as ACE-Step but through a completely different denoiser, so a pick that is
// right for one is usually wrong for the other -- and until this existed both
// backends read and wrote one `hs-inferMethod`, so changing the solver in MM3
// mode silently rewrote the ACE-Step setting (and vice versa).
//
// Key rule: 'ace' keeps the BARE `hs-*` key, every other backend gets
// `hs-*@<backendId>`. That means zero migration for existing installs -- the
// ACE settings people already have stay exactly where they are.
//
// First visit to a non-'ace' backend SEEDS (and persists) from the bare key,
// so switching forks the settings in force rather than snapping to defaults.
// After that one write the two are fully independent.
//
// Adding a field here is all it takes -- the initial hydrate, the backend
// switch and the preset/profile writer all read this table.
const BACKEND_SCOPED_FIELDS: { field: string; key: string; fallback: unknown }[] = [
  { field: 'inferMethod',   key: 'hs-inferMethod',   fallback: 'euler' },
  { field: 'scheduler',     key: 'hs-scheduler',     fallback: 'linear' },
  { field: 'guidanceMode',  key: 'hs-guidanceMode',  fallback: 'apg' },
  // The picks' declared params travel with the picks: "stork2:substeps" may
  // want a different value on each backend's sampler.
  { field: 'pluginParams',  key: 'hs-pluginParams',  fallback: {} as Record<string, string> },
  // Backend-declared extension knobs. Per backend by definition -- two
  // backends may each declare a knob called `steps` and mean different things.
  { field: 'backendParams', key: 'hs-backendParams', fallback: {} as Record<string, unknown> },
];

const BACKEND_SCOPED_KEYS: readonly string[] = BACKEND_SCOPED_FIELDS.map(f => f.key);

/** Live active backend id. 'ace' for every install with no second backend. */
function activeBackendId(): string {
  return useBackendStore.getState().activeBackendId || 'ace';
}

/**
 * localStorage key for `base` under `backendId`. Unscoped keys pass through
 * unchanged, so this is safe to call on any hs-* key.
 *
 * Exported because the preset/profile writer (utils/paramProfiles.ts) persists
 * store fields by key without going through the setters -- applying a preset
 * while MM3 is active must land in MM3's slot, not ACE's.
 */
export function scopedKey(base: string, backendId: string = activeBackendId()): string {
  if (!BACKEND_SCOPED_KEYS.includes(base)) return base;
  return backendId === 'ace' ? base : `${base}@${backendId}`;
}

function readScoped<T>(base: string, fallback: T, backendId: string = activeBackendId()): T {
  const key = scopedKey(base, backendId);
  if (key !== base) {
    try {
      if (localStorage.getItem(key) === null) {
        // First visit to this backend: fork from whatever is in force rather
        // than resetting the user to defaults, and persist it so a later
        // change on the OTHER backend can't drift this one.
        const seeded = readKey(base, fallback);
        writeKey(key, seeded);
        return seeded;
      }
    } catch { /* storage blocked -- fall through to the plain read */ }
  }
  return readKey(key, fallback);
}

/** Every per-backend field, read for `backendId`. Used for the initial store
 *  hydrate and again on each backend switch. */
function hydrateBackendScoped(backendId: string = activeBackendId()): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const f of BACKEND_SCOPED_FIELDS) out[f.field] = readScoped(f.key, f.fallback, backendId);
  return out;
}

// -- Store --

// eslint-disable-next-line @typescript-eslint/no-explicit-any
/**
 * Mirror the Models-tab selection to the server.
 *
 * This state lives in localStorage, which the server cannot see — and the
 * Training Studio needs it: a BULK run has no UI in the loop, and the engine
 * that knows its own loaded DiT is deliberately stopped while training runs. A
 * stale mirror is what made an overnight run train 18 datasets against the
 * stock base instead of the selected fine-tune (2026-07-31).
 *
 * Fire-and-forget on purpose: this must never block or fail a model change.
 */
function mirrorActiveModels(patch: Record<string, string>): void {
  try {
    void fetch('/api/training/active-models', {
      method: 'PUT',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(patch),
    }).catch(() => { /* offline / server down — retried on the next change and at boot */ });
  } catch { /* never let a model change throw */ }
}

export const useGlobalParamsStore = create<any>()((set, get) => ({
  // -- State (initialised from localStorage) --
  ditModel: readKey("hs-ditModel", ''),
  lmModel: readKey("hs-lmModel", ''),
  vaeModel: readKey("hs-vaeModel", ''),
  lmAdapter: readKey("hs-lmAdapter", ''),
  lmAdapterScale: readKey("hs-lmAdapterScale", 1.0),
  embeddingModel: readKey("hs-embeddingModel", ''),
  adapter: readKey("hs-adapter", ''),
  adapterScale: readKey("hs-adapterScale", 1.0),
  // Multi-adapter stack: a list of { path, scale } applied together, each with
  // its own scale. When non-empty it supersedes the single `adapter`. Group
  // scales, adapter mode and basin re-base apply globally to the whole stack.
  adapterStack: readKey("hs-adapterStack", [] as { path: string; scale: number }[]),
  // Stack scaling mode:
  //  'sum'   — each entry's `scale` is its absolute scale; the engine sums them
  //            (can deliberately over-drive: Σ scale may exceed 1).
  //  'blend' — each entry's `scale` is a relative weight; the effective scales
  //            are normalised to the budget so Σ effective = adapterStackBudget,
  //            keeping combined strength constant as adapters are added.
  adapterStackMode: readKey("hs-adapterStackMode", 'blend'),
  adapterStackBudget: readKey("hs-adapterStackBudget", 0.75),
  // Per-section masking (P2) tuning: alignment step fraction, and 0..1 regional
  // self-attention isolation (moderate default) to stop sections inheriting the
  // first section's voice.
  adapterSectionAlignAt: readKey("hs-adapterSectionAlignAt", 0.55),
  adapterSectionIsolation: readKey("hs-adapterSectionIsolation", 0.5),
  adapterMode: readKey("hs-adapterMode", 'runtime'),
  // Runtime adapter delta VRAM precision: 'bf16' (full), 'q8_0' (~½), 'q4_k' (~¼).
  // Lets many stacked adapters fit in VRAM; runtime mode only.
  adapterRuntimeQuant: readKey("hs-adapterRuntimeQuant", 'bf16'),
  // Merge (low VRAM): re-encode merged weights to the base's native quant instead
  // of F32 promotion (~¼ the merged-DiT VRAM on a Q8 base). Merge mode only.
  adapterMergeLowVram: readKey("hs-adapterMergeLowVram", false),
  adapterGroupScales: readKey("hs-adapterGroupScales", {
    self_attn: 1.0, cross_attn: 1.0, mlp: 1.0, cond_embed: 1.0, time_embed: 0.0, proj_in: 0.0,
  }),
  // Basin re-base: nudge a cross-base adapter into the basin of the model it was
  // trained on. rebaseSource = DiT model name (the adapter's "home base").
  // Merge mode only. Remembered across sessions.
  rebaseSource: readKey("hs-rebaseSource", ''),
  rebaseBeta: readKey("hs-rebaseBeta", 0.75),
  adapterFolder: readKey("hs-adapterFolder", ''),
  // '' = server default (hot-step-cpp/adapters/lm). Point at your archive to
  // pick planner adapters from there, like the DiT adapter folder.
  lmAdapterFolder: readKey("hs-lmAdapterFolder", ''),
  advancedAdapters: readKey("hs-advancedAdapters", false),
  // Optional 3rd output: low-step render on the bare DiT (adapter bypassed,
  // LM adapter kept, no post-processing) for A/B-ing the DiT adapter by ear.
  noAdapterRender: readKey("hs-noAdapterRender", false),
  adaptersOpen: readKey("hs-adaptersOpen", false),
  inferenceSteps: readKey("hs-inferenceSteps", 12),
  guidanceScale: readKey("hs-guidanceScale", 9.0),
  cfgCutoffRatio: readKey("hs-cfgCutoffRatio", 1.0),
  lmCfgCutoffRatio: readKey("hs-lmCfgCutoffRatio", 1.0),
  cacheRatio: readKey("hs-cacheRatio", 0),
  shift: readKey("hs-shift", 3.0),
  // DiT self-attention reach override. PARKED — the UI control is removed and
  // this is pinned to -1 (= the model's own 128). Deliberately NOT read from
  // localStorage: anyone who moved the slider while it was exposed still has a
  // value stored, and reading it back would keep silently applying an override
  // with no UI left to show or undo it. See docs/plans/attention-drift/.
  ditSlidingWindow: -1,
  // Solver / scheduler / guidance -- PER BACKEND, so does the spread below
  // cover `inferMethod`, `scheduler`, `guidanceMode`, `pluginParams` and
  // `backendParams` in one line. Declared in BACKEND_SCOPED_FIELDS above;
  // re-read on every backend switch by the subscription at the bottom of this
  // file. Changing a solver in MiniMax-Music3 mode must not touch ACE-Step's.
  ...hydrateBackendScoped(),
  seed: readKey("hs-seed", 42),
  randomSeed: readKey("hs-randomSeed", true),
  // `backendParams` -- backend-declared knobs (capabilities().extensions),
  // keyed by the schema's `key`. One persisted bag rather than a named field
  // per knob: the whole point of the extension mechanism is that a backend can
  // add a control WITHOUT a matching edit here. Spread into the request payload
  // below. Hydrated by the ...hydrateBackendScoped() spread above (per backend).
  // LM Seed — independent of the DiT/generation seed above, unless tied
  // via lmSeedFollowsDit (default true = original tied behavior).
  lmSeed: readKey("hs-lmSeed", 42),
  lmSeedFollowsDit: readKey("hs-lmSeedFollowsDit", true),
  batchSize: readKey("hs-batchSize", 1),
  storkSubsteps: readKey("hs-storkSubsteps", 10),
  beatStability: readKey("hs-beatStability", 0.25),
  frequencyDamping: readKey("hs-frequencyDamping", 0.4),
  temporalSmoothing: readKey("hs-temporalSmoothing", 0.13),
  apgMomentum: readKey("hs-apgMomentum", 0.75),
  apgNormThreshold: readKey("hs-apgNormThreshold", 2.5),
  dcwEnabled: readKey("hs-dcwEnabled", false),
  dcwMode: readKey("hs-dcwMode", 'double'),
  dcwLowScaler: readKey("hs-dcwLowScaler", readKey("hs-dcwScaler", 0.2)),
  dcwHighScaler: readKey("hs-dcwHighScaler", 0.2),
  latentShift: readKey("hs-latentShift", 0.0),
  latentRescale: readKey("hs-latentRescale", 1.0),
  customTimesteps: readKey("hs-customTimesteps", ''),
  denoiseStrength: readKey("hs-denoiseStrength", 0.0),
  denoiseSmoothing: readKey("hs-denoiseSmoothing", 0.7),
  denoiseMix: readKey("hs-denoiseMix", 0.25),
  lssStrength: readKey("hs-lssStrength", 0.0),
  lssVarThresh: readKey("hs-lssVarThresh", 0.15),
  lssDcRemove: readKey("hs-lssDcRemove", true),
  autoTrimEnabled: readKey("hs-autoTrimEnabled", false),
  durationBuffer: readKey("hs-durationBuffer", 15),
  autoTrimFadeMs: readKey("hs-autoTrimFadeMs", 2000),
  skipLm: readKey("hs-skipLm", false),
  skipLrc: readKey("hs-skipLrc", false),
  useCotCaption: readKey("hs-useCotCaption", true),
  lmTemperature: readKey("hs-lmTemperature", 0.8),
  lmCfgScale: readKey("hs-lmCfgScale", 2.2),
  lmTopK: readKey("hs-lmTopK", 0),
  lmTopP: readKey("hs-lmTopP", 0.92),
  lmRepPenalty: readKey("hs-lmRepPenalty", 1.1),
  lmRepWindow: readKey("hs-lmRepWindow", 64),
  lmRepMode: readKey<LmRepMode>("hs-lmRepMode", 'presence'),
  lmDryBase: readKey("hs-lmDryBase", 1.75),
  lmDryMinLen: readKey("hs-lmDryMinLen", 3),
  lmNegativePrompt: readKey("hs-lmNegativePrompt", 'NO USER INPUT'),
  lmCodesStrength: readKey("hs-lmCodesStrength", 1.0),
  // LM codes window mode: 'ratio' scales by fraction of the step budget
  // (lmCodesStrength), 'steps' pins an absolute step count (lmCodesSteps).
  // Both collapse to audio_cover_strength at request build — no engine field.
  lmCodesMode: readKey("hs-lmCodesMode", 'ratio' as 'ratio' | 'steps'),
  lmCodesSteps: readKey("hs-lmCodesSteps", 6),
  postProcessingEnabled: readKey("hs-postProcessingEnabled", true),
  spectralLifterEnabled: readKey("hs-spectralLifterEnabled", false),
  slDenoiseStrength: readKey("hs-slDenoiseStrength", 0.3),
  slNoiseFloor: readKey("hs-slNoiseFloor", 0.1),
  slHfMix: readKey("hs-slHfMix", 0.0),
  slTransientBoost: readKey("hs-slTransientBoost", 0.0),
  slShimmerReduction: readKey("hs-slShimmerReduction", 6.0),
  masteringEnabled: readKey("hs-masteringEnabled", false),
  masteringReference: readKey("hs-masteringReference", ''),
  timbreReference: readKey("hs-timbreReference", false),
  timbreAudioPath: readKey("hs-timbreAudioPath", ''),
  vocalNaturalizerEnabled: readKey("hs-vocalNaturalizerEnabled", false),
  gainOffsetDb: readKey("hs-gainOffsetDb", 0),
  naturalizeAmount: readKey("hs-naturalizeAmount", 0.5),
  natVibratoRate: readKey("hs-natVibratoRate", 4.5),
  natVibratoDepth: readKey("hs-natVibratoDepth", 1.0),
  natFormantStrength: readKey("hs-natFormantStrength", 1.0),
  natMetallicReduction: readKey("hs-natMetallicReduction", 1.0),
  natQuantizationMask: readKey("hs-natQuantizationMask", 0.0),
  natTransitionSmooth: readKey("hs-natTransitionSmooth", 1.0),
  ppVaeReencode: readKey("hs-ppVaeReencode", false),
  ppVaeBlend: readKey("hs-ppVaeBlend", 0.0),
  ppVaeUseOnnx: readKey("hs-ppVaeUseOnnx", true),
  stableStepOn: readKey("hs-stableStepOn", false),
  stableStepStrength: readKey("hs-stableStepStrength", 0.3),
  // Engine backend for the SA3 refine: 'auto' (engine picks) | 'onnx'
  // (ONNX Runtime/TensorRT, NVIDIA) | 'gguf' (GGML — CUDA/Vulkan/CPU).
  stableStepBackend: readKey("hs-stableStepBackend", 'auto'),
  // StableStep DoRA adapters: [{name, scale, enabled}] — persisted selection
  stableStepAdapters: readKey("hs-stableStepAdapters", [] as Array<{ name: string; scale: number; enabled: boolean }>),
  // Preserve source dynamics: envelope-match refined audio to the source
  stableStepPreserveDynamics: readKey("hs-stableStepPreserveDynamics", true),
  // PP-VAE re-encode of the vocal stem — OFF by default: the round trip is
  // lossy above ~4 kHz (see stableStepVocalPpVae in postProcessing.ts)
  stableStepVocalPpVae: readKey("hs-stableStepVocalPpVae", false),
  // Source blending: 'off' | 'crossover' (source lows + refined highs) | 'mix'
  stableStepBlendMode: readKey("hs-stableStepBlendMode", 'off'),
  stableStepCrossoverHz: readKey("hs-stableStepCrossoverHz", 250),
  stableStepCrossoverWidthHz: readKey("hs-stableStepCrossoverWidthHz", 200),
  stableStepMix: readKey("hs-stableStepMix", 1.0),
  // SA3 refine seed: follows the generation seed by default (LM-seed pattern)
  stableStepSeed: readKey("hs-stableStepSeed", 4242),
  stableStepSeedFollowsDit: readKey("hs-stableStepSeedFollowsDit", true),
  // SA3 sampler: steps + Lua solver/scheduler/guidance routing. Deliberately
  // SEPARATE fields from the generation-side inferMethod/scheduler/guidanceMode
  // — StableStep is a post-processing refine, and tying its sampler to the DiT's
  // would mean changing one silently changes the other.
  stableStepSteps: readKey("hs-stableStepSteps", 8),
  stableStepSolver: readKey("hs-stableStepSolver", ''),
  stableStepScheduler: readKey("hs-stableStepScheduler", ''),
  stableStepGuidanceMode: readKey("hs-stableStepGuidanceMode", ''),
  stableStepGuidanceScale: readKey("hs-stableStepGuidanceScale", 1.0),
  coverArtEnabled: readKey("hs-coverArtEnabled", false),
  coverArtSubject: readKey("hs-coverArtSubject", ''),
  qualityEvalEnabled: readKey("hs-qualityEvalEnabled", false),
  qualityEvalTarget: readKey("hs-qualityEvalTarget", 'unmastered'),

  // Dynamic Lua plugin params -- `pluginParams`, hydrated per backend by the
  // ...hydrateBackendScoped() spread above (BACKEND_SCOPED_FIELDS).

  // Whisper Lyrics Transcription
  whisperLyricsEnabled: readKey("hs-whisperLyricsEnabled", false),
  whisperModel: readKey("hs-whisperModel", ''),
  whisperLanguage: readKey("hs-whisperLang", 'auto'),
  whisperBeamSize: readKey("hs-whisperBeam", 5),
  whisperIsolateVocals: readKey("hs-whisperIsolate", false),

  // Postprocess plugin (replaces built-in VAE tiled decoder)
  postprocessEnabled: readKey('hs-postprocessEnabled', false),
  postprocessPlugin: readKey('hs-postprocessPlugin', ''),

  // VAE backend selection (ONNX Runtime / TensorRT)
  useOrtVae: readKey('hs-useOrtVae', false),

  // LUFS Normalization
  lufsEnabled: readKey('hs-lufsEnabled', false),
  lufsPreset: readKey('hs-lufsPreset', 'spotify'),
  lufsTarget: readKey('hs-lufsTarget', -14),

  // -- Actions --
  setDitModel: (v: any) => { set({ ditModel: v }); writeKey("hs-ditModel", v); mirrorActiveModels({ ditModel: v }); },
  setLmModel: (v: any) => { set({ lmModel: v }); writeKey("hs-lmModel", v); mirrorActiveModels({ lmModel: v }); },
  setLmAdapter: (v: any) => { set({ lmAdapter: v }); writeKey("hs-lmAdapter", v); },
  setLmAdapterScale: (v: any) => { set({ lmAdapterScale: v }); writeKey("hs-lmAdapterScale", v); },
  setVaeModel: (v: any) => {
    set({ vaeModel: v });
    writeKey("hs-vaeModel", v);
    // Auto-detect ORT backend from file extension
    const isOnnx = /\.onnx$/i.test(v || '');
    set({ useOrtVae: isOnnx });
    writeKey('hs-useOrtVae', isOnnx);
  },
  setEmbeddingModel: (v: any) => { set({ embeddingModel: v }); writeKey("hs-embeddingModel", v); },
  setAdapter: (v: any) => { set({ adapter: v }); writeKey("hs-adapter", v); },
  setAdapterScale: (v: any) => { set({ adapterScale: v }); writeKey("hs-adapterScale", v); },
  setAdapterStack: (v: any) => { set({ adapterStack: v }); writeKey("hs-adapterStack", v); },
  // Add/remove an adapter path from the stack (idempotent toggle).
  toggleAdapterInStack: (path: string, scale = 1.0) => {
    const cur: { path: string; scale: number }[] = get().adapterStack || [];
    const exists = cur.some(a => a.path === path);
    const next = exists ? cur.filter(a => a.path !== path) : [...cur, { path, scale }];
    set({ adapterStack: next }); writeKey("hs-adapterStack", next);
  },
  // Set the per-adapter scale (sum mode) or relative weight (blend mode) for one entry.
  setAdapterStackScale: (path: string, scale: number) => {
    const cur: { path: string; scale: number }[] = get().adapterStack || [];
    const next = cur.map(a => (a.path === path ? { ...a, scale } : a));
    set({ adapterStack: next }); writeKey("hs-adapterStack", next);
  },
  // Set one entry's active-timestep window (timestep-dependent adapters).
  // Axis is remaining-steps fraction: 1 = first step (noise), 0 = last step
  // (clean). Evaluated per STEP by the engine (gain_domain 'steps'), so 50%
  // means half the steps even on shift-skewed schedules.
  // Full range [0,1] means "always on" and is stripped to keep entries clean.
  setAdapterStackWindow: (path: string, stepStart: number, stepEnd: number) => {
    const cur: { path: string; scale: number; stepStart?: number; stepEnd?: number }[] = get().adapterStack || [];
    const full = stepStart <= 0 && stepEnd >= 1;
    const next = cur.map(a => {
      if (a.path !== path) return a;
      if (full) { const { stepStart: _s, stepEnd: _e, ...rest } = a; return rest; }
      return { ...a, stepStart, stepEnd };
    });
    set({ adapterStack: next }); writeKey("hs-adapterStack", next);
  },
  setAdapterStackMode: (v: any) => { set({ adapterStackMode: v }); writeKey("hs-adapterStackMode", v); },
  setAdapterStackBudget: (v: any) => { set({ adapterStackBudget: v }); writeKey("hs-adapterStackBudget", v); },
  setAdapterSectionAlignAt: (v: any) => { set({ adapterSectionAlignAt: v }); writeKey("hs-adapterSectionAlignAt", v); },
  setAdapterSectionIsolation: (v: any) => { set({ adapterSectionIsolation: v }); writeKey("hs-adapterSectionIsolation", v); },
  setAdapterMode: (v: any) => { set({ adapterMode: v }); writeKey("hs-adapterMode", v); },
  setAdapterRuntimeQuant: (v: any) => { set({ adapterRuntimeQuant: v }); writeKey("hs-adapterRuntimeQuant", v); },
  setAdapterMergeLowVram: (v: any) => { set({ adapterMergeLowVram: v }); writeKey("hs-adapterMergeLowVram", v); },
  setAdapterGroupScales: (v: any) => { set({ adapterGroupScales: v }); writeKey("hs-adapterGroupScales", v); },
  setRebaseSource: (v: any) => { set({ rebaseSource: v }); writeKey("hs-rebaseSource", v); },
  setRebaseBeta: (v: any) => { set({ rebaseBeta: v }); writeKey("hs-rebaseBeta", v); },
  setAdapterFolder: (v: any) => { set({ adapterFolder: v }); writeKey("hs-adapterFolder", v); },
  setLmAdapterFolder: (v: any) => { set({ lmAdapterFolder: v }); writeKey("hs-lmAdapterFolder", v); },
  setAdvancedAdapters: (v: any) => { set({ advancedAdapters: v }); writeKey("hs-advancedAdapters", v); },
  setNoAdapterRender: (v: any) => { set({ noAdapterRender: v }); writeKey("hs-noAdapterRender", v); },
  setAdaptersOpen: (v: any) => { set({ adaptersOpen: v }); writeKey("hs-adaptersOpen", v); },
  setInferenceSteps: (v: any) => { set({ inferenceSteps: v }); writeKey("hs-inferenceSteps", v); },
  setGuidanceScale: (v: any) => { set({ guidanceScale: v }); writeKey("hs-guidanceScale", v); },
  setCfgCutoffRatio: (v: any) => { set({ cfgCutoffRatio: v }); writeKey("hs-cfgCutoffRatio", v); },
  setLmCfgCutoffRatio: (v: any) => { set({ lmCfgCutoffRatio: v }); writeKey("hs-lmCfgCutoffRatio", v); },
  setCacheRatio: (v: any) => { set({ cacheRatio: v }); writeKey("hs-cacheRatio", v); },
  setShift: (v: any) => { set({ shift: v }); writeKey("hs-shift", v); },
  setDitSlidingWindow: (v: any) => { set({ ditSlidingWindow: v }); writeKey("hs-ditSlidingWindow", v); },
  // scopedKey(): these three land in the ACTIVE backend's slot, not a shared one.
  setInferMethod: (v: any) => { set({ inferMethod: v }); writeKey(scopedKey("hs-inferMethod"), v); },
  setScheduler: (v: any) => { set({ scheduler: v }); writeKey(scopedKey("hs-scheduler"), v); },
  setGuidanceMode: (v: any) => { set({ guidanceMode: v }); writeKey(scopedKey("hs-guidanceMode"), v); },
  setSeed: (v: any) => { set({ seed: v }); writeKey("hs-seed", v); },
  setBackendParam: (key: string, v: any) => {
    const next = { ...(get().backendParams || {}), [key]: v };
    set({ backendParams: next });
    writeKey(scopedKey("hs-backendParams"), next);
  },
  setRandomSeed: (v: any) => {
    set({ randomSeed: v }); writeKey("hs-randomSeed", v);
    // When disabling random, snap seed away from -1 (the random sentinel)
    if (!v && get().seed === -1) {
      set({ seed: 42 }); writeKey("hs-seed", 42);
    }
  },
  setLmSeed: (v: any) => { set({ lmSeed: v }); writeKey("hs-lmSeed", v); },
  setLmSeedFollowsDit: (v: any) => { set({ lmSeedFollowsDit: v }); writeKey("hs-lmSeedFollowsDit", v); },
  setBatchSize: (v: any) => { set({ batchSize: v }); writeKey("hs-batchSize", v); },
  setStorkSubsteps: (v: any) => { set({ storkSubsteps: v }); writeKey("hs-storkSubsteps", v); },
  setBeatStability: (v: any) => { set({ beatStability: v }); writeKey("hs-beatStability", v); },
  setFrequencyDamping: (v: any) => { set({ frequencyDamping: v }); writeKey("hs-frequencyDamping", v); },
  setTemporalSmoothing: (v: any) => { set({ temporalSmoothing: v }); writeKey("hs-temporalSmoothing", v); },
  setApgMomentum: (v: any) => { set({ apgMomentum: v }); writeKey("hs-apgMomentum", v); },
  setApgNormThreshold: (v: any) => { set({ apgNormThreshold: v }); writeKey("hs-apgNormThreshold", v); },
  setDcwEnabled: (v: any) => { set({ dcwEnabled: v }); writeKey("hs-dcwEnabled", v); },
  setDcwMode: (v: any) => { set({ dcwMode: v }); writeKey("hs-dcwMode", v); },
  setDcwLowScaler: (v: any) => { set({ dcwLowScaler: v }); writeKey("hs-dcwLowScaler", v); },
  setDcwHighScaler: (v: any) => { set({ dcwHighScaler: v }); writeKey("hs-dcwHighScaler", v); },
  setLatentShift: (v: any) => { set({ latentShift: v }); writeKey("hs-latentShift", v); },
  setLatentRescale: (v: any) => { set({ latentRescale: v }); writeKey("hs-latentRescale", v); },
  setCustomTimesteps: (v: any) => { set({ customTimesteps: v }); writeKey("hs-customTimesteps", v); },
  setDenoiseStrength: (v: any) => { set({ denoiseStrength: v }); writeKey("hs-denoiseStrength", v); },
  setDenoiseSmoothing: (v: any) => { set({ denoiseSmoothing: v }); writeKey("hs-denoiseSmoothing", v); },
  setDenoiseMix: (v: any) => { set({ denoiseMix: v }); writeKey("hs-denoiseMix", v); },
  setLssStrength: (v: any) => { set({ lssStrength: v }); writeKey("hs-lssStrength", v); },
  setLssVarThresh: (v: any) => { set({ lssVarThresh: v }); writeKey("hs-lssVarThresh", v); },
  setLssDcRemove: (v: any) => { set({ lssDcRemove: v }); writeKey("hs-lssDcRemove", v); },
  setAutoTrimEnabled: (v: any) => { set({ autoTrimEnabled: v }); writeKey("hs-autoTrimEnabled", v); },
  setDurationBuffer: (v: any) => { set({ durationBuffer: v }); writeKey("hs-durationBuffer", v); },
  setAutoTrimFadeMs: (v: any) => { set({ autoTrimFadeMs: v }); writeKey("hs-autoTrimFadeMs", v); },
  setSkipLm: (v: any) => { set({ skipLm: v }); writeKey("hs-skipLm", v); },
  setSkipLrc: (v: any) => { set({ skipLrc: v }); writeKey("hs-skipLrc", v); },
  setUseCotCaption: (v: any) => { set({ useCotCaption: v }); writeKey("hs-useCotCaption", v); },
  setLmTemperature: (v: any) => { set({ lmTemperature: v }); writeKey("hs-lmTemperature", v); },
  setLmCfgScale: (v: any) => { set({ lmCfgScale: v }); writeKey("hs-lmCfgScale", v); },
  setLmTopK: (v: any) => { set({ lmTopK: v }); writeKey("hs-lmTopK", v); },
  setLmTopP: (v: any) => { set({ lmTopP: v }); writeKey("hs-lmTopP", v); },
  setLmRepPenalty: (v: any) => { set({ lmRepPenalty: v }); writeKey("hs-lmRepPenalty", v); },
  setLmRepMode: (v: any) => { set({ lmRepMode: v }); writeKey("hs-lmRepMode", v); },
  setLmDryBase: (v: any) => { set({ lmDryBase: v }); writeKey("hs-lmDryBase", v); },
  setLmDryMinLen: (v: any) => { set({ lmDryMinLen: v }); writeKey("hs-lmDryMinLen", v); },
  setLmRepWindow: (v: any) => { set({ lmRepWindow: v }); writeKey("hs-lmRepWindow", v); },
  setLmNegativePrompt: (v: any) => { set({ lmNegativePrompt: v }); writeKey("hs-lmNegativePrompt", v); },
  setLmCodesStrength: (v: any) => { set({ lmCodesStrength: v }); writeKey("hs-lmCodesStrength", v); },
  setLmCodesMode: (v: any) => { set({ lmCodesMode: v }); writeKey("hs-lmCodesMode", v); },
  setLmCodesSteps: (v: any) => { set({ lmCodesSteps: v }); writeKey("hs-lmCodesSteps", v); },
  setPostProcessingEnabled: (v: any) => { set({ postProcessingEnabled: v }); writeKey("hs-postProcessingEnabled", v); },
  setSpectralLifterEnabled: (v: any) => { set({ spectralLifterEnabled: v }); writeKey("hs-spectralLifterEnabled", v); },
  setSlDenoiseStrength: (v: any) => { set({ slDenoiseStrength: v }); writeKey("hs-slDenoiseStrength", v); },
  setSlNoiseFloor: (v: any) => { set({ slNoiseFloor: v }); writeKey("hs-slNoiseFloor", v); },
  setSlHfMix: (v: any) => { set({ slHfMix: v }); writeKey("hs-slHfMix", v); },
  setSlTransientBoost: (v: any) => { set({ slTransientBoost: v }); writeKey("hs-slTransientBoost", v); },
  setSlShimmerReduction: (v: any) => { set({ slShimmerReduction: v }); writeKey("hs-slShimmerReduction", v); },
  setMasteringEnabled: (v: any) => { set({ masteringEnabled: v }); writeKey("hs-masteringEnabled", v); },
  setMasteringReference: (v: any) => { set({ masteringReference: v }); writeKey("hs-masteringReference", v); },
  setTimbreReference: (v: any) => { set({ timbreReference: v }); writeKey("hs-timbreReference", v); },
  setTimbreAudioPath: (v: any) => { set({ timbreAudioPath: v }); writeKey("hs-timbreAudioPath", v); },
  setVocalNaturalizerEnabled: (v: any) => { set({ vocalNaturalizerEnabled: v }); writeKey("hs-vocalNaturalizerEnabled", v); },
  setGainOffsetDb: (v: any) => { set({ gainOffsetDb: v }); writeKey("hs-gainOffsetDb", v); },
  setNaturalizeAmount: (v: any) => { set({ naturalizeAmount: v }); writeKey("hs-naturalizeAmount", v); },
  setNatVibratoRate: (v: any) => { set({ natVibratoRate: v }); writeKey("hs-natVibratoRate", v); },
  setNatVibratoDepth: (v: any) => { set({ natVibratoDepth: v }); writeKey("hs-natVibratoDepth", v); },
  setNatFormantStrength: (v: any) => { set({ natFormantStrength: v }); writeKey("hs-natFormantStrength", v); },
  setNatMetallicReduction: (v: any) => { set({ natMetallicReduction: v }); writeKey("hs-natMetallicReduction", v); },
  setNatQuantizationMask: (v: any) => { set({ natQuantizationMask: v }); writeKey("hs-natQuantizationMask", v); },
  setNatTransitionSmooth: (v: any) => { set({ natTransitionSmooth: v }); writeKey("hs-natTransitionSmooth", v); },
  setPpVaeReencode: (v: any) => { set({ ppVaeReencode: v }); writeKey("hs-ppVaeReencode", v); },
  setPpVaeBlend: (v: any) => { set({ ppVaeBlend: v }); writeKey("hs-ppVaeBlend", v); },
  setPpVaeUseOnnx: (v: any) => { set({ ppVaeUseOnnx: v }); writeKey("hs-ppVaeUseOnnx", v); },
  setStableStepOn: (v: any) => { set({ stableStepOn: v }); writeKey("hs-stableStepOn", v); },
  setStableStepStrength: (v: any) => { set({ stableStepStrength: v }); writeKey("hs-stableStepStrength", v); },
  setStableStepBackend: (v: any) => { set({ stableStepBackend: v }); writeKey("hs-stableStepBackend", v); },
  setStableStepAdapters: (v: any) => { set({ stableStepAdapters: v }); writeKey("hs-stableStepAdapters", v); },
  setStableStepPreserveDynamics: (v: any) => { set({ stableStepPreserveDynamics: v }); writeKey("hs-stableStepPreserveDynamics", v); },
  setStableStepVocalPpVae: (v: any) => { set({ stableStepVocalPpVae: v }); writeKey("hs-stableStepVocalPpVae", v); },
  setStableStepBlendMode: (v: any) => { set({ stableStepBlendMode: v }); writeKey("hs-stableStepBlendMode", v); },
  setStableStepCrossoverHz: (v: any) => { set({ stableStepCrossoverHz: v }); writeKey("hs-stableStepCrossoverHz", v); },
  setStableStepCrossoverWidthHz: (v: any) => { set({ stableStepCrossoverWidthHz: v }); writeKey("hs-stableStepCrossoverWidthHz", v); },
  setStableStepMix: (v: any) => { set({ stableStepMix: v }); writeKey("hs-stableStepMix", v); },
  setStableStepSeed: (v: any) => { set({ stableStepSeed: v }); writeKey("hs-stableStepSeed", v); },
  setStableStepSeedFollowsDit: (v: any) => { set({ stableStepSeedFollowsDit: v }); writeKey("hs-stableStepSeedFollowsDit", v); },
  setStableStepSteps: (v: any) => { set({ stableStepSteps: v }); writeKey("hs-stableStepSteps", v); },
  setStableStepSolver: (v: any) => { set({ stableStepSolver: v }); writeKey("hs-stableStepSolver", v); },
  setStableStepScheduler: (v: any) => { set({ stableStepScheduler: v }); writeKey("hs-stableStepScheduler", v); },
  setStableStepGuidanceMode: (v: any) => { set({ stableStepGuidanceMode: v }); writeKey("hs-stableStepGuidanceMode", v); },
  setStableStepGuidanceScale: (v: any) => { set({ stableStepGuidanceScale: v }); writeKey("hs-stableStepGuidanceScale", v); },
  setCoverArtEnabled: (v: any) => { set({ coverArtEnabled: v }); writeKey("hs-coverArtEnabled", v); },
  setCoverArtSubject: (v: any) => { set({ coverArtSubject: v }); writeKey("hs-coverArtSubject", v); },
  setQualityEvalEnabled: (v: any) => { set({ qualityEvalEnabled: v }); writeKey("hs-qualityEvalEnabled", v); },
  setQualityEvalTarget: (v: any) => { set({ qualityEvalTarget: v }); writeKey("hs-qualityEvalTarget", v); },
  setWhisperLyricsEnabled: (v: any) => { set({ whisperLyricsEnabled: v }); writeKey("hs-whisperLyricsEnabled", v); },
  setWhisperModel: (v: any) => { set({ whisperModel: v }); writeKey("hs-whisperModel", v); },
  setWhisperLanguage: (v: any) => { set({ whisperLanguage: v }); writeKey("hs-whisperLang", v); },
  setWhisperBeamSize: (v: any) => { set({ whisperBeamSize: v }); writeKey("hs-whisperBeam", v); },
  setWhisperIsolateVocals: (v: any) => { set({ whisperIsolateVocals: v }); writeKey("hs-whisperIsolate", v); },
  setPostprocessEnabled: (v: any) => { set({ postprocessEnabled: v }); writeKey("hs-postprocessEnabled", v); },
  setPostprocessPlugin: (v: any) => { set({ postprocessPlugin: v }); writeKey("hs-postprocessPlugin", v); },
  setUseOrtVae: (v: any) => { set({ useOrtVae: v }); writeKey('hs-useOrtVae', v); },
  setLufsEnabled: (v: any) => { set({ lufsEnabled: v }); writeKey('hs-lufsEnabled', v); },
  setLufsPreset: (v: any) => {
    set({ lufsPreset: v });
    writeKey('hs-lufsPreset', v);
    // Auto-set target from preset
    const presetTargets: Record<string, number> = {
      spotify: -14, apple: -16, ebu: -23, club: -8,
    };
    if (v !== 'custom' && presetTargets[v] !== undefined) {
      set({ lufsTarget: presetTargets[v] });
      writeKey('hs-lufsTarget', presetTargets[v]);
    }
  },
  setLufsTarget: (v: any) => { set({ lufsTarget: v }); writeKey('hs-lufsTarget', v); },

  // Plugin param helpers
  setPluginParam: (key: string, value: string) => {
    const prev = get().pluginParams;
    const next = { ...prev, [key]: value };
    set({ pluginParams: next });
    writeKey(scopedKey('hs-pluginParams'), next);
  },
  resetPluginParams: (pluginName: string) => {
    const prev = get().pluginParams;
    const next: Record<string, string> = {};
    for (const [k, v] of Object.entries(prev)) {
      if (!k.startsWith(pluginName + ':')) next[k] = v as string;
    }
    set({ pluginParams: next });
    writeKey(scopedKey('hs-pluginParams'), next);
  },

  // -- Derived: assemble generation params --
  getGlobalParams: (): Partial<GenerationParams> => {
    const s = get();
    const settings: AppSettings = readKey('ace-settings', DEFAULT_SETTINGS);

    // Effective adapter stack: the multi-adapter list when in Advanced mode,
    // otherwise the single `adapter` folded into a one-element stack. `primary`
    // drives the single-adapter features (trigger word, basin re-base, group
    // scales) which remain keyed on the first adapter. Gated on advancedAdapters
    // to match the badge/UI — a stack persisted from a previous Advanced session
    // must not override the Simple-mode selection.
    const isStack = !!(s.advancedAdapters && s.adapterStack && s.adapterStack.length > 0);
    const rawStack: { path: string; scale: number; stepStart?: number; stepEnd?: number }[] = isStack
      ? s.adapterStack
      : (s.adapter ? [{ path: s.adapter, scale: s.adapterScale }] : []);

    // Blend mode (multi-adapter stacks only): treat each entry's `scale` as a
    // relative weight and normalise so the effective scales sum to the budget,
    // keeping combined strength constant regardless of how many are stacked.
    // Sum mode (and the single-adapter fallback) sends the raw scales as-is.
    // Blend only applies with 2+ adapters — a single adapter's strength is just
    // its own scale, sent as-is.
    let stack = rawStack;
    if (isStack && s.adapterStackMode === 'blend' && rawStack.length >= 2) {
      const budget = s.adapterStackBudget ?? 0.75;
      const sumW = rawStack.reduce((acc, e) => acc + (e.scale || 0), 0);
      // Spread each entry so per-adapter extras (stepStart/stepEnd timestep
      // window, etc.) survive the blend re-scale.
      stack = sumW > 0
        ? rawStack.map(e => ({ ...e, scale: +(budget * (e.scale || 0) / sumW).toFixed(4) }))
        : rawStack.map(e => ({ ...e, scale: +(budget / rawStack.length).toFixed(4) }));
    }
    const primary = stack[0]?.path || '';
    // Timestep windows force runtime mode server-side regardless of the selected
    // adapter mode — runtime-only knobs must flow whenever they're present.
    const hasStepWindows = stack.some(
      (e: { stepStart?: number; stepEnd?: number }) => e.stepStart !== undefined || e.stepEnd !== undefined,
    );

    // Trigger words. The server resolves each adapter's EMBEDDED trigger from
    // its safetensors metadata — that is what the adapter was actually trained
    // with, so it wins and needs nothing from us. What we send is the filename
    // fallback for adapters that carry no embedded trigger, tagged with the
    // adapter path so the server can match it per-adapter rather than applying
    // it to the whole stack.
    // docs/plans/2026-07-28-adapter-trigger-embedding.md T6
    const triggerPlacement = (settings.triggerPlacement || 'prepend') as 'prepend' | 'append' | 'replace';
    const triggerSpecs = settings.triggerUseFilename
      ? stack
          .map(e => ({
            word: e.path.split(/[\\\/]/).pop()?.replace(/\.safetensors$/i, '') || '',
            placement: triggerPlacement,
            source: 'filename' as const,
            path: e.path,
          }))
          .filter(s => s.word)
      : [];
    const triggerWords: string[] = triggerSpecs.map(s => s.word);
    const triggerWord = triggerWords.join(', ');

    // LM codes window → audio_cover_strength fraction. Steps mode converts the
    // absolute count against the current step budget; +0.5 lands mid-bin so the
    // engine's floor(num_steps * strength) yields exactly that many steps
    // despite f32 rounding. At or above the budget → 1.0 (no silence switch).
    const lmCodesEff = s.lmCodesMode === 'steps'
      ? (s.lmCodesSteps >= s.inferenceSteps ? 1.0 : Math.max(0, s.lmCodesSteps + 0.5) / s.inferenceSteps)
      : s.lmCodesStrength;

    return {
      // Backend-declared knobs (capabilities().extensions), flattened alongside
      // the core params. Spread FIRST on purpose: later properties win in an
      // object literal, so a backend can never shadow a core field by picking a
      // colliding key.
      ...(s.backendParams || {}),
      // Multi-backend: which engine backend this request targets. Read directly
      // from backendStore (not a globalParamsStore field) so it's always the
      // live active id; defaults to 'ace' for installs with no second backend
      // registered (docs/plans/multi-backend-architecture.md §4.5).
      backend: useBackendStore.getState().activeBackendId || 'ace',
      ditModel: s.ditModel, lmModel: s.lmModel, vaeModel: s.vaeModel, embeddingModel: s.embeddingModel,
      // Planner-LM adapter (runtime LoRA on the 5Hz LM) — aceReq fields, so
      // they survive the LM-echo synth rebuild by construction.
      // lmAdapterScale is emitted UNCONDITIONALLY: like the DiT loraScale, the
      // global strength governs album-preset planner adapters too (the preset
      // supplies only the path).
      lmAdapter: s.lmAdapter || undefined,
      lmAdapterScale: s.lmAdapterScale,
      loraPath: primary, loraScale: stack[0]?.scale ?? 1.0,
      // Multi-adapter stack (>1 entry) — sent alongside loraPath; the engine
      // prefers the stack and applies each adapter with its own scale.
      loraStack: stack.length > 0 ? stack : undefined,
      // Stack scaling mode + budget — reused for per-section masking transforms.
      adapterStackMode: s.adapterStackMode,
      adapterStackBudget: s.adapterStackBudget,
      // Per-section masking tuning (only meaningful with a 2+ adapter stack).
      adapterSectionAlignAt: stack.length >= 2 ? s.adapterSectionAlignAt : undefined,
      adapterSectionIsolation: stack.length >= 2 ? s.adapterSectionIsolation : undefined,
      adapterGroupScales: primary ? s.adapterGroupScales : undefined,
      adapterMode: primary ? s.adapterMode : 'merge',
      // Runtime delta quantization (VRAM saver) — relevant in both runtime modes
      // (lowrank still stores full-size re-base corrections / Conv1d fallbacks).
      // Also sent when timestep windows are active: they force runtime mode
      // server-side even from Merge, and gating on the *selected* mode silently
      // killed the knob there (full BF16 deltas, 2×8 GB — the 32 GB bug).
      adapterRuntimeQuant: (primary && (s.adapterMode === 'runtime' || s.adapterMode === 'runtime_lowrank' || hasStepWindows))
        ? s.adapterRuntimeQuant : undefined,
      // Merge low-VRAM storage (native-quant re-encode) — only relevant in merge mode.
      adapterMergeLowVram: (primary && s.adapterMode !== 'runtime' && s.adapterMergeLowVram) ? true : undefined,
      // No-adapter reference render — only meaningful with a DiT adapter loaded.
      noAdapterRender: (primary && s.noAdapterRender) ? true : undefined,
      // Basin re-base: only sent with an adapter and a chosen source. Works in
      // both merge and runtime modes (runtime folds the nudge into the delta sum);
      // the engine skips it on the per-section masking path.
      rebaseSource: (primary && s.rebaseSource) ? s.rebaseSource : undefined,
      rebaseBeta: (primary && s.rebaseSource) ? s.rebaseBeta : undefined,
      triggerSpecs: triggerSpecs.length ? triggerSpecs : undefined,
      triggerWord: triggerWord || undefined,
      triggerWords: triggerWords.length ? triggerWords : undefined,
      // '|| prepend' fallback matches the queue path (audioGenQueueStore):
      // an ace-settings object saved before triggerPlacement existed has the
      // key undefined, and translateParams skips injection entirely without a
      // placement — which silently dropped trigger words on Create/custom-gen.
      triggerPlacement: triggerWords.length ? triggerPlacement : undefined,
      inferenceSteps: s.inferenceSteps, guidanceScale: s.guidanceScale, shift: s.shift,
      // ditSlidingWindow deliberately not sent while the feature is parked, so
      // requests are byte-identical to pre-feature generations.
      cfgCutoffRatio: s.cfgCutoffRatio < 1.0 ? s.cfgCutoffRatio : undefined,
      lmCfgCutoffRatio: s.lmCfgCutoffRatio < 1.0 ? s.lmCfgCutoffRatio : undefined,
      cacheRatio: s.cacheRatio > 0 ? s.cacheRatio : undefined,
      inferMethod: s.inferMethod, scheduler: s.scheduler, guidanceMode: s.guidanceMode,
      seed: s.seed, randomSeed: s.randomSeed,
      lmSeed: s.lmSeed, lmSeedFollowsDit: s.lmSeedFollowsDit,
      batchSize: s.batchSize,
      storkSubsteps: (s.inferMethod === 'stork2' || s.inferMethod === 'stork4') ? s.storkSubsteps : undefined,
      beatStability: s.inferMethod === 'jkass_fast' ? s.beatStability : undefined,
      frequencyDamping: s.inferMethod === 'jkass_fast' ? s.frequencyDamping : undefined,
      temporalSmoothing: s.inferMethod === 'jkass_fast' ? s.temporalSmoothing : undefined,
      apgMomentum: s.guidanceMode === 'apg' ? s.apgMomentum : undefined,
      apgNormThreshold: s.guidanceMode === 'apg' ? s.apgNormThreshold : undefined,
      skipLm: s.skipLm, useCotCaption: s.useCotCaption,
      skipLrc: s.skipLrc || undefined,
      lmTemperature: s.lmTemperature, lmCfgScale: s.lmCfgScale,
      lmTopK: s.lmTopK, lmTopP: s.lmTopP, lmNegativePrompt: s.lmNegativePrompt,
      lmRepPenalty: s.lmRepPenalty > 1.0 ? s.lmRepPenalty : undefined,
      lmRepWindow: s.lmRepPenalty > 1.0 ? s.lmRepWindow : undefined,
      lmRepMode: s.lmRepPenalty > 1.0 ? s.lmRepMode : undefined,
      lmDryBase: (s.lmRepPenalty > 1.0 && s.lmRepMode === 'dry') ? s.lmDryBase : undefined,
      lmDryMinLen: (s.lmRepPenalty > 1.0 && s.lmRepMode === 'dry') ? s.lmDryMinLen : undefined,
      audioCoverStrength: (!s.skipLm && lmCodesEff < 1.0) ? lmCodesEff : undefined,
      postProcessingEnabled: s.postProcessingEnabled,
      spectralLifterEnabled: s.postProcessingEnabled ? s.spectralLifterEnabled : false,
      slDenoiseStrength: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slDenoiseStrength : undefined,
      slNoiseFloor: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slNoiseFloor : undefined,
      slHfMix: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slHfMix : undefined,
      slTransientBoost: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slTransientBoost : undefined,
      slShimmerReduction: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slShimmerReduction : undefined,
      masteringEnabled: s.postProcessingEnabled ? s.masteringEnabled : false,
      masteringReference: (s.postProcessingEnabled && s.masteringEnabled) ? s.masteringReference : undefined,
      timbreReference: s.timbreAudioPath
        ? s.timbreAudioPath
        : (s.postProcessingEnabled && s.masteringEnabled && s.timbreReference && s.masteringReference) ? true : undefined,
      dcwEnabled: s.dcwEnabled,
      dcwMode: s.dcwEnabled ? s.dcwMode : undefined,
      // Route the correct scaler to dcw_scaler based on mode:
      // low/double/pix use dcwLowScaler, high uses dcwHighScaler
      dcwScaler: s.dcwEnabled
        ? (s.dcwMode === 'high' ? s.dcwHighScaler * 0.02 : s.dcwLowScaler * 0.05)
        : undefined,
      dcwHighScaler: (s.dcwEnabled && s.dcwMode === 'double') ? s.dcwHighScaler * 0.02 : undefined,
      latentShift: s.latentShift !== 0 ? s.latentShift : undefined,
      latentRescale: s.latentRescale !== 1 ? s.latentRescale : undefined,
      customTimesteps: s.customTimesteps || undefined,
      denoiseStrength: s.denoiseStrength > 0 ? s.denoiseStrength : undefined,
      denoiseSmoothing: s.denoiseStrength > 0 ? s.denoiseSmoothing : undefined,
      denoiseMix: s.denoiseStrength > 0 ? s.denoiseMix : undefined,
      lssStrength: s.lssStrength > 0 ? s.lssStrength : undefined,
      lssVarThresh: s.lssStrength > 0 ? s.lssVarThresh : undefined,
      lssDcRemove: s.lssStrength > 0 ? s.lssDcRemove : undefined,
      pluginParams: Object.keys(s.pluginParams).length > 0 ? s.pluginParams : undefined,
      autoTrimEnabled: s.autoTrimEnabled || undefined,
      durationBuffer: s.autoTrimEnabled ? s.durationBuffer : undefined,
      autoTrimFadeMs: s.autoTrimEnabled ? s.autoTrimFadeMs : undefined,
      vocalNaturalizerEnabled: s.postProcessingEnabled ? s.vocalNaturalizerEnabled : false,
      gainOffsetDb: (s.postProcessingEnabled && s.gainOffsetDb !== 0) ? s.gainOffsetDb : undefined,
      naturalizeAmount: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.naturalizeAmount : undefined,
      natVibratoRate: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natVibratoRate : undefined,
      natVibratoDepth: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natVibratoDepth : undefined,
      natFormantStrength: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natFormantStrength : undefined,
      natMetallicReduction: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natMetallicReduction : undefined,
      natQuantizationMask: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natQuantizationMask : undefined,
      natTransitionSmooth: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natTransitionSmooth : undefined,
      ppVaeReencode: (s.postProcessingEnabled && s.ppVaeReencode) || undefined,
      ppVaeBlend: (s.postProcessingEnabled && s.ppVaeReencode && s.ppVaeBlend > 0) ? s.ppVaeBlend : undefined,
      ppVaeUseOnnx: (s.postProcessingEnabled && s.ppVaeReencode) ? s.ppVaeUseOnnx : undefined,
      stableStepOn: (s.postProcessingEnabled && s.stableStepOn) || undefined,
      stableStepStrength: (s.postProcessingEnabled && s.stableStepOn) ? s.stableStepStrength : undefined,
      stableStepBackend: (s.postProcessingEnabled && s.stableStepOn && s.stableStepBackend !== 'auto')
        ? s.stableStepBackend : undefined,
      stableStepAdapters: (s.postProcessingEnabled && s.stableStepOn)
        ? (s.stableStepAdapters ?? [])
            .filter((a: any) => a.enabled && a.scale !== 0)
            .map((a: any) => ({ name: a.name, scale: a.scale }))
        : undefined,
      stableStepPreserveDynamics: (s.postProcessingEnabled && s.stableStepOn)
        ? s.stableStepPreserveDynamics !== false : undefined,
      // Opt-in only — omitted means "leave the AS1.5 vocals alone"
      stableStepVocalPpVae: (s.postProcessingEnabled && s.stableStepOn && s.stableStepVocalPpVae)
        || undefined,
      stableStepBlendMode: (s.postProcessingEnabled && s.stableStepOn && s.stableStepBlendMode !== 'off')
        ? s.stableStepBlendMode : undefined,
      stableStepCrossoverHz: (s.postProcessingEnabled && s.stableStepOn && s.stableStepBlendMode === 'crossover')
        ? s.stableStepCrossoverHz : undefined,
      stableStepCrossoverWidthHz: (s.postProcessingEnabled && s.stableStepOn && s.stableStepBlendMode === 'crossover')
        ? s.stableStepCrossoverWidthHz : undefined,
      stableStepMix: (s.postProcessingEnabled && s.stableStepOn && s.stableStepBlendMode === 'mix')
        ? s.stableStepMix : undefined,
      stableStepSeedFollowsDit: (s.postProcessingEnabled && s.stableStepOn)
        ? s.stableStepSeedFollowsDit !== false : undefined,
      stableStepSeed: (s.postProcessingEnabled && s.stableStepOn && s.stableStepSeedFollowsDit === false)
        ? s.stableStepSeed : undefined,
      // SA3 sampler routing. Each field is emitted only when it departs from the
      // engine default, so an untouched StableStep keeps the original
      // pingpong/euler path with nothing extra on the wire.
      stableStepSteps: (s.postProcessingEnabled && s.stableStepOn && s.stableStepSteps !== 8)
        ? s.stableStepSteps : undefined,
      stableStepSolver: (s.postProcessingEnabled && s.stableStepOn && s.stableStepSolver)
        ? s.stableStepSolver : undefined,
      stableStepScheduler: (s.postProcessingEnabled && s.stableStepOn && s.stableStepScheduler)
        ? s.stableStepScheduler : undefined,
      stableStepGuidanceMode: (s.postProcessingEnabled && s.stableStepOn && s.stableStepGuidanceMode)
        ? s.stableStepGuidanceMode : undefined,
      stableStepGuidanceScale: (s.postProcessingEnabled && s.stableStepOn && s.stableStepGuidanceMode
                                && s.stableStepGuidanceScale > 1)
        ? s.stableStepGuidanceScale : undefined,
      // Lua plugins read their declared params out of the shared pluginParams
      // map (keyed "<plugin>:<key>"), the same one the generation dropdowns
      // populate — so a plugin's knobs work identically in both places.
      stableStepPluginParams: (s.postProcessingEnabled && s.stableStepOn
                               && (s.stableStepSolver || s.stableStepScheduler || s.stableStepGuidanceMode))
        ? s.pluginParams : undefined,
      coverArtEnabled: (s.postProcessingEnabled && s.coverArtEnabled) || undefined,
      coverArtSubject: (s.postProcessingEnabled && s.coverArtEnabled && s.coverArtSubject) ? s.coverArtSubject : undefined,
      qualityEvalEnabled: (s.postProcessingEnabled && s.qualityEvalEnabled) || undefined,
      qualityEvalTarget: (s.postProcessingEnabled && s.qualityEvalEnabled) ? s.qualityEvalTarget : undefined,
      postprocessPlugin: (s.postProcessingEnabled && s.postprocessEnabled && s.postprocessPlugin) ? s.postprocessPlugin : undefined,
      lufsEnabled: (s.postProcessingEnabled && s.masteringEnabled && s.lufsEnabled) || undefined,
      lufsTarget: (s.postProcessingEnabled && s.masteringEnabled && s.lufsEnabled) ? s.lufsTarget : undefined,
      useOrtVae: s.useOrtVae || undefined,
      whisperLyricsEnabled: s.whisperLyricsEnabled,
      whisperModel: s.whisperLyricsEnabled ? s.whisperModel : undefined,
      whisperLanguage: s.whisperLyricsEnabled ? s.whisperLanguage : undefined,
      whisperBeamSize: s.whisperLyricsEnabled ? s.whisperBeamSize : undefined,
      whisperIsolateVocals: s.whisperLyricsEnabled ? s.whisperIsolateVocals : undefined,
    };
  },
}));

// -- Backend switch -> re-read the per-backend fields --
//
// One subscription rather than a call inside backendStore.switchBackend: this
// file already imports backendStore, and the reverse import would be a cycle.
// It also catches the OTHER way the active id moves -- fetchBackends() at boot
// resolving to whatever the server persisted, which the initial hydrate above
// could only guess at.
//
// Nothing is written here: hydrateBackendScoped() reads (and, on a backend's
// first visit, seeds) localStorage, so the settings the user leaves behind on
// the backend they are switching AWAY from were already persisted by its
// setters.
let lastScopedBackendId = activeBackendId();
useBackendStore.subscribe((st) => {
  const id = st.activeBackendId || 'ace';
  if (id === lastScopedBackendId) return;
  lastScopedBackendId = id;
  useGlobalParamsStore.setState(hydrateBackendScoped(id));
});

// One-shot boot sync. An existing install already has a DiT chosen in
// localStorage that the server has never been told about, and a user who never
// touches the dropdown again would otherwise never mirror it — leaving the
// Training Studio on its old "first BF16 in the catalogue" fallback.
mirrorActiveModels({
  ditModel: useGlobalParamsStore.getState().ditModel || '',
  lmModel: useGlobalParamsStore.getState().lmModel || '',
  vaeModel: useGlobalParamsStore.getState().vaeModel || '',
});
