import { z } from 'zod/v4';

/**
 * Create's full control dictionary: every field Node defaults or translates
 * for a Create (text2music) generation request, across the ace/yue2/
 * minimax-m3 backends. See docs/dev/frontend-create-controls.md for the
 * per-field default, backend applicability and omitted-vs-explicit table —
 * this file is the documented key list, that page is the narrative.
 *
 * Every field is `z.unknown().optional()` DELIBERATELY, not merely optional:
 * the existing `params: z.record(z.string(), z.unknown())` field on
 * `generationIntentSchema` (generation.ts) never checks a known field's
 * *type*, only that the body is an object, so a typed field here (e.g.
 * `z.number()`) would reject `{ inferenceSteps: "12" }` or
 * `{ skipLm: null }` even though the live route and the old schema both
 * accept them — a silent tightening of a published contract. Keep this
 * schema's job to "which keys exist and what they default to" (the comments
 * below), not "what shape must a caller send" — generationControls.test.ts
 * asserts the two schemas agree on every fixture, including those exact
 * type-mismatch and null cases. Unknown keys pass through via `.catchall`
 * for the same reason: a future control or a plugin/backend extension field
 * must never need a schema change here to keep working.
 */
export const generationControlsSchema = z.object({
  // Model routing
  ditModel: z.unknown().optional(), // string, default ''
  lmModel: z.unknown().optional(), // string, default ''
  vaeModel: z.unknown().optional(), // string, default ''
  embeddingModel: z.unknown().optional(), // string, default ''

  // Adapter (DiT LoRA) — single-slot Simple mode and multi-entry Advanced stack
  adapter: z.unknown().optional(), // string, default ''
  adapterScale: z.unknown().optional(), // number, default 1.0
  adapterStack: z.unknown().optional(), // { path, scale, stepStart?, stepEnd?, stepSoft?, gainCurve?, gainDomain? }[], default []
  adapterStackMode: z.unknown().optional(), // 'blend' | 'sum', default 'blend'
  adapterStackBudget: z.unknown().optional(), // number, default 0.75
  adapterSectionAlignAt: z.unknown().optional(), // number, default 0.55
  adapterSectionIsolation: z.unknown().optional(), // number, default 0.5
  adapterMode: z.unknown().optional(), // 'merge' | 'runtime' | 'runtime_lowrank', default 'runtime'
  adapterRuntimeQuant: z.unknown().optional(), // string, default 'bf16'
  adapterMergeLowVram: z.unknown().optional(), // boolean, default false
  adapterGroupScales: z.unknown().optional(), // { self_attn, cross_attn, mlp, cond_embed, time_embed, proj_in: number }
  rebaseSource: z.unknown().optional(), // string, default ''
  rebaseBeta: z.unknown().optional(), // number, default 0.75
  adapterFolder: z.unknown().optional(), // string, default ''
  lmAdapterFolder: z.unknown().optional(), // string, default ''
  advancedAdapters: z.unknown().optional(), // boolean, default false
  noAdapterRender: z.unknown().optional(), // boolean, default false
  adaptersOpen: z.unknown().optional(), // boolean, default false

  // Planner-LM runtime LoRA (independent of the DiT adapter stack above)
  lmAdapter: z.unknown().optional(), // string, default ''
  lmAdapterScale: z.unknown().optional(), // number, default 1.0

  // Core DiT sampling
  inferenceSteps: z.unknown().optional(), // number, default 12
  guidanceScale: z.unknown().optional(), // number, default 9.0
  cfgCutoffRatio: z.unknown().optional(), // number, default 1.0 (no-op)
  lmCfgCutoffRatio: z.unknown().optional(), // number, default 1.0 (no-op)
  cacheRatio: z.unknown().optional(), // number, default 0 (off)
  shift: z.unknown().optional(), // number, default 3.0
  ditSlidingWindow: z.unknown().optional(), // number, unset = model default (128)
  inferMethod: z.unknown().optional(), // string, default 'euler'
  scheduler: z.unknown().optional(), // string, default 'linear'
  guidanceMode: z.unknown().optional(), // string, default 'apg'
  storkSubsteps: z.unknown().optional(), // number, default 10
  beatStability: z.unknown().optional(), // number, default 0.25
  frequencyDamping: z.unknown().optional(), // number, default 0.4
  temporalSmoothing: z.unknown().optional(), // number, default 0.13
  apgMomentum: z.unknown().optional(), // number, default 0.75
  apgNormThreshold: z.unknown().optional(), // number, default 2.5

  // Seeding / batching
  seed: z.unknown().optional(), // number, default 42
  randomSeed: z.unknown().optional(), // boolean, default true
  lmSeed: z.unknown().optional(), // number, default 42
  lmSeedFollowsDit: z.unknown().optional(), // boolean, default true
  batchSize: z.unknown().optional(), // number, default 1

  // Metadata / song-level fields (legacy flat-body names; see translateParams.ts)
  bpm: z.unknown().optional(), // number, unset
  keyScale: z.unknown().optional(), // string, unset
  timeSignature: z.unknown().optional(), // string, unset
  vocalLanguage: z.unknown().optional(), // string, UI defaults to 'en'

  // Task mode and cover/repaint axis (ACE-only; YuE2/MM3 accept text2music only)
  taskType: z.unknown().optional(), // string, unset -> backend default 'text2music'
  audioCoverStrength: z.unknown().optional(), // number, derived from lmCodes*
  coverNoiseStrength: z.unknown().optional(), // number, unset
  coverNoiseMethod: z.unknown().optional(), // string, unset
  repaintingStart: z.unknown().optional(), // number, unset
  repaintingEnd: z.unknown().optional(), // number, unset
  seedStrength: z.unknown().optional(), // number, unset
  evictLm: z.unknown().optional(), // boolean, default false
  vaeChunk: z.unknown().optional(), // number | boolean, unset
  batchCfg: z.unknown().optional(), // boolean, unset
  trackName: z.unknown().optional(), // string, unset

  // LM / planner sampling
  skipLm: z.unknown().optional(), // boolean, default false
  skipLrc: z.unknown().optional(), // boolean, default false
  useCotCaption: z.unknown().optional(), // boolean, default true
  lmTemperature: z.unknown().optional(), // number, default 0.8
  lmCfgScale: z.unknown().optional(), // number, default 2.2
  lmTopK: z.unknown().optional(), // number, default 0 (meaningful — disables top-k)
  lmTopP: z.unknown().optional(), // number, default 0.92
  lmRepPenalty: z.unknown().optional(), // number, default 1.1
  lmRepWindow: z.unknown().optional(), // number, default 64
  lmRepMode: z.unknown().optional(), // 'presence' | 'dry', default 'presence'
  lmDryBase: z.unknown().optional(), // number, default 1.75
  lmDryMinLen: z.unknown().optional(), // number, default 3
  lmNegativePrompt: z.unknown().optional(), // string, default 'NO USER INPUT'
  negativePrompt: z.unknown().optional(), // string, unset
  lmCodesStrength: z.unknown().optional(), // number, default 1.0
  lmCodesMode: z.unknown().optional(), // 'ratio' | 'steps', default 'ratio'
  lmCodesSteps: z.unknown().optional(), // number, default 6

  // DCW
  dcwEnabled: z.unknown().optional(), // boolean, default false
  dcwMode: z.unknown().optional(), // 'low' | 'double' | 'high' | 'pix', default 'double'
  dcwLowScaler: z.unknown().optional(), // number, default 0.2
  dcwHighScaler: z.unknown().optional(), // number, default 0.2

  // Latent / denoise post-processing
  latentShift: z.unknown().optional(), // number, default 0.0
  latentRescale: z.unknown().optional(), // number, default 1.0
  customTimesteps: z.unknown().optional(), // string, default ''
  denoiseStrength: z.unknown().optional(), // number, default 0.0
  denoiseSmoothing: z.unknown().optional(), // number, default 0.7
  denoiseMix: z.unknown().optional(), // number, default 0.25
  lssStrength: z.unknown().optional(), // number, default 0.0
  lssVarThresh: z.unknown().optional(), // number, default 0.15
  lssDcRemove: z.unknown().optional(), // boolean, default true

  // Auto-trim
  autoTrimEnabled: z.unknown().optional(), // boolean, default false
  durationBuffer: z.unknown().optional(), // number, default 15
  autoTrimFadeMs: z.unknown().optional(), // number, default 2000

  // Post-processing master toggle and stages
  postProcessingEnabled: z.unknown().optional(), // boolean, default true
  spectralLifterEnabled: z.unknown().optional(), // boolean, default false
  slDenoiseStrength: z.unknown().optional(), // number, default 0.3
  slNoiseFloor: z.unknown().optional(), // number, default 0.1
  slHfMix: z.unknown().optional(), // number, default 0.0
  slTransientBoost: z.unknown().optional(), // number, default 0.0
  slShimmerReduction: z.unknown().optional(), // number, default 6.0
  masteringEnabled: z.unknown().optional(), // boolean, default false
  masteringReference: z.unknown().optional(), // string, default ''
  timbreReference: z.unknown().optional(), // boolean | string — string wire value is filled from timbreAudioPath
  timbreAudioPath: z.unknown().optional(), // string, default ''; folded into the `timbreReference` wire value, not sent as its own key
  vocalNaturalizerEnabled: z.unknown().optional(), // boolean, default false
  gainOffsetDb: z.unknown().optional(), // number, default 0
  naturalizeAmount: z.unknown().optional(), // number, default 0.5
  natVibratoRate: z.unknown().optional(), // number, default 4.5
  natVibratoDepth: z.unknown().optional(), // number, default 1.0
  natFormantStrength: z.unknown().optional(), // number, default 1.0
  natMetallicReduction: z.unknown().optional(), // number, default 1.0
  natQuantizationMask: z.unknown().optional(), // number, default 0.0
  natTransitionSmooth: z.unknown().optional(), // number, default 1.0
  ppVaeReencode: z.unknown().optional(), // boolean, default false
  ppVaeBlend: z.unknown().optional(), // number, default 0.0

  // StableStep (SA3) refinement
  stableStepOn: z.unknown().optional(), // boolean, default false
  stableStepStrength: z.unknown().optional(), // number, default 0.3
  stableStepBackend: z.unknown().optional(), // 'auto' | 'gguf', default 'auto'
  stableStepAdapters: z.unknown().optional(), // { name, scale, enabled? }[], default []
  stableStepPreserveDynamics: z.unknown().optional(), // boolean, default true (opt-out idiom: !== false)
  stableStepVocalPpVae: z.unknown().optional(), // boolean, default false
  stableStepVocalTrimDb: z.unknown().optional(), // number, default 0
  stableStepBlendMode: z.unknown().optional(), // 'off' | 'crossover' | 'mix', default 'off'
  stableStepCrossoverHz: z.unknown().optional(), // number, default 250
  stableStepCrossoverWidthHz: z.unknown().optional(), // number, default 200
  stableStepMix: z.unknown().optional(), // number, default 1.0
  stableStepSeed: z.unknown().optional(), // number, default 4242
  stableStepSeedFollowsDit: z.unknown().optional(), // boolean, default true
  stableStepSteps: z.unknown().optional(), // number, default 8
  stableStepSolver: z.unknown().optional(), // string, default ''
  stableStepScheduler: z.unknown().optional(), // string, default ''
  stableStepGuidanceMode: z.unknown().optional(), // string, default ''
  stableStepGuidanceScale: z.unknown().optional(), // number, default 1.0

  // Cover art / quality eval / whisper lyric recovery
  coverArtEnabled: z.unknown().optional(), // boolean, default false
  coverArtSubject: z.unknown().optional(), // string, default ''
  qualityEvalEnabled: z.unknown().optional(), // boolean, default false
  qualityEvalTarget: z.unknown().optional(), // string, default 'unmastered'
  whisperLyricsEnabled: z.unknown().optional(), // boolean, default false
  whisperModel: z.unknown().optional(), // string, default ''
  whisperLanguage: z.unknown().optional(), // string, default 'auto'
  whisperBeamSize: z.unknown().optional(), // number, default 5
  whisperIsolateVocals: z.unknown().optional(), // boolean, default false

  // YuE2 forced-alignment opt-in (sent unconditionally; ignored by backends that don't read it)
  yue2AlignLyrics: z.unknown().optional(), // boolean, default false

  // Generic Lua plugin params, keyed "<plugin>:<key>" — values are not
  // restricted to strings; a plugin's declared schema is what validates them
  pluginParams: z.unknown().optional(), // Record<string, unknown>, default {}
  postprocessPlugin: z.unknown().optional(), // string, default ''
  postprocessEnabled: z.unknown().optional(), // boolean, default false

  // Loudness normalization
  lufsEnabled: z.unknown().optional(), // boolean, default false
  lufsPreset: z.unknown().optional(), // string, default 'spotify'
  lufsTarget: z.unknown().optional(), // number, default -14
  lufsCeilingDb: z.unknown().optional(), // number, default -1

  // Backend-declared extension knobs (capabilities().extensions) — free-form
  // per backend; intent.ts validates these against the active backend's own
  // manifest, so this schema does not duplicate that per-key check.
  backendParams: z.unknown().optional(), // Record<string, unknown>, default {}
}).catchall(z.unknown());

export type GenerationControls = z.infer<typeof generationControlsSchema>;

/**
 * Non-throwing check against the documented control dictionary's key list.
 * Never gates a live request — `generationIntentSchema.params` (generation.ts)
 * remains the sole runtime validator so legacy and future fields keep working.
 */
export function parseGenerationControls(value: unknown) {
  return generationControlsSchema.safeParse(value);
}
