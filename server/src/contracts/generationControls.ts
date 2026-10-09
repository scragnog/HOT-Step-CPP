import { z } from 'zod/v4';

/**
 * Create's full control dictionary: every field Node defaults or translates
 * for a Create (text2music) generation request, across the ace/yue2/
 * minimax-m3 backends. See docs/dev/frontend-create-controls.md for the
 * per-field default, backend applicability and omitted-vs-explicit table —
 * this file is the typed shape, that page is the narrative.
 *
 * Permissive by design: every field is optional and unknown keys pass
 * through via `.catchall`. That makes this schema accept exactly what the
 * existing `params: z.record(z.string(), z.unknown())` field on
 * `generationIntentSchema` (generation.ts) already accepts — it documents
 * the dictionary without narrowing anything live. Arbitrary backend
 * extension knobs (`backendParams`) and Lua plugin knobs (`pluginParams`)
 * stay free-form maps for the same reason: new ones must not require a
 * schema change here.
 */
const adapterStackEntrySchema = z.object({
  path: z.string(),
  scale: z.number().optional(),
  stepStart: z.number().optional(),
  stepEnd: z.number().optional(),
  stepSoft: z.number().optional(),
  gainCurve: z.array(z.number()).optional(),
  gainDomain: z.enum(['steps', 't']).optional(),
}).catchall(z.unknown());

const stableStepAdapterEntrySchema = z.object({
  name: z.string(),
  scale: z.number(),
  enabled: z.boolean().optional(),
}).catchall(z.unknown());

export const generationControlsSchema = z.object({
  // Model routing
  ditModel: z.string().optional(),
  lmModel: z.string().optional(),
  vaeModel: z.string().optional(),
  embeddingModel: z.string().optional(),

  // Adapter (DiT LoRA) — single-slot Simple mode and multi-entry Advanced stack
  adapter: z.string().optional(),
  adapterScale: z.number().optional(),
  adapterStack: z.array(adapterStackEntrySchema).optional(),
  adapterStackMode: z.enum(['blend', 'sum']).optional(),
  adapterStackBudget: z.number().optional(),
  adapterSectionAlignAt: z.number().optional(),
  adapterSectionIsolation: z.number().optional(),
  adapterMode: z.enum(['merge', 'runtime', 'runtime_lowrank']).optional(),
  adapterRuntimeQuant: z.string().optional(),
  adapterMergeLowVram: z.boolean().optional(),
  adapterGroupScales: z.object({
    self_attn: z.number().optional(),
    cross_attn: z.number().optional(),
    mlp: z.number().optional(),
    cond_embed: z.number().optional(),
    time_embed: z.number().optional(),
    proj_in: z.number().optional(),
  }).catchall(z.number()).optional(),
  rebaseSource: z.string().optional(),
  rebaseBeta: z.number().optional(),
  adapterFolder: z.string().optional(),
  lmAdapterFolder: z.string().optional(),
  advancedAdapters: z.boolean().optional(),
  noAdapterRender: z.boolean().optional(),
  adaptersOpen: z.boolean().optional(),

  // Planner-LM runtime LoRA (independent of the DiT adapter stack above)
  lmAdapter: z.string().optional(),
  lmAdapterScale: z.number().optional(),

  // Core DiT sampling
  inferenceSteps: z.number().optional(),
  guidanceScale: z.number().optional(),
  cfgCutoffRatio: z.number().optional(),
  lmCfgCutoffRatio: z.number().optional(),
  cacheRatio: z.number().optional(),
  shift: z.number().optional(),
  ditSlidingWindow: z.number().optional(),
  inferMethod: z.string().optional(),
  scheduler: z.string().optional(),
  guidanceMode: z.string().optional(),
  storkSubsteps: z.number().optional(),
  beatStability: z.number().optional(),
  frequencyDamping: z.number().optional(),
  temporalSmoothing: z.number().optional(),
  apgMomentum: z.number().optional(),
  apgNormThreshold: z.number().optional(),

  // Seeding / batching
  seed: z.number().optional(),
  randomSeed: z.boolean().optional(),
  lmSeed: z.number().optional(),
  lmSeedFollowsDit: z.boolean().optional(),
  batchSize: z.number().optional(),

  // Metadata / song-level fields (legacy flat-body names; see translateParams.ts)
  bpm: z.union([z.number(), z.string()]).optional(),
  keyScale: z.string().optional(),
  timeSignature: z.string().optional(),
  vocalLanguage: z.string().optional(),

  // Task mode and cover/repaint axis (ACE-only; YuE2/MM3 accept text2music only)
  taskType: z.string().optional(),
  audioCoverStrength: z.number().optional(),
  coverNoiseStrength: z.number().optional(),
  coverNoiseMethod: z.string().optional(),
  repaintingStart: z.number().optional(),
  repaintingEnd: z.number().optional(),
  seedStrength: z.number().optional(),
  evictLm: z.boolean().optional(),
  vaeChunk: z.union([z.number(), z.boolean()]).optional(),
  batchCfg: z.boolean().optional(),
  trackName: z.string().optional(),

  // LM / planner sampling
  skipLm: z.boolean().optional(),
  skipLrc: z.boolean().optional(),
  useCotCaption: z.boolean().optional(),
  lmTemperature: z.number().optional(),
  lmCfgScale: z.number().optional(),
  lmTopK: z.number().optional(),
  lmTopP: z.number().optional(),
  lmRepPenalty: z.number().optional(),
  lmRepWindow: z.number().optional(),
  lmRepMode: z.enum(['presence', 'dry']).optional(),
  lmDryBase: z.number().optional(),
  lmDryMinLen: z.number().optional(),
  lmNegativePrompt: z.string().optional(),
  negativePrompt: z.string().optional(),
  lmCodesStrength: z.number().optional(),
  lmCodesMode: z.enum(['ratio', 'steps']).optional(),
  lmCodesSteps: z.number().optional(),

  // DCW
  dcwEnabled: z.boolean().optional(),
  dcwMode: z.enum(['low', 'double', 'high', 'pix']).optional(),
  dcwLowScaler: z.number().optional(),
  dcwHighScaler: z.number().optional(),

  // Latent / denoise post-processing
  latentShift: z.number().optional(),
  latentRescale: z.number().optional(),
  customTimesteps: z.string().optional(),
  denoiseStrength: z.number().optional(),
  denoiseSmoothing: z.number().optional(),
  denoiseMix: z.number().optional(),
  lssStrength: z.number().optional(),
  lssVarThresh: z.number().optional(),
  lssDcRemove: z.boolean().optional(),

  // Auto-trim
  autoTrimEnabled: z.boolean().optional(),
  durationBuffer: z.number().optional(),
  autoTrimFadeMs: z.number().optional(),

  // Post-processing master toggle and stages
  postProcessingEnabled: z.boolean().optional(),
  spectralLifterEnabled: z.boolean().optional(),
  slDenoiseStrength: z.number().optional(),
  slNoiseFloor: z.number().optional(),
  slHfMix: z.number().optional(),
  slTransientBoost: z.number().optional(),
  slShimmerReduction: z.number().optional(),
  masteringEnabled: z.boolean().optional(),
  masteringReference: z.string().optional(),
  timbreReference: z.union([z.boolean(), z.string()]).optional(),
  timbreAudioPath: z.string().optional(),
  vocalNaturalizerEnabled: z.boolean().optional(),
  gainOffsetDb: z.number().optional(),
  naturalizeAmount: z.number().optional(),
  natVibratoRate: z.number().optional(),
  natVibratoDepth: z.number().optional(),
  natFormantStrength: z.number().optional(),
  natMetallicReduction: z.number().optional(),
  natQuantizationMask: z.number().optional(),
  natTransitionSmooth: z.number().optional(),
  ppVaeReencode: z.boolean().optional(),
  ppVaeBlend: z.number().optional(),

  // StableStep (SA3) refinement
  stableStepOn: z.boolean().optional(),
  stableStepStrength: z.number().optional(),
  stableStepBackend: z.enum(['auto', 'gguf']).optional(),
  stableStepAdapters: z.array(stableStepAdapterEntrySchema).optional(),
  stableStepPreserveDynamics: z.boolean().optional(),
  stableStepVocalPpVae: z.boolean().optional(),
  stableStepVocalTrimDb: z.number().optional(),
  stableStepBlendMode: z.enum(['off', 'crossover', 'mix']).optional(),
  stableStepCrossoverHz: z.number().optional(),
  stableStepCrossoverWidthHz: z.number().optional(),
  stableStepMix: z.number().optional(),
  stableStepSeed: z.number().optional(),
  stableStepSeedFollowsDit: z.boolean().optional(),
  stableStepSteps: z.number().optional(),
  stableStepSolver: z.string().optional(),
  stableStepScheduler: z.string().optional(),
  stableStepGuidanceMode: z.string().optional(),
  stableStepGuidanceScale: z.number().optional(),

  // Cover art / quality eval / whisper lyric recovery
  coverArtEnabled: z.boolean().optional(),
  coverArtSubject: z.string().optional(),
  qualityEvalEnabled: z.boolean().optional(),
  qualityEvalTarget: z.string().optional(),
  whisperLyricsEnabled: z.boolean().optional(),
  whisperModel: z.string().optional(),
  whisperLanguage: z.string().optional(),
  whisperBeamSize: z.number().optional(),
  whisperIsolateVocals: z.boolean().optional(),

  // YuE2 forced-alignment opt-in (sent unconditionally; ignored by backends that don't read it)
  yue2AlignLyrics: z.boolean().optional(),

  // Generic Lua plugin params, keyed "<plugin>:<key>"
  pluginParams: z.record(z.string(), z.string()).optional(),
  postprocessPlugin: z.string().optional(),
  postprocessEnabled: z.boolean().optional(),

  // Loudness normalization
  lufsEnabled: z.boolean().optional(),
  lufsPreset: z.string().optional(),
  lufsTarget: z.number().optional(),
  lufsCeilingDb: z.number().optional(),

  // Backend-declared extension knobs (capabilities().extensions) — free-form
  // per backend; intent.ts validates these against the active backend's own
  // manifest, so this schema does not duplicate that per-key check.
  backendParams: z.record(z.string(), z.unknown()).optional(),
}).catchall(z.unknown());

export type GenerationControls = z.infer<typeof generationControlsSchema>;

/**
 * Non-throwing check against the documented control dictionary shape.
 * Never gates a live request — `generationIntentSchema.params` (generation.ts)
 * remains the sole runtime validator so legacy and future fields keep working.
 */
export function parseGenerationControls(value: unknown) {
  return generationControlsSchema.safeParse(value);
}
