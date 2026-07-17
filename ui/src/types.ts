// types.ts — All shared TypeScript types for HOT-Step CPP
//
// Keep types granular and composable. Import only what you need.

/** Song stored in the library */
export interface Song {
  id: string;
  title: string;
  lyrics: string;
  style: string;
  caption: string;
  // Audio URL — components use audioUrl, DB uses audio_url
  audioUrl: string;
  audio_url?: string;
  // Cover URL
  coverUrl?: string;
  cover_url?: string;
  // Duration — can be number (seconds) or formatted string
  duration: string | number;
  bpm?: number;
  key_scale?: string;
  time_signature?: string;
  tags: string[];
  artistName?: string;
  is_public?: boolean;
  isPublic?: boolean;
  dit_model?: string;
  // Generation params — snake_case from DB, camelCase for components
  generation_params?: Record<string, unknown>;
  generationParams?: Record<string, any>;
  // User-edited embed-tag overrides (JSON string: { artist, album, year, comment }) (#60)
  metadata_overrides?: string;
  created_at?: string;
  createdAt?: Date | string;
  // UI state
  isGenerating?: boolean;
  // Mastered version
  masteredAudioUrl?: string;
  mastered_audio_url?: string;
  kickStemUrl?: string;
  kick_stem_url?: string;
  snareStemUrl?: string;
  snare_stem_url?: string;
  hihatStemUrl?: string;
  hihat_stem_url?: string;
  discoDataUrl?: string;
  disco_data_url?: string;
  // Latent file
  latentUrl?: string;
  latent_url?: string;
  // Quality scores (JSON blob from DB)
  quality_scores?: string;
  // Cover art subject — custom prompt stored for "Regenerate Cover"
  cover_art_subject?: string;
}

/** Normalized recent song returned by /api/songs/recent — unified across all modes */
export interface UnifiedRecentSong {
  id: string;
  title: string;
  audio_url: string;
  mastered_audio_url: string;
  latent_url: string;
  cover_url: string;
  duration: number;
  lyrics: string;
  caption: string;
  style: string;
  bpm: number;
  key_scale: string;
  source: string;        // 'create' | 'lyric-studio' | 'cover-studio'
  created_at: string;
  artist_name: string;
  artist_image: string;
  album: string;
  generation_id: number | null;
}

/** Parameters sent to the generation API */
export interface GenerationParams {
  // Content
  caption: string;
  lyrics: string;
  instrumental: boolean;
  title?: string;
  artist?: string;
  subject?: string;

  // Metadata
  bpm: number;
  duration: number;
  keyScale: string;
  timeSignature: string;
  vocalLanguage: string;

  // LM settings
  skipLm: boolean;
  lmTemperature: number;
  lmCfgScale: number;
  lmTopK: number;
  lmTopP: number;
  lmNegativePrompt: string;
  useCotCaption: boolean;
  skipLrc?: boolean;  // Skip LRC (timed-lyrics) generation

  // DiT settings
  inferenceSteps: number;
  guidanceScale: number;
  cfgCutoffRatio?: number;
  lmCfgCutoffRatio?: number;
  cacheRatio?: number;
  shift: number;
  inferMethod: string;
  scheduler: string;
  guidanceMode: string;

  // Seed (DiT / generation phase)
  seed: number;
  randomSeed: boolean;

  // LM Seed — independent from the seed above. Controls the LM phase's
  // caption/lyrics/audio-code sampling. When lmSeedFollowsDit is true
  // (default), lmSeed is ignored and the LM seed is tied to the DiT seed —
  // the original engine behavior: locked seed -> both deterministic,
  // random -> both random. Set lmSeedFollowsDit to false to use lmSeed as
  // an independent fixed value instead.
  lmSeed?: number;
  lmSeedFollowsDit?: boolean;

  // Batch
  batchSize: number;

  // Model selection
  ditModel: string;
  lmModel: string;
  vaeModel: string;
  embeddingModel?: string;
  useOrtVae?: boolean;  // Use ONNX Runtime VAE instead of GGML

  // Adapter
  loraPath: string;
  loraScale: number;
  /** Multi-adapter stack: applied together, each with its own scale. When set
   *  (and non-empty) it supersedes the single loraPath/loraScale. */
  loraStack?: { path: string; scale: number }[];
  /** Stack scaling mode ('sum' | 'blend') and combined-strength budget — reused
   *  for per-section masking weight transforms. */
  adapterStackMode?: string;
  adapterStackBudget?: number;
  /** Per-section masking: alignment-timing fraction + (dormant) isolation flag. */
  adapterSectionAlignAt?: number;
  adapterSectionIsolation?: boolean;
  /** Runtime-mode delta quantization ('bf16' | 'q8_0' | 'q4_0' | aliased 'q4_k'). */
  adapterRuntimeQuant?: string;
  /** Merge-mode low-VRAM storage: re-encode merged weights to the base's native
   *  quant instead of F32 promotion. */
  adapterMergeLowVram?: boolean;
  /** Basin re-base: source DiT model name + nudge strength (merge mode only). */
  rebaseSource?: string;
  rebaseBeta?: number;
  adapterGroupScales?: {
    self_attn: number;
    cross_attn: number;
    mlp: number;
    cond_embed: number;
    time_embed: number;
    proj_in: number;
  };
  adapterMode: string;  // "merge" or "runtime"

  // Trigger word (applied server-side to caption). triggerWords carries every
  // loaded adapter's trigger; triggerWord is their joined form (back-compat).
  triggerWord?: string;
  triggerWords?: string[];
  triggerPlacement?: 'prepend' | 'append' | 'replace';

  // Task type
  taskType: string;
  trackName?: string;

  // Cover/repaint
  sourceAudioUrl?: string;
  audioCoverStrength?: number;
  coverNoiseStrength?: number;
  coverNoiseMethod?: string;
  repaintingStart?: number;
  repaintingEnd?: number;
  sourceLatentUrl?: string;  // HSLAT latent file URL (skips VAE encode)

  // Post-processing master toggle
  postProcessingEnabled?: boolean;

  // Post-processing — Spectral Lifter (native C++ in engine)
  spectralLifterEnabled?: boolean;
  slDenoiseStrength?: number;    // 0.0–1.0, gate aggressiveness
  slNoiseFloor?: number;         // 0.01–0.5, residual leakage
  slHfMix?: number;
  slTransientBoost?: number;
  slShimmerReduction?: number;
  masteringEnabled?: boolean;
  masteringReference?: string;
  timbreReference?: boolean | string;  // true = reuse mastering ref, string = dedicated timbre audio path

  // Pre-VST gain offset (dB adjustment to unmastered track before VST chain)
  gainOffsetDb?: number;           // -10 to +10, 0 = no change

  // Solver sub-parameters (conditional on selected solver)
  storkSubsteps?: number;
  beatStability?: number;
  frequencyDamping?: number;
  temporalSmoothing?: number;

  // Guidance sub-parameters (conditional on guidance mode)
  apgMomentum?: number;
  apgNormThreshold?: number;

  // Latent post-processing (applied after DiT, before VAE decode)
  latentShift?: number;         // 0.0 = no bias
  latentRescale?: number;       // 1.0 = no scaling
  customTimesteps?: string;     // CSV of descending floats, overrides scheduler

  // Post-VAE spectral denoiser
  denoiseStrength?: number;     // 0.0 = off, 1.0 = max suppression
  denoiseSmoothing?: number;    // 0.0 = sharp gate, 1.0 = very smooth
  denoiseMix?: number;          // 0.0 = all dry, 1.0 = all denoised

  // LSS: Latent Spectral Suppressor (pre-VAE latent channel gate, MDMAchine)
  lssStrength?: number;         // 0.0 = off; attenuation floor is 1-strength
  lssVarThresh?: number;        // relative variance threshold (default 0.15)
  lssDcRemove?: boolean;        // per-channel DC removal while LSS active

  // Duration buffer + auto-trim
  autoTrimEnabled?: boolean;    // Enable silence-detection trimming
  durationBuffer?: number;      // Extra seconds added to generation duration (default 15)
  autoTrimFadeMs?: number;      // Fade-out length in ms (default 2000 for forced, 500 for gap-detected)

  // Vocal Naturalizer (5-stage DSP humanisation, applied to full mix)
  vocalNaturalizerEnabled?: boolean;
  naturalizeAmount?: number;        // 0.0–1.0 master intensity
  natVibratoRate?: number;          // 3.0–7.0 Hz
  natVibratoDepth?: number;         // 0.0–1.0
  natFormantStrength?: number;      // 0.0–1.0
  natMetallicReduction?: number;    // 0.0–1.0
  natQuantizationMask?: number;     // 0.0–1.0
  natTransitionSmooth?: number;     // 0.0–1.0

  // PP-VAE re-encode (spectral cleanup via post-processing VAE)
  ppVaeReencode?: boolean;
  ppVaeBlend?: number;         // 0.0 = fully PP-VAE, 1.0 = fully original
  ppVaeUseOnnx?: boolean;      // true = prefer ONNX/TRT, false = force GGUF

  // DCW (Dynamic CFG Weighting)
  dcwEnabled?: boolean;
  dcwMode?: string;            // 'single' | 'double'
  dcwScaler?: number;
  dcwHighScaler?: number;

  // Cover/style caption (used by Cover Studio)
  style?: string;

  // Source tracking — which UI mode originated the generation
  source?: string;  // 'create' | 'lyric-studio' | 'cover-studio'

  // Lua plugin dynamic params
  pluginParams?: Record<string, string | number | boolean>;

  // Cover Art auto-generation
  coverArtEnabled?: boolean;
  coverArtSubject?: string;     // Override the auto-generated image subject

  // Audio Quality Evaluator
  qualityEvalEnabled?: boolean;
  qualityEvalTarget?: 'unmastered' | 'mastered' | 'both';

  // LUFS Normalization (final mastering stage)
  lufsEnabled?: boolean;
  lufsTarget?: number;          // target integrated LUFS (e.g. -14)

  // Postprocess plugin (replaces built-in VAE tiled decoder)
  postprocessPlugin?: string;

  // Whisper Lyrics Transcription
  whisperLyricsEnabled?: boolean;
  whisperModel?: string;
  whisperLanguage?: string;
  whisperBeamSize?: number;
  whisperIsolateVocals?: boolean;
}

/** Generation job status from the server */
export interface GenerationJob {
  jobId: string;
  status: 'pending' | 'lm_running' | 'synth_running' | 'saving' | 'succeeded' | 'failed' | 'cancelled';
  stage?: string;
  progress?: number;
  result?: {
    audioUrls: string[];
    songIds: string[];
    bpm?: number;
    duration?: number;
    keyScale?: string;
    timeSignature?: string;
    masteredAudioUrl?: string;
  };
  error?: string;
}

/** User profile */
export interface User {
  id: string;
  username: string;
  bio: string;
  avatar_url: string;
  banner_url: string;
  created_at: string;
}

/** Auth state from auto-login */
export interface AuthState {
  user: User | null;
  token: string | null;
}

/** Available models from ace-server /props */
export interface AceModels {
  models: {
    lm: string[];
    embedding: string[];
    dit: string[];
    vae: string[];
  };
  adapters: string[];
  config: {
    max_batch: number;
    mp3_bitrate: number;
  };
  defaults: Record<string, unknown>;
}

/** File or directory entry from the adapter browser */
export interface BrowseEntry {
  name: string;
  path: string;
  type: 'dir' | 'file';
  size?: number;
}

/** Adapter file from folder scan */
export interface AdapterFile {
  name: string;
  path: string;
  size: number;
}

/** Model registry file entry */
export interface RegistryFile {
  id: string;
  filename: string;
  role: 'dit' | 'lm' | 'embedding' | 'vae' | 'pp-vae' | 'supersep' | 'whisper';
  subdir?: string;
  displayName: string;
  scale?: 'standard' | 'xl' | null;
  variant?: string | null;
  quant: string;
  sizeBytes: number;
  repo: string;
  description: string;
  tags: string[];
  installed: boolean;
}

/** Starter pack definition */
export interface StarterPack {
  id: string;
  name: string;
  description: string;
  fileIds: string[];
}

/** Model registry response from server */
export interface ModelRegistry {
  packs: StarterPack[];
  files: RegistryFile[];
  modelsDir: string;
}

/** Download job status */
export interface DownloadJob {
  jobId: string;
  fileId: string;
  filename: string;
  status: 'queued' | 'downloading' | 'paused' | 'completed' | 'failed' | 'cancelled';
  bytesDownloaded: number;
  totalBytes: number;
  speed: number;
  error?: string;
}
