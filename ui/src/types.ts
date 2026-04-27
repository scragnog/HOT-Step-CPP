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
  created_at?: string;
  createdAt?: Date | string;
  // UI state
  isGenerating?: boolean;
  // Mastered version
  masteredAudioUrl?: string;
  mastered_audio_url?: string;
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

  // DiT settings
  inferenceSteps: number;
  guidanceScale: number;
  shift: number;
  inferMethod: string;
  scheduler: string;
  guidanceMode: string;

  // Seed
  seed: number;
  randomSeed: boolean;

  // Batch
  batchSize: number;

  // Model selection
  ditModel: string;
  lmModel: string;
  vaeModel: string;

  // Adapter
  loraPath: string;
  loraScale: number;
  adapterGroupScales?: {
    self_attn: number;
    cross_attn: number;
    mlp: number;
    cond_embed: number;
  };
  adapterMode: string;  // "merge" or "runtime"

  // Trigger word (applied server-side to caption)
  triggerWord?: string;
  triggerPlacement?: 'prepend' | 'append' | 'replace';

  // Task type
  taskType: string;
  trackName?: string;

  // Cover/repaint
  sourceAudioUrl?: string;
  audioCoverStrength?: number;
  coverNoiseStrength?: number;
  repaintingStart?: number;
  repaintingEnd?: number;

  // Post-processing
  spectralLifterEnabled?: boolean;
  slDenoisePasses?: number;
  slDenoiseThreshold?: number;
  slHfMix?: number;
  slTransientBoost?: number;
  slShimmerReduction?: number;
  masteringEnabled?: boolean;
  masteringReference?: string;
  timbreReference?: boolean;  // Also use mastering ref as timbre conditioner

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

  // Duration buffer + auto-trim
  autoTrimEnabled?: boolean;    // Enable silence-detection trimming
  durationBuffer?: number;      // Extra seconds added to generation duration (default 15)
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
