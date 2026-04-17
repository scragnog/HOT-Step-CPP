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
  audio_url: string;
  cover_url: string;
  duration: number;
  bpm: number;
  key_scale: string;
  time_signature: string;
  tags: string[];
  is_public: boolean;
  dit_model: string;
  generation_params: Record<string, unknown>;
  created_at: string;
}

/** Parameters sent to the generation API */
export interface GenerationParams {
  // Content
  caption: string;
  lyrics: string;
  instrumental: boolean;
  title?: string;

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

  // Seed
  seed: number;
  randomSeed: boolean;

  // Batch
  batchSize: number;

  // Model selection
  ditModel: string;
  lmModel: string;

  // Adapter
  loraPath: string;
  loraScale: number;

  // Task type
  taskType: string;
  trackName?: string;

  // Cover/repaint
  sourceAudioUrl?: string;
  audioCoverStrength?: number;
  coverNoiseStrength?: number;
  repaintingStart?: number;
  repaintingEnd?: number;
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
