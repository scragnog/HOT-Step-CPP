// types.ts — the skin-api's own types. Nothing here is imported from the app's internals, so
// a refactor behind the API cannot change what a skin sees (docs/dev/skins.md#versioning).

export type ThemeMode = 'system' | 'light' | 'dark';

/** A generation parameter id: a core control (`inferenceSteps`) or `backendParams.<key>`. */
export type ParamId = string;

export interface ParamOption { value: string; label: string }

/** What a control is: enough to draw it. `min`, `max` and `options` may be absent: draw a
 *  number or text field then, and fill select options from the catalogue hooks. */
export interface ParamMeta {
  id: ParamId;
  type: 'slider' | 'select' | 'toggle' | 'text' | 'array' | 'object';
  label: string;
  hint?: string;
  default: unknown;
  min?: number;
  max?: number;
  step?: number;
  options?: ParamOption[];
  /** False for content fields (caption, lyrics, bpm...): send them through generate(). */
  editable: boolean;
}

export interface ParamHistoryStep { label: string; at: number }

export interface SkinSong {
  id: string;
  title: string;
  caption: string;
  lyrics: string;
  audioUrl: string;
  coverUrl: string;
  /** Seconds, 0 when unknown. */
  duration: number;
  artistName: string;
  createdAt: string;
}

export type QueueState = 'waiting' | 'running' | 'done' | 'failed';

export interface SkinQueueItem {
  id: string;
  title: string;
  state: QueueState;
  /** What the engine is doing right now. */
  stage: string;
  /** 0 to 100; 0 when the engine has not reported any. */
  progress: number;
  error: string;
  /** The saved song, once there is one. */
  songId: string;
}

/** Content fields of a Create render. The global controls (models, steps, guidance...) are
 *  added by the host from the parameters; set those with params.set(). */
export interface GenerateRequest {
  caption: string;
  lyrics?: string;
  title?: string;
  instrumental?: boolean;
  /** Seconds. */
  duration?: number;
  bpm?: number;
  keyScale?: string;
  timeSignature?: string;
  vocalLanguage?: string;
}

export interface SkinBackend { id: string; name: string; active: boolean }

export interface SkinCapabilities {
  backend: string;
  /** False while the backend's server is not answering. */
  up: boolean;
  /** Longest render in seconds. */
  maxDuration: number;
  supportsBpm: boolean;
  supportsKeyScale: boolean;
  supportsNegativePrompt: boolean;
  maxBatch: number;
  /** Backend-declared extra controls, as parameter ids (`backendParams.<key>`). */
  extensionIds: ParamId[];
}

export interface SkinModelCatalogue {
  backend: string;
  /** Model files by bucket (`dit`, `lm`, `vae`...). Buckets differ per backend. */
  buckets: Readonly<Record<string, readonly string[]>>;
  adapters: readonly string[];
}

export interface SkinVram { usedMb: number; totalMb: number; freeMb: number }

export interface SkinLogLine { id: number; ts: number; text: string; source: 'engine' | 'server' }

/** Where openInStudio() can send you. Each reloads into the classic UI for now. */
export type StudioKind = 'library' | 'cover' | 'stem' | 'midi' | 'repaint';
