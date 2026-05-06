// llm/types.ts — Shared types for the LLM provider system

export interface ProviderInfo {
  id: string;
  name: string;
  available: boolean;
  models: string[];
  default_model: string;
}

export interface GenerationResponse {
  lyrics: string;
  provider: string;
  model: string;
  title: string;
  subject: string;
  bpm: number;
  key: string;
  caption: string;
  duration: number;
  system_prompt: string;
  user_prompt: string;
}

export type ChunkCallback = (chunk: string) => void;

export interface CallOptions {
  temperature?: number;
  top_p?: number;
  [key: string]: any;
}

// Global skip thinking signal
export let skipThinkingSignal = false;
export function setSkipThinking() {
  skipThinkingSignal = true;
  console.log('[LLM] Skip-thinking signal received');
}
export function resetSkipThinking() {
  skipThinkingSignal = false;
}
