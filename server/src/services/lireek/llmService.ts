// llmService.ts — Barrel re-export from the modular llm/ directory
//
// This file preserves the original import path (./llmService) for existing
// consumers. All implementation has been decomposed into llm/*.ts modules.
//
// New code should import from './llm/index.js' directly.

export {
  type ProviderInfo,
  type GenerationResponse,
  type ChunkCallback,
  type CallOptions,
  skipThinkingSignal,
  setSkipThinking,
  resetSkipThinking,
  LLMProvider,
  getProvider,
  listProviders,
  stripThinkingBlocks,
  postprocessLyrics,
  fixSectionLabels,
  enforceLineCounts,
  fixAPrefix,
  stripLyricQuotes,
  estimateDuration,
  selectBestBlueprint,
  generateLyricsStreaming,
  refineLyricsStreaming,
} from './llm/index.js';
