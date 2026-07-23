// llm/index.ts — Barrel re-export for the LLM provider module
//
// All external consumers should import from this module.
// Internal files within llm/ import from their specific submodules directly.

// Types
export type { ProviderInfo, GenerationResponse, ChunkCallback, CallOptions } from './types.js';
export { skipThinkingSignal, setSkipThinking, resetSkipThinking } from './types.js';

// Base class (for type use / extension)
export { LLMProvider } from './base.js';

// Registry
export { getProvider, listProviders } from './registry.js';

// Post-processing
export { stripThinkingBlocks, postprocessLyrics, fixSectionLabels, enforceLineCounts, fixAPrefix, stripLyricQuotes, estimateDuration } from './postprocess.js';

// Orchestration (high-level generation functions)
export { generateLyricsStreaming, refineLyricsStreaming } from './orchestration.js';
