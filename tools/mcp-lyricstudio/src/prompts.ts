// prompts.ts — re-export of the CANONICAL prompt module.
//
// The single source of truth for all Lyric Studio prompts (system prompts,
// prompt builders, blueprint helpers, slop lists) is:
//   server/src/services/lireek/prompts.ts
// Do NOT define or fork prompts here — edit them there. This works because the
// MCP server always runs from TypeScript source via tsx (see ../.mcp.json),
// which resolves cross-package .ts imports; there is no separate build step.

export * from '../../../server/src/services/lireek/prompts.js';
