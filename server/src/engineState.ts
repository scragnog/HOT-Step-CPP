// engineState.ts — Shared engine readiness state
//
// Extracted to avoid circular imports between index.ts and route modules.
// index.ts sets these values; routes read them.

/** True only after runtime DLLs are downloaded and ace-server is spawned. */
export let engineReady = false;

/** Human-readable status for what the engine is doing before it's ready. */
export let engineBootStatus = 'Initializing...';

/** Update the engine boot status (called from index.ts bootstrap) */
export function setEngineReady(ready: boolean, status: string) {
  engineReady = ready;
  engineBootStatus = status;
}
