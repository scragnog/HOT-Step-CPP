// generatePreflightApi.ts — POST /api/generate/preflight (server/src/contracts/generation.ts).
//
// Asks whether a Generate body would fit in VRAM now: fits, passes, refused or
// unknown, with the server's estimate. The UI shows the answer and never
// estimates VRAM itself. Same body as POST /api/generate; nothing is queued.
import type { GenerationPreflight } from '../../../server/src/contracts/generation';
import { apiClient, type ReadOptions } from './httpClient';

export type { GenerationPreflight, GenerationPreflightVerdict } from '../../../server/src/contracts/generation';

/** Silent: a failed check is no reason for an error toast. MM3 may read its
 *  tokenizer cold, so the timeout allows for that. */
export function preflightGeneration(body: unknown, opts: ReadOptions = {}): Promise<GenerationPreflight> {
  return apiClient.post<GenerationPreflight>('/generate/preflight', body, { signal: opts.signal, silent: true, timeout: 35_000 });
}
