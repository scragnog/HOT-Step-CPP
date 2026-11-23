// generation/preflight.ts — an engine admission dry run as the preflight contract

import type { GenerationPreflight, GenerationPreflightVerdict } from '../../contracts/generation.js';

const VERDICTS: readonly GenerationPreflightVerdict[] = ['fits', 'passes', 'refused', 'unknown'];
const gib = (bytes: number) => Math.round(bytes / 2 ** 30 * 100) / 100;

/** Map a /heartmula/preflight or /mm3/preflight answer. Both report the need at the
 *  request's largest stage and what it can use without unloading another backend. */
export function preflightFromEngine(r: { verdict?: unknown; need_bytes?: unknown; free_bytes?: unknown; detail?: unknown }): GenerationPreflight {
  const verdict = VERDICTS.find(v => v === r.verdict) ?? 'unknown';
  const out: GenerationPreflight = { verdict };
  if (verdict !== 'unknown' && typeof r.need_bytes === 'number' && typeof r.free_bytes === 'number') {
    out.estimateGb = gib(r.need_bytes);
    out.freeGb = gib(r.free_bytes);
  }
  if (typeof r.detail === 'string' && r.detail) out.detail = r.detail;
  return out;
}
