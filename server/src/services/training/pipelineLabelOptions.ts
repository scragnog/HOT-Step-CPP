// training/pipelineLabelOptions.ts — validate the batch pipeline's per-run
// label overrides. Field by field: anything absent or malformed is dropped, so
// the stored label defaults (and then the label route's own defaults) decide.

import { essentiaTempoProfile } from './essentiaClient.js';
import type { PipelineLabelOptions } from './types.js';

const POLICIES = ['fill_missing', 'overwrite_caption', 'overwrite_lyrics', 'overwrite_all'];

/** Throws only on out-of-range tempo bounds, which the label route would refuse
 *  per dataset anyway; better to refuse the whole batch up front. */
export function parsePipelineLabelOptions(raw: unknown): PipelineLabelOptions | undefined {
  if (!raw || typeof raw !== 'object') return undefined;
  const lo = raw as Record<string, unknown>;
  const out: PipelineLabelOptions = {};
  for (const k of ['useEssentia', 'useGenius', 'useCaption', 'localLyrics'] as const) {
    if (typeof lo[k] === 'boolean') out[k] = lo[k] as boolean;
  }
  if (lo.scope === 'all' || lo.scope === 'unlabeled') out.scope = lo.scope;
  if (lo.tempoSource === 'beat_this' || lo.tempoSource === 'essentia') out.tempoSource = lo.tempoSource;
  if (lo.keySource === 'sheetsage' || lo.keySource === 'essentia') out.keySource = lo.keySource;
  if (typeof lo.mergePolicy === 'string' && POLICIES.includes(lo.mergePolicy)) {
    out.mergePolicy = lo.mergePolicy as PipelineLabelOptions['mergePolicy'];
  }
  if (lo.essentiaTempo && typeof lo.essentiaTempo === 'object') {
    const t = lo.essentiaTempo as Record<string, unknown>;
    const bounds: NonNullable<PipelineLabelOptions['essentiaTempo']> = {};
    if (typeof t.minTempo === 'number') bounds.minTempo = t.minTempo;
    if (typeof t.maxTempo === 'number') bounds.maxTempo = t.maxTempo;
    if (essentiaTempoProfile(bounds)) out.essentiaTempo = bounds;
  }
  if (lo.caption && typeof lo.caption === 'object') {
    const c = lo.caption as Record<string, unknown>;
    const caption: NonNullable<PipelineLabelOptions['caption']> = {};
    if (typeof c.provider === 'string' && c.provider) caption.provider = c.provider;
    if (typeof c.model === 'string' && c.model) caption.model = c.model;
    if (Object.keys(caption).length > 0) out.caption = caption;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}
