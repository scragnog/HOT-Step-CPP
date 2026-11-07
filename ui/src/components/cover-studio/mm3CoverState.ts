// mm3CoverState.ts — Mm3CoverPanel's settings and their mapping onto the
// cover-render workflow's `mm3Cover` field (contracts/studioWorkflows.ts).

export interface Mm3CoverSettings {
  plank: string;
  semanticOnly: boolean;
  /** Blank = hard pin. */
  forceBias: string;
  patternMode: 'all' | 'prefix' | 'anchor' | 'ranges';
  /** Seconds, frames or "0-20, 45-60", by mode. */
  patternValue: string;
  invert: boolean;
  /** Blank = the whole plank. */
  trimSeconds: string;
  seed: number;
  arSeed: number;
}

export const DEFAULT_MM3_COVER: Mm3CoverSettings = {
  plank: '', semanticOnly: true, forceBias: '', patternMode: 'all', patternValue: '', invert: false,
  trimSeconds: '', seed: -1, arSeed: -1,
};

const positive = (text: string): number | null => {
  const n = Number(text.trim());
  return text.trim() !== '' && Number.isFinite(n) && n > 0 ? n : null;
};

/** "0-20, 45-60" -> [[0, 20], [45, 60]]; malformed parts are dropped. */
export function parseRanges(text: string): Array<[number, number]> {
  return text.split(',').map(part => part.split('-').map(s => Number(s.trim())))
    .filter(p => p.length === 2 && p.every(Number.isFinite) && p[0] >= 0 && p[1] > p[0])
    .map(p => [p[0], p[1]] as [number, number]);
}

/** The cover-render workflow's `mm3Cover` field. */
export function mm3CoverInput(s: Mm3CoverSettings) {
  const value = s.patternMode === 'ranges' ? null : positive(s.patternValue);
  const ranges = s.patternMode === 'ranges' ? parseRanges(s.patternValue) : [];
  const pattern = s.patternMode === 'prefix' && value ? { prefixSeconds: value }
    : s.patternMode === 'anchor' && value ? { anchorEvery: Math.round(value) }
      : ranges.length ? { pinnedRanges: ranges } : null;
  const bias = s.forceBias.trim() === '' ? NaN : Number(s.forceBias);
  return {
    plank: s.plank, semanticOnly: s.semanticOnly,
    forceBias: Number.isFinite(bias) && bias >= 0 ? Math.min(bias, 1000) : null,
    pattern: pattern ? { ...pattern, ...(s.invert ? { invert: true } : {}) } : null,
    trimSeconds: positive(s.trimSeconds), arSeed: s.arSeed,
  };
}
