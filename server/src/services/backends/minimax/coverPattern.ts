// minimax/coverPattern.ts — Cover Studio levers on a plank's codes.
//
// Pure functions over the engine's forced_* arrays (see plank.ts for the
// layout). Row 0 is the un-emitted warm-up iteration; row r >= 1 is audio
// frame r-1, and MM3 emits 25 frames a second.
//
// A free frame is semantic code -1: the engine samples it normally (EOS stays
// masked inside the forced span). Free frames and lm_force_bias are
// semantic-only features; the engine refuses them alongside forced_acoustic.

export const MM3_ROWS_PER_SEC = 25;

export interface Mm3ForcePattern {
  /** Pin the first N seconds. */
  prefixSeconds?: number;
  /** Pin every Nth frame. */
  anchorEvery?: number;
  /** Pin [start, end) second ranges. */
  pinnedRanges?: Array<[number, number]>;
  /** Swap pinned and free (free inside the rules, pinned outside). */
  invert?: boolean;
}

/** Keep the first `seconds` of codes (the warm-up row plus seconds * 25 frames). */
export function trimMm3Codes(sem: number[], ac: number[], seconds: number): { sem: number[]; ac: number[] } {
  const rows = Math.max(2, 1 + Math.round(seconds * MM3_ROWS_PER_SEC));
  if (rows >= sem.length) return { sem, ac };
  return { sem: sem.slice(0, rows), ac: ac.slice(0, rows * 7) };
}

/** Accept a pattern from the request body, keeping only well-typed fields.
 *  Null when no rule is set (pin everything, as today). */
export function parseMm3ForcePattern(raw: unknown): Mm3ForcePattern | null {
  if (!raw || typeof raw !== 'object') return null;
  const r = raw as Record<string, unknown>;
  const pos = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : undefined);
  const out: Mm3ForcePattern = {};
  const prefix = pos(r.prefixSeconds);
  if (prefix) out.prefixSeconds = prefix;
  const every = pos(r.anchorEvery);
  if (every) out.anchorEvery = Math.round(every);
  if (Array.isArray(r.pinnedRanges)) {
    const ranges = r.pinnedRanges.filter((x): x is [number, number] =>
      Array.isArray(x) && x.length === 2 && x.every(n => typeof n === 'number' && Number.isFinite(n) && n >= 0)
      && x[1] > x[0]);
    if (ranges.length) out.pinnedRanges = ranges;
  }
  if (!out.prefixSeconds && !out.anchorEvery && !out.pinnedRanges) return null;
  if (r.invert === true) out.invert = true;
  return out;
}

/** Write -1 into every free frame. A frame is pinned when any rule pins it;
 *  invert swaps that. Row 0 always keeps its code. */
export function applyMm3ForcePattern(sem: number[], p: Mm3ForcePattern): number[] {
  const R = MM3_ROWS_PER_SEC;
  return sem.map((code, row) => {
    if (row === 0) return code;
    const f = row - 1;
    const pinned = (p.prefixSeconds !== undefined && f < p.prefixSeconds * R)
      || (p.anchorEvery !== undefined && f % p.anchorEvery === 0)
      || (p.pinnedRanges ?? []).some(([s, e]) => f >= s * R && f < e * R);
    return pinned !== (p.invert === true) ? code : -1;
  });
}
