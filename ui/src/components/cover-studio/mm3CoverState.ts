// Settings for the laundered MM3 cover panel.

/** Cover modes (round 9 of the cover probe). Pin holds the plank's codes; the
 *  others let the planner pick from the encoder's candidates (the plank's guide). */
export type Mm3CoverMode = 'pin' | 'follow' | 'follow-depth' | 'follow-stream' | 'product' | 'blend';
export type Mm3StateFix = 'off' | 'depth' | 'stream' | 'both';

export interface Mm3CoverSettings {
  plank: string;
  launderSteps: number;
  launderSeed: number;
  launderCaption: string;
  /** Blank uses the whole plank. */
  trimSeconds: string;
  seed: number;
  arSeed: number;
  match: number;
  preparedSourceUrl: string;
  mode: Mm3CoverMode;
  /** Follow source: how often the source wins, 0..1. */
  strength: number;
  /** Encoder candidates per frame, 1..32. */
  candidates: number;
  stateFix: Mm3StateFix;
  /** Product rule weight on the encoder, 0..4. */
  weight: number;
  /** State blend toward the launder, percent. */
  blend: number;
}

/** Each mode's own sliders and their defaults. Changing mode applies these and
 *  leaves the source prep, seeds, trim, caption and lyrics alone. */
export const MM3_COVER_MODE_DEFAULTS: Record<Mm3CoverMode, Partial<Mm3CoverSettings>> = {
  pin: { match: 100 },
  follow: { strength: 0.9, candidates: 32, stateFix: 'off' },
  'follow-depth': { strength: 0.9, candidates: 32, stateFix: 'depth' },
  'follow-stream': { strength: 0.9, candidates: 32, stateFix: 'stream' },
  product: { weight: 1, candidates: 32 },
  blend: { blend: 25, strength: 0.9, candidates: 32 },
};
export const MM3_COVER_MODES = Object.keys(MM3_COVER_MODE_DEFAULTS) as Mm3CoverMode[];
const STATE_FIXES: Mm3StateFix[] = ['off', 'depth', 'stream', 'both'];

export const DEFAULT_MM3_COVER: Mm3CoverSettings = {
  plank: '', launderSteps: 30, launderSeed: 42,
  launderCaption: '',
  trimSeconds: '', seed: -1, arSeed: -1, match: 100, preparedSourceUrl: '',
  mode: 'pin', strength: 0.9, candidates: 32, stateFix: 'off', weight: 1, blend: 25,
};

export const withMm3CoverMode = (s: Mm3CoverSettings, mode: Mm3CoverMode): Mm3CoverSettings =>
  ({ ...s, mode, ...MM3_COVER_MODE_DEFAULTS[mode] });

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** Drop retired controls when restoring an older saved panel. Drafts from
 *  before cover modes open as Pin. */
export function restoreMm3Cover(saved: unknown): Mm3CoverSettings {
  const result = { ...DEFAULT_MM3_COVER };
  if (!saved || typeof saved !== 'object') return result;
  for (const key of Object.keys(result) as Array<keyof Mm3CoverSettings>) {
    const value = (saved as Record<string, unknown>)[key];
    if (typeof value === typeof result[key] && (typeof value !== 'number' || Number.isFinite(value))) {
      Object.assign(result, { [key]: value });
    }
  }
  if (!MM3_COVER_MODES.includes(result.mode)) result.mode = 'pin';
  if (!STATE_FIXES.includes(result.stateFix)) result.stateFix = 'off';
  result.match = clamp(result.match, 0, 100);
  result.strength = clamp(result.strength, 0, 1);
  result.candidates = clamp(Math.round(result.candidates), 1, 32);
  result.weight = clamp(result.weight, 0, 4);
  result.blend = clamp(result.blend, 0, 100);
  return result;
}

export const mm3MatchBias = (match: number) => match >= 100 ? undefined : Math.round(10 ** (3 * Math.max(0, match) / 100));

export function mm3CoverInput(s: Mm3CoverSettings) {
  const trim = Number(s.trimSeconds.trim());
  const base = {
    plank: s.plank || null, semanticOnly: true,
    trimSeconds: Number.isFinite(trim) && trim > 0 ? trim : null,
    arSeed: s.arSeed,
  };
  if (s.mode === 'pin') return { ...base, ...(mm3MatchBias(s.match) === undefined ? {} : { forceBias: mm3MatchBias(s.match) }) };
  if (s.mode === 'product') return { ...base, mode: s.mode, weight: s.weight, candidates: s.candidates };
  // Blend has no state-fix control, so it never inherits one from another mode.
  return { ...base, mode: s.mode, strength: s.strength, candidates: s.candidates,
    ...(s.mode === 'blend' ? { blend: s.blend } : { stateFix: s.stateFix }) };
}

/** What a mode needs from the plank beyond its codes. */
export const mm3ModeNeeds = (mode: Mm3CoverMode) => ({ guide: mode !== 'pin', state: mode === 'blend' });

export const mm3CoverReady = (s: Mm3CoverSettings, sourceUrl?: string) => !!s.plank && (sourceUrl === undefined || s.preparedSourceUrl === sourceUrl);
