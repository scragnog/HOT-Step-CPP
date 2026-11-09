// Settings for the laundered, semantic-only MM3 cover panel.
export interface Mm3CoverSettings {
  plank: string;
  launderSteps: number;
  launderSeed: number;
  launderCaption: string;
  /** Blank uses the whole plank. */
  trimSeconds: string;
  seed: number;
  arSeed: number;
}

export const DEFAULT_MM3_COVER: Mm3CoverSettings = {
  plank: '', launderSteps: 30, launderSeed: 42,
  launderCaption: 'a full song with vocals and instruments',
  trimSeconds: '', seed: -1, arSeed: -1,
};

/** Drop retired controls when restoring an older saved panel. */
export function restoreMm3Cover(saved: unknown): Mm3CoverSettings {
  const result = { ...DEFAULT_MM3_COVER };
  if (!saved || typeof saved !== 'object') return result;
  for (const key of Object.keys(result) as Array<keyof Mm3CoverSettings>) {
    const value = (saved as Record<string, unknown>)[key];
    if (typeof value === typeof result[key] && (typeof value !== 'number' || Number.isFinite(value))) {
      Object.assign(result, { [key]: value });
    }
  }
  return result;
}

export function mm3CoverInput(s: Mm3CoverSettings) {
  const trim = Number(s.trimSeconds.trim());
  return {
    plank: s.plank || null, semanticOnly: true,
    trimSeconds: Number.isFinite(trim) && trim > 0 ? trim : null,
    arSeed: s.arSeed,
  };
}

export const mm3CoverReady = (s: Mm3CoverSettings) => !!s.plank;
