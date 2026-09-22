import fs from 'fs';
import type { Yue2AitkRunRecord } from './yue2AitkRuns.js';

interface CaptionTrack {
  name: string;
  caption: string;
  styled: string;
  genre: string;
  bpm: string;
  key: string;
}

function readJson(file: unknown, maxBytes: number): Record<string, unknown> | undefined {
  if (typeof file !== 'string' || !file) return undefined;
  try {
    const stat = fs.statSync(file);
    if (!stat.isFile() || stat.size > maxBytes) return undefined;
    const value: unknown = JSON.parse(fs.readFileSync(file, 'utf8'));
    return value && typeof value === 'object' && !Array.isArray(value)
      ? value as Record<string, unknown> : undefined;
  } catch { return undefined; }
}

/** The native manifest holds the exact style string encoded for each training
 *  item — opener, caption and metadata tail already composed.
 *
 *  `captionFor` (the route's `datasetTrackCaption`) gets first refusal on the
 *  caption itself: a `.yue2.txt` written after the prepare ran is newer than
 *  anything baked in here. Only the CAPTION is swapped, by substring, so the
 *  trigger opener and the tail this run trained with survive untouched — a
 *  recomposed style is a near-miss, and a near-miss is off distribution in
 *  exactly the way the picker exists to avoid. */
export function jointCaptionTracks(
  run: Yue2AitkRunRecord,
  captionFor?: (audioPath: string, baked: string) => string,
): CaptionTrack[] {
  const prepared = readJson(run.options.dataset, 64 * 1024 * 1024);
  const items = prepared?.items;
  if (!Array.isArray(items) || items.length > 10000) return [];

  const preparation = run.options.preparation;
  const legacyFile = preparation && typeof preparation === 'object'
    ? (preparation as Record<string, unknown>).legacyManifest : undefined;
  const legacy = readJson(legacyFile, 16 * 1024 * 1024);
  const sourceByName = new Map<string, Record<string, unknown>>();
  if (Array.isArray(legacy?.sources)) {
    for (const source of legacy.sources) {
      if (source && typeof source === 'object' && typeof source.name === 'string') {
        sourceByName.set(source.name, source as Record<string, unknown>);
      }
    }
  }
  const str = (value: unknown): string => typeof value === 'string' ? value : '';
  return items.flatMap((value: unknown) => {
    if (!value || typeof value !== 'object') return [];
    const item = value as Record<string, unknown>;
    const name = str(item.id);
    let style = str(item.style);
    if (!name || !style) return [];
    const source = sourceByName.get(name);
    const baked = str(source?.caption);
    const fresh = captionFor ? captionFor(str(source?.source), baked) : baked;
    // Substring, not recompose: `style` is opener + baked caption + tail, and
    // only the middle is out of date. A baked caption that is not in the style
    // string means this run composed it some other way — leave it alone.
    if (fresh && baked && fresh !== baked && style.includes(baked)) {
      style = style.replace(baked, fresh);
    }
    return [{
      name, caption: style, styled: style,
      genre: str(source?.genre), bpm: str(source?.bpm), key: str(source?.key),
    }];
  });
}
