import type { Yue2SourceTrack } from './yue2CaptionSource';

/** Short phrases the selected dataset actually used as caption context. */
export function extractCaptionTags(tracks: readonly Pick<Yue2SourceTrack, 'styled' | 'caption'>[]): string[] {
  const counts = new Map<string, { phrase: string; count: number }>();
  for (const track of tracks) {
    const seen = new Set<string>();
    const source = track.styled?.trim() || track.caption || '';
    for (const raw of source.split(/[,.]/)) {
      const phrase = raw.trim();
      if (!phrase || phrase.length > 40
        || /^\d+(?:\.\d+)?(?:\s*BPM)?$/i.test(phrase)
        || /^key of\b/i.test(phrase)
        || /^in the style of\b/i.test(phrase)) continue;
      const key = phrase.toLocaleLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      const hit = counts.get(key);
      if (hit) hit.count++;
      else counts.set(key, { phrase, count: 1 });
    }
  }
  return [...counts.values()]
    .sort((a, b) => b.count - a.count || a.phrase.localeCompare(b.phrase, undefined, { sensitivity: 'base' }))
    .slice(0, 40)
    .map(({ phrase }) => phrase);
}
