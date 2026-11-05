import type { CoverRenderInput } from '../../../contracts/studioWorkflows.js';

/** Bare tags for MuLaCover's style encoder, never inferred from a caption. */
export function heartmulaCoverTags(fields: CoverRenderInput['heartmulaTags']): string {
  if (!fields) throw new Error('Enter the four HeartMuLa tag fields');
  const tags = (['topic', 'genre', 'instrument', 'mood'] as const).map(key => {
    const value = fields[key];
    if (typeof value !== 'string' || /[\[\];\r\n]/.test(value)) throw new Error(`Invalid HeartMuLa ${key} tag`);
    return `${key}:[${value.trim().toLowerCase()}]`;
  }).join('; ');
  if (Buffer.byteLength(tags) > 4096) throw new Error('HeartMuLa tags exceed 4096 bytes');
  return tags;
}

/** Keep explicit supported sections; unmarked text starts a Verse. */
export function heartmulaCoverLyrics(text: string): string {
  if (Buffer.byteLength(text) > 32768) throw new Error('HeartMuLa lyrics exceed 32768 bytes');
  const lyrics = text.trim();
  if (!lyrics) return '[Intro]';
  const normalized = lyrics.replace(/\[([^\]\r\n]+)\]/g, (_all, raw: string) => {
    const match = /^(intro|verse|chorus|bridge|outro)(?:\s+\d+)?$/i.exec(raw.trim());
    if (!match) throw new Error(`Unsupported HeartMuLa lyric section: ${raw}`);
    return `[${match[1][0].toUpperCase()}${match[1].slice(1).toLowerCase()}]`;
  });
  const result = normalized.startsWith('[') ? normalized : `[Verse]\n${normalized}`;
  if (Buffer.byteLength(result) > 32768) throw new Error('HeartMuLa lyrics exceed 32768 bytes');
  return result;
}
