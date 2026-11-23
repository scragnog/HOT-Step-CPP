// training/audioMeta.ts — duration + embedded-tag reader (music-metadata)
//
// Duration is measured here and ONLY here — never taken from /understand, whose
// `duration` is the LM's chain-of-thought guess rather than a measurement (§4.8).
// Strip NUL padding and control characters before using tag strings.

import { open } from 'node:fs/promises';
import path from 'node:path';
import { parseFile } from 'music-metadata';

export interface AudioMetaResult {
  duration: number;   // seconds, 0 when unknown
  artist: string;
  title: string;
  album: string;
  genre: string;      // first embedded genre tag, '' when absent
  bpm: number | null; // embedded BPM tag, when the container carries one
  sampleRate: number; // container sample rate in Hz, 0 when unknown
}

const EMPTY: AudioMetaResult = { duration: 0, artist: '', title: '', album: '', genre: '', bpm: null, sampleRate: 0 };

/** Control chars to drop from tag text — everything below 0x20 except \t and \n,
 *  plus DEL. Built from escapes so the source file stays plain ASCII. */
const CONTROL_CHARS = new RegExp('[\\u0000-\\u0008\\u000B\\u000C\\u000E-\\u001F\\u007F]', 'g');

function bpmTag(rawBpm: unknown): number | null {
  return typeof rawBpm === 'number' && Number.isFinite(rawBpm) && rawBpm > 0 ? Math.trunc(rawBpm) : null;
}

/** BPM tag only; analysis routes do not need a full duration scan. */
export async function readBpmTag(audioPath: string): Promise<number | null> {
  try { return bpmTag((await parseFile(audioPath, { skipCovers: true })).common.bpm); }
  catch { return null; }
}

/** Strip NULs and control chars (keeping \n and \t), trim, cap at 300 (§7.6). */
export function sanitizeTag(value: unknown): string {
  if (typeof value !== 'string') return '';
  return value.replace(CONTROL_CHARS, '').trim().slice(0, 300);
}

type WaveTagField = 'artist' | 'album' | 'title';
const INFO_FIELDS = new Map<string, WaveTagField>([
  ['IART', 'artist'], ['IPRD', 'album'], ['INAM', 'title'],
]);
type WaveTagRepairs = Partial<Record<WaveTagField, { ascii: string; utf8: string }>>;
const utf8 = new TextDecoder('utf-8', { fatal: true });

/** music-metadata 11.12 decodes WAV LIST/INFO as ASCII, discarding each high
 * bit. Recover valid UTF-8 from the original bytes, without reading audio
 * payloads. Keep the ASCII form to identify values produced by the old reader.
 */
export async function readWaveTagRepairs(audioPath: string): Promise<WaveTagRepairs> {
  if (path.extname(audioPath).toLowerCase() !== '.wav') return {};
  try {
    const file = await open(audioPath, 'r');
    try {
      const header = Buffer.alloc(12);
      if ((await file.read(header, 0, 12, 0)).bytesRead !== 12 ||
          header.toString('latin1', 0, 4) !== 'RIFF' ||
          header.toString('latin1', 8, 12) !== 'WAVE') return {};
      const end = Math.min(header.readUInt32LE(4) + 8, (await file.stat()).size);
      const repairs: WaveTagRepairs = {};
      const seen = new Set<WaveTagField>();
      for (let offset = 12; offset + 8 <= end;) {
        if ((await file.read(header, 0, 8, offset)).bytesRead !== 8) return {};
        const id = header.toString('latin1', 0, 4);
        const size = header.readUInt32LE(4);
        const body = offset + 8;
        const next = body + size + (size % 2);
        if (next > end) return {};
        offset = next;

        if (id === 'CSET' && size >= 8) {
          if ((await file.read(header, 0, 2, body)).bytesRead !== 2) return {};
          const codePage = header.readUInt16LE(0);
          if (codePage !== 0 && codePage !== 65001) return {};
        }
        if (id !== 'LIST' || size < 4) continue;
        if ((await file.read(header, 0, 4, body)).bytesRead !== 4) return {};
        if (header.toString('latin1', 0, 4) !== 'INFO') continue;

        for (let tag = body + 4; tag + 8 <= body + size;) {
          if ((await file.read(header, 0, 8, tag)).bytesRead !== 8) return {};
          const field = INFO_FIELDS.get(header.toString('latin1', 0, 4));
          const length = header.readUInt32LE(4);
          const valueOffset = tag + 8;
          tag = valueOffset + length + (length % 2);
          if (tag > body + size) return {};
          if (!field || seen.has(field)) continue;
          seen.add(field);
          if (!length || length > 64 * 1024) continue;
          const raw = Buffer.alloc(length);
          if ((await file.read(raw, 0, length, valueOffset)).bytesRead !== length) return {};
          try {
            const text = sanitizeTag(utf8.decode(raw));
            const ascii = sanitizeTag(raw.toString('ascii'));
            if (text && text !== ascii) repairs[field] = { ascii, utf8: text };
          } catch {
            // The bytes are not UTF-8. Keep the metadata reader's value.
          }
        }
      }
      return repairs;
    } finally {
      await file.close();
    }
  } catch {
    return {};
  }
}

/** Read duration + artist/title/album. Never throws — returns zeros on failure. */
export async function read(audioPath: string): Promise<AudioMetaResult> {
  try {
    const md = await parseFile(audioPath, { duration: true, skipCovers: true });
    const repairs = md.format.tagTypes.includes('exif') ? await readWaveTagRepairs(audioPath) : {};
    const text = (field: WaveTagField) => {
      const value = sanitizeTag(md.common[field]);
      return repairs[field]?.ascii === value ? repairs[field].utf8 : value;
    };
    const rawBpm = (md.common as { bpm?: unknown }).bpm;
    const bpm = bpmTag(rawBpm);
    return {
      duration: md.format.duration && Number.isFinite(md.format.duration) ? md.format.duration : 0,
      artist: text('artist'),
      title: text('title'),
      album: text('album'),
      genre: sanitizeTag(md.common.genre?.[0]),
      bpm,
      sampleRate: md.format.sampleRate && Number.isFinite(md.format.sampleRate) ? md.format.sampleRate : 0,
    };
  } catch (err: any) {
    console.warn(`[Training] Tag read failed for ${audioPath}: ${err?.message || err}`);
    return { ...EMPTY };
  }
}

/** Duration only, in seconds. 0 when it cannot be determined. */
export async function readDuration(audioPath: string): Promise<number> {
  const meta = await read(audioPath);
  return meta.duration;
}
