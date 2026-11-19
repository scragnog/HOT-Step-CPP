// Persistent vocal-stem cache for Cover Studio matching. Keyed on audio
// content, separation level and implementation version, so a byte-identical
// re-upload (new path, new mtime) is a hit. Nothing here is ever evicted.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { NoVocalsError } from './training/yue2Stems.js';

/** Bump when separation changes what a stem sounds like; old entries become misses. */
export const COVER_STEM_VERSION = 1;

export type StemSeparator = (audioPath: string, outPath: string, level: number) => Promise<void>;

/** A usable WAV: chunks stay inside the file, `fmt ` is a supported format, and
 *  `data` holds at least one whole audio frame. Exported for the tests. */
export function isCompleteWav(file: string): boolean {
  try {
    const buf = fs.readFileSync(file);
    if (buf.length < 12 || buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') return false;
    const end = buf.readUInt32LE(4) + 8;  // chunks must lie inside the declared RIFF
    if (end < 12 || end > buf.length) return false;
    let blockAlign = 0;
    for (let pos = 12; pos + 8 <= end;) {
      const id = buf.toString('ascii', pos, pos + 4);
      const size = buf.readUInt32LE(pos + 4);
      const body = pos + 8;
      if (body + size > end) return false;
      if (id === 'fmt ') {
        if (size < 16) return false;
        let tag = buf.readUInt16LE(body);
        const channels = buf.readUInt16LE(body + 2);
        const rate = buf.readUInt32LE(body + 4);
        blockAlign = buf.readUInt16LE(body + 12);
        const bits = buf.readUInt16LE(body + 14);
        if (tag === 0xfffe) {  // extensible: the subformat decides, as in engine/src/wav.h
          if (size < 40) return false;
          tag = buf.readUInt16LE(body + 24);
        }
        // Same formats the engine decodes: PCM 16/24-bit and float 32-bit.
        const ok = (tag === 1 && (bits === 16 || bits === 24)) || (tag === 3 && bits === 32);
        if (!ok || !channels || !rate || blockAlign !== channels * (bits / 8)) return false;
      } else if (id === 'data') {
        return blockAlign > 0 && size >= blockAlign && size % blockAlign === 0;
      }
      pos = body + size + (size & 1);
    }
    return false;  // no data chunk
  } catch { return false; }
}

export function createCoverStemCache(opts: { root: string; level: number; separate: StemSeparator; version?: number }) {
  const version = opts.version ?? COVER_STEM_VERSION;
  const inflight = new Map<string, Promise<string>>();

  async function produce(audioPath: string, base: string): Promise<string> {
    const wav = `${base}.wav`;
    const none = `${base}.no-vocals`;
    // A no-vocals result is cached as a marker and replayed as the same error.
    if (fs.existsSync(none)) throw new NoVocalsError('separation found no vocals');
    if (fs.existsSync(wav)) {
      if (isCompleteWav(wav)) return wav;
      fs.rmSync(wav, { force: true });  // corrupt entry: redo it
    }
    fs.mkdirSync(opts.root, { recursive: true });
    // Unique part name, renamed into place: a cut-short separation is never published.
    const part = `${base}.${process.pid}.${crypto.randomUUID()}.part`;
    try {
      await opts.separate(audioPath, part, opts.level);
      if (!isCompleteWav(part)) throw new Error('separation wrote an incomplete stem');
      fs.renameSync(part, wav);
    } catch (err) {
      fs.rmSync(part, { force: true });
      if (err instanceof NoVocalsError) fs.writeFileSync(none, 'No vocals in this audio.\n');
      throw err;
    }
    return wav;
  }

  /** Content-addressed, so a re-upload under a new path is the same stem. */
  function baseOf(audioPath: string): string {
    const hash = crypto.createHash('sha256').update(fs.readFileSync(audioPath)).digest('hex');
    return path.join(opts.root, `${hash}-L${opts.level}-v${version}`);
  }

  return {
    /**
     * The stem already on disk for this audio, or null. Never separates.
     *
     * Alignment uses this to take a free improvement when cover or stem work
     * has already produced one, instead of spending a GPU job on the chance
     * that separation helps.
     */
    find(audioPath: string): string | null {
      const wav = `${baseOf(audioPath)}.wav`;
      return fs.existsSync(wav) && isCompleteWav(wav) ? wav : null;
    },

    /** The cached vocal stem for this audio; throws NoVocalsError for a silent source. */
    async vocalStem(audioPath: string): Promise<string> {
      const base = baseOf(audioPath);
      let run = inflight.get(base);
      if (!run) {
        run = produce(audioPath, base).finally(() => inflight.delete(base));
        inflight.set(base, run);
      }
      return run;
    },
  };
}
