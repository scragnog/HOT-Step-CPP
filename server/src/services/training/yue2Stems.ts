// yue2/stems — vocal stems for the YuE2 lyric aligner, one dataset at a time.
//
// `ace-train yue2-align` reads `<stemsDir>/<source stem>/vocals.wav` and skips
// by name, so a dataset with no stems aligns nothing and still exits 0 — which
// is why the align route refuses to start without them rather than let a run
// look healthy. Nothing in the app produced them for a training dataset: Stem
// Studio separates ONE uploaded track into its own job directory, and the
// SuperSep route resolves paths inside data/references or data/audio only.
// This walks a dataset's audio instead and writes the one stem the aligner
// wants, in the layout it expects.
//
// It talks to the same engine endpoints Stem Studio does
// (/supersep/separate -> /progress -> /result -> /serve -> /release); the
// release call is not optional, since the engine's job pool has no eviction and
// a forgotten job holds about a gigabyte until ace-server restarts (#133).

import fs from 'fs';
import path from 'path';

import { config } from '../../config.js';
import { ensureEngineFormat } from '../audioConvert.js';
import { yue2StemsDir } from './yue2Align.js';

const ACE_URL = `http://127.0.0.1:${config.aceServer.port}`;

/** Which stem the aligner wants. SuperSep names its vocal stem in a handful of
 *  ways depending on level; the aligner only ever opens `vocals.wav`, so the
 *  match is on the name we are given and the file we write is fixed. */
const VOCAL_NAMES = ['vocals', 'vocal', 'lead vocals', 'lead_vocals'];

/** SUPERSEP_VOCALS_ONLY. The aligner opens one file, vocals.wav, so this stage
 *  has no use for a drum kit or a piano — and level 0 would produce them at
 *  real cost: a Leap Xe vocal pass, a Leap Xe instrumental pass, and a six-stem
 *  BS-RoFormer split on top, five stems of which are freed unheard. Level 4
 *  runs the same BS-RoFormer weights with the mask pinned to the vocal stem, so
 *  it is one pass and the other five stems never leave the frequency domain.
 *
 *  Its vocal is lead PLUS backing, which is the right call for lyrics: a
 *  karaoke-style lead-only split would drop the doubled and harmonised lines
 *  the aligner still has to place. */
export const YUE2_STEMS_DEFAULT_LEVEL = 4;

export interface Yue2StemsProgress {
  /** 1-based index of the track being separated. */
  index: number;
  total: number;
  name: string;
  /** The engine's own 0..1 for the current track, or null between tracks. */
  fraction: number | null;
}

export interface Yue2StemsResult {
  written: number;
  skipped: number;
  failed: Array<{ name: string; error: string }>;
  stemsDir: string;
}

const AUDIO_EXT = new Set(['.flac', '.wav', '.mp3', '.ogg', '.m4a', '.aac']);

/** The dataset's audio files, in the order the manifest lists them so progress
 *  reads the same as every other YuE2 stage. */
export function yue2StemSources(audioDir: string): string[] {
  let names: string[] = [];
  try {
    names = fs.readdirSync(audioDir);
  } catch {
    return [];
  }
  return names
    .filter(f => AUDIO_EXT.has(path.extname(f).toLowerCase()))
    .sort()
    .map(f => path.join(audioDir, f));
}

/** The stems folder name for a source: its basename without the extension,
 *  minus trailing dots and spaces. Windows drops those from a path, so
 *  "04..Get.Happy.Dirty..flac" wrote a folder the aligner could not open.
 *  Same rule as source_stem() in yue2-align-run.h. */
export function yue2StemName(sourceFile: string): string {
  return path.parse(sourceFile).name.replace(/[. ]+$/, '') || '_';
}

/** `<stemsDir>/<source stem>/vocals.wav`, the one path yue2-align opens. */
export function yue2VocalPath(stemsDir: string, sourceFile: string): string {
  return path.join(stemsDir, yue2StemName(sourceFile), 'vocals.wav');
}

/** Left in place of vocals.wav when separation finds no singing: the track's
 *  lyrics are wrong for the recording (an interlude with placeholder lyrics).
 *  yue2-align reads it as instrumental and the track trains untimed. */
export function yue2NoVocalsPath(stemsDir: string, sourceFile: string): string {
  return path.join(stemsDir, yue2StemName(sourceFile), 'no-vocals');
}

class NoVocalsError extends Error {}

async function separateOne(
  srcPath: string,
  outPath: string,
  level: number,
  onFraction: (f: number) => void,
): Promise<void> {
  let body: Buffer;
  try {
    // Convert into our own scratch dir, never beside the source: the source is
    // the user's dataset folder and the scanner walks it.
    body = ensureEngineFormat(srcPath, path.join(config.data.dir, 'tmp', 'yue2-stems'));
  } catch {
    body = fs.readFileSync(srcPath);  // the engine decodes more than we convert
  }

  const started = await fetch(`${ACE_URL}/supersep/separate?level=${level}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body,
  });
  if (!started.ok) throw new Error(`separate: ${await started.text()}`);
  const { id } = await started.json() as { id: string };

  try {
    // 500 ms between polls, the cadence Stem Studio uses; a long track on a
    // busy card is minutes, not hours, but the ceiling is generous because the
    // alternative is abandoning a job the engine is still holding.
    for (let i = 0; i < 14400; i++) {
      const res = await fetch(`${ACE_URL}/supersep/progress?id=${id}`);
      const p = await res.json() as { status: string; progress: number; error?: string };
      if (typeof p.progress === 'number') onFraction(p.progress);
      if (p.status === 'done') break;
      if (p.status === 'failed' || p.status === 'cancelled') {
        throw new Error(p.error || `separation ${p.status}`);
      }
      await new Promise(r => setTimeout(r, 500));
    }

    const listed = await fetch(`${ACE_URL}/supersep/result?id=${id}`);
    if (!listed.ok) throw new Error('result: engine would not list the stems');
    const { stems } = await listed.json() as {
      stems: Array<{ name: string; index: number; hidden?: boolean }>;
    };
    const vocal = stems.find(s => VOCAL_NAMES.includes(s.name.trim().toLowerCase()));
    if (!vocal) {
      // SuperSep drops a silent stem, so none named vocals means no singing.
      if (stems.length) throw new NoVocalsError('separation found no vocals');
      throw new Error('separation returned no stems');
    }

    const got = await fetch(`${ACE_URL}/supersep/serve?id=${id}&stem=${vocal.index}`);
    if (!got.ok) throw new Error(`serve: stem ${vocal.index} could not be fetched`);
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, Buffer.from(await got.arrayBuffer()));
  } finally {
    // Always, including on the throw paths above: the pool has no eviction.
    await fetch(`${ACE_URL}/supersep/release?id=${id}`, { method: 'POST' }).catch(() => {});
  }
}

/** Separate a whole dataset into the layout yue2-align reads.
 *
 *  Already-present stems are skipped rather than recomputed — separation is the
 *  most expensive stage in the AR pipeline by a wide margin, and a rerun after
 *  one failure should cost one track, not thirteen. Pass `force` to redo them.
 *  A track that fails is recorded and the walk continues: eleven aligned songs
 *  are worth more than none, and the align stage reports its own coverage. */
export async function yue2SeparateDataset(opts: {
  audioDir: string;
  slug: string;
  level?: number;
  force?: boolean;
  onProgress?: (p: Yue2StemsProgress) => void;
  isCancelled?: () => boolean;
}): Promise<Yue2StemsResult> {
  const stemsDir = yue2StemsDir(opts.slug);
  const sources = yue2StemSources(opts.audioDir);
  const out: Yue2StemsResult = { written: 0, skipped: 0, failed: [], stemsDir };

  for (let i = 0; i < sources.length; i++) {
    if (opts.isCancelled?.()) break;
    const src = sources[i];
    const name = path.basename(src);
    const dest = yue2VocalPath(stemsDir, src);

    const noVocals = yue2NoVocalsPath(stemsDir, src);
    if (!opts.force && ((fs.existsSync(dest) && fs.statSync(dest).size > 0) || fs.existsSync(noVocals))) {
      out.skipped++;
      opts.onProgress?.({ index: i + 1, total: sources.length, name, fraction: 1 });
      continue;
    }

    opts.onProgress?.({ index: i + 1, total: sources.length, name, fraction: null });
    try {
      await separateOne(src, dest, opts.level ?? YUE2_STEMS_DEFAULT_LEVEL, f => {
        opts.onProgress?.({ index: i + 1, total: sources.length, name, fraction: f });
      });
      out.written++;
    } catch (err) {
      if (err instanceof NoVocalsError) {
        fs.mkdirSync(path.dirname(noVocals), { recursive: true });
        fs.writeFileSync(noVocals, 'SuperSep found no vocals in this track.\n');
        out.written++;
        continue;
      }
      out.failed.push({ name, error: err instanceof Error ? err.message : String(err) });
    }
  }
  return out;
}
