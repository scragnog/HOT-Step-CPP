// training/yue2Align.ts — `ace-train yue2-align`: model resolution, defaults
// and the argv builder for the third of the three cache stages.
//
// Sibling of yue2Tokenize.ts, and the same shape for the same reason: this
// stage takes the manifest `yue2-preprocess` wrote, not an audio directory, and
// rewrites it in place (leaving a `.bak` beside it).
//
// WHAT IT PRODUCES: CTC forced alignment of each source's own lyrics against
// its vocal stem, through the native MMS_FA port — per-word spans written to
// `<manifest dir>/cursor/<source stem>.f32`, five f32 columns per word (start,
// end, score, and the two CODEPOINT offsets into the manifest's `lyrics`
// string). Every aligned source gains a manifest-relative `cursor_words` path.
// `yue2-ar-train --cursor-weight` reads them; a source without one simply
// trains with the cursor loss off, which is legal and silent, so the count
// below is worth showing before a run rather than after.
//
// It replaces engine/tools/yue2-cursor-bridge.py and the torchaudio install
// behind it. The bridge's whole REFUSED-on-mismatch path is gone with it: it
// aligned a separate lyrics file and had to compare it byte for byte against
// the manifest field, and this aligns the manifest's own string.
//
// THREE ENGINE FACTS THAT LOOK LIKE MISSING OPTIONS:
//
//   1. No `--models` discovery. Unlike every other YuE2 subcommand, this one
//      takes an explicit `--mmsfa <path>` and nothing else, so the server has
//      to resolve the file itself — see resolveYue2AlignerModel.
//   2. No `--force`, because there is no cache-skip to override: a cursor file
//      carries no shape a check could verify (its length depends on the
//      lyrics, not the audio), and a stale one written before a lyric edit is
//      exactly the input that trains a plausible, wrong cursor. `only` and
//      `limit` are how you do less work.
//   3. No chunk size. MMS_FA layer-normalises the waveform once over its whole
//      input, so a track aligned in pieces is a different model input in every
//      piece. One track, one forward: budget ~5 GB and ~100 s per four minutes
//      on 16 CPU threads.
//
// STEMS ARE AN INPUT THIS STAGE DOES NOT PRODUCE. The engine wants a VOCAL
// stem per source at `<stems>/<source stem>/vocals.wav` (the Demucs htdemucs
// layout) and SKIPS BY NAME when there is none. Separation happens elsewhere —
// SuperSep through the running engine, the way tools/vocal-end does it.

import fs from 'fs';
import path from 'path';

import { config } from '../../config.js';
import { datasetDir } from './paths.js';
import { yue2ModelDir } from './yue2Train.js';

// ── Model files ─────────────────────────────────────────────────────────────

/** The MMS_FA weights, or '' when none is installed.
 *
 *  `<models>/yue2` then `<models>`, which is the pair every other YuE2 tool
 *  searches — but here the server does the searching, because `yue2-align`
 *  has no `--models` flag and an unresolved path would reach the engine as
 *  "--mmsfa is required".
 *
 *  `mms-fa-f32.gguf` is preferred by name rather than by a quant ranking:
 *  convert-mms-fa.py emits F32 only, and the port's stage gates (14/14) were
 *  measured in fp32. An F16 build is listed as "later, if the gates hold", so
 *  a `mms-fa-*.gguf` fallback is here to find it without a code change — not
 *  as an endorsement of it. */
export function resolveYue2AlignerModel(): string {
  for (const dir of [yue2ModelDir(), config.aceServer.models]) {
    const preferred = path.join(dir, 'mms-fa-f32.gguf');
    if (fs.existsSync(preferred)) return preferred;
    let hits: string[] = [];
    try {
      hits = fs.readdirSync(dir).filter(f => f.startsWith('mms-fa-') && f.endsWith('.gguf')).sort();
    } catch {
      continue;  // a missing models dir reads as a missing file, never a throw
    }
    if (hits.length) return path.join(dir, hits[0]);
  }
  return '';
}

/** Which required files are missing, as user-facing names. Empty = ready.
 *
 *  The aligner is the ONLY model this stage loads: the lyrics come from the
 *  manifest and the audio from the stems directory, so there is no LM, no VAE
 *  and no tokenizer to check for.
 *
 *  NOTE FOR WHOEVER WIRES THE ROUTE: mms-fa-f32.gguf has no entry in
 *  server/src/data/model-registry.json, so the Model Manager cannot fetch it
 *  and a user who downloads a release has no way to get it. Until it is
 *  published and registered, this readiness check is the only thing standing
 *  between them and a stage that is simply dead — so surface the name, do not
 *  swallow it. */
export function missingYue2AlignModels(): string[] {
  return resolveYue2AlignerModel()
    ? []
    : ['the MMS_FA forced aligner (mms-fa-f32.gguf)'];
}

// ── Layout ──────────────────────────────────────────────────────────────────

/** `<training>/datasets/<slug>/yue2-stems` — where separated vocals for this
 *  dataset are expected, one folder per source stem holding `vocals.wav`.
 *
 *  Under the dataset for the reason yue2LatentsDir gives: deleting a dataset
 *  should take its derived data with it. The per-source subfolder is not a
 *  choice made here — the engine joins `<stems>/<name without extension>/
 *  vocals.wav` and skips anything it cannot find there. A caller with stems
 *  somewhere else passes its own path; this is only the default. */
export function yue2StemsDir(slug: string): string {
  return path.join(datasetDir(slug), 'yue2-stems');
}

// ── Defaults ────────────────────────────────────────────────────────────────
//
// The engine's own. There is no recipe here to tune: the alignment is
// deterministic given the stem and the lyrics, and every knob below chooses
// how much of it to do or where it runs.
export const YUE2_ALIGN_DEFAULTS = {
  /** Case-insensitive substring of the source name, and a cap on how many
   *  matching sources are taken. Also the re-run mechanism, since there is no
   *  `--force`: name the source whose lyrics you edited. */
  only: '',
  limit: 0,
  /** Pin the backend to CPU. Off: the GPU default is several times faster and
   *  the port's gates hold on both. The engine sets GGML_BACKEND before any
   *  context exists, which is the only moment it can be set — there is no
   *  later override, so this cannot be changed mid-run. */
  cpu: false,
} as const;

// ── Manifest state ──────────────────────────────────────────────────────────

/** What the manifest says about this stage. There is no `present` flag to read
 *  (unlike codec_ids): the aligner stamps provenance at the root and a path
 *  per source, so the count IS the state. */
export interface Yue2CursorWordsStatus {
  /** Basename of the aligner that produced them, as stamped into the manifest. */
  model: string;
  createdAt: string;
  sources: number;
  /** Sources carrying a `cursor_words` path. The rest train with the cursor
   *  loss off — the AR trainer notes it once and continues. */
  sourcesWithCursor: number;
}

/** Read that state. Null when there is no manifest, or when it is mid-write:
 *  this feeds a polled endpoint and a route guard, and neither may 500 over a
 *  partial file. Same contract as readYue2PreprocessSummary. */
export function readYue2CursorWordsStatus(manifestPath: string): Yue2CursorWordsStatus | null {
  try {
    const j = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as Record<string, unknown>;
    const sources = Array.isArray(j.sources) ? j.sources as Array<Record<string, unknown>> : [];
    return {
      model: typeof j.cursor_words_model === 'string' ? j.cursor_words_model : '',
      createdAt: typeof j.cursor_words_created_at === 'string' ? j.cursor_words_created_at : '',
      sources: sources.length,
      sourcesWithCursor: sources.filter(
        s => typeof s.cursor_words === 'string' && s.cursor_words !== '').length,
    };
  } catch {
    return null;
  }
}

// ── Arg building ────────────────────────────────────────────────────────────
//
// Every flag below is one cmd_yue2_align actually parses
// (engine/tools/ace-train.cpp). An unknown option is a hard exit 2, so nothing
// speculative belongs here.

export interface ResolvedYue2AlignOptions {
  /** `<latents>/yue2_preprocess.json` — REWRITTEN IN PLACE, with a `.bak` left
   *  beside it. Not an audio directory: the sources, and the lyrics the spans
   *  are measured against, are the ones this manifest already carries. */
  manifest: string;
  /** Holds `<source stem>/vocals.wav` per source. Required by the engine, and
   *  a mixed song here silently aligns against the instruments too. */
  stemsDir: string;
  only: string;
  limit: number;
  cpu: boolean;
  /** Carried for the runner's own log lines and the run record. */
  datasetSlug: string;
}

export function buildYue2AlignArgs(o: ResolvedYue2AlignOptions): string[] {
  // Resolved at spawn time rather than stored, so a model moved between
  // queueing and running fails with a named file instead of a stale path.
  const mmsfa = resolveYue2AlignerModel();
  const args = [
    'yue2-align',
    '--manifest', o.manifest,
    // Emitted even when unresolved: '' reaches the engine as its own
    // "--mmsfa <mms-fa-f32.gguf> is required" exit, which is the honest error.
    // The route should have refused on missingYue2AlignModels() before here.
    '--mmsfa', mmsfa,
    '--stems', o.stemsDir,
  ];
  if (o.only) args.push('--only', o.only);
  if (o.limit > 0) args.push('--limit', String(o.limit));
  if (o.cpu) args.push('--cpu');
  return args;
}
