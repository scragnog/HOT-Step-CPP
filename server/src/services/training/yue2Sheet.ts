// training/yue2Sheet.ts — `ace-train yue2-sheet`: model resolution, defaults
// and the argv builder for the seventh cache stage, "Lead sheets".
//
// Same shape as yue2Tokenize.ts, which this file is modelled on line for
// line: no audio directory, no dataset, no captions — it takes the manifest
// `yue2-preprocess` already wrote (yue2-tokenize's codes and yue2-align's
// cursor spans are siblings, not prerequisites) and rewrites it in place.
//
// WHAT IT PRODUCES: for every SOURCE without `abc`/`abc_error` (or all with
// `--force`), the SheetSage2 transcriber (engine/src/yue2/sheetsage-*.h)
// decodes that source's own audio and writes either `abc` (the rendered ABC
// lead sheet) or `abc_error` (the reference's own typed notation failure
// text) into its manifest row, plus `abc_producer`/`abc_windows`/`abc_ms`.
// `yue2-ar-train --abc-dropout` and `yue2-nar-train --abc-dropout` read
// `abc`/`abc_error` per source to draw cot=full vs cot=off; a source with
// neither (not yet run, or a decode-level infra failure) always trains
// cot=off, same as before this stage existed.
//
// SOFT FAILURE IS EXPECTED (docs/plans/yue2/19-sheetsage2-cot-training.md,
// "Decisions taken after phase 1"): roughly 4% of real tracks decode fine but
// fail to RENDER — a chord/melody interval too short for the ABC subbeat
// grid, no key decoded, etc. That is `abc_error`, not a job failure; the
// engine's own exit code stays 0 and this file's status reader counts
// `sourcesWithError` as progress, not as trouble.
//
// LICENCE: like the rest of the YuE2 family, CC BY-NC 4.0. Notice lives in
// services/backends/yue2/index.ts.

import fs from 'fs';
import path from 'path';

import { config } from '../../config.js';
import { yue2ModelDir } from './yue2Train.js';

// ── Model files ─────────────────────────────────────────────────────────────

/** The two directories the engine's own discovery scans (`find_model` in
 *  engine/src/train/yue2-sheet-run.h: `<models>/yue2` first, then
 *  `<models>`), mirrored so this readiness check agrees with what `--models`
 *  will actually find. */
function sheetModelSearchDirs(): string[] {
  return [yue2ModelDir(), config.aceServer.models];
}

/** The `sheetsage2-*.gguf` the engine would pick, or '' when there is none.
 *
 *  NOT passed to the engine — `buildYue2SheetArgs` sends `--models` and lets
 *  the engine's own quant ranking choose, same reasoning as
 *  resolveYue2TokenizerModel. This resolves the same file only to name it in
 *  a status payload and to answer "is the stage installed". */
export function resolveYue2SheetModel(): string {
  for (const dir of sheetModelSearchDirs()) {
    let hits: string[] = [];
    try {
      hits = fs.readdirSync(dir).filter(f => f.startsWith('sheetsage2-') && f.endsWith('.gguf')).sort();
    } catch {
      continue;  // a missing models dir reads as a missing file, never a throw
    }
    if (hits.length) return path.join(dir, hits[0]);
  }
  return '';
}

/** Which required files are missing, as user-facing names. Empty = ready.
 *
 *  One file, and it is optional to the rest of the app: nothing in generation
 *  needs the SheetSage2 transcriber, so a fresh install will not have it.
 *  Registry id `yue2-sheetsage2-f16`, which the Model Manager can download. */
export function missingYue2SheetModels(): string[] {
  return resolveYue2SheetModel()
    ? []
    : ['the SheetSage2 lead-sheet transcriber (sheetsage2-*.gguf)'];
}

// ── Defaults ────────────────────────────────────────────────────────────────
//
// The engine's own, unchanged: this stage has no recipe to tune. It is a
// deterministic transcription of audio the manifest already names.
export const YUE2_SHEET_DEFAULTS = {
  /** Case-insensitive substring of the source name. A partial run is legal —
   *  the sources it missed stay cot=off-only and the trainer's own draw
   *  reports that — so this is the way to do less work, not a failure mode. */
  only: '',
  /** Re-transcribe sources that already carry abc/abc_error. Without it a
   *  source with either field already set is skipped, which is what makes a
   *  resumed run cheap — a single transcription can run minutes. */
  force: false,
  /** `false` runs the engine's default `exact=true` load (every weight-
   *  bearing matmul in the encoder promoted to F32 at load time) — doc 23's
   *  encoder-precision investigation found this the only way to clear the G1
   *  gate on every fixture window. `true` sends `--fast` (skip the promotion)
   *  for a quicker but measurably less precise cache pass. */
  fast: false,
} as const;

// ── Manifest state ──────────────────────────────────────────────────────────

/** What the manifest says about this stage, for a status endpoint and for the
 *  trainers' own `cot=off,full` draw eligibility. */
export interface Yue2AbcStatus {
  /** The `abc_present` flag the stage sets. */
  present: boolean;
  /** This TOOL's own identity ("ace-train yue2-sheet <version>"), stamped at
   *  the manifest root — distinct from each source's own `abc_producer`
   *  (model file + precision mode + backend). */
  producer: string;
  createdAt: string;
  sources: number;
  /** Sources carrying a non-empty `abc` — the trainers' cot=full pool. */
  sourcesWithAbc: number;
  /** Sources carrying a non-empty `abc_error` — soft failures, trained as
   *  cot=off on every draw, never in the 50/50 pool. */
  sourcesWithError: number;
  /** Sources carrying a non-empty `abc_repaired` (engine/src/yue2/sheetsage-
   *  repair.h's own short description of what it dropped) — a subset of
   *  sourcesWithAbc in the common case, but can also overlap sourcesWithError
   *  when every repair round still ended in failure (the engine still
   *  records what it tried). */
  sourcesWithRepair: number;
}

/** Read that state. Null when there is no manifest, or when it is mid-write:
 *  this feeds a polled endpoint and a route guard, and neither may 500 over a
 *  partial file. Same contract as readYue2CodecIdsStatus. */
export function readYue2AbcStatus(manifestPath: string): Yue2AbcStatus | null {
  try {
    const j = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as Record<string, unknown>;
    const sources = Array.isArray(j.sources) ? j.sources as Array<Record<string, unknown>> : [];
    const hasAbc = (r: Record<string, unknown>) => typeof r.abc === 'string' && r.abc !== '';
    const hasErr = (r: Record<string, unknown>) => typeof r.abc_error === 'string' && r.abc_error !== '';
    const hasRepair = (r: Record<string, unknown>) => typeof r.abc_repaired === 'string' && r.abc_repaired !== '';
    return {
      present: j.abc_present === true,
      producer: typeof j.abc_producer === 'string' ? j.abc_producer : '',
      createdAt: typeof j.abc_created_at === 'string' ? j.abc_created_at : '',
      sources: sources.length,
      sourcesWithAbc: sources.filter(hasAbc).length,
      sourcesWithError: sources.filter(hasErr).length,
      sourcesWithRepair: sources.filter(hasRepair).length,
    };
  } catch {
    return null;
  }
}

// ── Per-source detail (Lead-sheet preview) ──────────────────────────────────
//
// readYue2AbcStatus above answers "how far has the stage got"; these answer
// "what does one source's row actually say", for the preview picker and its
// score/audio panel. Same manifest, same fields, read a second way.

/** One row of the picker list: enough to grey out a source with no lead
 *  sheet, or to show a soft failure's `abc_error` instead of a preview. */
export interface Yue2SheetSourceStatus {
  name: string;
  /** Carries a non-empty `abc` — the picker enables these. */
  ok: boolean;
  /** Non-empty `abc_error` — the soft-failure case; the picker lists these
   *  with the text but disabled, same as an untouched source. */
  error: string;
  /** Non-empty `abc_repaired` — the picker marks these (an amber note next to
   *  the name), whether the source ended up `ok` or still `error`. */
  repaired: string;
}

/** Every source the manifest names, with the lead-sheet stage's own verdict
 *  on each. Empty on a missing/partial manifest — same "never throws"
 *  contract as readYue2AbcStatus, since a picker list also feeds a poll. */
export function listYue2SheetSources(manifestPath: string): Yue2SheetSourceStatus[] {
  try {
    const j = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as Record<string, unknown>;
    const sources = Array.isArray(j.sources) ? j.sources as Array<Record<string, unknown>> : [];
    return sources.map(r => ({
      name: typeof r.name === 'string' ? r.name : '',
      ok: typeof r.abc === 'string' && r.abc !== '',
      error: typeof r.abc_error === 'string' ? r.abc_error : '',
      repaired: typeof r.abc_repaired === 'string' ? r.abc_repaired : '',
    }));
  } catch {
    return [];
  }
}

/** One source's full lead-sheet row, by its manifest `name` (the flat
 *  filename yue2-preprocess scanned it under — see routes/training.ts's own
 *  note that this stage's audio folder is scanned flat, never recursively,
 *  so a name never carries a path separator). */
export interface Yue2SheetSourceDetail {
  name: string;
  abc: string;
  abc_error: string;
  /** engine/src/yue2/sheetsage-repair.h's own short description of what it
   *  dropped en route to `abc`/`abc_error` above; "" when no repair round
   *  ever fired. */
  abc_repaired: string;
  /** The window count `sheetsage_transcribe` used for this source, 0 when
   *  untouched. */
  abc_windows: number;
  /** This source's own tag (model file + precision mode + backend) — distinct
   *  from the manifest-root `abc_producer` (this tool's identity) that
   *  readYue2AbcStatus reads. */
  abc_producer: string;
}

/** Null when the manifest is missing/partial, or the name isn't in it — the
 *  route turns either into a 404, never a 500. */
export function readYue2SheetSource(manifestPath: string, name: string): Yue2SheetSourceDetail | null {
  try {
    const j = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as Record<string, unknown>;
    const sources = Array.isArray(j.sources) ? j.sources as Array<Record<string, unknown>> : [];
    const hit = sources.find(r => typeof r.name === 'string' && r.name === name);
    if (!hit) return null;
    return {
      name,
      abc: typeof hit.abc === 'string' ? hit.abc : '',
      abc_error: typeof hit.abc_error === 'string' ? hit.abc_error : '',
      abc_repaired: typeof hit.abc_repaired === 'string' ? hit.abc_repaired : '',
      abc_windows: typeof hit.abc_windows === 'number' ? hit.abc_windows : 0,
      abc_producer: typeof hit.abc_producer === 'string' ? hit.abc_producer : '',
    };
  } catch {
    return null;
  }
}

// ── Arg building ────────────────────────────────────────────────────────────
//
// Every flag below is one cmd_yue2_sheet actually parses
// (engine/tools/ace-train.cpp). An unknown option is a hard exit 2, so nothing
// speculative belongs here.

export interface ResolvedYue2SheetOptions {
  /** `<latents>/yue2_preprocess.json` — REWRITTEN IN PLACE, atomically after
   *  every source (a single transcription can run minutes; see the engine
   *  file's own "Crash safety" note). */
  manifest: string;
  only: string;
  force: boolean;
  /** `--fast`: skip the exact-load precision fix. See YUE2_SHEET_DEFAULTS. */
  fast: boolean;
  /** Carried for the runner's own log lines and the run record. */
  datasetSlug: string;
}

export function buildYue2SheetArgs(o: ResolvedYue2SheetOptions): string[] {
  const args = [
    'yue2-sheet',
    '--manifest', o.manifest,
    '--models', config.aceServer.models,
  ];
  if (o.only) args.push('--only', o.only);
  if (o.force) args.push('--force');
  if (o.fast) args.push('--fast');
  // Deliberately NOT emitted: --model, --threads. `--models` lets the
  // engine's quant ranking pick (and it prints which file it chose, same
  // reasoning as buildYue2TokenizeArgs skipping --tok); --threads has no
  // server-side opinion, and the engine's own default is fine for an offline
  // cache stage.
  return args;
}
