// lyricsAlignJobs.ts — alignment runs that outlive the request that started them.
//
// A full alignment is a separation pass, several Whisper windows and a CTC
// forward: minutes, not seconds. Holding a request open for it means a refresh
// loses the run and a closed tab kills it. Each run therefore gets an id, is
// polled by the editor, and can be cancelled by name.
//
// It is deliberately not the generation queue. Whisper is CPU and the aligner
// is CPU, so neither may sit in front of a render on the GPU lane.
//
// The manual-timings rule is enforced here, once, and it now gates one door:
// retranscription. An alignment run is a preview that nobody writes until Apply,
// and the pipeline carries hand-checked lines back before it touches the rest,
// so the diff is where a lost placement is seen — `handPlacedLost` — rather than
// a 409 that asks the same question before the answer exists.

import crypto from 'node:crypto';
import fs from 'node:fs';
import { config } from '../config.js';
import { getDb } from '../db/database.js';
import { buildLrc, sidecarPath } from './lrcExport.js';
import { bumpRevision } from './lyricsRevision.js';
import { polishLines, runSongAlignment, type AlignRangeRequest, type PolishPasses } from './lyricsAlignment.js';
import { SerialQueue } from './serialQueue.js';
import { HOP_SEC, fileEnvelope } from './audioActivity.js';
import type { LyricsDiagnostics, LyricsJson, LyricsLine, PolishRepair } from './lyricsReconcile.js';

export type AlignJobStatus = 'running' | 'succeeded' | 'failed' | 'cancelled';

export interface AlignJobResult {
  lineCount: number;
  wordCount: number;
  method: string;
  wordsMoved: number;
  vocalsIsolated: boolean;
  /** A preview was produced and nothing was written. */
  dryRun?: boolean;
  /** True until the diff is accepted; `applyLyricsAlign` is the accept. */
  pending?: boolean;
  cutBack?: number;
  snapped?: number;
  untouched?: number;
  wordsFlagged?: number;
  /** Where the run worked: "the whole track", "line 6", "lines 6-9". Sent so
   *  a re-attached editor can title the review it did not start. */
  where?: string;
  /** The sheet Apply would write, so the editor can show the run's version
   *  beside the one on screen and let the person take it line by line. Built
   *  from the same array the accept writes, never a second rendering of it. */
  proposed?: Array<{
    start: number;
    end: number;
    text: string;
    section?: string;
    words: Array<{
      word: string;
      start: number;
      end: number;
      confidence?: number;
      layer?: 'lead' | 'backing';
    }>;
  }>;
  repairs?: PolishRepair[];
  /** Hand-placed lines the run kept, and the ones it could not match. */
  handPlacedKept?: number;
  handPlacedLost?: number;
}

/**
 * The sheet a preview would write, in the shape the review's "new" pane needs.
 *
 * `confidence` and `layer` travel with each word because a cherry-picked line
 * has to keep its doubt marks and its backing tag, not just its times; losing
 * either would quietly move a harmony line into the lead.
 */
export function proposedLines(lines: LyricsLine[]): NonNullable<AlignJobResult['proposed']> {
  return (Array.isArray(lines) ? lines : []).map(line => {
    const words = Array.isArray(line?.words) ? line.words : [];
    return {
      start: Number(line?.start) || 0,
      end: Number(line?.end) || 0,
      text: typeof line?.text === 'string' && line.text.length > 0
        ? line.text
        : words.map(w => w.word).join(' '),
      ...(line?.section ? { section: line.section } : {}),
      words: words.map(w => {
        const layer = (w as { layer?: 'lead' | 'backing' }).layer;
        return {
          word: w.word,
          start: Number(w.start) || 0,
          end: Number(w.end) || 0,
          ...(typeof w.confidence === 'number' ? { confidence: w.confidence } : {}),
          ...(layer ? { layer } : {}),
        };
      }),
    };
  });
}

export interface AlignJobView {
  jobId: string;
  audioUrl: string;
  status: AlignJobStatus;
  stage: string;
  startedAt: number;
  finishedAt?: number;
  error?: string;
  result?: AlignJobResult;
}

export class AlignRequestError extends Error {
  constructor(message: string, public readonly status = 400) { super(message); }
}

/** A finished run whose diff is waiting for the user to accept or reject it. */
export interface PendingApply {
  /** The sidecar as it would be written, before the revision bump. */
  data: any;
  lines: LyricsLine[];
  jsonPath: string;
  lrcPath: string;
  title: string;
  artist: string;
  /** The revision the preview was computed from, for the stale check. */
  expectedRevision: number;
}

interface AlignJob {
  view: AlignJobView;
  controller: AbortController;
  /** The sidecar this run works on, so the per-track lock can see every job. */
  key: string;
  /** Set when a run finished as a preview; consumed by the first apply. */
  apply?: PendingApply;
}

/**
 * The aligner, one forward at a time: ~5 GB of compute buffer and one engine
 * mutex. Queueing here, rather than letting the engine answer late, is what
 * turns "stuck" into "second in line".
 */
const alignerQueue = new SerialQueue();

const jobs = new Map<string, AlignJob>();
/** The newest job per track, so a reloaded editor can re-attach to one. */
const latest = new Map<string, string>();
/** Finished jobs older than this are dropped on the next start. */
const KEEP_FINISHED_MS = 10 * 60_000;
/** A preview is a decision someone has to come back to, so it lasts longer. */
const KEEP_PREVIEW_MS = 60 * 60_000;

function trackKey(audioUrl: string): string {
  return sidecarPath(config.data.audioDir, audioUrl, '.lyrics.json');
}

function prune(): void {
  const now = Date.now();
  for (const [id, job] of jobs) {
    if (job.view.status === 'running') continue;
    const keep = job.apply ? KEEP_PREVIEW_MS : KEEP_FINISHED_MS;
    if (now - (job.view.finishedAt ?? job.view.startedAt) > keep) {
      jobs.delete(id);
      // The re-attach pointer has to go with the job it points at, or a track
      // that was edited once keeps answering "latest job" with a 404 body.
      if (latest.get(job.key) === id) latest.delete(job.key);
    }
  }
}

/**
 * Whether a machine run may replace this track's timings.
 *
 * The rule is about the file, not about what the caller intends to write: every
 * run is a preview now, so `dryRun` is not a reason to pass. `force` is the one
 * way past it, and the editor asks before using it.
 */
export function refusesOverwrite(existing: any, force: boolean | undefined): boolean {
  return existing?.editedByHand === true && force !== true;
}

/**
 * The run holding this track, if any.
 *
 * Every job is checked, not just the one `latest` points at: `latest` is the
 * re-attach pointer, and a short polish run repoints it while a full run is
 * still going. Reading only `latest` let a second full run start beside the
 * first one, which is the state the per-track lock exists to prevent.
 *
 * Pure over the map so the rule can be tested without a database or a run.
 */
export function findRunning(
  jobs: ReadonlyMap<string, { key: string; view: AlignJobView }>, key: string,
): AlignJobView | null {
  for (const job of jobs.values()) {
    if (job.key === key && job.view.status === 'running') return job.view;
  }
  return null;
}

function runningFor(key: string): AlignJobView | null {
  return findRunning(jobs, key);
}

/**
 * Run work that needs the aligner, in the queue a background run uses.
 *
 * The retranscribe route is inline, but it calls the same engine forward, so it
 * has to take the same slot: two tracks aligning at once do not run in parallel
 * in the engine, they wait on its mutex and one of them burns its 900 s client
 * timeout with no sign of why.
 */
export function runOnAligner<T>(
  task: () => Promise<T>, onQueued?: (position: number) => void,
): Promise<T> {
  return alignerQueue.run(task, onQueued);
}

export interface PolishPreview {
  /** The sidecar as it would be written: repaired lines, new diagnostics. */
  data: any;
  lines: LyricsLine[];
  diagnostics: LyricsDiagnostics;
  result: AlignJobResult;
}

/**
 * The diff a polish run would apply, built without touching the disk.
 *
 * Extracted from the job so the contract can be checked without a database, an
 * engine or audio. The thing it must never do is hand `polishLines` its
 * `dryRun` option: that returns the *unrepaired* lines while still listing the
 * repairs, so the preview would promise work that Apply then does not write.
 */
export function buildPolishPreview(
  existing: any, envelope: number[], passes?: PolishPasses,
): PolishPreview {
  const started = Date.now();
  const wasChecked = countCheckedLines(existing?.lines);
  const result = polishLines(
    existing.lines as LyricsLine[], envelope, HOP_SEC, passes ? { passes } : undefined,
  );
  const keptChecked = countCheckedLines(result.lines);
  const diagnostics: LyricsDiagnostics = {
    // The polish pass does not re-time by model, so it must not relabel the
    // source: an `.lrc`-seeded file repaired here was still not timed by Whisper.
    method: existing.method === 'mms-fa' ? 'mms-fa'
      : existing.method === 'lrc-lines' ? 'lrc-lines' : 'whisper',
    stages: ['polish'],
    model: existing.diagnostics?.model ?? '',
    whisperModel: existing.diagnostics?.whisperModel ?? '',
    windows: existing.diagnostics?.windows ?? 0,
    dtw: existing.diagnostics?.dtw ?? false,
    vocalsIsolated: !!existing.vocalsIsolated,
    words: result.stats.words,
    untouched: result.stats.untouched,
    wordsMoved: result.stats.wordsMoved,
    wordsFlagged: result.stats.wordsFlagged,
    cutBack: result.stats.cutBack,
    snapped: result.stats.snapped,
    anchored: result.stats.anchored,
    repeatedLines: result.stats.repeatedLines,
    repeatWordsMoved: result.stats.repeatWordsMoved,
    monotonicRepairs: result.stats.monotonicRepairs,
    baseline: {
      medianSec: result.stats.medianSec,
      madSec: result.stats.madSec,
      lowSec: result.stats.lowSec,
      highSec: result.stats.highSec,
    },
    sections: result.stats.sections,
    repairs: result.repairs,
    // Carried over, not recomputed: the block describes the CTC pass against
    // Whisper's anchors, and a polish run has neither. Dropping it here would
    // erase the track's audit the first time somebody repaired it.
    ...(existing.diagnostics?.anchorDrift ? { anchorDrift: existing.diagnostics.anchorDrift } : {}),
    ...(result.repairsTruncated ? { repairsTruncated: true } : {}),
    fallback: null,
    elapsedMs: Date.now() - started,
    ...(existing.diagnostics?.range ? { range: existing.diagnostics.range } : {}),
  };
  return {
    data: { ...existing, lines: result.lines, diagnostics, editedByHand: false },
    lines: result.lines,
    diagnostics,
    result: {
      lineCount: result.lines.length,
      wordCount: result.stats.words,
      method: diagnostics.method,
      wordsMoved: result.stats.wordsMoved,
      vocalsIsolated: diagnostics.vocalsIsolated,
      dryRun: true,
      pending: true,
      cutBack: result.stats.cutBack,
      snapped: result.stats.snapped,
      untouched: result.stats.untouched,
      wordsFlagged: result.stats.wordsFlagged,
      repairs: result.repairs,
      handPlacedKept: keptChecked,
      handPlacedLost: Math.max(0, wasChecked - keptChecked),
    },
  };
}

/** Hand-checked lines in a sidecar, for the before/after count a run reports. */
function countCheckedLines(lines: unknown): number {
  return Array.isArray(lines)
    ? (lines as LyricsLine[]).filter(line => line?.checked === true).length
    : 0;
}

export interface StartAlignInput {
  audioUrl: string;
  /** `polish` repairs the current sidecar without Whisper or the aligner. */
  mode: 'full' | 'range' | 'polish';
  /**
   * Kept for the client that always sends it. Every run is a preview now — the
   * diff is held until `applyLyricsAlign` — so this no longer changes a run's
   * behaviour, and in particular it is not a way past the hand-placed guard.
   */
  dryRun?: boolean;
  fromLine?: unknown;
  toLine?: unknown;
  /** The selected span's own times. When the editor and the sidecar disagree
   *  about an index (a line the editor dropped), these pick the lines by time
   *  instead, so a re-align cannot land on the neighbour. */
  fromStart?: unknown;
  toStart?: unknown;
  force?: boolean;
  isolateVocals?: boolean;
  useCtc?: boolean;
  dtw?: boolean;
  /** Which repairs a `polish` run may make. Omitted fields default on. */
  polish?: PolishPasses;
}

/**
 * Validate, claim the track, and return the job immediately. The run itself is
 * started in the background; every failure after this point lands on the job,
 * where the poller reads it.
 */
export function startLyricsAlign(input: StartAlignInput): AlignJobView {
  const audioUrl = input.audioUrl.trim();
  if (!audioUrl) throw new AlignRequestError('audioUrl is required');

  const song = getDb().prepare('SELECT * FROM songs WHERE audio_url = ?').get(audioUrl) as any;
  if (!song) throw new AlignRequestError('Song not found', 404);

  const filename = typeof song.audio_url === 'string' ? String(song.audio_url).split('?')[0].split('/').pop() || '' : '';
  if (!filename) throw new AlignRequestError('This song has no audio file');
  const audioPath = `${config.data.audioDir}/${filename}`;
  if (!fs.existsSync(audioPath)) throw new AlignRequestError('The audio file is missing', 404);

  const key = trackKey(audioUrl);
  const active = runningFor(key);
  if (active) throw new AlignRequestError('An alignment is already running for this track.', 409);

  const jsonPath = sidecarPath(config.data.audioDir, audioUrl, '.lyrics.json');
  const existing = readJson(jsonPath);
  // No hand-placed refusal here. Nothing is written until Apply, and
  // `carryChecked()` restores ticked lines before the statistics pass touches
  // the rest, so the preview is where a placement can be seen to be lost —
  // `handPlacedLost` in its result — and Apply is where it is refused or
  // accepted. Retranscription keeps the 409: it applies itself and rewrites the
  // text that hand lines are matched by.

  let range: AlignRangeRequest | undefined;
  if (input.mode === 'range') {
    if (!existing || !Array.isArray(existing.lines)) {
      throw new AlignRequestError('This track has no word timings to re-align. Run a full alignment first.');
    }
    const nearestIndex = (seconds: number): number => {
      let best = 0;
      let bestDistance = Infinity;
      (existing.lines as any[]).forEach((line, index) => {
        const at = Number(line?.start);
        if (!Number.isFinite(at)) return;
        const distance = Math.abs(at - seconds);
        if (distance < bestDistance) { bestDistance = distance; best = index; }
      });
      return best;
    };
    const fromStart = Number(input.fromStart);
    const toStart = Number(input.toStart);
    const fromLine = Number.isFinite(fromStart) ? nearestIndex(fromStart) : Number(input.fromLine);
    const toLine = Number.isFinite(toStart) ? nearestIndex(toStart) : Number(input.toLine);
    if (!Number.isInteger(fromLine) || !Number.isInteger(toLine) || fromLine < 0 ||
        toLine < fromLine || toLine >= existing.lines.length) {
      throw new AlignRequestError('The selected lines are not in this track’s timings.');
    }
    const first = existing.lines[fromLine];
    const last = existing.lines[toLine];
    if (!Number.isFinite(first?.start) || !Number.isFinite(last?.end) || last.end <= first.start) {
      throw new AlignRequestError('The selected lines have no usable time range.');
    }
    range = { fromLine, toLine, startSec: first.start, endSec: last.end };
  }

  // Same wording the editor uses, computed once here so the job can carry it to
  // a window that did not start the run.
  const where = !range
    ? 'the whole track'
    : range.fromLine === range.toLine
      ? `line ${range.fromLine + 1}`
      : `lines ${range.fromLine + 1}-${range.toLine + 1}`;

  const params = song.generation_params ? safeJson(song.generation_params) : {};
  const lyrics = String(params?.lyrics || song.lyrics || '');
  // A polish run repairs the timings already on disk, so it needs no lyrics.
  if (input.mode !== 'polish' && !lyrics.trim()) {
    throw new AlignRequestError('No source lyrics available for this track');
  }
  if (input.mode === 'polish' && (!existing || !Array.isArray(existing.lines))) {
    throw new AlignRequestError('This track has no timings to repair.');
  }

  const jobId = crypto.randomUUID();
  const controller = new AbortController();
  const view: AlignJobView = {
    jobId,
    audioUrl,
    status: 'running',
    stage: input.mode === 'range' ? 'queued-selection' : 'queued',
    startedAt: Date.now(),
  };
  const job: AlignJob = { view, controller, key };
  jobs.set(jobId, job);
  latest.set(key, jobId);
  prune();

  void (async () => {
    try {
      if (input.mode === 'polish') {
        view.stage = 'polish';
        const preview = buildPolishPreview(existing, fileEnvelope(audioPath), input.polish);
        // Preview, not write. The diff is the point: the user accepts it in the
        // editor, and only then does anything reach the disk.
        const expectedRevision = Number(existing.revision) || 0;
        job.apply = {
          data: preview.data,
          lines: preview.lines,
          jsonPath,
          lrcPath: sidecarPath(config.data.audioDir, audioUrl, '.lrc'),
          title: song.title ?? '',
          artist: typeof params?.artist === 'string' ? params.artist : '',
          expectedRevision,
        };
        view.result = { ...preview.result, where, proposed: proposedLines(preview.lines) };
        view.status = 'succeeded';
        view.stage = 'done';
        view.finishedAt = Date.now();
        console.log(`[LyricsAlign] ${audioUrl}: polish preview, ` +
          `${preview.result.wordsMoved} moved of ${preview.result.wordCount} words`);
        return;
      }

      const outcome = await runOnAligner(
        () => {
          // A run cancelled while it was queued never takes the slot.
          if (controller.signal.aborted) throw new Error('Cancelled');
          return runSongAlignment({
            audioPath,
            lyrics,
            existing: existing as LyricsJson | null,
            ...(range ? { range } : {}),
            // Undefined stays undefined: that is the pipeline's "reuse a cached
            // stem if there is one, otherwise align the mix" default.
            isolateVocals: input.isolateVocals,
            useCtc: input.useCtc !== false,
            dtw: input.dtw !== false,
            signal: controller.signal,
            onProgress: stage => { view.stage = stage; },
          });
        },
        // Position 2 is "next", so it reads as a queue rather than a stall.
        position => { view.stage = `queued-aligner-${position}`; },
      );
      if (controller.signal.aborted) {
        view.status = 'cancelled';
        view.finishedAt = Date.now();
        return;
      }

      const expectedRevision = Number(existing?.revision) || 0;
      const wasChecked = countCheckedLines(existing?.lines);
      const keptChecked = countCheckedLines(outcome.lines);
      const data: any = { ...(existing ?? {}), version: 1 };
      data.method = outcome.diagnostics.method;
      data.whisperModel = outcome.diagnostics.whisperModel;
      data.vocalsIsolated = outcome.vocalsIsolated;
      data.diagnostics = outcome.diagnostics;
      data.lines = outcome.lines;
      data.editedByHand = false;
      // Preview, not write: `POST /lyrics/align/:jobId/apply` is the accept.
      job.apply = {
        data,
        lines: outcome.lines,
        jsonPath,
        lrcPath: sidecarPath(config.data.audioDir, audioUrl, '.lrc'),
        title: song.title ?? '',
        artist: typeof params?.artist === 'string' ? params.artist : '',
        expectedRevision,
      };

      view.status = 'succeeded';
      view.stage = 'done';
      view.finishedAt = Date.now();
      view.result = {
        lineCount: outcome.lines.length,
        wordCount: outcome.lines.reduce((n, line) => n + line.words.length, 0),
        method: outcome.diagnostics.method,
        wordsMoved: outcome.diagnostics.wordsMoved,
        vocalsIsolated: outcome.vocalsIsolated,
        dryRun: true,
        pending: true,
        cutBack: outcome.diagnostics.cutBack,
        snapped: outcome.diagnostics.snapped,
        untouched: outcome.diagnostics.untouched,
        wordsFlagged: outcome.diagnostics.wordsFlagged,
        where,
        proposed: proposedLines(outcome.lines),
        repairs: outcome.diagnostics.repairs,
        handPlacedKept: keptChecked,
        handPlacedLost: Math.max(0, wasChecked - keptChecked),
      };
      console.log(`[LyricsAlign] ${audioUrl}: ${view.result.lineCount} lines, ` +
        `${view.result.wordCount} words, ${outcome.diagnostics.method}` +
        `${outcome.diagnostics.fallback ? ` (fallback: ${outcome.diagnostics.fallback})` : ''}`);
    } catch (err: any) {
      if (controller.signal.aborted) {
        view.status = 'cancelled';
        view.stage = 'cancelled';
      } else {
        view.status = 'failed';
        view.stage = 'failed';
        view.error = err?.message || 'The alignment failed';
        console.error(`[LyricsAlign] ${audioUrl}: ${view.error}`);
      }
      view.finishedAt = Date.now();
    }
  })();

  return { ...view };
}

/** One job by id, or the newest job for a track when `audioUrl` is given. */
export function getLyricsAlignJob(jobId: string): AlignJobView | null {
  const job = jobs.get(jobId);
  return job ? { ...job.view } : null;
}

export function latestLyricsAlignJob(audioUrl: string): AlignJobView | null {
  const id = latest.get(trackKey(audioUrl.trim()));
  if (!id) return null;
  const job = jobs.get(id);
  return job ? { ...job.view } : null;
}

/**
 * Write a sidecar and its `.lrc`, as the one implementation of "commit timings".
 *
 * `applyLyricsAlign` is the reviewed path and the retranscribe route is the
 * confirmed one, but both files have to change together: the revision moves so
 * an open editor's next save is refused, `editedByHand` clears (these times are
 * not hand-placed), and the `.lrc` is rebuilt because the player prefers it and
 * it is what users export. Three copies of that drifting apart is how a route
 * hands out an `.lrc` that disagrees with the timeline on screen.
 */
export function writeTimingsFile(pending: PendingApply): {
  revision: number; lines: number; words: number;
} {
  const data = pending.data;
  data.lines = pending.lines;
  // A machine pass that carried hand-placed lines forward has not replaced
  // them, so the file is still partly hand-timed and the guard must keep
  // asking before the next run overwrites those words.
  data.editedByHand = pending.lines.some(line =>
    line.checked === true || line.words.some(word => word.checked === true));
  bumpRevision(data);
  fs.writeFileSync(pending.jsonPath, JSON.stringify(data, null, 2));
  fs.writeFileSync(pending.lrcPath, buildLrc(pending.lines, {
    title: pending.title,
    artist: pending.artist,
  }));
  return {
    revision: Number(data.revision) || 0,
    lines: pending.lines.length,
    words: pending.lines.reduce((n, line) => n + line.words.length, 0),
  };
}

/**
 * Accept the diff a run produced. This is the only path that writes a preview,
 * and it refuses when the file moved on since the diff was computed: applying an
 * old preview over a newer one is how a hand placement gets lost.
 */
export function applyLyricsAlign(jobId: string, opts: { force?: boolean } = {}): {
  revision: number; lines: number; words: number;
} {
  const job = jobs.get(jobId);
  if (!job) throw new AlignRequestError('That alignment preview is no longer available. Run it again.', 404);
  const pending = job.apply;
  if (!pending) throw new AlignRequestError('This run has no changes left to apply.', 409);

  const current = readJson(pending.jsonPath);
  const revision = Number(current?.revision) || 0;
  if (revision !== pending.expectedRevision && opts.force !== true) {
    throw new AlignRequestError(
      'The timings changed while you were reviewing. Discard this preview and align again, or apply it anyway.',
      409,
    );
  }

  // Consumed here rather than after the write: a second apply must not run the
  // revision bump twice over the same file.
  job.apply = undefined;
  const written = writeTimingsFile(pending);
  if (job.view.result) {
    job.view.result = {
      ...job.view.result,
      pending: false,
      lineCount: written.lines,
      wordCount: written.words,
    };
  }
  job.view.finishedAt = Date.now();
  console.log(`[LyricsAlign] ${job.view.audioUrl}: applied ${written.lines} lines, ${written.words} words`);
  return written;
}

/** Abort a run. Finished jobs answer false rather than pretending to stop. */
export function cancelLyricsAlign(jobId: string): boolean {
  const job = jobs.get(jobId);
  if (!job || job.view.status !== 'running') return false;
  job.controller.abort();
  return true;
}

function readJson(file: string): any {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf-8'));
  } catch {
    return null;
  }
}

function safeJson(value: string): any {
  try { return JSON.parse(value); } catch { return {}; }
}
