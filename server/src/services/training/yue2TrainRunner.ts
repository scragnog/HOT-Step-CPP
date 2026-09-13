// training/yue2TrainRunner.ts — runs one `ace-train yue2-preprocess` or
// `ace-train yue2-nar-train` job.
//
// Structurally mm3TrainRunner.ts, with one substitution that shapes the whole
// file: THE YuE2 TRAINERS EMIT NO JSONL. `mm3-lm-train` is launched with
// `--jsonl` and speaks a structured event vocabulary; `yue2-nar-train` and
// `yue2-preprocess` write human lines to STDERR and nothing to stdout
// (11 §7.1). So the relay here is a set of regexes over the child's stderr,
// anchored on the exact printf formats in engine/src/train/yue2-nar-train-run.h
// and yue2-preprocess-run.h.
//
// It emits the SAME TrainingStreamEvent vocabulary the MM3 relay does — metric
// step / milestone / data, progress, log — so the Monitor's loss chart and the
// store's applyStreamEvent need no YuE2 case. When the engine eventually grows
// a `--jsonl` mode, only parseTrainLine/parsePreprocessLine go away.
//
// PARSING A HUMAN LOG IS A CONTRACT WITH A printf. Each regex below names the
// line it mirrors, and every one is written to fail SOFT: an unmatched line is
// still recorded verbatim in the console log and in the stderr tail, so a
// format change costs the chart and never the run. What it must never do is
// match the wrong line and report a number that is not there — hence the
// anchored `^\[yue2-...\]` prefixes rather than loose field scraping.
//
// Both kinds are GPU-lane and both stop the engine. Training peaks at 19.6 GB
// at rank 256 and preprocess holds a 3.7 GB encode buffer on top of the VAE —
// neither is MM3's 31.7 GB, but both spawn ace-train, which owns the card.
//
// Phase 5 of docs/plans/yue2/08-nar-lora-trainer.md.

import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { spawn } from 'child_process';

import { pushLog } from '../../routes/logs.js';
import { buildGpuEnv } from '../gpuDevices.js';
import { restartAceServer, stopAceServer } from '../aceEngineProcess.js';
import { YUE2_LICENSE_NOTICE } from '../backends/yue2/index.js';
import { aceTrainExe } from './aceTrain.js';
import { getDataset } from './datasetsRepo.js';
import {
  buildYue2PreprocessArgs, buildYue2TrainArgs, missingYue2TrainModels,
  readYue2PreprocessSummary, YUE2_ADAPTER_STEM, YUE2_VRAM_MODEL,
  type ResolvedYue2PreprocessOptions, type ResolvedYue2TrainOptions,
} from './yue2Train.js';
import { writeYue2RunManifest } from './yue2Runs.js';
import {
  emitJob, emitProgress, finishJob, isCancelled, killJobChild, pushEvent, type TrainingJob,
} from './labelingQueue.js';

function log(job: TrainingJob, level: 'info' | 'warn' | 'error', message: string): void {
  pushEvent(job, { type: 'log', level, message, ts: Date.now() });
}

function numOr(v: string | undefined, d?: number): number | undefined {
  if (v === undefined) return d;
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

// ── stderr -> events ────────────────────────────────────────────────────────

interface RelayState {
  fatalMessage: string;
  doneSeen: boolean;
  lastStep: number;
  totalSteps: number;
  lastLoss?: number;
  /** The newest snapshot this run exported, absolute. */
  lastCkpt?: string;
  /** The final `<stem>.safetensors`, once the export line has been seen. */
  exported?: string;
  /** Counters off the preprocess `done:` line. */
  encoded?: number;
  cached?: number;
  skipped?: number;
  failed?: number;
  clips?: number;
  /** Written into the run's own train-log.jsonl by the caller. */
  onJsonl?: (ev: Record<string, unknown>) => void;
}

/** `[yue2-train] step %5lld/%lld  loss %.5f (win %.5f, run %.5f)  |g| %.4f
 *   lr %.2e  t %.3f  %.2fs/it  vram %zu/%zu MB (%s)  [%s]`
 *  — yue2-nar-train-run.h:2464. The `%5lld` pads with spaces, so the step
 *  number is matched after `\s+` rather than immediately. */
const RE_STEP = /^\[yue2-train\]\s+step\s+(\d+)\/(\d+)\s+loss\s+(\S+)\s+\(win\s+(\S+?),\s+run\s+([^)]+)\)\s+\|g\|\s+(\S+)\s+lr\s+(\S+)\s+t\s+(\S+)\s+([\d.]+)s\/it\s+vram\s+(\d+)\/(\d+)\s+MB/;
/** `[yue2-train] checkpoint at step %lld -> %s (+ snapshot)` — :2494. The path
 *  it prints is the RESUME STATE, not the snapshot, so the snapshot path is
 *  rebuilt from the run directory and the step instead of scraped. */
const RE_CKPT = /^\[yue2-train\]\s+checkpoint at step\s+(\d+)\s+->/;
/** `[yue2-train] exported %zu tensors -> %s` — :2099. Fires for snapshots too;
 *  the FINAL export is the one whose path has no `_step<N>` suffix. */
const RE_EXPORT = /^\[yue2-train\]\s+exported\s+(\d+)\s+tensors\s+->\s+(.+)$/;
/** `[yue2-train] done: %lld steps, mean loss %.5f, AR-prefix cache …` — :2506. */
const RE_DONE = /^\[yue2-train\]\s+done:\s+(\d+)\s+steps,\s+mean loss\s+([\d.eE+-]+)/;
/** `[yue2-train] %zu clip(s) of %lld frames (%.1f s at %.3g fps), %lld dropped as short` — :2208. */
const RE_CLIPS = /^\[yue2-train\]\s+(\d+)\s+clip\(s\)\s+of\s+(\d+)\s+frames/;
/** `[yue2-train] resumed from %s at step %lld/%lld (AdamW iter %d)` — :2280. */
const RE_RESUMED = /^\[yue2-train\]\s+resumed from\s+.*?\s+at step\s+(\d+)\/(\d+)/;
/** `[yue2-train] %zu LoRA tensors over %d NAR layers, rank %lld, alpha %.1f,
 *   target %s` — yue2-nar-train-run.h:513. The trainer's OWN count of what it
 *  is training, which is the one worth showing: the server's
 *  yue2TargetTensorCount() is an estimate from a hard-coded block count and
 *  this line is the model's answer. */
const RE_ADAPTERS = /^\[yue2-train\]\s+(\d+)\s+LoRA tensors over\s+(\d+)\s+NAR layers,\s+rank\s+(\d+),\s+alpha\s+(\S+?),\s+target\s+(\S+)/;

/** `[yue2-preprocess] %zu/%zu <encoded|cached|SKIP|FAIL> …` — the per-file
 *  lines at yue2-preprocess-run.h:744, :776, :791, :800, :829, :843, :861,
 *  :875 and :884. One regex covers all four verbs because they share the
 *  `i/n` prefix, which is the only part the progress bar needs. */
const RE_PP_FILE = /^\[yue2-preprocess\]\s+(\d+)\/(\d+)\s+(encoded|cached|SKIP|FAIL)\b\s*(.*)$/;
/** `[yue2-preprocess] %zu source file(s) in %s; VAE …` — :721. */
const RE_PP_HEAD = /^\[yue2-preprocess\]\s+(\d+)\s+source file\(s\)\s+in\s+/;
/** `[yue2-preprocess] done: %zu sources (%zu encoded, %zu cached, %zu skipped,
 *   %zu failed), %lld clips x %.1f s, …` — :1005. */
const RE_PP_DONE = /^\[yue2-preprocess\]\s+done:\s+(\d+)\s+sources\s+\((\d+)\s+encoded,\s+(\d+)\s+cached,\s+(\d+)\s+skipped,\s+(\d+)\s+failed\),\s+(\d+)\s+clips/;

/** A line the engine prints on its way out. Used ONLY to give the user that
 *  sentence instead of "exited with code 1" — never on its own to decide that
 *  a run failed. THE EXIT CODE IS THE AUTHORITY here, unlike MM3 where a
 *  `fatal` JSONL event is an explicit statement by the trainer: these tools
 *  print prose, and a heuristic that could fail a run which exited 0 would be
 *  strictly worse than no heuristic.
 *
 *  The alternatives are anchored on openings the engine actually uses
 *  (yue2-preprocess-run.h:542-691, yue2-nar-train-run.h:2142-2297) rather
 *  than on a loose "contains the word failed", which would fire on the
 *  perfectly ordinary "… 1 failed), 240 clips …" summary line. */
function fatalish(line: string): string {
  const m = /^\[yue2-(?:train|preprocess|fd)\]\s+(REFUSING:.+|cannot .+|no yue2-.+|every clip .+|nothing usable:.+|--\S+ .*(?:required|must be|needs) .+|VAE (?:load|config) .+|.+ not found:.+)$/.exec(line);
  return m ? m[1] : '';
}

function relayTrainLine(job: TrainingJob, line: string, st: RelayState, outDir: string): void {
  let m: RegExpExecArray | null;

  if ((m = RE_STEP.exec(line))) {
    const step = Number(m[1]);
    const total = Number(m[2]);
    const loss = numOr(m[3]);
    const runMean = numOr(m[5]);
    const gradNorm = numOr(m[6]);
    const lr = numOr(m[7]);
    const stepMs = (numOr(m[9]) ?? 0) * 1000;
    const usedMb = numOr(m[10]);
    const totalMb = numOr(m[11]);
    st.lastStep = Math.max(st.lastStep, step);
    st.totalSteps = total || st.totalSteps;
    st.lastLoss = loss ?? st.lastLoss;
    // Same shape the MM3 relay pushes, so the chart needs no YuE2 case. No log
    // line: one per logged step would flood the pane and the chart reads the
    // metric.
    pushEvent(job, {
      type: 'metric', metric: 'step', ts: Date.now(),
      step, totalSteps: total, loss, lr, gradNorm, stepMs,
      // The trainer's `run` mean is the whole-run average, which is the
      // steadier line to watch; `ma5` is the field the chart already draws a
      // moving average in for the DiT trainer.
      ma5: runMean,
      usedMb, totalMb,
    });
    st.onJsonl?.({ type: 'step', ts: Date.now(), step, totalSteps: total, loss, lr, gradNorm, stepMs });
    // The progress bar is driven from the same line: there is no separate
    // `progress` event to key off, unlike MM3.
    job.done = step;
    job.total = total || job.total;
    job.phase = 'training';
    emitProgress(job);
    return;
  }

  if ((m = RE_CKPT.exec(line))) {
    const step = Number(m[1]);
    const snap = path.join(outDir, `${YUE2_ADAPTER_STEM}_step${step}.safetensors`);
    st.lastCkpt = snap;
    pushEvent(job, {
      type: 'metric', metric: 'milestone', ts: Date.now(),
      step, loss: st.lastLoss, path: snap,
    });
    st.onJsonl?.({ type: 'milestone', ts: Date.now(), step, loss: st.lastLoss, path: snap });
    log(job, 'info', `Snapshot at step ${step}`
      + (st.lastLoss !== undefined ? ` (loss ${st.lastLoss.toFixed(4)})` : ''));
    return;
  }

  if ((m = RE_EXPORT.exec(line))) {
    const file = m[2].trim();
    const isFinal = path.basename(file) === `${YUE2_ADAPTER_STEM}.safetensors`;
    if (isFinal) st.exported = file;
    log(job, 'info', `Exported ${m[1]} tensors → ${path.basename(file)}`);
    st.onJsonl?.({ type: 'export', ts: Date.now(), tensors: Number(m[1]), path: file, final: isFinal });
    return;
  }

  if ((m = RE_DONE.exec(line))) {
    st.doneSeen = true;
    st.lastStep = Math.max(st.lastStep, Number(m[1]));
    const mean = numOr(m[2]);
    log(job, 'info', `Training finished: ${m[1]} steps, mean loss ${m[2]}`);
    st.onJsonl?.({ type: 'done', ts: Date.now(), steps: Number(m[1]), meanLoss: mean });
    return;
  }

  if ((m = RE_CLIPS.exec(line))) {
    const clips = Number(m[1]);
    st.clips = clips;
    pushEvent(job, {
      type: 'metric', metric: 'data', ts: Date.now(),
      samples: clips, totalSteps: st.totalSteps || undefined,
    });
    log(job, 'info', `${clips} clip(s) of ${m[2]} frames in the manifest`);
    st.onJsonl?.({ type: 'init', ts: Date.now(), samples: clips, clipFrames: Number(m[2]) });
    return;
  }

  if ((m = RE_ADAPTERS.exec(line))) {
    pushEvent(job, {
      type: 'metric', metric: 'data', ts: Date.now(),
      loraParams: Number(m[1]), layers: Number(m[2]),
    });
    log(job, 'info',
      `${m[1]} LoRA tensors over ${m[2]} NAR layers, rank ${m[3]}, alpha ${m[4]}, target ${m[5]}.`);
    st.onJsonl?.({
      type: 'adapters', ts: Date.now(), tensors: Number(m[1]), layers: Number(m[2]),
      rank: Number(m[3]), alpha: Number(m[4]), target: m[5],
    });
    return;
  }

  if ((m = RE_RESUMED.exec(line))) {
    st.lastStep = Math.max(st.lastStep, Number(m[1]));
    st.totalSteps = Number(m[2]) || st.totalSteps;
    log(job, 'info', `Resumed at step ${m[1]}/${m[2]} with optimizer state intact.`);
    return;
  }

  // Everything else the trainer says about the recipe, the conditioning or a
  // warning is worth showing verbatim — it is written for a human already.
  if (/^\[yue2-train\]\s+(lr |WARNING|NOTE)/.test(line)) {
    log(job, line.includes('WARNING') ? 'warn' : 'info', line.replace(/^\[yue2-train\]\s+/, ''));
    return;
  }
  const bad = fatalish(line);
  if (bad) st.fatalMessage = st.fatalMessage || bad;
}

function relayPreprocessLine(job: TrainingJob, line: string, st: RelayState): void {
  let m: RegExpExecArray | null;

  if ((m = RE_PP_HEAD.exec(line))) {
    job.total = Number(m[1]) || job.total;
    job.phase = 'encoding';
    emitProgress(job);
    log(job, 'info', line.replace(/^\[yue2-preprocess\]\s+/, ''));
    return;
  }

  if ((m = RE_PP_FILE.exec(line))) {
    const i = Number(m[1]);
    const n = Number(m[2]);
    const verb = m[3];
    job.done = i;
    job.total = n || job.total;
    job.phase = 'encoding';
    if (verb === 'FAIL') job.failed = (job.failed || 0) + 1;
    emitProgress(job);
    // SKIP and FAIL are the two the user has to know about — a track shorter
    // than one clip, or one that needs ffmpeg and did not get it, simply is
    // not in the training set. The successful lines are left to the console
    // log; one per track would be noise on a 15-track album.
    if (verb === 'SKIP' || verb === 'FAIL') {
      log(job, verb === 'FAIL' ? 'error' : 'warn', `${verb}: ${m[4].trim()}`);
    }
    return;
  }

  if ((m = RE_PP_DONE.exec(line))) {
    st.doneSeen = true;
    st.encoded = Number(m[2]);
    st.cached = Number(m[3]);
    st.skipped = Number(m[4]);
    st.failed = Number(m[5]);
    st.clips = Number(m[6]);
    // Reported in full rather than as "done": partial success IS success here
    // (the engine writes a manifest and returns 0), so the counts are the only
    // thing that says whether the training set is what the user expected.
    log(job, 'info',
      `${m[1]} source(s): ${st.encoded} encoded, ${st.cached} already cached, ${st.skipped} skipped, `
      + `${st.failed} failed — ${st.clips} clips.`);
    return;
  }

  const bad = fatalish(line);
  if (bad) st.fatalMessage = st.fatalMessage || bad;
}

// ── spawn ───────────────────────────────────────────────────────────────────

/** Open the run's log files. The directory comes out of `--out` in the argv
 *  rather than from a parameter, so both job kinds get it for free. A log that
 *  cannot be opened is never a reason to fail a run, so every failure here is
 *  swallowed. Append mode: a resumed run re-enters this function. */
function openRunLog(args: string[], jsonl: boolean): {
  jsonl: fs.WriteStream | null; console: fs.WriteStream | null;
} {
  const i = args.indexOf('--out');
  const dir = i >= 0 ? args[i + 1] : '';
  if (!dir) return { jsonl: null, console: null };
  try {
    fs.mkdirSync(dir, { recursive: true });
    return {
      jsonl: jsonl ? fs.createWriteStream(path.join(dir, 'train-log.jsonl'), { flags: 'a' }) : null,
      console: fs.createWriteStream(path.join(dir, 'train-console.log'), { flags: 'a' }),
    };
  } catch {
    return { jsonl: null, console: null };
  }
}

type Yue2Kind = 'yue2-preprocess' | 'yue2-nar-train';

/** Shared spawn + relay + engine restore. Mirrors runMm3AceTrain, including
 *  the two orderings that file learned the hard way: the runner marks the job
 *  running ITSELF (enqueue does not, and without it the UI shows "Queued…" for
 *  the whole run and every elapsed/ETA derived from startedAt never starts),
 *  and `timedOut` is checked BEFORE the exit-code branch so a killed child does
 *  not surface as "exited with code null". */
async function runYue2AceTrain(
  job: TrainingJob,
  kind: Yue2Kind,
  args: string[],
  timeoutMs: number,
  verifyOutput: () => string | null,
  onLine: (line: string, st: RelayState) => void,
  st: RelayState,
): Promise<RelayState> {
  const exe = aceTrainExe();
  if (!exe) {
    finishJob(job, 'failed', 'ace-train is not in this build — rebuild the engine');
    return st;
  }

  job.status = 'running';
  job.startedAt = Date.now();
  job.phase = 'engine-stop';
  emitJob(job);
  emitProgress(job);
  log(job, 'info', 'Stopping the engine to free VRAM…');
  // Not gated on a live child: a crashed engine leaves a respawn scheduled and
  // stopAceServer is what cancels it.
  const engineExited = await stopAceServer(`Paused for ${kind}`);

  try {
    if (!engineExited) {
      throw new Error('The engine did not shut down — YuE2 training needs its VRAM. '
        + 'Restart the app and try again.');
    }
    if (isCancelled(job)) return st;

    job.phase = 'loading-models';
    emitProgress(job);
    pushLog(`[Training] ${kind} job ${job.id}: ${exe} ${args[0]}`);

    const child = spawn(exe, args, { windowsHide: true, env: buildGpuEnv().env });
    job.child = child;

    const sinks = openRunLog(args, kind === 'yue2-nar-train');
    const record = (stream: fs.WriteStream | null, line: string): void => {
      if (!stream) return;
      try { stream.write(line + '\n'); } catch { /* a full disk must not kill a run */ }
    };
    st.onJsonl = (ev) => record(sinks.jsonl, JSON.stringify(ev));

    const stderrTail: string[] = [];
    // BOTH streams go through the same parser. The trainers write to stderr
    // today, but nothing in their contract says a future line cannot land on
    // stdout, and a parser that reads only one of the two would silently stop
    // reporting progress rather than fail.
    const handle = (line: string): void => {
      record(sinks.console, line);
      stderrTail.push(line);
      if (stderrTail.length > 30) stderrTail.shift();
      try { onLine(line, st); } catch { /* a bad line must never kill the run */ }
    };
    const rlErr = readline.createInterface({ input: child.stderr! });
    rlErr.on('line', l => { const t = l.trim(); if (t) handle(t); });
    const rlOut = readline.createInterface({ input: child.stdout! });
    rlOut.on('line', l => { const t = l.trim(); if (t) handle(t); });

    let timedOut = false;
    const killer = setTimeout(() => {
      timedOut = true;
      log(job, 'error', `${kind} exceeded its ${Math.round(timeoutMs / 60000)} min budget — stopping.`);
      killJobChild(job);
    }, timeoutMs);

    const code: number | null = await new Promise<number | null>((resolve, reject) => {
      child.on('error', err => reject(new Error(`Failed to launch ace-train: ${err.message}`)));
      child.on('close', (c, signal) => resolve(signal ? null : c));
    }).finally(() => {
      clearTimeout(killer);
      try { rlErr.close(); } catch { /* already closed */ }
      try { rlOut.close(); } catch { /* already closed */ }
      try { sinks.jsonl?.end(); } catch { /* already closed */ }
      try { sinks.console?.end(); } catch { /* already closed */ }
      job.child = undefined;
    });

    if (isCancelled(job)) return st;
    if (timedOut) {
      throw new Error(`${kind} timed out after ${Math.round(timeoutMs / 60000)} min and was stopped`
        + (stderrTail.length ? `: ${stderrTail.slice(-3).join(' | ')}` : ''));
    }
    if (code !== 0) {
      // The child's OWN last lines, not a generic message: these tools report
      // a real sentence ("no yue2-vae-standard-*.gguf under …", "--decode
      // ffmpeg needs --ffmpeg <path>") and throwing it away would leave the
      // user with an exit code.
      const tail = stderrTail.slice(-5).join(' | ');
      throw new Error(st.fatalMessage
        ? `${st.fatalMessage}${tail ? ` (${tail})` : ''}`
        : `ace-train exited with code ${code === null ? 'null (killed)' : code}${tail ? `: ${tail}` : ''}`);
    }
    // NOTE the missing `if (st.fatalMessage) throw` that mm3TrainRunner has
    // here. Its fatalMessage comes from an explicit `{"type":"fatal"}` event;
    // ours is a regex guess over prose, and failing a run that exited 0 on a
    // guess would be worse than losing the sentence. verifyOutput() is what
    // catches a clean exit that produced nothing.
    const problem = verifyOutput();
    if (problem) throw new Error(problem);
  } finally {
    // ALWAYS restore the engine — success, failure, cancel, timeout.
    job.phase = 'engine-restart';
    emitProgress(job);
    log(job, 'info', 'Restarting the engine…');
    pushLog(`[Training] ${kind} job ${job.id}: restarting the engine…`);
    const back = await restartAceServer();
    if (!back) {
      log(job, 'warn', 'Engine did not answer /health within 90 s — restart the app if generation fails');
    }
  }
  return st;
}

// ── yue2-preprocess ─────────────────────────────────────────────────────────

export async function runYue2PreprocessJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as (ResolvedYue2PreprocessOptions & { manifestPath: string }) | undefined;
  if (!opts?.audioDir || !opts.outDir || !opts.manifestPath) {
    finishJob(job, 'failed', 'yue2-preprocess job is missing its audio folder or output path');
    return;
  }
  // Re-checked here as well as at the route: the route's answer can be stale
  // by the time the job reaches the head of the GPU lane.
  const missing = missingYue2TrainModels('preprocess', { vaeVariant: opts.vaeVariant });
  if (missing.length) {
    finishJob(job, 'failed', `YuE2 models are missing: ${missing.join(', ')}`);
    return;
  }

  const args = buildYue2PreprocessArgs(opts);
  // ~4 min for a 12-15 track album, measured. Budget generously per track with
  // a 30 min floor that covers a cold model load on any corpus.
  const ds = getDataset(job.datasetId);
  const songs = Math.max(1, ds?.sampleCount ?? 1);
  const timeoutMs = Math.max(30 * 60 * 1000, songs * 3 * 60 * 1000);

  const st: RelayState = { fatalMessage: '', doneSeen: false, lastStep: 0, totalSteps: 0 };
  try {
    log(job, 'info',
      `Encoding to YuE2 latents at ${opts.clipSeconds} s clips (caption mode: ${opts.captionMode}). `
      + `Peak VRAM about ${(YUE2_VRAM_MODEL.encodeComputeMb / 1024).toFixed(1)} GB.`);
    // The one thing the engine cannot tell the user, because it never sees the
    // dataset: --audio is scanned FLAT and the dataset's exclusions and its
    // recursive flag do not reach it (11 §4.2). Said once, plainly, rather
    // than left to be discovered as a clip count that does not add up.
    log(job, 'warn',
      'yue2-preprocess scans the source folder directly: subfolders are not searched, and rows you '
      + 'excluded in the dataset grid ARE encoded. Compare the source count below with the dataset\'s.');

    await runYue2AceTrain(job, 'yue2-preprocess', args, timeoutMs, () => {
      if (!fs.existsSync(opts.manifestPath)) return 'yue2-preprocess finished but wrote no manifest';
      const s = readYue2PreprocessSummary(opts.manifestPath);
      return s && s.clips > 0 ? null : 'yue2-preprocess wrote a manifest with no clips in it';
    }, (line, state) => relayPreprocessLine(job, line, state), st);

    if (!isCancelled(job)) {
      const s = readYue2PreprocessSummary(opts.manifestPath);
      if (s) {
        log(job, 'info',
          `Latent cache ready: ${s.clips} clips of ${s.clipFrames} frames from ${s.sources} source(s). `
          + 'Re-running is cheap — the latents are cached per source and are clip-length independent.');
      }
      finishJob(job, 'done');
    }
  } catch (err: any) {
    if (!isCancelled(job)) finishJob(job, 'failed', err?.message || String(err));
  }
}

// ── yue2-nar-train ──────────────────────────────────────────────────────────

export async function runYue2TrainJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as ResolvedYue2TrainOptions | undefined;
  if (!opts?.manifest || !opts.outDir) {
    finishJob(job, 'failed', 'yue2-nar-train job is missing its manifest or output path');
    return;
  }
  const missing = missingYue2TrainModels('train', { lmType: opts.lmType });
  if (missing.length) {
    finishJob(job, 'failed', `YuE2 training models are missing: ${missing.join(', ')}`);
    return;
  }
  if (!fs.existsSync(opts.manifest)) {
    finishJob(job, 'failed',
      `The latent cache manifest is gone (${opts.manifest}) — run the YuE2 preprocess stage again`);
    return;
  }

  // Record the recipe BEFORE the first step, not after the last: a run that
  // dies at step 3 is exactly the one someone will want to continue, and a
  // manifest written on success would not be there.
  const ds = getDataset(job.datasetId);
  writeYue2RunManifest(opts, {
    datasetId: job.datasetId,
    datasetSlug: ds?.slug || opts.datasetSlug || '',
    datasetName: opts.datasetName || ds?.name || ds?.slug || '',
    clips: readYue2PreprocessSummary(opts.manifest) ?? undefined,
  });

  const st: RelayState = { fatalMessage: '', doneSeen: false, lastStep: 0, totalSteps: opts.steps };
  job.total = opts.steps;
  job.done = 0;

  try {
    // The derived-weight half of the licence. The backend already shows this
    // on the generation side; a trained adapter is a derivative of CC BY-NC
    // weights and inherits the restriction, so the run that produces one says
    // it too. Imported from the backend module rather than retyped.
    log(job, 'warn', YUE2_LICENSE_NOTICE
      + ' An adapter trained from them is a derivative work and inherits that restriction.');
    if (!opts.trigger) {
      log(job, 'warn',
        'No trigger word. The adapter will have nothing to address it by at generation time, which is '
        + 'the whole mechanism for reaching the trained style.');
    }
    const mins = (opts.steps * YUE2_VRAM_MODEL.secondsPerStep) / 60;
    log(job, 'info',
      `${opts.steps} steps at rank ${opts.rank}/alpha ${opts.alpha}, target ${opts.target} — `
      + `roughly ${mins < 1 ? '<1' : Math.round(mins)} min at the measured 0.065 s/step.`);

    // Budget on the steps still to run, with a 1 h floor so a cold model load
    // plus a short run never trips the killer. 0.065 s/step measured; 0.5 s
    // gives an 8x margin for a card that is sharing or thermally throttled.
    const timeoutMs = Math.max(60 * 60 * 1000, opts.steps * 500);
    const args = buildYue2TrainArgs(opts);

    await runYue2AceTrain(job, 'yue2-nar-train', args, timeoutMs, () => {
      const final = path.join(opts.outDir, `${YUE2_ADAPTER_STEM}.safetensors`);
      if (fs.existsSync(final)) return null;
      // A run can legitimately end without the final export only if it was
      // stopped; reaching here means exit 0, so the file should be there.
      const anySnap = fs.existsSync(opts.outDir)
        && fs.readdirSync(opts.outDir).some(f => f.startsWith(`${YUE2_ADAPTER_STEM}_step`));
      return anySnap
        ? 'yue2-nar-train exited cleanly but never wrote its final adapter (snapshots are on disk)'
        : 'yue2-nar-train finished but wrote no adapter';
    }, (line, state) => relayTrainLine(job, line, state, opts.outDir), st);

    if (!isCancelled(job)) {
      const final = path.join(opts.outDir, `${YUE2_ADAPTER_STEM}.safetensors`);
      log(job, 'info',
        `Adapter written to ${final}. Load it through the YuE2 model picker's adapter field`
        + (opts.trigger ? `, and put "${opts.trigger}" at the start of the style prompt.` : '.'));
      // Worth saying out loud because it is the opposite of MM3's behaviour:
      // the engine deletes its resume state on a clean export, so a finished
      // run cannot be continued past its step count.
      log(job, 'info',
        'The resume state was removed on the clean finish, so this run cannot be extended — start a new '
        + 'one with a higher step count instead. Its snapshots stay where they are.');
      finishJob(job, 'done');
    }
  } catch (err: any) {
    if (!isCancelled(job)) finishJob(job, 'failed', err?.message || String(err));
  }
}
