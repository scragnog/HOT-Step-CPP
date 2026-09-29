// training/preprocessRunner.ts — runs one `ace-train preprocess` job
//
// Structure mirrors routes/midiStudio.ts's child-process job: spawn, readline
// over stdout JSONL, a stderr ring buffer for the failure message, a timeout
// killer, and `job.child` so a cancel can actually stop the process.
//
// The difference is the VRAM handoff. ace-train loads VAE + text-enc + cond-enc
// (~3.2 GB) and the engine holds far more, so by default ace-server is stopped
// for the duration and restarted in a `finally` — success, failure, cancel and
// timeout all restore it (P32).
//
// Spec: docs/plans/2026-07-27-preprocess-implementation.md §4.5, §2.9

import fs from 'fs';
import { buildGpuEnv } from '../gpuDevices.js';
import path from 'path';
import readline from 'readline';
import { spawn } from 'child_process';
import { config, getFFmpegPath } from '../../config.js';
import { pushLog } from '../../routes/logs.js';
import {
  restartAceServer, stopAceServer,
} from '../aceEngineProcess.js';
import { aceTrainExe, buildPreprocessArgs, type ResolvedPreprocessOptions } from './aceTrain.js';
import { buildSamples } from './datasetScan.js';
import { getDataset } from './datasetsRepo.js';
import {
  emitJob, emitProgress, finishJob, isCancelled, killJobChild, pushEvent, type TrainingJob,
  trainerLine,
} from './labelingQueue.js';
import { buildPreprocessManifest, writePreprocessManifest } from './preprocessManifest.js';

/** 2 h floor, 5 min per song — ace-train is a long-running batch, not a request. */
function preprocessTimeoutMs(sampleCount: number): number {
  return Math.max(2 * 60 * 60 * 1000, sampleCount * 5 * 60 * 1000);
}

function log(job: TrainingJob, level: 'info' | 'warn' | 'error', message: string): void {
  pushEvent(job, { type: 'log', level, message, ts: Date.now() });
}

function int(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) ? Math.trunc(n) : 0;
}

function text(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/**
 * ace-train JSONL → the frozen TrainingStreamEvent union (§2.9). No new event
 * variants: per-song outcomes ride in `log`, counters ride in `progress`.
 * Defensive by contract — unknown `type` values are ignored, never fatal.
 */
function relay(job: TrainingJob, ev: Record<string, unknown>, sampleIdByCacheId: Map<string, string>): void {
  switch (ev.type) {
    case 'start': {
      job.total = int(ev.total) || job.total;
      job.done = 0;
      job.failed = 0;
      job.phase = 'preprocess';
      job.currentSampleId = null;
      job.engineQueueDepth = 0;
      emitProgress(job);
      log(job, 'info', `Preprocessing ${job.total} songs → ${text(ev.out)}`);
      break;
    }
    case 'model': {
      job.phase = 'loading-models';
      log(job, 'info', `Loaded ${text(ev.stage)} (${int(ev.ms)} ms)`);
      break;
    }
    case 'song_start': {
      // §2.9 says `job.currentSampleId = ev.id`, but ev.id is the 8-hex
      // dataset.json cache id while every other job kind — and the UI's
      // samplesById map and row highlight — uses the 16-hex sampleId. Taken
      // literally the "currently working on" indicator can never match, so the
      // id is translated back into the sampleId namespace here. The SSE event
      // shape is unchanged; only the value's namespace is corrected.
      const cacheId = text(ev.id);
      job.currentSampleId = sampleIdByCacheId.get(cacheId) || cacheId || null;
      job.phase = 'preprocess';
      emitProgress(job);
      break;
    }
    case 'song_done': {
      log(job, 'info', `${text(ev.file)} → T=${int(ev.tLatent)} S=${int(ev.sTotal)} (${int(ev.ms)} ms)`);
      break;
    }
    case 'song_skip': {
      log(job, 'info', `${text(ev.file)} — skipped (${text(ev.reason)})`);
      break;
    }
    case 'song_fail': {
      log(job, 'error', `${text(ev.file)} — ${text(ev.error)}`);
      break;
    }
    case 'progress': {
      job.done = int(ev.completed);
      job.failed = int(ev.failed);
      emitProgress(job);
      break;
    }
    case 'log': {
      const level = ev.level === 'warn' || ev.level === 'error' ? ev.level : 'info';
      log(job, level, text(ev.message));
      break;
    }
    case 'stats': {
      job.phase = 'stats';
      log(job, 'info', `Channel stats: ${int(ev.nSamples)} songs, ${int(ev.nFrames)} frames`);
      break;
    }
    case 'done': {
      // Recorded only — finishJob() emits the terminal status after the engine
      // restart completes, so the UI never sees "done" with the engine still down.
      log(job, 'info',
        `ace-train finished — ${int(ev.processed)} processed, ${int(ev.skipped)} skipped, ${int(ev.failed)} failed (${int(ev.ms)} ms)`);
      break;
    }
    default:
      break;   // unknown type — ignore, never fatal
  }
}

/**
 * Fail a preprocess that cached NOTHING, at the point it happens.
 *
 * ace-train exits 0 when every song was *skipped* — skips are per-song and
 * deliberately non-fatal (an all-FLAC dataset with no ffmpeg must not kill the
 * run). But a dataset where nothing at all landed in the cache is not a partial
 * success, it is a failed preprocess, and saying so here is the difference
 * between one clear message and a confusing one a whole stage later: a bulk run
 * lost britneyspears_oops to 14/14 "audio file not found" skips (mojibake in the
 * filenames) and the only error the user ever saw was train-dit's "no cached
 * songs in …\tensors\…" (2026-08-02).
 *
 * The test is `ok`, not `processed`: a resumed run that finds every song already
 * cached reports processed=0 / skipped=N with ok=true on each, and that is a
 * perfectly good cache. Only `ok === false` on every sample means empty.
 */
function assertCachedSomething(metaPath: string): void {
  let meta: any;
  try {
    meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
  } catch {
    return;   // unreadable meta is the existing failure mode; don't invent a new one
  }
  const samples: any[] = Array.isArray(meta?.samples) ? meta.samples : [];
  if (!samples.length) return;
  const cached = samples.filter(s => s?.ok === true).length;
  if (cached > 0) return;

  // Every sample carries the same class of error in practice, so the first one
  // is the diagnosis — quote it rather than making the user open the meta file.
  const reason = samples.find(s => typeof s?.error === 'string' && s.error)?.error;
  throw new Error(
    `preprocess cached 0 of ${samples.length} songs — nothing to train on`
    + (reason ? `. First error: ${reason}` : ''),
  );
}

export async function runPreprocessJob(job: TrainingJob): Promise<void> {
  if (isCancelled(job)) return;

  const opts = job.opts as ResolvedPreprocessOptions;

  try {
    const ds = getDataset(job.datasetId);
    if (!ds) { finishJob(job, 'failed', 'Dataset not found'); return; }

    job.status = 'running';
    job.startedAt = Date.now();
    job.phase = 'preprocess';
    emitJob(job);

    const exe = aceTrainExe();
    if (!exe) throw new Error('ace-train was not found next to ace-server — rebuild the engine');

    // ── 1. manifest ──────────────────────────────────────────────────────
    emitProgress(job);
    const all = await buildSamples(ds, { withDuration: true });
    const wanted = new Set(job.sampleIds);
    const included = all.filter(s =>
      !s.excluded && !s.fileMissing && (wanted.size === 0 || wanted.has(s.sampleId)));
    if (included.length === 0) throw new Error('Dataset has no includable samples');
    job.total = included.length;
    emitProgress(job);

    let datasetJson: unknown = null;
    try {
      if (ds.datasetJsonPath && fs.existsSync(ds.datasetJsonPath)) {
        datasetJson = JSON.parse(fs.readFileSync(ds.datasetJsonPath, 'utf-8'));
      }
    } catch (err: any) {
      // Cache ids fall back to the deterministic relPath hash — a warning, not a stop.
      log(job, 'warn', `Could not read dataset.json (${err.message}) — cache ids fall back to path hashes`);
    }

    const manifest = buildPreprocessManifest(ds, included, datasetJson);
    const manifestPath = writePreprocessManifest(job.id, manifest);

    // 8-hex cache id (what ace-train echoes) → 16-hex sampleId (what the UI
    // keys rows on). buildPreprocessManifest maps `included` 1:1 and in order.
    const sampleIdByCacheId = new Map<string, string>();
    manifest.samples.forEach((m, idx) => {
      const s = included[idx];
      if (s) sampleIdByCacheId.set(m.id, s.sampleId);
    });

    const outDir = opts.outputDir;
    fs.mkdirSync(outDir, { recursive: true });

    const args = buildPreprocessArgs({
      manifestPath,
      outDir,
      modelsDir: config.aceServer.models,
      dit: opts.ditModel,
      vae: opts.vaeModel,
      textEnc: opts.textEncoder,
      opts: {
        maxDuration: opts.maxDuration,
        normalize: opts.normalize,
        targetDb: opts.targetDb,
        dtype: opts.dtype,
        compat: opts.compat,
        maxCaptionTokens: opts.maxCaptionTokens,
        maxLyricTokens: opts.maxLyricTokens,
        vaeChunk: opts.vaeChunk,
        vaeOverlap: opts.vaeOverlap,
        overwrite: opts.overwrite,
      },
      ffmpeg: getFFmpegPath(),
    });

    // ── 2. engine handoff ────────────────────────────────────────────────
    let stopped = false;
    let engineExited = true;
    // NOT gated on getAceProcess() being non-null: a crashed engine leaves a
    // respawn scheduled 3 s out, and stopAceServer() is what cancels it. Skipping
    // the call would let that timer spawn an engine into the middle of the job
    // and compete for VRAM. With no live child the stop is a cheap no-op.
    if (opts.stopEngine) {
      job.phase = 'engine-stop';
      emitProgress(job);
      log(job, 'info', 'Stopping the engine to free VRAM…');
      engineExited = await stopAceServer('Paused for training preprocessing');
      stopped = true;
    }

    try {
      // Checked INSIDE the try so the `finally` still restores the engine.
      // A stop that timed out means the engine is still alive and still holding
      // VRAM: running ace-train alongside it defeats the whole handoff and risks
      // an OOM, and the restart would try to spawn a second engine on :8085.
      if (!engineExited) {
        throw new Error('The engine did not shut down — preprocessing needs its VRAM. Restart the app and try again.');
      }
      if (isCancelled(job)) return;

      // ── 3. spawn + relay ───────────────────────────────────────────────
      job.phase = 'loading-models';
      emitProgress(job);
      pushLog(`[Training] Preprocess job ${job.id}: ace-train ${included.length} songs → ${outDir}`);

      const child = spawn(exe, args, { windowsHide: true, env: buildGpuEnv().env });
      trainerLine(job, `$ ${exe} ${args.join(' ')}`);
      job.child = child;

      const stderrTail: string[] = [];
      child.stderr?.on('data', (buf: Buffer) => {
        for (const raw of buf.toString('utf-8').split(/[\r\n]+/)) {
          const line = raw.trim();
          if (!line) continue;
          trainerLine(job, line);
          stderrTail.push(line);
          if (stderrTail.length > 30) stderrTail.shift();
        }
      });

      const rl = readline.createInterface({ input: child.stdout! });
      rl.on('line', (line) => {
        trainerLine(job, line);
        try {
          const ev = JSON.parse(line) as Record<string, unknown>;
          if (ev && typeof ev === 'object') relay(job, ev, sampleIdByCacheId);
        } catch { /* non-JSON noise */ }
      });

      const timeoutMs = preprocessTimeoutMs(included.length);
      let timedOut = false;
      const killer = setTimeout(() => {
        timedOut = true;
        console.error(`[Training] Preprocess job ${job.id}: timed out — killing ace-train`);
        log(job, 'error', `Preprocessing exceeded its ${Math.round(timeoutMs / 60000)} min budget — stopping.`);
        // Tree kill: ace-train shells out to ffmpeg per song, and a plain
        // TerminateProcess on the parent leaves the transcode running.
        killJobChild(job);
      }, timeoutMs);

      const code: number | null = await new Promise<number | null>((resolve, reject) => {
        child.on('error', err => reject(new Error(`Failed to launch ace-train: ${err.message}`)));
        child.on('close', (c, signal) => resolve(signal ? null : c));
      }).finally(() => {
        clearTimeout(killer);
        try { rl.close(); } catch { /* already closed */ }
        job.child = undefined;
      });

      if (isCancelled(job)) return;
      if (timedOut) {
        // Checked BEFORE the exit-code branch: a killed child yields code null,
        // which would otherwise surface as the bare, unattributable message
        // "ace-train exited with code null".
        throw new Error(
          `Preprocessing timed out after ${Math.round(timeoutMs / 60000)} min and was stopped` +
          (stderrTail.length ? `: ${stderrTail.slice(-3).join(' | ')}` : ''));
      }
      if (code !== 0) {
        throw new Error(`ace-train exited with code ${code === null ? 'null (killed)' : code}: ${stderrTail.slice(-5).join(' | ')}`);
      }
      const metaPath = path.join(outDir, 'preprocess_meta.json');
      if (!fs.existsSync(metaPath)) {
        throw new Error('ace-train finished but wrote no preprocess_meta.json');
      }
      assertCachedSomething(metaPath);
    } finally {
      // ── 4. ALWAYS restore the engine — success, failure, cancel, timeout ─
      if (stopped) {
        job.phase = 'engine-restart';
        emitProgress(job);
        log(job, 'info', 'Restarting the engine…');
        // On a cancel the SSE stream is already closed (cancelJob emits the
        // terminal status and endStream()s synchronously), so everything from
        // here on is written to a job nobody is listening to. Mirror it into the
        // app log or the up-to-90 s restart happens completely invisibly.
        pushLog(`[Training] Preprocess job ${job.id}: restarting the engine…`);
        const back = await restartAceServer();
        if (!back) {
          log(job, 'warn',
            'Engine did not answer /health within 90 s — restart the app if generation fails');
          pushLog(`[Training] Preprocess job ${job.id}: engine did not answer /health within 90 s`);
        } else {
          pushLog(`[Training] Preprocess job ${job.id}: engine is back up`);
        }
      }
    }

    pushLog(`[Training] Preprocess job ${job.id} finished — ${job.done}/${job.total} (${job.failed} failed)`);
    finishJob(job, 'done');
  } catch (err: any) {
    if (isCancelled(job)) return;
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[Training] Preprocess job ${job.id} FAILED — ${message}`);
    finishJob(job, 'failed', message);
  }
}
