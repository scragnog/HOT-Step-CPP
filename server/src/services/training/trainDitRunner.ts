// training/trainDitRunner.ts — runs one `ace-train train-dit` job
//
// Structural clone of trainLmRunner.ts: engine stop → spawn → readline over
// stdout JSONL → relay into the SSE union → engine restart in a `finally`.
// Success, failure, cancel and timeout all restore the engine.
//
// The DiT trainer owns the whole card even harder than the LM one does: it
// builds a ~15 GB F32 weight mirror and then auto-fits its crop length to the
// free VRAM it measures AFTER the engine is gone (§3.7), so `stopEngine`
// defaults to true and the budget is hours. The adapter is exported EVERY epoch
// before the target-loss test (§3.5.3), so a cancelled, timed-out or crashed
// run still leaves a usable PEFT directory.
//
// Spec: docs/plans/2026-07-28-dit-trainer-implementation.md §2.8, §4.4

import fs from 'fs';
import { buildGpuEnv } from '../gpuDevices.js';
import path from 'path';
import readline from 'readline';
import { spawn } from 'child_process';
import { config } from '../../config.js';
import { pushLog } from '../../routes/logs.js';
import { restartAceServer, stopAceServer } from '../aceEngineProcess.js';
import { aceTrainExe, buildTrainDitArgs, type ResolvedTrainDitOptions } from './aceTrain.js';
import { getDataset } from './datasetsRepo.js';
import { refreshPresetsForNewRun } from './lyricStudioExport.js';
import {
  emitJob, emitProgress, finishJob, isCancelled, killJobChild, pushEvent, type TrainingJob,
  trainerLine,
} from './labelingQueue.js';

/**
 * 12 h floor, 5 s per song-epoch (§4.4).
 *
 * The arithmetic says the floor is the only part that ever binds: 17 songs ×
 * 100 epochs at the measured ~230 ms/step is ~11 min of compute. The floor is
 * insurance against a pathological auto-fit (a long crop on a small card walks
 * the depth ladder and can land far slower than the model predicts), not a
 * calibrated budget.
 */
function trainDitTimeoutMs(sampleCount: number, epochs: number): number {
  return Math.max(12 * 60 * 60 * 1000, Math.max(0, sampleCount) * Math.max(1, epochs) * 5 * 1000);
}

function log(job: TrainingJob, level: 'info' | 'warn' | 'error', message: string): void {
  pushEvent(job, { type: 'log', level, message, ts: Date.now() });
}

function int(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) ? Math.trunc(n) : 0;
}

function flt(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function text(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** Only forward numbers the engine actually sent — a metric field defaulted to
 *  0 is indistinguishable from a real 0 in the loss curve. */
function optNum(ev: Record<string, unknown>, key: string): number | undefined {
  const v = ev[key];
  if (v === undefined || v === null) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

/** Cross-event state the relay accumulates for the close handler. */
interface RelayState {
  /** Message from a `fatal` line — the reason the user must see instead of
   *  "ace-train exited with code 1". */
  fatalMessage: string;
  /** A `done` line arrived, i.e. the run reached its own clean end. */
  doneSeen: boolean;
}

function mb(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1);
}

/**
 * ace-train train-dit JSONL → the TrainingStreamEvent union (§2.8).
 *
 * Defensive by contract: every field access guarded, unknown `type` ignored,
 * every `metric` event stamped with `ts: Date.now()`. `step` deliberately emits
 * no log line — one per optimizer step would flood the log pane.
 *
 * `sample` is never emitted by a train-dit job (§2.8): the run's inputs are
 * cached tensor files, not dataset rows.
 */
function relay(job: TrainingJob, ev: Record<string, unknown>, state: RelayState): void {
  switch (ev.type) {
    case 'start': {
      job.done = 0;
      job.failed = 0;
      job.currentSampleId = null;
      job.engineQueueDepth = 0;
      // §2.1: a lokr `start` event carries lokrDim/lokrAlpha/lokrFactor instead
      // of rank/alpha (the contract lets those be absent or zero for lokr).
      const adapterType = text(ev.adapterType) || 'lora';
      const adapterDesc = adapterType === 'lokr' ? `dim${int(ev.lokrDim)}` : `r${int(ev.rank)}`;
      log(job, 'info',
        `Training DiT ${adapterType} ${adapterDesc} → ${text(ev.out)}`);
      break;
    }
    case 'stage': {
      if (ev.state !== 'begin') break;
      const stage = text(ev.stage);
      if (stage) job.phase = stage;
      const total = int(ev.total);
      if (total > 0) job.total = total;
      job.done = 0;
      emitProgress(job);
      log(job, 'info', `Stage: ${stage || 'unknown'}`);
      break;
    }
    case 'model': {
      job.phase = 'loading-models';
      log(job, 'info', `Loaded DiT (${int(ev.layers)} layers, ${int(ev.ms)} ms)`);
      break;
    }
    case 'vram': {
      pushEvent(job, {
        type: 'metric',
        metric: 'vram',
        ts: Date.now(),
        freeMb: optNum(ev, 'freeMb'),
        totalMb: optNum(ev, 'totalMb'),
        estMb: optNum(ev, 'estMb'),
        crop: optNum(ev, 'crop'),
        layers: optNum(ev, 'layers'),
      });
      log(job, 'info',
        `VRAM ${int(ev.freeMb)} MB free — crop ${int(ev.crop)} frames, ${int(ev.layers)} layers`);
      // The auto-fit walking the depth ladder is the single most consequential
      // thing that can happen silently: a top-K adapter trains fine and its
      // musical usefulness is unvalidated (D11 / R3). A user-chosen `--layers`
      // is their own decision and gets no warning.
      if (ev.layersSource === 'auto' && int(ev.layers) > 0 && int(ev.layers) < 32) {
        log(job, 'warn',
          `Auto-fit reduced trained depth to ${int(ev.layers)} layers to fit the available VRAM`
          + ' — adapter quality at partial depth is unvalidated');
      }
      break;
    }
    case 'data': {
      pushEvent(job, {
        type: 'metric',
        metric: 'data',
        ts: Date.now(),
        samples: optNum(ev, 'samples'),
        stepsPerEpoch: optNum(ev, 'stepsPerEpoch'),
        totalSteps: optNum(ev, 'totalSteps'),
      });
      log(job, 'info',
        `${int(ev.samples)} songs, ${int(ev.stepsPerEpoch)} steps/epoch, ${int(ev.totalSteps)} total steps`);
      if (int(ev.skipped) > 0) {
        log(job, 'warn', `${int(ev.skipped)} song(s) skipped — unreadable or missing a required tensor`);
      }
      if (ev.channelStats === false) {
        log(job, 'info', 'No channel_stats.json in this variant — channel balancing is off');
      }
      break;
    }
    case 'step': {
      // Metric only — no log line, a per-optimizer-step log would flood the pane.
      pushEvent(job, {
        type: 'metric',
        metric: 'step',
        ts: Date.now(),
        epoch: optNum(ev, 'epoch'),
        step: optNum(ev, 'step'),
        totalSteps: optNum(ev, 'totalSteps'),
        loss: optNum(ev, 'loss'),
        rawLoss: optNum(ev, 'rawLoss'),
        lr: optNum(ev, 'lr'),
        gradNorm: optNum(ev, 'gradNorm'),
        clipScale: optNum(ev, 'clipScale'),
        t: optNum(ev, 't'),
        crop: optNum(ev, 'crop'),
        ms: optNum(ev, 'ms'),
        vramMb: optNum(ev, 'vramMb'),
      });
      break;
    }
    case 'epoch': {
      pushEvent(job, {
        type: 'metric',
        metric: 'epoch',
        ts: Date.now(),
        epoch: optNum(ev, 'epoch'),
        epochs: optNum(ev, 'epochs'),
        loss: optNum(ev, 'loss'),
        ma5: optNum(ev, 'ma5'),
        lr: optNum(ev, 'lr'),
        gradNorm: optNum(ev, 'gradNorm'),
        etaMs: optNum(ev, 'etaMs'),
        ms: optNum(ev, 'ms'),
        best: ev.best === true ? true : undefined,
      });
      log(job, 'info',
        `Epoch ${int(ev.epoch)}/${int(ev.epochs)} loss ${flt(ev.loss).toFixed(6)}`
        + ` (MA5 ${flt(ev.ma5).toFixed(6)}) lr ${flt(ev.lr).toExponential(3)}`);
      break;
    }
    case 'milestone': {
      pushEvent(job, {
        type: 'metric',
        metric: 'milestone',
        ts: Date.now(),
        loss: optNum(ev, 'loss'),
        epoch: optNum(ev, 'epoch'),
        path: text(ev.path) || undefined,
      });
      log(job, 'info', `Milestone: loss ${flt(ev.loss).toFixed(1)} at epoch ${int(ev.epoch)} → ${text(ev.path)}`);
      break;
    }
    case 'progress': {
      job.done = int(ev.completed);
      const total = int(ev.total);
      if (total > 0) job.total = total;
      const phase = text(ev.phase);
      if (phase) job.phase = phase;
      // train-dit's §2.2 progress line carries no `failed`, but the guard is
      // kept from the shared shape: relaying it when present costs nothing and
      // its absence must not zero a count some other line set.
      if (ev.failed !== undefined) job.failed = int(ev.failed);
      emitProgress(job);
      break;
    }
    case 'target_stop': {
      log(job, 'info',
        `Target loss reached (MA5 ${flt(ev.ma5).toFixed(6)} ≤ ${flt(ev.targetLoss)}) at epoch ${int(ev.epoch)} — stopping`);
      break;
    }
    case 'export': {
      log(job, 'info',
        `Wrote ${int(ev.tensors)} tensors (${mb(int(ev.bytes))} MB) → ${text(ev.path)}`);
      break;
    }
    case 'log': {
      const level = ev.level === 'warn' || ev.level === 'error' ? ev.level : 'info';
      log(job, level, text(ev.message));
      break;
    }
    case 'fatal': {
      // Recorded, not thrown from here — the close handler turns it into the
      // job's failure message so the user sees "needs 15882 MB, 8100 free", not
      // "exit 1".
      const reason = text(ev.reason) || 'unknown';
      state.fatalMessage = text(ev.message) || `ace-train failed (${reason})`;
      job.error = state.fatalMessage;
      log(job, 'error', state.fatalMessage);
      break;
    }
    case 'done': {
      // Recorded only — finishJob() emits the terminal status after the engine
      // restart completes, so the UI never sees "done" with the engine still down.
      state.doneSeen = true;
      log(job, 'info',
        `ace-train finished — ${int(ev.epochsRun)} epoch(s), final loss ${flt(ev.finalLoss).toFixed(6)} (${int(ev.ms)} ms)`);
      break;
    }
    default:
      break;   // unknown type — ignore, never fatal
  }
}

/** How many songs the run will see — used only to size the timeout budget. */
function countTensorFiles(dir: string): number {
  try {
    let n = 0;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.isFile() && entry.name.toLowerCase().endsWith('.safetensors')) n++;
    }
    return n;
  } catch {
    return 0;
  }
}

/** Names that look like a quantized GGUF (Q4_K_M, q8_0, IQ3_XXS, …). D20: a
 *  quantized layer cannot carry an activation gradient, so the engine refuses
 *  such a base at mirror time — after the engine has already been stopped. */
const QUANTIZED_NAME_RE = /(^|[^a-z])(i?q\d)/i;

export async function runTrainDitJob(job: TrainingJob): Promise<void> {
  if (isCancelled(job)) return;

  const opts = job.opts as ResolvedTrainDitOptions;

  try {
    const ds = getDataset(job.datasetId);
    if (!ds) { finishJob(job, 'failed', 'Dataset not found'); return; }

    job.status = 'running';
    job.startedAt = Date.now();
    job.phase = opts.stages[0] ?? 'train';
    emitJob(job);

    const exe = aceTrainExe();
    if (!exe) throw new Error('ace-train was not found next to ace-server — rebuild the engine');

    // ── 1. output dir ────────────────────────────────────────────────────
    // Created up front so a permission problem fails before the engine is
    // stopped rather than after.
    fs.mkdirSync(path.dirname(opts.adapterDir), { recursive: true });
    fs.mkdirSync(opts.adapterDir, { recursive: true });

    const songCount = countTensorFiles(opts.tensorsDir);
    if (songCount > 0) job.total = songCount;
    emitProgress(job);

    // D20 guard: BF16/F16/F32 bases only. Tested POSITIVELY for a quantization
    // token rather than negatively for "bf16" — unlike the LM sizes, DiT bases
    // here carry free-form names (…-turbo-xl-thirds-BF16, scragvae, …) and a
    // legitimate F32 base would otherwise warn on every run.
    if (opts.ditModel && QUANTIZED_NAME_RE.test(opts.ditModel)) {
      log(job, 'warn',
        `${opts.ditModel} looks like a quantized base — the DiT trainer needs BF16, F16 or F32`);
    }

    const args = buildTrainDitArgs({ opts, modelsDir: config.aceServer.models });

    // ── 2. engine handoff ────────────────────────────────────────────────
    let stopped = false;
    let engineExited = true;
    // NOT gated on getAceProcess() being non-null: a crashed engine leaves a
    // respawn scheduled 3 s out, and stopAceServer() is what cancels it. Skipping
    // the call would let that timer spawn an engine into the middle of the job
    // and compete for VRAM — which for DiT training is worse than for anything
    // else, because ace-train solves its crop length and trained depth against
    // the free VRAM it measures while the card is empty (§3.7). With no live
    // child the stop is a cheap no-op.
    if (opts.stopEngine) {
      job.phase = 'engine-stop';
      emitProgress(job);
      log(job, 'info', 'Stopping the engine to free VRAM…');
      engineExited = await stopAceServer('Paused for DiT training');
      stopped = true;
    }

    try {
      // Checked INSIDE the try so the `finally` still restores the engine.
      if (!engineExited) {
        throw new Error('The engine did not shut down — DiT training needs its VRAM. Restart the app and try again.');
      }
      if (isCancelled(job)) return;

      // ── 3. spawn + relay ───────────────────────────────────────────────
      job.phase = 'loading-models';
      emitProgress(job);
      pushLog(`[Training] train-dit job ${job.id}: ace-train ${opts.adapterType} r${opts.rank} → ${opts.adapterDir}`);

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

      const state: RelayState = { fatalMessage: '', doneSeen: false };
      const rl = readline.createInterface({ input: child.stdout! });
      rl.on('line', (line) => {
        trainerLine(job, line);
        try {
          const ev = JSON.parse(line) as Record<string, unknown>;
          if (ev && typeof ev === 'object') relay(job, ev, state);
        } catch { /* non-JSON noise */ }
      });

      const timeoutMs = trainDitTimeoutMs(songCount, opts.epochs);
      let timedOut = false;
      const killer = setTimeout(() => {
        timedOut = true;
        console.error(`[Training] train-dit job ${job.id}: timed out — killing ace-train`);
        log(job, 'error', `Training exceeded its ${Math.round(timeoutMs / 60000)} min budget — stopping.`);
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
        // which would otherwise surface as "ace-train exited with code null".
        // Every finished epoch is already exported (§3.5.3), so the partial
        // adapter survives this.
        throw new Error(
          `DiT training timed out after ${Math.round(timeoutMs / 60000)} min and was stopped` +
          (stderrTail.length ? `: ${stderrTail.slice(-3).join(' | ')}` : ''));
      }
      if (code !== 0) {
        throw new Error(state.fatalMessage
          || `ace-train exited with code ${code === null ? 'null (killed)' : code}: ${stderrTail.slice(-5).join(' | ')}`);
      }
      // A structured `fatal` with a zero exit code should not happen, but if it
      // does the reason is the truth and the exit code is the bug.
      if (state.fatalMessage) throw new Error(state.fatalMessage);

      // ── 4. post-check ──────────────────────────────────────────────────
      // LoRA: adapter_config.json is as load-bearing as the weights — without it
      // the engine silently degrades the alpha/rank scale to 1.0 with only a
      // warning (D19), so "wrote the tensors but not the config" is a failure,
      // not a partial success.
      // LoKR: the exporter writes lokr_weights.safetensors and deliberately NO
      // adapter_config.json (that is PEFT-only; alpha rides per-module tensors +
      // __metadata__.lokr_config — lokr-dit-training plan §2.4). Demanding the
      // PEFT pair here failed every successful LoKR job.
      if (opts.stages.includes('export')) {
        const wrote = opts.adapterType === 'lokr'
          ? fs.existsSync(path.join(opts.adapterDir, 'lokr_weights.safetensors'))
          : fs.existsSync(path.join(opts.adapterDir, 'adapter_model.safetensors'))
            && fs.existsSync(path.join(opts.adapterDir, 'adapter_config.json'));
        if (!wrote) {
          throw new Error('ace-train finished but wrote no adapter');
        }
      }
    } finally {
      // ── 5. ALWAYS restore the engine — success, failure, cancel, timeout ─
      if (stopped) {
        job.phase = 'engine-restart';
        emitProgress(job);
        log(job, 'info', 'Restarting the engine…');
        // On a cancel the SSE stream is already closed (cancelJob emits the
        // terminal status and endStream()s synchronously), so everything from
        // here on is written to a job nobody is listening to. Mirror it into the
        // app log or the up-to-90 s restart happens completely invisibly.
        pushLog(`[Training] train-dit job ${job.id}: restarting the engine…`);
        const back = await restartAceServer();
        if (!back) {
          log(job, 'warn',
            'Engine did not answer /health within 90 s — restart the app if generation fails');
          pushLog(`[Training] train-dit job ${job.id}: engine did not answer /health within 90 s`);
        } else {
          pushLog(`[Training] train-dit job ${job.id}: engine is back up`);
        }
      }
    }

    // Album presets follow the newest run automatically (Rob, 2026-07-29) —
    // UNLESS calibration is queued: it has the final say on which adapter
    // serves (calibrateRunner PRESET RULE), so a fresh-but-worse run must not
    // capture the preset first.
    if (!(opts.calibrate && opts.stages.includes('export'))) {
      const presetsTouched = refreshPresetsForNewRun(opts.adapterDir, 'dit');
      if (presetsTouched) {
        pushLog(`[Training] train-dit job ${job.id}: ${presetsTouched} album preset(s) updated to the new run`);
      }
    }

    // Post-training DiT calibration — OPT-IN since 2026-08-13 (it was default
    // ON from 2026-08-11). Note the consequence in the branch above: with
    // calibration off the new run takes this artist's album presets outright,
    // under the older newest-run-wins rule, with no beat-the-previous guard.
    // Same GPU lane, runs after this job's finally has restarted the engine it
    // needs.
    if (opts.calibrate && opts.stages.includes('export')) {
      const { startDitCalibrateJob } = await import('./labelingQueue.js');
      const cal = startDitCalibrateJob(job.datasetId, {
        newRunDir: opts.adapterDir,
        oldRunDir: opts.initAdapter || '',
        variantKey: opts.variantKey,
        repoint: opts.calibrateRepoint,
      });
      pushLog(`[Training] train-dit job ${job.id}: calibration queued as job ${cal.id}`);
    }

    pushLog(`[Training] train-dit job ${job.id} finished — adapter at ${opts.adapterDir}`);
    finishJob(job, 'done');
  } catch (err: any) {
    if (isCancelled(job)) return;
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[Training] train-dit job ${job.id} FAILED — ${message}`);
    finishJob(job, 'failed', message);
  }
}
