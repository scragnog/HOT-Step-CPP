// training/mm3TrainRunner.ts — runs one `ace-train mm3-codes` or
// `ace-train mm3-lm-train` job.
//
// Structural clone of trainLmRunner.ts, which is the point: because the MM3
// binaries emit the SAME JSONL vocabulary as `train-lm` (init / step /
// milestone / progress / export / fatal / done), the relay below is the same
// relay and the Monitor's loss chart works with no new event types.
//
// Both kinds live in the GPU lane (labelingQueue laneFor) and both stop the
// engine first. That is not caution, it is the only configuration that fits:
// an MM3 LM training step peaks at 31.7 GB of a 32 GB card, so anything else
// resident means WDDM shared-memory thrash — measured at 38 s/step for work
// that takes 3.9 s when it fits.
//
// Spec: docs/plans/2026-08-20-mm3-training-server-design.md §2.3.

import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { spawn } from 'child_process';

import { pushLog } from '../../routes/logs.js';
import { restartAceServer, stopAceServer } from '../aceEngineProcess.js';
import { aceTrainExe } from './aceTrain.js';
import { getDataset } from './datasetsRepo.js';
import {
  buildMm3CodesArgs, buildMm3TrainLmArgs, missingMm3TrainModels,
  type Mm3CodesArgs, type ResolvedMm3TrainLmOptions,
} from './mm3Train.js';
import {
  emitJob, emitProgress, finishJob, isCancelled, killJobChild, pushEvent, type TrainingJob,
} from './labelingQueue.js';
import { planMm3Previews, renderMm3Preview, type Mm3PreviewPlan } from './mm3Preview.js';

function log(job: TrainingJob, level: 'info' | 'warn' | 'error', message: string): void {
  pushEvent(job, { type: 'log', level, message, ts: Date.now() });
}

function int(v: unknown): number {
  const n = Number(v);
  return Number.isFinite(n) ? Math.trunc(n) : 0;
}

function optNum(ev: Record<string, unknown>, key: string): number | undefined {
  const v = ev[key];
  if (v === undefined || v === null) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function text(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

interface RelayState {
  fatalMessage: string;
  doneSeen: boolean;
  /** Set when the trainer honoured a PAUSE sentinel and saved its state. The
   *  presence of this is what tells the segment loop "there is more to run". */
  pausedAt?: number;
  pauseState?: string;
  pauseCkpt?: string;
  /** Last checkpoint directory this segment exported, paused or not. */
  lastCkpt?: string;
  lastLoss?: number;
  /** Called on every step event, so the segment loop can decide when to ask
   *  for a pause without parsing the JSONL a second time. */
  onStep?: (step: number, loss: number | undefined) => void;
}

/** ace-train JSONL -> the TrainingStreamEvent union. Deliberately defensive:
 *  every field access guarded, unknown `type` ignored. `step` emits no log line
 *  — one per optimizer step would flood the pane, and the chart reads metrics. */
function relay(job: TrainingJob, ev: Record<string, unknown>, st: RelayState): void {
  switch (text(ev.type)) {
    case 'init': {
      // MM3 trains in shuffled passes, so it has REAL epochs: stepsPerEpoch is
      // the training-song count. Publishing it is what puts the step layer and
      // the epoch line on one x axis (fractional epochs), exactly as ACE does —
      // no MM3 special case in the chart at all.
      const perEpoch = int(ev.stepsPerEpoch);
      const totalSteps = int(ev.totalSteps);
      pushEvent(job, {
        type: 'metric', metric: 'data', ts: Date.now(),
        stepsPerEpoch: perEpoch || undefined,
        epochs: perEpoch > 0 && totalSteps > 0 ? Math.ceil(totalSteps / perEpoch) : undefined,
        samples: optNum(ev, 'samples'),
        totalSteps: totalSteps || undefined,
      });
      const hold = int(ev.holdout);
      log(job, 'info',
        `${int(ev.samples)} training songs${hold ? ` (+${hold} held out)` : ''}, prompt up to `
        + `${int(ev.maxPrompt)} tok, crops to ${int(ev.maxFrames)} frames `
        + `(sequence up to ${int(ev.seqMax)}), rank ${int(ev.rank)}`);
      if (!hold) {
        log(job, 'warn',
          'No held-out songs, so there is no evaluation — the training loss is the only signal, and it '
          + 'cannot tell learning from memorising.');
      }
      break;
    }
    case 'epoch':
      pushEvent(job, {
        type: 'metric', metric: 'epoch', ts: Date.now(),
        epoch: optNum(ev, 'epoch'), loss: optNum(ev, 'loss'), lr: optNum(ev, 'lr'),
        ms: optNum(ev, 'ms'), step: optNum(ev, 'step'),
      });
      break;
    case 'eval':
      // The line that actually answers "is it still learning?".
      pushEvent(job, {
        type: 'metric', metric: 'eval', ts: Date.now(),
        step: optNum(ev, 'step'), loss: optNum(ev, 'loss'), crops: optNum(ev, 'crops'),
      });
      log(job, 'info', `Held-out loss ${(optNum(ev, 'loss') ?? 0).toFixed(4)} at step ${int(ev.step)}`);
      break;
    case 'best':
      log(job, 'info',
        `Best held-out loss ${(optNum(ev, 'loss') ?? 0).toFixed(4)} at step ${int(ev.step)} — `
        + 'start the ear test at that checkpoint.');
      break;
    case 'vram': {
      const usedMb = int(ev.usedMb), totalMb = int(ev.totalMb);
      pushEvent(job, {
        type: 'metric', metric: 'vram', ts: Date.now(),
        usedMb: optNum(ev, 'usedMb'), totalMb: optNum(ev, 'totalMb'), freeMb: optNum(ev, 'freeMb'),
      });
      // The one failure this run can suffer that looks like "it works, slowly":
      // over the card, WDDM pages to host memory and a 4 s step becomes 40.
      // Say it plainly rather than leaving the user to wonder.
      if (totalMb > 0 && usedMb > totalMb - 512) {
        log(job, 'warn',
          `VRAM is at ${usedMb}/${totalMb} MB — this run is on the edge of the card. If steps are far `
          + 'slower than expected, it is spilling into shared memory: lower Crop (frames) or close other '
          + 'GPU users.');
      }
      break;
    }
    case 'optimizer': {
      const muon = int(ev.muon), tensors = int(ev.tensors);
      log(job, 'info',
        `Optimizer ${text(ev.name)}: ${muon}/${tensors} tensors on Muon in ${int(ev.buckets)} buckets`
        + (text(ev.name) === 'muon' ? ` at lr-scale ${optNum(ev, 'lrScale') ?? '?'}` : ''));
      // A Muon run that classified NOTHING onto Muon trains as AdamW and is
      // otherwise indistinguishable. Say so rather than let it pass silently.
      if (text(ev.name) === 'muon' && muon === 0) {
        log(job, 'warn',
          'Muon was selected but no parameter qualified — this run is training as AdamW. '
          + 'Rank is probably below the Muon minimum dimension.');
      }
      break;
    }
    case 'step':
      pushEvent(job, {
        type: 'metric', metric: 'step', ts: Date.now(),
        step: optNum(ev, 'step'), loss: optNum(ev, 'loss'), lr: optNum(ev, 'lr'),
        gradNorm: optNum(ev, 'gradNorm'), clipScale: optNum(ev, 'clipScale'),
        totalSteps: optNum(ev, 'totalSteps'), stepMs: optNum(ev, 'stepMs'),
      });
      st.lastLoss = optNum(ev, 'loss');
      st.onStep?.(int(ev.step), optNum(ev, 'loss'));
      break;
    case 'milestone':
      pushEvent(job, {
        type: 'metric', metric: 'milestone', ts: Date.now(),
        step: optNum(ev, 'step'), loss: optNum(ev, 'loss'), path: text(ev.path) || undefined,
      });
      st.lastCkpt = text(ev.path) || st.lastCkpt;
      log(job, 'info', `Checkpoint at step ${int(ev.step)} (loss ${(optNum(ev, 'loss') ?? 0).toFixed(4)})`);
      break;
    case 'cropAnchor':
      log(job, 'info', text(ev.mode) === 'zero'
        ? 'Crop anchor: zero — every crop is presented as the song\'s opening (the legacy convention).'
        : 'Crop anchor: song — each crop carries its true position in the track.');
      break;
    case 'resumed':
      log(job, 'info',
        `Resumed at step ${int(ev.step)}/${int(ev.totalSteps)} (epoch ${int(ev.epoch)}) with optimizer `
        + 'state intact.');
      break;
    case 'paused':
      // NOT a terminal state — the segment loop relaunches with --resume once
      // the preview has rendered.
      st.pausedAt = int(ev.step);
      st.pauseState = text(ev.state);
      st.pauseCkpt = text(ev.ckpt) || st.lastCkpt;
      st.lastLoss = optNum(ev, 'loss') ?? st.lastLoss;
      break;
    case 'progress': {
      job.done = int(ev.completed);
      const total = int(ev.total);
      if (total > 0) job.total = total;
      const phase = text(ev.phase);
      if (phase) job.phase = phase;
      if (ev.failed !== undefined) job.failed = int(ev.failed);
      emitProgress(job);
      break;
    }
    case 'export':
      log(job, 'info', `Exported ${int(ev.tensors)} tensors → ${text(ev.path)}`);
      break;
    case 'fatal':
      st.fatalMessage = text(ev.message) || 'ace-train reported a fatal error';
      log(job, 'error', st.fatalMessage);
      break;
    case 'done':
      st.doneSeen = true;
      break;
    default:
      break;
  }
}

/** Shared spawn + relay + engine restore. `kind` only shapes the log lines.
 *
 *  Returns the relay state so a caller can see whether the child PAUSED rather
 *  than finished — the engine is back up by then, which is exactly the window a
 *  preview render needs. */
async function runMm3AceTrain(
  job: TrainingJob,
  kind: 'mm3-codes' | 'mm3-lm-train',
  args: string[],
  timeoutMs: number,
  verifyOutput: () => string | null,
  onStep?: (step: number, loss: number | undefined) => void,
): Promise<RelayState> {
  const st: RelayState = { fatalMessage: '', doneSeen: false, onStep };

  const exe = aceTrainExe();
  if (!exe) {
    finishJob(job, 'failed', 'ace-train is not in this build — rebuild the engine');
    return st;
  }

  // `enqueue()` does NOT mark a job running — every runner does it itself
  // (trainLmRunner:301, preprocessRunner:171). Without this the UI shows
  // "Queued…" for the whole run and, because startedAt stays unset, elapsed
  // time and every ETA derived from it never start.
  job.status = 'running';
  job.startedAt = Date.now();
  job.phase = 'engine-stop';
  emitJob(job);
  emitProgress(job);
  log(job, 'info', 'Stopping the engine to free VRAM…');
  // Not gated on a live child: a crashed engine leaves a respawn scheduled, and
  // stopAceServer is what cancels it. An engine spawning into the middle of an
  // MM3 training run would compete for a card that is already at 97 %.
  const engineExited = await stopAceServer(`Paused for ${kind}`);

  try {
    if (!engineExited) {
      throw new Error('The engine did not shut down — MM3 training needs its VRAM. Restart the app and try again.');
    }
    if (isCancelled(job)) return st;

    job.phase = 'loading-models';
    emitProgress(job);
    pushLog(`[Training] ${kind} job ${job.id}: ${exe} ${args.slice(0, 2).join(' ')}`);

    const child = spawn(exe, args, { windowsHide: true });
    job.child = child;

    const stderrTail: string[] = [];
    child.stderr?.on('data', (buf: Buffer) => {
      for (const raw of buf.toString('utf-8').split(/[\r\n]+/)) {
        const line = raw.trim();
        if (!line) continue;
        stderrTail.push(line);
        if (stderrTail.length > 30) stderrTail.shift();
      }
    });

    const rl = readline.createInterface({ input: child.stdout! });
    rl.on('line', (line) => {
      try {
        const ev = JSON.parse(line) as Record<string, unknown>;
        if (ev && typeof ev === 'object') relay(job, ev, st);
      } catch { /* the human log also goes to stdout on some paths — ignore */ }
    });

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
      try { rl.close(); } catch { /* already closed */ }
      job.child = undefined;
    });

    if (isCancelled(job)) return st;
    if (timedOut) {
      // Before the exit-code branch: a killed child yields null, which would
      // otherwise surface as "exited with code null". Checkpoints already
      // written survive this.
      throw new Error(`${kind} timed out after ${Math.round(timeoutMs / 60000)} min and was stopped`
        + (stderrTail.length ? `: ${stderrTail.slice(-3).join(' | ')}` : ''));
    }
    if (code !== 0) {
      throw new Error(st.fatalMessage
        || `ace-train exited with code ${code === null ? 'null (killed)' : code}: ${stderrTail.slice(-5).join(' | ')}`);
    }
    if (st.fatalMessage) throw new Error(st.fatalMessage);

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

// ── mm3-codes ───────────────────────────────────────────────────────────────

export async function runMm3CodesJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as (Mm3CodesArgs & { datasetSlug: string }) | undefined;
  if (!opts?.datasetJson || !opts.outDir) {
    finishJob(job, 'failed', 'mm3-codes job is missing its dataset or output path');
    return;
  }
  const missing = missingMm3TrainModels('codes');
  if (missing.length) {
    finishJob(job, 'failed', `MiniMax-Music3 training models are missing: ${missing.join(', ')}`);
    return;
  }

  const args = buildMm3CodesArgs(opts);
  // Encoding is DAV + a 169M encoder per track: minutes, not hours. The floor
  // covers model load on a cold cache.
  const ds = getDataset(job.datasetId);
  const songs = Math.max(1, ds?.sampleCount ?? 1);
  const timeoutMs = Math.max(30 * 60 * 1000, songs * 60 * 1000);

  try {
    await runMm3AceTrain(job, 'mm3-codes', args, timeoutMs, () => {
      const dir = path.join(opts.outDir, 'codes');
      const n = fs.existsSync(dir) ? fs.readdirSync(dir).filter(f => f.endsWith('.codes')).length : 0;
      return n > 0 ? null : 'mm3-codes finished but wrote no .codes files';
    });
    if (!isCancelled(job)) finishJob(job, 'done');
  } catch (err: any) {
    if (!isCancelled(job)) finishJob(job, 'failed', err?.message || String(err));
  }
}

// ── mm3-lm-train ────────────────────────────────────────────────────────────

// A training run is one job but potentially SEVERAL ace-train processes, one
// per preview point. See mm3Preview.ts for why the trainer has to leave the
// card entirely for a render; the loop below is the other half of that.
//
// Each pass through the loop is a segment:
//
//   stop engine -> train -> PAUSE sentinel -> trainer saves state, exits 0
//     -> restart engine (runMm3AceTrain's finally) -> render previews -> repeat
//
// The engine restart is not extra work bought by previews: runMm3AceTrain
// already stops the engine on the way in and restores it on the way out, so a
// preview lands exactly in the window where the card is free and the engine is
// up. Nothing else in the job model changes — same job id, same SSE stream,
// same loss chart, one continuous step axis.

/** Ask the running trainer to stop at its next step boundary. */
function requestPause(outDir: string): void {
  try {
    fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(path.join(outDir, 'PAUSE'), '');
  } catch { /* the trainer simply runs on; the next trigger tries again */ }
}

/** Clear a sentinel left behind by a killed or crashed run. Without this the
 *  next run would pause at step 1 and never make progress. */
function clearPause(outDir: string): void {
  try { fs.rmSync(path.join(outDir, 'PAUSE'), { force: true }); } catch { /* ignore */ }
}

/** Render every preview in the plan against one checkpoint (or the base model
 *  when `adapterPath` is empty) and push each to the job's stream.
 *
 *  A preview failure NEVER fails the run. The renders are diagnostics; losing
 *  50 minutes of training because a WAV did not come back would be a strictly
 *  worse outcome than not hearing it. */
async function renderPreviewSet(
  job: TrainingJob,
  plan: Mm3PreviewPlan,
  runName: string,
  adapterPath: string,
  step: number,
  totalSteps: number,
  loss?: number,
): Promise<void> {
  for (const spec of plan.specs) {
    if (isCancelled(job)) return;
    const label = adapterPath ? `step ${step}` : 'base model';
    try {
      job.phase = 'preview';
      emitProgress(job);
      const preview = await renderMm3Preview({
        spec, seconds: plan.seconds, seed: plan.seed, adapterPath,
        step, totalSteps, dir: plan.dir, loss,
      });
      pushEvent(job, { type: 'preview', run: runName, preview });
      log(job, 'info',
        `Preview (${spec.kind}, ${label}): ${plan.seconds}s rendered in `
        + `${(preview.ms / 1000).toFixed(1)}s — ${preview.file}`);
    } catch (err: any) {
      log(job, 'warn',
        `Preview (${spec.kind}, ${label}) failed: ${err?.message || String(err)} — training continues.`);
    }
  }
}

export async function runMm3TrainLmJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as ResolvedMm3TrainLmOptions | undefined;
  if (!opts?.manifest || !opts.codesDir || !opts.outDir) {
    finishJob(job, 'failed', 'mm3-lm-train job is missing its manifest, codes or output path');
    return;
  }
  const missing = missingMm3TrainModels('train', opts.basePrecision);
  if (missing.length) {
    finishJob(job, 'failed', `MiniMax-Music3 training models are missing: ${missing.join(', ')}`);
    return;
  }

  const runName = path.basename(opts.outDir);
  const plan = planMm3Previews({
    preview: opts.preview,
    manifest: opts.manifest,
    captionsDir: opts.captionsDir,
    codesDir: opts.codesDir,
    outDir: opts.outDir,
    holdout: opts.holdout,
    trigger: opts.trigger,
    captionFile: opts.captionFile,
  });
  clearPause(opts.outDir);

  // The output check, shared by every segment. saveEvery 0 means "never
  // checkpoint" — a legitimate ask for a probe run, and demanding one anyway
  // failed a run that did exactly what it was told.
  const verifyOutput = () => {
    if (opts.saveEvery <= 0) return null;
    // LoKr writes lokr_weights.safetensors, LoRA writes a PEFT directory. Looking
    // only for the PEFT name marked a perfectly good LoKr run as "wrote no
    // checkpoint" with its checkpoints sitting on disk.
    const found = fs.existsSync(opts.outDir)
      && fs.readdirSync(opts.outDir).some(d =>
        fs.existsSync(path.join(opts.outDir, d, 'adapter_model.safetensors'))
        || fs.existsSync(path.join(opts.outDir, d, 'lokr_weights.safetensors')));
    return found ? null : 'mm3-lm-train finished but wrote no checkpoint';
  };

  try {
    if (plan) {
      const cadence = [
        plan.everySteps > 0 ? `every ${plan.everySteps} steps` : '',
        plan.everyMinutes > 0 ? `every ${plan.everyMinutes} min` : '',
      ].filter(Boolean).join(' or ');
      log(job, 'info',
        `Audio previews ${cadence}: ${plan.specs.map(s => s.kind).join(' + ')}, `
        + `${plan.seconds}s at seed ${plan.seed}. Each one pauses training for roughly a minute `
        + '(the trainer has to leave the card for the render).');

      // The step-0 reference, rendered before the engine is ever stopped. This
      // is the only render in the run that costs nothing extra — the engine is
      // already up — and it is the one the ear needs most: "worse than base" is
      // not a judgement anyone can make without base.
      if (plan.baseline && !isCancelled(job)) {
        await renderPreviewSet(job, plan, runName, '', 0, opts.steps, undefined);
      }
    }

    let resumeFrom = '';
    let lastStep = 0;
    for (;;) {
      if (isCancelled(job)) return;

      // Cadence state, reset per segment because the trainer restarts.
      let nextStepTrigger = plan && plan.everySteps > 0
        ? Math.floor(lastStep / plan.everySteps) * plan.everySteps + plan.everySteps
        : Number.POSITIVE_INFINITY;
      let pauseAsked = false;
      let minuteTimer: NodeJS.Timeout | undefined;

      const askPause = () => {
        if (pauseAsked) return;
        pauseAsked = true;
        requestPause(opts.outDir);
      };
      if (plan && plan.everyMinutes > 0) {
        minuteTimer = setTimeout(askPause, plan.everyMinutes * 60_000);
      }

      const onStep = (step: number, _loss: number | undefined) => {
        lastStep = Math.max(lastStep, step);
        if (!plan) return;
        // Never pause on the last step: the run is about to end and export a
        // checkpoint anyway, and the final preview is rendered after `done`
        // with the engine already back up.
        if (step >= opts.steps) return;
        if (step >= nextStepTrigger) {
          nextStepTrigger += plan.everySteps;
          askPause();
        }
      };

      // Budget the segment on the steps it can still run, with a 1 h floor so a
      // cold model load plus a short segment never trips the killer.
      const remaining = Math.max(1, opts.steps - lastStep);
      const timeoutMs = Math.max(60 * 60 * 1000, remaining * 20 * 1000);

      const args = buildMm3TrainLmArgs({ ...opts, resumeFrom });
      let st: RelayState;
      try {
        st = await runMm3AceTrain(job, 'mm3-lm-train', args, timeoutMs, verifyOutput, onStep);
      } finally {
        if (minuteTimer) clearTimeout(minuteTimer);
      }
      if (isCancelled(job)) return;

      if (st.pausedAt && st.pauseState && plan) {
        // A pause is mid-run, so `progress` has not reached total and the UI
        // would otherwise show a stalled bar while the render happens.
        log(job, 'info', `Paused at step ${st.pausedAt}/${opts.steps} to render a preview…`);
        clearPause(opts.outDir);
        await renderPreviewSet(job, plan, runName, st.pauseCkpt || '', st.pausedAt, opts.steps,
                               st.lastLoss);
        if (isCancelled(job)) return;
        resumeFrom = st.pauseState;
        lastStep = st.pausedAt;
        continue;
      }

      // Not paused = the run is over. Render the final checkpoint now, while
      // the engine is up, so the last thing in the strip is the finished
      // adapter rather than the last preview point before it.
      if (plan && st.lastCkpt && !isCancelled(job)) {
        await renderPreviewSet(job, plan, runName, st.lastCkpt, lastStep || opts.steps, opts.steps,
                               st.lastLoss);
      }
      break;
    }

    if (!isCancelled(job)) {
      log(job, 'info',
        'Checkpoints are in the MM3 adapter folder — they appear in the adapter picker with no install step.');
      finishJob(job, 'done');
    }
  } catch (err: any) {
    if (!isCancelled(job)) finishJob(job, 'failed', err?.message || String(err));
  } finally {
    clearPause(opts.outDir);
  }
}
