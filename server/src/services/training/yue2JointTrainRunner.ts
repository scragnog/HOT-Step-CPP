// Explicit AITK YuE2 joint trainer bridge. This is deliberately separate from
// the legacy NAR/AR runners: a missing prepared dataset must fail, never fall
// back to a legacy cache or silently change the requested training method.
import fs from 'fs';
import path from 'path';
import { emitProgress, finishJob, isCancelled, pushEvent, type TrainingJob } from './labelingQueue.js';
import { buildGpuEnv } from '../gpuDevices.js';
import { log, runYue2AceTrain, type RelayState } from './yue2TrainRunner.js';
import { checkpointRecords, listYue2AitkRuns, recordYue2AitkRun } from './yue2AitkRuns.js';
import { renderYue2JointPreview, Yue2PreviewCleanupError } from './yue2JointPreview.js';
import { yue2Unload } from '../backends/yue2/client.js';
import { ensureYue2PreparedDataset } from './yue2AutoPrepare.js';
import { getDataset } from './datasetsRepo.js';
import { refreshYue2PresetsForJointCheckpoint } from './lyricStudioExport.js';

export interface ResolvedYue2JointTrainOptions {
  checkpoint: string;
  dataset: string;
  outDir: string;
  steps: number;
  saveEvery: number;
  seed: number;
  device: string;
  resume?: string;
  datasetSlug?: string;
  spawnEnv?: NodeJS.ProcessEnv;
  preview?: import('./types.js').Yue2JointPreviewOptions;
  alignment?: import('./types.js').Yue2AlignmentOptions;
  pauseAt?: number;
  preparation?: import('./yue2AitkPrepareRunner.js').ResolvedYue2AitkPrepareOptions;
  /** Optimizer: native CUDA AdamW8bit (default) or the shared LmOptim path. */
  optimizer?: 'adamw' | 'prodigy' | 'muon';
  prodigyD0?: number;
  muonLrScale?: number;
  muonNsSteps?: number;
  /** LoRA rank / alpha; engine defaults are 32 / 32.0. Under adapterType
   *  'lokr', rank is unused and alpha is the LoKr alpha (engine default: the
   *  LoKr dim, i.e. scale 1). */
  rank?: number;
  alpha?: number;
  /** 'lora' (default) or 'lokr': kron-factor sites, dim/factor as in the DiT
   *  trainer. Engine defaults dim 32 / factor 8 (the DiT's 512/6 is larger
   *  than a rank-64 LoRA at YuE2's dims). */
  adapterType?: 'lora' | 'lokr';
  lokrDim?: number;
  lokrFactor?: number;
  /** 'loss' trains until the windowed composite loss <= targetLoss; 'kl' until
   *  the windowed AR KL to base >= targetKl. steps is the cap either way. */
  stopMode?: 'steps' | 'loss' | 'kl';
  targetLoss?: number;
  targetKl?: number;
  /** Advanced planner/optimizer knobs. Every one is optional and, when
   *  omitted, the engine's own default applies (lr 1e-4, weight decay 1e-4,
   *  KL 0.2, ABC dropout 0.5, planner scale 1.0). */
  lr?: number;
  weightDecay?: number;
  klWeight?: number;
  abcDropout?: number;
  /** Trigger-only style with this probability. Needs a dataset prepared after
   *  2026-09-20 (trigger-only prefixes); the engine refuses otherwise. */
  captionDropout?: number;
  plannerLrScale?: number;
}

/** Route and native runner share the public stop-mode contract. */
export function parseYue2JointStopMode(value: unknown): 'steps' | 'loss' | 'kl' | null {
  return value === undefined || value === 'steps' ? 'steps' : value === 'loss' ? 'loss' : value === 'kl' ? 'kl' : null;
}

export function buildYue2JointTrainArgs(o: ResolvedYue2JointTrainOptions): string[] {
  const args = [
    'yue2-joint-train', '--checkpoint', o.checkpoint, '--dataset', o.dataset,
    '--output', o.outDir, '--steps', String(o.steps), '--save-every', String(o.saveEvery),
    '--seed', String(o.seed), '--device', o.device,
  ];
  if (o.rank !== undefined) args.push('--rank', String(o.rank));
  if (o.alpha !== undefined) args.push('--alpha', String(o.alpha));
  if (o.adapterType === 'lokr') {
    args.push('--adapter-type', 'lokr');
    if (o.lokrDim !== undefined) args.push('--lokr-dim', String(o.lokrDim));
    if (o.lokrFactor !== undefined) args.push('--lokr-factor', String(o.lokrFactor));
  }
  const optimizer = o.optimizer ?? 'adamw';
  if (optimizer !== 'adamw') {
    args.push('--optimizer', optimizer);
    if (optimizer === 'prodigy' && o.prodigyD0 !== undefined) args.push('--prodigy-d0', String(o.prodigyD0));
    if (optimizer === 'muon') {
      if (o.muonLrScale !== undefined) args.push('--muon-lr-scale', String(o.muonLrScale));
      if (o.muonNsSteps !== undefined) args.push('--muon-ns-steps', String(o.muonNsSteps));
    }
  }
  if (o.stopMode === 'loss' && o.targetLoss !== undefined) args.push('--target-loss', String(o.targetLoss));
  if (o.stopMode === 'kl' && o.targetKl !== undefined) args.push('--target-kl', String(o.targetKl));
  if (o.lr !== undefined) args.push('--lr', String(o.lr));
  if (o.weightDecay !== undefined) args.push('--weight-decay', String(o.weightDecay));
  if (o.klWeight !== undefined) args.push('--kl-weight', String(o.klWeight));
  if (o.abcDropout !== undefined) args.push('--abc-dropout', String(o.abcDropout));
  if (o.captionDropout !== undefined) args.push('--caption-dropout', String(o.captionDropout));
  if (o.plannerLrScale !== undefined) args.push('--planner-lr-scale', String(o.plannerLrScale));
  if (o.resume) args.push('--resume', o.resume);
  if (o.alignment) {
    args.push('--cursor-weight', String(o.alignment.enabled ? o.alignment.cursorWeight : 0));
  }
  if (Number.isInteger(o.pauseAt) && (o.pauseAt as number) > 0) {
    args.push('--pause-at', String(o.pauseAt));
  }
  return args;
}

function preparedManifest(pathname: string): string | null {
  if (!fs.existsSync(pathname) || !fs.statSync(pathname).isFile()) {
    return 'prepared schema1 dataset manifest is missing';
  }
  if (fs.statSync(pathname).size > 16 * 1024 * 1024) return 'prepared schema1 dataset manifest exceeds 16 MiB';
  try {
    const value = JSON.parse(fs.readFileSync(pathname, 'utf8')) as Record<string, unknown>;
    if (value.schema_version !== 1 || value.recipe_version !== 'aitk-yue2-2026-09-16'
      || value.cot !== 'full' || !Array.isArray(value.items)
      || typeof value.base_sha256 !== 'string' || typeof value.source_manifest_sha256 !== 'string') {
      return 'dataset is not a validated joint-training schema1 manifest (run native preparation first)';
    }
  } catch {
    return 'prepared schema1 dataset manifest is not valid JSON';
  }
  return null;
}

function validateOptions(o: ResolvedYue2JointTrainOptions): string | null {
  if (!o.checkpoint || !fs.existsSync(o.checkpoint) || !fs.statSync(o.checkpoint).isFile()) return `raw ConvRot checkpoint is missing or is not a file: ${o.checkpoint || '(empty)'}`;
  const manifestError = preparedManifest(o.dataset);
  if (manifestError) return `${manifestError}: ${o.dataset || '(empty)'}`;
  if (!o.outDir) return 'Joint training requires a new output directory';
  if (fs.existsSync(o.outDir)) return `output directory already exists; choose a new directory: ${o.outDir}`;
  if (!fs.existsSync(path.dirname(o.outDir))) return `parent directory for joint-training output is missing: ${path.dirname(o.outDir)}`;
  if (!Number.isInteger(o.steps) || o.steps < 1) return 'steps must be a positive integer';
  if (!Number.isInteger(o.saveEvery) || o.saveEvery < 1 || o.saveEvery > o.steps) return 'saveEvery must be between 1 and steps';
  if (!Number.isInteger(o.seed) || o.seed < 0) return 'seed must be a non-negative integer';
  if (o.seed > 0xffffffff) return 'seed must fit uint32';
  if (o.steps > 0x7fffffff) return 'steps must fit int32';
  if (!/^CUDA[0-9]+$/i.test(o.device)) return 'device must be an explicit CUDA device such as CUDA0';
  if (o.optimizer && o.optimizer !== 'adamw' && o.optimizer !== 'prodigy' && o.optimizer !== 'muon') return 'optimizer must be adamw, prodigy or muon';
  if (o.rank !== undefined && (!Number.isInteger(o.rank) || o.rank < 1 || o.rank > 65536)) return 'rank must be an integer between 1 and 65536';
  if (o.alpha !== undefined && (!Number.isFinite(o.alpha) || o.alpha <= 0)) return 'alpha must be a positive finite number';
  if (o.adapterType !== undefined && o.adapterType !== 'lora' && o.adapterType !== 'lokr') return 'adapterType must be lora or lokr';
  if (o.lokrDim !== undefined && (!Number.isInteger(o.lokrDim) || o.lokrDim < 1 || o.lokrDim > 65536)) return 'lokrDim must be an integer between 1 and 65536';
  if (o.lokrFactor !== undefined && (!Number.isInteger(o.lokrFactor) || o.lokrFactor < 1 || o.lokrFactor > 65536)) return 'lokrFactor must be an integer between 1 and 65536';
  if (o.prodigyD0 !== undefined && (!Number.isFinite(o.prodigyD0) || o.prodigyD0 <= 0)) return 'prodigyD0 must be a positive finite number';
  if (o.muonLrScale !== undefined && (!Number.isFinite(o.muonLrScale) || o.muonLrScale <= 0)) return 'muonLrScale must be a positive finite number';
  if (o.muonNsSteps !== undefined && (!Number.isInteger(o.muonNsSteps) || o.muonNsSteps < 1 || o.muonNsSteps > 20)) return 'muonNsSteps must be an integer between 1 and 20';
  if (o.stopMode && o.stopMode !== 'steps' && o.stopMode !== 'loss' && o.stopMode !== 'kl') return 'stopMode must be steps, loss or kl';
  if (o.stopMode === 'loss' && (o.targetLoss === undefined || !Number.isFinite(o.targetLoss) || o.targetLoss < 0)) return 'targetLoss must be a non-negative finite number when stopMode is loss';
  if (o.stopMode === 'kl' && (o.targetKl === undefined || !Number.isFinite(o.targetKl) || o.targetKl <= 0)) return 'targetKl must be a positive finite number when stopMode is kl';
  if (o.resume && (!fs.existsSync(o.resume) || !fs.statSync(o.resume).isFile())) return `resume record is missing: ${o.resume}`;
  if (!o.spawnEnv) o.spawnEnv = buildGpuEnv().env;
  return null;
}

function relayJsonLine(job: TrainingJob, line: string, state: RelayState): void {
  const event = parseYue2JointEvent(line, state.totalSteps);
  if (!event) { log(job, 'info', line); return; }
  const raw = JSON.parse(line) as Record<string, unknown>;
  state.onJsonl?.(raw);
  const { stage, step } = event;
  if (stage === 'paused' && step !== undefined) {
    state.pausedAt = step;
    if (typeof raw.resume === 'string') state.pauseResume = raw.resume;
  }
  const opts = job.opts as ResolvedYue2JointTrainOptions | undefined;
  const catalogueStage = stage === 'load' || stage === 'checkpoint' || stage === 'checkpoint_stage' || stage === 'done';
  if (catalogueStage && opts) persistAitkCatalogue(job, opts, 'running');
  if (step !== undefined) state.lastStep = step;
  if (stage === 'joint' && step !== undefined) {
    job.done = step; job.total = state.totalSteps; job.phase = 'training';
    state.lastLoss = event.loss ?? state.lastLoss;
    pushEvent(job, { type: 'metric', metric: 'step', ts: Date.now(), step,
      totalSteps: state.totalSteps, ...(event.loss === undefined ? {} : { loss: event.loss }),
      ...(event.arKl === undefined ? {} : { arKl: event.arKl }),
      ...(event.gradNorm === undefined ? {} : { gradNorm: event.gradNorm }),
      ...(event.stepMs === undefined ? {} : { stepMs: event.stepMs }) });
    emitProgress(job);
    log(job, 'info', `Joint training step ${step}${event.loss === undefined ? '' : ` loss ${event.loss}`}`);
  } else if (stage === 'checkpoint' || stage === 'checkpoint_stage') {
    if (stage === 'checkpoint' && step !== undefined && opts) {
      const saved = checkpointRecords(opts.outDir).find(c => c.step === step && c.arPath && c.narPath);
      if (saved) pushEvent(job, { type: 'metric', metric: 'milestone', ts: Date.now(), step,
        loss: state.lastLoss, path: saved.dir });
    }
    log(job, 'info', `Joint training ${stage}${step === undefined ? '' : ` at step ${step}`}`);
  } else if (stage === 'target' && step !== undefined) {
    state.targetStopped = true;
    log(job, 'info', `Stop target reached at step ${step}; stopping early`);
  } else if (stage !== 'event') {
    if (stage === 'done') state.doneSeen = true;
    job.phase = stage;
    emitProgress(job);
    log(job, 'info', `Joint training ${stage}`);
  }
}

function persistAitkCatalogue(
  job: TrainingJob,
  opts: ResolvedYue2JointTrainOptions,
  status: 'running' | 'done' | 'failed' | 'cancelled',
): void {
  try {
    if (!fs.statSync(opts.outDir).isDirectory()) return;
    const { spawnEnv: _env, resume: _resume, ...persisted } = opts;
    recordYue2AitkRun({
      version: 1, jobId: job.id, datasetId: job.datasetId,
      datasetSlug: opts.datasetSlug ?? '', method: 'aitk', output: opts.outDir,
      options: persisted as Record<string, unknown>, status,
      createdAt: job.startedAt ?? Date.now(), updatedAt: Date.now(),
      ...(job.error ? { error: job.error } : {}), checkpoints: [],
    });
  } catch { /* output may not exist yet; catalogue is advisory */ }
}

/** Pure contract helper kept exportable for server-side event tests. */
export function parseYue2JointEvent(line: string, totalSteps: number): {
  stage: string; step?: number; loss?: number; arKl?: number; gradNorm?: number; stepMs?: number; totalSteps: number;
} | null {
  try {
    const event = JSON.parse(line) as Record<string, unknown>;
    if (!event || typeof event.stage !== 'string') return null;
    const step = typeof event.step === 'number' && Number.isInteger(event.step) && event.step >= 0 ? event.step : undefined;
    const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
    const cursor = finite(event.cursor_ce) && finite(event.cursor_weight)
      ? event.cursor_ce * event.cursor_weight : 0;
    const loss = finite(event.ar_ce) && finite(event.ar_kl) && finite(event.nar_mse)
      ? event.ar_ce + 0.2 * event.ar_kl + event.nar_mse + cursor : undefined;
    const stepMs = finite(event.step_ms) && event.step_ms >= 0 ? event.step_ms : undefined;
    return { stage: event.stage, ...(step === undefined ? {} : { step }),
      ...(loss === undefined ? {} : { loss }),
      ...(finite(event.ar_kl) ? { arKl: event.ar_kl } : {}),
      ...(finite(event.gradient_norm) ? { gradNorm: event.gradient_norm } : {}),
      ...(stepMs === undefined ? {} : { stepMs }), totalSteps };
  } catch { return null; }
}

export async function runYue2JointTrainJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as ResolvedYue2JointTrainOptions | undefined;
  if (isCancelled(job)) return;
  if (opts?.preparation && !opts.resume) {
    try {
      const manifest = await ensureYue2PreparedDataset(job, opts.preparation);
      if (!manifest || isCancelled(job)) return;
      opts.dataset = manifest;
      job.done = 0; job.total = opts.steps; job.phase = 'loading'; emitProgress(job);
    } catch (err) {
      if (!isCancelled(job)) finishJob(job, 'failed', err instanceof Error ? err.message : String(err));
      return;
    }
  }
  const error = opts ? validateOptions(opts) : 'job is missing joint-training options';
  if (error) { finishJob(job, 'failed', error); return; }
  const o = opts!;
  let nativeAttempted = false;
  try {
    log(job, 'info', `Starting YuE2 joint training (${o.device})`);
    nativeAttempted = true;
    const preview = o.preview?.enabled && o.preview.everySteps > 0 ? o.preview : undefined;
    if (preview) fs.mkdirSync(path.join(o.outDir, 'segments'), { recursive: true });
    let resume = o.resume || '';
    const resumeStep = resume ? Number((/checkpoint-step(\d+)/.exec(resume) || [])[1] || 0) : 0;
    let step = resumeStep;
    let segmentNo = 1;
    for (;;) {
      if (isCancelled(job)) return;
      const segmentOut = preview ? path.join(o.outDir, 'segments', `segment-${String(segmentNo).padStart(6, '0')}`) : o.outDir;
      const pauseAt = preview ? Math.min(o.steps, step + preview.everySteps) : 0;
      const segment = { ...o, outDir: segmentOut, resume: resume || undefined, pauseAt: pauseAt < o.steps ? pauseAt : undefined };
      const state: RelayState = { fatalMessage: '', doneSeen: false, lastStep: step, targetStopped: false, totalSteps: o.steps };
      // A target-loss stop ends the run early: the engine checkpoints the
      // last completed step, so the validator must accept that step, not o.steps.
      const wanted = () => state.targetStopped && state.lastStep > step
        ? state.lastStep : (pauseAt > 0 && pauseAt < o.steps ? pauseAt : o.steps);
      nativeAttempted = true;
      await runYue2AceTrain(job, 'yue2-joint-train', buildYue2JointTrainArgs(segment),
        Math.max(30 * 60 * 1000, (o.steps - step) * 10 * 60 * 1000), () => {
          if (!fs.existsSync(segmentOut)) return 'Joint trainer exited without creating its output directory';
          const expect = wanted();
          const checkpoint = path.join(segmentOut, `checkpoint-step${expect}`);
          if (!fs.existsSync(checkpoint)) return `Joint-training checkpoint-step${expect} is missing`;
          if (['adapter.safetensors', 'optimizer.resume', 'native-ar.safetensors', 'native-nar.safetensors']
            .some(name => !fs.existsSync(path.join(checkpoint, name)))) return `Joint-training checkpoint-step${expect} is incomplete`;
          return null;
        }, (line, current) => relayJsonLine(job, line, current), state, o.spawnEnv);
      if (isCancelled(job)) return;
      if (!state.pausedAt || !preview || state.pausedAt >= o.steps) break;
      const ckpt = checkpointRecords(segmentOut).find(c => c.step === state.pausedAt);
      if (!ckpt?.optimizerPath || !ckpt.arPath || !ckpt.narPath) throw new Error(`Joint-training pause at step ${state.pausedAt} has no complete paired checkpoint`);
      job.phase = 'preview'; emitProgress(job);
      try {
        await renderYue2JointPreview({ output: o.outDir, step: state.pausedAt, options: preview, arAdapter: ckpt.arPath, narAdapter: ckpt.narPath, dataset: o.dataset, signal: job.controller.signal });
      } catch (err: any) {
        if (err instanceof Yue2PreviewCleanupError) throw err;
        log(job, 'warn', `Preview at step ${state.pausedAt} failed; training will continue: ${err?.message || err}`);
      }
      if (isCancelled(job)) return;
      // renderYue2JointPreview owns the unload/restore transaction.  Do not
      // issue a second unload here: a failed cleanup is already surfaced as a
      // Yue2PreviewCleanupError and must abort before the next segment.
      resume = ckpt.optimizerPath; step = state.pausedAt; segmentNo++;
    }
    if (preview && !isCancelled(job)) {
      const finalDir = path.join(o.outDir, 'segments', `segment-${String(segmentNo).padStart(6, '0')}`);
      const final = checkpointRecords(finalDir).find(c => c.step === o.steps);
      if (final?.arPath && final.narPath) {
        job.phase = 'preview'; emitProgress(job);
        try { await renderYue2JointPreview({ output: o.outDir, step: o.steps, options: preview, arAdapter: final.arPath, narAdapter: final.narPath, dataset: o.dataset, signal: job.controller.signal }); }
        catch (err: any) {
          if (err instanceof Yue2PreviewCleanupError) throw err;
          log(job, 'warn', `Final preview failed; checkpoint is intact: ${err?.message || err}`);
        }
      }
    }
    /* runYue2AceTrain can fail early and marks the job failed itself. */
    if (!isCancelled(job) && job.status === 'running') {
      // Link only a complete, paired checkpoint after a successful run. This
      // also covers target-loss early stops and preview segmented runs: the
      // scanner walks every segment and the highest completed step wins.
      const final = checkpointRecords(o.outDir).find(c => c.arPath && c.narPath);
      if (final?.arPath && final.narPath) {
        const ds = getDataset(job.datasetId);
        if (ds) {
          const known = listYue2AitkRuns(job.datasetId, ds.slug).flatMap(run => run.checkpoints.flatMap(c => [c.arPath, c.narPath].filter((p): p is string => !!p)));
          refreshYue2PresetsForJointCheckpoint(ds, final.arPath, final.narPath, known);
        }
      }
      finishJob(job, 'done');
    }
    return;
  } catch (err: unknown) {
    if (!isCancelled(job)) finishJob(job, 'failed', err instanceof Error ? err.message : String(err));
  } finally {
    // The output directory is created by ace-train. Record only after that
    // boundary, including failed/cancelled partial runs, and never before it.
    let outputExists = false;
    try { outputExists = nativeAttempted && fs.statSync(o.outDir).isDirectory(); } catch { /* child may have failed before creating output */ }
    if (outputExists) {
      persistAitkCatalogue(job, o,
        job.status === 'done' ? 'done' : job.status === 'cancelled' ? 'cancelled' : 'failed');
    }
  }
}
