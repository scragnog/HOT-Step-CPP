// Explicit AITK YuE2 joint trainer bridge. This is deliberately separate from
// the legacy NAR/AR runners: a missing prepared dataset must fail, never fall
// back to a legacy cache or silently change the requested training method.
import fs from 'fs';
import path from 'path';
import { emitProgress, finishJob, isCancelled, pushEvent, type TrainingJob } from './labelingQueue.js';
import { buildGpuEnv } from '../gpuDevices.js';
import { log, runYue2AceTrain, type RelayState } from './yue2TrainRunner.js';
import { recordYue2AitkRun } from './yue2AitkRuns.js';

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
}

export function buildYue2JointTrainArgs(o: ResolvedYue2JointTrainOptions): string[] {
  const args = [
    'yue2-joint-train', '--checkpoint', o.checkpoint, '--dataset', o.dataset,
    '--output', o.outDir, '--steps', String(o.steps), '--save-every', String(o.saveEvery),
    '--seed', String(o.seed), '--device', o.device,
  ];
  if (o.resume) args.push('--resume', o.resume);
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
      return 'dataset is not a validated AITK schema1 manifest (run native preparation first)';
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
  if (!o.outDir) return 'AITK joint training requires a new output directory';
  if (fs.existsSync(o.outDir)) return `output directory already exists; choose a new directory: ${o.outDir}`;
  if (!fs.existsSync(path.dirname(o.outDir))) return `parent directory for AITK output is missing: ${path.dirname(o.outDir)}`;
  if (!Number.isInteger(o.steps) || o.steps < 1) return 'steps must be a positive integer';
  if (!Number.isInteger(o.saveEvery) || o.saveEvery < 1 || o.saveEvery > o.steps) return 'saveEvery must be between 1 and steps';
  if (!Number.isInteger(o.seed) || o.seed < 0) return 'seed must be a non-negative integer';
  if (o.seed > 0xffffffff) return 'seed must fit uint32';
  if (o.steps > 0x7fffffff) return 'steps must fit int32';
  if (!/^CUDA[0-9]+$/i.test(o.device)) return 'device must be an explicit CUDA device such as CUDA0';
  if (o.resume && (!fs.existsSync(o.resume) || !fs.statSync(o.resume).isFile())) return `resume record is missing: ${o.resume}`;
  if (!o.spawnEnv) o.spawnEnv = buildGpuEnv().env;
  return null;
}

function relayJsonLine(job: TrainingJob, line: string, state: RelayState): void {
  const event = parseYue2JointEvent(line, state.totalSteps);
  if (!event) { log(job, 'info', line); return; }
  state.onJsonl?.(JSON.parse(line) as Record<string, unknown>);
  const { stage, step } = event;
  const opts = job.opts as ResolvedYue2JointTrainOptions | undefined;
  const catalogueStage = stage === 'load' || stage === 'checkpoint' || stage === 'checkpoint_stage' || stage === 'done';
  if (catalogueStage && opts) persistAitkCatalogue(job, opts, 'running');
  if (step !== undefined) state.lastStep = step;
  if (stage === 'joint' && step !== undefined) {
    job.done = step; job.total = state.totalSteps; job.phase = 'training';
    pushEvent(job, { type: 'metric', metric: 'step', ts: Date.now(), step,
      totalSteps: state.totalSteps, ...(event.loss === undefined ? {} : { loss: event.loss }),
      ...(event.gradNorm === undefined ? {} : { gradNorm: event.gradNorm }) });
    emitProgress(job);
    log(job, 'info', `AITK joint step ${step}${event.loss === undefined ? '' : ` loss ${event.loss}`}`);
  } else if (stage === 'checkpoint' || stage === 'checkpoint_stage') {
    log(job, 'info', `AITK joint ${stage}${step === undefined ? '' : ` at step ${step}`}`);
  } else if (stage !== 'event') {
    if (stage === 'done') state.doneSeen = true;
    job.phase = stage;
    emitProgress(job);
    log(job, 'info', `AITK joint ${stage}`);
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
  stage: string; step?: number; loss?: number; gradNorm?: number; totalSteps: number;
} | null {
  try {
    const event = JSON.parse(line) as Record<string, unknown>;
    if (!event || typeof event.stage !== 'string') return null;
    const step = typeof event.step === 'number' && Number.isInteger(event.step) && event.step >= 0 ? event.step : undefined;
    const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
    const loss = finite(event.ar_ce) && finite(event.ar_kl) && finite(event.nar_mse)
      ? event.ar_ce + 0.2 * event.ar_kl + event.nar_mse : undefined;
    return { stage: event.stage, ...(step === undefined ? {} : { step }),
      ...(loss === undefined ? {} : { loss }),
      ...(finite(event.gradient_norm) ? { gradNorm: event.gradient_norm } : {}), totalSteps };
  } catch { return null; }
}

export async function runYue2JointTrainJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as ResolvedYue2JointTrainOptions | undefined;
  const error = opts ? validateOptions(opts) : 'job is missing AITK joint training options';
  if (error) { finishJob(job, 'failed', error); return; }
  const o = opts!;
  const state: RelayState = { fatalMessage: '', doneSeen: false, lastStep: 0, totalSteps: o.steps };
  let nativeAttempted = false;
  try {
    log(job, 'info', `Starting explicit AITK YuE2 joint training (${o.device})`);
    nativeAttempted = true;
    await runYue2AceTrain(job, 'yue2-joint-train', buildYue2JointTrainArgs(o),
      Math.max(30 * 60 * 1000, o.steps * 10 * 60 * 1000), () => {
        if (!fs.existsSync(o.outDir)) return 'AITK joint trainer exited without creating its output directory';
        const checkpoint = path.join(o.outDir, `checkpoint-step${o.steps}`);
        if (!fs.existsSync(checkpoint)) return `AITK joint trainer exited without final checkpoint-step${o.steps}`;
        if (['adapter.safetensors', 'optimizer.resume', 'native-ar.safetensors', 'native-nar.safetensors']
          .some(name => !fs.existsSync(path.join(checkpoint, name)))) {
          return `AITK checkpoint-step${o.steps} is incomplete (combined adapter, native AR/NAR exports and optimizer resume are required)`;
        }
        return null;
      }, (line, st) => relayJsonLine(job, line, st), state, o.spawnEnv);
    // runYue2AceTrain can fail early (for example when ace-train is absent)
    // and marks the job failed itself. Never overwrite that terminal state.
    if (!isCancelled(job) && job.status === 'running') finishJob(job, 'done');
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
