// yue2Optimise.ts — the YuE2 Optimise phase: optional measurements of a
// prepared album before training (Dataset-Calibrated Training).
//
// Everything this phase measures is saved in ONE file in the dataset's own
// folder, beside the audio and sidecars: _hotstep-optimisation.json. Not the
// database, so it travels with the dataset and is easy to read later. Each
// measurement is one top-level section; writing a section keeps the others.
//
// First measurement, baseLoss: the base model's loss on every song before any
// training (ace-train yue2-joint-train --eval-base-loss). The planner's CE says
// how unfamiliar the album's songs and plans are to YuE2; the decoder's flow
// MSE how unfamiliar its sound is.
import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import { emitProgress, finishJob, isCancelled, type TrainingJob } from './labelingQueue.js';
import { log, runYue2AceTrain, YUE2_IDLE_MS, type RelayState } from './yue2TrainRunner.js';
import { ensureYue2PreparedDataset } from './yue2AutoPrepare.js';
import type { ResolvedYue2AitkPrepareOptions } from './yue2AitkPrepareRunner.js';
import { getDataset } from './datasetsRepo.js';
import { datasetDir } from './paths.js';
import { yue2ModelDir } from './yue2Train.js';
import { config } from '../../config.js';

export const OPTIMISATION_FILE = '_hotstep-optimisation.json';
export const optimisationPath = (sourceDir: string) => path.join(sourceDir, OPTIMISATION_FILE);

export interface Yue2BaseLossItem { file: string; frames: number; arTokens: number; arCe: number; narMse: number; narMseByT: number[] }
export interface Yue2BaseLoss {
  measuredAt: string;
  base: string;
  companion: boolean;
  arTargets: string;
  narWindow: number;
  timesteps: number[];
  seed: number;
  items: Yue2BaseLossItem[];
  summary: { items: number; arCeMean: number; arCeTokenMean: number; narMseMean: number };
}
export interface Yue2Optimisation { version: 1; baseLoss?: Yue2BaseLoss; [section: string]: unknown }

export function readOptimisation(sourceDir: string): Yue2Optimisation | null {
  try {
    const parsed = JSON.parse(fs.readFileSync(optimisationPath(sourceDir), 'utf8'));
    return parsed && typeof parsed === 'object' ? parsed as Yue2Optimisation : null;
  } catch { return null; }
}

/** Replace one section, keep the rest. Atomic, next to the file. */
export function writeOptimisationSection(sourceDir: string, section: string, value: unknown): void {
  const next: Yue2Optimisation = { ...(readOptimisation(sourceDir) ?? { version: 1 }), version: 1, [section]: value };
  const file = optimisationPath(sourceDir);
  const tmp = `${file}.${randomUUID()}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(next, null, 2));
  fs.renameSync(tmp, file);
}

export interface Yue2BaseLossJobOptions {
  checkpoint: string;
  device: string;
  preparation: ResolvedYue2AitkPrepareOptions;
}

const EVAL_SEED = 42;
const EVAL_NAR_WINDOW = 1500; // the decoder crop every preset trains on

/** The job: prepare the dataset (reused when unchanged, as training does),
 *  run the base-loss pass, save the baseLoss section. */
export async function runYue2BaseLossJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as Yue2BaseLossJobOptions;
  const ds = getDataset(job.datasetId);
  if (!ds) { finishJob(job, 'failed', 'Dataset not found'); return; }
  let manifest: string | undefined;
  try {
    manifest = await ensureYue2PreparedDataset(job, opts.preparation);
  } catch (err) {
    if (!isCancelled(job)) finishJob(job, 'failed', err instanceof Error ? err.message : String(err));
    return;
  }
  if (!manifest || isCancelled(job) || job.status === 'failed') return;
  const out = path.join(datasetDir(ds.slug), `base-loss-${randomUUID()}`);
  const companion = [yue2ModelDir(), config.aceServer.models]
    .map(dir => path.join(dir, 'nar_lora_joint_v9.safetensors')).find(p => fs.existsSync(p));
  const args = ['yue2-joint-train', '--checkpoint', opts.checkpoint, '--dataset', manifest, '--output', out,
    '--device', opts.device, '--seed', String(EVAL_SEED), '--cursor-weight', '0', '--ar-targets', 'base',
    '--nar-crop-frames', String(EVAL_NAR_WINDOW), ...(companion ? ['--companion', companion] : []), '--eval-base-loss'];
  const items: Yue2BaseLossItem[] = [];
  let summary: Record<string, unknown> | null = null;
  let total = 0;
  try { total = (JSON.parse(fs.readFileSync(manifest, 'utf8')).items ?? []).length; } catch { /* progress shows done only */ }
  job.done = 0; job.total = total || 1; job.phase = 'measuring'; emitProgress(job);
  log(job, 'info', `Base loss: ${total || '?'} songs, base ${path.basename(opts.checkpoint)}${companion ? ' with the companion decoder' : ''}`);
  const onLine = (line: string) => {
    let ev: Record<string, unknown>;
    try { ev = JSON.parse(line); } catch { log(job, 'info', line); return; }
    // The engine's bare {"stage":"eval"} marks the start; song lines carry an item index.
    if (ev.stage === 'eval' && typeof ev.item === 'number') {
      items.push({ file: String(ev.id), frames: Number(ev.frames), arTokens: Number(ev.ar_tokens), arCe: Number(ev.ar_ce),
        narMse: Number(ev.nar_mse), narMseByT: Array.isArray(ev.nar_mse_t) ? (ev.nar_mse_t as unknown[]).map(Number) : [] });
      job.phase = 'measuring'; job.done = items.length; job.total = Math.max(total, items.length); emitProgress(job);
      log(job, 'info', `${ev.id}: planner CE ${Number(ev.ar_ce).toFixed(3)}, decoder MSE ${Number(ev.nar_mse).toFixed(3)}`);
    } else if (ev.stage === 'eval_summary') summary = ev;
  };
  // runYue2AceTrain throws on a failed exit and leaves finishing the job to
  // the caller.
  try {
    await runYue2AceTrain(job, 'yue2-base-loss', args, YUE2_IDLE_MS,
      () => summary ? null : 'the base-loss pass ended without a summary',
      (line: string) => onLine(line), {} as RelayState, undefined, false);
  } catch (err) {
    if (!isCancelled(job)) finishJob(job, 'failed', err instanceof Error ? err.message : String(err));
    return;
  } finally {
    try { fs.rmSync(out, { recursive: true, force: true }); } catch { /* scratch */ }
  }
  if (isCancelled(job) || !summary) return;
  const s = summary as Record<string, unknown>;
  const result: Yue2BaseLoss = {
    measuredAt: new Date().toISOString(), base: path.basename(opts.checkpoint), companion: !!companion,
    arTargets: String(s.ar_targets ?? 'base'), narWindow: EVAL_NAR_WINDOW, timesteps: [250, 500, 750], seed: EVAL_SEED, items,
    summary: { items: Number(s.items), arCeMean: Number(s.ar_ce_mean), arCeTokenMean: Number(s.ar_ce_token_mean), narMseMean: Number(s.nar_mse_mean) },
  };
  try { writeOptimisationSection(ds.sourceDir, 'baseLoss', result); }
  catch (err) { finishJob(job, 'failed', `Measured, but could not save ${optimisationPath(ds.sourceDir)}: ${err instanceof Error ? err.message : String(err)}`); return; }
  log(job, 'info', `Base loss saved to ${optimisationPath(ds.sourceDir)}: planner CE ${result.summary.arCeMean.toFixed(3)}, decoder MSE ${result.summary.narMseMean.toFixed(3)}`);
  job.done = job.total = items.length;
  finishJob(job, 'done');
}
