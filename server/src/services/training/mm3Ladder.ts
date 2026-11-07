// training/mm3Ladder.ts — an MM3 run's checkpoint ladder: the previews saved
// per checkpoint, a render on demand against a saved step, and "Use this
// checkpoint", which points the dataset's album presets at the chosen step.
//
// The UI draws it with the same CheckpointLadder as YuE2's. MM3 has no rung
// scores, so none are offered here.

import fs from 'fs';
import path from 'path';
import { z } from 'zod/v4';
import { engineReady } from '../../engineState.js';
import { isEngineSuspended } from '../aceEngineProcess.js';
import { restoreMm3Selection } from '../backends/minimax/index.js';
import { runOnGpuLane } from '../generation/gpuLane.js';
import { evictOtherFamilies } from '../generation/residency.js';
import { getDataset } from './datasetsRepo.js';
import { activeJobForDataset, reserveGpuForPreview, runReservedPreview } from './labelingQueue.js';
import { mm3PresetRefFor, pointMm3PresetsAt } from './lyricStudioExport.js';
import { listMm3RunPreviews, MM3_PREVIEW_DEFAULTS, planMm3Previews, renderMm3Preview, type Mm3PreviewPlan } from './mm3Preview.js';
import { listMm3Runs, mm3AdapterRoot, mm3RunManifestPath, readMm3RunManifest, type Mm3RecordedPreviewPlan, type Mm3RunManifest } from './mm3Runs.js';
import { mm3CodesDir } from './mm3Train.js';
import { acceptTrainingSnapshot, registerTrainingOperations, trainingOp, TrainingOperationFailure } from './operations.js';
import type { TrainingPreview } from './types.js';

const fail = (status: 400 | 404 | 409 | 503, error: string) => new TrainingOperationFailure(status, { error });

/** A checkpoint's weights file relative to the mm3-lm-adapters root: the
 *  reference album presets and the mm3LmAdapter param carry. */
export function mm3CheckpointRef(ckptDir: string): string | null {
  for (const name of ['adapter_model.safetensors', 'lokr_weights.safetensors']) {
    const file = path.join(ckptDir, name);
    const rel = path.relative(mm3AdapterRoot(), file);
    if (fs.existsSync(file) && rel && !rel.startsWith('..') && !path.isAbsolute(rel)) return rel;
  }
  return null;
}

function runOf(datasetId: string, runName: string) {
  const ds = getDataset(datasetId);
  if (!ds) throw fail(404, 'Dataset not found');
  const run = listMm3Runs(ds.id, ds.slug).find(r => r.runName === runName);
  if (!run) throw fail(404, 'No such training run for this dataset');
  const job = activeJobForDataset(ds.id);
  const outDir = job?.kind === 'mm3-train-lm' ? String((job.opts as { outDir?: string } | undefined)?.outDir || '') : '';
  const running = !!outDir && path.resolve(outDir) === path.resolve(run.dir);
  return { ds, run, running };
}

export type Mm3LadderPreview = TrainingPreview & { audioUrl: string };

export interface Mm3LadderResponse {
  dataset: { id: string; revision: string };
  run: string;
  running: boolean;
  /** The adapter the dataset's album presets use now, or null. */
  inUse: string | null;
  checkpoints: Array<{ step: number; ref: string; loss?: number }>;
  previews: Mm3LadderPreview[];
}

/** GET /datasets/:id/mm3-previews?run= */
export function mm3Ladder(datasetId: string, runName: string): Mm3LadderResponse {
  const { ds, run, running } = runOf(datasetId, runName);
  const p = readMm3RunManifest(run.dir)?.options.preview;
  const previews = listMm3RunPreviews(path.join(run.dir, 'previews'), {
    seconds: p?.seconds ?? MM3_PREVIEW_DEFAULTS.seconds, seed: p?.seed ?? MM3_PREVIEW_DEFAULTS.seed,
  }).map(v => ({ ...v, audioUrl: `/api/training/mm3/preview?run=${encodeURIComponent(run.runName)}&file=${encodeURIComponent(v.file)}` }));
  return {
    dataset: { id: ds.id, revision: ds.updatedAt },
    run: run.runName, running, inUse: mm3PresetRefFor(ds),
    checkpoints: run.checkpoints.flatMap(c => {
      const ref = mm3CheckpointRef(c.dir);
      return ref ? [{ step: c.step, ref, loss: c.loss }] : [];
    }),
    previews,
  };
}

/** The prompt "Render more" renders: the one the run recorded when it
 *  started, so editing a caption or clearing the codes cache later can neither
 *  change what the checkpoints are compared on nor break the render. A run from
 *  before the prompt was recorded resolves it once through `rebuild` (the
 *  dataset's captions and codes as they are now) and records it then. */
export function ladderPreviewPlan(runDir: string, manifest: Mm3RunManifest,
                                  rebuild: () => Mm3PreviewPlan | null): (Mm3RecordedPreviewPlan & { dir: string }) | null {
  const dir = path.join(runDir, 'previews');
  if (manifest.previewPlan?.specs.length) return { ...manifest.previewPlan, dir };
  const p = rebuild();
  if (!p) return null;
  const previewPlan: Mm3RecordedPreviewPlan = { specs: p.specs, seconds: p.seconds, seed: p.seed, scaleMlp: p.scaleMlp, scaleAttn: p.scaleAttn };
  try { fs.writeFileSync(mm3RunManifestPath(runDir), JSON.stringify({ ...manifest, previewPlan }, null, 2), 'utf-8'); }
  catch { /* rendered from the rebuilt plan anyway; recorded next time */ }
  return { ...previewPlan, dir };
}

/** POST /datasets/:id/mm3-previews/render — one more take of a saved step (0 =
 *  the base model) with the run's own preview recipe. The rung's first take
 *  uses the run's seed so it compares with the other rungs; later takes use a
 *  random seed. The run is over, so nothing pauses, but previews pin the LM to
 *  the base adapters render on, so the user's selection is put back after. */
export async function renderMm3LadderTake(datasetId: string, runName: string, step: number): Promise<TrainingPreview[]> {
  const { ds, run, running } = runOf(datasetId, runName);
  if (running) throw fail(409, 'This run is still training; its previews render at each checkpoint.');
  const ckpt = step === 0 ? null : run.checkpoints.find(c => c.step === step);
  if (step !== 0 && !ckpt) throw fail(400, 'That run has no saved checkpoint at that step');
  const manifest = readMm3RunManifest(run.dir);
  if (!manifest) throw fail(400, 'This run has no recorded recipe, so the caption its previews used is unknown.');
  const o = manifest.options;
  const plan = ladderPreviewPlan(run.dir, manifest, () => planMm3Previews({
    // everySteps only switches the plan on; nothing here is periodic.
    preview: { ...o.preview, everySteps: 1, previewSongId: o.preview?.previewSongId || manifest.previewSong?.id },
    manifest: ds.datasetJsonPath,
    captionsDir: ds.sourceDir,
    codesDir: fs.existsSync(o.codesDir) ? o.codesDir : path.join(mm3CodesDir(ds.slug), 'codes'),
    outDir: run.dir, holdout: o.holdout, trigger: o.trigger, captionFile: o.captionFile,
  }));
  if (!plan) throw fail(400, 'No preview caption could be built for this run.');
  if (isEngineSuspended() || !engineReady) throw fail(503, 'The engine is not running; previews need it up');
  const have = listMm3RunPreviews(plan.dir, { seconds: plan.seconds, seed: plan.seed });
  if (!reserveGpuForPreview()) throw fail(409, 'The GPU is busy with a training job or another preview; previews wait until it ends.');
  return runReservedPreview(() => runOnGpuLane(async () => {
    await evictOtherFamilies('minimax-m3');
    const out: TrainingPreview[] = [];
    try {
      for (const spec of plan.specs) {
        const extraTake = have.some(v => v.step === step && v.base === (step === 0) && v.kind === spec.kind && v.seed === plan.seed);
        out.push(await renderMm3Preview({
          spec, seconds: plan.seconds, seed: extraTake ? Math.floor(Math.random() * 2 ** 31) : plan.seed,
          adapterPath: ckpt?.dir ?? '', step, totalSteps: o.steps, dir: plan.dir, loss: ckpt?.loss,
          scaleMlp: plan.scaleMlp, scaleAttn: plan.scaleAttn, extraTake,
        }));
      }
    } finally {
      await restoreMm3Selection().catch(err => console.warn('[Training] MM3 ladder render: could not restore the model selection:', err?.message || err));
    }
    return out;
  }, { label: 'mm3 checkpoint preview', family: 'minimax-m3' }));
}

export const mm3SelectSchema = z.object({
  datasetId: z.string().min(1),
  runName: z.string().min(1),
  step: z.number().int().positive(),
});

registerTrainingOperations('mm3', (router, deps) => {
  /** "Use this checkpoint": the dataset's album presets move to this step.
   *  Idempotent; picking another step moves them again. */
  router.post('/select', trainingOp(async req => {
    const { snapshot } = await acceptTrainingSnapshot(req.body, mm3SelectSchema, 'review', deps);
    const pick = snapshot.payload;
    if (snapshot.dataset?.id !== pick.datasetId) throw fail(409, 'Dataset selection changed');
    const { ds, run } = runOf(pick.datasetId, pick.runName);
    if (activeJobForDataset(ds.id)) throw fail(409, 'A job is running for this dataset; pick a checkpoint when it ends.');
    const ckpt = run.checkpoints.find(c => c.step === pick.step);
    const ref = ckpt ? mm3CheckpointRef(ckpt.dir) : null;
    if (!ref) throw fail(409, 'That checkpoint is no longer on disk; reload the ladder');
    return { ref, presets: pointMm3PresetsAt(ds, ref) };
  }));
});
