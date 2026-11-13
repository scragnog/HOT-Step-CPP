// training/mm3Ladder.ts — an MM3 run's checkpoint ladder: the previews saved
// per checkpoint, a render on demand against a saved step, and "Use this
// checkpoint", which points the dataset's album presets at the chosen step.
//
// The UI draws it with the same ladder review as YuE2's: the response carries
// the blind letters and the rung and album scores (mm3Review.ts), and the ops
// below score, select and clean up.

import fs from 'fs';
import path from 'path';
import { z } from 'zod/v4';
import { engineReady } from '../../engineState.js';
import { isEngineSuspended } from '../aceEngineProcess.js';
import { restoreMm3Selection } from '../backends/minimax/index.js';
import { runOnGpuLane } from '../generation/gpuLane.js';
import { evictOtherFamilies } from '../generation/residency.js';
import { getDataset } from './datasetsRepo.js';
import { activeJobForDataset, gpuPreviewReserved, reserveGpuForPreview, runReservedPreview } from './labelingQueue.js';
import { clearMm3LinkedRun, mm3LinkedRefFor, pointMm3PresetsAt, readMm3Linked } from './lyricStudioExport.js';
import { hasActivePipeline } from './pipelineRunner.js';
import { withReviewMutation } from './review/operations.js';
import { deleteMm3RunScores, getMm3AlbumScore, listMm3RungScores, mm3BlindLabels, planMm3Cleanup, prunedMm3Previews, runMm3Cleanup,
  scoreMm3Album, scoreMm3Rung, type Mm3AlbumScore, type Mm3RungScore } from './mm3Review.js';
import { listMm3RunPreviews, MM3_PREVIEW_DEFAULTS, planMm3Previews, renderMm3Preview, type Mm3PreviewPlan } from './mm3Preview.js';
import { listMm3Runs, mm3AdapterRoot, mm3RunManifestPath, readMm3RunManifest, type Mm3RecordedPreviewPlan, type Mm3RunManifest } from './mm3Runs.js';
import { mm3CodesDir } from './mm3Train.js';
import { clearMm3AdapterPath, getAllPresets } from '../../db/lireekDb.js';
import { setYue2ReviewComplete, setYue2RunFinished, yue2ReviewComplete, yue2RunFinished } from './yue2AitkRuns.js';
import type { Yue2CleanupItem } from './yue2Cleanup.js';
import { deleteMm3RunFolder, mm3RunOrigin, withMm3RunLock } from './yue2Mirror.js';
import { acceptTrainingSnapshot, assertDatasetCurrent, registerTrainingOperations, trainingOp, TrainingOperationFailure } from './operations.js';
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

/** `pruned`: cleanup removed the audio, and `audioUrl` is empty. */
export type Mm3LadderPreview = TrainingPreview & { audioUrl: string; pruned?: boolean };

export interface Mm3LadderResponse {
  dataset: { id: string; revision: string };
  datasetName: string;
  run: string;
  running: boolean;
  createdAt: number;
  /** The worker this run was mirrored from; null = trained here. */
  origin: string | null;
  /** Marked reviewed without scoring every rung (the Review page counts it scored). */
  reviewComplete: boolean;
  /** The checkpoint recorded for this dataset, or null. */
  inUse: string | null;
  checkpoints: Array<{ step: number; ref: string; loss?: number }>;
  previews: Mm3LadderPreview[];
  /** Blind letter per checkpoint step. */
  blindLabels: Record<string, string>;
  scores: Mm3RungScore[];
  album: Mm3AlbumScore | null;
}

/** GET /datasets/:id/mm3-previews?run= */
export function mm3Ladder(datasetId: string, runName: string): Mm3LadderResponse {
  const { ds, run, running } = runOf(datasetId, runName);
  const p = readMm3RunManifest(run.dir)?.options.preview;
  const previews = listMm3RunPreviews(path.join(run.dir, 'previews'), {
    seconds: p?.seconds ?? MM3_PREVIEW_DEFAULTS.seconds, seed: p?.seed ?? MM3_PREVIEW_DEFAULTS.seed,
  }).map(v => ({ ...v, audioUrl: `/api/training/mm3/preview?run=${encodeURIComponent(run.runName)}&file=${encodeURIComponent(v.file)}` }))
    .concat(prunedMm3Previews(path.join(run.dir, 'previews')).map(v => ({ ...v, audioUrl: '' })));
  return {
    dataset: { id: ds.id, revision: ds.updatedAt }, datasetName: ds.name,
    run: run.runName, running, createdAt: run.startedAt ?? run.updatedAt, origin: mm3RunOrigin(run.runName),
    reviewComplete: yue2ReviewComplete(run.dir), inUse: mm3LinkedRefFor(ds),
    checkpoints: run.checkpoints.flatMap(c => {
      const ref = mm3CheckpointRef(c.dir);
      return ref ? [{ step: c.step, ref, loss: c.loss }] : [];
    }),
    previews,
    blindLabels: mm3BlindLabels(run),
    scores: listMm3RungScores(ds.id, run.runName),
    album: getMm3AlbumScore(run.runName),
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
 *  the base model) with the run's own preview recipe, at the length and MLP
 *  scale the form shows now when given (as YuE2's Render does). The rung's
 *  first take uses the run's seed so it compares with the other rungs; later
 *  takes use a random seed. The run is over, so nothing pauses, but previews pin the LM to
 *  the base adapters render on, so the user's selection is put back after.
 *  Holds the run's lock to the end, so no delete or cleanup removes the
 *  checkpoint or the previews folder while the render reads and writes them. */
export function renderMm3LadderTake(datasetId: string, runName: string, step: number,
                                    over: { seconds?: number; scaleMlp?: number } = {}): Promise<TrainingPreview[]> {
  return withMm3RunLock(runName, () => renderLocked(datasetId, runName, step, over));
}

async function renderLocked(datasetId: string, runName: string, step: number,
                            over: { seconds?: number; scaleMlp?: number }): Promise<TrainingPreview[]> {
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
  if (over.seconds !== undefined) plan.seconds = over.seconds;
  if (over.scaleMlp !== undefined) plan.scaleMlp = over.scaleMlp;
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
export const mm3CleanupSchema = mm3SelectSchema.extend({
  blindLabel: z.string().optional(),
  choice: z.object({ caches: z.boolean().optional(), otherCheckpoints: z.boolean().optional(),
    otherRuns: z.boolean().optional(), resume: z.boolean().optional(), otherPreviews: z.boolean().optional() }),
});
const rating = z.number().int().min(1).max(5).nullable().optional();
const scoreSchema = z.object({ datasetId: z.string(), runName: z.string(), step: z.number().int().positive(),
  likeness: rating, corruption: rating, notes: z.string().optional(), blind: z.boolean().optional(), blindLabel: z.string().optional() });
const direction = z.enum(['under', 'right', 'over']).nullable().optional();
const albumSchema = z.object({ datasetId: z.string(), runName: z.string(),
  score: rating, instruments: direction, vocals: direction, notes: z.string().optional() });

const sameRef = (a: string | null, b: string) => !!a && path.normalize(a.replace(/[\\/]/g, path.sep)).toLowerCase() === path.normalize(b.replace(/[\\/]/g, path.sep)).toLowerCase();
function parse<T extends z.ZodType>(schema: T, body: unknown): z.infer<T> {
  const r = schema.safeParse(body);
  if (!r.success) throw fail(400, r.error.issues.map(i => `${i.path.join('.')}: ${i.message}`).join('; '));
  return r.data;
}

/** The chosen checkpoint's adapter reference, or a 409 when it is gone. */
function chosenRef(run: { checkpoints: Array<{ step: number; dir: string }> }, step: number): string {
  const ckpt = run.checkpoints.find(c => c.step === step);
  const ref = ckpt ? mm3CheckpointRef(ckpt.dir) : null;
  if (!ref) throw fail(409, 'That checkpoint is no longer on disk; reload the ladder');
  return ref;
}

registerTrainingOperations('mm3', (router, deps) => {
  /** One rung's likeness, corruption or notes; a field not sent keeps its value. */
  router.post('/score', trainingOp(async req => {
    const { datasetId, runName, ...input } = parse(scoreSchema, req.body);
    const { ds, run } = runOf(datasetId, runName);
    try { return { score: scoreMm3Rung(ds, run, input) }; }
    catch (err) { throw fail(400, err instanceof Error ? err.message : String(err)); }
  }));
  /** The run's album verdict. */
  router.post('/album', trainingOp(async req => {
    const { datasetId, runName, ...input } = parse(albumSchema, req.body);
    const { ds, run } = runOf(datasetId, runName);
    return { score: scoreMm3Album(ds, run.runName, input) };
  }));
  /** Record the dataset's pick, update existing album presets, and return the
   *  cleanup plan even when there are no presets. */
  router.post('/select', trainingOp(async req => {
    const { snapshot } = await acceptTrainingSnapshot(req.body, mm3SelectSchema, 'review', deps);
    const pick = snapshot.payload;
    if (snapshot.dataset?.id !== pick.datasetId) throw fail(409, 'Dataset selection changed');
    // Under the run's lock, so a finishing mirror pass cannot link its own
    // pick over this one while it is made.
    return withReviewMutation(pick.datasetId, () => withMm3RunLock(pick.runName, async () => {
      assertDatasetCurrent(snapshot, deps);
      const { ds, run } = runOf(pick.datasetId, pick.runName);
      if (activeJobForDataset(ds.id)) throw fail(409, 'A job is running for this dataset; pick a checkpoint when it ends.');
      const ref = chosenRef(run, pick.step);
      const presets = pointMm3PresetsAt(ds, ref);
      return { ref, presets, plan: { ...planMm3Cleanup(ds, run, pick.step), otherRuns: otherRunsItem(otherMm3Runs(ds, run.runName)) } };
    }));
  }));
  /** Cleanup around the chosen step, only while it is still the linked
   *  adapter and nothing runs for the dataset. Never leaves the run's folder. */
  router.post('/cleanup', trainingOp(async req => {
    const { snapshot } = await acceptTrainingSnapshot(req.body, mm3CleanupSchema, 'review', deps);
    const pick = snapshot.payload;
    if (snapshot.dataset?.id !== pick.datasetId) throw fail(409, 'Dataset selection changed');
    return withReviewMutation(pick.datasetId, async () => {
      // Every check runs under the run's lock, right before anything goes: a
      // mirror pass or a ladder render holding it may have moved the linked
      // checkpoint or the dataset while this waited.
      const { ds, run, result } = await withMm3RunLock(pick.runName, async () => {
        assertDatasetCurrent(snapshot, deps);
        const { ds, run, running } = runOf(pick.datasetId, pick.runName);
        if (running || activeJobForDataset(ds.id) || hasActivePipeline()) throw fail(409, 'A job or pipeline is active for this dataset');
        if (gpuPreviewReserved()) throw fail(409, PREVIEW_BUSY);
        assertMm3CleanupLinked(mm3LinkedRefFor(ds), chosenRef(run, pick.step),
          pick.choice.otherCheckpoints ? run.checkpoints.filter(c => c.step !== pick.step).map(c => c.dir) : [],
          getAllPresets().flatMap(p => typeof p.mm3_adapter_path === 'string' ? [p.mm3_adapter_path] : []));
        if (pick.blindLabel && pick.blindLabel !== mm3BlindLabels(run)[pick.step]) throw fail(409, 'Blind label no longer identifies the selected rung');
        return { ds, run, result: runMm3Cleanup(ds, run, pick.step, pick.choice) };
      });
      // Each other run is re-checked under its own lock; one that changed
      // since the plan (a preview started, a preset moved to it) is kept.
      const others = pick.choice.otherRuns ? otherMm3Runs(ds, run.runName) : [];
      const removed: typeof others = [];
      for (const r of others) {
        try { await deleteMm3RunFolder(r.runName, r.dir, () => assertRemovable(ds.id, r.runName, false)); removed.push(r); }
        catch (err) { result.done.push(`kept ${r.runName} (${err instanceof Error ? err.message : String(err)})`); }
      }
      if (removed.length) {
        result.freedBytes += otherRunsItem(removed).bytes;
        result.done.push(`${removed.length} other run(s)`);
      }
      // As YuE2's: a picked and cleaned-up run has nothing left to review.
      setYue2RunFinished(run.dir, { pickedStep: pick.step, pickedBlind: !!pick.blindLabel, pickedLabel: pick.blindLabel ?? '' });
      return result;
    });
  }));
});

// ── the other runs, the Review page and run deletion ────────────────────────

/** The dataset's other runs that cleanup may delete: as YuE2's, never one
 *  training now and never one already finished (picked and cleaned up). */
function otherMm3Runs(ds: { id: string; slug: string }, keep: string) {
  const job = activeJobForDataset(ds.id);
  const live = job?.kind === 'mm3-train-lm' ? path.resolve(String((job.opts as { outDir?: string } | undefined)?.outDir || '')) : '';
  return listMm3Runs(ds.id, ds.slug).filter(r => r.runName !== keep && path.resolve(r.dir) !== live && !yue2RunFinished(r.dir) && !presetsUsing(r.runName).length && !recordUsesRun(r.runName));
}

/** Validate the link and every checkpoint folder about to be deleted. */
function assertMm3CleanupLinked(linked: string | null, chosen: string, deleting: string[], presetRefs: string[]): void {
  if (!sameRef(linked, chosen)) throw fail(409, 'This checkpoint is no longer the linked adapter');
  const norm = (p: string) => path.resolve(p.replace(/[\\/]/g, path.sep)).toLowerCase();
  if (deleting.some(dir => presetRefs.some(ref => norm(path.resolve(mm3AdapterRoot(), ref.replace(/[\\/]/g, path.sep))).startsWith(norm(dir) + path.sep))))
    throw fail(409, 'An album preset uses a checkpoint that cleanup would delete');
}
function recordUsesRun(runName: string): boolean {
  return Object.values(readMm3Linked()).some(p => p.ref.split(/[\\/]/)[0]?.toLowerCase() === runName.toLowerCase());
}
const otherRunsItem = (runs: Array<{ runName: string; sizeBytes: number }>): Yue2CleanupItem =>
  ({ count: runs.length, bytes: runs.reduce((s, r) => s + r.sizeBytes, 0), detail: runs.map(r => r.runName) });

/** The album presets that point at a checkpoint of this run. */
function presetsUsing(runName: string): Array<{ lyricsSetId: number; name: string }> {
  const name = runName.toLowerCase();
  return getAllPresets().filter(p => typeof p.mm3_adapter_path === 'string'
    && p.mm3_adapter_path.split(/[\\/]/).filter(Boolean)[0]?.toLowerCase() === name)
    .map(p => ({ lyricsSetId: p.lyrics_set_id as number, name: p.album ? `${p.artist_name} / ${p.album}` : String(p.artist_name) }));
}

const PREVIEW_BUSY = 'A checkpoint preview is rendering; try again when it ends.';

/** Whether a run's folder may go, checked under its lock just before it does
 *  (deleteMm3RunFolder): no preview rendering, not training, and no album
 *  preset using one of its checkpoints (a mirror pass may have linked one
 *  while the delete waited). `unlink` clears those presets' MM3 adapter
 *  instead of refusing. */
function assertRemovable(datasetId: string, runName: string, unlink: boolean): void {
  if (gpuPreviewReserved()) throw fail(409, PREVIEW_BUSY);
  if (runOf(datasetId, runName).running) throw fail(409, 'That run is still training; stop it first');
  if (recordUsesRun(runName) && !unlink) throw fail(409, 'The dataset uses a checkpoint of this run. Use a checkpoint of another run first.');
  const linked = presetsUsing(runName);
  if (linked.length && !unlink) throw fail(409, 'An album preset uses a checkpoint of this run. Use a checkpoint of another run first.');
  for (const p of linked) clearMm3AdapterPath(p.lyricsSetId);
  if (unlink) clearMm3LinkedRun(runName);
}

/** DELETE /datasets/:id/mm3-runs/:run — the run folder and its mirror record.
 *  Refused while a preview renders and while it trains. Without `purge` it is
 *  also refused while an album preset uses one of its checkpoints (the preset
 *  would point at nothing), and rung scores stay. `purge` (the Review page's
 *  delete) removes everything: it unlinks those presets' MM3 adapter and drops
 *  the run's rung and album scores. */
export function deleteMm3Run(datasetId: string, runName: string, purge = false): Promise<{ run: string; unlinked: string[] }> {
  return withReviewMutation(datasetId, async () => {
    const { run } = runOf(datasetId, runName);
    const unlinked = purge ? presetsUsing(run.runName).map(p => p.name) : [];
    await deleteMm3RunFolder(run.runName, run.dir, () => assertRemovable(datasetId, run.runName, purge));
    if (purge) deleteMm3RunScores(datasetId, run.runName);
    return { run: run.runName, unlinked };
  });
}

/** POST /datasets/:id/mm3-review-complete — mark a ladder reviewed without
 *  scoring every rung, as YuE2's Refine toggle does. */
export function setMm3ReviewComplete(datasetId: string, runName: string, complete: boolean): { reviewComplete: boolean } {
  const { run } = runOf(datasetId, runName);
  setYue2ReviewComplete(run.dir, complete);
  return { reviewComplete: yue2ReviewComplete(run.dir) };
}

/** One Review-page row per MM3 run with a checkpoint, in YuE2's row shape
 *  (`family: 'mm3'`), so the same panel lists both. */
export function mm3ReviewRows(ds: { id: string; slug: string; name: string }) {
  const job = activeJobForDataset(ds.id);
  const live = job?.kind === 'mm3-train-lm' ? path.resolve(String((job.opts as { outDir?: string } | undefined)?.outDir || '')) : '';
  return listMm3Runs(ds.id, ds.slug).flatMap(run => {
    const rungs = run.checkpoints.filter(c => mm3CheckpointRef(c.dir)).map(c => c.step);
    if (!rungs.length) return [];
    const scores = listMm3RungScores(ds.id, run.runName);
    const scored = new Set(scores.filter(s => s.likeness !== null || s.corruption !== null || s.notes).map(s => s.step));
    let best: { step: number; overall: number } | null = null;
    for (const s of [...scores].sort((a, b) => a.step - b.step)) {
      if (s.likeness === null || s.corruption === null || !rungs.includes(s.step)) continue;
      const overall = (s.likeness + (6 - s.corruption)) / 2;
      if (!best || overall > best.overall) best = { step: s.step, overall };
    }
    const manifest = readMm3RunManifest(run.dir);
    const p = manifest?.options.preview;
    const previews = listMm3RunPreviews(path.join(run.dir, 'previews'), { seconds: p?.seconds ?? MM3_PREVIEW_DEFAULTS.seconds, seed: p?.seed ?? MM3_PREVIEW_DEFAULTS.seed })
      .filter(v => !v.base && rungs.includes(v.step)).length;
    const running = !!live && path.resolve(run.dir) === live;
    return [{
      family: 'mm3' as const, datasetId: ds.id, datasetSlug: ds.slug, datasetName: ds.name, refineRun: run.runName,
      status: running ? 'running' : run.outcome, createdAt: run.startedAt ?? run.updatedAt, origin: mm3RunOrigin(run.runName),
      live: running, rungs: rungs.length, previews, scored: scored.size, unscored: rungs.filter(s => !scored.has(s)).length,
      reviewed: yue2ReviewComplete(run.dir),
      linkedAlbums: presetsUsing(run.runName).map(p => p.name),
      best: best ? { ...best, blindLabel: manifest?.blindLabels?.[best.step] ?? '' } : null,
      decoderOnly: false, klMin: null, klMax: null, baseMatched: true, finished: yue2RunFinished(run.dir),
    }];
  });
}
