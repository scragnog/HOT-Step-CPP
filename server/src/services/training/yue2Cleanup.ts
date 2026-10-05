// yue2Cleanup.ts — after a refinement rung is chosen as the final adapter,
// reclaim the disk around it: prepared caches, the run's other checkpoints,
// the dataset's other joint runs, the chosen rung's resume file, and the
// other rungs' previews. Never source audio, sidecars, labels or captions
// (preparedDataReset already draws that line), and never the chosen rung's
// adapter files.
import fs from 'fs';
import path from 'path';
import { listPreparedCaches, clearPreparedCaches } from './preparedDataReset.js';
import { listAllYue2AitkRuns, listYue2AitkRuns, deleteYue2AitkRun, yue2RunFinished, setYue2RunFinished, recordYue2AitkRun, moveYue2AitkRun,
  freeYue2RunDirectory, withYue2RunLock, yue2RemoteRunDirectory, type Yue2AitkRunRecord } from './yue2AitkRuns.js';
import { rebaseYue2JointLinks } from './lyricStudioExport.js';
import { getDataset } from './datasetsRepo.js';
import { config } from '../../config.js';
import { listYue2JointPreviews, pruneYue2JointPreviews } from './yue2JointPreview.js';
import { archiveYue2TrainLogs, noteYue2TrainLog, trainLogArchiveDir, trainLogName } from './datasetProfile.js';

export interface Yue2CleanupItem { count: number; bytes: number; detail?: string[] }
export interface Yue2CleanupPlan {
  run: string; step: number; keep: string;
  caches: Yue2CleanupItem;
  otherCheckpoints: Yue2CleanupItem;
  otherRuns: Yue2CleanupItem;
  resume: Yue2CleanupItem;
  otherPreviews: Yue2CleanupItem;
}
export interface Yue2CleanupChoice { caches?: boolean; otherCheckpoints?: boolean; otherRuns?: boolean; resume?: boolean; otherPreviews?: boolean }

function dirBytes(p: string): number {
  let total = 0;
  try {
    for (const e of fs.readdirSync(p, { withFileTypes: true })) {
      const f = path.join(p, e.name);
      if (e.isDirectory()) total += dirBytes(f);
      else if (e.isFile()) { try { total += fs.statSync(f).size; } catch { /* vanished */ } }
    }
  } catch { /* missing */ }
  return total;
}
const fileBytes = (p?: string) => { try { return p ? fs.statSync(p).size : 0; } catch { return 0; } };

function locate(ds: { id: string; slug: string }, runId: string, step: number) {
  const runs = listYue2AitkRuns(ds.id, ds.slug);
  const run = runs.find(r => r.jobId === runId);
  if (!run) throw new Error('Unknown run');
  const keep = run.checkpoints.find(c => c.step === step);
  if (!keep?.arPath || !keep.narPath) throw new Error(`No complete checkpoint at step ${step}`);
  return { runs, run, keep };
}

export function planYue2Cleanup(ds: { id: string; slug: string; sourceDir: string; lyricsSetId?: number }, runId: string, step: number): Yue2CleanupPlan {
  const { runs, run, keep } = locate(ds, runId, step);
  const caches = listPreparedCaches(ds.slug, ds.sourceDir);
  const others = run.checkpoints.filter(c => c.step !== step);
  // A finished run is a shipped adapter: never swept as an "other run".
  const otherRuns = runs.filter(r => r.jobId !== runId && r.status !== 'running' && !yue2RunFinished(r.output));
  const previews = listYue2JointPreviews(run.output).filter(p => p.step !== step);
  const previewBytes = previews.reduce((s, p) => s + (p.file ? fileBytes(path.join(run.output, 'previews', p.file)) : 0), 0);
  return {
    run: runId, step, keep: keep.dir,
    caches: { count: caches.length, bytes: caches.reduce((s, c) => s + c.bytes, 0), detail: caches.map(c => c.name) },
    otherCheckpoints: { count: others.length, bytes: others.reduce((s, c) => s + dirBytes(c.dir), 0) },
    otherRuns: { count: otherRuns.length, bytes: otherRuns.reduce((s, r) => s + dirBytes(r.output), 0), detail: otherRuns.map(r => path.basename(r.output)) },
    resume: { count: keep.optimizerPath ? 1 : 0, bytes: fileBytes(keep.optimizerPath) },
    otherPreviews: { count: previews.length, bytes: previewBytes },
  };
}

export async function runYue2Cleanup(ds: { id: string; slug: string; sourceDir: string; lyricsSetId?: number }, runId: string, step: number, choice: Yue2CleanupChoice,
  pick: { blind?: boolean; blindLabel?: string } = {}): Promise<{ freedBytes: number; done: string[]; finishError?: string; moveError?: string }> {
  const plan = planYue2Cleanup(ds, runId, step);
  const { runs, run, keep } = locate(ds, runId, step);
  if (pick.blind && (!pick.blindLabel || pick.blindLabel !== run.blindLabels?.[step])) throw new Error('Blind label does not match the chosen checkpoint');
  if (choice.resume && keep.optimizerPath && run.origin) {
    // A pulled rung revalidates against its worker manifest once the worker's
    // copy is gone, so the prune is recorded and read back before anything is
    // deleted: recordYue2AitkRun swallows write failures, and a resume file
    // deleted without this record would make the rung unusable offline.
    const name = path.basename(keep.optimizerPath);
    recordYue2AitkRun({ ...run, checkpoints: run.checkpoints.map(c => c.step === step
      ? { ...c, prunedFiles: [...new Set([...(c.prunedFiles ?? []), name])] } : c) });
    const landed = listYue2AitkRuns(ds.id, ds.slug).find(r => r.jobId === run.jobId)?.checkpoints.find(c => c.step === step)?.prunedFiles?.includes(name);
    if (!landed) throw new Error('Could not record the resume-file prune in the run index; nothing was deleted');
  }
  let freed = 0; const done: string[] = [];
  // The chosen run's loss curve, kept with the dataset: run folders get moved
  // and deleted by hand later, and calibration reads the curve per album.
  let archivedSegs: string[] = [];
  try { archivedSegs = archiveYue2TrainLogs(ds.slug, run.jobId, run.output); }
  catch (err: any) { console.warn(`[Training] YuE2 cleanup: could not archive the loss log of ${runId}: ${err?.message || err}`); }
  try { noteYue2TrainLog(ds.slug, run.jobId, { output: run.output, keptStep: step, status: 'finished', segments: archivedSegs }); }
  catch (err: any) { console.warn(`[Training] YuE2 cleanup: could not note the loss log of ${runId}: ${err?.message || err}`); }
  if (choice.otherRuns) {
    for (const r of runs.filter(r => r.jobId !== runId && r.status !== 'running' && !yue2RunFinished(r.output))) { deleteYue2AitkRun(r.jobId); }
    freed += plan.otherRuns.bytes; done.push(`${plan.otherRuns.count} other run(s)`);
  }
  if (choice.otherCheckpoints) {
    for (const c of run.checkpoints.filter(c => c.step !== step)) fs.rmSync(c.dir, { recursive: true, force: true });
    freed += plan.otherCheckpoints.bytes; done.push(`${plan.otherCheckpoints.count} other checkpoint(s)`);
  }
  if (choice.otherPreviews) {
    pruneYue2JointPreviews(run.output, step);
    freed += plan.otherPreviews.bytes; done.push(`${plan.otherPreviews.count} other preview(s)`);
  }
  if (choice.resume && keep.optimizerPath) {
    fs.rmSync(keep.optimizerPath, { force: true });
    freed += plan.resume.bytes; done.push('resume file');
  }
  if (choice.caches) {
    clearPreparedCaches(ds.slug, ds.sourceDir);
    freed += plan.caches.bytes; done.push(`${plan.caches.count} cache folder(s)`);
  }
  // Whatever the choice flags ("Keep everything" included), the ladder ends
  // here: the run stays where it is and is marked finished, which keeps it
  // off the Review page's Finish list and out of later otherRuns sweeps.
  try { setYue2RunFinished(run.output, { pickedStep: step, pickedBlind: pick.blind === true, pickedLabel: pick.blind ? pick.blindLabel! : '' }); }
  catch (err: any) {
    const finishError = `Cleanup completed, but the run could not be marked finished: ${err?.message || err}`;
    console.warn(`[Training] YuE2 cleanup: ${finishError}`);
    return { freedBytes: freed, done, finishError };
  }
  // A pulled ladder leaves its staging folder for the name a local run gets.
  // The run is already finished; a failed rename only leaves it where it was.
  if (run.origin) {
    try { const to = await settleYue2RemoteRun(run.jobId); if (to) done.push(`moved to ${path.basename(to)}`); }
    catch (err: any) {
      const moveError = `The run folder could not be renamed and stays at ${run.output}: ${err?.message || err}`;
      console.warn(`[Training] YuE2 cleanup: ${moveError}`);
      return { freedBytes: freed, done, moveError };
    }
  }
  return { freedBytes: freed, done };
}

/** Where a pulled run's folder belongs now (staging while under review, the
 *  local `<trigger>_<stamp>` name once finished), or null when it is already
 *  there, `-N` collision suffix included. */
export function plannedYue2RemoteMove(run: Yue2AitkRunRecord, taken?: Set<string>): string | null {
  if (!run.origin) return null;
  const ds = getDataset(run.datasetId);
  const wanted = path.resolve(yue2RemoteRunDirectory(config.aceServer.adapters, ds?.customTag || ds?.slug || run.datasetSlug,
    run.origin.worker, run.createdAt, yue2RunFinished(run.output)));
  const current = path.resolve(run.output);
  const name = path.basename(current).toLowerCase(), base = path.basename(wanted).toLowerCase();
  const settled = path.dirname(current).toLowerCase() === path.dirname(wanted).toLowerCase()
    && (name === base || (name.startsWith(`${base}-`) && /^\d+$/.test(name.slice(base.length + 1))));
  return settled ? null : freeYue2RunDirectory(wanted, taken);
}

/** Move a pulled run to plannedYue2RemoteMove's folder and repoint the run
 *  index, yue2-linked.json, album presets, rung scores and the loss-log note.
 *  Holds the run's lock, so no pull or checkpoint fetch of the same run is
 *  writing into the folder meanwhile, and plans from the index as it stands
 *  once the lock is held. Any failure puts the folder and every record back
 *  and rethrows. Returns the new folder, or null when nothing needed to move. */
export function settleYue2RemoteRun(jobId: string): Promise<string | null> {
  return withYue2RunLock(jobId, async () => {
    const run = listAllYue2AitkRuns().find(r => r.jobId === jobId);
    if (!run) throw new Error('Unknown run');
    const target = plannedYue2RemoteMove(run);
    if (!target) return null;
    const from = path.resolve(run.output);
    // Cleanup has just deleted most of this folder; Windows refuses the rename
    // (EPERM/EBUSY) while a scanner or a pending delete still holds a handle
    // inside it, for a second or so. The failed rename has changed nothing.
    for (let wait = 250; ; wait *= 2) {
      try { moveYue2AitkRun(run.jobId, target); break; }
      catch (err: any) {
        if (!['EPERM', 'EBUSY', 'EACCES'].includes(err?.code) || wait > 8000) throw err;
        await new Promise(r => setTimeout(r, wait));
      }
    }
    try { rebaseYue2JointLinks(from, target); }
    catch (err) {
      try { moveYue2AitkRun(run.jobId, from); }
      catch (back: any) { console.error(`[Training] YuE2: could not move ${target} back to ${from}: ${back?.message || back}`); }
      throw err;
    }
    // Informational only: the archived loss log notes where its run lives.
    try {
      if (fs.existsSync(path.join(trainLogArchiveDir(run.datasetSlug), `${trainLogName(run.jobId)}.json`))) noteYue2TrainLog(run.datasetSlug, run.jobId, { output: target });
    } catch (err: any) { console.warn(`[Training] YuE2: could not note the new folder of ${run.jobId}: ${err?.message || err}`); }
    return target;
  });
}

export interface Yue2RemoteFolderMove { jobId: string; dataset: string; finished: boolean; from: string; to: string; error?: string }

/** One-shot tidy of pulled runs still in the old `remote-<worker>-<uuid>`
 *  folders (or anywhere else they don't belong): finished runs get the local
 *  name, the rest the staging name. `apply: false` only lists the moves. */
export async function migrateYue2RemoteFolders(apply: boolean): Promise<Yue2RemoteFolderMove[]> {
  const out: Yue2RemoteFolderMove[] = [];
  const taken = new Set<string>();
  const runs = listAllYue2AitkRuns().filter(r => r.origin).sort((a, b) => a.createdAt - b.createdAt);
  for (const run of runs) {
    const to = plannedYue2RemoteMove(run, taken);
    if (!to) continue;
    taken.add(path.resolve(to).toLowerCase());
    const row: Yue2RemoteFolderMove = { jobId: run.jobId, dataset: run.datasetSlug, finished: yue2RunFinished(run.output), from: run.output, to };
    if (apply) {
      try { row.to = (await settleYue2RemoteRun(run.jobId)) ?? run.output; }
      catch (err: any) { row.error = err?.message || String(err); }
    }
    out.push(row);
  }
  return out;
}
