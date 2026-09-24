// yue2Cleanup.ts — after a refinement rung is chosen as the final adapter,
// reclaim the disk around it: prepared caches, the run's other checkpoints,
// the dataset's other joint runs, the chosen rung's resume file, and the
// other rungs' previews. Never source audio, sidecars, labels or captions
// (preparedDataReset already draws that line), and never the chosen rung's
// adapter files.
import fs from 'fs';
import path from 'path';
import { listPreparedCaches, clearPreparedCaches } from './preparedDataReset.js';
import { listYue2AitkRuns, deleteYue2AitkRun } from './yue2AitkRuns.js';
import { listYue2JointPreviews, pruneYue2JointPreviews } from './yue2JointPreview.js';

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

export function planYue2Cleanup(ds: { id: string; slug: string; sourceDir: string }, runId: string, step: number): Yue2CleanupPlan {
  const { runs, run, keep } = locate(ds, runId, step);
  const caches = listPreparedCaches(ds.slug, ds.sourceDir);
  const others = run.checkpoints.filter(c => c.step !== step);
  const otherRuns = runs.filter(r => r.jobId !== runId && r.status !== 'running');
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

export function runYue2Cleanup(ds: { id: string; slug: string; sourceDir: string }, runId: string, step: number, choice: Yue2CleanupChoice): { freedBytes: number; done: string[] } {
  const plan = planYue2Cleanup(ds, runId, step);
  const { runs, run, keep } = locate(ds, runId, step);
  let freed = 0; const done: string[] = [];
  if (choice.otherRuns) {
    for (const r of runs.filter(r => r.jobId !== runId && r.status !== 'running')) { deleteYue2AitkRun(r.jobId); }
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
  return { freedBytes: freed, done };
}
