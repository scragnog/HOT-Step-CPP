// training/mm3Review.ts — the listening test on an MM3 run's checkpoint ladder,
// the same one YuE2 has: blind rung letters, likeness/corruption scores and
// notes per checkpoint, one album verdict per run, and the cleanup after a
// checkpoint is chosen (the run's other checkpoints, their preview audio, the
// run's resume state, and the dataset's other prepared caches).
//
// Scores live in their own tables (mm3_rung_scores, mm3_album_scores), not
// YuE2's: Dataset-Calibrated Training and the rung-score export read those
// unfiltered. Blind letters live in the run's hotstep-run.json.
//
// Cleanup never leaves the chosen run's folder, so there is no "other runs"
// choice here; the routes (mm3Ladder.ts, ops/mm3) hold the dataset lock and the
// stale checks.

import fs from 'fs';
import path from 'path';
import { getDb } from '../../db/database.js';
import { listMm3RunPreviews, MM3_PREVIEW_DEFAULTS } from './mm3Preview.js';
import { mm3RunManifestPath, readMm3RunManifest, type Mm3RunSummary } from './mm3Runs.js';
import { listPreparedCaches } from './preparedDataReset.js';
import { dirBytes, type Yue2CleanupChoice, type Yue2CleanupItem } from './yue2Cleanup.js';
import { withBlindLabels } from './yue2AitkRuns.js';
import type { TrainingPreview } from './types.js';

type Ds = { id: string; slug: string };

// ── blind letters ───────────────────────────────────────────────────────────

/** The run's blind letter per checkpoint step. New checkpoints get theirs
 *  here, and the run's manifest keeps them, so a letter never moves. A run
 *  from before the manifest gets letters that are not saved. */
export function mm3BlindLabels(run: Pick<Mm3RunSummary, 'runName' | 'dir' | 'checkpoints'>): Record<string, string> {
  const manifest = readMm3RunManifest(run.dir);
  const added = withBlindLabels(run.runName, manifest?.blindLabels, run.checkpoints.map(c => c.step));
  if (!added) return manifest?.blindLabels ?? {};
  if (manifest) {
    try { fs.writeFileSync(mm3RunManifestPath(run.dir), JSON.stringify({ ...manifest, blindLabels: added }, null, 2), 'utf-8'); }
    catch (err: any) { console.warn(`[Training] MM3 blind labels for ${run.runName} not saved: ${err?.message || err}`); }
  }
  return added;
}

// ── scores ──────────────────────────────────────────────────────────────────

export interface Mm3RungScore {
  run: string; step: number; checkpointDir: string; loss: number | null;
  likeness: number | null; corruption: number | null; notes: string;
  blind: boolean; blindLabel: string; updatedAt: string;
}
export type Mm3TrainedDirection = 'under' | 'right' | 'over';
export interface Mm3AlbumScore {
  run: string; score: number | null; instruments: Mm3TrainedDirection | null; vocals: Mm3TrainedDirection | null;
  notes: string; updatedAt: string;
}

const rungRow = (r: Record<string, unknown>): Mm3RungScore => ({
  run: r.run as string, step: r.step as number, checkpointDir: r.checkpoint_dir as string, loss: r.loss as number | null,
  likeness: r.likeness as number | null, corruption: r.corruption as number | null, notes: (r.notes as string) ?? '',
  blind: r.blind === 1, blindLabel: (r.blind_label as string) ?? '', updatedAt: r.updated_at as string,
});
const clamp = (v: unknown) => { const n = Number(v); return Number.isInteger(n) && n >= 1 && n <= 5 ? n : null; };
const direction = (v: unknown): Mm3TrainedDirection | null => v === 'under' || v === 'right' || v === 'over' ? v : null;
const note = (v: unknown) => String(v).slice(0, 4000);

export function listMm3RungScores(datasetId: string, runName: string): Mm3RungScore[] {
  return (getDb().prepare('SELECT * FROM mm3_rung_scores WHERE dataset_id = ? AND run = ? ORDER BY step')
    .all(datasetId, runName) as Array<Record<string, unknown>>).map(rungRow);
}

/** Every saved MM3 rung score, for GET /mm3-rung-scores/export. */
export function listAllMm3RungScores(): Array<Mm3RungScore & { datasetSlug: string }> {
  return (getDb().prepare('SELECT * FROM mm3_rung_scores ORDER BY dataset_slug, run, step').all() as Array<Record<string, unknown>>)
    .map(r => ({ ...rungRow(r), datasetSlug: (r.dataset_slug as string) ?? '' }));
}

export function mm3RungScoresCsv(rows: ReturnType<typeof listAllMm3RungScores>): string {
  const cols = ['datasetSlug', 'run', 'step', 'loss', 'likeness', 'corruption', 'notes', 'blind', 'blindLabel', 'updatedAt'] as const;
  const cell = (v: unknown) => { const s = typeof v === 'string' ? v : v === null || v === undefined ? '' : JSON.stringify(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  return [cols.join(','), ...rows.map(r => cols.map(c => cell(r[c])).join(','))].join('\n') + '\n';
}

/** Upsert the judgement on one checkpoint. A field the client did not send
 *  keeps its stored value; a blind rating must name the step's own letter. */
export function scoreMm3Rung(ds: Ds, run: Pick<Mm3RunSummary, 'runName' | 'dir' | 'checkpoints'>,
  input: { step: number; likeness?: number | null; corruption?: number | null; notes?: string; blind?: boolean; blindLabel?: string }): Mm3RungScore {
  const ckpt = run.checkpoints.find(c => c.step === input.step);
  if (!ckpt) throw new Error(`No checkpoint at step ${input.step}`);
  const db = getDb();
  const prior = db.prepare('SELECT * FROM mm3_rung_scores WHERE checkpoint_dir = ?').get(ckpt.dir) as Record<string, unknown> | undefined;
  const was = prior ? rungRow(prior) : null;
  const rated = input.likeness !== undefined || input.corruption !== undefined;
  if (rated && input.blind === true && (!input.blindLabel || input.blindLabel !== mm3BlindLabels(run)[ckpt.step]))
    throw new Error('Blind label does not match this checkpoint');
  const blind = rated ? input.blind === true : !!was?.blind;
  db.prepare(`INSERT INTO mm3_rung_scores (checkpoint_dir, dataset_id, dataset_slug, run, step, loss, likeness, corruption, notes, blind, blind_label)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(checkpoint_dir) DO UPDATE SET loss = excluded.loss, likeness = excluded.likeness, corruption = excluded.corruption,
      notes = excluded.notes, blind = excluded.blind, blind_label = excluded.blind_label, updated_at = datetime('now')`)
    .run(ckpt.dir, ds.id, ds.slug, run.runName, ckpt.step, ckpt.loss ?? null,
      input.likeness === undefined ? (was?.likeness ?? null) : clamp(input.likeness),
      input.corruption === undefined ? (was?.corruption ?? null) : clamp(input.corruption),
      input.notes === undefined ? (was?.notes ?? '') : note(input.notes),
      blind ? 1 : 0, rated ? (blind ? input.blindLabel! : '') : (was?.blindLabel ?? ''));
  return rungRow(db.prepare('SELECT * FROM mm3_rung_scores WHERE checkpoint_dir = ?').get(ckpt.dir) as Record<string, unknown>);
}

/** Drop a deleted run's rung and album scores. */
export function deleteMm3RunScores(datasetId: string, runName: string): void {
  const db = getDb();
  db.prepare('DELETE FROM mm3_rung_scores WHERE dataset_id = ? AND run = ?').run(datasetId, runName);
  db.prepare('DELETE FROM mm3_album_scores WHERE run = ?').run(runName);
}

export function getMm3AlbumScore(runName: string): Mm3AlbumScore | null {
  const r = getDb().prepare('SELECT * FROM mm3_album_scores WHERE run = ?').get(runName) as Record<string, unknown> | undefined;
  return r ? { run: r.run as string, score: r.score as number | null, instruments: direction(r.instruments), vocals: direction(r.vocals),
    notes: (r.notes as string) ?? '', updatedAt: r.updated_at as string } : null;
}

export function scoreMm3Album(ds: Ds, runName: string,
  input: { score?: number | null; instruments?: string | null; vocals?: string | null; notes?: string }): Mm3AlbumScore {
  const prior = getMm3AlbumScore(runName);
  getDb().prepare(`INSERT INTO mm3_album_scores (run, dataset_id, dataset_slug, score, instruments, vocals, notes) VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(run) DO UPDATE SET score = excluded.score, instruments = excluded.instruments, vocals = excluded.vocals, notes = excluded.notes, updated_at = datetime('now')`)
    .run(runName, ds.id, ds.slug,
      input.score === undefined ? (prior?.score ?? null) : clamp(input.score),
      input.instruments === undefined ? (prior?.instruments ?? null) : direction(input.instruments),
      input.vocals === undefined ? (prior?.vocals ?? null) : direction(input.vocals),
      input.notes === undefined ? (prior?.notes ?? '') : note(input.notes));
  return getMm3AlbumScore(runName)!;
}

// ── preview audio removed by cleanup ────────────────────────────────────────

/** Takes whose audio cleanup removed: their sidecars stay, marked pruned, so
 *  the ladder can say what was there. */
export function prunedMm3Previews(previewsDir: string): Array<TrainingPreview & { pruned: true }> {
  let names: string[];
  try { names = fs.readdirSync(previewsDir); } catch { return []; }
  return names.filter(n => n.endsWith('.wav.json') && !names.includes(n.slice(0, -5))).flatMap(n => {
    try {
      const rec = JSON.parse(fs.readFileSync(path.join(previewsDir, n), 'utf-8')) as TrainingPreview & { pruned?: boolean };
      return rec.pruned ? [{ ...rec, file: n.slice(0, -5), bytes: 0, pruned: true as const }] : [];
    } catch { return []; }
  });
}

// ── cleanup ─────────────────────────────────────────────────────────────────

export type Mm3CleanupPlan = { run: string; step: number; keep: string;
  caches: Yue2CleanupItem; otherCheckpoints: Yue2CleanupItem; resume: Yue2CleanupItem; otherPreviews: Yue2CleanupItem };
export type Mm3CleanupChoice = Omit<Yue2CleanupChoice, 'otherRuns'>;

/** MM3's codes are kept, as YuE2 keeps its core: an album's codes take GPU
 *  time to rebuild and every later MM3 run reads them. */
const keepsCache = (name: string) => name.startsWith('mm3-codes');
const RESUME_FILES = ['resume-state.bin', 'resume-state.json'];

function inside(root: string, p: string): boolean {
  const rel = path.relative(path.resolve(root), path.resolve(p));
  return !!rel && !rel.startsWith('..') && !path.isAbsolute(rel);
}

/** Throw unless `p` (when it exists) is no link or junction and really lies
 *  inside the run folder. Cleanup deletes and writes these paths, so a
 *  `previews` junction to another run must stop it before anything changes. */
function assertOwn(runReal: string, p: string): void {
  let stat: fs.Stats;
  try { stat = fs.lstatSync(p); } catch { return; }
  if (stat.isSymbolicLink() || !inside(runReal, fs.realpathSync.native(p))) throw new Error(`Refusing to clean up a path outside the run folder: ${p}`);
}

function parts(ds: Ds & { sourceDir: string }, run: Pick<Mm3RunSummary, 'runName' | 'dir' | 'checkpoints'>, step: number) {
  const keep = run.checkpoints.find(c => c.step === step);
  if (!keep) throw new Error(`No checkpoint at step ${step}`);
  if (fs.lstatSync(run.dir).isSymbolicLink()) throw new Error(`Refusing to clean up a linked run folder: ${run.dir}`);
  const runReal = fs.realpathSync.native(run.dir);
  const others = run.checkpoints.filter(c => c.step !== step);
  // checkpointsIn lists the run's own ckpt-<step> folders; checked again,
  // through links, because this deletes them.
  for (const c of others) { if (!inside(run.dir, c.dir)) throw new Error('A checkpoint lies outside its run folder'); assertOwn(runReal, c.dir); }
  const p = readMm3RunManifest(run.dir)?.options.preview;
  const previewsDir = path.join(run.dir, 'previews');
  assertOwn(runReal, previewsDir);
  // The base takes stay: they are the reference every rung is heard against.
  const previews = listMm3RunPreviews(previewsDir, { seconds: p?.seconds ?? MM3_PREVIEW_DEFAULTS.seconds, seed: p?.seed ?? MM3_PREVIEW_DEFAULTS.seed })
    .filter(v => !v.base && v.step !== step);
  for (const v of previews) {
    const wav = path.join(previewsDir, v.file);
    for (const f of [wav, `${wav}.json`, wav.replace(/\.wav$/i, '.mp3')]) assertOwn(runReal, f);
  }
  const resume = RESUME_FILES.map(f => path.join(run.dir, f)).filter(f => fs.existsSync(f));
  for (const f of resume) assertOwn(runReal, f);
  const caches = listPreparedCaches(ds.slug, ds.sourceDir).filter(c => !keepsCache(c.name));
  return { keep, others, previewsDir, previews, resume, caches };
}

export function planMm3Cleanup(ds: Ds & { sourceDir: string }, run: Pick<Mm3RunSummary, 'runName' | 'dir' | 'checkpoints'>, step: number): Mm3CleanupPlan {
  const { keep, others, previews, resume, caches } = parts(ds, run, step);
  return {
    run: run.runName, step, keep: keep.dir,
    caches: { count: caches.length, bytes: caches.reduce((s, c) => s + c.bytes, 0), detail: caches.map(c => c.name) },
    otherCheckpoints: { count: others.length, bytes: others.reduce((s, c) => s + dirBytes(c.dir), 0), detail: others.map(c => c.name) },
    resume: { count: resume.length ? 1 : 0, bytes: resume.reduce((s, f) => s + fs.statSync(f).size, 0) },
    otherPreviews: { count: previews.length, bytes: previews.reduce((s, v) => s + v.bytes, 0) },
  };
}

/** Delete what `choice` names. The caller holds the dataset's review lock and
 *  has checked the chosen step is still the linked adapter. */
export function runMm3Cleanup(ds: Ds & { sourceDir: string }, run: Pick<Mm3RunSummary, 'runName' | 'dir' | 'checkpoints'>, step: number,
  choice: Mm3CleanupChoice): { freedBytes: number; done: string[] } {
  const plan = planMm3Cleanup(ds, run, step);
  const { others, previewsDir, previews, resume, caches } = parts(ds, run, step);
  let freed = 0; const done: string[] = [];
  if (choice.otherCheckpoints && others.length) {
    for (const c of others) fs.rmSync(c.dir, { recursive: true, force: true });
    freed += plan.otherCheckpoints.bytes; done.push(`${others.length} other checkpoint(s)`);
  }
  if (choice.otherPreviews && previews.length) {
    for (const v of previews) {
      const wav = path.join(previewsDir, v.file);
      const { audioUrl: _url, ...rec } = v as TrainingPreview & { audioUrl?: string };
      fs.writeFileSync(`${wav}.json`, JSON.stringify({ ...rec, pruned: true }, null, 2), 'utf-8');
      fs.rmSync(wav, { force: true });
      fs.rmSync(wav.replace(/\.wav$/i, '.mp3'), { force: true });
    }
    freed += plan.otherPreviews.bytes; done.push(`${previews.length} other preview(s)`);
  }
  if (choice.resume && resume.length) {
    for (const f of resume) fs.rmSync(f, { force: true });
    freed += plan.resume.bytes; done.push('resume state');
  }
  if (choice.caches && caches.length) {
    // listPreparedCaches has walked them (no links, nothing inside the source).
    for (const c of caches) fs.rmSync(c.path, { recursive: true, force: false });
    freed += plan.caches.bytes; done.push(`${caches.length} cache folder(s)`);
  }
  return { freedBytes: freed, done };
}
