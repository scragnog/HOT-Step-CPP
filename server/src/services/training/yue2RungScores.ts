// yue2RungScores.ts — Rob's ear scores per refinement rung, kept with the
// rung's facts (step, KL, reconstruction, drift, settings, previews) so a
// later pass can look across artists for what predicts a good planner stop.
//
// One row per checkpoint directory. The client sends only the judgement
// (likeness, corruption, notes); the server fills the facts from the run
// record and meters.json, so rows never carry hand-typed numbers.
import { getDb } from '../../db/database.js';
import { listYue2AitkRuns, type Yue2AitkRunRecord } from './yue2AitkRuns.js';
import { listYue2JointPreviews, type Yue2JointPreviewRecord } from './yue2JointPreview.js';

export interface Yue2RungScore {
  id: number;
  datasetId: string;
  datasetSlug: string;
  sourceRun: string;
  refineRun: string;
  checkpointDir: string;
  step: number;
  kl: number | null;
  recon: number | null;
  drift: number | null;
  rung: boolean;
  frozen: boolean;
  settings: Record<string, unknown>;
  previews: Array<Record<string, unknown>>;
  metrics: Record<string, unknown>;
  likeness: number | null;
  corruption: number | null;
  notes: string;
  blind: boolean;
  blindLabel: string;
  createdAt: string;
  updatedAt: string;
}

function row(r: Record<string, unknown>): Yue2RungScore {
  const json = (v: unknown, fallback: unknown) => { try { return typeof v === 'string' ? JSON.parse(v) : fallback; } catch { return fallback; } };
  return {
    id: r.id as number, datasetId: r.dataset_id as string, datasetSlug: r.dataset_slug as string,
    sourceRun: r.source_run as string, refineRun: r.refine_run as string, checkpointDir: r.checkpoint_dir as string,
    step: r.step as number, kl: r.kl as number | null, recon: r.recon as number | null, drift: r.drift as number | null,
    rung: r.rung === 1, frozen: r.frozen === 1,
    settings: json(r.settings, {}) as Record<string, unknown>, previews: json(r.previews, []) as Array<Record<string, unknown>>, metrics: json(r.metrics, {}) as Record<string, unknown>,
    likeness: r.likeness as number | null, corruption: r.corruption as number | null, notes: (r.notes as string) ?? '',
    blind: r.blind === 1, blindLabel: (r.blind_label as string) ?? '',
    createdAt: r.created_at as string, updatedAt: r.updated_at as string,
  };
}

export function listYue2RungScores(datasetId?: string, refineRun?: string): Yue2RungScore[] {
  const db = getDb();
  const where: string[] = []; const args: unknown[] = [];
  if (datasetId) { where.push('dataset_id = ?'); args.push(datasetId); }
  if (refineRun) { where.push('refine_run = ?'); args.push(refineRun); }
  const rows = db.prepare(`SELECT * FROM yue2_rung_scores${where.length ? ' WHERE ' + where.join(' AND ') : ''} ORDER BY dataset_slug, refine_run, step`).all(...args) as Array<Record<string, unknown>>;
  return rows.map(row);
}

/** Upsert the judgement for one checkpoint of a run; the facts come from the
 *  run's own records. Returns the stored row. */
export function scoreYue2Rung(ds: { id: string; slug: string }, input: { refineRun: string; step: number; likeness?: number | null; corruption?: number | null; notes?: string; blind?: boolean; blindLabel?: string },
  /** A run held elsewhere (a training worker): its record and previews, fetched by the caller. */
  remote?: { run: Yue2AitkRunRecord; previews: Yue2JointPreviewRecord[] }): Yue2RungScore {
  const run = remote ? remote.run : listYue2AitkRuns(ds.id, ds.slug).find(r => r.jobId === input.refineRun);
  if (!run) throw new Error('Unknown refinement run');
  const ckpt = run.checkpoints.find(c => c.step === input.step);
  if (!ckpt) throw new Error(`No checkpoint at step ${input.step}`);
  const o = run.options as Record<string, unknown>;
  const settings = {
    resume: o.resume, targetKl: o.targetKl, targetKlMode: o.targetKlMode, klCheckpointEvery: o.klCheckpointEvery, lr: o.lr,
    plannerLrScale: o.plannerLrScale, narLrScale: o.narLrScale, refineWarmup: o.refineWarmup, rungAdaptiveLr: o.rungAdaptiveLr,
    unfreezePlanner: o.unfreezePlanner, reconStop: o.reconStop, spikeFactor: o.spikeFactor, optimizer: o.optimizer,
    adapterType: o.adapterType, lokrDim: o.lokrDim, lokrFactor: o.lokrFactor, alpha: o.alpha, captionDropout: o.captionDropout, steps: o.steps, saveEvery: o.saveEvery,
    method: o.method, gradAccum: o.gradAccum, calibration: o.calibration,
  };
  const previews = (remote ? remote.previews : listYue2JointPreviews(run.output)).filter(p => p.step === input.step)
    .map(p => ({ id: p.id, kind: p.kind, seed: p.seed, seconds: p.seconds, status: p.status, endReason: p.endReason, verdict: p.score?.verdict, file: p.file,
      // Legibility counts ride along so the ear score and the plan's features
      // sit in one row, whatever later happens to the run folder.
      flags: p.score?.flags, legibility: p.score?.legibility, plannerReplans: p.plan ? p.plan.attempts.length - 1 : undefined, composerReplans: p.composerReplans }));
  const sourceRun = typeof o.resume === 'string' ? o.resume : '';
  const clamp = (v: unknown) => { const n = Number(v); return Number.isInteger(n) && n >= 1 && n <= 5 ? n : null; };
  const db = getDb();
  // Merge in code: a field the client did not send keeps its stored value
  // (a likeness click must not blank the notes), and the row always carries
  // a string for notes (the column is NOT NULL).
  const prior = db.prepare('SELECT likeness, corruption, notes, blind, blind_label FROM yue2_rung_scores WHERE checkpoint_dir = ?').get(ckpt.dir) as { likeness: number | null; corruption: number | null; notes: string; blind: number; blind_label: string } | undefined;
  const likeness = input.likeness === undefined ? (prior?.likeness ?? null) : clamp(input.likeness);
  const corruption = input.corruption === undefined ? (prior?.corruption ?? null) : clamp(input.corruption);
  const notes = input.notes === undefined ? (prior?.notes ?? '') : String(input.notes).slice(0, 4000);
  const rated = input.likeness !== undefined || input.corruption !== undefined;
  if (rated && input.blind === true && (input.blindLabel !== run.blindLabels?.[ckpt.step] || !input.blindLabel)) throw new Error('Blind label does not match this checkpoint');
  const blind = rated ? input.blind === true : prior?.blind === 1;
  const blindLabel = rated ? (blind ? input.blindLabel! : '') : (prior?.blind_label ?? '');
  db.prepare(`INSERT INTO yue2_rung_scores (dataset_id, dataset_slug, source_run, refine_run, checkpoint_dir, step, kl, recon, drift, rung, frozen, settings, previews, metrics, likeness, corruption, notes, blind, blind_label)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(checkpoint_dir) DO UPDATE SET kl = excluded.kl, recon = excluded.recon, drift = excluded.drift, rung = excluded.rung, frozen = excluded.frozen,
      settings = excluded.settings, previews = excluded.previews, metrics = excluded.metrics,
      likeness = excluded.likeness, corruption = excluded.corruption, notes = excluded.notes,
      blind = excluded.blind, blind_label = excluded.blind_label, updated_at = datetime('now')`)
    .run(ds.id, ds.slug, sourceRun, run.jobId, ckpt.dir, ckpt.step, ckpt.kl ?? null, ckpt.recon ?? null, ckpt.drift ?? null, ckpt.rung ? 1 : 0, ckpt.frozen ? 1 : 0,
      JSON.stringify(settings), JSON.stringify(previews), JSON.stringify({}), likeness, corruption, notes, blind ? 1 : 0, blindLabel);
  return row(db.prepare('SELECT * FROM yue2_rung_scores WHERE checkpoint_dir = ?').get(ckpt.dir) as Record<string, unknown>);
}

export function yue2RungScoresCsv(rows: Yue2RungScore[]): string {
  const cols = ['datasetSlug', 'refineRun', 'sourceRun', 'step', 'kl', 'recon', 'drift', 'rung', 'frozen', 'likeness', 'corruption', 'notes', 'blind', 'blind_label', 'settings', 'previews', 'updatedAt'] as const;
  const cell = (v: unknown) => { const s = typeof v === 'string' ? v : v === null || v === undefined ? '' : JSON.stringify(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  return [cols.join(','), ...rows.map(r => cols.map(c => cell(c === 'blind_label' ? r.blindLabel : r[c])).join(','))].join('\n') + '\n';
}

export type Yue2TrainedDirection = 'under' | 'right' | 'over';
export interface Yue2AlbumScore { refineRun: string; datasetId: string; datasetSlug: string; score: number | null;
  /** Which way each half missed: under-trained, right, over-trained (overcooked). */
  instruments: Yue2TrainedDirection | null; vocals: Yue2TrainedDirection | null; notes: string; updatedAt: string }
const direction = (v: unknown): Yue2TrainedDirection | null => v === 'under' || v === 'right' || v === 'over' ? v : null;

/** The listener's verdict on a whole run: how well the album trained, 1-5. */
export function getYue2AlbumScore(refineRun: string): Yue2AlbumScore | null {
  const r = getDb().prepare('SELECT * FROM yue2_album_scores WHERE refine_run = ?').get(refineRun) as Record<string, unknown> | undefined;
  return r ? { refineRun: r.refine_run as string, datasetId: r.dataset_id as string, datasetSlug: r.dataset_slug as string,
    score: r.score as number | null, instruments: direction(r.instruments), vocals: direction(r.vocals), notes: (r.notes as string) ?? '', updatedAt: r.updated_at as string } : null;
}

export function scoreYue2Album(ds: { id: string; slug: string }, input: { refineRun: string; score?: number | null; instruments?: string | null; vocals?: string | null; notes?: string },
  /** The caller has checked the run exists (on a training worker). */
  remoteRun = false): Yue2AlbumScore {
  if (!remoteRun && !listYue2AitkRuns(ds.id, ds.slug).some(r => r.jobId === input.refineRun)) throw new Error('Unknown run for this dataset');
  const prior = getYue2AlbumScore(input.refineRun);
  const n = Number(input.score);
  const score = input.score === undefined ? (prior?.score ?? null) : Number.isInteger(n) && n >= 1 && n <= 5 ? n : null;
  const notes = input.notes === undefined ? (prior?.notes ?? '') : String(input.notes).slice(0, 4000);
  const instruments = input.instruments === undefined ? (prior?.instruments ?? null) : direction(input.instruments);
  const vocals = input.vocals === undefined ? (prior?.vocals ?? null) : direction(input.vocals);
  getDb().prepare(`INSERT INTO yue2_album_scores (refine_run, dataset_id, dataset_slug, score, instruments, vocals, notes) VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(refine_run) DO UPDATE SET score = excluded.score, instruments = excluded.instruments, vocals = excluded.vocals, notes = excluded.notes, updated_at = datetime('now')`)
    .run(input.refineRun, ds.id, ds.slug, score, instruments, vocals, notes);
  return getYue2AlbumScore(input.refineRun)!;
}

/** Rows scored on a training worker before scores were kept here: inserted
 *  as they are, never over a row this machine already holds. */
export function importYue2RungScores(rows: Yue2RungScore[]): number {
  const db = getDb();
  const ins = db.prepare(`INSERT INTO yue2_rung_scores (dataset_id, dataset_slug, source_run, refine_run, checkpoint_dir, step, kl, recon, drift, rung, frozen, settings, previews, metrics, likeness, corruption, notes, blind, blind_label, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(checkpoint_dir) DO NOTHING`);
  let n = 0;
  for (const r of rows) {
    if (!r?.checkpointDir || !r.refineRun) continue;
    n += ins.run(r.datasetId, r.datasetSlug, r.sourceRun ?? '', r.refineRun, r.checkpointDir, r.step, r.kl ?? null, r.recon ?? null, r.drift ?? null, r.rung ? 1 : 0, r.frozen ? 1 : 0,
      JSON.stringify(r.settings ?? {}), JSON.stringify(r.previews ?? []), JSON.stringify(r.metrics ?? {}), r.likeness ?? null, r.corruption ?? null, r.notes ?? '', r.blind ? 1 : 0, r.blindLabel ?? '',
      r.createdAt ?? null, r.updatedAt ?? null).changes;
  }
  return n;
}
