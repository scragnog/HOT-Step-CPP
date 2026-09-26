// yue2BestRung.ts — the Refine scoreboard's "best by score" rung, server side,
// so a batch can finish a scored ladder without the page open.
//
// rungOverall mirrors ui/src/components/training-studio/RefinePanel.tsx
// (rungOverall + rungStats). Change both together.

import { listYue2AitkRuns } from './yue2AitkRuns.js';
import { listYue2JointPreviews } from './yue2JointPreview.js';
import { listYue2RungScores } from './yue2RungScores.js';

export function rungOverall(likeness: number | null | undefined, corruption: number | null | undefined, replans: number, takes: number,
  planFlags = 0, ownTakes = takes): number | null {
  if (typeof likeness !== 'number' || typeof corruption !== 'number') return null;
  const base = (likeness + (6 - corruption)) / 2;
  const penalty = Math.min(1, 0.25 * (replans / Math.max(1, takes)) + 0.1 * planFlags / Math.max(1, ownTakes));
  return Math.round((base - penalty) * 100) / 100;
}

/** The highest-scoring complete checkpoint of a ladder run; ties go to the
 *  lower step. null when no checkpoint has both scores. */
export function bestScoredRung(datasetId: string, runId: string, datasetSlug?: string): { step: number; dir: string; overall: number } | null {
  const run = listYue2AitkRuns(datasetId, datasetSlug).find(r => r.jobId === runId);
  if (!run) return null;
  const scores = new Map(listYue2RungScores(datasetId, runId).map(s => [s.step, s]));
  const previews = listYue2JointPreviews(run.output).filter(p => p.status === 'done');
  let best: { step: number; dir: string; overall: number } | null = null;
  for (const c of run.checkpoints.filter(c => c.arPath && c.narPath).sort((a, b) => a.step - b.step)) {
    const takes = previews.filter(p => p.step === c.step);
    const replans = takes.reduce((sum, p) => sum + (p.plan ? p.plan.attempts.length - 1 : 0)
      + (typeof p.composerReplans === 'number' ? p.composerReplans : 0), 0);
    const s = scores.get(c.step);
    const own = takes.filter(p => p.sheet !== 'shared');
    const flags = own.reduce((sum, p) => sum + (p.score?.flags?.length ?? 0), 0);
    const overall = rungOverall(s?.likeness, s?.corruption, replans, takes.length, flags, own.length);
    if (overall !== null && (!best || overall > best.overall)) best = { step: c.step, dir: c.dir, overall };
  }
  return best;
}
