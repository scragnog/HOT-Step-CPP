// yue2BestRung.ts — the Refine scoreboard's "best by score" rung, server side,
// so a batch can finish a scored ladder without the page open.
//
// rungOverall mirrors ui/src/components/training-studio/RefinePanel.tsx
// (rungOverall + rungStats). Change both together.

import { createHash } from 'node:crypto';
import { listYue2AitkRuns, type Yue2AitkRunRecord } from './yue2AitkRuns.js';
import { listYue2JointPreviews, type Yue2JointPreviewRecord } from './yue2JointPreview.js';
import { listYue2RungScores, type Yue2RungScore } from './yue2RungScores.js';

export function rungOverall(likeness: number | null | undefined, corruption: number | null | undefined, replans: number, takes: number,
  planFlags = 0, ownTakes = takes): number | null {
  if (typeof likeness !== 'number' || typeof corruption !== 'number') return null;
  const base = (likeness + (6 - corruption)) / 2;
  const penalty = Math.min(1, 0.25 * (replans / Math.max(1, takes)) + 0.1 * planFlags / Math.max(1, ownTakes));
  return Math.round((base - penalty) * 100) / 100;
}

export interface ReviewRungFact {
  step: number;
  dir: string;
  blindLabel: string;
  likeness: number | null;
  corruption: number | null;
  overall: number | null;
  replansPerTake: number;
  previewCount: number;
  planFlags: number;
}

/** One calculation for the scoreboard, best-rung selection and finish gate. */
export function reviewRungFacts(run: Yue2AitkRunRecord, scores: readonly Yue2RungScore[], previews: readonly Yue2JointPreviewRecord[]): ReviewRungFact[] {
  const byStep = new Map(scores.map(score => [score.step, score]));
  const done = previews.filter(preview => preview.status === 'done');
  return run.checkpoints.filter(checkpoint => checkpoint.arPath && checkpoint.narPath).sort((a, b) => a.step - b.step).map(checkpoint => {
    const takes = done.filter(preview => preview.step === checkpoint.step);
    const replans = takes.reduce((sum, preview) => sum + (preview.plan ? preview.plan.attempts.length - 1 : 0)
      + (typeof preview.composerReplans === 'number' ? preview.composerReplans : 0), 0);
    const own = takes.filter(preview => preview.sheet !== 'shared');
    const planFlags = own.reduce((sum, preview) => sum + (preview.score?.flags?.length ?? 0), 0);
    const score = byStep.get(checkpoint.step);
    return { step: checkpoint.step, dir: checkpoint.dir, blindLabel: run.blindLabels?.[checkpoint.step] ?? '',
      likeness: score?.likeness ?? null, corruption: score?.corruption ?? null,
      overall: rungOverall(score?.likeness, score?.corruption, replans, takes.length, planFlags, own.length),
      replansPerTake: replans / Math.max(1, takes.length), previewCount: takes.length, planFlags };
  });
}

/** A token for the exact run, judgements and previews the choice used. */
export function reviewRevision(run: Yue2AitkRunRecord, scores: readonly Yue2RungScore[], previews: readonly Yue2JointPreviewRecord[]): string {
  const facts = reviewRungFacts(run, scores, previews);
  return createHash('sha256').update(JSON.stringify({ run: run.jobId, updatedAt: run.updatedAt,
    checkpoints: run.checkpoints.map(c => [c.step, c.dir, c.arPath, c.narPath]),
    scores: scores.map(s => [s.step, s.likeness, s.corruption, s.updatedAt]),
    previews: previews.map(p => [p.id, p.status, p.step, p.plan, p.composerReplans, p.score?.flags, p.sheet]), facts })).digest('hex');
}

export function bestReviewFact(facts: readonly ReviewRungFact[]): ReviewRungFact | null {
  let best: ReviewRungFact | null = null;
  for (const fact of facts) if (fact.overall !== null && (!best || fact.overall > best.overall!)) best = fact;
  return best;
}

export function reviewSelectionError(facts: readonly ReviewRungFact[], revision: string,
  pick: { step: number; checkpointDir: string; reviewRevision: string }, requireBestScore: boolean): string | null {
  if (revision !== pick.reviewRevision) return 'Review changed since this rung was selected';
  const fact = facts.find(item => item.step === pick.step && item.dir === pick.checkpointDir);
  if (!fact) return 'The selected checkpoint changed';
  if (requireBestScore && (fact.overall === null || bestReviewFact(facts)?.step !== pick.step))
    return 'The best-scored rung changed or has no complete human score';
  return null;
}
/** The highest-scoring complete checkpoint; ties go to the earlier step. */
export function bestScoredRung(datasetId: string, runId: string, datasetSlug?: string): { step: number; dir: string; overall: number } | null {
  const run = listYue2AitkRuns(datasetId, datasetSlug).find(r => r.jobId === runId);
  if (!run) return null;
  const best = bestReviewFact(reviewRungFacts(run, listYue2RungScores(datasetId, runId), listYue2JointPreviews(run.output)));
  return best?.overall === null || !best ? null : { step: best.step, dir: best.dir, overall: best.overall };
}