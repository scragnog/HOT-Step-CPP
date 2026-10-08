import assert from 'node:assert/strict';
import test from 'node:test';
import { bestReviewFact, reviewRevision, reviewRungFacts, reviewSelectionError, rungOverall } from './yue2BestRung.js';
import type { Yue2AitkRunRecord } from './yue2AitkRuns.js';
import type { Yue2RungScore } from './yue2RungScores.js';
import type { Yue2JointPreviewRecord } from './yue2JointPreview.js';

test('rungOverall matches the Refine scoreboard formula', () => {
  assert.equal(rungOverall(5, 1, 0, 2), 5);           // (5 + 5) / 2, no replans
  assert.equal(rungOverall(4, 2, 2, 2), 3.75);        // 4 − 0.25 × (2 replans / 2 takes)
  assert.equal(rungOverall(4, 2, 40, 2), 3);          // penalty capped at 1
  assert.equal(rungOverall(4, null, 0, 2), null);     // needs both scores
  assert.equal(rungOverall(4, 2, 0, 2, 2, 1), 3.8);    // 2 flags on the one own-planned take: 0.1 × 2
  assert.equal(rungOverall(4, 2, 2, 2, 20, 1), 3);     // replans + flags capped at 1 together
});

test('review facts preserve null scores, penalties, blind labels and earlier-step ties', () => {
  const run = { jobId: 'run', updatedAt: 1, blindLabels: { 10: 'B', 20: 'A', 30: 'C' },
    checkpoints: [10, 20, 30].map(step => ({ step, dir: `step-${step}`, arPath: `ar-${step}`, narPath: `nar-${step}` })) } as unknown as Yue2AitkRunRecord;
  const scores = [
    { step: 10, likeness: 4, corruption: 2, updatedAt: 'one' },
    { step: 20, likeness: 4, corruption: 2, updatedAt: 'two' },
    { step: 30, likeness: 5, corruption: null, updatedAt: 'three' },
  ] as Yue2RungScore[];
  const previews = [
    { id: 'p10', step: 10, status: 'done', sheet: 'own', score: { flags: ['flag'] },
      composerReplans: 1, plan: { attempts: [{}, {}] } },
    { id: 'p20', step: 20, status: 'done', sheet: 'shared', score: { flags: ['ignored'] },
      composerReplans: 2 },
  ] as unknown as Yue2JointPreviewRecord[];
  const facts = reviewRungFacts(run, scores, previews);
  assert.equal(facts[0].overall, 3.4); // 4 minus 0.5 replans and 0.1 own-plan flag.
  assert.equal(facts[1].overall, 3.5); // Shared-sheet flags do not count.
  assert.equal(facts[2].overall, null); // One human axis is missing.
  assert.equal(bestReviewFact(facts)?.step, 20);
  assert.equal(facts[1].blindLabel, 'A');
  const revision = reviewRevision(run, scores, previews);
  assert.equal(reviewSelectionError(facts, revision, { step: 20, checkpointDir: 'step-20', reviewRevision: revision }, true), null);
  assert.match(reviewSelectionError(facts, revision, { step: 10, checkpointDir: 'step-10', reviewRevision: revision }, true)!, /best-scored/);
  assert.match(reviewSelectionError(facts, revision, { step: 20, checkpointDir: 'step-20', reviewRevision: 'old' }, true)!, /changed/);
  assert.match(reviewSelectionError(facts, revision, { step: 30, checkpointDir: 'step-30', reviewRevision: revision }, true)!, /human score/);
  assert.equal(reviewSelectionError(facts, revision, { step: 30, checkpointDir: 'step-30', reviewRevision: revision }, false), null);
  assert.notEqual(reviewRevision(run, [{ ...scores[0], likeness: 5 }, ...scores.slice(1)], previews), revision);
});

test('equal overall scores choose the earlier checkpoint', () => {
  assert.equal(bestReviewFact([
    { step: 10, dir: 'a', blindLabel: '', likeness: 4, corruption: 2, overall: 4, replansPerTake: 0, previewCount: 1, planFlags: 0 },
    { step: 20, dir: 'b', blindLabel: '', likeness: 4, corruption: 2, overall: 4, replansPerTake: 0, previewCount: 1, planFlags: 0 },
  ])?.step, 10);
});