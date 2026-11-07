import assert from 'node:assert/strict';
import test from 'node:test';
import { formatDuration, previewsFromRows, trainingRunEta } from './trainingRunEta.ts';

const take = (kind: string, step: number, renderMs: number | null, pauseMs: number | null) =>
  ({ kind, step, status: 'done', renderMs, pauseMs });

test('steps left plus the previews still to come, costed by the base take until a mid-run one exists', () => {
  // 720 steps, previews every 100, at step 150: takes at 200..700 (6) plus the final = 7.
  const early = trainingRunEta({ step: 150, totalSteps: 720, msPerStep: 5000, everySteps: 100, previews: [take('base', 0, 40_000, null)] });
  assert.deepEqual(early, { ms: 570 * 5000 + 7 * 40_000, previewsLeft: 7, previewMs: 40_000 });
  // After a mid-run take its pause (engine restart, render, reload) is the cost.
  const later = trainingRunEta({ step: 250, totalSteps: 720, msPerStep: 5000, everySteps: 100,
    previews: [take('base', 0, 40_000, null), take('step', 100, 41_000, 90_000), take('step', 200, 40_000, 95_000)] });
  assert.equal(later?.previewMs, 95_000);
  assert.equal(later?.previewsLeft, 6);
  // A total that is itself a multiple of the cadence: no extra checkpoint take, just the final.
  assert.equal(trainingRunEta({ step: 450, totalSteps: 500, msPerStep: 1, everySteps: 100, previews: [] })?.previewsLeft, 1);
});

test('no previews, no step time yet, or done', () => {
  assert.equal(trainingRunEta({ step: 10, totalSteps: 100, msPerStep: 1000, everySteps: 0, previews: [] })?.ms, 90_000);
  assert.equal(trainingRunEta({ step: 0, totalSteps: 100, msPerStep: null, everySteps: 0, previews: [] }), null);
  assert.equal(trainingRunEta({ step: 100, totalSteps: 100, msPerStep: 1000, everySteps: 0, previews: [] }), null);
  assert.equal(formatDuration(65 * 60_000), '1 h 05 min');
  assert.equal(formatDuration(20_000), '<1 min');
});

test('MM3 rows fold to one entry per checkpoint: renders add up, the pause rides once', () => {
  const rows = [
    { step: 0, base: true, ms: 20_000 }, { step: 0, base: true, ms: 25_000 },
    { step: 100, base: false, ms: 20_000 }, { step: 100, base: false, ms: 22_000, pauseMs: 120_000 },
  ];
  const previews = previewsFromRows(rows);
  assert.deepEqual(previews.map(p => [p.kind, p.step, p.renderMs, p.pauseMs]), [['base', 0, 45_000, null], ['step', 100, 42_000, 120_000]]);
  // Before the first pause the base take stands in; after it, the pause.
  const input = { step: 150, totalSteps: 500, msPerStep: 1000, everySteps: 100 };
  assert.equal(trainingRunEta({ ...input, previews: previews.slice(0, 1) })?.previewMs, 45_000);
  assert.equal(trainingRunEta({ ...input, previews })?.previewMs, 120_000);
});
