import assert from 'node:assert/strict';
import { test } from 'node:test';
import { chartMeans, klReading, reduceYue2Metric, smoothBest } from './yue2ChartData';
import type { TrainingMetricEvent } from '@/services/trainingApi';

const metric = (step: number, fields: Partial<TrainingMetricEvent> = {}): TrainingMetricEvent => ({ type: 'metric', metric: 'step', ts: step, step, ...fields });

test('smoothed best uses a full trailing window and marks its centre', () => {
  assert.deepEqual(chartMeans([3, 2, 4, 1], 3), [3, 2.5, 3, 7 / 3]);
  const points = [3, 2, 4, 1].map((loss, i) => ({ ep: (i + 1) * 10, loss }));
  assert.equal(smoothBest(points, 3), 30);
  assert.equal(smoothBest(points, 3, 31), undefined);
  assert.equal(smoothBest(points.slice(0, 2), 3), undefined);
});

test('SSE replay and reconnect deduplicate steps and keep sparse checkpoint meters', () => {
  const events = [metric(1, { loss: 4, arCe: 2, arKl: .1, narMse: 1 }), metric(2, { loss: 3, arCe: 1, arKl: .2 }), metric(2, { narRecon: .4, narDrift: .3 })];
  const first = events.reduce((p, e) => reduceYue2Metric(p, e), []);
  assert.equal(first.length, 2);
  assert.equal(first[1].narRecon, .4);
  assert.equal(first[1].narDrift, .3);
  assert.equal(first[1].arCe, 1);
  assert.deepEqual(events.reduce((p, e) => reduceYue2Metric(p, e), first), first);
  const reverse = [...events].reverse().reduce((p, e) => reduceYue2Metric(p, e), []);
  assert.deepEqual(reverse, first);
});

test('one recon point survives, frozen continuation leaves planner fields absent', () => {
  let points = reduceYue2Metric([], metric(5, { narRecon: .4 }));
  assert.equal(points.length, 1);
  assert.equal(points[0].loss, undefined);
  points = reduceYue2Metric(points, metric(5, { plannerFrozen: true, narMse: .3, gradNorm: 2, stepMs: 100 }));
  assert.equal(points[0].narRecon, .4);
  assert.equal(points[0].narMse, .3);
  assert.equal(points[0].frozen, true);
  assert.equal(points[0].arCe, undefined);
  assert.equal(points[0].ma5, undefined);
});

test('old runs do not acquire missing fields; nonfinite fields are ignored', () => {
  const p = reduceYue2Metric([], metric(1, { loss: 2, arCe: NaN }))[0];
  assert.equal(p.loss, 2);
  assert.equal(p.arCe, undefined);
  assert.equal(p.arKl, undefined);
  assert.equal(p.narRecon, undefined);
});

test('KL uses configured mean window and Refine partial mean from five readings', () => {
  assert.equal(klReading([1, 2, 3], 'mean', 3), 2);
  assert.equal(klReading([1, 2, 3, 4], 'trend', 20, true), undefined);
  assert.equal(klReading([1, 2, 3, 4, 5], 'trend', 20, true), 3);
  assert.equal(klReading(Array.from({ length: 30 }, (_, i) => i), 'trend'), 29);
  assert.equal(klReading([1, 2, 3, 4, 5], 'mean'), undefined);
});
