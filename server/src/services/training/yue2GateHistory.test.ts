import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseGateHistory } from './yue2GateHistory.js';
import fs from 'node:fs';
import path from 'node:path';

// A noisy held-out series and explicit freeze, also usable as a chart fixture.
const gateFixture = () => [3.9, 3.2, 3.5, 3.1, 3.3, 3.7].map((ce, i) => JSON.stringify({
  stage: 'planner_gate', step: (i + 1) * 10, epoch: i + 1, ce, best: 3.1,
  best_step: 40, stale: 0, patience: 3, armed: true, items: 3, tokens: 100, ms: 10,
})).concat(JSON.stringify({ stage: 'planner_frozen', step: 60, reason: 'planner_ce' })).join('\n');

test('gate history merges resumed segments, ignores malformed lines and preserves raw values', () => {
  const result = parseGateHistory([gateFixture(), '{broken\n' + JSON.stringify({ stage: 'joint', step: 61, nar_mse: .8, planner_frozen: true })]);
  assert.equal(result.readings.length, 6);
  assert.equal(result.readings[1].loss, 3.2);
  assert.equal(result.frozenAt, 60);
  assert.equal(result.steps[0].narMse, .8);
  assert.deepEqual(parseGateHistory(['']).readings, []);
});

test('freeze can be recovered from decoder-only steps in the saved log', () => {
  const line = (step: number, planner_frozen: boolean) => JSON.stringify({ stage: 'joint', step, nar_mse: 1, planner_frozen });
  assert.equal(parseGateHistory([[line(30, false), line(31, true)].join('\n')]).frozenAt, 30);
  assert.equal(parseGateHistory([line(31, true)]).frozenAt, undefined);
  assert.equal(parseGateHistory([[line(31, true), line(32, true)].join('\n')]).frozenAt, undefined);
});

test('write an optional synthetic chart fixture', { skip: !process.env.HOTSTEP_1319_FIXTURE }, () => {
  const dir = process.env.HOTSTEP_1319_FIXTURE!;
  const source = process.env.HOTSTEP_1319_SOURCE!;
  const target = process.env.HOTSTEP_1319_CHART_JSON!;
  fs.mkdirSync(dir, { recursive: false });
  const file = path.join(dir, 'train.jsonl');
  fs.copyFileSync(source, file, fs.constants.COPYFILE_EXCL);
  fs.appendFileSync(file, '\n' + gateFixture() + '\n');
  fs.writeFileSync(target, JSON.stringify(parseGateHistory([fs.readFileSync(file, 'utf8')])));
});
