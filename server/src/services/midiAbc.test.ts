import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { midiAbcArgs } from './midiAbc.js';

test('ABC export consumes raw events and the beat grid, with each export option once', () => {
  const args = midiAbcArgs('job', { tempoMode: 'map', quantize: true, grid: 4 }, 100, true);
  assert.equal(args[args.indexOf('--export') + 1], path.join('job', 'events.jsonl'));
  assert.equal(args[args.indexOf('--beats-in') + 1], path.join('job', 'beats.json'));
  assert.equal(args.filter(a => a === '--quantize').length, 1);
  assert.ok(args.includes('--tempo-map'));
  assert.ok(args.includes('--abc-melody-only'));
  assert.equal(args[args.indexOf('--abc-ins') + 1], '100');
  assert.ok(!midiAbcArgs('job', {}, 128).includes('--abc-ins'));
});
