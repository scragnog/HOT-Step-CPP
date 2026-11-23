import assert from 'node:assert/strict';
import test from 'node:test';
import { tensorCoreUnlockFromLog } from './tensorCoreUnlock.js';

const bootMs = 1_800_000_000_000;
const writtenMs = bootMs - 2000;
const handoff = 'HANDOFF-OUT: readout1=0x00000000 | tu10x:unlock=false ga/ada:ss0_unlock=true ss1_unlock=true plm_open=true';

test('worker unlock uses the Ada handoff for this boot, ignoring the Turing probe', () => {
  assert.equal(tensorCoreUnlockFromLog({ bootMs, writtenMs, handoff }), 'on');
  assert.equal(tensorCoreUnlockFromLog({ bootMs, writtenMs, handoff: handoff.replace('plm_open=true', 'plm_open=false') }), 'off');
  assert.equal(tensorCoreUnlockFromLog({ bootMs, writtenMs: writtenMs - 86_400_000, handoff }), 'off');
});

test('worker unlock requires recognizable evidence', () => {
  assert.equal(tensorCoreUnlockFromLog({ bootMs, writtenMs, handoff: '' }), 'unknown');
  assert.equal(tensorCoreUnlockFromLog({ bootMs, writtenMs, handoff: handoff.replace('0x00000000', '0x12345678') }), 'unknown');
  assert.equal(tensorCoreUnlockFromLog({ bootMs, writtenMs: bootMs + 60_000, handoff }), 'unknown');
});
