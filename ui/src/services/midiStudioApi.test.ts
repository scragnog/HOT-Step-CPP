import test from 'node:test';
import assert from 'node:assert/strict';
import { dettatoReady } from './midiStudioApi';

// The server's status for three folders present and the repairer missing: 'dettato' is the four-folder total.
const threeOfFour = {
  dettato: { downloaded: false }, 'dettato-transcriber': { downloaded: true },
  'dettato-evidence': { downloaded: false }, 'dettato-grouper': { downloaded: true }, 'dettato-drums': { downloaded: true },
};

test('with the repairer off, dettato is ready without the repairer folder', () => {
  assert.equal(dettatoReady(threeOfFour, false), true);
  assert.equal(dettatoReady(threeOfFour, true), false);
});

test('a missing grouper blocks dettato in either mode, and no status blocks it too', () => {
  const noGrouper = { ...threeOfFour, 'dettato-grouper': { downloaded: false } };
  assert.equal(dettatoReady(noGrouper, false), false);
  assert.equal(dettatoReady(undefined, false), false);
});
