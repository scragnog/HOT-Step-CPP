import test from 'node:test';
import assert from 'node:assert/strict';
import { aceClient, aceKeepLoadedLatched, type AceRequest } from './aceClient.js';

test('a keep_loaded request latches co-residence until the engine restores its policy', async () => {
  const real = globalThis.fetch;
  const seen: string[] = [];
  globalThis.fetch = (async (url: string | URL) => {
    seen.push(String(url));
    const body = String(url).includes('/models/restore-policy') ? { restored: true } : { id: '1' };
    return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
  }) as typeof fetch;
  try {
    const req = { caption: 'c' } as AceRequest;
    await aceClient.submitLm(req);
    assert.equal(aceKeepLoadedLatched(), false, 'a plain request does not latch');
    await aceClient.submitLm(req, undefined, true);
    assert.equal(aceKeepLoadedLatched(), true);
    await aceClient.submitLm(req);
    assert.equal(aceKeepLoadedLatched(), true, 'a later plain request finds the engine still resident');
    assert.equal(await aceClient.restoreEvictPolicy(), true);
    assert.equal(aceKeepLoadedLatched(), false);
    assert.ok(seen.some(u => u.includes('keep_loaded=1')));
  } finally {
    globalThis.fetch = real;
  }
});
