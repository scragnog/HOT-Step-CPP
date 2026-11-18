// The lease-free write hold (workspace export). maintenance.test.ts covers the
// update gate; nothing here touches a lease or the disk.
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import express from 'express';
import { createGate, type Lease } from './maintenance.js';

async function serve(gate: ReturnType<typeof createGate>) {
  const app = express();
  app.use(gate.middleware);
  let release: (() => void) | null = null;
  app.post('/api/slow', (_req, res) => { release = () => res.json({ ok: true }); });
  app.all('/api/*splat', (req, res) => res.json({ method: req.method }));
  const server = http.createServer(app);
  await new Promise<void>(r => server.listen(0, '127.0.0.1', r));
  const url = `http://127.0.0.1:${(server.address() as any).port}`;
  const status = async (method: string, p: string) => (await fetch(url + p, { method })).status;
  return { url, status, finishSlow: () => release?.(), close: () => new Promise<void>(r => { server.closeAllConnections(); server.close(() => r()); }) };
}

test('hold: writes get 503, reads and allowed paths pass, release reopens', async () => {
  const gate = createGate();
  const s = await serve(gate);
  try {
    const h = await gate.hold({ blockers: () => [], allow: ['/api/archive/cancel'] });
    assert.equal(gate.held, true);
    assert.deepEqual(gate.state, { closed: false, leaseId: null, deferred: 0 }, 'the update state is untouched');
    for (const m of ['POST', 'PUT', 'PATCH', 'DELETE']) assert.equal(await s.status(m, '/api/songs'), 503, m);
    for (const m of ['GET', 'HEAD', 'OPTIONS']) assert.equal(await s.status(m, '/api/songs'), 200, m);
    assert.equal(await s.status('POST', '/api/archive/cancel'), 200);
    const body = await (await fetch(s.url + '/api/songs', { method: 'POST' })).json() as any;
    assert.equal(body.maintenance.state, 'held');
    h.release();
    h.release();
    assert.equal(gate.held, false);
    assert.equal(await s.status('POST', '/api/songs'), 200);
  } finally { await s.close(); }
});

test('hold: refused while work runs or is queued, during an update, or while held', async () => {
  const gate = createGate();
  await assert.rejects(gate.hold({ blockers: () => ['Training job-1: running'] }),
    (e: any) => e.status === 409 && e.blockers[0] === 'Training job-1: running');
  assert.equal(gate.held, false);

  let queued = ['2 audio queue item(s) waiting'];
  gate.addBlocker(() => queued);
  await assert.rejects(gate.hold({ blockers: () => [] }), (e: any) => e.status === 409 && e.blockers[0] === queued[0]);
  queued = [];

  const h = await gate.hold({ blockers: () => [] });
  await assert.rejects(gate.hold({ blockers: () => [] }), /already holding writes/);
  queued = ['work appeared'];
  assert.deepEqual(h.check(), ['work appeared'], 'check() re-runs the gate blockers too');
  h.release();
  queued = [];

  gate.close({ id: 'x' } as Lease);
  await assert.rejects(gate.hold({ blockers: () => [] }), /An update is in progress/);
  gate.open();
});

test('hold: drains in-flight writes, and gives up (released) when one will not finish', async () => {
  const gate = createGate();
  const s = await serve(gate);
  try {
    const slow = fetch(s.url + '/api/slow', { method: 'POST' });
    await new Promise(r => setTimeout(r, 50));
    let settled = false;
    const pending = gate.hold({ blockers: () => [], drainMs: 5000 }).then(h => { settled = true; return h; });
    await new Promise(r => setTimeout(r, 150));
    assert.equal(settled, false, 'waits for the in-flight write');
    assert.equal(await s.status('POST', '/api/songs'), 503, 'new writes are refused while it drains');
    s.finishSlow();
    assert.equal((await slow).status, 200);
    (await pending).release();

    const stuck = fetch(s.url + '/api/slow', { method: 'POST' }).catch(() => null);
    await new Promise(r => setTimeout(r, 50));
    await assert.rejects(gate.hold({ blockers: () => [], drainMs: 100 }), (e: any) => e.status === 409 && e.blockers[0] === 'requests still in flight');
    assert.equal(gate.held, false);
    s.finishSlow();
    await stuck;
  } finally { await s.close(); }
});

test('hold: a hold taken inside a request handler does not wait for that request', async () => {
  const gate = createGate();
  const app = express();
  app.use(gate.middleware);
  app.post('/api/start', async (_req, res) => {
    const h = await gate.hold({ blockers: () => [], fromRequest: true, drainMs: 500 });
    h.release();
    res.json({ ok: true });
  });
  const server = http.createServer(app);
  await new Promise<void>(r => server.listen(0, '127.0.0.1', r));
  try {
    const res = await fetch(`http://127.0.0.1:${(server.address() as any).port}/api/start`, { method: 'POST' });
    assert.equal(res.status, 200);
  } finally { await new Promise<void>(r => server.close(() => r())); }
});
