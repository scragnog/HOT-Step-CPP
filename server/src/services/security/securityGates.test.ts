import { test } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import express from 'express';
import { applySecurityGates, type SecurityGateOptions } from './securityGates.js';

async function withApp(opts: SecurityGateOptions, fn: (port: number) => Promise<void>) {
  const app = express();
  applySecurityGates(app, opts);
  app.all('/api/ping', (_req, res) => { res.json({ ok: true }); });
  const server = app.listen(0, '127.0.0.1');
  await new Promise<void>(r => server.once('listening', () => r()));
  try { await fn((server.address() as AddressInfo).port); }
  finally { await new Promise<void>(r => server.close(() => r())); }
}

function req(port: number, method: string, headers: Record<string, string>) {
  return new Promise<{ status: number; headers: http.IncomingHttpHeaders }>((resolve, reject) => {
    const r = http.request({ host: '127.0.0.1', port, path: '/api/ping', method, headers }, res => {
      res.resume(); res.on('end', () => resolve({ status: res.statusCode ?? 0, headers: res.headers }));
    });
    r.on('error', reject); r.end();
  });
}

test('wiring: Host, Origin and CORS gates on a real HTTP server', async () => {
  await withApp({ serverHost: '127.0.0.1', allowedHosts: 'my-tailnet-name' }, async port => {
    const evilHost = await req(port, 'GET', { Host: `evil.example:${port}` });
    console.log('Host evil.example GET ->', evilHost.status);
    assert.equal(evilHost.status, 403);

    const evilPost = await req(port, 'POST', { Host: `127.0.0.1:${port}`, Origin: 'http://evil.example' });
    console.log('Origin evil POST ->', evilPost.status);
    assert.equal(evilPost.status, 403);

    const evilGet = await req(port, 'GET', { Host: `127.0.0.1:${port}`, Origin: 'http://evil.example' });
    console.log('Origin evil GET ACAO ->', evilGet.headers['access-control-allow-origin']);
    assert.equal(evilGet.headers['access-control-allow-origin'], undefined);

    const plainPost = await req(port, 'POST', { Host: `127.0.0.1:${port}` });
    console.log('no-Origin POST ->', plainPost.status);
    assert.equal(plainPost.status, 200);

    const tailnet = await req(port, 'GET', { Host: `my-tailnet-name:${port}` });
    console.log('Host my-tailnet-name GET ->', tailnet.status);
    assert.equal(tailnet.status, 200);

    const v6 = await req(port, 'GET', { Host: `[::1]:${port}`, Origin: `http://[::1]:${port}` });
    assert.equal(v6.status, 200);
  });
  // Without ALLOWED_HOSTS the tailnet name is refused.
  await withApp({ serverHost: '127.0.0.1' }, async port => {
    assert.equal((await req(port, 'GET', { Host: `my-tailnet-name:${port}` })).status, 403);
  });
});
