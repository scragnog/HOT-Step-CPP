import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import express from 'express';
import { serveUiDist } from './uiDist.js';
import { CSP_HEADER, contentSecurityPolicy } from '../contracts/csp.js';

function get(port: number, p: string) {
  return new Promise<{ status: number; headers: http.IncomingHttpHeaders; body: string }>((resolve, reject) => {
    http.get({ host: '127.0.0.1', port, path: p }, res => {
      let body = '';
      res.on('data', d => { body += d; });
      res.on('end', () => resolve({ status: res.statusCode ?? 0, headers: res.headers, body }));
    }).on('error', reject);
  });
}

test('production UI path: page, deep link and service worker carry the enforced CSP; API does not', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-uidist-'));
  fs.writeFileSync(path.join(dir, 'index.html'), '<!doctype html><title>ui</title>');
  fs.writeFileSync(path.join(dir, 'sw.js'), '// sw');
  const app = express();
  app.get('/api/ping', (_req, res) => { res.json({ ok: true }); });
  serveUiDist(app, dir);
  const server = app.listen(0, '127.0.0.1');
  await new Promise<void>(r => server.once('listening', () => r()));
  const port = (server.address() as AddressInfo).port;
  try {
    const policy = contentSecurityPolicy();
    assert.equal(CSP_HEADER, 'Content-Security-Policy');
    for (const p of ['/', '/library', '/lyric-studio/artist/1', '/sw.js']) {
      const r = await get(port, p);
      console.log(p, '->', r.status, r.headers['content-security-policy'] ? 'CSP' : 'no CSP');
      assert.equal(r.status, 200, p);
      assert.equal(r.headers['content-security-policy'], policy, p);
      assert.equal(r.headers['content-security-policy-report-only'], undefined, p);
    }
    assert.match((await get(port, '/library')).body, /<title>ui<\/title>/);
    assert.equal((await get(port, '/api/ping')).headers['content-security-policy'], undefined);
  } finally {
    await new Promise<void>(r => server.close(() => r()));
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
