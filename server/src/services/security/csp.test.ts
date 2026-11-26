import { test } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import express from 'express';
import { applySecurityGates, logCspReport } from './securityGates.js';
import { CSP_HEADER, CSP_REPORT_PATH, contentSecurityPolicy } from '../../contracts/csp.js';

const directive = (policy: string, name: string) =>
  policy.split('; ').find(d => d.startsWith(name + ' '))?.split(' ').slice(1);

test('production policy: scripts and connections stay on this origin', () => {
  const p = contentSecurityPolicy();
  assert.deepEqual(directive(p, 'script-src'), ["'self'", "'wasm-unsafe-eval'"]);
  assert.deepEqual(directive(p, 'connect-src'), ["'self'"]);
  assert.deepEqual(directive(p, 'default-src'), ["'self'"]);
  assert.deepEqual(directive(p, 'object-src'), ["'none'"]);
  assert.deepEqual(directive(p, 'report-uri'), [CSP_REPORT_PATH]);
  assert.ok(!p.includes("'unsafe-eval'"));
});

test('dev policy only adds inline scripts (Vite React Refresh preamble)', () => {
  assert.deepEqual(directive(contentSecurityPolicy(true), 'script-src'), ["'self'", "'wasm-unsafe-eval'", "'unsafe-inline'"]);
  assert.ok(!contentSecurityPolicy(true).includes("'unsafe-eval'"));
  const strip = (p: string) => p.replace(/script-src [^;]*/, '');
  assert.equal(strip(contentSecurityPolicy(true)), strip(contentSecurityPolicy()));
  assert.ok(CSP_HEADER === 'Content-Security-Policy' || CSP_HEADER === 'Content-Security-Policy-Report-Only');
});

test('report route accepts application/csp-report and logs each violation once', async () => {
  const app = express();
  applySecurityGates(app, { serverHost: '127.0.0.1' });
  const server = app.listen(0, '127.0.0.1');
  await new Promise<void>(r => server.once('listening', () => r()));
  const port = (server.address() as AddressInfo).port;
  const lines: string[] = [];
  const warn = console.warn;
  console.warn = (s: string) => { lines.push(s); };
  const body = JSON.stringify({ 'csp-report': { 'document-uri': 'http://localhost/x', 'effective-directive': 'img-src', 'blocked-uri': 'https://evil.example/p.png', disposition: 'enforce' } });
  const post = () => new Promise<number>((resolve, reject) => {
    const r = http.request({ host: '127.0.0.1', port, path: CSP_REPORT_PATH, method: 'POST', headers: { 'Content-Type': 'application/csp-report', Host: `127.0.0.1:${port}` } }, res => { res.resume(); res.on('end', () => resolve(res.statusCode ?? 0)); });
    r.on('error', reject); r.end(body);
  });
  try {
    assert.equal(await post(), 204);
    assert.equal(await post(), 204);
  } finally {
    console.warn = warn;
    await new Promise<void>(r => server.close(() => r()));
  }
  assert.deepEqual(lines, ['[CSP] enforce img-src https://evil.example/p.png on http://localhost/x']);
  assert.equal(logCspReport({ nothing: true }), null);
});

test('report fields are stripped of queries, control characters and excess length', () => {
  const warn = console.warn;
  console.warn = () => {};
  try {
    const line = logCspReport({ 'csp-report': {
      'document-uri': 'http://localhost:3000/library?token=secret#frag',
      'blocked-uri': 'https://evil.example/a.png?d=user-text',
      'effective-directive': 'img-src\n[CSP] forged line',
      'source-file': 'http://localhost:3000/x.js?v=1',
      'line-number': 7,
      disposition: 'enforce\u001b[31m',
    } });
    assert.equal(line, '[CSP] enforce?[31m img-src?[CSP] https://evil.example/a.png on http://localhost:3000/library at http://localhost:3000/x.js:7');
    assert.ok(!/secret|user-text|frag|\n|\u001b/.test(line ?? ''));
    const long = logCspReport({ 'csp-report': { 'document-uri': 'http://localhost/' + 'a'.repeat(5000), 'effective-directive': 'img-src' } });
    assert.ok(long && long.length < 300, `bounded, got ${long?.length}`);
    // Different queries on one page are one violation, not a new log line each.
    const q1 = logCspReport({ 'csp-report': { 'document-uri': 'http://localhost/q?a=1', 'effective-directive': 'font-src', 'blocked-uri': 'https://f.example/x' } });
    const q2 = logCspReport({ 'csp-report': { 'document-uri': 'http://localhost/q?a=2', 'effective-directive': 'font-src', 'blocked-uri': 'https://f.example/x' } });
    assert.ok(q1);
    assert.equal(q2, null);
  } finally {
    console.warn = warn;
  }
});
