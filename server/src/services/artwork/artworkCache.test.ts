import assert from 'node:assert/strict';
import fs from 'node:fs';
import https from 'node:https';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  ArtworkError, getArtwork, isBlockedAddress, isPreviewHost, parseArtworkUrl, pruneCache, readCached,
} from './artworkCache.js';

test('blocks private, loopback, link-local and mapped addresses', () => {
  for (const ip of ['127.0.0.1', '10.1.2.3', '192.168.50.173', '172.20.0.1', '169.254.169.254', '100.64.0.1',
    '::1', 'fe80::1', 'fd00::1', '::ffff:127.0.0.1', '::ffff:192.168.1.1', '0.0.0.0', 'nonsense']) {
    assert.equal(isBlockedAddress(ip), true, ip);
  }
  for (const ip of ['8.8.8.8', '151.101.1.1', '2606:4700:4700::1111', '::ffff:8.8.8.8']) {
    assert.equal(isBlockedAddress(ip), false, ip);
  }
});

test('parseArtworkUrl refuses non-https, credentials and literal private hosts', () => {
  for (const bad of ['http://images.genius.com/a.jpg', 'ftp://x/a', 'file:///etc/passwd', 'nope',
    'https://user:pw@images.genius.com/a.jpg', 'https://127.0.0.1/a.jpg', 'https://[::1]/a.jpg', 'https://169.254.169.254/x']) {
    assert.throws(() => parseArtworkUrl(bad), ArtworkError, bad);
  }
  assert.equal(parseArtworkUrl('https://images.genius.com/a.jpg').hostname, 'images.genius.com');
});

test('preview hosts are Genius only, not look-alikes', () => {
  assert.equal(isPreviewHost(new URL('https://images.genius.com/a.jpg')), true);
  assert.equal(isPreviewHost(new URL('https://images.rapgenius.com/a.jpg')), true);
  assert.equal(isPreviewHost(new URL('https://evilgenius.com/a.jpg')), false);
  assert.equal(isPreviewHost(new URL('https://genius.com.evil.example/a.jpg')), false);
});

test('getArtwork refuses a hostname that resolves to a private address, and caches nothing', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'artwork-'));
  await assert.rejects(getArtwork('https://localhost/a.jpg', { dir }), /Address not allowed|ECONN|refused/i);
  assert.deepEqual(fs.readdirSync(dir), []);
});

test('getArtwork downloads, caches by content type, then serves from cache', async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'artwork-'));
  const png = Buffer.from('89504e470d0a1a0a', 'hex');
  let calls = 0;
  t.mock.method(https, 'get', ((_u: unknown, _o: unknown, cb: (r: any) => void) => {
    calls++;
    const handlers: Record<string, (...a: any[]) => void> = {};
    const res: any = { statusCode: 200, headers: { 'content-type': 'image/png; charset=x' }, resume() {}, on(e: string, f: any) { handlers[e] = f; return res; } };
    queueMicrotask(() => { cb(res); handlers.data?.(png); handlers.end?.(); });
    return { on() { return this; }, destroy() {} };
  }) as any);
  const u = 'https://images.genius.com/a.png';
  const first = await getArtwork(u, { dir });
  assert.equal(first.type, 'image/png');
  assert.deepEqual(fs.readFileSync(first.file), png);
  await getArtwork(u, { dir });
  assert.equal(calls, 1);
  assert.equal(readCached(u, dir)?.file, first.file);
});

test('getArtwork rejects non-image responses', async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'artwork-'));
  t.mock.method(https, 'get', ((_u: unknown, _o: unknown, cb: (r: any) => void) => {
    queueMicrotask(() => cb({ statusCode: 200, headers: { 'content-type': 'image/svg+xml' }, resume() {}, on() {} }));
    return { on() { return this; }, destroy() {} };
  }) as any);
  await assert.rejects(getArtwork('https://images.genius.com/a.svg', { dir }), (e: ArtworkError) => e.status === 415);
});

test('a redirect to a private address is refused', async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'artwork-'));
  t.mock.method(https, 'get', ((_u: unknown, _o: unknown, cb: (r: any) => void) => {
    queueMicrotask(() => cb({ statusCode: 302, headers: { location: 'https://127.0.0.1/secret.png' }, resume() {}, on() {} }));
    return { on() { return this; }, destroy() {} };
  }) as any);
  await assert.rejects(getArtwork('https://images.genius.com/a.png', { dir }), (e: ArtworkError) => e.status === 403);
});

test('pruneCache removes oldest files first until under the cap', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'artwork-'));
  for (const [n, age] of [['old', 3000], ['mid', 2000], ['new', 1000]] as const) {
    const f = path.join(dir, n);
    fs.writeFileSync(f, Buffer.alloc(100));
    const t = new Date(Date.now() - age * 1000);
    fs.utimesSync(f, t, t);
  }
  pruneCache(dir, 150);
  assert.deepEqual(fs.readdirSync(dir), ['new']);
});
