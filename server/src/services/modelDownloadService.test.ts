// What this guards: the sha256 staleness check that lets a republished model
// reach an install that already holds the old bytes under the same filename.
// The YuE2 tokenizer head went v4 -> v9 at the same name, 832 bytes apart —
// inside the 5% tolerance the download validator allows and invisible to the
// existsSync test — so the sidecar cache is the only thing standing between a
// registry update and a silent split between old and new installs.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { describeDownloadError, modelDownloadService } from './modelDownloadService.js';

const svc = modelDownloadService as any;

function tmpFile(contents: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'model-sha-'));
  const p = path.join(dir, 'model.gguf');
  fs.writeFileSync(p, contents);
  return p;
}

const sha = (s: string) => createHash('sha256').update(s).digest('hex');
const entry = (sha256?: string) => ({ filename: 'model.gguf', sizeBytes: 0, sha256 } as any);

test('an entry without a sha256 is always considered current', async () => {
  const p = tmpFile('anything at all');
  assert.equal(await svc._matchesRegistry(p, entry(undefined)), true);
  assert.equal(fs.existsSync(`${p}.sha256`), false, 'nothing to cache, so no sidecar');
});

test('a matching sha256 passes and is remembered in a sidecar', async () => {
  const p = tmpFile('the published bytes');
  assert.equal(await svc._matchesRegistry(p, entry(sha('the published bytes'))), true);
  const [cached, stamp] = fs.readFileSync(`${p}.sha256`, 'utf8').trim().split(/\s+/);
  const stat = fs.statSync(p);
  assert.equal(cached, sha('the published bytes'));
  assert.equal(stamp, `${stat.size}:${Math.floor(stat.mtimeMs)}`);
});

test('a file holding the wrong bytes is reported stale', async () => {
  const p = tmpFile('the OLD bytes');
  assert.equal(await svc._matchesRegistry(p, entry(sha('the NEW bytes'))), false);
});

test('a sidecar stamped for the current file is trusted without rehashing', async () => {
  const p = tmpFile('real contents');
  const stat = fs.statSync(p);
  // A hash that is not this file's: only a cache read can return it.
  fs.writeFileSync(`${p}.sha256`, `${sha('not this file')} ${stat.size}:${Math.floor(stat.mtimeMs)}\n`);
  assert.equal(svc._cachedSha(p), sha('not this file'));
  assert.equal(await svc._matchesRegistry(p, entry(sha('real contents'))), false);
});

test('a sidecar whose stamp no longer fits the file is ignored and rehashed', async () => {
  const p = tmpFile('real contents');
  fs.writeFileSync(`${p}.sha256`, `${sha('not this file')} 1:1\n`);
  assert.equal(svc._cachedSha(p), null);
  assert.equal(await svc._matchesRegistry(p, entry(sha('real contents'))), true);
  assert.equal(fs.readFileSync(`${p}.sha256`, 'utf8').split(/\s+/)[0], sha('real contents'));
});

// ── Error reporting (#171) ──────────────────────────────────────────────────
// A Mac reporter got three lines of "Download failed after 3 attempts:
// yue2-lm-bf16.gguf — " with nothing after the dash. An AggregateError's
// message is the empty string, so the one thing we logged was the one thing it
// does not have.

test('an AggregateError reports its causes, not its empty message', () => {
  const err: any = new AggregateError([
    Object.assign(new Error('connect ECONNREFUSED'), { code: 'ECONNREFUSED', syscall: 'connect', address: '2600::1', port: 443 }),
    Object.assign(new Error('connect ECONNREFUSED'), { code: 'ECONNREFUSED', syscall: 'connect', address: '18.0.0.1', port: 443 }),
  ]);
  err.url = 'https://cdn-lfs.huggingface.co/repos/x/model.gguf';
  const out = describeDownloadError(err);
  assert.match(out, /ECONNREFUSED connect 2600::1:443/);
  assert.match(out, /ECONNREFUSED connect 18\.0\.0\.1:443/);
  assert.match(out, /while fetching cdn-lfs\.huggingface\.co/);
  assert.doesNotMatch(out, /^\s*—?\s*$/);
});

test('an ordinary errno error keeps its message and gains its code', () => {
  const err: any = Object.assign(new Error('getaddrinfo ENOTFOUND huggingface.co'),
    { code: 'ENOTFOUND', syscall: 'getaddrinfo' });
  assert.equal(describeDownloadError(err), 'getaddrinfo ENOTFOUND huggingface.co (ENOTFOUND getaddrinfo)');
});

test('an HTTP status error says which host answered', () => {
  const err: any = Object.assign(new Error('HTTP 403: Forbidden'), { url: 'https://huggingface.co/a/b.gguf' });
  assert.equal(describeDownloadError(err), 'HTTP 403: Forbidden while fetching huggingface.co');
});

test('nothing usable still produces something to read', () => {
  assert.equal(describeDownloadError(undefined), 'unknown error');
  assert.equal(describeDownloadError({}), '[object Object]');
});
