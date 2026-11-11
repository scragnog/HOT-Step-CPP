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

const catalogue = JSON.parse(fs.readFileSync(new URL('../data/model-registry.json', import.meta.url), 'utf8'));

test('all five generation backends have complete packs referencing real catalogue entries', () => {
  const ids = catalogue.files.map((f: any) => f.id);
  assert.equal(new Set(ids).size, ids.length, 'file ids must be unique');
  for (const family of ['as1.5', 'mm3', 'yue2', 'sa3', 'heartmula']) {
    assert.ok(catalogue.files.some((f: any) => f.family === family), family);
    assert.ok(catalogue.packs.some((p: any) => p.family === family && p.fileIds.length > 0), family);
  }
  for (const pack of catalogue.packs) {
    for (const id of pack.fileIds) assert.ok(ids.includes(id), `${pack.id}: ${id}`);
  }
  const sa3 = catalogue.packs.find((p: any) => p.id === 'sa3-generation');
  assert.equal(sa3.fileIds.length, 7, 'generation needs one DiT, not every alternative quant');
  assert.ok(sa3.fileIds.includes('stablestep-gguf-dit'));
  assert.ok(sa3.fileIds.includes('stablestep-tokenizer'));
  assert.ok(!sa3.fileIds.some((id: string) => id.startsWith('sa3-training')));
});

test('every HeartMuLa download agrees with the measured publication inventory', () => {
  const manifest = JSON.parse(fs.readFileSync(new URL('../../../docs/dev/heartmula-hf-manifest.json', import.meta.url), 'utf8'));
  const weights = manifest.files.filter((f: any) => f.upload && f.path.endsWith('.gguf'));
  const entries = catalogue.files.filter((f: any) => f.family === 'heartmula');
  assert.equal(entries.length, weights.length);
  for (const file of weights) {
    const entry = entries.find((f: any) => f.filename === file.path);
    assert.ok(entry, file.path);
    assert.equal(entry.repo, manifest.repo);
    assert.equal(entry.repoPath, file.path);
    assert.equal(entry.subdir, 'heartmula');
    assert.equal(entry.sizeBytes, file.sizeBytes);
    assert.equal(entry.sha256, file.sha256);
    assert.match(entry.sha256, /^[a-f0-9]{64}$/);
  }
});

test('each HeartMuLa variant offers the same full ladder for both LM parts', () => {
  const ladder = ['F16', 'Q8_0', 'Q6_K', 'Q5_K_M', 'Q5_K_S', 'Q4_K_M', 'Q4_K_S',
    'Q3_K_L', 'Q3_K_M', 'Q3_K_S', 'Q2_K', 'IQ4_XS', 'IQ3_XXS', 'IQ2_XS',
    'IQ2_XXS', 'IQ1_M', 'IQ1_S', 'MXFP4', 'NVFP4'].sort();
  for (const variant of ['heartmula-3b', 'heartmula-3b-happy-new-year', 'heartmula-3b-rl']) {
    for (const part of ['global', 'local']) {
      const files = catalogue.files.filter((f: any) => f.variant === `${variant}-${part}`);
      assert.deepEqual(files.map((f: any) => f.quant).sort(), ladder, `${variant}-${part}`);
    }
  }
});

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

test('a model already on disk still gets its missing companion files, side by side', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'companions-'));
  fs.writeFileSync(path.join(dir, 'model.gguf'), 'weights');
  const fetched: string[] = [];
  const realTarget = svc.getTargetDir, realSimple = svc._downloadSimple;
  svc.getTargetDir = () => dir;
  svc._downloadSimple = async (url: string, dest: string) => { fetched.push(url); fs.writeFileSync(dest, url); };
  try {
    const job = { status: 'queued', bytesDownloaded: 0, totalBytes: 0 } as any;
    for (const name of ['LICENSE', 'LICENSE-YuE2-RS-4.0x0.7']) {
      await svc._executeDownload(job, {
        filename: 'model.gguf', sizeBytes: 0, repo: 'o/r',
        companions: [{ filename: name, repoPath: 'LICENSE' }],
      });
    }
    await new Promise(r => setTimeout(r, 20)); // companions are fire-and-forget
    assert.equal(job.status, 'completed');
    assert.deepEqual(fs.readdirSync(dir).sort(), ['LICENSE', 'LICENSE-YuE2-RS-4.0x0.7', 'model.gguf']);
    assert.ok(fetched.every(u => u.endsWith('/o/r/resolve/main/LICENSE')));
  } finally {
    svc.getTargetDir = realTarget; svc._downloadSimple = realSimple;
  }
});

function hashSandbox() {
  const service = new (svc.constructor as any)();
  const dir = fs.mkdtempSync(path.join(process.cwd(), '.test-model-hashes-'));
  service._installedDir = () => dir;
  service.getInstalledFiles = () => new Set(fs.readdirSync(dir));
  return { service, dir, cleanup: () => fs.rmSync(dir, { recursive: true, force: true }) };
}

test('replacing a stale residual keeps its bytes; other stale models are removed', async () => {
  const service = new (svc.constructor as any)();
  const dir = fs.mkdtempSync(path.join(process.cwd(), '.test-residual-backup-'));
  service.getTargetDir = () => dir;
  try {
    for (const variant of ['residual', 'lm']) {
      const filename = `${variant}.safetensors`;
      const target = path.join(dir, filename);
      fs.writeFileSync(target, 'old bytes needed by adapters');
      const job = { status: 'queued', bytesDownloaded: 0, totalBytes: 0 };
      // Stop at the network boundary, after the replacement decision.
      service._downloadWithRedirects = async (_url: string, _dest: string, active: any) => { active.status = 'cancelled'; };
      await service._executeDownload(job, {
        filename, variant, repo: 'fixture/model', sha256: sha('published bytes'), sizeBytes: 0,
      });
      assert.equal(fs.existsSync(target), false);
      const backups = fs.readdirSync(dir).filter(name => name.startsWith(`${filename}.backup-`));
      assert.equal(backups.length, variant === 'residual' ? 1 : 0);
      if (variant === 'residual') {
        assert.equal(fs.readFileSync(path.join(dir, backups[0]), 'utf8'), 'old bytes needed by adapters');
        // A second replacement cannot overwrite the first backup.
        fs.writeFileSync(target, 'second residual');
        await service._executeDownload({ ...job, status: 'queued' }, {
          filename, variant, repo: 'fixture/model', sha256: sha('published bytes'), sizeBytes: 0,
        });
        assert.equal(fs.readdirSync(dir).filter(name => name.startsWith(`${filename}.backup-`)).length, 2);
      }
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a residual that cannot be backed up is never replaced', async () => {
  const service = new (svc.constructor as any)();
  const dir = fs.mkdtempSync(path.join(process.cwd(), '.test-residual-rename-'));
  const target = path.join(dir, 'residual.safetensors');
  fs.writeFileSync(target, 'only copy');
  service.getTargetDir = () => dir;
  let fetched = false;
  service._downloadWithRedirects = async () => { fetched = true; };
  const rename = fs.renameSync;
  fs.renameSync = (source, destination) => {
    if (source === target) throw new Error('backup rename refused');
    return rename(source, destination);
  };
  try {
    await assert.rejects(service._executeDownload({ status: 'queued' }, {
      filename: 'residual.safetensors', variant: 'residual', sha256: sha('published bytes'),
    }), /backup rename refused/);
    assert.equal(fs.readFileSync(target, 'utf8'), 'only copy');
    assert.equal(fetched, false);
  } finally {
    fs.renameSync = rename;
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

async function hashesSettled(service: any) {
  for (let i = 0; i < 500 && service._hashing.size; i++) {
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  assert.equal(service._hashing.size, 0, 'background hashes must settle');
}

test('a same-size GGUF with a valid header becomes outdated after its background hash', async () => {
  const { service, dir, cleanup } = hashSandbox();
  const file = catalogue.files.filter((f: any) => f.sha256 && f.filename.endsWith('.gguf'))
    .sort((a: any, b: any) => a.sizeBytes - b.sizeBytes)[0];
  const bytes = Buffer.alloc(file.sizeBytes);
  bytes.write('GGUF');
  const p = path.join(dir, file.filename);
  fs.writeFileSync(p, bytes);
  try {
    service._validateDownload(p, file);
    const row = () => service.getRegistry().files.find((f: any) => f.id === file.id);
    assert.equal(row().installed, true);
    assert.equal(row().outdated, false, 'unknown until the hash finishes');
    await hashesSettled(service);
    assert.equal(row().outdated, true);
    assert.equal(fs.readFileSync(`${p}.sha256`, 'utf8').split(/\s+/)[0],
      createHash('sha256').update(bytes).digest('hex'));
  } finally { cleanup(); }
});

test('background hashes run serially, smallest first, and duplicate reads stay queued once', async () => {
  const { service, dir, cleanup } = hashSandbox();
  const started: string[] = [];
  let active = 0, peak = 0;
  service._hashFile = async (p: string) => {
    started.push(path.basename(p));
    peak = Math.max(peak, ++active);
    await new Promise(resolve => setTimeout(resolve, 10));
    active--;
    return sha(fs.readFileSync(p, 'utf8'));
  };
  try {
    for (const [filename, sizeBytes] of [['large.gguf', 32], ['small.gguf', 8], ['medium.gguf', 16]] as const) {
      const contents = 'x'.repeat(sizeBytes);
      fs.writeFileSync(path.join(dir, filename), contents);
      const file = { filename, sizeBytes, sha256: sha(contents) };
      service._isKnownStale(file);
      service._isKnownStale(file);
    }
    await hashesSettled(service);
    assert.equal(peak, 1);
    assert.deepEqual(started, ['small.gguf', 'medium.gguf', 'large.gguf']);
  } finally { cleanup(); }
});

test('a fresh service trusts current sidecars and rechecks sidecars before draining queued files', async () => {
  const { service, dir, cleanup } = hashSandbox();
  const files = ['small.gguf', 'large.gguf'].map((filename, i) => {
    const contents = 'x'.repeat(8 + i * 8);
    fs.writeFileSync(path.join(dir, filename), contents);
    return { filename, sizeBytes: contents.length, sha256: sha(contents) };
  });
  const started: string[] = [];
  service._hashFile = async (p: string) => {
    started.push(path.basename(p));
    // A download or another instance can finish a queued file first.
    service._writeSidecar(path.join(dir, files[1].filename), files[1].sha256);
    return files[0].sha256;
  };
  try {
    for (const file of files) service._isKnownStale(file);
    await hashesSettled(service);
    assert.deepEqual(started, ['small.gguf']);
    const restarted = new (svc.constructor as any)();
    restarted._installedDir = () => dir;
    restarted._hashFile = async () => { assert.fail('a current sidecar must prevent disk reads'); };
    for (const file of files) assert.equal(restarted._isKnownStale(file), false);
    assert.equal(restarted._hashing.size, 0);
  } finally { cleanup(); }
});

test('background hashing waits for registered jobs, then resumes', async () => {
  const { registerBusyCheck } = await import('../dataroot/pointer.js');
  const { service, dir, cleanup } = hashSandbox();
  const contents = 'GGUFvalid';
  const file = { filename: 'model.gguf', sizeBytes: contents.length, sha256: sha(contents) };
  fs.writeFileSync(path.join(dir, file.filename), contents);
  let busy = true, reads = 0;
  registerBusyCheck('model-hash-test', () => busy ? ['test generation running'] : []);
  service._hashFile = async () => { reads++; return file.sha256; };
  try {
    service._isKnownStale(file);
    await new Promise(resolve => setTimeout(resolve, 25));
    assert.equal(reads, 0);
    busy = false;
    await hashesSettled(service);
    assert.equal(reads, 1);
  } finally {
    busy = false;
    registerBusyCheck('model-hash-test', () => []);
    cleanup();
  }
});
