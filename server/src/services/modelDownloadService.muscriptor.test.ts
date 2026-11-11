// MuScriptor and Beat This! in the Model Manager (#228).
//
// What this guards: the four variants share the filename model.safetensors, so
// installed status, download destination and deletion must go by exact path
// (not basename); config.json is part of the install and is fetched with the
// token and validated; MIDI Studio and the Model Manager share one job; and a
// complete pre-#228 download is reused with no network.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { hfAuthHeader, modelDownloadService } from './modelDownloadService.js';
import { muscriptorRoots } from './muscriptorPaths.js';

const svc = modelDownloadService as any;
const catalogue = JSON.parse(fs.readFileSync(new URL('../data/model-registry.json', import.meta.url), 'utf8'));
const sha = (s: string) => createHash('sha256').update(s).digest('hex');

interface Net {
  simple?: (url: string, dest: string, token?: string) => Promise<void>;
  weights?: (url: string, part: string, job: any) => Promise<void>;
}

/** Throwaway MuScriptor roots, a stubbed network and no retry delays. */
function sandbox(net: Net = {}) {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'muscriptor-mm-'));
  const roots = muscriptorRoots(undefined, path.join(base, 'models'), path.join(base, 'library'));
  const cls = svc.constructor as any;
  const saved = {
    installed: svc.getInstalledFiles, roots: svc._muscriptorRoots, simple: svc._downloadSimple, weights: svc._downloadWithRedirects,
    jobs: new Map<string, any>(svc.jobs), delays: [...cls.RETRY_DELAYS],
  };
  svc._muscriptorRoots = () => roots;
  svc.getInstalledFiles = () => new Set<string>(); // Keep background checks off the real model library.
  svc._downloadSimple = net.simple ?? (async () => { throw new Error('network used'); });
  svc._downloadWithRedirects = net.weights ?? (async () => { throw new Error('network used'); });
  svc.jobs.clear();
  cls.RETRY_DELAYS.splice(0, cls.RETRY_DELAYS.length, 0, 0, 0);
  return {
    shared: roots.shared,
    legacy: roots.legacy,
    restore() {
      svc.getInstalledFiles = saved.installed;
      svc._muscriptorRoots = saved.roots; svc._downloadSimple = saved.simple; svc._downloadWithRedirects = saved.weights;
      svc.jobs.clear();
      for (const [k, v] of saved.jobs) svc.jobs.set(k, v);
      cls.RETRY_DELAYS.splice(0, cls.RETRY_DELAYS.length, ...saved.delays);
    },
  };
}

function put(dir: string, files: Record<string, string>) {
  fs.mkdirSync(dir, { recursive: true });
  for (const [name, text] of Object.entries(files)) fs.writeFileSync(path.join(dir, name), text);
}

const installedIds = () => svc.getRegistry().files
  .filter((f: any) => f.installed && f.muscriptorModel).map((f: any) => f.id).sort();

async function settle(jobId: string, until = ['completed', 'failed', 'cancelled']) {
  for (let i = 0; i < 300 && !until.includes(svc.jobs.get(jobId).status); i++) await new Promise(r => setTimeout(r, 10));
  return svc.jobs.get(jobId);
}

test('the catalogue lists all four models under Shared with a required config', () => {
  const entries = catalogue.files.filter((f: any) => f.muscriptorModel);
  assert.deepEqual(entries.map((f: any) => f.muscriptorModel).sort(), ['beat-this', 'large', 'medium', 'small']);
  for (const e of entries) {
    assert.equal(e.family, 'shared');
    assert.equal(e.role, 'muscriptor');
    assert.equal(e.filename, 'model.safetensors');
    assert.equal(e.id, `muscriptor-${e.muscriptorModel}`);
    assert.deepEqual(e.companions, [{ filename: 'config.json', required: true }]);
  }
  assert.ok(entries.find((f: any) => f.id === 'muscriptor-large').tags.includes('gated'));
  assert.match(entries.find((f: any) => f.id === 'muscriptor-large').sha256, /^[a-f0-9]{64}$/, 'HF exposes the LFS hash even for gated weights');
  assert.match(entries.find((f: any) => f.id === 'muscriptor-beat-this').sha256, /^[a-f0-9]{64}$/);
  assert.deepEqual(catalogue.packs.find((p: any) => p.id === 'midi-transcription').fileIds, ['muscriptor-small', 'muscriptor-beat-this']);
});

test('two variants share one filename and only the installed one reads as installed', () => {
  const sb = sandbox();
  try {
    put(path.join(sb.shared, 'small'), { 'model.safetensors': 'w', 'config.json': '{}' });
    assert.deepEqual(installedIds(), ['muscriptor-small']);
    put(path.join(sb.shared, 'large'), { 'model.safetensors': 'w', 'config.json': '{}' });
    assert.deepEqual(installedIds(), ['muscriptor-large', 'muscriptor-small']);
  } finally { sb.restore(); }
});

test('weights without config.json are not installed', () => {
  const sb = sandbox();
  try {
    put(path.join(sb.shared, 'medium'), { 'model.safetensors': 'w' });
    assert.deepEqual(installedIds(), []);
  } finally { sb.restore(); }
});

test('a complete pre-#228 copy shows as installed and is reused without any network', async () => {
  const sb = sandbox();
  try {
    put(path.join(sb.legacy, 'small'), { 'model.safetensors': 'w', 'config.json': '{}' });
    // This fixture represents an already verified legacy install.
    svc._writeSidecar(path.join(sb.legacy, 'small', 'model.safetensors'),
      catalogue.files.find((f: any) => f.id === 'muscriptor-small').sha256);
    assert.deepEqual(installedIds(), ['muscriptor-small']);
    assert.deepEqual(svc.startMuscriptorDownload('small'), { started: false, error: 'Already downloaded' });
    const job = await settle(svc.startDownload('muscriptor-small'));
    assert.equal(job.status, 'completed');
    assert.equal(fs.existsSync(path.join(sb.shared, 'small')), false, 'nothing was fetched or copied into shared');
  } finally { sb.restore(); }
});

test('wrong legacy bytes without a current sidecar are refused by the reuse path', async () => {
  for (const staleSidecar of [false, true]) {
    const requested: string[] = [];
    const sb = sandbox({
      simple: async (url) => { requested.push(url); throw new Error('replacement fetch attempted'); },
    });
    const dir = path.join(sb.legacy, 'small');
    const weights = path.join(dir, 'model.safetensors');
    try {
      put(dir, { 'model.safetensors': 'wrong legacy bytes', 'config.json': '{}' });
      const expected = catalogue.files.find((f: any) => f.id === 'muscriptor-small').sha256;
      if (staleSidecar) fs.writeFileSync(`${weights}.sha256`, `${expected} 0:0\n`);
      assert.equal(svc._cachedSha(weights), null);
      const job = await settle(svc.startDownload('muscriptor-small'));
      assert.equal(job.status, 'failed', 'corrupt legacy weights cannot count as completed');
      assert.match(job.error, /replacement fetch attempted/);
      assert.ok(requested.length > 0, 'refusing reuse must attempt a replacement');
      assert.equal(svc._cachedSha(weights), sha('wrong legacy bytes'));
      assert.equal(fs.readFileSync(weights, 'utf8'), 'wrong legacy bytes');
    } finally { sb.restore(); }
  }
});

test('a download fetches config.json with the token before the weights, into the shared folder', async () => {
  const calls: string[] = [];
  const sb = sandbox({
    simple: async (url, dest, token) => { calls.push(`config ${url} ${token}`); fs.writeFileSync(dest, '{"a":1}'); },
    weights: async (_u, part) => { calls.push('weights'); fs.writeFileSync(part, 'weights'); },
  });
  try {
    const job = await settle(svc.startDownload('muscriptor-small', 'hf_abc'));
    assert.equal(job.status, 'failed', '7 bytes is nowhere near the declared size');
    assert.match(calls[0], /MuScriptor\/muscriptor-small\/resolve\/main\/config\.json hf_abc$/);
    assert.equal(calls[1], 'weights');
    assert.equal(fs.existsSync(path.join(sb.shared, 'small', 'config.json')), false, 'a failed job publishes no config');
    assert.equal(fs.existsSync(path.join(sb.shared, 'small', 'config.json.part')), false);
  } finally { sb.restore(); }
});

test('cancelling mid-weights leaves no config.json and no .part', async () => {
  let release!: () => void;
  const gate = new Promise<void>(r => { release = r; });
  const sb = sandbox({
    simple: async (_u, dest) => { fs.writeFileSync(dest, '{"a":1}'); },
    weights: async (_u, part) => { fs.writeFileSync(part, '12345'); await gate; },
  });
  try {
    const dir = path.join(sb.shared, 'small');
    const jobId = svc.startDownload('muscriptor-small');
    for (let i = 0; i < 200 && !fs.existsSync(path.join(dir, 'model.safetensors.part')); i++) await new Promise(r => setTimeout(r, 10));
    assert.ok(fs.existsSync(path.join(dir, 'config.json.part')), 'config is staged while weights download');
    assert.equal(fs.existsSync(path.join(dir, 'config.json')), false);
    assert.equal(svc.cancelDownload(jobId), true);
    release();
    await new Promise(r => setTimeout(r, 50));
    assert.deepEqual(fs.readdirSync(dir), []);
    assert.deepEqual(installedIds(), []);
  } finally { release(); sb.restore(); }
});

test('denied access fails the job on config.json before any weights are requested', async () => {
  let weights = 0;
  const sb = sandbox({
    simple: async url => { throw Object.assign(new Error('HTTP 403: Forbidden'), { url }); },
    weights: async () => { weights++; },
  });
  try {
    const job = await settle(svc.startDownload('muscriptor-large', 'hf_nope'));
    assert.equal(job.status, 'failed');
    assert.match(job.error, /HTTP 403/);
    assert.match(job.error, /Request access/);
    assert.deepEqual(svc.muscriptorState('large'), { downloading: false, receivedBytes: 0, totalBytes: job.totalBytes, error: job.error, gated: true });
    assert.equal(weights, 0);
    assert.equal(fs.existsSync(path.join(sb.shared, 'large', 'model.safetensors')), false);
  } finally { sb.restore(); }
});

test('an invalid config.json is rejected and leaves nothing behind', async () => {
  const sb = sandbox({ simple: async (_u, dest) => { fs.writeFileSync(dest, '<html>sign in</html>'); } });
  try {
    const job = await settle(svc.startDownload('muscriptor-small'));
    assert.equal(job.status, 'failed');
    assert.match(job.error, /config\.json/);
    assert.deepEqual(fs.readdirSync(path.join(sb.shared, 'small')), []);
  } finally { sb.restore(); }
});

test('a fresh download whose bytes miss the declared sha256 is rejected and removed', async () => {
  const sb = sandbox({ weights: async (_u, part) => { fs.writeFileSync(part, 'the wrong bytes'); } });
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'sha-fresh-'));
  const realTarget = svc.getTargetDir;
  svc.getTargetDir = () => dir;
  try {
    const job = { status: 'queued', bytesDownloaded: 0, totalBytes: 0 } as any;
    await svc._executeDownload(job, { filename: 'm.bin', sizeBytes: 0, repo: 'o/r', sha256: sha('the right bytes') });
    assert.equal(job.status, 'failed');
    assert.match(job.error, /sha256 mismatch/);
    assert.deepEqual(fs.readdirSync(dir), []);
  } finally { svc.getTargetDir = realTarget; sb.restore(); }
});

test('the HF token goes to huggingface.co over https and nowhere else', () => {
  const h = (u: string) => hfAuthHeader(new URL(u), 'hf_x');
  assert.deepEqual(h('https://huggingface.co/a/b/resolve/main/config.json'), { Authorization: 'Bearer hf_x' });
  assert.deepEqual(h('http://huggingface.co/x'), {}, 'never over plain http');
  assert.deepEqual(h('https://cas-bridge.xethub.hf.co/x'), {});
  assert.deepEqual(h('https://huggingface.co.evil.example/x'), {});
  assert.deepEqual(h('https://evilhuggingface.co/x'), {});
  assert.deepEqual(hfAuthHeader(new URL('https://huggingface.co/x'), undefined), {});
});

test('a redirect to another host is followed without the token', async () => {
  const seen: Array<string | undefined> = [];
  const target = http.createServer((req, res) => { seen.push(req.headers.authorization); res.end('{"ok":true}'); });
  await new Promise<void>(r => target.listen(0, '127.0.0.1', r));
  const hop = http.createServer((req, res) => {
    seen.push(req.headers.authorization);
    res.writeHead(302, { Location: `http://127.0.0.1:${(target.address() as any).port}/final` }).end();
  });
  await new Promise<void>(r => hop.listen(0, '127.0.0.1', r));
  const dest = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'redirect-')), 'config.json');
  try {
    await svc._downloadSimple(`http://127.0.0.1:${(hop.address() as any).port}/start`, dest, 'hf_secret');
    assert.equal(fs.readFileSync(dest, 'utf8'), '{"ok":true}');
    assert.deepEqual(seen, [undefined, undefined]);
  } finally { hop.close(); target.close(); }
});

test('cancel removes the partial weights, and a failed job resumes from the partial file', async () => {
  let release!: () => void;
  const gate = new Promise<void>(r => { release = r; });
  const sb = sandbox({
    simple: async (_u, dest) => { fs.writeFileSync(dest, '{}'); },
    weights: async (_u, part) => { fs.writeFileSync(part, '12345'); await gate; },
  });
  try {
    const jobId = svc.startDownload('muscriptor-medium');
    const part = path.join(sb.shared, 'medium', 'model.safetensors.part');
    for (let i = 0; i < 200 && !fs.existsSync(part); i++) await new Promise(r => setTimeout(r, 10));
    assert.equal(svc.cancelDownload(jobId), true);
    release();
    assert.equal(svc.jobs.get(jobId).status, 'cancelled');
    assert.equal(fs.existsSync(part), false);

    const job = svc.jobs.get(jobId);
    job.status = 'failed';
    fs.writeFileSync(part, '1234567');
    svc._downloadWithRedirects = async () => { throw new Error('stop here'); };
    svc.resumeDownload(jobId);
    assert.equal(job.bytesDownloaded, 7);
    await settle(jobId, ['failed']);
  } finally { release(); sb.restore(); }
});

test('deleting one model removes only its own files, in shared and legacy', () => {
  const sb = sandbox();
  try {
    for (const root of [sb.shared, sb.legacy]) {
      put(path.join(root, 'small'), { 'model.safetensors': 'w', 'config.json': '{}', 'notes.txt': 'mine' });
      put(path.join(root, 'medium'), { 'model.safetensors': 'w', 'config.json': '{}' });
    }
    assert.equal(svc.deleteFile('muscriptor-small'), true);
    for (const root of [sb.shared, sb.legacy]) {
      assert.deepEqual(fs.readdirSync(path.join(root, 'small')), ['notes.txt']);
      assert.deepEqual(fs.readdirSync(path.join(root, 'medium')).sort(), ['config.json', 'model.safetensors']);
    }
    assert.deepEqual(installedIds(), ['muscriptor-medium']);
    assert.equal(svc.deleteFile('muscriptor-large'), false, 'nothing there to delete');
    assert.equal(fs.existsSync(path.join(sb.shared, 'large')), false);
  } finally { sb.restore(); }
});

test('deleting leaves Beat This and other variants alone, and an override directory is never deleted from', () => {
  const sb = sandbox();
  try {
    for (const root of [sb.shared, sb.legacy]) {
      put(path.join(root, 'small'), { 'model.safetensors': 'w', 'config.json': '{}' });
      put(path.join(root, 'beat-this'), { 'model.safetensors': 'w', 'config.json': '{}', 'x.txt': 'x' });
    }
    assert.equal(svc.deleteFile('muscriptor-small'), true);
    for (const root of [sb.shared, sb.legacy]) {
      assert.equal(fs.existsSync(path.join(root, 'small')), false);
      assert.deepEqual(fs.readdirSync(path.join(root, 'beat-this')).sort(), ['config.json', 'model.safetensors', 'x.txt']);
    }
  } finally { sb.restore(); }

  const override = fs.mkdtempSync(path.join(os.tmpdir(), 'muscriptor-ovr-'));
  const saved = svc._muscriptorRoots;
  try {
    put(path.join(override, 'small'), { 'model.safetensors': 'w', 'config.json': '{}' });
    svc._muscriptorRoots = () => muscriptorRoots(override, path.join(override, 'm'), path.join(override, 'l'));
    assert.equal(svc.deleteFile('muscriptor-small'), false);
    assert.deepEqual(fs.readdirSync(path.join(override, 'small')).sort(), ['config.json', 'model.safetensors']);
  } finally { svc._muscriptorRoots = saved; }
});

test('a config that cannot be published fails the job and leaves no .part', async () => {
  const sb = sandbox({ simple: async (_u, dest) => { fs.writeFileSync(dest, '{"a":1}'); } });
  try {
    const dir = path.join(sb.shared, 'small');
    put(dir, { 'model.safetensors': 'w' });
    fs.mkdirSync(path.join(dir, 'config.json')); // rename onto a directory throws
    const job = await settle(svc.startDownload('muscriptor-small'));
    assert.equal(job.status, 'failed');
    assert.equal(fs.existsSync(path.join(dir, 'config.json.part')), false);
  } finally { sb.restore(); }
});

test('a model that is downloading cannot be deleted', () => {
  const sb = sandbox();
  try {
    svc.jobs.set('j1', { jobId: 'j1', fileId: 'muscriptor-small', filename: 'model.safetensors', status: 'downloading', bytesDownloaded: 0, totalBytes: 1, speed: 0, speedSamples: [] });
    assert.throws(() => svc.deleteFile('muscriptor-small'), /cancel it first/);
  } finally { sb.restore(); }
});

test('MIDI Studio and the Model Manager share one job per model', async () => {
  let release!: () => void;
  const gate = new Promise<void>(r => { release = r; });
  const sb = sandbox({
    simple: async (_u, dest) => { fs.writeFileSync(dest, '{}'); },
    weights: async () => { await gate; throw new Error('stop'); },
  });
  try {
    assert.deepEqual(svc.startMuscriptorDownload('small'), { started: true });
    assert.deepEqual(svc.startMuscriptorDownload('small'), { started: false, error: 'Download already in progress' });
    const [only] = [...svc.jobs.keys()];
    assert.equal(svc.startDownload('muscriptor-small'), only, 'the Model Manager joins the running job');
    assert.equal(svc.jobs.size, 1);
    assert.equal(svc.muscriptorState('small')?.downloading, true);
  } finally {
    release();
    await settle([...svc.jobs.keys()][0]);
    sb.restore();
  }
});

test('cancelling during the config fetch publishes nothing and the job stays cancelled (fresh and existing weights)', async () => {
  for (const existing of [false, true]) {
    let release!: () => void;
    const gate = new Promise<void>(r => { release = r; });
    let started!: () => void;
    const inFlight = new Promise<void>(r => { started = r; });
    let weights = 0;
    const sb = sandbox({
      simple: async (_u, dest) => { started(); await gate; fs.writeFileSync(dest, '{"late":true}'); },
      weights: async () => { weights++; },
    });
    try {
      const dir = path.join(sb.shared, 'small');
      if (existing) put(dir, { 'model.safetensors': 'w' });
      const jobId = svc.startDownload('muscriptor-small');
      await inFlight;
      assert.equal(svc.cancelDownload(jobId), true);
      release();
      await new Promise(r => setTimeout(r, 50));
      assert.equal(svc.jobs.get(jobId).status, 'cancelled', existing ? 'existing weights' : 'fresh');
      assert.equal(weights, 0);
      assert.equal(fs.existsSync(path.join(dir, 'config.json')), false);
      assert.equal(fs.existsSync(path.join(dir, 'config.json.part')), false);
      assert.deepEqual(installedIds(), []);
    } finally { release(); sb.restore(); }
  }
});

test('cancel aborts the config request itself, so the busy gate does not release over live I/O', async () => {
  let aborted = false;
  const server = http.createServer((req) => { req.on('close', () => { aborted = true; }); /* never answers */ });
  await new Promise<void>(r => server.listen(0, '127.0.0.1', r));
  const ac = new AbortController();
  const dest = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'abort-')), 'config.json.part');
  try {
    const p = svc._downloadSimple(`http://127.0.0.1:${(server.address() as any).port}/x`, dest, undefined, ac.signal);
    await new Promise(r => setTimeout(r, 30));
    ac.abort();
    await assert.rejects(p);
    await new Promise(r => setTimeout(r, 30));
    assert.equal(aborted, true);
  } finally { server.closeAllConnections?.(); server.close(); }
});
