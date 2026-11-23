// Disposable fixtures under the OS temp folder: never the real data root.
// HS_ARCHIVE_ZIP64=1 adds a round trip through a 5 GB file (ZIP64 sizes and
// offsets; about 15 GB of temp disk for a minute).
import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';
import Database from 'better-sqlite3';
import { parse as dotenvParse } from 'dotenv';

const base = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-import-'));
// config.ts (through db/backup.ts) bootstraps a data root at import: a temp one.
process.env.HOT_STEP_DATA_ROOT = path.join(base, 'config-root');
process.env.HOT_STEP_PROFILE_DIR = path.join(base, 'profile');
fs.mkdirSync(process.env.HOT_STEP_DATA_ROOT, { recursive: true });

const express = (await import('express')).default;
const { exportWorkspace } = await import('./export.js');
const { ImportCancelled, ImportError, REPORT_NAME, hooks, importPreview, importWorkspace, mergeEnvText } = await import('./import.js');
const { createWorkspaceArchiveRouter } = await import('../../routes/settingsWorkspaceArchive.js');
const { parseImportArgs } = await import('./cli.js');
const { isDataLayout } = await import('../../dataroot/paths.js');
type DataChildren = import('../../dataroot/locate.js').DataChildren;
type ImportHere = import('./import.js').ImportHere;

const sha = (b: Buffer | string) => crypto.createHash('sha256').update(b).digest('hex');
function put(file: string, data: string | Buffer): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, data);
}
const TABLES = `CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
  CREATE TABLE album_presets (id INTEGER PRIMARY KEY, adapter_path TEXT, reference_track_path TEXT);
  CREATE TABLE songs (id INTEGER PRIMARY KEY, generation_params TEXT);`;

function childrenOf(root: string, adapters = path.join(root, 'adapters')): DataChildren {
  const lib = path.join(root, 'library');
  return {
    library: lib, db: path.join(lib, 'hotstep.db'), audio: path.join(lib, 'audio'), logs: path.join(root, 'logs'),
    models: path.join(root, 'models'), adapters, training: path.join(lib, 'training'),
    muscriptor: path.join(lib, 'muscriptor'), muscriptorModels: path.join(root, 'models', 'muscriptor'),
    muscriptorLegacyModels: path.join(lib, 'models', 'muscriptor'), plugins: path.join(root, 'plugins'),
    envFile: path.join(root, 'settings', '.env'),
  };
}

/** Machine A's data (adapters on another drive) and machine B's current
 *  data folder, plus an app folder and a free spot for the new folder. */
function world(opts: { extraColumn?: boolean } = {}) {
  const dir = fs.mkdtempSync(path.join(base, 'w-'));
  const a = path.join(dir, 'A');
  const ca = childrenOf(a, path.join(dir, 'other-drive', 'Adapters'));
  const external = path.join(dir, 'Datasets', 'outside.wav');
  put(external, 'outside');
  put(path.join(ca.audio, 'a.wav'), crypto.randomBytes(70_000));
  put(path.join(ca.library, 'references', 'r.wav'), crypto.randomBytes(5_000));
  put(path.join(ca.training, 'datasets', 'd1', 'dataset.json'), JSON.stringify({ source: path.join(ca.training, 'datasets', 'd1'), ref: external }));
  put(path.join(ca.adapters, 'a1.safetensors'), crypto.randomBytes(3_000));
  put(path.join(ca.adapters, 'run1', 'adapter_config.json'), JSON.stringify({ base: path.join(ca.adapters, 'a1.safetensors') }));
  put(ca.envFile, ['GEMINI_API_KEY=A-SECRET-KEY', 'GEMINI_MODEL=gemini-from-A', 'SERVER_PORT=3001', `DATA_DIR=${ca.library}`, 'THEME_ACCENT=teal'].join('\n'));
  const dbA = new Database(ca.db);
  dbA.exec(TABLES + (opts.extraColumn ? `ALTER TABLE songs ADD COLUMN from_the_future TEXT;` : ''));
  dbA.prepare('INSERT INTO settings (key, value) VALUES (?, ?)').run('geminiApiKey', 'A-SECRET-ROW');
  dbA.prepare('INSERT INTO settings (key, value) VALUES (?, ?)').run('theme', 'dark');
  dbA.prepare('INSERT INTO album_presets (adapter_path, reference_track_path) VALUES (?, ?)').run(path.join(ca.adapters, 'a1.safetensors'), external);
  dbA.prepare('INSERT INTO songs (generation_params) VALUES (?)').run(JSON.stringify({ audio: '/audio/abc.wav', nested: [{ lora: path.join(ca.adapters, 'a1.safetensors') }], ds: path.join(ca.training, 'datasets', 'd1') }));
  dbA.close();

  const b = path.join(dir, 'B');
  const cb = childrenOf(b);
  put(path.join(cb.models, 'm.gguf'), 'weights here');
  put(path.join(b, 'tools', 'whisper', 'whisper-cli'), 'bin');
  put(cb.envFile, ['GEMINI_API_KEY=B-OWN-KEY', 'SERVER_PORT=3005', 'GEMINI_MODEL=gemini-from-B', 'CUDA_VISIBLE_DEVICES=1', `DATA_DIR=${cb.library}`, 'LYRICS_EXPORT_DIR=exports/lyrics'].join('\n'));
  fs.mkdirSync(cb.library, { recursive: true });
  const dbB = new Database(cb.db);
  dbB.exec(TABLES);
  dbB.prepare('INSERT INTO settings (key, value) VALUES (?, ?)').run('geminiApiKey', 'B-SECRET-ROW');
  dbB.prepare('INSERT INTO settings (key, value) VALUES (?, ?)').run('openaiApiKey', 'B-OPENAI-ROW');
  dbB.close();
  const app = path.join(dir, 'app');
  fs.mkdirSync(app);
  const here: ImportHere = {
    appRoot: app, dataRoot: b, children: cb,
    paths: { WHISPER_EXE: path.join(b, 'tools', 'whisper', 'whisper-cli'), TENSORRT_LIBS: path.join(b, 'tools', 'runtime-missing') },
  };
  return { dir, a, ca, b, cb, here, app, external, target: path.join(dir, 'New'), zip: path.join(dir, 'out', 'ws.zip') };
}

async function exported(w: ReturnType<typeof world>) {
  fs.mkdirSync(path.dirname(w.zip), { recursive: true });
  await exportWorkspace({ children: w.ca, dataRoot: w.a, outFile: w.zip, choice: { training: true, adapters: true, models: false, secrets: false }, appVersion: 'test' });
  return w.zip;
}

function treeHash(dirs: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  const walk = (d: string) => {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) walk(p); else out[p] = sha(fs.readFileSync(p));
    }
  };
  dirs.forEach(walk);
  return out;
}

async function refused(p: Promise<unknown>, re: RegExp, target?: string) {
  await assert.rejects(p, (e: any) => { assert.ok(e instanceof ImportError, `ImportError, got ${e?.constructor?.name}: ${e?.message}`); assert.match(e.message, re); return true; });
  if (target) assert.ok(!fs.existsSync(target) || fs.readdirSync(target).length === 0, 'nothing left in the target');
}

// ── Hand-made zips for names archiver will not write ─────────────────

interface RawEntry { name: string; data: Buffer | string; mode?: number }
/** Store-mode zip (no ZIP64). `mode` sets unix attributes (a link: 0o120777). */
function rawZip(file: string, entries: RawEntry[]): string {
  const locals: Buffer[] = [];
  const central: Buffer[] = [];
  let off = 0;
  for (const e of entries) {
    const data = Buffer.from(e.data);
    const name = Buffer.from(e.name, 'utf8');
    const crc = zlib.crc32(data);
    const lh = Buffer.alloc(30);
    lh.writeUInt32LE(0x04034b50, 0); lh.writeUInt16LE(20, 4); lh.writeUInt16LE(0x0800, 6); lh.writeUInt16LE(0, 8);
    lh.writeUInt32LE(crc, 14); lh.writeUInt32LE(data.length, 18); lh.writeUInt32LE(data.length, 22); lh.writeUInt16LE(name.length, 26);
    const ch = Buffer.alloc(46);
    ch.writeUInt32LE(0x02014b50, 0); ch.writeUInt16LE(0x031e, 4); ch.writeUInt16LE(20, 6); ch.writeUInt16LE(0x0800, 8);
    ch.writeUInt32LE(crc, 16); ch.writeUInt32LE(data.length, 20); ch.writeUInt32LE(data.length, 24); ch.writeUInt16LE(name.length, 28);
    ch.writeUInt32LE(((e.mode ?? 0o100644) << 16) >>> 0, 38); ch.writeUInt32LE(off, 42);
    locals.push(lh, name, data);
    central.push(ch, name);
    off += 30 + name.length + data.length;
  }
  const cd = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(entries.length, 8); end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(cd.length, 12); end.writeUInt32LE(off, 16);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, Buffer.concat([...locals, cd, end]));
  return file;
}

/** A minimal valid archive's entries: one library file plus its manifest. */
function withManifest(files: RawEntry[], over: Record<string, unknown> = {}): RawEntry[] {
  const manifest = {
    kind: 'hot-step-workspace', format: 1, createdAt: new Date().toISOString(), appVersion: 'test', platform: process.platform,
    dataRoot: path.join(base, 'src-root'), categories: { library: true, training: false, adapters: false, models: false, secrets: false },
    sources: [{ category: 'library', prefix: 'library', path: path.join(base, 'src-root', 'library'), files: files.length, bytes: 0 }],
    files: files.map(f => ({ name: f.name, bytes: Buffer.from(f.data).length, sha256: sha(Buffer.from(f.data)) })),
    db: null, env: null, skipped: [], skippedNotListed: 0, externals: [], ...over,
  };
  return [...files, { name: 'manifest.json', data: JSON.stringify(manifest) }];
}

// ── Round trip ───────────────────────────────────────────────────────

test('round trip A -> zip -> new folder on B: files verified, paths rewritten, settings merged, switch scheduled', async () => {
  const w = world();
  const zip = await exported(w);
  const hereBefore = treeHash([w.b]);
  const scheduled: string[] = [];
  const phases = new Set<string>();
  const r = await importWorkspace({ zipFile: zip, target: w.target, here: w.here, schedule: t => scheduled.push(t), onProgress: p => phases.add(p.phase) });

  assert.deepEqual(scheduled, [w.target]);
  assert.deepEqual([...phases], ['checking', 'extracting', 'rewriting', 'finishing']);
  assert.deepEqual(treeHash([w.b]), hereBefore, "this machine's data folder is untouched");
  assert.ok(isDataLayout(w.target), 'the new folder is a data folder repoint accepts');
  const t = (...p: string[]) => path.join(w.target, ...p);
  assert.equal(sha(fs.readFileSync(t('library', 'audio', 'a.wav'))), sha(fs.readFileSync(path.join(w.ca.audio, 'a.wav'))));
  assert.equal(sha(fs.readFileSync(t('adapters', 'a1.safetensors'))), sha(fs.readFileSync(path.join(w.ca.adapters, 'a1.safetensors'))));
  assert.ok(!fs.existsSync(t('library', 'hotstep.pre-transfer-import.db')), 'no rewrite backup left in the new folder');

  // Stored paths: carried ones follow, outside ones stay and are reported, URLs untouched.
  const db = new Database(t('library', 'hotstep.db'), { readonly: true });
  const preset = db.prepare('SELECT adapter_path, reference_track_path FROM album_presets').get() as any;
  assert.equal(preset.adapter_path, t('adapters', 'a1.safetensors'));
  assert.equal(preset.reference_track_path, w.external);
  const gp = JSON.parse((db.prepare('SELECT generation_params FROM songs').get() as any).generation_params);
  assert.deepEqual(gp, { audio: '/audio/abc.wav', nested: [{ lora: t('adapters', 'a1.safetensors') }], ds: t('library', 'training', 'datasets', 'd1') });
  const settings = Object.fromEntries((db.prepare('SELECT key, value FROM settings').all() as any[]).map(x => [x.key, x.value]));
  db.close();
  assert.deepEqual(settings, { theme: 'dark', geminiApiKey: 'B-SECRET-ROW', openaiApiKey: 'B-OPENAI-ROW' }, "the archive's secret stayed out; this machine's keys are kept");
  assert.equal(r.credentialRowsCopied, 2);
  assert.equal(JSON.parse(fs.readFileSync(t('library', 'training', 'datasets', 'd1', 'dataset.json'), 'utf-8')).source, t('library', 'training', 'datasets', 'd1'));
  assert.equal(JSON.parse(fs.readFileSync(t('adapters', 'run1', 'adapter_config.json'), 'utf-8')).base, t('adapters', 'a1.safetensors'));
  assert.deepEqual([r.pathsRewritten.db, r.pathsRewritten.jsonFiles], [2, 2], "rows and JSON files rewritten");

  // Settings: B's file first, A's values on top, models and tools point at B's folders.
  const env = dotenvParse(fs.readFileSync(t('settings', '.env'), 'utf-8'));
  assert.deepEqual(env, {
    GEMINI_API_KEY: 'B-OWN-KEY', SERVER_PORT: '3005', GEMINI_MODEL: 'gemini-from-A', CUDA_VISIBLE_DEVICES: '1', THEME_ACCENT: 'teal',
    LYRICS_EXPORT_DIR: path.join(w.b, 'exports', 'lyrics'), ACESTEPCPP_MODELS: w.cb.models, WHISPER_EXE: w.here.paths.WHISPER_EXE,
  });
  assert.ok(!Object.values(env).some(v => v.includes('A-SECRET')));

  const report = JSON.parse(fs.readFileSync(t(REPORT_NAME), 'utf-8'));
  assert.equal(report.scheduled, true);
  assert.equal(report.integrity, 'ok');
  assert.deepEqual(report.externals, [{ path: w.external, refs: 1, existsHere: true }]);
  assert.equal(report.files, r.files);
});

test('mergeEnvText: carried categories take the layout default, left-behind ones point here, archive credentials only with secrets', () => {
  const dir = fs.mkdtempSync(path.join(base, 'env-'));
  const adapters = path.join(dir, 'adapters');
  fs.mkdirSync(adapters);
  const carried = { library: true, training: true, adapters: false, models: true, secrets: true };
  const hereRoot = path.join(dir, 'here-root');
  const r = mergeEnvText('TRAINING_DIR=/x/t\nKEEP=1\nLYRICS_EXPORT_DIR=/my/lyrics\nDOWNLOAD_DIR=dl\nACESTEPCPP_EXE=bin/ace\n',
    'OPENAI_API_KEY=from-archive\nACESTEPCPP_EXE=/a/exe\nSERVER_HOST=0.0.0.0\n', carried,
    { ACESTEPCPP_ADAPTERS: adapters, ACESTEPCPP_MODELS: path.join(dir, 'models'), MISSING: path.join(dir, 'nope') }, hereRoot);
  assert.deepEqual(dotenvParse(r.text), {
    KEEP: '1', LYRICS_EXPORT_DIR: '/my/lyrics', DOWNLOAD_DIR: path.join(hereRoot, 'dl'), ACESTEPCPP_EXE: 'bin/ace',
    OPENAI_API_KEY: 'from-archive', ACESTEPCPP_ADAPTERS: adapters,
  }, 'a relative data-folder path keeps its location here; other relative values stay as they are');
  assert.deepEqual(r.fromArchive, ['OPENAI_API_KEY']);
  assert.equal(dotenvParse(mergeEnvText('', 'OPENAI_API_KEY=x\n', { ...carried, secrets: false }, {}, hereRoot).text).OPENAI_API_KEY, undefined);
});

// ── Negatives ────────────────────────────────────────────────────────

test('names: traversal, absolute, backslash, outside a category, case clash and links are refused before anything is written', async () => {
  const w = world();
  const cases: Array<[RawEntry[], RegExp]> = [
    [withManifest([{ name: 'library/../../evil.txt', data: 'x' }]), /invalid relative path/],
    [withManifest([{ name: '/etc/evil.txt', data: 'x' }]), /absolute path/],
    [withManifest([{ name: 'C:/evil.txt', data: 'x' }]), /absolute path/],
    [withManifest([{ name: 'library\\evil.txt', data: 'x' }]), /invalid characters/],
    [withManifest([{ name: 'plugins/evil.lua', data: 'x' }]), /not under library\/, adapters\/ or models\//],
    [withManifest([{ name: 'library/audio/a.txt', data: 'x' }, { name: 'library/audio/A.txt', data: 'y' }]), /differs only in case/],
    [withManifest([{ name: 'library/audio/link', data: '/etc/passwd', mode: 0o120777 }]), /a link/],
    [withManifest([{ name: 'adapters/a.safetensors', data: 'x' }]), /not under a category the archive carries/],
    [[...withManifest([{ name: 'library/audio/a.txt', data: 'x' }]), { name: 'library/audio/extra.txt', data: 'x' }], /in the zip, not in the manifest/],
    [withManifest([{ name: 'library/audio/a.txt', data: 'x' }], { files: [{ name: 'library/audio/a.txt', bytes: 1, sha256: sha('x') }, { name: 'library/audio/gone.txt', bytes: 1, sha256: sha('x') }] }), /missing from the zip/],
  ];
  for (const [i, [entries, re]] of cases.entries()) {
    const zip = rawZip(path.join(w.dir, 'neg', `n${i}.zip`), entries);
    await refused(importWorkspace({ zipFile: zip, target: w.target, here: w.here }), re, w.target);
    assert.ok(!fs.existsSync(w.target), `case ${i}: no folder created`);
  }
  assert.ok(!fs.existsSync(path.join(w.dir, 'evil.txt')) && !fs.existsSync(path.join(w.dir, 'neg', 'evil.txt')));
});

test('missing manifest, another platform, an unknown format: refused', async () => {
  const w = world();
  const z = (n: string, e: RawEntry[]) => rawZip(path.join(w.dir, 'neg', n), e);
  await refused(importWorkspace({ zipFile: z('nomanifest.zip', [{ name: 'library/audio/a.txt', data: 'x' }]), target: w.target, here: w.here }), /no manifest\.json/, w.target);
  const other = process.platform === 'win32' ? 'darwin' : 'win32';
  await refused(importPreview(z('plat.zip', withManifest([{ name: 'library/audio/a.txt', data: 'x' }], { platform: other })), null, w.here), new RegExp(`made on ${other}`));
  await refused(importPreview(z('fmt.zip', withManifest([{ name: 'library/audio/a.txt', data: 'x' }], { format: 2 })), null, w.here), /format 2 is not supported/);
  fs.writeFileSync(path.join(w.dir, 'neg', 'junk.zip'), 'not a zip');
  await refused(importPreview(path.join(w.dir, 'neg', 'junk.zip'), null, w.here), /not a readable zip/);
});

test('a damaged file fails its hash and the new folder is removed', async () => {
  const w = world();
  const zip = await exported(w);
  const buf = fs.readFileSync(zip);
  const at = buf.indexOf(fs.readFileSync(path.join(w.ca.audio, 'a.wav')).subarray(0, 64));
  assert.ok(at > 0);
  buf[at + 10] ^= 0xff;
  const bad = path.join(w.dir, 'out', 'bad.zip');
  fs.writeFileSync(bad, buf);
  await refused(importWorkspace({ zipFile: bad, target: w.target, here: w.here, schedule: () => assert.fail('must not switch') }), /a\.wav does not match its hash/);
  assert.ok(!fs.existsSync(w.target));
});

test('a database from a newer schema is refused after extraction, and the folder removed', async () => {
  const w = world({ extraColumn: true });
  const zip = await exported(w);
  await refused(importWorkspace({ zipFile: zip, target: w.target, here: w.here, schedule: () => assert.fail('must not switch') }), /newer HOT-Step \(songs\.from_the_future is unknown/);
  assert.ok(!fs.existsSync(w.target));
});

test('target checks: non-empty, inside the app, inside or around the current data folder, inside the adapters folder, extra checks', async () => {
  const w = world();
  const zip = await exported(w);
  const busy = path.join(w.dir, 'busy');
  put(path.join(busy, 'x.txt'), 'x');
  await refused(importWorkspace({ zipFile: zip, target: busy, here: w.here }), /is not empty/);
  assert.deepEqual(fs.readdirSync(busy), ['x.txt'], 'a non-empty folder is left as it was');
  await refused(importWorkspace({ zipFile: zip, target: path.join(w.app, 'data'), here: w.here }), /inside the app folder/);
  await refused(importWorkspace({ zipFile: zip, target: path.join(w.b, 'nested'), here: w.here }), /inside the current data folder/);
  await refused(importWorkspace({ zipFile: zip, target: w.target, here: { ...w.here, children: { ...w.cb, adapters: path.join(w.target, 'adapters') } } }), /contains the adapters folder/);
  await refused(importWorkspace({ zipFile: zip, target: 'relative/path', here: w.here }), /absolute path/);
  await refused(importWorkspace({ zipFile: zip, target: path.join(w.dir, 'no', 'parent'), here: w.here }), /does not exist/);
  await refused(importWorkspace({ zipFile: zip, target: w.target, here: w.here, targetErrors: () => ['belongs to another HOT-Step install'] }), /another HOT-Step install/);
  // An empty folder that already exists is used, and left empty by a failure before the switch.
  fs.mkdirSync(w.target);
  hooks.afterFile = () => { throw new Error('disk full'); };
  try {
    await refused(importWorkspace({ zipFile: zip, target: w.target, here: w.here }), /disk full/);
  } finally { hooks.afterFile = () => {}; }
  assert.deepEqual(fs.readdirSync(w.target), [], 'still there, still empty');
});

test('cancel during extraction removes the folder; a failing switch keeps the verified folder and deletes nothing; the switch waits for running work', async () => {
  const w = world();
  const zip = await exported(w);
  const ac = new AbortController();
  hooks.afterFile = () => { ac.abort(); };
  try {
    await assert.rejects(importWorkspace({ zipFile: zip, target: w.target, here: w.here, signal: ac.signal }), (e: any) => e instanceof ImportCancelled);
  } finally { hooks.afterFile = () => {}; }
  assert.ok(!fs.existsSync(w.target));

  // A user file lands in the published library just before scheduling fails.
  const userFile = path.join(w.target, 'library', 'audio', 'mine.wav');
  await assert.rejects(importWorkspace({
    zipFile: zip, target: w.target, here: w.here,
    schedule: () => { fs.writeFileSync(userFile, 'user'); throw Object.assign(new Error('pointer busy'), { status: 409 }); },
  }), (e: any) => {
    assert.ok(e instanceof ImportError && e.status === 409);
    assert.match(e.message, /is imported and verified, but switching to it failed \(pointer busy\)/);
    assert.equal((e.extra.report as { scheduled: boolean }).scheduled, false);
    return true;
  });
  assert.equal(fs.readFileSync(userFile, 'utf-8'), 'user', 'nothing deleted');
  assert.ok(isDataLayout(w.target) && fs.existsSync(path.join(w.target, 'library', 'audio', 'a.wav')), 'the verified folder is kept');
  assert.equal(JSON.parse(fs.readFileSync(path.join(w.target, REPORT_NAME), 'utf-8')).scheduled, false);
  fs.rmSync(w.target, { recursive: true });

  let busy = 2;
  const seen: string[][] = [];
  const done: string[] = [];
  await importWorkspace({
    zipFile: zip, target: w.target, here: w.here, waitMs: 5,
    blockers: () => (busy-- > 0 ? ['GPU work is running'] : []),
    onProgress: p => { if (p.blockers) seen.push(p.blockers); },
    schedule: t => done.push(t),
  });
  assert.deepEqual(done, [w.target]);
  assert.equal(seen.length, 2);
});

test('route: preview reads the manifest and checks the target; import runs to the switch; one job at a time', async () => {
  const w = world();
  const zip = await exported(w);
  const scheduled: string[] = [];
  const app = express();
  app.use(express.json());
  app.use('/wa', createWorkspaceArchiveRouter({
    locate: () => ({ root: w.b, children: w.cb }), blockers: () => [], appVersion: 'test',
    gate: { hold: async () => assert.fail('import never holds writes') },
    imports: { here: () => w.here, targetErrors: () => [], blockers: () => [], schedule: t => { scheduled.push(t); } },
  }));
  const server = http.createServer(app);
  await new Promise<void>(r => server.listen(0, '127.0.0.1', () => r()));
  const url = `http://127.0.0.1:${(server.address() as any).port}/wa`;
  const post = (p: string, body: unknown) => fetch(url + p, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  try {
    const pv = await post('/import/preview', { zip, target: w.target });
    assert.equal(pv.status, 200);
    const pj: any = await pv.json();
    assert.equal(pj.archive.platform, process.platform);
    assert.deepEqual(pj.target.errors, []);
    assert.ok(pj.bytes > 70_000 && pj.files > 5);
    assert.deepEqual(pj.externals, [{ path: w.external, refs: 1, existsHere: true }]);
    assert.equal((await post('/import/preview', { zip: 'rel.zip' })).status, 400);
    const busyTarget: any = await (await post('/import/preview', { zip, target: w.b })).json();
    assert.match(busyTarget.target.errors.join(';'), /not empty/);
    assert.equal((await post('/import', { zip, target: w.b })).status, 400);

    const start = await post('/import', { zip, target: w.target });
    assert.equal(start.status, 202);
    let job: any;
    for (let i = 0; i < 200; i++) {
      job = (await (await fetch(url + '/import')).json() as any).job;
      if (job.state !== 'running') break;
      await new Promise(r => setTimeout(r, 20));
    }
    assert.equal(job.state, 'done', job.error);
    assert.equal(job.report.scheduled, true);
    assert.deepEqual(scheduled, [w.target]);
    assert.equal((await post('/import/cancel', {})).status, 409);
  } finally { server.close(); }
});

test('cli: import arguments', () => {
  assert.deepEqual(parseImportArgs(['import', '--zip', 'a.zip', '--into', 'b']), { zip: path.resolve('a.zip'), into: path.resolve('b') });
  assert.throws(() => parseImportArgs(['import', '--zip', 'a.zip']), /usage/);
  assert.throws(() => parseImportArgs(['import', '--zip', 'a.zip', '--into', 'b', '--switch']), /unknown argument --switch/);
});

const leftovers = (dir: string) => fs.readdirSync(dir).filter(n => n.includes('.hs-import-'));

test('database metadata: a DB the manifest does not declare, a declared DB the zip lacks, and SQLite sidecars are refused', async () => {
  const w = world();
  const z = (n: string, e: RawEntry[]) => rawZip(path.join(w.dir, 'neg', n), e);
  const db = { name: 'library/hotstep.db', userVersion: 0, credentialRowsRemoved: 0 };
  await refused(importWorkspace({ zipFile: z('undeclared.zip', withManifest([{ name: 'library/hotstep.db', data: 'not checked' }])), target: w.target, here: w.here }),
    /manifest\.db does not match the zip \(the zip holds library\/hotstep\.db\)/, w.target);
  await refused(importWorkspace({ zipFile: z('missing.zip', withManifest([{ name: 'library/audio/a.txt', data: 'x' }], { db })), target: w.target, here: w.here }),
    /manifest\.db does not match the zip \(the zip lacks/, w.target);
  for (const [i, n] of ['library/hotstep.db-wal', 'library/hotstep.db-journal', 'library/lireek.db', 'library/training/other.db'].entries()) {
    await refused(importWorkspace({ zipFile: z(`side${i}.zip`, withManifest([{ name: n, data: 'x' }])), target: w.target, here: w.here }), /not a library entry HOT-Step exports/, w.target);
  }
  assert.deepEqual(leftovers(w.dir), []);
});

test('the target is re-checked before the switch: files that appear in it are never touched, and only the staging folder is removed', async () => {
  const w = world();
  const zip = await exported(w);
  fs.mkdirSync(w.target);
  const mine = path.join(w.target, 'mine.txt');
  hooks.afterFile = () => { if (!fs.existsSync(mine)) fs.writeFileSync(mine, 'user file'); };
  try {
    await refused(importWorkspace({ zipFile: zip, target: w.target, here: w.here, schedule: () => assert.fail('must not switch') }), /checked again before the switch: .*is not empty/);
  } finally { hooks.afterFile = () => {}; }
  assert.deepEqual(fs.readdirSync(w.target), ['mine.txt']);
  assert.equal(fs.readFileSync(mine, 'utf-8'), 'user file');
  assert.deepEqual(leftovers(w.dir), [], 'staging folder removed');

  // A folder that appears where there was none is left alone too.
  const fresh = path.join(w.dir, 'Fresh');
  hooks.afterFile = () => { if (!fs.existsSync(fresh)) { fs.mkdirSync(fresh); fs.writeFileSync(path.join(fresh, 'x'), 'x'); } };
  try {
    await refused(importWorkspace({ zipFile: zip, target: fresh, here: w.here }), /is not empty/);
  } finally { hooks.afterFile = () => {}; }
  assert.deepEqual(fs.readdirSync(fresh), ['x']);
  assert.deepEqual(leftovers(w.dir), []);
});

test('ZIP64: a 5 GB file round trips (opt-in: HS_ARCHIVE_ZIP64=1)', { skip: process.env.HS_ARCHIVE_ZIP64 !== '1' }, async () => {
  const w = world();
  const big = path.join(w.ca.adapters, 'big.safetensors');
  const fd = fs.openSync(big, 'w');
  try { fs.ftruncateSync(fd, 5 * 1024 ** 3 + 7); } finally { fs.closeSync(fd); }
  const zip = await exported(w);
  assert.ok(fs.statSync(zip).size > 5 * 1024 ** 3);
  const r = await importWorkspace({ zipFile: zip, target: w.target, here: w.here });
  assert.equal(fs.statSync(path.join(w.target, 'adapters', 'big.safetensors')).size, 5 * 1024 ** 3 + 7);
  // Entries after the 5 GB one sit past the 4 GiB offset limit.
  assert.ok(fs.existsSync(path.join(w.target, 'adapters', 'run1', 'adapter_config.json')));
  assert.equal(r.integrity, 'ok');
  fs.rmSync(w.dir, { recursive: true, force: true });
});

test.after(() => { fs.rmSync(base, { recursive: true, force: true }); });
