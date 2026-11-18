// Disposable fixtures under the OS temp folder: never the real data root.
import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import Database from 'better-sqlite3';
import { parse as dotenvParse } from 'dotenv';

const base = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-archive-'));
// config.ts (through db/backup.ts) bootstraps a data root at import: a temp one.
process.env.HOT_STEP_DATA_ROOT = path.join(base, 'config-root');
process.env.HOT_STEP_PROFILE_DIR = path.join(base, 'profile');
fs.mkdirSync(process.env.HOT_STEP_DATA_ROOT, { recursive: true });

const express = (await import('express')).default;
const { ExportCancelled, archiveEnvText, exportPreflight, exportWorkspace, hooks, otherLiveServers } = await import('./export.js');
const { createGate } = await import('../updates/maintenance.js');
const { ARCHIVE_BASE, createWorkspaceArchiveRouter } = await import('../../routes/settingsWorkspaceArchive.js');
const { parseArgs } = await import('./cli.js');
type DataChildren = import('../../dataroot/locate.js').DataChildren;
type Manifest = import('./manifest.js').Manifest;

const S = {
  setting: 'SENTINEL-SETTING-7f3a', deleted: 'SENTINEL-DELETED-91bc', hf: 'hf_SENTINELHFTOKEN0042',
  env: 'SENTINEL-ENV-55d1', before: 'SENTINEL-BEFORE-c0de', bak: 'SENTINEL-BAK-e11e',
  inline: 'SENTINEL-INLINE-ab12', adapterEnv: 'SENTINEL-ADAPTER-ENV-3c3c', tokenBak: 'SENTINEL-TOKEN-BAK-4d4d',
  nestedBak: 'SENTINEL-NESTED-BAK-5e5e',
};
const OFF = { training: true, adapters: true, models: false, secrets: false };

function put(file: string, data: string | Buffer): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, data);
}

/** A data root with a separate adapters folder (as on another drive), junk
 *  beside the library and secrets in every place they live. Returns the
 *  children plus the live DB connection (WAL, left open like the app's). */
function fixture() {
  const dir = fs.mkdtempSync(path.join(base, 'fx-'));
  const root = path.join(dir, 'root');
  const lib = path.join(root, 'library');
  const c: DataChildren = {
    library: lib, db: path.join(lib, 'hotstep.db'), audio: path.join(lib, 'audio'), logs: path.join(root, 'logs'),
    models: path.join(root, 'models'), adapters: path.join(dir, 'other-drive', 'Adapters'), training: path.join(lib, 'training'),
    muscriptor: path.join(lib, 'muscriptor'), muscriptorModels: path.join(root, 'models', 'muscriptor'),
    muscriptorLegacyModels: path.join(lib, 'models', 'muscriptor'), plugins: path.join(root, 'plugins'),
    envFile: path.join(root, 'settings', '.env'),
  };
  put(path.join(lib, 'audio', 'a.wav'), crypto.randomBytes(70_000));
  put(path.join(lib, 'audio', 'half.wav.tmp'), 'temp');
  put(path.join(lib, 'references', 'r.wav'), crypto.randomBytes(5_000));
  put(path.join(lib, 'muscriptor', 'hf_token'), S.hf);
  put(path.join(lib, 'peaks', 'p.json'), '{}');
  put(path.join(lib, 'hotstep_backup_tags_2026-08-08.db'), 'old db');
  put(path.join(lib, 'lireek - Copy.db'), 'old');
  put(path.join(lib, 'weird-folder', 'x.txt'), 'x');
  put(path.join(lib, 'training', 'datasets', 'd1', 'dataset.json'), JSON.stringify({ source: path.join(lib, 'training', 'datasets', 'd1') }));
  put(path.join(lib, 'training', '_backup-2026-10-05-mirror', 'x.json'), '{}');
  put(path.join(lib, 'training', 'yue2-aitk-runs.json'), '[]');
  put(path.join(lib, 'training', 'yue2-aitk-runs.json.123.456.tmp'), '[]');
  put(path.join(lib, 'muscriptor', 'hf_token.bak-1'), S.tokenBak);
  put(path.join(lib, 'audio', 'old', 'mix.wav.bak-2'), S.nestedBak);
  put(path.join(c.adapters, 'a1.safetensors'), crypto.randomBytes(3_000));
  put(path.join(c.adapters, 'run1', 'step-100', 'adapter.safetensors'), crypto.randomBytes(2_000));
  put(path.join(c.adapters, 'run1', 'finished'), '');
  put(path.join(c.adapters, '.env'), `OPENAI_API_KEY=${S.adapterEnv}`);
  put(path.join(c.adapters, 'run1', 'notes.xyz'), 'unknown type');
  put(path.join(c.adapters, 'run1', 'backup_2026-01-01', 'adapter.safetensors'), S.nestedBak);
  put(path.join(c.models, 'm.gguf'), crypto.randomBytes(4_000));
  put(path.join(c.models, 'tok', 'tokenizer.json'), '{}');
  put(path.join(c.models, 'tok', 'LICENSE'), 'MIT');
  put(path.join(c.models, 'dit.engine'), 'gpu-built');
  put(path.join(c.models, 'tool.exe'), 'binary');
  put(c.envFile, [
    '# GEMINI_API_KEY=' + S.env + ' (old)', `GEMINI_API_KEY=${S.env}`, 'SERVER_PORT=3001', 'SERVER_HOST=0.0.0.0',
    `GEMINI_MODEL_ALT=model # old API key ${S.inline}`,
    `DATA_DIR=${lib}`, 'CUDA_VISIBLE_DEVICES=1', 'GEMINI_MODEL=gemini-2.5-flash', 'TRAINING_WORKER_TOKEN=abc',
  ].join('\n'));
  put(path.join(root, 'settings', '.env.before-import-1'), `GEMINI_API_KEY=${S.before}`);
  put(path.join(root, 'settings', '.env.bak-1216'), `OPENAI_API_KEY=${S.bak}`);
  const external = path.join(dir, 'Datasets', 'outside.wav');
  put(external, 'outside');

  const db = new Database(c.db);
  db.pragma('journal_mode = WAL');
  db.exec(`CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE album_presets (id INTEGER PRIMARY KEY, adapter_path TEXT, reference_track_path TEXT);
    CREATE TABLE songs (id INTEGER PRIMARY KEY, generation_params TEXT);
    CREATE TABLE server_instances (id TEXT PRIMARY KEY, pid INTEGER NOT NULL, host TEXT NOT NULL, copy TEXT NOT NULL,
      started_at INTEGER NOT NULL, heartbeat_at INTEGER NOT NULL, stopped_at INTEGER);`);
  db.pragma('user_version = 7');
  const set = db.prepare('INSERT INTO settings (key, value) VALUES (?, ?)');
  set.run('geminiApiKey', S.setting);
  set.run('theme', 'dark');
  set.run('openaiApiKey', S.deleted);
  db.prepare('DELETE FROM settings WHERE key = ?').run('openaiApiKey');
  db.prepare('INSERT INTO album_presets (adapter_path, reference_track_path) VALUES (?, ?)').run(path.join(c.adapters, 'a1.safetensors'), external);
  db.prepare('INSERT INTO album_presets (adapter_path, reference_track_path) VALUES (?, ?)').run(path.join(dir, 'gone', 'missing.safetensors'), path.join(c.models, 'm.gguf'));
  db.prepare('INSERT INTO songs (generation_params) VALUES (?)').run(JSON.stringify({ audio: '/audio/abc.wav', ref: external, nested: [{ lora: path.join(c.adapters, 'a1.safetensors') }] }));
  // A committed row that is still only in the WAL when the export runs.
  set.run('lastRow', 'in-wal');
  return { dir, root, c, db, external, outDir: path.join(dir, 'out') };
}

/** Stored-entry reader for test zips (no ZIP64, method 0 only). */
function readZip(buf: Buffer): Map<string, Buffer> {
  const eocd = buf.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
  assert.ok(eocd > 0, 'end of central directory present');
  const count = buf.readUInt16LE(eocd + 10);
  let off = buf.readUInt32LE(eocd + 16);
  const out = new Map<string, Buffer>();
  for (let i = 0; i < count; i++) {
    assert.equal(buf.readUInt32LE(off), 0x02014b50);
    assert.equal(buf.readUInt16LE(off + 10), 0, 'store mode');
    const size = buf.readUInt32LE(off + 20);
    const [n, e, k] = [buf.readUInt16LE(off + 28), buf.readUInt16LE(off + 30), buf.readUInt16LE(off + 32)];
    const lho = buf.readUInt32LE(off + 42);
    const name = buf.toString('utf8', off + 46, off + 46 + n);
    const start = lho + 30 + buf.readUInt16LE(lho + 26) + buf.readUInt16LE(lho + 28);
    out.set(name, buf.subarray(start, start + size));
    off += 46 + n + e + k;
  }
  return out;
}

/** sha256 of every file under `dirs`, keyed by path; SQLite sidecars excluded. */
function treeHash(dirs: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  const walk = (d: string) => {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) walk(p);
      else if (!/-(wal|shm)$/.test(e.name)) out[p] = crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
    }
  };
  dirs.forEach(walk);
  return out;
}

const leftovers = (outDir: string) => fs.existsSync(outDir) ? fs.readdirSync(outDir) : [];
const sha = (b: Buffer) => crypto.createHash('sha256').update(b).digest('hex');

test('export with defaults: hashes verify, secrets absent from the bytes, junk skipped and listed, source unchanged', async () => {
  const f = fixture();
  fs.mkdirSync(f.outDir);
  const before = treeHash([f.root, f.c.adapters]);
  const outFile = path.join(f.outDir, 'ws.zip');
  const phases: string[] = [];
  const r = await exportWorkspace({ children: f.c, dataRoot: f.root, outFile, choice: OFF, appVersion: 'test', onProgress: p => phases.push(p.phase) });
  // Before closing the live connection: its close checkpoints the WAL into the DB file.
  assert.deepEqual(treeHash([f.root, f.c.adapters]), before, 'source tree unchanged');
  f.db.close();

  assert.deepEqual(leftovers(f.outDir), ['ws.zip'], 'no .part or staging left');
  assert.deepEqual([...new Set(phases)], ['database', 'files', 'finishing']);

  const bytes = fs.readFileSync(outFile);
  for (const [k, v] of Object.entries(S)) assert.equal(bytes.indexOf(v), -1, `sentinel ${k} must not be in the zip`);

  const zip = readZip(bytes);
  const names = [...zip.keys()];
  assert.equal(names.at(-1), 'manifest.json', 'manifest is the last entry');
  const m = JSON.parse(zip.get('manifest.json')!.toString()) as Manifest;
  assert.equal(m.kind, 'hot-step-workspace');
  assert.equal(m.platform, process.platform);
  assert.deepEqual(m.categories, { library: true, ...OFF });
  assert.deepEqual(names.slice(0, -1), m.files.map(x => x.name), 'manifest lists every entry in order');
  for (const x of m.files) assert.equal(sha(zip.get(x.name)!), x.sha256, `${x.name} hash`);
  assert.equal(names[0], 'library/hotstep.db');
  for (const want of ['library/audio/a.wav', 'library/references/r.wav', 'library/training/datasets/d1/dataset.json',
    'library/training/yue2-aitk-runs.json', 'adapters/a1.safetensors', 'adapters/run1/step-100/adapter.safetensors', 'settings/.env']) {
    assert.ok(zip.has(want), `${want} carried`);
  }
  assert.ok(!names.some(n => n.startsWith('models/')), 'models off');

  const skipped = Object.fromEntries(m.skipped.map(s => [s.path, s.reason]));
  assert.equal(skipped['library/hotstep_backup_tags_2026-08-08.db'], 'backup');
  assert.equal(skipped['library/lireek - Copy.db'], 'backup');
  assert.equal(skipped['library/weird-folder'], 'unknown');
  assert.equal(skipped['library/peaks'], 'cache');
  assert.equal(skipped['library/audio/half.wav.tmp'], 'temp');
  assert.equal(skipped['library/muscriptor/hf_token'], 'secret');
  assert.equal(skipped['library/training/_backup-2026-10-05-mirror'], 'backup');
  assert.equal(skipped['library/training/yue2-aitk-runs.json.123.456.tmp'], 'temp');
  assert.equal(skipped['settings/.env.before-import-1'], 'backup');
  assert.equal(skipped['settings/.env.bak-1216'], 'backup');
  // At any depth and in user-named folders: backups, credential files, unknown types.
  assert.equal(skipped['library/muscriptor/hf_token.bak-1'], 'backup');
  assert.equal(skipped['library/audio/old/mix.wav.bak-2'], 'backup');
  assert.equal(skipped['adapters/.env'], 'secret');
  assert.equal(skipped['adapters/run1/notes.xyz'], 'unknown');
  assert.equal(skipped['adapters/run1/backup_2026-01-01'], 'backup');
  assert.ok(zip.has('adapters/run1/finished'));

  const env = zip.get('settings/.env')!.toString();
  assert.equal(env, 'GEMINI_MODEL_ALT=model\nGEMINI_MODEL=gemini-2.5-flash\n', 'values only: no comments, inline ones included');
  assert.deepEqual(m.env!.droppedKeys.sort(), ['CUDA_VISIBLE_DEVICES', 'DATA_DIR', 'GEMINI_API_KEY', 'SERVER_HOST', 'SERVER_PORT', 'TRAINING_WORKER_TOKEN']);

  const dbFile = path.join(f.dir, 'snapshot-check.db');
  fs.writeFileSync(dbFile, zip.get('library/hotstep.db')!);
  const snap = new Database(dbFile, { readonly: true });
  assert.equal((snap.pragma('integrity_check') as any[])[0].integrity_check, 'ok');
  assert.deepEqual(snap.prepare('SELECT key FROM settings ORDER BY key').all().map((x: any) => x.key), ['lastRow', 'theme'], 'credential rows removed, WAL row kept');
  assert.equal(snap.pragma('journal_mode', { simple: true }), 'delete', 'self-contained file');
  snap.close();
  assert.deepEqual(m.db, { name: 'library/hotstep.db', userVersion: 7, credentialRowsRemoved: 1 });

  // Externals: exists here and outside every carried folder. Models are off, so m.gguf counts.
  assert.deepEqual(m.externals.map(e => [e.path, e.refs]).sort(), [[path.join(f.c.models, 'm.gguf'), 1], [f.external, 2]].sort());
  const src = Object.fromEntries(m.sources.map(s => [s.prefix, s]));
  assert.equal(src['adapters'].path, f.c.adapters);
  assert.equal(src['adapters'].files, 3);
  assert.equal(src['library/training'].files, 2);
  assert.equal(r.bytes, bytes.length);
});

test('secrets and models on: credentials and hf_token travel, models carried', async () => {
  const f = fixture();
  fs.mkdirSync(f.outDir);
  const outFile = path.join(f.outDir, 'ws.zip');
  const r = await exportWorkspace({ children: f.c, dataRoot: f.root, outFile, choice: { training: false, adapters: false, models: true, secrets: true }, appVersion: 'test' });
  f.db.close();
  const zip = readZip(fs.readFileSync(outFile));
  assert.equal(zip.get('library/muscriptor/hf_token')!.toString(), S.hf);
  assert.match(zip.get('settings/.env')!.toString(), new RegExp(`GEMINI_API_KEY=${S.env}`));
  assert.ok(zip.has('models/m.gguf'));
  assert.ok(zip.has('models/tok/tokenizer.json') && zip.has('models/tok/LICENSE'), 'tokenizers and licences travel');
  assert.equal(r.manifest.skipped.find(s => s.path === 'models/dit.engine')?.reason, 'machine');
  assert.equal(r.manifest.skipped.find(s => s.path === 'models/tool.exe')?.reason, 'machine');
  assert.equal(r.manifest.skipped.find(s => s.path === 'library/muscriptor/hf_token.bak-1')?.reason, 'backup', 'a backup stays out even with Secrets on');
  assert.ok(![...zip.keys()].some(n => n.startsWith('library/training/') || n.startsWith('adapters/')));
  assert.equal(r.manifest.skipped.find(s => s.path === 'library/training')?.reason, 'category-off');
  assert.equal(r.manifest.db!.credentialRowsRemoved, 0);
  assert.equal(fs.readFileSync(outFile).indexOf(S.before), -1, '.env backups never travel');
});

test('a file changed during read fails the export and leaves nothing behind', async () => {
  const f = fixture();
  fs.mkdirSync(f.outDir);
  const wav = path.join(f.c.library, 'audio', 'a.wav');
  hooks.afterRead = abs => { if (abs === wav) fs.appendFileSync(wav, 'more'); };
  try {
    await assert.rejects(exportWorkspace({ children: f.c, dataRoot: f.root, outFile: path.join(f.outDir, 'ws.zip'), choice: OFF, appVersion: 't' }),
      /a\.wav changed while it was being archived/);
  } finally { hooks.afterRead = () => {}; f.db.close(); }
  assert.deepEqual(leftovers(f.outDir), []);
});

test('cancel and a blocker that appears both abort cleanly', async () => {
  const f = fixture();
  fs.mkdirSync(f.outDir);
  const ac = new AbortController();
  hooks.afterRead = () => ac.abort();
  try {
    await assert.rejects(exportWorkspace({ children: f.c, dataRoot: f.root, outFile: path.join(f.outDir, 'a.zip'), choice: OFF, appVersion: 't', signal: ac.signal }),
      (e: unknown) => e instanceof ExportCancelled);
    assert.deepEqual(leftovers(f.outDir), []);

    let calls = 0;
    hooks.afterRead = () => new Promise(r => setTimeout(r, 40));
    await assert.rejects(exportWorkspace({
      children: f.c, dataRoot: f.root, outFile: path.join(f.outDir, 'b.zip'), choice: OFF, appVersion: 't',
      recheckMs: 5, blockers: () => (++calls > 1 ? ['Training job-1: running'] : []),
    }), /work started while it ran \(Training job-1: running\)/);
    assert.deepEqual(leftovers(f.outDir), []);
  } finally { hooks.afterRead = () => {}; f.db.close(); }
});

test('cancel in the middle of reading a big file closes its source stream before returning', async () => {
  const f = fixture();
  fs.mkdirSync(f.outDir);
  const big = path.join(f.c.library, 'audio', 'big.wav');
  fs.writeFileSync(big, Buffer.alloc(32 * 1024 * 1024, 1));
  const ac = new AbortController();
  let stream: fs.ReadStream | null = null;
  hooks.opened = (abs, rs) => { if (abs === big) { stream = rs; rs.once('data', () => ac.abort()); } };
  try {
    await assert.rejects(exportWorkspace({ children: f.c, dataRoot: f.root, outFile: path.join(f.outDir, 'c.zip'), choice: OFF, appVersion: 't', signal: ac.signal }),
      (e: unknown) => e instanceof ExportCancelled);
    assert.ok(stream, 'the big file was opened');
    const s = stream as unknown as fs.ReadStream;
    assert.equal(s.destroyed, true);
    assert.equal(s.closed, true, 'file handle closed before exportWorkspace returned');
    assert.ok(s.bytesRead < 32 * 1024 * 1024, 'stopped mid-file');
    assert.deepEqual(leftovers(f.outDir), []);
  } finally { hooks.opened = () => {}; f.db.close(); }
});

test('a blocker check that throws stops the export instead of crashing', async () => {
  const f = fixture();
  fs.mkdirSync(f.outDir);
  hooks.afterRead = () => new Promise(r => setTimeout(r, 30));
  try {
    await assert.rejects(exportWorkspace({
      children: f.c, dataRoot: f.root, outFile: path.join(f.outDir, 'd.zip'), choice: OFF, appVersion: 't',
      recheckMs: 5, blockers: () => { throw new Error('registry closed'); },
    }), /could not check for running work \(registry closed\)/);
    assert.deepEqual(leftovers(f.outDir), []);
  } finally { hooks.afterRead = () => {}; f.db.close(); }
});

test('preflight refuses a destination inside an exported folder or an existing file', () => {
  const f = fixture();
  f.db.close();
  fs.mkdirSync(f.outDir);
  assert.match(exportPreflight(path.join(f.c.adapters, 'x.zip'), f.c, OFF).join(), /inside the adapters folder/);
  assert.deepEqual(exportPreflight(path.join(f.c.models, 'x.zip'), f.c, OFF), [], 'models off: its folder is a valid destination');
  assert.match(exportPreflight(path.join(f.c.library, 'x.zip'), f.c, OFF).join(), /inside the library folder/);
  fs.writeFileSync(path.join(f.outDir, 'x.zip'), '');
  assert.match(exportPreflight(path.join(f.outDir, 'x.zip'), f.c, OFF).join(), /already exists/);
  assert.match(exportPreflight('relative.zip', f.c, OFF).join(), /absolute path ending in \.zip/);
});

test('archiveEnvText keeps workspace keys only', () => {
  const src = [
    'A=1', '# B_API_KEY=x', 'export OLLAMA_MODEL=llama3', 'VITE_PORT=3000', 'HOT_STEP_DATA_ROOT=x',
    'UNSLOTH_PASSWORD=p', 'ACESTEPCPP_ADAPTERS=/x', 'NAME=model # old key SECRET-IN-COMMENT',
    `SPACED="two words # not a comment"`, `MULTI="line one\nline two"`, `QUOTES="it's"`,
    `RAW='keep \\n literal'`, 'TICKS="a\'b`c\\"d"',
  ].join('\n');
  const r = archiveEnvText(src, false);
  assert.ok(!r.text.includes('SECRET-IN-COMMENT'), 'no comment text survives');
  assert.deepEqual(r.dropped, ['VITE_PORT', 'HOT_STEP_DATA_ROOT', 'UNSLOTH_PASSWORD', 'ACESTEPCPP_ADAPTERS', 'TICKS']);
  // What a later dotenv load reads back is exactly what was kept.
  const want = Object.fromEntries(Object.entries(dotenvParse(src)).filter(([k]) => ['A', 'OLLAMA_MODEL', 'NAME', 'SPACED', 'MULTI', 'QUOTES', 'RAW'].includes(k)));
  assert.deepEqual(dotenvParse(r.text), want);
  assert.equal(want.NAME, 'model');
  assert.equal(want.MULTI, 'line one\nline two');
  assert.match(archiveEnvText('UNSLOTH_PASSWORD=p\n', true).text, /UNSLOTH_PASSWORD=p/);
});

test('otherLiveServers names fresh rows of other live processes only', () => {
  const f = fixture();
  const now = Date.now();
  const ins = f.db.prepare('INSERT INTO server_instances (id, pid, host, copy, started_at, heartbeat_at, stopped_at) VALUES (?, ?, ?, ?, ?, ?, ?)');
  ins.run('self', process.pid, os.hostname(), 'me:3001', now, now, null);
  ins.run('other', process.pid, os.hostname(), 'pc:3002', now, now, null);
  ins.run('stale', process.pid, os.hostname(), 'pc:3003', now, now - 10 * 60_000, null);
  ins.run('stopped', process.pid, os.hostname(), 'pc:3004', now, now, now);
  f.db.close();
  assert.deepEqual(otherLiveServers(f.c.db, 'self'), [`HOT-Step is running in another copy (pc:3002, pid ${process.pid})`]);
  assert.deepEqual(otherLiveServers(path.join(f.dir, 'none.db'), null), []);
});

test('cli parseArgs: defaults, flags and a folder destination', () => {
  const a = parseArgs(['export', '--out', path.join(base, 'x.zip')]);
  assert.deepEqual(a, { out: path.join(base, 'x.zip'), choice: OFF });
  const b = parseArgs(['export', '--out', base, '--no-training', '--models', '--secrets']);
  assert.deepEqual(b.choice, { training: false, adapters: true, models: true, secrets: true });
  assert.match(path.basename(b.out), /^HOT-Step-workspace-.*\.zip$/);
  assert.throws(() => parseArgs(['export']), /usage/);
  assert.throws(() => parseArgs(['export', '--out', 'x', '--bogus']), /unknown argument --bogus/);
});

// ── Route + hold ─────────────────────────────────────────────────────

async function serve(app: ReturnType<typeof express>) {
  const server = http.createServer(app);
  await new Promise<void>(r => server.listen(0, '127.0.0.1', r));
  const url = `http://127.0.0.1:${(server.address() as any).port}`;
  const call = async (method: string, p: string, body?: unknown) => {
    const res = await fetch(url + p, { method, headers: body ? { 'content-type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined });
    return { status: res.status, body: await res.json() as any };
  };
  return { call, close: () => new Promise<void>(r => server.close(() => r())) };
}

function app(f: ReturnType<typeof fixture>, blockers: () => string[] = () => []) {
  const gate = createGate();
  const a = express();
  a.use(gate.middleware);
  a.use(express.json());
  let writes = 0;
  a.post('/api/songs', (_req, res) => { writes++; res.json({ writes }); });
  a.get('/api/songs', (_req, res) => res.json({ writes }));
  a.use(ARCHIVE_BASE, createWorkspaceArchiveRouter({ locate: () => ({ root: f.root, children: f.c }), blockers, gate, appVersion: 'test' }));
  return { a, gate };
}

async function until(fn: () => Promise<boolean>, ms = 5000) {
  const end = Date.now() + ms;
  while (!(await fn())) { if (Date.now() > end) throw new Error('timed out'); await new Promise(r => setTimeout(r, 20)); }
}

test('route: a write during the hold gets 503, GET passes, cancel leaves no zip and releases the hold', async () => {
  const f = fixture();
  fs.mkdirSync(f.outDir);
  const { a, gate } = app(f);
  const s = await serve(a);
  let resume!: () => void;
  const paused = new Promise<void>(r => { resume = r; });
  let reached = false;
  hooks.afterRead = async () => { reached = true; await paused; };
  try {
    const started = await s.call('POST', `${ARCHIVE_BASE}/export`, { destination: f.outDir });
    assert.equal(started.status, 202);
    assert.equal(started.body.job.state, 'running');
    await until(async () => reached);
    assert.equal(gate.held, true);

    const write = await s.call('POST', '/api/songs', {});
    assert.equal(write.status, 503);
    assert.equal(write.body.maintenance.state, 'held');
    assert.equal((await s.call('GET', '/api/songs')).status, 200, 'GET passes the hold');
    assert.equal((await s.call('GET', `${ARCHIVE_BASE}/export`)).body.job.state, 'running');
    assert.equal((await s.call('POST', `${ARCHIVE_BASE}/export`, { destination: f.outDir })).status, 503, 'a second start is a write too');

    assert.equal((await s.call('POST', `${ARCHIVE_BASE}/export/cancel`)).status, 200, 'cancel passes the hold');
    resume();
    await until(async () => (await s.call('GET', `${ARCHIVE_BASE}/export`)).body.job.state === 'cancelled');
    assert.equal(gate.held, false);
    assert.deepEqual(leftovers(f.outDir), []);
    assert.equal((await s.call('POST', '/api/songs', {})).status, 200, 'writes pass after the hold');
  } finally { hooks.afterRead = () => {}; resume(); await s.close(); f.db.close(); }
});

test('route: preview, a finished export, and refusal while work runs', async () => {
  const f = fixture();
  fs.mkdirSync(f.outDir);
  let busy: string[] = ['Training job-9: running'];
  const { a, gate } = app(f, () => busy);
  const s = await serve(a);
  try {
    const pv = await s.call('GET', `${ARCHIVE_BASE}/preview?destination=${encodeURIComponent(f.outDir)}`);
    assert.equal(pv.status, 200);
    assert.equal(pv.body.categories.adapters.files, 3);
    assert.equal(pv.body.categories.models.files, 3);
    assert.equal(pv.body.categories.secrets.files, 2, 'hf_token and adapters/.env');
    assert.ok(pv.body.categories.library.bytes > 70_000, 'library includes the DB');
    assert.deepEqual(pv.body.categories.adapters.paths, [f.c.adapters]);
    assert.deepEqual(pv.body.defaults, OFF);
    assert.equal(typeof pv.body.destination.freeBytes, 'number');
    assert.deepEqual(pv.body.destination.errors, []);
    assert.deepEqual(pv.body.blockers, busy);

    const refused = await s.call('POST', `${ARCHIVE_BASE}/export`, { destination: f.outDir });
    assert.equal(refused.status, 409);
    assert.deepEqual(refused.body.blockers, busy);
    assert.equal(gate.held, false);

    assert.equal((await s.call('POST', `${ARCHIVE_BASE}/export`, { destination: f.outDir, categories: { bogus: true } })).status, 400);
    assert.equal((await s.call('POST', `${ARCHIVE_BASE}/export`, { destination: 'relative' })).status, 400);

    busy = [];
    const ok = await s.call('POST', `${ARCHIVE_BASE}/export`, { destination: f.outDir, categories: { training: false } });
    assert.equal(ok.status, 202);
    await until(async () => (await s.call('GET', `${ARCHIVE_BASE}/export`)).body.job.state === 'done');
    const job = (await s.call('GET', `${ARCHIVE_BASE}/export`)).body.job;
    assert.equal(job.categories.training, false);
    assert.equal(job.bytes, fs.statSync(job.outFile).size);
    assert.equal(job.progress.filesDone, job.progress.filesTotal);
    assert.equal(gate.held, false);
    assert.deepEqual(leftovers(f.outDir), [path.basename(job.outFile)]);
  } finally { await s.close(); f.db.close(); }
});

test.after(() => fs.rmSync(base, { recursive: true, force: true }));
