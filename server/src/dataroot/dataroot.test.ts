// Disposable fixtures under the OS temp folder; nothing here reads the
// checkout's own data or profile.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { execFileSync } from 'child_process';
import { fileURLToPath } from 'url';
import Database from 'better-sqlite3';
import { resolveRuntime } from './bootstrap.js';
import { dataLayout, devPointerFile, initLayout, profileDir, isInside, otherPointerRoots, newPending, readPointer, rewritePath, writePointer, BUNDLED_DATA_NAMES, POINTER_FILE, LOCK_FILE } from './paths.js';
import { locateData } from './locate.js';
import { hooks, planOperation, rewriteDb, runPending, sharedRoot } from './transfer.js';

const quiet = () => {};

function fixture() {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-runtime-'));
  const app = path.join(base, 'HOT-Step-v2');
  const prof = path.join(base, 'profile');
  fs.mkdirSync(path.join(app, 'server', 'data'), { recursive: true });
  for (const n of BUNDLED_DATA_NAMES) fs.writeFileSync(path.join(app, 'server', 'data', n), 'bundled');
  const env = { HOT_STEP_ROOT: app, HOT_STEP_PROFILE_DIR: prof } as NodeJS.ProcessEnv;
  return { base, app, prof, env, sibling: path.join(base, 'HOT-Step-Data'), ptrFile: path.join(prof, POINTER_FILE) };
}

function write(p: string, text = 'x') {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, text);
}

function makeDb(file: string, rows: { adapter?: string; sourceDir?: string; params?: object }[]) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const db = new Database(file);
  db.exec(`CREATE TABLE songs (id INTEGER PRIMARY KEY, generation_params TEXT);
    CREATE TABLE album_presets (id INTEGER PRIMARY KEY, adapter_path TEXT, reference_track_path TEXT);
    CREATE TABLE training_datasets (id INTEGER PRIMARY KEY, source_dir TEXT NOT NULL, dataset_json_path TEXT NOT NULL DEFAULT '');
    CREATE UNIQUE INDEX idx_src ON training_datasets(source_dir);`);
  for (const r of rows) {
    if (r.adapter) db.prepare('INSERT INTO album_presets (adapter_path, reference_track_path) VALUES (?, ?)').run(r.adapter, 'https://example.com/a.flac');
    if (r.sourceDir) db.prepare('INSERT INTO training_datasets (source_dir) VALUES (?)').run(r.sourceDir);
    if (r.params) db.prepare('INSERT INTO songs (generation_params) VALUES (?)').run(JSON.stringify(r.params));
  }
  db.close();
}

test('dev: no HOT_STEP_ROOT and no dev pointer means no data root, nothing created', async () => {
  const f = fixture();
  const r = await resolveRuntime({ env: { HOT_STEP_PROFILE_DIR: f.prof }, log: quiet, checkoutRoot: f.app });
  assert.ok(r.ok);
  assert.equal(r.runtime.mode, 'dev');
  assert.equal(r.runtime.dataRoot, null);
  assert.equal(r.runtime.envPath, null);
  assert.equal(r.runtime.restartMarkerPath, null);
  assert.equal(r.runtime.pointerPath, devPointerFile(f.prof, f.app));
  assert.ok(!fs.existsSync(f.prof), 'profile untouched');
});

test('path boundaries: sibling prefixes are outside, case is ignored on Windows', () => {
  assert.equal(isInside(path.resolve('/x/data'), path.resolve('/x/data2/a')), false);
  assert.equal(isInside(path.resolve('/x/data'), path.resolve('/x/data/a')), true);
  assert.equal(rewritePath('https://host/x/data/a', [{ from: '/x/data', to: '/y' }]), null);
  if (process.platform === 'win32') {
    assert.equal(rewritePath('d:\\Old\\Lib\\Song.wav', [{ from: 'D:\\old\\lib', to: 'E:\\new' }]), 'E:\\new\\Song.wav');
  }
});

test('fresh unzip, no legacy data: sibling layout created and pointer saved', async () => {
  const f = fixture();
  const r = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
  assert.ok(r.ok);
  assert.equal(r.runtime.dataRoot, f.sibling);
  assert.ok(fs.existsSync(path.join(f.sibling, 'hot-step-data.json')));
  assert.equal(readPointer(f.ptrFile)!.root, f.sibling);
  assert.equal(r.runtime.restartMarkerPath, path.join(f.prof, 'restart-requested'));
});

test('legacy bundle with no .env: <app>/data, models, adapters move by rename and the DB paths follow', async () => {
  const f = fixture();
  const data = path.join(f.app, 'data');
  write(path.join(data, 'audio', 'a.wav'), 'audio');
  write(path.join(f.app, 'models', 'm.gguf'), 'model');
  write(path.join(f.app, 'adapters', 'lm-4b', 'x', 'adapter.safetensors'), 'w');
  write(path.join(f.app, 'logs', '2026-01-01', 'node_console.log'), 'log');
  makeDb(path.join(data, 'hotstep.db'), [
    { adapter: path.join(f.app, 'adapters', 'lm-4b', 'x') },
    { adapter: path.join(f.base, 'external', 'y') },
    { params: { lmAdapter: path.join(f.app, 'adapters', 'lm-4b', 'x'), audio: '/audio/a.wav' } },
  ]);
  const inoBefore = fs.statSync(path.join(f.app, 'models', 'm.gguf')).ino;

  const r = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
  assert.ok(r.ok, !r.ok ? r.error : '');
  const l = dataLayout(f.sibling);
  assert.equal(fs.statSync(path.join(l.models, 'm.gguf')).ino, inoBefore, 'renamed, not copied');
  assert.ok(fs.existsSync(path.join(l.library, 'audio', 'a.wav')));
  assert.ok(fs.existsSync(path.join(l.logs, '2026-01-01', 'node_console.log')));
  assert.ok(!fs.existsSync(path.join(f.app, 'models')));
  for (const n of BUNDLED_DATA_NAMES) assert.ok(fs.existsSync(path.join(f.app, 'server', 'data', n)), 'bundled files stay');

  const db = new Database(path.join(l.library, 'hotstep.db'), { readonly: true });
  const presets = db.prepare('SELECT adapter_path, reference_track_path FROM album_presets ORDER BY id').all() as any[];
  assert.equal(presets[0].adapter_path, path.join(l.adapters, 'lm-4b', 'x'));
  assert.equal(presets[1].adapter_path, path.join(f.base, 'external', 'y'), 'external path untouched');
  assert.equal(presets[0].reference_track_path, 'https://example.com/a.flac', 'URL untouched');
  const params = JSON.parse((db.prepare('SELECT generation_params FROM songs').get() as any).generation_params);
  assert.equal(params.lmAdapter, path.join(l.adapters, 'lm-4b', 'x'));
  assert.equal(params.audio, '/audio/a.wav');
  db.close();
  assert.ok(fs.readdirSync(l.library).some(n => n.includes('.pre-transfer-')), 'DB backed up before rewrite');
  assert.equal(readPointer(f.ptrFile)!.pending, null);

  // A second start finds the pointer and moves nothing.
  const again = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
  assert.ok(again.ok);
  assert.equal(again.runtime.dataRoot, f.sibling);
});

test('legacy .env DATA_DIR=./server/data: user data leaves server/data, settings carried with secrets, app-binary keys dropped', async () => {
  const f = fixture();
  const sd = path.join(f.app, 'server', 'data');
  write(path.join(sd, 'hotstep.db'), '');
  fs.rmSync(path.join(sd, 'hotstep.db'));
  makeDb(path.join(sd, 'hotstep.db'), []);
  write(path.join(sd, 'references', 'r.flac'), 'r');
  write(path.join(sd, 'training', 'datasets', 'a', 'manifest.json'), JSON.stringify({ src: path.join(sd, 'references', 'r.flac') }));
  const external = path.join(f.base, 'ext-adapters');
  fs.mkdirSync(external);
  write(path.join(f.app, '.env'), [
    'DATA_DIR=./server/data', 'GEMINI_API_KEY=secret123', `ACESTEPCPP_ADAPTERS=${external}`,
    `ACESTEPCPP_EXE=${path.join(f.app, 'engine', 'ace-server.exe')}`, 'ACESTEPCPP_PORT=8085',
  ].join('\n'));

  const cwd = process.cwd();
  process.chdir(os.tmpdir()); // nothing may depend on where the process starts
  try {
    const r = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
    assert.ok(r.ok, !r.ok ? r.error : '');
  } finally { process.chdir(cwd); }
  const l = dataLayout(f.sibling);
  assert.ok(fs.existsSync(path.join(l.library, 'hotstep.db')));
  assert.ok(fs.existsSync(path.join(l.library, 'references', 'r.flac')));
  assert.deepEqual(fs.readdirSync(sd).sort(), [...BUNDLED_DATA_NAMES].sort());
  const env = fs.readFileSync(l.envPath, 'utf-8');
  assert.match(env, /GEMINI_API_KEY=secret123/);
  assert.match(env, /ACESTEPCPP_PORT=8085/);
  assert.ok(env.includes(`ACESTEPCPP_ADAPTERS=${external}`), 'external kept');
  assert.doesNotMatch(env, /DATA_DIR=/);
  assert.doesNotMatch(env, /ACESTEPCPP_EXE=/);
  assert.ok(fs.existsSync(path.join(f.app, '.env.migrated')) && !fs.existsSync(path.join(f.app, '.env')));
  const manifest = JSON.parse(fs.readFileSync(path.join(l.library, 'training', 'datasets', 'a', 'manifest.json'), 'utf-8'));
  assert.equal(manifest.src, path.join(l.library, 'references', 'r.flac'));
});

test('missing data folder: falls back with one warning, preserves pointer and uses it again when restored', async () => {
  const f = fixture();
  const gone = path.join(f.base, 'unplugged-drive', 'HOT-Step-Data');
  writePointer(f.ptrFile, { version: 1, root: gone, pending: null });
  const before = fs.readFileSync(f.ptrFile, 'utf-8');
  const logs: string[] = [];
  const r = await resolveRuntime({ env: f.env, log: m => logs.push(m), skipEngineProbe: true });
  assert.ok(r.ok);
  assert.equal(r.runtime.dataRoot, f.sibling);
  assert.equal(r.runtime.libraryDir, dataLayout(f.sibling).library);
  assert.equal(logs.length, 1);
  assert.match(logs[0], /warning/i);
  assert.ok(logs[0].includes(gone) && logs[0].includes(f.sibling));
  assert.equal(fs.readFileSync(f.ptrFile, 'utf-8'), before);
  assert.ok(!fs.existsSync(gone));

  initLayout(dataLayout(gone));
  logs.length = 0;
  const restored = await resolveRuntime({ env: f.env, log: m => logs.push(m) });
  assert.ok(restored.ok);
  assert.equal(restored.runtime.dataRoot, gone);
  assert.deepEqual(logs, []);
  assert.equal(fs.readFileSync(f.ptrFile, 'utf-8'), before);
});

test('malformed pointer: refuses and keeps the pointer', async () => {
  const f = fixture();
  fs.mkdirSync(f.prof, { recursive: true });
  fs.writeFileSync(f.ptrFile, '{ not json');
  const bad = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
  assert.equal(bad.ok, false);
  assert.equal(fs.readFileSync(f.ptrFile, 'utf-8'), '{ not json');
});

test('HOT_STEP_DATA_ROOT wins over a stale pointer and never falls back itself', async () => {
  const f = fixture();
  const gone = path.join(f.base, 'gone');
  writePointer(f.ptrFile, { version: 1, root: gone, pending: null });
  const before = fs.readFileSync(f.ptrFile, 'utf-8');
  const override = path.join(f.base, 'override');
  fs.mkdirSync(override);
  const logs: string[] = [];
  const env = { ...f.env, HOT_STEP_DATA_ROOT: override };
  const r = await resolveRuntime({ env, log: m => logs.push(m) });
  assert.ok(r.ok);
  assert.equal(r.runtime.dataRoot, override);
  assert.equal(r.runtime.override, override);
  assert.ok(logs.every(m => !/warning/i.test(m)));
  assert.equal(fs.readFileSync(f.ptrFile, 'utf-8'), before);
  assert.ok(!fs.existsSync(f.sibling));
  const bad = await resolveRuntime({ env: { ...env, HOT_STEP_DATA_ROOT: gone }, log: quiet });
  assert.equal(bad.ok, false);
  assert.ok(!fs.existsSync(gone));
});

test('unreachable saved root falls back without rewriting the pointer', async t => {
  const f = fixture();
  const root = path.join(f.base, 'unreachable');
  initLayout(dataLayout(root));
  writePointer(f.ptrFile, { version: 1, root, pending: null });
  const before = fs.readFileSync(f.ptrFile, 'utf-8');
  const access = fs.accessSync;
  t.mock.method(fs, 'accessSync', (p: fs.PathLike, mode?: number) => {
    if (p === root) throw Object.assign(new Error('access denied'), { code: 'EACCES' });
    return access(p, mode);
  });
  const logs: string[] = [];
  const r = await resolveRuntime({ env: f.env, log: m => logs.push(m) });
  assert.ok(r.ok);
  assert.equal(r.runtime.dataRoot, f.sibling);
  assert.equal(logs.length, 1);
  assert.ok(logs[0].includes(root) && logs[0].includes(f.sibling));
  assert.equal(fs.readFileSync(f.ptrFile, 'utf-8'), before);
});

test('stale pointer refuses a fallback owned by another install or containing unrelated files', async () => {
  const f = fixture();
  writePointer(f.ptrFile, { version: 1, root: path.join(f.base, 'gone'), pending: null });
  write(path.join(f.sibling, 'unrelated.txt'));
  const unrelated = await resolveRuntime({ env: f.env, log: quiet });
  assert.equal(unrelated.ok, false);
  assert.ok(!fs.existsSync(dataLayout(f.sibling).marker));
  const other = fixture();
  writePointer(other.ptrFile, { version: 1, root: path.join(other.base, 'gone'), pending: null });
  writePointer(devPointerFile(other.prof, other.app), { version: 1, root: other.sibling, pending: null, checkout: other.app });
  const shared = await resolveRuntime({ env: other.env, log: quiet });
  assert.equal(shared.ok, false);
  assert.ok(!fs.existsSync(other.sibling));
});

test('dev stale pointer falls back to checkout paths with one warning and preserves pointer', async () => {
  const f = fixture();
  const ptrFile = devPointerFile(f.prof, f.app);
  const gone = path.join(f.base, 'gone');
  writePointer(ptrFile, { version: 1, root: gone, pending: null, checkout: f.app });
  const before = fs.readFileSync(ptrFile, 'utf-8');
  const logs: string[] = [];
  const r = await resolveRuntime({ env: { HOT_STEP_PROFILE_DIR: f.prof }, checkoutRoot: f.app, log: m => logs.push(m) });
  assert.ok(r.ok);
  assert.equal(r.runtime.dataRoot, null);
  assert.equal(r.runtime.envPath, null);
  assert.equal(logs.length, 1);
  assert.ok(logs[0].includes(gone) && logs[0].includes(f.app));
  assert.match(logs[0], /warning/i);
  assert.equal(fs.readFileSync(ptrFile, 'utf-8'), before);
  assert.ok(!fs.existsSync(gone));
});

test('legacy data next to a populated sibling refuses instead of choosing', async () => {
  const f = fixture();
  write(path.join(f.app, 'data', 'audio', 'a.wav'));
  write(path.join(f.sibling, 'library', 'hotstep.db'));
  const r = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
  assert.equal(r.ok, false);
  assert.ok(fs.existsSync(path.join(f.app, 'data', 'audio', 'a.wav')));
  assert.ok(!fs.existsSync(f.ptrFile));
});

test('crash between a rename and the journal write resumes; a real conflict stops without overwriting', () => {
  const f = fixture();
  const oldRoot = path.join(f.base, 'old-root');
  const newRoot = path.join(f.base, 'new-root');
  write(path.join(oldRoot, 'library', 'a.txt'), 'a');
  write(path.join(oldRoot, 'models', 'm.gguf'), 'm');
  const { op, errors } = planOperation('move', newRoot, { appRoot: f.app, currentRoot: oldRoot, currentDb: null });
  assert.deepEqual(errors, []);
  writePointer(f.ptrFile, { version: 1, root: oldRoot, pending: op });
  // Simulate the crash: "library" was renamed, the journal still says pending.
  fs.mkdirSync(newRoot, { recursive: true });
  fs.renameSync(path.join(oldRoot, 'library'), path.join(newRoot, 'library'));
  const ptr = readPointer(f.ptrFile)!;
  runPending(f.ptrFile, ptr, f.app, quiet);
  assert.equal(readPointer(f.ptrFile)!.root, newRoot);
  assert.ok(fs.existsSync(path.join(newRoot, 'library', 'a.txt')));
  assert.ok(fs.existsSync(path.join(newRoot, 'models', 'm.gguf')));

  // Both sides hold the same name: stop, keep both, record the error.
  write(path.join(oldRoot, 'models', 'dup.gguf'), 'old');
  const third = path.join(f.base, 'third-root');
  write(path.join(third, 'models', 'dup.gguf'), 'other');
  const p2 = { version: 1 as const, root: oldRoot, pending: { ...op, id: 'x2', destination: third, mappings: [{ name: 'data', from: oldRoot, to: third, state: 'pending' as const }] } };
  writePointer(f.ptrFile, p2);
  assert.throws(() => runPending(f.ptrFile, p2, f.app, quiet), /refusing to guess/);
  assert.equal(fs.readFileSync(path.join(oldRoot, 'models', 'dup.gguf'), 'utf-8'), 'old');
  assert.equal(fs.readFileSync(path.join(third, 'models', 'dup.gguf'), 'utf-8'), 'other');
  const kept = readPointer(f.ptrFile)!;
  assert.ok(kept.pending && kept.pending.error);
  assert.equal(kept.root, oldRoot);
});

test('cross-volume: copies, verifies sizes, keeps the source on failure, cleans only after success', () => {
  const f = fixture();
  const oldRoot = path.join(f.base, 'old-root');
  const newRoot = path.join(f.base, 'new-root');
  write(path.join(oldRoot, 'library', 'big.bin'), 'x'.repeat(4096));
  write(path.join(oldRoot, 'library', 'sub', 'b.txt'), 'b');
  const outside = path.join(f.base, 'outside-tree');
  write(path.join(outside, 'huge.bin'), 'not ours');
  fs.symlinkSync(outside, path.join(oldRoot, 'library', 'link'), 'junction');
  const { op } = planOperation('move', newRoot, { appRoot: f.app, currentRoot: oldRoot, currentDb: null });
  const ptr = { version: 1 as const, root: oldRoot, pending: op };
  writePointer(f.ptrFile, ptr);
  // An obstacle where a file must land makes the copy fail part way.
  fs.mkdirSync(path.join(newRoot, 'library', 'big.bin'), { recursive: true });
  const real = hooks.sameVolume;
  hooks.sameVolume = () => false;
  try {
    assert.throws(() => runPending(f.ptrFile, ptr, f.app, quiet));
    assert.ok(fs.existsSync(path.join(oldRoot, 'library', 'big.bin')), 'source kept after failure');
    assert.equal(readPointer(f.ptrFile)!.root, oldRoot);
    fs.rmSync(path.join(newRoot, 'library', 'big.bin'), { recursive: true });
    const resumed = readPointer(f.ptrFile)!;
    runPending(f.ptrFile, resumed, f.app, quiet);
  } finally { hooks.sameVolume = real; }
  assert.equal(fs.statSync(path.join(newRoot, 'library', 'big.bin')).size, 4096);
  assert.ok(fs.existsSync(path.join(newRoot, 'library', 'sub', 'b.txt')));
  assert.ok(fs.lstatSync(path.join(newRoot, 'library', 'link')).isSymbolicLink(), 'link recreated, not followed');
  assert.ok(!fs.existsSync(path.join(oldRoot, 'library')), 'source cleaned after success');
  assert.equal(fs.readFileSync(path.join(outside, 'huge.bin'), 'utf-8'), 'not ours', 'link target untouched');
  assert.equal(readPointer(f.ptrFile)!.root, newRoot);
});

test('cross-volume cleanup removes non-ASCII source entries', () => {
  const f = fixture();
  const oldRoot = path.join(f.base, 'old-root');
  const newRoot = path.join(f.base, 'new-root');
  const rel = path.join('library', 'Folder \u2014 Caf\u00e9', 'track.txt');
  write(path.join(oldRoot, rel), 'keep');
  const { op } = planOperation('move', newRoot, { appRoot: f.app, currentRoot: oldRoot, currentDb: null });
  const ptr = { version: 1 as const, root: oldRoot, pending: op };
  const real = hooks.sameVolume;
  hooks.sameVolume = () => false;
  try { runPending(f.ptrFile, ptr, f.app, quiet); }
  finally { hooks.sameVolume = real; }
  assert.equal(fs.existsSync(path.join(oldRoot, rel)), false);
  assert.equal(fs.readFileSync(path.join(newRoot, rel), 'utf8'), 'keep');
  assert.equal(readPointer(f.ptrFile)!.pending, null);
});

test('a cleanup failure warns without failing a verified move', t => {
  const f = fixture();
  const oldRoot = path.join(f.base, 'old-root');
  const newRoot = path.join(f.base, 'new-root');
  const source = path.join(oldRoot, 'library', 'locked.txt');
  write(source, 'keep');
  const { op } = planOperation('move', newRoot, { appRoot: f.app, currentRoot: oldRoot, currentDb: null });
  const ptr = { version: 1 as const, root: oldRoot, pending: op };
  const real = hooks.sameVolume;
  const rm = fs.rmSync;
  const failure = t.mock.method(fs, 'rmSync', (...args: Parameters<typeof fs.rmSync>) => {
    if (isInside(oldRoot, String(args[0]))) throw new Error('access denied');
    return rm(...args);
  });
  const logs: string[] = [];
  hooks.sameVolume = () => false;
  try { runPending(f.ptrFile, ptr, f.app, line => logs.push(line)); }
  finally { hooks.sameVolume = real; failure.mock.restore(); }
  assert.equal(readPointer(f.ptrFile)!.root, newRoot);
  assert.equal(readPointer(f.ptrFile)!.pending, null);
  assert.equal(fs.readFileSync(path.join(newRoot, 'library', 'locked.txt'), 'utf8'), 'keep');
  assert.ok(fs.existsSync(source));
  assert.ok(logs.some(line => line.includes('WARNING') && line.includes(oldRoot) && line.includes('access denied')));
});

test('DB rewrite: a unique-key collision rolls back everything', () => {
  const f = fixture();
  const from = path.join(f.base, 'old');
  const to = path.join(f.base, 'new');
  const db = path.join(f.base, 'lib', 'hotstep.db');
  makeDb(db, [
    { adapter: path.join(from, 'a') },
    { sourceDir: path.join(from, 'ds') },
    { sourceDir: path.join(to, 'ds') },
  ]);
  assert.throws(() => rewriteDb(db, [{ from, to }], 't1', quiet), /collides/);
  const conn = new Database(db, { readonly: true });
  assert.equal((conn.prepare('SELECT adapter_path FROM album_presets').get() as any).adapter_path, path.join(from, 'a'));
  conn.close();
});

test('lock: a stale lock is cleared, a live one blocks the transfer', async () => {
  const f = fixture();
  const oldRoot = path.join(f.base, 'old-root');
  write(path.join(oldRoot, 'library', 'a.txt'));
  const { op } = planOperation('move', path.join(f.base, 'new-root'), { appRoot: f.app, currentRoot: oldRoot, currentDb: null });
  writePointer(f.ptrFile, { version: 1, root: oldRoot, pending: op });
  fs.writeFileSync(path.join(f.prof, LOCK_FILE), String(process.pid));
  const blocked = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
  assert.equal(blocked.ok, false);
  assert.ok(fs.existsSync(path.join(oldRoot, 'library', 'a.txt')));
  fs.writeFileSync(path.join(f.prof, LOCK_FILE), '999999');
  const ok = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
  assert.ok(ok.ok, !ok.ok ? ok.error : '');
  assert.equal(ok.runtime.dataRoot, path.join(f.base, 'new-root'));
  assert.ok(!fs.existsSync(path.join(f.prof, LOCK_FILE)));
});

test('repoint needs an existing layout and moves nothing; move refuses targets inside the app', () => {
  const f = fixture();
  const cur = path.join(f.base, 'cur');
  initLayout(dataLayout(cur));
  assert.ok(planOperation('repoint', path.join(f.base, 'nothing'), { appRoot: f.app, currentRoot: cur, currentDb: null }).errors.length);
  assert.ok(planOperation('move', path.join(f.app, 'data2'), { appRoot: f.app, currentRoot: cur, currentDb: null }).errors.length);
  assert.ok(planOperation('move', path.join(cur, 'inner'), { appRoot: f.app, currentRoot: cur, currentDb: null }).errors.length);
  const other = path.join(f.base, 'other');
  initLayout(dataLayout(other));
  write(path.join(cur, 'library', 'keep.txt'));
  const { op, errors } = planOperation('repoint', other, { appRoot: f.app, currentRoot: cur, currentDb: null });
  assert.deepEqual(errors, []);
  const ptr = { version: 1 as const, root: cur, pending: op };
  writePointer(f.ptrFile, ptr);
  runPending(f.ptrFile, ptr, f.app, quiet);
  assert.equal(readPointer(f.ptrFile)!.root, other);
  assert.ok(fs.existsSync(path.join(cur, 'library', 'keep.txt')));
});

test('import from an older install: blank library set aside, data moved in, unshipped plugins copied', () => {
  const f = fixture();
  const cur = path.join(f.base, 'HOT-Step-Data');
  initLayout(dataLayout(cur));
  makeDb(path.join(cur, 'library', 'hotstep.db'), []);
  const oldApp = path.join(f.base, 'HOT-Step-v1');
  write(path.join(oldApp, 'data', 'audio', 'old.wav'));
  makeDb(path.join(oldApp, 'data', 'hotstep.db'), [{ params: { p: path.join(oldApp, 'data', 'audio', 'old.wav') } }]);
  write(path.join(oldApp, 'engine', 'plugins', 'solvers', 'mine.lua'), '-- user');
  write(path.join(oldApp, 'engine', 'plugins', 'solvers', 'euler.lua'), '-- shipped');
  write(path.join(f.app, 'engine', 'plugins', 'solvers', 'euler.lua'), '-- shipped');
  const { op, errors } = planOperation('import', oldApp, { appRoot: f.app, currentRoot: cur, currentDb: path.join(cur, 'library', 'hotstep.db') });
  assert.deepEqual(errors, []);
  const ptr = { version: 1 as const, root: cur, pending: op };
  writePointer(f.ptrFile, ptr);
  runPending(f.ptrFile, ptr, f.app, quiet);
  const l = dataLayout(cur);
  assert.ok(fs.existsSync(path.join(l.library, 'audio', 'old.wav')));
  assert.ok(fs.existsSync(path.join(l.plugins, 'solvers', 'mine.lua')));
  assert.ok(!fs.existsSync(path.join(l.plugins, 'solvers', 'euler.lua')));
  assert.ok(fs.readdirSync(cur).some(n => n.startsWith('library.blank-')));
  const conn = new Database(path.join(l.library, 'hotstep.db'), { readonly: true });
  assert.equal(JSON.parse((conn.prepare('SELECT generation_params FROM songs').get() as any).generation_params).p, path.join(l.library, 'audio', 'old.wav'));
  conn.close();
});

const SERVER_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

// config.ts evaluated for real in a child, so the import-order gate is covered too.
function loadConfig(env: NodeJS.ProcessEnv): any {
  const script = `const m = await import(${JSON.stringify(new URL('../config.ts', import.meta.url).href)});
console.log('@@' + JSON.stringify({ data: m.config.data.dir, training: m.config.training.dir, models: m.config.aceServer.models,
  adapters: m.config.aceServer.adapters, env: m.ENV_FILE_PATH, logs: m.LOGS_DIR, marker: m.RESTART_MARKER_PATH,
  trt: m.config.aceServer.trtLibs, portable: m.PORTABLE_MODE, root: m.PROJECT_ROOT, exe: m.config.aceServer.exe }));`;
  const clean = { ...process.env };
  for (const k of ['HOT_STEP_ROOT', 'HOT_STEP_DATA_ROOT', 'HOT_STEP_PROFILE_DIR', 'DATA_DIR', 'ACESTEPCPP_MODELS', 'ACESTEPCPP_ADAPTERS', 'TRAINING_DIR', 'TENSORRT_LIBS']) delete clean[k];
  const out = execFileSync(process.execPath, ['--import', 'tsx', '--input-type=module', '-e', script], {
    env: { ...clean, ...env }, cwd: SERVER_DIR, encoding: 'utf-8', timeout: 60_000,
  });
  return JSON.parse(out.split('\n').find(l => l.startsWith('@@'))!.slice(2));
}

test('config in dev keeps the checkout layout', () => {
  const serverDir = SERVER_DIR;
  const c = loadConfig({ HOT_STEP_PROFILE_DIR: fixture().prof });
  assert.equal(c.portable, false);
  assert.equal(c.root, path.dirname(serverDir));
  assert.equal(c.env, path.join(path.dirname(serverDir), '.env'));
  assert.equal(c.logs, path.join(path.dirname(serverDir), 'logs'));
  assert.equal(c.marker, path.join(path.dirname(serverDir), '.restart-requested'));
});

test('config in a release puts every writable default under the data folder', () => {
  const f = fixture();
  write(path.join(f.app, '.env.example'), 'DATA_DIR=./server/data\nSERVER_PORT=3001\n');
  const c = loadConfig(f.env);
  const l = dataLayout(f.sibling);
  assert.equal(c.portable, true);
  assert.equal(c.root, f.app);
  assert.equal(c.data, l.library);
  assert.equal(c.training, path.join(l.library, 'training'));
  assert.equal(c.models, l.models);
  assert.equal(c.adapters, l.adapters);
  assert.equal(c.env, l.envPath);
  assert.equal(c.logs, l.logs);
  assert.equal(c.trt, l.runtimeLibDir);
  assert.equal(c.marker, path.join(f.prof, 'restart-requested'));
  assert.doesNotMatch(fs.readFileSync(l.envPath, 'utf-8'), /^DATA_DIR=/m, 'example DATA_DIR dropped');
  for (const p of [c.data, c.models, c.adapters, c.env, c.logs]) assert.ok(!isInside(f.app, p), `${p} outside the app`);
});

// ── Review regressions ───────────────────────────────────────────────

test('an unreachable source stops the move; a finished rename with no source left still completes', () => {
  const f = fixture();
  const oldRoot = path.join(f.base, 'old-root');
  const newRoot = path.join(f.base, 'new-root');
  write(path.join(oldRoot, 'library', 'a.txt'));
  const { op } = planOperation('move', newRoot, { appRoot: f.app, currentRoot: oldRoot, currentDb: null });
  const ptr = { version: 1 as const, root: oldRoot, pending: op };
  writePointer(f.ptrFile, ptr);
  fs.renameSync(oldRoot, oldRoot + '-unplugged');
  assert.throws(() => runPending(f.ptrFile, ptr, f.app, quiet), /not reachable/);
  const kept = readPointer(f.ptrFile)!;
  assert.equal(kept.root, oldRoot);
  assert.ok(kept.pending && kept.pending.mappings[0].state === 'pending');

  // A journaled rename that finished and removed the emptied source: that is landed data.
  kept.pending!.mappings[0].method = 'rename';
  fs.mkdirSync(newRoot, { recursive: true });
  fs.renameSync(path.join(oldRoot + '-unplugged', 'library'), path.join(newRoot, 'library'));
  runPending(f.ptrFile, kept, f.app, quiet);
  assert.equal(readPointer(f.ptrFile)!.root, newRoot);
});

test('a target reached through a junction into the app or the source is refused, at plan time and again before moving', () => {
  const f = fixture();
  const cur = path.join(f.base, 'cur');
  write(path.join(cur, 'library', 'a.txt'));
  const ctx = { appRoot: f.app, currentRoot: cur, currentDb: null };
  const intoApp = path.join(f.base, 'looks-outside');
  fs.symlinkSync(f.app, intoApp, 'junction');
  assert.ok(planOperation('move', path.join(intoApp, 'Data'), ctx).errors.some(e => /inside the app/.test(e)));
  const intoSource = path.join(f.base, 'alias-of-cur');
  fs.symlinkSync(cur, intoSource, 'junction');
  assert.ok(planOperation('move', path.join(intoSource, 'Data'), ctx).errors.some(e => /overlap/.test(e)));

  // Passes at plan time, then the parent is swapped for a junction into the app.
  const parent = path.join(f.base, 'later');
  fs.mkdirSync(parent);
  const { op, errors } = planOperation('move', path.join(parent, 'Data'), ctx);
  assert.deepEqual(errors, []);
  const ptr = { version: 1 as const, root: cur, pending: op };
  writePointer(f.ptrFile, ptr);
  fs.rmdirSync(parent);
  fs.symlinkSync(f.app, parent, 'junction');
  assert.throws(() => runPending(f.ptrFile, ptr, f.app, quiet), /inside the app/);
  assert.ok(fs.existsSync(path.join(cur, 'library', 'a.txt')));
  assert.ok(!fs.existsSync(path.join(f.app, 'Data')));
});

test('import into a configured root: old settings win, current ones are kept as a backup', () => {
  const f = fixture();
  const cur = path.join(f.base, 'HOT-Step-Data');
  initLayout(dataLayout(cur));
  write(dataLayout(cur).envPath, 'GEMINI_API_KEY=new-install\n');
  const oldApp = path.join(f.base, 'HOT-Step-v1');
  write(path.join(oldApp, 'data', 'audio', 'old.wav'));
  write(path.join(oldApp, '.env'), 'GEMINI_API_KEY=old-install\n');
  const { op, errors } = planOperation('import', oldApp, { appRoot: f.app, currentRoot: cur, currentDb: null });
  assert.deepEqual(errors, []);
  const ptr = { version: 1 as const, root: cur, pending: op };
  writePointer(f.ptrFile, ptr);
  runPending(f.ptrFile, ptr, f.app, quiet);
  const l = dataLayout(cur);
  assert.match(fs.readFileSync(l.envPath, 'utf-8'), /old-install/);
  const backup = fs.readdirSync(l.settingsDir).find(n => n.startsWith('.env.before-import-'))!;
  assert.match(fs.readFileSync(path.join(l.settingsDir, backup), 'utf-8'), /new-install/);
  assert.ok(fs.existsSync(path.join(l.library, 'audio', 'old.wav')));
});

test('migration that stops on a DB collision resumes once the collision is fixed', async () => {
  const f = fixture();
  const data = path.join(f.app, 'data');
  const lib = dataLayout(f.sibling).library;
  makeDb(path.join(data, 'hotstep.db'), [
    { sourceDir: path.join(data, 'training', 'x') },
    { sourceDir: path.join(lib, 'training', 'x') },
  ]);
  write(path.join(f.app, '.env'), 'GEMINI_API_KEY=secret123\n');
  const first = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
  assert.equal(first.ok, false);
  assert.match((first as any).error, /collides/);
  assert.ok(readPointer(f.ptrFile)!.pending, 'pending kept');

  const db = new Database(path.join(lib, 'hotstep.db'));
  db.prepare('DELETE FROM training_datasets WHERE source_dir = ?').run(path.join(lib, 'training', 'x'));
  db.close();
  const second = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
  assert.ok(second.ok, !second.ok ? second.error : '');
  assert.match(fs.readFileSync(dataLayout(f.sibling).envPath, 'utf-8'), /GEMINI_API_KEY=secret123/);
  const conn = new Database(path.join(lib, 'hotstep.db'), { readonly: true });
  assert.equal((conn.prepare('SELECT source_dir FROM training_datasets').get() as any).source_dir, path.join(lib, 'training', 'x'));
  conn.close();
});

test('a copy cut short whose source then goes missing keeps pending and stops', () => {
  const f = fixture();
  const oldRoot = path.join(f.base, 'old-root');
  const newRoot = path.join(f.base, 'new-root');
  write(path.join(oldRoot, 'library', 'a.wav'), 'a');
  write(path.join(oldRoot, 'library', 'b.wav'), 'b');
  const { op } = planOperation('move', newRoot, { appRoot: f.app, currentRoot: oldRoot, currentDb: null });
  const ptr = { version: 1 as const, root: oldRoot, pending: op };
  writePointer(f.ptrFile, ptr);
  // a.wav lands, b.wav hits an obstacle.
  fs.mkdirSync(path.join(newRoot, 'library', 'b.wav'), { recursive: true });
  const real = hooks.sameVolume;
  hooks.sameVolume = () => false;
  try {
    assert.throws(() => runPending(f.ptrFile, ptr, f.app, quiet));
    assert.ok(fs.existsSync(path.join(newRoot, 'library', 'a.wav')));
    assert.equal(readPointer(f.ptrFile)!.pending!.mappings[0].method, 'copy', 'method journaled');
    fs.renameSync(oldRoot, oldRoot + '-unplugged');
    const again = readPointer(f.ptrFile)!;
    assert.throws(() => runPending(f.ptrFile, again, f.app, quiet), /not known to be complete/);
  } finally { hooks.sameVolume = real; }
  const kept = readPointer(f.ptrFile)!;
  assert.equal(kept.root, oldRoot);
  assert.equal(kept.pending!.mappings[0].state, 'pending');
  assert.ok(fs.existsSync(path.join(oldRoot + '-unplugged', 'library', 'b.wav')));
});

test('job-map busy checks block until every job has finished', async () => {
  const { registerJobsBusyCheck, getBusyBlockers } = await import('./pointer.js');
  const jobs = new Map([['a', { status: 'running' }], ['b', { status: 'done' }]]);
  registerJobsBusyCheck('fixture', jobs);
  assert.ok(getBusyBlockers().includes('1 fixture job(s) running or queued'));
  jobs.get('a')!.status = 'succeeded';
  assert.ok(!getBusyBlockers().some(b => b.includes('fixture')));
});

test('trackBusy keeps work busy after its map entry is deleted, until it settles, without leaking rejections', async () => {
  const { registerJobsBusyCheck, trackBusy, getBusyBlockers } = await import('./pointer.js');
  const jobs = new Map([['s', { status: 'separating' }]]);
  registerJobsBusyCheck('fixture-stem', jobs);
  let finish!: (v: unknown) => void, fail!: (e: Error) => void;
  void trackBusy('fixture-stem', new Promise(r => { finish = r; }));
  const failing = trackBusy('fixture-stem', new Promise((_r, j) => { fail = j; }));
  jobs.delete('s'); // DELETE /api/stem-studio/jobs/:id while the engine call is pending
  assert.ok(getBusyBlockers().includes('2 fixture-stem job(s) still running'));
  finish(undefined);
  fail(new Error('engine gone'));
  await assert.rejects(failing, /engine gone/);
  await new Promise(r => setImmediate(r));
  assert.ok(!getBusyBlockers().some(b => b.includes('fixture-stem')));
});

// ── Dev checkouts ────────────────────────────────────────────────────

/** A disposable checkout: git-initialised, with tracked server/src/data and
 *  tools files, and the untracked folders a dev checkout fills at runtime. */
function devCheckout() {
  const f = fixture();
  const co = path.join(f.base, 'checkout');
  write(path.join(co, 'server', 'src', 'data', 'model-registry.json'), '{}');
  write(path.join(co, 'tools', 'cg.cmd'), 'tracked');
  write(path.join(co, '.env.example'), 'DATA_DIR=./server/data\n');
  execFileSync('git', ['init', '-q'], { cwd: co });
  execFileSync('git', ['add', 'server/src/data/model-registry.json', 'tools/cg.cmd', '.env.example'], { cwd: co });
  return { ...f, co, env: { HOT_STEP_PROFILE_DIR: f.prof } as NodeJS.ProcessEnv, ptrFile: devPointerFile(f.prof, co) };
}

test('dev and release pointers are separate files, keyed per checkout, and never read each other', async () => {
  const f = devCheckout();
  const other = path.join(f.base, 'checkout2');
  assert.notEqual(devPointerFile(f.prof, f.co), devPointerFile(f.prof, other));
  assert.notEqual(path.basename(devPointerFile(f.prof, f.co)), POINTER_FILE);

  // Release first run writes data-location.json; dev keeps its checkout paths.
  const rel = await resolveRuntime({ env: { HOT_STEP_ROOT: f.app, HOT_STEP_PROFILE_DIR: f.prof }, log: quiet });
  assert.ok(rel.ok && rel.runtime.dataRoot === f.sibling);
  const dev0 = await resolveRuntime({ env: f.env, log: quiet, checkoutRoot: f.co });
  assert.ok(dev0.ok && dev0.runtime.dataRoot === null);

  // A dev pointer gives dev its own folder; the release pointer is unchanged.
  const devRoot = path.join(f.base, 'dev-data');
  initLayout(dataLayout(devRoot));
  writePointer(f.ptrFile, { version: 1, root: devRoot, pending: null, checkout: f.co });
  const dev1 = await resolveRuntime({ env: f.env, log: quiet, checkoutRoot: f.co });
  assert.ok(dev1.ok, !dev1.ok ? dev1.error : '');
  assert.equal(dev1.runtime.mode, 'dev');
  assert.equal(dev1.runtime.dataRoot, devRoot);
  assert.equal(dev1.runtime.restartMarkerPath, null, 'dev keeps the checkout restart marker');
  assert.equal(dev1.runtime.runtimeLibDir, null, 'dev runtime DLLs stay beside the dev engine');
  assert.equal(readPointer(f.ptrFile)!.checkout, f.co);
  assert.equal(readPointer(path.join(f.prof, POINTER_FILE))!.root, f.sibling);
  const rel2 = await resolveRuntime({ env: { HOT_STEP_ROOT: f.app, HOT_STEP_PROFILE_DIR: f.prof }, log: quiet });
  assert.ok(rel2.ok && rel2.runtime.dataRoot === f.sibling);

  // Each sees the other's folder as taken (Settings refuses those targets).
  assert.deepEqual(otherPointerRoots(f.prof, f.ptrFile), [f.sibling]);
  assert.deepEqual(otherPointerRoots(f.prof, path.join(f.prof, POINTER_FILE)), [devRoot]);
});

test('a release first run refuses a sibling folder a dev checkout already uses', async () => {
  const f = devCheckout();
  initLayout(dataLayout(f.sibling));
  writePointer(f.ptrFile, { version: 1, root: f.sibling, pending: null, checkout: f.co });
  const r = await resolveRuntime({ env: { HOT_STEP_ROOT: f.app, HOT_STEP_PROFILE_DIR: f.prof }, log: quiet });
  assert.ok(!r.ok && /development checkout/.test(r.error));
  assert.ok(!fs.existsSync(path.join(f.prof, POINTER_FILE)));
});

test('dev Move maps the checkout paths (DATA_DIR against server/), leaves tracked files, and only runs once scheduled', async () => {
  const f = devCheckout();
  // The root .env.example value resolves against server/, i.e. server/server/data.
  const lib = path.join(f.co, 'server', 'server', 'data');
  makeDb(path.join(lib, 'hotstep.db'), [{ adapter: path.join(f.co, 'adapters', 'a.safetensors'), params: { ref: path.join(lib, 'references', 'r.flac') } }]);
  write(path.join(lib, 'references', 'r.flac'));
  write(path.join(f.co, 'models', 'm.gguf'));
  write(path.join(f.co, 'adapters', 'a.safetensors'));
  write(path.join(f.co, 'logs', 'session.log'));
  write(path.join(f.co, 'tools', 'whisper', 'whisper-cli.exe'));
  const exe = path.join(f.co, 'engine', 'build', 'Debug', 'ace-server.exe');
  write(path.join(f.co, '.env'), ['DATA_DIR=./server/data', 'GEMINI_API_KEY=secret123', `ACESTEPCPP_EXE=${exe}`].join('\n'));

  // Booting dev with no pointer moves nothing.
  const before = await resolveRuntime({ env: f.env, log: quiet, checkoutRoot: f.co });
  assert.ok(before.ok && before.runtime.dataRoot === null);
  assert.ok(fs.existsSync(path.join(lib, 'hotstep.db')));

  const dest = path.join(f.base, 'Dev-Data');
  const { op, errors } = planOperation('migrate', dest, { appRoot: f.co, currentRoot: null, currentDb: null, dev: true });
  assert.deepEqual(errors, []);
  assert.deepEqual(Object.fromEntries(op.mappings.map(m => [m.name, m.from])), {
    library: lib, models: path.join(f.co, 'models'), adapters: path.join(f.co, 'adapters'),
    logs: path.join(f.co, 'logs'), whisper: path.join(f.co, 'tools', 'whisper'),
  });
  assert.ok(op.mappings.every(m => !m.exclude), 'server/data in dev is the library, not bundled resources');

  // What Settings does: write the dev pointer, then restart.
  writePointer(f.ptrFile, { version: 1, root: null, pending: op, checkout: f.co });
  const r = await resolveRuntime({ env: f.env, log: quiet, checkoutRoot: f.co, skipEngineProbe: true });
  assert.ok(r.ok, !r.ok ? r.error : '');
  const l = dataLayout(dest);
  assert.equal(r.runtime.mode, 'dev');
  assert.equal(r.runtime.dataRoot, dest);
  assert.equal(r.runtime.envPath, l.envPath);
  for (const p of [path.join(l.library, 'hotstep.db'), path.join(l.library, 'references', 'r.flac'), path.join(l.models, 'm.gguf'),
    path.join(l.adapters, 'a.safetensors'), path.join(l.logs, 'session.log'), path.join(l.whisperDir, 'whisper-cli.exe')]) {
    assert.ok(fs.existsSync(p), p);
  }
  // Tracked files and bundled resources stay in the checkout.
  assert.ok(fs.existsSync(path.join(f.co, 'server', 'src', 'data', 'model-registry.json')));
  assert.ok(fs.existsSync(path.join(f.co, 'tools', 'cg.cmd')));
  assert.ok(fs.existsSync(path.join(f.co, '.env.example')));
  // Settings follow; the checkout's own engine path stays valid, so it is kept.
  const env = fs.readFileSync(l.envPath, 'utf-8');
  assert.match(env, /GEMINI_API_KEY=secret123/);
  assert.doesNotMatch(env, /DATA_DIR=/);
  assert.ok(env.includes(`ACESTEPCPP_EXE=${exe}`));
  assert.ok(fs.existsSync(path.join(f.co, '.env.migrated')) && !fs.existsSync(path.join(f.co, '.env')));
  const conn = new Database(path.join(l.library, 'hotstep.db'), { readonly: true });
  assert.equal((conn.prepare('SELECT adapter_path FROM album_presets').get() as any).adapter_path, path.join(l.adapters, 'a.safetensors'));
  assert.equal(JSON.parse((conn.prepare('SELECT generation_params FROM songs').get() as any).generation_params).ref, path.join(l.library, 'references', 'r.flac'));
  conn.close();
  assert.deepEqual(readPointer(f.ptrFile), { version: 1, root: dest, pending: null, checkout: f.co });
});

test('dev Move with no .env uses server/data and refuses tracked sources or a populated target', () => {
  const f = devCheckout();
  write(path.join(f.co, 'server', 'data', 'hotstep.db'), 'db');
  const ctx = { appRoot: f.co, currentRoot: null, currentDb: null, dev: true };
  const ok = planOperation('migrate', path.join(f.base, 'D1'), ctx);
  assert.deepEqual(ok.errors, []);
  assert.deepEqual(ok.op.mappings.map(m => m.from), [path.join(f.co, 'server', 'data')]);

  write(path.join(f.base, 'D2', 'something.txt'));
  assert.ok(planOperation('migrate', path.join(f.base, 'D2'), ctx).errors.some(e => /not empty/.test(e)));
  assert.ok(planOperation('migrate', path.join(f.co, 'inside'), ctx).errors.some(e => /inside the app folder/.test(e)));

  // DATA_DIR=. would carry server/ itself, tracked sources included.
  write(path.join(f.co, '.env'), 'DATA_DIR=.\n');
  const bad = planOperation('migrate', path.join(f.base, 'D3'), ctx);
  assert.ok(bad.errors.some(e => /git tracks/.test(e) && e.includes('model-registry.json')), bad.errors.join('; '));
});

test('dev pointer with no root and nothing pending (an abandoned first move) keeps the checkout paths', async () => {
  const f = devCheckout();
  writePointer(f.ptrFile, { version: 1, root: null, pending: null, checkout: f.co });
  const r = await resolveRuntime({ env: f.env, log: quiet, checkoutRoot: f.co });
  assert.ok(r.ok && r.runtime.dataRoot === null);
});

test('config in dev with a dev pointer uses the data folder but keeps the checkout and its restart marker', () => {
  const f = fixture();
  const checkout = path.dirname(SERVER_DIR);
  const root = path.join(f.base, 'dev-data');
  initLayout(dataLayout(root));
  writePointer(devPointerFile(f.prof, checkout), { version: 1, root, pending: null, checkout });
  const c = loadConfig({ HOT_STEP_PROFILE_DIR: f.prof });
  const l = dataLayout(root);
  assert.equal(c.portable, false);
  assert.equal(c.root, checkout);
  assert.equal(c.data, l.library);
  assert.equal(c.models, l.models);
  assert.equal(c.adapters, l.adapters);
  assert.equal(c.env, l.envPath);
  assert.equal(c.logs, l.logs);
  assert.equal(c.marker, path.join(checkout, '.restart-requested'));
  assert.doesNotMatch(fs.readFileSync(l.envPath, 'utf-8'), /^DATA_DIR=/m, 'example DATA_DIR dropped');
});

test('dev Move refuses a source that is a junction onto tracked files, at plan time and again before moving', () => {
  const f = devCheckout();
  const tracked = path.join(f.co, 'server', 'src', 'data', 'model-registry.json');
  fs.mkdirSync(path.join(f.co, 'server'), { recursive: true });
  fs.symlinkSync(path.join(f.co, 'server', 'src', 'data'), path.join(f.co, 'server', 'data'), 'junction');
  const ctx = { appRoot: f.co, currentRoot: null, currentDb: null, dev: true };
  const bad = planOperation('migrate', path.join(f.base, 'D1'), ctx);
  assert.ok(bad.errors.some(e => /git tracks/.test(e) && e.includes('model-registry.json')), bad.errors.join('; '));

  // Planned against a real folder, swapped for the junction before restart.
  fs.unlinkSync(path.join(f.co, 'server', 'data')); // the junction, not its target
  write(path.join(f.co, 'server', 'data', 'hotstep.db'), 'db');
  const { op, errors } = planOperation('migrate', path.join(f.base, 'D2'), ctx);
  assert.deepEqual(errors, []);
  fs.rmSync(path.join(f.co, 'server', 'data'), { recursive: true });
  fs.symlinkSync(path.join(f.co, 'server', 'src', 'data'), path.join(f.co, 'server', 'data'), 'junction');
  const ptr = { version: 1 as const, root: null, pending: op, checkout: f.co };
  writePointer(f.ptrFile, ptr);
  assert.throws(() => runPending(f.ptrFile, ptr, f.co, quiet), /git tracks/);
  assert.ok(fs.existsSync(tracked), 'tracked resource untouched');
  assert.ok(isEmptyDirOrMissing(path.join(f.base, 'D2', 'library')));
});

test('dev Move refuses a tracked .env', () => {
  const f = devCheckout();
  write(path.join(f.co, '.env'), 'GEMINI_API_KEY=x\n');
  execFileSync('git', ['add', '-f', '.env'], { cwd: f.co });
  const r = planOperation('migrate', path.join(f.base, 'D1'), { appRoot: f.co, currentRoot: null, currentDb: null, dev: true });
  assert.ok(r.errors.some(e => /git tracks/.test(e) && e.includes('.env')), r.errors.join('; '));
});

test('a junction into another install\'s data folder counts as that folder, in Settings and at activation', async () => {
  const f = devCheckout();
  const rel = await resolveRuntime({ env: { HOT_STEP_ROOT: f.app, HOT_STEP_PROFILE_DIR: f.prof }, log: quiet });
  assert.ok(rel.ok);
  const link = path.join(f.base, 'looks-separate');
  fs.symlinkSync(f.sibling, link, 'junction');
  assert.equal(sharedRoot(f.prof, f.ptrFile, link), f.sibling);
  assert.equal(sharedRoot(f.prof, f.ptrFile, path.join(f.base, 'really-separate')), null);

  // Scheduled anyway (pointer edited by hand): refused before running.
  writePointer(f.ptrFile, { version: 1, root: null, pending: planOperation('repoint', link, { appRoot: f.co, currentRoot: null, currentDb: null, dev: true }).op, checkout: f.co });
  const pend = await resolveRuntime({ env: f.env, log: quiet, checkoutRoot: f.co, skipEngineProbe: true });
  assert.ok(!pend.ok && /another HOT-Step install/.test(pend.error));
  // Already active: refused at start-up.
  writePointer(f.ptrFile, { version: 1, root: link, pending: null, checkout: f.co });
  const act = await resolveRuntime({ env: f.env, log: quiet, checkoutRoot: f.co });
  assert.ok(!act.ok && /another HOT-Step install/.test(act.error));
});

function isEmptyDirOrMissing(dir: string): boolean {
  try { return fs.readdirSync(dir).length === 0; } catch { return true; }
}

// ── locate.ts: read-only report ──────────────────────────────────────

function snapshot(dir: string): string[] {
  return fs.readdirSync(dir, { recursive: true, withFileTypes: true }).map(e => `${path.join(e.parentPath, e.name)}:${e.isFile() ? fs.statSync(path.join(e.parentPath, e.name)).mtimeMs : 'd'}`).sort();
}

test('locateData: dev with no pointer reports checkout paths and touches nothing', () => {
  const f = fixture();
  const checkout = path.join(f.base, 'co');
  fs.mkdirSync(path.join(checkout, 'server'), { recursive: true });
  const before = snapshot(f.base);
  const r = locateData({ env: { HOT_STEP_PROFILE_DIR: f.prof } as NodeJS.ProcessEnv, checkoutRoot: checkout });
  assert.equal(r.mode, 'dev'); assert.equal(r.root, null); assert.equal(r.source, 'checkout');
  assert.equal(r.children!.library, path.join(checkout, 'server', 'data'));
  assert.equal(r.children!.models, path.join(checkout, 'models'));
  assert.deepEqual(snapshot(f.base), before);
});

test('locateData: dev pointer, release pointer, HOT_STEP_DATA_ROOT and .env overrides', () => {
  const f = fixture();
  const checkout = path.join(f.base, 'co');
  const dataDev = path.join(f.base, 'dev-data');
  initLayout(dataLayout(dataDev));
  writePointer(devPointerFile(f.prof, checkout), { version: 1, root: dataDev, pending: null });
  const dev = locateData({ env: { HOT_STEP_PROFILE_DIR: f.prof } as NodeJS.ProcessEnv, checkoutRoot: checkout });
  assert.equal(dev.root, dataDev); assert.equal(dev.source, 'pointer');
  assert.equal(dev.children!.db, path.join(dataDev, 'library', 'hotstep.db'));
  assert.equal(dev.children!.plugins, path.join(dataDev, 'plugins'));
  assert.deepEqual(dev.overrides, []);

  fs.writeFileSync(dataLayout(dataDev).envPath, 'ACESTEPCPP_MODELS=D:\\elsewhere\\m\n');
  const withEnv = locateData({ env: { HOT_STEP_PROFILE_DIR: f.prof } as NodeJS.ProcessEnv, checkoutRoot: checkout });
  assert.deepEqual(withEnv.overrides, ['ACESTEPCPP_MODELS']);
  const viaProcess = locateData({ env: { HOT_STEP_PROFILE_DIR: f.prof, TRAINING_DIR: path.join(f.base, 'tr') } as NodeJS.ProcessEnv, checkoutRoot: checkout });
  assert.equal(viaProcess.children!.training, path.join(f.base, 'tr'));

  initLayout(dataLayout(path.join(f.base, 'rel')));
  writePointer(f.ptrFile, { version: 1, root: path.join(f.base, 'rel'), pending: null });
  assert.equal(locateData({ env: f.env }).root, path.join(f.base, 'rel'));
  const o = locateData({ env: { ...f.env, HOT_STEP_DATA_ROOT: path.join(f.base, 'ov') } });
  assert.equal(o.root, path.join(f.base, 'ov')); assert.equal(o.source, 'HOT_STEP_DATA_ROOT');
});

test('locateData: reports a pending move and a bad pointer without acting on them', () => {
  const f = fixture();
  const checkout = path.join(f.base, 'co');
  const env = { HOT_STEP_PROFILE_DIR: f.prof } as NodeJS.ProcessEnv;
  const file = devPointerFile(f.prof, checkout);
  const src = path.join(f.base, 'a'); const dest = path.join(f.base, 'b');
  initLayout(dataLayout(src));
  writePointer(file, { version: 1, root: src, pending: newPending('move', src, dest) });
  const before = fs.readFileSync(file, 'utf-8');
  const r = locateData({ env, checkoutRoot: checkout });
  assert.equal(r.pendingMove!.kind, 'move'); assert.equal(r.pendingMove!.destination, dest);
  assert.equal(fs.readFileSync(file, 'utf-8'), before); assert.equal(fs.existsSync(dest), false);
  fs.writeFileSync(file, '{nope');
  const bad = locateData({ env, checkoutRoot: checkout });
  assert.match(bad.pointerError!, /not valid JSON/); assert.equal(bad.root, null);
});

for (const mode of ['release', 'dev'] as const) {
  for (const state of ['missing', 'inaccessible', 'valid', 'override'] as const) {
    test(`locateData and boot agree: ${mode} ${state} saved root`, async t => {
      const f = fixture();
      const saved = path.join(f.base, 'saved');
      const override = path.join(f.base, 'override');
      const env: NodeJS.ProcessEnv = { HOT_STEP_PROFILE_DIR: f.prof };
      if (mode === 'release') env.HOT_STEP_ROOT = f.app;
      if (state === 'override') {
        env.HOT_STEP_DATA_ROOT = override;
        initLayout(dataLayout(override));
      }
      if (state === 'valid' || state === 'inaccessible') initLayout(dataLayout(saved));
      const file = mode === 'release' ? f.ptrFile : devPointerFile(f.prof, f.app);
      writePointer(file, { version: 1, root: saved, pending: null });
      const pointerBefore = fs.readFileSync(file, 'utf-8');
      if (state === 'inaccessible') {
        const access = fs.accessSync;
        t.mock.method(fs, 'accessSync', (p: fs.PathLike, flags?: number) => {
          if (p === saved) throw Object.assign(new Error('access denied'), { code: 'EACCES' });
          return access(p, flags);
        });
      }
      const before = snapshot(f.base);
      const report = locateData({ env, checkoutRoot: f.app });
      const expectedRoot = mode === 'release' && state === 'override' ? override
        : state === 'valid' ? saved : mode === 'release' ? f.sibling : null;
      assert.equal(report.root, expectedRoot);
      assert.deepEqual(snapshot(f.base), before, 'locator never writes, even when fallback does not exist');
      const boot = await resolveRuntime({ env, checkoutRoot: f.app, log: quiet });
      assert.ok(boot.ok);
      assert.equal(report.root, boot.runtime.dataRoot);
      const library = boot.runtime.libraryDir ?? path.join(f.app, 'server', 'data');
      assert.equal(report.children!.library, library);
      assert.equal(report.children!.db, path.join(library, 'hotstep.db'));
      assert.equal(report.children!.envFile, boot.runtime.envPath ?? path.join(f.app, '.env'));
      assert.equal(fs.readFileSync(file, 'utf-8'), pointerBefore);
      assert.equal(locateData({ env, checkoutRoot: f.app }).root, boot.runtime.dataRoot);
    });
  }
}

test('locateData: an empty process variable beats the .env file, like dotenv', () => {
  const f = fixture();
  const checkout = path.join(f.base, 'co');
  const root = path.join(f.base, 'data');
  initLayout(dataLayout(root));
  writePointer(devPointerFile(f.prof, checkout), { version: 1, root, pending: null });
  fs.writeFileSync(dataLayout(root).envPath, `ACESTEPCPP_ADAPTERS=${path.join(f.base, 'elsewhere')}
`);
  const r = locateData({ env: { HOT_STEP_PROFILE_DIR: f.prof, ACESTEPCPP_ADAPTERS: '' } as NodeJS.ProcessEnv, checkoutRoot: checkout });
  assert.equal(r.children!.adapters, dataLayout(root).adapters);
  assert.deepEqual(r.overrides, []);
});

// ── Upgrade fixtures (GH #228): one profile, several unpacked releases ──

/** A second unpacked release folder (spaces in the name) sharing the fixture's profile. */
function secondRelease(f: ReturnType<typeof fixture>, name = 'HOT-Step v3 (new)') {
  // A different parent folder, so its sibling default differs from the first release's.
  const app = path.join(f.base, 'second parent', name);
  fs.mkdirSync(path.join(app, 'server', 'data'), { recursive: true });
  for (const n of BUNDLED_DATA_NAMES) fs.writeFileSync(path.join(app, 'server', 'data', n), 'bundled');
  return { app, env: { ...f.env, HOT_STEP_ROOT: app } as NodeJS.ProcessEnv };
}

test('two unpacked releases sharing one profile use one data folder, with its settings and tokens', async () => {
  const f = fixture();
  const first = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
  assert.ok(first.ok);
  const layout = dataLayout(f.sibling);
  write(layout.envPath, 'GEMINI_API_KEY=abc\n');
  write(path.join(layout.library, 'muscriptor', 'hf_token'), 'hf_secret');

  const v3 = secondRelease(f);
  const logs: string[] = [];
  const second = await resolveRuntime({ env: v3.env, log: m => logs.push(m), skipEngineProbe: true });
  assert.ok(second.ok, !second.ok ? second.error : '');
  // The pointer, not the new app folder's own sibling, decides.
  assert.equal(second.runtime.dataRoot, f.sibling);
  assert.equal(second.runtime.appRoot, v3.app);
  assert.equal(second.runtime.envPath, layout.envPath);
  assert.equal(fs.readFileSync(second.runtime.envPath!, 'utf-8'), 'GEMINI_API_KEY=abc\n');
  assert.equal(fs.readFileSync(path.join(second.runtime.libraryDir!, 'muscriptor', 'hf_token'), 'utf-8'), 'hf_secret');
  assert.equal(second.runtime.restartMarkerPath, path.join(f.prof, 'restart-requested'));
  assert.deepEqual(logs, []);
  assert.equal(readPointer(f.ptrFile)!.root, f.sibling);
});

test('a release with a saved pointer ignores its own sibling default and creates nothing', async () => {
  const f = fixture();
  const elsewhere = path.join(f.base, 'My Music Drive', 'HOT-Step Data');
  initLayout(dataLayout(elsewhere));
  writePointer(f.ptrFile, { version: 1, root: elsewhere, pending: null });
  const v3 = secondRelease(f);
  const r = await resolveRuntime({ env: v3.env, log: quiet, skipEngineProbe: true });
  assert.ok(r.ok);
  assert.equal(r.runtime.dataRoot, elsewhere);
  assert.ok(!fs.existsSync(f.sibling));
  assert.ok(!fs.existsSync(path.join(f.base, 'second parent', 'HOT-Step-Data')));
});

test('HOT_STEP_DATA_ROOT beats a healthy pointer without rewriting it; unset, the pointer wins again', async () => {
  const f = fixture();
  const saved = path.join(f.base, 'saved');
  const override = path.join(f.base, 'with space', 'override');
  initLayout(dataLayout(saved));
  initLayout(dataLayout(override));
  writePointer(f.ptrFile, { version: 1, root: saved, pending: null });
  const before = fs.readFileSync(f.ptrFile, 'utf-8');
  const over = await resolveRuntime({ env: { ...f.env, HOT_STEP_DATA_ROOT: override }, log: quiet, skipEngineProbe: true });
  assert.ok(over.ok);
  assert.equal(over.runtime.dataRoot, override);
  assert.equal(over.runtime.override, override);
  assert.equal(over.runtime.envPath, dataLayout(override).envPath);
  assert.equal(fs.readFileSync(f.ptrFile, 'utf-8'), before);
  const plain = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
  assert.ok(plain.ok);
  assert.equal(plain.runtime.dataRoot, saved);
  assert.equal(plain.runtime.override, null);
});

test('restart after a move: later starts use the new folder, with settings and token, and change nothing', async () => {
  const f = fixture();
  const first = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
  assert.ok(first.ok);
  const src = dataLayout(f.sibling);
  write(src.envPath, 'OPENAI_API_KEY=k\n');
  write(path.join(src.library, 'muscriptor', 'hf_token'), 'hf_secret');
  const dest = path.join(f.base, 'Moved Data');
  const { op, errors } = planOperation('move', dest, { appRoot: f.app, currentRoot: f.sibling, currentDb: null });
  assert.deepEqual(errors, []);
  writePointer(f.ptrFile, { version: 1, root: f.sibling, pending: op });
  const moved = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
  assert.ok(moved.ok, !moved.ok ? moved.error : '');
  assert.equal(moved.runtime.dataRoot, dest);
  const ptr = fs.readFileSync(f.ptrFile, 'utf-8');
  for (let i = 0; i < 2; i++) {
    const again = await resolveRuntime({ env: f.env, log: quiet, skipEngineProbe: true });
    assert.ok(again.ok);
    assert.equal(again.runtime.dataRoot, dest);
    assert.equal(fs.readFileSync(dataLayout(dest).envPath, 'utf-8'), 'OPENAI_API_KEY=k\n');
    assert.equal(fs.readFileSync(path.join(dataLayout(dest).library, 'muscriptor', 'hf_token'), 'utf-8'), 'hf_secret');
  }
  assert.equal(fs.readFileSync(f.ptrFile, 'utf-8'), ptr);
});

test('stale drive: a second app folder falls back, then returns to the saved drive once it is back', async () => {
  const f = fixture();
  const drive = path.join(f.base, 'E drive', 'HOT-Step Data');
  writePointer(f.ptrFile, { version: 1, root: drive, pending: null });
  const v3 = secondRelease(f);
  const logs: string[] = [];
  const away = await resolveRuntime({ env: v3.env, log: m => logs.push(m), skipEngineProbe: true });
  assert.ok(away.ok);
  assert.equal(away.runtime.dataRoot, path.join(f.base, 'second parent', 'HOT-Step-Data'));
  assert.equal(logs.length, 1);
  assert.ok(logs[0].includes(drive));
  assert.equal(readPointer(f.ptrFile)!.root, drive);
  initLayout(dataLayout(drive));
  logs.length = 0;
  const back = await resolveRuntime({ env: v3.env, log: m => logs.push(m), skipEngineProbe: true });
  assert.ok(back.ok);
  assert.equal(back.runtime.dataRoot, drive);
  assert.deepEqual(logs, []);
});

test('profile folder per platform, including macOS and Windows paths with spaces', () => {
  const mac = '/Users/Jo Bloggs';
  assert.equal(profileDir({}, 'darwin', mac), path.join(mac, 'Library', 'Application Support', 'HOT-Step'));
  assert.equal(profileDir({ XDG_CONFIG_HOME: '/cfg dir' }, 'linux', mac), path.join('/cfg dir', 'hot-step'));
  assert.equal(profileDir({}, 'linux', mac), path.join(mac, '.config', 'hot-step'));
  const local = path.join('C:', 'Users', 'Jo Bloggs', 'AppData', 'Local');
  assert.equal(profileDir({ LOCALAPPDATA: local }, 'win32', 'C:'), path.join(local, 'HOT-Step'));
  assert.equal(profileDir({}, 'win32', path.join('C:', 'Users', 'Jo Bloggs')), path.join('C:', 'Users', 'Jo Bloggs', 'AppData', 'Local', 'HOT-Step'));
  // The fixture override beats every platform default.
  assert.equal(profileDir({ HOT_STEP_PROFILE_DIR: path.resolve('p q') }, 'darwin', mac), path.resolve('p q'));
});

// ── Versioned layout: the stable launcher sets HOT_STEP_ROOT to app/<id> and HOT_STEP_INSTALL_ROOT to the unzipped folder ──

/** What a protocol-1 package puts in `install` (unzipped over an old one, or into an empty folder). */
function versioned(f: ReturnType<typeof fixture>, install = f.app) {
  const payload = path.join(install, 'app', 'v3.0.0-0123456789ab-cpu');
  for (const n of BUNDLED_DATA_NAMES) write(path.join(payload, 'server', 'data', n), 'bundled');
  write(path.join(payload, 'engine', 'ace-server.exe'), 'new');
  write(path.join(payload, '.env.example'), 'DATA_DIR=./server/data\nPP_STEM_CACHE_GB=4\n');
  for (const n of ['HOT-Step.bat', 'launcher.mjs', 'README.txt', path.join('runtime', 'node.exe'), path.join('updater', 'last-result.json')]) write(path.join(install, n));
  write(path.join(install, 'current.json'), JSON.stringify({ format: 1, payload: path.basename(payload) }));
  return { install, payload, env: { HOT_STEP_ROOT: payload, HOT_STEP_INSTALL_ROOT: install, HOT_STEP_PROFILE_DIR: f.prof } as NodeJS.ProcessEnv };
}

function settingsPlans(env: NodeJS.ProcessEnv, targets: string[], kind = 'move'): any {
  const script = `const m = await import(${JSON.stringify(new URL('./pointer.ts', import.meta.url).href)});
const targets = ${JSON.stringify(targets)};
const previews = targets.map(t => m.previewOperation(${JSON.stringify(kind)}, t, null));
const refusals = targets.map((t, i) => {
  if (previews[i].ok) return null;
  try { m.scheduleOperation(${JSON.stringify(kind)}, t, null); return 'accepted'; }
  catch (e) { return { status: e.status, message: e.message }; }
});
console.log('@@' + JSON.stringify({ previews, refusals, preflight: m.updatePreflight() }));`;
  const clean = { ...process.env };
  for (const k of ['HOT_STEP_ROOT', 'HOT_STEP_INSTALL_ROOT', 'HOT_STEP_DATA_ROOT', 'HOT_STEP_PROFILE_DIR', ...Object.keys(clean).filter(k => /^(DATA_DIR|ACESTEPCPP_|TRAINING_DIR|TENSORRT_LIBS)/.test(k))]) delete clean[k];
  const out = execFileSync(process.execPath, ['--import', 'tsx', '--input-type=module', '-e', script], {
    env: { ...clean, ...env }, cwd: SERVER_DIR, encoding: 'utf-8', timeout: 60_000,
  });
  return JSON.parse(out.split('\n').find(l => l.startsWith('@@'))!.slice(2));
}

test('Settings Move refuses destinations anywhere inside a versioned install', () => {
  const f = fixture();
  const v = versioned(f);
  initLayout(dataLayout(f.sibling));
  writePointer(f.ptrFile, { version: 1, root: f.sibling, pending: null });
  const alias = path.join(f.base, 'install-link');
  fs.symlinkSync(v.install, alias, process.platform === 'win32' ? 'junction' : 'dir');
  const targets = [v.install, path.join(v.install, 'my-data'), path.join(v.payload, 'my-data'), path.join(alias, 'my-data'), path.join(f.base, 'outside')];
  const r = settingsPlans(v.env, targets);
  for (let i = 0; i < targets.length - 1; i++) {
    assert.equal(r.previews[i].ok, false, targets[i]);
    assert.ok(r.previews[i].errors.includes(`the data folder cannot be inside the app folder (${v.install})`));
    assert.equal(r.refusals[i].status, 400);
  }
  assert.equal(r.previews.at(-1).ok, true);
  assert.equal(readPointer(f.ptrFile)!.pending, null);
});

test('Settings Move keeps the flat release and checkout destination boundaries', () => {
  for (const dev of [false, true]) {
    const f = fixture();
    initLayout(dataLayout(f.sibling));
    const app = dev ? path.dirname(SERVER_DIR) : f.app;
    const ptrFile = dev ? devPointerFile(f.prof, app) : f.ptrFile;
    writePointer(ptrFile, { version: 1, root: f.sibling, pending: null, ...(dev ? { checkout: app } : {}) });
    const env = dev ? { HOT_STEP_PROFILE_DIR: f.prof } : f.env;
    const r = settingsPlans(env, [path.join(app, 'my-data'), path.join(f.base, 'outside')]);
    assert.equal(r.previews[0].ok, false);
    assert.ok(r.previews[0].errors.includes(`the data folder cannot be inside the app folder (${app})`));
    assert.equal(r.refusals[0].status, 400);
    assert.equal(r.previews[1].ok, true);
  }
});

test('versioned Settings still imports from the install folder with a saved location', () => {
  const f = fixture();
  oldInAppData(f);
  const v = versioned(f);
  initLayout(dataLayout(f.sibling));
  writePointer(f.ptrFile, { version: 1, root: f.sibling, pending: null });
  assert.equal(settingsPlans(v.env, [v.install], 'import').previews[0].ok, true);
});

test('versioned update preflight allows an existing data folder inside the install', () => {
  const f = fixture();
  const v = versioned(f);
  const root = path.join(v.install, 'my-data');
  initLayout(dataLayout(root));
  writePointer(f.ptrFile, { version: 1, root, pending: null });
  assert.deepEqual(settingsPlans(v.env, []).preflight, []);
});

/** f.app as a release from before data folders: library in server/data, models and settings in the app folder. */
function oldInAppData(f: ReturnType<typeof fixture>) {
  const sd = path.join(f.app, 'server', 'data');
  makeDb(path.join(sd, 'hotstep.db'), [{ params: { p: path.join(sd, 'audio', 'a.wav') } }]);
  write(path.join(sd, 'audio', 'a.wav'), 'wav');
  write(path.join(f.app, 'models', 'm.gguf'), 'gguf');
  write(path.join(f.app, 'engine', 'ace-server.exe'), 'old flat engine');
  write(path.join(f.app, '.env'), ['DATA_DIR=./server/data', 'PP_STEM_CACHE_GB=7', 'ACESTEPCPP_EXE=./engine/ace-server.exe'].join('\n'));
  return sd;
}

test('versioned package unzipped over an old install: its in-app data moves beside the install, and the new engine runs', async () => {
  const f = fixture();
  const sd = oldInAppData(f);
  const v = versioned(f);
  const r = await resolveRuntime({ env: v.env, log: quiet, skipEngineProbe: true });
  assert.ok(r.ok, !r.ok ? r.error : '');
  assert.equal(r.runtime.dataRoot, f.sibling);
  assert.equal(r.runtime.appRoot, v.payload);
  assert.deepEqual(readPointer(f.ptrFile), { version: 1, root: f.sibling, pending: null });
  const l = dataLayout(f.sibling);
  assert.equal(fs.readFileSync(path.join(l.library, 'audio', 'a.wav'), 'utf-8'), 'wav');
  assert.ok(fs.existsSync(path.join(l.models, 'm.gguf')));
  assert.deepEqual(fs.readdirSync(sd).sort(), [...BUNDLED_DATA_NAMES].sort());
  const env = fs.readFileSync(l.envPath, 'utf-8');
  assert.match(env, /^PP_STEM_CACHE_GB=7$/m);
  assert.doesNotMatch(env, /ACESTEPCPP_EXE|DATA_DIR/);
  assert.ok(fs.existsSync(path.join(f.app, '.env.migrated')) && !fs.existsSync(path.join(f.app, '.env')));
  assert.ok(!fs.existsSync(path.join(f.app, 'app', 'HOT-Step-Data')));
  const conn = new Database(path.join(l.library, 'hotstep.db'), { readonly: true });
  assert.equal(JSON.parse((conn.prepare('SELECT generation_params FROM songs').get() as any).generation_params).p, path.join(l.library, 'audio', 'a.wav'));
  conn.close();
  // The flat engine still in the install folder is not the one config starts.
  const c = loadConfig(v.env);
  assert.equal(c.root, v.payload);
  assert.equal(c.exe, path.join(v.payload, 'engine', 'ace-server.exe'));
  assert.equal(c.data, l.library);
});

test('versioned package in a new folder: its own root is not old data, and the first start only records the default', async () => {
  const f = fixture();
  const v = versioned(f, path.join(f.base, 'HOT-Step'));
  const before = snapshot(v.install);
  const r = await resolveRuntime({ env: v.env, log: quiet, skipEngineProbe: true });
  assert.ok(r.ok, !r.ok ? r.error : '');
  assert.equal(r.runtime.dataRoot, f.sibling);
  assert.deepEqual(readPointer(f.ptrFile), { version: 1, root: f.sibling, pending: null });
  assert.deepEqual(snapshot(v.install), before);
  assert.deepEqual(fs.readdirSync(dataLayout(f.sibling).library), []);
});

test('versioned layout with a saved pointer: old data in the install folder stays put, and Import can still take it', async () => {
  const f = fixture();
  const sd = oldInAppData(f);
  const elsewhere = path.join(f.base, 'Data Drive', 'HOT-Step-Data');
  initLayout(dataLayout(elsewhere));
  writePointer(f.ptrFile, { version: 1, root: elsewhere, pending: null });
  const v = versioned(f);
  const r = await resolveRuntime({ env: v.env, log: quiet, skipEngineProbe: true });
  assert.ok(r.ok, !r.ok ? r.error : '');
  assert.equal(r.runtime.dataRoot, elsewhere);
  assert.ok(fs.existsSync(path.join(sd, 'hotstep.db')) && fs.existsSync(path.join(f.app, '.env')));
  assert.ok(!fs.existsSync(f.sibling));
  // Settings plans against the payload, so the install folder counts as an old install, not "this app folder".
  const { errors } = planOperation('import', v.install, { appRoot: v.payload, currentRoot: elsewhere, currentDb: null });
  assert.deepEqual(errors, []);
});

test('versioned layout: a populated default folder stops the migration, and one that stops half-way resumes on the next start', async () => {
  const f = fixture();
  oldInAppData(f);
  const v = versioned(f);
  write(path.join(f.sibling, 'library', 'hotstep.db'));
  const refused = await resolveRuntime({ env: v.env, log: quiet, skipEngineProbe: true });
  assert.equal(refused.ok, false);
  assert.match((refused as any).error, /already exists with other data/);
  assert.ok(!fs.existsSync(f.ptrFile) && fs.existsSync(path.join(f.app, '.env')));
  fs.rmSync(f.sibling, { recursive: true });

  // A DB path that collides once rewritten stops the run after the files moved; fixing it lets the next start finish.
  const lib = dataLayout(f.sibling).library;
  const db = new Database(path.join(f.app, 'server', 'data', 'hotstep.db'));
  const ins = db.prepare('INSERT INTO training_datasets (source_dir) VALUES (?)');
  ins.run(path.join(f.app, 'server', 'data', 'x'));
  ins.run(path.join(lib, 'x'));
  db.close();
  const first = await resolveRuntime({ env: v.env, log: quiet, skipEngineProbe: true });
  assert.equal(first.ok, false);
  assert.match((first as any).error, /collides/);
  const pending = readPointer(f.ptrFile)!.pending!;
  assert.equal(pending.kind, 'migrate');
  assert.equal(pending.source, f.app);
  const conn = new Database(path.join(lib, 'hotstep.db'));
  conn.prepare('DELETE FROM training_datasets WHERE source_dir = ?').run(path.join(lib, 'x'));
  conn.close();
  const second = await resolveRuntime({ env: v.env, log: quiet, skipEngineProbe: true });
  assert.ok(second.ok, !second.ok ? second.error : '');
  assert.deepEqual(readPointer(f.ptrFile), { version: 1, root: f.sibling, pending: null });
  assert.match(fs.readFileSync(dataLayout(f.sibling).envPath, 'utf-8'), /^PP_STEM_CACHE_GB=7$/m);
});

test('versioned layout: a saved data folder that went missing falls back beside the install, never inside app/', async () => {
  const f = fixture();
  writePointer(f.ptrFile, { version: 1, root: path.join(f.base, 'unplugged', 'HOT-Step-Data'), pending: null });
  const v = versioned(f);
  const r = await resolveRuntime({ env: v.env, log: quiet, skipEngineProbe: true });
  assert.ok(r.ok, !r.ok ? r.error : '');
  assert.equal(r.runtime.dataRoot, f.sibling);
  assert.ok(!fs.existsSync(path.join(f.app, 'app', 'HOT-Step-Data')));
});
