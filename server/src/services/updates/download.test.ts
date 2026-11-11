import test, { type TestContext } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { deflateRawSync } from 'node:zlib';
import { execFileSync } from 'node:child_process';
import { createDownloader, DOWNLOAD_LIMITS, prepareBlocker, type DownloadDeps } from './download.js';
import type { BuildInfo, InstalledIdentity } from '../../contracts/updates.js';

const sha = (b: Buffer | string) => createHash('sha256').update(b).digest('hex');
const SOURCE = 'a'.repeat(40);
const INFO: BuildInfo = { format: 1, tag: 'v2.0.0', sha: SOURCE, channel: 'release', os: 'windows', arch: 'x64', variant: 'cpu', engineSha: 'b'.repeat(40), updaterProtocol: 1 };
const NAME = 'HOT-Step-CPP-v2.0.0-win-x64-cpu.zip';
const URL = `https://github.com/scragnog/HOT-Step-CPP/releases/download/v2.0.0/${NAME}`;
const API = 'https://api.github.com/repos/scragnog/HOT-Step-CPP';
interface Entry { name: string; text?: string; attrs?: number; deflate?: boolean; localName?: string; declaredSize?: number; extra?: Buffer }

// Small independent fixture writer permits paths a normal archiver sanitizes.
function crc32(b: Buffer): number {
  let c = 0xffffffff;
  for (const byte of b) {
    c ^= byte;
    for (let bit = 0; bit < 8; bit++) c = (c >>> 1) ^ ((c & 1) ? 0xedb88320 : 0);
  }
  return (c ^ 0xffffffff) >>> 0;
}
function zip(entries: Entry[]): Buffer {
  const local: Buffer[] = [], central: Buffer[] = [];
  let offset = 0;
  for (const e of entries) {
    const name = Buffer.from(e.name), localName = Buffer.from(e.localName ?? e.name), extra = e.extra ?? Buffer.alloc(0);
    const data = Buffer.from(e.text ?? 'fixture'), packed = e.deflate ? deflateRawSync(data) : data;
    const h = Buffer.alloc(30);
    h.writeUInt32LE(0x04034b50); h.writeUInt16LE(20, 4); h.writeUInt16LE(0x800, 6); h.writeUInt16LE(e.deflate ? 8 : 0, 8);
    h.writeUInt32LE(crc32(data), 14); h.writeUInt32LE(packed.length, 18); h.writeUInt32LE(e.declaredSize ?? data.length, 22);
    h.writeUInt16LE(localName.length, 26); h.writeUInt16LE(extra.length, 28);
    local.push(h, localName, extra, packed);
    const c = Buffer.alloc(46);
    c.writeUInt32LE(0x02014b50); c.writeUInt16LE(0x314, 4); c.writeUInt16LE(20, 6); c.writeUInt16LE(0x800, 8); c.writeUInt16LE(e.deflate ? 8 : 0, 10);
    c.writeUInt32LE(crc32(data), 16); c.writeUInt32LE(packed.length, 20); c.writeUInt32LE(e.declaredSize ?? data.length, 24);
    c.writeUInt16LE(name.length, 28); c.writeUInt16LE(extra.length, 30); c.writeUInt32LE(e.attrs ?? 0, 38); c.writeUInt32LE(offset, 42);
    central.push(c, name, extra);
    offset += h.length + localName.length + extra.length + packed.length;
  }
  const directory = Buffer.concat(central), end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50); end.writeUInt16LE(entries.length, 8); end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(directory.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...local, directory, end]);
}
function layout(info: BuildInfo = INFO): Entry[] {
  return [
    { name: 'build-info.json', text: JSON.stringify(info) }, { name: 'VERSION', text: '2.0.0' },
    ...['node/node.exe', 'server/server.mjs', 'server/node_modules/better-sqlite3/build/Release/better_sqlite3.node', 'ui/dist/index.html', 'HOT-Step.bat', 'profile-path.mjs', '.env.example', 'README.txt',
      ...['ace-server', 'ace-train', 'ace-midi', 'ace-caption', 'mastering', 'mp3-codec', 'neural-codec', 'vst-host', 'quantize'].map(n => `engine/${n}.exe`)].map(name => ({ name, text: 'fixture', deflate: true })),
  ];
}

function setup(t: TestContext, entries = layout()) {
  const root = fs.mkdtempSync(path.join(process.cwd(), '.test-update-'));
  t.after(() => {
    assert.equal(path.dirname(root), process.cwd());
    fs.rmSync(root, { recursive: true, force: true });
  });
  for (const n of ['install/app/current', 'data']) fs.mkdirSync(path.join(root, n), { recursive: true });
  const archive = zip(entries);
  const sidecar = Buffer.from(`${sha(archive)}  ${NAME}`);
  const asset = (name: string, b: Buffer, id: number) => ({ id, name, size: b.length, digest: `sha256:${sha(b)}`, browser_download_url: name.endsWith('.sha256') ? `${URL}.sha256` : URL });
  const metadata = { tag_name: 'v2.0.0', published_at: '2026-11-01T12:00:00Z', draft: false, prerelease: false, assets: [asset(NAME, archive, 1), asset(`${NAME}.sha256`, sidecar, 2)] };
  let installed: InstalledIdentity = { kind: 'portable', version: '1.0.0', tag: 'v1.0.0', sha: SOURCE, variant: 'cpu', buildInfo: { ...INFO, tag: 'v1.0.0' }, buildInfoProblem: null, git: null };
  const calls: string[] = [];
  let respond: ((url: string, init: RequestInit) => Promise<Response> | Response | null) | null = null;
  let now = 1000;
  const deps: DownloadDeps = {
    fetch: async (url, init) => {
      calls.push(url);
      const override = respond && await respond(url, init);
      if (override) return override;
      if (url === `${API}/releases/latest`) return Response.json(metadata);
      if (url === `${API}/commits/v2.0.0`) return Response.json({ sha: SOURCE });
      if (url === `${URL}.sha256`) return new Response(sidecar);
      if (url === URL) return new Response(archive, { headers: { 'content-length': String(archive.length) } });
      throw new Error(`Unexpected fetch ${url}`);
    },
    install: () => installed, now: () => now, platform: 'win32', arch: 'x64',
    payloadRoot: path.join(root, 'install/app/current'), dataRoot: path.join(root, 'data'), stateRoot: path.join(root, 'install/updater'),
    freeBytes: async () => 32 * 1024 ** 3,
  };
  const d = createDownloader(deps);
  return { root, d, deps, metadata, archive, calls,
    install: (i: InstalledIdentity) => { installed = i; }, installed: () => installed,
    respond: (fn: NonNullable<typeof respond>) => { respond = fn; }, advance: (ms: number) => { now += ms; },
    async prepare() { await d.discover(); const id = d.status().candidate?.id; assert.ok(id, d.status().reason ?? 'no candidate'); d.prepare(id); await d.wait(); return d.status(); },
    assertClean() { assert.deepEqual(fs.readdirSync(deps.stateRoot!), []); assert.deepEqual(fs.readdirSync(deps.dataRoot!), []); assert.deepEqual(fs.readdirSync(deps.payloadRoot), []); },
  };
}

test('stages a complete stored/deflate archive, leaves active payload and data untouched', async t => {
  const s = setup(t);
  const result = await s.prepare();
  assert.equal(result.phase, 'ready', result.error ?? '');
  assert.equal(result.downloadedBytes, s.archive.length);
  const ready = s.d.getPrepared(); assert.ok(ready);
  assert.equal(ready.buildInfo.sha, SOURCE);
  assert.equal(fs.readFileSync(path.join(ready.payloadRoot, 'ui/dist/index.html'), 'utf8'), 'fixture');
  assert.ok(fs.existsSync(path.join(s.deps.stateRoot!, ready.id, 'ready.json')));
  assert.deepEqual(fs.readdirSync(s.deps.payloadRoot), []);
  assert.deepEqual(fs.readdirSync(s.deps.dataRoot!), []);
  assert.equal(s.d.status().offered, false);
  assert.throws(() => s.d.prepare(result.candidate!.id), /already staged/);
});

test('accepts empty directories, UTF-8 filenames and CRC across multiple inflate chunks', async t => {
  const s = setup(t, [...layout(), { name: 'engine/empty/', text: '', attrs: 0x10 }, { name: 'plugins/caf\u00e9.lua', text: 'fixture'.repeat(30_000), deflate: true }]);
  assert.equal((await s.prepare()).phase, 'ready', s.d.status().error ?? '');
});

for (const writer of ['powershell.exe', 'pwsh.exe']) test(`accepts the .NET release ZIP writer via ${writer}`, { skip: process.platform !== 'win32' }, async t => {
  const s = setup(t);
  const input = path.join(s.root, 'zip-input'), output = path.join(s.root, 'dotnet.zip');
  for (const e of layout()) {
    const file = path.join(input, e.name);
    fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, e.text ?? 'fixture');
  }
  execFileSync(writer, ['-NoProfile', '-NonInteractive', '-Command',
    'Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::CreateFromDirectory($env:UPDATE_FIXTURE_INPUT, $env:UPDATE_FIXTURE_OUTPUT, [System.IO.Compression.CompressionLevel]::Optimal, $false)'],
  { env: { ...process.env, UPDATE_FIXTURE_INPUT: input, UPDATE_FIXTURE_OUTPUT: output }, timeout: 30_000, windowsHide: true });
  const archive = fs.readFileSync(output), side = Buffer.from(`${sha(archive)}  ${NAME}`);
  s.metadata.assets[0].size = archive.length; s.metadata.assets[0].digest = `sha256:${sha(archive)}`;
  s.metadata.assets[1].size = side.length; s.metadata.assets[1].digest = `sha256:${sha(side)}`;
  s.respond(url => url === URL ? new Response(archive) : url === `${URL}.sha256` ? new Response(side) : null);
  assert.equal((await s.prepare()).phase, 'ready', s.d.status().error ?? '');
});

test('protocol 0, git, non-Windows, invalid identity, unknown variant and arch stay inert', async t => {
  const s = setup(t);
  for (const patch of [
    { buildInfo: { ...INFO, updaterProtocol: 0 } }, { kind: 'git' as const },
    { buildInfo: null }, { buildInfo: { ...INFO, engineSha: 'bad' } },
    { variant: null }, { buildInfo: { ...INFO, variant: 'rocm' as const } },
    { buildInfo: { ...INFO, arch: 'arm64' } },
    { version: '9.9.9' },
  ]) {
    s.install({ ...s.installed(), ...patch });
    await s.d.discover(); assert.equal(s.d.status().offered, false); assert.ok(s.d.status().reason);
    assert.throws(() => s.d.prepare('arbitrary'));
    s.install({ kind: 'portable', version: '1.0.0', tag: 'v1.0.0', sha: SOURCE, variant: 'cpu', buildInfo: { ...INFO, tag: 'v1.0.0' }, buildInfoProblem: null, git: null });
  }
  for (const platform of ['linux', 'darwin'] as const) assert.match(prepareBlocker({ ...s.deps, platform })!, /Windows/);
  assert.equal(s.calls.length, 0);
  assert.equal(fs.existsSync(s.deps.stateRoot!), false);
});

test('a checkout marker is refused even if Git could not identify the checkout', async t => {
  const s = setup(t); fs.writeFileSync(path.join(s.deps.payloadRoot, '.git'), 'gitdir: missing');
  await s.d.discover(); assert.equal(s.calls.length, 0); assert.match(s.d.status().reason!, /update.bat/);
});

test('state inside payload or data root, missing launcher state, and junctions are refused', t => {
  const s = setup(t);
  assert.match(prepareBlocker({ ...s.deps, stateRoot: path.join(s.deps.dataRoot!, 'updater') })!, /outside/);
  assert.match(prepareBlocker({ ...s.deps, stateRoot: s.deps.payloadRoot })!, /outside/);
  assert.match(prepareBlocker({ ...s.deps, stateRoot: null })!, /launcher/);
  const link = path.join(s.root, 'linked');
  fs.symlinkSync(s.deps.dataRoot!, link, process.platform === 'win32' ? 'junction' : 'dir');
  assert.match(prepareBlocker({ ...s.deps, stateRoot: path.join(link, 'updater') })!, /links|reparse/);
});

for (const [name, modify, expected] of [
  ['missing sidecar', (s: ReturnType<typeof setup>) => { s.metadata.assets.pop(); }, /Missing/],
  ['missing API digest', (s: ReturnType<typeof setup>) => { s.metadata.assets[0].digest = ''; }, /digest/],
  ['wrong variant', (s: ReturnType<typeof setup>) => { s.metadata.assets[0].name = NAME.replace('cpu.zip', 'vulkan.zip'); }, /Missing/],
  ['wrong asset URL', (s: ReturnType<typeof setup>) => { s.metadata.assets[0].browser_download_url = 'https://evil.example/a.zip'; }, /unexpected/],
  ['duplicate asset', (s: ReturnType<typeof setup>) => { s.metadata.assets.push(s.metadata.assets[0]); }, /Missing/],
  ['prerelease', (s: ReturnType<typeof setup>) => { s.metadata.prerelease = true; }, /stable/],
] as const) test(`candidate refusal: ${name}`, async t => {
  const s = setup(t); modify(s); await s.d.discover();
  assert.equal(s.d.status().offered, false); assert.match(s.d.status().reason!, expected);
  assert.equal(fs.existsSync(s.deps.stateRoot!), false);
});

test('unknown and expired candidate IDs never start downloads', async t => {
  const s = setup(t); await s.d.discover();
  assert.throws(() => s.d.prepare('https://evil.example/x.zip'), /Unknown/);
  const id = s.d.status().candidate!.id;
  s.advance(3600_001);
  assert.throws(() => s.d.prepare(id), /expired/);
  assert.equal(fs.existsSync(s.deps.stateRoot!), false);
});

test('low free space fails before downloading and releases lock', async t => {
  const s = setup(t); s.deps.freeBytes = async () => 1;
  const r = await s.prepare(); assert.equal(r.phase, 'failed'); assert.match(r.error!, /free space/);
  assert.ok(!s.calls.includes(URL)); s.assertClean();
});

test('space dropping before extraction removes the verified archive', async t => {
  const s = setup(t); let checks = 0;
  s.deps.freeBytes = async () => ++checks === 1 ? 32 * 1024 ** 3 : 1;
  const r = await s.prepare(); assert.equal(r.phase, 'failed'); assert.match(r.error!, /free space/);
  assert.ok(s.calls.includes(URL)); s.assertClean();
});

for (const [name, response, expected] of [
  ['truncated', (b: Buffer) => new Response(b.subarray(0, b.length - 1)), /truncated/],
  ['wrong hash', (b: Buffer) => { const bad = Buffer.from(b); bad[10] ^= 1; return new Response(bad); }, /SHA-256/],
  ['too long', (b: Buffer) => new Response(Buffer.concat([b, Buffer.from('extra')])), /byte limit/],
  ['wrong length header', (b: Buffer) => new Response(b, { headers: { 'content-length': '1' } }), /content length/],
  ['host redirect', () => new Response(null, { status: 302, headers: { location: 'https://evil.example/archive' } }), /GitHub asset host/],
] as const) test(`download ${name} cleans its files`, async t => {
  const s = setup(t); s.respond(url => url === URL ? response(s.archive) : null);
  const r = await s.prepare(); assert.equal(r.phase, 'failed'); assert.match(r.error!, expected);
  assert.equal(s.d.getPrepared(), null); s.assertClean();
  assert.ok(!s.calls.some(c => c.includes('evil.example')));
});

test('mismatched sidecar refuses before downloading the archive', async t => {
  const s = setup(t);
  const side = Buffer.from(`${'f'.repeat(64)}  ${NAME}`);
  s.metadata.assets[1].digest = `sha256:${sha(side)}`;
  s.respond(url => url === `${URL}.sha256` ? new Response(side) : null);
  const r = await s.prepare(); assert.match(r.error!, /sidecar/); assert.ok(!s.calls.includes(URL)); s.assertClean();
});

test('changed release candidate is refused at prepare', async t => {
  const s = setup(t); await s.d.discover(); const id = s.d.status().candidate!.id;
  s.metadata.assets[0].digest = `sha256:${'f'.repeat(64)}`;
  s.d.prepare(id); await s.d.wait(); assert.match(s.d.status().error!, /candidate changed/); s.assertClean();
});

test('same instance and separate instances cannot prepare concurrently', async t => {
  const s = setup(t); const other = createDownloader(s.deps);
  await s.d.discover(); await other.discover();
  const id = s.d.status().candidate!.id;
  s.d.prepare(id);
  assert.throws(() => s.d.prepare(id), /already running/);
  other.prepare(other.status().candidate!.id);
  await Promise.all([s.d.wait(), other.wait()]);
  const states = [s.d.status(), other.status()];
  assert.equal(states.filter(r => r.phase === 'ready').length, 1);
  assert.match(states.find(r => r.phase === 'failed')!.error!, /lock/);
  assert.equal(fs.existsSync(path.join(s.deps.stateRoot!, 'prepare.lock')), false);
});

test('a crashed owner lock is recovered and its incomplete stage is swept', async t => {
  const s = setup(t);
  const deadPid = Number(execFileSync(process.execPath, ['-e', 'process.stdout.write(String(process.pid))'], { encoding: 'utf8', timeout: 10_000, windowsHide: true }));
  assert.throws(() => process.kill(deadPid, 0), { code: 'ESRCH' });
  fs.mkdirSync(s.deps.stateRoot!, { recursive: true });
  fs.writeFileSync(path.join(s.deps.stateRoot!, 'prepare.lock'), JSON.stringify({ pid: deadPid }));
  const orphan = path.join(s.deps.stateRoot!, 'prepare-ab12CD');
  fs.mkdirSync(orphan); fs.writeFileSync(path.join(orphan, 'archive.zip.part'), 'interrupted');
  const r = await s.prepare();
  assert.equal(r.phase, 'ready', r.error ?? '');
  assert.equal(fs.existsSync(orphan), false);
  assert.equal(fs.existsSync(path.join(s.deps.stateRoot!, 'prepare.lock')), false);
});

test('a ready stage survives restart and prevents duplicate staging until launcher reconciliation', async t => {
  const s = setup(t); const r = await s.prepare(); assert.equal(r.phase, 'ready');
  const original = s.d.getPrepared()!;
  const restarted = createDownloader(s.deps); await restarted.discover();
  const downloads = s.calls.filter(url => url === URL).length;
  restarted.prepare(restarted.status().candidate!.id); await restarted.wait();
  assert.equal(restarted.status().phase, 'failed');
  assert.match(restarted.status().error!, /launcher.*reconcile/i);
  assert.ok(fs.existsSync(path.join(s.deps.stateRoot!, original.id, 'ready.json')));
  assert.deepEqual(fs.readdirSync(s.deps.stateRoot!), [original.id]);
  assert.equal(s.calls.filter(url => url === URL).length, downloads);
});

test('a live owner, including an EPERM probe, keeps its lock and staging', async t => {
  const s = setup(t);
  fs.mkdirSync(s.deps.stateRoot!, { recursive: true });
  const lock = path.join(s.deps.stateRoot!, 'prepare.lock');
  fs.writeFileSync(lock, JSON.stringify({ pid: process.pid }));
  const orphan = path.join(s.deps.stateRoot!, 'prepare-ab12CD'); fs.mkdirSync(orphan);
  for (const denied of [false, true]) {
    if (denied) t.mock.method(process, 'kill', () => { throw Object.assign(new Error('permission denied'), { code: 'EPERM' }); });
    const r = await s.prepare(); assert.equal(r.phase, 'failed'); assert.match(r.error!, /holds.*lock/);
    assert.ok(fs.existsSync(lock)); assert.ok(fs.existsSync(orphan)); assert.ok(!s.calls.includes(URL));
  }
});

test('old malformed locks recover; fresh malformed locks allow an owner to finish initialization', async t => {
  const s = setup(t);
  fs.mkdirSync(s.deps.stateRoot!, { recursive: true });
  const lock = path.join(s.deps.stateRoot!, 'prepare.lock');
  fs.writeFileSync(lock, '{');
  const fresh = await s.prepare(); assert.equal(fresh.phase, 'failed'); assert.match(fresh.error!, /initializing/);
  assert.ok(fs.existsSync(lock));
  const old = new Date(Date.now() - 60_000); fs.utimesSync(lock, old, old);
  const recovered = await s.prepare(); assert.equal(recovered.phase, 'ready', recovered.error ?? '');
});

test('sweeping incomplete stages leaves unrelated directories and junction targets untouched', async t => {
  const s = setup(t);
  fs.mkdirSync(s.deps.stateRoot!, { recursive: true });
  const unrelated = path.join(s.deps.stateRoot!, 'prepare-not-owned'); fs.mkdirSync(unrelated);
  const link = path.join(s.deps.stateRoot!, 'prepare-li12NK');
  fs.writeFileSync(path.join(s.deps.dataRoot!, 'keep.txt'), 'untouched');
  fs.symlinkSync(s.deps.dataRoot!, link, process.platform === 'win32' ? 'junction' : 'dir');
  assert.equal((await s.prepare()).phase, 'ready');
  assert.ok(fs.existsSync(unrelated)); assert.ok(fs.lstatSync(link).isSymbolicLink());
  assert.equal(fs.readFileSync(path.join(s.deps.dataRoot!, 'keep.txt'), 'utf8'), 'untouched');
});

test('a release recheck preserves a prior failed preparation phase and its error', async t => {
  const s = setup(t); s.deps.freeBytes = async () => 1;
  const failed = await s.prepare(); assert.equal(failed.phase, 'failed');
  await s.d.discover();
  assert.equal(s.d.status().phase, 'failed'); assert.equal(s.d.status().error, failed.error);
});

test('cancellation cleans .part and staging, then a retry starts from zero', async t => {
  const s = setup(t);
  let entered!: () => void;
  const downloading = new Promise<void>(resolve => { entered = resolve; });
  s.respond((url, init) => url === URL ? new Response(new ReadableStream({
    start(stream) { stream.enqueue(s.archive.subarray(0, 100)); entered(); init.signal!.addEventListener('abort', () => stream.error(new Error('aborted')), { once: true }); },
  })) : null);
  await s.d.discover(); const id = s.d.status().candidate!.id; s.d.prepare(id);
  await downloading; s.d.cancel(id); await s.d.wait();
  assert.equal(s.d.status().phase, 'cancelled'); s.assertClean();
  s.respond(() => null); s.d.prepare(id); await s.d.wait();
  assert.equal(s.d.status().phase, 'ready'); assert.equal(s.d.status().downloadedBytes, s.archive.length);
});

for (const e of [
  { name: '../escape' }, { name: '/absolute' }, { name: 'C:/absolute' }, { name: '\\\\server\\share' },
  { name: 'engine/../../escape' }, { name: 'engine\\..\\..\\escape' }, { name: 'engine/CON.txt' },
  { name: 'engine/com1.dll' }, { name: 'engine/alias.' }, { name: 'engine/ads:stream' },
  { name: 'engine/CONIN$' }, { name: 'engine/CONOUT$.txt' }, { name: 'engine/NUL .txt' },
  { name: 'models/weights' }, { name: 'HOT-Step-CPP/engine/ace-server.exe' },
  { name: 'engine/link', attrs: 0xa1ff0000 }, { name: 'engine/junction', attrs: 0x400 },
  { name: 'engine/device', attrs: 0x40 }, { name: 'engine/pipe', attrs: 0x11ff0000 },
  { name: 'engine/wrong-local', localName: 'engine/escape-name' },
  { name: 'engine/unix-link', extra: Buffer.from([0x0d, 0, 0, 0]) },
  { name: 'engine/zip64', extra: Buffer.from([1, 0, 0, 0]) },
] satisfies Entry[]) test(`archive refuses ${JSON.stringify(e.name)} attrs=${e.attrs ?? 0}`, async t => {
  const s = setup(t, [...layout(), e]); const r = await s.prepare();
  assert.equal(r.phase, 'failed', r.error ?? ''); assert.ok(r.error); s.assertClean();
  assert.equal(fs.existsSync(path.join(s.root, 'escape')), false);
});

for (const entries of [
  [...layout(), { name: 'engine/ACE-SERVER.exe' }],
  [...layout(), { name: 'ENGINE/new.exe' }],
  [...layout(), { name: 'engine/ace-server.exe' }],
  [...layout(), { name: 'engine' }],
]) test('case collisions, duplicate files and file-directory collisions are refused', async t => {
  const s = setup(t, entries); assert.equal((await s.prepare()).phase, 'failed'); s.assertClean();
});

test('caps entry count and expanded size, including dishonest deflate sizes', async t => {
  for (const [name, entries, limits] of [
    ['entry count', layout(), { entries: 2 }],
    ['expanded bytes', layout(), { expandedBytes: 1 }],
    ['dishonest size', [...layout(), { name: 'engine/bomb', text: 'x'.repeat(100_000), deflate: true, declaredSize: 1 }], {}],
  ] as const) {
    await t.test(name, async sub => {
      const s = setup(sub, [...entries]); s.deps.limits = limits;
      const d = createDownloader(s.deps); await d.discover(); d.prepare(d.status().candidate!.id); await d.wait();
      assert.equal(d.status().phase, 'failed'); s.assertClean();
    });
  }
});

for (const [name, entries] of [
  ['variant mismatch', layout({ ...INFO, variant: 'vulkan' })],
  ['source mismatch', layout({ ...INFO, sha: 'c'.repeat(40) })],
  ['protocol mismatch', layout({ ...INFO, updaterProtocol: 2 })],
  ['old protocol', layout({ ...INFO, updaterProtocol: 0 })],
  ['missing UI', layout().filter(e => e.name !== 'ui/dist/index.html')],
  ['unexpected backend', [...layout(), { name: 'engine/ggml-cuda.dll' }]],
  ['wrong marker', [...layout(), { name: 'engine/.variant', text: 'vulkan' }]],
] as const) test(`layout refuses ${name}`, async t => {
  const s = setup(t, [...entries]); assert.equal((await s.prepare()).phase, 'failed'); s.assertClean();
});

test('download caps match the non-ZIP64 format limit', () => {
  assert.ok(DOWNLOAD_LIMITS.archiveBytes < 0xffffffff);
  assert.ok(DOWNLOAD_LIMITS.entries < 65535);
});
