// Download and stage only. No payload activation, database or data-root writes.
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { createInflateRaw } from 'node:zlib';
import * as zlib from 'node:zlib';
import type { BuildInfo, InstalledIdentity, UpdateCandidate, UpdatePrepareStatus } from '../../contracts/updates.js';
import { parseBuildInfo, REPO } from './install.js';

const WINDOWS_VARIANTS = ['cuda13.1', 'cuda12.8', 'cuda12-volta', 'vulkan', 'cpu'];
const SHA = /^[0-9a-f]{40}$/;
const DIGEST = /^sha256:([0-9a-f]{64})$/;
const STABLE = /^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;
const API = `https://api.github.com/repos/${REPO}`;
const EXPIRES_MS = 3600_000;
export const DOWNLOAD_LIMITS = {
  archiveBytes: 2 * 1024 ** 3, expandedBytes: 8 * 1024 ** 3,
  entries: 20_000, directoryBytes: 16 * 1024 ** 2, reserveBytes: 256 * 1024 ** 2,
};
type Limits = typeof DOWNLOAD_LIMITS;
export type DownloadFetch = (url: string, init: RequestInit) => Promise<Response>;
export interface DownloadDeps {
  fetch: DownloadFetch;
  install(): InstalledIdentity;
  platform: NodeJS.Platform;
  arch: string;
  payloadRoot: string;
  dataRoot: string | null;
  /** Stable launcher's <install>/updater, never the profile's data folder. */
  stateRoot: string | null;
  now(): number;
  freeBytes?(directory: string): Promise<number>;
  limits?: Partial<Limits>;
}
interface Asset { id: number; name: string; size: number; browser_download_url: string; digest: string }
interface Candidate extends UpdateCandidate {
  asset: Asset;
  sidecar: Asset;
  expiresAt: number;
  protocol: number;
}
export class PrepareError extends Error {
  constructor(message: string, public status = 409) { super(message); }
}
const refuse = (message: string): never => { throw new PrepareError(message); };
const within = (parent: string, child: string) => {
  const rel = path.relative(parent, child);
  return rel === '' || (!rel.startsWith(`..${path.sep}`) && rel !== '..' && !path.isAbsolute(rel));
};
const overlap = (a: string, b: string) => within(a, b) || within(b, a);
const newer = (want: string, have: string) => {
  const a = STABLE.exec(want)?.slice(1).map(BigInt);
  const b = STABLE.exec(have)?.slice(1).map(BigInt);
  if (!a || !b) return false;
  for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] > b[i];
  return false;
};

/** Check every existing component, including junctions, before creating state. */
function noLinks(target: string): void {
  const absolute = path.resolve(target);
  const root = path.parse(absolute).root;
  let current = root;
  for (const part of absolute.slice(root.length).split(path.sep).filter(Boolean)) {
    current = path.join(current, part);
    try {
      const st = fs.lstatSync(current);
      if (st.isSymbolicLink() || !st.isDirectory()) refuse('Updater paths must be ordinary directories without links or reparse points');
      if (path.resolve(fs.realpathSync(current)).toLowerCase() !== current.toLowerCase()) refuse('Updater path resolves through a reparse point');
    } catch (e: any) { if (e.code !== 'ENOENT') throw e; }
  }
}

export function prepareBlocker(d: DownloadDeps): string | null {
  const id = d.install();
  if (id.kind === 'git' || id.git || fs.existsSync(path.join(d.payloadRoot, '.git'))) return 'Source checkouts update locally with update.bat';
  if (d.platform !== 'win32') return 'Automatic preparation is available on Windows portable installs only';
  if (id.kind !== 'portable') return 'Could not prove this is a portable installation';
  const info = id.buildInfo;
  if (!info || !validIdentity(info)) return id.buildInfoProblem ?? 'A valid build-info.json is required; install a current package manually';
  if (info.updaterProtocol < 1) return 'This package needs one manual upgrade to an updater-capable launcher';
  if (id.tag !== info.tag || id.sha !== info.sha || id.version.trim().replace(/^v/, '') !== info.tag!.slice(1)) return 'The installed version disagrees with build-info.json';
  if (info.os !== 'windows' || info.arch !== 'x64' || d.arch !== 'x64') return 'This build does not match Windows x64';
  if (!WINDOWS_VARIANTS.includes(info.variant) || id.variant !== info.variant) return 'The exact Windows package variant is unknown or unsupported';
  if (!d.stateRoot || !path.isAbsolute(d.stateRoot)) return 'The stable launcher has not supplied an updater state directory';
  if (!d.dataRoot || !path.isAbsolute(d.dataRoot)) return 'A separate data folder is required';
  if (overlap(path.resolve(d.stateRoot), path.resolve(d.payloadRoot)) || overlap(path.resolve(d.stateRoot), path.resolve(d.dataRoot))) return 'Updater state must be outside the payload and data folder';
  try { noLinks(d.stateRoot); noLinks(d.payloadRoot); noLinks(d.dataRoot); }
  catch (e) { return (e as Error).message; }
  return null;
}

function validIdentity(info: BuildInfo): boolean {
  return !!parseBuildInfo(JSON.stringify(info)).info && SHA.test(info.sha) && SHA.test(info.engineSha)
    && info.channel === 'release' && typeof info.tag === 'string' && STABLE.test(info.tag);
}

function assetUrl(raw: string, tag: string, name: string): string {
  const expected = `https://github.com/${REPO}/releases/download/${tag}/${name}`;
  if (raw !== expected) refuse('Release asset has an unexpected download URL');
  return raw;
}
function deliveryUrl(raw: string): URL {
  const u = new URL(raw);
  if (u.protocol !== 'https:' || u.username || u.password || u.port || u.hash
    || !['github.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com'].includes(u.hostname)) refuse('Download redirect host is not a GitHub asset host');
  if (u.hostname === 'github.com' && !u.pathname.startsWith(`/${REPO}/releases/download/`)) refuse('Unexpected GitHub download path');
  return u;
}
async function request(d: DownloadDeps, url: string, signal: AbortSignal, api = false): Promise<Response> {
  for (let redirects = 0; redirects <= 4; redirects++) {
    if (!api) deliveryUrl(url);
    const r = await d.fetch(url, {
      redirect: 'manual', signal,
      headers: { Accept: api ? 'application/vnd.github+json' : 'application/octet-stream', 'User-Agent': 'HOT-Step-updater', 'Accept-Encoding': 'identity' },
    });
    if ([301, 302, 303, 307, 308].includes(r.status)) {
      await r.body?.cancel();
      const location = r.headers.get('location');
      if (api || !location) refuse('Unexpected GitHub API redirect');
      url = new URL(location!, url).href;
      continue;
    }
    if (r.status !== 200) { await r.body?.cancel(); refuse(`GitHub answered HTTP ${r.status}`); }
    const encoding = r.headers.get('content-encoding');
    if (encoding && encoding !== 'identity') { await r.body?.cancel(); refuse('Unexpected HTTP content encoding'); }
    return r;
  }
  return refuse('Too many GitHub download redirects');
}
async function boundedBody(res: Response, limit: number, signal: AbortSignal): Promise<Buffer> {
  if (!res.body) return refuse('Empty GitHub response');
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of Readable.fromWeb(res.body as any)) {
    signal.throwIfAborted();
    const b = Buffer.from(chunk);
    size += b.length;
    if (size > limit) refuse('GitHub response exceeds the byte limit');
    chunks.push(b);
  }
  return Buffer.concat(chunks);
}

async function releaseCandidate(d: DownloadDeps, limits: Limits, signal: AbortSignal): Promise<Omit<Candidate, 'id' | 'expiresAt'>> {
  const info = d.install().buildInfo!;
  const raw = await boundedBody(await request(d, `${API}/releases/latest`, signal, true), 1024 ** 2, signal);
  const j = JSON.parse(raw.toString('utf8'));
  if (!j || j.draft !== false || j.prerelease !== false || typeof j.tag_name !== 'string' || !STABLE.test(j.tag_name)
    || !j.published_at || !Array.isArray(j.assets)) refuse('Latest release is not a published stable release');
  if (!newer(j.tag_name, info.tag!)) refuse('There is no newer stable release for this installation');
  const name = `HOT-Step-CPP-${j.tag_name}-win-x64-${info.variant}.zip`;
  const pick = (n: string, cap: number): Asset => {
    const hits = j.assets.filter((a: any) => a?.name === n);
    const a = hits[0];
    if (hits.length !== 1 || !a || !Number.isSafeInteger(a.id) || a.id <= 0 || !Number.isSafeInteger(a.size) || a.size <= 0 || a.size > cap
      || typeof a.digest !== 'string' || !DIGEST.test(a.digest)) refuse(`Missing or invalid asset or SHA-256 digest: ${n}`);
    assetUrl(a.browser_download_url, j.tag_name, n);
    return { id: a.id, name: n, size: a.size, digest: a.digest, browser_download_url: a.browser_download_url };
  };
  const asset = pick(name, limits.archiveBytes);
  const sidecar = pick(`${name}.sha256`, 4096);
  const commit = JSON.parse((await boundedBody(await request(d, `${API}/commits/${j.tag_name}`, signal, true), 1024 ** 2, signal)).toString('utf8'));
  if (typeof commit?.sha !== 'string' || !SHA.test(commit.sha)) refuse('Release tag has no exact source commit');
  return { tag: j.tag_name, sha: commit.sha, variant: info.variant, assetName: name, bytes: asset.size, asset, sidecar, protocol: info.updaterProtocol };
}

/** A strict stored/deflate ZIP reader using Node streams, with no shell or dependency.
 * ZIP64, split/encrypted archives and alternate filename/link extra fields fail closed.
 * Limits are deliberately below ZIP64's compressed size boundary. */
interface ZipEntry { name: string; rawName: Buffer; directory: boolean; flags: number; method: number; crc: number; packed: number; size: number; offset: number; start: number }
// The versioned layout release/updater/package-layout.mjs builds (protocol 1+):
// the stable root files, runtime/node.exe for the launcher, and app/<id>/.
const ROOT_FILES = new Set(['HOT-Step.bat', 'launcher.mjs', 'current.json', 'README.txt']);
const PAYLOAD_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const PAYLOAD_DIRS = new Set(['runtime', 'engine', 'server', 'ui', 'Essentia', 'noise_samples', 'plugins', 'presets']);
const PAYLOAD_FILES = new Set(['VERSION', 'build-info.json', 'payload-manifest.json', '.env.example', 'profile-path.mjs', 'open-browser-if-needed.mjs', 'README.txt']);
function allowedPath(parts: string[], directory: boolean): boolean {
  const [top, id, child] = parts;
  if (parts.length === 1) return directory ? top === 'app' || top === 'runtime' : ROOT_FILES.has(top);
  if (top === 'runtime') return parts.length === 2 && !directory && id === 'node.exe';
  if (top !== 'app' || !PAYLOAD_ID.test(id) || id.includes('..')) return false;
  if (parts.length === 2) return directory;
  return parts.length === 3 && !directory ? PAYLOAD_FILES.has(child) : PAYLOAD_DIRS.has(child);
}
function zipName(name: string): string[] {
  if (!name || name.length > 220 || /[\\\x00-\x1f\x7f<>:"|?*]/.test(name) || name.startsWith('/') || name.normalize('NFC') !== name) refuse('Archive has an unsafe or absolute path');
  const parts = name.replace(/\/$/, '').split('/');
  if (parts.some(p => !p || p === '.' || p === '..' || /[. ]$/.test(p) || /^(con|conin\$|conout\$|prn|aux|nul|com[0-9¹²³]|lpt[0-9¹²³])(?:[ .]|$)/i.test(p))) refuse('Archive has traversal, an alias or a Windows device name');
  if (!allowedPath(parts, name.endsWith('/'))) refuse('Archive has an unexpected root');
  return parts;
}
function zipExtras(b: Buffer): void {
  for (let p = 0; p < b.length;) {
    if (p + 4 > b.length) refuse('Malformed ZIP extra field');
    const id = b.readUInt16LE(p), n = b.readUInt16LE(p + 2);
    // Allow only extended timestamps and NTFS timestamps. Neither creates paths.
    if (![0x5455, 0x000a].includes(id) || p + 4 + n > b.length) refuse('Unsupported ZIP extra field, link or ZIP64 archive');
    p += 4 + n;
  }
}
async function readAt(h: fsp.FileHandle, position: number, size: number): Promise<Buffer> {
  const b = Buffer.alloc(size);
  let done = 0;
  while (done < size) {
    const r = await h.read(b, done, size - done, position + done);
    if (!r.bytesRead) refuse('Truncated ZIP archive');
    done += r.bytesRead;
  }
  return b;
}
async function zipEntries(file: string, limits: Limits, signal: AbortSignal): Promise<ZipEntry[]> {
  const h = await fsp.open(file, 'r');
  try {
    const size = (await h.stat()).size;
    if (size < 22) refuse('Truncated ZIP archive');
    const tail = await readAt(h, Math.max(0, size - 65557), Math.min(size, 65557));
    let end = -1;
    for (let i = tail.length - 22; i >= 0; i--) if (tail.readUInt32LE(i) === 0x06054b50 && i + 22 + tail.readUInt16LE(i + 20) === tail.length) { end = i; break; }
    if (end < 0) refuse('ZIP end record is missing');
    const count = tail.readUInt16LE(end + 10), bytes = tail.readUInt32LE(end + 12), start = tail.readUInt32LE(end + 16);
    if (tail.readUInt32LE(end + 4) !== 0 || count !== tail.readUInt16LE(end + 8) || count === 65535 || !count || count > limits.entries
      || bytes > limits.directoryBytes || start + bytes !== size - tail.length + end) refuse('ZIP directory, entry count or ZIP64 archive is unsupported');
    const central = await readAt(h, start, bytes);
    const entries: ZipEntry[] = [];
    const seen = new Map<string, { spelling: string; file: boolean; explicit: boolean }>();
    let p = 0, expanded = 0;
    for (let i = 0; i < count; i++) {
      signal.throwIfAborted();
      if (p + 46 > central.length || central.readUInt32LE(p) !== 0x02014b50) refuse('Malformed ZIP directory');
      const nameLen = central.readUInt16LE(p + 28), extraLen = central.readUInt16LE(p + 30), commentLen = central.readUInt16LE(p + 32);
      const next = p + 46 + nameLen + extraLen + commentLen;
      if (next > central.length || central.readUInt16LE(p + 34)) refuse('Truncated or split ZIP directory');
      const rawName = central.subarray(p + 46, p + 46 + nameLen);
      const flags = central.readUInt16LE(p + 8), method = central.readUInt16LE(p + 10);
      if (flags & ~0x080e || ![0, 8].includes(method)) refuse('Encrypted or unsupported ZIP entry');
      if (!(flags & 0x800) && rawName.some(v => v > 127)) refuse('ZIP filenames must be ASCII or UTF-8');
      // Older Windows .NET writers emit backslash separators. Validate their
      // canonical path while retaining the exact bytes for local-header checks.
      const name = new TextDecoder('utf-8', { fatal: true }).decode(rawName).replaceAll('\\', '/');
      const parts = zipName(name), directory = name.endsWith('/');
      const attrs = central.readUInt32LE(p + 38), mode = (attrs >>> 16) & 0xf000;
      if ((attrs & 0x440) || (mode && mode !== (directory ? 0x4000 : 0x8000)) || (!!(attrs & 0x10) && !directory)) refuse('Archive contains a link, device or reparse point');
      zipExtras(central.subarray(p + 46 + nameLen, p + 46 + nameLen + extraLen));
      for (let depth = 1; depth <= parts.length; depth++) {
        const spelling = parts.slice(0, depth).join('/'), key = spelling.toUpperCase();
        const last = depth === parts.length, file = last && !directory, old = seen.get(key);
        if (old && (old.spelling !== spelling || old.file || file || (last && old.explicit))) refuse('Archive contains case collisions or duplicate paths');
        seen.set(key, { spelling, file, explicit: !!old?.explicit || last });
      }
      const entry: ZipEntry = { name, rawName, directory, flags, method, crc: central.readUInt32LE(p + 16), packed: central.readUInt32LE(p + 20), size: central.readUInt32LE(p + 24), offset: central.readUInt32LE(p + 42), start: 0 };
      if (/(?:^|\/)(?:build-info\.json|VERSION|current\.json|\.variant|\.cuda-version)$/.test(name) && entry.size > 16_384) refuse('Package identity file is too large');
      expanded += entry.size;
      if (expanded > limits.expandedBytes || entry.packed === 0xffffffff || entry.size === 0xffffffff || entry.offset === 0xffffffff || (directory && (entry.size || entry.packed))) refuse('Archive expanded size exceeds the limit or uses ZIP64');
      const local = await readAt(h, entry.offset, 30);
      if (local.readUInt32LE(0) !== 0x04034b50 || local.readUInt16LE(6) !== flags || local.readUInt16LE(8) !== method || local.readUInt16LE(26) !== nameLen) refuse('ZIP local header disagrees with its directory');
      const localExtra = local.readUInt16LE(28);
      const localName = await readAt(h, entry.offset + 30, nameLen + localExtra);
      if (!localName.subarray(0, nameLen).equals(rawName)) refuse('ZIP local path disagrees with its directory');
      zipExtras(localName.subarray(nameLen));
      if (!(flags & 8) && (local.readUInt32LE(14) !== entry.crc || local.readUInt32LE(18) !== entry.packed || local.readUInt32LE(22) !== entry.size)) refuse('ZIP local size or checksum disagrees with its directory');
      entry.start = entry.offset + 30 + nameLen + localExtra;
      if (entry.start + entry.packed > start) refuse('ZIP entry overlaps its directory');
      entries.push(entry); p = next;
    }
    if (p !== central.length) refuse('Unexpected ZIP directory data');
    let previousEnd = 0;
    for (const e of [...entries].sort((a, b) => a.offset - b.offset)) {
      if (e.offset < previousEnd) refuse('Overlapping ZIP entries');
      previousEnd = e.start + e.packed;
    }
    return entries;
  } finally { await h.close(); }
}
const CRC_TABLE = Array.from({ length: 256 }, (_, i) => { let c = i; for (let j = 0; j < 8; j++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
function crcChunk(crc: number, b: Buffer): number {
  // Portable packages use Node 24; keep the Node 20 source-checkout fallback.
  if (typeof zlib.crc32 === 'function') return (zlib.crc32(b, (crc ^ 0xffffffff) >>> 0) ^ 0xffffffff) >>> 0;
  for (const v of b) crc = CRC_TABLE[(crc ^ v) & 255] ^ (crc >>> 8);
  return crc >>> 0;
}

/** Reclaim a proven dead owner once. A fresh malformed file may belong to a
 * process between exclusive creation and writing its owner record. */
function acquirePrepareLock(file: string, candidateId: string): number {
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const handle = fs.openSync(file, 'wx', 0o600);
      try { fs.writeFileSync(handle, JSON.stringify({ pid: process.pid, candidateId })); }
      catch (e) { fs.closeSync(handle); fs.unlinkSync(file); throw e; }
      return handle;
    } catch (e: any) { if (e.code !== 'EEXIST') throw e; }
    if (attempt) refuse('Another updater holds the preparation lock');
    let before: fs.Stats;
    try { before = fs.lstatSync(file); }
    catch (e: any) { if (e.code === 'ENOENT') continue; throw e; }
    if (!before.isFile() || before.isSymbolicLink()) refuse('Updater lock is not an ordinary file');
    let pid: unknown;
    try { if (before.size <= 4096) pid = JSON.parse(fs.readFileSync(file, 'utf8')).pid; }
    catch { /* An interrupted owner-record write can leave malformed JSON. */ }
    if (typeof pid === 'number' && Number.isSafeInteger(pid) && pid > 0 && pid <= 0x7fffffff) {
      let dead = false;
      try { process.kill(pid, 0); }
      catch (e: any) { if (e.code === 'ESRCH') dead = true; }
      // EPERM and other probe failures do not prove that the owner is dead.
      if (!dead) refuse('Another updater holds the preparation lock');
    } else if (Date.now() - before.mtimeMs < 30_000) refuse('Another updater is initializing the preparation lock');
    // Recheck identity before removal in case another process already recovered it.
    try {
      const after = fs.lstatSync(file);
      if (after.ino !== before.ino || after.dev !== before.dev || after.mtimeMs !== before.mtimeMs || after.size !== before.size) refuse('Updater lock changed during recovery; retry preparation');
      fs.unlinkSync(file);
    } catch (e: any) { if (e.code !== 'ENOENT') throw e; }
  }
  return refuse('Another updater holds the preparation lock');
}

async function sweepIncompleteStages(state: string): Promise<void> {
  let ready = false;
  for (const child of await fsp.readdir(state, { withFileTypes: true })) {
    if (!/^prepare-[a-zA-Z0-9]{6}$/.test(child.name) || !child.isDirectory() || child.isSymbolicLink()) continue;
    const folder = path.resolve(state, child.name);
    if (path.dirname(folder) !== state) refuse('Refusing cleanup outside updater staging');
    noLinks(folder);
    try {
      await fsp.lstat(path.join(folder, 'ready.json'));
      ready = true;
      continue;
    } catch (e: any) { if (e.code !== 'ENOENT') throw e; }
    await fsp.rm(folder, { recursive: true, force: true });
  }
  // The launcher owns every ready receipt, even after a server restart. Never
  // delete one or stage a second copy while it may be consumed for activation.
  if (ready) refuse('The launcher must reconcile an existing ready update before preparing another');
}

/** Size and SHA-256 of every staged file, keyed by its '/' path in the archive. */
type StagedFiles = Record<string, [number, string]>;

async function extract(file: string, target: string, entries: ZipEntry[], signal: AbortSignal): Promise<StagedFiles> {
  await fsp.mkdir(target, { mode: 0o700 });
  const staged: StagedFiles = {};
  for (const e of entries) {
    signal.throwIfAborted();
    const dest = path.join(target, e.name);
    if (!within(target, dest)) refuse('Archive path escapes staging');
    await fsp.mkdir(e.directory ? dest : path.dirname(dest), { recursive: true, mode: 0o700 });
    noLinks(e.directory ? dest : path.dirname(dest));
    if (e.directory) continue;
    let count = 0, crc = 0xffffffff;
    const sha = createHash('sha256');
    const guard = new Transform({ transform(chunk: Buffer, _enc, cb) {
      count += chunk.length;
      if (count > e.size) { cb(new Error('ZIP entry exceeds its expanded size')); return; }
      crc = crcChunk(crc, chunk); sha.update(chunk); cb(null, chunk);
    } });
    const input = e.packed ? fs.createReadStream(file, { start: e.start, end: e.start + e.packed - 1 }) : Readable.from([]);
    const output = fs.createWriteStream(dest, { flags: 'wx', mode: 0o600 });
    if (e.method === 8) await pipeline(input, createInflateRaw(), guard, output, { signal });
    else await pipeline(input, guard, output, { signal });
    if (count !== e.size || ((crc ^ 0xffffffff) >>> 0) !== e.crc) refuse('ZIP entry is truncated or has a wrong checksum');
    staged[e.name] = [count, sha.digest('hex')];
  }
  return staged;
}

/** The staged folder still holds exactly the files extraction recorded, by
 *  name and size; with `hashes`, also by SHA-256 read again from disk. */
function checkStagedFiles(root: string, files: StagedFiles): string[] {
  const rels: string[] = [];
  for (const e of fs.readdirSync(root, { recursive: true, withFileTypes: true }) as fs.Dirent[]) {
    if (e.isDirectory()) continue;
    const full = path.join(e.parentPath, e.name);
    const rel = path.relative(root, full).split(path.sep).join('/');
    const want = Object.hasOwn(files, rel) ? files[rel] : null;
    if (!want || e.isSymbolicLink() || !e.isFile()) refuse(`The staged update has an unexpected file (${rel}); download it again`);
    if (fs.statSync(full).size !== want![0]) refuse(`The staged update changed on disk (${rel}); download it again`);
    rels.push(rel);
  }
  if (rels.length !== Object.keys(files).length) refuse('The staged update is missing files; download it again');
  return rels;
}
async function checkStaged(root: string, files: StagedFiles): Promise<void> {
  for (const rel of checkStagedFiles(root, files)) {
    const h = createHash('sha256');
    for await (const chunk of fs.createReadStream(path.join(root, ...rel.split('/')))) h.update(chunk as Buffer);
    if (h.digest('hex') !== files[rel][1]) refuse(`The staged update changed on disk (${rel}); download it again`);
  }
}

/** Mirrors release/check-layout.mjs checkPackageBuildInfo for the versioned
 *  layout: release/*.mjs is not shipped in the server bundle. */
function checkLayout(root: string, c: Candidate): { info: BuildInfo; payloadId: string } {
  let pointer: any;
  try { pointer = JSON.parse(fs.readFileSync(path.join(root, 'current.json'), 'utf8')); } catch { refuse('The package has no readable current.json'); }
  const id = pointer?.payload;
  if (pointer?.format !== 1 || typeof id !== 'string' || !PAYLOAD_ID.test(id)) refuse('The package current.json is invalid');
  const apps = fs.readdirSync(path.join(root, 'app'));
  if (apps.length !== 1 || apps[0] !== id) refuse('The package must hold exactly the version current.json names');
  const payload = path.join(root, 'app', id);
  const read = (name: string) => fs.readFileSync(path.join(payload, name), 'utf8').trim();
  const raw = read('build-info.json');
  if (raw.length > 16_384) refuse('Embedded build identity is too large');
  const info = parseBuildInfo(raw).info;
  if (!info || !validIdentity(info) || info.tag !== c.tag || info.sha !== c.sha || info.os !== 'windows' || info.arch !== 'x64' || info.variant !== c.variant
    || info.updaterProtocol < 1 || info.updaterProtocol !== c.protocol) refuse('Embedded build-info.json does not match the candidate or updater protocol');
  if (id !== `${info!.tag}-${info!.sha.slice(0, 12)}-${info!.variant}`) refuse('The payload folder is named for another build');
  if (read('VERSION').replace(/^v/, '') !== c.tag.slice(1)) refuse('VERSION disagrees with the candidate');
  const inPayload = ['runtime/node.exe', 'server/server.mjs', 'server/updater-tool.mjs', 'server/node_modules/better-sqlite3/build/Release/better_sqlite3.node',
    'ui/dist/index.html', '.env.example', 'payload-manifest.json',
    ...['ace-server', 'ace-train', 'ace-midi', 'ace-caption', 'mastering', 'mp3-codec', 'neural-codec', 'vst-host', 'quantize'].map(n => `engine/${n}.exe`)];
  for (const name of ['HOT-Step.bat', 'launcher.mjs', 'runtime/node.exe', ...inPayload.map(n => `app/${id}/${n}`)]) {
    const st = fs.lstatSync(path.join(root, name), { throwIfNoEntry: false });
    if (!st || !st.isFile() || st.isSymbolicLink() || st.size === 0) refuse(`Required runtime file is missing or invalid: ${name}`);
  }
  const backend = c.variant.startsWith('cuda') ? 'cuda' : c.variant === 'vulkan' ? 'vulkan' : null;
  const libs = fs.readdirSync(path.join(payload, 'engine')).map(n => n.match(/^ggml-(cuda|vulkan|hip)\.dll$/)?.[1]).filter(Boolean);
  if ((backend && !libs.includes(backend)) || libs.some(l => l !== backend)) refuse('Packaged backend libraries disagree with the candidate variant');
  for (const [name, expected] of [['.variant', backend === 'cuda' ? 'cuda' : c.variant], ['.cuda-version', backend === 'cuda' ? c.variant.match(/^cuda(\d+)/)![1] : null]]) {
    const marker = path.join(payload, 'engine', name!);
    if (fs.existsSync(marker) && fs.readFileSync(marker, 'utf8').trim() !== expected) refuse('Packaged engine markers disagree with the candidate variant');
  }
  return { info: info!, payloadId: id };
}

export function createDownloader(d: DownloadDeps) {
  const limits = { ...DOWNLOAD_LIMITS, ...d.limits };
  let candidate: Candidate | null = null;
  let reason: string | null = 'Check for a newer stable release first';
  let progress: Omit<UpdatePrepareStatus, 'offered' | 'reason' | 'candidate'> = { phase: 'idle', downloadedBytes: 0, totalBytes: 0, error: null, preparedId: null };
  let discovering: Promise<void> | null = null;
  let running: Promise<void> | null = null;
  let controller: AbortController | null = null;
  /** payloadRoot is the staged package root; the launcher installs payloadRoot/app/<payloadId>. */
  let prepared: { id: string; payloadRoot: string; payloadId: string; buildInfo: BuildInfo; candidate: UpdateCandidate } | null = null;
  /** What extraction wrote, from the verified archive (ready.json after a restart). */
  let stagedFiles: StagedFiles | null = null;
  const publicCandidate = (c: Candidate): UpdateCandidate => ({ id: c.id, tag: c.tag, sha: c.sha, variant: c.variant, assetName: c.assetName, bytes: c.bytes });
  function status(): UpdatePrepareStatus {
    const blocker = prepareBlocker(d);
    const expired = !!candidate && d.now() > candidate.expiresAt;
    const why = blocker ?? (running ? 'An update is already being prepared' : discovering ? 'Checking release assets' : expired ? 'Candidate expired; check for updates again' : reason);
    return { ...progress, offered: !why && !!candidate && progress.phase !== 'ready', reason: why ?? (progress.phase === 'ready' ? 'The verified update is ready for the launcher' : null), candidate: blocker || expired || !candidate ? null : publicCandidate(candidate) };
  }
  async function discover(): Promise<void> {
    if (running || progress.phase === 'ready') return;
    if (discovering) return discovering;
    const blocker = prepareBlocker(d);
    if (blocker) { candidate = null; reason = blocker; return; }
    discovering = (async () => {
      const previousPhase = progress.phase;
      progress.phase = 'checking';
      try {
        const next = await releaseCandidate(d, limits, AbortSignal.timeout(30_000));
        const same = candidate && JSON.stringify({ ...candidate, id: '', expiresAt: 0 }) === JSON.stringify({ ...next, id: '', expiresAt: 0 });
        candidate = { ...next, id: same ? candidate!.id : randomUUID(), expiresAt: d.now() + EXPIRES_MS };
        reason = null;
      } catch (e) { candidate = null; reason = (e as Error).message; }
      finally { progress.phase = previousPhase; }
    })().finally(() => { discovering = null; });
    return discovering;
  }
  async function free(directory: string, need: number): Promise<void> {
    const available = d.freeBytes ? await d.freeBytes(directory) : await fsp.statfs(directory).then(s => Number(s.bavail) * Number(s.bsize));
    if (!Number.isFinite(available) || available < need + limits.reserveBytes) refuse('Not enough free space to stage and verify this update');
  }
  async function work(c: Candidate, signal: AbortSignal): Promise<void> {
    let folder: string | null = null;
    let lock: number | null = null;
    const state = path.resolve(d.stateRoot!);
    try {
      noLinks(state);
      await fsp.mkdir(state, { recursive: true, mode: 0o700 });
      noLinks(state);
      lock = acquirePrepareLock(path.join(state, 'prepare.lock'), c.id);
      const blocker = prepareBlocker(d); if (blocker) refuse(blocker);
      await sweepIncompleteStages(state);
      await free(state, c.bytes + limits.expandedBytes);
      const fresh = await releaseCandidate(d, limits, signal);
      if (JSON.stringify(fresh) !== JSON.stringify({ tag: c.tag, sha: c.sha, variant: c.variant, assetName: c.assetName, bytes: c.bytes, asset: c.asset, sidecar: c.sidecar, protocol: c.protocol })) refuse('Release candidate changed; check for updates again');
      const side = await boundedBody(await request(d, c.sidecar.browser_download_url, signal), 4096, signal);
      if (side.length !== c.sidecar.size || createHash('sha256').update(side).digest('hex') !== DIGEST.exec(c.sidecar.digest)![1]) refuse('Checksum sidecar size or API digest mismatch');
      const line = side.toString('utf8').replace(/^\uFEFF/, '').trim();
      const match = /^([0-9a-fA-F]{64}) [ *](\S+)$/.exec(line);
      if (!match || match[2] !== c.assetName || match[1].toLowerCase() !== DIGEST.exec(c.asset.digest)![1]) refuse('Missing or mismatched SHA-256 sidecar');
      folder = await fsp.mkdtemp(path.join(state, 'prepare-'));
      const part = path.join(folder, 'archive.zip.part'), archive = path.join(folder, 'archive.zip');
      progress.phase = 'downloading';
      const res = await request(d, c.asset.browser_download_url, signal);
      const length = res.headers.get('content-length');
      if (length !== null && Number(length) !== c.bytes) { await res.body?.cancel(); refuse('Download content length does not match the release asset'); }
      if (!res.body) refuse('Empty release download');
      const hash = createHash('sha256');
      const guard = new Transform({ transform(chunk: Buffer, _enc, cb) {
        progress.downloadedBytes += chunk.length;
        if (progress.downloadedBytes > c.bytes || progress.downloadedBytes > limits.archiveBytes) { cb(new Error('Download exceeds the byte limit')); return; }
        hash.update(chunk); cb(null, chunk);
      } });
      await pipeline(Readable.fromWeb(res.body! as any), guard, fs.createWriteStream(part, { flags: 'wx', mode: 0o600 }), { signal });
      progress.phase = 'verifying';
      if (progress.downloadedBytes !== c.bytes || hash.digest('hex') !== match![1].toLowerCase()) refuse('Download is truncated or has a wrong SHA-256');
      await fsp.rename(part, archive);
      const entries = await zipEntries(archive, limits, signal);
      await free(state, entries.reduce((sum, e) => sum + e.size, 0));
      progress.phase = 'extracting';
      const payloadRoot = path.join(folder, 'payload');
      const files = await extract(archive, payloadRoot, entries, signal);
      signal.throwIfAborted();
      const { info: buildInfo, payloadId } = checkLayout(payloadRoot, c);
      const id = path.basename(folder);
      prepared = { id, payloadRoot, payloadId, buildInfo, candidate: publicCandidate(c) };
      stagedFiles = files;
      await fsp.writeFile(path.join(folder, 'ready.json'), JSON.stringify({ ...prepared, archiveSha256: match![1].toLowerCase(), files }), { flag: 'wx', mode: 0o600 });
      signal.throwIfAborted();
      progress.phase = 'ready'; progress.preparedId = id;
    } catch (e) {
      prepared = null;
      progress.phase = signal.aborted ? 'cancelled' : 'failed';
      progress.error = signal.aborted ? 'Update preparation was cancelled or timed out' : (e as Error).message;
      if (folder) {
        // Only this invocation's randomly created child can be removed.
        const resolved = path.resolve(folder);
        if (path.dirname(resolved) !== state || !path.basename(resolved).startsWith('prepare-')) throw new Error('Refusing cleanup outside updater staging');
        try { await fsp.rm(resolved, { recursive: true, force: true }); }
        catch { progress.error += '; staging cleanup failed; no update was activated'; }
      }
    } finally {
      if (lock !== null) {
        try {
          fs.closeSync(lock);
          await fsp.unlink(path.join(state, 'prepare.lock'));
        } catch {
          prepared = null;
          progress.preparedId = null;
          progress.phase = 'failed';
          progress.error = 'Could not release the updater lock; no update was activated';
          if (folder && path.dirname(path.resolve(folder)) === state && path.basename(folder).startsWith('prepare-')) {
            try { await fsp.rm(folder, { recursive: true, force: true }); }
            catch { progress.error += '; staging cleanup failed'; }
          }
        }
      }
    }
  }
  function prepare(candidateId: string): UpdatePrepareStatus {
    const blocker = prepareBlocker(d); if (blocker) refuse(blocker);
    if (running || discovering) refuse('An update check or preparation is already running');
    if (progress.phase === 'ready') refuse('A verified update is already staged');
    if (!candidate || candidate.id !== candidateId || d.now() > candidate.expiresAt) refuse('Unknown or expired candidate; check for updates again');
    const selected = candidate!;
    controller = new AbortController();
    const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(30 * 60_000)]);
    progress = { phase: 'verifying', downloadedBytes: 0, totalBytes: selected.bytes, error: null, preparedId: null };
    running = work(selected, signal).catch(e => { progress.phase = 'failed'; progress.error = (e as Error).message; }).finally(() => { running = null; controller = null; });
    return status();
  }
  function cancel(candidateId: string): UpdatePrepareStatus {
    if (!candidate || candidate.id !== candidateId || !running || !controller) refuse('No matching update preparation is running');
    controller!.abort();
    return status();
  }
  /** A ready.json survives a restart only on disk. Take back a valid receipt for
   *  a newer release; delete one that is stale or no longer checks out (staging
   *  is updater-owned, and nothing installs while this server runs). */
  function restoreReceipt(): void {
    if (!d.stateRoot || prepareBlocker(d)) return;
    const state = path.resolve(d.stateRoot);
    let names: string[];
    try { names = fs.readdirSync(state); } catch { return; }
    for (const name of names.filter(n => /^prepare-[a-zA-Z0-9]{6}$/.test(n))) {
      const folder = path.join(state, name);
      const receipt = path.join(folder, 'ready.json');
      if (!fs.existsSync(receipt)) continue;
      try {
        noLinks(folder);
        const r = JSON.parse(fs.readFileSync(receipt, 'utf8'));
        const installed = d.install().buildInfo!;
        if (prepared || r?.id !== name || r.payloadRoot !== path.join(folder, 'payload') || typeof r.candidate?.tag !== 'string'
          || !newer(r.candidate.tag, installed.tag!)) throw new Error('stale receipt');
        const c = { ...r.candidate, protocol: installed.updaterProtocol, expiresAt: Infinity } as Candidate;
        const files = r.files as StagedFiles;
        if (!files || typeof files !== 'object' || Object.values(files).some(v => !Array.isArray(v) || !Number.isSafeInteger(v[0]) || !/^[0-9a-f]{64}$/.test(v[1]))) throw new Error('no file record');
        checkStagedFiles(r.payloadRoot, files); // names and sizes now; hashes again before handoff
        const { info, payloadId } = checkLayout(r.payloadRoot, c);
        stagedFiles = files;
        prepared = { id: name, payloadRoot: r.payloadRoot, payloadId, buildInfo: info, candidate: publicCandidate(c) };
        candidate = c;
        reason = null;
        progress = { phase: 'ready', downloadedBytes: c.bytes, totalBytes: c.bytes, error: null, preparedId: name };
      } catch {
        try { fs.rmSync(folder, { recursive: true, force: true }); } catch { /* the launcher removes it later */ }
      }
    }
  }
  /** The staged update, checked again just before it is handed to the launcher:
   *  every file against the SHA-256 recorded while extracting the verified archive. */
  async function revalidate(): Promise<NonNullable<typeof prepared>> {
    const blocker = prepareBlocker(d); if (blocker) refuse(blocker);
    if (!prepared || !candidate || !stagedFiles || progress.phase !== 'ready') refuse('No verified update is staged; download it first');
    if (!fs.existsSync(path.join(path.dirname(prepared!.payloadRoot), 'ready.json'))) refuse('The staged update is gone; download it again');
    await checkStaged(prepared!.payloadRoot, stagedFiles!);
    const { info, payloadId } = checkLayout(prepared!.payloadRoot, candidate!);
    if (payloadId !== prepared!.payloadId || info.sha !== prepared!.buildInfo.sha) refuse('The staged update changed on disk; download it again');
    return structuredClone(prepared!);
  }
  restoreReceipt();
  return { status, discover, prepare, cancel, revalidate, wait: async () => { await running; }, getPrepared: () => prepared ? structuredClone(prepared) : null };
}
