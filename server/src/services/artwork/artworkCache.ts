// artworkCache.ts — fetches remote artwork for GET /api/artwork and keeps it on disk,
// so the UI's img-src can stay 'self'. Outbound fetch rules: https only, no private,
// loopback or link-local address (checked on the address actually connected to, so a
// DNS answer cannot change between check and connect), redirects re-checked hop by
// hop, image types only (no SVG), size and time capped. The cache lives in the app
// data folder and prunes oldest-first past MAX_CACHE_BYTES.

import crypto from 'node:crypto';
import dns from 'node:dns';
import fs from 'node:fs';
import https from 'node:https';
import net from 'node:net';
import path from 'node:path';
import { config } from '../../config.js';

export const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
export const MAX_CACHE_BYTES = 256 * 1024 * 1024;
const MAX_REDIRECTS = 3;
const TIMEOUT_MS = 10_000;

/** Genius search previews: any URL on these hosts, stored or not. */
const PREVIEW_HOST_SUFFIXES = ['.genius.com', '.rapgenius.com'];

const EXT_BY_TYPE: Record<string, string> = {
  'image/jpeg': 'jpg', 'image/png': 'png', 'image/webp': 'webp', 'image/gif': 'gif', 'image/avif': 'avif',
};
const TYPE_BY_EXT = Object.fromEntries(Object.entries(EXT_BY_TYPE).map(([t, e]) => [e, t]));

export class ArtworkError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

const blocked = new net.BlockList();
for (const [a, bits] of [
  ['0.0.0.0', 8], ['10.0.0.0', 8], ['100.64.0.0', 10], ['127.0.0.0', 8], ['169.254.0.0', 16],
  ['172.16.0.0', 12], ['192.0.0.0', 24], ['192.168.0.0', 16], ['198.18.0.0', 15], ['224.0.0.0', 3],
] as const) blocked.addSubnet(a, bits, 'ipv4');
for (const [a, bits] of [['::', 128], ['::1', 128], ['fc00::', 7], ['fe80::', 10], ['ff00::', 8]] as const) {
  blocked.addSubnet(a, bits, 'ipv6');
}

export function isBlockedAddress(ip: string): boolean {
  const mapped = /^::ffff:(\d+\.\d+\.\d+\.\d+)$/i.exec(ip);
  if (mapped) return blocked.check(mapped[1], 'ipv4');
  const family = net.isIP(ip);
  if (!family) return true;
  return blocked.check(ip, family === 6 ? 'ipv6' : 'ipv4');
}

/** Parses and screens a URL before any connection; throws ArtworkError. */
export function parseArtworkUrl(raw: string): URL {
  let u: URL;
  try { u = new URL(raw); } catch { throw new ArtworkError(400, 'Invalid URL'); }
  if (u.protocol !== 'https:') throw new ArtworkError(400, 'Only https URLs');
  if (u.username || u.password) throw new ArtworkError(400, 'Credentials in URL');
  const host = u.hostname.replace(/^\[|\]$/g, '');
  if (net.isIP(host) && isBlockedAddress(host)) throw new ArtworkError(403, 'Address not allowed');
  return u;
}

export function isPreviewHost(u: URL): boolean {
  return PREVIEW_HOST_SUFFIXES.some(s => u.hostname.toLowerCase().endsWith(s));
}

// Resolves and screens every address; the connection then uses the one we checked.
const safeLookup = ((host: string, opts: dns.LookupOptions, cb: (...a: any[]) => void) => {
  dns.lookup(host, { all: true }, (err, addrs) => {
    if (err) return cb(err);
    if (!addrs.length || addrs.some(a => isBlockedAddress(a.address))) return cb(new Error('Address not allowed'));
    if (opts.all) return cb(null, addrs);
    cb(null, addrs[0].address, addrs[0].family);
  });
}) as net.LookupFunction;

function cacheDir(): string { return path.join(config.data.dir, 'cache', 'artwork'); }
function keyOf(url: string): string { return crypto.createHash('sha256').update(url).digest('hex'); }

export function readCached(url: string, dir = cacheDir()): { file: string; type: string } | null {
  const key = keyOf(url);
  for (const [ext, type] of Object.entries(TYPE_BY_EXT)) {
    const file = path.join(dir, `${key}.${ext}`);
    if (fs.existsSync(file)) return { file, type };
  }
  return null;
}

export function pruneCache(dir = cacheDir(), max = MAX_CACHE_BYTES): void {
  let files: { f: string; size: number; t: number }[];
  try {
    files = fs.readdirSync(dir).map(n => path.join(dir, n)).map(f => {
      const s = fs.statSync(f);
      return { f, size: s.size, t: s.mtimeMs };
    });
  } catch { return; }
  let total = files.reduce((n, x) => n + x.size, 0);
  for (const x of files.sort((a, b) => a.t - b.t)) {
    if (total <= max) break;
    try { fs.unlinkSync(x.f); total -= x.size; } catch { /* in use or gone */ }
  }
}

type Hop = { status: number; location?: string; type: string; body: Buffer };

function fetchOnce(u: URL, lookup: net.LookupFunction, signal: AbortSignal): Promise<Hop> {
  return new Promise((resolve, reject) => {
    const req = https.get(u, { lookup, signal, headers: { 'User-Agent': 'HOT-Step artwork cache', Accept: 'image/*' } }, res => {
      const status = res.statusCode ?? 0;
      const type = String(res.headers['content-type'] ?? '').split(';')[0].trim().toLowerCase();
      if (status >= 300 && status < 400) { req.destroy(); return resolve({ status, location: res.headers.location, type, body: Buffer.alloc(0) }); }
      if (status !== 200) { req.destroy(); return reject(new ArtworkError(502, `Upstream ${status}`)); }
      if (!EXT_BY_TYPE[type]) { req.destroy(); return reject(new ArtworkError(415, 'Not a supported image type')); }
      const chunks: Buffer[] = [];
      let n = 0;
      res.on('data', (c: Buffer) => {
        n += c.length;
        if (n > MAX_IMAGE_BYTES) { req.destroy(); reject(new ArtworkError(413, 'Image too large')); } else chunks.push(c);
      });
      res.on('end', () => resolve({ status, type, body: Buffer.concat(chunks) }));
      res.on('error', () => reject(new ArtworkError(signal.aborted ? 504 : 502, signal.aborted ? 'Upstream timeout' : 'Upstream read failed')));
    });
    req.on('error', e => reject(e instanceof ArtworkError ? e : signal.aborted ? new ArtworkError(504, 'Upstream timeout') : new ArtworkError(502, e.message)));
  });
}

/** Cached file for a URL, fetching it first when missing. Screens the URL on every hop. */
export async function getArtwork(
  url: string, opts: { dir?: string; lookup?: net.LookupFunction; timeoutMs?: number } = {},
): Promise<{ file: string; type: string }> {
  const dir = opts.dir ?? cacheDir();
  const hit = readCached(url, dir);
  if (hit) return hit;
  let u = parseArtworkUrl(url);
  // One deadline for DNS, every redirect hop and the body reads.
  const signal = AbortSignal.timeout(opts.timeoutMs ?? TIMEOUT_MS);
  for (let hop = 0; ; hop++) {
    const r = await fetchOnce(u, opts.lookup ?? safeLookup, signal);
    if (r.status === 200) {
      fs.mkdirSync(dir, { recursive: true });
      const file = path.join(dir, `${keyOf(url)}.${EXT_BY_TYPE[r.type]}`);
      fs.writeFileSync(file + '.tmp', r.body);
      fs.renameSync(file + '.tmp', file);
      pruneCache(dir);
      return { file, type: r.type };
    }
    if (hop >= MAX_REDIRECTS || !r.location) throw new ArtworkError(502, 'Too many redirects');
    u = parseArtworkUrl(new URL(r.location, u).toString());
  }
}
