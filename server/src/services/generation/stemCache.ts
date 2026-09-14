// generation/stemCache.ts — keep the vocal/instrumental split between PP runs
//
// The split that StableStep needs costs ~22 s of GPU on a 3.5-minute track —
// about half of a whole post-processing pass. It is also the same work every
// time: the chain always separates the RAW render, before any stage has
// touched it, so two runs over one track feed the separator byte-identical
// audio and get byte-identical stems back.
//
// That matters because of how post-processing is actually used: run the
// chain, listen, remove the result, change a setting, run it again. Every
// iteration of that loop paid the separation again for nothing.
//
// The cache is keyed by the hash of the audio handed to the separator plus
// the separation level, so it cannot serve stems from different audio or from
// a different model pass — an edited or re-rendered track simply misses.
// Entries are large (an f32 stereo stem is ~350 kB per second, and there are
// two), so the directory is capped and the oldest entries go first.

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { config } from '../../config.js';

/** Bump to invalidate every entry — the stored shape changed, or the stems a
 *  given level produces did. */
const CACHE_VERSION = 1;

/** How much disk the cache may hold before the oldest entries are dropped.
 *  Stems are f32, so a stereo pair costs ~42 MB per minute of audio — about
 *  145 MB for a 3.5-minute track, and the default keeps roughly 25 of them.
 *  PP_STEM_CACHE_GB=0 turns the cache off entirely. */
const MAX_BYTES = (() => {
  const gb = parseFloat(process.env.PP_STEM_CACHE_GB || '4');
  return Number.isFinite(gb) && gb > 0 ? gb * 1024 * 1024 * 1024 : 0;
})();

const ENABLED = MAX_BYTES > 0;

/** What a cached entry records besides the two stem files. `sepId` is the
 *  engine-side job the stems came from: still useful while the engine has it,
 *  and known to be stale otherwise (see readStemCache). */
interface StemCacheMeta {
  version: number;
  level: number;
  sepId: string;
  vocalIndex: number;
  stems: Array<{ index: number; category: string; hidden: boolean }>;
  vocalBytes: number;
  instBytes: number;
  /** A split that found no vocal to separate. Worth remembering: finding that
   *  out costs a full separation. */
  empty?: boolean;
  createdAt: string;
}

export interface CachedSplit {
  sepId: string;
  stems: Array<{ index: number; category: string; hidden: boolean }>;
  vocalIndex: number;
  vocalBuf: Buffer;
  instBuf: Buffer;
}

function cacheDir(): string {
  const dir = path.join(config.data.dir, 'stem-cache');
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

/**
 * The key for one split: what went in, and how it was split.
 *
 * Hashing the audio itself rather than the file it came from is what makes
 * this safe — the chain works on a fresh copy with a new path and mtime every
 * run, so anything path-shaped would miss every time, and anything
 * timestamp-shaped would be wrong.
 */
export function stemCacheKey(srcBuf: Buffer, level: number): string {
  const hash = crypto.createHash('sha1').update(srcBuf).digest('hex');
  return `v${CACHE_VERSION}-l${level}-${hash}`;
}

function entryPaths(key: string) {
  const dir = cacheDir();
  return {
    meta: path.join(dir, `${key}.json`),
    vocal: path.join(dir, `${key}.vocal.wav`),
    inst: path.join(dir, `${key}.inst.wav`),
  };
}

/**
 * The stems for this key, or null when there are none to serve.
 *
 * Returns `{ empty: true }` for a remembered "there was no vocal here", which
 * the caller must treat as the separation returning null rather than as a
 * miss.
 */
export function readStemCache(key: string): { empty: true } | CachedSplit | null {
  if (!ENABLED) return null;
  const p = entryPaths(key);
  let meta: StemCacheMeta;
  try {
    meta = JSON.parse(fs.readFileSync(p.meta, 'utf8')) as StemCacheMeta;
  } catch {
    return null;
  }
  if (meta.version !== CACHE_VERSION) return null;
  if (meta.empty) {
    touch(p.meta);
    return { empty: true };
  }

  try {
    const vocalBuf = fs.readFileSync(p.vocal);
    const instBuf = fs.readFileSync(p.inst);
    // A truncated stem — a crash mid-write, or a prune that caught one file of
    // the pair — must not reach the refine as if it were audio.
    if (vocalBuf.length !== meta.vocalBytes || instBuf.length !== meta.instBytes) return null;
    touch(p.meta);
    return { sepId: meta.sepId, stems: meta.stems, vocalIndex: meta.vocalIndex, vocalBuf, instBuf };
  } catch {
    return null;
  }
}

/** Store a split. Never throws — a cache that cannot write is a slow cache,
 *  not a broken pass. */
export function writeStemCache(key: string, split: CachedSplit | null, level: number): void {
  if (!ENABLED) return;
  const p = entryPaths(key);
  try {
    if (!split) {
      const meta: StemCacheMeta = {
        version: CACHE_VERSION, level, sepId: '', vocalIndex: -1, stems: [],
        vocalBytes: 0, instBytes: 0, empty: true, createdAt: new Date().toISOString(),
      };
      fs.writeFileSync(p.meta, JSON.stringify(meta));
      return;
    }

    // Stems first, metadata last: the metadata file is what a reader trusts,
    // so it must not exist before the audio it describes.
    fs.writeFileSync(p.vocal + '.tmp', split.vocalBuf);
    fs.renameSync(p.vocal + '.tmp', p.vocal);
    fs.writeFileSync(p.inst + '.tmp', split.instBuf);
    fs.renameSync(p.inst + '.tmp', p.inst);

    const meta: StemCacheMeta = {
      version: CACHE_VERSION,
      level,
      sepId: split.sepId,
      vocalIndex: split.vocalIndex,
      stems: split.stems,
      vocalBytes: split.vocalBuf.length,
      instBytes: split.instBuf.length,
      createdAt: new Date().toISOString(),
    };
    fs.writeFileSync(p.meta, JSON.stringify(meta));
    prune();
  } catch (err: any) {
    console.warn(`[StemCache] Could not store ${key}: ${err?.message || err}`);
  }
}

/** Mark an entry as just used, so pruning drops what nobody is coming back to. */
function touch(metaPath: string): void {
  try {
    const now = new Date();
    fs.utimesSync(metaPath, now, now);
  } catch { /* a cache hit is worth more than an accurate timestamp */ }
}

/** Drop whole entries, oldest first, until the directory is under the cap. */
function prune(): void {
  const dir = cacheDir();
  let names: string[];
  try { names = fs.readdirSync(dir); } catch { return; }

  const entries = new Map<string, { bytes: number; mtimeMs: number }>();
  for (const name of names) {
    const key = name.replace(/\.(json|vocal\.wav|inst\.wav)$/, '');
    if (key === name) continue; // not ours (a .tmp mid-write, say)
    let stat: fs.Stats;
    try { stat = fs.statSync(path.join(dir, name)); } catch { continue; }
    const acc = entries.get(key) || { bytes: 0, mtimeMs: 0 };
    acc.bytes += stat.size;
    // The metadata file carries the access time; the stems are never touched
    // after they are written.
    if (name.endsWith('.json')) acc.mtimeMs = stat.mtimeMs;
    entries.set(key, acc);
  }

  let total = 0;
  for (const e of entries.values()) total += e.bytes;
  if (total <= MAX_BYTES) return;

  const oldestFirst = [...entries.entries()].sort((a, b) => a[1].mtimeMs - b[1].mtimeMs);
  for (const [key, info] of oldestFirst) {
    if (total <= MAX_BYTES) break;
    const p = entryPaths(key);
    // Metadata first: an entry with no metadata is a miss, whereas stems with
    // no metadata are just bytes waiting to be deleted.
    for (const f of [p.meta, p.vocal, p.inst]) {
      try { if (fs.existsSync(f)) fs.unlinkSync(f); } catch { /* locked — next prune gets it */ }
    }
    total -= info.bytes;
    console.log(`[StemCache] Pruned ${key} (${(info.bytes / 1048576).toFixed(0)} MB)`);
  }
}
