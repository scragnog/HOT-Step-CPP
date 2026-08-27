// minimax/plank.ts — MM3 Plank: the AR-stage code cache.
//
// Captures the AR (LM) stage's output codes so a later render can replay them
// instead of sampling fresh ones.
//
// ── WHAT THIS IS FOR, AND WHAT IT IS NOT ────────────────────────────────────
//
// Replay does NOT make a render faster. The engine still executes every
// per-frame forward pass of the AR loop — the planning progress bar still
// crawls to 100%. Forcing pins WHICH codes come out, not how much compute runs.
//
// The value is a FIXED SEMANTIC BED: with the AR output held constant, an A/B
// of two flow-stage solver/scheduler/guidance settings compares those settings
// rather than two different AR samplings. Sold as a speed feature it
// disappoints; used as a reproducibility feature it does exactly what it says.
//
// (A real AR speedup needs the condition-encoder input HIDDENS captured rather
// than the codes. That is ./hiddens.ts, and it is built — but the blob is 4096x
// the size, so the two coexist: a plan for pinning a bed you are about to
// iterate against, a plank for anything you want to keep, share or inspect.)
//
// ── ON-DISK FORMAT ──────────────────────────────────────────────────────────
//
// <uuid>.mm3plank      little-endian i32 blob, written by engine mm3-job.h:
//                        [i32] n_semantic, [i32 * n_semantic] semantic
//                        [i32] n_acoustic, [i32 * n_acoustic] acoustic
//                      Iteration-indexed; entry 0 is the un-emitted iteration,
//                      so I codes render I-1 frames (engine mm3-ar-loop.h).
// <uuid>.mm3plank.json human-readable sidecar for the picker. Optional: a
//                      plank with no sidecar is still replayable.
//
// This module is deliberately free of engine/db/generation imports so the
// backend capability manifest can list planks without pulling the whole
// generation pipeline in at module-load time.

import fs from 'fs';
import path from 'path';

import { config } from '../../../config.js';

export const MM3_PLANK_EXT = '.mm3plank';

export interface Mm3PlankMeta {
  id: string;
  jobId: string;
  created: string;
  caption: string;
  lyrics: string;
  duration: number;
  seed: number;
  frames: number;
  sizeBytes: number;
}

/** Sibling of the audio directory, so planks travel with a data-dir move. */
export function mm3PlankDir(): string {
  return path.join(path.dirname(config.data.audioDir), 'mm3-planks');
}

/** Resolve a plank reference to an absolute path INSIDE the plank directory.
 *  A bare filename is the normal case — that is what the picker sends. This
 *  value reaches the server from the browser, so anything that escapes the
 *  plank directory returns null rather than being read. */
export function resolveMm3PlankPath(ref: string): string | null {
  if (!ref) return null;
  const dir = mm3PlankDir();
  const resolved = path.resolve(dir, ref);
  const rel = path.relative(dir, resolved);
  if (rel.startsWith('..') || path.isAbsolute(rel)) return null;
  return resolved;
}

/** One plank's sidecar, or null when it is missing/corrupt. */
export function readMm3PlankMeta(ref: string): Partial<Mm3PlankMeta> | null {
  const resolved = resolveMm3PlankPath(ref);
  if (!resolved) return null;
  const sidecar = resolved.endsWith('.json') ? resolved : `${resolved}.json`;
  try {
    return JSON.parse(fs.readFileSync(sidecar, 'utf-8'));
  } catch {
    return null;
  }
}

/** Saved planks, newest first. Never throws — a missing directory is []. */
export function listMm3Planks(): Array<{ file: string } & Partial<Mm3PlankMeta>> {
  try {
    const dir = mm3PlankDir();
    if (!fs.existsSync(dir)) return [];
    return fs.readdirSync(dir)
      .filter(f => f.endsWith(MM3_PLANK_EXT))
      // A plank with no readable sidecar is still replayable (the codes are in
      // the blob), so surface it with just its filename rather than hiding it.
      .map(file => ({ file, ...(readMm3PlankMeta(file) ?? {}) }))
      .sort((a, b) => String(b.created ?? '').localeCompare(String(a.created ?? '')));
  } catch {
    return [];
  }
}

/** Deserialize a .mm3plank blob into the engine's forced_* arrays. Returns null
 *  on any problem: a missing or corrupt plank downgrades the render to a normal
 *  AR pass rather than failing the generation outright. */
export function readMm3Plank(ref: string): { forced_semantic: number[]; forced_acoustic: number[] } | null {
  const plankPath = resolveMm3PlankPath(ref);
  if (!plankPath) {
    console.error(`[MM3 Plank] Rejected plank reference outside the plank directory: ${ref}`);
    return null;
  }
  try {
    const buf = fs.readFileSync(plankPath);
    let offset = 0;
    const readArray = (): number[] => {
      if (offset + 4 > buf.length) throw new Error('truncated before a length prefix');
      const n = buf.readInt32LE(offset); offset += 4;
      if (n < 0 || offset + n * 4 > buf.length) throw new Error('length prefix overruns the blob');
      const out = new Array<number>(n);
      for (let i = 0; i < n; i++) { out[i] = buf.readInt32LE(offset); offset += 4; }
      return out;
    };
    const forced_semantic = readArray();
    const forced_acoustic = readArray();
    // Mirrors the engine's own checks (mm3-request.h) so a bad plank is caught
    // here, with a filename to blame, rather than as a 400 mid-generation.
    if (forced_semantic.length < 2) throw new Error('needs at least 2 iterations');
    if (forced_acoustic.length !== forced_semantic.length * 7) {
      throw new Error(`acoustic ${forced_acoustic.length} != semantic ${forced_semantic.length} * 7`);
    }
    return { forced_semantic, forced_acoustic };
  } catch (err: any) {
    console.error(`[MM3 Plank] Failed to read ${plankPath}: ${err?.message ?? err}`);
    return null;
  }
}
