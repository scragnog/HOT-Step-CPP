// minimax/hiddens.ts — saved MM3 plans (`.mm3hiddens`).
//
// The AR cache (engine mm3-ar-cache.h) holds one stage-1 plan in host RAM and
// loses it on restart. A saved plan is that slot written to disk, so the same
// bed can be replayed tomorrow, or next to a different flow setting, without
// re-running the planner.
//
// ── HOW THIS DIFFERS FROM A PLANK ───────────────────────────────────────────
//
// A plank pins the AR *codes*; the engine then re-runs every per-frame forward
// pass to turn them back into hiddens, so replay costs the same as planning. A
// saved plan holds the hiddens themselves — what the condition encoder actually
// eats — so replay skips stage 1 outright, and with it the LM load and the
// staged handover.
//
// The trade is size: ~128 KB per frame, ~3 MB per second of audio, ~600 MB for
// a 200 s song. That is 4096x a plank. So:
//
//   saved plan  pin a bed you are about to iterate flow settings against, and
//               get the render time back too. A handful at a time.
//   plank       archive, share, inspect. Small enough to keep thousands, and
//               portable across a model or quant change, which a plan is not.
//
// ── ON-DISK ─────────────────────────────────────────────────────────────────
//
// <name>.mm3hiddens       binary, written and read ENGINE-side by
//                         mm3-hiddens-file.h. Self-describing: magic, shape,
//                         the AR cache key, the stage-1 byproducts, then the
//                         f32 block. Never parsed here — the server only ever
//                         hands the engine a path.
// <name>.mm3hiddens.json  sidecar for the picker (caption, seed, frames, size).
//                         Optional; a plan with no sidecar still replays.
//
// The engine refuses a plan whose saved LM, depth model or LM adapter is not
// the one currently loaded, so a stale file is a note in the log and a fresh
// plan, never a silently wrong render. Nothing on this side has to model that.
//
// Deliberately free of engine/db/generation imports, same as plank.ts, so the
// capability manifest can list plans without dragging the pipeline in.

import fs from 'fs';
import path from 'path';

import { config } from '../../../config.js';

export const MM3_HIDDENS_EXT = '.mm3hiddens';

export interface Mm3HiddensMeta {
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

/** Sibling of the audio directory, so plans travel with a data-dir move. */
export function mm3HiddensDir(): string {
  return path.join(path.dirname(config.data.audioDir), 'mm3-hiddens');
}

/** Create the directory on demand. Returns the path either way — a save that
 *  cannot create it will fail on the write, with a real errno to report. */
export function ensureMm3HiddensDir(): string {
  const dir = mm3HiddensDir();
  try {
    fs.mkdirSync(dir, { recursive: true });
  } catch {
    /* the write below reports it properly */
  }
  return dir;
}

/** Resolve a plan reference to an absolute path INSIDE the plan directory.
 *  A bare filename is the normal case — that is what the picker sends. This
 *  value reaches the server from the browser, so anything that escapes the
 *  directory returns null rather than being opened. Same rule as
 *  resolveMm3PlankPath, and for the same reason. */
export function resolveMm3HiddensPath(ref: string): string | null {
  if (!ref) return null;
  const dir = mm3HiddensDir();
  const resolved = path.resolve(dir, ref);
  const rel = path.relative(dir, resolved);
  if (rel.startsWith('..') || path.isAbsolute(rel)) return null;
  return resolved;
}

/** Turn a user-typed name into a filename that stays inside the plan directory.
 *  Returns null when nothing usable survives, so the caller can fall back to a
 *  uuid rather than writing a file called `.mm3hiddens`. */
export function sanitiseMm3HiddensName(raw: string): string | null {
  const cleaned = String(raw ?? '')
    .trim()
    // Strip the separators and the Windows-reserved set outright rather than
    // escaping them: this is a label, not a path, and `..` has no business
    // surviving even though resolveMm3HiddensPath would catch it after.
    .replace(/[/\\:*?"<>|]/g, '')
    .replace(/\.+/g, '.')
    .replace(/\s+/g, '_')
    .replace(/^[.\s]+|[.\s]+$/g, '')
    .slice(0, 120);
  return cleaned.length > 0 ? cleaned : null;
}

/** One plan's sidecar, or null when it is missing/corrupt. */
export function readMm3HiddensMeta(ref: string): Partial<Mm3HiddensMeta> | null {
  const resolved = resolveMm3HiddensPath(ref);
  if (!resolved) return null;
  const sidecar = resolved.endsWith('.json') ? resolved : `${resolved}.json`;
  try {
    return JSON.parse(fs.readFileSync(sidecar, 'utf-8'));
  } catch {
    return null;
  }
}

/** Saved plans, newest first. Never throws — a missing directory is [].
 *  `sizeBytes` is read from the file rather than the sidecar so the picker can
 *  show the real cost of each one; these are large enough that it matters. */
export function listMm3Hiddens(): Array<{ file: string } & Partial<Mm3HiddensMeta>> {
  try {
    const dir = mm3HiddensDir();
    if (!fs.existsSync(dir)) return [];
    return fs.readdirSync(dir)
      .filter(f => f.endsWith(MM3_HIDDENS_EXT))
      .map(file => {
        let sizeBytes = 0;
        try {
          sizeBytes = fs.statSync(path.join(dir, file)).size;
        } catch {
          /* listed anyway — an unreadable size is not a reason to hide it */
        }
        return { file, ...(readMm3HiddensMeta(file) ?? {}), sizeBytes };
      })
      .sort((a, b) => String(b.created ?? '').localeCompare(String(a.created ?? '')));
  } catch {
    return [];
  }
}

/** Delete a plan and its sidecar. Returns false when the reference fails
 *  containment or the file is not there. These are the largest artefacts the
 *  app writes, so the picker needs a way to reclaim the space. */
export function deleteMm3Hiddens(ref: string): boolean {
  const resolved = resolveMm3HiddensPath(ref);
  if (!resolved || !resolved.endsWith(MM3_HIDDENS_EXT)) return false;
  try {
    if (!fs.existsSync(resolved)) return false;
    fs.rmSync(resolved);
    fs.rmSync(`${resolved}.json`, { force: true });
    return true;
  } catch {
    return false;
  }
}
