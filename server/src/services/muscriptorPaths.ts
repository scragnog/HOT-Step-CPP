// muscriptorPaths.ts — where MuScriptor and Beat This! weights live (#228)
//
// Managed home: <models>/muscriptor/<variant>, beside the other shared models.
// Before #228 it was <library>/models/muscriptor/<variant>; a complete download
// there is still used as it stands, with no network access, until the user
// migrates it. An explicit MUSCRIPTOR_MODELS_DIR always wins and keeps its old
// meaning: the raw value, so a relative path resolves against the cwd as before.
//
// No config import: callers pass the roots, so tests and the data-root
// inspector (dataroot/locate.ts) use the same rules as the runtime.

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { pipeline } from 'stream/promises';
import { samePath } from '../dataroot/paths.js';

/** The two files every variant needs; both must exist for it to be usable. */
export const MODEL_FILES = ['model.safetensors', 'config.json'] as const;

export interface MuscriptorRoots {
  /** MUSCRIPTOR_MODELS_DIR as set, or null. */
  override: string | null;
  /** <models>/muscriptor */
  shared: string;
  /** <library>/models/muscriptor (pre-#228 default) */
  legacy: string;
}

export type ModelSource = 'override' | 'shared' | 'legacy';

export function muscriptorRoots(override: string | undefined | null, modelsDir: string, libraryDir: string): MuscriptorRoots {
  return {
    override: override || null,
    shared: path.join(modelsDir, 'muscriptor'),
    legacy: path.join(libraryDir, 'models', 'muscriptor'),
  };
}

/** Where downloads go and Settings shows: the override, else the shared root. */
export function targetRoot(r: MuscriptorRoots): string {
  return r.override ?? r.shared;
}

/** Both files present as regular files. A `.part` from an interrupted download does not count. */
export function isCompleteModelDir(dir: string): boolean {
  return MODEL_FILES.every(f => {
    try { return fs.statSync(path.join(dir, f)).isFile(); } catch { return false; }
  });
}

/** The folder a variant is read from. The override is used even when empty;
 *  otherwise a complete shared copy, then a complete legacy one, then the
 *  shared folder as the download destination. */
export function resolveModelDir(r: MuscriptorRoots, variant: string): { dir: string; source: ModelSource; complete: boolean } {
  if (r.override) {
    const dir = path.join(r.override, variant);
    return { dir, source: 'override', complete: isCompleteModelDir(dir) };
  }
  const shared = path.join(r.shared, variant);
  if (isCompleteModelDir(shared)) return { dir: shared, source: 'shared', complete: true };
  // dettato never had a pre-#228 home: its four folders are read from one root, or the launch could not find them all.
  const legacy = path.join(r.legacy, variant);
  if (!variant.startsWith('dettato') && isCompleteModelDir(legacy)) return { dir: legacy, source: 'legacy', complete: true };
  return { dir: shared, source: 'shared', complete: false };
}

// ── Explicit legacy → shared migration ───────────────────────────────────

export type MigrationOutcome =
  /** nothing complete under the legacy root */
  | 'none'
  /** copied and verified; legacy kept unless removeLegacy */
  | 'migrated'
  /** shared already held identical files */
  | 'identical'
  /** shared holds different files under the same name: nothing touched */
  | 'conflict'
  /** override set: the managed roots are not in use */
  | 'override'
  | 'error';

export interface MigrationResult { variant: string; outcome: MigrationOutcome; detail?: string }

async function sha256(file: string): Promise<string> {
  const h = crypto.createHash('sha256');
  await pipeline(fs.createReadStream(file), h);
  return h.digest('hex');
}

async function sameFile(a: string, b: string): Promise<boolean> {
  return fs.statSync(a).size === fs.statSync(b).size && await sha256(a) === await sha256(b);
}

/**
 * Copy complete legacy variants into the shared root. Never overwrites: a
 * final file already in shared must match byte for byte, or the variant is
 * reported as a conflict and left alone. Each file is copied to a temp name,
 * checked by size and SHA-256 against the source, then renamed into place.
 * The legacy copy is removed only with removeLegacy and only after every
 * file of that variant is verified in shared.
 */
export async function migrateLegacyModels(r: MuscriptorRoots, variants: readonly string[],
                                          opts: { removeLegacy?: boolean } = {}): Promise<MigrationResult[]> {
  const out: MigrationResult[] = [];
  for (const variant of variants) {
    if (r.override) { out.push({ variant, outcome: 'override' }); continue; }
    const from = path.join(r.legacy, variant);
    const to = path.join(r.shared, variant);
    if (!isCompleteModelDir(from)) { out.push({ variant, outcome: 'none' }); continue; }
    try {
      // Same folder under two names (ACESTEPCPP_MODELS=<library>/models, a
      // junction): nothing to copy, and removing "legacy" would remove shared.
      if (fs.existsSync(to) && samePath(fs.realpathSync(from), fs.realpathSync(to))) {
        out.push({ variant, outcome: 'identical', detail: 'shared and legacy are the same folder' });
        continue;
      }
      // Check every existing destination before publishing anything, so a
      // conflict never leaves a mixed shared pair that outranks legacy.
      let conflict: string | null = null;
      for (const f of MODEL_FILES) {
        const dst = path.join(to, f);
        if (fs.existsSync(dst) && !await sameFile(path.join(from, f), dst)) { conflict = f; break; }
      }
      const published: string[] = [];
      for (const f of conflict ? [] : MODEL_FILES) {
        const src = path.join(from, f);
        const dst = path.join(to, f);
        if (fs.existsSync(dst)) continue;
        fs.mkdirSync(to, { recursive: true });
        const tmp = `${dst}.migrate-${process.pid}`;
        try {
          fs.copyFileSync(src, tmp);
          if (!await sameFile(src, tmp)) throw new Error(`${f} did not verify after copy`);
          // Something may have written the name meanwhile: re-check before claiming it.
          if (fs.existsSync(dst)) {
            if (!await sameFile(src, dst)) { conflict = f; break; }
          } else {
            fs.renameSync(tmp, dst);
            published.push(dst);
          }
        } finally {
          fs.rmSync(tmp, { force: true });
        }
      }
      if (conflict) {
        for (const p of published) fs.rmSync(p, { force: true });
        out.push({ variant, outcome: 'conflict', detail: `${path.join(to, conflict)} differs from ${path.join(from, conflict)}` });
        continue;
      }
      if (opts.removeLegacy) for (const f of MODEL_FILES) fs.rmSync(path.join(from, f), { force: true });
      out.push({ variant, outcome: published.length ? 'migrated' : 'identical' });
    } catch (e: any) {
      out.push({ variant, outcome: 'error', detail: e?.message || String(e) });
    }
  }
  return out;
}
