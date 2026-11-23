// workspaceArchive/inventory.ts — Which files a workspace archive carries.
//
// Sources are locateData()'s children, not a walk of the data root: adapters
// and training can live on another drive. Library and Training carry an
// allowlist of top-level names; Adapters and Models take user-named folders but
// only known file types. Everywhere, at any depth: links, temp files, backups
// and credential files (unless Secrets is on) stay out. Anything left out is
// listed with a reason. Read-only.

import fs from 'node:fs';
import path from 'node:path';
import type { DataChildren } from '../../dataroot/locate.js';
import { isInside, samePath } from '../../dataroot/paths.js';
import { isCredentialKey } from '../generation/requestCapture.js';
import type { CategoryChoice, CategoryId, SkippedEntry } from './manifest.js';

/** Library top-level names that are the workspace. hotstep.db is carried as a snapshot. */
export const LIBRARY_NAMES = new Set([
  'audio', 'references', 'profiles', 'seeds', 'midi', 'stems', 'vst', 'lyrics', 'muscriptor',
  'mm3-planks', 'mm3-hiddens', 'stream-sessions',
]);
/** Known library names left out on purpose. */
const LIBRARY_LEFT: Record<string, SkippedEntry['reason']> = {
  peaks: 'cache', 'stem-cache': 'cache', cache: 'cache', models: 'cache',
  tmp: 'temp', download_temp: 'temp', import_temp: 'temp', mastering_temp: 'temp', daw_tmp: 'temp',
  'dev-captures': 'machine', 'updates.json': 'machine',
  backups: 'backup', 'lireek.db': 'backup', 'lireek.db.migrated': 'backup',
};
/** Carried by the DB snapshot, never as files. */
const DB_NAMES = new Set(['hotstep.db', 'hotstep.db-wal', 'hotstep.db-shm', 'hotstep.db-journal']);
export const TRAINING_NAMES = new Set([
  'datasets', 'jobs', 'pipelines', 'previews', 'tensors', 'worker-datasets', 'yue2-batches', 'yue2-preparation',
  'mm3-linked.json', 'mm3-mirror.json', 'yue2-linked.json', 'yue2-aitk-runs.json', 'yue2-mirror-tombstones.json',
]);
const TEMP = /(\.tmp|\.part|\.hspart|\.partial|\.crdownload|_temp)$/i;
const BACKUP = /(\.bak(\b|-)|^_?backup[-_]|_backup_|\.pre-transfer-|\.before-| - copy( \(\d+\))?\.)/i;

/** Dotfiles and key files that hold credentials, plus a credential-named
 *  plain file (hf_token, api_key.txt). Weights and configs never match: their
 *  names (tokenizer.json, added_tokens.json) do not end in "token". */
const SECRET_NAME = /^\.(env|netrc|git-credentials|npmrc|pypirc)(\.|$)|\.(pem|key|pfx|p12)$/i;
export function isSecretFile(name: string): boolean {
  const e = path.extname(name);
  return SECRET_NAME.test(name) || (/^(|\.txt|\.json)$/i.test(e) && isCredentialKey(path.basename(name, e)));
}

const ext = (name: string) => path.extname(name).slice(1).toLowerCase();
const NOTICE = /^(LICENSE|NOTICE|README|COPYING)([-_.].*)?$/i;
type FileRule = (name: string) => SkippedEntry['reason'] | null;
/** Adapter folders: weights, configs, run logs, audition audio and lead sheets. */
const ADAPTER_EXT = new Set(['safetensors', 'gguf', 'bin', 'pt', 'json', 'jsonl', 'yaml', 'yml', 'txt', 'md', 'log', 'abc', 'wav', 'flac', 'mp3', 'npz']);
const adapterFile: FileRule = n => ADAPTER_EXT.has(ext(n)) || n === 'finished' || NOTICE.test(n) ? null : 'unknown';
/** Model folders: weights, tokenizers, sidecars and repo code. GPU-built
 *  engines and binaries belong to this machine. */
const MODEL_EXT = new Set(['gguf', 'safetensors', 'bin', 'pt', 'pth', 'ckpt', 'onnx', 'json', 'jsonl', 'txt', 'md', 'model',
  'tiktoken', 'jinja', 'sha256', 'metadata', 'yaml', 'yml', 'py', 'npz', 'npy']);
const MACHINE_EXT = new Set(['engine', 'plan', 'exe', 'dll', 'so', 'dylib', 'lib', 'pdb']);
const modelFile: FileRule = n => MACHINE_EXT.has(ext(n)) ? 'machine' : ext(n) === 'pyc' ? 'cache'
  : MODEL_EXT.has(ext(n)) || NOTICE.test(n) ? null : 'unknown';

/** At most this many skipped entries are named; the rest are counted. */
export const SKIPPED_CAP = 1000;

export interface PlannedSource {
  category: Exclude<CategoryId, 'secrets'>;
  prefix: string;
  dir: string;
  selected: boolean;
  /** Top-level names carried; null = any name. */
  allow: Set<string> | null;
  /** Why a file is left out by type (null = carried); null = any type. */
  fileRule: FileRule | null;
}

export interface PlannedFile { abs: string; name: string; bytes: number; category: CategoryId }

export interface Inventory {
  sources: PlannedSource[];
  files: PlannedFile[];
  skipped: SkippedEntry[];
  skippedNotListed: number;
  totals: Record<CategoryId, { files: number; bytes: number }>;
}

/** Every source this machine has, selected or not (secrets has no folder). */
export function planSources(c: DataChildren, choice: CategoryChoice): PlannedSource[] {
  const out: PlannedSource[] = [
    { category: 'library', prefix: 'library', dir: c.library, selected: true, allow: LIBRARY_NAMES, fileRule: null },
    { category: 'training', prefix: 'library/training', dir: c.training, selected: choice.training, allow: TRAINING_NAMES, fileRule: null },
    { category: 'adapters', prefix: 'adapters', dir: c.adapters, selected: choice.adapters, allow: null, fileRule: adapterFile },
    { category: 'models', prefix: 'models', dir: c.models, selected: choice.models, allow: null, fileRule: modelFile },
  ];
  if (!isInside(c.models, c.muscriptorModels)) {
    out.push({ category: 'models', prefix: 'models/muscriptor', dir: c.muscriptorModels, selected: choice.models, allow: null, fileRule: modelFile });
  }
  // A source nested in another is walked on its own; two at one path are one.
  return out.filter((s, i) => !out.slice(0, i).some(o => samePath(o.dir, s.dir)));
}

/** The source that owns `p` (the most specific folder containing it), or null. */
export function ownerOf(sources: PlannedSource[], p: string): PlannedSource | null {
  let best: PlannedSource | null = null;
  for (const s of sources) if (isInside(s.dir, p) && (!best || s.dir.length > best.dir.length)) best = s;
  return best;
}

const emptyTotals = () => Object.fromEntries((['library', 'training', 'adapters', 'models', 'secrets'] as CategoryId[])
  .map(k => [k, { files: 0, bytes: 0 }])) as Inventory['totals'];

/** Walk the selected sources. Sync: a big training folder takes a few seconds.
 *  ponytail: sync walk blocks the event loop on a 100k-file tree; move it to a worker if that shows. */
export function takeInventory(c: DataChildren, choice: CategoryChoice): Inventory {
  const sources = planSources(c, choice);
  const inv: Inventory = { sources, files: [], skipped: [], skippedNotListed: 0, totals: emptyTotals() };
  const skip = (name: string, reason: SkippedEntry['reason']) => {
    if (inv.skipped.length < SKIPPED_CAP) inv.skipped.push({ path: name, reason }); else inv.skippedNotListed++;
  };

  for (const src of sources.filter(s => s.selected)) {
    let top: fs.Dirent[];
    try { top = fs.readdirSync(src.dir, { withFileTypes: true }); } catch { continue; }
    const add = (abs: string, name: string, category: CategoryId) => {
      const st = fs.statSync(abs);
      inv.files.push({ abs, name, bytes: st.size, category });
      inv.totals[category].files++; inv.totals[category].bytes += st.size;
    };

    const walk = (abs: string, name: string, d: fs.Dirent): void => {
      if (d.isSymbolicLink()) { skip(name, 'link'); return; }
      if (TEMP.test(d.name)) { skip(name, 'temp'); return; }
      if (BACKUP.test(d.name)) { skip(name, 'backup'); return; }
      if (d.isDirectory()) {
        const nested = sources.find(o => o !== src && samePath(o.dir, abs));
        if (nested) { if (!nested.selected) skip(name, 'category-off'); return; }
        let kids: fs.Dirent[];
        try { kids = fs.readdirSync(abs, { withFileTypes: true }); } catch { return; }
        for (const k of kids) walk(path.join(abs, k.name), `${name}/${k.name}`, k);
        return;
      }
      if (!d.isFile()) { skip(name, 'unknown'); return; }
      if (isSecretFile(d.name)) {
        if (choice.secrets) add(abs, name, 'secrets'); else skip(name, 'secret');
        return;
      }
      const reason = src.fileRule?.(d.name) ?? null;
      if (reason) { skip(name, reason); return; }
      add(abs, name, src.category);
    };

    for (const d of top) {
      const name = `${src.prefix}/${d.name}`;
      if (src.category === 'library' && DB_NAMES.has(d.name)) continue;
      const left = src.category === 'library' ? LIBRARY_LEFT[d.name] : undefined;
      if (left) { skip(name, left); continue; }
      // A nested source's folder is walked (or listed) by walk().
      const nested = d.isDirectory() && sources.some(o => o !== src && samePath(o.dir, path.join(src.dir, d.name)));
      if (src.allow && !src.allow.has(d.name) && !nested) {
        skip(name, TEMP.test(d.name) ? 'temp' : BACKUP.test(d.name) ? 'backup' : 'unknown');
        continue;
      }
      walk(path.join(src.dir, d.name), name, d);
    }
  }
  return inv;
}

/** Why `outDir` cannot take an archive of these sources (empty when it can). */
export function destinationErrors(outDir: string, sources: PlannedSource[]): string[] {
  if (!path.isAbsolute(outDir)) return ['the destination must be an absolute folder path'];
  let st: fs.Stats;
  try { st = fs.statSync(outDir); } catch { return [`${outDir} does not exist`]; }
  if (!st.isDirectory()) return [`${outDir} is not a folder`];
  return sources.filter(s => s.selected && isInside(s.dir, outDir))
    .map(s => `the destination is inside the ${s.category} folder being exported (${s.dir})`);
}

/** Free bytes on the volume holding `dir`, or null when unknown. */
export function freeBytes(dir: string): number | null {
  try { const s = fs.statfsSync(dir); return s.bavail * s.bsize; } catch { return null; }
}
