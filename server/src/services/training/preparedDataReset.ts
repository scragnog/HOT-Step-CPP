import fs from 'fs';
import path from 'path';
import { datasetDir, isInside } from './paths.js';
import { tensorsRoot } from './aceTrain.js';
import { snapshotYue2Sheets } from './datasetProfile.js';

export interface PreparedCache {
  name: string;
  path: string;
  files: number;
  bytes: number;
}

/** Only app-owned, reproducible training outputs. Source audio, sidecars,
 * edited labels, dataset.json, adapters and job records are excluded. */
export function preparedCachePaths(slug: string): Array<{ name: string; path: string }> {
  const root = datasetDir(slug);
  return [
    'yue2-latents', 'yue2-stems', 'mm3-codes', 'mm3-codes-laundered',
    'mm3-retarget',
  ].map(name => ({ name, path: path.join(root, name) }))
    .concat({ name: 'ACE tensor caches', path: tensorsRoot(slug) });
}

function measure(target: string): { files: number; bytes: number } {
  const stat = fs.lstatSync(target);
  if (stat.isSymbolicLink()) throw new Error(`Prepared cache contains a link: ${target}`);
  if (!stat.isDirectory()) return { files: 1, bytes: stat.size };
  let files = 0;
  let bytes = 0;
  for (const entry of fs.readdirSync(target)) {
    const child = measure(path.join(target, entry));
    files += child.files;
    bytes += child.bytes;
  }
  return { files, bytes };
}

/** YuE2's core prepared data: latents, codes, lead sheets, alignment and the
 *  prepared training sets, ~40 MiB an album that costs minutes of GPU to
 *  rebuild (lead sheets alone ~3 min). Every clear keeps it unless asked
 *  (2026-09-30): what goes stale is rebuilt anyway, since the prepared set is
 *  fingerprinted and a loudness or caption change re-cuts the cache. */
export const YUE2_CORE_CACHE = 'yue2-latents';
export interface PreparedCacheScope { includeYue2Core?: boolean }

export function listPreparedCaches(slug: string, sourceDir: string, scope: PreparedCacheScope = {}): PreparedCache[] {
  const datasetRoot = datasetDir(slug);
  const tensorRoot = tensorsRoot(slug);
  const roots = [datasetRoot, tensorRoot];
  for (const root of roots) {
    if (fs.existsSync(root) && fs.lstatSync(root).isSymbolicLink()) throw new Error(`Cache root is a link: ${root}`);
  }
  // Every path passes the safety walk, kept or not, before the scope filter.
  return preparedCachePaths(slug).flatMap(item => {
    const target = path.resolve(item.path);
    const owner = item.name === 'ACE tensor caches' ? path.dirname(tensorRoot) : datasetRoot;
    if (!isInside(owner, target) || isInside(target, sourceDir) || path.resolve(sourceDir) === target) {
      throw new Error(`Refusing to clear a cache that may contain source files: ${target}`);
    }
    if (!fs.existsSync(target)) return [];
    const stat = fs.lstatSync(target);
    if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error(`Unexpected cache path: ${target}`);
    return [{ ...item, ...measure(target) }];
  }).filter(cache => scope.includeYue2Core || cache.name !== YUE2_CORE_CACHE);
}

export function clearPreparedCaches(slug: string, sourceDir: string, scope: PreparedCacheScope = {}): PreparedCache[] {
  // Complete the safety walk before removing anything.
  const caches = listPreparedCaches(slug, sourceDir, scope);
  if (caches.some(c => c.name === YUE2_CORE_CACHE)) {
    try { snapshotYue2Sheets(slug); }
    catch (err: any) { console.warn(`[Training] Could not keep the YuE2 lead sheets for ${slug}: ${err?.message || err}`); }
  }
  for (const cache of caches) fs.rmSync(cache.path, { recursive: true, force: false });
  return caches;
}
