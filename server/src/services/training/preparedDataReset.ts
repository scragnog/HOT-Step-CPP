import fs from 'fs';
import path from 'path';
import { datasetDir, isInside } from './paths.js';
import { tensorsRoot } from './aceTrain.js';

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

export function listPreparedCaches(slug: string, sourceDir: string): PreparedCache[] {
  const datasetRoot = datasetDir(slug);
  const tensorRoot = tensorsRoot(slug);
  const roots = [datasetRoot, tensorRoot];
  for (const root of roots) {
    if (fs.existsSync(root) && fs.lstatSync(root).isSymbolicLink()) throw new Error(`Cache root is a link: ${root}`);
  }
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
  });
}

export function clearPreparedCaches(slug: string, sourceDir: string): PreparedCache[] {
  // Complete the safety walk before removing anything.
  const caches = listPreparedCaches(slug, sourceDir);
  for (const cache of caches) fs.rmSync(cache.path, { recursive: true, force: false });
  return caches;
}
