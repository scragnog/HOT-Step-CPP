// training/paths.ts — Dataset Studio path helpers
//
// Slugging, studio-private directory layout, deterministic sample ids and the
// path-traversal guard every server-side write goes through.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §4.1, §7.8

import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { config } from '../../config.js';

/** Audio extensions the scanner accepts (lowercase, leading dot). */
export const AUDIO_EXTENSIONS: ReadonlySet<string> = new Set([
  '.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.aac',
]);

/** Preference order when two audio files in one folder share a sidecar (§7.5). */
export const EXTENSION_PRIORITY: readonly string[] = [
  '.flac', '.wav', '.m4a', '.aac', '.ogg', '.opus', '.mp3',
];

/** Studio-private root: data/training/ (D6). */
export const trainingBaseDir: string = config.training.dir;
try {
  fs.mkdirSync(trainingBaseDir, { recursive: true });
} catch (err: any) {
  console.warn(`[Training] Could not create ${trainingBaseDir}: ${err.message}`);
}

/** Lowercase, [a-z0-9-_] only, runs collapsed, trimmed, max 64 chars. */
export function slugify(name: string): string {
  const s = String(name ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9\-_]+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^[-_]+/, '')
    .replace(/[-_]+$/, '')
    .slice(0, 64)
    .replace(/[-_]+$/, '');
  return s || 'dataset';
}

/** First free slug in the `base`, `base-2`, `base-3`, … series. */
export function uniqueSlug(base: string, taken: Set<string>): string {
  const root = slugify(base);
  if (!taken.has(root)) return root;
  for (let n = 2; n < 10_000; n++) {
    const suffix = `-${n}`;
    const candidate = root.slice(0, 64 - suffix.length) + suffix;
    if (!taken.has(candidate)) return candidate;
  }
  return `${root.slice(0, 50)}-${Date.now().toString(36)}`;
}

/** <trainingBaseDir>/datasets/<slug> — slug re-sanitised before use (§7.8). */
export function datasetDir(slug: string): string {
  return path.join(trainingBaseDir, 'datasets', slugify(slug));
}

/** <trainingBaseDir>/datasets/<slug>/labels */
export function labelsDir(slug: string): string {
  return path.join(datasetDir(slug), 'labels');
}

/** <trainingBaseDir>/datasets/<slug>/mm3-retarget — edited audio for tracks that were over MM3's 6:00 cap. */
export function mm3RetargetDir(slug: string): string {
  return path.join(datasetDir(slug), 'mm3-retarget');
}

/** The derived dataset.json mm3-retarget writes. A build artifact at a known path, so the codes pass and the
 *  trainer can both point at it without threading state between jobs. */
export function mm3RetargetManifest(slug: string): string {
  return path.join(mm3RetargetDir(slug), 'dataset.json');
}

/** <trainingBaseDir>/jobs */
export function jobsDir(): string {
  return path.join(trainingBaseDir, 'jobs');
}

/** Deterministic sample id. Lowercased before hashing — Windows paths are
 *  case-insensitive, so `Track.flac` and `track.flac` must not yield two ids. */
export function sampleIdFor(relPath: string): string {
  return crypto.createHash('sha1').update(relPath.toLowerCase(), 'utf8').digest('hex').slice(0, 16);
}

/** `id` field written into dataset.json — first 8 hex of the same hash (D9). */
export function datasetSampleId(relPath: string): string {
  return sampleIdFor(relPath).slice(0, 8);
}

/** Path relative to sourceDir, forward slashes, no leading './'. */
export function toRelPath(sourceDir: string, abs: string): string {
  const rel = path.relative(sourceDir, abs);
  return rel.split(path.sep).join('/').replace(/^\.\//, '');
}

/** Path-traversal guard: is `candidate` at or below `root`? */
export function isInside(root: string, candidate: string): boolean {
  try {
    const a = path.resolve(root);
    const b = path.resolve(candidate);
    if (a === b) return true;
    let rel = path.relative(a, b);
    if (process.platform === 'win32') {
      rel = path.relative(a.toLowerCase(), b.toLowerCase());
    }
    return !!rel && !rel.startsWith('..') && !path.isAbsolute(rel);
  } catch {
    return false;
  }
}

/** Same dir, same stem, `.txt` — literal extension replacement (§6.1). */
export function sidecarPathFor(audioPath: string): string {
  const dir = path.dirname(audioPath);
  const ext = path.extname(audioPath);
  const stem = ext ? path.basename(audioPath, ext) : path.basename(audioPath);
  return path.join(dir, `${stem}.txt`);
}

/** Extension-stripped absolute path — used for the split-file conventions
 *  (`<stem>.caption.txt`) which are built by concatenation, not replacement. */
export function stemPathFor(audioPath: string): string {
  const ext = path.extname(audioPath);
  return ext ? audioPath.slice(0, audioPath.length - ext.length) : audioPath;
}
