// YuE2 scans a flat folder. Keep the dataset's included audio and its sidecars
// under the preprocess output: later stages reopen the manifest's source paths.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import type { TrainingSample } from './types.js';

type AudioSample = Pick<TrainingSample, 'audioPath' | 'relPath' | 'excluded' | 'fileMissing'>;
export const YUE2_AUDIO_EXTS = new Set(['.wav', '.flac', '.mp3', '.ogg', '.m4a']);

export function selectYue2Audio<T extends AudioSample>(samples: T[]) {
  const included = samples.filter(s => !s.excluded && !s.fileMissing);
  const files = included.filter(s => YUE2_AUDIO_EXTS.has(path.extname(s.audioPath).toLowerCase()));
  return {
    files,
    unsupported: included.length - files.length,
    needFfmpeg: files.filter(s => !['.wav', '.mp3'].includes(path.extname(s.audioPath).toLowerCase())).length,
  };
}

const stem = (name: string) => name.slice(0, -path.extname(name).length);
const SIDECARS = ['.txt', '.yue2.txt', '.mm3.txt'];
const destinations = (name: string) => [name, ...SIDECARS.map(suffix => `${stem(name)}${suffix}`)]
  .map(file => file.toLowerCase());
const SOURCE_MAP = '_staged_audio.sources.json';
interface StagedSource { name: string; source: string; relPath: string }

function putFile(source: string, destination: string, link: (source: string, destination: string) => void): void {
  try { link(source, destination); }
  catch {
    fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);
    const st = fs.statSync(source);
    fs.utimesSync(destination, st.atime, st.mtime);
  }
}

/** Not fs.rmSync: on Windows, Node 24.11 silently skips a path with non-ASCII
 * characters (an em dash in a title), so the copy after it hit EEXIST. */
function removeFile(file: string): void {
  try { fs.unlinkSync(file); }
  catch (e) { if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e; }
}

/** Separation must use the same flattened names as preprocessing. Old caches
 * without an audio_dir retain the original-folder fallback, and so does a
 * cache whose audio_dir was moved away (the manifest is fingerprinted, so it
 * is not rewritten to follow the move). */
export function yue2SeparationAudioDir(manifestPath: string, sourceDir: string): string {
  if (!fs.existsSync(manifestPath)) return sourceDir;
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8')) as { audio_dir?: string };
  const dir = manifest.audio_dir;
  return typeof dir === 'string' && dir && fs.existsSync(dir) ? dir : sourceDir;
}

/** Re-pair current original sidecars with the stable staged audio names.
 * Replace links rather than overwriting them, which would edit originals.
 * This also handles atomic sidecar replacement, copy fallback and deletion. */
export function refreshYue2StagedSidecars(manifestPath: string): void {
  const output = path.dirname(path.resolve(manifestPath));
  const mapPath = path.join(output, SOURCE_MAP);
  if (!fs.existsSync(mapPath)) return; // Legacy cache, not staged by this helper.
  const sources = JSON.parse(fs.readFileSync(mapPath, 'utf8')) as StagedSource[];
  const dir = path.join(output, '_staged_audio');
  for (const entry of sources) {
    if (!entry.name || path.basename(entry.name) !== entry.name || !path.isAbsolute(entry.source)) {
      throw new Error('Invalid YuE2 staged source mapping');
    }
    for (const suffix of SIDECARS) {
      const source = `${stem(entry.source)}${suffix}`;
      const destination = path.join(dir, `${stem(entry.name)}${suffix}`);
      // Older staging names may overlap suffixes; never unlink original files.
      if (path.resolve(source) === path.resolve(destination)) throw new Error('Invalid YuE2 staged sidecar mapping');
      removeFile(destination);
      if (fs.existsSync(source)) putFile(source, destination, fs.linkSync);
    }
  }
}

/** Staged name -> the ORIGINAL audio in the dataset folder, for a manifest whose
 *  sources were staged by stageYue2Audio. Empty for a legacy cache (no map). */
export function readYue2StagedSources(manifestPath: string): Map<string, string> {
  const mapPath = path.join(path.dirname(path.resolve(manifestPath)), SOURCE_MAP);
  const out = new Map<string, string>();
  if (!fs.existsSync(mapPath)) return out;
  for (const e of JSON.parse(fs.readFileSync(mapPath, 'utf8')) as StagedSource[]) {
    if (e?.name && path.isAbsolute(e.source)) out.set(e.name, e.source);
  }
  return out;
}

/** Keep unique basenames; flatten colliding paths and protect sidecar stems too.
 * Reserve all potential sidecars, even absent ones: captions may be added later.
 * A hash resolves collisions with literal `__` names or flattened paths. */
export function yue2StagedNames(samples: AudioSample[]): Map<string, string> {
  const counts = new Map<string, number>();
  for (const s of samples) {
    const key = stem(path.basename(s.audioPath)).toLowerCase();
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  const result = new Map<string, string>();
  const used = new Set<string>();
  // Reserve unique names before assigning flattened ones.
  for (const s of samples) {
    const name = path.basename(s.audioPath);
    if (counts.get(stem(name).toLowerCase()) === 1 && !destinations(name).some(file => used.has(file))) {
      result.set(s.relPath, name);
      destinations(name).forEach(file => used.add(file));
    }
  }
  for (const s of [...samples].sort((a, b) => a.relPath.localeCompare(b.relPath))) {
    if (result.has(s.relPath)) continue;
    const flat = s.relPath.replace(/[/\\]/g, '__');
    let name = flat;
    let attempt = 0;
    while (destinations(name).some(file => used.has(file))) {
      const hash = createHash('sha256').update(`${s.relPath}:${attempt++}`).digest('hex').slice(0, 12);
      name = `${stem(flat)}__${hash}${path.extname(flat)}`;
    }
    destinations(name).forEach(file => used.add(file));
    result.set(s.relPath, name);
  }
  return result;
}

/** Rebuild only our staging directory. Hardlinks avoid copying large audio;
 * cross-volume or unsupported links fall back to a copy preserving mtime so
 * the engine's source cache stays reusable. Never remove this on job finish. */
export function stageYue2Audio(
  samples: AudioSample[], outDir: string,
  link: (source: string, destination: string) => void = fs.linkSync,
): string {
  const output = path.resolve(outDir);
  const dir = path.resolve(output, '_staged_audio');
  if (path.dirname(dir) !== output) throw new Error('Invalid YuE2 staging directory');
  const files = selectYue2Audio(samples).files;
  if (files.length === 0) throw new Error('Dataset has no included YuE2 audio files');
  for (const s of files) {
    const relative = path.relative(dir, path.resolve(s.audioPath));
    if (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative)) {
      throw new Error('YuE2 staging cannot contain its own source audio');
    }
  }
  const names = yue2StagedNames(files);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const sources: StagedSource[] = [];
  for (const s of files) {
    const name = names.get(s.relPath)!;
    putFile(s.audioPath, path.join(dir, name), link);
    for (const suffix of SIDECARS) {
      const source = `${stem(s.audioPath)}${suffix}`;
      if (fs.existsSync(source)) putFile(source, path.join(dir, `${stem(name)}${suffix}`), link);
    }
    sources.push({ name, source: path.resolve(s.audioPath), relPath: s.relPath });
  }
  fs.writeFileSync(path.join(output, SOURCE_MAP), JSON.stringify(sources, null, 2));
  return dir;
}
