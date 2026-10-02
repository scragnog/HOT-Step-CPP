import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { listDatasets } from './training/datasetsRepo.js';
import { dedupeBySidecar, scanAudioFiles } from './training/datasetScan.js';
import { sampleIdFor } from './training/paths.js';
import type { TrainingDatasetRow } from './training/types.js';
import { writeAbcSidecar } from './training/abcSidecar.js';

export interface CoverDatasetSource {
  datasetId: string;
  sampleId: string;
  audioPath: string;
}

interface IndexedSource extends CoverDatasetSource {
  canonicalPath: string;
  sizeBytes: number;
}

interface DatasetIndex {
  stamp: string;
  byPath: Map<string, IndexedSource>;
  bySize: Map<number, IndexedSource[]>;
}

const datasetIndexes = new Map<string, DatasetIndex>();

function canonical(file: string): string {
  const value = fs.realpathSync(file);
  return process.platform === 'win32' ? value.toLowerCase() : value;
}

function sha256(file: string): string {
  return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

function indexFor(dataset: TrainingDatasetRow): DatasetIndex {
  // These row fields change when the dataset is edited or its sample count is
  // refreshed. The source path and recursion flag also affect the scan.
  const stamp = JSON.stringify([dataset.sourceDir, dataset.recursive, dataset.updatedAt, dataset.sampleCount]);
  const cached = datasetIndexes.get(dataset.id);
  if (cached?.stamp === stamp) return cached;
  const index: DatasetIndex = { stamp, byPath: new Map(), bySize: new Map() };
  for (const file of dedupeBySidecar(scanAudioFiles(dataset.sourceDir, dataset.recursive))) {
    try {
      const source: IndexedSource = { datasetId: dataset.id, sampleId: sampleIdFor(file.relPath),
        audioPath: file.absPath, canonicalPath: canonical(file.absPath), sizeBytes: file.sizeBytes };
      if (!index.byPath.has(source.canonicalPath)) index.byPath.set(source.canonicalPath, source);
      const sized = index.bySize.get(source.sizeBytes) ?? [];
      sized.push(source);
      index.bySize.set(source.sizeBytes, sized);
    } catch { /* a file removed during the scan is not a candidate */ }
  }
  datasetIndexes.set(dataset.id, index);
  return index;
}

/** Dataset sample IDs hash the relative filename, so compare paths first.
 * Library songs have no dataset origin; copied audio falls back to exact bytes. */
export function resolveCoverDatasetSource(
  audioPath: string, datasets: TrainingDatasetRow[] = listDatasets(),
): CoverDatasetSource | null {
  let sourcePath: string;
  let sourceSize: number;
  try { sourcePath = canonical(audioPath); sourceSize = fs.statSync(audioPath).size; }
  catch { return null; }
  const indexes: DatasetIndex[] = [];
  for (const dataset of datasets) {
    try { indexes.push(indexFor(dataset)); }
    catch { /* one unreadable dataset must not block cover selection */ }
  }
  for (const index of indexes) {
    const direct = index.byPath.get(sourcePath);
    if (direct) {
      return { datasetId: direct.datasetId, sampleId: direct.sampleId, audioPath: direct.audioPath };
    }
  }
  const candidates = indexes.flatMap(index => index.bySize.get(sourceSize) ?? []);
  if (!candidates.length) return null;
  const digest = sha256(audioPath);
  const matches = candidates.filter(c => {
    try { return sha256(c.audioPath) === digest; } catch { return false; }
  });
  // Several datasets may register the same source path. Different paths with
  // identical bytes may carry conflicting sidecars, so do not guess.
  const paths = new Set(matches.map(c => c.canonicalPath));
  return paths.size === 1 ? { datasetId: matches[0].datasetId, sampleId: matches[0].sampleId,
    audioPath: matches[0].audioPath } : null;
}

export function saveDatasetCoverAbc(
  audioPath: string, abc: string, match = resolveCoverDatasetSource,
): boolean {
  let datasetSource: CoverDatasetSource | null;
  try { datasetSource = match(audioPath); }
  catch { return false; }
  return datasetSource ? writeAbcSidecar(datasetSource.audioPath, abc) : false;
}
