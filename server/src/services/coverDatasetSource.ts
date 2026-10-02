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

function canonical(file: string): string {
  const value = fs.realpathSync(file);
  return process.platform === 'win32' ? value.toLowerCase() : value;
}

function sha256(file: string): string {
  return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
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
  const candidates: CoverDatasetSource[] = [];
  for (const dataset of datasets) {
    try {
      for (const file of dedupeBySidecar(scanAudioFiles(dataset.sourceDir, dataset.recursive))) {
        if (file.sizeBytes !== sourceSize) continue;
        candidates.push({ datasetId: dataset.id, sampleId: sampleIdFor(file.relPath), audioPath: file.absPath });
      }
    } catch { /* one unreadable dataset must not block cover selection */ }
  }
  const direct = candidates.filter(c => canonical(c.audioPath) === sourcePath);
  if (direct.length) return direct[0];
  if (!candidates.length) return null;
  const digest = sha256(audioPath);
  const matches = candidates.filter(c => {
    try { return sha256(c.audioPath) === digest; } catch { return false; }
  });
  // Several datasets may register the same source path. Different paths with
  // identical bytes may carry conflicting sidecars, so do not guess.
  const paths = new Set(matches.map(c => canonical(c.audioPath)));
  return paths.size === 1 ? matches[0] : null;
}

export function saveDatasetCoverAbc(
  audioPath: string, abc: string, match = resolveCoverDatasetSource,
): boolean {
  let datasetSource: CoverDatasetSource | null;
  try { datasetSource = match(audioPath); }
  catch { return false; }
  return datasetSource ? writeAbcSidecar(datasetSource.audioPath, abc) : false;
}
