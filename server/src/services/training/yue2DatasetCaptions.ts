// yue2DatasetCaptions.ts — the captions a YuE2 training dataset offers as
// caption sources. Shared by GET /api/training/yue2-dataset-captions and the
// server-side request resolver (services/generation/resolve).

import fs from 'fs';
import path from 'path';
import * as repo from './datasetsRepo.js';
import { buildSamples } from './datasetScan.js';
import { jointRunForAdapter } from './yue2AitkRuns.js';
import { yue2StyleString } from '../backends/yue2/style.js';
import type { TrainingDatasetRow } from './types.js';

/** The caption a dataset track contributes, newest source first: the YuE2
 *  planner sentence beside the audio, then whatever the cache baked in (the ACE
 *  caption), then the MM3 structured caption flattened to one line.
 *
 *  Sidecar FIRST, not manifest first. A cache is cut once and then outlives
 *  several rounds of captioning, so the manifest is a snapshot of what the
 *  labels said on the day — reading it in preference to the file on disk is how
 *  an album with eleven `.yue2.txt` beside its audio rendered from its ACE
 *  captions for weeks without anything saying so.
 *
 *  Note the consequence: a caption served here may be one the adapter never
 *  trained on, if the cache was cut in `ace` mode. That is the user's call —
 *  re-cut and retrain to make the two agree. */
export function datasetTrackCaption(audioPath: string, baked: string): string {
  const read = (file: string): string => {
    try { return fs.readFileSync(file, 'utf8').trim(); } catch { return ''; }
  };
  if (!audioPath) return baked.trim();
  const stem = audioPath.slice(0, audioPath.length - path.extname(audioPath).length);
  // The MM3 caption is a multi-line structured block; as a YuE2 style it is a
  // last resort and it goes in as one line, never as its own field layout.
  return read(`${stem}.yue2.txt`)
    || baked.trim()
    || read(`${stem}.mm3.txt`).replace(/\s+/g, ' ').trim();
}

export interface Yue2DatasetCaptionTrack {
  name: string;
  caption: string;
  genre: string;
  bpm: string;
  key: string;
  styled: string;
}

export interface Yue2DatasetCaptions {
  datasetId: string;
  datasetSlug: string;
  datasetName: string;
  tracks: Yue2DatasetCaptionTrack[];
}

/** One of `dataset` (id), `lyricsSet` (resolved through
 *  training_datasets.lyrics_set_id) or `adapter` (resolved through the run that
 *  trained it). Every failure answers all-empty. */
export async function yue2DatasetCaptions(
  by: { dataset?: string; lyricsSet?: number; adapter?: string },
): Promise<Yue2DatasetCaptions> {
  const empty: Yue2DatasetCaptions = { datasetId: '', datasetSlug: '', datasetName: '', tracks: [] };
  try {
    let ds: TrainingDatasetRow | null = null;
    if (by.dataset) ds = repo.getDataset(by.dataset);
    else if (by.lyricsSet !== undefined) {
      const id = Number(by.lyricsSet);
      ds = Number.isFinite(id) ? repo.listDatasets().find(d => Number(d.lyricsSetId) === id) ?? null : null;
    } else if (by.adapter) {
      const run = jointRunForAdapter(by.adapter);
      ds = run ? repo.getDataset(run.datasetId) ?? repo.listDatasets().find(d => d.slug === run.datasetSlug) ?? null : null;
    }
    if (!ds) return empty;
    const samples = await buildSamples(ds);
    // Parity with the prepared style: a `.yue2.txt` caption is the whole
    // trained sentence (it carries its own BPM), so it gets no tail; an ACE
    // caption gets the trainer's "<genre>, <bpm> BPM, key of <key>." tail.
    const yue2Sidecar = (audioPath: string): string => {
      try { return fs.readFileSync(audioPath.slice(0, audioPath.length - path.extname(audioPath).length) + '.yue2.txt', 'utf8').replace(/\s+/g, ' ').trim(); }
      catch { return ''; }
    };
    const tracks = samples
      .filter(s => !s.excluded && !s.fileMissing)
      .map(s => ({ name: s.filename, caption: datasetTrackCaption(s.audioPath, s.caption || ''), yue2: yue2Sidecar(s.audioPath),
        genre: s.genre || '', bpm: s.bpm === null || s.bpm === undefined ? '' : String(s.bpm), key: s.key || '' }))
      .filter(t => t.caption)
      .map(({ yue2, ...t }) => ({ ...t, styled: yue2 || yue2StyleString(t) }));
    return { datasetId: ds.id, datasetSlug: ds.slug, datasetName: ds.albumName || ds.name || ds.slug, tracks };
  } catch (err: any) {
    console.warn(`[Training] yue2-dataset-captions failed: ${err?.message || err}`);
    return empty;
  }
}
