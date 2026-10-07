// resolve/loadIntentData.ts — read the data an intent resolves against: the
// song row, the album preset and the caption source tracks. The same reads the
// browser makes through the API today (lyrics set, preset, YuE2 dataset
// captions), done directly.

import type { CreateIntent, WrittenSongIntent } from '../../../contracts/resolution.js';
import * as lireekDb from '../../../db/lireekDb.js';
import { yue2DatasetCaptions } from '../../training/yue2DatasetCaptions.js';
import { collectMm3SourceTracks, MM3_ENGINE_ID, YUE2_ENGINE_ID } from './captionSource.js';
import type { CreateData, WrittenSongData } from './resolveIntent.js';

export class IntentDataError extends Error {
  constructor(readonly status: 400 | 404, message: string) { super(message); }
}

function mm3TracksForLyricsSet(lyricsSetId: number | undefined) {
  if (!lyricsSetId) return [];
  const set = lireekDb.getLyricsSet(lyricsSetId);
  const songs = set ? (typeof set.songs === 'string' ? JSON.parse(set.songs) : set.songs || []) : [];
  return collectMm3SourceTracks(songs);
}

export async function loadCreateData(intent: CreateIntent, engine: string): Promise<CreateData> {
  const src = intent.captionSource;
  if (src?.engine === YUE2_ENGINE_ID && engine === YUE2_ENGINE_ID && src.datasetId) {
    return { yue2Tracks: (await yue2DatasetCaptions({ dataset: src.datasetId })).tracks };
  }
  if (src?.engine === MM3_ENGINE_ID && engine === MM3_ENGINE_ID) {
    return { mm3Tracks: src.lyricsSetId ? mm3TracksForLyricsSet(src.lyricsSetId) : src.tracks ?? [] };
  }
  return {};
}

export async function loadWrittenSongData(intent: WrittenSongIntent, engine: string): Promise<WrittenSongData> {
  const gen = lireekDb.getGeneration(intent.generationId);
  if (!gen) throw new IntentDataError(404, `Generation ${intent.generationId} not found`);
  const preset = intent.lyricsSetId ? lireekDb.getPreset(intent.lyricsSetId) : null;
  const data: WrittenSongData = { gen, preset };
  if (engine === MM3_ENGINE_ID) data.mm3Tracks = mm3TracksForLyricsSet(intent.lyricsSetId);
  if (engine === YUE2_ENGINE_ID) {
    const ds = intent.lyricsSetId ? await yue2DatasetCaptions({ lyricsSet: intent.lyricsSetId }) : null;
    data.yue2Dataset = { datasetId: ds?.datasetId ?? '', tracks: ds?.tracks ?? [] };
  }
  return data;
}
