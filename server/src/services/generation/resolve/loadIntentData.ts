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

/** The reads, injectable so tests can see which source each intent reaches. */
export interface IntentDataSources {
  getGeneration: (id: number) => Record<string, any> | null;
  getPreset: (lyricsSetId: number) => Record<string, any> | null;
  getLyricsSet: (id: number) => Record<string, any> | null;
  yue2DatasetCaptions: typeof yue2DatasetCaptions;
}

export const defaultIntentDataSources: IntentDataSources = {
  getGeneration: id => lireekDb.getGeneration(id),
  getPreset: id => lireekDb.getPreset(id),
  getLyricsSet: id => lireekDb.getLyricsSet(id),
  yue2DatasetCaptions,
};

function mm3TracksForLyricsSet(src: IntentDataSources, lyricsSetId: number | undefined) {
  if (!lyricsSetId) return [];
  const set = src.getLyricsSet(lyricsSetId);
  const songs = set ? (typeof set.songs === 'string' ? JSON.parse(set.songs) : set.songs || []) : [];
  return collectMm3SourceTracks(songs);
}

/** Create: the caption source's tracks, only for the engine it names. YuE2 by
 *  dataset id; MM3 by the album's lyrics set, else the tracks handed over. */
export async function loadCreateData(
  intent: CreateIntent, engine: string, src: IntentDataSources = defaultIntentDataSources,
): Promise<CreateData> {
  const cs = intent.captionSource;
  if (cs?.engine === YUE2_ENGINE_ID && engine === YUE2_ENGINE_ID && cs.datasetId) {
    return { yue2Tracks: (await src.yue2DatasetCaptions({ dataset: cs.datasetId })).tracks };
  }
  if (cs?.engine === MM3_ENGINE_ID && engine === MM3_ENGINE_ID) {
    return { mm3Tracks: cs.lyricsSetId ? mm3TracksForLyricsSet(src, cs.lyricsSetId) : cs.tracks ?? [] };
  }
  return {};
}

/** Written song: the song row, and the preset and caption sources of the album
 *  it renders AS (`lyricsSetId`), never of its own album when they differ. */
export async function loadWrittenSongData(
  intent: WrittenSongIntent, engine: string, src: IntentDataSources = defaultIntentDataSources,
): Promise<WrittenSongData> {
  const gen = src.getGeneration(intent.generationId);
  if (!gen) throw new IntentDataError(404, `Generation ${intent.generationId} not found`);
  const preset = intent.lyricsSetId ? src.getPreset(intent.lyricsSetId) : null;
  const data: WrittenSongData = { gen, preset };
  if (engine === MM3_ENGINE_ID) data.mm3Tracks = mm3TracksForLyricsSet(src, intent.lyricsSetId);
  if (engine === YUE2_ENGINE_ID) {
    const ds = intent.lyricsSetId ? await src.yue2DatasetCaptions({ lyricsSet: intent.lyricsSetId }) : null;
    data.yue2Dataset = { datasetId: ds?.datasetId ?? '', tracks: ds?.tracks ?? [] };
  }
  return data;
}
