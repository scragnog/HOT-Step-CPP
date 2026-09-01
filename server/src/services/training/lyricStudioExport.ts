// training/lyricStudioExport.ts — dataset → Lyric Studio artist/album export
//
// A labeled dataset already carries everything a lireek artist/album entry
// needs: Genius lyrics, LLM caption+genre, Essentia bpm/key/signature, plus
// embedded audio tags. This service turns one dataset into (or refreshes) one
// lyrics_set, enriched per song, and optionally wires the dataset's trained
// adapters AND one of its source tracks (the timbre/mastering reference) into
// the album preset so generation is one click away.
//
// Artist/album detection is a majority vote over the embedded tags — the most
// reliable source we have and already scanned into every sample — falling back
// to the dataset defaults, then the source folder name. The caller always gets
// the vote back first (preview) and can override both names before committing.
//
// An existing artist+album pair is UPDATED IN PLACE (Rob, 2026-07-29): the
// song list is replaced wholesale, never duplicated into a second set.

import fs from 'fs';
import path from 'path';
import { config } from '../../config.js';
import {
  findArtistByName, findLyricsSetByAlbum, getAllPresets, getOrCreateArtist, getPreset,
  replaceLyricsSetSongs, saveLyricsSet, updateArtistImage, upsertPreset,
} from '../../db/lireekDb.js';
import { getAlbumImageUrl, getArtistImageUrl } from '../lireek/geniusService.js';
import { resolveArtistTitle } from './enhanceService.js';
import * as audioMeta from './audioMeta.js';
import { updateDataset } from './datasetsRepo.js';
import { findDitAdapter, findLmAdapter } from './datasetAssets.js';
import type {
  LyricStudioExportInput, LyricStudioExportPreview,
  LyricStudioExportResult, LyricStudioExportSong, LyricStudioFieldSource,
  TrainingDatasetRow, TrainingSample,
} from './types.js';

// ── Detection ────────────────────────────────────────────────────────────

/** Most common non-empty value, case-insensitively grouped, first-seen casing
 *  returned. null when nothing usable at all. */
function majority(values: string[]): string | null {
  const counts = new Map<string, { display: string; n: number }>();
  for (const raw of values) {
    const v = String(raw ?? '').trim();
    if (!v) continue;
    const k = v.toLowerCase();
    const hit = counts.get(k);
    if (hit) hit.n++;
    else counts.set(k, { display: v, n: 1 });
  }
  let best: { display: string; n: number } | null = null;
  for (const c of counts.values()) if (!best || c.n > best.n) best = c;
  return best ? best.display : null;
}

function detectArtist(
  ds: TrainingDatasetRow, included: TrainingSample[],
): { value: string; source: LyricStudioFieldSource } {
  const byTags = majority(included.map(s => s.tagArtist));
  if (byTags) return { value: byTags, source: 'tags' };
  if (ds.defaultArtist.trim()) return { value: ds.defaultArtist.trim(), source: 'default' };
  const byName = majority(included.map(s => resolveArtistTitle(s, ds, {}).artist));
  if (byName) return { value: byName, source: 'filename' };
  return { value: ds.name, source: 'dataset-name' };
}

function detectAlbum(
  ds: TrainingDatasetRow, included: TrainingSample[],
): { value: string; source: LyricStudioFieldSource } {
  const byTags = majority(included.map(s => s.tagAlbum));
  if (byTags) return { value: byTags, source: 'tags' };
  if (ds.defaultAlbum.trim()) return { value: ds.defaultAlbum.trim(), source: 'default' };
  const folder = path.basename(ds.sourceDir).trim();
  if (folder) return { value: folder, source: 'folder' };
  return { value: ds.name, source: 'dataset-name' };
}

/**
 * Fill in tagArtist/tagAlbum from the AUDIO FILES when the label store has none.
 *
 * buildSamples reads those two off the label record (datasetScan.ts:250) and
 * only the Label job ever writes it. A pipeline run that skips Label therefore
 * arrives here with every tag blank, and detectAlbum — which has no filename
 * fallback, unlike detectArtist — drops all the way through to the SOURCE
 * FOLDER NAME. That is how a bulk run published "4yearstrong_someway" as a new
 * album alongside the existing "In Some Way, Shape Or Form" instead of merging
 * into it (2026-07-31): the existing-album lookup was working fine, it was just
 * being handed a name no album could ever match.
 *
 * The tags are sitting in the files either way, so read them. Only the fields
 * that are actually empty are filled, so a labelled dataset is untouched and
 * the label store stays authoritative where it has an opinion.
 *
 * Bounded and best-effort: a dataset only needs enough of a majority to name an
 * artist and an album, and an unreadable file must never fail an export.
 */
const TAG_PROBE_LIMIT = 24;

async function fillTagsFromFiles(included: TrainingSample[]): Promise<void> {
  const needs = included.filter(s => !s.tagArtist.trim() || !s.tagAlbum.trim());
  if (needs.length === 0) return;
  for (const s of needs.slice(0, TAG_PROBE_LIMIT)) {
    try {
      const md = await audioMeta.read(s.audioPath);
      if (!s.tagArtist.trim() && md.artist) s.tagArtist = md.artist;
      if (!s.tagAlbum.trim() && md.album) s.tagAlbum = md.album;
      if (!s.tagTitle.trim() && md.title) s.tagTitle = md.title;
    } catch { /* unreadable file — the existing fallbacks still apply */ }
  }
}

// ── Song assembly ────────────────────────────────────────────────────────

interface SongSplit {
  included: TrainingSample[];
  skippedInstrumental: number;
  skippedNoLyrics: number;
  skippedExcluded: number;
}

function splitSamples(samples: TrainingSample[]): SongSplit {
  const out: SongSplit = { included: [], skippedInstrumental: 0, skippedNoLyrics: 0, skippedExcluded: 0 };
  for (const s of samples) {
    if (s.excluded || s.fileMissing) { out.skippedExcluded++; continue; }
    if (s.isInstrumental) { out.skippedInstrumental++; continue; }
    if (!s.lyrics.trim()) { out.skippedNoLyrics++; continue; }
    out.included.push(s);
  }
  return out;
}

/** The enriched song object stored in lyrics_sets.songs. Extra fields beyond
 *  {title, album, lyrics} are additive — every existing consumer reads only
 *  those three (profilerService, prompts) and ignores the rest. */
function toStoredSong(s: TrainingSample, ds: TrainingDatasetRow, setAlbum: string): Record<string, any> {
  const song: Record<string, any> = {
    title: resolveArtistTitle(s, ds, {}).title || s.filename.replace(/\.[^.]+$/, ''),
    album: s.tagAlbum.trim() || setAlbum,
    lyrics: s.lyrics,
  };
  if (s.caption.trim()) song.caption = s.caption.trim();
  // The MM3 Structured Caption lives beside the audio (<stem>.mm3.txt), not in
  // the sidecar, so the sample object never carries it — read it here or lose
  // it. It is the second of the two caption formats a MOSS labeling run
  // writes, and Lyric Studio is where Rob reviews source tracks.
  try {
    const mm3Path = `${s.audioPath.replace(/\.[^.\\/]+$/, '')}.mm3.txt`;
    if (fs.existsSync(mm3Path)) {
      const mm3 = fs.readFileSync(mm3Path, 'utf8').trim();
      if (mm3) song.mm3Caption = mm3;
    }
  } catch { /* a caption we cannot read is a caption we do not export */ }
  if (s.genre.trim()) song.genre = s.genre.trim();
  if (typeof s.bpm === 'number' && Number.isFinite(s.bpm) && s.bpm > 0) song.bpm = s.bpm;
  if (s.key.trim()) song.key = s.key.trim();
  if (s.signature.trim()) song.signature = s.signature.trim();
  if (s.language.trim()) song.language = s.language.trim();
  // Duration is what lets Lyric Studio measure THIS artist's vocal pacing
  // (words/sec) — per-artist medians span 0.51-3.29 w/s, and without it the
  // planner falls back to a global constant that misprices most artists.
  if (typeof s.duration === 'number' && Number.isFinite(s.duration) && s.duration > 0) {
    song.duration = Math.round(s.duration * 100) / 100;
  }
  return song;
}

// ── Reference track ────────────────────────────────────────────────

/** Longest track the engine will take as reference audio, in seconds.
 *  /synth 413s at MAX_T_LATENT frames — 600 s at 48 kHz — and
 *  generation/sourceAudio.ts enforces the same bound at render time. Mirrored
 *  (not imported) because that module pulls in the whole mastering route. */
const REFERENCE_MAX_SECONDS = 598;

/**
 * One of the dataset's own tracks, to hang on the album preset as the timbre /
 * mastering reference.
 *
 * Any track off the album is a valid answer — the point is that the preset
 * carries the artist's real recorded sound, not that one particular song does.
 * Two preferences make the automatic pick less likely to be a bad one:
 *
 *  - vocal tracks over instrumentals, because vocal timbre is most of what the
 *    reference is being asked for;
 *  - the MEDIAN duration rather than the first track, because the front of an
 *    album's file order is where intros, skits and 40-second interludes live.
 *
 * Over-long tracks are skipped outright: the engine rejects reference audio at
 * 600 s and a 21-minute closer would fail every render that rolled onto it.
 *
 * Note this also seeds "Randomize timbre reference", which re-rolls across the
 * WHOLE folder the picked file sits in (sourceAudio.ts) — so the pick doubles
 * as pointing the preset at this dataset's audio.
 */
export function pickReferenceTrack(samples: TrainingSample[]): string | null {
  const usable = samples.filter(s =>
    !s.excluded && !s.fileMissing
    && !(typeof s.duration === 'number' && s.duration >= REFERENCE_MAX_SECONDS));
  const pool = usable.filter(s => !s.isInstrumental);
  const candidates = (pool.length ? pool : usable)
    .slice()
    .sort((a, b) => (a.duration || 0) - (b.duration || 0) || a.relPath.localeCompare(b.relPath));
  if (!candidates.length) return null;
  return candidates[Math.floor((candidates.length - 1) / 2)].audioPath;
}

// ── Public API ───────────────────────────────────────────────────────────

export async function previewLyricStudioExport(
  ds: TrainingDatasetRow, samples: TrainingSample[],
): Promise<LyricStudioExportPreview> {
  const split = splitSamples(samples);
  // Before detection, not after: detectAlbum's folder-name fallback is
  // indistinguishable from a real answer once it has been taken.
  await fillTagsFromFiles(split.included);
  const artist = detectArtist(ds, split.included);
  const album = detectAlbum(ds, split.included);

  const songs: LyricStudioExportSong[] = split.included.map(s => ({
    sampleId: s.sampleId,
    title: resolveArtistTitle(s, ds, {}).title || s.filename.replace(/\.[^.]+$/, ''),
    album: s.tagAlbum.trim() || album.value,
    hasCaption: !!s.caption.trim(),
  }));

  const existingArtist = findArtistByName(artist.value);
  const existingSet = existingArtist ? findLyricsSetByAlbum(existingArtist.id, album.value) : null;

  return {
    artist: artist.value,
    album: album.value,
    artistSource: artist.source,
    albumSource: album.source,
    songs,
    skippedInstrumental: split.skippedInstrumental,
    skippedNoLyrics: split.skippedNoLyrics,
    skippedExcluded: split.skippedExcluded,
    existingArtistId: existingArtist ? existingArtist.id : null,
    existingLyricsSetId: existingSet ? existingSet.id : null,
    existingSongCount: existingSet ? existingSet.total_songs : 0,
    ditAdapter: findDitAdapter(ds),
    lmAdapter: findLmAdapter(ds),
    // Whole sample list, not split.included: a reference track needs audio on
    // disk, not lyrics, so instrumentals and un-Geniused tracks still count.
    referenceTrack: pickReferenceTrack(samples),
    geniusConfigured: !!config.lireek.geniusAccessToken,
  };
}

export class LyricStudioExportError extends Error {}

/** An album_presets row mapped back into upsertPreset's input — upsertPreset
 *  overwrites EVERY column, so any partial update must round-trip the rest. */
function presetDataFromRow(row: Record<string, any> | null): {
  adapterPath: string | null; adapterScale: number | null; adapterGroupScales: any;
  referenceTrackPath: string | null; audioCoverStrength: number | null;
  lmAdapterPath: string | null; lmAdapterScale: number | null;
} {
  let groupScales: any = row?.adapter_group_scales ?? null;
  if (typeof groupScales === 'string') {
    try { groupScales = JSON.parse(groupScales); } catch { groupScales = null; }
  }
  return {
    adapterPath: row?.adapter_path ?? null,
    adapterScale: row?.adapter_scale ?? null,
    adapterGroupScales: groupScales,
    referenceTrackPath: row?.reference_track_path ?? null,
    audioCoverStrength: row?.audio_cover_strength ?? null,
    lmAdapterPath: row?.lm_adapter_path ?? null,
    lmAdapterScale: row?.lm_adapter_scale ?? null,
  };
}

/** Windows-safe path identity for run-dir comparison. */
function normPath(p: string): string {
  return path.resolve(String(p ?? '').replace(/[\\/]+$/, '')).toLowerCase();
}

/**
 * A training run just exported into `newRunDir` (`<adapters>/…/<artist>/<stamp>`).
 * Point every Lyric Studio album preset that references an OLDER run of the same
 * artist folder — or the unversioned artist folder itself — at the new run, so
 * presets always play the most recent training (Rob, 2026-07-29). Presets
 * referencing other adapters entirely are never touched. Returns the number of
 * presets updated; never throws (callers run inside a training job's success
 * path where a lireek hiccup must not fail the run).
 */
export function refreshPresetsForNewRun(newRunDir: string, kind: 'dit' | 'lm'): number {
  let updated = 0;
  try {
    // A run whose stages skipped 'export' (or whose export failed upstream)
    // has no weights — never point a preset at an empty dir.
    const hasWeights = ['adapter_model.safetensors', 'lokr_weights.safetensors']
      .some(f => fs.existsSync(path.join(newRunDir, f)));
    if (!hasWeights) return 0;
    const newDir = normPath(newRunDir);
    const artistDir = path.dirname(newDir);
    const artistName = path.basename(artistDir);
    const column = kind === 'dit' ? 'adapter_path' : 'lm_adapter_path';
    const otherColumn = kind === 'dit' ? 'lm_adapter_path' : 'adapter_path';
    for (const preset of getAllPresets()) {
      const stored = preset[column];
      let retarget = false;
      if (stored && typeof stored === 'string') {
        const storedNorm = normPath(stored);
        if (storedNorm === newDir) continue;                   // already current
        retarget = path.dirname(storedNorm) === artistDir      // older stamped run
          || storedNorm === artistDir;                         // unversioned legacy dir
      } else {
        // Column empty — e.g. the album was exported after LM training but
        // before any DiT run existed. If the preset's OTHER column already
        // points at this same artist's adapter, this run belongs to the same
        // album: fill the gap so the preset carries both adapters.
        const other = preset[otherColumn];
        if (other && typeof other === 'string') {
          const otherNorm = normPath(other);
          retarget = path.basename(path.dirname(otherNorm)) === artistName
            || path.basename(otherNorm) === artistName;
        }
      }
      if (!retarget) continue;
      const data = presetDataFromRow(preset);
      if (kind === 'dit') data.adapterPath = newRunDir;
      else data.lmAdapterPath = newRunDir;
      upsertPreset(preset.lyrics_set_id, data);
      updated++;
      console.log(`[Training] Album preset (lyrics set ${preset.lyrics_set_id}) ${column} → ${newRunDir}`);
    }
  } catch (err: any) {
    console.warn(`[Training] Preset refresh after training failed: ${err?.message ?? err}`);
  }
  return updated;
}

export async function commitLyricStudioExport(
  ds: TrainingDatasetRow, samples: TrainingSample[], input: LyricStudioExportInput,
): Promise<LyricStudioExportResult> {
  const split = splitSamples(samples);
  if (!split.included.length) {
    throw new LyricStudioExportError('No exportable songs — every sample is excluded, instrumental, or has no lyrics.');
  }

  // Only when the caller did NOT name both: the manual path posts explicit
  // artist/album from the preview the user confirmed, and probing files behind
  // an explicit choice would be pure latency.
  if (!(input.artist ?? '').trim() || !(input.album ?? '').trim()) {
    await fillTagsFromFiles(split.included);
  }
  const artistName = (input.artist ?? '').trim() || detectArtist(ds, split.included).value;
  const albumName = (input.album ?? '').trim() || detectAlbum(ds, split.included).value;

  const songs = split.included.map(s => toStoredSong(s, ds, albumName));

  // Album art first: both the insert and the update paths take it, and a
  // Genius miss (or no token) must never fail the export.
  let imageUrl = '';
  if (config.lireek.geniusAccessToken) {
    try { imageUrl = (await getAlbumImageUrl(albumName, artistName)) ?? ''; }
    catch { /* art stays '' */ }
  }

  const artist = getOrCreateArtist(artistName);
  const existingSet = findLyricsSetByAlbum(artist.id, albumName);

  let lyricsSetId: number;
  if (existingSet) {
    replaceLyricsSetSongs(existingSet.id, albumName, songs, imageUrl || null);
    lyricsSetId = existingSet.id;
  } else {
    const created = saveLyricsSet(artist.id, albumName, songs.length, songs, imageUrl || null);
    lyricsSetId = Number(created.id);
  }

  // Persist the dataset → album link on the row (Rob, 2026-08-13): the
  // audition's Lyric Studio prompt source reads it without re-detecting.
  try { updateDataset(ds.id, { lyricsSetId }); } catch { /* link is a convenience — never fail the export */ }

  if (!artist.image_url && config.lireek.geniusAccessToken) {
    try {
      const artistImage = await getArtistImageUrl(artistName);
      if (artistImage) updateArtistImage(artist.id, artistImage);
    } catch { /* portrait stays empty */ }
  }

  // Album preset: wire the dataset's trained adapters and one of its own
  // tracks in, preserving whatever the user already dialed in (upsertPreset
  // overwrites EVERY column, so an existing preset must be read back and
  // merged, not patched blind).
  let presetUpdated = false;
  if (input.linkAdapters !== false) {
    const dit = findDitAdapter(ds);
    const lm = findLmAdapter(ds);
    const data = presetDataFromRow(getPreset(lyricsSetId));
    // A reference the user chose by hand outranks our pick; one pointing at
    // a file that has since gone does not.
    const refStale = !data.referenceTrackPath
      || !fs.existsSync(data.referenceTrackPath);
    const ref = refStale ? pickReferenceTrack(samples) : null;
    if (dit || lm || ref) {
      if (dit) data.adapterPath = dit.path;
      if (lm) data.lmAdapterPath = lm.path;
      if (ref) data.referenceTrackPath = ref;
      upsertPreset(lyricsSetId, data);
      presetUpdated = true;
    }
  }

  console.log(
    `[Training] Exported '${ds.name}' → Lyric Studio: ${artistName} / ${albumName} ` +
    `(${songs.length} songs, ${existingSet ? 'updated in place' : 'new set'}${presetUpdated ? ', preset linked' : ''})`,
  );

  return {
    artistId: artist.id,
    lyricsSetId,
    updatedExisting: !!existingSet,
    songCount: songs.length,
    presetUpdated,
    imageUrl,
  };
}
