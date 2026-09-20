/**
 * mm3CaptionSync.ts — keep a lyrics set's source-song MM3 captions in step
 * with the training dataset the set was exported from.
 *
 * Why this exists (2026-09-09): a written song renders on MiniMax-Music3 under
 * one of the album's own training captions by default (utils/mm3CaptionSource
 * in the UI), because a training caption with new lyrics is what reliably
 * lands in the band's style and reaches a natural ending. The captions the UI
 * can offer come from the lyrics set's stored songs, and those were copied
 * from `<stem>.mm3.txt` ONCE, at export time. Every set exported before that
 * copy existed (August) holds none, and every set exported before a dataset
 * was re-captioned holds the old text. The album R set showed "No source track
 * on this album has an MM3 caption" while fourteen Gemini captions sat on
 * disk beside the audio.
 *
 * So the captions are refreshed from disk whenever a set is read. The dataset
 * is found through `training_datasets.lyrics_set_id` (the export writes it),
 * songs are matched to samples by their lyrics text (the export copied it
 * verbatim; 14/14 matched on the set that prompted this), and a song whose
 * caption file changed gets the new text. Cheap: one JSON read and a few
 * small file reads per set, and nothing is written unless something changed.
 */

import fs from 'fs';
import path from 'path';
import { listDatasets } from '../training/datasetsRepo.js';
import { updateLyricsSetSongs } from '../../db/lireekDb.js';

function norm(s: unknown): string {
  return String(s ?? '').replace(/\s+/g, ' ').trim().toLowerCase();
}

/** Returns the set with `songs` refreshed (parsed to an array). Persists when
 *  any caption changed. Never throws: a set with no linked dataset, or a
 *  dataset whose files are unreadable, comes back untouched. */
export function refreshMm3CaptionsFromDataset(set: Record<string, any>): Record<string, any> {
  try {
    const setId = Number(set?.id);
    if (!setId) return set;
    const ds = listDatasets().find(d => Number(d.lyricsSetId) === setId);
    if (!ds || !ds.datasetJsonPath || !fs.existsSync(ds.datasetJsonPath)) return set;

    const manifest = JSON.parse(fs.readFileSync(ds.datasetJsonPath, 'utf-8'));
    const samples: any[] = manifest?.samples ?? manifest?.tracks ?? [];
    if (!Array.isArray(samples) || samples.length === 0) return set;
    const dir = path.dirname(ds.datasetJsonPath);

    // lyrics text -> caption file path, for every sample that has one. Two
    // sidecar formats live beside the audio: <stem>.mm3.txt (MM3) and
    // <stem>.yue2.txt (the one-sentence YuE2 planner caption, 2026-09-20).
    const SIDECARS: Array<{ suffix: string; field: string }> = [
      { suffix: '.mm3.txt', field: 'mm3Caption' },
      { suffix: '.yue2.txt', field: 'yue2Caption' },
    ];
    const byLyrics = new Map<string, Partial<Record<string, string>>>();
    for (const s of samples) {
      const audio = String(s?.audio_path ?? s?.audioPath ?? '');
      const filename = String(s?.filename ?? path.basename(audio));
      if (!filename) continue;
      const stem = filename.replace(/\.[^.]+$/, '');
      const key = norm(s?.lyrics) || norm(s?.raw_lyrics);
      if (!key) continue;
      for (const { suffix, field } of SIDECARS) {
        const candidates = [
          audio ? audio.replace(/\.[^.\\/]+$/, '') + suffix : '',
          path.join(ds.sourceDir || dir, stem + suffix),
          path.join(dir, stem + suffix),
        ].filter(Boolean);
        const file = candidates.find(p => fs.existsSync(p));
        if (!file) continue;
        byLyrics.set(key, { ...(byLyrics.get(key) ?? {}), [field]: file });
      }
    }
    if (byLyrics.size === 0) return set;

    const songs: any[] = Array.isArray(set.songs) ? set.songs
      : (typeof set.songs === 'string' ? JSON.parse(set.songs) : []);
    let changed = false;
    for (const song of songs) {
      const files = byLyrics.get(norm(song?.lyrics));
      if (!files) continue;
      for (const [field, file] of Object.entries(files)) {
        if (!file) continue;
        let text = '';
        try { text = fs.readFileSync(file, 'utf-8').replace(/^﻿/, '').trim(); } catch { continue; }
        if (text && text !== (song[field] || '')) {
          song[field] = text;
          changed = true;
        }
      }
    }
    if (changed) {
      updateLyricsSetSongs(setId, songs);
      console.log(`[Lireek] MM3/YuE2 captions refreshed from ${ds.slug} for lyrics set ${setId}`);
    }
    return { ...set, songs };
  } catch (err: any) {
    console.warn(`[Lireek] MM3 caption refresh skipped: ${err?.message || err}`);
    return set;
  }
}
