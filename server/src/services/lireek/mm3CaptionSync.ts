/**
 * mm3CaptionSync.ts — keep a lyrics set's source-song MM3 and YuE2 captions in
 * step with the training dataset the set was exported from.
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
 *
 * It runs inside BOTH getLyricsSet functions (server lireekDb.ts and the MCP
 * server's db.ts), so it takes the caller's SQLite handle and imports nothing
 * from the server's DB layer. 2026-09-28: it used to run only in two UI read
 * routes, so the MCP tools and in-app generation read whatever text the last
 * UI visit had cached — the YuE2 "house dialect" still quoted a re-captioned
 * track's old sidecar.
 */

import fs from 'fs';
import path from 'path';
import type Database from 'better-sqlite3';

function norm(s: unknown): string {
  return String(s ?? '').replace(/\s+/g, ' ').trim().toLowerCase();
}

/** Refreshes `set.songs` (already parsed to an array) in place. Persists when
 *  any caption changed. Never throws: a set with no linked dataset, or a
 *  dataset whose files are unreadable, is left untouched. */
export function refreshSidecarCaptions(db: Database.Database, set: Record<string, any>): void {
  try {
    const setId = Number(set?.id);
    if (!setId || !Array.isArray(set.songs)) return;
    const ds = db.prepare(
      'SELECT slug, source_dir, dataset_json_path FROM training_datasets WHERE lyrics_set_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1',
    ).get(setId) as { slug: string; source_dir: string; dataset_json_path: string } | undefined;
    if (!ds?.dataset_json_path || !fs.existsSync(ds.dataset_json_path)) return;

    const manifest = JSON.parse(fs.readFileSync(ds.dataset_json_path, 'utf-8'));
    const samples: any[] = manifest?.samples ?? manifest?.tracks ?? [];
    if (!Array.isArray(samples) || samples.length === 0) return;
    const dir = path.dirname(ds.dataset_json_path);

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
          path.join(ds.source_dir || dir, stem + suffix),
          path.join(dir, stem + suffix),
        ].filter(Boolean);
        const file = candidates.find(p => fs.existsSync(p));
        if (!file) continue;
        byLyrics.set(key, { ...(byLyrics.get(key) ?? {}), [field]: file });
      }
    }
    if (byLyrics.size === 0) return;

    let changed = false;
    for (const song of set.songs) {
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
      db.prepare('UPDATE lyrics_sets SET songs = ? WHERE id = ?').run(JSON.stringify(set.songs), setId);
      console.error(`[Lireek] MM3/YuE2 captions refreshed from ${ds.slug} for lyrics set ${setId}`);
    }
  } catch (err: any) {
    console.warn(`[Lireek] MM3 caption refresh skipped: ${err?.message || err}`);
  }
}
