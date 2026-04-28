// lireekDb.ts — SQLite schema and queries for Lyric Studio / Lireek
//
// Separate database from hotstep.db to preserve portability.
// Uses better-sqlite3 for synchronous, fast access.

import Database from 'better-sqlite3';
import fs from 'fs';
import { config } from '../config.js';

let db: Database.Database;

export function getLireekDb(): Database.Database {
  if (!db) {
    throw new Error('Lireek database not initialized. Call initLireekDb() first.');
  }
  return db;
}

export function initLireekDb(): void {
  fs.mkdirSync(config.data.dir, { recursive: true });

  db = new Database(config.lireek.dbPath);

  // Performance pragmas
  db.pragma('journal_mode = WAL');
  db.pragma('synchronous = NORMAL');
  db.pragma('foreign_keys = ON');

  // Create core tables
  db.exec(`
    CREATE TABLE IF NOT EXISTS artists (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      name        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
      created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS lyrics_sets (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      artist_id   INTEGER NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
      album       TEXT,
      max_songs   INTEGER NOT NULL DEFAULT 10,
      songs       TEXT    NOT NULL,
      fetched_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS profiles (
      id              INTEGER PRIMARY KEY AUTOINCREMENT,
      lyrics_set_id   INTEGER NOT NULL REFERENCES lyrics_sets(id) ON DELETE CASCADE,
      provider        TEXT    NOT NULL,
      model           TEXT    NOT NULL,
      profile_data    TEXT    NOT NULL,
      created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS generations (
      id                  INTEGER PRIMARY KEY AUTOINCREMENT,
      profile_id          INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
      provider            TEXT    NOT NULL,
      model               TEXT    NOT NULL,
      extra_instructions  TEXT,
      title               TEXT    NOT NULL DEFAULT '',
      subject             TEXT    NOT NULL DEFAULT '',
      lyrics              TEXT    NOT NULL,
      system_prompt       TEXT    NOT NULL DEFAULT '',
      user_prompt         TEXT    NOT NULL DEFAULT '',
      created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS settings (
      key    TEXT PRIMARY KEY,
      value  TEXT NOT NULL
    );

    -- HOT-Step integration tables
    CREATE TABLE IF NOT EXISTS album_presets (
      id                    INTEGER PRIMARY KEY AUTOINCREMENT,
      lyrics_set_id         INTEGER NOT NULL REFERENCES lyrics_sets(id) ON DELETE CASCADE,
      adapter_path          TEXT,
      adapter_scale         REAL,
      adapter_group_scales  TEXT,
      reference_track_path  TEXT,
      audio_cover_strength  REAL,
      created_at            TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS audio_generations (
      id              INTEGER PRIMARY KEY AUTOINCREMENT,
      generation_id   INTEGER NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
      hotstep_job_id  TEXT NOT NULL,
      audio_url       TEXT,
      cover_url       TEXT,
      created_at      TEXT DEFAULT (datetime('now'))
    );
  `);

  // Migrations — add columns that may not exist in older databases
  const migrations = [
    "ALTER TABLE generations ADD COLUMN subject TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE generations ADD COLUMN title TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE generations ADD COLUMN system_prompt TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE generations ADD COLUMN user_prompt TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE generations ADD COLUMN bpm INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE generations ADD COLUMN key TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE generations ADD COLUMN caption TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE generations ADD COLUMN duration INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE generations ADD COLUMN parent_generation_id INTEGER REFERENCES generations(id) ON DELETE SET NULL",
    "ALTER TABLE artists ADD COLUMN image_url TEXT",
    "ALTER TABLE artists ADD COLUMN genius_id INTEGER",
    "ALTER TABLE lyrics_sets ADD COLUMN image_url TEXT",
  ];
  for (const sql of migrations) {
    try { db.exec(sql); } catch { /* column already exists */ }
  }

  console.log(`[LireekDB] Initialized: ${config.lireek.dbPath}`);
}

export function closeLireekDb(): void {
  if (db) {
    db.close();
    console.log('[LireekDB] Closed');
  }
}


// ── Artists ──────────────────────────────────────────────────────────────────

export function getOrCreateArtist(name: string): Record<string, any> {
  const existing = db.prepare(
    'SELECT * FROM artists WHERE name = ? COLLATE NOCASE'
  ).get(name) as any;
  if (existing) return existing;

  const now = new Date().toISOString();
  const result = db.prepare(
    'INSERT INTO artists (name, created_at) VALUES (?, ?)'
  ).run(name, now);
  return { id: result.lastInsertRowid, name, created_at: now, image_url: null, genius_id: null };
}

export function listArtists(): Record<string, any>[] {
  return db.prepare(
    `SELECT a.*, COUNT(ls.id) AS lyrics_set_count
     FROM artists a LEFT JOIN lyrics_sets ls ON ls.artist_id = a.id
     GROUP BY a.id ORDER BY a.name`
  ).all() as any[];
}

export function deleteArtist(id: number): boolean {
  const result = db.prepare('DELETE FROM artists WHERE id = ?').run(id);
  return result.changes > 0;
}

export function updateArtistImage(id: number, imageUrl: string | null): void {
  db.prepare('UPDATE artists SET image_url = ? WHERE id = ?').run(imageUrl, id);
}

export function updateArtistGeniusId(id: number, geniusId: number | null): void {
  db.prepare('UPDATE artists SET genius_id = ? WHERE id = ?').run(geniusId, id);
}

export function getArtist(id: number): Record<string, any> | undefined {
  return db.prepare('SELECT * FROM artists WHERE id = ?').get(id) as any;
}


// ── Lyrics Sets ─────────────────────────────────────────────────────────────

export function saveLyricsSet(
  artistId: number, album: string | null, maxSongs: number, songs: any[],
  imageUrl?: string | null,
): Record<string, any> {
  const now = new Date().toISOString();
  const songsJson = JSON.stringify(songs);
  const result = db.prepare(
    'INSERT INTO lyrics_sets (artist_id, album, max_songs, songs, image_url, fetched_at) VALUES (?, ?, ?, ?, ?, ?)'
  ).run(artistId, album, maxSongs, songsJson, imageUrl ?? null, now);
  return {
    id: result.lastInsertRowid, artist_id: artistId, album, max_songs: maxSongs,
    total_songs: songs.length, image_url: imageUrl ?? null, fetched_at: now,
  };
}

export function getLyricsSets(artistId?: number): Record<string, any>[] {
  const query = artistId
    ? db.prepare(
        `SELECT ls.*, a.name as artist_name FROM lyrics_sets ls
         JOIN artists a ON a.id = ls.artist_id
         WHERE ls.artist_id = ? ORDER BY ls.fetched_at DESC`
      )
    : db.prepare(
        `SELECT ls.*, a.name as artist_name FROM lyrics_sets ls
         JOIN artists a ON a.id = ls.artist_id
         ORDER BY ls.fetched_at DESC`
      );

  const rows = (artistId ? query.all(artistId) : query.all()) as any[];
  return rows.map(r => {
    const songs = JSON.parse(r.songs);
    const { songs: _, ...rest } = r;
    return { ...rest, total_songs: songs.length };
  });
}

export function getLyricsSet(id: number): Record<string, any> | null {
  const row = db.prepare(
    `SELECT ls.*, a.name as artist_name FROM lyrics_sets ls
     JOIN artists a ON a.id = ls.artist_id WHERE ls.id = ?`
  ).get(id) as any;
  if (!row) return null;
  row.songs = JSON.parse(row.songs);
  row.total_songs = row.songs.length;
  return row;
}

export function deleteLyricsSet(id: number): boolean {
  return db.prepare('DELETE FROM lyrics_sets WHERE id = ?').run(id).changes > 0;
}

export function removeSongFromSet(lyricsSetId: number, songIndex: number): Record<string, any> | null {
  const row = db.prepare('SELECT songs FROM lyrics_sets WHERE id = ?').get(lyricsSetId) as any;
  if (!row) return null;
  const songs = JSON.parse(row.songs);
  if (songIndex < 0 || songIndex >= songs.length) return null;
  songs.splice(songIndex, 1);
  db.prepare('UPDATE lyrics_sets SET songs = ? WHERE id = ?').run(JSON.stringify(songs), lyricsSetId);
  return getLyricsSet(lyricsSetId);
}

export function editSongInSet(lyricsSetId: number, songIndex: number, newLyrics: string): Record<string, any> | null {
  const row = db.prepare('SELECT songs FROM lyrics_sets WHERE id = ?').get(lyricsSetId) as any;
  if (!row) return null;
  const songs = JSON.parse(row.songs);
  if (songIndex < 0 || songIndex >= songs.length) return null;
  songs[songIndex].lyrics = newLyrics;
  db.prepare('UPDATE lyrics_sets SET songs = ? WHERE id = ?').run(JSON.stringify(songs), lyricsSetId);
  return getLyricsSet(lyricsSetId);
}

export function addSongToSet(lyricsSetId: number, song: { title: string; album?: string; lyrics: string }): Record<string, any> | null {
  const row = db.prepare('SELECT songs FROM lyrics_sets WHERE id = ?').get(lyricsSetId) as any;
  if (!row) return null;
  const songs = JSON.parse(row.songs);
  songs.push(song);
  db.prepare('UPDATE lyrics_sets SET songs = ? WHERE id = ?').run(JSON.stringify(songs), lyricsSetId);
  return getLyricsSet(lyricsSetId);
}

export function updateLyricsSetImage(id: number, imageUrl: string | null): void {
  db.prepare('UPDATE lyrics_sets SET image_url = ? WHERE id = ?').run(imageUrl, id);
}


// ── Profiles ────────────────────────────────────────────────────────────────

export function saveProfile(
  lyricsSetId: number, provider: string, model: string, profileData: any,
): Record<string, any> {
  const now = new Date().toISOString();
  const result = db.prepare(
    'INSERT INTO profiles (lyrics_set_id, provider, model, profile_data, created_at) VALUES (?, ?, ?, ?, ?)'
  ).run(lyricsSetId, provider, model, JSON.stringify(profileData), now);
  return {
    id: result.lastInsertRowid, lyrics_set_id: lyricsSetId,
    provider, model, profile_data: profileData, created_at: now,
  };
}

export function getProfiles(lyricsSetId?: number): Record<string, any>[] {
  const query = lyricsSetId
    ? db.prepare('SELECT * FROM profiles WHERE lyrics_set_id = ? ORDER BY created_at DESC')
    : db.prepare('SELECT * FROM profiles ORDER BY created_at DESC');
  const rows = (lyricsSetId ? query.all(lyricsSetId) : query.all()) as any[];
  return rows.map(r => ({ ...r, profile_data: JSON.parse(r.profile_data) }));
}

export function getProfile(id: number): Record<string, any> | null {
  const row = db.prepare('SELECT * FROM profiles WHERE id = ?').get(id) as any;
  if (!row) return null;
  row.profile_data = JSON.parse(row.profile_data);
  return row;
}

export function deleteProfile(id: number): boolean {
  return db.prepare('DELETE FROM profiles WHERE id = ?').run(id).changes > 0;
}

export function updateProfileData(id: number, profileData: any): void {
  db.prepare('UPDATE profiles SET profile_data = ? WHERE id = ?').run(JSON.stringify(profileData), id);
}


// ── Generations ─────────────────────────────────────────────────────────────

export interface SaveGenerationParams {
  profileId: number;
  provider: string;
  model: string;
  lyrics: string;
  extraInstructions?: string;
  title?: string;
  subject?: string;
  bpm?: number;
  key?: string;
  caption?: string;
  duration?: number;
  systemPrompt?: string;
  userPrompt?: string;
  parentGenerationId?: number | null;
}

export function saveGeneration(p: SaveGenerationParams): Record<string, any> {
  const now = new Date().toISOString();
  const result = db.prepare(
    `INSERT INTO generations
     (profile_id, provider, model, extra_instructions, title, subject, bpm, key, caption, duration, lyrics, system_prompt, user_prompt, parent_generation_id, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(
    p.profileId, p.provider, p.model, p.extraInstructions ?? null,
    p.title ?? '', p.subject ?? '', p.bpm ?? 0, p.key ?? '', p.caption ?? '', p.duration ?? 0,
    p.lyrics, p.systemPrompt ?? '', p.userPrompt ?? '', p.parentGenerationId ?? null, now,
  );
  return {
    id: result.lastInsertRowid, profile_id: p.profileId, provider: p.provider,
    model: p.model, extra_instructions: p.extraInstructions ?? null,
    title: p.title ?? '', subject: p.subject ?? '', bpm: p.bpm ?? 0,
    key: p.key ?? '', caption: p.caption ?? '', duration: p.duration ?? 0,
    lyrics: p.lyrics, system_prompt: p.systemPrompt ?? '', user_prompt: p.userPrompt ?? '',
    parent_generation_id: p.parentGenerationId ?? null, created_at: now,
  };
}

export function getGenerations(profileId?: number, lyricsSetId?: number): Record<string, any>[] {
  if (profileId) {
    return db.prepare('SELECT * FROM generations WHERE profile_id = ? ORDER BY created_at DESC').all(profileId) as any[];
  }
  if (lyricsSetId) {
    return db.prepare(
      `SELECT g.* FROM generations g
       JOIN profiles p ON p.id = g.profile_id
       WHERE p.lyrics_set_id = ? ORDER BY g.created_at DESC`
    ).all(lyricsSetId) as any[];
  }
  return db.prepare('SELECT * FROM generations ORDER BY created_at DESC').all() as any[];
}

export function getAllGenerationsWithContext(): Record<string, any>[] {
  return db.prepare(
    `SELECT g.*, a.name AS artist_name, ls.album, ls.artist_id
     FROM generations g
     JOIN profiles p ON p.id = g.profile_id
     JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
     JOIN artists a ON a.id = ls.artist_id
     ORDER BY g.created_at DESC`
  ).all() as any[];
}

export function getGeneration(id: number): Record<string, any> | null {
  return (db.prepare('SELECT * FROM generations WHERE id = ?').get(id) as any) ?? null;
}

export function updateGenerationMetadata(
  id: number, bpm: number, key: string, caption: string, duration: number = 0,
): void {
  db.prepare(
    'UPDATE generations SET bpm = ?, key = ?, caption = ?, duration = ? WHERE id = ?'
  ).run(bpm, key, caption, duration, id);
}

export function updateGenerationFields(id: number, fields: Record<string, any>): void {
  const allowed = ['title', 'subject', 'lyrics', 'bpm', 'key', 'caption', 'duration', 'extra_instructions'];
  const sets: string[] = [];
  const values: any[] = [];
  for (const [k, v] of Object.entries(fields)) {
    if (allowed.includes(k)) {
      sets.push(`${k} = ?`);
      values.push(v);
    }
  }
  if (sets.length === 0) return;
  values.push(id);
  db.prepare(`UPDATE generations SET ${sets.join(', ')} WHERE id = ?`).run(...values);
}

export function deleteGeneration(id: number): boolean {
  return db.prepare('DELETE FROM generations WHERE id = ?').run(id).changes > 0;
}

export function purgeProfilesAndGenerations(): { generations_deleted: number; profiles_deleted: number } {
  const genResult = db.prepare('DELETE FROM generations').run();
  const profResult = db.prepare('DELETE FROM profiles').run();
  return { generations_deleted: genResult.changes, profiles_deleted: profResult.changes };
}


// ── Settings ────────────────────────────────────────────────────────────────

export function getSetting(key: string, defaultValue = ''): string {
  const row = db.prepare('SELECT value FROM settings WHERE key = ?').get(key) as any;
  return row?.value ?? defaultValue;
}

export function setSetting(key: string, value: string): void {
  db.prepare(
    'INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value'
  ).run(key, value);
}


// ── Album Presets ───────────────────────────────────────────────────────────

export function getPreset(lyricsSetId: number): Record<string, any> | null {
  return (db.prepare('SELECT * FROM album_presets WHERE lyrics_set_id = ?').get(lyricsSetId) as any) ?? null;
}

export function getAllPresets(): Record<string, any>[] {
  return db.prepare(
    `SELECT ap.*, a.name as artist_name, ls.album, ls.artist_id
     FROM album_presets ap
     JOIN lyrics_sets ls ON ls.id = ap.lyrics_set_id
     JOIN artists a ON a.id = ls.artist_id
     ORDER BY ap.created_at DESC`
  ).all() as any[];
}

export function upsertPreset(lyricsSetId: number, data: {
  adapterPath?: string | null;
  adapterScale?: number | null;
  adapterGroupScales?: any;
  referenceTrackPath?: string | null;
  audioCoverStrength?: number | null;
}): Record<string, any> {
  const existing = getPreset(lyricsSetId);
  const groupScalesJson = data.adapterGroupScales ? JSON.stringify(data.adapterGroupScales) : null;

  if (existing) {
    db.prepare(
      `UPDATE album_presets SET adapter_path = ?, adapter_scale = ?, adapter_group_scales = ?,
       reference_track_path = ?, audio_cover_strength = ? WHERE lyrics_set_id = ?`
    ).run(
      data.adapterPath ?? null, data.adapterScale ?? null, groupScalesJson,
      data.referenceTrackPath ?? null, data.audioCoverStrength ?? null, lyricsSetId,
    );
  } else {
    db.prepare(
      `INSERT INTO album_presets (lyrics_set_id, adapter_path, adapter_scale, adapter_group_scales, reference_track_path, audio_cover_strength)
       VALUES (?, ?, ?, ?, ?, ?)`
    ).run(
      lyricsSetId, data.adapterPath ?? null, data.adapterScale ?? null, groupScalesJson,
      data.referenceTrackPath ?? null, data.audioCoverStrength ?? null,
    );
  }
  return getPreset(lyricsSetId)!;
}

export function deletePreset(lyricsSetId: number): boolean {
  return db.prepare('DELETE FROM album_presets WHERE lyrics_set_id = ?').run(lyricsSetId).changes > 0;
}


// ── Audio Generations ───────────────────────────────────────────────────────

export function linkAudioGeneration(generationId: number, jobId: string): Record<string, any> {
  const now = new Date().toISOString();
  const result = db.prepare(
    'INSERT INTO audio_generations (generation_id, hotstep_job_id, created_at) VALUES (?, ?, ?)'
  ).run(generationId, jobId, now);
  return { id: result.lastInsertRowid, generation_id: generationId, hotstep_job_id: jobId, created_at: now };
}

export function getAudioGenerations(generationId: number): Record<string, any>[] {
  return db.prepare(
    'SELECT * FROM audio_generations WHERE generation_id = ? ORDER BY created_at DESC'
  ).all(generationId) as any[];
}

export function resolveAudioGeneration(jobId: string, audioUrl: string, coverUrl?: string): void {
  db.prepare(
    'UPDATE audio_generations SET audio_url = ?, cover_url = ? WHERE hotstep_job_id = ?'
  ).run(audioUrl, coverUrl ?? null, jobId);
}

export function deleteAudioGeneration(id: number): boolean {
  return db.prepare('DELETE FROM audio_generations WHERE id = ?').run(id).changes > 0;
}

/** Delete audio_generations rows matching the given hotstep job IDs (used when songs are deleted from the main library). */
export function deleteAudioGenerationsByJobIds(jobIds: string[]): number {
  if (jobIds.length === 0) return 0;
  const placeholders = jobIds.map(() => '?').join(',');
  return db.prepare(`DELETE FROM audio_generations WHERE hotstep_job_id IN (${placeholders})`).run(...jobIds).changes;
}

export function getRecentGenerationsWithAudio(limit = 50): Record<string, any>[] {
  return db.prepare(
    `SELECT g.title AS song_title, g.subject, g.caption, g.lyrics, g.duration,
       g.created_at AS ag_created_at, g.id AS generation_id,
       a.name AS artist_name, a.image_url AS artist_image, a.id AS artist_id,
       ls.album, ls.id AS lyrics_set_id,
       ag.id AS ag_id, ag.audio_url, ag.cover_url, ag.hotstep_job_id
     FROM audio_generations ag
     JOIN generations g ON g.id = ag.generation_id
     JOIN profiles p ON p.id = g.profile_id
     JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
     JOIN artists a ON a.id = ls.artist_id
     WHERE ag.audio_url IS NOT NULL
     ORDER BY ag.created_at DESC
     LIMIT ?`
  ).all(limit) as any[];
}
