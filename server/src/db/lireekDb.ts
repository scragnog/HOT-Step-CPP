// lireekDb.ts — Query functions for Lyric Studio / Lireek tables
//
// These tables now live in the unified hotstep.db (previously lireek.db).
// All functions use getDb() from database.ts — there is no separate connection.

import { getDb } from './database.js';

// ── Legacy exports (no-ops, kept for compatibility during transition) ────────
// initLireekDb/closeLireekDb are no longer needed — the tables are created
// and migrated in database.ts initDb(). These are kept temporarily so callers
// that import them don't break at compile time.
/** @deprecated Tables are now in hotstep.db — no separate init needed */
export function initLireekDb(): void {
  // No-op — tables created in initDb()
}
/** @deprecated Tables are now in hotstep.db — no separate close needed */
export function closeLireekDb(): void {
  // No-op — closed by closeDb()
}
/** @deprecated Use getDb() directly */
export function getLireekDb(): ReturnType<typeof getDb> {
  return getDb();
}


// ── Artists ──────────────────────────────────────────────────────────────────

export function getOrCreateArtist(name: string): Record<string, any> {
  const db = getDb();
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
  return getDb().prepare(
    `SELECT a.*, COUNT(ls.id) AS lyrics_set_count
     FROM artists a LEFT JOIN lyrics_sets ls ON ls.artist_id = a.id
     GROUP BY a.id ORDER BY a.name`
  ).all() as any[];
}

export function deleteArtist(id: number): boolean {
  const result = getDb().prepare('DELETE FROM artists WHERE id = ?').run(id);
  return result.changes > 0;
}

export function updateArtistImage(id: number, imageUrl: string | null): void {
  getDb().prepare('UPDATE artists SET image_url = ? WHERE id = ?').run(imageUrl, id);
}

export function updateArtistGeniusId(id: number, geniusId: number | null): void {
  getDb().prepare('UPDATE artists SET genius_id = ? WHERE id = ?').run(geniusId, id);
}

export function getArtist(id: number): Record<string, any> | undefined {
  return getDb().prepare('SELECT * FROM artists WHERE id = ?').get(id) as any;
}


// ── Lyrics Sets ─────────────────────────────────────────────────────────────

export function saveLyricsSet(
  artistId: number, album: string | null, maxSongs: number, songs: any[],
  imageUrl?: string | null,
): Record<string, any> {
  const now = new Date().toISOString();
  const songsJson = JSON.stringify(songs);
  const result = getDb().prepare(
    'INSERT INTO lyrics_sets (artist_id, album, max_songs, songs, image_url, fetched_at) VALUES (?, ?, ?, ?, ?, ?)'
  ).run(artistId, album, maxSongs, songsJson, imageUrl ?? null, now);
  return {
    id: result.lastInsertRowid, artist_id: artistId, album, max_songs: maxSongs,
    total_songs: songs.length, image_url: imageUrl ?? null, fetched_at: now,
  };
}

export function getLyricsSets(artistId?: number): Record<string, any>[] {
  const db = getDb();
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
  const row = getDb().prepare(
    `SELECT ls.*, a.name as artist_name FROM lyrics_sets ls
     JOIN artists a ON a.id = ls.artist_id WHERE ls.id = ?`
  ).get(id) as any;
  if (!row) return null;
  row.songs = JSON.parse(row.songs);
  row.total_songs = row.songs.length;
  return row;
}

export function deleteLyricsSet(id: number): boolean {
  return getDb().prepare('DELETE FROM lyrics_sets WHERE id = ?').run(id).changes > 0;
}

export function removeSongFromSet(lyricsSetId: number, songIndex: number): Record<string, any> | null {
  const db = getDb();
  const row = db.prepare('SELECT songs FROM lyrics_sets WHERE id = ?').get(lyricsSetId) as any;
  if (!row) return null;
  const songs = JSON.parse(row.songs);
  if (songIndex < 0 || songIndex >= songs.length) return null;
  songs.splice(songIndex, 1);
  db.prepare('UPDATE lyrics_sets SET songs = ? WHERE id = ?').run(JSON.stringify(songs), lyricsSetId);
  return getLyricsSet(lyricsSetId);
}

export function editSongInSet(lyricsSetId: number, songIndex: number, newLyrics: string): Record<string, any> | null {
  const db = getDb();
  const row = db.prepare('SELECT songs FROM lyrics_sets WHERE id = ?').get(lyricsSetId) as any;
  if (!row) return null;
  const songs = JSON.parse(row.songs);
  if (songIndex < 0 || songIndex >= songs.length) return null;
  songs[songIndex].lyrics = newLyrics;
  db.prepare('UPDATE lyrics_sets SET songs = ? WHERE id = ?').run(JSON.stringify(songs), lyricsSetId);
  return getLyricsSet(lyricsSetId);
}

export function addSongToSet(lyricsSetId: number, song: { title: string; album?: string; lyrics: string }): Record<string, any> | null {
  const db = getDb();
  const row = db.prepare('SELECT songs FROM lyrics_sets WHERE id = ?').get(lyricsSetId) as any;
  if (!row) return null;
  const songs = JSON.parse(row.songs);
  songs.push(song);
  db.prepare('UPDATE lyrics_sets SET songs = ? WHERE id = ?').run(JSON.stringify(songs), lyricsSetId);
  return getLyricsSet(lyricsSetId);
}

export function updateLyricsSetImage(id: number, imageUrl: string | null): void {
  getDb().prepare('UPDATE lyrics_sets SET image_url = ? WHERE id = ?').run(imageUrl, id);
}


// ── Profiles ────────────────────────────────────────────────────────────────

export function saveProfile(
  lyricsSetId: number, provider: string, model: string, profileData: any,
): Record<string, any> {
  const now = new Date().toISOString();
  const result = getDb().prepare(
    'INSERT INTO profiles (lyrics_set_id, provider, model, profile_data, created_at) VALUES (?, ?, ?, ?, ?)'
  ).run(lyricsSetId, provider, model, JSON.stringify(profileData), now);
  return {
    id: result.lastInsertRowid, lyrics_set_id: lyricsSetId,
    provider, model, profile_data: profileData, created_at: now,
  };
}

export function getProfiles(lyricsSetId?: number): Record<string, any>[] {
  const db = getDb();
  const query = lyricsSetId
    ? db.prepare('SELECT * FROM profiles WHERE lyrics_set_id = ? ORDER BY created_at DESC')
    : db.prepare('SELECT * FROM profiles ORDER BY created_at DESC');
  const rows = (lyricsSetId ? query.all(lyricsSetId) : query.all()) as any[];
  return rows.map(r => ({ ...r, profile_data: JSON.parse(r.profile_data) }));
}

export function getProfile(id: number): Record<string, any> | null {
  const row = getDb().prepare('SELECT * FROM profiles WHERE id = ?').get(id) as any;
  if (!row) return null;
  row.profile_data = JSON.parse(row.profile_data);
  return row;
}

export function deleteProfile(id: number): boolean {
  return getDb().prepare('DELETE FROM profiles WHERE id = ?').run(id).changes > 0;
}

export function updateProfileData(id: number, profileData: any): void {
  getDb().prepare('UPDATE profiles SET profile_data = ? WHERE id = ?').run(JSON.stringify(profileData), id);
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
  const result = getDb().prepare(
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
  const db = getDb();
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
  return getDb().prepare(
    `SELECT g.*, a.name AS artist_name, ls.album, ls.artist_id
     FROM generations g
     JOIN profiles p ON p.id = g.profile_id
     JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
     JOIN artists a ON a.id = ls.artist_id
     ORDER BY g.created_at DESC`
  ).all() as any[];
}

export function getGeneration(id: number): Record<string, any> | null {
  return (getDb().prepare('SELECT * FROM generations WHERE id = ?').get(id) as any) ?? null;
}

export function updateGenerationMetadata(
  id: number, bpm: number, key: string, caption: string, duration: number = 0,
): void {
  getDb().prepare(
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
  getDb().prepare(`UPDATE generations SET ${sets.join(', ')} WHERE id = ?`).run(...values);
}

export function deleteGeneration(id: number): boolean {
  return getDb().prepare('DELETE FROM generations WHERE id = ?').run(id).changes > 0;
}

export function purgeProfilesAndGenerations(): { generations_deleted: number; profiles_deleted: number } {
  const db = getDb();
  const genResult = db.prepare('DELETE FROM generations').run();
  const profResult = db.prepare('DELETE FROM profiles').run();
  return { generations_deleted: genResult.changes, profiles_deleted: profResult.changes };
}

export function purgeGenerationsOnly(): { generations_deleted: number } {
  const result = getDb().prepare('DELETE FROM generations').run();
  return { generations_deleted: result.changes };
}

export function purgeProfilesOnly(): { profiles_deleted: number; generations_deleted: number } {
  const db = getDb();
  // Generations depend on profiles via FK, so delete generations first
  const genResult = db.prepare('DELETE FROM generations').run();
  const profResult = db.prepare('DELETE FROM profiles').run();
  return { profiles_deleted: profResult.changes, generations_deleted: genResult.changes };
}


// ── Settings ────────────────────────────────────────────────────────────────

export function getSetting(key: string, defaultValue = ''): string {
  const row = getDb().prepare('SELECT value FROM settings WHERE key = ?').get(key) as any;
  return row?.value ?? defaultValue;
}

export function setSetting(key: string, value: string): void {
  getDb().prepare(
    'INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value'
  ).run(key, value);
}


// ── Album Presets ───────────────────────────────────────────────────────────

export function getPreset(lyricsSetId: number): Record<string, any> | null {
  return (getDb().prepare('SELECT * FROM album_presets WHERE lyrics_set_id = ?').get(lyricsSetId) as any) ?? null;
}

export function getAllPresets(): Record<string, any>[] {
  return getDb().prepare(
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
  const db = getDb();
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
  return getDb().prepare('DELETE FROM album_presets WHERE lyrics_set_id = ?').run(lyricsSetId).changes > 0;
}


// ── Audio Generations ───────────────────────────────────────────────────────

export function linkAudioGeneration(generationId: number, jobId: string): Record<string, any> {
  const now = new Date().toISOString();
  const result = getDb().prepare(
    'INSERT INTO audio_generations (generation_id, hotstep_job_id, created_at) VALUES (?, ?, ?)'
  ).run(generationId, jobId, now);
  return { id: result.lastInsertRowid, generation_id: generationId, hotstep_job_id: jobId, created_at: now };
}

export function getAudioGenerations(generationId: number): Record<string, any>[] {
  return getDb().prepare(
    `SELECT ag.*, s.mastered_audio_url
     FROM audio_generations ag
     LEFT JOIN songs s ON s.audio_url = ag.audio_url
     WHERE ag.generation_id = ? ORDER BY ag.created_at DESC`
  ).all(generationId) as any[];
}

export function resolveAudioGeneration(jobId: string, audioUrl: string, coverUrl?: string): void {
  getDb().prepare(
    'UPDATE audio_generations SET audio_url = ?, cover_url = ? WHERE hotstep_job_id = ?'
  ).run(audioUrl, coverUrl ?? null, jobId);
}

export function deleteAudioGeneration(id: number): boolean {
  return getDb().prepare('DELETE FROM audio_generations WHERE id = ?').run(id).changes > 0;
}

/** Delete audio_generations rows matching the given hotstep job IDs (used when songs are deleted from the main library). */
export function deleteAudioGenerationsByJobIds(jobIds: string[]): number {
  if (jobIds.length === 0) return 0;
  const placeholders = jobIds.map(() => '?').join(',');
  return getDb().prepare(`DELETE FROM audio_generations WHERE hotstep_job_id IN (${placeholders})`).run(...jobIds).changes;
}

export function getRecentGenerationsWithAudio(limit = 50): Record<string, any>[] {
  return getDb().prepare(
    `SELECT g.title AS song_title, g.subject, g.caption, g.lyrics, g.duration,
       g.created_at AS ag_created_at, g.id AS generation_id,
       a.name AS artist_name, a.image_url AS artist_image, a.id AS artist_id,
       ls.album, ls.id AS lyrics_set_id,
       ag.id AS ag_id, ag.audio_url, ag.cover_url, ag.hotstep_job_id,
       s.mastered_audio_url
     FROM audio_generations ag
     JOIN generations g ON g.id = ag.generation_id
     JOIN profiles p ON p.id = g.profile_id
     JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
     JOIN artists a ON a.id = ls.artist_id
     LEFT JOIN songs s ON s.audio_url = ag.audio_url
     WHERE ag.audio_url IS NOT NULL
     ORDER BY ag.created_at DESC
     LIMIT ?`
  ).all(limit) as any[];
}
