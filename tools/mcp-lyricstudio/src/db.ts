// db.ts — Database access layer for mcp-lyricstudio
//
// Connects directly to hotstep.db via better-sqlite3.
// Query functions mirror those in server/src/db/lireekDb.ts.

import Database from 'better-sqlite3';
import path from 'path';

let db: Database.Database;

/**
 * Resolve the database path from environment or default.
 * Priority: HOTSTEP_DB env > default relative path.
 */
function resolveDbPath(): string {
  if (process.env.HOTSTEP_DB) {
    return path.resolve(process.env.HOTSTEP_DB);
  }
  // Default: relative to this package's location in tools/mcp-lyricstudio/
  return path.resolve(import.meta.dirname, '..', '..', 'server', 'data', 'hotstep.db');
}

export function initDb(): void {
  const dbPath = resolveDbPath();
  console.error(`[mcp-lyricstudio] Opening database: ${dbPath}`);
  db = new Database(dbPath);
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');
}

export function getDb(): Database.Database {
  if (!db) throw new Error('Database not initialized');
  return db;
}

// ── Artists ──────────────────────────────────────────────────────────────────

export function listArtists(): any[] {
  return getDb().prepare(
    `SELECT a.*, COUNT(ls.id) AS lyrics_set_count
     FROM artists a LEFT JOIN lyrics_sets ls ON ls.artist_id = a.id
     GROUP BY a.id ORDER BY a.name`
  ).all();
}

export function getArtist(id: number): any {
  return getDb().prepare('SELECT * FROM artists WHERE id = ?').get(id);
}

// ── Lyrics Sets ─────────────────────────────────────────────────────────────

export function getLyricsSet(id: number): any {
  const row = getDb().prepare(
    `SELECT ls.*, a.name as artist_name FROM lyrics_sets ls
     JOIN artists a ON a.id = ls.artist_id WHERE ls.id = ?`
  ).get(id) as any;
  if (!row) return null;
  row.songs = JSON.parse(row.songs);
  row.total_songs = row.songs.length;
  return row;
}

export function getLyricsSets(artistId?: number): any[] {
  const db = getDb();
  // profile_id is NULL when the set has never been profiled — this is what makes
  // "which artists still need a profile?" answerable without querying SQLite directly.
  const profileSubquery =
    `(SELECT p.id FROM profiles p WHERE p.lyrics_set_id = ls.id
      ORDER BY p.created_at DESC LIMIT 1) AS profile_id`;
  const query = artistId
    ? db.prepare(
        `SELECT ls.*, a.name as artist_name, ${profileSubquery} FROM lyrics_sets ls
         JOIN artists a ON a.id = ls.artist_id
         WHERE ls.artist_id = ? ORDER BY ls.fetched_at DESC`
      )
    : db.prepare(
        `SELECT ls.*, a.name as artist_name, ${profileSubquery} FROM lyrics_sets ls
         JOIN artists a ON a.id = ls.artist_id
         ORDER BY a.name, ls.fetched_at DESC`
      );
  const rows = (artistId ? query.all(artistId) : query.all()) as any[];
  return rows.map(r => {
    const songs = JSON.parse(r.songs);
    const { songs: _, ...rest } = r;
    return { ...rest, total_songs: songs.length };
  });
}

// ── Profiles ────────────────────────────────────────────────────────────────

export function listProfiles(artistId?: number): any[] {
  const db = getDb();
  let rows: any[];
  if (artistId) {
    rows = db.prepare(
      `SELECT p.*, ls.album, ls.artist_id, a.name as artist_name
       FROM profiles p
       JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
       JOIN artists a ON a.id = ls.artist_id
       WHERE ls.artist_id = ?
       ORDER BY p.created_at DESC`
    ).all(artistId);
  } else {
    rows = db.prepare(
      `SELECT p.*, ls.album, ls.artist_id, a.name as artist_name
       FROM profiles p
       JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
       JOIN artists a ON a.id = ls.artist_id
       ORDER BY a.name, p.created_at DESC`
    ).all();
  }
  return rows.map(r => ({
    ...r,
    profile_data: JSON.parse(r.profile_data),
  }));
}

export function getProfile(id: number): any {
  const row = getDb().prepare(
    `SELECT p.*, ls.album, ls.artist_id, a.name as artist_name
     FROM profiles p
     JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
     JOIN artists a ON a.id = ls.artist_id
     WHERE p.id = ?`
  ).get(id) as any;
  if (!row) return null;
  row.profile_data = JSON.parse(row.profile_data);
  return row;
}

export function saveProfile(
  lyricsSetId: number, provider: string, model: string, profileData: any
): any {
  const now = new Date().toISOString();
  const result = getDb().prepare(
    'INSERT INTO profiles (lyrics_set_id, provider, model, profile_data, created_at) VALUES (?, ?, ?, ?, ?)'
  ).run(lyricsSetId, provider, model, JSON.stringify(profileData), now);
  return {
    id: result.lastInsertRowid, lyrics_set_id: lyricsSetId,
    provider, model, profile_data: profileData, created_at: now,
  };
}

// ── Generations ─────────────────────────────────────────────────────────────

export interface SaveGenerationParams {
  profileId: number;
  provider: string;
  model: string;
  lyrics: string;
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

export function saveGeneration(p: SaveGenerationParams): any {
  const now = new Date().toISOString();
  const result = getDb().prepare(
    `INSERT INTO generations
     (profile_id, provider, model, extra_instructions, title, subject, bpm, key, caption, duration, lyrics, system_prompt, user_prompt, parent_generation_id, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(
    p.profileId, p.provider, p.model, null,
    p.title ?? '', p.subject ?? '', p.bpm ?? 0, p.key ?? '', p.caption ?? '', p.duration ?? 0,
    p.lyrics, p.systemPrompt ?? '', p.userPrompt ?? '', p.parentGenerationId ?? null, now,
  );
  return {
    id: result.lastInsertRowid, profile_id: p.profileId, provider: p.provider,
    model: p.model, title: p.title ?? '', subject: p.subject ?? '',
    bpm: p.bpm ?? 0, key: p.key ?? '', caption: p.caption ?? '', duration: p.duration ?? 0,
    lyrics: p.lyrics, parent_generation_id: p.parentGenerationId ?? null, created_at: now,
  };
}

export function getGeneration(id: number): any {
  return (getDb().prepare('SELECT * FROM generations WHERE id = ?').get(id) as any) ?? null;
}

export function listGenerations(profileId?: number, artistId?: number, limit = 20): any[] {
  const db = getDb();
  if (profileId) {
    return db.prepare(
      `SELECT g.*, a.name AS artist_name, ls.album
       FROM generations g
       JOIN profiles p ON p.id = g.profile_id
       JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
       JOIN artists a ON a.id = ls.artist_id
       WHERE g.profile_id = ?
       ORDER BY g.created_at DESC LIMIT ?`
    ).all(profileId, limit);
  }
  if (artistId) {
    return db.prepare(
      `SELECT g.*, a.name AS artist_name, ls.album
       FROM generations g
       JOIN profiles p ON p.id = g.profile_id
       JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
       JOIN artists a ON a.id = ls.artist_id
       WHERE ls.artist_id = ?
       ORDER BY g.created_at DESC LIMIT ?`
    ).all(artistId, limit);
  }
  return db.prepare(
    `SELECT g.*, a.name AS artist_name, ls.album
     FROM generations g
     JOIN profiles p ON p.id = g.profile_id
     JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
     JOIN artists a ON a.id = ls.artist_id
     ORDER BY g.created_at DESC LIMIT ?`
  ).all(limit);
}

/**
 * Get generation history for diversity tracking (used subjects, BPMs, keys, titles).
 */
export function getGenerationHistory(artistId: number): {
  usedSubjects: string[];
  usedBpms: number[];
  usedKeys: string[];
  usedTitles: string[];
  usedDurations: number[];
} {
  const rows = getDb().prepare(
    `SELECT g.subject, g.bpm, g.key, g.title, g.duration
     FROM generations g
     JOIN profiles p ON p.id = g.profile_id
     JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
     WHERE ls.artist_id = ?`
  ).all(artistId) as any[];

  return {
    usedSubjects: rows.map(r => r.subject).filter(Boolean),
    usedBpms: rows.map(r => r.bpm).filter((b: number) => b > 0),
    usedKeys: rows.map(r => r.key).filter(Boolean),
    usedTitles: rows.map(r => r.title).filter(Boolean),
    usedDurations: rows.map(r => r.duration).filter((d: number) => d > 0),
  };
}
