// database.ts — SQLite schema and queries for HOT-Step CPP
//
// Uses better-sqlite3 for synchronous, fast SQLite access.
// Schema covers: users, songs (Phase 1), playlists, lyric-studio (future).

import Database from 'better-sqlite3';
import fs from 'fs';
import path from 'path';
import { config } from '../config.js';

let db: Database.Database;

export function getDb(): Database.Database {
  if (!db) {
    throw new Error('Database not initialized. Call initDb() first.');
  }
  return db;
}

export function initDb(): void {
  // Ensure data directory exists
  fs.mkdirSync(config.data.dir, { recursive: true });
  fs.mkdirSync(config.data.audioDir, { recursive: true });

  db = new Database(config.data.dbPath);

  // Performance pragmas
  db.pragma('journal_mode = WAL');
  db.pragma('synchronous = NORMAL');
  db.pragma('foreign_keys = ON');

  // Create tables
  db.exec(`
    -- Users (simplified: single-user local app)
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      username TEXT NOT NULL UNIQUE,
      bio TEXT DEFAULT '',
      avatar_url TEXT DEFAULT '',
      banner_url TEXT DEFAULT '',
      created_at TEXT DEFAULT (datetime('now'))
    );

    -- Songs
    CREATE TABLE IF NOT EXISTS songs (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL REFERENCES users(id),
      title TEXT NOT NULL DEFAULT 'Untitled',
      lyrics TEXT DEFAULT '',
      style TEXT DEFAULT '',
      caption TEXT DEFAULT '',
      audio_url TEXT DEFAULT '',
      cover_url TEXT DEFAULT '',
      duration REAL DEFAULT 0,
      bpm INTEGER DEFAULT 0,
      key_scale TEXT DEFAULT '',
      time_signature TEXT DEFAULT '',
      tags TEXT DEFAULT '[]',
      is_public INTEGER DEFAULT 0,
      like_count INTEGER DEFAULT 0,
      view_count INTEGER DEFAULT 0,
      dit_model TEXT DEFAULT '',
      generation_params TEXT DEFAULT '{}',
      mastered_audio_url TEXT DEFAULT '',
      created_at TEXT DEFAULT (datetime('now'))
    );

    -- Playlists (Phase 3)
    CREATE TABLE IF NOT EXISTS playlists (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL REFERENCES users(id),
      name TEXT NOT NULL,
      description TEXT DEFAULT '',
      cover_url TEXT DEFAULT '',
      is_public INTEGER DEFAULT 0,
      created_at TEXT DEFAULT (datetime('now'))
    );

    -- Playlist-Song junction
    CREATE TABLE IF NOT EXISTS playlist_songs (
      playlist_id TEXT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
      song_id TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
      position INTEGER DEFAULT 0,
      added_at TEXT DEFAULT (datetime('now')),
      PRIMARY KEY (playlist_id, song_id)
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_songs_user ON songs(user_id);
    CREATE INDEX IF NOT EXISTS idx_songs_created ON songs(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_playlists_user ON playlists(user_id);
  `);

  console.log(`[DB] Initialized: ${config.data.dbPath}`);
}

export function closeDb(): void {
  if (db) {
    db.close();
    console.log('[DB] Closed');
  }
}
