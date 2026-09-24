// database.ts — Unified SQLite schema for HOT-Step CPP
//
// Uses better-sqlite3 for synchronous, fast SQLite access.
// Schema covers: users, songs, playlists, AND lireek/lyric-studio tables.
//
// Previously the lireek tables lived in a separate lireek.db file.
// As of v2 they are consolidated into hotstep.db for simpler queries,
// unified recent-songs endpoints, and less cross-DB gymnastics.

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

  // ── Core HOT-Step tables ──────────────────────────────────────────────────
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
      backend TEXT DEFAULT 'ace',
      created_at TEXT DEFAULT (datetime('now'))
    );

    -- Playlists
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

    -- Indexes (core)
    CREATE INDEX IF NOT EXISTS idx_songs_user ON songs(user_id);
    CREATE INDEX IF NOT EXISTS idx_songs_created ON songs(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_playlists_user ON playlists(user_id);
  `);

  // ── Lireek / Lyric Studio tables (consolidated from lireek.db) ────────────
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
      lm_adapter_path       TEXT,
      lm_adapter_scale      REAL,
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

    -- ── Song Builder (Udio-style section-by-section generation) ──────────────
    -- A project is one song being assembled from ordered sections. Each section
    -- generates N candidate songs (variants); the user picks one (chosen_song_id)
    -- and the next section outpaint-extends from the chosen variant's latent.
    CREATE TABLE IF NOT EXISTS builder_projects (
      id              TEXT PRIMARY KEY,
      user_id         TEXT NOT NULL REFERENCES users(id),
      title           TEXT NOT NULL DEFAULT 'Untitled Song',
      -- Shared musical params reused across every section
      style           TEXT DEFAULT '',
      bpm             INTEGER DEFAULT 0,
      key_scale       TEXT DEFAULT '',
      time_signature  TEXT DEFAULT '',
      vocal_language  TEXT DEFAULT '',
      -- Default seconds per generated section (user-overridable per section)
      section_length  REAL DEFAULT 30,
      -- Default number of variants generated per section
      variant_count   INTEGER DEFAULT 4,
      gen_params      TEXT DEFAULT '{}',
      created_at      TEXT DEFAULT (datetime('now')),
      updated_at      TEXT DEFAULT (datetime('now'))
    );

    -- One row per committed/in-progress section. position orders sections along
    -- the timeline (may be negative or fractional to allow prepend/insert without
    -- renumbering). candidate_song_ids is a JSON array of song ids (the variants);
    -- chosen_song_id is the committed pick (NULL until the user chooses).
    CREATE TABLE IF NOT EXISTS builder_sections (
      id                 TEXT PRIMARY KEY,
      project_id         TEXT NOT NULL REFERENCES builder_projects(id) ON DELETE CASCADE,
      position           REAL NOT NULL DEFAULT 0,
      label              TEXT DEFAULT '',
      lyrics             TEXT DEFAULT '',
      direction          TEXT DEFAULT 'append',  -- 'first' | 'append' | 'prepend'
      section_length     REAL DEFAULT 30,
      candidate_song_ids TEXT DEFAULT '[]',
      chosen_song_id     TEXT,
      job_id             TEXT,
      status             TEXT DEFAULT 'pending',  -- 'pending' | 'generating' | 'ready' | 'chosen' | 'failed'
      created_at         TEXT DEFAULT (datetime('now')),
      updated_at         TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_builder_projects_user ON builder_projects(user_id);
    CREATE INDEX IF NOT EXISTS idx_builder_sections_project ON builder_sections(project_id, position);

    -- Training datasets (Dataset Studio). Disk is the source of truth for
    -- sample data; this row exists for listing, status and settings only.
    CREATE TABLE IF NOT EXISTS training_datasets (
      id                TEXT PRIMARY KEY,
      slug              TEXT NOT NULL UNIQUE,
      name              TEXT NOT NULL DEFAULT 'Untitled',
      source_dir        TEXT NOT NULL,
      recursive         INTEGER NOT NULL DEFAULT 1,
      custom_tag        TEXT NOT NULL DEFAULT '',
      tag_position      TEXT NOT NULL DEFAULT 'prepend',
      genre_ratio       INTEGER NOT NULL DEFAULT 0,
      default_artist    TEXT NOT NULL DEFAULT '',
      default_album     TEXT NOT NULL DEFAULT '',
      default_genre     TEXT NOT NULL DEFAULT '',
      default_language  TEXT NOT NULL DEFAULT 'english',
      sample_count      INTEGER NOT NULL DEFAULT 0,
      labeled_count     INTEGER NOT NULL DEFAULT 0,
      excluded_count    INTEGER NOT NULL DEFAULT 0,
      status            TEXT NOT NULL DEFAULT 'draft',
      built_at          TEXT NOT NULL DEFAULT '',
      dataset_json_path TEXT NOT NULL DEFAULT '',
      album_name        TEXT NOT NULL DEFAULT '',
      created_at        TEXT DEFAULT (datetime('now')),
      updated_at        TEXT DEFAULT (datetime('now'))
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_training_datasets_src ON training_datasets(source_dir);

    -- YuE2 refinement rung scores (Refine tab): Rob's ear judgement per rung
    -- checkpoint beside the rung's facts, for cross-artist analysis later.
    CREATE TABLE IF NOT EXISTS yue2_rung_scores (
      id             INTEGER PRIMARY KEY AUTOINCREMENT,
      dataset_id     TEXT NOT NULL,
      dataset_slug   TEXT NOT NULL,
      source_run     TEXT NOT NULL DEFAULT '',
      refine_run     TEXT NOT NULL,
      checkpoint_dir TEXT NOT NULL UNIQUE,
      step           INTEGER NOT NULL,
      kl             REAL,
      recon          REAL,
      drift          REAL,
      rung           INTEGER NOT NULL DEFAULT 0,
      frozen         INTEGER NOT NULL DEFAULT 0,
      settings       TEXT NOT NULL DEFAULT '{}',
      previews       TEXT NOT NULL DEFAULT '[]',
      metrics        TEXT NOT NULL DEFAULT '{}',
      likeness       INTEGER,
      corruption     INTEGER,
      notes          TEXT NOT NULL DEFAULT '',
      created_at     TEXT DEFAULT (datetime('now')),
      updated_at     TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_yue2_rung_scores_ds ON yue2_rung_scores(dataset_id, refine_run, step);
  `);

  // ── Migrations — add columns that may not exist in older databases ────────
  // Training datasets migrations
  const trainingMigrations: Array<{ check: string; alter: string }> = [
    {
      check: `SELECT COUNT(*) as c FROM pragma_table_info('training_datasets') WHERE name='default_language'`,
      alter: `ALTER TABLE training_datasets ADD COLUMN default_language TEXT NOT NULL DEFAULT 'english'`,
    },
    // Friendly album name detected from the tracks' embedded tags. Cached here
    // because the dataset LIST must not parse audio files on every request —
    // datasetAssets.ts fills it once, then refreshes it whenever a scan has the
    // samples in hand.
    {
      check: `SELECT COUNT(*) as c FROM pragma_table_info('training_datasets') WHERE name='album_name'`,
      alter: `ALTER TABLE training_datasets ADD COLUMN album_name TEXT NOT NULL DEFAULT ''`,
    },
    // Persistent dataset → Lyric Studio link, written by the export commit and
    // lazily backfilled when the audition's Lyric Studio prompt source resolves
    // the album by detection. 0 = never linked.
    {
      check: `SELECT COUNT(*) as c FROM pragma_table_info('training_datasets') WHERE name='lyrics_set_id'`,
      alter: `ALTER TABLE training_datasets ADD COLUMN lyrics_set_id INTEGER NOT NULL DEFAULT 0`,
    },
  ];
  for (const m of trainingMigrations) {
    const row = db.prepare(m.check).get() as { c: number };
    if (row.c === 0) db.exec(m.alter);
  }

  // Songs table migrations
  const songsMigrations: Array<{ check: string; alter: string }> = [
    {
      check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='mastered_audio_url'`,
      alter: `ALTER TABLE songs ADD COLUMN mastered_audio_url TEXT DEFAULT ''`,
    },
    {
      check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='latent_url'`,
      alter: `ALTER TABLE songs ADD COLUMN latent_url TEXT DEFAULT ''`,
    },
    {
      check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='quality_scores'`,
      alter: `ALTER TABLE songs ADD COLUMN quality_scores TEXT DEFAULT ''`,
    },
    {
      check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='cover_art_subject'`,
      alter: `ALTER TABLE songs ADD COLUMN cover_art_subject TEXT DEFAULT ''`,
    },
    {
      check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='kick_stem_url'`,
      alter: `ALTER TABLE songs ADD COLUMN kick_stem_url TEXT DEFAULT ''`,
    },
    {
      check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='snare_stem_url'`,
      alter: `ALTER TABLE songs ADD COLUMN snare_stem_url TEXT DEFAULT ''`,
    },
    {
      check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='hihat_stem_url'`,
      alter: `ALTER TABLE songs ADD COLUMN hihat_stem_url TEXT DEFAULT ''`,
    },
    {
      check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='disco_data_url'`,
      alter: `ALTER TABLE songs ADD COLUMN disco_data_url TEXT DEFAULT ''`,
    },
    {
      // User-edited embed-tag overrides (JSON: { artist, album, year, comment }).
      // Used verbatim by gatherSongMetadata when set — see metadata editor (#60).
      check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='metadata_overrides'`,
      alter: `ALTER TABLE songs ADD COLUMN metadata_overrides TEXT DEFAULT ''`,
    },
    {
      // No-adapter reference render: low-step bare-DiT output (adapter bypassed,
      // no post-processing) for A/B-ing what the DiT adapter contributes.
      check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='noadapter_audio_url'`,
      alter: `ALTER TABLE songs ADD COLUMN noadapter_audio_url TEXT DEFAULT ''`,
    },
    {
      // Which generation backend produced this song ('ace', future: 'minimax-m3', ...).
      // See docs/plans/multi-backend-architecture.md §3.5.
      check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='backend'`,
      alter: `ALTER TABLE songs ADD COLUMN backend TEXT DEFAULT 'ace'`,
    },
  ];
  for (const m of songsMigrations) {
    const row = db.prepare(m.check).get() as any;
    if (row.c === 0) {
      db.exec(m.alter);
      console.log(`[DB] Migration: ${m.alter}`);
    }
  }

  // Lireek table migrations (same as previously in lireekDb.ts)
  const lireekMigrations = [
    "ALTER TABLE generations ADD COLUMN subject TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE generations ADD COLUMN title TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE generations ADD COLUMN system_prompt TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE generations ADD COLUMN user_prompt TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE generations ADD COLUMN bpm INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE generations ADD COLUMN key TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE generations ADD COLUMN caption TEXT NOT NULL DEFAULT ''",
    // MiniMax-Music3 Structured Caption — a SECOND caption in MM3's own
    // three-heading format, not a reformatting of `caption`. The two backends
    // want genuinely different text and neither works in the other's slot.
    "ALTER TABLE generations ADD COLUMN caption_mm3 TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE generations ADD COLUMN duration INTEGER NOT NULL DEFAULT 0",
    // YuE2 planner caption (2026-09-20): ONE sentence in the fixed order
    // language → genre → vocal → instruments → mood → production → BPM, the
    // shape the planner's own training captions take. A third caption, not a
    // trimming of `caption`.
    "ALTER TABLE generations ADD COLUMN caption_yue2 TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE generations ADD COLUMN parent_generation_id INTEGER REFERENCES generations(id) ON DELETE SET NULL",
    "ALTER TABLE artists ADD COLUMN image_url TEXT",
    "ALTER TABLE artists ADD COLUMN genius_id INTEGER",
    "ALTER TABLE lyrics_sets ADD COLUMN image_url TEXT",
    // Planner-LM adapter per album (local HOT-Step feature)
    "ALTER TABLE album_presets ADD COLUMN lm_adapter_path TEXT",
    "ALTER TABLE album_presets ADD COLUMN lm_adapter_scale REAL",
    // MM3 LM adapter per album (2026-09-11): the checkpoint's weights file,
    // relative to the mm3-lm-adapters root — the same reference the
    // mm3LmAdapter request param carries. ACE's lm_adapter_path above is the
    // 5 Hz planner adapter and stays separate: one preset serves both backends.
    "ALTER TABLE album_presets ADD COLUMN mm3_adapter_path TEXT",
    // YuE2 adapters per album (2026-09-15). TWO columns, not one, because YuE2
    // is two experts and a stack needs both: the AR plans and carries the
    // artist likeness, the NAR renders those tokens to audio. They are trained
    // by separate runs into separate roots and are selected in separate slots
    // (the dual-slot picker, 58aff871), so one column could only ever hold half
    // a preset. Absolute paths, matching what the YuE2 catalogue reports.
    "ALTER TABLE album_presets ADD COLUMN yue2_ar_adapter_path TEXT",
    "ALTER TABLE album_presets ADD COLUMN yue2_nar_adapter_path TEXT",
  ];
  for (const sql of lireekMigrations) {
    try { db.exec(sql); } catch { /* column already exists */ }
  }

  // ── "These lyrics have been generated" marker ─────────────────────────────
  // Stored on the generations row rather than derived from audio_generations,
  // because the whole point is that it survives the audio being deleted — from
  // disk or from the DB. Backfilled once from the audio_generations rows that
  // already exist so the "never generated" filter is accurate on day one.
  const genCols = db.prepare("SELECT name FROM pragma_table_info('generations')").all() as { name: string }[];
  const hasGenCol = (n: string) => genCols.some(c => c.name === n);
  const needsGeneratedBackfill = !hasGenCol('audio_generated_count');

  if (!hasGenCol('audio_generated_count')) {
    db.exec('ALTER TABLE generations ADD COLUMN audio_generated_count INTEGER NOT NULL DEFAULT 0');
  }
  if (!hasGenCol('first_generated_at')) {
    db.exec('ALTER TABLE generations ADD COLUMN first_generated_at TEXT');
  }

  // Downloads: "I liked this one enough to keep it". Same reasoning as above —
  // stored on the generations row so deleting the audio afterwards, which is
  // exactly what happens once a track has been downloaded, doesn't lose it.
  // downloaded_at on audio_generations makes the count per-track rather than
  // per-click, so grabbing both the original and the master, or downloading the
  // same file twice, still counts as one kept version.
  // No backfill is possible: nothing recorded downloads before now.
  if (!hasGenCol('download_count')) {
    db.exec('ALTER TABLE generations ADD COLUMN download_count INTEGER NOT NULL DEFAULT 0');
  }
  if (!hasGenCol('first_downloaded_at')) {
    db.exec('ALTER TABLE generations ADD COLUMN first_downloaded_at TEXT');
  }
  try { db.exec('ALTER TABLE audio_generations ADD COLUMN downloaded_at TEXT'); } catch { /* exists */ }
  if (needsGeneratedBackfill) {
    const filled = db.prepare(`
      UPDATE generations SET
        audio_generated_count = (SELECT COUNT(*) FROM audio_generations ag WHERE ag.generation_id = generations.id),
        first_generated_at    = (SELECT MIN(ag.created_at) FROM audio_generations ag WHERE ag.generation_id = generations.id)
      WHERE EXISTS (SELECT 1 FROM audio_generations ag WHERE ag.generation_id = generations.id)
    `).run();
    console.log(`[DB] Backfilled generated-marker for ${filled.changes} lyric generations`);
  }

  // ── One-time migration: import data from lireek.db if it exists ───────────
  migrateLireekData();

  console.log(`[DB] Initialized: ${config.data.dbPath}`);
}

/**
 * One-time migration: copies all data from the legacy lireek.db into hotstep.db,
 * then renames lireek.db → lireek.db.migrated as a backup.
 *
 * Safe to run repeatedly — it's a no-op if lireek.db doesn't exist or has
 * already been migrated.
 */
function migrateLireekData(): void {
  const lireekPath = path.join(config.data.dir, 'lireek.db');
  if (!fs.existsSync(lireekPath)) {
    return; // Nothing to migrate
  }

  // Check if we already have data — if artists table has rows, assume migration is done
  const existingArtists = (db.prepare('SELECT COUNT(*) as c FROM artists').get() as any).c;
  if (existingArtists > 0) {
    console.log(`[DB] Lireek data already present (${existingArtists} artists) — skipping migration`);
    // If lireek.db still exists, rename it now
    const backupPath = lireekPath + '.migrated';
    if (!fs.existsSync(backupPath)) {
      fs.renameSync(lireekPath, backupPath);
      console.log(`[DB] Renamed lireek.db → lireek.db.migrated`);
    }
    return;
  }

  console.log(`[DB] ═══════════════════════════════════════════════════════════`);
  console.log(`[DB] Migrating lireek.db data into hotstep.db...`);

  // Temporarily disable foreign keys for the migration
  db.pragma('foreign_keys = OFF');

  try {
    // Attach the old database
    db.exec(`ATTACH DATABASE '${lireekPath.replace(/'/g, "''")}' AS lireek_old`);

    // Tables to migrate, in dependency order (parents first)
    const tables = [
      'artists',
      'lyrics_sets',
      'profiles',
      'generations',
      'settings',
      'album_presets',
      'audio_generations',
    ];

    const counts: Record<string, number> = {};

    for (const table of tables) {
      // Check if the source table exists in lireek_old
      const exists = db.prepare(
        `SELECT COUNT(*) as c FROM lireek_old.sqlite_master WHERE type='table' AND name=?`
      ).get(table) as any;

      if (exists.c === 0) {
        console.log(`[DB]   ${table}: skipped (not in lireek.db)`);
        continue;
      }

      // Get column names from source table
      const cols = (db.prepare(`PRAGMA lireek_old.table_info('${table}')`).all() as any[])
        .map(c => c.name);

      // Filter to only columns that exist in the target table
      const targetCols = (db.prepare(`PRAGMA table_info('${table}')`).all() as any[])
        .map(c => c.name);
      const commonCols = cols.filter(c => targetCols.includes(c));

      if (commonCols.length === 0) {
        console.log(`[DB]   ${table}: skipped (no common columns)`);
        continue;
      }

      const colList = commonCols.join(', ');
      const result = db.prepare(
        `INSERT OR IGNORE INTO ${table} (${colList}) SELECT ${colList} FROM lireek_old.${table}`
      ).run();

      counts[table] = result.changes;
      console.log(`[DB]   ${table}: ${result.changes} rows migrated`);
    }

    db.exec('DETACH DATABASE lireek_old');

    // Rename the old file
    const backupPath = lireekPath + '.migrated';
    fs.renameSync(lireekPath, backupPath);

    const totalRows = Object.values(counts).reduce((a, b) => a + b, 0);
    console.log(`[DB] Migration complete: ${totalRows} total rows imported`);
    console.log(`[DB] Old file preserved as: lireek.db.migrated`);
    console.log(`[DB] ═══════════════════════════════════════════════════════════`);
  } catch (err: any) {
    console.error(`[DB] Migration failed: ${err.message}`);
    console.error(`[DB] lireek.db was NOT modified — data is safe`);
    try { db.exec('DETACH DATABASE lireek_old'); } catch { /* may not be attached */ }
  } finally {
    db.pragma('foreign_keys = ON');
  }
}

export function closeDb(): void {
  if (db) {
    db.close();
    console.log('[DB] Closed');
  }
}
