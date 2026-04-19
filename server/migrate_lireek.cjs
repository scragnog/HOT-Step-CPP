/**
 * migrate_lireek.cjs — Migrate data from hot-step-9000 hotstep_lyrics.db
 *                       into hot-step-cpp lireek.db
 *
 * Schema differences handled:
 *   - album_presets: adapter_scales (JSON) → adapter_scale + adapter_group_scales
 *   - album_presets: matchering_ref_path → reference_track_path
 *   - album_presets: updated_at → created_at
 *
 * Run: node migrate_lireek.cjs
 */

const Database = require('better-sqlite3');
const path = require('path');

const SRC_PATH = 'D:/Ace-Step-Latest/hot-step-9000/data/hotstep_lyrics.db';
const DST_PATH = path.resolve(__dirname, 'data/lireek.db');

console.log('=== Lireek DB Migration ===');
console.log(`Source: ${SRC_PATH}`);
console.log(`Target: ${DST_PATH}`);
console.log();

const src = new Database(SRC_PATH, { readonly: true });
const dst = new Database(DST_PATH);

// Enable foreign keys and WAL mode on target
dst.pragma('journal_mode = WAL');
dst.pragma('foreign_keys = ON');

// ── Step 1: Ensure target schema exists (run migrations) ──
// The server would normally do this on startup, but ensure columns exist
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
  try { dst.exec(sql); } catch { /* already exists */ }
}

// ── Step 2: Clear target tables (in safe order due to FK) ──
console.log('Clearing target tables...');
dst.exec('DELETE FROM audio_generations');
dst.exec('DELETE FROM album_presets');
dst.exec('DELETE FROM generations');
dst.exec('DELETE FROM profiles');
dst.exec('DELETE FROM lyrics_sets');
dst.exec('DELETE FROM artists');
dst.exec('DELETE FROM settings');
// Reset autoincrement counters
dst.exec("DELETE FROM sqlite_sequence WHERE name IN ('artists','lyrics_sets','profiles','generations','album_presets','audio_generations')");

// ── Step 3: Migrate artists ──
const srcArtists = src.prepare('SELECT * FROM artists ORDER BY id').all();
const insertArtist = dst.prepare(
  'INSERT INTO artists (id, name, created_at, image_url, genius_id) VALUES (?, ?, ?, ?, ?)'
);
let artistCount = 0;
for (const a of srcArtists) {
  insertArtist.run(a.id, a.name, a.created_at, a.image_url || null, a.genius_id || null);
  artistCount++;
}
console.log(`✓ Artists: ${artistCount}`);

// ── Step 4: Migrate lyrics_sets ──
const srcSets = src.prepare('SELECT * FROM lyrics_sets ORDER BY id').all();
const insertSet = dst.prepare(
  'INSERT INTO lyrics_sets (id, artist_id, album, max_songs, songs, fetched_at, image_url) VALUES (?, ?, ?, ?, ?, ?, ?)'
);
let setCount = 0;
for (const s of srcSets) {
  insertSet.run(s.id, s.artist_id, s.album, s.max_songs, s.songs, s.fetched_at, s.image_url || null);
  setCount++;
}
console.log(`✓ Lyrics Sets: ${setCount}`);

// ── Step 5: Migrate profiles ──
const srcProfiles = src.prepare('SELECT * FROM profiles ORDER BY id').all();
const insertProfile = dst.prepare(
  'INSERT INTO profiles (id, lyrics_set_id, provider, model, profile_data, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);
let profileCount = 0;
for (const p of srcProfiles) {
  insertProfile.run(p.id, p.lyrics_set_id, p.provider, p.model, p.profile_data, p.created_at);
  profileCount++;
}
console.log(`✓ Profiles: ${profileCount}`);

// ── Step 6: Migrate generations ──
const srcGens = src.prepare('SELECT * FROM generations ORDER BY id').all();
const insertGen = dst.prepare(
  `INSERT INTO generations (id, profile_id, provider, model, extra_instructions, title, subject,
   lyrics, system_prompt, user_prompt, bpm, key, caption, duration, parent_generation_id, created_at)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
);
let genCount = 0;
for (const g of srcGens) {
  insertGen.run(
    g.id, g.profile_id, g.provider, g.model, g.extra_instructions || null,
    g.title || '', g.subject || '', g.lyrics, g.system_prompt || '', g.user_prompt || '',
    g.bpm || 0, g.key || '', g.caption || '', g.duration || 0,
    g.parent_generation_id || null, g.created_at
  );
  genCount++;
}
console.log(`✓ Generations: ${genCount}`);

// ── Step 7: Migrate album_presets (schema differs!) ──
const srcPresets = src.prepare('SELECT * FROM album_presets ORDER BY id').all();
const insertPreset = dst.prepare(
  `INSERT INTO album_presets (id, lyrics_set_id, adapter_path, adapter_scale, adapter_group_scales,
   reference_track_path, audio_cover_strength, created_at)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
);
let presetCount = 0;
for (const p of srcPresets) {
  // Parse adapter_scales JSON → separate scale + group_scales
  let adapterScale = null;
  let groupScales = null;
  if (p.adapter_scales) {
    try {
      const parsed = JSON.parse(p.adapter_scales);
      adapterScale = parsed.scale ?? 1.0;
      if (parsed.group_scales) {
        groupScales = JSON.stringify(parsed.group_scales);
      }
    } catch (e) {
      console.warn(`  ⚠ Failed to parse adapter_scales for preset ${p.id}: ${e.message}`);
    }
  }

  insertPreset.run(
    p.id, p.lyrics_set_id, p.adapter_path || null,
    adapterScale, groupScales,
    p.matchering_ref_path || null,  // renamed column
    p.audio_cover_strength || null,
    p.updated_at || new Date().toISOString()  // updated_at → created_at
  );
  presetCount++;
}
console.log(`✓ Album Presets: ${presetCount}`);

// ── Step 8: Migrate audio_generations ──
const srcAGs = src.prepare('SELECT * FROM audio_generations ORDER BY id').all();
const insertAG = dst.prepare(
  'INSERT INTO audio_generations (id, generation_id, hotstep_job_id, audio_url, cover_url, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);
let agCount = 0;
for (const ag of srcAGs) {
  insertAG.run(ag.id, ag.generation_id, ag.hotstep_job_id, ag.audio_url || null, ag.cover_url || null, ag.created_at);
  agCount++;
}
console.log(`✓ Audio Generations: ${agCount}`);

// ── Step 9: Migrate settings ──
const srcSettings = src.prepare('SELECT * FROM settings ORDER BY key').all();
const insertSetting = dst.prepare(
  'INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)'
);
let settingCount = 0;
for (const s of srcSettings) {
  insertSetting.run(s.key, s.value);
  settingCount++;
}
console.log(`✓ Settings: ${settingCount}`);

// ── Step 10: Update sqlite_sequence to match max IDs ──
const tables = ['artists', 'lyrics_sets', 'profiles', 'generations', 'album_presets', 'audio_generations'];
for (const table of tables) {
  const max = dst.prepare(`SELECT MAX(id) as m FROM ${table}`).get();
  if (max && max.m) {
    dst.prepare('INSERT OR REPLACE INTO sqlite_sequence (name, seq) VALUES (?, ?)').run(table, max.m);
  }
}

// ── Verification ──
console.log('\n--- Verification ---');
for (const table of tables) {
  const srcCount = src.prepare(`SELECT count(*) as c FROM ${table}`).get();
  const dstCount = dst.prepare(`SELECT count(*) as c FROM ${table}`).get();
  const match = srcCount.c === dstCount.c ? '✓' : '✗ MISMATCH!';
  console.log(`${match} ${table}: source=${srcCount.c} target=${dstCount.c}`);
}

src.close();
dst.close();
console.log('\n✅ Migration complete!');
