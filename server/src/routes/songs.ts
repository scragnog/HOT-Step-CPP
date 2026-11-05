// songs.ts — Song CRUD routes + audio file serving
//
// Songs are stored in SQLite. Audio files are saved to data/audio/.

import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import multer from 'multer';
import { getDb } from '../db/database.js';
import { config } from '../config.js';
import { getUserId } from './auth.js';
import { deleteAudioGenerationsByJobIds } from '../db/lireekDb.js';
import { cleanupDeletedSongs } from '../services/compare/store.js';
import { cropWavFile, cropLrcFile, cropLyricsJsonFile } from '../services/audioCrop.js';
import { analyzeAndSaveDiscoData } from '../services/disco-analyzer.js';
import {
  startRePostProcess, checkPpEligibility, requestedPpStages, revertPostProcessing,
  getRePostProcessJob, findRePostProcessJobBySong,
} from '../services/generation/rePostProcess.js';
import { importTrackFile, IMPORT_EXTENSIONS } from '../services/library/importTrack.js';
import { SONG_EDITABLE_FIELDS } from '../contracts/songs.js';
import { planZipEntries, parseZipIds, streamZip, MAX_ZIP_IDS, MAX_ZIP_BYTES } from '../services/library/exportZip.js';

const router = Router();

/**
 * Delete the lyric sidecars that share a render's filename stem.
 *
 * Deleting a song removed its audio variants but left `.lrc` and
 * `.lyrics.json` behind, so every deleted track leaked two orphan files into
 * the audio folder. Sidecars live beside the BASE render, so one stem covers
 * every variant of the song.
 */
function deleteLyricSidecars(audioUrl: string | null | undefined): void {
  if (!audioUrl) return;
  const stem = path.basename(audioUrl).replace(/\.[^.]+$/, '');
  for (const ext of ['.lrc', '.lyrics.json']) {
    const filepath = path.join(config.data.audioDir, stem + ext);
    try {
      if (fs.existsSync(filepath)) fs.unlinkSync(filepath);
    } catch { /* best effort — never fail a delete over a sidecar */ }
  }
}

// Uploads land on disk rather than in memory: an album's worth of FLAC at
// 100 MB a file is not something to hold in the heap while ffmpeg works.
const importTempDir = path.join(config.data.dir, 'import_temp');
const importUpload = multer({
  dest: importTempDir,
  limits: { fileSize: 500 * 1024 * 1024, files: 50 },
});

// GET /api/songs — list user's songs
router.get('/', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  // Optional source filter — filters by generation_params JSON field
  const source = req.query.source as string | undefined;
  let query = 'SELECT * FROM songs WHERE user_id = ?';
  const params: any[] = [userId];
  if (source) {
    query += ` AND json_extract(generation_params, '$.source') = ?`;
    params.push(source);
  }
  query += ' ORDER BY created_at DESC';

  const songs = getDb().prepare(query).all(...params);

  // Parse tags JSON string
  const parsed = songs.map((s: any) => ({
    ...s,
    tags: JSON.parse(s.tags || '[]'),
    is_public: !!s.is_public,
  }));

  res.json({ songs: parsed });
});

// GET /api/songs/ids — bare id list, used by the UI queue to prune entries
// whose songs were deleted (nuke, other tab, individual delete)
// IMPORTANT: Must be defined BEFORE /:id to avoid Express matching 'ids' as an id
router.get('/ids', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const rows = getDb().prepare('SELECT id FROM songs WHERE user_id = ?').all(userId) as Array<{ id: string }>;
  res.json({ ids: rows.map(r => r.id) });
});

// GET /api/songs/recent — unified recent songs across all modes
// Supports ?source=create|lyric-studio|cover-studio&limit=50
// Returns a normalized shape compatible with the frontend's RecentSong interface
// IMPORTANT: Must be defined BEFORE /:id to avoid Express matching 'recent' as an id
router.get('/recent', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const source = req.query.source as string | undefined;
  const limit = parseInt(req.query.limit as string, 10) || 50;

  let query = 'SELECT * FROM songs WHERE user_id = ?';
  const params: any[] = [userId];

  if (source && source !== 'all') {
    query += ` AND json_extract(generation_params, '$.source') = ?`;
    params.push(source);
  }

  query += ' ORDER BY created_at DESC LIMIT ?';
  params.push(limit);

  const songs = getDb().prepare(query).all(...params) as any[];

  // Enrich songs with Lyric Studio metadata where available
  // For lyric-studio songs, look up artist name via the audio_generations → generations → profiles → lyrics_sets → artists chain
  const audioUrls = songs.map(s => s.audio_url).filter(Boolean);
  let lireekMetaMap = new Map<string, { artist_name: string; artist_image?: string; album?: string; generation_id?: number }>();

  if (audioUrls.length > 0) {
    try {
      const placeholders = audioUrls.map(() => '?').join(',');
      const enrichRows = getDb().prepare(
        `SELECT ag.audio_url, a.name AS artist_name, a.image_url AS artist_image,
                ls.album, g.id AS generation_id
         FROM audio_generations ag
         JOIN generations g ON g.id = ag.generation_id
         JOIN profiles p ON p.id = g.profile_id
         JOIN lyrics_sets ls ON ls.id = COALESCE(ag.lyrics_set_id, p.lyrics_set_id)
         JOIN artists a ON a.id = ls.artist_id
         WHERE ag.audio_url IN (${placeholders})`
      ).all(...audioUrls) as any[];
      for (const row of enrichRows) {
        lireekMetaMap.set(row.audio_url, row);
      }
    } catch { /* lireek tables may not exist yet */ }
  }

  // Build normalized response
  const result = songs.map((s: any) => {
    const genParams = JSON.parse(s.generation_params || '{}');
    const lireekMeta = lireekMetaMap.get(s.audio_url);
    return {
      id: s.id,
      title: s.title || 'Untitled',
      audio_url: s.audio_url || '',
      mastered_audio_url: s.mastered_audio_url || '',
      noadapter_audio_url: s.noadapter_audio_url || '',
      latent_url: s.latent_url || '',
      kick_stem_url: s.kick_stem_url || '',
      snare_stem_url: s.snare_stem_url || '',
      hihat_stem_url: s.hihat_stem_url || '',
      disco_data_url: s.disco_data_url || '',
      cover_url: s.cover_url || '',
      duration: s.duration || 0,
      lyrics: s.lyrics || '',
      caption: s.caption || '',
      style: s.style || '',
      bpm: s.bpm || 0,
      key_scale: s.key_scale || '',
      time_signature: s.time_signature || '',
      metadata_overrides: s.metadata_overrides || '',
      source: genParams.source || 'create',
      created_at: s.created_at,
      // Enriched from Lyric Studio metadata
      artist_name: lireekMeta?.artist_name || genParams.artist_name || '',
      artist_image: lireekMeta?.artist_image || '',
      album: lireekMeta?.album || genParams.album || '',
      generation_id: lireekMeta?.generation_id || null,
    };
  });

  res.json({ songs: result });
});

// GET /api/songs/:id — get single song
router.get('/:id', (req, res) => {
  const song = getDb()
    .prepare('SELECT * FROM songs WHERE id = ?')
    .get(req.params.id) as any;

  if (!song) { res.status(404).json({ error: 'Song not found' }); return; }

  res.json({
    song: {
      ...song,
      tags: JSON.parse(song.tags || '[]'),
      is_public: !!song.is_public,
    },
  });
});

// POST /api/songs — create a new song
router.post('/', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const {
    id, title, lyrics, style, caption, audio_url, cover_url,
    duration, bpm, key_scale, time_signature, tags, dit_model,
    generation_params,
  } = req.body;

  const songId = id || crypto.randomUUID();
  const tagsJson = JSON.stringify(tags || []);
  const genParamsJson = typeof generation_params === 'string'
    ? generation_params
    : JSON.stringify(generation_params || {});

  getDb().prepare(`
    INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url, cover_url,
                       duration, bpm, key_scale, time_signature, tags, dit_model, generation_params)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    songId, userId, title || 'Untitled', lyrics || '', style || '',
    caption || '', audio_url || '', cover_url || '',
    duration || 0, bpm || 0, key_scale || '', time_signature || '',
    tagsJson, dit_model || '', genParamsJson,
  );

  const song = getDb().prepare('SELECT * FROM songs WHERE id = ?').get(songId) as any;
  res.json({
    song: { ...song, tags: JSON.parse(song.tags || '[]'), is_public: !!song.is_public },
  });
});

// POST /api/songs/import — bring existing audio files into the library
//
// Multipart, field "audio", one or more files. Each becomes a song row whose
// audio_url is a raw WAV in data/audio — the shape every later stage expects,
// post-processing included. Files are handled one at a time and reported one
// at a time: one unreadable file in a folder-full must not lose the rest.
router.post('/import', importUpload.array('audio', 50), async (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const files = (req.files as Express.Multer.File[] | undefined) || [];
  if (files.length === 0) { res.status(400).json({ error: 'No files uploaded' }); return; }

  // One description for the batch — what StableStep would refine these towards.
  // Per-track edits happen afterwards in Edit Metadata.
  const description = typeof req.body?.description === 'string' ? req.body.description : '';

  const songs: any[] = [];
  const errors: { file: string; error: string }[] = [];

  for (const file of files) {
    try {
      const song = await importTrackFile({
        userId,
        sourcePath: file.path,
        originalName: file.originalname,
        description,
      });
      songs.push({ ...song, tags: JSON.parse(song.tags || '[]'), is_public: !!song.is_public });
    } catch (err: any) {
      console.error(`[Import] ${file.originalname} failed:`, err.message);
      errors.push({ file: file.originalname, error: err.message });
    } finally {
      // The upload's temp copy has served its purpose either way.
      try { fs.unlinkSync(file.path); } catch { /* already gone */ }
    }
  }

  // Nothing got in — report it as the failure it is rather than as an empty
  // success the UI has to squint at.
  if (songs.length === 0) {
    res.status(400).json({ error: errors[0]?.error || 'Import failed', errors });
    return;
  }
  res.json({ songs, errors, accepted: IMPORT_EXTENSIONS });
});

// PATCH /api/songs/:id — update song
router.patch('/:id', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const song = getDb().prepare('SELECT * FROM songs WHERE id = ? AND user_id = ?')
    .get(req.params.id, userId) as any;
  if (!song) { res.status(404).json({ error: 'Song not found' }); return; }

  const updates = req.body;
  for (const key of SONG_EDITABLE_FIELDS) {
    if (updates[key] !== undefined) {
      const value = key === 'is_public' ? (updates[key] ? 1 : 0) : updates[key];
      getDb().prepare(`UPDATE songs SET ${key} = ? WHERE id = ?`).run(value, req.params.id);
    }
  }

  if (updates.tags) {
    getDb().prepare('UPDATE songs SET tags = ? WHERE id = ?')
      .run(JSON.stringify(updates.tags), req.params.id);
  }

  // Metadata-editor overrides (#60) — embed-tag values used verbatim on export.
  // Accept an object, store as JSON; an empty/null value clears the overrides.
  if (updates.metadata_overrides !== undefined) {
    const val = updates.metadata_overrides && typeof updates.metadata_overrides === 'object'
      ? JSON.stringify(updates.metadata_overrides)
      : '';
    getDb().prepare('UPDATE songs SET metadata_overrides = ? WHERE id = ?')
      .run(val, req.params.id);
  }

  const updated = getDb().prepare('SELECT * FROM songs WHERE id = ?').get(req.params.id) as any;
  res.json({
    song: { ...updated, tags: JSON.parse(updated.tags || '[]'), is_public: !!updated.is_public },
  });
});

// DELETE /api/songs/:id — delete song + audio file
router.delete('/:id', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const song = getDb().prepare('SELECT * FROM songs WHERE id = ? AND user_id = ?')
    .get(req.params.id, userId) as any;
  if (!song) { res.status(404).json({ error: 'Song not found' }); return; }

  // Delete audio file if it exists
  if (song.audio_url) {
    const filename = path.basename(song.audio_url);
    const filepath = path.join(config.data.audioDir, filename);
    if (fs.existsSync(filepath)) {
      fs.unlinkSync(filepath);
    }
  }
  // Delete mastered audio file if it exists
  if (song.mastered_audio_url) {
    const masteredFilename = path.basename(song.mastered_audio_url);
    const masteredFilepath = path.join(config.data.audioDir, masteredFilename);
    if (fs.existsSync(masteredFilepath)) {
      fs.unlinkSync(masteredFilepath);
    }
  }
  // Delete no-adapter reference render if it exists
  if (song.noadapter_audio_url) {
    const refFilename = path.basename(song.noadapter_audio_url);
    const refFilepath = path.join(config.data.audioDir, refFilename);
    if (fs.existsSync(refFilepath)) {
      fs.unlinkSync(refFilepath);
    }
  }
  // Delete latent file if it exists
  if (song.latent_url) {
    const latentFilename = path.basename(song.latent_url);
    const latentFilepath = path.join(config.data.audioDir, latentFilename);
    if (fs.existsSync(latentFilepath)) {
      fs.unlinkSync(latentFilepath);
    }
  }
  // Delete drum stem files if they exist
  for (const stemUrl of [song.kick_stem_url, song.snare_stem_url, song.hihat_stem_url]) {
    if (stemUrl && !stemUrl.startsWith('extracting:')) {
      const stemFilename = path.basename(stemUrl);
      const stemFilepath = path.join(config.data.audioDir, stemFilename);
      if (fs.existsSync(stemFilepath)) {
        fs.unlinkSync(stemFilepath);
      }
    }
  }

  deleteLyricSidecars(song.audio_url);

  getDb().prepare('DELETE FROM songs WHERE id = ?').run(req.params.id);
  cleanupDeletedSongs([req.params.id]);

  // Cascade to Lireek DB — remove matching audio_generation records
  try { deleteAudioGenerationsByJobIds([req.params.id]); } catch { /* lireek DB may not be initialized */ }

  res.json({ success: true });
});

// DELETE /api/songs — delete all user's songs
router.delete('/', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const songs = getDb().prepare('SELECT audio_url FROM songs WHERE user_id = ?').all(userId) as any[];

  // Delete audio files
  for (const song of songs) {
    if (song.audio_url) {
      const filename = path.basename(song.audio_url);
      const filepath = path.join(config.data.audioDir, filename);
      if (fs.existsSync(filepath)) {
        fs.unlinkSync(filepath);
      }
      deleteLyricSidecars(song.audio_url);
    }
  }

  // Get all song IDs before deleting (for Lireek cascade)
  const allSongs = getDb().prepare('SELECT id FROM songs WHERE user_id = ?').all(userId) as any[];
  const allIds = allSongs.map((s: any) => s.id);

  const result = getDb().prepare('DELETE FROM songs WHERE user_id = ?').run(userId);
  cleanupDeletedSongs(allIds);

  // Cascade to Lireek DB
  try { deleteAudioGenerationsByJobIds(allIds); } catch { /* lireek DB may not be initialized */ }

  res.json({ success: true, deletedCount: result.changes });
});

// POST /api/songs/bulk-delete — delete multiple songs by ID
router.post('/bulk-delete', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const { ids } = req.body;
  if (!Array.isArray(ids) || ids.length === 0) {
    res.status(400).json({ error: 'ids must be a non-empty array' });
    return;
  }

  // Fetch all matching songs to get file paths
  const placeholders = ids.map(() => '?').join(',');
  const songs = getDb()
    .prepare(`SELECT id, audio_url, mastered_audio_url, noadapter_audio_url FROM songs WHERE id IN (${placeholders}) AND user_id = ?`)
    .all(...ids, userId) as any[];

  // Delete audio files from disk
  for (const song of songs) {
    if (song.audio_url) {
      const filepath = path.join(config.data.audioDir, path.basename(song.audio_url));
      if (fs.existsSync(filepath)) fs.unlinkSync(filepath);
    }
    if (song.mastered_audio_url) {
      const filepath = path.join(config.data.audioDir, path.basename(song.mastered_audio_url));
      if (fs.existsSync(filepath)) fs.unlinkSync(filepath);
    }
    if (song.noadapter_audio_url) {
      const filepath = path.join(config.data.audioDir, path.basename(song.noadapter_audio_url));
      if (fs.existsSync(filepath)) fs.unlinkSync(filepath);
    }
    deleteLyricSidecars(song.audio_url);
  }

  // Delete from DB
  const result = getDb()
    .prepare(`DELETE FROM songs WHERE id IN (${placeholders}) AND user_id = ?`)
    .run(...ids, userId);
  cleanupDeletedSongs(ids.filter((id: string) => !getDb().prepare('SELECT 1 FROM songs WHERE id = ?').get(id)));

  console.log(`[Songs] Bulk deleted ${result.changes}/${ids.length} songs`);

  // Cascade to Lireek DB
  try { deleteAudioGenerationsByJobIds(ids); } catch { /* lireek DB may not be initialized */ }

  res.json({ success: true, deletedCount: result.changes });
});

// POST /api/songs/export-zip — stream the selected songs' audio as one zip
router.post('/export-zip', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const ids = parseZipIds(req.body?.ids);
  if (!ids) {
    res.status(400).json({ error: `ids must be a non-empty array of id strings (max ${MAX_ZIP_IDS})` });
    return;
  }

  const placeholders = ids.map(() => '?').join(',');
  const rows = getDb()
    .prepare(`SELECT id, title, audio_url FROM songs WHERE id IN (${placeholders}) AND user_id = ?`)
    .all(...ids, userId) as any[];
  const order = new Map(ids.map((id, i) => [id, i]));
  rows.sort((a, b) => (order.get(a.id) ?? 0) - (order.get(b.id) ?? 0));

  const { entries, totalBytes } = planZipEntries(rows, config.data.audioDir);
  if (entries.length === 0) {
    res.status(404).json({ error: 'No audio files found for the selected songs' });
    return;
  }
  if (totalBytes > MAX_ZIP_BYTES) {
    res.status(413).json({
      error: `Selection is ${Math.ceil(totalBytes / 1048576)} MB; a zip export is limited to ${MAX_ZIP_BYTES / 1048576} MB. Select fewer tracks.`,
    });
    return;
  }

  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  res.setHeader('Content-Type', 'application/zip');
  res.setHeader('Content-Disposition', `attachment; filename="hotstep-library-${stamp}.zip"`);
  streamZip(res, entries, err => console.error('[Songs] export-zip failed:', err));
});

// POST /api/songs/nuke-generations — delete ALL generated audio across both databases + disk
router.post('/nuke-generations', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  // 1. Collect all file-reference columns from songs table
  const songs = getDb()
    .prepare(`SELECT id, audio_url, mastered_audio_url, noadapter_audio_url, latent_url,
                     kick_stem_url, snare_stem_url, hihat_stem_url,
                     disco_data_url, cover_url
              FROM songs WHERE user_id = ?`)
    .all(userId) as any[];

  // 2. Delete referenced files from disk
  let filesDeleted = 0;
  const fileColumns = [
    'audio_url', 'mastered_audio_url', 'noadapter_audio_url', 'latent_url',
    'kick_stem_url', 'snare_stem_url', 'hihat_stem_url',
    'disco_data_url', 'cover_url',
  ];
  for (const song of songs) {
    for (const col of fileColumns) {
      const url = song[col];
      if (url && !url.startsWith('extracting:')) {
        const filepath = path.join(config.data.audioDir, path.basename(url));
        if (fs.existsSync(filepath)) {
          try { fs.unlinkSync(filepath); filesDeleted++; } catch { /* best effort */ }
        }
      }
    }
    // Also delete companion files (lyrics JSON, LRC) that share the audio filename stem
    if (song.audio_url) {
      const baseName = path.basename(song.audio_url).replace(/\.[^.]+$/, '');
      for (const ext of ['.lyrics.json', '.lrc']) {
        const companionPath = path.join(config.data.audioDir, baseName + ext);
        if (fs.existsSync(companionPath)) {
          try { fs.unlinkSync(companionPath); filesDeleted++; } catch { /* best effort */ }
        }
      }
    }
  }

  // 3. Delete all songs from main DB
  const songIds = songs.map((s: any) => s.id);
  const songResult = getDb().prepare('DELETE FROM songs WHERE user_id = ?').run(userId);
  cleanupDeletedSongs(songIds);

  // 4. Delete all audio_generations (same DB now)
  let lireekDeleted = 0;
  try {
    if (songIds.length > 0) {
      lireekDeleted += deleteAudioGenerationsByJobIds(songIds);
    }
    // Also nuke ALL audio_generations (catches any orphans)
    const allResult = getDb().prepare('DELETE FROM audio_generations').run();
    lireekDeleted = Math.max(lireekDeleted, allResult.changes);
  } catch (err) {
    console.error('[Songs] NUKE audio_generations cleanup error:', err);
  }

  // 5. Sweep the audio directory — remove any orphan files the DB didn't know about
  let orphansDeleted = 0;
  try {
    if (fs.existsSync(config.data.audioDir)) {
      const remaining = fs.readdirSync(config.data.audioDir);
      for (const file of remaining) {
        const filePath = path.join(config.data.audioDir, file);
        try {
          const stat = fs.statSync(filePath);
          if (stat.isFile()) {
            fs.unlinkSync(filePath);
            orphansDeleted++;
          }
        } catch { /* best effort */ }
      }
      if (orphansDeleted > 0) {
        console.log(`[Songs] NUKE: swept ${orphansDeleted} orphan files from audio directory`);
      }
    }
  } catch (err) {
    console.error('[Songs] NUKE: audio directory sweep error:', err);
  }

  console.log(`[Songs] NUKE: ${songResult.changes} songs, ${filesDeleted} referenced files, ${orphansDeleted} orphans, ${lireekDeleted} lireek audio_gens deleted`);

  res.json({
    success: true,
    songsDeleted: songResult.changes,
    filesDeleted: filesDeleted + orphansDeleted,
    lireekAudioGensDeleted: lireekDeleted,
  });
});

// POST /api/songs/:id/crop — crop audio to IN/OUT range (destructive)
router.post('/:id/crop', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  let { inPoint, outPoint, audioUrl } = req.body;
  if (inPoint == null || outPoint == null) {
    res.status(400).json({ error: 'inPoint and outPoint are required' });
    return;
  }

  // Auto-swap if reversed
  if (inPoint > outPoint) [inPoint, outPoint] = [outPoint, inPoint];

  // Look up song by ID first, then fall back to audio_url match
  // (Lireek/Lyric Studio tracks use hotstep_job_id as their ID, which
  // doesn't match the songs.id column — but the audio_url is the same)
  let song = getDb().prepare('SELECT * FROM songs WHERE id = ?')
    .get(req.params.id) as any;
  if (!song && audioUrl) {
    song = getDb().prepare('SELECT * FROM songs WHERE audio_url = ?')
      .get(audioUrl) as any;
  }
  if (!song) { res.status(404).json({ error: 'Song not found' }); return; }

  try {
    // 1. Crop original audio
    const audioFilename = path.basename(song.audio_url);
    const audioPath = path.join(config.data.audioDir, audioFilename);
    if (!fs.existsSync(audioPath)) {
      res.status(404).json({ error: 'Audio file not found' });
      return;
    }
    const result = cropWavFile(audioPath, inPoint, outPoint);

    // 2. Crop mastered audio (if exists)
    if (song.mastered_audio_url) {
      const masteredFilename = path.basename(song.mastered_audio_url);
      const masteredPath = path.join(config.data.audioDir, masteredFilename);
      if (fs.existsSync(masteredPath)) {
        cropWavFile(masteredPath, inPoint, outPoint);
      }
    }

    // 3. Crop companion lyric sidecars (if they exist). Both, because the
    //    player prefers `.lyrics.json` while exports read the `.lrc`: cropping
    //    only one leaves the other on the pre-crop timeline.
    const lrcFilename = audioFilename.replace(/\.[^.]+$/, '.lrc');
    const lrcPath = path.join(config.data.audioDir, lrcFilename);
    if (fs.existsSync(lrcPath)) {
      cropLrcFile(lrcPath, inPoint, outPoint);
    }
    const lyricsJsonPath = path.join(config.data.audioDir, audioFilename.replace(/\.[^.]+$/, '.lyrics.json'));
    if (fs.existsSync(lyricsJsonPath)) {
      try { cropLyricsJsonFile(lyricsJsonPath, inPoint, outPoint); } catch (err: any) {
        console.warn(`[Songs] Crop: word-level lyrics not cropped: ${err.message}`);
      }
    }

    // 4. Update duration in DB
    getDb().prepare('UPDATE songs SET duration = ? WHERE id = ?')
      .run(result.newDurationSec, req.params.id);

    console.log(`[Songs] Cropped ${req.params.id}: ${inPoint.toFixed(1)}s–${outPoint.toFixed(1)}s → ${result.newDurationSec.toFixed(1)}s`);
    res.json({ cropped: true, newDuration: result.newDurationSec });
  } catch (err: any) {
    console.error(`[Songs] Crop failed:`, err.message);
    res.status(500).json({ error: err.message });
  }
});


// POST /api/songs/lyrics/offset — move a track's lyric timing by hand.
//
// Body: { audioUrl, deltaMs }. Positive moves the lyrics later. Every line and
// word shifts by the same amount, and the `.lrc` is rebuilt from the word
// timings so the export keeps agreeing with the player. This is the manual half
// of alignment: Whisper's own error is usually one constant shift.

/** The song row's generation params, or an empty object when it has none. */
function generationParams(song: any): any {
  try { return JSON.parse(song?.generation_params || '{}') || {}; } catch { return {}; }
}

/** A JSON sidecar as an object, or null when it is missing or unreadable. */
function readJsonOrNull(file: string): any | null {
  try { return JSON.parse(fs.readFileSync(file, 'utf-8')); } catch { return null; }
}

router.post('/lyrics/offset', async (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const audioUrl = typeof req.body?.audioUrl === 'string' ? req.body.audioUrl.trim() : '';
  const deltaMs = Number(req.body?.deltaMs);
  if (!audioUrl) { res.status(400).json({ error: 'audioUrl is required' }); return; }
  if (!Number.isFinite(deltaMs) || Math.abs(deltaMs) > 30_000) {
    res.status(400).json({ error: 'deltaMs must be a number within ±30000' });
    return;
  }

  try {
    const { shiftLyrics } = await import('../services/lyricsOffset.js');
    const song = getDb().prepare('SELECT * FROM songs WHERE audio_url = ?').get(audioUrl) as any;
    const gp = generationParams(song);
    const result = shiftLyrics(config.data.audioDir, audioUrl, deltaMs / 1000, {
      title: song?.title,
      artist: typeof gp.artist === 'string' ? gp.artist : '',
    });
    if (!result) {
      res.status(404).json({ error: 'No synced lyrics for this track' });
      return;
    }
    console.log(`[Lyrics] Shifted ${audioUrl} by ${deltaMs}ms (${result.lines} lines, ${result.source})`);
    res.json({ success: true, ...result });
  } catch (err: any) {
    console.error('[Lyrics] Offset failed:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/songs/lyrics/split — give a line a boundary at the playhead.
//
// Body: { audioUrl, atSec }. Whisper times a phrase as a unit, so on a slow
// passage one line covers several seconds and lights up long before its words
// are sung. Splitting at the playhead adds the missing boundary; word times stay
// where they are and the `.lrc` is rebuilt, so the strip, the waveform and the
// export all agree on where the second half starts.
router.post('/lyrics/split', async (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const audioUrl = typeof req.body?.audioUrl === 'string' ? req.body.audioUrl.trim() : '';
  const atSec = Number(req.body?.atSec);
  if (!audioUrl) { res.status(400).json({ error: 'audioUrl is required' }); return; }
  if (!Number.isFinite(atSec) || atSec <= 0 || atSec > 86_400) {
    res.status(400).json({ error: 'atSec must be a positive number of seconds' });
    return;
  }

  try {
    const { splitLyricLine } = await import('../services/lyricsSplit.js');
    const song = getDb().prepare('SELECT * FROM songs WHERE audio_url = ?').get(audioUrl) as any;
    const gp = generationParams(song);
    const result = splitLyricLine(config.data.audioDir, audioUrl, atSec, {
      title: song?.title,
      artist: typeof gp.artist === 'string' ? gp.artist : '',
    });
    if (!result.ok) {
      if (result.reason === 'no-sidecar') {
        res.status(404).json({ error: 'No synced lyrics for this track' });
        return;
      }
      res.status(400).json({
        error: result.reason === 'no-line'
          ? 'No lyric line at that position'
          : 'Nothing to split there: move the playhead into the middle of a line',
      });
      return;
    }
    console.log(`[Lyrics] Split ${audioUrl} at ${atSec.toFixed(2)}s: "${result.left}" / "${result.right}"`);
    res.json({ success: true, ...result });
  } catch (err: any) {
    console.error('[Lyrics] Split failed:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/songs/lyrics/timings — save hand-placed word timings from the
// lyric editor.
//
// Body: { audioUrl, lines }. The editor sends the whole array; the service sorts
// it, drops broken entries, recomputes each line's start/end from its words and
// rebuilds the `.lrc`, so a line no longer starts before its first syllable.
router.post('/lyrics/timings', async (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const audioUrl = typeof req.body?.audioUrl === 'string' ? req.body.audioUrl.trim() : '';
  if (!audioUrl) { res.status(400).json({ error: 'audioUrl is required' }); return; }

  try {
    const { saveLyricTimings } = await import('../services/lyricTimings.js');
    const song = getDb().prepare('SELECT * FROM songs WHERE audio_url = ?').get(audioUrl) as any;
    const gp = generationParams(song);
    const expectedRevision = typeof req.body?.expectedRevision === 'number' ? req.body.expectedRevision : undefined;
    const result = saveLyricTimings(config.data.audioDir, audioUrl, req.body?.lines, {
      title: song?.title,
      artist: typeof gp.artist === 'string' ? gp.artist : '',
    }, expectedRevision);
    if (!result.ok) {
      if (result.reason === 'no-sidecar') {
        res.status(404).json({ error: 'This track has no word timings to edit' });
        return;
      }
      if (result.reason === 'stale') {
        res.status(409).json({
          error: 'The lyric timings changed on disk. Reload the editor before saving.',
          code: 'stale',
          revision: result.revision,
        });
        return;
      }
      res.status(400).json({ error: 'No timed lyrics in that request' });
      return;
    }
    console.log(`[Lyrics] Saved timings for ${audioUrl}: ${result.lines} lines, ${result.words} words`);
    res.json({ success: true, ...result });
  } catch (err: any) {
    console.error('[Lyrics] Save timings failed:', err.message);
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/songs/retranscribe/model — which Whisper model a retranscription
 * would pick, plus the models installed.
 *
 * The pick is made server-side by priority, so the UI cannot name the model in
 * its progress toast without asking. No song id is involved: the choice is
 * global, and this route answers before a run starts.
 */
router.get('/retranscribe/model', async (_req, res) => {
  try {
    const { findWhisperModel, whisperModelLabel, listWhisperModels } =
      await import('../services/whisperTranscribe.js');
    const selected = findWhisperModel(undefined, { quiet: true });
    res.json({
      model: selected ? path.basename(selected) : null,
      label: selected ? whisperModelLabel(path.basename(selected)) : null,
      models: listWhisperModels(),
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

/** In-flight retranscriptions, so a disconnect or a Cancel click can stop one. */
const activeRetranscribes = new Map<string, AbortController>();

// POST /api/songs/:id/retranscribe — Re-run Whisper transcription
//
// Runs inline (CPU only, never the GPU lane) and can take minutes on a long
// track. Abortable: the response closing kills whisper-cli, and a cancelled
// run writes no sidecars.
router.post('/:id/retranscribe', async (req, res) => {
  const { id } = req.params;
  if (activeRetranscribes.has(id)) {
    return res.status(409).json({ error: 'A retranscription is already running for this track.' });
  }
  const controller = new AbortController();
  activeRetranscribes.set(id, controller);
  res.on('close', () => { if (!res.writableEnded) controller.abort(); });

  try {
    const song = getDb().prepare('SELECT * FROM songs WHERE id = ?').get(id) as any;
    if (!song) return res.status(404).json({ error: 'Song not found' });

    const audioFilename = song.audio_url ? path.basename(song.audio_url) : '';
    if (!audioFilename) return res.status(400).json({ error: 'No audio file' });
    const audioPath = path.join(config.data.audioDir, audioFilename);
    if (!fs.existsSync(audioPath)) return res.status(404).json({ error: 'Audio file not found' });

    const genParams = song.generation_params ? JSON.parse(song.generation_params) : {};
    const sourceLyrics = genParams.lyrics || song.lyrics || '';
    if (!sourceLyrics.trim()) {
      return res.status(400).json({ error: 'No source lyrics available' });
    }

    const { ensureWhisperCli, findWhisperModel, transcribeWithWhisper } = await import('../services/whisperTranscribe.js');
    const { reconcileLyrics } = await import('../services/lyricsReconcile.js');
    const { buildLrc } = await import('../services/lrcExport.js');

    const whisperReady = await ensureWhisperCli();
    if (!whisperReady.ok) {
      return res.status(400).json({ error: whisperReady.reason });
    }

    // Hand-placed timings are authoritative: a retranscription would replace
    // them with Whisper's guess, so it takes an explicit `force` to overwrite.
    const sidecarPath = path.join(config.data.audioDir, audioFilename.replace(/\.[^.]+$/, '.lyrics.json'));
    const existing = readJsonOrNull(sidecarPath);
    if (existing?.editedByHand === true && req.body?.force !== true) {
      return res.status(409).json({
        error: "This track's lyric timings were edited by hand. Click Retranscribe again to overwrite them.",
        code: 'manual-timings',
      });
    }
    const previousRevision = Number(existing?.revision) || 0;

    const whisperModel = req.body?.model || '';
    const modelPath = findWhisperModel(whisperModel);
    if (!modelPath) {
      return res.status(400).json({ error: 'No Whisper model found. Download one from the Model Manager.' });
    }

    console.log(`[Retranscribe] Song ${id}: starting`);

    const whisperResult = await transcribeWithWhisper(audioPath, sourceLyrics, {
      model: whisperModel,
      language: req.body?.language || 'auto',
      beamSize: req.body?.beamSize || 5,
      signal: controller.signal,
    });

    // Cancelled or navigated away: whisper-cli is dead, and writing a half run
    // over the existing sidecars would be worse than writing nothing.
    if (controller.signal.aborted) return;
    if (!whisperResult || !whisperResult.segments?.length) {
      return res.status(500).json({ error: 'Whisper returned no transcription' });
    }

    // Record the model that actually ran, not the (usually empty) request value.
    const modelName = path.basename(modelPath);
    const lyricsJson = reconcileLyrics(whisperResult, sourceLyrics, modelName, false);
    // Freshly machine-timed: clear the manual marker and move the revision on,
    // so an editor that was open on the old file is refused on its next save.
    (lyricsJson as any).editedByHand = false;
    (lyricsJson as any).revision = previousRevision + 1;

    const lyricsJsonFilename = audioFilename.replace(/\.[^.]+$/, '.lyrics.json');
    const lyricsJsonPath = path.join(config.data.audioDir, lyricsJsonFilename);
    fs.writeFileSync(lyricsJsonPath, JSON.stringify(lyricsJson, null, 2));

    // Refresh the `.lrc` too. The player prefers this sidecar, and it is the
    // file users export — leaving the engine's original alignment in place
    // would hand out timings that no longer match what the UI shows.
    const lrcFilename = audioFilename.replace(/\.[^.]+$/, '.lrc');
    const lrcPath = path.join(config.data.audioDir, lrcFilename);
    fs.writeFileSync(lrcPath, buildLrc(lyricsJson.lines, {
      title: song.title,
      artist: typeof genParams.artist === 'string' ? genParams.artist : '',
    }));

    const wordCount = lyricsJson.lines.reduce((n: number, l: any) => n + l.words.length, 0);
    console.log(`[Retranscribe] Song ${id}: saved ${lyricsJsonFilename} and ${lrcFilename} (${lyricsJson.lines.length} lines, ${wordCount} words)`);

    res.json({ success: true, lineCount: lyricsJson.lines.length, wordCount, model: modelName });
  } catch (err: any) {
    // The socket is gone when this is a cancel; there is nobody to answer.
    if (controller.signal.aborted) return;
    console.error('[Retranscribe] Error:', err.message);
    res.status(500).json({ error: err.message });
  } finally {
    activeRetranscribes.delete(id);
  }
});

// POST /api/songs/:id/postprocess — run the PP chain on an existing render
//
// For the render you made with the global post-processing toggle off and then
// decided you liked. The body carries the user's live PP settings; the master
// switch is forced on server-side, because sending this request IS the
// override. Returns immediately with a job id — the pass itself waits its turn
// on the GPU lane behind any running generation.
router.post('/:id/postprocess', (req, res) => {
  try {
    const song = getDb().prepare('SELECT * FROM songs WHERE id = ?').get(req.params.id) as any;

    const eligibility = checkPpEligibility(song);
    if (!eligibility.ok) {
      res.status(eligibility.status).json({
        error: eligibility.error,
        alreadyProcessed: eligibility.alreadyProcessed,
        masteredAudioUrl: eligibility.masteredAudioUrl,
      });
      return;
    }

    const params = (req.body && typeof req.body === 'object') ? req.body : {};

    // A chain with nothing switched on copies the WAV, changes nothing, deletes
    // the copy and reports success. Say so instead of spending a GPU slot on it.
    const stages = requestedPpStages({ ...params, postProcessingEnabled: true });
    if (stages.length === 0) {
      res.status(400).json({
        error: 'No post-processing stages are enabled. Turn on at least one (StableStep, PP-VAE, Spectral Lifter, Vocal Naturalizer, Gain Offset, a VST chain, Mastering or the Final Normalizer) and try again.',
      });
      return;
    }

    const job = startRePostProcess(song, params);
    res.json({ jobId: job.id, status: job.status, stage: job.stage, stages });
  } catch (err: any) {
    console.error('[PostProcess] Start failed:', err.message);
    res.status(err.status || 500).json({ error: err.message });
  }
});

// GET /api/songs/:id/postprocess — poll a re-run
//
// Scoped under the song rather than sitting at /postprocess/:jobId so it cannot
// be shadowed by GET /:id. With no jobId it reports whatever is in flight for
// this song, which is how a reloaded tab re-attaches to a running pass.
router.get('/:id/postprocess', (req, res) => {
  const jobId = req.query.jobId as string | undefined;
  const job = jobId ? getRePostProcessJob(jobId) : findRePostProcessJobBySong(req.params.id);
  if (!job) { res.status(404).json({ error: 'No post-processing job found' }); return; }
  res.json({
    jobId: job.id,
    songId: job.songId,
    status: job.status,
    stage: job.stage,
    error: job.error,
    masteredAudioUrl: job.masteredAudioUrl,
  });
});

// DELETE /api/songs/:id/postprocess — throw away the post-processed version
//
// The raw render is untouched; only the derived _mastered.wav goes. Exists so
// the no-double-cook rule cannot trap a track: a pass you regret (or one run
// with the wrong settings) can be undone and re-run properly.
router.delete('/:id/postprocess', (req, res) => {
  try {
    const song = getDb().prepare('SELECT * FROM songs WHERE id = ?').get(req.params.id) as any;
    const removed = revertPostProcessing(song);
    res.json({ removed, songId: req.params.id });
  } catch (err: any) {
    console.error('[PostProcess] Revert failed:', err.message);
    res.status(err.status || 500).json({ error: err.message });
  }
});

// Track in-flight extractions to prevent duplicate SuperSep jobs
const extractionsInFlight = new Set<string>();

// POST /api/songs/:id/extract-kick — extract kick drum stem for beat visualization
router.post('/:id/extract-kick', async (req, res) => {
  try {
    const song = getDb().prepare('SELECT * FROM songs WHERE id = ?')
      .get(req.params.id) as any;
    if (!song) { res.status(404).json({ error: 'Song not found' }); return; }
    if (!song.audio_url) { res.status(400).json({ error: 'No audio file' }); return; }

    // Already has disco data? (stems were analyzed and cleaned up)
    if (song.disco_data_url) {
      res.json({ status: 'exists', discoDataUrl: song.disco_data_url });
      return;
    }

    // Already has all drum stems but no disco data? Backfill analysis.
    if (song.kick_stem_url && song.snare_stem_url && song.hihat_stem_url) {
      let discoDataUrl = '';
      try {
        discoDataUrl = analyzeAndSaveDiscoData(req.params.id, config.data.audioDir, {
          kick: song.kick_stem_url,
          snare: song.snare_stem_url,
          hihat: song.hihat_stem_url,
        });
        if (discoDataUrl) {
          getDb().prepare('UPDATE songs SET disco_data_url = ? WHERE id = ?')
            .run(discoDataUrl, req.params.id);
        }
      } catch (err: any) {
        console.warn(`[KickExtract] Disco analysis backfill failed: ${err.message}`);
      }
      res.json({ status: 'exists', discoDataUrl });
      return;
    }

    // Guard against duplicate in-flight extractions
    if (extractionsInFlight.has(req.params.id)) {
      res.json({ status: 'in-progress' });
      return;
    }

    const ACE_URL = config.aceServer?.url || 'http://127.0.0.1:8085';

    // Resolve audio file path
    const audioFilename = path.basename(song.audio_url);
    const audioPath = path.join(config.data.audioDir, audioFilename);
    if (!fs.existsSync(audioPath)) {
      res.status(404).json({ error: 'Audio file not found on disk' });
      return;
    }

    console.log(`[KickExtract] Song ${req.params.id}: starting SuperSep level 2...`);

    // Send audio to ace-server SuperSep at level 2 (FULL — includes drum sub-separation)
    const audioBuf = fs.readFileSync(audioPath);
    const sepRes = await fetch(`${ACE_URL}/supersep/separate?level=2`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: audioBuf,
    });
    if (!sepRes.ok) {
      const errText = await sepRes.text();
      throw new Error(`SuperSep engine error: ${errText}`);
    }
    const { id: aceJobId } = await sepRes.json() as { id: string };

    console.log(`[KickExtract] Song ${req.params.id}: ace-server job ${aceJobId}`);
    extractionsInFlight.add(req.params.id);
    res.json({ status: 'started', aceJobId, stems: ['kick', 'snare', 'hihat'] });

    // Continue extraction in background (don't await in request handler)
    extractDrumStemsBackground(req.params.id, aceJobId, ACE_URL).catch(err => {
      console.error(`[KickExtract] Background extraction failed for ${req.params.id}:`, err.message);
      // Clear the extracting status
      getDb().prepare('UPDATE songs SET kick_stem_url = ?, snare_stem_url = ?, hihat_stem_url = ? WHERE id = ?')
        .run('', '', '', req.params.id);
    }).finally(() => {
      extractionsInFlight.delete(req.params.id);
    });
  } catch (err: any) {
    console.error('[KickExtract] Error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/songs/:id/analyze-disco — generate disco data from existing stems (no re-extraction)
router.post('/:id/analyze-disco', (req, res) => {
  try {
    const song = getDb().prepare('SELECT * FROM songs WHERE id = ?')
      .get(req.params.id) as any;
    if (!song) { res.status(404).json({ error: 'Song not found' }); return; }

    // Already has disco data?
    if (song.disco_data_url) {
      res.json({ status: 'exists', discoDataUrl: song.disco_data_url });
      return;
    }

    // Need at least one stem
    const stemUrls = {
      kick: song.kick_stem_url || undefined,
      snare: song.snare_stem_url || undefined,
      hihat: song.hihat_stem_url || undefined,
    };
    const hasStem = stemUrls.kick || stemUrls.snare || stemUrls.hihat;
    if (!hasStem) {
      res.status(400).json({ error: 'No stem files to analyze' });
      return;
    }

    const discoDataUrl = analyzeAndSaveDiscoData(req.params.id, config.data.audioDir, stemUrls);
    if (discoDataUrl) {
      getDb().prepare('UPDATE songs SET disco_data_url = ? WHERE id = ?')
        .run(discoDataUrl, req.params.id);
    }

    res.json({ status: 'created', discoDataUrl });
  } catch (err: any) {
    console.error('[DiscoAnalyze] Error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

async function extractDrumStemsBackground(songId: string, aceJobId: string, aceUrl: string): Promise<void> {
  // Poll until separation completes
  const MAX_POLLS = 3600; // 30 minutes max
  for (let i = 0; i < MAX_POLLS; i++) {
    const progRes = await fetch(`${aceUrl}/supersep/progress?id=${aceJobId}`);
    const progData = await progRes.json() as { status: string; progress: number; message: string; error?: string };

    if (progData.status === 'done') break;
    if (progData.status === 'failed' || progData.status === 'cancelled') {
      throw new Error(progData.error || `Separation ${progData.status}`);
    }
    await new Promise(r => setTimeout(r, 500));
  }

  // Fetch stem list
  const resultRes = await fetch(`${aceUrl}/supersep/result?id=${aceJobId}`);
  if (!resultRes.ok) throw new Error('Failed to fetch SuperSep result');
  const resultData = await resultRes.json() as { stems: Array<{ name: string; category: string; index: number; stage?: number; hidden?: boolean }> };

  // Helper: find, download, and save a stem
  async function downloadStem(
    stems: typeof resultData.stems,
    matchFn: (name: string) => boolean,
    suffix: string,
    label: string,
  ): Promise<string> {
    const stem = stems.find(s => matchFn(s.name.toLowerCase()));
    if (!stem) {
      console.warn(`[DrumStems] Song ${songId}: no ${label} stem found`);
      return '';
    }
    const stemRes = await fetch(`${aceUrl}/supersep/serve?id=${aceJobId}&stem=${stem.index}`);
    if (!stemRes.ok) {
      console.warn(`[DrumStems] Song ${songId}: failed to download ${label} stem`);
      return '';
    }
    const buf = Buffer.from(await stemRes.arrayBuffer());
    const filename = `${songId}_${suffix}.wav`;
    fs.writeFileSync(path.join(config.data.audioDir, filename), buf);
    console.log(`[DrumStems] Song ${songId}: ${label} stem saved (${(buf.length / 1024).toFixed(0)} KB)`);
    return `/audio/${filename}`;
  }

  // Download all three drum stems
  const kickUrl = await downloadStem(
    resultData.stems,
    name => name.includes('kick'),
    'kick', 'kick',
  );
  const snareUrl = await downloadStem(
    resultData.stems,
    name => name.includes('snare'),
    'snare', 'snare',
  );
  const hihatUrl = await downloadStem(
    resultData.stems,
    name => name.includes('hi-hat') || name.includes('hihat'),
    'hihat', 'hi-hat',
  );

  // Update DB with all stems at once
  getDb().prepare('UPDATE songs SET kick_stem_url = ?, snare_stem_url = ?, hihat_stem_url = ? WHERE id = ?')
    .run(kickUrl, snareUrl, hihatUrl, songId);

  // Analyze stems and save compact disco data JSON
  try {
    const discoDataUrl = analyzeAndSaveDiscoData(songId, config.data.audioDir, {
      kick: kickUrl,
      snare: snareUrl,
      hihat: hihatUrl,
    });
    if (discoDataUrl) {
      getDb().prepare('UPDATE songs SET disco_data_url = ? WHERE id = ?')
        .run(discoDataUrl, songId);
      console.log(`[DrumStems] Song ${songId}: disco data saved → ${discoDataUrl}`);

      // Clean up stem WAV files — disco JSON has all the data we need
      for (const stemUrl of [kickUrl, snareUrl, hihatUrl]) {
        if (!stemUrl) continue;
        const stemPath = path.join(config.data.audioDir, path.basename(stemUrl));
        try {
          if (fs.existsSync(stemPath)) {
            fs.unlinkSync(stemPath);
            console.log(`[DrumStems] Song ${songId}: deleted ${path.basename(stemUrl)}`);
          }
        } catch { /* non-fatal */ }
      }
      // Clear stem URLs from DB — files no longer exist
      getDb().prepare('UPDATE songs SET kick_stem_url = \'\', snare_stem_url = \'\', hihat_stem_url = \'\' WHERE id = ?')
        .run(songId);
    }
  } catch (err: any) {
    console.error(`[DrumStems] Song ${songId}: disco analysis failed (non-fatal):`, err.message);
  }

  console.log(`[DrumStems] Song ${songId}: all drum stems processed`);
}


export default router;
