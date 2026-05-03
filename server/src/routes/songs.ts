// songs.ts — Song CRUD routes + audio file serving
//
// Songs are stored in SQLite. Audio files are saved to data/audio/.

import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { getDb } from '../db/database.js';
import { config } from '../config.js';
import { getUserId } from './auth.js';
import { deleteAudioGenerationsByJobIds } from '../db/lireekDb.js';
import { cropWavFile, cropLrcFile } from '../services/audioCrop.js';

const router = Router();

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

// PATCH /api/songs/:id — update song
router.patch('/:id', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const song = getDb().prepare('SELECT * FROM songs WHERE id = ? AND user_id = ?')
    .get(req.params.id, userId) as any;
  if (!song) { res.status(404).json({ error: 'Song not found' }); return; }

  const updates = req.body;
  const allowed = ['title', 'lyrics', 'style', 'caption', 'cover_url', 'is_public',
    'bpm', 'key_scale', 'time_signature', 'dit_model'];

  for (const key of allowed) {
    if (updates[key] !== undefined) {
      const value = key === 'is_public' ? (updates[key] ? 1 : 0) : updates[key];
      getDb().prepare(`UPDATE songs SET ${key} = ? WHERE id = ?`).run(value, req.params.id);
    }
  }

  if (updates.tags) {
    getDb().prepare('UPDATE songs SET tags = ? WHERE id = ?')
      .run(JSON.stringify(updates.tags), req.params.id);
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

  getDb().prepare('DELETE FROM songs WHERE id = ?').run(req.params.id);

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
    }
  }

  // Get all song IDs before deleting (for Lireek cascade)
  const allSongs = getDb().prepare('SELECT id FROM songs WHERE user_id = ?').all(userId) as any[];
  const allIds = allSongs.map((s: any) => s.id);

  const result = getDb().prepare('DELETE FROM songs WHERE user_id = ?').run(userId);

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
    .prepare(`SELECT id, audio_url, mastered_audio_url FROM songs WHERE id IN (${placeholders}) AND user_id = ?`)
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
  }

  // Delete from DB
  const result = getDb()
    .prepare(`DELETE FROM songs WHERE id IN (${placeholders}) AND user_id = ?`)
    .run(...ids, userId);

  console.log(`[Songs] Bulk deleted ${result.changes}/${ids.length} songs`);

  // Cascade to Lireek DB
  try { deleteAudioGenerationsByJobIds(ids); } catch { /* lireek DB may not be initialized */ }

  res.json({ success: true, deletedCount: result.changes });
});

// POST /api/songs/nuke-generations — delete ALL generated audio across both databases + disk
router.post('/nuke-generations', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  // 1. Collect all audio file paths from songs table
  const songs = getDb()
    .prepare('SELECT id, audio_url, mastered_audio_url FROM songs WHERE user_id = ?')
    .all(userId) as any[];

  // 2. Delete audio files from disk
  let filesDeleted = 0;
  for (const song of songs) {
    for (const urlField of ['audio_url', 'mastered_audio_url']) {
      const url = song[urlField];
      if (url) {
        const filepath = path.join(config.data.audioDir, path.basename(url));
        if (fs.existsSync(filepath)) {
          try { fs.unlinkSync(filepath); filesDeleted++; } catch { /* best effort */ }
        }
      }
    }
  }

  // 3. Delete all songs from main DB
  const songIds = songs.map((s: any) => s.id);
  const songResult = getDb().prepare('DELETE FROM songs WHERE user_id = ?').run(userId);

  // 4. Delete all audio_generations (same DB now)
  let lireekDeleted = 0;
  try {
    // Delete by matching job IDs first (precise)
    if (songIds.length > 0) {
      lireekDeleted += deleteAudioGenerationsByJobIds(songIds);
    }
    // Also nuke ALL audio_generations (catches any orphans)
    const allResult = getDb().prepare('DELETE FROM audio_generations').run();
    lireekDeleted = Math.max(lireekDeleted, allResult.changes);
  } catch (err) {
    console.error('[Songs] NUKE audio_generations cleanup error:', err);
  }

  console.log(`[Songs] NUKE: ${songResult.changes} songs, ${filesDeleted} files, ${lireekDeleted} lireek audio_gens deleted`);

  res.json({
    success: true,
    songsDeleted: songResult.changes,
    filesDeleted,
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

    // 3. Crop companion LRC file (if exists)
    const lrcFilename = audioFilename.replace(/\.[^.]+$/, '.lrc');
    const lrcPath = path.join(config.data.audioDir, lrcFilename);
    if (fs.existsSync(lrcPath)) {
      cropLrcFile(lrcPath, inPoint, outPoint);
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

export default router;
