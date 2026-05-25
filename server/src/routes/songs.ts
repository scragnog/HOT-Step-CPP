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
import { analyzeAndSaveDiscoData } from '../services/disco-analyzer.js';

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
         JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
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

// PATCH /api/songs/:id — update song
router.patch('/:id', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const song = getDb().prepare('SELECT * FROM songs WHERE id = ? AND user_id = ?')
    .get(req.params.id, userId) as any;
  if (!song) { res.status(404).json({ error: 'Song not found' }); return; }

  const updates = req.body;
  const allowed = ['title', 'lyrics', 'style', 'caption', 'cover_url', 'is_public',
    'bpm', 'key_scale', 'time_signature', 'dit_model', 'cover_art_subject'];

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


// POST /api/songs/:id/retranscribe — Re-run Whisper transcription
router.post('/:id/retranscribe', async (req, res) => {
  try {
    const { id } = req.params;
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

    const whisperReady = await ensureWhisperCli();
    if (!whisperReady) {
      return res.status(400).json({ error: 'Whisper CLI not available and auto-download failed.' });
    }

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
    });

    if (!whisperResult || !whisperResult.segments?.length) {
      return res.status(500).json({ error: 'Whisper returned no transcription' });
    }

    const lyricsJson = reconcileLyrics(whisperResult, sourceLyrics, whisperModel || 'auto', false);

    const lyricsJsonFilename = audioFilename.replace(/\.[^.]+$/, '.lyrics.json');
    const lyricsJsonPath = path.join(config.data.audioDir, lyricsJsonFilename);
    fs.writeFileSync(lyricsJsonPath, JSON.stringify(lyricsJson, null, 2));

    const wordCount = lyricsJson.lines.reduce((n: number, l: any) => n + l.words.length, 0);
    console.log(`[Retranscribe] Song ${id}: saved ${lyricsJsonFilename} (${lyricsJson.lines.length} lines, ${wordCount} words)`);

    res.json({ success: true, lineCount: lyricsJson.lines.length, wordCount });
  } catch (err: any) {
    console.error('[Retranscribe] Error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/songs/:id/extract-kick — extract kick drum stem for beat visualization
router.post('/:id/extract-kick', async (req, res) => {
  try {
    const song = getDb().prepare('SELECT * FROM songs WHERE id = ?')
      .get(req.params.id) as any;
    if (!song) { res.status(404).json({ error: 'Song not found' }); return; }
    if (!song.audio_url) { res.status(400).json({ error: 'No audio file' }); return; }

    // Already has all drum stems?
    if (song.kick_stem_url && song.snare_stem_url && song.hihat_stem_url) {
      // Backfill disco data if stems exist but analysis doesn't
      let discoDataUrl = song.disco_data_url || '';
      if (!discoDataUrl) {
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
      }
      res.json({
        status: 'exists',
        kickStemUrl: song.kick_stem_url,
        snareStemUrl: song.snare_stem_url,
        hihatStemUrl: song.hihat_stem_url,
        discoDataUrl,
      });
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

    // Return immediately — client will poll for status
    // Store the ace job ID on the song for polling
    getDb().prepare('UPDATE songs SET kick_stem_url = ? WHERE id = ?')
      .run(`extracting:${aceJobId}`, req.params.id);

    console.log(`[KickExtract] Song ${req.params.id}: ace-server job ${aceJobId}`);
    res.json({ status: 'started', aceJobId, stems: ['kick', 'snare', 'hihat'] });

    // Continue extraction in background (don't await in request handler)
    extractDrumStemsBackground(req.params.id, aceJobId, ACE_URL).catch(err => {
      console.error(`[KickExtract] Background extraction failed for ${req.params.id}:`, err.message);
      // Clear the extracting status
      getDb().prepare('UPDATE songs SET kick_stem_url = ?, snare_stem_url = ?, hihat_stem_url = ? WHERE id = ?')
        .run('', '', '', req.params.id);
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
