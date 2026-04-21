// lireek.ts — Express routes for Lyric Studio / Lireek
//
// All endpoints under /api/lireek/*
// Phase 1: CRUD routes for artists, lyrics sets, profiles, generations
// Phase 2: LLM-powered routes (build-profile, generate, refine) — TODO

import { Router, type Request, type Response } from 'express';
import { getDb } from '../db/database.js';
import * as db from '../db/lireekDb.js';
import * as genius from '../services/lireek/geniusService.js';
import { exportGeneration } from '../services/lireek/exportService.js';
import { scanForSlop, BLACKLISTED_WORDS, BLACKLISTED_PHRASES } from '../services/lireek/slopDetector.js';
import * as llmService from '../services/lireek/llmService.js';
import * as profilerService from '../services/lireek/profilerService.js';

const router = Router();

// ── Helper: enrich rows with mastered_audio_url from the main songs DB ──────
function enrichWithMasteredUrls(rows: Record<string, any>[]): Record<string, any>[] {
  const audioUrls = rows.map(r => r.audio_url).filter(Boolean);
  if (audioUrls.length === 0) return rows;
  try {
    const placeholders = audioUrls.map(() => '?').join(',');
    const songs = getDb().prepare(
      `SELECT audio_url, mastered_audio_url FROM songs WHERE audio_url IN (${placeholders})`
    ).all(...audioUrls) as any[];
    const map = new Map(songs.map((s: any) => [s.audio_url, s.mastered_audio_url]));
    return rows.map(r => ({
      ...r,
      mastered_audio_url: map.get(r.audio_url) || '',
    }));
  } catch {
    return rows;
  }
}

/** Safely extract a route param as string (Express 5 types params as string | string[]) */
function param(req: Request, name: string): string {
  const v = req.params[name];
  return Array.isArray(v) ? v[0] : v;
}

/** Parse an integer route param */
function intParam(req: Request, name: string): number {
  return parseInt(param(req, name), 10);
}

// ── Artists ─────────────────────────────────────────────────────────────────

router.get('/artists', (_req: Request, res: Response) => {
  try {
    const artists = db.listArtists();
    res.json(artists);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/artists/create', (req: Request, res: Response) => {
  try {
    const { name } = req.body;
    if (!name?.trim()) {
      res.status(400).json({ error: 'Artist name required' });
      return;
    }
    const artist = db.getOrCreateArtist(name.trim());
    res.json(artist);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.delete('/artists/:id', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const deleted = db.deleteArtist(id);
    if (!deleted) { res.status(404).json({ error: 'Artist not found' }); return; }
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/artists/:id/refresh-image', async (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const artist = db.getArtist(id);
    if (!artist) { res.status(404).json({ error: 'Artist not found' }); return; }
    const imageUrl = await genius.getArtistImageUrl(artist.name);
    if (imageUrl) db.updateArtistImage(id, imageUrl);
    res.json({ image_url: imageUrl });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/artists/:id/set-image', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const { image_url } = req.body;
    db.updateArtistImage(id, image_url ?? null);
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Lyrics Sets ─────────────────────────────────────────────────────────────

router.get('/lyrics-sets', (req: Request, res: Response) => {
  try {
    const artistId = req.query.artist_id ? parseInt(req.query.artist_id as string, 10) : undefined;
    const sets = db.getLyricsSets(artistId);
    res.json(sets);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/lyrics-sets/create', (req: Request, res: Response) => {
  try {
    const { artist_name, album, songs } = req.body;
    if (!artist_name?.trim() || !Array.isArray(songs)) {
      res.status(400).json({ error: 'artist_name and songs array required' });
      return;
    }
    const artist = db.getOrCreateArtist(artist_name.trim());
    const set = db.saveLyricsSet(artist.id as number, album ?? null, songs.length, songs);
    res.json(set);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/lyrics-sets/:id', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const set = db.getLyricsSet(id);
    if (!set) { res.status(404).json({ error: 'Lyrics set not found' }); return; }
    res.json(set);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/lyrics-sets/:id/full-detail', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const set = db.getLyricsSet(id);
    if (!set) { res.status(404).json({ error: 'Lyrics set not found' }); return; }
    const profiles = db.getProfiles(id);
    const generations = db.getGenerations(undefined, id);
    const preset = db.getPreset(id);
    res.json({ lyrics_set: set, profiles, generations, preset });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.delete('/lyrics-sets/:id', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const deleted = db.deleteLyricsSet(id);
    if (!deleted) { res.status(404).json({ error: 'Lyrics set not found' }); return; }
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.delete('/lyrics-sets/:id/songs/:index', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const index = intParam(req, 'index');
    const updated = db.removeSongFromSet(id, index);
    if (!updated) { res.status(404).json({ error: 'Song not found' }); return; }
    res.json(updated);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.put('/lyrics-sets/:id/songs/:index', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const index = intParam(req, 'index');
    const { lyrics } = req.body;
    const updated = db.editSongInSet(id, index, lyrics);
    if (!updated) { res.status(404).json({ error: 'Song not found' }); return; }
    res.json(updated);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/lyrics-sets/:id/refresh-image', async (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const set = db.getLyricsSet(id);
    if (!set) { res.status(404).json({ error: 'Lyrics set not found' }); return; }
    const imageUrl = set.album
      ? await genius.getAlbumImageUrl(set.album, set.artist_name)
      : await genius.getArtistImageUrl(set.artist_name);
    if (imageUrl) db.updateLyricsSetImage(id, imageUrl);
    res.json({ image_url: imageUrl });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/lyrics-sets/:id/set-image', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const { image_url } = req.body;
    db.updateLyricsSetImage(id, image_url ?? null);
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/lyrics-sets/:id/add-song', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const { title, album, lyrics } = req.body;
    if (!title || !lyrics) {
      res.status(400).json({ error: 'title and lyrics required' });
      return;
    }
    const updated = db.addSongToSet(id, { title, album, lyrics });
    if (!updated) { res.status(404).json({ error: 'Lyrics set not found' }); return; }
    res.json(updated);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Fetch Lyrics (Genius) ───────────────────────────────────────────────────

router.post('/fetch-lyrics', async (req: Request, res: Response) => {
  try {
    const { artist, album, max_songs = 10 } = req.body;
    if (!artist?.trim()) {
      res.status(400).json({ error: 'Artist name required' });
      return;
    }

    const result = await genius.fetchLyrics(artist.trim(), album?.trim() || null, max_songs);

    // Save to database
    const artistRow = db.getOrCreateArtist(result.artist);

    // Try to get artist image if we don't have one
    if (!artistRow.image_url) {
      genius.getArtistImageUrl(result.artist).then(url => {
        if (url) db.updateArtistImage(artistRow.id as number, url);
      }).catch(() => {});
    }

    // Try to get album image
    let albumImageUrl: string | null = null;
    if (result.album) {
      try {
        albumImageUrl = await genius.getAlbumImageUrl(result.album, result.artist);
      } catch {}
    }

    const lyricsSet = db.saveLyricsSet(
      artistRow.id as number,
      result.album,
      result.songs.length,
      result.songs,
      albumImageUrl,
    );

    res.json({
      ...result,
      artist_id: artistRow.id,
      lyrics_set_id: lyricsSet.id,
    });
  } catch (err: any) {
    const status = err.message?.includes('not found') || err.message?.includes('No lyrics') ? 404 : 500;
    res.status(status).json({ error: err.message });
  }
});

router.post('/search-song-lyrics', async (req: Request, res: Response) => {
  try {
    const { artist, title } = req.body;
    if (!artist || !title) {
      res.status(400).json({ error: 'artist and title required' });
      return;
    }
    const result = await genius.searchSongLyrics(artist, title);
    if (!result) { res.status(404).json({ error: 'Song not found' }); return; }
    res.json(result);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Profiles ────────────────────────────────────────────────────────────────

router.get('/profiles', (req: Request, res: Response) => {
  try {
    const lyricsSetId = req.query.lyrics_set_id ? parseInt(req.query.lyrics_set_id as string, 10) : undefined;
    res.json(db.getProfiles(lyricsSetId));
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/profiles/:id', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const profile = db.getProfile(id);
    if (!profile) { res.status(404).json({ error: 'Profile not found' }); return; }
    res.json(profile);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.delete('/profiles/:id', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const deleted = db.deleteProfile(id);
    if (!deleted) { res.status(404).json({ error: 'Profile not found' }); return; }
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Generations ─────────────────────────────────────────────────────────────

router.get('/generations', (req: Request, res: Response) => {
  try {
    const profileId = req.query.profile_id ? parseInt(req.query.profile_id as string, 10) : undefined;
    const lyricsSetId = req.query.lyrics_set_id ? parseInt(req.query.lyrics_set_id as string, 10) : undefined;
    res.json(db.getGenerations(profileId, lyricsSetId));
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/generations/all', (_req: Request, res: Response) => {
  try {
    res.json(db.getAllGenerationsWithContext());
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/generations/:id', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const gen = db.getGeneration(id);
    if (!gen) { res.status(404).json({ error: 'Generation not found' }); return; }
    res.json(gen);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.patch('/generations/:id', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    db.updateGenerationFields(id, req.body);
    const updated = db.getGeneration(id);
    if (!updated) { res.status(404).json({ error: 'Generation not found' }); return; }
    res.json(updated);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.delete('/generations/:id', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const deleted = db.deleteGeneration(id);
    if (!deleted) { res.status(404).json({ error: 'Generation not found' }); return; }
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Export ───────────────────────────────────────────────────────────────────

router.post('/generations/:id/export', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const gen = db.getGeneration(id);
    if (!gen) { res.status(404).json({ error: 'Generation not found' }); return; }

    // Get context (artist name, album)
    const profile = db.getProfile(gen.profile_id);
    let artistName = 'Unknown';
    let albumName: string | undefined;
    if (profile) {
      const lyricsSet = db.getLyricsSet(profile.lyrics_set_id);
      if (lyricsSet) {
        artistName = lyricsSet.artist_name;
        albumName = lyricsSet.album ?? undefined;
      }
    }

    const paths = exportGeneration({
      title: gen.title,
      lyrics: gen.lyrics,
      artistName,
      albumName,
      provider: gen.provider,
      model: gen.model,
      bpm: gen.bpm,
      key: gen.key,
      caption: gen.caption,
      duration: gen.duration,
      subject: gen.subject,
      extraInstructions: gen.extra_instructions,
      createdAt: gen.created_at,
    });
    res.json({ success: true, ...paths });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Audio Generations ───────────────────────────────────────────────────────

router.post('/generations/:id/audio', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const { job_id } = req.body;
    if (!job_id) { res.status(400).json({ error: 'job_id required' }); return; }
    const link = db.linkAudioGeneration(id, job_id);
    res.json(link);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/generations/:id/audio', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const rows = db.getAudioGenerations(id);
    res.json({ audio_generations: enrichWithMasteredUrls(rows) });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.delete('/audio-generations/:id', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const deleted = db.deleteAudioGeneration(id);
    if (!deleted) { res.status(404).json({ error: 'Audio generation not found' }); return; }
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.patch('/audio-generations/resolve', (req: Request, res: Response) => {
  try {
    const { job_id, audio_url, cover_url } = req.body;
    if (!job_id || !audio_url) { res.status(400).json({ error: 'job_id and audio_url required' }); return; }
    db.resolveAudioGeneration(job_id, audio_url, cover_url);
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Album Presets ───────────────────────────────────────────────────────────

/** Parse adapter_group_scales from JSON string to object if needed */
function hydratePreset(preset: any): any {
  if (!preset) return null;
  const hydrated = { ...preset };
  if (typeof hydrated.adapter_group_scales === 'string') {
    try { hydrated.adapter_group_scales = JSON.parse(hydrated.adapter_group_scales); }
    catch { hydrated.adapter_group_scales = null; }
  }
  return hydrated;
}

router.get('/lyrics-sets/:id/preset', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const preset = db.getPreset(id);
    res.json({ preset: hydratePreset(preset) });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.put('/lyrics-sets/:id/preset', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const preset = db.upsertPreset(id, {
      adapterPath: req.body.adapter_path,
      adapterScale: req.body.adapter_scale,
      adapterGroupScales: req.body.adapter_group_scales,
      referenceTrackPath: req.body.reference_track_path,
      audioCoverStrength: req.body.audio_cover_strength,
    });
    res.json({ preset: hydratePreset(preset) });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.delete('/lyrics-sets/:id/preset', (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    db.deletePreset(id);
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/presets', (_req: Request, res: Response) => {
  try {
    const presets = db.getAllPresets().map(hydratePreset);
    res.json({ presets });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Slop Scanner ────────────────────────────────────────────────────────────

router.post('/slop-scan', (req: Request, res: Response) => {
  try {
    const { text, fingerprint, statistical_weight } = req.body;
    if (!text) { res.status(400).json({ error: 'text required' }); return; }
    const result = scanForSlop(text, fingerprint, statistical_weight);
    res.json(result);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Purge ───────────────────────────────────────────────────────────────────

router.post('/purge', (_req: Request, res: Response) => {
  try {
    const result = db.purgeProfilesAndGenerations();
    res.json(result);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Settings / Prompts ──────────────────────────────────────────────────────

router.get('/prompts', (_req: Request, res: Response) => {
  // Return list of customizable prompt names with current values
  const names = ['generation_system', 'metadata_system', 'profile_system', 'refine_system'];
  const prompts = names.map(name => ({
    name,
    custom: db.getSetting(`prompt_${name}`) || null,
  }));
  res.json(prompts);
});

router.put('/prompts/:name', (req: Request, res: Response) => {
  try {
    const promptName = param(req, 'name');
    const { value } = req.body;
    if (!value) { res.status(400).json({ error: 'value required' }); return; }
    db.setSetting(`prompt_${promptName}`, value);
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.delete('/prompts/:name', (req: Request, res: Response) => {
  try {
    const promptName = param(req, 'name');
    db.setSetting(`prompt_${promptName}`, '');
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Recent Songs ────────────────────────────────────────────────────────────

router.get('/recent-songs', (req: Request, res: Response) => {
  try {
    const limit = parseInt(req.query.limit as string, 10) || 50;
    const rows = db.getRecentGenerationsWithAudio(limit);
    res.json({ songs: enrichWithMasteredUrls(rows) });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── LLM Providers ───────────────────────────────────────────────────────────

router.get('/providers', async (_req: Request, res: Response) => {
  try {
    const providers = await llmService.listProviders();
    res.json(providers);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Build Profile ───────────────────────────────────────────────────────────

router.post('/lyrics-sets/:id/build-profile', async (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const { provider_name, model } = req.body;
    const lyricsSet = db.getLyricsSet(id);
    if (!lyricsSet) return res.status(404).json({ error: 'Lyrics set not found' });
    const artist = db.getArtist(lyricsSet.artist_id);
    if (!artist) return res.status(404).json({ error: 'Artist not found' });
    
    const profileData = await profilerService.buildProfile(
      artist?.name || 'Unknown', 
      null, 
      lyricsSet.songs, 
      provider_name, 
      model
    );
    const profile = db.saveProfile(lyricsSet.id, provider_name, model || '', profileData);
    res.json(profile);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/lyrics-sets/:id/build-profile-stream', async (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const { provider_name, model } = req.body;
    const lyricsSet = db.getLyricsSet(id);
    if (!lyricsSet) throw new Error('Lyrics set not found');
    const artist = db.getArtist(lyricsSet.artist_id);
    if (!artist) throw new Error('Artist not found');

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const sendSse = (type: string, data: any) => {
      res.write(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    const profileData = await profilerService.buildProfile(
      artist?.name || 'Unknown', 
      null, 
      lyricsSet.songs, 
      provider_name, 
      model,
      (phase) => sendSse('phase', { phase })
    );
    const profile = db.saveProfile(lyricsSet.id, provider_name, model || '', profileData);
    sendSse('complete', profile);
    res.end();
  } catch (err: any) {
    res.write(`event: error\ndata: ${JSON.stringify({ error: err.message })}\n\n`);
    res.end();
  }
});

// ── Generate Lyrics ─────────────────────────────────────────────────────────

router.post('/profiles/:id/generate', async (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const profile = db.getProfile(id);
    if (!profile) return res.status(404).json({ error: 'Profile not found' });
    
    const { provider_name, model, extra_instructions, auto_save = true } = req.body;
    
    // Resolve artist_id via the lyrics_set (profiles table doesn't have artist_id)
    const lyricsSet = db.getLyricsSet(profile.lyrics_set_id);
    const artistId = lyricsSet?.artist_id;
    const pastGenerations = artistId
      ? db.getAllGenerationsWithContext().filter((g: any) => g.artist_id === artistId)
      : [];
    const usedSubjects = pastGenerations.map((g: any) => g.song_subject).filter(Boolean);
    const usedBpms = pastGenerations.map((g: any) => g.bpm).filter((b: any): b is number => b !== null && b > 0);
    const usedKeys = pastGenerations.map((g: any) => g.song_key).filter(Boolean);
    const usedTitles = pastGenerations.map((g: any) => g.title).filter(Boolean);

    const generated = await llmService.generateLyricsStreaming(
      profile.profile_data, provider_name, model, extra_instructions, 
      usedSubjects as string[], usedBpms as number[], usedKeys as string[], usedTitles as string[]
    );
    
    if (auto_save) {
      const saved = db.saveGeneration({
        profileId: id,
        provider: provider_name,
        model: generated.model,
        lyrics: generated.lyrics,
        title: generated.title,
        subject: generated.subject,
        bpm: generated.bpm || undefined,
        key: generated.key,
        caption: generated.caption,
        duration: generated.duration || undefined,
        systemPrompt: generated.system_prompt,
        userPrompt: generated.user_prompt
      });
      res.json(saved);
    } else {
      res.json(generated);
    }
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/profiles/:id/generate-stream', async (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const profile = db.getProfile(id);
    if (!profile) throw new Error('Profile not found');
    
    const { provider_name, model, extra_instructions, auto_save = true } = req.body;
    
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const sendSse = (type: string, data: any) => {
      res.write(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    llmService.resetSkipThinking();

    // Resolve artist_id via the lyrics_set (profiles table doesn't have artist_id)
    const lyricsSet = db.getLyricsSet(profile.lyrics_set_id);
    const artistId = lyricsSet?.artist_id;
    const pastGenerations = artistId
      ? db.getAllGenerationsWithContext().filter((g: any) => g.artist_id === artistId)
      : [];
    const usedSubjects = pastGenerations.map((g: any) => g.subject).filter(Boolean);
    const usedBpms = pastGenerations.map((g: any) => g.bpm).filter((b: any): b is number => b !== null && b > 0);
    const usedKeys = pastGenerations.map((g: any) => g.key).filter(Boolean);
    const usedTitles = pastGenerations.map((g: any) => g.title).filter(Boolean);

    const generated = await llmService.generateLyricsStreaming(
      profile.profile_data, provider_name, model, extra_instructions,
      usedSubjects as string[], usedBpms as number[], usedKeys as string[], usedTitles as string[],
      (chunk) => sendSse('chunk', { text: chunk }),
      (phase) => sendSse('phase', { phase })
    );

    if (auto_save) {
      const saved = db.saveGeneration({
        profileId: id,
        provider: provider_name,
        model: generated.model,
        lyrics: generated.lyrics,
        title: generated.title,
        subject: generated.subject,
        bpm: generated.bpm || undefined,
        key: generated.key,
        caption: generated.caption,
        duration: generated.duration || undefined,
        systemPrompt: generated.system_prompt,
        userPrompt: generated.user_prompt
      });
      sendSse('complete', saved);
    } else {
      sendSse('complete', generated);
    }
    res.end();
  } catch (err: any) {
    res.write(`event: error\ndata: ${JSON.stringify({ error: err.message })}\n\n`);
    res.end();
  }
});

// ── Refine Lyrics ───────────────────────────────────────────────────────────

router.post('/generations/:id/refine', async (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const existing = db.getGeneration(id);
    if (!existing) return res.status(404).json({ error: 'Generation not found' });
    
    const { provider_name, model, auto_save = true } = req.body;
    const profile = db.getProfile(existing.profile_id);
    const lyricsSet = profile ? db.getLyricsSet(profile.lyrics_set_id) : null;
    const artist = lyricsSet ? db.getArtist(lyricsSet.artist_id) : undefined;

    const refined = await llmService.refineLyricsStreaming(
      existing.lyrics, artist?.name || 'Unknown', existing.title, provider_name, model, profile?.profile_data || undefined
    );

    if (auto_save) {
      const saved = db.saveGeneration({
        profileId: existing.profile_id,
        provider: provider_name,
        model: refined.model,
        lyrics: refined.lyrics,
        title: refined.title,
        subject: existing.song_subject,
        bpm: existing.bpm || undefined,
        key: existing.song_key,
        caption: existing.caption,
        duration: existing.duration || undefined,
        systemPrompt: refined.system_prompt,
        userPrompt: refined.user_prompt,
        parentGenerationId: existing.id
      });
      res.json(saved);
    } else {
      res.json(refined);
    }
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/generations/:id/refine-stream', async (req: Request, res: Response) => {
  try {
    const id = intParam(req, 'id');
    const existing = db.getGeneration(id);
    if (!existing) throw new Error('Generation not found');
    
    const { provider_name, model, auto_save = true } = req.body;
    const profile = db.getProfile(existing.profile_id);
    const lyricsSet = profile ? db.getLyricsSet(profile.lyrics_set_id) : null;
    const artist = lyricsSet ? db.getArtist(lyricsSet.artist_id) : undefined;
    
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const sendSse = (type: string, data: any) => {
      res.write(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    llmService.resetSkipThinking();

    const refined = await llmService.refineLyricsStreaming(
      existing.lyrics, artist?.name || 'Unknown', existing.title, provider_name, model, profile?.profile_data || undefined,
      (chunk) => sendSse('chunk', { text: chunk })
    );

    if (auto_save) {
      const saved = db.saveGeneration({
        profileId: existing.profile_id,
        provider: provider_name,
        model: refined.model,
        lyrics: refined.lyrics,
        title: refined.title,
        subject: existing.song_subject,
        bpm: existing.bpm || undefined,
        key: existing.song_key,
        caption: existing.caption,
        duration: existing.duration || undefined,
        systemPrompt: refined.system_prompt,
        userPrompt: refined.user_prompt,
        parentGenerationId: existing.id
      });
      sendSse('complete', saved);
    } else {
      sendSse('complete', refined);
    }
    res.end();
  } catch (err: any) {
    res.write(`event: error\ndata: ${JSON.stringify({ error: err.message })}\n\n`);
    res.end();
  }
});

// ── Curated Profile ─────────────────────────────────────────────────────────

router.post('/artists/:id/curated-profile', async (_req: Request, res: Response) => {
  res.status(501).json({ error: 'Curated profiles not yet implemented in TS' });
});

router.post('/artists/:id/curated-profile-stream', async (_req: Request, res: Response) => {
  res.status(501).json({ error: 'Curated profiles not yet implemented in TS' });
});

// ── Skip Thinking ───────────────────────────────────────────────────────────

router.post('/skip-thinking', (_req: Request, res: Response) => {
  llmService.setSkipThinking();
  res.json({ success: true });
});

export default router;
