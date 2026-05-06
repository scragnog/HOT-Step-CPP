// lireek/llmRoutes.ts — LLM-powered routes for Lyric Studio
//
// Build Profile, Generate Lyrics, Refine Lyrics (streaming + non-streaming),
// Provider listing, Recalculate Stats, Skip Thinking.
// Registered on the parent router by lireek.ts.

import type { Router, Request, Response } from 'express';
import * as db from '../../db/lireekDb.js';
import * as llmService from '../../services/lireek/llmService.js';
import * as profilerService from '../../services/lireek/profilerService.js';

/** Safely extract a route param as string (Express 5 types params as string | string[]) */
function param(req: Request, name: string): string {
  const v = req.params[name];
  return Array.isArray(v) ? v[0] : v;
}

/** Parse an integer route param */
function intParam(req: Request, name: string): number {
  return parseInt(param(req, name), 10);
}

/** Set up SSE headers on the response */
function initSse(res: Response): (type: string, data: any) => void {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();
  return (type: string, data: any) => {
    res.write(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`);
  };
}

/** Resolve past generation history for diversity tracking */
function resolveHistory(artistId: number | undefined) {
  const pastGenerations = artistId
    ? db.getAllGenerationsWithContext().filter((g: any) => g.artist_id === artistId)
    : [];
  return {
    usedSubjects: pastGenerations.map((g: any) => g.subject || g.song_subject).filter(Boolean) as string[],
    usedBpms: pastGenerations.map((g: any) => g.bpm).filter((b: any): b is number => b !== null && b > 0) as number[],
    usedKeys: pastGenerations.map((g: any) => g.key || g.song_key).filter(Boolean) as string[],
    usedTitles: pastGenerations.map((g: any) => g.title).filter(Boolean) as string[],
    usedDurations: pastGenerations.map((g: any) => g.duration).filter((d: any): d is number => d !== null && d > 0) as number[],
  };
}

export function registerLlmRoutes(router: Router): void {

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

      const sendSse = initSse(res);

      const profileData = await profilerService.buildProfile(
        artist?.name || 'Unknown', 
        null, 
        lyricsSet.songs, 
        provider_name, 
        model,
        (phase) => sendSse('phase', { phase }),
        (chunk) => sendSse('chunk', { text: chunk })
      );
      const profile = db.saveProfile(lyricsSet.id, provider_name, model || '', profileData);
      sendSse('complete', profile);
      res.end();
    } catch (err: any) {
      res.write(`event: error\ndata: ${JSON.stringify({ error: err.message })}\n\n`);
      res.end();
    }
  });

  // ── Recalculate Stats (no LLM) ──────────────────────────────────────────────

  router.post('/profiles/recalculate-stats', async (_req: Request, res: Response) => {
    try {
      const profiles = db.getProfiles();
      let updated = 0;
      for (const profile of profiles) {
        const lyricsSet = db.getLyricsSet(profile.lyrics_set_id);
        if (!lyricsSet) continue;
        const songs = (lyricsSet.songs || []) as { title: string; album?: string; lyrics: string }[];
        if (!songs.length) continue;
        const patched = profilerService.recalculateProfileStats(songs, profile.profile_data);
        db.updateProfileData(profile.id, patched);
        updated++;
      }
      res.json({ updated, total: profiles.length });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Generate Lyrics ─────────────────────────────────────────────────────────

  router.post('/profiles/:id/generate', async (req: Request, res: Response) => {
    try {
      const id = intParam(req, 'id');
      const profile = db.getProfile(id);
      if (!profile) return res.status(404).json({ error: 'Profile not found' });
      
      const { provider_name, model, extra_instructions, auto_save = true } = req.body;
      
      const lyricsSet = db.getLyricsSet(profile.lyrics_set_id);
      const artistId = lyricsSet?.artist_id;
      const { usedSubjects, usedBpms, usedKeys, usedTitles, usedDurations } = resolveHistory(artistId);

      const generated = await llmService.generateLyricsStreaming(
        profile.profile_data, provider_name, model, extra_instructions, 
        usedSubjects, usedBpms, usedKeys, usedTitles, usedDurations
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
      
      const sendSse = initSse(res);
      llmService.resetSkipThinking();

      const lyricsSet = db.getLyricsSet(profile.lyrics_set_id);
      const artistId = lyricsSet?.artist_id;
      const { usedSubjects, usedBpms, usedKeys, usedTitles, usedDurations } = resolveHistory(artistId);

      const generated = await llmService.generateLyricsStreaming(
        profile.profile_data, provider_name, model, extra_instructions,
        usedSubjects, usedBpms, usedKeys, usedTitles, usedDurations,
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
      
      const sendSse = initSse(res);
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
}
