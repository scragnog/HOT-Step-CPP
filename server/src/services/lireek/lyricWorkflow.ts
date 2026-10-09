// Durable Lyric Studio batches. The submit route captures every source and option.
import { z } from 'zod/v4';
import * as db from '../../db/lireekDb.js';
import * as genius from './geniusService.js';
import * as llm from './llmService.js';
import * as profiler from './profilerService.js';
import { computeAlbumEnrichment } from './prompts.js';
import { requestVersion } from '../generation/resolve/resolveIntent.js';
import { resolveWrittenSongIntent } from '../generation/resolve/resolveIntent.js';
import { loadWrittenSongData } from '../generation/resolve/loadIntentData.js';
import { getActiveBackendId } from '../backends/registry.js';
import { writtenSongIntentSchema, type WrittenSongIntent } from '../../contracts/resolution.js';
import type { LyricsProfile } from './profilerService.js';
import { registerWorkflowKind } from '../../routes/workflows.js';
import type { WorkflowContext } from '../workflows/workflowJobs.js';
import type { AudioIntentItem } from '../../contracts/audioQueue.js';
import {
  lyricBatchInputSchema as lyricBatchInput, lyricItemBase as base,
  lyricFetchItem as fetchItem, lyricRenderItem as render, type LyricBatchInput,
} from '../../contracts/studioWorkflows.js';

export { lyricBatchInput, type LyricBatchInput };
type Item = LyricBatchInput['items'][number];
export type LyricRequest = { type: Item['type']; targetId?: number; provider?: string; model?: string; count?: number; extraInstructions?: string; userSubject?: string; noThink?: boolean; artist?: string; album?: string; maxSongs?: number };
const revision = (value: unknown) => requestVersion(value as Record<string, unknown>);

/** A completed item and its library write commit together. Retry can skip it. */
function completedItem(jobId: string, index: number): Record<string, unknown> | null {
  const database = db.getLireekDb();
  database.exec(`CREATE TABLE IF NOT EXISTS lyric_workflow_items (
    job_id TEXT NOT NULL, item_index INTEGER NOT NULL, result TEXT NOT NULL,
    PRIMARY KEY (job_id, item_index)
  )`);
  const row = database.prepare('SELECT result FROM lyric_workflow_items WHERE job_id = ? AND item_index = ?')
    .get(jobId, index) as { result: string } | undefined;
  return row ? JSON.parse(row.result) : null;
}

function saveItem(jobId: string, index: number, write: () => Record<string, unknown>): Record<string, unknown> {
  const database = db.getLireekDb();
  return database.transaction(() => {
    const existing = completedItem(jobId, index);
    if (existing) return existing;
    const result = write();
    database.prepare('INSERT INTO lyric_workflow_items (job_id, item_index, result) VALUES (?, ?, ?)')
      .run(jobId, index, JSON.stringify(result));
    return result;
  })();
}

function captureHistory(artistId: number | undefined) {
  const past = artistId ? db.getAllGenerationsWithContext().filter((g: any) => g.artist_id === artistId) : [];
  return {
    usedSubjects: past.map((g: any) => g.subject || g.song_subject).filter(Boolean) as string[],
    usedBpms: past.map((g: any) => g.bpm).filter((b: any) => b !== null && b > 0) as number[],
    usedKeys: past.map((g: any) => g.key || g.song_key).filter(Boolean) as string[],
    usedTitles: past.map((g: any) => g.title).filter(Boolean) as string[],
    usedDurations: past.map((g: any) => g.duration).filter((d: any) => d !== null && d > 0) as number[],
  };
}

export function captureLyricItems(requested: LyricRequest[]): LyricBatchInput {
  if (requested.length < 1 || requested.length > 200) throw new Error('Batch must contain 1 to 200 items');
  const items: Item[] = [];
  for (const req of requested) {
    try {
    if (!['profile', 'generate', 'refine', 'fetch'].includes(req.type)) throw new Error('Unknown Lyric Studio operation');
    if (req.type === 'fetch') {
      items.push(fetchItem.parse({ type: 'fetch', artist: req.artist, album: req.album, maxSongs: req.maxSongs ?? 50 }));
      continue;
    }
    const options = base.parse({ provider: req.provider, model: req.model });
    const id = req.targetId;
    if (!Number.isInteger(id) || !id || id < 1) throw new Error('Valid source id required');
    let captured: Item;
    if (req.type === 'profile') {
      const source = db.getLyricsSet(id);
      if (!source) throw new Error(`Lyrics set ${id} not found`);
      captured = { ...options, type: 'profile', sourceId: id, sourceRevision: revision(source), artist: source.artist_name, songs: source.songs };
    } else if (req.type === 'generate') {
      const source = db.getProfile(id);
      if (!source) throw new Error(`Profile ${id} not found`);
      const set = db.getLyricsSet(source.lyrics_set_id);
      const profileData = structuredClone(source.profile_data);
      if (set) profileData.audio_enrichment = computeAlbumEnrichment(set.songs);
      captured = { ...options, type: 'generate', sourceId: id, sourceRevision: revision(source), lyricsSetId: set?.id, lyricsSetRevision: set ? revision(set) : undefined, profileData, artistId: set?.artist_id, extraInstructions: req.extraInstructions, userSubject: req.userSubject, noThink: req.noThink, history: captureHistory(set?.artist_id) };
    } else {
      const source = db.getGeneration(id);
      if (!source) throw new Error(`Generation ${id} not found`);
      const p = db.getProfile(source.profile_id);
      const set = p ? db.getLyricsSet(p.lyrics_set_id) : null;
      const artist = set ? db.getArtist(set.artist_id) : null;
      captured = { ...options, type: 'refine', sourceId: id, sourceRevision: revision(source), profileId: p?.id, profileRevision: p ? revision(p) : undefined, source, profileData: p?.profile_data, artist: artist?.name || 'Unknown' };
    }
    const count = req.count ?? 1;
    if (!Number.isInteger(count) || count < 1 || count > 50) throw new Error('Count must be between 1 and 50');
    for (let i = 0; i < count; i++) items.push(captured);
    } catch (error) {
      const count = Number.isInteger(req?.count) && req.count! > 0 && req.count! <= 50 ? req.count! : 1;
      for (let i = 0; i < count; i++) items.push({ type: 'preflight-error', error: error instanceof Error ? error.message : String(error) });
    }
  }
  return lyricBatchInput.parse({ items });
}

export async function captureRenderItems(intents: WrittenSongIntent[]): Promise<LyricBatchInput> {
  if (intents.length < 1 || intents.length > 200) throw new Error('Batch must contain 1 to 200 renders');
  const items: Item[] = [];
  for (const value of intents) {
    try {
      const parsed = writtenSongIntentSchema.parse(value);
      const intent = { ...parsed, engine: parsed.engine ?? getActiveBackendId() };
      const data = await loadWrittenSongData(intent, intent.engine);
      const resolved = resolveWrittenSongIntent(intent, intent.engine, data);
      items.push({ type: 'render', intent, sourceRevision: revision(resolved.request) });
    } catch (error) {
      items.push({ type: 'preflight-error', error: error instanceof Error ? error.message : String(error) });
    }
  }
  return lyricBatchInput.parse({ items });
}

async function queueRender(item: z.infer<typeof render>, ctx: WorkflowContext<LyricBatchInput>, index: number): Promise<AudioIntentItem> {
  const engine = item.intent.engine!;
  const data = await loadWrittenSongData(item.intent, engine);
  const resolved = resolveWrittenSongIntent(item.intent, engine, data);
  if (revision(resolved.request) !== item.sourceRevision) throw new Error('Written song or preset changed since submission');
  ctx.throwIfCancelled();
  return ctx.audio.enqueue(`render-${index}`, resolved.request);
}

async function execute(item: Item, ctx: WorkflowContext<LyricBatchInput>, index: number, prepared?: AudioIntentItem) {
  if (item.type === 'preflight-error') throw new Error(item.error);
  const prior = completedItem(ctx.jobId, index);
  if (prior) return prior;
  if (item.type === 'render') {
    const audio = prepared ?? await queueRender(item, ctx, index);
    const finished = await ctx.audio.wait(audio.id);
    ctx.throwIfCancelled();
    if (finished.status !== 'succeeded') throw new Error(finished.error || `Audio ${finished.status}`);
    if (!finished.jobId) throw new Error('Audio completed without a generation job id');
    return saveItem(ctx.jobId, index, () => {
      if (finished.jobId) {
        const existing = db.getAudioGenerations(item.intent.generationId, item.intent.lyricsSetId)
          .some((row: any) => row.hotstep_job_id === finished.jobId);
        if (!existing) db.linkAudioGeneration(item.intent.generationId, finished.jobId, item.intent.lyricsSetId);
        const url = finished.result?.audioUrl || (finished.result?.audioUrls as string[] | undefined)?.[0];
        if (typeof url === 'string' && url) db.resolveAudioGeneration(finished.jobId, url);
      }
      return { audioIntentId: finished.id, generationId: item.intent.generationId, jobId: finished.jobId };
    });
  }
  if (item.type === 'fetch') {
    const result = await genius.fetchLyrics(item.artist, item.album || null, item.maxSongs);
    ctx.throwIfCancelled();
    let image: string | null = null;
    let artistImage: string | null = null;
    try { artistImage = await genius.getArtistImageUrl(result.artist); } catch { /* optional art */ }
    if (result.album) {
      try { image = await genius.getAlbumImageUrl(result.album, result.artist); } catch { /* optional art */ }
    }
    ctx.throwIfCancelled();
    return saveItem(ctx.jobId, index, () => {
      const artist = db.getOrCreateArtist(result.artist);
      if (!artist.image_url && artistImage) db.updateArtistImage(artist.id as number, artistImage);
      const set = db.saveLyricsSet(artist.id as number, result.album, result.songs.length, result.songs, image);
      return { artist_id: artist.id, lyrics_set_id: set.id, songs_fetched: result.songs.length };
    });
  }
  const read = () => item.type === 'profile' ? db.getLyricsSet(item.sourceId) : item.type === 'generate' ? db.getProfile(item.sourceId) : db.getGeneration(item.sourceId);
  const sourceUnchanged = () => revision(read()) === item.sourceRevision
    && (item.type !== 'generate' || !item.lyricsSetId || revision(db.getLyricsSet(item.lyricsSetId)) === item.lyricsSetRevision)
    && (item.type !== 'refine' || !item.profileId || revision(db.getProfile(item.profileId)) === item.profileRevision);
  if (!sourceUnchanged()) throw new Error('Source changed since submission');
  const chunk = (text: string) => ctx.emit('chunk', { text });
  const phase = (text: string) => ctx.emit('phase', { phase: text });
  if (item.type === 'profile') {
    const data = await profiler.buildProfile(item.artist, null, item.songs, item.provider, item.model, phase, chunk);
    ctx.throwIfCancelled();
    if (!sourceUnchanged()) throw new Error('Source changed during profile build');
    return saveItem(ctx.jobId, index, () => {
      const saved = db.saveProfile(item.sourceId, item.provider, item.model || '', data);
      return { id: saved.id };
    });
  }
  if (item.type === 'generate') {
    const h = item.history;
    llm.resetSkipThinking();
    const result = await llm.generateLyricsStreaming(item.profileData as LyricsProfile, item.provider, item.model, item.extraInstructions, h.usedSubjects, h.usedBpms, h.usedKeys, h.usedTitles, h.usedDurations, chunk, phase, item.userSubject, item.noThink ? { noThink: true } : undefined);
    ctx.throwIfCancelled();
    if (!sourceUnchanged()) throw new Error('Source changed during generation');
    return saveItem(ctx.jobId, index, () => {
      const saved = db.saveGeneration({ profileId: item.sourceId, provider: item.provider, model: result.model, lyrics: result.lyrics, title: result.title, subject: result.subject, bpm: result.bpm || undefined, key: result.key, caption: result.caption, captionMm3: result.caption_mm3, captionYue2: result.caption_yue2, duration: result.duration || undefined, systemPrompt: result.system_prompt, userPrompt: result.user_prompt });
      return { id: saved.id, subject: saved.subject, bpm: saved.bpm, key: saved.key, title: saved.title, duration: saved.duration };
    });
  }
  const s = item.source;
  llm.resetSkipThinking();
  const result = await llm.refineLyricsStreaming(s.lyrics, item.artist, s.title, item.provider, item.model, item.profileData as LyricsProfile | undefined, chunk);
  ctx.throwIfCancelled();
  if (!sourceUnchanged()) throw new Error('Source changed during refinement');
  return saveItem(ctx.jobId, index, () => {
    const saved = db.saveGeneration({ profileId: s.profile_id, provider: item.provider, model: result.model, lyrics: result.lyrics, title: result.title, subject: s.subject || s.song_subject, bpm: s.bpm || undefined, key: s.key || s.song_key, caption: s.caption, captionMm3: s.caption_mm3, captionYue2: s.caption_yue2, duration: s.duration || undefined, systemPrompt: result.system_prompt, userPrompt: result.user_prompt, parentGenerationId: s.id });
    return { id: saved.id };
  });
}

export async function runLyricBatch(ctx: WorkflowContext<LyricBatchInput>) {
  const results: Array<{ index: number; status: 'done' | 'error'; value?: unknown; error?: string }> = [];
  // Queue all bulk renders before waiting so the audio queue can admit a
  // supported batch together. A failed preflight remains local to its item.
  const prepared = new Map<number, AudioIntentItem | Error>();
  if (ctx.input.items.every(item => item.type === 'render' || item.type === 'preflight-error')) {
    for (const [index, item] of ctx.input.items.entries()) {
      if (item.type !== 'render' || completedItem(ctx.jobId, index)) continue;
      try { prepared.set(index, await queueRender(item, ctx, index)); }
      catch (error) { ctx.throwIfCancelled(); prepared.set(index, error instanceof Error ? error : new Error(String(error))); }
    }
  }
  for (const [index, item] of ctx.input.items.entries()) {
    ctx.throwIfCancelled();
    ctx.emit('item-start', { index, type: item.type });
    try {
      const queued = prepared.get(index);
      if (queued instanceof Error) throw queued;
      const value = await execute(item, ctx, index, queued);
      ctx.throwIfCancelled();
      const result = { index, status: 'done' as const, value };
      results.push(result);
      ctx.emit('item-result', result);
      if (item.type === 'generate') {
        const saved = value as { subject?: string; bpm?: number; key?: string; title?: string; duration?: number };
        for (const later of ctx.input.items.slice(index + 1)) {
          if (later.type !== 'generate' || later.artistId !== item.artistId) continue;
          if (saved.subject) later.history.usedSubjects.push(saved.subject);
          if (saved.bpm) later.history.usedBpms.push(saved.bpm);
          if (saved.key) later.history.usedKeys.push(saved.key);
          if (saved.title) later.history.usedTitles.push(saved.title);
          if (saved.duration) later.history.usedDurations.push(saved.duration);
        }
      }
    } catch (error) {
      ctx.throwIfCancelled();
      const result = { index, status: 'error' as const, error: error instanceof Error ? error.message : String(error) };
      results.push(result);
      ctx.emit('item-error', result);
    }
  }
  return { results };
}

registerWorkflowKind({ kind: 'lyric-batch', input: lyricBatchInput, run: runLyricBatch, maxConcurrent: 1 });
