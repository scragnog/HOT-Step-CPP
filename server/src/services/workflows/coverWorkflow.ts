import { z } from 'zod/v4';
import { parseFile } from 'music-metadata';
import { config } from '../../config.js';
import { getDb } from '../../db/database.js';
import { resolveAudioAsset } from '../assets/audioAssets.js';
import { getActiveBackendId, getBackend } from '../backends/registry.js';
import { resolveYue2Caption } from '../generation/resolve/captionSource.js';
import type { Yue2SourceTrack } from '../../contracts/resolution.js';
import { analyzeWithEssentia } from '../training/essentiaClient.js';
import { yue2CoverService } from '../yue2Cover.js';
import * as lireekDb from '../../db/lireekDb.js';
import { getProvider } from '../lireek/llm/registry.js';
import { STYLE_CAPTION_PROMPT } from '../lireek/prompts.js';
import { registerWorkflowKind, workflowDocuments } from '../../routes/workflows.js';
import { WorkflowError, type WorkflowKind } from './workflowJobs.js';
import type { WorkflowDocuments } from './revisions.js';

const ref = z.object({ documentId: z.string().uuid(), revision: z.number().int().min(1) });
const open = z.object({ assetId: z.string().uuid(), cached: z.object({
  metadata: z.object({ artist: z.string(), title: z.string(), album: z.string(),
    duration: z.number().finite().nullable() }),
  analysis: z.object({ bpm: z.number().finite(), key: z.string(), scale: z.string().optional() }),
}).optional() });
const transcribe = ref.extend({ force: z.boolean().default(false) });
const captionInput = ref.extend({
  artistId: z.number().int().positive(), provider: z.string(), model: z.string(), force: z.boolean(),
});
const render = ref.extend({
  expectedBackend: z.enum(['ace', 'yue2']), engineParams: z.record(z.string(), z.unknown()),
  title: z.string(), artistName: z.string(), targetArtistName: z.string(),
  lyrics: z.string(), caption: z.string(),
  instrumental: z.boolean(), lyricsSource: z.string().nullable(),
  scoreSource: z.enum(['dataset']).nullable(),
  analysis: z.object({ bpm: z.number().finite(), key: z.string(), scale: z.string().optional() }).nullable().optional(),
  settings: z.record(z.string(), z.unknown()),
  controls: z.object({
    bpmOverride: z.number().finite().positive().nullable(), bpmCorrection: z.number().finite().positive(),
    keyOverride: z.string().nullable(), tempoScale: z.number().finite().positive(),
    pitchShift: z.number().int().min(-12).max(12), noFsq: z.boolean(),
    audioCoverStrength: z.number().finite(), coverNoiseStrength: z.number().finite(),
    coverNoiseMethod: z.string(), vocalLanguage: z.string(), sourceLatentUrl: z.string(),
    timbreOverridePath: z.string(), presetAdapterPath: z.string(), presetReferencePath: z.string(),
    triggerUseFilename: z.boolean(), triggerPlacement: z.string(),
    voices: z.enum(['vocal', 'both']), keepChords: z.boolean(),
    tempoMode: z.enum(['free', 'source', 'set']), coverBpm: z.number().finite().positive(),
    keyShift: z.number().int().min(-6).max(6), cfgScale: z.number().finite(),
    lmAdapterAr: z.string(), lmAdapterNar: z.string(), pairMode: z.enum(['base', 'pair']),
    yue2Pick: z.record(z.string(), z.union([z.string(), z.number()])),
    captionMode: z.string(), captionTracks: z.array(z.object({
      name: z.string(), styled: z.string().optional(), caption: z.string().optional(),
      bpm: z.union([z.string(), z.number()]).optional(),
    })),
  }),
});
export type CoverRenderInput = z.infer<typeof render>;
export interface CoverDraft { assetId: string; sha256: string; sourceLabel?: string;
  analysis?: { bpm: number; key: string; scale?: string } | null;
  abc?: string; approvedAbc?: string; scoreSource?: string }

const NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const INDEX: Record<string, number> = { C: 0, 'C#': 1, Db: 1, D: 2, 'D#': 3, Eb: 3,
  E: 4, Fb: 4, F: 5, 'F#': 6, Gb: 6, G: 7, 'G#': 8, Ab: 8, A: 9, 'A#': 10, Bb: 10, B: 11, Cb: 11 };
export function transposeCoverKey(key: string, shift: number): string {
  if (!key || !shift) return key;
  const [note, ...quality] = key.trim().split(/\s+/);
  const index = INDEX[note];
  return index === undefined ? key : [NOTES[((index + shift) % 12 + 12) % 12], ...quality].join(' ');
}
function scoreKey(abc: string): string {
  const m = abc.match(/^K:\s*([A-Ga-g][#b]?)(m)?(?:\s+(minor|major|maj))?(?=\s|$)/m);
  return m ? `${m[1]} ${m[2] || m[3]?.toLowerCase() === 'minor' ? 'minor' : 'major'}` : '';
}
export function coverScoreDetails(abc: string, factor: number): { bpm: number; key: string } {
  const raw = Number(abc.match(/^Q:[^=\r\n]*=\s*(\d+(?:\.\d+)?)/m)?.[1]) || 0;
  const key = scoreKey(abc).replace(/(major|minor)$/, value => value[0].toUpperCase() + value.slice(1));
  if (!Number.isFinite(factor) || factor <= 0 || !raw || !key) {
    throw new WorkflowError(400, 'Score needs a readable tempo and key');
  }
  return { bpm: Math.round(raw * factor), key };
}
function coverTitle(i: CoverRenderInput) {
  return i.artistName ? `${i.title || 'Cover'} (${i.artistName} Cover)` : (i.title || 'Cover');
}
export function effectiveCoverRequest(i: CoverRenderInput, d: CoverDraft, url: string): Record<string, unknown> {
  const c = i.controls, e = i.engineParams, s = i.settings;
  const analysis = i.analysis ?? d.analysis;
  if (i.expectedBackend === 'ace') {
    const bpm = c.bpmOverride ?? ((analysis?.bpm || 120) * c.bpmCorrection);
    const key = c.keyOverride || analysis?.key || 'C major';
    const request: Record<string, unknown> = {
      ...e, customMode: true, lyrics: i.instrumental ? '[Instrumental]' : i.lyrics,
      style: i.caption || e.style || '', title: coverTitle(i),
      taskType: c.noFsq ? 'cover-nofsq' : 'cover', sourceAudioUrl: url,
      audioCoverStrength: c.audioCoverStrength, coverNoiseStrength: c.coverNoiseStrength,
      bpm: Math.round(bpm * c.tempoScale), keyScale: transposeCoverKey(key, c.pitchShift),
      duration: 0, instrumental: i.instrumental, vocalLanguage: c.vocalLanguage,
      source: 'cover-studio', artistName: i.targetArtistName || i.artistName, sourceArtist: i.artistName,
      expectedBackend: i.expectedBackend,
    };
    if (c.coverNoiseMethod) request.coverNoiseMethod = c.coverNoiseMethod;
    if (c.tempoScale !== 1) request.tempoScale = c.tempoScale;
    if (c.pitchShift) request.pitchShift = c.pitchShift;
    if (i.lyricsSource) request.lyricsSource = i.lyricsSource;
    if (c.sourceLatentUrl) request.sourceLatentUrl = c.sourceLatentUrl;
    if (c.presetAdapterPath) {
      request.loraPath = c.presetAdapterPath;
      if (c.triggerUseFilename) {
        const trigger = c.presetAdapterPath.split(/[\\/]/).pop()?.replace(/\.safetensors$/i, '');
        if (trigger) { request.triggerWord = trigger; request.triggerPlacement = c.triggerPlacement || 'prepend'; }
      }
    }
    if (c.presetReferencePath) {
      request.referenceAudioUrl = c.presetReferencePath;
      request.masteringEnabled = true;
      request.masteringReference = c.presetReferencePath;
      request.timbreReference = true;
    }
    if (c.timbreOverridePath) request.timbreReference = c.timbreOverridePath;
    else if (typeof e.timbreReference === 'string' && e.timbreReference) request.timbreReference = e.timbreReference;
    return request;
  }
  if (!d.approvedAbc) throw new WorkflowError(409, 'Review and approve the score first');
  if (c.pairMode === 'pair' && (!c.lmAdapterAr || !c.lmAdapterNar)) throw new WorkflowError(400, 'Choose both YuE2 adapter halves');
  const selection = c.captionMode === 'auto' ? { mode: 'auto' as const }
    : c.captionMode.startsWith('track:') ? { mode: 'track' as const, selectedName: c.captionMode.slice(6) }
      : { mode: 'custom' as const };
  const caption = resolveYue2Caption(i.caption, analysis?.bpm, c.captionTracks as Yue2SourceTrack[], selection).caption;
  const yue2 = Object.fromEntries(Object.entries(e).filter(([key]) => key.startsWith('yue2')));
  const keys = ['seed', 'randomSeed', 'batchSize', 'postProcessingEnabled', 'masteringEnabled',
    'masteringReference', 'lufsEnabled', 'lufsTarget', 'lufsCeilingDb', 'stableStepOn',
    'stableStepStrength', 'stableStepBackend', 'stableStepAdapters', 'stableStepSeed',
    'stableStepSeedFollowsDit', 'stableStepSteps', 'stableStepSolver', 'stableStepScheduler',
    'stableStepGuidanceMode', 'stableStepGuidanceScale', 'stableStepPluginParams',
    'whisperLyricsEnabled', 'whisperModel', 'whisperLanguage', 'whisperBeamSize',
    'whisperIsolateVocals', 'qualityEvalEnabled', 'qualityEvalTarget', 'coverArtEnabled', 'coverArtSubject'];
  const shared = Object.fromEntries(keys.map(k => [k, e[k]]));
  const moved = c.keyShift && scoreKey(d.approvedAbc) ? transposeCoverKey(scoreKey(d.approvedAbc), c.keyShift) : '';
  return { ...shared, ...yue2, backend: 'yue2', duration: -1, expectedBackend: 'yue2',
    coResident: s.coResident, cacheLmCodes: s.cacheLmCodes,
    parallelWhisper: s.parallelWhisper, parallelQualityEval: s.parallelQualityEval,
    parallelCoverArt: s.parallelCoverArt,
    yue2CfgScale: c.cfgScale, customMode: true, taskType: 'text2music',
    title: coverTitle(i), caption, style: caption, lyrics: i.instrumental ? '' : i.lyrics,
    ...(i.lyricsSource ? { lyricsSource: i.lyricsSource } : {}),
    ...(i.scoreSource === 'dataset' || d.scoreSource === 'dataset' ? { scoreSource: 'dataset-sidecar' } : {}),
    instrumental: i.instrumental, source: 'cover-studio', sourceAudioUrl: url,
    yue2Cover: { sourceId: url, sourceLabel: d.sourceLabel || url,
      voices: c.voices, keepChords: c.keepChords,
      tempo: c.tempoMode === 'set' ? c.coverBpm : c.tempoMode,
      key: moved ? moved.replace(/ major$/, '').replace(/ minor$/, 'm') : 'source', cfgScale: c.cfgScale },
    yue2Abc: d.approvedAbc, yue2Cot: c.keepChords ? 'full' : 'melody',
    yue2Pick: { ...c.yue2Pick, lmAdapterAr: c.pairMode === 'pair' ? c.lmAdapterAr : '',
      lmAdapterNar: c.pairMode === 'pair' ? c.lmAdapterNar : '' } };
}

export interface CoverDeps {
  documents?: WorkflowDocuments;
  asset?: (id: string, user: string) => { url: string; path: string; filename: string; sha256: string };
  analyze?: (path: string, signal: AbortSignal) => Promise<{ bpm: number | null; key: string; scale: string } | null>;
  metadata?: (path: string) => Promise<{ artist: string; title: string; album: string; duration: number | null }>;
  transcribe?: (url: string, label: string, user: string, signal: AbortSignal, force: boolean) => Promise<{ abc: string; scoreSource?: string }>;
  capability?: (backend: string) => Promise<boolean>;
  caption?: (artistId: number, provider: string, model: string, force: boolean, signal: AbortSignal) => Promise<{ text: string; generatedProfileId?: number }>;
}
const defaultAsset = (id: string, user: string) => resolveAudioAsset(getDb(), id, user, config.data.dir);
async function defaultMetadata(path: string) {
  try {
    const m = await parseFile(path);
    return { artist: m.common.artist || '', title: m.common.title || '', album: m.common.album || '', duration: m.format.duration ?? null };
  } catch { return { artist: '', title: '', album: '', duration: null }; }
}
async function defaultTranscribe(url: string, label: string, user: string, signal: AbortSignal, force: boolean) {
  const started = await yue2CoverService.start({ sourceAudioUrl: url, sourceLabel: label.slice(0, 120), force }, user);
  if (signal.aborted) {
    if (started.jobId) yue2CoverService.cancel(started.jobId, user);
    throw new Error('Cancelled');
  }
  if (started.abc) return { abc: started.abc, scoreSource: started.scoreSource };
  const id = started.jobId;
  if (!id) throw new Error('Transcription returned no score or job');
  const cancel = () => { try { yue2CoverService.cancel(id, user); } catch { /* already final */ } };
  signal.addEventListener('abort', cancel, { once: true });
  try {
    for (;;) {
      if (signal.aborted) throw new Error('Cancelled');
      const result = yue2CoverService.find(id, user);
      if (result.abc) return { abc: result.abc };
      if (result.job?.status === 'done') throw new Error('Transcription finished without a score');
      if (result.job?.status === 'failed' || result.job?.status === 'cancelled') throw new Error(result.job.error || `Transcription ${result.job.status}`);
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  } finally { signal.removeEventListener('abort', cancel); }
}
async function defaultCapability(id: string) {
  if (getActiveBackendId() !== id) throw new WorkflowError(409, 'Active backend changed; start again');
  const backend = getBackend(id);
  if (!backend) throw new WorkflowError(400, 'Unknown backend');
  const cap = await backend.capabilities();
  if (getActiveBackendId() !== id) throw new WorkflowError(409, 'Active backend changed; start again');
  return cap.features.cover;
}
async function defaultCaption(artistId: number, provider: string, model: string, force: boolean, signal: AbortSignal) {
  const artist = lireekDb.getArtist(artistId);
  if (!artist) throw new WorkflowError(404, 'Artist not found');
  const sets = lireekDb.getLyricsSets(artistId);
  if (!force) {
    for (const set of sets) {
      const found = lireekDb.getGenerations(undefined, set.id).find(g => g.caption?.trim());
      if (found) return { text: String(found.caption) };
    }
    for (const set of sets) {
      const found = lireekDb.getProfiles(set.id).find(p => p.profile_data?.style_caption);
      if (found) return { text: String(found.profile_data.style_caption) };
    }
  }
  if (!provider) return { text: '' };
  let profile: Record<string, any> | undefined;
  for (const set of sets) {
    profile = lireekDb.getProfiles(set.id)[0];
    if (profile) break;
  }
  if (!profile) throw new WorkflowError(404, 'No profile found for this artist');
  const data = profile.profile_data || {};
  const prompt = [`Artist: ${artist.name}`, data.album ? `Album: ${data.album}` : '',
    data.themes?.length ? `Themes: ${data.themes.join(', ')}` : '',
    data.tone_and_mood ? `Tone and mood: ${data.tone_and_mood}` : '',
    data.vocabulary_notes ? `Vocabulary: ${data.vocabulary_notes}` : ''].filter(Boolean).join('\n');
  const llm = getProvider(provider);
  const raw = await llm.call(STYLE_CAPTION_PROMPT, prompt, model || llm.defaultModel);
  if (signal.aborted) throw new Error('Cancelled');
  return { text: raw.replace(/^["'`]+|["'`]+$/g, '').trim(), generatedProfileId: profile.id };
}
function draft(docs: WorkflowDocuments, id: string, revision: number, user: string): CoverDraft {
  const doc = docs.get(id, user);
  if (doc.kind !== 'cover-draft') throw new WorkflowError(400, 'Document is not a cover draft');
  if (doc.revision !== revision) throw new WorkflowError(409, `Stale cover draft revision ${revision}`);
  const data = doc.data as unknown as CoverDraft;
  if (!data.assetId) throw new WorkflowError(400, 'Cover draft has no asset id');
  return data;
}
export function createCoverKinds(deps: CoverDeps = {}): WorkflowKind<any>[] {
  const docs = () => deps.documents || workflowDocuments();
  const asset = deps.asset || defaultAsset;
  return [{ kind: 'cover-open', input: open, async run(ctx) {
    const input = ctx.input as z.infer<typeof open>;
    const { assetId } = input;
    const source = asset(assetId, ctx.userId);
    ctx.emit('stage', { stage: 'Analyzing source' });
    const [metadata, result] = input.cached ? [input.cached.metadata, null] : await Promise.all([
      (deps.metadata || defaultMetadata)(source.path),
      (deps.analyze || analyzeWithEssentia)(source.path, ctx.signal),
    ]);
    ctx.throwIfCancelled();
    const analysis = input.cached?.analysis || (result
      ? { bpm: result.bpm || 120, key: `${result.key || 'C'} ${result.scale || 'major'}`, scale: result.scale }
      : null);
    const document = docs().create(ctx.userId, 'cover-draft', {
      assetId, sha256: source.sha256, sourceLabel: source.filename, metadata, analysis,
      abc: '', approvedAbc: '',
    });
    return { documentId: document.id, revision: document.revision, metadata, analysis };
  } }, { kind: 'cover-caption', input: captionInput, async run(ctx) {
    const input = ctx.input as z.infer<typeof captionInput>;
    const data = draft(docs(), input.documentId, input.revision, ctx.userId);
    const source = asset(data.assetId, ctx.userId);
    if (source.sha256 !== data.sha256) throw new WorkflowError(409, 'Cover source changed');
    ctx.emit('stage', { stage: 'Resolving cover caption' });
    const result = await (deps.caption || defaultCaption)(input.artistId, input.provider, input.model, input.force, ctx.signal);
    ctx.throwIfCancelled();
    const updated = docs().update(input.documentId, ctx.userId, input.revision, current => {
      if (current.assetId !== data.assetId) throw new WorkflowError(409, 'Cover source changed');
      return { ...current, caption: result.text, artistId: input.artistId };
    });
    if (result.generatedProfileId) {
      try {
        const profile = lireekDb.getProfile(result.generatedProfileId);
        if (profile) lireekDb.updateProfileData(result.generatedProfileId,
          { ...profile.profile_data, style_caption: result.text });
      } catch (err) { console.warn('[CoverWorkflow] Profile caption cache failed:', err); }
    }
    return { documentId: updated.id, revision: updated.revision, caption: result.text };
  } }, { kind: 'cover-transcribe', input: transcribe, async run(ctx) {
    const input = ctx.input as z.infer<typeof transcribe>;
    const data = draft(docs(), input.documentId, input.revision, ctx.userId);
    const source = asset(data.assetId, ctx.userId);
    if (source.sha256 !== data.sha256) throw new WorkflowError(409, 'Cover source changed');
    ctx.emit('stage', { stage: 'Transcribing score' });
    const result = await (deps.transcribe || defaultTranscribe)(source.url, data.sourceLabel || source.filename,
      ctx.userId, ctx.signal, input.force);
    ctx.throwIfCancelled();
    const updated = docs().update(input.documentId, ctx.userId, input.revision, current => {
      if (current.assetId !== data.assetId) throw new WorkflowError(409, 'Cover source changed');
      return { ...current, abc: result.abc, approvedAbc: '', scoreSource: result.scoreSource || '' };
    });
    return { documentId: updated.id, revision: updated.revision, abc: result.abc, scoreSource: result.scoreSource };
  } }, { kind: 'cover-render', input: render, async run(ctx) {
    const input = ctx.input as CoverRenderInput;
    if (!input.instrumental && !input.lyrics.trim()) throw new WorkflowError(400, 'Enter lyrics or enable Instrumental mode');
    const data = draft(docs(), input.documentId, input.revision, ctx.userId);
    const source = asset(data.assetId, ctx.userId);
    if (source.sha256 !== data.sha256) throw new WorkflowError(409, 'Cover source changed');
    if (!await (deps.capability || defaultCapability)(input.expectedBackend)) throw new WorkflowError(400, 'Backend does not support covers');
    const request = effectiveCoverRequest(input, data, source.url);
    ctx.throwIfCancelled();
    const item = ctx.audio.enqueue('render', request, { source: 'cover-studio', workflowJobId: ctx.jobId });
    const finished = await ctx.audio.wait(item.id);
    ctx.throwIfCancelled();
    if (finished.status !== 'succeeded') throw new Error(finished.error || `Audio ${finished.status}`);
    return { request, audioIntentId: item.id, audio: finished.result };
  } }];
}
export function registerCoverWorkflows(deps: CoverDeps = {}) {
  for (const kind of createCoverKinds(deps)) registerWorkflowKind(kind);
}
