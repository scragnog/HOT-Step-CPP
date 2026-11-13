import { z } from 'zod/v4';
import fs from 'node:fs';
import path from 'node:path';
import { parseFile } from 'music-metadata';
import { config } from '../../config.js';
import { getDb } from '../../db/database.js';
import { resolveAudioAsset } from '../assets/audioAssets.js';
import { getActiveBackendId, getBackend } from '../backends/registry.js';
import { resolveYue2Caption } from '../generation/resolve/captionSource.js';
import type { Yue2SourceTrack } from '../../contracts/resolution.js';
import { analyzeWithEssentia } from '../training/essentiaClient.js';
import { readCoverDatasetAnalysis } from '../coverDatasetSource.js';
import { yue2CoverService } from '../yue2Cover.js';
import * as lireekDb from '../../db/lireekDb.js';
import { getProvider } from '../lireek/llm/registry.js';
import { STYLE_CAPTION_PROMPT } from '../lireek/prompts.js';
import { registerWorkflowKind, workflowDocuments } from '../../routes/workflows.js';
import { WorkflowError, type WorkflowKind } from './workflowJobs.js';
import type { WorkflowDocuments } from './revisions.js';
import {
  coverDraftRefSchema as ref, coverOpenSchema as open, coverTranscribeSchema as transcribe,
  coverCaptionSchema as captionInput, coverRenderSchema as render,
  coverSourceSchema, type CoverLeadSheetSource, type CoverLeadSheetResult,
  type CoverRenderInput, type CoverDraft,
} from '../../contracts/studioWorkflows.js';
import { instrumentalConflict, preparedPair } from '../../contracts/yue2Prepare.js';
import { tryAbcCoverLeadSheet, coverRenderRolls, clearCoverScore } from '../coverLeadSheet.js';
import { sheetTranscriberSetting } from '../training/yue2Sheet.js';
import { heartmulaCoverLyrics, heartmulaCoverTags } from '../backends/heartmula/cover.js';
import { mm3PlankDir, mm3PlankSiblings, readMm3PlankMeta } from '../backends/minimax/plank.js';

export type { CoverRenderInput, CoverDraft };

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
/** The plank sidecar's launder record as generation params. A plank without a
 *  sidecar, or one captured from a render, reports nothing. */
export function mm3PlankProvenance(plank: string): Record<string, unknown> {
  const meta = readMm3PlankMeta(plank);
  if (typeof meta?.laundered !== 'boolean') return {};
  return { mm3PlankLaundered: meta.laundered,
    ...(meta.laundered ? {
      mm3PlankLaunderModels: [meta.rec7, meta.encoder].filter(Boolean).join(', '),
      ...(meta.launderSteps !== undefined ? { mm3PlankLaunderSteps: meta.launderSteps } : {}),
      ...(meta.launderSeed !== undefined ? { mm3PlankLaunderSeed: meta.launderSeed } : {}),
      ...(meta.launderCaption !== undefined ? { mm3PlankLaunderCaption: meta.launderCaption } : {}),
    } : {}) };
}

const MM3_STATE_FIX = { off: 0, depth: 1, stream: 2, both: 3 } as const;

/** The guided cover modes (round 9 of the cover probe): the planner picks from
 *  the plank's guide instead of being pinned to it. Null for Pin. Refuses with
 *  "Re-encode" when the plank predates the guide or the launder state, rather
 *  than falling back to Pin. */
export function mm3CoverModeParams(m: NonNullable<CoverRenderInput['mm3Cover']>): Record<string, unknown> | null {
  const mode = m.mode ?? 'pin';
  if (mode === 'pin') return null;
  if (!m.plank) throw new WorkflowError(400, 'Encode the source first');
  const sib = mm3PlankSiblings(m.plank);
  const has = (file: string) => fs.existsSync(path.join(mm3PlankDir(), file));
  if (!has(sib.guide) || (mode === 'blend' && !has(sib.state))) {
    throw new WorkflowError(409, 'This source was encoded before cover modes. Re-encode it to use this mode.');
  }
  const fix = mode === 'follow-depth' ? 'depth' : mode === 'follow-stream' ? 'stream' : 'off';
  return {
    mm3CoverMode: mode, mm3CoverPlank: m.plank,
    mm3GuidePath: sib.guide, mm3GuideScale: m.strength ?? 0.9, mm3GuideK: m.candidates ?? 32, mm3GuideOutside: -100,
    mm3GuideSeconds: m.trimSeconds ?? '',
    ...(mode === 'product' ? { mm3GuideProduct: m.weight ?? 1 } : { mm3GuideStateFix: MM3_STATE_FIX[m.stateFix ?? fix] }),
    ...(mode === 'blend' ? { mm3BlendPath: sib.state, mm3BlendAmount: (m.blend ?? 25) / 100 } : {}),
  };
}

export function effectiveCoverRequest(i: CoverRenderInput, d: CoverDraft, url: string): Record<string, unknown> {
  const c = i.controls, e = i.engineParams, s = i.settings;
  const analysis = i.analysis ?? d.analysis;
  const scoreSource = d.scoreSource || i.scoreSource;
  const scoreMetadata = scoreSource ? { scoreSource: scoreSource === 'dataset' ? 'dataset-sidecar' : scoreSource } : {};
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
  const keys = ['seed', 'randomSeed', 'batchSize', 'postProcessingEnabled', 'masteringEnabled',
    'masteringReference', 'lufsEnabled', 'lufsTarget', 'lufsCeilingDb', 'stableStepOn',
    'stableStepStrength', 'stableStepBackend', 'stableStepAdapters', 'stableStepSeed',
    'stableStepSeedFollowsDit', 'stableStepSteps', 'stableStepSolver', 'stableStepScheduler',
    'stableStepGuidanceMode', 'stableStepGuidanceScale', 'stableStepPluginParams',
    'whisperLyricsEnabled', 'whisperModel', 'whisperLanguage', 'whisperBeamSize',
    'whisperIsolateVocals', 'qualityEvalEnabled', 'qualityEvalTarget', 'coverArtEnabled', 'coverArtSubject'];
  const shared = Object.fromEntries(keys.map(k => [k, e[k]]));
  if (i.expectedBackend === 'sa3') {
    return { ...shared, backend: 'sa3', expectedBackend: 'sa3', taskType: 'cover',
      customMode: true, sourceAudioUrl: url, style: i.caption, title: coverTitle(i),
      instrumental: true, lyrics: '', duration: 0, source: 'cover-studio',
      artistName: i.targetArtistName || i.artistName, sourceArtist: i.artistName,
      caption: i.caption, sa3Steps: i.steps ?? 8,
      ...(i.seed !== undefined ? { seed: i.seed, randomSeed: i.seed < 0 } : {}),
      sa3Cover: { transform: i.strength ?? c.sa3Transform ?? 0.3,
        adapters: i.adapter ?? c.sa3Adapters ?? '' } };
  }
  if (i.expectedBackend === 'minimax-m3') {
    const m = i.mm3Cover;
    if (!m || (!m.plank && !m.soundStrength)) throw new WorkflowError(400, 'Turn on Plan (encode or pick a reference) or Sound');
    // The global bar's MM3 settings (LM knobs, adapter, flow) ride along; the
    // plank or the source pins the length, so takes, endings and saved plans stand down.
    const mm3 = Object.fromEntries(Object.entries(e).filter(([key]) => key.startsWith('mm3')));
    const guided = mm3CoverModeParams(m);
    return { ...shared, ...mm3, backend: 'minimax-m3', expectedBackend: 'minimax-m3', taskType: 'text2music',
      customMode: true, caption: i.caption, style: i.caption, title: coverTitle(i),
      lyrics: i.instrumental ? '' : i.lyrics, instrumental: i.instrumental, batchSize: 1,
      source: 'cover-studio', sourceAudioUrl: url,
      artistName: i.targetArtistName || i.artistName, sourceArtist: i.artistName,
      ...(i.seed !== undefined ? { seed: i.seed, randomSeed: i.seed < 0 } : {}),
      mm3PlankPath: guided ? '' : m.plank ?? '', mm3PlankSemanticOnly: m.semanticOnly,
      ...guided,
      // Provenance for the song facts: a laundered plank sounds different, and
      // A/Bs need to know which kind they rendered from.
      ...(m.plank ? mm3PlankProvenance(m.plank) : {}),
      // Sound: the source's latents in the flow. Trim covers both when both are on.
      mm3A2aStrength: m.soundStrength ?? '', mm3A2aHold: m.soundStrength ? m.soundHold ?? 0 : '',
      mm3A2aSourceSeconds: m.soundStrength && m.plank ? m.trimSeconds ?? '' : '',
      mm3ForceBias: m.forceBias ?? '', mm3ForcePattern: m.pattern ?? undefined,
      mm3PlankTrimSeconds: m.trimSeconds ?? undefined, mm3ArSeed: m.arSeed >= 0 ? m.arSeed : '',
      mm3Takes: 1, mm3HiddensPath: '', mm3SaveHiddens: false };
  }
  if (i.expectedBackend === 'heartmula') {
    if (!d.approvedAbc) throw new WorkflowError(409, 'Review and approve the score first');
    try {
      const { rolls, path } = coverRenderRolls(d);
      const tags = heartmulaCoverTags(i.heartmulaTags);
      return { ...shared, ...Object.fromEntries(Object.entries(e).filter(([key]) => key.startsWith('heartmula'))),
        backend: 'heartmula', expectedBackend: 'heartmula', taskType: 'cover', customMode: true,
        heartmulaCover: rolls, leadSheetRenderingPath: path,
        ...scoreMetadata, heartmulaTags: tags, caption: tags, style: tags,
        lyrics: heartmulaCoverLyrics(i.instrumental ? '' : i.lyrics), instrumental: i.instrumental,
        duration: -1, batchSize: 1, title: coverTitle(i), sourceAudioUrl: url, source: 'cover-studio',
        artistName: i.targetArtistName || i.artistName, sourceArtist: i.artistName,
        ...(i.seed !== undefined ? { seed: i.seed, randomSeed: i.seed < 0 } : {}),
        heartmulaTopK: e.heartmulaTopK ?? 250 };
    } catch (error) { throw new WorkflowError(400, error instanceof Error ? error.message : String(error)); }
  }
  if (i.expectedBackend !== 'yue2') throw new WorkflowError(400, 'Backend does not support covers');
  if (!d.approvedAbc) throw new WorkflowError(409, 'Review and approve the score first');
  if (c.pairMode === 'pair' && (!c.lmAdapterAr || !c.lmAdapterNar)) throw new WorkflowError(400, 'Choose both YuE2 adapter halves');
  const selection = c.captionMode === 'auto' ? { mode: 'auto' as const }
    : c.captionMode.startsWith('track:') ? { mode: 'track' as const, selectedName: c.captionMode.slice(6) }
      : { mode: 'custom' as const };
  const caption = resolveYue2Caption(i.caption, analysis?.bpm, c.captionTracks as Yue2SourceTrack[], selection).caption;
  const yue2 = Object.fromEntries(Object.entries(e).filter(([key]) => key.startsWith('yue2')));
  const moved = c.keyShift && scoreKey(d.approvedAbc) ? transposeCoverKey(scoreKey(d.approvedAbc), c.keyShift) : '';
  return { ...shared, ...yue2, backend: 'yue2', duration: -1, expectedBackend: 'yue2',
    coResident: s.coResident, cacheLmCodes: s.cacheLmCodes,
    parallelWhisper: s.parallelWhisper, parallelQualityEval: s.parallelQualityEval,
    parallelCoverArt: s.parallelCoverArt,
    yue2CfgScale: c.cfgScale, yue2LyricSchedule: c.lyricSchedule, customMode: true, taskType: 'text2music',
    title: coverTitle(i), caption, style: caption, lyrics: i.instrumental ? '' : i.lyrics,
    ...(i.lyricsSource ? { lyricsSource: i.lyricsSource } : {}),
    ...scoreMetadata,
    leadSheetRenderingPath: d.leadSheet && d.leadSheet.abc !== d.approvedAbc ? 'abc-edited' : 'abc',
    instrumental: i.instrumental, source: 'cover-studio', sourceAudioUrl: url,
    // The covering band names the saved file and tags it, as on the ACE path.
    artistName: i.targetArtistName || i.artistName, sourceArtist: i.artistName,
    yue2Cover: { sourceId: url, sourceLabel: d.sourceLabel || url,
      voices: c.voices, keepChords: c.keepChords,
      tempo: c.tempoMode === 'set' ? c.coverBpm : c.tempoMode,
      key: moved ? moved.replace(/ major$/, '').replace(/ minor$/, 'm') : 'source',
      ...(moved ? { keyShift: c.keyShift } : {}), cfgScale: c.cfgScale },
    yue2Abc: d.approvedAbc, yue2Cot: c.keepChords ? 'full' : 'melody',
    yue2Pick: { ...c.yue2Pick, lmAdapterAr: c.pairMode === 'pair' ? c.lmAdapterAr : '',
      lmAdapterNar: c.pairMode === 'pair' ? c.lmAdapterNar : '' } };
}

export interface CoverDeps {
  documents?: WorkflowDocuments;
  asset?: (id: string, user: string) => { url: string; path: string; filename: string; sha256: string };
  analyze?: (path: string, signal: AbortSignal) => Promise<{ bpm: number | null; key: string; scale: string } | null>;
  datasetAnalysis?: (path: string) => { bpm: number; key: string; scale: string } | null;
  metadata?: (path: string) => Promise<{ artist: string; title: string; album: string; duration: number | null }>;
  transcribe?: (url: string, label: string, user: string, signal: AbortSignal, force: boolean, source?: CoverLeadSheetSource) => Promise<{ abc: string; scoreSource?: string; leadSheet?: CoverLeadSheetResult }>;
  leadSheetDefault?: () => CoverLeadSheetSource;
  capability?: (backend: string) => Promise<boolean>;
  caption?: (artistId: number, provider: string, model: string, force: boolean, signal: AbortSignal) => Promise<{ text: string; generatedProfileId?: number }>;
}
const defaultLeadSheetSource = (): CoverLeadSheetSource => sheetTranscriberSetting().refined ? 'sheetsage2-refined' : 'sheetsage2';
const defaultAsset = (id: string, user: string) => resolveAudioAsset(getDb(), id, user, config.data.dir);
async function defaultMetadata(path: string) {
  try {
    const m = await parseFile(path);
    return { artist: m.common.artist || '', title: m.common.title || '', album: m.common.album || '', duration: m.format.duration ?? null };
  } catch { return { artist: '', title: '', album: '', duration: null }; }
}
async function defaultTranscribe(url: string, label: string, user: string, signal: AbortSignal, force: boolean, source: CoverLeadSheetSource = 'sheetsage2'): Promise<{ abc: string; scoreSource?: string; leadSheet?: CoverLeadSheetResult }> {
  if (source === 'yourmt3-chordnet') throw new WorkflowError(400, yue2CoverService.readiness(source).message!);
  if (source === 'muscriptor') {
    const { transcribeCoverMidi } = await import('../../routes/midiStudio.js');
    return transcribeCoverMidi(url, signal);
  }
  const started = await yue2CoverService.start({ sourceAudioUrl: url, sourceLabel: label.slice(0, 120), force, transcriber: source }, user);
  if (signal.aborted) {
    if (started.jobId) yue2CoverService.cancel(started.jobId, user);
    throw new Error('Cancelled');
  }
  if (started.abc) return { abc: started.abc, scoreSource: started.scoreSource || 'sheetsage2' };
  const id = started.jobId;
  if (!id) throw new Error('Transcription returned no score or job');
  const cancel = () => { try { yue2CoverService.cancel(id, user); } catch { /* already final */ } };
  signal.addEventListener('abort', cancel, { once: true });
  try {
    for (;;) {
      if (signal.aborted) throw new Error('Cancelled');
      const result = yue2CoverService.find(id, user);
      if (result.abc) return { abc: result.abc, scoreSource: 'sheetsage2' };
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
/** Overrides for the Cover kinds routes/yue2Cover.ts registers at import,
 *  read each time a step runs. Only an isolated harness sets them (the
 *  frontend reference suite, which must never reach a real LLM or the YuE2
 *  transcriber); production never calls this, so every default stands.
 *  Returns the function that clears them. */
let runtimeOverrides: CoverDeps = {};
export function overrideCoverWorkflowDeps(overrides: CoverDeps): () => void {
  runtimeOverrides = overrides;
  return () => { runtimeOverrides = {}; };
}

export function createCoverKinds(deps: CoverDeps = {}): WorkflowKind<any>[] {
  const dep = <K extends keyof CoverDeps>(key: K): CoverDeps[K] => runtimeOverrides[key] ?? deps[key];
  const docs = () => deps.documents || workflowDocuments();
  const asset = deps.asset || defaultAsset;
  return [{ kind: 'cover-open', input: open, async run(ctx) {
    const input = ctx.input as z.infer<typeof open>;
    const { assetId } = input;
    const source = asset(assetId, ctx.userId);
    ctx.emit('stage', { stage: 'Analyzing source' });
    const datasetAnalysis = input.cached ? null : (dep('datasetAnalysis') || readCoverDatasetAnalysis)(source.path);
    const [metadata, result] = input.cached ? [input.cached.metadata, null] : await Promise.all([
      (dep('metadata') || defaultMetadata)(source.path),
      datasetAnalysis ? Promise.resolve(null) : (dep('analyze') || analyzeWithEssentia)(source.path, ctx.signal),
    ]);
    ctx.throwIfCancelled();
    const analysis = input.cached?.analysis || datasetAnalysis || (result
      ? { bpm: result.bpm || 120, key: `${result.key || 'C'} ${result.scale || 'major'}`, scale: result.scale }
      : null);
    const document = docs().create(ctx.userId, 'cover-draft', {
      assetId, sha256: source.sha256, sourceLabel: source.filename, metadata, analysis,
      abc: '', approvedAbc: '', leadSheetSource: input.leadSheetSource || (dep('leadSheetDefault') || defaultLeadSheetSource)(),
    });
    return { documentId: document.id, revision: document.revision, metadata, analysis, leadSheetSource: document.data.leadSheetSource };
  } }, { kind: 'cover-source', input: coverSourceSchema, async run(ctx) {
    const input = ctx.input as z.infer<typeof coverSourceSchema>;
    const data = draft(docs(), input.documentId, input.revision, ctx.userId);
    const selected = asset(input.assetId || data.assetId, ctx.userId);
    const changed = selected.sha256 !== data.sha256 || (input.assetId || data.assetId) !== data.assetId;
    const next = changed || input.leadSheetSource !== data.leadSheetSource
      ? clearCoverScore(data, input.leadSheetSource) : data;
    const updated = docs().update(input.documentId, ctx.userId, input.revision, {
      ...next, assetId: input.assetId || data.assetId, sha256: selected.sha256,
      ...(changed ? { sourceLabel: selected.filename, analysis: null, metadata: null } : {}),
    });
    return { documentId: updated.id, revision: updated.revision, leadSheetSource: input.leadSheetSource };
  } }, { kind: 'cover-caption', input: captionInput, async run(ctx) {
    const input = ctx.input as z.infer<typeof captionInput>;
    const data = draft(docs(), input.documentId, input.revision, ctx.userId);
    const source = asset(data.assetId, ctx.userId);
    if (source.sha256 !== data.sha256) throw new WorkflowError(409, 'Cover source changed');
    ctx.emit('stage', { stage: 'Resolving cover caption' });
    const result = await (dep('caption') || defaultCaption)(input.artistId, input.provider, input.model, input.force, ctx.signal);
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
    const leadSheetSource = data.leadSheetSource || (dep('leadSheetDefault') || defaultLeadSheetSource)();
    ctx.emit('stage', { stage: 'Transcribing score' });
    const result = await (dep('transcribe') || defaultTranscribe)(source.url, data.sourceLabel || source.filename,
      ctx.userId, ctx.signal, input.force, leadSheetSource);
    ctx.throwIfCancelled();
    const updated = docs().update(input.documentId, ctx.userId, input.revision, current => {
      if (current.assetId !== data.assetId) throw new WorkflowError(409, 'Cover source changed');
      if (asset(data.assetId, ctx.userId).sha256 !== data.sha256) throw new WorkflowError(409, 'Cover source changed');
      const scoreSource = result.scoreSource || (leadSheetSource === 'sheetsage2-refined' ? 'sheetsage2' : leadSheetSource);
      const sheet = result.leadSheet || tryAbcCoverLeadSheet(result.abc, leadSheetSource, scoreSource as CoverLeadSheetResult['scoreSource']);
      return { ...current, leadSheetSource, abc: result.abc, approvedAbc: '', scoreSource, leadSheet: sheet };
    });
    return { documentId: updated.id, revision: updated.revision, abc: result.abc, scoreSource: updated.data.scoreSource, leadSheet: updated.data.leadSheet };
  } }, { kind: 'cover-render', input: render, async run(ctx) {
    let input = ctx.input as CoverRenderInput;
    const data = draft(docs(), input.documentId, input.revision, ctx.userId);
    if (input.prepare !== undefined && (input.expectedBackend === 'yue2' || input.expectedBackend === 'heartmula')) {
      const pair = preparedPair(input.prepare, data.approvedAbc || undefined);
      if ('error' in pair) throw new WorkflowError(409, pair.error);
      const conflict = instrumentalConflict(pair, input.instrumental);
      if (conflict) throw new WorkflowError(409, conflict);
      input = { ...input, lyrics: pair.lyrics };
    }
    if (input.expectedBackend !== 'sa3' && !input.instrumental && !input.lyrics.trim()) throw new WorkflowError(400, 'Enter lyrics or enable Instrumental mode');
    const source = asset(data.assetId, ctx.userId);
    if (source.sha256 !== data.sha256) throw new WorkflowError(409, 'Cover source changed');
    if (!await (dep('capability') || defaultCapability)(input.expectedBackend)) throw new WorkflowError(400, 'Backend does not support covers');
    ctx.throwIfCancelled();
    const current = draft(docs(), input.documentId, input.revision, ctx.userId);
    if (current.assetId !== data.assetId || current.sha256 !== source.sha256) throw new WorkflowError(409, 'Cover source changed');
    const request = effectiveCoverRequest(input, current, source.url);
    const item = ctx.audio.enqueue('render', request, { source: 'cover-studio', workflowJobId: ctx.jobId });
    const finished = await ctx.audio.wait(item.id);
    ctx.throwIfCancelled();
    if (finished.status !== 'succeeded') throw new Error(finished.error || `Audio ${finished.status}`);
    return { request, audioIntentId: item.id, audio: finished.result, elapsedSeconds: finished.elapsedSeconds };
  } }];
}
export function registerCoverWorkflows(deps: CoverDeps = {}) {
  for (const kind of createCoverKinds(deps)) registerWorkflowKind(kind);
}
