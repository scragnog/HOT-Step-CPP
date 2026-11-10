// contracts/studioWorkflows.ts — wire input/result shapes for the Insta-Gen,
// Cover, Repaint and Lego (layer) studio kinds registered on /api/workflows
// (contracts/workflow.ts owns the generic job/document envelope). Services
// under services/workflows/ and services/lireek/ import these schemas
// instead of redefining them, so the published shape is the one they
// actually validate against — not a parallel description of it.
//
// Stem separation (stemStudio.ts, supersep.ts) and Create's control
// dictionary (generation.ts, generationControls.ts) are published
// separately; this file does not cover them.

import { z } from 'zod/v4';
import { writtenSongIntentSchema } from './resolution.js';

// ── Insta-Gen (kinds: insta-preview, insta-direct, insta-approve) ──────────

const instaMode = z.enum(['instrumental', 'lyrics', 'lyrics-ai']);

/** POST /api/workflows/jobs { kind: 'insta-preview' | 'insta-direct', input }.
 *  `model` is a required field — it must be present, though `''` satisfies
 *  this schema — and `systemPrompt` is genuinely optional. Validation runs
 *  first; only after it passes does createInstaGenKinds' `.transform` fill
 *  a falsy `model` from the caller's saved provider default and a missing
 *  `systemPrompt` from their saved prompt. A request missing the `model`
 *  key outright is still rejected here. */
export const instaInputSchema = z.object({
  caption: z.string().min(1), genres: z.array(z.string()), lyricMode: instaMode,
  subject: z.string(), randomSubject: z.boolean(), provider: z.string(), model: z.string(),
  vocalLanguage: z.string(), thinking: z.boolean(),
  engineParams: z.record(z.string(), z.unknown()), expectedBackend: z.string().min(1),
  coResident: z.boolean(), cacheLmCodes: z.boolean(),
  systemPrompt: z.string().optional(),
}).superRefine((input, ctx) => {
  if (input.lyricMode === 'lyrics-ai' && !input.provider) ctx.addIssue({ code: 'custom', path: ['provider'], message: 'An LLM provider is required' });
  if (input.lyricMode === 'lyrics-ai' && !input.randomSubject && !input.subject.trim()) ctx.addIssue({ code: 'custom', path: ['subject'], message: 'A subject is required' });
});
export type InstaInput = z.infer<typeof instaInputSchema>;

/** `insta-preview` result, and the `result` field of the stored preview
 *  document consumed by `insta-approve`. */
export interface InstaResult {
  caption: string; lyrics: string; title?: string; bpm?: number; duration?: number;
  keyScale?: string; timeSignature?: string; vocalLanguage: string;
}

/** POST /api/workflows/jobs { kind: 'insta-approve', input }. References an
 *  `insta-preview` document; 409 if `revision` is stale. */
export const instaApproveSchema = z.object({ documentId: z.string().uuid(), revision: z.number().int().min(1) });

/** Common `insta-direct`/`insta-approve` render result: the engine request
 *  actually sent, the audio queue item it became, and that item's result
 *  once the queue wait resolves (`/api/audio-queue` shapes own that part). */
export interface InstaRenderResult { request: Record<string, unknown>; audioIntentId: string; audio: unknown }

// ── Cover (kinds: cover-open, cover-caption, cover-transcribe, cover-render) ─

/** Every Cover step after `cover-open` references its draft document by
 *  id + revision; a stale revision is a 409 naming the current one. */
export const coverDraftRefSchema = z.object({ documentId: z.string().uuid(), revision: z.number().int().min(1) });
export const leadSheetSourceSchema = z.enum(['sheetsage2', 'sheetsage2-refined', 'muscriptor', 'yourmt3-chordnet']);
export type CoverLeadSheetSource = z.infer<typeof leadSheetSourceSchema>;
export const coverScoreSourceSchema = z.enum(['dataset', 'musicxml', 'sheetsage2', 'muscriptor', 'yourmt3-chordnet']);
/** `cover-source` changes the transcriber or audio at a pinned draft revision.
 * A changed choice or audio clears ABC, approval and timing together. Use this
 * kind rather than copying those fields into a generic document update.
 * Readiness is GET /api/yue2/cover/readiness?source=<leadSheetSource>; the
 * existing {source, ready, model, refined, message} response is unchanged.
 * Native remains not installed until the native spawn wrapper is available.
 * Approved MusicXML/dataset scores can render without invoking readiness.
 * `dataset` stays the stored spelling; generation metadata uses dataset-sidecar. */
export const coverSourceSchema = coverDraftRefSchema.extend({
  leadSheetSource: leadSheetSourceSchema, assetId: z.string().uuid().optional(),
});

/** POST /api/workflows/jobs { kind: 'cover-open', input }. `leadSheetSource`
 *  overrides the stored Models choice, which is snapshotted on creation.
 *  A subsequent global preference change never rewrites a saved draft.
 *  `cached` lets a
 *  caller that already has metadata/analysis for this asset skip Node's
 *  parse + Essentia pass. Creates a `cover-draft` document. */
export const coverOpenSchema = z.object({
  assetId: z.string().uuid(),
  leadSheetSource: leadSheetSourceSchema.optional(),
  cached: z.object({
    metadata: z.object({ artist: z.string(), title: z.string(), album: z.string(), duration: z.number().finite().nullable() }),
    analysis: z.object({ bpm: z.number().finite(), key: z.string(), scale: z.string().optional() }),
  }).optional(),
});
export interface CoverOpenResult {
  documentId: string; revision: number;
  leadSheetSource?: CoverLeadSheetSource;
  metadata: { artist: string; title: string; album: string; duration: number | null };
  analysis: { bpm: number; key: string; scale?: string } | null;
}

/** POST /api/workflows/jobs { kind: 'cover-caption', input }. */
export const coverCaptionSchema = coverDraftRefSchema.extend({
  artistId: z.number().int().positive(), provider: z.string(), model: z.string(), force: z.boolean(),
});
export interface CoverCaptionResult { documentId: string; revision: number; caption: string }

/** POST /api/workflows/jobs { kind: 'cover-transcribe', input }. Resolves an
 *  editable ABC score using the draft's leadSheetSource. Legacy drafts without
 *  that field snapshot the stored Models choice on first transcription.
 *  `force` skips a cached score. Source/audio edits reject late results through
 *  the document revision and source hash. Persist the returned revision.
 *  Incomplete ABC is saved for editing without leadSheet timing. Rendering
 *  still requires a readable tempo and supported notation.
 *  HeartMuLa renders record leadSheetRenderingPath: native uses raw seconds,
 *  abc uses parsed score seconds, abc-edited regenerates from approved edits. */
export const coverTranscribeSchema = coverDraftRefSchema.extend({ force: z.boolean().default(false) });
export interface CoverLeadSheetResult {
  abc: string; scoreSource: z.infer<typeof coverScoreSourceSchema>; leadSheetSource: CoverLeadSheetSource;
  chordQuality: 'estimated' | 'transcribed' | 'provided';
  timing: {
    conversionVersion: 1; path: 'abc' | 'native'; bpm: number;
    /** Raw seconds, never Beat This! positions or sixteenth indices. */
    rawEvents: { melody: Array<{ onset: number; offset: number; pitch: number }>;
      chords: Array<{ onset: number; offset: number; pitch: number }>;
      drums?: Array<{ onset: number; offset: number; pitch: number }> };
    grid?: { identity: string; data: unknown };
  };
}
export interface CoverTranscribeResult { documentId: string; revision: number; abc: string; scoreSource?: string; leadSheet?: CoverLeadSheetResult }

/** POST /api/workflows/jobs { kind: 'cover-render', input }. `analysis` and
 *  the ACE/YuE2/SA3 `controls` block carry every per-run override the engine
 *  request needs; the draft document supplies the resolved asset and, for
 *  YuE2 or HeartMuLa, the approved ABC score (409 if not approved).
 *  HeartMuLa requires heartmulaTags, ignores caption and YuE2 controls, and
 *  derives fresh rolls from that approved score. Its generation backend still
 *  returns "HeartMuLa covers are not available yet" until the engine is ready.
 *  `prepare` (YuE2/HeartMuLa, optional) is contracts/yue2Prepare.ts's document:
 *  when sent, both its confirmations must be current and its structure must
 *  be the approved score, and the lyrics rendered are the ones it exports,
 *  replacing `lyrics` (409 otherwise). */
export const coverRenderSchema = coverDraftRefSchema.extend({
  expectedBackend: z.enum(['ace', 'yue2', 'sa3', 'heartmula', 'minimax-m3']), engineParams: z.record(z.string(), z.unknown()),
  /** MuLaCover uses four explicit fields; caption is never parsed into tags. */
  heartmulaTags: z.object({ topic: z.string().max(512), genre: z.string().max(512),
    instrument: z.string().max(512), mood: z.string().max(512) }).optional(),
  // SA3 overrides. The existing controls/engineParams fields remain accepted.
  strength: z.number().min(0.1).max(0.9).optional(),
  seed: z.number().int().min(-1).max(Number.MAX_SAFE_INTEGER).optional(),
  adapter: z.string().regex(/^[^/\\:,]+:-?(?:\d+(?:\.\d*)?|\.\d+)$/).optional(),
  steps: z.number().int().min(1).max(64).optional(),
  /** MM3 cover: Plan = the generation request's mm3Plank and force levers (plank
   *  null = Plan off); Sound = audio to audio from the source (soundStrength null = off). */
  mm3Cover: z.object({
    plank: z.string().min(1).nullable(), semanticOnly: z.boolean(),
    soundStrength: z.number().gt(0).max(1).nullable().optional(), soundHold: z.number().min(0).max(1).optional(),
    forceBias: z.number().min(0).max(1000).nullable().optional(),
    pattern: z.object({ prefixSeconds: z.number().positive().optional(), anchorEvery: z.number().int().positive().optional(),
      pinnedRanges: z.array(z.tuple([z.number().min(0), z.number().positive()])).optional(), invert: z.boolean().optional() }).nullable().optional(),
    trimSeconds: z.number().positive().nullable(),
    arSeed: z.number().int().min(-1).max(Number.MAX_SAFE_INTEGER),
    /** Cover mode (absent = pin). The guided modes read the plank's guide and,
     *  for blend, its launder state; see coverWorkflow.ts. */
    mode: z.enum(['pin', 'follow', 'follow-depth', 'follow-stream', 'product', 'blend']).optional(),
    strength: z.number().min(0).max(1).optional(),
    candidates: z.number().int().min(1).max(32).optional(),
    stateFix: z.enum(['off', 'depth', 'stream', 'both']).optional(),
    weight: z.number().min(0).max(4).optional(),
    blend: z.number().min(0).max(100).optional(),
  }).optional(),
  title: z.string(), artistName: z.string(), targetArtistName: z.string(),
  lyrics: z.string(), caption: z.string(),
  instrumental: z.boolean(), lyricsSource: z.string().nullable(),
  scoreSource: coverScoreSourceSchema.nullable(),
  prepare: z.unknown().optional(),
  analysis: z.object({ bpm: z.number().finite(), key: z.string(), scale: z.string().optional() }).nullable().optional(),
  settings: z.record(z.string(), z.unknown()),
  controls: z.object({
    sa3Transform: z.number().min(0.1).max(0.9).default(0.3),
    sa3Adapters: z.string().default(''), // Comma-separated engine name:scale pairs.
    bpmOverride: z.number().finite().positive().nullable(), bpmCorrection: z.number().finite().positive(),
    keyOverride: z.string().nullable(), tempoScale: z.number().finite().positive(),
    pitchShift: z.number().int().min(-12).max(12), noFsq: z.boolean(),
    audioCoverStrength: z.number().finite(), coverNoiseStrength: z.number().finite(),
    coverNoiseMethod: z.string(), vocalLanguage: z.string(), sourceLatentUrl: z.string(),
    timbreOverridePath: z.string(), presetAdapterPath: z.string(), presetReferencePath: z.string(),
    triggerUseFilename: z.boolean(), triggerPlacement: z.string(),
    voices: z.enum(['vocal', 'both']), keepChords: z.boolean(),
    tempoMode: z.enum(['free', 'source', 'set']), coverBpm: z.number().finite().positive(),
    keyShift: z.number().int().min(-12).max(12), cfgScale: z.number().finite(),
    lyricSchedule: z.enum(['off', 'bias', 'mask']).default('off'),
    lmAdapterAr: z.string(), lmAdapterNar: z.string(), pairMode: z.enum(['base', 'pair']),
    yue2Pick: z.record(z.string(), z.union([z.string(), z.number()])),
    captionMode: z.string(), captionTracks: z.array(z.object({
      name: z.string(), styled: z.string().optional(), caption: z.string().optional(),
      bpm: z.union([z.string(), z.number()]).optional(),
    })),
  }),
});
export type CoverRenderInput = z.infer<typeof coverRenderSchema>;
/** MuLaCover JSON wire, contiguous row-major binary values represented as f32
 * numbers. Shapes: pianoroll [length,256], chord [length,12], drums [length,128],
 * mask [length]. Melody columns are pitch*2 onset, pitch*2+1 sustain.
 * context_mask is zero, including rests. At most 5000 sixteenth steps. */
export interface HeartmulaCoverRolls {
  version: 1; bpm: number; length: number; dtype: 'float32';
  pianoroll: number[]; chord: number[]; drum_pianoroll: number[]; context_mask: number[];
}
/** A persisted `cover-draft` document's body, as `cover-render` reads it. */
export interface CoverDraft {
  assetId: string; sha256: string; sourceLabel?: string;
  analysis?: { bpm: number; key: string; scale?: string } | null;
  abc?: string; approvedAbc?: string; scoreSource?: string;
  leadSheetSource?: CoverLeadSheetSource;
  leadSheet?: CoverLeadSheetResult;
}
export interface CoverRenderResult { request: Record<string, unknown>; audioIntentId: string; audio: unknown }

// ── Repaint and Lego layer (kinds: repaint-render, layer-render) ───────────

/** Both kinds take either an uploaded asset or a library song as source,
 *  pinned by `expectedUrl`: a 409 if the asset/song's current URL no longer
 *  matches, so a stale picker selection cannot render against the wrong audio. */
export const repaintLayerSourceSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('asset'), id: z.string().uuid(), expectedUrl: z.string().min(1) }),
  z.object({ kind: z.literal('song'), id: z.string().min(1), expectedUrl: z.string().min(1) }),
]);
const repaintLayerCommonSchema = z.object({
  source: repaintLayerSourceSchema, expectedBackend: z.string().min(1), engineParams: z.record(z.string(), z.unknown()),
});

/** POST /api/workflows/jobs { kind: 'repaint-render', input }. Region is
 *  checked against the source's real duration at run time (400 if it reads
 *  past the end); there is no draft document, so a retry resends the body. */
export const repaintRenderSchema = repaintLayerCommonSchema.extend({
  regionStart: z.number().finite().nonnegative(), regionEnd: z.number().finite().positive(),
  lyrics: z.string(), styleCaption: z.string(), sourceName: z.string(),
  repaintMode: z.enum(['conservative', 'balanced', 'aggressive']),
  crossfadeFrames: z.number().int().nonnegative(),
});
export type RepaintInput = z.infer<typeof repaintRenderSchema>;

/** POST /api/workflows/jobs { kind: 'layer-render', input }. `buildModel`
 *  must be a plain Base DiT checkpoint (400 otherwise); isolates one stem
 *  track from the source into its own render. */
export const layerRenderSchema = repaintLayerCommonSchema.extend({
  trackName: z.enum(['vocals', 'backing_vocals', 'drums', 'bass', 'guitar', 'keyboard', 'percussion', 'strings', 'synth', 'fx', 'brass', 'woodwinds']),
  buildModel: z.string().regex(/^acestep-v15-(?:xl-)?base-/, 'A plain Base DiT model is required'), caption: z.string(),
});
export type LayerInput = z.infer<typeof layerRenderSchema>;

/** Shared `repaint-render`/`layer-render` result shape. */
export interface RepaintLayerResult { request: Record<string, unknown>; audioIntentId: string; audio: unknown }

// ── Lyric Studio batch (kind: lyric-batch) ──────────────────────────────────
//
// One job kind, not six: the Lyric Studio's profile/fetch/generate/refine/
// render operations are all items of one discriminated-union batch body
// (1-200 items, mixed types allowed), captured and version-pinned at submit
// by services/lireek/lyricWorkflow.ts (captureLyricItems/captureRenderItems).
// That capture step, not this file, is the real input boundary: it is what
// turns a client's loose request (ids, provider, free-text subject) into
// the exact pinned item this schema accepts. The schema itself lives here,
// not there, so importing it never pulls in the DB/LLM services or the
// registerWorkflowKind side effect lyricWorkflow.ts carries at module load;
// that module imports it back from here as `lyricBatchInput`.

export const lyricItemBase = z.object({ provider: z.string().min(1), model: z.string().optional() });
const lyricProfileItem = lyricItemBase.extend({ type: z.literal('profile'), sourceId: z.number().int().positive(), sourceRevision: z.string(), artist: z.string(), songs: z.array(z.any()) });
const lyricHistorySchema = z.object({ usedSubjects: z.array(z.string()), usedBpms: z.array(z.number()), usedKeys: z.array(z.string()), usedTitles: z.array(z.string()), usedDurations: z.array(z.number()) });
const lyricGenerateItem = lyricItemBase.extend({ type: z.literal('generate'), sourceId: z.number().int().positive(), sourceRevision: z.string(), lyricsSetId: z.number().int().positive().optional(), lyricsSetRevision: z.string().optional(), profileData: z.record(z.string(), z.any()).optional(), artistId: z.number().int().positive().optional(), extraInstructions: z.string().optional(), userSubject: z.string().optional(), noThink: z.boolean().optional(), history: lyricHistorySchema.optional() });
const lyricRefineItem = lyricItemBase.extend({ type: z.literal('refine'), sourceId: z.number().int().positive(), sourceRevision: z.string(), profileId: z.number().int().positive().optional(), profileRevision: z.string().optional(), source: z.record(z.string(), z.any()), profileData: z.record(z.string(), z.any()).optional(), artist: z.string() });
export const lyricFetchItem = z.object({ type: z.literal('fetch'), artist: z.string().trim().min(1), album: z.string().optional(), maxSongs: z.number().int().min(1).max(200) });
export const lyricRenderItem = z.object({ type: z.literal('render'), intent: writtenSongIntentSchema, sourceRevision: z.string() });
const lyricPreflightItem = z.object({ type: z.literal('preflight-error'), error: z.string() });
// Generate items share one copy of each profile (keyed by profile id) and
// each artist's history (keyed by artist id). Jobs captured before that
// carry profileData and history inline on every item.
export const lyricBatchInputSchema = z.object({
  items: z.array(z.discriminatedUnion('type', [lyricProfileItem, lyricGenerateItem, lyricRefineItem, lyricFetchItem, lyricRenderItem, lyricPreflightItem])).min(1).max(10_000),
  profiles: z.record(z.string(), z.record(z.string(), z.any())).optional(),
  histories: z.record(z.string(), lyricHistorySchema).optional(),
  // How many artists run side by side; one artist's songs always run in order.
  concurrency: z.number().int().min(1).max(16).optional(),
});
export type LyricBatchInput = z.infer<typeof lyricBatchInputSchema>;

/** One `lyric-batch` result entry (the job's `result.results[]`). `value`'s
 *  shape depends on the item's type: `{ id, ... }` for profile/generate,
 *  `{ artist_id, lyrics_set_id, songs_fetched }` for fetch, `{ audioIntentId,
 *  generationId, jobId }` for render. A `preflight-error` item (a source
 *  that failed to resolve at submit) always reports `error`. */
export interface LyricBatchResultItem { index: number; status: 'done' | 'error'; value?: unknown; error?: string }
export interface LyricBatchResult { results: LyricBatchResultItem[] }
