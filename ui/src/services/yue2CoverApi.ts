import { queryClient } from '../queries/client';
import { serverKey } from '../queries/keys';
import { apiClient, type ClientOptions, type ReadOptions } from './httpClient';
import { readSnapshot, changed } from '../queries/cover-lyrics/reads';
import { getSessionEpoch } from '../queries/keys';
import type { PrepareV2 } from '../utils/yue2Prepare';
// Authenticated YuE2 cover transcription endpoints.
const BASE = '/api/yue2-cover';

export interface Yue2CoverSourceInput {
  sourceAudioUrl?: string;
  songId?: string;
  sourceLabel?: string;
  abc?: string;
  force?: boolean;
  /** Lead-sheet transcriber (the YuE2 Models dropdown); SheetSage2 when absent. */
  transcriber?: 'sheetsage2' | 'sheetsage2-refined';
  /** With SheetSage2-Refined: keys with modes. */
  keyMode?: boolean;
}

export interface Yue2CoverReadiness {
  ready: boolean;
  model: string | null;
  message: string | null;
  /** SheetSage2-Refined's overlay is installed beside SheetSage2. */
  refined?: boolean;
}

export interface Yue2CoverJob {
  id: string;
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  phase: string;
  done: number;
  total: number;
  error: string | null;
}

export interface Yue2CoverResult {
  status?: 'queued' | 'done';
  jobId?: string;
  job?: Yue2CoverJob;
  abc?: string;
  sourceId: string;
  sourceLabel: string;
  scoreSource?: 'dataset';
  sections?: Yue2ScoreSection[];
}

export interface Yue2ScoreSection { label: string; startBar: number }
export interface Yue2SectionReview {
  sections: Yue2ScoreSection[];
  lint: { ok: boolean; scoreCount: number; lyricCount: number; message: string };
}

/** POST /sections/match: lyrics retagged from where the source vocal sings each block. */
export interface Yue2SectionMatch {
  lyrics: string;
  blocks: Array<{ index: number; tag: string | null; newTag: string | null; section: number | null;
    firstWordSeconds: number | null; status: 'kept' | 'renamed' | 'merged' | 'unsure' | 'dropped' }>;
  filled: Array<{ section: number; label: string; copiedFrom: number | null }>;
  unchanged: boolean;
  datasetSong: boolean;
  /** For a dataset song: what the proposal was made against; a save sends it back. */
  revision?: string;
}

/** Match score to lyrics: the score's new sections (1-based bars) and the
 *  lyrics retagged to them. */
export interface Yue2ScorePlan {
  sections: Array<{ label: string; bar: number; block: number | null; tag: string | null }>;
  lyrics: string;
  unsure: Array<{ block: number; tag: string | null }>;
  unchanged: boolean;
  datasetSong: boolean;
  revision?: string;
}

/** POST /sections/suggest: proposals for a Prepare document, never applied by the server. */
export interface Yue2AssignmentSuggestions {
  /** What it was computed for; drop the reply if the document no longer matches. */
  sourceKey: string;
  structureHash: string;
  lyricRevision: number;
  /** Section id -> block ids sung there in order, or 'instrumental'. */
  assignments: Record<string, string[] | 'instrumental'>;
  /** Sections to fill with a copy of a block placed elsewhere. */
  reuse: Array<{ sectionId: string; blockId: string }>;
  uncertain: string[];
  unplaced: string[];
  /** True when the recording placed the blocks. */
  recording: boolean;
  /** Set when the recording was asked for and failed; the proposals are the instant ones. */
  recordingError?: string;
}

export interface Yue2CoverDatasetMetadata {
  matched: boolean;
  metadataAvailable?: boolean;
  datasetId?: string;
  sampleId?: string;
  lyrics?: string;
  bpm?: number;
  key?: string;
  isInstrumental?: boolean;
  abc?: string;
  sections?: Yue2ScoreSection[];
  lyricsSource?: 'dataset-sidecar';
}

export interface Yue2CoverDriftResult {
  metricVersion: number;
  tempoBpm: number;
  meter: string;
  secondsPerBar: number;
  tempoSource: 'rendered-score' | 'source-score-fallback';
  stemChecked: boolean;
  sectionWarning: string | null;
  sungLyricBlocks: number;
  unmatchedLyricBlocks: Array<{ index: number; label: string }>;
  sections: Array<{
    scoreLabel: string;
    lyricTag: string | null;
    startBar: number;
    endBar: number;
    expected: { start: number | null; end: number };
    boundary: { start: number; end: number };
    sung: { start: number; end: number } | null;
    offsetBars: number | null;
    unscoredReason: 'no_matching_lyric_tag' | 'no_vocal_note' | 'no_aligned_words' | 'low_word_confidence' |
      'mix_stem_disagreement' | 'whisper_transcript_unreliable' | 'no_whisper_match' |
      'whisper_disagreement' | null;
    disagreementBars: number | null;
  }>;
  meanAbsoluteOffsetBars: number | null;
  firstOverOneBar: { index: number; label: string; offsetBars: number } | null;
  inputHash?: string;
}

/** A Build-by-section session (server/src/services/yue2Takes.ts). Frames
 *  are codec frames, 25 per second; sections are 0-based. */
export interface Yue2BuildTake {
  id: string;
  /** The build revision it was made at; a take of an older one cannot be kept. */
  revision: number;
  from: number;
  to: number;
  seed: number;
  audioUrl: string;
  frames: number;
  starts: Array<number | null>;
  /** Loudness 0..1 of its new part, from 4 s before its first section. */
  peaks?: number[];
  /** The kept take it resumed (null: it starts the song). */
  parent: string | null;
  /** Codec frames it resumed; its new part starts here (older takes: starts[from]). */
  prefix?: number;
  /** A crop of this take: same take, cut short. */
  cropOf?: string;
  /** The codec frame its audio file starts at (a take renders from a lead-in before its new part); missing: 0. */
  audioStart?: number;
  /** Where the next take resumes when a crop cut into a section it composes again, or kept part of the one it ran on into. */
  resume?: number;
  /** The source section id (or improv) its prompt ran on into after `to`; missing when it composed an ending. */
  after?: string;
  /** Ids of the occurrences it composed, from..to. */
  occurrences: string[];
  /** The style prompt it was sent, adapter trigger included (older takes: missing). */
  style?: string;
  /** Its own caption; missing when it used the session's. */
  caption?: string;
}
/** A section of the score, as the arrangement names it (s0, s1, ... in score order). */
export interface Yue2SourceSection { id: string; label: string; startBar: number; bars: number; lyricBlock: number | null }
/** The output: which source section each occurrence (o0, o1, ...) sings. Kept ones come first. */
export interface Yue2Arrangement {
  sections: Yue2SourceSection[];
  /** `lyrics`: the occurrence's own sung lines, when it has been given some. */
  /** section is a source section id, or 'improv' with its own bar count. */
  plan: Array<{ id: string; section: string; lyrics?: string; bars?: number }>;
  /** The plan ends the song; until then a take runs a few seconds into the source section after the last one. */
  finished: boolean;
}
export interface Yue2BuildSession {
  id: string;
  /** Counts changes to what the next take is sent (lyrics, plan, settings); mutations name the one they saw. */
  revision: number;
  sections: Array<{ label: string; startBar: number }>;
  takes: Yue2BuildTake[];
  kept: string[];
  pending: Array<{ from: number; to: number; count: number; activeCount?: number; phase?: string; progress?: number }>;
  error: string | null;
  params: Record<string, unknown>;
  arrangement: Yue2Arrangement;
  /** Where Save song starts the song, faded in: seconds into a first-section take (`take` is the uncropped original). */
  trim?: { take: string; seconds: number } | null;
  /** The whole build rendered once for Play build; stale once `spine` is no longer the last kept take. */
  preview?: { spine: string; status: 'rendering' | 'done' | 'failed'; progress?: number; audioUrl?: string; error?: string } | null;
}

async function request<T>(path: string, init?: RequestInit, clientOpts?: ClientOptions): Promise<T> {
  const target = { session: getSessionEpoch(), api: apiClient.baseUrl };
  const url = `${BASE}${path}`.replace(/^\/api/, '');
  if (!init?.method || init.method === 'GET') return readSnapshot<T>('cover', url, { signal: init?.signal ?? undefined }, /\/(build|transcriptions)\//.test(path) ? 0 : undefined);
  const opts = { signal: init.signal ?? undefined, ...clientOpts };
  const body = init.body ? JSON.parse(String(init.body)) : undefined;
  if (path === '/source-metadata' || path === '/sections/review') {
    return queryClient.query({
      queryKey: serverKey('cover-lyrics/cover', { auth: true, id: path, filters: body }),
      queryFn: ({ signal }) => apiClient.post<T>(url, body, { signal: opts.signal ? AbortSignal.any([signal, opts.signal]) : signal }),
    });
  }
  const result = init.method === 'DELETE' ? await apiClient.delete<T>(url, opts) : await apiClient.post<T>(url, body, opts);
  await changed('cover', target);
  return result;
}

export const yue2CoverApi = {
  readiness: (options: ReadOptions = {}) => request<Yue2CoverReadiness>('/readiness', options),
  reviewSections: (abc: string, lyrics: string, options: ReadOptions = {}) => request<Yue2SectionReview>('/sections/review', {
    ...options, method: 'POST', body: JSON.stringify({ abc, lyrics }),
  }),
  /** A 2/4 score rewritten in 4/4, for preview; nothing is stored. With `pickup` the
   *  first bar stays 2/4. `twoFourBars` counts the 2/4 bars left (pickup, odd bars
   *  before a section, the last bar). Fails (422) on anything it cannot convert exactly. */
  // Silent: CoverStudio renders the remeter failure itself (its inline line).
  remeterScore: (abc: string, pickup: boolean) =>
    apiClient.post<{ abc: string; twoFourBars: number }>(`${BASE}/score/remeter`.replace(/^\/api/, ''), { abc, pickup }, { silent: true }),
  shiftBars: (abc: string, beats: number) =>
    apiClient.post<{ abc: string }>(`${BASE}/score/shift-bars`.replace(/^\/api/, ''), { abc, beats }, { silent: true }),
  matchSections: (input: Yue2CoverSourceInput & { abc: string; lyrics: string }) =>
    request<Yue2SectionMatch>('/sections/match', { method: 'POST', body: JSON.stringify(input) }),
  // All the build commands are silent: Yue2Build renders their failure in its own error line.
  /** `lyrics` is the build's own text; without it the session follows params.lyrics. */
  buildStart: (params: Record<string, unknown>, lyrics?: string) =>
    request<Yue2BuildSession>('/build', { method: 'POST', body: JSON.stringify({ params, ...(lyrics === undefined ? {} : { lyrics }) }) }, { silent: true }),
  buildLyricsCheck: (lyrics: string) =>
    request<{ ok: true }>('/build/lyrics-check', { method: 'POST', body: JSON.stringify({ lyrics }) }, { silent: true }),
  buildLyrics: (id: string, revision: number, lyrics: string) =>
    request<Yue2BuildSession>(`/build/${id}/lyrics`, { method: 'POST', body: JSON.stringify({ revision, lyrics }) }, { silent: true }),
  /** Unbuilt occurrences' own lines (occurrence id to text), or null to sing the build's again. */
  buildOccurrenceLyrics: (id: string, revision: number, edits: Record<string, string | null>) =>
    request<Yue2BuildSession>(`/build/${id}/occurrence-lyrics`, { method: 'POST', body: JSON.stringify({ revision, edits }) }, { silent: true }),
  /** Before the first keep: the cover's voices, tempo, key, keyShift and cfgScale (as
   *  params.yue2Cover carries them) and AR strength. Takes made before it cannot be kept. */
  buildSettings: (id: string, revision: number, settings: Record<string, unknown>) =>
    request<Yue2BuildSession>(`/build/${id}/settings`, { method: 'POST', body: JSON.stringify({ revision, settings }) }, { silent: true }),
  /** Replaces the unbuilt part of the plan: `upcoming` lists source section ids (s0...) for new
   *  occurrences, or ids of upcoming occurrences (o3...) to keep them, in the order to build. */
  buildPlan: (id: string, revision: number, upcoming: string[], finished: boolean) =>
    request<Yue2BuildSession>(`/build/${id}/plan`, { method: 'POST', body: JSON.stringify({ revision, upcoming, finished }) }, { silent: true }),
  buildGet: (id: string, options: ReadOptions = {}) => request<Yue2BuildSession>(`/build/${id}`, options),
  /** `caption`, when non-empty, replaces the session's caption for these takes only. */
  buildGenerate: (id: string, revision: number, to: number, count: number, caption?: string) =>
    request<Yue2BuildSession>(`/build/${id}/generate`, { method: 'POST', body: JSON.stringify({ revision, to, count, ...(caption ? { caption } : {}) }) }, { silent: true }),
  buildCancel: (id: string) =>
    request<Yue2BuildSession>(`/build/${id}/cancel`, { method: 'POST' }, { silent: true }),
  buildKeep: (id: string, revision: number, take: string) =>
    request<Yue2BuildSession>(`/build/${id}/keep`, { method: 'POST', body: JSON.stringify({ revision, take }) }, { silent: true }),
  /** Drops the last kept take, or every kept take from `section` on. */
  buildUndo: (id: string, revision: number, section?: number) =>
    request<Yue2BuildSession>(`/build/${id}/undo`, { method: 'POST', body: JSON.stringify(section === undefined ? { revision } : { revision, section }) }, { silent: true }),
  /** Cuts the last kept take at codec frame `frame`, or restores it with null. `mode` answers a cut inside
   *  a section: compose it again from the cut, or move on to the next one there. */
  buildCrop: (id: string, revision: number, take: string, frame: number | null, mode: 'regenerate' | 'advance' | undefined) =>
    request<Yue2BuildSession>(`/build/${id}/crop`, { method: 'POST', body: JSON.stringify({ revision, take, frame, mode }) }, { silent: true }),
  /** Renders everything kept so far once, for Play build. */
  buildPreview: (id: string) =>
    request<Yue2BuildSession>(`/build/${id}/preview`, { method: 'POST', body: JSON.stringify({}) }, { silent: true }),
  /** Where Save song starts the song: `seconds` into first-section take `take`, or null for the top. */
  buildTrim: (id: string, revision: number, take: string, seconds: number | null) =>
    request<Yue2BuildSession>(`/build/${id}/trim`, { method: 'POST', body: JSON.stringify({ revision, take, seconds }) }, { silent: true }),
  buildDelete: (id: string, revision: number, take: string) =>
    request<Yue2BuildSession>(`/build/${id}/takes/${take}?revision=${revision}`, { method: 'DELETE' }, { silent: true }),
  buildDeleteAll: (id: string) =>
    request<{ ok: true }>(`/build/${id}`, { method: 'DELETE' }, { silent: true }),
  planScoreSections: (input: Yue2CoverSourceInput & { abc: string; lyrics: string }) =>
    request<Yue2ScorePlan>('/sections/plan-score', { method: 'POST', body: JSON.stringify(input) }),
  suggestAssignments: (input: Yue2CoverSourceInput & { prepare: PrepareV2; useRecording?: boolean }) =>
    request<Yue2AssignmentSuggestions>('/sections/suggest', { method: 'POST', body: JSON.stringify(input) }),
  /** Lyrics (.txt) and/or the revised score (.abc) of a matched proposal, with the
   *  `revision` it returned; or the score's tempo and key into the .txt. The two
   *  files are separate writes: `abcSaved: false` means the lyrics went in and the score did not. */
  // Silent: CoverStudio's score sheet renders the save failure itself (its
  // "Save to dataset:" line is the sole notifier).
  saveDatasetDetails: (input: Yue2CoverSourceInput & ({ lyrics?: string; abc?: string; revision: string } | { bpm: number; key: string })) =>
    request<{ saved: true; file: string; abcSaved?: boolean; abcError?: string }>('/sections/save-dataset', { method: 'POST', body: JSON.stringify(input) }, { silent: true }),
  sourceMetadata: (input: Yue2CoverSourceInput, options: ReadOptions = {}) => request<Yue2CoverDatasetMetadata>('/source-metadata', {
    ...options, method: 'POST', body: JSON.stringify(input),
  }),
  start: (input: Yue2CoverSourceInput) => request<Yue2CoverResult>('/transcriptions', {
    method: 'POST', body: JSON.stringify(input),
  }),
  status: (jobId: string, options: ReadOptions = {}) => request<Yue2CoverResult>(`/transcriptions/${encodeURIComponent(jobId)}`, options),
  cancel: (jobId: string) => request<Yue2CoverResult>(`/transcriptions/${encodeURIComponent(jobId)}`, { method: 'DELETE' }),
  // Yue2CoverDrift renders this failure itself (its inline error line), so the
  // transport stays quiet — one notifier per failure.
  measureDrift: (songId: string) => request<Yue2CoverDriftResult>(
    `/drift/${encodeURIComponent(songId)}`, { method: 'POST' }, { silent: true }),
};
