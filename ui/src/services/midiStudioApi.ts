import { apiClient, type ReadOptions } from './httpClient';
import type { TFunction } from 'i18next';
// midiStudioApi.ts — API client for MIDI Studio (audio → MIDI transcription)
//
// Communicates with Node server /api/midi-studio endpoints, which run the
// native ace-midi engine (MuScriptor by Kyutai & Mirelo, Beat This! by CPJKU).

const API_BASE = '/api/midi-studio';
/** Path prefix for apiClient calls (the client already adds `/api`). */
const P = '/midi-studio';

// ── Types ────────────────────────────────────────────────────────────────

export type MuscriptorModel = 'small' | 'medium' | 'large';
/** What MIDI Studio can transcribe with: a MuScriptor size, or dettato (four folders, one download). */
export type MidiModel = MuscriptorModel | 'dettato';
/** dettato's four weight folders; its repairer is needed only while the Evidence repairer switch is on. */
export const DETTATO_PARTS = ['dettato', 'dettato-evidence', 'dettato-grouper', 'dettato-drums'] as const;
export type DettatoPart = typeof DETTATO_PARTS[number];
/** The status key holding each part's own state. 'dettato' itself is the total of all four. */
export const DETTATO_PART_STATUS_KEY: Record<DettatoPart, string> = {
  dettato: 'dettato-transcriber', 'dettato-evidence': 'dettato-evidence', 'dettato-grouper': 'dettato-grouper', 'dettato-drums': 'dettato-drums',
};

export interface ModelState {
  downloaded: boolean;
  sizeBytes: number;
  downloading: boolean;
  receivedBytes: number;
  totalBytes: number;
  error?: string;
  gated?: boolean;
}

/** Beat This! weights: tempo, time signature and bar lines for the MIDI */
export const BEAT_MODEL = 'beat-this';

/** dettato can transcribe when the folders this run reads are there: the repairer's only while its switch is on. */
export function dettatoReady(models: Record<string, { downloaded: boolean } | undefined> | undefined, repair: boolean): boolean {
  return DETTATO_PARTS.every(p => (!repair && p === 'dettato-evidence') || !!models?.[DETTATO_PART_STATUS_KEY[p]]?.downloaded);
}

export interface MidiStudioStatus {
  /** ace-midi engine binary present next to ace-server */
  engineAvailable: boolean;
  hfTokenSet: boolean;
  models: Record<string, ModelState>;
}

export interface MidiJobProgress {
  status: 'queued' | 'transcribing' | 'done' | 'failed' | 'cancelled';
  chunksDone?: number;
  chunksTotal?: number;
  /** dettato's stage: transcribe, repair, group, drums or done */
  phase?: string;
  sub?: number;
  noteCount?: number;
  error?: string;
  gated?: boolean;
  partial?: boolean;
}

export interface MidiJobSummary {
  id: string;
  status: 'queued' | 'transcribing' | 'done' | 'failed' | 'cancelled';
  sourceFileName: string;
  /** URL of the original audio (for synced playback in the player) */
  sourceAudioUrl?: string;
  songId?: string;
  model: MidiModel;
  noteCount: number;
  durationSec: number;
  createdAt: string;
  error?: string;
  /** Failure looks like gated-model / HF auth trouble */
  gated?: boolean;
  /** Failed, but the notes decoded before the failure were saved as a .mid */
  partial?: boolean;
  /** chunk progress — present on in-flight jobs */
  chunksDone?: number;
  chunksTotal?: number;
  phase?: string;
  /** progress inside the current step, 0 to 1 */
  sub?: number;
  /** options the MIDI was last re-exported with */
  exportOptions?: MidiExportOptions;
}

/** Both the live player and the polled job list display the engine's current phase. */
export function midiPhaseLabel(t: TFunction, phase?: string, sub?: number): string | undefined {
  switch (phase) {
    case 'repair': return t('midiStudio.statusRepairing');
    case 'group': return t('midiStudio.statusGrouping');
    case 'drums': return sub == null ? t('midiStudio.statusDrums')
      : t('midiStudio.drumProgress', { percent: Math.round(sub * 100) });
  }
}

/**
 * The MuScriptor weights are GATED on Hugging Face — each user must request
 * access on the model page (free, instant after accepting the conditions)
 * and provide a read token before the first download.
 */
export const HF_MODEL_URLS: Record<MuscriptorModel, string> = {
  small: 'https://huggingface.co/MuScriptor/muscriptor-small',
  medium: 'https://huggingface.co/MuScriptor/muscriptor-medium',
  large: 'https://huggingface.co/MuScriptor/muscriptor-large',
};

export const HF_TOKEN_SETTINGS_URL = 'https://huggingface.co/settings/tokens';

export interface MidiNote {
  pitch: number;
  velocity: number;
  channel: number;
  start: number;
  duration: number;
}

export interface MidiChannelInfo {
  channel: number;
  program: number;
  sourceProgram?: number;
  isDrums: boolean;
  noteCount: number;
}

/** The grid a tempo-aware MIDI was written on; times are seconds of the audio. */
export interface MidiGrid {
  bpm: number;
  /** 0 for older exports without a time signature or when beat tracking failed */
  beatsPerBar: number;
  /** false: the MIDI follows the performance with a tempo map */
  steady: boolean;
  onsetDelay: number;
  firstDownbeat?: number;
  subdivision: number;
  audioShift: number;
  beats: number[];
  bars: number[];
}

/** Re-export corrections (ace-midi's MuScriptor export options). */
export interface MidiExportOptions {
  bpm?: number;
  beatsPerBar?: number;
  firstDownbeat?: number;
  tempoMode?: 'auto' | 'map' | 'fixed';
  beatScale?: number;
  quantize?: boolean;
  grid?: number;
  strength?: number;
  fillGaps?: number;
  trimLeadingBars?: boolean;
  fullVelocity?: boolean;
  programOverrides?: Record<number, number>;
}

export interface ParsedMidi {
  durationSec: number;
  noteCount: number;
  channels: MidiChannelInfo[];
  notes: MidiNote[];
  /** present when the MIDI has tempo and bars (beat model downloaded) */
  grid?: MidiGrid;
}

export interface TranscribeParams {
  sourceAudioUrl: string;
  sourceFileName: string;
  songId?: string;
  model: MidiModel;
  /** speculative decoding (large only; the server defaults it on) */
  speculative?: boolean;
  /** dettato only: run the evidence repairer (the server defaults it on) */
  repair?: boolean;
  electricGuitarProgram?: number;
}

// ── API Functions ────────────────────────────────────────────────────────

/** Get MuScriptor install/environment status. */
export async function getMidiStatus(opts?: ReadOptions): Promise<MidiStudioStatus> {
  return apiClient.get<MidiStudioStatus>(`${P}/status`, opts);
}

/** Save (or clear, with '') the Hugging Face read token for gated weights.
 *  Silent: MidiStudio renders failures itself (its inline error line). */
export async function saveHfToken(token: string): Promise<void> {
  await apiClient.post(`${P}/hf-token`, { token }, { silent: true });
}

/** Begin downloading a model's gated weights. Poll getMidiStatus() for progress.
 *  Silent: MidiStudio renders failures itself (its inline error line). */
export async function startModelDownload(model: MidiModel | typeof BEAT_MODEL): Promise<void> {
  await apiClient.post(`${P}/models/${model}/download`, undefined, { silent: true });
}

/** Poll a transcription job's progress. */
export async function getMidiProgress(jobId: string, opts?: ReadOptions): Promise<MidiJobProgress> {
  return apiClient.get<MidiJobProgress>(`${P}/${jobId}/progress`, opts);
}

/** SSE URL streaming live note events for an in-flight job (Phase 6 UI). */
export function getMidiStreamUrl(jobId: string): string {
  return `${API_BASE}/${jobId}/stream`;
}

/** Queue a transcription job. Returns the job ID. Silent: MidiStudio renders
 *  failures itself (its inline error line). */
export async function submitTranscription(params: TranscribeParams): Promise<string> {
  const data = await apiClient.post<{ id: string }>(`${P}/transcribe`, params, { silent: true });
  return data.id;
}

/** Fetch the parsed note data for the piano-roll preview. */
export async function getMidiNotes(jobId: string, opts?: ReadOptions): Promise<ParsedMidi> {
  return apiClient.get<ParsedMidi>(`${P}/${jobId}/notes`, opts);
}

/** List all transcription jobs (newest first). */
export async function listMidiJobs(opts?: ReadOptions): Promise<MidiJobSummary[]> {
  return apiClient.get<MidiJobSummary[]>(`${P}/jobs`, opts);
}

/** Cancel and/or delete a job. Silent: MidiStudio renders failures itself
 *  (its inline error line). */
export async function deleteMidiJob(jobId: string): Promise<void> {
  await apiClient.delete(`${P}/${jobId}`, { silent: true });
}

/** Rewrite a job's MIDI on a corrected and/or quantized grid. Silent:
 *  MidiExportPanel renders failures itself (its inline error line). */
export async function exportMidi(jobId: string, options: MidiExportOptions): Promise<{ grid: MidiGrid | null; noteCount: number }> {
  return apiClient.post(`${P}/${jobId}/export`, options, { silent: true });
}

export interface MidiAbcResult {
  abc: string;
  summary: { vocal_notes: number; ins_notes: number; ins: string; key: string; chords: number };
}

/** The transcription as a YuE2 ABC lead sheet on the job's current grid. */
export async function getMidiAbc(jobId: string, options: { ins?: number; melodyOnly?: boolean }): Promise<MidiAbcResult> {
  // Silent: MidiExportPanel renders the failure in its inline error line.
  return apiClient.post(`${P}/${jobId}/abc`, options, { silent: true });
}

/** Direct URL for downloading the .mid file of a completed job. */
export function getMidiFileUrl(jobId: string): string {
  return `${API_BASE}/${jobId}/file`;
}

// ── GM instrument names (for channel legend) ─────────────────────────────

const GM_FAMILIES = [
  'Piano', 'Chromatic Perc.', 'Organ', 'Guitar', 'Bass', 'Strings',
  'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad',
  'Synth FX', 'Ethnic', 'Percussive', 'Sound FX',
];

/** Coarse GM name for a channel (family-level; channel 9 = drums). */
export function channelLabel(ch: MidiChannelInfo): string {
  if (ch.isDrums) return 'Drums';
  return GM_FAMILIES[Math.floor(ch.program / 8)] || `Program ${ch.program}`;
}
