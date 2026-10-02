// Authenticated YuE2 cover transcription endpoints.
const BASE = '/api/yue2-cover';

export interface Yue2CoverSourceInput {
  sourceAudioUrl?: string;
  songId?: string;
  sourceLabel?: string;
  abc?: string;
  force?: boolean;
}

export interface Yue2CoverReadiness {
  ready: boolean;
  model: string | null;
  message: string | null;
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
  lyricsSource?: 'dataset-sidecar';
}

async function request<T>(path: string, token: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { Authorization: `Bearer ${token}`, ...(init?.body ? { 'Content-Type': 'application/json' } : {}) },
  });
  const result = await response.json() as T & { error?: string };
  if (!response.ok) throw new Error(result.error || `Cover transcription failed (${response.status})`);
  return result;
}

export const yue2CoverApi = {
  readiness: (token: string) => request<Yue2CoverReadiness>('/readiness', token),
  sourceMetadata: (input: Yue2CoverSourceInput, token: string) => request<Yue2CoverDatasetMetadata>('/source-metadata', token, {
    method: 'POST', body: JSON.stringify(input),
  }),
  start: (input: Yue2CoverSourceInput, token: string) => request<Yue2CoverResult>('/transcriptions', token, {
    method: 'POST', body: JSON.stringify(input),
  }),
  status: (jobId: string, token: string) => request<Yue2CoverResult>(`/transcriptions/${encodeURIComponent(jobId)}`, token),
  cancel: (jobId: string, token: string) => request<Yue2CoverResult>(`/transcriptions/${encodeURIComponent(jobId)}`, token, { method: 'DELETE' }),
};
