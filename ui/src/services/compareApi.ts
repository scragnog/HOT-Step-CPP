// compareApi.ts - typed client for /api/compare (Compare studio).

import { apiClient, type ReadOptions } from './httpClient';

async function req<T>(method: string, path: string, body?: unknown, opts: ReadOptions = {}): Promise<T> {
  return apiClient.request<T>(method, `/compare${path}`, body, opts);
}

export type CompareStatus = 'scored' | 'partial' | 'unscored';

export interface CompareCriterion {
  id: string; label: string; type: 'scale' | 'keep' | 'note'; min?: number; max?: number;
}

export interface CompareSessionSummary {
  id: string; name: string; group: string; baselineSongId: string | null;
  archived: boolean; status: CompareStatus; scored: number; total: number;
  createdAt: string; updatedAt: string;
}

export interface CompareScore {
  criterion: string; value: number | null; keep: boolean | null; note: string; updatedAt?: string;
}

export interface CompareTake {
  songId: string; position: number; title: string | null; audioUrl: string | null;
  duration: number | null; backend: string | null; createdAt: string | null;
  seed: number | null; seedB: number | null; seedBlend: number | null;
  generationParams: Record<string, unknown>;
  scores: CompareScore[];
}

export interface CompareSession extends CompareSessionSummary {
  criteria: CompareCriterion[];
  takes: CompareTake[];
}

export interface ScoreInput {
  songId: string; criterion: string; value?: number | null; keep?: boolean | null; note?: string;
}

export interface ScoreResult { status: CompareStatus; archived: boolean; scored: number; total: number; }

export interface CompareAnalysis {
  songId: string; cached: boolean; analyzerVersion?: string;
  integratedLufs: number | null; shortTermMaxLufs: number | null;
  truePeakDbtp: number | null; samplePeakDbfs: number | null; rmsDbfs: number | null;
  crestDb: number | null; centroidHz: number | null; flatness: number | null;
  stereoCorr: number | null; durationSec: number;
  peaks: { min: number[]; max: number[] };
}

export interface CompareDiff {
  a: string; b: string; hopSec: number;
  loudnessDeltaLu: (number | null)[]; logSpectralDistanceDb: number[];
}

export interface CreateSessionInput {
  name: string; songIds: string[]; criteria?: CompareCriterion[];
  baselineSongId?: string | null; group?: string;
}

const enc = encodeURIComponent;

export const compareApi = {
  listSessions: (opts: { archived?: boolean; group?: string } = {}, read: ReadOptions = {}) => {
    const q = new URLSearchParams();
    if (opts.archived !== undefined) q.set('archived', opts.archived ? '1' : '0');
    if (opts.group !== undefined) q.set('group', opts.group);
    const s = q.toString();
    return req<{ sessions: CompareSessionSummary[] }>('GET', `/sessions${s ? '?' + s : ''}`, undefined, read);
  },
  createSession: (body: CreateSessionInput) => req<CompareSession>('POST', '/sessions', body),
  getSession: (id: string, read: ReadOptions = {}) => req<CompareSession>('GET', `/sessions/${enc(id)}`, undefined, read),
  deleteSession: (id: string) => req<{ ok: boolean }>('DELETE', `/sessions/${enc(id)}`),
  addTakes: (id: string, songIds: string[]) => req<CompareSession>('POST', `/sessions/${enc(id)}/takes`, { songIds }),
  removeTake: (id: string, songId: string) => req<CompareSession>('DELETE', `/sessions/${enc(id)}/takes/${enc(songId)}`),
  putScores: (id: string, items: ScoreInput[]) => req<ScoreResult>('PUT', `/sessions/${enc(id)}/scores`, items),
  setArchived: (id: string, archived: boolean) => req<CompareSessionSummary>('POST', `/sessions/${enc(id)}/archive`, { archived }),
  getAnalysis: (songId: string, read: ReadOptions = {}) => req<CompareAnalysis>('GET', `/analysis/${enc(songId)}`, undefined, read),
  getDiff: (a: string, b: string, read: ReadOptions = {}) => req<CompareDiff>('GET', `/diff?a=${enc(a)}&b=${enc(b)}`, undefined, read),
};

export interface CompareSongHit {
  id: string; title: string; style: string; duration: number; backend: string; createdAt: string;
}

export function searchCompareSongs(q: string, limit = 25, offset = 0, read: ReadOptions = {}) {
  const p = new URLSearchParams({ q, limit: String(limit), offset: String(offset) });
  return req<{ total: number; limit: number; offset: number; songs: CompareSongHit[] }>('GET', `/songs?${p.toString()}`, undefined, read);
}

// Library -> Compare hand-off: song ids picked in the library, consumed once by the studio.
let _pendingSongIds: string[] = [];
export function setPendingCompareSongs(ids: string[]): void { _pendingSongIds = [...ids]; }
export function takePendingCompareSongs(): string[] { const ids = _pendingSongIds; _pendingSongIds = []; return ids; }
