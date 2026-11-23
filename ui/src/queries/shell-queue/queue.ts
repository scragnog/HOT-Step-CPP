// queue.ts — the generation queue's server reads.
//
// The queue executor (stores/audioGenQueueStore.ts) keeps its own sequencing,
// legacy owner/migration rules and completion effects; Query owns the reads
// it makes. Every read goes through query(), so two loops asking for the
// same thing at once share one request, and a loop that is happy with a
// recent answer takes the cache instead of asking again. Reads never retry
// here: each executor loop already decides what a failed read means.
// Submits, cancels and other commands stay direct transport calls.

import { queryOptions } from '@tanstack/react-query';
import { generateApi, songApi } from '@/services/api';
import { audioQueueApi } from '@/services/audioQueueApi';
import { apiClient } from '@/services/httpClient';
import { queryClient } from '@/queries/client';
import { ACTIVE_POLL_MS } from '@/queries/contract';
import { serverKey } from '@/queries/keys';
import { songOptions } from './songs';
import { workflowOptions } from '@/queries/cover-lyrics/workflows';

export const audioQueueItemsOptions = () => queryOptions({
  queryKey: serverKey('shell-queue/audio-queue', { auth: true }),
  queryFn: ({ signal }) => audioQueueApi.list({ signal }),
  retry: false,
});

export const generationStatusOptions = (jobId: string, options: { signal?: AbortSignal } = {}) => queryOptions({
  queryKey: serverKey('shell-queue/generation-status', { id: jobId }),
  queryFn: ({ signal }) => generateApi.status(jobId, {
    signal: options.signal ? AbortSignal.any([signal, options.signal]) : signal,
  }),
  retry: false,
});

export interface CoverArtJob {
  status: string;
  error?: string;
  result?: { coverUrl?: string };
}

/** One cover-art generation job, polled by the prompt modal. */
export const coverArtJobOptions = (jobId: string) => queryOptions({
  queryKey: serverKey('shell-queue/cover-art-job', { id: jobId }),
  queryFn: ({ signal }) => apiClient.get<CoverArtJob>(`/cover-art/generate/${encodeURIComponent(jobId)}`, { signal }),
  staleTime: 0,
});

export interface CoverArtFileProgress {
  filename: string;
  description: string;
  status: string;
  bytesDownloaded: number;
  totalBytes: number;
  speed: number;
}
export interface CoverArtStatus {
  installed: boolean;
  missingFiles: string[];
  download: {
    phase: string;
    files: CoverArtFileProgress[];
    totalBytes: number;
    downloadedBytes: number;
    overallProgress: number;
  };
}
/** Cover-art model install status, polled while a download runs. */
export const coverArtStatusOptions = () => queryOptions({
  queryKey: serverKey('shell-queue/cover-art-status'),
  queryFn: ({ signal }) => apiClient.get<CoverArtStatus>('/cover-art/status', { signal }),
});

export interface SavedSeed {
  name: string;
  seed: number;
  saved_at: string | null;
  description: string;
  tags: string[];
  favorite: boolean;
}
/** The seed manager's saved list. Written by the drawer's own mutations. */
export const seedsOptions = () => queryOptions({
  queryKey: serverKey('shell-queue/seeds'),
  queryFn: ({ signal }) => apiClient.get<{ seeds: SavedSeed[] }>('/seeds', { signal }),
});

/** The Directed Drift panel's live control snapshot (GET /storm/control). */
export interface DriftSnapshot {
  pos: Record<string, number>;
  captionWeights: Record<string, number>;
  seed: number;
  slot: number;
  pull: number;
  rate: number;
  jitter: number;
  targetCaption: string[];
}

/** Polled by the panel only while its stream is playing; no retry, because a
 *  dropped tick is replaced by the next one anyway. */
export const stormDriftOptions = (streamId: string) => queryOptions({
  queryKey: serverKey('shell-queue/storm-drift', { filters: { streamId } }),
  queryFn: ({ signal }) => apiClient.get<{ drift?: DriftSnapshot }>(
    `/generate/storm/control?streamId=${encodeURIComponent(streamId)}`, { signal }),
  staleTime: 0,
  retry: false,
});

export const songIdsOptions = () => queryOptions({
  queryKey: serverKey('shell-queue/song-ids', { auth: true }),
  queryFn: ({ signal }) => songApi.listIds({ signal }),
  retry: false,
});

/** The server queue's items. `maxAgeMs` 0 always asks the server (joining a
 *  request already in flight); a waiter that only needs "recent" passes the
 *  poll cadence and rides the projection's own reads. */
export const readAudioQueue = (maxAgeMs = 0) =>
  queryClient.query({ ...audioQueueItemsOptions(), staleTime: maxAgeMs });

/** One job's status. Every loop that follows a job polls at about the active
 *  cadence, so answers younger than a poll are shared rather than re-asked. */
export const readGenerationStatus = (jobId: string) =>
  queryClient.query({ ...generationStatusOptions(jobId), staleTime: ACTIVE_POLL_MS - 500 });

export const readWorkflowJob = (jobId: string) =>
  queryClient.query({ ...workflowOptions(jobId), retry: false, staleTime: 0 });

/** A finished render's song row, fresh: the queue reads it the moment the
 *  render is saved. Shares the canonical song key with every other reader. */
export const readSong = (id: string) =>
  queryClient.query({ ...songOptions(id), staleTime: 0 });

export const readSongIds = () =>
  queryClient.query({ ...songIdsOptions(), staleTime: 0 });
