import { queryClient } from '@/queries/client';
import { separationProgressOptions, separationResultOptions } from '@/queries/studios/resources';
import { ACTIVE_POLL_MS } from '@/queries/contract';
import { waitForQuery } from '@/queries/wait';
import { apiClient, type ReadOptions } from './httpClient';
// supersepApi.ts — API client for SuperSep stem separation
//
// Communicates with the Node server /api/supersep endpoints.

export interface StemInfo {
  name: string;
  category: 'vocals' | 'instruments' | 'drums' | 'other';
  stem_type: string;
  n_frames: number;
  stage: number;
  index: number;
}

export interface SeparateResult {
  id: string;
}

export interface ProgressResult {
  status: 'running' | 'done' | 'failed' | 'cancelled';
  progress: number;
  message: string;
  n_stems?: number;
  error?: string;
}

export interface StemListResult {
  id: string;
  stems: StemInfo[];
}

// Mirrors SuperSepLevel in engine/src/supersep.h — keep the numbering in sync.
export type SeparationLevel = 0 | 1 | 2 | 3 | 4 | 5;

export const SEPARATION_LEVELS: { value: SeparationLevel; label: string; description: string }[] = [
  { value: 0, label: 'Basic', description: '6 stems: vocals, bass, drums, guitar, piano, other' },
  { value: 1, label: 'Vocal Split', description: '8 stems: + lead/backing vocals' },
  // The engine emits 14 at this level, but the aggregate Vocals and Drums that
  // feed the splits are hidden, so 12 reach the user (#134).
  { value: 2, label: 'Full', description: '12 stems: + 6 drum components' },
  // Level 3 ("Maximum") is retired — its HTDemucs "Other" refinement pass was
  // removed. The enum value still exists engine-side and now behaves as Full,
  // so anyone with 3 persisted keeps working; it is just not offered any more.
  {
    value: 4,
    label: '2-Stem (BS-RoFormer)',
    description: '2 stems: vocals from the 6-stem model, instrumental as mix − vocals '
      + '(exact complement, inherits the vocal model\'s errors inverted)',
  },
  {
    value: 5,
    label: '2-Stem (Leap Xe)',
    description: '2 stems, each from its own dedicated model — no mix-minus residual. '
      + 'Native GGML. Requires the Leap Xe pair from Model Manager.',
  },
];

const API_BASE = '/api/supersep';
/** Path prefix for apiClient calls (the client already adds `/api`). */
const P = '/supersep';

/** Start a separation job. Returns the job ID. Silent: CoverStudio's
 *  handleSeparate toasts the failure itself (its "Separation failed" line is
 *  the sole notifier). */
export async function startSeparation(
  audioUrl: string,
  level: SeparationLevel = 0,
): Promise<string> {
  const data = await apiClient.post<{ id: string }>(`${P}/separate?level=${level}`, { audioUrl }, { silent: true });
  return data.id;
}

/** Poll job progress. */
export async function getProgress(jobId: string, opts?: ReadOptions): Promise<ProgressResult> {
  return apiClient.get<ProgressResult>(`${P}/${jobId}/progress`, opts);
}

/** Get stem list (metadata only, no audio). */
export async function getStemList(jobId: string, opts?: ReadOptions): Promise<StemListResult> {
  return apiClient.get<StemListResult>(`${P}/${jobId}/result`, opts);
}

/** Get the URL for a specific stem's audio. */
export function getStemAudioUrl(jobId: string, stemIndex: number): string {
  return `${API_BASE}/${jobId}/stem/${stemIndex}`;
}

/** Recombine stems with volume/mute controls, returns WAV blob. */
export async function recombineStems(
  jobId: string,
  stemControls: { index: number; volume: number; muted: boolean }[],
): Promise<Blob> {
  return apiClient.post<Blob>(`${P}/recombine`, { id: jobId, stems: stemControls }, { responseType: 'blob' });
}

/** Release a completed/abandoned job's server-side resources. Fire-and-forget:
 * the route doesn't exist server-side yet (added separately, #132), so a 404
 * here is expected for now — swallow it with a console.warn rather than
 * throwing, since a missing release endpoint must never block the UI. */
export async function releaseSeparation(jobId: string): Promise<void> {
  try {
    // silent: this call's failures are expected and handled below, not toasted.
    await apiClient.post(`${P}/${jobId}/release`, undefined, { silent: true });
  } catch (err) {
    console.warn(`[supersepApi] release ${jobId} failed:`, err);
  }
}

/** Helper: poll until job completes, calling onProgress along the way. */
export async function waitForCompletion(
  jobId: string,
  onProgress?: (progress: number, message: string) => void,
  pollIntervalMs = ACTIVE_POLL_MS,
): Promise<StemListResult> {
  const p = await waitForQuery<ProgressResult>(separationProgressOptions(jobId),
    p => p.status === 'running', p => onProgress?.(p.progress, p.message), { pollMs: pollIntervalMs });
  if (p.status !== 'done') throw new Error(p.error || `Separation ${p.status}`);
  return queryClient.query(separationResultOptions(jobId));
}
