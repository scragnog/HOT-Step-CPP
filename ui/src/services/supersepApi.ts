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
      + 'Native GGML, no ONNX Runtime. Requires the Leap Xe pair from Model Manager.',
  },
];

const API_BASE = '/api/supersep';

/** Start a separation job. Returns the job ID. */
export async function startSeparation(
  audioUrl: string,
  level: SeparationLevel = 0,
): Promise<string> {
  const res = await fetch(`${API_BASE}/separate?level=${level}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ audioUrl }),
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    throw new Error(errBody.error || `Separation failed: ${res.status}`);
  }
  const data: SeparateResult = await res.json();
  return data.id;
}

/** Poll job progress. */
export async function getProgress(jobId: string): Promise<ProgressResult> {
  const res = await fetch(`${API_BASE}/${jobId}/progress`);
  if (!res.ok) throw new Error(`Progress fetch failed: ${res.status}`);
  return res.json();
}

/** Get stem list (metadata only, no audio). */
export async function getStemList(jobId: string): Promise<StemListResult> {
  const res = await fetch(`${API_BASE}/${jobId}/result`);
  if (!res.ok) throw new Error(`Result fetch failed: ${res.status}`);
  return res.json();
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
  const res = await fetch(`${API_BASE}/recombine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: jobId, stems: stemControls }),
  });
  if (!res.ok) throw new Error(`Recombine failed: ${res.status}`);
  return res.blob();
}

/** Helper: poll until job completes, calling onProgress along the way. */
export async function waitForCompletion(
  jobId: string,
  onProgress?: (progress: number, message: string) => void,
  pollIntervalMs = 500,
): Promise<StemListResult> {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const p = await getProgress(jobId);
        onProgress?.(p.progress, p.message);

        if (p.status === 'done') {
          const result = await getStemList(jobId);
          resolve(result);
        } else if (p.status === 'failed' || p.status === 'cancelled') {
          reject(new Error(p.error || `Separation ${p.status}`));
        } else {
          setTimeout(poll, pollIntervalMs);
        }
      } catch (err) {
        reject(err);
      }
    };
    poll();
  });
}
