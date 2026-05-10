// inspireApi.ts — Frontend API client for the inspire endpoint
//
// Provides submit + poll interface for the Insta-Gen inspire flow.

const BASE = '/api/inspire';

export interface InspireParams {
  caption: string;
  subject?: string;
  lyrics?: string;
  bpm?: number;
  duration?: number;
  keyScale?: string;
  timeSignature?: string;
  vocalLanguage?: string;
  // LM settings (optional overrides)
  lmModel?: string;
  lmTemperature?: number;
  lmCfgScale?: number;
  lmTopP?: number;
  useCotCaption?: boolean;
}

export interface InspireResult {
  caption: string;
  lyrics: string;
  title?: string;
  bpm: number;
  duration: number;
  keyScale: string;
  timeSignature: string;
  vocalLanguage: string;
}

export interface InspireJobStatus {
  jobId: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';
  stage?: string;
  progress?: number;
  result?: InspireResult;
  error?: string;
}

/** Submit an inspire request, returns job ID */
export async function submitInspire(params: InspireParams, token?: string): Promise<string> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(BASE, {
    method: 'POST',
    headers,
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(err.error || `Inspire failed (${res.status})`);
  }

  const data = await res.json();
  return data.jobId;
}

/** Poll inspire job status */
export async function pollInspireStatus(jobId: string, token?: string): Promise<InspireJobStatus> {
  const headers: Record<string, string> = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}/status/${jobId}`, { headers });
  if (!res.ok) {
    throw new Error(`Poll failed (${res.status})`);
  }
  return res.json();
}

/** Cancel an inspire job */
export async function cancelInspire(jobId: string, token?: string): Promise<void> {
  const headers: Record<string, string> = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;

  await fetch(`${BASE}/cancel/${jobId}`, { method: 'POST', headers });
}

/**
 * Submit inspire and poll until complete.
 * Returns the InspireResult or throws on failure.
 */
export async function runInspireAndWait(
  params: InspireParams,
  token?: string,
  onProgress?: (stage: string, progress: number) => void,
): Promise<InspireResult> {
  const jobId = await submitInspire(params, token);

  const POLL_MS = 800;
  const MAX_POLLS = 300; // ~4 minutes

  for (let i = 0; i < MAX_POLLS; i++) {
    await new Promise(r => setTimeout(r, POLL_MS));
    const status = await pollInspireStatus(jobId, token);

    if (onProgress && status.stage && status.progress !== undefined) {
      onProgress(status.stage, status.progress);
    }

    if (status.status === 'succeeded' && status.result) {
      return status.result;
    }
    if (status.status === 'failed') {
      throw new Error(status.error || 'Inspire failed');
    }
    if (status.status === 'cancelled') {
      throw new Error('Inspire was cancelled');
    }
  }

  throw new Error('Inspire timed out');
}

// ── External LLM lyric generation ──────────────────────────────────

export interface LlmInspireParams {
  provider: string;
  model?: string;
  genres: string[];
  subject: string;
  language?: string;
}

export interface LlmInspireResult {
  lyrics: string;
  caption: string;
  title?: string;
  provider: string;
  model: string;
}

export interface InspireProvider {
  id: string;
  name: string;
  available: boolean;
  models: string[];
  default_model: string;
}

/** Generate lyrics via an external LLM provider (synchronous call) */
export async function runLlmInspire(
  params: LlmInspireParams,
  token?: string,
): Promise<LlmInspireResult> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}/llm`, {
    method: 'POST',
    headers,
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(err.error || `LLM inspire failed (${res.status})`);
  }

  return res.json();
}

/** Fetch available LLM providers for the Insta-Gen provider dropdown */
export async function fetchInspireProviders(): Promise<InspireProvider[]> {
  const res = await fetch(`${BASE}/llm/providers`);
  if (!res.ok) throw new Error(`Failed to fetch providers: ${res.statusText}`);
  return res.json();
}

/** Ask the LLM to generate a random song subject for the given genres */
export async function generateRandomSubject(
  params: { provider: string; model?: string; genres?: string[] },
  token?: string,
): Promise<string> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}/llm/subject`, {
    method: 'POST',
    headers,
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(err.error || `Random subject generation failed (${res.status})`);
  }

  const data = await res.json();
  return data.subject || '';
}
