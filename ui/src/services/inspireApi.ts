import { serverKey } from '@/queries/keys';
import { waitForSnapshot } from '@/queries/studios/wait';
import { apiClient, type ClientOptions, type ReadOptions } from './httpClient';
import { getAuthToken, SLOW_ROUTE_TIMEOUT_MS } from './axiosClient';
import { newId } from '@/utils/id';
// inspireApi.ts — Frontend API client for the inspire endpoint
//
// Provides submit + poll interface for the Insta-Gen inspire flow.

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
export async function submitInspire(params: InspireParams): Promise<string> {
  const data = await apiClient.post<{ jobId: string }>('/inspire', params);
  return data.jobId;
}

/** Poll inspire job status */
export async function pollInspireStatus(jobId: string, opts?: ReadOptions): Promise<InspireJobStatus> {
  return apiClient.get<InspireJobStatus>(`/inspire/status/${jobId}`, opts);
}

/** Cancel an inspire job */
export async function cancelInspire(jobId: string): Promise<void> {
  await apiClient.post(`/inspire/cancel/${jobId}`);
}

/**
 * Submit inspire and poll until complete.
 * Returns the InspireResult or throws on failure.
 */
export async function runInspireAndWait(
  params: InspireParams,
  onProgress?: (stage: string, progress: number) => void,
): Promise<InspireResult> {
  const jobId = await submitInspire(params);

  const status = await waitForSnapshot(serverKey('studios/inspire-job', { auth: getAuthToken() !== null, id: jobId }),
    signal => pollInspireStatus(jobId, { signal }),
    job => job.status === 'pending' || job.status === 'running',
    job => { if (job.stage && job.progress !== undefined) onProgress?.(job.stage, job.progress); }, { timeoutMs: 240_000 });
  if (status.status === 'succeeded' && status.result) return status.result;
  throw new Error(status.error || `Inspire ${status.status}`);
}

// ── External LLM lyric generation ──────────────────────────────────

export interface LlmInspireParams {
  provider: string;
  model?: string;
  genres: string[];
  subject: string;
  language?: string;
  systemPrompt?: string;  // Optional client-side prompt override
}

export interface LlmInspireResult {
  lyrics: string;
  caption: string;
  title?: string;
  bpm?: number;
  key?: string;
  timeSignature?: string;
  duration?: number;
  structured: boolean;  // true if LLM returned full JSON metadata
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
  opts?: ClientOptions,
): Promise<LlmInspireResult> {
  return apiClient.post<LlmInspireResult>('/inspire/llm', params, { timeout: SLOW_ROUTE_TIMEOUT_MS, ...opts });
}

/** Fetch available LLM providers for the Insta-Gen provider dropdown */
export async function fetchInspireProviders(opts?: ReadOptions): Promise<InspireProvider[]> {
  return apiClient.get<InspireProvider[]>('/inspire/llm/providers', opts);
}

/** Ask the LLM to generate a random song subject for the given genres */
export async function generateRandomSubject(
  params: { provider: string; model?: string; genres?: string[] },
  opts?: ClientOptions,
): Promise<string> {
  const data = await apiClient.post<{ subject?: string }>('/inspire/llm/subject', params, { timeout: SLOW_ROUTE_TIMEOUT_MS, ...opts });
  return data.subject || '';
}

// ── InstaGen system prompt CRUD ──────────────────────────────────

export interface InstagenPromptData {
  name: string;
  default_content: string;
  custom: string | null;
}

/** Fetch the current InstaGen system prompt (custom + default) */
export async function fetchInstagenPrompt(opts?: ReadOptions): Promise<InstagenPromptData> {
  return apiClient.get<InstagenPromptData>('/inspire/llm/prompt', opts);
}

/** Save a custom InstaGen system prompt */
export async function saveInstagenPrompt(value: string): Promise<void> {
  await apiClient.put('/inspire/llm/prompt', { value });
}

/** Reset the InstaGen system prompt to its default */
export async function resetInstagenPrompt(): Promise<void> {
  await apiClient.delete('/inspire/llm/prompt');
}

// Durable Insta-Gen workflows. The preview document is the authority for
// edits; approval names its exact revision.
import { studioWorkflowApi as workflowApi, followStudioJob as followJob } from '@/queries/studios/workflows';
export { followJob };

export interface InstaWorkflowInput {
  caption: string; genres: string[]; lyricMode: 'instrumental' | 'lyrics' | 'lyrics-ai';
  subject: string; randomSubject: boolean; provider: string; model: string;
  vocalLanguage: string; thinking: boolean; engineParams: Record<string, unknown>;
  expectedBackend: string; coResident: boolean; cacheLmCodes: boolean;
}

// The Insta-Gen panel displays every one of these failures itself (its inline
// error line covers the submit leg and the SSE job follow alike), so the JSON
// legs stay silent — one notifier per failure.
export const instaWorkflowApi = {
  preview: (input: InstaWorkflowInput, key = newId()) =>
    workflowApi.submit('insta-preview', key, input as unknown as Record<string, unknown>, { silent: true }),
  direct: (input: InstaWorkflowInput, key = newId()) =>
    workflowApi.submit('insta-direct', key, input as unknown as Record<string, unknown>, { silent: true }),
  approve: (documentId: string, revision: number, key = `approve:${documentId}:${revision}`) =>
    workflowApi.submit('insta-approve', key, { documentId, revision }, { silent: true }),
  get: workflowApi.get,
  list: workflowApi.list,
  cancel: workflowApi.cancel,
  getDocument: workflowApi.getDocument,
  updateDocument: workflowApi.updateDocument,
};
