import { coverLyricWorkflowApi, followCoverLyricJob } from '@/queries/cover-lyrics/workflows';
import { newId } from '@/utils/id';

export interface ScratchBrief {
  theme: string; style: string; structure: string; existingLyrics: string;
}

export function scratchRequest(lyricsSetId: number, brief: ScratchBrief, model: string) {
  const fields = Object.fromEntries(Object.entries(brief).map(([key, value]) => [key, value.trim()]));
  if (!fields.theme && !fields.existingLyrics) throw new Error('Enter a theme or some starter lyrics');
  return { lyricsSetId, ...fields, provider: 'ollama', model: model || undefined };
}

export async function followScratchLyrics(jobId: string, signal: AbortSignal,
  deps = { follow: followCoverLyricJob, get: coverLyricWorkflowApi.get }) {
  const status = await deps.follow(jobId, { onEvent: () => {} }, { signal });
  if (!status || signal.aborted) return false;
  const { job } = await deps.get(jobId, 0, { signal });
  if (status !== 'succeeded') throw new Error(job.error || `Workflow ${status}`);
  return true;
}

export const submitScratchLyrics = (input: ReturnType<typeof scratchRequest>) =>
  coverLyricWorkflowApi.submit('lyric-from-prompt', newId(), input);
