import { followStudioJob as followJob, studioWorkflowApi as workflowApi } from '@/queries/studios/workflows';
import { newId } from '@/utils/id';
export type SourceRef = { kind: 'asset' | 'song'; id: string; expectedUrl: string };
interface CommonCapture { source: SourceRef; expectedBackend: string; engineParams: Record<string, unknown> }
export interface RepaintCapture extends CommonCapture {
  regionStart: number; regionEnd: number; lyrics: string; styleCaption: string; sourceName: string;
  repaintMode: 'conservative' | 'balanced' | 'aggressive'; crossfadeFrames: number;
}
export interface LayerCapture extends CommonCapture { trackName: string; buildModel: string; caption: string }

export interface RenderResult {
  request: Record<string, unknown>;
  audioIntentId: string;
  audio: { audioUrls?: string[]; songIds?: string[]; masteredAudioUrl?: string; noAdapterAudioUrl?: string; duration?: number } | null;
}

export async function submitStudioRender(kind: 'repaint-render' | 'layer-render',
  input: RepaintCapture | LayerCapture, idempotencyKey = newId()) {
  return workflowApi.submit(kind, idempotencyKey, input as unknown as Record<string, unknown>);
}

export async function followStudioRender(jobId: string,
  onStage?: (stage: string) => void, opts?: { signal?: AbortSignal }): Promise<RenderResult> {
  const status = await followJob(jobId, {
    onEvent: event => {
      if (event.type === 'audio') onStage?.('Generating...');
      if (event.type === 'request') onStage?.('Queued...');
    },
  }, opts);
  if (opts?.signal?.aborted) throw new DOMException('Observation cancelled', 'AbortError');
  const { job } = await workflowApi.get(jobId);
  if (status !== 'succeeded' || !job.result) throw new Error(job.error || `Workflow ${status}`);
  return job.result as RenderResult;
}
