import { followJob, workflowApi } from './workflowApi';
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

export async function submitStudioRender(token: string, kind: 'repaint-render' | 'layer-render',
  input: RepaintCapture | LayerCapture, idempotencyKey = crypto.randomUUID()) {
  return workflowApi.submit(token, kind, idempotencyKey, input as unknown as Record<string, unknown>);
}

export async function followStudioRender(token: string, jobId: string,
  onStage?: (stage: string) => void): Promise<RenderResult> {
  const status = await followJob(token, jobId, {
    onEvent: event => {
      if (event.type === 'audio') onStage?.('Generating...');
      if (event.type === 'request') onStage?.('Queued...');
    },
  });
  const { job } = await workflowApi.get(token, jobId);
  if (status !== 'succeeded' || !job.result) throw new Error(job.error || `Workflow ${status}`);
  return job.result as RenderResult;
}
