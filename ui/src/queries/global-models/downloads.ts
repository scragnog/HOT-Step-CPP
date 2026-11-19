import type { QueryClient } from '@tanstack/react-query';
import type { DownloadJob } from '@/types';
import { serverKey } from '@/queries/keys';
import { bridgeEvents, type BridgeOptions } from '@/queries/sseBridge';
import { modelChangeKeys, invalidateModels } from './options';

export const downloadsKey = () => serverKey('global-models/downloads');
export function downloadFrame(data: string): { jobs: DownloadJob[] } | undefined {
  try {
    const frame = JSON.parse(data);
    if (!Array.isArray(frame.jobs) || !frame.jobs.every((job: DownloadJob) =>
      job && typeof job.jobId === 'string' && typeof job.fileId === 'string' && typeof job.filename === 'string' &&
      [job.bytesDownloaded, job.totalBytes, job.speed].every(value => Number.isFinite(value) && value >= 0) &&
      (job.error === undefined || typeof job.error === 'string') &&
      ['queued', 'downloading', 'paused', 'completed', 'failed', 'cancelled'].includes(job.status))) return;
    return frame;
  } catch { return; }
}

const completedByClient = new WeakMap<QueryClient, Set<string>>();

export function bridgeDownloads(client: QueryClient, url: string, onComplete?: () => void,
  subscribe?: BridgeOptions<{ jobs: DownloadJob[] }>['subscribe']) {
  const key = downloadsKey();
  let completedIds = completedByClient.get(client);
  if (!completedIds) completedByClient.set(client, completedIds = new Set());
  const seen = completedIds;
  return bridgeEvents<{ jobs: DownloadJob[] }>(client, {
    url, subscribe, parse: downloadFrame, reconnect: true,
    reconnectInvalidate: modelChangeKeys(),
    map: frame => ({
      update: [{ key, apply: () => frame.jobs }],
      // A per-client ledger also covers remount and frames with older completions.
      once: {
        id: `downloads-frame:${frame.jobs.filter(j => j.status === 'completed').map(j => j.jobId).sort().join(',')}`,
        run: () => {
          const completed = frame.jobs.filter(j => j.status === 'completed' && !seen.has(j.jobId));
          if (completed.length) {
            for (const job of completed) seen.add(job.jobId);
            void invalidateModels(client);
            onComplete?.();
          }
        },
      },
    }),
  });
}
