import { useQuery, useQueryClient } from '@tanstack/react-query';
import { bridgeDownloads, downloadsKey } from '@/queries/global-models/downloads';
import { subscribeEventSource } from '@/services/sharedEventSource';
// useDownloadStream.ts — SSE hook for real-time download progress

import { useState, useEffect, useCallback, useRef } from 'react';
import { modelManagerApi } from '@/services/api';
import type { DownloadJob } from '@/types';

interface UseDownloadStreamOptions {
  /** Called when any job transitions to 'completed' */
  onComplete?: () => void;
}

export function useDownloadStream(opts?: UseDownloadStreamOptions) {
  const { data: jobs = [] } = useQuery({
    queryKey: downloadsKey(), queryFn: async (): Promise<DownloadJob[]> => [],
    enabled: false, initialData: [] as DownloadJob[],
  });
  const [isConnected, setIsConnected] = useState(false);
  const client = useQueryClient();
  const callback = useRef(opts?.onComplete);
  callback.current = opts?.onComplete;
  useEffect(() => {
    const url = modelManagerApi.downloadsStreamUrl;
    const detachBridge = bridgeDownloads(client, url, () => callback.current?.());
    const detachStatus = subscribeEventSource(url, {
      open: () => setIsConnected(true), error: () => setIsConnected(false),
    }, { reconnect: true });
    return () => { detachBridge(); detachStatus(); };
  }, [client]);

  const getJob = useCallback((jobId: string): DownloadJob | undefined => {
    return jobs.find(j => j.jobId === jobId);
  }, [jobs]);

  const getJobByFileId = useCallback((fileId: string): DownloadJob | undefined => {
    return jobs.find(j => j.fileId === fileId && (j.status === 'queued' || j.status === 'downloading' || j.status === 'paused'));
  }, [jobs]);

  const hasActiveDownloads = jobs.some(j => j.status === 'downloading' || j.status === 'queued');

  return {
    jobs,
    isConnected,
    getJob,
    getJobByFileId,
    hasActiveDownloads,
  };
}
