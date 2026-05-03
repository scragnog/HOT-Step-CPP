// useDownloadStream.ts — SSE hook for real-time download progress

import { useState, useEffect, useCallback, useRef } from 'react';
import { modelManagerApi } from '../../services/api';
import type { DownloadJob } from '../../types';

interface UseDownloadStreamOptions {
  /** Called when any job transitions to 'completed' */
  onComplete?: () => void;
}

export function useDownloadStream(opts?: UseDownloadStreamOptions) {
  const [jobs, setJobs] = useState<DownloadJob[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const completedRef = useRef(new Set<string>());
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const url = modelManagerApi.downloadsStreamUrl;
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onopen = () => setIsConnected(true);

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as { jobs: DownloadJob[] };
        setJobs(data.jobs);

        // Check for newly completed jobs
        if (opts?.onComplete) {
          for (const job of data.jobs) {
            if (job.status === 'completed' && !completedRef.current.has(job.jobId)) {
              completedRef.current.add(job.jobId);
              opts.onComplete();
            }
          }
        }
      } catch { /* ignore parse errors */ }
    };

    es.onerror = () => {
      setIsConnected(false);
      // EventSource auto-reconnects
    };

    return () => {
      es.close();
      eventSourceRef.current = null;
    };
  }, [opts?.onComplete]);

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
