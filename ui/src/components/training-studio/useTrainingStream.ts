// useTrainingStream.ts — EventSource hook for training job progress
//
// The server replays the whole event buffer on connect, so every handler in
// the store must be idempotent (sample rows are replaced by id, never
// appended). EventSource reconnects on its own; we only close on unmount.

import { useEffect } from 'react';
import { useTrainingStore } from '../../stores/trainingStore';
import { getJob, jobStreamUrl, type TrainingStreamEvent } from '../../services/trainingApi';
import { subscribeEventSource } from '../../services/sharedEventSource';

export function useTrainingStream(jobId: string | null) {
  const applyStreamEvent = useTrainingStore(s => s.applyStreamEvent);

  useEffect(() => {
    if (!jobId) return;
    return subscribeEventSource(jobStreamUrl(jobId), {
      message: (data) => {
        try { applyStreamEvent(JSON.parse(data) as TrainingStreamEvent); } catch { /* ignore malformed frame */ }
      },
      error: () => {
        // EventSource auto-reconnects and the buffer replay makes that safe. But
        // if the server restarted, the in-memory job is gone (stream 404s forever)
        // and no terminal event will ever arrive — the UI would show "running"
        // and keep Start disabled indefinitely. Poll once per error: a live job
        // is a no-op; a terminal or vanished job synthesizes its final status.
        void getJob(jobId).then(
          (job) => {
            if (job.status !== 'queued' && job.status !== 'running') {
              applyStreamEvent({ type: 'status', status: job.status, ...(job.error ? { error: job.error } : {}) });
            }
          },
          () => applyStreamEvent({ type: 'status', status: 'failed', error: 'Job lost — the server restarted' }),
        );
      },
    });
  }, [jobId, applyStreamEvent]);
}
