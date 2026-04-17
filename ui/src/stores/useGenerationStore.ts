// useGenerationStore.ts — Generation state management
//
// Tracks active jobs, handles submission + polling + completion.
// Decoupled from UI — components just read state and call actions.

import { useState, useCallback, useRef } from 'react';
import { generateApi, songApi } from '../services/api';
import type { GenerationParams, GenerationJob, Song } from '../types';

interface GenerationStore {
  jobs: GenerationJob[];
  isGenerating: boolean;
  submit: (params: GenerationParams, token: string) => Promise<void>;
  cancel: (jobId: string) => Promise<void>;
  cancelAll: () => Promise<void>;
  clearCompleted: () => void;
  onSongCreated?: (song: Song) => void;
}

export function useGenerationStore(
  onSongCreated?: (song: Song) => void,
): GenerationStore {
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  const pollTimers = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map());

  const updateJob = useCallback((jobId: string, update: Partial<GenerationJob>) => {
    setJobs(prev => prev.map(j => j.jobId === jobId ? { ...j, ...update } : j));
  }, []);

  const pollJob = useCallback(async (jobId: string, _token: string) => {
    try {
      const status = await generateApi.status(jobId);
      updateJob(jobId, status);

      if (status.status === 'succeeded' || status.status === 'failed' || status.status === 'cancelled') {
        // Stop polling
        const timer = pollTimers.current.get(jobId);
        if (timer) clearInterval(timer);
        pollTimers.current.delete(jobId);

        // Fetch created songs if succeeded
        if (status.status === 'succeeded' && status.result?.songIds && onSongCreated) {
          for (const songId of status.result.songIds) {
            try {
              const { song } = await songApi.get(songId);
              onSongCreated(song);
            } catch (err) {
              console.error('[Generation] Failed to fetch created song:', err);
            }
          }
        }
      }
    } catch (err) {
      console.error('[Generation] Poll error:', err);
    }
  }, [updateJob, onSongCreated]);

  const submit = useCallback(async (params: GenerationParams, token: string) => {
    const { jobId, status } = await generateApi.submit(params, token);

    const newJob: GenerationJob = {
      jobId,
      status: status as GenerationJob['status'],
      stage: 'Queued',
      progress: 0,
    };

    setJobs(prev => [newJob, ...prev]);

    // Start polling
    const timer = setInterval(() => pollJob(jobId, token), 1000);
    pollTimers.current.set(jobId, timer);
  }, [pollJob]);

  const cancel = useCallback(async (jobId: string) => {
    await generateApi.cancel(jobId);
    updateJob(jobId, { status: 'cancelled', stage: 'Cancelled' });

    const timer = pollTimers.current.get(jobId);
    if (timer) clearInterval(timer);
    pollTimers.current.delete(jobId);
  }, [updateJob]);

  const cancelAll = useCallback(async () => {
    await generateApi.cancelAll();
    for (const [_id, timer] of pollTimers.current) {
      clearInterval(timer);
    }
    pollTimers.current.clear();
    setJobs(prev => prev.map(j =>
      ['pending', 'lm_running', 'synth_running'].includes(j.status)
        ? { ...j, status: 'cancelled' as const, stage: 'Cancelled' }
        : j
    ));
  }, []);

  const clearCompleted = useCallback(() => {
    setJobs(prev => prev.filter(j =>
      ['pending', 'lm_running', 'synth_running', 'saving'].includes(j.status)
    ));
  }, []);

  const isGenerating = jobs.some(j =>
    ['pending', 'lm_running', 'synth_running', 'saving'].includes(j.status)
  );

  return { jobs, isGenerating, submit, cancel, cancelAll, clearCompleted };
}
