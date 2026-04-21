// useGenerationStore.ts — Generation state management
//
// Tracks active jobs, handles submission + polling + completion.
// Decoupled from UI — components just read state and call actions.

import { useState, useCallback, useRef, useEffect } from 'react';
import { generateApi, songApi } from '../services/api';
import type { GenerationParams, GenerationJob, Song } from '../types';

interface GenerationStore {
  jobs: GenerationJob[];
  isGenerating: boolean;
  submit: (params: GenerationParams, token: string) => Promise<void>;
  cancel: (jobId: string) => Promise<void>;
  cancelAll: () => Promise<void>;
  clearCompleted: () => void;
  removeJob: (jobId: string) => void;
  onSongCreated?: (song: Song) => void;
}

const MAX_POLL_ERRORS = 5;
const MAX_POLL_DURATION_MS = 30 * 60 * 1000; // 30 minutes

export function useGenerationStore(
  onSongCreated?: (song: Song) => void,
): GenerationStore {
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  const pollTimers = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map());
  const pollErrors = useRef<Map<string, number>>(new Map());
  const pollStart = useRef<Map<string, number>>(new Map());

  // Keep latest onSongCreated in a ref to avoid stale closures in setInterval
  const onSongCreatedRef = useRef(onSongCreated);
  useEffect(() => { onSongCreatedRef.current = onSongCreated; }, [onSongCreated]);

  const updateJob = useCallback((jobId: string, update: Partial<GenerationJob>) => {
    setJobs(prev => prev.map(j => j.jobId === jobId ? { ...j, ...update } : j));
  }, []);

  const stopPolling = useCallback((jobId: string) => {
    const timer = pollTimers.current.get(jobId);
    if (timer) clearInterval(timer);
    pollTimers.current.delete(jobId);
    pollErrors.current.delete(jobId);
    pollStart.current.delete(jobId);
  }, []);

  const pollJob = useCallback(async (jobId: string) => {
    // Safety: check if we've exceeded the maximum poll duration
    const startTime = pollStart.current.get(jobId) ?? Date.now();
    if (Date.now() - startTime > MAX_POLL_DURATION_MS) {
      console.error(`[Generation] Job ${jobId} exceeded max poll duration, marking failed`);
      updateJob(jobId, { status: 'failed', stage: 'Timed out', error: 'Generation timed out' });
      stopPolling(jobId);
      return;
    }

    try {
      const status = await generateApi.status(jobId);
      updateJob(jobId, status);
      // Reset error count on success
      pollErrors.current.set(jobId, 0);

      if (status.status === 'succeeded' || status.status === 'failed' || status.status === 'cancelled') {
        stopPolling(jobId);

        // Fetch created songs if succeeded
        if (status.status === 'succeeded' && status.result?.songIds && onSongCreatedRef.current) {
          for (const songId of status.result.songIds) {
            try {
              const { song } = await songApi.get(songId);
              onSongCreatedRef.current(song);
            } catch (err) {
              console.error('[Generation] Failed to fetch created song:', err);
            }
          }
        }
      }
    } catch (err) {
      console.error('[Generation] Poll error:', err);
      const errorCount = (pollErrors.current.get(jobId) ?? 0) + 1;
      pollErrors.current.set(jobId, errorCount);

      if (errorCount >= MAX_POLL_ERRORS) {
        console.error(`[Generation] Job ${jobId}: ${errorCount} consecutive poll errors, marking failed`);
        updateJob(jobId, { status: 'failed', stage: 'Connection lost', error: 'Lost connection to server' });
        stopPolling(jobId);
      }
    }
  }, [updateJob, stopPolling]);

  // Keep pollJob in a ref so setInterval always calls the latest version
  const pollJobRef = useRef(pollJob);
  useEffect(() => { pollJobRef.current = pollJob; }, [pollJob]);

  const submit = useCallback(async (params: GenerationParams, token: string) => {
    const { jobId, status } = await generateApi.submit(params, token);

    const newJob: GenerationJob = {
      jobId,
      status: status as GenerationJob['status'],
      stage: 'Queued',
      progress: 0,
    };

    setJobs(prev => [newJob, ...prev]);
    pollStart.current.set(jobId, Date.now());

    // Start polling — uses ref so interval always calls latest pollJob
    const timer = setInterval(() => pollJobRef.current(jobId), 1000);
    pollTimers.current.set(jobId, timer);
  }, []);

  const cancel = useCallback(async (jobId: string) => {
    await generateApi.cancel(jobId);
    updateJob(jobId, { status: 'cancelled', stage: 'Cancelled' });
    stopPolling(jobId);
  }, [updateJob, stopPolling]);

  const cancelAll = useCallback(async () => {
    await generateApi.cancelAll();
    for (const [id] of pollTimers.current) {
      stopPolling(id);
    }
    setJobs(prev => prev.map(j =>
      ['pending', 'lm_running', 'synth_running'].includes(j.status)
        ? { ...j, status: 'cancelled' as const, stage: 'Cancelled' }
        : j
    ));
  }, [stopPolling]);

  const clearCompleted = useCallback(() => {
    setJobs(prev => prev.filter(j =>
      ['pending', 'lm_running', 'synth_running', 'saving'].includes(j.status)
    ));
  }, []);

  const removeJob = useCallback((jobId: string) => {
    setJobs(prev => prev.filter(j => j.jobId !== jobId));
    stopPolling(jobId);
  }, [stopPolling]);

  const isGenerating = jobs.some(j =>
    ['pending', 'lm_running', 'synth_running', 'saving'].includes(j.status)
  );

  // Auto-clear completed jobs after 5 seconds
  useEffect(() => {
    const hasCompleted = jobs.some(j => ['succeeded', 'failed', 'cancelled'].includes(j.status));
    if (hasCompleted) {
      const timer = setTimeout(() => {
        clearCompleted();
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [jobs, clearCompleted]);

  return { jobs, isGenerating, submit, cancel, cancelAll, clearCompleted, removeJob };
}
