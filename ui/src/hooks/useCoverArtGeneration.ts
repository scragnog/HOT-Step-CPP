// useCoverArtGeneration.ts — submit a cover-art generation and track its job,
// for CoverArtPromptModal (#67). The follow (the QueryObserver poll of the
// job) stays in the component: it closes the modal on success, which is the
// component's state. This hook owns only the submit and what the submit knows:
// the job id to follow, the inline error, and whether the modal is locked.
//
// The submit is a *silent* mutation: its failures need context the global
// toast cannot carry — a 503 means the weights are absent, not that anything
// went wrong, and "cover art not installed" on its own sent people hunting
// through Settings and the Model Manager for a download that lives in the
// Post-Processing menu instead (issue #108) — so it opts out of the
// interceptor's notification and reports inline.

import { useCallback, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import type { Song } from '@/types';
import { ApiError, HttpStatus, apiClient } from '@/services/httpClient';

/** Build the metadata payload used for both preview and generation. */
export function songPromptInputs(song: Song) {
  const params: any = song.generationParams || (song as any).generation_params || {};
  return {
    title: song.title || '',
    style: song.style || params?.style || '',
    lyrics: song.lyrics || params?.lyrics || '',
    subject: (song as any).cover_art_subject || params?.coverArtSubject || params?.subject || '',
  };
}

export function useCoverArtGeneration(song: Song | null, prompt: string) {
  // The generation being followed. Clearing it stops the poll.
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState('');

  const generateMut = useMutation({
    mutationFn: async () => {
      if (!song) throw new Error('No song selected');
      return apiClient.post<{ jobId: string }>('/cover-art/generate', {
        songId: song.id,
        ...songPromptInputs(song),
        prompt: prompt.trim(),
      }, { silent: true });
    },
    onSuccess: d => setJobId(d.jobId),
    onError: err => {
      if (err instanceof ApiError && err.status === HttpStatus.serviceUnavailable) {
        const missing: string[] = Array.isArray((err.body as any)?.missingFiles) ? (err.body as any).missingFiles : [];
        setError(
          'Cover art models are not downloaded yet. Open the Cover Art section '
          + 'of the Post-Processing menu in the top bar and download a model there.'
          + (missing.length ? ` Missing: ${missing.join(', ')}.` : ''),
        );
      } else {
        setError(err instanceof Error ? err.message : 'Failed to start generation');
      }
    },
  });

  // Submit in flight, or a generation being followed — either locks the modal.
  const generating = generateMut.isPending || jobId !== null;

  /** Fire the submit with a fresh error line — a retry must not show the old failure. */
  const generate = useCallback(() => {
    if (!song || !prompt.trim()) return;
    setError('');
    generateMut.mutate();
  }, [song, prompt, generateMut]);

  const clearError = useCallback(() => setError(''), []);
  /** Stop following the current job (clearing it ends the poll). */
  const resetJob = useCallback(() => setJobId(null), []);
  /** The followed job ended in failure: stop following and report its message. */
  const fail = useCallback((message: string) => {
    setJobId(null);
    setError(message);
  }, []);

  return { jobId, error, generating, generate, clearError, resetJob, fail } as const;
}
