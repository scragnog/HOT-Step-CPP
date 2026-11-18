import React from 'react';
import ReactDOM from 'react-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Loader2, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { songApi } from '../../services/api';
import { apiClient, ApiError } from '../../services/httpClient';
import { SLOW_ROUTE_TIMEOUT_MS } from '../../services/axiosClient';
import { songOptions } from '../../queries/shell-queue/songs';
import { events, writeAndInvalidate } from '../../queries/writes';
import { measuredMetadata } from './measuredMetadata';

/** Measurement is a command. Opening the dialog never analyzes or saves audio. */
export const MeasureMetadataDialog: React.FC<{ songId: string; onClose: () => void }> = ({ songId, onClose }) => {
  const { t } = useTranslation();
  const client = useQueryClient();
  const { data, isPending, error: songError, refetch } = useQuery({ ...songOptions(songId), staleTime: 0 });
  const measure = useMutation({
    mutationFn: async () => {
      const { song } = (await refetch({ throwOnError: true })).data!;
      const audioUrl = song.audioUrl || song.audio_url;
      if (!audioUrl) throw new Error(t('metadata.measureNoAudio', 'This track has no audio to measure.'));
      const raw = await apiClient.post<{ bpm?: number; key?: string; scale?: string }>(
        '/analyze', { audioUrl }, { silent: true, timeout: SLOW_ROUTE_TIMEOUT_MS });
      const patch = measuredMetadata(raw);
      if (!patch) throw new Error(t('metadata.measureNoResult', 'No BPM or key could be measured. No metadata was saved.'));
      return { song, patch };
    },
  });
  const save = useMutation({
    mutationFn: async () => {
      const result = measure.data;
      if (!result) return;
      // A later manual edit must be reviewed, rather than silently replaced.
      const { song } = (await refetch({ throwOnError: true })).data!;
      if (song.bpm !== result.song.bpm || song.key_scale !== result.song.key_scale || song.audioUrl !== result.song.audioUrl) {
        throw new Error(t('metadata.measureChanged', 'This track changed during measurement. Measure again to review its current values.'));
      }
      await writeAndInvalidate(client, () => songApi.update(result.song.id, result.patch), events('song:written'));
    },
    onSuccess: onClose,
  });
  const busy = measure.isPending || save.isPending;
  const result = measure.data;
  const error = measure.error || (save.error instanceof ApiError ? null : save.error);
  const unknown = t('metadata.measureUnknown', 'Not available');

  return ReactDOM.createPortal(
    <div data-portal-layer className="fixed inset-0 z-[150] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4"
      onClick={e => { e.stopPropagation(); if (!busy) onClose(); }}>
      <section role="dialog" aria-modal="true" aria-labelledby="measure-metadata-title"
        className="w-full max-w-lg max-h-[88vh] overflow-y-auto rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-white/10 p-5 space-y-4 shadow-2xl"
        onClick={e => e.stopPropagation()} onKeyDown={e => { e.stopPropagation(); if (e.key === 'Escape' && !busy) onClose(); }}>
        <div className="flex items-center justify-between gap-3">
          <h2 id="measure-metadata-title" className="font-bold text-zinc-900 dark:text-white">{t('metadata.measure', 'Measure BPM and key')}</h2>
          <button type="button" autoFocus disabled={busy} onClick={onClose} aria-label={t('common.close', 'Close')}
            className="p-1 text-zinc-500 hover:text-zinc-800 dark:hover:text-white disabled:opacity-50"><X size={18} /></button>
        </div>
        <p className="text-sm text-zinc-700 dark:text-zinc-300">{result?.song.title || data?.song.title}</p>
        <p className="text-sm text-zinc-500">{t('metadata.measureHint', 'Essentia measures the original audio on the CPU. Review its estimates before replacing saved metadata. Planned tempo is unchanged.')}</p>
        {error && <p role="alert" className="text-sm text-red-600 dark:text-red-400">{error.message}</p>}
        {result && <table className="w-full text-sm text-left text-zinc-700 dark:text-zinc-300">
          <thead><tr><th></th><th className="p-2">{t('metadata.measureSaved', 'Saved')}</th><th className="p-2">{t('metadata.measureNew', 'Measured')}</th></tr></thead>
          <tbody>
            <tr><th className="p-2">{t('metadata.bpm', 'BPM')}</th><td className="p-2">{result.song.bpm || unknown}</td><td className="p-2">{result.patch.bpm ?? unknown}</td></tr>
            <tr><th className="p-2">{t('metadata.key', 'Key')}</th><td className="p-2">{result.song.key_scale || unknown}</td><td className="p-2">{result.patch.key_scale || unknown}</td></tr>
          </tbody>
        </table>}
        {result && <p className="text-xs text-zinc-500">{t('metadata.measureConfirmHint', 'Replace saved values also replaces manual edits. Unavailable measurements keep the saved value. Cancel saves nothing.')}</p>}
        <div className="flex flex-wrap justify-end gap-2">
          <button type="button" disabled={busy} onClick={onClose}
            className="px-3 py-2 rounded-lg text-sm text-zinc-600 dark:text-zinc-300 hover:bg-black/5 dark:hover:bg-white/5 disabled:opacity-50">{t('common.cancel', 'Cancel')}</button>
          <button type="button" disabled={busy || isPending || !!songError} onClick={() => { save.reset(); measure.mutate(); }}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm bg-zinc-100 dark:bg-zinc-800 text-zinc-800 dark:text-zinc-200 disabled:opacity-50">
            {measure.isPending && <Loader2 size={15} className="animate-spin" />}
            {measure.isPending ? t('metadata.measuring', 'Measuring...') : result ? t('metadata.measureAgain', 'Measure again') : t('metadata.measure', 'Measure BPM and key')}
          </button>
          {result && <button type="button" disabled={busy || measure.isError} onClick={() => save.mutate()}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-semibold bg-cyan-600 hover:bg-cyan-500 text-white disabled:opacity-50">
            {save.isPending && <Loader2 size={15} className="animate-spin" />}{t('metadata.measureReplace', 'Replace saved values')}
          </button>}
        </div>
      </section>
    </div>, document.body);
};
