import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Archive, FolderOpen } from 'lucide-react';
import { workspaceArchiveApi, type ArchiveCategoryId, type ArchiveChoice } from '../../services/workspaceArchiveApi';
import { workspaceArchiveImportJobOptions, workspaceArchiveJobOptions, workspaceArchivePreviewOptions } from '../../queries/shell-queue/settings';
import { queryManager } from '../../queries/invalidation';
import { FileBrowserModal } from '../shared/FileBrowserModal';
import { ParamLabel } from '../shared/ParamLabel';
import { Toggle } from './SettingsPrimitives';
import { WorkspaceArchiveImport } from './WorkspaceArchiveImport';

const CATEGORIES: ArchiveCategoryId[] = ['library', 'training', 'adapters', 'models', 'secrets'];
const gb = (n: number) => `${(n / 1e9).toFixed(2)} GB`;

/** Settings > Storage & Data: export the workspace to a zip, or import one into a new data folder. */
export function WorkspaceArchivePanel() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [destination, setDestination] = useState('');
  const [checked, setChecked] = useState('');
  const [choice, setChoice] = useState<ArchiveChoice | null>(null);
  const [browsing, setBrowsing] = useState(false);
  const [error, setError] = useState('');

  const jobQuery = useQuery(workspaceArchiveJobOptions());
  const job = jobQuery.data?.job ?? null;
  const running = job?.state === 'holding' || job?.state === 'running';
  // The server answers 409 to an export while an import runs; disable it here too.
  const importing = useQuery(workspaceArchiveImportJobOptions()).data?.job?.state === 'running';
  const previewQuery = useQuery(workspaceArchivePreviewOptions(checked));
  const preview = previewQuery.data ?? null;
  const picked = choice ?? preview?.defaults ?? null;

  const refresh = () => {
    void queryManager.invalidate(queryClient, workspaceArchiveJobOptions().queryKey);
    void queryManager.invalidate(queryClient, workspaceArchivePreviewOptions(checked).queryKey);
  };
  const fail = (e: unknown) => setError(e instanceof Error ? e.message : String(e));
  // Both commands are silent at the transport: the inline line is their only notifier.
  const startMut = useMutation({
    mutationFn: () => workspaceArchiveApi.start(destination.trim(), picked!),
    onMutate: () => setError(''),
    onError: fail,
    onSettled: refresh,
  });
  const cancelMut = useMutation({
    mutationFn: () => workspaceArchiveApi.cancel(),
    onMutate: () => setError(''),
    onError: fail,
    onSettled: refresh,
  });

  const setDest = (v: string) => { setDestination(v); setChecked(''); };
  const check = () => { const d = destination.trim(); if (d === checked) void previewQuery.refetch(); else setChecked(d); };
  const on = (id: ArchiveCategoryId) => id === 'library' || !!picked?.[id as keyof ArchiveChoice];
  const bytesFor = (id: ArchiveCategoryId) => preview?.categories[id].bytes ?? 0;
  // Selected categories, plus the library again: the database is snapshotted before it is zipped.
  const needed = preview ? CATEGORIES.filter(on).reduce((sum, id) => sum + bytesFor(id), 0) + bytesFor('library') : 0;
  const dest = preview?.destination ?? null;
  const free = dest?.freeBytes ?? null;
  const tooSmall = free !== null && needed > free;
  const problems = [...(preview?.blockers ?? []), ...(dest?.errors ?? []), ...(tooSmall ? [t('settings.workspaceArchive.noSpace')] : [])];
  const canStart = !running && !importing && !startMut.isPending && !previewQuery.isFetching && !previewQuery.isError && !!picked && !!dest && checked === destination.trim() && problems.length === 0;
  const pct = job?.progress && job.progress.bytesTotal > 0 ? Math.min(100, Math.round(100 * job.progress.bytesDone / job.progress.bytesTotal)) : 0;

  return <section className="settings-section">
    <div className="settings-section-header"><Archive size={16} className="settings-section-icon" /><span className="settings-section-title">{t('settings.workspaceArchive.title')}</span></div>
    <div className="settings-grid text-sm text-zinc-700 dark:text-zinc-300">
      <p>{t('settings.workspaceArchive.intro')}</p>
      {error && <p role="alert" className="text-red-600 dark:text-red-400">{error}</p>}
      {jobQuery.error && <p role="alert" className="text-red-600 dark:text-red-400">{jobQuery.error.message}</p>}
      {previewQuery.error && <p role="alert" className="text-red-600 dark:text-red-400">{previewQuery.error.message}</p>}

      {running && job && <div className="space-y-2" role="status">
        <p>{t('settings.workspaceArchive.held')}</p>
        <p className="break-all font-mono">{job.outFile}</p>
        {job.state === 'holding'
          ? <p>{t('settings.workspaceArchive.holding')}</p>
          : <>
            <progress className="w-full" max={100} value={pct} aria-label={t('settings.workspaceArchive.progress')} />
            <p>{t(`settings.workspaceArchive.phase.${job.progress?.phase ?? 'database'}`)}{job.progress ? ` · ${gb(job.progress.bytesDone)} / ${gb(job.progress.bytesTotal)} · ${job.progress.filesDone} / ${job.progress.filesTotal}` : ''}</p>
          </>}
        <button type="button" className="settings-save-btn" disabled={cancelMut.isPending} onClick={() => cancelMut.mutate()}>{t('settings.workspaceArchive.cancel')}</button>
      </div>}
      {job?.state === 'done' && <p role="status" className="break-all">{t('settings.workspaceArchive.done', { path: job.outFile, size: gb(job.bytes ?? 0), skipped: job.skipped, externals: job.externals })}</p>}
      {job?.state === 'cancelled' && <p role="status">{t('settings.workspaceArchive.cancelled')}</p>}
      {job?.state === 'failed' && <div role="alert" className="text-red-600 dark:text-red-400">
        <p>{t('settings.workspaceArchive.failed', { error: job.error ?? '' })}</p>
        {job.blockers.map((b, i) => <p key={i}>{b}</p>)}
      </div>}

      <div className="setting-row">
        <div className="setting-info"><ParamLabel className="setting-label" label={t('settings.workspaceArchive.destination')} info={t('settings.workspaceArchive.destinationInfo')} /></div>
        <div className="settings-path-actions">
          <input className="env-input" aria-label={t('settings.workspaceArchive.destination')} disabled={running} value={destination} onChange={e => setDest(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && destination.trim()) check(); }} />
          <button type="button" className="env-browse-btn" disabled={running} title={t('settings.workspaceArchive.browse')} aria-label={t('settings.workspaceArchive.browse')} onClick={() => setBrowsing(true)}><FolderOpen size={16} /></button>
          <button type="button" className="settings-save-btn" disabled={running || !destination.trim()} onClick={check}>{t('settings.workspaceArchive.check')}</button>
        </div>
      </div>

      {preview && picked && <>
        {CATEGORIES.map(id => <div className="setting-row" key={id}>
          <div className="setting-info">
            <ParamLabel className="setting-label" label={t(`settings.workspaceArchive.cat.${id}`)} info={t(`settings.workspaceArchive.cat.${id}Info`)} />
            <p className="setting-description">{gb(bytesFor(id))} · {t('settings.workspaceArchive.files', { count: preview.categories[id].files })}</p>
          </div>
          <Toggle id={`archive-${id}`} checked={on(id)} disabled={running || id === 'library'} onChange={v => setChoice({ ...picked, [id]: v })} />
        </div>)}
        <p>{t('settings.workspaceArchive.needed', { size: gb(needed) })}{dest && ` · ${t('settings.workspaceArchive.free', { size: free === null ? t('settings.workspaceArchive.unknown') : gb(free) })}`}</p>
        {preview.skipped.length + preview.skippedNotListed > 0 && <p>{t('settings.workspaceArchive.skipped', { count: preview.skipped.length + preview.skippedNotListed })}</p>}
        {problems.map((p, i) => <p key={i} role="alert" className="text-red-600 dark:text-red-400">{p}</p>)}
        <button type="button" className="settings-save-btn" disabled={!canStart} onClick={() => startMut.mutate()}>{t('settings.workspaceArchive.start')}</button>
      </>}
      <WorkspaceArchiveImport exporting={running} />
    </div>
    <FileBrowserModal open={browsing} mode="folder" startPath={destination} title={t('settings.workspaceArchive.browse')}
      onClose={() => setBrowsing(false)} onSelect={p => { setBrowsing(false); setDestination(p); setChecked(p.trim()); }} />
  </section>;
}
