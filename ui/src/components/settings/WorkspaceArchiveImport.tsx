import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { FolderOpen } from 'lucide-react';
import { workspaceArchiveApi, type ArchiveImportPreview } from '../../services/workspaceArchiveApi';
import { dataFolderOptions, workspaceArchiveImportJobOptions } from '../../queries/shell-queue/settings';
import { queryManager } from '../../queries/invalidation';
import { ConfirmDialog } from '../shared/ConfirmDialog';
import { FileBrowserModal } from '../shared/FileBrowserModal';
import { ParamLabel } from '../shared/ParamLabel';

const gb = (n: number) => `${(n / 1e9).toFixed(2)} GB`;
// The accepted import's target survives the restart's page reload here, so the outcome can be checked afterwards.
const PENDING_KEY = 'hotstep:workspace-import-pending';
const norm = (p: string) => p.replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase();
const parentDir = (p: string) => p.replace(/[\\/]+[^\\/]*$/, '');
const readPending = (): string => { try { return sessionStorage.getItem(PENDING_KEY) ?? ''; } catch { return ''; } };
const writePending = (target: string) => { try { if (target) sessionStorage.setItem(PENDING_KEY, target); else sessionStorage.removeItem(PENDING_KEY); } catch { /* storage unavailable: no outcome check */ } };

/** Import half of the workspace archive: pick a zip and a new folder, preview, confirm, restart. */
export function WorkspaceArchiveImport({ exporting }: { exporting: boolean }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [zip, setZip] = useState('');
  const [target, setTarget] = useState('');
  const [preview, setPreview] = useState<ArchiveImportPreview | null>(null);
  const [browse, setBrowse] = useState<'zip' | 'target' | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState('');
  const [errors, setErrors] = useState<string[]>([]);
  const [pending, setPending] = useState(readPending);
  const [forceCheck, setForceCheck] = useState(false);
  // Set at start until the poll first shows the job, so the stale pre-start `null` is not read as a restart.
  const [unseen, setUnseen] = useState(false);

  const jobQuery = useQuery(workspaceArchiveImportJobOptions());
  const job = jobQuery.data?.job ?? null;
  const running = job?.state === 'running';
  const waiting = running && job.progress?.phase === 'waiting';

  // The server restarts about 300 ms after `done`; tell the app shell so it shows its restart overlay.
  // The poll can miss that window, so the outcome below does not depend on seeing `done`.
  const done = job?.state === 'done';
  useEffect(() => { if (done) window.dispatchEvent(new CustomEvent('hotstep:restarting')); }, [done]);
  // A failed or cancelled job means nothing was switched. A new server has no job (it lives in memory).
  useEffect(() => { if (pending && (job?.state === 'failed' || job?.state === 'cancelled')) { writePending(''); setPending(''); } }, [pending, job?.state]);
  useEffect(() => { if (job) setUnseen(false); }, [job]);
  const restarted = !!pending && !unseen && jobQuery.isSuccess && !job;
  const locationQuery = useQuery({ ...dataFolderOptions(), enabled: restarted || (!!pending && done && forceCheck), refetchInterval: q => (q.state.data ? false : 2000) });
  const where = locationQuery.data;
  const switched = !!where && !!pending && norm(where.dataRoot ?? '') === norm(pending);
  const reportPath = pending ? `${pending}${/[\\/]$/.test(pending) ? '' : '/'}import-report.json` : '';
  const dismiss = () => { writePending(''); setPending(''); setForceCheck(false); };

  const refresh = () => void queryManager.invalidate(queryClient, workspaceArchiveImportJobOptions().queryKey);
  const fail = (e: unknown) => {
    setError(e instanceof Error ? e.message : String(e));
    const body = (e as { body?: { errors?: string[] } } | null)?.body;
    setErrors(body?.errors ?? []);
  };
  const clear = () => { setPreview(null); setError(''); setErrors([]); };

  // Every command is silent at the transport: the inline lines are their only notifier.
  const previewMut = useMutation({
    mutationFn: () => workspaceArchiveApi.importPreview(zip.trim(), target.trim()),
    onMutate: clear,
    onSuccess: setPreview,
    onError: fail,
  });
  const startMut = useMutation({
    mutationFn: () => workspaceArchiveApi.importStart(zip.trim(), target.trim()),
    onMutate: () => { setError(''); setErrors([]); setConfirming(false); },
    onError: fail,
    onSuccess: ({ job: started }) => { writePending(started.target); setPending(started.target); setForceCheck(false); setUnseen(true); },
    onSettled: refresh,
  });
  const cancelMut = useMutation({
    mutationFn: () => workspaceArchiveApi.importCancel(),
    onMutate: () => setError(''),
    onError: fail,
    onSettled: refresh,
  });

  const busy = running || done || exporting || previewMut.isPending || startMut.isPending;
  const problems = [...(preview?.blockers ?? []), ...(preview?.target?.errors ?? [])];
  const need = preview?.bytes ?? 0;
  const free = preview?.target?.freeBytes ?? null;
  const tooSmall = free !== null && need > free;
  const canStart = !busy && !!preview && !!preview.target && problems.length === 0 && !tooSmall;
  const pct = job?.progress && job.progress.bytesTotal > 0 ? Math.min(100, Math.round(100 * job.progress.bytesDone / job.progress.bytesTotal)) : 0;
  const date = (iso: string) => { const d = new Date(iso); return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(); };

  return <div className="space-y-2">
    <p className="font-semibold">{t('settings.workspaceArchive.import.title')}</p>
    <p>{t('settings.workspaceArchive.import.intro')}</p>
    {exporting && <p>{t('settings.workspaceArchive.import.exporting')}</p>}
    {error && <div role="alert" className="text-red-600 dark:text-red-400"><p>{error}</p>{errors.filter(e => e !== error).map((e, i) => <p key={i}>{e}</p>)}</div>}
    {jobQuery.error && !done && <p role="alert" className="text-red-600 dark:text-red-400">{jobQuery.error.message}</p>}

    {running && job && <div className="space-y-2" role="status">
      <p className="break-all">{job.zipFile} → {job.target}</p>
      {waiting
        ? <><p>{t('settings.workspaceArchive.import.waiting')}</p>{job.progress?.blockers?.map((b, i) => <p key={i}>{b}</p>)}</>
        : <>
          <progress className="w-full" max={100} value={pct} aria-label={t('settings.workspaceArchive.import.progress')} />
          <p>{t(`settings.workspaceArchive.import.phase.${job.progress?.phase ?? 'checking'}`)}{job.progress && job.progress.bytesTotal > 0 ? ` · ${gb(job.progress.bytesDone)} / ${gb(job.progress.bytesTotal)} · ${job.progress.filesDone} / ${job.progress.filesTotal}` : ''}</p>
        </>}
      <button type="button" className="settings-save-btn" disabled={cancelMut.isPending} onClick={() => cancelMut.mutate()}>{t('settings.workspaceArchive.import.cancel')}</button>
      <p>{t('settings.workspaceArchive.import.cancelNote')}</p>
    </div>}
    {done && job && !forceCheck && <div role="status" className="space-y-1">
      <p>{t('settings.workspaceArchive.import.restarting', { target: job.target })}</p>
      <p className="break-all">{t('settings.workspaceArchive.import.reportAt', { path: reportPath })}</p>
      <button type="button" className="settings-save-btn" onClick={() => setForceCheck(true)}>{t('settings.workspaceArchive.import.checkNow')}</button>
    </div>}
    {pending && !running && (!done || forceCheck) && !where && <p role="status">{t('settings.workspaceArchive.import.checking')}</p>}
    {pending && !running && where && <div role="status" className="space-y-1">
      <p>{t(switched ? 'settings.workspaceArchive.import.switched' : 'settings.workspaceArchive.import.notSwitched', { target: pending, root: where.dataRoot ?? '' })}</p>
      <p className="break-all">{t('settings.workspaceArchive.import.reportAt', { path: reportPath })}</p>
      <button type="button" className="settings-save-btn" onClick={dismiss}>{t('settings.workspaceArchive.import.dismiss')}</button>
    </div>}
    {job?.state === 'cancelled' && <p role="status">{t('settings.workspaceArchive.import.cancelled')}</p>}
    {job?.state === 'failed' && <div role="alert" className="text-red-600 dark:text-red-400">
      <p>{t('settings.workspaceArchive.import.failed', { error: job.error ?? '' })}</p>
      {job.errors.filter(e => e !== job.error).map((e, i) => <p key={i}>{e}</p>)}
    </div>}

    <div className="setting-row">
      <div className="setting-info"><ParamLabel className="setting-label" label={t('settings.workspaceArchive.import.zip')} info={t('settings.workspaceArchive.import.zipInfo')} /></div>
      <div className="settings-path-actions">
        <input className="env-input" aria-label={t('settings.workspaceArchive.import.zip')} disabled={busy} value={zip} onChange={e => { setZip(e.target.value); clear(); }} />
        <button type="button" className="env-browse-btn" disabled={busy} title={t('settings.workspaceArchive.import.browseZip')} aria-label={t('settings.workspaceArchive.import.browseZip')} onClick={() => setBrowse('zip')}><FolderOpen size={16} /></button>
      </div>
    </div>
    <div className="setting-row">
      <div className="setting-info"><ParamLabel className="setting-label" label={t('settings.workspaceArchive.import.target')} info={t('settings.workspaceArchive.import.targetInfo')} /></div>
      <div className="settings-path-actions">
        <input className="env-input" aria-label={t('settings.workspaceArchive.import.target')} disabled={busy} value={target} onChange={e => { setTarget(e.target.value); clear(); }} />
        <button type="button" className="env-browse-btn" disabled={busy} title={t('settings.workspaceArchive.import.browseTarget')} aria-label={t('settings.workspaceArchive.import.browseTarget')} onClick={() => setBrowse('target')}><FolderOpen size={16} /></button>
        <button type="button" className="settings-save-btn" disabled={busy || !zip.trim()} onClick={() => previewMut.mutate()}>{t('settings.workspaceArchive.import.preview')}</button>
      </div>
    </div>

    {preview && <div className="space-y-1">
      <p>{t('settings.workspaceArchive.import.archive', { date: date(preview.archive.createdAt), version: preview.archive.appVersion, platform: preview.archive.platform })}</p>
      {preview.sources.map(s => <p key={s.prefix}>{t(`settings.workspaceArchive.cat.${s.category}`)}: {gb(s.bytes)} · {t('settings.workspaceArchive.files', { count: s.files })}</p>)}
      <p>{t('settings.workspaceArchive.import.total', { size: gb(preview.bytes), count: preview.files })}</p>
      {preview.target && <p>{t('settings.workspaceArchive.free', { size: free === null ? t('settings.workspaceArchive.unknown') : gb(free) })}</p>}
      {!preview.target && <p>{t('settings.workspaceArchive.import.needTarget')}</p>}
      {preview.externals.length > 0 && <div>
        <p>{t('settings.workspaceArchive.import.externals')}</p>
        <ul className="font-mono break-all">{preview.externals.map(x => <li key={x.path} className={x.existsHere ? '' : 'text-amber-600 dark:text-amber-400'}>{x.path}{x.existsHere ? '' : ` — ${t('settings.workspaceArchive.import.missing')}`}</li>)}</ul>
      </div>}
      {tooSmall && <p role="alert" className="text-red-600 dark:text-red-400">{t('settings.workspaceArchive.noSpace')}</p>}
      {problems.map((p, i) => <p key={i} role="alert" className="text-red-600 dark:text-red-400">{p}</p>)}
      <button type="button" className="settings-save-btn" disabled={!canStart} onClick={() => setConfirming(true)}>{t('settings.workspaceArchive.import.start')}</button>
    </div>}

    <ConfirmDialog isOpen={confirming} title={t('settings.workspaceArchive.import.confirmTitle')} message={t('settings.workspaceArchive.import.confirmMessage', { target: target.trim() })}
      confirmLabel={t('settings.workspaceArchive.import.start')} onCancel={() => setConfirming(false)} onConfirm={() => startMut.mutate()} />
    <FileBrowserModal open={browse !== null} mode={browse === 'zip' ? 'file' : 'folder'} startPath={browse === 'zip' ? parentDir(zip.trim()) : target}
      title={t(browse === 'zip' ? 'settings.workspaceArchive.import.browseZip' : 'settings.workspaceArchive.import.browseTarget')}
      onClose={() => setBrowse(null)} onSelect={p => { if (browse === 'zip') setZip(p); else setTarget(p); setBrowse(null); clear(); }} />
  </div>;
}
