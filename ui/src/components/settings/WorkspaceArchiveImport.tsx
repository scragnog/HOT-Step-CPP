import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { FolderOpen } from 'lucide-react';
import { workspaceArchiveApi, type ArchiveImportPreview } from '../../services/workspaceArchiveApi';
import { workspaceArchiveImportJobOptions } from '../../queries/shell-queue/settings';
import { serverKey } from '../../queries/keys';
import { dataFolderApi } from '../../services/dataFolderApi';
import { queryManager } from '../../queries/invalidation';
import { ConfirmDialog } from '../shared/ConfirmDialog';
import { FileBrowserModal } from '../shared/FileBrowserModal';
import { ParamLabel } from '../shared/ParamLabel';

const gb = (n: number) => `${(n / 1e9).toFixed(2)} GB`;
// The accepted import (id and target) survives the restart's page reload here, so the outcome can be checked afterwards.
const PENDING_KEY = 'hotstep:workspace-import-pending';
type Pending = { id: string; target: string };
// Windows paths ignore case; elsewhere /data/A and /data/a are different folders.
const samePath = (a: string, b: string) => {
  const n = (p: string) => p.replace(/\\/g, '/').replace(/\/+$/, '');
  const win = /^[a-z]:|\\/i.test(a) || /^[a-z]:|\\/i.test(b);
  return win ? n(a).toLowerCase() === n(b).toLowerCase() : n(a) === n(b);
};
const parentDir = (p: string) => p.replace(/[\\/]+[^\\/]*$/, '');
const readPending = (): Pending | null => { try { const v = JSON.parse(sessionStorage.getItem(PENDING_KEY) ?? 'null'); return v && typeof v.id === 'string' && typeof v.target === 'string' ? v : null; } catch { return null; } };
const writePending = (v: Pending | null) => { try { if (v) sessionStorage.setItem(PENDING_KEY, JSON.stringify(v)); else sessionStorage.removeItem(PENDING_KEY); } catch { /* storage unavailable: no outcome check */ } };

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
  const [reloaded] = useState(() => !!readPending());
  const [pending, setPending] = useState(readPending);

  const jobQuery = useQuery(workspaceArchiveImportJobOptions());
  const job = jobQuery.data?.job ?? null;
  const running = job?.state === 'running';
  const waiting = running && job.progress?.phase === 'waiting';

  // The server restarts about 300 ms after `done`; the app shell then shows its restart overlay and
  // reloads the page when the server answers. The poll can miss that window, so the outcome check
  // below runs from the saved mark after the reload, not from seeing `done`.
  const done = job?.state === 'done';
  useEffect(() => { if (done) window.dispatchEvent(new CustomEvent('hotstep:restarting')); }, [done]);
  const mine = !!pending && job?.id === pending.id;
  // Only this import's own failure or cancel clears the mark. After a reload, any other job is a stale mark.
  useEffect(() => {
    if (!pending || !jobQuery.isSuccess) return;
    if (mine ? job.state === 'failed' || job.state === 'cancelled' : !!job && reloaded) { writePending(null); setPending(null); }
  }, [pending, jobQuery.isSuccess, mine, job, reloaded]);
  // A new server has no job (it lives in memory). After a reload, a `done` job of ours means the old server still runs.
  const verifying = !!pending && jobQuery.isSuccess && (!job || (reloaded && mine && job.state === 'done'));
  // Own key and no cache: DataFolderPanel's cached location would still show the old root.
  const locationQuery = useQuery({
    queryKey: serverKey('shell-queue/data-folder', { filters: { verifyImport: pending?.id ?? '' } }),
    queryFn: ({ signal }) => dataFolderApi.location({ signal }),
    enabled: verifying, gcTime: 0, staleTime: 0, refetchInterval: q => (q.state.data ? false : 2000),
  });
  const verdict = verifying && locationQuery.isSuccess ? locationQuery.data : null;
  const switched = !!verdict && !!pending && samePath(verdict.dataRoot ?? '', pending.target);
  const reportPath = pending ? `${pending.target}${/[\\/]$/.test(pending.target) ? '' : '/'}import-report.json` : '';
  const dismiss = () => { writePending(null); setPending(null); };

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
    onSuccess: async ({ job: started }) => {
      // Seed the job cache with this import, so a stale poll of an earlier job cannot clear or confirm it.
      const key = workspaceArchiveImportJobOptions().queryKey;
      await queryClient.cancelQueries({ queryKey: key });
      queryClient.setQueryData(key, { job: started });
      const mark = { id: started.id, target: started.target };
      writePending(mark);
      setPending(mark);
    },
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
    {done && job && <div role="status" className="space-y-1">
      <p>{t('settings.workspaceArchive.import.restarting', { target: job.target })}</p>
      <p className="break-all">{t('settings.workspaceArchive.import.reportAt', { path: reportPath })}</p>
    </div>}
    {verifying && !verdict && <p role="status">{t('settings.workspaceArchive.import.checking')}</p>}
    {verdict && pending && <div role="status" className="space-y-1">
      <p>{t(switched ? 'settings.workspaceArchive.import.switched' : 'settings.workspaceArchive.import.notSwitched', { target: pending.target, root: verdict.dataRoot ?? '' })}</p>
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
