import { useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { dataFolderApi, type DataFolderLocation, type DataFolderOperation, type DataFolderPreview } from '../../services/dataFolderApi';
import { shutdownApi } from '../../services/api';
import { ParamLabel } from '../shared/ParamLabel';
import { dataFolderOptions } from '../../queries/shell-queue/settings';
import { queryManager } from '../../queries/invalidation';
import { StyledSelect } from '../shared/StyledSelect';
import { FolderOpen } from 'lucide-react';

export function DataFolderPanel() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const locationQuery = useQuery(dataFolderOptions());
  const location: DataFolderLocation | null = locationQuery.data ?? null;
  const [kind, setKind] = useState<DataFolderOperation>('move');
  const [target, setTarget] = useState('');
  const [preview, setPreview] = useState<DataFolderPreview | null>(null);
  const [error, setError] = useState('');
  const revision = useRef(0);

  // Every command below is silent at the transport — this panel's inline line
  // is each one's only notifier, so each mutation owns its failure display.
  // Writes re-read location when they settle: a failed or half-applied command
  // can change what the server reports (a pending op, blockers).
  const invalidate = () => void queryManager.invalidate(queryClient, dataFolderOptions().queryKey);
  const fail = (e: unknown) => setError(e instanceof Error ? e.message : String(e));

  const previewMut = useMutation({
    mutationFn: async ({ kind, target }: { kind: DataFolderOperation; target: string }) => {
      // A late response must not overwrite a newer one: compare this click's
      // revision against the current one when it lands.
      const rev = revision.current;
      return { result: await dataFolderApi.preview(kind, target), rev };
    },
    onMutate: () => setError(''),
    onSuccess: ({ result, rev }) => { if (rev === revision.current) setPreview(result); },
    onError: fail,
    onSettled: invalidate,
  });

  const scheduleMut = useMutation({
    mutationFn: ({ kind, target }: { kind: DataFolderOperation; target: string }) => dataFolderApi.schedule(kind, target),
    onMutate: () => setError(''),
    onSuccess: () => window.dispatchEvent(new CustomEvent('hotstep:restarting')),
    onError: e => { fail(e); invalidate(); },
  });

  const cancelMut = useMutation({
    mutationFn: () => dataFolderApi.cancel(),
    onMutate: () => setError(''),
    onError: fail,
    onSettled: invalidate,
  });

  const restartMut = useMutation({
    mutationFn: async () => {
      // The server tears the connection down as it goes, so a rejected request
      // is expected; surface the localized line, not a network error.
      try { await shutdownApi.restart(); } catch { throw new Error(t('settings.dataFolder.restartFailed')); }
    },
    onMutate: () => setError(''),
    onSuccess: () => window.dispatchEvent(new CustomEvent('hotstep:restarting')),
    onError: fail,
  });

  // Named zero-arg handlers so the buttons can pass them straight to onClick —
  // mutate itself takes `void` variables, which a click event is not.
  const runPreview = () => previewMut.mutate({ kind, target: target.trim() });
  const runSchedule = () => { if (preview) scheduleMut.mutate({ kind: preview.kind, target: preview.target }); };
  const runCancel = () => cancelMut.mutate();
  const runRestart = () => restartMut.mutate();

  const busy = previewMut.isPending || scheduleMut.isPending || cancelMut.isPending || restartMut.isPending;
  const change = () => { revision.current++; setPreview(null); setError(''); };
  const bytes = (value: number) => `${(value / 1e9).toFixed(2)} GB`;
  const blocked = busy || !location || !!location.override || !!location.pending;
  return <section className="settings-section">
    <div className="settings-section-header"><FolderOpen size={16} className="settings-section-icon" /><span className="settings-section-title">{t('settings.dataFolder.title')}</span></div>
    <div className="settings-grid text-sm text-zinc-700 dark:text-zinc-300">
      {locationQuery.error && <p role="alert" className="text-red-600 dark:text-red-400">{locationQuery.error.message}</p>}
      {error && <p role="alert" className="text-red-600 dark:text-red-400">{error}</p>}
      {!location && !locationQuery.error && <p>{t('settings.dataFolder.loading')}</p>}
      {location && <>
        <div className="setting-row">
          <div className="setting-info"><ParamLabel className="setting-label" label={t('settings.dataFolder.current')} info={t('settings.dataFolder.currentInfo')} /></div>
          <p className="setting-value font-mono">{location.dataRoot || t('settings.dataFolder.dev')}</p>
        </div>
        {location.override && <p className="break-all">{t('settings.dataFolder.override', { path: location.override })}</p>}
        {location.externals.length > 0 && <div className="setting-row">
          <div className="setting-info"><ParamLabel className="setting-label" label={t('settings.dataFolder.externals')} info={t('settings.dataFolder.externalsInfo')} /></div>
          <ul className="setting-value font-mono">{location.externals.map(item => <li key={item.key}>{item.key}: {item.path}</li>)}</ul>
        </div>}
        {location.blockers.length > 0 && <div><p>{t('settings.dataFolder.wait')}</p><ul>{location.blockers.map((item, i) => <li key={i}>{item}</li>)}</ul></div>}
        {location.pending && <div className="space-y-2">
          <p>{t('settings.dataFolder.pending', { action: t(`settings.dataFolder.${location.pending.kind}`) })}</p>
          <p className="break-all">{location.pending.source || location.dataRoot} → {location.pending.destination}</p>
          <button className="settings-save-btn" disabled={busy} onClick={runRestart}>{t('settings.dataFolder.restart')}</button>
          <button className="settings-save-btn" disabled={busy} onClick={runCancel}>{t('settings.dataFolder.cancel')}</button>
        </div>}
        <div className="setting-row">
        <div className="setting-info">
          <ParamLabel className="setting-label" label={t('settings.dataFolder.action')} info={t('settings.dataFolder.actionInfo')} />
          <p className="setting-description">{t(`settings.dataFolder.${kind}Info`)}</p>
        </div>
        <StyledSelect value={kind} disabled={blocked} accent="emerald" onChange={value => { change(); setKind(value as DataFolderOperation); }} options={(['move', 'repoint', 'create', 'import'] as const).map(value => ({ value, label: t(`settings.dataFolder.${value}`) }))} />
        </div>
        <div className="setting-row">
          <div className="setting-info"><ParamLabel className="setting-label" label={t(kind === 'import' ? 'settings.dataFolder.previousInstall' : 'settings.dataFolder.target')} info={t('settings.dataFolder.targetInfo')} /></div>
          <div className="settings-path-actions">
            <input className="env-input" aria-label={t(kind === 'import' ? 'settings.dataFolder.previousInstall' : 'settings.dataFolder.target')} disabled={blocked} value={target} onChange={e => { change(); setTarget(e.target.value); }} />
            <button type="button" className="settings-save-btn" disabled={blocked || !target.trim()} onClick={runPreview}>{t('settings.dataFolder.preview')}</button>
          </div>
        </div>
        {preview && <div className="space-y-2">
          <p className="break-all">{t('settings.dataFolder.current')}: {location.dataRoot || t('settings.dataFolder.dev')}</p>
          <p className="break-all">{t(kind === 'import' ? 'settings.dataFolder.previousInstall' : 'settings.dataFolder.target')}: {preview.target}</p>
          <p>{t('settings.dataFolder.size', { size: bytes(preview.bytes) })}</p>
          {preview.crossVolume && <p>{t('settings.dataFolder.space', { needed: bytes(preview.needBytes), free: preview.freeBytes === null ? t('settings.dataFolder.unknown') : bytes(preview.freeBytes) })}</p>}
          {preview.mappings.map(item => <p key={item.name} className="break-all">{item.name}: {item.from} → {item.to}</p>)}
          {preview.errors.map((item, i) => <p role="alert" key={i} className="text-red-600 dark:text-red-400">{item}</p>)}
          {preview.blockers.map((item, i) => <p key={i}>{item}</p>)}
          <button className="settings-save-btn" disabled={blocked || !preview.ok || preview.blockers.length > 0 || location.blockers.length > 0} onClick={runSchedule}>{t('settings.dataFolder.confirm')}</button>
        </div>}
      </>}
    </div>
  </section>;
}
