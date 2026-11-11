import { useContext, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { ExternalLink, RefreshCw } from 'lucide-react';
import { updatesApi, type UpdateChannel, type UpdatesStatus } from '../../services/updatesApi';
import { updatesStatusOptions } from '../../queries/shell-queue/settings';
import { ParamLabel } from '../shared/ParamLabel';
import { SelectRow } from './SettingsPrimitives';
import { ConfirmDialog } from '../shared/ConfirmDialog';
import { UpdateOutcome, UpdateRestartContext } from './UpdateReconnectOverlay';
import { updateRefusal, updatesViewState } from './updatesViewState';

const short = (sha: string | null) => (sha ? sha.slice(0, 8) : null);

/** Explain local changes or an identity that cannot be compared. */
function noticeKey(s: UpdatesStatus): string | null {
  if (s.installed.git?.dirty) return 'dirty';
  if (s.installed.kind === 'unknown' || (s.installed.kind === 'portable' && !s.installed.tag && !s.installed.sha)) return 'unknownIdentity';
  if (s.status === 'unknown') return 'cannotCompare';
  return null;
}

export function UpdatesPanel() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [confirmCandidate, setConfirmCandidate] = useState<string | null>(null);
  const { attempt, setAttempt } = useContext(UpdateRestartContext)!;
  const releaseReset = useRef(false);
  const query = useQuery({
    ...updatesStatusOptions(),
    enabled: !attempt,
  });
  const check = useMutation({
    mutationFn: (channel?: UpdateChannel) => updatesApi.check(channel),
    onSuccess: result => queryClient.setQueryData(updatesStatusOptions().queryKey, result),
  });
  const { mutate: checkFeed } = check;
  useEffect(() => {
    if (query.data && updatesViewState(query.data).resetToRelease && !releaseReset.current) {
      releaseReset.current = true;
      checkFeed('release');
    }
  }, [query.data, checkFeed]);
  const command = useMutation({
    mutationFn: async ({ kind, candidateId }: { kind: 'prepare' | 'cancel' | 'apply'; candidateId: string }) => {
      await queryClient.cancelQueries({ queryKey: updatesStatusOptions().queryKey });
      if (kind === 'apply') {
        const previousResult = queryClient.getQueryData(updatesStatusOptions().queryKey)?.install?.lastResult ?? null;
        const startedAt = Date.now();
        const response = await updatesApi.apply(candidateId);
        queryClient.setQueryData(updatesStatusOptions().queryKey, (old: UpdatesStatus | undefined) => old && {
          ...old, install: old.install && { ...old.install, phase: 'stopping' as const },
        });
        setAttempt({ ...response, previousResult, startedAt });
      } else {
        const prepare = await updatesApi[kind](candidateId);
        queryClient.setQueryData(updatesStatusOptions().queryKey, (old: UpdatesStatus | undefined) => old && { ...old, prepare });
      }
    },
    onError: () => { void query.refetch(); },
  });
  const s = query.data;
  const view = s ? updatesViewState(s, command.error ? updateRefusal(command.error) : null) : null;
  const busy = check.isPending || !!s?.checking || command.isPending || !!view?.active || !!attempt;
  const error = check.error?.message ?? query.error?.message ?? '';
  const when = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : t('settings.updates.never'));
  const notice = s ? noticeKey(s) : null;
  const latest = s?.channel === 'release' ? s.latest.release?.tag : short(s?.latest.master?.sha ?? null);
  const installedLabel = s && [s.installed.tag ?? short(s.installed.sha) ?? t('settings.updates.unknown'), s.installed.variant].filter(Boolean).join(' · ');
  const notesLink = s?.notesUrl && <a className="settings-save-btn" href={s.notesUrl} target="_blank" rel="noreferrer">
    {t(s.channel === 'release' ? 'settings.updates.notes' : 'settings.updates.commit')}
    <ExternalLink size={14} aria-hidden="true" />
  </a>;

  return <section className="settings-section settings-updates">
    <div className="settings-section-header">
      <RefreshCw size={16} className="settings-section-icon" aria-hidden="true" />
      <span className="settings-section-title">{t('settings.updates.title')}</span>
    </div>
    <div className="settings-grid text-sm text-zinc-700 dark:text-zinc-300">
      {error && <p role="alert" className="env-save-status env-save-status--error">{error}</p>}
      {!s && !error && <p className="setting-description">{t('settings.updates.loading')}</p>}
      {s && <>
        <div className="setting-row">
          <div className="setting-info">
            <ParamLabel className="setting-label" label={t('settings.updates.installed')} info={t('settings.updates.installedInfo')} />
          </div>
          <span className="break-all text-right font-mono text-sm">{installedLabel}</span>
        </div>
        {view!.showChannel && <SelectRow id="setting-update-feed" label={t('settings.updates.channel')} description={t('settings.updates.channelInfo')}
          value={s.channel} disabled={busy}
          onChange={value => check.mutate(value as UpdateChannel)}
          options={(['release', 'master'] as const).map(value => ({ value, label: t(`settings.updates.${value}`) }))} />}
        <div className="setting-row">
          <div className="setting-info">
            <p className="setting-description">{t('settings.updates.lastCheck', { time: when(s.lastSuccessAt) })}</p>
          </div>
          <button type="button" className="settings-save-btn" disabled={busy} onClick={() => check.mutate(undefined)}>
            <RefreshCw size={14} className={busy ? 'animate-spin' : ''} aria-hidden="true" />
            {busy ? t('settings.updates.checking') : t('settings.updates.checkNow')}
          </button>
        </div>
        {check.data?.throttled && <p className="setting-description">{t('settings.updates.throttled')}</p>}
        {s.retryAfter && <p className="setting-description">{t('settings.updates.retryAfter', { time: when(s.retryAfter) })}</p>}
        {s.status === 'could-not-check' && <p role="alert" className="env-save-status env-save-status--error">{t('settings.updates.couldNotCheck', { error: s.error ?? '' })}</p>}
        {s.status === 'up-to-date' && <p role="status" className="env-save-status env-save-status--success">{t('settings.updates.upToDate')}</p>}
        {s.status === 'ahead' && <p role="status" className="setting-description">{t('settings.updates.ahead')}</p>}
        {s.status === 'update-available' && <div role="status" className="rounded border border-emerald-500/40 bg-emerald-500/10 p-3">
          <p>{t('settings.updates.available', { version: latest ?? '' })}</p>
          {!view!.offered && <>
            <p className="mt-1">{t(view!.gitInstructions ? 'settings.updates.gitUpdate' : 'settings.updates.noInstall')}</p>
            {view!.reason && <p className="setting-description mt-1">{view!.reason}</p>}
          </>}
          {view!.offered && <div className="settings-update-download">
            {view!.prepareError && <p role="alert" className="env-save-status env-save-status--error">{view!.prepareError}</p>}
            {view!.active && <>
              <p>{t(`settings.updates.phase.${view!.phase}`)}</p>
              <progress max={1} value={view!.progress ?? undefined} aria-label={t(`settings.updates.phase.${view!.phase}`)} />
              <p className="setting-description">{t('settings.updates.bytes', {
                downloaded: formatBytes(s.prepare!.downloadedBytes), total: formatBytes(s.prepare!.totalBytes),
              })}</p>
            </>}
            <div className="settings-actions">
              {view!.active && <button type="button" className="settings-save-btn" disabled={command.isPending}
                onClick={() => command.mutate({ kind: 'cancel', candidateId: view!.candidate!.id })}>{t('settings.updates.cancel')}</button>}
              {view!.action && <button type="button" className="settings-save-btn" disabled={busy}
                onClick={() => view!.ready ? setConfirmCandidate(view!.candidate!.id) : command.mutate({ kind: 'prepare', candidateId: view!.candidate!.id })}>
                {view!.ready ? t('settings.updates.install') : t('settings.updates.download', { size: formatBytes(view!.candidate!.bytes) })}
              </button>}
            </div>
          </div>}
          {notesLink && <div className="mt-3">{notesLink}</div>}
        </div>}
        {view!.refusal && <div role="alert" className="env-save-status env-save-status--error">
          <p>{view!.refusal.error}</p>
          {view!.refusal.blockers.length > 0 && <>
            <ul>{view!.refusal.blockers.map((b, i) => <li key={i}>{b}</li>)}</ul>
            <p>{t('settings.updates.finishJobs')}</p>
          </>}
          {view!.refusal.errors.length > 0 && <ul>{view!.refusal.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>}
        </div>}
        {view!.lastResult && <div className="setting-row setting-row--wide settings-update-result" role="status">
          <ParamLabel className="setting-label" label={t('settings.updates.lastOutcome')} info={t('settings.updates.lastOutcomeInfo')} />
          <div><UpdateOutcome result={view!.lastResult} /></div>
        </div>}
        {notice && <p role="status" className="setting-description">{t(`settings.updates.${notice}`)}</p>}
        {s.status !== 'update-available' && notesLink && <div>{notesLink}</div>}
      </>}
    </div>
    {createPortal(<ConfirmDialog isOpen={!!confirmCandidate} title={t('settings.updates.confirmTitle')} message={t('settings.updates.confirmMessage')}
      confirmLabel={t('settings.updates.install')} onCancel={() => setConfirmCandidate(null)} onConfirm={() => {
        const candidateId = confirmCandidate;
        setConfirmCandidate(null);
        if (candidateId && !command.isPending && view?.ready && candidateId === view.candidate?.id) command.mutate({ kind: 'apply', candidateId });
      }} />, document.body)}
  </section>;
}

function formatBytes(bytes: number) {
  const unit = bytes >= 1024 ** 3 ? 'GB' : 'MB';
  return `${(bytes / (unit === 'GB' ? 1024 ** 3 : 1024 ** 2)).toLocaleString(undefined, { maximumFractionDigits: 1 })} ${unit}`;
}
