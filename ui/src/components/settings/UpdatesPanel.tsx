import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { ExternalLink, RefreshCw } from 'lucide-react';
import { updatesApi, type UpdateChannel, type UpdatesStatus } from '../../services/updatesApi';
import { updatesStatusOptions } from '../../queries/shell-queue/settings';
import { ParamLabel } from '../shared/ParamLabel';
import { SelectRow } from './SettingsPrimitives';

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
  const query = useQuery({
    ...updatesStatusOptions(),
    // A check started elsewhere (startup, another tab) reports checking:true; follow it until it ends.
    refetchInterval: q => (q.state.data?.checking ? 2000 : false),
  });
  const check = useMutation({
    mutationFn: (channel?: UpdateChannel) => updatesApi.check(channel),
    onSuccess: result => queryClient.setQueryData(updatesStatusOptions().queryKey, result),
  });
  const s = query.data;
  const busy = check.isPending || !!s?.checking;
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
        <SelectRow id="setting-update-feed" label={t('settings.updates.channel')} description={t('settings.updates.channelInfo')}
          value={s.channel} disabled={busy}
          onChange={value => check.mutate(value as UpdateChannel)}
          options={(['release', 'master'] as const).map(value => ({ value, label: t(`settings.updates.${value}`) }))} />
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
          <p className="mt-1">{t('settings.updates.noInstall')}</p>
          {notesLink && <div className="mt-3">{notesLink}</div>}
        </div>}
        {notice && <p role="status" className="setting-description">{t(`settings.updates.${notice}`)}</p>}
        {s.status !== 'update-available' && notesLink && <div>{notesLink}</div>}
      </>}
    </div>
  </section>;
}
