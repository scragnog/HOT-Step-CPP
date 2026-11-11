import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useQueryClient } from '@tanstack/react-query';
import type { BackupPreview, SettingsBackup } from '../../../../server/src/services/settingsBackup';
import type { AppSettings } from './SettingsPanel';
import { http } from '../../services/axiosClient';
import { queryManager } from '../../queries/invalidation';
import { ConfirmDialog } from '../shared/ConfirmDialog';
import { ParamLabel } from '../shared/ParamLabel';
import { Archive } from 'lucide-react';

interface Props { settings: AppSettings; onSettingsChange: (settings: AppSettings) => void }
const MAX_BYTES = 500 * 1024 * 1024; // server accepts 512 MB (BACKUP_BODY_LIMIT)

export function SettingsBackupPanel({ settings, onSettingsChange }: Props) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const running = useRef(false);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);
  const [pending, setPending] = useState<{ backup: unknown; preview: BackupPreview } | null>(null);
  const [confirm, setConfirm] = useState(false);

  const run = async (action: () => Promise<void>) => {
    if (running.current) return;
    running.current = true;
    setBusy(true); setError(''); setSaved(false);
    try { await action(); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { running.current = false; setBusy(false); }
  };
  const download = () => run(async () => {
    const { data } = await http.post<SettingsBackup>('/settings/backup/export', { app: settings });
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `hotstep-settings-${data.exportedAt.slice(0, 10)}.json`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  const choose = (file: File) => run(async () => {
    setPending(null);
    if (file.size > MAX_BYTES) throw new Error(t('settings.backup.tooLarge', 'Backup files must be smaller than 500 MB.'));
    let backup: unknown;
    try { backup = JSON.parse(await file.text()); }
    catch { throw new Error(t('settings.backup.badJson', 'This file is not valid JSON.')); }
    const { data: preview } = await http.post<BackupPreview>('/settings/backup/preview', { backup });
    setPending({ backup, preview });
  });
  const restore = () => {
    setConfirm(false);
    const selected = pending;
    if (!selected) return;
    void run(async () => {
      const { data } = await http.post<{ app: AppSettings }>('/settings/backup/import', { backup: selected.backup, confirmed: true });
      onSettingsChange(data.app);
      setPending(null); setSaved(true);
      await queryManager.all(queryClient);
    });
  };
  const labels: Record<keyof BackupPreview['current'], string> = {
    artists: t('settings.backup.artists', 'Artists'), lyrics_sets: t('settings.backup.albums', 'Lyric collections'),
    profiles: t('settings.backup.profiles', 'Profiles'), generations: t('settings.backup.lyrics', 'Generated lyrics'),
    album_presets: t('settings.backup.presets', 'Album presets'), audio_generations: t('settings.backup.audio', 'Audio links'),
    settings: t('settings.backup.preferences', 'Lyric Studio preferences'),
  };
  return <section className="settings-section text-sm text-zinc-700 dark:text-zinc-300">
    <div className="settings-section-header"><Archive size={16} className="settings-section-icon" /><span className="settings-section-title">{t('settings.backup.title', 'Settings and Lyric Studio backup')}</span></div>
    <div className="setting-row">
      <div className="setting-info">
        <ParamLabel className="setting-label" label={t('settings.backup.title', 'Settings and Lyric Studio backup')}
          info={t('settings.backup.info', 'Save application settings and Lyric Studio data to JSON. Import replaces them after confirmation. Audio files, models, environment settings, and other browser preferences are excluded.')} />
        <p className="setting-description">{t('settings.backup.scope', 'Includes General and Performance settings, Lyric Studio preferences, artists, lyric collections, profiles, generated lyrics, presets, and audio links. Media files and environment settings are excluded.')}</p>
      </div>
    <div className="settings-actions">
      <button className="settings-save-btn" disabled={busy} onClick={() => void download()}>{t('settings.backup.export', 'Export JSON')}</button>
      <button className="settings-save-btn" disabled={busy} onClick={() => fileInput.current?.click()}>{t('settings.backup.choose', 'Choose JSON to import')}</button>
      <input ref={fileInput} type="file" accept="application/json,.json" hidden disabled={busy}
        aria-label={t('settings.backup.choose', 'Choose JSON to import')} onChange={event => {
          const file = event.target.files?.[0]; event.target.value = '';
          if (file) void choose(file);
        }} />
    </div>
    </div>
    {pending && <div className="space-y-2">
      <p>{t('settings.backup.replacement', 'Import will replace application settings and all Lyric Studio data. Current records will be removed, including any created since this preview.')}</p>
      <table className="w-full text-left"><thead><tr>
        <th>{t('settings.backup.data', 'Data')}</th><th>{t('settings.backup.current', 'Current')}</th><th>{t('settings.backup.incoming', 'In file')}</th>
      </tr></thead><tbody>{(Object.keys(labels) as (keyof typeof labels)[]).map(key => <tr key={key}>
        <td>{labels[key]}</td><td>{pending.preview.current[key]}</td><td>{pending.preview.incoming[key]}</td>
      </tr>)}</tbody></table>
      <p>{t('settings.backup.paths', 'Audio links, images, reference tracks, and adapters still need their original files. Training collections remain unchanged. Export a backup first if you want to keep the current data.')}</p>
      <button className="placement-button" disabled={busy} onClick={() => setConfirm(true)}>{t('settings.backup.replace', 'Replace settings and Lyric Studio data')}</button>
      <button className="placement-button ml-2" disabled={busy} onClick={() => setPending(null)}>{t('common.cancel', 'Cancel')}</button>
    </div>}
    <div role="status" aria-live="polite">
      {busy && <p>{t('settings.backup.working', 'Working...')}</p>}
      {error && <p className="text-red-600 dark:text-red-400">{error}</p>}
      {saved && <p>{t('settings.backup.saved', 'Backup imported. Application settings and Lyric Studio data have been replaced.')}</p>}
    </div>
    <ConfirmDialog isOpen={confirm} title={t('settings.backup.confirmTitle', 'Replace settings and Lyric Studio data?')}
      message={t('settings.backup.confirmMessage', 'This permanently replaces your current application settings and every Lyric Studio record with the selected JSON file. Continue?')}
      confirmLabel={t('settings.backup.confirm', 'Replace data')} danger onConfirm={restore} onCancel={() => setConfirm(false)} />
  </section>;
}
