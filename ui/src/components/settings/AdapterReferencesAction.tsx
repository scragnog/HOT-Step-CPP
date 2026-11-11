import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { queryManager } from '../../queries/invalidation';
import { lireekApi } from '../../services/lireekApi';

export function AdapterReferencesAction() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const updateMut = useMutation({
    mutationFn: lireekApi.updateAdapterReferences,
    // The command rewrites preset adapter references inside lyrics sets, so
    // every cached Lyric Studio snapshot that embeds one is stale.
    onSuccess: () => queryManager.coverLyrics.lireek(client),
  });
  const running = updateMut.isPending;
  const result = updateMut.data ?? null;
  const error = updateMut.error?.message ?? null;
  return (
    <div className="space-y-2">
      <div className="setting-row">
        <div className="setting-info">
          <p className="setting-description">{t('settings.adapters.updateInfo')}</p>
        </div>
        <button className="settings-save-btn" disabled={running} onClick={() => updateMut.mutate()}>
          {t(running ? 'settings.adapters.updating' : 'settings.adapters.update')}
        </button>
      </div>
      <div role="status" aria-live="polite" className="text-sm text-zinc-700 dark:text-zinc-300">
        {error && <p className="text-red-600 dark:text-red-400">{t('settings.adapters.failed', { error })}</p>}
        {result && <>
          <p>{t('settings.adapters.result', { scanned: result.scanned, updated: result.updated, count: result.changes.length })}</p>
          {result.changes.length > 0 && <ul className="mt-2 space-y-1 break-all">
            {result.changes.map(change => <li key={`${change.lyricsSetId}:${change.field}`}>
              {t('settings.adapters.dropped', { preset: change.presetName, adapter: change.adapter })}
            </li>)}
          </ul>}
        </>}
      </div>
    </div>
  );
}
