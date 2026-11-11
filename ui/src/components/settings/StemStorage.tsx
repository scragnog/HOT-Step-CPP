// StemStorage.tsx — extracted stem storage stats + clear-all for the settings
// storage tab. The clear is a two-click confirm; its failure renders inline
// here (the call is silent at the transport), so this row is the only notifier.

import React, { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Scissors, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { deleteAllJobs, formatBytes, type StemStats } from '../../services/stemStudioApi';
import { stemStatsOptions } from '../../queries/shell-queue/settings';
import { queryManager } from '../../queries/invalidation';
import { QueryError } from '../shared/QueryError';

export const StemStorage: React.FC = () => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const stemStatsQuery = useQuery(stemStatsOptions());
  const stemStats = stemStatsQuery.data ?? null;
  const [stemClearConfirm, setStemClearConfirm] = useState(false);
  const [stemClearError, setStemClearError] = useState<string | null>(null);

  const clearMut = useMutation({
    mutationFn: () => deleteAllJobs(),
    // A new attempt clears the stale line from a previous failure.
    onMutate: () => setStemClearError(null),
    onSuccess: () => {
      // Zero the row immediately so it doesn't flash the old totals while the
      // invalidation refetch lands.
      queryClient.setQueryData<StemStats>(stemStatsOptions().queryKey, { totalBytes: 0, jobCount: 0, stemCount: 0 });
      void queryManager.invalidate(queryClient, stemStatsOptions().queryKey);
      setStemClearConfirm(false);
    },
    onError: e => setStemClearError(e instanceof Error ? e.message : 'Clear failed'),
  });
  const stemClearing = clearMut.isPending;
  // Named zero-arg handler so the button can pass it straight to onClick —
  // mutate itself takes `void` variables, which a click event is not.
  const clearStems = () => clearMut.mutate();

  return (
    <div className="settings-section">
      <div className="settings-section-header">
        <Scissors size={16} className="settings-section-icon" />
        <span className="settings-section-title">{t('settings.storage.stemStorage')}</span>
      </div>

      {stemStatsQuery.isError && (
        <QueryError error={stemStatsQuery.error} onRetry={() => void stemStatsQuery.refetch()} />
      )}

      <div className="setting-row">
        <div className="setting-info">
          <div className="setting-label">{t('settings.storage.extractedStems')}</div>
          <div className="setting-description">
            {stemStats ? (
              <>
                {stemStats.jobCount} extraction{stemStats.jobCount !== 1 ? 's' : ''} · {stemStats.stemCount} stem{stemStats.stemCount !== 1 ? 's' : ''} · {formatBytes(stemStats.totalBytes)}
              </>
            ) : (
              t('settings.storage.loading')
            )}
          </div>
        </div>
        <div className="settings-danger-actions">
          {!stemClearConfirm ? (
            <button
              id="clear-stems-btn"
              onClick={() => setStemClearConfirm(true)}
              disabled={stemClearing || !stemStats || stemStats.jobCount === 0}
              className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
              style={{
                background: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                cursor: (!stemStats || stemStats.jobCount === 0) ? 'not-allowed' : 'pointer',
                opacity: (!stemStats || stemStats.jobCount === 0) ? 0.5 : 1,
              }}
            >
              🗑️ {t('settings.storage.clearAllStems')}
            </button>
          ) : (
            <div className="settings-actions">
              <span className="text-xs text-red-400" style={{ whiteSpace: 'nowrap' }}>{t('settings.storage.deleteAllStems')}</span>
              <button
                id="clear-stems-confirm-btn"
                onClick={clearStems}
                disabled={stemClearing}
                className="px-4 py-2 rounded-lg text-sm font-bold transition-all duration-200"
                style={{
                  background: '#dc2626',
                  border: '1px solid #ef4444',
                  color: 'white',
                  cursor: stemClearing ? 'wait' : 'pointer',
                  opacity: stemClearing ? 0.6 : 1,
                }}
              >
                {stemClearing ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Loader2 size={14} className="animate-spin" /> {t('settings.storage.clearing')}
                  </span>
                ) : (
                  '🗑️ ' + t('settings.storage.confirm')
                )}
              </button>
              <button
                onClick={() => setStemClearConfirm(false)}
                disabled={stemClearing}
                className="px-3 py-2 rounded-lg text-sm text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors"
                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', cursor: 'pointer' }}
              >
                {t('settings.storage.cancel')}
              </button>
            </div>
          )}
          {stemClearError && (
            <span className="text-xs text-red-400" style={{ whiteSpace: 'nowrap' }}>{stemClearError}</span>
          )}
        </div>
      </div>
    </div>
  );
};
