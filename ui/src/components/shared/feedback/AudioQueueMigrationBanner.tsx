import React, { useEffect, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useAuth } from '@/context/AuthContext';
import { audioQueueItemsOptions } from '@/queries/shell-queue/queue';
import { queryManager } from '@/queries/invalidation';
import {
  ensureQueueOwnerDecided, getAudioQueueOwner, isAudioQueueRollbackPending, retryQueueMigration, rollbackAudioQueue,
  useAudioGenQueueSelector,
} from '@/stores/audioGenQueueStore';
import { shouldShowMigrationBanner } from '@/stores/audioQueueOwnerDecision';
import { plural } from '@/utils/format/plural';

export const AudioQueueMigrationBanner: React.FC = () => {
  const { token } = useAuth();
  const count = useAudioGenQueueSelector(s => s.items.length);
  const [ready, setReady] = useState(false);
  const [owner, setOwner] = useState(getAudioQueueOwner());
  const [receipt, setReceipt] = useState('');

  // A never-decided browser migrates (or opts into server ownership) silently
  // on load — don't render the manual prompt until that decision has landed.
  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    void ensureQueueOwnerDecided(token).finally(() => {
      if (!cancelled) { setOwner(getAudioQueueOwner()); setReady(true); }
    });
    return () => { cancelled = true; };
  }, [token]);

  const client = useQueryClient();
  const afterOwnershipChange = () => {
    setOwner(getAudioQueueOwner());
    // Both commands move the queue's server-side ownership; the cached item
    // list must not keep the previous owner's snapshot.
    void queryManager.invalidate(client, audioQueueItemsOptions().queryKey);
  };

  const retryMut = useMutation({
    mutationFn: async () => { await retryQueueMigration(token!); setReceipt('Move to the server finished.'); },
    onSettled: afterOwnershipChange,
  });
  const rollbackMut = useMutation({
    mutationFn: async () => { await rollbackAudioQueue(); setReceipt('Browser ownership restored.'); },
    onSettled: afterOwnershipChange,
  });

  if (!shouldShowMigrationBanner(!!token, ready, owner) || !token) return null;
  const busy = retryMut.isPending || rollbackMut.isPending;
  const error = retryMut.error?.message ?? rollbackMut.error?.message ?? '';
  return <div className="fixed bottom-40 left-1/2 -translate-x-1/2 z-[60] max-w-[92vw] rounded-xl border border-amber-500/30 bg-white/95 dark:bg-zinc-900/95 px-4 py-3 shadow-2xl text-sm">
    <div className="font-semibold">Audio queue: {isAudioQueueRollbackPending() ? 'roll back stuck' : 'migration stuck'}</div>
    <div className="text-xs text-zinc-500">
      {isAudioQueueRollbackPending()
        ? 'Rolling the queue back to this browser didn’t finish.'
        : `Moving ${count} queued ${plural(count, 'item')} to the server didn’t finish.`} The browser
      queue stays paused (nothing runs twice) until you retry the move to the server or roll back.
    </div>
    {receipt && <div className="text-xs text-emerald-600">{receipt}</div>}
    {error && <div role="alert" className="text-xs text-red-500">{error}</div>}
    <div className="mt-2 flex gap-2">
      <button type="button" disabled={busy} onClick={() => retryMut.mutate()} className="rounded bg-amber-500/20 px-3 py-1">Retry</button>
      <button type="button" disabled={busy} onClick={() => rollbackMut.mutate()} className="rounded bg-amber-500/20 px-3 py-1">Export and roll back</button>
    </div>
  </div>;
};
