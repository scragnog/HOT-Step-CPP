import React, { useEffect, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useAuth } from '../../context/AuthContext';
import { audioQueueItemsOptions } from '../../queries/shell-queue/queue';
import { queryManager } from '../../queries/invalidation';
import {
  ensureQueueOwnerDecided, getAudioQueueOwner, retryQueueMigration, useAudioGenQueueSelector,
} from '../../stores/audioGenQueueStore';
import { shouldShowMigrationBanner } from '../../stores/audioQueueOwnerDecision';

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
    // Retry moves the queue's server-side ownership; the cached item list
    // must not keep the previous owner's snapshot.
    void queryManager.invalidate(client, audioQueueItemsOptions().queryKey);
  };

  const retryMut = useMutation({
    mutationFn: async () => { await retryQueueMigration(token!); setReceipt('Move to the server finished.'); },
    onSettled: afterOwnershipChange,
  });

  if (!shouldShowMigrationBanner(!!token, ready, owner) || !token) return null;
  const error = retryMut.error?.message ?? '';
  return <div className="fixed bottom-40 left-1/2 -translate-x-1/2 z-[60] max-w-[92vw] rounded-xl border border-amber-500/30 bg-white/95 dark:bg-zinc-900/95 px-4 py-3 shadow-2xl text-sm">
    <div className="font-semibold">Audio queue: migration stuck</div>
    <div className="text-xs text-zinc-500">
      Moving {count} queued item{count === 1 ? '' : 's'} to the server didn’t finish. The queue stays
      paused until the move is retried. The items stay saved in this browser.
    </div>
    {receipt && <div className="text-xs text-emerald-600">{receipt}</div>}
    {error && <div role="alert" className="text-xs text-red-500">{error}</div>}
    <div className="mt-2 flex gap-2">
      <button type="button" disabled={retryMut.isPending} onClick={() => retryMut.mutate()} className="rounded bg-amber-500/20 px-3 py-1">Retry</button>
    </div>
  </div>;
};
