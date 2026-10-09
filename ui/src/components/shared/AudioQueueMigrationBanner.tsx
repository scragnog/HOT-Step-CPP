import React, { useEffect, useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import {
  ensureQueueOwnerDecided, getAudioQueueOwner, retryQueueMigration, rollbackAudioQueue,
  useAudioGenQueueSelector,
} from '../../stores/audioGenQueueStore';
import { shouldShowMigrationBanner } from '../../stores/audioQueueOwnerDecision';

export const AudioQueueMigrationBanner: React.FC = () => {
  const { token } = useAuth();
  const count = useAudioGenQueueSelector(s => s.items.length);
  const [ready, setReady] = useState(false);
  const [owner, setOwner] = useState(getAudioQueueOwner());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
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

  if (!shouldShowMigrationBanner(!!token, ready, owner) || !token) return null;
  const run = async (action: () => Promise<void>) => {
    setBusy(true); setError('');
    try { await action(); setOwner(getAudioQueueOwner()); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); setOwner(getAudioQueueOwner()); }
    finally { setBusy(false); }
  };
  return <div className="fixed bottom-40 left-1/2 -translate-x-1/2 z-[60] max-w-[92vw] rounded-xl border border-amber-500/30 bg-white/95 dark:bg-zinc-900/95 px-4 py-3 shadow-2xl text-sm">
    <div className="font-semibold">Audio queue: migration stuck</div>
    <div className="text-xs text-zinc-500">
      Moving {count} queued item{count === 1 ? '' : 's'} to the server didn't finish. The browser queue stays
      paused — nothing runs twice — until you retry or roll back.
    </div>
    {receipt && <div className="text-xs text-emerald-600">{receipt}</div>}
    {error && <div role="alert" className="text-xs text-red-500">{error}</div>}
    <div className="mt-2 flex gap-2">
      <button type="button" disabled={busy} onClick={() => run(async () => { await retryQueueMigration(token); setReceipt('Move to the server finished.'); })} className="rounded bg-amber-500/20 px-3 py-1">Retry</button>
      <button type="button" disabled={busy} onClick={() => run(async () => { await rollbackAudioQueue(token); setReceipt('Browser ownership restored.'); })} className="rounded bg-amber-500/20 px-3 py-1">Export and roll back</button>
    </div>
  </div>;
};
