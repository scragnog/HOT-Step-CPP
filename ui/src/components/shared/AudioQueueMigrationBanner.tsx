import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import {
  getAudioQueueOwner, migrateAudioQueue, rollbackAudioQueue,
  useAudioGenQueueSelector,
} from '../../stores/audioGenQueueStore';

export const AudioQueueMigrationBanner: React.FC = () => {
  const { token } = useAuth();
  const count = useAudioGenQueueSelector(s => s.items.length);
  const [owner, setOwner] = useState(getAudioQueueOwner());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [receipt, setReceipt] = useState('');

  if (!token || (owner === 'browser' && count === 0)) return null;
  const run = async (action: () => Promise<void>) => {
    setBusy(true); setError('');
    try { await action(); setOwner(getAudioQueueOwner()); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); setOwner(getAudioQueueOwner()); }
    finally { setBusy(false); }
  };
  return <div className="fixed bottom-40 left-1/2 -translate-x-1/2 z-[60] max-w-[92vw] rounded-xl border border-amber-500/30 bg-white/95 dark:bg-zinc-900/95 px-4 py-3 shadow-2xl text-sm">
    <div className="font-semibold">Audio queue: {owner === 'server' ? 'server-owned' : owner === 'migrating' ? 'migration in progress' : 'browser-owned'}</div>
    {owner === 'browser' && <div className="text-xs text-zinc-500">Move {count} queued item{count === 1 ? '' : 's'} to the server. A versioned backup downloads first; browser data stays available for rollback.</div>}
    {owner === 'server' && <div className="text-xs text-zinc-500">This queue follows the server across reloads and tabs. Rollback pauses it and downloads a reconciliation export.</div>}
    {receipt && <div className="text-xs text-emerald-600">{receipt}</div>}
    {error && <div role="alert" className="text-xs text-red-500">{error}</div>}
    <div className="mt-2 flex gap-2">
      {owner === 'browser' && <>
        <button type="button" disabled={busy} onClick={() => run(async () => { const { receipt: r } = await migrateAudioQueue(token, 'hold'); setReceipt(`Imported ${r.imported} items; ${r.existing} already present. Pending work is held until Resume.`); })} className="rounded bg-amber-500/20 px-3 py-1">Back up and import</button>
      </>}
      {owner === 'server' && <button type="button" disabled={busy} onClick={() => run(async () => { await rollbackAudioQueue(token); setReceipt('Browser ownership restored. Pending work is held for review.'); })} className="rounded bg-amber-500/20 px-3 py-1">Export and roll back</button>}
    </div>
  </div>;
};
