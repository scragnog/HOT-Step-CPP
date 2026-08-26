/**
 * RestoredQueueBanner.tsx — asks before running a queue restored from storage.
 *
 * The audio generation queue is persisted in IndexedDB so it survives a reload,
 * a browser restart, or the machine being rebooted. It used to restart itself
 * on the next page load, which is right when you reloaded a moment ago and
 * wrong when a browser restores a tab by itself days later: a full render would
 * begin with nobody at the machine and nothing clicked (issue #100).
 *
 * Items that were already submitted still reconnect without asking. There is
 * a job running on the server either way, and reattaching to it loses nothing.
 * Only items that never left the browser wait here.
 */

import React from 'react';
import { ListRestart } from 'lucide-react';
import {
  useAudioGenQueueSelector,
  resumeRestoredQueue,
  discardRestoredQueue,
} from '../../stores/audioGenQueueStore';
import { useAuth } from '../../context/AuthContext';

export const RestoredQueueBanner: React.FC = () => {
  const { token } = useAuth();
  const count = useAudioGenQueueSelector(s => s.awaitingResume);

  if (count === 0) return null;

  return (
    <div className="fixed bottom-24 left-1/2 -translate-x-1/2 z-[60] max-w-[92vw]">
      <div className="flex items-center gap-3 px-4 py-3 rounded-xl border border-amber-500/30
                      bg-white/95 dark:bg-zinc-900/95 backdrop-blur shadow-2xl shadow-black/40">
        <ListRestart size={18} className="flex-shrink-0 text-amber-400" />
        <div className="min-w-0">
          <div className="text-sm text-zinc-800 dark:text-zinc-100">
            {count === 1
              ? '1 queued generation from a previous session'
              : `${count} queued generations from a previous session`}
          </div>
          <div className="text-[11px] text-zinc-500">
            Nothing is running. Start them, or clear them out.
          </div>
        </div>
        <button
          type="button"
          onClick={() => resumeRestoredQueue(token || undefined)}
          className="flex-shrink-0 px-3 py-1.5 rounded-lg text-xs font-semibold
                     bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 transition-colors"
        >
          Resume
        </button>
        <button
          type="button"
          onClick={discardRestoredQueue}
          className="flex-shrink-0 px-3 py-1.5 rounded-lg text-xs font-semibold
                     bg-white/[0.06] text-zinc-400 hover:bg-red-500/20 hover:text-red-300 transition-colors"
        >
          Discard
        </button>
      </div>
    </div>
  );
};
