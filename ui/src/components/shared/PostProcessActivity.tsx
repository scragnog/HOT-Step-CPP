// PostProcessActivity.tsx — live view of after-the-fact post-processing runs.
//
// A pass takes minutes and runs entirely server-side. Without this panel the
// only sign one was happening was the 3-dot glyph on the row you clicked
// turning into a spinner — so firing four of them and walking away looked
// exactly like the feature doing nothing at all. That is the bug this fixes.
//
// Finished runs stay until dismissed rather than fading on a timer, so a pass
// that completed while you were on another page still leaves evidence.

import React from 'react';
import { Loader2, CheckCircle2, XCircle, X, Sparkles } from 'lucide-react';
import { usePostProcessRunList, dismissRun } from '../../stores/postProcessStore';

export const PostProcessActivity: React.FC = () => {
  const runs = usePostProcessRunList();
  if (runs.length === 0) return null;

  return (
    <div className="px-2 py-1.5 space-y-1">
      {runs.map(run => {
        const busy = run.status === 'starting' || run.status === 'pending' || run.status === 'running';
        const failed = run.status === 'failed';
        const done = run.status === 'succeeded';

        return (
          <div
            key={run.songId}
            className={`flex items-center gap-2 rounded-lg px-2 py-1.5 border ${
              failed
                ? 'border-red-500/30 bg-red-500/[0.07]'
                : done
                  ? 'border-emerald-500/30 bg-emerald-500/[0.07]'
                  : 'border-violet-500/30 bg-violet-500/[0.07]'
            }`}
          >
            <span className="flex-shrink-0">
              {busy && <Loader2 className="w-3.5 h-3.5 text-violet-400 animate-spin" />}
              {done && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />}
              {failed && <XCircle className="w-3.5 h-3.5 text-red-400" />}
            </span>

            <div className="flex-1 min-w-0">
              <p className="text-[11px] font-medium text-zinc-800 dark:text-zinc-200 truncate leading-tight">
                {run.title || 'Untitled'}
              </p>
              <p
                className={`text-[9px] truncate leading-tight ${
                  failed ? 'text-red-400' : done ? 'text-emerald-400' : 'text-zinc-500'
                }`}
                title={run.error || run.stage}
              >
                {failed
                  ? (run.error || 'Failed')
                  : done
                    ? 'Post-processing complete'
                    : (run.stage || 'Working...')}
              </p>
            </div>

            {/* Only a finished run can be dismissed — hiding a running one would
                just lose track of GPU work that is still happening. */}
            {!busy && (
              <button
                onClick={() => dismissRun(run.songId)}
                className="p-0.5 rounded text-zinc-600 hover:text-white transition-colors flex-shrink-0"
                title="Dismiss"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
};

/** Count of runs still working — drives the section badge. */
export function usePostProcessActiveCount(): number {
  const runs = usePostProcessRunList();
  return runs.filter(r =>
    r.status === 'starting' || r.status === 'pending' || r.status === 'running'
  ).length;
}

export { Sparkles as PostProcessIcon };

/**
 * Always-visible dock for post-processing runs.
 *
 * Mounted once at the app root rather than inside ActivitySidebar, because a
 * pass can be started from anywhere — including the Library, which has no
 * activity sidebar at all. Tying the only progress indicator to a panel that
 * half the app does not render is how four runs completed with nothing to show
 * for them.
 *
 * Sits above the player bar and renders nothing when idle.
 */
export const PostProcessDock: React.FC = () => {
  const runs = usePostProcessRunList();
  if (runs.length === 0) return null;

  return (
    <div
      data-portal-layer
      className="fixed bottom-28 right-4 z-[120] w-[280px] max-h-[40vh] overflow-y-auto
                 rounded-xl border border-zinc-300 dark:border-white/10
                 bg-white/95 dark:bg-zinc-900/95 backdrop-blur-sm shadow-xl"
    >
      <div className="flex items-center gap-1.5 px-2.5 pt-2 pb-1 text-[10px] font-semibold
                      uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
        <Sparkles className="w-3 h-3 text-violet-400" />
        Post-Processing
      </div>
      <PostProcessActivity />
    </div>
  );
};
