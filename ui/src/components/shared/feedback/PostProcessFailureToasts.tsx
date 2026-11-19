// PostProcessFailureToasts.tsx — failures that arrive while the dock is not
// being watched.
//
// A post-processing run is minutes long, so it is usually started and left. When
// it fails, the dock shows the error, but only there: a pass started from the
// Library, or from a menu that has since closed, fails silently unless someone
// happens to open the dock. A toast is the only surface that follows the run.
//
// The toast is also where the failure becomes actionable. A missing stage is a
// switch that can be turned on, so it offers "Enable … & retry" and re-queues
// the run in the same click. A memory failure has no switch to offer: nothing
// the app can turn on makes VRAM appear, so that toast says what to free instead
// of showing a button that would only fail again.
//
// One toast per failure, not per poll: the store keeps a failure until it is
// dismissed, and `announced` remembers which `(song, job, message)` triples have
// already been shown. A retry is a new job, so its own failure gets its own
// toast.

import React, { useEffect, useRef } from 'react';
import { toast } from 'react-toastify';
import {
  enablePpDependency, ppDependency, ppFailureKind, retryPostProcessing,
  usePostProcessRunList,
  type PpDependencyInfo,
} from '@/stores/postProcessStore';

const RETRY_BTN = 'mt-0.5 w-fit rounded-md border border-pink-400 px-2 py-0.5 text-[11px] '
  + 'font-semibold text-pink-400 hover:bg-pink-500/10';

export const PostProcessFailureToasts: React.FC = () => {
  const runs = usePostProcessRunList();
  const announced = useRef(new Set<string>());

  useEffect(() => {
    for (const run of runs) {
      if (run.status !== 'failed' || !run.error) continue;
      const key = `${run.songId}:${run.jobId ?? ''}:${run.error}`;
      if (announced.current.has(key)) continue;
      announced.current.add(key);

      const kind = ppFailureKind(run.error);
      const dependency: PpDependencyInfo | null = kind === 'dependency' ? ppDependency(run.error) : null;

      let id = 0 as unknown as ReturnType<typeof toast.error>;
      id = toast.error(
        <div className="flex max-w-[340px] flex-col gap-1">
          <span className="text-[11px] font-semibold">
            {run.title || 'Untitled'}: post-processing failed
          </span>
          <span className="text-[11px] opacity-90">{run.error}</span>
          {kind === 'resources' ? (
            <span className="text-[11px] opacity-90">
              There are not enough GPU resources to run this chain. Free up VRAM by closing another
              render or a GPU app, then run post-processing again.
            </span>
          ) : kind === 'dependency' ? (
            <span className="text-[11px] opacity-90">
              {dependency
                ? `The chain needs ${dependency.label}. Enable it and run the chain again?`
                : 'A model or plugin this chain needs is missing. Install it in Model Manager, then run post-processing again.'}
            </span>
          ) : null}
          {dependency && (
            <button
              type="button"
              className={RETRY_BTN}
              onClick={() => {
                toast.dismiss(id);
                enablePpDependency(dependency);
                void retryPostProcessing(run.songId, run.title).catch(() => {
                  // The retry's own failure reaches this component through the
                  // store and gets its own toast; nothing to add here.
                });
              }}
            >
              Enable {dependency.label} &amp; retry
            </button>
          )}
        </div>,
        // Stays until dismissed, and clicking the text does not take the
        // button away with it.
        { autoClose: false, closeOnClick: false },
      );
    }
  }, [runs]);

  return null;
};

export default PostProcessFailureToasts;
