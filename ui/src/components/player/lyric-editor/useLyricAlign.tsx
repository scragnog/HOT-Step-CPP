// useLyricAlign.tsx — starting an alignment, following it, and asking for the
// diff to be accepted.
//
// Every run is a preview. Alignment is minutes of separation, Whisper windows
// and a CTC forward, and what it produces is a set of placements that overwrite
// whatever was there, so the server writes nothing until the user has seen what
// would change. This hook owns the polling, the preview, the two answers
// (apply, discard), and the toasts that carry them.
//
// A 409 on apply means the file moved on while the diff was on screen. The
// confirmation is in the toast, and taking it re-sends the same apply with
// `force`, so nothing is overwritten by accident and nothing is lost by a stale
// preview.

import { useCallback, useEffect, useRef, useState } from 'react';
import { toast } from 'react-toastify';
import {
  applyLyricsAlign, cancelLyricsAlign, getLyricsAlignJob, latestLyricsAlignJob, startLyricsAlign,
  type LyricsAlignJob,
} from '@/services/api';
import type { RealignSettings } from '@/storage/lyricEditorSettings';

/** The pipeline's stage names, as a sentence fragment. */
const STAGE_LABEL: Record<string, string> = {
  starting: 'starting',
  queued: 'waiting for the lane',
  'queued-selection': 'waiting for the lane',
  'queued-aligner': 'waiting for the aligner',
  cut: 'cutting the selection',
  stem: 'separating the vocal stem',
  whisper: 'transcribing',
  align: 'aligning words',
  reconcile: 'matching words',
  polish: 'checking onsets',
  write: 'writing the sidecar',
  done: 'done',
};

const POLL_MS = 900;

/**
 * A stage as a sentence fragment, including the aligner queue's place in line.
 * The server sends `queued-aligner-2` for "one run is ahead of you", because a
 * wait with no number is indistinguishable from a hang.
 */
function stageLabel(stage: string): string {
  const known = STAGE_LABEL[stage];
  if (known) return known;
  const queued = /^queued-aligner(?:-(\d+))?$/.exec(stage);
  if (queued) {
    const place = Number(queued[1]);
    return place > 1 ? `waiting for the aligner (${place} in line)` : 'waiting for the aligner';
  }
  return stage;
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/** The HTTP status on a failed request, without narrowing through `any`. */
function errorStatus(err: unknown): number | undefined {
  if (typeof err !== 'object' || err === null || !('status' in err)) return undefined;
  const status = Number((err as { status?: unknown }).status);
  return Number.isFinite(status) ? status : undefined;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : '';
}

export interface AlignRunRange {
  fromLine: number;
  toLine: number;
  fromStart?: number;
  toStart?: number;
}

/** `polish` repairs the current timings; the other two produce them. */
type AlignMode = 'full' | 'range' | 'polish';

/** A finished run nobody has accepted yet: the diff and the job that holds it. */
export interface AlignPreview {
  jobId: string;
  /** "the whole track", "line 6" or "lines 6-9", for the panel's title. */
  where: string;
  result: NonNullable<LyricsAlignJob['result']>;
  /** Kept on screen after Apply, so the rows can still jump to each moved
   *  word — that review is the reason the diff exists. */
  applied?: boolean;
}

export interface LyricAlignOptions {
  audioUrl: string;
  /** Called once a preview has been applied and the sidecars rewritten. */
  onDone: () => Promise<void> | void;
}

export function useLyricAlign({ audioUrl, onDone }: LyricAlignOptions) {
  const [running, setRunning] = useState(false);
  const [stage, setStage] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<AlignPreview | null>(null);
  const [applying, setApplying] = useState(false);
  const alive = useRef(true);
  const jobRef = useRef<string | null>(null);
  // The toast is built inside `run`, before `run` and `cancel` exist; the refs
  // are what let its buttons reach them without the callbacks naming themselves.
  const runRef = useRef<((range: AlignRunRange | null, opts?: { force?: boolean }) => Promise<void>) | null>(null);
  const cancelRef = useRef<(() => Promise<void>) | null>(null);
  const applyRef = useRef<((opts?: { force?: boolean }) => Promise<void>) | null>(null);
  const previewRef = useRef<AlignPreview | null>(null);
  const applyingRef = useRef(false);

  previewRef.current = preview;
  applyingRef.current = applying;

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  /**
   * Re-attach to a run that is already going for this track.
   *
   * The run outlives the editor and the toast that announced it, so a song
   * switched away from and back — or a second window — has to pick the job up
   * again: otherwise the title shows nothing while Whisper works, and a diff
   * nobody saw is thrown away an hour later. Only the first mount attaches; a
   * run started from this editor owns the job from its first poll.
   */
  const attached = useRef(false);
  useEffect(() => {
    if (attached.current) return;
    attached.current = true;
    let dead = false;
    void (async () => {
      const job = await latestLyricsAlignJob(audioUrl).catch(() => null);
      if (dead || !job || jobRef.current) return;
      const attachWhere = job.result?.where ?? 'the whole track';
      if (job.status === 'running') {
        jobRef.current = job.jobId;
        setRunning(true);
        setStage(stageLabel(job.stage));
        let current: LyricsAlignJob | null = job;
        while (current && current.status === 'running' && !dead) {
          await delay(POLL_MS);
          current = await getLyricsAlignJob(job.jobId).catch(() => null);
        }
        jobRef.current = null;
        if (dead) return;
        setRunning(false);
        if (current?.status === 'succeeded' && current.result?.pending) {
          setPreview({ jobId: current.jobId, where: current.result.where ?? attachWhere, result: current.result });
          toast.info(`The run on ${attachWhere} finished. Review the diff.`);
        } else if (current?.status === 'failed') {
          setError(current.error ?? 'The alignment failed');
        }
        return;
      }
      // A finished preview is still a decision to make, and it is the reason the
      // server keeps one for an hour.
      if (job.status === 'succeeded' && job.result?.pending) {
        setPreview({ jobId: job.jobId, where: attachWhere, result: job.result });
      }
    })();
    return () => { dead = true; };
  }, [audioUrl]);

  const execute = useCallback(async (
    mode: AlignMode, range: AlignRunRange | null, opts: { force?: boolean } = {}, polish?: RealignSettings,
  ) => {
    if (jobRef.current) return;
    setRunning(true);
    setError(null);
    setPreview(null);
    setStage(STAGE_LABEL.starting);

    // One line is "line 6", not "lines 6-6": the numbers are 1-based positions
    // in the editor, which is the only place the user sees them.
    const where = !range
      ? 'the whole track'
      : range.fromLine === range.toLine
        ? `line ${range.fromLine + 1}`
        : `lines ${range.fromLine + 1}-${range.toLine + 1}`;
    const counted = (lines: number, words: number) =>
      `${lines} ${lines === 1 ? 'line' : 'lines'}, ${words} ${words === 1 ? 'word' : 'words'}`;
    // The loading toast carries Cancel at every stage, because the run can be
    // minutes long and the editor must not be a trap while it works.
    const progressToast = (label: string) => (
      <div className="flex items-center gap-3">
        <span>{`${mode === 'polish' ? `Repairing timings on ${where}` : `Aligning ${where}`}: ${label}`}</span>
        <button
          type="button"
          className="shrink-0 rounded-md border border-red-500 px-2 py-0.5 text-xs font-semibold text-red-500 hover:bg-red-500/10 dark:border-red-400/70 dark:text-red-400"
          onClick={() => { void cancelRef.current?.(); }}
        >
          Cancel
        </button>
      </div>
    );
    // Dismissible even while it runs: the run continues in the background, and
    // a toast nobody can close is the one thing that makes a long job a trap.
    const toastId = toast.loading(progressToast(STAGE_LABEL.starting), { closeButton: true, autoClose: false });

    try {
      let job: LyricsAlignJob | null = await startLyricsAlign(audioUrl, {
        mode,
        // A polish run repairs the whole sidecar; the other two take a range.
        ...(mode === 'polish' ? {} : (range ?? {})),
        // Every run is a preview: the diff is reviewed before anything is
        // written. Separation is not requested either — a cached stem is used
        // when one exists, and the mix is aligned otherwise.
        dryRun: true,
        force: opts.force === true,
        // Only a repair run has passes; the others produce timings rather than
        // judging them, and sending the object would just be ignored.
        ...(mode === 'polish' && polish ? { polish } : {}),
      });
      jobRef.current = job.jobId;

      while (job && job.status === 'running' && alive.current) {
        const label = stageLabel(job.stage);
        setStage(label);
        toast.update(toastId, { render: progressToast(label) });
        await delay(POLL_MS);
        if (!alive.current) return;
        job = await getLyricsAlignJob(job.jobId).catch(() => null);
      }
      if (!alive.current) return;
      if (!job) throw new Error('The alignment job is no longer on the server');

      if (job.status === 'succeeded') {
        const result = job.result;
        if (result?.pending) {
          // The job's own wording wins: a run that was started in another window
          // knows its range even when this one does not.
          setPreview({ jobId: job.jobId, where: result.where ?? where, result });
          const ready = mode === 'polish' ? 'Timing repairs ready' : 'Alignment ready';
          toast.update(toastId, {
            render: `${ready} on ${where}: ${counted(result.lineCount, result.wordCount)}, ` +
              `${result.wordsMoved} ${result.wordsMoved === 1 ? 'word' : 'words'} would move. Review the diff.`,
            type: 'info',
            isLoading: false,
            autoClose: 6000,
          });
        } else {
          // A run that wrote immediately is older than this editor expects;
          // adopt it rather than pretending there is something to review.
          await onDone();
          toast.update(toastId, {
            render: result
              ? `Lyrics aligned on ${where}: ${counted(result.lineCount, result.wordCount)} (${result.method})`
              : `Lyrics aligned on ${where}`,
            type: 'success',
            isLoading: false,
            autoClose: 6000,
          });
        }
        setStage(STAGE_LABEL.done);
      } else if (job.status === 'cancelled') {
        toast.update(toastId, {
          render: `${mode === 'polish' ? 'Repair' : 'Alignment'} cancelled on ${where}`,
          type: 'info',
          isLoading: false,
          autoClose: 4000,
        });
        setStage('');
      } else {
        throw new Error(job.error || 'The alignment failed');
      }
    } catch (err) {
      if (errorStatus(err) === 409) {
        toast.update(toastId, {
          render: (
            <div className="flex items-center gap-3">
              <span>{errorMessage(err) || 'These timings were placed by hand.'}</span>
              <button
                type="button"
                className="shrink-0 rounded-md border border-amber-500 px-2 py-0.5 text-xs font-semibold text-amber-500 hover:bg-amber-500/10"
                onClick={() => { toast.dismiss(toastId); void runRef.current?.(range, { force: true }); }}
              >
                Overwrite
              </button>
            </div>
          ),
          type: 'warning',
          isLoading: false,
          autoClose: 12000,
        });
      } else {
        const message = errorMessage(err) || 'The alignment failed';
        // A repair run with no saved timings has nothing to repair. Saying so
        // in red leaves the person stuck; offer the run that does place them.
        if (mode === 'polish' && /no timings to repair/i.test(message)) {
          setStage('');
          toast.update(toastId, {
            render: (
              <div className="flex items-center gap-3">
                <span>This track has no saved word timings yet, so there is nothing to repair.</span>
                <button
                  type="button"
                  className="shrink-0 rounded-md border border-pink-400 px-2 py-0.5 text-xs font-semibold text-pink-400 hover:bg-pink-500/10"
                  onClick={() => { toast.dismiss(toastId); void runRef.current?.(null); }}
                >
                  Align track
                </button>
              </div>
            ),
            type: 'info',
            isLoading: false,
            autoClose: 12000,
          });
          return;
        }
        setError(message);
        setStage('');
        toast.update(toastId, {
          render: `Alignment failed: ${message}`,
          type: 'error',
          isLoading: false,
          autoClose: 12000,
        });
      }
    } finally {
      jobRef.current = null;
      if (alive.current) setRunning(false);
    }
  }, [audioUrl, onDone]);

  /** Align the selection, or the whole track. Always a preview. */
  const run = useCallback(
    (range: AlignRunRange | null, opts: { force?: boolean } = {}) =>
      execute(range ? 'range' : 'full', range, opts),
    [execute],
  );

  /**
   * Re-run only the statistics pass over the current timings. No Whisper, no
   * aligner, no model: this is the loop for tuning the cut-back and snap rules,
   * and it previews like every other run.
   */
  const runPolish = useCallback(
    (polish?: RealignSettings) => execute('polish', null, {}, polish),
    [execute],
  );

  useEffect(() => { runRef.current = run; }, [run]);

  /** Ask the server to stop. Polling then reports the cancelled state. */
  const cancel = useCallback(async () => {
    const jobId = jobRef.current;
    if (!jobId) return;
    await cancelLyricsAlign(jobId).catch(() => null);
  }, []);

  useEffect(() => { cancelRef.current = cancel; }, [cancel]);

  /** Write the previewed diff. The only path from a run to the sidecar. */
  const apply = useCallback(async (opts: { force?: boolean } = {}) => {
    const pending = previewRef.current;
    if (!pending || applyingRef.current) return;
    applyingRef.current = true;
    setApplying(true);
    try {
      await applyLyricsAlign(pending.jobId, { force: opts.force === true });
      await onDone();
      // Not cleared: the strip stays as the receipt, and each row jumps to its
      // word at the time just written. Discard is what closes it.
      setPreview(prev => (prev ? { ...prev, applied: true } : prev));
      toast.success(`Timings applied on ${pending.where}`);
    } catch (err) {
      if (errorStatus(err) === 409) {
        toast.error(
          <div className="flex items-center gap-3">
            <span>{errorMessage(err) || 'The timings changed while you were reviewing.'}</span>
            <button
              type="button"
              className="shrink-0 rounded-md border border-amber-500 px-2 py-0.5 text-xs font-semibold text-amber-500 hover:bg-amber-500/10"
              onClick={() => { void applyRef.current?.({ force: true }); }}
            >
              Apply anyway
            </button>
          </div>,
          { autoClose: 12000 },
        );
      } else {
        toast.error(errorMessage(err) || 'Could not apply the alignment');
      }
    } finally {
      applyingRef.current = false;
      if (alive.current) setApplying(false);
    }
  }, [onDone]);

  useEffect(() => { applyRef.current = apply; }, [apply]);

  /** Throw the preview away. Nothing was written, so there is nothing to undo. */
  const discard = useCallback(() => {
    if (!previewRef.current) return;
    setPreview(null);
    toast.info('Alignment preview discarded');
  }, []);

  /**
   * The diff was written by a mix, not by the server's accept.
   *
   * The strip stays as the receipt; without this it would keep asking which
   * rows to take after the choice has already been saved.
   */
  const markApplied = useCallback(() => {
    setPreview(prev => (prev ? { ...prev, applied: true } : prev));
  }, []);

  return {
    data: { running, stage, error, preview, applying },
    run, runPolish, cancel, apply, discard, markApplied,
  };
}

export type LyricAlign = ReturnType<typeof useLyricAlign>;
