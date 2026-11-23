// LyricEditorAlignmentReview.tsx — the run's sheet beside yours, row by row.
//
// Alignment never writes on its own. A run finishes with the sheet it would
// write, and this strip is the question: each row shows the run's version above
// the one on screen, and the person decides which of the two is kept. Apply
// then writes that mix — all of it the run's, or a choice row by row. Discard
// writes nothing at all.
//
// New goes above old because the run is the proposal and the file is the
// reference: reading down a row answers "what would change here". The rows that
// need no answer are not drawn at all — a re-align rebuilds its whole span, and
// the lines that come back identical are noise, not proposals.
//
// The repairs list is kept for the case where the server could not send a
// sheet (an older job): there is nothing to choose between then, and the diff
// is still worth reading.

import React, { useMemo, useState } from 'react';
import { Check, Sparkles, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { HintButton } from '@/components/shared/controls/HintButton';
import { BTN } from './editorStyles';
import { changedProposedRows, fmt, mergeProposed, type EditLine } from './timingModel';
import type { AlignPreview } from './useLyricAlign';

/** How many repairs are listed when there is no sheet to compare. */
const MAX_LISTED = 100;
/** How many paired rows are drawn. The rest are counted. */
const MAX_ROWS = 400;

/** The repair classes, in the words the editor uses. Exported for the timeline
 *  marker's tooltip, so the two never describe a repair differently. */
export const REPAIR_REASON: Record<string, string> = {
  repeat: 'reused from the earlier repeat',
  monotonic: 'out of order',
  'cut-back': 'cut back',
  snap: 'snapped to onset',
};

export interface LyricEditorAlignmentReviewProps {
  preview: AlignPreview;
  /** The sheet on screen: the review's old side, and what a row keeps. */
  current: EditLine[];
  applying?: boolean;
  /** The diff has been written: the rows become a receipt, not a question. */
  applied?: boolean;
  /** All-new: the server writes the run's sheet, diagnostics included. */
  onApplyNew: () => void;
  /** A mix: an ordinary edit, so it is one undo entry and autosaves. */
  onApplyMix: (lines: EditLine[], summary: { taken: number; kept: number }) => void;
  /** Take one row now; the other rows stay open for their own decisions. */
  onApplyLine?: (lineIndex: number) => void;
  /** Select a repaired word, seek just before it and scroll it into view. */
  onJump?: (ref: { line: number; word: number }) => void;
  /** Seek to a row's time, so the old and new placements can be heard in place. */
  onJumpTime?: (time: number) => void;
  /** Select a row's line, so the timeline expands that line into its diff lanes. */
  onFocusLine?: (lineIndex: number) => void;
  onDiscard: () => void;
}

export const LyricEditorAlignmentReview: React.FC<LyricEditorAlignmentReviewProps> = ({
  preview, current, applying, applied, onApplyNew, onApplyMix, onApplyLine, onJump, onJumpTime, onFocusLine,
  onDiscard,
}) => {
  const { t } = useTranslation();
  const { result } = preview;
  const proposed = useMemo(() => result.proposed ?? [], [result.proposed]);
  // Only the rows the run would change. A re-align rebuilds its span, so most
  // rows come back identical; listing them asked the person to choose between
  // a line and the same line.
  const changedRows = useMemo(() => changedProposedRows(current, proposed), [current, proposed]);
  // Which side each changed row keeps. New is the default, because the run is
  // the reason this strip is on screen; the person opts back to their own line.
  // An unchanged row is never taken: taking it would write the same line back.
  const [take, setTake] = useState<boolean[]>(() => {
    const next = new Array(proposed.length).fill(false);
    for (const index of changedRows) next[index] = true;
    return next;
  });
  // Rows decided one at a time. Applied rows were written as ordinary edits;
  // kept rows stay on screen and are excluded from the bulk Apply.
  const [appliedRows, setAppliedRows] = useState<Set<number>>(() => new Set());
  const [keptRows, setKeptRows] = useState<Set<number>>(() => new Set());
  const openRows = changedRows.filter(index => !appliedRows.has(index) && !keptRows.has(index));
  const takenCount = openRows.filter(index => take[index] !== false).length;
  const repairs = result.repairs ?? [];
  const listed = repairs.slice(0, MAX_LISTED);

  const summary = [
    `${result.lineCount} ${t('lyrics.editor.reviewLines', 'lines')}`,
    `${result.wordCount} ${t('lyrics.editor.reviewWords', 'words')}`,
    result.wordsMoved > 0
      ? t('lyrics.editor.reviewMoved', '{{count}} would move', { count: result.wordsMoved })
      : t('lyrics.editor.reviewNothingMoved', 'nothing would move'),
    result.method,
    result.vocalsIsolated
      ? t('lyrics.editor.reviewOnStem', 'on the vocal stem')
      : t('lyrics.editor.reviewOnMix', 'on the mix'),
  ].join(' · ');

  const apply = () => {
    if (openRows.length === 0) return;
    // Applied rows already have the run's line in the editor, so the bulk write
    // must not take them again; kept rows never take.
    const payload = take.map((yes, index) => (openRows.includes(index) ? yes !== false : false));
    if (takenCount === openRows.length && appliedRows.size === 0) {
      onApplyNew();
      return;
    }
    onApplyMix(mergeProposed(current, proposed, payload), {
      taken: takenCount, kept: openRows.length - takenCount,
    });
  };

  const setAll = (keepNew: boolean) => {
    setTake(prev => prev.map((yes, index) => (openRows.includes(index) ? keepNew : yes)));
  };

  /** Take this row now, as one ordinary edit. */
  const takeLine = (index: number) => {
    if (!onApplyLine) return;
    onApplyLine(index);
    setAppliedRows(prev => new Set(prev).add(index));
  };

  /** Keep the line on screen and stop asking about this row. */
  const keepLine = (index: number) => {
    setTake(prev => prev.map((yes, at) => (at === index ? false : yes)));
    setKeptRows(prev => new Set(prev).add(index));
  };

  return (
    <div className="flex shrink-0 flex-col gap-1 border-b border-amber-500/20 bg-amber-500/5 px-3 py-2 text-[11px]">
      <div className="flex flex-wrap items-center gap-2">
        <Sparkles size={12} className="text-amber-500" />
        <span className="font-semibold text-amber-600 dark:text-amber-400">
          {applied
            ? t('lyrics.editor.reviewAppliedTitle', 'Applied repairs')
            : t('lyrics.editor.reviewTitle', 'Alignment preview')}
          {preview.where ? ` · ${preview.where}` : ''}
        </span>
        <span className="opacity-70">{summary}</span>
        {proposed.length > 0 && !applied && (openRows.length > 0 ? (
          <span className="opacity-70">
            {t('lyrics.editor.reviewTaking', 'taking {{taken}} of {{rows}} rows from the run', {
              taken: takenCount, rows: openRows.length,
            })}
          </span>
        ) : changedRows.length > 0 ? (
          <span className="opacity-70">
            {t('lyrics.editor.reviewAllDecided', 'every changed row has been taken or kept')}
          </span>
        ) : (
          <span className="opacity-70">
            {t('lyrics.editor.reviewNoChanges', 'nothing to change: the run agrees with every line on screen')}
          </span>
        ))}
        <span className="ml-auto flex items-center gap-1.5">
          {applied ? (
            <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 font-semibold text-emerald-600 dark:text-emerald-400">
              {t('lyrics.editor.reviewApplied', 'Applied · the moved words are marked on the timeline')}
            </span>
          ) : (
            <>
              {changedRows.length > 0 && (
                <>
                  <HintButton
                    label={t('lyrics.editor.reviewAllNew', 'All new')}
                    info={t('lyrics.editor.reviewAllNewInfo', 'Takes the run’s version of every row. Apply then writes the whole sheet through the server, so the run’s own diagnostics travel with it.')}
                    className={BTN}
                    onClick={() => setAll(true)}
                  >
                    {t('lyrics.editor.reviewAllNew', 'All new')}
                  </HintButton>
                  <HintButton
                    label={t('lyrics.editor.reviewAllOld', 'All old')}
                    info={t('lyrics.editor.reviewAllOldInfo', 'Keeps every row as it is on screen. Nothing is written, which is the same as Discard.')}
                    className={BTN}
                    onClick={() => setAll(false)}
                  >
                    {t('lyrics.editor.reviewAllOld', 'All old')}
                  </HintButton>
                </>
              )}
              <HintButton
                label={t('lyrics.editor.reviewApply', 'Apply timings')}
                info={t('lyrics.editor.reviewApplyInfo', 'Writes the rows shown as taken: the run’s lines, the ones you kept, and nothing else. Until this is pressed nothing on disk has changed.')}
                className={BTN}
                onClick={apply}
                disabled={applying || openRows.length === 0}
              >
                <Check size={12} /> {applying
                  ? t('lyrics.editor.reviewApplying', 'Applying…')
                  : t('lyrics.editor.reviewApplyCount', 'Apply {{count}} lines', { count: takenCount })}
              </HintButton>
            </>
          )}
          {/* After Apply the same button only closes the receipt, so it stops
              asking about writes that have already happened. */}
          <HintButton
            label={applied
              ? t('lyrics.editor.reviewClose', 'Close review')
              : t('lyrics.editor.reviewDiscard', 'Discard')}
            info={applied
              ? t('lyrics.editor.reviewCloseInfo', 'Removes this receipt from above the timeline. The timings are already written; closing it changes nothing. The marks on the timeline stay until the next edit renumbers the words.')
              : t('lyrics.editor.reviewDiscardInfo', 'Throws the preview away. Nothing was written, so the timings stay exactly as they were and no undo entry is used.')}
            className={BTN}
            onClick={onDiscard}
            disabled={applying}
          >
            <X size={12} /> {applied
              ? t('lyrics.editor.reviewClose', 'Close review')
              : t('lyrics.editor.reviewDiscard', 'Discard')}
          </HintButton>
        </span>
      </div>

      {result.cutBack || result.snapped ? (
        <span className="opacity-70">
          {t('lyrics.editor.reviewSplit', '{{cut}} cut back · {{snap}} snapped to an onset · {{untouched}} untouched', {
            cut: result.cutBack ?? 0, snap: result.snapped ?? 0, untouched: result.untouched ?? 0,
          })}
        </span>
      ) : null}

      {result.handPlacedLost ? (
        <span className="text-red-500">
          {t('lyrics.editor.reviewHandLost', '{{count}} line(s) you had checked by ear could not be matched. Check them before applying.', {
            count: result.handPlacedLost,
          })}
        </span>
      ) : null}

      {changedRows.length > 0 ? (
        <div className="max-h-36 overflow-auto rounded border border-amber-500/20 bg-black/5 p-1 font-mono text-[10px] leading-4 dark:bg-black/20">
          {changedRows.slice(0, MAX_ROWS).map(index => {
            const fresh = proposed[index];
            const old = current[index];
            const keepsNew = take[index] !== false;
            return (
              <div
                key={index}
                className={`grid w-full grid-cols-[3.2rem_1fr_auto] gap-2 rounded px-1 py-0.5 hover:bg-amber-500/10 ${
                  appliedRows.has(index) ? 'bg-emerald-500/10' : keptRows.has(index) ? 'opacity-50' : ''
                }`}
              >
                <span className="pt-0.5 text-[9px] uppercase tracking-wider opacity-60">
                  line {index + 1}
                </span>
                <button
                  type="button"
                  onClick={() => {
                    // Switching the row and jumping to the side just chosen are one
                    // gesture: the two lines are the same words at two times, and
                    // hearing each in place is how a person decides between them.
                    const nowNew = !keepsNew;
                    setTake(prev => prev.map((yes, at) => (at === index ? !yes : yes)));
                    onJumpTime?.(nowNew ? fresh.start : (old?.start ?? fresh.start));
                    // Selecting the line is what asks the timeline for the two
                    // lanes; without it only the strip above would move.
                    onFocusLine?.(index);
                  }}
                  disabled={applied || appliedRows.has(index) || keptRows.has(index)}
                  title={applied
                    ? ''
                    : t('lyrics.editor.reviewRowHint', 'Click to switch this row between the run’s version and the one on screen; the timeline jumps to the one you chose, so it can be played there.')}
                  className="flex flex-col gap-0.5 text-left disabled:cursor-default"
                >
                  {/* New above: the run's line, struck through once it is not taken. */}
                  <span className={`flex gap-2 ${keepsNew ? '' : 'line-through opacity-40'}`}>
                    <span className="w-12 shrink-0 text-pink-400">new</span>
                    <span className="w-12 shrink-0 opacity-70">{fmt(fresh.start)}</span>
                    <span className="truncate">{fresh.text}</span>
                  </span>
                  {/* Old below: what is on screen now. */}
                  <span className={`flex gap-2 ${keepsNew ? 'opacity-45' : ''}`}>
                    <span className="w-12 shrink-0 text-zinc-400">old</span>
                    <span className="w-12 shrink-0 opacity-70">{old ? fmt(old.start) : '—'}</span>
                    <span className="truncate">{old?.text ?? t('lyrics.editor.reviewNoOld', 'nothing here')}</span>
                  </span>
                </button>
                {/* One row at a time: Take writes this line as an ordinary edit;
                    Keep stops asking and leaves the others open. */}
                <span className="flex items-center gap-1 self-center">
                  {appliedRows.has(index) ? (
                    <span className="text-[9px] font-semibold uppercase tracking-wider text-emerald-500">
                      {t('lyrics.editor.reviewLineTaken', 'taken')}
                    </span>
                  ) : keptRows.has(index) ? (
                    <span className="text-[9px] font-semibold uppercase tracking-wider opacity-60">
                      {t('lyrics.editor.reviewLineKept', 'kept')}
                    </span>
                  ) : (
                    <>
                      <HintButton
                        label={t('lyrics.editor.reviewTakeLine', 'Take this line')}
                        info={t('lyrics.editor.reviewTakeLineInfo', 'Writes this one line from the run as an ordinary edit, without waiting for the rest. The other rows stay open, and Apply later writes only those still undecided.')}
                        className={BTN}
                        onClick={() => takeLine(index)}
                        disabled={applied || !onApplyLine}
                      >
                        <Check size={10} /> {t('lyrics.editor.reviewTake', 'Take')}
                      </HintButton>
                      <HintButton
                        label={t('lyrics.editor.reviewKeepLine', 'Keep this line')}
                        info={t('lyrics.editor.reviewKeepLineInfo', 'Leaves this line exactly as it is on screen and stops asking about it. The other rows can still be taken, one at a time or with Apply.')}
                        className={BTN}
                        onClick={() => keepLine(index)}
                        disabled={applied}
                      >
                        <X size={10} /> {t('lyrics.editor.reviewKeep', 'Keep')}
                      </HintButton>
                    </>
                  )}
                </span>
              </div>
            );
          })}
          {changedRows.length > MAX_ROWS && (
            <div className="opacity-60">
              {t('lyrics.editor.reviewMoreRows', '+{{count}} more rows, all taken from the run', {
                count: changedRows.length - MAX_ROWS,
              })}
            </div>
          )}
        </div>
      ) : listed.length > 0 ? (
        <div className="max-h-24 overflow-auto rounded border border-amber-500/20 bg-black/5 p-1 font-mono text-[10px] leading-4 dark:bg-black/20">
          {listed.map((repair, index) => (
            <HintButton
              key={`${repair.line}:${repair.word}:${index}`}
              label={t('lyrics.editor.reviewJumpLabel', 'Line {{line}}, word {{word}}', {
                line: repair.line + 1, word: repair.word + 1,
              })}
              info={t(
                'lyrics.editor.reviewJump',
                'Selects this word, seeks just before it and scrolls the timeline to it, so you can hear the repair.',
              )}
              onClick={() => onJump?.({ line: repair.line, word: repair.word })}
              disabled={!onJump}
              rootClassName="w-full"
              className="flex w-full gap-2 text-left hover:bg-amber-500/10 disabled:cursor-default"
            >
              <span className="w-14 shrink-0 opacity-60">line {repair.line + 1}</span>
              <span className="w-12 shrink-0 opacity-60">#{repair.word + 1}</span>
              <span className="shrink-0">{fmt(repair.from.start)} → {fmt(repair.to.start)}</span>
              <span className="shrink-0 opacity-60">{REPAIR_REASON[repair.reason] ?? repair.reason}</span>
            </HintButton>
          ))}
          {repairs.length > listed.length && (
            <div className="opacity-60">
              {t('lyrics.editor.reviewMore', '+{{count}} more', { count: repairs.length - listed.length })}
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
};

export default LyricEditorAlignmentReview;
