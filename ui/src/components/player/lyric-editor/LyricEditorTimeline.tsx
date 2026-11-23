// LyricEditorTimeline.tsx — the scrolling map: ruler, one row per line, a chip
// per word, the drag readout, the playhead, and the paging arrows.
//
// It draws. What a gesture may do to a word is `timingModel`'s rule and where
// the pointer is relative to the window is `useWordGestures`' problem; this
// component only turns the two into elements.
//
// Chips and resize handles keep a plain `title` instead of the house hover card:
// they are hundreds of drag targets inside a scrolling pane, and a portalled
// card per chip would cost more than it explains. The name and times are on the
// chip and in its aria-label. The two line buttons — tick and select — are one
// per line, so they use the house card like every other button in the editor.
//
// A chip is also where a word gets retyped: a double-click (or F2 on a focused
// chip) opens a text box over it, because correcting a misheard word and moving
// its timing are the same job done on the same syllable. Words the aligner was
// unsure of carry a dotted underline, so the ones worth listening to are
// visible without hunting.
//
// While an alignment preview is open, the selected line grows a second lane
// above its words, carrying the run's placements as dashed amber chips on the
// same time scale. The run is on top and the line on screen below, the same
// way the review strip reads. Two lanes rather than two chips on top of each
// other: a moved word overlaps what it replaces, and an overlap shows neither
// shape. Only the selected line is expanded — every changed row is stripes.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { HintButton } from '@/components/shared/controls/HintButton';
import { PAN_BTN } from './editorStyles';
import {
  BOTTOM_PAD, PAN_END_PAD_PX, ROW_H, RULER_H, fmt, refKey, rulerTicks, sameProposedLine, selectedKeys,
  type EditLine, type ProposedLine, type SelectionSpan, type WordRef,
} from './timingModel';
import type { DragLabel, WordGestures } from './useWordGestures';

/** A chip never shrinks below this, so a very short word stays clickable. */
const MIN_CHIP_PX = 10;
/** A row with the proposal lane above it: two 22px chips and the gaps. */
const DIFF_ROW_H = ROW_H + 24;
/** How far the current-line lane drops in an expanded row, clearing the run's. */
const DIFF_LANE = 24;
/** A signed seconds value at the preview's two-decimal resolution. Rounding
 *  first keeps a value like -0.001 from printing as `-0.00`. */
const signed = (value: number): string => {
  const rounded = Math.round(value * 100) / 100;
  return `${rounded < 0 ? '-' : '+'}${Math.abs(rounded).toFixed(2)}`;
};

/** The layer a row is drawn as: its own tag, else its first tagged word. */
function lineLayerOf(line: EditLine): 'lead' | 'backing' | 'adlib' {
  if (line.layer === 'backing' || line.layer === 'adlib') return line.layer;
  const tagged = line.words.find(w => w.layer === 'backing' || w.layer === 'adlib');
  return tagged?.layer === 'backing' ? 'backing' : tagged?.layer === 'adlib' ? 'adlib' : 'lead';
}

/**
 * The chip's text box, opened on a double-click or F2.
 *
 * Retyping a misheard word used to mean selecting it and then finding the word
 * field in the toolbar above: two targets for one word. Editing on the chip is
 * the same correction with the pointer already on the word. The row's field
 * stays for a clean single-word edit and for keyboard-only use.
 */
const ChipEditor: React.FC<{
  text: string;
  left: number;
  width: number;
  label: string;
  onCommit: (text: string) => void;
  onCancel: () => void;
  top: number;
}> = ({ text, left, top, width, label, onCommit, onCancel }) => {
  const [draft, setDraft] = useState(text);
  const cancelled = useRef(false);
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    ref.current?.focus();
    ref.current?.select();
  }, []);

  const commit = () => {
    const next = draft.replace(/\s+/g, ' ').trim();
    if (next && next !== text) onCommit(next);
    else onCancel();
  };

  return (
    <input
      ref={ref}
      value={draft}
      onChange={e => setDraft(e.target.value)}
      onBlur={() => { if (!cancelled.current) commit(); }}
      onPointerDown={e => e.stopPropagation()}
      onKeyDown={e => {
        if (e.key === 'Enter') { e.preventDefault(); e.currentTarget.blur(); }
        if (e.key === 'Escape') { cancelled.current = true; e.preventDefault(); onCancel(); }
      }}
      aria-label={label}
      className="pointer-events-auto absolute z-40 h-[22px] rounded border border-pink-400 bg-white px-1 text-[10px] text-zinc-900 shadow outline-none dark:bg-zinc-900 dark:text-zinc-100"
      style={{ left, top, width }}
    />
  );
};

export interface LyricEditorTimelineProps {
  lines: EditLine[];
  /** The selected span, not one word: a range is what re-alignment works on. */
  selection: SelectionSpan | null;
  /** Select every word on a line; `extend` is a shift-click. */
  onLineSelect: (lineIndex: number, extend: boolean) => void;
  /** Tick a line as placed by ear. The fixture store keeps only those lines. */
  onLineCheck: (lineIndex: number) => void;
  /** Retype a word where it is. Opened by a double-click or F2 on its chip. */
  onRenameWord?: (ref: WordRef, text: string) => void;
  /** Words the aligner was unsure of, keyed `line:word`, underlined here. */
  doubtful?: ReadonlySet<string>;
  /** Words the recogniser heard but the sheet never asked for, marked apart. */
  adlib?: ReadonlySet<string>;
  /** Words a just-applied repair moved, keyed `line:word`, with its reason. */
  repaired?: ReadonlyMap<string, string>;
  /** A run's sheet while its preview is open: each row draws these words over
   *  the current ones at their proposed times, so the change is read in place. */
  proposed?: ProposedLine[] | null;
  currentTime: number;
  total: number;
  pxPerSec: number;
  scrollerRef: React.RefObject<HTMLDivElement | null>;
  onScrollLeft: (left: number) => void;
  /** The readout for the gesture in flight, from `useWordGestures().data`. */
  dragLabel: DragLabel | null;
  /** The gesture handlers: drag, resize, pan. This row cannot reach the state. */
  gestures: WordGestures;
  canPageLeft: boolean;
  canPageRight: boolean;
  onPage: (dir: -1 | 1) => void;
}

export const LyricEditorTimeline: React.FC<LyricEditorTimelineProps> = ({
  lines, selection, onLineSelect, onLineCheck, onRenameWord, doubtful, adlib, repaired, proposed, currentTime,
  total, pxPerSec, scrollerRef, onScrollLeft, dragLabel, gestures, canPageLeft, canPageRight,
  onPage,
}) => {
  const { t } = useTranslation();
  /** The chip whose text box is open, if any. */
  const [editing, setEditing] = useState<{ line: number; word: number } | null>(null);
  // A double-press is detected here rather than through `dblclick`: the chip's
  // pointerdown is cancelled by the drag gesture, and a cancelled pointerdown
  // may stop the browser's own double-click from arriving at all.
  const press = useRef<{ key: string; at: number } | null>(null);
  // The ruler is this row's own concern: it is thinned to the current zoom, so
  // the labels exist only where they are drawn.
  const ticks = useMemo(() => rulerTicks(total, pxPerSec), [total, pxPerSec]);
  // One Set for the whole row strip: selected state is read per chip and a
  // linear search of the selection inside a render would be quadratic.
  const active = useMemo(() => selectedKeys(lines, selection), [lines, selection]);
  const focusKey = selection ? refKey(selection.focus) : '';
  // The row whose proposal is on show: the selected line, and only while the
  // run actually changes it. The comparison follows the selection rather than
  // drawing every changed row, which reads as noise at timeline widths.
  const selected = selection ? lines[selection.focus.line] : undefined;
  const proposedLine = selection ? proposed?.[selection.focus.line] : undefined;
  const diffLine = selection && proposedLine && proposedLine.words.length > 0
    && (!selected || !sameProposedLine(selected, proposedLine))
    ? selection.focus.line
    : null;
  // The run and the file can disagree on word count — an insertion or a drop —
  // so pairing by index would report a delta against a different word. Pair by
  // text, nth occurrence; an unmatched word is new on this line.
  const paired = useMemo(() => {
    const out: Array<EditLine['words'][number] | undefined> = [];
    if (!selected || !proposedLine) return out;
    const used = new Set<number>();
    for (const fresh of proposedLine.words) {
      let match: EditLine['words'][number] | undefined;
      for (let i = 0; i < selected.words.length; i += 1) {
        const word = selected.words[i];
        if (!used.has(i) && word.word === fresh.word) {
          match = word;
          used.add(i);
          break;
        }
      }
      out.push(match);
    }
    return out;
  }, [selected, proposedLine]);
  // `proposedToLine` recomputes a line's bounds rather than trusting the
  // server's `start`, so the offset is measured from the earliest word too.
  const proposalStart = proposedLine && proposedLine.words.length > 0
    ? Math.min(...proposedLine.words.map(word => word.start))
    : 0;
  // That row carries a second lane, so every row below it moves down by the
  // extra height: tops are a prefix sum, not `index * ROW_H`.
  const rows = useMemo(() => {
    const heights = lines.map((_, index) => (index === diffLine ? DIFF_ROW_H : ROW_H));
    const tops: number[] = [];
    let y = 0;
    for (const height of heights) {
      tops.push(y);
      y += height;
    }
    return { heights, tops, total: y };
  }, [lines, diffLine]);
  /** The current-line lane drops in the expanded row; every other row is flat. */
  const laneTopFor = (index: number) => (index === diffLine ? DIFF_LANE : 0);

  return (
    <div className="relative flex min-h-[240px] flex-1">
      <div
        ref={scrollerRef}
        onScroll={e => onScrollLeft(e.currentTarget.scrollLeft)}
        // The pan's capture lives on this pane, so the move and release
        // handlers have to be here too: captured events are retargeted to the
        // pane, not to the content the pointer went down on.
        onPointerMove={gestures.movePan}
        onPointerUp={gestures.endPan}
        onPointerCancel={gestures.cancelPan}
        onLostPointerCapture={gestures.cancelPan}
        className="relative flex-1 overflow-auto"
        title={t('lyrics.editor.panHint', 'Drag the timeline to pan, click to seek')}
      >
        {lines.length === 0 ? (
          <div className="p-6 text-center text-xs text-zinc-500">
            {t('lyrics.editor.noWords', 'This track has no word timings to edit. Run Retranscribe Lyrics first.')}
          </div>
        ) : (
          <div
            className="relative cursor-grab active:cursor-grabbing"
            style={{
              width: Math.max(1, total * pxPerSec + PAN_END_PAD_PX),
              height: RULER_H + rows.total + BOTTOM_PAD,
            }}
            onPointerDown={gestures.startPan}
          >
            {ticks.map(time => (
              <div
                key={time}
                className="pointer-events-none absolute top-0 border-l border-zinc-300 text-[9px] tabular-nums text-zinc-500 dark:border-white/15"
                style={{ left: time * pxPerSec, height: RULER_H }}
              >
                <span className="absolute left-1 top-1 whitespace-nowrap">{fmt(time)}</span>
              </div>
            ))}

            {lines.map((line, li) => (
              <div
                key={li}
                data-line-index={li}
                className="pointer-events-none absolute left-0 right-0 border-b border-zinc-100 dark:border-white/5"
                style={{ top: RULER_H + rows.tops[li], height: rows.heights[li] }}
              >
                {line.checked === true && line.words.length > 0 && (
                  // The locked zone: everything inside this strip was placed by
                  // ear, and a re-align moves the words around it, never it.
                  <div
                    aria-hidden
                    className="pointer-events-none absolute h-[26px] rounded border border-emerald-400/50 bg-emerald-400/10"
                    style={{
                      top: 1 + laneTopFor(li),
                      left: line.start * pxPerSec,
                      width: Math.max(2, (line.end - line.start) * pxPerSec),
                    }}
                  />
                )}
                {line.words.length === 0 && (
                  <span
                    className="absolute whitespace-nowrap text-[10px] italic text-zinc-500"
                    style={{ top: 7 + laneTopFor(li), left: line.start * pxPerSec }}
                    title={t('lyrics.editor.untimedLine', 'This line has no word timings, so it can only be kept as it is')}
                  >
                    {line.text}
                  </span>
                )}
                {line.words.length > 0 && (
                  <div
                    className="pointer-events-auto absolute z-10"
                    style={{ top: 8 + laneTopFor(li), left: Math.max(0, line.start * pxPerSec - 34) }}
                    onPointerDown={e => e.stopPropagation()}
                  >
                    <HintButton
                      label={t('lyrics.editor.markCheckedLabel', 'Placed by ear')}
                      info={t(
                        'lyrics.editor.markCheckedInfo',
                        'Tick a line once you have listened to it and its timing is right. A re-align keeps a ticked line’s words where they are and moves the other words around them, and only ticked lines are saved to the alignment fixtures — an unticked line is a machine guess, not a reference.',
                      )}
                      onClick={() => onLineCheck(li)}
                      ariaLabel={t('lyrics.editor.markChecked', 'Mark this line as placed by ear')}
                      ariaPressed={!!line.checked}
                      className={`flex h-[14px] w-[14px] items-center justify-center rounded border text-[9px] leading-none ${
                        line.checked
                          ? 'border-emerald-400 bg-emerald-500/20 text-emerald-500'
                          : 'border-zinc-300 bg-white/80 text-zinc-400 hover:border-emerald-400 hover:text-emerald-500 dark:border-white/15 dark:bg-zinc-900/80'
                      }`}
                    >
                      ✓
                    </HintButton>
                  </div>
                )}
                {line.words.length > 0 && (
                  <div
                    className="pointer-events-auto absolute z-10"
                    style={{ top: 8 + laneTopFor(li), left: Math.max(0, line.start * pxPerSec - 16) }}
                    onPointerDown={e => e.stopPropagation()}
                  >
                    <HintButton
                      label={t('lyrics.editor.selectLineLabel', 'Select line')}
                      info={t(
                        'lyrics.editor.selectLine',
                        'Selects every word on this line, which is the span Re-align works on. Shift-click another line to extend the selection.',
                      )}
                      onClick={e => onLineSelect(li, e.shiftKey)}
                      ariaLabel={t('lyrics.editor.selectLineAria', 'Select every word on this line')}
                      className="flex h-[14px] w-[14px] items-center justify-center rounded border border-zinc-300 bg-white/80 text-[9px] leading-none text-zinc-500 hover:border-pink-400 hover:text-pink-500 dark:border-white/15 dark:bg-zinc-900/80 dark:text-zinc-400"
                    >
                      ≡
                    </HintButton>
                  </div>
                )}
                {line.words.length > 0 && lineLayerOf(line) !== 'lead' && (
                  // A duplicate row under the lead is invisible otherwise: the
                  // badge says why this line may overlap instead of pushing.
                  <div
                    className="pointer-events-none absolute z-10 whitespace-nowrap rounded border border-sky-400/60 bg-sky-500/10 px-1 text-[9px] font-semibold uppercase leading-[14px] tracking-wide text-sky-500"
                    style={{ top: 8 + laneTopFor(li), left: line.end * pxPerSec + 6 }}
                  >
                    {lineLayerOf(line) === 'backing'
                      ? t('lyrics.editor.layerBacking', 'Backing')
                      : t('lyrics.editor.layerAdlib', 'Ad-lib')}
                  </div>
                )}
                {line.words.map((w, wi) => {
                  const key = `${li}:${wi}`;
                  const isFocus = focusKey === key;
                  const isSelected = active.has(key);
                  // A word a just-applied statistics pass moved keeps a mark
                  // until the array changes underneath it (`repaired` is empty
                  // once the editor is dirty, because the indices would lie).
                  const repair = repaired?.get(key);
                  const isDoubtful = doubtful?.has(key) ?? false;
                  const isAdlib = adlib?.has(key) ?? false;
                  // Ad-libs already have their own (heard, not scripted) mark;
                  // backing is the written harmony, so it gets its own tint.
                  const isBacking = !isAdlib && (w.layer === 'backing' || lineLayerOf(line) === 'backing');
                  return (
                    <button
                      key={wi}
                      type="button"
                      onPointerDown={e => {
                        const key = `${li}:${wi}`;
                        const now = performance.now();
                        const last = press.current;
                        press.current = { key, at: now };
                        if (onRenameWord && e.button === 0 && last?.key === key && now - last.at < 450) {
                          // The second press opens the text box instead of a drag:
                          // two presses on one syllable is "retype", never "move
                          // it twice".
                          press.current = null;
                          e.preventDefault();
                          e.stopPropagation();
                          setEditing({ line: li, word: wi });
                          return;
                        }
                        gestures.startDrag(e, li, wi);
                      }}
                      onPointerMove={gestures.dragMove}
                      onPointerUp={gestures.endDrag}
                      onPointerCancel={gestures.endDrag}
                      aria-pressed={isSelected}
                      aria-label={`${w.word}, ${fmt(w.start)} to ${fmt(w.end)}`}
                      onKeyDown={e => {
                        if (e.key === 'F2' && onRenameWord) {
                          e.preventDefault();
                          setEditing({ line: li, word: wi });
                        }
                      }}
                      className={`pointer-events-auto absolute h-[22px] cursor-grab overflow-hidden rounded px-1 text-left text-[10px] leading-[22px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-300 ${
                        isFocus
                          ? 'z-20 bg-pink-500 text-white ring-2 ring-pink-300'
                          : isSelected
                            ? 'z-10 bg-pink-500/40 text-white ring-1 ring-pink-400'
                            : isBacking
                              ? 'bg-sky-500/15 text-sky-300 hover:bg-sky-500/30'
                              : 'bg-pink-500/15 text-pink-300 hover:bg-pink-500/30'
                      }${repair ? ' ring-1 ring-amber-400' : ''}`}
                      style={{
                        top: 3 + laneTopFor(li),
                        left: w.start * pxPerSec,
                        width: Math.max(MIN_CHIP_PX, (w.end - w.start) * pxPerSec),
                      }}
                      title={`${w.word} · ${fmt(w.start)} → ${fmt(w.end)}${repair ? ` · repaired: ${repair}` : ''}${isAdlib ? ' · heard, not in the lyrics' : ''}${isBacking ? ' · backing vocal' : ''}${onRenameWord ? ' · double-click to retype' : ''}`}
                    >
                      <span className={`pointer-events-none block truncate${
                        isDoubtful
                          ? ' underline decoration-dotted decoration-amber-400 underline-offset-2'
                          : isAdlib
                            ? ' underline decoration-dashed decoration-violet-400 underline-offset-2'
                            : ''
                      }`}>{w.word}</span>
                      {isFocus && (
                        <>
                          <span
                            onPointerDown={e => gestures.startResize(e, li, wi, 'left')}
                            onPointerMove={gestures.resizeMove}
                            onPointerUp={gestures.endResize}
                            onPointerCancel={gestures.endResize}
                            className="pointer-events-auto absolute left-0 top-0 z-30 h-full w-[6px] cursor-ew-resize rounded-l bg-white/80"
                            title={t('lyrics.editor.resizeLeft', 'Drag to move the start of this word')}
                          />
                          <span
                            onPointerDown={e => gestures.startResize(e, li, wi, 'right')}
                            onPointerMove={gestures.resizeMove}
                            onPointerUp={gestures.endResize}
                            onPointerCancel={gestures.endResize}
                            className="pointer-events-auto absolute right-0 top-0 z-30 h-full w-[6px] cursor-ew-resize rounded-r bg-white/80"
                            title={t('lyrics.editor.resizeRight', 'Drag to change how long this word lasts')}
                          />
                        </>
                      )}
                    </button>
                  );
                })}
                {/* The selected row's proposal, one lane above the current
                    words on the same time scale: start is position, duration
                    is width, so the two shapes compare directly. */}
                {li === diffLine && proposedLine && proposedLine.words.map((w, wi) => {
                  const current = paired[wi];
                  const duration = Math.max(0, w.end - w.start);
                  const backing = (w.layer ?? 'lead') === 'backing';
                  return (
                    <span
                      key={`proposed-${wi}`}
                      title={[
                        `${t('lyrics.editor.reviewProposedWord', 'Proposed')} “${w.word}”`,
                        `start ${fmt(w.start)}`,
                        `duration ${duration.toFixed(2)}s`,
                        `offset ${signed(w.start - proposalStart)}s from the line`,
                        backing ? t('lyrics.editor.reviewBacking', 'backing vocal') : '',
                        current
                          ? `current “${current.word}” at ${fmt(current.start)} · move ${signed(w.start - current.start)}s`
                          : t('lyrics.editor.reviewNoCounterpart', 'not on this line now'),
                      ].filter(Boolean).join(' · ')}
                      className={`pointer-events-auto absolute top-[2px] z-10 h-[22px] cursor-default select-none overflow-hidden rounded border border-dashed px-1 text-[10px] leading-[22px] ${
                        backing
                          ? 'border-sky-400 bg-sky-400/15 text-sky-700 dark:text-sky-300'
                          : 'border-amber-400 bg-amber-400/15 text-amber-700 dark:text-amber-200'
                      }`}
                      style={{ left: w.start * pxPerSec, width: Math.max(MIN_CHIP_PX, duration * pxPerSec) }}
                    >
                      <span className="pointer-events-none block truncate">{w.word}</span>
                    </span>
                  );
                })}
                {editing?.line === li && onRenameWord && (() => {
                  const w = line.words[editing.word];
                  if (!w) return null;
                  return (
                    <ChipEditor
                      key={`${li}:${editing.word}`}
                      text={w.word}
                      left={w.start * pxPerSec}
                      top={3 + laneTopFor(li)}
                      width={Math.max(90, Math.min(180, (w.end - w.start) * pxPerSec + 40))}
                      label={t('lyrics.editor.renameWord', 'Word text')}
                      onCommit={text => { onRenameWord({ line: li, word: editing.word }, text); setEditing(null); }}
                      onCancel={() => setEditing(null)}
                    />
                  );
                })()}
              </div>
            ))}

            {dragLabel && (
              <>
                <div
                  className="pointer-events-none absolute z-20 w-px bg-pink-300/80"
                  style={{
                    left: dragLabel.start * pxPerSec,
                    top: RULER_H + (rows.tops[dragLabel.line] ?? 0),
                    height: rows.heights[dragLabel.line] ?? ROW_H,
                  }}
                />
                <div
                  className="pointer-events-none absolute z-20 w-px bg-pink-300/40"
                  style={{
                    left: dragLabel.end * pxPerSec,
                    top: RULER_H + (rows.tops[dragLabel.line] ?? 0),
                    height: rows.heights[dragLabel.line] ?? ROW_H,
                  }}
                />
                <div
                  className="pointer-events-none absolute z-30 -translate-x-1/2 whitespace-nowrap rounded bg-pink-600 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-white shadow"
                  style={{ left: dragLabel.start * pxPerSec, top: RULER_H + (rows.tops[dragLabel.line] ?? 0) - 17 }}
                >
                  {fmt(dragLabel.start)} → {fmt(dragLabel.end)}
                </div>
              </>
            )}

            <div
              className="pointer-events-none absolute top-0 bottom-0 w-px bg-pink-400"
              style={{ left: currentTime * pxPerSec }}
            />
          </div>
        )}
      </div>

      <HintButton
        label={t('lyrics.editor.pageBack', 'Page back')}
        info={t('lyrics.editor.pageBackInfo', 'Scrolls left so the last word on screen becomes the first one, about one window at a time. Drag the timeline for smaller moves.')}
        className={PAN_BTN}
        rootClassName="absolute left-1 top-1/2 z-30 -translate-y-1/2"
        onClick={() => onPage(-1)}
        disabled={!canPageLeft}
        ariaLabel={t('lyrics.editor.pageBack', 'Page back')}
      >
        <ChevronLeft size={18} />
      </HintButton>
      <HintButton
        label={t('lyrics.editor.pageForward', 'Page forward')}
        info={t('lyrics.editor.pageForwardInfo', 'Scrolls right so the first word past the right edge becomes the first one on screen. Jumps to the next window of words instead of a fixed number of seconds.')}
        className={PAN_BTN}
        rootClassName="absolute right-1 top-1/2 z-30 -translate-y-1/2"
        onClick={() => onPage(1)}
        disabled={!canPageRight}
        ariaLabel={t('lyrics.editor.pageForward', 'Page forward')}
      >
        <ChevronRight size={18} />
      </HintButton>
    </div>
  );
};

export default LyricEditorTimeline;
