// LyricEditorModal.tsx — the fine tool for placing words by ear.
//
// This file is layout and gestures. The rules live beside it, under
// `lyric-editor/`: `timingModel` (pure clamps, transforms and paging),
// `useEditHistory` (undo), `useLyricSaver` (writing and its revision rules),
// `useTimelineView` (zoom and Follow) and `useVisiblePeaks` (the waveform).
// Anything here that looks like a rule for what a legal lyric timeline is
// belongs there instead.
//
// Every button carries the house hover card through `HintButton` (which wraps
// `ParamLabel`'s children mode), including play/pause, so a control explains
// itself the same way the Auto save switch does. Word chips and their resize
// handles keep a plain title: they are hundreds of drag targets inside the
// scrolling timeline, and a portalled card per chip would cost more than it
// explains. Their name and times are on the chip and in its aria-label.
//
// Words cannot overlap. A move is clamped to the previous word's end and the
// next word's start; a resize either stops at the next word or, with Auto shift,
// pushes the words after it. The save endpoint repairs any overlap the same way,
// so nobody can produce two words fighting for one instant, including through
// the API.
//
// Keys: Space plays and pauses, left and right move the selected word, Shift
// with them resizes it, Ctrl with - and = zooms, Ctrl+Z undoes, Escape closes.
// The cards name those keys too.
//
// Edits save themselves about a second after the last change unless Auto save
// is switched off, in which case Save is shown and nothing is written until it
// is pressed. A failed save is not retried on a timer. If the file moved on
// underneath (a crop, a retranscription, another window), the save is refused
// and the footer offers Reload instead of writing stale timings back. The
// backdrop does not close the modal.

import React, { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import ReactDOM from 'react-dom';
import { useQuery } from '@tanstack/react-query';
import {
  ArrowLeftToLine, ChevronLeft, ChevronRight, ChevronsDown, ChevronsUp, Loader2, LocateFixed,
  Maximize2, Pause, Pencil, Play, Redo2, RefreshCw, Save, Scissors, Undo2, Volume2, X,
  ZoomIn, ZoomOut,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { toast } from 'react-toastify';
import {
  getSnapshot as getEngineSnapshot, seek, setRate, setVolume,
  subscribe as subscribeEngine, toggle,
} from '../../audio/engine';
import { lyricsJsonOptions } from '../../queries/shell-queue/lyrics';
import { HintButton } from '../shared/HintButton';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { WordHighlighter } from './WordHighlighter';
import {
  AUTOSAVE_KEY, BOTTOM_PAD, EDGE_SCROLL_PX, EDGE_SCROLL_STEP, MAX_PX, MIN_PX, MIN_WORD_SEC,
  PAN_THRESHOLD_PX, PEAKS_H, ROW_H, RULER_H,
  clampStart, cloneLines, fmt, lineAtOrBefore, mergeLineAt, moveWordTo, pageTargetTime, readFlag,
  resizeWordEdge, shiftEveryWord, shiftFrom, splitLineAt, type Selection,
} from './lyric-editor/timingModel';
import { useEditHistory } from './lyric-editor/useEditHistory';
import { useLyricSaver } from './lyric-editor/useLyricSaver';
import { useTimelineView } from './lyric-editor/useTimelineView';
import { useVisiblePeaks } from './lyric-editor/useVisiblePeaks';

const BTN = 'flex items-center gap-1 rounded-md border border-zinc-300 px-1.5 py-0.5 text-[11px] '
  + 'text-zinc-600 hover:bg-black/5 disabled:opacity-40 disabled:pointer-events-none '
  + 'dark:border-white/10 dark:text-zinc-300 dark:hover:bg-white/10';

const ICON_BTN = 'rounded-md border border-zinc-300 p-1 text-zinc-600 hover:bg-black/5 '
  + 'disabled:opacity-40 disabled:pointer-events-none dark:border-white/10 dark:text-zinc-300 dark:hover:bg-white/10';

const PAN_BTN = 'rounded-full border border-zinc-300 bg-white/85 p-1.5 text-zinc-600 shadow-md '
  + 'backdrop-blur transition-colors hover:bg-white disabled:opacity-30 disabled:pointer-events-none '
  + 'dark:border-white/15 dark:bg-zinc-900/85 dark:text-zinc-200 dark:hover:bg-zinc-900';

const SAVE_BTN = 'flex items-center gap-1 rounded-md border border-emerald-600 bg-emerald-600 px-2 py-0.5 '
  + 'text-[11px] font-semibold text-white hover:bg-emerald-500 disabled:pointer-events-none disabled:opacity-40';

export interface LyricEditorModalProps {
  audioUrl: string;
  title?: string;
  duration: number;
  currentTime: number;
  onClose: () => void;
}

export const LyricEditorModal: React.FC<LyricEditorModalProps> = ({
  audioUrl, title, duration, currentTime, onClose,
}) => {
  const { t } = useTranslation();
  const wordData = useQuery(lyricsJsonOptions(audioUrl)).data ?? null;
  const engine = useSyncExternalStore(subscribeEngine, getEngineSnapshot);

  const {
    lines, setLines, linesRef, record, undo, redo, reset, canUndo, canRedo,
  } = useEditHistory(cloneLines(wordData?.lines));

  const [selected, setSelected] = useState<Selection | null>(null);
  const [step, setStep] = useState(20);
  // The box holds what is typed and only commits on blur or Enter, so typing
  // 100 no longer passes through 1 and 10 on the way.
  const [stepDraft, setStepDraft] = useState('20');
  const [loopWord, setLoopWord] = useState(false);
  const [autoSave, setAutoSave] = useState(() => readFlag(AUTOSAVE_KEY, true));
  // Lengthening a word normally pushes what follows it along, which is what
  // keeps a resize from colliding with the next word.
  const [autoShift, setAutoShift] = useState(true);
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const [dragLabel, setDragLabel] = useState<{ line: number; start: number; end: number } | null>(null);
  const [viewportPx, setViewportPx] = useState(0);

  const scrollerRef = useRef<HTMLDivElement>(null);
  const peaksWrapRef = useRef<HTMLDivElement>(null);
  const peaksCanvasRef = useRef<HTMLCanvasElement>(null);
  // `scrolled` accumulates the pixels an edge auto-scroll moved under the
  // pointer, so a drag keeps travelling past the edge of the window.
  const dragRef = useRef<{ line: number; word: number; x: number; start: number; span: number; pushed: boolean; scrolled: number } | null>(null);
  const resizeRef = useRef<{ line: number; word: number; edge: 'left' | 'right'; x: number; start: number; end: number; pushed: boolean; scrolled: number } | null>(null);
  const panRef = useRef<{ x: number; left: number; moved: boolean; followReleased: boolean } | null>(null);
  const edgeRef = useRef<{ dx: number } | null>(null);
  const edgeRaf = useRef<number | null>(null);


  const total = duration || engine.duration || lines[lines.length - 1]?.end || 0;
  const selectedWord = selected ? lines[selected.line]?.words?.[selected.word] ?? null : null;

  const saver = useLyricSaver({
    audioUrl,
    lines,
    getLines: () => linesRef.current,
    autoSave,
    enabled: !!wordData,
    revision: Number(wordData?.revision) || undefined,
  });
  const { dirty, saving, saveError, stale, save } = saver;

  const {
    pxPerSec, setPxPerSec, zoomBy, zoomToSpan, scrollLeft, setScrollLeft,
    follow, setFollow, goToPlayhead,
  } = useTimelineView({
    scrollerRef,
    currentTime,
    total,
    anchorTime: selectedWord?.start ?? currentTime,
  });

  useVisiblePeaks({
    wrapRef: peaksWrapRef, canvasRef: peaksCanvasRef, peaks: engine.peaks,
    total, pxPerSec, scrollLeft, height: PEAKS_H,
  });

  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    const update = () => setViewportPx(el.clientWidth);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // The sidecar can land after the modal opens (slow disk, first read).
  const seeded = useRef(!!wordData);
  useEffect(() => {
    if (seeded.current || !wordData) return;
    seeded.current = true;
    const next = cloneLines(wordData.lines);
    reset(next);
    saver.adopt(next, Number(wordData.revision) || 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wordData]);

  /** The line the line-actions apply to: the selected one, else the playhead's. */
  const targetLine = selected ? selected.line : lineAtOrBefore(lines, currentTime);

  // Karaoke preview, from the editor's own state so it shows edits immediately.
  const previewIdx = lineAtOrBefore(lines, currentTime);
  const previewLine = previewIdx >= 0 ? lines[previewIdx] : null;
  const previewWords = useMemo(() => (previewLine?.words ?? []).map(w => ({
    word: w.word,
    start: w.start,
    end: w.end,
    confidence: w.confidence ?? 0,
    source: (w.source as 'matched' | 'whisper' | 'ad-lib') ?? 'whisper',
  })), [previewLine]);
  const previewWordIdx = previewWords.filter(w => w.start <= currentTime).length - 1;

  // ── Word and line actions ───────────────────────────────────────────────

  const wordStep = (dir: 1 | -1) => {
    if (!selected) return;
    const w = lines[selected.line]?.words?.[selected.word];
    if (!w) return;
    record();
    setLines(moveWordTo(lines, selected.line, selected.word, w.start + (dir * step) / 1000));
  };

  const setWordAtPlayhead = () => {
    if (!selected) return;
    record();
    setLines(moveWordTo(lines, selected.line, selected.word, currentTime));
  };

  /** Start the selected word exactly where the word before it ended. */
  const snapToPrevious = () => {
    if (!selected) return;
    const prevLine = selected.line > 0 ? lines[selected.line - 1] : null;
    const prevEnd = selected.word > 0
      ? lines[selected.line].words[selected.word - 1].end
      : prevLine && prevLine.words.length > 0
        ? prevLine.words[prevLine.words.length - 1].end
        : null;
    if (prevEnd == null) {
      toast.info(t('lyrics.editor.noPrevious', 'This is the first word of the track'));
      return;
    }
    record();
    setLines(moveWordTo(lines, selected.line, selected.word, prevEnd));
  };

  /** Lengthen or shorten the selected word by the step, from its end. */
  const resizeWordBy = (dir: 1 | -1) => {
    if (!selected) return;
    const w = lines[selected.line]?.words?.[selected.word];
    if (!w) return;
    record();
    const res = resizeWordEdge(lines, selected.line, selected.word, 'right', w.end + (dir * step) / 1000, { autoShift, total });
    if (!res) return;
    setLines(res.lines);
    if (res.noRoom) toast.warn(t('lyrics.editor.noRoom', 'No room before the end of the track'));
  };

  /** Every word and line by the same amount, for a uniform offset. */
  const shiftAll = (ms: number) => {
    record();
    setLines(shiftEveryWord(lines, ms / 1000));
  };

  /** Ripple: the selection and everything after it, for drift that builds up. */
  const shiftFromSelection = (ms: number) => {
    if (!selected) return;
    record();
    setLines(shiftFrom(lines, selected.line, selected.word, ms / 1000));
  };

  /**
   * A selected word IS the cut point, so splitting does not depend on where the
   * playhead happens to be. With nothing selected, the playhead's word is used.
   */
  const splitLine = () => {
    const idx = selected ? selected.line : lineAtOrBefore(lines, currentTime);
    if (idx < 0) {
      toast.info(t('lyrics.editor.noLine', 'Select a word to split from, or move the playhead into a line'));
      return;
    }
    const line = lines[idx];
    const cut = selected ? selected.word : line.words.findIndex(w => w.start >= currentTime);
    const next = splitLineAt(lines, idx, cut);
    if (!next) {
      toast.info(cut <= 0
        ? t('lyrics.editor.noCutFirst', 'That word already starts its line. Pick a later word to split there')
        : t('lyrics.editor.noCut', 'Put the playhead on the word the new line should start with'));
      return;
    }
    record();
    setLines(next);
    setSelected({ line: idx + 1, word: 0 });
  };

  /** Fold a line into the one above or below it, words re-sorted by time. */
  const mergeLine = (dir: -1 | 1) => {
    const merged = mergeLineAt(lines, targetLine, dir);
    if (!merged) {
      toast.info(t('lyrics.editor.noNeighbour', 'Nothing to merge with on that side'));
      return;
    }
    record();
    setLines(merged.lines);
    setSelected({ line: merged.at, word: 0 });
  };

  const revert = () => {
    record();
    setLines(cloneLines(saver.savedLines));
    setSelected(null);
  };

  const reloadFromDisk = async () => {
    const fresh = await saver.reload();
    reset(fresh.lines);
    setSelected(null);
  };

  const requestClose = () => {
    if (dirty && !stale) setConfirmDiscard(true);
    else onClose();
  };

  /** Arming the loop seeks just before the word, so the check starts at once. */
  const armLoop = () => {
    const next = !loopWord;
    setLoopWord(next);
    if (next && selectedWord) seek(Math.max(0, selectedWord.start - 0.2));
  };

  const commitStep = () => {
    const value = Math.max(1, Math.min(1000, Math.round(Number(stepDraft) || 20)));
    setStep(value);
    setStepDraft(String(value));
  };

  useEffect(() => {
    if (!loopWord || !selectedWord) return;
    if (currentTime > selectedWord.end + 0.05) {
      seek(Math.max(0, selectedWord.start - 0.2));
      if (!engine.playing) toggle();
    }
  }, [currentTime, loopWord, selectedWord, engine.playing]);

  /**
   * Follow owns the view, so the moment the user pans by hand it is switched
   * off and a toast says so. Leaving it on would snap the view back on the next
   * playhead tick and make the drag feel broken.
   */
  const releaseFollow = () => {
    setFollow(false);
    toast.info(t('lyrics.editor.followDisabled', 'Follow has been disabled.'));
  };

  /**
   * Page the timeline by words, not by pixels: back puts the last visible word
   * at the left edge, forward puts the first word past the right edge there.
   */
  const panPage = (dir: -1 | 1) => {
    const el = scrollerRef.current;
    if (!el) return;
    const target = pageTargetTime(lines, dir, {
      scrollLeft: el.scrollLeft, viewportPx: el.clientWidth || viewportPx, pxPerSec,
    });
    if (target == null) return;
    if (follow) releaseFollow();
    el.scrollLeft = Math.max(0, target * pxPerSec - 4);
  };

  const canPageLeft = lines.length > 0 && scrollLeft > 1;
  const canPageRight = pageTargetTime(lines, 1, { scrollLeft, viewportPx, pxPerSec }) != null;

  // ── Keyboard ────────────────────────────────────────────────────────────

  const actionsRef = useRef({
    undo, redo, requestClose, wordStep, resizeWordBy, zoomBy,
    hasSelection: false, playPause: () => {},
  });
  actionsRef.current = {
    undo, redo, requestClose, wordStep, resizeWordBy, zoomBy,
    hasSelection: !!selected,
    playPause: () => toggle(),
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        actionsRef.current.requestClose();
        return;
      }
      const target = e.target as HTMLElement | null;
      const typing = target instanceof HTMLInputElement
        || target instanceof HTMLTextAreaElement
        || target instanceof HTMLSelectElement
        || !!target?.isContentEditable;
      if (typing) return;

      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        e.stopPropagation();
        if (e.shiftKey) actionsRef.current.redo(); else actionsRef.current.undo();
        return;
      }
      if (mod && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        e.stopPropagation();
        actionsRef.current.redo();
        return;
      }
      if (mod && (e.key === '+' || e.key === '=')) {
        e.preventDefault();
        e.stopPropagation();
        actionsRef.current.zoomBy(1.25);
        return;
      }
      if (mod && (e.key === '-' || e.key === '_')) {
        e.preventDefault();
        e.stopPropagation();
        actionsRef.current.zoomBy(1 / 1.25);
        return;
      }
      // Space is the transport everywhere else in the app, and a focused word
      // chip must not swallow it into a click or scroll the pane.
      if (e.code === 'Space') {
        e.preventDefault();
        e.stopPropagation();
        actionsRef.current.playPause();
        return;
      }
      if (!mod && !e.altKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
        if (!actionsRef.current.hasSelection) return;
        e.preventDefault();
        e.stopPropagation();
        const dir = e.key === 'ArrowRight' ? 1 : -1;
        if (e.shiftKey) actionsRef.current.resizeWordBy(dir as 1 | -1);
        else actionsRef.current.wordStep(dir as 1 | -1);
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, []);

  // ── Pointer gestures ────────────────────────────────────────────────────

  const startDrag = (e: React.PointerEvent, li: number, wi: number) => {
    const w = lines[li]?.words?.[wi];
    if (!w) return;
    e.preventDefault();
    e.stopPropagation();
    setSelected({ line: li, word: wi });
    dragRef.current = {
      line: li, word: wi, x: e.clientX, start: w.start,
      span: Math.max(MIN_WORD_SEC, w.end - w.start), pushed: false, scrolled: 0,
    };
    setDragLabel({ line: li, start: w.start, end: w.end });
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
  };

  const dragMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    const line = d ? lines[d.line] : null;
    if (!d || !line) return;
    trackEdge(e.clientX);
    const nextLineStart = d.line + 1 < lines.length ? lines[d.line + 1]?.words[0]?.start ?? Infinity : Infinity;
    const wanted = d.start + (e.clientX - d.x + d.scrolled) / pxPerSec;
    const start = clampStart(line, d.word, wanted, d.span, nextLineStart);
    // One history entry per drag, however many frames it takes.
    if (!d.pushed) { record(); d.pushed = true; }
    setLines(prev => moveWordTo(prev, d.line, d.word, wanted));
    setDragLabel({ line: d.line, start, end: start + d.span });
  };

  const endDrag = () => {
    dragRef.current = null;
    setDragLabel(null);
    stopEdgeScroll();
  };

  const startResize = (e: React.PointerEvent, li: number, wi: number, edge: 'left' | 'right') => {
    const w = lines[li]?.words?.[wi];
    if (!w) return;
    e.preventDefault();
    e.stopPropagation();
    setSelected({ line: li, word: wi });
    resizeRef.current = { line: li, word: wi, edge, x: e.clientX, start: w.start, end: w.end, pushed: false, scrolled: 0 };
    setDragLabel({ line: li, start: w.start, end: w.end });
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
  };

  const resizeMove = (e: React.PointerEvent) => {
    const d = resizeRef.current;
    if (!d) return;
    trackEdge(e.clientX);
    const dx = (e.clientX - d.x + d.scrolled) / pxPerSec;
    const res = resizeWordEdge(lines, d.line, d.word, d.edge, d.edge === 'left' ? d.start + dx : d.end + dx, { autoShift, total });
    if (!res) return;
    if (!d.pushed) { record(); d.pushed = true; }
    setLines(res.lines);
    setDragLabel({ line: d.line, start: res.start, end: res.end });
  };

  const endResize = () => {
    resizeRef.current = null;
    setDragLabel(null);
    stopEdgeScroll();
  };

  /** Background drag pans the timeline; a click that never moved seeks. */
  const startPan = (e: React.PointerEvent) => {
    const el = scrollerRef.current;
    if (!el) return;
    panRef.current = { x: e.clientX, left: el.scrollLeft, moved: false, followReleased: false };
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
  };

  const movePan = (e: React.PointerEvent) => {
    const el = scrollerRef.current;
    const p = panRef.current;
    if (!el || !p) return;
    const dx = e.clientX - p.x;
    if (Math.abs(dx) <= PAN_THRESHOLD_PX) return;
    p.moved = true;
    // Only when Follow is on, and only once per gesture, so one drag never
    // stacks a toast per pointer move.
    if (follow && !p.followReleased) {
      p.followReleased = true;
      releaseFollow();
    }
    el.scrollLeft = p.left - dx;
  };

  const endPan = (e: React.PointerEvent) => {
    const p = panRef.current;
    panRef.current = null;
    const el = scrollerRef.current;
    if (!p || p.moved || !el) return;
    const rect = el.getBoundingClientRect();
    seek(Math.max(0, (e.clientX - rect.left + el.scrollLeft) / pxPerSec));
  };

  /** Keep the view scrolling while a drag sits near an edge. */
  function startEdgeScroll() {
    if (edgeRaf.current !== null) return;
    const tick = () => {
      const el = scrollerRef.current;
      const d = edgeRef.current;
      if (!el || !d || d.dx === 0) { edgeRaf.current = null; return; }
      el.scrollLeft = Math.max(0, el.scrollLeft + d.dx);
      if (dragRef.current) dragRef.current.scrolled += d.dx;
      if (resizeRef.current) resizeRef.current.scrolled += d.dx;
      edgeRaf.current = requestAnimationFrame(tick);
    };
    edgeRaf.current = requestAnimationFrame(tick);
  }

  function stopEdgeScroll() {
    if (edgeRaf.current !== null) cancelAnimationFrame(edgeRaf.current);
    edgeRaf.current = null;
    edgeRef.current = null;
  }

  function trackEdge(clientX: number) {
    const el = scrollerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const dx = clientX < rect.left + EDGE_SCROLL_PX ? -EDGE_SCROLL_STEP
      : clientX > rect.right - EDGE_SCROLL_PX ? EDGE_SCROLL_STEP
        : 0;
    edgeRef.current = { dx };
    if (dx !== 0) startEdgeScroll();
    else stopEdgeScroll();
  }

  const tickStep = pxPerSec >= 240 ? 1 : pxPerSec >= 90 ? 5 : 15;
  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let time = 0; time <= total; time += tickStep) out.push(time);
    return out;
  }, [total, tickStep]);

  const status = stale
    ? { label: t('lyrics.editor.stale', 'Changed on disk'), cls: 'bg-red-500/15 text-red-400' }
    : saving
      ? { label: t('lyrics.editor.saving', 'Saving…'), cls: 'bg-amber-500/15 text-amber-500' }
      : saveError
        ? { label: t('lyrics.editor.notSaved', 'Not saved'), cls: 'bg-red-500/15 text-red-400' }
        : dirty
          ? { label: t('lyrics.editor.unsaved', 'Unsaved'), cls: 'bg-amber-500/15 text-amber-500' }
          : { label: t('lyrics.editor.savedBadge', 'Saved'), cls: 'bg-emerald-500/15 text-emerald-400' };

  return ReactDOM.createPortal(
    <div
      data-portal-layer
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-[160] flex items-center justify-center bg-black/60 p-3 backdrop-blur-sm"
    >
      <div className="flex max-h-[94vh] w-[96vw] max-w-[1800px] flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-2xl dark:border-white/10 dark:bg-zinc-950">
        {/* Header */}
        <div className="flex items-center gap-2 border-b border-zinc-200 px-3 py-2 dark:border-white/10">
          <Pencil size={13} className="text-pink-500" />
          <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-600 dark:text-zinc-300">
            {t('lyrics.editor.title', 'Lyric editor')}
          </span>
          <div className="flex items-center gap-1">
            <HintButton
              label={t('lyrics.editor.undo', 'Undo')}
              info={t('lyrics.editor.undoInfo', 'Steps back one change, with a whole drag counting as one. Nothing is saved by undoing: the file gets the undone state at the next save. Ctrl+Z here.')}
              className={ICON_BTN}
              onClick={undo}
              disabled={!canUndo}
              ariaLabel={t('lyrics.editor.undo', 'Undo')}
            >
              <Undo2 size={13} />
            </HintButton>
            <HintButton
              label={t('lyrics.editor.redo', 'Redo')}
              info={t('lyrics.editor.redoInfo', 'Steps forward again after an undo. The stack is cleared by the next new change. Ctrl+Shift+Z here.')}
              className={ICON_BTN}
              onClick={redo}
              disabled={!canRedo}
              ariaLabel={t('lyrics.editor.redo', 'Redo')}
            >
              <Redo2 size={13} />
            </HintButton>
          </div>
          {title && <span className="truncate text-xs text-zinc-500">{title}</span>}
          <span className={`rounded px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider ${status.cls}`}>
            {status.label}
          </span>
          <HintButton
            label={t('common.close', 'Close')}
            info={t('lyrics.editor.closeInfo', 'Closes the editor. With unsaved edits it asks first, so closing never silently loses a placement. Escape does the same.')}
            className="rounded p-1 text-zinc-500 hover:bg-black/5 dark:hover:bg-white/10"
            onClick={requestClose}
            rootClassName="ml-auto"
            ariaLabel={t('common.close', 'Close')}
          >
            <X size={14} />
          </HintButton>
        </div>

        {/* Peaks for the visible window */}
        <div
          ref={peaksWrapRef}
          className="shrink-0 border-b border-zinc-200 bg-zinc-100 dark:border-white/10 dark:bg-black/40"
          style={{ height: PEAKS_H }}
        >
          <canvas ref={peaksCanvasRef} className="block" />
        </div>

        {/* Timeline, with paging arrows on both edges */}
        <div className="relative flex min-h-[240px] flex-1">
          <div
            ref={scrollerRef}
            onScroll={e => setScrollLeft(e.currentTarget.scrollLeft)}
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
                style={{ width: Math.max(1, total * pxPerSec), height: RULER_H + lines.length * ROW_H + BOTTOM_PAD }}
                onPointerDown={startPan}
                onPointerMove={movePan}
                onPointerUp={endPan}
                onPointerLeave={() => { panRef.current = null; }}
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
                    className="pointer-events-none absolute left-0 right-0 border-b border-zinc-100 dark:border-white/5"
                    style={{ top: RULER_H + li * ROW_H, height: ROW_H }}
                  >
                    {line.words.length === 0 && (
                      <span
                        className="absolute top-[7px] whitespace-nowrap text-[10px] italic text-zinc-500"
                        style={{ left: line.start * pxPerSec }}
                        title={t('lyrics.editor.untimedLine', 'This line has no word timings, so it can only be kept as it is')}
                      >
                        {line.text}
                      </span>
                    )}
                    {line.words.map((w, wi) => {
                      const isSelected = selected?.line === li && selected?.word === wi;
                      return (
                        <button
                          key={wi}
                          type="button"
                          onPointerDown={e => startDrag(e, li, wi)}
                          onPointerMove={dragMove}
                          onPointerUp={endDrag}
                          onPointerCancel={endDrag}
                          aria-pressed={isSelected}
                          aria-label={`${w.word}, ${fmt(w.start)} to ${fmt(w.end)}`}
                          className={`pointer-events-auto absolute top-[3px] h-[22px] cursor-grab overflow-hidden rounded px-1 text-left text-[10px] leading-[22px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-300 ${
                            isSelected
                              ? 'z-20 bg-pink-500 text-white ring-2 ring-pink-300'
                              : 'bg-pink-500/15 text-pink-300 hover:bg-pink-500/30'
                          }`}
                          style={{ left: w.start * pxPerSec, width: Math.max(10, (w.end - w.start) * pxPerSec) }}
                          title={`${w.word} · ${fmt(w.start)} → ${fmt(w.end)}`}
                        >
                          <span className="pointer-events-none block truncate">{w.word}</span>
                          {isSelected && (
                            <>
                              <span
                                onPointerDown={e => startResize(e, li, wi, 'left')}
                                onPointerMove={resizeMove}
                                onPointerUp={endResize}
                                onPointerCancel={endResize}
                                className="pointer-events-auto absolute left-0 top-0 z-30 h-full w-[6px] cursor-ew-resize rounded-l bg-white/80"
                                title={t('lyrics.editor.resizeLeft', 'Drag to move the start of this word')}
                              />
                              <span
                                onPointerDown={e => startResize(e, li, wi, 'right')}
                                onPointerMove={resizeMove}
                                onPointerUp={endResize}
                                onPointerCancel={endResize}
                                className="pointer-events-auto absolute right-0 top-0 z-30 h-full w-[6px] cursor-ew-resize rounded-r bg-white/80"
                                title={t('lyrics.editor.resizeRight', 'Drag to change how long this word lasts')}
                              />
                            </>
                          )}
                        </button>
                      );
                    })}
                  </div>
                ))}

                {dragLabel && (
                  <>
                    <div
                      className="pointer-events-none absolute z-20 w-px bg-pink-300/80"
                      style={{ left: dragLabel.start * pxPerSec, top: RULER_H + dragLabel.line * ROW_H, height: ROW_H }}
                    />
                    <div
                      className="pointer-events-none absolute z-20 w-px bg-pink-300/40"
                      style={{ left: dragLabel.end * pxPerSec, top: RULER_H + dragLabel.line * ROW_H, height: ROW_H }}
                    />
                    <div
                      className="pointer-events-none absolute z-30 -translate-x-1/2 whitespace-nowrap rounded bg-pink-600 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-white shadow"
                      style={{ left: dragLabel.start * pxPerSec, top: RULER_H + dragLabel.line * ROW_H - 17 }}
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
            onClick={() => panPage(-1)}
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
            onClick={() => panPage(1)}
            disabled={!canPageRight}
            ariaLabel={t('lyrics.editor.pageForward', 'Page forward')}
          >
            <ChevronRight size={18} />
          </HintButton>
        </div>

        {/* Selected word + line actions */}
        <div className="flex flex-wrap items-center gap-1.5 border-t border-zinc-200 px-3 py-2 text-[11px] text-zinc-600 dark:border-white/10 dark:text-zinc-300">
          {selectedWord ? (
            <>
              <span className="font-semibold text-pink-400">“{selectedWord.word}”</span>
              <span className="font-mono tabular-nums">{fmt(selectedWord.start)} → {fmt(selectedWord.end)}</span>
              <span className="tabular-nums opacity-50">({(selectedWord.end - selectedWord.start).toFixed(2)}s)</span>
              <ParamLabel
                label={t('lyrics.editor.step', 'Step')}
                info={t('lyrics.editor.stepInfo', 'How far the nudge, length and ripple buttons move a word, in milliseconds. Applied when you leave the box or press Enter, so typing a three-digit value does not step through one and ten.')}
              >
                <input
                  type="number" min={1} max={1000} value={stepDraft}
                  onChange={e => setStepDraft(e.target.value)}
                  onBlur={commitStep}
                  onKeyDown={e => { if (e.key === 'Enter') { e.currentTarget.blur(); } }}
                  className="w-14 rounded border border-zinc-300 bg-transparent px-1 py-0.5 text-center dark:border-white/10"
                  aria-label={t('lyrics.editor.step', 'Nudge step in milliseconds')}
                />
              </ParamLabel>
              <span className="opacity-60">ms</span>
              <HintButton
                label={t('lyrics.editor.nudgeEarlier', 'Move earlier')}
                info={t('lyrics.editor.shiftEarlierHint', 'Moves this word earlier by the step. It stops at the end of the word before it, so two words never share an instant. Left arrow does the same.')}
                className={BTN}
                onClick={() => wordStep(-1)}
              >
                -{step}
              </HintButton>
              <HintButton
                label={t('lyrics.editor.nudgeLater', 'Move later')}
                info={t('lyrics.editor.shiftLaterHint', 'Moves this word later by the step, stopping before the next word. Right arrow does the same.')}
                className={BTN}
                onClick={() => wordStep(1)}
              >
                +{step}
              </HintButton>
              <HintButton
                label={t('lyrics.editor.shorter', 'Shorten')}
                info={t('lyrics.editor.shorterHint', 'Makes this word end earlier by the step. It cannot end before it starts, and the next word does not move. Shift+Left arrow does the same.')}
                className={BTN}
                onClick={() => resizeWordBy(-1)}
              >
                {t('lyrics.editor.len', 'Len')} -{step}
              </HintButton>
              <HintButton
                label={t('lyrics.editor.longer', 'Lengthen')}
                info={t('lyrics.editor.longerHint', 'Makes this word last longer by the step. With Auto shift on, everything after it moves later by the overhang; off, it stops at the next word. Shift+Right arrow does the same.')}
                className={BTN}
                onClick={() => resizeWordBy(1)}
              >
                {t('lyrics.editor.len', 'Len')} +{step}
              </HintButton>
              <HintButton
                label={t('lyrics.editor.snap', 'Snap to prev')}
                info={t('lyrics.editor.snapHint', 'Starts this word exactly where the previous word ends, across a line break too. The fix for a word that landed with a gap before it.')}
                className={BTN}
                onClick={snapToPrevious}
              >
                <ArrowLeftToLine size={12} /> {t('lyrics.editor.snap', 'Snap to prev')}
              </HintButton>
              <HintButton
                label={t('lyrics.editor.setStart', 'Start at playhead')}
                info={t('lyrics.editor.setStartInfo', 'Moves this word\u2019s onset to the playhead, clamped like any move: after the previous word and early enough to finish before the next one.')}
                className={BTN}
                onClick={setWordAtPlayhead}
              >
                {t('lyrics.editor.setStart', 'Start at playhead')}
              </HintButton>
              <HintButton
                label={t('lyrics.editor.playFrom', 'Play from word')}
                info={t('lyrics.editor.playFromHint', 'Seeks a quarter second before this word and plays, so a check starts on the syllable instead of a minute earlier.')}
                className={BTN}
                onClick={() => {
                  seek(Math.max(0, selectedWord.start - 0.25));
                  if (!engine.playing) toggle();
                }}
              >
                <Play size={12} /> {t('lyrics.editor.playFrom', 'Play from word')}
              </HintButton>
              <HintButton
                label={t('lyrics.editor.loop', 'Loop word')}
                info={t('lyrics.editor.loopHint', 'Replays this word until it is switched off. Arming it seeks just before the word, and the loop restarts whenever playback passes its end.')}
                className={`${BTN} ${loopWord ? 'border-pink-400 text-pink-400' : ''}`}
                onClick={armLoop}
              >
                {t('lyrics.editor.loop', 'Loop word')}
              </HintButton>
            </>
          ) : (
            <span className="opacity-60">
              {t('lyrics.editor.selectHint', 'Click a word on the timeline to select it, or drag it to move it.')}
            </span>
          )}

          <div className="ml-auto flex items-center gap-1.5">
            <HintButton
              label={t('lyrics.editor.mergePrev', 'Merge previous')}
              info={t('lyrics.editor.mergePrevHint', 'Folds this line into the one above it. The words are re-sorted by time and the earlier line keeps its section label.')}
              className={BTN}
              onClick={() => mergeLine(-1)}
              disabled={targetLine <= 0 || lines.length < 2}
            >
              <ChevronsUp size={12} /> {t('lyrics.editor.mergePrev', 'Merge previous')}
            </HintButton>
            <HintButton
              label={t('lyrics.editor.mergeNext', 'Merge next')}
              info={t('lyrics.editor.mergeNextHint', 'Folds the line below into this one. The words are re-sorted by time and this line keeps its section label.')}
              className={BTN}
              onClick={() => mergeLine(1)}
              disabled={targetLine < 0 || targetLine >= lines.length - 1}
            >
              <ChevronsDown size={12} /> {t('lyrics.editor.mergeNext', 'Merge next')}
            </HintButton>
            <HintButton
              label={t('lyrics.editor.split', 'Split line')}
              info={t('lyrics.editor.splitHint', 'Starts a new line at the selected word, or at the word under the playhead when nothing is selected. The selected word becomes the first word of the new line, so the break lands where that syllable is sung.')}
              className={BTN}
              onClick={splitLine}
            >
              <Scissors size={12} /> {t('lyrics.editor.split', 'Split line')}
            </HintButton>
          </div>
        </div>

        {/* Transport, along the bottom like a DAW's */}
        <div className="flex flex-wrap items-center gap-1.5 border-t border-zinc-200 px-3 py-2 text-[11px] text-zinc-600 dark:border-white/10 dark:text-zinc-300">
          <HintButton
            label={t('player.playPause', 'Play or pause')}
            info={t('lyrics.editor.playInfo', 'Plays the same audio as the player bar, from the playhead. Space does the same anywhere in the editor, including while a word is selected, so it never scrolls the pane instead.')}
            className={ICON_BTN}
            onClick={() => toggle()}
            ariaLabel={t('player.playPause', 'Play or pause')}
          >
            {engine.playing ? <Pause size={13} /> : <Play size={13} />}
          </HintButton>
          <HintButton
            label={t('lyrics.editor.back1', 'Back 1 second')}
            info={t('lyrics.editor.back1Info', 'Moves the playhead one second earlier. Each click is one second; hold nothing, the loop is what repeats.')}
            className={BTN}
            onClick={() => seek(Math.max(0, currentTime - 1))}
          >
            <ChevronLeft size={12} />1s
          </HintButton>
          <HintButton
            label={t('lyrics.editor.backTenth', 'Back 0.1 second')}
            info={t('lyrics.editor.backTenthInfo', 'Moves the playhead a tenth of a second earlier, the finest seek offered. Useful for landing on a transient before nudging a word onto it.')}
            className={BTN}
            onClick={() => seek(Math.max(0, currentTime - 0.1))}
          >
            -0.1s
          </HintButton>
          <HintButton
            label={t('lyrics.editor.forwardTenth', 'Forward 0.1 second')}
            info={t('lyrics.editor.forwardTenthInfo', 'Moves the playhead a tenth of a second later.')}
            className={BTN}
            onClick={() => seek(currentTime + 0.1)}
          >
            +0.1s
          </HintButton>
          <HintButton
            label={t('lyrics.editor.forward1', 'Forward 1 second')}
            info={t('lyrics.editor.forward1Info', 'Moves the playhead one second later.')}
            className={BTN}
            onClick={() => seek(currentTime + 1)}
          >
            1s<ChevronRight size={12} />
          </HintButton>
          <span className="ml-1 font-mono tabular-nums">{fmt(currentTime)} / {fmt(total)}</span>
          <div className="ml-2 flex items-center gap-1">
            <ParamLabel
              label={t('lyrics.editor.speed', 'Speed')}
              info={t('lyrics.editor.speedInfo', 'Playback speed for both the editor and the player bar, because they share one audio element. Half speed widens a syllable enough to hear where it starts.')}
            >
              <span className="flex items-center gap-1">
                <span className="opacity-60">{t('lyrics.editor.speed', 'Speed')}</span>
                <StyledSelect
                  accent="pink"
                  size="sm"
                  value={engine.rate}
                  onChange={r => setRate(Number(r))}
                  options={[0.5, 0.75, 1].map(r => ({ value: r, label: `${r}×` }))}
                />
              </span>
            </ParamLabel>
          </div>
          <ParamLabel
            label={t('lyrics.editor.volume', 'Volume')}
            info={t('lyrics.editor.volumeInfo', 'The shared player volume. Lower it against the track rather than muting: a quiet syllable is easier to place when you can still hear its neighbour.')}
          >
            <label className="flex items-center gap-1">
              <Volume2 size={12} />
              <input
                type="range" min={0} max={1} step={0.01} value={engine.volume}
                onChange={e => setVolume(Number(e.target.value))}
                className="w-20 accent-pink-500"
                aria-label={t('lyrics.editor.volume', 'Volume')}
              />
            </label>
          </ParamLabel>
          <Toggle
            size="sm"
            accent="pink"
            checked={follow}
            onChange={setFollow}
            label={t('lyrics.editor.follow', 'Follow')}
            info={t(
              'lyrics.editor.followHint',
              'Keeps the timeline on the playhead while the track plays. Off: the view stays where you put it. The choice is remembered for the next time the editor opens. Dragging the timeline switches it off for you, with a toast saying Follow has been disabled, so a pan is never snapped back. The paging arrows do the same.',
            )}
          />
          <Toggle
            size="sm"
            accent="pink"
            checked={autoShift}
            onChange={setAutoShift}
            label={t('lyrics.editor.autoShift', 'Auto shift')}
            info={t(
              'lyrics.editor.autoShiftHint',
              'What happens when a word is made longer than the gap before the next one. On: the words after it move later by the overhang, so the spacing you set survives and nothing overlaps. Off: the word stops at the next word instead. Either way two words can never share an instant, and a save repairs any overlap the same way.',
            )}
          />
        </div>

        {/* Footer: view controls, shifts, save */}
        <div className="flex flex-wrap items-center gap-1.5 border-t border-zinc-200 px-3 py-2 text-[11px] text-zinc-600 dark:border-white/10 dark:text-zinc-300">
          <div className="flex items-center gap-1">
            <HintButton
              label={t('lyrics.editor.zoomOut', 'Zoom out')}
              info={t('lyrics.editor.zoomOutInfo', 'Shows more of the track per pixel, so a whole line fits at once. Ctrl+- does the same from anywhere.')}
              className={ICON_BTN}
              onClick={() => zoomBy(1 / 1.25)}
              ariaLabel={t('lyrics.editor.zoomOut', 'Zoom out')}
            >
              <ZoomOut size={12} />
            </HintButton>
            <ParamLabel
              label={t('lyrics.editor.zoom', 'Timeline zoom')}
              info={t('lyrics.editor.zoomHint', 'Pixels per second. At 30 a phrase is a few pixels; at 300 a syllable is wide enough to read and drag. Ctrl+- and Ctrl+= change it, and the chosen zoom is remembered.')}
            >
              <input
                type="range" min={MIN_PX} max={MAX_PX} step={10} value={pxPerSec}
                onChange={e => setPxPerSec(Number(e.target.value))}
                className="w-24 accent-pink-500"
                aria-label={t('lyrics.editor.zoom', 'Timeline zoom')}
              />
            </ParamLabel>
            <HintButton
              label={t('lyrics.editor.zoomIn', 'Zoom in')}
              info={t('lyrics.editor.zoomInInfo', 'Shows less of the track per pixel, which is how a syllable gets wide enough to place. Ctrl+= does the same.')}
              className={ICON_BTN}
              onClick={() => zoomBy(1.25)}
              ariaLabel={t('lyrics.editor.zoomIn', 'Zoom in')}
            >
              <ZoomIn size={12} />
            </HintButton>
            <span className="w-10 tabular-nums text-zinc-500">{pxPerSec}</span>
            <span className="text-[10px] text-zinc-500">px/s</span>
            <HintButton
              label={t('lyrics.editor.goToPlayhead', 'Scroll to the playhead')}
              info={t('lyrics.editor.goToPlayheadInfo', 'Brings the view back to the current time after a pan, without turning Follow on. The playhead keeps moving; this only moves the window.')}
              className={ICON_BTN}
              onClick={goToPlayhead}
              ariaLabel={t('lyrics.editor.goToPlayhead', 'Scroll to the playhead')}
            >
              <LocateFixed size={12} />
            </HintButton>
            <HintButton
              label={t('lyrics.editor.zoomSelected', 'Frame the selected word')}
              info={t('lyrics.editor.zoomSelectedInfo', 'Zooms so the selected word fills most of the window, for work at syllable level. Needs a selected word.')}
              className={ICON_BTN}
              onClick={() => { if (selectedWord) zoomToSpan(Math.max(0.3, (selectedWord.end - selectedWord.start) * 8)); }}
              disabled={!selectedWord}
              ariaLabel={t('lyrics.editor.zoomSelected', 'Frame the selected word')}
            >
              <Maximize2 size={12} />
            </HintButton>
          </div>

          <span className="ml-2 opacity-60">{t('lyrics.editor.wholeTrack', 'Whole track')}</span>
          <HintButton
            label={t('lyrics.editor.wholeTrack', 'Whole track')}
            info={t('lyrics.editor.wholeTrackInfo', 'Moves every word and line by the step, for an offset that is wrong from the first line to the last. Stops at zero rather than going negative.')}
            className={BTN}
            onClick={() => shiftAll(-step)}
          >
            -{step} ms
          </HintButton>
          <HintButton
            label={t('lyrics.editor.wholeTrack', 'Whole track')}
            info={t('lyrics.editor.wholeTrackInfo', 'Moves every word and line by the step, for an offset that is wrong from the first line to the last.')}
            className={BTN}
            onClick={() => shiftAll(step)}
          >
            +{step} ms
          </HintButton>

          <span className="ml-2 opacity-60">{t('lyrics.editor.fromSelected', 'From selected')}</span>
          <HintButton
            label={t('lyrics.editor.fromSelected', 'From selected')}
            info={t('lyrics.editor.shiftLeftHint', 'Moves the selected word and everything after it earlier by the step, leaving the part already right where it is. The fix for drift that builds up through the track.')}
            className={BTN}
            onClick={() => shiftFromSelection(-step)}
            disabled={!selected}
          >
            ← {step} ms
          </HintButton>
          <HintButton
            label={t('lyrics.editor.fromSelected', 'From selected')}
            info={t('lyrics.editor.shiftRightHint', 'Moves the selected word and everything after it later by the step, leaving the part before it alone.')}
            className={BTN}
            onClick={() => shiftFromSelection(step)}
            disabled={!selected}
          >
            {step} ms →
          </HintButton>

          <div className="ml-auto flex items-center gap-2">
            <Toggle
              size="sm"
              accent="pink"
              checked={autoSave}
              onChange={setAutoSave}
              label={t('lyrics.editor.autoSave', 'Auto save')}
              info={t(
                'lyrics.editor.autoSaveHint',
                'On: a change is written about a second after you stop, and the Save button is hidden because there is nothing to press. Off: nothing is written until Save, which appears when this is switched off, and closing with unsaved edits asks first. The choice is remembered.',
              )}
            />
            {stale ? (
              <>
                <span className="text-red-400">{t('lyrics.editor.staleNote', 'This file changed on disk')}</span>
                <HintButton
                  label={t('lyrics.editor.reload', 'Reload')}
                  info={t('lyrics.editor.reloadHint', 'Reads the file as it is now and throws away the edits on screen, which is the only safe move when a crop, a retranscription or another window moved the timings on.')}
                  className={BTN}
                  onClick={() => { void reloadFromDisk(); }}
                >
                  <RefreshCw size={12} /> {t('lyrics.editor.reload', 'Reload')}
                </HintButton>
              </>
            ) : confirmDiscard ? (
              <>
                <span className="text-amber-500">{t('lyrics.editor.discard', 'Discard changes?')}</span>
                <HintButton
                  label={t('lyrics.editor.discardYes', 'Discard')}
                  info={t('lyrics.editor.discardInfo', 'Closes the editor and throws away the changes that have not been saved. There is no undo for this once the modal is gone.')}
                  className={BTN}
                  onClick={onClose}
                >
                  {t('lyrics.editor.discardYes', 'Discard')}
                </HintButton>
                <HintButton
                  label={t('lyrics.editor.keep', 'Keep editing')}
                  info={t('lyrics.editor.keepInfo', 'Closes the question and leaves the editor open with the edits still on screen.')}
                  className={BTN}
                  onClick={() => setConfirmDiscard(false)}
                >
                  {t('lyrics.editor.keep', 'Keep editing')}
                </HintButton>
              </>
            ) : (
              <>
                <HintButton
                  label={t('lyrics.editor.revert', 'Reload saved')}
                  info={t('lyrics.editor.revertHint', 'Throws away the edits on screen and goes back to the last saved file. Unlike Undo this is not one step, so it is disabled when there is nothing pending.')}
                  className={BTN}
                  onClick={revert}
                  disabled={!dirty}
                >
                  <RefreshCw size={12} /> {t('lyrics.editor.revert', 'Reload saved')}
                </HintButton>
                {!autoSave && (
                  <HintButton
                    label={t('lyrics.editor.save', 'Save')}
                    info={t('lyrics.editor.saveHint', 'Writes the changes now. Shown only while Auto save is off; with it on the write happens about a second after you stop.')}
                    className={SAVE_BTN}
                    onClick={() => { void save(); }}
                    disabled={!dirty || saving}
                  >
                    {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
                    {t('lyrics.editor.save', 'Save')}
                  </HintButton>
                )}
              </>
            )}
          </div>
        </div>

        {/* Karaoke preview, the last row: what is being sung, from the editor's own state */}
        {previewLine && (
          <div className="flex items-center justify-center gap-2 border-t border-zinc-200 bg-zinc-50/60 px-3 py-2 dark:border-white/10 dark:bg-white/5">
            {previewLine.section && (
              <span className="rounded-full border border-violet-500/20 bg-violet-500/15 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-violet-300">
                {previewLine.section}
              </span>
            )}
            <span className="text-center text-sm font-semibold text-zinc-800 md:text-base dark:text-zinc-100">
              {previewWords.length > 0
                ? <WordHighlighter words={previewWords} activeWordIndex={previewWordIdx} />
                : previewLine.text}
            </span>
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
};
