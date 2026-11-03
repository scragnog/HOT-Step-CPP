// LyricEditorModal.tsx — the editor's shell and the state it composes.
//
// This file is the connector: it reads the sidecar, owns the selection, the step
// and the switches, and wires the hooks to the rows. Everything visible lives in
// `lyric-editor/`, so a row can be read, or changed, without scrolling past the
// others:
//
//   `LyricEditorHeader`      title, save badge, undo/redo, close
//   `LyricPeaks`             the waveform strip for the visible window
//   `LyricEditorTimeline`    ruler, rows, word chips, playhead, paging arrows
//   `LyricEditorWordActions` the selected word's row plus split and merge
//   `LyricEditorTransport`   play, seek, speed, volume, Follow, Auto shift, export
//   `LyricEditorFooter`      zoom and view, the offsets, Reload/Save
//   `LyricEditorPreview`     the line being sung, from the editor's own state
//
// Each row owns what only it needs: the header decides the save badge from the
// four facts about the file, the timeline thins its own ruler, the word row
// holds the half-typed step, the transport reads the engine, and the preview
// finds the current line. What is left here is shared: the selection, the
// committed step, the switches, and the three commands that span several hooks
// (close, reload, export).
//
// The behaviour lives beside this file, and belongs there rather than here:
// `timingModel` (clamps, transforms, paging, the ruler, every distance),
// `useEditHistory` (undo), `useTimingEdits` (every word and line change),
// `useLyricSaver` (writing and its revision rules), `useTimelineView` (zoom,
// Follow, paging), `useWordGestures` (dragging, resizing, panning),
// `useWordPlayback` (playing and looping one word) and `useVisiblePeaks` (the
// waveform).
//
// Every button carries the house hover card through `HintButton`, including
// play/pause, so a control explains itself the same way the Auto save switch
// does. Word chips keep a plain title: they are hundreds of drag targets inside
// the scrolling timeline, and their name and times are on the chip and in its
// aria-label.
//
// Words cannot overlap. A move is clamped to the previous word's end and the
// next word's start; a resize either stops at the next word or, with Auto shift,
// pushes the words after it. The save endpoint repairs any overlap the same way,
// so nobody can produce two words fighting for one instant, including through
// the API.
//
// Keys are declared as a list of bindings for the shared `useKeyboardShortcuts`
// (`hooks/`), which owns modifier matching, the focused-field rule and consuming
// the event: Space plays and pauses, left and right move the selected word,
// Shift with them resizes it, Ctrl with - and = zooms, Ctrl+Z undoes, Escape
// closes. The cards name those keys too.
//
// Edits save themselves about a second after the last change unless Auto save
// is switched off, in which case Save is shown and nothing is written until it
// is pressed. A failed save is not retried on a timer. If the file moved on
// underneath (a crop, a retranscription, another window), the save is refused
// and the footer offers Reload instead of writing stale timings back. The
// backdrop does not close the modal.

import React, { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import ReactDOM from 'react-dom';
import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { toast } from 'react-toastify';
import {
  getSnapshot as getEngineSnapshot, subscribe as subscribeEngine, toggle,
} from '../../audio/engine';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';
import { lyricsJsonOptions } from '../../queries/shell-queue/lyrics';
import { downloadLyricsLrcForAudio } from '../../utils/downloadTrack';
import { LyricEditorFooter } from './lyric-editor/LyricEditorFooter';
import { LyricEditorHeader } from './lyric-editor/LyricEditorHeader';
import { LyricEditorPreview } from './lyric-editor/LyricEditorPreview';
import { LyricEditorTimeline } from './lyric-editor/LyricEditorTimeline';
import { LyricEditorTransport } from './lyric-editor/LyricEditorTransport';
import { LyricEditorWordActions } from './lyric-editor/LyricEditorWordActions';
import { LyricPeaks } from './lyric-editor/LyricPeaks';
import { useEditHistory } from './lyric-editor/useEditHistory';
import { useLyricSaver } from './lyric-editor/useLyricSaver';
import { useTimelineView } from './lyric-editor/useTimelineView';
import { useTimingEdits } from './lyric-editor/useTimingEdits';
import { useWordGestures } from './lyric-editor/useWordGestures';
import { useWordPlayback } from './lyric-editor/useWordPlayback';
import {
  AUTOSAVE_KEY, FRAME_MIN_SPAN, FRAME_SPAN_FACTOR, STEP_DEFAULT, ZOOM_FACTOR,
  cloneLines, readFlag, type Selection,
} from './lyric-editor/timingModel';

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

  const { data: historyData, ...history } = useEditHistory(cloneLines(wordData?.lines));
  const { lines, linesRef, canUndo, canRedo } = historyData;
  const { setLines, record, reset, undo, redo } = history;

  const [selected, setSelected] = useState<Selection | null>(null);
  const [step, setStep] = useState(STEP_DEFAULT);
  const [autoSave, setAutoSave] = useState(() => readFlag(AUTOSAVE_KEY, true));
  // Lengthening a word normally pushes what follows it along, which is what
  // keeps a resize from colliding with the next word.
  const [autoShift, setAutoShift] = useState(true);
  const [confirmDiscard, setConfirmDiscard] = useState(false);

  const scrollerRef = useRef<HTMLDivElement>(null);

  const total = duration || engine.duration || lines[lines.length - 1]?.end || 0;
  const selectedWord = selected ? lines[selected.line]?.words?.[selected.word] ?? null : null;

  const { data: saveState, ...saver } = useLyricSaver({
    audioUrl,
    lines,
    getLines: () => linesRef.current,
    autoSave,
    enabled: !!wordData,
    revision: Number(wordData?.revision) || undefined,
  });
  const { dirty, saving, saveError, stale, savedLines } = saveState;
  const { save, adopt, reload } = saver;

  const { data: view, ...timeline } = useTimelineView({
    scrollerRef,
    lines,
    currentTime,
    total,
    anchorTime: selectedWord?.start ?? currentTime,
  });
  const {
    pxPerSec, scrollLeft, follow, canPageLeft, canPageRight,
  } = view;
  const {
    setPxPerSec, zoomBy, zoomToSpan, setScrollLeft, setFollow, goToPlayhead,
    releaseFollow, pageBy,
  } = timeline;

  // The sidecar can land after the modal opens (slow disk, first read).
  const seeded = useRef(!!wordData);
  useEffect(() => {
    if (seeded.current || !wordData) return;
    seeded.current = true;
    const next = cloneLines(wordData.lines);
    reset(next);
    adopt(next, Number(wordData.revision) || 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wordData]);

  const { data: gestureData, ...gestures } = useWordGestures({
    lines,
    setLines,
    record,
    onSelect: setSelected,
    pxPerSec,
    autoShift,
    total,
    follow,
    onPanWhileFollowing: releaseFollow,
    scrollerRef,
  });

  const { data: editData, ...edits } = useTimingEdits({
    lines,
    setLines,
    record,
    selected,
    onSelect: setSelected,
    step,
    currentTime,
    autoShift,
    total,
  });
  const { targetLine } = editData;

  const { data: playbackData, ...playback } = useWordPlayback({ selectedWord, currentTime });
  const { loopWord } = playbackData;

  // ── Commands that span more than one hook ──────────────────────────────

  const revert = () => {
    record();
    setLines(cloneLines(savedLines));
    setSelected(null);
  };

  const reloadFromDisk = async () => {
    const fresh = await reload();
    reset(fresh.lines);
    setSelected(null);
  };

  const requestClose = () => {
    // Clean or already refused on disk: closing loses nothing, so it just closes.
    if (!dirty || stale) {
      onClose();
      return;
    }
    setConfirmDiscard(true);
  };

  const frameSelection = () => {
    if (!selectedWord) return;
    zoomToSpan(Math.max(FRAME_MIN_SPAN, (selectedWord.end - selectedWord.start) * FRAME_SPAN_FACTOR));
  };

  /** Download the .lrc for what is on screen; pending edits are written first. */
  const exportLyrics = async () => {
    if (stale) {
      toast.error(t('lyrics.editor.exportStale', 'The timings changed on disk. Reload before exporting.'));
      return;
    }
    if (dirty) await save();
    await downloadLyricsLrcForAudio(audioUrl);
  };

  // ── Keyboard ────────────────────────────────────────────────────────────

  // Declared as bindings rather than one switch, so every key names what it
  // does. Unspecified modifiers may be either; `false` means the key must not be
  // held, which is what separates Ctrl+Z from Ctrl+Shift+Z. The first match
  // wins, so the more specific binding comes first.
  useKeyboardShortcuts([
    // Escape works inside the step box too: it closes, it does not edit.
    { key: 'Escape', allowInField: true, run: requestClose },
    { key: 'z', ctrl: true, shift: false, run: undo },
    { key: 'z', ctrl: true, shift: true, run: redo },
    { key: 'y', ctrl: true, run: redo },
    { key: '+', ctrl: true, run: () => zoomBy(ZOOM_FACTOR) },
    { key: '=', ctrl: true, run: () => zoomBy(ZOOM_FACTOR) },
    { key: '-', ctrl: true, run: () => zoomBy(1 / ZOOM_FACTOR) },
    { key: '_', ctrl: true, run: () => zoomBy(1 / ZOOM_FACTOR) },
    // Space is the transport everywhere else in the app, and a focused word
    // chip must not swallow it into a click or scroll the pane.
    { code: 'Space', run: () => toggle() },
    {
      key: 'ArrowLeft', ctrl: false, alt: false, shift: false,
      when: () => !!selected,
      run: () => edits.wordStep(-1),
    },
    {
      key: 'ArrowRight', ctrl: false, alt: false, shift: false,
      when: () => !!selected,
      run: () => edits.wordStep(1),
    },
    {
      key: 'ArrowLeft', ctrl: false, alt: false, shift: true,
      when: () => !!selected,
      run: () => edits.resizeWordBy(-1),
    },
    {
      key: 'ArrowRight', ctrl: false, alt: false, shift: true,
      when: () => !!selected,
      run: () => edits.resizeWordBy(1),
    },
  ]);

  return ReactDOM.createPortal(
    <div
      data-portal-layer
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-[160] flex items-center justify-center bg-black/60 p-3 backdrop-blur-sm"
    >
      <div className="flex max-h-[94vh] w-[96vw] max-w-[1800px] flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-2xl dark:border-white/10 dark:bg-zinc-950">
        <LyricEditorHeader
          title={title}
          save={{ stale, saving, saveError, dirty }}
          canUndo={canUndo}
          canRedo={canRedo}
          handlers={history}
          onClose={requestClose}
        />

        <LyricPeaks
          peaks={engine.peaks}
          total={total}
          pxPerSec={pxPerSec}
          scrollLeft={scrollLeft}
        />

        <LyricEditorTimeline
          lines={lines}
          selected={selected}
          currentTime={currentTime}
          total={total}
          pxPerSec={pxPerSec}
          scrollerRef={scrollerRef}
          onScrollLeft={setScrollLeft}
          dragLabel={gestureData.dragLabel}
          gestures={gestures}
          canPageLeft={canPageLeft}
          canPageRight={canPageRight}
          onPage={pageBy}
        />

        <LyricEditorWordActions
          selectedWord={selectedWord}
          step={step}
          onStep={setStep}
          onNudge={edits.wordStep}
          onResize={edits.resizeWordBy}
          onSnapToPrevious={edits.snapToPrevious}
          onStartAtPlayhead={edits.setWordAtPlayhead}
          onPlayFromWord={playback.playFromWord}
          onToggleLoop={playback.toggleLoop}
          loopWord={loopWord}
          canMergePrev={targetLine > 0 && lines.length >= 2}
          canMergeNext={targetLine >= 0 && targetLine < lines.length - 1}
          onMerge={edits.mergeLine}
          onSplit={edits.splitLine}
        />

        <LyricEditorTransport
          currentTime={currentTime}
          total={total}
          follow={follow}
          onFollow={setFollow}
          autoShift={autoShift}
          onAutoShift={setAutoShift}
          onExport={() => { void exportLyrics(); }}
          exportDisabled={saving}
        />

        <LyricEditorFooter
          pxPerSec={pxPerSec}
          onZoom={setPxPerSec}
          onZoomBy={zoomBy}
          onGoToPlayhead={goToPlayhead}
          onFrameSelection={frameSelection}
          hasSelection={!!selectedWord}
          step={step}
          onShiftAll={edits.shiftAll}
          onShiftFromSelection={edits.shiftFromSelection}
          autoSave={autoSave}
          onAutoSave={setAutoSave}
          stale={stale}
          confirmDiscard={confirmDiscard}
          dirty={dirty}
          saving={saving}
          onReload={() => { void reloadFromDisk(); }}
          onDiscard={onClose}
          onKeepEditing={() => setConfirmDiscard(false)}
          onRevert={revert}
          onSave={() => { void save(); }}
        />

        <LyricEditorPreview lines={lines} currentTime={currentTime} />
      </div>
    </div>,
    document.body,
  );
};
