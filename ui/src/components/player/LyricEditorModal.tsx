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
//   `LyricEditorWordActions` line tools, the selected word's row, Re-align and
//                            Repair timings
//   `LyricEditorAlignmentReview` the run's diff, per-repair rows and Apply
//   `LyricEditorPreview`     the line being sung, with the View controls beside
//                            it: zoom at its left edge, playhead/jump/frame at
//                            its right
//   `LyricEditorTransport`   play, seek, speed, volume, Follow and Auto shift, with
//                            the File cluster (Auto save, Reload, Save, Export) at
//                            its right end and Export as the last control
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
// `useWordPlayback` (playing and looping one word), `useVisiblePeaks` (the
// waveform) and `useLyricAlign` (starting a run, following its stage and holding
// the preview until Apply). `lrcSeed` is the one input that is not a sidecar: the
// words of a line-level `.lrc`, spread over the time that file gives each line.
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
// Ctrl with them walks to the previous and next word, Shift+left/right resizes,
// Ctrl with - and = zooms, Ctrl+Z undoes, Escape closes. On a focused chip F2
// opens its text box. The cards name those keys too.
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
import { useTranslation } from 'react-i18next';
import { toast } from 'react-toastify';
import {
  getSnapshot as getEngineSnapshot, seek, subscribe as subscribeEngine, toggle,
} from '@/audio/engine';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';
import { lyricsJsonOptions } from '@/queries/shell-queue/lyrics';
import { downloadLyricsLrcForAudio } from '@/utils/audio/downloadTrack';
import { convertLyricsSectionToMelody } from '@/services/api';
import { LyricEditorAlignmentReview, REPAIR_REASON } from './lyric-editor/LyricEditorAlignmentReview';
import { LyricEditorFileControls } from './lyric-editor/LyricEditorFileControls';
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
import { useLyricAlign } from './lyric-editor/useLyricAlign';
import {
  ANCHOR_RATIO, PLAY_LEAD_SEC, STEP_DEFAULT, ZOOM_FACTOR,
  adlibRefs, cloneLines, doubtfulRefs, lineSpan, mergeProposed, refKey, selectedRefs,
  linesInSpan, sectionRuns, setLineChecked,
  setWordText, spanLineRange, spanTimeRange, spanWordCount, timingsLookRough, wordOrder,
  type EditLine, type SelectionSpan, type WordRef,
} from './lyric-editor/timingModel';
import { scopeFromAudioUrl } from '@/storage/trackScope';
import {
  editorPrefs, loadRealignSettings, saveRealignSettings, type RealignSettings,
} from '@/storage/lyricEditorSettings';
import { errorMessage } from '@/utils/format/errorMessage';

/** How long an undo holds autosave off, so the state it landed in is visible. */
const UNDO_HOLD_MS = 2500;

export interface LyricEditorModalProps {
  audioUrl: string;
  title?: string;
  duration: number;
  currentTime: number;
  /** What to open on when the track has no `.lyrics.json`: the words of a
   *  line-level `.lrc`, spread over the time it gives each line. The first save
   *  creates the word sidecar beside the audio. */
  seed?: EditLine[] | null;
  onClose: () => void;
}

export const LyricEditorModal: React.FC<LyricEditorModalProps> = ({
  audioUrl, title, duration, currentTime, seed, onClose,
}) => {
  const { t } = useTranslation();
  const wordData = useQuery(lyricsJsonOptions(audioUrl)).data ?? null;
  const engine = useSyncExternalStore(subscribeEngine, getEngineSnapshot);

  /**
   * The lines the editor opens on: the sidecar when the track has one, the
   * seeded ones otherwise. Read once — the query above may still be in flight,
   * and the effect below adopts a sidecar that lands later.
   */
  const [initialLines] = useState(() => {
    const fromSidecar = cloneLines(wordData?.lines);
    const fromSeed = cloneLines(seed);
    // A sidecar can carry lines and not one word; that is not something to
    // place, so the seed is only secondary to a sidecar that has words in it.
    return fromSidecar.some(line => line.words.length > 0)
      ? fromSidecar
      : fromSeed.length > 0 ? fromSeed : fromSidecar;
  });
  const { data: historyData, ...history } = useEditHistory(initialLines);
  const { lines, linesRef, canUndo, canRedo } = historyData;
  const { setLines, record, reset, replace, undo, redo } = history;

  // A selection is a span, not a word: re-alignment works on the lines it
  // touches, and shift-click has to be able to extend it without moving the
  // word the nudge buttons act on.
  const [selection, setSelection] = useState<SelectionSpan | null>(null);
  /**
   * The stretch drawn on the waveform, in seconds, when the selection came from
   * there.
   *
   * Kept apart from the word selection because its edges have to stay where the
   * person put them: a band that snapped to whole words would not move when a
   * handle is dragged past a long word. Every timeline selection clears it, so
   * the two can never point at different stretches of the track.
   */
  const [waveSpan, setWaveSpan] = useState<{ start: number; end: number } | null>(null);
  /** True while a selected section is being cut out and transcribed. */
  const [melodyBusy, setMelodyBusy] = useState(false);
  const selected = selection?.focus ?? null;
  /** The word just added, so its text box takes the cursor once. */
  const [freshWord, setFreshWord] = useState<WordRef | null>(null);

  /** Plain click selects one word; shift-click moves the far end. */
  const selectWord = (ref: WordRef, extend: boolean) => {
    setFreshWord(null);
    setWaveSpan(null);
    setSelection(prev => (extend && prev ? { anchor: prev.anchor, focus: ref } : { anchor: ref, focus: ref }));
  };

  /** The line's chip selects every word on it, widening an existing selection. */
  const selectLine = (li: number, extend: boolean) => {
    const span = lineSpan(lines[li], li);
    if (!span) return;
    setFreshWord(null);
    setWaveSpan(null);
    setSelection(prev => (extend && prev ? { anchor: prev.anchor, focus: span.focus } : span));
  };

  /** Split and merge set one word; they never keep a range alive. */
  const selectSingle = (ref: WordRef | null) => {
    setFreshWord(null);
    setWaveSpan(null);
    setSelection(ref ? { anchor: ref, focus: ref } : null);
  };
  const [step, setStep] = useState(STEP_DEFAULT);
  const [autoSave, setAutoSave] = useState(() => editorPrefs.autosave.read(true));
  useEffect(() => { editorPrefs.autosave.write(autoSave); }, [autoSave]);

  // Repair settings are per track: a sparse verse and a legato ballad want
  // different passes. The key is `realign-<track scope>:1`; a missing or older
  // value opens on the documented defaults.
  const trackScope = useMemo(() => scopeFromAudioUrl(audioUrl), [audioUrl]);
  const [realignSettings, setRealignSettings] = useState<RealignSettings>(
    () => loadRealignSettings(trackScope),
  );
  const changeRealignSettings = (next: RealignSettings) => {
    setRealignSettings(next);
    saveRealignSettings(trackScope, next);
  };
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
    // A seeded track has no sidecar to wait for; its first save creates one.
    enabled: !!wordData || !!seed?.length,
    revision: Number(wordData?.revision) || undefined,
  });
  const { dirty, saving, saveError, stale, savedLines } = saveState;
  const { save, adopt, reload, delayAutosave } = saver;

  /**
   * After a run the sidecar on disk is the truth: adopt it and drop the range.
   * `replace`, not `reset`: an Apply is one undoable step, so a diff that turns
   * out wrong is Ctrl+Z away rather than a fresh alignment.
   */
  const refreshAfterAlign = async () => {
    const fresh = await reload();
    replace(fresh.lines);
    setSelection(null);
    setWaveSpan(null);
  };
  const {
    data: alignData, run: runAlign, runPolish: runPolishAlign, apply: applyAlign, discard: discardAlign,
    markApplied,
  } = useLyricAlign({ audioUrl, onDone: refreshAfterAlign });

  /**
   * Take the run's sheet line by line.
   *
   * A mix is not the run's output, so it is written as an ordinary edit — one
   * undo entry, saved by the same autosave — rather than through the server's
   * accept, which writes all of it and would attach the run's diagnostics to a
   * sheet they no longer describe.
   */
  const applyMixedAlign = (next: EditLine[], summary: { taken: number; kept: number }) => {
    record();
    setLines(next);
    markApplied();
    toast.success(t(
      'lyrics.editor.reviewMixed',
      'Took {{taken}} lines from the run and kept {{kept}} as they were',
      summary,
    ));
  };

  /**
   * Take one row of the run now, leaving the other rows undecided.
   *
   * This is an ordinary edit, not the server's accept: one undo entry, saved by
   * the same autosave. The review stays on screen so the rest can be judged on
   * their own evidence, and a later bulk Apply writes only what is still open.
   */
  const applyAlignLine = (lineIndex: number) => {
    const proposed = alignData.preview?.result.proposed;
    if (!proposed?.[lineIndex]) return;
    const take = proposed.map(() => false);
    take[lineIndex] = true;
    record();
    setLines(mergeProposed(lines, proposed, take));
    toast.success(t(
      'lyrics.editor.reviewLineApplied',
      'Took line {{line}} from the run',
      { line: lineIndex + 1 },
    ));
  };

  /**
   * Take a section drawn, moved or resized on the waveform.
   *
   * The band is kept exactly as drawn, because it is what the loop replays; the
   * word selection is the lines it covers, because that is what Re-align turns
   * into whole lines. A section over no words still keeps its band — it has
   * simply nothing to align.
   */
  const setWaveSection = (next: { start: number; end: number } | null) => {
    setFreshWord(null);
    setWaveSpan(next);
    if (!next) {
      setSelection(null);
      return;
    }
    const covered = linesInSpan(lines, next.start, next.end);
    const last = covered ? lines[covered.toLine] : null;
    if (!covered || !last?.words.length) {
      setSelection(null);
      return;
    }
    setSelection({
      anchor: { line: covered.fromLine, word: 0 },
      focus: { line: covered.toLine, word: last.words.length - 1 },
    });
  };

  const { data: view, ...timeline } = useTimelineView({
    scrollerRef,
    lines,
    currentTime,
    total,
    // A section drawn on the waveform is what the person is looking at, and the
    // word selection it makes covers whole lines: its last word can sit well
    // past the band's end, so anchoring there zoomed the band out of the window.
    anchorTime: waveSpan
      ? (waveSpan.start + waveSpan.end) / 2
      : selectedWord?.start ?? currentTime,
  });
  const {
    pxPerSec, scrollLeft, follow, canPageLeft, canPageRight,
  } = view;
  const {
    setPxPerSec, zoomBy, setScrollLeft, setFollow, goToPlayhead,
    revealWord, frameWord, releaseFollow, pageBy,
  } = timeline;

  // The sidecar can land after the modal opens (slow disk, first read).
  const seeded = useRef(!!wordData);
  useEffect(() => {
    if (seeded.current || !wordData) return;
    seeded.current = true;
    const next = cloneLines(wordData.lines);
    // Same rule as the opening state: a wordless sidecar does not replace the
    // seeded lines someone is already placing against.
    if (seed?.length && !next.some(line => line.words.length > 0)) return;
    reset(next);
    adopt(next, Number(wordData.revision) || 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wordData]);

  const { data: gestureData, ...gestures } = useWordGestures({
    lines,
    setLines,
    record,
    onSelect: selectWord,
    selection,
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
    selection,
    onSelect: selectSingle,
    step,
    currentTime,
    autoShift,
    total,
  });
  const { targetLine } = editData;
  const targetSection = targetLine >= 0 ? lines[targetLine]?.section ?? '' : '';

  /**
   * The track's sections, in order, each with the line it starts on.
   *
   * A section is a label on a line, not a row of its own, so this is what the
   * add and delete buttons act on: the run start is the line whose label has to
   * come off, and a chorus that comes back is listed once per appearance.
   */
  const sectionRunsInTrack = useMemo(() => sectionRuns(lines), [lines]);

  /** Add a word, then hand the cursor to its box so typing replaces the placeholder. */
  const addWord = () => {
    const at = edits.addWord();
    if (at) setFreshWord(at);
  };

  /**
   * Merging across a section boundary is legal but lossy: one label goes. It is
   * said out loud with both names instead of refused, because a line that
   * straddles a chorus is a real thing to have to represent.
   */
  const mergeLines = (dir: -1 | 1) => {
    const from = lines[targetLine]?.section?.trim();
    const to = lines[targetLine + dir]?.section?.trim();
    edits.mergeLine(dir);
    if (from && to && from !== to) {
      toast.info(t(
        'lyrics.editor.mergeSections',
        'Merged [{{from}}] with [{{to}}]: the line keeps [{{from}}]. Use the section field if it should read otherwise.',
        { from, to },
      ));
    }
  };

  // Checking a range by ear means hearing the range: playback starts at the
  // first selected word and loops to the end of the last one.
  const picked = selectedRefs(lines, selection);
  const pickedFirst = picked[0] ? lines[picked[0].line]?.words?.[picked[0].word] ?? null : null;
  const pickedLast = picked.length > 1 && picked.at(-1)
    ? lines[picked.at(-1)!.line]?.words?.[picked.at(-1)!.word] ?? null
    : null;
  const playbackWord = pickedFirst && pickedLast ? { ...pickedFirst, end: pickedLast.end } : selectedWord;

  /** The band on the waveform: the drawn section, or the words' own span. */
  const waveBand = useMemo(
    () => waveSpan ?? (selection ? spanTimeRange(lines, selection) : null),
    [waveSpan, selection, lines],
  );

  // The line loop needs a stable span, or its restart effect re-arms on every
  // render and the audio never gets past the line's first word.
  const targetLineSpan = useMemo(() => {
    const line = targetLine >= 0 ? lines[targetLine] : null;
    return line ? { start: line.start, end: line.end } : null;
  }, [lines, targetLine]);

  const { data: playbackData, ...playback } = useWordPlayback({
    selectedWord: playbackWord, span: waveBand, line: targetLineSpan, currentTime,
  });
  const { loopWord, loopLine } = playbackData;

  /**
   * Jump to a word a repair moved: select it, seek just before its onset and
   * scroll it into view. The selection is what makes the word actions row act
   * on it, so the next press of Play from word plays the syllable in question.
   */
  const jumpToWord = (ref: WordRef) => {
    const word = lines[ref.line]?.words?.[ref.word];
    if (!word) return;
    selectWord(ref, false);
    seek(Math.max(0, word.start - PLAY_LEAD_SEC));
    const el = scrollerRef.current;
    if (el) el.scrollLeft = Math.max(0, word.start * pxPerSec - el.clientWidth * ANCHOR_RATIO);
  };

  /**
   * Jump to a section's first line from its chip: the same move a repair row
   * makes, so the timeline scrolls there and the word actions target the line.
   * A line with no words (an instrumental marker) can only seek.
   */
  const jumpToSection = (lineIndex: number) => {
    const line = lines[lineIndex];
    if (!line) return;
    if (line.words.length > 0) {
      jumpToWord({ line: lineIndex, word: 0 });
      return;
    }
    seek(Math.max(0, line.start - PLAY_LEAD_SEC));
    const el = scrollerRef.current;
    if (el) el.scrollLeft = Math.max(0, line.start * pxPerSec - el.clientWidth * ANCHOR_RATIO);
  };

  /**
   * Jump to a preview row's time. A review row can be heard either as the run
   * placed it or as the line is on screen, so this seeks only; whether it is
   * the old or the new time is the row's choice, not this handler's.
   */
  const jumpToTime = (time: number) => {
    seek(Math.max(0, time - PLAY_LEAD_SEC));
    const el = scrollerRef.current;
    if (el) el.scrollLeft = Math.max(0, time * pxPerSec - el.clientWidth * ANCHOR_RATIO);
  };

  /**
   * Retype a word from its chip, leaving its times where they are. Kept apart
   * from `edits.renameWord` because that one acts on the selection: a chip being
   * edited is not always the selection the toolbar would act on.
   */
  const renameWordAt = (ref: WordRef, text: string) => {
    const next = setWordText(lines, ref.line, ref.word, text);
    if (next === lines) return;
    record();
    setLines(next);
  };

  /**
   * The words the aligner was least sure of, in reading order. They are
   * underlined on the timeline and the transport's button walks them: the run
   * already knows where it was weak, and that list is the shortest way to spend
   * a minute of listening.
   */
  const doubtful = useMemo(() => doubtfulRefs(lines), [lines]);
  // Ad-libs are marked but never queued: they are the singer's own additions,
  // so walking them as "check this" mostly teaches people to delete true words.
  const adlib = useMemo(() => adlibRefs(lines), [lines]);
  const doubtfulKeys = useMemo(() => new Set(doubtful.map(refKey)), [doubtful]);
  const adlibKeys = useMemo(() => new Set(adlib.map(refKey)), [adlib]);

  /** Walk the doubtful words, wrapping; the same jump the repair rows use. */
  const checkNextDoubtful = () => {
    if (doubtful.length === 0) return;
    const at = selected
      ? doubtful.findIndex(ref => ref.line === selected.line && ref.word === selected.word)
      : -1;
    // With nothing selected the queue starts at the playhead, not at the top of
    // the track: the person is already somewhere, and the next doubtful word is
    // the one after them.
    const after = doubtful.findIndex(
      ref => (lines[ref.line]?.words?.[ref.word]?.start ?? 0) >= currentTime,
    );
    jumpToWord(at >= 0
      ? doubtful[(at + 1) % doubtful.length]
      : doubtful[after < 0 ? 0 : after]);
  };

  /**
   * The words a just-applied pass moved, keyed `line:word`.
   *
   * Only while the array still is the one that was applied: the repairs are
   * positions, and the first edit, merge or delete renumbers them, so a mark
   * that outlived an edit would point at the wrong syllable.
   */
  const repaired = useMemo(() => {
    const marks = new Map<string, string>();
    if (!alignData.preview?.applied || dirty) return marks;
    for (const repair of alignData.preview.result.repairs ?? []) {
      marks.set(`${repair.line}:${repair.word}`, REPAIR_REASON[repair.reason] ?? repair.reason);
    }
    return marks;
  }, [alignData.preview, dirty]);

  /**
   * Tick a line as placed by ear, and move on to the next one.
   *
   * The tick ends a check, so the next thing the person needs is the next line
   * ready to play: selecting its first word and seeking just before it turns
   * "listen, tick, find the next line" into "listen, tick". Ticking off does not
   * move anything — it is a correction, not a step in the loop.
   */
  const toggleChecked = (li: number) => {
    const line = lines[li];
    if (!line) return;
    const turningOn = !line.checked;
    record();
    setLines(prev => setLineChecked(prev, li, turningOn));
    if (!turningOn) return;
    const next = lines[li + 1]?.words?.[0];
    if (!next) return;
    selectWord({ line: li + 1, word: 0 }, false);
    // While it is playing, seeking would skip the music being judged; the
    // selection simply moves on and the audio carries the next line in.
    if (!engine.playing) seek(Math.max(0, next.start - PLAY_LEAD_SEC));
    const el = scrollerRef.current;
    if (el) el.scrollLeft = Math.max(0, next.start * pxPerSec - el.clientWidth * ANCHOR_RATIO);
  };

  // ── Commands that span more than one hook ──────────────────────────────

  const revert = () => {
    record();
    setLines(cloneLines(savedLines));
    setSelection(null);
    setWaveSpan(null);
  };

  /**
   * Undo still saves itself, a moment later: an undo that reverses an applied
   * diff should be visible, and redoable, before the file changes to match it.
   */
  const undoEdit = () => {
    undo();
    delayAutosave(UNDO_HOLD_MS);
  };
  const redoEdit = () => {
    redo();
    delayAutosave(UNDO_HOLD_MS);
  };

  const reloadFromDisk = async () => {
    const fresh = await reload();
    replace(fresh.lines);
    setSelection(null);
    setWaveSpan(null);
  };

  const requestClose = () => {
    // Clean or already refused on disk: closing loses nothing, so it just closes.
    if (!dirty || stale) {
      onClose();
      return;
    }
    setConfirmDiscard(true);
  };

  /**
   * Whether to offer an alignment from the header. The reason this editor can
   * tell is the timings themselves: a sheet split out of line times, or one
   * whose words all share an instant, is not something anyone can check by ear,
   * and "go to the transport and press Re-align" assumes a word was selected
   * first. A track the aligner placed is left alone — its remaining work is
   * listening, which is what the doubtful queue is for.
   */
  // A seeded editor is rough by construction even though its chips have lengths:
  // the words were spread inside line times nobody listened to, which is exactly
  // the case that wants the aligner. Once a sidecar is what is open, the
  // timings themselves decide.
  const roughTimings = useMemo(
    () => timingsLookRough(lines, total) || (!wordData && !!seed?.length),
    [lines, total, wordData, seed],
  );

  /** Frame the selected word and jump to it on both axes. */
  const frameSelection = () => {
    if (!selectedWord || !selected) return;
    frameWord(selectedWord.start, selectedWord.end, selected.line);
  };

  /** Bring the selected word into view without zooming or moving the playhead. */
  const jumpToSelection = () => {
    if (!selectedWord || !selected) return;
    revealWord(selectedWord.start, selected.line);
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

  /**
   * Turn the selected stretch of waveform into a song of its own.
   *
   * This is not an edit: the source stays untouched. The server cuts the range
   * into a new library render named after this track and the moment, transcribes
   * its YuE2 ABC lead sheet, and stores both, so the section can be arranged or
   * covered on its own.
   */
  const convertSectionToMelody = async () => {
    if (!waveBand || melodyBusy) return;
    setMelodyBusy(true);
    try {
      const result = await convertLyricsSectionToMelody(
        audioUrl, waveBand.start, waveBand.end, title,
      );
      toast.success(t(
        'lyrics.editor.melodySaved',
        'Saved “{{title}}” to the library with its YuE2 ABC score',
        { title: result.song.title },
      ));
    } catch (err) {
      toast.error(t(
        'lyrics.editor.melodyFailed',
        'Could not convert this section: {{error}}',
        { error: errorMessage(err) },
      ));
    } finally {
      setMelodyBusy(false);
    }
  };

  /**
   * Repair with the panel's switches. No saved word timings means there is
   * nothing to repair, and the useful run is the full alignment.
   */
  const applyRepair = () => {
    if (wordData) { void runPolishAlign(realignSettings); return; }
    toast.info(t(
      'lyrics.editor.repairNeedsAlign',
      'This track has no saved word timings yet, so there is nothing to repair. Starting a full alignment instead.',
    ));
    void realign();
  };

  /** Re-align the selection, or the whole track with nothing selected. */
  const realign = async () => {
    if (stale) {
      toast.error(t('lyrics.editor.realignStale', 'The timings changed on disk. Reload before aligning.'));
      return;
    }
    // The server reads the sidecar, so pending placements must land first or the
    // re-aligned lines would replace timings the editor has already moved on.
    if (dirty) await save();
    const linesRange = spanLineRange(lines, selection);
    const times = spanTimeRange(lines, selection);
    // The server picks the lines by time when an index drifted, and it reads
    // those times as `fromStart`/`toStart`; `spanTimeRange` calls them
    // start/end, so spreading it sent keys the run never looked at.
    await runAlign(linesRange
      ? { ...linesRange, ...(times ? { fromStart: times.start, toStart: times.end } : {}) }
      : null);
  };

  /**
   * Move the selection one word, without moving its timing: walking the words
   * is the other half of placing them, and Ctrl with the arrows keeps the hand
   * on the keys while checking a line by ear. It scrolls only when the word
   * would be off screen, so it does not fight Follow.
   */
  const stepSelection = (dir: 1 | -1) => {
    const order = wordOrder(lines);
    if (order.length === 0) return;
    const at = selected
      ? order.findIndex(ref => ref.line === selected.line && ref.word === selected.word)
      : -1;
    let index: number;
    if (at >= 0) {
      index = Math.max(0, Math.min(order.length - 1, at + dir));
    } else {
      // Nothing selected yet: walk from the playhead, not from an end of the
      // track, so a key press lands near where the person is listening.
      const after = order.findIndex(
        ref => (lines[ref.line]?.words?.[ref.word]?.start ?? 0) >= currentTime,
      );
      index = after < 0 ? order.length - 1 : (dir === 1 ? after : Math.max(0, after - 1));
    }
    const ref = order[index];
    selectWord(ref, false);
    const word = lines[ref.line]?.words?.[ref.word];
    const el = scrollerRef.current;
    if (!word || !el) return;
    const x = word.start * pxPerSec;
    if (x < el.scrollLeft || x > el.scrollLeft + el.clientWidth - 40) {
      el.scrollLeft = Math.max(0, x - el.clientWidth * ANCHOR_RATIO);
    }
  };

  // ── Keyboard ────────────────────────────────────────────────────────────

  // Declared as bindings rather than one switch, so every key names what it
  // does. Unspecified modifiers may be either; `false` means the key must not be
  // held, which is what separates Ctrl+Z from Ctrl+Shift+Z. The first match
  // wins, so the more specific binding comes first.
  useKeyboardShortcuts([
    // Escape works inside the step box too: it closes, it does not edit.
    { key: 'Escape', allowInField: true, run: requestClose },
    { key: 'z', ctrl: true, shift: false, run: undoEdit },
    { key: 'z', ctrl: true, shift: true, run: redoEdit },
    { key: 'y', ctrl: true, run: redoEdit },
    // Ctrl with the arrows walks the words; plain arrows nudge the selected one,
    // so the two jobs stay one modifier apart.
    { key: 'ArrowLeft', ctrl: true, alt: false, shift: false, run: () => stepSelection(-1) },
    { key: 'ArrowRight', ctrl: true, alt: false, shift: false, run: () => stepSelection(1) },
    // Page keys move the window, not the playhead: the same page the arrows by
    // the timeline's edge do, for a keyboard that has no reason to reach for a
    // mouse to look further along the track.
    { key: 'PageUp', run: () => pageBy(-1) },
    { key: 'PageDown', run: () => pageBy(1) },
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

  const provenance = wordData ? {
    method: wordData.method,
    vocalsIsolated: wordData.vocalsIsolated,
    dtw: wordData.diagnostics?.dtw,
    wordsMoved: wordData.diagnostics?.wordsMoved,
    fallback: wordData.diagnostics?.fallback,
    // Only a full-track run records this, so a file aligned by a selection run
    // alone shows the badge it always did.
    anchorDrift: wordData.diagnostics?.anchorDrift ?? null,
  } : seed?.length ? {
    // No word sidecar: these chips came from the `.lrc`'s line times, and the
    // badge says so rather than implying that anything heard the words.
    method: 'lrc lines',
    detail: 'Split from the .lrc\u2019s line times: this track has no word timings yet, so each word starts somewhere inside its line and the placements are a guess until they are checked by ear.',
  } : null;

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
          provenance={provenance}
          save={{ stale, saving, saveError, dirty }}
          busy={alignData.running ? { label: alignData.stage || 'starting' } : null}
          align={roughTimings && !alignData.preview ? {
            onAlign: () => { void realign(); },
            aligning: alignData.running,
            disabled: saving || stale,
          } : undefined}
          canUndo={canUndo}
          canRedo={canRedo}
          handlers={{ ...history, undo: undoEdit, redo: redoEdit }}
          onClose={requestClose}
        />

        {alignData.preview && (
          <LyricEditorAlignmentReview
            key={alignData.preview.jobId}
            preview={alignData.preview}
            current={lines}
            applying={alignData.applying}
            applied={alignData.preview.applied}
            onApplyNew={() => { void applyAlign(); }}
            onApplyMix={applyMixedAlign}
            onJump={jumpToWord}
            onJumpTime={jumpToTime}
            onFocusLine={index => {
              selectLine(index, false);
              // The lanes exist only for the selected line, so a row clicked in
              // the strip has to be brought into view or it is selected off
              // screen. Wait for that selection to render: the selected row may
              // grow into its two-lane form, and the scroll has to measure the
              // row it actually became, not the one it was a frame earlier.
              const line = lines[index];
              if (line) requestAnimationFrame(() => revealWord(line.start, index));
            }}
            onApplyLine={applyAlignLine}
            onDiscard={discardAlign}
          />
        )}

        {/* The waveform never shrinks out of the modal: a preview above it is
            capped and scrolls, so the picture being judged stays visible. */}
        <div className="shrink-0">
          <LyricPeaks
            peaks={engine.peaks}
            total={total}
            pxPerSec={pxPerSec}
            scrollLeft={scrollLeft}
            span={waveBand}
            onSpanChange={setWaveSection}
            scrollerRef={scrollerRef}
            looping={loopWord}
            onLoop={playback.toggleLoop}
            onRealign={() => { void realign(); }}
            // A band over no words has nothing to align, and the whole-track run
            // is what the header's Align button is for.
            realignDisabled={alignData.running || saving || !selection}
            onMelody={() => { void convertSectionToMelody(); }}
            melodyBusy={melodyBusy}
          />
        </div>

        <LyricEditorTimeline
          lines={lines}
          selection={selection}
          onLineSelect={selectLine}
          onLineCheck={toggleChecked}
          onRenameWord={renameWordAt}
          doubtful={doubtfulKeys}
          adlib={adlibKeys}
          repaired={repaired}
          proposed={alignData.preview && !alignData.preview.applied ? alignData.preview.result.proposed : null}
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

        {/* The sung line sits above the controls: it is feedback on the timing
            rows below it, not another row at the very bottom of the editor. */}
        <LyricEditorPreview
          lines={lines}
          currentTime={currentTime}
          pxPerSec={pxPerSec}
          onZoom={setPxPerSec}
          onZoomBy={zoomBy}
          onGoToPlayhead={goToPlayhead}
          onJumpToSelection={jumpToSelection}
          onFrameSelection={frameSelection}
          hasSelection={!!selectedWord}
        />

        <LyricEditorWordActions
          selectedWord={selectedWord}
          selectionCount={spanWordCount(lines, selection)}
          selectionLines={spanLineRange(lines, selection)}
          selectionRange={waveBand}
          onRealign={() => { void realign(); }}
          realignDisabled={alignData.running || saving}
          // Repair needs timings on disk. With none saved yet there is nothing
          // to repair, and the task the button promises is a full alignment.
          onRepairTimings={applyRepair}
          repairDisabled={alignData.running || saving}
          repairSettings={realignSettings}
          onRepairSettingsChange={changeRealignSettings}
          alignStage={alignData.running ? alignData.stage : null}
          alignError={alignData.error}
          targetLine={targetLine}
          targetSection={targetSection}
          onSectionChange={edits.renameSection}
          onAddSection={edits.addSection}
          onDeleteSection={edits.deleteSection}
          onJumpSection={jumpToSection}
          targetLineData={targetLine >= 0 ? lines[targetLine] ?? null : null}
          onApplyLine={edits.applyLinePlan}
          sections={sectionRunsInTrack}
          autoFocusWord={!!freshWord && selected?.line === freshWord.line && selected?.word === freshWord.word}
          onWordAutoFocus={() => setFreshWord(null)}
          onAddWord={addWord}
          onDeleteWord={edits.deleteWord}
          step={step}
          onStep={setStep}
          onShiftAll={edits.shiftAll}
          onShiftFromSelection={edits.shiftFromSelection}
          onNudge={edits.wordStep}
          onResize={edits.resizeWordBy}
          onSnapToPrevious={edits.snapToPrevious}
          onStartAtPlayhead={edits.setWordAtPlayhead}
          onPlayFromWord={playback.playFromWord}
          onToggleLoop={playback.toggleLoop}
          loopWord={loopWord}
          onToggleLineLoop={playback.toggleLineLoop}
          loopLine={loopLine}
          canMergePrev={targetLine > 0 && lines.length >= 2}
          canMergeNext={targetLine >= 0 && targetLine < lines.length - 1}
          onMerge={mergeLines}
          onSplit={edits.splitLine}
        />

        <LyricEditorTransport
          currentTime={currentTime}
          total={total}
          follow={follow}
          onFollow={setFollow}
          autoShift={autoShift}
          onAutoShift={setAutoShift}
          doubtfulCount={doubtful.length}
          onCheckDoubtful={checkNextDoubtful}
          right={(
            <LyricEditorFileControls
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
              onExport={() => { void exportLyrics(); }}
              exportDisabled={saving}
            />
          )}
        />

      </div>
    </div>,
    document.body,
  );
};
