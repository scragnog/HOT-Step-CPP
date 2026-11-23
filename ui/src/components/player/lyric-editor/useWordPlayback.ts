// useWordPlayback.ts — checking a word, or a whole line, by ear.
//
// Three behaviours: play from just before a word, loop that word, and loop the
// line it belongs to. A loop that merely re-seeked once past the end played the
// gap first, so arming it seeks a fifth of a second early and the check starts
// on the syllable; the effects below restart it the moment playback passes the
// end. The two loops are exclusive: arming one releases the other, so the audio
// is never fighting two restart rules at once.
//
// The audio is the app's one player, so the hook reads the engine's snapshot
// itself: the modal no longer has to hand it `playing` just to decide whether a
// loop should restart. It reads it at the moment of the decision rather than
// from the last render, because the two disagree for as long as it takes React
// to catch up — long enough for a second click on Play from word (or a restart
// at the loop's end) to find a stale `false` and pause the audio it just
// started.

import { useEffect, useMemo, useState } from 'react';
import { getSnapshot as getEngineSnapshot, seek, toggle } from '@/audio/engine';
import {
  LOOP_LEAD_SEC, LOOP_SLACK_SEC, PLAY_LEAD_SEC, type EditWord,
} from './timingModel';

export interface WordPlaybackOptions {
  /** The word under the selection, if any. */
  selectedWord: EditWord | null;
  /** A section drawn on the waveform, when there is one: the loop replays this
   *  instead of the words' own span, so moving its handles changes what is
   *  heard rather than only what is highlighted. */
  span?: { start: number; end: number } | null;
  /** The line the line tools act on, as its span in seconds. */
  line?: { start: number; end: number } | null;
  currentTime: number;
}

/** Start the audio only if it is not already running. */
function playIfPaused(): void {
  if (!getEngineSnapshot().playing) toggle();
}

export function useWordPlayback({ selectedWord, span, line, currentTime }: WordPlaybackOptions) {
  const [loopWord, setLoopWord] = useState(false);
  const [loopLine, setLoopLine] = useState(false);
  /** The span the loop was armed on, kept so clearing the selection does not
   *  leave an armed toggle with nothing to replay. */
  const [armedSpan, setArmedSpan] = useState<{ start: number; end: number } | null>(null);

  /** A waveform section outranks the words' span: its handles are what the
   *  person moved, so they are what should be heard. */
  const loopSpan = useMemo(
    () => span ?? (selectedWord ? { start: selectedWord.start, end: selectedWord.end } : null),
    [span, selectedWord],
  );

  /** Arming the loop seeks just before the span, so the check starts at once. */
  const toggleLoop = () => {
    if (loopWord) {
      setLoopWord(false);
      setArmedSpan(null);
      return;
    }
    setLoopLine(false);
    setLoopWord(true);
    setArmedSpan(loopSpan);
    if (loopSpan) seek(Math.max(0, loopSpan.start - LOOP_LEAD_SEC));
  };

  /** The same, for the phrase: a line checked as a whole keeps its rhythm. */
  const toggleLineLoop = () => {
    if (loopLine) {
      setLoopLine(false);
      return;
    }
    setLoopWord(false);
    setLoopLine(true);
    if (line) seek(Math.max(0, line.start - LOOP_LEAD_SEC));
  };

  /** A lead-in, so the first syllable is audible. */
  const playFromWord = () => {
    if (!selectedWord) return;
    seek(Math.max(0, selectedWord.start - PLAY_LEAD_SEC));
    playIfPaused();
  };

  // The live span wins while there is one, so dragging the waveform's handles
  // replays the new stretch at once; the span captured when the loop was armed
  // is the fallback once the selection is cleared.
  const heard = loopSpan ?? armedSpan;
  useEffect(() => {
    if (!loopWord || !heard) return;
    if (currentTime > heard.end + LOOP_SLACK_SEC) {
      seek(Math.max(0, heard.start - LOOP_LEAD_SEC));
      playIfPaused();
    }
  }, [currentTime, loopWord, heard]);

  useEffect(() => {
    if (!loopLine || !line) return;
    if (currentTime > line.end + LOOP_SLACK_SEC) {
      seek(Math.max(0, line.start - LOOP_LEAD_SEC));
      playIfPaused();
    }
  }, [currentTime, loopLine, line]);

  return {
    data: { loopWord, loopLine },
    toggleLoop, toggleLineLoop, playFromWord,
  };
}

/** The hook's whole return, so its shape is never written twice. */
export type WordPlayback = ReturnType<typeof useWordPlayback>;
/** What a component may call: the handlers, without the state. */
export type WordPlaybackHandlers = Omit<WordPlayback, 'data'>;
