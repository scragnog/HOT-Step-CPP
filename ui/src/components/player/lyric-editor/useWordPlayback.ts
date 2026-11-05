// useWordPlayback.ts — checking one word by ear.
//
// Two behaviours, both about a single word: play from just before it, and loop
// it. A loop that merely re-seeked once past the end played the gap first, so
// arming it seeks a fifth of a second early and the check starts on the
// syllable; the effect below restarts it the moment playback passes the word's
// end.
//
// The audio is the app's one player, so the hook reads the engine's snapshot
// itself: the modal no longer has to hand it `playing` just to decide whether a
// loop should restart.

import { useEffect, useState, useSyncExternalStore } from 'react';
import {
  getSnapshot as getEngineSnapshot, seek, subscribe as subscribeEngine, toggle,
} from '../../../audio/engine';
import {
  LOOP_LEAD_SEC, LOOP_SLACK_SEC, PLAY_LEAD_SEC, type EditWord,
} from './timingModel';

export interface WordPlaybackOptions {
  /** The word under the selection, if any. */
  selectedWord: EditWord | null;
  currentTime: number;
}

export function useWordPlayback({ selectedWord, currentTime }: WordPlaybackOptions) {
  const engine = useSyncExternalStore(subscribeEngine, getEngineSnapshot);
  const [loopWord, setLoopWord] = useState(false);

  /** Arming the loop seeks just before the word, so the check starts at once. */
  const toggleLoop = () => {
    if (loopWord) {
      setLoopWord(false);
      return;
    }
    setLoopWord(true);
    if (selectedWord) seek(Math.max(0, selectedWord.start - LOOP_LEAD_SEC));
  };

  /** A lead-in, so the first syllable is audible. */
  const playFromWord = () => {
    if (!selectedWord) return;
    seek(Math.max(0, selectedWord.start - PLAY_LEAD_SEC));
    if (!engine.playing) toggle();
  };

  useEffect(() => {
    if (!loopWord || !selectedWord) return;
    if (currentTime > selectedWord.end + LOOP_SLACK_SEC) {
      seek(Math.max(0, selectedWord.start - LOOP_LEAD_SEC));
      if (!engine.playing) toggle();
    }
  }, [currentTime, loopWord, selectedWord, engine.playing]);

  return { data: { loopWord }, toggleLoop, playFromWord };
}

/** The hook's whole return, so its shape is never written twice. */
export type WordPlayback = ReturnType<typeof useWordPlayback>;
/** What a component may call: the handlers, without the state. */
export type WordPlaybackHandlers = Omit<WordPlayback, 'data'>;
