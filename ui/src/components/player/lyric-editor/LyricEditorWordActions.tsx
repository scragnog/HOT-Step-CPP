// LyricEditorWordActions.tsx — the selected word's row, and the line actions
// that sit at its right.
//
// Left: what is selected, the step size, and the ways to move or stretch it.
// Right: the line tools, split and the two merges.
//
// The step box's draft lives here rather than in the editor's shell. Nothing
// else can type into it, so the half-typed value is this row's private state;
// only the committed number travels up, through `onStep`. The clamping rule is
// `timingModel`'s `clampStep`, so the range is one definition for the box and
// the keyboard that reads the same step.

import React, { useState } from 'react';
import {
  ArrowLeftToLine, ChevronsDown, ChevronsUp, Play, Scissors,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { HintButton } from '../../shared/HintButton';
import { ParamLabel } from '../../shared/ParamLabel';
import { BTN, ROW } from './editorStyles';
import {
  STEP_MAX, STEP_MIN, clampStep, fmt, type EditWord,
} from './timingModel';

export interface LyricEditorWordActionsProps {
  selectedWord: EditWord | null;
  /** The committed step, in milliseconds, shared with the offsets and the keys. */
  step: number;
  onStep: (step: number) => void;
  onNudge: (dir: 1 | -1) => void;
  onResize: (dir: 1 | -1) => void;
  onSnapToPrevious: () => void;
  onStartAtPlayhead: () => void;
  onPlayFromWord: () => void;
  onToggleLoop: () => void;
  loopWord: boolean;
  canMergePrev: boolean;
  canMergeNext: boolean;
  onMerge: (dir: -1 | 1) => void;
  onSplit: () => void;
}

export const LyricEditorWordActions: React.FC<LyricEditorWordActionsProps> = ({
  selectedWord, step, onStep, onNudge, onResize, onSnapToPrevious, onStartAtPlayhead,
  onPlayFromWord, onToggleLoop, loopWord, canMergePrev, canMergeNext, onMerge, onSplit,
}) => {
  const { t } = useTranslation();
  // What is typed, until it is committed. Typing 100 no longer passes through
  // 1 and 10, because the value is read on blur or Enter.
  const [draft, setDraft] = useState(String(step));

  const commitStep = () => {
    const value = clampStep(Number(draft));
    setDraft(String(value));
    onStep(value);
  };

  return (
    <div className={ROW}>
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
              type="number" min={STEP_MIN} max={STEP_MAX} value={draft}
              onChange={e => setDraft(e.target.value)}
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
            onClick={() => onNudge(-1)}
          >
            -{step}
          </HintButton>
          <HintButton
            label={t('lyrics.editor.nudgeLater', 'Move later')}
            info={t('lyrics.editor.shiftLaterHint', 'Moves this word later by the step, stopping before the next word. Right arrow does the same.')}
            className={BTN}
            onClick={() => onNudge(1)}
          >
            +{step}
          </HintButton>
          <HintButton
            label={t('lyrics.editor.shorter', 'Shorten')}
            info={t('lyrics.editor.shorterHint', 'Makes this word end earlier by the step. It cannot end before it starts, and the next word does not move. Shift+Left arrow does the same.')}
            className={BTN}
            onClick={() => onResize(-1)}
          >
            {t('lyrics.editor.len', 'Len')} -{step}
          </HintButton>
          <HintButton
            label={t('lyrics.editor.longer', 'Lengthen')}
            info={t('lyrics.editor.longerHint', 'Makes this word last longer by the step. With Auto shift on, everything after it moves later by the overhang; off, it stops at the next word. Shift+Right arrow does the same.')}
            className={BTN}
            onClick={() => onResize(1)}
          >
            {t('lyrics.editor.len', 'Len')} +{step}
          </HintButton>
          <HintButton
            label={t('lyrics.editor.snap', 'Snap to prev')}
            info={t('lyrics.editor.snapHint', 'Starts this word exactly where the previous word ends, across a line break too. The fix for a word that landed with a gap before it.')}
            className={BTN}
            onClick={onSnapToPrevious}
          >
            <ArrowLeftToLine size={12} /> {t('lyrics.editor.snap', 'Snap to prev')}
          </HintButton>
          <HintButton
            label={t('lyrics.editor.setStart', 'Start at playhead')}
            info={t('lyrics.editor.setStartInfo', 'Moves this word\u2019s onset to the playhead, clamped like any move: after the previous word and early enough to finish before the next one.')}
            className={BTN}
            onClick={onStartAtPlayhead}
          >
            {t('lyrics.editor.setStart', 'Start at playhead')}
          </HintButton>
          <HintButton
            label={t('lyrics.editor.playFrom', 'Play from word')}
            info={t('lyrics.editor.playFromHint', 'Seeks a quarter second before this word and plays, so a check starts on the syllable instead of a minute earlier.')}
            className={BTN}
            onClick={onPlayFromWord}
          >
            <Play size={12} /> {t('lyrics.editor.playFrom', 'Play from word')}
          </HintButton>
          <HintButton
            label={t('lyrics.editor.loop', 'Loop word')}
            info={t('lyrics.editor.loopHint', 'Replays this word until it is switched off. Arming it seeks just before the word, and the loop restarts whenever playback passes its end.')}
            className={`${BTN} ${loopWord ? 'border-pink-400 text-pink-400' : ''}`}
            onClick={onToggleLoop}
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
          onClick={() => onMerge(-1)}
          disabled={!canMergePrev}
        >
          <ChevronsUp size={12} /> {t('lyrics.editor.mergePrev', 'Merge previous')}
        </HintButton>
        <HintButton
          label={t('lyrics.editor.mergeNext', 'Merge next')}
          info={t('lyrics.editor.mergeNextHint', 'Folds the line below into this one. The words are re-sorted by time and this line keeps its section label.')}
          className={BTN}
          onClick={() => onMerge(1)}
          disabled={!canMergeNext}
        >
          <ChevronsDown size={12} /> {t('lyrics.editor.mergeNext', 'Merge next')}
        </HintButton>
        <HintButton
          label={t('lyrics.editor.split', 'Split line')}
          info={t('lyrics.editor.splitHint', 'Starts a new line at the selected word, or at the word under the playhead when nothing is selected. The selected word becomes the first word of the new line, so the break lands where that syllable is sung.')}
          className={BTN}
          onClick={onSplit}
        >
          <Scissors size={12} /> {t('lyrics.editor.split', 'Split line')}
        </HintButton>
      </div>
    </div>
  );
};

export default LyricEditorWordActions;
