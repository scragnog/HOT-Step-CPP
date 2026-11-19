// LyricEditorWordActions.tsx — the editing scope rows: the line, the word, and
// the timing commands that act on whole runs.
//
// Each row is one scope, named in the same gutter as every other control row in
// the editor, and the buttons inside it are divided into clusters in the order
// the work goes:
//
//   Line    the section label, the whole line's text, merge and split
//   Word    what is selected, the step, the nudge/length/snap commands, the
//           word-level playback, add, delete and the two loops
//   Timing  the two AI passes over whole spans, their licence badge, and the
//           offsets that move every word or everything after the selection
//
// Three rows rather than one long run: eighteen buttons in a single row read as
// a wall, and the one you want is never where you look first. The gutter means
// the first button of each scope starts at the same x.
//
// The step box's draft lives here rather than in the editor's shell. Nothing
// else can type into it, so the half-typed value is this row's private state;
// only the committed number travels up, through `onStep`. The clamping rule is
// `timingModel`'s `clampStep`, so the range is one definition for the box and
// the keyboard that reads the same step.

import React, { useEffect, useRef, useState } from 'react';
import {
  ArrowLeftToLine, Check, ChevronsDown, ChevronsUp, Play, Plus, Scissors, Sparkles, Trash2, X,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { HintButton } from '@/components/shared/controls/HintButton';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { BTN, DIVIDER, ROW, ROW_BODY, ROW_LABEL } from './editorStyles';
import { LyricEditorRepairPopover } from './LyricEditorRepairPopover';
import type { RealignSettings } from '@/storage/lyricEditorSettings';
import {
  STEP_MAX, STEP_MIN, clampStep, fmt, planLineText,
  type EditLine, type EditWord, type LineEditPlan, type SectionRun,
} from './timingModel';
import { formatDecimal } from '@/utils/format/format';
import { isPresent } from '@/utils/guards';

export interface LyricEditorWordActionsProps {
  selectedWord: EditWord | null;
  /** Words in the current selection; more than one is a range. */
  selectionCount: number;
  /** The lines the selection touches, or null with nothing selected. */
  selectionLines: { fromLine: number; toLine: number } | null;
  /** From the first selected onset to the last selected end. */
  selectionRange?: { start: number; end: number } | null;
  onRealign: () => void;
  realignDisabled?: boolean;
  /** Re-runs only the statistics pass: seconds, no Whisper and no model. */
  onRepairTimings: () => void;
  repairDisabled?: boolean;
  /** The pass's switches, remembered per track. */
  repairSettings: RealignSettings;
  onRepairSettingsChange: (next: RealignSettings) => void;
  /** The pipeline stage while a run is in flight, for the label beside it. */
  alignStage?: string | null;
  alignError?: string | null;
  /** The line the line tools act on, or -1 when the track has no lines. */
  targetLine: number;
  /** That line's section label, empty when it starts no section. */
  targetSection: string;
  onSectionChange: (section: string) => void;
  /** The line the tools act on, as it is now. Its text is edited as a whole:
   *  retyping a chorus one word at a time is the tedium this replaces. */
  targetLineData: EditLine | null;
  /** Writes a rewritten line, after its diff has been shown and accepted. */
  onApplyLine: (plan: LineEditPlan) => void;
  /** The track's sections, each with the line it starts on. */
  sections?: SectionRun[];
  /** Starts a section at the playhead, named by the label box. */
  onAddSection: (name: string) => void;
  /** Takes a section's label off the line it starts on. */
  onDeleteSection: (lineIndex: number) => void;
  /** Scrolls the timeline to a section's first line, from its chip. */
  onJumpSection: (lineIndex: number) => void;
  /** A word was just added, so the text box should take the cursor. */
  autoFocusWord?: boolean;
  onWordAutoFocus?: () => void;
  /** Inserts a word after the selection, or after the playhead's word. */
  onAddWord: () => void;
  onDeleteWord: () => void;
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
  /** Replays the target line, for checking a phrase rather than a syllable. */
  onToggleLineLoop: () => void;
  loopLine: boolean;
  /** Moves every word by `ms`, for an offset that is wrong throughout. */
  onShiftAll: (ms: number) => void;
  /** Moves the selection and everything after it by `ms`. */
  onShiftFromSelection: (ms: number) => void;
  canMergePrev: boolean;
  canMergeNext: boolean;
  onMerge: (dir: -1 | 1) => void;
  onSplit: () => void;
}

/**
 * One line's section label, committed on blur or Enter.
 *
 * The draft lives here and the component is keyed by the line and its label, so
 * moving the playhead (or an undo) remounts it with the right value. That is
 * the whole reason it is a component rather than an effect syncing state: a
 * half-typed label must never be written by an autosave timer, and must not
 * survive into another line's box.
 */
const SectionBar: React.FC<{
  line: number;
  section: string;
  onChange: (section: string) => void;
  onAdd: (name: string) => void;
  onDelete: (lineIndex: number) => void;
  onJump: (lineIndex: number) => void;
  /** The track's sections in order: one entry per place a label starts. */
  sections: SectionRun[];
}> = ({ line, section, onChange, onAdd, onDelete, onJump, sections }) => {
  const { t } = useTranslation();
  const [draft, setDraft] = useState(section);
  /** While the box has focus, its own text is what matters, not the card. */
  const [editing, setEditing] = useState(false);

  const commit = (value: string) => {
    const label = value.trim();
    setDraft(label);
    if (label !== section.trim()) onChange(label);
  };

  return (
    // On its own line within this row's controls: section chips grow with the
    // song, and sharing a line with the text box squeezed that box into wrapping
    // mid-word.
    <span className="flex basis-full flex-wrap items-center gap-1">
      {/* The card belongs to the box alone: hovering a chip or Add section
          should not explain the Section field, and while the box has focus the
          card would cover the text being edited. */}
      <ParamLabel
        label={t('lyrics.editor.section', 'Section')}
        info={t('lyrics.editor.sectionInfo', 'The label the player shows above this line, such as Verse 2 or Chorus. A section is a label on the line where it starts, not a row of its own: Add section starts one at the playhead — splitting the line there first when the playhead is inside it — and each chip below deletes a section from wherever you are. A line with no label belongs to the section above it.')}
        openOnFocus={false}
        suppress={editing}
      >
        <input
          type="text"
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onFocus={() => setEditing(true)}
          onBlur={() => { setEditing(false); commit(draft); }}
          onKeyDown={e => {
            if (e.key === 'Enter') e.currentTarget.blur();
            if (e.key === 'Escape') { setDraft(section); e.currentTarget.blur(); }
          }}
          disabled={line < 0}
          placeholder={t('lyrics.editor.sectionNone', 'No section')}
          className="w-24 rounded border border-zinc-300 bg-transparent px-1 py-0.5 dark:border-white/10"
          aria-label={t('lyrics.editor.section', 'Section label for this line')}
        />
      </ParamLabel>
      <HintButton
        label={t('lyrics.editor.addSection', 'Add section')}
        info={t('lyrics.editor.addSectionInfo', 'Starts a section at the playhead with the name in the box. If the playhead is inside the line rather than before its first word, the line is split there first, so the new section gets its own line; no word changes its time either way.')}
        className={BTN}
        onClick={() => onAdd(draft || section)}
      >
        <Plus size={11} /> {t('lyrics.editor.addSection', 'Add section')}
      </HintButton>
      {sections.map(run => (
        <span
          key={`${run.section}-${run.from}`}
          className="flex items-center gap-0.5 rounded-full border border-violet-400/40 bg-violet-500/10 px-1.5 py-0.5 text-[10px] text-violet-300"
        >
          <button
            type="button"
            onClick={() => onJump(run.from)}
            aria-label={t('lyrics.editor.jumpToSectionAria', 'Jump to the {{section}} section on line {{line}}', { section: run.section, line: run.from + 1 })}
            className="rounded-full px-0.5 hover:text-violet-100"
          >
            {run.section}
          </button>
          <HintButton
            label={t('lyrics.editor.deleteSection', 'Delete section')}
            info={t('lyrics.editor.deleteSectionInfo', 'Takes the {{section}} label off line {{line}}, where it starts. No lyric and no word time changes: the lines join the section above, and the words stay exactly where they are.', { section: run.section, line: run.from + 1 })}
            className="flex h-3.5 w-3.5 items-center justify-center rounded-full text-violet-300 hover:bg-violet-500/30 hover:text-violet-100"
            onClick={() => onDelete(run.from)}
            ariaLabel={t('lyrics.editor.deleteSectionAria', 'Delete the {{section}} section', { section: run.section })}
          >
            <X size={9} />
          </HintButton>
        </span>
      ))}
    </span>
  );
};

/**
 * The whole line's text, with the diff shown before anything is written.
 *
 * This replaces retyping a line one word at a time. Committing the box (Enter or
 * leaving it) never writes: it builds a plan and shows it — the words that keep
 * their slot, the ones renamed, the ones added and the ones dropped — and Apply
 * is the write. Cancel puts the line back exactly as it was.
 *
 * The word count may change. Words pair by position, new ones split the last
 * word's own interval, and a line that loses words keeps its old end, so a
 * spelling fix never moves the lines after it.
 */
const LineField: React.FC<{
  line: EditLine | null;
  lineIndex: number;
  onApply: (plan: LineEditPlan) => void;
  autoFocus?: boolean;
  onAutoFocus?: () => void;
}> = ({ line, lineIndex, onApply, autoFocus, onAutoFocus }) => {
  const { t } = useTranslation();
  const text = line?.text ?? '';
  const [draft, setDraft] = useState(text);
  const [plan, setPlan] = useState<LineEditPlan | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!autoFocus) return;
    inputRef.current?.focus();
    inputRef.current?.setSelectionRange(text.length, text.length);
    onAutoFocus?.();
  }, [autoFocus, onAutoFocus, text.length]);

  /** Build the plan, or leave the box holding the line's own text. */
  const review = (value: string) => {
    const next = value.replace(/\s+/g, ' ').trim();
    setDraft(next);
    if (!line) return;
    setPlan(planLineText(line, lineIndex, next));
  };

  const applyPlan = () => {
    if (!plan) return;
    onApply(plan);
    setDraft(plan.edited.text);
    setPlan(null);
  };

  const cancelPlan = () => {
    setPlan(null);
    setDraft(text);
  };

  if (plan) {
    const counts = [
      plan.renamed ? t('lyrics.editor.lineRenamed', '{{count}} renamed', { count: plan.renamed }) : '',
      plan.added ? t('lyrics.editor.lineAdded', '{{count}} added', { count: plan.added }) : '',
      plan.removed ? t('lyrics.editor.lineRemoved', '{{count}} dropped', { count: plan.removed }) : '',
    ].filter(isPresent).join(' · ');
    return (
      <div className="flex items-center gap-1.5 rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5">
        <span className="font-mono text-[10px] leading-4">
          {plan.changes.map((change, index) => (
            <span
              key={index}
              className={
                change.kind === 'added' ? 'ml-1 text-emerald-500'
                  : change.kind === 'removed' ? 'ml-1 text-red-400 line-through'
                    : change.kind === 'renamed' ? 'ml-1 text-amber-500'
                      : 'ml-1 opacity-60'
              }
            >
              {change.kind === 'renamed'
                ? `${change.from}\u2192${change.to}`
                : change.kind === 'added'
                  ? `+${change.to}@${fmt(change.start ?? 0)}`
                  : change.kind === 'removed' ? `\u2212${change.from}` : change.to}
            </span>
          ))}
        </span>
        <span className="shrink-0 opacity-70">{counts}</span>
        <HintButton
          label={t('lyrics.editor.lineApply', 'Apply')}
          info={t('lyrics.editor.lineApplyInfo', 'Writes this line\u2019s text. A word that kept its slot keeps its times; added words take their time from the last word\u2019s own interval; dropped words hand theirs to the word kept before them. One undo step for the whole line.')}
          className={BTN}
          onClick={applyPlan}
        >
          <Check size={11} /> {t('lyrics.editor.lineApply', 'Apply')}
        </HintButton>
        <HintButton
          label={t('lyrics.editor.lineCancel', 'Cancel')}
          info={t('lyrics.editor.lineCancelInfo', 'Leaves the line exactly as it is. Nothing was written while the diff was on screen.')}
          className={BTN}
          onClick={cancelPlan}
        >
          <X size={11} /> {t('lyrics.editor.lineCancel', 'Cancel')}
        </HintButton>
      </div>
    );
  }

  return (
    <ParamLabel
      label={t('lyrics.editor.line', 'Line')}
      info={t('lyrics.editor.lineInfo', 'The whole line as it should read, in one box. Enter, or leaving the box, shows what changes \u2014 which words are renamed, which are added with a time taken from the last word, and which are dropped \u2014 and nothing is written until Apply. The word count may differ from what the aligner produced.')}
    >
      <input
        ref={inputRef}
        type="text"
        value={draft}
        onChange={e => setDraft(e.target.value)}
        onBlur={() => review(draft)}
        onKeyDown={e => {
          if (e.key === 'Enter') review(draft);
          if (e.key === 'Escape') { setDraft(text); e.currentTarget.blur(); }
        }}
        disabled={!line}
        placeholder={t('lyrics.editor.lineEmpty', 'No line here')}
        className="w-72 rounded border border-zinc-300 bg-transparent px-1 py-0.5 font-semibold text-pink-400 dark:border-white/10"
        aria-label={t('lyrics.editor.line', 'Text for this line')}
      />
    </ParamLabel>
  );
};

export const LyricEditorWordActions: React.FC<LyricEditorWordActionsProps> = ({
  selectedWord, selectionCount, selectionLines, selectionRange,
  onRealign, realignDisabled, onRepairTimings, repairDisabled, repairSettings, onRepairSettingsChange,
  alignStage, alignError,
  targetLine, targetSection, onSectionChange, onAddSection, onDeleteSection, onJumpSection, targetLineData, onApplyLine,
  sections = [], autoFocusWord, onWordAutoFocus, onAddWord, onDeleteWord,
  step, onStep, onNudge, onResize, onSnapToPrevious, onStartAtPlayhead,
  onPlayFromWord, onToggleLoop, loopWord, onToggleLineLoop, loopLine,
  onShiftAll, onShiftFromSelection,
  canMergePrev, canMergeNext, onMerge, onSplit,
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

  const deleteLabel = selectionCount > 1
    ? t('lyrics.editor.deleteWords', 'Delete {{count}} words', { count: selectionCount })
    : t('lyrics.editor.deleteWord', 'Delete word');

  // A selection is what the loop replays, so the label says which one it will
  // be rather than promising a single word it is not going to play.
  const loopLabel = selectionCount > 1
    ? t('lyrics.editor.loopSelection', 'Loop {{count}} words', { count: selectionCount })
    : t('lyrics.editor.loop', 'Loop word');

  // One selected line says "line 6", not "lines 6-6".
  const realignLabel = !selectionLines
    ? t('lyrics.editor.realignAll', 'Re-align track')
    : selectionLines.fromLine === selectionLines.toLine
      ? t('lyrics.editor.realignOneLine', 'Re-align line {{line}}', { line: selectionLines.fromLine + 1 })
      : t('lyrics.editor.realignRange', 'Re-align lines {{from}}-{{to}}', {
          from: selectionLines.fromLine + 1, to: selectionLines.toLine + 1,
        });

  // A range shows what the commands act on: the span, not the focus word.
  const fromSec = selectionCount > 1 && selectionRange ? selectionRange.start : selectedWord?.start ?? 0;
  const toSec = selectionCount > 1 && selectionRange ? selectionRange.end : selectedWord?.end ?? 0;

  return (
    <>
      {/* Line scope: the coarser label and the whole line, then its structure. */}
      <div className={ROW}>
        <span className={ROW_LABEL}>{t('lyrics.editor.scopeLine', 'Line')}</span>
        <div className={ROW_BODY}>
          <SectionBar
            key={`${targetLine}:${targetSection}:${sections.length}`}
            line={targetLine}
            section={targetSection}
            onChange={onSectionChange}
            onAdd={onAddSection}
            onDelete={onDeleteSection}
            onJump={onJumpSection}
            sections={sections}
          />
          <LineField
            key={`${targetLine}:${targetLineData?.text ?? ''}`}
            line={targetLineData}
            lineIndex={targetLine}
            onApply={onApplyLine}
            autoFocus={autoFocusWord}
            onAutoFocus={onWordAutoFocus}
          />
          <span className={DIVIDER} />
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

      {/* Word scope: the selection, then one cluster per kind of edit. */}
      <div className={ROW}>
        <span className={ROW_LABEL}>{t('lyrics.editor.scopeWord', 'Word')}</span>
        <div className={ROW_BODY}>
          {selectedWord ? (
            <>
              {selectionCount > 1 && (
                <ParamLabel
                  label={t('lyrics.editor.range', 'Selection')}
                  info={t('lyrics.editor.rangeHint', 'Shift-click another word to extend the selection. Re-align works on whole lines.')}
                >
                  <span className="rounded bg-pink-500/15 px-1.5 py-0.5 font-semibold text-pink-300">
                    {selectionCount} {t('lyrics.editor.words', 'words')}
                  </span>
                </ParamLabel>
              )}
              <span className="font-mono tabular-nums">{fmt(fromSec)} → {fmt(toSec)}</span>
              <span className="tabular-nums opacity-50">({formatDecimal((toSec - fromSec), 2)}s)</span>
              <span className={DIVIDER} />
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
                label={deleteLabel}
                info={t('lyrics.editor.deleteWordHint', 'Removes every selected word. A line that loses all of its words goes with them; the last word of the track is kept, so the lyrics cannot end up empty.')}
                className={BTN}
                onClick={onDeleteWord}
              >
                <Trash2 size={12} /> {deleteLabel}
              </HintButton>
            </>
          ) : (
            // An empty row reads as broken; say what to do instead.
            <span className="opacity-60">
              {t('lyrics.editor.noWord', 'Select a word on the timeline to edit its timing')}
            </span>
          )}
          <span className={DIVIDER} />
          <HintButton
            label={t('lyrics.editor.addWord', 'Add word')}
            info={t('lyrics.editor.addWordHint', 'Inserts a word after the selected one, or after the word the playhead is on, and puts the cursor in its box so you can type over the placeholder. It takes the gap to the next word; if that gap is too short, the words after it move along by the shortfall.')}
            className={BTN}
            onClick={onAddWord}
            disabled={targetLine < 0}
          >
            <Plus size={12} /> {t('lyrics.editor.addWord', 'Add word')}
          </HintButton>
          {/* The loop toggles live outside the selection branch: a loop that is on
              has to stay clickable when the selection moves on or is cleared, or
              the only way to silence it is closing the editor. */}
          <span className={DIVIDER} />
          <HintButton
            label={loopLabel}
            info={t('lyrics.editor.loopHint', 'Replays the selected word, or the whole selection when several are selected, until it is switched off. Arming it seeks just before the start, and the loop restarts whenever playback passes the end. It stays here while it is on, so clearing the selection never traps you with a loop you cannot switch off.')}
            className={`${BTN} ${loopWord ? 'border-pink-400 text-pink-400' : ''}`}
            onClick={onToggleLoop}
            disabled={!selectedWord && !loopWord}
            ariaPressed={loopWord}
          >
            {loopLabel}
          </HintButton>
          <HintButton
            label={t('lyrics.editor.loopLine', 'Loop line')}
            info={t('lyrics.editor.loopLineHint', 'Replays the line the tools above act on, from a fifth of a second before its first word. Clicking a row and arming this is a two-click check of a phrase; Loop word does the same for one syllable, and arming either releases the other. It stays here while it is on, including when nothing is selected.')}
            className={`${BTN} ${loopLine ? 'border-pink-400 text-pink-400' : ''}`}
            onClick={onToggleLineLoop}
            disabled={targetLine < 0 && !loopLine}
            ariaPressed={loopLine}
          >
            {t('lyrics.editor.loopLine', 'Loop line')}
          </HintButton>
        </div>
      </div>

      {/* Timing scope: the AI passes first, then the offsets that move runs. */}
      <div className={ROW}>
        <span className={ROW_LABEL}>{t('lyrics.editor.scopeTiming', 'Timing')}</span>
        <div className={ROW_BODY}>
          <LyricEditorRepairPopover
            settings={repairSettings}
            onChange={onRepairSettingsChange}
            onApply={onRepairTimings}
            disabled={repairDisabled}
          />
          <HintButton
            label={realignLabel}
            info={t('lyrics.editor.realignHint', 'Runs Whisper over this audio and then the CTC forced aligner over the same span, replacing only the lines you selected. Nothing outside the selection is re-timed. With nothing selected it runs the whole track.')}
            className={BTN}
            onClick={onRealign}
            disabled={realignDisabled}
          >
            <Sparkles size={12} />
            {realignLabel}
          </HintButton>
          {/* The licence notice belongs where the button is: the aligner's weights
              are non-commercial, and the timings a run produces are not. The same
              hover card as every other explained control: a native `title` here
              looked nothing like the rest of the app (rule 3 of
              docs/dev/ui-design.md). */}
          <ParamLabel
            label={t('lyrics.editor.alignerLicence', 'CC BY-NC 4.0')}
            info={t(
              'lyrics.editor.alignerLicenceInfo',
              'Whisper is MIT-licensed. The CTC word aligner runs Meta’s MMS_FA weights, licensed CC BY-NC 4.0: non-commercial use. HOT-Step itself is open source and non-commercial, and the timings it produces are just times, so they carry no extra restriction.',
            )}
          >
            <span className="rounded bg-zinc-500/10 px-1 py-0.5 text-[9px] uppercase tracking-wider opacity-60">
              {t('lyrics.editor.alignerLicence', 'CC BY-NC 4.0')}
            </span>
          </ParamLabel>
          <span className={DIVIDER} />
          <span className="opacity-60">{t('lyrics.editor.wholeTrack', 'Whole track')}</span>
          <HintButton
            label={t('lyrics.editor.wholeTrackEarlier', 'Whole track earlier')}
            info={t('lyrics.editor.wholeTrackInfo', 'Moves every word and line by the step, for an offset that is wrong from the first line to the last. Stops at zero rather than going negative.')}
            className={BTN}
            onClick={() => onShiftAll(-step)}
          >
            -{step} ms
          </HintButton>
          <HintButton
            label={t('lyrics.editor.wholeTrackLater', 'Whole track later')}
            info={t('lyrics.editor.wholeTrackInfo', 'Moves every word and line by the step, for an offset that is wrong from the first line to the last.')}
            className={BTN}
            onClick={() => onShiftAll(step)}
          >
            +{step} ms
          </HintButton>
          <span className={DIVIDER} />
          <span className="opacity-60">{t('lyrics.editor.fromSelected', 'From selected')}</span>
          <HintButton
            label={t('lyrics.editor.rippleEarlier', 'Ripple earlier')}
            info={t('lyrics.editor.shiftLeftHint', 'Moves the selected word and everything after it earlier by the step, leaving the part already right where it is. The fix for drift that builds up through the track.')}
            className={BTN}
            onClick={() => onShiftFromSelection(-step)}
            disabled={!selectedWord}
          >
            ← {step} ms
          </HintButton>
          <HintButton
            label={t('lyrics.editor.rippleLater', 'Ripple later')}
            info={t('lyrics.editor.shiftRightHint', 'Moves the selected word and everything after it later by the step, leaving the part before it alone.')}
            className={BTN}
            onClick={() => onShiftFromSelection(step)}
            disabled={!selectedWord}
          >
            {step} ms →
          </HintButton>
          {alignStage && <span className="ml-1 text-[10px] text-amber-500">{alignStage}…</span>}
          {alignError && <span className="ml-1 text-[10px] text-red-400">{alignError}</span>}
        </div>
      </div>
    </>
  );
};

export default LyricEditorWordActions;
