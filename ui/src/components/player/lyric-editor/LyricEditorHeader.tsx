// LyricEditorHeader.tsx — the top row: what is open, its save state, undo/redo,
// and the way out.
//
// The badge belongs to this row, so the state that produces it does too: the
// caller passes the four facts about the file and the header decides which one
// wins and how to say it. That keeps the precedence rule beside the thing it
// renders instead of in the shell above it.

import React from 'react';
import { Info, Loader2, Pencil, Redo2, Undo2, Wand2, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { HintButton } from '@/components/shared/controls/HintButton';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { BTN, ICON_BTN } from './editorStyles';
import type { EditHistoryHandlers } from './useEditHistory';
import { isPresent } from '@/utils/guards';

export interface LyricEditorHeaderProps {
  title?: string;
  /** What produced the current timings, from the sidecar's `diagnostics`. */
  provenance?: {
    method?: string;
    vocalsIsolated?: boolean;
    dtw?: boolean;
    wordsMoved?: number;
    fallback?: string | null;
    /** Replaces the generated title when the timings did not come from an
     *  aligner — seeded from a `.lrc`'s line times, say. */
    detail?: string;
    /** The CTC-against-Whisper audit, on a sidecar written by a full-track run.
     *  The same block `npm run lyrics:drift` pools across tracks. */
    anchorDrift?: {
      anchors: number;
      medianMs: number;
      p90Ms: number;
      outliers: number;
      worst?: Array<{ line: number; driftMs: number; section?: string }>;
    } | null;
  } | null;
  /** What the file is doing, from `useLyricSaver().data`. */
  save: {
    stale: boolean;
    saving: boolean;
    saveError: boolean;
    dirty: boolean;
  };
  /**
   * Offered when the timings this editor opened on are too rough to check by
   * ear — words stacked on one instant, or a sheet that was only ever split
   * from line times. Absent means there is nothing to offer: a track the
   * aligner already placed is judged by listening, not by running it again.
   */
  align?: {
    onAlign: () => void;
    aligning: boolean;
    disabled: boolean;
  };
  /**
   * A run is working on this track right now. It belongs in the title row, not
   * only in a toast: a toast can be dismissed, and the run outlives both the
   * toast and the editor, so the row that names the open file is where its
   * state has to stay readable.
   */
  busy?: { label: string } | null;
  /** From `useEditHistory().data`. */
  canUndo: boolean;
  canRedo: boolean;
  /** The history's handler slice: this row can undo and redo, nothing else. */
  handlers: Pick<EditHistoryHandlers, 'undo' | 'redo'>;
  onClose: () => void;
}

export const LyricEditorHeader: React.FC<LyricEditorHeaderProps> = ({
  title, provenance, save, align, busy, canUndo, canRedo, handlers, onClose,
}) => {
  const { t } = useTranslation();

  const drift = provenance?.anchorDrift ?? null;

  /**
   * The audit as one line of title text: the numbers first, then the places
   * worth listening to. Line numbers are the editor's, so they are 1-based here
   * even though the sidecar's are not.
   */
  const driftDetail = drift
    ? [
        `drift ${drift.medianMs} ms median, ${drift.p90Ms} ms P90 against the words heard`,
        `${drift.outliers} of ${drift.anchors} words over tolerance`,
        ...(drift.worst ?? []).slice(0, 3).map(worst =>
          `line ${worst.line + 1}${worst.section ? ` (${worst.section})` : ''} at ${worst.driftMs} ms`),
      ].join(', ')
    : '';

  /** One badge for the stage that produced these timings, with the detail in
   *  its title: the badge is a fact, not a paragraph. */
  const provenanceTitle = provenance?.detail
    ? provenance.detail
    : provenance?.method
      ? [
        `Aligned by ${provenance.method}`,
        provenance.dtw ? 'with token DTW' : 'from decoder timestamps',
        provenance.vocalsIsolated ? 'on the vocal stem' : 'on the mix',
        provenance.wordsMoved ? `${provenance.wordsMoved} words repaired` : '',
        driftDetail,
        provenance.fallback ? `Fallback: ${provenance.fallback}` : '',
        ].filter(isPresent).join(', ')
      : '';

  /**
   * The badge's wording, in precedence order: the first reason that is true is
   * the one that matters. Guards rather than a chain of ternaries, so another
   * state is one more line and not one more indent level.
   */
  const badge = (): { label: string; cls: string } => {
    if (save.stale) return { label: t('lyrics.editor.stale', 'Changed on disk'), cls: 'bg-red-500/15 text-red-400' };
    if (save.saving) return { label: t('lyrics.editor.saving', 'Saving…'), cls: 'bg-amber-500/15 text-amber-500' };
    if (save.saveError) return { label: t('lyrics.editor.notSaved', 'Not saved'), cls: 'bg-red-500/15 text-red-400' };
    if (save.dirty) return { label: t('lyrics.editor.unsaved', 'Unsaved'), cls: 'bg-amber-500/15 text-amber-500' };
    return { label: t('lyrics.editor.savedBadge', 'Saved'), cls: 'bg-emerald-500/15 text-emerald-400' };
  };
  const status = badge();

  return (
    <div className="flex items-center gap-2 border-b border-zinc-200 px-3 py-2 dark:border-white/10">
      <Pencil size={13} className="text-pink-500" />
      <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-600 dark:text-zinc-300">
        {t('lyrics.editor.title', 'Lyric editor')}
      </span>
      {busy && (
        <ParamLabel
          label={t('lyrics.editor.busyLabel', 'Run in progress')}
          info={t(
            'lyrics.editor.busyInfo',
            'A run is working on this track. It keeps going if you close this editor or switch songs, and reopening the track re-attaches to the same run with its diff still waiting.',
          )}
        >
          <span
            className="flex items-center gap-1 rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-semibold lowercase tracking-wide text-amber-600 dark:text-amber-400"
            aria-live="polite"
            aria-busy="true"
          >
            <Loader2 size={11} className="animate-spin" />
            {busy.label}
          </span>
        </ParamLabel>
      )}
      <div className="flex items-center gap-1">
        <HintButton
          label={t('lyrics.editor.undo', 'Undo')}
          info={t('lyrics.editor.undoInfo', 'Steps back one change, with a whole drag counting as one. Nothing is saved by undoing: the file gets the undone state at the next save. Ctrl+Z here.')}
          className={ICON_BTN}
          onClick={handlers.undo}
          disabled={!canUndo}
          ariaLabel={t('lyrics.editor.undo', 'Undo')}
        >
          <Undo2 size={13} />
        </HintButton>
        <HintButton
          label={t('lyrics.editor.redo', 'Redo')}
          info={t('lyrics.editor.redoInfo', 'Steps forward again after an undo. The stack is cleared by the next new change. Ctrl+Shift+Z here.')}
          className={ICON_BTN}
          onClick={handlers.redo}
          disabled={!canRedo}
          ariaLabel={t('lyrics.editor.redo', 'Redo')}
        >
          <Redo2 size={13} />
        </HintButton>
      </div>
      {title && <span className="truncate text-xs text-zinc-500">{title}</span>}
      {align && (
        <HintButton
          label={t('lyrics.editor.alignOffer', 'Align my lyrics')}
          info={t(
            'lyrics.editor.alignOfferInfo',
            'These timings are too rough to check by ear: words share an instant, or the words were only split out of line times and never placed against the audio. This runs the aligner over the whole track — it hears where each word is sung and keeps your lyrics as the text — then shows the diff. Nothing is written until you apply it, and you can cancel while it runs.',
          )}
          className={BTN}
          onClick={align.onAlign}
          disabled={align.disabled || align.aligning}
          ariaLabel={t('lyrics.editor.alignOffer', 'Align my lyrics')}
        >
          <Wand2 size={12} />
          <span>
            {align.aligning
              ? t('lyrics.editor.alignOfferRunning', 'Aligning\u2026')
              : t('lyrics.editor.alignOffer', 'Align my lyrics')}
          </span>
        </HintButton>
      )}
      {provenance?.method && (
        <ParamLabel
          label={t('lyrics.editor.provenance', 'How these timings were made')}
          info={provenanceTitle}
        >
          <span className="rounded bg-sky-500/10 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-sky-500">
            {provenance.method}{provenance.vocalsIsolated ? ' · vocals' : ''}
            {drift ? ` · drift ${drift.medianMs} ms` : ''}
          </span>
        </ParamLabel>
      )}
      <span className={`rounded px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider ${status.cls}`}>
        {status.label}
      </span>
      {/* Editor-wide help sits with the document controls at the right edge,
          not in the middle of a scope row. */}
      <div className="ml-auto flex items-center gap-1">
        <HintButton
          label={t('lyrics.editor.shortcuts', 'Keyboard shortcuts')}
          info={t(
            'lyrics.editor.shortcutsInfo',
            'Space — play or pause the track.\n'
            + 'Escape — close the editor; it asks first when there are unsaved edits.\n'
            + 'Ctrl+Z — undo. Ctrl+Shift+Z or Ctrl+Y — redo.\n'
            + 'Ctrl+← / Ctrl+→ — previous or next word, without moving its timing.\n'
            + '← / → — nudge the selected word earlier or later by the step.\n'
            + 'Shift+← / Shift+→ — shorten or lengthen the selected word.\n'
            + 'PageUp / PageDown — page the timeline back or forward.\n'
            + 'Ctrl+= or Ctrl++ — zoom in. Ctrl+- or Ctrl+_ — zoom out.\n'
            + 'F2 or double-click a word — retype that word.\n'
            + 'Enter — keep the box you are typing in. Escape — leave it unchanged.',
          )}
          className={ICON_BTN}
          ariaLabel={t('lyrics.editor.shortcuts', 'Keyboard shortcuts')}
        >
          <Info size={13} />
        </HintButton>
        <HintButton
          label={t('common.close', 'Close')}
          info={t('lyrics.editor.closeInfo', 'Closes the editor. With unsaved edits it asks first, so closing never silently loses a placement. Escape does the same.')}
          className="rounded p-1 text-zinc-500 hover:bg-black/5 dark:hover:bg-white/10"
          onClick={onClose}
          ariaLabel={t('common.close', 'Close')}
        >
          <X size={14} />
        </HintButton>
      </div>
    </div>
  );
};

export default LyricEditorHeader;
