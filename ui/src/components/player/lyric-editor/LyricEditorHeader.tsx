// LyricEditorHeader.tsx — the top row: what is open, its save state, undo/redo,
// and the way out.
//
// The badge belongs to this row, so the state that produces it does too: the
// caller passes the four facts about the file and the header decides which one
// wins and how to say it. That keeps the precedence rule beside the thing it
// renders instead of in the shell above it.

import React from 'react';
import { Pencil, Redo2, Undo2, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { HintButton } from '../../shared/HintButton';
import { ICON_BTN } from './editorStyles';
import type { EditHistoryHandlers } from './useEditHistory';

export interface LyricEditorHeaderProps {
  title?: string;
  /** What the file is doing, from `useLyricSaver().data`. */
  save: {
    stale: boolean;
    saving: boolean;
    saveError: boolean;
    dirty: boolean;
  };
  /** From `useEditHistory().data`. */
  canUndo: boolean;
  canRedo: boolean;
  /** The history's handler slice: this row can undo and redo, nothing else. */
  handlers: Pick<EditHistoryHandlers, 'undo' | 'redo'>;
  onClose: () => void;
}

export const LyricEditorHeader: React.FC<LyricEditorHeaderProps> = ({
  title, save, canUndo, canRedo, handlers, onClose,
}) => {
  const { t } = useTranslation();

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
      <span className={`rounded px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider ${status.cls}`}>
        {status.label}
      </span>
      <HintButton
        label={t('common.close', 'Close')}
        info={t('lyrics.editor.closeInfo', 'Closes the editor. With unsaved edits it asks first, so closing never silently loses a placement. Escape does the same.')}
        className="rounded p-1 text-zinc-500 hover:bg-black/5 dark:hover:bg-white/10"
        onClick={onClose}
        rootClassName="ml-auto"
        ariaLabel={t('common.close', 'Close')}
      >
        <X size={14} />
      </HintButton>
    </div>
  );
};

export default LyricEditorHeader;
