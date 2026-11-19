// LyricEditorFileControls.tsx — the File cluster at the right end of the bottom
// row: Auto save, the three save states, Save and Export.
//
// The row is one rail: Play on the left, File on the right, and Export is the
// last control on screen. Keeping the document actions together means the two
// questions a person asks at the end of a session — "is this written?" and
// "give me the .lrc" — are answered in one place instead of in two rows.
//
// Three states compete for the cluster and only one can be shown: the file moved
// on disk (Reload), closing with unsaved edits (Discard or Keep editing), or the
// ordinary case (Reload saved, and Save when Auto save is off). They are three
// guards in one small render function rather than a chain of ternaries, so the
// precedence is stated once and adding a fourth is one more block, not one more
// indent.

import React from 'react';
import { Download, Loader2, RefreshCw, Save } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { HintButton } from '@/components/shared/controls/HintButton';
import { Toggle } from '@/components/shared/controls/Toggle';
import { BTN, SAVE_BTN } from './editorStyles';

export interface LyricEditorFileControlsProps {
  autoSave: boolean;
  onAutoSave: (autoSave: boolean) => void;
  /** The file changed under the editor; it must be reloaded, not written. */
  stale: boolean;
  /** The close question is open. */
  confirmDiscard: boolean;
  dirty: boolean;
  saving: boolean;
  onReload: () => void;
  onDiscard: () => void;
  onKeepEditing: () => void;
  onRevert: () => void;
  onSave: () => void;
  onExport: () => void;
  /** True while a write is in flight, so the export cannot race it. */
  exportDisabled: boolean;
}

export const LyricEditorFileControls: React.FC<LyricEditorFileControlsProps> = ({
  autoSave, onAutoSave, stale, confirmDiscard, dirty, saving,
  onReload, onDiscard, onKeepEditing, onRevert, onSave, onExport, exportDisabled,
}) => {
  const { t } = useTranslation();

  /** The save state: stale wins, then the close question, then the usual. */
  const saveStateControls = () => {
    if (stale) {
      return (
        <>
          <span className="text-red-400">{t('lyrics.editor.staleNote', 'This file changed on disk')}</span>
          <HintButton
            label={t('lyrics.editor.reload', 'Reload')}
            info={t('lyrics.editor.reloadHint', 'Reads the file as it is now and throws away the edits on screen, which is the only safe move when a crop, a retranscription or another window moved the timings on.')}
            className={BTN}
            onClick={onReload}
          >
            <RefreshCw size={12} /> {t('lyrics.editor.reload', 'Reload')}
          </HintButton>
        </>
      );
    }

    if (confirmDiscard) {
      return (
        <>
          <span className="text-amber-500">{t('lyrics.editor.discard', 'Discard changes?')}</span>
          <HintButton
            label={t('lyrics.editor.discardYes', 'Discard')}
            info={t('lyrics.editor.discardInfo', 'Closes the editor and throws away the changes that have not been saved. There is no undo for this once the modal is gone.')}
            className={BTN}
            onClick={onDiscard}
          >
            {t('lyrics.editor.discardYes', 'Discard')}
          </HintButton>
          <HintButton
            label={t('lyrics.editor.keep', 'Keep editing')}
            info={t('lyrics.editor.keepInfo', 'Closes the question and leaves the editor open with the edits still on screen.')}
            className={BTN}
            onClick={onKeepEditing}
          >
            {t('lyrics.editor.keep', 'Keep editing')}
          </HintButton>
        </>
      );
    }

    return (
      <>
        <HintButton
          label={t('lyrics.editor.revert', 'Reload saved')}
          info={t('lyrics.editor.revertHint', 'Throws away the edits on screen and goes back to the last saved file. Unlike Undo this is not one step, so it is disabled when there is nothing pending.')}
          className={BTN}
          onClick={onRevert}
          disabled={!dirty}
        >
          <RefreshCw size={12} /> {t('lyrics.editor.revert', 'Reload saved')}
        </HintButton>
        {!autoSave && (
          <HintButton
            label={t('lyrics.editor.save', 'Save')}
            info={t('lyrics.editor.saveHint', 'Writes the changes now. Shown only while Auto save is off; with it on the write happens about a second after you stop.')}
            className={SAVE_BTN}
            onClick={onSave}
            disabled={!dirty || saving}
          >
            {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
            {t('lyrics.editor.save', 'Save')}
          </HintButton>
        )}
      </>
    );
  };

  return (
    <>
      {/* The second scope in the row, in the same small caps as the rail. */}
      <span className="text-[9px] font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500">
        {t('lyrics.editor.scopeFile', 'File')}
      </span>
      <Toggle
        size="sm"
        accent="pink"
        checked={autoSave}
        onChange={onAutoSave}
        label={t('lyrics.editor.autoSave', 'Auto save')}
        info={t(
          'lyrics.editor.autoSaveHint',
          'On: a change is written about a second after you stop, and the Save button is hidden because there is nothing to press. Off: nothing is written until Save, which appears when this is switched off, and closing with unsaved edits asks first. The choice is remembered.',
        )}
      />
      {saveStateControls()}
      <HintButton
        label={t('lyrics.editor.exportLyrics', 'Export lyrics')}
        info={t(
          'lyrics.editor.exportLyricsHint',
          'Downloads this track\u2019s .lrc with the timings on screen, named to match the audio download so a player pairs the two. Pending changes are saved first, and Enhanced LRC in Settings adds per-word stamps. A track with no word timings exports its line-level .lrc instead.',
        )}
        className={BTN}
        onClick={onExport}
        disabled={exportDisabled}
      >
        <Download size={12} /> {t('lyrics.editor.exportLyrics', 'Export lyrics')}
      </HintButton>
    </>
  );
};

export default LyricEditorFileControls;
