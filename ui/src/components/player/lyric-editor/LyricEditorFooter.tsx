// LyricEditorFooter.tsx — zoom and view actions, the offsets, and the save
// state's buttons.
//
// Three states compete for the right-hand side and only one can be shown: the
// file moved on disk (Reload), closing with unsaved edits (Discard or Keep
// editing), or the ordinary case (Reload saved, and Save when Auto save is off).
// They are three guards in one small render function rather than a chain of
// ternaries, so the precedence is stated once and adding a fourth is one more
// block, not one more indent.

import React from 'react';
import {
  Loader2, LocateFixed, Maximize2, RefreshCw, Save, ZoomIn, ZoomOut,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { HintButton } from '../../shared/HintButton';
import { ParamLabel } from '../../shared/ParamLabel';
import { Toggle } from '../../shared/Toggle';
import { BTN, ICON_BTN, ROW, SAVE_BTN } from './editorStyles';
import { MAX_PX, MIN_PX, ZOOM_FACTOR } from './timingModel';

/** The zoom slider's granularity, in pixels per second. */
const ZOOM_SLIDER_STEP = 10;

export interface LyricEditorFooterProps {
  pxPerSec: number;
  onZoom: (pxPerSec: number) => void;
  onZoomBy: (factor: number) => void;
  onGoToPlayhead: () => void;
  onFrameSelection: () => void;
  hasSelection: boolean;
  /** Committed step in milliseconds, shared with the word row. */
  step: number;
  onShiftAll: (ms: number) => void;
  onShiftFromSelection: (ms: number) => void;
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
}

export const LyricEditorFooter: React.FC<LyricEditorFooterProps> = ({
  pxPerSec, onZoom, onZoomBy, onGoToPlayhead, onFrameSelection, hasSelection,
  step, onShiftAll, onShiftFromSelection, autoSave, onAutoSave,
  stale, confirmDiscard, dirty, saving, onReload, onDiscard, onKeepEditing, onRevert, onSave,
}) => {
  const { t } = useTranslation();

  /** The right-hand side: stale wins, then the close question, then the usual. */
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
    <div className={ROW}>
      <div className="flex items-center gap-1">
        <HintButton
          label={t('lyrics.editor.zoomOut', 'Zoom out')}
          info={t('lyrics.editor.zoomOutInfo', 'Shows more of the track per pixel, so a whole line fits at once. Ctrl+- does the same from anywhere.')}
          className={ICON_BTN}
          onClick={() => onZoomBy(1 / ZOOM_FACTOR)}
          ariaLabel={t('lyrics.editor.zoomOut', 'Zoom out')}
        >
          <ZoomOut size={12} />
        </HintButton>
        <ParamLabel
          label={t('lyrics.editor.zoom', 'Timeline zoom')}
          info={t('lyrics.editor.zoomHint', 'Pixels per second. At 30 a phrase is a few pixels; at 300 a syllable is wide enough to read and drag. Ctrl+- and Ctrl+= change it, and the chosen zoom is remembered.')}
        >
          <input
            type="range" min={MIN_PX} max={MAX_PX} step={ZOOM_SLIDER_STEP} value={pxPerSec}
            onChange={e => onZoom(Number(e.target.value))}
            className="w-24 accent-pink-500"
            aria-label={t('lyrics.editor.zoom', 'Timeline zoom')}
          />
        </ParamLabel>
        <HintButton
          label={t('lyrics.editor.zoomIn', 'Zoom in')}
          info={t('lyrics.editor.zoomInInfo', 'Shows less of the track per pixel, which is how a syllable gets wide enough to place. Ctrl+= does the same.')}
          className={ICON_BTN}
          onClick={() => onZoomBy(ZOOM_FACTOR)}
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
          onClick={onGoToPlayhead}
          ariaLabel={t('lyrics.editor.goToPlayhead', 'Scroll to the playhead')}
        >
          <LocateFixed size={12} />
        </HintButton>
        <HintButton
          label={t('lyrics.editor.zoomSelected', 'Frame the selected word')}
          info={t('lyrics.editor.zoomSelectedInfo', 'Zooms so the selected word fills most of the window, for work at syllable level. Needs a selected word.')}
          className={ICON_BTN}
          onClick={onFrameSelection}
          disabled={!hasSelection}
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
        onClick={() => onShiftAll(-step)}
      >
        -{step} ms
      </HintButton>
      <HintButton
        label={t('lyrics.editor.wholeTrack', 'Whole track')}
        info={t('lyrics.editor.wholeTrackInfo', 'Moves every word and line by the step, for an offset that is wrong from the first line to the last.')}
        className={BTN}
        onClick={() => onShiftAll(step)}
      >
        +{step} ms
      </HintButton>

      <span className="ml-2 opacity-60">{t('lyrics.editor.fromSelected', 'From selected')}</span>
      <HintButton
        label={t('lyrics.editor.fromSelected', 'From selected')}
        info={t('lyrics.editor.shiftLeftHint', 'Moves the selected word and everything after it earlier by the step, leaving the part already right where it is. The fix for drift that builds up through the track.')}
        className={BTN}
        onClick={() => onShiftFromSelection(-step)}
        disabled={!hasSelection}
      >
        ← {step} ms
      </HintButton>
      <HintButton
        label={t('lyrics.editor.fromSelected', 'From selected')}
        info={t('lyrics.editor.shiftRightHint', 'Moves the selected word and everything after it later by the step, leaving the part before it alone.')}
        className={BTN}
        onClick={() => onShiftFromSelection(step)}
        disabled={!hasSelection}
      >
        {step} ms →
      </HintButton>

      <div className="ml-auto flex items-center gap-2">
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
      </div>
    </div>
  );
};

export default LyricEditorFooter;
