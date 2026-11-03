// LyricEditorTransport.tsx — play, seek, speed, volume, Follow, Auto shift and
// the export, along the bottom like a DAW's transport.
//
// The audio is the app's single player element, so this row talks to the engine
// itself instead of being handed four pass-through callbacks and three copies of
// the same state: it is the transport, and what it does here is what the player
// bar does. Its own numbers — how far a seek button jumps, the speeds offered —
// are constants in this file, where the buttons that use them are.

import React, { useSyncExternalStore } from 'react';
import {
  ChevronLeft, ChevronRight, Download, Pause, Play, Volume2,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import {
  getSnapshot as getEngineSnapshot, seek, setRate, setVolume,
  subscribe as subscribeEngine, toggle,
} from '../../../audio/engine';
import { HintButton } from '../../shared/HintButton';
import { ParamLabel } from '../../shared/ParamLabel';
import { StyledSelect } from '../../shared/StyledSelect';
import { Toggle } from '../../shared/Toggle';
import { BTN, ICON_BTN, ROW } from './editorStyles';
import { fmt } from './timingModel';

/** One second for the coarse buttons, a tenth for the fine pair. */
const SEEK_SEC = 1;
const FINE_SEEK_SEC = 0.1;
/** Half speed widens a syllable enough to hear where it starts. */
const PLAYBACK_RATES = [0.5, 0.75, 1];

export interface LyricEditorTransportProps {
  currentTime: number;
  total: number;
  follow: boolean;
  onFollow: (follow: boolean) => void;
  autoShift: boolean;
  onAutoShift: (autoShift: boolean) => void;
  onExport: () => void;
  /** True while a write is in flight, so the export cannot race it. */
  exportDisabled: boolean;
}

export const LyricEditorTransport: React.FC<LyricEditorTransportProps> = ({
  currentTime, total, follow, onFollow, autoShift, onAutoShift, onExport, exportDisabled,
}) => {
  const { t } = useTranslation();
  const engine = useSyncExternalStore(subscribeEngine, getEngineSnapshot);

  return (
    <div className={ROW}>
      <HintButton
        label={t('player.playPause', 'Play or pause')}
        info={t('lyrics.editor.playInfo', 'Plays the same audio as the player bar, from the playhead. Space does the same anywhere in the editor, including while a word is selected, so it never scrolls the pane instead.')}
        className={ICON_BTN}
        onClick={() => toggle()}
        ariaLabel={t('player.playPause', 'Play or pause')}
      >
        {engine.playing ? <Pause size={13} /> : <Play size={13} />}
      </HintButton>
      <HintButton
        label={t('lyrics.editor.back1', 'Back 1 second')}
        info={t('lyrics.editor.back1Info', 'Moves the playhead one second earlier. Each click is one second; hold nothing, the loop is what repeats.')}
        className={BTN}
        onClick={() => seek(Math.max(0, currentTime - SEEK_SEC))}
      >
        <ChevronLeft size={12} />{SEEK_SEC}s
      </HintButton>
      <HintButton
        label={t('lyrics.editor.backTenth', 'Back 0.1 second')}
        info={t('lyrics.editor.backTenthInfo', 'Moves the playhead a tenth of a second earlier, the finest seek offered. Useful for landing on a transient before nudging a word onto it.')}
        className={BTN}
        onClick={() => seek(Math.max(0, currentTime - FINE_SEEK_SEC))}
      >
        -{FINE_SEEK_SEC}s
      </HintButton>
      <HintButton
        label={t('lyrics.editor.forwardTenth', 'Forward 0.1 second')}
        info={t('lyrics.editor.forwardTenthInfo', 'Moves the playhead a tenth of a second later.')}
        className={BTN}
        onClick={() => seek(currentTime + FINE_SEEK_SEC)}
      >
        +{FINE_SEEK_SEC}s
      </HintButton>
      <HintButton
        label={t('lyrics.editor.forward1', 'Forward 1 second')}
        info={t('lyrics.editor.forward1Info', 'Moves the playhead one second later.')}
        className={BTN}
        onClick={() => seek(currentTime + SEEK_SEC)}
      >
        {SEEK_SEC}s<ChevronRight size={12} />
      </HintButton>
      <span className="ml-1 font-mono tabular-nums">{fmt(currentTime)} / {fmt(total)}</span>
      <div className="ml-2 flex items-center gap-1">
        <ParamLabel
          label={t('lyrics.editor.speed', 'Speed')}
          info={t('lyrics.editor.speedInfo', 'Playback speed for both the editor and the player bar, because they share one audio element. Half speed widens a syllable enough to hear where it starts.')}
        >
          <span className="flex items-center gap-1">
            <span className="opacity-60">{t('lyrics.editor.speed', 'Speed')}</span>
            <StyledSelect
              accent="pink"
              size="sm"
              value={engine.rate}
              onChange={r => setRate(Number(r))}
              options={PLAYBACK_RATES.map(r => ({ value: r, label: `${r}×` }))}
            />
          </span>
        </ParamLabel>
      </div>
      <ParamLabel
        label={t('lyrics.editor.volume', 'Volume')}
        info={t('lyrics.editor.volumeInfo', 'The shared player volume. Lower it against the track rather than muting: a quiet syllable is easier to place when you can still hear its neighbour.')}
      >
        <label className="flex items-center gap-1">
          <Volume2 size={12} />
          <input
            type="range" min={0} max={1} step={0.01} value={engine.volume}
            onChange={e => setVolume(Number(e.target.value))}
            className="w-20 accent-pink-500"
            aria-label={t('lyrics.editor.volume', 'Volume')}
          />
        </label>
      </ParamLabel>
      <Toggle
        size="sm"
        accent="pink"
        checked={follow}
        onChange={onFollow}
        label={t('lyrics.editor.follow', 'Follow')}
        info={t(
          'lyrics.editor.followHint',
          'Keeps the timeline on the playhead while the track plays. Off: the view stays where you put it. The choice is remembered for the next time the editor opens. Dragging the timeline switches it off for you, with a toast saying Follow has been disabled, so a pan is never snapped back. The paging arrows do the same.',
        )}
      />
      <Toggle
        size="sm"
        accent="pink"
        checked={autoShift}
        onChange={onAutoShift}
        label={t('lyrics.editor.autoShift', 'Auto shift')}
        info={t(
          'lyrics.editor.autoShiftHint',
          'What happens when a word is made longer than the gap before the next one. On: the words after it move later by the overhang, so the spacing you set survives and nothing overlaps. Off: the word stops at the next word instead. Either way two words can never share an instant, and a save repairs any overlap the same way.',
        )}
      />
      <HintButton
        label={t('lyrics.editor.exportLyrics', 'Export lyrics')}
        info={t(
          'lyrics.editor.exportLyricsHint',
          'Downloads this track\u2019s .lrc with the timings on screen, named to match the audio download so a player pairs the two. Pending changes are saved first, and Enhanced LRC in Settings adds per-word stamps. A track with no word timings exports its line-level .lrc instead.',
        )}
        className={BTN}
        onClick={onExport}
        disabled={exportDisabled}
        rootClassName="ml-auto"
      >
        <Download size={12} /> {t('lyrics.editor.exportLyrics', 'Export lyrics')}
      </HintButton>
    </div>
  );
};

export default LyricEditorTransport;
