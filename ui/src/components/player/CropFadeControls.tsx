// CropFadeControls.tsx — crop and fade mode controls, above the waveform.
//
// Pick Intro or Outro, drag on the waveform to set where that end is cut, set
// a fade length and curve for it, then Done. Each end keeps its own settings,
// so both can be set before anything is written. The waveform draws the edit
// from store state and the player follows it while the mode is on, so it can
// be heard first. Done writes it straight away: crop, then the fades.

import React, { useState } from 'react';
import { toast } from 'react-toastify';
import { X, Check, Scissors } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { songApi } from '../../services/api';
import { invalidateEvents } from '../../queries/invalidation';
import { HintButton } from '../shared/HintButton';
import { StyledSelect } from '../shared/StyledSelect';
import { ParamLabel } from '../shared/ParamLabel';
import {
  usePlaybackSelector, setEditEnd, setEndFade, setEndCurve, setEditMode, reloadCurrentTrack,
} from '../../stores/playbackStore';
import type { FadeCurve, FadeKind } from '../../../../server/src/contracts/fade';

interface CropFadeControlsProps {
  songId: string | null;
  audioUrl: string | null;
}

const formatTime = (s: number) => {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}.${Math.floor((s % 1) * 10)}`;
};

export const CropFadeControls: React.FC<CropFadeControlsProps> = ({ songId, audioUrl }) => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const end = usePlaybackSelector(s => s.editEnd);
  const edit = usePlaybackSelector(s => s.endEdit);
  const duration = usePlaybackSelector(s => s.duration);
  const [error, setError] = useState<string | null>(null);

  const cutIn = edit.cropIn;
  const cutOut = edit.cropOut ?? duration;
  const cropping = cutIn > 0.05 || edit.cropOut !== null;
  const changed = cropping || edit.fadeIn > 0 || edit.fadeOut > 0;
  const fade = end === 'in' ? edit.fadeIn : edit.fadeOut;
  const curve = end === 'in' ? edit.curveIn : edit.curveOut;
  const fadeMax = Math.max(0.1, Math.min(30, cutOut - cutIn));

  const applyMut = useMutation({
    mutationFn: async () => {
      const id = songId!;
      const url = audioUrl ?? undefined;
      let wrote = false;
      let newDuration: number | undefined;
      try {
        if (cropping) {
          const r = await songApi.crop(id, cutIn, cutOut, url);
          wrote = r.cropped;
          if (r.cropped) newDuration = r.newDuration;
        }
        // After the crop both cuts are the file's edges, which is where a fade starts.
        if (edit.fadeIn > 0) {
          await songApi.fade(id, { kind: 'in', lengthSec: edit.fadeIn, curve: edit.curveIn, audioUrl: url });
          wrote = true;
        }
        if (edit.fadeOut > 0) {
          await songApi.fade(id, { kind: 'out', lengthSec: edit.fadeOut, curve: edit.curveOut, audioUrl: url });
          wrote = true;
        }
      } catch (err) {
        // Part of the edit is already on disk: show the file as it now is.
        if (wrote) {
          reloadCurrentTrack(newDuration);
          void invalidateEvents(queryClient, ['song:written', 'sidecars:changed']);
          toast.error(t('cropFade.partial', 'Only part of the edit was applied: {{error}}', {
            error: err instanceof Error ? err.message : String(err),
          }));
        }
        throw err;
      }
      return newDuration;
    },
    onMutate: () => { setError(null); },
    onSuccess: newDuration => {
      // Same file, same URL: reload busts the browser and peaks caches and
      // leaves the mode.
      reloadCurrentTrack(newDuration);
      void invalidateEvents(queryClient, ['song:written', 'sidecars:changed']);
    },
    onError: err => setError(err instanceof Error ? err.message : t('cropFade.failed', 'Crop and fade failed')),
  });
  const busy = applyMut.isPending;

  const curves: Array<{ value: FadeCurve; label: string; hint: string }> = [
    { value: 'linear', label: t('fade.curve.linear', 'Linear'), hint: t('fade.curve.linearHint', 'Straight ramp in gain.') },
    { value: 'exponential', label: t('fade.curve.exponential', 'Exponential'), hint: t('fade.curve.exponentialHint', 'Changes slowly near silence: a gentle intro, an outro that drops then tails off.') },
    { value: 'logarithmic', label: t('fade.curve.logarithmic', 'Logarithmic'), hint: t('fade.curve.logarithmicHint', 'Changes quickly near silence: a fast-rising intro, an outro that holds then drops.') },
    { value: 'scurve', label: t('fade.curve.scurve', 'S-curve'), hint: t('fade.curve.scurveHint', 'Eased at both ends.') },
  ];

  const endButton = (k: FadeKind, label: string) => (
    <button
      onClick={() => setEditEnd(k)}
      disabled={busy}
      className={`px-2 py-0.5 text-[11px] font-medium transition-colors ${
        end === k ? 'bg-cyan-500/20 text-cyan-400' : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'
      }`}
    >
      {label}
    </button>
  );

  return (
    <div
      className="flex-shrink-0 flex items-center gap-3 px-4 py-1.5"
      style={{ background: 'rgba(6,182,212,0.06)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}
    >
      <Scissors size={14} className="text-cyan-400 flex-shrink-0" />

      <div className="flex rounded-md overflow-hidden border border-zinc-300 dark:border-white/10 flex-shrink-0">
        {endButton('in', t('fade.intro', 'Intro'))}
        {endButton('out', t('fade.outro', 'Outro'))}
      </div>

      <span className="text-[11px] text-zinc-600 dark:text-zinc-400 min-w-0 truncate">
        {busy
          ? <span className="text-amber-400 animate-pulse">{t('cropFade.applying', 'Applying...')}</span>
          : end === 'in'
            ? t('cropFade.dragIn', 'Drag on the waveform to set where the track starts')
            : t('cropFade.dragOut', 'Drag on the waveform to set where the track ends')}
      </span>

      <div className="flex-1" />

      <span className="text-[11px] font-mono flex-shrink-0">
        <span className="text-green-400">{formatTime(cutIn)}</span>
        <span className="text-zinc-500"> – </span>
        <span className="text-red-400">{formatTime(cutOut)}</span>
      </span>

      <ParamLabel
        label={t('cropFade.fade', 'Fade')}
        info={t('cropFade.fadeInfo', 'How long this end fades, measured inwards from the cut. 0 means no fade.')}
        rootClassName="flex items-center gap-1.5 flex-shrink-0"
      >
        <input
          type="range"
          min={0}
          max={fadeMax}
          step={0.1}
          value={Math.min(fade, fadeMax)}
          onChange={e => setEndFade(Number(e.target.value))}
          disabled={busy}
          aria-label={t('cropFade.fade', 'Fade')}
          className="w-24 accent-amber-500"
        />
        <span className="text-[11px] font-mono text-amber-400 w-10">{fade.toFixed(1)} s</span>
      </ParamLabel>

      <ParamLabel
        label={t('fade.curveLabel', 'Curve')}
        info={t('cropFade.curveInfo', 'The shape of this end\'s fade. The player follows it while the mode is on, so play the track to hear it first.')}
        rootClassName="flex items-center gap-1.5 flex-shrink-0"
      >
        <StyledSelect
          accent="amber"
          size="sm"
          value={curve}
          onChange={setEndCurve}
          options={curves}
          disabled={busy || fade <= 0}
          aria-label={t('fade.curveLabel', 'Curve')}
          className="w-32"
        />
      </ParamLabel>

      {error && (
        <span className="text-[10px] text-red-400 truncate max-w-[200px]" title={error}>{error}</span>
      )}

      <div className="flex items-center gap-1 flex-shrink-0">
        <HintButton
          label={t('cropFade.done', 'Done')}
          info={t('cropFade.doneInfo', 'Writes the crop and both fades into the audio file (and the mastered file, if there is one) and cuts the lyric sidecars to match. There is no confirmation and it cannot be undone.')}
          onClick={() => applyMut.mutate()}
          disabled={busy || !changed || !songId}
          className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-green-500/15 text-green-400 hover:bg-green-500/25 disabled:opacity-40 transition-colors"
        >
          <Check size={11} />
          {t('cropFade.done', 'Done')}
        </HintButton>
        <HintButton
          label={t('cropFade.cancel', 'Cancel')}
          info={t('cropFade.cancelInfo', 'Leaves crop and fade mode. Nothing is written, so the audio is unchanged.')}
          onClick={() => setEditMode(false)}
          disabled={busy}
          className="p-1 rounded text-zinc-500 hover:text-zinc-900 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-white/5 transition-colors"
          ariaLabel={t('cropFade.cancel', 'Cancel')}
        >
          <X size={14} />
        </HintButton>
      </div>
    </div>
  );
};
