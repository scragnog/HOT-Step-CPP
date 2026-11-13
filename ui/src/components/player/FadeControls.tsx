// FadeControls.tsx — fade mode controls, above the waveform.
//
// Pick intro or outro, drag the length out on the waveform, pick a curve,
// apply. The waveform draws the fade from store state and the player volume
// follows it while the mode is on, so it can be heard before it is written.

import React, { useState } from 'react';
import { X, Check, AlertTriangle, TrendingUp } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { songApi } from '../../services/api';
import { invalidateEvents } from '../../queries/invalidation';
import { HintButton } from '../shared/HintButton';
import { StyledSelect } from '../shared/StyledSelect';
import { ParamLabel } from '../shared/ParamLabel';
import { usePlaybackSelector, setFadeKind, setFadeCurve, setFadeMode, reloadCurrentTrack } from '../../stores/playbackStore';
import type { FadeCurve, FadeKind } from '../../../../server/src/contracts/fade';

interface FadeControlsProps {
  songId: string | null;
  audioUrl: string | null;
}

export const FadeControls: React.FC<FadeControlsProps> = ({ songId, audioUrl }) => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const kind = usePlaybackSelector(s => s.fadeKind);
  const length = usePlaybackSelector(s => s.fadeLength);
  const curve = usePlaybackSelector(s => s.fadeCurve);
  const [showConfirm, setShowConfirm] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fadeMut = useMutation({
    mutationFn: () => songApi.fade(songId!, { kind, lengthSec: length, curve, audioUrl: audioUrl ?? undefined }),
    onMutate: () => { setError(null); },
    onSuccess: () => {
      // Same file, same URL: reload busts the browser and peaks caches and
      // leaves fade mode.
      reloadCurrentTrack();
      void invalidateEvents(queryClient, ['song:written']);
    },
    onError: err => setError(err instanceof Error ? err.message : t('fade.failed', 'Fade failed')),
  });
  const busy = fadeMut.isPending;
  const canApply = length > 0 && !!songId;

  const curves: Array<{ value: FadeCurve; label: string; hint: string }> = [
    { value: 'linear', label: t('fade.curve.linear', 'Linear'), hint: t('fade.curve.linearHint', 'Straight ramp in gain.') },
    { value: 'exponential', label: t('fade.curve.exponential', 'Exponential'), hint: t('fade.curve.exponentialHint', 'Changes slowly near silence: a gentle intro, an outro that drops then tails off.') },
    { value: 'logarithmic', label: t('fade.curve.logarithmic', 'Logarithmic'), hint: t('fade.curve.logarithmicHint', 'Changes quickly near silence: a fast-rising intro, an outro that holds then drops.') },
    { value: 'scurve', label: t('fade.curve.scurve', 'S-curve'), hint: t('fade.curve.scurveHint', 'Eased at both ends.') },
  ];

  const kindButton = (k: FadeKind, label: string) => (
    <button
      onClick={() => setFadeKind(k)}
      disabled={busy}
      className={`px-2 py-0.5 text-[11px] font-medium transition-colors ${
        kind === k ? 'bg-amber-500/20 text-amber-400' : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'
      }`}
    >
      {label}
    </button>
  );

  return (
    <div
      className="flex-shrink-0 flex items-center gap-3 px-4 py-1.5"
      style={{ background: 'rgba(245,158,11,0.06)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}
    >
      <TrendingUp size={14} className="text-amber-400 flex-shrink-0" />

      <div className="flex rounded-md overflow-hidden border border-zinc-300 dark:border-white/10 flex-shrink-0">
        {kindButton('in', t('fade.intro', 'Intro'))}
        {kindButton('out', t('fade.outro', 'Outro'))}
      </div>

      <span className="text-[11px] text-zinc-600 dark:text-zinc-400 flex-shrink-0">
        {busy
          ? <span className="text-amber-400 animate-pulse">{t('fade.applying', 'Applying fade...')}</span>
          : kind === 'in'
            ? t('fade.dragIn', 'Drag on the waveform to set where the fade-in ends')
            : t('fade.dragOut', 'Drag on the waveform to set where the fade-out starts')}
      </span>

      <div className="flex-1" />

      {length > 0 && (
        <span className="text-[11px] font-mono text-amber-400 flex-shrink-0">{length.toFixed(1)} s</span>
      )}

      <ParamLabel
        label={t('fade.curveLabel', 'Curve')}
        info={t('fade.curveInfo', 'The shape of the fade. The player volume follows it while fade mode is on, so play the track to hear it before applying.')}
        rootClassName="flex items-center gap-1.5 flex-shrink-0"
      >
        <StyledSelect
          accent="amber"
          size="sm"
          value={curve}
          onChange={setFadeCurve}
          options={curves}
          disabled={busy}
          aria-label={t('fade.curveLabel', 'Curve')}
          className="w-32"
        />
      </ParamLabel>

      {error && (
        <span className="text-[10px] text-red-400 truncate max-w-[200px]" title={error}>{error}</span>
      )}

      <div className="flex items-center gap-1 flex-shrink-0">
        {canApply && !showConfirm && (
          <button
            onClick={() => setShowConfirm(true)}
            disabled={busy}
            className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-amber-500/15 text-amber-400 hover:bg-amber-500/25 transition-colors"
          >
            <Check size={11} />
            {t('fade.apply', 'Apply fade')}
          </button>
        )}
        {showConfirm && (
          <div className="flex items-center gap-1.5">
            <AlertTriangle size={12} className="text-amber-400" />
            <span className="text-[10px] text-amber-400">{t('trim.cantUndo')}</span>
            <button
              onClick={() => { setShowConfirm(false); fadeMut.mutate(); }}
              disabled={busy}
              className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-green-500/15 text-green-400 hover:bg-green-500/25 transition-colors"
            >
              <Check size={11} />
              {t('trim.confirm')}
            </button>
            <button
              onClick={() => setShowConfirm(false)}
              className="p-0.5 rounded text-zinc-500 hover:text-zinc-900 dark:hover:text-white transition-colors"
            >
              <X size={12} />
            </button>
          </div>
        )}
        <HintButton
          label={t('fade.exit', 'Exit fade mode')}
          info={t('fade.exitInfo', 'Leaves fade mode. Nothing is written, so the audio is unchanged.')}
          onClick={() => setFadeMode(false)}
          disabled={busy}
          className="p-1 rounded text-zinc-500 hover:text-zinc-900 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-white/5 transition-colors"
          ariaLabel={t('fade.exit', 'Exit fade mode')}
        >
          <X size={14} />
        </HintButton>
      </div>
    </div>
  );
};
