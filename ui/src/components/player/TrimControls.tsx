// TrimControls.tsx — Trim mode controls overlay
//
// Renders above the waveform when trim mode is active. Shows contextual
// instructions, IN/OUT time readout, and Crop/Reset/Cancel buttons.
// The waveform draws the IN/OUT markers itself from store state.

import React, { useState } from 'react';
import { Scissors, X, RotateCcw, Check, AlertTriangle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { songApi } from '../../services/api';
import { invalidateEvents } from '../../queries/invalidation';
import { useAuth } from '../../context/AuthContext';
import { HintButton } from '../shared/HintButton';

interface TrimControlsProps {
  trimInPoint: number | null;
  trimOutPoint: number | null;
  trimClickCount: number;
  songId: string | null;
  audioUrl: string | null;
  onReload: (newDuration?: number) => void;
  onCancel: () => void;
}

const formatTime = (s: number) => {
  if (!s || !isFinite(s)) return '0:00.0';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const ms = Math.floor((s % 1) * 10);
  return `${m}:${sec.toString().padStart(2, '0')}.${ms}`;
};

export const TrimControls: React.FC<TrimControlsProps> = ({
  trimInPoint,
  trimOutPoint,
  trimClickCount,
  songId,
  audioUrl,
  onReload,
  onCancel,
}) => {
  const { t } = useTranslation();
  const { token } = useAuth();
  const queryClient = useQueryClient();
  const [showConfirm, setShowConfirm] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The IN/OUT markers are drawn by the waveform itself, which reads them from
  // the store. Nothing to push anywhere.

  const canCrop = trimInPoint !== null && trimOutPoint !== null && songId;

  // Crop is a single user command: one mutation. songApi.crop is silent (its
  // sole consumer renders an inline error), so this line is the only notifier
  // on failure (R2). A successful crop rewrites the row's duration in every
  // song view and cuts its lyric sidecars, so invalidate both via the manager.
  const cropMut = useMutation({
    mutationFn: () => songApi.crop(songId!, trimInPoint!, trimOutPoint!, audioUrl ?? undefined),
    onMutate: () => { setError(null); },
    onSuccess: result => {
      if (!result.cropped) return;
      onReload(result.newDuration);
      void invalidateEvents(queryClient, ['song:written', 'sidecars:changed']);
    },
    onError: err => setError(err instanceof Error ? err.message : t('trim.cropFailed')),
  });
  const isCropping = cropMut.isPending;
  const confirmCrop = () => {
    if (!canCrop || !token) return;
    setShowConfirm(false);
    cropMut.mutate();
  };

  // Instruction text based on state
  let instruction = '';
  if (trimClickCount === 0) {
    instruction = t('trim.clickInPoint');
  } else if (trimClickCount === 1) {
    instruction = t('trim.clickOutPoint');
  } else {
    instruction = t('trim.adjustOrCrop');
  }

  const trimDuration = (trimInPoint !== null && trimOutPoint !== null)
    ? trimOutPoint - trimInPoint
    : null;

  return (
    <div className="flex-shrink-0 relative">
      {/* Main controls bar */}
      <div
        className="flex items-center gap-3 px-4 py-1.5"
        style={{
          background: 'linear-gradient(90deg, rgba(6,182,212,0.08) 0%, rgba(6,182,212,0.03) 50%, rgba(239,68,68,0.03) 50%, rgba(239,68,68,0.08) 100%)',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
        }}
      >
        {/* Trim icon */}
        <Scissors size={14} className="text-cyan-400 flex-shrink-0" />

        {/* Instruction / status */}
        <span className="text-[11px] text-zinc-600 dark:text-zinc-400 flex-shrink-0">
          {isCropping ? (
            <span className="text-amber-400 animate-pulse">{t('trim.cropping')}</span>
          ) : (
            instruction
          )}
        </span>

        {/* Spacer */}
        <div className="flex-1" />

        {/* IN/OUT readout */}
        {trimInPoint !== null && (
          <span className="text-[11px] font-mono text-green-400 flex-shrink-0">
            IN {formatTime(trimInPoint)}
          </span>
        )}
        {trimOutPoint !== null && (
          <span className="text-[11px] font-mono text-red-400 flex-shrink-0">
            OUT {formatTime(trimOutPoint)}
          </span>
        )}
        {trimDuration !== null && (
          <span className="text-[11px] font-mono text-zinc-500 flex-shrink-0">
            ({formatTime(trimDuration)})
          </span>
        )}

        {/* Error */}
        {error && (
          <span className="text-[10px] text-red-400 truncate max-w-[200px]" title={error}>
            {error}
          </span>
        )}

        {/* Actions */}
        <div className="flex items-center gap-1 flex-shrink-0">
          {/* Reset markers */}
          {trimClickCount > 0 && (
            <HintButton
              label={t('trim.resetMarkers')}
              info={t('trim.resetMarkersInfo', 'Clears the in and out points so new ones can be clicked on the waveform. Nothing has been written yet, so this costs nothing.')}
              onClick={() => {
                onCancel();
                // Re-enter trim mode (cancel + re-enable handled by parent)
              }}
              className="p-1 rounded text-zinc-500 hover:text-zinc-900 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-white/5 transition-colors"
              disabled={isCropping}
              ariaLabel={t('trim.resetMarkers')}
            >
              <RotateCcw size={13} />
            </HintButton>
          )}

          {/* Crop button */}
          {canCrop && !showConfirm && (
            <button
              onClick={() => setShowConfirm(true)}
              className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-cyan-500/15 text-cyan-400 hover:bg-cyan-500/25 transition-colors"
              disabled={isCropping}
            >
              <Scissors size={11} />
              {t('trim.crop')}
            </button>
          )}

          {/* Confirmation */}
          {showConfirm && (
            <div className="flex items-center gap-1.5 animate-in fade-in">
              <AlertTriangle size={12} className="text-amber-400" />
              <span className="text-[10px] text-amber-400">{t('trim.cantUndo')}</span>
              <button
                onClick={confirmCrop}
                className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-green-500/15 text-green-400 hover:bg-green-500/25 transition-colors"
                disabled={isCropping}
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

          {/* Cancel trim mode */}
          <HintButton
            label={t('trim.exitTrimMode')}
            info={t('trim.exitTrimModeInfo', 'Leaves trim mode and clears the markers. Nothing is cropped, so the audio is unchanged.')}
            onClick={onCancel}
            className="p-1 rounded text-zinc-500 hover:text-zinc-900 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-white/5 transition-colors"
            disabled={isCropping}
            ariaLabel={t('trim.exitTrimMode')}
          >
            <X size={14} />
          </HintButton>
        </div>
      </div>
    </div>
  );
};
