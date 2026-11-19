import { useQuery } from '@tanstack/react-query';
import { monitorOptions } from '@/queries/global-models/options';
// MonitorBar.tsx – Persistent VST monitor status strip
//
// Always polls /api/vst/monitor/status (slow when idle, fast when active)
// so it self-activates after page reload even if monitor was already running.

import React from 'react';
import { useTranslation } from 'react-i18next';
import { Square, Pause, Play } from 'lucide-react';
import { HintButton } from '@/components/shared/controls/HintButton';
import { useVstChainStore } from '@/stores/vstChainStore';
import { usePlaybackSelector } from '@/stores/playbackStore';
import { formatClock } from '@/utils/format/format';

const formatTime = (s: number) => formatClock(s);

export const MonitorBar: React.FC = () => {
  const { t } = useTranslation();
  const {
    monitoring, monitorPaused,
    monitorPosition, monitorDuration,
    stopMonitor, pauseMonitor, resumeMonitor,
    seekMonitor,
  } = useVstChainStore();

  const currentTrack = usePlaybackSelector(s => s.currentTrack);

  useQuery(monitorOptions());

  if (!monitoring) return null;

  const progress = monitorDuration > 0
    ? Math.min(100, (monitorPosition / monitorDuration) * 100)
    : 0;

  return (
    <div className="flex items-center gap-2 w-full min-w-0 overflow-hidden px-2.5 py-1.5 rounded-xl bg-violet-500/10 border border-violet-500/25">
      {/* Live / paused indicator */}
      {monitorPaused ? (
        <span className="h-2 w-2 rounded-full bg-violet-500/40 flex-shrink-0" />
      ) : (
        <span className="relative flex h-2 w-2 flex-shrink-0">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-violet-500" />
        </span>
      )}

      {/* Track name */}
      <span className="text-[10px] text-violet-300 font-mono truncate max-w-[72px]">
        {currentTrack?.title || 'monitor'}
      </span>

      {/* Seek slider */}
      {monitorDuration > 0 && (
        <input
          type="range"
          min={0}
          max={monitorDuration}
          step={0.5}
          value={monitorPosition}
          onChange={e => seekMonitor(parseFloat(e.target.value))}
          className="flex-1 min-w-0 h-1 rounded-full appearance-none cursor-pointer"
          style={{
            background: `linear-gradient(to right, rgb(139 92 246) ${progress}%, rgb(63 63 70) ${progress}%)`,
          }}
        />
      )}

      {/* Time */}
      <span className="text-[10px] text-violet-300/60 font-mono flex-shrink-0 tabular-nums">
        {formatTime(monitorPosition)}
        {monitorDuration > 0 && <span className="text-violet-500/40">/{formatTime(monitorDuration)}</span>}
      </span>

      {/* Pause / Resume */}
      <HintButton
        onClick={monitorPaused ? resumeMonitor : pauseMonitor}
        label={monitorPaused ? t('monitor.resume', 'Resume') : t('monitor.pause', 'Pause')}
        info={monitorPaused
          ? t('monitor.resumeInfo', 'Resumes playback in the monitor without restarting the plugin chain.')
          : t('monitor.pauseInfo', 'Pauses monitor playback. The plugin chain and position are kept.')}
        ariaLabel={monitorPaused ? t('monitor.resume', 'Resume') : t('monitor.pause', 'Pause')}
        rootClassName="flex-shrink-0"
        className="p-1 rounded hover:bg-violet-500/20 text-violet-400 hover:text-violet-200 transition-colors flex-shrink-0"
      >
        {monitorPaused ? <Play size={11} /> : <Pause size={11} />}
      </HintButton>

      {/* Stop */}
      <HintButton
        onClick={stopMonitor}
        label={t('monitor.stop', 'Stop monitor')}
        info={t('monitor.stopInfo', 'Stops the monitor and unloads its plugin chain. This cannot be undone from this bar.')}
        ariaLabel={t('monitor.stop', 'Stop monitor')}
        rootClassName="flex-shrink-0"
        className="p-1 rounded hover:bg-red-500/10 text-violet-400 hover:text-red-400 transition-colors flex-shrink-0"
      >
        <Square size={11} />
      </HintButton>
    </div>
  );
};
