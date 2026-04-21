// RightSidebar.tsx — Selected song details panel
// Ported from hot-step-9000's RightSidebar, simplified for current feature set.

import React from 'react';
import { X, Play, Pause, RotateCcw, Trash2, Music, Clock, Gauge, Download } from 'lucide-react';
import type { Song } from '../../types';
import { useLanguage } from '../../context/LanguageContext';

interface RightSidebarProps {
  song: Song;
  onClose: () => void;
  onReuse: (song: Song) => void;
  onDelete: (song: Song) => void;
  onPlay: (song: Song) => void;
  isPlaying: boolean;
  onDownload?: (song: Song) => void;
}

export const RightSidebar: React.FC<RightSidebarProps> = ({
  song,
  onClose,
  onReuse,
  onDelete,
  onPlay,
  isPlaying,
  onDownload,
}) => {
  const { t } = useLanguage();
  const gp = song.generationParams;

  const formatDuration = (val: string | number | undefined) => {
    if (!val) return '--:--';
    if (typeof val === 'string') return val;
    const m = Math.floor(val / 60);
    const s = Math.floor(val % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
        <h3 className="text-sm font-semibold text-zinc-300 truncate">Song Details</h3>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-white/5 transition-colors"
        >
          <X size={16} />
        </button>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto hide-scrollbar p-4 space-y-4">
        {/* Cover Art Placeholder */}
        <div className="aspect-square w-48 h-48 mx-auto rounded-xl bg-gradient-to-br from-pink-500/20 to-purple-600/20 border border-white/5 flex items-center justify-center overflow-hidden">
          {song.coverUrl ? (
            <img src={song.coverUrl} alt={song.title} className="w-full h-full object-cover" />
          ) : (
            <Music size={48} className="text-zinc-600" />
          )}
        </div>

        {/* Title & Style */}
        <div>
          <h2 className="text-lg font-bold text-white leading-tight">{song.title || 'Untitled'}</h2>
          {song.style && (
            <p className="mt-1 text-sm text-zinc-400 line-clamp-2">{song.style}</p>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onPlay(song)}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl bg-pink-600 hover:bg-pink-500 text-white font-semibold transition-colors"
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={() => onReuse(song)}
            className="p-2.5 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors"
            title="Reuse Prompt"
          >
            <RotateCcw size={16} />
          </button>
          <button
            onClick={() => onDelete(song)}
            className="p-2.5 rounded-xl bg-zinc-800 hover:bg-red-900/50 text-zinc-300 hover:text-red-400 transition-colors"
            title="Delete"
          >
            <Trash2 size={16} />
          </button>
          {onDownload && (
            <button
              onClick={() => onDownload(song)}
              className="p-2.5 rounded-xl bg-zinc-800 hover:bg-emerald-900/50 text-zinc-300 hover:text-emerald-400 transition-colors"
              title="Download"
            >
              <Download size={16} />
            </button>
          )}
        </div>

        <div className="grid grid-cols-2 gap-3 mb-6">
          <MetaBadge icon={<Clock size={16} />} label={t('meta_duration')} value={formatDuration(song.duration)} />
          <MetaBadge icon={<Gauge size={16} />} label={t('meta_bpm')} value={String(song.bpm || gp?.bpm || '---')} />
          <MetaBadge 
            icon={<Music size={16} />} 
            label={t('meta_key').split(' / ')[0]} 
            value={String(gp?.keyScale || song.key_scale || t('meta_random'))} 
          />
          <MetaBadge 
            icon={<Music size={16} />} 
            label={t('meta_time_sig').split(' (')[0]} 
            value={String(gp?.timeSignature || song.time_signature || t('meta_random'))} 
          />
        </div>

        {/* Generation Info */}
        {gp && (
          <div className="space-y-3">
            <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Generation Info</h4>

            {gp.ditModel && (
              <InfoRow label="DiT Model" value={gp.ditModel.split('/').pop() || gp.ditModel} />
            )}
            {gp.inferenceSteps && (
              <InfoRow label="Steps" value={String(gp.inferenceSteps)} />
            )}
            {gp.guidanceScale !== undefined && (
              <InfoRow label="Guidance" value={String(gp.guidanceScale)} />
            )}
            {gp.seed !== undefined && (
              <InfoRow label="Seed" value={String(gp.seed)} />
            )}
            {gp.adapter && (
              <InfoRow label="Adapter" value={gp.adapter.split('/').pop() || gp.adapter} />
            )}
          </div>
        )}

        {/* Lyrics */}
        {song.lyrics && (
          <div className="space-y-2">
            <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Lyrics</h4>
            <pre className="text-sm text-zinc-300 whitespace-pre-wrap font-sans leading-relaxed bg-zinc-900/50 rounded-xl p-3 border border-white/5 max-h-64 overflow-y-auto">
              {song.lyrics}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

const MetaBadge: React.FC<{ icon: React.ReactNode; label: string; value: string }> = ({ icon, label, value }) => (
  <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-800/50 border border-white/5">
    <div className="text-zinc-500">{icon}</div>
    <div className="min-w-0">
      <div className="text-[10px] text-zinc-500 uppercase tracking-wider">{label}</div>
      <div className="text-sm text-zinc-200 font-medium truncate">{value}</div>
    </div>
  </div>
);

const InfoRow: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="flex items-center justify-between text-sm">
    <span className="text-zinc-500">{label}</span>
    <span className="text-zinc-200 font-medium truncate ml-4 text-right">{value}</span>
  </div>
);
