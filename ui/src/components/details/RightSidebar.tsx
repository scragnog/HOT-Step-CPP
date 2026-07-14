// RightSidebar.tsx — Selected song details panel
// Ported from hot-step-9000's RightSidebar, simplified for current feature set.

import React from 'react';
import { X, Play, Pause, RotateCcw, Trash2, Music, Clock, Hash, Gauge, Download, Upload, Cpu, Terminal, Settings2, Zap, Radio, Activity, Layers, Sparkles, SlidersHorizontal, Pencil, Disc3, Tags, Image as ImageIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { Song } from '../../types';
import { HoverFullText } from '../shared/HoverFullText';
import { openCoverArtPrompt } from '../library/CoverArtPromptModal';
import { formatDitModel, formatLmModel } from '../global-bar/modelLabels';

interface RightSidebarProps {
  song: Song;
  onClose: () => void;
  onReuse: (song: Song) => void;
  onDelete: (song: Song) => void;
  onPlay: (song: Song) => void;
  isPlaying: boolean;
  onDownload?: (song: Song) => void;
  onRename?: (song: Song, newTitle: string) => void;
  onSendToCover?: (song: Song) => void;
  onEditMetadata?: (song: Song) => void;
}

export const RightSidebar: React.FC<RightSidebarProps> = ({
  song,
  onClose,
  onReuse,
  onDelete,
  onPlay,
  isPlaying,
  onDownload,
  onRename,
  onSendToCover,
  onEditMetadata,
}) => {
  const { t } = useTranslation();
  const gp = song.generationParams;

  // Inline rename state
  const [editing, setEditing] = React.useState(false);
  const [editTitle, setEditTitle] = React.useState(song.title || '');
  const inputRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  // Reset edit state when song changes
  React.useEffect(() => {
    setEditing(false);
    setEditTitle(song.title || '');
  }, [song.id]);

  const commitRename = () => {
    const trimmed = editTitle.trim();
    if (trimmed && trimmed !== (song.title || '')) {
      onRename?.(song, trimmed);
    }
    setEditing(false);
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-white/5">
        <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 truncate">{t('details.songDetails')}</h3>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-zinc-600 dark:text-zinc-400 hover:text-white hover:bg-white/5 transition-colors"
        >
          <X size={16} />
        </button>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto hide-scrollbar p-4 space-y-4">
        {/* Cover Art Placeholder */}
        <div className="aspect-square w-full rounded-xl bg-gradient-to-br from-pink-500/20 to-purple-600/20 border border-zinc-200 dark:border-white/5 flex items-center justify-center">
          {song.coverUrl ? (
            <img src={song.coverUrl} alt={song.title} className="w-full h-full object-cover rounded-xl" />
          ) : (
            <Music size={48} className="text-zinc-600" />
          )}
        </div>

        {/* Title & Style */}
        <div>
          {editing ? (
            <input
              ref={inputRef}
              className="w-full text-lg font-bold bg-zinc-800 border border-pink-500/40 rounded-lg px-2 py-0.5 text-white outline-none focus:border-pink-500"
              value={editTitle}
              onChange={e => setEditTitle(e.target.value)}
              onBlur={commitRename}
              onKeyDown={e => {
                if (e.key === 'Enter') commitRename();
                if (e.key === 'Escape') { setEditTitle(song.title || ''); setEditing(false); }
              }}
            />
          ) : (
            <div className="flex items-center gap-1.5 group/title">
              <h2 className="text-lg font-bold text-white leading-tight truncate">{song.title || 'Untitled'}</h2>
              {onRename && (
                <button
                  onClick={() => { setEditTitle(song.title || ''); setEditing(true); }}
                  className="flex-shrink-0 p-1 rounded-lg text-zinc-600 hover:text-zinc-300 opacity-0 group-hover/title:opacity-100 transition-opacity"
                  title={t('library.rename')}
                >
                  <Pencil size={13} />
                </button>
              )}
            </div>
          )}
          {song.style && (
            <HoverFullText
              as="p"
              text={song.style}
              className="mt-1 text-sm text-zinc-600 dark:text-zinc-400 line-clamp-2 cursor-help"
            />
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onPlay(song)}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl bg-pink-600 hover:bg-pink-500 text-white font-semibold transition-colors"
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            {isPlaying ? t('details.pause') : t('details.play')}
          </button>
          <button
            onClick={() => onReuse(song)}
            className="p-2.5 rounded-xl bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-300 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300 transition-colors"
            title={t('details.edit')}
          >
            <RotateCcw size={16} />
          </button>
          <button
            onClick={() => onDelete(song)}
            className="p-2.5 rounded-xl bg-zinc-100 dark:bg-zinc-800 hover:bg-red-100 dark:hover:bg-red-900/50 text-zinc-700 dark:text-zinc-300 hover:text-red-400 transition-colors"
            title={t('details.delete')}
          >
            <Trash2 size={16} />
          </button>
          {onDownload && (
            <button
              onClick={() => onDownload(song)}
              className="p-2.5 rounded-xl bg-zinc-100 dark:bg-zinc-800 hover:bg-emerald-100 dark:hover:bg-emerald-900/50 text-zinc-700 dark:text-zinc-300 hover:text-emerald-400 transition-colors"
              title={t('details.download')}
            >
              <Download size={16} />
            </button>
          )}
          <button
            onClick={() => openCoverArtPrompt(song)}
            className="p-2.5 rounded-xl bg-zinc-100 dark:bg-zinc-800 hover:bg-pink-100 dark:hover:bg-pink-900/50 text-zinc-700 dark:text-zinc-300 hover:text-pink-400 transition-colors"
            title={(song.coverUrl || song.cover_url)
              ? t('coverArt.regenerateTitle', 'Regenerate Cover Art')
              : t('coverArt.generateTitle', 'Generate Cover Art')}
          >
            <ImageIcon size={16} />
          </button>
          {onSendToCover && (
            <button
              onClick={() => onSendToCover(song)}
              className="p-2.5 rounded-xl bg-zinc-100 dark:bg-zinc-800 hover:bg-cyan-100 dark:hover:bg-cyan-900/50 text-zinc-700 dark:text-zinc-300 hover:text-cyan-400 transition-colors"
              title={t('library.sendToCover', 'Send to Cover Studio')}
            >
              <Disc3 size={16} />
            </button>
          )}
          {onEditMetadata && (
            <button
              onClick={() => onEditMetadata(song)}
              className="p-2.5 rounded-xl bg-zinc-100 dark:bg-zinc-800 hover:bg-amber-100 dark:hover:bg-amber-900/50 text-zinc-700 dark:text-zinc-300 hover:text-amber-400 transition-colors"
              title={t('metadata.editTitle', 'Edit Metadata')}
            >
              <Tags size={16} />
            </button>
          )}
          {gp && (
            <button
              onClick={() => {
                const params = song.generationParams || song.generation_params || {};
                const exportData = { _format: 'hot-step-preset', _version: 1, ...params, title: song.title || '', caption: (params as any).caption || song.style || '', lyrics: (params as any).lyrics || song.lyrics || '' };
                const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${(song.title || 'song').slice(0, 40).replace(/[^a-zA-Z0-9]/g, '_').toLowerCase()}_params.json`;
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="p-2.5 rounded-xl bg-zinc-100 dark:bg-zinc-800 hover:bg-sky-100 dark:hover:bg-sky-900/50 text-zinc-700 dark:text-zinc-300 hover:text-sky-400 transition-colors"
              title={t('details.exportParams')}
            >
              <Upload size={16} />
            </button>
          )}
        </div>

        {/* Metadata Badges */}
        <div className="grid grid-cols-2 gap-2">
          {song.duration && (
            <MetaBadge icon={<Clock size={14} />} label={t('details.duration')} value={String(song.duration)} gradient="from-amber-500/10 to-orange-500/10 border-amber-200 dark:border-amber-500/30" iconColor="text-amber-600 dark:text-amber-400" />
          )}
          {(song.bpm || gp?.bpm) ? (
            <MetaBadge icon={<Gauge size={14} />} label={t('details.bpm')} value={String(song.bpm || gp?.bpm)} gradient="from-rose-500/10 to-pink-500/10 border-rose-200 dark:border-rose-500/30" iconColor="text-rose-600 dark:text-rose-400" />
          ) : null}
          {gp?.keyScale && (
            <MetaBadge icon={<Hash size={14} />} label={t('details.key')} value={gp.keyScale} gradient="from-emerald-500/10 to-teal-500/10 border-emerald-200 dark:border-emerald-500/30" iconColor="text-emerald-600 dark:text-emerald-400" />
          )}
          {gp?.timeSignature && (
            <MetaBadge icon={<Music size={14} />} label={t('details.timeSig')} value={gp.timeSignature} gradient="from-violet-500/10 to-purple-500/10 border-violet-200 dark:border-violet-500/30" iconColor="text-violet-600 dark:text-violet-400" />
          )}
        </div>

        {/* Generation Parameters Grid — HOT-Step 9000 style */}
        {gp && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-indigo-500/20 to-purple-500/20 border border-indigo-200 dark:border-indigo-500/30 flex items-center justify-center">
                <SlidersHorizontal size={12} className="text-indigo-600 dark:text-indigo-400" />
              </div>
              <h4 className="text-xs font-bold text-zinc-800 dark:text-zinc-200 uppercase tracking-wider">{t('details.generationInfo')}</h4>
            </div>

            <div className="grid grid-cols-2 gap-2">
              {/* Models — blue accent */}
              {gp.ditModel && (
                <ParamCell
                  label="DiT Model"
                  value={formatDitModel(gp.ditModel)}
                  title={gp.ditModel}
                  gradient="from-blue-500/10 to-cyan-500/10 border-blue-200 dark:border-blue-500/30"
                  iconColor="text-blue-600 dark:text-blue-400"
                  icon={<Cpu size={12} />}
                />
              )}
              {gp.lmModel && (
                <ParamCell
                  label="LM Model"
                  value={formatLmModel(gp.lmModel)}
                  title={gp.lmModel}
                  gradient="from-blue-500/10 to-cyan-500/10 border-blue-200 dark:border-blue-500/30"
                  iconColor="text-blue-600 dark:text-blue-400"
                  icon={<Terminal size={12} />}
                />
              )}

              {/* Engine — tech accent */}
              {gp.inferenceSteps && (
                <ParamCell
                  label="Steps"
                  value={String(gp.inferenceSteps)}
                  gradient="from-slate-500/10 to-zinc-500/10 border-slate-200 dark:border-slate-500/30"
                  iconColor="text-slate-600 dark:text-slate-400"
                  icon={<Gauge size={12} />}
                />
              )}
              {gp.guidanceScale !== undefined && (
                <ParamCell
                  label="CFG Scale"
                  value={String(gp.guidanceScale)}
                  gradient="from-slate-500/10 to-zinc-500/10 border-slate-200 dark:border-slate-500/30"
                  iconColor="text-slate-600 dark:text-slate-400"
                  icon={<Settings2 size={12} />}
                />
              )}

              {/* Solver + Scheduler — violet accent */}
              {gp.inferMethod && (
                <ParamCell
                  label="Solver"
                  value={gp.inferMethod.toUpperCase()}
                  gradient="from-violet-500/10 to-purple-500/10 border-violet-200 dark:border-violet-500/30"
                  iconColor="text-violet-600 dark:text-violet-400"
                  icon={<Zap size={12} />}
                />
              )}
              {gp.scheduler && gp.scheduler !== 'linear' && (
                <ParamCell
                  label="Schedule"
                  value={gp.scheduler.split(':')[0].replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase())}
                  gradient="from-violet-500/10 to-purple-500/10 border-violet-200 dark:border-violet-500/30"
                  iconColor="text-violet-600 dark:text-violet-400"
                  icon={<Clock size={12} />}
                />
              )}

              {/* Guidance — emerald accent */}
              {gp.guidanceMode && (
                <ParamCell
                  label="Guidance"
                  value={gp.guidanceMode.toUpperCase()}
                  gradient="from-emerald-500/10 to-teal-500/10 border-emerald-200 dark:border-emerald-500/30"
                  iconColor="text-emerald-600 dark:text-emerald-400"
                  icon={<Radio size={12} />}
                />
              )}

              {/* Shift */}
              {gp.shift !== undefined && (
                <ParamCell
                  label="Shift"
                  value={gp.shift < 0 ? 'Auto' : String(gp.shift)}
                  gradient="from-amber-500/10 to-orange-500/10 border-amber-200 dark:border-amber-500/30"
                  iconColor="text-amber-600 dark:text-amber-400"
                  icon={<Activity size={12} />}
                />
              )}

              {/* Seed — mono */}
              {gp.seed !== undefined && (
                <ParamCell
                  label="Seed"
                  value={String(gp.seed).substring(0, 12) + (String(gp.seed).length > 12 ? '…' : '')}
                  gradient="from-slate-500/10 to-zinc-500/10 border-slate-200 dark:border-slate-500/30"
                  iconColor="text-slate-600 dark:text-slate-400"
                  icon={<Hash size={12} />}
                  mono
                />
              )}

              {/* LM Seed — mono */}
              {gp.lmSeed !== undefined && (
                <ParamCell
                  label="LM Seed"
                  value={String(gp.lmSeed).substring(0, 12) + (String(gp.lmSeed).length > 12 ? '…' : '')}
                  gradient="from-slate-500/10 to-zinc-500/10 border-slate-200 dark:border-slate-500/30"
                  iconColor="text-slate-600 dark:text-slate-400"
                  icon={<Hash size={12} />}
                  mono
                />
              )}

              {/* Batch Size */}
              {gp.batchSize && gp.batchSize > 1 && (
                <ParamCell
                  label="Batch"
                  value={String(gp.batchSize)}
                  gradient="from-slate-500/10 to-zinc-500/10 border-slate-200 dark:border-slate-500/30"
                  iconColor="text-slate-600 dark:text-slate-400"
                  icon={<Layers size={12} />}
                />
              )}

              {/* Adapter — pink accent */}
              {(gp.adapter || gp.loraPath) && (
                <ParamCell
                  label="Adapter"
                  value={getModelShortName(gp.adapter || gp.loraPath || '')}
                  title={gp.adapter || gp.loraPath || ''}
                  gradient="from-pink-500/10 to-rose-500/10 border-pink-200 dark:border-pink-500/30"
                  iconColor="text-pink-600 dark:text-pink-400"
                  icon={<Sparkles size={12} />}
                  span2
                />
              )}
              {gp.loraScale !== undefined && gp.loraScale !== 1 && (gp.adapter || gp.loraPath) && (
                <ParamCell
                  label="Adapter Scale"
                  value={String(gp.loraScale)}
                  gradient="from-pink-500/10 to-rose-500/10 border-pink-200 dark:border-pink-500/30"
                  iconColor="text-pink-600 dark:text-pink-400"
                  icon={<SlidersHorizontal size={12} />}
                />
              )}

              {/* Thinking */}
              {gp.useCotCaption !== undefined && (
                <ParamCell
                  label="Thinking"
                  value={gp.useCotCaption ? 'ON' : 'OFF'}
                  gradient={gp.useCotCaption
                    ? "from-emerald-500/10 to-green-500/10 border-emerald-200 dark:border-emerald-500/30"
                    : "from-slate-500/10 to-zinc-500/10 border-slate-200 dark:border-slate-500/30"}
                  iconColor={gp.useCotCaption ? "text-emerald-600 dark:text-emerald-400" : "text-slate-600 dark:text-slate-400"}
                  icon={<Zap size={12} />}
                />
              )}
            </div>
          </div>
        )}

        {/* Lyrics */}
        {song.lyrics && (
          <div className="space-y-2">
            <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">{t('details.lyrics')}</h4>
            <pre className="text-sm text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap font-sans leading-relaxed bg-zinc-50/80 dark:bg-zinc-900/50 rounded-xl p-3 border border-zinc-200 dark:border-white/5 max-h-64 overflow-y-auto">
              {song.lyrics}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

/** Extract basename from a full model path */
const getModelShortName = (modelId: string): string => {
  const base = modelId.split(/[\\/]/).filter(Boolean).pop() || modelId;
  return base.replace(/^acestep-/, '');
};

/** Color-coded metadata badge (top section) */
const MetaBadge: React.FC<{ icon: React.ReactNode; label: string; value: string; gradient: string; iconColor: string }> = ({ icon, label, value, gradient, iconColor }) => (
  <div className={`flex items-center gap-2 px-3 py-2 rounded-lg bg-gradient-to-r ${gradient} border`}>
    <div className={iconColor}>{icon}</div>
    <div className="min-w-0">
      <div className="text-[10px] text-zinc-500 uppercase tracking-wider">{label}</div>
      <div className="text-sm text-zinc-800 dark:text-zinc-200 font-medium truncate">{value}</div>
    </div>
  </div>
);

/** Color-coded generation parameter cell (2-column grid) */
const ParamCell: React.FC<{
  label: string;
  value: string;
  gradient: string;
  iconColor: string;
  icon: React.ReactNode;
  mono?: boolean;
  span2?: boolean;
  title?: string;  // full text shown on hover (the cell value is truncated)
}> = ({ label, value, gradient, iconColor, icon, mono, span2, title }) => (
  <div className={`flex items-center gap-2 px-2.5 py-2 rounded-lg bg-gradient-to-r ${gradient} border ${span2 ? 'col-span-2' : ''}`}>
    <div className={`${iconColor} flex-shrink-0`}>{icon}</div>
    <div className="min-w-0 flex-1">
      <div className="text-[9px] text-zinc-500 uppercase tracking-wider leading-none mb-0.5">{label}</div>
      <div className={`text-xs text-zinc-800 dark:text-zinc-200 font-semibold truncate ${mono ? 'font-mono' : ''}`} title={title ?? value}>{value}</div>
    </div>
  </div>
);
