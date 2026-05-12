// InspirePreview.tsx — Preview/edit panel for inspire results
//
// Shows generated lyrics + metadata after the inspire step.
// User can edit lyrics and caption before committing to full generation.

import React from 'react';
import { Music, ArrowLeft, Pencil } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { InspireResult } from '../../services/inspireApi';

interface InspirePreviewProps {
  result: InspireResult;
  editedLyrics: string;
  editedCaption: string;
  onLyricsChange: (lyrics: string) => void;
  onCaptionChange: (caption: string) => void;
  onGenerate: () => void;
  onBack: () => void;
  onRefine?: () => void;
  isGenerating: boolean;
}

export const InspirePreview: React.FC<InspirePreviewProps> = ({
  result,
  editedLyrics,
  editedCaption,
  onLyricsChange,
  onCaptionChange,
  onGenerate,
  onBack,
  onRefine,
  isGenerating,
}) => {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 pt-3 pb-2">
        <button
          onClick={onBack}
          disabled={isGenerating}
          className="flex items-center gap-1.5 text-sm text-zinc-400 hover:text-white transition-colors disabled:opacity-40"
        >
          <ArrowLeft size={16} />
          {t('instaGen.preview.back')}
        </button>
        <h2 className="text-sm font-semibold text-zinc-300">
          {t('instaGen.preview.title')}
        </h2>
      </div>

      {/* Metadata badges */}
      <div className="px-4 py-2 flex flex-wrap gap-2">
        <MetaBadge label="BPM" value={String(result.bpm)} color="amber" />
        <MetaBadge label="Key" value={result.keyScale} color="emerald" />
        <MetaBadge label="Time" value={`${result.timeSignature}/4`} color="sky" />
        <MetaBadge label="Duration" value={`${result.duration}s`} color="purple" />
      </div>

      {/* Caption editor */}
      <div className="px-4 py-2">
        <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1">
          {t('instaGen.preview.editCaption')}
        </label>
        <textarea
          value={editedCaption}
          onChange={(e) => onCaptionChange(e.target.value)}
          disabled={isGenerating}
          rows={2}
          className="w-full rounded-lg border border-zinc-300 dark:border-white/10 bg-zinc-50 dark:bg-white/5 px-3 py-2 text-sm text-zinc-900 dark:text-white placeholder:text-zinc-400 resize-none outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 transition-all disabled:opacity-50"
        />
      </div>

      {/* Lyrics editor */}
      <div className="flex-1 px-4 py-2 min-h-0 flex flex-col">
        <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1">
          {t('instaGen.preview.lyrics')}
        </label>
        <textarea
          value={editedLyrics}
          onChange={(e) => onLyricsChange(e.target.value)}
          disabled={isGenerating}
          className="flex-1 w-full rounded-lg border border-zinc-300 dark:border-white/10 bg-zinc-50 dark:bg-white/5 px-3 py-2 text-sm text-zinc-900 dark:text-white font-mono leading-relaxed placeholder:text-zinc-400 resize-none outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 transition-all disabled:opacity-50"
          placeholder="[Verse 1]&#10;..."
        />
      </div>

      {/* Action buttons */}
      <div className="px-4 py-3 space-y-2">
        <button
          onClick={onGenerate}
          disabled={isGenerating}
          className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-pink-600 to-violet-600 hover:from-pink-500 hover:to-violet-500 shadow-lg shadow-pink-500/20 hover:shadow-pink-500/30 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          <Music size={16} />
          {isGenerating ? t('instaGen.inspireLoading') : t('instaGen.preview.generate')}
        </button>
        {onRefine && (
          <button
            onClick={onRefine}
            disabled={isGenerating}
            className="w-full py-2.5 rounded-xl text-sm font-medium text-zinc-300 border border-white/10 bg-white/5 hover:bg-white/10 hover:text-white transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <Pencil size={14} />
            {t('instaGen.preview.refine')}
          </button>
        )}
      </div>
    </div>
  );
};

/** Coloured metadata badge */
const MetaBadge: React.FC<{ label: string; value: string; color: string }> = ({ label, value, color }) => {
  const colorClasses: Record<string, string> = {
    amber: 'bg-amber-500/15 text-amber-400 border-amber-500/20',
    emerald: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20',
    sky: 'bg-sky-500/15 text-sky-400 border-sky-500/20',
    purple: 'bg-purple-500/15 text-purple-400 border-purple-500/20',
  };
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border ${colorClasses[color] || colorClasses.amber}`}>
      <span className="text-zinc-500 dark:text-zinc-400">{label}:</span>
      {value}
    </span>
  );
};
