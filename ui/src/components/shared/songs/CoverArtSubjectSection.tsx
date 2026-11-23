// CoverArtSubjectSection.tsx — Optional cover art prompt override accordion
//
// Appears in the left-hand column of CreatePanel and InstaGenPanel.
// Only visible when cover art is enabled in post-processing settings.
// Allows users to override the auto-generated image subject while keeping
// the art direction (genre visuals, quality modifiers) automated.

import React from 'react';
import { Image, RotateCcw, ChevronDown } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useShallow } from 'zustand/react/shallow';
import { useGlobalParamsStore } from '@/stores/globalParamsStore';
import { usePersistedState } from '@/hooks/usePersistedState';

export const CoverArtSubjectSection: React.FC = () => {
  const { t } = useTranslation();
  const gp = useGlobalParamsStore(useShallow(s => ({
    coverArtEnabled: s.coverArtEnabled, coverArtSubject: s.coverArtSubject,
    setCoverArtSubject: s.setCoverArtSubject,
  })));
  const [open, setOpen] = usePersistedState('hs-coverArtSubjectOpen', false);

  // Only render when cover art is enabled
  if (!gp.coverArtEnabled) return null;

  return (
    <div className="rounded-xl border border-pink-500/15 bg-pink-500/[0.03] overflow-hidden">
      {/* Header */}
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-pink-500/5 transition-colors"
      >
        <Image size={14} className="text-pink-400 flex-shrink-0" />
        <span className="text-sm text-zinc-300 font-medium flex-1">
          {t('coverArt.subjectTitle')}
        </span>
        {gp.coverArtSubject && (
          <span className="text-[10px] text-pink-400/60 font-mono flex-shrink-0">
            Custom
          </span>
        )}
        <ChevronDown
          size={14}
          className={`text-zinc-500 transition-transform flex-shrink-0 ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {/* Content */}
      {open && (
        <div className="px-3 pb-3 pt-1 border-t border-pink-500/10 space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-[11px] text-zinc-500">
              {t('coverArt.subjectLabel')}
            </label>
            {gp.coverArtSubject && (
              <button
                type="button"
                onClick={() => gp.setCoverArtSubject('')}
                className="flex items-center gap-1 text-[10px] text-zinc-500 hover:text-pink-400 transition-colors"
                title={t('coverArt.subjectReset')}
              >
                <RotateCcw size={10} />
                {t('coverArt.subjectReset')}
              </button>
            )}
          </div>
          <textarea
            className="w-full px-3 py-2 rounded-lg bg-zinc-900/50 border border-white/10
                       text-sm text-zinc-200 placeholder-zinc-600
                       focus:border-pink-500/30 focus:ring-1 focus:ring-pink-500/20
                       outline-none transition-colors resize-none"
            rows={3}
            value={gp.coverArtSubject}
            onChange={e => gp.setCoverArtSubject(e.target.value)}
            placeholder={t('coverArt.subjectPlaceholder')}
          />
          <p className="text-[10px] text-zinc-600 leading-relaxed">
            {t('coverArt.subjectHelp')}
          </p>
        </div>
      )}
    </div>
  );
};
