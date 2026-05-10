// InstaGenPanel.tsx — Main Insta-Gen studio container
//
// Genre-first music generation with optional lyrics preview.
// State machine: Input → (optional) Preview → Generate
//
// "Preview Lyrics" toggle controls the flow:
//   ON:  genres → subject → Inspire → preview/edit → Generate
//   OFF: genres → subject → Generate (engine does everything)

import React, { useState, useMemo, useCallback } from 'react';
import { Wand2, Sparkles, Music, Eye, EyeOff } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../../context/AuthContext';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { usePersistedState } from '../../hooks/usePersistedState';
import { GenreSelector } from './GenreSelector';
import { InspirePreview } from './InspirePreview';
import { runInspireAndWait, type InspireResult } from '../../services/inspireApi';
import type { GenerationParams } from '../../types';

// Vocal language options (subset matching the engine's VALID_LANGUAGES)
const LANGUAGES = [
  { value: 'en', label: 'English' },
  { value: 'zh', label: '中文 (Chinese)' },
  { value: 'ja', label: '日本語 (Japanese)' },
  { value: 'ko', label: '한국어 (Korean)' },
  { value: 'es', label: 'Español' },
  { value: 'fr', label: 'Français' },
  { value: 'de', label: 'Deutsch' },
  { value: 'it', label: 'Italiano' },
  { value: 'pt', label: 'Português' },
  { value: 'ru', label: 'Русский' },
  { value: 'ar', label: 'العربية' },
  { value: 'hi', label: 'हिन्दी' },
  { value: 'tr', label: 'Türkçe' },
  { value: 'vi', label: 'Tiếng Việt' },
  { value: 'th', label: 'ไทย' },
  { value: 'sv', label: 'Svenska' },
  { value: 'pl', label: 'Polski' },
  { value: 'nl', label: 'Nederlands' },
];

type Phase = 'input' | 'inspiring' | 'preview' | 'generating';

interface InstaGenPanelProps {
  onGenerate: (params: Partial<GenerationParams>) => void;
  activeJobCount: number;
}

export const InstaGenPanel: React.FC<InstaGenPanelProps> = ({ onGenerate, activeJobCount }) => {
  const { t } = useTranslation();
  const { token } = useAuth();
  const globalParams = useGlobalParams();

  // ── Persisted state ──
  const [previewEnabled, setPreviewEnabled] = usePersistedState('hs-instagen-preview', true);
  const [selectedGenres, setSelectedGenres] = usePersistedState<string[]>('hs-instagen-genres', []);
  const [vocalLanguage, setVocalLanguage] = usePersistedState('hs-instagen-language', 'en');

  // ── Ephemeral state ──
  const [subject, setSubject] = usePersistedState('hs-instagen-subject', '');
  const [additionalCaption, setAdditionalCaption] = useState('');
  const [phase, setPhase] = useState<Phase>('input');
  const [inspireResult, setInspireResult] = useState<InspireResult | null>(null);
  const [editedLyrics, setEditedLyrics] = useState('');
  const [editedCaption, setEditedCaption] = useState('');
  const [inspireProgress, setInspireProgress] = useState('');
  const [error, setError] = useState('');

  // ── Computed caption ──
  const computedCaption = useMemo(() => {
    const parts: string[] = [];
    if (selectedGenres.length > 0) {
      parts.push(selectedGenres.join(', '));
    }
    if (additionalCaption.trim()) {
      parts.push(additionalCaption.trim());
    }
    return parts.join(', ');
  }, [selectedGenres, additionalCaption]);

  // ── Validation ──
  const canSubmit = selectedGenres.length > 0 || additionalCaption.trim().length > 0;

  // ── Build generation params ──
  const buildParams = useCallback((lyrics: string, caption: string): Partial<GenerationParams> => ({
    caption: caption || computedCaption,
    lyrics,
    instrumental: false,
    subject: subject.trim() || undefined,
    vocalLanguage,
    source: 'insta-gen',
    useCotCaption: true,
  }), [computedCaption, subject, vocalLanguage]);

  // ── Inspire flow (preview ON) ──
  const handleInspire = useCallback(async () => {
    if (!canSubmit) return;
    setError('');
    setPhase('inspiring');
    setInspireProgress('Starting...');

    try {
      const inspireCaption = subject.trim()
        ? `${computedCaption}. Song about: ${subject.trim()}`
        : computedCaption;
      console.log('[InstaGen] Inspire request:', { inspireCaption, subject, computedCaption, vocalLanguage, lmModel: globalParams.lmModel });
      const result = await runInspireAndWait(
        {
          caption: inspireCaption,
          subject: subject.trim() || undefined,
          vocalLanguage,
          useCotCaption: true,
          // Global LM params — ensures the correct model + sampling settings
          lmModel: globalParams.lmModel || undefined,
          lmTemperature: globalParams.lmTemperature,
          lmCfgScale: globalParams.lmCfgScale,
          lmTopP: globalParams.lmTopP,
        },
        token || undefined,
        (stage, _progress) => setInspireProgress(stage),
      );

      setInspireResult(result);
      setEditedLyrics(result.lyrics);
      setEditedCaption(result.caption);
      setPhase('preview');
    } catch (err: any) {
      setError(err.message || 'Inspire failed');
      setPhase('input');
    }
  }, [canSubmit, computedCaption, subject, vocalLanguage, globalParams, token]);

  // ── Generate from preview ──
  const handleGenerateFromPreview = useCallback(() => {
    if (!inspireResult) return;
    const params = buildParams(editedLyrics, editedCaption);
    // Include metadata from inspire result
    params.bpm = inspireResult.bpm;
    params.duration = inspireResult.duration;
    params.keyScale = inspireResult.keyScale;
    params.timeSignature = inspireResult.timeSignature;
    onGenerate(params);
    // Return to input after queuing
    setPhase('input');
    setInspireResult(null);
  }, [inspireResult, editedLyrics, editedCaption, buildParams, onGenerate]);

  // ── Direct generate (preview OFF) ──
  const handleDirectGenerate = useCallback(() => {
    if (!canSubmit) return;
    const params = buildParams('', computedCaption);
    onGenerate(params);
  }, [canSubmit, computedCaption, buildParams, onGenerate]);

  // ── Back to input from preview ──
  const handleBack = useCallback(() => {
    setPhase('input');
  }, []);

  // ── Render ──
  if (phase === 'preview' && inspireResult) {
    return (
      <div className="h-full flex flex-col bg-white dark:bg-suno overflow-hidden">
        <InspirePreview
          result={inspireResult}
          editedLyrics={editedLyrics}
          editedCaption={editedCaption}
          onLyricsChange={setEditedLyrics}
          onCaptionChange={setEditedCaption}
          onGenerate={handleGenerateFromPreview}
          onBack={handleBack}
          isGenerating={false}
        />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-white dark:bg-suno overflow-y-auto">
      {/* Header */}
      <div className="px-4 pt-4 pb-2 flex-shrink-0">
        <div className="flex items-center gap-2.5 mb-1">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-pink-500 to-violet-600 flex items-center justify-center shadow-lg shadow-pink-500/20">
            <Wand2 size={16} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-zinc-900 dark:text-white leading-tight">
              {t('instaGen.title')}
            </h1>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              {t('instaGen.subtitle')}
            </p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 px-4 space-y-4 pb-4">
        {/* Genre Selector */}
        <GenreSelector selected={selectedGenres} onChange={setSelectedGenres} />

        {/* Subject */}
        <div>
          <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1.5">
            {t('instaGen.subjectLabel')}
          </label>
          <input
            type="text"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            placeholder={t('instaGen.subjectPlaceholder')}
            className="w-full rounded-xl border border-zinc-300 dark:border-white/10 bg-zinc-50 dark:bg-white/5 px-3 py-2.5 text-sm text-zinc-900 dark:text-white placeholder:text-zinc-400 dark:placeholder:text-zinc-500 outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 transition-all"
          />
        </div>

        {/* Additional caption */}
        <div>
          <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1.5">
            {t('instaGen.captionAdditional')}
          </label>
          <input
            type="text"
            value={additionalCaption}
            onChange={(e) => setAdditionalCaption(e.target.value)}
            placeholder="e.g. female vocals, melancholic, reverb-heavy guitar"
            className="w-full rounded-xl border border-zinc-300 dark:border-white/10 bg-zinc-50 dark:bg-white/5 px-3 py-2.5 text-sm text-zinc-900 dark:text-white placeholder:text-zinc-400 dark:placeholder:text-zinc-500 outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 transition-all"
          />
        </div>

        {/* Language selector */}
        <div>
          <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1.5">
            {t('instaGen.languageLabel')}
          </label>
          <select
            value={vocalLanguage}
            onChange={(e) => setVocalLanguage(e.target.value)}
            className="w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors cursor-pointer"
          >
            {LANGUAGES.map(lang => (
              <option key={lang.value} value={lang.value}>{lang.label}</option>
            ))}
          </select>
        </div>

        {/* Caption preview */}
        {computedCaption && (
          <div>
            <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1.5">
              {t('instaGen.captionLabel')}
            </label>
            <div className="w-full rounded-xl border border-zinc-300 dark:border-white/10 bg-zinc-100 dark:bg-white/[0.03] px-3 py-2.5 text-sm text-zinc-700 dark:text-zinc-300 italic">
              {computedCaption}
            </div>
          </div>
        )}

        {/* Preview toggle */}
        <div className="flex items-center justify-between py-2">
          <div className="flex items-center gap-2">
            {previewEnabled ? <Eye size={14} className="text-violet-400" /> : <EyeOff size={14} className="text-zinc-400" />}
            <span className="text-sm text-zinc-700 dark:text-zinc-300">
              {t('instaGen.previewToggle')}
            </span>
          </div>
          <button
            onClick={() => setPreviewEnabled(!previewEnabled)}
            className={`
              relative w-10 h-5 rounded-full transition-colors duration-200
              ${previewEnabled ? 'bg-violet-500' : 'bg-zinc-300 dark:bg-zinc-600'}
            `}
          >
            <div className={`
              absolute top-0.5 w-4 h-4 rounded-full bg-white shadow-sm transition-transform duration-200
              ${previewEnabled ? 'translate-x-5' : 'translate-x-0.5'}
            `} />
          </button>
        </div>
        {previewEnabled && (
          <p className="text-xs text-zinc-400 dark:text-zinc-500 -mt-2">
            {t('instaGen.previewToggleHint')}
          </p>
        )}

        {/* Error display */}
        {error && (
          <div className="rounded-xl bg-red-500/10 border border-red-500/20 px-3 py-2 text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Action button */}
        {phase === 'inspiring' ? (
          <div className="space-y-2">
            <button
              disabled
              className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-violet-600 to-pink-600 opacity-80 flex items-center justify-center gap-2"
            >
              <div className="w-4 h-4 rounded-full border-2 border-white/30 border-t-white animate-spin" />
              {inspireProgress || t('instaGen.inspireLoading')}
            </button>
          </div>
        ) : previewEnabled ? (
          <button
            onClick={handleInspire}
            disabled={!canSubmit || activeJobCount > 0}
            className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-violet-600 to-pink-600 hover:from-violet-500 hover:to-pink-500 shadow-lg shadow-violet-500/20 hover:shadow-violet-500/30 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <Sparkles size={16} />
            {t('instaGen.inspire')}
          </button>
        ) : (
          <button
            onClick={handleDirectGenerate}
            disabled={!canSubmit || activeJobCount > 0}
            className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-pink-600 to-violet-600 hover:from-pink-500 hover:to-violet-500 shadow-lg shadow-pink-500/20 hover:shadow-pink-500/30 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <Music size={16} />
            {t('instaGen.generate')}
          </button>
        )}
      </div>
    </div>
  );
};
