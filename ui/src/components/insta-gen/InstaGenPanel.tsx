// InstaGenPanel.tsx — Main Insta-Gen studio container
//
// Genre-first music generation with three lyric modes:
//   Instrumental:  no lyrics at all
//   Lyrics:        built-in ACE-Step LM generates lyrics (random topic)
//   Lyrics + AI:   external LLM generates subject-aware lyrics
//
// State machine: Input → (optional) Preview → Generate

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Wand2, Sparkles, Music, Eye, EyeOff, Mic, MicOff, Bot, Brain } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../../context/AuthContext';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { usePersistedState } from '../../hooks/usePersistedState';
import { GenreSelector } from './GenreSelector';
import { InspirePreview } from './InspirePreview';
import {
  runInspireAndWait,
  runLlmInspire,
  fetchInspireProviders,
  generateRandomSubject,
  type InspireResult,
  type InspireProvider,
} from '../../services/inspireApi';
import { generateApi, songApi } from '../../services/api';
import {
  addManualQueueItem,
  updateManualQueueItem,
  completeManualQueueItem,
  failManualQueueItem,
} from '../../stores/audioGenQueueStore';
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

type LyricMode = 'instrumental' | 'lyrics' | 'lyrics-ai';
type Phase = 'input' | 'inspiring' | 'preview' | 'generating';

interface InstaGenPanelProps {
  onGenerate: (params: Partial<GenerationParams>) => void;
  onSongCreated?: (song: any) => void;
  activeJobCount: number;
}

/** Derive a song title from lyrics. Prefers [Chorus] first line, then [Verse 1], then first lyric line. */
function deriveTitleFromLyrics(lyrics: string): string {
  if (!lyrics || lyrics === '[Instrumental]') return '';

  const lines = lyrics.split(/\r?\n/);
  const sectionRe = /^\s*\[(.+?)\]\s*$/;

  // Build a map of section → first meaningful lyric line
  const sections: Record<string, string> = {};
  let currentSection = '';
  for (const line of lines) {
    const m = line.match(sectionRe);
    if (m) {
      currentSection = m[1].trim().toLowerCase();
      continue;
    }
    const trimmed = line.trim();
    // Skip empty lines, parenthetical backing vocals, and "[Instrumental]" markers
    if (!trimmed || trimmed.startsWith('(') || trimmed.toLowerCase() === '[instrumental]') continue;
    if (currentSection && !sections[currentSection]) {
      sections[currentSection] = trimmed;
    }
  }

  // Priority: chorus → verse 1 → verse → first any section
  const chorusKey = Object.keys(sections).find(k => k.startsWith('chorus'));
  if (chorusKey) return cleanTitle(sections[chorusKey]);

  const verse1Key = Object.keys(sections).find(k => k === 'verse 1');
  if (verse1Key) return cleanTitle(sections[verse1Key]);

  const verseKey = Object.keys(sections).find(k => k.startsWith('verse'));
  if (verseKey) return cleanTitle(sections[verseKey]);

  // Fallback: first value in any section
  const firstVal = Object.values(sections)[0];
  return firstVal ? cleanTitle(firstVal) : '';
}

/** Clean up a lyric line for use as a title */
function cleanTitle(line: string): string {
  // Remove trailing punctuation, parenthetical asides, and limit length
  let t = line.replace(/\s*\(.*?\)\s*/g, '').trim();
  t = t.replace(/[,.!?;:]+$/, '').trim();
  if (t.length > 60) t = t.substring(0, 57) + '...';
  return t;
}

// ── Module-level serial queue ──
// Ensures InstaGen jobs run one at a time (inspire → generate → poll → next).
// Without this, concurrent inspire calls stomp on each other's engine logs.
const _instaQueue: Array<() => Promise<void>> = [];
let _instaRunning = false;
function enqueueInstaJob(fn: () => Promise<void>) {
  _instaQueue.push(fn);
  if (!_instaRunning) _drainInstaQueue();
}
async function _drainInstaQueue() {
  _instaRunning = true;
  while (_instaQueue.length > 0) {
    const job = _instaQueue.shift()!;
    await job();
  }
  _instaRunning = false;
}

export const InstaGenPanel: React.FC<InstaGenPanelProps> = ({ onGenerate, onSongCreated, activeJobCount }) => {
  const { t } = useTranslation();
  const { token } = useAuth();
  const globalParams = useGlobalParams();

  // ── Persisted state ──
  const [previewEnabled, setPreviewEnabled] = usePersistedState('hs-instagen-preview', true);
  const [selectedGenres, setSelectedGenres] = usePersistedState<string[]>('hs-instagen-genres', []);
  const [vocalLanguage, setVocalLanguage] = usePersistedState('hs-instagen-language', 'en');
  const [lyricMode, setLyricMode] = usePersistedState<LyricMode>('hs-instagen-lyricmode', 'lyrics');
  const [subject, setSubject] = usePersistedState('hs-instagen-subject', '');
  const [selectedProvider, setSelectedProvider] = usePersistedState('hs-instagen-llm-provider', '');
  const [selectedModel, setSelectedModel] = usePersistedState('hs-instagen-llm-model', '');
  const [thinking, setThinking] = usePersistedState('hs-instagen-thinking', true);

  // ── Ephemeral state ──
  const [additionalCaption, setAdditionalCaption] = useState('');
  const [phase, setPhase] = useState<Phase>('input');
  const [inspireResult, setInspireResult] = useState<InspireResult | null>(null);
  const [editedLyrics, setEditedLyrics] = useState('');
  const [editedCaption, setEditedCaption] = useState('');
  const [inspireProgress, setInspireProgress] = useState('');
  const [error, setError] = useState('');
  const [providers, setProviders] = useState<InspireProvider[]>([]);
  const [providersLoaded, setProvidersLoaded] = useState(false);
  const [randomSubjectLoading, setRandomSubjectLoading] = useState(false);

  // ── Load LLM providers on mount ──
  useEffect(() => {
    if (providersLoaded) return;
    fetchInspireProviders()
      .then((list) => {
        setProviders(list.filter(p => p.available));
        setProvidersLoaded(true);
        // Auto-select first available provider if none persisted
        if (!selectedProvider && list.length > 0) {
          const first = list.find(p => p.available);
          if (first) {
            setSelectedProvider(first.id);
            setSelectedModel(first.default_model);
          }
        }
      })
      .catch(() => setProvidersLoaded(true));
  }, [providersLoaded, selectedProvider, setSelectedProvider, setSelectedModel]);

  // ── Current provider info ──
  const currentProvider = useMemo(
    () => providers.find(p => p.id === selectedProvider),
    [providers, selectedProvider]
  );

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
  const canSubmit = useMemo(() => {
    if (selectedGenres.length === 0 && !additionalCaption.trim()) return false;
    if (lyricMode === 'lyrics-ai' && !subject.trim()) return false;
    if (lyricMode === 'lyrics-ai' && !selectedProvider) return false;
    return true;
  }, [selectedGenres, additionalCaption, lyricMode, subject, selectedProvider]);

  // ── Build generation params ──
  const buildParams = useCallback((lyrics: string, caption: string): Partial<GenerationParams> => ({
    caption: caption || computedCaption,
    lyrics,
    instrumental: lyricMode === 'instrumental',
    vocalLanguage: lyricMode === 'instrumental' ? undefined : vocalLanguage,
    source: 'insta-gen',
    useCotCaption: thinking,
    skipLm: !thinking, // Thinking ON = LM runs (audio codes + CoT); OFF = skip LM (faster)
  }), [computedCaption, lyricMode, vocalLanguage, thinking]);

  // ── Random subject via LLM ──
  const handleRandomSubject = useCallback(async () => {
    if (!selectedProvider || randomSubjectLoading) return;
    setRandomSubjectLoading(true);
    try {
      const result = await generateRandomSubject(
        { provider: selectedProvider, model: selectedModel || undefined, genres: selectedGenres },
        token || undefined,
      );
      if (result) setSubject(result);
    } catch (err: any) {
      console.error('[InstaGen] Random subject failed:', err.message);
    } finally {
      setRandomSubjectLoading(false);
    }
  }, [selectedProvider, selectedModel, selectedGenres, token, randomSubjectLoading, setSubject]);

  // ── Inspire flow (preview ON) ──
  const handleInspire = useCallback(async () => {
    if (!canSubmit) return;
    setError('');
    setPhase('inspiring');

    try {
      if (lyricMode === 'lyrics-ai') {
        // ── External LLM path ──
        setInspireProgress('Generating lyrics via AI...');
        const llmResult = await runLlmInspire(
          {
            provider: selectedProvider,
            model: selectedModel || undefined,
            genres: selectedGenres,
            subject: subject.trim(),
            language: vocalLanguage,
          },
          token || undefined,
        );

        // Run inspire with the LLM lyrics to get metadata (bpm, duration, key, timesig)
        setInspireProgress('Resolving song metadata...');
        const metaResult = await runInspireAndWait(
          {
            caption: llmResult.caption || computedCaption,
            lyrics: llmResult.lyrics,
            vocalLanguage,
            useCotCaption: thinking,
            lmModel: globalParams.lmModel || undefined,
            lmTemperature: globalParams.lmTemperature,
            lmCfgScale: globalParams.lmCfgScale,
            lmTopP: globalParams.lmTopP,
          },
          token || undefined,
          (stage, _progress) => setInspireProgress(stage),
        );

        const result: InspireResult = {
          caption: llmResult.caption || computedCaption,
          lyrics: llmResult.lyrics,  // Keep LLM lyrics, not inspire's
          title: llmResult.title,
          bpm: metaResult.bpm,
          duration: metaResult.duration,
          keyScale: metaResult.keyScale,
          timeSignature: metaResult.timeSignature,
          vocalLanguage,
        };

        setInspireResult(result);
        setEditedLyrics(result.lyrics);
        setEditedCaption(result.caption);
        setPhase('preview');

      } else {
        // ── Built-in LM path ──
        setInspireProgress('Starting...');
        const result = await runInspireAndWait(
          {
            caption: computedCaption,
            vocalLanguage,
            useCotCaption: thinking,
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
      }
    } catch (err: any) {
      setError(err.message || 'Inspire failed');
      setPhase('input');
    }
  }, [canSubmit, lyricMode, selectedProvider, selectedModel, selectedGenres, subject, vocalLanguage, computedCaption, globalParams, token]);

  // ── Generate from preview ──
  const handleGenerateFromPreview = useCallback(() => {
    if (!inspireResult) return;
    const params = buildParams(editedLyrics, editedCaption);
    // Prefer LLM-generated title, then derive from lyrics, then caption
    params.title = inspireResult.title || deriveTitleFromLyrics(editedLyrics) || computedCaption;
    // Include metadata from inspire result
    if (inspireResult.bpm) params.bpm = inspireResult.bpm;
    if (inspireResult.duration) params.duration = inspireResult.duration;
    if (inspireResult.keyScale) params.keyScale = inspireResult.keyScale;
    if (inspireResult.timeSignature) params.timeSignature = inspireResult.timeSignature;
    onGenerate(params);
    // Return to input after queuing
    setPhase('input');
    setInspireResult(null);
  }, [inspireResult, editedLyrics, editedCaption, computedCaption, buildParams, onGenerate]);

  // ── Direct generate (preview OFF) ──
  // Non-blocking: creates a queue item immediately, runs inspire + generate
  // in the background. User can queue more items without waiting.
  const handleDirectGenerate = useCallback(() => {
    if (!canSubmit || !token) return;

    // Capture current state for async closure
    const capturedCaption = computedCaption;
    const capturedLyricMode = lyricMode;
    const capturedThinking = thinking;
    const capturedVocalLang = vocalLanguage;
    const capturedProvider = selectedProvider;
    const capturedModel = selectedModel;
    const capturedGenres = [...selectedGenres];
    const capturedSubject = subject.trim();
    const capturedGlobalParams = { ...globalParams };
    const capturedToken = token;

    // Create queue item immediately — user sees it right away
    const queueId = addManualQueueItem({
      title: capturedCaption || 'Insta-Gen',
      caption: capturedCaption,
    });
    updateManualQueueItem(queueId, {
      stage: _instaRunning ? 'Queued…' : (capturedLyricMode === 'lyrics' ? 'Generating lyrics…' : 'Preparing…'),
    });

    // Run the full pipeline via serial queue (one at a time)
    enqueueInstaJob(async () => {
      try {
        // Step 1: Resolve lyrics
        let resolvedLyrics = '';
        let resolvedCaption = capturedCaption;
        let llmTitle = '';

        if (capturedLyricMode === 'instrumental') {
          resolvedLyrics = '[Instrumental]';
        } else if (capturedLyricMode === 'lyrics-ai') {
          updateManualQueueItem(queueId, { stage: 'Generating lyrics via AI…' });
          const llmResult = await runLlmInspire(
            {
              provider: capturedProvider,
              model: capturedModel || undefined,
              genres: capturedGenres,
              subject: capturedSubject,
              language: capturedVocalLang,
            },
            capturedToken,
          );
          resolvedLyrics = llmResult.lyrics;
          resolvedCaption = llmResult.caption || capturedCaption;
          llmTitle = llmResult.title || '';
        }

        // Step 2: Run inspire for metadata (+ lyrics if not resolved)
        updateManualQueueItem(queueId, {
          stage: capturedLyricMode === 'lyrics' ? 'Generating lyrics…' : 'Resolving metadata…',
        });
        const inspireParams: any = {
          caption: resolvedCaption,
          vocalLanguage: capturedVocalLang,
          useCotCaption: capturedThinking,
          lmModel: capturedGlobalParams.lmModel || undefined,
          lmTemperature: capturedGlobalParams.lmTemperature,
          lmCfgScale: capturedGlobalParams.lmCfgScale,
          lmTopP: capturedGlobalParams.lmTopP,
        };
        if (resolvedLyrics) inspireParams.lyrics = resolvedLyrics;
        if (capturedLyricMode === 'instrumental') inspireParams.instrumental = true;

        const inspireResult = await runInspireAndWait(
          inspireParams,
          capturedToken,
          (stage) => updateManualQueueItem(queueId, { stage }),
        );

        // Step 3: Build generation params
        const finalLyrics = resolvedLyrics || inspireResult.lyrics;
        const params = buildParams(finalLyrics, resolvedCaption);
        params.title = llmTitle || deriveTitleFromLyrics(finalLyrics) || resolvedCaption;
        if (inspireResult.bpm) params.bpm = inspireResult.bpm;
        if (inspireResult.duration) params.duration = inspireResult.duration;
        if (inspireResult.keyScale) params.keyScale = inspireResult.keyScale;
        if (inspireResult.timeSignature) params.timeSignature = inspireResult.timeSignature;

        // Update queue item title now we have lyrics
        updateManualQueueItem(queueId, {
          title: params.title || resolvedCaption,
          stage: 'Submitting to engine…',
        });

        // Step 4: Merge with global engine params and submit
        const engineParams = globalParams.getGlobalParams();
        const enrichedParams = {
          ...engineParams,
          ...params,
          source: 'insta-gen',
          coResident: ((): boolean => {
            try { return JSON.parse(localStorage.getItem('ace-settings') || '{}').coResident; }
            catch { return false; }
          })(),
          cacheLmCodes: ((): boolean => {
            try { return JSON.parse(localStorage.getItem('ace-settings') || '{}').cacheLmCodes; }
            catch { return true; }
          })(),
        };

        const res = await generateApi.submit(enrichedParams as any, capturedToken);
        updateManualQueueItem(queueId, { jobId: res.jobId, stage: 'Generating audio…' });

        // Step 5: Poll until done
        const startTime = Date.now();
        while (true) {
          await new Promise(r => setTimeout(r, 1500));
          const status = await generateApi.status(res.jobId);
          const progress = status.progress !== undefined
            ? Math.min(100, Math.max(0, (status.progress > 1 ? status.progress / 100 : status.progress) * 100))
            : undefined;
          updateManualQueueItem(queueId, {
            progress,
            stage: status.stage || 'Generating…',
            elapsed: Math.round((Date.now() - startTime) / 1000),
          });

          if (status.status === 'succeeded') {
            const audioUrl = status.result?.audioUrls?.[0] || '';
            const songId = status.result?.songIds?.[0];
            completeManualQueueItem(queueId, {
              audioUrl,
              songId,
              masteredAudioUrl: status.result?.masteredAudioUrl,
              audioDuration: status.result?.duration,
            });
            // Notify App to refresh library
            if (songId) {
              try {
                const { song } = await songApi.get(songId);
                onSongCreated?.(song);
              } catch { /* non-fatal */ }
            }
            return;
          }
          if (status.status === 'failed' || status.status === 'cancelled') {
            throw new Error(status.error || 'Generation failed');
          }
          if (Date.now() - startTime > 30 * 60 * 1000) {
            throw new Error('Generation timed out');
          }
        }
      } catch (err: any) {
        failManualQueueItem(queueId, err.message || 'Generation failed');
      }
    });
  }, [canSubmit, token, lyricMode, selectedProvider, selectedModel, selectedGenres, subject, vocalLanguage, computedCaption, thinking, buildParams, globalParams, onSongCreated]);

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

        {/* ── Lyric Mode Selector (3-way segmented control) ── */}
        <div>
          <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1.5">
            Vocal Mode
          </label>
          <div className="flex rounded-xl overflow-hidden border border-zinc-300 dark:border-white/10">
            {([
              { value: 'instrumental' as LyricMode, label: 'Instrumental', icon: MicOff },
              { value: 'lyrics' as LyricMode, label: 'Lyrics', icon: Mic },
              { value: 'lyrics-ai' as LyricMode, label: 'Lyrics + AI', icon: Bot },
            ]).map(({ value, label, icon: Icon }) => (
              <button
                key={value}
                onClick={() => setLyricMode(value)}
                className={`
                  flex-1 flex items-center justify-center gap-1.5 py-2 text-xs font-medium transition-all duration-200
                  ${lyricMode === value
                    ? 'bg-gradient-to-r from-violet-600 to-pink-600 text-white shadow-inner'
                    : 'bg-zinc-50 dark:bg-white/5 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-white/10'
                  }
                `}
              >
                <Icon size={13} />
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* ── Subject (only for Lyrics + AI) ── */}
        {lyricMode === 'lyrics-ai' && (
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
                Song Subject <span className="text-pink-500">*</span>
              </label>
              <button
                onClick={handleRandomSubject}
                disabled={randomSubjectLoading || !selectedProvider}
                className="text-xs text-zinc-400 hover:text-violet-400 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"
              >
                {randomSubjectLoading && (
                  <div className="w-3 h-3 rounded-full border border-zinc-400 border-t-violet-400 animate-spin" />
                )}
                Random
              </button>
            </div>
            <input
              type="text"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              placeholder="e.g. a man tired from a life of working 9 to 5"
              className="w-full rounded-xl border border-zinc-300 dark:border-white/10 bg-zinc-50 dark:bg-white/5 px-3 py-2.5 text-sm text-zinc-900 dark:text-white placeholder:text-zinc-400 dark:placeholder:text-zinc-500 outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 transition-all"
            />
          </div>
        )}

        {/* ── LLM Provider & Model (only for Lyrics + AI) ── */}
        {lyricMode === 'lyrics-ai' && (
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1.5">
                LLM Provider
              </label>
              <select
                value={selectedProvider}
                onChange={(e) => {
                  setSelectedProvider(e.target.value);
                  const prov = providers.find(p => p.id === e.target.value);
                  if (prov) setSelectedModel(prov.default_model);
                }}
                className="w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors cursor-pointer"
              >
                {providers.length === 0 && (
                  <option value="">No providers available</option>
                )}
                {providers.map(p => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1.5">
                Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors cursor-pointer"
              >
                {(currentProvider?.models || []).map(m => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
          </div>
        )}

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

        {/* Language selector (hidden for instrumental) */}
        {lyricMode !== 'instrumental' && (
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
        )}

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

        {/* Thinking toggle */}
        <div className="flex items-center justify-between py-2">
          <div className="flex items-center gap-2">
            <Brain size={14} className={thinking ? 'text-amber-400' : 'text-zinc-400'} />
            <span className="text-sm text-zinc-700 dark:text-zinc-300">Thinking</span>
            <span className="text-[10px] text-zinc-400 dark:text-zinc-500">(CoT)</span>
          </div>
          <button
            onClick={() => setThinking(!thinking)}
            className={`
              relative w-10 h-5 rounded-full transition-colors duration-200
              ${thinking ? 'bg-amber-500' : 'bg-zinc-300 dark:bg-zinc-600'}
            `}
          >
            <div className={`
              absolute top-0.5 w-4 h-4 rounded-full bg-white shadow-sm transition-transform duration-200
              ${thinking ? 'translate-x-5' : 'translate-x-0.5'}
            `} />
          </button>
        </div>

        {/* Preview toggle (hidden for instrumental) */}
        {lyricMode !== 'instrumental' && (
          <>
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
          </>
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
        ) : lyricMode !== 'instrumental' && previewEnabled ? (
          <button
            onClick={handleInspire}
            disabled={!canSubmit}
            className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-violet-600 to-pink-600 hover:from-violet-500 hover:to-pink-500 shadow-lg shadow-violet-500/20 hover:shadow-violet-500/30 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {lyricMode === 'lyrics-ai' ? <Bot size={16} /> : <Sparkles size={16} />}
            {lyricMode === 'lyrics-ai' ? 'Generate Lyrics' : t('instaGen.inspire')}
          </button>
        ) : (
          <button
            onClick={handleDirectGenerate}
            disabled={!canSubmit}
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
