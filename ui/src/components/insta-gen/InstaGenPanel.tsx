// InstaGenPanel.tsx — Main Insta-Gen studio container
//
// Genre-first music generation with three lyric modes:
//   Instrumental:  no lyrics at all
//   Lyrics:        built-in ACE-Step LM generates lyrics (random topic)
//   Lyrics + AI:   external LLM generates subject-aware lyrics
//
// State machine: Input → (optional) Preview → Generate

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Sparkles, Music, Eye, EyeOff, Mic, MicOff, Bot, PenLine, Code2, Save, RotateCcw, ChevronDown, ChevronRight } from 'lucide-react';
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
  fetchInstagenPrompt,
  saveInstagenPrompt,
  resetInstagenPrompt,
  type InspireResult,
  type InspireProvider,
} from '../../services/inspireApi';
import { generateApi, songApi } from '../../services/api';
import { createGenerationTimer, getGenerationTimeoutMinutes } from '../../utils/generationTimer';
import {
  addManualQueueItem,
  updateManualQueueItem,
  completeManualQueueItem,
  failManualQueueItem,
} from '../../stores/audioGenQueueStore';
import type { GenerationParams } from '../../types';
import { VOCAL_LANGUAGES } from '../../constants/languages';
import { useBackendStore } from '../../stores/backendStore';
import { MM3_BACKEND_ID } from '../../utils/captionForBackend';
import { CoverArtSubjectSection } from '../shared/CoverArtSubjectSection';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { ParamLabel } from '../shared/ParamLabel';

type LyricMode = 'instrumental' | 'lyrics' | 'lyrics-ai';
type Phase = 'input' | 'inspiring' | 'preview' | 'generating';

/** True when the render about to be submitted will run on MiniMax-Music3.
 *
 *  MM3 has no length input: a duration becomes a frame cap and nothing else, so
 *  the LLM's estimate can only cut the planner's own ending off. Every MM3
 *  render is auto, enforced server-side in backends/minimax/generate.ts — this
 *  just keeps the estimate out of the request in the first place.
 *
 *  Read the live selection at submission, as audioGenQueueStore does. The
 *  server freezes its active backend at POST time for the accepted job. */
const isMm3Render = (): boolean =>
  useBackendStore.getState().activeBackendId === MM3_BACKEND_ID;

interface InstaGenPanelProps {
  onSongCreated?: (song: any) => void;
  activeJobCount: number;
  onNavigate?: (view: string) => void;
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

export const InstaGenPanel: React.FC<InstaGenPanelProps> = ({ onSongCreated, activeJobCount: _activeJobCount, onNavigate }) => {
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
  const [randomSubject, setRandomSubject] = useState(false);

  // ── Prompt editor state ──
  const [promptEditorOpen, setPromptEditorOpen] = useState(false);
  const [promptContent, setPromptContent] = useState('');
  const [promptDefault, setPromptDefault] = useState('');
  const [promptDirty, setPromptDirty] = useState(false);
  const [promptIsCustom, setPromptIsCustom] = useState(false);
  const [promptSaving, setPromptSaving] = useState(false);
  const [promptLoaded, setPromptLoaded] = useState(false);

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
    if (lyricMode === 'lyrics-ai' && !randomSubject && !subject.trim()) return false;
    if (lyricMode === 'lyrics-ai' && !selectedProvider) return false;
    return true;
  }, [selectedGenres, additionalCaption, lyricMode, subject, selectedProvider, randomSubject]);

  // ── Build generation params ──
  const buildParams = useCallback((lyrics: string, caption: string): Partial<GenerationParams> => ({
    caption: caption || computedCaption,
    lyrics,
    instrumental: lyricMode === 'instrumental',
    vocalLanguage: lyricMode === 'instrumental' ? undefined : vocalLanguage,
    source: 'insta-gen',
    useCotCaption: thinking,
    skipLm: false, // InstaGen always needs the LM for metadata (BPM/key/timesig)
  }), [computedCaption, lyricMode, vocalLanguage, thinking]);

  // ── Random subject toggle ──
  const handleRandomSubject = useCallback(() => {
    setRandomSubject(true);
    setSubject('');
  }, [setSubject]);

  // ── Inspire flow (preview ON) ──
  const handleInspire = useCallback(async () => {
    if (!canSubmit) return;
    setError('');
    setPhase('inspiring');

    try {
      if (lyricMode === 'lyrics-ai') {
        // ── External LLM path ──
        // If random subject mode, generate a subject first
        let effectiveSubject = subject.trim();
        if (randomSubject || !effectiveSubject) {
          setInspireProgress('Generating random subject...');
          effectiveSubject = await generateRandomSubject(
            { provider: selectedProvider, model: selectedModel || undefined, genres: selectedGenres },
            token || undefined,
          );
        }
        setInspireProgress('Generating lyrics via AI...');
        const llmResult = await runLlmInspire(
          {
            provider: selectedProvider,
            model: selectedModel || undefined,
            genres: selectedGenres,
            subject: effectiveSubject,
            language: vocalLanguage,
          },
          token || undefined,
        );

        let result: InspireResult;

        if (llmResult.structured && llmResult.bpm) {
          // ── Structured path: LLM returned full metadata, skip inspire ──
          console.log('[InstaGen] Structured LLM response — skipping inspire step');
          result = {
            caption: llmResult.caption || computedCaption,
            lyrics: llmResult.lyrics,
            title: llmResult.title,
            bpm: llmResult.bpm,
            duration: llmResult.duration || 200,
            keyScale: llmResult.key || 'C major',
            timeSignature: llmResult.timeSignature || '4/4',
            vocalLanguage,
          };
        } else {
          // ── Legacy path: run inspire for metadata ──
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

          result = {
            caption: llmResult.caption || computedCaption,
            lyrics: llmResult.lyrics,
            title: llmResult.title,
            bpm: metaResult.bpm,
            duration: metaResult.duration,
            keyScale: metaResult.keyScale,
            timeSignature: metaResult.timeSignature,
            vocalLanguage,
          };
        }

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
        setEditedCaption(thinking ? result.caption : computedCaption);
        setPhase('preview');
      }
    } catch (err: any) {
      setError(err.message || 'Inspire failed');
      setPhase('input');
    }
  }, [canSubmit, lyricMode, selectedProvider, selectedModel, selectedGenres, subject, vocalLanguage, computedCaption, thinking, globalParams, token]);

  // ── Generate from preview ──
  const handleGenerateFromPreview = useCallback(() => {
    if (!inspireResult || !token) return;
    const params = buildParams(editedLyrics, editedCaption);
    // Prefer LLM-generated title, then derive from lyrics, then caption
    params.title = inspireResult.title || deriveTitleFromLyrics(editedLyrics) || computedCaption;
    // Include metadata from inspire result
    if (inspireResult.bpm) params.bpm = inspireResult.bpm;
    if (inspireResult.duration && !isMm3Render()) params.duration = inspireResult.duration;
    if (inspireResult.keyScale) params.keyScale = inspireResult.keyScale;
    if (inspireResult.timeSignature) params.timeSignature = inspireResult.timeSignature;

    const capturedToken = token;

    // Create queue item immediately
    const queueId = addManualQueueItem({
      title: params.title || computedCaption || 'Auto-Gen',
      caption: editedCaption,
    });

    // Submit via serial queue
    enqueueInstaJob(async () => {
      try {
        updateManualQueueItem(queueId, { stage: 'Submitting to engine…' });
        const engineParams = globalParams.getGlobalParams();
        const enrichedParams = {
          ...engineParams,
          ...params,
          source: 'insta-gen',
          coResident: (() => { try { return JSON.parse(localStorage.getItem('ace-settings') || '{}').coResident; } catch { return false; } })(),
          cacheLmCodes: (() => { try { return JSON.parse(localStorage.getItem('ace-settings') || '{}').cacheLmCodes; } catch { return true; } })(),
        };

        const res = await generateApi.submit(enrichedParams as any, capturedToken);
        updateManualQueueItem(queueId, { jobId: res.jobId, stage: 'Generating audio…' });

        // Poll until done — clock ignores server-queue wait.
        const timer = createGenerationTimer();
        while (true) {
          await new Promise(r => setTimeout(r, 1500));
          const status = await generateApi.status(res.jobId);
          const t = timer.tick(status.status);
          const progress = status.progress !== undefined
            ? Math.min(100, Math.max(0, (status.progress > 1 ? status.progress / 100 : status.progress) * 100))
            : undefined;
          updateManualQueueItem(queueId, {
            progress,
            stage: status.stage || 'Generating…',
            elapsed: t.elapsed,
          });

          if (status.status === 'succeeded') {
            const audioUrl = status.result?.audioUrls?.[0] || '';
            const songId = status.result?.songIds?.[0];
            completeManualQueueItem(queueId, {
              audioUrl,
              songId,
              masteredAudioUrl: status.result?.masteredAudioUrl,
              noAdapterAudioUrl: status.result?.noAdapterAudioUrl,
              audioDuration: status.result?.duration,
            });
            // Notify App to refresh library
            if (songId) {
              try {
                const { song } = await songApi.get(songId);
                onSongCreated?.(song);
              } catch { /* non-fatal */ }
            }
            break;
          }
          if (status.status === 'failed' || status.status === 'cancelled') {
            failManualQueueItem(queueId, status.error || 'Generation failed');
            break;
          }
          if (t.timedOut) {
            failManualQueueItem(queueId, `Generation timed out after ${getGenerationTimeoutMinutes()} minutes`);
            break;
          }
        }
      } catch (err: any) {
        failManualQueueItem(queueId, err.message || 'Generation failed');
      }
    });

    // Return to input after queuing
    setPhase('input');
    setInspireResult(null);
  }, [inspireResult, editedLyrics, editedCaption, computedCaption, buildParams, token, globalParams, onSongCreated]);

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
    const capturedRandomSubject = randomSubject;
    const capturedGlobalParams = { ...globalParams };
    const capturedToken = token;

    // Create queue item immediately — user sees it right away
    const queueId = addManualQueueItem({
      title: capturedCaption || 'Auto-Gen',
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
          // If random subject mode, generate a subject first
          let effectiveSubject = capturedSubject;
          if (capturedRandomSubject || !effectiveSubject) {
            updateManualQueueItem(queueId, { stage: 'Generating random subject…' });
            effectiveSubject = await generateRandomSubject(
              { provider: capturedProvider, model: capturedModel || undefined, genres: capturedGenres },
              capturedToken,
            );
          }
          updateManualQueueItem(queueId, { stage: 'Generating lyrics via AI…' });
          const llmResult = await runLlmInspire(
            {
              provider: capturedProvider,
              model: capturedModel || undefined,
              genres: capturedGenres,
              subject: effectiveSubject,
              language: capturedVocalLang,
            },
            capturedToken,
          );
          resolvedLyrics = llmResult.lyrics;
          resolvedCaption = llmResult.caption || capturedCaption;
          llmTitle = llmResult.title || '';

          // If structured response, use LLM metadata directly — skip inspire
          if (llmResult.structured && llmResult.bpm) {
            console.log('[InstaGen] Structured LLM response — skipping inspire step');
            const finalLyrics = resolvedLyrics;
            const params = buildParams(finalLyrics, resolvedCaption);
            params.title = llmTitle || deriveTitleFromLyrics(finalLyrics) || resolvedCaption;
            if (llmResult.bpm) params.bpm = llmResult.bpm;
            if (llmResult.duration && !isMm3Render()) params.duration = llmResult.duration;
            if (llmResult.key) params.keyScale = llmResult.key;
            if (llmResult.timeSignature) params.timeSignature = llmResult.timeSignature;

            updateManualQueueItem(queueId, {
              title: params.title || resolvedCaption,
              stage: 'Submitting to engine…',
            });

            // Merge with global engine params and submit
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

            // Poll until done — clock ignores server-queue wait.
            const timer = createGenerationTimer();
            while (true) {
              await new Promise(r => setTimeout(r, 1500));
              const status = await generateApi.status(res.jobId);
              const t = timer.tick(status.status);
              const progress = status.progress !== undefined
                ? Math.min(100, Math.max(0, (status.progress > 1 ? status.progress / 100 : status.progress) * 100))
                : undefined;
              updateManualQueueItem(queueId, {
                progress,
                stage: status.stage || 'Generating…',
                elapsed: t.elapsed,
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
              if (t.timedOut) {
                throw new Error(`Generation timed out after ${getGenerationTimeoutMinutes()} minutes`);
              }
            }
          }
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
        // Caption rewrite ON → use inspire's rich caption; OFF → user's original
        const finalCaption = capturedThinking
          ? (inspireResult.caption || resolvedCaption)
          : resolvedCaption;
        const params = buildParams(finalLyrics, finalCaption);
        params.title = llmTitle || deriveTitleFromLyrics(finalLyrics) || resolvedCaption;
        if (inspireResult.bpm) params.bpm = inspireResult.bpm;
        if (inspireResult.duration && !isMm3Render()) params.duration = inspireResult.duration;
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

        // Step 5: Poll until done — clock ignores server-queue wait.
        const timer = createGenerationTimer();
        while (true) {
          await new Promise(r => setTimeout(r, 1500));
          const status = await generateApi.status(res.jobId);
          const t = timer.tick(status.status);
          const progress = status.progress !== undefined
            ? Math.min(100, Math.max(0, (status.progress > 1 ? status.progress / 100 : status.progress) * 100))
            : undefined;
          updateManualQueueItem(queueId, {
            progress,
            stage: status.stage || 'Generating…',
            elapsed: t.elapsed,
          });

          if (status.status === 'succeeded') {
            const audioUrl = status.result?.audioUrls?.[0] || '';
            const songId = status.result?.songIds?.[0];
            completeManualQueueItem(queueId, {
              audioUrl,
              songId,
              masteredAudioUrl: status.result?.masteredAudioUrl,
              noAdapterAudioUrl: status.result?.noAdapterAudioUrl,
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
          if (t.timedOut) {
            throw new Error(`Generation timed out after ${getGenerationTimeoutMinutes()} minutes`);
          }
        }
      } catch (err: any) {
        failManualQueueItem(queueId, err.message || 'Generation failed');
      }
    });
  }, [canSubmit, token, lyricMode, selectedProvider, selectedModel, selectedGenres, subject, vocalLanguage, computedCaption, thinking, buildParams, globalParams, onSongCreated]);

  // ── Refine in Custom-Gen — write preview data to CreatePanel's persisted state ──
  const handleRefineInCustomGen = useCallback(() => {
    if (!inspireResult) return;
    try {
      // Write to the same localStorage keys that CreatePanel's usePersistedState reads
      localStorage.setItem('hs-caption', JSON.stringify(editedCaption));
      localStorage.setItem('hs-lyrics', JSON.stringify(editedLyrics));
      localStorage.setItem('hs-instrumental', JSON.stringify(lyricMode === 'instrumental'));
      if (inspireResult.bpm) localStorage.setItem('hs-bpm', JSON.stringify(inspireResult.bpm));
      // Not in MM3 mode: the Create panel hides its duration control there, so
      // writing one would leave a number in a box nobody can see or clear.
      if (inspireResult.duration && !isMm3Render()) {
        localStorage.setItem('hs-duration', JSON.stringify(inspireResult.duration));
      }
      if (inspireResult.keyScale) localStorage.setItem('hs-keyScale', JSON.stringify(inspireResult.keyScale));
      if (inspireResult.timeSignature) localStorage.setItem('hs-timeSignature', JSON.stringify(inspireResult.timeSignature));
      if (vocalLanguage) localStorage.setItem('hs-vocalLanguage', JSON.stringify(vocalLanguage));
      // Title from LLM or derived
      const title = inspireResult.title || '';
      if (title) localStorage.setItem('hs-title', JSON.stringify(title));
    } catch { /* ignore storage errors */ }

    setPhase('input');
    setInspireResult(null);
    onNavigate?.('create');
  }, [inspireResult, editedCaption, editedLyrics, lyricMode, vocalLanguage, onNavigate]);

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
          onRefine={handleRefineInCustomGen}
          isGenerating={false}
        />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-white dark:bg-suno overflow-y-auto">

      {/* Header — matches CreatePanel */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-white/5 flex-shrink-0">
        <h2 className="text-lg font-bold text-zinc-900 dark:text-white">{t('instaGen.title')}</h2>
        <span className="text-xs text-zinc-500 font-medium">{t('instaGen.panelSubtitle')}</span>
      </div>

      {/* Content */}
      <div className="flex-1 px-4 pt-4 space-y-4 pb-4">
        {/* Genre Selector */}
        <GenreSelector selected={selectedGenres} onChange={setSelectedGenres} />

        {/* ── Lyric Mode Selector (3-way segmented control) ── */}
        <div>
          <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1.5">
            <ParamLabel
              label="Vocal Mode"
              info="How the song gets its lyrics. Instrumental: no lyrics at all. Lyrics: the built-in LM writes lyrics on its own, on a random topic. Lyrics + AI: an external LLM writes lyrics from a subject you give it, using the provider and model chosen below."
            />
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

        {/* ── System Prompt Editor (Lyrics + AI only) ── */}
        {lyricMode === 'lyrics-ai' && (
          <div className="rounded-xl border border-zinc-200 dark:border-white/5 overflow-hidden">
            <button
              onClick={() => {
                if (!promptEditorOpen && !promptLoaded) {
                  // Load prompt on first open
                  fetchInstagenPrompt().then(data => {
                    setPromptDefault(data.default_content);
                    setPromptContent(data.custom || data.default_content);
                    setPromptIsCustom(!!data.custom);
                    setPromptLoaded(true);
                  }).catch(() => setPromptLoaded(true));
                }
                setPromptEditorOpen(!promptEditorOpen);
              }}
              className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium text-zinc-500 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-white/[0.02] transition-colors"
            >
              <span className="flex items-center gap-1.5">
                <Code2 size={12} className="text-cyan-400" />
                <ParamLabel
                  label="System Prompt"
                  info="The system prompt sent to the external LLM when it writes lyrics in Lyrics + AI mode. Save stores your edited version as a custom prompt used from then on; Reset discards it and restores the built-in default."
                  className="text-xs font-medium text-zinc-500 dark:text-zinc-400"
                />
                {promptIsCustom && (
                  <span className="text-[9px] text-cyan-400 bg-cyan-400/10 px-1 rounded">custom</span>
                )}
                {promptDirty && (
                  <span className="text-[9px] text-amber-400 bg-amber-400/10 px-1 rounded">unsaved</span>
                )}
              </span>
              {promptEditorOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            </button>
            {promptEditorOpen && (
              <div className="border-t border-zinc-200 dark:border-white/5">
                <textarea
                  value={promptContent}
                  onChange={(e) => { setPromptContent(e.target.value); setPromptDirty(true); }}
                  className="w-full h-48 p-3 bg-black/10 dark:bg-black/30 text-xs text-zinc-700 dark:text-zinc-300 font-mono leading-relaxed resize-y focus:outline-none"
                  spellCheck={false}
                  placeholder="Loading prompt..."
                />
                <div className="flex items-center justify-between px-3 py-1.5 bg-zinc-50 dark:bg-white/[0.02]">
                  <div className="flex items-center gap-1">
                    {promptIsCustom && (
                      <button
                        onClick={async () => {
                          if (!confirm('Reset to default prompt? Your customizations will be lost.')) return;
                          try {
                            await resetInstagenPrompt();
                            setPromptContent(promptDefault);
                            setPromptIsCustom(false);
                            setPromptDirty(false);
                          } catch { /* ignore */ }
                        }}
                        className="flex items-center gap-1 px-2 py-1 rounded text-[10px] text-zinc-500 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 hover:bg-white/5 transition-colors"
                      >
                        <RotateCcw size={10} /> Reset
                      </button>
                    )}
                  </div>
                  <button
                    disabled={!promptDirty || promptSaving}
                    onClick={async () => {
                      setPromptSaving(true);
                      try {
                        await saveInstagenPrompt(promptContent);
                        setPromptDirty(false);
                        setPromptIsCustom(true);
                      } catch { /* ignore */ }
                      setPromptSaving(false);
                    }}
                    className="flex items-center gap-1 px-2.5 py-1 rounded text-[10px] font-medium bg-cyan-500 text-black hover:bg-cyan-400 disabled:opacity-30 transition-all"
                  >
                    <Save size={10} />
                    {promptSaving ? 'Saving…' : 'Save'}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Subject (only for Lyrics + AI) ── */}
        {lyricMode === 'lyrics-ai' && (
          <div>
            <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1.5">
              <ParamLabel
                label="Song Subject"
                info="What the song is about, in your own words. The external LLM uses this to write the lyrics for Lyrics + AI mode. Required unless Random is on, which asks the LLM to invent a subject instead of using this field."
              />
              {!randomSubject && <span className="text-pink-500"> *</span>}
            </label>
            <div className="flex items-stretch gap-2">
              <input
                type="text"
                value={subject}
                onChange={(e) => { setSubject(e.target.value); setRandomSubject(false); }}
                placeholder={randomSubject ? 'Subject will be chosen at random' : 'e.g. a man tired from a life of working 9 to 5'}
                className={`flex-1 rounded-xl border bg-zinc-50 dark:bg-white/5 px-3 py-2.5 text-sm text-zinc-900 dark:text-white outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 transition-all ${
                  randomSubject
                    ? 'border-violet-500/30 placeholder:text-violet-400 dark:placeholder:text-violet-400/70'
                    : 'border-zinc-300 dark:border-white/10 placeholder:text-zinc-400 dark:placeholder:text-zinc-500'
                }`}
              />
              <button
                onClick={handleRandomSubject}
                disabled={!selectedProvider}
                className={`px-3 rounded-xl text-xs font-semibold text-white shadow-md transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5 flex-shrink-0 ${
                  randomSubject
                    ? 'bg-gradient-to-r from-violet-500 to-purple-500 shadow-violet-500/30 ring-1 ring-violet-400/50'
                    : 'bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 shadow-violet-500/20 hover:shadow-violet-500/30'
                }`}
              >
                Random
              </button>
            </div>
          </div>
        )}

        {/* ── LLM Provider & Model (only for Lyrics + AI) ── */}
        {lyricMode === 'lyrics-ai' && (
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1.5">
                <ParamLabel
                  label="LLM Provider"
                  info="Which configured external LLM writes the lyrics for Lyrics + AI mode. Only providers that responded successfully to a connectivity check appear here; add or fix credentials under Settings > AI Services > API Keys if none are available. Switching provider resets Model to that provider's default."
                />
              </label>
              <StyledSelect
                accent="pink"
                value={selectedProvider}
                onChange={(v) => {
                  setSelectedProvider(v);
                  const prov = providers.find(p => p.id === v);
                  if (prov) setSelectedModel(prov.default_model);
                }}
                options={providers.length === 0
                  ? [{ value: '', label: 'No providers available', disabled: true }]
                  : providers.map(p => ({ value: p.id, label: p.name }))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1.5">
                <ParamLabel
                  label="Model"
                  info="Which model of the selected LLM Provider writes the lyrics. Different models vary in writing style and how closely they follow the subject; there is no single better choice, it depends on the provider and what the lyrics need."
                />
              </label>
              <StyledSelect
                accent="pink"
                value={selectedModel}
                onChange={setSelectedModel}
                options={(currentProvider?.models || []).map(m => ({ value: m, label: m }))}
                className="w-full"
              />
            </div>
          </div>
        )}

        {/* Additional caption */}
        <div>
          <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1.5">
            <ParamLabel
              label={t('instaGen.captionAdditional')}
              info={t('instaGen.captionAdditionalInfo')}
            />
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
              <ParamLabel
                label={t('instaGen.languageLabel')}
                info={t('instaGen.languageLabelInfo')}
              />
            </label>
            <StyledSelect
              accent="pink"
              value={vocalLanguage}
              onChange={setVocalLanguage}
              options={VOCAL_LANGUAGES.map(lang => ({ value: lang.value, label: lang.label }))}
              className="w-full"
            />
          </div>
        )}

        {/* Caption preview */}
        {computedCaption && (
          <div>
            <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1.5">
              <ParamLabel
                label={t('instaGen.captionLabel')}
                info={t('instaGen.captionLabelInfo')}
              />
            </label>
            <div className="w-full rounded-xl border border-zinc-300 dark:border-white/10 bg-zinc-100 dark:bg-white/[0.03] px-3 py-2.5 text-sm text-zinc-700 dark:text-zinc-300 italic">
              {computedCaption}
            </div>
          </div>
        )}

        {/* Cover Art prompt override (only when enabled) */}
        <CoverArtSubjectSection />

        {/* Caption Rewrite toggle */}
        <div className="flex items-center gap-2 py-2">
          <PenLine size={14} className={thinking ? 'text-amber-400' : 'text-zinc-400'} />
          <Toggle
            accent="pink"
            checked={thinking}
            onChange={setThinking}
            label={t('instaGen.captionRewriteLabel')}
            info={t('instaGen.captionRewriteInfo')}
          />
        </div>

        {/* Preview toggle (hidden for instrumental) */}
        {lyricMode !== 'instrumental' && (
          <div className="flex items-center gap-2 py-2">
            {previewEnabled ? <Eye size={14} className="text-violet-400" /> : <EyeOff size={14} className="text-zinc-400" />}
            <Toggle
              accent="pink"
              checked={previewEnabled}
              onChange={setPreviewEnabled}
              label={t('instaGen.previewToggle')}
              info={t('instaGen.previewToggleInfo')}
            />
          </div>
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
