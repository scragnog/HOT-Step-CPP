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
  fetchInspireProviders,
  fetchInstagenPrompt,
  saveInstagenPrompt,
  resetInstagenPrompt,
  instaWorkflowApi,
  followJob,
  type InstaWorkflowInput,
  type InspireResult,
  type InspireProvider,
} from '../../services/inspireApi';
import { songApi } from '../../services/api';
import {
  addManualQueueItem,
  updateManualQueueItem,
  completeManualQueueItem,
  failManualQueueItem,
} from '../../stores/audioGenQueueStore';
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
  const [activePreviewJob, setActivePreviewJob] = useState<string | null>(null);
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

  // ── Random subject toggle ──
  const handleRandomSubject = useCallback(() => {
    setRandomSubject(true);
    setSubject('');
  }, [setSubject]);

  const [previewDocument, setPreviewDocument] = useState<{ id: string; revision: number; lyrics: string; caption: string } | null>(null);
  // Recover a saved preview after a browser reconnect. The server document
  // remains authoritative, including edits made by another client.
  useEffect(() => {
    if (!token) return;
    const id = localStorage.getItem('hs-instagen-preview-document');
    if (!id) return;
    let alive = true;
    instaWorkflowApi.getDocument(token, id).then(({ document }) => {
      if (!alive || document.kind !== 'insta-preview') return;
      const data = document.data as { result: InspireResult; edits: { lyrics: string; caption: string } };
      setPreviewDocument({ id, revision: document.revision, lyrics: data.edits.lyrics, caption: data.edits.caption });
      setInspireResult(data.result);
      setEditedLyrics(data.edits.lyrics);
      setEditedCaption(data.edits.caption);
      setPhase('preview');
    }).catch(() => { localStorage.removeItem('hs-instagen-preview-document'); });
    return () => { alive = false; };
  }, [token]);

  const captureInput = useCallback((): InstaWorkflowInput => {
    let settings: Record<string, unknown> = {};
    try { settings = JSON.parse(localStorage.getItem('ace-settings') || '{}'); } catch { /* defaults */ }
    return {
      caption: computedCaption, genres: [...selectedGenres], lyricMode, subject: subject.trim(),
      randomSubject, provider: selectedProvider, model: selectedModel, vocalLanguage, thinking,
      engineParams: { ...globalParams.getGlobalParams() },
      expectedBackend: useBackendStore.getState().activeBackendId,
      coResident: typeof settings.coResident === 'boolean' ? settings.coResident : false,
      cacheLmCodes: typeof settings.cacheLmCodes === 'boolean' ? settings.cacheLmCodes : true,
    };
  }, [computedCaption, selectedGenres, lyricMode, subject, randomSubject, selectedProvider, selectedModel, vocalLanguage, thinking, globalParams]);

  const trackRender = useCallback(async (jobId: string, queueId: string, capturedToken: string) => {
    updateManualQueueItem(queueId, { workflowJob: { id: jobId, token: capturedToken }, stage: 'Preparing song...' });
    try {
      await followJob(capturedToken, jobId, {
        onSnapshot: job => {
          if (job.status === 'interrupted') updateManualQueueItem(queueId, { stage: 'Interrupted by server restart' });
        },
        onEvent: event => {
          if (event.type === 'stage') {
            const stage = (event.data as { stage?: string } | null)?.stage;
            if (stage) updateManualQueueItem(queueId, { stage });
          }
          if (event.type === 'request') {
            const title = (event.data as { title?: string } | null)?.title;
            if (title) updateManualQueueItem(queueId, { title, stage: 'Generating audio...' });
          }
        },
      });
      const { job } = await instaWorkflowApi.get(capturedToken, jobId);
      if (job.status !== 'succeeded') throw new Error(job.error || `Song ${job.status}`);
      const output = job.result as { audio?: { audioUrls?: string[]; songIds?: string[]; masteredAudioUrl?: string; noAdapterAudioUrl?: string; duration?: number } };
      const audio = output.audio || {};
      const songId = audio.songIds?.[0];
      completeManualQueueItem(queueId, {
        audioUrl: audio.audioUrls?.[0] || '', songId,
        masteredAudioUrl: audio.masteredAudioUrl, noAdapterAudioUrl: audio.noAdapterAudioUrl,
        audioDuration: audio.duration,
      });
      if (songId) {
        try { const { song } = await songApi.get(songId); onSongCreated?.(song); } catch { /* library refresh still runs */ }
      }
    } catch (err: any) {
      failManualQueueItem(queueId, err.message || 'Generation failed');
    }
  }, [onSongCreated]);

  // Preview only resolves metadata and creates a revisioned document. The
  // render is a separate approval against that document's exact revision.
  const handleInspire = useCallback(async () => {
    if (!canSubmit || !token) return;
    setError(''); setPhase('inspiring'); setInspireProgress('Starting...');
    try {
      const { job } = await instaWorkflowApi.preview(token, captureInput());
      setActivePreviewJob(job.id);
      await followJob(token, job.id, {
        onEvent: event => {
          if (event.type === 'stage') setInspireProgress((event.data as { stage?: string } | null)?.stage || 'Working...');
        },
      });
      const { job: finished } = await instaWorkflowApi.get(token, job.id);
      if (finished.status !== 'succeeded') throw new Error(finished.error || `Preview ${finished.status}`);
      const output = finished.result as { documentId: string; revision: number; result: InspireResult };
      localStorage.setItem('hs-instagen-preview-document', output.documentId);
      setPreviewDocument({ id: output.documentId, revision: output.revision, lyrics: output.result.lyrics, caption: output.result.caption });
      setInspireResult(output.result);
      setEditedLyrics(output.result.lyrics);
      setEditedCaption(output.result.caption);
      setPhase('preview');
    } catch (err: any) { setError(err.message || 'Preview failed'); setPhase('input'); }
    finally { setActivePreviewJob(null); }
  }, [canSubmit, token, captureInput]);

  const handleGenerateFromPreview = useCallback(async () => {
    if (!inspireResult || !previewDocument || !token) return;
    setError(''); setPhase('generating');
    try {
      let revision = previewDocument.revision;
      if (editedLyrics !== previewDocument.lyrics || editedCaption !== previewDocument.caption) {
        const current = await instaWorkflowApi.getDocument(token, previewDocument.id);
        if (current.document.revision !== revision) throw new Error('Preview changed elsewhere; reopen it before generating');
        const { document } = await instaWorkflowApi.updateDocument(token, previewDocument.id, revision, {
          ...current.document.data, edits: { lyrics: editedLyrics, caption: editedCaption },
        });
        revision = document.revision;
      }
      const { job } = await instaWorkflowApi.approve(token, previewDocument.id, revision);
      const queueId = addManualQueueItem({ title: inspireResult.title || editedCaption || 'Auto-Gen', caption: editedCaption });
      void trackRender(job.id, queueId, token);
      localStorage.removeItem('hs-instagen-preview-document');
      setPhase('input'); setInspireResult(null); setPreviewDocument(null);
    } catch (err: any) { setError(err.message || 'Approval failed'); setPhase('preview'); }
  }, [inspireResult, previewDocument, token, editedLyrics, editedCaption, trackRender]);

  const handleDirectGenerate = useCallback(async () => {
    if (!canSubmit || !token) return;
    const input = captureInput();
    setError('');
    try {
      const { job } = await instaWorkflowApi.direct(token, input);
      const queueId = addManualQueueItem({ title: input.caption || 'Auto-Gen', caption: input.caption });
      void trackRender(job.id, queueId, token);
    } catch (err: any) { setError(err.message || 'Generation failed'); }
  }, [canSubmit, token, captureInput, trackRender]);

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

    localStorage.removeItem('hs-instagen-preview-document');
    setPhase('input');
    setInspireResult(null);
    onNavigate?.('create');
  }, [inspireResult, editedCaption, editedLyrics, lyricMode, vocalLanguage, onNavigate]);

  // ── Back to input from preview ──
  const handleBack = useCallback(() => {
    localStorage.removeItem('hs-instagen-preview-document');
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
              onReset={lyricMode !== 'lyrics' ? () => setLyricMode('lyrics') : undefined}
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
                onReset={vocalLanguage !== 'en' ? () => setVocalLanguage('en') : undefined}
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
            defaultValue={true}
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
            {activePreviewJob && <button onClick={() => { if (token) void instaWorkflowApi.cancel(token, activePreviewJob); }} className="w-full py-2 text-sm text-zinc-400 hover:text-white">{t('common.cancel')}</button>}
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
