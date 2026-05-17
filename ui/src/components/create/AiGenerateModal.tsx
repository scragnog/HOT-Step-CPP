// AiGenerateModal.tsx — Modal for generating song content via external LLM
//
// Lets the user pick an LLM provider/model, enter a subject and genre/style,
// and generates lyrics + caption + metadata in one shot. Results populate
// the CreatePanel form fields.
//
// Uses the same /api/inspire/llm endpoint that InstaGen uses, so zero
// backend changes required.

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { X, Sparkles, Dice5, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../../context/AuthContext';
import {
  runLlmInspire,
  fetchInspireProviders,
  generateRandomSubject,
  type InspireProvider,
} from '../../services/inspireApi';
import { VOCAL_LANGUAGES } from '../../constants/languages';

// ── Result type for the parent callback ──────────────────────────────────
export interface AiGenerateResult {
  caption: string;
  lyrics: string;
  title: string;
  subject: string;
  bpm: number;
  keyScale: string;
  timeSignature: string;
  duration: number;
  vocalLanguage: string;
}

// ── Props ────────────────────────────────────────────────────────────────
interface AiGenerateModalProps {
  isOpen: boolean;
  onClose: () => void;
  onResult: (result: AiGenerateResult) => void;
}

// ── localStorage keys for persisting provider/model selection ─────────
const STORAGE_KEY_PROVIDER = 'hs-customgen-ai-provider';
const STORAGE_KEY_MODEL = 'hs-customgen-ai-model';

function loadPersisted(key: string): string {
  try { return JSON.parse(localStorage.getItem(key) || '""'); } catch { return ''; }
}
function savePersisted(key: string, value: string) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* ignore */ }
}

// ── Component ────────────────────────────────────────────────────────────
export const AiGenerateModal: React.FC<AiGenerateModalProps> = ({ isOpen, onClose, onResult }) => {
  const { } = useTranslation();
  const { token } = useAuth();

  // ── Provider state ──
  const [providers, setProviders] = useState<InspireProvider[]>([]);
  const [providersLoading, setProvidersLoading] = useState(true);
  const [selectedProvider, setSelectedProvider] = useState(() => loadPersisted(STORAGE_KEY_PROVIDER));
  const [selectedModel, setSelectedModel] = useState(() => loadPersisted(STORAGE_KEY_MODEL));

  // ── Form state ──
  const [subject, setSubject] = useState('');
  const [genreText, setGenreText] = useState('');
  const [language, setLanguage] = useState('en');

  // ── Submission state ──
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('');
  const [error, setError] = useState('');
  const [randomLoading, setRandomLoading] = useState(false);

  // ── Load providers on open ──
  useEffect(() => {
    if (!isOpen) return;
    setProvidersLoading(true);
    fetchInspireProviders()
      .then(list => {
        const available = list.filter(p => p.available);
        setProviders(available);
        // Auto-select first if none persisted or persisted one is unavailable
        if (available.length > 0 && !available.find(p => p.id === selectedProvider)) {
          const first = available[0];
          setSelectedProvider(first.id);
          setSelectedModel(first.default_model);
          savePersisted(STORAGE_KEY_PROVIDER, first.id);
          savePersisted(STORAGE_KEY_MODEL, first.default_model);
        }
      })
      .catch(() => { /* leave empty */ })
      .finally(() => setProvidersLoading(false));
  }, [isOpen]);

  // ── Current provider's models ──
  const currentProvider = useMemo(
    () => providers.find(p => p.id === selectedProvider),
    [providers, selectedProvider],
  );
  const models = currentProvider?.models || [];

  // ── Provider change handler ──
  const handleProviderChange = useCallback((id: string) => {
    setSelectedProvider(id);
    savePersisted(STORAGE_KEY_PROVIDER, id);
    const prov = providers.find(p => p.id === id);
    if (prov) {
      setSelectedModel(prov.default_model);
      savePersisted(STORAGE_KEY_MODEL, prov.default_model);
    }
  }, [providers]);

  const handleModelChange = useCallback((model: string) => {
    setSelectedModel(model);
    savePersisted(STORAGE_KEY_MODEL, model);
  }, []);

  // ── Parse genre text into array ──
  const parseGenres = useCallback((text: string): string[] => {
    return text
      .split(/[,;]+/)
      .map(s => s.trim())
      .filter(s => s.length > 0);
  }, []);

  // ── Validation ──
  const canSubmit = useMemo(() => {
    if (!selectedProvider) return false;
    if (!subject.trim() && !genreText.trim()) return false;
    return true;
  }, [selectedProvider, subject, genreText]);

  // ── Random subject ──
  const handleRandomSubject = useCallback(async () => {
    if (!selectedProvider) return;
    setRandomLoading(true);
    try {
      const genres = parseGenres(genreText);
      const result = await generateRandomSubject(
        { provider: selectedProvider, model: selectedModel || undefined, genres: genres.length > 0 ? genres : undefined },
        token || undefined,
      );
      setSubject(result);
    } catch (err: any) {
      setError(err.message || 'Failed to generate subject');
    } finally {
      setRandomLoading(false);
    }
  }, [selectedProvider, selectedModel, genreText, parseGenres, token]);

  // ── Submit ──
  const handleSubmit = useCallback(async () => {
    if (!canSubmit || loading) return;
    setError('');
    setLoading(true);
    setLoadingStage('Generating song with AI…');

    try {
      const genres = parseGenres(genreText);
      // If no genres provided, use a neutral default so the LLM has context
      const effectiveGenres = genres.length > 0 ? genres : ['any genre'];
      const effectiveSubject = subject.trim() || 'a creative and interesting topic of your choice';

      const result = await runLlmInspire(
        {
          provider: selectedProvider,
          model: selectedModel || undefined,
          genres: effectiveGenres,
          subject: effectiveSubject,
          language,
        },
        token || undefined,
      );

      // Build the result for the parent
      const aiResult: AiGenerateResult = {
        caption: result.caption || effectiveGenres.join(', '),
        lyrics: result.lyrics || '',
        title: result.title || '',
        subject: effectiveSubject,
        bpm: result.bpm || 0,
        keyScale: result.key || '',
        timeSignature: result.timeSignature || '',
        duration: result.duration || 0,
        vocalLanguage: language,
      };

      onResult(aiResult);
      onClose();

      // Reset form for next use (keep provider/model persisted)
      setSubject('');
      setGenreText('');
      setError('');
    } catch (err: any) {
      setError(err.message || 'AI generation failed');
    } finally {
      setLoading(false);
      setLoadingStage('');
    }
  }, [canSubmit, loading, parseGenres, genreText, subject, selectedProvider, selectedModel, language, token, onResult, onClose]);

  // ── Keyboard shortcuts ──
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !loading) onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, loading, onClose]);

  if (!isOpen) return null;

  // ── Shared input classes ──
  const inputClass = 'w-full px-3 py-2.5 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 placeholder:text-zinc-400 dark:placeholder:text-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors';
  const selectClass = 'w-full px-3 py-2.5 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors cursor-pointer appearance-none';
  const labelClass = 'block text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider mb-1.5';

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
        onClick={loading ? undefined : onClose}
      />

      {/* Modal */}
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
        <div
          className="w-full max-w-lg bg-zinc-50 dark:bg-zinc-900/95 rounded-2xl border border-zinc-200 dark:border-white/10 shadow-2xl pointer-events-auto overflow-hidden"
          onClick={e => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-200 dark:border-white/5">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-pink-500 flex items-center justify-center">
                <Sparkles size={16} className="text-white" />
              </div>
              <div>
                <h3 className="text-base font-bold text-zinc-900 dark:text-white">Generate with AI</h3>
                <p className="text-[11px] text-zinc-500">Describe your song, let AI fill in the rest</p>
              </div>
            </div>
            <button
              onClick={onClose}
              disabled={loading}
              className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors disabled:opacity-50"
            >
              <X size={18} />
            </button>
          </div>

          {/* Body */}
          <div className="px-5 py-4 space-y-4 max-h-[60vh] overflow-y-auto">
            {/* Provider / Model */}
            <div>
              <label className={labelClass}>LLM Provider</label>
              {providersLoading ? (
                <div className="flex items-center gap-2 text-xs text-zinc-500 py-2">
                  <Loader2 size={14} className="animate-spin" />
                  Loading providers…
                </div>
              ) : providers.length === 0 ? (
                <div className="text-xs text-amber-400 py-2">
                  ⚠ No LLM providers configured. Set up LM Studio, Ollama, or another OpenAI-compatible API in Settings.
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-2">
                  <select
                    className={selectClass}
                    value={selectedProvider}
                    onChange={e => handleProviderChange(e.target.value)}
                    disabled={loading}
                    title="LLM Provider"
                  >
                    {providers.map(p => (
                      <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                  </select>
                  <select
                    className={selectClass}
                    value={selectedModel}
                    onChange={e => handleModelChange(e.target.value)}
                    disabled={loading}
                    title="Model"
                  >
                    {models.map(m => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                    {models.length === 0 && <option value="">No models</option>}
                  </select>
                </div>
              )}
            </div>

            {/* Genre / Style */}
            <div>
              <label className={labelClass}>Genre / Style</label>
              <input
                type="text"
                className={inputClass}
                placeholder="e.g. indie folk, acoustic, dreamy female vocals"
                value={genreText}
                onChange={e => setGenreText(e.target.value)}
                disabled={loading}
              />
              <p className="mt-1 text-[10px] text-zinc-500">
                Comma-separated genres and style descriptors. Used to generate a rich caption.
              </p>
            </div>

            {/* Subject */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">
                  Song Subject
                </label>
                <button
                  onClick={handleRandomSubject}
                  disabled={loading || randomLoading || !selectedProvider}
                  className="flex items-center gap-1 px-2 py-0.5 rounded-lg text-[10px] font-medium text-violet-400 hover:text-violet-300 hover:bg-violet-400/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Generate a random subject"
                >
                  {randomLoading ? (
                    <Loader2 size={10} className="animate-spin" />
                  ) : (
                    <Dice5 size={10} />
                  )}
                  Random
                </button>
              </div>
              <textarea
                className={`${inputClass} resize-none`}
                placeholder="e.g. A night drive through empty streets, thinking about someone who left"
                value={subject}
                onChange={e => setSubject(e.target.value)}
                disabled={loading}
                rows={2}
              />
              <p className="mt-1 text-[10px] text-zinc-500">
                What the song should be about. Leave empty for AI to choose.
              </p>
            </div>

            {/* Language */}
            <div>
              <label className={labelClass}>Vocal Language</label>
              <select
                className={selectClass}
                value={language}
                onChange={e => setLanguage(e.target.value)}
                disabled={loading}
              >
                {VOCAL_LANGUAGES.map(l => (
                  <option key={l.value} value={l.value}>{l.label}</option>
                ))}
              </select>
            </div>

            {/* Error */}
            {error && (
              <div className="px-3 py-2 rounded-xl bg-red-500/10 border border-red-500/20 text-xs text-red-400">
                {error}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-5 py-4 border-t border-zinc-200 dark:border-white/5">
            <button
              onClick={handleSubmit}
              disabled={!canSubmit || loading || providers.length === 0}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-violet-600 to-pink-600 hover:from-violet-500 hover:to-pink-500 text-white font-semibold text-sm transition-all duration-200 hover:shadow-lg hover:shadow-violet-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  {loadingStage}
                </>
              ) : (
                <>
                  <Sparkles size={16} />
                  Generate Song
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </>
  );
};
