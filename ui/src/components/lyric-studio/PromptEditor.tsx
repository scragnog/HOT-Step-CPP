import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { X, Save, RotateCcw, Code2, Loader2 } from 'lucide-react';
import { lireekApi } from '../../services/lireekApi';

interface PromptData {
  name: string;
  source: string;
  content: string;
  default_content: string;
  has_default: boolean;
}

const FRIENDLY_NAMES: Record<string, string> = {
  generation_system: 'Generation',
  metadata_system: 'Metadata Planning',
  profile_system: 'Artist Profiler',
  refine_system: 'Refinement',
};

interface Props {
  open: boolean;
  onClose: () => void;
}

export const PromptEditor: React.FC<Props> = ({ open, onClose }) => {
  const [prompts, setPrompts] = useState<PromptData[]>([]);
  const { t } = useTranslation();
  const [selected, setSelected] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    if (open) loadPrompts();
  }, [open]);

  const loadPrompts = async () => {
    setLoading(true);
    try {
      const res = await lireekApi.listPrompts();
      const rawData = Array.isArray(res?.prompts) ? res.prompts : Array.isArray(res) ? res : [];
      const data: PromptData[] = rawData.map((p: any) => ({
        name: p.name,
        source: p.custom != null ? 'custom' : 'default',
        content: p.custom ?? p.default_content ?? '',
        default_content: p.default_content ?? '',
        has_default: !!p.default_content,
      }));
      setPrompts(data);
      if (data.length > 0 && !selected) {
        setSelected(data[0].name);
        setEditContent(data[0].content);
        setDirty(false);
      }
    } catch { } finally {
      setLoading(false);
    }
  };

  const selectPrompt = (name: string) => {
    const p = prompts.find(pp => pp.name === name);
    if (p) {
      setSelected(name);
      setEditContent(p.content);
      setDirty(false);
    }
  };

  const handleSave = async () => {
    if (!selected) return;
    setSaving(true);
    try {
      await lireekApi.savePrompt(selected, editContent);
      setDirty(false);
      setToast('Saved');
      setTimeout(() => setToast(null), 2000);
      loadPrompts();
    } catch (err) {
      setToast(`Save failed: ${(err as Error).message}`);
      setTimeout(() => setToast(null), 3000);
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    if (!selected) return;
    if (!confirm('Reset this prompt to its default? Your customizations will be lost.')) return;
    try {
      await lireekApi.resetPrompt(selected);
      setToast('Reset to default');
      setTimeout(() => setToast(null), 2000);
      await loadPrompts();
      const updated = prompts.find(p => p.name === selected);
      if (updated) setEditContent(updated.content);
    } catch (err) {
      setToast(`Reset failed: ${(err as Error).message}`);
      setTimeout(() => setToast(null), 3000);
    }
  };

  if (!open) return null;

  const currentPrompt = prompts.find(p => p.name === selected);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 dark:bg-black/60 backdrop-blur-sm">
      <div className="bg-white dark:bg-zinc-900 rounded-2xl border border-zinc-300 dark:border-white/10 shadow-2xl w-[900px] h-[95vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-white/5">
          <div className="flex items-center gap-2">
            <Code2 className="w-5 h-5 text-cyan-400" />
            <h2 className="text-lg font-bold text-white">{t('lyric.systemPrompts')}</h2>
          </div>
          <div className="flex items-center gap-2">
            {toast && <span className="text-xs text-green-400">{toast}</span>}
            <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/10 text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex flex-1 min-h-0">
          {/* Sidebar */}
          <div className="w-56 flex-shrink-0 border-r border-zinc-200 dark:border-white/5 overflow-y-auto py-2">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-5 h-5 animate-spin text-zinc-500" />
              </div>
            ) : (
              prompts.map(p => (
                <button
                  key={p.name}
                  onClick={() => selectPrompt(p.name)}
                  className={`w-full text-left px-4 py-2.5 text-sm transition-colors flex items-center justify-between ${
                    selected === p.name
                      ? 'bg-white/10 text-white border-l-2 border-cyan-400'
                      : 'text-zinc-600 dark:text-zinc-400 hover:bg-white/5 hover:text-zinc-800 dark:text-zinc-200 border-l-2 border-transparent'
                  }`}
                >
                  <span className="truncate">{FRIENDLY_NAMES[p.name] || p.name}</span>
                  {p.source === 'file' && (
                    <span className="text-[9px] text-cyan-400 bg-cyan-400/10 px-1 rounded">{t('lyric.custom')}</span>
                  )}
                </button>
              ))
            )}
          </div>

          {/* Editor */}
          <div className="flex-1 flex flex-col min-w-0">
            {selected ? (
              <>
                <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-200 dark:border-white/5">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">{FRIENDLY_NAMES[selected] || selected}</span>
                    {currentPrompt?.source === 'file' && (
                      <span className="text-[10px] text-cyan-400 bg-cyan-400/10 px-1.5 py-0.5 rounded">{t('lyric.customized')}</span>
                    )}
                    {dirty && (
                      <span className="text-[10px] text-amber-400 bg-amber-400/10 px-1.5 py-0.5 rounded">{t('lyric.unsavedChanges')}</span>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5">
                    {currentPrompt?.has_default && currentPrompt?.source === 'file' && (
                      <button onClick={handleReset}
                        className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs text-zinc-600 dark:text-zinc-400 hover:text-zinc-800 dark:text-zinc-200 hover:bg-white/5 transition-colors"
                        title="Reset to default"
                      >
                        <RotateCcw className="w-3 h-3" /> {t('lyric.reset')}
                      </button>
                    )}
                    <button onClick={handleSave} disabled={!dirty || saving}
                      className="flex items-center gap-1 px-3 py-1 rounded-lg text-xs font-medium bg-cyan-500 text-black hover:bg-cyan-400 disabled:opacity-30 transition-all"
                    >
                      {saving ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
                      {t('common.save')}
                    </button>
                  </div>
                </div>
                <textarea
                  value={editContent}
                  onChange={(e) => { setEditContent(e.target.value); setDirty(true); }}
                  className="flex-1 p-4 bg-black/20 dark:bg-black/40 text-sm text-zinc-800 dark:text-zinc-200 font-mono leading-relaxed resize-none focus:outline-none"
                  spellCheck={false}
                  style={{ minHeight: 0 }}
                />
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center text-zinc-500 text-sm">
                {t('lyric.selectPromptToEdit')}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
