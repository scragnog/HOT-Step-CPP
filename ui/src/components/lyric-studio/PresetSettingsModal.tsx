import React, { useState, useEffect } from 'react';
import { X, Save, Loader2, ChevronDown, ChevronRight, Zap, Music, FolderSearch } from 'lucide-react';
import { lireekApi } from '../../services/lireekApi';
import { FileBrowserModal } from '../shared/FileBrowserModal';

interface PresetForm {
  adapter_path: string;
  self_attn: number;
  cross_attn: number;
  mlp: number;
  cond_embed: number;
  reference_track_path: string;
  audio_cover_strength: number;
}

const DEFAULT_FORM: PresetForm = {
  adapter_path: '',
  self_attn: 1.0,
  cross_attn: 1.0,
  mlp: 1.0,
  cond_embed: 1.0,
  reference_track_path: '',
  audio_cover_strength: 0.5,
};

interface PresetSettingsModalProps {
  isOpen: boolean;
  lyricsSetId: number;
  albumName: string;
  onClose: () => void;
  showToast: (msg: string) => void;
}

// Simple inline slider
const Slider: React.FC<{
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; help?: string;
}> = ({ label, value, min, max, step, onChange, help }) => (
  <div className="space-y-1">
    <div className="flex items-center justify-between">
      <label className="text-xs font-medium text-zinc-400">{label}</label>
      <span className="text-xs text-zinc-500 font-mono">{value.toFixed(2)}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value}
      onChange={e => onChange(parseFloat(e.target.value))}
      className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-pink-500"
    />
    {help && <p className="text-[10px] text-zinc-600">{help}</p>}
  </div>
);

export const PresetSettingsModal: React.FC<PresetSettingsModalProps> = ({
  isOpen, lyricsSetId, albumName, onClose, showToast,
}) => {
  const [form, setForm] = useState<PresetForm>(DEFAULT_FORM);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [groupsExpanded, setGroupsExpanded] = useState(false);
  const [browserOpen, setBrowserOpen] = useState(false);
  const [browserTarget, setBrowserTarget] = useState<'adapter' | 'reference'>('adapter');

  // Load existing preset
  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    lireekApi.getPreset(lyricsSetId)
      .then(res => {
        if (res.preset) {
          setForm({
            adapter_path: res.preset.adapter_path || '',
            self_attn: res.preset.adapter_group_scales?.self_attn ?? 1.0,
            cross_attn: res.preset.adapter_group_scales?.cross_attn ?? 1.0,
            mlp: res.preset.adapter_group_scales?.mlp ?? 1.0,
            cond_embed: res.preset.adapter_group_scales?.cond_embed ?? 1.0,
            reference_track_path: res.preset.reference_track_path || '',
            audio_cover_strength: res.preset.audio_cover_strength ?? 0.5,
          });
        } else {
          setForm(DEFAULT_FORM);
        }
      })
      .catch(err => showToast(`Failed to load preset: ${err.message}`))
      .finally(() => setLoading(false));
  }, [isOpen, lyricsSetId, showToast]);

  const save = async () => {
    setSaving(true);
    try {
      await lireekApi.upsertPreset(lyricsSetId, {
        adapter_path: form.adapter_path || undefined,
        adapter_group_scales: { self_attn: form.self_attn, cross_attn: form.cross_attn, mlp: form.mlp, cond_embed: form.cond_embed },
        reference_track_path: form.reference_track_path || undefined,
        audio_cover_strength: form.audio_cover_strength,
      });
      showToast('Preset saved');
      onClose();
    } catch (err: any) {
      showToast(`Save failed: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  const clear = async () => {
    setSaving(true);
    try {
      await lireekApi.deletePreset(lyricsSetId);
      setForm(DEFAULT_FORM);
      showToast('Preset cleared');
      onClose();
    } catch (err: any) {
      showToast(`Failed: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  if (!isOpen) return null;

  const adapterFileName = form.adapter_path ? form.adapter_path.split(/[\\/]/).pop() || '' : '';
  const matchFileName = form.reference_track_path ? form.reference_track_path.split(/[\\/]/).pop() || '' : '';

  return (
    <>
      <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
        <div className="w-full max-w-lg rounded-2xl bg-zinc-900 border border-white/10 shadow-2xl pointer-events-auto" onClick={(e) => e.stopPropagation()}>
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
            <div>
              <h2 className="text-base font-bold text-white">Album Preset</h2>
              <p className="text-xs text-zinc-500 mt-0.5">{albumName || 'Top Songs'}</p>
            </div>
            <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/5 text-zinc-400 hover:text-white transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Content */}
          <div className="px-6 py-5 space-y-5 max-h-[70vh] overflow-y-auto">
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-6 h-6 text-zinc-500 animate-spin" />
              </div>
            ) : (
              <>
                {/* Adapter Section */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-zinc-300">
                    <Zap className="w-4 h-4 text-pink-400" />
                    Adapter (LoRA/LoKR)
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium text-zinc-400">Adapter Path</label>
                    <div className="flex gap-2">
                      <input type="text" value={form.adapter_path}
                        onChange={e => setForm(p => ({ ...p, adapter_path: e.target.value }))}
                        placeholder="Path to .safetensors file or adapter folder"
                        className="flex-1 bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-pink-500 transition-colors"
                      />
                      <button onClick={() => { setBrowserTarget('adapter'); setBrowserOpen(true); }}
                        className="px-2.5 py-2 rounded-lg text-xs font-semibold bg-pink-900/20 text-pink-400 hover:bg-pink-900/30 transition-colors flex items-center gap-1 flex-shrink-0">
                        <FolderSearch size={12} /> Browse
                      </button>
                    </div>
                    {form.adapter_path && (
                      <span className="text-[10px] text-zinc-500 truncate block" title={form.adapter_path}>{adapterFileName}</span>
                    )}
                  </div>
                  {/* Group Scales */}
                  <div className="space-y-2">
                    <button onClick={() => setGroupsExpanded(!groupsExpanded)}
                      className="flex items-center gap-1.5 text-[10px] font-semibold text-zinc-500 hover:text-zinc-300 transition-colors uppercase tracking-wider"
                    >
                      {groupsExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                      Group Scales
                    </button>
                    {groupsExpanded && (
                      <div className="space-y-2 pl-3 border-l-2 border-pink-500/20">
                        <Slider label="Self-Attn" value={form.self_attn} min={0} max={4} step={0.05}
                          onChange={v => setForm(p => ({ ...p, self_attn: v }))} help="Temporal coherence" />
                        <Slider label="Cross-Attn" value={form.cross_attn} min={0} max={4} step={0.05}
                          onChange={v => setForm(p => ({ ...p, cross_attn: v }))} help="Prompt adherence" />
                        <Slider label="MLP" value={form.mlp} min={0} max={4} step={0.05}
                          onChange={v => setForm(p => ({ ...p, mlp: v }))} help="Timbre/tonal texture" />
                        <Slider label="Cond" value={form.cond_embed} min={0} max={4} step={0.05}
                          onChange={v => setForm(p => ({ ...p, cond_embed: v }))} help="Prompt interpretation" />
                      </div>
                    )}
                  </div>
                </div>

                <div className="border-t border-white/5" />

                {/* Reference Track Section */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-zinc-300">
                    <Music className="w-4 h-4 text-amber-400" />
                    Reference Track
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium text-zinc-400">Reference Audio</label>
                    <div className="flex gap-2">
                      <input type="text" value={form.reference_track_path}
                        onChange={e => setForm(p => ({ ...p, reference_track_path: e.target.value }))}
                        placeholder="Path to reference audio (.wav, .mp3, .flac)"
                        className="flex-1 bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-amber-500 transition-colors"
                      />
                      <button onClick={() => { setBrowserTarget('reference'); setBrowserOpen(true); }}
                        className="px-2.5 py-2 rounded-lg text-xs font-semibold bg-amber-900/20 text-amber-400 hover:bg-amber-900/30 transition-colors flex items-center gap-1 flex-shrink-0">
                        <FolderSearch size={12} /> Browse
                      </button>
                    </div>
                    {form.reference_track_path && (
                      <span className="text-[10px] text-zinc-500 truncate block" title={form.reference_track_path}>{matchFileName}</span>
                    )}
                  </div>
                  {form.reference_track_path && (
                    <Slider label="Reference Strength" value={form.audio_cover_strength} min={0} max={1} step={0.05}
                      onChange={v => setForm(p => ({ ...p, audio_cover_strength: v }))}
                      help="How strongly the reference track influences generation timbre"
                    />
                  )}
                  <p className="text-[10px] text-zinc-600">
                    Used for timbre conditioning during generation
                  </p>
                </div>
              </>
            )}
          </div>

          {/* Footer */}
          {!loading && (
            <div className="flex items-center justify-between px-6 py-4 border-t border-white/5">
              <button onClick={clear} disabled={saving}
                className="px-4 py-2 rounded-lg text-xs text-zinc-400 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-50"
              >Clear Preset</button>
              <div className="flex items-center gap-2">
                <button onClick={onClose} className="px-4 py-2 rounded-lg text-sm text-zinc-400 hover:bg-white/5 transition-colors">Cancel</button>
                <button onClick={save} disabled={saving}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 text-white text-sm font-semibold transition-all disabled:opacity-50 shadow-lg shadow-pink-500/10"
                >
                  {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                  Save
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* File Browser sub-modal */}
      <FileBrowserModal
        open={browserOpen}
        onClose={() => setBrowserOpen(false)}
        onSelect={(path) => {
          if (browserTarget === 'adapter') setForm(p => ({ ...p, adapter_path: path }));
          else setForm(p => ({ ...p, reference_track_path: path }));
          setBrowserOpen(false);
        }}
        mode="file"
        filter={browserTarget === 'reference' ? 'audio' : 'adapters'}
        title={browserTarget === 'reference' ? 'Select Reference Audio' : 'Select Adapter File'}
      />
    </>
  );
};
