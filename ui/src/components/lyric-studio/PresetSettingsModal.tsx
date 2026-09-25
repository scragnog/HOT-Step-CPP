import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { X, Save, Loader2, ChevronDown, ChevronRight, Zap, Music, FolderSearch, FolderOpen, Search, Brain } from 'lucide-react';
import { lireekApi, type Mm3PresetAdapter } from '../../services/lireekApi';
import { FileBrowserModal } from '../shared/FileBrowserModal';
import { StyledSelect, type SelectOption } from '../shared/StyledSelect';
import { useBackendStore } from '../../stores/backendStore';
import { MM3_BACKEND_ID } from '../../utils/captionForBackend';
import { YUE2_BACKEND_ID } from '../../utils/yue2CaptionSource';

interface PresetForm {
  adapter_path: string;
  self_attn: number;
  cross_attn: number;
  mlp: number;
  cond_embed: number;
  reference_track_path: string;
  lm_adapter_path: string;
  /** MM3 LM adapter, relative to the mm3-lm-adapters root ('' = base model). */
  mm3_adapter_path: string;
  /** YuE2's two halves, absolute paths ('' = that half runs unmerged). */
  yue2_ar_adapter_path: string;
  yue2_nar_adapter_path: string;
}

const DEFAULT_FORM: PresetForm = {
  adapter_path: '',
  self_attn: 1.0,
  cross_attn: 1.0,
  mlp: 1.0,
  cond_embed: 1.0,
  reference_track_path: '',
  lm_adapter_path: '',
  mm3_adapter_path: '',
  yue2_ar_adapter_path: '',
  yue2_nar_adapter_path: '',
};

/** "run-stamp · ckpt-300" — the part of an MM3 adapter reference a human recognises. */
const mm3Label = (a: Mm3PresetAdapter): string =>
  `${a.run}${a.ckpt ? ' · ' + a.ckpt : ''}${a.trainedSteps && !a.ckpt ? ` · ${a.trainedSteps} steps` : ''}`;

interface PresetSettingsModalProps {
  isOpen: boolean;
  lyricsSetId: number;
  albumName: string;
  onClose: () => void;
  showToast: (msg: string) => void;
}

/** The weights file is always adapter_model.safetensors, so a pasted or
 *  browsed file path normalises to its folder — the only part that matters. */
const stripWeightsFile = (p: string): string =>
  p.replace(/[\\/]adapter_model\.safetensors$/i, '');

// Simple inline slider
const Slider: React.FC<{
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; help?: string;
}> = ({ label, value, min, max, step, onChange, help }) => (
  <div className="space-y-1">
    <div className="flex items-center justify-between">
      <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{label}</label>
      <span className="text-xs text-zinc-500 font-mono">{value.toFixed(2)}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value}
      onChange={e => onChange(parseFloat(e.target.value))}
      className="w-full h-1.5 bg-zinc-100 dark:bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-pink-500"
    />
    {help && <p className="text-[10px] text-zinc-600">{help}</p>}
  </div>
);

export const PresetSettingsModal: React.FC<PresetSettingsModalProps> = ({
  isOpen, lyricsSetId, albumName, onClose, showToast,
}) => {
  const [form, setForm] = useState<PresetForm>(DEFAULT_FORM);
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [groupsExpanded, setGroupsExpanded] = useState(false);
  const [browserOpen, setBrowserOpen] = useState(false);
  const [browserTarget, setBrowserTarget] = useState<'adapter' | 'lmAdapter' | 'reference' | 'yue2Folder'>('adapter');
  // MM3 mode shows the MM3 adapter for this album instead of the two ACE
  // adapters (2026-09-11): the preset row is shared, the backends' adapters
  // are not interchangeable, so each mode edits its own column.
  const mm3Mode = useBackendStore(s => s.activeBackendId) === MM3_BACKEND_ID;
  const yue2Mode = useBackendStore(s => s.activeBackendId) === YUE2_BACKEND_ID;
  // The installed YuE2 adapters, straight off the catalogue the global picker
  // reads — there is no per-album lookup route for YuE2 the way there is for
  // MM3, and the catalogue already records each file's half and trigger.
  const yue2Catalogue = useBackendStore(s => s.models[YUE2_BACKEND_ID] ?? null);
  const fetchBackendModels = useBackendStore(s => s.fetchModels);
  const selectModels = useBackendStore(s => s.selectModels);
  useEffect(() => {
    if (yue2Mode && !yue2Catalogue) void fetchBackendModels(YUE2_BACKEND_ID);
  }, [yue2Mode, yue2Catalogue, fetchBackendModels]);
  const [mm3Adapters, setMm3Adapters] = useState<{
    datasetSlug: string | null; candidates: Mm3PresetAdapter[]; others: Mm3PresetAdapter[];
  } | null>(null);

  // Repointing which folder the two YuE2 halves are picked from — same folder
  // scan the global Adapters picker uses (Yue2LmAdapterDropdown), so adapters
  // moved into a subfolder (e.g. .../refined) show up here too.
  const [yue2Folder, setYue2Folder] = useState('');
  const [yue2FolderBusy, setYue2FolderBusy] = useState(false);
  const [yue2FolderNote, setYue2FolderNote] = useState<string | null>(null);
  useEffect(() => {
    const saved = (yue2Catalogue?.defaults as Record<string, unknown> | undefined)?.lmAdapterFolder;
    if (typeof saved === 'string') setYue2Folder(saved);
  }, [yue2Catalogue]);

  const scanYue2Folder = async (dir: string) => {
    const trimmed = dir.trim();
    setYue2Folder(dir);
    setYue2FolderBusy(true);
    setYue2FolderNote(null);
    const ok = await selectModels({ lmAdapterFolder: trimmed }, YUE2_BACKEND_ID);
    setYue2FolderBusy(false);
    if (!ok) {
      setYue2FolderNote(t('lyric.yue2AdapterFolderFailed',
        'That folder could not be used. It needs a full path to a folder that exists.'));
      return;
    }
    // selectModels already refetched the catalogue — read it fresh rather than
    // the stale closure over this render's yue2Catalogue.
    const cat = useBackendStore.getState().models[YUE2_BACKEND_ID];
    const meta = cat?.lmAdapterMeta ?? {};
    const all = cat?.lmAdapters ?? [];
    const norm = (p: string) => p.toLowerCase().replace(/\\/g, '/');
    const folderNorm = norm(trimmed);
    const inFolder = all.filter(p => norm(p).startsWith(folderNorm));
    const pickBest = (half: 'ar' | 'nar'): string | null => {
      const candidates = inFolder.filter(p => (meta[p] as { kind?: string } | undefined)?.kind === half);
      if (candidates.length === 0) return null;
      return candidates.reduce((best, p) => {
        const bestSteps = (meta[best] as { steps?: number } | undefined)?.steps ?? -1;
        const pSteps = (meta[p] as { steps?: number } | undefined)?.steps ?? -1;
        return pSteps > bestSteps ? p : best;
      }, candidates[0]);
    };
    const arPick = pickBest('ar');
    const narPick = pickBest('nar');
    setForm(p => ({
      ...p,
      yue2_ar_adapter_path: arPick ?? p.yue2_ar_adapter_path,
      yue2_nar_adapter_path: narPick ?? p.yue2_nar_adapter_path,
    }));
    if (!arPick && !narPick) {
      setYue2FolderNote(t('lyric.yue2AdapterFolderNoneFound', 'No AR/NAR adapter found under that folder.'));
    }
  };

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
            lm_adapter_path: res.preset.lm_adapter_path || '',
            mm3_adapter_path: res.preset.mm3_adapter_path || '',
            yue2_ar_adapter_path: res.preset.yue2_ar_adapter_path || '',
            yue2_nar_adapter_path: res.preset.yue2_nar_adapter_path || '',
          });
        } else {
          setForm(DEFAULT_FORM);
        }
      })
      .catch(err => showToast(`Failed to load preset: ${err.message}`))
      .finally(() => setLoading(false));
  }, [isOpen, lyricsSetId, showToast]);

  // The MM3 adapters that belong to this album (runs on the dataset it was
  // exported from), plus every other installed one. Advisory: a failure here
  // leaves the picker empty, never blocks the modal.
  useEffect(() => {
    if (!isOpen || !mm3Mode) return;
    let cancelled = false;
    setMm3Adapters(null);
    lireekApi.mm3AdaptersForLyricsSet(lyricsSetId)
      .then(res => { if (!cancelled) setMm3Adapters(res); })
      .catch(() => { if (!cancelled) setMm3Adapters({ datasetSlug: null, candidates: [], others: [] }); });
    return () => { cancelled = true; };
  }, [isOpen, mm3Mode, lyricsSetId]);

  const save = async () => {
    setSaving(true);
    try {
      await lireekApi.upsertPreset(lyricsSetId, {
        // Normalise typed/pasted file paths to their folder on the way out too.
        adapter_path: stripWeightsFile(form.adapter_path) || undefined,
        adapter_group_scales: { self_attn: form.self_attn, cross_attn: form.cross_attn, mlp: form.mlp, cond_embed: form.cond_embed },
        reference_track_path: form.reference_track_path || undefined,
        lm_adapter_path: stripWeightsFile(form.lm_adapter_path) || undefined,
        mm3_adapter_path: form.mm3_adapter_path || undefined,
        yue2_ar_adapter_path: form.yue2_ar_adapter_path || undefined,
        yue2_nar_adapter_path: form.yue2_nar_adapter_path || undefined,
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

  // Last two path segments — for a per-run folder that reads "artist/stamp",
  // which is the part a human recognises.
  const adapterFileName = form.adapter_path
    ? form.adapter_path.split(/[\\/]/).filter(Boolean).slice(-2).join('/')
    : '';
  const matchFileName = form.reference_track_path ? form.reference_track_path.split(/[\\/]/).pop() || '' : '';

  return (
    <>
      <div className="fixed inset-0 z-50 bg-black/30 dark:bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
        <div className="w-full max-w-lg rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 shadow-2xl pointer-events-auto" onClick={(e) => e.stopPropagation()}>
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-white/5">
            <div>
              <h2 className="text-base font-bold text-white">{t('lyric.albumPreset')}</h2>
              <p className="text-xs text-zinc-500 mt-0.5">{albumName || 'Top Songs'}</p>
            </div>
            <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/5 text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Content */}
          <div className="px-6 py-5 space-y-5 max-h-[70vh] overflow-y-auto">
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-6 h-6 text-zinc-500 animate-spin" />
              </div>
            ) : yue2Mode ? (
              <>
                {/* YuE2 is TWO adapters, so the preset stores two. Both are
                    offered because a stack needs both halves: the AR plans the
                    song and carries the artist likeness, the NAR renders those
                    tokens to audio. Selecting an album applies the pair to the
                    engine — on this backend the adapter is merged into the
                    resident LM, so it is engine state rather than a per-request
                    field. Strengths stay in the global picker's dials. */}
                <div className="space-y-2">
                  <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
                    {t('lyric.yue2AdapterFolder', 'Adapter folder')}
                  </label>
                  <div className="flex gap-2">
                    <input type="text" value={yue2Folder}
                      onChange={e => setYue2Folder(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter') void scanYue2Folder(yue2Folder); }}
                      disabled={yue2FolderBusy}
                      placeholder="Folder of AR/NAR adapters, e.g. …\yue2-joint-adapters\refined"
                      className="flex-1 bg-zinc-200 dark:bg-black/20 border border-zinc-300 dark:border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-pink-500 transition-colors disabled:opacity-50"
                    />
                    <button onClick={() => void scanYue2Folder(yue2Folder)} disabled={yue2FolderBusy}
                      className="px-2.5 py-2 rounded-lg text-xs font-semibold bg-pink-900/20 text-pink-400 hover:bg-pink-900/30 transition-colors flex items-center gap-1 flex-shrink-0 disabled:opacity-50">
                      {yue2FolderBusy ? <Loader2 size={12} className="animate-spin" /> : <Search size={12} />}
                    </button>
                    <button onClick={() => { setBrowserTarget('yue2Folder'); setBrowserOpen(true); }} disabled={yue2FolderBusy}
                      className="px-2.5 py-2 rounded-lg text-xs font-semibold bg-pink-900/20 text-pink-400 hover:bg-pink-900/30 transition-colors flex items-center gap-1 flex-shrink-0 disabled:opacity-50">
                      <FolderOpen size={12} />
                    </button>
                  </div>
                  <p className="text-[10px] text-zinc-600">
                    {t('lyric.yue2AdapterFolderHint',
                      'Point this at a folder of adapters, e.g. one you moved under yue2-joint-adapters\\refined; every AR and NAR file inside it, including subfolders, is added to the lists below and the two halves are filled in from it.')}
                  </p>
                  {yue2FolderNote && <p className="text-[10px] text-amber-500/90">{yue2FolderNote}</p>}
                </div>
                {(['ar', 'nar'] as const).map(half => {
                  const field = half === 'ar' ? 'yue2_ar_adapter_path' : 'yue2_nar_adapter_path';
                  const value = form[field];
                  const meta = yue2Catalogue?.lmAdapterMeta ?? {};
                  const all = yue2Catalogue?.lmAdapters ?? [];
                  // An entry whose half the server did not record is shown in
                  // BOTH lists, for the same reason the global picker does it:
                  // the engine's format gate is the real authority, and hiding
                  // a file someone trained is worse than letting them try it.
                  const paths = all.filter(p2 => {
                    const k = (meta[p2] as { kind?: string } | undefined)?.kind;
                    return k === undefined || k === half;
                  });
                  const label = (p2: string) => {
                    const m = meta[p2];
                    const bits = [m?.trigger || m?.runName || p2.split(/[\\/]/).pop()];
                    if (m?.steps) bits.push(`${m.steps} steps`);
                    return bits.filter(Boolean).join(' · ');
                  };
                  const options: Array<SelectOption<string>> = [
                    { value: '', label: t('lyric.yue2AdapterNone', 'None — base model') },
                    ...(value && !paths.includes(value)
                      ? [{ value, label: `${value} (${t('lyric.yue2AdapterMissing', 'not installed')})` }]
                      : []),
                    ...paths.map(p2 => ({ value: p2, label: label(p2), hint: meta[p2]?.trigger || meta[p2]?.runName ? p2 : undefined })),
                  ];
                  return (
                    <div className="space-y-3" key={half}>
                      <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                        <Brain className="w-4 h-4 text-emerald-400" />
                        {half === 'ar'
                          ? t('lyric.yue2ArAdapter', 'YuE2 AR adapter (plans the song)')
                          : t('lyric.yue2NarAdapter', 'YuE2 NAR adapter (renders the audio)')}
                      </div>
                      <div className="space-y-2">
                        <StyledSelect value={value} onChange={v => setForm(p2 => ({ ...p2, [field]: v }))}
                          options={options} accent="pink" className="w-full" searchable={paths.length >= 8} />
                        {value && (
                          <span className="text-[10px] text-zinc-500 truncate block" title={value}>{value}</span>
                        )}
                      </div>
                    </div>
                  );
                })}
                <p className="text-[10px] text-zinc-600">
                  {t('lyric.yue2AdapterHint',
                    'A finished YuE2 training run on this album’s dataset fills these in by itself — the AR at its '
                    + 'pick rung rather than its last, since the likeness curve can be trained past. Strength lives in the '
                    + 'global adapter menu.')}
                </p>
              </>
            ) : mm3Mode ? (
              <>
                {/* MM3 Adapter Section — the one adapter that exists for this backend */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                    <Brain className="w-4 h-4 text-emerald-400" />
                    {t('lyric.mm3Adapter', 'MM3 Adapter')}
                  </div>
                  <div className="space-y-2">
                    <StyledSelect
                      value={form.mm3_adapter_path}
                      onChange={v => setForm(p => ({ ...p, mm3_adapter_path: v }))}
                      accent="pink"
                      className="w-full"
                      searchable={!!mm3Adapters && mm3Adapters.candidates.length + mm3Adapters.others.length >= 8}
                      options={[
                        { value: '', label: t('lyric.mm3AdapterNone', 'None — base model') },
                        ...(form.mm3_adapter_path && mm3Adapters
                          && ![...mm3Adapters.candidates, ...mm3Adapters.others].some(a => a.file === form.mm3_adapter_path)
                          ? [{ value: form.mm3_adapter_path, label: `${form.mm3_adapter_path} (${t('lyric.mm3AdapterMissing', 'not installed')})` }]
                          : []),
                        ...(mm3Adapters?.candidates ?? []).map(a => ({
                          value: a.file, label: mm3Label(a), hint: t('lyric.mm3AdapterTrainedHere', 'Trained on this album') as string,
                        })),
                        ...(mm3Adapters?.others ?? []).map(a => ({
                          value: a.file, label: mm3Label(a), hint: t('lyric.mm3AdapterOthers', 'Other installed adapter') as string,
                        })),
                      ]}
                    />
                    {form.mm3_adapter_path && (
                      <span className="text-[10px] text-zinc-500 truncate block" title={form.mm3_adapter_path}>{form.mm3_adapter_path}</span>
                    )}
                    <p className="text-[10px] text-zinc-600">
                      {mm3Adapters === null
                        ? t('lyric.mm3AdapterLoading', 'Looking up this album’s training runs…')
                        : mm3Adapters.datasetSlug
                          ? t('lyric.mm3AdapterHint', 'Newest run first. A finished MM3 training run on this album’s dataset ({{slug}}) selects its final checkpoint here automatically. Strength comes from the global LM Adapter menu.', { slug: mm3Adapters.datasetSlug })
                          : t('lyric.mm3AdapterNoDataset', 'This album was not exported from a training dataset, so no run is linked to it; pick any installed adapter.')}
                    </p>
                  </div>
                </div>
                {/* No reference track here: the timbre reference is an ACE-Step
                    conditioning input (generate.ts) and the MM3 runner never
                    reads it. The column is kept for the ACE view of this preset. */}
              </>
            ) : (
              <>
                {/* Adapter Section */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                    <Zap className="w-4 h-4 text-pink-400" />
                    {t('lyric.adapter')}
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">Adapter Folder</label>
                    <div className="flex gap-2">
                      <input type="text" value={form.adapter_path}
                        onChange={e => setForm(p => ({ ...p, adapter_path: e.target.value }))}
                        placeholder="Folder containing adapter_model.safetensors"
                        className="flex-1 bg-zinc-200 dark:bg-black/20 border border-zinc-300 dark:border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-pink-500 transition-colors"
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
                      className="flex items-center gap-1.5 text-[10px] font-semibold text-zinc-500 hover:text-zinc-700 dark:text-zinc-300 transition-colors uppercase tracking-wider"
                    >
                      {groupsExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                      {t('lyric.groupScales')}
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

                <div className="border-t border-zinc-200 dark:border-white/5" />

                {/* Planner Adapter (LM) Section — song structure */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                    <Brain className="w-4 h-4 text-violet-400" />
                    Planner Adapter (LM)
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">Adapter Folder</label>
                    <div className="flex gap-2">
                      <input type="text" value={form.lm_adapter_path}
                        onChange={e => setForm(p => ({ ...p, lm_adapter_path: e.target.value }))}
                        placeholder="Folder containing adapter_model.safetensors — empty = base planner"
                        className="flex-1 bg-zinc-200 dark:bg-black/20 border border-zinc-300 dark:border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-violet-500 transition-colors"
                      />
                      <button onClick={() => { setBrowserTarget('lmAdapter'); setBrowserOpen(true); }}
                        className="px-2.5 py-2 rounded-lg text-xs font-semibold bg-violet-900/20 text-violet-400 hover:bg-violet-900/30 transition-colors flex items-center gap-1 flex-shrink-0">
                        <FolderSearch size={12} /> Browse
                      </button>
                    </div>
                    <p className="text-[10px] text-zinc-600">
                      Shapes song structure/phrasing via the 5Hz planner — pairs with the DiT adapter above (same trigger word).
                      Strength comes from the global Adapters menu, like the DiT adapter scale.
                    </p>
                  </div>
                </div>

                <div className="border-t border-zinc-200 dark:border-white/5" />

                {/* Reference Track Section */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                    <Music className="w-4 h-4 text-amber-400" />
                    {t('lyric.referenceTrack')}
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">Reference Audio</label>
                    <div className="flex gap-2">
                      <input type="text" value={form.reference_track_path}
                        onChange={e => setForm(p => ({ ...p, reference_track_path: e.target.value }))}
                        placeholder="Path to reference audio (.wav, .mp3, .flac)"
                        className="flex-1 bg-zinc-200 dark:bg-black/20 border border-zinc-300 dark:border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-amber-500 transition-colors"
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

                  <p className="text-[10px] text-zinc-600">
                    Used for timbre conditioning during generation
                  </p>
                </div>
              </>
            )}
          </div>

          {/* Footer */}
          {!loading && (
            <div className="flex items-center justify-between px-6 py-4 border-t border-zinc-200 dark:border-white/5">
              <button onClick={clear} disabled={saving}
                className="px-4 py-2 rounded-lg text-xs text-zinc-600 dark:text-zinc-400 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-50"
              >{t('lyric.clearPreset')}</button>
              <div className="flex items-center gap-2">
                <button onClick={onClose} className="px-4 py-2 rounded-lg text-sm text-zinc-600 dark:text-zinc-400 hover:bg-white/5 transition-colors">{t('common.cancel')}</button>
                <button onClick={save} disabled={saving}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 text-white text-sm font-semibold transition-all disabled:opacity-50 shadow-lg shadow-pink-500/10"
                >
                  {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                  {t('common.save')}
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
          // Both adapter targets take the FOLDER — the weights file inside is
          // always adapter_model.safetensors, so the filename carries nothing.
          if (browserTarget === 'adapter') setForm(p => ({ ...p, adapter_path: stripWeightsFile(path) }));
          else if (browserTarget === 'lmAdapter') setForm(p => ({ ...p, lm_adapter_path: stripWeightsFile(path) }));
          else if (browserTarget === 'yue2Folder') void scanYue2Folder(path);
          else setForm(p => ({ ...p, reference_track_path: path }));
          setBrowserOpen(false);
        }}
        mode={browserTarget === 'reference' ? 'file' : 'folder'}
        filter={browserTarget === 'reference' ? 'audio' : 'adapters'}
        title={browserTarget === 'reference' ? 'Select Reference Audio'
          : browserTarget === 'lmAdapter' ? 'Select Planner Adapter Folder'
          : browserTarget === 'yue2Folder' ? 'Select YuE2 adapter folder'
          : 'Select Adapter Folder'}
      />
    </>
  );
};
