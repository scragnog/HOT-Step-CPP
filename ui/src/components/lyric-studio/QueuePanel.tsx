/**
 * QueuePanel.tsx — Bulk operations panel for Lyric Studio V2.
 *
 * Modes:
 *   - Build Profiles: Queue profile builds for unprofiled albums
 *   - Generate Lyrics: Queue lyric generation for profiled albums
 *   - Assign Presets: Bulk-assign adapter + reference track presets to albums
 *   - Fetch Lyrics: Batch-fetch lyrics from Genius for new artists/albums
 *
 * Adapted for C++ engine: removed adapter type detection (not needed).
 */

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  X, Loader2, CheckCircle, AlertCircle, ListOrdered, Sparkles,
  Wand2, Settings2, FolderSearch, ChevronDown, ChevronRight,
  RefreshCw, Zap, Music, Search, Plus, Trash2, ClipboardPaste,
  LayoutList, Play, Square,
} from 'lucide-react';
import {
  useStreamingStore,
  addBulkToQueue,
  removeFromQueue,
  clearQueue,
} from '../../stores/streamingStore';
import type { QueueItem, QueueItemType } from '../../stores/streamingStore';
import { lireekApi } from '../../services/lireekApi';
import type { Artist, LyricsSet, Profile, AlbumPreset } from '../../services/lireekApi';
import { useAuth } from '../../context/AuthContext';
import { FileBrowserModal } from '../shared/FileBrowserModal';
import { EditableSlider } from '../shared/EditableSlider';

interface QueuePanelProps {
  open: boolean;
  onClose: () => void;
  artists: Artist[];
  lyricsSets: LyricsSet[];
  profiles: Profile[];
  profilingModel: { provider: string; model?: string };
  generationModel: { provider: string; model?: string };
  refinementModel: { provider: string; model?: string };
  showToast?: (msg: string) => void;
  onFetchComplete?: () => void;
}

type QueueMode = 'profile' | 'generate' | 'presets' | 'fetch-lyrics';

interface FetchEntry { artist: string; album: string; }
type FetchStatus = 'pending' | 'running' | 'done' | 'error' | 'skipped';
interface FetchQueueItem extends FetchEntry {
  id: string; status: FetchStatus; error?: string; songsFetched?: number;
}
type FetchInputMode = 'paste' | 'structured';

type PresetStatus = 'complete' | 'partial' | 'missing';

function getPresetStatus(preset?: AlbumPreset | null): PresetStatus {
  if (!preset) return 'missing';
  const hasAdapter = !!preset.adapter_path;
  const hasRef = !!preset.reference_track_path;
  if (hasAdapter && hasRef) return 'complete';
  if (hasAdapter || hasRef) return 'partial';
  return 'missing';
}

const STATUS_BADGE: Record<PresetStatus, { label: string; color: string; icon: string }> = {
  complete: { label: 'PRESET', color: 'bg-green-900/30 text-green-400', icon: '✓' },
  partial:  { label: 'PARTIAL', color: 'bg-amber-900/30 text-amber-400', icon: '⚠' },
  missing:  { label: 'NONE', color: 'bg-red-900/30 text-red-400', icon: '✕' },
};

export const QueuePanel: React.FC<QueuePanelProps> = ({
  open, onClose, artists, lyricsSets, profiles,
  profilingModel, generationModel, refinementModel, showToast,
  onFetchComplete,
}) => {
  const stream = useStreamingStore();
  const { token } = useAuth();
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [mode, setMode] = useState<QueueMode>('profile');
  const [genCount, setGenCount] = useState(4);

  // Generation counts for the "generate" tab
  const [genCountsMap, setGenCountsMap] = useState<Map<number, number>>(new Map());
  const [genCountsLoading, setGenCountsLoading] = useState(false);
  const [genFilterThreshold, setGenFilterThreshold] = useState('');

  // Presets state
  const [presetMap, setPresetMap] = useState<Map<number, AlbumPreset>>(new Map());
  const [presetsLoading, setPresetsLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [adapterPath, setAdapterPath] = useState('');
  const [matcheringPath, setMatcheringPath] = useState('');
  const [adapterScale, setAdapterScale] = useState(1.0);
  const [selfAttn, setSelfAttn] = useState(1.0);
  const [crossAttn, setCrossAttn] = useState(1.0);
  const [mlp, setMlp] = useState(1.0);
  const [condEmbed, setCondEmbed] = useState(1.0);
  const [groupsExpanded, setGroupsExpanded] = useState(false);
  const [presetFilter, setPresetFilter] = useState('');
  const [browserOpen, setBrowserOpen] = useState(false);
  const [browserTarget, setBrowserTarget] = useState<'adapter' | 'matchering'>('adapter');

  // Fetch lyrics state
  const [fetchInputMode, setFetchInputMode] = useState<FetchInputMode>('paste');
  const [pasteText, setPasteText] = useState('');
  const [structuredRows, setStructuredRows] = useState<FetchEntry[]>([{ artist: '', album: '' }]);
  const [fetchMaxSongs, setFetchMaxSongs] = useState(50);
  const [fetchQueue, setFetchQueue] = useState<FetchQueueItem[]>([]);
  const [fetchRunning, setFetchRunning] = useState(false);
  const fetchAbortRef = useRef(false);

  const loadPresets = useCallback(async () => {
    setPresetsLoading(true);
    try {
      const res = await lireekApi.listAllPresets();
      const map = new Map<number, AlbumPreset>();
      for (const p of res.presets) map.set(p.lyrics_set_id, p);
      setPresetMap(map);
    } catch (err) {
      console.error('[QueuePanel] Failed to load presets:', err);
    } finally {
      setPresetsLoading(false);
    }
  }, []);

  const loadGenerationCounts = useCallback(async () => {
    setGenCountsLoading(true);
    try {
      const res = await lireekApi.listAllGenerations();
      // API returns raw array (server) but type says { generations }, handle both
      const gens = Array.isArray(res) ? res : (res.generations || []);
      const counts = new Map<number, number>();
      for (const g of gens) {
        counts.set(g.profile_id, (counts.get(g.profile_id) || 0) + 1);
      }
      setGenCountsMap(counts);
    } catch (err) {
      console.error('[QueuePanel] Failed to load generation counts:', err);
    } finally {
      setGenCountsLoading(false);
    }
  }, []);

  if (!open) return null;

  const parsePasteText = (text: string): FetchEntry[] => {
    return text.split('\n').map(l => l.trim()).filter(l => l.length > 0 && !l.startsWith('#')).map(line => {
      let parts: string[];
      if (line.includes('|')) parts = line.split('|').map(s => s.trim());
      else if (line.includes(' - ') && !line.startsWith('http')) parts = line.split(' - ', 2).map(s => s.trim());
      else parts = [line.trim()];
      return { artist: parts[0] || '', album: parts[1] || '' };
    }).filter(e => e.artist.length > 0);
  };

  const addStructuredRow = () => setStructuredRows(prev => [...prev, { artist: '', album: '' }]);
  const removeStructuredRow = (idx: number) => setStructuredRows(prev => prev.filter((_, i) => i !== idx));
  const updateStructuredRow = (idx: number, field: 'artist' | 'album', value: string) => {
    setStructuredRows(prev => prev.map((row, i) => i === idx ? { ...row, [field]: value } : row));
  };

  const startFetchQueue = async () => {
    const entries = fetchInputMode === 'paste' ? parsePasteText(pasteText) : structuredRows.filter(r => r.artist.trim().length > 0);
    if (entries.length === 0) { showToast?.('No valid entries to fetch'); return; }

    const existingArtistAlbums = new Set(
      lyricsSets.map(ls => `${(ls.artist_name || '').toLowerCase()}|||${(ls.album || '').toLowerCase()}`)
    );

    const items: FetchQueueItem[] = entries.map((e, i) => {
      const key = `${e.artist.toLowerCase()}|||${e.album.toLowerCase()}`;
      const alreadyExists = existingArtistAlbums.has(key);
      return { ...e, id: `fetch-${Date.now()}-${i}`, status: alreadyExists ? 'skipped' as FetchStatus : 'pending' as FetchStatus, error: alreadyExists ? 'Already exists' : undefined };
    });

    setFetchQueue(items);
    setFetchRunning(true);
    fetchAbortRef.current = false;
    let completed = 0, failed = 0, skipped = 0;

    for (let i = 0; i < items.length; i++) {
      if (fetchAbortRef.current) break;
      const item = items[i];
      if (item.status === 'skipped') { skipped++; continue; }
      setFetchQueue(prev => prev.map((q, qi) => qi === i ? { ...q, status: 'running' } : q));
      try {
        const res = await lireekApi.fetchLyrics({ artist: item.artist, album: item.album || undefined, max_songs: fetchMaxSongs });
        completed++;
        setFetchQueue(prev => prev.map((q, qi) => qi === i ? { ...q, status: 'done', songsFetched: res.songs_fetched } : q));
      } catch (err: any) {
        failed++;
        setFetchQueue(prev => prev.map((q, qi) => qi === i ? { ...q, status: 'error', error: err.message || 'Fetch failed' } : q));
      }
    }

    setFetchRunning(false);
    const parts = [];
    if (completed > 0) parts.push(`${completed} fetched`);
    if (skipped > 0) parts.push(`${skipped} skipped`);
    if (failed > 0) parts.push(`${failed} failed`);
    showToast?.(parts.join(', ') || 'Queue complete');
    onFetchComplete?.();
  };

  const stopFetchQueue = () => { fetchAbortRef.current = true; };
  const clearFetchQueue = () => { if (!fetchRunning) setFetchQueue([]); };

  const fetchQueuePending = fetchQueue.filter(q => q.status === 'pending').length;
  const fetchQueueDone = fetchQueue.filter(q => q.status === 'done').length;
  const fetchQueueErrors = fetchQueue.filter(q => q.status === 'error').length;
  const fetchQueueSkipped = fetchQueue.filter(q => q.status === 'skipped').length;

  const toggleItem = (id: number) => {
    setSelected(prev => { const next = new Set(prev); next.has(id) ? next.delete(id) : next.add(id); return next; });
  };

  const selectAll = () => {
    if (mode === 'profile') {
      const profiledSetIds = new Set(profiles.map(p => p.lyrics_set_id));
      setSelected(new Set(lyricsSets.filter(ls => !profiledSetIds.has(ls.id)).map(ls => ls.id)));
    } else if (mode === 'generate') {
      const threshold = genFilterThreshold.trim() !== '' ? parseInt(genFilterThreshold) : null;
      const visible = profiles.filter(p => {
        if (threshold == null || isNaN(threshold)) return true;
        return (genCountsMap.get(p.id) || 0) < threshold;
      });
      setSelected(new Set(visible.map(p => p.id)));
    } else {
      setSelected(new Set(lyricsSets.map(ls => ls.id)));
    }
  };

  const selectMissing = () => {
    setSelected(new Set(lyricsSets.filter(ls => getPresetStatus(presetMap.get(ls.id)) === 'missing').map(ls => ls.id)));
  };

  const selectIncomplete = () => {
    setSelected(new Set(lyricsSets.filter(ls => {
      const st = getPresetStatus(presetMap.get(ls.id));
      return st === 'missing' || st === 'partial';
    }).map(ls => ls.id)));
  };

  const handleQueue = () => {
    if (selected.size === 0) return;
    if (mode === 'profile') {
      addBulkToQueue(Array.from(selected).map(lsId => {
        const ls = lyricsSets.find(l => l.id === lsId);
        return { type: 'profile' as QueueItemType, targetId: lsId, label: `Profile: ${ls?.artist_name || '?'} — ${ls?.album || 'Unknown'}`, provider: profilingModel.provider, model: profilingModel.model };
      }));
    } else {
      addBulkToQueue(Array.from(selected).map(profileId => {
        const profile = profiles.find(p => p.id === profileId);
        const ls = lyricsSets.find(l => l.id === profile?.lyrics_set_id);
        return { type: 'generate' as QueueItemType, targetId: profileId, label: `Generate: ${ls?.artist_name || '?'} — ${ls?.album || 'Unknown'}`, provider: generationModel.provider, model: generationModel.model, count: genCount };
      }));
    }
    setSelected(new Set());
  };

  const handleApplyPresets = async () => {
    if (selected.size === 0) return;
    const hasAdapter = adapterPath.trim().length > 0;
    const hasRef = matcheringPath.trim().length > 0;
    if (!hasAdapter && !hasRef) { showToast?.('Set at least one path (adapter or reference) to apply'); return; }

    setApplying(true);
    let success = 0, failed = 0;
    for (const lsId of Array.from(selected)) {
      try {
        const params: any = {};
        if (hasAdapter) {
          params.adapter_path = adapterPath.trim();
          params.adapter_scale = adapterScale;
          params.adapter_group_scales = { self_attn: selfAttn, cross_attn: crossAttn, mlp, cond_embed: condEmbed };
        }
        if (hasRef) params.reference_track_path = matcheringPath.trim();
        await lireekApi.upsertPreset(lsId, params);
        success++;
      } catch (err) {
        failed++;
        console.error(`[QueuePanel] Failed to upsert preset for ls_id=${lsId}:`, err);
      }
    }
    showToast?.(failed === 0 ? `Applied presets to ${success} album${success !== 1 ? 's' : ''}` : `Applied to ${success}, failed ${failed}`);
    setSelected(new Set());
    await loadPresets();
    setApplying(false);
  };

  const queueItems = stream.queue;
  const pendingCount = queueItems.filter(q => q.status === 'pending').length;
  const runningItem = queueItems.find(q => q.status === 'running');
  const doneCount = queueItems.filter(q => q.status === 'done').length;

  const adapterFileName = adapterPath ? adapterPath.split(/[\\/]/).pop() || '' : '';
  const matchFileName = matcheringPath ? matcheringPath.split(/[\\/]/).pop() || '' : '';

  const presetStats = {
    complete: lyricsSets.filter(ls => getPresetStatus(presetMap.get(ls.id)) === 'complete').length,
    partial: lyricsSets.filter(ls => getPresetStatus(presetMap.get(ls.id)) === 'partial').length,
    missing: lyricsSets.filter(ls => getPresetStatus(presetMap.get(ls.id)) === 'missing').length,
  };

  return (
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
        <div className="bg-zinc-900 rounded-2xl border border-white/10 shadow-2xl w-[680px] max-h-[85vh] flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
            <div className="flex items-center gap-2">
              <ListOrdered className="w-5 h-5 text-pink-400" />
              <h2 className="text-lg font-bold text-white">Bulk Operations</h2>
            </div>
            <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/10 text-zinc-400 hover:text-white transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Mode tabs */}
          <div className="flex items-center gap-2 px-6 pt-4">
            <button onClick={() => { setMode('profile'); setSelected(new Set()); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${mode === 'profile' ? 'bg-amber-500/20 text-amber-300' : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'}`}>
              <Sparkles className="w-3.5 h-3.5" /> Build Profiles
            </button>
            <button onClick={() => { setMode('generate'); setSelected(new Set()); loadGenerationCounts(); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${mode === 'generate' ? 'bg-green-500/20 text-green-300' : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'}`}>
              <Wand2 className="w-3.5 h-3.5" /> Generate Lyrics
            </button>
            <button onClick={() => { setMode('presets'); setSelected(new Set()); loadPresets(); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${mode === 'presets' ? 'bg-pink-500/20 text-pink-300' : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'}`}>
              <Settings2 className="w-3.5 h-3.5" /> Assign Presets
            </button>
            <button onClick={() => { setMode('fetch-lyrics'); setSelected(new Set()); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${mode === 'fetch-lyrics' ? 'bg-cyan-500/20 text-cyan-300' : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'}`}>
              <Search className="w-3.5 h-3.5" /> Fetch Lyrics
            </button>
          </div>

          {/* ══ Presets mode config panel ══ */}
          {mode === 'presets' && (
            <div className="px-6 pt-3 pb-2 space-y-3 border-b border-white/5">
              <div className="flex items-center gap-3 text-[10px] font-semibold">
                <span className="text-green-400">{presetStats.complete} ✓ complete</span>
                <span className="text-amber-400">{presetStats.partial} ⚠ partial</span>
                <span className="text-red-400">{presetStats.missing} ✕ missing</span>
              </div>
              <div className="relative">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-500 pointer-events-none" />
                <input type="text" value={presetFilter} onChange={e => setPresetFilter(e.target.value)}
                  placeholder="Filter by artist or album…"
                  className="w-full bg-black/20 border border-white/10 rounded-lg pl-8 pr-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-pink-500/50 transition-colors" />
                {presetFilter && (
                  <button onClick={() => setPresetFilter('')} className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 rounded hover:bg-white/10 text-zinc-500 hover:text-white transition-colors">
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
              {/* Adapter path */}
              <div className="space-y-1.5">
                <div className="flex items-center gap-2 text-xs font-semibold text-zinc-300">
                  <Zap className="w-3.5 h-3.5 text-pink-400" /> Adapter to Apply
                </div>
                <div className="flex gap-2">
                  <input type="text" value={adapterPath} onChange={e => setAdapterPath(e.target.value)}
                    placeholder="Path to .safetensors adapter file"
                    className="flex-1 bg-black/20 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-pink-500 transition-colors" />
                  <button onClick={() => { setBrowserTarget('adapter'); setBrowserOpen(true); }}
                    className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-pink-900/20 text-pink-400 hover:bg-pink-900/30 transition-colors flex items-center gap-1 flex-shrink-0">
                    <FolderSearch size={12} /> Browse
                  </button>
                </div>
                {adapterPath && (
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-pink-900/30 text-pink-400">ADAPTER</span>
                    <span className="text-[10px] text-zinc-500 truncate" title={adapterPath}>{adapterFileName}</span>
                  </div>
                )}
              </div>
              {/* Adapter scale */}
              {adapterPath && (
                <div className="space-y-1">
                  <EditableSlider label="Adapter Scale" value={adapterScale} min={0} max={4} step={0.05} onChange={setAdapterScale} formatDisplay={v => v.toFixed(2)} />
                  <button onClick={() => setGroupsExpanded(!groupsExpanded)}
                    className="flex items-center gap-1.5 text-[10px] font-semibold text-zinc-500 hover:text-zinc-300 transition-colors uppercase tracking-wider">
                    {groupsExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                    Group Scales
                  </button>
                  {groupsExpanded && (
                    <div className="space-y-1 pl-3 border-l-2 border-pink-500/20">
                      <EditableSlider label="Self-Attn" value={selfAttn} min={0} max={4} step={0.05} onChange={setSelfAttn} formatDisplay={v => v.toFixed(2)} />
                      <EditableSlider label="Cross-Attn" value={crossAttn} min={0} max={4} step={0.05} onChange={setCrossAttn} formatDisplay={v => v.toFixed(2)} />
                      <EditableSlider label="MLP" value={mlp} min={0} max={4} step={0.05} onChange={setMlp} formatDisplay={v => v.toFixed(2)} />
                      <EditableSlider label="Cond" value={condEmbed} min={0} max={4} step={0.05} onChange={setCondEmbed} formatDisplay={v => v.toFixed(2)} />
                    </div>
                  )}
                </div>
              )}
              {/* Reference track */}
              <div className="space-y-1.5">
                <div className="flex items-center gap-2 text-xs font-semibold text-zinc-300">
                  <Music className="w-3.5 h-3.5 text-amber-400" /> Reference Track to Apply
                </div>
                <div className="flex gap-2">
                  <input type="text" value={matcheringPath} onChange={e => setMatcheringPath(e.target.value)}
                    placeholder="Path to reference audio (.wav, .mp3, .flac)"
                    className="flex-1 bg-black/20 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-amber-500 transition-colors" />
                  <button onClick={() => { setBrowserTarget('matchering'); setBrowserOpen(true); }}
                    className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-amber-900/20 text-amber-400 hover:bg-amber-900/30 transition-colors flex items-center gap-1 flex-shrink-0">
                    <FolderSearch size={12} /> Browse
                  </button>
                </div>
                {matcheringPath && (
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-900/30 text-amber-400">REF</span>
                    <span className="text-[10px] text-zinc-500 truncate" title={matcheringPath}>{matchFileName}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ══ Fetch Lyrics mode config panel ══ */}
          {mode === 'fetch-lyrics' && (
            <div className="px-6 pt-3 pb-2 space-y-3 border-b border-white/5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <button onClick={() => setFetchInputMode('paste')}
                    className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-colors ${fetchInputMode === 'paste' ? 'bg-cyan-500/20 text-cyan-300' : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'}`}>
                    <ClipboardPaste className="w-3 h-3" /> Paste
                  </button>
                  <button onClick={() => setFetchInputMode('structured')}
                    className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-colors ${fetchInputMode === 'structured' ? 'bg-cyan-500/20 text-cyan-300' : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'}`}>
                    <LayoutList className="w-3 h-3" /> Rows
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-zinc-500 uppercase tracking-wider">Max songs</span>
                  <input type="number" value={fetchMaxSongs} onChange={e => setFetchMaxSongs(Math.max(1, Math.min(200, parseInt(e.target.value) || 50)))} min={1} max={200}
                    className="w-14 px-2 py-1 rounded-lg bg-black/20 border border-white/10 text-xs text-white text-center font-mono focus:outline-none focus:border-cyan-500/50 transition-all" />
                </div>
              </div>
              {fetchInputMode === 'paste' && (
                <div className="space-y-1.5">
                  <textarea value={pasteText} onChange={e => setPasteText(e.target.value)}
                    placeholder={`Paste one entry per line:\nArtist | Album\nArtist | Album\n\nAlso supports:\nArtist - Album\nArtist (fetches top songs)`}
                    rows={6} disabled={fetchRunning}
                    className="w-full bg-black/20 border border-white/10 rounded-xl px-3 py-2.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-cyan-500/40 transition-all resize-none font-mono leading-relaxed disabled:opacity-50" />
                  <p className="text-[10px] text-zinc-600">{parsePasteText(pasteText).length} entries detected · lines starting with # are ignored</p>
                </div>
              )}
              {fetchInputMode === 'structured' && (
                <div className="space-y-1.5 max-h-[200px] overflow-y-auto scrollbar-hide">
                  {structuredRows.map((row, idx) => (
                    <div key={idx} className="flex items-center gap-1.5">
                      <input type="text" value={row.artist} onChange={e => updateStructuredRow(idx, 'artist', e.target.value)} placeholder="Artist" disabled={fetchRunning}
                        className="flex-1 bg-black/20 border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-cyan-500/40 transition-all disabled:opacity-50" />
                      <input type="text" value={row.album} onChange={e => updateStructuredRow(idx, 'album', e.target.value)} placeholder="Album (optional)" disabled={fetchRunning}
                        className="flex-1 bg-black/20 border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-cyan-500/40 transition-all disabled:opacity-50" />
                      <button onClick={() => removeStructuredRow(idx)} disabled={fetchRunning || structuredRows.length <= 1}
                        className="p-1 rounded-lg text-zinc-600 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-30 disabled:pointer-events-none">
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                  <button onClick={addStructuredRow} disabled={fetchRunning}
                    className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[11px] text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/10 transition-colors disabled:opacity-50">
                    <Plus className="w-3 h-3" /> Add Row
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Selection list */}
          <div className="flex-1 overflow-y-auto px-6 py-3 space-y-1 scrollbar-hide" style={{ maxHeight: mode === 'presets' ? '250px' : '300px' }}>
            {mode === 'profile' ? (() => {
              const profiledSetIds = new Set(profiles.map(p => p.lyrics_set_id));
              const unprofiled = lyricsSets.filter(ls => !profiledSetIds.has(ls.id));
              return unprofiled.length === 0 ? (
                <p className="text-zinc-500 text-sm text-center py-4">{lyricsSets.length === 0 ? 'No albums available' : 'All albums already have profiles ✓'}</p>
              ) : (<>{unprofiled.map(ls => (
                <label key={ls.id} className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors ${selected.has(ls.id) ? 'bg-amber-500/10 border border-amber-500/20' : 'bg-white/5 border border-transparent hover:bg-white/10'}`}>
                  <input type="checkbox" checked={selected.has(ls.id)} onChange={() => toggleItem(ls.id)} className="accent-amber-500" />
                  <div className="flex-1 min-w-0">
                    <span className="text-sm text-white truncate block">{ls.album || 'Unknown Album'}</span>
                    <span className="text-[10px] text-zinc-500">{ls.artist_name}</span>
                  </div>
                </label>
              ))}</>);
            })() : mode === 'generate' ? (
              profiles.length === 0 ? (
                <p className="text-zinc-500 text-sm text-center py-4">No profiles available — build profiles first</p>
              ) : (() => {
                const threshold = genFilterThreshold.trim() !== '' ? parseInt(genFilterThreshold) : null;
                const filtered = profiles.filter(profile => {
                  if (threshold == null || isNaN(threshold)) return true;
                  const count = genCountsMap.get(profile.id) || 0;
                  return count < threshold;
                });
                return filtered.length === 0 ? (
                  <p className="text-zinc-500 text-sm text-center py-4">
                    {threshold != null ? `All ${profiles.length} profiles have ≥ ${threshold} generation${threshold !== 1 ? 's' : ''} — try a higher threshold` : 'No profiles match'}
                  </p>
                ) : (<>{filtered.map(profile => {
                  const ls = lyricsSets.find(l => l.id === profile.lyrics_set_id);
                  const genCt = genCountsMap.get(profile.id) || 0;
                  return (
                    <label key={profile.id} className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors ${selected.has(profile.id) ? 'bg-green-500/10 border border-green-500/20' : 'bg-white/5 border border-transparent hover:bg-white/10'}`}>
                      <input type="checkbox" checked={selected.has(profile.id)} onChange={() => toggleItem(profile.id)} className="accent-green-500" />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-white truncate">{ls?.album || 'Unknown Album'} — {ls?.artist_name || '?'}</span>
                          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded flex-shrink-0 ${genCt === 0 ? 'bg-red-900/30 text-red-400' : genCt < 3 ? 'bg-amber-900/30 text-amber-400' : 'bg-green-900/30 text-green-400'}`}>
                            {genCt} gen{genCt !== 1 ? 's' : ''}
                          </span>
                        </div>
                        <span className="text-[10px] text-zinc-500">{profile.provider}/{profile.model} · {new Date(profile.created_at).toLocaleDateString()}</span>
                      </div>
                    </label>
                  );
                })}</>);
              })()
            ) : mode === 'presets' ? (
              presetsLoading ? (
                <div className="flex items-center justify-center py-8"><Loader2 className="w-5 h-5 text-zinc-500 animate-spin" /></div>
              ) : lyricsSets.length === 0 ? (
                <p className="text-zinc-500 text-sm text-center py-4">No albums available</p>
              ) : (() => {
                const needle = presetFilter.toLowerCase().trim();
                const filtered = lyricsSets
                  .filter(ls => !needle || (ls.artist_name || '').toLowerCase().includes(needle) || (ls.album || '').toLowerCase().includes(needle))
                  .sort((a, b) => {
                    const cmp = (a.artist_name || '').localeCompare(b.artist_name || '', undefined, { sensitivity: 'base' });
                    return cmp !== 0 ? cmp : (a.album || '').localeCompare(b.album || '', undefined, { sensitivity: 'base' });
                  });
                return filtered.length === 0 ? (
                  <p className="text-zinc-500 text-sm text-center py-4">No albums match "{presetFilter}"</p>
                ) : (<>{filtered.map(ls => {
                  const preset = presetMap.get(ls.id);
                  const status = getPresetStatus(preset);
                  const badge = STATUS_BADGE[status];
                  return (
                    <label key={ls.id} className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors ${selected.has(ls.id) ? 'bg-pink-500/10 border border-pink-500/20' : 'bg-white/5 border border-transparent hover:bg-white/10'}`}>
                      <input type="checkbox" checked={selected.has(ls.id)} onChange={() => toggleItem(ls.id)} className="accent-pink-500" />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-white truncate">{ls.artist_name}</span>
                          <span className="text-[10px] text-zinc-600">—</span>
                          <span className="text-sm text-zinc-300 truncate">{ls.album || 'Top Songs'}</span>
                          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded flex-shrink-0 ${badge.color}`}>{badge.icon} {badge.label}</span>
                        </div>
                        <div className="flex items-center gap-2 mt-0.5">
                          {preset?.adapter_path && <span className="text-[9px] text-zinc-600 truncate max-w-[120px]" title={preset.adapter_path}>🔌 {preset.adapter_path.split(/[\\/]/).pop()}</span>}
                          {preset?.reference_track_path && <span className="text-[9px] text-zinc-600 truncate max-w-[120px]" title={preset.reference_track_path}>🎵 {preset.reference_track_path.split(/[\\/]/).pop()}</span>}
                        </div>
                      </div>
                    </label>
                  );
                })}</>);
              })()
            ) : (
              /* Fetch Lyrics queue list */
              fetchQueue.length === 0 ? (
                <p className="text-zinc-500 text-sm text-center py-4">{fetchInputMode === 'paste' ? 'Paste artist/album pairs above, then click Fetch All' : 'Add artist/album rows above, then click Fetch All'}</p>
              ) : (<>{fetchQueue.map(item => (
                <div key={item.id} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 border border-transparent">
                  {item.status === 'pending' && <div className="w-2 h-2 rounded-full bg-zinc-500 flex-shrink-0" />}
                  {item.status === 'running' && <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400 flex-shrink-0" />}
                  {item.status === 'done' && <CheckCircle className="w-3.5 h-3.5 text-green-400 flex-shrink-0" />}
                  {item.status === 'error' && <AlertCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />}
                  {item.status === 'skipped' && <span className="text-[10px] text-amber-400 flex-shrink-0">SKIP</span>}
                  <div className="flex-1 min-w-0">
                    <span className="text-sm text-white truncate block">{item.artist}</span>
                    <span className="text-[10px] text-zinc-500">
                      {item.album || 'Top songs'}
                      {item.songsFetched != null && <span className="text-green-400"> · {item.songsFetched} songs</span>}
                      {item.error && <span className="text-red-400"> · {item.error}</span>}
                    </span>
                  </div>
                </div>
              ))}</>)
            )}
          </div>

          {/* Generation count + filter */}
          {mode === 'generate' && (
            <div className="px-6 py-2 space-y-2">
              <div className="flex items-center gap-3">
                <span className="text-xs text-zinc-400">Generations per profile:</span>
                <input type="number" min={1} max={20} value={genCount}
                  onChange={e => setGenCount(Math.max(1, Math.min(20, parseInt(e.target.value) || 1)))}
                  className="w-16 px-2 py-1 rounded-lg bg-zinc-800 border border-white/10 text-sm text-white text-center focus:outline-none focus:border-green-500/50" />
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs text-zinc-400">Hide profiles with ≥</span>
                <input type="number" min={0} value={genFilterThreshold}
                  onChange={e => setGenFilterThreshold(e.target.value)}
                  placeholder="—"
                  className="w-16 px-2 py-1 rounded-lg bg-zinc-800 border border-white/10 text-sm text-white text-center focus:outline-none focus:border-green-500/50 placeholder-zinc-600" />
                <span className="text-xs text-zinc-400">existing gens</span>
                {genFilterThreshold.trim() !== '' && (
                  <button onClick={() => setGenFilterThreshold('')}
                    className="text-[10px] text-zinc-500 hover:text-zinc-300 transition-colors">clear</button>
                )}
                {genCountsLoading && <Loader2 className="w-3 h-3 animate-spin text-zinc-500" />}
              </div>
            </div>
          )}

          {/* Action bar */}
          <div className="px-6 py-3 border-t border-white/5 flex items-center justify-between">
            {mode === 'fetch-lyrics' ? (
              <>
                <div className="flex items-center gap-2">
                  {fetchQueue.length > 0 && !fetchRunning && (
                    <button onClick={clearFetchQueue} className="px-3 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-zinc-200 hover:bg-white/5 transition-colors">Clear Results</button>
                  )}
                  {fetchQueue.length > 0 && (
                    <span className="text-[10px] text-zinc-500">
                      {fetchQueueDone > 0 && <span className="text-green-400">{fetchQueueDone} done</span>}
                      {fetchQueueErrors > 0 && <span className="text-red-400"> · {fetchQueueErrors} failed</span>}
                      {fetchQueueSkipped > 0 && <span className="text-amber-400"> · {fetchQueueSkipped} skipped</span>}
                      {fetchQueuePending > 0 && <span> · {fetchQueuePending} pending</span>}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {fetchRunning ? (
                    <button onClick={stopFetchQueue} className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold bg-red-600 hover:bg-red-500 text-white transition-all">
                      <Square className="w-3.5 h-3.5" /> Stop
                    </button>
                  ) : (
                    <button onClick={startFetchQueue}
                      disabled={fetchInputMode === 'paste' ? parsePasteText(pasteText).length === 0 : structuredRows.filter(r => r.artist.trim()).length === 0}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold transition-all disabled:opacity-30 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-lg shadow-cyan-500/10">
                      <Play className="w-3.5 h-3.5" /> Fetch All
                    </button>
                  )}
                </div>
              </>
            ) : (
              <>
                <div className="flex items-center gap-2">
                  <button onClick={selectAll} className="px-3 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-zinc-200 hover:bg-white/5 transition-colors">Select All</button>
                  {mode === 'presets' && (
                    <>
                      <button onClick={selectMissing} className="px-3 py-1.5 rounded-lg text-xs text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-colors">Select Missing</button>
                      <button onClick={selectIncomplete} className="px-3 py-1.5 rounded-lg text-xs text-amber-400 hover:text-amber-300 hover:bg-amber-500/10 transition-colors">Select Incomplete</button>
                    </>
                  )}
                  <button onClick={() => setSelected(new Set())} className="px-3 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-zinc-200 hover:bg-white/5 transition-colors">Clear</button>
                </div>
                {mode === 'presets' ? (
                  <div className="flex items-center gap-2">
                    <button onClick={loadPresets} disabled={presetsLoading}
                      className="p-2 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-white/5 transition-colors disabled:opacity-50" title="Refresh preset data">
                      <RefreshCw className={`w-3.5 h-3.5 ${presetsLoading ? 'animate-spin' : ''}`} />
                    </button>
                    <button onClick={handleApplyPresets}
                      disabled={selected.size === 0 || applying || (!adapterPath.trim() && !matcheringPath.trim())}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold transition-all disabled:opacity-30 bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 text-white shadow-lg shadow-pink-500/10">
                      {applying ? <Loader2 className="w-4 h-4 animate-spin" /> : <Settings2 className="w-4 h-4" />}
                      Apply to {selected.size} Album{selected.size !== 1 ? 's' : ''}
                    </button>
                  </div>
                ) : (
                  <button onClick={handleQueue} disabled={selected.size === 0}
                    className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold transition-all disabled:opacity-30 ${mode === 'profile' ? 'bg-amber-500 text-black hover:bg-amber-400' : 'bg-green-500 text-black hover:bg-green-400'}`}>
                    <ListOrdered className="w-4 h-4" />
                    Queue {selected.size} {mode === 'profile' ? 'Profile Build' : 'Generation Run'}{selected.size !== 1 ? 's' : ''}
                  </button>
                )}
              </>
            )}
          </div>

          {/* Queue status */}
          {queueItems.length > 0 && (
            <div className="px-6 py-3 border-t border-white/5">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Queue Progress</span>
                <button onClick={clearQueue} className="text-[10px] text-zinc-500 hover:text-red-400 transition-colors">Clear Finished</button>
              </div>
              <div className="space-y-1 max-h-32 overflow-y-auto scrollbar-hide">
                {queueItems.map(item => (
                  <div key={item.id} className="flex items-center gap-2 px-2 py-1 rounded-lg bg-white/5 text-xs">
                    {item.status === 'pending' && <div className="w-2 h-2 rounded-full bg-zinc-500" />}
                    {item.status === 'running' && <Loader2 className="w-3 h-3 animate-spin text-pink-400" />}
                    {item.status === 'done' && <CheckCircle className="w-3 h-3 text-green-400" />}
                    {item.status === 'error' && <AlertCircle className="w-3 h-3 text-red-400" />}
                    <span className="text-zinc-300 flex-1 truncate">{item.label}</span>
                    {item.count && item.count > 1 && <span className="text-[10px] text-zinc-500">{item.countCompleted || 0}/{item.count}</span>}
                    {item.status === 'pending' && (
                      <button onClick={() => removeFromQueue(item.id)} className="p-0.5 rounded hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-colors">
                        <X className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
              {(pendingCount > 0 || runningItem) && (
                <div className="mt-2 text-[10px] text-zinc-500">
                  {runningItem ? `Running: ${runningItem.label}` : ''} 
                  {pendingCount > 0 ? ` · ${pendingCount} pending` : ''}
                  {doneCount > 0 ? ` · ${doneCount} done` : ''}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* File Browser sub-modal */}
      <FileBrowserModal
        open={browserOpen}
        onClose={() => setBrowserOpen(false)}
        onSelect={(path) => {
          if (browserTarget === 'adapter') setAdapterPath(path);
          else setMatcheringPath(path);
          setBrowserOpen(false);
        }}
        mode="file"
        filter={browserTarget === 'matchering' ? 'audio' : 'adapters'}
        title={browserTarget === 'matchering' ? 'Select Reference Audio' : 'Select Adapter File'}
      />
    </>
  );
};
