// StemBuilder.tsx — Main Stem Builder orchestrator
//
// Lego mode UI: generates new instrument tracks layered over backing audio.
// Supports iterative composition: output → new source → add another layer.
//
// Layout follows the same 3-column pattern as CoverStudio / StemStudio:
// [Source + Track] | [Build Controls + Layer Stack] | [Recent + Queue]
//
// Engine requirements:
//   - task_type: 'lego'
//   - Base model only (turbo/SFT not supported)
//   - Source audio required
//   - Single track name required

import React, { useState, useEffect, useCallback } from 'react';
import { Upload, X, Loader2, Info, AlertTriangle, Layers, Clock, ListOrdered } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../../context/AuthContext';
import { useGlobalParamsStore } from '../../context/GlobalParamsContext';
import { usePersistedState } from '../../hooks/usePersistedState';
import { generateApi, modelApi } from '../../services/api';
import { TrackPicker, type TrackName } from './TrackPicker';
import { LayerStack, type LayerInfo } from './LayerStack';
import { RecentBuilds } from './RecentBuilds';
import { PreviewPlayer } from './PreviewPlayer';
import { Section } from '../shared/ActivitySidebar';
import { InlineAudioQueue } from '../lyric-studio/InlineAudioQueue';
import { useAudioGenQueueSelector } from '../../stores/audioGenQueueStore';

/** Filter DiT model list to only pure base models (no merge/sft/turbo) */
function getBaseModels(ditModels: string[]): string[] {
  return ditModels.filter(m => {
    const lower = m.toLowerCase();
    return lower.startsWith('acestep-v15-base-') || lower.startsWith('acestep-v15-xl-base-');
  });
}

export const StemBuilder: React.FC = () => {
  const { token } = useAuth();
  const { t } = useTranslation();
  const gp = useGlobalParamsStore();

  // ── Source audio ──
  const [sourceAudioUrl, setSourceAudioUrl] = useState(() => localStorage.getItem('hs-sb-sourceUrl') || '');
  const [sourceFileName, setSourceFileName] = useState(() => localStorage.getItem('hs-sb-sourceFile') || '');
  const [isUploading, setIsUploading] = useState(false);

  // ── Track selection ──
  const [selectedTrack, setSelectedTrack] = useState<TrackName | null>(null);

  // ── Style hint (collapsible) ──
  const [caption, setCaption] = useState('');

  // ── Model selection ──
  const [baseModels, setBaseModels] = useState<string[]>([]);
  const [buildModel, setBuildModel] = useState<string>('');
  const [modelsLoading, setModelsLoading] = useState(true);

  // ── Generation state ──
  const [isGenerating, setIsGenerating] = useState(false);
  const [genProgress, setGenProgress] = useState(0);
  const [genStage, setGenStage] = useState('');
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  // ── Iterative composition ──
  const [layers, setLayers] = useState<LayerInfo[]>([]);

  // ── Preview player ──
  const [previewStemUrl, setPreviewStemUrl] = useState('');
  const [previewLabel, setPreviewLabel] = useState('');

  // ── Sidebar ──
  const [sidebarWidth, setSidebarWidth] = usePersistedState('hs-activitySidebarWidth', 320);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const queueCount = useAudioGenQueueSelector(s =>
    s.items.filter(i => i.status === 'pending' || i.status === 'loading-adapter' || i.status === 'generating').length
  );
  const completionCounter = useAudioGenQueueSelector(s => s.completionCounter);

  // ── Toast ──
  const [toast, setToast] = useState('');
  const showToast = (msg: string) => { setToast(msg); setTimeout(() => setToast(''), 4000); };

  // ── Persist source ──
  useEffect(() => { localStorage.setItem('hs-sb-sourceUrl', sourceAudioUrl); }, [sourceAudioUrl]);
  useEffect(() => { localStorage.setItem('hs-sb-sourceFile', sourceFileName); }, [sourceFileName]);

  // ── Persist model ──
  useEffect(() => {
    if (buildModel) localStorage.setItem('hs-sb-model', buildModel);
  }, [buildModel]);

  // ── Load base models on mount ──
  useEffect(() => {
    modelApi.list()
      .then(data => {
        const base = getBaseModels(data.models.dit || []);
        setBaseModels(base);
        if (base.length > 0) {
          const stored = localStorage.getItem('hs-sb-model');
          setBuildModel(stored && base.includes(stored) ? stored : base[0]);
        }
      })
      .catch(err => console.error('[StemBuilder] Failed to load models:', err))
      .finally(() => setModelsLoading(false));
  }, []);

  // ── File upload ──
  const handleFileSelected = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = ''; // reset so same file can be re-selected

    setSourceFileName(file.name);
    setIsUploading(true);
    setLayers([]); // reset layer stack on new source

    try {
      const fd = new FormData();
      fd.append('audio', file);
      const res = await fetch('/api/upload/audio', { method: 'POST', body: fd });
      if (!res.ok) throw new Error('Upload failed');
      const data = await res.json();
      setSourceAudioUrl(data.audio_url || '');
      showToast('Source audio uploaded!');
    } catch (err: any) {
      showToast(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  }, []);

  const handleClearSource = () => {
    setSourceAudioUrl('');
    setSourceFileName('');
    setLayers([]);
    setSelectedTrack(null);
    setCaption('');
  };

  // ── Use layer output as new source (iterative composition) ──
  const handleUseAsSource = useCallback((layer: LayerInfo) => {
    setSourceAudioUrl(layer.audioUrl);
    setSourceFileName(`Layer: ${layer.trackName}`);
    setSelectedTrack(null);
    setCaption('');
    showToast(`Using ${layer.trackName} output as new source`);
  }, []);

  // ── Preview layer audio (opens in dual-track player) ──
  const handlePlayLayer = useCallback((layer: LayerInfo) => {
    setPreviewStemUrl(layer.audioUrl);
    setPreviewLabel(layer.trackName.replace('_', ' '));
  }, []);

  // ── Use recent build as source ──
  const handleUseRecentAsSource = useCallback((build: { audioUrl: string; title: string }) => {
    setSourceAudioUrl(build.audioUrl);
    setSourceFileName(build.title);
    setLayers([]);
    setSelectedTrack(null);
    showToast(`Loaded "${build.title}" as source`);
  }, []);

  // ── Preview recent build (opens in dual-track player) ──
  const handlePlayRecent = useCallback((build: { audioUrl: string; trackName?: string }) => {
    setPreviewStemUrl(build.audioUrl);
    setPreviewLabel((build as any).trackName?.replace('_', ' ') || 'stem');
  }, []);

  // ── Generate ──
  const handleGenerate = async () => {
    if (!token || !sourceAudioUrl || !selectedTrack || !buildModel) {
      showToast('Missing source audio, track, or model');
      return;
    }

    setIsGenerating(true);
    setGenProgress(0);
    setGenStage('Submitting...');

    try {
      const engineParams = gp.getGlobalParams();

      const params: Record<string, any> = {
        ...engineParams,
        customMode: true,
        taskType: 'lego',
        trackName: selectedTrack,
        sourceAudioUrl,
        caption: caption || '',
        lyrics: selectedTrack === 'vocals' ? '' : '[Instrumental]',
        duration: 0,      // locked to source length by engine
        instrumental: selectedTrack !== 'vocals',
        source: 'stem-builder',
        title: `${selectedTrack.replace('_', ' ')} layer`,
        // Force base model — override whatever's in global params
        ditModel: buildModel,
        // ── Force-disable adapters (same isolation as StemStudio extract) ──
        loraPath: '',
        loraScale: 0,
        adapterGroupScales: undefined,
        adapterMode: undefined,
        // ── Force-disable mastering/timbre (not applicable to lego) ──
        masteringEnabled: false,
        masteringReference: undefined,
        timbreReference: undefined,
        // ── Source conditioning ──
        audioCoverStrength: 1.0,  // full source conditioning for lego
        // ── Clear metadata — let engine infer from source audio ──
        bpm: 0,
        keyScale: '',
        timeSignature: '',
      };

      const res = await generateApi.submit(params as any, token);
      const jobId = res.jobId;
      setActiveJobId(jobId);
      showToast(`Building ${selectedTrack} layer...`);

      // Poll for progress
      const iv = setInterval(async () => {
        try {
          const s = await generateApi.status(jobId);
          const rawProg = s.progress;
          const pct = rawProg != null
            ? Math.min(100, Math.max(0, Math.round(rawProg > 1 ? rawProg : rawProg * 100)))
            : undefined;
          if (pct != null) setGenProgress(pct);
          if (s.stage) setGenStage(s.stage);

          if (s.status === 'succeeded') {
            clearInterval(iv);
            setGenProgress(100);
            setGenStage('Complete!');
            setIsGenerating(false);
            setActiveJobId(null);
            setRefreshTrigger(p => p + 1);

            // Add to layer stack + auto-open preview
            const audioUrl = s.result?.audioUrls?.[0] || '';
            if (audioUrl) {
              setLayers(prev => [...prev, {
                trackName: selectedTrack,
                caption,
                audioUrl,
                songId: s.result?.songIds?.[0],
                timestamp: Date.now(),
              }]);
              // Auto-open preview player with the new stem
              setPreviewStemUrl(audioUrl);
              setPreviewLabel(selectedTrack.replace('_', ' '));
            }

            showToast(`${selectedTrack} layer complete!`);
            setTimeout(() => { setGenProgress(0); setGenStage(''); }, 3000);
          } else if (s.status === 'failed') {
            clearInterval(iv);
            setIsGenerating(false);
            setActiveJobId(null);
            setGenProgress(0);
            setGenStage('');
            showToast(`Failed: ${s.error || 'Unknown error'}`);
          }
        } catch { /* polling error — keep trying */ }
      }, 2000);

      // Safety timeout
      setTimeout(() => clearInterval(iv), 1_800_000);
    } catch (err: any) {
      showToast(`Generation failed: ${err.message}`);
      setIsGenerating(false);
    }
  };

  const handleCancel = async () => {
    if (activeJobId) {
      try { await generateApi.cancel(activeJobId); } catch { /* ignore */ }
    }
    setIsGenerating(false);
    setActiveJobId(null);
    setGenProgress(0);
    setGenStage('');
  };

  const canGenerate = !!sourceAudioUrl && !!selectedTrack && !!buildModel && !isGenerating;

  // ── Sidebar resize ──
  const handleSidebarResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = sidebarWidth;
    const onMove = (ev: MouseEvent) => {
      const newW = Math.min(700, Math.max(240, startW + startX - ev.clientX));
      setSidebarWidth(newW);
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [sidebarWidth, setSidebarWidth]);

  // ── Render ──
  return (
    <div className="flex flex-col w-full h-full overflow-hidden">
      {/* Toast */}
      {toast && (
        <div className="absolute top-16 right-6 z-50 px-4 py-2 rounded-xl bg-white dark:bg-zinc-900 text-zinc-900 dark:text-white text-sm shadow-xl border border-zinc-300 dark:border-white/10 animate-in fade-in slide-in-from-top-2">
          {toast}
        </div>
      )}

      <div className="flex-1 flex overflow-hidden">
        {/* Left — Source Audio + Track Picker */}
        <div className="flex flex-col gap-4 p-4 overflow-y-auto border-r border-zinc-200 dark:border-white/5 flex-shrink-0" style={{ width: 300 }}>
          {/* Source Audio Upload */}
          <div>
            <div className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2">
              {t('stemBuilder.sourceAudio')}
            </div>
            {sourceAudioUrl ? (
              <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-amber-500/5 border border-amber-500/15">
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-zinc-300 truncate">{sourceFileName}</div>
                  <div className="text-[10px] text-zinc-600">{t('stemBuilder.readyForLayering')}</div>
                </div>
                <button
                  type="button"
                  onClick={handleClearSource}
                  className="flex-shrink-0 w-6 h-6 rounded-full hover:bg-white/10 flex items-center justify-center transition-colors"
                  title={t('stemBuilder.clearSource')}
                >
                  <X size={12} className="text-zinc-500" />
                </button>
              </div>
            ) : (
              <label className="flex flex-col items-center gap-2 px-4 py-6 rounded-xl border-2 border-dashed border-white/[0.08] hover:border-amber-500/30 cursor-pointer transition-colors group">
                <input
                  type="file"
                  accept=".wav,.mp3,.flac,.ogg"
                  onChange={handleFileSelected}
                  className="hidden"
                  disabled={isUploading}
                />
                {isUploading ? (
                  <Loader2 size={24} className="text-amber-400 animate-spin" />
                ) : (
                  <Upload size={24} className="text-zinc-600 group-hover:text-amber-400 transition-colors" />
                )}
                <span className="text-xs text-zinc-500 group-hover:text-zinc-300 transition-colors">
                  {isUploading ? 'Uploading...' : 'Drop or click to upload backing track'}
                </span>
              </label>
            )}
          </div>

          {/* Model selector */}
          {!modelsLoading && baseModels.length === 0 && (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs">
              <AlertTriangle size={14} />
              <span>{t('stemBuilder.noBaseModels')}</span>
            </div>
          )}
          {baseModels.length > 0 && (
            <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-amber-500/[0.04] border border-amber-500/10">
              <Info size={13} className="text-amber-400 flex-shrink-0" />
              <span className="text-[11px] font-semibold text-amber-400 uppercase tracking-wider whitespace-nowrap">{t('stemBuilder.model')}</span>
              <select
                value={buildModel}
                onChange={e => setBuildModel(e.target.value)}
                disabled={isGenerating}
                className="flex-1 px-2 py-1 rounded-md border border-white/[0.08] bg-white/[0.04] text-zinc-300 text-xs outline-none cursor-pointer focus:border-amber-500/40 transition-colors"
              >
                {baseModels.map(m => (
                  <option key={m} value={m}>{m.replace(/\.gguf$/i, '')}</option>
                ))}
              </select>
            </div>
          )}

          {/* Track Picker */}
          <TrackPicker
            selectedTrack={selectedTrack}
            onTrackChange={setSelectedTrack}
            disabled={isGenerating}
          />

          {/* Collapsible Style Hint */}
          <details>
            <summary className="text-xs text-zinc-500 cursor-pointer font-medium select-none">
              Style Hint (optional)
            </summary>
            <div className="mt-2">
              <input
                type="text"
                value={caption}
                onChange={e => setCaption(e.target.value)}
                placeholder="e.g. tight house drums, warm vintage tone"
                disabled={isGenerating}
                className="w-full px-2.5 py-2 rounded-md border border-white/[0.08] bg-white/[0.04] text-zinc-300 text-xs outline-none placeholder-zinc-600 focus:border-amber-500/40 transition-colors"
              />
              <div className="text-[10px] text-zinc-600 mt-1">
                Describes the style of the generated stem
              </div>
            </div>
          </details>
        </div>

        {/* Center — Generate Button + Progress + Layer Stack */}
        <div className="flex-1 flex flex-col gap-4 p-4 overflow-y-auto border-r border-zinc-200 dark:border-white/5">
          {/* Generate / Cancel */}
          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleGenerate}
              disabled={!canGenerate}
              className={`
                flex-1 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200
                ${canGenerate
                  ? 'bg-gradient-to-r from-amber-500 to-orange-500 text-white hover:from-amber-400 hover:to-orange-400 shadow-lg shadow-amber-500/20'
                  : 'bg-zinc-800 text-zinc-600 cursor-not-allowed'
                }
              `}
            >
              {isGenerating ? (
                <span className="flex items-center justify-center gap-2">
                  <Loader2 size={16} className="animate-spin" />
                  Building...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  <Layers size={16} />
                  Build {selectedTrack ? selectedTrack.replace('_', ' ') : 'Layer'}
                </span>
              )}
            </button>
            {isGenerating && (
              <button
                type="button"
                onClick={handleCancel}
                className="px-4 py-3 rounded-xl text-sm font-medium bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors"
              >
                Cancel
              </button>
            )}
          </div>

          {/* Progress */}
          {isGenerating && (
            <div className="flex items-center gap-3">
              <div className="text-xs text-amber-400 font-medium whitespace-nowrap">
                {genStage || 'Generating...'}
              </div>
              <div className="flex-1 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-amber-500 to-orange-500 transition-all duration-300"
                  style={{ width: `${genProgress}%` }}
                />
              </div>
              <span className="text-[11px] text-zinc-500 font-mono w-8 text-right">{genProgress}%</span>
            </div>
          )}

          {/* Info banner */}
          {!isGenerating && !sourceAudioUrl && (
            <div className="flex items-start gap-3 px-4 py-3 rounded-xl bg-amber-500/[0.04] border border-amber-500/10">
              <Layers size={18} className="text-amber-400 flex-shrink-0 mt-0.5" />
              <div>
                <div className="text-sm font-medium text-zinc-300 mb-1">{t('stemBuilder.buildLayerByLayer')}</div>
                <div className="text-xs text-zinc-500 leading-relaxed">
                  Upload a backing track, select an instrument, and generate a new stem that harmonises
                  with the existing audio. Use the output as a new source to build up a full arrangement
                  one layer at a time.
                </div>
              </div>
            </div>
          )}

          {/* Readiness checklist */}
          {sourceAudioUrl && !isGenerating && (
            <div className="flex flex-wrap gap-2 text-xs">
              <span className={`px-2 py-0.5 rounded-full ${sourceAudioUrl ? 'bg-emerald-500/10 text-emerald-400' : 'bg-zinc-800 text-zinc-600'}`}>
                ✓ Source audio
              </span>
              <span className={`px-2 py-0.5 rounded-full ${selectedTrack ? 'bg-emerald-500/10 text-emerald-400' : 'bg-zinc-800 text-zinc-600'}`}>
                {selectedTrack ? `✓ ${selectedTrack}` : '○ Select track'}
              </span>
              <span className={`px-2 py-0.5 rounded-full ${buildModel ? 'bg-emerald-500/10 text-emerald-400' : 'bg-zinc-800 text-zinc-600'}`}>
                {buildModel ? '✓ Base model' : '○ Need base model'}
              </span>
            </div>
          )}

          {/* Preview Player — dual-track (source + stem) */}
          {previewStemUrl && sourceAudioUrl && (
            <PreviewPlayer
              sourceUrl={sourceAudioUrl}
              stemUrl={previewStemUrl}
              stemLabel={previewLabel}
              onClose={() => setPreviewStemUrl('')}
            />
          )}

          {/* Layer Stack */}
          <LayerStack
            layers={layers}
            sourceFileName={sourceFileName}
            onPlayLayer={handlePlayLayer}
            onUseAsSource={handleUseAsSource}
          />
        </div>

        {/* Resize handle */}
        <div
          className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-amber-500/20 active:bg-amber-500/30 transition-colors"
          onMouseDown={handleSidebarResize}
        >
          <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-amber-400 transition-colors" />
        </div>

        {/* Right — Recent Builds + Queue */}
        <div className="h-full flex-shrink-0 border-l border-zinc-200 dark:border-white/5 overflow-hidden flex flex-col" style={{ width: sidebarWidth }}>
          <Section
            title={t('stemBuilder.recentBuilds')}
            icon={<Clock className="w-3 h-3" />}
            defaultOpen={true}
          >
            <RecentBuilds
              refreshTrigger={refreshTrigger + completionCounter}
              onPlay={handlePlayRecent}
              onUseAsSource={handleUseRecentAsSource}
            />
          </Section>

          <Section
            title={t('stemBuilder.queue')}
            icon={<ListOrdered className="w-3 h-3" />}
            count={queueCount}
            countColor="bg-amber-500/20 text-amber-300"
            defaultOpen={true}
          >
            <InlineAudioQueue />
          </Section>
        </div>
      </div>
    </div>
  );
};

export default StemBuilder;
