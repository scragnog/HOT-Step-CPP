import { useQuery, useQueryClient } from '@tanstack/react-query';
import { modelsOptions, songsOptions } from '../../queries/studios/shared';
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

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Upload, X, Loader2, Info, AlertTriangle, Layers, Clock, ListOrdered } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../../context/AuthContext';
import { useGlobalParamsStore } from '../../context/GlobalParamsContext';
import { usePersistedState } from '../../hooks/usePersistedState';
import { submitStudioRender, followStudioRender, type SourceRef } from '../../services/repaintLayerWorkflowApi';
import { useUpload } from '../../queries/writes';
import { studioWorkflowApi as workflowApi } from '../../queries/studios/workflows';
import { useBackendStore } from '../../stores/backendStore';
import { TrackPicker, type TrackName } from './TrackPicker';
import { LayerStack, type LayerInfo } from './LayerStack';
import { RecentBuilds } from './RecentBuilds';
import { PreviewPlayer } from './PreviewPlayer';
import { Section } from '../shared/ActivitySidebar';
import { StyledSelect } from '../shared/StyledSelect';
import { ParamLabel } from '../shared/ParamLabel';
import { BackendCapabilityGate } from '../shared/BackendCapabilityGate';
import { InlineAudioQueue } from '../lyric-studio/InlineAudioQueue';
import { useAudioGenQueueSelector } from '../../stores/audioGenQueueStore';
import { queryManager } from '../../queries/invalidation';

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
  const [sourceRef, setSourceRef] = useState<SourceRef | null>(() => {
    try { return JSON.parse(localStorage.getItem('hs-sb-sourceRef') || 'null'); } catch { return null; }
  });
  const sourceUrlRef = useRef(sourceAudioUrl);
  sourceUrlRef.current = sourceAudioUrl;
  const [sourceFileName, setSourceFileName] = useState(() => localStorage.getItem('hs-sb-sourceFile') || '');
  const { mutate: uploadSource, isPending: isUploading } = useUpload<{ audio_url: string; asset_id?: string }>();

  // ── Track selection ──
  const [selectedTrack, setSelectedTrack] = useState<TrackName | null>(null);

  // ── Style hint (collapsible) ──
  const [caption, setCaption] = useState('');

  // ── Model selection ──
  const modelsQuery = useQuery(modelsOptions());
  const baseModels = getBaseModels(modelsQuery.data?.models.dit || []);
  const [buildModel, setBuildModel] = useState<string>('');
  const modelsLoading = modelsQuery.isPending;

  // ── Generation state ──
  const [isGenerating, setIsGenerating] = useState(false);
  const [genStage, setGenStage] = useState('');
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const submissionKeyRef = useRef<string | null>(null);

  // ── Iterative composition ──
  const [layers, setLayers] = useState<LayerInfo[]>([]);

  // ── Preview player ──
  const [previewStemUrl, setPreviewStemUrl] = useState('');
  const [previewLabel, setPreviewLabel] = useState('');

  // ── Narrow-screen pane switcher ──
  const [narrowPane, setNarrowPane] = useState<'source' | 'build' | 'recent'>('build');

  // Portrait-screen (>= 1024px, orientation portrait) — behaves like narrow
  const [isPortrait, setIsPortrait] = useState(() =>
    window.matchMedia('(orientation: portrait) and (min-width: 1024px)').matches
  );
  useEffect(() => {
    const mq = window.matchMedia('(orientation: portrait) and (min-width: 1024px)');
    const handler = (e: MediaQueryListEvent) => setIsPortrait(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  // ── Sidebar ──
  const [sidebarWidth, setSidebarWidth] = usePersistedState('hs-activitySidebarWidth', 320);
  const queueCount = useAudioGenQueueSelector(s =>
    s.items.filter(i => i.status === 'pending' || i.status === 'loading-adapter' || i.status === 'generating').length
  );
  const completionCounter = useAudioGenQueueSelector(s => s.completionCounter);
  const client = useQueryClient();
  // Queue completions are written by the shell, not by this page; refresh the
  // shared song feed when one lands so a finished layer appears in Recent.
  useEffect(() => {
    if (completionCounter) void queryManager.invalidate(client, songsOptions().queryKey);
  }, [client, completionCounter]);

  // ── Toast ──
  const [toast, setToast] = useState('');
  const showToast = (msg: string) => { setToast(msg); setTimeout(() => setToast(''), 4000); };

  // ── Persist source ──
  useEffect(() => { localStorage.setItem('hs-sb-sourceUrl', sourceAudioUrl); }, [sourceAudioUrl]);
  useEffect(() => { localStorage.setItem('hs-sb-sourceRef', JSON.stringify(sourceRef)); }, [sourceRef]);
  useEffect(() => { localStorage.setItem('hs-sb-sourceFile', sourceFileName); }, [sourceFileName]);

  useEffect(() => {
    const id = localStorage.getItem('hs-sb-workflowJob');
    if (!id || !token) return;
    let mounted = true;
    const observation = new AbortController();
    setIsGenerating(true);
    setActiveJobId(id);
    void followStudioRender(id, stage => { if (mounted) setGenStage(stage); }, { signal: observation.signal })
      .then(async result => {
        if (!mounted) return;
        const { job } = await workflowApi.get(id);
        if (!mounted) return;
        const captured = job.input as { source?: SourceRef; trackName?: string; caption?: string };
        const audioUrl = result.audio?.audioUrls?.[0];
        void queryManager.invalidate(client, songsOptions().queryKey);
        if (audioUrl && captured.source?.expectedUrl === sourceUrlRef.current) {
          setLayers(prev => [...prev, { trackName: captured.trackName || 'stem', caption: captured.caption || '',
            audioUrl, songId: result.audio?.songIds?.[0], timestamp: Date.now() }]);
          setPreviewStemUrl(audioUrl);
          setPreviewLabel((captured.trackName || 'stem').replace('_', ' '));
        }
        showToast('Layer complete!');
      }).catch(err => { if (mounted) showToast(`Layer ${String((err as Error).message)}`); })
      .finally(() => {
        if (mounted && localStorage.getItem('hs-sb-workflowJob') === id) localStorage.removeItem('hs-sb-workflowJob');
        if (mounted) { setIsGenerating(false); setActiveJobId(null); }
      });
    return () => { mounted = false; observation.abort(); };
  }, [token, client]);

  // ── Persist model ──
  useEffect(() => {
    if (buildModel) localStorage.setItem('hs-sb-model', buildModel);
  }, [buildModel]);

  // ── Load base models on mount ──
  useEffect(() => {
    const base = getBaseModels(modelsQuery.data?.models.dit || []);
    if (base.length && !buildModel) {
      const stored = localStorage.getItem('hs-sb-model');
      setBuildModel(stored && base.includes(stored) ? stored : base[0]);
    }
  }, [modelsQuery.data, buildModel]);

  // ── File upload ──
  const handleFileSelected = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = ''; // reset so same file can be re-selected

    setSourceFileName(file.name);
    setSourceRef(null);
    setLayers([]); // reset layer stack on new source

    uploadSource({
      path: '/upload/audio',
      files: [{ field: 'audio', file }],
      onSuccess: data => {
        setSourceAudioUrl(data.audio_url || '');
        setSourceRef(data.asset_id ? { kind: 'asset', id: data.asset_id, expectedUrl: data.audio_url } : null);
        showToast('Source audio uploaded!');
      },
    });
  }, [uploadSource]);

  const handleClearSource = () => {
    setSourceAudioUrl('');
    setSourceRef(null);
    setSourceFileName('');
    setLayers([]);
    setSelectedTrack(null);
    setCaption('');
  };

  // ── Use layer output as new source (iterative composition) ──
  const handleUseAsSource = useCallback((layer: LayerInfo) => {
    setSourceAudioUrl(layer.audioUrl);
    setSourceRef(layer.songId ? { kind: 'song', id: layer.songId, expectedUrl: layer.audioUrl } : null);
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
  const handleUseRecentAsSource = useCallback((build: { id: string; audioUrl: string; title: string }) => {
    setSourceAudioUrl(build.audioUrl);
    setSourceRef({ kind: 'song', id: build.id, expectedUrl: build.audioUrl });
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
    if (submissionKeyRef.current || localStorage.getItem('hs-sb-workflowJob')) return;
    if (!token || !sourceAudioUrl || !sourceRef || sourceRef.expectedUrl !== sourceAudioUrl ||
        !selectedTrack || !buildModel) {
      showToast('Select a source, track, and model');
      return;
    }
    const key = crypto.randomUUID();
    submissionKeyRef.current = key;
    setIsGenerating(true);
    setGenStage('Submitting...');
    let submittedJobId: string | null = null;
    try {
      const { job } = await submitStudioRender('layer-render', {
        source: sourceRef, expectedBackend: useBackendStore.getState().activeBackendId,
        engineParams: gp.getGlobalParams(), trackName: selectedTrack, buildModel, caption,
      }, key);
      if (submissionKeyRef.current !== key) { await workflowApi.cancel(job.id); return; }
      setActiveJobId(job.id);
      submittedJobId = job.id;
      localStorage.setItem('hs-sb-workflowJob', job.id);
      showToast(`Building ${selectedTrack} layer...`);
      const result = await followStudioRender(job.id, stage => setGenStage(stage));
      const audio = result.audio || {};
      setGenStage('Complete!');
      void queryManager.invalidate(client, songsOptions().queryKey);
      const audioUrl = audio.audioUrls?.[0] || '';
      if (audioUrl && sourceUrlRef.current === sourceAudioUrl) {
        setLayers(prev => [...prev, {
          trackName: selectedTrack, caption, audioUrl,
          songId: audio.songIds?.[0], timestamp: Date.now(),
        }]);
        setPreviewStemUrl(audioUrl);
        setPreviewLabel(selectedTrack.replace('_', ' '));
      }
      showToast(`${selectedTrack} layer complete!`);
      setTimeout(() => setGenStage(''), 3000);
    } catch (err: any) {
      showToast(`Generation failed: ${err.message}`);
      setGenStage('');
    } finally {
      if (submissionKeyRef.current === key) {
        submissionKeyRef.current = null;
        if (localStorage.getItem('hs-sb-workflowJob') === submittedJobId) localStorage.removeItem('hs-sb-workflowJob');
        setIsGenerating(false);
        setActiveJobId(null);
      }
    }
  };

  const handleCancel = async () => {
    submissionKeyRef.current = null;
    if (activeJobId && token) {
      try { await workflowApi.cancel(activeJobId); } catch { /* ignore */ }
    }
    localStorage.removeItem('hs-sb-workflowJob');
    setIsGenerating(false);
    setActiveJobId(null);
    setGenStage('');
  };

  const canGenerate = !!sourceAudioUrl && !!sourceRef && !!selectedTrack && !!buildModel && !isGenerating;

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
    <BackendCapabilityGate feature="stems">
    <div className="flex flex-col w-full h-full overflow-hidden">
      {(modelsQuery.error) && <div role="alert" className="text-sm text-red-500">{(modelsQuery.error)?.message}</div>}
      {/* Toast */}
      {toast && (
        <div className="absolute top-16 right-6 z-50 px-4 py-2 rounded-xl bg-white dark:bg-zinc-900 text-zinc-900 dark:text-white text-sm shadow-xl border border-zinc-300 dark:border-white/10 animate-in fade-in slide-in-from-top-2">
          {toast}
        </div>
      )}

      {/* Narrow/portrait pane switcher — rendered below 1024px and on portrait >= 1024px */}
      <div className="max-[1279px]:flex min-[1280px]:hidden portrait-wide:flex flex-shrink-0 border-b border-zinc-200 dark:border-white/5">
        {(['source', 'build', 'recent'] as const).map(pane => (
          <button
            key={pane}
            type="button"
            onClick={() => setNarrowPane(pane)}
            className={`flex-1 py-2 text-xs font-semibold transition-colors ${
              narrowPane === pane
                ? 'text-amber-400 border-b-2 border-amber-400'
                : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {pane === 'source' ? 'Source' : pane === 'build' ? 'Build' : 'Recent'}
          </button>
        ))}
      </div>

      <div className="flex-1 flex overflow-hidden min-h-0">
        {/* Left — Source Audio + Track Picker */}
        <div className={`flex flex-col gap-4 p-4 overflow-y-auto border-r border-zinc-200 dark:border-white/5 flex-shrink-0 ${!isPortrait ? 'min-[1280px]:w-[280px] min-[1280px]:min-w-[200px] min-[1280px]:shrink min-[1280px]:flex' : ''} ${narrowPane === 'source' ? 'flex w-full min-w-0' : 'hidden'}`}>
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
              <label
                // The zone said "Drop or click" but only click was wired, so a
                // dropped file opened as a browser tab (#128).
                onDragOver={e => e.preventDefault()}
                onDrop={e => {
                  e.preventDefault();
                  const files = e.dataTransfer?.files;
                  if (files && files.length > 0 && !isUploading) {
                    void handleFileSelected({ target: { files, value: '' } } as unknown as React.ChangeEvent<HTMLInputElement>);
                  }
                }}
                className="flex flex-col items-center gap-2 px-4 py-6 rounded-xl border-2 border-dashed border-white/[0.08] hover:border-amber-500/30 cursor-pointer transition-colors group">
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
              <ParamLabel
                label={t('stemBuilder.model')}
                info={t('stemBuilder.modelInfo')}
                className="text-[11px] font-semibold text-amber-400 uppercase tracking-wider whitespace-nowrap"
              />
              <StyledSelect
                accent="amber"
                value={buildModel}
                onChange={setBuildModel}
                disabled={isGenerating}
                options={baseModels.map(m => ({ value: m, label: m.replace(/\.gguf$/i, '') }))}
                className="flex-1"
              />
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
              <ParamLabel
                label="Style Hint (optional)"
                info="Free-text description of the stem's character, sent to the engine as the generation caption. Leave it blank to let the engine infer style from the source audio alone; fill it in to steer tone and instrumentation, for example 'tight house drums, warm vintage tone'."
                className="text-xs text-zinc-500 font-medium"
              />
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
            </div>
          </details>
        </div>

        {/* Center — Generate Button + Progress + Layer Stack */}
        <div className={`flex-col gap-4 p-4 overflow-y-auto border-r border-zinc-200 dark:border-white/5 flex-1 min-w-0 ${!isPortrait ? 'min-[1280px]:flex min-[1280px]:min-w-[220px]' : ''} ${narrowPane === 'build' ? 'flex' : 'hidden'}`}>
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

          {/* Render stage */}
          {isGenerating && (
            <div className="flex items-center gap-3">
              <Loader2 size={14} className="animate-spin text-amber-400" />
              <div className="text-xs text-amber-400 font-medium whitespace-nowrap">
                {genStage || 'Generating...'}
              </div>
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

        {/* Resize handle — hidden on narrow screens */}
        <div
          className={`${isPortrait ? 'hidden' : 'max-[1279px]:hidden'} flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-amber-500/20 active:bg-amber-500/30 transition-colors`}
          onMouseDown={handleSidebarResize}
        >
          <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-amber-400 transition-colors" />
        </div>

        {/* Right — Recent Builds + Queue */}
        <div
          className={`h-full flex-shrink-0 border-l border-zinc-200 dark:border-white/5 overflow-hidden flex-col ${!isPortrait ? 'min-[1280px]:flex min-[1280px]:max-w-[220px]' : ''} max-[1279px]:!w-full ${narrowPane === 'recent' ? 'flex' : 'hidden'}`}
          style={{ width: sidebarWidth }}
        >
          <Section
            title={t('stemBuilder.recentBuilds')}
            icon={<Clock className="w-3 h-3" />}
            defaultOpen={true}
          >
            <RecentBuilds
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
    </BackendCapabilityGate>
  );
};

export default StemBuilder;
