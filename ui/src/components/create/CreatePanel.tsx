// CreatePanel.tsx — The composition panel (Content + Metadata only)
//
// Global engine parameters (Models, Adapters, Generation Settings, LM, Mastering)
// have been moved to the GlobalParamBar. This panel now only handles
// per-song content and metadata.

import React, { useEffect, useRef } from 'react';
import { Zap, Loader2, Download, Upload } from 'lucide-react';
import { usePersistedState } from '../../hooks/usePersistedState';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { ContentSection } from './ContentSection';
import { MetadataSection } from './MetadataSection';
import type { GenerationParams, Song } from '../../types';

interface CreatePanelProps {
  onGenerate: (params: Partial<GenerationParams>) => void;
  isGenerating: boolean;
  reuseData?: { song: Song; timestamp: number } | null;
}

export const CreatePanel: React.FC<CreatePanelProps> = ({ onGenerate, isGenerating, reuseData }) => {
  // ── Content (per-song) ──
  const [caption, setCaption] = usePersistedState('hs-caption', '');
  const [lyrics, setLyrics] = usePersistedState('hs-lyrics', '');
  const [instrumental, setInstrumental] = usePersistedState('hs-instrumental', false);

  // ── Metadata (per-song) ──
  const [bpm, setBpm] = usePersistedState('hs-bpm', 0);
  const [keyScale, setKeyScale] = usePersistedState('hs-keyScale', '');
  const [timeSignature, setTimeSignature] = usePersistedState('hs-timeSignature', '');
  const [duration, setDuration] = usePersistedState('hs-duration', -1);
  const [vocalLanguage, setVocalLanguage] = usePersistedState('hs-vocalLanguage', 'en');

  // Global params context — for preset import/export
  const gp = useGlobalParams();

  // ── Reuse data ──
  useEffect(() => {
    if (!reuseData) return;
    const gpData = reuseData.song.generationParams;
    if (!gpData) return;

    if (reuseData.song.caption || gpData.caption) setCaption(reuseData.song.caption || gpData.caption || '');
    if (reuseData.song.lyrics || gpData.lyrics) setLyrics(reuseData.song.lyrics || gpData.lyrics || '');
    if (reuseData.song.style || gpData.style) setCaption(reuseData.song.style || gpData.style || '');
    if (gpData.bpm) setBpm(gpData.bpm);
    if (gpData.keyScale) setKeyScale(gpData.keyScale);
    if (gpData.timeSignature) setTimeSignature(gpData.timeSignature);
    if (gpData.duration) setDuration(typeof gpData.duration === 'string' ? parseFloat(gpData.duration) : gpData.duration);
    if (gpData.inferenceSteps) gp.setInferenceSteps(gpData.inferenceSteps);
    if (gpData.guidanceScale !== undefined) gp.setGuidanceScale(gpData.guidanceScale);
    if (gpData.seed !== undefined) gp.setSeed(gpData.seed);
  }, [reuseData?.timestamp]);

  // ── JSON Import / Export ────────────────────────────────────────────────
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleExport = () => {
    const preset: Record<string, unknown> = {
      _format: 'hot-step-preset',
      _version: 1,
      // Content
      caption, lyrics, instrumental,
      // Metadata
      bpm, duration, keyScale, timeSignature, vocalLanguage,
      // Global engine params
      inferenceSteps: gp.inferenceSteps, guidanceScale: gp.guidanceScale, shift: gp.shift,
      inferMethod: gp.inferMethod, scheduler: gp.scheduler, guidanceMode: gp.guidanceMode,
      seed: gp.seed, randomSeed: gp.randomSeed, batchSize: gp.batchSize,
      useCotCaption: gp.useCotCaption, skipLm: gp.skipLm,
      lmTemperature: gp.lmTemperature, lmCfgScale: gp.lmCfgScale,
      lmTopK: gp.lmTopK, lmTopP: gp.lmTopP, lmNegativePrompt: gp.lmNegativePrompt,
      ditModel: gp.ditModel, lmModel: gp.lmModel, vaeModel: gp.vaeModel,
      adapter: gp.adapter, adapterScale: gp.adapterScale,
      adapterGroupScales: gp.adapterGroupScales, adapterMode: gp.adapterMode,
      masteringEnabled: gp.masteringEnabled, masteringReference: gp.masteringReference,
      timbreReference: gp.timbreReference,
      // Solver sub-params
      storkSubsteps: gp.storkSubsteps, beatStability: gp.beatStability,
      frequencyDamping: gp.frequencyDamping, temporalSmoothing: gp.temporalSmoothing,
      // Guidance sub-params
      apgMomentum: gp.apgMomentum, apgNormThreshold: gp.apgNormThreshold,
      // DCW
      dcwEnabled: gp.dcwEnabled, dcwMode: gp.dcwMode,
      dcwScaler: gp.dcwScaler, dcwHighScaler: gp.dcwHighScaler,
    };
    const blob = new Blob([JSON.stringify(preset, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const slug = (caption || 'preset').slice(0, 40).replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();
    a.download = `${slug}_params.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const p = JSON.parse(reader.result as string);
        // Content
        if (p.caption !== undefined) setCaption(p.caption);
        if (p.lyrics !== undefined) setLyrics(p.lyrics);
        if (p.instrumental !== undefined) setInstrumental(p.instrumental);
        // Metadata
        if (p.bpm !== undefined) setBpm(p.bpm);
        if (p.duration !== undefined) setDuration(p.duration);
        if (p.keyScale !== undefined) setKeyScale(p.keyScale);
        if (p.timeSignature !== undefined) setTimeSignature(p.timeSignature);
        if (p.vocalLanguage !== undefined) setVocalLanguage(p.vocalLanguage);
        // Global engine params → write to context
        if (p.inferenceSteps !== undefined) gp.setInferenceSteps(p.inferenceSteps);
        if (p.guidanceScale !== undefined) gp.setGuidanceScale(p.guidanceScale);
        if (p.shift !== undefined) gp.setShift(p.shift);
        if (p.inferMethod !== undefined) gp.setInferMethod(p.inferMethod);
        if (p.scheduler !== undefined) gp.setScheduler(p.scheduler);
        if (p.guidanceMode !== undefined) gp.setGuidanceMode(p.guidanceMode);
        if (p.seed !== undefined) gp.setSeed(p.seed);
        if (p.randomSeed !== undefined) gp.setRandomSeed(p.randomSeed);
        if (p.batchSize !== undefined) gp.setBatchSize(p.batchSize);
        if (p.useCotCaption !== undefined) gp.setUseCotCaption(p.useCotCaption);
        if (p.skipLm !== undefined) gp.setSkipLm(p.skipLm);
        // LM
        if (p.lmTemperature !== undefined) gp.setLmTemperature(p.lmTemperature);
        if (p.lmCfgScale !== undefined) gp.setLmCfgScale(p.lmCfgScale);
        if (p.lmTopK !== undefined) gp.setLmTopK(p.lmTopK);
        if (p.lmTopP !== undefined) gp.setLmTopP(p.lmTopP);
        if (p.lmNegativePrompt !== undefined) gp.setLmNegativePrompt(p.lmNegativePrompt);
        // Models
        if (p.ditModel !== undefined) gp.setDitModel(p.ditModel);
        if (p.lmModel !== undefined) gp.setLmModel(p.lmModel);
        if (p.vaeModel !== undefined) gp.setVaeModel(p.vaeModel);
        if (p.adapter !== undefined) gp.setAdapter(p.adapter);
        if (p.adapterScale !== undefined) gp.setAdapterScale(p.adapterScale);
        if (p.adapterGroupScales !== undefined) gp.setAdapterGroupScales(p.adapterGroupScales);
        if (p.adapterMode !== undefined) gp.setAdapterMode(p.adapterMode);
        // Mastering
        if (p.masteringEnabled !== undefined) gp.setMasteringEnabled(p.masteringEnabled);
        if (p.masteringReference !== undefined) gp.setMasteringReference(p.masteringReference);
        if (p.timbreReference !== undefined) gp.setTimbreReference(p.timbreReference);
        // Solver sub-params
        if (p.storkSubsteps !== undefined) gp.setStorkSubsteps(p.storkSubsteps);
        if (p.beatStability !== undefined) gp.setBeatStability(p.beatStability);
        if (p.frequencyDamping !== undefined) gp.setFrequencyDamping(p.frequencyDamping);
        if (p.temporalSmoothing !== undefined) gp.setTemporalSmoothing(p.temporalSmoothing);
        // Guidance sub-params
        if (p.apgMomentum !== undefined) gp.setApgMomentum(p.apgMomentum);
        if (p.apgNormThreshold !== undefined) gp.setApgNormThreshold(p.apgNormThreshold);
        // DCW
        if (p.dcwEnabled !== undefined) gp.setDcwEnabled(p.dcwEnabled);
        if (p.dcwMode !== undefined) gp.setDcwMode(p.dcwMode);
        if (p.dcwScaler !== undefined) gp.setDcwScaler(p.dcwScaler / 0.02);
        if (p.dcwHighScaler !== undefined) gp.setDcwHighScaler(p.dcwHighScaler / 0.02);
      } catch (err) {
        console.error('[Preset Import] Invalid JSON:', err);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const handleGenerate = () => {
    // Send only content + metadata params — global engine params are merged in App.tsx
    const params: Partial<GenerationParams> = {
      caption,
      lyrics: instrumental ? '[Instrumental]' : lyrics,
      instrumental,
      bpm,
      duration,
      keyScale,
      timeSignature,
      vocalLanguage,
      taskType: 'text2music',
    };
    onGenerate(params);
  };

  return (
    <div className="h-full flex flex-col bg-zinc-50 dark:bg-suno-panel">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-white/5">
        <h2 className="text-lg font-bold text-zinc-900 dark:text-white">Create</h2>
        <div className="flex items-center gap-1.5">
          <button onClick={handleExport} title="Export preset"
            className="p-1.5 rounded-lg hover:bg-white/10 text-zinc-500 hover:text-emerald-400 transition-colors">
            <Upload size={14} />
          </button>
          <button onClick={() => fileInputRef.current?.click()} title="Import preset"
            className="p-1.5 rounded-lg hover:bg-white/10 text-zinc-500 hover:text-sky-400 transition-colors">
            <Download size={14} />
          </button>
          <input ref={fileInputRef} type="file" accept=".json" className="hidden"
            onChange={handleImport} />
          <span className="text-xs text-zinc-500 font-medium ml-1">text2music</span>
        </div>
      </div>

      {/* Scrollable body — now much slimmer */}
      <div className="flex-1 overflow-y-auto hide-scrollbar px-4 py-3 space-y-1">
        <ContentSection
          caption={caption} onCaptionChange={setCaption}
          lyrics={lyrics} onLyricsChange={setLyrics}
          instrumental={instrumental} onInstrumentalChange={setInstrumental}
        />

        <MetadataSection
          bpm={bpm} onBpmChange={setBpm}
          keyScale={keyScale} onKeyScaleChange={setKeyScale}
          timeSignature={timeSignature} onTimeSignatureChange={setTimeSignature}
          duration={duration} onDurationChange={setDuration}
          vocalLanguage={vocalLanguage} onVocalLanguageChange={setVocalLanguage}
        />
      </div>

      {/* Generate button */}
      <div className="px-4 py-3 border-t border-zinc-200 dark:border-white/5">
        <button
          className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 text-white font-semibold text-sm transition-all duration-200 hover:shadow-lg hover:shadow-pink-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
          onClick={handleGenerate}
          disabled={isGenerating || (!caption.trim() && !lyrics.trim() && !instrumental)}
        >
          {isGenerating ? (
            <>
              <Loader2 size={18} className="spinner" />
              Generating...
            </>
          ) : (
            <>
              <Zap size={18} />
              Generate
            </>
          )}
        </button>
      </div>
    </div>
  );
};
