// GlobalParamBar.tsx — Horizontal top bar with hover-to-expand engine config sections
//
// Renders 5 sections: Models, Adapters, Generation, LM/Thinking, Post-Processing.
// Each section shows a summary badge and expands on hover to reveal controls.
// Sits full-width at the top of the entire window (above sidebar).

import React, { useState, useCallback, useRef } from 'react';
import { Cpu, Plug, Sliders, Brain, AudioWaveform, Upload, Download } from 'lucide-react';
import { BarSection, ToggleSwitch } from './BarSection';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { ModelsDropdown, ModelsBadge } from './ModelsDropdown';
import { AdaptersDropdown, AdaptersBadge } from './AdaptersDropdown';
import { GenerationDropdown, GenerationBadge } from './GenerationDropdown';
import { LmThinkingDropdown, LmThinkingBadge } from './LmThinkingDropdown';
import { PostProcessingDropdown, PostProcessingBadge } from './PostProcessingDropdown';
import { VramIndicator } from '../shared/VramIndicator';

type SectionId = 'models' | 'adapters' | 'generation' | 'lm' | 'postprocessing' | null;

export const GlobalParamBar: React.FC = () => {
  const [openSection, setOpenSection] = useState<SectionId>(null);
  const gp = useGlobalParams();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Preset Export ────────────────────────────────────────────────
  const handleExport = useCallback(() => {
    // Read content params from localStorage (same keys CreatePanel uses)
    const readLS = (key: string, fallback: string) => {
      try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; }
    };
    const preset: Record<string, unknown> = {
      _format: 'hot-step-preset', _version: 1,
      // Content (from localStorage)
      caption: readLS('hs-caption', ''), lyrics: readLS('hs-lyrics', ''), instrumental: readLS('hs-instrumental', false),
      bpm: readLS('hs-bpm', 0), duration: readLS('hs-duration', -1),
      keyScale: readLS('hs-keyScale', ''), timeSignature: readLS('hs-timeSignature', ''),
      vocalLanguage: readLS('hs-vocalLanguage', 'en'),
      // Global engine params
      ...gp.getGlobalParams(),
    };
    const blob = new Blob([JSON.stringify(preset, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const slug = (String(preset.caption || 'preset')).slice(0, 40).replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();
    a.download = `${slug}_params.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [gp]);

  // ── Preset Import ────────────────────────────────────────────────
  const handleImport = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const p = JSON.parse(reader.result as string);
        const writeLS = (key: string, val: unknown) => localStorage.setItem(key, JSON.stringify(val));
        // Content → localStorage (CreatePanel reads from there)
        if (p.caption !== undefined) writeLS('hs-caption', p.caption);
        if (p.lyrics !== undefined) writeLS('hs-lyrics', p.lyrics);
        if (p.instrumental !== undefined) writeLS('hs-instrumental', p.instrumental);
        if (p.bpm !== undefined) writeLS('hs-bpm', p.bpm);
        if (p.duration !== undefined) writeLS('hs-duration', p.duration);
        if (p.keyScale !== undefined) writeLS('hs-keyScale', p.keyScale);
        if (p.timeSignature !== undefined) writeLS('hs-timeSignature', p.timeSignature);
        if (p.vocalLanguage !== undefined) writeLS('hs-vocalLanguage', p.vocalLanguage);
        // Global engine params → context
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
        if (p.lmTemperature !== undefined) gp.setLmTemperature(p.lmTemperature);
        if (p.lmCfgScale !== undefined) gp.setLmCfgScale(p.lmCfgScale);
        if (p.lmTopK !== undefined) gp.setLmTopK(p.lmTopK);
        if (p.lmTopP !== undefined) gp.setLmTopP(p.lmTopP);
        if (p.lmNegativePrompt !== undefined) gp.setLmNegativePrompt(p.lmNegativePrompt);
        if (p.ditModel !== undefined) gp.setDitModel(p.ditModel);
        if (p.lmModel !== undefined) gp.setLmModel(p.lmModel);
        if (p.vaeModel !== undefined) gp.setVaeModel(p.vaeModel);
        if (p.adapter !== undefined) gp.setAdapter(p.adapter);
        if (p.adapterScale !== undefined) gp.setAdapterScale(p.adapterScale);
        if (p.adapterGroupScales !== undefined) gp.setAdapterGroupScales(p.adapterGroupScales);
        if (p.adapterMode !== undefined) gp.setAdapterMode(p.adapterMode);
        if (p.spectralLifterEnabled !== undefined) gp.setSpectralLifterEnabled(p.spectralLifterEnabled);
        if (p.masteringEnabled !== undefined) gp.setMasteringEnabled(p.masteringEnabled);
        if (p.masteringReference !== undefined) gp.setMasteringReference(p.masteringReference);
        if (p.timbreReference !== undefined) gp.setTimbreReference(p.timbreReference);
        if (p.storkSubsteps !== undefined) gp.setStorkSubsteps(p.storkSubsteps);
        if (p.beatStability !== undefined) gp.setBeatStability(p.beatStability);
        if (p.frequencyDamping !== undefined) gp.setFrequencyDamping(p.frequencyDamping);
        if (p.temporalSmoothing !== undefined) gp.setTemporalSmoothing(p.temporalSmoothing);
        if (p.apgMomentum !== undefined) gp.setApgMomentum(p.apgMomentum);
        if (p.apgNormThreshold !== undefined) gp.setApgNormThreshold(p.apgNormThreshold);
        if (p.dcwEnabled !== undefined) gp.setDcwEnabled(p.dcwEnabled);
        if (p.dcwMode !== undefined) gp.setDcwMode(p.dcwMode);
        if (p.dcwScaler !== undefined) gp.setDcwScaler(p.dcwScaler);
        if (p.dcwHighScaler !== undefined) gp.setDcwHighScaler(p.dcwHighScaler);
        if (p.latentShift !== undefined) gp.setLatentShift(p.latentShift);
        if (p.latentRescale !== undefined) gp.setLatentRescale(p.latentRescale);
        if (p.customTimesteps !== undefined) gp.setCustomTimesteps(p.customTimesteps);
        // Force a page reload to pick up localStorage content changes
        window.location.reload();
      } catch (err) {
        console.error('[Preset Import] Invalid JSON:', err);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  }, [gp]);

  const handleOpen = useCallback((id: SectionId) => {
    setOpenSection(id);
  }, []);

  // Only close if the requesting section is still the one that's open.
  // Prevents the leaving section's delayed close from killing a newly-opened neighbour.
  const handleClose = useCallback((id: SectionId) => {
    setOpenSection(prev => prev === id ? null : prev);
  }, []);

  return (
    <div className="flex-shrink-0 relative z-40 bg-zinc-900/95 border-b border-white/5"
         style={{ backdropFilter: 'blur(20px)' }}>
      <div className="flex items-stretch">
        {/* Logo */}
        <div className="flex items-center gap-2.5 px-5 flex-shrink-0 border-r border-white/5" style={{ width: '210px' }}>
          <div className="w-7 h-7 rounded-full bg-gradient-to-br from-pink-500 to-purple-600 flex items-center justify-center shadow-lg flex-shrink-0">
            <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <span className="text-base font-bold text-white whitespace-nowrap">HOT-Step CPP</span>
        </div>

        {/* Sections — separated by dividers */}
        <div className="flex-1 flex items-stretch divide-x divide-white/5">
          <BarSection
            id="models"
            label="Models"
            icon={<Cpu size={14} />}
            badge={<ModelsBadge />}
            accentColor="pink"
            isOpen={openSection === 'models'}
            onOpen={() => handleOpen('models')}
            onClose={() => handleClose('models')}
          >
            <ModelsDropdown />
          </BarSection>

          <BarSection
            id="adapters"
            label="Adapters"
            icon={<Plug size={14} />}
            badge={<AdaptersBadge />}
            accentColor="emerald"
            isOpen={openSection === 'adapters'}
            onOpen={() => handleOpen('adapters')}
            onClose={() => handleClose('adapters')}
          >
            <AdaptersDropdown />
          </BarSection>

          <BarSection
            id="generation"
            label="Generation"
            icon={<Sliders size={14} />}
            badge={<GenerationBadge />}
            accentColor="sky"
            isOpen={openSection === 'generation'}
            onOpen={() => handleOpen('generation')}
            onClose={() => handleClose('generation')}
          >
            <GenerationDropdown />
          </BarSection>

          <BarSection
            id="lm"
            label="LM / Thinking"
            icon={<Brain size={14} />}
            badge={<LmThinkingBadge />}
            accentColor="purple"
            isOpen={openSection === 'lm'}
            onOpen={() => handleOpen('lm')}
            onClose={() => handleClose('lm')}
            headerToggle={
              <ToggleSwitch
                checked={!gp.skipLm}
                onChange={(on) => gp.setSkipLm(!on)}
                accentColor="purple"
              />
            }
          >
            <LmThinkingDropdown />
          </BarSection>

          <BarSection
            id="postprocessing"
            label="Post-Processing"
            icon={<AudioWaveform size={14} />}
            badge={<PostProcessingBadge />}
            accentColor="amber"
            isOpen={openSection === 'postprocessing'}
            onOpen={() => handleOpen('postprocessing')}
            onClose={() => handleClose('postprocessing')}
          >
            <PostProcessingDropdown />
          </BarSection>
        </div>

        {/* Right — Export/Import + VRAM */}
        <div className="flex items-center gap-2 flex-shrink-0 px-3 border-l border-white/5" style={{ width: '210px' }}>
          <button onClick={handleExport} title="Export preset"
            className="p-1.5 rounded-lg hover:bg-white/10 text-zinc-500 hover:text-emerald-400 transition-colors">
            <Upload size={13} />
          </button>
          <button onClick={() => fileInputRef.current?.click()} title="Import preset"
            className="p-1.5 rounded-lg hover:bg-white/10 text-zinc-500 hover:text-sky-400 transition-colors">
            <Download size={13} />
          </button>
          <input ref={fileInputRef} type="file" accept=".json" className="hidden" onChange={handleImport} />
          <div className="w-px h-4 bg-white/5" />
          <VramIndicator compact />
        </div>
      </div>
    </div>
  );
};
