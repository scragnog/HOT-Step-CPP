// GlobalParamBar.tsx — Horizontal top bar with hover-to-expand engine config sections
//
// Renders 5 sections: Models, Adapters, Generation, LM/Thinking, Mastering.
// Each section shows a summary badge and expands on hover to reveal controls.
// Sits full-width at the top of the entire window (above sidebar).

import React, { useState, useCallback } from 'react';
import { Cpu, Plug, Sliders, Brain, AudioWaveform } from 'lucide-react';
import { BarSection, ToggleSwitch } from './BarSection';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { ModelsDropdown, ModelsBadge } from './ModelsDropdown';
import { AdaptersDropdown, AdaptersBadge } from './AdaptersDropdown';
import { GenerationDropdown, GenerationBadge } from './GenerationDropdown';
import { LmThinkingDropdown, LmThinkingBadge } from './LmThinkingDropdown';
import { MasteringDropdown, MasteringBadge } from './MasteringDropdown';
import { VramIndicator } from '../shared/VramIndicator';

type SectionId = 'models' | 'adapters' | 'generation' | 'lm' | 'mastering' | null;

export const GlobalParamBar: React.FC = () => {
  const [openSection, setOpenSection] = useState<SectionId>(null);
  const gp = useGlobalParams();

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
        <div className="flex items-center gap-2.5 px-4 flex-shrink-0 border-r border-white/5">
          <div className="w-7 h-7 rounded-full bg-gradient-to-br from-pink-500 to-purple-600 flex items-center justify-center shadow-lg flex-shrink-0">
            <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <span className="text-sm font-bold text-white whitespace-nowrap">HOT-Step CPP</span>
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
            id="mastering"
            label="Mastering"
            icon={<AudioWaveform size={14} />}
            badge={<MasteringBadge />}
            accentColor="amber"
            isOpen={openSection === 'mastering'}
            onOpen={() => handleOpen('mastering')}
            onClose={() => handleClose('mastering')}
            headerToggle={
              <ToggleSwitch
                checked={gp.masteringEnabled}
                onChange={gp.setMasteringEnabled}
                accentColor="amber"
              />
            }
          >
            <MasteringDropdown />
          </BarSection>
        </div>

        {/* Right — VRAM indicator (matches logo width for symmetry) */}
        <div className="flex items-center justify-center flex-shrink-0 px-4 border-l border-white/5" style={{ width: '172px' }}>
          <VramIndicator compact />
        </div>
      </div>
    </div>
  );
};
