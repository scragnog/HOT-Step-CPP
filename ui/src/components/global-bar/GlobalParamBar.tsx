// GlobalParamBar.tsx — Horizontal top bar with hover-to-expand engine config sections
//
// Renders 5 sections: Models, Adapters, Generation, LM/Thinking, Mastering.
// Each section shows a summary badge and expands on hover to reveal controls.
// Sits full-width at the top of the entire window (above sidebar).

import React, { useState, useCallback } from 'react';
import { Cpu, Plug, Sliders, Brain, AudioWaveform } from 'lucide-react';
import { BarSection } from './BarSection';
import { ModelsDropdown, ModelsBadge } from './ModelsDropdown';
import { AdaptersDropdown, AdaptersBadge } from './AdaptersDropdown';
import { GenerationDropdown, GenerationBadge } from './GenerationDropdown';
import { LmThinkingDropdown, LmThinkingBadge } from './LmThinkingDropdown';
import { MasteringDropdown, MasteringBadge } from './MasteringDropdown';

type SectionId = 'models' | 'adapters' | 'generation' | 'lm' | 'mastering' | null;

export const GlobalParamBar: React.FC = () => {
  const [openSection, setOpenSection] = useState<SectionId>(null);

  const handleOpen = useCallback((id: SectionId) => {
    setOpenSection(id);
  }, []);

  const handleClose = useCallback(() => {
    setOpenSection(null);
  }, []);

  return (
    <div className="flex-shrink-0 relative z-40 bg-zinc-900/95 border-b border-white/5"
         style={{ backdropFilter: 'blur(20px)' }}>
      <div className="flex items-stretch divide-x divide-white/5">
        <BarSection
          id="models"
          label="Models"
          icon={<Cpu size={14} />}
          badge={<ModelsBadge />}
          accentColor="pink"
          isOpen={openSection === 'models'}
          onOpen={() => handleOpen('models')}
          onClose={handleClose}
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
          onClose={handleClose}
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
          onClose={handleClose}
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
          onClose={handleClose}
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
          onClose={handleClose}
        >
          <MasteringDropdown />
        </BarSection>
      </div>
    </div>
  );
};
