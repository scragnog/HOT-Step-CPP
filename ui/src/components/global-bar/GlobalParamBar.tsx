// GlobalParamBar.tsx — Horizontal top bar with hover-to-expand engine config sections
//
// Renders 5 sections: Models, Adapters, Generation, LM/Thinking, Post-Processing.
// Each section shows a summary badge and expands on hover to reveal controls.
// Sits full-width at the top of the entire window (above sidebar).

import React, { useState, useCallback, useEffect } from 'react';
import { Cpu, Plug, Sliders, Brain, AudioWaveform, Bookmark } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { BarSection, ToggleSwitch } from './BarSection';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { modelApi } from '../../services/api';
import { ModelManagerModal } from '../model-manager/ModelManagerModal';
import { ModelsDropdown, ModelsBadge } from './ModelsDropdown';
import { AdaptersDropdown, AdaptersBadge } from './AdaptersDropdown';
import { GenerationDropdown, GenerationBadge } from './GenerationDropdown';
import { LmThinkingDropdown, LmThinkingBadge } from './LmThinkingDropdown';
import { PostProcessingDropdown, PostProcessingBadge } from './PostProcessingDropdown';
import { VramIndicator } from '../shared/VramIndicator';
import { DiscoPulseWrapper } from '../shared/DiscoPulseWrapper';
import { MonitorBar } from './MonitorBar';
import { useVstChainStore } from '../../stores/vstChainStore';
import { ProfilesModal } from './ProfilesModal';

type SectionId = 'models' | 'adapters' | 'generation' | 'lm' | 'postprocessing' | null;

export const GlobalParamBar: React.FC = () => {
  const { t } = useTranslation();
  const [openSection, setOpenSection] = useState<SectionId>(null);
  const gp = useGlobalParams();
  const monitoring = useVstChainStore(s => s.monitoring);

  // ── Auto-select models when engine becomes ready ────────────────
  // Polls the engine until it returns a model list, then auto-selects
  // the first available model for any empty slot. Runs independently
  // of the Model Manager modal state.
  const [showModelManager, setShowModelManager] = useState(false);
  const [showProfiles, setShowProfiles] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let retries = 0;
    const MAX_RETRIES = 20; // ~60 seconds of polling

    const tryAutoSelect = () => {
      if (cancelled) return;
      modelApi.list()
        .then((data) => {
          if (cancelled) return;
          const dit = data?.models?.dit || [];
          const lm = data?.models?.lm || [];
          const vae = data?.models?.vae || [];
          const emb = data?.models?.embedding || [];

          // Auto-select first available model for any empty slot
          if (dit.length > 0 && !gp.ditModel) gp.setDitModel(dit[0]);
          if (lm.length > 0 && !gp.lmModel) gp.setLmModel(lm[0]);
          if (vae.length > 0 && !gp.vaeModel) gp.setVaeModel(vae[0]);
          if (emb.length > 0 && !gp.embeddingModel) gp.setEmbeddingModel(emb[0]);

          // If we got models, we're done. If empty, keep polling
          // (user might be downloading via Model Manager right now)
          if (dit.length === 0 && retries < MAX_RETRIES) {
            retries++;
            setTimeout(tryAutoSelect, 3000);
          }
        })
        .catch(() => {
          // Engine not ready yet — retry
          if (!cancelled && retries < MAX_RETRIES) {
            retries++;
            setTimeout(tryAutoSelect, 3000);
          }
        });
    };

    // Initial check after a short delay (let engine boot)
    setTimeout(tryAutoSelect, 1500);
    return () => { cancelled = true; };
  }, []);

  // ── Auto-open Model Manager on first launch ──────────────────────
  // Separate from auto-select — only opens the modal if, after giving
  // the engine time to start, there are genuinely no models available.
  useEffect(() => {
    if (sessionStorage.getItem('mm-auto-dismissed')) return;

    const timer = setTimeout(() => {
      modelApi.list()
        .then((data) => {
          const allModels = [
            ...(data?.models?.dit || []),
            ...(data?.models?.lm || []),
            ...(data?.models?.vae || []),
          ];
          if (allModels.length === 0) {
            setShowModelManager(true);
          }
        })
        .catch(() => {
          // Engine still not running after 8s — likely no models at all
          setShowModelManager(true);
        });
    }, 8000); // 8s delay: engine needs time to scan models + cuBLAS download

    return () => clearTimeout(timer);
  }, []);

  const handleOpen = useCallback((id: SectionId) => {
    setOpenSection(id);
  }, []);

  // Only close if the requesting section is still the one that's open.
  // Prevents the leaving section's delayed close from killing a newly-opened neighbour.
  const handleClose = useCallback((id: SectionId) => {
    setOpenSection(prev => prev === id ? null : prev);
  }, []);

  return (
    <div className="flex-shrink-0 relative z-40 bg-white/95 dark:bg-zinc-900/95 border-b border-zinc-200 dark:border-white/5"
         style={{ backdropFilter: 'blur(20px)' }}>
      <div className="flex items-stretch">
        {/* Logo */}
        <div className="flex items-center justify-center flex-shrink-0 border-r border-zinc-200 dark:border-white/5" style={{ width: '199px', backgroundColor: '#000' }}>
          <img src="/logo.webp" alt="HOT-Step" style={{ width: '140px' }} className="h-auto object-contain" draggable={false} />
        </div>

        {/* Sections — separated by dividers */}
        <div className="flex-1 flex items-stretch divide-x divide-white/5">
          <DiscoPulseWrapper hue={0} stem="snare" className="flex-1 min-w-0">
          <BarSection
            id="models"
            label={t('globalBar.models')}
            icon={<Cpu size={14} />}
            badge={<ModelsBadge />}
            accentColor="pink"
            isOpen={openSection === 'models'}
            onOpen={() => handleOpen('models')}
            onClose={() => handleClose('models')}
          >
            <ModelsDropdown />
          </BarSection>
          </DiscoPulseWrapper>

          <DiscoPulseWrapper hue={72} stem="snare" className="flex-1 min-w-0">
          <BarSection
            id="adapters"
            label={t('globalBar.adapters')}
            icon={<Plug size={14} />}
            badge={<AdaptersBadge />}
            accentColor="emerald"
            isOpen={openSection === 'adapters'}
            onOpen={() => handleOpen('adapters')}
            onClose={() => handleClose('adapters')}
          >
            <AdaptersDropdown />
          </BarSection>
          </DiscoPulseWrapper>

          <DiscoPulseWrapper hue={144} stem="snare" className="flex-1 min-w-0">
          <BarSection
            id="generation"
            label={t('globalBar.generation')}
            icon={<Sliders size={14} />}
            badge={<GenerationBadge />}
            accentColor="sky"
            isOpen={openSection === 'generation'}
            onOpen={() => handleOpen('generation')}
            onClose={() => handleClose('generation')}
          >
            <GenerationDropdown />
          </BarSection>
          </DiscoPulseWrapper>

          <DiscoPulseWrapper hue={216} stem="snare" className="flex-1 min-w-0">
          <BarSection
            id="lm"
            label={t('globalBar.lm')}
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
          </DiscoPulseWrapper>

          <DiscoPulseWrapper hue={288} stem="snare" className="flex-1 min-w-0">
          <BarSection
            id="postprocessing"
            label={t('globalBar.postProcessing')}
            icon={<AudioWaveform size={14} />}
            badge={<PostProcessingBadge />}
            accentColor="amber"
            isOpen={openSection === 'postprocessing'}
            onOpen={() => handleOpen('postprocessing')}
            onClose={() => handleClose('postprocessing')}
            headerToggle={
              <ToggleSwitch
                checked={gp.postProcessingEnabled}
                onChange={(on) => gp.setPostProcessingEnabled(on)}
                accentColor="amber"
              />
            }
          >
            <PostProcessingDropdown />
          </BarSection>
          </DiscoPulseWrapper>
        </div>

        {/* Right — MonitorBar when active, otherwise Export/Import + VRAM */}
        <div className={`flex items-center gap-2 flex-shrink-0 px-3 border-l border-zinc-200 dark:border-white/5 transition-all overflow-hidden ${monitoring ? 'w-[300px]' : 'w-[240px]'}`}>
          {monitoring ? (
            <MonitorBar />
          ) : (
            <>
              {/* Mini version of the BarSection tabs to the left */}
              <button onClick={() => setShowProfiles(true)} title={t('globalBar.profiles')}
                className="group flex items-center gap-1.5 px-2 py-1.5 rounded-lg hover:bg-pink-500/10 transition-colors duration-150">
                <Bookmark size={13} className="flex-shrink-0 text-zinc-500 group-hover:text-pink-400 transition-colors duration-150" />
                <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-600 dark:text-zinc-400 group-hover:text-zinc-800 dark:group-hover:text-zinc-200 transition-colors duration-150">
                  {t('globalBar.profiles')}
                </span>
              </button>
              <div className="w-px h-4 bg-white/5" />
              <VramIndicator compact />
            </>
          )}
        </div>
      </div>

      {/* Parameter Profiles Modal */}
      {showProfiles && <ProfilesModal onClose={() => setShowProfiles(false)} />}

      {/* Model Manager Modal — rendered here (always mounted) so auto-open works */}
      {showModelManager && (
        <ModelManagerModal onClose={() => {
          setShowModelManager(false);
          sessionStorage.setItem('mm-auto-dismissed', '1');
        }} />
      )}
    </div>
  );
};
