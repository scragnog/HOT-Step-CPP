// GlobalParamBar.tsx — Horizontal top bar with hover-to-expand engine config sections
//
// Renders 5 sections: Models, Adapters, Generation, LM/Thinking, Post-Processing.
// Each section shows a summary badge and expands on hover to reveal controls.
// Sits full-width at the top of the entire window (above sidebar).

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Cpu, Plug, Sliders, Brain, AudioWaveform, Bookmark, AlertTriangle, Download, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { BarSection, ToggleSwitch } from './BarSection';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { useCapabilities } from '../../hooks/useCapabilities';
import { modelApi } from '../../services/api';
import { ModelManagerModal } from '../model-manager/ModelManagerModal';
import { ModelsDropdown, ModelsBadge } from './ModelsDropdown';
import { BackendModelsDropdown, BackendModelsBadge } from './BackendModelsDropdown';
import { BackendToggle } from './BackendToggle';
import { AdaptersDropdown, AdaptersBadge } from './AdaptersDropdown';
import { Mm3LmAdapterDropdown, Mm3LmAdapterBadge } from './Mm3LmAdapterDropdown';
import { GenerationDropdown, GenerationBadge } from './GenerationDropdown';
import { BackendGenerationDropdown, BackendGenerationBadge } from './BackendGenerationDropdown';
import { LmThinkingDropdown, LmThinkingBadge } from './LmThinkingDropdown';
import { BackendLmDropdown, BackendLmBadge } from './BackendLmDropdown';
import { PostProcessingDropdown, PostProcessingBadge } from './PostProcessingDropdown';
import { VramIndicator } from '../shared/VramIndicator';
import { DiscoPulseWrapper } from '../shared/DiscoPulseWrapper';
import { MonitorBar } from './MonitorBar';
import { useVstChainStore } from '../../stores/vstChainStore';
import { ProfilesModal } from './ProfilesModal';

type SectionId = 'models' | 'adapters' | 'generation' | 'lm' | 'postprocessing' | null;

/** Placeholder body for a cluster whose subsystem the ACTIVE backend doesn't
 *  have yet. The section stays in the bar (its absence reads as a bug) but says
 *  plainly that there is nothing to configure, rather than showing ACE's
 *  controls where they would do nothing. */
const NotYetPanel: React.FC<{ feature: string }> = ({ feature }) => (
  <p className="text-[11px] text-zinc-500 leading-relaxed">
    {feature} isn’t supported by the active backend yet. This section will fill in
    as the capability lands.
  </p>
);

const UnsupportedBadge: React.FC = () => (
  <span className="text-[10px] text-zinc-600 dark:text-zinc-500 italic truncate">not available</span>
);

/** Holds a hidden cluster's slot in the bar.
 *
 *  Every section is `flex-1`, so simply omitting one makes the survivors share
 *  the freed width and the whole bar re-flows — the tabs end up noticeably
 *  wider and in different places than in ACE mode. Rendering an inert slot of
 *  the same width keeps section size and position identical across backends,
 *  leaving a gap where the missing cluster would be. */
const SectionSpacer: React.FC = () => (
  <div className="flex-1 min-w-0" aria-hidden="true" />
);

export const GlobalParamBar: React.FC = () => {
  const { t } = useTranslation();
  const [openSection, setOpenSection] = useState<SectionId>(null);
  const gp = useGlobalParams();
  const monitoring = useVstChainStore(s => s.monitoring);
  const { capabilities } = useCapabilities();

  // Capability gating (docs/plans/multi-backend-architecture.md §4.5,
  // §2 principle 2): undefined/loading capabilities default to SHOWING every
  // cluster (ACE behavior today) — never flash-hide while /api/capabilities
  // is in flight. Only hide once a manifest has actually loaded and says no.
  //
  // PostProcessingDropdown has no dedicated capability flag in the manifest
  // (BackendFeatureCapabilities has no stems/postprocess-shaped field) — it
  // mixes ACE-VAE-specific panels (PP-VAE, Spectral Lifter) with genuinely
  // backend-agnostic ones (Mastering, VST chain, per §3.4 of the plan doc).
  // Gated behind `plugins` as the closest existing flag per the task brief;
  // a future finer-grained pass could split the agnostic sub-panels out.
  // Models is gated on its own `models` flag, not on `lm`: MiniMax-Music3 has
  // a selectable quant ladder but no ACE-style LM stage, and the old
  // `features.lm` gate hid its model picker entirely.
  const showModels = !capabilities || capabilities.features.models !== false;
  // Which picker: backends that hold model choice as engine state (MM3 loads
  // and pins one quant) get the generic bucket-driven one; ACE keeps its own,
  // which writes per-request globalParams. Keyed on the capability, not on the
  // backend id (plan §2 principle 2).
  const useBackendModelPicker = !!capabilities && capabilities.features.lm === false;

  // Adapters / Post-Processing stay VISIBLE for every backend and render an
  // empty placeholder when the subsystem doesn't exist yet, rather than
  // vanishing — the clusters are part of the app's shape, and a missing
  // section reads as a bug. Their contents are still capability-gated inside.
  const showAdapters = true;
  const showPostProcessing = true;
  const adaptersSupported = !capabilities || capabilities.features.adapters;
  // A backend can have adapters without having ACE's adapter STACK: MM3's are
  // runtime LM LoRAs with their own strength dials and nothing else from that
  // UI (no merge/runtime modes, no per-section masking, no DiT group scales).
  // So the cluster has a third state between "ACE's panel" and "not yet".
  const lmAdaptersSupported = !!capabilities && capabilities.features.lmAdapters === true;
  // Post-processing is NOT all-or-nothing. The VST chain, reference mastering
  // and StableStep read the sample rate from the audio rather than assuming
  // ACE's 48 kHz, so they work for any backend; only PP-VAE re-encode and
  // Spectral Lifter are genuinely ACE-model-coupled. Gate the cluster on the
  // agnostic subset and let the dropdown hide the coupled panels.
  const postProcessingSupported = !capabilities
    || capabilities.features.plugins
    || capabilities.features.postProcess !== false;

  // Generation is NOT only plugins — the seed lives here, and seed is
  // backend-agnostic. Gating the whole cluster on `plugins` left MiniMax-Music3
  // with no way to change the seed, so every render of a given prompt came back
  // identical. Show it whenever the backend has plugins, a seed, or any
  // declared extension knob; the contents pick themselves below.
  const backendHasGenControls = !!capabilities &&
    (capabilities.core?.seed !== false || (capabilities.extensions?.length ?? 0) > 0);
  const showGeneration = !capabilities || capabilities.features.plugins || backendHasGenControls;
  // ACE's dropdown is built around the Lua plugin registry; a backend without
  // it gets the generic seed + declared-extensions cluster instead.
  const useBackendGenPicker = !!capabilities && !capabilities.features.plugins;

  // The LM cluster is not ACE-only. `features.lm` means "has ACE's CoT
  // metadata LM" — a stage that is optional and can be switched off. A backend
  // whose LM is an autoregressive planner has no such switch but very much has
  // LM controls, and it says so by tagging declared knobs `group: 'lm'`. Both
  // get the tab; only the first gets the on/off toggle, because only there does
  // skipping mean anything.
  const aceLm = !capabilities || capabilities.features.lm;
  const backendLmParams = (capabilities?.extensions ?? []).some(p => p.group === 'lm');
  const showLm = aceLm || backendLmParams;

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

  // ── MiniMax-Music3 "models missing" banner ────────────────────────
  // capabilities().core.modelsMissing (server: backends/minimax/index.ts) is
  // true only when the MM3 backend is active, the engine is reachable, and
  // its weight files specifically weren't found — never for engine-down or
  // corrupt-file cases. Surfacing is opt-in: a dismissible banner pointing at
  // the Model Manager, never an auto-started 24 GB download.
  const mm3ModelsMissing = capabilities?.backend === 'minimax-m3' && capabilities.core?.modelsMissing === true;
  const [mm3BannerDismissed, setMm3BannerDismissed] = useState(false);
  const lastBackendRef = useRef<string | undefined>(undefined);
  useEffect(() => {
    const b = capabilities?.backend;
    // Re-arm the banner whenever the user switches INTO minimax-m3, so a
    // dismissal doesn't stick forever across backend switches.
    if (b === 'minimax-m3' && lastBackendRef.current !== 'minimax-m3') {
      setMm3BannerDismissed(false);
    }
    lastBackendRef.current = b;
  }, [capabilities?.backend]);

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
          {showModels && (
          <DiscoPulseWrapper hue={0} stem="snare" className="flex-1 min-w-0">
          <BarSection
            id="models"
            label={t('globalBar.models')}
            icon={<Cpu size={14} />}
            badge={useBackendModelPicker ? <BackendModelsBadge /> : <ModelsBadge />}
            accentColor="pink"
            isOpen={openSection === 'models'}
            onOpen={() => handleOpen('models')}
            onClose={() => handleClose('models')}
          >
            {useBackendModelPicker ? <BackendModelsDropdown /> : <ModelsDropdown />}
          </BarSection>
          </DiscoPulseWrapper>
          )}
          {!showModels && <SectionSpacer />}

          <BackendToggle />

          {showAdapters && (
          <DiscoPulseWrapper hue={72} stem="snare" className="flex-1 min-w-0">
          <BarSection
            id="adapters"
            label={t('globalBar.adapters')}
            icon={<Plug size={14} />}
            badge={adaptersSupported ? <AdaptersBadge />
                 : lmAdaptersSupported ? <Mm3LmAdapterBadge />
                 : <UnsupportedBadge />}
            accentColor="emerald"
            isOpen={openSection === 'adapters'}
            onOpen={() => handleOpen('adapters')}
            onClose={() => handleClose('adapters')}
          >
            {adaptersSupported
              ? <AdaptersDropdown />
              : lmAdaptersSupported
              ? <Mm3LmAdapterDropdown />
              : <NotYetPanel feature="Adapters" />}
          </BarSection>
          </DiscoPulseWrapper>
          )}

          {showGeneration && (
          <DiscoPulseWrapper hue={144} stem="snare" className="flex-1 min-w-0">
          <BarSection
            id="generation"
            label={t('globalBar.generation')}
            icon={<Sliders size={14} />}
            badge={useBackendGenPicker ? <BackendGenerationBadge /> : <GenerationBadge />}
            accentColor="sky"
            isOpen={openSection === 'generation'}
            onOpen={() => handleOpen('generation')}
            onClose={() => handleClose('generation')}
          >
            {useBackendGenPicker ? <BackendGenerationDropdown /> : <GenerationDropdown />}
          </BarSection>
          </DiscoPulseWrapper>
          )}
          {!showGeneration && <SectionSpacer />}

          {showLm && (
          <DiscoPulseWrapper hue={216} stem="snare" className="flex-1 min-w-0">
          <BarSection
            id="lm"
            label={t('globalBar.lm')}
            icon={<Brain size={14} />}
            badge={aceLm ? <LmThinkingBadge /> : <BackendLmBadge />}
            accentColor="purple"
            isOpen={openSection === 'lm'}
            onOpen={() => handleOpen('lm')}
            onClose={() => handleClose('lm')}
            headerToggle={aceLm ? (
              <ToggleSwitch
                checked={!gp.skipLm}
                onChange={(on) => gp.setSkipLm(!on)}
                accentColor="purple"
              />
            ) : undefined}
          >
            {aceLm ? <LmThinkingDropdown /> : <BackendLmDropdown />}
          </BarSection>
          </DiscoPulseWrapper>
          )}
          {!showLm && <SectionSpacer />}

          {showPostProcessing && (
          <DiscoPulseWrapper hue={288} stem="snare" className="flex-1 min-w-0">
          <BarSection
            id="postprocessing"
            label={t('globalBar.postProcessing')}
            icon={<AudioWaveform size={14} />}
            badge={postProcessingSupported ? <PostProcessingBadge /> : <UnsupportedBadge />}
            accentColor="amber"
            isOpen={openSection === 'postprocessing'}
            onOpen={() => handleOpen('postprocessing')}
            onClose={() => handleClose('postprocessing')}
            headerToggle={postProcessingSupported ? (
              <ToggleSwitch
                checked={gp.postProcessingEnabled}
                onChange={(on) => gp.setPostProcessingEnabled(on)}
                accentColor="amber"
              />
            ) : undefined}
          >
            {postProcessingSupported
              ? <PostProcessingDropdown />
              : <NotYetPanel feature="Post-processing" />}
          </BarSection>
          </DiscoPulseWrapper>
          )}
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

      {/* MiniMax-Music3 models-missing banner — dismissible, links to the
          Model Manager rather than auto-starting a 24 GB download. */}
      {mm3ModelsMissing && !mm3BannerDismissed && (
        <div className="flex items-center gap-2 px-4 py-2 border-t border-amber-500/20 bg-amber-500/10 text-xs text-amber-700 dark:text-amber-400">
          <AlertTriangle size={14} className="flex-shrink-0" />
          <span className="flex-1">{t('globalBar.mm3ModelsMissing')}</span>
          <button
            onClick={() => setShowModelManager(true)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 text-amber-700 dark:text-amber-300 font-medium transition-colors"
          >
            <Download size={12} />
            {t('globalBar.mm3GetModels')}
          </button>
          <button
            onClick={() => setMm3BannerDismissed(true)}
            className="p-1 rounded-lg hover:bg-amber-500/15 text-amber-600 dark:text-amber-400 transition-colors"
            title={t('common.dismiss')}
          >
            <X size={12} />
          </button>
        </div>
      )}

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
