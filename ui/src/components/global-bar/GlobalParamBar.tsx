import { useQuery } from '@tanstack/react-query';
import { aceModelsOptions } from '../../queries/global-models/options';
// GlobalParamBar.tsx — Horizontal top bar with hover-to-expand engine config sections
//
// Renders 5 sections: Models, Adapters, Generation, LM/Thinking, Post-Processing.
// Each section shows a summary badge and expands on hover to reveal controls.
// Sits full-width at the top of the entire window (above sidebar).

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Cpu, Plug, Sliders, Brain, AudioWaveform, Bookmark, AlertTriangle, Download, X, Menu, ChevronDown, Settings, ListMusic } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { BarSection, ToggleSwitch, ACCENT_STYLES } from './BarSection';
import { useGlobalParams, useGlobalParamsStore } from '../../context/GlobalParamsContext';
import { useCapabilities } from '../../hooks/useCapabilities';
import { useBackendStore } from '../../stores/backendStore';
import { ModelManagerModal } from '../model-manager/ModelManagerModal';
import { ModelsDropdown, ModelsBadge } from './ModelsDropdown';
import { BackendModelsDropdown, BackendModelsBadge } from './BackendModelsDropdown';
import { AdaptersDropdown, AdaptersBadge } from './AdaptersDropdown';
import { Mm3LmAdapterDropdown, Mm3LmAdapterBadge } from './Mm3LmAdapterDropdown';
import { Yue2LmAdapterDropdown, Yue2LmAdapterBadge } from './Yue2LmAdapterDropdown';
import { HeartmulaAdapterDropdown, HeartmulaAdapterBadge } from './HeartmulaAdapterDropdown';
import { GenerationDropdown, GenerationBadge } from './GenerationDropdown';
import { BackendGenerationDropdown, BackendGenerationBadge } from './BackendGenerationDropdown';
import { LmThinkingDropdown, LmThinkingBadge } from './LmThinkingDropdown';
import { BackendLmDropdown, BackendLmBadge } from './BackendLmDropdown';
import { PostProcessingDropdown, PostProcessingBadge } from './PostProcessingDropdown';
import { DiscoPulseWrapper } from '../shared/DiscoPulseWrapper';
import { MonitorBar } from './MonitorBar';
import { useVstChainStore } from '../../stores/vstChainStore';
import { ProfilesModal } from './ProfilesModal';
import { QueryError } from '../shared/QueryError';
import { healthOptions } from '../../queries/shell-queue/settings';
import { engineNotice, noModelsRefetchInterval } from '../shared/noModelsPoll';
import { usePlaylist } from '../lyric-studio/playlistStore';

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

// ── SheetSection — accordion wrapper for the mobile settings sheet ─────────
// Renders inline (no absolute positioning) unlike BarSection. Shares the same
// accent palette so section identity is visually consistent.

interface SheetSectionProps {
  label: string;
  icon: React.ReactNode;
  badge: React.ReactNode;
  accentColor?: string;
  children: React.ReactNode;
  isOpen: boolean;
  onToggle: () => void;
  headerToggle?: React.ReactNode;
}

const SheetSection: React.FC<SheetSectionProps> = ({
  label, icon, badge, accentColor = 'pink', children,
  isOpen, onToggle, headerToggle,
}) => {
  const accent = ACCENT_STYLES[accentColor] || ACCENT_STYLES.pink;
  return (
    <div className="border border-zinc-200 dark:border-white/10 rounded-xl overflow-hidden">
      <div
        role="button"
        tabIndex={0}
        onClick={onToggle}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onToggle(); } }}
        className={`flex items-center gap-3 px-4 py-3 cursor-pointer transition-all duration-150 border-b-2
          ${isOpen ? `${accent.bgHover} ${accent.border}` : `${accent.bg} border-transparent ${accent.hoverBg}`}`}
      >
        <span className={`flex-shrink-0 transition-colors duration-150 ${isOpen ? accent.iconColor : 'text-zinc-500'}`}>
          {icon}
        </span>
        <span className={`text-[12px] font-semibold uppercase tracking-wider flex-shrink-0 transition-colors duration-150 ${
          isOpen ? 'text-zinc-800 dark:text-zinc-200' : 'text-zinc-600 dark:text-zinc-400'
        }`}>
          {label}
        </span>
        {headerToggle && (
          <div className="flex-shrink-0" onClick={e => e.stopPropagation()}>
            {headerToggle}
          </div>
        )}
        <div className="flex-1 min-w-0 flex justify-end items-center gap-2">
          <div className="min-w-0 flex-1 flex justify-end">{badge}</div>
          <ChevronDown
            size={14}
            className={`flex-shrink-0 text-zinc-500 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
          />
        </div>
      </div>
      {isOpen && (
        <div className="p-4 space-y-3 bg-white dark:bg-zinc-900">
          {children}
        </div>
      )}
    </div>
  );
};

export const GlobalParamBar: React.FC<{ onOpenMobileNav?: () => void; onOpenMobilePlaylist?: () => void }> = ({ onOpenMobileNav, onOpenMobilePlaylist }) => {
  const { t } = useTranslation();
  const { items: playlistItems } = usePlaylist();
  const [openSection, setOpenSection] = useState<SectionId>(null);
  const gp = useGlobalParams();
  const monitoring = useVstChainStore(s => s.monitoring);
  const { capabilities } = useCapabilities();
  const backends = useBackendStore(s => s.backends);
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  // What to show while the active backend's manifest has not arrived.
  //
  // Every gate below used to default to ACE's answer whenever `capabilities`
  // was null, on the reasoning that the null window is a brief one at first
  // paint. It is not always brief: a manifest reporting the engine down is
  // deliberately never cached (backendStore, issue #153), so any backend whose
  // readiness probe answers "down" leaves this null for the whole session —
  // and the top bar then shows ACE's models, ACE's adapter stack, ACE's
  // sampler plugins and ACE's thinking LM while a completely different backend
  // is the one generating. That is not a loading state, it is a lie about
  // which engine the user is driving.
  //
  // So the fallback is now "ACE's shape only when ACE is actually active".
  // Another backend with no manifest yet gets the honest empty placeholder.
  const assumeAce = activeBackendId === 'ace';

  // Capability gating (docs/plans/multi-backend-architecture.md §4.5,
  // §2 principle 2): a loaded manifest decides; an absent one falls back to
  // `assumeAce` above, so ACE never flash-hides its own clusters while
  // /api/capabilities is in flight and no other backend ever borrows them.
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
  const showModels = capabilities ? capabilities.features.models !== false : assumeAce;
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
  const adaptersSupported = capabilities ? capabilities.features.adapters : assumeAce;
  // A backend can have adapters without having ACE's adapter STACK: MM3's are
  // runtime LM LoRAs with their own strength dials and nothing else from that
  // UI (no merge/runtime modes, no per-section masking, no DiT group scales).
  // So the cluster has a third state between "ACE's panel" and "not yet".
  const lmAdaptersSupported = !!capabilities && capabilities.features.lmAdapters === true;
  // ...and a second split inside that third state. MM3's LM LoRA rides on the
  // generation request (dials are live per render); YuE2's is merged into the
  // resident weights at load, so choosing one is a POST that evicts the model.
  // Same cluster, different panel — keyed on the capability, never the backend
  // id (plan §2 principle 2).
  const lmAdaptersAreEngineState = lmAdaptersSupported
    && capabilities.features.lmAdapterSelectable === true;
  const heartmulaAdapters = lmAdaptersAreEngineState && activeBackendId === 'heartmula';
  // Post-processing is NOT all-or-nothing. The VST chain, reference mastering
  // and StableStep read the sample rate from the audio rather than assuming
  // ACE's 48 kHz, so they work for any backend; only PP-VAE re-encode and
  // Spectral Lifter are genuinely ACE-model-coupled. Gate the cluster on the
  // agnostic subset and let the dropdown hide the coupled panels.
  const postProcessingSupported = capabilities
    ? (capabilities.features.plugins || capabilities.features.postProcess !== false)
    : assumeAce;

  // Generation is NOT only plugins — the seed lives here, and seed is
  // backend-agnostic. Gating the whole cluster on `plugins` left MiniMax-Music3
  // with no way to change the seed, so every render of a given prompt came back
  // identical. Show it whenever the backend has plugins, a seed, or any
  // declared extension knob; the contents pick themselves below.
  const backendHasGenControls = !!capabilities &&
    (capabilities.core?.seed !== false || (capabilities.extensions?.length ?? 0) > 0);
  const showGeneration = capabilities ? (capabilities.features.plugins || backendHasGenControls) : assumeAce;
  // ACE's dropdown is built around the Lua plugin registry; a backend without
  // it gets the generic seed + declared-extensions cluster instead.
  const useBackendGenPicker = !capabilities || !capabilities.features.plugins;

  // The LM cluster is not ACE-only. `features.lm` means "has ACE's CoT
  // metadata LM" — a stage that is optional and can be switched off. A backend
  // whose LM is an autoregressive planner has no such switch but very much has
  // LM controls, and it says so by tagging declared knobs `group: 'lm'`. Both
  // get the tab; only the first gets the on/off toggle, because only there does
  // skipping mean anything.
  const aceLm = capabilities ? capabilities.features.lm : assumeAce;
  const backendLmParams = (capabilities?.extensions ?? []).some(p => p.group === 'lm');
  const showLm = aceLm || backendLmParams;

  // ── Auto-select models when engine becomes ready ────────────────
  // Polls the engine until it returns a model list, then auto-selects
  // the first available model for any empty slot. Runs independently
  // of the Model Manager modal state.
  const [showModelManager, setShowModelManager] = useState(false);
  const [showProfiles, setShowProfiles] = useState(false);
  const [showMobileSettings, setShowMobileSettings] = useState(false);
  const [sheetOpenSection, setSheetOpenSection] = useState<SectionId>(null);
  const mobileSheetRef = useRef<HTMLDivElement>(null);

  // Settings sheet a11y: focus into the sheet on open, Escape closes, focus returns to the opener on close.
  useEffect(() => {
    if (!showMobileSettings) return;
    const opener = document.activeElement as HTMLElement | null;
    mobileSheetRef.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setShowMobileSettings(false); };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      if (opener && document.contains(opener)) opener.focus();
    };
  }, [showMobileSettings]);

  // Close the sheet when the viewport crosses into the desktop layout (where its opener is hidden).
  useEffect(() => {
    const mq = window.matchMedia('(min-width: 1024px) and (orientation: landscape)');
    const onChange = (e: MediaQueryListEvent) => { if (e.matches) setShowMobileSettings(false); };
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  const modelsQuery = useQuery(aceModelsOptions());
  const healthQuery = useQuery({ ...healthOptions(), refetchInterval: noModelsRefetchInterval });
  const notice = engineNotice(healthQuery.data?.engine);
  const availableModels = modelsQuery.data;
  useEffect(() => {
    const models = availableModels?.models;
    if (!models) return;
    const params = useGlobalParamsStore.getState();
    if (models.dit?.length && !params.ditModel) params.setDitModel(models.dit[0]);
    if (models.lm?.length && !params.lmModel) params.setLmModel(models.lm[0]);
    if (models.vae?.length && !params.vaeModel) params.setVaeModel(models.vae[0]);
    if (models.embedding?.length && !params.embeddingModel) params.setEmbeddingModel(models.embedding[0]);
  }, [availableModels]);

  // ── Auto-open Model Manager on first launch ──────────────────────
  // Separate from auto-select — only opens the modal if, after giving
  // the engine time to start, there are genuinely no models available.
  useEffect(() => {
    if (sessionStorage.getItem('mm-auto-dismissed')) return;
    if (notice !== 'no-models') return;

    const timer = setTimeout(() => {
      setShowModelManager(true);
    }, 8000);
    return () => clearTimeout(timer);
  }, [notice]);

  // ── "Models missing" banner (generic — any backend) ─────────────────
  // capabilities().core.modelsMissing is true only when the ACTIVE backend is
  // reachable and ITS OWN weight files specifically weren't found — never for
  // engine-down or corrupt-file cases (docs/plans/yue2/01-seam-design.md
  // §4.2). Was gated on `backend === 'minimax-m3'`; any backend can report it
  // now (YuE2 will), so the gate reads only the flag. Surfacing stays
  // opt-in: a dismissible banner pointing at the Model Manager, never an
  // auto-started multi-GB download.
  const modelsMissing = capabilities?.core?.modelsMissing === true;
  const modelsMissingHint = typeof capabilities?.core?.modelsMissingHint === 'string'
    ? capabilities.core.modelsMissingHint
    : undefined;
  const modelsMissingBackendName =
    backends.find(b => b.id === capabilities?.backend)?.displayName || capabilities?.backend || '';
  const [modelsMissingBannerDismissed, setModelsMissingBannerDismissed] = useState(false);
  const lastBackendRef = useRef<string | undefined>(undefined);
  useEffect(() => {
    const b = capabilities?.backend;
    // Re-arm the banner on ANY backend switch (not just entering MM3), so a
    // dismissal on one backend doesn't stick forever after switching to
    // another whose models also happen to be missing.
    if (b !== lastBackendRef.current) {
      setModelsMissingBannerDismissed(false);
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

  const toggleSheetSection = useCallback((id: SectionId) => {
    setSheetOpenSection(prev => prev === id ? null : id);
  }, []);

  return (
    <div className="flex-shrink-0 relative z-40 bg-white/95 dark:bg-zinc-900/95 border-b border-zinc-200 dark:border-white/5"
         style={{ backdropFilter: 'blur(20px)' }}>
      {modelsQuery.isError && (
        <div className="px-3 pt-1.5">
          <QueryError error={modelsQuery.error} onRetry={() => void modelsQuery.refetch()} compact />
        </div>
      )}

      {/* Mobile compact bar — hamburger + logo + Settings button (hidden >= 1024px) */}
      <div className="hidden max-[1023px]:flex portrait:flex items-center gap-3 px-3 py-2">
        <button
          onClick={onOpenMobileNav}
          className="p-2 rounded-lg hover:bg-zinc-100 dark:hover:bg-white/10 text-zinc-600 dark:text-zinc-400 transition-colors"
          aria-label={t('app.openNav')}
          title={t('app.openNav')}
        >
          <Menu size={20} />
        </button>
        <img src="/logo.webp" alt="HOT-Step" style={{ height: '28px' }} className="w-auto object-contain" draggable={false} />
        <div className="flex-1" />
        <button
          onClick={onOpenMobilePlaylist}
          className="relative p-2 rounded-lg bg-zinc-200 dark:bg-white/5 hover:bg-zinc-300 dark:hover:bg-white/10 text-zinc-600 dark:text-zinc-400 transition-colors"
          aria-label={t('playlist.title')}
          title={t('playlist.title')}
        >
          <ListMusic size={20} />
          {playlistItems.length > 0 && (
            <span className="absolute -top-1 -right-1 min-w-[14px] h-[14px] px-0.5 flex items-center justify-center rounded-full bg-pink-500 text-white text-[8px] font-bold">
              {playlistItems.length}
            </span>
          )}
        </button>
        <button
          onClick={() => setShowMobileSettings(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-100 dark:bg-white/10 hover:bg-zinc-200 dark:hover:bg-white/15 text-zinc-600 dark:text-zinc-400 transition-colors text-[12px] font-semibold"
          aria-label={t('globalBar.mobileSettingsTitle')}
        >
          <Settings size={14} />
          {t('globalBar.mobileSettingsTitle')}
        </button>
      </div>

      {/* Desktop bar — full 5-section layout (hidden < 1024px) */}
      <div className="flex max-[1023px]:hidden portrait:hidden items-stretch">
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
            {useBackendModelPicker ? <BackendModelsDropdown key={activeBackendId} /> : <ModelsDropdown />}
          </BarSection>
          </DiscoPulseWrapper>
          )}
          {!showModels && <SectionSpacer />}


          {showAdapters && (
          <DiscoPulseWrapper hue={72} stem="snare" className="flex-1 min-w-0">
          <BarSection
            id="adapters"
            label={t('globalBar.adapters')}
            icon={<Plug size={14} />}
            badge={adaptersSupported ? <AdaptersBadge />
                 : heartmulaAdapters ? <HeartmulaAdapterBadge />
                 : lmAdaptersAreEngineState ? <Yue2LmAdapterBadge />
                 : lmAdaptersSupported ? <Mm3LmAdapterBadge />
                 : <UnsupportedBadge />}
            accentColor="emerald"
            isOpen={openSection === 'adapters'}
            onOpen={() => handleOpen('adapters')}
            onClose={() => handleClose('adapters')}
          >
            {adaptersSupported
              ? <AdaptersDropdown />
              : heartmulaAdapters
              ? <HeartmulaAdapterDropdown />
              : lmAdaptersAreEngineState
              ? <Yue2LmAdapterDropdown />
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

        {/* Right — MonitorBar when active, otherwise Profiles (VRAM lives in the Terminal box) */}
        <div className={`flex items-center gap-2 flex-shrink-0 px-3 border-l border-zinc-200 dark:border-white/5 transition-all overflow-hidden ${monitoring ? 'w-[300px]' : 'w-auto'}`}>
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
            </>
          )}
        </div>
      </div>

      {/* Models-missing banner (any backend) — dismissible, links to the
          Model Manager rather than auto-starting a multi-GB download. */}
      {modelsMissing && !modelsMissingBannerDismissed && (
        <div className="flex items-center gap-2 px-4 py-2 border-t border-amber-500/20 bg-amber-500/10 text-xs text-amber-700 dark:text-amber-400">
          <AlertTriangle size={14} className="flex-shrink-0" />
          <span className="flex-1">
            {t('globalBar.modelsMissing', {
              backend: modelsMissingBackendName,
              hint: modelsMissingHint ? ` — ${modelsMissingHint}` : '',
            })}
          </span>
          <button
            onClick={() => setShowModelManager(true)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 text-amber-700 dark:text-amber-300 font-medium transition-colors"
          >
            <Download size={12} />
            {t('globalBar.getModels')}
          </button>
          <button
            onClick={() => setModelsMissingBannerDismissed(true)}
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

      {/* Mobile Settings Sheet — full-height overlay for narrow screens */}
      {showMobileSettings && createPortal(
        <div
          className="fixed inset-0 z-[55] flex"
          role="dialog"
          aria-modal="true"
          aria-label={t('globalBar.mobileSettingsTitle')}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setShowMobileSettings(false)}
          />
          {/* Sheet panel */}
          <div
            data-portal-layer
            ref={mobileSheetRef}
            tabIndex={-1}
            className="outline-none relative ml-auto w-full max-w-[420px] h-full bg-white dark:bg-zinc-900 flex flex-col shadow-2xl shadow-black/40"
          >
            {/* Header */}
            <div className="flex items-center gap-3 px-4 py-3 border-b border-zinc-200 dark:border-white/10 flex-shrink-0">
              <Settings size={16} className="text-zinc-500" />
              <span className="flex-1 text-[13px] font-semibold text-zinc-800 dark:text-zinc-200">
                {t('globalBar.mobileSettingsTitle')}
              </span>
              <button
                onClick={() => setShowMobileSettings(false)}
                className="p-1.5 rounded-lg hover:bg-zinc-100 dark:hover:bg-white/10 text-zinc-500 transition-colors"
                aria-label={t('globalBar.mobileSettingsClose')}
              >
                <X size={16} />
              </button>
            </div>
            {/* Sections — scrollable */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {showModels && (
                <SheetSection
                  label={t('globalBar.models')}
                  icon={<Cpu size={14} />}
                  badge={useBackendModelPicker ? <BackendModelsBadge /> : <ModelsBadge />}
                  accentColor="pink"
                  isOpen={sheetOpenSection === 'models'}
                  onToggle={() => toggleSheetSection('models')}
                >
                  {useBackendModelPicker ? <BackendModelsDropdown key={activeBackendId} /> : <ModelsDropdown />}
                </SheetSection>
              )}
              <SheetSection
                label={t('globalBar.adapters')}
                icon={<Plug size={14} />}
                badge={adaptersSupported ? <AdaptersBadge />
                     : heartmulaAdapters ? <HeartmulaAdapterBadge />
                     : lmAdaptersAreEngineState ? <Yue2LmAdapterBadge />
                     : lmAdaptersSupported ? <Mm3LmAdapterBadge />
                     : <UnsupportedBadge />}
                accentColor="emerald"
                isOpen={sheetOpenSection === 'adapters'}
                onToggle={() => toggleSheetSection('adapters')}
              >
                {adaptersSupported
                  ? <AdaptersDropdown />
                  : heartmulaAdapters
                  ? <HeartmulaAdapterDropdown />
                  : lmAdaptersAreEngineState
                  ? <Yue2LmAdapterDropdown />
                  : lmAdaptersSupported
                  ? <Mm3LmAdapterDropdown />
                  : <NotYetPanel feature="Adapters" />}
              </SheetSection>
              {showGeneration && (
                <SheetSection
                  label={t('globalBar.generation')}
                  icon={<Sliders size={14} />}
                  badge={useBackendGenPicker ? <BackendGenerationBadge /> : <GenerationBadge />}
                  accentColor="sky"
                  isOpen={sheetOpenSection === 'generation'}
                  onToggle={() => toggleSheetSection('generation')}
                >
                  {useBackendGenPicker ? <BackendGenerationDropdown /> : <GenerationDropdown />}
                </SheetSection>
              )}
              {showLm && (
                <SheetSection
                  label={t('globalBar.lm')}
                  icon={<Brain size={14} />}
                  badge={aceLm ? <LmThinkingBadge /> : <BackendLmBadge />}
                  accentColor="purple"
                  isOpen={sheetOpenSection === 'lm'}
                  onToggle={() => toggleSheetSection('lm')}
                  headerToggle={aceLm ? (
                    <ToggleSwitch
                      checked={!gp.skipLm}
                      onChange={(on) => gp.setSkipLm(!on)}
                      accentColor="purple"
                    />
                  ) : undefined}
                >
                  {aceLm ? <LmThinkingDropdown /> : <BackendLmDropdown />}
                </SheetSection>
              )}
              {showPostProcessing && (
                <SheetSection
                  label={t('globalBar.postProcessing')}
                  icon={<AudioWaveform size={14} />}
                  badge={postProcessingSupported ? <PostProcessingBadge /> : <UnsupportedBadge />}
                  accentColor="amber"
                  isOpen={sheetOpenSection === 'postprocessing'}
                  onToggle={() => toggleSheetSection('postprocessing')}
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
                </SheetSection>
              )}
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
};
