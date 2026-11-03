// SettingsPanel.tsx — Application settings with persistent state
//
// All settings use localStorage via usePersistedState and survive page refreshes.
// Environment settings read from / write to the server's .env file.

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Zap, Download, Tag, Loader2, Settings2,
  ChevronRight, Save,
  Key, Database, Globe, Gauge, Bug, Image as ImageIcon
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { FALLBACK_LANGUAGE, SUPPORTED_LANGUAGES } from '../../i18n';
import { engineWorkersOptions, healthOptions } from '../../queries/shell-queue/settings';
import { useEnvSettingsForm } from './useEnvSettingsForm';
import { DataFolderPanel } from './DataFolderPanel';
import { SettingsBackupPanel } from './SettingsBackupPanel';
import { UpdatesPanel } from './UpdatesPanel';
import { ReportIssuePanel } from './ReportIssuePanel';
import { StemStorage } from './StemStorage';
import { DangerZone } from './DangerZone';
import { AdapterReferencesAction } from './AdapterReferencesAction';
import { FileBrowserModal } from '../shared/FileBrowserModal';
import { StyledSelect } from '../shared/StyledSelect';
import { ParamLabel } from '../shared/ParamLabel';
import {
  SettingRow, SelectRow, Toggle,
  EnvTextRow, EnvPasswordRow, EnvPathRow, EnvSubsection,
} from './SettingsPrimitives';
import './SettingsPanel.css';
import { COVER_SETS, FALLBACK_SET, getCoverSet, setCoverSet, type CoverSet } from '../../utils/defaultCover';

export interface AppSettings {
  coResident: boolean;
  cacheLmCodes: boolean;
  speculativeDecoding: boolean;
  // Download defaults
  downloadFormat: 'wav' | 'flac' | 'opus' | 'mp3';
  downloadMp3Bitrate: number;
  downloadOpusBitrate: number;
  downloadVersion: 'original' | 'mastered' | 'both';
  downloadIncludeLatent: boolean;
  // Adapter trigger word
  triggerUseFilename: boolean;
  triggerPlacement: 'prepend' | 'append' | 'replace';
  discoKickExtract: boolean;
  // Pipeline parallelism (opt-in)
  parallelWhisper: boolean;
  parallelQualityEval: boolean;
  parallelCoverArt: boolean;
  // Generation timeout (minutes) — how long to wait before declaring a generation timed out
  generationTimeoutMinutes: number;
}

export const DEFAULT_SETTINGS: AppSettings = {
  coResident: false,
  cacheLmCodes: true,
  speculativeDecoding: true,
  downloadFormat: 'flac',
  downloadMp3Bitrate: 192,
  downloadOpusBitrate: 192,
  downloadVersion: 'mastered',
  downloadIncludeLatent: false,
  triggerUseFilename: false,
  triggerPlacement: 'prepend',
  discoKickExtract: false,
  parallelWhisper: false,
  parallelQualityEval: false,
  parallelCoverArt: false,
  generationTimeoutMinutes: 30,
};

interface SettingsPanelProps {
  settings: AppSettings;
  onSettingsChange: (settings: AppSettings) => void;
  onNukeComplete?: () => void;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  settings,
  onSettingsChange,
  onNukeComplete,
}) => {
  const { t, i18n } = useTranslation();
  const [coverSet, setCoverSetState] = useState<CoverSet>(getCoverSet);
  // The release tag the server reports; a git checkout says "dev" (#185).
  const appVersion = String(useQuery(healthOptions()).data?.version || '');

  const update = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  // ── Environment (.env) form ─────────────────────────────────────
  // The hook owns the draft/seed/save lifecycle over envOptions(); this panel
  // only renders values and calls its handlers.
  const {
    values: envValues,
    loading: envLoading,
    status: envStatus,
    dirty: envDirty,
    saving: envSaving,
    restarting,
    pendingRestartKeys,
    gpus: detectedGpus,
    change: handleEnvChange,
    save: handleEnvSave,
    restartNow: handleRestartNow,
  } = useEnvSettingsForm();

  // Engine workers for the experimental multi-GPU section: observed and polled
  // while mounted, separate from the editable env form (read-only status).
  const engineWorkers = useQuery(engineWorkersOptions()).data ?? null;

  // Subsection open/close state
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    engine: true, server: true, apiKeys: true, llmConfig: true, llmEndpoints: false,
    labeling: false, paths: true,
  });
  const toggleSection = (key: string) =>
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));

  // File browser modal state
  const [browseKey, setBrowseKey] = useState<string | null>(null);

  // The GPU picker stores a GPU UUID, not an index (issue #153 — nvidia-smi
  // and CUDA number GPUs differently, so an index could select the other card).
  // A .env written before that change still holds a bare index, so map it onto
  // the matching card for display; anything we cannot resolve gets its own
  // option rather than silently reading as "Auto".
  const gpuRaw = (envValues.CUDA_VISIBLE_DEVICES || '').trim();
  const gpuMatch = gpuRaw
    ? detectedGpus.find(g => g.uuid && g.uuid.toLowerCase() === gpuRaw.toLowerCase())
      ?? detectedGpus.find(g => String(g.index) === gpuRaw)
    : undefined;
  const gpuSelectValue = gpuRaw ? (gpuMatch ? (gpuMatch.uuid || String(gpuMatch.index)) : gpuRaw) : '';
  const gpuSelectUnknown = !!gpuRaw && !gpuMatch;

  // Multi-GPU opt-in: comma-separated UUIDs, same stable ids as the picker above.
  const workerGpuSet = new Set((envValues.ACESTEPCPP_WORKER_GPUS || '').split(',').map(x => x.trim().toLowerCase()).filter(Boolean));
  const toggleWorkerGpu = (uuid: string) => {
    const key = uuid.toLowerCase();
    const chosen = detectedGpus.filter(g => g.uuid && (g.uuid.toLowerCase() === key ? !workerGpuSet.has(key) : workerGpuSet.has(g.uuid.toLowerCase())));
    handleEnvChange('ACESTEPCPP_WORKER_GPUS', chosen.map(g => g.uuid).join(','));
  };

  const handleBrowse = (key: string) => {
    setBrowseKey(key);
  };

  const handleBrowseSelect = (selectedPath: string) => {
    if (browseKey) {
      handleEnvChange(browseKey, selectedPath);
    }
    setBrowseKey(null);
  };

  type TabId = 'general' | 'environment' | 'ai' | 'storage' | 'performance' | 'report';
  const [activeTab, setActiveTab] = useState<TabId>('general');

  const tabs: Array<{ id: TabId; label: string; icon: React.ReactNode }> = [
    { id: 'general',     label: t('settings.tabs.general'),        icon: <Zap size={15} className="settings-tab-icon" /> },
    { id: 'environment', label: t('settings.tabs.environment'),     icon: <Settings2 size={15} className="settings-tab-icon" /> },
    { id: 'ai',          label: t('settings.tabs.ai'),     icon: <Key size={15} className="settings-tab-icon" /> },
    { id: 'storage',     label: t('settings.tabs.storage'),  icon: <Database size={15} className="settings-tab-icon" /> },
    { id: 'performance', label: 'Performance', icon: <Gauge size={15} className="settings-tab-icon" /> },
    { id: 'report',      label: t('settings.tabs.report', 'Report an issue'), icon: <Bug size={15} className="settings-tab-icon" /> },
  ];

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h1 className="settings-title">{t('settings.title')}{appVersion && <span className="ml-3 align-middle text-xs font-normal text-zinc-500 dark:text-zinc-400">v{appVersion}</span>}</h1>
        <p className="settings-subtitle">
          {t('settings.subtitle')}
        </p>
      </div>

      {/* ── Tab Navigation ── */}
      <div className="settings-tabs">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={`settings-tab${activeTab === t.id ? ' settings-tab--active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </div>

      {/* ═══════════════ ENVIRONMENT TAB ═══════════════ */}
      {activeTab === 'environment' && (
      <div className="settings-section">
        <div className="settings-section-header">
          <Settings2 size={16} className="settings-section-icon" />
          <span className="settings-section-title">{t('settings.tabs.environment')}</span>
        </div>

        {envLoading ? (
          <div className="setting-row" style={{ justifyContent: 'center' }}>
            <Loader2 size={16} className="animate-spin" style={{ color: 'rgba(255,255,255,0.4)' }} />
            <span style={{ marginLeft: '8px', color: 'rgba(255,255,255,0.4)', fontSize: '0.8125rem' }}>{t('settings.env.loading')}</span>
          </div>
        ) : (
          <>
            {/* Engine */}
            <EnvSubsection title={t('settings.env.engine')} isOpen={openSections.engine} onToggle={() => toggleSection('engine')}>
              <EnvPathRow envKey="ACESTEPCPP_MODELS" label={t('settings.env.modelsDir')} description={t('settings.env.modelsDesc')}
                value={envValues.ACESTEPCPP_MODELS || ''} onChange={handleEnvChange} onBrowse={handleBrowse} />
              <EnvPathRow envKey="ACESTEPCPP_ADAPTERS" label={t('settings.env.adaptersDir')} description={t('settings.env.adaptersDesc')}
                value={envValues.ACESTEPCPP_ADAPTERS || ''} onChange={handleEnvChange} onBrowse={handleBrowse} />
              <EnvTextRow envKey="ACESTEPCPP_PORT" label={t('settings.env.enginePort')} description={t('settings.env.enginePortDesc')}
                value={envValues.ACESTEPCPP_PORT || ''} onChange={handleEnvChange} type="number" placeholder="8085" />
              <EnvTextRow envKey="ACESTEPCPP_HOST" label={t('settings.env.engineHost')} description={t('settings.env.engineHostDesc')}
                value={envValues.ACESTEPCPP_HOST || ''} onChange={handleEnvChange} placeholder="127.0.0.1" />
              <EnvTextRow envKey="ACESTEPCPP_VAE_CHUNK" label={t('settings.env.vaeChunk')} description={t('settings.env.vaeChunkDesc')}
                value={envValues.ACESTEPCPP_VAE_CHUNK || ''} onChange={handleEnvChange} type="number" placeholder="1024" />
              <EnvTextRow envKey="ACESTEPCPP_VAE_OVERLAP" label={t('settings.env.vaeOverlap')} description={t('settings.env.vaeOverlapDesc')}
                value={envValues.ACESTEPCPP_VAE_OVERLAP || ''} onChange={handleEnvChange} type="number" placeholder="64" />

              {/* GPU Device selector — auto-detects available GPUs */}
              <div className="setting-row">
                <div className="setting-info">
                  <ParamLabel
                    label="GPU Device"
                    className="setting-label"
                    info="Which GPU the engine should use. Auto picks the card with the most VRAM. Pick a specific card to pin the engine to it instead, useful with more than one GPU in the machine. Requires a restart to take effect."
                  />
                </div>
                {detectedGpus.length > 0 ? (
                  <StyledSelect
                    id="env-CUDA_VISIBLE_DEVICES"
                    accent="emerald"
                    value={gpuSelectValue}
                    onChange={(v) => handleEnvChange('CUDA_VISIBLE_DEVICES', v)}
                    className="min-w-[180px] max-w-[360px]"
                    options={[
                      { value: '', label: 'Auto (most VRAM)' },
                      ...detectedGpus.map((gpu) => ({
                        value: gpu.uuid || String(gpu.index),
                        label: `GPU ${gpu.index}: ${gpu.name} (${Math.round(gpu.memoryMB / 1024)} GB)`,
                      })),
                      ...(gpuSelectUnknown ? [{ value: gpuSelectValue, label: `${gpuSelectValue} (from .env)` }] : []),
                    ]}
                  />
                ) : (
                  <input
                    id="env-CUDA_VISIBLE_DEVICES"
                    className="env-input"
                    type="text"
                    value={envValues.CUDA_VISIBLE_DEVICES || ''}
                    onChange={(e) => handleEnvChange('CUDA_VISIBLE_DEVICES', e.target.value)}
                    placeholder="(auto)"
                    style={{ width: '80px', textAlign: 'center' }}
                  />
                )}
              </div>

              {/* Multi-GPU (experimental) — one engine per ticked card. Only shown with 2+ detected
                  cards. With two or more ticked cards, jobs run in parallel, one per card. */}
              {detectedGpus.length > 1 && (
                <div className="setting-row">
                  <div className="setting-info">
                    <ParamLabel
                      label="Multi-GPU workers"
                      className="setting-label"
                      meta="experimental · off by default"
                      info="Starts one engine per ticked card instead of one engine in total. Each model must fit in a single card's VRAM; a model is never split across cards. CUDA-only: Vulkan, CPU and Apple Silicon keep the single engine. With two or more cards ticked, jobs run in parallel, one per card; training, post-processing and previews still take every card. Leave all unticked for the normal single engine, which still uses the GPU Device pick above. Requires a restart to take effect."
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    {detectedGpus.map((gpu) => (
                      <label key={gpu.uuid || gpu.index} className="flex items-center gap-2">
                        <Toggle
                          id={`env-worker-gpu-${gpu.index}`}
                          disabled={!gpu.uuid}
                          checked={!!gpu.uuid && workerGpuSet.has(gpu.uuid.toLowerCase())}
                          onChange={() => toggleWorkerGpu(gpu.uuid)}
                        />
                        <span>{`GPU ${gpu.index}: ${gpu.name} (${Math.round(gpu.memoryMB / 1024)} GB)`}</span>
                      </label>
                    ))}
                  </div>
                </div>
              )}

              {/* Per-worker engine status: shown once a second worker exists (after a restart with the opt-in set) */}
              {engineWorkers && engineWorkers.workers.length > 1 && (
                <div className="setting-row">
                  <div className="setting-info">
                    <ParamLabel
                      label="Engine workers"
                      className="setting-label"
                      info="One row per running engine. w0 is the primary. “busy” means a job is running on that worker; a worker is only restarted while idle."
                    />
                  </div>
                  <div className="flex flex-col gap-1 text-sm">
                    {engineWorkers.workers.map((w) => {
                      const card = detectedGpus.find(g => g.uuid && g.uuid === w.gpuUuid);
                      return (
                        <div key={w.id}>
                          {`${w.id}${w.primary ? ' (primary)' : ''} · ${card ? `GPU ${card.index}` : 'auto'} · port ${w.port} · ${w.ready ? 'ready' : w.status}${w.busy > 0 ? ' · busy' : ''}`}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Keep models in VRAM — passes --keep-loaded to the engine.
                  Restart-required (spawn-time flag). Default off (VRAM cost). */}
              <div className="setting-row">
                <div className="setting-info">
                  <ParamLabel
                    label="Keep Models in VRAM"
                    className="setting-label"
                    meta="default off · holds ~13 GB VRAM"
                    info="Keeps the DiT, adapter, and VAE resident in VRAM between generations, instead of loading them fresh each time. On: skips the one-time ~17s adapter precompute on every render, so repeat generations are much faster, at the cost of ~13 GB of VRAM held continuously. Off: frees that VRAM between renders, better on VRAM-tight GPUs. Requires a restart to take effect."
                  />
                </div>
                <Toggle
                  id="env-ACESTEPCPP_KEEP_LOADED"
                  checked={(envValues.ACESTEPCPP_KEEP_LOADED || '0') === '1'}
                  onChange={(v) => handleEnvChange('ACESTEPCPP_KEEP_LOADED', v ? '1' : '0')}
                />
              </div>

            </EnvSubsection>

            {/* Server */}
            <EnvSubsection title={t('settings.env.server')} isOpen={openSections.server} onToggle={() => toggleSection('server')}>
              <EnvTextRow envKey="SERVER_PORT" label={t('settings.env.serverPort')} description={t('settings.env.serverPortDesc')}
                value={envValues.SERVER_PORT || ''} onChange={handleEnvChange} type="number" placeholder="3001" />
              <EnvPathRow envKey="DATA_DIR" label={t('settings.env.dataDir')} description={t('settings.env.dataDirDesc')}
                value={envValues.DATA_DIR || ''} onChange={handleEnvChange} onBrowse={handleBrowse} placeholder="./data" />
              <EnvTextRow envKey="TRAINING_WORKERS" label={t('settings.env.trainingWorkers', 'Training workers')}
                description={t('settings.env.trainingWorkersDesc', 'Other PCs running HOT-Step that this PC can train on, as Name=http://address:3001, comma-separated. They appear under "Train on" in the Training Studio.')}
                value={envValues.TRAINING_WORKERS || ''} onChange={handleEnvChange} placeholder="LivingRoom=http://192.168.1.50:3001" />
              <EnvPasswordRow envKey="TRAINING_WORKER_TOKEN" label={t('settings.env.trainingWorkerToken', 'Training worker token')}
                description={t('settings.env.trainingWorkerTokenDesc', 'Sent to the training workers above. Must match the Worker token set on each worker.')}
                value={envValues.TRAINING_WORKER_TOKEN || ''} onChange={handleEnvChange} />
              <EnvPasswordRow envKey="WORKER_TOKEN" label={t('settings.env.workerToken', 'Worker token (this PC as a worker)')}
                description={t('settings.env.workerTokenDesc', 'Set only on a PC that trains for another. When set, API calls from other machines must carry this token.')}
                value={envValues.WORKER_TOKEN || ''} onChange={handleEnvChange} />
            </EnvSubsection>

            {/* Dataset labeling throughput */}
            <EnvSubsection title={t('settings.env.labeling')} isOpen={openSections.labeling} onToggle={() => toggleSection('labeling')}>
              <EnvTextRow envKey="LABEL_ESSENTIA_CONCURRENCY" label={t('settings.env.labelEssentiaConcurrency')} description={t('settings.env.labelEssentiaConcurrencyDesc')}
                value={envValues.LABEL_ESSENTIA_CONCURRENCY || ''} onChange={handleEnvChange} type="number" placeholder="2" />
              <EnvTextRow envKey="LABEL_GENIUS_CONCURRENCY" label={t('settings.env.labelGeniusConcurrency')} description={t('settings.env.labelGeniusConcurrencyDesc')}
                value={envValues.LABEL_GENIUS_CONCURRENCY || ''} onChange={handleEnvChange} type="number" placeholder="1" />
              <EnvTextRow envKey="LABEL_GENIUS_MIN_INTERVAL_MS" label={t('settings.env.labelGeniusInterval')} description={t('settings.env.labelGeniusIntervalDesc')}
                value={envValues.LABEL_GENIUS_MIN_INTERVAL_MS || ''} onChange={handleEnvChange} type="number" placeholder="400" />
              <EnvTextRow envKey="LABEL_CAPTION_CONCURRENCY" label={t('settings.env.labelCaptionConcurrency')} description={t('settings.env.labelCaptionConcurrencyDesc')}
                value={envValues.LABEL_CAPTION_CONCURRENCY || ''} onChange={handleEnvChange} type="number" placeholder="1" />
              <EnvTextRow envKey="LABEL_CAPTION_MIN_INTERVAL_MS" label={t('settings.env.labelCaptionInterval')} description={t('settings.env.labelCaptionIntervalDesc')}
                value={envValues.LABEL_CAPTION_MIN_INTERVAL_MS || ''} onChange={handleEnvChange} type="number" placeholder="250" />
            </EnvSubsection>

            {/* Paths */}
            <EnvSubsection title={t('settings.env.paths')} isOpen={openSections.paths} onToggle={() => toggleSection('paths')}>
              <EnvPathRow envKey="LYRICS_EXPORT_DIR" label={t('settings.env.lyricsExportDir')} description={t('settings.env.lyricsExportDirDesc')}
                value={envValues.LYRICS_EXPORT_DIR || ''} onChange={handleEnvChange} onBrowse={handleBrowse} />
              <EnvPathRow envKey="DOWNLOAD_DIR" label={t('settings.env.downloadDir')} description={t('settings.env.downloadDirDesc')}
                value={envValues.DOWNLOAD_DIR || ''} onChange={handleEnvChange} onBrowse={handleBrowse} />
              <EnvPathRow envKey="MUSCRIPTOR_MODELS_DIR" label={t('settings.env.muscriptorModelsDir')} description={t('settings.env.muscriptorModelsDirDesc')}
                value={envValues.MUSCRIPTOR_MODELS_DIR || ''} onChange={handleEnvChange} onBrowse={handleBrowse} />
            </EnvSubsection>

            {/* Save bar */}
            <div className="env-save-bar">
              {envStatus && (
                <span className={`env-save-status env-save-status--${envStatus.type}`}>
                  {envStatus.text}
                </span>
              )}
            {pendingRestartKeys.length > 0 && (
              <button
                className="env-save-btn"
                onClick={handleRestartNow}
                disabled={restarting}
                title="Restart so the saved values take effect"
              >
                {restarting ? 'Restarting…' : 'Restart now'}
              </button>
            )}
              <button
                className="env-save-btn"
                onClick={handleEnvSave}
                disabled={!envDirty || envSaving}
              >
                {envSaving ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Loader2 size={14} className="animate-spin" /> {t('settings.env.saving')}
                  </span>
                ) : (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Save size={14} /> {t('settings.env.saveEnvironment')}
                  </span>
                )}
              </button>
            </div>
          </>
        )}
      </div>
      )}

      {/* File Browser Modal for path settings */}
      <FileBrowserModal
        open={browseKey !== null}
        onClose={() => setBrowseKey(null)}
        onSelect={handleBrowseSelect}
        mode="folder"
        startPath={browseKey ? envValues[browseKey] || '' : ''}
        title={browseKey ? t('settings.env.selectFolderFor', { key: browseKey }) : t('settings.env.selectFolder')}
      />

      {/* ═══════════════ AI SERVICES TAB ═══════════════ */}
      {activeTab === 'ai' && (
      <>
      {/* API Keys Section */}
      <div className="settings-section">
        <div className="settings-section-header">
          <Key size={16} className="settings-section-icon" />
          <span className="settings-section-title">{t('settings.ai.apiKeys')}</span>
        </div>
        <EnvPasswordRow envKey="GENIUS_ACCESS_TOKEN" label={t('settings.ai.geniusToken')} description={t('settings.ai.geniusDesc')}
          value={envValues.GENIUS_ACCESS_TOKEN || ''} onChange={handleEnvChange} />
        <EnvPasswordRow envKey="GEMINI_API_KEY" label={t('settings.ai.geminiKey')} description={t('settings.ai.geminiDesc')}
          value={envValues.GEMINI_API_KEY || ''} onChange={handleEnvChange} />
        <EnvPasswordRow envKey="OPENAI_API_KEY" label={t('settings.ai.openaiKey')} description={t('settings.ai.openaiDesc')}
          value={envValues.OPENAI_API_KEY || ''} onChange={handleEnvChange} />
        <EnvPasswordRow envKey="ANTHROPIC_API_KEY" label={t('settings.ai.anthropicKey')} description={t('settings.ai.anthropicDesc')}
          value={envValues.ANTHROPIC_API_KEY || ''} onChange={handleEnvChange} />
      </div>

      {/* LLM Configuration Section */}
      <div className="settings-section">
        <div className="settings-section-header">
          <Settings2 size={16} className="settings-section-icon" />
          <span className="settings-section-title">{t('settings.ai.llmConfig')}</span>
        </div>
        <div className="setting-row">
          <div className="setting-info">
            <ParamLabel
              label={t('settings.ai.defaultProvider')}
              className="setting-label"
              info={t('settings.ai.defaultProviderDesc')}
            />
          </div>
          <StyledSelect
            id="env-DEFAULT_LLM_PROVIDER"
            accent="emerald"
            className="min-w-[180px] max-w-[360px]"
            value={envValues.DEFAULT_LLM_PROVIDER || 'gemini'}
            onChange={(v) => handleEnvChange('DEFAULT_LLM_PROVIDER', v)}
            options={[
              { value: 'gemini', label: 'Gemini' },
              { value: 'openai', label: 'OpenAI' },
              { value: 'anthropic', label: 'Anthropic' },
              { value: 'ollama', label: 'Ollama' },
              { value: 'lmstudio', label: 'LM Studio' },
              { value: 'llamacpp', label: 'llama.cpp' },
              { value: 'unsloth', label: 'Unsloth' },
              { value: 'openai-compat', label: envValues.OPENAI_COMPAT_NAME || 'OpenAI Compatible' },
            ]}
          />
        </div>
        <EnvTextRow envKey="LLM_TIMEOUT_MS" label={t('settings.ai.llmTimeout')} description={t('settings.ai.llmTimeoutDesc')}
          value={envValues.LLM_TIMEOUT_MS || ''} onChange={handleEnvChange} type="number" placeholder="300000" />
        <EnvTextRow envKey="GEMINI_MODEL" label={t('settings.ai.geminiModel')} description={t('settings.ai.geminiModelDesc')}
          value={envValues.GEMINI_MODEL || ''} onChange={handleEnvChange} placeholder="gemini-2.5-flash" />
        <EnvTextRow envKey="OPENAI_MODEL" label={t('settings.ai.openaiModel')} description={t('settings.ai.openaiModelDesc')}
          value={envValues.OPENAI_MODEL || ''} onChange={handleEnvChange} placeholder="gpt-4o-mini" />
        <EnvTextRow envKey="ANTHROPIC_MODEL" label={t('settings.ai.anthropicModel')} description={t('settings.ai.anthropicModelDesc')}
          value={envValues.ANTHROPIC_MODEL || ''} onChange={handleEnvChange} placeholder="claude-haiku-4-5-20251001" />
        <EnvTextRow envKey="OLLAMA_MODEL" label={t('settings.ai.ollamaModel')} description={t('settings.ai.ollamaModelDesc')}
          value={envValues.OLLAMA_MODEL || ''} onChange={handleEnvChange} placeholder="llama3" />
        <EnvTextRow envKey="LMSTUDIO_MODEL" label={t('settings.ai.lmstudioModel')} description={t('settings.ai.lmstudioModelDesc')}
          value={envValues.LMSTUDIO_MODEL || ''} onChange={handleEnvChange} />
        <EnvTextRow envKey="UNSLOTH_MODEL" label={t('settings.ai.unslothModel')} description={t('settings.ai.unslothModelDesc')}
          value={envValues.UNSLOTH_MODEL || ''} onChange={handleEnvChange} />
        <EnvTextRow envKey="LLAMACPP_MODEL" label={t('settings.ai.llamacppModel')} description={t('settings.ai.llamacppModelDesc')}
          value={envValues.LLAMACPP_MODEL || ''} onChange={handleEnvChange} />
        <EnvTextRow envKey="OPENAI_COMPAT_MODEL" label={t('settings.ai.openaiCompatModel')} description={t('settings.ai.openaiCompatModelDesc')}
          value={envValues.OPENAI_COMPAT_MODEL || ''} onChange={handleEnvChange} placeholder="auto-discovered" />
      </div>

      {/* LLM Endpoints Section */}
      <div className="settings-section">
        <div className="settings-section-header">
          <ChevronRight size={16} className="settings-section-icon" />
          <span className="settings-section-title">{t('settings.ai.llmEndpoints')}</span>
        </div>
        <EnvTextRow envKey="OLLAMA_BASE_URL" label={t('settings.ai.ollamaUrl')} description={t('settings.ai.ollamaUrlDesc')}
          value={envValues.OLLAMA_BASE_URL || ''} onChange={handleEnvChange} placeholder="http://localhost:11434" />
        <EnvTextRow envKey="LMSTUDIO_BASE_URL" label={t('settings.ai.lmstudioUrl')} description={t('settings.ai.lmstudioUrlDesc')}
          value={envValues.LMSTUDIO_BASE_URL || ''} onChange={handleEnvChange} placeholder="http://localhost:1234/v1" />
        <EnvPasswordRow envKey="LMSTUDIO_API_KEY" label={t('settings.ai.lmstudioApiKey')} description={t('settings.ai.lmstudioApiKeyDesc')}
          value={envValues.LMSTUDIO_API_KEY || ''} onChange={handleEnvChange} />
        <EnvTextRow envKey="UNSLOTH_BASE_URL" label={t('settings.ai.unslothUrl')} description={t('settings.ai.unslothUrlDesc')}
          value={envValues.UNSLOTH_BASE_URL || ''} onChange={handleEnvChange} placeholder="http://127.0.0.1:8888" />
        <EnvTextRow envKey="UNSLOTH_USERNAME" label={t('settings.ai.unslothUsername')} description={t('settings.ai.unslothUsernameDesc')}
          value={envValues.UNSLOTH_USERNAME || ''} onChange={handleEnvChange} />
        <EnvPasswordRow envKey="UNSLOTH_PASSWORD" label={t('settings.ai.unslothPassword')} description={t('settings.ai.unslothPasswordDesc')}
          value={envValues.UNSLOTH_PASSWORD || ''} onChange={handleEnvChange} />
        <div className="setting-row">
          <div className="setting-info">
            <ParamLabel
              label={t('settings.ai.unslothKeyless')}
              className="setting-label"
              info={t('settings.ai.unslothKeylessDesc')}
            />
          </div>
          <Toggle
            id="env-UNSLOTH_KEYLESS"
            checked={(envValues.UNSLOTH_KEYLESS || '0') === '1'}
            onChange={(v) => handleEnvChange('UNSLOTH_KEYLESS', v ? '1' : '0')}
          />
        </div>
        <EnvTextRow envKey="LLAMACPP_BASE_URL" label={t('settings.ai.llamacppUrl')} description={t('settings.ai.llamacppUrlDesc')}
          value={envValues.LLAMACPP_BASE_URL || ''} onChange={handleEnvChange} placeholder="http://127.0.0.1:8080/v1" />
        <EnvTextRow envKey="OPENAI_COMPAT_BASE_URL" label={t('settings.ai.openaiCompatUrl')} description={t('settings.ai.openaiCompatUrlDesc')}
          value={envValues.OPENAI_COMPAT_BASE_URL || ''} onChange={handleEnvChange} placeholder="http://mac-server:8000/v1" />
        <EnvPasswordRow envKey="OPENAI_COMPAT_API_KEY" label={t('settings.ai.openaiCompatApiKey')} description={t('settings.ai.openaiCompatApiKeyDesc')}
          value={envValues.OPENAI_COMPAT_API_KEY || ''} onChange={handleEnvChange} />
        <EnvTextRow envKey="OPENAI_COMPAT_NAME" label={t('settings.ai.openaiCompatName')} description={t('settings.ai.openaiCompatNameDesc')}
          value={envValues.OPENAI_COMPAT_NAME || ''} onChange={handleEnvChange} placeholder="OpenAI Compatible" />
      </div>

      {/* Save bar for AI tab */}
      <div className="env-save-bar">
        {envStatus && (
          <span className={`env-save-status env-save-status--${envStatus.type}`}>{envStatus.text}</span>
        )}
        {pendingRestartKeys.length > 0 && (
          <button
            className="env-save-btn"
            onClick={handleRestartNow}
            disabled={restarting}
            title="Restart so the saved values take effect"
          >
            {restarting ? 'Restarting…' : 'Restart now'}
          </button>
        )}
        <button className="env-save-btn" onClick={handleEnvSave} disabled={!envDirty || envSaving}>
          {envSaving ? (
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Loader2 size={14} className="animate-spin" /> {t('settings.env.saving')}
            </span>
          ) : (
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Save size={14} /> {t('settings.env.saveChanges')}
            </span>
          )}
        </button>
      </div>
      </>
      )}

      {/* ═══════════════ PERFORMANCE TAB ═══════════════ */}
      {activeTab === 'performance' && (
      <>
      {/* Pipeline Parallelism Section */}
      <div className="settings-section">
        <div className="settings-section-header">
          <Gauge size={16} className="settings-section-icon" />
          <span className="settings-section-title">Pipeline Parallelism</span>
        </div>

        <div className="setting-row">
          <div className="setting-info">
            <div className="setting-description" style={{ marginTop: 0 }}>
              Run independent pipeline stages concurrently instead of sequentially.
              Enable these to reduce total generation time. Disable if you experience
              out-of-memory errors or instability.
            </div>
          </div>
        </div>

        <SettingRow
          id="setting-parallel-whisper"
          label="Parallel Whisper Transcription"
          description="Run lyrics transcription concurrently with post-processing. Whisper is CPU-only — no VRAM impact."
          checked={settings.parallelWhisper}
          onChange={(v) => update('parallelWhisper', v)}
          defaultValue={DEFAULT_SETTINGS.parallelWhisper}
          badges={[
            { text: '~6s saved', type: 'speed' as const },
          ]}
        />

        <SettingRow
          id="setting-parallel-qe"
          label="Parallel Quality Evaluation"
          description="Run audio quality analysis concurrently with other stages. CPU-only — no VRAM impact."
          checked={settings.parallelQualityEval}
          onChange={(v) => update('parallelQualityEval', v)}
          defaultValue={DEFAULT_SETTINGS.parallelQualityEval}
          badges={[
            { text: '~2s saved', type: 'speed' as const },
          ]}
        />

        <SettingRow
          id="setting-parallel-cover-art"
          label="Parallel Cover Art"
          description="Start cover art generation during post-processing instead of waiting until after. Uses GPU (Flux) — may need VRAM headroom."
          checked={settings.parallelCoverArt}
          onChange={(v) => update('parallelCoverArt', v)}
          defaultValue={DEFAULT_SETTINGS.parallelCoverArt}
          badges={[
            { text: '~5s saved', type: 'speed' as const },
            { text: 'GPU', type: 'vram' as const },
          ]}
        />
      </div>

      {/* Existing Performance Settings */}
      <div className="settings-section">
        <div className="settings-section-header">
          <Zap size={16} className="settings-section-icon" />
          <span className="settings-section-title">Engine</span>
        </div>

        <SettingRow
          id="setting-co-resident-perf"
          label={t('settings.general.coResident')}
          description={t('settings.general.coResidentDesc')}
          checked={settings.coResident}
          onChange={(v) => update('coResident', v)}
          defaultValue={DEFAULT_SETTINGS.coResident}
          badges={[
            { text: '−13s', type: 'speed' as const },
            { text: '+8GB VRAM', type: 'vram' as const },
          ]}
        />

        <SettingRow
          id="setting-cache-lm-perf"
          label={t('settings.general.cacheLm')}
          description={t('settings.general.cacheLmDesc')}
          checked={settings.cacheLmCodes}
          onChange={(v) => update('cacheLmCodes', v)}
          defaultValue={DEFAULT_SETTINGS.cacheLmCodes}
          badges={[
            { text: '−12s', type: 'speed' as const },
          ]}
        />

        <SelectRow
          id="setting-gen-timeout"
          label="Generation Timeout"
          description="Maximum time to wait for a single generation before it's considered timed out. Increase if you use high step counts or have slower hardware. This is per track, not per batch — the clock starts when the engine picks a track up, so a 9-track batch gets this budget nine times over."
          value={settings.generationTimeoutMinutes}
          options={[
            { value: 10, label: '10 minutes' },
            { value: 15, label: '15 minutes' },
            { value: 20, label: '20 minutes' },
            { value: 30, label: '30 minutes (default)' },
            { value: 45, label: '45 minutes' },
            { value: 60, label: '60 minutes' },
            { value: 90, label: '90 minutes' },
            { value: 120, label: '120 minutes' },
            { value: 180, label: '3 hours' },
            { value: 240, label: '4 hours' },
            { value: 360, label: '6 hours' },
          ]}
          onChange={(v) => update('generationTimeoutMinutes', parseInt(v))}
          defaultValue={DEFAULT_SETTINGS.generationTimeoutMinutes}
        />
      </div>
      </>
      )}

      {/* ═══════════════ GENERAL TAB ═══════════════ */}
      {activeTab === 'general' && (
      <>
      <UpdatesPanel />
      {/* Language Section */}
      <div className="settings-section">
        <div className="settings-section-header">
          <Globe size={16} className="settings-section-icon" />
          <span className="settings-section-title">{t('settings.general.language')}</span>
        </div>

        <SelectRow
          id="setting-language"
          label={t('settings.general.displayLanguage')}
          description={t('settings.general.displayLanguageDesc')}
          value={i18n.language?.split('-')[0] || FALLBACK_LANGUAGE}
          options={SUPPORTED_LANGUAGES.map(l => ({ value: l.code, label: `${l.flag} ${l.name}` }))}
          onChange={(v) => i18n.changeLanguage(v)}
          defaultValue={FALLBACK_LANGUAGE}
        />
      </div>



      {/* Generic cover art */}
      <div className="settings-section">
        <div className="settings-section-header">
          <ImageIcon size={16} className="settings-section-icon" />
          <span className="settings-section-title">Cover art</span>
        </div>

        <SelectRow
          id="setting-cover-set"
          label="Generic artwork"
          description="Which set of stand-in covers a track shows until it has one of its own. Each track keeps the same image within a set."
          value={coverSet}
          options={COVER_SETS.map(s => ({ value: s.id, label: s.label }))}
          onChange={(v) => { setCoverSetState(v as CoverSet); setCoverSet(v as CoverSet); }}
          defaultValue={FALLBACK_SET}
        />

        <div className="flex gap-1.5 mt-3">
          {[0, 7, 13, 21, 34, 42].map(i => (
            <img key={i} src={`/covers/${coverSet}/${String(i).padStart(2, '0')}.webp`}
              alt="" loading="lazy"
              className="w-12 h-12 rounded-md object-cover border border-zinc-200 dark:border-white/10" />
          ))}
        </div>
      </div>

      {/* Disco Mode */}
      <div className="border-t border-zinc-200 dark:border-white/5 pt-4 mt-4">
        <h4 className="text-sm font-medium text-zinc-900 dark:text-white mb-3 flex items-center gap-2">
          🪩 Disco Mode
        </h4>
        <SettingRow
          id="setting-disco-kick-extract"
          label="Extract drum stems for visualisations"
          description="After generation, extract kick, snare, and hi-hat stems for beat-reactive visual effects (rainbow borders, snare flash, hi-hat particles). Takes ~60-90s extra per song."
          checked={settings.discoKickExtract}
          onChange={v => update('discoKickExtract', v)}
          defaultValue={DEFAULT_SETTINGS.discoKickExtract}
        />
      </div>

      {/* Downloads Section */}
      <div className="settings-section">
        <div className="settings-section-header">
          <Download size={16} className="settings-section-icon" />
          <span className="settings-section-title">{t('settings.general.downloads')}</span>
        </div>

        <SelectRow
          id="setting-dl-format"
          label={t('settings.general.defaultFormat')}
          description={t('settings.general.defaultFormatDesc')}
          value={settings.downloadFormat}
          options={[
            { value: 'wav', label: 'WAV (lossless)' },
            { value: 'flac', label: 'FLAC (lossless)' },
            { value: 'opus', label: 'Opus (lossy)' },
            { value: 'mp3', label: 'MP3 (lossy)' },
          ]}
          onChange={(v) => update('downloadFormat', v as AppSettings['downloadFormat'])}
          defaultValue={DEFAULT_SETTINGS.downloadFormat}
        />

        <SelectRow
          id="setting-dl-mp3-bitrate"
          label={t('settings.general.mp3Bitrate')}
          description={t('settings.general.mp3BitrateDesc')}
          value={settings.downloadMp3Bitrate}
          options={[
            { value: 128, label: '128 kbps' },
            { value: 192, label: '192 kbps' },
            { value: 256, label: '256 kbps' },
            { value: 320, label: '320 kbps' },
          ]}
          onChange={(v) => update('downloadMp3Bitrate', parseInt(v))}
          defaultValue={DEFAULT_SETTINGS.downloadMp3Bitrate}
        />

        <SelectRow
          id="setting-dl-opus-bitrate"
          label={t('settings.general.opusBitrate')}
          description={t('settings.general.opusBitrateDesc')}
          value={settings.downloadOpusBitrate}
          options={[
            { value: 96, label: '96 kbps' },
            { value: 128, label: '128 kbps' },
            { value: 192, label: '192 kbps' },
            { value: 256, label: '256 kbps' },
          ]}
          onChange={(v) => update('downloadOpusBitrate', parseInt(v))}
          defaultValue={DEFAULT_SETTINGS.downloadOpusBitrate}
        />

        <SelectRow
          id="setting-dl-version"
          label={t('settings.general.downloadVersion')}
          description={t('settings.general.downloadVersionDesc')}
          value={settings.downloadVersion}
          options={[
            { value: 'mastered', label: t('settings.general.versionMastered') },
            { value: 'original', label: t('settings.general.versionOriginal') },
            { value: 'both', label: t('settings.general.versionBoth') },
          ]}
          onChange={(v) => update('downloadVersion', v as AppSettings['downloadVersion'])}
          defaultValue={DEFAULT_SETTINGS.downloadVersion}
        />

        <SettingRow
          id="setting-dl-latent"
          label={t('settings.general.includeLatent')}
          description={t('settings.general.includeLatentDesc')}
          checked={settings.downloadIncludeLatent}
          onChange={(v) => update('downloadIncludeLatent', v)}
          defaultValue={DEFAULT_SETTINGS.downloadIncludeLatent}
        />
      </div>

      {/* Adapters Section (still in General tab) */}
      <div className="settings-section">
        <div className="settings-section-header">
          <Tag size={16} className="settings-section-icon" />
          <span className="settings-section-title">{t('settings.general.adapters')}</span>
        </div>

        <AdapterReferencesAction />

        <SettingRow
          id="setting-trigger-filename"
          label={t('settings.general.triggerFilename')}
          description={t('settings.general.triggerFilenameDesc')}
          checked={settings.triggerUseFilename}
          onChange={(v) => update('triggerUseFilename', v)}
          defaultValue={DEFAULT_SETTINGS.triggerUseFilename}
        />

        {settings.triggerUseFilename && (
          <div className="setting-row">
            <div className="setting-info">
              <ParamLabel
                label={t('settings.general.triggerPlacement')}
                className="setting-label"
                onReset={settings.triggerPlacement !== DEFAULT_SETTINGS.triggerPlacement ? () => update('triggerPlacement', DEFAULT_SETTINGS.triggerPlacement) : undefined} />
              <div className="setting-description">
                {settings.triggerPlacement === 'prepend' && t('settings.general.triggerPrepend')}
                {settings.triggerPlacement === 'append' && t('settings.general.triggerAppend')}
                {settings.triggerPlacement === 'replace' && t('settings.general.triggerReplace')}
              </div>
            </div>
            <div className="placement-button-group">
              {(['prepend', 'append', 'replace'] as const).map((p) => (
                <button
                  key={p}
                  className={`placement-button ${settings.triggerPlacement === p ? 'placement-button--active' : ''}`}
                  onClick={() => update('triggerPlacement', p)}
                >
                  {p.charAt(0).toUpperCase() + p.slice(1)}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
      </>
      )}

      {activeTab === 'report' && <ReportIssuePanel />}

      {/* ═══════════════ STORAGE & DATA TAB ═══════════════ */}
      {activeTab === 'storage' && (
      <>
      <DataFolderPanel />
      <SettingsBackupPanel settings={settings} onSettingsChange={onSettingsChange} />
      <StemStorage />

      <DangerZone onNukeComplete={onNukeComplete} />
      </>
      )}
    </div>
  );
};
