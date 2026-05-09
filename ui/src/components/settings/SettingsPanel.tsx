// SettingsPanel.tsx — Application settings with persistent state
//
// All settings use localStorage via usePersistedState and survive page refreshes.
// Environment settings read from / write to the server's .env file.

import React, { useState, useEffect, useCallback } from 'react';
import {
  Zap, Download, Tag, AlertTriangle, Loader2, Settings2,
  ChevronRight, Save, Scissors,
  Key, Database, Globe
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { SUPPORTED_LANGUAGES } from '../../i18n';
import { getStemStats, deleteAllJobs, formatBytes, type StemStats } from '../../services/stemStudioApi';
import { useAuth } from '../../context/AuthContext';
import { songApi, settingsApi } from '../../services/api';
import { lireekApi } from '../../services/lireekApi';
import { FileBrowserModal } from '../shared/FileBrowserModal';
import {
  SettingRow, SelectRow,
  EnvTextRow, EnvPasswordRow, EnvPathRow, EnvSubsection,
} from './SettingsPrimitives';
import './SettingsPanel.css';

export interface AppSettings {
  coResident: boolean;
  cacheLmCodes: boolean;
  // Download defaults
  downloadFormat: 'wav' | 'flac' | 'opus' | 'mp3';
  downloadMp3Bitrate: number;
  downloadOpusBitrate: number;
  // Adapter trigger word
  triggerUseFilename: boolean;
  triggerPlacement: 'prepend' | 'append' | 'replace';
}

export const DEFAULT_SETTINGS: AppSettings = {
  coResident: false,
  cacheLmCodes: true,
  downloadFormat: 'flac',
  downloadMp3Bitrate: 192,
  downloadOpusBitrate: 192,
  triggerUseFilename: false,
  triggerPlacement: 'prepend',
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
  const { token } = useAuth();
  const [nukeConfirm, setNukeConfirm] = useState(false);
  const [nukeRunning, setNukeRunning] = useState(false);
  const [nukeResult, setNukeResult] = useState<string | null>(null);
  const [nukeLyricsConfirm, setNukeLyricsConfirm] = useState(false);
  const [nukeLyricsRunning, setNukeLyricsRunning] = useState(false);
  const [nukeProfilesConfirm, setNukeProfilesConfirm] = useState(false);
  const [nukeProfilesRunning, setNukeProfilesRunning] = useState(false);

  // Stem storage state
  const [stemStats, setStemStats] = useState<StemStats | null>(null);
  const [stemClearConfirm, setStemClearConfirm] = useState(false);
  const [stemClearing, setStemClearing] = useState(false);

  const update = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  const handleNuke = async () => {
    if (!token) return;
    setNukeRunning(true);
    setNukeResult(null);
    try {
      const res = await songApi.nukeGenerations(token);
      setNukeResult(
        `Nuked ${res.songsDeleted} songs, ${res.filesDeleted} audio files, ${res.lireekAudioGensDeleted} Lyric Studio entries.`
      );
      setNukeConfirm(false);
      onNukeComplete?.();
    } catch (err: any) {
      setNukeResult(`Failed: ${err.message}`);
    } finally {
      setNukeRunning(false);
    }
  };

  const handleNukeLyrics = async () => {
    setNukeLyricsRunning(true);
    setNukeResult(null);
    try {
      const res = await lireekApi.purgeGenerations();
      setNukeResult(`Deleted ${res.generations_deleted} generated lyrics.`);
      setNukeLyricsConfirm(false);
    } catch (err: any) {
      setNukeResult(`Failed: ${err.message}`);
    } finally {
      setNukeLyricsRunning(false);
    }
  };

  const handleNukeProfiles = async () => {
    setNukeProfilesRunning(true);
    setNukeResult(null);
    try {
      const res = await lireekApi.purgeProfiles();
      setNukeResult(`Deleted ${res.profiles_deleted} profiles and ${res.generations_deleted} dependent lyrics.`);
      setNukeProfilesConfirm(false);
    } catch (err: any) {
      setNukeResult(`Failed: ${err.message}`);
    } finally {
      setNukeProfilesRunning(false);
    }
  };

  // Load stem storage stats
  useEffect(() => {
    getStemStats().then(setStemStats).catch(() => setStemStats(null));
  }, []);

  const handleClearStems = async () => {
    setStemClearing(true);
    try {
      await deleteAllJobs();
      setStemStats({ totalBytes: 0, jobCount: 0, stemCount: 0 });
      setStemClearConfirm(false);
    } catch (err: any) {
      console.error('[Settings] Clear stems failed:', err);
    } finally {
      setStemClearing(false);
    }
  };

  // ── Environment (.env) state ────────────────────────────────────
  const [envValues, setEnvValues] = useState<Record<string, string>>({});
  const [envOriginal, setEnvOriginal] = useState<Record<string, string>>({});
  const [envLoading, setEnvLoading] = useState(true);
  const [envSaving, setEnvSaving] = useState(false);
  const [envStatus, setEnvStatus] = useState<{ type: 'success' | 'error' | 'warning'; text: string } | null>(null);

  // Subsection open/close state
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    engine: true, server: true, apiKeys: true, llmConfig: true, llmEndpoints: false, paths: true,
  });
  const toggleSection = (key: string) =>
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));

  // File browser modal state
  const [browseKey, setBrowseKey] = useState<string | null>(null);

  // Load env settings on mount
  const loadEnvSettings = useCallback(async () => {
    try {
      setEnvLoading(true);
      const data = await settingsApi.getEnv();
      setEnvValues(data.values);
      setEnvOriginal(data.values);
    } catch (err: any) {
      console.error('[Settings] Failed to load .env:', err.message);
    } finally {
      setEnvLoading(false);
    }
  }, []);

  useEffect(() => { loadEnvSettings(); }, [loadEnvSettings]);

  // Track which keys have changed
  const envDirty = Object.keys(envValues).some((k) => envValues[k] !== envOriginal[k]);

  const handleEnvChange = (key: string, value: string) => {
    setEnvValues((prev) => ({ ...prev, [key]: value }));
    setEnvStatus(null);
  };

  const handleEnvSave = async () => {
    // Build diff — only send changed keys
    const diff: Record<string, string> = {};
    for (const [key, value] of Object.entries(envValues)) {
      if (value !== envOriginal[key]) {
        diff[key] = value;
      }
    }
    if (Object.keys(diff).length === 0) return;

    setEnvSaving(true);
    setEnvStatus(null);
    try {
      const res = await settingsApi.updateEnv(diff);
      setEnvOriginal({ ...envValues });
      if (res.restartRequired) {
        setEnvStatus({ type: 'warning', text: `Saved ${res.updated.length} setting(s). Some changes require a restart.` });
      } else {
        setEnvStatus({ type: 'success', text: `Saved ${res.updated.length} setting(s). Changes are live.` });
      }
    } catch (err: any) {
      setEnvStatus({ type: 'error', text: `Failed to save: ${err.message}` });
    } finally {
      setEnvSaving(false);
    }
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

  type TabId = 'general' | 'environment' | 'ai' | 'storage';
  const [activeTab, setActiveTab] = useState<TabId>('general');

  const tabs: Array<{ id: TabId; label: string; icon: React.ReactNode }> = [
    { id: 'general',     label: t('settings.tabs.general'),        icon: <Zap size={15} className="settings-tab-icon" /> },
    { id: 'environment', label: t('settings.tabs.environment'),     icon: <Settings2 size={15} className="settings-tab-icon" /> },
    { id: 'ai',          label: t('settings.tabs.ai'),     icon: <Key size={15} className="settings-tab-icon" /> },
    { id: 'storage',     label: t('settings.tabs.storage'),  icon: <Database size={15} className="settings-tab-icon" /> },
  ];

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h1 className="settings-title">{t('settings.title')}</h1>
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
            </EnvSubsection>

            {/* Server */}
            <EnvSubsection title={t('settings.env.server')} isOpen={openSections.server} onToggle={() => toggleSection('server')}>
              <EnvTextRow envKey="SERVER_PORT" label={t('settings.env.serverPort')} description={t('settings.env.serverPortDesc')}
                value={envValues.SERVER_PORT || ''} onChange={handleEnvChange} type="number" placeholder="3001" />
              <EnvPathRow envKey="DATA_DIR" label={t('settings.env.dataDir')} description={t('settings.env.dataDirDesc')}
                value={envValues.DATA_DIR || ''} onChange={handleEnvChange} onBrowse={handleBrowse} placeholder="./data" />
            </EnvSubsection>

            {/* Paths */}
            <EnvSubsection title={t('settings.env.paths')} isOpen={openSections.paths} onToggle={() => toggleSection('paths')}>
              <EnvPathRow envKey="LYRICS_EXPORT_DIR" label={t('settings.env.lyricsExportDir')} description={t('settings.env.lyricsExportDirDesc')}
                value={envValues.LYRICS_EXPORT_DIR || ''} onChange={handleEnvChange} onBrowse={handleBrowse} />
            </EnvSubsection>

            {/* Save bar */}
            <div className="env-save-bar">
              {envStatus && (
                <span className={`env-save-status env-save-status--${envStatus.type}`}>
                  {envStatus.text}
                </span>
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
            <div className="setting-label">{t('settings.ai.defaultProvider')}</div>
            <div className="setting-description">{t('settings.ai.defaultProviderDesc')}</div>
          </div>
          <select id="env-DEFAULT_LLM_PROVIDER" className="env-select"
            value={envValues.DEFAULT_LLM_PROVIDER || 'gemini'}
            onChange={(e) => handleEnvChange('DEFAULT_LLM_PROVIDER', e.target.value)}>
            <option value="gemini">Gemini</option>
            <option value="openai">OpenAI</option>
            <option value="anthropic">Anthropic</option>
            <option value="ollama">Ollama</option>
            <option value="lmstudio">LM Studio</option>
            <option value="unsloth">Unsloth</option>
          </select>
        </div>
        <EnvTextRow envKey="GEMINI_MODEL" label={t('settings.ai.geminiModel')} description={t('settings.ai.geminiModelDesc')}
          value={envValues.GEMINI_MODEL || ''} onChange={handleEnvChange} placeholder="gemini-2.5-flash" />
        <EnvTextRow envKey="OPENAI_MODEL" label={t('settings.ai.openaiModel')} description={t('settings.ai.openaiModelDesc')}
          value={envValues.OPENAI_MODEL || ''} onChange={handleEnvChange} placeholder="gpt-4o-mini" />
        <EnvTextRow envKey="ANTHROPIC_MODEL" label={t('settings.ai.anthropicModel')} description={t('settings.ai.anthropicModelDesc')}
          value={envValues.ANTHROPIC_MODEL || ''} onChange={handleEnvChange} placeholder="claude-3-5-haiku-20241022" />
        <EnvTextRow envKey="OLLAMA_MODEL" label={t('settings.ai.ollamaModel')} description={t('settings.ai.ollamaModelDesc')}
          value={envValues.OLLAMA_MODEL || ''} onChange={handleEnvChange} placeholder="llama3" />
        <EnvTextRow envKey="LMSTUDIO_MODEL" label={t('settings.ai.lmstudioModel')} description={t('settings.ai.lmstudioModelDesc')}
          value={envValues.LMSTUDIO_MODEL || ''} onChange={handleEnvChange} />
        <EnvTextRow envKey="UNSLOTH_MODEL" label={t('settings.ai.unslothModel')} description={t('settings.ai.unslothModelDesc')}
          value={envValues.UNSLOTH_MODEL || ''} onChange={handleEnvChange} />
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
        <EnvTextRow envKey="UNSLOTH_BASE_URL" label={t('settings.ai.unslothUrl')} description={t('settings.ai.unslothUrlDesc')}
          value={envValues.UNSLOTH_BASE_URL || ''} onChange={handleEnvChange} placeholder="http://127.0.0.1:8888" />
        <EnvTextRow envKey="UNSLOTH_USERNAME" label={t('settings.ai.unslothUsername')} description={t('settings.ai.unslothUsernameDesc')}
          value={envValues.UNSLOTH_USERNAME || ''} onChange={handleEnvChange} />
        <EnvPasswordRow envKey="UNSLOTH_PASSWORD" label={t('settings.ai.unslothPassword')} description={t('settings.ai.unslothPasswordDesc')}
          value={envValues.UNSLOTH_PASSWORD || ''} onChange={handleEnvChange} />
      </div>

      {/* Save bar for AI tab */}
      <div className="env-save-bar">
        {envStatus && (
          <span className={`env-save-status env-save-status--${envStatus.type}`}>{envStatus.text}</span>
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

      {/* ═══════════════ GENERAL TAB ═══════════════ */}
      {activeTab === 'general' && (
      <>
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
          value={i18n.language?.split('-')[0] || 'en'}
          options={SUPPORTED_LANGUAGES.map(l => ({ value: l.code, label: `${l.flag} ${l.name}` }))}
          onChange={(v) => i18n.changeLanguage(v)}
        />
      </div>

      {/* Performance Section */}
      <div className="settings-section">
        <div className="settings-section-header">
          <Zap size={16} className="settings-section-icon" />
          <span className="settings-section-title">{t('settings.general.performance')}</span>
        </div>

        <SettingRow
          id="setting-co-resident"
          label={t('settings.general.coResident')}
          description={t('settings.general.coResidentDesc')}
          checked={settings.coResident}
          onChange={(v) => update('coResident', v)}
          badges={[
            { text: '−13s', type: 'speed' },
            { text: '+8GB VRAM', type: 'vram' },
          ]}
        />

        <SettingRow
          id="setting-cache-lm"
          label={t('settings.general.cacheLm')}
          description={t('settings.general.cacheLmDesc')}
          checked={settings.cacheLmCodes}
          onChange={(v) => update('cacheLmCodes', v)}
          badges={[
            { text: '−12s', type: 'speed' },
          ]}
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
        />
      </div>

      {/* Adapters Section (still in General tab) */}
      <div className="settings-section">
        <div className="settings-section-header">
          <Tag size={16} className="settings-section-icon" />
          <span className="settings-section-title">{t('settings.general.adapters')}</span>
        </div>

        <SettingRow
          id="setting-trigger-filename"
          label={t('settings.general.triggerFilename')}
          description={t('settings.general.triggerFilenameDesc')}
          checked={settings.triggerUseFilename}
          onChange={(v) => update('triggerUseFilename', v)}
        />

        {settings.triggerUseFilename && (
          <div className="setting-row">
            <div className="setting-info">
              <div className="setting-label">{t('settings.general.triggerPlacement')}</div>
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

      {/* ═══════════════ STORAGE & DATA TAB ═══════════════ */}
      {activeTab === 'storage' && (
      <>
      {/* Stem Storage Section */}
      <div className="settings-section">
        <div className="settings-section-header">
          <Scissors size={16} className="settings-section-icon" />
          <span className="settings-section-title">{t('settings.storage.stemStorage')}</span>
        </div>

        <div className="setting-row">
          <div className="setting-info">
            <div className="setting-label">{t('settings.storage.extractedStems')}</div>
            <div className="setting-description">
              {stemStats ? (
                <>
                  {stemStats.jobCount} extraction{stemStats.jobCount !== 1 ? 's' : ''} · {stemStats.stemCount} stem{stemStats.stemCount !== 1 ? 's' : ''} · {formatBytes(stemStats.totalBytes)}
                </>
              ) : (
                t('settings.storage.loading')
              )}
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '6px', flexShrink: 0 }}>
            {!stemClearConfirm ? (
              <button
                id="clear-stems-btn"
                onClick={() => setStemClearConfirm(true)}
                disabled={stemClearing || !stemStats || stemStats.jobCount === 0}
                className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
                style={{
                  background: 'rgba(239, 68, 68, 0.15)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  color: '#fca5a5',
                  cursor: (!stemStats || stemStats.jobCount === 0) ? 'not-allowed' : 'pointer',
                  opacity: (!stemStats || stemStats.jobCount === 0) ? 0.5 : 1,
                }}
              >
                🗑️ {t('settings.storage.clearAllStems')}
              </button>
            ) : (
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span className="text-xs text-red-400" style={{ whiteSpace: 'nowrap' }}>{t('settings.storage.deleteAllStems')}</span>
                <button
                  id="clear-stems-confirm-btn"
                  onClick={handleClearStems}
                  disabled={stemClearing}
                  className="px-4 py-2 rounded-lg text-sm font-bold transition-all duration-200"
                  style={{
                    background: '#dc2626',
                    border: '1px solid #ef4444',
                    color: 'white',
                    cursor: stemClearing ? 'wait' : 'pointer',
                    opacity: stemClearing ? 0.6 : 1,
                  }}
                >
                  {stemClearing ? (
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Loader2 size={14} className="animate-spin" /> {t('settings.storage.clearing')}
                    </span>
                  ) : (
                    '🗑️ ' + t('settings.storage.confirm')
                  )}
                </button>
                <button
                  onClick={() => setStemClearConfirm(false)}
                  disabled={stemClearing}
                  className="px-3 py-2 rounded-lg text-sm text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors"
                  style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', cursor: 'pointer' }}
                >
                  {t('settings.storage.cancel')}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Danger Zone */}
      <div className="settings-section" style={{ borderColor: 'rgba(239, 68, 68, 0.2)' }}>
        <div className="settings-section-header">
          <AlertTriangle size={16} style={{ color: '#ef4444' }} />
          <span className="settings-section-title" style={{ color: '#ef4444' }}>{t('settings.storage.dangerZone')}</span>
        </div>

        <div className="setting-row">
          <div className="setting-info">
            <div className="setting-label" style={{ color: '#fca5a5' }}>{t('settings.storage.nukeGenerations')}</div>
            <div className="setting-description">
              Permanently delete <strong>all</strong> generated audio — songs from the library, audio files from disk,
              and all Lyric Studio audio generation entries. Lyrics, profiles, and artist data are preserved.
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '6px', flexShrink: 0 }}>
            {!nukeConfirm ? (
              <button
                id="nuke-generations-btn"
                onClick={() => setNukeConfirm(true)}
                disabled={nukeRunning}
                className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
                style={{
                  background: 'rgba(239, 68, 68, 0.15)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  color: '#fca5a5',
                  cursor: 'pointer',
                }}
                onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.25)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.5)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.15)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.3)'; }}
              >
                ☢️ {t('settings.storage.nukeGenerations')}
              </button>
            ) : (
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span className="text-xs text-red-400" style={{ whiteSpace: 'nowrap' }}>{t('settings.storage.areYouSure')}</span>
                <button
                  id="nuke-confirm-btn"
                  onClick={handleNuke}
                  disabled={nukeRunning}
                  className="px-4 py-2 rounded-lg text-sm font-bold transition-all duration-200"
                  style={{
                    background: '#dc2626',
                    border: '1px solid #ef4444',
                    color: 'white',
                    cursor: nukeRunning ? 'wait' : 'pointer',
                    opacity: nukeRunning ? 0.6 : 1,
                  }}
                >
                  {nukeRunning ? (
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Loader2 size={14} className="animate-spin" /> {t('settings.storage.nuking')}
                    </span>
                  ) : (
                    t('settings.storage.confirmNuke')
                  )}
                </button>
                <button
                  onClick={() => setNukeConfirm(false)}
                  disabled={nukeRunning}
                  className="px-3 py-2 rounded-lg text-sm text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors"
                  style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', cursor: 'pointer' }}
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Nuke Lyrics */}
        <div className="setting-row">
          <div className="setting-info">
            <div className="setting-label" style={{ color: '#fca5a5' }}>{t('settings.storage.nukeLyrics')}</div>
            <div className="setting-description">
              Permanently delete <strong>all</strong> generated/written song lyrics across all artists.
              Profiles and source lyrics are preserved.
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '6px', flexShrink: 0 }}>
            {!nukeLyricsConfirm ? (
              <button
                id="nuke-lyrics-btn"
                onClick={() => setNukeLyricsConfirm(true)}
                disabled={nukeLyricsRunning}
                className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
                style={{
                  background: 'rgba(239, 68, 68, 0.15)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  color: '#fca5a5',
                  cursor: 'pointer',
                }}
                onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.25)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.5)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.15)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.3)'; }}
              >
                🗑️ {t('settings.storage.nukeLyrics')}
              </button>
            ) : (
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span className="text-xs text-red-400" style={{ whiteSpace: 'nowrap' }}>{t('settings.storage.areYouSure')}</span>
                <button
                  id="nuke-lyrics-confirm-btn"
                  onClick={handleNukeLyrics}
                  disabled={nukeLyricsRunning}
                  className="px-4 py-2 rounded-lg text-sm font-bold transition-all duration-200"
                  style={{
                    background: '#dc2626',
                    border: '1px solid #ef4444',
                    color: 'white',
                    cursor: nukeLyricsRunning ? 'wait' : 'pointer',
                    opacity: nukeLyricsRunning ? 0.6 : 1,
                  }}
                >
                  {nukeLyricsRunning ? (
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Loader2 size={14} className="animate-spin" /> {t('settings.storage.nuking')}
                    </span>
                  ) : (
                    t('settings.storage.confirmNukeLyrics')
                  )}
                </button>
                <button
                  onClick={() => setNukeLyricsConfirm(false)}
                  disabled={nukeLyricsRunning}
                  className="px-3 py-2 rounded-lg text-sm text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors"
                  style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', cursor: 'pointer' }}
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Nuke Profiles */}
        <div className="setting-row">
          <div className="setting-info">
            <div className="setting-label" style={{ color: '#fca5a5' }}>{t('settings.storage.nukeProfiles')}</div>
            <div className="setting-description">
              Permanently delete <strong>all</strong> artist profiles and their dependent written lyrics.
              Source lyrics and artist data are preserved.
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '6px', flexShrink: 0 }}>
            {!nukeProfilesConfirm ? (
              <button
                id="nuke-profiles-btn"
                onClick={() => setNukeProfilesConfirm(true)}
                disabled={nukeProfilesRunning}
                className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
                style={{
                  background: 'rgba(239, 68, 68, 0.15)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  color: '#fca5a5',
                  cursor: 'pointer',
                }}
                onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.25)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.5)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.15)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.3)'; }}
              >
                🧬 {t('settings.storage.nukeProfiles')}
              </button>
            ) : (
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span className="text-xs text-red-400" style={{ whiteSpace: 'nowrap' }}>{t('settings.storage.lyricsAlsoDeleted')}</span>
                <button
                  id="nuke-profiles-confirm-btn"
                  onClick={handleNukeProfiles}
                  disabled={nukeProfilesRunning}
                  className="px-4 py-2 rounded-lg text-sm font-bold transition-all duration-200"
                  style={{
                    background: '#dc2626',
                    border: '1px solid #ef4444',
                    color: 'white',
                    cursor: nukeProfilesRunning ? 'wait' : 'pointer',
                    opacity: nukeProfilesRunning ? 0.6 : 1,
                  }}
                >
                  {nukeProfilesRunning ? (
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Loader2 size={14} className="animate-spin" /> {t('settings.storage.nuking')}
                    </span>
                  ) : (
                    t('settings.storage.confirmNukeProfiles')
                  )}
                </button>
                <button
                  onClick={() => setNukeProfilesConfirm(false)}
                  disabled={nukeProfilesRunning}
                  className="px-3 py-2 rounded-lg text-sm text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors"
                  style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', cursor: 'pointer' }}
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
        </div>
        {nukeResult && (
          <div
            className="text-sm px-4 py-3 rounded-lg"
            style={{
              background: nukeResult.startsWith('Failed')
                ? 'rgba(239, 68, 68, 0.1)' : 'rgba(34, 197, 94, 0.1)',
              border: `1px solid ${nukeResult.startsWith('Failed')
                ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)'}`,
              color: nukeResult.startsWith('Failed') ? '#fca5a5' : '#86efac',
              marginTop: '8px',
            }}
          >
            {nukeResult}
          </div>
        )}
      </div>
      </>
      )}
    </div>
  );
};
