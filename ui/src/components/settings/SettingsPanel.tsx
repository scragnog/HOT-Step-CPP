// SettingsPanel.tsx — Application settings with persistent state
//
// All settings use localStorage via usePersistedState and survive page refreshes.
// Environment settings read from / write to the server's .env file.

import React, { useState, useEffect, useCallback } from 'react';
import {
  Zap, Download, Tag, AlertTriangle, Loader2, Settings2,
  FolderOpen, Eye, EyeOff, ChevronRight, Save,
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { songApi, settingsApi } from '../../services/api';
import { lireekApi } from '../../services/lireekApi';
import { FileBrowserModal } from '../shared/FileBrowserModal';
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

/** Toggle switch component */
const Toggle: React.FC<{ checked: boolean; onChange: (v: boolean) => void; id: string }> = ({
  checked,
  onChange,
  id,
}) => (
  <label className="toggle-switch" htmlFor={id}>
    <input
      type="checkbox"
      id={id}
      checked={checked}
      onChange={(e) => onChange(e.target.checked)}
    />
    <div className="toggle-track" />
    <div className="toggle-thumb" />
  </label>
);

/** Single setting row */
const SettingRow: React.FC<{
  id: string;
  label: string;
  description: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  badges?: Array<{ text: string; type: 'speed' | 'vram' | 'rebuild' }>;
}> = ({ id, label, description, checked, onChange, badges }) => (
  <div className="setting-row">
    <div className="setting-info">
      <div className="setting-label">
        {label}
        {badges?.map((b, i) => (
          <span key={i} className={`setting-badge setting-badge--${b.type}`}>
            {b.text}
          </span>
        ))}
      </div>
      <div className="setting-description">{description}</div>
    </div>
    <Toggle checked={checked} onChange={onChange} id={id} />
  </div>
);

/** Select dropdown row for settings */
const SelectRow: React.FC<{
  id: string;
  label: string;
  description: string;
  value: string | number;
  options: Array<{ value: string | number; label: string }>;
  onChange: (v: string) => void;
}> = ({ id, label, description, value, options, onChange }) => (
  <div className="setting-row">
    <div className="setting-info">
      <div className="setting-label">{label}</div>
      <div className="setting-description">{description}</div>
    </div>
    <select
      id={id}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="px-3 py-1.5 rounded-lg bg-zinc-800 border border-white/10 text-sm text-zinc-200 focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 outline-none cursor-pointer min-w-[100px]"
    >
      {options.map(o => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  </div>
);

// ── Environment (.env) sub-components ───────────────────────────────

/** Restart-required keys set (mirrors server) */
const RESTART_KEYS = new Set([
  'ACESTEPCPP_MODELS', 'ACESTEPCPP_ADAPTERS', 'ACESTEPCPP_PORT', 'ACESTEPCPP_HOST',
  'SERVER_PORT', 'DATA_DIR',
]);

/** Sensitive keys — display as masked password fields */
const SENSITIVE_KEYS = new Set([
  'GENIUS_ACCESS_TOKEN', 'GEMINI_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
  'UNSLOTH_PASSWORD',
]);

/** Text / number input row for env settings */
const EnvTextRow: React.FC<{
  envKey: string;
  label: string;
  description: string;
  value: string;
  onChange: (key: string, value: string) => void;
  type?: 'text' | 'number';
  placeholder?: string;
}> = ({ envKey, label, description, value, onChange, type = 'text', placeholder }) => (
  <div className="setting-row">
    <div className="setting-info">
      <div className="setting-label">
        {label}
        {RESTART_KEYS.has(envKey) && (
          <span className="setting-badge setting-badge--restart">⚠️ Restart</span>
        )}
      </div>
      <div className="setting-description">{description}</div>
    </div>
    <input
      id={`env-${envKey}`}
      type={type}
      className={`env-input${type === 'number' ? ' env-input--number' : ''}`}
      value={value}
      onChange={(e) => onChange(envKey, e.target.value)}
      placeholder={placeholder}
    />
  </div>
);

/** Password input row with show/hide toggle */
const EnvPasswordRow: React.FC<{
  envKey: string;
  label: string;
  description: string;
  value: string;
  onChange: (key: string, value: string) => void;
  placeholder?: string;
}> = ({ envKey, label, description, value, onChange, placeholder }) => {
  const [visible, setVisible] = useState(false);
  return (
    <div className="setting-row">
      <div className="setting-info">
        <div className="setting-label">{label}</div>
        <div className="setting-description">{description}</div>
      </div>
      <div className="env-password-wrapper">
        <input
          id={`env-${envKey}`}
          type={visible ? 'text' : 'password'}
          className="env-input"
          value={value}
          onChange={(e) => onChange(envKey, e.target.value)}
          placeholder={placeholder || '••••••••'}
        />
        <button
          type="button"
          className="env-password-toggle"
          onClick={() => setVisible(!visible)}
          title={visible ? 'Hide' : 'Show'}
        >
          {visible ? <EyeOff size={14} /> : <Eye size={14} />}
        </button>
      </div>
    </div>
  );
};

/** Path input row with browse folder button */
const EnvPathRow: React.FC<{
  envKey: string;
  label: string;
  description: string;
  value: string;
  onChange: (key: string, value: string) => void;
  onBrowse: (key: string) => void;
  placeholder?: string;
}> = ({ envKey, label, description, value, onChange, onBrowse, placeholder }) => (
  <div className="setting-row">
    <div className="setting-info">
      <div className="setting-label">
        {label}
        {RESTART_KEYS.has(envKey) && (
          <span className="setting-badge setting-badge--restart">⚠️ Restart</span>
        )}
      </div>
      <div className="setting-description">{description}</div>
    </div>
    <div className="env-path-row">
      <input
        id={`env-${envKey}`}
        type="text"
        className="env-input"
        value={value}
        onChange={(e) => onChange(envKey, e.target.value)}
        placeholder={placeholder || 'Select folder...'}
      />
      <button
        type="button"
        className="env-browse-btn"
        onClick={() => onBrowse(envKey)}
        title="Browse..."
      >
        <FolderOpen size={16} />
      </button>
    </div>
  </div>
);

/** Collapsible subsection header */
const EnvSubsection: React.FC<{
  title: string;
  isOpen: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}> = ({ title, isOpen, onToggle, children }) => (
  <div>
    <div className="env-subsection-header" onClick={onToggle}>
      <ChevronRight
        size={14}
        className={`env-subsection-chevron${isOpen ? ' env-subsection-chevron--open' : ''}`}
      />
      <span className="env-subsection-title">{title}</span>
    </div>
    {isOpen && children}
  </div>
);

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  settings,
  onSettingsChange,
  onNukeComplete,
}) => {
  const { token } = useAuth();
  const [nukeConfirm, setNukeConfirm] = useState(false);
  const [nukeRunning, setNukeRunning] = useState(false);
  const [nukeResult, setNukeResult] = useState<string | null>(null);
  const [nukeLyricsConfirm, setNukeLyricsConfirm] = useState(false);
  const [nukeLyricsRunning, setNukeLyricsRunning] = useState(false);
  const [nukeProfilesConfirm, setNukeProfilesConfirm] = useState(false);
  const [nukeProfilesRunning, setNukeProfilesRunning] = useState(false);

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

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h1 className="settings-title">Settings</h1>
        <p className="settings-subtitle">
          Configure performance and behavior options
        </p>
      </div>

      {/* ── Environment Section ──────────────────────────────────── */}
      <div className="settings-section">
        <div className="settings-section-header">
          <Settings2 size={16} className="settings-section-icon" />
          <span className="settings-section-title">Environment</span>
        </div>

        {envLoading ? (
          <div className="setting-row" style={{ justifyContent: 'center' }}>
            <Loader2 size={16} className="animate-spin" style={{ color: 'rgba(255,255,255,0.4)' }} />
            <span style={{ marginLeft: '8px', color: 'rgba(255,255,255,0.4)', fontSize: '0.8125rem' }}>Loading .env…</span>
          </div>
        ) : (
          <>
            {/* Engine */}
            <EnvSubsection title="Engine" isOpen={openSections.engine} onToggle={() => toggleSection('engine')}>
              <EnvPathRow envKey="ACESTEPCPP_MODELS" label="Models directory" description="Path to GGUF model files for the synthesis engine."
                value={envValues.ACESTEPCPP_MODELS || ''} onChange={handleEnvChange} onBrowse={handleBrowse} />
              <EnvPathRow envKey="ACESTEPCPP_ADAPTERS" label="Adapters directory" description="Path to LoRA/LoKR adapter files (.safetensors)."
                value={envValues.ACESTEPCPP_ADAPTERS || ''} onChange={handleEnvChange} onBrowse={handleBrowse} />
              <EnvTextRow envKey="ACESTEPCPP_PORT" label="Engine port" description="HTTP port for the ace-server C++ engine."
                value={envValues.ACESTEPCPP_PORT || ''} onChange={handleEnvChange} type="number" placeholder="8085" />
              <EnvTextRow envKey="ACESTEPCPP_HOST" label="Engine host" description="Bind address for the engine server."
                value={envValues.ACESTEPCPP_HOST || ''} onChange={handleEnvChange} placeholder="127.0.0.1" />
            </EnvSubsection>

            {/* Server */}
            <EnvSubsection title="Server" isOpen={openSections.server} onToggle={() => toggleSection('server')}>
              <EnvTextRow envKey="SERVER_PORT" label="Server port" description="HTTP port for the Node.js server."
                value={envValues.SERVER_PORT || ''} onChange={handleEnvChange} type="number" placeholder="3001" />
              <EnvPathRow envKey="DATA_DIR" label="Data directory" description="Root directory for databases, audio files, and app data."
                value={envValues.DATA_DIR || ''} onChange={handleEnvChange} onBrowse={handleBrowse} placeholder="./data" />
            </EnvSubsection>

            {/* API Keys */}
            <EnvSubsection title="API Keys" isOpen={openSections.apiKeys} onToggle={() => toggleSection('apiKeys')}>
              <EnvPasswordRow envKey="GENIUS_ACCESS_TOKEN" label="Genius API token" description="For fetching reference lyrics from Genius."
                value={envValues.GENIUS_ACCESS_TOKEN || ''} onChange={handleEnvChange} />
              <EnvPasswordRow envKey="GEMINI_API_KEY" label="Google Gemini key" description="For Gemini-powered lyric generation."
                value={envValues.GEMINI_API_KEY || ''} onChange={handleEnvChange} />
              <EnvPasswordRow envKey="OPENAI_API_KEY" label="OpenAI key" description="For GPT-powered lyric generation."
                value={envValues.OPENAI_API_KEY || ''} onChange={handleEnvChange} />
              <EnvPasswordRow envKey="ANTHROPIC_API_KEY" label="Anthropic key" description="For Claude-powered lyric generation."
                value={envValues.ANTHROPIC_API_KEY || ''} onChange={handleEnvChange} />
            </EnvSubsection>

            {/* LLM Config */}
            <EnvSubsection title="LLM Configuration" isOpen={openSections.llmConfig} onToggle={() => toggleSection('llmConfig')}>
              <div className="setting-row">
                <div className="setting-info">
                  <div className="setting-label">Default LLM provider</div>
                  <div className="setting-description">Which provider to use by default for lyric generation.</div>
                </div>
                <select
                  id="env-DEFAULT_LLM_PROVIDER"
                  className="env-select"
                  value={envValues.DEFAULT_LLM_PROVIDER || 'gemini'}
                  onChange={(e) => handleEnvChange('DEFAULT_LLM_PROVIDER', e.target.value)}
                >
                  <option value="gemini">Gemini</option>
                  <option value="openai">OpenAI</option>
                  <option value="anthropic">Anthropic</option>
                  <option value="ollama">Ollama</option>
                  <option value="lmstudio">LM Studio</option>
                  <option value="unsloth">Unsloth</option>
                </select>
              </div>
              <EnvTextRow envKey="GEMINI_MODEL" label="Gemini model" description="Model name for Gemini API."
                value={envValues.GEMINI_MODEL || ''} onChange={handleEnvChange} placeholder="gemini-2.5-flash" />
              <EnvTextRow envKey="OPENAI_MODEL" label="OpenAI model" description="Model name for OpenAI API."
                value={envValues.OPENAI_MODEL || ''} onChange={handleEnvChange} placeholder="gpt-4o-mini" />
              <EnvTextRow envKey="ANTHROPIC_MODEL" label="Anthropic model" description="Model name for Anthropic API."
                value={envValues.ANTHROPIC_MODEL || ''} onChange={handleEnvChange} placeholder="claude-3-5-haiku-20241022" />
              <EnvTextRow envKey="OLLAMA_MODEL" label="Ollama model" description="Model name for local Ollama instance."
                value={envValues.OLLAMA_MODEL || ''} onChange={handleEnvChange} placeholder="llama3" />
              <EnvTextRow envKey="LMSTUDIO_MODEL" label="LM Studio model" description="Model name for LM Studio."
                value={envValues.LMSTUDIO_MODEL || ''} onChange={handleEnvChange} />
              <EnvTextRow envKey="UNSLOTH_MODEL" label="Unsloth model" description="Model name for Unsloth."
                value={envValues.UNSLOTH_MODEL || ''} onChange={handleEnvChange} />
            </EnvSubsection>

            {/* LLM Endpoints */}
            <EnvSubsection title="LLM Endpoints" isOpen={openSections.llmEndpoints} onToggle={() => toggleSection('llmEndpoints')}>
              <EnvTextRow envKey="OLLAMA_BASE_URL" label="Ollama URL" description="Base URL for the Ollama API."
                value={envValues.OLLAMA_BASE_URL || ''} onChange={handleEnvChange} placeholder="http://localhost:11434" />
              <EnvTextRow envKey="LMSTUDIO_BASE_URL" label="LM Studio URL" description="Base URL for the LM Studio API."
                value={envValues.LMSTUDIO_BASE_URL || ''} onChange={handleEnvChange} placeholder="http://localhost:1234/v1" />
              <EnvTextRow envKey="UNSLOTH_BASE_URL" label="Unsloth URL" description="Base URL for the Unsloth API."
                value={envValues.UNSLOTH_BASE_URL || ''} onChange={handleEnvChange} placeholder="http://127.0.0.1:8888" />
              <EnvTextRow envKey="UNSLOTH_USERNAME" label="Unsloth username" description="Username for Unsloth authentication."
                value={envValues.UNSLOTH_USERNAME || ''} onChange={handleEnvChange} />
              <EnvPasswordRow envKey="UNSLOTH_PASSWORD" label="Unsloth password" description="Password for Unsloth authentication."
                value={envValues.UNSLOTH_PASSWORD || ''} onChange={handleEnvChange} />
            </EnvSubsection>

            {/* Paths */}
            <EnvSubsection title="Paths" isOpen={openSections.paths} onToggle={() => toggleSection('paths')}>
              <EnvPathRow envKey="LYRICS_EXPORT_DIR" label="Lyrics export directory" description="Where exported lyrics files are saved."
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
                    <Loader2 size={14} className="animate-spin" /> Saving…
                  </span>
                ) : (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Save size={14} /> Save Environment
                  </span>
                )}
              </button>
            </div>
          </>
        )}
      </div>

      {/* File Browser Modal for path settings */}
      <FileBrowserModal
        open={browseKey !== null}
        onClose={() => setBrowseKey(null)}
        onSelect={handleBrowseSelect}
        mode="folder"
        startPath={browseKey ? envValues[browseKey] || '' : ''}
        title={browseKey ? `Select folder for ${browseKey}` : 'Select Folder'}
      />

      {/* Performance Section */}
      <div className="settings-section">
        <div className="settings-section-header">
          <Zap size={16} className="settings-section-icon" />
          <span className="settings-section-title">Performance</span>
        </div>

        <SettingRow
          id="setting-co-resident"
          label="Keep DiT & VAE loaded"
          description="Keep both the DiT diffusion model and VAE decoder in VRAM simultaneously instead of swapping them. Eliminates ~13s of model load/unload per generation. Uses more VRAM (~8.2GB for XL)."
          checked={settings.coResident}
          onChange={(v) => update('coResident', v)}
          badges={[
            { text: '−13s', type: 'speed' },
            { text: '+8GB VRAM', type: 'vram' },
          ]}
        />

        <SettingRow
          id="setting-cache-lm"
          label="Cache LM audio codes"
          description="When generating with the same seed and parameters, reuse previously computed audio codes instead of re-running the LM. Saves ~12s on repeat generations."
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
          <span className="settings-section-title">Downloads</span>
        </div>

        <SelectRow
          id="setting-dl-format"
          label="Default format"
          description="Preferred audio format when downloading tracks."
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
          label="MP3 bitrate"
          description="Default bitrate for MP3 downloads."
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
          label="Opus bitrate"
          description="Default bitrate for Opus downloads."
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

      {/* Adapters Section */}
      <div className="settings-section">
        <div className="settings-section-header">
          <Tag size={16} className="settings-section-icon" />
          <span className="settings-section-title">Adapters</span>
        </div>

        <SettingRow
          id="setting-trigger-filename"
          label="Use filename as trigger word"
          description="Auto-inject the adapter filename into the style description at generation time. The trigger word is derived from the adapter filename (without .safetensors extension)."
          checked={settings.triggerUseFilename}
          onChange={(v) => update('triggerUseFilename', v)}
        />

        {settings.triggerUseFilename && (
          <div className="setting-row">
            <div className="setting-info">
              <div className="setting-label">Trigger word placement</div>
              <div className="setting-description">
                {settings.triggerPlacement === 'prepend' && 'Trigger word is added before your style description.'}
                {settings.triggerPlacement === 'append' && 'Trigger word is added after your style description.'}
                {settings.triggerPlacement === 'replace' && 'Trigger word replaces your entire style description.'}
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

      {/* Danger Zone */}
      <div className="settings-section" style={{ borderColor: 'rgba(239, 68, 68, 0.2)' }}>
        <div className="settings-section-header">
          <AlertTriangle size={16} style={{ color: '#ef4444' }} />
          <span className="settings-section-title" style={{ color: '#ef4444' }}>Danger Zone</span>
        </div>

        <div className="setting-row">
          <div className="setting-info">
            <div className="setting-label" style={{ color: '#fca5a5' }}>Nuke Generations</div>
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
                ☢️ Nuke Generations
              </button>
            ) : (
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span className="text-xs text-red-400" style={{ whiteSpace: 'nowrap' }}>Are you sure?</span>
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
                      <Loader2 size={14} className="animate-spin" /> Nuking...
                    </span>
                  ) : (
                    '☢️ Confirm Nuke'
                  )}
                </button>
                <button
                  onClick={() => setNukeConfirm(false)}
                  disabled={nukeRunning}
                  className="px-3 py-2 rounded-lg text-sm text-zinc-400 hover:text-white transition-colors"
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
            <div className="setting-label" style={{ color: '#fca5a5' }}>Nuke Written Lyrics</div>
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
                🗑️ Nuke Lyrics
              </button>
            ) : (
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span className="text-xs text-red-400" style={{ whiteSpace: 'nowrap' }}>Are you sure?</span>
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
                      <Loader2 size={14} className="animate-spin" /> Nuking...
                    </span>
                  ) : (
                    '🗑️ Confirm Nuke'
                  )}
                </button>
                <button
                  onClick={() => setNukeLyricsConfirm(false)}
                  disabled={nukeLyricsRunning}
                  className="px-3 py-2 rounded-lg text-sm text-zinc-400 hover:text-white transition-colors"
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
            <div className="setting-label" style={{ color: '#fca5a5' }}>Nuke Profiles</div>
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
                🧬 Nuke Profiles
              </button>
            ) : (
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span className="text-xs text-red-400" style={{ whiteSpace: 'nowrap' }}>Lyrics will also be deleted!</span>
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
                      <Loader2 size={14} className="animate-spin" /> Nuking...
                    </span>
                  ) : (
                    '🧬 Confirm Nuke'
                  )}
                </button>
                <button
                  onClick={() => setNukeProfilesConfirm(false)}
                  disabled={nukeProfilesRunning}
                  className="px-3 py-2 rounded-lg text-sm text-zinc-400 hover:text-white transition-colors"
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
    </div>
  );
};
