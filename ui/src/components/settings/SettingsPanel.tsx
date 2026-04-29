// SettingsPanel.tsx — Application settings with persistent state
//
// All settings use localStorage via usePersistedState and survive page refreshes.

import React, { useState } from 'react';
import { Zap, Download, Tag, AlertTriangle, Loader2 } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { songApi } from '../../services/api';
import { lireekApi } from '../../services/lireekApi';
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

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  settings,
  onSettingsChange,
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

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h1 className="settings-title">Settings</h1>
        <p className="settings-subtitle">
          Configure performance and behavior options
        </p>
      </div>

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
