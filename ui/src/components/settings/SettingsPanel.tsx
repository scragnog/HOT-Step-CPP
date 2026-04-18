// SettingsPanel.tsx — Application settings with persistent state
//
// First settings page for HOT-Step CPP. All settings use localStorage
// via usePersistedState and survive page refreshes.

import React from 'react';
import { Zap, Brain, Settings } from 'lucide-react';
import { usePersistedState } from '../../hooks/usePersistedState';
import './SettingsPanel.css';

export interface AppSettings {
  coResident: boolean;
  cacheLmCodes: boolean;
}

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

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  settings,
  onSettingsChange,
}) => {
  const update = (key: keyof AppSettings, value: boolean) => {
    onSettingsChange({ ...settings, [key]: value });
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
    </div>
  );
};
