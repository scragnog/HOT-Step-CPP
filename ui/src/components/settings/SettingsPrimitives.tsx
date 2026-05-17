// settings/SettingsPrimitives.tsx — Reusable primitives for the SettingsPanel
//
// These small components are used across all settings tabs.
// Extracted to keep the main SettingsPanel focused on layout and logic.

import React, { useState } from 'react';
import { Eye, EyeOff, FolderOpen, ChevronRight } from 'lucide-react';

/** Toggle switch component */
export const Toggle: React.FC<{ checked: boolean; onChange: (v: boolean) => void; id: string }> = ({
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
export const SettingRow: React.FC<{
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
export const SelectRow: React.FC<{
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
      className="px-3 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 outline-none cursor-pointer min-w-[100px]"
    >
      {options.map(o => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  </div>
);

// ── Environment (.env) sub-components ───────────────────────────────

/** Restart-required keys set (mirrors server) */
export const RESTART_KEYS = new Set([
  'ACESTEPCPP_MODELS', 'ACESTEPCPP_ADAPTERS', 'ACESTEPCPP_PORT', 'ACESTEPCPP_HOST',
  'SERVER_PORT', 'DATA_DIR',
]);

/** Sensitive keys — display as masked password fields */
export const SENSITIVE_KEYS = new Set([
  'GENIUS_ACCESS_TOKEN', 'GEMINI_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
  'UNSLOTH_PASSWORD', 'OPENAI_COMPAT_API_KEY',
]);

/** Text / number input row for env settings */
export const EnvTextRow: React.FC<{
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
export const EnvPasswordRow: React.FC<{
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
export const EnvPathRow: React.FC<{
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
export const EnvSubsection: React.FC<{
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
