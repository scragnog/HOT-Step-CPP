// settings/SettingsPrimitives.tsx — Reusable primitives for the SettingsPanel
//
// These small components are used across all settings tabs.
// Extracted to keep the main SettingsPanel focused on layout and logic.

import React, { useState } from 'react';
import { Eye, EyeOff, FolderOpen, ChevronRight } from 'lucide-react';
import { Toggle as SharedToggle } from '../shared/Toggle';
import { StyledSelect } from '../shared/StyledSelect';
import { ParamLabel } from '../shared/ParamLabel';

/** This file's own highlight colour — its one existing accent cue was the
 *  select's emerald focus ring. Passed to every StyledSelect/Toggle here. */
const ACCENT = 'emerald' as const;

/** The shared Toggle (components/shared/Toggle.tsx); this name stays for the
 *  settings tabs. New code imports Toggle from shared. */
export const Toggle: React.FC<{ checked: boolean; onChange: (v: boolean) => void; id: string }> = ({ checked, onChange, id }) => (
  <SharedToggle id={id} checked={checked} onChange={onChange} accent={ACCENT} />
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
        <ParamLabel label={label} info={description} className="" />
        {badges?.map((b, i) => (
          <span key={i} className={`setting-badge setting-badge--${b.type}`}>
            {b.text}
          </span>
        ))}
      </div>
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
      <ParamLabel label={label} info={description} className="setting-label" />
    </div>
    <StyledSelect
      id={id}
      accent={ACCENT}
      value={value}
      onChange={(v) => onChange(String(v))}
      options={options}
      className="min-w-[100px]"
    />
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
  'UNSLOTH_PASSWORD', 'OPENAI_COMPAT_API_KEY', 'LMSTUDIO_API_KEY',
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
        <ParamLabel label={label} info={description} className="" />
        {RESTART_KEYS.has(envKey) && (
          <span className="setting-badge setting-badge--restart">⚠️ Restart</span>
        )}
      </div>
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
        <ParamLabel label={label} info={description} className="setting-label" />
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
        <ParamLabel label={label} info={description} className="" />
        {RESTART_KEYS.has(envKey) && (
          <span className="setting-badge setting-badge--restart">⚠️ Restart</span>
        )}
      </div>
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
