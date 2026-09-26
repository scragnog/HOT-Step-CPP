// Toggle.tsx — THE on/off control. Every boolean setting in the app is one of
// these, never a native checkbox (docs/dev/ui-design.md).
//
// One component for what used to be two: settings/SettingsPrimitives.tsx's
// 44×24 CSS toggle and global-bar/BarSection.tsx's 32×16 Tailwind one. Both
// still export their old names as thin wrappers over this, so nothing broke
// when they merged; new code imports from here.
//
// A <button role="switch">, not a hidden <input type="checkbox">: it needs no
// id/htmlFor pairing to be clickable, and stopPropagation on the click keeps a
// toggle inside an accordion header from also toggling the accordion, which
// is where the global bar's copy first grew that line.
//
// With `label` the row is complete on its own: the switch, the label, and —
// with `info` — the hover card that explains it (ParamLabel). A toggle with
// nothing explaining it is the case the rulebook exists to stop.

import React from 'react';
import { ParamLabel } from './ParamLabel';

export type ToggleAccent = 'pink' | 'amber' | 'sky' | 'emerald' | 'purple' | 'teal' | 'cyan' | 'violet';

/** Tailwind needs whole class names in the source, so these cannot be built
 *  by interpolating the accent into a template string. */
const ON: Record<ToggleAccent, string> = {
  pink: 'bg-pink-500', amber: 'bg-amber-500', sky: 'bg-sky-500', emerald: 'bg-emerald-500',
  purple: 'bg-purple-500', teal: 'bg-teal-500', cyan: 'bg-cyan-500', violet: 'bg-violet-500',
};

const SIZE = {
  /** The global bar's rows: dense, many per panel. */
  sm: { track: 'h-4 w-8', thumb: 'h-3 w-3', on: 'translate-x-[17px]', off: 'translate-x-[3px]' },
  /** Settings and studio forms. */
  md: { track: 'h-6 w-11', thumb: 'h-5 w-5', on: 'translate-x-[22px]', off: 'translate-x-[2px]' },
} as const;

export interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  accent?: ToggleAccent;
  size?: keyof typeof SIZE;
  disabled?: boolean;
  id?: string;
  /** Label text beside the switch. Pass a plain string; it is rendered through
   *  ParamLabel so `info` and `meta` behave as they do on any field. */
  label?: string;
  /** What this switch does and what changes when it is on — the hover card. */
  info?: string;
  meta?: string;
  /** Classes for the wrapper row when `label` is set. */
  className?: string;
  title?: string;
  'aria-label'?: string;
}

export const Toggle: React.FC<ToggleProps> = ({
  checked, onChange, accent = 'pink', size = 'md', disabled = false, id, label, info, meta, className = '', title, 'aria-label': ariaLabel,
}) => {
  const s = SIZE[size];
  const button = (
    <button
      type="button"
      role="switch"
      id={id}
      aria-checked={checked}
      aria-label={ariaLabel ?? (label ? undefined : title)}
      title={title}
      disabled={disabled}
      onClick={(e) => { e.stopPropagation(); if (!disabled) onChange(!checked); }}
      className={`relative inline-flex ${s.track} items-center rounded-full transition-colors duration-200 flex-shrink-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-1 focus-visible:ring-zinc-400 ${
        checked ? ON[accent] : 'bg-zinc-300 dark:bg-zinc-700'} ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
    >
      <span className={`inline-block ${s.thumb} rounded-full bg-white shadow-sm transform transition-transform duration-200 ${checked ? s.on : s.off}`} />
    </button>
  );
  if (!label) return button;
  return (
    <div className={`flex items-center gap-2 ${disabled ? 'opacity-60' : ''} ${className}`}>
      {button}
      <ParamLabel
        label={label}
        info={info}
        meta={meta}
        underline={!!info}
        className="text-xs text-zinc-700 dark:text-zinc-300 cursor-pointer select-none"
        rootClassName="min-w-0"
      />
    </div>
  );
};

export default Toggle;
