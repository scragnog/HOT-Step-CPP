// kit/Toggle.tsx — the kit switch (tint when on); the original ../Toggle is untouched. No accent prop: the tint is the colour. Was: THE on/off control. Every boolean setting in the app is one of
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
import { ParamLabel } from '@/components/shared/controls/ParamLabel';


const SIZE = {
  /** The global bar's rows: dense, many per panel. */
  sm: { track: 'h-4 w-8', thumb: 'h-3 w-3', on: 'translate-x-[17px]', off: 'translate-x-[3px]' },
  /** Settings and studio forms. */
  md: { track: 'h-6 w-11', thumb: 'h-5 w-5', on: 'translate-x-[22px]', off: 'translate-x-[2px]' },
} as const;

export interface KitToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
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
  /** The value to restore when the reset icon is clicked. Set this to show the
   *  icon while `checked` differs from it. */
  defaultValue?: boolean;
  /** Called instead of onChange(defaultValue) when the reset icon is
   *  clicked. For a server-default overlay form, resetting means deleting
   *  the edit key rather than writing a value — the caller supplies that. */
  onReset?: () => void;
}

export const KitToggle: React.FC<KitToggleProps> = ({
  checked, onChange, size = 'md', disabled = false, id, label, info, meta, className = '', title, 'aria-label': ariaLabel, defaultValue, onReset,
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
      className={`relative inline-flex ${s.track} items-center rounded-full transition-colors duration-glow motion-reduce:transition-none flex-shrink-0 kit-focus ${checked ? 'kit-fill' : 'kit-well'} ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
    >
      <span className={`inline-block ${s.thumb} rounded-full bg-ink shadow-bevel transform transition-transform duration-glow motion-reduce:transition-none ${checked ? s.on : s.off}`} />
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
        className="text-xs text-ink-muted cursor-pointer select-none"
        rootClassName="min-w-0"
        onReset={defaultValue !== undefined && checked !== defaultValue && !disabled ? (onReset ?? (() => onChange(defaultValue))) : undefined}
      />
    </div>
  );
};

export default KitToggle;
