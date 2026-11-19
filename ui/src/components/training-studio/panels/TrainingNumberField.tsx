// TrainingNumberField.tsx — the numeric field and byte formatter shared by the
// YuE2 stage cards. The reset affordance lives here so every card compares
// against its own server default the same way.

import React from 'react';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { INPUT } from '@/components/training-studio/lib/yue2CardUi';

export const NumField: React.FC<{
  label: string; value: number; onChange: (v: number) => void; step?: number; hint?: string;
  info?: string; meta?: string; disabled?: boolean;
  /** The server-default value to compare against; the icon calls onReset,
   *  not a write of this value. */
  defaultValue?: number;
  onReset?: () => void;
}> = ({ label, value, onChange, step = 1, hint, info, meta, disabled, defaultValue, onReset }) => (
  <label className={`flex flex-col gap-1${disabled ? ' opacity-50' : ''}`}>
    <ParamLabel
      label={label}
      info={info}
      meta={meta}
      className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
      onReset={defaultValue !== undefined && value !== defaultValue && !disabled ? onReset : undefined}
    />
    <input
      type="number" className={INPUT} value={value} step={step} disabled={disabled}
      onChange={e => onChange(Number(e.target.value))}
    />
    {hint && <span className="text-[10px] text-zinc-500 leading-snug">{hint}</span>}
  </label>
);
