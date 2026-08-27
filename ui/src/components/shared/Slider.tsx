// Slider.tsx — Reusable slider with label, value display, and optional number input
// Ported to Tailwind styling.
//
// Pass `info` to explain the knob. The global param bar's dropdowns used to
// print that explanation as a paragraph under every slider, which was useful
// and enormous — five knobs of prose and the panel outgrew the screen. It now
// hangs off a "?" beside the label instead (see ParamLabel).

import React from 'react';
import { ParamLabel } from './ParamLabel';

interface SliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  showInput?: boolean;
  /** What the knob does, shown in a hover card off a "?" beside the label. */
  info?: string;
  /** Optional one-liner above the explanation — default, range, units. */
  infoMeta?: string;
  /** Extra control pinned to the right of the header row, beside the value.
   *  For per-knob modes that aren't points on the scale — Duration's Auto,
   *  where the backend decides the number instead of the user. */
  headerRight?: React.ReactNode;
}

export const Slider: React.FC<SliderProps> = ({
  label, value, onChange, min, max, step, suffix = '', showInput = false, headerRight,
  info, infoMeta,
}) => {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <ParamLabel
          label={label}
          info={info}
          meta={infoMeta}
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
        />
        <div className="flex items-center gap-1.5">
        {showInput ? (
          <input
            type="number"
            className="w-16 px-2 py-0.5 text-xs text-right text-zinc-700 dark:text-zinc-300 bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 rounded-lg outline-none focus:border-pink-500/50"
            value={value}
            onChange={e => onChange(parseFloat(e.target.value) || min)}
            min={min}
            max={max}
            step={step}
          />
        ) : (
          <span className="text-xs text-zinc-600 dark:text-zinc-400 font-mono">{value}{suffix}</span>
        )}
        {headerRight}
        </div>
      </div>
      <input
        type="range"
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
        className="w-full"
      />
    </div>
  );
};
