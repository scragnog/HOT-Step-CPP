// EditableSlider.tsx — Slider with inline editable value display
import React, { useState } from 'react';
import { RotateCcw } from 'lucide-react';

interface EditableSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  formatDisplay?: (v: number) => string;
  helpText?: string;
  /** The value to restore when the reset icon is clicked. Set this to show the
   *  icon while `value` differs from it. */
  defaultValue?: number;
}

export const EditableSlider: React.FC<EditableSliderProps> = ({
  label, value, min, max, step, onChange, formatDisplay, helpText, defaultValue,
}) => {
  const [editing, setEditing] = useState(false);
  const [editVal, setEditVal] = useState('');
  const display = formatDisplay ? formatDisplay(value) : value.toString();
  const showReset = defaultValue !== undefined && value !== defaultValue;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="inline-flex items-center gap-1">
          <label className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{label}</label>
          {showReset && (
            <button
              type="button"
              onClick={() => onChange(defaultValue as number)}
              title={`Reset ${label} to default`}
              className="text-zinc-400/70 hover:text-cyan-500 transition-colors"
            >
              <RotateCcw size={11} />
            </button>
          )}
        </span>
        {editing ? (
          <input
            autoFocus
            type="text"
            value={editVal}
            onChange={e => setEditVal(e.target.value)}
            onBlur={() => {
              const n = parseFloat(editVal);
              if (!isNaN(n)) onChange(Math.min(max, Math.max(min, n)));
              setEditing(false);
            }}
            onKeyDown={e => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
            className="w-20 px-1.5 py-0.5 text-[10px] text-right bg-zinc-200 dark:bg-black/20 border border-cyan-500/50 rounded text-white outline-none font-mono"
          />
        ) : (
          <span
            className="text-[10px] text-zinc-600 dark:text-zinc-400 font-mono cursor-pointer hover:text-cyan-400 transition-colors"
            onClick={() => { setEditing(true); setEditVal(String(value)); }}
            title="Click to edit"
          >{display}</span>
        )}
      </div>
      <input
        type="range" value={value} onChange={e => onChange(parseFloat(e.target.value))}
        min={min} max={max} step={step} className="w-full h-1.5 accent-cyan-500"
      />
      {helpText && <p className="text-[9px] text-zinc-600">{helpText}</p>}
    </div>
  );
};
