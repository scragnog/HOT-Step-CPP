// EditableSlider.tsx — Slider with inline editable value display
import React, { useState } from 'react';

interface EditableSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  formatDisplay?: (v: number) => string;
  helpText?: string;
}

export const EditableSlider: React.FC<EditableSliderProps> = ({
  label, value, min, max, step, onChange, formatDisplay, helpText,
}) => {
  const [editing, setEditing] = useState(false);
  const [editVal, setEditVal] = useState('');
  const display = formatDisplay ? formatDisplay(value) : value.toString();

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <label className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{label}</label>
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
            className="w-20 px-1.5 py-0.5 text-[10px] text-right bg-black/20 border border-cyan-500/50 rounded text-white outline-none font-mono"
          />
        ) : (
          <span
            className="text-[10px] text-zinc-400 font-mono cursor-pointer hover:text-cyan-400 transition-colors"
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
