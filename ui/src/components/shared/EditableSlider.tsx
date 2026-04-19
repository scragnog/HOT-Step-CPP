import React, { useState, useEffect, useRef } from 'react';

interface EditableSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  onChangeCommitted?: (value: number) => void;
  formatDisplay?: (value: number) => string;
  helpText?: string;
  tooltip?: string;
  title?: string;
  autoLabel?: string;
  disabled?: boolean;
  disabledReason?: string;
}

export const EditableSlider: React.FC<EditableSliderProps> = ({
  label, value, min, max, step, onChange, onChangeCommitted,
  formatDisplay, helpText, tooltip, title = '', autoLabel = 'Auto',
  disabled = false, disabledReason,
}) => {
  const [inputValue, setInputValue] = useState(value.toString());
  const [isEditing, setIsEditing] = useState(false);
  const [dragValue, setDragValue] = useState<number | null>(null);
  const isDragging = useRef(false);
  const effectiveValue = dragValue ?? value;

  useEffect(() => {
    if (!isEditing) setInputValue(value.toString());
  }, [value, isEditing]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleInputBlur = () => {
    setIsEditing(false);
    const numValue = parseFloat(inputValue);
    if (!isNaN(numValue)) {
      const clampedValue = Math.max(min, Math.min(max, numValue));
      (onChangeCommitted || onChange)(clampedValue);
      setInputValue(clampedValue.toString());
    } else {
      setInputValue(value.toString());
    }
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') handleInputBlur();
    else if (e.key === 'Escape') { setInputValue(value.toString()); setIsEditing(false); }
  };

  const handleRangeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return;
    const v = Number(e.target.value);
    if (onChangeCommitted) { setDragValue(v); isDragging.current = true; }
    else { onChange(v); }
  };

  const handleRangeCommit = () => {
    if (onChangeCommitted && isDragging.current && dragValue !== null) {
      onChangeCommitted(dragValue);
      setDragValue(null);
      isDragging.current = false;
    }
  };

  const displayValue = formatDisplay ? formatDisplay(effectiveValue) : (effectiveValue === min && autoLabel ? autoLabel : effectiveValue.toString());

  return (
    <div className={`space-y-2 ${disabled ? 'opacity-50' : ''}`}>
      <div className="flex items-center justify-between">
        <div className="relative flex items-center group/tip">
          <label className="text-xs font-medium text-zinc-300 cursor-default" title={title}>{label}</label>
          {tooltip && (
            <div className="absolute left-0 bottom-full mb-1.5 hidden group-hover/tip:block z-50 w-56 bg-zinc-800 text-zinc-100 text-[10px] leading-relaxed rounded-lg px-2.5 py-2 shadow-xl border border-white/10 pointer-events-none">
              {tooltip}
            </div>
          )}
        </div>
        {isEditing && !disabled ? (
          <input type="number" value={inputValue} onChange={handleInputChange}
            onBlur={handleInputBlur} onKeyDown={handleInputKeyDown}
            onFocus={() => setIsEditing(true)} min={min} max={max} step={step} autoFocus
            className="text-xs font-mono text-white bg-zinc-800 border border-pink-500 px-2 py-0.5 rounded-lg w-20 text-right shadow-sm focus:outline-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
        ) : (
          <span onClick={() => !disabled && setIsEditing(true)}
            className={`text-xs font-mono text-zinc-200 bg-zinc-800 border border-zinc-700 px-2.5 py-1 rounded-lg transition-all shadow-sm ${disabled ? 'cursor-not-allowed' : 'cursor-pointer hover:bg-zinc-700'}`}
          >{displayValue}</span>
        )}
      </div>
      <div className="relative h-2 bg-zinc-800 rounded-full shadow-inner">
        <input type="range" min={min} max={max} step={step} value={effectiveValue}
          onChange={handleRangeChange} onPointerUp={handleRangeCommit} onTouchEnd={handleRangeCommit}
          className={`absolute inset-0 w-full h-full opacity-0 ${disabled ? 'cursor-not-allowed' : 'cursor-pointer'}`}
          disabled={disabled}
        />
        <div className="absolute top-0 left-0 h-full bg-gradient-to-r from-pink-500 to-rose-600 rounded-full pointer-events-none transition-all duration-150"
          style={{ width: `${Math.max(0, Math.min(100, ((effectiveValue - min) / (max - min)) * 100))}%` }}
        />
        <div className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-zinc-200 rounded-full shadow-md border-2 border-pink-500 pointer-events-none transition-all duration-150"
          style={{ left: `clamp(0px, calc(${((effectiveValue - min) / (max - min)) * 100}% - 8px), calc(100% - 16px))` }}
        />
      </div>
      {(disabled && disabledReason) ? (
        <p className="text-[10px] text-amber-400 flex items-center gap-1">
          <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v.01M12 9v3m0 8a9 9 0 110-18 9 9 0 010 18z" /></svg>
          {disabledReason}
        </p>
      ) : helpText ? (
        <p className="text-[10px] text-zinc-500">{helpText}</p>
      ) : null}
    </div>
  );
};
