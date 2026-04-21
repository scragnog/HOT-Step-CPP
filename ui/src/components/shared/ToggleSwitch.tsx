import React from 'react';

interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  icon?: React.ReactNode;
}

export const ToggleSwitch: React.FC<ToggleSwitchProps> = ({ checked, onChange, label, icon }) => {
  return (
    <label className="flex items-center gap-2.5 cursor-pointer group">
      <div className="relative flex-shrink-0" style={{ width: 36, height: 20 }}>
        <input
          type="checkbox"
          checked={checked}
          onChange={e => onChange(e.target.checked)}
          className="sr-only peer"
        />
        <div className="absolute inset-0 bg-zinc-700 rounded-full peer-checked:bg-pink-500 transition-colors" />
        <div className="absolute top-[2px] left-[2px] w-4 h-4 bg-white rounded-full shadow-sm transition-transform peer-checked:translate-x-4" />
      </div>
      {label && (
        <div className="flex items-center gap-1.5">
          {icon && <span className="text-zinc-500">{icon}</span>}
          <span className="text-sm text-zinc-400 group-hover:text-zinc-300 transition-colors">
            {label}
          </span>
        </div>
      )}
    </label>
  );
};
