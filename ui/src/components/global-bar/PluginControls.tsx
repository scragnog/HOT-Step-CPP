// PluginControls.tsx — Dynamic UI controls rendered from Lua plugin param schemas
//
// Takes a plugin's `params` array and renders the appropriate controls
// (sliders, selects, toggles, text inputs) with a Reset button.
// Values are stored in a flat { "pluginName:key": value } map.
//
// Renders as a collapsible accordion, collapsed by default.
// Open/closed state is persisted per-plugin via localStorage.
//
// A param's `hint` is its explanation and now lands in a hover card off a "?"
// beside the label, not as a paragraph underneath. Plugins need no change for
// this — the same `hint` field feeds it. `text` params keep using the hint as
// placeholder text too, since an empty box with no example is hard to guess at.

import React from 'react';
import { RotateCcw, ChevronDown } from 'lucide-react';
import { Slider } from '../shared/Slider';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle, type ToggleAccent } from '../shared/Toggle';
import { usePersistedState } from '../../hooks/usePersistedState';
import type { PluginParamSchema } from '../../types/pluginTypes';

const inputClasses = "w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors";

// Accent color mapping — plugins declare an accent name, we map to Tailwind
const accentMap: Record<string, { border: string; bg: string; text: string; hover: string }> = {
  amber:   { border: 'border-amber-500/20',   bg: 'bg-amber-500/5',   text: 'text-amber-400',   hover: 'hover:text-amber-300' },
  cyan:    { border: 'border-cyan-500/20',    bg: 'bg-cyan-500/5',    text: 'text-cyan-400',    hover: 'hover:text-cyan-300' },
  blue:    { border: 'border-blue-500/20',    bg: 'bg-blue-500/5',    text: 'text-blue-400',    hover: 'hover:text-blue-300' },
  teal:    { border: 'border-teal-500/20',    bg: 'bg-teal-500/5',    text: 'text-teal-400',    hover: 'hover:text-teal-300' },
  green:   { border: 'border-green-500/20',   bg: 'bg-green-500/5',   text: 'text-green-400',   hover: 'hover:text-green-300' },
  emerald: { border: 'border-emerald-500/20', bg: 'bg-emerald-500/5', text: 'text-emerald-400', hover: 'hover:text-emerald-300' },
  purple:  { border: 'border-purple-500/20',  bg: 'bg-purple-500/5',  text: 'text-purple-400',  hover: 'hover:text-purple-300' },
  indigo:  { border: 'border-indigo-500/20',  bg: 'bg-indigo-500/5',  text: 'text-indigo-400',  hover: 'hover:text-indigo-300' },
  orange:  { border: 'border-orange-500/20',  bg: 'bg-orange-500/5',  text: 'text-orange-400',  hover: 'hover:text-orange-300' },
  pink:    { border: 'border-pink-500/20',    bg: 'bg-pink-500/5',    text: 'text-pink-400',    hover: 'hover:text-pink-300' },
  rose:    { border: 'border-rose-500/20',    bg: 'bg-rose-500/5',    text: 'text-rose-400',    hover: 'hover:text-rose-300' },
  sky:     { border: 'border-sky-500/20',     bg: 'bg-sky-500/5',     text: 'text-sky-400',     hover: 'hover:text-sky-300' },
  violet:  { border: 'border-violet-500/20',  bg: 'bg-violet-500/5',  text: 'text-violet-400',  hover: 'hover:text-violet-300' },
};
const defaultAccent = accentMap.cyan;

// StyledSelect/Toggle only know the app's 8-accent palette; plugins declare a
// wider set of loose Tailwind color names (accentMap above), so map the odd
// ones onto their nearest match instead of widening the shared components.
const SHARED_ACCENT: Record<string, ToggleAccent> = {
  amber: 'amber', cyan: 'cyan', teal: 'teal', emerald: 'emerald', purple: 'purple',
  pink: 'pink', sky: 'sky', violet: 'violet',
  blue: 'sky', green: 'emerald', indigo: 'violet', orange: 'amber', rose: 'pink',
};

interface PluginControlsProps {
  pluginName: string;
  displayName: string;
  accent?: string;
  params: PluginParamSchema[];
  values: Record<string, string>;
  onChange: (key: string, value: string) => void;
  onReset: () => void;
}

export const PluginControls: React.FC<PluginControlsProps> = ({
  pluginName,
  displayName,
  accent,
  params,
  values,
  onChange,
  onReset,
}) => {
  const [isOpen, setIsOpen] = usePersistedState(`hs-pluginAccordion-${pluginName}`, false);

  if (!params || params.length === 0) return null;

  const a = (accent && accentMap[accent]) || defaultAccent;
  const sharedAccent: ToggleAccent = (accent && SHARED_ACCENT[accent]) || 'pink';

  // Get value for a param, falling back to its declared default
  const getVal = (p: PluginParamSchema): string => {
    const k = `${pluginName}:${p.key}`;
    if (values[k] !== undefined) return values[k];
    if (p.default !== undefined) return String(p.default);
    if (p.type === 'slider') return String(p.min ?? 0);
    if (p.type === 'toggle') return 'false';
    return '';
  };

  // Check visibility condition
  const isVisible = (p: PluginParamSchema): boolean => {
    if (!p.visible_when) return true;
    const depVal = getVal(params.find(pp => pp.key === p.visible_when!.key) || p);
    return depVal === p.visible_when.equals;
  };

  const visibleParams = params.filter(isVisible);
  if (visibleParams.length === 0) return null;

  return (
    <div className={`rounded-xl border ${a.border} ${a.bg} transition-all overflow-hidden`}>
      {/* Accordion header */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full flex items-center justify-between px-3 py-2 ${a.hover.replace('hover:text-', 'hover:bg-').replace('300', '500/5')} transition-colors`}
      >
        <div className="flex items-center gap-2">
          <ChevronDown size={12} className={`${a.text} transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
          <span className={`text-[10px] font-semibold ${a.text} uppercase tracking-wider`}>
            {displayName} Controls
          </span>
        </div>
        <button type="button" onClick={(e) => { e.stopPropagation(); onReset(); }}
          className={`flex items-center gap-1 text-[10px] ${a.text} ${a.hover} transition-colors`}>
          <RotateCcw size={10} /> Reset
        </button>
      </button>

      {/* Collapsible param content */}
      {isOpen && (
        <div className="px-3 pb-3 space-y-3">
          {visibleParams.map(p => {
            const val = getVal(p);
            const fullKey = `${pluginName}:${p.key}`;

            switch (p.type) {
              case 'slider':
                return (
                  <div key={p.key}>
                    <Slider
                      label={p.label}
                      info={p.hint}
                      value={parseFloat(val) || 0}
                      onChange={v => onChange(fullKey, String(v))}
                      min={p.min ?? 0}
                      max={p.max ?? 1}
                      step={p.step ?? 0.01}
                      showInput
                    />
                  </div>
                );

              case 'select':
                return (
                  <div key={p.key}>
                    <ParamLabel label={p.label} info={p.hint} className={`text-[10px] ${a.text}`} rootClassName="flex mb-1" />
                    <StyledSelect
                      accent={sharedAccent}
                      value={val}
                      onChange={v => onChange(fullKey, v)}
                      options={(p.options || []).map(o => ({ value: o.value, label: o.label }))}
                      className="w-full"
                    />
                  </div>
                );

              case 'toggle':
                return (
                  <div key={p.key} className="flex items-center justify-between">
                    <ParamLabel label={p.label} info={p.hint} className="text-xs text-zinc-400" />
                    <Toggle
                      size="sm"
                      accent={sharedAccent}
                      checked={val === 'true'}
                      onChange={v => onChange(fullKey, v ? 'true' : 'false')}
                      aria-label={p.label}
                    />
                  </div>
                );

              case 'text':
                return (
                  <div key={p.key}>
                    <ParamLabel label={p.label} info={p.hint} className={`text-[10px] ${a.text}`} rootClassName="flex mb-1" />
                    <input
                      className={inputClasses}
                      value={val}
                      onChange={e => onChange(fullKey, e.target.value)}
                      placeholder={p.hint || ''}
                    />
                  </div>
                );

              default:
                return null;
            }
          })}
        </div>
      )}
    </div>
  );
};
