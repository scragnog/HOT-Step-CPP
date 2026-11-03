// AccordionSection.tsx — the collapsible section shell the post-processing
// panel is built from: persisted open state, info tooltip, optional toggle and
// reset, and a badge slot. Extracted so other bars can share the same shell.

import React from 'react';
import { ChevronDown } from 'lucide-react';
import { usePersistedState } from '../../hooks/usePersistedState';
import { ParamLabel } from '../shared/ParamLabel';
import { ToggleSwitch } from './BarSection';

// ── Accordion Section ───────────────────────────────────────────

export interface AccordionProps {
  icon: React.ReactNode;
  label: string;
  /** What the section does. Each of these used to be a paragraph pinned to the
   *  top of the open section, which meant every section cost three lines of
   *  prose before its first control. It now hangs off a "?" beside the label,
   *  readable while the section is still shut. */
  info?: string;
  badge?: React.ReactNode;
  accentColor: string;
  toggle?: { checked: boolean; onChange: (v: boolean) => void };
  /** Resets the section's own on/off toggle to its default. Leaves whatever
   *  the section contains alone — that's what its own inner group Reset (if
   *  any) already covers. */
  onReset?: () => void;
  persistKey: string;  // localStorage key for remembering open/closed state
  children: React.ReactNode;
}

export const Accordion: React.FC<AccordionProps> = ({
  icon, label, info, badge, accentColor, toggle, onReset, persistKey, children,
}) => {
  const [open, setOpen] = usePersistedState(persistKey, false);

  const accentMap: Record<string, string> = {
    cyan: 'border-cyan-500/20 bg-cyan-500/5',
    violet: 'border-violet-500/20 bg-violet-500/5',
    amber: 'border-amber-500/20 bg-amber-500/5',
    emerald: 'border-emerald-500/20 bg-emerald-500/5',
    pink: 'border-pink-500/20 bg-pink-500/5',
    sky: 'border-sky-500/20 bg-sky-500/5',
  };
  const activeStyle = toggle?.checked ? (accentMap[accentColor] || '') : '';

  return (
    <div className={`rounded-xl border transition-all ${activeStyle || 'border-zinc-200 dark:border-white/5 bg-zinc-100/30 dark:bg-zinc-800/30'}`}>
      {/* Header */}
      <div
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={() => setOpen(o => !o)}
        onKeyDown={e => {
          if (e.target === e.currentTarget && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault();
            setOpen(o => !o);
          }
        }}
        className="w-full flex items-center gap-2 px-3 py-2 text-left cursor-pointer"
      >
        <span className={`flex-shrink-0 ${toggle?.checked ? `text-${accentColor}-400` : 'text-zinc-500'}`}>
          {icon}
        </span>
        <ParamLabel label={label} info={info} underline={false}
          className="text-sm text-zinc-700 dark:text-zinc-300 font-medium"
          rootClassName="flex-1"
          onReset={onReset} />
        {badge && <span className="flex-shrink-0">{badge}</span>}
        {toggle && (
          <span onClick={e => e.stopPropagation()}>
            <ToggleSwitch checked={toggle.checked} onChange={toggle.onChange} accentColor={accentColor as 'pink' | 'emerald' | 'sky' | 'purple' | 'amber'} />
          </span>
        )}
        <ChevronDown
          size={14}
          className={`text-zinc-500 transition-transform flex-shrink-0 ${open ? 'rotate-180' : ''}`}
        />
      </div>

      {/* Content */}
      {open && (
        <div className="px-3 pb-3 pt-0 border-t border-zinc-200 dark:border-white/5">
          {children}
        </div>
      )}
    </div>
  );
};

