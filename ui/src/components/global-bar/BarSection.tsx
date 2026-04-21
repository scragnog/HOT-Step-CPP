// BarSection.tsx — Reusable hover-to-expand section for the global param bar
//
// Shows a compact header with label + summary badge.
// On hover (or click), expands a floating dropdown panel below.
// Each section has a unique accent tint that is always visible as its background.
// Optionally shows a toggle switch in the header (for LM / Mastering).

import React, { useRef, useCallback, useEffect } from 'react';

// ── Accent color lookup ────────────────────────────────────────────────────
// Tailwind JIT can't compile dynamic class names like `bg-${color}-500/10`,
// so we map accent names to concrete classes.

const ACCENT_STYLES: Record<string, {
  bg: string;        // resting background tint
  bgHover: string;   // hover/active background tint (stronger)
  border: string;    // active bottom border
  iconColor: string; // icon color when active
}> = {
  pink:    { bg: 'bg-pink-500/5',    bgHover: 'bg-pink-500/10',    border: 'border-pink-500',    iconColor: 'text-pink-400' },
  emerald: { bg: 'bg-emerald-500/5', bgHover: 'bg-emerald-500/10', border: 'border-emerald-500', iconColor: 'text-emerald-400' },
  sky:     { bg: 'bg-sky-500/5',     bgHover: 'bg-sky-500/10',     border: 'border-sky-500',     iconColor: 'text-sky-400' },
  purple:  { bg: 'bg-purple-500/5',  bgHover: 'bg-purple-500/10',  border: 'border-purple-500',  iconColor: 'text-purple-400' },
  amber:   { bg: 'bg-amber-500/5',   bgHover: 'bg-amber-500/10',   border: 'border-amber-500',   iconColor: 'text-amber-400' },
};

interface BarSectionProps {
  id: string;
  label: string;
  icon: React.ReactNode;
  badge: React.ReactNode;
  accentColor?: string;
  children: React.ReactNode;
  isOpen: boolean;
  onOpen: () => void;
  onClose: () => void;
  /** Optional toggle rendered in the header bar (e.g. LM on/off, Mastering on/off).
   *  The element handles its own onClick and should call e.stopPropagation(). */
  headerToggle?: React.ReactNode;
}

const HOVER_CLOSE_DELAY = 200; // ms

export const BarSection: React.FC<BarSectionProps> = ({
  id, label, icon, badge, accentColor = 'pink', children,
  isOpen, onOpen, onClose, headerToggle,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const accent = ACCENT_STYLES[accentColor] || ACCENT_STYLES.pink;

  const cancelClose = useCallback(() => {
    if (closeTimer.current) {
      clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
  }, []);

  const scheduleClose = useCallback(() => {
    cancelClose();
    closeTimer.current = setTimeout(() => {
      onClose();
    }, HOVER_CLOSE_DELAY);
  }, [onClose, cancelClose]);

  const handleMouseEnter = useCallback(() => {
    cancelClose();
    onOpen();
  }, [onOpen, cancelClose]);

  const handleMouseLeave = useCallback(() => {
    scheduleClose();
  }, [scheduleClose]);

  const handleClick = useCallback(() => {
    if (isOpen) {
      onClose();
    } else {
      onOpen();
    }
  }, [isOpen, onOpen, onClose]);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (closeTimer.current) clearTimeout(closeTimer.current);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative flex-1 min-w-0"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Header */}
      <button
        id={`global-bar-${id}`}
        onClick={handleClick}
        className={`
          w-full h-10 px-3 flex items-center gap-2 transition-all duration-150 cursor-pointer
          border-b-2 ${isOpen ? `${accent.bgHover} ${accent.border}` : `${accent.bg} border-transparent hover:${accent.bgHover}`}
        `}
      >
        <span className={`flex-shrink-0 transition-colors duration-150 ${isOpen ? accent.iconColor : 'text-zinc-500'}`}>
          {icon}
        </span>
        <span className={`text-[11px] font-semibold uppercase tracking-wider flex-shrink-0 hidden xl:inline transition-colors duration-150 ${
          isOpen ? 'text-zinc-200' : 'text-zinc-400'
        }`}>
          {label}
        </span>
        {/* Optional inline toggle */}
        {headerToggle && (
          <div className="flex-shrink-0" onClick={e => e.stopPropagation()}>
            {headerToggle}
          </div>
        )}
        <div className="flex-1 min-w-0 flex justify-end">
          {badge}
        </div>
      </button>

      {/* Dropdown — matches section width */}
      {isOpen && (
        <div
          className="absolute top-full left-0 z-50 w-full min-w-[300px] max-h-[calc(100vh-120px)] overflow-y-auto
                     bg-zinc-900 border border-white/10 border-t-0 rounded-b-xl shadow-2xl shadow-black/60
                     global-bar-dropdown-enter hide-scrollbar"
        >
          <div className="p-4 space-y-3">
            {children}
          </div>
        </div>
      )}
    </div>
  );
};

// ── Inline Toggle Switch ─────────────────────────────────────────────────────

interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  accentColor?: 'pink' | 'emerald' | 'sky' | 'purple' | 'amber';
}

const TOGGLE_COLORS: Record<string, string> = {
  pink: 'bg-pink-500',
  emerald: 'bg-emerald-500',
  sky: 'bg-sky-500',
  purple: 'bg-purple-500',
  amber: 'bg-amber-500',
};

export const ToggleSwitch: React.FC<ToggleSwitchProps> = ({ checked, onChange, accentColor = 'pink' }) => {
  const activeColor = TOGGLE_COLORS[accentColor] || TOGGLE_COLORS.pink;
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={(e) => {
        e.stopPropagation();
        onChange(!checked);
      }}
      className={`
        relative inline-flex h-4 w-8 items-center rounded-full transition-colors duration-200 flex-shrink-0
        ${checked ? activeColor : 'bg-zinc-700'}
      `}
    >
      <span
        className={`
          inline-block h-3 w-3 rounded-full bg-white shadow-sm transform transition-transform duration-200
          ${checked ? 'translate-x-[17px]' : 'translate-x-[3px]'}
        `}
      />
    </button>
  );
};
