// BarSection.tsx — Reusable hover-to-expand section for the global param bar
//
// Shows a compact header with label + summary badge.
// On hover (or click), expands a floating dropdown panel below.
// Each section has a unique accent tint that is always visible as its background.
// Optionally shows a toggle switch in the header (for LM / Mastering).

import React, { useRef, useCallback, useEffect } from 'react';
import { hoverCardIsOpen } from '../shared/ParamLabel';

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
  violet:  { bg: 'bg-violet-500/5',  bgHover: 'bg-violet-500/10',  border: 'border-violet-500',  iconColor: 'text-violet-400' },
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

const HOVER_CLOSE_DELAY = 400; // ms

/** True when the browser window itself has focus.
 *
 *  A native OS file picker takes focus away from the page, which fires
 *  mouseleave on whatever the pointer was over. For a hover-close panel that
 *  reads as "the user left", so the panel unmounts, taking the
 *  <input type="file"> with it, and the file the user then picks is delivered
 *  to a detached element that nobody is listening to (issue #117, reported on
 *  Wayland/Firefox but not platform-specific). Suppressing the close while the
 *  window is unfocused keeps the input mounted until the picker returns. */
function windowHasFocus(): boolean {
  try {
    return typeof document !== 'undefined' && document.hasFocus();
  } catch {
    return true; // never let a focus probe be the reason a panel sticks open
  }
}

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

  // Set when the user opened this section by clicking the header rather than
  // by hovering over it. A clicked-open panel stays open until it is clicked
  // shut, so moving the pointer away to a file dialog, a colour picker or a
  // second monitor cannot collapse work in progress (issue #117).
  const pinned = useRef(false);

  const scheduleClose = useCallback(() => {
    cancelClose();
    if (pinned.current) return;
    closeTimer.current = setTimeout(() => {
      // Focus left the page (file picker, alt-tab). Wait rather than close:
      // the mouseleave that got us here was a side effect of losing focus,
      // not the user walking away from the panel.
      if (!windowHasFocus()) {
        scheduleClose();
        return;
      }
      // A ParamLabel help card is up. It is portalled to <body> and floats
      // beside this panel, so the pointer moving to read it counts as leaving
      // — closing now would unmount the card the user is reaching for.
      if (hoverCardIsOpen()) {
        scheduleClose();
        return;
      }
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
    if (isOpen && pinned.current) {
      pinned.current = false;
      cancelClose();
      onClose();
    } else {
      // Covers both "closed, so open it" and "hover-opened, so pin it".
      pinned.current = true;
      cancelClose();
      onOpen();
    }
  }, [isOpen, onOpen, onClose, cancelClose]);

  // The parent owns which section is open, so a different section opening
  // closes this one without any of our handlers running. Drop the pin then,
  // otherwise it survives into the next time this section opens by hover.
  useEffect(() => {
    if (!isOpen) pinned.current = false;
  }, [isOpen]);

  // A pinned panel still has to close on click-outside, or there is no way to
  // dismiss it without going back to the header.
  useEffect(() => {
    if (!isOpen) return;
    const onDocPointerDown = (e: PointerEvent) => {
      if (!containerRef.current) return;
      if (containerRef.current.contains(e.target as Node)) return;
      // A help card belongs to this panel even though it is portalled outside
      // it — clicking into one to select its text is not clicking away.
      if ((e.target as HTMLElement)?.closest?.('[role="tooltip"]')) return;
      pinned.current = false;
      cancelClose();
      onClose();
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      pinned.current = false;
      cancelClose();
      onClose();
    };
    document.addEventListener('pointerdown', onDocPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('pointerdown', onDocPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [isOpen, onClose, cancelClose]);

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
          absolute inset-0 w-full px-3 pt-1 flex items-center gap-2 transition-all duration-150 cursor-pointer
          border-b-2 ${isOpen ? `${accent.bgHover} ${accent.border}` : `${accent.bg} border-transparent hover:${accent.bgHover}`}
        `}
      >
        <span className={`flex-shrink-0 transition-colors duration-150 ${isOpen ? accent.iconColor : 'text-zinc-500'}`}>
          {icon}
        </span>
        <span className={`text-[11px] font-semibold uppercase tracking-wider flex-shrink-0 hidden xl:inline transition-colors duration-150 ${
          isOpen ? 'text-zinc-800 dark:text-zinc-200' : 'text-zinc-600 dark:text-zinc-400'
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

      {/* Dropdown — matches section width.
          data-hovercard-boundary tells ParamLabel to float its help card beside
          this panel rather than under the label that opened it: a card anchored
          to the label lands squarely on the knobs below it, and you have to
          dismiss it before you can reach them. */}
      {isOpen && (
        <div
          data-hovercard-boundary
          className="absolute top-full left-0 z-50 w-full min-w-[300px] max-h-[calc(100vh-120px)] overflow-y-auto
                     bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 border-t-0 rounded-b-xl shadow-2xl shadow-black/30 dark:shadow-black/60
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
  accentColor?: 'pink' | 'emerald' | 'sky' | 'purple' | 'amber' | 'teal';
}

const TOGGLE_COLORS: Record<string, string> = {
  pink: 'bg-pink-500',
  emerald: 'bg-emerald-500',
  sky: 'bg-sky-500',
  purple: 'bg-purple-500',
  amber: 'bg-amber-500',
  teal: 'bg-teal-500',
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
        ${checked ? activeColor : 'bg-zinc-200 dark:bg-zinc-700'}
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
