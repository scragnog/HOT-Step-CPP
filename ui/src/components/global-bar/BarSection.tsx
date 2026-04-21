// BarSection.tsx — Reusable hover-to-expand section for the global param bar
//
// Shows a compact header with label + summary badge.
// On hover (or click), expands a floating dropdown panel below.

import React, { useState, useRef, useCallback, useEffect } from 'react';

interface BarSectionProps {
  id: string;
  label: string;
  icon: React.ReactNode;
  badge: React.ReactNode;
  accentColor?: string;      // Tailwind color for active state (e.g., 'pink')
  children: React.ReactNode;
  isOpen: boolean;
  onOpen: () => void;
  onClose: () => void;
}

const HOVER_CLOSE_DELAY = 200; // ms

export const BarSection: React.FC<BarSectionProps> = ({
  id, label, icon, badge, accentColor = 'pink', children,
  isOpen, onOpen, onClose,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

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

  // Accent color mapping for the active indicator
  const accentBg = isOpen ? `bg-${accentColor}-500/10` : '';
  const accentBorder = isOpen ? `border-b-2 border-${accentColor}-500` : 'border-b-2 border-transparent';

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
          hover:bg-white/5 ${accentBg} ${accentBorder}
        `}
      >
        <span className="text-zinc-500 flex-shrink-0">{icon}</span>
        <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider flex-shrink-0 hidden xl:inline">
          {label}
        </span>
        <div className="flex-1 min-w-0 flex justify-end">
          {badge}
        </div>
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div
          className="absolute top-full left-0 z-50 min-w-[340px] max-w-[480px] max-h-[calc(100vh-120px)] overflow-y-auto
                     bg-zinc-900/98 border border-white/10 border-t-0 rounded-b-xl shadow-2xl shadow-black/40
                     global-bar-dropdown-enter hide-scrollbar"
          style={{ backdropFilter: 'blur(20px)' }}
        >
          <div className="p-4 space-y-3">
            {children}
          </div>
        </div>
      )}
    </div>
  );
};
