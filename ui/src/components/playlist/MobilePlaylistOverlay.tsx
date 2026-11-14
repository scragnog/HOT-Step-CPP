/**
 * MobilePlaylistOverlay.tsx — Full-screen mobile playlist overlay.
 *
 * Wraps PlaylistSidebar with swipe gestures enabled. Accessible from the
 * global bar on mobile. Player bar remains visible at the bottom.
 * Mutually exclusive with the mobile nav drawer.
 */

import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { X, ListMusic } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { PlaylistSidebar } from './PlaylistSidebar';
import type { PlaylistItem } from '../lyric-studio/playlistStore';

interface MobilePlaylistOverlayProps {
  isOpen: boolean;
  onClose: () => void;
  onSwipeLeft?: (item: PlaylistItem) => void;
  onSwipeRight?: (item: PlaylistItem) => void;
}

/** Approximate height of the mobile player bar at the bottom of the screen.
 *  Includes transport row, lyrics bar, and margin for A/B / post-process docks. */
const PLAYER_BAR_HEIGHT = 160;

export const MobilePlaylistOverlay: React.FC<MobilePlaylistOverlayProps> = ({
  isOpen,
  onClose,
  onSwipeLeft,
  onSwipeRight,
}) => {
  const { t } = useTranslation();
  const overlayRef = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  // a11y: focus in on open, Escape closes, focus returns to the opener on close.
  useEffect(() => {
    if (!isOpen) return;
    const opener = document.activeElement as HTMLElement | null;
    overlayRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCloseRef.current();
    };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      if (opener && document.contains(opener)) opener.focus();
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-50 pointer-events-none"
      role="dialog"
      aria-modal="true"
      aria-label={t('playlist.title')}
    >
      {/* Backdrop — covers everything except the player bar; closes on click */}
      <div
        className="absolute inset-x-0 top-0 bg-black/60 pointer-events-auto"
        style={{ bottom: `${PLAYER_BAR_HEIGHT}px` }}
        onClick={onClose}
        aria-hidden="true"
      />
      {/* Overlay panel — full width, leaves space for the player bar at the bottom */}
      <div
        ref={overlayRef}
        tabIndex={-1}
        className="outline-none absolute inset-x-0 top-0 bg-white dark:bg-suno flex flex-col shadow-2xl shadow-black/40 pointer-events-auto"
        style={{ bottom: `${PLAYER_BAR_HEIGHT}px` }}
      >
        {/* Header with close button */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-white/5 flex-shrink-0">
          <div className="flex items-center gap-2">
            <ListMusic className="w-4 h-4 text-pink-400" />
            <span className="text-xs font-bold text-zinc-800 dark:text-zinc-200 uppercase tracking-wider">
              {t('playlist.title')}
            </span>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg bg-zinc-200 dark:bg-white/5 hover:bg-zinc-300 dark:hover:bg-white/10 flex items-center justify-center transition-all flex-shrink-0"
            aria-label={t('playlist.closePlaylist')}
          >
            <X size={16} className="text-zinc-600 dark:text-zinc-400" />
          </button>
        </div>
        {/* Playlist sidebar with swipe gestures enabled */}
        <div className="flex-1 min-h-0">
          <PlaylistSidebar
            onClose={onClose}
            embedded
            swipeEnabled={true}
            onSwipeLeft={onSwipeLeft}
            onSwipeRight={onSwipeRight}
          />
        </div>
      </div>
    </div>,
    document.body
  );
};
