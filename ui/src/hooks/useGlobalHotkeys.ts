import { useEffect } from 'react';
import { mapKey, shouldIgnore, seekTarget, LIBRARY_ACTIONS, SEEK_STEP_SEC, type HotkeyAction } from '@/utils/hotkeys';
import { togglePlay, seek, getPlaybackState } from '@/stores/playbackStore';

/** Event the mounted SongList listens for to run list actions. */
export const HOTKEY_EVENT = 'hs-hotkey';

function isModalOpen(): boolean {
  return !!document.querySelector('[role="dialog"], [aria-modal="true"], [data-hotkeys-block]');
}

/** Install the desktop global hotkeys. onHelp opens the cheat sheet. */
export function useGlobalHotkeys(onHelp: () => void, onPalette?: () => void): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return;
      const action: HotkeyAction | null = mapKey(e);
      if (!action) return;
      // Ctrl/Cmd+K works while typing too; only an open dialog blocks it.
      if (action === 'openPalette') {
        if (!onPalette || isModalOpen()) return;
        e.preventDefault();
        onPalette();
        return;
      }
      if (shouldIgnore({ target: e.target as HTMLElement | null, modalOpen: isModalOpen(), pathname: window.location.pathname })) return;
      // Let focused buttons and links keep their native Space/Enter behaviour.
      const tag = (e.target as HTMLElement | null)?.tagName;
      if ((action === 'togglePlay' && e.key === ' ' || action === 'playFocused') && (tag === 'BUTTON' || tag === 'A')) return;
      if (action === 'togglePlay') {
        togglePlay();
      } else if (action === 'seekBack' || action === 'seekForward') {
        const st = getPlaybackState();
        if (!st.currentTrack) return;
        seek(seekTarget(st.currentTime, action === 'seekBack' ? -SEEK_STEP_SEC : SEEK_STEP_SEC, st.duration));
      } else if (action === 'showHelp') {
        onHelp();
      } else if (LIBRARY_ACTIONS.has(action)) {
        const ev = new CustomEvent(HOTKEY_EVENT, { detail: { action }, cancelable: true });
        window.dispatchEvent(ev);
        if (!ev.defaultPrevented) return; // nobody handled it: leave native behaviour (scrolling etc.)
      }
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onHelp, onPalette]);
}
