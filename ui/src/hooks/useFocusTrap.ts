import { useEffect, useRef, type RefObject } from 'react';

/** Same selector the dialogs used by hand: only things that can actually take focus. */
const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export interface FocusTrapOptions {
  /** Escape pressed inside the trap. */
  onEscape?: () => void;
  /** preventDefault + stopPropagation on Escape, for a layer that must not leak it. */
  consumeEscape?: boolean;
  /** Skip the trap while a higher layer owns focus (nested dialog or popover). */
  shouldIgnore?: () => boolean;
}

/**
 * Keep keyboard focus inside an open dialog: focus the first control on open,
 * cycle Tab at the edges, and hand focus back to the opener on close.
 *
 * BottomSheet, the mobile playlist overlay and the hotkey sheet each wrote this
 * by hand — three copies of the same Tab math, with different bugs (one never
 * restored focus, one had no trap at all). `shouldIgnore` is for a sheet with a
 * portalled child layer above it.
 */
export function useFocusTrap(
  ref: RefObject<HTMLElement | null>,
  active: boolean,
  options: FocusTrapOptions = {},
): void {
  const optionsRef = useRef(options);
  optionsRef.current = options;

  useEffect(() => {
    if (!active) return;
    const opener = document.activeElement as HTMLElement | null;
    // After the layer paints: a control that is still display:none cannot take focus.
    const focusTimer = setTimeout(() => {
      if (optionsRef.current.shouldIgnore?.()) return;
      const el = ref.current;
      if (!el) return;
      (el.querySelector<HTMLElement>(FOCUSABLE) ?? el).focus();
    }, 0);

    const onKey = (e: KeyboardEvent) => {
      if (optionsRef.current.shouldIgnore?.()) return;
      const el = ref.current;
      if (!el) return;
      if (e.key === 'Escape') {
        if (!optionsRef.current.onEscape) return;
        if (optionsRef.current.consumeEscape) { e.preventDefault(); e.stopPropagation(); }
        optionsRef.current.onEscape();
        return;
      }
      if (e.key !== 'Tab') return;
      const els = Array.from(el.querySelectorAll<HTMLElement>(FOCUSABLE));
      if (els.length === 0) { e.preventDefault(); return; }
      const first = els[0];
      const last = els[els.length - 1];
      const inside = el.contains(document.activeElement);
      if (e.shiftKey && (document.activeElement === first || !inside)) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && (document.activeElement === last || !inside)) { e.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', onKey, true);
    return () => {
      clearTimeout(focusTimer);
      document.removeEventListener('keydown', onKey, true);
      if (opener && document.contains(opener)) opener.focus();
    };
  }, [active, ref]);
}
