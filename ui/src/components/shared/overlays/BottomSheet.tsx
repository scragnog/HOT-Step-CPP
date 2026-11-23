// BottomSheet.tsx: a mobile bottom sheet (portal, backdrop, slide up,
// drag handle, swipe down to dismiss, focus trap, Esc closes).
//
// Tagged data-portal-layer so BarSection's outside-click handler ignores it.

import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom';
import { useFocusTrap } from '@/hooks/useFocusTrap';

/** True while the media query matches; follows changes. */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window !== 'undefined' && !!window.matchMedia && window.matchMedia(query).matches);
  useEffect(() => {
    if (!window.matchMedia) return;
    const mq = window.matchMedia(query);
    const on = () => setMatches(mq.matches);
    on();
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, [query]);
  return matches;
}

/** Phone layout: matches the app's max-[1023px] breakpoint. */
export const useIsMobile = () => useMediaQuery('(max-width: 1023px)');

const DISMISS_PX = 80;

interface BottomSheetProps {
  open: boolean;
  onClose: () => void;
  title?: React.ReactNode;
  ariaLabel?: string;
  children: React.ReactNode;
}

export const BottomSheet: React.FC<BottomSheetProps> = ({ open, onClose, title, ariaLabel, children }) => {
  const sheetRef = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);
  const [dragY, setDragY] = useState(0);
  const drag = useRef<{ y: number; id: number } | null>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  // Child dialogs and dropdowns portal outside the sheet. Let the higher
  // layer own keyboard focus and dismissal until it has closed.
  const childLayerOpen = () => {
    const sheet = sheetRef.current;
    if (!sheet) return false;
    const z = Number(getComputedStyle(sheet).zIndex);
    return Array.from(document.querySelectorAll<HTMLElement>('[data-portal-layer], [role="dialog"]'))
      .some(layer => !sheet.contains(layer) && !layer.contains(sheet)
        && Number(getComputedStyle(layer).zIndex) > z && layer.getClientRects().length > 0);
  };
  const childLayerOpenRef = useRef(childLayerOpen);
  childLayerOpenRef.current = childLayerOpen;

  useLayoutEffect(() => {
    if (!open) { setShown(false); setDragY(0); return; }
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, [open]);

  useFocusTrap(sheetRef, open, {
    onEscape: () => onCloseRef.current(),
    consumeEscape: true,
    shouldIgnore: () => childLayerOpenRef.current(),
  });

  if (!open) return null;

  const onHandleDown = (e: React.PointerEvent) => {
    drag.current = { y: e.clientY, id: e.pointerId };
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
  };
  const onHandleMove = (e: React.PointerEvent) => {
    if (!drag.current || drag.current.id !== e.pointerId) return;
    setDragY(Math.max(0, e.clientY - drag.current.y));
  };
  const onHandleUp = () => {
    if (!drag.current) return;
    drag.current = null;
    if (dragY > DISMISS_PX) onClose();
    else setDragY(0);
  };

  return ReactDOM.createPortal(
    <>
      <div
        data-portal-layer
        className={`fixed inset-0 z-[59] bg-black/50 transition-opacity duration-200 ${shown ? 'opacity-100' : 'opacity-0'}`}
        onClick={(e) => { e.stopPropagation(); if (!childLayerOpen()) onClose(); }}
      />
      <div
        data-portal-layer
        ref={sheetRef}
        role="dialog"
        aria-modal="true"
        aria-label={ariaLabel}
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
        className="fixed inset-x-0 bottom-0 z-[60] max-h-[85vh] flex flex-col rounded-t-2xl bg-white dark:bg-zinc-900 border-t border-zinc-300 dark:border-white/10 shadow-2xl outline-none"
        style={{
          transform: shown ? `translateY(${dragY}px)` : 'translateY(100%)',
          transition: drag.current ? 'none' : 'transform 200ms ease-out',
          paddingBottom: 'env(safe-area-inset-bottom)',
        }}
      >
        <div
          className="flex justify-center py-3 cursor-grab touch-none"
          onPointerDown={onHandleDown}
          onPointerMove={onHandleMove}
          onPointerUp={onHandleUp}
          onPointerCancel={onHandleUp}
        >
          <div className="w-10 h-1.5 rounded-full bg-zinc-400 dark:bg-zinc-600" />
        </div>
        {title && <div className="px-4 pb-2 text-sm font-semibold text-zinc-800 dark:text-zinc-200 truncate">{title}</div>}
        <div className="overflow-y-auto pb-2">{children}</div>
      </div>
    </>,
    document.body,
  );
};

export default BottomSheet;
