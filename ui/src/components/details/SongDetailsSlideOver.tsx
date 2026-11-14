// SongDetailsSlideOver.tsx — the Song Details column, sliding in from the right.
//
// The outer box animates its width between 0 and the panel's width, so the
// centre column beside it (flex-1) reflows smoothly instead of jumping. The
// panel inside is pinned to the right edge and slides in with it, so it reads
// as coming out from under the column to its right. The panel stays mounted
// until the close animation ends, then unmounts so a hidden panel does not
// keep fetching details for every track that plays.

import React, { useState } from 'react';

const HANDLE_W = 6;
const EASE = '300ms cubic-bezier(0.4, 0, 0.2, 1)';

interface SongDetailsSlideOverProps {
  open: boolean;
  width: number;
  onResize: (width: number) => void;
  /** Hide entirely on narrow and portrait layouts (Create's stacked view). */
  hideNarrow?: boolean;
  children: React.ReactNode;
}

export const SongDetailsSlideOver: React.FC<SongDetailsSlideOverProps> = ({
  open, width, onResize, hideNarrow, children,
}) => {
  const [mounted, setMounted] = useState(open);
  const [dragging, setDragging] = useState(false);
  // Mount on the render that opens it, so the slide starts from closed.
  if (open && !mounted) setMounted(true);

  // A drag must follow the pointer, not ease after it.
  const transition = dragging ? 'none' : `width ${EASE}, transform ${EASE}`;
  const full = width + HANDLE_W;

  const startResize = (e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = width;
    setDragging(true);
    const onMove = (ev: MouseEvent) => onResize(Math.min(600, Math.max(280, startW + startX - ev.clientX)));
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      setDragging(false);
    };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  };

  return (
    <div
      className={`relative flex-shrink-0 h-full overflow-hidden max-xl:max-w-[calc(26vw+6px)] ${hideNarrow ? 'max-[1023px]:hidden portrait-wide:hidden' : ''}`}
      style={{ width: open ? full : 0, transition }}
      onTransitionEnd={(e) => { if (e.target === e.currentTarget && !open) setMounted(false); }}
      aria-hidden={!open}
    >
      {mounted && (
        <div
          className="absolute top-0 right-0 h-full flex max-xl:max-w-[calc(26vw+6px)]"
          style={{ width: full, transform: open ? 'translateX(0)' : 'translateX(100%)', transition }}
        >
          <div
            className="flex-shrink-0 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors"
            style={{ width: HANDLE_W }}
            onMouseDown={startResize}
          >
            <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-pink-400 transition-colors" />
          </div>
          <div className="flex-1 min-w-0 h-full">{children}</div>
        </div>
      )}
    </div>
  );
};
