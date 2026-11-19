import type { MouseEvent } from 'react';

interface Props {
  label: string;
  direction: 'horizontal' | 'vertical';
  onMouseDown: (event: MouseEvent<HTMLDivElement>) => void;
}

// Matches the app's permanent right column (Activity: playlist/queue/terminal)
// resize handle exactly — same classes, same grip, unchanged down to the pink
// hover/active colour. See App.tsx's Activity column handle (lines ~1443-1465)
// and ActivitySidebar.tsx's section handles for the canonical pattern. Do not
// swap in a Cover Studio accent colour here — Lead's spec is an exact match,
// idle look, hover, cursor and hit area included.
export function CoverResizeHandle({ label, direction, onMouseDown }: Props) {
  const horizontal = direction === 'horizontal';
  return (
    <div role="separator" aria-label={label}
      aria-orientation={horizontal ? 'vertical' : 'horizontal'}
      onMouseDown={onMouseDown}
      className={horizontal
        ? 'flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors'
        : 'flex-shrink-0 h-1.5 w-full cursor-row-resize group z-20 flex justify-center items-center hover:bg-pink-500/20 active:bg-pink-500/30 rounded transition-colors'}
    >
      <div className={horizontal
        ? 'w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-pink-400 transition-colors'
        : 'h-0.5 w-8 rounded-full bg-zinc-600 group-hover:bg-pink-400 transition-colors'} />
    </div>
  );
}
