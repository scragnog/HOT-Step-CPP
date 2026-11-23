import type { MouseEvent } from 'react';

/** Drag a size from a handle's mousedown: `sign` -1 grows it as the pointer
 *  moves up or left (a handle above or before the thing it sizes). */
export function startDragResize(event: MouseEvent, axis: 'x' | 'y', size: number,
  setSize: (value: number) => void, min: number, max: number, sign: 1 | -1 = 1) {
  event.preventDefault();
  const start = axis === 'x' ? event.clientX : event.clientY;
  const onMove = (move: globalThis.MouseEvent) =>
    setSize(Math.min(max, Math.max(min, size + sign * ((axis === 'x' ? move.clientX : move.clientY) - start))));
  const onUp = () => {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  };
  document.body.style.cursor = axis === 'x' ? 'col-resize' : 'row-resize';
  document.body.style.userSelect = 'none';
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}
