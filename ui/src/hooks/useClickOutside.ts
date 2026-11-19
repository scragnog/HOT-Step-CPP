import { useEffect, useRef, type RefObject } from 'react';

export type OutsidePress = 'mousedown' | 'pointerdown';

/**
 * Call `onOutside` for a press that lands outside every ref in `refs`, while
 * `enabled`. The refs' live `.current` is read at event time, so a panel that
 * renders its contents a frame later is still recognised as inside.
 *
 * Nine popovers each subscribed their own mousedown listener for this, and they
 * disagreed about whether the trigger, the portalled panel, or both counted as
 * inside — several closed when the user clicked the thing they had just opened.
 */
export function useClickOutside(
  refs: ReadonlyArray<RefObject<HTMLElement | null>>,
  onOutside: () => void,
  enabled = true,
  event: OutsidePress = 'mousedown',
): void {
  const handler = useRef(onOutside);
  handler.current = onOutside;
  const refList = useRef(refs);
  refList.current = refs;

  useEffect(() => {
    if (!enabled) return;
    const onDown = (e: Event) => {
      const target = e.target as Node | null;
      if (!target) return;
      for (const ref of refList.current) {
        if (ref.current?.contains(target)) return;
      }
      handler.current();
    };
    document.addEventListener(event, onDown);
    return () => document.removeEventListener(event, onDown);
  }, [enabled, event]);
}
