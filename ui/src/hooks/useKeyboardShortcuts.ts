// useKeyboardShortcuts.ts — one window listener for a list of key bindings.
//
// Every dialog that owns keys was writing the same preamble: ignore keys typed
// into a field, compare modifiers, consume what it handled, and let nobody else
// see it. That preamble does not depend on what the keys mean, so it lives here
// and a caller declares only the meanings:
//
//   useKeyboardShortcuts([
//     { key: 'Escape', allowInField: true, run: close },
//     { key: 'z', ctrl: true, shift: false, run: undo },
//     { key: 'z', ctrl: true, shift: true, run: redo },
//     { code: 'Space', run: togglePlay },
//   ]);
//
// The listener is installed once for the component's life and reads the current
// list from a ref, so a binding may close over fresh state without the effect
// re-subscribing, and a list rebuilt on every render costs nothing.
//
// Capture phase by default: a focused button inside a dialog must not turn Space
// into a click of that button before the dialog has had its say.

import { useEffect, useLayoutEffect, useRef } from 'react';

export interface KeyBinding {
  /** `KeyboardEvent.key`, compared case-insensitively. */
  key?: string;
  /** `KeyboardEvent.code`, for keys whose `key` varies by layout or is ' '. */
  code?: string;
  /**
   * Ctrl, or Cmd on macOS. `undefined` accepts either, `true` requires it and
   * `false` requires that it is not held — which is what separates Ctrl+Z from
   * a plain Z.
   */
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  /** Run only while this says so. False leaves the event unhandled. */
  when?: () => boolean;
  /** Consume the event, so the browser and other listeners do not also act. Default true. */
  consume?: boolean;
  /** False by default: a focused field keeps its own keys. */
  allowInField?: boolean;
  run: () => void;
}

/** The three elements a keystroke is text rather than a shortcut. */
function isTypingTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLInputElement
    || target instanceof HTMLTextAreaElement
    || target instanceof HTMLSelectElement
    || (target instanceof HTMLElement && target.isContentEditable);
}

function matches(e: KeyboardEvent, b: KeyBinding): boolean {
  if (!b.allowInField && isTypingTarget(e.target)) return false;
  if (b.ctrl !== undefined && b.ctrl !== (e.ctrlKey || e.metaKey)) return false;
  if (b.shift !== undefined && b.shift !== e.shiftKey) return false;
  if (b.alt !== undefined && b.alt !== e.altKey) return false;
  if (b.code !== undefined) return e.code === b.code;
  if (b.key === undefined) return false;
  return e.key.toLowerCase() === b.key.toLowerCase();
}

export interface KeyboardShortcutOptions {
  /** Listen in the capture phase, ahead of the focused element. Default true. */
  capture?: boolean;
  /** Where to listen. Default `window`. */
  target?: 'window' | 'document';
}

/**
 * Install `bindings` for the life of the component. The first binding that
 * matches wins, so order is the precedence: put a more specific key first.
 */
export function useKeyboardShortcuts(
  bindings: KeyBinding[],
  options: KeyboardShortcutOptions = {},
): void {
  const { capture = true, target = 'window' } = options;
  const ref = useRef(bindings);
  useLayoutEffect(() => { ref.current = bindings; }, [bindings]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // A nearer listener already claimed this keystroke.
      if (e.defaultPrevented) return;
      for (const b of ref.current) {
        if (!matches(e, b)) continue;
        if (b.when && !b.when()) continue;
        if (b.consume !== false) {
          e.preventDefault();
          e.stopPropagation();
        }
        b.run();
        return;
      }
    };
    const el: EventTarget = target === 'document' ? document : window;
    el.addEventListener('keydown', onKey as EventListener, capture);
    return () => el.removeEventListener('keydown', onKey as EventListener, capture);
  }, [capture, target]);
}
