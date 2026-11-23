// The page's keyboard shortcuts, registered by the features that own them (the concept's onKey, 02-events.js). The
// page listens once (App.tsx) and asks each binding in order; the first whose `when` is true runs, and no other. A
// feature adds a shortcut by registering it in its own file (src/app/features/); none edits another's.
//
//   onKey({ id: 'undo', when: e => isMod(e, 'z'), run: e => { ... } })
//
// `order` decides who is asked first where two could answer the same key (lower first; 50 by default): a text field's
// own Ctrl+Z (the lyrics package) registers before the page's Undo with a lower order and declines (`when` false)
// once the field has nothing of its own left to undo.

export interface KeyBinding {
  /** A name, unique on the page: two features never bind under one name. */
  id: string;
  when(e: KeyboardEvent): boolean;
  run(e: KeyboardEvent): void;
  order?: number;
}

const KEYS: KeyBinding[] = [];

export function onKey(b: KeyBinding): void {
  if (KEYS.some(k => k.id === b.id)) throw new Error(`the key binding "${b.id}" is registered twice`);
  KEYS.push(b);
  KEYS.sort((a, c) => (a.order ?? 50) - (c.order ?? 50));
}

/** Asks each binding in order; the first that answers runs. */
export function dispatchKey(e: KeyboardEvent): void {
  for (const k of KEYS) if (k.when(e)) { k.run(e); return; }
}

/** Ctrl (or Cmd) and a key, without Alt; `shift` says whether Shift must be held (undefined: either). */
export const isMod = (e: KeyboardEvent, key: string, shift?: boolean): boolean =>
  (e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === key && (shift === undefined || e.shiftKey === shift);

/** A field that takes typing: its own keys (Ctrl+Z, arrows) are its own while it has focus. */
export const isTextField = (el: Element | null): boolean => !!el && (el.tagName === 'TEXTAREA'
  || (el.tagName === 'INPUT' && !/^(checkbox|radio|button|submit|range|file|color)$/i.test((el as HTMLInputElement).type))
  || (el as HTMLElement).isContentEditable);
