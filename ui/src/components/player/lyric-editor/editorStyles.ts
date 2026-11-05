// editorStyles.ts — the button shells every editor row shares.
//
// The rows are one toolbar stack under one modal, so a button has to look the
// same in the header as in the footer. They live here rather than being copied
// into each row because a Tailwind string has no compiler to catch the drift.

/** The default outlined button: a label with an optional icon. */
export const BTN = 'flex items-center gap-1 rounded-md border border-zinc-300 px-1.5 py-0.5 text-[11px] '
  + 'text-zinc-600 hover:bg-black/5 disabled:opacity-40 disabled:pointer-events-none '
  + 'dark:border-white/10 dark:text-zinc-300 dark:hover:bg-white/10';

/** The same button, icon only. */
export const ICON_BTN = 'rounded-md border border-zinc-300 p-1 text-zinc-600 hover:bg-black/5 '
  + 'disabled:opacity-40 disabled:pointer-events-none dark:border-white/10 dark:text-zinc-300 dark:hover:bg-white/10';

/** The floating paging arrows over the timeline, which sit above the audio. */
export const PAN_BTN = 'rounded-full border border-zinc-300 bg-white/85 p-1.5 text-zinc-600 shadow-md '
  + 'backdrop-blur transition-colors hover:bg-white disabled:opacity-30 disabled:pointer-events-none '
  + 'dark:border-white/15 dark:bg-zinc-900/85 dark:text-zinc-200 dark:hover:bg-zinc-900';

/** Save, the one filled button: with Auto save off it is the only write. */
export const SAVE_BTN = 'flex items-center gap-1 rounded-md border border-emerald-600 bg-emerald-600 px-2 py-0.5 '
  + 'text-[11px] font-semibold text-white hover:bg-emerald-500 disabled:pointer-events-none disabled:opacity-40';

/** One editor row: the same top border, padding and type from top to bottom. */
export const ROW = 'flex flex-wrap items-center gap-1.5 border-t border-zinc-200 px-3 py-2 text-[11px] '
  + 'text-zinc-600 dark:border-white/10 dark:text-zinc-300';
