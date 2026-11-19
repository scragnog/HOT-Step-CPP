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

/** Between two clusters in a row, so the groups read as groups and not as one run. */
export const DIVIDER = 'mx-0.5 h-4 w-px shrink-0 bg-zinc-200 dark:bg-white/10';

/**
 * One editor row: a scope label in a fixed left gutter, that scope's controls
 * to its right. Every row uses this shape, so the controls line up down the
 * whole editor instead of each row inventing its own start.
 */
export const ROW = 'flex items-start gap-2 border-t border-zinc-200 px-3 py-2 text-[11px] '
  + 'text-zinc-600 dark:border-white/10 dark:text-zinc-300';

/** The scope's name in the gutter. Short words only: the column is 48px. */
export const ROW_LABEL = 'w-12 shrink-0 pt-[3px] text-right text-[9px] font-semibold uppercase '
  + 'tracking-wider text-zinc-400 dark:text-zinc-500';

/** The controls of one row. Clusters are divided; the body does the wrapping. */
export const ROW_BODY = 'flex min-w-0 flex-1 flex-wrap items-center gap-1.5';
