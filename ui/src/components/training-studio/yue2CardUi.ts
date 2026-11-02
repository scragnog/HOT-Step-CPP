// yue2CardUi.ts — the shared card/field/button classes the YuE2 stage cards use.
// These were copy-pasted per card; one home keeps the surfaces consistent.

export const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
export const INPUT = 'w-full px-2.5 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 '
            + 'dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 outline-none '
            + 'focus:border-amber-500/50';
export const BTN_SM = 'shrink-0 px-2.5 py-1.5 rounded-lg text-[11px] font-medium border border-zinc-300 '
             + 'dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 '
             + 'dark:hover:bg-zinc-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors';
export const BTN_GO = 'px-4 py-2 rounded-lg text-sm font-semibold bg-amber-500 text-black hover:bg-amber-400 '
             + 'disabled:opacity-40 transition-colors flex items-center gap-2';
