// The one drawn icon set (the concept's ICON_D, 00-util.js): 24-unit grid, 1.75 stroke, currentColor.

const D = {
  check: <path d="M5 12.5l4.5 4.5L19 7.5" />,
  x: <path d="M6 6l12 12M18 6L6 18" />,
  grip: <g fill="currentColor" stroke="none"><circle cx="9" cy="6" r="1.5" /><circle cx="15" cy="6" r="1.5" /><circle cx="9" cy="12" r="1.5" /><circle cx="15" cy="12" r="1.5" /><circle cx="9" cy="18" r="1.5" /><circle cx="15" cy="18" r="1.5" /></g>,
  plus: <path d="M12 5v14M5 12h14" />,
  chevR: <path d="M9.5 6l6 6-6 6" />,
  chevL: <path d="M14.5 6l-6 6 6 6" />,
  caret: <path d="M7 10l5 5 5-5" />,
  /* the Workflows panel's and the Drafts list's (package 7) */
  info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6" /><path d="M12 7.5v.01" /></>,
  pencil: <><path d="M4 20h4L19 9l-4-4L4 16v4z" /><path d="M13.5 6.5l4 4" /></>,
  download: <><path d="M12 4v11" /><path d="M7 10.5l5 5 5-5" /><path d="M5 20h14" /></>,
  upload: <><path d="M12 16V5" /><path d="M7 9.5l5-5 5 5" /><path d="M5 20h14" /></>,
  trash: <><path d="M4.5 7h15" /><path d="M9.5 7V4.5h5V7" /><path d="M6.5 7l1 13h9l1-13" /></>,
  undo: <><path d="M9 7L4.5 11.5 9 16" /><path d="M4.5 11.5H14a5.5 5.5 0 0 1 0 11h-2" /></>,
  more: <><circle cx="5.5" cy="12" r="1.2" /><circle cx="12" cy="12" r="1.2" /><circle cx="18.5" cy="12" r="1.2" /></>,
  saveIn: <><path d="M4 14v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" /><path d="M8 9.5l3 3 5.5-6" /></>,
  pagePlus: <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /><path d="M12 11v6M9 14h6" /></>,
  openIn: <><path d="M14 5h5v5" /><path d="M19 5l-8 8" /><path d="M17 14v4a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1h4" /></>,
  lines: <path d="M5 7h14M5 12h14M5 17h8" />,
  note: <><path d="M9 18V6l10-2v12" /><circle cx="6.5" cy="18" r="2.5" /><circle cx="16.5" cy="16" r="2.5" /></>,
} as const;

export type IconName = keyof typeof D;

export function Icon({ name, size = 14 }: { name: IconName; size?: number }) {
  return (
    <svg className="ico" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {D[name]}
    </svg>
  );
}
