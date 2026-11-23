// The one drawn icon set (the concept's ICON_D, 00-util.js): 24-unit grid, 1.75 stroke, currentColor.

const D = {
  check: <path d="M5 12.5l4.5 4.5L19 7.5" />,
  x: <path d="M6 6l12 12M18 6L6 18" />,
  grip: <g fill="currentColor" stroke="none"><circle cx="9" cy="6" r="1.5" /><circle cx="15" cy="6" r="1.5" /><circle cx="9" cy="12" r="1.5" /><circle cx="15" cy="12" r="1.5" /><circle cx="9" cy="18" r="1.5" /><circle cx="15" cy="18" r="1.5" /></g>,
  plus: <path d="M12 5v14M5 12h14" />,
  chevR: <path d="M9.5 6l6 6-6 6" />,
  chevL: <path d="M14.5 6l-6 6 6 6" />,
  caret: <path d="M7 10l5 5 5-5" />,
} as const;

export type IconName = keyof typeof D;

export function Icon({ name, size = 14 }: { name: IconName; size?: number }) {
  return (
    <svg className="ico" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {D[name]}
    </svg>
  );
}
