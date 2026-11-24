// The icons Latest and the player draw (the concept's ICON_D, 00-util.js: play, pause, stop, more, x), in the kit's
// markup (kit/icons.tsx). Kept here so the kit's own list stays the slice's; package 7 adds `more` to the kit too:
// once both are on one branch, these can be the kit's.

const D = {
  play: <path d="M8 5.5v13l10.5-6.5z" fill="currentColor" />,
  pause: <path d="M8.5 5.5v13M15.5 5.5v13" />,
  stop: <rect x="6.5" y="6.5" width="11" height="11" rx="1.5" fill="currentColor" />,
  more: <><circle cx="5.5" cy="12" r="1.2" /><circle cx="12" cy="12" r="1.2" /><circle cx="18.5" cy="12" r="1.2" /></>,
  x: <path d="M6 6l12 12M18 6L6 18" />,
} as const;

export function GIcon({ name, size = 14 }: { name: keyof typeof D; size?: number }) {
  return (
    <svg className="ico" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {D[name]}
    </svg>
  );
}
