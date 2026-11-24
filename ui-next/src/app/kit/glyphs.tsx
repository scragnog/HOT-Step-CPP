// Three more icons of the concept's set (00-util.js ICON_D) for package 3c's instruments: play and stop (a take, the
// recording's chosen part, the score preview) and the padlock on a value a keep locks. Drawn in the one icon frame
// (icons.tsx); kept in a file of their own so that the icon set's list is not edited by two packages at once.

const D = {
  play: <path d="M8 5.5v13l10.5-6.5z" fill="currentColor" />,
  stop: <rect x="6.5" y="6.5" width="11" height="11" rx="1.5" fill="currentColor" />,
  lock: <><rect x="5" y="11" width="14" height="10" rx="2" /><path d="M8 11V7a4 4 0 0 1 8 0v4" /></>,
} as const;

export type GlyphName = keyof typeof D;
export function Glyph({ name, size = 12 }: { name: GlyphName; size?: number }) {
  return (
    <svg className="ico" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {D[name]}
    </svg>
  );
}
