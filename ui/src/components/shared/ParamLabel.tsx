// ParamLabel.tsx — a form field label that explains itself on hover.
//
// The Training Studio's Advanced drawers are ~30 numeric knobs each, most of
// them named after the thing they scale rather than the thing they do
// ("Timestep bias", "SNR gamma", "Unconditional dropout"). The inline help
// lines under a field are one clause long by necessity — there is no room for
// "what happens if I raise it" next to every box. This puts that sentence in a
// hover card instead: point at the label, get the paragraph.
//
// Portalled and fixed-positioned for the same reason StyledSelect's panel is —
// these live inside <details> drawers and scrolling panes, both of which clip
// an absolutely-positioned card. Flips above the label near the bottom of the
// viewport, and clamps to the horizontal edges.
//
// Keyboard/touch: the icon is focusable and handles Enter/Space, so the card
// also opens on focus and on tap, not hover alone. It is a <span role="button">
// rather than a <button> on purpose — the global bar's accordion headers are
// themselves <button>s, and a nested <button> is invalid HTML that the parser
// reparents out of its ancestor, silently breaking the accordion's click
// target. A span nests anywhere and keeps the same semantics for a screen
// reader. Clicks stopPropagation for the same reason: pointing at "?" inside a
// header must not toggle the section.

import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom';
import { HelpCircle } from 'lucide-react';

const CARD_W = 340;

interface Props {
  /** The field name, shown as the label and as the hover card's heading. */
  label: string;
  /** The explanation. Nothing renders the hover affordance without it. */
  info?: string;
  /** Optional one-liner under the heading — default value, accepted range. */
  meta?: string;
  /** Classes for the label text. Defaults to the studios' field-label style. */
  className?: string;
  /** Layout classes for the wrapper (margins, `block`, alignment). Keep these
   *  off `className`, which lands on the inner text span where a `block` or a
   *  margin has no effect inside the flex row. */
  rootClassName?: string;
  /** Dotted underline hinting the label is explainable. Off for section
   *  headings, where the whole row is already an affordance and the underline
   *  reads as noise under uppercase tracking. */
  underline?: boolean;
}

interface CardPos {
  left: number;
  top?: number;
  bottom?: number;
}

export const ParamLabel: React.FC<Props> = ({
  label,
  info,
  meta,
  className = 'text-xs font-semibold text-zinc-600 dark:text-zinc-400',
  rootClassName = '',
  underline = true,
}) => {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<CardPos | null>(null);
  const anchorRef = useRef<HTMLSpanElement>(null);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cancelHide = () => {
    if (hideTimer.current) {
      clearTimeout(hideTimer.current);
      hideTimer.current = null;
    }
  };
  // A short grace period so the pointer can travel from the label to the card
  // (they are 6px apart) without the card vanishing underneath it.
  const scheduleHide = () => {
    cancelHide();
    hideTimer.current = setTimeout(() => setOpen(false), 140);
  };
  useEffect(() => () => cancelHide(), []);

  const measure = useCallback(() => {
    const el = anchorRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const below = window.innerHeight - r.bottom;
    // 220 is a comfortable card height; below that, put it above the label.
    const flip = below < 220 && r.top > below;
    const left = Math.max(8, Math.min(r.left, window.innerWidth - CARD_W - 8));
    setPos(flip
      ? { left, bottom: window.innerHeight - r.top + 6 }
      : { left, top: r.bottom + 6 });
  }, []);

  useLayoutEffect(() => {
    if (!open) return;
    measure();
    const onScroll = () => measure();
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onScroll);
    };
  }, [open, measure]);

  if (!info) return <span className={`${className} ${rootClassName}`}>{label}</span>;

  const show = () => { cancelHide(); setOpen(true); };

  return (
    <span
      ref={anchorRef}
      className={`inline-flex items-center gap-1 w-fit ${rootClassName}`}
      onMouseEnter={show}
      onMouseLeave={scheduleHide}
    >
      <span className={`${className} cursor-help${underline ? ' decoration-dotted underline underline-offset-[3px] decoration-zinc-400/40' : ''}`}>
        {label}
      </span>
      <span
        role="button"
        tabIndex={0}
        aria-label={`About ${label}`}
        aria-expanded={open}
        onFocus={show}
        onBlur={scheduleHide}
        onClick={(e) => { e.preventDefault(); e.stopPropagation(); setOpen(o => !o); }}
        onKeyDown={(e) => {
          if (e.key !== 'Enter' && e.key !== ' ') return;
          e.preventDefault();
          e.stopPropagation();
          setOpen(o => !o);
        }}
        className="shrink-0 inline-flex text-zinc-400/70 hover:text-amber-500 focus:text-amber-500 outline-none transition-colors cursor-help"
      >
        <HelpCircle size={12} />
      </span>

      {open && pos && ReactDOM.createPortal(
        <div
          role="tooltip"
          onMouseEnter={cancelHide}
          onMouseLeave={scheduleHide}
          className="fixed z-[10000] rounded-xl p-3
                     bg-white dark:bg-zinc-900
                     border border-zinc-200 dark:border-white/10
                     shadow-xl shadow-black/25"
          style={{
            left: pos.left,
            width: CARD_W,
            ...(pos.top !== undefined ? { top: pos.top } : { bottom: pos.bottom }),
          }}
        >
          <div className="text-xs font-bold text-zinc-800 dark:text-zinc-100">{label}</div>
          {meta && (
            <div className="mt-0.5 text-[10px] font-mono text-amber-600 dark:text-amber-500/90">{meta}</div>
          )}
          <p className="mt-1.5 text-[11px] leading-relaxed text-zinc-600 dark:text-zinc-300 max-h-72 overflow-y-auto">
            {info}
          </p>
        </div>,
        document.body,
      )}
    </span>
  );
};

export default ParamLabel;
