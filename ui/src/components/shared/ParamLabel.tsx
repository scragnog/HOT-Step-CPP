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
// an absolutely-positioned card.
//
// WHERE the card lands depends on whether an ancestor is marked
// data-hovercard-boundary. Without one — the Training Studio's drawers — it
// sits under the label and flips above near the bottom of the viewport. With
// one, it flanks that element instead: pinned just outside its left or right
// edge, vertically level with the label. The global bar's dropdown panels mark
// themselves, because a card anchored under a label there covers the very
// knobs you opened the panel to reach, and you have to dismiss it to get at
// them.
//
// Flanking picks the side with room for a full-width card, preferring the one
// with more space when neither fits outright; if the boundary is centred in a
// narrow window with no room either side, it gives up and falls back to the
// under-the-label placement rather than hanging the card off-screen.
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
const GAP = 10;      // breathing room between the card and whatever it flanks
const MIN_CARD_H = 120;

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
  /** Set when flanking a boundary: the card is top-anchored and scrolls
   *  internally rather than growing past the bottom of the window. */
  maxHeight?: number;
}

// How many hover cards are open anywhere on the page.
//
// A flanking card is portalled to <body> and lands outside the panel it
// describes, so reaching for it fires that panel's mouseleave — and the global
// bar's panels close themselves ~400ms after the pointer leaves. The panel
// would take the card down with it on the way to being read. A hover-close
// container asks here first and defers while a card is up, the same way it
// already defers while the window has lost focus.
let openCardCount = 0;
export function hoverCardIsOpen(): boolean {
  return openCardCount > 0;
}

/** The nearest ancestor the card should sit beside instead of on top of. */
function findBoundary(el: HTMLElement | null): HTMLElement | null {
  return el?.closest<HTMLElement>('[data-hovercard-boundary]') ?? null;
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
    // A card that flanks the panel is a long way from its label — the pointer
    // has the whole panel to cross before it lands. 140ms was enough when the
    // card sat 6px below the label and is not enough now.
    hideTimer.current = setTimeout(() => setOpen(false), 320);
  };
  useEffect(() => () => cancelHide(), []);

  const measure = useCallback(() => {
    const el = anchorRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();

    // Flank the panel, if this label lives in one.
    const boundary = findBoundary(el);
    if (boundary) {
      const b = boundary.getBoundingClientRect();
      const roomRight = window.innerWidth - b.right - GAP * 2;
      const roomLeft = b.left - GAP * 2;
      const side = roomRight >= CARD_W ? 'right'
        : roomLeft >= CARD_W ? 'left'
        : roomRight >= roomLeft ? 'right' : 'left';
      const room = side === 'right' ? roomRight : roomLeft;

      if (room >= CARD_W) {
        const left = side === 'right' ? b.right + GAP : b.left - GAP - CARD_W;
        // Level with the label, but never so low that the card has no height
        // left to open into.
        const top = Math.max(GAP, Math.min(r.top - 4, window.innerHeight - MIN_CARD_H - GAP));
        setPos({ left, top, maxHeight: window.innerHeight - top - GAP });
        return;
      }
      // No room either side — fall through to the under-the-label placement
      // rather than parking the card half off-screen.
    }

    const below = window.innerHeight - r.bottom;
    // 220 is a comfortable card height; below that, put it above the label.
    const flip = below < 220 && r.top > below;
    const left = Math.max(8, Math.min(r.left, window.innerWidth - CARD_W - 8));
    setPos(flip
      ? { left, bottom: window.innerHeight - r.top + 6, maxHeight: r.top - 6 - GAP }
      : { left, top: r.bottom + 6, maxHeight: below - 6 - GAP });
  }, []);

  // Publish this card's open state for hover-close containers to consult, and
  // make sure a card that unmounts while open (its panel closed underneath it)
  // still gives its count back.
  useEffect(() => {
    if (!open) return;
    openCardCount += 1;
    return () => { openCardCount -= 1; };
  }, [open]);

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
            ...(pos.maxHeight !== undefined
              ? { maxHeight: pos.maxHeight, overflowY: 'auto' as const }
              : {}),
          }}
        >
          <div className="text-xs font-bold text-zinc-800 dark:text-zinc-100">{label}</div>
          {meta && (
            <div className="mt-0.5 text-[10px] font-mono text-amber-600 dark:text-amber-500/90">{meta}</div>
          )}
          <p className="mt-1.5 text-[11px] leading-relaxed text-zinc-600 dark:text-zinc-300">
            {info}
          </p>
        </div>,
        document.body,
      )}
    </span>
  );
};

export default ParamLabel;
