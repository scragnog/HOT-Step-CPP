// HoverFullText.tsx — truncated text whose full content appears in a hover
// tooltip with a copy-to-clipboard button. Truncation style is controlled by
// the caller's `className` (e.g. "truncate" for one line, "line-clamp-2" for
// two). Used by the library table Style cell and the Song Details panel.

import React, { useRef, useState, useEffect } from 'react';
import ReactDOM from 'react-dom';
import { Copy, Check } from 'lucide-react';

interface HoverFullTextProps {
  text: string;
  /** Classes for the trigger element — controls truncation/appearance. */
  className?: string;
  /** Trigger tag. Default 'span'. */
  as?: 'span' | 'p' | 'div';
}

export const HoverFullText: React.FC<HoverFullTextProps> = ({ text, className = '', as = 'span' }) => {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number; width: number } | null>(null);
  const ref = useRef<HTMLElement>(null);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const hasText = !!text && text.trim() !== '' && text !== '—';

  const cancelHide = () => { if (hideTimer.current) { clearTimeout(hideTimer.current); hideTimer.current = null; } };
  const scheduleHide = () => { cancelHide(); hideTimer.current = setTimeout(() => { setOpen(false); setCopied(false); }, 140); };
  const show = () => {
    cancelHide();
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setPos({ left: r.left, top: r.bottom + 4, width: r.width });
    setOpen(true);
  };
  useEffect(() => () => cancelHide(), []);

  const copy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(true);
      cancelHide();
      hideTimer.current = setTimeout(() => { setOpen(false); setCopied(false); }, 1200);
    }).catch(() => {});
  };

  const Tag: React.ElementType = as;

  return (
    <>
      <Tag
        ref={ref as React.Ref<any>}
        className={className}
        onMouseEnter={hasText ? show : undefined}
        onMouseLeave={hasText ? scheduleHide : undefined}
      >
        {text}
      </Tag>
      {open && hasText && pos && ReactDOM.createPortal(
        <div
          className="fixed z-[9999] p-2.5 rounded-lg bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-white/10 shadow-xl text-xs text-zinc-700 dark:text-zinc-200"
          style={{ left: pos.left, top: pos.top, maxWidth: 420, minWidth: Math.min(Math.max(pos.width, 200), 420) }}
          onMouseEnter={cancelHide}
          onMouseLeave={scheduleHide}
          onClick={e => e.stopPropagation()}
        >
          <div className="flex items-start gap-2">
            <span className="whitespace-pre-wrap break-words leading-relaxed flex-1 max-h-60 overflow-y-auto">{text}</span>
            <button
              onClick={copy}
              className="flex-shrink-0 p-1 rounded text-zinc-500 hover:text-zinc-800 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
              title="Copy to clipboard"
            >
              {copied ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} />}
            </button>
          </div>
        </div>,
        document.body,
      )}
    </>
  );
};
