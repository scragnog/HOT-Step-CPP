import { useEffect, useRef } from 'react';

interface Props {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  /** The character range to highlight and its colour, or null for none. */
  highlight: { start: number; end: number; colour: string } | null;
}

// Text, padding, border width and scrollbar gutter shared by the textarea and
// the layer behind it, so both wrap the lyrics onto the same lines.
const TEXT = 'absolute inset-0 w-full h-full rounded-xl border px-4 py-3 text-sm font-mono leading-relaxed whitespace-pre-wrap break-words [scrollbar-gutter:stable]';

/** The lyrics textarea with one block highlighted behind its text. A textarea
 *  cannot colour part of its own text, so a copy of the lyrics sits behind a
 *  transparent textarea with the block in a <mark>, scrolled in step with it.
 *  A new highlight scrolls its block into view if it is off screen. */
export function LyricsSectionHighlight({ value, onChange, placeholder, highlight }: Props) {
  const textRef = useRef<HTMLTextAreaElement>(null);
  const layerRef = useRef<HTMLDivElement>(null);
  const markRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const text = textRef.current, mark = markRef.current;
    if (!text || !mark) return;
    const top = mark.offsetTop, bottom = top + mark.offsetHeight;
    if (top < text.scrollTop || bottom > text.scrollTop + text.clientHeight)
      text.scrollTop = Math.max(0, top - 12);
    if (layerRef.current) layerRef.current.scrollTop = text.scrollTop;
  }, [highlight?.start, highlight?.end]);

  const range = highlight && highlight.end > highlight.start ? highlight : null;
  return (
    <div className="relative w-full min-h-0 flex-1 rounded-xl bg-white dark:bg-black/20">
      <div ref={layerRef} aria-hidden className={`${TEXT} overflow-hidden border-transparent text-transparent`}>
        {range ? <>
          {value.slice(0, range.start)}
          <mark ref={markRef} className="rounded text-transparent transition-colors" style={{ backgroundColor: `${range.colour}33` }}>
            {value.slice(range.start, range.end)}
          </mark>
          {value.slice(range.end)}
        </> : value}
        {/* A trailing newline needs a character after it to take up a line. */}
        {'​'}
      </div>
      <textarea ref={textRef} value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
        onScroll={e => { if (layerRef.current) layerRef.current.scrollTop = e.currentTarget.scrollTop; }}
        className={`${TEXT} resize-none overflow-y-auto bg-transparent border-zinc-200 dark:border-white/10 text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-cyan-500 transition-colors`} />
    </div>
  );
}
