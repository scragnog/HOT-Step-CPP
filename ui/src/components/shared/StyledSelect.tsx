// StyledSelect.tsx — the app's dropdown look, for plain value lists.
//
// The global param bar's model pickers (ModelSelect) established the house
// dropdown style: a rounded-xl trigger with a rotating chevron, and a floating
// panel with a check on the selected row. That component is model-specific —
// it hardcodes GGUF/SafeTensors format badges and a "Filter models…" box — so
// panels that just need a styled list (the Training Studio, mostly) were left
// on native <select>, which renders as an OS widget and looks nothing like the
// rest of the app. This is the same look, generalised.
//
// Two differences from ModelSelect, both deliberate:
//
//   * The panel is PORTALLED and fixed-positioned, not absolutely positioned
//     inside the trigger's parent. Training forms live inside <details> drawers
//     and scrolling panes, and an absolute panel gets clipped by both. It also
//     flips above the trigger when there is no room below.
//   * `accent` picks the highlight colour, because the param bar is pink and
//     the Training Studio is amber.

import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom';
import { Check, ChevronDown, Search } from 'lucide-react';

export interface SelectOption<T extends string | number> {
  value: T;
  label: string;
  /** Second line in the panel row. Never shown on the trigger. */
  hint?: string;
  disabled?: boolean;
}

type Accent = 'pink' | 'amber';

/** Tailwind needs whole class names in the source, so these cannot be built by
 *  interpolating the accent into a template string. */
const ACCENT: Record<Accent, { focus: string; row: string; text: string; icon: string }> = {
  pink: {
    focus: 'focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20',
    row: 'bg-pink-500/10 dark:bg-pink-500/15',
    text: 'text-pink-400',
    icon: 'text-pink-400',
  },
  amber: {
    focus: 'focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20',
    row: 'bg-amber-500/10 dark:bg-amber-500/15',
    text: 'text-amber-500',
    icon: 'text-amber-500',
  },
};

const SIZE = {
  sm: { trigger: 'px-2.5 py-1.5 text-xs', row: 'px-2.5 py-1.5 text-xs', chevron: 13 },
  md: { trigger: 'px-3 py-2 text-sm', row: 'px-3 py-2 text-sm', chevron: 14 },
} as const;

interface Props<T extends string | number> {
  value: T;
  onChange: (value: T) => void;
  options: Array<SelectOption<T>>;
  disabled?: boolean;
  /** Shown when `value` matches no option (e.g. an empty selection). */
  placeholder?: string;
  accent?: Accent;
  size?: keyof typeof SIZE;
  /** Extra classes on the trigger button — width, alignment, error borders. */
  className?: string;
  /** Force the filter box on or off. Default: on from 8 options up. */
  searchable?: boolean;
  searchPlaceholder?: string;
  emptyLabel?: string;
  id?: string;
  title?: string;
  'aria-label'?: string;
}

/** Where the floating panel goes, in viewport coordinates. */
interface PanelPos {
  left: number;
  width: number;
  /** Set for a panel that hangs below the trigger. */
  top?: number;
  /** Set instead of `top` when the panel flipped above it. */
  bottom?: number;
  maxHeight: number;
}

export function StyledSelect<T extends string | number>({
  value,
  onChange,
  options,
  disabled,
  placeholder = '—',
  accent = 'pink',
  size = 'md',
  className = '',
  searchable,
  searchPlaceholder = 'Filter…',
  emptyLabel = 'Nothing matches.',
  id,
  title,
  'aria-label': ariaLabel,
}: Props<T>) {
  const [open, setOpen] = useState(false);
  const [focusIdx, setFocusIdx] = useState(-1);
  const [query, setQuery] = useState('');
  const [pos, setPos] = useState<PanelPos | null>(null);

  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const a = ACCENT[accent];
  const s = SIZE[size];

  const withFilter = searchable ?? options.length >= 8;
  const q = query.trim().toLowerCase();
  const filtered = q ? options.filter(o => o.label.toLowerCase().includes(q) || String(o.value).toLowerCase().includes(q)) : options;

  const selected = options.find(o => o.value === value);

  // Derived, not stored: `disabled` can flip to true from outside (a job
  // starts) while the panel is open, and a disabled control must never be left
  // showing one. Deriving it means there is no state to resynchronise.
  const panelOpen = open && !disabled;

  /** Open and close set the transient panel state together, in the event
   *  handler — the alternative (an effect keyed on `open`) is a cascading
   *  render for state that never comes from outside this component. */
  const openPanel = () => {
    setQuery('');
    setFocusIdx(Math.max(0, options.findIndex(o => o.value === value)));
    setOpen(true);
  };
  const closePanel = () => {
    setOpen(false);
    setQuery('');
  };

  const measure = useCallback(() => {
    const el = triggerRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const below = window.innerHeight - r.bottom - 8;
    const above = r.top - 8;
    // Prefer below; flip only when below is genuinely cramped AND above is
    // roomier, so a dropdown near the bottom of a long form stays readable.
    const flip = below < 200 && above > below;
    setPos({
      left: r.left,
      width: r.width,
      ...(flip ? { bottom: window.innerHeight - r.top + 4 } : { top: r.bottom + 4 }),
      maxHeight: Math.max(140, Math.min(320, flip ? above : below)),
    });
  }, []);

  useLayoutEffect(() => {
    if (!panelOpen) return;
    measure();
    // `true` so the panel also follows ancestors that scroll, not just window.
    const onScroll = () => measure();
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onScroll);
    };
  }, [panelOpen, measure]);

  // Close on click outside either the trigger or the (portalled) panel.
  useEffect(() => {
    if (!panelOpen) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as Node;
      if (triggerRef.current?.contains(target) || panelRef.current?.contains(target)) return;
      setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [panelOpen]);

  // Moving DOM focus into the filter box is a genuine external-system effect,
  // unlike the query/focus-row reset, which openPanel() does directly.
  useEffect(() => {
    if (!panelOpen || !withFilter) return;
    const t = setTimeout(() => inputRef.current?.focus(), 0);
    return () => clearTimeout(t);
  }, [panelOpen, withFilter]);

  useEffect(() => {
    if (!panelOpen || focusIdx < 0 || !listRef.current) return;
    (listRef.current.children[focusIdx] as HTMLElement | undefined)?.scrollIntoView({ block: 'nearest' });
  }, [focusIdx, panelOpen]);

  const commit = (opt: SelectOption<T>) => {
    if (opt.disabled) return;
    onChange(opt.value);
    closePanel();
    triggerRef.current?.focus();
  };

  /** Shared by the trigger (when open, without a filter box) and the filter
   *  input, so arrow keys behave the same either way. */
  const navKeys = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setFocusIdx(i => Math.min(i + 1, filtered.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setFocusIdx(i => Math.max(i - 1, 0));
        break;
      case 'Home':
        e.preventDefault();
        setFocusIdx(0);
        break;
      case 'End':
        e.preventDefault();
        setFocusIdx(filtered.length - 1);
        break;
      case 'Enter':
        e.preventDefault();
        if (focusIdx >= 0 && focusIdx < filtered.length) commit(filtered[focusIdx]);
        break;
      case 'Escape':
        e.preventDefault();
        closePanel();
        triggerRef.current?.focus();
        break;
      case 'Tab':
        closePanel();
        break;
    }
  };

  const onTriggerKeyDown = (e: React.KeyboardEvent) => {
    if (!panelOpen) {
      if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openPanel();
      }
      return;
    }
    if (!withFilter) navKeys(e);
    else if (e.key === 'Escape') {
      e.preventDefault();
      closePanel();
    }
  };

  return (
    <>
      <button
        ref={triggerRef}
        id={id}
        type="button"
        role="combobox"
        aria-expanded={panelOpen}
        aria-haspopup="listbox"
        aria-label={ariaLabel}
        disabled={disabled}
        title={title ?? selected?.label}
        onClick={() => (panelOpen ? closePanel() : openPanel())}
        onKeyDown={onTriggerKeyDown}
        className={`w-full flex items-center gap-2 rounded-xl ${s.trigger}
                    bg-zinc-100 dark:bg-zinc-800
                    border border-zinc-300 dark:border-white/10
                    text-zinc-800 dark:text-zinc-200
                    hover:border-zinc-400 dark:hover:border-white/20
                    ${a.focus}
                    outline-none transition-colors cursor-pointer
                    disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:border-zinc-300
                    dark:disabled:hover:border-white/10 ${className}`}
      >
        <span className={`truncate flex-1 text-left${selected ? '' : ' text-zinc-400'}`}>
          {selected ? selected.label : placeholder}
        </span>
        <ChevronDown
          size={s.chevron}
          className={`shrink-0 text-zinc-400 transition-transform duration-150 ${panelOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {panelOpen && pos && ReactDOM.createPortal(
        <div
          ref={panelRef}
          data-portal-layer
          className="fixed z-[9999] rounded-xl bg-white dark:bg-zinc-800
                     border border-zinc-200 dark:border-white/10
                     shadow-lg shadow-black/20 overflow-hidden"
          style={{
            left: pos.left,
            width: pos.width,
            minWidth: 140,
            ...(pos.top !== undefined ? { top: pos.top } : { bottom: pos.bottom }),
          }}
        >
          {withFilter && (
            <div className="p-1.5 border-b border-zinc-200 dark:border-white/10">
              <div className="relative">
                <Search size={13} className="absolute left-2 top-1/2 -translate-y-1/2 text-zinc-400 pointer-events-none" />
                <input
                  ref={inputRef}
                  type="text"
                  value={query}
                  onChange={(e) => { setQuery(e.target.value); setFocusIdx(0); }}
                  onKeyDown={navKeys}
                  placeholder={searchPlaceholder}
                  className={`w-full pl-7 pr-2 py-1.5 rounded-lg
                              bg-zinc-100 dark:bg-zinc-900
                              border border-zinc-200 dark:border-white/10
                              text-sm text-zinc-800 dark:text-zinc-200 placeholder-zinc-400
                              outline-none ${a.focus}`}
                />
              </div>
            </div>
          )}

          <div
            ref={listRef}
            role="listbox"
            className="overflow-auto py-1"
            style={{ maxHeight: pos.maxHeight - (withFilter ? 46 : 0) }}
          >
            {filtered.length === 0 ? (
              <div className="px-3 py-2 text-sm text-zinc-400">{emptyLabel}</div>
            ) : (
              filtered.map((opt, i) => {
                const isSel = opt.value === value;
                const isFocus = i === focusIdx;
                return (
                  <button
                    key={String(opt.value)}
                    type="button"
                    role="option"
                    aria-selected={isSel}
                    disabled={opt.disabled}
                    onClick={() => commit(opt)}
                    onMouseEnter={() => setFocusIdx(i)}
                    title={opt.hint ?? opt.label}
                    className={`w-full flex items-start gap-2 text-left transition-colors ${s.row}
                      ${isFocus && !opt.disabled ? a.row : ''}
                      ${isSel ? a.text : 'text-zinc-700 dark:text-zinc-200'}
                      ${opt.disabled ? 'opacity-40 cursor-not-allowed' : ''}`}
                  >
                    <span className="min-w-0 flex-1">
                      <span className="block truncate">{opt.label}</span>
                      {opt.hint && (
                        <span className="block truncate text-[10px] text-zinc-500">{opt.hint}</span>
                      )}
                    </span>
                    {isSel && <Check size={14} className={`shrink-0 mt-0.5 ${a.icon}`} />}
                  </button>
                );
              })
            )}
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}

export default StyledSelect;
