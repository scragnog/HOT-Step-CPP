// ModelSelect.tsx — Custom dropdown for model selection with format badges
//
// Replaces native <select> to allow rich rendering of options with
// GGUF/SafeTensors format indicators. Uses click-outside and keyboard
// navigation for accessibility.

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ChevronDown, Check, Search } from 'lucide-react';

/** Detect model format from the raw model name/path.
 *  ONNX detection: .onnx extension OR known ONNX directory name patterns
 *  (the C++ registry registers ONNX subdirectories by directory name). */
export function getModelFormat(name: string): 'gguf' | 'safetensors' | 'onnx' {
  if (/\.onnx$/i.test(name)) return 'onnx';
  if (/\.gguf$/i.test(name)) return 'gguf';
  // ONNX model directories: names like 'lm-4B', 'dit-xl', etc.
  // that don't have .gguf or .safetensors extensions
  // and don't match safetensors naming patterns (acestep-*, Qwen3-*, vae*, scragvae*)
  const lower = name.toLowerCase();
  if (/^lm-\d/i.test(name)) return 'onnx';
  if (/^dit-/i.test(name) && !lower.includes('acestep')) return 'onnx';
  if (/^vae-/i.test(name) && !lower.endsWith('.safetensors')) return 'onnx';
  if (/^text[_-]enc/i.test(name)) return 'onnx';
  return 'safetensors';
}

/** Middle-truncate a long label so BOTH ends stay visible (the tail often holds
 *  the distinguishing suffix, e.g. "…-xlremap-s0.3"). CSS truncates only the
 *  right, which hides exactly that. Returns the string unchanged if short. */
export function middleEllipsis(s: string, max = 40): string {
  if (s.length <= max) return s;
  const keep = max - 1; // room for the ellipsis
  const head = Math.ceil(keep * 0.55);
  const tail = keep - head;
  return `${s.slice(0, head)}…${s.slice(s.length - tail)}`;
}

interface FormatBadgeProps {
  format: 'gguf' | 'safetensors' | 'onnx';
  compact?: boolean;
}

/** Tiny pill showing GGUF, ST, or ONNX format */
export const FormatBadge: React.FC<FormatBadgeProps> = ({ format, compact }) => {
  const colorClass = format === 'gguf'
    ? 'bg-sky-500/15 text-sky-400 ring-1 ring-sky-500/20'
    : format === 'onnx'
    ? 'bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/20'
    : 'bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/20';
  const icon = format === 'gguf' ? '◆' : format === 'onnx' ? '⬡' : '◈';
  const label = format === 'gguf' ? (compact ? 'GG' : 'GGUF')
    : format === 'onnx' ? (compact ? 'OX' : 'ONNX')
    : (compact ? 'ST' : 'ST');
  const title = format === 'gguf' ? 'GGUF quantized format'
    : format === 'onnx' ? 'ONNX TensorRT-accelerated format'
    : 'SafeTensors native format';
  return (
    <span
      className={`inline-flex items-center gap-0.5 shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold leading-none tracking-wide uppercase ${colorClass}`}
      title={title}
    >
      <span className="text-[9px]">{icon}</span>
      {label}
    </span>
  );
};

interface ModelSelectProps {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  formatLabel?: (name: string) => string;
  placeholder?: string;
  id?: string;
}

export const ModelSelect: React.FC<ModelSelectProps> = ({
  value,
  onChange,
  options,
  formatLabel = (n) => n,
  placeholder = 'Select model…',
  id,
}) => {
  const [open, setOpen] = useState(false);
  const [focusIdx, setFocusIdx] = useState(-1);
  const [query, setQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Text-filtered options: case-insensitive substring match against the raw
  // model name and its formatted label.
  const q = query.trim().toLowerCase();
  const filtered = q
    ? options.filter((o) => o.toLowerCase().includes(q) || formatLabel(o).toLowerCase().includes(q))
    : options;

  // Close on click outside
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  // On open: reset the query and focus the filter input. On close: clear query.
  useEffect(() => {
    if (open) {
      setQuery('');
      setFocusIdx(Math.max(0, options.indexOf(value)));
      const t = setTimeout(() => inputRef.current?.focus(), 0);
      return () => clearTimeout(t);
    }
    setQuery('');
  }, [open]);

  // Scroll focused item into view
  useEffect(() => {
    if (!open || focusIdx < 0 || !listRef.current) return;
    const el = listRef.current.children[focusIdx] as HTMLElement | undefined;
    el?.scrollIntoView({ block: 'nearest' });
  }, [focusIdx, open]);

  // Trigger-button keys: only used to open the dropdown.
  const handleTriggerKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (!open && (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ')) {
        e.preventDefault();
        setOpen(true);
      }
    },
    [open]
  );

  // Filter-input keys: navigate + select within the filtered list.
  const handleInputKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setFocusIdx((i) => Math.min(i + 1, filtered.length - 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setFocusIdx((i) => Math.max(i - 1, 0));
          break;
        case 'Enter':
          e.preventDefault();
          if (focusIdx >= 0 && focusIdx < filtered.length) {
            onChange(filtered[focusIdx]);
            setOpen(false);
          }
          break;
        case 'Escape':
          e.preventDefault();
          setOpen(false);
          break;
      }
    },
    [focusIdx, filtered, onChange]
  );

  const selectedFormat = value ? getModelFormat(value) : null;

  return (
    <div ref={containerRef} className="relative" id={id}>
      {/* Trigger button */}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        onKeyDown={handleTriggerKeyDown}
        title={value || undefined}
        className="w-full flex items-center gap-2 px-3 py-2 rounded-xl
                   bg-zinc-100 dark:bg-zinc-800
                   border border-zinc-300 dark:border-white/10
                   text-sm text-zinc-800 dark:text-zinc-200
                   hover:border-zinc-400 dark:hover:border-white/20
                   focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20
                   outline-none transition-colors cursor-pointer"
      >
        {value ? (
          <>
            {selectedFormat && <FormatBadge format={selectedFormat} compact />}
            <span className="truncate flex-1 text-left">{middleEllipsis(formatLabel(value))}</span>
          </>
        ) : (
          <span className="truncate flex-1 text-left text-zinc-400">{placeholder}</span>
        )}
        <ChevronDown
          size={14}
          className={`shrink-0 text-zinc-400 transition-transform duration-150 ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {/* Dropdown panel: filter box + scrollable list */}
      {open && (
        <div
          className="absolute z-50 mt-1 w-full rounded-xl
                     bg-white dark:bg-zinc-800
                     border border-zinc-200 dark:border-white/10
                     shadow-lg shadow-black/20"
        >
          {/* Text filter */}
          <div className="p-1.5 border-b border-zinc-200 dark:border-white/10">
            <div className="relative">
              <Search size={13} className="absolute left-2 top-1/2 -translate-y-1/2 text-zinc-400 pointer-events-none" />
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => { setQuery(e.target.value); setFocusIdx(0); }}
                onKeyDown={handleInputKeyDown}
                placeholder="Filter models…"
                className="w-full pl-7 pr-2 py-1.5 rounded-lg
                           bg-zinc-100 dark:bg-zinc-900
                           border border-zinc-200 dark:border-white/10
                           text-sm text-zinc-800 dark:text-zinc-200 placeholder-zinc-400
                           outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20"
              />
            </div>
          </div>

          {/* Filtered list */}
          <div ref={listRef} className="max-h-56 overflow-auto py-1" role="listbox">
            {filtered.length === 0 ? (
              <div className="px-3 py-2 text-sm text-zinc-400">No models match.</div>
            ) : (
              filtered.map((opt, i) => {
                const fmt = getModelFormat(opt);
                const selected = opt === value;
                const focused = i === focusIdx;
                return (
                  <button
                    key={opt}
                    type="button"
                    role="option"
                    aria-selected={selected}
                    onClick={() => {
                      onChange(opt);
                      setOpen(false);
                    }}
                    onMouseEnter={() => setFocusIdx(i)}
                    title={opt}
                    className={`w-full flex items-center gap-2 px-3 py-2 text-sm text-left transition-colors
                      ${focused ? 'bg-pink-500/10 dark:bg-pink-500/15' : ''}
                      ${selected ? 'text-pink-400' : 'text-zinc-700 dark:text-zinc-200'}
                      hover:bg-pink-500/10 dark:hover:bg-pink-500/15`}
                  >
                    <FormatBadge format={fmt} />
                    <span className="truncate flex-1">{middleEllipsis(formatLabel(opt), 48)}</span>
                    {selected && <Check size={14} className="shrink-0 text-pink-400" />}
                  </button>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
};
