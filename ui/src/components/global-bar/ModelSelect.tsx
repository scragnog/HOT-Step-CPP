// ModelSelect.tsx — Custom dropdown for model selection with format badges
//
// Replaces native <select> to allow rich rendering of options with
// GGUF/SafeTensors format indicators. Uses click-outside and keyboard
// navigation for accessibility.

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ChevronDown, Check, Search } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { splitModel } from './modelLabels';
import { useClickOutside } from '@/hooks/useClickOutside';

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
  /** Override how an option's format badge is derived. The default sniffs the
   *  file name, which only works when the option IS a file name — backends
   *  whose options are something else (MM3 lists quant tokens like "q8_0")
   *  pass their own, or `null` to drop the badge entirely. */
  formatOf?: ((name: string) => 'gguf' | 'safetensors' | 'onnx') | null;
  /** Hide the filter box. Pointless for short, fixed option lists. */
  filterable?: boolean;
  disabled?: boolean;
  /** An unavailable option stays visible, with this explanation. */
  disabledReason?: (name: string) => string | undefined;
}

export const ModelSelect: React.FC<ModelSelectProps> = ({
  value,
  onChange,
  options,
  formatLabel = (n) => n,
  placeholder = 'Select model…',
  id,
  formatOf = getModelFormat,
  filterable = true,
  disabled = false,
  disabledReason,
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
  useClickOutside([containerRef], () => setOpen(false), open);

  // On open: reset the query and focus the filter input. On close: clear query.
  useEffect(() => {
    if (open) {
      setQuery('');
      setFocusIdx(options.includes(value) && !disabledReason?.(value)
        ? options.indexOf(value) : options.findIndex(o => !disabledReason?.(o)));
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

  // Filter-input keys: navigate + select within the filtered list.
  // Declared before the trigger handler, which depends on it.
  const handleInputKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setFocusIdx((i) => {
            const next = filtered.findIndex((o, idx) => idx > i && !disabledReason?.(o));
            return next < 0 ? i : next;
          });
          break;
        case 'ArrowUp':
          e.preventDefault();
          setFocusIdx((i) => {
            for (let idx = i - 1; idx >= 0; idx--) if (!disabledReason?.(filtered[idx])) return idx;
            return i;
          });
          break;
        case 'Enter':
          e.preventDefault();
          if (focusIdx >= 0 && focusIdx < filtered.length && !disabledReason?.(filtered[focusIdx])) {
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
    [focusIdx, filtered, onChange, disabledReason]
  );

  // Trigger-button keys: opens the dropdown. When the filter box is hidden
  // there is no input to take focus, so the trigger also has to drive
  // navigation once open — otherwise the list is mouse-only.
  const handleTriggerKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (!open && (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ')) {
        e.preventDefault();
        setOpen(true);
        return;
      }
      if (open && !filterable) {
        handleInputKeyDown(e);
      }
    },
    [open, filterable, handleInputKeyDown]
  );

  const selectedFormat = value && formatOf ? formatOf(value) : null;

  return (
    <div ref={containerRef} className="relative">
      {/* Trigger button */}
      <button
        id={id}
        type="button"
        onClick={() => !disabled && setOpen(!open)}
        onKeyDown={handleTriggerKeyDown}
        disabled={disabled}
        title={(value && disabledReason?.(value)) || value || undefined}
        className="w-full flex items-center gap-2 px-3 py-2 rounded-xl
                   bg-zinc-100 dark:bg-zinc-800
                   border border-zinc-300 dark:border-white/10
                   text-sm text-zinc-800 dark:text-zinc-200
                   hover:border-zinc-400 dark:hover:border-white/20
                   focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20
                   outline-none transition-colors cursor-pointer
                   disabled:opacity-50 disabled:cursor-not-allowed"
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
          {filterable && (
          <div className="p-1.5 border-b border-zinc-200 dark:border-white/10">
            <div className="relative">
              <Search size={13} className="absolute left-2 top-1/2 -translate-y-1/2 text-zinc-400 pointer-events-none" />
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => {
                  const nextQuery = e.target.value.trim().toLowerCase();
                  setQuery(e.target.value);
                  setFocusIdx(options.filter(o => o.toLowerCase().includes(nextQuery)
                    || formatLabel(o).toLowerCase().includes(nextQuery)).findIndex(o => !disabledReason?.(o)));
                }}
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
          )}

          {/* Filtered list */}
          <div ref={listRef} className="max-h-56 overflow-auto py-1" role="listbox">
            {filtered.length === 0 ? (
              <div className="px-3 py-2 text-sm text-zinc-400">No models match.</div>
            ) : (
              filtered.map((opt, i) => {
                const fmt = formatOf ? formatOf(opt) : null;
                const selected = opt === value;
                const focused = i === focusIdx;
                const reason = disabledReason?.(opt);
                return (
                  <button
                    key={opt}
                    type="button"
                    role="option"
                    aria-selected={selected}
                    aria-disabled={!!reason}
                    disabled={!!reason}
                    onClick={() => {
                      if (reason) return;
                      onChange(opt);
                      setOpen(false);
                    }}
                    onMouseEnter={() => { if (!reason) setFocusIdx(i); }}
                    title={reason ? `${opt}: ${reason}` : opt}
                    className={`w-full flex items-center gap-2 px-3 py-2 text-sm text-left transition-colors
                      ${focused ? 'bg-pink-500/10 dark:bg-pink-500/15' : ''}
                      ${selected ? 'text-pink-400' : 'text-zinc-700 dark:text-zinc-200'}
                      ${reason ? 'opacity-50 cursor-not-allowed' : 'hover:bg-pink-500/10 dark:hover:bg-pink-500/15'}`}
                  >
                    {fmt && <FormatBadge format={fmt} />}
                    <span className="min-w-0 flex-1">
                      <span className="block truncate">{middleEllipsis(formatLabel(opt), 48)}</span>
                      {reason && <span className="block text-[10px] text-zinc-500 dark:text-zinc-400">{reason}</span>}
                    </span>
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

/** Two-step picker: variant on top, quants of that variant below (the
 *  HeartMuLa layout). Falls back to one plain ModelSelect when every option
 *  belongs to the same variant, so single-model backends look unchanged. */
export const VariantModelSelect: React.FC<ModelSelectProps & {
  variantLabel?: (variant: string) => string;
  quantLabel?: (name: string) => string;
}> = ({ variantLabel, quantLabel, ...props }) => {
  const { t } = useTranslation();
  const groups = new Map<string, string[]>();
  for (const o of props.options) {
    const v = splitModel(o).variant;
    groups.set(v, [...(groups.get(v) ?? []), o]);
  }
  // Nothing to group when every option is its own variant (or all share one).
  if (groups.size < 2 || groups.size === props.options.length) return <ModelSelect {...props} />;

  const formatLabel = props.formatLabel ?? ((n: string) => n);
  const labelVariant = variantLabel ?? formatLabel;
  const current = splitModel(props.value);
  const pick = (variant: string) => {
    const quants = groups.get(variant) ?? [];
    const usable = quants.filter(o => !props.disabledReason?.(o));
    // Keep the quant you had when the new variant has it.
    return usable.find(o => splitModel(o).quant.toLowerCase() === current.quant.toLowerCase())
      ?? usable[0] ?? quants[0];
  };
  return (
    <div className="space-y-2">
      <ModelSelect
        {...props}
        value={current.variant}
        options={[...groups.keys()]}
        onChange={(v) => props.onChange(pick(v))}
        formatLabel={labelVariant}
        formatOf={null}
        disabledReason={undefined}
        // '' is a real variant (bare quant tokens); without this it renders as "unset".
        placeholder={props.value ? labelVariant(current.variant) : props.placeholder}
      />
      <div>
        <label htmlFor={props.id && `${props.id}-quant`}
          className="block text-[10px] font-medium text-zinc-500 uppercase tracking-wider mb-1">
          {t('models.quantization', 'Quantization')}
        </label>
        <ModelSelect
          {...props}
          id={props.id && `${props.id}-quant`}
          options={groups.get(current.variant) ?? []}
          formatLabel={quantLabel ?? ((o) => splitModel(o).quant || formatLabel(o))}
          filterable={false}
        />
      </div>
    </div>
  );
};
