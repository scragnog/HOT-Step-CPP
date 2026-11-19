// LmAdapterPicker.tsx — filterable combobox over the planner-LM adapter registry
//
// `adapterApi.lmList()` returns every PEFT dir under adapters/lm — 194 of them
// on this machine today — so a bare <select> is unusable. This is a text filter
// plus a keyboard-navigable listbox, which is also what lets the milestone flow
// inject entries (`extraOptions`) that no registry scan will ever produce.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, Loader2, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useQuery } from '@tanstack/react-query';
import { trainingQueries } from '@/queries/training-data/queries';
import { formatBytes } from '@/utils/format/format';
import { useClickOutside } from '@/hooks/useClickOutside';
import { isPresent } from '@/utils/guards';

export interface LmAdapterOption {
  label: string;
  value: string;
  /** 'peft' | 'lokr' | 'safetensors' | '' (an injected option carries no kind). */
  kind?: string;
  size?: number;
  /** Eval score (marginal+transition JS to the artist, lower = closer) from the
   *  adapter's hot_step_eval.json sidecar; undefined/null = never evaluated. */
  evalScore?: number | null;
  evalVerdict?: string;
}

interface LmAdapterPickerProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  placeholder?: string;
  /** Milestone dirs and other non-registry entries, listed first. */
  extraOptions?: LmAdapterOption[];
}

export const LmAdapterPicker: React.FC<LmAdapterPickerProps> = ({
  value, onChange, disabled = false, placeholder, extraOptions,
}) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState('');
  const [cursor, setCursor] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // The registry is a filesystem walk on the server, and every picker on
  // screen shares one read of it.
  const lmQuery = useQuery(trainingQueries.lmAdapters());
  const loading = lmQuery.isPending;
  // Advisory: a failed scan must still leave the field usable — the value is a
  // plain string and the server validates it either way (C16).
  const loadError = lmQuery.error ? lmQuery.error.message : '';
  // Per-base + per-run layout: names are unsuffixed and an artist can have
  // several sizes and several training runs — tag both so every entry stays
  // distinguishable.
  const registry = useMemo<LmAdapterOption[]>(() => (lmQuery.data?.adapters ?? []).map(a => ({
    label: [a.name, a.lmSize, a.run].filter(isPresent).join(' · '),
    value: a.path, kind: a.kind, size: a.size,
    evalScore: a.evalScore, evalVerdict: a.evalVerdict,
  })), [lmQuery.data]);

  const options = useMemo<LmAdapterOption[]>(
    () => [...(extraOptions ?? []), ...registry],
    [extraOptions, registry],
  );

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return options;
    return options.filter(o =>
      o.label.toLowerCase().includes(q) || o.value.toLowerCase().includes(q));
  }, [options, filter]);

  // Close on outside click — the listbox is absolutely positioned over the card.
  useClickOutside([rootRef], () => close(), open);

  // -1 = the base-LM row, the resting position. See onKeyDown.
  useEffect(() => { setCursor(-1); }, [filter, open]);

  // Keep the highlighted row in view while arrowing through 194 entries.
  useEffect(() => {
    if (!open) return;
    const el = listRef.current?.querySelector<HTMLElement>(`[data-idx="${cursor}"]`);
    el?.scrollIntoView({ block: 'nearest' });
  }, [cursor, open]);

  const selected = options.find(o => o.value === value);
  const display = selected?.label ?? value;

  const commit = (v: string) => { onChange(v); setOpen(false); setFilter(''); };

  /** Close without committing, and DROP the filter. Leaving it behind means the
   *  next focus reopens the list pre-filtered by stale text with no visible
   *  cause. */
  const close = () => { setOpen(false); setFilter(''); };

  const onKeyDown = (e: React.KeyboardEvent) => {
    // cursor === -1 is the explicit base-LM row (data-idx={-1}), not "nothing".
    // It is the default and most-used value, so it has to be reachable by
    // keyboard; making it the resting cursor also stops the first ArrowDown
    // after opening from skipping straight past entry 0.
    if (e.key === 'ArrowDown') { e.preventDefault(); setOpen(true); setCursor(c => Math.min(c + 1, filtered.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setCursor(c => Math.max(c - 1, -1)); }
    else if (e.key === 'Enter') {
      e.preventDefault();
      const pick = cursor >= 0 ? filtered[cursor] : undefined;
      if (pick) { commit(pick.value); return; }
      // No highlighted row: commit the RAW typed text. Without this the picker
      // has no free-text path at all — when adapterApi.lmList() rejects, registry
      // stays empty, `filtered` is empty, Enter is a no-op, and the only
      // selectable value is '' (base LM). The adapterScanFailed hint literally
      // says "type or paste a path instead", which was impossible to act on, and
      // side B could never be given an adapter. It also reaches a milestone dir
      // that lmList() did not enumerate, which C16 explicitly permits.
      // An empty filter trims to '' — which IS the base-LM row's value.
      commit(filter.trim());
    } else if (e.key === 'Escape') { e.preventDefault(); close(); }
  };

  const field = 'w-full rounded-lg px-3 py-2 text-xs bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-amber-500 disabled:opacity-50';

  return (
    <div ref={rootRef} className="relative min-w-0">
      <div className="flex items-center gap-1">
        <input
          type="text"
          disabled={disabled}
          // While open, the box holds the filter — but an EMPTY filter must not
          // read as "nothing selected". Showing the current selection as the
          // placeholder keeps it visible, so a single click on a field that
          // already holds an adapter no longer appears to blank it.
          value={open ? filter : display}
          placeholder={open && display
            ? display
            : (placeholder ?? t('trainingStudio.audition.adapterPlaceholder'))}
          onFocus={() => setOpen(true)}
          onChange={(e) => { setOpen(true); setFilter(e.target.value); }}
          onKeyDown={onKeyDown}
          className={`${field} pr-14 font-mono truncate`}
          title={value || undefined}
        />
        {/* Kept INSIDE the field's own width — this picker lives in a
            grid-cols-2 cell and anything hung off the right edge clips. */}
        {value && !disabled && (
          <button
            type="button"
            onClick={() => commit('')}
            title={t('trainingStudio.audition.clearAdapter')}
            className="absolute right-6 p-0.5 rounded text-zinc-500 hover:text-red-500 transition-colors"
          >
            <X size={13} />
          </button>
        )}
        <div className="absolute right-2 flex items-center gap-1 pointer-events-none">
          {loading && <Loader2 size={12} className="animate-spin text-zinc-500" />}
          <ChevronDown size={13} className="text-zinc-500" />
        </div>
      </div>

      {loadError && (
        <p className="text-[10px] text-amber-600 dark:text-amber-400 mt-1">
          {t('trainingStudio.audition.adapterScanFailed')}
        </p>
      )}

      {open && (
        <div
          ref={listRef}
          className="absolute z-50 mt-1 w-full max-h-64 overflow-y-auto rounded-lg border border-zinc-200 dark:border-white/10 bg-white dark:bg-suno-card shadow-xl"
        >
          {/* Explicit base-LM row: '' is a legal, meaningful value here (C16). */}
          <button
            type="button"
            data-idx={-1}
            onMouseEnter={() => setCursor(-1)}
            onMouseDown={(e) => { e.preventDefault(); commit(''); }}
            className={`w-full text-left px-3 py-1.5 text-[11px] text-zinc-600 dark:text-zinc-400 hover:bg-amber-500/10 border-b border-zinc-200 dark:border-white/10 ${
              cursor === -1 ? 'bg-amber-500/10' : ''
            }`}
          >
            {t('trainingStudio.audition.baseLm')}
          </button>
          {filtered.length === 0 && (
            <div className="px-3 py-2 text-[11px] text-zinc-500">
              {t('trainingStudio.audition.noAdapters')}
            </div>
          )}
          {filtered.map((o, i) => (
            <button
              key={`${o.value}-${i}`}
              type="button"
              data-idx={i}
              onMouseEnter={() => setCursor(i)}
              onMouseDown={(e) => { e.preventDefault(); commit(o.value); }}
              title={o.value}
              className={`w-full text-left px-3 py-1.5 flex items-center gap-2 ${
                i === cursor ? 'bg-amber-500/10' : ''
              } ${o.value === value ? 'text-amber-600 dark:text-amber-400' : 'text-zinc-700 dark:text-zinc-300'}`}
            >
              <span className="flex-1 min-w-0 truncate text-[11px] font-mono">{o.label}</span>
              {/* Eval score from the adapter's sidecar — lower = closer to the
                  artist. Green when the eval verdict was 'toward'. */}
              {typeof o.evalScore === 'number' && (
                <span
                  title={`Artist-match score (marginal+transition JS, lower = closer) — verdict: ${o.evalVerdict || 'n/a'}`}
                  className={`text-[9px] tabular-nums flex-shrink-0 ${
                    o.evalVerdict === 'toward'
                      ? 'text-emerald-600 dark:text-emerald-400'
                      : o.evalVerdict === 'away'
                        ? 'text-red-500 dark:text-red-400'
                        : 'text-zinc-500'
                  }`}
                >
                  {o.evalScore.toFixed(3)}
                </span>
              )}
              {o.kind && <span className="text-[9px] text-zinc-500 flex-shrink-0">{o.kind}</span>}
              {formatBytes(o.size, { zero: '' }) && (
                <span className="text-[9px] text-zinc-500 tabular-nums flex-shrink-0">{formatBytes(o.size, { zero: '' })}</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default LmAdapterPicker;
