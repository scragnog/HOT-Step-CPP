// BackendToggle.tsx — compact backend picker for the global param bar
//
// Lists registered engine backends (GET /api/backends) and switches the
// active one via backendStore.switchBackend(). Styled to match the compact
// controls in the bar's right-hand cluster (see the Profiles button in
// GlobalParamBar.tsx) rather than the full BarSection accordion — this is a
// small pill, not a params dropdown.
//
// CRITICAL: renders null while only one backend is registered. Every install
// today has exactly one ('ace'), so this control must stay completely
// invisible until a second backend (e.g. MiniMax-Music3) actually exists —
// see docs/plans/multi-backend-architecture.md §4.5.

import React, { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Layers, Check } from 'lucide-react';
import { useBackendStore } from '../../stores/backendStore';
import { useCapabilities } from '../../hooks/useCapabilities';

export const BackendToggle: React.FC = () => {
  const { t } = useTranslation();
  const backends = useBackendStore(s => s.backends);
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const fetchBackends = useBackendStore(s => s.fetchBackends);
  const switchBackend = useBackendStore(s => s.switchBackend);
  const { capabilities } = useCapabilities();
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => { void fetchBackends(); }, [fetchBackends]);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, [open]);

  // The whole feature must be invisible until a second backend is registered.
  if (backends.length < 2) return null;

  const active = backends.find(b => b.id === activeBackendId) || backends[0];

  // Optional plain-text notice a backend's manifest can carry (backendStore.ts
  // BackendCoreCapabilities.notice) — e.g. a weights-licence reminder with an
  // upstream contact address. Generic: keyed off the manifest field,
  // never a backend id. Surfaced as a tooltip rather than an extra line under
  // the picker so a notice never changes the bar's fixed height.
  const notice = typeof capabilities?.license === 'string' ? capabilities.license : (typeof capabilities?.core?.notice === 'string' ? capabilities.core.notice : undefined);
  const pickerTitle = notice ? `${t('globalBar.backend')} — ${notice}` : t('globalBar.backend');

  return (
    <div ref={wrapRef} className="relative flex-shrink-0 flex items-center">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        title={pickerTitle}
        className="group flex items-center gap-1.5 px-2 py-1.5 rounded-lg hover:bg-pink-500/10 transition-colors duration-150"
      >
        <Layers size={13} className="flex-shrink-0 text-zinc-500 group-hover:text-pink-400 transition-colors duration-150" />
        <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-600 dark:text-zinc-400 group-hover:text-zinc-800 dark:group-hover:text-zinc-200 transition-colors duration-150 max-w-[120px] truncate">
          {/*
            The MiniMax-Music3 community license (§4.2 "commercial products
            must prominently display 'MiniMax-Music3'") is satisfied by
            rendering the server's displayName string verbatim here — do not
            reformat, truncate-with-ellipsis-only-partial, or substitute a
            friendlier label for that backend.
          */}
          {active?.displayName || active?.id}
        </span>
      </button>

      {open && (
        <div className="absolute top-full left-0 mt-1 z-50 min-w-[180px] rounded-xl border border-zinc-300 dark:border-white/10 bg-white dark:bg-zinc-900 shadow-2xl shadow-black/30 dark:shadow-black/60 p-1 global-bar-dropdown-enter">
          {backends.map(b => (
            <button
              key={b.id}
              type="button"
              onClick={() => {
                setOpen(false);
                if (b.id !== activeBackendId) void switchBackend(b.id);
              }}
              className={`w-full flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors ${
                b.id === activeBackendId
                  ? 'bg-pink-500/10 text-pink-400'
                  : 'text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800'
              }`}
            >
              <span className="truncate">{b.displayName}</span>
              {b.id === activeBackendId && <Check size={12} className="flex-shrink-0" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
