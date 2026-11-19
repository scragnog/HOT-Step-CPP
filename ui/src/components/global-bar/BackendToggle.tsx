import { useQuery } from '@tanstack/react-query';
import { backendsOptions } from '@/queries/global-models/options';
// BackendToggle.tsx — the generation-engine picker at the top of the left
// sidebar, directly under the logo.
//
// Lists registered engine backends (GET /api/backends) and switches the
// active one via backendStore.switchBackend(). Deliberately large and bold:
// it is where the engine is changed, and the sidebar's studio list follows it.
// The option list is portaled to document.body because the sidebar scrolls
// (overflow-y-auto would clip an absolute dropdown).
//
// CRITICAL: renders null while only one backend is registered — see
// docs/plans/multi-backend-architecture.md §4.5.

import React, { useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { Layers, Check, ChevronDown } from 'lucide-react';
import { useBackendStore } from '@/stores/backendStore';
import { useCapabilities } from '@/hooks/useCapabilities';
import { useClickOutside } from '@/hooks/useClickOutside';

export const BackendToggle: React.FC<{ collapsed?: boolean }> = ({ collapsed = false }) => {
  const { t } = useTranslation();
  const backends = useBackendStore(s => s.backends);
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const switchBackend = useBackendStore(s => s.switchBackend);
  const { capabilities } = useCapabilities();
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useQuery(backendsOptions());

  useClickOutside([buttonRef, listRef], () => setPos(null), pos !== null);

  // The whole feature must be invisible until a second backend is registered.
  if (backends.length < 2) return null;

  const active = backends.find(b => b.id === activeBackendId) || backends[0];

  // Optional plain-text notice a backend's manifest can carry (e.g. a
  // weights-licence reminder). Generic: keyed off the manifest field, never a
  // backend id. Surfaced as a tooltip.
  const notice = typeof capabilities?.license === 'string' ? capabilities.license : (typeof capabilities?.core?.notice === 'string' ? capabilities.core.notice : undefined);
  const pickerTitle = notice ? `${t('globalBar.backend')} — ${notice}` : t('globalBar.backend');

  const toggle = () => {
    if (pos) { setPos(null); return; }
    const r = buttonRef.current?.getBoundingClientRect();
    if (r) setPos(collapsed ? { top: r.top, left: r.right + 6 } : { top: r.bottom + 4, left: r.left });
  };

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        onClick={toggle}
        title={pickerTitle}
        className={`w-full rounded-xl border-2 border-pink-500/60 bg-pink-500/10 hover:bg-pink-500/20 transition-colors text-left ${
          collapsed ? 'flex flex-col items-center gap-1 px-1 py-2' : 'flex items-center gap-2 px-3 py-2.5'
        }`}
      >
        <Layers size={collapsed ? 18 : 16} className="flex-shrink-0 text-pink-400" />
        <span className={`min-w-0 ${collapsed ? 'text-center' : 'flex-1'}`}>
          {!collapsed && (
            <span className="block text-[9px] font-semibold uppercase tracking-wider text-pink-400/80">
              {t('globalBar.engine', 'Generation engine')}
            </span>
          )}
          {/*
            The MiniMax-Music3 community license (§4.2 "commercial products
            must prominently display 'MiniMax-Music3'") is satisfied by
            rendering the server's displayName string verbatim here — wrapped,
            never truncated, and never a friendlier substitute label.
          */}
          <span className={`block font-bold text-zinc-900 dark:text-white break-words ${collapsed ? 'text-[10px] leading-tight' : 'text-sm leading-snug'}`}>
            {active?.displayName || active?.id}
          </span>
        </span>
        {!collapsed && <ChevronDown size={14} className={`flex-shrink-0 text-pink-400 transition-transform ${pos ? 'rotate-180' : ''}`} />}
      </button>

      {pos && createPortal(
        <div
          ref={listRef}
          data-portal-layer
          className="fixed z-[9999] min-w-[200px] rounded-xl border border-zinc-300 dark:border-white/10 bg-white dark:bg-zinc-900 shadow-2xl shadow-black/30 dark:shadow-black/60 p-1 global-bar-dropdown-enter"
          style={{ top: pos.top, left: pos.left }}
        >
          {backends.map(b => (
            <button
              key={b.id}
              type="button"
              onClick={() => {
                setPos(null);
                if (b.id !== activeBackendId) void switchBackend(b.id);
              }}
              className={`w-full flex items-center justify-between gap-2 px-2.5 py-2 rounded-lg text-sm text-left transition-colors ${
                b.id === activeBackendId
                  ? 'bg-pink-500/10 text-pink-400 font-semibold'
                  : 'text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800'
              }`}
            >
              <span>{b.displayName}</span>
              {b.id === activeBackendId && <Check size={14} className="flex-shrink-0" />}
            </button>
          ))}
        </div>,
        document.body,
      )}
    </>
  );
};
