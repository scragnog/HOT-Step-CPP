import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useVram } from '@/queries/global-models/hooks';
import { loadedModelsOptions } from '@/queries/global-models/options';
import { apiClient } from '@/services/httpClient';
import { sharedKeys } from '@/queries/keys';
import { QueryError } from '@/components/shared/feedback/QueryError';
import { queryManager } from '@/queries/invalidation';
// VramIndicator.tsx — GPU VRAM usage badge (top of the Terminal box) +
// manual model-unload dropdown
//
// Polls the shared /api/logs/vram query every 1.5 seconds for the used/total bar. On hover it
// opens a dropdown listing the GPU modules currently resident in the engine
// (LM / DiT / VAE / encoders …) with a per-module unload button — a manual
// fallback to free VRAM at any time. In-use modules are disabled; under
// keep-loaded an unloaded module simply reloads on next use, so it's safe.
//
// The dropdown is rendered via a portal to document.body so it escapes the
// right column's stacking context, left-aligned under the badge.

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Cpu, X, Loader2 } from 'lucide-react';
import { formatDecimal } from '@/utils/format/format';

export const VramIndicator: React.FC = () => {
  const { data: vram } = useVram();
  const client = useQueryClient();
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top?: number; bottom?: number; left: number } | null>(null);
  const loadedQuery = useQuery({ ...loadedModelsOptions(), enabled: open });
  const loaded = loadedQuery.data?.loaded ?? null;
  // No onError: this renders no inline error line, so the global handler is
  // the notifier for a failed unload.
  const unloadMut = useMutation({
    mutationFn: (label: string) => apiClient.post('/logs/models-unload', { label }),
    onSuccess: () => { void queryManager.invalidate(client, sharedKeys.loadedModels(), sharedKeys.vram()); },
  });
  const unloading = unloadMut.isPending ? unloadMut.variables : null;
  const wrapRef = useRef<HTMLDivElement>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => { if (closeTimer.current) clearTimeout(closeTimer.current); }, []);

  const openMenu = useCallback(() => {
    if (closeTimer.current) clearTimeout(closeTimer.current);
    const r = wrapRef.current?.getBoundingClientRect();
    // 240 px wide (w-60): keep it on screen when the badge sits near the right
    // edge, and open upwards when the Terminal sits low in the window.
    if (r) {
      const left = Math.max(8, Math.min(r.left, window.innerWidth - 248));
      setPos(window.innerHeight - r.bottom < 280
        ? { bottom: window.innerHeight - r.top + 6, left }
        : { top: r.bottom + 6, left });
    }
    setOpen(true);
  }, []);

  // Hover-intent: small delay so moving from the indicator into the portal
  // dropdown (not a DOM child of the wrapper) doesn't immediately close it.
  const scheduleClose = useCallback(() => {
    if (closeTimer.current) clearTimeout(closeTimer.current);
    closeTimer.current = setTimeout(() => setOpen(false), 160);
  }, []);

  if (!vram || vram.total_mb <= 0) return null;

  const percent = Math.round((vram.used_mb / vram.total_mb) * 100);
  const colorText = percent > 90 ? 'text-red-400' : percent > 70 ? 'text-yellow-400' : 'text-emerald-400';
  const colorBg = percent > 90 ? 'bg-red-500' : percent > 70 ? 'bg-yellow-500' : 'bg-emerald-500';

  return (
    <div ref={wrapRef} className="relative" onMouseEnter={openMenu} onMouseLeave={scheduleClose}>
      <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-zinc-100/80 dark:bg-zinc-800/80 border border-zinc-200 dark:border-white/5 cursor-default">
        <Cpu size={11} className="text-zinc-500 flex-shrink-0" />
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-zinc-500">VRAM</span>
          <span className={`text-[10px] font-mono font-medium ${colorText}`}>
            {formatDecimal((vram.used_mb / 1024), 1)} / {formatDecimal((vram.total_mb / 1024), 1)} GB
          </span>
          <div className="w-12 h-1 rounded-full bg-zinc-200 dark:bg-zinc-700 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${colorBg}`}
              style={{ width: `${percent}%` }}
            />
          </div>
        </div>
      </div>

      {open && pos && createPortal(
        <div
          data-portal-layer
          className="fixed z-[9999] w-60 rounded-lg border border-white/10 bg-white dark:bg-zinc-900 shadow-2xl p-2"
          style={pos}
          onMouseEnter={openMenu}
          onMouseLeave={scheduleClose}
        >
          <div className="flex items-center justify-between px-1 pb-1.5 mb-1 border-b border-black/5 dark:border-white/5">
            <span className="text-[10px] uppercase tracking-wide text-zinc-500">Loaded models</span>
            <span className="text-[10px] font-mono text-zinc-500">{formatDecimal((vram.used_mb / 1024), 1)} GB used</span>
          </div>

          {loadedQuery.isError ? (
            <QueryError error={loadedQuery.error} onRetry={() => void loadedQuery.refetch()} compact />
          ) : loaded === null ? (
            <div className="text-[11px] text-zinc-500 px-1 py-2 flex items-center gap-1.5">
              <Loader2 size={11} className="animate-spin" /> Loading…
            </div>
          ) : loaded.length === 0 ? (
            <div className="text-[11px] text-zinc-500 px-1 py-2">No models resident in VRAM.</div>
          ) : (
            <div className="space-y-0.5">
              {loaded.map(m => (
                <div key={m.label} className="flex items-center gap-2 px-1 py-1 rounded hover:bg-black/5 dark:hover:bg-white/5">
                  <span className="text-[11px] text-zinc-800 dark:text-white flex-1 truncate">{m.label}</span>
                  {m.in_use && <span className="text-[9px] text-amber-500 uppercase">in use</span>}
                  <span className="text-[10px] text-zinc-500 font-mono">{formatDecimal((m.mb / 1024), 1)} GB</span>
                  <button
                    onClick={() => unloadMut.mutate(m.label)}
                    disabled={m.in_use || unloading === m.label}
                    title={m.in_use ? "In use — can't unload right now" : `Unload ${m.label}`}
                    className="w-5 h-5 rounded flex items-center justify-center text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-30 disabled:hover:text-zinc-500 disabled:hover:bg-transparent"
                  >
                    {unloading === m.label ? <Loader2 size={11} className="animate-spin" /> : <X size={11} />}
                  </button>
                </div>
              ))}
            </div>
          )}

          {loaded && loaded.length > 0 && (() => {
            const weights = loaded.reduce((a, m) => a + m.mb, 0);
            const other = Math.max(0, vram.used_mb - weights);
            return (
              <div className="flex items-center gap-2 px-1 py-1 mt-0.5 border-t border-black/5 dark:border-white/5">
                <span className="text-[11px] text-zinc-500 flex-1">Compute + buffers</span>
                <span className="text-[10px] font-mono text-zinc-500">{formatDecimal((other / 1024), 1)} GB</span>
                <span className="w-5 flex-shrink-0" />
              </div>
            );
          })()}

          <div className="text-[10px] text-zinc-500 px-1 pt-1.5 mt-1 border-t border-black/5 dark:border-white/5">
            Weights are listed above; the rest is activations/CUDA (not unloadable). Unloaded models reload when next needed.
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
};
