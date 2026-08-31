// VramIndicator.tsx — GPU VRAM usage display + manual model-unload dropdown
//
// Polls /api/logs/vram every 5 seconds for the used/total bar. On hover it
// opens a dropdown listing the GPU modules currently resident in the engine
// (LM / DiT / VAE / encoders …) with a per-module unload button — a manual
// fallback to free VRAM at any time. In-use modules are disabled; under
// keep-loaded an unloaded module simply reloads on next use, so it's safe.
//
// The dropdown is rendered via a portal to document.body so it escapes the
// GlobalParamBar's (z-40) stacking context and paints above the terminal etc.

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Cpu, X, Loader2 } from 'lucide-react';

interface VramInfo {
  used_mb: number;
  total_mb: number;
  free_mb: number;
}

interface LoadedModel {
  label: string;
  mb: number;
  in_use: boolean;
}

interface VramIndicatorProps {
  /** Polling interval in ms (default 5000) */
  pollInterval?: number;
  /** Compact mode — hides the label and uses smaller text */
  compact?: boolean;
}

export const VramIndicator: React.FC<VramIndicatorProps> = ({
  pollInterval = 5000,
  compact = false,
}) => {
  const [vram, setVram] = useState<VramInfo | null>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; right: number } | null>(null);
  const [loaded, setLoaded] = useState<LoadedModel[] | null>(null);
  const [unloading, setUnloading] = useState<string | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const fetchVram = async () => {
      try {
        const res = await fetch('/api/logs/vram');
        if (res.ok) {
          const data = await res.json();
          if (data.total_mb > 0) setVram(data);
        }
      } catch { /* ignore */ }
    };
    fetchVram();
    const interval = setInterval(fetchVram, pollInterval);
    return () => clearInterval(interval);
  }, [pollInterval]);

  const fetchLoaded = useCallback(async () => {
    try {
      const res = await fetch('/api/logs/models-loaded');
      if (res.ok) {
        const d = await res.json();
        setLoaded(Array.isArray(d.loaded) ? d.loaded : []);
      } else setLoaded([]);
    } catch { setLoaded([]); }
  }, []);

  // While the dropdown is open, fetch the resident-model list and refresh it.
  useEffect(() => {
    if (!open) return;
    fetchLoaded();
    const iv = setInterval(fetchLoaded, 2000);
    return () => clearInterval(iv);
  }, [open, fetchLoaded]);

  useEffect(() => () => { if (closeTimer.current) clearTimeout(closeTimer.current); }, []);

  const openMenu = useCallback(() => {
    if (closeTimer.current) clearTimeout(closeTimer.current);
    const r = wrapRef.current?.getBoundingClientRect();
    if (r) setPos({ top: r.bottom + 6, right: Math.max(8, window.innerWidth - r.right) });
    setOpen(true);
  }, []);

  // Hover-intent: small delay so moving from the indicator into the portal
  // dropdown (not a DOM child of the wrapper) doesn't immediately close it.
  const scheduleClose = useCallback(() => {
    if (closeTimer.current) clearTimeout(closeTimer.current);
    closeTimer.current = setTimeout(() => setOpen(false), 160);
  }, []);

  const unload = useCallback(async (label: string) => {
    setUnloading(label);
    try {
      await fetch('/api/logs/models-unload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label }),
      });
      await fetchLoaded();
    } catch { /* ignore */ } finally {
      setUnloading(null);
    }
  }, [fetchLoaded]);

  if (!vram || vram.total_mb <= 0) return null;

  const percent = Math.round((vram.used_mb / vram.total_mb) * 100);
  const colorText = percent > 90 ? 'text-red-400' : percent > 70 ? 'text-yellow-400' : 'text-emerald-400';
  const colorBg = percent > 90 ? 'bg-red-500' : percent > 70 ? 'bg-yellow-500' : 'bg-emerald-500';

  return (
    <div ref={wrapRef} className="relative" onMouseEnter={openMenu} onMouseLeave={scheduleClose}>
      <div className="flex items-center gap-1.5 cursor-default">
        <Cpu size={compact ? 10 : 11} className="text-zinc-500 flex-shrink-0" />
        <div className="flex items-center gap-1">
          {!compact && <span className="text-[10px] text-zinc-500">VRAM</span>}
          <span className={`text-[10px] font-mono font-medium ${colorText}`}>
            {(vram.used_mb / 1024).toFixed(1)}/{(vram.total_mb / 1024).toFixed(1)}
          </span>
          <div className="w-10 h-1 rounded-full bg-zinc-200 dark:bg-zinc-700 overflow-hidden">
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
          style={{ top: pos.top, right: pos.right }}
          onMouseEnter={openMenu}
          onMouseLeave={scheduleClose}
        >
          <div className="flex items-center justify-between px-1 pb-1.5 mb-1 border-b border-black/5 dark:border-white/5">
            <span className="text-[10px] uppercase tracking-wide text-zinc-500">Loaded models</span>
            <span className="text-[10px] font-mono text-zinc-500">{(vram.used_mb / 1024).toFixed(1)} GB used</span>
          </div>

          {loaded === null ? (
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
                  <span className="text-[10px] text-zinc-500 font-mono">{(m.mb / 1024).toFixed(1)} GB</span>
                  <button
                    onClick={() => unload(m.label)}
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
                <span className="text-[10px] font-mono text-zinc-500">{(other / 1024).toFixed(1)} GB</span>
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
