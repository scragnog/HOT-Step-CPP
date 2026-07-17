// TerminalPanel.tsx — Live engine log viewer with search and VRAM display
//
// Mirrors ace-server stderr output in the app. Supports search filtering,
// auto-scroll, and color-coded log lines.
//
// Performance: memoized line components + content-visibility: auto to skip
// layout/paint for offscreen lines. Combined with batched SSE updates in
// useEventSource, this keeps the UI responsive even during heavy logging.

import React, { useState, useEffect, useLayoutEffect, useRef, useCallback, useMemo } from 'react';
import { Search, X, Trash2, Cpu, ArrowDown, Pin, PinOff, Wifi, WifiOff } from 'lucide-react';
import { useEventSource, type LogLine } from '../../hooks/useEventSource';

interface VramInfo {
  used_mb: number;
  total_mb: number;
  free_mb: number;
}

interface TerminalPanelProps {
  onClose: () => void;
}

// Color classes by log prefix
function getLineColor(text: string): string {
  if (text.includes('[DiT]')) return 'text-pink-400';
  if (text.includes('[LM-Phase1]') || text.includes('[LM-Phase2]')) return 'text-purple-400';
  if (text.includes('[LM-Generate]')) return 'text-purple-300';
  if (text.includes('[Adapter]')) return 'text-cyan-400';
  if (text.includes('[VAE]') || text.includes('vae_decode')) return 'text-emerald-400';
  if (text.includes('[FSQ]')) return 'text-teal-400';
  if (text.includes('[Server]')) return 'text-zinc-600 dark:text-zinc-400';
  if (text.includes('[Mastering]')) return 'text-amber-400';
  if (/ERROR|FAIL/i.test(text)) return 'text-red-400';
  if (/WARNING|WARN/i.test(text)) return 'text-yellow-400';
  return 'text-zinc-700 dark:text-zinc-300';
}

/** Format timestamp once at line creation time, not on every render */
function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  const s = String(d.getSeconds()).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

/** Highlight search matches in a log line */
function highlightMatch(text: string, search: string): React.ReactNode {
  if (!search) return text;
  const idx = text.toLowerCase().indexOf(search.toLowerCase());
  if (idx < 0) return text;
  return (
    <>
      {text.substring(0, idx)}
      <span className="bg-yellow-500/30 text-yellow-200 rounded-sm px-0.5">
        {text.substring(idx, idx + search.length)}
      </span>
      {text.substring(idx + search.length)}
    </>
  );
}

/** Memoized log line — only re-renders when its own id or the search term changes */
const LogLineItem = React.memo<{
  line: LogLine;
  search: string;
}>(({ line, search }) => (
  <div
    className={`whitespace-pre-wrap break-all hover:bg-white/[0.02] px-1 rounded ${getLineColor(line.text)}`}
    style={{ contentVisibility: 'auto', containIntrinsicSize: 'auto 18px' }}
  >
    <span className="text-zinc-600 select-none">
      {formatTimestamp(line.ts)}
    </span>
    {' '}
    {search ? highlightMatch(line.text, search) : line.text}
  </div>
), (prev, next) => prev.line.id === next.line.id && prev.search === next.search);

LogLineItem.displayName = 'LogLineItem';

export const TerminalPanel: React.FC<TerminalPanelProps> = ({ onClose }) => {
  const { lines, connected, clear } = useEventSource('/api/logs', true);
  const [search, setSearch] = useState('');
  const [vram, setVram] = useState<VramInfo | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  // ── Stick-to-bottom pin ──────────────────────────────────────────────────
  // The pin only disengages on PROVABLE user intent (wheel-up, touch, scrollbar
  // drag, scroll keys). Any other displacement while pinned — contentVisibility
  // height corrections, line-cap trimming, panel resizes — is treated as a
  // layout shift and snapped back to the bottom. The previous design guarded
  // programmatic scrolls for one frame and disengaged on anything else, so
  // async layout shifts from content-visibility (estimated 18px vs real wrapped
  // height) silently killed the pin and stranded the view mid-log.
  const [pinned, setPinnedState] = useState(true);
  const pinnedRef = useRef(true);
  const setPinned = useCallback((v: boolean) => {
    pinnedRef.current = v;
    setPinnedState(v);
  }, []);
  // User-intent tracking: recent upward wheel/touch/key, or an active pointer
  // drag (scrollbar). Timestamps beat booleans here — no cleanup races.
  const lastIntentRef = useRef(0);
  const draggingRef = useRef(false);

  const snapToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);

  const pinAndSnap = useCallback(() => {
    setPinned(true);
    snapToBottom();
  }, [setPinned, snapToBottom]);

  // Filter lines by search term
  const filteredLines = useMemo(() => {
    if (!search.trim()) return lines;
    const term = search.toLowerCase();
    return lines.filter(l => l.text.toLowerCase().includes(term));
  }, [lines, search]);

  // Snap on new/changed lines — layout effect so the pinned view never paints
  // at the stale position first.
  useLayoutEffect(() => {
    if (pinnedRef.current) snapToBottom();
  }, [filteredLines, snapToBottom]);

  // Snap on ANY height change of the container or its content: catches
  // content-visibility layout corrections, wrapping changes, and panel resizes
  // that happen without a lines update.
  useEffect(() => {
    const el = scrollRef.current;
    const content = contentRef.current;
    if (!el || !content) return;
    const ro = new ResizeObserver(() => {
      if (pinnedRef.current) snapToBottom();
    });
    ro.observe(el);
    ro.observe(content);
    return () => ro.disconnect();
  }, [snapToBottom]);

  // Re-engage the pin when search is cleared (user finished filtering)
  const prevSearch = useRef(search);
  useEffect(() => {
    if (prevSearch.current && !search) pinAndSnap();
    prevSearch.current = search;
  }, [search, pinAndSnap]);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
    const userIntent = draggingRef.current || performance.now() - lastIntentRef.current < 250;
    if (pinnedRef.current) {
      if (dist < 4) return; // at bottom — all good
      if (userIntent) {
        setPinned(false);   // user deliberately scrolled away
      } else {
        snapToBottom();     // layout shift — correct it, keep the pin
      }
    } else if (dist < 48) {
      setPinned(true);      // user came back to the bottom — re-pin
    }
  }, [setPinned, snapToBottom]);

  // Intent signals. Wheel-up is the classic "let me read old logs" gesture;
  // touch and scroll keys count regardless of direction (downward ones land at
  // the bottom and fall into the re-pin path anyway); pointer-down covers
  // scrollbar drags until release.
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (e.deltaY < 0) lastIntentRef.current = performance.now();
  }, []);
  const handleTouchMove = useCallback(() => {
    lastIntentRef.current = performance.now();
  }, []);
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (['ArrowUp', 'PageUp', 'Home', 'ArrowDown', 'PageDown', 'End', ' '].includes(e.key)) {
      lastIntentRef.current = performance.now();
    }
  }, []);
  const handlePointerDown = useCallback(() => {
    draggingRef.current = true;
    const release = () => {
      draggingRef.current = false;
      window.removeEventListener('pointerup', release);
      window.removeEventListener('pointercancel', release);
    };
    window.addEventListener('pointerup', release);
    window.addEventListener('pointercancel', release);
  }, []);

  // Poll VRAM every 5 seconds
  useEffect(() => {
    const fetchVram = async () => {
      try {
        const res = await fetch('/api/logs/vram');
        if (res.ok) {
          const data = await res.json();
          if (data.total_mb > 0) {
            setVram(data);
          }
        }
      } catch { /* ignore */ }
    };
    fetchVram();
    const interval = setInterval(fetchVram, 5000);
    return () => clearInterval(interval);
  }, []);

  const vramPercent = vram && vram.total_mb > 0
    ? Math.round((vram.used_mb / vram.total_mb) * 100)
    : 0;

  return (
    <div className="h-full flex flex-col overflow-hidden bg-white dark:bg-[#0d0d0f]">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-3 border-b border-zinc-200 dark:border-white/5 bg-zinc-100/80 dark:bg-zinc-900/80">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            {connected
              ? <Wifi size={12} className="text-green-400" />
              : <WifiOff size={12} className="text-red-400" />}
            <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">Terminal</span>
          </div>

          {/* VRAM badge */}
          {vram && vram.total_mb > 0 && (
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-zinc-100/80 dark:bg-zinc-800/80 border border-zinc-200 dark:border-white/5">
              <Cpu size={11} className="text-zinc-500" />
              <div className="flex items-center gap-1">
                <span className="text-[10px] text-zinc-500">VRAM</span>
                <span className={`text-[10px] font-mono font-medium ${
                  vramPercent > 90 ? 'text-red-400' :
                  vramPercent > 70 ? 'text-yellow-400' :
                  'text-emerald-400'
                }`}>
                  {(vram.used_mb / 1024).toFixed(1)} / {(vram.total_mb / 1024).toFixed(1)} GB
                </span>
                <div className="w-12 h-1 rounded-full bg-zinc-200 dark:bg-zinc-700 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      vramPercent > 90 ? 'bg-red-500' :
                      vramPercent > 70 ? 'bg-yellow-500' :
                      'bg-emerald-500'
                    }`}
                    style={{ width: `${vramPercent}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Line count */}
          <span className="text-[10px] text-zinc-600 font-mono">
            {filteredLines.length}{search ? `/${lines.length}` : ''} lines
          </span>
        </div>

        <div className="flex items-center gap-1">
          {/* Auto-scroll pin toggle — always visible */}
          <button
            onClick={() => (pinned ? setPinned(false) : pinAndSnap())}
            className={`p-1 rounded transition-colors ${
              pinned
                ? 'text-blue-400 bg-blue-500/10 hover:bg-blue-500/20'
                : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'
            }`}
            title={pinned ? 'Pinned to bottom (click to unpin)' : 'Unpinned (click to pin to bottom)'}
          >
            {pinned ? <Pin size={12} /> : <PinOff size={12} />}
          </button>
          {/* Jump to bottom (only when not pinned) */}
          {!pinned && (
            <button
              onClick={pinAndSnap}
              className="p-1 rounded text-zinc-500 hover:text-blue-400 hover:bg-blue-500/10 transition-colors"
              title="Jump to bottom"
            >
              <ArrowDown size={12} />
            </button>
          )}
          <button
            onClick={clear}
            className="p-1 rounded text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-white/5 transition-colors"
            title="Clear"
          >
            <Trash2 size={12} />
          </button>
          <button
            onClick={onClose}
            className="p-1 rounded text-zinc-500 hover:text-white hover:bg-white/5 transition-colors"
            title="Close terminal"
          >
            <X size={12} />
          </button>
        </div>
      </div>

      {/* Search bar */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-zinc-200 dark:border-white/5 bg-zinc-50/80 dark:bg-zinc-900/50">
        <Search size={12} className="text-zinc-500 flex-shrink-0" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter logs..."
          className="flex-1 bg-transparent text-xs text-zinc-700 dark:text-zinc-300 placeholder:text-zinc-400 dark:placeholder:text-zinc-600 outline-none font-mono"
        />
        {search && (
          <button
            onClick={() => setSearch('')}
            className="p-0.5 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
          >
            <X size={11} />
          </button>
        )}
      </div>

      {/* Log lines. overflow-anchor: none — the browser's native scroll
          anchoring fights the pin logic (it re-anchors to visible lines when
          heights shift); the pin handles positioning itself. */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        onWheel={handleWheel}
        onTouchMove={handleTouchMove}
        onKeyDown={handleKeyDown}
        onPointerDown={handlePointerDown}
        style={{ overflowAnchor: 'none' }}
        className="flex-1 overflow-y-auto overflow-x-hidden font-mono text-[11px] leading-[18px] p-2
                   scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent"
      >
        <div ref={contentRef}>
          {filteredLines.map((line) => (
            <LogLineItem key={line.id} line={line} search={search} />
          ))}
        </div>
      </div>
    </div>
  );
};
