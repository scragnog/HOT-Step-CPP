/**
 * ActivitySidebar.tsx — the app's right-hand activity column.
 *
 * Three boxed, collapsible, vertically resizable sections: Playlist / Recent
 * Songs (tabbed), the generation Queue, and the engine Terminal. Mounted once
 * in App.tsx for every view, so the queue and the log are always one glance
 * away. Section heights persist; the last open section absorbs the remainder.
 */

import React, { useCallback, useRef, useState } from 'react';
import {
  ListOrdered, TerminalSquare,
  ChevronDown, ChevronRight,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { UnifiedRecentSongs } from './UnifiedRecentSongs';
import { InlineAudioQueue } from '../lyric-studio/InlineAudioQueue';
import { useAudioGenQueueSelector } from '../../stores/audioGenQueueStore';
import { PlaylistSidebar } from '../playlist/PlaylistSidebar';
import { TerminalPanel } from '../terminal/TerminalPanel';
import { usePlaylist } from '../lyric-studio/playlistStore';
import { usePersistedState } from '../../hooks/usePersistedState';

/** Smallest a section may be dragged to, and the room left for the one below. */
const MIN_SECTION = 90;

// ── Shared Section Container ─────────────────────────────────────────────────

interface SectionTab {
  id: string;
  label: string;
  count?: number;
}

interface SectionProps {
  title?: string;
  icon?: React.ReactNode;
  count?: number;
  countColor?: string;
  defaultOpen?: boolean;
  /** Render a tab strip in the header instead of a single title. */
  tabs?: SectionTab[];
  activeTab?: string;
  onTabChange?: (id: string) => void;
  /** Controlled collapse. Falls back to internal state when omitted. */
  open?: boolean;
  onToggle?: () => void;
  /** Draw the box chrome — off for the legacy in-panel call sites. */
  boxed?: boolean;
  children: React.ReactNode;
}

export const Section: React.FC<SectionProps> = ({
  title, icon, count, countColor = 'bg-zinc-200 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300', defaultOpen = true,
  tabs, activeTab, onTabChange, open: openProp, onToggle, boxed = false, children,
}) => {
  const [openState, setOpenState] = useState(defaultOpen);
  const open = openProp ?? openState;
  const toggle = onToggle ?? (() => setOpenState(v => !v));

  return (
    <div
      className={`flex flex-col overflow-hidden ${boxed
        ? 'h-full rounded-xl border border-zinc-200 dark:border-white/10 bg-white dark:bg-zinc-900/60 shadow-sm'
        : ''}`}
      style={boxed ? undefined : { flex: open ? '1 1 0%' : '0 0 auto', minHeight: open ? 0 : 'auto' }}>
      <div
        onClick={toggle}
        className="flex items-center gap-2 px-4 py-3 min-h-[44px] border-b border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-zinc-950/50 hover:bg-zinc-50/80 dark:hover:bg-zinc-950/80 transition-colors flex-shrink-0 cursor-pointer select-none">
        {open ? <ChevronDown className="w-3 h-3 text-zinc-500 flex-shrink-0" /> : <ChevronRight className="w-3 h-3 text-zinc-500 flex-shrink-0" />}

        {tabs ? (
          <div className="flex items-center gap-1 min-w-0">
            {tabs.map(tab => {
              const active = tab.id === activeTab;
              return (
                <button key={tab.id}
                  onClick={(e) => { e.stopPropagation(); if (!open) toggle(); onTabChange?.(tab.id); }}
                  className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-semibold uppercase tracking-wider transition-colors whitespace-nowrap ${
                    active
                      ? 'text-pink-500 dark:text-pink-400 bg-pink-500/10'
                      : 'text-zinc-500 dark:text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-white/5'
                  }`}>
                  {tab.label}
                  {tab.count !== undefined && tab.count > 0 && (
                    <span className="min-w-[16px] h-4 px-1 rounded-full text-[10px] font-bold flex items-center justify-center bg-pink-500/20 text-pink-400">
                      {tab.count}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        ) : (
          <>
            <span className="flex items-center gap-1.5 text-xs font-semibold text-zinc-600 dark:text-zinc-400 uppercase tracking-wider">
              {icon} {title}
            </span>
            {count !== undefined && count > 0 && (
              <span className={`min-w-[18px] h-4 px-1 rounded-full text-[10px] font-bold flex items-center justify-center ${countColor}`}>
                {count}
              </span>
            )}
          </>
        )}
      </div>
      {open && (
        <div className="flex-1 min-h-0 overflow-y-auto scrollbar-hide">
          {children}
        </div>
      )}
    </div>
  );
};

// ── ActivitySidebar ──────────────────────────────────────────────────────────

interface ActivitySidebarProps {
  showToast: (msg: string, type?: 'success' | 'error') => void;
  /** Filter recent songs by source — e.g. 'lyric-studio', 'cover-studio', 'create', or undefined for all */
  source?: string;
  /** External refresh trigger (e.g. when a generation completes elsewhere) */
  refreshKey?: number;
  /** Use single-column layout for recent songs */
  compact?: boolean;
  /** Color accent for queue count badge */
  queueCountColor?: string;
  /** Show the Terminal section at the bottom of the column. */
  showTerminal?: boolean;
}

export const ActivitySidebar: React.FC<ActivitySidebarProps> = ({
  showToast,
  source,
  refreshKey = 0,
  compact = false,
  queueCountColor = 'bg-pink-500/20 text-pink-300',
  showTerminal = false,
}) => {
  const { t } = useTranslation();
  const queueCount = useAudioGenQueueSelector(s =>
    s.items.filter(i => i.status === 'pending' || i.status === 'loading-adapter' || i.status === 'generating').length
  );
  const completionCounter = useAudioGenQueueSelector(s => s.completionCounter);
  const playlistCount = usePlaylist().items.length;
  const [topTab, setTopTab] = useState<'playlist' | 'recent'>('playlist');

  // Combine external refresh key with queue completion counter
  const recentRefreshKey = completionCounter + refreshKey;

  // ── Section sizing ──
  // Heights are keyed by section name, not by index, so hiding one section does
  // not shuffle the others into each other's saved size.
  const [heights, setHeights] = usePersistedState<Record<string, number>>(
    'hs-activitySectionHeights', { top: 340, queue: 220 },
  );
  const [openMap, setOpenMap] = usePersistedState<Record<string, boolean>>(
    'hs-activitySectionOpen', { top: true, queue: true, terminal: true },
  );
  const columnRef = useRef<HTMLDivElement>(null);

  const isOpen = (key: string) => openMap[key] !== false;
  const toggle = (key: string) => setOpenMap(m => ({ ...m, [key]: m[key] === false }));

  /** Drag the boundary below `key`, growing or shrinking that section. */
  const startResize = useCallback((key: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    const startY = e.clientY;
    const startH = columnRef.current
      ?.querySelector<HTMLElement>(`[data-section="${key}"]`)?.offsetHeight ?? MIN_SECTION;
    const columnH = columnRef.current?.offsetHeight ?? 0;

    const onMove = (ev: MouseEvent) => {
      const next = Math.max(MIN_SECTION, Math.min(columnH - MIN_SECTION, startH + (ev.clientY - startY)));
      setHeights(h => ({ ...h, [key]: next }));
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [setHeights]);

  const sections: { key: string; node: React.ReactNode }[] = [
    {
      key: 'top',
      node: (
        <Section boxed
          open={isOpen('top')} onToggle={() => toggle('top')}
          tabs={[
            { id: 'playlist', label: t('playlist.title'), count: playlistCount },
            { id: 'recent', label: t('activity.recentSongs') },
          ]}
          activeTab={topTab}
          onTabChange={(id) => setTopTab(id as 'playlist' | 'recent')}
        >
          {topTab === 'playlist' ? (
            <PlaylistSidebar embedded />
          ) : (
            <UnifiedRecentSongs
              showToast={showToast}
              refreshKey={recentRefreshKey}
              compact={compact}
              source={source}
            />
          )}
        </Section>
      ),
    },
    {
      key: 'queue',
      node: (
        <Section boxed title={t('activity.queue')}
          icon={<ListOrdered className="w-3 h-3" />}
          count={queueCount}
          countColor={queueCountColor}
          open={isOpen('queue')} onToggle={() => toggle('queue')}>
          <InlineAudioQueue />
        </Section>
      ),
    },
  ];

  if (showTerminal) {
    sections.push({
      key: 'terminal',
      node: (
        <Section boxed title="Terminal"
          icon={<TerminalSquare className="w-3 h-3" />}
          open={isOpen('terminal')} onToggle={() => toggle('terminal')}>
          <TerminalPanel embedded />
        </Section>
      ),
    });
  }

  return (
    <div ref={columnRef} className="h-full flex flex-col gap-1 p-1.5 bg-zinc-100 dark:bg-zinc-950 overflow-hidden">
      {sections.map((section, i) => {
        const last = i === sections.length - 1;
        const open = isOpen(section.key);
        // The last section absorbs whatever is left, so it has no handle below
        // it — its size is set by dragging the boundary above.
        const draggable = !last && open;
        return (
          <React.Fragment key={section.key}>
            <div data-section={section.key} className="flex flex-col overflow-hidden"
              style={{
                flex: !open ? '0 0 auto' : last ? '1 1 0%' : `0 0 ${heights[section.key] ?? 240}px`,
                minHeight: 0,
              }}>
              {section.node}
            </div>
            {draggable && (
              <div
                onMouseDown={startResize(section.key)}
                className="flex-shrink-0 h-1.5 w-full cursor-row-resize group z-20 flex justify-center items-center hover:bg-pink-500/20 active:bg-pink-500/30 rounded transition-colors"
              >
                <div className="h-0.5 w-8 rounded-full bg-zinc-600 group-hover:bg-pink-400 transition-colors" />
              </div>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};
