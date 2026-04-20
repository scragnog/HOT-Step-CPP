/**
 * RightSidebarPanel.tsx — Persistent right-side panel for Lyric Studio V2.
 *
 * Layout varies by navigation level:
 *   album-detail: Generated Songs (1/3) + Recent Songs (1/3) + Queue (1/3)
 *   artists/albums: Recent Songs (1/2) + Queue (1/2)
 */

import React, { useState } from 'react';
import {
  Headphones, Clock, ListOrdered,
  ChevronDown, ChevronRight,
} from 'lucide-react';
import type { Generation } from '../../services/lireekApi';
import { RecordingsTab } from './RecordingsTab';
import { RecentSongsList } from './RecentSongsList';
import { InlineAudioQueue } from './InlineAudioQueue';
import { useAudioGenQueue } from '../../stores/audioGenQueueStore';

type NavLevel = 'artists' | 'albums' | 'album-detail';

interface RightSidebarPanelProps {
  navLevel: NavLevel;
  generations?: Generation[];
  showToast: (msg: string) => void;
  recordingsFilter?: number | null;
  onClearRecordingsFilter?: () => void;
  onSongCountChange?: (count: number) => void;
  recordingsRefreshKey?: number;
  artistName?: string;
}

interface SectionProps {
  title: string;
  icon: React.ReactNode;
  count?: number;
  countColor?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

const Section: React.FC<SectionProps> = ({
  title, icon, count, countColor = 'bg-zinc-700 text-zinc-300', defaultOpen = true, children,
}) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="flex flex-col overflow-hidden"
      style={{ flex: open ? '1 1 0%' : '0 0 auto', minHeight: open ? 0 : 'auto' }}>
      <button onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-4 py-2.5 border-b border-white/5 bg-zinc-950/50 hover:bg-zinc-950/80 transition-colors flex-shrink-0">
        {open ? <ChevronDown className="w-3 h-3 text-zinc-500" /> : <ChevronRight className="w-3 h-3 text-zinc-500" />}
        <span className="flex items-center gap-1.5 text-xs font-semibold text-zinc-400 uppercase tracking-wider">
          {icon} {title}
        </span>
        {count !== undefined && count > 0 && (
          <span className={`min-w-[18px] h-4 px-1 rounded-full text-[10px] font-bold flex items-center justify-center ${countColor}`}>
            {count}
          </span>
        )}
      </button>
      {open && (
        <div className="flex-1 min-h-0 overflow-y-auto scrollbar-hide">
          {children}
        </div>
      )}
    </div>
  );
};

export const RightSidebarPanel: React.FC<RightSidebarPanelProps> = ({
  navLevel, generations, showToast,
  recordingsFilter, onClearRecordingsFilter, onSongCountChange,
  recordingsRefreshKey = 0, artistName,
}) => {
  const queue = useAudioGenQueue();
  const queueCount = queue.items.filter(i => i.status === 'pending' || i.status === 'loading-adapter' || i.status === 'generating').length;
  const recentRefreshKey = queue.completionCounter + recordingsRefreshKey;

  const showRecordings = navLevel === 'album-detail' && generations;

  return (
    <div className="h-full flex flex-col">
      {showRecordings && (
        <Section title="Generated Songs"
          icon={<Headphones className="w-3 h-3" />}
          countColor="bg-pink-500/20 text-pink-300"
          defaultOpen={true}>
          <RecordingsTab
            generations={generations}
            showToast={showToast}
            filterGenerationId={recordingsFilter}
            onClearFilter={onClearRecordingsFilter}
            onSongCountChange={onSongCountChange}
            refreshKey={recordingsRefreshKey}
            artistName={artistName}
          />
        </Section>
      )}

      <Section title="Recent Songs"
        icon={<Clock className="w-3 h-3" />}
        defaultOpen={true}>
        <RecentSongsList
          showToast={showToast}
          refreshKey={recentRefreshKey}
        />
      </Section>

      <Section title="Queue"
        icon={<ListOrdered className="w-3 h-3" />}
        count={queueCount}
        countColor="bg-pink-500/20 text-pink-300"
        defaultOpen={true}>
        <InlineAudioQueue />
      </Section>
    </div>
  );
};
