/**
 * CoverSidebarPanel.tsx — Right sidebar for Cover Studio.
 *
 * Layout: Recent Covers (top half) + Queue (bottom half)
 * Mirrors Lyric Studio's RightSidebarPanel pattern.
 */

import React, { useState } from 'react';
import {
  Clock, ListOrdered,
  ChevronDown, ChevronRight,
} from 'lucide-react';
import { CoverRecentSongs } from './CoverRecentSongs';
import { InlineAudioQueue } from '../lyric-studio/InlineAudioQueue';
import { useAudioGenQueue } from '../../stores/audioGenQueueStore';

interface CoverSidebarPanelProps {
  showToast: (msg: string) => void;
  refreshKey?: number;
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
        className="flex items-center gap-2 px-4 py-3 min-h-[44px] border-b border-white/5 bg-zinc-950/50 hover:bg-zinc-950/80 transition-colors flex-shrink-0">
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

export const CoverSidebarPanel: React.FC<CoverSidebarPanelProps> = ({
  showToast, refreshKey = 0,
}) => {
  const queue = useAudioGenQueue();
  const queueCount = queue.items.filter(i => i.status === 'pending' || i.status === 'loading-adapter' || i.status === 'generating').length;

  return (
    <div className="h-full flex flex-col">

      <Section title="Recent Covers"
        icon={<Clock className="w-3 h-3" />}
        defaultOpen={true}>
        <CoverRecentSongs
          showToast={showToast}
          refreshKey={refreshKey}
        />
      </Section>

      <Section title="Queue"
        icon={<ListOrdered className="w-3 h-3" />}
        count={queueCount}
        countColor="bg-cyan-500/20 text-cyan-300"
        defaultOpen={true}>
        <InlineAudioQueue />
      </Section>
    </div>
  );
};
