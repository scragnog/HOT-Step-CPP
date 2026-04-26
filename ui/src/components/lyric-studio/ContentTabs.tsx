import React from 'react';
import { FileText, Users, Music2, Headphones } from 'lucide-react';

export type TabId = 'source-lyrics' | 'profiles' | 'written-songs' | 'recordings';

interface Tab {
  id: TabId;
  label: string;
  icon: React.ReactNode;
  badge?: number;
}

interface ContentTabsProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  sourceLyricsCount?: number;
  profilesCount?: number;
  writtenSongsCount?: number;
  recordingsCount?: number;
  children: React.ReactNode;
}

export const ContentTabs: React.FC<ContentTabsProps> = ({
  activeTab, onTabChange, sourceLyricsCount, profilesCount, writtenSongsCount, recordingsCount, children,
}) => {
  const tabs: Tab[] = [
    { id: 'source-lyrics', label: 'Source Lyrics', icon: <FileText className="w-4 h-4" />, badge: sourceLyricsCount },
    { id: 'profiles', label: 'Profiles', icon: <Users className="w-4 h-4" />, badge: profilesCount },
    { id: 'written-songs', label: 'Generated Lyrics', icon: <Music2 className="w-4 h-4" />, badge: writtenSongsCount },
    { id: 'recordings', label: 'Generated Songs', icon: <Headphones className="w-4 h-4" />, badge: recordingsCount },
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Tab bar */}
      <div className="flex-shrink-0 border-b border-white/5 bg-zinc-950/30">
        <div className="flex">
          {tabs.map(tab => {
            const isActive = tab.id === activeTab;
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`flex items-center gap-2 px-5 py-3 text-sm font-medium transition-all border-b-2 ${
                  isActive
                    ? 'text-pink-400 border-pink-500 bg-pink-500/5'
                    : 'text-zinc-400 border-transparent hover:text-zinc-200 hover:bg-white/[0.02]'
                }`}
              >
                {tab.icon}
                <span className="hidden sm:inline">{tab.label}</span>
                {tab.badge != null && tab.badge > 0 && (
                  <span className={`min-w-[20px] h-5 px-1.5 rounded-full text-[11px] font-bold flex items-center justify-center ${
                    isActive ? 'bg-pink-500/20 text-pink-300' : 'bg-white/10 text-zinc-400'
                  }`}>
                    {tab.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab content */}
      <div key={activeTab} className="flex-1 overflow-y-auto ls2-tab-content">
        {children}
      </div>
    </div>
  );
};
