// Sidebar.tsx — Left navigation sidebar
// Ported from hot-step-9000, simplified for current cpp feature set.

import React from 'react';
import { Disc, Library, Settings, Power } from 'lucide-react';

interface SidebarProps {
  activeView: string;
  onViewChange: (view: string) => void;
  onQuit: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeView,
  onViewChange,
  onQuit,
}) => {
  const [isOpen, setIsOpen] = React.useState(true);

  return (
    <div className={`
      flex flex-col h-full bg-white dark:bg-suno-sidebar border-r border-zinc-200 dark:border-white/5 flex-shrink-0 py-4 overflow-y-auto hide-scrollbar transition-all duration-300
      ${isOpen ? 'w-[200px]' : 'w-[72px]'}
    `}>
      {/* Logo & Brand */}
      <div className={`mb-8 flex items-center ${isOpen ? 'px-3' : 'justify-center'}`}>
        <div className="flex items-center gap-3">
          <button
            className="w-10 h-10 rounded-full bg-gradient-to-br from-pink-500 to-purple-600 flex items-center justify-center shadow-lg hover:scale-105 transition-transform flex-shrink-0"
            onClick={() => setIsOpen(!isOpen)}
            title={isOpen ? 'Collapse' : 'HOT-Step CPP'}
          >
            <svg className={`w-5 h-5 text-white transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
          {isOpen && (
            <div className="flex flex-col items-center leading-tight">
              <span className="text-lg font-bold text-zinc-900 dark:text-white whitespace-nowrap">HOT-Step</span>
              <span className="text-xs font-semibold tracking-[0.25em] text-zinc-500 dark:text-zinc-400">CPP ⚡</span>
            </div>
          )}
        </div>
      </div>

      <nav className="flex-1 flex flex-col gap-2 w-full px-3">
        <NavItem
          icon={<Disc size={20} />}
          label="Create"
          active={activeView === 'create'}
          onClick={() => onViewChange('create')}
          isExpanded={isOpen}
        />
        <NavItem
          icon={<Library size={20} />}
          label="Library"
          active={activeView === 'library'}
          onClick={() => onViewChange('library')}
          isExpanded={isOpen}
        />

        <NavItem
          icon={<Settings size={20} />}
          label="Settings"
          active={activeView === 'settings'}
          onClick={() => onViewChange('settings')}
          isExpanded={isOpen}
        />

        <div className="mt-auto flex flex-col gap-2">
          {/* Quit / Shutdown */}
          <button
            onClick={onQuit}
            className={`
              w-full rounded-xl flex items-center gap-3 transition-all duration-200 text-red-400 hover:text-red-300 hover:bg-red-500/10
              ${isOpen ? 'px-3 py-2.5 justify-start' : 'aspect-square justify-center'}
            `}
            title="Quit HOT-Step CPP"
          >
            <div className="flex-shrink-0"><Power size={20} /></div>
            {isOpen && (
              <span className="text-sm font-medium whitespace-nowrap">Quit</span>
            )}
          </button>
        </div>
      </nav>
    </div>
  );
};

interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  onClick: () => void;
  isExpanded?: boolean;
}

const NavItem: React.FC<NavItemProps> = ({ icon, label, active, onClick, isExpanded }) => (
  <button
    onClick={onClick}
    className={`
      w-full rounded-xl flex items-center gap-3 transition-all duration-200 group relative overflow-hidden
      ${isExpanded ? 'px-3 py-2.5 justify-start' : 'aspect-square justify-center'}
      ${active ? 'bg-zinc-100 dark:bg-white/10 text-black dark:text-white' : 'text-zinc-500 hover:text-black dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-white/5'}
    `}
    title={label}
  >
    {active && <div className="absolute left-0 top-1/2 -translate-y-1/2 h-8 w-1 bg-pink-500 rounded-r-full"></div>}
    <div className="flex-shrink-0">{icon}</div>
    {isExpanded && (
      <span className="text-sm font-medium whitespace-nowrap">{label}</span>
    )}
  </button>
);
