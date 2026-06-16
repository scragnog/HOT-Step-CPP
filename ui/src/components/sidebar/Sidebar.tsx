// Sidebar.tsx — Left navigation sidebar
// Ported from hot-step-9000, simplified for current cpp feature set.

import React from 'react';
import { Disc, Library, Mic, Guitar, Paintbrush, Scissors, Layers, Blocks, Settings, Power, Terminal, RotateCcw, Sun, Moon, Sparkles, Wand2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { usePersistedState } from '../../hooks/usePersistedState';

interface SidebarProps {
  activeView: string;
  onViewChange: (view: string) => void;
  onQuit: () => void;
  onRestart?: () => void;
  theme?: 'dark' | 'light';
  onToggleTheme?: () => void;
  showTerminal?: boolean;
  onToggleTerminal?: () => void;
  showAssistant?: boolean;
  onToggleAssistant?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeView,
  onViewChange,
  onQuit,
  onRestart,
  theme = 'dark',
  onToggleTheme,
  showTerminal = false,
  onToggleTerminal,
  showAssistant = false,
  onToggleAssistant,
}) => {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = usePersistedState('hs-sidebar-open', true);

  return (
    <div className={`
      flex flex-col h-full bg-white dark:bg-suno-sidebar border-r border-zinc-200 dark:border-white/5 flex-shrink-0 py-4 overflow-y-auto hide-scrollbar transition-all duration-300
      ${isOpen ? 'w-[200px]' : 'w-[72px]'}
    `}>
      {/* Collapse / Expand toggle (logo moved to GlobalParamBar) */}
      <div className={`mb-6 flex items-center ${isOpen ? 'px-3' : 'justify-center'}`}>
        <button
          className="w-8 h-8 rounded-lg bg-zinc-200 dark:bg-white/5 hover:bg-zinc-300 dark:hover:bg-white/10 flex items-center justify-center transition-all flex-shrink-0"
          onClick={() => setIsOpen(!isOpen)}
          title={isOpen ? t('sidebar.collapse') : t('sidebar.expand')}
        >
          <svg className={`w-4 h-4 text-zinc-600 dark:text-zinc-400 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </div>

      <nav className="flex-1 flex flex-col gap-2 w-full px-3">
        {/* ── Create section — framed by purple bars ── */}
        {isOpen && <div className="h-[3px] bg-purple-500/40 mx-1 mt-1" />}
        {isOpen && (
          <div className="text-center -mb-1">
            <span className="text-[1.2rem] font-bold text-purple-400 tracking-wide">{t('sidebar.createSection')}</span>
          </div>
        )}
        <NavItem
          icon={<Wand2 size={20} />}
          label={t('sidebar.instaGen')}
          active={activeView === 'insta-gen'}
          onClick={() => onViewChange('insta-gen')}
          isExpanded={isOpen}
        />
        <NavItem
          icon={<Disc size={20} />}
          label={t('sidebar.create')}
          active={activeView === 'create'}
          onClick={() => onViewChange('create')}
          isExpanded={isOpen}
        />
        {isOpen && <div className="h-[3px] bg-purple-500/40 mx-1" />}
        {!isOpen && <div className="h-px bg-white/5 mx-2" />}
        <div className={isOpen ? 'mb-1' : 'mb-0'} />
        <NavItem
          icon={<Library size={20} />}
          label={t('sidebar.library')}
          active={activeView === 'library'}
          onClick={() => onViewChange('library')}
          isExpanded={isOpen}
        />
        <NavItem
          icon={<Mic size={20} />}
          label={t('sidebar.lyricStudio')}
          active={activeView === 'lyric-studio'}
          onClick={() => onViewChange('lyric-studio')}
          isExpanded={isOpen}
        />

        <NavItem
          icon={<Guitar size={20} />}
          label={t('sidebar.coverStudio')}
          active={activeView === 'cover-studio'}
          onClick={() => onViewChange('cover-studio')}
          isExpanded={isOpen}
        />

        <NavItem
          icon={<Paintbrush size={20} />}
          label={t('sidebar.repaint')}
          active={activeView === 'repaint'}
          onClick={() => onViewChange('repaint')}
          isExpanded={isOpen}
        />

        <NavItem
          icon={<Scissors size={20} />}
          label={t('sidebar.stemStudio')}
          active={activeView === 'stem-studio'}
          onClick={() => onViewChange('stem-studio')}
          isExpanded={isOpen}
        />

        <NavItem
          icon={<Layers size={20} />}
          label={t('sidebar.stemBuilder')}
          active={activeView === 'stem-builder'}
          onClick={() => onViewChange('stem-builder')}
          isExpanded={isOpen}
        />

        <NavItem
          icon={<Blocks size={20} />}
          label={t('sidebar.songBuilder')}
          active={activeView === 'song-builder'}
          onClick={() => onViewChange('song-builder')}
          isExpanded={isOpen}
        />

        <NavItem
          icon={<Settings size={20} />}
          label={t('sidebar.settings')}
          active={activeView === 'settings'}
          onClick={() => onViewChange('settings')}
          isExpanded={isOpen}
        />

        {onToggleTheme && (
          <button onClick={onToggleTheme}
            className={`w-full rounded-xl flex items-center gap-3 transition-all duration-200
              ${isOpen ? 'px-3 py-2.5 justify-start' : 'aspect-square justify-center'}
              text-zinc-500 hover:text-black dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-white/5`}
            title={theme === 'dark' ? t('sidebar.theme.switchToLight') : t('sidebar.theme.switchToDark')}>
            <div className="flex-shrink-0">
              {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
            </div>
            {isOpen && <span className="text-sm font-medium whitespace-nowrap">
              {theme === 'dark' ? t('sidebar.theme.lightMode') : t('sidebar.theme.darkMode')}
            </span>}
          </button>
        )}

        <div className="mt-auto flex flex-col gap-2">
          {/* Terminal toggle */}
          {onToggleTerminal && (
            <button
              onClick={onToggleTerminal}
              className={`
                w-full rounded-xl flex items-center gap-3 transition-all duration-200
                ${isOpen ? 'px-3 py-2.5 justify-start' : 'aspect-square justify-center'}
                ${showTerminal
                  ? 'bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/15'
                  : 'text-zinc-500 hover:text-zinc-700 dark:text-zinc-300 hover:bg-white/5'}
              `}
              title={showTerminal ? t('sidebar.terminal.hide') : t('sidebar.terminal.show')}
            >
              <div className="flex-shrink-0"><Terminal size={20} /></div>
              {isOpen && (
                <span className="text-sm font-medium whitespace-nowrap">{t('sidebar.terminal.label')}</span>
              )}
            </button>
          )}

          {/* Assistant toggle */}
          {onToggleAssistant && (
            <button
              onClick={onToggleAssistant}
              className={`
                w-full rounded-xl flex items-center gap-3 transition-all duration-200
                ${isOpen ? 'px-3 py-2.5 justify-start' : 'aspect-square justify-center'}
                ${showAssistant
                  ? 'bg-violet-500/10 text-violet-400 hover:bg-violet-500/15'
                  : 'text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-white/5'}
              `}
              title={showAssistant ? t('sidebar.assistant.hide') : t('sidebar.assistant.show')}
            >
              <div className="flex-shrink-0"><Sparkles size={20} /></div>
              {isOpen && (
                <span className="text-sm font-medium whitespace-nowrap">{t('sidebar.assistant.label')}</span>
              )}
            </button>
          )}

          {/* Restart */}
          {onRestart && (
            <button
              onClick={onRestart}
              className={`
                w-full rounded-xl flex items-center gap-3 transition-all duration-200 text-amber-400 hover:text-amber-300 hover:bg-amber-500/10
                ${isOpen ? 'px-3 py-2.5 justify-start' : 'aspect-square justify-center'}
              `}
              title={t('sidebar.restartTitle')}
            >
              <div className="flex-shrink-0"><RotateCcw size={20} /></div>
              {isOpen && (
                <span className="text-sm font-medium whitespace-nowrap">{t('sidebar.restart')}</span>
              )}
            </button>
          )}

          {/* Quit / Shutdown */}
          <button
            onClick={onQuit}
            className={`
              w-full rounded-xl flex items-center gap-3 transition-all duration-200 text-red-400 hover:text-red-300 hover:bg-red-500/10
              ${isOpen ? 'px-3 py-2.5 justify-start' : 'aspect-square justify-center'}
            `}
            title={t('sidebar.quitTitle')}
          >
            <div className="flex-shrink-0"><Power size={20} /></div>
            {isOpen && (
              <span className="text-sm font-medium whitespace-nowrap">{t('sidebar.quit')}</span>
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

