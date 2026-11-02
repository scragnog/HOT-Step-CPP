// Sidebar.tsx — Left navigation sidebar
// Ported from hot-step-9000, simplified for current cpp feature set.

import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Disc, Library, Mic, Guitar, Paintbrush, Scissors, Layers, Blocks, Settings, Power, RotateCcw, Sun, Moon, Sparkles, Wand2, Zap, Piano, GraduationCap, X, GitCompare } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { usePersistedState } from '../../hooks/usePersistedState';
import { useJobBadge } from '../../hooks/useJobProgress';
import { useCapabilities } from '../../hooks/useCapabilities';
import { useBackendStore, type BackendCapabilities } from '../../stores/backendStore';
import { BackendToggle } from '../global-bar/BackendToggle';

// ── Studios per generation engine ───────────────────────────────────────────
// A studio that only works on some engines is listed only while one of them
// is active. Unlisted views (Auto-Gen, Custom-Gen, Library, Lyric Studio,
// Stem Separator, Compare, MIDI Studio, Settings) work on every engine.
// Unknown capabilities (still loading, or the engine is down and reports
// every feature off) list everything rather than flicker the menu.
const STUDIO_NEEDS: Record<string, (caps: BackendCapabilities, backendId: string) => boolean> = {
  'cover-studio': caps => caps.features.cover,
  'repaint': caps => caps.features.repaint,
  'stem-builder': caps => caps.features.stems,
  // Section regeneration is repaint outpainting.
  'song-builder': caps => caps.features.repaint,
  'storm': caps => caps.features.streaming,
};

/** Which views the active engine supports. */
export function useStudioAvailable(): (view: string) => boolean {
  const { capabilities } = useCapabilities();
  const backendId = useBackendStore(s => s.activeBackendId);
  return (view: string) => {
    const needs = STUDIO_NEEDS[view];
    if (!needs || !capabilities || !capabilities.up) return true;
    return needs(capabilities, backendId);
  };
}

interface SidebarProps {
  activeView: string;
  onViewChange: (view: string) => void;
  onQuit: () => void;
  onRestart?: () => void;
  theme?: 'dark' | 'light';
  onToggleTheme?: () => void;
  showAssistant?: boolean;
  onToggleAssistant?: () => void;
  /** Mobile drawer: controlled open state (ignored on desktop >= 1024px) */
  mobileOpen?: boolean;
  /** Called when the mobile drawer should close (backdrop tap, nav item tap, X button) */
  onMobileClose?: () => void;
}

// ── Shared nav content ──────────────────────────────────────────────────────
// Rendered both inside the desktop sidebar and the mobile overlay drawer.
// `isExpanded` controls icon-only vs icon+label mode.
// `onNav` is the navigation callback (may also close the mobile drawer).

interface SidebarNavProps {
  activeView: string;
  isExpanded: boolean;
  onNav: (view: string) => void;
  onQuit: () => void;
  onRestart?: () => void;
  theme: 'dark' | 'light';
  onToggleTheme?: () => void;
  showAssistant: boolean;
  onToggleAssistant?: () => void;
}

const SidebarNav: React.FC<SidebarNavProps> = ({
  activeView,
  isExpanded,
  onNav,
  onQuit,
  onRestart,
  theme,
  onToggleTheme,
  showAssistant,
  onToggleAssistant,
}) => {
  const { t } = useTranslation();
  const jobBadge = useJobBadge();
  const available = useStudioAvailable();
  return (
    <nav className="flex-1 flex flex-col min-h-0 w-full min-w-0">
      {/* Items scroll on short screens; the account/actions footer below stays
          pinned so Theme/Assistant/Restart/Quit are always reachable. */}
      <div className="flex-1 min-h-0 overflow-y-auto sidebar-scroll flex flex-col gap-2 short:gap-1 w-full min-w-0 px-3 pb-2">
      {/* ── Create section — framed by purple bars ── */}
      {isExpanded && <div className="h-[3px] flex-shrink-0 bg-purple-500/40 mx-1 mt-1" />}
      {isExpanded && (
        <div className="text-center -mb-1 flex-shrink-0">
          <span className="text-[1.2rem] font-bold text-purple-400 tracking-wide">{t('sidebar.createSection')}</span>
        </div>
      )}
      <NavItem
        icon={<Wand2 size={20} />}
        label={t('sidebar.instaGen')}
        active={activeView === 'insta-gen'}
        onClick={() => onNav('insta-gen')}
        isExpanded={isExpanded}
      />
      <NavItem
        icon={<Disc size={20} />}
        label={t('sidebar.create')}
        active={activeView === 'create'}
        onClick={() => onNav('create')}
        isExpanded={isExpanded}
        badge={jobBadge}
      />
      {isExpanded && <div className="h-[3px] flex-shrink-0 bg-purple-500/40 mx-1" />}
      {!isExpanded && <div className="h-px flex-shrink-0 bg-white/5 mx-2" />}
      <div className={`flex-shrink-0 ${isExpanded ? 'mb-1' : 'mb-0'}`} />
      <NavItem
        icon={<Library size={20} />}
        label={t('sidebar.library')}
        active={activeView === 'library'}
        onClick={() => onNav('library')}
        isExpanded={isExpanded}
      />
      <NavItem
        icon={<Mic size={20} />}
        label={t('sidebar.lyricStudio')}
        active={activeView === 'lyric-studio'}
        onClick={() => onNav('lyric-studio')}
        isExpanded={isExpanded}
      />

      {available('cover-studio') && <NavItem
        icon={<Guitar size={20} />}
        label={t('sidebar.coverStudio')}
        active={activeView === 'cover-studio'}
        onClick={() => onNav('cover-studio')}
        isExpanded={isExpanded}
      />}

      {available('repaint') && <NavItem
        icon={<Paintbrush size={20} />}
        label={t('sidebar.repaint')}
        active={activeView === 'repaint'}
        onClick={() => onNav('repaint')}
        isExpanded={isExpanded}
      />}

      <NavItem
        icon={<Scissors size={20} />}
        label={t('sidebar.stemStudio')}
        active={activeView === 'stem-studio'}
        onClick={() => onNav('stem-studio')}
        isExpanded={isExpanded}
      />

      <NavItem
        icon={<GitCompare size={20} />}
        label={t('sidebar.compare')}
        active={activeView === 'compare-studio'}
        onClick={() => onNav('compare-studio')}
        isExpanded={isExpanded}
      />

      {available('stem-builder') && <NavItem
        icon={<Layers size={20} />}
        label={t('sidebar.stemBuilder')}
        active={activeView === 'stem-builder'}
        onClick={() => onNav('stem-builder')}
        isExpanded={isExpanded}
      />}

      {available('song-builder') && <NavItem
        icon={<Blocks size={20} />}
        label={t('sidebar.songBuilder')}
        active={activeView === 'song-builder'}
        onClick={() => onNav('song-builder')}
        isExpanded={isExpanded}
      />}

      {available('storm') && <NavItem
        icon={<Zap size={20} />}
        label={t('sidebar.storm')}
        active={activeView === 'storm'}
        onClick={() => onNav('storm')}
        isExpanded={isExpanded}
      />}

      <NavItem
        icon={<Piano size={20} />}
        label={t('sidebar.midiStudio')}
        active={activeView === 'midi-studio'}
        onClick={() => onNav('midi-studio')}
        isExpanded={isExpanded}
      />

      {available('training-studio') && <NavItem
        icon={<GraduationCap size={20} />}
        label={t('sidebar.trainingStudio')}
        active={activeView === 'training-studio'}
        onClick={() => onNav('training-studio')}
        isExpanded={isExpanded}
      />}

      <NavItem
        icon={<Settings size={20} />}
        label={t('sidebar.settings')}
        active={activeView === 'settings'}
        onClick={() => onNav('settings')}
        isExpanded={isExpanded}
      />
      </div>

      {/* Pinned footer — always on screen, even when the nav list scrolls. */}
      <div className={`flex-shrink-0 w-full min-w-0 px-3 pt-2 border-t border-zinc-200/70 dark:border-white/5 flex flex-col gap-2 ${isExpanded ? 'short:grid short:grid-cols-4 short:gap-1' : 'short:grid short:grid-cols-2 short:gap-1'}`}>
        {onToggleTheme && (
          <button onClick={onToggleTheme}
            className={`w-full flex-shrink-0 rounded-xl flex items-center gap-3 transition-all duration-200
              ${isExpanded ? 'px-3 py-2.5 short:py-1.5 justify-start short:justify-center short:px-0' : 'aspect-square justify-center'}
              text-zinc-500 hover:text-black dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-white/5`}
            title={theme === 'dark' ? t('sidebar.theme.switchToLight') : t('sidebar.theme.switchToDark')}
            aria-label={theme === 'dark' ? t('sidebar.theme.switchToLight') : t('sidebar.theme.switchToDark')}>
            <div className="flex-shrink-0">
              {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
            </div>
            {isExpanded && <span className="text-sm font-medium truncate min-w-0 short:hidden">
              {theme === 'dark' ? t('sidebar.theme.lightMode') : t('sidebar.theme.darkMode')}
            </span>}
          </button>
        )}

        {/* Assistant toggle */}
        {onToggleAssistant && (
          <button
            onClick={onToggleAssistant}
            className={`
              w-full flex-shrink-0 rounded-xl flex items-center gap-3 transition-all duration-200
              ${isExpanded ? 'px-3 py-2.5 short:py-1.5 justify-start short:justify-center short:px-0' : 'aspect-square justify-center'}
              ${showAssistant
                ? 'bg-violet-500/10 text-violet-400 hover:bg-violet-500/15'
                : 'text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-white/5'}
            `}
            title={showAssistant ? t('sidebar.assistant.hide') : t('sidebar.assistant.show')}
            aria-label={showAssistant ? t('sidebar.assistant.hide') : t('sidebar.assistant.show')}
          >
            <div className="flex-shrink-0"><Sparkles size={20} /></div>
            {isExpanded && (
              <span className="text-sm font-medium truncate min-w-0 short:hidden">{t('sidebar.assistant.label')}</span>
            )}
          </button>
        )}

        {/* Restart */}
        {onRestart && (
          <button
            onClick={onRestart}
            className={`
              w-full flex-shrink-0 rounded-xl flex items-center gap-3 transition-all duration-200 text-amber-400 hover:text-amber-300 hover:bg-amber-500/10
              ${isExpanded ? 'px-3 py-2.5 short:py-1.5 justify-start short:justify-center short:px-0' : 'aspect-square justify-center'}
            `}
            title={t('sidebar.restartTitle')}
            aria-label={t('sidebar.restartTitle')}
          >
            <div className="flex-shrink-0"><RotateCcw size={20} /></div>
            {isExpanded && (
              <span className="text-sm font-medium truncate min-w-0 short:hidden">{t('sidebar.restart')}</span>
            )}
          </button>
        )}

        {/* Quit / Shutdown */}
        <button
          onClick={onQuit}
          className={`
            w-full flex-shrink-0 rounded-xl flex items-center gap-3 transition-all duration-200 text-red-400 hover:text-red-300 hover:bg-red-500/10
            ${isExpanded ? 'px-3 py-2.5 short:py-1.5 justify-start short:justify-center short:px-0' : 'aspect-square justify-center'}
          `}
          title={t('sidebar.quitTitle')}
          aria-label={t('sidebar.quitTitle')}
        >
          <div className="flex-shrink-0"><Power size={20} /></div>
          {isExpanded && (
            <span className="text-sm font-medium truncate min-w-0 short:hidden">{t('sidebar.quit')}</span>
          )}
        </button>
      </div>
    </nav>
  );
};

// ── Main Sidebar component ──────────────────────────────────────────────────

export const Sidebar: React.FC<SidebarProps> = ({
  activeView,
  onViewChange,
  onQuit,
  onRestart,
  theme = 'dark',
  onToggleTheme,
  showAssistant = false,
  onToggleAssistant,
  mobileOpen = false,
  onMobileClose,
}) => {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = usePersistedState('hs-sidebar-open', true);

  // Switching to an engine that does not list the open studio moves to Custom-Gen.
  const available = useStudioAvailable();
  const activeAvailable = available(activeView);
  useEffect(() => {
    if (!activeAvailable) onViewChange('create');
  }, [activeAvailable]); // eslint-disable-line react-hooks/exhaustive-deps

  // Nav handler for mobile overlay: navigate then close the drawer
  const handleMobileNav = (view: string) => {
    onViewChange(view);
    onMobileClose?.();
  };

  const drawerRef = useRef<HTMLDivElement>(null);
  const onMobileCloseRef = useRef(onMobileClose);
  onMobileCloseRef.current = onMobileClose;

  // Drawer a11y: focus in on open, Escape closes, focus returns to the opener on close.
  useEffect(() => {
    if (!mobileOpen) return;
    const opener = document.activeElement as HTMLElement | null;
    drawerRef.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onMobileCloseRef.current?.(); };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      if (opener && document.contains(opener)) opener.focus();
    };
  }, [mobileOpen]);

  // Reset the drawer when the viewport crosses into the desktop layout (where it is hidden).
  useEffect(() => {
    const mq = window.matchMedia('(min-width: 1024px) and (orientation: landscape)');
    const onChange = (e: MediaQueryListEvent) => { if (e.matches) onMobileCloseRef.current?.(); };
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  return (
    <>
      {/* ── Mobile overlay backdrop — shown behind drawer at narrow widths ── */}
      {/* Portaled to body: the disco wrapper around the sidebar sets will-change: transform,
          which makes position: fixed relative to that (hidden) wrapper instead of the viewport. */}
      {mobileOpen && createPortal(<>
        <div
          className="fixed inset-0 bg-black/60 z-40 min-[1024px]:landscape:hidden"
          onClick={onMobileClose}
          aria-hidden="true"
        />
      {/* ── Mobile overlay drawer — slides in from the left ── */}
        <div ref={drawerRef} tabIndex={-1} role="dialog" aria-modal="true" aria-label="Navigation" className="outline-none fixed inset-y-0 left-0 z-50 min-[1024px]:landscape:hidden flex flex-col w-[240px] h-full bg-white dark:bg-suno-sidebar border-r border-zinc-200 dark:border-white/5 py-4 overflow-hidden hide-scrollbar">
          {/* Close button */}
          <div className="mb-6 flex flex-shrink-0 items-center justify-between px-3">
            <span className="text-sm font-semibold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Menu</span>
            <button
              className="w-8 h-8 rounded-lg bg-zinc-200 dark:bg-white/5 hover:bg-zinc-300 dark:hover:bg-white/10 flex items-center justify-center transition-all flex-shrink-0"
              onClick={onMobileClose}
              aria-label="Close navigation"
            >
              <X size={16} className="text-zinc-600 dark:text-zinc-400" />
            </button>
          </div>
          <div className="px-3 mb-4 flex-shrink-0">
            <BackendToggle />
          </div>
          <SidebarNav
            activeView={activeView}
            isExpanded={true}
            onNav={handleMobileNav}
            onQuit={onQuit}
            onRestart={onRestart}
            theme={theme}
            onToggleTheme={onToggleTheme}
            showAssistant={showAssistant}
            onToggleAssistant={onToggleAssistant}
          />
        </div>
      </>, document.body)}

      {/* ── Desktop sidebar — hidden at narrow widths ── */}
      <div className={[
        'flex flex-col h-full bg-white dark:bg-suno-sidebar border-r border-zinc-200 dark:border-white/5 flex-shrink-0 py-4 overflow-hidden hide-scrollbar transition-all duration-300',
        'max-[1023px]:hidden portrait:hidden',
        isOpen ? 'w-[200px]' : 'w-[72px]',
      ].join(' ')}>
        {/* Generation engine — directly under the logo (GlobalParamBar) */}
        <div className="px-3 mb-3 flex-shrink-0">
          <BackendToggle collapsed={!isOpen} />
        </div>

        {/* Collapse / Expand toggle */}
        <div className={`mb-6 flex flex-shrink-0 items-center ${isOpen ? 'px-3' : 'justify-center'}`}>
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

        <SidebarNav
          activeView={activeView}
          isExpanded={isOpen}
          onNav={onViewChange}
          onQuit={onQuit}
          onRestart={onRestart}
          theme={theme}
          onToggleTheme={onToggleTheme}
          showAssistant={showAssistant}
          onToggleAssistant={onToggleAssistant}
        />
      </div>
    </>
  );
};

interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  onClick: () => void;
  isExpanded?: boolean;
  /** Live job badge, e.g. "2 · 43%". */
  badge?: string | null;
}

const NavItem: React.FC<NavItemProps> = ({ icon, label, active, onClick, isExpanded, badge }) => (
  <button
    onClick={onClick}
    className={`
      w-full flex-shrink-0 rounded-xl flex items-center gap-3 transition-all duration-200 group relative overflow-hidden
      ${isExpanded ? 'px-3 py-2.5 short:py-1.5 justify-start' : 'aspect-square justify-center'}
      ${active ? 'bg-zinc-100 dark:bg-white/10 text-black dark:text-white' : 'text-zinc-500 hover:text-black dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-white/5'}
    `}
    title={label}
  >
    {active && <div className="absolute left-0 top-1/2 -translate-y-1/2 h-8 w-1 bg-pink-500 rounded-r-full"></div>}
    <div className="flex-shrink-0">{icon}</div>
    {isExpanded && (
      <span className="text-sm font-medium truncate min-w-0">{label}</span>
    )}
    {badge && (isExpanded
      ? <span className="ml-auto flex-shrink-0 px-1.5 py-0.5 rounded-full bg-pink-500 text-white text-[10px] font-semibold tabular-nums whitespace-nowrap">{badge}</span>
      : <span className="absolute top-1 right-1 w-2.5 h-2.5 rounded-full bg-pink-500 animate-pulse" aria-hidden="true" />)}
  </button>
);
