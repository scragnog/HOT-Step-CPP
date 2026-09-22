// RightSidebar.tsx — Selected song details panel
//
// Reads its parameters through songFacts, which picks the field set by backend.
// Nothing here may reach into generation_params directly: that blob holds every
// panel's settings, not the ones this render used, and reading it generically
// is what put AS1.5 numbers on MM3 and YuE2 songs.

import React from 'react';
import { X, Play, Pause, RotateCcw, Pencil, ChevronDown, ChevronRight } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { Song } from '../../types';
import { HoverFullText } from '../shared/HoverFullText';
import { SongActionsMenu } from '../shared/SongActionsMenu';
import { CoverImage } from '../shared/CoverImage';
import { displayTitle, songArtist, songSubject } from '../../utils/songDisplay';
import {
  BACKEND_LABELS, buildFactGroups, buildTrackChips, songBackend,
  type Fact,
} from './songFacts';

interface RightSidebarProps {
  song: Song;
  onClose: () => void;
  onReuse: (song: Song) => void;
  onDelete: (song: Song) => void;
  onPlay: (song: Song) => void;
  isPlaying: boolean;
  onDownload?: (song: Song) => void;
  onRename?: (song: Song, newTitle: string) => void;
  onSendToCover?: (song: Song) => void;
  onEditMetadata?: (song: Song) => void;
}

/** Per-backend accent, so the chip is recognisable at a glance. */
const BACKEND_CHIP: Record<string, string> = {
  'ace': 'bg-pink-500/10 text-pink-600 dark:text-pink-400 border-pink-500/30',
  'minimax-m3': 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border-cyan-500/30',
  'yue2': 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/30',
};

export const RightSidebar: React.FC<RightSidebarProps> = ({
  song,
  onClose,
  onReuse,
  onDelete,
  onPlay,
  isPlaying,
  onDownload,
  onRename,
  onSendToCover,
  onEditMetadata,
}) => {
  const { t } = useTranslation();

  const backend = songBackend(song);
  const chips = React.useMemo(() => buildTrackChips(song), [song]);
  const groups = React.useMemo(() => buildFactGroups(song), [song]);
  const artist = songArtist(song);
  const subject = songSubject(song);

  // Inline rename state
  const [editing, setEditing] = React.useState(false);
  const [editTitle, setEditTitle] = React.useState(song.title || '');
  const inputRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  // Reset edit state when song changes
  React.useEffect(() => {
    setEditing(false);
    setEditTitle(song.title || '');
  }, [song.id]);

  const commitRename = () => {
    const trimmed = editTitle.trim();
    if (trimmed && trimmed !== (song.title || '')) {
      onRename?.(song, trimmed);
    }
    setEditing(false);
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-white/5">
        <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 truncate">{t('details.songDetails')}</h3>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-zinc-600 dark:text-zinc-400 hover:text-white hover:bg-white/5 transition-colors"
        >
          <X size={16} />
        </button>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto hide-scrollbar p-4 space-y-4">
        {/* Cover */}
        <div className="aspect-square w-full rounded-xl overflow-hidden bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-white/5 flex items-center justify-center">
          <CoverImage url={song.coverUrl} seed={song.id} alt={song.title}
            className="w-full h-full object-cover" iconSize={48} />
        </div>

        {/* Title, artist, what it is about */}
        <div>
          {editing ? (
            <input
              ref={inputRef}
              className="w-full text-lg font-bold bg-zinc-100 dark:bg-zinc-800 border border-pink-500/40 rounded-lg px-2 py-0.5 text-zinc-900 dark:text-white outline-none focus:border-pink-500"
              value={editTitle}
              onChange={e => setEditTitle(e.target.value)}
              onBlur={commitRename}
              onKeyDown={e => {
                if (e.key === 'Enter') commitRename();
                if (e.key === 'Escape') { setEditTitle(song.title || ''); setEditing(false); }
              }}
            />
          ) : (
            <div className="flex items-center gap-1.5 group/title">
              <h2 className="text-lg font-bold text-zinc-900 dark:text-white leading-tight truncate">
                {displayTitle(song)}
              </h2>
              {onRename && (
                <button
                  onClick={() => { setEditTitle(song.title || ''); setEditing(true); }}
                  className="flex-shrink-0 p-1 rounded-lg text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200 opacity-0 group-hover/title:opacity-100 transition-opacity"
                  title={t('library.rename')}
                >
                  <Pencil size={13} />
                </button>
              )}
            </div>
          )}

          <div className="flex items-center gap-2 mt-1.5 flex-wrap">
            <span className={`px-2 py-0.5 rounded-md border text-[10px] font-semibold uppercase tracking-wider ${BACKEND_CHIP[backend]}`}>
              {BACKEND_LABELS[backend]}
            </span>
            {artist && (
              <span className="text-xs text-zinc-600 dark:text-zinc-400 truncate">{artist}</span>
            )}
          </div>

          {subject && (
            <HoverFullText
              as="p"
              text={subject}
              className="mt-2 text-sm text-zinc-700 dark:text-zinc-300 line-clamp-3 cursor-help"
            />
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onPlay(song)}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl bg-pink-600 hover:bg-pink-500 text-white font-semibold transition-colors"
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            {isPlaying ? t('details.pause') : t('details.play')}
          </button>
          <button
            onClick={() => onReuse(song)}
            className="p-2.5 rounded-xl bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-300 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300 transition-colors"
            title={t('details.edit')}
          >
            <RotateCcw size={16} />
          </button>
          {/* Everything below Play/Edit collapsed into the shared menu — the
              row was seven icon buttons deep and still had no way to run
              post-processing. */}
          <SongActionsMenu
            song={song}
            size={16}
            className="!p-2.5 rounded-xl bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-300 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300"
            onDownload={onDownload ? () => onDownload(song) : undefined}
            onSendToCover={onSendToCover ? () => onSendToCover(song) : undefined}
            onEditMetadata={onEditMetadata ? () => onEditMetadata(song) : undefined}
            onDelete={() => onDelete(song)}
          />
        </div>

        {/* Track chips — only the ones this backend actually measured */}
        {chips.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {chips.map(chip => (
              <span key={chip.label}
                className="px-2 py-1 rounded-lg bg-zinc-100 dark:bg-white/[0.04] border border-zinc-200 dark:border-white/5 text-[11px]">
                <span className="text-zinc-500">{chip.label} </span>
                <span className="text-zinc-800 dark:text-zinc-200 font-medium">{chip.value}</span>
              </span>
            ))}
          </div>
        )}

        {/* Style / caption — the prompt, kept below the human-readable bits */}
        {song.style && (
          <Collapsible title="Prompt" defaultOpen={!subject}>
            <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed whitespace-pre-wrap break-words">
              {song.style}
            </p>
          </Collapsible>
        )}

        {/* How it was made */}
        {groups.length > 0 && (
          <Collapsible title="How it was made" defaultOpen>
            <div className="space-y-3">
              {groups.map(group => (
                <div key={group.title}>
                  <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-1">
                    {group.title}
                  </div>
                  <div className="rounded-lg border border-zinc-200 dark:border-white/5 divide-y divide-zinc-200 dark:divide-white/5 overflow-hidden">
                    {group.facts.map(fact => <FactRow key={fact.label} fact={fact} />)}
                  </div>
                </div>
              ))}
            </div>
          </Collapsible>
        )}

        {/* Lyrics */}
        {song.lyrics && (
          <Collapsible title={t('details.lyrics')} defaultOpen={false}>
            <pre className="text-sm text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap font-sans leading-relaxed max-h-64 overflow-y-auto">
              {song.lyrics}
            </pre>
          </Collapsible>
        )}
      </div>
    </div>
  );
};

/** One label → value line. Deliberately plain: a dozen gradient tiles read as
 *  decoration, a list of aligned pairs reads as data. */
const FactRow: React.FC<{ fact: Fact }> = ({ fact }) => (
  <div className="flex items-baseline gap-3 px-2.5 py-1.5 bg-zinc-50/60 dark:bg-white/[0.02]">
    <span className="text-[11px] text-zinc-500 flex-shrink-0">{fact.label}</span>
    <span
      title={fact.title ?? fact.value}
      className={`ml-auto text-right text-xs font-medium truncate
        ${fact.mono ? 'font-mono text-[11px]' : ''}
        ${fact.tone === 'warn' ? 'text-amber-600 dark:text-amber-400'
          : fact.tone === 'good' ? 'text-emerald-600 dark:text-emerald-400'
          : 'text-zinc-800 dark:text-zinc-200'}`}
    >
      {fact.value}
    </span>
  </div>
);

const Collapsible: React.FC<{
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}> = ({ title, defaultOpen = true, children }) => {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-zinc-50/60 dark:bg-zinc-900/40 overflow-hidden">
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center gap-1.5 px-3 py-2 text-left hover:bg-zinc-100/60 dark:hover:bg-white/[0.03] transition-colors"
      >
        {open ? <ChevronDown size={12} className="text-zinc-500" /> : <ChevronRight size={12} className="text-zinc-500" />}
        <span className="text-xs font-semibold text-zinc-700 dark:text-zinc-300 uppercase tracking-wider">{title}</span>
      </button>
      {open && <div className="px-3 pb-3">{children}</div>}
    </div>
  );
};
