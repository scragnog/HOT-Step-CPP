// SongActionsMenu.tsx — the one 3-dot menu for a generated track.
//
// Every surface that lists songs (Library, Recent Songs, the Lyric Studio
// queue, the playlist, Song Details, Cover Studio) mounts this same component,
// so an action added here appears everywhere at once. Before it existed the
// menu was copy-pasted twice inside SongList alone, and most surfaces had no
// menu at all.
//
// Two kinds of action live here:
//   - self-contained ones the component implements itself (Export Params,
//     Retranscribe, Cover Art, A/B pinning, Post-Processing) — always present,
//     because they need nothing from the host
//   - host-supplied ones passed as callbacks (Edit, Download, Send to Cover,
//     Edit Metadata, Add to Playlist, Delete) — each hidden when its callback
//     is absent, since "Add to Playlist" makes no sense inside the playlist
//
// The dropdown portals to document.body to escape overflow clipping, and tags
// itself data-portal-layer so a click inside it does not read as a click
// outside to BarSection's dismiss handler (#129/#126).

import React, { useRef, useState, useLayoutEffect } from 'react';
import ReactDOM from 'react-dom';
import {
  MoreHorizontal, RotateCcw, Download, Disc3, Tags, Trash2, Upload,
  Mic2, Image, ArrowLeftRight, ListPlus, Sparkles, Loader2, AlertCircle,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { Song, UnifiedRecentSong } from '../../types';
import { songToTrack } from '../../stores/playbackStore';
import { useABCompareSelector, setTrackA, setTrackB } from '../../stores/abCompareStore';
import {
  startPostProcessing, canPostProcess, dismissPostProcessing, usePostProcessRun,
} from '../../stores/postProcessStore';
import { openCoverArtPrompt } from '../library/CoverArtPromptModal';

// ── Portal ───────────────────────────────────────────────────────────────────

interface PortalMenuProps {
  anchorRef: React.RefObject<HTMLElement | null>;
  onClose: () => void;
  children: React.ReactNode;
}

const MENU_WIDTH = 220;
const MENU_MAX_HEIGHT = 420;

const PortalMenu: React.FC<PortalMenuProps> = ({ anchorRef, onClose, children }) => {
  const [pos, setPos] = useState({ top: 0, left: 0, flipped: false });

  useLayoutEffect(() => {
    if (!anchorRef.current) return;
    const rect = anchorRef.current.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom;
    const flipped = spaceBelow < MENU_MAX_HEIGHT && rect.top > spaceBelow;
    setPos({
      top: flipped ? rect.top : rect.bottom + 4,
      // Right-align to the button, then keep the whole menu on screen.
      left: Math.min(
        Math.max(8, rect.right - MENU_WIDTH),
        Math.max(8, window.innerWidth - MENU_WIDTH - 8),
      ),
      flipped,
    });
  }, [anchorRef]);

  return ReactDOM.createPortal(
    <>
      <div data-portal-layer className="fixed inset-0 z-[9998]" onClick={onClose} />
      <div
        data-portal-layer
        className="fixed z-[9999] bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 rounded-xl shadow-xl py-1 overflow-y-auto"
        style={{
          top: pos.flipped ? undefined : pos.top,
          bottom: pos.flipped ? (window.innerHeight - pos.top + 4) : undefined,
          left: pos.left,
          width: MENU_WIDTH,
          maxHeight: MENU_MAX_HEIGHT,
        }}
      >
        {children}
      </div>
    </>,
    document.body,
  );
};

// ── Menu items ───────────────────────────────────────────────────────────────

const ITEM_BASE = 'w-full flex items-center gap-2 px-3 py-2 text-sm text-left transition-colors';

interface ItemProps {
  icon: React.ReactNode;
  label: React.ReactNode;
  onClick: () => void;
  /** Tailwind colour classes for the enabled state. */
  tone?: string;
  disabled?: boolean;
  /** Shown on hover — the place to explain a disabled item. */
  title?: string;
}

const Item: React.FC<ItemProps> = ({ icon, label, onClick, tone, disabled, title }) => (
  <button
    type="button"
    title={title}
    disabled={disabled}
    onClick={(e) => { e.stopPropagation(); if (!disabled) onClick(); }}
    className={`${ITEM_BASE} ${
      disabled
        ? 'text-zinc-400 dark:text-zinc-600 cursor-not-allowed'
        : (tone || 'text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white')
    }`}
  >
    <span className="flex-shrink-0">{icon}</span>
    <span className="truncate">{label}</span>
  </button>
);

const Divider: React.FC = () => (
  <div className="border-t border-zinc-200 dark:border-white/5 my-1" />
);

// ── Shape adapters ───────────────────────────────────────────────────────────

/**
 * Widen a /songs/recent row into the Song shape this menu (and downloadTrack,
 * and the cover-art modal) expects.
 *
 * mastered_audio_url is the field that matters most here: it is what decides
 * whether the post-processing item is offered, so dropping it would offer a
 * second pass on an already-mastered track.
 */
export function songFromRecent(rs: UnifiedRecentSong): Song {
  return {
    id: rs.id,
    title: rs.title || 'Untitled',
    style: rs.style || rs.caption || '',
    caption: rs.caption || '',
    lyrics: rs.lyrics || '',
    audioUrl: rs.audio_url || '',
    audio_url: rs.audio_url || '',
    coverUrl: rs.cover_url || rs.artist_image || '',
    cover_url: rs.cover_url || '',
    duration: rs.duration || 0,
    bpm: rs.bpm,
    key_scale: rs.key_scale,
    tags: [],
    masteredAudioUrl: rs.mastered_audio_url || '',
    mastered_audio_url: rs.mastered_audio_url || '',
    noAdapterAudioUrl: rs.noadapter_audio_url || '',
    latentUrl: rs.latent_url || '',
    latent_url: rs.latent_url || '',
  };
}

/**
 * Widen a playlist entry into the Song shape.
 *
 * A playlist item is a thin projection stored in localStorage, so it carries no
 * lyrics and its `style` doubles as the caption. That is enough for every menu
 * action: the post-processing gate reads masteredAudioUrl and audioUrl, and the
 * server re-reads the real row before it runs anything.
 */
export function songFromPlaylistItem(item: {
  id: string;
  title: string;
  audioUrl: string;
  masteredAudioUrl?: string;
  noAdapterAudioUrl?: string;
  coverUrl?: string;
  duration?: number;
  style?: string;
  generationParams?: any;
}): Song {
  return {
    id: item.id,
    title: item.title || 'Untitled',
    style: item.style || '',
    caption: item.style || '',
    lyrics: '',
    audioUrl: item.audioUrl || '',
    audio_url: item.audioUrl || '',
    coverUrl: item.coverUrl || '',
    duration: item.duration || 0,
    tags: [],
    masteredAudioUrl: item.masteredAudioUrl || '',
    mastered_audio_url: item.masteredAudioUrl || '',
    noAdapterAudioUrl: item.noAdapterAudioUrl || '',
    generationParams: item.generationParams,
  };
}

/**
 * Widen a finished audio-queue item into the Song shape.
 *
 * Returns null without `songId`: that field is the DB row id, and every action
 * here that reaches the server needs one. A queue item can be playable before
 * its row exists, and offering post-processing on it would 404.
 */
export function songFromQueueItem(item: {
  id: string;
  songId?: string;
  audioUrl?: string;
  masteredAudioUrl?: string;
  noAdapterAudioUrl?: string;
  coverUrl?: string;
  artistImageUrl?: string;
  audioDuration?: number;
  artistName?: string;
  generation?: { title?: string; caption?: string; lyrics?: string };
}): Song | null {
  if (!item.songId) return null;
  return {
    id: item.songId,
    title: item.generation?.title || 'Untitled',
    lyrics: item.generation?.lyrics || '',
    style: item.generation?.caption || '',
    caption: item.generation?.caption || '',
    audioUrl: item.audioUrl || '',
    audio_url: item.audioUrl || '',
    masteredAudioUrl: item.masteredAudioUrl || '',
    mastered_audio_url: item.masteredAudioUrl || '',
    noAdapterAudioUrl: item.noAdapterAudioUrl || '',
    coverUrl: item.coverUrl || item.artistImageUrl || '',
    duration: item.audioDuration || 0,
    artistName: item.artistName || '',
    tags: [],
  };
}

// ── Actions the component owns ───────────────────────────────────────────────

/** Download this song's params as a hot-step preset JSON. */
function exportParams(song: Song): void {
  const params = (song.generationParams || (song as any).generation_params || {}) as Record<string, any>;
  const exportData = {
    _format: 'hot-step-preset',
    _version: 1,
    ...params,
    title: song.title || '',
    caption: params.caption || song.style || '',
    lyrics: params.lyrics || song.lyrics || '',
  };
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${(song.title || 'song').slice(0, 40).replace(/[^a-zA-Z0-9]/g, '_').toLowerCase()}_params.json`;
  a.click();
  URL.revokeObjectURL(url);
}

async function retranscribe(song: Song): Promise<void> {
  try {
    const { retranscribeLyrics } = await import('../../services/api');
    const result = await retranscribeLyrics(song.id);
    console.log(`[Retranscribe] ${result.wordCount} words, ${result.lineCount} lines`);
  } catch (err: any) {
    console.error('[Retranscribe] Failed:', err.message);
  }
}

// ── Component ────────────────────────────────────────────────────────────────

export interface SongActionsMenuProps {
  song: Song;

  // Host-supplied actions. Each item is omitted when its handler is absent.
  onReuse?: () => void;
  onDownload?: () => void;
  onSendToCover?: () => void;
  onEditMetadata?: () => void;
  onAddToPlaylist?: () => void;
  onDelete?: () => void;

  /** Extra items appended before the destructive section — for surface-specific
   *  actions like the playlist's "Remove from playlist". */
  extraItems?: React.ReactNode;

  /** Hide the self-contained block (Export Params, Retranscribe, Cover Art,
   *  A/B) on surfaces where it is noise — a compact queue row, say. */
  compact?: boolean;

  /** Size of the trigger glyph. */
  size?: number;
  className?: string;
  /** Rendered instead of the default 3-dot button. */
  trigger?: React.ReactNode;
}

export const SongActionsMenu: React.FC<SongActionsMenuProps> = ({
  song,
  onReuse,
  onDownload,
  onSendToCover,
  onEditMetadata,
  onAddToPlaylist,
  onDelete,
  extraItems,
  compact = false,
  size = 16,
  className = '',
  trigger,
}) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);
  const abTrackAId = useABCompareSelector(s => s.trackA?.id);
  const abTrackBId = useABCompareSelector(s => s.trackB?.id);
  const run = usePostProcessRun(song.id);

  const pp = canPostProcess(song);
  const ppBusy = !!run && (run.status === 'starting' || run.status === 'pending' || run.status === 'running');
  const ppFailed = run?.status === 'failed';
  const ppDone = run?.status === 'succeeded';

  const close = () => setOpen(false);
  const act = (fn: () => void) => { fn(); close(); };

  const handlePostProcess = async () => {
    close();
    try {
      await startPostProcessing(song);
    } catch (err: any) {
      // startPostProcessing has already parked the message on the run, which
      // the trigger badge surfaces. Nothing to do but keep it out of the void.
      console.error('[PostProcess]', err.message);
    }
  };

  // The trigger doubles as the progress indicator: this menu is the only place
  // a re-run can be started from, so it is the natural place to watch one.
  const triggerIcon = ppBusy
    ? <Loader2 size={size} className="animate-spin text-violet-400" />
    : ppFailed
      ? <AlertCircle size={size} className="text-red-400" />
      : <MoreHorizontal size={size} />;

  const triggerTitle = ppBusy
    ? `Post-processing: ${run?.stage || 'running'}`
    : ppFailed
      ? `Post-processing failed: ${run?.error || 'unknown error'}`
      : ppDone
        ? 'Post-processing complete'
        : 'Song actions';

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        title={triggerTitle}
        onClick={(e) => { e.stopPropagation(); setOpen(o => !o); }}
        className={`p-1.5 rounded-lg text-zinc-600 dark:text-zinc-400 hover:text-white hover:bg-white/10 transition-colors ${className}`}
      >
        {trigger ?? triggerIcon}
      </button>

      {open && (
        <PortalMenu anchorRef={btnRef} onClose={close}>
          {/* Post-processing — the reason this menu exists. Deliberately first
              and deliberately shown (disabled) rather than hidden when it can't
              run, so "why can't I master this?" has an answer on hover. */}
          <Item
            icon={ppBusy ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
            label={ppBusy ? (run?.stage || 'Post-processing...') : t('songActions.postProcess', 'Run Post-Processing')}
            onClick={handlePostProcess}
            disabled={!pp.enabled}
            title={pp.enabled
              ? t('songActions.postProcessHint',
                  'Runs the full PP chain with your current settings, even if the global toggle is off')
              : pp.reason}
            tone="text-violet-400 hover:bg-violet-500/10"
          />
          {ppFailed && (
            <Item
              icon={<AlertCircle size={14} />}
              label="Dismiss PP error"
              onClick={() => act(() => dismissPostProcessing(song.id))}
              title={run?.error}
              tone="text-red-400 hover:bg-red-500/10"
            />
          )}

          <Divider />

          {onReuse && (
            <Item icon={<RotateCcw size={14} />} label={t('library.edit')} onClick={() => act(onReuse)} />
          )}
          {onAddToPlaylist && (
            <Item
              icon={<ListPlus size={14} />}
              label={t('library.addToPlaylist')}
              onClick={() => act(onAddToPlaylist)}
              tone="text-pink-400 hover:bg-pink-500/10"
            />
          )}
          {onDownload && (
            <Item icon={<Download size={14} />} label={t('library.download')} onClick={() => act(onDownload)} />
          )}
          {onSendToCover && (
            <Item
              icon={<Disc3 size={14} />}
              label={t('library.sendToCover', 'Send to Cover Studio')}
              onClick={() => act(onSendToCover)}
              tone="text-cyan-400 hover:bg-cyan-500/10"
            />
          )}
          {onEditMetadata && (
            <Item
              icon={<Tags size={14} />}
              label={t('metadata.editTitle', 'Edit Metadata')}
              onClick={() => act(onEditMetadata)}
              tone="text-amber-400 hover:bg-amber-500/10"
            />
          )}

          {extraItems}

          {!compact && (
            <>
              <Divider />
              <Item
                icon={<Upload size={14} />}
                label={t('library.exportParams')}
                onClick={() => act(() => exportParams(song))}
              />
              <Item
                icon={<Mic2 size={14} />}
                label={t('songActions.retranscribe', 'Retranscribe Lyrics')}
                onClick={() => act(() => { void retranscribe(song); })}
                tone="text-sky-400 hover:bg-sky-500/10"
              />
              <Item
                icon={<Image size={14} />}
                label={song.coverUrl
                  ? t('songActions.regenerateCoverArt', 'Regenerate Cover Art')
                  : t('songActions.generateCoverArt', 'Generate Cover Art')}
                onClick={() => act(() => openCoverArtPrompt(song))}
                tone="text-violet-400 hover:bg-violet-500/10"
              />

              <Divider />
              <Item
                icon={<ArrowLeftRight size={14} />}
                label={t('songActions.setTrackA', 'Set as Track A')}
                onClick={() => act(() => setTrackA(songToTrack(song)))}
                tone={song.id === abTrackAId
                  ? 'text-blue-400 bg-blue-500/10'
                  : 'text-blue-400/70 hover:bg-blue-500/10 hover:text-blue-400'}
              />
              <Item
                icon={<ArrowLeftRight size={14} />}
                label={t('songActions.setTrackB', 'Set as Track B')}
                onClick={() => act(() => setTrackB(songToTrack(song)))}
                tone={song.id === abTrackBId
                  ? 'text-orange-400 bg-orange-500/10'
                  : 'text-orange-400/70 hover:bg-orange-500/10 hover:text-orange-400'}
              />
            </>
          )}

          {onDelete && (
            <>
              <Divider />
              <Item
                icon={<Trash2 size={14} />}
                label={t('library.delete')}
                onClick={() => act(onDelete)}
                tone="text-red-400 hover:bg-red-500/10"
              />
            </>
          )}
        </PortalMenu>
      )}
    </>
  );
};

export default SongActionsMenu;
