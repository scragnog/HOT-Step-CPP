/**
 * UnifiedRecentSongs.tsx — Shows recently generated songs across ALL modes.
 *
 * Data flow: GET /api/songs/recent?source=X returns normalized rows, cached
 * per source by the shell-queue recent-songs query. The cache outlives every
 * mount, so navigation renders instantly; song writes (App.tsx) patch and
 * invalidate it.
 * Visual design matches the original Lyric Studio RecentSongsList.
 */

import React, { useCallback } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Play, Loader2, Music, ListPlus, Check } from 'lucide-react';
import { songApi } from '../../services/api';
import type { UnifiedRecentSong } from '../../types';
import { useAuth } from '../../context/AuthContext';
import { downloadTrack } from '../../utils/downloadTrack';
import { usePlaylist } from '../../stores/playlistStore';
import { playFromList, togglePlay, unifiedRecentSongToTrack, usePlaybackSelector } from '../../stores/playbackStore';
import { SongActionsMenu, songFromRecent } from './SongActionsMenu';
import { useDisguiseMode } from '../../hooks/useDisguiseMode';
import { CoverImage } from './CoverImage';
import { recentSongsOptions, removeSongs } from '../../queries/shell-queue/songs';
import { QueryError } from './QueryError';

interface UnifiedRecentSongsProps {
  showToast: (msg: string, type?: 'success' | 'error') => void;
  compact?: boolean;
  /** Filter by source mode — undefined or 'all' shows everything */
  source?: string;
}

const NO_SONGS: UnifiedRecentSong[] = [];
const playable = (res: { songs: UnifiedRecentSong[] }) => (res.songs || []).filter(s => !!s.audio_url);

// ── Component ────────────────────────────────────────────────────────────────

export const UnifiedRecentSongs: React.FC<UnifiedRecentSongsProps> = ({
  showToast, compact = false, source = 'all',
}) => {
  const { token, sessionEpoch } = useAuth();
  void sessionEpoch;
  const queryClient = useQueryClient();
  const query = useQuery({ ...recentSongsOptions(source || 'all'), enabled: !!token, select: playable });
  const songs = query.data ?? NO_SONGS;
  // Spin only with nothing cached at all; an empty result is an answer.
  const loading = query.isPending && !!token;
  const { disguiseArtist, disguiseTitle } = useDisguiseMode();

  const currentTrackId = usePlaybackSelector(s => s.currentTrack?.id ?? null);
  const handlePlay = useCallback(async (rs: UnifiedRecentSong) => {
    const track = unifiedRecentSongToTrack(rs);
    // The current track pauses/resumes in place; reloading it would restart it.
    if (track.id === currentTrackId) { togglePlay(); return; }
    const allTracks = songs.map(unifiedRecentSongToTrack);
    const playbackSource = source === 'cover-studio' ? 'cover-studio' as const
      : source === 'lyric-studio' ? 'lireek-recent' as const
      : 'library' as const;
    playFromList(track, allTracks, playbackSource);
  }, [songs, source, currentTrackId]);

  const deleteMut = useMutation({
    mutationFn: (rs: UnifiedRecentSong) => songApi.delete(rs.id),
    onSuccess: (_res, rs) => {
      void removeSongs(queryClient, [rs.id]);
      showToast('Song deleted');
    },
  });
  const handleDelete = useCallback((rs: UnifiedRecentSong) => {
    if (!token || !rs.audio_url) return;
    deleteMut.mutate(rs);
  }, [token, deleteMut]);

  const handleDownload = useCallback((rs: UnifiedRecentSong) => {
    if (!rs.audio_url) return;
    downloadTrack(songFromRecent(rs), { artistName: rs.artist_name || '' });
  }, []);

  if (query.isError && songs.length === 0) {
    return <QueryError error={query.error} onRetry={() => void query.refetch()} className="mx-2 my-2" />;
  }

  if (loading && songs.length === 0) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-4 h-4 text-zinc-500 animate-spin" />
      </div>
    );
  }

  if (songs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center px-4">
        <Music className="w-5 h-5 text-zinc-600 mb-2" />
        <p className="text-xs text-zinc-500">No recent generations yet</p>
      </div>
    );
  }

  return (
    <>
      <div className={`grid ${compact ? 'grid-cols-1' : 'grid-cols-2'} auto-rows-[4.5rem] gap-1 px-2 py-1.5 overflow-y-auto scrollbar-hide`} style={{ maxHeight: '100%' }}>
        {songs.slice(0, 50).map((rs) => {
          const dur = rs.duration || 0;
          const mins = Math.floor(dur / 60);
          const secs = String(Math.floor(dur % 60)).padStart(2, '0');
          const coverUrl = rs.cover_url || rs.artist_image || '';
          return (
            <div key={rs.id}
              className="flex items-center gap-2.5 rounded-lg hover:bg-white/[0.06] transition-colors text-left group px-2 overflow-hidden relative cursor-pointer"
              onClick={() => handlePlay(rs)}>
              <div className="w-14 h-14 rounded-md flex-shrink-0 overflow-hidden bg-zinc-100 dark:bg-zinc-800 relative">
                <CoverImage url={coverUrl} seed={String(rs.id)} iconSize={20} />
                <div className="absolute inset-0 bg-black/20 dark:bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <Play className="w-4 h-4 text-white ml-0.5" />
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-zinc-800 dark:text-zinc-200 truncate leading-snug">
                  {disguiseTitle(rs.title || 'Untitled')}
                </p>
                {rs.artist_name && (
                  <p className="text-[10px] text-zinc-500 truncate leading-snug">{disguiseArtist(rs.artist_name)}</p>
                )}
                {dur > 0 && (
                  <p className="text-[10px] text-zinc-600 font-mono mt-0.5">{mins}:{secs}</p>
                )}
              </div>
              {/* A/B pinning, download and delete used to be five buttons
                  crammed into a 4.5rem row. They live on the shared menu now;
                  only the playlist toggle stays out here, because it is the one
                  that shows state (in / not in) rather than firing an action. */}
              <div className="absolute right-1 top-1/2 -translate-y-1/2 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <AddToPlaylistBtn rs={rs} />
                <SongActionsMenu
                  song={songFromRecent(rs)}
                  size={14}
                  className="bg-zinc-100/80 dark:bg-zinc-800/80 rounded-md"
                  onDownload={() => handleDownload(rs)}
                  onDelete={() => handleDelete(rs)}
                />
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
};

// ── Add-to-playlist helper ───────────────────────────────────────────────────

const AddToPlaylistBtn: React.FC<{ rs: UnifiedRecentSong }> = ({ rs }) => {
  const playlist = usePlaylist();
  const inPlaylist = playlist.isIn(rs.id);

  const toggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (inPlaylist) {
      playlist.remove(rs.id);
    } else {
      playlist.add({
        id: rs.id,
        title: rs.title || 'Untitled',
        audioUrl: rs.audio_url || '',
        masteredAudioUrl: rs.mastered_audio_url || '',
        noAdapterAudioUrl: rs.noadapter_audio_url || '',
        artistName: rs.artist_name || '',
        coverUrl: rs.cover_url || rs.artist_image || '',
        duration: rs.duration || 0,
      });
    }
  };

  return (
    <button onClick={toggle}
      className={`p-1.5 rounded-md transition-colors ${
        inPlaylist ? 'bg-pink-500/20 text-pink-400 hover:bg-pink-500/30'
          : 'bg-zinc-100/80 dark:bg-zinc-800/80 hover:bg-zinc-200 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-400 hover:text-pink-400'
      }`}
      title={inPlaylist ? 'Remove from playlist' : 'Add to playlist'}>
      {inPlaylist ? <Check className="w-3 h-3" /> : <ListPlus className="w-3 h-3" />}
    </button>
  );
};
