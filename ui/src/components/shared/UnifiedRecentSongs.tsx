/**
 * UnifiedRecentSongs.tsx — Shows recently generated songs across ALL modes.
 *
 * Data flow: GET /api/songs/recent?source=X returns normalized rows.
 * MODULE-LEVEL CACHE ensures instant render on navigation.
 * Visual design matches the original Lyric Studio RecentSongsList.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Loader2, Music, ListPlus, Check } from 'lucide-react';
import { songApi } from '../../services/api';
import type { UnifiedRecentSong } from '../../types';
import { useAuth } from '../../context/AuthContext';
import { downloadTrack } from '../../utils/downloadTrack';
import { usePlaylist } from '../lyric-studio/playlistStore';
import { playFromList, unifiedRecentSongToTrack } from '../../stores/playbackStore';
import { SongActionsMenu, songFromRecent } from './SongActionsMenu';
import { useDisguiseMode } from '../../hooks/useDisguiseMode';

interface UnifiedRecentSongsProps {
  showToast: (msg: string, type?: 'success' | 'error') => void;
  refreshKey?: number;
  compact?: boolean;
  /** Filter by source mode — undefined or 'all' shows everything */
  source?: string;
}

// ── Module-level cache (keyed by source) ─────────────────────────────────────

const _cache = new Map<string, { songs: UnifiedRecentSong[]; key: number }>();
const _fetchInFlight = new Set<string>();  // per-source to avoid cross-blocking

const CACHE_CLEARED_EVENT = 'recent-songs-cache-cleared';

/** Drop all cached recent-song lists (e.g. after Nuke All Generations) and
 *  tell any mounted instances to empty themselves. */
export function clearRecentSongsCache(): void {
  _cache.clear();
  window.dispatchEvent(new Event(CACHE_CLEARED_EVENT));
}

// Cache correction at MODULE scope, not inside the component.
//
// The per-source cache outlives every mount, so patching it only from a
// mounted instance meant a post-processing pass that finished while you were
// on another page left the cache claiming the track was still unprocessed.
// The menu then offered "Run Post-Processing" (refused by the server) and hid
// "Remove Post-Processed Version" — both decisions driven by the same stale
// flag. Listening here fixes the cache whether or not anything is on screen.
function patchCachedSong(songId: string, patch: Partial<UnifiedRecentSong>): void {
  for (const [key, entry] of _cache) {
    if (!entry.songs.some(s => s.id === songId)) continue;
    _cache.set(key, {
      ...entry,
      songs: entry.songs.map(s => (s.id === songId ? { ...s, ...patch } : s)),
    });
  }
}

window.addEventListener('song-postprocessed', (e: Event) => {
  const { songId, masteredAudioUrl } = (e as CustomEvent).detail || {};
  if (songId && masteredAudioUrl) patchCachedSong(songId, { mastered_audio_url: masteredAudioUrl });
});

window.addEventListener('song-postprocess-reverted', (e: Event) => {
  const { songId } = (e as CustomEvent).detail || {};
  if (songId) patchCachedSong(songId, { mastered_audio_url: '' });
});

// ── Component ────────────────────────────────────────────────────────────────

export const UnifiedRecentSongs: React.FC<UnifiedRecentSongsProps> = ({
  showToast, refreshKey = 0, compact = false, source = 'all',
}) => {
  const { token } = useAuth();
  const cacheKey = source || 'all';
  const cached = _cache.get(cacheKey);
  const [songs, setSongs] = useState<UnifiedRecentSong[]>(cached?.songs || []);
  // Only show loading spinner if we have NO cached data at all (not even an empty result).
  // A cache entry with songs=[] means "we fetched and there are none" — don't spin for that.
  const [loading, setLoading] = useState(!cached);
  const mountedRef = useRef(true);
  const { disguiseArtist, disguiseTitle } = useDisguiseMode();

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, [cacheKey]);

  useEffect(() => {
    const onCleared = () => { setSongs([]); setLoading(false); };
    window.addEventListener(CACHE_CLEARED_EVENT, onCleared);
    return () => window.removeEventListener(CACHE_CLEARED_EVENT, onCleared);
  }, []);

  // A post-processing re-run gives an existing row a mastered file. This list
  // keeps its own module-level cache, so it has to be patched directly —
  // App.tsx's song state is a different copy.
  useEffect(() => {
    const onProcessed = (e: Event) => {
      const { songId, masteredAudioUrl } = (e as CustomEvent).detail || {};
      if (!songId || !masteredAudioUrl) return;
      setSongs(prev => {
        const updated = prev.map(s =>
          s.id === songId ? { ...s, mastered_audio_url: masteredAudioUrl } : s
        );
        _cache.set(cacheKey, { songs: updated, key: refreshKey });
        return updated;
      });
    };
    const onReverted = (e: Event) => {
      const { songId } = (e as CustomEvent).detail || {};
      if (!songId) return;
      setSongs(prev => {
        const updated = prev.map(s =>
          s.id === songId ? { ...s, mastered_audio_url: '' } : s
        );
        _cache.set(cacheKey, { songs: updated, key: refreshKey });
        return updated;
      });
    };
    window.addEventListener('song-postprocessed', onProcessed);
    window.addEventListener('song-postprocess-reverted', onReverted);
    return () => {
      window.removeEventListener('song-postprocessed', onProcessed);
      window.removeEventListener('song-postprocess-reverted', onReverted);
    };
  }, [cacheKey, refreshKey]);

  useEffect(() => {
    if (!token) { setLoading(false); return; }
    // Check cache inline (not via closure) to avoid stale dep issues
    const entry = _cache.get(cacheKey);
    if (entry && entry.key === refreshKey) {
      // Cache hit — still sync state so component reflects the correct source
      // (React may reuse this instance across view changes, keeping stale useState)
      setSongs(entry.songs);
      setLoading(false);
      return;
    }
    if (_fetchInFlight.has(cacheKey)) return;
    // Only show spinner if we have no cached data at all
    if (!entry) setLoading(true);

    _fetchInFlight.add(cacheKey);
    songApi.getRecentSongs(token, source, 50).then(res => {
      const resolved = (res.songs || []).filter(s => !!s.audio_url);
      _cache.set(cacheKey, { songs: resolved, key: refreshKey });
      _fetchInFlight.delete(cacheKey);
      if (mountedRef.current) { setSongs(resolved); setLoading(false); }
    }).catch(() => {
      _fetchInFlight.delete(cacheKey);
      if (mountedRef.current) setLoading(false);
    });
  }, [refreshKey, token, source, cacheKey]);

  const handlePlay = useCallback(async (rs: UnifiedRecentSong) => {
    const track = unifiedRecentSongToTrack(rs);
    const allTracks = songs.map(unifiedRecentSongToTrack);
    const playbackSource = source === 'cover-studio' ? 'cover-studio' as const
      : source === 'lyric-studio' ? 'lireek-recent' as const
      : 'library' as const;
    playFromList(track, allTracks, playbackSource);
  }, [songs, source]);

  const handleDelete = useCallback(async (rs: UnifiedRecentSong) => {
    if (!token || !rs.audio_url) return;
    try {
      await songApi.delete(rs.id, token);
      setSongs(prev => {
        const updated = prev.filter(s => s.id !== rs.id);
        _cache.set(cacheKey, { songs: updated, key: refreshKey });
        return updated;
      });
      showToast('Song deleted');
    } catch {
      showToast('Failed to delete song', 'error');
    }
  }, [token, showToast, cacheKey, refreshKey]);

  const handleDownload = useCallback((rs: UnifiedRecentSong) => {
    if (!rs.audio_url) return;
    downloadTrack(songFromRecent(rs), { artistName: rs.artist_name || '' });
  }, []);

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
                {coverUrl ? (
                  <img src={coverUrl} alt="" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <Music className="w-5 h-5 text-zinc-600" />
                  </div>
                )}
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
