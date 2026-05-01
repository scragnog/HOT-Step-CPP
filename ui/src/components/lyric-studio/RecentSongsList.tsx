/**
 * RecentSongsList.tsx — Shows recently generated songs across ALL Lireek artists.
 *
 * Data flow: GET /api/lireek/recent-songs returns rows with pre-resolved audio_url + cover_url.
 * MODULE-LEVEL CACHE ensures instant render on navigation.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Loader2, Music, Download, Trash2, ListPlus, Check } from 'lucide-react';
import { lireekApi } from '../../services/lireekApi';
import type { RecentSong } from '../../services/lireekApi';
// songApi removed — not used in this component
import { useAuth } from '../../context/AuthContext';
import type { Song } from '../../types';
import { DownloadModal } from '../shared/DownloadModal';
import { usePlaylist } from './playlistStore';
import { playFromList, recentSongToTrack } from '../../stores/playbackStore';

interface RecentSongsListProps {
  showToast: (msg: string, type?: 'success' | 'error') => void;
  refreshKey?: number;
  compact?: boolean;
}

// ── Module-level cache ───────────────────────────────────────────────────────

let _cachedSongs: RecentSong[] = [];
let _cachedRefreshKey = -1;
let _fetchInFlight = false;

async function _loadRecentSongs(): Promise<RecentSong[]> {
  const res = await lireekApi.getRecentSongs(50);
  return (res.songs || []).filter(s => !!s.audio_url);
}

// ── Component ────────────────────────────────────────────────────────────────

export const RecentSongsList: React.FC<RecentSongsListProps> = ({ showToast, refreshKey = 0, compact = false }) => {
  const { token } = useAuth();
  const [songs, setSongs] = useState<RecentSong[]>(_cachedSongs);
  const [loading, setLoading] = useState(_cachedSongs.length === 0);
  const mountedRef = useRef(true);
  const [downloadSong, setDownloadSong] = useState<Song | null>(null);
  const [downloadArtist, setDownloadArtist] = useState('');

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      // Invalidate cache on unmount so returning to Lyric Studio fetches fresh data
      // (handles cross-page deletions, e.g. deleting songs from the Create page)
      _cachedRefreshKey = -1;
    };
  }, []);

  useEffect(() => {
    if (!token) { setLoading(false); return; }
    if (_cachedRefreshKey === refreshKey && _cachedSongs.length > 0) return;
    if (_fetchInFlight) return;
    if (_cachedSongs.length === 0) setLoading(true);

    _fetchInFlight = true;
    _loadRecentSongs().then(resolved => {
      _cachedSongs = resolved;
      _cachedRefreshKey = refreshKey;
      _fetchInFlight = false;
      if (mountedRef.current) { setSongs(resolved); setLoading(false); }
    }).catch(() => {
      _fetchInFlight = false;
      if (mountedRef.current) setLoading(false);
    });
  }, [refreshKey, token]);

  const handlePlay = useCallback(async (rs: RecentSong) => {
    const track = recentSongToTrack(rs);
    const allTracks = songs.map(recentSongToTrack);
    playFromList(track, allTracks, 'lireek-recent');
  }, [songs]);

  const handleDelete = useCallback(async (e: React.MouseEvent, rs: RecentSong) => {
    e.stopPropagation();
    if (!token || !rs.audio_url) return;
    try {
      // Delete the audio generation record from lireek DB
      if (rs.ag_id) {
        await lireekApi.deleteAudioGeneration(rs.ag_id);
      }
      setSongs(prev => {
        const updated = prev.filter(s => s.ag_id !== rs.ag_id);
        _cachedSongs = updated;
        return updated;
      });
      showToast('Song deleted');
    } catch {
      showToast('Failed to delete song', 'error');
    }
  }, [token, showToast]);

  const handleDownloadClick = useCallback(async (e: React.MouseEvent, rs: RecentSong) => {
    e.stopPropagation();
    if (!rs.audio_url) return;
    // Build a Song object for the DownloadModal
    const song: Song = {
      id: rs.hotstep_job_id || `recent-${rs.ag_id}`,
      title: rs.song_title || 'Untitled',
      style: rs.caption || '',
      caption: rs.caption || '',
      lyrics: rs.lyrics || '',
      audioUrl: rs.audio_url || '',
      coverUrl: rs.cover_url || '',
      duration: rs.duration || 0,
      tags: [],
      masteredAudioUrl: rs.mastered_audio_url || '',
    };
    setDownloadSong(song);
    setDownloadArtist(rs.artist_name || '');
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
          const coverUrl = rs.cover_url || rs.album_image || rs.artist_image || '';
          return (
            <div key={rs.ag_id}
              className="flex items-center gap-2.5 rounded-lg hover:bg-white/[0.06] transition-colors text-left group px-2 overflow-hidden relative cursor-pointer"
              onClick={() => handlePlay(rs)}>
              <div className="w-14 h-14 rounded-md flex-shrink-0 overflow-hidden bg-zinc-800 relative">
                {coverUrl ? (
                  <img src={coverUrl} alt="" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <Music className="w-5 h-5 text-zinc-600" />
                  </div>
                )}
                <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <Play className="w-4 h-4 text-white ml-0.5" />
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-zinc-200 truncate leading-snug">
                  {rs.song_title || 'Untitled'}
                </p>
                <p className="text-[10px] text-zinc-500 truncate leading-snug">{rs.artist_name}</p>
                {dur > 0 && (
                  <p className="text-[10px] text-zinc-600 font-mono mt-0.5">{mins}:{secs}</p>
                )}
              </div>
              <div className="absolute right-1 top-1/2 -translate-y-1/2 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <AddToPlaylistBtn rs={rs} />
                <button onClick={(e) => handleDownloadClick(e, rs)}
                  className="p-1.5 rounded-md bg-zinc-800/80 hover:bg-zinc-700 text-zinc-400 hover:text-white transition-colors"
                  title="Download">
                  <Download className="w-3 h-3" />
                </button>
                <button onClick={(e) => handleDelete(e, rs)}
                  className="p-1.5 rounded-md bg-zinc-800/80 hover:bg-red-900/60 text-zinc-400 hover:text-red-400 transition-colors"
                  title="Delete">
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Download Modal */}
      {downloadSong && (
        <DownloadModal
          song={downloadSong}
          isOpen={!!downloadSong}
          onClose={() => setDownloadSong(null)}
          artistName={downloadArtist}
        />
      )}
    </>
  );
};

// ── Add-to-playlist helper ───────────────────────────────────────────────────

const AddToPlaylistBtn: React.FC<{ rs: RecentSong }> = ({ rs }) => {
  const playlist = usePlaylist();
  const itemId = String(rs.ag_id) || `recent-${rs.song_title}`;
  const inPlaylist = playlist.isIn(itemId);

  const toggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (inPlaylist) {
      playlist.remove(itemId);
    } else {
      playlist.add({
        id: itemId,
        title: rs.song_title || 'Untitled',
        audioUrl: rs.audio_url || '',
        masteredAudioUrl: rs.mastered_audio_url || '',
        artistName: rs.artist_name || '',
        coverUrl: rs.cover_url || rs.album_image || rs.artist_image || '',
        duration: rs.duration || 0,
      });
    }
  };

  return (
    <button onClick={toggle}
      className={`p-1.5 rounded-md transition-colors ${
        inPlaylist ? 'bg-pink-500/20 text-pink-400 hover:bg-pink-500/30'
          : 'bg-zinc-800/80 hover:bg-zinc-700 text-zinc-400 hover:text-pink-400'
      }`}
      title={inPlaylist ? 'Remove from playlist' : 'Add to playlist'}>
      {inPlaylist ? <Check className="w-3 h-3" /> : <ListPlus className="w-3 h-3" />}
    </button>
  );
};
