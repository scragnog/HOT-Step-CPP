/**
 * PlaylistSidebar.tsx — Global playlist panel, rendered as a right-side sidebar.
 *
 * Replaces the floating playlist. Lives in App.tsx so it's accessible from any view.
 * Track list rendering adapted from the original FloatingPlaylist.
 */

import React, { useCallback, useMemo, useState } from 'react';
import {
  Play, X, Trash2, ChevronUp, ChevronDown,
  Music, ListPlus, ListMusic, Square, Download,
} from 'lucide-react';
import { usePlaylist, type PlaylistItem } from '../lyric-studio/playlistStore';
import { playFromList, playlistItemToTrack, usePlayback } from '../../stores/playbackStore';
import { DownloadModal } from '../shared/DownloadModal';
import type { Song } from '../../types';

interface PlaylistSidebarProps {
  onClose: () => void;
}

export const PlaylistSidebar: React.FC<PlaylistSidebarProps> = ({ onClose }) => {
  const playlist = usePlaylist();
  const pb = usePlayback();
  const currentSongId = pb.currentTrack?.id ?? null;
  const [downloadSong, setDownloadSong] = useState<Song | null>(null);

  const handlePlay = useCallback((item: PlaylistItem) => {
    const allTracks = playlist.items.map(playlistItemToTrack);
    const clickedTrack = playlistItemToTrack(item);
    playFromList(clickedTrack, allTracks, 'playlist');
  }, [playlist.items]);

  const totalDuration = useMemo(() => {
    const total = playlist.items.reduce((sum, i) => sum + (i.duration || 0), 0);
    if (total <= 0) return '0:00';
    return `${Math.floor(total / 60)}:${String(Math.floor(total % 60)).padStart(2, '0')}`;
  }, [playlist.items]);

  return (
    <div className="flex flex-col h-full bg-zinc-950/80 backdrop-blur-sm">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/5 flex-shrink-0">
        <div className="flex items-center gap-2">
          <ListMusic className="w-4 h-4 text-pink-400" />
          <span className="text-xs font-bold text-zinc-200 uppercase tracking-wider">Playlist</span>
          {playlist.items.length > 0 && (
            <span className="text-[10px] text-zinc-500 font-normal">
              {playlist.items.length} · {totalDuration}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {playlist.items.length > 0 && (
            <button onClick={playlist.clear}
              className="p-1 rounded-md text-zinc-600 hover:text-red-400 hover:bg-red-500/10 transition-colors"
              title="Clear playlist">
              <Trash2 className="w-3 h-3" />
            </button>
          )}
          <button onClick={onClose}
            className="p-1 rounded-md text-zinc-500 hover:text-white hover:bg-white/10 transition-colors"
            title="Close playlist">
            <X className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Track list */}
      <div className="flex-1 min-h-0 overflow-y-auto scrollbar-hide">
        {playlist.items.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-6 py-8 gap-3">
            <div className="w-14 h-14 rounded-full bg-zinc-900 flex items-center justify-center border border-white/5">
              <Music className="w-6 h-6 text-zinc-700" />
            </div>
            <div>
              <p className="text-xs text-zinc-500 font-medium mb-1">Playlist empty</p>
              <p className="text-[10px] text-zinc-600 leading-relaxed max-w-[200px]">
                Click the <ListPlus className="w-3 h-3 inline text-pink-400" /> button next to any track to add it here.
              </p>
            </div>
          </div>
        ) : (
          <div className="divide-y divide-white/5">
            {playlist.items.map((item, idx) => {
              const isPlaying = currentSongId === item.id;
              return (
                <div key={item.id}
                  className={`group flex items-center gap-2 px-3 py-2 hover:bg-white/[0.04] transition-colors ${isPlaying ? 'bg-pink-500/10' : ''}`}>
                  {/* Track number / play button */}
                  <div className="w-6 flex-shrink-0 flex items-center justify-center">
                    <span className={`text-[10px] font-mono group-hover:hidden ${isPlaying ? 'text-pink-400 font-bold' : 'text-zinc-600'}`}>
                      {idx + 1}
                    </span>
                    <button onClick={() => handlePlay(item)}
                      className="hidden group-hover:flex w-5 h-5 items-center justify-center rounded-full hover:bg-white/10 transition-colors"
                      title={isPlaying ? 'Now Playing' : 'Play'}>
                      {isPlaying
                        ? <Square className="w-2.5 h-2.5 text-pink-400" />
                        : <Play className="w-2.5 h-2.5 text-white ml-0.5" />}
                    </button>
                  </div>

                  {/* Cover art */}
                  {item.coverUrl ? (
                    <div className="w-9 h-9 rounded-md overflow-hidden flex-shrink-0 bg-zinc-800">
                      <img src={item.coverUrl} alt="" className="w-full h-full object-cover" />
                    </div>
                  ) : (
                    <div className="w-9 h-9 rounded-md flex-shrink-0 bg-zinc-800 flex items-center justify-center">
                      <Music className="w-3.5 h-3.5 text-zinc-600" />
                    </div>
                  )}

                  {/* Title / Artist */}
                  <div className="flex-1 min-w-0 cursor-pointer" onClick={() => handlePlay(item)}>
                    <p className={`text-[11px] font-medium truncate leading-tight ${isPlaying ? 'text-pink-300' : 'text-zinc-200'}`}>
                      {item.title || 'Untitled'}
                    </p>
                    {item.artistName && (
                      <p className="text-[9px] text-zinc-500 truncate leading-tight">{item.artistName}</p>
                    )}
                  </div>

                  {/* Actions — appear on hover between title and duration */}
                  <div className="hidden group-hover:flex items-center gap-0 flex-shrink-0">
                    <button onClick={(e) => {
                        e.stopPropagation();
                        setDownloadSong({
                          id: item.id,
                          title: item.title || 'Untitled',
                          style: item.style || '',
                          caption: item.style || '',
                          lyrics: '',
                          audioUrl: item.audioUrl,
                          masteredAudioUrl: item.masteredAudioUrl || '',
                          coverUrl: item.coverUrl || '',
                          duration: item.duration || 0,
                          artistName: item.artistName || '',
                          tags: [],
                        });
                      }}
                      className="p-0.5 text-zinc-600 hover:text-emerald-400 transition-colors" title="Download">
                      <Download className="w-3 h-3" />
                    </button>
                    <button onClick={() => playlist.move(item.id, 'up')} disabled={idx === 0}
                      className="p-0.5 text-zinc-600 hover:text-white transition-colors disabled:opacity-20" title="Move up">
                      <ChevronUp className="w-3 h-3" />
                    </button>
                    <button onClick={() => playlist.move(item.id, 'down')} disabled={idx === playlist.items.length - 1}
                      className="p-0.5 text-zinc-600 hover:text-white transition-colors disabled:opacity-20" title="Move down">
                      <ChevronDown className="w-3 h-3" />
                    </button>
                    <button onClick={() => playlist.remove(item.id)}
                      className="p-0.5 text-zinc-600 hover:text-red-400 transition-colors ml-0.5" title="Remove">
                      <X className="w-3 h-3" />
                    </button>
                  </div>

                  {/* Duration — always visible, pinned right */}
                  {item.duration && item.duration > 0 && (
                    <span className="text-[9px] text-zinc-600 font-mono flex-shrink-0 tabular-nums w-8 text-right ml-auto">
                      {Math.floor(item.duration / 60)}:{String(Math.floor(item.duration % 60)).padStart(2, '0')}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer */}
      {playlist.items.length > 0 && (
        <div className="flex items-center justify-between px-4 py-2 border-t border-white/5 flex-shrink-0">
          <button onClick={() => handlePlay(playlist.items[0])}
            className="flex items-center gap-1.5 text-[10px] font-semibold text-pink-400 hover:text-pink-300 transition-colors">
            <Play className="w-3 h-3" /> Play All
          </button>
          <span className="text-[9px] text-zinc-600 font-mono">{totalDuration}</span>
        </div>
      )}

      {/* Download Modal */}
      {downloadSong && (
        <DownloadModal
          song={downloadSong}
          isOpen={!!downloadSong}
          onClose={() => setDownloadSong(null)}
          artistName={downloadSong.artistName}
        />
      )}
    </div>
  );
};
