/**
 * InlineAudioQueue.tsx — Inline audio generation queue display.
 * Shows active, pending, and completed audio generation jobs from the audioGenQueueStore.
 *
 * Order: Active (generating/loading) → Queued (pending) → Completed (succeeded/failed, newest first)
 * Completed items play through the main player (onPlaySong).
 */

import React, { useCallback } from 'react';
import { Loader2, CheckCircle2, XCircle, X, Music, Clock, Play, Square, ListPlus, Check } from 'lucide-react';
import {
  useAudioGenQueue,
  removeFromAudioQueue,
  clearFinishedFromAudioQueue,
  forceFailQueueItem,
} from '../../stores/audioGenQueueStore';
import type { AudioQueueItem } from '../../stores/audioGenQueueStore';
import { usePlaylist } from './playlistStore';
import type { Song } from '../../types';

interface InlineAudioQueueProps {
  onPlaySong?: (song: Song) => void;
  currentSongId?: string | null;
}

export const InlineAudioQueue: React.FC<InlineAudioQueueProps> = ({ onPlaySong, currentSongId }) => {
  const { items } = useAudioGenQueue();

  const active = items.filter(i => i.status === 'loading-adapter' || i.status === 'generating');
  const queued = items.filter(i => i.status === 'pending');
  const finished = items
    .filter(i => i.status === 'succeeded' || i.status === 'failed')
    .slice().reverse();

  const pendingCount = queued.length;
  const finishedCount = finished.length;

  const handlePlay = useCallback((item: AudioQueueItem) => {
    if (!item.audioUrl || !onPlaySong) return;
    const song: Song = {
      id: item.songId || item.id,
      title: item.generation.title || 'Untitled',
      lyrics: '',
      style: item.generation.caption || '',
      caption: item.generation.caption || '',
      audioUrl: item.audioUrl,
      masteredAudioUrl: item.masteredAudioUrl || '',
      coverUrl: '',
      duration: item.audioDuration || 0,
      tags: [],
    };
    onPlaySong(song);
  }, [onPlaySong]);

  if (items.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center px-4">
        <Music className="w-5 h-5 text-zinc-600 mb-2" />
        <p className="text-xs text-zinc-500">No queued generations</p>
      </div>
    );
  }

  return (
    <div className="space-y-1 px-2">
      <div className="flex items-center justify-between px-1 py-1">
        <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">
          {pendingCount > 0 && `${pendingCount} pending`}
          {pendingCount > 0 && finishedCount > 0 && ' · '}
          {finishedCount > 0 && `${finishedCount} done`}
        </span>
        {finishedCount > 0 && (
          <button onClick={clearFinishedFromAudioQueue}
            className="text-[10px] text-zinc-500 hover:text-red-400 transition-colors">
            Clear Done
          </button>
        )}
      </div>

      {active.length > 0 && (
        <>
          <GroupLabel label="Active" color="text-pink-400" />
          {active.map(item => (
            <QueueItemRow key={item.id} item={item} isPlayingInMain={currentSongId === item.id} onPlay={handlePlay} />
          ))}
        </>
      )}

      {queued.length > 0 && (
        <>
          <GroupLabel label="Queued" color="text-zinc-400" />
          {queued.map(item => (
            <QueueItemRow key={item.id} item={item} isPlayingInMain={currentSongId === item.id} onPlay={handlePlay} />
          ))}
        </>
      )}

      {finished.length > 0 && (
        <>
          <GroupLabel label="Completed" color="text-green-400" />
          {finished.map(item => (
            <QueueItemRow key={item.id} item={item} isPlayingInMain={currentSongId === item.id} onPlay={handlePlay} />
          ))}
        </>
      )}
    </div>
  );
};

const GroupLabel: React.FC<{ label: string; color: string }> = ({ label, color }) => (
  <p className={`text-[9px] font-bold uppercase tracking-widest ${color} px-1 pt-2 pb-0.5`}>
    {label}
  </p>
);

interface QueueItemRowProps {
  item: AudioQueueItem;
  isPlayingInMain: boolean;
  onPlay: (item: AudioQueueItem) => void;
}

const QueueItemRow: React.FC<QueueItemRowProps> = ({ item, isPlayingInMain, onPlay }) => {
  const isRunning = item.status === 'loading-adapter' || item.status === 'generating';
  const isSucceeded = item.status === 'succeeded';
  const isFailed = item.status === 'failed';
  const isPending = item.status === 'pending';

  // Show audio duration for completed items, generation elapsed for running/pending
  const displaySeconds = isSucceeded && item.audioDuration ? item.audioDuration : (item.elapsed || 0);
  const mins = Math.floor(displaySeconds / 60);
  const secs = displaySeconds % 60;
  const timeStr = displaySeconds > 0
    ? `${mins}:${String(Math.floor(secs)).padStart(2, '0')}`
    : '';

  const borderColor = isSucceeded ? 'border-green-500/20'
    : isFailed ? 'border-red-500/20'
    : isRunning ? 'border-pink-500/20'
    : 'border-white/5';

  return (
    <div className={`rounded-lg border ${borderColor} bg-white/[0.02] px-3 py-2 transition-all ${isPlayingInMain ? 'ring-1 ring-pink-500/40 bg-pink-500/5' : ''}`}>
      <div className="flex items-center gap-2">
        <div className="flex-shrink-0">
          {isPending && <div className="w-2 h-2 rounded-full bg-zinc-500" />}
          {isRunning && <Loader2 className="w-3.5 h-3.5 text-pink-400 animate-spin" />}
          {isSucceeded && item.audioUrl ? (
            <button onClick={() => onPlay(item)}
              className={`p-0.5 rounded-full transition-colors ${isPlayingInMain ? 'bg-green-500/20 text-green-300' : 'text-green-400 hover:bg-green-500/20'}`}
              title={isPlayingInMain ? 'Playing' : 'Play'}>
              {isPlayingInMain ? <Square className="w-3 h-3" /> : <Play className="w-3 h-3" />}
            </button>
          ) : isSucceeded ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
          ) : null}
          {isFailed && <XCircle className="w-3.5 h-3.5 text-red-400" />}
        </div>

        <div className="flex-1 min-w-0">
          <p className={`text-xs font-medium truncate ${isPlayingInMain ? 'text-pink-300' : 'text-zinc-200'}`}>
            {item.generation.title || 'Untitled'}
          </p>
          <p className="text-[10px] text-zinc-500 truncate">{item.artistName}</p>
        </div>

        <div className="flex items-center gap-1 flex-shrink-0">
          {timeStr && (
            <span className="text-[10px] text-zinc-500 font-mono flex items-center gap-0.5">
              <Clock className="w-2.5 h-2.5" />{timeStr}
            </span>
          )}
          {isPending && (
            <button onClick={() => removeFromAudioQueue(item.id)}
              className="p-0.5 rounded hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-colors"
              title="Remove from queue">
              <X className="w-3 h-3" />
            </button>
          )}
          {isRunning && (
            <button onClick={() => forceFailQueueItem(item.id)}
              className="p-0.5 rounded hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-colors"
              title="Dismiss stuck item">
              <X className="w-3 h-3" />
            </button>
          )}
          {isSucceeded && item.audioUrl && <QueueAddToPlaylistBtn item={item} />}
        </div>
      </div>

      {isRunning && (
        <div className="mt-1.5 space-y-1">
          <div className="h-1 rounded-full bg-white/10 overflow-hidden">
            <div className={`h-full bg-gradient-to-r from-pink-500 to-purple-600 transition-all duration-500 ${!item.progress ? 'animate-pulse opacity-40 w-full' : ''}`}
              style={item.progress ? { width: `${item.progress}%` } : undefined} />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[9px] text-zinc-500">{item.stage || 'Processing…'}</span>
            {item.progress !== undefined && item.progress > 0 && (
              <span className="text-[9px] font-bold text-pink-400">{Math.round(item.progress)}%</span>
            )}
          </div>
        </div>
      )}

      {isFailed && item.error && (
        <p className="mt-1 text-[9px] text-red-400 truncate">{item.error}</p>
      )}
    </div>
  );
};

const QueueAddToPlaylistBtn: React.FC<{ item: AudioQueueItem }> = ({ item }) => {
  const playlist = usePlaylist();
  const inPlaylist = playlist.isIn(item.songId || item.id);

  const toggle = () => {
    const resolvedId = item.songId || item.id;
    if (inPlaylist) { playlist.remove(resolvedId); }
    else {
      playlist.add({
        id: resolvedId,
        title: item.generation.title || 'Untitled',
        audioUrl: item.audioUrl || '',
        artistName: item.artistName || '',
        coverUrl: '',
        duration: item.audioDuration || 0,
      });
    }
  };

  return (
    <button onClick={toggle}
      className={`p-0.5 rounded transition-colors ${inPlaylist ? 'text-pink-400 bg-pink-500/10' : 'text-zinc-600 hover:text-pink-400 hover:bg-pink-500/10'}`}
      title={inPlaylist ? 'Remove from playlist' : 'Add to playlist'}>
      {inPlaylist ? <Check className="w-3 h-3" /> : <ListPlus className="w-3 h-3" />}
    </button>
  );
};
