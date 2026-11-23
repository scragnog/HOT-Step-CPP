// library.ts — songs, the render queue, and starting and stopping renders.

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useAuth } from '../context/AuthContext';
import { songsOptions, recentSongsOptions } from '../queries/shell-queue/songs';
import { forceFailQueueItem, getAudioQueueOwner, removeFromAudioQueue, useAudioGenQueueSelector, type AudioQueueItem } from '../stores/audioGenQueueStore';
import { activeRuntime } from '../app/skinContext';
import type { Song, UnifiedRecentSong } from '../types';
import type { GenerateRequest, QueueState, SkinQueueItem, SkinSong } from './types';

const NO_SONGS: SkinSong[] = [];

const fromSong = (s: Song): SkinSong => ({
  id: s.id, title: s.title, caption: s.caption || s.style || '', lyrics: s.lyrics || '',
  audioUrl: s.audioUrl || s.audio_url || '', coverUrl: s.coverUrl || s.cover_url || '',
  duration: Number(s.duration) || 0, artistName: s.artistName || '',
  createdAt: String(s.createdAt ?? s.created_at ?? ''),
});
const fromRecent = (s: UnifiedRecentSong): SkinSong => ({
  id: s.id, title: s.title, caption: s.caption || s.style || '', lyrics: s.lyrics || '',
  audioUrl: s.mastered_audio_url || s.audio_url || '', coverUrl: s.cover_url || '',
  duration: s.duration || 0, artistName: s.artist_name || '', createdAt: s.created_at || '',
});

/** The whole library, newest first. */
export function useSongs(): { songs: SkinSong[]; loading: boolean } {
  const { token } = useAuth();
  const q = useQuery({ ...songsOptions(), enabled: !!token });
  const songs = useMemo(() => q.data?.songs.map(fromSong) ?? NO_SONGS, [q.data]);
  return { songs, loading: q.isLoading };
}

/** The latest 50 renders from one source: 'all' (default), 'create', 'lyric-studio' or 'cover-studio'. */
export function useRecentSongs(source = 'all'): { songs: SkinSong[]; loading: boolean } {
  const { token } = useAuth();
  const q = useQuery({ ...recentSongsOptions(source), enabled: !!token });
  const songs = useMemo(() => q.data?.songs.map(fromRecent) ?? NO_SONGS, [q.data]);
  return { songs, loading: q.isLoading };
}

const STATE: Record<AudioQueueItem['status'], QueueState> = {
  pending: 'waiting', 'loading-adapter': 'running', generating: 'running', succeeded: 'done', failed: 'failed',
};

/** The render queue, oldest first. `active` counts waiting and running items. */
export function useQueue(): { items: SkinQueueItem[]; active: number } {
  const raw = useAudioGenQueueSelector(s => s.items);
  return useMemo(() => {
    const items = raw.map<SkinQueueItem>(i => ({
      id: i.id, title: i.generation?.title || 'Untitled', state: STATE[i.status] ?? 'running',
      stage: i.stage ?? '', progress: i.progress ?? 0, error: i.error ?? '', songId: i.songId ?? '',
    }));
    return { items, active: items.filter(i => i.state === 'waiting' || i.state === 'running').length };
  }, [raw]);
}

/** Start a Create render. Returns at once; follow it with useQueue(). Call it from a click
 *  (the first call asks for notification permission, which needs a user gesture). */
export function generate(req: GenerateRequest): void {
  if (!activeRuntime) throw new Error('generate() needs a shell rendered by ShellHost');
  activeRuntime.generate({ ...req });
}

/** Stop a queue item, or drop it if it never started. */
export function cancel(queueItemId: string): void {
  forceFailQueueItem(queueItemId);
  if (getAudioQueueOwner() !== 'server') removeFromAudioQueue(queueItemId);
}
