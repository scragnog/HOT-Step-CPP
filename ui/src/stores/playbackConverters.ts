/**
 * playbackConverters.ts — Track format converters for the playback system.
 *
 * Pure functions that convert various song/item shapes into the canonical
 * PlaybackTrack format used by playbackStore. Zero dependencies on store state.
 */

import type { Song } from '../types';
import type { PlaylistItem } from '../components/lyric-studio/playlistStore';
import type { PlaybackTrack } from './playbackStore';

// ── Duration Coercion ────────────────────────────────────────────────────────

function coerceDuration(d: string | number | undefined | null): number | undefined {
  if (d == null) return undefined;
  if (typeof d === 'number') return d;
  if (typeof d === 'string') {
    if (d.includes(':')) {
      const [m, s] = d.split(':').map(Number);
      return (m || 0) * 60 + (s || 0);
    }
    const n = parseFloat(d);
    return isNaN(n) ? undefined : n;
  }
  return undefined;
}

// ── Converters ───────────────────────────────────────────────────────────────

export function songToTrack(song: Song): PlaybackTrack {
  return {
    id: song.id,
    title: song.title || 'Untitled',
    audioUrl: song.audioUrl || song.audio_url || '',
    masteredAudioUrl: song.masteredAudioUrl || song.mastered_audio_url || '',
    artistName: song.artistName || '',
    coverUrl: song.coverUrl || song.cover_url || '',
    duration: coerceDuration(song.duration),
    style: song.style || '',
    lyrics: song.lyrics || '',
    caption: song.caption || '',
    generationParams: song.generationParams || song.generation_params as any,
  };
}

export function playlistItemToTrack(item: PlaylistItem): PlaybackTrack {
  return {
    id: item.id,
    title: item.title || 'Untitled',
    audioUrl: item.audioUrl || '',
    masteredAudioUrl: item.masteredAudioUrl || '',
    artistName: item.artistName || '',
    coverUrl: item.coverUrl || '',
    duration: coerceDuration(item.duration),
    style: item.style || '',
    generationParams: item.generationParams,
  };
}

// Generic converter for RecentSong-shaped objects (lireekApi types)
export function recentSongToTrack(rs: {
  ag_id?: number;
  hotstep_job_id?: string;
  song_title?: string;
  audio_url?: string;
  mastered_audio_url?: string;
  artist_name?: string;
  cover_url?: string;
  album_image?: string;
  artist_image?: string;
  duration?: number;
  caption?: string;
  lyrics?: string;
}): PlaybackTrack {
  return {
    id: rs.hotstep_job_id || `recent-${rs.ag_id}`,
    title: rs.song_title || 'Untitled',
    audioUrl: rs.audio_url || '',
    masteredAudioUrl: rs.mastered_audio_url || '',
    artistName: rs.artist_name || '',
    coverUrl: rs.cover_url || rs.album_image || rs.artist_image || '',
    duration: coerceDuration(rs.duration),
    caption: rs.caption || '',
    lyrics: rs.lyrics || '',
  };
}

// Converter for UnifiedRecentSong objects (from /api/songs/recent)
export function unifiedRecentSongToTrack(rs: {
  id: string;
  title: string;
  audio_url: string;
  mastered_audio_url?: string;
  artist_name?: string;
  cover_url?: string;
  artist_image?: string;
  duration?: number;
  caption?: string;
  lyrics?: string;
  style?: string;
}): PlaybackTrack {
  return {
    id: rs.id,
    title: rs.title || 'Untitled',
    audioUrl: rs.audio_url || '',
    masteredAudioUrl: rs.mastered_audio_url || '',
    artistName: rs.artist_name || '',
    coverUrl: rs.cover_url || rs.artist_image || '',
    duration: coerceDuration(rs.duration),
    caption: rs.caption || '',
    lyrics: rs.lyrics || '',
    style: rs.style || '',
  };
}

// Generic converter for AudioQueueItem-shaped objects
export function audioQueueItemToTrack(item: {
  id: string;
  songId?: string;
  audioUrl?: string;
  masteredAudioUrl?: string;
  artistName?: string;
  artistImageUrl?: string;
  coverUrl?: string;
  audioDuration?: number;
  generation: { title?: string; caption?: string };
}): PlaybackTrack {
  return {
    id: item.songId || item.id,
    title: item.generation.title || 'Untitled',
    audioUrl: item.audioUrl || '',
    masteredAudioUrl: item.masteredAudioUrl || '',
    artistName: item.artistName || '',
    coverUrl: item.coverUrl || item.artistImageUrl || '',
    duration: coerceDuration(item.audioDuration),
    caption: item.generation.caption || '',
  };
}
