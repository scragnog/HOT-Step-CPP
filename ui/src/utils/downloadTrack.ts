// downloadTrack.ts — Instant download utility using settings-page preferences
//
// Reads format, bitrate, version, and latent settings from AppSettings (localStorage).
// Triggers browser downloads via hidden <a> element clicks.

import type { Song } from '../types';

type AudioFormat = 'wav' | 'flac' | 'opus' | 'mp3';
type DownloadVersion = 'original' | 'mastered' | 'both';

interface DownloadSettings {
  downloadFormat: AudioFormat;
  downloadMp3Bitrate: number;
  downloadOpusBitrate: number;
  downloadVersion: DownloadVersion;
  downloadIncludeLatent: boolean;
}

const DEFAULTS: DownloadSettings = {
  downloadFormat: 'flac',
  downloadMp3Bitrate: 192,
  downloadOpusBitrate: 192,
  downloadVersion: 'mastered',
  downloadIncludeLatent: false,
};

/** Read current download settings from localStorage (same key as AppSettings). */
function readSettings(): DownloadSettings {
  try {
    const raw = localStorage.getItem('ace-settings');
    if (raw) {
      const parsed = JSON.parse(raw);
      return { ...DEFAULTS, ...parsed };
    }
  } catch { /* fallback to defaults */ }
  return { ...DEFAULTS };
}

/** Read the optional filename prepend from Lyric Studio's localStorage key. */
function readFilenamePrepend(): string {
  try {
    const raw = localStorage.getItem('lireek-downloadFilenamePrepend');
    return raw ? JSON.parse(raw) : '';
  } catch { return ''; }
}

/** Trigger a single file download via hidden <a> click. */
function triggerBrowserDownload(url: string): void {
  const a = document.createElement('a');
  a.href = url;
  a.download = ''; // Let server set the filename
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

/** Small delay helper — used between multiple downloads so the browser doesn't block them. */
const delay = (ms: number) => new Promise<void>(r => setTimeout(r, ms));

/**
 * Download a track instantly using settings-page preferences.
 *
 * @param song     The song to download
 * @param options  Optional overrides (artistName for covers, prepend for filename prefix)
 */
export async function downloadTrack(
  song: Song,
  options?: { artistName?: string; prepend?: string },
): Promise<void> {
  const settings = readSettings();
  const format = settings.downloadFormat;
  const isLossy = format === 'mp3' || format === 'opus';
  const bitrate = format === 'opus' ? settings.downloadOpusBitrate : settings.downloadMp3Bitrate;
  const hasMastered = !!(song.masteredAudioUrl || song.mastered_audio_url);

  const finalArtist = options?.artistName || song.artistName
    || (song.generationParams as any)?.artist || (song.generation_params as any)?.artist || '';
  const finalPrepend = options?.prepend || readFilenamePrepend();

  // Build common query params
  const buildParams = (version: 'original' | 'mastered') => {
    const params = new URLSearchParams({
      format,
      version,
      ...(isLossy ? { bitrate: String(bitrate) } : {}),
      ...(song.audioUrl ? { audioUrl: song.audioUrl } : {}),
      ...(finalArtist ? { artist: finalArtist } : {}),
      ...(finalPrepend ? { prepend: finalPrepend } : {}),
    });
    return params;
  };

  // Determine which audio version(s) to download
  const versionPref = settings.downloadVersion;
  if (versionPref === 'both' && hasMastered) {
    triggerBrowserDownload(`/api/download/${song.id}?${buildParams('original')}`);
    await delay(500);
    triggerBrowserDownload(`/api/download/${song.id}?${buildParams('mastered')}`);
  } else if (versionPref === 'mastered' && hasMastered) {
    triggerBrowserDownload(`/api/download/${song.id}?${buildParams('mastered')}`);
  } else {
    triggerBrowserDownload(`/api/download/${song.id}?${buildParams('original')}`);
  }

  // Latent download — if enabled and the song has a latent file
  if (settings.downloadIncludeLatent && (song.latentUrl || song.latent_url)) {
    await delay(500);
    const latentParams = new URLSearchParams({ version: 'latent' });
    triggerBrowserDownload(`/api/download/${song.id}?${latentParams}`);
  }
}

/**
 * Bulk-download multiple tracks sequentially.
 * Stagers downloads 800ms apart so the browser doesn't throttle/block them.
 */
export async function downloadAll(
  songs: Song[],
  options?: { artistName?: string; prepend?: string },
): Promise<void> {
  for (let i = 0; i < songs.length; i++) {
    if (i > 0) await delay(800);
    await downloadTrack(songs[i], options);
  }
}
