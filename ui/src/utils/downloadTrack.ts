// downloadTrack.ts — Instant download utility using settings-page preferences
//
// Reads format, bitrate, version, and latent settings from AppSettings (localStorage).
// Saves into the DOWNLOAD_DIR folder when one is set, else triggers browser downloads via hidden <a> clicks.

import type { Song } from '../types';
import { resolveExports } from '../services/exportImportApi';
import { HttpStatus } from '../services/httpClient';

type AudioFormat = 'wav' | 'flac' | 'opus' | 'mp3';
type DownloadVersion = 'original' | 'mastered' | 'both';

interface DownloadSettings {
  downloadFormat: AudioFormat;
  downloadMp3Bitrate: number;
  downloadOpusBitrate: number;
  downloadVersion: DownloadVersion;
  downloadIncludeLatent: boolean;
  downloadEnhancedLrc: boolean;
}

const DEFAULTS: DownloadSettings = {
  downloadFormat: 'flac',
  downloadMp3Bitrate: 192,
  downloadOpusBitrate: 192,
  downloadVersion: 'mastered',
  downloadIncludeLatent: false,
  downloadEnhancedLrc: false,
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

/**
 * Download one /api/download URL. With a download folder set (DOWNLOAD_DIR) the
 * server saves the file there and the path is announced to the app shell as a
 * `hotstep:download-saved` event; with none set (409) it is the browser download.
 */
async function triggerDownload(url: string): Promise<void> {
  const [pathname, query] = url.split('?');
  let res: Response;
  try {
    res = await fetch(`${pathname}/save${query ? `?${query}` : ''}`, { method: 'POST' });
  } catch {
    triggerBrowserDownload(url);
    return;
  }
  if (res.status === HttpStatus.conflict) { triggerBrowserDownload(url); return; }
  const body = await res.json().catch(() => null);
  window.dispatchEvent(new CustomEvent('hotstep:download-saved', {
    detail: res.ok ? { path: body?.path } : { error: body?.error || `HTTP ${res.status}` },
  }));
}

/** Small delay helper — used between multiple downloads so the browser doesn't block them. */
const delay = (ms: number) => new Promise<void>(r => setTimeout(r, ms));

/** A specific rendered variant of a track, matching the playbar's variant switch. */
export type TrackVersion = 'original' | 'mastered' | 'noadapter';

function artistFor(song: Song, options?: { artistName?: string }): string {
  return options?.artistName || song.artistName
    || (song.generationParams as any)?.artist || (song.generation_params as any)?.artist
    // Cover Studio records the covering band as artistName.
    || (song.generationParams as any)?.artistName || (song.generation_params as any)?.artistName || '';
}

/**
 * Build the /api/download query string for one version of a track.
 * Shared by the settings-driven download and the per-variant playbar buttons so
 * both produce identical filenames (prepend - artist - title[ - suffix]).
 */
function buildDownloadParams(
  song: Song,
  version: TrackVersion,
  settings: DownloadSettings,
  options?: { artistName?: string; prepend?: string },
): URLSearchParams {
  const format = settings.downloadFormat;
  const isLossy = format === 'mp3' || format === 'opus';
  const bitrate = format === 'opus' ? settings.downloadOpusBitrate : settings.downloadMp3Bitrate;

  const finalArtist = artistFor(song, options);
  const finalPrepend = options?.prepend || readFilenamePrepend();

  // The no-adapter render may live only on a queue item, not in the songs row —
  // pass it explicitly so the server can serve it either way.
  const noAdapterUrl = version === 'noadapter'
    ? ((song as any).noAdapterAudioUrl || (song as any).noadapter_audio_url || '')
    : '';

  return new URLSearchParams({
    format,
    version,
    ...(isLossy ? { bitrate: String(bitrate) } : {}),
    ...(song.audioUrl ? { audioUrl: song.audioUrl } : {}),
    ...(noAdapterUrl ? { srcUrl: noAdapterUrl } : {}),
    ...(finalArtist ? { artist: finalArtist } : {}),
    ...(finalPrepend ? { prepend: finalPrepend } : {}),
    // Only reaches the sidecar written beside a folder-mode download.
    ...(settings.downloadEnhancedLrc ? { enhanced: '1' } : {}),
  });
}

/**
 * Download one specific version of a track, bypassing the settings-page
 * "which version" preference. Format, bitrate, artist and filename prepend
 * still come from settings — this only pins which render is fetched.
 * Used by the playbar's per-variant download buttons.
 */
export async function downloadTrackVersion(
  song: Song,
  version: TrackVersion,
  options?: { artistName?: string; prepend?: string },
): Promise<void> {
  const params = buildDownloadParams(song, version, readSettings(), options);
  await triggerDownload(`/api/download/${song.id}?${params}`);
}

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
  const hasMastered = !!(song.masteredAudioUrl || song.mastered_audio_url);

  const buildParams = (version: TrackVersion) => buildDownloadParams(song, version, settings, options);

  // Determine which audio version(s) to download
  const versionPref = settings.downloadVersion;
  if (versionPref === 'both' && hasMastered) {
    await triggerDownload(`/api/download/${song.id}?${buildParams('original')}`);
    await delay(500);
    await triggerDownload(`/api/download/${song.id}?${buildParams('mastered')}`);
  } else if (versionPref === 'mastered' && hasMastered) {
    await triggerDownload(`/api/download/${song.id}?${buildParams('mastered')}`);
  } else {
    await triggerDownload(`/api/download/${song.id}?${buildParams('original')}`);
  }

  // Latent download — if enabled and the song has a latent file
  if (settings.downloadIncludeLatent && (song.latentUrl || song.latent_url)) {
    await delay(500);
    const latentParams = new URLSearchParams({ version: 'latent' });
    await triggerDownload(`/api/download/${song.id}?${latentParams}`);
  }
}

/**
 * Download a track's synced lyrics as an `.lrc` sidecar — the file external
 * players, DAWs and datasets read.
 *
 * Goes through the server so a DOWNLOAD_DIR folder is honoured, and so a track
 * whose only timing data is `.lyrics.json` still exports: the server rebuilds
 * the LRC when no `.lrc` was ever written.
 */
export async function downloadLyricsLrc(
  song: Song,
  options?: { artistName?: string; prepend?: string },
): Promise<void> {
  const artist = artistFor(song, options);
  const prepend = options?.prepend || readFilenamePrepend();
  const settings = readSettings();
  // Name the .lrc after the same render the Download action would fetch: the
  // unmastered file carries a " - Unmastered" suffix and a player pairs a
  // sidecar by exact filename, so the two names have to agree.
  const version: TrackVersion = settings.downloadVersion === 'original' ? 'original' : 'mastered';
  const params = new URLSearchParams({
    version,
    ...(settings.downloadEnhancedLrc ? { enhanced: '1' } : {}),
    ...(song.audioUrl ? { audioUrl: song.audioUrl } : {}),
    ...(artist ? { artist } : {}),
    ...(prepend ? { prepend } : {}),
  });
  await triggerDownload(`/api/download/${song.id}/lrc?${params}`);
}

/**
 * The same `.lrc` export for a surface that only knows the audio URL, such as
 * the lyric editor. The server resolves the song row from `audioUrl`, so the
 * filename, title and saved artist still come out right.
 */
export async function downloadLyricsLrcForAudio(audioUrl: string, songId?: string): Promise<void> {
  const settings = readSettings();
  const version: TrackVersion = settings.downloadVersion === 'original' ? 'original' : 'mastered';
  const prepend = readFilenamePrepend();
  const params = new URLSearchParams({
    version,
    ...(settings.downloadEnhancedLrc ? { enhanced: '1' } : {}),
    ...(audioUrl ? { audioUrl } : {}),
    ...(prepend ? { prepend } : {}),
  });
  await triggerDownload(`/api/download/${songId || '_'}/lrc?${params}`);
}

/**
 * Bulk-download multiple tracks. The server resolves which versions exist and
 * which files to fetch; the browser only triggers the returned URLs, 800ms
 * apart so it doesn't throttle/block them.
 *
 * @returns one message per item the server could not resolve
 */
export async function downloadAll(
  songs: Song[],
  options?: { artistName?: string; prepend?: string },
): Promise<string[]> {
  const settings = readSettings();
  const format = settings.downloadFormat;
  const bitrate = format === 'opus' ? settings.downloadOpusBitrate
    : format === 'mp3' ? settings.downloadMp3Bitrate : undefined;
  const prepend = options?.prepend || readFilenamePrepend();

  // ponytail: one request per artist because the contract takes a single artist
  // and songs has no artist column; a per-item artist field would collapse this.
  const byArtist = new Map<string, Song[]>();
  for (const song of songs) {
    const artist = artistFor(song, options);
    byArtist.set(artist, [...(byArtist.get(artist) || []), song]);
  }

  const failures: string[] = [];
  let triggered = 0;
  for (const [artist, group] of byArtist) {
    for (let offset = 0; offset < group.length; offset += 100) {
      const batch = group.slice(offset, offset + 100);
      const results = await resolveExports(
        batch.map(song => ({ songId: song.id, ...(song.audioUrl ? { audioUrl: song.audioUrl } : {}) })),
        {
          format, bitrate, downloadVersion: settings.downloadVersion, includeLatent: settings.downloadIncludeLatent,
          ...(artist ? { artist } : {}), ...(prepend ? { prepend } : {}),
        },
      );
      for (const result of results) {
        if (!result.url) {
          failures.push(`${batch[result.index]?.title || result.songId}: ${result.error || 'not available'}`);
          continue;
        }
        if (triggered++ > 0) await delay(800);
        await triggerDownload(result.url);
      }
    }
  }
  return failures;
}
