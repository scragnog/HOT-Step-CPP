// download.ts — Audio download route with format conversion + metadata embedding
//
// GET /api/songs/:id/download?format=wav|flac|opus|mp3&bitrate=192&version=original|mastered|noadapter
//
// Converts the source WAV to the requested format, embeds metadata tags
// and cover art (when available), and streams it back with a
// Content-Disposition header for browser download.

import { Router, type Request, type Response } from 'express';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { config, getFFmpegPath } from '../config.js';
import { getDb } from '../db/database.js';
import { markDownloadedByAudioUrl } from '../db/lireekDb.js';
import { saveIntoDir } from '../services/downloadSave.js';
import {
  gatherSongMetadata, buildMetadataArgs, buildCoverArtArgs,
  type AudioMetadata,
} from '../services/audioMetadata.js';
import { readTrackLrc } from '../services/lrcExport.js';

const execFileAsync = promisify(execFile);
const router = Router();

/** Get mp3-codec binary path (platform-aware: no .exe on macOS/Linux) */
function getMp3CodecPath(): string {
  const ext = process.platform === 'win32' ? '.exe' : '';
  return path.join(path.dirname(config.aceServer.exe), `mp3-codec${ext}`);
}

/** Convert WAV to target format with optional metadata embedding */
async function convertAudio(
  sourcePath: string,
  format: string,
  bitrate: number,
  outputPath: string,
  metadata?: AudioMetadata,
): Promise<void> {
  const hasMeta = !!metadata;

  // WAV without metadata — fast copy, no conversion
  if (format === 'wav' && !hasMeta) {
    fs.copyFileSync(sourcePath, outputPath);
    return;
  }

  // MP3 without metadata — use mp3-codec.exe if available (faster, no ffmpeg dep)
  // When metadata IS present, we must use ffmpeg so ID3 tags get written.
  if (format === 'mp3' && !hasMeta) {
    const codec = getMp3CodecPath();
    if (fs.existsSync(codec)) {
      await execFileAsync(codec, [
        '-i', sourcePath, '-o', outputPath, '-b', String(bitrate),
      ], { timeout: 120_000 });
      return;
    }
    // Fallback to ffmpeg
  }

  // All other cases use ffmpeg (conversion + metadata + cover art)
  const ffmpegPath = getFFmpegPath();
  if (!ffmpegPath) {
    // No ffmpeg — for WAV, fall back to raw copy (no metadata)
    if (format === 'wav') {
      fs.copyFileSync(sourcePath, outputPath);
      return;
    }
    throw new Error(`Cannot convert to ${format} — ffmpeg not available`);
  }

  // ── Build ffmpeg command ──
  const args = ['-y', '-i', sourcePath];

  // Cover art: add as second input (PNG → JPEG transcoded on-the-fly)
  let coverArtArgs: { inputArgs: string[]; outputArgs: string[] } = { inputArgs: [], outputArgs: [] };
  if (metadata?.coverArtPath) {
    coverArtArgs = buildCoverArtArgs(metadata.coverArtPath, format);
    args.push(...coverArtArgs.inputArgs);
  }

  // Audio codec selection
  switch (format) {
    case 'wav':
      // Re-encode through ffmpeg so INFO chunks get written
      args.push('-c:a', 'pcm_s16le');
      break;
    case 'flac':
      args.push('-c:a', 'flac', '-sample_fmt', 's32', '-compression_level', '8');
      break;
    case 'opus':
      args.push('-c:a', 'libopus', '-b:a', `${bitrate}k`);
      break;
    case 'mp3':
      args.push('-c:a', 'libmp3lame', '-b:a', `${bitrate}k`);
      break;
    default:
      throw new Error(`Unsupported format: ${format}`);
  }

  // Cover art output args (stream mapping, codec, disposition)
  if (coverArtArgs.outputArgs.length > 0) {
    args.push(...coverArtArgs.outputArgs);
  }

  // Metadata tags
  if (metadata) {
    args.push(...buildMetadataArgs(metadata, format));
  }

  args.push(outputPath);

  try {
    await execFileAsync(ffmpegPath, args, { timeout: 120_000 });
  } catch (err: any) {
    throw new Error(`ffmpeg conversion failed: ${err.message}`);
  }
}

/** Filename suffix marking which variant of the track this is. */
function versionSuffix(version: string): string {
  if (version === 'original') return ' - Unmastered';
  if (version === 'noadapter') return ' - No Adapter';
  return '';
}

/** MIME types for audio formats */
const mimeTypes: Record<string, string> = {
  wav: 'audio/wav',
  mp3: 'audio/mpeg',
  flac: 'audio/flac',
  opus: 'audio/ogg',
};

/** Put exported synced lyrics next to a saved audio file, same stem. */
function writeLrcBeside(savedAudioPath: string, lrcText: string): void {
  const dir = path.dirname(savedAudioPath);
  const stem = path.basename(savedAudioPath).replace(/\.[^.]+$/, '');
  const tempDir = path.join(config.data.dir, 'download_temp');
  fs.mkdirSync(tempDir, { recursive: true });
  const tempFile = path.join(tempDir, `lrc_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}.lrc`);
  fs.writeFileSync(tempFile, lrcText, 'utf-8');
  try {
    saveIntoDir(dir, `${stem}.lrc`, tempFile);
  } finally {
    try { fs.unlinkSync(tempFile); } catch {}
  }
}

/**
 * Hand the finished file to the caller: stream it as an attachment, or, in
 * save mode, copy it into the DOWNLOAD_DIR folder and answer with the path.
 * `tempFile` (when set) is removed once delivered. `sidecarLrc` (save mode
 * only) is written beside the audio as a matching `.lrc` file.
 */
function deliver(
  res: Response, save: boolean, filePath: string, filename: string, mime: string,
  tempFile?: string, sidecarLrc?: string,
): void {
  const cleanup = () => {
    if (!tempFile) return;
    try { fs.unlinkSync(tempFile); } catch {}
    try { fs.rmdirSync(path.dirname(tempFile)); } catch {}
  };
  if (save) {
    try {
      const saved = saveIntoDir(config.data.downloadDir, filename, filePath);
      // A sidecar that cannot be written must not fail the download itself.
      if (sidecarLrc) {
        try { writeLrcBeside(saved, sidecarLrc); } catch (err: any) {
          console.warn('[Download] LRC sidecar failed (non-fatal):', err.message);
        }
      }
      res.json({ path: saved, filename: path.basename(saved) });
    } catch (err: any) {
      console.error('[Download] Save failed:', err.message);
      res.status(500).json({ error: err.message });
    } finally { cleanup(); }
    return;
  }
  res.setHeader('Content-Type', mime);
  res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
  res.setHeader('Content-Length', fs.statSync(filePath).size);
  const stream = fs.createReadStream(filePath);
  stream.pipe(res);
  if (tempFile) { stream.on('end', cleanup); stream.on('error', cleanup); }
}

// GET /api/download/:id?format=wav&bitrate=192&version=original&artist=Name&prepend=Prefix
// version: original | mastered | noadapter | latent.  srcUrl is an optional explicit
// source override (used by the playbar's per-variant download buttons for tracks whose
// songs row is missing) — resolved by basename inside the audio dir, same as audioUrl.
// POST /api/download/:id/save takes the same query and writes the file into the
// DOWNLOAD_DIR setting instead of streaming it; 409 when no folder is set.
const downloadHandler = (save: boolean) => async (req: Request, res: Response) => {
  if (save && !config.data.downloadDir) {
    res.status(409).json({ error: 'No download folder is set' });
    return;
  }
  const { id } = req.params;
  const format = (req.query.format as string || 'wav').toLowerCase();
  const bitrate = parseInt(req.query.bitrate as string) || 192;
  const version = (req.query.version as string || 'original').toLowerCase();
  const artistName = (req.query.artist as string || '').trim();
  const prepend = (req.query.prepend as string || '').trim();
  const srcUrlParam = (req.query.srcUrl as string || '').trim();

  // Validate format
  if (!['wav', 'mp3', 'flac', 'opus'].includes(format)) {
    res.status(400).json({ error: `Invalid format: ${format}. Use wav, mp3, flac, or opus.` });
    return;
  }

  // Get song from DB — by ID first, then by audio_url (Lyric Studio queue items)
  const song = findSong(id, req.query.audioUrl as string);
  if (!song) {
    // Last resort: serve the audio file directly without DB metadata
    const audioUrlParam = (req.query.audioUrl as string) || srcUrlParam;
    if (audioUrlParam) {
      const filename = path.basename(srcUrlParam || audioUrlParam);
      const sourcePath = path.join(config.data.audioDir, filename);
      if (fs.existsSync(sourcePath)) {
        // Same marking as the main path — this branch serves Lyric Studio queue
        // items whose songs row isn't there, which is exactly the case that
        // matters most for tracking what's been kept.
        try { markDownloadedByAudioUrl(audioUrlParam); } catch { /* lireek tables may not exist */ }
        // Clean up parsed parameters just in case DB doesn't have standard naming
        const badPrefixes = /^_?(XL|STD)(\s*\(CPP\))?_?\s*-?\s*/i;
        const cleanPrepend = prepend.trim();
        const cleanArtist = artistName.replace(badPrefixes, '').trim();
        const titleSuffix = versionSuffix(version);
        const titleParts = [cleanPrepend, cleanArtist, 'Untitled'].filter(Boolean);
        const downloadFilename = `${titleParts.join(' - ')}${titleSuffix}.${format}`;
        if (format === 'wav' && sourcePath.endsWith('.wav')) {
          deliver(res, save, sourcePath, downloadFilename, mimeTypes.wav);
          return;
        }
        // Convert
        const tempDir = path.join(config.data.dir, 'download_temp');
        fs.mkdirSync(tempDir, { recursive: true });
        const tempFile = path.join(tempDir, `dl_${Date.now().toString(36)}.${format}`);
        await convertAudio(sourcePath, format, bitrate, tempFile);
        deliver(res, save, tempFile, downloadFilename, mimeTypes[format] || 'application/octet-stream', tempFile);
        return;
      }
    }
    res.status(404).json({ error: 'Song not found' });
    return;
  }

  // Latent download — raw HSLAT binary, no format conversion
  if (version === 'latent') {
    const latentUrl = song?.latent_url;
    if (!latentUrl) {
      res.status(404).json({ error: 'No latent file available for this track' });
      return;
    }
    const latentFilename = path.basename(latentUrl);
    const latentPath = path.join(config.data.audioDir, latentFilename);
    if (!fs.existsSync(latentPath)) {
      res.status(404).json({ error: 'Latent file not found on disk' });
      return;
    }
    const rawTitle = song.title || 'track';
    const downloadName = `${rawTitle.replace(/[^a-zA-Z0-9 _()-]/g, '').trim() || 'track'}.latent`;
    deliver(res, save, latentPath, downloadName, 'application/octet-stream');
    return;
  }

  // Flag the lyrics this track came from as having a version worth keeping.
  // Deliberately after the latent branch above — grabbing the latent file is not
  // the same as keeping the song. Best-effort: never fail a download over it.
  try { markDownloadedByAudioUrl(song.audio_url); } catch { /* lireek tables may not exist */ }

  // Determine which audio URL to use
  let audioUrl: string;
  if (version === 'mastered' && song.mastered_audio_url) {
    audioUrl = song.mastered_audio_url;
  } else if (version === 'noadapter') {
    // srcUrl covers queue-backed tracks whose no-adapter render never landed in the row
    audioUrl = song.noadapter_audio_url || srcUrlParam || song.audio_url;
  } else {
    audioUrl = song.audio_url;
  }

  if (!audioUrl) {
    res.status(404).json({ error: 'No audio file available' });
    return;
  }

  // Resolve to filesystem path
  const audioFilename = path.basename(audioUrl);
  const sourcePath = path.join(config.data.audioDir, audioFilename);

  if (!fs.existsSync(sourcePath)) {
    res.status(404).json({ error: `Audio file not found on disk: ${audioFilename}` });
    return;
  }

  // Gather metadata for embedding into the output file
  let metadata: AudioMetadata | undefined;
  try {
    metadata = gatherSongMetadata(song);
  } catch (metaErr: any) {
    // Non-fatal — proceed without metadata if gathering fails
    console.warn(`[Download] Metadata gathering failed (non-fatal): ${metaErr.message}`);
  }

  // Synced lyrics travel with the audio into the download folder. Save mode
  // only — a browser stream stays one file, and the .lrc has its own button.
  const lrcText = save
    ? readTrackLrc(config.data.audioDir, song.audio_url || audioUrl, {
        title: song.title, artist: metadata?.artist,
      }, { enhanced: wantsEnhanced(req.query.enhanced) })?.text
    : undefined;

  // Build download filename: Prepend - Artist - Title_suffix.format
  // Strip leading/trailing underscores and the old backend-injected prefix patterns
  const badPrefixes = /^_?(XL|STD)(\s*\(CPP\))?_?\s*-?\s*/i;
  
  // Strip backend-generated prefixes if they were committed to the DB, and the
  // mastered marker, which is the default render anyway.
  const songTitle = (song.title || 'Untitled')
    .replace(badPrefixes, '')
    .replace(/_mastered/g, '')
    .replace(/[^a-zA-Z0-9 _()-]/g, '')
    .trim();

  const suffix = versionSuffix(version);
  // The caller's artist, else the one the song was made with (a cover's
  // covering band is recorded as artistName).
  const gp = generationParams(song);
  const savedArtist = [gp.artist, gp.artistName, song.artist].find(v => typeof v === 'string' && v.trim()) as string | undefined;
  const resolvedArtist = artistName || (savedArtist || '').replace(badPrefixes, '').replace(/[^a-zA-Z0-9 _()-]/g, '').trim();

  // Strip leading "Artist - " prefix from title if it duplicates the resolved artist.
  // generate.ts stores titles as "Artist - Song Title", so without this the artist
  // would appear twice in the download filename.
  const finalTitle = stripArtistPrefix(songTitle, resolvedArtist);

  // Clean prepend: the user typed this, just trim whitespace
  const cleanPrepend = prepend.trim();
  const parts = [cleanPrepend, resolvedArtist, `${finalTitle}${suffix}`].filter(Boolean);
  const downloadFilename = `${parts.join(' - ')}.${format}`;

  try {
    if (format === 'wav' && sourcePath.endsWith('.wav') && !metadata) {
      // Source is already WAV with no metadata to embed — stream directly
      deliver(res, save, sourcePath, downloadFilename, mimeTypes.wav, undefined, lrcText);
      return;
    }

    // Convert to temp file, then stream
    const tempDir = path.join(config.data.dir, 'download_temp');
    fs.mkdirSync(tempDir, { recursive: true });
    const tempFile = path.join(tempDir, `dl_${Date.now().toString(36)}.${format}`);

    await convertAudio(sourcePath, format, bitrate, tempFile, metadata);

    deliver(res, save, tempFile, downloadFilename, mimeTypes[format] || 'application/octet-stream', tempFile, lrcText);
  } catch (err: any) {
    console.error(`[Download] Conversion failed:`, err.message);
    res.status(500).json({ error: err.message });
  }
};

// ── Synced lyrics export (.lrc) ─────────────────────────────────────────

/** Cleanup of old backend-injected filename prefixes. */
const TRACK_PREFIXES = /^_?(XL|STD)(\s*\(CPP\))?_?\s*-?\s*/i;

/** Song row by id, else by audio_url (Lyric Studio queue items have no row). */
function findSong(id: unknown, audioUrl?: string): any {
  const byId = typeof id === 'string' && id
    ? getDb().prepare('SELECT * FROM songs WHERE id = ?').get(id) as any
    : null;
  if (byId) return byId;
  if (!audioUrl) return null;
  return getDb().prepare('SELECT * FROM songs WHERE audio_url = ?').get(audioUrl) as any ?? null;
}

/** The song row's generation params, or an empty object when it has none. */
function generationParams(song: any): { artist?: unknown; artistName?: unknown } {
  try { return JSON.parse(song?.generation_params || '{}') || {}; } catch { return {}; }
}

/** Drop a leading "Artist - " from a title, which generation may have added. */
function stripArtistPrefix(title: string, artist: string): string {
  if (!artist) return title;
  const prefix = new RegExp(`^${artist.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*-\\s*`, 'i');
  return title.replace(prefix, '').trim() || title;
}

/** `enhanced=1` asks for A2 word stamps (`<mm:ss.xx>`) in the exported LRC. */
function wantsEnhanced(value: unknown): boolean {
  return value === '1' || value === 'true';
}

/** Artist as recorded on the song row, cleaned of old backend prefixes. */
function savedArtistFor(song: any): string {
  const gp = generationParams(song);
  const saved = [gp.artist, gp.artistName, song?.artist]
    .find(v => typeof v === 'string' && v.trim()) as string | undefined;
  return (saved || '').replace(TRACK_PREFIXES, '').replace(/[^a-zA-Z0-9 _()-]/g, '').trim();
}

/**
 * Filename stem for an exported `.lrc`, matching the audio download name
 * (`Prepend - Artist - Title`) so a folder of tracks keeps its pairs.
 */
function lrcNameStem(song: any, audioUrl: string, artistName: string, prepend: string): string {
  const artist = artistName || savedArtistFor(song);
  const rawTitle = song?.title || path.basename(audioUrl || '').replace(/\.[^.]+$/, '') || 'track';
  const title = stripArtistPrefix(
    rawTitle.replace(TRACK_PREFIXES, '').replace(/_mastered/g, '')
      .replace(/[^a-zA-Z0-9 _()-]/g, '').trim() || 'track',
    artist,
  );
  return [prepend, artist, title].filter(Boolean).join(' - ');
}

// GET /api/download/:id/lrc — the track's synced lyrics as an .lrc file.
// POST /api/download/:id/lrc/save — same text, written into DOWNLOAD_DIR.
// `enhanced=1` adds `<mm:ss.xx>` word stamps, rebuilt from `.lyrics.json`.
//
// Reads the `.lrc` beside the render, else rebuilds it from `.lyrics.json`, so
// a retranscribed track exports even when the engine never wrote an LRC.
const lrcHandler = (save: boolean) => async (req: Request, res: Response) => {
  if (save && !config.data.downloadDir) {
    res.status(409).json({ error: 'No download folder is set' });
    return;
  }
  const audioUrlParam = (req.query.audioUrl as string || '').trim();
  const artistName = (req.query.artist as string || '').trim();
  const prepend = (req.query.prepend as string || '').trim();
  // Which render the caller would have downloaded, so the .lrc gets the same
  // " - Unmastered" suffix. External players pair a sidecar by exact filename.
  const version = (req.query.version as string || 'mastered').toLowerCase();

  const song = findSong(req.params.id, audioUrlParam);

  const audioUrl = song?.audio_url || audioUrlParam;
  if (!audioUrl) {
    res.status(404).json({ error: 'Song not found' });
    return;
  }

  const artist = artistName || savedArtistFor(song);
  const lrc = readTrackLrc(
    config.data.audioDir, audioUrl, { title: song?.title, artist },
    { enhanced: wantsEnhanced(req.query.enhanced) },
  );
  if (!lrc) {
    res.status(404).json({ error: 'No synced lyrics for this track — run Retranscribe Lyrics first.' });
    return;
  }

  const filename = `${lrcNameStem(song, audioUrl, artistName, prepend)}${versionSuffix(version)}.lrc`;
  try {
    const tempDir = path.join(config.data.dir, 'download_temp');
    fs.mkdirSync(tempDir, { recursive: true });
    const tempFile = path.join(tempDir, `lrc_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}.lrc`);
    fs.writeFileSync(tempFile, lrc.text, 'utf-8');
    deliver(res, save, tempFile, filename, 'text/plain; charset=utf-8', tempFile);
  } catch (err: any) {
    console.error('[Download] LRC export failed:', err.message);
    res.status(500).json({ error: err.message });
  }
};

router.get('/:id/lrc', lrcHandler(false));
router.post('/:id/lrc/save', lrcHandler(true));
router.get('/:id', downloadHandler(false));
router.post('/:id/save', downloadHandler(true));

export default router;
