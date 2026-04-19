// download.ts — Audio download route with format conversion
//
// GET /api/songs/:id/download?format=wav|flac|opus|mp3&bitrate=192&version=original|mastered
//
// Converts the source WAV to the requested format and streams it back
// with a Content-Disposition header for browser download.

import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { config } from '../config.js';
import { getDb } from '../db/database.js';

const execFileAsync = promisify(execFile);
const router = Router();

/** Get mp3-codec.exe path */
function getMp3CodecPath(): string {
  const aceExe = config.aceServer.exe;
  if (aceExe) return path.join(path.dirname(aceExe), 'mp3-codec.exe');
  return path.resolve(process.cwd(), '..', 'engine', 'build', 'Release', 'mp3-codec.exe');
}

/** Convert WAV to target format, return temp file path */
async function convertAudio(
  sourcePath: string,
  format: string,
  bitrate: number,
  outputPath: string,
): Promise<void> {
  if (format === 'wav') {
    // No conversion needed — copy source
    fs.copyFileSync(sourcePath, outputPath);
    return;
  }

  if (format === 'mp3') {
    // Use mp3-codec.exe for MP3 (no ffmpeg dependency)
    const codec = getMp3CodecPath();
    if (fs.existsSync(codec)) {
      await execFileAsync(codec, [
        '-i', sourcePath, '-o', outputPath, '-b', String(bitrate),
      ], { timeout: 120_000 });
      return;
    }
    // Fallback to ffmpeg
  }

  // Use ffmpeg for FLAC, Opus, or MP3 fallback
  const args = ['-y', '-i', sourcePath];

  switch (format) {
    case 'flac':
      args.push('-c:a', 'flac', '-compression_level', '8');
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

  args.push(outputPath);

  try {
    await execFileAsync('ffmpeg', args, { timeout: 120_000 });
  } catch (err: any) {
    throw new Error(`ffmpeg conversion failed: ${err.message}. Is ffmpeg installed?`);
  }
}

/** MIME types for audio formats */
const mimeTypes: Record<string, string> = {
  wav: 'audio/wav',
  mp3: 'audio/mpeg',
  flac: 'audio/flac',
  opus: 'audio/ogg',
};

// GET /api/download/:id?format=wav&bitrate=192&version=original&artist=Name&prepend=Prefix
router.get('/:id', async (req, res) => {
  const { id } = req.params;
  const format = (req.query.format as string || 'wav').toLowerCase();
  const bitrate = parseInt(req.query.bitrate as string) || 192;
  const version = (req.query.version as string || 'original').toLowerCase();
  const artistName = (req.query.artist as string || '').trim();
  const prepend = (req.query.prepend as string || '').trim();

  // Validate format
  if (!['wav', 'mp3', 'flac', 'opus'].includes(format)) {
    res.status(400).json({ error: `Invalid format: ${format}. Use wav, mp3, flac, or opus.` });
    return;
  }

  // Get song from DB — try by ID first, then by audio_url
  let song = getDb().prepare('SELECT * FROM songs WHERE id = ?').get(id) as any;
  if (!song) {
    // Fallback: try looking up by audio_url (for Lyric Studio queue items)
    const audioUrlParam = req.query.audioUrl as string;
    if (audioUrlParam) {
      song = getDb().prepare('SELECT * FROM songs WHERE audio_url = ?').get(audioUrlParam) as any;
    }
  }
  if (!song) {
    // Last resort: serve the audio file directly without DB metadata
    const audioUrlParam = req.query.audioUrl as string;
    if (audioUrlParam) {
      const filename = path.basename(audioUrlParam);
      const sourcePath = path.join(config.data.audioDir, filename);
      if (fs.existsSync(sourcePath)) {
        // Clean up parsed parameters just in case DB doesn't have standard naming
        const badPrefixes = /^_(XL|STD)(\s*\(CPP\))?(\s*-\s*_)?\s*-?\s*/i;
        const cleanPrepend = prepend.replace(badPrefixes, '').trim();
        const cleanArtist = artistName.replace(badPrefixes, '').trim();
        const titleSuffix = version === 'original' ? ' - Unmastered' : '';
        const titleParts = [cleanPrepend, cleanArtist, 'Untitled'].filter(Boolean);
        const downloadFilename = `${titleParts.join(' - ')}${titleSuffix}.${format}`;
        if (format === 'wav' && sourcePath.endsWith('.wav')) {
          res.setHeader('Content-Type', mimeTypes.wav);
          res.setHeader('Content-Disposition', `attachment; filename="${downloadFilename}"`);
          res.setHeader('Content-Length', fs.statSync(sourcePath).size);
          fs.createReadStream(sourcePath).pipe(res);
          return;
        }
        // Convert
        const tempDir = path.join(config.data.dir, 'download_temp');
        fs.mkdirSync(tempDir, { recursive: true });
        const tempFile = path.join(tempDir, `dl_${Date.now().toString(36)}.${format}`);
        await convertAudio(sourcePath, format, bitrate, tempFile);
        const stat = fs.statSync(tempFile);
        res.setHeader('Content-Type', mimeTypes[format] || 'application/octet-stream');
        res.setHeader('Content-Disposition', `attachment; filename="${downloadFilename}"`);
        res.setHeader('Content-Length', stat.size);
        const stream = fs.createReadStream(tempFile);
        stream.pipe(res);
        stream.on('end', () => { try { fs.unlinkSync(tempFile); } catch {} });
        return;
      }
    }
    res.status(404).json({ error: 'Song not found' });
    return;
  }

  // Determine which audio URL to use
  let audioUrl: string;
  if (version === 'mastered' && song.mastered_audio_url) {
    audioUrl = song.mastered_audio_url;
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

  // Build download filename: Prepend - Artist - Title_suffix.format
  const badPrefixes = /^_(XL|STD)(\s*\(CPP\))?(\s*-\s*_)?\s*-?\s*/i;
  
  let rawTitle = song.title || 'Untitled';
  // Strip backend-generated prefix strings if they accidentally got committed to the DB
  rawTitle = rawTitle.replace(badPrefixes, '');
  rawTitle = rawTitle.replace(/_mastered/g, ''); // User wants mastered as default, so explicitly strip it out just in case
  const songTitle = rawTitle.replace(/[^a-zA-Z0-9 _-]/g, '');

  const suffix = version === 'original' ? ' - Unmastered' : '';
  const resolvedArtist = artistName || (song.artist || '').replace(badPrefixes, '').replace(/[^a-zA-Z0-9 _-]/g, '');
  const cleanPrepend = prepend.replace(badPrefixes, '').trim();
  const parts = [cleanPrepend, resolvedArtist, `${songTitle}${suffix}`].filter(Boolean);
  const downloadFilename = `${parts.join(' - ')}.${format}`;

  try {
    if (format === 'wav' && sourcePath.endsWith('.wav')) {
      // Source is already WAV — stream directly, no temp file
      res.setHeader('Content-Type', mimeTypes.wav);
      res.setHeader('Content-Disposition', `attachment; filename="${downloadFilename}"`);
      res.setHeader('Content-Length', fs.statSync(sourcePath).size);
      fs.createReadStream(sourcePath).pipe(res);
      return;
    }

    // Convert to temp file, then stream
    const tempDir = path.join(config.data.dir, 'download_temp');
    fs.mkdirSync(tempDir, { recursive: true });
    const tempFile = path.join(tempDir, `dl_${Date.now().toString(36)}.${format}`);

    await convertAudio(sourcePath, format, bitrate, tempFile);

    const stat = fs.statSync(tempFile);
    res.setHeader('Content-Type', mimeTypes[format] || 'application/octet-stream');
    res.setHeader('Content-Disposition', `attachment; filename="${downloadFilename}"`);
    res.setHeader('Content-Length', stat.size);

    const stream = fs.createReadStream(tempFile);
    stream.pipe(res);

    // Clean up temp file after stream completes
    stream.on('end', () => {
      try { fs.unlinkSync(tempFile); } catch {}
      try { fs.rmdirSync(tempDir); } catch {}
    });
    stream.on('error', () => {
      try { fs.unlinkSync(tempFile); } catch {}
    });
  } catch (err: any) {
    console.error(`[Download] Conversion failed:`, err.message);
    res.status(500).json({ error: err.message });
  }
});

export default router;
