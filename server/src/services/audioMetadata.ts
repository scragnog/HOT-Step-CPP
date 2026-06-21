// audioMetadata.ts — Gather song metadata and build ffmpeg args for embedding
//
// Reads metadata from the songs DB row, enriches with Lyric Studio artist/album
// data and Cover Studio source/target artist formatting, then produces ffmpeg
// CLI arguments for -metadata tags and cover art attachment.
//
// Format support:
//   FLAC  — Vorbis comments + PICTURE block (full metadata + cover art)
//   MP3   — ID3v2 tags + APIC frame (full metadata + cover art)
//   Opus  — Vorbis comments (full text metadata, no cover art — OGG limitation)
//   WAV   — INFO chunk only (title, artist, comment — no cover art)

import fs from 'fs';
import path from 'path';
import { getDb } from '../db/database.js';
import { config } from '../config.js';

// ── Types ───────────────────────────────────────────────────────────────

export interface AudioMetadata {
  title?: string;
  artist?: string;
  albumArtist?: string;   // original artist for covers (ALBUMARTIST tag)
  album?: string;
  genre?: string;          // style/caption
  lyrics?: string;
  bpm?: number;
  key?: string;            // key_scale
  date?: string;           // year from created_at
  comment?: string;        // generator attribution + model info
  coverArtPath?: string;   // absolute path to cover image on disk
}

// ── Metadata gathering ──────────────────────────────────────────────────

/**
 * Gather all available metadata from a song DB row.
 *
 * Works uniformly across all sources (Auto-Gen, Custom-Gen, Lyric Studio,
 * Cover Studio) because they all write to the same songs table. The
 * differences are handled by checking generation_params.source and
 * enriching where possible.
 */
export function gatherSongMetadata(song: any): AudioMetadata {
  const meta: AudioMetadata = {};

  // Parse generation_params (stored as JSON string in DB)
  let genParams: Record<string, any> = {};
  try {
    genParams = typeof song.generation_params === 'string'
      ? JSON.parse(song.generation_params || '{}')
      : (song.generation_params || {});
  } catch { /* malformed JSON — proceed with empty */ }

  const source = genParams.source || 'create';

  // ── Title ──
  // Strip "Artist - " prefix from DB title (generate.ts stores as "Artist - Song Title")
  let rawTitle = song.title || 'Untitled';
  const artist = genParams.artist || genParams.artistName || '';
  if (artist) {
    const prefix = new RegExp(`^${escapeRegex(artist)}\\s*-\\s*`, 'i');
    rawTitle = rawTitle.replace(prefix, '').trim() || rawTitle;
  }
  // For cover studio, the title may already be "Song (Artist Cover)" — extract clean title
  if (source === 'cover-studio') {
    const coverSuffix = /\s*\(.*?\bCover\b\)\s*$/i;
    rawTitle = rawTitle.replace(coverSuffix, '').trim() || rawTitle;
  }
  meta.title = rawTitle;

  // ── Artist ──
  // For Lyric Studio and Cover Studio, append "(AI-Generated)" to be
  // transparent that this is AI-generated content in the style of the artist.
  if (source === 'cover-studio') {
    // Cover Studio: format as "TargetArtist (SourceArtist Cover) (AI-Generated)"
    const targetArtist = genParams.artistName || '';
    const sourceArtist = genParams.sourceArtist || '';
    if (targetArtist && sourceArtist) {
      meta.artist = `${targetArtist} (${sourceArtist} Cover) (AI-Generated)`;
      meta.albumArtist = sourceArtist;
    } else if (targetArtist) {
      meta.artist = `${targetArtist} (AI-Generated)`;
    } else if (sourceArtist) {
      meta.artist = `${sourceArtist} (AI-Generated)`;
    }
  } else if (source === 'lyric-studio') {
    // Lyric Studio: enrich from the lireek join chain
    const lireekMeta = enrichFromLireek(song.audio_url);
    if (lireekMeta) {
      const lireekArtist = lireekMeta.artistName || artist;
      meta.artist = lireekArtist ? `${lireekArtist} (AI-Generated)` : undefined;
      meta.album = lireekMeta.album || undefined;
    } else {
      meta.artist = artist ? `${artist} (AI-Generated)` : undefined;
    }
  } else {
    // Auto-Gen / Custom-Gen
    meta.artist = artist || undefined;
  }

  // ── Genre (music style from caption, NOT from subject) ──
  // song.style stores the subject (what the song is about), NOT the genre.
  // song.caption stores the LM's music style description (e.g. "metalcore,
  // aggressive, heavy guitars") which is what genre should be based on.
  meta.genre = song.caption || genParams.caption || genParams.style || undefined;

  // ── Lyrics ──
  if (song.lyrics && song.lyrics !== '[Instrumental]') {
    meta.lyrics = song.lyrics;
  }

  // ── BPM ──
  if (song.bpm && song.bpm > 0) {
    meta.bpm = song.bpm;
  }

  // ── Key ──
  if (song.key_scale) {
    meta.key = song.key_scale;
  }

  // ── Date (year) ──
  if (song.created_at) {
    try {
      const year = new Date(song.created_at).getFullYear();
      if (year > 2000) meta.date = String(year);
    } catch { /* invalid date */ }
  }

  // ── Comment (attribution) ──
  const modelName = song.dit_model || genParams.ditModel || '';
  const parts = ['Generated by HOT-Step'];
  if (modelName) parts.push(`Model: ${modelName}`);
  if (source === 'cover-studio') parts.push('(AI Cover)');
  meta.comment = parts.join(' | ');

  // ── Cover art ──
  if (song.cover_url) {
    const coverFilename = path.basename(song.cover_url);
    const coverPath = path.join(config.data.audioDir, coverFilename);
    if (fs.existsSync(coverPath)) {
      meta.coverArtPath = coverPath;
    }
  }

  // ── User overrides (metadata editor, #60) ──
  // When the user has edited a tag-only field, embed it VERBATIM — overriding
  // the auto-derivation above (e.g. the "(AI-Generated)" artist suffix).
  // Columns (title/genre/bpm/key/lyrics/cover) are edited directly and already
  // read above, so only the columnless fields live here.
  if (song.metadata_overrides) {
    try {
      const ov = typeof song.metadata_overrides === 'string'
        ? JSON.parse(song.metadata_overrides || '{}')
        : (song.metadata_overrides || {});
      if (typeof ov.artist === 'string' && ov.artist.trim() !== '') meta.artist = ov.artist;
      if (typeof ov.album === 'string' && ov.album.trim() !== '') meta.album = ov.album;
      if (typeof ov.year === 'string' && ov.year.trim() !== '') meta.date = ov.year;
      if (typeof ov.comment === 'string' && ov.comment.trim() !== '') meta.comment = ov.comment;
    } catch { /* malformed overrides — ignore */ }
  }

  return meta;
}

// ── ffmpeg argument builders ────────────────────────────────────────────

/**
 * Build ffmpeg -metadata arguments for text tags.
 *
 * Returns an array of strings to splice into the ffmpeg command.
 * WAV format gets a reduced set (INFO chunk supports fewer fields).
 */
export function buildMetadataArgs(meta: AudioMetadata, format: string): string[] {
  const args: string[] = [];
  const isWav = format === 'wav';

  const add = (key: string, value: string | number | undefined) => {
    if (value === undefined || value === null || value === '') return;
    args.push('-metadata', `${key}=${String(value)}`);
  };

  // Tags supported by all formats (including WAV INFO chunks)
  add('title', meta.title);
  add('artist', meta.artist);
  add('comment', meta.comment);

  // Tags NOT supported by WAV INFO chunks
  if (!isWav) {
    add('album', meta.album);
    add('album_artist', meta.albumArtist);
    add('genre', meta.genre);
    add('date', meta.date);

    // BPM and Key — use standardized tag names per format so media players
    // actually populate their "Beats-per-minute" and "Initial key" fields.
    // MP3 (ID3v2): TBPM and TKEY are the standard frame names.
    // FLAC/Opus (Vorbis): BPM is standard, INITIALKEY is more widely
    // recognized than KEY by media players and DJ software.
    if (meta.bpm) {
      if (format === 'mp3') {
        add('TBPM', meta.bpm);
      } else {
        add('BPM', meta.bpm);
      }
    }
    if (meta.key) {
      if (format === 'mp3') {
        add('TKEY', meta.key);
      } else {
        add('INITIALKEY', meta.key);
      }
    }

    // Lyrics — use UNSYNCEDLYRICS for better player compatibility.
    // Most players (foobar2000, MusicBee, VLC, Strawberry) look for this
    // tag name in Vorbis comments. For MP3, ffmpeg writes it as a custom
    // text frame (TXXX); full USLT support requires a dedicated ID3 library.
    if (meta.lyrics) {
      // Truncate very long lyrics to avoid bloating (16KB limit is generous)
      const truncated = meta.lyrics.length > 16_000
        ? meta.lyrics.substring(0, 16_000) + '\n[truncated]'
        : meta.lyrics;
      add('UNSYNCEDLYRICS', truncated);
    }
  }

  return args;
}

/**
 * Build ffmpeg arguments for embedding cover art.
 *
 * Returns separate input args (to prepend) and output args (to append).
 *
 * Cover art is transcoded from PNG to JPEG and downscaled to 1024×1024
 * on-the-fly, keeping the embedded image ~150-250KB instead of 2-5MB PNG.
 *
 * Supported: FLAC (PICTURE block), MP3 (ID3v2 APIC frame)
 * NOT supported: Opus (OGG container can't hold video streams — would
 *   need METADATA_BLOCK_PICTURE in Vorbis comments which requires base64
 *   encoding), WAV (no standard mechanism)
 */
export function buildCoverArtArgs(
  coverPath: string,
  format: string,
): { inputArgs: string[]; outputArgs: string[] } {
  // WAV and Opus: no cover art via the video-stream approach
  // WAV has no standard, Opus/OGG can't hold video streams
  if (format === 'wav' || format === 'opus') {
    return { inputArgs: [], outputArgs: [] };
  }

  // Verify cover file exists
  if (!fs.existsSync(coverPath)) {
    return { inputArgs: [], outputArgs: [] };
  }

  // Input: add cover image as second input
  const inputArgs = ['-i', coverPath];

  // Output: map audio from input 0, video (cover) from input 1
  // Transcode to JPEG, scale to 1024×1024, quality 5 (~85%)
  const outputArgs: string[] = [];

  if (format === 'mp3') {
    // MP3/ID3v2: APIC frame with JPEG transcode
    outputArgs.push(
      '-map', '0:a',
      '-map', '1:v',
      '-c:v', 'mjpeg',
      '-vf', 'scale=1024:1024',
      '-q:v', '5',
      '-metadata:s:v', 'title=Cover',
      '-metadata:s:v', 'comment=Cover (front)',
      '-disposition:v', 'attached_pic',
      '-id3v2_version', '3',
    );
  } else {
    // FLAC: PICTURE block with JPEG transcode
    outputArgs.push(
      '-map', '0:a',
      '-map', '1:v',
      '-c:v', 'mjpeg',
      '-vf', 'scale=1024:1024',
      '-q:v', '5',
      '-metadata:s:v', 'title=Cover',
      '-metadata:s:v', 'comment=Cover (front)',
      '-disposition:v', 'attached_pic',
    );
  }

  return { inputArgs, outputArgs };
}

// ── Helpers ─────────────────────────────────────────────────────────────

/** Escape special regex characters in a string */
function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Enrich metadata from the Lyric Studio join chain.
 * Returns artist name and album if the song was generated via Lyric Studio.
 */
function enrichFromLireek(audioUrl: string): { artistName?: string; album?: string } | null {
  if (!audioUrl) return null;

  try {
    const row = getDb().prepare(
      `SELECT a.name AS artist_name, ls.album
       FROM audio_generations ag
       JOIN generations g ON g.id = ag.generation_id
       JOIN profiles p ON p.id = g.profile_id
       JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
       JOIN artists a ON a.id = ls.artist_id
       WHERE ag.audio_url = ?`
    ).get(audioUrl) as any;

    if (row) {
      return {
        artistName: row.artist_name || undefined,
        album: row.album || undefined,
      };
    }
  } catch {
    // Lireek tables may not exist — graceful fallback
  }

  return null;
}
