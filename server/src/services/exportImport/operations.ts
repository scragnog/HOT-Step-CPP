import fs from 'node:fs';
import path from 'node:path';
import type Database from 'better-sqlite3';
import { config } from '../../config.js';
import { resolveAudioAsset } from '../assets/audioAssets.js';
import { importTrackFile } from '../library/importTrack.js';
import type { ExportRequest, ImportRequest } from './contracts.js';

const CLEAN_PREFIX = /^_?(XL|STD)(\s*\(CPP\))?_?\s*-?\s*/i;
const clean = (text: string) => text.replace(CLEAN_PREFIX, '').replace(/[^a-zA-Z0-9 _()-]/g, '').trim();

interface ResolvedExport {
  index: number;
  songId: string;
  variant?: string;
  filename?: string;
  url?: string;
  error?: string;
}

export function resolveExports(db: Database.Database, userId: string, request: ExportRequest): ResolvedExport[] {
  return request.items.flatMap((item, index): ResolvedExport[] => {
    try {
      const song = db.prepare('SELECT * FROM songs WHERE id = ? AND user_id = ?').get(item.songId, userId) as Record<string, any> | undefined;
      const audioUrl = song?.audio_url || item.audioUrl;
      if (!song && !audioUrl && !item.srcUrl) throw new Error('Song not found');
      const variants = item.variant ? [item.variant === 'mastered' && !song?.mastered_audio_url ? 'original' : item.variant]
        : request.downloadVersion === 'both' && song?.mastered_audio_url ? ['original', 'mastered']
        : [request.downloadVersion === 'mastered' && song?.mastered_audio_url ? 'mastered' : 'original'];
      if (!item.variant && request.includeLatent && song?.latent_url) variants.push('latent');
      return variants.map((variant) => {
        try {
          const source = variant === 'mastered' ? song?.mastered_audio_url
            : variant === 'noadapter' ? song?.noadapter_audio_url || item.srcUrl || audioUrl
            : variant === 'latent' ? song?.latent_url : audioUrl;
          if (!source) throw new Error(`${variant} variant unavailable`);
          const file = path.join(config.data.audioDir, path.basename(source));
          if (!fs.existsSync(file)) throw new Error(`${variant} file missing`);
          const format = variant === 'latent' ? 'latent' : request.format;
          const suffix = variant === 'original' ? ' - Unmastered' : variant === 'noadapter' ? ' - No Adapter' : '';
          const artist = song ? request.artist || clean(song.artist || '') : clean(request.artist || '');
          const title = song ? clean(String(song.title || 'Untitled').replace(/_mastered/g, '')) || 'Untitled' : 'Untitled';
          const finalTitle = artist && title.toLowerCase().startsWith(`${artist.toLowerCase()} - `)
            ? title.slice(artist.length + 3) || title : title;
          const filename = variant === 'latent' ? `${title}.latent`
            : `${[request.prepend?.trim(), artist, `${finalTitle}${suffix}`].filter(Boolean).join(' - ')}.${format}`;
          const query = new URLSearchParams({ format: request.format, version: variant });
          if (request.bitrate !== undefined) query.set('bitrate', String(request.bitrate));
          if (request.artist) query.set('artist', request.artist);
          if (request.prepend) query.set('prepend', request.prepend);
          if (item.audioUrl) query.set('audioUrl', item.audioUrl);
          if (item.srcUrl) query.set('srcUrl', item.srcUrl);
          return { index, songId: item.songId, variant, filename, url: `/api/download/${encodeURIComponent(item.songId)}?${query}` };
        } catch (error) {
          return { index, songId: item.songId, variant, error: (error as Error).message };
        }
      });
    } catch (error) {
      return [{ index, songId: item.songId, error: (error as Error).message }];
    }
  });
}

export async function importAssets(db: Database.Database, userId: string, request: ImportRequest) {
  const results = [];
  for (const [index, item] of request.items.entries()) {
    try {
      const asset = resolveAudioAsset(db, item.assetId, userId, config.data.dir);
      const song = await importTrackFile({ userId, sourcePath: asset.path, originalName: asset.filename, description: item.description });
      results.push({ index, assetId: item.assetId, song: { ...song, tags: JSON.parse(song.tags || '[]'), is_public: !!song.is_public } });
    } catch (error) {
      results.push({ index, assetId: item.assetId, error: (error as Error).message });
    }
  }
  return results;
}
