// exportService.ts — Export lyric generations to JSON + TXT files
//
// Port of Python export_service.py

import fs from 'fs';
import path from 'path';
import { config } from '../../config.js';

export interface ExportData {
  title: string;
  lyrics: string;
  artistName: string;
  albumName?: string;
  provider: string;
  model: string;
  bpm?: number;
  key?: string;
  caption?: string;
  /** MiniMax-Music3 Structured Caption — the second, separately-formatted
   *  caption. Absent on generations written before the field existed. */
  captionMm3?: string;
  captionYue2?: string;
  duration?: number;
  subject?: string;
  extraInstructions?: string;
  createdAt?: string;
}

/**
 * Export a generation to both JSON and TXT files.
 * Returns the paths of the exported files.
 */
export function exportGeneration(data: ExportData): { jsonPath: string; txtPath: string } {
  const exportDir = config.lireek.exportDir;
  fs.mkdirSync(exportDir, { recursive: true });

  // Build safe filename: "Artist - Title" or "Artist - Album - Title"
  const safeName = (s: string) => s.replace(/[<>:"/\\|?*]/g, '_').trim();
  const parts = [safeName(data.artistName)];
  if (data.albumName) parts.push(safeName(data.albumName));
  parts.push(safeName(data.title || 'Untitled'));
  const baseName = parts.join(' - ');

  // Deduplicate: if file exists, append (2), (3), etc.
  let finalBase = baseName;
  let counter = 1;
  while (fs.existsSync(path.join(exportDir, `${finalBase}.json`))) {
    counter++;
    finalBase = `${baseName} (${counter})`;
  }

  const jsonPath = path.join(exportDir, `${finalBase}.json`);
  const txtPath = path.join(exportDir, `${finalBase}.txt`);

  // JSON export (full metadata)
  const jsonData = {
    title: data.title,
    artist: data.artistName,
    album: data.albumName ?? null,
    lyrics: data.lyrics,
    provider: data.provider,
    model: data.model,
    bpm: data.bpm ?? null,
    key: data.key ?? null,
    caption: data.caption ?? null,
    caption_mm3: data.captionMm3 ?? null,
    caption_yue2: data.captionYue2 ?? null,
    duration: data.duration ?? null,
    subject: data.subject ?? null,
    extra_instructions: data.extraInstructions ?? null,
    exported_at: new Date().toISOString(),
    created_at: data.createdAt ?? null,
  };
  fs.writeFileSync(jsonPath, JSON.stringify(jsonData, null, 2), 'utf-8');

  // TXT export (human-readable)
  const txtLines = [
    `Title: ${data.title || 'Untitled'}`,
    `Artist: ${data.artistName}`,
  ];
  if (data.albumName) txtLines.push(`Album Style: ${data.albumName}`);
  if (data.bpm) txtLines.push(`BPM: ${data.bpm}`);
  if (data.key) txtLines.push(`Key: ${data.key}`);
  if (data.caption) txtLines.push(`Caption: ${data.caption}`);
  if (data.duration) txtLines.push(`Duration: ${data.duration}s`);
  // Multi-line and structured, so it goes in its own block rather than inline
  // with the single-line metadata above.
  if (data.captionMm3) txtLines.push('', '--- MM3 Caption ---', '', data.captionMm3);
  if (data.captionYue2) txtLines.push('', '--- YuE2 Caption ---', '', data.captionYue2);
  txtLines.push('', '---', '', data.lyrics);
  fs.writeFileSync(txtPath, txtLines.join('\n'), 'utf-8');

  console.log(`[Export] Saved ${jsonPath}`);
  return { jsonPath, txtPath };
}
