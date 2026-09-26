// training/datasetScan.ts — disk is the source of truth (D3)
//
// Every dataset read re-scans the source folder and re-parses the sidecars.
// Nothing about a sample is cached in SQLite; the only per-sample state we own
// lives in labels/<sampleId>.json, and even that is a side-car of a side-car.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §4.3, §6.6, §7.4, §7.5

import fs from 'fs';
import path from 'path';
import { config } from '../../config.js';
import {
  AUDIO_EXTENSIONS, EXTENSION_PRIORITY, isInside, sampleIdFor, sidecarPathFor,
  stemPathFor, toRelPath,
} from './paths.js';
import { parseTxtMetadata, readSidecar } from './sidecarIO.js';
import { getTransientStatus, readAllLabels, type SampleLabelRecord } from './labelStore.js';
import * as audioMeta from './audioMeta.js';
import type { SampleLabelStatus, TrainingDatasetRow, TrainingSample } from './types.js';

export interface ScannedFile {
  relPath: string;
  absPath: string;
  sizeBytes: number;
  mtimeMs: number;
}

export interface ScanResult {
  sourceDir: string;
  files: ScannedFile[];
}

/** Thrown when a folder blows past config.training.maxScanFiles (§7.3). */
export class ScanLimitError extends Error {
  constructor(limit: number) {
    super(`Folder contains more than ${limit} audio files`);
    this.name = 'ScanLimitError';
  }
}

// ── Scanning ─────────────────────────────────────────────────────────────

/**
 * Recursive audio scan. Case-insensitive extension match, skips dotted dirs and
 * node_modules, sorted by relPath, hard-capped at config.training.maxScanFiles.
 */
export function scanAudioFiles(sourceDir: string, recursive: boolean): ScannedFile[] {
  const limit = config.training.maxScanFiles;
  const out: ScannedFile[] = [];

  const walk = (dir: string) => {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;   // unreadable sub-folder — skip it, never fail the whole scan
    }
    for (const entry of entries) {
      const abs = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (!recursive) continue;
        if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
        walk(abs);
        continue;
      }
      if (!entry.isFile()) continue;
      if (!AUDIO_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) continue;
      // audioConvert.ts once cached `<track>.engine.wav` beside the source;
      // leftovers are duplicates of a real track, not songs.
      if (entry.name.toLowerCase().endsWith('.engine.wav')) continue;
      let st: fs.Stats;
      try { st = fs.statSync(abs); } catch { continue; }
      out.push({
        relPath: toRelPath(sourceDir, abs),
        absPath: abs,
        sizeBytes: st.size,
        mtimeMs: st.mtimeMs,
      });
      if (out.length > limit) throw new ScanLimitError(limit);
    }
  };

  walk(sourceDir);
  out.sort((a, b) => a.relPath.localeCompare(b.relPath));
  return out;
}

/**
 * §7.5 — `song.wav` and `song.mp3` in one folder both map to `song.txt`.
 * Keep the highest-priority extension, drop the rest, and warn.
 */
export function dedupeBySidecar(files: ScannedFile[], warnings?: string[]): ScannedFile[] {
  const byKey = new Map<string, ScannedFile[]>();
  for (const f of files) {
    const key = sidecarPathFor(f.absPath).toLowerCase();
    const bucket = byKey.get(key);
    if (bucket) bucket.push(f); else byKey.set(key, [f]);
  }

  const kept: ScannedFile[] = [];
  for (const bucket of byKey.values()) {
    if (bucket.length === 1) {
      kept.push(bucket[0]);
      continue;
    }
    const rank = (f: ScannedFile) => {
      const i = EXTENSION_PRIORITY.indexOf(path.extname(f.absPath).toLowerCase());
      return i === -1 ? EXTENSION_PRIORITY.length : i;
    };
    const ordered = [...bucket].sort((a, b) => rank(a) - rank(b) || a.relPath.localeCompare(b.relPath));
    const winner = ordered[0];
    kept.push(winner);
    if (warnings) {
      const dropped = ordered.slice(1).map(f => path.basename(f.absPath));
      warnings.push(
        `${dropped.join(' and ')} share one sidecar with ${path.basename(winner.absPath)} — ` +
        `only ${path.basename(winner.absPath)} will be used`,
      );
    }
  }
  kept.sort((a, b) => a.relPath.localeCompare(b.relPath));
  return kept;
}

// ── Sidecar convention auto-detect (§6.6) ────────────────────────────────

function readTextFile(p: string): string {
  try {
    let raw = fs.readFileSync(p, 'utf-8');
    if (raw.charCodeAt(0) === 0xfeff) raw = raw.slice(1);
    return raw.trim();
  } catch {
    return '';
  }
}

/** Side-Step's 3-convention auto-detect — first usable match wins (§6.6). */
export function loadSidecarMetadata(audioPath: string): Record<string, string> {
  const stem = stemPathFor(audioPath);
  const optionA = sidecarPathFor(audioPath);

  // 1. Option A — <stem>.txt parsed as key: value
  let parsedA: Record<string, string> = {};
  const hasOptionA = fs.existsSync(optionA);
  if (hasOptionA) {
    parsedA = parseTxtMetadata(readTextFile(optionA));
    if (parsedA.caption) return parsedA;
  }

  // 2. Split files — <stem>.caption.txt + <stem>.lyrics.txt (string concat, not
  //    extension replacement: filenames containing dots must still resolve).
  const captionPath = `${stem}.caption.txt`;
  const lyricsPath = `${stem}.lyrics.txt`;
  const hasCaptionFile = fs.existsSync(captionPath);
  if (hasCaptionFile) {
    const caption = readTextFile(captionPath);
    if (caption) {
      const lyrics = fs.existsSync(lyricsPath) ? readTextFile(lyricsPath) : '';
      return lyrics ? { caption, lyrics } : { caption };
    }
  }

  // 3. Lyrics-only fallback — a <stem>.txt with no caption and no .caption.txt.
  //
  //    ONLY when the Option-A parse produced no structured key at all. Every
  //    sidecar this server writes carries the six FIELD_ORDER keys (often
  //    empty, e.g. after a quick label that found BPM but no caption), so
  //    flattening those into `lyrics` would paste the whole file into the
  //    lyrics block and blank bpm/key/signature on the very next save.
  if (hasOptionA && !hasCaptionFile) {
    if (Object.keys(parsedA).length > 0) return parsedA;
    const raw = readTextFile(optionA);
    if (raw) return { lyrics: raw };
  }

  return {};
}

// ── Sample assembly ──────────────────────────────────────────────────────

function parseBpm(raw: string | undefined): number | null {
  if (!raw) return null;
  const n = parseFloat(raw);
  return Number.isFinite(n) ? Math.trunc(n) : null;
}

function parseBool(raw: string | undefined): boolean {
  return ['true', '1', 'yes'].includes(String(raw ?? '').trim().toLowerCase());
}

function parseRepeat(raw: string | undefined): number {
  const n = parseInt(String(raw ?? ''), 10);
  return Number.isFinite(n) && n >= 1 ? n : 1;
}

/**
 * 'pending'/'processing' come from the in-memory job registry, never from disk
 * (§2.0 defines both as live-job state). 'error' and the caption test are the
 * only persistent inputs.
 */
function resolveStatus(
  hasCaption: boolean,
  label: SampleLabelRecord | undefined,
  transient: SampleLabelStatus | null,
): SampleLabelStatus {
  if (transient) return transient;
  if (hasCaption) return 'labeled';
  if (label?.labelStatus === 'error') return 'error';
  return 'unlabeled';
}

/** Build the API-shaped sample from one scanned file + its sidecar + its label. */
export function sampleFromParts(
  datasetId: string,
  file: ScannedFile,
  meta: Record<string, string>,
  label: SampleLabelRecord | undefined,
  duration: number,
  fileMissing: boolean,
): TrainingSample {
  const sidecarPath = sidecarPathFor(file.absPath);
  const caption = meta.caption ?? '';
  const sampleId = sampleIdFor(file.relPath);
  return {
    sampleId,
    relPath: file.relPath,
    audioPath: file.absPath,
    filename: path.basename(file.absPath),
    sidecarPath,
    sidecarExists: (() => { try { return fs.existsSync(sidecarPath); } catch { return false; } })(),
    fileMissing,

    caption,
    genre: meta.genre ?? '',
    bpm: parseBpm(meta.bpm),
    key: meta.key ?? '',
    signature: meta.signature ?? '',
    language: meta.language ?? '',
    isInstrumental: parseBool(meta.is_instrumental),
    lyrics: meta.lyrics ?? '',
    customTag: (meta.custom_tag ?? '').trim(),
    repeat: parseRepeat(meta.repeat),
    promptOverride: meta.prompt_override ? meta.prompt_override : null,

    excluded: label?.excluded ?? false,
    labelStatus: resolveStatus(!!caption.trim(), label, getTransientStatus(datasetId, sampleId)),
    error: label?.error ?? null,
    hasAudioCodes: !!label?.audioCodes,
    sources: (label?.sources ?? {}) as TrainingSample['sources'],

    duration,
    sizeBytes: file.sizeBytes,
    tagArtist: label?.tags?.artist ?? '',
    tagTitle: label?.tags?.title ?? '',
    tagAlbum: label?.tags?.album ?? '',
    labeledAt: label?.labeledAt ?? '',
  };
}

/**
 * Join scan + sidecar + labelStore (+ optional measured duration).
 *
 * `withDuration` defaults FALSE: `GET /datasets/:id` must stay fast on a
 * 500-file folder, so duration comes from the size+mtime-keyed cache in
 * labels/<id>.json. Only the build job pays to parse every file (§4.3).
 *
 * `opts.warnings` is an optional sink for dataset-level scan warnings — the
 * contract's `buildSamples` returns only samples, so warnings ride out here.
 */
export async function buildSamples(
  ds: TrainingDatasetRow,
  opts?: { withDuration?: boolean; warnings?: string[] },
): Promise<TrainingSample[]> {
  const withDuration = opts?.withDuration === true;
  const warnings = opts?.warnings;

  const scanned = dedupeBySidecar(scanAudioFiles(ds.sourceDir, ds.recursive), warnings);
  const labels = readAllLabels(ds.slug);
  const seen = new Set<string>();
  const samples: TrainingSample[] = [];

  for (const file of scanned) {
    const sampleId = sampleIdFor(file.relPath);
    seen.add(sampleId);
    const label = labels.get(sampleId);
    const meta = loadSidecarMetadata(file.absPath);

    let duration = 0;
    const cache = label?.durationCache;
    if (cache && cache.sizeBytes === file.sizeBytes && Math.abs(cache.mtimeMs - file.mtimeMs) < 1) {
      duration = cache.seconds;
    } else if (withDuration) {
      duration = await audioMeta.readDuration(file.absPath);
    }

    samples.push(sampleFromParts(ds.id, file, meta, label, duration, false));
  }

  // §7.4 — a label record whose audio has vanished still gets a row, so the
  // user's edits are not silently lost. Files with no label simply disappear.
  for (const [sampleId, label] of labels) {
    if (seen.has(sampleId) || !label.relPath) continue;
    const absPath = path.join(ds.sourceDir, ...label.relPath.split('/'));
    // §7.8 — this is the one path we reconstruct rather than scan, so it goes
    // through the traversal guard before anything reads it.
    if (!isInside(ds.sourceDir, absPath)) continue;
    const ghost: ScannedFile = { relPath: label.relPath, absPath, sizeBytes: 0, mtimeMs: 0 };
    const meta = loadSidecarMetadata(absPath);
    samples.push(sampleFromParts(ds.id, ghost, meta, label, label.durationCache?.seconds ?? 0, true));
  }

  samples.sort((a, b) => a.relPath.localeCompare(b.relPath));
  return samples;
}

// ── Scan preview (§2.2) ──────────────────────────────────────────────────

export interface ScanPreviewResult {
  root: string;
  audioFiles: number;
  withSidecar: number;
  withCaption: number;
  hasDatasetJson: boolean;
  extensions: Record<string, number>;
  sampleNames: string[];
}

export function scanPreview(root: string, recursive: boolean): ScanPreviewResult {
  const files = dedupeBySidecar(scanAudioFiles(root, recursive));
  const extensions: Record<string, number> = {};
  let withSidecar = 0;
  let withCaption = 0;

  for (const f of files) {
    const ext = path.extname(f.absPath).toLowerCase();
    extensions[ext] = (extensions[ext] || 0) + 1;
    const sidecarPath = sidecarPathFor(f.absPath);
    if (fs.existsSync(sidecarPath)) {
      withSidecar++;
      if (readSidecar(sidecarPath).caption) withCaption++;
    }
  }

  return {
    root,
    audioFiles: files.length,
    withSidecar,
    withCaption,
    hasDatasetJson: fs.existsSync(path.join(root, 'dataset.json')),
    extensions,
    sampleNames: files.slice(0, 10).map(f => path.basename(f.absPath)),
  };
}
