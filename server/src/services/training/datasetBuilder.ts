// training/datasetBuilder.ts — writes dataset.json where Side-Step writes it
//
// The acceptance criterion for this file is ROUND-TRIP PRESERVATION: an existing
// dataset.json may carry enrichment keys we know nothing about (lyrics_source,
// mbid, data_source, artist, title, album, …). Every one of them survives a
// rebuild. Never strip an unknown field.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §4.11, §6.8

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { datasetSampleId } from './paths.js';
import { loadSidecarMetadata } from './datasetScan.js';
import { normalizeLanguage } from '../languageCodes.js';
import * as audioMeta from './audioMeta.js';
import type { TrainingDatasetRow, TrainingSample } from './types.js';
import { normalizeKeyscale, normalizeTimeSignature } from './metadataVocab.js';

/** Keys this builder owns and therefore rewrites on every build. */
const OWNED_SAMPLE_KEYS = [
  'id', 'audio_path', 'filename', 'caption', 'genre', 'lyrics', 'raw_lyrics',
  'formatted_lyrics', 'bpm', 'keyscale', 'timesignature', 'duration', 'language',
  'is_instrumental', 'custom_tag', 'labeled', 'prompt_override', 'repeat',
] as const;

const OWNED_METADATA_KEYS = [
  'name', 'custom_tag', 'tag_position', 'created_at', 'num_samples',
  'all_instrumental', 'genre_ratio', 'default_artist', 'default_album', 'default_genre',
  'default_language',
] as const;

interface PriorDataset {
  metadata: Record<string, unknown>;
  samplesByPath: Map<string, Record<string, unknown>>;
}

function readPrior(outputPath: string): PriorDataset {
  const empty: PriorDataset = { metadata: {}, samplesByPath: new Map() };
  try {
    if (!fs.existsSync(outputPath)) return empty;
    const parsed = JSON.parse(fs.readFileSync(outputPath, 'utf-8')) as Record<string, unknown>;
    if (!parsed || typeof parsed !== 'object') return empty;

    const metadata = (parsed.metadata && typeof parsed.metadata === 'object')
      ? parsed.metadata as Record<string, unknown>
      : {};
    const samplesByPath = new Map<string, Record<string, unknown>>();
    const samples = Array.isArray(parsed.samples) ? parsed.samples : [];
    for (const s of samples) {
      if (!s || typeof s !== 'object') continue;
      const row = s as Record<string, unknown>;
      const key = typeof row.audio_path === 'string' ? row.audio_path.toLowerCase() : '';
      if (key) samplesByPath.set(key, row);
    }
    return { metadata, samplesByPath };
  } catch (err: any) {
    console.warn(`[Training] Could not read existing dataset.json (${err.message}) — building fresh`);
    return empty;
  }
}

/**
 * `metadata` of an existing `<dir>/dataset.json`, or `{}`.
 *
 * `POST /datasets` seeds the new row from this so a later rebuild rewrites the
 * user's own `custom_tag` / `genre_ratio` / `default_artist` / `default_album`
 * with the same values instead of blanking them (§8.5 acceptance criterion).
 */
export function readDatasetJsonMetadata(dir: string): Record<string, unknown> {
  return readPrior(path.join(dir, 'dataset.json')).metadata;
}

function parseBpmValue(raw: string | undefined): number | null {
  if (!raw) return null;
  const n = parseFloat(raw);
  return Number.isFinite(n) ? Math.trunc(n) : null;
}

/**
 * Build dataset.json for every non-excluded sample.
 * Durations are measured here (music-metadata), never guessed.
 */
export async function buildDataset(
  ds: TrainingDatasetRow,
  samples: TrainingSample[],
  outputPath: string,
  onProgress?: (done: number, total: number) => void,
): Promise<{ path: string; total: number; withMetadata: number }> {
  const included = samples.filter(s => !s.excluded);
  const prior = readPrior(outputPath);

  const rows: Array<Record<string, unknown>> = [];
  let withMetadata = 0;
  let allInstrumental = true;

  for (let i = 0; i < included.length; i++) {
    const sample = included[i];
    const meta = loadSidecarMetadata(sample.audioPath);

    const stem = path.basename(sample.filename, path.extname(sample.filename));
    const caption = meta.caption || stem.replace(/[_-]/g, ' ');

    // Order matters: is_instrumental is decided from the ORIGINAL lyrics,
    // before the '[Instrumental]' substitution, and raw_lyrics keeps the original.
    const rawLyrics = meta.lyrics ?? '';
    const isInstrumental = ('is_instrumental' in meta)
      ? ['true', '1', 'yes'].includes(String(meta.is_instrumental).toLowerCase())
      : (!rawLyrics.trim() || rawLyrics.includes('[Instrumental]'));
    const lyrics = rawLyrics.trim() ? rawLyrics : '[Instrumental]';

    const duration = sample.duration > 0
      ? sample.duration
      : await audioMeta.readDuration(sample.audioPath);

    if (meta.caption) withMetadata++;
    if (!isInstrumental) allInstrumental = false;

    const repeatRaw = parseInt(String(meta.repeat ?? ''), 10);

    // The prior row for this file (case-insensitive path match) — read BEFORE
    // the new row is assembled so owned-but-unpopulated fields can fall back to
    // it instead of blanking real data (§4.11 "never strip").
    const before = prior.samplesByPath.get(sample.audioPath.toLowerCase());
    const priorString = (key: string): string => {
      const v = before?.[key];
      return typeof v === 'string' ? v : '';
    };

    const row: Record<string, unknown> = {
      id: datasetSampleId(sample.relPath),
      audio_path: sample.audioPath,
      filename: sample.filename,
      caption,
      genre: meta.genre || '',
      lyrics,
      raw_lyrics: rawLyrics,
      // A later phase populates formatted_lyrics; until then keep whatever the
      // previous dataset.json had rather than blanking it.
      formatted_lyrics: priorString('formatted_lyrics'),
      bpm: parseBpmValue(meta.bpm),
      // Normalised to the FSM's vocabulary, not written through raw: the sidecar
      // holds "F# Major" / "4/4" for humans, but the LM can only ever emit
      // "F# major" / "4", so the raw forms would train it on unreachable tokens.
      keyscale: normalizeKeyscale(meta.key),
      timesignature: normalizeTimeSignature(meta.signature),
      duration,
      // Sidecars carry no `language:` line in the real corpus — don't downgrade
      // a known language to 'unknown' on a rebuild.
      language: meta.language || priorString('language') || 'unknown',
      is_instrumental: isInstrumental,
      custom_tag: (meta.custom_tag || '').trim() || ds.customTag,
      labeled: !!meta.caption,
      prompt_override: meta.prompt_override ? meta.prompt_override : null,
      repeat: Number.isFinite(repeatRaw) ? Math.max(1, repeatRaw) : 1,
    };

    // Carry forward every key we do not own (case-insensitive path match).
    if (before) {
      for (const [key, value] of Object.entries(before)) {
        if ((OWNED_SAMPLE_KEYS as readonly string[]).includes(key)) continue;
        row[key] = value;
      }
    }

    rows.push(row);
    onProgress?.(i + 1, included.length);
  }

  // "created" means created — a rebuild must not restamp it.
  const priorCreatedAt = typeof prior.metadata.created_at === 'string' && prior.metadata.created_at
    ? prior.metadata.created_at
    : '';

  const metadata: Record<string, unknown> = {
    name: ds.name,
    custom_tag: ds.customTag || '',
    tag_position: ds.tagPosition,
    created_at: priorCreatedAt || new Date().toISOString(),
    num_samples: rows.length,
    // §6.8: "every emitted sample has is_instrumental === true" — vacuously
    // true when nothing is emitted.
    all_instrumental: allInstrumental,
    genre_ratio: ds.genreRatio,
    default_artist: ds.defaultArtist || '',
    default_album: ds.defaultAlbum || '',
    default_genre: ds.defaultGenre || '',
    // ISO code, never a full name — see languageCodes.ts.
    default_language: ds.defaultLanguage ? normalizeLanguage(ds.defaultLanguage) : 'auto',
  };
  for (const [key, value] of Object.entries(prior.metadata)) {
    if ((OWNED_METADATA_KEYS as readonly string[]).includes(key)) continue;
    metadata[key] = value;
  }

  const dataset = { metadata, samples: rows };

  // Atomic write, same directory, UTF-8 no BOM, no ASCII escaping.
  const dir = path.dirname(outputPath);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = path.join(dir, `.dataset_${crypto.randomBytes(6).toString('hex')}.tmp`);
  try {
    fs.writeFileSync(tmp, JSON.stringify(dataset, null, 2), 'utf-8');
    fs.renameSync(tmp, outputPath);
  } catch (err) {
    try { fs.unlinkSync(tmp); } catch { /* already gone */ }
    throw err;
  }

  return { path: outputPath, total: rows.length, withMetadata };
}

/**
 * Adopt a dataset.json that already exists on disk but that the DB has no
 * record of.
 *
 * `builtAt` is written by exactly one thing — the Build job — so a dataset row
 * created for a folder that ALREADY contains a dataset.json (imported earlier,
 * built by a previous install, or produced outside the app) reads as unbuilt.
 * Preprocess and both trainers then refuse it with "Dataset must be built
 * first", which is untrue and unfixable from the UI without re-running a Build
 * the user deliberately skipped. A 19-folder bulk run failed every item this
 * way (2026-07-31).
 *
 * The file IS the build product, so the honest fix is to believe it. Adoption
 * is conservative: the shape must be the {metadata, samples[]} object
 * buildDataset writes, with at least one sample. Anything else is left alone
 * and the caller's existing refusal stands.
 *
 * Returns the row to carry on with — updated in place when adoption happened.
 */
export function adoptExistingDatasetJson(
  ds: TrainingDatasetRow,
  updateDataset: (id: string, patch: Partial<TrainingDatasetRow>) => unknown,
): TrainingDatasetRow {
  if (ds.builtAt && ds.datasetJsonPath && fs.existsSync(ds.datasetJsonPath)) {
    return ds;   // already known-built; nothing to do
  }
  const candidate = ds.datasetJsonPath || path.join(ds.sourceDir, 'dataset.json');
  let stat: fs.Stats;
  try {
    stat = fs.statSync(candidate);
    if (!stat.isFile()) return ds;
  } catch {
    return ds;
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(candidate, 'utf-8')) as {
      metadata?: unknown; samples?: unknown;
    };
    const samples = Array.isArray(parsed?.samples) ? parsed.samples : null;
    if (!samples || samples.length === 0) return ds;
    if (!parsed.metadata || typeof parsed.metadata !== 'object') return ds;
  } catch {
    return ds;   // unreadable or not JSON — not a build product
  }

  // mtime, not now(): the honest answer to "when was this built" is when the
  // file was written, and a Monitor row showing today for a month-old build
  // would be its own small lie.
  const builtAt = new Date(stat.mtimeMs).toISOString();
  updateDataset(ds.id, { builtAt, datasetJsonPath: candidate, status: 'built' });
  console.log(`[Training] Adopted existing dataset.json for ${ds.slug}: ${candidate}`);
  return { ...ds, builtAt, datasetJsonPath: candidate, status: 'built' };
}
