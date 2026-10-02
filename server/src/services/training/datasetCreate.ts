// training/datasetCreate.ts — the POST /datasets creation block, as a service
//
// Extracted so the batch pipeline creates datasets EXACTLY the way the route
// does (batch-pipeline §3.3) instead of duplicating the seeding rules. Every
// check, message string and derived value is unchanged; the status code the
// route used to send inline now rides on DatasetCreateError.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §2.3, §8.5

import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import { readDatasetJsonMetadata } from './datasetBuilder.js';
import { detailFor, syncCounters } from './datasetDetail.js';
import { datasetLanguage } from '../languageCodes.js';
import { scanPreview as scanPreviewFolder } from './datasetScan.js';
import * as repo from './datasetsRepo.js';
import { uniqueDatasetTrigger } from './datasetTrigger.js';
import { labelsDir, slugify, uniqueSlug } from './paths.js';
import type { CreateDatasetInput, TrainingDatasetDetail, TrainingDatasetRow } from './types.js';

/** Carries the HTTP status the route answers this message with. */
export class DatasetCreateError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = 'DatasetCreateError';
  }
}

function metaString(meta: Record<string, unknown>, key: string): string {
  const v = meta[key];
  return typeof v === 'string' ? v : '';
}

function isWritableDir(dir: string): boolean {
  const probe = path.join(dir, '.hotstep-write-test');
  try {
    fs.writeFileSync(probe, 'x');
    fs.unlinkSync(probe);
    return true;
  } catch {
    return false;
  }
}

export async function createDatasetFromFolder(input: CreateDatasetInput): Promise<TrainingDatasetDetail> {
  // Typed at the boundary, untyped inside: these are the route's own defensive
  // reads of a raw request body and they must keep accepting anything.
  const body = (input || {}) as Record<string, any>;
  const name = typeof body.name === 'string' ? body.name.trim() : '';
  const rawDir = typeof body.sourceDir === 'string' ? body.sourceDir.trim() : '';

  if (!name || name.length > 100) {
    throw new DatasetCreateError(400, 'name is required');
  }
  if (!rawDir) {
    throw new DatasetCreateError(400, 'sourceDir is required');
  }

  const sourceDir = path.resolve(rawDir);
  if (!fs.existsSync(sourceDir) || !fs.statSync(sourceDir).isDirectory()) {
    throw new DatasetCreateError(400, 'sourceDir is not a directory');
  }
  if (!isWritableDir(sourceDir)) {
    throw new DatasetCreateError(400, 'sourceDir is not writable — sidecars cannot be saved there');
  }
  if (repo.getDatasetBySourceDir(sourceDir)) {
    throw new DatasetCreateError(409, 'A dataset already exists for this folder');
  }

  const recursive = body.recursive !== false;
  const preview = scanPreviewFolder(sourceDir, recursive);
  if (preview.audioFiles === 0) {
    throw new DatasetCreateError(400, 'No audio files found in sourceDir');
  }

  const slug = uniqueSlug(slugify(name), repo.listSlugs());

  // §8.5 — a folder that already has a dataset.json owns these settings.
  // Seeding the row from it is what makes "existing labels will be imported
  // and kept" true: the builder rewrites the same values instead of blanking
  // custom_tag / genre_ratio / default_artist / default_album on the first build.
  const priorMeta = readDatasetJsonMetadata(sourceDir);
  const baseTrigger = typeof body.customTag === 'string'
    ? body.customTag.trim() : (metaString(priorMeta, 'custom_tag') || slug);
  const takenTriggers = new Set(repo.listDatasets().map(ds => ds.customTag.toLowerCase()));
  const priorTagPosition = metaString(priorMeta, 'tag_position');
  const priorGenreRatio = Number(priorMeta.genre_ratio);

  const now = new Date().toISOString();
  const row: TrainingDatasetRow = {
    id: randomUUID(),
    slug,
    name,
    sourceDir,
    recursive,
    customTag: baseTrigger ? uniqueDatasetTrigger(baseTrigger, slug, takenTriggers) : '',
    tagPosition: ['prepend', 'append', 'replace'].includes(body.tagPosition)
      ? body.tagPosition
      : (['prepend', 'append', 'replace'].includes(priorTagPosition)
        ? priorTagPosition as TrainingDatasetRow['tagPosition']
        : 'prepend'),
    genreRatio: Number.isFinite(Number(body.genreRatio))
      ? Math.min(100, Math.max(0, Math.trunc(Number(body.genreRatio))))
      : (Number.isFinite(priorGenreRatio) ? Math.min(100, Math.max(0, Math.trunc(priorGenreRatio))) : 30),
    defaultArtist: typeof body.defaultArtist === 'string' ? body.defaultArtist : metaString(priorMeta, 'default_artist'),
    defaultAlbum: typeof body.defaultAlbum === 'string' ? body.defaultAlbum : metaString(priorMeta, 'default_album'),
    defaultGenre: typeof body.defaultGenre === 'string' ? body.defaultGenre : metaString(priorMeta, 'default_genre'),
    // Normalized to an ISO code the inference FSM can emit — a full name like
    // 'english' trains adapters to produce a token the sampler forbids
    // (languageCodes.ts has the full account).
    // 'auto' (mixed-language corpora, #191) stores '' — no override, each
    // file keeps the local AI's own language guess.
    defaultLanguage: datasetLanguage(
      typeof body.defaultLanguage === 'string' && body.defaultLanguage.trim()
        ? body.defaultLanguage
        : metaString(priorMeta, 'default_language')),
    sampleCount: preview.audioFiles,
    labeledCount: preview.withCaption,
    excludedCount: 0,
    status: 'draft',
    builtAt: '',
    datasetJsonPath: '',
    // Detected from the tracks' tags on the first scan below (syncCounters →
    // syncAlbumName), or lazily by the list endpoint — never guessed here.
    albumName: '',
    createdAt: now,
    updatedAt: now,
  };

  repo.insertDataset(row);
  fs.mkdirSync(labelsDir(slug), { recursive: true });

  const detail = await detailFor(row);
  syncCounters(row, detail.samples);
  console.log(`[Training] Created dataset "${name}" (${slug}) — ${detail.samples.length} files in ${sourceDir}`);
  return detail;
}
