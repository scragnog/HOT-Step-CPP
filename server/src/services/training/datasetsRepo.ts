// training/datasetsRepo.ts — training_datasets CRUD
//
// One row per dataset, never per sample (D3): disk is the source of truth for
// sample data, and this table exists so GET /datasets is a single query with no
// disk I/O. snake_case ↔ camelCase conversion happens here and nowhere else.
//
// getDb() is only ever called inside a function — never at module import time.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §3, §4.12

import { getDb } from '../../db/database.js';
import { normalizeLanguage } from '../languageCodes.js';
import type { TagPosition, TrainingDatasetRow } from './types.js';

interface DbRow {
  id: string;
  slug: string;
  name: string;
  source_dir: string;
  recursive: number;
  custom_tag: string;
  tag_position: string;
  genre_ratio: number;
  default_artist: string;
  default_album: string;
  default_genre: string;
  default_language: string;
  sample_count: number;
  labeled_count: number;
  excluded_count: number;
  status: string;
  built_at: string;
  dataset_json_path: string;
  album_name: string;
  lyrics_set_id: number;
  created_at: string;
  updated_at: string;
}

const VALID_TAG_POSITIONS: readonly TagPosition[] = ['prepend', 'append', 'replace'];
const VALID_STATUSES: readonly TrainingDatasetRow['status'][] = ['draft', 'labeling', 'labeled', 'built', 'error'];

function toRow(r: DbRow): TrainingDatasetRow {
  return {
    id: r.id,
    slug: r.slug,
    name: r.name,
    sourceDir: r.source_dir,
    recursive: !!r.recursive,
    customTag: r.custom_tag,
    tagPosition: VALID_TAG_POSITIONS.includes(r.tag_position as TagPosition)
      ? r.tag_position as TagPosition
      : 'prepend',
    genreRatio: r.genre_ratio,
    defaultArtist: r.default_artist,
    defaultAlbum: r.default_album,
    defaultGenre: r.default_genre,
    // Normalized on READ so the ~183 rows still storing the old 'english'
    // literal cannot re-seed it into sidecars (labelingQueue force-writes this
    // value on every label pass) and from there back into training data. An
    // EMPTY value stays empty — it means "do not force a language", which is
    // not the same as 'en'. See services/languageCodes.ts.
    defaultLanguage: r.default_language ? normalizeLanguage(r.default_language) : '',
    sampleCount: r.sample_count,
    labeledCount: r.labeled_count,
    excludedCount: r.excluded_count,
    status: VALID_STATUSES.includes(r.status as TrainingDatasetRow['status'])
      ? r.status as TrainingDatasetRow['status']
      : 'draft',
    builtAt: r.built_at,
    datasetJsonPath: r.dataset_json_path,
    albumName: r.album_name ?? '',
    lyricsSetId: r.lyrics_set_id ?? 0,
    createdAt: r.created_at,
    updatedAt: r.updated_at,
  };
}

/** camelCase field → column name, for PATCH-style partial updates. */
const COLUMN_OF: Record<string, string> = {
  name: 'name',
  recursive: 'recursive',
  customTag: 'custom_tag',
  tagPosition: 'tag_position',
  genreRatio: 'genre_ratio',
  defaultArtist: 'default_artist',
  defaultAlbum: 'default_album',
  defaultGenre: 'default_genre',
  defaultLanguage: 'default_language',
  sampleCount: 'sample_count',
  labeledCount: 'labeled_count',
  excludedCount: 'excluded_count',
  status: 'status',
  builtAt: 'built_at',
  datasetJsonPath: 'dataset_json_path',
  albumName: 'album_name',
  lyricsSetId: 'lyrics_set_id',
};

export function listDatasets(): TrainingDatasetRow[] {
  const rows = getDb()
    .prepare('SELECT * FROM training_datasets ORDER BY created_at DESC, rowid DESC')
    .all() as DbRow[];
  return rows.map(toRow);
}

export function getDataset(id: string): TrainingDatasetRow | null {
  const row = getDb()
    .prepare('SELECT * FROM training_datasets WHERE id = ?')
    .get(id) as DbRow | undefined;
  return row ? toRow(row) : null;
}

export function getDatasetBySourceDir(dir: string): TrainingDatasetRow | null {
  // §2.3's 409 must hold on Windows, where `D:\Music\Artist` and `d:\music\artist`
  // are the same folder but different strings (path.resolve preserves casing and
  // the UNIQUE index collates BINARY).
  const sql = process.platform === 'win32'
    ? 'SELECT * FROM training_datasets WHERE source_dir = ? COLLATE NOCASE'
    : 'SELECT * FROM training_datasets WHERE source_dir = ?';
  const row = getDb().prepare(sql).get(dir) as DbRow | undefined;
  return row ? toRow(row) : null;
}

export function listSlugs(): Set<string> {
  const rows = getDb().prepare('SELECT slug FROM training_datasets').all() as Array<{ slug: string }>;
  return new Set(rows.map(r => r.slug));
}

export function insertDataset(row: TrainingDatasetRow): void {
  getDb().prepare(`
    INSERT INTO training_datasets (
      id, slug, name, source_dir, recursive, custom_tag, tag_position, genre_ratio,
      default_artist, default_album, default_genre, default_language, sample_count, labeled_count,
      excluded_count, status, built_at, dataset_json_path, album_name, created_at, updated_at
    ) VALUES (
      @id, @slug, @name, @source_dir, @recursive, @custom_tag, @tag_position, @genre_ratio,
      @default_artist, @default_album, @default_genre, @default_language, @sample_count, @labeled_count,
      @excluded_count, @status, @built_at, @dataset_json_path, @album_name, @created_at, @updated_at
    )
  `).run({
    id: row.id,
    slug: row.slug,
    name: row.name,
    source_dir: row.sourceDir,
    recursive: row.recursive ? 1 : 0,
    custom_tag: row.customTag,
    tag_position: row.tagPosition,
    genre_ratio: row.genreRatio,
    default_artist: row.defaultArtist,
    default_album: row.defaultAlbum,
    default_genre: row.defaultGenre,
    default_language: row.defaultLanguage,
    sample_count: row.sampleCount,
    labeled_count: row.labeledCount,
    excluded_count: row.excludedCount,
    status: row.status,
    built_at: row.builtAt,
    dataset_json_path: row.datasetJsonPath,
    album_name: row.albumName ?? '',
    created_at: row.createdAt,
    updated_at: row.updatedAt,
  });
}

export function updateDataset(id: string, patch: Partial<TrainingDatasetRow>): void {
  const sets: string[] = [];
  const params: Record<string, unknown> = { id };

  for (const [key, value] of Object.entries(patch)) {
    const column = COLUMN_OF[key];
    if (!column || value === undefined) continue;
    sets.push(`${column} = @${column}`);
    params[column] = typeof value === 'boolean' ? (value ? 1 : 0) : value;
  }
  if (sets.length === 0) return;

  sets.push(`updated_at = @updated_at`);
  params.updated_at = new Date().toISOString();

  getDb().prepare(`UPDATE training_datasets SET ${sets.join(', ')} WHERE id = @id`).run(params);
}

/** Cache the album name detected from tags. Display-only, so it is not an
 *  edit: updated_at (the dataset revision) stays put. */
export function cacheAlbumName(id: string, album: string): void {
  getDb().prepare('UPDATE training_datasets SET album_name = ? WHERE id = ?').run(album, id);
}

export function updateCounters(
  id: string,
  c: { sampleCount: number; labeledCount: number; excludedCount: number; status: string },
): void {
  getDb().prepare(`
    UPDATE training_datasets
       SET sample_count = @sample_count, labeled_count = @labeled_count,
           excluded_count = @excluded_count, status = @status, updated_at = @updated_at
     WHERE id = @id
       -- Unchanged counters are not an edit: updated_at is the dataset's
       -- revision, and a read-only view must not invalidate accepted work.
       AND (sample_count IS NOT @sample_count OR labeled_count IS NOT @labeled_count
            OR excluded_count IS NOT @excluded_count OR status IS NOT @status)
  `).run({
    id,
    sample_count: c.sampleCount,
    labeled_count: c.labeledCount,
    excluded_count: c.excludedCount,
    status: c.status,
    updated_at: new Date().toISOString(),
  });
}

export function deleteDataset(id: string): void {
  getDb().prepare('DELETE FROM training_datasets WHERE id = ?').run(id);
}
