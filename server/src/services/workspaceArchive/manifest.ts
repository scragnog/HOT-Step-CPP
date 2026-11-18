// workspaceArchive/manifest.ts — The workspace archive format (GH #228).
//
// A store-mode ZIP64 whose entries mirror a default data-folder layout
// (library/..., library/training/..., adapters/..., models/..., settings/.env)
// and whose last entry is manifest.json. A zip without a manifest is
// unfinished; import refuses it. Types and constants only: the CLI imports
// this before config may load.

export const MANIFEST_NAME = 'manifest.json';
export const ARCHIVE_KIND = 'hot-step-workspace';
export const ARCHIVE_FORMAT = 1;

export type CategoryId = 'library' | 'training' | 'adapters' | 'models' | 'secrets';
export const CATEGORY_IDS: CategoryId[] = ['library', 'training', 'adapters', 'models', 'secrets'];

/** What the caller chose. Library is always carried. */
export interface CategoryChoice { training: boolean; adapters: boolean; models: boolean; secrets: boolean }
export const DEFAULT_CHOICE: CategoryChoice = { training: true, adapters: true, models: false, secrets: false };

export interface ManifestSource {
  category: CategoryId;
  /** Zip prefix, forward slashes (e.g. "library/training"). */
  prefix: string;
  /** Absolute folder on the exporting machine (the rewrite pair's `from`). */
  path: string;
  files: number;
  bytes: number;
}

export interface ManifestFile { name: string; bytes: number; sha256: string }
export interface SkippedEntry { path: string; reason: 'unknown' | 'cache' | 'temp' | 'backup' | 'link' | 'secret' | 'machine' | 'category-off' }
/** An absolute path stored in the DB that exists on the exporting machine but
 *  lies outside every carried folder: import lists it, never rewrites it. */
export interface ExternalRef { path: string; refs: number }

export interface Manifest {
  kind: typeof ARCHIVE_KIND;
  format: typeof ARCHIVE_FORMAT;
  createdAt: string;
  appVersion: string;
  platform: NodeJS.Platform;
  dataRoot: string | null;
  categories: Record<CategoryId, boolean>;
  sources: ManifestSource[];
  /** Every entry except manifest.json, in zip order. */
  files: ManifestFile[];
  db: { name: string; userVersion: number; credentialRowsRemoved: number } | null;
  env: { name: string; droppedKeys: string[] } | null;
  skipped: SkippedEntry[];
  /** Entries beyond the cap are counted, not named. */
  skippedNotListed: number;
  externals: ExternalRef[];
}
