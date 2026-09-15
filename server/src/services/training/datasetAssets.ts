// training/datasetAssets.ts — "what has this dataset actually got?" off disk
//
// The dataset row says whether a dataset was BUILT and nothing else, so the
// list used to be silent about the three stages that matter most: preprocess,
// train-dit, train-lm. Every one of those leaves an artefact on disk, and disk
// stays the source of truth (D3) — so the flags are re-read on every request
// rather than cached in a column. Deleting a tensors folder or an adapter shows
// up on the next refresh.
//
// The friendly ALBUM name is the one exception: detecting it means reading
// embedded tags, which the list cannot afford per request. It is detected once
// (label store first, then a bounded probe of the audio files) and cached in
// training_datasets.album_name, then refreshed for free whenever a caller
// already holds the sample list.
//
// Never throws — an unreadable adapter root or tensors dir degrades to
// "nothing trained", never to a 500.

import fs from 'fs';
import path from 'path';
import * as audioMeta from './audioMeta.js';
import { latestRunDir, lmSizeFromSlug } from './adapterLayout.js';
import { scanAudioFiles } from './datasetScan.js';
import * as repo from './datasetsRepo.js';
import { readAllLabels } from './labelStore.js';
import { findMm3LmAdapter, findMm3LmAdaptersFor } from './mm3Runs.js';
import { mm3CodesDir } from './mm3Train.js';
import { countPreprocessedVariants } from './preprocessStatus.js';
import { readTrainDitStatus } from './trainDitStatus.js';
import { adapterLmRoot, lmArtistDirFor, safeAdapterName } from './trainLmStatus.js';
import type {
  DatasetAssets, LmSize, TrainingAdapterHit, TrainingDatasetRow, TrainingSample,
} from './types.js';
import { findYue2ArAdapter, findYue2ArAdaptersFor } from './yue2ArRuns.js';
import { findYue2NarAdapter, findYue2NarAdaptersFor } from './yue2Runs.js';
import { yue2PreprocessManifest } from './yue2Train.js';

// ── Trained-adapter lookup ───────────────────────────────────────────────
//
// Both finders live here rather than in lyricStudioExport.ts (their original
// home) because the dataset list, the batch wizard and the Lyric Studio export
// must all agree on what "this dataset has an adapter" means.

const LM_SIZES: LmSize[] = ['4B', '1.7B', '0.6B'];

function weightsMtime(dir: string): number {
  for (const f of ['adapter_model.safetensors', 'lokr_weights.safetensors']) {
    try { return fs.statSync(path.join(dir, f)).mtimeMs; } catch { /* next */ }
  }
  return 0;
}

/** Newest trained planner-LM adapter for this dataset across every size root
 *  (plus the legacy flat `lm/<name>-<size>` dir). null when none trained. */
export function findLmAdapter(ds: TrainingDatasetRow): TrainingAdapterHit | null {
  let best: { hit: TrainingAdapterHit; mtime: number } | null = null;
  for (const size of LM_SIZES) {
    const candidates = [
      latestRunDir(lmArtistDirFor(ds.slug, size)),
      path.join(adapterLmRoot(), `${safeAdapterName(ds.slug)}-${size}`),
    ];
    for (const dir of candidates) {
      if (!dir) continue;
      const mtime = weightsMtime(dir);
      if (!mtime || (best && mtime <= best.mtime)) continue;
      // A legacy flat dir's parent is `lm/`, whose slug lookup fails → keep the
      // loop's size; a per-size run dir confirms it from the folder itself.
      const sizeOfDir = lmSizeFromSlug(path.basename(path.dirname(path.dirname(dir)))) || size;
      best = {
        mtime,
        hit: { path: dir, kind: 'lm', detail: sizeOfDir, trainedAt: new Date(mtime).toISOString() },
      };
    }
  }
  return best ? best.hit : null;
}

/** The dataset's newest trained DiT adapter, resolved exactly the way the
 *  Train panel resolves it (newest preprocess variant → base → latest run). */
export function findDitAdapter(ds: TrainingDatasetRow): TrainingAdapterHit | null {
  return ditHitFrom(readTrainDitStatus(ds, {}));
}

/** The hit for an already-read DiT status — so readDatasetAssets pays for the
 *  variant scan once instead of twice. */
function ditHitFrom(status: ReturnType<typeof readTrainDitStatus>): TrainingAdapterHit | null {
  if (!status.adapterExists) return null;
  return {
    path: status.adapterDir,
    kind: 'dit',
    // The per-base folder the run lives in (…/dit-<base>/<artist>/<stamp>).
    detail: path.basename(path.dirname(path.dirname(status.adapterDir))),
    trainedAt: status.trainedAt || new Date(weightsMtime(status.adapterDir) || Date.now()).toISOString(),
  };
}

// ── MM3 / YuE2 cheap stage reads ─────────────────────────────────────────
//
// Both blocks below are computed unconditionally on every row (GET /datasets
// has no backend query param), so each read has to stay as cheap as the ACE
// reads above — no readLog, no dirSize, no per-checkpoint header parse. The
// expensive run listers (listMm3Runs/listYue2Runs/listYue2ArRuns) are never
// called from here; findMm3LmAdapter(sFor)/findYue2NarAdapter(sFor)/
// findYue2ArAdapter(sFor) are the cheap analogues built for this file.

/** MM3's codes cache count — one readdir of `<dataset>/mm3-codes/codes`.
 *  Mirrors the pattern GET /datasets/:id/mm3 uses (routes/training.ts). */
function readMm3CodesCount(slug: string): number {
  try {
    const inner = path.join(mm3CodesDir(slug), 'codes');
    return fs.readdirSync(inner).filter(f => f.endsWith('.codes')).length;
  } catch {
    return 0;
  }
}

/** The three YuE2 preprocess-stage flags, off ONE read of the dataset's
 *  yue2_preprocess.json. readYue2PreprocessSummary/readYue2CodecIdsStatus/
 *  readYue2CursorWordsStatus (yue2Train.ts/yue2Tokenize.ts/yue2Align.ts) each
 *  reopen and reparse that same file — fine for a single status poll, not for
 *  a per-row list loop, so this reads and parses it once and derives all
 *  three. Never throws: no manifest yet just means nothing is ready. */
function readYue2StageFlags(slug: string): {
  latentsReady: boolean; clips: number; codesReady: boolean; cursorReady: boolean;
} {
  const empty = { latentsReady: false, clips: 0, codesReady: false, cursorReady: false };
  let j: Record<string, unknown>;
  try {
    j = JSON.parse(fs.readFileSync(yue2PreprocessManifest(slug), 'utf-8')) as Record<string, unknown>;
  } catch {
    return empty;
  }
  const nClips = Number(j.n_clips);
  const sources = Array.isArray(j.sources) ? j.sources as Array<Record<string, unknown>> : [];
  return {
    latentsReady: Number.isFinite(nClips) && nClips > 0,
    clips: Number.isFinite(nClips) ? nClips : 0,
    codesReady: j.codec_ids_present === true,
    cursorReady: sources.some(s => typeof s.cursor_words === 'string' && s.cursor_words !== ''),
  };
}

function safeMm3Adapter(ds: TrainingDatasetRow): TrainingAdapterHit | null {
  try { return findMm3LmAdapter(ds); } catch { return null; }
}
function safeYue2NarAdapter(ds: TrainingDatasetRow): TrainingAdapterHit | null {
  try { return findYue2NarAdapter(ds); } catch { return null; }
}
function safeYue2ArAdapter(ds: TrainingDatasetRow): TrainingAdapterHit | null {
  try { return findYue2ArAdapter(ds); } catch { return null; }
}

// ── Assets ───────────────────────────────────────────────────────────────

/**
 * Every pipeline artefact this dataset has on disk. Cheap enough for the list
 * endpoint: one readdir of the tensors root, one of the newest variant, a
 * couple of stats per adapter candidate, and one small JSON parse.
 *
 * `labeled` comes off the row's cached counter, not a fresh scan — the list
 * already shows that counter, and re-scanning N source folders to confirm it
 * would cost far more than the flag is worth.
 *
 * `precomputed` lets listDatasetsWithAssets() hand in the MM3/YuE2 adapter
 * hits it already found in one pass over each adapter root, instead of this
 * function re-scanning that root per row. Omit it (as datasetDetail.ts does,
 * for one dataset) and it does the single-dataset lookup itself.
 */
export function readDatasetAssets(
  ds: TrainingDatasetRow,
  precomputed?: {
    mm3Adapter: TrainingAdapterHit | null;
    yue2NarAdapter: TrainingAdapterHit | null;
    yue2ArAdapter: TrainingAdapterHit | null;
  },
): DatasetAssets {
  const assets: DatasetAssets = {
    labeled: ds.labeledCount > 0,
    built: ds.status === 'built' || !!ds.builtAt,
    tensorVariants: 0,
    tensorVariantKey: '',
    tensorSamples: 0,
    ditBase: '',
    lm: null,
    dit: null,
    mm3: { codesReady: false, codesCount: 0, lmAdapter: null },
    yue2: { latentsReady: false, clips: 0, codesReady: false, cursorReady: false, narAdapter: null, arAdapter: null },
  };

  // A row that says 'built' but whose dataset.json has since been deleted is
  // not built any more — the file is what the trainers actually read.
  if (assets.built && ds.datasetJsonPath) {
    try { assets.built = fs.existsSync(ds.datasetJsonPath); } catch { /* keep the row's word */ }
  }

  try { assets.tensorVariants = countPreprocessedVariants(ds.slug); } catch { /* stays 0 */ }

  try {
    const dit = readTrainDitStatus(ds, {});
    assets.tensorVariantKey = dit.variantKey;
    assets.tensorSamples = dit.sampleCount;
    assets.ditBase = dit.ditModel;
    assets.dit = ditHitFrom(dit);
  } catch { /* stays null/0 */ }

  try { assets.lm = findLmAdapter(ds); } catch { /* stays null */ }

  const mm3CodesCount = readMm3CodesCount(ds.slug);
  assets.mm3 = {
    codesReady: mm3CodesCount > 0,
    codesCount: mm3CodesCount,
    lmAdapter: precomputed ? precomputed.mm3Adapter : safeMm3Adapter(ds),
  };

  const yue2Flags = readYue2StageFlags(ds.slug);
  assets.yue2 = {
    ...yue2Flags,
    narAdapter: precomputed ? precomputed.yue2NarAdapter : safeYue2NarAdapter(ds),
    arAdapter: precomputed ? precomputed.yue2ArAdapter : safeYue2ArAdapter(ds),
  };

  return assets;
}

// ── Album detection ──────────────────────────────────────────────────────

/** Most common non-empty value, case-insensitively grouped, first-seen casing
 *  returned. '' when nothing usable at all. */
function majority(values: string[]): string {
  const counts = new Map<string, { display: string; n: number }>();
  for (const raw of values) {
    const v = String(raw ?? '').trim();
    if (!v) continue;
    const k = v.toLowerCase();
    const hit = counts.get(k);
    if (hit) hit.n++;
    else counts.set(k, { display: v, n: 1 });
  }
  let best: { display: string; n: number } | null = null;
  for (const c of counts.values()) if (!best || c.n > best.n) best = c;
  return best ? best.display : '';
}

/**
 * Album name from a sample list already in hand — tag majority first, then the
 * dataset's own default.
 *
 * The SOURCE FOLDER NAME is deliberately not a fallback here (unlike the Lyric
 * Studio export's detectAlbum): this value is displayed as "the album these
 * tracks say they are", and a folder name dressed up as a tag read is exactly
 * the confusion that published "4yearstrong_someway" as an album (2026-07-31).
 * '' means "unknown", and the UI shows the dataset name instead.
 */
export function albumFromSamples(ds: TrainingDatasetRow, samples: TrainingSample[]): string {
  const included = samples.filter(s => !s.excluded && !s.fileMissing);
  return majority(included.map(s => s.tagAlbum)) || ds.defaultAlbum.trim();
}

/** Persist a freshly detected album name when it actually changed. */
function cacheAlbum(ds: TrainingDatasetRow, album: string): void {
  if (!album || album === ds.albumName) return;
  ds.albumName = album;
  try { repo.updateDataset(ds.id, { albumName: album }); } catch { /* display-only */ }
}

/** Refresh the cached album from a scan that already has the samples — free,
 *  and it upgrades a probe-derived guess the moment the Label job writes tags. */
export function syncAlbumName(ds: TrainingDatasetRow, samples: TrainingSample[]): void {
  cacheAlbum(ds, albumFromSamples(ds, samples));
}

/** Datasets whose files have already been probed this process. A probe that
 *  finds nothing must not re-read the same audio on every list request; a
 *  server restart is a cheap enough retry. */
const probed = new Set<string>();

/** How many files a cold dataset is worth reading tags out of — an album only
 *  needs a majority, not a census. Same bound the Lyric Studio export uses. */
const TAG_PROBE_LIMIT = 12;

/**
 * The album name for a dataset that has never been opened: label-store tags
 * first (no audio parsing at all), then a bounded read of the files themselves.
 * Whatever it finds is cached, so this costs something exactly once.
 */
export async function ensureAlbumName(ds: TrainingDatasetRow): Promise<string> {
  if (ds.albumName.trim()) return ds.albumName;
  if (probed.has(ds.id)) return '';
  probed.add(ds.id);

  try {
    const labels = [...readAllLabels(ds.slug).values()];
    const fromLabels = majority(labels.map(l => l.tags?.album ?? ''));
    if (fromLabels) { cacheAlbum(ds, fromLabels); return fromLabels; }
  } catch { /* no labels dir — fall through to the file probe */ }

  let files: string[] = [];
  try {
    files = scanAudioFiles(ds.sourceDir, ds.recursive).slice(0, TAG_PROBE_LIMIT).map(f => f.absPath);
  } catch { /* folder gone or too large — the default below still applies */ }

  const albums: string[] = [];
  for (const file of files) {
    try {
      const md = await audioMeta.read(file);
      if (md.album) albums.push(md.album);
    } catch { /* unreadable file — the rest still vote */ }
  }

  const album = majority(albums) || ds.defaultAlbum.trim();
  cacheAlbum(ds, album);
  return album;
}

/** How long ONE list request will spend detecting albums before handing the
 *  rest off to a background pass. A corpus of never-opened datasets must not
 *  turn the first list of the session into a multi-second stall; whatever the
 *  background finishes is cached and shows up on the next refresh. */
const ALBUM_BUDGET_MS = 1500;

function probeInBackground(rows: TrainingDatasetRow[]): void {
  void (async () => {
    for (const ds of rows) {
      try { await ensureAlbumName(ds); } catch { /* album stays '' */ }
    }
  })();
}

/** The list payload: every row with its album filled in and its disk artefacts
 *  attached. Album detection is bounded and once-per-dataset; the asset read is
 *  fresh every time. */
export async function listDatasetsWithAssets(): Promise<TrainingDatasetRow[]> {
  const rows = repo.listDatasets();
  const deadline = Date.now() + ALBUM_BUDGET_MS;
  const deferred: TrainingDatasetRow[] = [];

  // One pass over each adapter root for the whole list, not one per row —
  // see the finders' own headers (mm3Runs.ts/yue2Runs.ts/yue2ArRuns.ts) for
  // why readMm3Run/readYue2Run/readYue2ArRun must never be called per row.
  const idsAndSlugs = rows.map(r => ({ id: r.id, slug: r.slug }));
  const mm3Adapters = findMm3LmAdaptersFor(idsAndSlugs);
  const yue2NarAdapters = findYue2NarAdaptersFor(idsAndSlugs);
  const yue2ArAdapters = findYue2ArAdaptersFor(idsAndSlugs);

  for (const ds of rows) {
    if (!ds.albumName.trim() && !probed.has(ds.id)) {
      if (Date.now() < deadline) {
        try { await ensureAlbumName(ds); } catch { /* album stays '' */ }
      } else {
        deferred.push(ds);
      }
    }
    ds.assets = readDatasetAssets(ds, {
      mm3Adapter: mm3Adapters.get(ds.id) ?? null,
      yue2NarAdapter: yue2NarAdapters.get(ds.id) ?? null,
      yue2ArAdapter: yue2ArAdapters.get(ds.id) ?? null,
    });
  }

  if (deferred.length) probeInBackground(deferred);
  return rows;
}
