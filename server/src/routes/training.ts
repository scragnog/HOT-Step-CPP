// training.ts — Dataset Studio HTTP surface (Training Studio phase 1)
//
// Import audio → local label (/understand + Essentia) → optional cloud enhance
// (Genius lyrics, LLM captions) → review/edit grid → build dataset.json.
//
// Disk is the source of truth (D3): sidecar .txt files next to the audio plus
// dataset.json in the source folder. SQLite holds ONE row per dataset for
// listing and status — never per-sample rows. Every GET re-scans.
//
// There is no global error middleware, so every handler is wrapped in
// try/catch and must respond, or the request hangs. Static routes are declared
// before any /:param routes.
//
// Mounts at: /api/training
// Routes:
//   GET    /capabilities                                — engine/Essentia/Genius/LLM probes
//   GET    /scan-preview                                — pre-create folder summary
//   GET    /datasets                                    — list (newest first)
//   POST   /datasets                                    — create
//   POST   /pipeline                                    — start a batch import + stage chain
//   GET    /pipeline                                    — active + recent pipelines
//   GET    /pipeline/:id                                — one pipeline
//   DELETE /pipeline/:id                                — cancel a pipeline
//   POST   /pipeline/:id/pause                          — hold at the next stage boundary
//   POST   /pipeline/:id/resume                         — continue a paused OR ended/cancelled pipeline
//   GET    /defaults                                    — stored per-stage defaults
//   PUT    /defaults                                    — set per-stage defaults
//   GET    /jobs                                        — active + finished jobs
//   GET    /jobs/:jobId                                 — poll a job
//   DELETE /jobs/:jobId                                 — cancel/forget a job
//   GET    /jobs/:jobId/stream                          — SSE: live job events
//   GET    /datasets/:id                                — detail (re-scans disk)
//   PATCH  /datasets/:id                                — settings
//   POST   /datasets/:id/rescan                         — pick up added/removed files
//   DELETE /datasets/:id                                — forget (never touches sourceDir)
//   POST   /datasets/:id/samples/bulk                   — bulk field set
//   PATCH  /datasets/:id/samples/:sampleId              — edit one sample
//   GET    /datasets/:id/samples/:sampleId/audio        — stream audio (Range aware)
//   GET    /datasets/:id/samples/:sampleId/mm3          — read <stem>.mm3.txt
//   PUT    /datasets/:id/samples/:sampleId/mm3          — write <stem>.mm3.txt
//   POST   /datasets/:id/label                          — start a labeling job
//   POST   /datasets/:id/enhance/genius                 — start a Genius job
//   POST   /datasets/:id/enhance/caption                — start an LLM caption job
//   POST   /datasets/:id/build                          — start a build job
//   GET    /datasets/:id/dataset-json                   — read back the built file
//   GET    /datasets/:id/lyric-studio                   — export preview (detected artist/album, adapters)
//   POST   /datasets/:id/lyric-studio                   — commit export into Lyric Studio
//   POST   /datasets/:id/preprocess                     — start a tensor-cache job
//   GET    /datasets/:id/preprocess                     — tensor-cache status
//   DELETE /datasets/:id/preprocess/:variantKey         — delete one cache variant
//   GET    /datasets/:id/mm3                            — MM3 codes cache + model readiness
//   POST   /datasets/:id/mm3-codes                      — audio -> RVQ codes (MM3)
//   POST   /datasets/:id/mm3-train-lm                   — start an MM3 LM LoRA training job
//   POST   /datasets/:id/train-lm                       — start an LM LoRA training job
//   GET    /datasets/:id/train-lm                       — LM adapter / codes status
//   POST   /datasets/:id/train-dit                      — start a DiT LoRA training job
//   GET    /datasets/:id/train-dit                      — DiT adapter / variant status
//   GET    /previews/:previewId/:slot                   — stream a codes preview (Range aware)
//   POST   /datasets/:id/audition                       — start an A/B codes-audition job
//   GET    /datasets/:id/audition                       — recent previews for a dataset
//   POST   /datasets/:id/samples/:sampleId/audition     — SYNC decode of a sample's stored codes

import { Router, Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { config } from '../config.js';
import { engineReady } from '../engineState.js';
import { aceClient } from '../services/aceClient.js';
import { listProviders, getProvider } from '../services/lireek/llm/registry.js';
import * as repo from '../services/training/datasetsRepo.js';
import {
  buildSamples, loadSidecarMetadata, sampleFromParts,
  scanPreview as scanPreviewFolder, ScanLimitError,
} from '../services/training/datasetScan.js';
import { isInside, trainingBaseDir } from '../services/training/paths.js';
import { resolveMossPaths } from '../services/training/mossCaption.js';
import { deleteLabel, deleteLabels, patchLabel, readLabel } from '../services/training/labelStore.js';
import { listDatasetsWithAssets } from '../services/training/datasetAssets.js';
import { createDatasetFromFolder, DatasetCreateError } from '../services/training/datasetCreate.js';
import { detailFor, syncCounters } from '../services/training/datasetDetail.js';
import {
  cancelPipeline, getPipeline, hasActivePipeline, listPipelines, pausePipeline, PIPELINE_STAGES,
  resumePipeline, startPipeline,
} from '../services/training/pipelineRunner.js';
import { getTrainingDefaults, setTrainingDefaults } from '../services/training/trainingDefaults.js';
import {
  availableMm3Bases, MM3_VRAM_MODEL, recommendMm3Config,
  MM3_LM_DEFAULTS, missingMm3TrainModels, mm3AdapterRunDir, mm3CodesDir, mm3PriorDir,
  mm3RunName,
  type Mm3BasePrecision,
} from '../services/training/mm3Train.js';
import {
  listMm3Runs, readMm3Run, resolveMm3RunDir, resumeOptionsFor,
} from '../services/training/mm3Runs.js';
import { writeSidecar } from '../services/training/sidecarIO.js';
import { essentiaAvailable } from '../services/training/essentiaClient.js';
import { engineQueueDepth, engineUnderstandReady, pickBestLm } from '../services/training/understandClient.js';
import * as queue from '../services/training/labelingQueue.js';
import { isEngineSuspended } from '../services/aceEngineProcess.js';
import {
  aceTrainExe, findRegCorpora, getModelSnapshot, pickBf16, pickDitBaseFor, pickLmFor, refreshModelSnapshot,
  tensorsDir, tensorsRoot, variantKeyFor,
  type ResolvedPreprocessOptions, type ResolvedTrainDitOptions, type ResolvedTrainLmOptions,
} from '../services/training/aceTrain.js';
import { readPreprocessStatus } from '../services/training/preprocessStatus.js';
import {
  adapterLmRoot, lmRunDirFor, newestVariantKey, readTrainLmStatus,
  safeAdapterName, variantDitModel, variantExists,
} from '../services/training/trainLmStatus.js';
import {
  adapterDitRoot, ditRunDirFor, readTrainDitStatus,
} from '../services/training/trainDitStatus.js';
import { hasWeights, lmAdapterRoots } from '../services/training/adapterLayout.js';
import { adoptExistingDatasetJson } from '../services/training/datasetBuilder.js';
import {
  getActiveModels, resolveTrainingDit, setActiveModels,
} from '../services/training/activeModels.js';
import {
  deleteDatasetPreviews, isPreviewFileKey, isPreviewId, listPreviews, previewsRoot,
  prunePreviews, resolvePreviewFile,
} from '../services/training/auditionStore.js';
import { AuditionError, decodeStoredCodes } from '../services/training/auditionService.js';
import {
  commitLyricStudioExport, LyricStudioExportError, previewLyricStudioExport,
} from '../services/training/lyricStudioExport.js';
import { getGenerations, getLyricsSet } from '../db/lireekDb.js';
import type {
  AuditionListResponse, AuditionOptions, AuditionSideSpec,
  BulkSetInput, CaptionOptions, CreateDatasetInput, FieldSource, GeniusOptions, LabelOptions, LmSize,
  LyricStudioExportInput, Mm3PreviewOptions,
  PatchSampleInput, PipelineFolderSpec, PipelineLabelOptions, PipelineStage,
  PreprocessCompat, PreprocessDtype, PreprocessNormalize, PreprocessOptions,
  TrainingCapabilities, TrainingDatasetRow, TrainingDefaults, TrainingSample,
  DitAdapterType, TrainDitOptions, TrainDitStage, TrainLmOptions, TrainLmStage,
} from '../services/training/types.js';

const router = Router();

const MIME_BY_EXT: Record<string, string> = {
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
  '.opus': 'audio/ogg',
  '.m4a': 'audio/mp4',
  '.aac': 'audio/aac',
};

// ── Helpers ──────────────────────────────────────────────────────────────

/** Re-read one sample from disk after a write — avoids a full rescan per edit. */
function reloadSample(ds: TrainingDatasetRow, sample: TrainingSample): TrainingSample {
  let sizeBytes = sample.sizeBytes;
  let mtimeMs = 0;
  try {
    const st = fs.statSync(sample.audioPath);
    sizeBytes = st.size;
    mtimeMs = st.mtimeMs;
  } catch { /* file vanished — keep the sizes we already had */ }
  const label = readLabel(ds.slug, sample.sampleId) ?? undefined;
  const meta = loadSidecarMetadata(sample.audioPath);
  return sampleFromParts(
    ds.id,
    { relPath: sample.relPath, absPath: sample.audioPath, sizeBytes, mtimeMs },
    meta, label, label?.durationCache?.seconds ?? sample.duration, sample.fileMissing,
  );
}

/** Sidecar keys for the editable sample fields (§6.4 write whitelist). */
const SIDECAR_KEY_OF: Record<string, string> = {
  caption: 'caption',
  genre: 'genre',
  bpm: 'bpm',
  key: 'key',
  signature: 'signature',
  language: 'language',
  isInstrumental: 'is_instrumental',
  lyrics: 'lyrics',
  customTag: 'custom_tag',
  repeat: 'repeat',
  promptOverride: 'prompt_override',
};

/** Sample fields whose provenance the contract tracks (§2.0 `sources`). */
const SOURCE_TRACKED_FIELDS: ReadonlySet<string> = new Set([
  'caption', 'lyrics', 'genre', 'bpm', 'key', 'signature', 'language',
]);

// ── Capabilities (§2.1) ──────────────────────────────────────────────────

router.get('/capabilities', async (_req: Request, res: Response) => {
  // Every probe degrades independently — this endpoint never throws upward.
  const caps: TrainingCapabilities = {
    engine: { up: false, ready: engineReady, understandSupported: false, missingModels: [], queueDepth: 0, lmModels: [], defaultLmModel: '' },
    essentia: { available: false, binPath: config.essentia.bin },
    genius: { configured: false },
    llm: { configured: false, defaultProvider: config.lireek.defaultProvider, providers: [] },
    // Local audio captioning. Probed synchronously below — it is two fs.existsSync
    // calls, not a subprocess, so it cannot hang this endpoint.
    moss: { available: false, missing: '' },
    preprocess: {
      available: false, binPath: '', ditModels: [], vaeModels: [], textEncoders: [],
      defaultDit: '', defaultVae: '', defaultTextEnc: '', modelsCachedAt: 0,
      engineSuspended: false,
    },
    trainLm: {
      available: false, lmModels: [], sizes: ['0.6B', '1.7B', '4B'],
      defaultLmBySize: { '0.6B': '', '1.7B': '', '4B': '' }, adaptersRoot: '',
    },
    trainDit: {
      // 8192 is ADVISORY and deliberately below what a comfortable run wants:
      // the engine decides, per run, and it can now express configurations that
      // fit well under this (bf16 mirror + --attn flash + --ckpt + a shallow
      // top-K). See the comment on TrainDitCapabilities.minVramMb.
      available: false, adapterTypes: ['lora', 'lokr'], adaptersRoot: '', minVramMb: 8192,
    },
  };

  try { caps.engine.up = await aceClient.isReachable(); } catch { /* stays false */ }
  if (caps.engine.up) {
    try {
      const ready = await engineUnderstandReady();
      caps.engine.understandSupported = ready.ok;
      caps.engine.missingModels = ready.missing;
      caps.engine.lmModels = ready.lmModels;
      caps.engine.defaultLmModel = pickBestLm(ready.lmModels);
    } catch { caps.engine.missingModels = ['lm', 'dit', 'vae']; }
    try { caps.engine.queueDepth = await engineQueueDepth(); } catch { /* stays 0 */ }
  }

  try { caps.essentia.available = essentiaAvailable(); } catch { /* stays false */ }
  try { caps.genius.configured = !!config.lireek.geniusAccessToken; } catch { /* stays false */ }

  // `missing` is surfaced rather than swallowed: the two failure modes are
  // "binary not built" and "weights not downloaded", and the user's next action
  // is completely different for each. A bare `available: false` would send
  // everyone to rebuild the engine when the usual cause is the 8 GB GGUF.
  try {
    const probe = resolveMossPaths();
    if ('paths' in probe) caps.moss.available = true;
    else caps.moss.missing = probe.missing;
  } catch (err: any) { caps.moss.missing = String(err?.message || err).slice(0, 200); }

  try {
    const providers = await listProviders();
    caps.llm.providers = providers.map(p => ({
      id: p.id,
      name: p.name,
      available: p.available,
      models: p.models,
      defaultModel: p.default_model,
      local: p.local,
    }));
    caps.llm.configured = caps.llm.providers.some(p => p.available);
  } catch (err: any) {
    console.warn(`[Training] Provider probe failed: ${err.message}`);
  }

  // ── Preprocess (phase 2). Every probe degrades independently; the model
  // lists come from a CACHED /props snapshot so the picker survives the engine
  // being stopped by a running preprocess job (P28).
  try {
    const exe = aceTrainExe();
    caps.preprocess.available = !!exe;
    caps.preprocess.binPath = exe ?? '';
  } catch { /* stays unavailable */ }
  try { caps.preprocess.engineSuspended = isEngineSuspended(); } catch { /* stays false */ }
  try {
    const snap = caps.engine.up ? await refreshModelSnapshot() : getModelSnapshot();
    caps.preprocess.ditModels = snap.dit;
    caps.preprocess.vaeModels = snap.vae;
    caps.preprocess.textEncoders = snap.textEnc;
    caps.preprocess.defaultDit = pickBf16(snap.dit);
    caps.preprocess.defaultVae = pickBf16(snap.vae) || snap.vae[0] || '';
    caps.preprocess.defaultTextEnc = snap.textEnc[0] || '';
    caps.preprocess.modelsCachedAt = snap.cachedAt;
  } catch { /* stays empty */ }

  // ── LM trainer (phase 3). Same rules: every probe individually caught, the
  // model list comes from the cached snapshot so the picker survives the engine
  // being stopped by a running training job.
  try { caps.trainLm.available = !!aceTrainExe(); } catch { /* stays false */ }
  try { caps.trainLm.adaptersRoot = adapterLmRoot(); } catch { /* stays '' */ }
  try {
    const snap = getModelSnapshot();
    caps.trainLm.lmModels = snap.lm;
    caps.trainLm.defaultLmBySize = {
      '0.6B': pickLmFor('0.6B', snap.lm),
      '1.7B': pickLmFor('1.7B', snap.lm),
      '4B': pickLmFor('4B', snap.lm),
    };
  } catch { /* stays empty */ }

  // ── DiT trainer (phase 4). No model list: the base is forced to the one the
  // chosen preprocess variant was made against (§4.2 base-match guard), so
  // there is nothing for the UI to pick. `minVramMb` is ADVISORY — the real
  // gate is ace-train's own footprint solve, which can only run once the base
  // is loaded and the engine is already down (§4.5). Since 2026-09-02 that
  // solve is the ONLY gate: the engine's flat 16 GB card refusal is retired,
  // so this number dropped to 8192 and must never be used to block a run.
  try { caps.trainDit.available = !!aceTrainExe(); } catch { /* stays false */ }
  try { caps.trainDit.adaptersRoot = adapterDitRoot(); } catch { /* stays '' */ }

  res.json(caps);
});

// ── Scan preview (§2.2) ──────────────────────────────────────────────────

router.get('/scan-preview', (req: Request, res: Response) => {
  try {
    const raw = (req.query.path as string) || '';
    if (!raw.trim()) {
      res.status(400).json({ error: 'path is required' });
      return;
    }
    let root: string;
    try { root = path.resolve(raw); } catch {
      res.status(400).json({ error: 'Invalid path' });
      return;
    }
    if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) {
      res.status(404).json({ error: 'Directory not found' });
      return;
    }
    const recursive = req.query.recursive !== '0' && req.query.recursive !== 'false';
    res.json(scanPreviewFolder(root, recursive));
  } catch (err: any) {
    if (err instanceof ScanLimitError) {
      res.status(400).json({ error: err.message });
      return;
    }
    console.error(`[Training] scan-preview failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Dataset list / create (§2.3) ─────────────────────────────────────────

router.get('/datasets', async (_req: Request, res: Response) => {
  try {
    // Every row carries what it has ON DISK (built / tensors / LM / DiT adapter)
    // plus its detected album name, so the list and the batch wizard can show
    // pipeline progress without opening each dataset — datasetAssets.ts.
    res.json({ datasets: await listDatasetsWithAssets() });
  } catch (err: any) {
    console.error(`[Training] List datasets failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.post('/datasets', async (req: Request, res: Response) => {
  try {
    // The whole creation block lives in the service so the batch pipeline can
    // create datasets without going through HTTP; DatasetCreateError carries the
    // status this handler used to send inline.
    const detail = await createDatasetFromFolder((req.body || {}) as CreateDatasetInput);
    res.status(201).json({ dataset: detail });
  } catch (err: any) {
    if (err instanceof DatasetCreateError) {
      res.status(err.status).json({ error: err.message });
      return;
    }
    if (err instanceof ScanLimitError) {
      res.status(400).json({ error: err.message });
      return;
    }
    console.error(`[Training] Create dataset failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Job control (§2.8, §2.9) — declared before /datasets/:id ──────────────

router.get('/jobs', (req: Request, res: Response) => {
  try {
    const datasetId = typeof req.query.datasetId === 'string' ? req.query.datasetId : undefined;
    res.json({ jobs: queue.listJobs(datasetId) });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/jobs/:jobId', (req: Request, res: Response) => {
  try {
    const job = queue.getJob(req.params.jobId as string);
    if (job) {
      res.json(queue.toSummary(job));
      return;
    }
    const finished = queue.listJobs().find(j => j.id === req.params.jobId);
    if (!finished) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }
    res.json(finished);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.delete('/jobs/:jobId', (req: Request, res: Response) => {
  try {
    const jobId = req.params.jobId as string;
    if (!queue.cancelJob(jobId)) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }
    res.json({ ok: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/jobs/:jobId/stream', (req: Request, res: Response) => {
  try {
    const job = queue.getJob(req.params.jobId as string);
    if (!job) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }
    queue.attachStream(job, res);
  } catch (err: any) {
    if (!res.headersSent) res.status(500).json({ error: err.message });
    else { try { res.end(); } catch { /* already closed */ } }
  }
});

// ── Codes-audition preview streaming (codes-preview §3.2) ────────────────
//
// Declared here, in the static-first-segment block, per [DS] §2 — `/previews`
// cannot collide with `/datasets/:id`, but the house rule is the house rule.
//
// The served path is BUILT SERVER-SIDE from two validated tokens (a uuid v4 and
// `base|adapter`) and re-checked with isInside(previewsRoot(), file). No
// client-supplied path ever reaches fs.
//
// Range handling is a verbatim clone of the sample-audio route below with
// sample.audioPath swapped for the validated preview path. Deliberately NOT
// refactored into a shared helper: the sample route is outside this plan's
// editable set and a shared helper would drag it in.
router.get('/previews/:previewId/:slot', (req: Request, res: Response) => {
  try {
    const previewId = req.params.previewId;
    const slot = req.params.slot;   // 'base' | 'adapter' | '<slot>-render'
    if (!isPreviewId(previewId) || !isPreviewFileKey(slot)) {
      res.status(404).json({ error: 'Preview not found' });
      return;
    }
    const file = resolvePreviewFile(previewId, slot);
    if (!file || !isInside(previewsRoot(), file) || !fs.existsSync(file)) {
      res.status(404).json({ error: 'Preview not found' });
      return;
    }

    const stat = fs.statSync(file);
    const mime = MIME_BY_EXT[path.extname(file).toLowerCase()] || 'application/octet-stream';
    res.setHeader('Content-Type', mime);
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Cache-Control', 'no-store');

    const range = req.headers.range;
    if (range) {
      const m = /bytes=(\d*)-(\d*)/.exec(range);
      if (m) {
        const start = m[1] ? parseInt(m[1], 10) : 0;
        const end = m[2] ? parseInt(m[2], 10) : stat.size - 1;
        if (Number.isFinite(start) && start < stat.size && end >= start) {
          const last = Math.min(end, stat.size - 1);
          res.status(206);
          res.setHeader('Content-Range', `bytes ${start}-${last}/${stat.size}`);
          res.setHeader('Content-Length', last - start + 1);
          fs.createReadStream(file, { start, end: last }).pipe(res);
          return;
        }
      }
    }

    res.setHeader('Content-Length', stat.size);
    fs.createReadStream(file).pipe(res);
  } catch (err: any) {
    console.error(`[Training] Preview stream failed: ${err.message}`);
    if (!res.headersSent) res.status(500).json({ error: err.message });
  }
});

// ── Batch pipeline (batch-pipeline §2.2) — declared before /datasets/:id ──

router.post('/pipeline', (req: Request, res: Response) => {
  try {
    const body = req.body || {};
    const rawFolders = Array.isArray(body.folders) ? body.folders : [];
    if (rawFolders.length === 0) {
      res.status(400).json({ error: 'folders is required' });
      return;
    }

    const folders: PipelineFolderSpec[] = [];
    for (const raw of rawFolders) {
      const dir = typeof raw?.sourceDir === 'string' ? raw.sourceDir.trim() : '';
      if (!dir) {
        res.status(400).json({ error: 'folders is required' });
        return;
      }
      // Checked up front for EVERY folder: a typo in the last row of a 20-album
      // batch should not surface an hour into the run.
      const sourceDir = path.resolve(dir);
      if (!fs.existsSync(sourceDir) || !fs.statSync(sourceDir).isDirectory()) {
        res.status(400).json({ error: `Folder does not exist: ${dir}` });
        return;
      }
      folders.push({
        sourceDir,
        name: typeof raw?.name === 'string' ? raw.name.trim() : undefined,
        customTag: typeof raw?.customTag === 'string' ? raw.customTag.trim() : undefined,
      });
    }

    let stages: PipelineStage[] = [...PIPELINE_STAGES];
    if (Array.isArray(body.stages)) {
      for (const stage of body.stages) {
        if (!PIPELINE_STAGES.includes(stage)) {
          res.status(400).json({ error: `Unknown stage: ${stage}` });
          return;
        }
      }
      // Canonical order, whatever order the client listed them in.
      stages = PIPELINE_STAGES.filter(s => body.stages.includes(s));
    }

    if (hasActivePipeline()) {
      res.status(409).json({ error: 'A pipeline is already running' });
      return;
    }

    // Per-run label-stage overrides (bulk re-caption etc.) — validated field
    // by field; anything absent falls through to the stored label defaults.
    let labelOptions: PipelineLabelOptions | undefined;
    if (body.labelOptions && typeof body.labelOptions === 'object') {
      const lo = body.labelOptions as Record<string, unknown>;
      labelOptions = {};
      for (const k of ['useEssentia', 'useGenius', 'useCaption'] as const) {
        if (typeof lo[k] === 'boolean') labelOptions[k] = lo[k] as boolean;
      }
      if (lo.scope === 'all' || lo.scope === 'unlabeled') labelOptions.scope = lo.scope;
      const policies = ['fill_missing', 'overwrite_caption', 'overwrite_lyrics', 'overwrite_all'];
      if (typeof lo.mergePolicy === 'string' && policies.includes(lo.mergePolicy)) {
        labelOptions.mergePolicy = lo.mergePolicy as PipelineLabelOptions['mergePolicy'];
      }
      if (Object.keys(labelOptions).length === 0) labelOptions = undefined;
    }

    res.status(202).json({ pipeline: startPipeline(folders, stages, labelOptions) });
  } catch (err: any) {
    console.error(`[Training] Pipeline start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/pipeline', (_req: Request, res: Response) => {
  try {
    res.json({ pipelines: listPipelines() });
  } catch (err: any) {
    console.error(`[Training] Pipeline list failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/pipeline/:id', (req: Request, res: Response) => {
  try {
    const pipeline = getPipeline(req.params.id as string);
    if (!pipeline) {
      res.status(404).json({ error: 'Pipeline not found' });
      return;
    }
    res.json(pipeline);
  } catch (err: any) {
    console.error(`[Training] Pipeline read failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.delete('/pipeline/:id', (req: Request, res: Response) => {
  try {
    if (!cancelPipeline(req.params.id as string)) {
      res.status(404).json({ error: 'Pipeline not found' });
      return;
    }
    res.json({ ok: true });
  } catch (err: any) {
    console.error(`[Training] Pipeline cancel failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// Pause/resume are boundary holds, not kills — the in-flight stage always
// finishes first (pipelineRunner.ts). Both are idempotent on an active
// pipeline; a finished/cancelled one answers 409.
router.post('/pipeline/:id/pause', (req: Request, res: Response) => {
  try {
    const result = pausePipeline(req.params.id as string);
    if (result === 'not_found') {
      res.status(404).json({ error: 'Pipeline not found' });
      return;
    }
    if (result === 'not_active') {
      res.status(409).json({ error: 'Pipeline is not running' });
      return;
    }
    res.json({ ok: true });
  } catch (err: any) {
    console.error(`[Training] Pipeline pause failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// Resume also revives ended/cancelled/restart-recovered pipelines: the runner
// rebuilds its loop from the persisted record, keeps everything already done,
// and re-runs the rest (failures retry). 409 only when another pipeline is
// already active.
router.post('/pipeline/:id/resume', (req: Request, res: Response) => {
  try {
    const result = resumePipeline(req.params.id as string);
    if (result === 'not_found') {
      res.status(404).json({ error: 'Pipeline not found' });
      return;
    }
    if (result === 'busy') {
      res.status(409).json({ error: 'A pipeline is already running' });
      return;
    }
    res.json({ ok: true });
  } catch (err: any) {
    console.error(`[Training] Pipeline resume failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Stage defaults (batch-pipeline §2.2) ─────────────────────────────────
//
// No deep validation on PUT: a section is an opaque bag of the stage route's
// own option fields, and that route re-validates every one of them when the
// pipeline POSTs it — which is the honest gate.

const DEFAULTS_SECTIONS: ReadonlyArray<keyof TrainingDefaults> =
  ['label', 'preprocess', 'trainLm', 'trainDit'];

router.get('/defaults', (_req: Request, res: Response) => {
  try {
    res.json(getTrainingDefaults());
  } catch (err: any) {
    console.error(`[Training] Defaults read failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

/**
 * Server-side mirror of the Models tab selection.
 *
 * The UI owns this state in localStorage; training needs it on the server,
 * because a bulk run has no UI in the loop and the engine (which knows its own
 * loaded DiT) is deliberately stopped while a training job runs.
 */
router.get('/active-models', (_req: Request, res: Response) => {
  res.json(getActiveModels());
});

router.put('/active-models', (req: Request, res: Response) => {
  try {
    const body = (req.body || {}) as Record<string, unknown>;
    res.json(setActiveModels({
      ditModel: typeof body.ditModel === 'string' ? body.ditModel : undefined,
      lmModel: typeof body.lmModel === 'string' ? body.lmModel : undefined,
      vaeModel: typeof body.vaeModel === 'string' ? body.vaeModel : undefined,
      textEncoder: typeof body.textEncoder === 'string' ? body.textEncoder : undefined,
    }));
  } catch (err: any) {
    console.error(`[Training] Active-models write failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.put('/defaults', (req: Request, res: Response) => {
  try {
    const body = req.body || {};
    const patch: Partial<TrainingDefaults> = {};
    for (const section of DEFAULTS_SECTIONS) {
      const value = body[section];
      if (value === undefined) continue;
      if (!value || typeof value !== 'object' || Array.isArray(value)) {
        res.status(400).json({ error: `${section} must be an object` });
        return;
      }
      patch[section] = value as Record<string, unknown>;
    }
    res.json(setTrainingDefaults(patch));
  } catch (err: any) {
    console.error(`[Training] Defaults write failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Dataset detail / patch / rescan / delete (§2.3) ───────────────────────

router.get('/datasets/:id', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const detail = await detailFor(ds);
    syncCounters(ds, detail.samples);
    res.json(detail);
  } catch (err: any) {
    if (err instanceof ScanLimitError) {
      res.status(400).json({ error: err.message });
      return;
    }
    console.error(`[Training] Dataset detail failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.patch('/datasets/:id', (req: Request, res: Response) => {
  try {
    const id = req.params.id as string;
    const ds = repo.getDataset(id);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const body = req.body || {};
    const patch: Partial<TrainingDatasetRow> = {};

    if (typeof body.name === 'string') {
      const name = body.name.trim();
      if (!name || name.length > 100) {
        res.status(400).json({ error: 'name is required' });
        return;
      }
      patch.name = name;
    }
    if (typeof body.customTag === 'string') patch.customTag = body.customTag.trim();
    if (typeof body.tagPosition === 'string') {
      if (!['prepend', 'append', 'replace'].includes(body.tagPosition)) {
        res.status(400).json({ error: 'Invalid tagPosition' });
        return;
      }
      patch.tagPosition = body.tagPosition;
    }
    if (body.genreRatio !== undefined) {
      const n = Number(body.genreRatio);
      if (!Number.isFinite(n)) {
        res.status(400).json({ error: 'genreRatio must be a number' });
        return;
      }
      patch.genreRatio = Math.min(100, Math.max(0, Math.trunc(n)));
    }
    if (typeof body.defaultArtist === 'string') patch.defaultArtist = body.defaultArtist;
    if (typeof body.defaultAlbum === 'string') patch.defaultAlbum = body.defaultAlbum;
    if (typeof body.defaultGenre === 'string') patch.defaultGenre = body.defaultGenre;
    if (typeof body.defaultLanguage === 'string') patch.defaultLanguage = body.defaultLanguage.trim().toLowerCase();
    if (typeof body.recursive === 'boolean') patch.recursive = body.recursive;

    // slug and sourceDir are immutable after creation.
    repo.updateDataset(id, patch);
    const updated = repo.getDataset(id);
    res.json({ dataset: updated });
  } catch (err: any) {
    console.error(`[Training] Patch dataset failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.post('/datasets/:id/rescan', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is running for this dataset' });
      return;
    }

    let detail = await detailFor(ds);

    // §7.4 — prune orphaned label records only after TWO consecutive misses;
    // a disconnected network drive must not wipe the user's edits.
    const now = Date.now();
    let pruned = false;
    for (const sample of detail.samples) {
      const rec = readLabel(ds.slug, sample.sampleId);
      if (!rec) continue;
      if (!sample.fileMissing) {
        if (rec.missingSince) patchLabel(ds.slug, sample.sampleId, { missingSince: null });
        continue;
      }
      if (!rec.missingSince) {
        patchLabel(ds.slug, sample.sampleId, { missingSince: now });
        continue;
      }
      // Second consecutive miss — the ghost row goes, and with it the Build block.
      deleteLabel(ds.slug, sample.sampleId);
      pruned = true;
    }
    if (pruned) detail = await detailFor(ds);

    syncCounters(ds, detail.samples);
    res.json(detail);
  } catch (err: any) {
    if (err instanceof ScanLimitError) {
      res.status(400).json({ error: err.message });
      return;
    }
    console.error(`[Training] Rescan failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.delete('/datasets/:id', (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is running for this dataset' });
      return;
    }
    // D20: the DB row and data/training/datasets/<slug>/ only. The user's audio,
    // sidecars and dataset.json are their files and are never touched.
    repo.deleteDataset(ds.id);
    deleteLabels(ds.slug);
    deleteDatasetPreviews(ds.id);
    console.log(`[Training] Deleted dataset ${ds.slug} (source folder untouched: ${ds.sourceDir})`);
    res.json({ ok: true });
  } catch (err: any) {
    console.error(`[Training] Delete dataset failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Sample edits (§2.4) ──────────────────────────────────────────────────

/** Apply a patch to one sample: sidecar fields to disk, `excluded` to the label. */
async function applySamplePatch(
  ds: TrainingDatasetRow,
  sample: TrainingSample,
  patch: PatchSampleInput,
): Promise<TrainingSample> {
  const meta = loadSidecarMetadata(sample.audioPath);
  const sources: Record<string, FieldSource> = {};
  let touchedSidecar = false;

  for (const [field, value] of Object.entries(patch)) {
    if (value === undefined) continue;
    if (field === 'excluded') continue;
    const key = SIDECAR_KEY_OF[field];
    if (!key) continue;

    if (value === null) meta[key] = '';
    else if (typeof value === 'boolean') meta[key] = value ? 'true' : 'false';
    else meta[key] = String(value);

    touchedSidecar = true;
    // §2.0 types `sources` over seven field names only — editing
    // isInstrumental/customTag/promptOverride must not inject keys outside it.
    if (SOURCE_TRACKED_FIELDS.has(key)) sources[key] = 'user';
  }

  if (touchedSidecar) await writeSidecar(sample.sidecarPath, meta);

  patchLabel(ds.slug, sample.sampleId, {
    relPath: sample.relPath,
    sources,
    ...(typeof patch.excluded === 'boolean' ? { excluded: patch.excluded } : {}),
  });

  return reloadSample(ds, sample);
}

router.post('/datasets/:id/samples/bulk', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const body = req.body || {};
    const sampleIds: string[] = Array.isArray(body.sampleIds) ? body.sampleIds.filter((s: unknown) => typeof s === 'string') : [];
    const set = (body.set || {}) as BulkSetInput;
    if (sampleIds.length === 0) {
      res.status(400).json({ error: 'sampleIds is required' });
      return;
    }
    if (Object.keys(set).length === 0) {
      res.status(400).json({ error: 'set is empty' });
      return;
    }

    const samples = await buildSamples(ds);
    const byId = new Map(samples.map(s => [s.sampleId, s]));
    const failed: Array<{ sampleId: string; error: string }> = [];
    let updated = 0;

    for (const sampleId of sampleIds) {
      const sample = byId.get(sampleId);
      if (!sample) {
        failed.push({ sampleId, error: 'Sample not found' });
        continue;
      }
      if (sample.fileMissing && Object.keys(set).some(k => k !== 'excluded')) {
        failed.push({ sampleId, error: 'Audio file is missing from disk' });
        continue;
      }
      try {
        await applySamplePatch(ds, sample, set as PatchSampleInput);
        updated++;
      } catch (err: any) {
        failed.push({ sampleId, error: err.message });
      }
    }

    res.json({ updated, failed });
  } catch (err: any) {
    console.error(`[Training] Bulk edit failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.patch('/datasets/:id/samples/:sampleId', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const body = (req.body || {}) as PatchSampleInput;
    const editable = Object.keys(body).filter(
      k => (k in SIDECAR_KEY_OF || k === 'excluded') && (body as Record<string, unknown>)[k] !== undefined,
    );
    if (editable.length === 0) {
      res.status(400).json({ error: 'No editable fields in body' });
      return;
    }
    if (body.repeat !== undefined && (!Number.isFinite(Number(body.repeat)) || Number(body.repeat) < 1)) {
      res.status(400).json({ error: 'repeat must be >= 1' });
      return;
    }

    const samples = await buildSamples(ds);
    const sample = samples.find(s => s.sampleId === req.params.sampleId);
    if (!sample) {
      res.status(404).json({ error: 'Sample not found' });
      return;
    }
    if (sample.fileMissing) {
      res.status(409).json({ error: 'Audio file is missing from disk' });
      return;
    }

    const updated = await applySamplePatch(ds, sample, body);
    res.json({ sample: updated });
  } catch (err: any) {
    console.error(`[Training] Sample patch failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── MM3 Structured Caption (§2.4a) ───────────────────────────────────────
//
// Lives in `<stem>.mm3.txt` beside the audio (what `ace-train mm3-condition`
// reads), NOT in the sidecar — so it cannot ride the PATCH whitelist above.
// Served on demand rather than embedded in the samples list: at ~3 KB a
// caption it would double the list payload for a field only the drawer shows.

/** `<stem>.mm3.txt` for a sample, mirroring enhanceService's write path. */
function mm3PathOf(sample: TrainingSample): string {
  return `${sample.audioPath.replace(/\.[^.\\/]+$/, '')}.mm3.txt`;
}

router.get('/datasets/:id/samples/:sampleId/mm3', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const samples = await buildSamples(ds);
    const sample = samples.find(s => s.sampleId === req.params.sampleId);
    if (!sample) {
      res.status(404).json({ error: 'Sample not found' });
      return;
    }
    const mm3Path = mm3PathOf(sample);
    // Absent file is `text: ''`, not a 404 — "no MM3 caption yet" is a normal
    // state for a dataset labeled before MOSS existed.
    const text = fs.existsSync(mm3Path) ? fs.readFileSync(mm3Path, 'utf8') : '';
    res.json({ text });
  } catch (err: any) {
    console.error(`[Training] MM3 read failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.put('/datasets/:id/samples/:sampleId/mm3', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const text = String((req.body || {}).text ?? '');
    const samples = await buildSamples(ds);
    const sample = samples.find(s => s.sampleId === req.params.sampleId);
    if (!sample) {
      res.status(404).json({ error: 'Sample not found' });
      return;
    }
    if (sample.fileMissing) {
      res.status(409).json({ error: 'Audio file is missing from disk' });
      return;
    }
    const mm3Path = mm3PathOf(sample);
    if (text.trim()) {
      fs.writeFileSync(mm3Path, text, 'utf8');
    } else if (fs.existsSync(mm3Path)) {
      // Clearing the editor deletes the file — an empty .mm3.txt would still
      // be picked up by `ace-train mm3-condition` and condition on nothing.
      fs.unlinkSync(mm3Path);
    }
    res.json({ text });
  } catch (err: any) {
    console.error(`[Training] MM3 write failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/datasets/:id/samples/:sampleId/audio', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Sample not found' });
      return;
    }
    const samples = await buildSamples(ds);
    const sample = samples.find(s => s.sampleId === req.params.sampleId);
    // The path always comes from our own scan, never from the client (§7.8).
    if (!sample || !isInside(ds.sourceDir, sample.audioPath)) {
      res.status(404).json({ error: 'Sample not found' });
      return;
    }
    if (!fs.existsSync(sample.audioPath)) {
      res.status(404).json({ error: 'Audio file is missing from disk' });
      return;
    }

    const stat = fs.statSync(sample.audioPath);
    const mime = MIME_BY_EXT[path.extname(sample.audioPath).toLowerCase()] || 'application/octet-stream';
    res.setHeader('Content-Type', mime);
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Cache-Control', 'no-store');

    const range = req.headers.range;
    if (range) {
      const m = /bytes=(\d*)-(\d*)/.exec(range);
      if (m) {
        const start = m[1] ? parseInt(m[1], 10) : 0;
        const end = m[2] ? parseInt(m[2], 10) : stat.size - 1;
        if (Number.isFinite(start) && start < stat.size && end >= start) {
          const last = Math.min(end, stat.size - 1);
          res.status(206);
          res.setHeader('Content-Range', `bytes ${start}-${last}/${stat.size}`);
          res.setHeader('Content-Length', last - start + 1);
          fs.createReadStream(sample.audioPath, { start, end: last }).pipe(res);
          return;
        }
      }
    }

    res.setHeader('Content-Length', stat.size);
    fs.createReadStream(sample.audioPath).pipe(res);
  } catch (err: any) {
    console.error(`[Training] Audio stream failed: ${err.message}`);
    if (!res.headersSent) res.status(500).json({ error: err.message });
  }
});

// ── Labeling job (§2.5) ──────────────────────────────────────────────────

/** Resolve the sample ids a job should work on. */
function pickTargets(
  samples: TrainingSample[],
  sampleIds: unknown,
  scope: 'all' | 'unlabeled',
  extraFilter?: (s: TrainingSample) => boolean,
): string[] {
  if (Array.isArray(sampleIds) && sampleIds.length > 0) {
    const wanted = new Set(sampleIds.filter((s): s is string => typeof s === 'string'));
    return samples.filter(s => wanted.has(s.sampleId) && !s.fileMissing).map(s => s.sampleId);
  }
  return samples
    .filter(s => !s.excluded && !s.fileMissing)
    .filter(s => (scope === 'all' ? true : !s.caption.trim()))
    .filter(s => (extraFilter ? extraFilter(s) : true))
    .map(s => s.sampleId);
}

router.post('/datasets/:id/label', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }

    const body = (req.body || {}) as LabelOptions;
    const useEssentia = body.useEssentia !== false;
    // 2026-07-27 pivot: understand is LEGACY and opt-in; the default flow is
    // Essentia + Genius + LLM caption, all engine-free.
    //
    // Genius and caption DEFAULT ON (2026-07-30), matching LabelPanel, which has
    // shipped with all three boxes ticked. They were `=== true` — default OFF —
    // so the batch pipeline, which POSTs only the stored per-stage defaults
    // (`{}` in practice), silently ran Essentia-only and produced datasets with
    // no lyrics and no captions. Same class of bug as the train-dit adapterType
    // and train-lm lmSize divergences: the form's numbers and this handler's
    // numbers are two different sets reached by two different callers.
    const useUnderstand = body.useUnderstand === true;
    let useGenius = body.useGenius !== false;
    let useCaption = body.useCaption !== false;

    // Capability handling has to distinguish ASKED-FOR from DEFAULTED-ON, now
    // that these default on. An explicit `useGenius: true` with no token is a
    // request the server cannot honour and must refuse. A default-on step with
    // no token is just a machine that does not have that capability — skip it,
    // exactly as LabelPanel does by disabling the checkbox (effectiveGenius =
    // useGenius && geniusOk). Otherwise every keyless install would 503 on a
    // plain label call that never mentioned Genius.
    const geniusAsked  = body.useGenius === true;
    const captionAsked = body.useCaption === true;

    if (useGenius && !config.lireek.geniusAccessToken) {
      if (geniusAsked) {
        res.status(503).json({ error: 'GENIUS_ACCESS_TOKEN is not set' });
        return;
      }
      useGenius = false;
    }
    if (useCaption && (body.caption?.provider === 'moss')) {
      // MOSS is a local binary, not a chat API — getProvider() throws on it, and
      // its availability is "built + weights present", not "API key set". Handled
      // before the LLM branch so labeling can run with no cloud credentials at
      // all, which is the entire point of the local captioner.
      const probe = resolveMossPaths();
      if ('missing' in probe) {
        if (captionAsked) {
          res.status(503).json({ error: `MOSS captioning is not available: ${probe.missing}` });
          return;
        }
        useCaption = false;
      }
    } else if (useCaption) {
      const providerName = body.caption?.provider || config.lireek.defaultProvider;
      let provider;
      try {
        provider = getProvider(providerName);
      } catch (err: any) {
        if (captionAsked) {
          res.status(400).json({ error: err.message });
          return;
        }
        provider = null;
      }
      if (provider && !provider.isAvailable()) {
        if (captionAsked) {
          res.status(503).json({
            error: `Provider ${providerName} is not available. Check API keys in Settings → AI Services.`,
          });
          return;
        }
        provider = null;
      }
      if (!provider) {
        useCaption = false;
      }
    }

    // Checked AFTER the capability downgrades above, or a keyless machine would
    // pass this test on steps that are about to be switched off.
    if (!useEssentia && !useUnderstand && !useGenius && !useCaption) {
      res.status(400).json({ error: 'No labeling steps enabled' });
      return;
    }

    if (useUnderstand) {
      let ready: { ok: boolean; missing: string[]; lmModels: string[] };
      try {
        ready = await engineUnderstandReady();
      } catch {
        res.status(503).json({ error: 'Engine is not running' });
        return;
      }
      if (!ready.ok) {
        res.status(503).json({ error: `Engine is missing models: ${ready.missing.join(', ')}` });
        return;
      }
      // Without an explicit model the engine falls back to whatever is loaded,
      // else the alphabetically-first registry entry — the 0.6B. Always pin the
      // best LM unless the caller chose one.
      if (!body.understand?.lmModel) {
        const best = pickBestLm(ready.lmModels);
        if (best) body.understand = { ...body.understand, lmModel: best };
      }
    }

    const samples = await buildSamples(ds);
    const scope = body.scope === 'all' ? 'all' : 'unlabeled';
    const targets = pickTargets(samples, body.sampleIds, scope);
    if (targets.length === 0) {
      // A dataset that is ALREADY fully labelled is not an error — it is the
      // desired end state. Answering 400 made the bulk pipeline treat it as a
      // stage failure and abandon the whole item, so a folder that arrived with
      // sidecars already written got no preprocess, no LM train and no DiT
      // train (2026-07-31). `allowEmpty` lets a caller that can act on "nothing
      // to do" say so; the UI keeps the 400, where it is a useful toast.
      if (body.allowEmpty === true) {
        res.status(200).json({ jobId: null, skipped: 'nothing-to-label' });
        return;
      }
      res.status(400).json({ error: 'Nothing to label' });
      return;
    }

    const job = queue.startLabelJob(ds.id, targets, {
      ...body,
      useEssentia,
      useUnderstand,
      useGenius,
      useCaption,
      mergePolicy: body.mergePolicy || 'fill_missing',
    });
    repo.updateDataset(ds.id, { status: 'labeling' });
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] Label start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Enhance jobs (§2.6) ──────────────────────────────────────────────────

router.post('/datasets/:id/enhance/genius', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }
    if (!config.lireek.geniusAccessToken) {
      res.status(503).json({ error: 'GENIUS_ACCESS_TOKEN is not set' });
      return;
    }

    const body = (req.body || {}) as GeniusOptions;
    const samples = await buildSamples(ds);
    const targets = pickTargets(samples, body.sampleIds, 'all', s => !s.isInstrumental);
    if (targets.length === 0) {
      res.status(400).json({ error: 'Nothing to enhance' });
      return;
    }

    const job = queue.startGeniusJob(ds.id, targets, {
      ...body,
      mergePolicy: body.mergePolicy || 'overwrite_lyrics',
      sanitizeHeaders: body.sanitizeHeaders !== false,
    });
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] Genius start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.post('/datasets/:id/enhance/caption', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }

    const body = (req.body || {}) as CaptionOptions;
    const providerName = body.provider || config.lireek.defaultProvider;
    // MOSS is not in the LLM registry — it is a local binary, not a chat API — so
    // it must be admitted BEFORE getProvider(), which throws on an unknown id.
    // Its availability check is "binary built + weights on disk", and the failure
    // message has to say which, since neither is fixed in Settings → AI Services.
    if (providerName === 'moss') {
      const probe = resolveMossPaths();
      if ('missing' in probe) {
        res.status(503).json({ error: `MOSS captioning is not available: ${probe.missing}` });
        return;
      }
    } else {
      let provider;
      try {
        provider = getProvider(providerName);
      } catch (err: any) {
        res.status(400).json({ error: err.message });
        return;
      }
      if (!provider.isAvailable()) {
        res.status(503).json({
          error: `Provider ${providerName} is not available. Check API keys in Settings → AI Services.`,
        });
        return;
      }
    }

    const samples = await buildSamples(ds);
    const targets = pickTargets(samples, body.sampleIds, 'all');
    if (targets.length === 0) {
      res.status(400).json({ error: 'Nothing to enhance' });
      return;
    }

    const job = queue.startCaptionJob(ds.id, targets, {
      ...body,
      provider: providerName,
      mergePolicy: body.mergePolicy || 'overwrite_caption',
      includeLyricsExcerpt: body.includeLyricsExcerpt !== false,
      temperature: typeof body.temperature === 'number' ? body.temperature : 0.45,
    });
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] Caption start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Build (§2.7) ─────────────────────────────────────────────────────────

router.post('/datasets/:id/build', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }

    const raw = typeof req.body?.outputPath === 'string' ? req.body.outputPath.trim() : '';
    let outputPath = path.join(ds.sourceDir, 'dataset.json');
    if (raw) {
      const resolved = path.resolve(raw);
      if (!isInside(ds.sourceDir, resolved) && !isInside(trainingBaseDir, resolved)) {
        res.status(400).json({ error: 'outputPath must be inside the dataset folder' });
        return;
      }
      outputPath = resolved;
    }

    const samples = await buildSamples(ds);
    if (samples.filter(s => !s.excluded).length === 0) {
      res.status(400).json({ error: 'Dataset has no includable samples' });
      return;
    }

    const job = queue.startBuildJob(ds.id, { outputPath });
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] Build start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/datasets/:id/dataset-json', (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const target = ds.datasetJsonPath || path.join(ds.sourceDir, 'dataset.json');
    if (!ds.datasetJsonPath || !fs.existsSync(target)) {
      res.status(404).json({ error: 'dataset.json has not been built yet' });
      return;
    }
    const dataset = JSON.parse(fs.readFileSync(target, 'utf-8')) as unknown;
    res.json({ path: target, builtAt: ds.builtAt, dataset });
  } catch (err: any) {
    console.error(`[Training] dataset-json read failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Lyric Studio export ──────────────────────────────────────────────────

router.get('/datasets/:id/lyric-studio', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const samples = await buildSamples(ds, { warnings: [] });
    res.json(await previewLyricStudioExport(ds, samples));
  } catch (err: any) {
    if (err instanceof ScanLimitError) {
      res.status(400).json({ error: err.message });
      return;
    }
    console.error(`[Training] Lyric Studio preview failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.post('/datasets/:id/lyric-studio', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const body = (req.body ?? {}) as LyricStudioExportInput;
    const input: LyricStudioExportInput = {
      artist: typeof body.artist === 'string' ? body.artist.slice(0, 200) : undefined,
      album: typeof body.album === 'string' ? body.album.slice(0, 200) : undefined,
      linkAdapters: body.linkAdapters !== false,
    };
    const samples = await buildSamples(ds, { warnings: [] });
    const result = await commitLyricStudioExport(ds, samples, input);
    res.json(result);
  } catch (err: any) {
    if (err instanceof LyricStudioExportError || err instanceof ScanLimitError) {
      res.status(400).json({ error: err.message });
      return;
    }
    console.error(`[Training] Lyric Studio export failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Preprocess (§2.8, phase 2) ───────────────────────────────────────────

/** Clamp a numeric option to its default when the client omitted it. */
function numOpt(value: unknown, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

router.post('/datasets/:id/preprocess', async (req: Request, res: Response) => {
  try {
    let ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }
    // The Build step is what produces dataset.json — and with it the stable
    // sample ids the tensor cache filenames are keyed on.
    // A folder that already carries a dataset.json is built — the DB just may
    // never have been told (a fresh row for a previously-built folder, or a run
    // that skipped the Build stage). Believe the file before refusing.
    ds = adoptExistingDatasetJson(ds, repo.updateDataset);
    if (!ds.builtAt || !ds.datasetJsonPath || !fs.existsSync(ds.datasetJsonPath)) {
      res.status(400).json({ error: 'Dataset must be built first — run Build before Preprocess' });
      return;
    }
    if (!aceTrainExe()) {
      res.status(503).json({ error: 'ace-train was not found next to ace-server — rebuild the engine' });
      return;
    }

    const body = (req.body || {}) as PreprocessOptions;

    const samples = await buildSamples(ds);
    const wanted = Array.isArray(body.sampleIds)
      ? new Set(body.sampleIds.filter((s): s is string => typeof s === 'string'))
      : null;
    const targets = samples
      .filter(s => !s.excluded && !s.fileMissing)
      .filter(s => (wanted ? wanted.has(s.sampleId) : true))
      .map(s => s.sampleId);
    if (targets.length === 0) {
      res.status(400).json({ error: 'Dataset has no includable samples' });
      return;
    }

    // Models come from the CACHED snapshot — the engine may be stopped by a job
    // on another dataset. An empty snapshot means /props was never read, in
    // which case an explicit name is passed straight through to ace-train.
    // A never-probed snapshot (fresh boot, or a /capabilities read that raced the
    // engine coming up) would otherwise 400 with 'No DiT base model available'
    // even though the engine is reachable — probe once instead of rejecting.
    let snap = getModelSnapshot();
    if (!snap.cachedAt && !isEngineSuspended()) snap = await refreshModelSnapshot();
    // The DiT chosen HERE is the one the adapter ends up bound to: train-dit
    // reads --dit back out of preprocess_meta.json. Falling back to
    // pickBf16(snap.dit) — the first BF16 in the catalogue, i.e. the stock base
    // — is what made an 18-dataset overnight bulk run train against the base
    // instead of the user's fine-tune (2026-07-31). Prefer what they actually
    // have selected in the Models tab, mirrored server-side by
    // PUT /api/training/active-models, and fall back to the old behaviour only
    // when nothing is recorded.
    const dit = (typeof body.ditModel === 'string' ? body.ditModel.trim() : '')
      || resolveTrainingDit(getActiveModels().ditModel, snap.dit, pickBf16(snap.dit));
    if (!dit) {
      res.status(400).json({ error: 'No DiT base model available' });
      return;
    }
    if (body.ditModel && snap.dit.length > 0 && !snap.dit.includes(dit)) {
      res.status(400).json({ error: `Unknown DiT model: ${dit}` });
      return;
    }
    const vae = (typeof body.vaeModel === 'string' ? body.vaeModel.trim() : '')
      || pickBf16(snap.vae) || snap.vae[0] || '';
    const textEnc = (typeof body.textEncoder === 'string' ? body.textEncoder.trim() : '')
      || snap.textEnc[0] || '';

    // 600 (Rob, 2026-08-29) — was 240, which truncated 34% of the corpus and
    // taught the LM a false mid-phrase song ending on every capped track (the
    // hard-cut latent gets an im_end appended at extraction). 600 s matches the
    // engine's own 10-minute generation ceiling; tracks longer than that are
    // rejected elsewhere anyway. Longer songs cost preprocess storage and LM
    // sequence length linearly — the trainer's max-len auto-fit absorbs it.
    const maxDuration = numOpt(body.maxDuration, 600);
    const vaeChunk = numOpt(body.vaeChunk, 384);
    const vaeOverlap = numOpt(body.vaeOverlap, 48);
    // 512 / 2048, raised from Side-Step's 256 / 512 on 2026-07-30. Measured on
    // 7 datasets: ALL 83 captions exceeded 256 (median ~352) and ~half the
    // lyrics exceeded 512 (max ~1685), so every adapter trained so far never saw
    // the tail of any caption. E4 pads encoder states to a dataset-wide enc_S,
    // so the cap is a SAFETY VALVE, not a cost driver — enc_S is set by the
    // longest actual song, not by this number. Ceiling is
    // DIT_REPEAT_BACK_MAX/n_kv_heads/B = 4096; worst case here is
    // 2048 + 1 + 512 = 2561. ace-train's own defaults stay 256/512 for
    // reference parity. Full record + rollback:
    // docs/plans/2026-07-30-conditioning-token-caps.md
    const maxCaptionTokens = numOpt(body.maxCaptionTokens, 512);
    const maxLyricTokens = numOpt(body.maxLyricTokens, 2048);
    const targetDb = numOpt(body.targetDb, -1.0);

    if (maxDuration < 0) {
      res.status(400).json({ error: 'maxDuration must be >= 0' });
      return;
    }
    if (vaeChunk < 64) {
      res.status(400).json({ error: 'vaeChunk must be >= 64' });
      return;
    }
    if (vaeOverlap < 0 || vaeOverlap >= vaeChunk) {
      res.status(400).json({ error: 'vaeOverlap must be >= 0 and less than vaeChunk' });
      return;
    }
    if (maxCaptionTokens < 16 || maxCaptionTokens > 4096) {
      res.status(400).json({ error: 'maxCaptionTokens must be between 16 and 4096' });
      return;
    }
    if (maxLyricTokens < 16 || maxLyricTokens > 4096) {
      res.status(400).json({ error: 'maxLyricTokens must be between 16 and 4096' });
      return;
    }
    if (targetDb < -60 || targetDb > 0) {
      res.status(400).json({ error: 'targetDb must be between -60 and 0' });
      return;
    }

    const variantKey = variantKeyFor(dit);
    let outputDir = tensorsDir(ds.slug, dit);
    if (typeof body.outputDir === 'string' && body.outputDir.trim()) {
      // Containment, same rule the sibling DELETE handler applies (§7.8). This
      // path is mkdir'd, ace-train creates <out>/.tmp/ in it and deletes orphan
      // *.__writing__ files there, so an unchecked absolute path from the
      // request body is a write primitive. Staying under the dataset's tensors
      // root also keeps the cache visible to GET/DELETE .../preprocess — a
      // cache written anywhere else is unmanaged disk the UI can never see.
      const root = tensorsRoot(ds.slug);
      const resolved = path.resolve(body.outputDir.trim());
      if (!isInside(root, resolved) || path.resolve(root) === resolved) {
        res.status(400).json({ error: `outputDir must be a subdirectory of ${root}` });
        return;
      }
      outputDir = resolved;
    }

    const opts: ResolvedPreprocessOptions = {
      ditModel: dit,
      vaeModel: vae,
      textEncoder: textEnc,
      maxDuration: Math.trunc(maxDuration),
      normalize: (body.normalize === 'none' ? 'none' : 'peak') as PreprocessNormalize,
      targetDb,
      dtype: (body.dtype === 'bf16' ? 'bf16' : 'f32') as PreprocessDtype,
      compat: (body.compat === 'sidestep' ? 'sidestep' : 'hotstep') as PreprocessCompat,
      maxCaptionTokens: Math.trunc(maxCaptionTokens),
      maxLyricTokens: Math.trunc(maxLyricTokens),
      vaeChunk: Math.trunc(vaeChunk),
      vaeOverlap: Math.trunc(vaeOverlap),
      overwrite: body.overwrite === true,
      stopEngine: body.stopEngine !== false,
      outputDir,
      variantKey,
    };

    const job = queue.startPreprocessJob(ds.id, targets, opts);
    console.log(`[Training] Preprocess job ${job.id} queued — ${targets.length} songs, variant ${variantKey}`);
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] Preprocess start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/datasets/:id/preprocess', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    res.json(readPreprocessStatus(ds, await buildSamples(ds)));
  } catch (err: any) {
    console.error(`[Training] Preprocess status failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.delete('/datasets/:id/preprocess/:variantKey', (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is running for this dataset' });
      return;
    }
    const root = tensorsRoot(ds.slug);
    const dir = path.join(root, path.basename(String(req.params.variantKey ?? '')));
    // §7.8 — the client-supplied key never escapes the dataset's tensors root.
    if (!isInside(root, dir) || root === path.resolve(dir) || !fs.existsSync(dir)) {
      res.status(404).json({ error: 'Variant not found' });
      return;
    }
    fs.rmSync(dir, { recursive: true, force: true });
    console.log(`[Training] Deleted tensor cache ${dir}`);
    res.json({ ok: true });
  } catch (err: any) {
    console.error(`[Training] Preprocess delete failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── LM LoRA training (§2.8, phase 3) ─────────────────────────────────────

const TRAIN_LM_STAGES: readonly TrainLmStage[] = ['extract', 'train', 'export'];
const ADAPTER_NAME_RE = /^[A-Za-z0-9._-]{1,64}$/;

/** `null` when in range, else the §2.8-shaped 400 message. */
function outOfRange(name: string, n: number, min: number, max: number): string | null {
  if (!Number.isFinite(n) || n < min || n > max) return `${name} must be between ${min} and ${max}`;
  return null;
}

// ─── MiniMax-Music3 training (S4) ───────────────────────────────────────────
//
// Two stages, both GPU-lane and both engine-stopping:
//   POST /datasets/:id/mm3-codes     audio -> RVQ codes  (the LM's input)
//   POST /datasets/:id/mm3-train-lm  codes + captions -> an LM LoRA
//   GET  /datasets/:id/mm3           what exists and what is missing
//
// Spec: docs/plans/2026-08-20-mm3-training-server-design.md §2.4.

/** Shared guards: dataset exists, nothing else is running on it, ace-train is
 *  in the build, and the dataset has actually been built. Returns the dataset
 *  or null after answering. */
function mm3Preflight(req: Request, res: Response): TrainingDatasetRow | null {
  let ds = repo.getDataset(req.params.id as string);
  if (!ds) {
    res.status(404).json({ error: 'Dataset not found' });
    return null;
  }
  if (queue.activeJobForDataset(ds.id)) {
    res.status(409).json({ error: 'A job is already running for this dataset' });
    return null;
  }
  if (!aceTrainExe()) {
    res.status(503).json({ error: 'ace-train was not found next to ace-server — rebuild the engine' });
    return null;
  }
  ds = adoptExistingDatasetJson(ds, repo.updateDataset);
  if (!ds.builtAt || !ds.datasetJsonPath || !fs.existsSync(ds.datasetJsonPath)) {
    res.status(400).json({ error: 'Dataset must be built first — run Build before MiniMax-Music3 training' });
    return null;
  }
  return ds;
}

/** How many manifest rows have a `<stem>.mm3.txt` beside their audio. Mirrors
 *  the trainer's own rule (mm3-lm-train-run.h): stem = filename minus its last
 *  extension, caption = `<captionsDir>/<stem>.mm3.txt`. Never throws. */
function countMm3Captions(manifest: string, captionsDir: string): { total: number; captioned: number } {
  let rows: unknown[] = [];
  try {
    const parsed = JSON.parse(fs.readFileSync(manifest, 'utf-8'));
    rows = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.samples) ? parsed.samples : [];
  } catch {
    return { total: 0, captioned: 0 };
  }
  let total = 0, captioned = 0;
  for (const raw of rows) {
    const filename = (raw as { filename?: unknown })?.filename;
    if (typeof filename !== 'string' || !filename) continue;
    total++;
    const stem = filename.replace(/\.[^.]*$/, '');
    if (fs.existsSync(path.join(captionsDir, `${stem}.mm3.txt`))) captioned++;
  }
  return { total, captioned };
}

/** GET /datasets/:id/mm3 — codes cache state + which model files are missing.
 *  Cheap and never throws: the UI polls it to decide what to enable. */
router.get('/datasets/:id/mm3', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const codesDir = mm3CodesDir(ds.slug);
    let codes = 0;
    let encoder = '';
    try {
      const inner = path.join(codesDir, 'codes');
      codes = fs.readdirSync(inner).filter(f => f.endsWith('.codes')).length;
      const meta = JSON.parse(fs.readFileSync(path.join(inner, 'codes.json'), 'utf-8'));
      encoder = typeof meta?.encoder === 'string' ? path.basename(meta.encoder) : '';
    } catch { /* no cache yet */ }
    // The laundered cache is a sibling, never a replacement — both counts are
    // reported so the card can say which kinds exist.
    let codesLaundered = 0;
    try {
      const innerL = path.join(mm3CodesDir(ds.slug, true), 'codes');
      codesLaundered = fs.readdirSync(innerL).filter(f => f.endsWith('.codes')).length;
    } catch { /* no laundered cache yet */ }
    // The card, from the engine's own reading rather than a guess. Short
    // timeout and a 0 fallback: this endpoint is polled, and a picker that
    // stalls because ace-server is restarting is worse than one without a fit
    // estimate for a second.
    let gpuTotalMb = 0;
    try {
      const r = await fetch(`${config.aceServer.url}/vram`, { signal: AbortSignal.timeout(1500) });
      if (r.ok) {
        const v: any = await r.json();
        gpuTotalMb = Number(v?.total_mb) || 0;
      }
    } catch { /* engine down or CPU-only — leave it unknown */ }

    // The dataset-wide caption, so the form can show what is actually in force
    // rather than an empty box that silently means "use per-song captions".
    let sharedCaption = '';
    try {
      const p = path.join(path.dirname(ds.datasetJsonPath || ''), '_shared-caption.txt');
      if (ds.datasetJsonPath && fs.existsSync(p)) sharedCaption = fs.readFileSync(p, 'utf-8').trim();
    } catch { /* unreadable — treat as absent */ }

    res.json({
      codesDir,
      codes,
      sharedCaption,
      encoder,
      // Reported separately because the two stages need different files: the
      // codes job wants the encoders, training wants the F16 LM + depth.
      codesLaundered,
      missingForCodes: missingMm3TrainModels('codes'),
      missingForLaunder: missingMm3TrainModels('launder'),
      missingForTrain: missingMm3TrainModels('train'),
      // Only offer bases that are installed — a picker listing a file that is
      // not there just moves the failure to spawn time. Sized at the DEFAULT
      // recipe; the UI re-estimates locally as the user moves rank/frames.
      bases: availableMm3Bases(),
      // Datasets that could serve as a PRIOR-PRESERVATION corpus: anything with
      // RVQ codes that is not this one. Listed by the server because "has codes"
      // is a filesystem fact the browser cannot see, and offering a dataset
      // without them would fail at spawn time instead of at pick time.
      regCandidates: repo.listDatasets()
        .filter(d => d.id !== ds.id)
        .map(d => {
          const inner = path.join(mm3CodesDir(d.slug), 'codes');
          let songs = 0;
          try {
            songs = fs.readdirSync(inner).filter(f => f.endsWith('.codes')).length;
          } catch { /* no codes: filtered out below */ }
          return { id: d.id, name: d.name || d.slug, songs };
        })
        .filter(d => d.songs > 0),
      // What the card can actually hold. 0 when ace-server is not up, which the
      // UI must treat as "unknown" rather than "no VRAM" — the difference is a
      // greyed-out estimate versus a false "will not fit" on a 5090.
      gpuTotalMb,
      // Base AND rank: on a 16 GB card nothing fits at the default rank 256,
      // so recommending only a base would hand the user a red warning and no
      // way out of it.
      recommended: recommendMm3Config(gpuTotalMb),
      // The coefficients, not just the answer — the form re-estimates locally
      // as rank and crop length move, and must not carry its own copy.
      vramModel: MM3_VRAM_MODEL,
      defaults: MM3_LM_DEFAULTS,
    });
  } catch (err: any) {
    res.status(500).json({ error: err?.message || String(err) });
  }
});

/** POST /datasets/:id/mm3-codes */
router.post('/datasets/:id/mm3-codes', (req: Request, res: Response) => {
  try {
    const ds = mm3Preflight(req, res);
    if (!ds) return;
    const body = (req.body || {}) as { maxDuration?: unknown; launder?: unknown };
    // The launder gate. Off (absent/false) = plain mm3-codes into mm3-codes/,
    // byte-identical to the pipeline before the gate existed. On = ace-train
    // mm3-launder into mm3-codes-laundered/ — a SEPARATE cache, so the toggle
    // can never silently serve the other kind's codes.
    const launder = body.launder === true;
    const missing = missingMm3TrainModels(launder ? 'launder' : 'codes');
    if (missing.length) {
      res.status(400).json({ error: `MiniMax-Music3 encoder files are missing: ${missing.join(', ')}` });
      return;
    }
    const maxDuration = Number(body.maxDuration);
    const job = queue.startMm3CodesJob(ds.id, {
      datasetSlug: ds.slug,
      datasetJson: ds.datasetJsonPath,
      outDir: mm3CodesDir(ds.slug, launder),
      maxDuration: Number.isFinite(maxDuration) && maxDuration > 0 ? maxDuration : undefined,
      launder,
    });
    res.json({ jobId: job.id, kind: job.kind });
  } catch (err: any) {
    res.status(500).json({ error: err?.message || String(err) });
  }
});

/** Mid-run preview options off the request body, clamped.
 *
 *  Returns undefined for "off", which is what the runner checks. Both cadence
 *  fields at zero is off — a preview block with captions but no cadence would
 *  otherwise look configured and never fire. */
function parseMm3PreviewOptions(raw: unknown): Mm3PreviewOptions | undefined {
  if (!raw || typeof raw !== 'object') return undefined;
  const p = raw as Record<string, unknown>;
  const int = (k: string, d: number, lo: number, hi: number): number => {
    const v = Number(p[k]);
    return Number.isFinite(v) ? Math.min(hi, Math.max(lo, Math.trunc(v))) : d;
  };
  const everySteps = int('everySteps', 0, 0, 100000);
  const everyMinutes = int('everyMinutes', 0, 0, 1440);
  if (everySteps <= 0 && everyMinutes <= 0) return undefined;
  const str = (k: string): string | undefined =>
    typeof p[k] === 'string' ? (p[k] as string) : undefined;
  // undefined, not a default: the planner owns the default, and substituting one
  // here would mean two places to change it.
  const flt = (k: string): number | undefined => {
    const v = Number(p[k]);
    return Number.isFinite(v) ? Math.min(2, Math.max(0, v)) : undefined;
  };
  return {
    everySteps,
    everyMinutes,
    seconds: int('seconds', 40, 8, 120),
    seed: int('seed', 424242, 0, 2 ** 31 - 1),
    caption: str('caption'),
    lyrics: str('lyrics'),
    control: p.control !== false,
    controlCaption: str('controlCaption'),
    baseline: p.baseline !== false,
    scaleMlp: flt('scaleMlp'),
    scaleAttn: flt('scaleAttn'),
  };
}

/** Resolve the regularisation corpus off the request body.
 *
 *  Returns {} when prior preservation is off, and THROWS when it is on but the
 *  corpus cannot be used — a silently-dropped regularisation set would train a
 *  different objective than the one asked for and look identical in the logs. */
function resolveMm3Regularisation(raw: unknown, styleDatasetId: string): Partial<{
  regManifest: string; regCaptionsDir: string; regCodesDir: string;
  regPriorDir: string; regEvery: number; regTopK: number;
}> {
  if (!raw || typeof raw !== 'object') return {};
  const r = raw as Record<string, unknown>;
  const every = Number.isFinite(Number(r.every)) ? Math.trunc(Number(r.every)) : MM3_LM_DEFAULTS.regEvery;
  const id = typeof r.datasetId === 'string' ? r.datasetId : '';
  if (!id || every <= 0) return {};
  if (every < 2) {
    throw new Error('Regularisation cadence must be at least 2 — at 1 every step would be a '
                  + 'regularisation step and nothing would learn the artist.');
  }
  if (id === styleDatasetId) {
    throw new Error('The regularisation corpus must be a DIFFERENT dataset. Preserving the prior '
                  + 'over the same songs the adapter is learning cancels itself out.');
  }
  const ds = repo.getDataset(id);
  if (!ds) throw new Error('Regularisation dataset not found');
  const codes = path.join(mm3CodesDir(ds.slug), 'codes');
  const n = fs.existsSync(codes) ? fs.readdirSync(codes).filter(f => f.endsWith('.codes')).length : 0;
  if (n === 0) {
    throw new Error(`"${ds.name || ds.slug}" has no RVQ codes — run the codes export on it first. `
                  + 'A regularisation corpus needs exactly what a training corpus needs.');
  }
  const topK = Number.isFinite(Number(r.topK)) ? Math.trunc(Number(r.topK)) : MM3_LM_DEFAULTS.regTopK;
  return {
    regManifest:    ds.datasetJsonPath,
    regCaptionsDir: ds.sourceDir,
    regCodesDir:    codes,
    regPriorDir:    mm3PriorDir(ds.slug),
    regEvery:       every,
    regTopK:        Math.min(256, Math.max(1, topK)),
  };
}

/** GET /mm3/preview?run=<run>&file=<file> — one rendered preview WAV.
 *
 *  Served rather than base64'd down the SSE stream: a 24 s 16-bit stereo WAV is
 *  ~4 MB, and the stream is also carrying a step event every four seconds. */
router.get('/mm3/preview', (req: Request, res: Response) => {
  const run = String(req.query.run || '');
  const file = String(req.query.file || '');
  // Containment: both components reach us from the browser. Anything with a
  // separator or a dot-segment is refused outright rather than normalised.
  if (!run || !file || /[\\/]|\.\./.test(run) || /[\\/]|\.\./.test(file) || !file.endsWith('.wav')) {
    res.status(400).json({ error: 'bad preview reference' });
    return;
  }
  const full = path.join(mm3AdapterRunDir(run), 'previews', file);
  if (!fs.existsSync(full)) {
    res.status(404).json({ error: 'preview not found' });
    return;
  }
  res.setHeader('Content-Type', 'audio/wav');
  res.setHeader('Cache-Control', 'no-store');
  res.sendFile(full);
});

/** POST /datasets/:id/mm3-train-lm */
router.post('/datasets/:id/mm3-train-lm', (req: Request, res: Response) => {
  try {
    const ds = mm3Preflight(req, res);
    if (!ds) return;
    // Any installed base is trainable since the quant-cpy-kquant patch, so the
    // check is "is this file here", not "is this file f16". Anything unknown
    // falls back to the default rather than being passed to spawn.
    const installed = availableMm3Bases();
    const askedBase = String((req.body || {}).basePrecision || '');
    const chosenBase: Mm3BasePrecision =
      installed.some(b => b.id === askedBase) ? askedBase : MM3_LM_DEFAULTS.basePrecision;
    const missing = missingMm3TrainModels('train', chosenBase);
    if (missing.length) {
      res.status(400).json({
        error: `MiniMax-Music3 training models are missing: ${missing.join(', ')}. `
             + 'Install one from the Model Manager, or pick a base that is already present.',
      });
      return;
    }

    // The training half of the launder gate: opt-in per run, and the two code
    // caches are separate dirs, so an absent flag trains on exactly the codes
    // it always has.
    const trainLaunder = (req.body || {}).launder === true;
    const codesDir = mm3CodesDir(ds.slug, trainLaunder);
    const codesInner = path.join(codesDir, 'codes');
    const nCodes = fs.existsSync(codesInner)
      ? fs.readdirSync(codesInner).filter(f => f.endsWith('.codes')).length : 0;
    if (nCodes === 0) {
      res.status(400).json({ error: trainLaunder
        ? 'No laundered codes for this dataset — run the codes export with laundering enabled first'
        : 'No RVQ codes for this dataset — run the codes export first' });
      return;
    }

    // The captions the trainer reads are the MM3-native `<stem>.mm3.txt` files
    // beside the audio, NOT the ACE caption in the sidecar. The trainer refuses
    // to fall back (an ACE caption trains the wrong genre, measured), so point
    // it at the source folder and let it skip what has none.
    const captionsDir = ds.sourceDir;

    const b = (req.body || {}) as Record<string, unknown>;
    const num = (k: string, d: number, lo: number, hi: number): number => {
      const v = Number(b[k]);
      return Number.isFinite(v) && v >= lo && v <= hi ? v : d;
    };
    const D = MM3_LM_DEFAULTS;
    // Three-way now. The old two-way collapsed anything that was not 'adamw'
    // onto the default, which with a prodigy default would have silently
    // ignored a request for muon.
    const optimizer: 'muon' | 'adamw' | 'prodigy' =
      b.optimizer === 'adamw' || b.optimizer === 'muon' || b.optimizer === 'prodigy'
        ? b.optimizer
        : D.optimizer;
    const basePrecision: Mm3BasePrecision = chosenBase;
    // Enumerated, not a two-way test. The old form collapsed anything that was
    // not 'beginning' onto the default, so a request for 'structured' would have
    // been silently ignored — the same manual-whitelist shape that has already
    // produced a set of dead-looking knobs elsewhere in this file.
    const cropMode: 'random' | 'beginning' | 'structured' =
      b.cropMode === 'beginning' || b.cropMode === 'random' || b.cropMode === 'structured'
        ? b.cropMode
        : D.cropMode;
    const runName   = mm3RunName(ds.slug);
    // Per-track `<stem>.mm3.txt` captions (Enhance panel: MOSS or Gemini) are
    // the intended input. One shared caption for the whole album is the
    // FALLBACK for datasets without them. Convention: `_shared-caption.txt`
    // beside the dataset.
    const sharedCaptionPath = (() => {
      if (typeof b.captionFile === 'string' && b.captionFile.trim()) return b.captionFile.trim();
      const p = path.join(captionsDir, '_shared-caption.txt');
      // A caption submitted with the job WINS and is persisted, so the file the
      // engine reads, the preview renderer and any hand-run command all agree.
      const typed = typeof b.sharedCaption === 'string' ? b.sharedCaption.trim() : '';
      if (typed) {
        try {
          fs.writeFileSync(p, typed + '\n', 'utf-8');
        } catch (e: any) {
          throw new Error(`cannot write the shared caption to ${p}: ${e?.message || e}`);
        }
        return p;
      }
      return fs.existsSync(p) ? p : undefined;
    })();

    // The trainer skips every row without a `.mm3.txt` and, with no rows left,
    // exits 1 with nothing but SKIP lines in the log. A user read that as "the
    // file needs .mm3 in its name", renamed the ACE sidecars, and trained on
    // ACE captions. Count here and say what to do instead.
    if (!sharedCaptionPath) {
      const c = countMm3Captions(ds.datasetJsonPath, captionsDir);
      if (c.total > 0 && c.captioned === 0) {
        res.status(400).json({
          error: `None of the ${c.total} tracks has a MiniMax-Music3 caption (<stem>.mm3.txt beside `
               + 'the audio). Generate them in the Enhance panel with MOSS (local) or Gemini, '
               + 'both of which hear the audio. As a fallback, fill in the Dataset-wide caption '
               + 'under Advanced. Renaming ACE sidecar .txt files does not work: the trainer '
               + 'needs the MM3 Structured Caption format.',
        });
        return;
      }
      if (c.captioned < c.total) {
        console.warn(`[Training] mm3-train-lm: ${c.total - c.captioned} of ${c.total} tracks have no `
                   + '.mm3.txt and will be skipped by the trainer');
      }
    }

    const previewOpts = parseMm3PreviewOptions(b.preview);

    // Resolved before the job is built so a bad regularisation corpus is a 400
    // the user can act on, not a 500 from inside the queue.
    let reg: ReturnType<typeof resolveMm3Regularisation>;
    try {
      reg = resolveMm3Regularisation(b.regularisation, ds.id);
    } catch (err: any) {
      res.status(400).json({ error: err?.message || String(err) });
      return;
    }

    // The engine refuses `eval` targeting without a held-out set, and it does so
    // with exit code 2 five minutes into a model load. Catch it here, where it
    // is a 400 with the fix in it.
    if (b.stopMode === 'loss' && b.targetLossMetric === 'eval') {
      const hold = num('holdout', D.holdout, 0, 0.5);
      const ev   = num('evalEvery', D.evalEvery, 0, 100000);
      if (hold <= 0 || ev <= 0) {
        res.status(400).json({
          error: 'Targeting the held-out loss needs a hold-out fraction above 0 and evaluation '
               + 'switched on. Either raise both, or target the training loss instead.',
        });
        return;
      }
    }

    const job = queue.startMm3TrainLmJob(ds.id, {
      manifest:    ds.datasetJsonPath,
      captionsDir,
      codesDir:    codesInner,
      outDir:      mm3AdapterRunDir(runName),
      rank:        num('rank', D.rank, 1, 512),
      alpha:       num('alpha', D.alpha, 1, 2048),
      lr:          num('lr', D.lr, 1e-7, 1e-2),
      steps:       num('steps', D.steps, 1, 100000),
      saveEvery:   num('saveEvery', D.saveEvery, 0, 100000),
      warmup:      num('warmup', D.warmup, 0, 100000),
      gradAccum:   num('gradAccum', D.gradAccum, 1, 64),
      seed:        num('seed', D.seed, 0, 2 ** 31 - 1),
      maxFrames:   num('maxFrames', D.maxFrames, 64, 9000),
      cropMode,
      cropStartFrac: num('cropStartFrac', D.cropStartFrac, 0, 1),
      cropEndFrac:   num('cropEndFrac', D.cropEndFrac, 0, 1),
      cropStartTiles: num('cropStartTiles', D.cropStartTiles, 1, 64),
      depthLossWeight: num('depthLossWeight', D.depthLossWeight, 0, 10),
      depthLossFrames: num('depthLossFrames', D.depthLossFrames, 1, 1024),
      optimizer,
      muonLrScale: num('muonLrScale', D.muonLrScale, 0.01, 4096),
      basePrecision,
      holdout:     num('holdout', D.holdout, 0, 0.5),
      evalEvery:   num('evalEvery', D.evalEvery, 0, 100000),
      // Clamped to maxFrames: a larger eval crop than the training crop cannot
      // fit the sequence limit and every eval is silently skipped.
      evalCrop:    Math.min(num('evalCrop', D.evalCrop, 8, 9000),
                            num('maxFrames', D.maxFrames, 64, 9000)),
      rankDropout: num('rankDropout', D.rankDropout, 0, 0.9),
      captionFile: sharedCaptionPath,
      adapterType: b.adapterType === 'lora' || b.adapterType === 'lokr'
        ? b.adapterType : D.adapterType,
      lokrFactor:  num('lokrFactor', D.lokrFactor, 1, 64),
      lokrDim:     num('lokrDim', D.lokrDim, 1, 8192),
      lokrAlpha:   num('lokrAlpha', D.lokrAlpha, 1, 8192),
      trigger:     typeof b.trigger === 'string' ? b.trigger.trim() : (ds.customTag || ''),
      triggerPrepend: b.triggerPrepend !== false,
      datasetName: ds.name || ds.slug,
      // `song` is the default and the correct convention; `zero` exists only to
      // reproduce a pre-2026-08-23 run. See Mm3TrainLmRequest.cropAnchor.
      cropAnchor:  b.cropAnchor === 'zero' ? 'zero' : 'song',
      // 0 = off. The upper bound is the engine's own sequence ceiling; the
      // store is linear in this, so a long prefix is affordable in a way a
      // long crop is not.
      prefixFrames: num('prefixFrames', D.prefixFrames, 0, 9000),
      prefixChunk:  num('prefixChunk', D.prefixChunk, 32, 2048),
      prefixSelftest: b.prefixSelftest !== false,
      lrEndFrac:   num('lrEndFrac', D.lrEndFrac, 0, 1),
      // Stopping strategy. `steps` above is the cap in BOTH modes, which is the
      // whole reason it is still sent in loss mode: a target that never arrives
      // has to end somewhere.
      stopMode:    b.stopMode === 'loss' ? 'loss' : D.stopMode,
      targetLoss:  num('targetLoss', D.targetLoss, 0, 100),
      targetLossMetric: b.targetLossMetric === 'eval' ? 'eval' : 'train',
      targetLossEpochs: num('targetLossEpochs', D.targetLossEpochs, 1, 10000),
      ...reg,
      preview:     previewOpts,
    });
    res.json({ jobId: job.id, kind: job.kind, runName, outDir: mm3AdapterRunDir(runName) });
  } catch (err: any) {
    res.status(500).json({ error: err?.message || String(err) });
  }
});

// -- continuing a previous run ----------------------------------------------
//
//   GET  /datasets/:id/mm3-runs        what previous runs exist and their state
//   POST /datasets/:id/mm3-resume-lm   train more steps into one of them
//
// The engine has taken --resume since the preview loop was built; what was
// missing was any way to ask for it after the job that owned the run had ended.
// See services/training/mm3Runs.ts for what is read off disk and why.

/** GET /datasets/:id/mm3-runs */
router.get('/datasets/:id/mm3-runs', (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const runs = listMm3Runs(ds.id, ds.slug);
    // A run whose job is live right now is not "halted", whatever its log tail
    // says - the log tail of a running job always looks like an interrupted one.
    const active = queue.activeJobForDataset(ds.id);
    const activeDir = active && active.kind === 'mm3-train-lm'
      ? String((active.opts as { outDir?: string } | undefined)?.outDir || '') : '';
    const out = runs.map(r => (
      activeDir && path.resolve(activeDir) === path.resolve(r.dir)
        ? { ...r, outcome: 'unknown' as const, running: true }
        : { ...r, running: false }
    ));
    res.json({ runs: out, busy: !!active });
  } catch (err: any) {
    res.status(500).json({ error: err?.message || String(err) });
  }
});

/** POST /datasets/:id/mm3-resume-lm
 *
 *  Body: { runName, addSteps? | steps?, saveEvery?, stopMode?, targetLoss?,
 *          targetLossMetric?, targetLossEpochs?, preview? }
 *
 *  Everything else comes from the run's own manifest, deliberately. A resume
 *  that quietly re-derived rank, optimizer or the held-out split from today's
 *  form would be refused by the engine's fingerprint check at best, and at
 *  worst would train a different recipe under the same adapter name.
 */
router.post('/datasets/:id/mm3-resume-lm', (req: Request, res: Response) => {
  try {
    const ds = mm3Preflight(req, res);
    if (!ds) return;
    const b = (req.body || {}) as Record<string, unknown>;
    const dir = resolveMm3RunDir(String(b.runName || ''));
    if (!dir) {
      res.status(404).json({ error: 'No such training run' });
      return;
    }
    const run = readMm3Run(dir);
    if (!run) {
      res.status(404).json({ error: 'No such training run' });
      return;
    }
    if (!run.resume) {
      res.status(400).json({
        error: 'This run has no saved optimizer state, so there is nothing to continue from. A run '
             + 'is resumable from its last preview pause, or - for runs that finished after this '
             + 'feature shipped - from the step it ended on.',
      });
      return;
    }

    // The recipe: the run's own if it recorded one, otherwise reconstructed
    // from its log with today's defaults filling the gaps the log does not
    // carry. resumeOptionsFor returns null when neither exists.
    const base = resumeOptionsFor(dir);
    if (!base) {
      res.status(400).json({
        error: 'This run directory has neither a recorded recipe nor a training log, so there is '
             + 'no way to know what it was trained with.',
      });
      return;
    }
    // The DATA paths always come from the dataset as it is NOW, not from the
    // manifest: a dataset re-scanned since has a new manifest path, and
    // pointing at the old one would train on a stale song list. A changed song
    // COUNT is refused by the engine outright, which is the check that matters.
    const opts = {
      ...base,
      manifest:    ds.datasetJsonPath,
      captionsDir: ds.sourceDir,
      codesDir:    path.join(mm3CodesDir(ds.slug), 'codes'),
      outDir:      dir,
      datasetName: ds.name || ds.slug,
    };

    const missing = missingMm3TrainModels('train', opts.basePrecision);
    if (missing.length) {
      res.status(400).json({
        error: `MiniMax-Music3 training models are missing: ${missing.join(', ')}. A resume has to `
             + 'run on the same base the run was trained on.',
      });
      return;
    }

    // How much further. `addSteps` is the natural way to ask ("another 250");
    // `steps` sets the new total outright. Either way the cap has to be ahead
    // of where the state sits, or the engine loads 22 GB of model to do nothing.
    const from = run.resume.step;
    const addSteps = Number(b.addSteps);
    const total = Number.isFinite(Number(b.steps)) && Number(b.steps) > 0
      ? Math.trunc(Number(b.steps))
      : from + (Number.isFinite(addSteps) && addSteps > 0 ? Math.trunc(addSteps) : 250);
    if (total <= from) {
      res.status(400).json({
        error: `That run already reached step ${from} - ask for a step cap above it.`,
      });
      return;
    }
    opts.steps = Math.min(total, 100000);
    if (Number.isFinite(Number(b.saveEvery))) {
      opts.saveEvery = Math.max(0, Math.trunc(Number(b.saveEvery)));
    }
    // The stopping strategy is re-decidable on a resume: "run it 250 more" and
    // "run it until the loss reaches 0.4, capped at 1000" are both reasonable
    // ways to continue, and neither should be forced by what the first launch
    // happened to choose.
    if (b.stopMode === 'loss' || b.stopMode === 'steps') opts.stopMode = b.stopMode;
    if (Number.isFinite(Number(b.targetLoss))) {
      opts.targetLoss = Math.max(0, Number(b.targetLoss));
    }
    if (b.targetLossMetric === 'eval' || b.targetLossMetric === 'train') {
      opts.targetLossMetric = b.targetLossMetric;
    }
    if (Number.isFinite(Number(b.targetLossEpochs))) {
      opts.targetLossEpochs = Math.max(1, Math.trunc(Number(b.targetLossEpochs)));
    }
    if (opts.stopMode === 'loss' && opts.targetLossMetric === 'eval'
        && (opts.holdout <= 0 || opts.evalEvery <= 0)) {
      res.status(400).json({
        error: 'Targeting the held-out loss needs the hold-out set and the evaluation cadence this '
             + 'run was trained with, and this one has neither. Target the training loss instead.',
      });
      return;
    }
    // Previews are a per-launch choice too. Absent = whatever the run used.
    if (b.preview !== undefined) opts.preview = parseMm3PreviewOptions(b.preview);
    opts.resumeFrom = run.resume.statePath;
    opts.resumeStep = from;

    const job = queue.startMm3TrainLmJob(ds.id, opts);
    res.json({
      jobId: job.id, kind: job.kind, runName: run.runName, outDir: dir,
      from, steps: opts.steps, optionsSource: run.optionsSource,
    });
  } catch (err: any) {
    res.status(500).json({ error: err?.message || String(err) });
  }
});

router.post('/datasets/:id/train-lm', async (req: Request, res: Response) => {
  try {
    let ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }
    if (!aceTrainExe()) {
      res.status(503).json({ error: 'ace-train was not found next to ace-server — rebuild the engine' });
      return;
    }
    // Same adoption as preprocess: an existing dataset.json IS the build.
    ds = adoptExistingDatasetJson(ds, repo.updateDataset);
    if (!ds.builtAt || !ds.datasetJsonPath || !fs.existsSync(ds.datasetJsonPath)) {
      res.status(400).json({ error: 'Dataset must be built first — run Build before Training' });
      return;
    }

    const body = (req.body || {}) as TrainLmOptions & { ditModel?: unknown };

    // ── variant ──────────────────────────────────────────────────────────
    const requestedVariant = typeof body.variantKey === 'string' ? body.variantKey.trim() : '';
    // isSafeVariantKey (inside variantExists) rejects any key that is not a
    // single directory name: without it `../../otherslug/…` escapes the tensors
    // root and ace-train would read from — and write lm_codes.jsonl into — an
    // arbitrary directory. Same §7.8 rule the two preprocess routes apply.
    if (requestedVariant && !variantExists(ds.slug, requestedVariant)) {
      res.status(400).json({ error: `Unknown preprocess variant: ${requestedVariant}` });
      return;
    }
    const variantKey = requestedVariant || newestVariantKey(ds.slug);
    if (!variantKey) {
      res.status(400).json({ error: 'Dataset has no preprocessed tensors — run Preprocess first' });
      return;
    }

    // ── base size ────────────────────────────────────────────────────────
    // All three sizes are accepted: 4B trains through the engine's low-VRAM
    // path (per-layer checkpointing + chunked CE). An unaffordable 4B request
    // is now refused by ace-train's own VRAM solve with real numbers, not by a
    // blanket 400 here.
    // DEFAULT IS 4B, matching TRAIN_LM_DEFAULTS (2026-07-30). It was '0.6B' —
    // the safe-for-any-GPU seed — while the form seeded 4B, and those are two
    // different sets of numbers reached by two different callers. The batch
    // pipeline POSTs only the STORED per-stage defaults (`{}` unless someone
    // has PUT /defaults, and no UI does), so every field it omits falls through
    // to this handler. A batch run was therefore silently training 0.6B
    // adapters while the identical manual run trained 4B — the one field where
    // the two disagreed.
    //
    // An unaffordable 4B is refused by ace-train's own VRAM solve with real
    // numbers, so a small card now gets a loud, specific failure instead of a
    // quietly wrong-sized adapter. That is the better of the two failure modes.
    const lmSize: LmSize =
      body.lmSize === '1.7B' ? '1.7B' : body.lmSize === '0.6B' ? '0.6B' : '4B';

    // ── models ───────────────────────────────────────────────────────────
    // Same never-probed race fix as preprocess: a fresh boot would otherwise
    // 400 with "No LM base model available" while the engine is reachable.
    let snap = getModelSnapshot();
    if (!snap.cachedAt && !isEngineSuspended()) snap = await refreshModelSnapshot();
    const requestedLm = typeof body.lmModel === 'string' ? body.lmModel.trim() : '';
    const lmModel = requestedLm || pickLmFor(lmSize, snap.lm);
    if (!lmModel) {
      res.status(400).json({ error: `No LM base model available for ${lmSize}` });
      return;
    }
    if (requestedLm && snap.lm.length > 0 && !snap.lm.includes(lmModel)) {
      res.status(400).json({ error: `Unknown LM model: ${lmModel}` });
      return;
    }

    // The FSQ tokenizer lives inside the DiT, and it must be the one the
    // latents were made against — the variant's own record wins by default.
    const ditOverride = typeof body.ditModel === 'string' ? body.ditModel.trim() : '';
    const ditModel = ditOverride || variantDitModel(ds.slug, variantKey);

    // ── adapter name / dir ───────────────────────────────────────────────
    const adapterName = (typeof body.adapterName === 'string' ? body.adapterName.trim() : '') || ds.slug;
    // The regex alone accepts '.', '..', '.hidden' — all of which safeAdapterName()
    // silently REWRITES ('..' and '.' both collapse to 'adapter', '.hidden' to
    // 'hidden'). The stored/logged/returned adapterName would then name a
    // directory that does not exist, and two distinct requests would write the
    // same dir. Reject anything the sanitiser would have to change.
    if (!ADAPTER_NAME_RE.test(adapterName) || safeAdapterName(adapterName) !== adapterName) {
      res.status(400).json({ error: 'adapterName must match [A-Za-z0-9._-]{1,64}' });
      return;
    }
    // Per-base + per-run layout: every run writes a FRESH
    // <adapters>/lm-<sizeSlug>/<name>/<stamp> dir, so retraining an artist
    // never overwrites an earlier adapter. Contain against the adapters root,
    // refusing the root itself and any size root — this path is mkdir'd and
    // written into by a spawned process (§7.8).
    const adaptersRoot = config.aceServer.adapters;
    const adapterDir = lmRunDirFor(adapterName, lmSize);
    const isARoot = [adaptersRoot, adapterLmRoot(), ...lmAdapterRoots().map(r => r.dir)]
      .some(r => path.resolve(r) === path.resolve(adapterDir));
    if (!isInside(adaptersRoot, adapterDir) || isARoot) {
      res.status(400).json({ error: 'adapterName must match [A-Za-z0-9._-]{1,64}' });
      return;
    }

    // ── numeric clamps (§4.5 step 8) ─────────────────────────────────────
    const epochs = numOpt(body.epochs, 150);
    // 1.5 (Rob, 2026-09-02) — was 0.1 from 2026-08-12, 0.2 earlier that day, 2.0
    // before that. The render-scored stage ladder (docs/plans/lm-attr-probe/
    // RESULTS.md §4, 9 artists) showed every measurable gain of an LM adapter is
    // in by CE ~2.0 and the last leg to 0.1 only doubled the looping-plan rate
    // (26 % vs 11 % at 1.5). With the ladder below, 1.5 means one 2.0 leg then
    // one 1.5 leg.
    // Tracks TRAIN_LM_DEFAULTS.targetLoss: the batch pipeline POSTs the STORED
    // per-stage defaults (`{}` in practice), so this fallback IS the number a
    // bulk run trains to, and it has to be the one the form shows.
    const targetLoss = numOpt(body.targetLoss, 1.5);
    // ── Staged target chain (Rob, 2026-08-29: "this MUST be the new default") ─
    // `targetLoss` is the FINAL target; the run always descends through the
    // 2.0 → 1.5 ladder first, one full ace-train leg per rung with
    // --init-adapter chaining between them. Resetting the optimizer state and
    // LR schedule at each rung is what prevents the straight-dive loop
    // attractor (gojira: 95.7% plan loops straight vs 4% chained, same data,
    // same final CE) and measurably improves songwriting (nirvana E3 > D by
    // ear). Rungs at or below the final target are dropped, so an explicit
    // high target (e.g. 2.0) degenerates to the legacy single leg. A caller
    // that genuinely wants one straight leg sends targetLossStages: [final].
    const stageLadder = [2.0, 1.5];
    let targetLossStages: number[];
    if (Array.isArray(body.targetLossStages) && body.targetLossStages.length > 0) {
      targetLossStages = body.targetLossStages.map((v: unknown) => Number(v));
      if (targetLossStages.some(v => !Number.isFinite(v) || v < 0 || v > 20)) {
        res.status(400).json({ error: 'targetLossStages entries must be numbers between 0 and 20' });
        return;
      }
      // Stages must strictly descend (0 = "no auto-stop" is only legal last).
      for (let i = 1; i < targetLossStages.length; i++) {
        const prev = targetLossStages[i - 1], cur = targetLossStages[i];
        if (prev === 0 || (cur !== 0 && cur >= prev)) {
          res.status(400).json({ error: 'targetLossStages must strictly descend (0 only as the last entry)' });
          return;
        }
      }
    } else {
      // targetLoss 0 = "no auto-stop": still rest at the ladder rungs first.
      targetLossStages = [
        ...stageLadder.filter(r => targetLoss === 0 || r > targetLoss),
        targetLoss,
      ];
    }
    // LoRA rank 128 / alpha 256 is the DEFAULT (Rob, 2026-09-03; LoKr dim 128
    // was the default from 2026-07-30). An omitted adapterType means LoRA, so a
    // caller that wants LoKr must say so. The batch pipeline stores partial
    // option bags, and an absent field there lands on the same default a user
    // sees in the form (TRAIN_LM_DEFAULTS).
    const rank = numOpt(body.rank, 128);
    const lmIsLokr = body.adapterType === 'lokr';
    const lmMuonLrScale = numOpt(body.muonLrScale, 20.0);
    const lmMuonNsSteps = numOpt(body.muonNsSteps, 5);
    if (body.optimizer !== undefined && body.optimizer !== 'adamw' && body.optimizer !== 'muon'
        && body.optimizer !== 'prodigy') {
      res.status(400).json({ error: 'optimizer must be adamw, muon or prodigy' });
      return;
    }
    if (lmMuonLrScale < 0.001 || lmMuonLrScale > 1000) {
      res.status(400).json({ error: 'muonLrScale must be between 0.001 and 1000' });
      return;
    }
    if (lmMuonNsSteps < 1 || lmMuonNsSteps > 20) {
      res.status(400).json({ error: 'muonNsSteps must be between 1 and 20' });
      return;
    }
    const lmLokrDim = numOpt(body.lokrDim, 128);
    const lmLokrAlpha = numOpt(body.lokrAlpha, 128);
    const lmLokrFactor = numOpt(body.lokrFactor, 6);
    if (body.adapterType !== undefined && body.adapterType !== 'lora' && body.adapterType !== 'lokr') {
      res.status(400).json({ error: 'adapterType must be lora or lokr' });
      return;
    }
    if (lmIsLokr && (lmLokrDim < 4 || lmLokrDim > 4096)) {
      res.status(400).json({ error: 'lokrDim must be between 4 and 4096' });
      return;
    }
    if (lmIsLokr && (lmLokrFactor !== -1 && (lmLokrFactor < 2 || lmLokrFactor > 64))) {
      res.status(400).json({ error: 'lokrFactor must be -1 or between 2 and 64' });
      return;
    }
    const alpha = numOpt(body.alpha, 256);
    const learningRate = numOpt(body.learningRate, 0.0001);
    const gradAccum = numOpt(body.gradAccum, 2);
    const gradClip = numOpt(body.gradClip, 1.0);
    const warmupRatio = numOpt(body.warmupRatio, 0.05);
    const weightDecay = numOpt(body.weightDecay, 0.01);
    const maxLen = numOpt(body.maxLen, 0);
    const seed = numOpt(body.seed, 42);
    // 0 = milestones OFF by default (Rob, 2026-08-12). History: the default was
    // 0.1 from 2026-08-09 (a 0 fallback had silently disabled snapshots for the
    // whole 2026-08 LoKr corpus, back when nobody wanted that); now off is the
    // intended default for both the form and pipeline/batch runs. Callers that
    // want milestone snapshots must send milestoneStep explicitly.
    const milestoneStep = numOpt(body.milestoneStep, 0);
    const milestoneKeep = numOpt(body.milestoneKeep, 6);

    const rangeFailure =
      outOfRange('epochs', epochs, 1, 200)
      ?? outOfRange('targetLoss', targetLoss, 0, 20)
      ?? outOfRange('rank', rank, 1, 256)
      ?? outOfRange('alpha', alpha, 1, 1024)
      ?? outOfRange('gradAccum', gradAccum, 1, 64)
      ?? outOfRange('gradClip', gradClip, 0, 100)
      ?? outOfRange('warmupRatio', warmupRatio, 0, 0.5)
      ?? outOfRange('weightDecay', weightDecay, 0, 1)
      ?? outOfRange('seed', seed, 0, 2 ** 31 - 1)
      ?? outOfRange('milestoneStep', milestoneStep, 0, 5)
      ?? outOfRange('milestoneKeep', milestoneKeep, 0, 64);
    if (rangeFailure) {
      res.status(400).json({ error: rangeFailure });
      return;
    }
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) {
      res.status(400).json({ error: 'learningRate must be greater than 0 and at most 1' });
      return;
    }
    // 0 means "auto-fit from free VRAM" (L8); any explicit value must be usable.
    if (maxLen !== 0 && (maxLen < 512 || maxLen > 16384)) {
      res.status(400).json({ error: 'maxLen must be 0 (auto) or between 512 and 16384' });
      return;
    }

    // ── low-VRAM knobs (4B §2.5) ─────────────────────────────────────────
    // 'auto' is the engine's own default, so an omitted field emits no flag at
    // all (see buildTrainLmArgs) and the argv stays byte-identical to today.
    const lowVram = body.lowVram === undefined ? 'auto' : body.lowVram;
    if (lowVram !== 'auto' && lowVram !== 'on' && lowVram !== 'off') {
      res.status(400).json({ error: 'lowVram must be auto, on or off' });
      return;
    }
    const attnHeadBlock = numOpt(body.attnHeadBlock, 0);
    const chunk = numOpt(body.chunk, 0);
    if (!Number.isFinite(attnHeadBlock) || attnHeadBlock < 0 || attnHeadBlock > 128) {
      res.status(400).json({ error: 'attnHeadBlock must be between 0 and 128' });
      return;
    }
    // 0 = "engine default (128)"; any explicit value must be a usable chunk.
    if (chunk !== 0 && (chunk < 16 || chunk > 1024)) {
      res.status(400).json({ error: 'chunk must be between 16 and 1024' });
      return;
    }

    // Attention backend (2026-09-02 lm-flash-attn plan, Stream B — the DiT's
    // attnBackend ported to train-lm). Refused, not coerced, same rule as
    // mirror/optimizer/bwd on the DiT route. Default 'flash' since 2026-09-03
    // (Rob), after the gated port: byte-identical off, 5.5 % faster at 4B,
    // drift in the bf16 class. An explicit 'exact' is honoured.
    const attnBackend = body.attnBackend === 'exact' ? 'exact' as const
      : body.attnBackend === 'flash-f32' ? 'flash-f32' as const
      : 'flash' as const;
    if (body.attnBackend !== undefined && body.attnBackend !== 'exact' && body.attnBackend !== 'flash' &&
        body.attnBackend !== 'flash-f32') {
      res.status(400).json({ error: 'attnBackend must be exact, flash or flash-f32' });
      return;
    }
    // The fused flash op has no S² term to cut, so head-blocking is meaningless
    // under it — the engine exits 2 on this exact pair (D3 in the plan doc, the
    // same shape as the `weights bf16` + `bwd mm` refusal above). Caught here so
    // the runner never stops ace-server for a combination ace-train would only
    // reject after the engine is already down.
    if (attnBackend !== 'exact' && attnHeadBlock > 0) {
      res.status(400).json({
        error: 'attnBackend flash/flash-f32 and attnHeadBlock > 0 cannot be combined — the fused attention '
             + 'op has no S² term for head-blocking to cut, and the engine exits 2 on the pair. Leave '
             + 'attnHeadBlock at 0 (engine picks) when training with flash attention.',
      });
      return;
    }

    // ── speed levers (2026-07-28 plan §2.5) ──────────────────────────────
    // 'weights' went bf16 (2026-07-29) and back to 'f32-window' (Rob,
    // 2026-07-30, final) to match the DiT's F32 mirror. That is once again the
    // CLI's own default, so an omitted field emits no --weights flag at all.
    //
    // KNOWN AND ACCEPTED COST: unlike the DiT's mirror this is not a free
    // precision dial — bf16 is ALSO the only route to the mul_mat backward on
    // the LM (lm-bf16.h rewrites out_prod in place, ~1.7-1.8x on the GEMM mix)
    // and --bwd mm cannot substitute; see the bwd comment below.
    //
    // The engine owns the semantic rules (bf16 needs a BF16 base; batch>1
    // implies low-VRAM) — this is a value whitelist only, so a stale UI can
    // never make the runner stop ace-server for an argument ace-train rejects.
    // 'bf16' (Rob, 2026-09-03) — the LM's equivalent of the DiT's bf16 mirror:
    // BF16 transposed projection window + the patched BF16 out_prod, 1.256x at
    // 4B, same gradient-error class the DiT accepted. The engine falls back to
    // f32-window on a non-CUDA backend or a non-BF16 base with a warning.
    const weights = body.weights === undefined ? 'bf16' : body.weights;
    if (weights !== 'f32-window' && weights !== 'bf16') {
      res.status(400).json({ error: 'weights must be f32-window or bf16' });
      return;
    }
    // MUL_MAT activation-gradient formulation.
    //
    // THE LM DEFAULT IS 'outprod', NOT 'mm' — deliberately different from
    // train-dit's. `weights` above already defaults to 'bf16', and lm-bf16.h's
    // Lever A reaches the same mul_mat backward by rewriting ggml's out_prod
    // nodes in place, asserting exactly 7 rewrites per segment. Under --bwd mm
    // ggml emits mul_mat directly, that surgery finds nothing, and the S18
    // tripwire GGML_ABORTs the run. Defaulting the LM to 'mm' would therefore
    // brick the DEFAULT LM training job. It would also buy nothing on the
    // f32-window path, where the transposed weight is the F32 window and the
    // GEMM stays TF32 while paying an extra cont. --bwd mm is a train-dit win.
    const bwd = body.bwd === undefined ? 'outprod' : body.bwd;
    if (bwd !== 'outprod' && bwd !== 'mm') {
      res.status(400).json({ error: 'bwd must be outprod or mm' });
      return;
    }
    // Refused HERE so the user gets a 400 instead of the runner stopping
    // ace-server for a run ace-train exits 2 on (same rule as the other levers).
    if (weights === 'bf16' && bwd === 'mm') {
      res.status(400).json({
        error: 'weights bf16 and bwd mm are two routes to the same mul_mat backward and cannot be '
             + 'combined — the bf16 lever rewrites ggml out_prod nodes in place and aborts when there '
             + 'are none. Use weights bf16 on its own, or pair bwd mm with weights f32-window.',
      });
      return;
    }
    const rawBatch = body.batch === undefined ? 1 : body.batch;
    let batch: number | 'auto';
    if (rawBatch === 'auto') {
      batch = 'auto';
    } else {
      const n = Number(rawBatch);
      if (!Number.isFinite(n) || Math.trunc(n) !== n || n < 1 || n > 8) {
        res.status(400).json({ error: 'batch must be 1-8 or auto' });
        return;
      }
      batch = n;
    }
    // Lever B (micro-batching) was never written — its §6.1 build gate measures
    // 9.3% amortisable host overhead at 4B against a 10% bar — so ace-train
    // refuses anything but 1 with exit 2. That refusal has to happen HERE and
    // not there: trainLmRunner stops ace-server before it spawns ace-train, so
    // letting a batch>1 request through would shut the user's engine down, fail
    // instantly, and restart it — an engine bounce as the reward for touching a
    // dropdown. Reject before the job is queued.
    if (batch !== 1) {
      res.status(400).json({
        error: 'batch must be 1 — micro-batching is not built in this engine. The host overhead it would '
             + 'amortise measures 9.3% at 4B, under the 10% bar its build gate required. Use gradient '
             + 'accumulation to change the effective batch size.',
      });
      return;
    }

    // ── caption dropout + prior preservation (2026-09-02) ────────────────
    // Ported from the MM3 LM trainer's two levers (mm3-lm-train-run.h,
    // mm3-lm-prior.h) — see docs/plans/lm-attr-probe/HANDOFF.md. Both are
    // recipe knobs like --lr, so they are validated here and carried
    // unchanged through every leg of the staged chain (opts is spread as-is
    // into each leg by trainLmRunner.ts).
    // Defaults 0.3 / 3 (Rob, 2026-09-03; were 0 / 0): the measured best recipe,
    // docs/plans/lm-attr-probe/RESULTS.md §6-7. Tracks TRAIN_LM_DEFAULTS in
    // TrainLmForm.tsx — the batch pipeline POSTs an empty option bag, so these
    // fallbacks ARE what a bulk run trains with. `*Explicit` records whether the
    // caller chose the value: a DEFAULT that cannot be honoured (no trigger for
    // caption dropout, no other 600 s corpus for prior preservation) is dropped
    // with a warning; an EXPLICIT one is a 400, because the engine would exit 2.
    const captionDropoutExplicit = body.captionDropout !== undefined;
    let captionDropout = numOpt(body.captionDropout, 0.3);
    if (outOfRange('captionDropout', captionDropout, 0, 1)) {
      res.status(400).json({ error: outOfRange('captionDropout', captionDropout, 0, 1) });
      return;
    }
    const regEveryExplicit = body.regEvery !== undefined;
    let regEvery = Math.trunc(numOpt(body.regEvery, 3));
    const regTopk = Math.trunc(numOpt(body.regTopk, 64));
    const regSongs = Math.trunc(numOpt(body.regSongs, 24));
    // Prior teacher (docs/plans/lm-attr-probe/OVERNIGHT.md "Live teacher
    // spec"). Refused, not coerced, same rule as attnBackend above. Default
    // stays 'cached' — today's byte-identical top-K prior.
    const regTeacher = body.regTeacher === 'live' ? 'live' as const : 'cached' as const;
    if (body.regTeacher !== undefined && body.regTeacher !== 'cached' && body.regTeacher !== 'live') {
      res.status(400).json({ error: 'regTeacher must be cached or live' });
      return;
    }
    let regCodes = '';
    let regPriorDir = '';
    if (regEvery > 0) {
      if (regEvery < 2) {
        res.status(400).json({
          error: 'regEvery must be at least 2 — at 1 every step would be a regularisation step and '
               + 'nothing would learn the artist.',
        });
        return;
      }
      if (outOfRange('regTopk', regTopk, 1, 256) || outOfRange('regSongs', regSongs, 1, 200)) {
        res.status(400).json({ error: outOfRange('regTopk', regTopk, 1, 256) ?? outOfRange('regSongs', regSongs, 1, 200) });
        return;
      }
      const rawCorpora = body.regCorpora;
      if (Array.isArray(rawCorpora)) {
        const paths = rawCorpora.filter((p): p is string => typeof p === 'string' && p.trim() !== '');
        const allTensorsRoot = path.join(trainingBaseDir, 'tensors');
        const bad = paths.find(p => !isInside(allTensorsRoot, path.resolve(p)) || path.basename(p) !== 'lm_codes.jsonl'
          || !fs.existsSync(p));
        if (!paths.length || bad) {
          res.status(400).json({
            error: bad
              ? `regCorpora entry "${bad}" must be an existing lm_codes.jsonl under data/training/tensors/`
              : 'regCorpora must list at least one lm_codes.jsonl path when not "auto"',
          });
          return;
        }
        regCodes = paths.map(p => path.resolve(p)).join(',');
      } else {
        const picked = findRegCorpora(ds.slug, 6);
        if (!picked.length) {
          if (regEveryExplicit) {
            res.status(400).json({
              error: 'Prior preservation needs at least one OTHER artist preprocessed at the 600 s cap — '
                   + 'none were found. Preprocess another dataset first, or set regEvery to 0.',
            });
            return;
          }
          console.warn(`[Training] train-lm ${ds.slug}: prior preservation is the default but no other 600 s `
                     + 'corpus exists — training WITHOUT it (regEvery 0). Preprocess another dataset to enable it.');
          regEvery = 0;
        } else {
          console.log(`[Training] train-lm ${ds.slug}: auto reg corpus — ${picked.map(p => p.slug).join(', ')}`);
          regCodes = picked.map(p => p.codesPath).join(',');
        }
      }
      // Fixed at the TOP-LEVEL adapter dir, not per-stage — every leg of a
      // staged chain must share one cache, captured once (see aceTrain.ts
      // ResolvedTrainLmOptions.regPriorDir).
      if (regEvery > 0) regPriorDir = path.join(adapterDir, 'reg-prior');
    }

    // Caption dropout needs a trigger word: the engine resolves it from the
    // variant's preprocess_meta.json (custom_tag, and a `replace` position
    // counts as none — lm_resolve_trigger) and exits 2 when --caption-dropout
    // finds nothing to drop to. Mirror that here so a DEFAULT run on an
    // untagged dataset degrades instead of failing after the model load.
    if (captionDropout > 0) {
      let trigger = '';
      try {
        const metaPath = path.join(tensorsDir(ds.slug, variantKey), 'preprocess_meta.json');
        const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8')) as { custom_tag?: unknown; tag_position?: unknown };
        const tag = typeof meta.custom_tag === 'string' ? meta.custom_tag.trim() : '';
        trigger = (tag && meta.tag_position !== 'replace') ? tag : '';
      } catch { trigger = ''; }
      if (!trigger) {
        if (captionDropoutExplicit) {
          res.status(400).json({
            error: 'captionDropout needs a trigger word (the variant\'s preprocess custom_tag, not in `replace` '
                 + 'position) — this variant has none. Re-preprocess with a custom tag, or set captionDropout to 0.',
          });
          return;
        }
        console.warn(`[Training] train-lm ${ds.slug}: caption dropout is the default but the variant has no trigger `
                   + 'word — training WITHOUT it (captionDropout 0). Preprocess with a custom tag to enable it.');
        captionDropout = 0;
      }
    }

    // ── resume + post-training calibration (2026-08-10) ─────────────────
    // initAdapter must be a real adapter run dir BEFORE the runner stops the
    // engine — the engine would also refuse it, but only after a full engine
    // stop/restart cycle (same rule as every other pre-flight check here).
    //
    // The sentinel 'latest' (the UI checkbox) resolves to the newest run of
    // THIS adapter name that holds weights, excluding -calibrated bakes: a
    // bake's factors carry the baked scale, so resuming one would train from
    // rescaled weights and shift the adapter's effective strength mid-lineage.
    //
    // An omitted initAdapter meant 'latest' from 2026-08-12 — SUPERSEDED
    // 2026-08-29 by the staged-chain default: a chain MUST start from scratch
    // (that is the validated procedure; the first smoke run proved the failure
    // mode — 'latest' resumed an existing memorized adapter, every rung was
    // instantly satisfied and the "chain" was three 1-epoch no-ops). An
    // EXPLICIT initAdapter ('latest' or a path) is still honored as a resume,
    // and a resume collapses the ladder to its final target below — re-running
    // the 2.0/1.5 rungs on an adapter already past them is meaningless.
    const initRequested = typeof body.initAdapter === 'string' && body.initAdapter.trim() !== '';
    let initAdapter = initRequested
      ? (body.initAdapter as string).trim()
      : (targetLossStages.length > 1 ? '' : 'latest');
    if (initAdapter === 'latest') {
      const artistDir = path.dirname(adapterDir);
      let newest = '';
      try {
        const runs = fs.readdirSync(artistDir, { withFileTypes: true })
          .filter(e => e.isDirectory() && !/-calibrated$/i.test(e.name)
            && hasWeights(path.join(artistDir, e.name)))
          .map(e => e.name)
          .sort();
        if (runs.length) newest = path.join(artistDir, runs[runs.length - 1]);
      } catch { /* artist dir may not exist yet */ }
      if (!newest && hasWeights(artistDir)) newest = artistDir;   // legacy unversioned
      // "Nothing to resume" is NOT an error any more. It was a 400 while resume
      // was a deliberate opt-in; now that it is the default for both the form
      // and the pipeline, a first-ever run for an adapter name would fail on the
      // one thing it cannot possibly satisfy. Fall through to a scratch run and
      // say so in the log, so a bulk sweep can mix new and existing datasets.
      if (!newest) {
        console.log(
          `[Training] train-lm: nothing to resume for "${adapterName}" (${lmSize}) — training from scratch`);
      }
      initAdapter = newest;
    }
    if (initAdapter && !hasWeights(initAdapter)) {
      res.status(400).json({
        error: `initAdapter ${initAdapter} holds no adapter weights (adapter_model/lokr_weights.safetensors)`,
      });
      return;
    }
    // A resume skips straight to the final target unless the caller pinned an
    // explicit stage list — see the initAdapter note above.
    if (initAdapter && !Array.isArray(body.targetLossStages)) {
      targetLossStages = [targetLoss];
    }
    // Calibration is OPT-IN (Rob, 2026-08-12) — it was default ON from
    // 2026-08-10. Only an explicit `true` runs it, so the batch pipeline's empty
    // bag no longer appends an eval pass to every adapter in a bulk sweep.
    const calibrate = body.calibrate === true;
    const calibrateRepoint = body.calibrateRepoint !== false;

    // ── stages ───────────────────────────────────────────────────────────
    const requestedStages = Array.isArray(body.stages) ? body.stages : [];
    const stages = TRAIN_LM_STAGES.filter(s => requestedStages.includes(s));
    const resolvedStages: TrainLmStage[] = stages.length > 0 ? [...stages] : [...TRAIN_LM_STAGES];

    // Belt and braces: the key is already segment-validated above, but this is
    // the path a spawned process reads and writes, so assert containment too.
    const tensorsPath = path.join(tensorsRoot(ds.slug), variantKey);
    if (!isInside(tensorsRoot(ds.slug), tensorsPath)) {
      res.status(400).json({ error: `Unknown preprocess variant: ${variantKey}` });
      return;
    }
    // Soft prompt defaults (Rob, 2026-09-04): the joint recipe is the default. A
    // request that says nothing gets a token named after the adapter, k=32,
    // LR 5e-3 and an 8-column prefix; an explicit artistToken: '' switches it off.
    const artistTokenResolved = body.artistToken === undefined
      ? (String(adapterName).replace(/[^A-Za-z0-9_-]/g, '').slice(0, 64) || 'artist')
      : (typeof body.artistToken === 'string' ? body.artistToken.trim().replace(/[^A-Za-z0-9_-]/g, '').slice(0, 64) : '');
    const prefixNResolved = Math.min(64, Math.max(0, Math.trunc(numOpt(body.prefixN, artistTokenResolved ? 8 : 0))));
    // A KV prefix runs under exact attention only (the flash probe does not cover
    // S_kv != S), and the prefix is on by default now, so coerce rather than fail
    // a default run 10 s in. Logged so a flash request knows why it went exact.
    const attnBackendEff = (artistTokenResolved && prefixNResolved > 0 && attnBackend !== 'exact') ? 'exact' as const : attnBackend;
    if (attnBackendEff !== attnBackend) {
      console.log(`[Training] train-lm: attnBackend ${attnBackend} -> exact (a KV prefix of ${prefixNResolved} columns needs exact attention)`);
    }
    const opts: ResolvedTrainLmOptions = {
      lmSize,
      lmModel,
      ditModel,
      variantKey,
      tensorsDir: tensorsPath,
      codesPath: path.join(tensorsPath, 'lm_codes.jsonl'),
      adapterName,
      adapterDir,
      targetLoss,
      targetLossStages,
      epochs: Math.trunc(epochs),
      // Only the exact string 'lokr' opts in; anything else is a LoRA.
      adapterType: lmIsLokr ? 'lokr' : 'lora',
      // Prodigy is the LM default (Rob, 2026-09-03): the step size is estimated
      // online, so there is no hand-tuned LR to get wrong per artist. The
      // exact strings 'adamw' / 'muon' opt out.
      optimizer: body.optimizer === 'adamw' ? 'adamw' : body.optimizer === 'muon' ? 'muon' : 'prodigy',
      muonLrScale: lmMuonLrScale,
      muonNsSteps: Math.trunc(lmMuonNsSteps),
      rank: Math.trunc(rank),
      alpha: Math.trunc(alpha),
      lokrDim: Math.trunc(lmLokrDim),
      lokrAlpha: lmLokrAlpha,
      lokrFactor: Math.trunc(lmLokrFactor),
      lokrDecomposeBoth: body.lokrDecomposeBoth !== false,
      learningRate,
      gradAccum: Math.trunc(gradAccum),
      gradClip,
      warmupRatio,
      weightDecay,
      maxLen: Math.trunc(maxLen),
      seed: Math.trunc(seed),
      lossOnCot: body.lossOnCot !== false,
      order: body.order === 'fixed' ? 'fixed' : 'shuffle',
      milestoneStep,
      milestoneKeep: Math.trunc(milestoneKeep),
      stages: resolvedStages,
      overwrite: body.overwrite === true,
      stopEngine: body.stopEngine !== false,
      initAdapter,
      calibrate,
      calibrateRepoint,
      lowVram,
      attnHeadBlock: Math.trunc(attnHeadBlock),
      chunk: Math.trunc(chunk),
      weights,
      batch,
      bwd,
      captionDropout,
      rslora: body.rslora === true,
      loraPlusRatio: numOpt(body.loraPlusRatio, 1),
      // Soft prompt: the name becomes a safetensors key, so keep it to a slug.
      artistToken: artistTokenResolved,
      artistTokenK: Math.min(256, Math.max(1, Math.trunc(numOpt(body.artistTokenK, 32)))),
      artistTokenLr: Math.min(1, Math.max(0, numOpt(body.artistTokenLr, 0.005))),
      prefixN: prefixNResolved,
      regEvery,
      regTopk,
      regSongs,
      regCodes,
      regPriorDir,
      regTeacher,
      attnBackend: attnBackendEff,
    };

    const job = queue.startTrainLmJob(ds.id, opts);
    console.log(
      `[Training] train-lm job ${job.id} queued — ${lmSize} ${lmModel}, variant ${variantKey} → ${adapterDir}`);
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] train-lm start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/datasets/:id/train-lm', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const lmSizeQuery: LmSize | undefined =
      req.query.lmSize === '1.7B' ? '1.7B'
        : req.query.lmSize === '4B' ? '4B'
          : req.query.lmSize === '0.6B' ? '0.6B'
            : undefined;
    // `[]`, not `await buildSamples(ds)`: readTrainLmStatus names the parameter
    // `_samples` and never reads it (its counters are measured against the tensor
    // cache, not the dataset). buildSamples walks the whole source tree, stats
    // every audio file and reads every sidecar — a full recursive scan per poll,
    // thrown away, on the same process relaying the training JSONL.
    res.json(readTrainLmStatus(ds, [], {
      variantKey: typeof req.query.variantKey === 'string' ? req.query.variantKey : undefined,
      adapterName: typeof req.query.adapterName === 'string' ? req.query.adapterName : undefined,
      lmSize: lmSizeQuery,
    }));
  } catch (err: any) {
    console.error(`[Training] train-lm status failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── DiT LoRA training (§2.7, phase 4) ────────────────────────────────────

const TRAIN_DIT_STAGES: readonly TrainDitStage[] = ['train', 'export'];

router.post('/datasets/:id/train-dit', async (req: Request, res: Response) => {
  try {
    // ── 1. dataset / job / binary ────────────────────────────────────────
    let ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }
    if (!aceTrainExe()) {
      res.status(503).json({ error: 'ace-train was not found next to ace-server — rebuild the engine' });
      return;
    }
    // ── 2. built ─────────────────────────────────────────────────────────
    // Same adoption as preprocess: an existing dataset.json IS the build.
    ds = adoptExistingDatasetJson(ds, repo.updateDataset);
    if (!ds.builtAt || !ds.datasetJsonPath || !fs.existsSync(ds.datasetJsonPath)) {
      res.status(400).json({ error: 'Dataset must be built first — run Build before Training' });
      return;
    }

    const body = (req.body || {}) as TrainDitOptions;

    // ── 3. variant ───────────────────────────────────────────────────────
    // isSafeVariantKey (inside variantExists) rejects any key that is not a
    // single directory name: without it `../../otherslug/…` escapes the tensors
    // root and ace-train would read an arbitrary directory. Same §7.8 rule the
    // preprocess and train-lm routes apply.
    const requestedVariant = typeof body.variantKey === 'string' ? body.variantKey.trim() : '';
    if (requestedVariant && !variantExists(ds.slug, requestedVariant)) {
      res.status(400).json({ error: `Unknown preprocess variant: ${requestedVariant}` });
      return;
    }
    const variantKey = requestedVariant || newestVariantKey(ds.slug);
    if (!variantKey) {
      res.status(400).json({ error: 'Dataset has no preprocessed tensors — run Preprocess first' });
      return;
    }
    // Belt and braces: the key is already segment-validated above, but this is
    // the path a spawned process reads, so assert containment too.
    const tensorsPath = path.join(tensorsRoot(ds.slug), variantKey);
    if (!isInside(tensorsRoot(ds.slug), tensorsPath)) {
      res.status(400).json({ error: `Unknown preprocess variant: ${variantKey}` });
      return;
    }

    // ── 4. adapter type ──────────────────────────────────────────────────
    // Refused HERE rather than by the engine (which answers `fatal
    // reason=unsupported-adapter`, exit 1): the engine's refusal arrives only
    // after the runner has already stopped the app's engine, so a typo would
    // cost a full stop/restart cycle.
    if (body.adapterType !== undefined && body.adapterType !== 'lora' && body.adapterType !== 'lokr') {
      res.status(400).json({ error: 'adapterType must be "lora" or "lokr"' });
      return;
    }
    // DEFAULT IS LoKR (2026-07-30), matching the form: TrainPanel seeds
    // TRAIN_DIT_LOKR_DEFAULTS, not TRAIN_DIT_DEFAULTS — LoKR is the UI's default
    // adapter type (K1/K2, Rob's validated Uber-LoKR preference), and the base
    // constant only exists as the object the LoKR preset spreads.
    //
    // This handler is where the BATCH pipeline lands: it POSTs only the stored
    // per-stage defaults (`{}` in practice) so every omitted field resolves
    // here. Defaulting to 'lora' meant a batch run trained a LoRA where the
    // identical manual run trained a LoKR.
    //
    // NOTE the sense: `!== 'lora'`, not `=== 'lokr'`. Everything downstream
    // keys off isLokr — learningRate 0.002 vs 5e-4, weightDecay 0.001 vs 0.01,
    // lossWeighting none vs flow_snr — so getting this branch wrong silently
    // retunes four other parameters, not one.
    const adapterType: DitAdapterType = body.adapterType === 'lora' ? 'lora' : 'lokr';
    const isLokr = adapterType === 'lokr';

    // ── 5. base model ────────────────────────────────────────────────────
    // NEVER from user input: the cached encoder states and context latents are
    // this exact model's outputs, so training against another base is silently
    // wrong (§4.2 base-match guard).
    const ditModel = variantDitModel(ds.slug, variantKey);
    const ditPath = pickDitBaseFor(variantKey, tensorsPath);
    if (!ditPath) {
      res.status(400).json({
        error: 'This preprocess variant was made against a base that is no longer installed',
      });
      return;
    }

    // ── 6. adapter name / dir ────────────────────────────────────────────
    const adapterName = (typeof body.adapterName === 'string' ? body.adapterName.trim() : '') || ds.slug;
    // The regex alone accepts '.', '..', '.hidden' — all of which
    // safeAdapterName() silently REWRITES, so the returned/logged adapterName
    // would name a directory that does not exist and two distinct requests would
    // write the same dir. Reject anything the sanitiser would have to change.
    if (!ADAPTER_NAME_RE.test(adapterName) || safeAdapterName(adapterName) !== adapterName) {
      res.status(400).json({ error: 'adapterName must match [A-Za-z0-9._-]{1,64}' });
      return;
    }
    const adaptersRoot = adapterDitRoot();
    // Per-base + per-run layout: <adapters>/dit-<shorthand>/<name>/<stamp>,
    // the shorthand coming from the variant's base resolved above
    // (adapterLayout.ts). A fresh stamped dir per run — retraining an artist
    // never overwrites an earlier adapter.
    const adapterDir = ditRunDirFor(adapterName, ditModel);
    // Containment, same rule the preprocess outputDir uses (§7.8): this path is
    // mkdir'd and written into by a spawned process. The root itself is refused
    // — a run writing adapter_model.safetensors into the adapters root would
    // put a nameless adapter in every user's dropdown — and so are the LM
    // adapter roots (legacy flat `lm/` and the per-size `lm-*` dirs), where a
    // DiT PEFT dir would show up in the planner-adapter dropdown.
    const clashesRoot = [adaptersRoot, adapterLmRoot(), ...lmAdapterRoots().map(r => r.dir)]
      .some(r => path.resolve(r) === path.resolve(adapterDir));
    if (!isInside(adaptersRoot, adapterDir) || clashesRoot) {
      res.status(400).json({ error: 'adapterName must match [A-Za-z0-9._-]{1,64}' });
      return;
    }

    // ── 7. numeric clamps (§4.5 step 7) ──────────────────────────────────
    // train-dit defaults. These are what the UI path actually gets: TrainDitForm
    // sends every field, but any client that omits one lands here.
    //
    // K2 (lokr-dit-training plan §0): when adapterType==='lokr', four of these
    // omitted-field fallbacks change to Rob's Uber-LoKR-4 preset values. The
    // 'lora' path's fallbacks are untouched — byte-identical to before LoKR.
    // 250 for LoKR (2026-07-30 retune): a 400-epoch cosine horizon left every
    // measured run stopping at ~50% of peak LR, so the schedule never decayed
    // into the target. 250 cut epochs-to-0.6 from 228 to 203. LoRA keeps 400.
    const epochs = numOpt(body.epochs, 500);
    // 0.1 (Rob, 2026-08-13) — was 0.2 from 2026-07-30. Tracks
    // TRAIN_DIT_DEFAULTS.targetLoss: the batch pipeline POSTs the STORED
    // per-stage defaults (`{}` in practice), so this fallback IS the number a
    // bulk run trains to, and it has to be the one the form shows.
    // 0.3 (Rob, 2026-09-03): R1's loss-0.3 milestone and its final adapter were
    // indistinguishable by ear; 0.3 -> 0.1 is mostly memorising the set.
    const targetLoss = numOpt(body.targetLoss, 0.3);
    const rank = numOpt(body.rank, 128);
    const alpha = numOpt(body.alpha, 256);
    const lokrDim = numOpt(body.lokrDim, 512);
    const lokrAlpha = numOpt(body.lokrAlpha, 512);
    const lokrFactor = numOpt(body.lokrFactor, 6);
    const lokrDecomposeBoth = body.lokrDecomposeBoth !== false;
    const layers = numOpt(body.layers, 0);
    const crop = numOpt(body.crop, 0);
    const cropMin = numOpt(body.cropMin, 375);
    // Parsed early because cropMax's default depends on it. In flash mode the
    // default is 0 = "no pin": aceTrain then omits --crop-max entirely and the
    // engine lifts the cap to the dataset's longest track. An explicit number
    // is a pin, in either mode.
    // 'flash-f32' is the same fused graph as 'flash' — only the kernels'
    // arithmetic differs — so it takes the flash crop-cap default too.
    const attnBackend = body.attnBackend === 'flash' ? 'flash' as const
      : body.attnBackend === 'flash-f32' ? 'flash-f32' as const
      : 'exact' as const;
    // 0 = engine default cap (800 since 2026-09-03, every attention mode). A
    // number is an explicit pin the engine honours in both directions.
    const cropMax = numOpt(body.cropMax, 0);
    // Crop regime (2026-08-29): song-anchored positions + structured draws are
    // the defaults for every run — the batch pipeline POSTs {} and inherits
    // them. 'zero'/'random' remain reachable for A/B archaeology only.
    const cropAnchor = body.cropAnchor === 'zero' ? 'zero' as const : 'song' as const;
    const cropMode = body.cropMode === 'random' ? 'random' as const : 'structured' as const;
    const cropStartFrac = numOpt(body.cropStartFrac, 0.2);
    // Back to 0.2 (2026-09-01, was 0 for a day). The 2026-08-30 zeroing left
    // song endings unsupervised (~1.4 flush-end draws per 500-epoch run), and
    // every adapter of the 2026-08-31 overnight batch cut off mid-stream with
    // no ending. The engine's end share now splits flush-jitter /
    // closing-region draws — see dit_sample_crop_structured in
    // engine/src/train/dit-data.h for the corrected story.
    const cropEndFrac = numOpt(body.cropEndFrac, 0.2);
    const cropJitter = body.cropJitter === true;
    if (cropStartFrac < 0 || cropEndFrac < 0 || cropStartFrac + cropEndFrac > 1) {
      res.status(400).json({ error: 'cropStartFrac/cropEndFrac must be >= 0 and sum to <= 1' });
      return;
    }
    // LoKR 2e-3 @ GA 4 (2026-07-30 retune) replaces 1e-2 @ GA 20. Side-Step
    // reaches an effective batch of 20 as batch 5 x GA 4; we reached it by
    // accumulating 20, which is the same effective LR per sample under linear
    // scaling. Measured on gunship_unicorn: IDENTICAL epochs-to-target (227 vs
    // 228) with strictly better-behaved gradients — median grad-norm 0.062 vs
    // 0.031 (the sqrt(5) a 5x smaller batch predicts) and no warmup spike, where
    // GA 20 peaked at 13.5 on epoch 1. THE TWO MOVE TOGETHER: 2e-3 at GA 20, or
    // 1e-2 at GA 4, are both untested configurations. The LoRA path (5e-4, GA 4)
    // is unchanged, as is ace-train's own CLI default.
    const learningRate = numOpt(body.learningRate, isLokr ? 0.002 : 0.0005);
    const gradAccum = numOpt(body.gradAccum, 4);
    const gradClip = numOpt(body.gradClip, 1.0);
    const warmupRatio = numOpt(body.warmupRatio, 0.05);
    const weightDecay = numOpt(body.weightDecay, isLokr ? 0.001 : 0.01);
    const snrGamma = numOpt(body.snrGamma, 5);
    const tBias = numOpt(body.tBias, 0.5);
    const timestepMu = numOpt(body.timestepMu, -0.4);
    const timestepSigma = numOpt(body.timestepSigma, 1.0);
    const tMin = numOpt(body.tMin, 0);
    const tMax = numOpt(body.tMax, 1);
    const cfgRatio = numOpt(body.cfgRatio, 0.15);
    // Percent of micro-steps conditioned on the dataset's genre text instead of
    // the caption (D14). Default 30 — enough for the adapter to learn the genre
    // handle without the caption path going untrained.
    const genreRatio = numOpt(body.genreRatio, 30);
    const seed = numOpt(body.seed, 42);
    // 0.1, NOT 0 — same fix as train-lm above: the 0 fallback silently
    // disabled milestone snapshots for any caller omitting the field.
    const milestoneStep = numOpt(body.milestoneStep, 0.1);
    const milestoneKeep = numOpt(body.milestoneKeep, 6);
    const vramReserveMb = numOpt(body.vramReserveMb, 2048);
    // Micro-batching / checkpointing (design §2.2). ckptSegments mirrors the
    // engine's --ckpt semantics directly: 0=off, 1=auto, 2-32=fixed segments.
    // batch defaults to 1 = OFF (2026-07-29): measured ~2.5x SLOWER at full depth
    // on a 32 GB card, ~2.4x faster on shallow/partial-depth runs. Same default
    // as the engine's own DitTrainArgs, so an omitted field and an absent flag
    // land on the same behaviour.
    const batch = numOpt(body.batch, 1);
    const ckptSegments = numOpt(body.ckptSegments, 1);
    // Optimizer (2026-07-30). Muon is the DEFAULT for the DiT, after a full-run
    // A/B (161 epochs to target vs AdamW's 227) and ear validation. Resolved
    // below as `=== 'adamw' ? 'adamw' : 'muon'`, so an omitted field means Muon.
    const muonLrScale = numOpt(body.muonLrScale, 20.0);
    const muonMomentum = numOpt(body.muonMomentum, 0.95);
    const muonNsSteps = numOpt(body.muonNsSteps, 5);
    const muonMinDim = numOpt(body.muonMinDim, 16);

    const rangeFailure =
      outOfRange('epochs', epochs, 1, 2000)
      ?? outOfRange('targetLoss', targetLoss, 0, 20)
      ?? outOfRange('rank', rank, 1, 256)
      ?? outOfRange('alpha', alpha, 1, 1024)
      ?? outOfRange('lokrDim', lokrDim, 4, 4096)
      // §2.1: (0,8192] with 0 as the "-> dim" sentinel (K6) — 0 is valid input.
      ?? outOfRange('lokrAlpha', lokrAlpha, 0, 8192)
      ?? (lokrFactor !== -1 && (lokrFactor < 2 || lokrFactor > 64)
        ? 'lokrFactor must be -1 or between 2 and 64' : null)
      ?? outOfRange('layers', layers, 0, 64)
      ?? outOfRange('cropMin', cropMin, 128, 8192)
      ?? (cropMax > 0 ? outOfRange('cropMax', cropMax, 128, 8192) : null)
      ?? outOfRange('gradAccum', gradAccum, 1, 64)
      ?? outOfRange('gradClip', gradClip, 0, 100)
      ?? outOfRange('warmupRatio', warmupRatio, 0, 0.5)
      ?? outOfRange('weightDecay', weightDecay, 0, 1)
      ?? outOfRange('snrGamma', snrGamma, 1, 100)
      ?? outOfRange('tBias', tBias, 0, 4)
      ?? outOfRange('timestepMu', timestepMu, -4, 4)
      ?? outOfRange('tMin', tMin, 0, 1)
      ?? outOfRange('tMax', tMax, 0, 1)
      ?? outOfRange('cfgRatio', cfgRatio, 0, 1)
      ?? outOfRange('genreRatio', genreRatio, 0, 100)
      ?? outOfRange('seed', seed, 0, 2 ** 31 - 1)
      ?? outOfRange('milestoneStep', milestoneStep, 0, 5)
      ?? outOfRange('milestoneKeep', milestoneKeep, 0, 64)
      ?? outOfRange('vramReserveMb', vramReserveMb, 0, 16384)
      ?? outOfRange('batch', batch, 1, 16)
      ?? outOfRange('muonLrScale', muonLrScale, 0.001, 1000)
      ?? outOfRange('muonMomentum', muonMomentum, 0, 0.999)
      ?? outOfRange('muonNsSteps', muonNsSteps, 1, 20)
      ?? outOfRange('muonMinDim', muonMinDim, 1, 4096)
      // ckptSegments: 0=off, 1=auto, 2-32=fixed segment count (design §2.2).
      ?? (ckptSegments !== 0 && ckptSegments !== 1 && (ckptSegments < 2 || ckptSegments > 32)
        ? 'ckptSegments must be 0, 1, or between 2 and 32' : null);
    if (rangeFailure) {
      res.status(400).json({ error: rangeFailure });
      return;
    }
    // 0 means "auto-fit from free VRAM" (D10); any explicit value must be usable.
    if (crop !== 0 && (crop < 128 || crop > 8192)) {
      res.status(400).json({ error: 'crop must be 0 or between 128 and 8192 frames' });
      return;
    }
    if (cropMax > 0 && cropMax < cropMin) {
      res.status(400).json({ error: 'cropMax must be greater than or equal to cropMin' });
      return;
    }
    // Refused here rather than coerced: an unrecognised value must not
    // silently land on either side. 'bf16' halves the frozen-weight mirror
    // and is the default (2026-07-29); the engine falls back to 'f32' itself
    // on a non-CUDA backend, so an explicit 'f32' remains the opt-out.
    // 'bf16-f32' (2026-09-02) stores like 'bf16' and computes like 'f32' — see
    // the trainingApi.ts doc comment; it is the Training Studio's default now.
    if (body.mirror !== undefined && body.mirror !== 'f32' && body.mirror !== 'bf16' &&
        body.mirror !== 'bf16-f32') {
      res.status(400).json({ error: 'mirror must be f32, bf16 or bf16-f32' });
      return;
    }
    // Same rule for the MUL_MAT activation-gradient formulation: refused, not
    // coerced. Default is 'mm' (engine/patches/mm-backward.patch), not
    // ace-train's own 'outprod'.
    if (body.optimizer !== undefined && body.optimizer !== 'adamw' && body.optimizer !== 'muon'
        && body.optimizer !== 'prodigy') {
      res.status(400).json({ error: 'optimizer must be adamw, muon or prodigy' });
      return;
    }
    if (body.bwd !== undefined && body.bwd !== 'outprod' && body.bwd !== 'mm') {
      res.status(400).json({ error: 'bwd must be outprod or mm' });
      return;
    }
    // Attention backend (2026-09-01 flash-attn-backward plan §11). Refused, not
    // coerced, same rule as mirror/optimizer/bwd above. Default stays 'exact' —
    // the byte-identical dit_attn_f32 graph — until a caller opts into 'flash'.
    // 'flash-f32' is the API-only strict-f32 variant of 'flash' (the scalar
    // kernels instead of the TF32 tensor-core ones); the UI never sends it.
    if (body.attnBackend !== undefined && body.attnBackend !== 'exact' && body.attnBackend !== 'flash' &&
        body.attnBackend !== 'flash-f32') {
      res.status(400).json({ error: 'attnBackend must be exact, flash or flash-f32' });
      return;
    }
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) {
      res.status(400).json({ error: 'learningRate must be greater than 0 and at most 1' });
      return;
    }
    // Exclusive lower bound: sigma 0 makes the logit-normal timestep sampler
    // degenerate to a single t, which trains one point of the schedule (D12).
    if (!Number.isFinite(timestepSigma) || timestepSigma <= 0 || timestepSigma > 4) {
      res.status(400).json({ error: 'timestepSigma must be greater than 0 and at most 4' });
      return;
    }
    // An empty interval makes dit_sample_t's rejection loop exhaust its 64 tries
    // on every micro-step and clamp — silently training one timestep.
    if (tMin >= tMax) {
      res.status(400).json({ error: 'tMin must be less than tMax' });
      return;
    }

    // ── 8. stages ────────────────────────────────────────────────────────
    const requestedStages = Array.isArray(body.stages) ? body.stages : [];
    const stages = TRAIN_DIT_STAGES.filter(s => requestedStages.includes(s));
    const resolvedStages: TrainDitStage[] = stages.length > 0 ? [...stages] : [...TRAIN_DIT_STAGES];

    // ── 8b. resume (--init-adapter, 2026-08-11) — same rules as train-lm:
    // 'latest' resolves to the newest non-calibrated run of this adapter name
    // (bakes excluded — their factors carry the baked scale), and any explicit
    // path must hold weights BEFORE the runner stops the engine.
    // AN OMITTED initAdapter MEANS 'latest' (Rob, 2026-08-13), matching the
    // form's now-ticked Continue checkbox and the train-lm route's 2026-08-12
    // flip. The batch pipeline POSTs `{}`, so without this a bulk re-run over
    // already-trained datasets would restart every adapter from scratch while
    // the identical manual run continued it.
    let ditInitAdapter = typeof body.initAdapter === 'string' ? body.initAdapter.trim() : 'latest';
    if (ditInitAdapter === 'latest') {
      const artistDir = path.dirname(adapterDir);
      let newest = '';
      try {
        const runs = fs.readdirSync(artistDir, { withFileTypes: true })
          .filter(e => e.isDirectory() && !/-calibrated$/i.test(e.name)
            && hasWeights(path.join(artistDir, e.name)))
          .map(e => e.name)
          .sort();
        if (runs.length) newest = path.join(artistDir, runs[runs.length - 1]);
      } catch { /* artist dir may not exist yet */ }
      if (!newest && hasWeights(artistDir)) newest = artistDir;
      // "Nothing to resume" is NOT an error any more — same reasoning as
      // train-lm: now that resume is the default for both the form and the
      // pipeline, a first-ever run for an adapter name would fail on the one
      // thing it cannot possibly satisfy. Fall through to a scratch run and
      // say so in the log, so a bulk sweep can mix new and existing datasets.
      if (!newest) {
        console.log(
          `[Training] train-dit: nothing to resume for "${adapterName}" — training from scratch`);
      }
      ditInitAdapter = newest;
    }
    if (ditInitAdapter && !hasWeights(ditInitAdapter)) {
      res.status(400).json({ error: `initAdapter ${ditInitAdapter} holds no adapter weights` });
      return;
    }

    // ── 9. queue ─────────────────────────────────────────────────────────
    // No VRAM gating here (§4.5): only ace-train knows the mirror size, and only
    // after the base is loaded with the engine already stopped.
    // capabilities.trainDit.minVramMb is advisory for the UI banner alone — and
    // since the engine's flat 16 GB refusal was retired (2026-09-02) it is not
    // even a floor, just a hint. ace-train's per-run footprint solve refuses,
    // and its message names the settings that would make the run fit.
    const opts: ResolvedTrainDitOptions = {
      variantKey,
      tensorsDir: tensorsPath,
      ditModel,
      ditPath,
      adapterName,
      adapterDir,
      adapterType,
      rank: Math.trunc(rank),
      alpha: Math.trunc(alpha),
      lokrDim: Math.trunc(lokrDim),
      lokrAlpha,
      lokrFactor: Math.trunc(lokrFactor),
      lokrDecomposeBoth,
      // Default ON: an attention-only DiT LoRA leaves the MLP projections —
      // where most of the timbre lives — frozen. Same !== false shape as
      // channelBalance/stopEngine, so an omitting client gets the default.
      targetMlp: body.targetMlp !== false,
      dora: body.dora === true,
      hira: body.hira === true,
      loha: body.loha === true,
      pissa: body.pissa === true,
      hra: body.hra === true,
      rslora: body.rslora === true,
      loraPlusRatio: numOpt(body.loraPlusRatio, 1),
      layers: Math.trunc(layers),
      crop: Math.trunc(crop),
      cropMin: Math.trunc(cropMin),
      cropMax: Math.trunc(cropMax),
      cropAnchor,
      cropMode,
      cropStartFrac,
      cropEndFrac,
      cropJitter,
      targetLoss,
      epochs: Math.trunc(epochs),
      learningRate,
      gradAccum: Math.trunc(gradAccum),
      gradClip,
      warmupRatio,
      weightDecay,
      // K2: omitted-field fallback is 'none' for lokr, 'flow_snr' for lora —
      // an explicit value from the client always wins either way.
      lossWeighting: body.lossWeighting === 'none' ? 'none'
        : body.lossWeighting === 'flow_snr' ? 'flow_snr'
          : (isLokr ? 'none' : 'flow_snr'),
      snrGamma,
      tBias,
      channelBalance: body.channelBalance !== false,
      timestepMu,
      timestepSigma,
      tMin,
      tMax,
      cfgRatio,
      genreRatio: Math.trunc(genreRatio),
      seed: Math.trunc(seed),
      order: body.order === 'fixed' ? 'fixed' : 'shuffle',
      milestoneStep,
      milestoneKeep: Math.trunc(milestoneKeep),
      vramReserveMb: Math.trunc(vramReserveMb),
      // Frozen-weight mirror precision. Default is 'bf16' (2026-07-29); only
      // the exact string 'f32' opts back out. The engine itself falls back to
      // f32 with a warning on a non-CUDA backend (dit-train-run.h).
      // Default 'bf16-f32' (Rob, 2026-09-02): bf16 storage, f32 compute — the
      // f32 mirror's numerics (measured bit-identical over 12 epochs) at the
      // bf16 mirror's VRAM, so the flash-mode crop stays long. History: f32 was
      // Rob's 2026-07-30 call for precision; 'bf16' became the default on
      // 2026-09-01 for VRAM, but it also ran the GEMMs in bf16 and adapters
      // trained that way rendered coarse and "bitty". 'bf16' is reachable only
      // by name now; callers that omit the field (batch pipeline, stored
      // presets) get the clean mode.
      mirror: body.mirror === 'f32' ? 'f32' : body.mirror === 'bf16' ? 'bf16' : 'bf16-f32',
      // MUL_MAT activation-gradient formulation. Default 'mm' (2026-07-29);
      // only the exact string 'outprod' opts back out to upstream ggml's
      // F32-only out_prod backward.
      bwd: body.bwd === 'outprod' ? 'outprod' : 'mm',
      // DEFAULT MUON (2026-07-30, after the ear test). Only the exact string
      // 'adamw' opts back out. Measured on gunship_unicorn: 161 epochs to ma5
      // 0.6 vs AdamW's 227, and with bucketing that is ~1.23x on wall-clock —
      // Rob's own run reached 0.6 in ~5 minutes and the adapter was judged
      // perfect by ear, which is what made Muon a default rather than a flag.
      // Prodigy replaced it as the default on 2026-09-03 (Rob): the step size
      // is estimated online, so there is no per-artist LR to tune. Muon and
      // AdamW remain one select away.
      optimizer: body.optimizer === 'adamw' ? 'adamw' : body.optimizer === 'muon' ? 'muon' : 'prodigy',
      muonLrScale,
      muonMomentum,
      muonNsSteps: Math.trunc(muonNsSteps),
      muonMinDim: Math.trunc(muonMinDim),
      batch: Math.trunc(batch),
      ckptSegments: Math.trunc(ckptSegments),
      stages: resolvedStages,
      overwrite: body.overwrite === true,
      stopEngine: body.stopEngine !== false,
      initAdapter: ditInitAdapter,
      // Calibration is OPT-IN (Rob, 2026-08-13) — it was default ON from
      // 2026-08-11. Only an explicit `true` runs it, so the batch pipeline's
      // empty bag no longer appends an eval pass to every adapter in a sweep.
      calibrate: body.calibrate === true,
      calibrateRepoint: body.calibrateRepoint !== false,
      attnBackend,
    };

    const job = queue.startTrainDitJob(ds.id, opts);
    const adapterDesc = isLokr ? `lokr dim${opts.lokrDim}` : `lora r${opts.rank}`;
    console.log(
      `[Training] train-dit job ${job.id} queued — ${adapterDesc}, variant ${variantKey} → ${adapterDir}`);
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] train-dit start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/datasets/:id/train-dit', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    res.json(readTrainDitStatus(ds, {
      variantKey: typeof req.query.variantKey === 'string' ? req.query.variantKey : undefined,
      adapterName: typeof req.query.adapterName === 'string' ? req.query.adapterName : undefined,
    }));
  } catch (err: any) {
    console.error(`[Training] train-dit status failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Codes audition (codes-preview §3.2) ──────────────────────────────────

const AUDITION_SLOTS: readonly string[] = ['base', 'adapter'];

/**
 * C16 — stricter than the Create panel, which accepts any absolute path.
 *
 * An lmAdapter is either '' (base LM), a bare registry name, or a directory
 * INSIDE adapters/lm that actually holds an adapter_model.safetensors. The third
 * clause is what lets a milestone dir
 * (`<adapters>/lm/<name>-<size>/milestones/loss_<v>`) through while refusing an
 * arbitrary absolute path. A Training-Studio route has no reason to be as
 * permissive as the Create panel.
 */
function auditionAdapterError(value: string): string | null {
  if (value === '') return null;
  // The regex alone accepts '.', '..', '...' and '-'. Both other adapter-name
  // validators in this file (train-lm and train-dit) pair it with the
  // safeAdapterName cross-check for exactly that reason; being the one looser
  // validator in the file is how a future change to the engine's
  // resolve_lm_adapter_path heuristic turns into a real traversal. Today it is
  // contained — the engine only takes its path-fallback branch when the value
  // holds a '/' or '\' — but C16 is this plan's stated security clause and it
  // should not be the weakest of the three.
  if (ADAPTER_NAME_RE.test(value)) {
    return safeAdapterName(value) === value
      ? null
      : 'lmAdapter must match [A-Za-z0-9._-]{1,64} and cannot start with a dot';
  }
  // Any planner-adapter root counts: the per-size lm-* dirs plus the legacy
  // flat lm/ (adapterLayout.ts). Milestone dirs inside an adapter dir pass too.
  const roots = lmAdapterRoots().map(r => r.dir);
  const resolved = path.resolve(value);
  const insideSome = roots.some(r => isInside(r, resolved) && path.resolve(r) !== resolved);
  if (!insideSome) {
    return `lmAdapter must be a registry name or a directory inside ${roots.join(' | ')}`;
  }
  // A LoKr adapter dir has NO adapter_model.safetensors — its weights live in
  // lokr_weights.safetensors and there is deliberately no adapter_config.json
  // (alpha rides the per-module tensors + __metadata__.lokr_config). Checking
  // only the PEFT name rejected every LoKr adapter before the engine ever saw
  // it; lm-adapter.h reads both layouts.
  if (!hasWeights(resolved)) {
    return `lmAdapter has no adapter_model.safetensors or lokr_weights.safetensors: ${resolved}`;
  }
  return null;
}

/**
 * The linked Lyric Studio album's generated lyrics, as audition prompts.
 *
 * Link resolution: the persisted `lyrics_set_id` on the dataset row (written by
 * the export commit) wins; a never-linked dataset falls back to the export
 * preview's artist/album detection (tag majority vote), and a successful
 * detection is persisted back onto the row so the scan only ever runs once.
 */
router.get('/datasets/:id/ls-generations', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }

    let lyricsSetId = ds.lyricsSetId ?? 0;
    // A stale link (album deleted in Lyric Studio) degrades to re-detection.
    if (lyricsSetId > 0 && !getLyricsSet(lyricsSetId)) lyricsSetId = 0;
    if (lyricsSetId <= 0) {
      const samples = await buildSamples(ds);
      const preview = await previewLyricStudioExport(ds, samples);
      lyricsSetId = preview.existingLyricsSetId ?? 0;
      if (lyricsSetId > 0) {
        try { repo.updateDataset(ds.id, { lyricsSetId }); } catch { /* lazy backfill only */ }
      }
    }

    if (lyricsSetId <= 0) {
      res.json({ lyricsSetId: 0, artist: '', album: '', generations: [] });
      return;
    }

    const set = getLyricsSet(lyricsSetId);
    const gens = getGenerations(undefined, lyricsSetId).map(g => ({
      id: Number(g.id),
      title: String(g.title ?? ''),
      caption: String(g.caption ?? ''),
      lyrics: String(g.lyrics ?? ''),
      bpm: Math.trunc(Number(g.bpm) || 0),
      key: String(g.key ?? ''),
      duration: Math.trunc(Number(g.duration) || 0),
      createdAt: String(g.created_at ?? ''),
    }));
    res.json({
      lyricsSetId,
      artist: String(set?.artist_name ?? ''),
      album: String(set?.album ?? ''),
      generations: gens,
    });
  } catch (err: any) {
    console.error(`[Training] ls-generations failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.post('/datasets/:id/audition', async (req: Request, res: Response) => {
  try {
    const body = (req.body || {}) as AuditionOptions;

    // A fresh server boot has an EMPTY model snapshot, and the audition's
    // adapter-base derivation reads it (pickLmFor) — twice on 2026-08-29 a
    // valid job died instantly with "needs a 4B LM base, but none is
    // installed" because nothing had refreshed the cache yet. Same pre-flight
    // the preprocess/train routes already do.
    if (!getModelSnapshot().cachedAt && !isEngineSuspended()) await refreshModelSnapshot();

    // ── 1. dataset + queue ───────────────────────────────────────────────
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }

    // ── 2. the engine must be UP — the opposite of every ace-train job ───
    if (isEngineSuspended()) {
      res.status(503).json({
        error: 'Engine is stopped for a training run — audition needs ace-server up',
      });
      return;
    }
    // Suspended and down are different things, and this feature's own list
    // response already reports them separately (engineReady, engineSuspended) so
    // the client can tell them apart. Without this check a crashed ace-server
    // whose 3 s respawn has not landed yields a 202 + jobId, and ~15 s later the
    // job dies carrying the bare string 'fetch failed' — an error the user
    // cannot act on, where an immediate 503 is actionable.
    if (!engineReady) {
      res.status(503).json({
        error: 'Engine is not running — audition needs ace-server up',
      });
      return;
    }

    // ── 3. caption ───────────────────────────────────────────────────────
    // An EMPTY caption is legal when a sampleId is supplied, and that is not a
    // loophole — it is the only way §5.3's documented fallback can ever fire.
    // C12(a) says a sample-sourced audition must use "the literal strings the
    // trainer conditioned on", which is the lm_codes.jsonl row's caption, i.e.
    // lm_apply_tag(sidecar caption, custom_tag, position) — the TAGGED string
    // carrying the trigger word. The dataset sidecar caption the client can see
    // is the UNTAGGED one; sending it would run the adapter's prompt without its
    // trigger, both sides would emit near-identical plans, and the UI would
    // report "the adapter had no effect" — the exact misattribution C14 exists
    // to prevent. With an unconditional 400 here, resolveAuditionInputs' row
    // fallback was dead code for this route.
    const caption = typeof body.caption === 'string' ? body.caption.trim() : '';
    const hasSampleId = typeof body.sampleId === 'string' && body.sampleId.trim() !== '';
    if (!caption && !hasSampleId) {
      res.status(400).json({ error: 'caption is required' });
      return;
    }
    if (caption.length > 4000) {
      res.status(400).json({ error: 'caption must be at most 4000 characters' });
      return;
    }

    // ── 4. sides ─────────────────────────────────────────────────────────
    const rawSides = Array.isArray(body.sides) ? body.sides : [];
    if (rawSides.length < 1 || rawSides.length > 2) {
      res.status(400).json({ error: 'sides must hold 1 or 2 entries' });
      return;
    }
    const seenSlots = new Set<string>();
    const sides: AuditionSideSpec[] = [];
    for (const raw of rawSides) {
      const slot = typeof raw?.slot === 'string' ? raw.slot : '';
      if (!AUDITION_SLOTS.includes(slot)) {
        res.status(400).json({ error: "every side needs a slot of 'base' or 'adapter'" });
        return;
      }
      if (seenSlots.has(slot)) {
        res.status(400).json({ error: 'sides must have distinct slots' });
        return;
      }
      seenSlots.add(slot);

      const label = typeof raw?.label === 'string' ? raw.label : '';
      if (label.length > 64) {
        res.status(400).json({ error: 'side label must be at most 64 characters' });
        return;
      }

      // ── 5. adapter path safety (C16) ───────────────────────────────────
      const lmAdapter = typeof raw?.lmAdapter === 'string' ? raw.lmAdapter.trim() : '';
      const adapterFailure = auditionAdapterError(lmAdapter);
      if (adapterFailure) {
        res.status(400).json({ error: adapterFailure });
        return;
      }

      sides.push({
        slot: slot as AuditionSideSpec['slot'],
        label,
        lmAdapter,
        lmAdapterScale: numOpt(raw?.lmAdapterScale, 1.0),
      });
    }

    // ── 6. variantKey ────────────────────────────────────────────────────
    const requestedVariant = typeof body.variantKey === 'string' ? body.variantKey.trim() : '';
    if (requestedVariant && !variantExists(ds.slug, requestedVariant)) {
      res.status(400).json({ error: `Unknown preprocess variant: ${requestedVariant}` });
      return;
    }

    // ── 6b. DiT-adapter render name — same bar as the train-dit adapterName
    // validator: the regex plus the safeAdapterName cross-check, because this
    // string reaches path.join inside adapterDitDirFor. The resolved dir is
    // then server-derived (never a client path), so this is the whole surface.
    const rdaName = typeof (body as { renderDitAdapterName?: unknown }).renderDitAdapterName === 'string'
      ? (body.renderDitAdapterName as string).trim()
      : '';
    if (rdaName && (!ADAPTER_NAME_RE.test(rdaName) || safeAdapterName(rdaName) !== rdaName)) {
      res.status(400).json({
        error: 'renderDitAdapterName must match [A-Za-z0-9._-]{1,64} and cannot start with a dot',
      });
      return;
    }

    // ── 7. numeric fields clamp silently in resolveAuditionInputs (§3.2.7).
    // They never 400: an out-of-range temperature is a slider mishap, not a
    // reason to refuse an audition.
    const opts: AuditionOptions = {
      ...body,
      caption,
      sides,
      variantKey: requestedVariant || undefined,
    };

    const job = queue.startAuditionJob(ds.id, opts);
    console.log(
      `[Training] audition job ${job.id} queued — ${sides.length} side(s), dataset ${ds.slug}`);
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] audition start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/datasets/:id/audition', (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    // Lazy prune (C11) — there is deliberately no boot hook.
    prunePreviews();

    const limit = numOpt(req.query.limit, 20);
    const payload: AuditionListResponse = {
      previews: listPreviews(ds.id, limit),
      engineReady,
      engineSuspended: isEngineSuspended(),
    };
    res.json(payload);
  } catch (err: any) {
    console.error(`[Training] audition list failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

/**
 * SYNCHRONOUS (C7) — stored codes straight through the detokenizer + VAE. There
 * is no /lm call, which is exactly why this is fast enough not to be a job.
 * If gate V4 ever measures a warm total over 10 s, promote it to the 'audition'
 * job kind (the runner already handles a single-sided decode-only job).
 */
router.post('/datasets/:id/samples/:sampleId/audition', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (isEngineSuspended()) {
      res.status(503).json({
        error: 'Engine is stopped for a training run — audition needs ace-server up',
      });
      return;
    }

    const samples = await buildSamples(ds);
    const sample = samples.find(s => s.sampleId === req.params.sampleId);
    if (!sample) {
      res.status(404).json({ error: 'Sample not found' });
      return;
    }

    const format = (req.body || {}).format === 'mp3' ? 'mp3' : 'wav16';
    res.json(await decodeStoredCodes(ds, sample.sampleId, format));
  } catch (err: any) {
    if (err instanceof AuditionError) {
      res.status(err.status).json({ error: err.message });
      return;
    }
    const message = err?.message || String(err);
    // aceClient.codesDecode throws these two by exact text.
    if (/timed out/i.test(message)) {
      res.status(504).json({ error: 'Decode timed out after 90s' });
      return;
    }
    if (/fetch failed|ECONNREFUSED|unreachable|aborted/i.test(message)) {
      res.status(503).json({ error: `Engine unreachable: ${message}` });
      return;
    }
    console.error(`[Training] sample audition failed: ${message}`);
    res.status(500).json({ error: message });
  }
});

export default router;
