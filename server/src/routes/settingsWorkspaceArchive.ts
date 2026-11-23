// settingsWorkspaceArchive.ts — Workspace archive export and import (GH #228).
// Mounted by routes/settings.ts at /api/settings/workspace-archive.
//
//   GET  /preview[?destination=<abs folder>]  bytes per category, free space, blockers
//   POST /export  { destination, categories? }  hold writes, then 202 { job }
//   GET  /export                                 { job } (null before the first export)
//   POST /export/cancel                          { job }; passes the write hold
//   POST /import/preview  { zip, target? }       read the archive's manifest, check the target
//   POST /import  { zip, target }                202 { job }: extract into target, then switch to it
//   GET  /import                                 { job } (null before the first import)
//   POST /import/cancel                          { job }; removes the half-written folder
//
// One export or import at a time. While an export runs every other mutating
// request gets 503. An import writes only into its new folder; when it is done
// it schedules a repoint to that folder and restarts the app.

import { Router, type Request, type Response } from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { APP_VERSION, config } from '../config.js';
import { getBusyBlockers, getDataLocation, previewOperation, scheduleOperation } from '../dataroot/pointer.js';
import { locateData, type DataChildren } from '../dataroot/locate.js';
import { instanceId } from '../services/instance.js';
import { maintenance, type Gate } from '../services/updates/maintenance.js';
import { ExportCancelled, exportPreflight, exportWorkspace, otherLiveServers, type ExportProgress } from '../services/workspaceArchive/export.js';
import { destinationErrors, freeBytes, planSources, takeInventory } from '../services/workspaceArchive/inventory.js';
import { ImportCancelled, importPreview, importWorkspace, type ImportHere, type ImportPreview, type ImportProgress, type ImportReport } from '../services/workspaceArchive/import.js';
import { requestWorkerRestart } from './shutdown.js';
import { CATEGORY_IDS, DEFAULT_CHOICE, type CategoryChoice, type CategoryId, type SkippedEntry } from '../services/workspaceArchive/manifest.js';

export const ARCHIVE_BASE = '/api/settings/workspace-archive';

export interface ArchivePreview {
  dataRoot: string | null;
  categories: Record<CategoryId, { files: number; bytes: number; paths: string[] }>;
  defaults: CategoryChoice;
  destination: { path: string; freeBytes: number | null; errors: string[] } | null;
  skipped: SkippedEntry[];
  skippedNotListed: number;
  blockers: string[];
}

export interface ArchiveJob {
  id: string;
  state: 'holding' | 'running' | 'done' | 'failed' | 'cancelled';
  outFile: string;
  categories: CategoryChoice;
  progress: ExportProgress | null;
  /** Set when done. */
  bytes: number | null;
  skipped: number;
  externals: number;
  error: string | null;
  blockers: string[];
  startedAt: string;
  finishedAt: string | null;
}

export interface ImportJob {
  id: string;
  state: 'running' | 'done' | 'failed' | 'cancelled';
  zipFile: string;
  target: string;
  progress: ImportProgress | null;
  /** Set when done: the folder is complete and the switch is scheduled. */
  report: ImportReport | null;
  error: string | null;
  errors: string[];
  startedAt: string;
  finishedAt: string | null;
}

export type ImportPreviewResponse = ImportPreview & { blockers: string[] };

export interface ImportDeps {
  here(): ImportHere;
  /** Why the app could not repoint to `target` once it holds data (empty when it can). */
  targetErrors(target: string): string[];
  blockers(): string[];
  /** Schedule the repoint and restart. Throws only when nothing was scheduled. */
  schedule(target: string): void;
}

interface Deps {
  locate(): { root: string | null; children: DataChildren };
  blockers(): string[];
  gate: Pick<Gate, 'hold'>;
  appVersion: string;
  now?: () => Date;
  imports?: ImportDeps;
}

const stamp = (d: Date) => d.toISOString().slice(0, 19).replace(/[:T]/g, '-');

function parseChoice(v: unknown): CategoryChoice {
  if (v === undefined) return { ...DEFAULT_CHOICE };
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw Object.assign(new Error('categories must be an object of booleans'), { status: 400 });
  const out = { ...DEFAULT_CHOICE };
  for (const [k, b] of Object.entries(v)) {
    if (k === 'library' && b === true) continue;
    if (!(k in out) || typeof b !== 'boolean') throw Object.assign(new Error(`categories.${k} is not a choice (training, adapters, models, secrets: true or false)`), { status: 400 });
    out[k as keyof CategoryChoice] = b;
  }
  return out;
}

export function createWorkspaceArchiveRouter(deps: Deps): Router {
  const router = Router();
  const now = deps.now ?? (() => new Date());
  let job: ArchiveJob | null = null;
  let controller: AbortController | null = null;
  const active = () => job?.state === 'holding' || job?.state === 'running';
  let importJob: ImportJob | null = null;
  let importController: AbortController | null = null;
  const importing = () => importJob?.state === 'running';
  const fail = (res: Response, e: any) => {
    const x = e.extra ?? e;
    res.status(e.status >= 400 && e.status < 600 ? e.status : 500).json({ error: e.message, blockers: x.blockers, errors: x.errors });
  };

  router.get('/preview', (req: Request, res: Response) => {
    try {
      const { root, children } = deps.locate();
      const all: CategoryChoice = { training: true, adapters: true, models: true, secrets: true };
      const inv = takeInventory(children, all);
      const categories = Object.fromEntries(CATEGORY_IDS.map(k => [k, {
        ...inv.totals[k],
        paths: k === 'secrets' ? [] : inv.sources.filter(s => s.category === k).map(s => s.dir),
      }])) as ArchivePreview['categories'];
      if (fs.existsSync(children.db)) { categories.library.files++; categories.library.bytes += fs.statSync(children.db).size; }
      const dest = typeof req.query.destination === 'string' && req.query.destination.trim() ? req.query.destination.trim() : null;
      const body: ArchivePreview = {
        dataRoot: root, categories, defaults: { ...DEFAULT_CHOICE },
        destination: dest ? { path: dest, freeBytes: path.isAbsolute(dest) ? freeBytes(dest) : null, errors: destinationErrors(dest, planSources(children, all)) } : null,
        skipped: inv.skipped, skippedNotListed: inv.skippedNotListed,
        blockers: deps.blockers(),
      };
      res.json(body);
    } catch (e) { fail(res, e); }
  });

  router.get('/export', (_req, res) => res.json({ job }));

  router.post('/export', async (req: Request, res: Response) => {
    if (active()) { res.status(409).json({ error: 'An export is already running', job }); return; }
    if (importing()) { res.status(409).json({ error: 'An import is running', job: importJob }); return; }
    let choice: CategoryChoice;
    const destination = typeof req.body?.destination === 'string' ? req.body.destination.trim() : '';
    try { choice = parseChoice(req.body?.categories); } catch (e) { fail(res, e); return; }
    if (!path.isAbsolute(destination)) { res.status(400).json({ error: 'destination must be an absolute folder path' }); return; }
    const { root, children } = deps.locate();
    const started = now();
    const outFile = path.join(destination, `HOT-Step-workspace-${stamp(started)}.zip`);
    const errors = exportPreflight(outFile, children, choice);
    if (errors.length) { res.status(400).json({ error: errors.join('; '), errors }); return; }

    const mine: ArchiveJob = {
      id: `${started.getTime()}`, state: 'holding', outFile, categories: choice, progress: null, bytes: null,
      skipped: 0, externals: 0, error: null, blockers: [], startedAt: started.toISOString(), finishedAt: null,
    };
    job = mine;
    const ac = new AbortController();
    controller = ac;
    let held: Awaited<ReturnType<Gate['hold']>>;
    try {
      held = await deps.gate.hold({ blockers: deps.blockers, allow: [`${ARCHIVE_BASE}/export/cancel`], fromRequest: true });
    } catch (e: any) {
      Object.assign(mine, { state: 'failed', error: e.message, blockers: e.blockers ?? [], finishedAt: now().toISOString() });
      fail(res, e);
      return;
    }
    if (ac.signal.aborted) {
      held.release();
      Object.assign(mine, { state: 'cancelled', finishedAt: now().toISOString() });
      res.status(409).json({ error: 'Export cancelled', job: mine });
      return;
    }
    mine.state = 'running';
    res.status(202).json({ job: mine });

    exportWorkspace({
      children, dataRoot: root, outFile, choice, appVersion: deps.appVersion,
      blockers: held.check, signal: ac.signal, onProgress: p => { mine.progress = p; },
    }).then(result => {
      Object.assign(mine, {
        state: 'done', bytes: result.bytes, skipped: result.manifest.skipped.length + result.manifest.skippedNotListed,
        externals: result.manifest.externals.length,
      });
    }).catch((e: any) => {
      Object.assign(mine, e instanceof ExportCancelled ? { state: 'cancelled' } : { state: 'failed', error: e.message, blockers: e.extra?.blockers ?? [] });
      if (!(e instanceof ExportCancelled)) console.error('[WorkspaceArchive] export failed:', e.message);
    }).finally(() => {
      held.release();
      mine.finishedAt = now().toISOString();
      if (controller === ac) controller = null;
    });
  });

  router.post('/export/cancel', (_req, res) => {
    if (!active() || !controller) { res.status(409).json({ error: 'No export is running', job }); return; }
    controller.abort();
    res.json({ job });
  });

  const imp = deps.imports;
  if (imp) {
    const body = (req: Request) => ({
      zip: typeof req.body?.zip === 'string' ? req.body.zip.trim() : '',
      target: typeof req.body?.target === 'string' && req.body.target.trim() ? req.body.target.trim() : null,
    });

    router.post('/import/preview', async (req: Request, res: Response) => {
      const { zip, target } = body(req);
      if (!path.isAbsolute(zip)) { res.status(400).json({ error: 'zip must be an absolute path to a workspace archive' }); return; }
      try {
        const out: ImportPreviewResponse = { ...await importPreview(zip, target, imp.here(), imp.targetErrors), blockers: imp.blockers() };
        res.json(out);
      } catch (e) { fail(res, e); }
    });

    router.get('/import', (_req, res) => res.json({ job: importJob }));

    router.post('/import', async (req: Request, res: Response) => {
      if (importing()) { res.status(409).json({ error: 'An import is already running', job: importJob }); return; }
      if (active()) { res.status(409).json({ error: 'An export is running', job }); return; }
      const { zip, target } = body(req);
      if (!path.isAbsolute(zip) || !target || !path.isAbsolute(target)) {
        res.status(400).json({ error: 'zip and target must be absolute paths' }); return;
      }
      let here: ImportHere;
      let pre: ImportPreview;
      try { here = imp.here(); pre = await importPreview(zip, target, here, imp.targetErrors); } catch (e) { fail(res, e); return; }
      const errors = pre.target?.errors ?? [];
      if (errors.length) { res.status(400).json({ error: errors.join('; '), errors }); return; }
      const blockers = imp.blockers();
      if (blockers.length) { res.status(409).json({ error: `Wait for running work to finish: ${blockers.join('; ')}`, blockers }); return; }
      // The preview awaited: another request may have started one meanwhile.
      if (importing() || active()) { res.status(409).json({ error: 'An export or import is already running', job: importJob }); return; }

      const started = now();
      const mine: ImportJob = {
        id: `${started.getTime()}`, state: 'running', zipFile: zip, target, progress: null, report: null,
        error: null, errors: [], startedAt: started.toISOString(), finishedAt: null,
      };
      importJob = mine;
      const ac = new AbortController();
      importController = ac;
      res.status(202).json({ job: mine });

      importWorkspace({
        zipFile: zip, target, here, targetErrors: imp.targetErrors, blockers: imp.blockers, signal: ac.signal,
        onProgress: p => { mine.progress = p; },
        schedule: t => imp.schedule(t),
      }).then(report => {
        Object.assign(mine, { state: 'done', report });
      }).catch((e: any) => {
        Object.assign(mine, e instanceof ImportCancelled ? { state: 'cancelled' } : { state: 'failed', error: e.message, errors: e.extra?.errors ?? e.errors ?? [] });
        if (!(e instanceof ImportCancelled)) console.error('[WorkspaceArchive] import failed:', e.message);
      }).finally(() => {
        mine.finishedAt = now().toISOString();
        if (importController === ac) importController = null;
      });
    });

    router.post('/import/cancel', (_req, res) => {
      if (!importing() || !importController) { res.status(409).json({ error: 'No import is running', job: importJob }); return; }
      importController.abort();
      res.json({ job: importJob });
    });
  }

  return router;
}

export default createWorkspaceArchiveRouter({
  locate: () => {
    const d = locateData();
    if (!d.children) throw Object.assign(new Error('This install has no data folder yet'), { status: 400 });
    return { root: d.root, children: d.children };
  },
  blockers: () => [...getBusyBlockers(), ...otherLiveServers(locateData().children?.db ?? '', instanceId())],
  gate: maintenance,
  appVersion: APP_VERSION,
  imports: {
    here: () => {
      const d = locateData();
      if (!d.children) throw Object.assign(new Error('This install has no data folder yet'), { status: 400 });
      return {
        appRoot: d.installRoot, dataRoot: d.root, children: d.children,
        paths: { WHISPER_EXE: config.whisper.exe, WHISPER_MODELS_DIR: config.whisper.modelsDir, TENSORRT_LIBS: config.aceServer.trtLibs },
      };
    },
    // The folder holds no data until the import finishes, so the repoint's
    // "not a data folder" check cannot pass yet; every other check applies.
    // ponytail: matched by message text; give planOperation error codes if this wording changes.
    targetErrors: t => [
      ...(getDataLocation().pending ? ['a data-folder change is already scheduled; restart to finish it first'] : []),
      ...previewOperation('repoint', t, config.data.dbPath).errors.filter(e => !e.endsWith('is not a HOT-Step data folder')),
    ],
    blockers: getBusyBlockers,
    schedule: t => {
      scheduleOperation('repoint', t, config.data.dbPath);
      // Scheduled: from here a failure must not remove the folder the pointer names.
      try { requestWorkerRestart(); } catch (e: any) { console.error(`[WorkspaceArchive] switch scheduled, restart failed (${e.message}); restart the app to finish`); }
    },
  },
});
