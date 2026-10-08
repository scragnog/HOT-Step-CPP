// adapters.ts — Adapter filesystem browsing and scanning
//
// Provides server-side endpoints for:
//   1. GET /browse — directory navigation with file type filtering
//   2. POST /scan  — flat listing of .safetensors files in a folder

import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { config } from '../config.js';
import { readAdapterTrigger } from '../services/adapters/stMetadata.js';
import { lmAdapterRoots } from '../services/training/adapterLayout.js';
import type { BrowseResponse, BrowseErrorResponse, ScanResponse, LmAdapterEntry, LmAdaptersResponse } from '../contracts/adapters.js';

const router = Router();

/** Extensions accepted per filter category */
const FILTER_EXTENSIONS: Record<string, string[]> = {
  adapters: ['.safetensors'],
  audio: ['.wav', '.mp3', '.flac', '.ogg', '.opus'],
  // Training Studio folder picker — deliberately separate from `audio`, which
  // other callers depend on staying as-is.
  trainingAudio: ['.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.aac'],
};

/**
 * GET /api/adapters/browse?path=...&filter=adapters
 *
 * Lists the contents of a directory, returning sub-directories and
 * files that match the optional filter.  Always includes a '..'
 * parent entry unless already at a filesystem root.
 *
 * Response: { current: string, entries: BrowseEntry[] }
 */
router.get('/browse', (req, res) => {
  const rawPath = (req.query.path as string) || '';
  const filter = (req.query.filter as string) || '';
  const allowedExts = FILTER_EXTENSIONS[filter] || [];

  // Resolve to an absolute path
  let dirPath: string;
  try {
    dirPath = rawPath ? path.resolve(rawPath) : path.resolve('.');
  } catch {
    res.status(400).json({ error: 'Invalid path' });
    return;
  }

  // Verify path exists and is a directory
  if (!fs.existsSync(dirPath) || !fs.statSync(dirPath).isDirectory()) {
    res.status(404).json({ error: 'Directory not found', current: dirPath, entries: [] } satisfies BrowseErrorResponse);
    return;
  }

  try {
    const rawEntries = fs.readdirSync(dirPath, { withFileTypes: true });
    const entries: Array<{ name: string; path: string; type: 'dir' | 'file'; size?: number }> = [];

    // Parent directory (unless at root)
    const parent = path.dirname(dirPath);
    if (parent !== dirPath) {
      entries.push({ name: '..', path: parent, type: 'dir' });
    }

    // Directories first (skip hidden)
    for (const entry of rawEntries) {
      if (entry.name.startsWith('.')) continue;
      const fullPath = path.join(dirPath, entry.name);
      if (entry.isDirectory()) {
        entries.push({ name: entry.name, path: fullPath, type: 'dir' });
      }
    }

    // Then files (filtered, skip hidden)
    for (const entry of rawEntries) {
      if (entry.name.startsWith('.')) continue;
      if (!entry.isFile()) continue;
      const fullPath = path.join(dirPath, entry.name);
      const ext = path.extname(entry.name).toLowerCase();
      if (allowedExts.length === 0 || allowedExts.includes(ext)) {
        try {
          const stat = fs.statSync(fullPath);
          entries.push({ name: entry.name, path: fullPath, type: 'file', size: stat.size });
        } catch {
          // Skip files we can't stat (locked, permissions, etc.)
        }
      }
    }

    res.json({ current: dirPath, entries } satisfies BrowseResponse);
  } catch (err: any) {
    res.status(500).json({ error: err.message, current: dirPath, entries: [] } satisfies BrowseErrorResponse);
  }
});

/**
 * POST /api/adapters/scan
 *
 * Flat scan of a single directory for adapters, in two forms:
 *   * bare `.safetensors` files (the hand-installed convention), and
 *   * PEFT sub-directories — `adapter_model.safetensors` + `adapter_config.json`
 *     — which is what the DiT trainer writes.
 *
 * The PEFT half is what makes a freshly trained adapter appear in the Create
 * view's dropdown WITHOUT an engine restart: `path` is the directory, and the
 * engine's path-fallback resolver already accepts a PEFT dir (adapter-merge.h).
 * Without it a just-finished training run would be invisible until relaunch.
 *
 * Returns an empty array if the folder doesn't exist or is empty.
 *
 * Body: { folder: string }
 * Response: { files: AdapterFile[] }
 */
router.post('/scan', (req, res) => {
  const folder = req.body?.folder;
  if (!folder || typeof folder !== 'string') {
    res.json({ files: [] } satisfies ScanResponse);
    return;
  }

  const dirPath = path.resolve(folder);
  if (!fs.existsSync(dirPath) || !fs.statSync(dirPath).isDirectory()) {
    res.json({ files: [] } satisfies ScanResponse);
    return;
  }

  try {
    const rawEntries = fs.readdirSync(dirPath, { withFileTypes: true });
    const files = rawEntries
      .filter(e => e.isFile() && e.name.endsWith('.safetensors'))
      .map(e => {
        const fullPath = path.join(dirPath, e.name);
        const stat = fs.statSync(fullPath);
        const tg = readAdapterTrigger(fullPath);
        return { name: e.name, path: fullPath, size: stat.size, trigger: tg.trigger, triggerPosition: tg.position };
      });

    // Adapter directories, mirroring GET /adapters/lm. Only the weights file
    // is required: PEFT dirs from our trainers always carry adapter_config.json
    // too, but an enfoldered LyCORIS adapter (bare archive file moved into
    // <name>/adapter_model.safetensors) has its alphas per-tensor and no
    // config — hiding it from the dropdown would be worse than listing it.
    //
    // A DiT LoKR export (K8, lokr-dit-training plan §2.4) writes
    // `lokr_weights.safetensors` instead — same out-dir convention, different
    // filename, checked second so an (unexpected) dir with both prefers PEFT.
    const pushPeftDir = (dir: string, displayName: string) => {
      let model = path.join(dir, 'adapter_model.safetensors');
      if (!fs.existsSync(model)) model = path.join(dir, 'lokr_weights.safetensors');
      if (!fs.existsSync(model)) return;
      try {
        const tg = readAdapterTrigger(dir);
        files.push({ name: displayName, path: dir, size: fs.statSync(model).size, trigger: tg.trigger,
                     triggerPosition: tg.position });
      } catch { /* skip */ }
    };
    for (const e of rawEntries) {
      if (!e.isDirectory() || e.name.startsWith('.')) continue;
      const dir = path.join(dirPath, e.name);
      pushPeftDir(dir, e.name);
      // Per-base layout (adapterLayout.ts): DiT adapters live one level down in
      // dit-<shorthand> folders. Descend into those — and only those; lm-* is
      // the planner-adapter tree and arbitrary subfolders are not ours to walk.
      if (!/^dit-/i.test(e.name)) continue;
      try {
        for (const sub of fs.readdirSync(dir, { withFileTypes: true })) {
          if (sub.name.startsWith('.')) continue;
          const subPath = path.join(dir, sub.name);
          if (sub.isDirectory()) {
            // Unversioned adapter directly in the artist dir (legacy)…
            pushPeftDir(subPath, `${e.name}/${sub.name}`);
            // …and every stamped training run beneath it (per-run layout).
            try {
              for (const run of fs.readdirSync(subPath, { withFileTypes: true })) {
                if (!run.isDirectory() || run.name.startsWith('.')) continue;
                pushPeftDir(path.join(subPath, run.name), `${e.name}/${sub.name}/${run.name}`);
              }
            } catch { /* unreadable artist dir — skip its runs */ }
          } else if (sub.isFile() && sub.name.endsWith('.safetensors')) {
            try {
              const stat = fs.statSync(subPath);
              const tg = readAdapterTrigger(subPath);
              files.push({ name: `${e.name}/${sub.name}`, path: subPath, size: stat.size,
                           trigger: tg.trigger, triggerPosition: tg.position });
            } catch { /* skip */ }
          }
        }
      } catch { /* unreadable subdir — skip */ }
    }
    files.sort((a, b) => a.name.localeCompare(b.name));

    res.json({ files } satisfies ScanResponse);
  } catch {
    res.json({ files: [] } satisfies ScanResponse);
  }
});

/**
 * GET /api/adapters/lm?folder=...
 *
 * Lists planner-LM adapters (local HOT-Step feature): PEFT directories
 * (adapter_model.safetensors + adapter_config.json) and bare .safetensors
 * files.  Scans `folder` when given (the user's archive, like the DiT
 * adapter folder), else the adapters root's `lm/` subtree.  Filesystem-based
 * so freshly trained adapters appear WITHOUT an engine restart — the UI
 * sends the absolute path and the engine's path-fallback resolver loads it.
 *
 * Response: { root: string, adapters: { name, path, kind, size, mtime }[] }
 */
router.get('/lm', (req, res) => {
  const folderParam = (req.query.folder as string) || '';
  // LmAdapterEntry is now contracts/adapters.ts's shared definition.
  const adapters: LmAdapterEntry[] = [];

  const readEvalSidecar = (dir: string): { evalScore: number | null; evalVerdict: string } => {
    try {
      const sc = JSON.parse(fs.readFileSync(path.join(dir, 'hot_step_eval.json'), 'utf-8')) as
        { score?: unknown; verdict?: unknown };
      if (typeof sc.score === 'number' && Number.isFinite(sc.score)) {
        return { evalScore: sc.score, evalVerdict: typeof sc.verdict === 'string' ? sc.verdict : '' };
      }
    } catch { /* no sidecar / torn file — not evaluated */ }
    return { evalScore: null, evalVerdict: '' };
  };

  // Both trained dir layouts count: PEFT (adapter_model.safetensors) and the
  // LyCORIS LoKr export ace-train writes (lokr_weights.safetensors). Checking
  // only the PEFT leaf made the ENTIRE LoKr-era corpus (183 adapters,
  // 2026-08) invisible to every picker — the same bug hasWeights() already
  // fixed for latestRunDir, re-fixed here (found 2026-08-09: /api/adapters/lm
  // returned zero adapters).
  const pushPeft = (dir: string, name: string, lmSize: string, run: string) => {
    let model = path.join(dir, 'adapter_model.safetensors');
    let kind: 'peft' | 'lokr' = 'peft';
    if (!fs.existsSync(model)) {
      model = path.join(dir, 'lokr_weights.safetensors');
      kind = 'lokr';
      if (!fs.existsSync(model)) return false;
    }
    const stat = fs.statSync(model);
    const tg = readAdapterTrigger(dir);
    adapters.push({ name, path: dir, kind, size: stat.size, mtime: stat.mtimeMs,
                    lmSize, run, trigger: tg.trigger, triggerPosition: tg.position,
                    ...readEvalSidecar(dir) });
    return true;
  };

  const scanOne = (dir: string, size: string) => {
    if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) return;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name.startsWith('.')) continue;
      const fullPath = path.join(dir, entry.name);
      try {
        // Legacy flat layout carries the size in the name; per-size dirs carry
        // it in `size`.
        const suffix = /-(0\.6B|1\.7B|4B)$/i.exec(entry.name.replace(/\.safetensors$/i, ''));
        const lmSize = size || (suffix ? suffix[1] : '');
        if (entry.isDirectory()) {
          // Unversioned adapter directly in the artist dir (legacy)…
          pushPeft(fullPath, entry.name, lmSize, '');
          // …and every stamped training run beneath it (per-run layout).
          for (const runEntry of fs.readdirSync(fullPath, { withFileTypes: true })) {
            if (!runEntry.isDirectory() || runEntry.name.startsWith('.')) continue;
            pushPeft(path.join(fullPath, runEntry.name), entry.name, lmSize, runEntry.name);
          }
        } else if (entry.isFile() && entry.name.endsWith('.safetensors')) {
          const stat = fs.statSync(fullPath);
          const tg = readAdapterTrigger(fullPath);
          adapters.push({ name: entry.name, path: fullPath, kind: 'safetensors', size: stat.size,
                          mtime: stat.mtimeMs, lmSize, run: '', trigger: tg.trigger, triggerPosition: tg.position,
                          evalScore: null, evalVerdict: '' });
        }
      } catch { /* skip unreadable entries */ }
    }
  };

  let root: string;
  try {
    if (folderParam) {
      // Explicit folder (the user's archive override). Scan it flat, and when
      // it contains lm-* per-size subdirs — e.g. the adapters root itself —
      // scan those too so the override survives the layout change.
      root = path.resolve(folderParam);
      scanOne(root, '');
      for (const r of lmAdapterRoots()) {
        const sub = path.join(root, path.basename(r.dir));
        if (path.resolve(sub) !== root && fs.existsSync(sub)) scanOne(sub, r.size);
      }
    } else {
      // Default: every planner-adapter root — the per-size lm-06b/lm-17b/lm-4b
      // dirs plus the legacy flat lm/ (adapterLayout.ts).
      root = config.aceServer.adapters;
      for (const r of lmAdapterRoots()) scanOne(r.dir, r.size);
    }
    // Alphabetical by artist; within an artist, newest run first.
    adapters.sort((a, b) =>
      a.name.localeCompare(b.name) || a.lmSize.localeCompare(b.lmSize) || b.run.localeCompare(a.run));
    res.json({ root, adapters } satisfies LmAdaptersResponse);
  } catch (err: any) {
    res.json({ root: folderParam || config.aceServer.adapters, adapters: [], error: err.message } satisfies LmAdaptersResponse);
  }
});

export default router;
