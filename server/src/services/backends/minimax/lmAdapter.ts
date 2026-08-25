// minimax/lmAdapter.ts — MM3 LM LoRA adapters (runtime, engine-applied).
//
// Adapters are PEFT safetensors from SimpleTuner's LM-training mode
// (`language_model.`-prefixed keys, q/k/v/o + gate/up/down on all 36 layers).
// The engine applies them as runtime low-rank deltas with live group scales
// (mm3-lm-adapter.h) — nothing here touches tensor data, this module only
// resolves references and lists what is installed.
//
// Layout (Rob-approved 2026-08-20):
//   <adapters root>/mm3-lm-adapters/<run>/<name>.safetensors
//   <same>.json                       optional sidecar: trigger word, rank,
//                                     dataset, recommended scales, run params
//
// The default *recommended* scales come from the 2026-08-20 ablation grid on
// the alk3 r256 adapter: attention 1.0 / MLP 0.5 ("mlp-half", the ear
// winner). Sidecars may override per adapter; the UI prefills from here.

import fs from 'fs';
import path from 'path';

import { config } from '../../../config.js';

export interface Mm3LmAdapterMeta {
  name: string;
  trigger?: string;
  rank?: number;
  dataset?: string;
  encoder?: string;
  trainedSteps?: number;
  recommendedScales?: Partial<Mm3LmAdapterScales>;
  notes?: string;
}

export interface Mm3LmAdapterScales {
  scale: number;       // global
  scaleAttn: number;
  scaleMlp: number;
  scaleEarly: number;
  scaleMid: number;
  scaleLate: number;
}

export const MM3_LM_ADAPTER_DEFAULT_SCALES: Mm3LmAdapterScales = {
  scale: 1.0,
  scaleAttn: 1.0,
  /** 1.0 as of 2026-08-25. The old 0.5 was the ear-validated crutch for the
   *  timbre fault — semantic-only training let adapters drift the hidden state
   *  the depth decoder reads, and halving MLP was what kept vocals human. The
   *  acoustic loss now holds that state during training, so adapters trained
   *  with it run at full strength. PRE-FIX adapters still want ~0.5-0.75: use
   *  their sidecar recommendedScales, or retrain. */
  scaleMlp: 1.0,
  scaleEarly: 1.0,
  scaleMid: 1.0,
  scaleLate: 1.0,
};

export function mm3LmAdapterDir(): string {
  return path.join(config.aceServer.adapters, 'mm3-lm-adapters');
}

/** Resolve an adapter reference to an absolute path INSIDE the adapter
 *  directory. The reference reaches the server from the browser, so anything
 *  escaping the directory returns null rather than being read. */
export function resolveMm3LmAdapterPath(ref: string): string | null {
  if (!ref) return null;
  const dir = mm3LmAdapterDir();
  const resolved = path.resolve(dir, ref);
  const rel = path.relative(dir, resolved);
  if (rel.startsWith('..') || path.isAbsolute(rel)) return null;
  return resolved;
}

/** Installed adapters (any .safetensors under the root, one level of run
 *  directories deep), with sidecar metadata when present. Never throws. */
export function listMm3LmAdapters(): Array<{ file: string } & Mm3LmAdapterMeta> {
  const out: Array<{ file: string } & Mm3LmAdapterMeta> = [];
  const dir = mm3LmAdapterDir();
  const scan = (sub: string, depth: number) => {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(path.join(dir, sub), { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      const rel = sub ? path.join(sub, e.name) : e.name;
      if (e.isDirectory() && depth < 2) {
        scan(rel, depth + 1);
      } else if (e.isFile() && e.name.endsWith('.safetensors')) {
        let meta: Mm3LmAdapterMeta = { name: rel };
        try {
          meta = { ...meta, ...JSON.parse(fs.readFileSync(path.join(dir, `${rel}.json`), 'utf-8')) };
        } catch { /* sidecar optional */ }
        out.push({ file: rel, ...meta });
      }
    }
  };
  scan('', 0);
  return out.sort((a, b) => a.file.localeCompare(b.file));
}
