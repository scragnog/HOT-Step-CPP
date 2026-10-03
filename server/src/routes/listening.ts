// listening.ts — local-only static server + save endpoint for ear-test score
// sheets (see .claude/skills/ear-test-scoresheet/SKILL.md).
//
// Mounts at: /listening (top-level, like /audio — not under /api)
// Routes:
//   GET  /listening/:folder/{*splat}  — serve a file from _experiments/_LISTENING/<folder>/
//   POST /listening/:folder/scores    — write that folder's scores.json (body = JSON)
//
// These pages write to disk with no auth, so every request is restricted to
// loopback. Both routes confine the resolved path to _experiments/_LISTENING:
// '..' segments are rejected outright, and a symlink anywhere in the chain
// that resolves outside the root is rejected too (SAFE_NAME + realpath walk
// in resolveConfined).
import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { PROJECT_ROOT } from '../config.js';

const router = Router();

const LISTENING_ROOT = path.resolve(PROJECT_ROOT, '_experiments', '_LISTENING');
const LOOPBACK = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1']);
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

router.use((req, res, next) => {
  if (!LOOPBACK.has(req.socket.remoteAddress ?? '')) {
    res.status(403).json({ error: 'Listening pages are local-only' });
    return;
  }
  next();
});

/** Resolves `<folder>/<segments...>` under `root`, or null if the
 *  folder/segments look unsafe or the path (lexically or via a symlink)
 *  escapes the root. Exported for the path-confinement test. */
export function resolveConfined(root: string, folder: string, segments: string[]): string | null {
  if (!SAFE_NAME.test(folder)) return null;
  for (const seg of segments) {
    if (!seg || seg === '.' || seg === '..' || seg.includes('/') || seg.includes('\\')) return null;
  }
  const rootReal = fs.realpathSync(root);
  const target = path.resolve(root, folder, ...segments);
  if (target !== rootReal && !target.startsWith(rootReal + path.sep)) return null;
  // Walk up to the nearest existing ancestor and realpath it, to catch a
  // symlink anywhere in the chain (including one for a file that doesn't
  // exist yet, e.g. scores.json on first save).
  let check = target;
  while (!fs.existsSync(check)) {
    const parent = path.dirname(check);
    if (parent === check) return null;
    check = parent;
  }
  const real = fs.realpathSync(check);
  if (real !== rootReal && !real.startsWith(rootReal + path.sep)) return null;
  return target;
}

router.get('/:folder/{*splat}', (req, res) => {
  const splat = ([] as string[]).concat((req.params as any).splat ?? []);
  const segments = splat.length ? splat : ['index.html'];
  const target = resolveConfined(LISTENING_ROOT, req.params.folder, segments);
  if (!target) {
    res.status(400).json({ error: 'Invalid path' });
    return;
  }
  fs.stat(target, (err, stat) => {
    if (err || !stat.isFile()) {
      res.status(404).json({ error: 'Not found' });
      return;
    }
    res.sendFile(target);
  });
});

router.post('/:folder/scores', (req, res) => {
  const target = resolveConfined(LISTENING_ROOT, req.params.folder, ['scores.json']);
  if (!target) {
    res.status(400).json({ error: 'Invalid path' });
    return;
  }
  const tmp = `${target}.tmp-${crypto.randomBytes(6).toString('hex')}`;
  try {
    fs.writeFileSync(tmp, JSON.stringify(req.body, null, 1));
    fs.renameSync(tmp, target);
    res.json({ ok: true });
  } catch (err: any) {
    try { fs.unlinkSync(tmp); } catch {}
    console.error('[Listening] scores write failed:', err.message);
    res.status(500).json({ error: err.message });
  }
});

export default router;
