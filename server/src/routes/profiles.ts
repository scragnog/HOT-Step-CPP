// profiles.ts — Parameter profile REST API
//
// A profile is a named snapshot of every generation parameter (the UI's
// 'hot-step-preset' JSON — same format as the preset export file), stored
// server-side so the user can switch configs in-app without juggling files.
// Files are plain preset JSON wrapped with { name, saved_at, data }, so an
// exported preset can be dropped in by hand and vice versa.
//
// Mounts at: /api/profiles
// Routes:
//   GET    /api/profiles         — list all profiles (full data inline)
//   GET    /api/profiles/:name   — load one profile
//   POST   /api/profiles         — save/overwrite { name, data }
//   DELETE /api/profiles/:name   — delete profile

import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { config } from '../config.js';

const router = Router();

function profilesDir(): string {
  const dir = path.join(config.data.dir, 'profiles');
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

// Sanitize name — strip path separators and shell-hostile chars (same regex as seeds.ts)
function safeName(name: string): string {
  return name.replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').slice(0, 100).trim();
}

function profilePath(name: string): string {
  return path.join(profilesDir(), `${safeName(name)}.json`);
}

interface ProfileFile {
  name: string;
  saved_at: string;
  data: Record<string, unknown>;
}

function readProfile(name: string): ProfileFile | null {
  const p = profilePath(name);
  if (!fs.existsSync(p)) return null;
  try {
    const raw = JSON.parse(fs.readFileSync(p, 'utf8'));
    // Tolerate a bare preset JSON dropped into the folder by hand
    if (raw && typeof raw === 'object' && raw.data === undefined && raw._format === 'hot-step-preset') {
      return { name, saved_at: '', data: raw };
    }
    return raw;
  } catch { return null; }
}

// GET /api/profiles — list all with data inline (profiles are a few KB each)
router.get('/', (_req, res) => {
  try {
    const names = fs.readdirSync(profilesDir())
      .filter(f => f.endsWith('.json'))
      .map(f => f.slice(0, -5))
      .sort((a, b) => a.localeCompare(b));
    const profiles = names
      .map(name => readProfile(name))
      .filter((p): p is ProfileFile => p !== null && !!p.data);
    res.json({ profiles, count: profiles.length });
  } catch (err: any) {
    console.error('[Profiles] list failed:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/profiles/:name — load one
router.get('/:name', (req, res) => {
  const profile = readProfile(req.params.name);
  if (!profile) {
    res.status(404).json({ error: `profile '${req.params.name}' not found` });
    return;
  }
  res.json(profile);
});

// POST /api/profiles — save/overwrite { name, data }
router.post('/', (req, res) => {
  const { name, data } = req.body ?? {};
  if (!name || typeof name !== 'string' || !safeName(name)) {
    res.status(400).json({ error: 'name is required' });
    return;
  }
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    res.status(400).json({ error: 'data must be an object' });
    return;
  }
  const profile: ProfileFile = {
    name: safeName(name),
    saved_at: new Date().toISOString(),
    data,
  };
  try {
    fs.writeFileSync(profilePath(name), JSON.stringify(profile, null, 2), 'utf8');
    console.log(`[Profiles] Saved '${profile.name}'`);
    res.json({ ok: true, name: profile.name, saved_at: profile.saved_at });
  } catch (err: any) {
    console.error('[Profiles] save failed:', err.message);
    res.status(500).json({ error: 'failed to write profile' });
  }
});

// PATCH /api/profiles/:name — rename { newName } (data unchanged)
router.patch('/:name', (req, res) => {
  const oldName = req.params.name;
  const { newName } = req.body ?? {};
  if (!newName || typeof newName !== 'string' || !safeName(newName)) {
    res.status(400).json({ error: 'newName is required' });
    return;
  }
  const src = profilePath(oldName);
  if (!fs.existsSync(src)) {
    res.status(404).json({ error: `profile '${oldName}' not found` });
    return;
  }
  const dstName = safeName(newName);
  const dst = profilePath(newName);
  const sameFile = path.resolve(dst) === path.resolve(src);
  if (fs.existsSync(dst) && !sameFile) {
    res.status(409).json({ error: `a profile named '${dstName}' already exists` });
    return;
  }
  const profile = readProfile(oldName);
  if (!profile) {
    res.status(404).json({ error: `profile '${oldName}' not found` });
    return;
  }
  profile.name = dstName;
  try {
    fs.writeFileSync(dst, JSON.stringify(profile, null, 2), 'utf8');
    if (!sameFile) fs.unlinkSync(src);
    console.log(`[Profiles] Renamed '${oldName}' -> '${dstName}'`);
    res.json({ ok: true, name: dstName });
  } catch (err: any) {
    console.error('[Profiles] rename failed:', err.message);
    res.status(500).json({ error: 'failed to rename profile' });
  }
});

// DELETE /api/profiles/:name — delete
router.delete('/:name', (req, res) => {
  const p = profilePath(req.params.name);
  if (!fs.existsSync(p)) {
    res.status(404).json({ error: `profile '${req.params.name}' not found` });
    return;
  }
  try {
    fs.unlinkSync(p);
    console.log(`[Profiles] Deleted '${req.params.name}'`);
    res.json({ ok: true, deleted: req.params.name });
  } catch (err: any) {
    console.error('[Profiles] delete failed:', err.message);
    res.status(500).json({ error: 'delete failed' });
  }
});

export default router;
