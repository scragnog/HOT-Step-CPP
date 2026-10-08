// seeds.ts — Seed management REST API
// MDMAchine / A&E Concepts 2026
// GPL v3 — safe for public repo
//
// File format is intentionally identical to MD_Nodes/SeedSaver (ComfyUI):
//   { seed: number, saved_at: string, metadata: { description?, tags?, ... } }
// Drop your existing ComfyUI seeds/ directory into the output dir and they load immediately.
//
// Mounts at: /api/seeds
// Routes:
//   GET    /api/seeds              — list all saved seeds (flat, default dir)
//   GET    /api/seeds/:name        — load one seed by name
//   POST   /api/seeds              — save seed { name, seed, description?, tags? }
//   DELETE /api/seeds/:name        — delete seed
//   GET    /api/seeds/favorites    — list favorites
//   POST   /api/seeds/:name/favorite — toggle favorite
//   GET    /api/seeds/random       — return a random saved seed
//
// Subdirectory support: pass ?subdir=mydir to scope to a subfolder.
// The UI currently uses the flat default (no subdir param).

import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { config } from '../config.js';
import type {
  ListSeedsResponse, ListFavoritesResponse, FavoriteSeedEntry, RandomSeedResponse, GetSeedResponse,
  SaveSeedResponse, DeleteSeedResponse, ToggleFavoriteResponse,
} from '../contracts/seeds.js';

const router = Router();

// ─── Storage root ─────────────────────────────────────────────────────────────

function seedsRoot(): string {
  return path.join(config.data.dir, 'seeds');
}

function seedsDir(subdir = ''): string {
  const base = seedsRoot();
  const dir  = subdir ? path.join(base, subdir) : base;
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function seedPath(name: string, subdir = ''): string {
  // Sanitize name — strip path separators and shell-hostile chars
  const safe = name.replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').slice(0, 200);
  return path.join(seedsDir(subdir), `${safe}.json`);
}

function favoritesPath(): string {
  fs.mkdirSync(seedsRoot(), { recursive: true });
  return path.join(seedsRoot(), '_favorites.json');
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function loadFavorites(): string[] {
  try {
    const raw = fs.readFileSync(favoritesPath(), 'utf8');
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch { return []; }
}

function saveFavorites(list: string[]): void {
  try { fs.writeFileSync(favoritesPath(), JSON.stringify(list, null, 2), 'utf8'); } catch {}
}

function listSeedNames(subdir = ''): string[] {
  const dir = seedsDir(subdir);
  try {
    return fs.readdirSync(dir)
      .filter(f => f.endsWith('.json') && !f.startsWith('_'))
      .map(f => f.slice(0, -5))  // strip .json
      .sort((a, b) => a.localeCompare(b));
  } catch { return []; }
}

function readSeedFile(name: string, subdir = ''): { seed: number; saved_at: string; metadata: Record<string, unknown> } | null {
  const p = seedPath(name, subdir);
  if (!fs.existsSync(p)) return null;
  try {
    const raw = fs.readFileSync(p, 'utf8');
    return JSON.parse(raw);
  } catch { return null; }
}

function writeSeedFile(
  name: string,
  seed: number,
  subdir = '',
  meta: Record<string, unknown> = {},
): boolean {
  const p = seedPath(name, subdir);
  const data = {
    seed,
    saved_at: new Date().toISOString(),
    metadata: meta,
  };
  try {
    fs.writeFileSync(p, JSON.stringify(data, null, 2), 'utf8');
    return true;
  } catch (e) {
    console.error('[seeds] write failed:', e);
    return false;
  }
}

// ─── Routes ───────────────────────────────────────────────────────────────────

// GET /api/seeds — list all seeds with metadata inline (for the drawer list)
router.get('/', (req, res) => {
  const subdir    = String(req.query.subdir || '');
  const favorites = loadFavorites();
  const names     = listSeedNames(subdir);

  const seeds = names.map(name => {
    const data = readSeedFile(name, subdir);
    return {
      name,
      seed:        data?.seed ?? null,
      saved_at:    data?.saved_at ?? null,
      description: (data?.metadata?.description as string) || '',
      tags:        (data?.metadata?.tags as string[]) || [],
      favorite:    favorites.includes(name),
    };
  });

  res.json({ seeds, count: seeds.length } satisfies ListSeedsResponse);
});

// GET /api/seeds/favorites — list favorite seed names
router.get('/favorites', (_req, res) => {
  const favorites = loadFavorites();
  const seeds = favorites
    .map(name => {
      const data = readSeedFile(name);
      return data ? { name, seed: data.seed, saved_at: data.saved_at, favorite: true } : null;
    })
    .filter((s): s is FavoriteSeedEntry => s !== null);
  res.json({ seeds } satisfies ListFavoritesResponse);
});

// GET /api/seeds/random — return a random saved seed
router.get('/random', (req, res) => {
  const subdir = String(req.query.subdir || '');
  const names  = listSeedNames(subdir);
  if (names.length === 0) return res.status(404).json({ error: 'no seeds saved' });

  const name = names[Math.floor(Math.random() * names.length)];
  const data = readSeedFile(name, subdir);
  if (!data) return res.status(404).json({ error: 'seed file missing' });

  res.json({ name, seed: data.seed, saved_at: data.saved_at } satisfies RandomSeedResponse);
});

// GET /api/seeds/:name — load a single seed
router.get('/:name', (req, res) => {
  const { name } = req.params;
  const subdir   = String(req.query.subdir || '');
  const data     = readSeedFile(name, subdir);

  if (!data) return res.status(404).json({ error: `seed '${name}' not found` });

  const favorites = loadFavorites();
  res.json({
    name,
    seed:        data.seed,
    saved_at:    data.saved_at,
    description: (data.metadata?.description as string) || '',
    tags:        (data.metadata?.tags as string[]) || [],
    favorite:    favorites.includes(name),
  } satisfies GetSeedResponse);
});

// POST /api/seeds — save a seed
//   body: { name: string, seed: number, description?: string, tags?: string[] }
router.post('/', (req, res) => {
  const { name, seed, description = '', tags = [], subdir = '' } = req.body;

  if (!name || typeof name !== 'string' || !name.trim()) {
    return res.status(400).json({ error: 'name is required' });
  }
  if (seed === undefined || seed === null || isNaN(Number(seed))) {
    return res.status(400).json({ error: 'seed must be a number' });
  }

  const seedVal = Math.max(0, Math.min(Number(seed), 9007199254740991));
  const meta    = {
    description: String(description).slice(0, 500),
    tags:        Array.isArray(tags) ? tags.map(String) : [],
    source:      'hot-step-cpp',
  };

  const ok = writeSeedFile(name.trim(), seedVal, String(subdir), meta);
  if (!ok) return res.status(500).json({ error: 'failed to write seed file' });

  res.json({ ok: true, name: name.trim(), seed: seedVal } satisfies SaveSeedResponse);
});

// DELETE /api/seeds/:name — delete a seed
router.delete('/:name', (req, res) => {
  const { name } = req.params;
  const subdir   = String(req.query.subdir || '');
  const p        = seedPath(name, subdir);

  if (!fs.existsSync(p)) return res.status(404).json({ error: `seed '${name}' not found` });

  try {
    fs.unlinkSync(p);
    // Also remove from favorites if present
    const favs = loadFavorites();
    const next = favs.filter(f => f !== name);
    if (next.length !== favs.length) saveFavorites(next);
    res.json({ ok: true, deleted: name } satisfies DeleteSeedResponse);
  } catch (e) {
    res.status(500).json({ error: 'delete failed' });
  }
});

// POST /api/seeds/:name/favorite — toggle favorite
router.post('/:name/favorite', (req, res) => {
  const { name } = req.params;
  const favs     = loadFavorites();
  const idx      = favs.indexOf(name);
  let nowFavorite: boolean;

  if (idx >= 0) {
    favs.splice(idx, 1);
    nowFavorite = false;
  } else {
    favs.push(name);
    nowFavorite = true;
  }

  saveFavorites(favs);
  res.json({ ok: true, name, favorite: nowFavorite } satisfies ToggleFavoriteResponse);
});

export default router;
