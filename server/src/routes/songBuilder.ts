// songBuilder.ts — Song Builder (Udio-style section-by-section generation)
//
// A "project" is one song assembled from an ordered chain of sections. Each
// section generates N candidate songs (variants) via the normal /api/generate
// pipeline (text2music for the first section, outpaint-repaint extending the
// previously chosen variant's latent for every section after). The user picks
// one variant per section; that pick becomes the source for the next section.
//
// This router is pure bookkeeping. Generation runs through /api/generate; the
// UI records the returned jobId/songIds here and tracks the chosen variant.

import { Router } from 'express';
import { randomUUID } from 'crypto';
import { getDb } from '../db/database.js';
import { getUserId } from './auth.js';

const router = Router();

// ── Helpers ────────────────────────────────────────────────────────────────

/** Resolve a list of song ids into full song rows (parsed), preserving order. */
function resolveSongs(ids: string[]): any[] {
  if (!ids.length) return [];
  const placeholders = ids.map(() => '?').join(',');
  const rows = getDb()
    .prepare(`SELECT * FROM songs WHERE id IN (${placeholders})`)
    .all(...ids) as any[];
  const byId = new Map(rows.map(r => [r.id, r]));
  return ids
    .map(id => byId.get(id))
    .filter(Boolean)
    .map((s: any) => ({ ...s, tags: JSON.parse(s.tags || '[]'), is_public: !!s.is_public }));
}

/** Load a project's sections (ordered by position) with resolved candidate + chosen songs. */
function loadSections(projectId: string): any[] {
  const sections = getDb()
    .prepare(`SELECT * FROM builder_sections WHERE project_id = ? ORDER BY position ASC, created_at ASC`)
    .all(projectId) as any[];

  return sections.map(sec => {
    const candidateIds: string[] = JSON.parse(sec.candidate_song_ids || '[]');
    const candidates = resolveSongs(candidateIds);
    const chosen = sec.chosen_song_id ? resolveSongs([sec.chosen_song_id])[0] || null : null;
    return { ...sec, candidate_song_ids: candidateIds, candidates, chosen };
  });
}

/** Verify a project belongs to the user; returns the row or null. */
function ownedProject(projectId: string, userId: string): any | null {
  const p = getDb()
    .prepare(`SELECT * FROM builder_projects WHERE id = ? AND user_id = ?`)
    .get(projectId, userId) as any;
  return p || null;
}

/** Verify a section belongs to a project owned by the user; returns {section, project} or null. */
function ownedSection(sectionId: string, userId: string): { section: any; project: any } | null {
  const section = getDb()
    .prepare(`SELECT * FROM builder_sections WHERE id = ?`)
    .get(sectionId) as any;
  if (!section) return null;
  const project = ownedProject(section.project_id, userId);
  if (!project) return null;
  return { section, project };
}

const touchProject = (id: string) =>
  getDb().prepare(`UPDATE builder_projects SET updated_at = datetime('now') WHERE id = ?`).run(id);

// ── Project routes ───────────────────────────────────────────────────────────

// GET /api/builder/projects — list projects (with section count)
router.get('/projects', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const projects = getDb().prepare(`
    SELECT p.*,
      (SELECT COUNT(*) FROM builder_sections s WHERE s.project_id = p.id) AS section_count
    FROM builder_projects p
    WHERE p.user_id = ?
    ORDER BY p.updated_at DESC
  `).all(userId);
  res.json({ projects });
});

// GET /api/builder/projects/:id — full project with ordered, resolved sections
router.get('/projects/:id', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const project = ownedProject(req.params.id, userId);
  if (!project) { res.status(404).json({ error: 'Project not found' }); return; }

  res.json({ project, sections: loadSections(project.id) });
});

// POST /api/builder/projects — create a project
router.post('/projects', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const b = req.body || {};
  const id = randomUUID();
  getDb().prepare(`
    INSERT INTO builder_projects
      (id, user_id, title, style, bpm, key_scale, time_signature, vocal_language,
       section_length, variant_count, gen_params)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    id, userId,
    b.title || 'Untitled Song',
    b.style || '',
    b.bpm || 0,
    b.keyScale || '',
    b.timeSignature || '',
    b.vocalLanguage || '',
    b.sectionLength ?? 30,
    b.variantCount ?? 4,
    JSON.stringify(b.genParams || {}),
  );

  const project = getDb().prepare(`SELECT * FROM builder_projects WHERE id = ?`).get(id);
  res.json({ project, sections: [] });
});

// PATCH /api/builder/projects/:id — update shared params / title
router.patch('/projects/:id', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const project = ownedProject(req.params.id, userId);
  if (!project) { res.status(404).json({ error: 'Project not found' }); return; }

  const b = req.body || {};
  const map: Record<string, string> = {
    title: 'title', style: 'style', bpm: 'bpm', keyScale: 'key_scale',
    timeSignature: 'time_signature', vocalLanguage: 'vocal_language',
    sectionLength: 'section_length', variantCount: 'variant_count',
  };
  const sets: string[] = [];
  const vals: any[] = [];
  for (const [k, col] of Object.entries(map)) {
    if (b[k] !== undefined) { sets.push(`${col} = ?`); vals.push(b[k]); }
  }
  if (b.genParams !== undefined) { sets.push(`gen_params = ?`); vals.push(JSON.stringify(b.genParams)); }
  if (sets.length) {
    sets.push(`updated_at = datetime('now')`);
    vals.push(project.id);
    getDb().prepare(`UPDATE builder_projects SET ${sets.join(', ')} WHERE id = ?`).run(...vals);
  }

  const updated = getDb().prepare(`SELECT * FROM builder_projects WHERE id = ?`).get(project.id);
  res.json({ project: updated, sections: loadSections(project.id) });
});

// DELETE /api/builder/projects/:id
router.delete('/projects/:id', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const project = ownedProject(req.params.id, userId);
  if (!project) { res.status(404).json({ error: 'Project not found' }); return; }

  // ON DELETE CASCADE removes sections. Candidate songs are left in the library
  // (they are normal songs and may be referenced elsewhere).
  getDb().prepare(`DELETE FROM builder_projects WHERE id = ?`).run(project.id);
  res.json({ ok: true });
});

// ── Section routes ───────────────────────────────────────────────────────────

// POST /api/builder/projects/:id/sections — create a section record
// Called by the UI right after it kicks off generation via /api/generate.
router.post('/projects/:id/sections', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const project = ownedProject(req.params.id, userId);
  if (!project) { res.status(404).json({ error: 'Project not found' }); return; }

  const b = req.body || {};
  const id = randomUUID();

  // Default position: append after the current max (or before the min for prepend).
  let position = b.position;
  if (position === undefined) {
    const agg = getDb()
      .prepare(`SELECT MIN(position) AS lo, MAX(position) AS hi FROM builder_sections WHERE project_id = ?`)
      .get(project.id) as any;
    if (b.direction === 'prepend') position = (agg.lo ?? 0) - 1;
    else position = (agg.hi ?? -1) + 1;
  }

  getDb().prepare(`
    INSERT INTO builder_sections
      (id, project_id, position, label, lyrics, direction, section_length,
       candidate_song_ids, chosen_song_id, job_id, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    id, project.id, position,
    b.label || '',
    b.lyrics || '',
    b.direction || 'append',
    b.sectionLength ?? project.section_length ?? 30,
    JSON.stringify(b.candidateSongIds || []),
    b.chosenSongId || null,
    b.jobId || null,
    b.status || (b.jobId ? 'generating' : 'pending'),
  );
  touchProject(project.id);

  const section = loadSections(project.id).find(s => s.id === id);
  res.json({ section });
});

// PATCH /api/builder/sections/:id — update a section
// Used to record candidate song ids when a job completes, to choose a variant,
// and to edit label/lyrics.
router.patch('/sections/:id', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const owned = ownedSection(req.params.id, userId);
  if (!owned) { res.status(404).json({ error: 'Section not found' }); return; }

  const b = req.body || {};
  const sets: string[] = [];
  const vals: any[] = [];

  if (b.label !== undefined) { sets.push(`label = ?`); vals.push(b.label); }
  if (b.lyrics !== undefined) { sets.push(`lyrics = ?`); vals.push(b.lyrics); }
  if (b.position !== undefined) { sets.push(`position = ?`); vals.push(b.position); }
  if (b.sectionLength !== undefined) { sets.push(`section_length = ?`); vals.push(b.sectionLength); }
  if (b.jobId !== undefined) { sets.push(`job_id = ?`); vals.push(b.jobId); }
  if (b.candidateSongIds !== undefined) {
    sets.push(`candidate_song_ids = ?`);
    vals.push(JSON.stringify(b.candidateSongIds));
  }
  if (b.chosenSongId !== undefined) { sets.push(`chosen_song_id = ?`); vals.push(b.chosenSongId); }
  if (b.status !== undefined) { sets.push(`status = ?`); vals.push(b.status); }

  if (sets.length) {
    sets.push(`updated_at = datetime('now')`);
    vals.push(owned.section.id);
    getDb().prepare(`UPDATE builder_sections SET ${sets.join(', ')} WHERE id = ?`).run(...vals);
    touchProject(owned.project.id);
  }

  const section = loadSections(owned.project.id).find(s => s.id === owned.section.id);
  res.json({ section });
});

// DELETE /api/builder/sections/:id
router.delete('/sections/:id', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const owned = ownedSection(req.params.id, userId);
  if (!owned) { res.status(404).json({ error: 'Section not found' }); return; }

  getDb().prepare(`DELETE FROM builder_sections WHERE id = ?`).run(owned.section.id);
  touchProject(owned.project.id);
  res.json({ ok: true });
});

export default router;
