// songBuilder.ts — Song Builder (Udio-style section-by-section generation)
//
// A "project" is one song assembled from an ordered chain of sections. Each
// section generates N candidate songs (variants) via the normal /api/generate
// pipeline (text2music for the first section, outpaint-repaint extending the
// previously chosen variant's latent for every section after). The user picks
// one variant per section; that pick becomes the source for the next section.
//
// Projects are plain CRUD. Sections are Node-owned operations: geometry,
// variant renders and candidates live in services/songBuilder/sectionOps.ts,
// and each section's renders run as a builder-section workflow job.

import { Router, type Request, type Response } from 'express';
import { randomUUID } from 'crypto';
import { getDb } from '../db/database.js';
import { getUserId } from './auth.js';
import { registerWorkflowKind, workflowJobs } from './workflows.js';
import { WorkflowError } from '../services/workflows/workflowJobs.js';
import {
  generateSectionSchema, generateSection, chooseCandidate, stopSection, deleteSection, editSection,
  reconcileSections, sectionKind,
} from '../services/songBuilder/sectionOps.js';

registerWorkflowKind(sectionKind(getDb));

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

/** The project and its sections, after settling any whose job has ended. */
function projectView(projectId: string): { project: any; sections: any[] } {
  reconcileSections(getDb(), workflowJobs(), projectId);
  return {
    project: getDb().prepare(`SELECT * FROM builder_projects WHERE id = ?`).get(projectId),
    sections: loadSections(projectId),
  };
}

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

  res.json(projectView(project.id));
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
    // Last write wins per field, as before, but other clients' section edits
    // based on the old settings now see a newer revision.
    sets.push(`updated_at = datetime('now')`, `revision = revision + 1`);
    vals.push(project.id);
    getDb().prepare(`UPDATE builder_projects SET ${sets.join(', ')} WHERE id = ?`).run(...vals);
  }

  res.json(projectView(project.id));
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
// Node owns section geometry, the variant renders and candidate insertion
// (services/songBuilder/sectionOps.ts). Every edit names the project revision
// it was based on; a stale one is a 409 carrying currentRevision.

/** Run an operation and answer with the whole project, or its error. */
function sectionOp(fn: (req: Request, userId: string) => string) {
  return (req: Request, res: Response) => {
    const userId = getUserId(req);
    if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }
    try {
      const projectId = fn(req, userId);
      res.json(projectView(projectId));
    } catch (err) {
      if (err instanceof WorkflowError) { res.status(err.status).json({ error: err.message, ...err.extra }); return; }
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  };
}

const revisionOf = (req: Request) => {
  const r = Number(req.body?.expectedRevision ?? req.query.expectedRevision);
  if (!Number.isInteger(r) || r < 0) throw new WorkflowError(400, 'expectedRevision is required');
  return r;
};
const projectOfSection = (id: string) =>
  (getDb().prepare(`SELECT project_id FROM builder_sections WHERE id = ?`).get(id) as { project_id: string } | undefined)?.project_id;

// POST /api/builder/projects/:id/sections/generate — create the next section
// and start its variants. Returns the project plus { jobId, sectionId }.
router.post('/projects/:id/sections/generate', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }
  const parsed = generateSectionSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: 'Invalid section request', issues: parsed.error.issues.map(i => ({ path: i.path.join('.'), message: i.message })) });
    return;
  }
  try {
    const out = generateSection(getDb(), workflowJobs(), userId, req.params.id, parsed.data);
    res.json({ ...projectView(req.params.id), jobId: out.job.id, sectionId: out.sectionId });
  } catch (err) {
    if (err instanceof WorkflowError) { res.status(err.status).json({ error: err.message, ...err.extra }); return; }
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// POST /api/builder/sections/:id/choose { songId, expectedRevision }
router.post('/sections/:id/choose', sectionOp((req, userId) => {
  const projectId = projectOfSection(String(req.params.id));
  if (typeof req.body?.songId !== 'string') throw new WorkflowError(400, 'songId is required');
  chooseCandidate(getDb(), workflowJobs(), userId, String(req.params.id), req.body.songId, revisionOf(req));
  return projectId!;
}));

// POST /api/builder/sections/:id/stop { expectedRevision } — keep what landed
router.post('/sections/:id/stop', sectionOp((req, userId) => {
  const projectId = projectOfSection(String(req.params.id));
  stopSection(getDb(), workflowJobs(), userId, String(req.params.id), revisionOf(req));
  return projectId!;
}));

// PATCH /api/builder/sections/:id { label?, lyrics?, expectedRevision }
router.patch('/sections/:id', sectionOp((req, userId) => {
  const projectId = projectOfSection(String(req.params.id));
  const b = req.body || {};
  if ((b.label !== undefined && typeof b.label !== 'string') || (b.lyrics !== undefined && typeof b.lyrics !== 'string')) {
    throw new WorkflowError(400, 'label and lyrics must be strings');
  }
  editSection(getDb(), userId, String(req.params.id), { label: b.label, lyrics: b.lyrics }, revisionOf(req));
  return projectId!;
}));

// DELETE /api/builder/sections/:id?expectedRevision=N — also stops its variants
router.delete('/sections/:id', sectionOp((req, userId) => {
  const projectId = projectOfSection(String(req.params.id));
  deleteSection(getDb(), workflowJobs(), userId, String(req.params.id), revisionOf(req));
  return projectId!;
}));

export default router;
