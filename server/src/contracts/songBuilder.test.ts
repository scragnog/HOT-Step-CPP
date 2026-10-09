import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// A fresh DATA_DIR and database, always, before anything reads config. No
// section here ever reaches a render: every generate is refused before its
// job, and sections are seeded directly.
const DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'builder-contract-'));
process.env.DATA_DIR = DATA_DIR;

const express = (await import('express')).default;
const database = await import('../db/database.js');
database.initDb();
const authRoutes = (await import('../routes/auth.js')).default;
const builderRoutes = (await import('../routes/songBuilder.js')).default;
const { getActiveBackendId } = await import('../services/backends/registry.js');
const contract = await import('./songBuilder.js');
const sectionOps = await import('../services/songBuilder/sectionOps.js');
type View = import('./songBuilder.js').BuilderProjectView;

const app = express();
app.use(express.json());
app.use('/api/auth', authRoutes);
app.use('/api/builder', builderRoutes);
const server = app.listen(0);
await new Promise(resolve => server.once('listening', resolve));
const base = `http://127.0.0.1:${(server.address() as { port: number }).port}/api`;
test.after(() => new Promise(resolve => server.close(() => { database.closeDb(); fs.rmSync(DATA_DIR, { recursive: true, force: true }); resolve(undefined); })));

const { token, user } = await (await fetch(`${base}/auth/auto`)).json() as { token: string; user: { id: string } };
const call = async (method: string, p: string, body?: unknown) => {
  const res = await fetch(`${base}/builder${p}`, { method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body) });
  return { status: res.status, body: await res.json() as any };
};
const db = database.getDb();
const PROJECT_FIELDS = ['id', 'user_id', 'title', 'style', 'bpm', 'key_scale', 'time_signature', 'vocal_language', 'section_length',
  'variant_count', 'gen_params', 'created_at', 'updated_at', 'revision'];
const SECTION_FIELDS = ['id', 'project_id', 'position', 'label', 'lyrics', 'direction', 'section_length', 'candidate_song_ids',
  'chosen_song_id', 'job_id', 'status', 'created_at', 'updated_at', 'candidates', 'chosen'];

function seedSection(projectId: string, id: string, position: number, candidates: string[]) {
  for (const songId of candidates) db.prepare('INSERT OR IGNORE INTO songs (id, user_id, title, bpm, key_scale) VALUES (?, ?, ?, 120, ?)').run(songId, user.id, songId, 'C major');
  db.prepare(`INSERT INTO builder_sections (id, project_id, position, label, direction, candidate_song_ids, status)
    VALUES (?, ?, ?, ?, 'append', ?, 'ready')`).run(id, projectId, position, id, JSON.stringify(candidates));
}

test('the service path re-exports the published generate schema', () => {
  assert.equal(sectionOps.generateSectionSchema, contract.generateSectionSchema);
});

test('projects: create, view and list shapes; a project edit moves the revision', async () => {
  const created = await call('POST', '/projects', { title: 'Song', bpm: 0, variantCount: 2, genParams: { steps: 8 } });
  assert.equal(created.status, 200);
  const view = created.body as View;
  assert.deepEqual(Object.keys(view.project), PROJECT_FIELDS);
  assert.deepEqual([view.project.revision, view.project.gen_params, view.sections], [0, '{"steps":8}', []]);
  const patched = (await call('PATCH', `/projects/${view.project.id}`, { style: 'rock', ignored: 1 })).body as View;
  assert.deepEqual([patched.project.style, patched.project.revision], ['rock', 1]);
  const list = (await call('GET', '/projects')).body;
  assert.deepEqual(Object.keys(list.projects[0]), [...PROJECT_FIELDS, 'section_count']);
  assert.equal((await call('GET', '/projects/nope')).status, 404);
});

test('section commands: stale revision is 409 with currentRevision; accepted ones move it by one', async () => {
  const projectId = (await call('POST', '/projects', { title: 'S' })).body.project.id as string;
  // Reordered positions: the view sorts by position, not insertion.
  seedSection(projectId, 'sec-b', 2, ['cand-3']);
  seedSection(projectId, 'sec-a', 1, ['cand-1', 'cand-gone', 'cand-2']);
  db.prepare("DELETE FROM songs WHERE id = 'cand-gone'").run();
  let view = (await call('GET', `/projects/${projectId}`)).body as View;
  assert.deepEqual(view.sections.map(s => s.id), ['sec-a', 'sec-b']);
  assert.deepEqual(Object.keys(view.sections[0]), SECTION_FIELDS);
  assert.deepEqual(view.sections[0].candidate_song_ids, ['cand-1', 'cand-gone', 'cand-2']);
  assert.deepEqual(view.sections[0].candidates.map(s => s.id), ['cand-1', 'cand-2'], 'a deleted candidate song is dropped');

  const stale = await call('POST', '/sections/sec-a/choose', { songId: 'cand-1', expectedRevision: 5 });
  assert.deepEqual([stale.status, stale.body.currentRevision], [409, 0]);
  const notCandidate = await call('POST', '/sections/sec-a/choose', { songId: 'cand-3', expectedRevision: 0 });
  assert.equal(notCandidate.status, 409);
  assert.equal((await call('POST', '/sections/sec-a/choose', { songId: 'cand-1' })).status, 400, 'expectedRevision is required');
  view = (await call('POST', '/sections/sec-a/choose', { songId: 'cand-1', expectedRevision: 0 })).body as View;
  const chosen = view.sections.find(s => s.id === 'sec-a')!;
  assert.deepEqual([view.project.revision, chosen.status, chosen.chosen_song_id, chosen.chosen?.id], [1, 'chosen', 'cand-1', 'cand-1']);
  assert.deepEqual([view.project.bpm, view.project.key_scale], [120, 'C major'], 'auto BPM/key filled from the chosen song');

  view = (await call('PATCH', '/sections/sec-b', { label: 'Chorus', expectedRevision: 1 })).body as View;
  assert.deepEqual([view.project.revision, view.sections[1].label], [2, 'Chorus']);
  assert.equal((await call('PATCH', '/sections/sec-b', { label: 7, expectedRevision: 2 })).status, 400);
  view = (await call('POST', '/sections/sec-b/stop', { expectedRevision: 2 })).body as View;
  assert.equal(view.project.revision, 3);
  view = (await call('DELETE', '/sections/sec-b?expectedRevision=3')).body as View;
  assert.deepEqual([view.project.revision, view.sections.map(s => s.id)], [4, ['sec-a']]);
  assert.equal((await call('POST', '/sections/sec-b/stop', { expectedRevision: 4 })).status, 404, 'a deleted section is gone');
});

test('generate refuses a bad body, another engine and a stale revision before any job exists', async () => {
  const projectId = (await call('POST', '/projects', { title: 'G', bpm: 100 })).body.project.id as string;
  const body = { idempotencyKey: 'k1', expectedRevision: 0, direction: 'first', length: { bars: 8 },
    expectedBackend: getActiveBackendId(), engineParams: {} };
  const bad = await call('POST', `/projects/${projectId}/sections/generate`, { ...body, direction: 'sideways' });
  assert.equal(bad.status, 400);
  assert.ok(Array.isArray(bad.body.issues));
  const engine = await call('POST', `/projects/${projectId}/sections/generate`, { ...body, expectedBackend: 'not-this-one' });
  assert.equal(engine.status, 409);
  const stale = await call('POST', `/projects/${projectId}/sections/generate`, { ...body, expectedRevision: 3 });
  assert.deepEqual([stale.status, stale.body.currentRevision], [409, 0]);
  const jobs = db.prepare("SELECT COUNT(*) AS n FROM workflow_jobs WHERE kind = 'builder-section'").get() as { n: number };
  assert.equal(jobs.n, 0);
  assert.equal((db.prepare('SELECT COUNT(*) AS n FROM builder_sections WHERE project_id = ?').get(projectId) as { n: number }).n, 0);
});
