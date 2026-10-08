import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { WorkflowJobs, type WorkflowDeps } from '../workflows/workflowJobs.js';
import type { AudioIntentItem } from '../../contracts/audioQueue.js';
import {
  generateSectionSchema, generateSection, chooseCandidate, stopSection, deleteSection, editSection,
  addCandidates, reconcileSections, sectionKind, type GenerateSection,
} from './sectionOps.js';

function fakeAudio() {
  const items = new Map<string, AudioIntentItem>();
  const byKey = new Map<string, string>();
  const audio: WorkflowDeps['audio'] = {
    enqueue: input => {
      const known = byKey.get(input.idempotencyKey);
      if (known) return { item: items.get(known)!, created: false };
      const id = `intent-${items.size + 1}`;
      const item = { id, idempotencyKey: input.idempotencyKey, status: 'pending', request: input.request, meta: input.meta ?? null, result: null, error: null } as AudioIntentItem;
      items.set(id, item); byKey.set(input.idempotencyKey, id);
      return { item, created: true };
    },
    get: id => items.get(id)!,
    cancel: id => { const i = items.get(id)!; i.status = 'cancelled'; return i; },
  };
  const land = (id: string, songId: string) => { const i = items.get(id)!; i.status = 'succeeded'; i.result = { songIds: [songId] }; };
  return { audio, items, land };
}

function setup(db = new Database(':memory:'), audio = fakeAudio()) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS songs (id TEXT PRIMARY KEY, duration REAL DEFAULT 0, bpm INTEGER DEFAULT 0, key_scale TEXT DEFAULT '',
      latent_url TEXT DEFAULT '', audio_url TEXT DEFAULT '');
    CREATE TABLE IF NOT EXISTS builder_projects (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT DEFAULT 'Song', style TEXT DEFAULT '',
      bpm INTEGER DEFAULT 0, key_scale TEXT DEFAULT '', time_signature TEXT DEFAULT '', vocal_language TEXT DEFAULT '',
      section_length REAL DEFAULT 30, variant_count INTEGER DEFAULT 2, gen_params TEXT DEFAULT '{}',
      created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')), revision INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS builder_sections (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES builder_projects(id) ON DELETE CASCADE,
      position REAL NOT NULL DEFAULT 0, label TEXT DEFAULT '', lyrics TEXT DEFAULT '', direction TEXT DEFAULT 'append',
      section_length REAL DEFAULT 30, candidate_song_ids TEXT DEFAULT '[]', chosen_song_id TEXT, job_id TEXT,
      status TEXT DEFAULT 'pending', created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')));
    INSERT OR IGNORE INTO builder_projects (id, user_id, title, style) VALUES ('p', 'u', 'Song', 'synthpop');
  `);
  const jobs = new WorkflowJobs({ db, audio: audio.audio, audioPollMs: 5 });
  jobs.register(sectionKind(() => db));
  return { db, jobs, audio };
}

const body = (over: Partial<Record<keyof GenerateSection, unknown>> = {}): GenerateSection => generateSectionSchema.parse({
  idempotencyKey: `k-${Math.random()}`, expectedRevision: 0, direction: 'first', label: 'Intro', lyrics: '',
  length: { seconds: 20 }, engineParams: { inferenceSteps: 8 }, ...over,
});
const revision = (db: Database.Database) => (db.prepare(`SELECT revision FROM builder_projects WHERE id = 'p'`).get() as { revision: number }).revision;
const section = (db: Database.Database, id: string) => db.prepare('SELECT * FROM builder_sections WHERE id = ?').get(id) as any;
/** Cancel what a test left running, so no run keeps polling. */
async function finish(jobs: WorkflowJobs) {
  for (const j of jobs.list('u')) if (j.status === 'pending' || j.status === 'running') jobs.cancel(j.id, 'u');
  await jobs.settled();
}
const tick = () => new Promise(r => setTimeout(r, 30));

/** A chosen head section: song 'h' of `duration` s, created first. */
function withHead(db: Database.Database, duration: number, extra: { id?: string; position?: number; label?: string; lyrics?: string; latent?: string } = {}) {
  const id = extra.id ?? 'h';
  db.prepare(`INSERT INTO songs (id, duration, latent_url, audio_url) VALUES (?, ?, ?, ?)`).run(id, duration, extra.latent ?? `/l/${id}`, `/a/${id}`);
  db.prepare(`INSERT INTO builder_sections (id, project_id, position, label, lyrics, section_length, candidate_song_ids, chosen_song_id, status, created_at)
    VALUES (?, 'p', ?, ?, ?, 20, ?, ?, 'chosen', ?)`).run(`s-${id}`, extra.position ?? 0, extra.label ?? 'Verse', extra.lyrics ?? 'la la', JSON.stringify([id]), id, `2026-01-01 00:00:0${db.prepare('SELECT COUNT(*) AS n FROM builder_sections').pluck().get()}`);
}

test('first section: one transaction makes the section and bumps the revision; candidates stream in, then ready', async () => {
  const { db, jobs, audio } = setup();
  const out = generateSection(db, jobs, 'u', 'p', body());
  assert.equal(revision(db), 1);
  const s = section(db, out.sectionId);
  assert.equal(s.status, 'generating');
  assert.equal(s.job_id, out.job.id);
  const [a, b] = [...audio.items.values()];
  assert.equal(audio.items.size, 2);
  assert.equal(a.request.duration, 20);
  assert.equal(a.request.taskType, undefined);
  assert.equal(a.request.lyrics, '[Intro]');
  assert.equal(a.request.style, 'synthpop');
  assert.equal(a.request.evictLm, false);
  assert.equal(a.request.masteringEnabled, false);
  assert.equal(a.request.inferenceSteps, 8);
  audio.land(a.id, 'c1');
  await tick();
  assert.deepEqual(JSON.parse(section(db, out.sectionId).candidate_song_ids), ['c1']);
  assert.equal(section(db, out.sectionId).status, 'generating');
  audio.land(b.id, 'c2');
  await jobs.settled();
  assert.equal(section(db, out.sectionId).status, 'ready');
  assert.deepEqual(JSON.parse(section(db, out.sectionId).candidate_song_ids), ['c1', 'c2']);
  assert.equal(jobs.get(out.job.id).status, 'succeeded');
});

test('append, prepend and clip points match the old studio geometry; overlap is clamped', async () => {
  const { db, jobs, audio } = setup();
  withHead(db, 60, { lyrics: 'verse one' });
  const reqOf = (b: GenerateSection) => { const n = audio.items.size; const out = generateSection(db, jobs, 'u', 'p', b); return { req: [...audio.items.values()][n].request, out }; };

  let { req, out } = reqOf(body({ direction: 'append', label: 'Chorus', lyrics: 'oh oh', expectedRevision: 0 }));
  assert.equal(req.taskType, 'repaint');
  assert.equal(req.duration, 0);
  assert.equal(req.repaintingStart, 56);
  assert.equal(req.repaintingEnd, 80);
  assert.equal(req.sourceLatentUrl, '/l/h');
  assert.equal(req.sourceAudioUrl, '/a/h');
  assert.equal(req.evictLm, true);
  assert.equal(req.lyrics, '[Verse]\nverse one\n\n[Chorus]\noh oh');
  assert.equal(section(db, out.sectionId).position, 1);

  ({ req, out } = reqOf(body({ direction: 'prepend', label: 'Intro', expectedRevision: 1 })));
  assert.equal(req.repaintingStart, -20);
  assert.equal(req.repaintingEnd, 4);
  assert.equal(req.lyrics, '[Intro]\n\n[Verse]\nverse one');
  assert.equal(section(db, out.sectionId).position, -1);

  ({ req } = reqOf(body({ direction: 'append', clipPoint: 30, expectedRevision: 2 })));
  assert.equal(req.repaintingStart, 26);
  assert.equal(req.repaintingEnd, 60);  // the old tail past 30 s is regenerated

  ({ req } = reqOf(body({ direction: 'prepend', clipPoint: 5, overlap: 50, length: { seconds: 10 }, expectedRevision: 3 })));
  assert.equal(req.repaintingEnd, 15);  // overlap clamped to the section length

  // Bars at the project tempo, with the time signature's beats per bar.
  db.prepare(`UPDATE builder_projects SET bpm = 120, time_signature = '3/4' WHERE id = 'p'`).run();
  ({ req, out } = reqOf(body({ direction: 'append', length: { bars: 8 }, expectedRevision: 4 })));
  assert.equal(section(db, out.sectionId).section_length, 12);
  assert.equal(req.repaintingEnd, 72);
  assert.equal(req.bpm, 120);
  await finish(jobs);
});

test('a head stored with duration 0 uses the measured length; a structural seed is passed through', async () => {
  const { db, jobs, audio } = setup();
  withHead(db, 0, { id: 'old', position: 0 });
  withHead(db, 0, { id: 'h', position: 1 });
  generateSection(db, jobs, 'u', 'p', body({ direction: 'append', headDuration: 3, seedSectionId: 's-old', seedStrength: 0.5 }));
  const req = [...audio.items.values()][0].request;
  assert.equal(req.sourceLatentUrl, '/l/h');  // the newest chosen section is the head
  assert.equal(req.repaintingStart, 1);       // overlap min(4, 3 - 1, 20) = 2
  assert.equal(req.repaintingEnd, 23);
  assert.equal(req.seedLatentUrl, '/l/old');
  assert.equal(req.seedSeconds, 20);
  assert.equal(req.seedStrength, 0.5);
  await finish(jobs);
});

test('direction must fit the song: first only when empty, extensions only with a head', async () => {
  const { db, jobs } = setup();
  assert.throws(() => generateSection(db, jobs, 'u', 'p', body({ direction: 'append' })), /Choose a section first/);
  withHead(db, 30);
  assert.throws(() => generateSection(db, jobs, 'u', 'p', body({ direction: 'first' })), /already has a section/);
  assert.equal(revision(db), 0);  // the failed plan rolled the bump back
  assert.throws(() => generateSection(db, jobs, 'u', 'p', body({ direction: 'append', length: { bars: 4 } })), /Bars need the project BPM/);
  await finish(jobs);
});

test('idempotent submit: the same key returns the same section and job even after the revision moved; another body is a 409', async () => {
  const { db, jobs, audio } = setup();
  const b = body();
  const first = generateSection(db, jobs, 'u', 'p', b);
  const again = generateSection(db, jobs, 'u', 'p', generateSectionSchema.parse(JSON.parse(JSON.stringify(b))));
  assert.equal(again.created, false);
  assert.equal(again.job.id, first.job.id);
  assert.equal(again.sectionId, first.sectionId);
  assert.equal(revision(db), 1);
  assert.equal(audio.items.size, 2);
  assert.equal((db.prepare('SELECT COUNT(*) AS n FROM builder_sections').get() as { n: number }).n, 1);
  assert.throws(() => generateSection(db, jobs, 'u', 'p', { ...b, label: 'Other' }), (e: any) => e.status === 409);
  await finish(jobs);
});

test('stale two-client edits get a 409 and change nothing', async () => {
  const { db, jobs } = setup();
  withHead(db, 30);
  editSection(db, 'u', 's-h', { lyrics: 'client A' }, 0);
  assert.throws(() => editSection(db, 'u', 's-h', { lyrics: 'client B' }, 0), (e: any) => e.status === 409 && e.extra.currentRevision === 1);
  assert.equal(section(db, 's-h').lyrics, 'client A');
  assert.throws(() => generateSection(db, jobs, 'u', 'p', body({ direction: 'append', expectedRevision: 0 })), (e: any) => e.status === 409);
  assert.equal((db.prepare('SELECT COUNT(*) AS n FROM builder_sections').get() as { n: number }).n, 1);
  assert.throws(() => deleteSection(db, jobs, 'u', 's-h', 0), (e: any) => e.status === 409);
  assert.ok(section(db, 's-h'));
  assert.throws(() => editSection(db, 'other', 's-h', { lyrics: 'x' }, 1), (e: any) => e.status === 404);
  await finish(jobs);
});

test('choosing a candidate stops the rest, fills auto BPM/key, and a late completion cannot change the choice', async () => {
  const { db, jobs, audio } = setup();
  const out = generateSection(db, jobs, 'u', 'p', body());
  const [a, b] = [...audio.items.values()];
  db.prepare(`INSERT INTO songs (id, duration, bpm, key_scale) VALUES ('c1', 20, 97.6, 'D minor')`).run();
  audio.land(a.id, 'c1');
  await tick();
  assert.throws(() => chooseCandidate(db, jobs, 'u', out.sectionId, 'not-a-candidate', 1), (e: any) => e.status === 409);
  chooseCandidate(db, jobs, 'u', out.sectionId, 'c1', 1);
  assert.equal(revision(db), 2);
  const s = section(db, out.sectionId);
  assert.equal(s.status, 'chosen');
  assert.equal(s.chosen_song_id, 'c1');
  assert.equal(audio.items.get(b.id)!.status, 'cancelled');
  const p = db.prepare(`SELECT bpm, key_scale FROM builder_projects WHERE id = 'p'`).get() as { bpm: number; key_scale: string };
  assert.deepEqual(p, { bpm: 98, key_scale: 'D minor' });
  await jobs.settled();
  assert.equal(jobs.get(out.job.id).status, 'cancelled');
  addCandidates(db, out.sectionId, ['late']);  // a render that finished anyway
  const after = section(db, out.sectionId);
  assert.equal(after.status, 'chosen');
  assert.equal(after.chosen_song_id, 'c1');
});

test('stop keeps the candidates that landed; delete mid-render stops it and nothing recreates the section', async () => {
  const { db, jobs, audio } = setup();
  const out = generateSection(db, jobs, 'u', 'p', body());
  audio.land([...audio.items.values()][0].id, 'c1');
  await tick();
  stopSection(db, jobs, 'u', out.sectionId, 1);
  assert.equal(section(db, out.sectionId).status, 'ready');
  assert.deepEqual(JSON.parse(section(db, out.sectionId).candidate_song_ids), ['c1']);
  await jobs.settled();

  const two = generateSection(db, jobs, 'u', 'p', body({ direction: 'first', expectedRevision: 2 }));
  const pending = [...audio.items.values()].slice(2);
  deleteSection(db, jobs, 'u', two.sectionId, 3);
  assert.equal(section(db, two.sectionId), undefined);
  assert.ok(pending.every(i => i.status === 'cancelled'));
  await jobs.settled();
  addCandidates(db, two.sectionId, ['late']);
  assert.equal(section(db, two.sectionId), undefined);
});

test('a job cancelled elsewhere settles its section on the next read', async () => {
  const { db, jobs } = setup();
  const out = generateSection(db, jobs, 'u', 'p', body());
  jobs.cancel(out.job.id, 'u');
  await jobs.settled();
  assert.equal(section(db, out.sectionId).status, 'generating');
  reconcileSections(db, jobs, 'p');
  assert.equal(section(db, out.sectionId).status, 'failed');
});

test('restart: the job is interrupted, never rerun, the section settles, and a retry reuses its audio items', async () => {
  const db = new Database(':memory:');
  const audio = fakeAudio();
  const before = setup(db, audio);
  const out = generateSection(db, before.jobs, 'u', 'p', body());
  audio.land([...audio.items.values()][0].id, 'c1');
  await tick();
  // A new process on the same database; the old one's run is gone.
  const after = new WorkflowJobs({ db, audio: audio.audio, audioPollMs: 5 });
  assert.equal(after.reconcileAfterRestart(), 1);
  after.register(sectionKind(() => db));
  assert.equal(after.get(out.job.id).status, 'interrupted');
  reconcileSections(db, after, 'p');
  assert.equal(section(db, out.sectionId).status, 'ready');
  assert.equal(audio.items.size, 2);  // nothing re-queued
  after.retry(out.job.id, 'u');
  audio.land([...audio.items.values()][1].id, 'c2');
  await after.settled();
  assert.equal(audio.items.size, 2);  // the retry got the same items back
  assert.deepEqual(JSON.parse(section(db, out.sectionId).candidate_song_ids), ['c1', 'c2']);
});

test('a reconnecting reader replays the variant events after its cursor', async () => {
  const { db, jobs, audio } = setup();
  const out = generateSection(db, jobs, 'u', 'p', body());
  const cursor = jobs.get(out.job.id).lastSeq;
  for (const [i, item] of [...audio.items.values()].entries()) audio.land(item.id, `c${i}`);
  await jobs.settled();
  const replay = jobs.replay(out.job.id, 'u', cursor);
  assert.equal(replay.gap, false);
  const variants = replay.events.filter(e => e.type === 'variant').map(e => (e.data as { songIds: string[] }).songIds[0]).sort();
  assert.deepEqual(variants, ['c0', 'c1']);
  assert.equal(replay.events.at(-1)!.type, 'status');
  assert.equal(section(db, out.sectionId).status, 'ready');
});
