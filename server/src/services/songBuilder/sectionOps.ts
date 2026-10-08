// songBuilder/sectionOps.ts — Song Builder section operations, owned by Node.
//
// Generating a section is a workflow job (kind builder-section). The route
// checks the project revision, works out the geometry from the stored project
// (head song, first/append/prepend repaint bounds, overlap clamp, structural
// seed, cumulative lyrics) and creates the section, all in one transaction;
// the job then queues the variants on the audio intent queue and adds each
// candidate as it lands.
//
// Rules:
//   - Every user edit (generate, choose, stop, edit, delete, project fields)
//     bumps builder_projects.revision inside its own transaction. Section
//     edits name the revision they were based on, so a second client working
//     from an older view gets a 409 instead of overwriting the first.
//   - The job only ever appends candidates to its own section and moves it
//     out of 'generating'. It never touches chosen_song_id, and a write to a
//     deleted section changes nothing, so a late completion cannot overwrite
//     a choice or bring a deleted section back.
//   - Submission is idempotent on the client's key: the same key and body
//     returns the same job and section, even after the revision moved on.
//   - A section whose job is gone (cancelled elsewhere, interrupted by a
//     restart, never started) is settled on the next read: ready if any
//     candidate landed, failed if none. Nothing is rerun automatically.

import { randomUUID } from 'node:crypto';
import type Database from 'better-sqlite3';
import { z } from 'zod/v4';
import { bumpRevision } from '../workflows/revisions.js';
import { WorkflowError, type WorkflowJobs, type WorkflowKind } from '../workflows/workflowJobs.js';
import { requestVersion } from '../generation/resolve/resolveIntent.js';

export const SECTION_KIND = 'builder-section';
const MAX_VARIANTS = 16;

/** POST /api/builder/projects/:id/sections/generate */
export const generateSectionSchema = z.object({
  idempotencyKey: z.string().min(1).max(200),
  expectedRevision: z.number().int().min(0),
  direction: z.enum(['first', 'append', 'prepend']),
  label: z.string().max(200).default(''),
  lyrics: z.string().max(20_000).default(''),
  /** Bars need the project's BPM; seconds are used as given. */
  length: z.union([
    z.object({ bars: z.number().int().min(1).max(512) }),
    z.object({ seconds: z.number().positive().max(600) }),
  ]),
  /** Seconds of existing audio the extension overwrites at the seam. */
  overlap: z.number().min(0).max(120).default(4),
  /** Extend-from (append) or connect-at (prepend) point on the song so far;
   *  null = the very end (append) or start (prepend). */
  clipPoint: z.number().min(0).nullable().default(null),
  seedSectionId: z.string().nullable().default(null),
  seedStrength: z.number().min(0).max(1).default(0.4),
  previewMastering: z.boolean().default(false),
  /** The browser's "Keep Models in VRAM" setting at submit. */
  coResident: z.boolean().default(false),
  /** The head's measured length, used only when its song row stores 0
   *  (older repaint sections were saved that way). */
  headDuration: z.number().min(0).optional(),
  /** The engine the browser had active at submit; must still be active. */
  expectedBackend: z.string().min(1),
  /** getGlobalParams() at submit, captured like any other request. */
  engineParams: z.record(z.string(), z.unknown()),
});
export type GenerateSection = z.infer<typeof generateSectionSchema>;

const sectionJobSchema = z.object({
  projectId: z.string(),
  sectionId: z.string(),
  variants: z.number().int().min(1).max(MAX_VARIANTS),
  request: z.record(z.string(), z.unknown()),
  /** The submitted body, for the idempotency check. */
  client: z.record(z.string(), z.unknown()),
});
type SectionJobInput = z.infer<typeof sectionJobSchema>;

interface ProjectRow {
  id: string; user_id: string; title: string; style: string; bpm: number; key_scale: string;
  time_signature: string; vocal_language: string; variant_count: number; revision: number;
}
interface ChosenRow {
  id: string; label: string; lyrics: string; position: number; section_length: number; created_at: string;
  song_duration: number; latent_url: string | null; audio_url: string | null;
}

/** One section's contribution to the cumulative lyric sheet. */
function fmtBlock(label: string, lyrics: string): string {
  const body = (lyrics || '').trim();
  const tag = label ? `[${label}]` : '';
  if (!body) return tag;  // instrumental / structural section → just the tag
  return tag ? `${tag}\n${body}` : body;
}

function ownedProject(db: Database.Database, projectId: string, userId: string): ProjectRow {
  const p = db.prepare('SELECT * FROM builder_projects WHERE id = ? AND user_id = ?').get(projectId, userId) as ProjectRow | undefined;
  if (!p) throw new WorkflowError(404, 'Project not found');
  return p;
}

function ownedSection(db: Database.Database, sectionId: string, userId: string) {
  const section = db.prepare('SELECT * FROM builder_sections WHERE id = ?').get(sectionId) as
    { id: string; project_id: string; status: string; job_id: string | null; candidate_song_ids: string } | undefined;
  if (!section) throw new WorkflowError(404, 'Section not found');
  return { section, project: ownedProject(db, section.project_id, userId) };
}

/** Seconds for the section: bars at the project's tempo, or as given. */
export function sectionSeconds(length: GenerateSection['length'], project: Pick<ProjectRow, 'bpm' | 'time_signature'>): number {
  if ('seconds' in length) return length.seconds;
  if (!(project.bpm > 0)) throw new WorkflowError(400, 'Bars need the project BPM; give the length in seconds');
  const beatsPerBar = parseInt((project.time_signature || '').split('/')[0], 10) || 4;
  return Math.max(1, Math.round((length.bars * beatsPerBar * 60) / project.bpm));
}

/** The /api/generate body for the next section, from the stored project and
 *  the captured composer state. Same body the browser built before. */
export function planSection(db: Database.Database, project: ProjectRow, input: GenerateSection) {
  // Chosen sections with a song, in musical order; the head is the newest.
  const timeline = db.prepare(`SELECT s.id, s.label, s.lyrics, s.position, s.section_length, s.created_at,
      g.duration AS song_duration, g.latent_url, g.audio_url
    FROM builder_sections s JOIN songs g ON g.id = s.chosen_song_id
    WHERE s.project_id = ? AND s.status = 'chosen' ORDER BY s.position ASC, s.created_at ASC`).all(project.id) as ChosenRow[];
  const head = timeline.length ? timeline.reduce((a, b) => (a.created_at >= b.created_at ? a : b)) : null;
  if (input.direction === 'first' && head) throw new WorkflowError(400, 'This song already has a section; extend it with append or prepend');
  if (input.direction !== 'first' && !head) throw new WorkflowError(400, 'Choose a section first to extend from');

  const sectionLength = sectionSeconds(input.length, project);
  const blocks = timeline.map(s => fmtBlock(s.label, s.lyrics));
  const newBlock = fmtBlock(input.label, input.lyrics);
  const lyrics = (input.direction === 'prepend' ? [newBlock, ...blocks] : [...blocks, newBlock]).filter(Boolean).join('\n\n') || '[Instrumental]';

  const ep = input.engineParams;
  const request: Record<string, unknown> = {
    ...ep,
    customMode: true,
    source: 'builder',
    title: `${project.title} — ${input.label || input.direction}`,
    style: project.style || ep.style || '',
    lyrics,
    batchSize: 1,        // one variant per job → options stream in progressively
    randomSeed: true,    // fresh seed per job, so the variants differ
    // Respect "Keep Models in VRAM" rather than forcing it: forced keep-loaded
    // made models pile up across sections until VRAM ran out.
    coResident: input.coResident,
    // Repaint sections never use the LM; free it once past the first section.
    evictLm: input.direction !== 'first',
    // Smaller VAE tiles cut the decode peak (~7 GB → ~2 GB) at negligible cost.
    vaeChunk: 256,
    // Enrichment steps are never useful mid-build.
    coverArtEnabled: false,
    parallelCoverArt: false,
    whisperLyricsEnabled: false,
    qualityEvalEnabled: false,
    parallelQualityEval: false,
    autoTrimEnabled: false,
    skipLrc: true,
  };
  // The cosmetic post-processing chain runs on the finished track only,
  // unless the user asked for a per-section preview.
  if (!input.previewMastering) {
    Object.assign(request, {
      postProcessingEnabled: false, masteringEnabled: false, ppVaeReencode: false,
      stableStepOn: false, spectralLifterEnabled: false, lufsEnabled: false,
    });
  }
  if (project.bpm) request.bpm = project.bpm;
  if (project.key_scale) request.keyScale = project.key_scale;
  if (project.time_signature) request.timeSignature = project.time_signature;
  if (project.vocal_language) request.vocalLanguage = project.vocal_language;

  if (input.direction === 'first') {
    request.duration = sectionLength;
  } else {
    const h = head!;
    const headDuration = h.song_duration > 0 ? h.song_duration : (input.headDuration ?? 0);
    // Overwrite `overlap` seconds at the seam so the prior section's ending
    // (append) or the song start (prepend) is regenerated as a transition.
    // Never overwrite the whole source or more than the new section is long.
    const overlap = Math.max(0, Math.min(input.overlap, headDuration - 1, sectionLength));
    request.taskType = 'repaint';
    request.duration = 0;  // the engine derives it from the source canvas
    if (input.direction === 'prepend') {
      const at = input.clipPoint ?? 0;
      request.repaintingStart = -sectionLength;      // generate before the song
      request.repaintingEnd = at + overlap;          // and overwrite [0, at + overlap]
    } else {
      const from = input.clipPoint ?? headDuration;
      request.repaintingStart = from - overlap;      // overwrite back from the attach point
      // Regenerate the whole tail past `from`, never keeping a sliver of it.
      request.repaintingEnd = Math.max(from + sectionLength, headDuration);
    }
    request.repaintInjectionRatio = 0.5;
    request.repaintCrossfadeFrames = 10;
    if (h.latent_url) request.sourceLatentUrl = h.latent_url;
    if (h.audio_url) request.sourceAudioUrl = h.audio_url;
    if (input.seedSectionId && input.seedStrength > 0) {
      const seed = timeline.find(s => s.id === input.seedSectionId);
      if (seed?.latent_url) {
        request.seedLatentUrl = seed.latent_url;
        request.seedSeconds = seed.section_length;
        request.seedStrength = input.seedStrength;
      }
    }
  }
  const position = input.direction === 'prepend'
    ? ((db.prepare('SELECT MIN(position) AS lo FROM builder_sections WHERE project_id = ?').get(project.id) as { lo: number | null }).lo ?? 0) - 1
    : ((db.prepare('SELECT MAX(position) AS hi FROM builder_sections WHERE project_id = ?').get(project.id) as { hi: number | null }).hi ?? -1) + 1;
  const variants = Math.max(1, project.variant_count || 4);
  if (variants > MAX_VARIANTS) throw new WorkflowError(400, `A section renders at most ${MAX_VARIANTS} variants; this project asks for ${variants}`);
  return { request, sectionLength, position, variants };
}

/** Create the section and its job, or return the ones this key already made. */
export function generateSection(db: Database.Database, jobs: WorkflowJobs, userId: string, projectId: string, input: GenerateSection, activeEngine: string) {
  const project = ownedProject(db, projectId, userId);
  const prior = db.prepare('SELECT id, input FROM workflow_jobs WHERE user_id = ? AND kind = ? AND idempotency_key = ?')
    .get(userId, SECTION_KIND, input.idempotencyKey) as { id: string; input: string } | undefined;
  if (prior) {
    const captured = JSON.parse(prior.input) as SectionJobInput;
    if (captured.projectId !== projectId || requestVersion(captured.client) !== requestVersion(JSON.parse(JSON.stringify(input)))) {
      throw new WorkflowError(409, `Idempotency key '${input.idempotencyKey}' is already used by a different section request`);
    }
    return { job: jobs.get(prior.id), sectionId: captured.sectionId, created: false };
  }
  // The engine is pinned now, as the browser saw it: a section that starts
  // after a switch waits for this engine instead of rendering on another.
  if (input.expectedBackend !== activeEngine) {
    throw new WorkflowError(409, `The active engine is now ${activeEngine}, not ${input.expectedBackend}; reload and generate again`);
  }
  const sectionId = randomUUID();
  // Revision, section, job and their link land together or not at all; the
  // job starts only after the commit.
  const jobId = db.transaction(() => {
    bumpRevision(db, 'builder_projects', project.id, input.expectedRevision, userId);
    const p = planSection(db, project, input);
    p.request.expectedBackend = input.expectedBackend;
    db.prepare(`INSERT INTO builder_sections (id, project_id, position, label, lyrics, direction, section_length, status)
      VALUES (?, ?, ?, ?, ?, ?, ?, 'generating')`)
      .run(sectionId, project.id, p.position, input.label, input.lyrics, input.direction, p.sectionLength);
    db.prepare(`UPDATE builder_projects SET updated_at = datetime('now') WHERE id = ?`).run(project.id);
    const { job } = jobs.submit({
      kind: SECTION_KIND, idempotencyKey: input.idempotencyKey,
      input: { projectId: project.id, sectionId, variants: p.variants, request: p.request, client: input },
    }, userId, { defer: true });
    db.prepare('UPDATE builder_sections SET job_id = ? WHERE id = ?').run(job.id, sectionId);
    return job.id;
  })();
  jobs.pump();
  return { job: jobs.get(jobId), sectionId, created: true };
}

/** Add landed songs to a section's candidates. A deleted section stays
 *  deleted; a choice is never touched. */
export function addCandidates(db: Database.Database, sectionId: string, songIds: string[]): void {
  db.transaction(() => {
    const row = db.prepare('SELECT candidate_song_ids FROM builder_sections WHERE id = ?').get(sectionId) as { candidate_song_ids: string } | undefined;
    if (!row) return;
    const ids = JSON.parse(row.candidate_song_ids || '[]') as string[];
    for (const id of songIds) if (!ids.includes(id)) ids.push(id);
    // A section settled as failed before this landed (its job retried after
    // a restart) has something to choose from now.
    db.prepare(`UPDATE builder_sections SET candidate_song_ids = ?, updated_at = datetime('now'),
      status = CASE WHEN status = 'failed' THEN 'ready' ELSE status END WHERE id = ?`).run(JSON.stringify(ids), sectionId);
  })();
}

/** Move a generating section to ready (has candidates) or failed (none). */
export function settleSection(db: Database.Database, sectionId: string): void {
  db.prepare(`UPDATE builder_sections SET updated_at = datetime('now'),
    status = CASE WHEN candidate_song_ids IS NOT NULL AND candidate_song_ids <> '[]' THEN 'ready' ELSE 'failed' END
    WHERE id = ? AND status = 'generating'`).run(sectionId);
}

const LIVE = new Set(['pending', 'running']);

/** Settle generating sections whose job is no longer live. Run on read. */
export function reconcileSections(db: Database.Database, jobs: WorkflowJobs, projectId: string): void {
  const rows = db.prepare(`SELECT id, job_id FROM builder_sections WHERE project_id = ? AND status = 'generating'`).all(projectId) as Array<{ id: string; job_id: string | null }>;
  for (const r of rows) {
    let live = false;
    if (r.job_id) { try { live = LIVE.has(jobs.get(r.job_id).status); } catch { /* job gone */ } }
    if (!live) settleSection(db, r.id);
  }
}

function cancelJob(jobs: WorkflowJobs, jobId: string | null): void {
  if (!jobId) return;
  try { jobs.cancel(jobId); } catch { /* already finished */ }
}

/** Choose a candidate. Stops the section's remaining variants, and fills the
 *  project's auto BPM/key from the chosen song so later sections inherit them. */
export function chooseCandidate(db: Database.Database, jobs: WorkflowJobs, userId: string, sectionId: string, songId: string, expectedRevision: number): void {
  const { section, project } = ownedSection(db, sectionId, userId);
  db.transaction(() => {
    bumpRevision(db, 'builder_projects', project.id, expectedRevision, userId);
    const now = db.prepare('SELECT candidate_song_ids FROM builder_sections WHERE id = ?').get(sectionId) as { candidate_song_ids: string };
    if (!(JSON.parse(now.candidate_song_ids || '[]') as string[]).includes(songId)) {
      throw new WorkflowError(409, 'That song is not one of this section\'s candidates');
    }
    db.prepare(`UPDATE builder_sections SET chosen_song_id = ?, status = 'chosen', updated_at = datetime('now') WHERE id = ?`).run(songId, sectionId);
    const song = db.prepare('SELECT bpm, key_scale FROM songs WHERE id = ?').get(songId) as { bpm: number | null; key_scale: string | null } | undefined;
    if (!project.bpm && song?.bpm) db.prepare('UPDATE builder_projects SET bpm = ? WHERE id = ?').run(Math.round(song.bpm), project.id);
    if (!project.key_scale && song?.key_scale) db.prepare('UPDATE builder_projects SET key_scale = ? WHERE id = ?').run(song.key_scale, project.id);
    db.prepare(`UPDATE builder_projects SET updated_at = datetime('now') WHERE id = ?`).run(project.id);
  })();
  cancelJob(jobs, section.job_id);
}

/** Stop the remaining variants, keeping the candidates that already landed. */
export function stopSection(db: Database.Database, jobs: WorkflowJobs, userId: string, sectionId: string, expectedRevision: number): void {
  const { section, project } = ownedSection(db, sectionId, userId);
  db.transaction(() => {
    bumpRevision(db, 'builder_projects', project.id, expectedRevision, userId);
    settleSection(db, sectionId);
  })();
  cancelJob(jobs, section.job_id);
}

export function deleteSection(db: Database.Database, jobs: WorkflowJobs, userId: string, sectionId: string, expectedRevision: number): void {
  const { section, project } = ownedSection(db, sectionId, userId);
  db.transaction(() => {
    bumpRevision(db, 'builder_projects', project.id, expectedRevision, userId);
    db.prepare('DELETE FROM builder_sections WHERE id = ?').run(sectionId);
    db.prepare(`UPDATE builder_projects SET updated_at = datetime('now') WHERE id = ?`).run(project.id);
  })();
  cancelJob(jobs, section.job_id);
}

/** Edit a section's label and lyrics (the sheet fed into later sections). */
export function editSection(db: Database.Database, userId: string, sectionId: string, edit: { label?: string; lyrics?: string }, expectedRevision: number): void {
  const { project } = ownedSection(db, sectionId, userId);
  db.transaction(() => {
    bumpRevision(db, 'builder_projects', project.id, expectedRevision, userId);
    if (edit.label !== undefined) db.prepare('UPDATE builder_sections SET label = ? WHERE id = ?').run(edit.label, sectionId);
    if (edit.lyrics !== undefined) db.prepare('UPDATE builder_sections SET lyrics = ? WHERE id = ?').run(edit.lyrics, sectionId);
    db.prepare(`UPDATE builder_sections SET updated_at = datetime('now') WHERE id = ?`).run(sectionId);
  })();
}

/** The builder-section job: queue the variants, add each candidate as it
 *  lands, then settle the section. */
export function sectionKind(dbOf: () => Database.Database): WorkflowKind<SectionJobInput> {
  return {
    kind: SECTION_KIND,
    input: sectionJobSchema,
    // Variants queue behind every other GPU job; the per-render timeout in
    // the request still applies to each of them.
    timeoutMs: 6 * 3600_000,
    // A section job holds no GPU, it only waits on the audio queue, which
    // admits the renders. One project's section must not hold up another's.
    maxConcurrent: 16,
    async run(ctx) {
      const { sectionId, request, variants } = ctx.input;
      const items = Array.from({ length: variants }, (_, i) =>
        ctx.audio.enqueue(`variant-${i}`, request, { builderSectionId: sectionId }));
      await Promise.all(items.map(async item => {
        const done = await ctx.audio.wait(item.id);
        const songIds = done.status === 'succeeded' && Array.isArray(done.result?.songIds) ? done.result.songIds as string[] : [];
        ctx.throwIfCancelled();
        if (songIds.length) addCandidates(dbOf(), sectionId, songIds);
        ctx.emit('variant', { intentId: item.id, status: done.status, songIds, error: done.error });
      }));
      ctx.throwIfCancelled();
      settleSection(dbOf(), sectionId);
      return { sectionId };
    },
  };
}
