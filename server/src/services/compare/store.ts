// store.ts - SQLite access for the Compare studio (sessions, takes, scores,
// cached audio metrics). All functions query lazily via getDb().

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { getDb } from '../../db/database.js';
import { registerBusyCheck } from '../../dataroot/pointer.js';
import { config } from '../../config.js';
import { analyze, ANALYZER_VERSION, type AnalysisResult } from './analyzer.js';

export interface Criterion {
  id: string;
  label: string;
  type: 'scale' | 'keep' | 'note';
  min?: number;
  max?: number;
}

export const DEFAULT_CRITERIA: Criterion[] = [
  { id: 'character', label: 'Character', type: 'scale', min: 1, max: 5 },
  { id: 'quality', label: 'Quality', type: 'scale', min: 1, max: 5 },
  { id: 'keep', label: 'Keep', type: 'keep' },
  { id: 'note', label: 'Note', type: 'note' },
];

export type SessionStatus = 'scored' | 'partial' | 'unscored';

export class NotFoundError extends Error {}
export class BadRequestError extends Error {}

interface SessionRow {
  id: string; name: string; criteria_json: string; baseline_song_id: string | null;
  group_name: string; archived: number; created_at: string; updated_at: string;
}

function songExists(songId: string): boolean {
  return !!getDb().prepare('SELECT 1 FROM songs WHERE id = ?').get(songId);
}

function getSessionRow(id: string): SessionRow {
  const row = getDb().prepare('SELECT * FROM compare_sessions WHERE id = ?').get(id) as SessionRow | undefined;
  if (!row) throw new NotFoundError('Session not found');
  return row;
}

/** Scored-take count for a session: a take counts if any criterion has a value,
 *  a keep flag, or a non-empty note. */
function scoreCounts(sessionId: string): { scored: number; total: number } {
  const db = getDb();
  const total = (db.prepare('SELECT COUNT(*) AS c FROM compare_takes WHERE session_id = ?').get(sessionId) as any).c as number;
  const scored = (db.prepare(`
    SELECT COUNT(DISTINCT t.song_id) AS c FROM compare_takes t
    JOIN compare_scores s ON s.session_id = t.session_id AND s.song_id = t.song_id
    WHERE t.session_id = ? AND (s.value IS NOT NULL OR s.keep IS NOT NULL OR s.note <> '')
  `).get(sessionId) as any).c as number;
  return { scored, total };
}

function statusOf(scored: number, total: number): SessionStatus {
  if (total > 0 && scored >= total) return 'scored';
  return scored > 0 ? 'partial' : 'unscored';
}

function summarize(row: SessionRow) {
  const { scored, total } = scoreCounts(row.id);
  return {
    id: row.id,
    name: row.name,
    group: row.group_name,
    baselineSongId: row.baseline_song_id,
    archived: !!row.archived,
    status: statusOf(scored, total),
    scored,
    total,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

export function listSessions(opts: { archived?: boolean; group?: string }) {
  const where: string[] = [];
  const args: unknown[] = [];
  if (opts.archived !== undefined) { where.push('archived = ?'); args.push(opts.archived ? 1 : 0); }
  if (opts.group !== undefined) { where.push('group_name = ?'); args.push(opts.group); }
  const sql = `SELECT * FROM compare_sessions ${where.length ? 'WHERE ' + where.join(' AND ') : ''} ORDER BY updated_at DESC`;
  return (getDb().prepare(sql).all(...args) as SessionRow[]).map(summarize);
}

function validateCriteria(raw: unknown): Criterion[] {
  if (raw === undefined || raw === null) return DEFAULT_CRITERIA;
  if (!Array.isArray(raw) || raw.length === 0) throw new BadRequestError('criteria must be a non-empty array');
  if (raw.length > MAX_CRITERIA) throw new BadRequestError(`criteria: at most ${MAX_CRITERIA} allowed`);
  const seen = new Set<string>();
  return raw.map((c: any) => {
    if (!c || typeof c.id !== 'string' || !c.id) throw new BadRequestError('each criterion needs an id');
    if (c.id.length > MAX_NAME_LEN || (typeof c.label === 'string' && c.label.length > MAX_NAME_LEN)) throw new BadRequestError('criterion id/label too long');
    if (!['scale', 'keep', 'note'].includes(c.type)) throw new BadRequestError(`criterion ${c.id}: bad type`);
    if (seen.has(c.id)) throw new BadRequestError(`duplicate criterion ${c.id}`);
    seen.add(c.id);
    const out: Criterion = { id: c.id, label: typeof c.label === 'string' ? c.label : c.id, type: c.type };
    if (c.type === 'scale') {
      out.min = Number.isInteger(c.min) ? c.min : 1;
      out.max = Number.isInteger(c.max) ? c.max : 5;
    }
    return out;
  });
}

function addTakesTx(sessionId: string, songIds: string[]) {
  const db = getDb();
  const missing = songIds.filter(id => !songExists(id));
  if (missing.length) throw new NotFoundError(`Song not found: ${missing.join(', ')}`);
  const maxPos = (db.prepare('SELECT COALESCE(MAX(position), -1) AS p FROM compare_takes WHERE session_id = ?').get(sessionId) as any).p as number;
  const ins = db.prepare('INSERT OR IGNORE INTO compare_takes (session_id, song_id, position) VALUES (?, ?, ?)');
  let pos = maxPos + 1;
  for (const id of songIds) if (ins.run(sessionId, id, pos).changes) pos++;
}

function validateSongIds(raw: unknown, allowEmpty = false): string[] {
  if (!Array.isArray(raw) || (!allowEmpty && raw.length === 0) || !raw.every(s => typeof s === 'string' && s)) {
    throw new BadRequestError('songIds must be a non-empty array of strings');
  }
  if (raw.length > MAX_SONG_IDS) throw new BadRequestError(`songIds: at most ${MAX_SONG_IDS} per request`);
  return [...new Set(raw as string[])];
}

export const MAX_SONG_IDS = 200;
export const MAX_NAME_LEN = 200;
export const MAX_NOTE_LEN = 4000;
export const MAX_CRITERIA = 20;

/** Remove compare takes, scores and cached metrics for deleted songs. Call after deleting songs. */
export function cleanupDeletedSongs(songIds: string[]) {
  if (!songIds.length) return;
  const db = getDb();
  const tx = db.transaction((ids: string[]) => {
    for (let i = 0; i < ids.length; i += 500) {
      const chunk = ids.slice(i, i + 500);
      const ph = chunk.map(() => '?').join(',');
      db.prepare(`DELETE FROM compare_scores WHERE song_id IN (${ph})`).run(...chunk);
      db.prepare(`DELETE FROM compare_takes WHERE song_id IN (${ph})`).run(...chunk);
      db.prepare(`DELETE FROM audio_metrics WHERE song_id IN (${ph})`).run(...chunk);
    }
  });
  tx(songIds);
}

export function createSession(body: any) {
  if (!body || typeof body.name !== 'string' || !body.name.trim()) throw new BadRequestError('name is required');
  if (body.name.length > MAX_NAME_LEN) throw new BadRequestError(`name: at most ${MAX_NAME_LEN} chars`);
  if (typeof body.group === 'string' && body.group.length > MAX_NAME_LEN) throw new BadRequestError(`group: at most ${MAX_NAME_LEN} chars`);
  const songIds = validateSongIds(body.songIds);
  const criteria = validateCriteria(body.criteria);
  const baseline = body.baselineSongId ?? null;
  if (baseline !== null && (typeof baseline !== 'string' || !songIds.includes(baseline))) {
    throw new BadRequestError('baselineSongId must be one of songIds');
  }
  const group = typeof body.group === 'string' ? body.group : '';
  const id = crypto.randomUUID();
  const db = getDb();
  db.transaction(() => {
    db.prepare('INSERT INTO compare_sessions (id, name, criteria_json, baseline_song_id, group_name) VALUES (?, ?, ?, ?, ?)')
      .run(id, body.name.trim(), JSON.stringify(criteria), baseline, group);
    addTakesTx(id, songIds);
  })();
  return getSession(id);
}

function parseJson(s: string | null | undefined): any {
  try { return s ? JSON.parse(s) : {}; } catch { return {}; }
}

export function getSession(id: string) {
  const row = getSessionRow(id);
  const db = getDb();
  const takes = db.prepare(`
    SELECT t.song_id, t.position, s.title, s.audio_url, s.duration, s.backend, s.generation_params, s.created_at
    FROM compare_takes t LEFT JOIN songs s ON s.id = t.song_id
    WHERE t.session_id = ? ORDER BY t.position
  `).all(id) as any[];
  const scores = db.prepare('SELECT song_id, criterion, value, keep, note, updated_at FROM compare_scores WHERE session_id = ?').all(id) as any[];
  return {
    ...summarize(row),
    criteria: parseJson(row.criteria_json) as Criterion[],
    takes: takes.map(t => {
      const gp = parseJson(t.generation_params);
      return {
        songId: t.song_id,
        position: t.position,
        title: t.title ?? null,
        audioUrl: t.audio_url || null,
        duration: t.duration ?? null,
        backend: t.backend ?? null,
        createdAt: t.created_at ?? null,
        seed: gp.seed ?? null,
        seedB: gp.seedB ?? null,
        seedBlend: gp.seedBlend ?? null,
        generationParams: gp,
        scores: scores.filter(s => s.song_id === t.song_id).map(s => ({
          criterion: s.criterion, value: s.value, keep: s.keep === null ? null : !!s.keep, note: s.note, updatedAt: s.updated_at,
        })),
      };
    }),
  };
}

export function addTakes(id: string, rawSongIds: unknown) {
  getSessionRow(id);
  const songIds = validateSongIds(rawSongIds);
  const db = getDb();
  db.transaction(() => {
    addTakesTx(id, songIds);
    db.prepare("UPDATE compare_sessions SET updated_at = datetime('now') WHERE id = ?").run(id);
  })();
  return getSession(id);
}

export function removeTake(id: string, songId: string) {
  getSessionRow(id);
  const db = getDb();
  db.transaction(() => {
    const r = db.prepare('DELETE FROM compare_takes WHERE session_id = ? AND song_id = ?').run(id, songId);
    if (!r.changes) throw new NotFoundError('Take not found');
    db.prepare('DELETE FROM compare_scores WHERE session_id = ? AND song_id = ?').run(id, songId);
    db.prepare("UPDATE compare_sessions SET updated_at = datetime('now') WHERE id = ?").run(id);
  })();
  return getSession(id);
}

export function deleteSession(id: string) {
  getSessionRow(id);
  const db = getDb();
  db.transaction(() => {
    db.prepare('DELETE FROM compare_scores WHERE session_id = ?').run(id);
    db.prepare('DELETE FROM compare_takes WHERE session_id = ?').run(id);
    db.prepare('DELETE FROM compare_sessions WHERE id = ?').run(id);
  })();
}

/** Upsert scores. Each item sets only the fields it carries. When every take is
 *  scored the session is archived (idempotent; never un-archives). */
export function putScores(id: string, items: unknown) {
  const row = getSessionRow(id);
  if (!Array.isArray(items) || items.length === 0) throw new BadRequestError('body must be a non-empty array of scores');
  const criteria = parseJson(row.criteria_json) as Criterion[];
  const db = getDb();
  const takeIds = new Set((db.prepare('SELECT song_id FROM compare_takes WHERE session_id = ?').all(id) as any[]).map(r => r.song_id));
  for (const it of items as any[]) {
    if (!it || typeof it.songId !== 'string' || !takeIds.has(it.songId)) throw new BadRequestError(`songId ${it?.songId} is not a take in this session`);
    const crit = criteria.find(c => c.id === it.criterion);
    if (!crit) throw new BadRequestError(`unknown criterion ${it.criterion}`);
    if ('value' in it && it.value !== null) {
      if (!Number.isInteger(it.value)) throw new BadRequestError('value must be an integer or null');
      if (crit.type === 'scale' && (it.value < (crit.min ?? 1) || it.value > (crit.max ?? 5))) {
        throw new BadRequestError(`value for ${crit.id} must be ${crit.min}-${crit.max}`);
      }
    }
    if ('keep' in it && it.keep !== null && typeof it.keep !== 'boolean') throw new BadRequestError('keep must be boolean or null');
    if ('note' in it && typeof it.note !== 'string') throw new BadRequestError('note must be a string');
    if ('note' in it && it.note.length > MAX_NOTE_LEN) throw new BadRequestError(`note: at most ${MAX_NOTE_LEN} chars`);
  }
  let archived = !!row.archived;
  db.transaction(() => {
    const ensure = db.prepare('INSERT OR IGNORE INTO compare_scores (session_id, song_id, criterion) VALUES (?, ?, ?)');
    for (const it of items as any[]) {
      ensure.run(id, it.songId, it.criterion);
      const sets: string[] = [];
      const args: unknown[] = [];
      if ('value' in it) { sets.push('value = ?'); args.push(it.value); }
      if ('keep' in it) { sets.push('keep = ?'); args.push(it.keep === null ? null : it.keep ? 1 : 0); }
      if ('note' in it) { sets.push('note = ?'); args.push(it.note); }
      sets.push("updated_at = datetime('now')");
      db.prepare(`UPDATE compare_scores SET ${sets.join(', ')} WHERE session_id = ? AND song_id = ? AND criterion = ?`)
        .run(...args, id, it.songId, it.criterion);
    }
    const { scored, total } = scoreCounts(id);
    // Lenient by design (Alex's choice): any criterion value, keep flag or non-empty note counts a take as scored.
    if (!archived && statusOf(scored, total) === 'scored') archived = true;
    db.prepare("UPDATE compare_sessions SET archived = ?, updated_at = datetime('now') WHERE id = ?").run(archived ? 1 : 0, id);
  })();
  const { scored, total } = scoreCounts(id);
  return { status: statusOf(scored, total), archived, scored, total };
}

export function setArchived(id: string, archived: unknown) {
  getSessionRow(id);
  if (typeof archived !== 'boolean') throw new BadRequestError('archived must be boolean');
  getDb().prepare("UPDATE compare_sessions SET archived = ?, updated_at = datetime('now') WHERE id = ?").run(archived ? 1 : 0, id);
  return summarize(getSessionRow(id));
}

// ── Audio analysis ──────────────────────────────────────────────────────────

/** Resolve a song's audio file under the served /audio root, or throw. */
export function songAudioPath(songId: string): string {
  const song = getDb().prepare('SELECT audio_url FROM songs WHERE id = ?').get(songId) as { audio_url: string } | undefined;
  if (!song) throw new NotFoundError('Song not found');
  const url = (song.audio_url || '').split('?')[0];
  if (!url.startsWith('/audio/')) throw new NotFoundError('Song has no local audio file');
  let rel: string;
  try { rel = decodeURIComponent(url.slice('/audio/'.length)); } catch { throw new NotFoundError('Bad audio url'); }
  const root = path.resolve(config.data.audioDir);
  const full = path.resolve(root, rel);
  if (!full.startsWith(root + path.sep)) throw new NotFoundError('Bad audio url');
  if (!fs.existsSync(full)) throw new NotFoundError('Audio file missing on disk');
  return full;
}

// One analysis at a time so a burst of requests does not hog the CPU.
let queue: Promise<unknown> = Promise.resolve();
let queued = 0;
function enqueue<T>(fn: () => Promise<T>): Promise<T> {
  const run = queue.then(fn, fn);
  queue = run.catch(() => undefined);
  queued++;
  run.then(() => { queued--; }, () => { queued--; });
  return run;
}
registerBusyCheck('compare', () => queued ? [`${queued} comparison analysis job(s) running or queued`] : []);
const inflight = new Map<string, Promise<AnalysisResult>>();

export async function getAnalysis(songId: string): Promise<{ cached: boolean; result: AnalysisResult }> {
  const file = songAudioPath(songId);
  const mtime = fs.statSync(file).mtimeMs;
  const db = getDb();
  const row = db.prepare('SELECT file_mtime, metrics_json, peaks_json FROM audio_metrics WHERE song_id = ? AND analyzer_ver = ?')
    .get(songId, ANALYZER_VERSION) as any;
  if (row && row.file_mtime === mtime) {
    return { cached: true, result: { ...JSON.parse(row.metrics_json), peaks: JSON.parse(row.peaks_json) } };
  }
  const key = `${songId}:${mtime}`;
  let p = inflight.get(key);
  if (!p) {
    p = enqueue(() => analyze(file));
    inflight.set(key, p);
    p.finally(() => inflight.delete(key)).catch(() => undefined);
  }
  const result = await p;
  const { peaks, ...metrics } = result;
  getDb().prepare('INSERT OR REPLACE INTO audio_metrics (song_id, file_mtime, analyzer_ver, metrics_json, peaks_json) VALUES (?, ?, ?, ?, ?)')
    .run(songId, mtime, ANALYZER_VERSION, JSON.stringify(metrics), JSON.stringify(peaks));
  return { cached: false, result };
}

export function runQueued<T>(fn: () => Promise<T>): Promise<T> {
  return enqueue(fn);
}

/** Paginated library search for the New comparison picker. Matches title,
 *  style and caption; newest first. Never returns the whole library. */
export function searchSongs(opts: { q?: string; limit?: number; offset?: number }) {
  const limit = Math.min(Math.max(Math.floor(opts.limit ?? 25), 1), 100);
  const offset = Math.max(Math.floor(opts.offset ?? 0), 0);
  const q = (opts.q ?? '').trim();
  const like = `%${q.replace(/[!%_]/g, m => '!' + m)}%`;
  const args = q ? [like, like, like] : [];
  const whereSql = q ? "WHERE title LIKE ? ESCAPE '!' OR style LIKE ? ESCAPE '!' OR caption LIKE ? ESCAPE '!'" : '';
  const db = getDb();
  const total = (db.prepare(`SELECT COUNT(*) AS n FROM songs ${whereSql}`).get(...args) as { n: number }).n;
  const rows = db.prepare(
    `SELECT id, title, style, duration, backend, created_at FROM songs ${whereSql} ORDER BY created_at DESC LIMIT ? OFFSET ?`,
  ).all(...args, limit, offset) as Array<{ id: string; title: string; style: string; duration: number; backend: string; created_at: string }>;
  return {
    total, limit, offset,
    songs: rows.map(r => ({ id: r.id, title: r.title, style: r.style, duration: r.duration, backend: r.backend, createdAt: r.created_at })),
  };
}
