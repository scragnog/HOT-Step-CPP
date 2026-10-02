// YuE2 cover transcription boundary. The existing training queue owns the job.
import fs from 'node:fs';
import path from 'node:path';
import { config } from '../config.js';
import { getDb } from '../db/database.js';
import * as queue from './training/labelingQueue.js';
import { jobsDir } from './training/paths.js';
import { readDuration } from './training/audioMeta.js';
import { missingYue2SheetModels, readYue2CoverAbc, resolveYue2SheetModel } from './training/yue2Sheet.js';
import { runYue2CoverSheetJob } from './training/yue2ArTrainRunner.js';

const MAX_AUDIO_BYTES = 100 * 1024 * 1024;
const MAX_AUDIO_SECONDS = 10 * 60;
const MAX_ABC_LENGTH = 64 * 1024;
const AUDIO_EXT = new Set(['.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.aac']);

export class CoverRequestError extends Error {
  constructor(message: string, public readonly status = 400) { super(message); }
}

export interface CoverSource { sourceId: string; sourceLabel: string; audioPath: string }
type SavedSource = Pick<CoverSource, 'sourceId' | 'sourceLabel'> & { userId: string; name: string };
type CoverInput = { sourceAudioUrl?: unknown; songId?: unknown; sourceLabel?: unknown; abc?: unknown };

export function createYue2CoverService(deps = {
  queue,
  referenceDir: path.join(config.data.dir, 'references'),
  libraryDir: config.data.audioDir,
  jobRoot: jobsDir(),
  duration: readDuration,
  missingModels: missingYue2SheetModels,
  model: resolveYue2SheetModel,
  run: runYue2CoverSheetJob,
  song: (id: string, userId: string) => getDb().prepare('SELECT audio_url, title FROM songs WHERE id = ? AND user_id = ?').get(id, userId) as { audio_url: string; title: string } | undefined,
}) {
  function readiness() {
    const missing = deps.missingModels();
    return { ready: missing.length === 0, model: missing.length ? null : path.basename(deps.model()),
      message: missing.length ? 'SheetSage2 is missing. Download the SheetSage2 Lead-Sheet Transcriber (yue2-sheetsage2-f16) in Model Manager, or supply an approved ABC score.' : null };
  }

  async function source(input: CoverInput, userId: string): Promise<CoverSource> {
    if (input.songId !== undefined && input.sourceAudioUrl !== undefined) throw new CoverRequestError('Choose a library song or an uploaded source, not both.');
    let sourceId: string;
    let sourceLabel: string;
    let url: string;
    let root: string;
    if (typeof input.songId === 'string' && input.songId.length > 0 && input.songId.length <= 128) {
      const song = deps.song(input.songId, userId);
      if (!song) throw new CoverRequestError('Library song not found', 404);
      sourceId = input.songId;
      sourceLabel = song.title || input.songId;
      url = song.audio_url;
      root = deps.libraryDir;
      if (!url.startsWith('/audio/')) throw new CoverRequestError('Library song has no local audio source.');
    } else if (typeof input.sourceAudioUrl === 'string') {
      sourceId = input.sourceAudioUrl;
      sourceLabel = input.sourceAudioUrl.split('/').at(-1) || '';
      url = input.sourceAudioUrl;
      root = deps.referenceDir;
      if (!url.startsWith('/references/')) throw new CoverRequestError('Use an uploaded /references/ source or a library songId.');
    } else {
      throw new CoverRequestError('sourceAudioUrl or songId is required.');
    }
    const name = url.slice(url.lastIndexOf('/') + 1);
    if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,180}$/.test(name) || url !== `${root === deps.libraryDir ? '/audio' : '/references'}/${name}` || !AUDIO_EXT.has(path.extname(name).toLowerCase())) {
      throw new CoverRequestError('Invalid audio source.');
    }
    const audioPath = path.join(root, name);
    let stat: fs.Stats;
    try {
      stat = fs.lstatSync(audioPath);
      if (!stat.isFile()) throw new Error('not a file');
    } catch { throw new CoverRequestError('Audio source not found. Re-upload or choose another library song.', 404); }
    if (stat.size === 0 || stat.size > MAX_AUDIO_BYTES) throw new CoverRequestError('Audio source must be between 1 byte and 100 MB.');
    const duration = await deps.duration(audioPath);
    if (!Number.isFinite(duration) || duration <= 0 || duration > MAX_AUDIO_SECONDS) throw new CoverRequestError('Audio source must have a readable duration of at most 10 minutes.');
    if (input.sourceLabel !== undefined) {
      if (typeof input.sourceLabel !== 'string' || input.sourceLabel.trim().length > 120) throw new CoverRequestError('sourceLabel must be at most 120 characters.');
      if (input.sourceLabel.trim()) sourceLabel = input.sourceLabel.trim();
    }
    return { sourceId, sourceLabel, audioPath };
  }

  async function start(input: CoverInput, userId: string) {
    const selected = await source(input, userId);
    if (input.abc !== undefined) {
      if (typeof input.abc !== 'string' || !input.abc.trim() || input.abc.length > MAX_ABC_LENGTH) throw new CoverRequestError('ABC must be non-empty and at most 64 KB.');
      return { status: 'done' as const, abc: input.abc.trim(), sourceId: selected.sourceId, sourceLabel: selected.sourceLabel };
    }
    const ready = readiness();
    if (!ready.ready) throw new CoverRequestError(ready.message!, 400);
    const job = deps.queue.createJob('yue2-sheet', `cover:${userId}`, [], { sourceId: selected.sourceId });
    const dir = path.join(deps.jobRoot, job.id);
    try {
      fs.mkdirSync(dir, { recursive: true });
      const saved: SavedSource = { userId, sourceId: selected.sourceId, sourceLabel: selected.sourceLabel, name: path.basename(selected.audioPath) };
      fs.writeFileSync(path.join(dir, 'cover-source.json'), JSON.stringify(saved), { mode: 0o600 });
    } catch (err) {
      deps.queue.cancelJob(job.id);
      throw err;
    }
    deps.queue.enqueue(job, async j => {
      if (j.controller.signal.aborted) return;
      try { await deps.run(j, selected.audioPath, dir); } catch { /* runner records failed/cancelled status */ }
    });
    return { status: 'queued' as const, jobId: job.id, sourceId: selected.sourceId, sourceLabel: selected.sourceLabel };
  }

  function find(jobId: string, userId: string) {
    if (!/^[0-9a-f-]{36}$/.test(jobId)) throw new CoverRequestError('Cover job not found', 404);
    let saved: SavedSource;
    try { saved = JSON.parse(fs.readFileSync(path.join(deps.jobRoot, jobId, 'cover-source.json'), 'utf8')) as SavedSource; }
    catch { throw new CoverRequestError('Cover job not found', 404); }
    if (saved.userId !== userId) throw new CoverRequestError('Cover job not found', 404);
    const job = deps.queue.getJob(jobId);
    const summary = job ? deps.queue.toSummary(job) : deps.queue.listJobs().find(j => j.id === jobId);
    if (!summary) throw new CoverRequestError('Cover job not found', 404);
    const abc = summary.status === 'done' ? readYue2CoverAbc(path.join(deps.jobRoot, jobId, 'cover-sheet.json'), saved.name) : undefined;
    return { job: summary, sourceId: saved.sourceId, sourceLabel: saved.sourceLabel, ...(abc ? { abc } : {}) };
  }

  function cancel(jobId: string, userId: string) {
    const current = find(jobId, userId);
    if (current.job.status === 'queued' || current.job.status === 'running') deps.queue.cancelJob(jobId);
    return find(jobId, userId);
  }

  return { readiness, source, start, find, cancel };
}

export const yue2CoverService = createYue2CoverService();
