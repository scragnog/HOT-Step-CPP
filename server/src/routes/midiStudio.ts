// midiStudio.ts — MIDI Studio audio→MIDI transcription route
//
// Transcribes library tracks to multi-instrument MIDI via the NATIVE ace-midi
// engine binary (MuScriptor GGML port — docs/plans/muscriptor-cpp-port.md).
// ace-midi reads WAV/MP3 directly (FLAC etc. are converted first), streams note events as JSONL on stdout
// (relayed live over SSE for the piano roll), and writes the .mid.
// Weights are gated on Hugging Face; the download endpoints fetch them with
// the user's stored read token. Results persist to data/midi/<jobId>/.
//
// Mounts at: /api/midi-studio
// Routes:
//   GET    /api/midi-studio/status                — engine/models/token status
//   POST   /api/midi-studio/hf-token              — save/clear HF token
//   POST   /api/midi-studio/models/:size/download — download gated weights
//   POST   /api/midi-studio/transcribe            — queue a transcription job
//   GET    /api/midi-studio/jobs                  — list jobs (disk + active)
//   GET    /api/midi-studio/:jobId/progress       — poll a job
//   GET    /api/midi-studio/:jobId/stream         — SSE: live note events
//   GET    /api/midi-studio/:jobId/notes          — parsed notes (piano roll)
//   POST   /api/midi-studio/:jobId/export         — re-export on a corrected / quantized grid
//   POST   /api/midi-studio/:jobId/abc            — YuE2 ABC lead sheet from the transcription
//   GET    /api/midi-studio/:jobId/file           — download the .mid
//   DELETE /api/midi-studio/:jobId                — cancel/delete a job

import { Router, Request, Response } from 'express';
import path from 'path';
import fs from 'fs';
import { spawn, ChildProcess } from 'child_process';
import { randomUUID } from 'crypto';
import readline from 'readline';
import { config } from '../config.js';
import {
  getHfToken, setHfToken, looksLikeGatedError, aceMidiExe,
  modelDir, isModelDownloaded, getModelStates,
  MUSCRIPTOR_MODELS, type MuscriptorModel, partialMidiFromEvents,
  BEAT_MODEL, readGrid, alignToAudio, exportArgs, speculativeArgs, type MidiExportOptions,
} from '../services/muscriptor.js';
import { modelDownloadService } from '../services/modelDownloadService.js';
import { parseMidiFile } from '../services/midiParser.js';
import { engineFormatPath } from '../services/audioConvert.js';
import { exportMidiAbc, MidiAbcError, coverMidiModel } from '../services/midiAbc.js';
import { tryAbcCoverLeadSheet } from '../services/coverLeadSheet.js';
import { yue2CoverService } from '../services/yue2Cover.js';
import { addMidiLyrics } from '../services/midiLyrics.js';
import { registerJobsBusyCheck, trackBusy } from '../dataroot/pointer.js';
import { runOnGpuLane } from '../services/generation/gpuLane.js';

const router = Router();

const midiBaseDir = path.join(config.data.dir, 'midi');
fs.mkdirSync(midiBaseDir, { recursive: true });

/** Kill ace-midi if it prints nothing for this long (#227: a fixed total
 *  timeout killed long songs that were still making chunk progress). */
const TRANSCRIBE_IDLE_TIMEOUT_MS = 20 * 60 * 1000;
/** Hard cap on one transcription, however steady its progress. */
const TRANSCRIBE_MAX_MS = 6 * 60 * 60 * 1000;

/** Fires onTimeout('idle') when touch() is not called for idleMs, or
 *  onTimeout('cap') after capMs total. touch() on each line of output. */
export function startStallTimer(onTimeout: (why: 'idle' | 'cap') => void, idleMs: number, capMs: number) {
  let idle = setTimeout(() => onTimeout('idle'), idleMs);
  const cap = setTimeout(() => onTimeout('cap'), capMs);
  return {
    touch() {
      clearTimeout(idle);
      idle = setTimeout(() => onTimeout('idle'), idleMs);
    },
    clear() {
      clearTimeout(idle);
      clearTimeout(cap);
    },
  };
}

interface MidiJob {
  id: string;
  status: 'queued' | 'transcribing' | 'done' | 'failed' | 'cancelled';
  sourceAudioUrl: string;
  sourceFileName: string;
  songId?: string;
  model: MuscriptorModel;
  /** speculative decoding with the small model as draft (large only) */
  speculative: boolean;
  // live event stream (JSONL objects from ace-midi, replayed to SSE clients)
  events: any[];
  chunksDone: number;
  chunksTotal: number;
  noteCount: number;
  error?: string;
  gated?: boolean;
  /** A failed run whose streamed notes were saved as out.mid. */
  partial?: boolean;
  createdAt: number;
  child?: ChildProcess;
  listeners: Set<Response>;
}

const jobs = new Map<string, MidiJob>();
registerJobsBusyCheck('MIDI', jobs);

/** Run MIDI work on the shared GPU lane, behind any render or post-process
 *  already running or queued (#227: running beside YuE2 starved both). */
export function runOnMidiLane(jobId: string, fn: () => Promise<void>): Promise<void> {
  return runOnGpuLane(fn, { label: 'midi:' + jobId, family: 'midi' });
}

/** Resolve a URL-style audio path to an absolute filesystem path */
function resolveAudioPath(audioUrl: string): string {
  if (audioUrl.startsWith('/references/')) {
    return path.join(config.data.dir, 'references', path.basename(audioUrl));
  }
  if (audioUrl.startsWith('/audio/')) {
    return path.join(config.data.audioDir, path.basename(audioUrl));
  }
  if (path.isAbsolute(audioUrl)) {
    return audioUrl;
  }
  return path.join(config.data.dir, 'references', path.basename(audioUrl));
}

function jobDir(id: string): string { return path.join(midiBaseDir, id); }
function midPath(id: string): string { return path.join(jobDir(id), 'out.mid'); }
/** ace-midi's note events, kept so the MIDI can be re-exported on a new grid */
function eventsPath(id: string): string { return path.join(jobDir(id), 'events.jsonl'); }
/** Tracked beats + the grid the MIDI was written on (ace-midi --beats-out) */
function beatsPath(id: string): string { return path.join(jobDir(id), 'beats.json'); }

function broadcast(job: MidiJob, ev: any): void {
  const line = `data: ${JSON.stringify(ev)}\n\n`;
  for (const res of job.listeners) {
    try { res.write(line); } catch { /* client gone; cleanup on close */ }
  }
}

function pushEvent(job: MidiJob, ev: any): void {
  job.events.push(ev);
  if (ev.type === 'progress') {
    job.chunksDone = ev.completed ?? job.chunksDone;
    job.chunksTotal = ev.total ?? job.chunksTotal;
  } else if (ev.type === 'note_start') {
    job.noteCount++;
  }
  broadcast(job, ev);
}

function endStream(job: MidiJob): void {
  broadcast(job, { type: 'status', status: job.status, error: job.error, noteCount: job.noteCount });
  for (const res of job.listeners) {
    try { res.end(); } catch { /* ignore */ }
  }
  job.listeners.clear();
}

async function runTranscription(job: MidiJob): Promise<void> {
  if ((job.status as string) === 'cancelled') return;
  const dir = jobDir(job.id);
  fs.mkdirSync(dir, { recursive: true });

  try {
    const exe = aceMidiExe();
    if (!exe) throw new Error('ace-midi engine binary not found — rebuild the engine or reinstall');
    if (!isModelDownloaded(job.model)) throw new Error(`Model '${job.model}' is not downloaded`);
    const srcPath = resolveAudioPath(job.sourceAudioUrl);
    if (!fs.existsSync(srcPath)) throw new Error(`Source audio not found: ${srcPath}`);

    job.status = 'transcribing';
    // ace-midi decodes only WAV (16-bit) and MP3; FLAC and the rest go through ffmpeg
    const enginePath = engineFormatPath(srcPath, dir);
    console.log(`[MidiStudio] Job ${job.id}: ace-midi ${path.basename(enginePath)} (model=${job.model})`);

    // With the beat model the MIDI gets tempo, time signature and bar lines
    const beatArgs = isModelDownloaded(BEAT_MODEL)
      ? ['--beat-model', modelDir(BEAT_MODEL), '--beats-out', beatsPath(job.id)]
      : [];
    const child = spawn(exe, [
      '--model', modelDir(job.model),
      '--transcribe', enginePath,
      '--out', midPath(job.id),
      '--jsonl',
      ...beatArgs,
      ...speculativeArgs(job.model, job.speculative, isModelDownloaded('small')),
    ], { windowsHide: true });
    job.child = child;

    const stderrTail: string[] = [];
    child.stderr?.on('data', (buf: Buffer) => {
      for (const raw of buf.toString('utf-8').split(/[\r\n]+/)) {
        const line = raw.trim();
        if (!line) continue;
        stderrTail.push(line);
        if (stderrTail.length > 30) stderrTail.shift();
      }
    });

    const killer = startStallTimer((why) => {
      console.error(`[MidiStudio] Job ${job.id}: ${why === 'idle' ? 'no progress' : 'timed out'} — killing ace-midi`);
      child.kill();
    }, TRANSCRIBE_IDLE_TIMEOUT_MS, TRANSCRIBE_MAX_MS);

    const rl = readline.createInterface({ input: child.stdout! });
    rl.on('line', (line) => {
      killer.touch();
      try { pushEvent(job, JSON.parse(line)); } catch { /* non-JSON noise */ }
    });

    const code: number | null = await new Promise((resolve, reject) => {
      child.on('error', (err) => reject(new Error(`Failed to launch ace-midi: ${err.message}`)));
      child.on('close', (c, signal) => resolve(signal ? null : c));
    }).finally(() => {
      killer.clear();
      job.child = undefined;
    }) as number | null;

    if ((job.status as string) === 'cancelled') return;
    if (code !== 0) {
      throw new Error(`ace-midi exited with code ${code}: ${stderrTail.slice(-30).join(' | ')}`);
    }
    if (!fs.existsSync(midPath(job.id))) throw new Error('ace-midi finished but produced no MIDI file');

    addLyrics(job.id, srcPath);
    const { noteCount, durationSec } = saveResult(job);
    job.status = 'done';
    console.log(`[MidiStudio] Job ${job.id}: done (${noteCount} notes, ${durationSec.toFixed(1)}s)`);
  } catch (err: any) {
    if ((job.status as string) !== 'cancelled') {
      job.status = 'failed';
      job.error = err.message || 'Unknown error';
      job.gated = looksLikeGatedError(job.error || '');
      console.error(`[MidiStudio] Job ${job.id}: FAILED — ${job.error}`);
      savePartial(job);
    }
  } finally {
    endStream(job);
  }
}

/** Embed lyrics from an .lrc next to the source audio into out.mid, if any. */
function addLyrics(jobId: string, srcPath: string): void {
  const lrcPath = path.join(path.dirname(srcPath), `${path.parse(srcPath).name}.lrc`);
  if (!fs.existsSync(lrcPath)) return;
  const midiPath = midPath(jobId);
  const tempPath = path.join(jobDir(jobId), 'out.lyrics.tmp');
  try {
    const original = fs.readFileSync(midiPath);
    const withLyrics = addMidiLyrics(original, fs.readFileSync(lrcPath, 'utf8'));
    if (withLyrics !== original) {
      fs.writeFileSync(tempPath, withLyrics);
      fs.renameSync(tempPath, midiPath);
    }
  } catch (err: any) {
    console.warn(`[MidiStudio] Job ${jobId}: MIDI lyric post-pass skipped (${err.message})`);
  } finally {
    try {
      if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
    } catch (err: any) {
      console.warn(`[MidiStudio] Job ${jobId}: MIDI lyric temp cleanup failed (${err.message})`);
    }
  }
}

/** Parse out.mid into notes.json for the piano roll. A bar-aligned MIDI is
 *  moved back onto the audio's clock and carries its grid. */
function writeNotesJson(jobId: string): { noteCount: number; durationSec: number } {
  let parsed: any = parseMidiFile(fs.readFileSync(midPath(jobId)));
  const grid = fs.existsSync(beatsPath(jobId)) ? readGrid(fs.readFileSync(beatsPath(jobId), 'utf-8')) : null;
  if (grid) parsed = alignToAudio(parsed, grid);
  fs.writeFileSync(path.join(jobDir(jobId), 'notes.json'), JSON.stringify(parsed));
  return { noteCount: parsed.noteCount, durationSec: parsed.durationSec };
}

/** A failed run keeps the notes it streamed before it died (#227). */
function savePartial(job: MidiJob): void {
  try {
    if (!fs.existsSync(midPath(job.id))) {
      const mid = partialMidiFromEvents(job.events);
      if (!mid) return;
      fs.mkdirSync(jobDir(job.id), { recursive: true });
      fs.writeFileSync(midPath(job.id), mid);
    }
    const { noteCount } = saveResult(job, true);
    job.partial = true;
    console.log(`[MidiStudio] Job ${job.id}: kept partial MIDI (${noteCount} notes, ${job.chunksDone}/${job.chunksTotal} chunks)`);
  } catch (err: any) {
    console.warn(`[MidiStudio] Job ${job.id}: could not keep partial MIDI (${err.message})`);
  }
}

/** Write notes.json and _meta.json for out.mid; returns what the preview parsed. */
function saveResult(job: MidiJob, partial = false): { noteCount: number; durationSec: number } {
  const dir = jobDir(job.id);
  // Parse for the piano-roll preview (preview failure is non-fatal)
  let noteCount = job.noteCount;
  let durationSec = 0;
  try {
    fs.writeFileSync(eventsPath(job.id), job.events
      .filter(e => e?.type === 'note_start' || e?.type === 'note_end')
      .map(e => JSON.stringify(e)).join('\n') + '\n');
  } catch (err: any) {
    console.warn(`[MidiStudio] Job ${job.id}: could not keep note events (${err.message})`);
  }
  try {
    ({ noteCount, durationSec } = writeNotesJson(job.id));
  } catch (err: any) {
    console.warn(`[MidiStudio] Job ${job.id}: MIDI parse for preview failed (${err.message})`);
  }

  fs.writeFileSync(path.join(dir, '_meta.json'), JSON.stringify({
    id: job.id,
    sourceAudioUrl: job.sourceAudioUrl,
    sourceFileName: job.sourceFileName,
    songId: job.songId,
    model: job.model,
    noteCount,
    durationSec,
    createdAt: new Date(job.createdAt).toISOString(),
    ...(partial ? { partial: true, error: job.error, chunksDone: job.chunksDone, chunksTotal: job.chunksTotal } : {}),
  }, null, 2));
  return { noteCount, durationSec };
}

// ── Routes ───────────────────────────────────────────────────────────────

/** GET /status — engine, models, token */
router.get('/status', (_req: Request, res: Response) => {
  try {
    res.json({
      engineAvailable: aceMidiExe() !== null,
      hfTokenSet: getHfToken() !== null,
      models: getModelStates(m => modelDownloadService.muscriptorState(m)),
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

/** POST /hf-token — store (or clear, with empty string) the HF read token */
router.post('/hf-token', (req: Request, res: Response) => {
  try {
    const token = typeof req.body?.token === 'string' ? req.body.token : '';
    setHfToken(token);
    console.log(`[MidiStudio] HF token ${token.trim() ? 'saved' : 'cleared'}`);
    res.json({ ok: true, hfTokenSet: !!token.trim() });
  } catch (err: any) {
    res.status(err.status || 500).json({ error: err.message });
  }
});

/** POST /models/:size/download — begin downloading gated weights (or the beat model) */
router.post('/models/:size/download', (req: Request, res: Response) => {
  const size = req.params.size as MuscriptorModel | typeof BEAT_MODEL;
  if (size !== BEAT_MODEL && !MUSCRIPTOR_MODELS.includes(size)) {
    res.status(400).json({ error: `Unknown model '${size}'` });
    return;
  }
  const r = modelDownloadService.startMuscriptorDownload(size);
  if (!r.started) {
    res.status(409).json({ error: r.error });
    return;
  }
  res.json({ ok: true });
});

/** POST /transcribe — queue a transcription job */
router.post('/transcribe', (req: Request, res: Response) => {
  const { sourceAudioUrl, sourceFileName, songId, model, speculative } = req.body || {};
  if (!sourceAudioUrl || typeof sourceAudioUrl !== 'string') {
    res.status(400).json({ error: 'sourceAudioUrl is required' });
    return;
  }
  const m: MuscriptorModel = MUSCRIPTOR_MODELS.includes(model) ? model : 'small';
  if (!aceMidiExe()) {
    res.status(503).json({ error: 'ace-midi engine binary not found' });
    return;
  }
  if (!isModelDownloaded(m)) {
    res.status(409).json({ error: `Model '${m}' is not downloaded — download it first` });
    return;
  }

  const job: MidiJob = {
    id: randomUUID(),
    status: 'queued',
    sourceAudioUrl,
    sourceFileName: sourceFileName || path.basename(sourceAudioUrl),
    songId: songId || undefined,
    model: m,
    speculative: speculative !== false,
    events: [],
    chunksDone: 0,
    chunksTotal: 0,
    noteCount: 0,
    createdAt: Date.now(),
    listeners: new Set(),
  };
  jobs.set(job.id, job);
  // Tracked from enqueue: a queued job deleted from the map still runs later.
  void trackBusy('MIDI', runOnMidiLane(job.id, () => runTranscription(job))).catch((err: any) => {
    // Only a queue reset rejects here; runTranscription settles its own errors.
    if (job.status !== 'queued') return;
    job.status = 'cancelled';
    job.error = err.message || 'Queue reset';
    endStream(job);
  });

  console.log(`[MidiStudio] Job ${job.id} queued: ${job.sourceFileName} (model=${m})`);
  res.json({ id: job.id });
});

/** GET /jobs — completed jobs from disk + in-flight jobs */
router.get('/jobs', (_req: Request, res: Response) => {
  try {
    const summaries: any[] = [];
    const onDisk = new Set<string>();
    if (fs.existsSync(midiBaseDir)) {
      for (const entry of fs.readdirSync(midiBaseDir, { withFileTypes: true })) {
        if (!entry.isDirectory()) continue;
        const metaPath = path.join(midiBaseDir, entry.name, '_meta.json');
        if (!fs.existsSync(metaPath)) continue;
        try {
          const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
          onDisk.add(meta.id || entry.name);
          summaries.push({
            id: meta.id || entry.name,
            // A kept partial stays a failure, before and after a restart.
            status: meta.partial ? 'failed' : 'done',
            sourceFileName: meta.sourceFileName || 'unknown',
            sourceAudioUrl: meta.sourceAudioUrl,
            songId: meta.songId,
            model: meta.model || 'small',
            noteCount: meta.noteCount || 0,
            durationSec: meta.durationSec || 0,
            createdAt: meta.createdAt || '',
            exportOptions: meta.export,
            ...(meta.partial ? {
              partial: true, error: meta.error, chunksDone: meta.chunksDone, chunksTotal: meta.chunksTotal,
            } : {}),
          });
        } catch { /* skip corrupted meta */ }
      }
    }
    for (const [, job] of jobs) {
      if (job.status === 'done' || onDisk.has(job.id)) continue;
      summaries.push({
        id: job.id,
        status: job.status,
        sourceFileName: job.sourceFileName,
        sourceAudioUrl: job.sourceAudioUrl,
        songId: job.songId,
        model: job.model,
        noteCount: job.noteCount,
        durationSec: 0,
        chunksDone: job.chunksDone,
        chunksTotal: job.chunksTotal,
        createdAt: new Date(job.createdAt).toISOString(),
        error: job.error,
        gated: job.gated,
        partial: job.partial,
      });
    }
    summaries.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
    res.json(summaries);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

/** GET /:jobId/progress — poll job progress */
router.get('/:jobId/progress', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const job = jobs.get(jobId);
  if (!job) {
    const metaPath = path.join(jobDir(jobId), '_meta.json');
    if (fs.existsSync(metaPath)) {
      let meta: any = {};
      try { meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8')); } catch { /* treat as done */ }
      res.json(meta.partial ? { status: 'failed', partial: true, error: meta.error } : { status: 'done' });
      return;
    }
    res.status(404).json({ error: 'Job not found' });
    return;
  }
  res.json({
    status: job.status,
    chunksDone: job.chunksDone,
    chunksTotal: job.chunksTotal,
    noteCount: job.noteCount,
    error: job.error,
    gated: job.gated,
    partial: job.partial,
  });
});

/** GET /:jobId/stream — SSE: replay buffered events then tail live ones */
router.get('/:jobId/stream', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const job = jobs.get(jobId);
  if (!job) {
    res.status(404).json({ error: 'Job not found or already finished (use /notes)' });
    return;
  }
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  for (const ev of job.events) {
    res.write(`data: ${JSON.stringify(ev)}\n\n`);
  }
  if (job.status === 'done' || job.status === 'failed' || job.status === 'cancelled') {
    res.write(`data: ${JSON.stringify({ type: 'status', status: job.status, error: job.error, noteCount: job.noteCount })}\n\n`);
    res.end();
    return;
  }
  job.listeners.add(res);
  req.on('close', () => job.listeners.delete(res));
});

/** GET /:jobId/notes — parsed note data for the piano roll */
router.get('/:jobId/notes', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const notesPath = path.join(jobDir(jobId), 'notes.json');
  try {
    if (fs.existsSync(notesPath)) {
      res.setHeader('Content-Type', 'application/json');
      fs.createReadStream(notesPath).pipe(res);
      return;
    }
    if (fs.existsSync(midPath(jobId))) {
      writeNotesJson(jobId);
      res.setHeader('Content-Type', 'application/json');
      fs.createReadStream(notesPath).pipe(res);
      return;
    }
    res.status(404).json({ error: 'No MIDI data for this job' });
  } catch (err: any) {
    res.status(500).json({ error: `MIDI parse failed: ${err.message}` });
  }
});

/** POST /:jobId/export — rewrite out.mid on a corrected grid (tempo, meter,
 *  downbeat, beat scale) and/or quantized, from the kept note events and
 *  tracked beats. Grid maths only: no model, no GPU. */
router.post('/:jobId/export', async (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const live = jobs.get(jobId);
  if (live && (live.status === 'queued' || live.status === 'transcribing')) {
    res.status(409).json({ error: 'Still transcribing' });
    return;
  }
  if (!fs.existsSync(eventsPath(jobId)) || !fs.existsSync(beatsPath(jobId))) {
    res.status(409).json({ error: 'This transcription has no beat grid. Download the tempo model and transcribe again.' });
    return;
  }
  const exe = aceMidiExe();
  if (!exe) {
    res.status(503).json({ error: 'ace-midi engine binary not found' });
    return;
  }
  const opts: MidiExportOptions = req.body || {};
  const dir = jobDir(jobId);
  const tmpMid = path.join(dir, 'out.export.tmp');
  const tmpBeats = path.join(dir, 'beats.export.tmp');
  try {
    const stderr: string[] = [];
    const code: number | null = await new Promise((resolve, reject) => {
      const child = spawn(exe, [
        '--export', eventsPath(jobId), '--beats-in', beatsPath(jobId),
        '--out', tmpMid, '--beats-out', tmpBeats, ...exportArgs(opts),
      ], { windowsHide: true });
      child.stderr?.on('data', (b: Buffer) => stderr.push(b.toString('utf-8')));
      child.on('error', reject);
      child.on('close', c => resolve(c));
    });
    if (code !== 0 || !fs.existsSync(tmpMid)) {
      throw new Error(`ace-midi export failed (${code}): ${stderr.join('').trim().split(/\r?\n/).slice(-3).join(' | ')}`);
    }
    fs.renameSync(tmpMid, midPath(jobId));
    fs.renameSync(tmpBeats, beatsPath(jobId));

    const metaPath = path.join(dir, '_meta.json');
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
    if (meta.sourceAudioUrl) addLyrics(jobId, resolveAudioPath(meta.sourceAudioUrl));
    const { noteCount, durationSec } = writeNotesJson(jobId);
    fs.writeFileSync(metaPath, JSON.stringify({ ...meta, noteCount, durationSec, export: opts }, null, 2));
    console.log(`[MidiStudio] Job ${jobId}: re-exported (${exportArgs(opts).join(' ') || 'detected grid'})`);
    res.json({ ok: true, grid: readGrid(fs.readFileSync(beatsPath(jobId), 'utf-8')), noteCount });
  } catch (err: any) {
    for (const p of [tmpMid, tmpBeats]) { try { fs.rmSync(p, { force: true }); } catch { /* ignore */ } }
    console.error(`[MidiStudio] Job ${jobId}: export FAILED — ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

/** POST /:jobId/abc — the transcription as a YuE2 two-voice ABC lead sheet
 *  (voice -> Vocal, one instrument -> Ins, estimated key and chords) on the
 *  job's current grid. Body: { ins?: GM program, melodyOnly?: boolean }. */
router.post('/:jobId/abc', async (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  if (!fs.existsSync(eventsPath(jobId)) || !fs.existsSync(beatsPath(jobId))) {
    res.status(409).json({ error: 'This transcription has no beat grid. Download the tempo model and transcribe again.' });
    return;
  }
  const exe = aceMidiExe();
  if (!exe) {
    res.status(503).json({ error: 'ace-midi engine binary not found' });
    return;
  }
  const dir = jobDir(jobId);
  let opts: MidiExportOptions = {};
  try { opts = JSON.parse(fs.readFileSync(path.join(dir, '_meta.json'), 'utf-8')).export || {}; } catch { /* detected grid */ }
  const ins = Number(req.body?.ins);
  try {
    res.json(await exportMidiAbc(dir, opts, ins, !!req.body?.melodyOnly));
  } catch (err: any) {
    console.error(`[MidiStudio] Job ${jobId}: ABC FAILED: ${err.message}`);
    res.status(err instanceof MidiAbcError ? err.status : 500).json({ error: err.message,
      ...(err instanceof MidiAbcError && err.summary ? { summary: err.summary } : {}) });
  }
});

/** GET /:jobId/file — download the .mid */
router.get('/:jobId/file', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const p = midPath(jobId);
  if (!fs.existsSync(p)) {
    res.status(404).json({ error: 'MIDI file not found' });
    return;
  }
  let base = 'transcription';
  try {
    const meta = JSON.parse(fs.readFileSync(path.join(jobDir(jobId), '_meta.json'), 'utf-8'));
    base = (meta.sourceFileName || base).replace(/\.[^.]+$/, '').replace(/[^\w\s.-]/g, '_');
  } catch { /* keep default */ }
  res.download(p, `${base}.mid`);
});

/** DELETE /:jobId — cancel a running job and/or delete its files */
router.delete('/:jobId', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const job = jobs.get(jobId);
  if (job && (job.status === 'queued' || job.status === 'transcribing')) {
    job.status = 'cancelled';
    job.child?.kill();
    endStream(job);
    console.log(`[MidiStudio] Job ${jobId} cancelled`);
  }
  jobs.delete(jobId);

  const dir = jobDir(jobId);
  if (fs.existsSync(dir)) {
    fs.rmSync(dir, { recursive: true, force: true });
    console.log(`[MidiStudio] Deleted job ${jobId}`);
  }
  res.json({ ok: true });
});

export default router;

/** Cover Prepare reuses MIDI Studio's lane, cancellation and Beat This! export. */
export async function transcribeCoverMidi(url: string, signal: AbortSignal) {
  if (signal.aborted) throw new Error('Cancelled');
  const model = coverMidiModel(yue2CoverService.readiness('muscriptor'));
  const job: MidiJob = { id: randomUUID(), status: 'queued', sourceAudioUrl: url,
    sourceFileName: path.basename(url), model, speculative: true, events: [], chunksDone: 0,
    chunksTotal: 0, noteCount: 0, createdAt: Date.now(), listeners: new Set() };
  jobs.set(job.id, job);
  const cancel = () => { job.status = 'cancelled'; job.child?.kill(); };
  signal.addEventListener('abort', cancel, { once: true });
  try {
    await trackBusy('MIDI', runOnMidiLane(job.id, () => runTranscription(job)));
    if (signal.aborted) throw new Error('Cancelled');
    if (job.status !== 'done') throw new Error(job.error || `Transcription ${job.status}`);
    const result = await exportMidiAbc(jobDir(job.id), {}, undefined, false, signal);
    const sheet = tryAbcCoverLeadSheet(result.abc, 'muscriptor');
    if (sheet) sheet.timing.grid = { identity: job.id, data: JSON.parse(fs.readFileSync(beatsPath(job.id), 'utf8')) };
    return { abc: result.abc, scoreSource: 'muscriptor' as const, leadSheet: sheet };
  } finally { signal.removeEventListener('abort', cancel); }
}
