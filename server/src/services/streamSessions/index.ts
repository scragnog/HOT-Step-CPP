// streamSessions — Node's record of the engine audio streams it proxies.
//
// The STORM and MM3 stream routes (routes/generate.ts) hand every whole WAV
// they forward to a session here, after it has been written to the client.
// A session analyses each chunk (BPM/key, analysis.ts) and, between an
// explicit start and stop, appends the chunk's samples byte-for-byte to a
// recording on disk. The export is that canonical concatenation: the engine's
// own samples, no client mix, crossfade or device in the path. MM3 windows
// are consecutive spans of one signal, so their concatenation is the song as
// rendered; STORM slots are independent renders joined end to end.
//
// Nothing here may disturb playback: a session that cannot be opened (cap
// reached) or a chunk that cannot be parsed only loses the Node-side copy.
// MM3 allows one engine reader per job, so Node never opens its own stream;
// it records what a client is already playing.
//
// Lifetime and caps: at most MAX_SESSIONS sessions (the oldest ended one is
// evicted first; when all are live a new stream simply gets no session); a
// recording stops itself at MAX_RECORDING_BYTES ('capped'); an ended session
// and its file are deleted ENDED_TTL_MS after the stream ends; a live session
// with no chunk for LIVE_IDLE_MS is treated as ended; at most
// MAX_CONCURRENT_EXPORTS exports run at once (429 beyond).
import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { pipeline } from 'stream/promises';
import type { StreamChunkAnalysis, StreamExportFormat, StreamSessionSummary } from '../../contracts/streamSessions.js';
import { config, getFFmpegPath } from '../../config.js';
import { detectBpm, detectKey } from './analysis.js';
import { channel0, parseWav, sameFormat, WavSplitter, wavHeader, type WavFormat } from './wav.js';

export const LIMITS = {
  maxSessions: 16,
  maxRecordingBytes: 2 * 1024 ** 3,
  endedTtlMs: 30 * 60_000,
  liveIdleMs: 2 * 60 * 60_000,
  keptChunkAnalyses: 64,
  /** Exports being built or downloaded at once; each holds temp files and maybe an ffmpeg. */
  maxConcurrentExports: 2,
};

export class StreamSessionError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

interface Recording {
  status: StreamSessionSummary['recording']['status'];
  /** Its own file, so a discarded recording's late delete never touches the next one. */
  file: string;
  format: WavFormat | null;
  fromChunk: number | null;
  toChunk: number | null;
  bytes: number;
  error: string | null;
  /** Serialises appends so chunks land in order. */
  writes: Promise<void>;
}

export class StreamSession {
  readonly id = randomUUID();
  readonly createdAt: number;
  endedAt: number | null = null;
  lastChunkAt: number;
  chunkCount = 0;
  private analyses: StreamChunkAnalysis[] = [];
  private recording: Recording = StreamSession.idle();

  constructor(private readonly owner: StreamSessions, readonly kind: 'storm' | 'mm3',
    readonly source: StreamSessionSummary['source']) {
    this.createdAt = this.lastChunkAt = owner.now();
  }

  private static idle(file = ''): Recording {
    return { status: 'idle', file, format: null, fromChunk: null, toChunk: null, bytes: 0, error: null, writes: Promise.resolve() };
  }
  get live(): boolean { return this.endedAt === null; }

  /** Raw bytes of a stream made of concatenated WAVs (MM3). Never throws. */
  feed(bytes: Uint8Array): void {
    if (!this.live) return;
    try { for (const wav of this.splitter.push(bytes)) this.chunk(wav); } catch (err) {
      this.fail(`Stream framing failed: ${(err as Error).message}`);
    }
  }
  private readonly splitter = new WavSplitter();

  /** One whole WAV as it went to the client. Never throws. */
  chunk(bytes: Uint8Array): void {
    try { this.take(bytes); } catch (err) { this.fail(`Chunk handling failed: ${(err as Error).message}`); }
  }

  private take(bytes: Uint8Array): void {
    if (!this.live) return;
    const index = this.chunkCount++;
    this.lastChunkAt = this.owner.now();
    let wav;
    try { wav = parseWav(bytes); } catch (err) {
      this.fail(`Chunk ${index} unreadable: ${(err as Error).message}`);
      return;
    }
    const samples = channel0(wav);
    const { bpm, firstBeat } = detectBpm(samples, wav.sampleRate);
    this.analyses.push({ index, sampleRate: wav.sampleRate, channels: wav.channels, frames: wav.frames,
      bpm, key: detectKey(samples, wav.sampleRate), firstBeat });
    if (this.analyses.length > LIMITS.keptChunkAnalyses) this.analyses.shift();

    const r = this.recording;
    if (r.status !== 'recording') return;
    if (r.format && !sameFormat(r.format, wav)) { this.fail(`Chunk ${index} changed the audio format; recording stopped`); return; }
    if (r.bytes + wav.data.length > this.owner.limits.maxRecordingBytes) { r.status = 'capped'; return; }
    r.format ??= { audioFormat: wav.audioFormat, channels: wav.channels, sampleRate: wav.sampleRate, bitsPerSample: wav.bitsPerSample };
    r.fromChunk ??= index;
    r.toChunk = index;
    r.bytes += wav.data.length;
    const data = Buffer.from(wav.data);
    // A write can fail after stop, end or the cap: it still loses bytes the
    // header would claim, so it fails this recording whatever its status.
    r.writes = r.writes.then(() => fs.promises.appendFile(r.file, data)).catch(err => {
      StreamSession.failRecording(r, `Recording write failed: ${(err as Error).message}`);
    });
  }

  end(): void {
    if (!this.live) return;
    this.endedAt = this.owner.now();
    if (this.recording.status === 'recording') this.recording.status = 'stopped';
  }

  record(action: 'start' | 'stop' | 'discard'): void {
    const r = this.recording;
    if (action === 'start') {
      if (!this.live) throw new StreamSessionError(409, 'The stream has ended');
      if (r.status === 'recording') return;
      if (r.bytes > 0) throw new StreamSessionError(409, 'Export or discard the current recording first');
      fs.mkdirSync(this.owner.dir, { recursive: true });
      this.recording = { ...StreamSession.idle(path.join(this.owner.dir, `${this.id}-${randomUUID()}.pcm`)), status: 'recording' };
      return;
    }
    if (action === 'stop') { if (r.status === 'recording') r.status = 'stopped'; return; }
    this.recording = StreamSession.idle();
    if (r.file) void r.writes.then(() => fs.promises.rm(r.file, { force: true }));
  }

  private fail(message: string): void {
    if (this.recording.status === 'recording') StreamSession.failRecording(this.recording, message);
  }

  private static failRecording(r: Recording, message: string): void {
    if (r.status === 'failed') return;
    r.status = 'failed';
    r.error = message;
  }

  /** A finished recording as WAV/FLAC/MP3/Opus in a temporary file the caller deletes. */
  async exportTo(format: StreamExportFormat, bitrate?: number): Promise<{ file: string; cleanup: () => void }> {
    const r = this.recording;
    if (r.status === 'recording') throw new StreamSessionError(409, 'Stop the recording before exporting it');
    if (!r.format || r.bytes === 0) throw new StreamSessionError(409, 'Nothing has been recorded');
    await r.writes;
    if (r.status === 'failed') throw new StreamSessionError(409, r.error ?? 'Recording failed');
    const release = this.owner.admitExport();
    const stem = path.join(this.owner.dir, `${this.id}-${randomUUID()}`);
    const wavFile = `${stem}.wav`;
    const cleanups = [wavFile];
    let cleaned = false;
    const cleanup = () => {
      if (cleaned) return;
      cleaned = true;
      for (const f of cleanups) fs.rmSync(f, { force: true });
      release();
    };
    try {
      fs.writeFileSync(wavFile, wavHeader(r.format, r.bytes));
      // Streamed, so a 2 GiB recording never sits in memory.
      await pipeline(fs.createReadStream(r.file, { start: 0, end: r.bytes - 1 }), fs.createWriteStream(wavFile, { flags: 'a' }));
      if (format === 'wav') return { file: wavFile, cleanup };
      const out = `${stem}.${format}`;
      cleanups.push(out);
      await this.owner.transcode(wavFile, format, bitrate, out);
      return { file: out, cleanup };
    } catch (err) { cleanup(); throw err; }
  }

  summary(): StreamSessionSummary {
    const r = this.recording;
    const bytesPerFrame = r.format ? r.format.channels * r.format.bitsPerSample / 8 : 0;
    return {
      id: this.id, kind: this.kind, source: this.source, status: this.live ? 'live' : 'ended',
      createdAt: this.createdAt, endedAt: this.endedAt,
      expiresAt: this.endedAt === null ? null : this.endedAt + this.owner.limits.endedTtlMs,
      chunkCount: this.chunkCount, chunks: [...this.analyses],
      recording: { status: r.status, fromChunk: r.fromChunk, toChunk: r.toChunk,
        frames: bytesPerFrame ? r.bytes / bytesPerFrame : 0, bytes: r.bytes,
        sampleRate: r.format?.sampleRate ?? null, channels: r.format?.channels ?? null, error: r.error },
      limits: { maxRecordingBytes: this.owner.limits.maxRecordingBytes, endedTtlMs: this.owner.limits.endedTtlMs,
        maxSessions: this.owner.limits.maxSessions },
    };
  }

  /** Delete the recording file (eviction/expiry). */
  dispose(): void {
    const r = this.recording;
    if (r.file) void r.writes.then(() => fs.promises.rm(r.file, { force: true }));
  }
}

export type Transcoder = (wavFile: string, format: Exclude<StreamExportFormat, 'wav'>, bitrate: number | undefined, out: string) => Promise<void>;

export class StreamSessions {
  private readonly sessions = new Map<string, StreamSession>();
  private exportsInFlight = 0;
  constructor(readonly dir: string, readonly transcode: Transcoder,
    readonly limits = LIMITS, readonly now: () => number = Date.now) {}

  /** A new session for a stream that is starting, or null at the cap (the stream plays on without one). */
  open(kind: 'storm' | 'mm3', source: StreamSessionSummary['source']): StreamSession | null {
    this.sweep();
    if (this.sessions.size >= this.limits.maxSessions) {
      const oldestEnded = [...this.sessions.values()].filter(s => !s.live).sort((a, b) => a.endedAt! - b.endedAt!)[0];
      if (!oldestEnded) return null;
      this.remove(oldestEnded);
    }
    const session = new StreamSession(this, kind, source);
    this.sessions.set(session.id, session);
    return session;
  }

  /** Reserve an export slot (429 when full). Call the returned release exactly once. */
  admitExport(): () => void {
    if (this.exportsInFlight >= this.limits.maxConcurrentExports)
      throw new StreamSessionError(429, 'Too many recording exports in progress; try again shortly');
    this.exportsInFlight++;
    return () => { this.exportsInFlight--; };
  }

  get(id: string): StreamSession {
    this.sweep();
    const session = this.sessions.get(id);
    if (!session) throw new StreamSessionError(404, 'Stream session not found or expired');
    return session;
  }

  list(): StreamSessionSummary[] {
    this.sweep();
    return [...this.sessions.values()].sort((a, b) => b.createdAt - a.createdAt).map(s => s.summary());
  }

  /** End idle live sessions and delete expired ended ones. */
  sweep(): void {
    const now = this.now();
    for (const s of [...this.sessions.values()]) {
      if (s.live && now - s.lastChunkAt > this.limits.liveIdleMs) s.end();
      if (!s.live && now - s.endedAt! > this.limits.endedTtlMs) this.remove(s);
    }
  }

  private remove(session: StreamSession): void {
    this.sessions.delete(session.id);
    session.dispose();
  }
}

const execFileAsync = promisify(execFile);

/** ffmpeg with the download route's codec settings (routes/download.ts). */
// ponytail: duplicates download.ts's private codec switch; fold into 8b's shared export helper when it lands.
export async function ffmpegTranscode(wavFile: string, format: Exclude<StreamExportFormat, 'wav'>, bitrate: number | undefined, out: string): Promise<void> {
  const ffmpeg = getFFmpegPath();
  if (!ffmpeg) throw new StreamSessionError(501, `Cannot export ${format}: ffmpeg is not available`);
  const codec = format === 'flac' ? ['-c:a', 'flac', '-sample_fmt', 's32', '-compression_level', '8']
    : format === 'opus' ? ['-c:a', 'libopus', '-b:a', `${bitrate ?? 160}k`]
      : ['-c:a', 'libmp3lame', '-b:a', `${bitrate ?? 320}k`];
  await execFileAsync(ffmpeg, ['-y', '-i', wavFile, ...codec, out], { timeout: 600_000 });
}

let instance: StreamSessions | undefined;
export function getStreamSessions(): StreamSessions {
  if (!instance) {
    const dir = path.join(config.data.dir, 'stream-sessions');
    // Recordings never outlive the process that owns their sessions. A locked
    // leftover only wastes disk; new files get fresh names.
    try { fs.rmSync(dir, { recursive: true, force: true }); } catch (err) {
      console.warn(`[stream-sessions] could not clear ${dir}: ${(err as Error).message}`);
    }
    instance = new StreamSessions(dir, ffmpegTranscode);
    setInterval(() => instance!.sweep(), 60_000).unref();
  }
  return instance;
}

/** For the stream routes: a session, or null if one cannot be opened for any
 *  reason. Playback must never depend on Node's copy. */
export function openStreamSession(kind: 'storm' | 'mm3', source: StreamSessionSummary['source'],
  store: () => StreamSessions = getStreamSessions): StreamSession | null {
  try { return store().open(kind, source); } catch (err) {
    console.warn(`[stream-sessions] no session for ${kind} stream: ${(err as Error).message}`);
    return null;
  }
}
