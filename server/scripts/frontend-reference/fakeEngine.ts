// fakeEngine.ts — a fake ace-server, standing in for the real engine process.
//
// Implements just enough of ace-server's HTTP wire protocol (/lm, /synth,
// /job, /understand, /props, /health, /supersep/*) for the reference client
// to exercise Create, Insta-Gen, Cover, Repaint, Lego and stem separation
// end to end through the REAL Node routes/services, without ever spawning a
// real engine, downloading a model or calling a remote worker. Node code is
// never faked here — only the engine process it talks to over HTTP is.
//
// Wire protocol verified against server/src/services/aceClient.ts,
// server/src/services/training/understandClient.ts, server/src/routes/
// stemStudio.ts and server/src/routes/supersep.ts (job submit/poll/result
// shapes, multipart field names, SuperSep's separate contract).

import express, { type Express } from 'express';
import multer from 'multer';
import { randomUUID } from 'node:crypto';
import type { Server } from 'node:http';

type AceRequest = Record<string, unknown>;

interface FakeJob {
  kind: 'lm' | 'synth' | 'understand';
  requests: AceRequest[];
  format: 'mp3' | 'wav';
  status: 'running' | 'done' | 'failed' | 'cancelled';
  polls: number;
  createdAt: number;
}

interface SuperSepJob {
  level: number;
  stems: Array<{ name: string; category: string; stem_type: string; n_frames: number; stage: number; index: number; hidden: boolean }>;
  released: boolean;
}

/** A tiny, structurally valid silent WAV — just enough PCM for a
 *  Content-Type/byte-length check, never for a listening test. */
function makeWav(sampleRate: number, channels: number, bitsPerSample: number, frames = 100): Buffer {
  const bytesPerSample = bitsPerSample / 8;
  const dataSize = frames * channels * bytesPerSample;
  const buf = Buffer.alloc(44 + dataSize);
  buf.write('RIFF', 0, 'ascii');
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write('WAVE', 8, 'ascii');
  buf.write('fmt ', 12, 'ascii');
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(channels, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * channels * bytesPerSample, 28);
  buf.writeUInt16LE(channels * bytesPerSample, 32);
  buf.writeUInt16LE(bitsPerSample, 34);
  buf.write('data', 36, 'ascii');
  buf.writeUInt32LE(dataSize, 40);
  return buf; // PCM payload stays zeroed (silence) — content, not shape, is what's faked.
}

const PHASES = ['queued', 'loading_dit', 'dit_inference', 'vae_decode', 'done'] as const;
/** Polls to reach 'done'. Every poll before that returns a different phase
 *  (and a growing phase_step), which is what keeps pollUntilDone.ts's
 *  stall/watchdog logic — which treats a changing phase/step as progress —
 *  from firing, since the fake engine produces none of the stdout log lines
 *  the real stall detector also reads. */
const POLLS_TO_DONE = PHASES.length - 1;

export interface FakeEngineHooks {
  /** Force the next job (any kind) to fail once polled, with this message. */
  failNextJob?: string | null;
}

export interface FakeAceEngine {
  app: Express;
  origin: string;
  hooks: FakeEngineHooks;
  close(): Promise<void>;
}

export async function startFakeAceEngine(): Promise<FakeAceEngine> {
  const hooks: FakeEngineHooks = { failNextJob: null };
  const jobs = new Map<string, FakeJob>();
  const superSepJobs = new Map<string, SuperSepJob>();
  const upload = multer();

  const app = express();
  app.use(express.json({ limit: '50mb' }));

  app.get('/health', (_req, res) => { res.json({ status: 'ok' }); });

  app.get('/props', (_req, res) => {
    res.json({
      models: { lm: ['fake-lm'], embedding: ['fake-embedding'], dit: ['fake-dit'], vae: ['fake-vae'] },
      adapters: [],
      lm_adapters: [],
      cli: { max_batch: 4, mp3_bitrate: 192 },
      default: {},
    });
  });

  app.get('/jobs', (_req, res) => {
    res.json([...jobs.entries()].map(([id, j]) => ({ id, status: j.status, phase: PHASES[Math.min(j.polls, POLLS_TO_DONE)] })));
  });

  app.post('/lm', (req, res) => {
    const body = req.body as AceRequest | AceRequest[];
    const id = randomUUID();
    jobs.set(id, { kind: 'lm', requests: Array.isArray(body) ? body : [body], format: 'mp3', status: 'running', polls: 0, createdAt: Date.now() });
    res.json({ id });
  });

  app.post('/synth', upload.any(), (req, res) => {
    const multipart = req.is('multipart/form-data');
    const raw = multipart ? JSON.parse((req.body as Record<string, string>).request ?? '{}') : req.body;
    const requests: AceRequest[] = Array.isArray(raw) ? raw : [raw];
    const format = String(req.query.format ?? 'mp3') === 'wav' ? 'wav' : 'mp3';
    const id = randomUUID();
    jobs.set(id, { kind: 'synth', requests, format, status: 'running', polls: 0, createdAt: Date.now() });
    res.json({ id });
  });

  app.post('/understand', upload.any(), (req, res) => {
    const files = (req.files as Express.Multer.File[] | undefined) ?? [];
    if (!files.some(f => f.fieldname === 'audio')) {
      res.status(400).send('audio part is required');
      return;
    }
    const requestPart = (req.body as Record<string, string>).request;
    const id = randomUUID();
    jobs.set(id, { kind: 'understand', requests: [requestPart ? JSON.parse(requestPart) : {}], format: 'mp3', status: 'running', polls: 0, createdAt: Date.now() });
    res.json({ id });
  });

  app.get('/job', (req, res) => {
    const id = String(req.query.id ?? '');
    const job = jobs.get(id);
    if (!job) { res.status(404).send('unknown job'); return; }

    if (req.query.cancel !== undefined) {
      job.status = 'cancelled';
      res.json({ ok: true });
      return;
    }

    if (req.query.latent !== undefined) {
      // No latent captured — legal and the simplest faithful response.
      res.status(204).end();
      return;
    }

    if (req.query.result !== undefined) {
      if (job.status === 'cancelled') { res.status(409).send('job was cancelled'); return; }
      if (job.status === 'failed') { res.status(500).send('fake engine: forced failure'); return; }
      if (job.kind === 'lm') { res.json(job.requests); return; }
      if (job.kind === 'understand') {
        const r = job.requests[0] ?? {};
        res.json({
          caption: r.caption ?? 'a fake understood caption',
          lyrics: r.lyrics ?? '',
          bpm: r.bpm ?? 120,
          keyscale: r.keyscale ?? 'C major',
          timesignature: r.timesignature ?? '4',
          vocal_language: r.vocal_language ?? 'en',
          duration: r.duration ?? 12,
          seed: r.seed ?? 1,
        });
        return;
      }
      // synth: binary audio, extension/content-type driven by the submit's format.
      const wav = makeWav(44100, 2, 16, 4410);
      res.setHeader('Content-Type', job.format === 'wav' ? 'audio/wav' : 'audio/mpeg');
      res.send(job.format === 'wav' ? wav : wav); // fake mp3 bytes are just the WAV buffer — only Content-Type is read for extension choice
      return;
    }

    // Plain status poll: advance one step per call, deterministically.
    if (hooks.failNextJob && job.polls === 0) {
      job.status = 'failed';
      hooks.failNextJob = null;
      res.json({ status: 'failed', phase: 'failed', phase_step: 0, phase_total: 0, end_reason: 'fake engine: forced failure' });
      return;
    }
    job.polls = Math.min(job.polls + 1, POLLS_TO_DONE);
    if (job.polls >= POLLS_TO_DONE) job.status = 'done';
    res.json({
      status: job.status,
      phase: PHASES[job.polls],
      phase_step: job.polls,
      phase_total: POLLS_TO_DONE,
      adapter_progress: -1,
      songs_done: job.status === 'done' ? job.requests.length : 0,
    });
  });

  // ── SuperSep (ONNX neural separation) — a different, non-/job protocol ──
  app.post('/supersep/separate', express.raw({ type: 'application/octet-stream', limit: '50mb' }), (req, res) => {
    const level = parseInt(String(req.query.level ?? '0'), 10);
    const id = randomUUID();
    const names = level >= 2 ? ['vocals', 'drums', 'bass', 'other', 'guitar', 'piano'] : ['vocals', 'instrumental'];
    superSepJobs.set(id, {
      level,
      released: false,
      stems: names.map((name, index) => ({ name, category: 'stem', stem_type: name, n_frames: 44100, stage: level, index, hidden: false })),
    });
    res.json({ id });
  });

  app.get('/supersep/progress', (req, res) => {
    const id = String(req.query.id ?? '');
    const job = superSepJobs.get(id);
    if (!job) { res.status(404).json({ status: 'not_found', error: 'unknown job' }); return; }
    res.json({ status: 'done', progress: 100, message: 'done', n_stems: job.stems.length });
  });

  app.get('/supersep/result', (req, res) => {
    const id = String(req.query.id ?? '');
    const job = superSepJobs.get(id);
    if (!job) { res.status(404).json({ error: 'unknown job' }); return; }
    res.json({ id, stems: job.stems });
  });

  app.get('/supersep/serve', (req, res) => {
    const id = String(req.query.id ?? '');
    const job = superSepJobs.get(id);
    if (!job) { res.status(404).send('unknown job'); return; }
    res.setHeader('Content-Type', 'audio/wav');
    res.send(makeWav(44100, 1, 16, 4410));
  });

  app.post('/supersep/recombine', (req, res) => {
    const id = String((req.body as { id?: string })?.id ?? '');
    if (!superSepJobs.has(id)) { res.status(404).send('unknown job'); return; }
    res.setHeader('Content-Type', 'audio/wav');
    res.send(makeWav(48000, 2, 16, 4800));
  });

  app.post('/supersep/release', (req, res) => {
    const id = String(req.query.id ?? '');
    const job = superSepJobs.get(id);
    if (job) job.released = true;
    res.json({ ok: true });
  });

  const server: Server = await new Promise(resolve => {
    const s = app.listen(0, '127.0.0.1', () => resolve(s));
  });
  const port = (server.address() as { port: number }).port;

  return {
    app,
    origin: `http://127.0.0.1:${port}`,
    hooks,
    close: () => new Promise<void>(resolve => server.close(() => resolve())),
  };
}
