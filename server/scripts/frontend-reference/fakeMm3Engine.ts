// fakeMm3Engine.ts — the MiniMax-Music3 part of the fake ace-server.
//
// Adds the /mm3/* endpoints the MiniMax backend calls (props, tokenize-check,
// synth, the live window stream) to the fake engine's own Express app, so the
// real MM3 backend code and the real GET /api/generate/mm3/stream/:id route
// run end to end. Engine behaviour only, never Node's:
//
//   - POST /mm3/synth hands the render to the fake engine's own POST /synth,
//     so its id is an ordinary /job id: GET /job polling, cancel and the
//     result fetch work exactly as for an ACE render.
//   - GET /mm3/stream?id= sends WINDOWS complete WAVs back to back, then ends,
//     the way the engine streams a finished take. One reader per take, as the
//     engine allows: a second open is 409.
//   - GET /mm3/job is not served (404): the backend treats stage detail as
//     optional garnish.

import type { FakeAceEngine } from './fakeEngine.js';
import { makeFixtureWav } from './fixtures.js';

export const MM3_WINDOWS = 3;
/** One window: 0.2 s of 16-bit stereo silence at 44.1 kHz (the MM3 rate). */
export const mm3Window = (index: number) => {
  const wav = makeFixtureWav(0.2, 44100, 2);
  wav.writeInt16LE(index + 1, 44);   // distinct first sample per window
  return wav;
};

export function installFakeMm3(engine: FakeAceEngine): { streamed: Map<string, number> } {
  const streamed = new Map<string, number>();
  const readers = new Set<string>();
  const app = engine.app;

  app.get('/mm3/props', (_req, res) => {
    res.json({
      backend: 'minimax-m3', model: 'MiniMax-Music3', available: true, loaded: true, synth_ready: true,
      prompt_token_limit: 5000, max_audio_frames_limit: 9000, errors: [],
    });
  });

  app.post('/mm3/tokenize-check', (req, res) => {
    const lyrics = String(req.body?.lyrics ?? '');
    res.json({ tokens: 12, limit: 5000, ok: true, instrumental: !lyrics.trim() || lyrics.includes('[Instrumental]') });
  });

  app.post('/mm3/synth', async (req, res) => {
    const body = (req.body ?? {}) as Record<string, unknown>;
    const submitted = await fetch(`${engine.origin}/synth?format=wav`, { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify([{ caption: body.caption ?? '' }]) });
    const { id } = await submitted.json() as { id: string };
    res.json({
      job_id: id, id, seed: 1, seed_str: '1', takes: 1, max_frames: 250, duration: 10,
      prompt_tokens: 12, prompt_token_limit: 5000, instrumental: true, steps: 8, cfg_flow: 1, wav_bits: 16,
      streaming: body.stream === true,
    });
  });

  // A finished take's whole WAV (the backend saves each take from here).
  app.get('/mm3/take', (_req, res) => {
    res.setHeader('Content-Type', 'audio/wav');
    res.send(makeFixtureWav(0.6, 44100, 2));
  });

  app.get('/mm3/stream', async (req, res) => {
    const key = `${String(req.query.id ?? '')}:${String(req.query.take ?? '0')}`;
    if (readers.has(key)) { res.status(409).json({ error: 'This take already has a reader' }); return; }
    readers.add(key);
    res.setHeader('Content-Type', 'audio/wav');
    for (let i = 0; i < MM3_WINDOWS; i++) {
      res.write(mm3Window(i));
      streamed.set(key, i + 1);
      await new Promise(r => setTimeout(r, 50));
    }
    res.end();
  });

  return { streamed };
}
