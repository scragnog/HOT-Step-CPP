import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { encodeWav } from '../../generation/audioLevel.js';
import type { GenerationJob } from '../../generation/jobTypes.js';
import { runAceGeneration } from './generate.js';

test('ACE complete sends its source audio and track to synthesis without an LM pass', async (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ace-complete-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const source = path.join(dir, 'stem.wav');
  const wav = encodeWav({
    channels: [Float32Array.of(0.125, -0.25)], sampleRate: 48000, numChannels: 1, sourceFormat: 's16',
  }, 's16');
  fs.writeFileSync(source, wav);

  const requests: Array<{ route: string; init?: RequestInit }> = [];
  t.mock.method(globalThis, 'fetch', async (url: string, init?: RequestInit) => {
    requests.push({ route: new URL(url).pathname, init });
    // Refuse VAE encoding to exercise the raw-audio fallback. Also stop at
    // synth submission to avoid polling the engine or writing a song.
    return new Response('test engine', { status: 400 });
  });

  const job = {
    id: 'complete-test', status: 'running', params: {
      prompt: 'Arrange around this stem', taskType: 'complete', trackName: 'vocals', sourceAudioUrl: source,
    },
  } as GenerationJob;
  await runAceGeneration(job, {
    signal: new AbortController().signal,
    pollUntilDone: async () => assert.fail('synth submission should stop the test'),
  });

  assert.deepEqual(requests.map(({ route }) => route), ['/mm3/unload', '/vae', '/synth']);
  const { body, headers } = requests[2].init!;
  const form = await new Response(body, { headers }).formData();
  const request = form.get('request');
  assert.ok(typeof request === 'string');
  const fields = JSON.parse(request);
  assert.equal(fields.task_type, 'complete');
  assert.equal(fields.track, 'vocals');
  const audio = form.get('audio');
  assert.ok(audio instanceof Blob);
  assert.deepEqual(Buffer.from(await audio.arrayBuffer()), wav);
});
