import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import type { GenerationJob } from '../generation/jobTypes.js';
import type { GenerationContext } from './types.js';
import { runMinimaxGeneration } from './minimax/generate.js';
import { runYue2Generation } from './yue2/generate.js';
import { runSa3Generation } from './sa3/generate.js';
import { runHeartmulaGeneration } from './heartmula/generate.js';
import { pushLog } from '../../routes/logs.js';

test('decoder failure stage and count reach the failed Node job for every backend', async t => {
  const { config } = await import('../../config.js');
  const { TokenizerLoader } = await import('@lenml/tokenizers');
  const root = fs.mkdtempSync(path.resolve('.tmp-finite-'));
  const previousModels = config.aceServer.models;
  t.after(() => {
    config.aceServer.models = previousModels;
    fs.rmSync(root, { recursive: true, force: true });
  });
  config.aceServer.models = root;
  const tokenizerDir = path.join(root, 'onnx', 'sa3');
  fs.mkdirSync(tokenizerDir, { recursive: true });
  for (const file of ['tokenizer.json', 'tokenizer_config.json']) fs.writeFileSync(path.join(tokenizerDir, file), '{}');
  const source = path.join(root, 'source.wav');
  const wav = Buffer.alloc(44 + 16000);
  wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(8000, 24); wav.writeUInt32LE(16000, 28);
  wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
  wav.write('data', 36); wav.writeUInt32LE(16000, 40);
  fs.writeFileSync(source, wav);
  t.mock.method(TokenizerLoader, 'fromPreTrained', () => Object.assign(
    () => ({ input_ids: [1], attention_mask: [1] }), { pad_token_id: 0 },
  ));
  const failPoll = async () => { throw new Error('Generation failed on ace-server'); };
  for (const backend of ['minimax', 'yue2', 'sa3', 'sa3-refine', 'heartmula']) {
    const stage = { minimax: 'MM3 vocoder decode', yue2: 'YuE2 VAE decode',
      sa3: 'SA3 SAME decode', 'sa3-refine': 'SA3 final PCM', heartmula: 'HeartMuLa codec decode' }[backend];
    const message = `${stage}: 3 non-finite values out of 64`;
    const job = {
      id: `finite-${backend}`, status: 'pending', params: { caption: 'instrumental test',
        mm3DitBackend: 'ggml', yue2Coalesce: false, yue2AutoReplan: false, seed: 42,
        ...(backend === 'sa3-refine' ? { taskType: 'cover', sourceAudioUrl: source } : {}),
        yue2Abc: 'X:1\nM:4/4\nL:1/8\nK:C\nC8|\n' },
      envelope: { models: {}, options: { heartmula: { tags: 'instrumental', seed: 42 } },
        policy: {}, backendId: backend },
    } as unknown as GenerationJob;
    const deps = { signal: new AbortController().signal, pollUntilDone: failPoll,
      attempt: { effective: {}, engineJobIds: [] }, envelope: job.envelope,
      hooks: { onEngineJob() {} } } as unknown as GenerationContext;
    const requested: string[] = [];
    let delayedLog: Promise<void> | undefined;
    let logArrived = false;
    const mock = t.mock.method(globalThis, 'fetch', async (input: string | URL | Request) => {
      const url = new URL(String(input));
      const route = url.pathname;
      requested.push(url.href);
      if (route.endsWith('/props')) return Response.json({ max_lm_batch: 1 });
      if (route.endsWith('/tokenize-check')) return Response.json({ ok: true, tokens: 1, limit: 5000 });
      if (route === '/sa3-synth' || route === '/sa3-refine') {
        delayedLog = new Promise(resolve => setTimeout(() => {
          logArrived = true;
          pushLog(`[Synth-Finite] ${message}`, 'engine');
          resolve();
        }, 20));
        return Response.json({ error: message }, { status: 500 });
      }
      if (route.endsWith('/synth')) return Response.json({ id: 'engine-finite', job_id: 'engine-finite', seed: 42, duration: 4 });
      if (route === '/heartmula/job') return Response.json({ status: 'failed', error: { code: 'load_failed', message } });
      if (route.endsWith('/job')) return Response.json({ status: 'failed', error: message });
      throw new Error(`Unexpected request ${route}`);
    });
    try {
      if (backend === 'minimax') await runMinimaxGeneration(job, deps);
      if (backend === 'yue2') await runYue2Generation(job, deps);
      if (backend.startsWith('sa3')) await runSa3Generation(job, deps);
      if (backend === 'heartmula') await runHeartmulaGeneration(job, deps);
      assert.equal(job.status, 'failed', `${backend}: ${job.error}`);
      if (backend.startsWith('sa3')) assert.equal(logArrived, false, 'job must fail before the delayed log arrives');
      assert.ok(job.error?.includes(message), `${backend}: ${job.error}`);
      assert.ok(!requested.some(url => new URL(url).searchParams.has('result')), backend);
    } finally {
      await delayedLog;
      mock.mock.restore();
    }
  }
});
