import { test } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { heartmulaStageText } from './progress.js';
import { withHeartmulaTrigger } from './trigger.js';

// Isolate the settings database, audio files and engine transport from the app.
test('HeartMuLa contract, queued picks, validation, persistence, cancellation and residency', t => {
  const root = fs.mkdtempSync(path.resolve('.tmp-heartmula-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const script = `
    import assert from 'node:assert/strict';
    import fs from 'node:fs';
    import path from 'node:path';
    import { config } from './src/config.js';
    import { initDb, getDb, closeDb } from './src/db/database.js';
    import { heartmulaBackend } from './src/services/backends/heartmula/index.js';
    import { heartmulaClient } from './src/services/backends/heartmula/client.js';
    import { listHeartmulaAdapters } from './src/services/backends/heartmula/models.js';
    import { buildEnvelope } from './src/services/generation/envelope.js';
    import { getBackend, setActiveBackendId } from './src/services/backends/registry.js';
    import { evictOtherFamilies } from './src/services/generation/residency.js';
    import { trackBackendRelease, awaitBackendRelease } from './src/services/generation/backendRelease.js';
    import { pollUntilDone } from './src/services/generation/pollUntilDone.js';
    const calls = [];
    let rejectPick = false, failJob = false, cancelOnSubmit = false, cancelOnPoll = false, rejectUnload = false, polls = 0;
    let controller, residentBytes;
    const wav = Buffer.alloc(48);
    wav.write('RIFF'); wav.writeUInt32LE(40, 4); wav.write('WAVE', 8);
    wav.write('fmt ', 12); wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20);
    wav.writeUInt16LE(2, 22); wav.writeUInt32LE(48000, 24); wav.writeUInt32LE(192000, 28);
    wav.writeUInt16LE(4, 32); wav.writeUInt16LE(16, 34); wav.write('data', 36); wav.writeUInt32LE(4, 40);
    globalThis.fetch = async (url, init = {}) => {
      const route = new URL(url).pathname;
      const body = init.body ? JSON.parse(init.body) : undefined;
      calls.push({ url: String(url), route, body });
      if (route === '/models/loaded') return Response.json({loaded: []});
      if (route === '/heartmula/unload' && rejectUnload) return Response.json({unloaded: false});
      if (route.endsWith('/select-model')) {
        return rejectPick ? Response.json({ error: 'incompatible fixture pair', code: 'incompatible' }, { status: 409 })
          : Response.json({ selected: {}, requested: {}, adapter: {} });
      }
      if (route === '/heartmula/models') return Response.json({ variants: ['3B'], components: {
        global: [{type: 'Q8_0', variant: '3B', compatible: true, bytes: 123}, {type: 'F16', variant: '3B', compatible: false, error: 'RVQ mismatch'}],
        local: [{type: 'Q4_K_M', variant: '3B', compatible: true}], flow: [], scalar_decoder: [],
      }});
      if (route === '/heartmula/props') return Response.json({ synth_ready: true, available: true, encode_ready: false,
        ...(residentBytes === undefined ? {} : { vram: { resident_bytes: residentBytes } }) });
      if (route === '/heartmula/synth') {
        if (cancelOnSubmit) controller.abort();
        return Response.json({ id: 'engine-fixture' });
      }
      if (route === '/heartmula/job') return Response.json({
        status: failJob ? 'failed' : 'done', stage: 'frames', seed: '18446744073709551615', noise_seed: '17',
        pick: { global: 'Q8_0', local: 'Q4_K_M' }, end_reason: 'limit_hit',
        ...(failJob ? {error: {code: 'vram_insufficient', message: 'fixture VRAM refusal'}, admission: {need_bytes: 9000, free_bytes: 1000}} : {}),
      });
      if (route === '/job') {
        const query = new URL(url).searchParams;
        if (query.has('result')) return new Response(wav, { headers: {'content-type': 'audio/wav'} });
        if (query.has('cancel')) return Response.json({});
        if (cancelOnPoll) { controller.abort(); return Response.json({status: 'running'}); }
        if (failJob) return Response.json({status: 'failed'});
        return Response.json(polls++ === 0 ? {status: 'running', phase: 'dit_inference', phase_step: 2, phase_total: 3} : {status: 'done'});
      }
      return Response.json({unloaded: true});
    };
    initDb();
    try {
      getDb().prepare('INSERT INTO users (id, username) VALUES (?, ?)').run('fixture-user', 'fixture-user');
      const adapterDir = path.join(config.aceServer.adapters, 'heartmula-lm-adapters');
      fs.mkdirSync(path.join(adapterDir, 'run'), {recursive: true});
      const adapter = path.join(adapterDir, 'run', 'fixture.gguf');
      fs.writeFileSync(adapter, 'fixture');
      fs.writeFileSync(path.join(adapterDir, 'fixture.safetensors'), 'not a runtime adapter');
      assert.deepEqual(listHeartmulaAdapters(), [adapter]);
      assert.equal(getBackend('heartmula'), heartmulaBackend);
      assert.equal((await heartmulaBackend.capabilities()).core.duration.max, 240);
      const catalogue = await heartmulaBackend.models();
      assert.deepEqual(catalogue.buckets.global, ['Q8_0', 'F16']);
      assert.equal(catalogue.meta.global.F16.disabledReason, 'RVQ mismatch');
      assert.deepEqual(catalogue.lmAdapters, [adapter]);
      assert.equal(catalogue.buckets.muencoder, undefined, 'generation never requires encoder models');
      await heartmulaBackend.selectModel({variant: '3B', global: 'Q8_0', local: 'Q4_K_M', flow: 'Q8_0', scalar_decoder: 'F16', lmAdapter: adapter});
      setActiveBackendId('heartmula');
      const raw = {caption: 'ambient, piano', lyrics: '[verse]\\nfixture', duration: -1,
        seed: '18446744073709551615', heartmulaNoiseSeed: 17, heartmulaAdapterStrength: 0,
        postProcessingEnabled: false};
      const envelope = buildEnvelope(raw, 'fixture-user', 'fixture-job', Date.now());
      assert.equal(envelope.submission.duration, -1, 'retain the app Auto sentinel in the submission');
      assert.equal(envelope.common.duration, 240, 'Auto uses the maximum ceiling');
      assert.equal(envelope.options.heartmula.duration_s, 240);
      assert.equal(heartmulaBackend.resolveRequest({...raw, duration: undefined}).common.duration, 120);
      for (const duration of [1, 120, 240]) {
        assert.equal(heartmulaBackend.resolveRequest({...raw, duration}).options.duration_s, duration);
      }
      await heartmulaBackend.selectModel({global: 'F16', lmAdapter: ''});
      setActiveBackendId('ace');
      assert.equal(envelope.backendId, 'heartmula');
      assert.equal(envelope.models.global, 'Q8_0');
      assert.equal(envelope.options.heartmula.adapter, adapter);
      rejectPick = true;
      await assert.rejects(heartmulaBackend.selectModel({local: 'bad'}), /incompatible fixture pair/);
      assert.equal(heartmulaBackend.resolveRequest(raw).models.local, 'Q4_K_M');
      rejectPick = false;
      for (const bad of [
        {caption: ''}, {tags: []}, {lyrics: 123}, {duration: 241}, {duration: 0}, {duration: -2}, {duration: '-1'}, {duration: NaN},
        {seed: -1}, {seed: Number.MAX_SAFE_INTEGER + 1}, {seed: '18446744073709551616'},
        {heartmulaNoiseSeed: -2}, {heartmulaTopK: 1.2}, {heartmulaTemperature: 0},
        {heartmulaCfgScale: 11}, {heartmulaCodecSteps: 101}, {heartmulaCodecGuidance: 0},
        {heartmulaAdapterStrength: 3}, {heartmulaAdapter: '../escape.gguf'}, {heartmulaAdapter: 'fixture.safetensors'},
        {heartmulaGlobalModel: 8}, {batchSize: 2}, {randomSeed: 'yes'},
        {tags: '界'.repeat(1366)}, {lyrics: '界'.repeat(10923)},
      ]) assert.throws(() => heartmulaBackend.resolveRequest({...raw, ...bad}), undefined, JSON.stringify(bad));
      assert.equal((await heartmulaBackend.capabilities()).features.cover, true);
      const callsBeforeCover = calls.length;
      assert.throws(() => buildEnvelope({...raw, backendOverride: 'heartmula', taskType: 'cover'}, 'fixture-user', 'other', Date.now()), /HeartMuLa covers are not available yet/);
      assert.throws(() => heartmulaBackend.resolveRequest({...raw, heartmulaCover: {}}), /HeartMuLa covers are not available yet/);
      assert.equal(calls.length, callsBeforeCover, 'cover must not call a missing endpoint or fall back to synth');
      assert.equal(heartmulaBackend.resolveRequest({...raw, instrumental: true}).common.lyrics, '');
      // Adapter dials ride on the wire only with an adapter; plugin picks only with the toggle on.
      const dialled = heartmulaBackend.resolveRequest({...raw, heartmulaAdapter: adapter, heartmulaAdapterScaleMlp: 0, heartmulaAdapterScaleLate: 1.5}).options;
      assert.equal(dialled.adapter_scale_mlp, 0);
      assert.equal(dialled.adapter_scale_late, 1.5);
      assert.equal(heartmulaBackend.resolveRequest({...raw, heartmulaAdapter: ''}).options.adapter_scale_attn, undefined);
      assert.throws(() => heartmulaBackend.resolveRequest({...raw, heartmulaAdapter: adapter, heartmulaAdapterScaleAttn: 2.5}), /attention scale/);
      const picks = {inferMethod: 'heun', scheduler: 'linear', guidanceMode: 'apg', pluginParams: {'heun:x': 1}};
      assert.equal(heartmulaBackend.resolveRequest({...raw, ...picks}).options.infer_method, undefined);
      const plugged = heartmulaBackend.resolveRequest({...raw, ...picks, samplerPluginsEnabled: true}).options;
      assert.deepEqual([plugged.infer_method, plugged.scheduler, plugged.guidance_mode, plugged.flow_shift, plugged.plugin_params],
        ['heun', 'linear', 'apg', 1, {'heun:x': 1}]);
      // A deleted adapter file fails the render with a clear message and does not block other picks.
      const doomed = path.join(adapterDir, 'doomed.gguf');
      fs.writeFileSync(doomed, 'fixture');
      await heartmulaBackend.selectModel({lmAdapter: doomed});
      fs.rmSync(doomed);
      assert.throws(() => heartmulaBackend.resolveRequest({...raw, heartmulaAdapter: undefined}), /no longer installed/);
      await heartmulaBackend.selectModel({global: 'Q8_0'});
      assert.equal((await heartmulaBackend.models()).defaults.lmAdapter, '');
      const makeJob = () => ({id: 'fixture-job', userId: 'fixture-user', envelope, params: {...raw}, attempts: [], status: 'pending', createdAt: Date.now()});
      const makeContext = () => {
        controller = new AbortController();
        return {envelope, attempt: {attempt: 1, startedAt: Date.now(), effective: {models: {}}, reseeded: false, engineJobIds: []},
          signal: controller.signal, lease: {isCurrent: () => true}, pollUntilDone,
          hooks: {onEngineJob() {}, onStage() {}, onArtifact() {}}};
      };
      let job = makeJob(), ctx = makeContext();
      const outcome = await heartmulaBackend.generate(job, ctx);
      assert.equal(outcome.endReason, 'completed');
      assert.equal(job.acePhaseProgress, 'step 2/3');
      assert.deepEqual(ctx.attempt.engineJobIds, ['engine-fixture']);
      const sent = calls.find(c => c.route === '/heartmula/synth').body;
      assert.deepEqual(sent, {variant: '3B', global_type: 'Q8_0', local_type: 'Q4_K_M', flow_type: 'Q8_0', scalar_decoder_type: 'F16',
        adapter, adapter_scale: 0, adapter_scale_attn: 1, adapter_scale_mlp: 1, adapter_scale_early: 1, adapter_scale_mid: 1,
        adapter_scale_late: 1, adapter_rebase: 0.75, tags: raw.caption, lyrics: raw.lyrics, duration_s: 240, seed: raw.seed, noise_seed: 17,
        temperature: 1, topk: 50, cfg_scale: 1.5, codec_steps: 10, codec_guidance: 1.25});
      const row = getDb().prepare('SELECT * FROM songs WHERE id=?').get(outcome.songIds[0]);
      assert.equal(row.backend, 'heartmula');
      assert.equal(JSON.parse(row.generation_params).heartmulaResult.seed, raw.seed);
      assert.equal(outcome.artifacts[0].kind, 'audio');
      assert.ok(fs.existsSync(path.join(config.data.audioDir, path.basename(row.audio_url))));
      cancelOnSubmit = true;
      job = makeJob(); ctx = makeContext();
      assert.equal((await heartmulaBackend.generate(job, ctx)).endReason, 'cancelled');
      assert.ok(calls.some(c => c.url.includes('cancel=1')));
      assert.equal(getDb().prepare('SELECT count(*) AS n FROM songs').get().n, 1);
      cancelOnSubmit = false; cancelOnPoll = true;
      job = makeJob(); ctx = makeContext();
      assert.equal((await heartmulaBackend.generate(job, ctx)).endReason, 'cancelled');
      assert.equal(getDb().prepare('SELECT count(*) AS n FROM songs').get().n, 1);
      cancelOnPoll = false; failJob = true;
      job = makeJob(); ctx = makeContext();
      assert.equal((await heartmulaBackend.generate(job, ctx)).endReason, 'failed');
      assert.match(job.error, /vram_insufficient/);
      assert.equal(ctx.attempt.effective.admission.need_bytes, 9000);
      const before = calls.length;
      await evictOtherFamilies('heartmula');
      assert.ok(!calls.slice(before).some(c => c.route === '/heartmula/unload'));
      await evictOtherFamilies('ace');
      assert.ok(calls.slice(before).some(c => c.route === '/heartmula/unload'));
      await trackBackendRelease('heartmula', () => heartmulaBackend.releaseVram());
      await awaitBackendRelease();
      rejectUnload = true;
      await assert.rejects(trackBackendRelease('heartmula', () => heartmulaBackend.releaseVram()), /did not confirm/);
      await assert.rejects(awaitBackendRelease(), /not confirmed it freed/);
      rejectUnload = false;
      await awaitBackendRelease();
      // Nothing resident: no unload, which would queue on the engine's GPU worker (GH #341).
      residentBytes = 0;
      const idle = calls.length;
      await heartmulaBackend.releaseVram();
      assert.ok(calls.slice(idle).some(c => c.route === '/heartmula/props'));
      assert.ok(!calls.slice(idle).some(c => c.route === '/heartmula/unload'));
      residentBytes = 1;
      await heartmulaBackend.releaseVram();
      assert.ok(calls.slice(idle).some(c => c.route === '/heartmula/unload'));
    } finally { closeDb(); }
  `;
  try { execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], {
    timeout: 60_000, stdio: 'pipe',
    env: { ...process.env, DATA_DIR: path.join(root, 'library'),
      ACESTEPCPP_MODELS: path.join(root, 'models'), ACESTEPCPP_ADAPTERS: path.join(root, 'adapters'),
      TRAINING_DIR: path.join(root, 'training') },
  }); } catch (err: any) { throw new Error(err.stderr?.toString() || err.message); }
});

test('HeartMuLa queue card text follows the engine stage and ticks for the stall watchdog', () => {
  assert.equal(heartmulaStageText({ stage: 'loading_lm' })?.stage, 'Loading HeartMuLa LM...');
  const lm = heartmulaStageText({ stage: 'frames', frames_done: 750, max_frames: 3000 });
  assert.equal(lm?.stage, 'HeartMuLa LM: 60.0s of up to 240s: Step 750/3000');
  assert.equal(lm?.progress, 25);
  const codec = heartmulaStageText({ stage: 'codec_decode', codec_chunk: 1, codec_chunks: 3, codec_step: 15, codec_steps: 10 });
  assert.equal(codec?.stage, 'HeartCodec decode: chunk 2/3: Step 15/30');
  assert.equal(codec?.progress, 83);
  assert.match(lm!.stage, /: Step \d+/);
  assert.equal(heartmulaStageText({ stage: 'done' }), null);
  assert.equal(heartmulaStageText({ stage: 'frames', frames_done: 5, max_frames: 0 })?.progress, 10);
});

test('HeartMuLa adapter trigger leads the tags once, as in training', () => {
  assert.equal(withHeartmulaTrigger('pop, piano', 'zx9'), 'zx9, pop, piano');
  assert.equal(withHeartmulaTrigger('pop, ZX9, piano', 'zx9'), 'pop, ZX9, piano', 'already typed: left alone');
  assert.equal(withHeartmulaTrigger('pop', undefined), 'pop');
  assert.equal(withHeartmulaTrigger('', 'zx9'), 'zx9');
});
