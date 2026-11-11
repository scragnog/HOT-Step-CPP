import { test } from 'node:test';
import assert from 'node:assert/strict';
import { releaseYue2RenderLane, jobCover, mapYue2Params, planYue2Jobs, yue2PlanMode, yue2TrackReplayParams, type Yue2PreparedJob } from './generate.js';
import { yue2Seed, yue2SeedPlus } from './client.js';
import type { GenerationJob } from '../../generation/jobTypes.js';
import type { Yue2PersistedSelection } from './index.js';

// #S3: a cover job's source identity and approved ABC are captured into the
// envelope at submit (index.ts's resolveRequest) and read ONLY from there —
// never from job.params, the raw request body. These tests build a job whose
// envelope and params deliberately disagree, so a regression that reads the
// old params path instead shows up immediately.

function fakeJob(envelopeOptions: Record<string, unknown> | undefined, params: Record<string, unknown>): GenerationJob {
  return {
    envelope: { backendId: 'yue2', options: envelopeOptions ? { yue2: envelopeOptions } : {} },
    params,
  } as unknown as GenerationJob;
}

test('jobCover reads the ABC from the envelope, not from job.params', () => {
  const job = fakeJob(
    { yue2Cover: { sourceId: 'song-1' }, yue2Abc: 'X:1\nK:C\nENVELOPE\n' },
    { yue2Abc: 'X:1\nK:C\nSTALE PARAMS COPY\n' },
  );
  const captured = jobCover(job);
  assert.equal(captured?.abc, 'X:1\nK:C\nENVELOPE\n');
});

test('jobCover reads source identity from the envelope, not from job.params', () => {
  const job = fakeJob(
    { yue2Cover: { sourceId: 'envelope-id', sourceLabel: 'Envelope Label' }, yue2Abc: 'X:1\n' },
    { yue2Cover: { sourceId: 'stale-params-id', sourceLabel: 'Stale Params Label' }, yue2Abc: 'X:1\n' },
  );
  const captured = jobCover(job);
  assert.deepEqual(captured?.cover, { sourceId: 'envelope-id', sourceLabel: 'Envelope Label' });
});

test('jobCover reads full score, chord choice and rendered ABC from the envelope only', () => {
  const job = fakeJob(
    { yue2Cover: { sourceId: 'source', keepChords: false, fullScore: 'X:1\n"Am"C|' }, yue2Abc: 'X:1\nC|' },
    { yue2Cover: { sourceId: 'stale', keepChords: true, fullScore: 'wrong' }, yue2Abc: 'wrong' },
  );
  assert.deepEqual(jobCover(job), {
    cover: { sourceId: 'source', keepChords: false, fullScore: 'X:1\n"Am"C|' },
    abc: 'X:1\nC|',
  });
});

test('jobCover reads all five score choices from the captured envelope', () => {
  const cover = { sourceId: 'source', voices: 'vocal', keepChords: false,
    tempo: 'free', key: 'F#m', cfgScale: 1.25, fullScore: 'X:1\nK:Em\n"Em"E|' };
  const job = fakeJob(
    { yue2Cover: cover, yue2Abc: 'X:1\nK:F#m\nF|' },
    { yue2Cover: { ...cover, key: 'Bm', cfgScale: 2 }, yue2Abc: 'wrong' },
  );
  assert.deepEqual(jobCover(job), { cover, abc: 'X:1\nK:F#m\nF|' });
});

test('MusicXML cover reaches the YuE2 request from its captured score without source audio', () => {
  const score = 'X:1\nK:C\nC4|';
  const job = fakeJob(
    { yue2Cover: { sourceKind: 'musicxml' }, yue2Abc: score },
    { caption: 'quiet strings', yue2Abc: 'stale', sourceAudioUrl: '/audio/old.wav' },
  );
  const captured = jobCover(job);
  assert.deepEqual(captured, { cover: { sourceKind: 'musicxml' }, abc: score });
  const mapped = mapYue2Params({ ...job.params, yue2Abc: captured!.abc }, {
    lm: '', vae_variant: '', adapters: {
      ar: { path: '', scales: { global: 1, attn: 1, mlp: 1, early: 1, mid: 1, late: 1 } },
      nar: { path: '', scales: { global: 1, attn: 1, mlp: 1, early: 1, mid: 1, late: 1 } },
    },
  });
  assert.equal(mapped.req.abc, score);
  assert.equal('sourceAudioUrl' in mapped.req, false);
});

test('an ordinary (non-cover) job has no captured cover, regardless of job.params', () => {
  const job = fakeJob({ yue2Abc: 'X:1\n' }, { yue2Cover: { sourceId: 'ignored' } });
  assert.equal(jobCover(job), undefined);
});

test('a job with no envelope options for this backend has no captured cover', () => {
  const job = fakeJob(undefined, {});
  assert.equal(jobCover(job), undefined);
});

// Off by default (2026-10-12: condition strength 1 with the old
// soft-by-default ruined lyric adherence). Soft (bias) and hard (mask) both
// stay selectable, an explicit choice (current or saved) always wins, and a
// schedule that cannot be built must never fail the render.

const scales = { global: 1, attn: 1, mlp: 1, early: 1, mid: 1, late: 1 };
const pick: Yue2PersistedSelection = {
  lm: '', vae_variant: '',
  adapters: { ar: { path: '', scales }, nar: { path: '', scales } },
};
const abc = ['X:1', 'M:4/4', 'L:1/4', 'Q:1/4=60', 'K:C', '% verse', 'V: Vocal', 'C4|', ''].join('\n');
const lyrics = '[Verse]\nfirst line';

test('batch member metadata replays its own score and exact semantic and noise seeds', () => {
  const firstAbc = 'X:1\nK:C\nC4|';
  const secondAbc = 'X:2\nK:C\nD4|';
  const { req } = mapYue2Params({ caption: 'folk', seed: 7, yue2Abc: firstAbc }, pick);
  const saved = yue2TrackReplayParams({ ...req, lm_batch_size: 3, synth_batch_size: 2,
    songs: [{ style: 'folk', abc: firstAbc, seed: 7 }] }, {
    song: 1, variation: 1, seed: '18446744073709551610', noise_seed: '18446744073709551611', abc: secondAbc,
  });
  assert.equal(saved.yue2Abc, secondAbc);
  assert.equal(saved.yue2Request.abc, secondAbc);
  assert.equal(saved.yue2Request.songs, undefined);
  assert.equal(saved.yue2Request.lm_batch_size, undefined);
  assert.equal(saved.yue2Request.synth_batch_size, undefined);
  const replay = mapYue2Params({ caption: 'folk', ...saved, randomSeed: false, batchSize: 1,
    yue2BatchSize: 1, yue2Variations: 1 }, pick).req;
  assert.equal(replay.abc, secondAbc);
  assert.equal(replay.seed, '18446744073709551610');
  assert.equal(replay.noise_seed, '18446744073709551611');
  assert.equal(replay.lm_batch_size, undefined);
  assert.equal(replay.synth_batch_size, undefined);
});

test('single-track metadata retains supplied scores and falls back to the final score on older engines', () => {
  const { req } = mapYue2Params({ caption: 'folk', lyrics, seed: 0, yue2Abc: abc }, pick);
  const td = { song: 0, variation: 0, seed: 0, noise_seed: 0 };
  assert.equal(yue2TrackReplayParams(req, td).yue2Abc, req.abc);
  const saved = yue2TrackReplayParams({ ...req, abc: undefined }, td, abc);
  assert.equal(saved.yue2Abc, abc);
  assert.equal(saved.seed, 0);
  assert.equal(saved.yue2NoiseSeed, 0);
});

test('score provenance survives app planning, engine fallback and instrumental planning', () => {
  for (const instrumental of [false, true]) {
    for (const yue2AutoReplan of [false, true]) {
      const mapped = mapYue2Params({ caption: 'folk', lyrics, instrumental, yue2AutoReplan }, pick);
      assert.equal(mapped.abcSupplied, false, 'no user score, including engine fallback');
      // planYue2Jobs installs the chosen LM score into the mutable request.
      mapped.req.abc = abc;
      assert.equal(mapped.abcSupplied, false, 'an app-planned score is still LM-planned');
    }
  }
});

test('score provenance counts only supplied scores the request actually uses', () => {
  for (const yue2Abc of [undefined, '', '   ', abc]) {
    for (const yue2Cot of ['full', 'off']) {
      const mapped = mapYue2Params({ caption: 'folk', lyrics, yue2Abc, yue2Cot }, pick);
      assert.equal(mapped.abcSupplied, yue2Abc === abc && yue2Cot !== 'off');
    }
  }
});

test('yue2LyricSchedule defaults to off when a score and lyrics are supplied', () => {
  const { req, notes } = mapYue2Params({ caption: 'folk', lyrics, yue2Abc: abc }, pick);
  assert.equal(req.lyric_schedule, undefined);
  assert.ok(!notes.some(n => n.startsWith('Lyric schedule')));
});

test('an explicit "bias" is honoured with a usable score', () => {
  const { req, notes } = mapYue2Params({ caption: 'folk', lyrics, yue2Abc: abc, yue2LyricSchedule: 'bias' }, pick);
  assert.equal(req.lyric_schedule?.mode, 'bias');
  assert.equal(req.lyric_schedule?.bias, -4);
  assert.ok(notes.some(n => n.startsWith('Lyric schedule: bias')));
});

test('an explicit "off" is honoured even with a usable score', () => {
  const { req, notes } = mapYue2Params({ caption: 'folk', lyrics, yue2Abc: abc, yue2LyricSchedule: 'off' }, pick);
  assert.equal(req.lyric_schedule, undefined);
  assert.ok(!notes.some(n => n.startsWith('Lyric schedule')));
});

test('hard mask stays selectable but is never the default', () => {
  const { req } = mapYue2Params({ caption: 'folk', lyrics, yue2Abc: abc, yue2LyricSchedule: 'mask' }, pick);
  assert.equal(req.lyric_schedule?.mode, 'mask');
});

test('an explicit CFG other than 1 skips an explicitly requested schedule instead of the engine rejecting the job', () => {
  // yue2-pipeline.h:596 refuses lyric_schedule outright once CFG is on
  // ("needs cfg_scale 1 (no rule for the guidance set)"); the mapper must
  // never emit a schedule the engine will reject.
  const { req, notes } = mapYue2Params(
    { caption: 'folk', lyrics, yue2Abc: abc, yue2LyricSchedule: 'bias', yue2CfgScale: '1.2' }, pick);
  assert.equal(req.lyric_schedule, undefined);
  assert.equal(req.cfg_scale, 1.2);
  assert.ok(notes.some(n => n.includes('skipped') && n.includes('CFG 1.2')));
});

test('an explicitly requested schedule that cannot be built falls back instead of failing the render', () => {
  // No "% section" label line: buildYue2LyricSchedule throws "needs a score
  // with labelled sections" — the render must still go ahead without C6.
  const unlabelled = ['X:1', 'M:4/4', 'L:1/4', 'Q:1/4=60', 'K:C', 'C4|', ''].join('\n');
  const { req, notes } = mapYue2Params(
    { caption: 'folk', lyrics, yue2Abc: unlabelled, yue2LyricSchedule: 'bias' }, pick);
  assert.equal(req.lyric_schedule, undefined);
  assert.ok(notes.some(n => n.includes('not built') && n.includes('rendering without it')));
});

test('no score or lyrics: an explicitly requested schedule never fires and never fails', () => {
  const { req } = mapYue2Params({ caption: 'folk', yue2LyricSchedule: 'bias' }, pick);
  assert.equal(req.lyric_schedule, undefined);
});

test('yue2PlanMode: every auto-replan, instrumental, cot and supplied-score combination', () => {
  for (const autoReplan of [undefined, true, false]) {
    for (const instrumental of [true, false]) {
      for (const cot of ['full', 'melody', 'off'] as const) {
        for (const supplied of [false, true]) {
          const params = { instrumental, ...(autoReplan === undefined ? {} : { yue2AutoReplan: autoReplan }) };
          const req = { cot, lyrics: instrumental ? '' : lyrics, ...(supplied ? { abc } : {}) };
          const want = cot === 'off' || supplied ? null : autoReplan !== false ? 'replan' : instrumental ? 'once' : null;
          assert.equal(yue2PlanMode(params, req), want, JSON.stringify({ autoReplan, instrumental, cot, supplied }));
        }
      }
    }
  }
  // Empty lyrics without the Instrumental switch is an instrumental too.
  assert.equal(yue2PlanMode({ yue2AutoReplan: false }, { cot: 'melody', lyrics: '' }), 'once');
  // A planner strength of its own needs a plan job apart from the render.
  assert.equal(yue2PlanMode({ yue2AutoReplan: false }, { cot: 'full', lyrics }, true), 'once');
  assert.equal(yue2PlanMode({ yue2AutoReplan: false }, { cot: 'off', lyrics }, true), null);
});

test('a supplied score is moved for an instrumental whatever re-planning is, and left alone for a vocal cover', () => {
  const native = ['X:1', 'T:', 'M:4/4', 'L:1/32', 'Q:1/4=90', 'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
    'V: Ins clef=treble name="Ins Melody" snm="Inst."', 'K:C', '% verse', 'V: Vocal', '"C"c16"G"d16|', 'V: Ins', 'Z|', ''].join('\n');
  for (const yue2AutoReplan of [true, false]) {
    for (const cot of ['full', 'melody']) {
      const inst = mapYue2Params({ caption: 'piano', instrumental: true, yue2Abc: native, yue2Cot: cot, yue2AutoReplan }, pick).req;
      assert.match(inst.abc!, /\nV: Vocal\n"C"z16"G"z16\|\nV: Ins\nc16d16\|$/, `instrumental, autoReplan ${yue2AutoReplan}, cot ${cot}`);
      assert.equal(yue2PlanMode({ instrumental: true, yue2AutoReplan }, inst), null);
      const vocal = mapYue2Params({ caption: 'folk', lyrics, yue2Abc: native, yue2Cot: cot, yue2AutoReplan }, pick).req;
      assert.equal(vocal.abc, native.trim(), `vocal, autoReplan ${yue2AutoReplan}, cot ${cot}`);
    }
  }
});

test('re-plan seeds follow from the job seed, so a pinned seed replays the same plans', async () => {
  const { yue2ReplanSeed } = await import('./generate.js');
  assert.equal(yue2ReplanSeed(2110292260, 1), 2110292260);
  const run = (base: number) => Array.from({ length: 10 }, (_, i) => yue2ReplanSeed(base, i + 1));
  assert.deepEqual(run(4242), run(4242));
  assert.equal(new Set(run(4242)).size, 10);
  for (const s of run(4294967295)) assert.ok(Number.isInteger(s) && Number(s) >= 0 && Number(s) < 2 ** 32);
  assert.equal(yue2ReplanSeed('15213961153666092475', 1), '15213961153666092475');
  assert.equal(yue2ReplanSeed('15213961153666092475', 2), '15213961153667092478');
});

test('a random seed is drawn exact (32 bits), so a take replays under the seed it recorded', () => {
  for (let i = 0; i < 200; i++) {
    const seed = mapYue2Params({ caption: 'folk', lyrics: 'la', randomSeed: true }, pick).req.seed!;
    assert.ok(Number.isSafeInteger(seed) && Number(seed) >= 0);
    assert.equal(Number(JSON.stringify({ seed }).match(/\d+/)![0]), seed);
  }
  assert.equal(mapYue2Params({ caption: 'folk', lyrics: 'la', seed: 12345 }, pick).req.seed, 12345);
});

test('the largest random seed stays exact after the engine derives from it (song offset, retries)', () => {
  const real = Math.random;
  Math.random = () => 1 - Number.EPSILON;
  try {
    const seed = mapYue2Params({ caption: 'folk', lyrics: 'la', randomSeed: true }, pick).req.seed!;
    assert.equal(seed, 2 ** 32 - 1);
    const derived = BigInt(seed) + 1000003n * 100n + 64n;
    assert.equal(BigInt(JSON.parse(JSON.stringify({ s: Number(derived) })).s), derived);
  } finally { Math.random = real; }
});

test('a seed above 2^53 reaches the engine request as its exact decimal string', () => {
  const big = '15213961153666092475';
  const { req } = mapYue2Params({ caption: 'folk', lyrics: 'la', seed: big, yue2NoiseSeed: big }, pick);
  assert.equal(req.seed, big);
  assert.equal(req.noise_seed, big);
  assert.match(JSON.stringify(req), /"seed":"15213961153666092475"/);
  assert.equal(yue2SeedPlus(big, 3), '15213961153666092478');
  assert.equal(yue2Seed(big), big);
  assert.equal(yue2Seed('42'), 42);        // a safe value stays a plain number
  assert.equal(yue2Seed(123456789), 123456789); // legacy numeric sessions
  assert.equal(yue2Seed('18446744073709551616'), undefined); // past uint64
});

test('planning keeps a pinned uint64 seed: engine plan request, then the render request', async t => {
  const big = '15213961153666092475';
  const abc = ['X:1', 'T:', 'M:4/4', 'L:1/32', 'Q:1/4=90', 'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
    'V: Ins clef=treble name="Ins Melody" snm="Inst."', 'K:C', '% verse', 'V: Vocal', '"C"c16"G"d16|', 'V: Ins', 'Z|', ''].join('\n');
  const sent: Array<Record<string, any>> = [];
  t.mock.method(globalThis, 'fetch', async (input: string | URL | Request, init?: RequestInit) => {
    const url = new URL(String(input));
    if (url.pathname === '/yue2/props') return Response.json({ max_plan_batch: 1 });
    if (url.pathname === '/yue2/synth') { sent.push(JSON.parse(String(init?.body))); return Response.json({ id: 'p1' }); }
    assert.equal(url.pathname, '/job');
    const songs = sent[sent.length - 1].songs as Array<{ seed: string }>;
    return Response.json({ status: 'done', tracks: songs.map((s, song) => ({ song, seed: s.seed, noise_seed: s.seed, variation: 0, abc, end_reason: 'completed' })) });
  });
  for (const yue2AutoReplan of [true, false]) {
    sent.length = 0;
    const { req } = mapYue2Params({ caption: 'folk', lyrics: 'la', seed: big, instrumental: !yue2AutoReplan, yue2AutoReplan, yue2ReplanAttempts: 1 }, pick);
    assert.equal(req.seed, big);
    const m = { job: { ...fakeJob(undefined, { instrumental: !yue2AutoReplan, yue2AutoReplan, yue2ReplanAttempts: 1 }), envelope: { backendId: 'yue2', models: {}, options: {} }, status: 'queued' }, req, log: () => {} } as unknown as Yue2PreparedJob;
    await planYue2Jobs([m]);
    assert.equal(sent[0].songs[0].seed, big, `plan request, autoReplan ${yue2AutoReplan}`);
    assert.equal(m.req.seed, big, `render request, autoReplan ${yue2AutoReplan}`);
  }
});

test('the experimental full-loop solvers and Plan Overdrive map to their plugins and params', () => {
  const lie = mapYue2Params({ caption: 'folk', yue2NarSolver: 'tlie', yue2LieAmount: -0.1, yue2LieCurve: 'sections',
    yue2LieSections: 'chorus=0.25', yue2NarScheduler: 'plan_overdrive', yue2OdShape: 'pulse', yue2OdDrive: 2 }, pick);
  assert.equal(lie.req.infer_method, 'md_tlie_yue2');
  assert.equal(lie.req.scheduler, 'md_plan_overdrive_yue2');
  assert.equal(lie.req.plugin_params?.['md_tlie_yue2:lie'], -0.1);
  assert.equal(lie.req.plugin_params?.['md_tlie_yue2:lie_curve'], 'sections');
  assert.equal(lie.req.plugin_params?.['md_tlie_yue2:section_lies'], 'chorus=0.25');
  assert.equal(lie.req.plugin_params?.['md_plan_overdrive_yue2:shape'], 'pulse');
  assert.equal(lie.req.plugin_params?.['md_plan_overdrive_yue2:drive'], 2);
  const eq = mapYue2Params({ caption: 'folk', yue2NarSolver: 'latent_eq', yue2EqGains: '0=1.2', yue2EqSection: 'chorus' }, pick);
  assert.equal(eq.req.infer_method, 'md_latent_eq_yue2');
  assert.equal(eq.req.plugin_params?.['md_latent_eq_yue2:gains'], '0=1.2');
  assert.equal(eq.req.plugin_params?.['md_latent_eq_yue2:section_only'], 'chorus');
});

test('an unlisted select value or an oversized text param falls back to the plugin default with a note', () => {
  const { req, notes } = mapYue2Params({ caption: 'folk', yue2NarSolver: 'tlie', yue2LieCurve: 'sideways',
    yue2LieSections: 'x'.repeat(201) }, pick);
  assert.equal(req.plugin_params?.['md_tlie_yue2:lie_curve'], undefined);
  assert.equal(req.plugin_params?.['md_tlie_yue2:section_lies'], undefined);
  assert.ok(notes.includes('Invalid yue2LieCurve — using the plugin default'));
  assert.ok(notes.includes('Invalid yue2LieSections — using the plugin default'));
});

// #286: the BPM/key tail is optional per render; absent means on.
test('yue2CaptionBpmKey: on and absent keep the tail, off leaves it out', async () => {
  const fs = await import('node:fs');
  const os = await import('node:os');
  const nodePath = await import('node:path');
  const dir = fs.mkdtempSync(nodePath.join(os.tmpdir(), 'yue2-tail-'));
  try {
    const header = Buffer.from(JSON.stringify({ __metadata__: { trigger: 'tt', style_template: 'upstream' } }));
    const len = Buffer.alloc(8);
    len.writeBigUInt64LE(BigInt(header.length));
    const file = nodePath.join(dir, 'native-nar.safetensors');
    fs.writeFileSync(file, Buffer.concat([len, header]));
    const adapted: Yue2PersistedSelection = {
      ...pick, adapters: { ar: { path: '', scales }, nar: { path: file, scales } },
    };
    const base = { caption: 'sludgy riffs doom, 72 BPM, key of D minor.', lyrics, bpm: 90, keyScale: 'A minor' };
    const style = (extra: Record<string, unknown>) => mapYue2Params({ ...base, ...extra }, adapted).req.style;
    const on = 'tt, in the style of tt. sludgy riffs doom 90 BPM, key of A minor.';
    assert.equal(style({}), on, 'older request without the field');
    assert.equal(style({ yue2CaptionBpmKey: true }), on);
    assert.equal(style({ yue2CaptionBpmKey: false }), 'tt, in the style of tt. sludgy riffs doom');
    // No adapter: the caption goes verbatim, so off still peels its own tail.
    assert.equal(mapYue2Params({ ...base, yue2CaptionBpmKey: false }, pick).req.style, 'sludgy riffs doom');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('inline YuE2 renders hold the lane until completion; eligible renders hand it over', async t => {
  const { config } = await import('../../../config.js');
  const lane = await import('../../generation/gpuLane.js');
  const saved = config.aceServer.keepLoaded;
  t.after(() => { config.aceServer.keepLoaded = saved; });
  for (const keepLoaded of [false, true]) {
    for (const params of [{}, { yue2NarSolver: 'midpoint', yue2NarScheduler: 'uniform' },
      { yue2NarSolver: 'tlie', yue2NarScheduler: 'plan_overdrive' },
      { yue2NarSolver: 'wasserstein' }, { yue2NarSolver: 'ancestral' },
      { yue2NarSolver: 'latent_eq' }, { yue2NarSolver: 'tlie' }, { yue2NarScheduler: 'ht_v3' },
      { yue2NarScheduler: 'plan_aware' }, { yue2NarScheduler: 'plan_loose' }, { yue2NarScheduler: 'plan_overdrive' }]) {
      for (const phase of ['queued', 'plan', 'semantic', 'nar', 'vae_decode', 'done']) {
        config.aceServer.keepLoaded = keepLoaded;
        const { req } = mapYue2Params({ caption: 'folk', ...params }, pick);
        const expected = keepLoaded && !req.infer_method && !req.scheduler
          && (phase === 'nar' || phase === 'vae_decode');
        let finish!: () => void;
        const rendering = new Promise<void>(resolve => { finish = resolve; });
        let lease!: Parameters<typeof lane.releaseGpuLane>[0];
        const first = lane.runOnWorkerLane(async held => { lease = held; await rendering; }, { family: 'yue2' });
        let secondStarted = false;
        const second = lane.runOnWorkerLane(async () => { secondStarted = true; }, { family: 'yue2' });
        await Promise.resolve();
        let releases = 0;
        const released = releaseYue2RenderLane(req, phase, {
          releaseLane: () => { releases++; return lane.releaseGpuLane(lease); },
          nextLaneFamily: lane.gpuLaneNextFamily,
        });
        await Promise.resolve();
        try {
          const label = JSON.stringify({ keepLoaded, params, phase });
          assert.equal(released, expected, label);
          assert.equal(releases, Number(expected), label);
          assert.equal(secondStarted, expected, label);
        } finally {
          finish();
          await Promise.all([first, second]);
        }
        assert.equal(secondStarted, true, 'the next job starts after completion');
      }
    }
  }
});

test('YuE2 handoff waits for other families and tolerates absent callbacks', async () => {
  const { config } = await import('../../../config.js');
  const saved = config.aceServer.keepLoaded;
  config.aceServer.keepLoaded = true;
  try {
    const { req } = mapYue2Params({ caption: 'folk' }, pick);
    let releases = 0;
    for (const family of [undefined, 'ace', 'minimax-m3']) {
      assert.equal(releaseYue2RenderLane(req, 'nar', {
        releaseLane: () => { releases++; return true; }, nextLaneFamily: () => family,
      }), false);
    }
    assert.equal(releases, 0);
    assert.equal(releaseYue2RenderLane(req, 'nar', { nextLaneFamily: () => 'yue2' }), false);
    assert.equal(releaseYue2RenderLane(req, 'nar', { releaseLane: () => true }), false);
    assert.equal(releaseYue2RenderLane(req, 'nar', {
      releaseLane: () => false, nextLaneFamily: () => 'yue2',
    }), false);
  } finally { config.aceServer.keepLoaded = saved; }
});
