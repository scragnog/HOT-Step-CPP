import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { config } from '../config.js';
import { buildSessionRender, createBuildSession, editBuildPlan, generateBuildTakes, getBuildSession, keepBuildTake, undoBuildKeep } from './yue2Takes.js';
import type { Yue2SynthRequest, Yue2TrackDetail } from './backends/yue2/client.js';
import { listBackends } from './backends/registry.js';

for (const b of listBackends()) if (b.id !== 'yue2' && b.releaseVram) b.releaseVram = async () => {};

const score = [
  'X:1', 'M:4/4', 'L:1/16', 'Q:1/4=120', 'V: Vocal clef=treble', 'V: Ins clef=treble', 'K:C',
  '% intro', 'V: Vocal', 'Z2|', 'V: Ins', 'C16|E16|',
  '% verse', 'V: Vocal', 'C4D4E4F4|G16|', 'V: Ins', 'Z2|',
  '% chorus', 'V: Vocal', 'c16|c16|', 'V: Ins', 'Z2|',
  '% outro', 'V: Vocal', 'Z2|', 'V: Ins', 'C16|E16|', '',
].join('\n');
const lyrics = '[intro]\n\n[verse]\nfirst line here\n\n[chorus]\nhook line now\n\n[outro]\n';

function pcm(): Buffer {
  const wav = Buffer.alloc(44 + 1920);
  wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(2, 22);
  wav.writeUInt32LE(48000, 24); wav.writeUInt16LE(16, 34); wav.write('data', 36);
  wav.writeUInt32LE(1920, 40);
  return wav;
}

/** The label a plan prefix ends on, and the section lines a draw writes: two
 *  bars, with a chord named for the section and seed so draws differ. */
const labelOf = (prefix: string) => prefix.trimEnd().split('\n').at(-1)!.slice(1).trim();
const section = (label: string, seed: number) => `V: Vocal\n"${label}${seed}"c16|c16|\nV: Ins\nZ2|\n`;
const chordIn = (abc: string, label: string) => new RegExp(`"${label}(\\d+)"`).exec(abc)?.[1];

function setup(t: test.TestContext, behaviour: { props?: object; draw?: (label: string, seed: number, attempt: number) => 'ok' | 'limit_hit' | 'empty' } = {}) {
  const previousDir = config.data.dir;
  const dir = fs.mkdtempSync(path.resolve('.test-yue2-plan-'));
  config.data.dir = dir;
  t.after(() => { config.data.dir = previousDir; fs.rmSync(dir, { recursive: true, force: true }); });
  const plans: Yue2SynthRequest[] = [], takes: Yue2SynthRequest[] = [];
  const results = new Map<string, Yue2TrackDetail[]>();
  t.mock.method(globalThis, 'fetch', async (input: string | URL | Request, init?: RequestInit) => {
    const url = new URL(String(input));
    if (url.pathname === '/yue2/props') return Response.json(behaviour.props ?? { plan_continue: true });
    if (url.pathname === '/yue2/synth') {
      const req = JSON.parse(String(init?.body)) as Yue2SynthRequest;
      const id = String(plans.length + takes.length + 1);
      if (req.plan_only) {
        const attempt = plans.filter(p => p.songs?.[0]?.abc === req.songs?.[0]?.abc).length;
        plans.push(req);
        results.set(id, req.songs!.map((song, i) => {
          const label = labelOf(song.abc!), draw = behaviour.draw?.(label, song.seed as number, attempt) ?? 'ok';
          return { song: i, seed: song.seed, noise_seed: song.seed, variation: 0,
            abc: draw === 'empty' ? '' : song.abc + section(label, song.seed as number),
            stage_end_reasons: { plan: draw === 'limit_hit' ? 'limit_hit' : 'section_end' } } as unknown as Yue2TrackDetail;
        }));
      } else {
        takes.push(req);
        const seeds = req.songs?.map(s => s.seed!) ?? [req.seed!];
        results.set(id, seeds.map((seed, song) => ({ song, seed, noise_seed: seed, variation: 0,
          semantic_ids: [...(req.codec_ids ?? []), ...Array<number>(400).fill(1)], frames: (req.codec_ids?.length ?? 0) + 400 }) as unknown as Yue2TrackDetail));
      }
      return Response.json({ id });
    }
    if (url.pathname === '/yue2/align') return Response.json({ words: [] });
    const tracks = results.get(url.searchParams.get('id')!)!;
    if (url.searchParams.has('result')) {
      if (tracks.length === 1) return new Response(new Uint8Array(pcm()), { headers: { 'content-type': 'audio/wav' } });
      const body = Buffer.concat(tracks.flatMap(() => [Buffer.from('--t\r\nContent-Type: audio/wav\r\n\r\n'), pcm(), Buffer.from('\r\n')])
        .concat(Buffer.from('--t--\r\n')));
      return new Response(new Uint8Array(body), { headers: { 'content-type': 'multipart/mixed; boundary=t' } });
    }
    return Response.json({ status: 'done', tracks });
  });
  const settle = async (id: string) => {
    const deadline = Date.now() + 5000;
    while (getBuildSession(id, 'user').pending.length) {
      assert.ok(Date.now() < deadline, 'finished within deadline');
      await new Promise(r => setTimeout(r, 5));
    }
    assert.equal(getBuildSession(id, 'user').error, null);
  };
  const start = () => createBuildSession({ lyrics }, { style: 'pop', cot: 'full', abc: score, lyrics }, 'user');
  const generate = async (id: string, to: number, count: number, newChords = true) => {
    const before = getBuildSession(id, 'user');
    generateBuildTakes(id, 'user', to, count, before.revision, undefined, undefined, newChords, () => undefined);
    await settle(id);
    return getBuildSession(id, 'user').takes.slice(before.takes.length);
  };
  return { plans, takes, start, generate };
}

test('new chords per take: each take plans its sections after the score so far, Keep writes them in, repeats follow', async t => {
  const { plans, takes, start, generate } = setup(t);
  const s = start();
  // Intro and verse: two sections, so two chained plan passes of two songs each.
  const first = await generate(s.id, 1, 2);
  assert.equal(plans.length, 2);
  assert.ok(plans.every(p => p.plan_only && p.plan_continue && p.plan_max_tokens === 1024 && p.songs!.length === 2));
  assert.deepEqual(plans.map(p => labelOf(p.songs![0].abc!)), ['intro', 'verse']);
  // The verse is planned after the take's own intro.
  for (const [i, take] of first.entries()) {
    assert.ok(plans[1].songs![i].abc!.includes(`"intro${take.seed}"`), 'the verse prefix holds this take\'s intro');
    assert.equal(take.planNote, undefined);
    assert.equal('abc' in take, false, 'the page does not get the score');
  }
  // Each take is composed under its own score: different chords per take.
  const sent = takes.at(-1)!.songs!.map(x => x.abc!);
  assert.notEqual(sent[0], sent[1]);
  assert.ok(sent[0].includes(`"verse${first[0].seed}"`) && !sent[0].includes(`"verse${first[1].seed}"`));
  // Nothing is written to the session until a take is kept.
  assert.equal(chordIn(score, 'verse'), undefined);
  keepBuildTake(s.id, 'user', first[0].id, 0);
  const kept = buildSessionRender(s.id, 'user').abc!;
  assert.equal(chordIn(kept, 'intro'), String(first[0].seed));
  assert.equal(chordIn(kept, 'verse'), String(first[0].seed));
  assert.ok(kept.includes('c16|c16|\nV: Ins\nZ2|\n% outro'), 'the sections not built are as they were');

  // The chorus is planned after the kept score.
  const [chorus] = await generate(s.id, 2, 1);
  assert.ok(plans.at(-1)!.songs![0].abc!.includes(`"verse${first[0].seed}"`));
  assert.equal(labelOf(plans.at(-1)!.songs![0].abc!), 'chorus');
  keepBuildTake(s.id, 'user', chorus.id, 0);
  // A second chorus follows the kept one: its lines are the same.
  const revision = getBuildSession(s.id, 'user').revision;
  editBuildPlan(s.id, 'user', revision, ['s2', 'o3'], true);
  const plan = getBuildSession(s.id, 'user').arrangement;
  assert.equal(plan.plan.length, 5);
  const render = buildSessionRender(s.id, 'user').abc!;
  assert.equal([...render.matchAll(new RegExp(`"chorus${chorus.seed}"`, 'g'))].length, 2, 'the kept chord is in each of the two choruses');
  // Undo of the chorus puts its old lines back and leaves the verse's.
  undoBuildKeep(s.id, 'user', 2, revision + 1);
  const undone = buildSessionRender(s.id, 'user').abc!;
  assert.equal(chordIn(undone, 'chorus'), undefined);
  assert.equal(chordIn(undone, 'verse'), String(first[0].seed));
});

test('a draw that fails the checks is redrawn once, and a take still without a plan follows the score and says so', async t => {
  const failing = setup(t, { draw: (_label, _seed, attempt) => (attempt === 0 ? 'limit_hit' : 'ok') });
  const a = failing.start();
  const [redrawn] = await failing.generate(a.id, 0, 1);
  assert.equal(failing.plans.length, 2, 'one more pass with new seeds');
  assert.notEqual(failing.plans[1].songs![0].seed, failing.plans[0].songs![0].seed);
  assert.equal(redrawn.planNote, undefined);
});

test('a take with no usable plan follows the score and says so; Keep leaves the score', async t => {
  const { plans, takes, start, generate } = setup(t, { draw: () => 'limit_hit' });
  const s = start();
  const [take] = await generate(s.id, 0, 1);
  assert.equal(plans.length, 2);
  assert.match(take.planNote!, /intro follows the score: it ran past the plan limit/);
  assert.equal(takes.at(-1)!.abc!.includes('% intro\nV: Vocal\nZ2|'), true, 'composed under the score as it was');
  keepBuildTake(s.id, 'user', take.id, 0);
  assert.equal(buildSessionRender(s.id, 'user').abc, score);
});

test('an engine that cannot plan from a kept score, and Follow the score, send no plan pass', async t => {
  const old = setup(t, { props: {} });
  const a = old.start();
  const [take] = await old.generate(a.id, 0, 1);
  assert.equal(old.plans.length, 0);
  assert.match(take.planNote!, /cannot plan/);
  const b = old.start();
  const [plain] = await old.generate(b.id, 0, 1, false);
  assert.equal(old.plans.length, 0);
  assert.equal(plain.planNote, undefined);
  assert.equal(old.takes.at(-1)!.abc, score);
});
