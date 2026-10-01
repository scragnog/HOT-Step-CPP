import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveCoverScene } from './sceneLlm.js';
import { buildCoverArtPrompt, type CoverArtPromptOpts } from './promptBuilder.js';

function fakeProvider(reply: () => Promise<string>) {
  let calls = 0;
  let input: { instruction: string; context: string } | null = null;
  return {
    provider: {
      isAvailable: () => true,
      call: async (instruction: string, context: string) => {
        calls++;
        input = { instruction, context };
        return reply();
      },
    },
    get calls() { return calls; },
    get input() { return input; },
  };
}

test('manual prompt or subject skips the caption provider', async () => {
  const fake = fakeProvider(async () => 'a forest');
  for (const opts of [
    { prompt: 'a user scene' },
    { subject: 'a chosen subject' },
    { prompt: 'a user scene', subject: '   ' },
    { prompt: '   ', subject: 'a chosen subject' },
  ]) {
    assert.equal(await resolveCoverScene(opts, { provider: fake.provider }), null);
  }
  assert.equal(fake.calls, 0);
});

test('blank and whitespace-only inputs make one configured-provider call', async () => {
  for (const opts of [{}, { prompt: '   ', subject: '\t' }]) {
    const fake = fakeProvider(async () => '  a lantern glowing beside a river  ');
    const scene = await resolveCoverScene({
      ...opts, title: 'Night river', style: 'folk', lyrics: '[Verse]\nA lantern gleams\nThe river flows',
    }, { provider: fake.provider });
    assert.equal(scene, 'a lantern glowing beside a river');
    assert.equal(fake.calls, 1);
    assert.match(fake.input!.context, /Night river/);
    assert.match(fake.input!.context, /folk/);
    assert.match(fake.input!.context, /A lantern gleams/);
    assert.doesNotMatch(fake.input!.context, /\[Verse\]/);
  }
});

test('the scene is used as the built prompt subject', async () => {
  const opts: CoverArtPromptOpts = { style: 'folk', lyrics: 'the river flows' };
  const fake = fakeProvider(async () => 'a lantern glowing beside a river');
  const scene = await resolveCoverScene(opts, { provider: fake.provider });
  const prompt = buildCoverArtPrompt({ ...opts, subject: scene ?? opts.subject });
  assert.match(prompt, /^a lantern glowing beside a river,/);
  assert.doesNotMatch(prompt, /a scene evoking/);
});

test('empty, unavailable, thrown and timed-out providers use the keyword fallback', async () => {
  const opts: CoverArtPromptOpts = { lyrics: 'silver rivers silver moonlight' };
  const fallback = buildCoverArtPrompt(opts);
  const cases = [
    { provider: null },
    { provider: { isAvailable: () => false, call: async () => 'unused' } },
    { provider: fakeProvider(async () => '   ').provider },
    { provider: fakeProvider(async () => { throw new Error('offline'); }).provider },
    { provider: fakeProvider(() => new Promise<string>(() => {})).provider, timeoutMs: 5 },
  ];
  for (const deps of cases) {
    const scene = await resolveCoverScene(opts, deps);
    assert.equal(scene, null);
    assert.equal(buildCoverArtPrompt({ ...opts, subject: scene ?? opts.subject }), fallback);
  }
});

test('reply is limited to one trimmed line and 220 characters', async () => {
  const fake = fakeProvider(async () => `\n  ${'a'.repeat(260)}\nsecond scene`);
  const scene = await resolveCoverScene({}, { provider: fake.provider });
  assert.equal(scene, 'a'.repeat(220));
});

test('text-rendering words in a reply fall back to keywords', async () => {
  const fake = fakeProvider(async () => 'a poster with a title');
  assert.equal(await resolveCoverScene({}, { provider: fake.provider }), null);
});
