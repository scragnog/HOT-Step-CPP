import test from 'node:test';
import assert from 'node:assert/strict';
import { buildProfile, recalculateProfileStats } from './profilerService.js';
import { computeAlbumEnrichment } from './prompts.js';
import { getProvider } from './llm/registry.js';

test('instrumental profile refuses before provider lookup or LLM calls', async () => {
  await assert.rejects(buildProfile('Fixture', null,
    [{ title: 'Quiet', lyrics: 'stale words', instrumental: true }], 'provider-must-not-be-called'),
  /no lyrics to profile: this album is instrumental/);
});

test('mixed lyric statistics equal vocal-only statistics and empty stats stay finite', () => {
  const vocal = { title: 'Voice', lyrics: '[Verse]\nThe road is long\nWe walk along\n[Chorus]\nThe lights return\nWe watch them burn' };
  const instrumental = { title: 'Quiet', instrumental: true, lyrics: 'excludedword '.repeat(100), duration: 180 };
  assert.deepEqual(recalculateProfileStats([vocal, instrumental], {}), recalculateProfileStats([vocal], {}));
  const empty = recalculateProfileStats([instrumental], {});
  assert.equal(empty.vocabulary_stats.total_words, 0);
  const check = (value: unknown): void => {
    if (typeof value === 'number') assert.ok(Number.isFinite(value));
    else if (value && typeof value === 'object') Object.values(value).forEach(check);
  };
  check(empty);
  const enrichment = computeAlbumEnrichment(Array.from({ length: 4 }, () => ({ ...instrumental, bpm: 80 })));
  assert.equal(enrichment?.wordsPerSec, 0);
});

test('mixed profile sends only vocal songs to the LLM and matches a vocal-only profile', async ctx => {
  const prompts: string[] = [];
  ctx.mock.method(getProvider('ollama'), 'call', async (_system: string, prompt: string) => {
    prompts.push(prompt);
    return '{}';
  });
  const vocal = { title: 'Voice', lyrics: '[Verse]\nThe road is long\nWe walk along' };
  const inst = { title: 'DO_NOT_PROFILE', lyrics: 'excludedword', instrumental: true };
  const mixed = await buildProfile('Fixture', null, [vocal, inst], 'ollama', 'fixture');
  const only = await buildProfile('Fixture', null, [vocal], 'ollama', 'fixture');
  assert.deepEqual(mixed, only);
  assert.equal(prompts.length, 10);
  assert.ok(prompts.every(prompt => !prompt.includes('DO_NOT_PROFILE') && !prompt.includes('excludedword')));
});
