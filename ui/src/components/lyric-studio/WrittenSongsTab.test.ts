// WrittenSongsTab.test.ts — provider-fallback guard for issue b676d752.
// "Generate Lyrics" read generationModel.provider straight off the lifted
// selection, which is still '' until the picker's own provider fetch
// resolves — a fast first-load click sent an empty provider the server
// correctly rejected. resolveGenerationProvider fills it from the same
// cache the picker itself falls back to. Run with the server's tsx:
//   (cd server && node --import tsx --test ../ui/src/components/lyric-studio/WrittenSongsTab.test.ts)
import assert from 'node:assert/strict';
import test from 'node:test';
import { resolveGenerationProvider } from './WrittenSongsTab.tsx';

const providers = [
  { id: 'gemini', default_model: 'gemini-2.5-flash' },
  { id: 'openai', default_model: 'gpt-5' },
];

test('leaves an already-wired selection alone', () => {
  const model = { provider: 'openai', model: 'gpt-5-mini' };
  assert.deepEqual(resolveGenerationProvider(model, providers), model);
});

test('falls back to the first cached provider when empty', () => {
  const resolved = resolveGenerationProvider({ provider: '', model: '' }, providers);
  assert.deepEqual(resolved, { provider: 'gemini', model: 'gemini-2.5-flash' });
});

test('keeps an explicit model when only the provider was empty', () => {
  const resolved = resolveGenerationProvider({ provider: '', model: 'gpt-5-mini' }, providers);
  assert.deepEqual(resolved, { provider: 'gemini', model: 'gpt-5-mini' });
});

test('stays empty when no providers are cached either', () => {
  const model = { provider: '', model: '' };
  assert.deepEqual(resolveGenerationProvider(model, []), model);
});
