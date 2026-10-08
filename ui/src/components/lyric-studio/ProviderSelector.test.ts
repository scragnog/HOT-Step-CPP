// ProviderSelector.test.ts — selection-to-command capture for 94b61c2a.
// Picking a provider in AlbumHeader/ArtistPageSidebar used to update a local
// copy of ModelSelections that WrittenSongsTab/QueuePanel never saw, so the
// job they queued still carried whatever loadSelections() returned on
// LyricStudioV2's last unrelated render. Run with the server's tsx:
//   (cd server && node --import tsx --test ../ui/src/components/lyric-studio/ProviderSelector.test.ts)
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { updateModelSelection, type ModelSelections } from './ProviderSelector.tsx';

const here = dirname(fileURLToPath(import.meta.url));

function selections(): ModelSelections {
  return {
    profiling: { provider: 'openai', model: 'gpt-5' },
    generation: { provider: 'openai', model: 'gpt-5' },
    refinement: { provider: 'openai', model: 'gpt-5' },
    coverCaption: { provider: 'openai', model: 'gpt-5' },
  };
}

test('updating one role leaves the others untouched', () => {
  const before = selections();
  const after = updateModelSelection(before, 'generation', 'gemini', 'gemini-2.5-flash');
  assert.deepEqual(after.generation, { provider: 'gemini', model: 'gemini-2.5-flash' });
  assert.deepEqual(after.profiling, before.profiling);
  assert.deepEqual(after.refinement, before.refinement);
  assert.deepEqual(after.coverCaption, before.coverCaption);
  // Input untouched — downstream props can't alias a mutated object.
  assert.deepEqual(before, selections());
});

// Regression guard for the actual root cause: AlbumHeader and
// ArtistPageSidebar must take modelSelections as a prop from LyricStudioV2,
// not keep their own useState(loadSelections) copy — that duplication is
// exactly what let a pick in one panel never reach the queue in the other.
for (const file of ['AlbumHeader.tsx', 'ArtistPageSidebar.tsx']) {
  test(`${file} does not keep its own ModelSelections state`, () => {
    const source = readFileSync(join(here, file), 'utf8');
    assert.match(source, /modelSelections:\s*ModelSelections/, `${file} must accept modelSelections as a prop`);
    assert.doesNotMatch(source, /useState(?:<[^>]*>)?\(\s*loadSelections\s*\)/, `${file} must not load its own selections copy`);
  });
}
