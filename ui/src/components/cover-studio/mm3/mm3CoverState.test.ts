import assert from 'node:assert/strict';
import test from 'node:test';
import { DEFAULT_MM3_COVER, mm3CoverInput, mm3CoverReady, restoreMm3Cover, mm3MatchBias, withMm3CoverMode } from './mm3CoverState';

test('old saved controls are ignored and cannot change semantic hard pinning', () => {
  const restored = restoreMm3Cover({ plank: 'a.mm3plank', plan: false, sound: true,
    semanticOnly: false, forceBias: '16', patternMode: 'prefix', launder: false });
  assert.deepEqual(restored, { ...DEFAULT_MM3_COVER, plank: 'a.mm3plank' });
  assert.deepEqual(mm3CoverInput(restored), { plank: 'a.mm3plank', semanticOnly: true, trimSeconds: null, arSeed: -1 });
  assert.equal(mm3CoverReady(DEFAULT_MM3_COVER), false);
  assert.equal(mm3CoverReady(restored), true);
  assert.deepEqual(restoreMm3Cover(null), DEFAULT_MM3_COVER);
});

test('Match maps to measured semantic bias and 100 omits it', () => {
  assert.equal(mm3MatchBias(50), 32);
  assert.equal(mm3MatchBias(33), 10);
  assert.equal(mm3MatchBias(25), 6);
  assert.equal(mm3MatchBias(0), 1);
  assert.equal('forceBias' in mm3CoverInput(DEFAULT_MM3_COVER), false);
  assert.equal(mm3CoverInput({ ...DEFAULT_MM3_COVER, match: 50 }).forceBias, 32);
  assert.equal(restoreMm3Cover({ match: 101 }).match, 100);
  assert.equal(restoreMm3Cover({ match: -2 }).match, 0);
  assert.equal(mm3CoverReady({ ...DEFAULT_MM3_COVER, plank: 'a.mm3plank', preparedSourceUrl: '/references/a.wav' }, '/references/b.wav'), false);
});

test('trim, planner seed and launder settings survive restore', () => {
  const restored = restoreMm3Cover({ launderSteps: 40, launderSeed: -1, launderCaption: 'acoustic guitar', trimSeconds: '12', arSeed: 9 });
  assert.equal(restored.launderSteps, 40);
  assert.equal(restored.launderSeed, -1);
  assert.equal(restored.launderCaption, 'acoustic guitar');
  assert.equal(mm3CoverInput(restored).trimSeconds, 12);
  assert.equal(mm3CoverInput(restored).arSeed, 9);
  assert.equal(mm3CoverInput({ ...restored, trimSeconds: 'bad' }).trimSeconds, null);
});

test('MM3 cover modes: a mode change applies only that mode\'s defaults, and old drafts open as Pin', () => {
  const prepared = { ...DEFAULT_MM3_COVER, plank: 'a.mm3plank', seed: 7, arSeed: 3, trimSeconds: '40', match: 60 };
  const follow = withMm3CoverMode({ ...prepared, strength: 0.4, stateFix: 'both' }, 'follow-depth');
  assert.equal(follow.strength, 0.9);
  assert.equal(follow.stateFix, 'depth');
  assert.equal(follow.seed, 7);
  assert.equal(follow.trimSeconds, '40');
  assert.equal(withMm3CoverMode(follow, 'pin').match, 100);
  assert.deepEqual(mm3CoverInput(follow), { plank: 'a.mm3plank', semanticOnly: true, trimSeconds: 40, arSeed: 3,
    mode: 'follow-depth', strength: 0.9, candidates: 32, stateFix: 'depth' });
  const blend = mm3CoverInput(withMm3CoverMode(follow, 'blend'));
  assert.equal(blend.blend, 25);
  assert.equal('stateFix' in blend, false);
  assert.deepEqual(mm3CoverInput(withMm3CoverMode(follow, 'product')), { plank: 'a.mm3plank', semanticOnly: true,
    trimSeconds: 40, arSeed: 3, mode: 'product', weight: 1, candidates: 32 });
  assert.equal(restoreMm3Cover({ plank: 'a.mm3plank' }).mode, 'pin');
  assert.equal(restoreMm3Cover({ mode: 'greedy', stateFix: 'x', candidates: 99, weight: 9 }).mode, 'pin');
  assert.equal(restoreMm3Cover({ stateFix: 'x' }).stateFix, 'off');
  assert.equal(restoreMm3Cover({ candidates: 99 }).candidates, 32);
  assert.equal(restoreMm3Cover({ mode: 'blend', blend: 40 }).blend, 40);
  assert.equal('mode' in mm3CoverInput(DEFAULT_MM3_COVER), false);
});
