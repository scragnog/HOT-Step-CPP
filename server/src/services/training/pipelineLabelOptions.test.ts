import test from 'node:test';
import assert from 'node:assert/strict';
import { parsePipelineLabelOptions } from './pipelineLabelOptions.js';

test('keeps every label-panel option the batch wizard sends', () => {
  const opts = {
    scope: 'all', useEssentia: true, tempoSource: 'beat_this', keySource: 'sheetsage',
    essentiaTempo: { minTempo: 70, maxTempo: 160 }, useGenius: false, localLyrics: true,
    useCaption: true, caption: { provider: 'flamingo' }, mergePolicy: 'overwrite_caption',
  };
  assert.deepEqual(parsePipelineLabelOptions(opts), opts);
});

test('drops malformed fields and returns undefined when nothing is left', () => {
  assert.equal(parsePipelineLabelOptions(undefined), undefined);
  assert.equal(parsePipelineLabelOptions({
    scope: 'selected', tempoSource: 'x', keySource: 1, mergePolicy: 'nope',
    localLyrics: 'yes', caption: { provider: '' }, essentiaTempo: {},
  }), undefined);
});

test('refuses out-of-range tempo bounds', () => {
  assert.throws(() => parsePipelineLabelOptions({ essentiaTempo: { minTempo: 10 } }), /minTempo/);
});
