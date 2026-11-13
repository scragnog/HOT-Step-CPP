// Mm3LadderRows.test.ts — the MM3 ladder's rungs and their actions.
//   (cd server && node --import tsx --test ../ui/src/components/training-studio/Mm3LadderRows.test.ts)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import type { Mm3Ladder, Mm3LadderPreview } from '../../services/trainingApi';
import { Mm3LadderRows, type Mm3ReviewState } from './Mm3LadderRows';

await i18n.use(initReactI18next).init({ lng: 'en', fallbackLng: 'en', resources: {}, interpolation: { escapeValue: false } });

const take = (o: Partial<Mm3LadderPreview> = {}): Mm3LadderPreview => ({
  id: 'take', step: 100, totalSteps: 300, kind: 'artist', base: false, file: 'step-000100-artist.wav',
  seconds: 40, seed: 424242, caption: 'A held-out caption', bytes: 100, ms: 16000, ts: 1,
  audioUrl: '/api/training/mm3/preview?run=r&file=step-000100-artist.wav', ...o,
});
const ladder = (o: Partial<Mm3Ladder> = {}): Mm3Ladder => ({
  dataset: { id: 'd', revision: '1' }, datasetName: 'Album', run: 'r', running: false, createdAt: 1, origin: null, reviewComplete: false, inUse: null, blindLabels: { 100: 'B', 200: 'A' }, scores: [], album: null,
  checkpoints: [{ step: 100, ref: 'r\\ckpt-100\\adapter_model.safetensors', loss: 2.5 }, { step: 200, ref: 'r\\ckpt-200\\adapter_model.safetensors' }],
  previews: [take(), take({ id: 'base', base: true, step: 0, file: 'base-artist.wav' }), take({ id: 'gone', step: 50 })], ...o,
});
const review = (o: Partial<Mm3ReviewState> = {}): Mm3ReviewState => ({ blind: false, scores: {}, notes: {}, onNote: () => {}, onScore: () => {}, idPrefix: 't', ...o });
const render = (l: Mm3Ladder, r = review(), o: { rendering?: number | null; renderError?: { step: number; message: string }; pickError?: { step: number; message: string } } = {}) =>
  renderToStaticMarkup(createElement(Mm3LadderRows, { ladder: l, review: r, rendering: o.rendering ?? null, renderError: o.renderError, pickError: o.pickError, picking: false, onRender: () => {}, onUse: () => {} }));

// Base first, then every checkpoint, plus a step whose checkpoint is gone
// (listen only: it has no Use button).
test('rungs draw takes, Render and Use; the in-use step says so, either slash', () => {
  const html = render(ladder({ inUse: 'r/ckpt-200/adapter_model.safetensors' }));
  assert.match(html, /Base[\s\S]*Step 50[\s\S]*Step 100[\s\S]*Loss 2\.500[\s\S]*Step 200/);
  assert.match(html, /40 s · seed 424242 · rendered in 16 s/);
  assert.equal(html.match(/Use this checkpoint/g)?.length, 1);
  assert.equal(html.match(/In use/g)?.length, 1);
  assert.equal(html.match(/Render more/g)?.length, 3);
  assert.match(html, />Render</);
});

test('a running run cannot render or pick, and an empty ladder draws nothing', () => {
  const html = render(ladder({ running: true }));
  assert.equal(html.match(/disabled=""/g)?.length, 6);
  assert.equal(render(ladder({ checkpoints: [], previews: [] })), '');
});

// The YuE2 review on MM3 rungs: blind letters in letter order with the loss
// hidden, scores per checkpoint, the best rung, and takes cleanup pruned.
test('blind rungs sort by letter and hide the loss; scored rungs show overall and the best', () => {
  const l = ladder({ scores: [
    { run: 'r', step: 100, checkpointDir: 'c1', loss: null, likeness: 5, corruption: 2, notes: 'nice', blind: true, blindLabel: 'B', updatedAt: '' },
    { run: 'r', step: 200, checkpointDir: 'c2', loss: null, likeness: 3, corruption: 3, notes: '', blind: true, blindLabel: 'A', updatedAt: '' }] });
  const scores = Object.fromEntries(l.scores.map(s => [s.step, s]));
  const blind = render(l, review({ blind: true, scores }));
  assert.match(blind, /Base[\s\S]*Rung A[\s\S]*Rung B[\s\S]*Step 50/);
  assert.doesNotMatch(blind, /Loss 2\.500|Step 100|Step 200/);
  assert.match(blind, /overall 4\.50[\s\S]*best by score|best by score[\s\S]*overall 4\.50/);
  assert.equal(blind.match(/Likeness/g)?.length, 2);
  assert.match(blind, /value="nice"/);
  assert.match(render(l), /Step 100[\s\S]*Loss 2\.500/);
});

test('the fixed-seed take comes first; a Render more take and a pruned take say what they are', () => {
  const html = render(ladder({ previews: [
    take({ id: 'extra', file: 'step-000100-artist-s9.wav', seed: 9, ts: 0 }), take({ ts: 5 }),
    take({ id: 'gone', step: 200, file: 'step-000200-artist.wav', audioUrl: '', pruned: true })] }));
  assert.match(html, /seed 424242[^<]*same seed on every rung[\s\S]*seed 9[^<]*random seed/);
  assert.match(html, /Take 1: audio removed by cleanup/);
});

test('takes download as WAV under the dataset, rung and seed; a pending or failed render shows on its rung', () => {
  assert.match(render(ladder()), /download="Album_step100_take1_seed424242\.wav"/);
  assert.match(render(ladder(), review({ blind: true })), /download="Album_rungB_take1_seed424242\.wav"/);
  assert.match(render(ladder(), review(), { rendering: 200 }), /Step 200[\s\S]*Take 1: pending/);
  assert.match(render(ladder(), review(), { renderError: { step: 100, message: 'engine down' } }), /Take 2: failed — engine down/);
});

test('a failed pick appears only beside the rung that was clicked', () => {
  const html = render(ladder(), review(), { pickError: { step: 100, message: 'Checkpoint changed; reload the ladder' } });
  assert.match(html, /Step 100[\s\S]*role="alert"[^>]*>Checkpoint changed; reload the ladder[\s\S]*Step 200/);
  assert.equal(html.match(/Checkpoint changed; reload the ladder/g)?.length, 1);
});
