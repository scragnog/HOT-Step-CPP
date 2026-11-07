// Mm3LadderRows.test.ts — the MM3 ladder's rungs and their actions.
//   (cd server && node --import tsx --test ../ui/src/components/training-studio/Mm3LadderRows.test.ts)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import type { Mm3Ladder, Mm3LadderPreview } from '../../services/trainingApi';
import { Mm3LadderRows } from './Mm3LadderRows';

await i18n.use(initReactI18next).init({ lng: 'en', fallbackLng: 'en', resources: {}, interpolation: { escapeValue: false } });

const take = (o: Partial<Mm3LadderPreview> = {}): Mm3LadderPreview => ({
  id: 'take', step: 100, totalSteps: 300, kind: 'artist', base: false, file: 'step-000100-artist.wav',
  seconds: 40, seed: 424242, caption: 'A held-out caption', bytes: 100, ms: 16000, ts: 1,
  audioUrl: '/api/training/mm3/preview?run=r&file=step-000100-artist.wav', ...o,
});
const ladder = (o: Partial<Mm3Ladder> = {}): Mm3Ladder => ({
  dataset: { id: 'd', revision: '1' }, run: 'r', running: false, inUse: null,
  checkpoints: [{ step: 100, ref: 'r\\ckpt-100\\adapter_model.safetensors', loss: 2.5 }, { step: 200, ref: 'r\\ckpt-200\\adapter_model.safetensors' }],
  previews: [take(), take({ id: 'base', base: true, step: 0, file: 'base-artist.wav' }), take({ id: 'gone', step: 50 })], ...o,
});
const render = (l: Mm3Ladder) => renderToStaticMarkup(createElement(Mm3LadderRows, { ladder: l, rendering: null, picking: false, onRender: () => {}, onUse: () => {} }));

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
