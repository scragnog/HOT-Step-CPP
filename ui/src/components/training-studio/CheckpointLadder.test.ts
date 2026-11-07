// CheckpointLadder.test.ts — the shared ladder: slots, in-use, listen-only rungs.
//   (cd server && node --import tsx --test ../ui/src/components/training-studio/CheckpointLadder.test.ts)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { CheckpointLadder, type CheckpointLadderRung } from './CheckpointLadder';

await i18n.use(initReactI18next).init({ lng: 'en', fallbackLng: 'en', resources: {}, interpolation: { escapeValue: false } });

const rung = (step: number, o: Partial<CheckpointLadderRung> = {}): CheckpointLadderRung => ({
  key: step, step, title: `Step ${step}`, takes: [], inUse: false, canRender: true, onUse: () => {}, ...o });
const html = (rungs: CheckpointLadderRung[], o: Record<string, unknown> = {}) => renderToStaticMarkup(createElement(CheckpointLadder,
  { rungs, rendering: null, onRender: () => {}, picking: false, useLabel: 'Use this rung', ...o }));

test('engine slots render: header above the rungs, badges and children inside them', () => {
  const out = html([rung(1, { badges: createElement('b', null, 'KL 0.10'), children: createElement('i', null, 'scores') })],
    { header: createElement('u', null, 'album score') });
  assert.match(out, /<u>album score<\/u>[\s\S]*Step 1[\s\S]*<b>KL 0\.10<\/b>[\s\S]*<i>scores<\/i>/);
});

test('in use overrides the rung tint; a rung without onUse or weights is listen only; locked turns all off', () => {
  const out = html([rung(1, { inUse: true, className: 'border-sky-500/70' }), rung(2, { onUse: undefined, canRender: false })]);
  assert.match(out, /border-emerald-500\/60[^"]*"[\s\S]*In use/);
  assert.doesNotMatch(out, /border-sky-500\/70/);
  assert.equal(out.match(/Use this rung/g)?.length ?? 0, 0);
  assert.equal(out.match(/disabled=""/g)?.length, 1);
  assert.equal(html([rung(1), rung(2)], { locked: true }).match(/disabled=""/g)?.length, 4);
  assert.equal(html([]), '');
});
