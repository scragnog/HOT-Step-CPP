import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { Yue2SplitChart } from './Yue2SplitChart';
import { TrainingChart } from './TrainingChart';

await i18n.use(initReactI18next).init({ lng: 'en', fallbackLng: 'en', resources: {}, interpolation: { escapeValue: false } });

test('a single reconstruction sample draws a dot and old AR fields remain missing', () => {
  const html = renderToStaticMarkup(createElement(Yue2SplitChart, { points: [{ step: 20, ep: 20, loss: 9, narRecon: .4 }] }));
  assert.match(html, /Reconstruction/);
  assert.match(html, /w-\[3px\]/);
  assert.match(html, /Not recorded for this run/);
  assert.doesNotMatch(html, /epoch|Composite loss/);
});

test('standalone panels identify the untrained stage and token-grammar validation', () => {
  const points = [{ step: 1, ep: 1, loss: 2, gradNorm: 1 }];
  const ar = renderToStaticMarkup(createElement(Yue2SplitChart, { mode: 'ar', points, evals: [{ ep: 1, loss: 3 }] }));
  assert.match(ar, /Stage 2 not trained in this run/);
  assert.match(ar, /Token-grammar check/);
  const nar = renderToStaticMarkup(createElement(Yue2SplitChart, { mode: 'nar', points }));
  assert.match(nar, /Stage 1 not trained in this run/);
  assert.match(nar, /Decoder loss/);
});

test('Refine draws its first stop point at five readings and shared grad appears once', () => {
  const points = Array.from({ length: 5 }, (_, i) => ({ step: i + 1, ep: i + 1, loss: 9, arCe: 2, arKl: i / 10, narMse: 3, gradNorm: 4 }));
  const html = renderToStaticMarkup(createElement(Yue2SplitChart, { points, options: { unfreezePlanner: true, targetKlMode: 'trend', stopMode: 'kl', targetKl: 1 } }));
  assert.match(html, /KL trend \(30 readings\)/);
  assert.equal(html.match(/Grad norm/g)?.length, 1);
  assert.doesNotMatch(html, /Cursor CE|Drift|Composite loss/);
});

test('composite target is a separate readout; planner freeze marks the AR endpoint', () => {
  const html = renderToStaticMarkup(createElement(Yue2SplitChart, { points: [
    { step: 1, ep: 1, loss: 9, ma20: 9, arCe: 2, arKl: .1, narMse: 3 },
    { step: 2, ep: 2, frozen: true, narMse: 2 },
  ], options: { stopMode: 'loss', targetLoss: 4 } }));
  assert.match(html, /Planner frozen at step 1/);
  assert.match(html, /Composite loss stop mean 9\.0000 \/ target 4/);
  assert.doesNotMatch(html, /stroke="#10b981"/);
});

test('configured charts ignore steps carrying only coordinates and elapsed time', () => {
  const props = { epochs: [], target: 0, config: { series: ['arCe'] }, steps: [
    { step: 1, ep: 1, loss: NaN, arCe: 2 },
  ] };
  const expected = renderToStaticMarkup(createElement(TrainingChart, props));
  const actual = renderToStaticMarkup(createElement(TrainingChart, { ...props, steps: [
    ...props.steps, { step: 100, ep: 100, loss: NaN, elapsedMs: 1000 },
  ] }));
  assert.equal(actual, expected);
});

test('unconfigured shared chart retains the existing loss and learning-rate defaults', () => {
  const html = renderToStaticMarkup(createElement(TrainingChart, { epochs: [], target: 0, steps: [
    { step: 1, ep: 1, loss: 2, lr: .01 }, { step: 2, ep: 2, loss: 1, lr: .02 },
  ] }));
  assert.match(html, /learning rate/);
  assert.doesNotMatch(html, /More|AR CE|Cursor CE/);
});
