import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { TrainingPreviewRung, TrainingPreviewSettings } from './TrainingPreviewPanel';

await i18n.use(initReactI18next).init({ lng: 'en', fallbackLng: 'en', resources: {}, interpolation: { escapeValue: false } });

const settings = (enabled: boolean) => renderToStaticMarkup(createElement(TrainingPreviewSettings, {
  title: 'Previews', enabled, onEnabledChange: () => {}, enableLabel: 'Render previews', enableInfo: 'info',
  lead: createElement('p', null, 'LEAD'), afterToggle: createElement('p', null, 'AFTER'),
}, createElement('i', null, 'FIELDS')));

test('settings draw the lead and after slots always and the fields only while enabled', () => {
  const off = settings(false);
  assert.match(off, /Previews/);
  assert.match(off, /LEAD[\s\S]*Render previews[\s\S]*AFTER/);
  assert.doesNotMatch(off, /FIELDS/);
  assert.match(settings(true), /FIELDS/);
});

test('a rung draws playable takes as players and the rest as their own text, with the caller slots', () => {
  const html = renderToStaticMarkup(createElement(TrainingPreviewRung, {
    id: 'rung-5', title: 'Step 5', badges: createElement('b', null, 'BADGE'), actions: createElement('button', null, 'ACTION'),
    takes: [
      { id: 'a', src: '/audio/a.wav', label: 'Take A', sublabel: '30 s' },
      { id: 'b', label: 'Take B', text: 'Take B: failed — boom' },
    ],
  }, createElement('div', null, 'SCORES')));
  assert.match(html, /id="rung-5"/);
  assert.match(html, /Step 5[\s\S]*BADGE[\s\S]*ACTION[\s\S]*SCORES/);
  assert.match(html, /<audio[^>]*src="\/audio\/a\.wav"/);
  assert.match(html, /Take B: failed — boom/);
  assert.equal(html.match(/<audio/g)?.length, 1);
});
