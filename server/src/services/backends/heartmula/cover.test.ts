import test from 'node:test';
import assert from 'node:assert/strict';
import { heartmulaCoverLyrics, heartmulaCoverTags } from './cover.js';

test('four explicit tag fields have fixed order and cannot inject another field', () => {
  assert.equal(heartmulaCoverTags({ topic: ' Sky ', genre: 'ROCK', instrument: '', mood: 'Warm' }),
    'topic:[sky]; genre:[rock]; instrument:[]; mood:[warm]');
  assert.throws(() => heartmulaCoverTags(undefined), /four/);
  for (const topic of ['x]; mood:[hidden', 'a\nb']) {
    assert.throws(() => heartmulaCoverTags({ topic, genre: '', instrument: '', mood: '' }), /Invalid/);
  }
});

test('lyrics preserve supported sections, normalize numbered headings and reject unknown markers', () => {
  assert.equal(heartmulaCoverLyrics('[intro]\n[Verse 2]\nline\n[chorus]\n[Bridge]\n[OUTRO]'),
    '[Intro]\n[Verse]\nline\n[Chorus]\n[Bridge]\n[Outro]');
  assert.equal(heartmulaCoverLyrics('line'), '[Verse]\nline');
  assert.equal(heartmulaCoverLyrics(''), '[Intro]');
  assert.throws(() => heartmulaCoverLyrics('[Solo]\nline'), /Unsupported/);
  assert.throws(() => heartmulaCoverLyrics('x'.repeat(32768)), /32768/);
});
