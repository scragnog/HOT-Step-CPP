import assert from 'node:assert/strict';
import test from 'node:test';
import { withSourceAttributes } from './mm3CaptionSource';

test('source facts replace only the attributes and keep genre and arrangement', () => {
  const original = 'Global Metadata\nBasic Attributes: bpm is 80. key is D, and scale is minor. Jazz.\nVocal Details\nWarm voice\nArrangement\nGuitar';
  assert.equal(withSourceAttributes(original, 123.6, 'F# Major', 'major'),
    original.replace('bpm is 80. key is D, and scale is minor.', 'bpm is 124. key is F#, and scale is major.'));
});

test('missing attributes insert under Global Metadata, including CRLF captions', () => {
  assert.equal(withSourceAttributes('Global Metadata\r\nVocal Details\r\nAiry', 90, 'Bb minor'),
    'Global Metadata\nBasic Attributes: bpm is 90. key is Bb, and scale is minor.\r\nVocal Details\r\nAiry');
  assert.equal(withSourceAttributes('Arrangement\nPiano', 100, 'C', 'major'),
    'Global Metadata\nBasic Attributes: bpm is 100. key is C, and scale is major.\nArrangement\nPiano');
});
