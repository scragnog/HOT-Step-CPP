import assert from 'node:assert/strict';
import test from 'node:test';
import { alignerLyrics, pronounceWord } from './lyricsPronunciation.js';

test('an acronym in the table is spelled out the way it is sung', () => {
  assert.deepEqual(pronounceWord('AGI'), ['ay', 'gee', 'eye']);
  assert.deepEqual(pronounceWord('NVDA'), ['en', 'vee', 'dee', 'ay']);
  assert.deepEqual(pronounceWord('RLHF'), ['are', 'el', 'aitch', 'eff']);
});

test('capitals with no vowel to hold are spelled out, capitals with one are sung', () => {
  // XML has nothing to sing; NASA does, so spelling it out would be the error.
  assert.deepEqual(pronounceWord('XML'), ['ex', 'em', 'el']);
  assert.deepEqual(pronounceWord('NASA'), ['nasa']);
  assert.deepEqual(pronounceWord('FOOM'), ['foom']);
});

test('a number becomes the words a singer uses for it', () => {
  assert.equal(pronounceWord('2024').join(' '), 'twenty twenty four');
  assert.equal(pronounceWord('7').join(' '), 'seven');
  assert.equal(pronounceWord('42').join(' '), 'forty two');
  // A number nobody sings as a number is read digit by digit.
  assert.deepEqual(pronounceWord('12345'), ['one', 'two', 'three', 'four', 'five']);
});

test('ordinary words survive as themselves', () => {
  assert.deepEqual(pronounceWord('Hello!'), ['hello']);
  assert.deepEqual(pronounceWord("don't"), ["don't"]);
  assert.deepEqual(pronounceWord('pre-training'), ['pre', 'training']);
  assert.deepEqual(pronounceWord('“Hold'), ['hold']);
});

test('a token with no letters has nothing to align, and says so', () => {
  assert.deepEqual(pronounceWord('—'), []);
  assert.deepEqual(pronounceWord('♪'), []);
  assert.deepEqual(pronounceWord('...'), []);
});

test('section labels are carried through and are not words', () => {
  const plan = alignerLyrics('[Verse 1]\nhold the line\n\n[Chorus]\nlet it go\n');
  assert.equal(plan.text, '[Verse 1]\nhold the line\n[Chorus]\nlet it go');
  // Six words, none of them a label: the text has the labels, the map does not.
  assert.equal(plan.words.length, 6);
  assert.equal(plan.units.length, 6);
});

test('every word keeps its own span in the lyrics that were handed in', () => {
  const lyrics = '[Verse]\nyou are an AGI now';
  const plan = alignerLyrics(lyrics);
  const chars = Array.from(lyrics);

  const agi = plan.words.find(word => chars.slice(word.start, word.end).join('') === 'AGI')!;
  assert.ok(agi, 'the acronym is located');
  // Three units — "ay gee eye" — for the one word the sidecar stores.
  assert.equal(agi.lastUnit - agi.firstUnit + 1, 3);
  assert.deepEqual(
    plan.units.slice(agi.firstUnit, agi.lastUnit + 1).map(unit => unit.text),
    ['ay', 'gee', 'eye'],
  );
  // The unit spans land on the aligner text they were cut from.
  for (const unit of plan.units) {
    assert.equal(plan.text.slice(unit.start, unit.end), unit.text);
  }
});

test('a lyric of ordinary words is unchanged apart from case and punctuation', () => {
  const plan = alignerLyrics('Hold the line,\nlet it go!');
  assert.equal(plan.text, 'hold the line\nlet it go');
  assert.equal(plan.expanded, 0);
});

test('only the words that were respelled are counted', () => {
  const plan = alignerLyrics('an AGI in 2024');
  assert.equal(plan.expanded, 2);
});

test('a word with nothing to align is left out of the text, not timed by guess', () => {
  const plan = alignerLyrics('hold — the line');
  assert.equal(plan.text, 'hold the line');
  assert.equal(plan.words.length, 3);
});
