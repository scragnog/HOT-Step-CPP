import assert from 'node:assert/strict';
import test from 'node:test';
import { arbitrateSidecarGenre, assembleYue2FromFields, genreHint, instrumentalContradiction, scanGenreHints } from './mossCaption.js';

test('Local YuE2 fields assemble in order from labelled (Flamingo) answers, dropping parentheticals', () => {
  const raw = [
    'language: English',
    'genre: Lounge; Parody (as a cover of a famous song)',
    'vocal: Male baritone, clear and slightly nasal, delivered in a theatrical style. More detail here.',
    'instruments: Acoustic piano, upright bass, brushed drum kit. The mix is polished.',
    'mood: Humorous, self-deprecating, ironic. The arrangement keeps the feel.',
    'production: Clean modern lounge recording',
  ].join('\n');
  assert.equal(assembleYue2FromFields(raw),
    'English, Lounge / Parody, Male baritone, clear and slightly nasal, delivered in a theatrical style, '
    + 'Acoustic piano, upright bass, brushed drum kit, Humorous, self-deprecating, ironic, Clean modern lounge recording');
});

test('Local YuE2 fields accept six bare lines in order (MOSS)', () => {
  const raw = 'English\n1950s Rockabilly\nMale baritone\nPiano, upright bass\nSardonic, playful\nWarm analog mix';
  assert.equal(assembleYue2FromFields(raw),
    'English, 1950s Rockabilly, Male baritone, Piano, upright bass, Sardonic, playful, Warm analog mix');
});

test('Local YuE2 fields over the word cap end on a clean boundary, not a dangling word', () => {
  const out = assembleYue2FromFields([
    'language: English', 'genre: Lounge',
    'vocal: Male baritone, clean and slightly theatrical, delivered with a detached, almost spoken style',
    'instruments: Clean electric guitar, walking bass line, light drum kit, and a prominent vibraphone that carries the main melody',
    'mood: playful, ironic, slightly melancholic',
    'production: polished studio recording with clear separation of parts, modest compression on drums for a tight feel',
  ].join('\n'));
  assert.doesNotMatch(out, /\b(?:with|for|a|and|the)\s*,/i);
  assert.match(out, /^English, Lounge, Male baritone/);
  assert.match(out, /walking bass line, light drum kit, /);
  assert.match(out, /polished studio recording with clear separation of parts$/);
});

test('Genre hints: lounge beats the song it covers, coffee-house is not House', () => {
  assert.equal(scanGenreHints('A laid-back lounge cover of a punk rock anthem'), 'Lounge');
  assert.equal(scanGenreHints('soft piano and a coffee‑house atmosphere'), undefined);
  assert.equal(scanGenreHints('a deep house groove'), 'Deep House');
  assert.equal(arbitrateSidecarGenre({ providerGenre: 'Pop', providerCaption: 'a melancholy ballad', existing: 'Lounge; Parody' }), 'Lounge');
});

test('genreHint is empty without a tag and single-line with one', () => {
  assert.equal(genreHint(''), '');
  assert.match(genreHint('Lounge;\tParody\n'), /^The file is tagged with the genre "Lounge; Parody"\./);
  assert.doesNotMatch(genreHint('Rock'), /cover|original/);
});

test('instrumentalContradiction flags a sung track captioned instrumental', () => {
  const sung = 'language: English\nvocal: Male, mid-range, shouted.  \ninstruments: guitars';
  const flagged = 'This track is an instrumental rock piece. No vocals are present; a clean guitar carries the lead.';
  assert.equal(instrumentalContradiction(flagged, sung), 'Male, mid-range, shouted');
  // A caption that describes the voice is fine, even with an instrumental break.
  assert.equal(instrumentalContradiction('Punk rock with a shouted male vocal and an instrumental bridge.', sung), null);
  // No contradiction when YuE2 heard no singer, or gave no answer.
  assert.equal(instrumentalContradiction(flagged, 'vocal: instrumental, lead guitar'), null);
  assert.equal(instrumentalContradiction(flagged, undefined), null);
});
