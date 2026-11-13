import assert from 'node:assert/strict';
import test from 'node:test';
import { applyFactSubstitution } from './mossCaption.js';

const FACTS = { bpm: 115, keyscale: 'D Major', genre: 'Lounge' };

// Shape of a Music Flamingo MM3 answer with the genre hint after the template.
const DRIFTED = [
  'Global Metadata',
  'Basic Elements: bpm is 60, key is D major, and scale is minor. Genre / Subgenre: Lounge-style cover.',
  'Global Emotional Progression: Relaxed throughout. Instrumental Scenarios & Imagery: A late-night bar.',
  'Sonics & Production Profile: Clean and spacious.',
  'Vocal Details',
  'Vocal Gender & Timbre: Male.',
  'Vocal Style: Crooned.',
  'Harmony/Backing Vocals: None.',
  'Vocal FX: Light reverb.',
  'Arrangement',
  'Instrumental Description (Primary/Secondary Layering):',
  'Primary: Piano.',
  'Secondary: Brushes.',
  'Groove & Foundation Progression: Steady swing.',
  'Embellishments & Textures & Spatial FX: Soft sweeps.',
].join('\n');

const WANT = [
  'Global Metadata',
  'Basic Attributes: bpm is 115. key is D, and scale is major. Lounge.',
  'Global Emotional Progression: Relaxed throughout.',
  'Application Scenarios & Imagery: A late-night bar.',
  'Sonics & Production Profile: Clean and spacious.',
  'Vocal Details',
  'Vocal Gender & Timbre: Male.',
  'Vocal Style: Crooned.',
  'Harmony/Backing Vocals: None.',
  'Vocal FX: Light reverb.',
  'Arrangement',
  'Instrument Lifecycle Description (Primary/Secondary Layering):',
  'Primary: Piano.',
  'Secondary: Brushes.',
  'Groove & Foundation Progression: Steady swing.',
  'Embellishments, Textures & Spatial FX: Soft sweeps.',
].join('\n');

test('MM3 near-miss labels are canonicalised and the drifted Basic line replaced, not doubled', () => {
  assert.equal(applyFactSubstitution(DRIFTED, FACTS), WANT);
});

test('MM3 repair is idempotent over an already-doubled caption', () => {
  // What the pre-fix pipeline wrote: a rebuilt line inserted above the drifted one.
  const doubled = DRIFTED.replace('Global Metadata\n', `Global Metadata\n${WANT.split('\n')[1]}\n`);
  assert.equal(applyFactSubstitution(doubled, FACTS), WANT);
  assert.equal(applyFactSubstitution(WANT, FACTS), WANT);
});

test('A section header welded onto its first field is split off', () => {
  const welded = DRIFTED.replace('Vocal Details\nVocal Gender', 'Vocal Details Vocal Gender');
  assert.equal(applyFactSubstitution(welded, FACTS), WANT);
});
