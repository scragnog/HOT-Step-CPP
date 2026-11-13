import assert from 'node:assert/strict';
import test from 'node:test';
import { buildFactGroups } from './songFacts';
import type { Song } from '../../types';

test('Cover Studio facts retain source identity, choices and covering tag', () => {
  const groups = buildFactGroups({ backend: 'yue2', generationParams: {
    artistName: 'iteration 0.9', yue2Cover: { sourceId: '/references/source.wav', sourceLabel: 'source.wav',
      voices: 'both', keepChords: false, tempo: 90, key: 'Dm', keyShift: -2, cfgScale: 0 },
  } } as unknown as Song);
  const facts = groups.find(g => g.title === 'Cover Studio')?.facts;
  assert.ok(facts, 'cover provenance must be visible');
  const values = Object.fromEntries(facts.map(f => [f.label, f.value]));
  assert.equal(values['Source'], 'source.wav');
  assert.equal(values['Source ID'], '/references/source.wav');
  assert.equal(values['Covering band'], 'iteration 0.9');
  assert.equal(values['Keep chords'], 'Off');
  assert.equal(values['Tempo'], '90 bpm');
  assert.equal(values['Key shift'], '-2 semitones');
  assert.equal(values['Cover CFG'], '0');
});

test('MusicXML provenance does not invent source audio; ordinary tracks have no cover group', () => {
  const facts = buildFactGroups({ backend: 'yue2', generation_params: {
    yue2Cover: { sourceKind: 'musicxml', sourceLabel: 'score.musicxml' },
  } } as Song).find(g => g.title === 'Cover Studio')!.facts;
  assert.equal(facts.find(f => f.label === 'Source type')?.value, 'MusicXML');
  assert.equal(facts.find(f => f.label === 'Source ID'), undefined);
  assert.equal(buildFactGroups({ generationParams: { seed: 1 } } as Song).some(g => g.title === 'Cover Studio'), false);
  assert.equal(buildFactGroups({ backend: 'ace', generationParams: { yue2Cover: { sourceId: 'stale' } } } as Song)
    .some(g => g.title === 'Cover Studio'), false);
});

test('ACE cover facts use its own source and preserve zero strengths', () => {
  const facts = buildFactGroups({ backend: 'ace', generationParams: {
    source: 'cover-studio', sourceAudioUrl: '/references/recording.wav', audioCoverStrength: 0,
    coverNoiseStrength: 0, yue2Cover: { sourceId: 'stale' },
  } } as Song).find(g => g.title === 'Cover Studio')!.facts;
  assert.equal(facts.find(f => f.label === 'Source ID')?.value, '/references/recording.wav');
  assert.equal(facts.find(f => f.label === 'Structure fidelity')?.value, '0');
  assert.equal(facts.find(f => f.label === 'Source preservation')?.value, '0');
});

function planStage(yue2: Record<string, unknown>, yue2Request?: Record<string, unknown>): string | undefined {
  const song = { backend: 'yue2', generationParams: { backend: 'yue2', yue2, yue2Request } } as unknown as Song;
  const outcome = buildFactGroups(song).find(g => g.title === 'Outcome');
  return outcome?.facts.find(f => f.label === 'Plan stage')?.value;
}

test('an app-planned score does not read as a plan stage that never ran (#271)', () => {
  const stage_end_reasons = { plan: 'skipped', semantic: 'eos' };
  assert.equal(planStage({ abc_supplied: false, abc_planned: true, stage_end_reasons }), 'Planned by the app');
  assert.equal(planStage({ abc_supplied: true, stage_end_reasons }), undefined, 'a supplied score drops the row');
  assert.equal(planStage({ abc_supplied: false, abc_planned: true, stage_end_reasons }, { abc: 'X:1', cot: 'full' }), 'Planned by the app',
    'the persisted request carries the planned abc; the flag still wins');
  assert.equal(planStage({ abc_supplied: false, stage_end_reasons }), 'Not run', 'cot off still reads Not run');
});


test('MM3 facts show the launder recipe including a zero seed', () => {
  const groups = buildFactGroups({ backend: 'minimax-m3', generationParams: {
    mm3PlankPath: 'source.mm3plank', mm3PlankSemanticOnly: true,
    mm3PlankLaundered: true, mm3PlankLaunderSteps: 40,
    mm3PlankLaunderSeed: 0, mm3PlankLaunderCaption: 'acoustic guitar and vocals',
  } } as unknown as Song);
  const facts = groups.flatMap(g => g.facts);
  assert.equal(facts.find(f => f.label === 'Launder steps')?.value, '40');
  assert.equal(facts.find(f => f.label === 'Launder seed')?.value, '0');
  assert.equal(facts.find(f => f.label === 'Source caption')?.value, 'acoustic guitar and vocals');
});
