import test from 'node:test';
import assert from 'node:assert/strict';
import { abcCoverLeadSheet, tryAbcCoverLeadSheet, coverRenderRolls, coverScoreEvents, nativeCoverLeadSheet } from './coverLeadSheet.js';
import { abcLeadSheetEvents, leadSheetRoll } from './leadSheetRoll.js';

const abc = 'X:1\nM:4/4\nL:1/8\nQ:1/4=90\nK:C\nV:Vocal\n"C" C | D2 [Q:1/4=180] E2 F2 |';

test('MuScriptor preview can fail without discarding editable ABC or accepting it for render', () => {
  assert.equal(tryAbcCoverLeadSheet('X:1\nK:C\nC|', 'muscriptor'), undefined);
  assert.throws(() => coverRenderRolls({ assetId: 'a', sha256: 'h', approvedAbc: 'X:1\nK:C\nC|' }), /explicit Q: tempo/);
  assert.deepEqual(tryAbcCoverLeadSheet(abc, 'muscriptor'), abcCoverLeadSheet(abc, 'muscriptor'));
});

test('ABC timing retains pickups and tempo changes in seconds before one quantiser', () => {
  const sheet = abcCoverLeadSheet(abc, 'sheetsage2');
  assert.equal(sheet.timing.bpm, 90);
  assert.deepEqual(sheet.timing.rawEvents.melody, abcLeadSheetEvents(abc).melody);
  assert.ok(sheet.timing.rawEvents.melody[0].offset < 1);
  const { rolls, path } = coverRenderRolls({ assetId: 'a', sha256: 'h', approvedAbc: abc, leadSheet: sheet, leadSheetSource: 'sheetsage2' });
  assert.equal(path, 'abc');
  assert.deepEqual(rolls.pianoroll, Array.from(leadSheetRoll(abcLeadSheetEvents(abc)).pianoroll));
});

test('a sounding Vocal wins; silent or absent Vocal falls back to Ins', () => {
  const header = 'X:1\nM:4/4\nL:1/4\nQ:1/4=120\nK:C\n';
  assert.equal(coverScoreEvents(header + 'V:Vocal\nC4|\nV:Ins\nG4|').melody[0].pitch, 60);
  assert.equal(coverScoreEvents(header + 'V:Vocal\nz4|\nV:Ins\nG4|').melody[0].pitch, 67);
  assert.equal(coverScoreEvents(header + 'V:Ins\nG4|').melody[0].pitch, 67);
  assert.equal(coverScoreEvents(header + 'V:Other\nC4|\nV:Ins\nG4|').melody[0].pitch, 67);
});

test('MuScriptor chords are labelled estimates and imports keep their source', () => {
  assert.equal(abcCoverLeadSheet(abc, 'muscriptor').chordQuality, 'estimated');
  for (const source of ['dataset', 'musicxml'] as const) {
    const sheet = abcCoverLeadSheet(abc, 'muscriptor', source);
    assert.equal(sheet.scoreSource, source);
    assert.equal(sheet.chordQuality, 'provided');
  }
});

test('native rolls use raw seconds and whole-beat chord rounding, edits discard them', () => {
  const sheet = nativeCoverLeadSheet(abc, { path: 'native', conversionVersion: 1, bpm: 120,
    grid: { identity: 'grid-1', data: { beats: [0, 0.6, 1.1] } },
    rawEvents: { melody: [{ onset: 0.0625, offset: 0.1875, pitch: 72 }],
      chords: [{ onset: 0.25, offset: 0.75, pitch: 60 }] } });
  const draft = { assetId: 'a', sha256: 'h', approvedAbc: abc, leadSheet: sheet, leadSheetSource: 'yourmt3-chordnet' as const };
  const direct = coverRenderRolls(draft);
  assert.equal(direct.path, 'native');
  assert.equal(direct.rolls.pianoroll[256 + 72 * 2], 1, 'half sixteenth rounds up');
  assert.equal(direct.rolls.chord[0], 1, 'half beat rounds to even before expansion');
  assert.equal(direct.rolls.chord[7 * 12], 1);
  const edited = coverRenderRolls({ ...draft, approvedAbc: abc.replace('"C" C', '"C" D') });
  assert.equal(edited.path, 'abc-edited');
  assert.equal(edited.rolls.pianoroll[62 * 2], 1);
  assert.equal(edited.rolls.pianoroll[72 * 2], 0);
  const otherSource = coverRenderRolls({ ...draft, leadSheetSource: 'muscriptor' });
  assert.equal(otherSource.path, 'abc-edited');
});

test('native boundary rejects missing grid identity and invalid raw seconds', () => {
  assert.throws(() => nativeCoverLeadSheet(abc, { path: 'native', conversionVersion: 1, bpm: 120,
    rawEvents: { melody: [], chords: [] } }), /grid identity/);
  assert.throws(() => nativeCoverLeadSheet(abc, { path: 'native', conversionVersion: 1, bpm: 120,
    grid: { identity: 'a', data: {} }, rawEvents: { melody: [{ onset: -1, offset: 0, pitch: 60 }], chords: [] } }), /Invalid/);
});
