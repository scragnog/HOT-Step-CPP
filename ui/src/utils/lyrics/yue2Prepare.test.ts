import assert from 'node:assert/strict';
import test from 'node:test';
import {
  confirmLyrics, confirmStructure, createPrepare, editLyrics, editStructure, exportPrepare, migrateLegacyPrepare,
  structureConfirmed, structureHash, type PrepareV2,
} from './yue2Prepare';
// The server's own readers: its section lint and section bars.
import { lintScoreSections, scoreSections } from '../../../../server/src/services/backends/yue2/scoreSections';

// A verse transcribed as three fragments with junk labels, as reported.
const ABC = [
  'X:1', 'M:4/4', 'L:1/8', 'K:C',
  '% intro', 'V:Vocal', 'Z4|', 'V:Ins', '"C"C8|"F"F8|"G"G8|"C"C8|',
  '% verse', 'V:Vocal', 'C2D2E2F2|G8|', 'V:Ins', '"C"C8|"G"G8|',
  '% pre-chorus', 'V:Vocal', 'A8|', 'V:Ins', '"F"F8|',
  '% chorus', 'V:Vocal', 'G8|', 'V:Ins', '"C"C8|',
  '% chorus', 'V:Vocal', 'c8|B8|A8|G8|', 'V:Ins', '"F"F8|"G"G8|"A"A8|"C"C8|',
].join('\n');
const PASTE = '[Verse 1]\none two three\nfour five six\n\n[Chorus]\nseven eight\nnine ten\n';

const ok = (r: PrepareV2 | { error: string }): PrepareV2 => {
  if ('error' in r) assert.fail(r.error);
  return r;
};
const bars = (p: PrepareV2) => scoreSections(p.abc).map(s => `${s.label}@${s.startBar}`);

test('the reported loop: fix structure, confirm, paste fresh lyrics; the score never moves', () => {
  let p = createPrepare('src', ABC);
  // Merge the three fragments into one verse, then name the real chorus.
  p = ok(editStructure(p, { kind: 'merge', index: 1, keep: 'first' }));
  p = ok(editStructure(p, { kind: 'merge', index: 1, keep: 'first' }));
  assert.deepEqual(bars(p), ['intro@1', 'verse@5', 'chorus@9']);
  p = ok(confirmStructure(p));
  const hash = structureHash(p.abc), shape = bars(p);
  // Paste, assign, then paste again: only the map changes.
  p = ok(editLyrics(p, { kind: 'paste', paste: PASTE }));
  const [verse, chorus] = p.blocks;
  p = ok(editLyrics(p, { kind: 'instrumental', sectionId: p.sectionIds[0] }));
  p = ok(editLyrics(p, { kind: 'assign', blockId: verse.id, sectionId: p.sectionIds[1] }));
  p = ok(editLyrics(p, { kind: 'assign', blockId: chorus.id, sectionId: p.sectionIds[2] }));
  p = ok(editLyrics(p, { kind: 'paste', paste: PASTE.replace('nine ten', 'nine ten eleven') }));
  assert.equal(structureHash(p.abc), hash);
  assert.deepEqual(bars(p), shape);
  assert.ok(structureConfirmed(p));
  // The unchanged verse kept its place; the changed chorus is back to Unassigned and flagged.
  assert.deepEqual(p.assignments[p.sectionIds[1]], { kind: 'words', blockIds: [verse.id] });
  assert.equal(p.assignments[p.sectionIds[2]], undefined);
  assert.deepEqual(p.flagged, [p.sectionIds[2]]);
  assert.match((confirmLyrics(p) as { error: string }).error, /chorus \(section 3\) needs words/);
  p = ok(editLyrics(p, { kind: 'assign', blockId: p.blocks[1].id, sectionId: p.sectionIds[2] }));
  assert.deepEqual(p.flagged, []);
  p = ok(confirmLyrics(p));
  const pair = exportPrepare(p);
  if ('error' in pair) assert.fail(pair.error);
  assert.equal(pair.lyrics, '[intro]\n\n[verse]\none two three\nfour five six\n\n[chorus]\nseven eight\nnine ten eleven');
  // The server's lint pairs the exported lyrics with the confirmed score.
  assert.equal(lintScoreSections(scoreSections(pair.abc), pair.lyrics).ok, true);
  // Any later word change makes the lyric confirmation stale.
  const edited = ok(editLyrics(p, { kind: 'text', blockId: verse.id, text: 'one two' }));
  assert.match((exportPrepare(edited) as { error: string }).error, /Confirm the lyrics/);
});

test('a structure edit clears only the sections it changed', () => {
  let p = createPrepare('src', ABC, PASTE);
  const [s1, s2, s3, s4, s5] = p.sectionIds;
  p = ok(editLyrics(p, { kind: 'instrumental', sectionId: s1 }));
  p = ok(editLyrics(p, { kind: 'assign', blockId: p.blocks[0].id, sectionId: s2 }));
  p = ok(editLyrics(p, { kind: 'assign', blockId: p.blocks[1].id, sectionId: s5 }));
  const before = p.lyricRevision;
  // Relabel the pre-chorus (no words): nothing assigned changes, nothing is flagged.
  const renamed = ok(editStructure(p, { kind: 'relabel', index: 2, label: 'bridge' }));
  assert.deepEqual(renamed.sectionIds, p.sectionIds);
  assert.deepEqual(renamed.assignments, p.assignments);
  assert.equal(renamed.lyricRevision, before);
  // Merging verse with pre-chorus changes the verse: it loses its words, flagged.
  const merged = ok(editStructure(p, { kind: 'merge', index: 1, keep: 'first' }));
  assert.deepEqual(merged.sectionIds, [s1, s2, s4, s5]);
  assert.equal(merged.assignments[s2], undefined);
  assert.deepEqual(merged.assignments[s1], { kind: 'instrumental' });
  assert.deepEqual(merged.assignments[s5], p.assignments[s5]);
  assert.deepEqual(merged.flagged, [s2]);
  assert.equal(merged.lyricRevision, before + 1);
  assert.equal(merged.blocks.length, p.blocks.length, 'words are never dropped');
  // A split adds a new id after the split section; both halves changed.
  const split = ok(editStructure(p, { kind: 'split', index: 4, bar: 11, label: 'outro' }));
  assert.deepEqual(split.sectionIds, [s1, s2, s3, s4, s5, `s${p.nextId}`]);
  assert.equal(split.assignments[s5], undefined);
  assert.deepEqual(bars(split).slice(-2), ['chorus@9', 'outro@11']);
  // Moving a boundary changes both neighbours.
  const moved = ok(editStructure(p, { kind: 'move', index: 1, bar: 6 }));
  assert.equal(moved.assignments[s2], undefined);
  assert.deepEqual(moved.assignments[s1], undefined);
  // A raw rewrite with a different section count makes every section new.
  const raw = ok(editStructure(p, { kind: 'replace', abc: ABC.replace('% pre-chorus\n', '') }));
  assert.equal(raw.sectionIds.length, 4);
  assert.ok(raw.sectionIds.every(id => !p.sectionIds.includes(id)));
  assert.deepEqual(raw.assignments, {});
  // Score edits never touch lyric text.
  assert.equal(merged.paste, PASTE);
  assert.deepEqual(merged.blocks, p.blocks);
});

test('assigning never silently replaces words', () => {
  let p = createPrepare('src', ABC, '[A]\nfirst\n\n[B]\nsecond\n\n[C]\nthird');
  const [a, b, c] = p.blocks.map(x => x.id);
  const [s1, s2] = p.sectionIds;
  p = ok(editLyrics(p, { kind: 'assign', blockId: a, sectionId: s1 }));
  p = ok(editLyrics(p, { kind: 'assign', blockId: b, sectionId: s2 }));
  assert.match((editLyrics(p, { kind: 'assign', blockId: c, sectionId: s1 }) as { error: string }).error, /append or swap/);
  assert.match((editLyrics(p, { kind: 'instrumental', sectionId: s1 }) as { error: string }).error, /Clear/);
  const appended = ok(editLyrics(p, { kind: 'assign', blockId: c, sectionId: s1, mode: 'append' }));
  assert.deepEqual(appended.assignments[s1], { kind: 'words', blockIds: [a, c] });
  // Swap: the target's words take the moved block's old place.
  const swapped = ok(editLyrics(p, { kind: 'assign', blockId: b, sectionId: s1, mode: 'swap' }));
  assert.deepEqual(swapped.assignments[s1], { kind: 'words', blockIds: [b] });
  assert.deepEqual(swapped.assignments[s2], { kind: 'words', blockIds: [a] });
  // Moving a block takes it from its old section.
  const movedBlock = ok(editLyrics(p, { kind: 'assign', blockId: b, sectionId: p.sectionIds[2] }));
  assert.equal(movedBlock.assignments[s2], undefined);
  // Excluding takes it out of the map; assigning brings it back in.
  const excluded = ok(editLyrics(p, { kind: 'exclude', blockId: a }));
  assert.equal(excluded.assignments[s1], undefined);
  assert.deepEqual(excluded.excluded, [a]);
  assert.deepEqual(ok(editLyrics(excluded, { kind: 'assign', blockId: a, sectionId: s1 })).excluded, []);
  assert.ok([appended, swapped, excluded].every(x => x.lyricRevision > p.lyricRevision));
});

test('one pasted chorus reused for later choruses gives independent copies', () => {
  let p = createPrepare('src', ABC, '[Chorus]\nla la');
  const [, , , s4, s5] = p.sectionIds;
  const chorus = p.blocks[0].id;
  p = ok(editLyrics(p, { kind: 'assign', blockId: chorus, sectionId: s4 }));
  p = ok(editLyrics(p, { kind: 'reuse', blockId: chorus, sectionIds: [s5] }));
  const copy = p.blocks.at(-1)!;
  assert.equal(copy.copyOf, chorus);
  assert.deepEqual(p.assignments[s5], { kind: 'words', blockIds: [copy.id] });
  p = ok(editLyrics(p, { kind: 'text', blockId: copy.id, text: 'la la la' }));
  assert.equal(p.blocks[0].text, 'la la', 'editing a copy leaves the original');
  assert.match((editLyrics(p, { kind: 'reuse', blockId: chorus, sectionIds: [s4] }) as { error: string }).error, /none/);
});

test('split and join blocks keep their place in the map', () => {
  let p = createPrepare('src', ABC, '[Verse]\none\ntwo\nthree');
  const [, s2, s3] = p.sectionIds;
  p = ok(editLyrics(p, { kind: 'assign', blockId: p.blocks[0].id, sectionId: s2 }));
  const split = ok(editLyrics(p, { kind: 'splitBlock', blockId: p.blocks[0].id, line: 2 }));
  assert.deepEqual(split.blocks.map(b => b.text), ['one\ntwo', 'three']);
  assert.deepEqual(split.assignments[s2], { kind: 'words', blockIds: split.blocks.map(b => b.id) });
  // The second half can go to the next section on its own.
  const moved = ok(editLyrics(split, { kind: 'assign', blockId: split.blocks[1].id, sectionId: s3 }));
  assert.deepEqual(moved.assignments[s2], { kind: 'words', blockIds: [split.blocks[0].id] });
  const joined = ok(editLyrics(moved, { kind: 'joinNext', blockId: split.blocks[0].id }));
  assert.deepEqual(joined.blocks.map(b => b.text), ['one\ntwo\nthree']);
  assert.equal(joined.assignments[s3], undefined);
  assert.match((editLyrics(p, { kind: 'splitBlock', blockId: p.blocks[0].id, line: 3 }) as { error: string }).error, /between/);
});

test('re-paste re-suggests identical duplicates instead of guessing', () => {
  let p = createPrepare('src', ABC, '[Chorus]\nsame\n\n[Chorus]\nsame\n\n[Verse]\nonly');
  p = ok(editLyrics(p, { kind: 'assign', blockId: p.blocks[0].id, sectionId: p.sectionIds[3] }));
  p = ok(editLyrics(p, { kind: 'assign', blockId: p.blocks[2].id, sectionId: p.sectionIds[1] }));
  const again = ok(editLyrics(p, { kind: 'paste', paste: '[Chorus]\nsame\n\n[Chorus]\nsame\n\n[Verse]\nonly' }));
  assert.equal(again.assignments[p.sectionIds[3]], undefined, 'which "same" was assigned is ambiguous');
  assert.deepEqual(again.assignments[p.sectionIds[1]], p.assignments[p.sectionIds[1]]);
  assert.equal(again.blocks.length, 3);
});

test('legacy drafts migrate from the saved score, proposing a map only when tags match', () => {
  const lyrics = '[intro]\n\n[verse]\none\n\n[pre-chorus]\ntwo\n\n[chorus]\nthree\n\n[Chorus]\nfour';
  const p = migrateLegacyPrepare('src', ABC, lyrics);
  assert.equal(p.abc, ABC);
  assert.deepEqual(p.assignments[p.sectionIds[0]], { kind: 'instrumental' });
  assert.deepEqual(p.sectionIds.slice(1).map(id => p.assignments[id]),
    p.blocks.map(b => ({ kind: 'words', blockIds: [b.id] })));
  assert.deepEqual(p.confirmed, { structure: null, lyrics: null }, 'the first V2 confirmation is the user\'s');
  // Confirming both reproduces the legacy pair the server lint accepted.
  const done = ok(confirmLyrics(ok(confirmStructure(p))));
  const pair = exportPrepare(done);
  if ('error' in pair) assert.fail(pair.error);
  assert.equal(lintScoreSections(scoreSections(pair.abc), pair.lyrics).ok, true);
  // A mismatch keeps every word, all Unassigned.
  const off = migrateLegacyPrepare('src', ABC, '[Verse]\none\n\n[Chorus]\nthree');
  assert.deepEqual(off.assignments, {});
  assert.deepEqual(off.blocks.map(b => b.text), ['one', 'three']);
  // Words above the first tag also count as a mismatch.
  assert.deepEqual(migrateLegacyPrepare('src', ABC, `stray\n${lyrics}`).assignments, {});
});

test('structure confirmation refuses what the score check would miscount', () => {
  const directive = createPrepare('src', `${ABC}\n%%MIDI program 1`);
  assert.match((confirmStructure(directive) as { error: string }).error, /%%/);
  assert.match((confirmStructure(createPrepare('src', 'X:1\nK:C\nC|')) as { error: string }).error, /no sections/);
  assert.match((confirmLyrics(createPrepare('src', ABC)) as { error: string }).error, /structure first/);
});
