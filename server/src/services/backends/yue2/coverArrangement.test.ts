import assert from 'node:assert/strict';
import test from 'node:test';
import {
  composeArrangement, editPlan, generateRefusal, keepRefusal, lookaheadSection, migrateLegacySession, planDrawRefusal, planPrefix, revisionRefusal,
  sourceSections, spliceSection, takeIdentity, type ArrangedSession, type LegacySession,
} from './coverArrangement.js';
import { scoreSections } from './scoreSections.js';
import { scoreBarClock } from './scoreClock.js';
import { buildYue2LyricSchedule } from './lyricSchedule.js';
import { splitLyricBlocks } from './sectionMatch.js';
import { lyricsThrough } from '../../yue2Takes.js';
import type { Yue2Take } from '../../yue2Takes.js';

// Laid out the way the transcriber writes a cover score: Vocal and Ins lines
// per section, and a meter change restated in both voices. The chorus ends in
// 2/4 and the bridge carries that 2/4 on, so moving either one moves meter.
const score = [
  'X:1', 'T:Fixture', 'M:4/4', 'L:1/16', 'Q:1/4=120',
  'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"', 'V: Ins clef=treble name="Ins Melody" snm="Inst."', 'K:C',
  '% intro', 'V: Vocal', 'Z2|', 'V: Ins', 'C16|E16|',
  '% verse', 'V: Vocal', 'C4D4E4F4|G16|', 'V: Ins', 'Z2|',
  '% chorus', 'V: Vocal', 'c16|', 'V: Ins', 'Z|', 'V: Vocal', 'M:2/4', 'c8|', 'V: Ins', 'M:2/4', 'Z|',
  '% bridge', 'V: Vocal', 'B8|A8|', 'V: Ins', 'Z2|',
  '% outro', 'V: Vocal', 'M:4/4', 'Z|', 'V: Ins', 'M:4/4', 'C16|', '',
].join('\n');
const lyrics = '[Intro]\n\n[Verse]\nwalking down the line\n\n[Chorus]\nsing it out\n\n[Bridge]\nturn around\n\n[Outro]\n';
const INTRO = 0, VERSE = 1, CHORUS = 2, BRIDGE = 3, OUTRO = 4;
const schedule = { mode: 'mask' as const, bias: -4, abc: false, leadSec: 0, behind: -1 };

/** Each bar's length in seconds, grouped by section. */
function barLengths(abc: string): number[][] {
  const bars = scoreBarClock(abc), sections = scoreSections(abc);
  return sections.map((s, i) => bars.slice(s.startBar - 1, (sections[i + 1]?.startBar ?? bars.length + 1) - 1)
    .map(b => b.end - b.start));
}

test('source sections carry their bar counts and lyric blocks', () => {
  assert.deepEqual(sourceSections(score, lyrics).map(s => [s.id, s.label, s.startBar, s.bars, s.lyricBlock]), [
    ['s0', 'intro', 1, 2, null], ['s1', 'verse', 3, 2, 0], ['s2', 'chorus', 5, 2, 1],
    ['s3', 'bridge', 7, 2, 2], ['s4', 'outro', 9, 1, null],
  ]);
});

test('the source order composes the source text exactly', () => {
  const out = composeArrangement(score, lyrics, [0, 1, 2, 3, 4]);
  assert.equal(out.abc, score);
  assert.equal(out.lyrics, lyrics);
  assert.equal(out.textPairingMatches, true);
  assert.deepEqual(out.unplacedBlocks, []);
  assert.deepEqual(out.occurrences.map(o => o.lyric), [null, [9, 40], [40, 62], [62, 84], null]);
});

for (const [name, plan] of [
  ['skip (the chorus that sets 2/4)', [INTRO, VERSE, BRIDGE, OUTRO]],
  ['reorder (bridge first)', [INTRO, BRIDGE, VERSE, CHORUS, OUTRO]],
  ['repeat (chorus twice, then again after the bridge)', [INTRO, VERSE, CHORUS, CHORUS, BRIDGE, CHORUS, OUTRO]],
] as const) {
  test(`${name}: section labels, bar numbers and bar lengths follow the plan`, () => {
    const out = composeArrangement(score, lyrics, [...plan]);
    const source = sourceSections(score, lyrics), sourceBars = barLengths(score);
    assert.deepEqual(scoreSections(out.abc).map(s => [s.label, s.startBar]),
      out.occurrences.map(o => [source[o.section].label, o.startBar]));
    // Each occurrence plays at its source meter and tempo wherever it lands.
    assert.deepEqual(barLengths(out.abc), plan.map(s => sourceBars[s]));
    assert.equal(out.textPairingMatches, true);
  });
}

test('a chorus repeated after itself restates 4/4 in each voice, once', () => {
  const out = composeArrangement(score, lyrics, [INTRO, VERSE, CHORUS, CHORUS, OUTRO]);
  const lines = out.abc.split('\n');
  const second = lines.lastIndexOf('% chorus');
  assert.deepEqual(lines.slice(second, second + 11),
    ['% chorus', 'V: Vocal', 'M:4/4', 'c16|', 'V: Ins', 'M:4/4', 'Z|', 'V: Vocal', 'M:2/4', 'c8|', 'V: Ins']);
});

test('repeated lyric blocks: each sung occurrence gets its own copy, in order', () => {
  const plan = [INTRO, VERSE, CHORUS, CHORUS, BRIDGE, CHORUS, OUTRO];
  const out = composeArrangement(score, lyrics, plan);
  assert.deepEqual(splitLyricBlocks(out.lyrics).map(b => b.tag),
    ['Intro', 'Verse', 'Chorus', 'Chorus', 'Bridge', 'Chorus', 'Outro']);
  const chars = Array.from(out.lyrics);
  assert.deepEqual(out.occurrences.map(o => (o.lyric ? chars.slice(...o.lyric).join('').trim() : null)), [
    null, '[Verse]\nwalking down the line', '[Chorus]\nsing it out', '[Chorus]\nsing it out',
    '[Bridge]\nturn around', '[Chorus]\nsing it out', null,
  ]);
  assert.deepEqual(out.unplacedBlocks, []);
  // The lyric schedule pairs every sung occurrence with its own block.
  const wire = buildYue2LyricSchedule(out.abc, out.lyrics, schedule).wire;
  assert.deepEqual(wire.sections.map(s => s.lyric[0]), out.occurrences.flatMap(o => (o.lyric ? [o.lyric[0]] : [])));
});

test('a skipped sung section reports its block as unplaced', () => {
  assert.deepEqual(composeArrangement(score, lyrics, [INTRO, VERSE, CHORUS, OUTRO]).unplacedBlocks, [2]);
});

test('the retained prefix: text before the next occurrence depends only on the kept ones', () => {
  // Kept intro, verse, chorus; then the plan changes what follows.
  const a = composeArrangement(score, lyrics, [INTRO, VERSE, CHORUS, BRIDGE, CHORUS, OUTRO]);
  const b = composeArrangement(score, lyrics, [INTRO, VERSE, CHORUS, CHORUS, OUTRO]);
  const cut = a.occurrences[3].abcStart;
  assert.equal(b.occurrences[3].abcStart, cut);
  assert.equal(Array.from(a.abc).slice(0, cut).join(''), Array.from(b.abc).slice(0, cut).join(''));
  const lyricCut = a.occurrences[2].lyric![1];
  assert.equal(b.occurrences[2].lyric![1], lyricCut);
  assert.equal(Array.from(a.lyrics).slice(0, lyricCut).join(''), Array.from(b.lyrics).slice(0, lyricCut).join(''));
  // So a resumed take replays the kept frames under the same schedule.
  const timed = (abc: string, text: string) => buildYue2LyricSchedule(abc, text, schedule).wire.sections.slice(0, 2);
  assert.deepEqual(timed(a.abc, a.lyrics), timed(b.abc, b.lyrics));
  // And yue2Takes' cut for aligning a take that stops early still lands on
  // the end of the occurrence after it.
  assert.equal(lyricsThrough(b.lyrics, b.occurrences.map(o => o.lyric), 3), Array.from(b.lyrics).slice(0, b.occurrences[3].lyric![1]).join(''));
});

test('kept text and offsets survive leaving the source order, with odd whitespace and empty tags', () => {
  const abc = ['X:1', 'M:4/4', 'L:1/16', 'Q:1/4=120', 'K:C', '% intro', 'V: Vocal', 'Z|', '% verse', 'V: Vocal', 'C16|',
    '% chorus', 'V: Vocal', 'c16|', '% outro', 'V: Vocal', 'Z|', ''].join('\n');
  for (const text of [
    '[Intro]\n\n[Verse]\nla la\n\n[Chorus]\noh oh\n\n[Outro]\n',
    'spoken first\n[Intro]\n[Verse]\n  la la  \n\n\n[Chorus]\noh oh\n[Outro]',
    '[Verse]\nla la\n[Chorus]\noh oh',
  ]) {
    const source = composeArrangement(abc, text, [0, 1, 2, 3]);
    const repeat = composeArrangement(abc, text, [0, 1, 2, 2, 3]);
    assert.equal(source.lyrics, text.endsWith('\n') ? text : `${text}\n`);
    // Intro and verse kept: everything up to the chorus is the same text.
    const keptLyrics = source.occurrences[2].lyric![0];
    assert.deepEqual(repeat.occurrences.slice(0, 3).map(o => [o.lyric, o.abcStart]), source.occurrences.slice(0, 3).map(o => [o.lyric, o.abcStart]));
    assert.equal(Array.from(repeat.lyrics).slice(0, keptLyrics).join(''), Array.from(source.lyrics).slice(0, keptLyrics).join(''));
    const cut = source.occurrences[2].abcStart;
    assert.equal(Array.from(repeat.abc).slice(0, cut).join(''), Array.from(source.abc).slice(0, cut).join(''));
    const timed = (o: typeof source) => buildYue2LyricSchedule(o.abc, o.lyrics, schedule).wire.sections[0];
    assert.deepEqual(timed(repeat), timed(source));
    assert.equal(repeat.textPairingMatches, true);
  }
});

test('an occurrence sings its own lines in place of its block, keeping the tag and the rest', () => {
  const plan = [INTRO, VERSE, CHORUS, CHORUS, OUTRO];
  const plain = composeArrangement(score, lyrics, plan);
  const out = composeArrangement(score, lyrics, plan, [undefined, undefined, undefined, 'sing it again\nlouder', undefined]);
  const chars = Array.from(out.lyrics);
  assert.deepEqual(out.occurrences.map(o => (o.lyric ? chars.slice(...o.lyric).join('').trim() : null)), [
    null, '[Verse]\nwalking down the line', '[Chorus]\nsing it out', '[Chorus]\nsing it again\nlouder', null,
  ]);
  // The first chorus, and everything before the second, are untouched.
  const cut = plain.occurrences[3].lyric![0];
  assert.equal(chars.slice(0, cut).join(''), Array.from(plain.lyrics).slice(0, cut).join(''));
  assert.equal(out.textPairingMatches, true);
  // An unsung section has no lines to replace.
  assert.equal(composeArrangement(score, lyrics, plan, ['words']).lyrics, plain.lyrics);
});

test('two sections of one kind pair with their own tags, the empty one included', () => {
  // Prepare writes an empty tag for an instrumental section: the first intro
  // is that one, the second intro sings.
  const abc = ['X:1', 'M:4/4', 'L:1/16', 'Q:1/4=120', 'K:C', '% intro', 'V: Vocal', 'Z|', '% intro', 'V: Vocal', 'C16|',
    '% verse', 'V: Vocal', 'c16|', ''].join('\n');
  const text = '[Intro]\n\n[Intro]\nla la\n\n[Verse]\noh oh\n';
  assert.deepEqual(sourceSections(abc, text).map(s => s.lyricBlock), [null, 0, 1]);
  const chars = (o: ReturnType<typeof composeArrangement>) => o.occurrences.map(x => (x.lyric ? Array.from(o.lyrics).slice(...x.lyric).join('').trim() : null));
  // Skipping the sung intro leaves its words out, not in the silent one.
  const skip = composeArrangement(abc, text, [0, 2]);
  assert.deepEqual(chars(skip), [null, '[Verse]\noh oh']);
  assert.equal(skip.lyrics.includes('la la'), false);
  assert.deepEqual(chars(composeArrangement(abc, text, [0, 1, 2])), [null, '[Intro]\nla la', '[Verse]\noh oh']);
});

test('an unsung section moved before a sung one of its kind breaks re-pairing the text', () => {
  const abc = ['X:1', 'M:4/4', 'L:1/16', 'Q:1/4=120', 'K:C', '% verse', 'V: Vocal', 'C16|',
    '% chorus', 'V: Vocal', 'c16|', '% chorus', 'V: Vocal', 'Z|', ''].join('\n');
  const text = '[Verse]\nla la\n\n[Chorus]\noh oh\n';
  const out = composeArrangement(abc, text, [0, 2, 1]);
  assert.deepEqual(out.occurrences.map(o => o.lyric !== null), [true, false, true]);
  assert.equal(out.textPairingMatches, false);
});

// --- session identity ------------------------------------------------------

function take(id: string, from: number, to: number, starts: Array<number | null>): Yue2Take {
  return { id, from, to, seed: 1, audioUrl: `/audio/takes/x/${id}.wav`, frames: 300, starts };
}

function legacy(): LegacySession {
  return {
    sections: scoreSections(score).map(s => ({ label: s.label, startBar: s.startBar })),
    takes: [
      take('a', 0, 1, [0, 50, 120, null, null]),
      take('b', 2, 2, [0, 50, 120, 200, null]),
      take('c', 2, 2, [0, 50, 120, 210, null]),
      take('d', 0, 0, [0, 60, null, null, null]),
    ],
    kept: ['a'],
    request: { abc: score, lyrics },
  };
}
const codes: Record<string, number[]> = {
  a: Array.from({ length: 130 }, (_, i) => i),
  b: Array.from({ length: 120 }, (_, i) => i).concat(9, 9, 9),
  c: Array.from({ length: 125 }, () => 7),
  d: Array.from({ length: 70 }, (_, i) => i),
};

test('a legacy session migrates to the source-order plan, with parents from the codes', () => {
  const s = migrateLegacySession(legacy(), id => codes[id]);
  assert.equal(s.version, 2);
  assert.equal(s.revision, 0);
  assert.deepEqual(s.arrangement.plan.map(o => `${o.id}:${o.section}`), ['o0:s0', 'o1:s1', 'o2:s2', 'o3:s3', 'o4:s4']);
  assert.equal(s.arrangement.finished, true);
  assert.deepEqual(s.takes.map(t => [t.id, t.parent, t.occurrences.join()]),
    [['a', null, 'o0,o1'], ['b', 'a', 'o2'], ['c', null, 'o2'], ['d', null, 'o0']]);
  assert.equal(keepRefusal(s, s.takes[1]), null);
  assert.match(keepRefusal(s, s.takes[2])!, /does not follow/);
  assert.match(keepRefusal(s, s.takes[3])!, /does not follow/);
});

test('plan edits keep built occurrences, give new ones fresh ids, and retire old candidates', () => {
  const s: ArrangedSession = migrateLegacySession(legacy(), id => codes[id]);
  // Reorder and repeat what is unbuilt: a new chorus, the bridge, the old chorus.
  const arrangement = editPlan(s, ['s2', 'o3', 'o2'], false);
  assert.deepEqual(arrangement.plan.map(o => `${o.id}:${o.section}`), ['o0:s0', 'o1:s1', 'o5:s2', 'o3:s3', 'o2:s2']);
  assert.equal(arrangement.nextOccurrence, 6);
  assert.throws(() => editPlan(s, ['o0'], true), /No such section/);
  const edited: ArrangedSession = { ...s, arrangement, revision: s.revision + 1 };
  assert.match(revisionRefusal(edited, 0)!, /changed/);
  assert.equal(revisionRefusal(edited, 1), null);
  assert.match(keepRefusal(edited, edited.takes[1])!, /before the last change/);
  assert.match(keepRefusal(edited, { ...edited.takes[1], revision: 1 })!, /different arrangement/);
  // An open plan builds its last occurrence too, running on into the source
  // section after it; a finished one, or one ending on the score's last, adds none.
  assert.equal(generateRefusal(edited, 4), null);
  assert.match(generateRefusal(edited, 5)!, /Pick an occurrence/);
  assert.equal(lookaheadSection(edited.arrangement), 3);
  assert.equal(lookaheadSection({ ...edited.arrangement, finished: true }), null);
  assert.equal(lookaheadSection(editPlan(s, ['s4'], false)), null);
  assert.deepEqual(takeIdentity(edited, 3), { revision: 1, parent: 'a', occurrences: ['o5', 'o3'] });
});

test('undo makes a spine\'s candidates keepable again; a kept ending locks the plan', () => {
  const s: ArrangedSession = migrateLegacySession(legacy(), id => codes[id]);
  const kept = { ...s, kept: ['a', 'b'] };
  assert.match(keepRefusal(kept, s.takes[1])!, /does not follow/);
  assert.equal(keepRefusal({ ...kept, kept: ['a'] }, s.takes[1]), null);
  const ending = { ...s, takes: [...s.takes, { ...s.takes[1], id: 'e', from: 2, to: 4, occurrences: ['o2', 'o3', 'o4'] }], kept: ['a', 'e'] };
  assert.throws(() => editPlan(ending, ['s2'], true), /already ends the song/);
  assert.throws(() => editPlan(ending, [], false), /already ends the song/);
  assert.equal(editPlan(ending, [], true).plan.length, 5);
  // A kept take that ran on into a lookahead composed no ending: the song can still grow, or stop ending.
  const ranOn = { ...ending, takes: ending.takes.map(t => t.id === 'e' ? { ...t, starts: [0, 100, 200, 300, 400, 500] } : t) };
  assert.equal(editPlan(ranOn, ['s2'], true).plan.length, 6);
  assert.equal(editPlan(ranOn, [], false).finished, false);
});

test('an improv is rest bars in the meter in force, in every voice, sung by nobody', () => {
  // After the chorus the meter is 2/4 at L:1/16: a bar is z8.
  const out = composeArrangement(score, lyrics, [VERSE, CHORUS, { improv: 4 }, OUTRO]);
  const lines = out.abc.split('\n');
  const at = lines.indexOf('% Improv');
  assert.deepEqual(lines.slice(at, at + 5), ['% Improv', 'V: Vocal', 'z8|z8|z8|z8|', 'V: Ins', 'z8|z8|z8|z8|']);
  assert.deepEqual(scoreSections(out.abc).map(s => [s.label, s.startBar]), [['verse', 1], ['chorus', 3], ['Improv', 5], ['outro', 9]]);
  assert.deepEqual(barLengths(out.abc)[2], [1, 1, 1, 1]);
  assert.deepEqual(out.occurrences[2], { section: -1, startBar: 5, bars: 4, abcStart: Array.from(lines.slice(0, at).join('\n')).length + 1, lyric: null });
  assert.equal(out.lyrics, composeArrangement(score, lyrics, [VERSE, CHORUS, OUTRO]).lyrics);
  assert.equal(out.textPairingMatches, true);
  // At the start, before any meter change: 4/4 at L:1/16 is z16.
  assert.match(composeArrangement(score, lyrics, [{ improv: 2 }, VERSE]).abc, /% Improv\nV: Vocal\nz16\|z16\|\nV: Ins\nz16\|z16\|/);
  assert.throws(() => composeArrangement(score, lyrics, [{ improv: 3 }]), /No such section/);
});

test('plan edits add improvs by bar count; the source runs on after the last source section', () => {
  const s: ArrangedSession = migrateLegacySession(legacy(), id => codes[id]);
  const arrangement = editPlan(s, ['s2', 'improv:8'], false);
  assert.deepEqual(arrangement.plan.slice(2), [{ id: 'o5', section: 's2' }, { id: 'o6', section: 'improv', bars: 8 }]);
  assert.equal(lookaheadSection(arrangement), 3);
  assert.throws(() => editPlan(s, ['improv:3'], false), /2, 4, 8, 16 bars/);
  // Keeping an upcoming improv by its id keeps its bars.
  const kept = editPlan({ ...s, arrangement }, ['o6', 'o5'], false);
  assert.deepEqual(kept.plan.slice(2), [{ id: 'o6', section: 'improv', bars: 8 }, { id: 'o5', section: 's2' }]);
});

test('plan_continue: the prefix is the composed score up to an occurrence plus its label line', () => {
  // Verse then chorus then chorus again: the second chorus is planned after the first.
  const plan = [VERSE, CHORUS, CHORUS];
  const composed = composeArrangement(score, lyrics, plan);
  const prefix = planPrefix(score, lyrics, plan, [], 2);
  assert.ok(prefix.endsWith('% chorus\n'));
  assert.equal(prefix.split('% chorus').length - 1, 2, 'the first chorus is inside the prefix');
  assert.ok(composed.abc.startsWith(prefix.slice(0, -'% chorus\n'.length)));
  assert.ok(planPrefix(score, lyrics, plan, [], 0).endsWith('% verse\n'));
});

test('plan_continue: a section\'s lines are replaced, and the labels stay', () => {
  const lines = 'V: Vocal\nd16|\nV: Ins\nZ|\n';
  const out = spliceSection(score, CHORUS, lines);
  assert.equal(scoreSections(out).length, scoreSections(score).length);
  assert.ok(out.includes(`% chorus\n${lines}% bridge`));
  assert.ok(!out.includes('c16|\n'), 'the old chorus lines are gone');
  assert.ok(spliceSection(score, OUTRO, 'V: Vocal\nZ|\nV: Ins\nC8|\n').endsWith('C8|\n'), 'a score ending in a newline still does');
  assert.equal(out.split('% outro')[1], score.split('% outro')[1]);
});

test('plan_continue: the five checks on a draw', () => {
  const prefix = '% chorus\n', ok = 'V: Vocal\n"C"c16|c16|\nV: Ins\nZ2|\n';
  const refuse = (text: string, end = 'section_end', slot = { bars: 2, sung: true }) => planDrawRefusal(prefix, text, end, slot);
  assert.equal(refuse(prefix + ok), null);
  assert.equal(refuse(prefix + ok, 'eos'), null, 'a planner that ended the song here is accepted');
  assert.match(refuse(prefix + ok, 'limit_hit')!, /plan limit/);
  assert.match(refuse(`% verse\n${ok}`)!, /did not keep/);
  assert.match(refuse(`${prefix}${ok}% bridge\n${ok}`)!, /more than one section/);
  assert.match(refuse(`${prefix}M:3/4\n${ok}`)!, /meter/);
  assert.match(refuse(`${prefix}V: Vocal\n"C"c16|c16|\nV: Ins\nZ|\n`)!, /differ/);
  assert.match(refuse(`${prefix}V: Vocal\n"C"c16|c16|c16|c16|c16|\nV: Ins\nZ5|\n`)!, /far from/, 'over twice the slot');
  const silent = `${prefix}V: Vocal\nz16|z16|\nV: Ins\nZ2|\n`;
  assert.equal(refuse(silent, 'section_end', { bars: 2, sung: false }), null);
  assert.match(refuse(silent)!, /sounding vocal/, 'a lyric needs a bar to sing in');
});
