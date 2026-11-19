import assert from 'node:assert/strict';
import test from 'node:test';
import { barRanges, mergeScoreSections, mergeSections, relabelScoreSection, relabelSection, splitScoreSection, splitSection, type ScoreEdit, type SectionEdit } from './abcSectionMove';
import { abcSectionMarks, lyricBlockLines, withBarNumbers } from './abcSections';
// The server's own readers: its section lint, and the planner's block split.
import { lintScoreSections, scoreSections } from '../../../../server/src/services/backends/yue2/scoreSections';
import { splitLyricBlocks } from '../../../../server/src/services/backends/yue2/sectionMatch';
import { isPresent } from '@/utils/guards';

const ABC = [
  'X:1', 'T:t', 'M:4/4', 'L:1/8', 'K:C',
  '% intro', 'V:Vocal', 'Z4|', 'V:Ins', '"C"C8|"F"F8|"G"G8|"C"C8|',
  '% verse', 'V:Vocal', 'C2D2E2F2|G8|A8|G8|', 'V:Ins', '"C"C8|"G"G8|"F"F8|"C"C8|',
  '% chorus', 'V:Vocal', 'c8|B8|A8|G8|', 'V:Ins', '"F"F8|"G"G8|"A"A8|"C"C8|',
].join('\n');

const LYRICS = '[Intro]\n\n[Verse]\none two three\nfour five six\n\n[Chorus]\nseven eight\nnine ten\n';

const ok = (edit: SectionEdit) => {
  if ('error' in edit) assert.fail(edit.error);
  assert.equal(edit.removedMusic, false);
  // The server's section check passes on the result.
  assert.equal(lintScoreSections(scoreSections(edit.abc), edit.lyrics).ok, true, lintScoreSections(scoreSections(edit.abc), edit.lyrics).message);
  return edit;
};
const words = (lyrics: string) => lyrics.replace(/^\s*\[[^\]\n]+\]\s*$/gm, '').split(/\s+/).filter(isPresent);
const sections = (abc: string) => scoreSections(abc).map(s => `${s.label}@${s.startBar}`);
// What the planner sings from: sectionMatch.ts drops blocks with no words.
const sung = (lyrics: string) => splitLyricBlocks(lyrics).filter(b => b.body.trim()).map(b => `${b.tag}: ${words(b.body).join(' ')}`);
// Each voice's bars in order, whatever groups they sit in; a split Z4 is Z2|Z2.
const music = (abc: string) => {
  const voices: Record<string, string[]> = {};
  let voice = '';
  for (const line of abc.split('\n').slice(5)) {
    if (line.startsWith('V:')) voice = line.slice(2);
    else if (!line.startsWith('%')) (voices[voice] ??= []).push(...line.split('|').filter(isPresent)
      .flatMap(bar => /^Z\d*$/.test(bar) ? Array(Number(bar.slice(1) || 1)).fill('Z') : [bar]));
  }
  return voices;
};

test('merge keeps the first label, joins the lyric blocks and leaves every bar alone', () => {
  const edit = ok(mergeSections(ABC, LYRICS, 1, 'first'));
  assert.deepEqual(sections(edit.abc), ['intro@1', 'verse@5']);
  assert.equal(edit.abc, ABC.replace('% chorus\n', ''));
  assert.deepEqual(sung(edit.lyrics), ['Verse: one two three four five six seven eight nine ten']);
});

test('merge can keep the second label instead', () => {
  const edit = ok(mergeSections(ABC, LYRICS, 1, 'second'));
  assert.deepEqual(sections(edit.abc), ['intro@1', 'chorus@5']);
  assert.deepEqual(sung(edit.lyrics), ['Chorus: one two three four five six seven eight nine ten']);
  assert.deepEqual(words(edit.lyrics), words(LYRICS));
});

test('merging an empty instrumental block into a sung one keeps the words where they were', () => {
  const first = ok(mergeSections(ABC, LYRICS, 0, 'first'));
  assert.deepEqual(sections(first.abc), ['intro@1', 'chorus@9']);
  assert.deepEqual(sung(first.lyrics), ['Intro: one two three four five six', 'Chorus: seven eight nine ten']);
  const second = ok(mergeSections(ABC, LYRICS, 0, 'second'));
  assert.deepEqual(sections(second.abc), ['verse@1', 'chorus@9']);
  assert.deepEqual(sung(second.lyrics), ['Verse: one two three four five six', 'Chorus: seven eight nine ten']);
});

test('relabel renames the marker and the tag, keeping their spacing', () => {
  const edit = ok(relabelSection(ABC.replace('% chorus', '%chorus'), LYRICS, 2, ' Bridge '));
  assert.match(edit.abc, /^%Bridge$/m);
  assert.match(edit.lyrics, /^\[Bridge\]$/m);
  assert.deepEqual(words(edit.lyrics), words(LYRICS));
});

test('split moves the lyric lines from the chosen boundary into the new block', () => {
  assert.deepEqual(lyricBlockLines(LYRICS, 1), ['one two three', 'four five six']);
  const edit = ok(splitSection(ABC, LYRICS, 1, 7, 1, 'pre-chorus'));
  assert.deepEqual(sections(edit.abc), ['intro@1', 'verse@5', 'pre-chorus@7', 'chorus@9']);
  assert.deepEqual(music(edit.abc), music(ABC));
  assert.deepEqual(words(edit.lyrics), words(LYRICS));
  // The planner sees all three sung blocks, each with its own words.
  assert.deepEqual(sung(edit.lyrics), ['Verse: one two three', 'pre-chorus: four five six', 'Chorus: seven eight nine ten']);
});

test('a split at either end of the lyrics leaves one half instrumental, and the planner still pairs the sung blocks', () => {
  const none = ok(splitSection(ABC, LYRICS, 1, 7, 2, 'break'));
  assert.deepEqual(sections(none.abc), ['intro@1', 'verse@5', 'break@7', 'chorus@9']);
  assert.deepEqual(sung(none.lyrics), ['Verse: one two three four five six', 'Chorus: seven eight nine ten']);
  const all = ok(splitSection(ABC, LYRICS, 1, 6, 0, 'verse'));
  assert.deepEqual(sections(all.abc), ['intro@1', 'verse@5', 'verse@6', 'chorus@9']);
  assert.deepEqual(sung(all.lyrics), ['verse: one two three four five six', 'Chorus: seven eight nine ten']);
  // An empty instrumental section splits into two empty ones.
  const intro = ok(splitSection(ABC, LYRICS, 0, 3, 0, 'intro'));
  assert.deepEqual(sections(intro.abc), ['intro@1', 'intro@3', 'verse@5', 'chorus@9']);
  assert.deepEqual(sung(intro.lyrics), sung(LYRICS));
  assert.deepEqual(music(intro.abc), music(ABC));
});

test('edits refuse when the lyrics do not already pair with the sections', () => {
  const short = LYRICS.replace('[Intro]\n\n', '');
  const swapped = LYRICS.replace('[Chorus]', '[Bridge]');
  for (const lyrics of [short, swapped, LYRICS.replace('[Verse]\n', '[Verse] one\n')]) {
    for (const edit of [mergeSections(ABC, lyrics, 1, 'first'), relabelSection(ABC, lyrics, 1, 'hook'), splitSection(ABC, lyrics, 1, 7, 1, 'hook')])
      assert.ok('error' in edit, JSON.stringify(edit));
  }
  const counts = mergeSections(ABC, short, 0, 'first');
  assert.ok('error' in counts && /3 sections but the lyrics have 2 tags/.test(counts.error));
  const label = relabelSection(ABC, swapped, 0, 'hook');
  assert.ok('error' in label && /section 3 is chorus but lyric tag 3 is Bridge/.test(label.error));
});

test('score-only edits never read lyrics and give the paired edits\' score', () => {
  const abcOf = (edit: SectionEdit | ScoreEdit) => { if ('error' in edit) assert.fail(edit.error); return edit.abc; };
  assert.equal(abcOf(mergeScoreSections(ABC, 1, 'second')), abcOf(mergeSections(ABC, LYRICS, 1, 'second')));
  assert.equal(abcOf(relabelScoreSection(ABC, 1, 'hook')), abcOf(relabelSection(ABC, LYRICS, 1, 'hook')));
  assert.equal(abcOf(splitScoreSection(ABC, 1, 7, 'hook')), abcOf(splitSection(ABC, LYRICS, 1, 7, 1, 'hook')));
  assert.deepEqual(scoreSections(abcOf(splitScoreSection(ABC, 1, 7, 'hook'))).map(s => `${s.label}@${s.startBar}`),
    ['intro@1', 'verse@5', 'hook@7', 'chorus@9']);
  assert.ok('error' in mergeScoreSections(ABC, 2, 'first'));
  assert.ok('error' in relabelScoreSection(ABC, 0, '[x]'));
});

test('edits refuse bad indexes, bars, lyric lines and names', () => {
  for (const edit of [
    mergeSections(ABC, LYRICS, 2, 'first'), relabelSection(ABC, LYRICS, 3, 'x'), relabelSection(ABC, LYRICS, 0, ' '),
    relabelSection(ABC, LYRICS, 0, 'a]b'), relabelSection(ABC, LYRICS, 0, '%x'),
    splitSection(ABC, LYRICS, 1, 5, 1, 'x'), splitSection(ABC, LYRICS, 1, 9, 1, 'x'), splitSection(ABC, LYRICS, 1, 7, 3, 'x'),
  ]) assert.ok('error' in edit, JSON.stringify(edit));
});

test('a split refuses a bar that is not a whole bar number', () => {
  for (const bar of [7.5, NaN, Infinity, -Infinity])
    assert.ok('error' in splitSection(ABC, LYRICS, 1, bar, 1, 'x'), String(bar));
});

test('a name with $ sequences is written literally', () => {
  const edit = ok(relabelSection(ABC, LYRICS, 1, "$& $1 $$ $'"));
  assert.match(edit.lyrics, /^\[\$& \$1 \$\$ \$'\]$/m);
  assert.match(edit.abc, /^% \$& \$1 \$\$ \$'$/m);
  const merged = ok(mergeSections(edit.abc, edit.lyrics, 0, 'second'));
  assert.match(merged.lyrics, /^\[\$& \$1 \$\$ \$'\]$/m);
  assert.equal(merged.lyrics.match(/^\[/gm)?.length, 2);
});

test('edits refuse a %% directive the lint counts, and words above the first tag', () => {
  const midi = ABC.replace('% intro', '%%MIDI program 1\n% intro');
  assert.ok('error' in relabelSection(midi, LYRICS, 0, 'x'));
  for (const lyrics of ['stray words\n' + LYRICS, '\nstray words\n\n' + LYRICS])
    assert.ok('error' in mergeSections(ABC, lyrics, 1, 'first'), lyrics);
  // Blank lines above the first tag are fine.
  ok(mergeSections(ABC, '\n\n' + LYRICS, 1, 'first'));
});

test('lyrics with no tags at all are left alone and only the score changes', () => {
  const edit = mergeSections(ABC, 'la la la', 1, 'first');
  assert.ok(!('error' in edit));
  assert.equal(edit.lyrics, 'la la la');
  assert.deepEqual(sections(edit.abc), ['intro@1', 'verse@5']);
});

test('CRLF lyrics keep their line endings', () => {
  const edit = ok(splitSection(ABC, LYRICS.replace(/\n/g, '\r\n'), 1, 7, 1, 'pre-chorus'));
  assert.ok(!/[^\r]\n/.test(edit.lyrics));
});

test('numbering every bar on the lead sheet changes neither its bars nor its sections', () => {
  const abc = ['X:1', 'M:4/4', 'L:1/8', 'K:C', '% intro', 'V: Vocal', 'z8|z8|', '% verse', 'V: Vocal', 'C8|D8|E8|', ''].join('\n');
  const numbered = withBarNumbers(abc);
  assert.equal(numbered.split('\n')[1], '%%barnumbers 1');
  assert.deepEqual(barRanges(numbered).map(r => r.bar), barRanges(abc).map(r => r.bar));
  assert.deepEqual(abcSectionMarks(numbered).map(m => m.label), ['intro', 'verse']);
});
