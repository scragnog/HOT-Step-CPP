import assert from 'node:assert/strict';
import test from 'node:test';
import {
  anchorDrift, carryChecked, carrySections, clampToNeighbours, nonRomanisedRatio, polishLines, reuseRepeats,
  summariseDrift,
} from './lyricsAlignment.js';
import type { AnchorDrift, LyricsLine, LyricsWord } from './lyricsReconcile.js';

const HOP = 0.05;

test('a repeated line takes its word offsets from the first time it was sung', () => {
  // Same words in the same rhythm, but the second chorus drifted inside the
  // line: only its first word is right, and the earlier occurrence says where
  // the other two belong.
  const first = line([word('hold', 10, 10.4), word('on', 10.4, 10.7), word('me', 10.7, 11.2)]);
  // Same 1.2 s span as the first, so the pace is unchanged and the offsets
  // are applied as they were.
  const second = line([word('hold', 30, 30.4), word('on', 30.9, 31.05), word('me', 31.05, 31.2)]);
  const stats = reuseRepeats([first, second], null);

  assert.equal(stats.repeatedLines, 1);
  assert.equal(stats.wordsMoved, 2);
  assert.equal(second.words[1].start, 30.4, 'the offset from the earlier line is copied');
  assert.equal(second.words[2].start, 30.7);
  assert.equal(second.words[0].start, 30, 'a word the repeat already agrees with stays put');
});

test('every later occurrence of a repeated line is anchored to its own start', () => {
  // Three choruses, the second and third both swallowed around their middle
  // word. Each takes the first occurrence's offsets relative to its own first
  // word — never the first occurrence's absolute times — and the third is not
  // left carrying the second's placement either.
  const first = line([word('hold', 10, 10.4), word('on', 10.4, 10.7), word('me', 10.7, 11.2)]);
  const second = line([word('hold', 30, 30.4), word('on', 30.9, 31.05), word('me', 31.05, 31.2)]);
  const third = line([word('hold', 50, 50.4), word('on', 50.9, 51.05), word('me', 51.05, 51.2)]);

  const stats = reuseRepeats([first, second, third], null);

  assert.equal(stats.repeatedLines, 2, 'every later occurrence is matched');
  assert.equal(stats.wordsMoved, 4);
  close(second.words[1].start, 30.4);
  close(second.words[2].start, 30.7);
  close(third.words[1].start, 50.4, 'the third occurrence is anchored to its own line, not the first');
  close(third.words[2].start, 50.7);
});

test('a swallowed earlier occurrence does not pace the later repeats', () => {
  // The first occurrence was collapsed by the aligner: every word lasts a few
  // hundredths. Its offsets are not a rhythm, so both repeats keep the timing
  // the aligner gave them instead of being rebuilt bunched at their starts.
  const first = line([
    word('we', 0, 0.04), word('could', 0.05, 0.09), word('run', 0.1, 0.14), word('away', 0.15, 0.19),
  ]);
  const second = line([
    word('we', 30, 30.45), word('could', 30.5, 30.95), word('run', 31, 31.45), word('away', 31.5, 31.95),
  ]);
  const third = line([
    word('we', 60, 60.45), word('could', 60.5, 60.95), word('run', 61, 61.45), word('away', 61.5, 61.95),
  ]);

  const stats = reuseRepeats([first, second, third], null);

  assert.equal(stats.repeatedLines, 2, 'the repeats are still recognised');
  assert.equal(stats.wordsMoved, 0, 'a collapsed source moves nothing');
  assert.equal(second.words[1].start, 30.5);
  assert.equal(second.words[3].start, 31.5);
  assert.equal(third.words[1].start, 60.5);
  assert.equal(third.words[3].start, 61.5);
});

test('a repeated line placed by ear is copied from, never written over', () => {
  const first = line([word('hold', 10, 10.4), word('on', 10.4, 10.7)]);
  const placed = { ...line([word('hold', 30, 30.4), word('on', 31, 31.4)]), checked: true };
  const stats = reuseRepeats([first, placed], null);

  assert.equal(stats.repeatedLines, 1, 'it is still recognised as the same line');
  assert.equal(stats.wordsMoved, 0);
  assert.equal(placed.words[1].start, 31, 'the person’s placement wins over the repeat');
});

test('the repeat pass runs inside polish and reports its own reason', () => {
  const first = line([word('hold', 10, 10.4), word('on', 10.4, 10.7), word('me', 10.7, 11.2)]);
  const second = line([word('hold', 30, 30.4), word('on', 30.9, 31.05), word('me', 31.05, 31.2)]);
  const result = polishLines([first, second], []);

  assert.equal(result.stats.repeatedLines, 1);
  assert.ok(result.repairs.some(repair => repair.reason === 'repeat'));
  assert.equal(result.lines[1].words[1].start, 30.4);
});

test('a repeat with one changed word reuses the words it shares', () => {
  // Chorus 1, then a chorus that swaps "used to" for "still": the leading "i"
  // and the trailing "love you" are the same delivery, and the later line's
  // copy of them has drifted. The word in between is the variation and keeps
  // whatever timing the aligner gave it.
  const first = line([
    word('i', 10, 10.2), word('used', 10.2, 10.5), word('to', 10.5, 10.7),
    word('love', 10.7, 10.9), word('you', 10.9, 11.2),
  ]);
  const second = line([
    word('i', 30, 30.2), word('still', 30.2, 30.5),
    word('love', 31, 31.15), word('you', 31.15, 31.2),
  ]);
  const stats = reuseRepeats([first, second], null);

  assert.equal(stats.repeatedLines, 1, 'a one-word variation is still the same line');
  assert.equal(stats.wordsMoved, 2);
  assert.equal(second.words[0].start, 30, 'a shared word already in place stays put');
  assert.equal(second.words[1].start, 30.2, 'the word that differs is the aligner’s own');
  close(second.words[2].start, 30.7);
  close(second.words[3].start, 30.9);
  close(second.words[3].end, 31.2);
});

test('a repeat is not matched on its opening words alone', () => {
  // Two shared words out of four is not enough to call this the same line, so
  // the later line is left to the aligner and the repairs as usual.
  const first = line([word('i', 10, 10.4), word('know', 10.4, 10.8), word('why', 10.8, 11.2)]);
  const second = line([
    word('i', 30, 30.4), word('know', 30.4, 30.8), word('the', 30.8, 31), word('way', 31, 31.2),
  ]);
  const stats = reuseRepeats([first, second], null);

  assert.equal(stats.repeatedLines, 0);
  assert.equal(stats.wordsMoved, 0);
  assert.equal(second.words[2].start, 30.8);
});

test('a doubtful source word is not copied onto the repeat', () => {
  // The aligner was unsure of the earlier line's last word, so it is not a
  // prediction: the repeat keeps its own timing for that word even though the
  // two disagree, while the word before it — which the earlier line is sure of
  // — is still re-placed.
  const first = line([word('hold', 10, 10.4), word('on', 10.4, 10.7), word('me', 10.7, 11.2, 0.1)]);
  const second = line([word('hold', 30, 30.4), word('on', 30.9, 31.05), word('me', 31.05, 31.2)]);
  const stats = reuseRepeats([first, second], null);

  assert.equal(stats.repeatedLines, 1);
  assert.equal(stats.wordsMoved, 1, 'only the word the earlier line is sure of moves');
  assert.equal(second.words[1].start, 30.4);
  assert.equal(second.words[2].start, 31.05, 'the doubtful word keeps this run’s timing');
});

test('a re-align puts hand-placed lines back before repairing around them', () => {
  const old = [
    { ...line([word('hold', 10, 10.4), word('on', 10.4, 10.7)]), checked: true },
    line([word('free', 20, 20.3)]),
  ];
  const rerun = [
    line([word('hold', 10.2, 10.6), word('on', 10.9, 11.2)]),
    line([word('free', 20, 20.3)]),
  ];
  const carried = carryChecked(old, rerun);

  assert.equal(carried.kept, 1);
  assert.equal(carried.lines[0].checked, true);
  assert.equal(carried.lines[0].words[1].start, 10.4, 'the person’s timing is restored');
  assert.equal(carried.lines[0].words[1].checked, true, 'and every word of it is an anchor');
  assert.equal(carried.lines[1].words[0].start, 20, 'an unticked line keeps the machine timings');
});

test('a selection run matches hand-placed lines through the crop offset', () => {
  // The crop starts 58 s into the track, so the rerun's times are relative to
  // it: 2.3 s into the crop is 60.3 s into the song, which is the line it is.
  const old = [{ ...line([word('hold', 60, 60.4), word('on', 60.4, 60.7)]), checked: true }];
  const rerun = [line([word('hold', 2.3, 2.7), word('on', 2.9, 3.2)])];
  const carried = carryChecked(old, rerun, 58);

  assert.equal(carried.kept, 1);
  // The restored words stay in the rerun's clock, so the caller's shift of the
  // whole block lands them back at 60.4 s — restoring absolute times here would
  // shift them twice.
  close(carried.lines[0].words[1].start, 2.4);
  assert.equal(carried.lines[0].start, 2);
});

test('a hand-placed line is left alone when the rerun put it somewhere else', () => {
  const old = [{ ...line([word('hold', 10, 10.4), word('on', 10.4, 10.7)]), checked: true }];
  const rerun = [line([word('hold', 30, 30.4), word('on', 30.4, 30.7)])];
  const carried = carryChecked(old, rerun);

  assert.equal(carried.kept, 0);
  assert.equal(carried.skipped, 1);
  assert.equal(carried.lines[0].words[1].start, 30.4);
});

test('a tail may not bleed into a hand-placed line', () => {
  // Both words are a normal length for this section, so neither is flagged as
  // too long — and the second still ends 0.6 s inside the anchor. An anchor is
  // a wall whether or not the tail before it is suspect.
  const lead = line([word('we', 7.8, 9.2), word('hold', 9.2, 10.6)]);
  const anchor = { ...line([word('keep', 10, 12)]), checked: true };
  const { lines, stats, repairs } = polishLines([lead, anchor], []);

  assert.equal(lines[0].words[1].end, 10);
  assert.equal(stats.cutBack, 1);
  assert.equal(repairs.find(r => r.reason === 'cut-back')?.to.end, 10);
  assert.equal(lines[1].words[0].start, 10, 'the anchor is untouched');
  assert.equal(lines[1].words[0].end, 12);
});

test('a word pushed past an anchor keeps its duration', () => {
  // The machine put this word inside the anchor. There is room after the anchor,
  // so the whole span moves instead of collapsing to the 50 ms floor.
  const anchor = { ...line([word('keep', 10, 12)]), checked: true };
  const late = line([word('late', 11, 11.5)]);
  const { lines, repairs } = polishLines([anchor, late], []);

  assert.equal(lines[1].words[0].start, 12);
  assert.equal(lines[1].words[0].end, 12.5, 'half a second stays half a second');
  assert.equal(repairs.find(r => r.reason === 'monotonic')?.to.end, 12.5);
});

test('a selection run is held inside the lines either side of it', () => {
  // The crop's own words were aligned without ever seeing the neighbour lines,
  // so the first is pushed clear of the one before and the last stops at the
  // one after instead of bleeding past it.
  const { lines, stats } = polishLines(
    [line([word('first', 1, 1.4), word('out', 4.9, 5.6)])], [], HOP,
    { bounds: { startSec: 2, endSec: 5.5 } },
  );

  assert.equal(lines[0].words[0].start, 2);
  assert.equal(lines[0].words[0].end, 2.4, 'the duration moves with it');
  assert.equal(lines[0].words[1].end, 5.5, 'the last word stops at the next line');
  assert.equal(stats.cutBack, 1);
});

test('a spliced selection is pushed and trimmed clear of its neighbours', () => {
  const before = line([word('prev', 8, 10)]);
  const after = line([word('next', 30, 30.4)]);
  const block = [line([word('a', 9.5, 9.9), word('b', 29.9, 30.6)])];
  const { lines, pushed, trimmed } = clampToNeighbours(block, before, after);

  assert.equal(pushed, 1);
  assert.equal(trimmed, 1);
  assert.equal(lines[0].words[0].start, 10);
  assert.equal(lines[0].words[0].end, 10.4, 'the push keeps the duration');
  assert.equal(lines[0].words[1].end, 30, 'the tail stops at the next line');
});

test('a seam does not move a hand-placed word either', () => {
  const before = line([word('prev', 8, 10)]);
  const placed = { ...line([word('keep', 9.6, 12)]), checked: true };
  const { lines, pushed } = clampToNeighbours([placed], before, null);

  assert.equal(pushed, 0);
  assert.equal(lines[0].words[0].start, 9.6);
  assert.equal(lines[0].words[0].end, 12);
});

test('a backing line is its own stream, so a harmony overlap is not a sequence error', () => {
  // The harmony starts under the lead's held note. With one cursor across the
  // track the harmony would be pushed to 15.5 and crushed to the floor; with a
  // cursor per layer both keep the times they were aligned with.
  const lead = line([word('hold', 13.9, 14.3), word('the', 14.3, 14.7), word('line', 14.7, 15.5)]);
  const backing: LyricsLine = { ...line([word('ooh', 14.8, 15.2), word('ah', 15.2, 15.6)]), layer: 'backing' };
  const { lines, stats } = polishLines([lead, backing], []);

  assert.equal(stats.monotonicRepairs, 0, 'neither stream pushes the other');
  assert.equal(stats.cutBack, 0);
  assert.equal(lines[0].words[2].end, 15.5, 'the held note keeps its tail');
  assert.equal(lines[1].words[0].start, 14.8, 'the harmony keeps its onset');
  assert.equal(lines[1].words[0].end, 15.2, 'and its whole duration');
});

test('a lead held note is not cut back to a backing onset', () => {
  // Without the layer this long tail collides with the harmony's 14.8 onset and
  // is cut there. The harmony is not a boundary in the lead's stream.
  const lead = line([word('hold', 12.9, 13.3), word('the', 13.3, 13.7), word('line', 13.7, 15.5)]);
  const backing: LyricsLine = { ...line([word('ooh', 14.8, 15.2)]), layer: 'backing' };
  const { lines, stats } = polishLines([lead, backing], []);

  assert.equal(stats.cutBack, 0);
  assert.equal(lines[0].words[2].end, 15.5);
  assert.equal(lines[1].words[0].start, 14.8);
});

test('a word’s own layer wins over its line’s', () => {
  // A shouted word inside a lead line is a second stream too, so it neither
  // moves nor pushes the lead words around it.
  const mixed = line([
    word('lead', 10, 10.4),
    { ...word('back', 10.2, 10.6), layer: 'backing' },
    word('lead2', 10.4, 10.8),
  ]);
  const { lines, stats } = polishLines([mixed], []);

  assert.equal(stats.monotonicRepairs, 0);
  assert.equal(lines[0].words[1].start, 10.2);
  assert.equal(lines[0].words[2].start, 10.4);
});

test('a repeat that lands on an anchor is cut back, then pushed, around it', () => {
  // Case B from the repeat/anchor audit: the reused line's second word is moved
  // under the anchor's start and its third starts under the anchor's end. The
  // cut-back trims the first collision, the monotonic pass moves the second,
  // and the anchor itself is never written.
  const source = line([word('one', 10, 10.4), word('two', 10.4, 10.8), word('three', 10.8, 11.2)]);
  const repeatLine = line([word('one', 30, 30.3), word('two', 30.2, 30.5), word('three', 30.9, 31.5)]);
  const anchor: LyricsLine = { ...line([word('keep', 30.7, 31.4)]), checked: true };

  const { lines, repairs, stats } = polishLines([source, repeatLine, anchor], []);
  const placed = lines.find(l => l.checked)!;
  const later = lines.filter(l => !l.checked).sort((a, b) => a.start - b.start)[1];

  assert.deepEqual([placed.words[0].start, placed.words[0].end], [30.7, 31.4], 'the anchor is never written');
  assert.equal(placed.words[0].checked, undefined, 'and it is not re-tagged by the repair');

  // The repeat moves `two` to the earlier occurrence's offset (1.15x pace)...
  close(later.words[1].start, 30.46);
  // ...the cut-back ends it at the anchor's start...
  close(later.words[1].end, 30.7);
  // ...and the push moves `three` past the anchor's end, keeping its 0.6 s span.
  close(later.words[2].start, 31.4);
  close(later.words[2].end, 32);

  assert.deepEqual(repairs.map(r => r.reason), ['repeat', 'cut-back', 'monotonic'],
    'cut-back handles the word before the anchor, monotonic the word after');
  // The repair says where the word went, not just where it was: this is the
  // record-order fix, pinned so a repeat can never be reported as a no-op again.
  close(repairs[0].to.start, 30.46);
  close(repairs[0].to.end, 30.9);
  assert.equal(stats.anchored, 1);
  assert.equal(stats.wordsMoved, 2);
});

test('a backing-majority section does not flag the lead’s ordinary words', () => {
  // The pooled median follows whichever layer has more words: 12 lead words at
  // 0.40 s against 20 harmony syllables at 0.15 s put the band at 0.325 and
  // flagged every lead word. Each layer is now its own cohort.
  const leadPairs = [
    ['one', 'two'], ['three', 'four'], ['five', 'six'],
    ['seven', 'eight'], ['nine', 'ten'], ['eleven', 'twelve'],
  ];
  const lines: LyricsLine[] = leadPairs.map((pair, i) => ({
    ...line([word(pair[0], 10 + i, 10.4 + i), word(pair[1], 10.4 + i, 10.8 + i)]),
    section: 'Chorus',
  }));
  for (let i = 0; i < 10; i++) {
    const a = String.fromCharCode(97 + i) + 'ah';
    const b = String.fromCharCode(97 + i) + 'oh';
    lines.push({
      ...line([word(a, 20 + i * 0.4, 20.15 + i * 0.4), word(b, 20.15 + i * 0.4, 20.3 + i * 0.4)]),
      section: 'Chorus',
      layer: 'backing',
    });
  }

  const { stats } = polishLines(lines, []);
  assert.equal(stats.wordsFlagged, 0, 'the lead is judged against the lead, not the harmony');
});

test('a two-word harmony line is judged by the harmony elsewhere, not by the lead', () => {
  // The target harmony is its own section, and the ±4-line window holds only
  // four harmony words. Only the same-layer widening reaches the eight the
  // cohort needs; without it the 0.95 s note was over the lead's band.
  const lines: LyricsLine[] = [
    { ...line([word('aah', 10, 10.85), word('aoh', 10.85, 11.7)]), layer: 'backing' },
    line([word('alpha', 12, 12.4), word('beta', 12.4, 12.8)]),
    line([word('gamma', 13, 13.4), word('delta', 13.4, 13.8)]),
    { ...line([word('baah', 14, 14.85), word('baoh', 14.85, 15.7)]), layer: 'backing' },
    line([word('epsilon', 16, 16.4), word('zeta', 16.4, 16.8)]),
    line([word('eta', 17, 17.4), word('theta', 17.4, 17.8)]),
    line([word('iota', 18, 18.4), word('kappa', 18.4, 18.8)]),
    { ...line([word('caah', 20, 20.95), word('caoh', 20.95, 21.8)]), layer: 'backing', section: 'Refrain' },
    line([word('lambda', 23, 23.4), word('mu', 23.4, 23.8)]),
    line([word('nu', 24, 24.4), word('xi', 24.4, 24.8)]),
    line([word('omicron', 25, 25.4), word('pi', 25.4, 25.8)]),
    { ...line([word('daah', 27, 27.85), word('daoh', 27.85, 28.7)]), layer: 'backing' },
    line([word('rho', 30, 30.4), word('sigma', 30.4, 30.8)]),
    line([word('tau', 31, 31.4), word('upsilon', 31.4, 31.8)]),
    line([word('phi', 32, 32.4), word('chi', 32.4, 32.8)]),
  ];
  lines.forEach(l => { if (!l.section) l.section = 'Chorus'; });

  const { stats } = polishLines(lines, []);
  assert.equal(stats.wordsFlagged, 0, 'the held harmony note is within the harmony’s own band');
});

test('a line placed by ear is an anchor, not a measurement', () => {
  // The ticked line overlaps the machine's word after it. The person's timing is
  // kept exactly, and the machine's word is the one that moves.
  const placed = { ...line([word('keep', 1, 2)]), checked: true };
  const machine = line([word('push', 1.5, 1.9)]);
  const result = polishLines([placed, machine], []);

  assert.equal(result.lines[0].words[0].start, 1);
  assert.equal(result.lines[0].words[0].end, 2);
  assert.equal(result.lines[1].words[0].start, 2);
  assert.equal(result.stats.anchored, 1);
});

test('digits are neither Latin nor foreign', () => {
  // `\p{N}` is not `Script=Latin`, so counting digits made a year a non-Latin
  // word and could push a romanised lyric over the Whisper-only threshold.
  assert.equal(nonRomanisedRatio('Hello 2024 déjà vu'), 0);
  assert.equal(nonRomanisedRatio('夜に駆ける'), 1);
  assert.equal(nonRomanisedRatio('Hello 夜'), 1 / 6);
});

function word(text: string, start: number, end: number, confidence = 1): LyricsWord {
  return { word: text, start, end, confidence, source: 'matched' };
}

/** Copied offsets are arithmetic, so compare them with a tolerance. */
function close(actual: number, expected: number, because = ''): void {
  assert.ok(
    Math.abs(actual - expected) < 1e-6,
    `${actual} should be ${expected}${because ? ` (${because})` : ''}`,
  );
}

function line(words: LyricsWord[], section?: string): LyricsLine {
  return {
    start: words[0].start,
    end: words[words.length - 1].end,
    text: words.map(w => w.word).join(' '),
    ...(section ? { section } : {}),
    words,
  };
}

/** A voicing envelope: `level` until `until` seconds, near silence after. */
function voiced(seconds: number, until: number, level = 0.5): number[] {
  const frames = Math.ceil(seconds / HOP);
  const env = new Array(frames).fill(0.001);
  for (let i = 0; i < Math.min(frames, Math.floor(until / HOP)); i++) env[i] = level;
  return env;
}

/** Consecutive words of the given durations, so sections can be built quickly. */
function sequence(durations: number[], from = 0, prefix = 'w'): LyricsWord[] {
  let at = from;
  return durations.map((duration, index) => {
    const next = word(`${prefix}${index}`, at, at + duration);
    at += duration;
    return next;
  });
}

/** Six short words, one long one, then a word after a further gap. */
function heldNote(nextStart: number): LyricsWord[] {
  const lead = sequence([0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 0, 'lead');
  return [...lead, word('held', 2.4, 4.9), word('after', nextStart, nextStart + 0.4)];
}

test('the aligner script check passes Latin, accents included, and rejects CJK', () => {
  assert.equal(nonRomanisedRatio('hold the line'), 0);
  assert.equal(nonRomanisedRatio('café déjà vu'), 0);
  assert.equal(nonRomanisedRatio(''), 0);
  assert.ok(nonRomanisedRatio('君の声が聞こえる') > 0.9);
  // Section markers are labels, not lyrics.
  assert.equal(nonRomanisedRatio('[Chorus]\nlet it go'), 0);
});

test('a word that starts before its neighbour ends is pushed forward', () => {
  const { lines, stats, repairs } = polishLines([
    line([word('one', 1, 2), word('two', 1.5, 2.5)]),
  ]);
  assert.equal(stats.monotonicRepairs, 1);
  assert.equal(lines[0].words[1].start, 2);
  assert.ok(lines[0].words[1].end > 2);
  assert.equal(repairs[0].reason, 'monotonic');
});

test('a pass switched off in the settings does not run', () => {
  // `b` starts before `a` ends. The monotonic pass is the one that would push
  // it later; with that pass off the overlap stays exactly as written.
  const words = [word('a', 1, 1.4), word('b', 1.2, 1.6)];
  const { lines, stats } = polishLines(
    [line(words)], [], HOP, { passes: { monotonic: false } },
  );
  assert.equal(lines[0].words[1].start, 1.2, 'the out-of-order word is left alone');
  assert.equal(stats.monotonicRepairs, 0);
});

test('a held note that keeps sounding is melisma, not a swallowed gap', () => {
  // The tail runs from the expected 2.8 s to 4.9 s and is voiced throughout.
  // Its next onset is far away, so nothing about it says "stray tail".
  const { lines, stats } = polishLines([line(heldNote(5.2))], voiced(6, 5.6), HOP);
  const held = lines[0].words.find(w => w.word === 'held')!;
  assert.equal(held.end, 4.9);
  assert.equal(stats.cutBack, 0);
  assert.equal(stats.snapped, 0);
  assert.equal(stats.wordsFlagged, 1, 'it is long, so it is flagged, but left alone');
});

test('a tail that collides with the next onset is cut back to that onset', () => {
  const { lines, stats, repairs } = polishLines([line(heldNote(4.7))], voiced(6, 6), HOP);
  const held = lines[0].words.find(w => w.word === 'held')!;
  assert.equal(held.end, 4.7);
  assert.equal(stats.cutBack, 1);
  assert.equal(repairs.find(r => r.reason === 'cut-back')?.to.end, 4.7);
});

test('a long tail over silence is cut back to its expected end', () => {
  // Voiced only until 2.6 s; the expected end is 2.4 + 0.4 = 2.8 s.
  const { lines, stats } = polishLines([line(heldNote(5.2))], voiced(6, 2.6), HOP);
  const held = lines[0].words.find(w => w.word === 'held')!;
  assert.ok(Math.abs(held.end - 2.8) < 1e-6, `expected 2.8, got ${held.end}`);
  assert.equal(stats.cutBack, 1);
});

test('a short staccato syllable is not flagged on duration alone', () => {
  const words = [...sequence([0.4, 0.4, 0.4, 0.4, 0.4], 0, 's'), word('short', 2, 2.12)];
  const { lines, stats } = polishLines([line(words)], voiced(3, 3), HOP);
  const short = lines[0].words.find(w => w.word === 'short')!;
  assert.equal(stats.wordsFlagged, 0);
  assert.equal(stats.snapped, 0);
  assert.equal(short.start, 2);
  assert.equal(short.end, 2.12);
});

test('sections are judged against their own pace, not a blended global median', () => {
  // A slow verse at 0.9 s per word and a fast chorus at 0.15 s, plus one word
  // in the chorus that really is long. Only that one should be flagged.
  const verse = sequence(new Array(6).fill(0.9), 0, 'v');
  const chorus = sequence([...new Array(20).fill(0.15), 0.9], verse[verse.length - 1].end, 'c');
  const { stats } = polishLines([line(verse, 'Verse'), line(chorus, 'Chorus')], [], HOP);
  assert.equal(stats.sections.length, 2);
  assert.deepEqual(stats.sections.map(s => s.section), ['Verse', 'Chorus']);
  assert.ok(Math.abs(stats.sections[0].medianSec - 0.9) < 1e-9);
  assert.ok(Math.abs(stats.sections[1].medianSec - 0.15) < 1e-9);
  assert.equal(stats.wordsFlagged, 1, 'the slow verse is normal for its own section');
});

test('a dry run reports the diff and returns the timings untouched', () => {
  const { lines, stats, repairs } = polishLines([line(heldNote(4.7))], voiced(6, 6), HOP, { dryRun: true });
  const held = lines[0].words.find(w => w.word === 'held')!;
  assert.equal(held.end, 4.9, 'dry run writes nothing');
  assert.equal(stats.cutBack, 1, 'but it reports what it would have done');
  assert.deepEqual(repairs[0].from, { start: 2.4, end: 4.9 });
  assert.deepEqual(repairs[0].to, { start: 2.4, end: 4.7 });
});

test('a low-confidence word moves onto a nearby energy rise, inside its neighbours', () => {
  // 50 ms hops; a clear rise at 2.30 s. The 0.02 s word at 2.20 s is the only
  // low-confidence word and is the only thing eligible to move.
  const env = new Array(80).fill(0.001);
  for (let i = 46; i < 80; i++) env[i] = 0.5;
  const words = [
    word('a', 0, 0.5), word('b', 0.5, 1.0), word('c', 1.0, 1.5),
    word('tiny', 2.2, 2.22, 0.1), word('d', 2.6, 3.0),
  ];
  const { lines, stats, repairs } = polishLines([line(words)], env, HOP);
  const tiny = lines[0].words.find(w => w.word === 'tiny')!;
  assert.equal(stats.snapped, 1);
  assert.ok(tiny.start >= 2.25 && tiny.start < 2.6, `got ${tiny.start}`);
  assert.ok(tiny.end <= 2.6);
  assert.equal(repairs.find(r => r.reason === 'snap')?.word, 3);
});

test('the anchor audit measures drift and names the worst words, moving nothing', () => {
  const anchors = [line(sequence([0.4, 0.4, 0.4, 0.4], 1, 'w'))];
  const exact = [line(sequence([0.4, 0.4, 0.4, 0.4], 1, 'w'))];
  const clean = anchorDrift(exact, anchors);
  assert.equal(clean.medianMs, 0);
  assert.equal(clean.outliers, 0);
  assert.equal(clean.anchors, 4);

  const early = [line(sequence([0.4, 0.4, 0.4, 0.4], 0.88, 'w'))];
  const shifted = anchorDrift(early, anchors);
  assert.equal(shifted.medianMs, 120);
  assert.equal(shifted.outliers, 0, '120 ms is inside the tolerance');

  const oneBad = [line(sequence([0.4, 0.4, 0.4, 0.4], 1, 'w'))];
  oneBad[0].words[2] = { ...oneBad[0].words[2], start: oneBad[0].words[2].start + 0.4 };
  const flagged = anchorDrift(oneBad, anchors);
  assert.equal(flagged.outliers, 1);
  assert.equal(flagged.worst[0].word, 2);
  assert.equal(flagged.worst[0].driftMs, 400);
});

test('a chorus that comes back is scored apart from the chorus it repeats', () => {
  const chorus = () => line([
    word('hold', 10, 10.4), word('on', 10.4, 10.7), word('me', 10.7, 11.2),
  ], 'Chorus');
  const verse = () => line([word('low', 5, 5.4), word('tide', 5.4, 5.9)], 'Verse');
  const anchors = [chorus(), verse(), chorus()];
  const ctc = [chorus(), verse(), chorus()];
  // The second chorus lands 300 ms late; the first and the verse are exact.
  ctc[2].words = ctc[2].words.map(w => ({ ...w, start: w.start + 0.3, end: w.end + 0.3 }));
  const drift = anchorDrift(ctc, anchors);

  const first = drift.sections.find(entry => entry.section === 'Chorus' && entry.occurrence === 1)!;
  const second = drift.sections.find(entry => entry.section === 'Chorus' && entry.occurrence === 2)!;
  assert.equal(first.outliers, 0, 'the first chorus is on time');
  assert.equal(second.outliers, 3, 'the repeat is the row a corridor would have fixed');
  assert.equal(second.medianMs, 300);
  assert.equal(second.words, 3);
  assert.equal(drift.sections.find(entry => entry.section === 'Verse')?.words, 2);
  assert.equal(drift.medianMs, 0, 'the whole-track median still hides the repeat');
  assert.equal(drift.worst[0].section, 'Chorus', 'a worst entry says which section it came from');
});

test('the pooled report adds up sections across tracks and puts the worst first', () => {
  const track = (medianMs: number, outliers: number, sections: AnchorDrift['sections']): AnchorDrift => ({
    anchors: 10, words: 10, medianMs, p90Ms: medianMs, outliers, worst: [], sections,
  });
  const summary = summariseDrift([
    {
      track: 'a', drift: track(60, 1, [
        { section: 'Chorus', occurrence: 1, words: 6, medianMs: 40, p90Ms: 90, outliers: 0 },
        { section: 'Chorus', occurrence: 2, words: 6, medianMs: 260, p90Ms: 400, outliers: 4 },
      ]),
    },
    {
      track: 'b', drift: track(80, 2, [
        { section: 'Chorus', occurrence: 1, words: 4, medianMs: 60, p90Ms: 120, outliers: 1 },
        { section: 'Chorus', occurrence: 2, words: 4, medianMs: 200, p90Ms: 300, outliers: 3 },
      ]),
    },
  ]);

  assert.equal(summary.tracks, 2);
  assert.equal(summary.words, 20, 'pairs pooled, not words aligned');
  assert.equal(summary.outliers, 3, 'the per-track outlier counts, added up');
  assert.equal(summary.medianMs, 60, 'nearest-rank median of the per-track medians, one track one vote');
  assert.equal(summary.worstP90Ms, 80, 'the tail of the worst track, not an average');

  const second = summary.sections.find(entry => entry.occurrence === 2)!;
  assert.equal(second.tracks, 2);
  assert.equal(second.words, 10);
  assert.equal(second.outliers, 7);
  assert.equal(second.medianMs, 200, 'the median of the per-track section medians');
  assert.equal(summary.sections[0].occurrence, 2, 'the row with the most outliers leads');
});

test('a range run keeps its sections on the lines they belong to, not the times', () => {
  const before = [line([word('a', 1, 1.4)], 'Chorus'), line([word('b', 2, 2.4)], 'Verse')];
  // The run returned the same two lines in the other order.
  const after = [line([word('b', 1, 1.4)]), line([word('a', 2, 2.4)])];
  const { lines, matched } = carrySections(before, after);
  assert.deepEqual(lines.map(l => l.section), ['Verse', 'Chorus']);
  assert.deepEqual(matched, { text: 2, index: 0, time: 0 });
});

test('a rewritten line falls back to its position, and says it did', () => {
  const before = [line([word('a', 1, 1.4)], 'Chorus'), line([word('b', 2, 2.4)], 'Verse')];
  const after = [line([word('a', 1, 1.4)]), line([word('fixed', 2, 2.4)])];
  const { lines, matched } = carrySections(before, after);
  assert.deepEqual(lines.map(l => l.section), ['Chorus', 'Verse']);
  assert.deepEqual(matched, { text: 1, index: 1, time: 0 });
});

test('a repeated line inside the selection keeps one label per occurrence', () => {
  const before = [
    line([word('la', 1, 1.4)], 'Chorus'), line([word('la', 5, 5.4)], 'Chorus'),
  ];
  const after = [line([word('la', 1, 1.4)]), line([word('la', 5, 5.4)])];
  const { lines } = carrySections(before, after);
  assert.deepEqual(lines.map(l => l.section), ['Chorus', 'Chorus']);
});

test('with a different line count and no text match, the nearest time is used and reported', () => {
  const before = [line([word('a', 1, 1.4)], 'Chorus'), line([word('b', 5, 5.4)], 'Bridge')];
  const after = [line([word('c', 4.9, 5.3)])];
  const { lines, matched } = carrySections(before, after);
  assert.equal(lines[0].section, 'Bridge');
  assert.deepEqual(matched, { text: 0, index: 0, time: 1 });
});

test('a word one side dropped does not shift the rest of the audit', () => {
  const anchors = [line(sequence([0.4, 0.4, 0.4], 1, 'w'))];
  const dropped = [line(sequence([0.4, 0.4], 1, 'w'))];
  dropped[0].words.push({ word: 'w2', start: 2.2, end: 2.6, confidence: 1, source: 'matched' });
  const result = anchorDrift(dropped, anchors);
  assert.equal(result.anchors, 3);
  assert.equal(result.words, 3);
  assert.equal(result.medianMs, 0, 'the remaining words are compared to their own partners');
});

test('line bounds and the line order follow the repaired words', () => {
  const { lines } = polishLines([
    line([word('late', 9, 9.4)]),
    line([word('early', 1, 1.4)]),
  ]);
  assert.deepEqual(lines.map(l => l.text), ['early', 'late']);
  assert.deepEqual(lines.map(l => [l.start, l.end]), [[1, 1.4], [9, 9.4]]);
});

test('an overlap under the duration floor is pulled clear of the anchor', () => {
  // The machine word is only 60 ms long and 10 ms of it lands inside the hand
  // placed line. Trimming would leave a 10 ms word, and leaving it would put
  // two words on the bar at once, so the whole span slides back instead.
  const lead = line([word('a', 2, 2.9), word('hold', 2.99, 3.05)]);
  const anchor = { ...line([word('keep', 3, 3.2)]), checked: true };
  const { lines, stats, repairs } = polishLines([lead, anchor], []);

  assert.equal(lines[0].words[1].end, 3, 'the tail stops at the checked start');
  close(lines[0].words[1].start, 2.94, 'the 60 ms span moves, it does not shrink');
  assert.equal(stats.cutBack, 1);
  const moved = repairs.find(r => r.reason === 'cut-back')!;
  close(moved.to.start, 2.94);
  close(moved.to.end, 3);
  assert.deepEqual([lines[1].words[0].start, lines[1].words[0].end], [3, 3.2], 'the anchor is untouched');
});

test('a word with only part of its length free is cut back to the anchor', () => {
  // 0.45 s of word, 0.1 s of room before the anchor: too short to keep the
  // length, long enough to keep a legal word.
  const lead = line([word('a', 2, 2.9), word('hold', 2.95, 3.4)]);
  const anchor = { ...line([word('keep', 3, 3.2)]), checked: true };
  const { lines, stats } = polishLines([lead, anchor], []);

  assert.equal(lines[0].words[1].end, 3);
  assert.ok(lines[0].words[1].start >= 2.9, `got ${lines[0].words[1].start}`);
  assert.ok(lines[0].words[1].end - lines[0].words[1].start >= 0.05 - 1e-9, 'still a legal word');
  assert.equal(stats.cutBack, 1);
});

test('a collapsed repeat span is not used as the repeat’s pace', () => {
  // The aligner swallowed the second occurrence: four words in 0.19 s. Scaling
  // by that span put the last word 1.1 s from where the first occurrence says
  // it belongs, so a span whose words are not singable lengths is ignored.
  const first = line([word('we', 0, 0.45), word('could', 0.5, 0.95), word('run', 1, 1.45), word('away', 1.5, 1.95)]);
  const collapsed = line([word('we', 10, 10.04), word('could', 10.05, 10.09), word('run', 10.1, 10.14), word('away', 10.15, 10.19)]);
  reuseRepeats([first, collapsed], null);

  // Only the last word can move: the guard that keeps a reuse from overlapping
  // its neighbour still reads the earlier words' collapsed starts, so the words
  // before it stay where the aligner put them. What the collapse guard changes
  // is where the last word lands.
  close(collapsed.words[3].start, 11.5, 'the first occurrence’s own spacing');

  // A fast occurrence is not a collapse, but its span is still outside the band
  // (0.41 of the first one's pace), so it is not a measurement either: the
  // earlier pace is used whole. It is still reused — the band was never a
  // rejection, and 0.85 of a 2.4x error just moves the word by 15% of the
  // mistake.
  const fast = line([word('we', 20, 20.2), word('could', 20.2, 20.4), word('run', 20.4, 20.6), word('away', 20.6, 20.8)]);
  reuseRepeats([first, fast], null);
  close(fast.words[3].start, 21.5, 'the first occurrence’s own spacing, not a clamped 0.85');

  // Inside the band the measured pace is used, so a repeat sung 13% slower is
  // copied at its own tempo instead of the first occurrence's.
  const slower = line([
    word('we', 30, 30.2), word('could', 30.55, 30.75),
    word('run', 31.1, 31.3), word('away', 31.9, 32.2),
  ]);
  reuseRepeats([first, slower], null);
  close(slower.words[3].start, 30 + 1.5 * (2.2 / 1.95), 'scaled by the 1.13 pace the two lines share');
});

test('a line that only resembles one in another section is not its repeat', () => {
  // A shared prefix and a shared suffix with an unrelated middle: on text alone
  // this is a near repeat, and copying the chorus's offsets onto the verse's
  // words is exactly the cross-section match the section guard exists to stop.
  const chorus = line([
    word('i', 10, 10.4), word('know', 10.4, 10.8), word('what', 10.8, 11.2),
    word('you', 11.2, 11.6), word('mean', 11.6, 12),
  ], 'Chorus');
  const verse = line([
    word('i', 30, 30.4), word('know', 30.4, 30.8), word('the', 30.8, 31),
    word('you', 31, 31.4), word('mean', 31.4, 31.8),
  ], 'Verse 2');
  const stats = reuseRepeats([chorus, verse], null);

  assert.equal(stats.repeatedLines, 0, 'the same words in another section are another passage');
  assert.equal(stats.wordsMoved, 0);
  assert.equal(verse.words[2].start, 30.8, 'the verse keeps the timing the aligner gave it');
});

test('a near repeat inside one section is still the same passage', () => {
  // The label is on the first line only; the line under it belongs to the same
  // section by carry-forward, which is how a sidecar actually records a chorus.
  const first = line([
    word('i', 10, 10.2), word('used', 10.2, 10.5), word('to', 10.5, 10.7),
    word('love', 10.7, 10.9), word('you', 10.9, 11.2),
  ], 'Chorus');
  const second = line([
    word('i', 30, 30.2), word('still', 30.2, 30.5),
    word('love', 31, 31.15), word('you', 31.15, 31.2),
  ]);
  const stats = reuseRepeats([first, second], null);

  assert.equal(stats.repeatedLines, 1);
  assert.ok(stats.wordsMoved > 0, 'the shared words are copied from the earlier line');
  close(second.words[2].start, 30.7);
});

test('an exact repeat still copies across a section label', () => {
  // A chorus returning in the outro is the same lyric even though the label
  // changed; exact repeats are found by their text before the section guard is
  // consulted, so the guard never disables reuse.
  const chorus = line([word('hold', 10, 10.4), word('on', 10.4, 10.7), word('me', 10.7, 11.2)], 'Chorus');
  const outro = line([word('hold', 40, 40.4), word('on', 40.9, 41.05), word('me', 41.05, 41.2)], 'Outro');
  const stats = reuseRepeats([chorus, outro], null);

  assert.equal(stats.repeatedLines, 1);
  assert.equal(outro.words[1].start, 40.4, 'the earlier occurrence still supplies the offsets');
});

test('repairs name the line they describe after the output is sorted', () => {
  // The input arrives out of order, both lines have a word starting before its
  // neighbour ends, and the output is sorted — so the repair indices have to be
  // remapped with it or the diagnostics point at the wrong line.
  const late = line([word('second', 10, 10.6), word('line', 10.2, 10.5)]);
  const early = line([word('first', 2, 2.6), word('line', 2.2, 2.5)]);
  const { lines, repairs } = polishLines([late, early], []);

  assert.deepEqual(lines.map(l => l.text), ['first line', 'second line']);
  assert.equal(repairs.length, 2);
  for (const repair of repairs) {
    const named = lines[repair.line].words[repair.word];
    assert.deepEqual(
      { start: named.start, end: named.end }, repair.to,
      `repair on "${lines[repair.line].text}" was filed against line ${repair.line}`,
    );
  }
  assert.equal(lines[repairs.find(r => r.from.start === 10.2)!.line].text, 'second line');
  assert.equal(lines[repairs.find(r => r.from.start === 2.2)!.line].text, 'first line');
});
