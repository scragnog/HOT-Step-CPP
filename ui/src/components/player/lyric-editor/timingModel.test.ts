import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  cloneLines, lineAtOrBefore, mergeLineAt, moveWordTo, pageTargetTime, resizeWordEdge,
  shiftEveryWord, shiftFrom, splitLineAt, withBounds, type EditLine,
} from './timingModel';

function line(words: Array<[string, number, number]>, section?: string): EditLine {
  return {
    start: words[0][1],
    end: words[words.length - 1][2],
    text: words.map(w => w[0]).join(' '),
    ...(section ? { section } : {}),
    words: words.map(([word, start, end]) => ({ word, start, end })),
  };
}

const twoLines = (): EditLine[] => [
  line([['one', 1, 1.4], ['two', 1.4, 1.8]], 'verse'),
  line([['three', 2, 2.4], ['four', 2.4, 2.8]], 'chorus'),
];

describe('cloning a sidecar', () => {
  it('keeps a line with text but no words, and drops an empty one', () => {
    const lines = cloneLines([
      { start: 1, end: 2, text: 'first', words: [{ word: 'first', start: 1, end: 2 }] },
      { start: 3, end: 4, text: 'untimed line' },
      { start: 5, end: 6, text: '   ' },
    ]);
    assert.deepEqual(lines.map(l => l.text), ['first', 'untimed line']);
    assert.deepEqual(lines[1].words, []);
  });

  it('drops blank words and keeps their neighbours', () => {
    const lines = cloneLines([{ start: 0, end: 1, words: [{ word: '  ', start: 0, end: 0.4 }, { word: 'kept', start: 0.4, end: 1 }] }]);
    assert.deepEqual(lines[0].words.map(w => w.word), ['kept']);
  });
});

describe('moving a word', () => {
  it('stops at the end of the previous word', () => {
    const lines = moveWordTo(twoLines(), 0, 1, 0.2);
    assert.equal(lines[0].words[1].start, 1.4);
  });

  it('clamps the start so a long word still ends before its neighbour', () => {
    // Span 0.4 into a gap that ends at 1.4 leaves 1.0 as the latest onset.
    const lines = moveWordTo(twoLines(), 0, 0, 1.2);
    const [first, second] = lines[0].words;
    assert.equal(first.start, 1);
    assert.ok(first.end! <= second.start);
  });

  it('cannot cross into the next line', () => {
    const lines = moveWordTo(twoLines(), 0, 1, 9);
    assert.ok(lines[0].words[1].start! <= 2);
  });
});

describe('resizing a word', () => {
  it('stops the right edge at the next word when Auto shift is off', () => {
    const res = resizeWordEdge(twoLines(), 0, 0, 'right', 2.5, { autoShift: false, total: 10 });
    assert.equal(res?.end, 1.4);
    assert.equal(res?.lines[0].words[1].start, 1.4);
  });

  it('pushes the words after it when Auto shift is on', () => {
    const res = resizeWordEdge(twoLines(), 0, 0, 'right', 1.6, { autoShift: true, total: 10 });
    assert.ok(res);
    assert.equal(res?.end, 1.6);
    assert.ok(Math.abs(res.lines[0].words[1].start - 1.6) < 0.5 * 10 ** -3);
    assert.ok(Math.abs(res.lines[1].words[0].start - 2.2) < 0.5 * 10 ** -3);
  });

  it('refuses to push words past the end of the audio', () => {
    // Two seconds of room in the file, none of it after the last word.
    const res = resizeWordEdge(twoLines(), 0, 0, 'right', 2, { autoShift: true, total: 3 });
    assert.equal(res?.noRoom, true);
    assert.equal(res?.end, 1.4);
    assert.equal(res?.lines[1].words[0].start, 2);
  });

  it('never moves the left edge before the previous word', () => {
    const res = resizeWordEdge(twoLines(), 0, 1, 'left', 0, { autoShift: true, total: 10 });
    assert.equal(res?.start, 1.4);
  });
});

describe('shifting', () => {
  it('moves every word and keeps line bounds derived from them', () => {
    const lines = shiftEveryWord(twoLines(), 0.5);
    assert.equal(lines[0].start, 1.5);
    assert.equal(lines[0].text, 'one two');
    assert.equal(withBounds(lines[1]).start, 2.5);
  });

  it('ripples only from the selection onwards', () => {
    const lines = shiftFrom(twoLines(), 0, 1, 1);
    assert.equal(lines[0].words[0].start, 1);
    assert.equal(lines[0].words[1].start, 2.4);
    assert.equal(lines[1].words[0].start, 3);
  });
});

describe('splitting and merging lines', () => {
  it('splits at the chosen word and keeps both halves timed', () => {
    const lines = splitLineAt(twoLines(), 0, 1);
    assert.notEqual(lines, null);
    assert.equal(lines![0].text, 'one');
    assert.equal(lines![1].text, 'two');
    assert.equal(lines![1].start, 1.4);
    assert.equal(lines!.length, 3);
  });

  it('refuses a cut at the first word', () => {
    assert.equal(splitLineAt(twoLines(), 0, 0), null);
  });

  it('merges into the line above, sorted by time', () => {
    const merged = mergeLineAt(twoLines(), 1, -1);
    assert.equal(merged?.at, 0);
    assert.equal(merged?.lines.length, 1);
    assert.deepEqual(merged?.lines[0].words.map(w => w.word), ['one', 'two', 'three', 'four']);
    assert.equal(merged?.lines[0].start, 1);
    assert.equal(merged?.lines[0].end, 2.8);
  });

  it('refuses a merge with no neighbour', () => {
    assert.equal(mergeLineAt(twoLines(), 0, -1), null);
    assert.equal(mergeLineAt(twoLines(), 1, 1), null);
  });
});

describe('paging the timeline', () => {
  const wide: EditLine[] = [
    line([['a', 1, 1.2], ['b', 2, 2.2], ['c', 3, 3.2], ['d', 4, 4.2]]),
  ];

  it('pages forward so the last word on screen becomes the first', () => {
    assert.equal(pageTargetTime(wide, 1, { scrollLeft: 0, viewportPx: 250, pxPerSec: 100 }), 2);
  });

  it('pages forward again from the new window', () => {
    assert.equal(pageTargetTime(wide, 1, { scrollLeft: 200, viewportPx: 500, pxPerSec: 100 }), 4);
  });

  it('has no forward target once the last word is at the left edge', () => {
    assert.equal(pageTargetTime(wide, 1, { scrollLeft: 400, viewportPx: 500, pxPerSec: 100 }), null);
  });

  it('pages back, always to an earlier time than the current left edge', () => {
    const target = pageTargetTime(wide, -1, { scrollLeft: 300, viewportPx: 100, pxPerSec: 100 });
    assert.equal(target, 2);
    assert.ok(target! * 100! < 300);
  });

  it('pages back to the word before the window when none is visible', () => {
    assert.equal(pageTargetTime(wide, -1, { scrollLeft: 500, viewportPx: 100, pxPerSec: 100 }), 4);
  });

  it('pages back to the start of the track near zero', () => {
    assert.equal(pageTargetTime(wide, -1, { scrollLeft: 50, viewportPx: 200, pxPerSec: 100 }), 0);
  });

  it('has nowhere to go with no viewport', () => {
    assert.equal(pageTargetTime(wide, -1, { scrollLeft: 0, viewportPx: 0, pxPerSec: 100 }), null);
  });
});

describe('playhead lookup', () => {
  it('finds the last line that has started', () => {
    assert.equal(lineAtOrBefore(twoLines(), 0), -1);
    assert.equal(lineAtOrBefore(twoLines(), 1.9), 0);
    assert.equal(lineAtOrBefore(twoLines(), 99), 1);
  });
});
