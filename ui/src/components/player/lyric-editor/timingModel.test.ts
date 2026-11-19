import { describe, expect, it } from 'vitest';
import assert from 'node:assert/strict';
import {
  addWordAfter, adlibRefs, cloneLines, deleteWords, doubtfulRefs, dragSelection, lineAtOrBefore, lineSpan,
  mergeLineAt, moveWordTo,
  orderSpan, pageTargetTime, resizeWordEdge, selectedKeys, selectedRefs, selectionPrevEnd,
  setLineChecked, setLineSection, setWordText, shiftEveryWord, shiftFrom, shiftSelection, spanLineRange,
  addSectionAt, changedProposedRows, linesInSpan, mergeProposed, planLineText, removeSectionAt, sectionRuns,
  spanTimeRange, spanWordCount, splitCutAt, splitLineAt,
  timingsLookRough,
  withBounds, type EditLine,
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

describe('adding and deleting sections', () => {
  it('lists a run per place a label starts, at that line', () => {
    const lines: EditLine[] = [
      line([['a', 1, 1.2]], 'verse'),
      line([['b', 2, 2.2]]),
      line([['c', 3, 3.2]], 'chorus'),
      line([['d', 4, 4.2]]),
      line([['e', 5, 5.2]], 'verse'),
      line([['f', 6, 6.2]]),
      line([['g', 7, 7.2]], 'chorus'),
    ];
    expect(sectionRuns(lines)).toEqual([
      { section: 'verse', from: 0 },
      { section: 'chorus', from: 2 },
      { section: 'verse', from: 4 },
      { section: 'chorus', from: 6 },
    ]);
  });

  it('starts a section on the line when the playhead is before its first word', () => {
    const result = addSectionAt(twoLines(), 1, 1.9, ' Chorus ');
    expect(result?.at).toBe(1);
    expect(result?.lines.map(l => l.section)).toEqual(['verse', 'Chorus']);
    expect(result?.lines).toHaveLength(2);
  });

  it('splits the line at the playhead so the section gets its own line', () => {
    // 2.2 is between `three` (2.0) and `four` (2.4): inside the line, so the
    // boundary needs a row of its own. 2.5 would be past the line, i.e. 2.5 is
    // the next line's business and no split happens.
    const result = addSectionAt(twoLines(), 1, 2.2, 'Bridge');
    expect(result?.at).toBe(2);
    expect(result?.lines.map(l => l.section)).toEqual(['verse', 'chorus', 'Bridge']);
    // The split moved no word: every time on the sheet is the one it had.
    const times = result!.lines.flatMap(l => l.words.map(w => [w.word, w.start, w.end]));
    expect(times).toEqual([
      ['one', 1, 1.4], ['two', 1.4, 1.8],
      ['three', 2, 2.4],
      ['four', 2.4, 2.8],
    ]);
    expect(result?.lines.map(l => l.words.length)).toEqual([2, 1, 1]);
  });

  it('refuses an empty name or a line that is not there', () => {
    expect(addSectionAt(twoLines(), 0, 1, '   ')).toBeNull();
    expect(addSectionAt(twoLines(), 9, 1, 'Chorus')).toBeNull();
  });

  it('deletes the whole run, so a repeated label does not survive and resurface', () => {
    const lines: EditLine[] = [
      line([['a', 1, 1.2]], 'chorus'),
      line([['b', 2, 2.2]], 'chorus'),
      line([['c', 3, 3.2]], 'verse'),
    ];
    const next = removeSectionAt(lines, 0)!;
    expect(next.map(l => l.section)).toEqual([undefined, undefined, 'verse']);
    expect(sectionRuns(next).map(run => run.from)).toEqual([2]);
    expect(next[0].words[0].start).toBe(1);
  });

  it('stops at the next different label, so a later repeat is its own section', () => {
    const lines: EditLine[] = [
      line([['a', 1, 1.2]], 'chorus'),
      line([['b', 2, 2.2]], 'verse'),
      line([['c', 3, 3.2]], 'chorus'),
    ];
    const next = removeSectionAt(lines, 0)!;
    expect(next.map(l => l.section)).toEqual([undefined, 'verse', 'chorus']);
    expect(sectionRuns(next).map(run => run.from)).toEqual([1, 2]);
  });

  it('deletes only the label, leaving every word time alone', () => {
    const removed = removeSectionAt(twoLines(), 1);
    expect(removed?.map(l => l.section)).toEqual(['verse', undefined]);
    expect(removed?.[1].words).toEqual(twoLines()[1].words);
    expect(removeSectionAt(twoLines(), 0)).not.toBeNull();
    const unlabelled = removeSectionAt(removeSectionAt(twoLines(), 0)!, 0);
    expect(unlabelled).toBeNull();
  });
});

describe('rough timings', () => {
  it('is rough when every word sits on one instant', () => {
    const piled: EditLine[] = [line([['a', 0, 0], ['b', 0, 0]]), line([['c', 0, 0], ['d', 0, 0]])];
    expect(timingsLookRough(piled, 180)).toBe(true);
  });

  it('is rough when most words have no length and the rest do', () => {
    const mostly: EditLine[] = [line([['a', 1, 1.3], ['b', 0, 0], ['c', 0, 0]])];
    expect(timingsLookRough(mostly, 180)).toBe(true);
  });

  it('is rough when the words stop inside the first fifth of a long track', () => {
    expect(timingsLookRough(twoLines(), 300)).toBe(true);
  });

  it('leaves a placed track alone, including a short one and one with no words', () => {
    expect(timingsLookRough(twoLines(), 4)).toBe(false);
    expect(timingsLookRough([], 180)).toBe(false);
    expect(timingsLookRough([line([['only', 3, 3.4]])], 180)).toBe(false);
  });
});

describe('doubtful words', () => {
  it('lists low-confidence words only, and treats a missing probability as unknown', () => {
    const lines: EditLine[] = [
      {
        start: 0, end: 0.9, text: 'a b',
        words: [
          { word: 'a', start: 0, end: 0.4, confidence: 0.2, source: 'whisper' },
          { word: 'b', start: 0.4, end: 0.9, confidence: 0.95, source: 'matched' },
        ],
      },
      {
        start: 1, end: 1.9, text: 'c d',
        words: [
          // No probability reported: not a doubt, just an absence.
          { word: 'c', start: 1, end: 1.4 },
          // Heard but not in the sheet, and heard clearly: marked, not queued.
          { word: 'd', start: 1.4, end: 1.9, confidence: 0.9, source: 'ad-lib' },
        ],
      },
    ];
    expect(doubtfulRefs(lines)).toEqual([{ line: 0, word: 0 }]);
    // The ad-lib is marked, but it is the singer's own word, not a doubt.
    expect(adlibRefs(lines)).toEqual([{ line: 1, word: 1 }]);
  });
});

describe('a backing layer', () => {
  it('survives a clone of the sidecar', () => {
    const lines = cloneLines([{
      start: 1, end: 1.5, text: 'a', layer: 'backing',
      words: [{ word: 'a', start: 1, end: 1.5, layer: 'backing' }],
    }]);
    expect(lines[0].layer).toBe('backing');
    expect(lines[0].words[0].layer).toBe('backing');
  });
});

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

describe('hand-checked lines', () => {
  it('ticks one line, keeps every other line as it was, and can clear the tick', () => {
    const lines = twoLines();
    const checked = setLineChecked(lines, 1, true);
    expect(checked[1].checked).toBe(true);
    expect(checked[0].checked).toBeUndefined();
    // Fresh objects, so React sees a change and undo has something to store.
    expect(checked).not.toBe(lines);
    expect(lines[1].checked).toBeUndefined();

    const cleared = setLineChecked(checked, 1, false);
    expect('checked' in cleared[1]).toBe(false);
  });

  it('survives the clone a save or a reload goes through', () => {
    const checked = setLineChecked(twoLines(), 0, true);
    const cloned = cloneLines(checked as any);
    expect(cloned[0].checked).toBe(true);
    expect(cloned[1].checked).toBeUndefined();
  });

  it('ignores an index that is not there', () => {
    const lines = twoLines();
    expect(setLineChecked(lines, 7, true)).toEqual(lines);
  });
});

describe('section labels', () => {
  it('sets, trims and clears one line without touching its neighbours', () => {
    const lines = twoLines();
    const set = setLineSection(lines, 1, '  Chorus  ');
    expect(set[1].section).toBe('Chorus');
    expect(set[0].section).toBe('verse');

    const cleared = setLineSection(set, 1, '   ');
    expect(cleared[1].section).toBeUndefined();
    expect(cleared[0].section).toBe('verse');
  });
});

describe('commands on a multi-word selection', () => {
  const block = (): EditLine[] => [
    line([['one', 1, 1.4], ['two', 1.6, 2], ['three', 2.4, 2.8], ['four', 3.2, 3.6]]),
  ];
  const firstTwo = { anchor: { line: 0, word: 0 }, focus: { line: 0, word: 1 } };

  it('moves every selected word by the same amount', () => {
    const lines = shiftSelection(block(), firstTwo, 0.2);
    expect(lines[0].words.map(w => w.start)).toEqual([1.2, 1.8, 2.4, 3.2]);
    expect(lines[0].words.map(w => w.end)).toEqual([1.6, 2.2, 2.8, 3.6]);
  });

  it('stops the block at the word after it', () => {
    // "two" ends at 2.0 and "three" starts at 2.4: 0.4 s of room, not the 1 s asked for.
    const lines = shiftSelection(block(), firstTwo, 1);
    expect(lines[0].words[1].end).toBe(2.4);
    expect(lines[0].words[2].start).toBe(2.4);
  });

  it('cannot push the block before the start of the track', () => {
    const lines = shiftSelection(block(), firstTwo, -5);
    expect(Math.min(...lines[0].words.map(w => w.start))).toBe(0);
  });

  it('finds the end of the word before the selection, across a line break', () => {
    expect(selectionPrevEnd(twoLines(), { anchor: { line: 1, word: 0 }, focus: { line: 1, word: 1 } })).toBe(1.8);
    expect(selectionPrevEnd(twoLines(), { anchor: { line: 0, word: 0 }, focus: { line: 0, word: 0 } })).toBeNull();
  });

  it('leaves an empty or unmoved selection alone', () => {
    const lines = block();
    expect(shiftSelection(lines, null, 1)).toBe(lines);
    expect(shiftSelection(lines, firstTwo, 0)).toBe(lines);
  });

  it('carries the block when one of its words is dragged, within the same clamps', () => {
    const lines = dragSelection(block(), firstTwo, 0.2, 20);
    expect(lines[0].words.map(w => w.start)).toEqual([1.2, 1.8, 2.4, 3.2]);
    // The word after the block still holds it back.
    expect(dragSelection(block(), firstTwo, 1, 20)[0].words[1].end).toBe(2.4);
  });

  it('stops a dragged block before the end of the audio', () => {
    // "two", the block's last word, ends at 2.0; the next word would allow 2.4.
    const lines = dragSelection(block(), firstTwo, 5, 2.15);
    expect(lines[0].words[1].end).toBe(2.15);
  });

  it('leaves a drag with no block unchanged', () => {
    const lines = block();
    expect(dragSelection(lines, null, 1, 20)).toBe(lines);
  });
});

describe('adding and deleting words', () => {
  it('inserts after the selected word, taking the gap before the next one', () => {
    const spaced = [line([['one', 1, 1.4], ['two', 1.6, 2]])];
    const res = addWordAfter(spaced, 0, 0, 'mid', { total: 10 })!;
    expect(res.at).toBe(1);
    expect(res.lines[0].words.map(w => w.word)).toEqual(['one', 'mid', 'two']);
    expect(res.lines[0].words[1]).toMatchObject({ start: 1.4, end: 1.6 });
    expect(res.lines[0].text).toBe('one mid two');
  });

  it('pushes what follows when the gap is too short for a word', () => {
    const res = addWordAfter(twoLines(), 0, 0, 'new', { total: 10 })!;
    const [one, added, two] = res.lines[0].words;
    expect(one.end).toBe(1.4);
    expect(added).toMatchObject({ start: 1.4 });
    expect(added.end - added.start).toBeCloseTo(0.05, 3);
    expect(two.start).toBeCloseTo(added.end, 3);
  });

  it('inserts before a line\u2019s first word with -1', () => {
    const res = addWordAfter(twoLines(), 1, -1, 'lead', { total: 10 })!;
    expect(res.at).toBe(0);
    expect(res.lines[1].words[0]).toMatchObject({ word: 'lead', start: 1.8, end: 2 });
    expect(res.lines[1].words[1].start).toBe(2);
  });

  it('has a default length after the last word', () => {
    const solo = [line([['solo', 1, 1.2]])];
    expect(addWordAfter(solo, 0, 0, 'tail', { total: 10 })!.lines[0].words[1]).toMatchObject({ start: 1.2, end: 1.45 });
  });

  it('refuses when there is no room before the end of the audio', () => {
    const solo = [line([['solo', 1, 1.2]])];
    expect(addWordAfter(solo, 0, 0, 'tail', { total: 1.21 })).toBeNull();
  });

  it('deletes one word and rebuilds the line from the rest', () => {
    const lines = deleteWords(twoLines(), [{ line: 0, word: 0 }]);
    expect(lines[0].words.map(w => w.word)).toEqual(['two']);
    expect(lines[0]).toMatchObject({ start: 1.4, end: 1.8, text: 'two' });
  });

  it('drops a line whose words all went', () => {
    const lines = deleteWords(twoLines(), [{ line: 0, word: 0 }, { line: 0, word: 1 }]);
    expect(lines.map(l => l.text)).toEqual(['three four']);
  });

  it('deletes a range that crosses a line break', () => {
    const lines = deleteWords(twoLines(), [{ line: 0, word: 1 }, { line: 1, word: 0 }]);
    expect(lines.map(l => l.text)).toEqual(['one', 'four']);
  });
});

describe('renaming a word', () => {
  it('replaces the text, keeps the times, and rebuilds the line text', () => {
    const lines = setWordText(twoLines(), 0, 1, 'too');
    expect(lines[0].words[1]).toMatchObject({ word: 'too', start: 1.4, end: 1.8 });
    expect(lines[0].text).toBe('one too');
    expect(lines[0].start).toBe(1);
    expect(lines[0].end).toBe(1.8);
  });

  it('trims, and collapses inner whitespace', () => {
    expect(setWordText(twoLines(), 0, 0, '  o  one ')[0].words[0].word).toBe('o one');
  });

  it('refuses an empty word and an unchanged one', () => {
    const lines = twoLines();
    expect(setWordText(lines, 0, 0, '   ')).toBe(lines);
    expect(setWordText(lines, 0, 0, 'one')).toBe(lines);
    expect(setWordText(lines, 9, 0, 'gone')).toBe(lines);
  });
});

describe('selecting more than one word', () => {
  it('orders a backwards shift-selection by reading order', () => {
    const lines = twoLines();
    const span = { anchor: { line: 1, word: 1 }, focus: { line: 0, word: 0 } };
    expect(orderSpan(lines, span)).toEqual({ from: { line: 0, word: 0 }, to: { line: 1, word: 1 } });
    expect(spanWordCount(lines, span)).toBe(4);
  });

  it('selecting a line covers every word on it', () => {
    const lines = twoLines();
    const span = lineSpan(lines[1], 1)!;
    expect(spanWordCount(lines, span)).toBe(2);
    expect(spanLineRange(lines, span)).toEqual({ fromLine: 1, toLine: 1 });
    expect(spanTimeRange(lines, span)).toEqual({ start: 2, end: 2.8 });
  });

  it('reports every line a multi-line selection touches', () => {
    const lines = twoLines();
    const span = { anchor: { line: 0, word: 1 }, focus: { line: 1, word: 0 } };
    expect(spanLineRange(lines, span)).toEqual({ fromLine: 0, toLine: 1 });
    expect(selectedRefs(lines, span).map(r => `${r.line}:${r.word}`)).toEqual(['0:1', '1:0']);
    expect([...selectedKeys(lines, span)].sort()).toEqual(['0:1', '1:0']);
  });

  it('a stale reference selects nothing instead of inventing a word', () => {
    const lines = twoLines();
    const stale = { anchor: { line: 4, word: 0 }, focus: { line: 0, word: 0 } };
    expect(orderSpan(lines, stale)).toBeNull();
    expect(spanWordCount(lines, stale)).toBe(0);
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

  it('breaks before the first selected word, not the last one touched', () => {
    const lines = twoLines();
    // A row chip selects the whole line and leaves the focus on its last word;
    // breaking there used to strand that word on a line of its own.
    const row = lineSpan(lines[0], 0)!;
    expect(row.focus.word).toBe(1);
    expect(splitCutAt(lines, 0, row, 5)).toBe(0);
    // A range reaching into the next line still breaks where the range starts.
    expect(splitCutAt(lines, 0, { anchor: { line: 0, word: 1 }, focus: { line: 1, word: 1 } }, 5)).toBe(1);
  });

  it('falls back to the first word at the playhead when nothing is selected', () => {
    expect(splitCutAt(twoLines(), 0, null, 0)).toBe(0);
    expect(splitCutAt(twoLines(), 0, null, 1.2)).toBe(1);
    // A playhead past the row's last word has nothing to break before.
    expect(splitCutAt(twoLines(), 0, null, 1.5)).toBe(-1);
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

describe('rewriting a whole line', () => {
  const row = () => line([['hold', 1, 1.2], ['on', 1.4, 1.6], ['me', 1.8, 2.4]]);

  it('renames in place and leaves every timing where it was', () => {
    const plan = planLineText(row(), 0, 'hold on the');
    expect(plan?.renamed).toBe(1);
    expect(plan?.added).toBe(0);
    expect(plan?.removed).toBe(0);
    expect(plan?.edited.text).toBe('hold on the');
    expect(plan?.edited.words[2].word).toBe('the');
    expect(plan?.edited.words[2].start).toBe(1.8);
    expect(plan?.edited.words[2].end).toBe(2.4);
  });

  it('splits the last word’s own interval for the words added after it', () => {
    const plan = planLineText(row(), 0, 'hold on me now');
    expect(plan?.added).toBe(1);
    expect(plan?.edited.words[2].end).toBe(2.1);
    expect(plan?.edited.words[3].start).toBe(2.1);
    expect(plan?.edited.words[3].end).toBe(2.4);
    expect(plan?.edited.end).toBe(2.4);
  });

  it('gives a dropped word’s time to the last one kept', () => {
    const plan = planLineText(row(), 0, 'hold on');
    expect(plan?.removed).toBe(1);
    expect(plan?.edited.words[1].end).toBe(2.4);
    expect(plan?.edited.end).toBe(2.4);
    expect(plan?.changes.at(-1)?.kind).toBe('removed');
  });

  it('has nothing to ask about for empty text or the same text', () => {
    expect(planLineText(row(), 0, '   ')).toBeNull();
    expect(planLineText(row(), 0, 'hold  on   me')).toBeNull();
  });

  it('splits a line that had no words across its own span', () => {
    const plan = planLineText({ start: 2, end: 3, text: '', words: [] }, 4, 'a b');
    expect(plan?.added).toBe(2);
    expect(plan?.line).toBe(4);
    expect(plan?.edited.words[0].start).toBe(2);
    expect(plan?.edited.words[1].start).toBe(2.5);
    expect(plan?.edited.end).toBe(3);
  });

  it('drops the probability of a word it renames, and keeps the rest', () => {
    const scored = { start: 1, end: 2, text: 'hold on', words: [
      { word: 'hold', start: 1, end: 1.5, confidence: 0.8 },
      { word: 'on', start: 1.5, end: 2, confidence: 0.4 },
    ] };
    const plan = planLineText(scored, 0, 'hold in');
    expect(plan?.edited.words[0].confidence).toBe(0.8);
    expect(plan?.edited.words[1].confidence).toBeUndefined();
  });
});

describe('a dragged span on the waveform', () => {
  const rows = () => [
    line([['one', 1, 1.2], ['two', 1.4, 1.6]]),
    line([['three', 2, 2.2], ['four', 2.4, 2.8]]),
  ];

  it('selects the lines the span covers, and only those', () => {
    expect(linesInSpan(rows(), 1.1, 1.9)).toEqual({ fromLine: 0, toLine: 0 });
    expect(linesInSpan(rows(), 1.9, 2.5)).toEqual({ fromLine: 1, toLine: 1 });
    expect(linesInSpan(rows(), 1.5, 2.1)).toEqual({ fromLine: 0, toLine: 1 });
    expect(linesInSpan(rows(), 3, 4)).toBeNull();
    expect(linesInSpan(rows(), 2, 2)).toBeNull();
  });
});

describe('taking a run line by line', () => {
  const old = () => [
    line([['one', 1, 1.2], ['two', 1.4, 1.6]]),
    line([['three', 2, 2.2], ['four', 2.4, 2.8]]),
  ];
  const fresh = () => [
    { start: 5, end: 5.6, text: 'one two', words: [{ word: 'one', start: 5, end: 5.2 }, { word: 'two', start: 5.4, end: 5.6 }] },
    { start: 6, end: 6.8, text: 'three four', words: [{ word: 'three', start: 6, end: 6.2 }, { word: 'four', start: 6.4, end: 6.8 }] },
  ];

  it('keeps the line on screen where the row was left on old', () => {
    const mixed = mergeProposed(old(), fresh(), [false, true]);
    expect(mixed[0].words[0].start).toBe(1);
    expect(mixed[1].words[0].start).toBe(6);
    expect(mixed.length).toBe(2);
  });

  it('appends a line the run found and the sheet does not have', () => {
    const extra = [...fresh(), { start: 7, end: 7.4, text: 'five', words: [{ word: 'five', start: 7, end: 7.4 }] }];
    expect(mergeProposed(old(), extra, [true, true, true]).length).toBe(3);
    expect(mergeProposed(old(), extra, [true, true, false]).length).toBe(2);
  });

  it('drops the by-ear tick from a line the run replaced', () => {
    const ticked = [{ ...old()[0], checked: true }, old()[1]];
    expect(mergeProposed(ticked, fresh(), [true, false])[0].checked).toBeUndefined();
  });
});

describe('which proposed rows are changes', () => {
  const proposed = (start: number, text: string, words: Array<[string, number, number]>) => ({
    start,
    end: words[words.length - 1][2],
    text,
    words: words.map(([word, s, e]) => ({ word, start: s, end: e })),
  });
  const sheet = (): EditLine[] => [
    line([['one', 1, 1.4], ['two', 1.4, 1.8]]),
    line([['three', 2, 2.4]]),
  ];
  const agreed = () => [
    proposed(1, 'one two', [['one', 1, 1.4], ['two', 1.4, 1.8]]),
    proposed(2, 'three', [['three', 2, 2.4]]),
  ];

  it('reports nothing when the run agrees with the sheet', () => {
    expect(changedProposedRows(sheet(), agreed())).toEqual([]);
  });

  it('reports a word moved inside a line whose start and text did not move', () => {
    const moved = agreed();
    moved[0] = proposed(1, 'one two', [['one', 1, 1.5], ['two', 1.5, 1.8]]);
    expect(changedProposedRows(sheet(), moved)).toEqual([0]);
  });

  it('reports a line the run adds and a line whose text changed', () => {
    const rows = agreed();
    rows[1] = proposed(2, 'three!', [['three!', 2, 2.4]]);
    rows.push(proposed(3, 'four', [['four', 3, 3.2]]));
    expect(changedProposedRows(sheet(), rows)).toEqual([1, 2]);
  });

  it('treats empty line text as the words it already holds', () => {
    const bare = [{ ...sheet()[0], text: '' }, sheet()[1]];
    expect(changedProposedRows(bare, agreed())).toEqual([]);
  });

  it('reports a line the run re-sections', () => {
    const rows = agreed();
    rows[0] = { ...rows[0], section: 'chorus' };
    expect(changedProposedRows(sheet(), rows)).toEqual([0]);
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
