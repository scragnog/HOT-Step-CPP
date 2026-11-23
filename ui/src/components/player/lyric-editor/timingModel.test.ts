import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
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
    assert.deepEqual(sectionRuns(lines), [
      { section: 'verse', from: 0 },
      { section: 'chorus', from: 2 },
      { section: 'verse', from: 4 },
      { section: 'chorus', from: 6 },
    ]);
  });

  it('starts a section on the line when the playhead is before its first word', () => {
    const result = addSectionAt(twoLines(), 1, 1.9, ' Chorus ');
    assert.equal(result?.at, 1);
    assert.deepEqual(result?.lines.map(l => l.section), ['verse', 'Chorus']);
    assert.equal((result?.lines).length, 2);
  });

  it('splits the line at the playhead so the section gets its own line', () => {
    // 2.2 is between `three` (2.0) and `four` (2.4): inside the line, so the
    // boundary needs a row of its own. 2.5 would be past the line, i.e. 2.5 is
    // the next line's business and no split happens.
    const result = addSectionAt(twoLines(), 1, 2.2, 'Bridge');
    assert.equal(result?.at, 2);
    assert.deepEqual(result?.lines.map(l => l.section), ['verse', 'chorus', 'Bridge']);
    // The split moved no word: every time on the sheet is the one it had.
    const times = result!.lines.flatMap(l => l.words.map(w => [w.word, w.start, w.end]));
    assert.deepEqual(times, [
      ['one', 1, 1.4], ['two', 1.4, 1.8],
      ['three', 2, 2.4],
      ['four', 2.4, 2.8],
    ]);
    assert.deepEqual(result?.lines.map(l => l.words.length), [2, 1, 1]);
  });

  it('refuses an empty name or a line that is not there', () => {
    assert.equal(addSectionAt(twoLines(), 0, 1, '   '), null);
    assert.equal(addSectionAt(twoLines(), 9, 1, 'Chorus'), null);
  });

  it('deletes the whole run, so a repeated label does not survive and resurface', () => {
    const lines: EditLine[] = [
      line([['a', 1, 1.2]], 'chorus'),
      line([['b', 2, 2.2]], 'chorus'),
      line([['c', 3, 3.2]], 'verse'),
    ];
    const next = removeSectionAt(lines, 0)!;
    assert.deepEqual(next.map(l => l.section), [undefined, undefined, 'verse']);
    assert.deepEqual(sectionRuns(next).map(run => run.from), [2]);
    assert.equal(next[0].words[0].start, 1);
  });

  it('stops at the next different label, so a later repeat is its own section', () => {
    const lines: EditLine[] = [
      line([['a', 1, 1.2]], 'chorus'),
      line([['b', 2, 2.2]], 'verse'),
      line([['c', 3, 3.2]], 'chorus'),
    ];
    const next = removeSectionAt(lines, 0)!;
    assert.deepEqual(next.map(l => l.section), [undefined, 'verse', 'chorus']);
    assert.deepEqual(sectionRuns(next).map(run => run.from), [1, 2]);
  });

  it('deletes only the label, leaving every word time alone', () => {
    const removed = removeSectionAt(twoLines(), 1);
    assert.deepEqual(removed?.map(l => l.section), ['verse', undefined]);
    assert.deepEqual(removed?.[1].words, twoLines()[1].words);
    assert.notEqual(removeSectionAt(twoLines(), 0), null);
    const unlabelled = removeSectionAt(removeSectionAt(twoLines(), 0)!, 0);
    assert.equal(unlabelled, null);
  });
});

describe('rough timings', () => {
  it('is rough when every word sits on one instant', () => {
    const piled: EditLine[] = [line([['a', 0, 0], ['b', 0, 0]]), line([['c', 0, 0], ['d', 0, 0]])];
    assert.equal(timingsLookRough(piled, 180), true);
  });

  it('is rough when most words have no length and the rest do', () => {
    const mostly: EditLine[] = [line([['a', 1, 1.3], ['b', 0, 0], ['c', 0, 0]])];
    assert.equal(timingsLookRough(mostly, 180), true);
  });

  it('is rough when the words stop inside the first fifth of a long track', () => {
    assert.equal(timingsLookRough(twoLines(), 300), true);
  });

  it('leaves a placed track alone, including a short one and one with no words', () => {
    assert.equal(timingsLookRough(twoLines(), 4), false);
    assert.equal(timingsLookRough([], 180), false);
    assert.equal(timingsLookRough([line([['only', 3, 3.4]])], 180), false);
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
    assert.deepEqual(doubtfulRefs(lines), [{ line: 0, word: 0 }]);
    // The ad-lib is marked, but it is the singer's own word, not a doubt.
    assert.deepEqual(adlibRefs(lines), [{ line: 1, word: 1 }]);
  });
});

describe('a backing layer', () => {
  it('survives a clone of the sidecar', () => {
    const lines = cloneLines([{
      start: 1, end: 1.5, text: 'a', layer: 'backing',
      words: [{ word: 'a', start: 1, end: 1.5, layer: 'backing' }],
    }]);
    assert.equal(lines[0].layer, 'backing');
    assert.equal(lines[0].words[0].layer, 'backing');
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
    assert.equal(checked[1].checked, true);
    assert.equal(checked[0].checked, undefined);
    // Fresh objects, so React sees a change and undo has something to store.
    assert.notEqual(checked, lines);
    assert.equal(lines[1].checked, undefined);

    const cleared = setLineChecked(checked, 1, false);
    assert.equal('checked' in cleared[1], false);
  });

  it('survives the clone a save or a reload goes through', () => {
    const checked = setLineChecked(twoLines(), 0, true);
    const cloned = cloneLines(checked as any);
    assert.equal(cloned[0].checked, true);
    assert.equal(cloned[1].checked, undefined);
  });

  it('ignores an index that is not there', () => {
    const lines = twoLines();
    assert.deepEqual(setLineChecked(lines, 7, true), lines);
  });
});

describe('section labels', () => {
  it('sets, trims and clears one line without touching its neighbours', () => {
    const lines = twoLines();
    const set = setLineSection(lines, 1, '  Chorus  ');
    assert.equal(set[1].section, 'Chorus');
    assert.equal(set[0].section, 'verse');

    const cleared = setLineSection(set, 1, '   ');
    assert.equal(cleared[1].section, undefined);
    assert.equal(cleared[0].section, 'verse');
  });
});

describe('commands on a multi-word selection', () => {
  const block = (): EditLine[] => [
    line([['one', 1, 1.4], ['two', 1.6, 2], ['three', 2.4, 2.8], ['four', 3.2, 3.6]]),
  ];
  const firstTwo = { anchor: { line: 0, word: 0 }, focus: { line: 0, word: 1 } };

  it('moves every selected word by the same amount', () => {
    const lines = shiftSelection(block(), firstTwo, 0.2);
    assert.deepEqual(lines[0].words.map(w => w.start), [1.2, 1.8, 2.4, 3.2]);
    assert.deepEqual(lines[0].words.map(w => w.end), [1.6, 2.2, 2.8, 3.6]);
  });

  it('stops the block at the word after it', () => {
    // "two" ends at 2.0 and "three" starts at 2.4: 0.4 s of room, not the 1 s asked for.
    const lines = shiftSelection(block(), firstTwo, 1);
    assert.equal(lines[0].words[1].end, 2.4);
    assert.equal(lines[0].words[2].start, 2.4);
  });

  it('cannot push the block before the start of the track', () => {
    const lines = shiftSelection(block(), firstTwo, -5);
    assert.equal(Math.min(...lines[0].words.map(w => w.start)), 0);
  });

  it('finds the end of the word before the selection, across a line break', () => {
    assert.equal(selectionPrevEnd(twoLines(), { anchor: { line: 1, word: 0 }, focus: { line: 1, word: 1 } }), 1.8);
    assert.equal(selectionPrevEnd(twoLines(), { anchor: { line: 0, word: 0 }, focus: { line: 0, word: 0 } }), null);
  });

  it('leaves an empty or unmoved selection alone', () => {
    const lines = block();
    assert.equal(shiftSelection(lines, null, 1), lines);
    assert.equal(shiftSelection(lines, firstTwo, 0), lines);
  });

  it('carries the block when one of its words is dragged, within the same clamps', () => {
    const lines = dragSelection(block(), firstTwo, 0.2, 20);
    assert.deepEqual(lines[0].words.map(w => w.start), [1.2, 1.8, 2.4, 3.2]);
    // The word after the block still holds it back.
    assert.equal(dragSelection(block(), firstTwo, 1, 20)[0].words[1].end, 2.4);
  });

  it('stops a dragged block before the end of the audio', () => {
    // "two", the block's last word, ends at 2.0; the next word would allow 2.4.
    const lines = dragSelection(block(), firstTwo, 5, 2.15);
    assert.equal(lines[0].words[1].end, 2.15);
  });

  it('leaves a drag with no block unchanged', () => {
    const lines = block();
    assert.equal(dragSelection(lines, null, 1, 20), lines);
  });
});

describe('adding and deleting words', () => {
  it('inserts after the selected word, taking the gap before the next one', () => {
    const spaced = [line([['one', 1, 1.4], ['two', 1.6, 2]])];
    const res = addWordAfter(spaced, 0, 0, 'mid', { total: 10 })!;
    assert.equal(res.at, 1);
    assert.deepEqual(res.lines[0].words.map(w => w.word), ['one', 'mid', 'two']);
    assertMatchObject(res.lines[0].words[1], { start: 1.4, end: 1.6 });
    assert.equal(res.lines[0].text, 'one mid two');
  });

  it('pushes what follows when the gap is too short for a word', () => {
    const res = addWordAfter(twoLines(), 0, 0, 'new', { total: 10 })!;
    const [one, added, two] = res.lines[0].words;
    assert.equal(one.end, 1.4);
    assertMatchObject(added, { start: 1.4 });
    assert.ok(Math.abs((added.end - added.start) - (0.05)) < 10 ** -3 / 2);
    assert.ok(Math.abs((two.start) - (added.end)) < 10 ** -3 / 2);
  });

  it('inserts before a line\u2019s first word with -1', () => {
    const res = addWordAfter(twoLines(), 1, -1, 'lead', { total: 10 })!;
    assert.equal(res.at, 0);
    assertMatchObject(res.lines[1].words[0], { word: 'lead', start: 1.8, end: 2 });
    assert.equal(res.lines[1].words[1].start, 2);
  });

  it('has a default length after the last word', () => {
    const solo = [line([['solo', 1, 1.2]])];
    assertMatchObject(addWordAfter(solo, 0, 0, 'tail', { total: 10 })!.lines[0].words[1], { start: 1.2, end: 1.45 });
  });

  it('refuses when there is no room before the end of the audio', () => {
    const solo = [line([['solo', 1, 1.2]])];
    assert.equal(addWordAfter(solo, 0, 0, 'tail', { total: 1.21 }), null);
  });

  it('deletes one word and rebuilds the line from the rest', () => {
    const lines = deleteWords(twoLines(), [{ line: 0, word: 0 }]);
    assert.deepEqual(lines[0].words.map(w => w.word), ['two']);
    assertMatchObject(lines[0], { start: 1.4, end: 1.8, text: 'two' });
  });

  it('drops a line whose words all went', () => {
    const lines = deleteWords(twoLines(), [{ line: 0, word: 0 }, { line: 0, word: 1 }]);
    assert.deepEqual(lines.map(l => l.text), ['three four']);
  });

  it('deletes a range that crosses a line break', () => {
    const lines = deleteWords(twoLines(), [{ line: 0, word: 1 }, { line: 1, word: 0 }]);
    assert.deepEqual(lines.map(l => l.text), ['one', 'four']);
  });
});

describe('renaming a word', () => {
  it('replaces the text, keeps the times, and rebuilds the line text', () => {
    const lines = setWordText(twoLines(), 0, 1, 'too');
    assertMatchObject(lines[0].words[1], { word: 'too', start: 1.4, end: 1.8 });
    assert.equal(lines[0].text, 'one too');
    assert.equal(lines[0].start, 1);
    assert.equal(lines[0].end, 1.8);
  });

  it('trims, and collapses inner whitespace', () => {
    assert.equal(setWordText(twoLines(), 0, 0, '  o  one ')[0].words[0].word, 'o one');
  });

  it('refuses an empty word and an unchanged one', () => {
    const lines = twoLines();
    assert.equal(setWordText(lines, 0, 0, '   '), lines);
    assert.equal(setWordText(lines, 0, 0, 'one'), lines);
    assert.equal(setWordText(lines, 9, 0, 'gone'), lines);
  });
});

describe('selecting more than one word', () => {
  it('orders a backwards shift-selection by reading order', () => {
    const lines = twoLines();
    const span = { anchor: { line: 1, word: 1 }, focus: { line: 0, word: 0 } };
    assert.deepEqual(orderSpan(lines, span), { from: { line: 0, word: 0 }, to: { line: 1, word: 1 } });
    assert.equal(spanWordCount(lines, span), 4);
  });

  it('selecting a line covers every word on it', () => {
    const lines = twoLines();
    const span = lineSpan(lines[1], 1)!;
    assert.equal(spanWordCount(lines, span), 2);
    assert.deepEqual(spanLineRange(lines, span), { fromLine: 1, toLine: 1 });
    assert.deepEqual(spanTimeRange(lines, span), { start: 2, end: 2.8 });
  });

  it('reports every line a multi-line selection touches', () => {
    const lines = twoLines();
    const span = { anchor: { line: 0, word: 1 }, focus: { line: 1, word: 0 } };
    assert.deepEqual(spanLineRange(lines, span), { fromLine: 0, toLine: 1 });
    assert.deepEqual(selectedRefs(lines, span).map(r => `${r.line}:${r.word}`), ['0:1', '1:0']);
    assert.deepEqual([...selectedKeys(lines, span)].sort(), ['0:1', '1:0']);
  });

  it('a stale reference selects nothing instead of inventing a word', () => {
    const lines = twoLines();
    const stale = { anchor: { line: 4, word: 0 }, focus: { line: 0, word: 0 } };
    assert.equal(orderSpan(lines, stale), null);
    assert.equal(spanWordCount(lines, stale), 0);
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
    assert.equal(row.focus.word, 1);
    assert.equal(splitCutAt(lines, 0, row, 5), 0);
    // A range reaching into the next line still breaks where the range starts.
    assert.equal(splitCutAt(lines, 0, { anchor: { line: 0, word: 1 }, focus: { line: 1, word: 1 } }, 5), 1);
  });

  it('falls back to the first word at the playhead when nothing is selected', () => {
    assert.equal(splitCutAt(twoLines(), 0, null, 0), 0);
    assert.equal(splitCutAt(twoLines(), 0, null, 1.2), 1);
    // A playhead past the row's last word has nothing to break before.
    assert.equal(splitCutAt(twoLines(), 0, null, 1.5), -1);
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
    assert.equal(plan?.renamed, 1);
    assert.equal(plan?.added, 0);
    assert.equal(plan?.removed, 0);
    assert.equal(plan?.edited.text, 'hold on the');
    assert.equal(plan?.edited.words[2].word, 'the');
    assert.equal(plan?.edited.words[2].start, 1.8);
    assert.equal(plan?.edited.words[2].end, 2.4);
  });

  it('splits the last word’s own interval for the words added after it', () => {
    const plan = planLineText(row(), 0, 'hold on me now');
    assert.equal(plan?.added, 1);
    assert.equal(plan?.edited.words[2].end, 2.1);
    assert.equal(plan?.edited.words[3].start, 2.1);
    assert.equal(plan?.edited.words[3].end, 2.4);
    assert.equal(plan?.edited.end, 2.4);
  });

  it('gives a dropped word’s time to the last one kept', () => {
    const plan = planLineText(row(), 0, 'hold on');
    assert.equal(plan?.removed, 1);
    assert.equal(plan?.edited.words[1].end, 2.4);
    assert.equal(plan?.edited.end, 2.4);
    assert.equal(plan?.changes.at(-1)?.kind, 'removed');
  });

  it('has nothing to ask about for empty text or the same text', () => {
    assert.equal(planLineText(row(), 0, '   '), null);
    assert.equal(planLineText(row(), 0, 'hold  on   me'), null);
  });

  it('splits a line that had no words across its own span', () => {
    const plan = planLineText({ start: 2, end: 3, text: '', words: [] }, 4, 'a b');
    assert.equal(plan?.added, 2);
    assert.equal(plan?.line, 4);
    assert.equal(plan?.edited.words[0].start, 2);
    assert.equal(plan?.edited.words[1].start, 2.5);
    assert.equal(plan?.edited.end, 3);
  });

  it('drops the probability of a word it renames, and keeps the rest', () => {
    const scored = { start: 1, end: 2, text: 'hold on', words: [
      { word: 'hold', start: 1, end: 1.5, confidence: 0.8 },
      { word: 'on', start: 1.5, end: 2, confidence: 0.4 },
    ] };
    const plan = planLineText(scored, 0, 'hold in');
    assert.equal(plan?.edited.words[0].confidence, 0.8);
    assert.equal(plan?.edited.words[1].confidence, undefined);
  });
});

describe('a dragged span on the waveform', () => {
  const rows = () => [
    line([['one', 1, 1.2], ['two', 1.4, 1.6]]),
    line([['three', 2, 2.2], ['four', 2.4, 2.8]]),
  ];

  it('selects the lines the span covers, and only those', () => {
    assert.deepEqual(linesInSpan(rows(), 1.1, 1.9), { fromLine: 0, toLine: 0 });
    assert.deepEqual(linesInSpan(rows(), 1.9, 2.5), { fromLine: 1, toLine: 1 });
    assert.deepEqual(linesInSpan(rows(), 1.5, 2.1), { fromLine: 0, toLine: 1 });
    assert.equal(linesInSpan(rows(), 3, 4), null);
    assert.equal(linesInSpan(rows(), 2, 2), null);
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
    assert.equal(mixed[0].words[0].start, 1);
    assert.equal(mixed[1].words[0].start, 6);
    assert.equal(mixed.length, 2);
  });

  it('appends a line the run found and the sheet does not have', () => {
    const extra = [...fresh(), { start: 7, end: 7.4, text: 'five', words: [{ word: 'five', start: 7, end: 7.4 }] }];
    assert.equal(mergeProposed(old(), extra, [true, true, true]).length, 3);
    assert.equal(mergeProposed(old(), extra, [true, true, false]).length, 2);
  });

  it('drops the by-ear tick from a line the run replaced', () => {
    const ticked = [{ ...old()[0], checked: true }, old()[1]];
    assert.equal(mergeProposed(ticked, fresh(), [true, false])[0].checked, undefined);
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
    assert.deepEqual(changedProposedRows(sheet(), agreed()), []);
  });

  it('reports a word moved inside a line whose start and text did not move', () => {
    const moved = agreed();
    moved[0] = proposed(1, 'one two', [['one', 1, 1.5], ['two', 1.5, 1.8]]);
    assert.deepEqual(changedProposedRows(sheet(), moved), [0]);
  });

  it('reports a line the run adds and a line whose text changed', () => {
    const rows = agreed();
    rows[1] = proposed(2, 'three!', [['three!', 2, 2.4]]);
    rows.push(proposed(3, 'four', [['four', 3, 3.2]]));
    assert.deepEqual(changedProposedRows(sheet(), rows), [1, 2]);
  });

  it('treats empty line text as the words it already holds', () => {
    const bare = [{ ...sheet()[0], text: '' }, sheet()[1]];
    assert.deepEqual(changedProposedRows(bare, agreed()), []);
  });

  it('reports a line the run re-sections', () => {
    const rows = agreed();
    rows[0] = { ...rows[0], section: 'chorus' };
    assert.deepEqual(changedProposedRows(sheet(), rows), [0]);
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

/** Every key in `expected` matches `actual`, recursively; extra keys in `actual` are allowed. */
function assertMatchObject(actual: unknown, expected: unknown): void {
  if (expected === null || typeof expected !== 'object') { assert.deepEqual(actual, expected); return; }
  assert.ok(actual !== null && typeof actual === 'object', 'expected an object');
  if (Array.isArray(expected)) { assert.ok(Array.isArray(actual)); assert.equal(actual.length, expected.length); }
  for (const [k, v] of Object.entries(expected)) assertMatchObject((actual as Record<string, unknown>)[k], v);
}
