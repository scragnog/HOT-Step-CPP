import assert from 'node:assert/strict';
import test from 'node:test';

const store = new Map<string, string>();
(globalThis as any).localStorage = {
  getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v),
};
const { buildCaption, cropNeedsMode, finishSavedBuild, moveCropEdge, onBuildSaved, settingsSummary, submittedCaption } = await import('./Yue2Build');
const saved = () => JSON.parse(store.get('cover-studio-buildSessionId') ?? '""');

test('a Save finishing after the studio was rebuilt clears its build and tells the mounted studio', () => {
  store.set('cover-studio-buildSessionId', '"a"');
  const seen: string[] = [];
  const off = onBuildSaved(id => seen.push(id));
  assert.equal(finishSavedBuild('a'), true);
  assert.equal(saved(), '');
  assert.deepEqual(seen, ['a']);
  off();
});

test('a Save finishing after a newer build began leaves the newer build', () => {
  store.set('cover-studio-buildSessionId', '"b"');
  const seen: string[] = [];
  const off = onBuildSaved(id => seen.push(id));
  assert.equal(finishSavedBuild('a'), false);
  assert.equal(saved(), 'b');
  assert.deepEqual(seen, []);
  off();
});

test('lyrics edits wait for a saved build to load instead of becoming a draft', async () => {
  const { lyricsTarget } = await import('./Yue2Build');
  assert.equal(lyricsTarget('saved-id', null), 'loading');
  assert.equal(lyricsTarget('saved-id', { id: 'saved-id' }), 'session');
  assert.equal(lyricsTarget('', null), 'draft');
});

test('a draft belongs to the source and score it was written for', async () => {
  const { draftText } = await import('./Yue2Build');
  const first = JSON.stringify(['musicxml', '', '', '', 'X:1 first score']);
  const second = JSON.stringify(['musicxml', '', '', '', 'X:1 other score']);
  const draft = { source: first, text: '[verse]\nmine' };
  assert.equal(draftText(draft, first), '[verse]\nmine');
  assert.equal(draftText(draft, second), null);
  assert.equal(draftText(null, first), null);
});

const arranged = (plan: string[], kept = 0, lyricBlocks: Array<number | null> = [0, 1, null]) => ({
  id: 'x', revision: 3, kept: kept ? ['t'] : [], pending: [], error: null, params: {}, sections: [],
  takes: kept ? [{ id: 't', to: kept - 1, parent: null, occurrences: [], revision: 3 }] : [],
  arrangement: { finished: true, plan: plan.map((s, i) => ({ id: `o${i}`, section: s })),
    sections: lyricBlocks.map((lyricBlock, i) => ({ id: `s${i}`, label: `S${i}`, startBar: 1, bars: 4, lyricBlock })) },
}) as any;

test('the output bar lists occurrences in plan order, kept ones locked, a repeat with its own id', async () => {
  const { planItems } = await import('./Yue2Build');
  const items = planItems(arranged(['s0', 's1', 's1', 's0'], 2));
  assert.deepEqual(items.map(i => [i.id, i.section, i.kept]), [['o0', 0, true], ['o1', 1, true], ['o2', 1, false], ['o3', 0, false]]);
  // Before a build exists the song is empty: it is built up from the source bar.
  assert.deepEqual(planItems(null), []);
});

test('plan edits send only the unbuilt part: add a source section, remove, move', async () => {
  const { planItems, editedPlan } = await import('./Yue2Build');
  const items = planItems(arranged(['s0', 's1', 's2'], 1));
  assert.deepEqual(editedPlan(items, 1, { add: 1 }), ['o1', 'o2', 's1']);
  assert.deepEqual(editedPlan(items, 1, { remove: 'o1' }), ['o2']);
  assert.deepEqual(editedPlan(items, 1, { move: 'o2', by: -1 }), ['o2', 'o1']);
  assert.deepEqual(editedPlan(items, 1, { move: 'o1', by: -1 }), ['o1', 'o2']);
});

test('a repeated chorus shows its lyrics under each occurrence; an unsung section shows none', async () => {
  const { planItems, occurrenceLyrics } = await import('./Yue2Build');
  const lyrics = '[verse]\none\n\n[chorus]\nsing\n';
  const session = arranged(['s0', 's1', 's1', 's2']);
  const text = planItems(session).map(i => occurrenceLyrics(lyrics, session, i));
  assert.deepEqual(text, ['[verse]\none', '[chorus]\nsing', '[chorus]\nsing', '']);
});

test('a take is keepable only if it follows the spine at this revision and composed the planned occurrences', async () => {
  const { keepable } = await import('./Yue2Build');
  const session = arranged(['s0', 's1', 's2'], 1);
  const take = (over = {}) => ({ id: 'n', from: 1, to: 2, revision: 3, parent: 't', occurrences: ['o1', 'o2'], ...over }) as any;
  assert.equal(keepable(session, take()), true);
  assert.equal(keepable(session, take({ revision: 2 })), false);
  assert.equal(keepable(session, take({ occurrences: ['o2', 'o1'] })), false);
  assert.equal(keepable(session, take({ parent: null })), false);
});

test('Generate sends a caption only when the field differs from the build\'s', () => {
  assert.equal(submittedCaption('  bright pop ', 'bright pop'), '');
  assert.equal(submittedCaption('quiet, acoustic guitar, whispers ', 'bright pop'), 'quiet, acoustic guitar, whispers');
  assert.equal(buildCaption({ caption: 'c', style: 's' }), 'c');
  assert.equal(buildCaption({ prompt: 'p', caption: 'c' }), 'p');
});

test('the lyrics pane edits lines without the tag; Generate applies only the changed ones', async () => {
  const { planItems, occurrenceLines, pendingLyrics } = await import('./Yue2Build');
  const lyrics = '[verse]\none\ntwo\n\n[chorus]\nsing\n';
  const session = arranged(['s0', 's1', 's1', 's2']);
  session.arrangement.plan[2].lyrics = 'sing again';
  assert.deepEqual(planItems(session).map(i => occurrenceLines(lyrics, session, i)), ['one\ntwo', 'sing', 'sing again', null]);
  assert.deepEqual(pendingLyrics({ o0: 'one\ntwo ', o1: 'louder', o3: 'x', o9: 'y' }, { o0: 'one\ntwo', o1: 'sing', o3: null }), { o1: 'louder' });
});


test('take crop: the in point only on a first-section take, a second clear of the out point, snapped to frames', () => {
  // 10 s, first section (no prefix): the in point can move; dragging back to the top clears it.
  const first = { id: 'a', from: 0, to: 0, frames: 250, starts: [0, 200], audioUrl: '', seed: 1, occurrences: ['o0'] } as any;
  const none = { in: null, out: null, mode: null };
  assert.deepEqual(moveCropEdge(first, none, 'in', 1.234), { in: 1.24, out: null, mode: null });
  assert.equal(moveCropEdge(first, { ...none, in: 2 }, 'in', -3).in, null);
  assert.equal(moveCropEdge(first, { ...none, out: 4 }, 'in', 3.9).in, 3, 'a second before the out point');
  assert.equal(moveCropEdge(first, none, 'out', 99).out, null, 'dragged to the end is no crop');
  assert.equal(moveCropEdge(first, { ...none, in: 5 }, 'out', 2).out, 6, 'a second after the in point');
  // A later take resumes 150 frames: no in point, and the out point stays in the part it composed.
  const later = { ...first, from: 1, to: 1, prefix: 150, starts: [0, 150, 220] };
  assert.equal(moveCropEdge(later, none, 'in', 2).in, null);
  assert.equal(moveCropEdge(later, none, 'out', 1).out, 6.04);
  // Inside the section it composed, or inside the one it ran on into, needs a choice; at a section's start does not.
  assert.equal(cropNeedsMode(later, { ...none, out: 7 }), true);
  assert.equal(cropNeedsMode(later, { ...none, out: 8.8 }), false);
  assert.equal(cropNeedsMode(later, { ...none, out: 9.2 }), true);
  assert.equal(cropNeedsMode(later, none), false);
});

test('a cut with nothing after it ends the song only when the plan does', async () => {
  const { advanceLabel } = await import('./Yue2Build');
  assert.equal(advanceLabel('verse', 'chorus', false), 'moves on to chorus');
  assert.equal(advanceLabel('verse', null, true), 'keeps verse cut short and ends the song');
  assert.match(advanceLabel('verse', null, false), /next section you add starts at the cut/);
});

test('locked build settings read as one line', () => {
  assert.equal(settingsSummary({ voices: 'vocal', tempo: 'free', keyShift: 0, cfgScale: 1, arStrength: 0.6 }, 'C major'),
    'vocal melody only, free tempo, source key, condition strength 1, AR strength 0.6');
  assert.match(settingsSummary({ voices: 'both', tempo: 96, keyShift: 2, cfgScale: 1.2, arStrength: 0.5 }, 'C major'), /^vocal \+ instrumental line, 96 BPM, \+2 st \(D major\)/);
});

test('an improv is its own occurrence with its bars, added by bar count, unsung', async () => {
  const { planItems, editedPlan, occurrenceLines } = await import('./Yue2Build');
  const session = arranged(['s0', 'improv'], 1);
  session.arrangement.plan[1].bars = 8;
  const items = planItems(session);
  assert.deepEqual(items.map(i => [i.id, i.section, i.improv]), [['o0', 0, undefined], ['o1', -1, 8]]);
  assert.deepEqual(editedPlan(items, 1, { improv: 4 }), ['o1', 'improv:4']);
  assert.equal(occurrenceLines('[S0]\nwords\n', session, items[1]), null);
});
