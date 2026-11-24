// Tests of the core's additions for package 3c (src/derive/build.ts, recording.ts, score.ts; the node rule for a control
// that stands back in src/derive/desk.ts; a strip's "Also set" line in models.ts and tiers.ts), each against what the
// concept does (25-module-build.js buildSecs, buildFace, buildDrift, staleKeep, replanTitle and the step handlers;
// 25-module-cont.js setCont, contHit and its keys; 25-module-score.js abcFoot; 34-score-modal.js planScore's drawing,
// abcInfo and scoreLayout).
//
//   node tools/test-instruments-3c.ts        exit 1 on a failure

import { DEFINITIONS } from '../src/definitions/index.ts';
import type { EngineId, Json, Node, Tier, Value } from '../src/definitions/types.ts';
import {
  abcInfo, bindPart, buildSections, cleanBuild, indexOf, nodeKeys, nodeShown, partAct, partModel, readsAsAbc, scoreAsk, scoreLayout, scoreModel, sectionColour, spanMove, tileModel,
  type Change, type Desk, type Msg, type Score,
} from '../src/derive.ts';
import { say } from '../src/derive/messages.ts';

const ix = indexOf(DEFINITIONS);
const J = (x: unknown) => JSON.stringify(x);
const fails: string[] = [];
let n = 0;
const ok = (what: string, cond: boolean, detail: unknown = '') => { n++; if (!cond) fails.push(`${what}${detail === '' ? '' : ': ' + (typeof detail === 'string' ? detail : J(detail))}`); };
const desk = (engine: EngineId, tier: Tier, on: string[], vals: Desk['vals'] = {}, state: Desk['state'] = {}, sessions?: Desk['sessions']): Desk =>
  ({ engine, tier, vals, auto: {}, state, on: new Set([...DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id), ...on]), ...(sessions ? { sessions } : {}) });
const def = (id: string) => DEFINITIONS.find(d => d.id === id)!;
const s = (m: unknown) => say(m as Msg);
const node = (id: string, part: string): Node => {
  let out: Node | undefined;
  const walk = (ns: Node[]) => { for (const x of ns) { if (!out && x.part === part) out = x; walk(x.children ?? []); } };
  walk(('face' in def(id) ? (def(id) as { face: Node[] }).face : []) ?? []);
  return out!;
};
const keyOf = (id: string, x: Node) => nodeKeys(def(id)).get(x) ?? '';
const model = (id: string, part: string, d: Desk) => partModel(ix, def(id), node(id, part), d, keyOf(id, node(id, part)), 'tile');
function act(id: string, part: string, d0: Desk, name: string, arg?: unknown) {
  let d = d0; const steps: Change[] = [], told: [string, unknown][] = [];
  const props = bindPart(ix, def(id), node(id, part), () => d, c => { steps.push(c); d = { ...d, vals: { ...d.vals, ...Object.fromEntries(Object.entries(c.vals).filter((e): e is [string, Value] => e[1] !== null)) }, auto: { ...d.auto, ...c.auto }, state: { ...d.state, ...c.state } }; });
  const done = partAct(ix, props.node, props, d, name, arg, (m, how) => told.push([s(m), how]));
  return { steps, told, done, desk: d };
}

/* ---- the sections the build knows */
{
  const secs = buildSections('untagged\n[Verse]\na\nb\n[Chorus]\nc\n[Verse]\nd');
  ok('words before a tag are a Verse, and a repeated name is numbered (B-05)', J(secs.map(x => [x.id, x.label, x.n])) === J([['Verse#1', 'Verse 1', 1], ['Verse#2', 'Verse 2', 2], ['Chorus#1', 'Chorus', 1], ['Verse#3', 'Verse 3', 1]]), secs);
  ok('a section keeps its words', secs[1]!.words === 'a\nb');
  ok('a section has its colour by its name', sectionColour('Verse 2') === '#3f5ef0' && sectionColour('Pre-Chorus') === '#9440f5' && sectionColour('Solo') === '#8b8496');
  ok('a stored build is cleaned: a keep is kept, junk dropped', J(cleanBuild({ session: 's1', seed: 5, plan: ['Verse#1', 3], kept: { 'Verse#1': 't' }, songs: ['a', 1] } as unknown as Json)) === J({ session: 's1', seed: 5, plan: ['Verse#1'], kept: { 'Verse#1': 't' }, dropped: [], rendered: null, songs: ['a'] }));
}

/* ---- the build board through its life */
const LYR = '[Verse]\nline\n[Chorus]\nline';
const take = (id: string, words = 'line', ready = true) => ({ id, ready, hue: 10, dur: 20, w: 1, words, note: '' });
const SESS = (takes: object, cur?: string) => ({ build: { planning: false, sessions: { s1: { takes, ...(cur ? { cur } : {}) } } } as unknown as Json });
{
  const empty = desk('yue2', 'lab', ['build'], { 'lyrics.text': LYR });
  const m0 = model('build', 'buildBoard', empty)!.instrument!.build!;
  ok('empty, the board is not planned and asks to plan the sections as they stand', !m0.planned && J(m0.plan.sections) === J([{ id: 'Verse#1', words: 'line' }, { id: 'Chorus#1', words: 'line' }]));
  const p = act('build', 'buildBoard', empty, 'planned', { session: 's1', seed: 42 });
  ok('the first plan is the record\'s plan step', p.steps.length === 1 && s(p.steps[0]!.words.label) === 'Plan the score' && J((p.desk.state.build as { plan: string[] }).plan) === J(['Verse#1', 'Chorus#1']), p.steps.map(c => s(c.words.label)));
  ok('and is said, not toasted', J(p.told) === J([['Score planned', 'say']]), p.told);
  const planned = { ...p.desk, sessions: SESS({ 'Verse#1': [take('a'), take('b'), take('c', 'line', false)] }, 'Verse#1') };
  const m1 = model('build', 'buildBoard', planned)!.instrument!.build!;
  ok('planned: the sections, the first being built, its takes (one still rendering)', m1.planned && J(m1.sections.map(x => [x.id, x.cur])) === J([['Verse#1', true], ['Chorus#1', false]]) && m1.takes.length === 3 && !m1.takes[2]!.ready && s(m1.takes[2]!.meta) === 'rendering…');
  ok('the make button names more takes once there are some; with none, the count of Takes per section (E-09)', s(m1.cur!.make.label) === 'More takes' && s(model('build', 'buildBoard', { ...p.desk, vals: { ...p.desk.vals, 'yue.btakes': 3 } })!.instrument!.build!.cur!.make.label) === 'Generate 3 takes');
  ok('Re-plan names what it would drop before the press (E-04)', s(m1.replanTip) === 'Plans the score again with a new seed (this one is 42) and drops this build\'s 2 takes');
  ok('section mode New chords: each take its own chords', s(m1.cur!.says) === 'Each take gets its own chords and melody.' && m1.cur!.make.ask.ownChords === true);
  const k = act('build', 'buildBoard', planned, 'keep', 'b');
  ok('Keep is the record\'s keep step, named for the take and the section', s(k.steps[0]!.words.label) === 'Keep take 2 for Verse', s(k.steps[0]!.words.label));
  ok('the first keep says the settings are locked', k.told.some(([m, how]) => /now locked/.test(m) && how === undefined) && k.told.some(([m, how]) => m === 'Kept take 2 for Verse. Next: Chorus' && how === 'say'), k.told);
  const kept = { ...k.desk, sessions: planned.sessions };
  const m2 = model('build', 'buildBoard', kept)!.instrument!.build!;
  ok('kept: the section checked, Next offered, the two settings locked as read-outs (E-25)', m2.sections[0]!.kept && m2.cur!.next?.id === 'Chorus#1' && m2.locked?.length === 2 && J(m2.locked.map(l => l.at)) === J([2, 3]) && s(m2.locked[0]!.aria) === 'Follow score: 1.00, locked by the first keep. Re-plan to change it.');
  ok('the foot: 1 of 2 kept, the seed at Lab, Render 1 of 2 sections', s(m2.foot.count) === '1 of 2 kept · seed 42' && s(m2.foot.render) === 'Render 1 of 2 sections' && m2.foot.canRender);
  ok('at Studio the locks do not show, the settings that hold them not being drawn', model('build', 'buildBoard', { ...kept, tier: 'studio' })!.instrument!.build!.locked === null);
  const dk = act('build', 'buildBoard', kept, 'discard', 'b');
  ok('discarding the only keep is a step that frees the locks (E-24)', s(dk.steps[0]!.words.label) === 'Discard take 2' && model('build', 'buildBoard', { ...dk.desk, sessions: planned.sessions })!.instrument!.build!.locked === null && J(dk.told[0]) === J(['Discarded take 2', true]));
  ok('a discarded take is gone from the board', model('build', 'buildBoard', { ...dk.desk, sessions: planned.sessions })!.instrument!.build!.takes.length === 2);
  const uk = act('build', 'buildBoard', kept, 'keep', 'b');
  ok('a second press lets go of the keep, said with the settings freed', s(uk.steps[0]!.words.label) === 'Stop keeping take 2 for Verse' && uk.told[0]![0] === 'Verse is not kept any more. Follow score and Plan strength are free again');
  const r = act('build', 'buildBoard', kept, 'rendered', { song: 'song1', title: 'Built song' });
  ok('Render is the record\'s render step, the build kept (B-06)', s(r.steps[0]!.words.label) === 'Render "Built song"' && J((r.desk.state.build as { songs: string[] }).songs) === J(['song1']) && J((r.desk.state.build as { kept: object }).kept) === J({ 'Verse#1': 'b' }));
  ok('said with Undo', J(r.told[0]) === J(['Saved “Built song” to the Library. The build stays here', true]));
  const rm = model('build', 'buildBoard', { ...r.desk, sessions: planned.sessions })!.instrument!.build!;
  ok('rendered: the count says so, Render again, Start a new build', s(rm.foot.count) === 'Rendered as “Built song” · 1 of 2 kept · seed 42' && s(rm.foot.render) === 'Render again' && rm.foot.reset);
  const nb = act('build', 'buildBoard', { ...r.desk, sessions: planned.sessions }, 'reset');
  ok('Start a new build empties the board and keeps the songs it rendered', s(nb.steps[0]!.words.label) === 'Start a new build' && !model('build', 'buildBoard', nb.desk)!.instrument!.build!.planned && J((nb.desk.state.build as { songs: string[] }).songs) === J(['song1']));
  const rp = act('build', 'buildBoard', kept, 'planned', { session: 's2', seed: 7 });
  ok('Re-plan replaces the build in one step, said with what it dropped (E-04)', rp.steps.length === 1 && s(rp.steps[0]!.words.label) === 'Re-plan the score' && J(rp.told[0]) === J(['Planned a new score. 2 takes were dropped, 1 of them kept', true]), rp.told);
  /* the lyrics change under the build */
  const moved = { ...kept, vals: { ...kept.vals, 'lyrics.text': '[Intro]\nhi\n[Verse]\nnew words\n[Chorus]\nline' } };
  const md = model('build', 'buildBoard', moved)!.instrument!.build!;
  ok('the keep stays with its named section, flagged as sung to the old words, the new one hatched (B-05, E-26)', md.sections.find(x => x.id === 'Verse#1')!.kept && md.sections.find(x => x.id === 'Verse#1')!.stale && md.sections.find(x => x.id === 'Intro#1')!.fresh);
  ok('the banner says what changed and offers Keep building', md.drift?.goon === true && s(md.drift.text) === 'The lyrics changed since this score was planned. 1 kept take was sung to the old words. 1 new section is not built yet.', md.drift && s(md.drift.text));
  const gl = act('build', 'buildBoard', moved, 'relyric');
  const mg = model('build', 'buildBoard', { ...gl.desk, sessions: planned.sessions })!.instrument!.build!;
  ok('Keep building is a step; the plan is the lyrics as they are', s(gl.steps[0]!.words.label) === 'Keep building on the new lyrics' && mg.drift?.goon === false && !mg.sections.some(x => x.fresh));
  ok('the section looked at is the session\'s, so it stays after a keep (B-37)', model('build', 'buildBoard', { ...kept, sessions: SESS({ 'Verse#1': [take('a'), take('b')] }, 'Verse#1') })!.instrument!.build!.cur!.id === 'Verse#1' && model('build', 'buildBoard', { ...kept, sessions: SESS({ 'Verse#1': [take('a'), take('b')] }) })!.instrument!.build!.cur!.id === 'Chorus#1');
  ok('at Quick the board draws none of its settings (E-19)', model('build', 'buildBoard', { ...planned, tier: 'quick' })!.instrument!.build!.ctl === false && model('build', 'buildBoard', { ...planned, tier: 'studio' })!.instrument!.build!.ctl === true);
}

/* ---- Build by section's strip at Quick names Studio for its settings (E-25) */
{
  const t = tileModel(ix, def('build'), desk('yue2', 'quick', ['build'], { 'yue.btakes': 3 }));
  ok('a strip says what is set at another tier, and names the tier its face edits it at', !!t.also && t.also.link === 'studio' && /Takes per section 3/.test(s(t.also.text)), t.also && s(t.also.text));
}

/* ---- the recording span */
{
  ok('arrows move it a second, Shift its end, never past 30 s (B-28)', J(spanMove(10, 30, 200, { key: 'ArrowRight' })) === J([11, 31]) && J(spanMove(10, 30, 200, { key: 'ArrowRight', shift: true })) === J([10, 31]) && J(spanMove(10, 40, 200, { key: 'ArrowRight', shift: true })) === J([10, 40]));
  ok('Page moves it by its length, Home and End to the file\'s ends', J(spanMove(10, 30, 231, { key: 'PageUp' })) === J([30, 50]) && J(spanMove(10, 30, 231, { key: 'End' })) === J([211, 231]) && J(spanMove(10, 30, 231, { key: 'Home' })) === J([0, 20]));
  ok('a drag moves the edge it pressed, carries the part, or draws a new one of at most 30 s (G-B-01)', J(spanMove(10, 30, 200, { mode: 'a', t: 0.025 })) === J([5, 30]) && J(spanMove(10, 30, 200, { mode: 'move', t0: 0.1, t: 0.2, a0: 10, z0: 30 })) === J([30, 50]) && J(spanMove(10, 30, 200, { mode: 'new', t0: 0.5, t: 0.9 })) === J([100, 130]));
  ok('a double-click puts back the first 20 s', J(spanMove(50, 70, 200, { reset: true })) === J([0, 20]));
  const d = desk('yue2', 'lab', ['cont'], {}, { contFile: 'take.wav', contSel: { start: 10, end: 30 } }, { recording: { 'take.wav': 231 } as unknown as Json });
  const m = model('cont', 'recordingSpan', d)!.instrument!.span!;
  ok('the span in the recording\'s times, the ruler in round steps ending on its length (B-28)', s(m.cap) === '0:10 – 0:30 · 20 s of 30 s max' && s(m.text) === '0:10 to 0:30' && J(m.ticks.map(x => x.text)) === J(['0:00', '1:00', '2:00', '3:00', '3:51']));
  ok('a span the recording no longer holds is its first 20 s', (() => { const x = model('cont', 'recordingSpan', { ...d, state: { ...d.state, contSel: { start: 220, end: 240 } } })!.instrument!.span!; return x.a === 0 && x.z === 20; })());
  const k = act('cont', 'recordingSpan', d, 'move', { key: 'ArrowRight' });
  ok('a key is the span\'s move step, in the recording\'s times', s(k.steps[0]!.words.label) === 'Part to continue from 0:10–0:30 → 0:11–0:31');
  const k0 = act('cont', 'recordingSpan', { ...d, state: { contFile: 'take.wav' } }, 'move', { key: 'ArrowRight' });
  ok('the first move names the part it started from (E-23)', s(k0.steps[0]!.words.label) === 'Part to continue from 0:00–0:20 → 0:01–0:21', s(k0.steps[0]!.words.label));
  ok('a key it does not take is refused', act('cont', 'recordingSpan', d, 'move', { key: 'a' }).done === false);
  ok('without a recording it draws nothing', model('cont', 'recordingSpan', desk('yue2', 'lab', ['cont']))!.instrument!.span === undefined);
  ok('a file the page was told the length of says it (the sleeve\'s second line)', s(model('cont', 'dropZone', d)!.state![0]!.detail) === '3:51 long');
}

/* ---- the lead sheet */
{
  const foot = (vals: Desk['vals']) => model('score', 'abcFoot', desk('yue2', 'studio', ['score', 'planner'], vals))!.instrument!.abc!;
  const kinds = (m: { show: { kind: string }[] }) => m.show.map(x => x.kind).join(',');
  ok('empty: Insert example and Plan and preview a score', kinds(foot({})) === 'example,preview' && s(foot({}).preview) === 'Plan and preview a score' && foot({}).note === null);
  ok('Chain of Thought off: the example and the way back, said (E-12)', kinds(foot({ 'lm.cotmode': 'off' })) === 'example,back' && s(foot({ 'lm.cotmode': 'off' }).note!.text) === 'The Planner is set to write nothing down.');
  ok('ABC on the sheet: its preview and its lines counted', kinds(foot({ 'yue.abc': 'X:1\nK:G\nG A | B c |' })) === 'preview' && s(foot({ 'yue.abc': 'X:1\nK:G\nG A | B c |' }).note!.text) === '3 lines. The composer follows this instead of planning one.');
  ok('text that is not ABC is said, in the warning\'s ink (E-26)', foot({ 'yue.abc': 'words' }).note!.warn && !readsAsAbc('words') && readsAsAbc('K:G\nA|'));
  const ask = scoreAsk(ix, desk('yue2', 'studio', ['score'], { 'lyrics.text': '[Verse]\na\n[Chorus]\nb', 'prompt.text': 'folk rock, banjo' }));
  ok('the preview asks the planner for the lyrics\' sections, the title as Generate names a song, and chords for a full lead sheet', J(ask) === J({ abc: '', names: ['Verse', 'Chorus'], title: 'Folk rock', chords: true }), ask);
  ok('Melody only: no chords', scoreAsk(ix, desk('yue2', 'studio', ['score', 'planner'], { 'lm.cotmode': 'melody' }))!.chords === false);
  const info = abcInfo('X:1\nT:Stay a little longer\nM:4/4\nK:G\n"G" G A B d | "C" c B A G |]');
  ok('a pasted sheet is read from its own header (G-B-02)', info.title === 'Stay a little longer' && s(info.meta) === 'Key G · 4/4 · 2 bars · pasted on the lead sheet' && info.file === 'Stay_a_little_longer.abc.txt', s(info.meta));
  const sc: Score = { seed: 9, order: ['Verse', 'Chorus'], meas: [...Array.from({ length: 8 }, (_, i) => ({ b: [0, 2, 4, 8], si: i < 4 ? 0 : 1, first: i % 4 === 0 })), ...Array.from({ length: 4 }, (_, i) => ({ b: [1, 1, 1, 1], si: 1, loop: true, first: i === 0 }))], lyricBars: 8, verdict: 'runaway', est: 30, abc: '', title: 'T', chords: false };
  const L = scoreLayout(sc), M = scoreModel(sc);
  ok('systems of six bars; the playhead\'s places, one a note, on their systems (E-01)', L.rows.length === 2 && L.rows[0]!.bars.length === 6 && M.play.length === 48 && M.play[24]!.row === 1 && M.play[24]!.y === L.rows[1]!.y0 - 6);
  ok('a runaway plan: a repeat sign, "and again", the staff open at its end, said in the meta (E-23)', M.staff.marks.some(m => m.cls === 'sc-sec st-more') && M.staff.marks.filter(m => m.cls === 'st-line open').length === 5 && s(M.meta) === 'seed 9 · plays 0:30, then loops · 2 sections' && s(M.verdict.text) === 'runaway');
  ok('A5 has its ledger line, and the notes play at their pitches', M.staff.marks.some(m => m.cls === 'st-line led') && M.freqs.slice(0, 4).join() === '196,247,294,440');
  ok('no chords on a melody-only plan; a full one carries them', !M.staff.marks.some(m => m.cls === 'st-chord') && scoreModel({ ...sc, chords: true }).staff.marks.some(m => m.cls === 'st-chord'));
  const long = scoreModel({ ...sc, verdict: 'long', meas: sc.meas.map(m => ({ ...m, loop: false, past: m.si === 1 && !!m.loop })) });
  ok('a long plan: bars past the lyrics washed and labelled, said why', long.staff.marks.some(m => m.cls === 'st-past') && long.staff.marks.some(m => m.cls === 'sc-sec st-pastlbl') && /runs 4 bars past the lyrics/.test(s(long.why)));
  /* the lead sheet's paper with Chain of Thought off: it stands back, says why, and its placeholder says so */
  const off = desk('yue2', 'studio', ['score', 'planner'], { 'lm.cotmode': 'off' });
  const ta = model('score', 'textArea', off);
  ok('with Chain of Thought off the sheet stands back and says why', !!ta && nodeShown(ix, node('score', 'textArea'), off) && /Not used while the Planner is set to write nothing down/.test(s(ta.dim)) && /Planner is set to write nothing down/.test(s(ta.params[0]!.placeholder)));
}

if (fails.length) { console.log(`instruments (3c): ${fails.length} of ${n} checks failed:\n  ` + fails.join('\n  ')); process.exit(1); }
console.log(`instruments (3c): ${n} checks of package 3c's core: ok`);
