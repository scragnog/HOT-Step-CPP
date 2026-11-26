// Tests of the core's additions for the prompt card and the lyric sheet (package 4, src/derive/fields.ts), each against
// what the concept does (33-prompt.js, 25-module-lyrics.js, 01-engines.js) and the pending findings the rebuild puts
// right (S-07: one tempo for every estimate).
//
//   node tools/test-fields.ts        exit 1 on a failure

import { DEFINITIONS } from '../src/definitions/index.ts';
import type { EngineId, Node, Tier } from '../src/definitions/types.ts';
import {
  applyChange, composeCaption, copyStep, datasetStep, expandStep, expandWild, hasItem, indexOf, insertTag, latentStep, lyricsPlace, moveToLyrics, plainCaption,
  promptModel, sectionColour, sections, sectionStart, sheetModel, stripLabels, tagStep, tempoStep, toggleItem, trackNow, writeForMe, type Desk, type Msg,
} from '../src/derive.ts';
import { say } from '../src/derive/messages.ts';

const ix = indexOf(DEFINITIONS);
const J = (x: unknown) => JSON.stringify(x);
const fails: string[] = [];
let n = 0;
const ok = (what: string, cond: boolean, detail: unknown = '') => { n++; if (!cond) fails.push(`${what}${detail === '' ? '' : ': ' + (typeof detail === 'string' ? detail : J(detail))}`); };
const s = (m: unknown) => say(m as Msg);
/* the caption datasets as the simulated engine lists them (services/mockEngine.ts), trimmed */
const LISTS: Desk['lists'] = {
  languages: [{ value: 'en', label: 'English' }, { value: 'ja', label: 'Japanese' }, { value: 'fr', label: 'French' }],
  'caption.datasets': [
    { value: 'my-band-ep', label: 'my-band-ep', data: { tracks: [{ title: 'Track 01 · Night drive', bpm: 92, caption: 'synthwave, 92 BPM', structured: 'Global Metadata: synthwave' }, { title: 'Track 04 · Paper boats', bpm: 118, caption: 'indie pop, 118 BPM', structured: 'Global Metadata: indie pop' }], tags: ['synthwave', 'gated drums'] } },
    { value: 'folk-sessions', label: 'folk-sessions', data: { tracks: [{ title: 'Track 02 · Hollow pine', bpm: 84, caption: 'acoustic folk', structured: 'Global Metadata: acoustic folk' }], tags: ['room reverb'] } },
  ],
};
const desk = (engine: EngineId, tier: Tier, on: string[], vals: Desk['vals'] = {}, auto: Desk['auto'] = {}, state: Desk['state'] = {}): Desk =>
  ({ engine, tier, vals, auto, state, lists: LISTS, on: new Set([...DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id), ...on]) });
const sheetNode = DEFINITIONS.find(d => d.id === 'lyrics')!.face!.find(x => x.part === 'lyricSheet') as Node;
const K = 'prompt.text';

/* ---- the lyric sheet's text (25-module-lyrics.js sections, sectionStart, insertTag, fitStrip) */
ok('lines before the first tag are a Verse; each tag starts a section', J(sections('a\n[Chorus]\nb\nc\n\n[Bridge]')) === J([{ name: 'Verse', n: 1 }, { name: 'Chorus', n: 2 }, { name: 'Bridge', n: 0 }]));
ok('a section starts on the line after its tag', sectionStart('[Verse]\nhello there\n\n[Chorus]\n', 0) === 8 && sectionStart('[Verse]\nhello there\n\n[Chorus]\n', 1) === 30);
ok('section colours: by name, a numbered one as its name, an unknown one grey', sectionColour('Chorus 2') === '#e0225a' && sectionColour('Solo') === '#8b8496');
ok('a tag goes in at the end with no caret', J(insertTag('[Verse]\nhello there', null, '[Chorus]')) === J({ text: '[Verse]\nhello there\n\n[Chorus]\n', caret: 30 }));
ok('a tag inside a word goes after the line', insertTag('one two\nthree', 2, '[Bridge]').text === 'one two\n\n[Bridge]\nthree');
ok('an empty sheet takes the tag first', J(insertTag('', null, '[Verse]')) === J({ text: '[Verse]\n', caret: 8 }));
{
  const l = stripLabels(['Pre-Chorus', 'Verse', 'Verse', 'Big Finale Section']);
  ok('a strip label shortens: Pre-Chorus, Pre, P (L-20)', J(l[0]!.slice(-2)) === J(['Pre', 'P']) && l[0]![0] === 'Pre-Chorus', l[0]);
  ok('repeats are numbered when shortened', l[1]!.includes('Ver 1') && l[2]!.includes('Ver 2') && l[2]!.at(-1) === 'V2', [l[1], l[2]]);
  ok('a name of several words is cut with an ellipsis first (Q-22)', l[3]![1]!.endsWith('…'), l[3]);
}

/* ---- the sheet's model: the stat, the stamp, Instrumental, the tabs */
{
  const text = '[Verse]\na\nb\n[Chorus]\nc';
  const m = sheetModel(ix, sheetNode, desk('ace', 'studio', [], { 'lyrics.text': text }))!;
  ok('the line count guesses at 120 BPM with no tempo (E-16)', s(m.stat) === '3 lines · ≈ 0:12 at about 120 BPM', s(m.stat));
  const t = sheetModel(ix, sheetNode, desk('ace', 'studio', ['tempo'], { 'lyrics.text': text, 'tempo.bpm': 90 }, { 'tempo.bpm': false }))!;
  ok('with the BPM field set, at that tempo', s(t.stat) === '3 lines · ≈ 0:16 at 90 BPM', s(t.stat));
  const off = sheetModel(ix, sheetNode, desk('ace', 'studio', [], { 'lyrics.text': text, 'tempo.bpm': 90 }, { 'tempo.bpm': false }))!;
  ok('the field of a Tempo module that is not on the desk is not the tempo', /about 120 BPM/.test(s(off.stat)), s(off.stat));
  const hm = sheetModel(ix, sheetNode, desk('heartmula', 'studio', ['tempo'], { 'lyrics.text': text, 'prompt.text': 'folk, 96 BPM' }))!;
  ok('S-07: a BPM written in the tags is the estimate\'s, and it says where it came from', s(hm.stat) === '3 lines · ≈ 0:15 at 96 BPM (from your tags)', s(hm.stat));
  const instr = sheetModel(ix, sheetNode, desk('ace', 'lab', [], { 'lyrics.text': text, 'lyrics.instr': true }))!;
  ok('Instrumental: no count, no strip, its words, read-only (Q-27, L-22)', instr.instr && s(instr.stat) === '' && !instr.strip.length && s(instr.stripSays) === 'Lyrics are ignored' && !!instr.readOnlyTip);
  ok('Lab has the seven tabs; Studio none', instr.tabs?.length === 7 && m.tabs === null);
  ok('the stamp: at Studio always, at Quick only when not the default (Q-07)', m.stamp && !sheetModel(ix, sheetNode, desk('ace', 'quick', []))!.stamp && sheetModel(ix, sheetNode, desk('ace', 'quick', [], { 'lyrics.lang': 'fr' }))!.stamp);
  ok('no stamp on an engine whose lyrics have no language', !sheetModel(ix, sheetNode, desk('yue2', 'studio', []))!.stamp);
  ok('the placeholder\'s second line is the engine\'s', /planner writes them/.test(s(m.placeholder)) && /empty for an instrumental/.test(s(sheetModel(ix, sheetNode, desk('heartmula', 'studio', []))!.placeholder)));
  ok('an engine with no lyrics: the note, which says lyrics written are kept (S-11)', !lyricsPlace(ix, desk('sa3', 'studio', [], { 'lyrics.text': 'x' })).here && /kept for engines that sing them/.test(s(lyricsPlace(ix, desk('sa3', 'studio', [], { 'lyrics.text': 'x' })).note)) && !/kept/.test(s(lyricsPlace(ix, desk('sa3', 'studio', [])).note)));
}

/* ---- the box (33-prompt.js renderSuggest, BOX) */
{
  ok('the box\'s name per engine (S-05)', ['ace', 'minimax-m3', 'sa3', 'heartmula'].map(e => s(promptModel(ix, K, desk(e as EngineId, 'studio', [])).aria)).join() === 'Describe the song,Caption,Prompt,Tags');
  ok('tags engines show tags as examples', /, /.test(s(promptModel(ix, K, desk('heartmula', 'quick', [])).examples[0])) && s(promptModel(ix, K, desk('heartmula', 'quick', [])).examples[0]) !== s(promptModel(ix, K, desk('ace', 'quick', [])).examples[0]));
  const sug = (e: EngineId, text: string, on: string[] = [], vals: Desk['vals'] = {}, auto: Desk['auto'] = {}) => promptModel(ix, K, desk(e, 'studio', on, { [K]: text, ...vals }, auto)).suggestions.map(x => x.kind).join();
  ok('lyrics and a BPM are noticed', sug('ace', 'slow jam 92 bpm\n[Verse]\nla la') === 'lyrics,bpm');
  ok('no lyrics suggestion on an engine with no lyrics, nor in a Structured Caption', sug('sa3', '[Verse] x') === '' && sug('minimax-m3', 'Global Metadata: x\n[Verse]') === '');
  ok('the BPM is offered only where the tempo field reaches the song (E-38), in range, and not when it is already set', sug('minimax-m3', '92 bpm') === '' && sug('yue2', '92 bpm') === 'bpm' && sug('ace', '300 bpm') === '' && sug('ace', '92 bpm', ['tempo'], { 'tempo.bpm': 92 }, { 'tempo.bpm': false }) === '');
  const w = promptModel(ix, K, desk('ace', 'studio', [], { [K]: 'a {x|y} b {p|q}' })).suggestions[0]!;
  ok('wildcards: the line names them and says what Generate does (E-05)', w.kind === 'wild' && J(w.items) === J(['{x|y}', '{p|q}']) && s(w.lead) === 'Wildcards: ' && /Sent as written/.test(s(w.tail)));
  ok('on an engine that never expands them it offers to expand now', !!promptModel(ix, K, desk('heartmula', 'studio', [], { [K]: '{x|y}' })).suggestions[0]!.link);
  ok('a Structured Caption on another engine is offered as plain words', sug('yue2', 'Global Metadata: x\nVocal Details: y') === 'plain');
  ok('plain words keep the style and the vocals', plainCaption('Global Metadata: smoky jazz trio, warm\nVocal Details: male lead; low\nArrangement: x') === 'smoky jazz trio, warm, male lead, low');
  ok('an item of the box, as a tag is in it or not (E-31)', hasItem('synthwave-ish, warm', 'warm') && !hasItem('synthwave-ish, warm', 'synthwave') && toggleItem('a, b', 'b') === 'a' && toggleItem('a', 'c') === 'a, c');
  let r = 0; const rnd = () => (r++ % 2 ? 0.9 : 0.1);
  ok('one option of each wildcard', expandWild('{a|b} and {c|d}', rnd) === 'a and d');
}

/* ---- the caption a dataset supplies (E-06, E-08, E-22) */
{
  const mm = (vals: Desk['vals'], on: string[] = []) => desk('minimax-m3', 'studio', on, vals);
  ok('Custom supplies nothing', trackNow(ix, mm({})) === null && !promptModel(ix, K, mm({ [K]: 'mine' })).locked);
  const p = promptModel(ix, K, mm({ [K]: 'mine', 'cap.src': 'Track 04 · Paper boats' }));
  ok('a track locks the box with its Structured Caption on MiniMax-Music3, and says where it comes from', p.shown === 'Global Metadata: indie pop' && s(p.locked!.line) === 'From my-band-ep · Track 04 · Paper boats' && !p.tools.write);
  ok('the locked box says why in the control\'s own noun (E-31)', s(p.locked!.says) === 'The caption comes from the dataset (Caption source: Track 04 · Paper boats). Edit a copy to change it.', s(p.locked!.says));
  ok('Automatic takes the track nearest the tempo field (100 without one), on YuE2 its words', trackNow(ix, desk('yue2', 'studio', [], { 'cap.src': 'auto' }))!.track.title === 'Track 01 · Night drive' && trackNow(ix, desk('yue2', 'studio', ['tempo'], { 'cap.src': 'auto', 'tempo.bpm': 118 }, { 'tempo.bpm': false }))!.track.title === 'Track 04 · Paper boats' && promptModel(ix, K, desk('yue2', 'studio', [], { 'cap.src': 'auto' })).shown === 'synthwave, 92 BPM');
  const c = copyStep(ix, K, mm({ [K]: 'mine', 'cap.src': 'Track 04 · Paper boats' }))!;
  ok('Edit a copy: the supplied caption becomes the box\'s text, the source Custom, one step', J(c.change.vals) === J({ [K]: 'Global Metadata: indie pop', 'cap.src': 'custom' }) && s(c.change.words.label) === 'Edit a copy of the caption', s(c.change.words.label));
  const t = tagStep(ix, K, desk('yue2', 'studio', [], { 'cap.src': 'auto' }), 'gated drums')!;
  ok('a tag on a supplied caption edits a copy of it, one step, and says so', t.change.vals[K] === 'synthwave, 92 BPM, gated drums' && t.change.vals['cap.src'] === 'custom' && s(t.change.words.label) === 'Add "gated drums"' && /Editing a copy/.test(s(t.said)));
  const ds = datasetStep(ix, desk('yue2', 'studio', [], { 'cap.src': 'Track 04 · Paper boats' }), 'folk-sessions')!;
  ok('a dataset without the track puts the source on Custom in the same step, and says so (E-22)', ds.change.vals['cap.src'] === 'custom' && /Paper boats isn't in folk-sessions/.test(s(ds.said)));
  ok('a dataset with it leaves the source', datasetStep(ix, desk('yue2', 'studio', [], { 'cap.src': 'Track 04 · Paper boats', 'cap.ds': 'folk-sessions' }), 'my-band-ep')!.change.vals['cap.src'] === undefined);
  ok('at Quick, the From chip only while the source uses it (E-22)', !promptModel(ix, K, desk('yue2', 'quick', [])).tools.dataset && promptModel(ix, K, desk('yue2', 'quick', [], { 'cap.src': 'auto' })).tools.dataset);
}

/* ---- the tools' steps */
{
  const w = writeForMe(ix, K, desk('ace', 'studio', []));
  ok('Write it for me fills an empty box and empty lyrics, one step, said with what it wrote', 'change' in w && /intimate room sound/.test(String(w.change.vals[K])) && /Kettle on/.test(String(w.change.vals['lyrics.text'])) && s(w.change.words.label) === 'Write it for me' && s(w.said) === 'Wrote a description and lyrics (simulated)', 'change' in w ? s(w.said) : w);
  ok('on a tags engine it writes tags (S-33)', (r => 'change' in r && s(r.said) === 'Wrote tags and lyrics (simulated)' && !/room sound/.test(String(r.change.vals[K])))(writeForMe(ix, K, desk('heartmula', 'studio', []))));
  ok('with both filled it asks (F-17)', 'ask' in writeForMe(ix, K, desk('ace', 'studio', [], { [K]: 'x', 'lyrics.text': 'y' })));
  ok('the answer writes new lyrics', (r => 'change' in r && r.change.vals[K] === 'x' && s(r.said) === 'Wrote new lyrics (simulated)')(writeForMe(ix, K, desk('ace', 'studio', [], { [K]: 'x', 'lyrics.text': 'y' }), true)));
  ok('a filled box on an engine with no lyrics says why', (r => 'says' in r && /Stable Audio 3 has no lyrics/.test(s(r.says)))(writeForMe(ix, K, desk('sa3', 'studio', [], { [K]: 'x' }))));
  const c = composeCaption(ix, K, desk('minimax-m3', 'studio', ['tempo'], { [K]: 'dream pop, hazy', 'tempo.bpm': 100, 'tempo.key': 'A minor', 'lyrics.lang': 'ja' }, { 'tempo.bpm': false }));
  ok('Compose caption builds on the first words with the BPM, the key and the language (E-14)', 'change' in c && /^Global Metadata: dream pop, 100 BPM, A minor, warm and intimate\nVocal Details: lead vocal, sung in Japanese, close-mic/.test(String(c.change.vals[K])) && s(c.said) === 'Compose caption built a Structured Caption with your BPM and key', 'change' in c ? c.change.vals[K] : c);
  ok('it needs words, and not twice', 'says' in composeCaption(ix, K, desk('minimax-m3', 'studio', [])) && (r => 'says' in r && /already/.test(s(r.says)))(composeCaption(ix, K, desk('minimax-m3', 'studio', [], { [K]: 'Global Metadata: x' }))));
  const e = expandStep(ix, K, desk('ace', 'lab', [], { 'lyrics.text': 'go {fast|slow}' }), 'lyrics', () => 0);
  ok('Expand in the lyrics: one option, one step named for it', 'change' in e && e.change.vals['lyrics.text'] === 'go fast' && s(e.change.words.label) === 'Expand wildcards');
  ok('with none it says so', (r => 'says' in r && s(r.says) === 'No {a|b|c} wildcards in the lyrics')(expandStep(ix, K, desk('ace', 'lab', []), 'lyrics', () => 0)));
  const m = moveToLyrics(ix, K, desk('ace', 'studio', [], { [K]: 'slow jam, [Verse]\nla la', 'lyrics.text': '[Intro]\nhm' }))!;
  ok('Move to lyrics: under the lyrics there, the box keeps what came before', m.change.vals['lyrics.text'] === '[Intro]\nhm\n\n[Verse]\nla la' && m.change.vals[K] === 'slow jam' && s(m.said) === 'Added below the lyrics you already had');
  const tp = tempoStep(ix, K, desk('ace', 'studio', [], { [K]: 'slow 92 bpm' }))!;
  ok('Set the tempo field: the BPM on, at the box\'s tempo, bringing Tempo when it is not on the desk', tp.vals['tempo.bpm'] === 92 && tp.auto['tempo.bpm'] === false && tp.brings === 'tempo' && s(tp.words.label) === 'Set the tempo to 92 BPM' && /Tempo module is on the desk/.test(s(tp.said)), s(tp.words.label));
  const lt = latentStep(ix, K, desk('ace', 'studio', [], { [K]: 'mine' }), 'night.latent');
  ok('a latent brings its tempo and key and fills empty lyrics, not the box, one step', 'state' in lt && lt.state.latent === 'night.latent' && lt.vals['tempo.bpm'] === 96 && !(K in lt.vals) && !!lt.vals['lyrics.text'] && s(lt.words.label) === 'Import latent' && lt.brings === 'tempo');
  ok('a file that is not a latent is refused', (r => 'says' in r && s(r.says) === 'song.wav isn\'t a .latent or .hslat file')(latentStep(ix, K, desk('ace', 'studio', []), 'song.wav')));
  void applyChange;
}

if (fails.length) { console.log(`fields: ${fails.length} of ${n} checks fail`); fails.forEach(f => console.log('  FAIL ' + f)); process.exitCode = 1; }
else console.log(`fields: ${n} checks of the prompt card's and the lyric sheet's core: ok`);
