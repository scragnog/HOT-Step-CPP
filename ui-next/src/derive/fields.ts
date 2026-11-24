// The words the page types most: the prompt card and the lyric sheet (the concept's 33-prompt.js and
// 25-module-lyrics.js). What each draws (sheetModel, promptModel: the sections and their colours, the line count and its
// estimate, the suggestions, the caption a dataset supplies, which tools show), and what each of their tools writes, as
// pure functions of the definitions and the desk; a tool's step is one of the prompt's named steps (prompt.ts `steps`),
// so the core names it as it names every step. Nothing here reads the page.

import type { Definition, EngineId, Json, Node, Value } from '../definitions/types.ts';
import { engineName } from '../definitions/engines.ts';
import { type Desk, type Index, applies, changed, has, isAuto, mmss, onDesk, valueOf } from './core.ts';
import { EN, type Msg, type Slot, msg, say } from './messages.ts';
import { optionsOf, paramWords, textMsg } from './strings.ts';
import { optionFaces, songTempo, type OptionFace } from './controls.ts';
import { applyChange, type Change } from './parts.ts';
import { stepWords } from './steps.ts';
import { setValue } from './values.ts';
import { boxWord } from './workflows.ts';

/* ---------------------------------------------------------------- the main box's word on each engine (S-05) */

/** What the main text box is called on an engine, everywhere the page names it (the concept's BOX, 01-engines.js):
 *  MiniMax-Music3 reads a caption, Stable Audio 3 a prompt, HeartMuLa tags, the others a description. `tags`: the box
 *  holds tags, and its examples are tags. */
const BOX: { [engine: string]: { w: string; tags?: boolean } } = { 'minimax-m3': { w: 'caption' }, sa3: { w: 'prompt' }, heartmula: { w: 'tags', tags: true } };
const boxKey = (desk: Desk) => BOX[desk.engine]?.w ?? 'description';
/** The box's word with its article ("a caption", "some tags"), and whether it is a plural; the word itself is
 *  workflows.ts boxWord, the one statement of it. */
export const boxA = (desk: Desk): Msg => msg(`box.${boxKey(desk)}.a`);
/** The example a refusal offers to put in the box: the first of the box's examples (33-prompt.js PH, PHT), as text. */
export const exampleText = (e: EngineId): string => say(msg(`prompt.ph.${BOX[e]?.tags ? 'tags' : 'words'}.0`));
const boxPl = (desk: Desk): boolean => boxKey(desk) === 'tags';

/** The engines whose tempo field reaches the song (E-38): there a BPM written in the box is offered to the field. */
const TEMPO_IN_SONG = ['ace', 'yue2'];
/** The engine that structures its caption: Compose caption is its, and a Structured Caption is its own. */
const STRUCTURED = 'minimax-m3';

const own = (ix: Index, k: string) => ix.params.get(k);
/** A parameter on the desk in force: its module on the desk, on the engine. */
const here = (ix: Index, k: string, desk: Desk) => { const h = own(ix, k); return !!h && onDesk(h.d, desk) && has(ix, k, desk.engine); };
const str = (v: Value | undefined) => (typeof v === 'string' ? v : '');
const read = (ix: Index, k: string, desk: Desk): Value | undefined => (own(ix, k) && has(ix, k, desk.engine) ? valueOf(ix, k, desk) : undefined);

/* ---------------------------------------------------------------- the lyric sheet */

/** The section tags the sheet's tabs insert, in order, and each section's colour (the strip, the tabs). */
export const SECTION_TAGS = ['Intro', 'Verse', 'Pre-Chorus', 'Chorus', 'Bridge', 'Outro', 'Hook'] as const;
const SECTION_COLOUR: { [name: string]: string } = { verse: '#3f5ef0', chorus: '#e0225a', 'pre-chorus': '#9440f5', bridge: '#0d7f62', intro: '#6b7280', outro: '#6b7280', hook: '#a85a00' };
export const sectionColour = (name: string): string => SECTION_COLOUR[name.toLowerCase().replace(/\s*\d+$/, '')] ?? '#8b8496';

/** The sections of a lyric sheet, in order: each [Tag] starts one; lines before the first tag are a Verse. `n` counts
 *  its lines that hold words. */
export function sections(text: string): { name: string; n: number }[] {
  const out: { name: string; n: number }[] = []; let cur: { name: string; n: number } | null = null;
  for (const l of text.split('\n')) {
    const m = /^\[([^\]]+)\]/.exec(l.trim());
    if (m) { cur = { name: m[1]!.trim(), n: 0 }; out.push(cur); } else if (l.trim()) { if (!cur) { cur = { name: 'Verse', n: 0 }; out.push(cur); } cur.n++; }
  }
  return out;
}
/** Where section j starts: the line after its tag (or the first line, for untagged lines at the top). */
export function sectionStart(text: string, j: number): number {
  let k = -1, pos = 0, implicit = false; const ls = text.split('\n');
  for (let i = 0; i < ls.length; i++) {
    const l = ls[i]!;
    if (/^\s*\[[^\]]+\]/.test(l.trim())) { k++; if (k === j) return pos + l.length + (i < ls.length - 1 ? 1 : 0); }
    else if (l.trim() && k < 0 && !implicit) { implicit = true; k++; if (k === j) return pos; }
    pos += l.length + 1;
  }
  return text.length;
}
/** A section tag put in at a line boundary, never inside a word: after the line the caret is on (the end with no caret),
 *  a blank line before it unless there is one, and the caret on the line after it. */
export function insertTag(text: string, caret: number | null, tag: string): { text: string; caret: number } {
  let pos = caret ?? text.length;
  if (pos > 0 && text[pos - 1] !== '\n') { const eol = text.indexOf('\n', pos); pos = eol < 0 ? text.length : eol; }
  const before = text.slice(0, pos), after = text.slice(pos);
  const pre = !before ? '' : before.endsWith('\n\n') ? '' : before.endsWith('\n') ? '\n' : '\n\n', ins = pre + tag + (after.startsWith('\n') ? '' : '\n');
  return { text: before + ins + after, caret: pos + ins.length + (after.startsWith('\n') ? 1 : 0) };
}

/** What a song's tempo is for an estimate (S-07): the BPM field when it is on the desk and set, else a BPM written in the
 *  box, else none. `from`: where it came from. */
export function estimateTempo(ix: Index, desk: Desk, bpmKey: string | undefined, textKey: string | undefined): { bpm: number | null; from: 'field' | 'box' | null } {
  const field = bpmKey && here(ix, bpmKey, desk) ? { value: valueOf(ix, bpmKey, desk) ?? null, auto: isAuto(ix, bpmKey, desk) } : undefined;
  const text = textKey ? read(ix, textKey, desk) : undefined;
  const bpm = songTempo(field, text);
  return { bpm, from: bpm === null ? null : field && !field.auto ? 'field' : 'box' };
}

export interface SheetSection { name: string; n: number; colour: string; start: number; tip: Msg; aria: Msg; labels: string[] }
export interface SheetModel {
  lab: boolean;
  /** Instrumental is on: the text is kept but not sung, and everything that edits or measures it stands back. */
  instr: boolean;
  /** The section tabs (Lab): each tag, its colour and its tip. */
  tabs: { tag: string; colour: string; tip: Msg }[] | null;
  tabsLabel: Msg;
  /** The language stamp shows: at Quick only when it is not the default (Q-07); each language's face. */
  stamp: boolean;
  faces: OptionFace[];
  placeholder: Msg;
  readOnlyTip: Msg | null;
  /** The structure strip: its sections, or what it says instead. */
  strip: SheetSection[];
  stripSays: Msg | null;
  /** The line count and its estimate ('' with Instrumental on, or no lines: Q-27). */
  stat: Slot;
  /** The wildcard tools (Lab): Expand now, and Expand on Generate where the engine has it. */
  wild: { expand: Msg; tip: Msg; auto: boolean } | null;
}
/** The lyric sheet on the desk: `node` is its node, which binds the text (param) and the language, and reads Instrumental,
 *  the BPM field and the box (in that order). */
export function sheetModel(ix: Index, node: Node, desk: Desk): SheetModel | undefined {
  const key = node.param; if (!key || !has(ix, key, desk.engine)) return undefined;
  const [instrKey, bpmKey, boxKey_] = node.reads ?? [];
  const lab = desk.tier === 'lab', text = str(valueOf(ix, key, desk)), instr = !!instrKey && read(ix, instrKey, desk) === true;
  const lang = (node.params ?? []).find(k => own(ix, k)?.p.type === 'choice'), wildKey = (node.params ?? []).find(k => own(ix, k)?.p.type === 'boolean');
  const secs = sections(text), lines = secs.reduce((a, s) => a + s.n, 0);
  const t = estimateTempo(ix, desk, bpmKey, boxKey_), bpm = t.bpm ?? 120;
  const stat: Slot = instr || !lines ? '' : msg('lyrics.stat', { n: lines, time: mmss(lines * 8 * 60 / bpm), bpm, how: t.from ?? 'guess', box: boxWord(ix, desk) });
  const ph = paramWords(ix, key, desk), labels = stripLabels(secs.map(s => s.name));
  return {
    lab, instr,
    tabs: lab ? SECTION_TAGS.map(x => ({ tag: `[${x}]`, colour: sectionColour(x), tip: msg('lyrics.tab.tip', { tag: `[${x}]` }) })) : null,
    tabsLabel: msg('lyrics.tabs'),
    stamp: !!lang && applies(ix, lang, desk) && (desk.tier !== 'quick' || changed(ix, lang, desk)),
    faces: lang && has(ix, lang, desk.engine) ? optionFaces(ix, lang, desk, 'stamp') : [],
    placeholder: msg('lyrics.placeholder', { more: ph.face.placeholder ? { id: `create.${own(ix, key)!.d.id}.p.${key.replace(/\./g, '-')}.placeholder${(own(ix, key)!.p.per || []).some(l => 'engines' in l && l.engines.includes(desk.engine) && 'placeholder' in l) ? '@' + ((own(ix, key)!.p.per || []).find(l => 'engines' in l && l.engines.includes(desk.engine) && 'placeholder' in l) as { engines: string[] }).engines.join('+') : ''}`, en: ph.face.placeholder } : '' }),
    readOnlyTip: instr ? msg('lyrics.instrTip') : null,
    strip: instr ? [] : secs.map((s, j) => ({ name: s.name, n: s.n, colour: sectionColour(s.name), start: sectionStart(text, j), tip: msg('lyrics.sec.tip', { name: s.name, n: s.n }), aria: msg('lyrics.sec.aria', { name: s.name, n: s.n }), labels: labels[j]! })),
    stripSays: instr ? msg('lyrics.ignored') : secs.length ? null : msg('lyrics.noSections'),
    stat,
    wild: lab ? { expand: msg('lyrics.expand'), tip: msg('wild.tip'), auto: !!wildKey && applies(ix, wildKey, desk) } : null,
  };
}

/** The short labels a strip segment tries, longest first, until one fits (L-20, Q-22): the full name, cut with an
 *  ellipsis word by word when it is several words, its first three letters with its number, its initial with its number.
 *  A repeated section is numbered when shortened, so the repeats stay apart. */
export function stripLabels(names: string[]): string[][] {
  const base = (s: string) => s.replace(/\s*\d+$/, ''), seen: { [k: string]: number } = {};
  return names.map(full => {
    const short = base(full), k = short.toLowerCase(), n = (seen[k] = (seen[k] ?? 0) + 1), many = names.filter(x => base(x).toLowerCase() === k).length > 1;
    const num = /\d+$/.exec(full)?.[0] ?? (many ? String(n) : ''), abbr = short.split(/[-\s]/)[0]!.slice(0, 3);
    const cuts = /\s/.test(short) && short.length > 8 ? Array.from({ length: short.length - 4 }, (_, i) => short.slice(0, short.length - 1 - i).trimEnd() + '…') : [];
    return [full, ...cuts, abbr + (num ? ' ' + num : ''), short.slice(0, 1).toLocaleUpperCase() + num];
  });
}

/** Where the lyric sheet goes on the desk's engine: a sheet, or the note that the engine has no lyrics (S-11: kept lyrics
 *  are said kept). */
export function lyricsPlace(ix: Index, desk: Desk): { here: boolean; note: Msg | null } {
  const d = ix.defs.find(x => x.kind === 'module' && x.builtIn && (x.face ?? []).some(n => n.part === 'lyricSheet'));
  const key = (d?.face ?? []).find(n => n.part === 'lyricSheet')?.param;
  if (!d || !key) return { here: false, note: null };
  const here_ = d.engines.includes(desk.engine);
  return { here: here_, note: here_ ? null : msg('lyrics.none', { engine: engineName(desk.engine), box: boxWord(ix, desk), kept: String(!!str(desk.vals[key]).trim()) }) };
}

/* ---------------------------------------------------------------- the prompt card */

const WILD = /\{[^{}]*\|[^{}]*\}/g;
const STRUCT = /^Global Metadata:/m;
const LYRICISH = /\[(verse|chorus|bridge|intro|outro|pre-chorus|hook)/i;

/** One option of each {a|b|c} wildcard, chosen by `rnd` (0 to 1). */
export const expandWild = (t: string, rnd: () => number): string => t.replace(/\{([^{}]*\|[^{}]*)\}/g, (_, g: string) => { const o = g.split('|'); return o[Math.floor(rnd() * o.length)]!.trim(); });
/** A Structured Caption's Global Metadata and Vocal Details as plain words (carried item). */
export const plainCaption = (p: string): string => { const g = /^Global Metadata:\s*(.+)$/m.exec(p)?.[1], v = /^Vocal Details:\s*(.+)$/m.exec(p)?.[1]; return g ? [g, v].filter(Boolean).join(', ').replace(/;/g, ',') : p; };
const parts = (t: string) => t.split(',').map(x => x.trim()).filter(Boolean);
/** A comma-separated item of the box, as a tag is in it or not (E-31). */
export const hasItem = (t: string, item: string): boolean => parts(t).some(x => x.toLowerCase() === item.toLowerCase());
export const toggleItem = (t: string, item: string): string => (hasItem(t, item) ? parts(t).filter(x => x.toLowerCase() !== item.toLowerCase()) : [...parts(t), item]).join(', ');

/** A caption dataset's tracks, as the server lists them (desk.lists 'caption.datasets', each with its `data`). */
export interface Track { title: string; bpm: number; caption: string; structured?: string }
interface Dataset { tracks: Track[]; tags: string[] }
const datasetOf = (desk: Desk, name: string): { name: string; set: Dataset } | null => {
  const l = desk.lists?.['caption.datasets'] ?? [], x = l.find(o => o.value === name) ?? (name === '' ? l[0] : undefined);
  const set = x?.data as unknown as Dataset | undefined;
  return x && set && Array.isArray(set.tracks) ? { name: String(x.value), set } : null;
};

export interface Suggestion {
  kind: 'lyrics' | 'bpm' | 'wild' | 'plain';
  /** A button: its words and the bold words after them; a note: its words around the wildcards, and its link. */
  lead: Msg; bold?: Msg; items?: string[]; tail?: Msg; link?: Msg; bpm?: number;
}
export interface PromptModel {
  /** The box's accessible name and its example lines (the placeholder turns through them while the box is empty). */
  aria: Msg; examples: Msg[];
  /** What the box shows: the text in force, or a caption a dataset supplies (locked). */
  shown: string;
  locked: { text: string; line: Msg; says: Msg } | null;
  suggestions: Suggestion[];
  wildcards: boolean;
  /** The tools that show. */
  tools: { write: boolean; presets: boolean; compose: boolean; latent: boolean; expand: boolean; wildAuto: boolean; trigger: boolean; source: boolean; dataset: boolean; tags: boolean };
  /** The caption source's chip tip (with the track Automatic resolves to), the dataset tags and their heading. */
  sourceTip: Msg | null;
  tags: { tag: string; on: boolean }[];
  tagsTip: Msg;
  /** Song info: something is filled in; the cover art subject shows (the Cover art unit is on). */
  info: { any: boolean; art: boolean };
  latent: string | null;
}

const KEYS = { src: 'cap.src', ds: 'cap.ds', wild: 'wild.auto', trig: 'ad.trigger', art: 'pp.art.subject', info: ['song.artist', 'song.title', 'song.subject'], lyrics: 'lyrics.text', instr: 'lyrics.instr', bpm: 'tempo.bpm', key: 'tempo.key', gender: 'lyrics.gender', lang: 'lyrics.lang' };

/** The track a caption source stands for: none on Custom; on Automatic the track nearest the tempo (100 without one). */
export function trackNow(ix: Index, desk: Desk): { ds: string; track: Track } | null {
  if (!has(ix, KEYS.src, desk.engine)) return null;
  const src = str(valueOf(ix, KEYS.src, desk)), ds = datasetOf(desk, str(read(ix, KEYS.ds, desk)));
  if (!src || src === 'custom' || !ds) return null;
  if (src === 'auto') {
    const t = estimateTempo(ix, desk, KEYS.bpm, undefined), bpm = t.from === 'field' ? t.bpm! : 100;
    const best = ds.set.tracks.slice().sort((a, b) => Math.abs(a.bpm - bpm) - Math.abs(b.bpm - bpm))[0];
    return best ? { ds: ds.name, track: best } : null;
  }
  const tr = ds.set.tracks.find(x => x.title === src);
  return tr ? { ds: ds.name, track: tr } : null;
}
/** The caption a track supplies on the desk's engine: MiniMax-Music3 its Structured Caption, the others its words. */
const trackText = (desk: Desk, t: Track) => (desk.engine === STRUCTURED && t.structured ? t.structured : t.caption);
const srcWords = (ix: Index, desk: Desk): Slot => { const o = optionsOf(ix, KEYS.src, desk).find(x => x.value === valueOf(ix, KEYS.src, desk)); return o?.label ?? str(valueOf(ix, KEYS.src, desk)); };
/** The one sentence for a supplied caption, in the control's own noun (E-31): "… Edit a copy to change it." */
export const lockedMsg = (ix: Index, desk: Desk, what: 'change' | 'write' | 'reuse' = 'change'): Msg =>
  msg('prompt.locked', { box: boxWord(ix, desk), label: paramWords(ix, KEYS.src, desk).label, src: srcWords(ix, desk), what });

/** The prompt card on the desk: `key` is its box's parameter. */
export function promptModel(ix: Index, key: string, desk: Desk): PromptModel {
  const text = str(valueOf(ix, key, desk)), tr = trackNow(ix, desk), shown = tr ? trackText(desk, tr.track) : text;
  const lyricsHere = has(ix, KEYS.lyrics, desk.engine), structured = desk.engine === STRUCTURED;
  const sug: Suggestion[] = [];
  if (LYRICISH.test(shown) && !STRUCT.test(shown) && lyricsHere) sug.push({ kind: 'lyrics', lead: msg('sg.lyrics'), bold: msg('sg.lyrics.b') });
  const b = /(\d{2,3})\s?bpm/i.exec(shown);
  if (b && TEMPO_IN_SONG.includes(desk.engine) && +b[1]! >= 40 && +b[1]! <= 220 && !(here(ix, KEYS.bpm, desk) && !isAuto(ix, KEYS.bpm, desk) && valueOf(ix, KEYS.bpm, desk) === +b[1]!)) sug.push({ kind: 'bpm', lead: msg('sg.bpm'), bold: msg('sg.bpm.b', { bpm: +b[1]! }), bpm: +b[1]! });
  const w = shown.match(WILD);
  if (w) {
    const autoHere = has(ix, KEYS.wild, desk.engine) && applies(ix, KEYS.wild, desk), on = autoHere && valueOf(ix, KEYS.wild, desk) === true;
    sug.push({ kind: 'wild', lead: msg('sg.wild', { n: w.length }), items: [...w], tail: !autoHere ? msg('sg.wild.never', { engine: engineName(desk.engine) }) : on ? msg('sg.wild.each') : msg('sg.wild.asWritten'), ...(!autoHere ? { link: msg('sg.wild.now') } : {}) });
  }
  if (STRUCT.test(shown) && !structured) sug.push({ kind: 'plain', lead: msg('sg.plain', { engine: engineName(desk.engine) }), bold: msg('sg.plain.b', { box: boxWord(ix, desk), pl: String(boxPl(desk)) }) });
  const ds = has(ix, KEYS.src, desk.engine) ? datasetOf(desk, str(read(ix, KEYS.ds, desk))) : null;
  const latent = desk.state.latent;
  return {
    aria: paramWords(ix, key, desk).label,
    examples: [0, 1, 2, 3].map(i => msg(`prompt.ph.${BOX[desk.engine]?.tags ? 'tags' : 'words'}.${i}`)),
    shown,
    locked: tr ? { text: shown, line: msg('prompt.lockLine', { ds: tr.ds, track: tr.track.title }), says: lockedMsg(ix, desk) } : null,
    suggestions: sug,
    wildcards: !!w,
    tools: {
      write: !tr, presets: !tr, compose: structured && !tr, latent: latentHere(ix, desk),
      expand: !!w && !tr, wildAuto: !!w && has(ix, KEYS.wild, desk.engine) && applies(ix, KEYS.wild, desk),
      trigger: has(ix, KEYS.trig, desk.engine) && applies(ix, KEYS.trig, desk),
      source: has(ix, KEYS.src, desk.engine) && applies(ix, KEYS.src, desk),
      dataset: has(ix, KEYS.src, desk.engine) && (desk.tier !== 'quick' || str(valueOf(ix, KEYS.src, desk)) !== 'custom'),
      tags: has(ix, KEYS.src, desk.engine) && !structured && desk.tier !== 'quick',
    },
    sourceTip: has(ix, KEYS.src, desk.engine) ? msg('prompt.srcTip', { label: paramWords(ix, KEYS.src, desk).label, src: srcWords(ix, desk), track: tr && valueOf(ix, KEYS.src, desk) === 'auto' ? msg('prompt.srcTip.auto', { track: tr.track.title.split(' · ')[0]!, bpm: tr.track.bpm }) : '' }) : null,
    tags: ds ? ds.set.tags.map(t => ({ tag: t, on: hasItem(shown, t) })) : [],
    tagsTip: msg('prompt.tagsTip', { box: boxWord(ix, desk) }),
    info: { any: KEYS.info.some(k => !!str(read(ix, k, desk))), art: !!own(ix, KEYS.art) && here(ix, KEYS.art, desk) && applies(ix, KEYS.art, desk) && (() => { const u = own(ix, KEYS.art)!.d; return u.kind !== 'module' && !!u.on && valueOf(ix, u.on, desk) === true; })() },
    latent: typeof latent === 'string' && latent && latentHere(ix, desk) ? latent : null,
  };
}
/** The latent the next render continues from: a state of the prompt's, sent on the engines its wire names. */
function latentHere(ix: Index, desk: Desk): boolean {
  const s = ix.states.get('latent'); if (!s) return false;
  const w = s.s.wire; return Array.isArray(w) && w.some(x => (x as { engines: string[] }).engines.includes(desk.engine));
}

/* ---------------------------------------------------------------- what each tool writes */

/** A named step of the box's parameter that sets these values (cleaned on the engine in force), with what it says
 *  after it is done; null when it changes nothing. */
export interface ToolStep { change: Change; said: Msg | null; undo: boolean }
/** A tool's step is one of the desk's (the concept's deskUndo): what Undo puts back ends with "Anything else you did
 *  since stays as it is", as every desk step's does (RC-16). */
const asDesk = (w: Change['words']): Change['words'] => (w.put.id === 'step.put' ? { ...w, put: msg('step.put.desk', w.put.slots ?? {}) } : w);
function named(ix: Index, desk: Desk, key: string, step: string, vals: { [k: string]: Value }, auto: { [k: string]: boolean } = {}, slots?: { [s: string]: string | number }): Change | null {
  const v: Change['vals'] = {};
  for (const [k, x] of Object.entries(vals)) if (has(ix, k, desk.engine)) v[k] = setValue(ix, desk, k, x).value;
  const after = applyChange(desk, { vals: v, auto, state: {}, words: { label: msg(''), put: msg('') } });
  const words = stepWords(ix, desk, after, { named: { key, step, ...(slots ? { slots } : {}) } });
  return words ? { vals: v, auto, state: {}, words: asDesk(words) } : null;
}
/* what Write it for me writes (simulated): content the engine reads, in its words; on a tags engine, tags (S-33) */
const WRITTEN = { words: 'warm indie folk, fingerpicked acoustic guitar, soft female vocals, intimate room sound', tags: 'indie folk, acoustic guitar, fingerpicked, warm, soft female vocals' };
const LYRICS_WRITTEN = '[Verse]\nKettle on, the window fogged\nYour coat still hanging by the door\n\n[Chorus]\nStay a little longer, stay\nLet the morning wait\n\n[Verse]\nRadio is playing low\nSomething that we used to know\n\n[Bridge]\nNo one has to leave today\n\n[Chorus]\nStay a little longer, stay';

/** Write it for me (F-17): it fills what is empty, the box and the lyrics; with everything filled it asks before
 *  replacing the lyrics (`ask`), and a press that would change nothing says why. `replace`: the answer to that question. */
export function writeForMe(ix: Index, key: string, desk: Desk, replace = false): ToolStep | { ask: Msg } | { says: Msg } {
  const locked = !!trackNow(ix, desk), text = str(valueOf(ix, key, desk));
  const lyrOk = has(ix, KEYS.lyrics, desk.engine) && read(ix, KEYS.instr, desk) !== true, hadLyr = lyrOk && !!str(read(ix, KEYS.lyrics, desk)).trim();
  const cap = !replace && !text.trim() && !locked, lyr = lyrOk && (!hadLyr || replace);
  if (!cap && !lyr) {
    if (hadLyr && !locked) return { ask: msg('write.ask') };
    return { says: locked ? lockedMsg(ix, desk, 'write') : msg('write.full', { box: boxWord(ix, desk), pl: String(boxPl(desk)), why: has(ix, KEYS.lyrics, desk.engine) ? 'instr' : 'none', engine: engineName(desk.engine) }) };
  }
  const vals: { [k: string]: Value } = { [key]: cap ? WRITTEN[BOX[desk.engine]?.tags ? 'tags' : 'words'] : text };
  if (lyr) vals[KEYS.lyrics] = LYRICS_WRITTEN;
  const change = named(ix, desk, key, 'write', vals);
  const what = desk.engine === STRUCTURED ? msg('write.plain') : boxPl(desk) ? boxWord(ix, desk) : boxA(desk);
  return change ? { change, undo: true, said: msg('write.said', { what, kind: cap && lyr ? 'both' : cap ? 'box' : replace ? 'new' : 'lyrics' }) } : { says: msg('write.full', { box: boxWord(ix, desk), pl: String(boxPl(desk)), why: 'instr', engine: engineName(desk.engine) }) };
}

/** Compose caption (MiniMax-Music3): a Structured Caption built on the box's first words, the BPM and key when the
 *  Tempo module sets them, and the vocals (the gender, the language when it is chosen: E-14). The caption is what the
 *  engine reads, so it is written in the words the engine reads, not the page's language. */
export function composeCaption(ix: Index, key: string, desk: Desk): ToolStep | { says: Msg; focus?: boolean } {
  const p = str(valueOf(ix, key, desk)).trim();
  if (!p) return { says: msg('compose.empty'), focus: true };
  if (STRUCT.test(p)) return { says: msg('compose.already') };
  const bits = [p.split(/[,.\n]/)[0]!.trim()], bpm = here(ix, KEYS.bpm, desk) && !isAuto(ix, KEYS.bpm, desk), k = here(ix, KEYS.key, desk) && valueOf(ix, KEYS.key, desk) !== '';
  if (bpm) bits.push(`${Number(valueOf(ix, KEYS.bpm, desk))} BPM`);
  if (k) bits.push(say(textMsg(ix, KEYS.key, desk)));
  const g = str(read(ix, KEYS.gender, desk)), lang = has(ix, KEYS.lang, desk.engine) && applies(ix, KEYS.lang, desk) && changed(ix, KEYS.lang, desk) ? optionsOf(ix, KEYS.lang, desk).find(o => o.value === valueOf(ix, KEYS.lang, desk))?.label ?? str(valueOf(ix, KEYS.lang, desk)) : undefined;
  const voice = (g === 'duet' ? 'male and female duet' : g ? `${g} lead vocal` : 'lead vocal') + (lang ? `, sung in ${say(lang)}` : '');
  const text = `Global Metadata: ${bits.join(', ')}, warm and intimate\nVocal Details: ${read(ix, KEYS.instr, desk) === true ? 'instrumental; the lead instrument carries the melody' : voice + ', close-mic, light harmonies on the chorus'}\nArrangement: [Intro] solo instrument, [Verse] sparse, [Chorus] full band, [Outro] fades back to the intro motif`;
  const change = named(ix, desk, key, 'compose', { [key]: text });
  return change ? { change, undo: true, said: msg('compose.said', { with: bpm && k ? 'both' : bpm ? 'bpm' : k ? 'key' : 'none' }) } : { says: msg('compose.already') };
}

/** Expand wildcards in the box or the lyrics: one option of each; with none, it says so. */
export function expandStep(ix: Index, key: string, desk: Desk, where: 'prompt' | 'lyrics', rnd: () => number): ToolStep | { says: Msg } {
  const k = where === 'lyrics' ? KEYS.lyrics : key, src = str(read(ix, k, desk)), out = expandWild(src, rnd);
  if (out === src) return { says: msg('wild.none', { where, box: boxWord(ix, desk) }) };
  const change = named(ix, desk, key, 'expand', { [k]: out });
  return change ? { change, undo: true, said: msg('wild.done') } : { says: msg('wild.none', { where, box: boxWord(ix, desk) }) };
}

/** Move to lyrics: from the first [ on, under any lyrics already there; the box keeps what came before it. */
export function moveToLyrics(ix: Index, key: string, desk: Desk): ToolStep | null {
  const p = str(valueOf(ix, key, desk)), i = p.search(/\[/); if (i < 0) return null;
  const moved = p.slice(i).trim(), had = str(read(ix, KEYS.lyrics, desk)).trim();
  const change = named(ix, desk, key, 'toLyrics', { [KEYS.lyrics]: had ? had + '\n\n' + moved : moved, [key]: p.slice(0, i).replace(/[\s,;:]+$/, '') });
  return change ? { change, undo: true, said: msg(had ? 'sg.lyrics.below' : 'sg.lyrics.moved') } : null;
}

/** Set the tempo field to a BPM the box names: the field on, at that BPM. `brings`: the module the field is on, when it
 *  is not on the desk (it comes with the step). */
export function tempoStep(ix: Index, key: string, desk: Desk): { vals: Change['vals']; auto: Change['auto']; words: Change['words']; brings: string | null; said: Msg } | null {
  const bpm = promptModel(ix, key, desk).suggestions.find(s => s.kind === 'bpm')?.bpm;
  const h = own(ix, KEYS.bpm); if (bpm === undefined || !h || !has(ix, KEYS.bpm, desk.engine)) return null;
  const brings = onDesk(h.d, desk) ? null : h.d.id, at = brings ? { ...desk, on: new Set([...desk.on, brings]) } : desk;
  const v = setValue(ix, at, KEYS.bpm, bpm).value;
  const after = applyChange(at, { vals: { [KEYS.bpm]: v }, auto: { [KEYS.bpm]: false }, state: {}, words: { label: msg(''), put: msg('') } });
  const words = stepWords(ix, desk, after, { named: { key, step: 'tempo', slots: { bpm } } }) ?? { label: msg('step.change', { what: '' }), put: msg('') };
  return { vals: { [KEYS.bpm]: v }, auto: { [KEYS.bpm]: false }, words: asDesk(words), brings, said: msg('sg.bpm.said', { bpm, added: String(!!brings) }) };
}

/** A Structured Caption on another engine, as plain words. */
export function plainStep(ix: Index, key: string, desk: Desk): ToolStep | null {
  const p = str(valueOf(ix, key, desk)), change = named(ix, desk, key, 'plain', { [key]: plainCaption(p) }, {}, undefined);
  return change ? { change, undo: true, said: msg('sg.plain.said', { box: boxWord(ix, desk), pl: String(boxPl(desk)) }) } : null;
}

/** Edit a copy (E-06): the supplied caption becomes the box's text, and the source Custom, in one step. */
export function copyStep(ix: Index, key: string, desk: Desk): ToolStep | null {
  const tr = trackNow(ix, desk); if (!tr) return null;
  const change = named(ix, desk, key, 'copy', { [key]: trackText(desk, tr.track), [KEYS.src]: 'custom' });
  return change ? { change, undo: true, said: msg('prompt.copied', { box: boxWord(ix, desk) }) } : null;
}

/** A dataset tag pressed: in or out of the box's items; on a supplied caption it edits a copy first, in the same step. */
export function tagStep(ix: Index, key: string, desk: Desk, tag: string): ToolStep | null {
  const tr = trackNow(ix, desk), text = tr ? trackText(desk, tr.track) : str(valueOf(ix, key, desk)), out = hasItem(text, tag);
  const change = named(ix, desk, key, out ? 'removeTag' : 'addTag', { [key]: toggleItem(text, tag), ...(tr ? { [KEYS.src]: 'custom' } : {}) }, {}, { tag });
  return change ? { change, undo: true, said: tr ? msg('prompt.tagCopy', { box: boxWord(ix, desk), tag, out: String(out) }) : null } : null;
}

/** A dataset chosen: a dataset that does not hold the chosen track leaves the source on Custom for good, in the same
 *  step, and says so (E-08, E-22). */
export function datasetStep(ix: Index, desk: Desk, ds: string): ToolStep | null {
  if (!has(ix, KEYS.ds, desk.engine)) return null;
  const src = str(valueOf(ix, KEYS.src, desk)), set = datasetOf(desk, ds);
  const drops = !!src && src !== 'custom' && src !== 'auto' && !!set && !set.set.tracks.some(t => t.title === src);
  const vals: Change['vals'] = { [KEYS.ds]: setValue(ix, desk, KEYS.ds, ds).value, ...(drops ? { [KEYS.src]: 'custom' } : {}) };
  const after = applyChange(desk, { vals, auto: {}, state: {}, words: { label: msg(''), put: msg('') } });
  const words = stepWords(ix, desk, after); if (!words) return null;
  return { change: { vals, auto: {}, state: {}, words }, undo: false, said: drops ? msg('prompt.dsDrops', { src, ds: set!.name, box: boxWord(ix, desk), label: paramWords(ix, KEYS.src, desk).label }) : null };
}

/** The tracks the caption source lists (the server's list 'caption.sources'): those of the dataset in force. `ds` names
 *  it, so the page asks again only when it changes. */
export function captionSources(ix: Index, desk: Desk): { ds: string; list: { value: string; label: string }[] } | null {
  if (!has(ix, KEYS.ds, desk.engine) || !desk.lists?.['caption.datasets']) return null;
  const ds = datasetOf(desk, str(valueOf(ix, KEYS.ds, desk)));
  return { ds: ds?.name ?? '', list: (ds?.set.tracks ?? []).map(t => ({ value: t.title, label: t.title })) };
}

/** A latent chosen (ACE-Step): its tempo and key, and the lyrics and the box where they are empty; the module that
 *  carries the tempo comes onto the desk with it (C-21). A file that is not a latent is refused. */
export function latentStep(ix: Index, key: string, desk: Desk, name: string): { state: { [k: string]: Json }; vals: Change['vals']; auto: Change['auto']; words: Change['words']; brings: string | null; said: Msg } | { says: Msg } {
  if (!/\.(latent|hslat)$/i.test(name)) return { says: msg('latent.wrong', { name }) };
  const h = own(ix, KEYS.bpm), brings = h && has(ix, KEYS.bpm, desk.engine) && !onDesk(h.d, desk) ? h.d.id : null, at = brings ? { ...desk, on: new Set([...desk.on, brings]) } : desk;
  const vals: Change['vals'] = {};
  if (h && has(ix, KEYS.bpm, desk.engine)) vals[KEYS.bpm] = setValue(ix, at, KEYS.bpm, 96).value;
  if (has(ix, KEYS.key, desk.engine)) vals[KEYS.key] = setValue(ix, at, KEYS.key, 'A minor').value;
  if (has(ix, KEYS.lyrics, desk.engine) && !str(read(ix, KEYS.lyrics, desk)).trim()) vals[KEYS.lyrics] = '[Verse]\nDust on the needle, the night comes slow\n[Chorus]\nPlay it again, the way we know';
  if (!str(valueOf(ix, key, desk)).trim()) vals[key] = 'dusty boom-bap, vinyl crackle, mellow keys';
  const auto = KEYS.bpm in vals ? { [KEYS.bpm]: false } : {}, state = { latent: name };
  const after = applyChange(at, { vals, auto, state, words: { label: msg(''), put: msg('') } });
  const words = stepWords(ix, desk, after, { named: { key: 'latent', step: 'load' } }) ?? { label: msg('step.change', { what: '' }), put: msg('') };
  return { state, vals, auto, words: asDesk(words), brings, said: msg('latent.said', { name, added: String(!!brings) }) };
}
/** The latent let go: back to a fresh render. */
export function latentClear(ix: Index, desk: Desk): ToolStep | null {
  const after = { ...desk, state: { ...desk.state, latent: null } }, words = stepWords(ix, desk, after, { named: { key: 'latent', step: 'clear' } });
  return words ? { change: { vals: {}, auto: {}, state: { latent: null }, words: asDesk(words) }, undo: true, said: msg('latent.cleared') } : null;
}

/** Reuse (a song's menu) puts its text in the box; a caption a dataset supplies is not written over, and says so in
 *  the source's own words (E-31, 35-songs.js). Null when Reuse may write. */
export const reuseRefused = (ix: Index, desk: Desk): Msg | null => (trackNow(ix, desk) ? lockedMsg(ix, desk, 'reuse') : null);

/** Presets & genres (33-prompt.js PRESETS, GENRES): style presets that replace the box, and genres that go in or out of
 *  its items (C-20). The words are content the engine reads, in its words. */
export const STYLE_PRESETS = ['Acoustic guitar, upright bass and brushed drums, warm and intimate', 'Analog synths, gated drums and a pulsing bassline, nocturnal', 'String quartet with felt piano, slow and cinematic', 'Distorted guitars, live drums and gang vocals, high energy', 'Rhodes, muted trumpet and a lazy swing groove, smoky', '808s, sub bass and airy pads, late-night'];
export const GENRES = ['indie folk', 'synthwave', 'lo-fi hip hop', 'post-rock', 'drum and bass', 'city pop', 'shoegaze', 'afrobeats', 'bossa nova', 'metalcore', 'ambient', 'trap', 'gospel', 'UK garage'];
/** A style preset chosen: the box's text replaced, one step; said with its Undo when it replaced words. A supplied
 *  caption is not replaced, and says why. */
export function presetStep(ix: Index, key: string, desk: Desk, preset: string): ToolStep | { says: Msg } | null {
  if (trackNow(ix, desk)) return { says: lockedMsg(ix, desk) };
  const had = !!str(valueOf(ix, key, desk)).trim(), change = named(ix, desk, key, 'preset', { [key]: preset });
  return change ? { change, undo: true, said: had ? msg('prompt.replaced', { box: boxWord(ix, desk) }) : null } : null;
}
/** A genre pressed: in the box's items, or out again (C-20); typing in the box, as the concept's is. */
export function genreChange(ix: Index, key: string, desk: Desk, genre: string): Change | { says: Msg } | null {
  if (trackNow(ix, desk)) return { says: lockedMsg(ix, desk) };
  const text = str(valueOf(ix, key, desk)), next = hasItem(text, genre) ? toggleItem(text, genre) : text.trimEnd().replace(/[,.]$/, '') + (text.trim() ? ', ' : '') + genre;
  const vals = { [key]: setValue(ix, desk, key, next).value }, after = applyChange(desk, { vals, auto: {}, state: {}, words: { label: msg(''), put: msg('') } });
  const words = stepWords(ix, desk, after);
  return words ? { vals, auto: {}, state: {}, words } : null;
}
/** Which genres the box holds now. */
export const genresIn = (ix: Index, key: string, desk: Desk): boolean[] => GENRES.map(g => hasItem(str(valueOf(ix, key, desk)), g));

/** The definition and node of the part that edits a key (for the page, which binds it). */
export function nodeOf(ix: Index, part: string): { d: Definition; node: Node } | null {
  for (const d of ix.defs) for (const n of 'face' in d ? d.face ?? [] : []) if (n.part === part) return { d, node: n };
  return null;
}

/* ---------------------------------------------------------------- the words */

Object.assign(EN, {
  'box.description.a': 'a description',
  'box.caption.a': 'a caption',
  'box.prompt.a': 'a prompt',
  'box.tags.a': 'some tags',
  'lyrics.stat': '{n, plural, one {# line} other {# lines}} · ≈ {time} at {how, select, guess {about {bpm} BPM} box {{bpm} BPM (from your {box})} other {{bpm} BPM}}',
  'lyrics.tabs': 'Insert a section tag',
  'lyrics.tab.tip': 'Insert {tag}',
  'lyrics.placeholder': 'Type lyrics here, with [Verse] and [Chorus] tags.\n{more}',
  'lyrics.instrTip': 'Instrumental is on, so the lyrics are kept but not sung. Turn it off to edit them.',
  'lyrics.instrOn': 'Instrumental is on. Turn it off to edit the lyrics.',
  'lyrics.sec.tip': '{name} · {n, plural, one {# line} other {# lines}}. Click to go there',
  'lyrics.sec.aria': '{name}, {n, plural, one {# line} other {# lines}}: go to it in the lyrics',
  'lyrics.ignored': 'Lyrics are ignored',
  'lyrics.noSections': 'The song\'s structure shows here as you tag sections',
  'lyrics.expand': 'Expand {a|b} now',
  'lyrics.none': '{engine} has no lyrics. Describe everything, vocals included, in the {box}.{kept, select, true { Your lyrics are kept for engines that sing them.} other {}}',
  'wild.tip': 'Replace each {a|b|c} with one of its choices',
  'wild.none': 'No {a|b|c} wildcards in the {where, select, lyrics {lyrics} other {{box}}}',
  'wild.done': 'Wildcards expanded',
  'sg.lyrics': 'This looks like lyrics. ',
  'sg.lyrics.b': 'Move them to the lyrics sheet',
  'sg.lyrics.moved': 'Moved into the lyrics sheet',
  'sg.lyrics.below': 'Added below the lyrics you already had',
  'sg.bpm': 'Set the tempo field to ',
  'sg.bpm.b': '{bpm} BPM',
  'sg.bpm.said': 'Tempo set to {bpm} BPM{added, select, true {; the Tempo module is on the desk} other {}}',
  'sg.wild': '{n, plural, one {Wildcard} other {Wildcards}}: ',
  'sg.wild.never': '{engine} doesn\'t expand them at Generate.',
  'sg.wild.now': 'Expand them now',
  'sg.wild.each': 'Each song picks one option.',
  'sg.wild.asWritten': 'Sent as written unless you expand them, or turn on Expand on Generate.',
  'sg.plain': 'This is MiniMax-Music3\'s Structured Caption. {engine} reads plain words. ',
  'sg.plain.b': 'Use it as {pl, select, true {plain} other {a plain}} {box}',
  'sg.plain.said': 'Kept the style and vocals as {pl, select, true {plain} other {a plain}} {box}',
  'prompt.ph.words.0': 'dreamy indie folk, warm acoustic guitar, soft female vocals',
  'prompt.ph.words.1': 'late-night synthwave, gated drums, moody male vocals',
  'prompt.ph.words.2': 'lo-fi hip hop with tape hiss and a lazy Rhodes',
  'prompt.ph.words.3': 'big post-rock build, crashing cymbals, no vocals',
  'prompt.ph.tags.0': 'indie folk, acoustic guitar, warm, female vocals',
  'prompt.ph.tags.1': 'synthwave, gated drums, 80s, male vocals',
  'prompt.ph.tags.2': 'lo-fi hip hop, rhodes, tape, chill',
  'prompt.ph.tags.3': 'post-rock, crescendo, cymbals, instrumental',
  'prompt.locked': 'The {box} comes from the dataset ({label}: {src}). Edit a copy {what, select, write {to have one written} reuse {to reuse this one} other {to change it}}.',
  'prompt.lockLine': 'From {ds} · {track}',
  'prompt.srcTip': '{label}: {src}{track}',
  'prompt.srcTip.auto': ', {track} · {bpm} BPM',
  'prompt.tagsTip': 'Click a tag to add it to the {box} or take it out. On a supplied {box} that edits a copy.',
  'prompt.replaced': 'Replaced the {box}',
  'presets.title': 'Style presets',
  'presets.genres': 'Add a genre',
  'prompt.copied': 'The {box} is yours to edit; the dataset\'s is behind Undo',
  'prompt.tagCopy': 'Editing a copy of the dataset\'s {box}, with "{tag}" {out, select, true {taken out} other {added}}',
  'prompt.dsDrops': '{src} isn\'t in {ds}, so the {box} is yours again ({label}: Custom)',
  'write.plain': 'plain words for the caption',
  'write.said': '{kind, select, both {Wrote {what} and lyrics} box {Wrote {what}} new {Wrote new lyrics} other {Wrote lyrics}} (simulated)',
  'write.ask': 'Both are filled in. Replace the lyrics with new ones?',
  'write.replace': 'Replace lyrics',
  'write.full': 'Your {box} {pl, select, true {stay as they are} other {stays as it is}}: Write it for me only fills an empty {pl, select, true {box} other {one}}{why, select, none {, and {engine} has no lyrics} other {, and Instrumental is on}}.',
  'compose.empty': 'Write a few words of style first. Compose caption builds on them.',
  'compose.already': 'This is already a Structured Caption',
  'compose.said': 'Compose caption built a Structured Caption{with, select, both { with your BPM and key} bpm { with your BPM} key { with your key} other {}}',
  'latent.wrong': '{name} isn\'t a .latent or .hslat file',
  'latent.said': 'Continuing from {name}; tempo and key set from it{added, select, true { on the Tempo module} other {}}',
  'latent.cleared': 'Back to a fresh render',
});
