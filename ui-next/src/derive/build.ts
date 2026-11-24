// Build by section (YuE2's buildBoard, the concept's 25-module-build.js): what the board draws and what a hand on it
// does. The build is two things (docs/09, package 3c):
//
//   - the user's decisions, the record state its node names (`build`): the session it builds in, the score's seed, the
//     sections planned, the take kept for each section, the takes discarded, the songs it rendered. Every change of it
//     is the record's step of that name (plan, replan, keep, unkeep, discard, relyric, render, reset), so Undo covers it;
//   - the build session the server keeps (`desk.sessions.build`, told by the build service, services/build.ts): its
//     takes for each section as they land, the section being looked at, a plan being made. Never a step: the server
//     owns it, and an arrival is no step.
//
// Takes belong to a named section (its name and its ordinal among sections of that name, "Verse 2"), never to a place
// in the list, so they survive lyric edits (B-05); a take remembers the words it was sung to, and a keep sung to words
// its section no longer has is flagged (E-26). Nothing here has a branch for a module or an engine: the board's node says
// what it reads.
//
//   buildSections(text)                 the sections the lyrics hold, as the build knows them
//   buildModel(ix, node, desk)          InstrumentModel.build: what the board draws
//   the board's hands (partAct): planned, keep, discard, relyric, rendered, reset

import type { Json, Node } from '../definitions/types.ts';
import { type Desk, type Index, formatValue, has, mmss, resolve, stateOf, valueOf } from './core.ts';
import { EN, type Slot, list, msg } from './messages.ts';
import { paramWords } from './strings.ts';
import { nodeShown } from './desk.ts';
import { defineInstrument } from './instruments.ts';

Object.assign(EN, {
  'build.empty.title': 'Build the song one section at a time.',
  'build.empty.text': 'The planner writes a score from your style and lyrics. Then pick a section, make a few takes, listen, keep one, and move on.',
  'build.plan': 'Plan the score',
  'build.planning': 'Planning',
  'build.sections': 'Sections',
  'build.sec.kept': '{label}, kept',
  'build.sec.stale': '{label}, kept take was sung to the old words',
  'build.sec.tip.kept': '{label} · kept',
  'build.sec.tip.stale': '{label} · kept take was sung to the old words',
  'build.drift': 'The lyrics changed since this score was planned.{lost, plural, =0 {} one { # kept take no longer has a section.} other { # kept takes no longer have a section.}}{stale, plural, =0 {} one { # kept take was sung to the old words.} other { # kept takes were sung to the old words.}}{added, plural, =0 {} one { # new section is not built yet.} other { # new sections are not built yet.}}',
  'build.replan': 'Re-plan',
  'build.replan.tip': 'Plans the score again with a new seed (this one is {seed}){takes, plural, =0 {} one { and drops this build\'s # take} other { and drops this build\'s # takes}}',
  'build.goon': 'Keep building',
  'build.mode.new': 'Each take gets its own chords and melody.',
  'build.mode.follow': 'Every take follows the score.',
  'build.make': 'Generate {n, plural, one {# take} other {# takes}}',
  'build.more': 'More takes',
  'build.next': 'Next: {label}',
  'build.take': 'Take {n}',
  'build.take.play': 'Play {take} of {section}',
  'build.take.stop': 'Stop {take} of {section}',
  'build.take.play.tip': 'Play',
  'build.take.stop.tip': 'Stop',
  'build.take.discard': 'Discard {take}',
  'build.take.discard.tip': 'Discard this take',
  'build.take.meta': '{dur}{note, select, none {} other { · {note}}}',
  'build.take.old': '{dur} · old words',
  'build.take.tip': '{dur}{note, select, none {} other { · {note}}}{old, select, true { · sung to the old words} other {}}',
  'build.take.busy': 'rendering…',
  'build.take.busy.tip': 'Rendering',
  'build.keep': 'Keep',
  'build.kept': 'Kept',
  'build.new': 'Start a new build',
  'build.count': '{rendered, select, none {} other {Rendered as “{rendered}” · }}{k} of {n} kept{seed, select, none {} other { · seed {seed}}}',
  'build.render': 'Render song',
  'build.render.part': 'Render {k} of {n} sections',
  'build.render.again': 'Render again',
  'build.playKept': 'Play kept',
  'build.playKept.stop': 'Stop',
  'build.playKept.tip': 'Play every kept take in section order',
  'build.locked.tip': 'Locked by the first keep, so every kept section matches. Re-plan to change it.',
  'build.locked.aria': '{label}: {value}, locked by the first keep. Re-plan to change it.',
  'build.locknote': 'Locked for this build by the first keep',
  'build.planned': 'Score planned',
  'build.replanned': 'Planned a new score. {n, plural, one {# take was} other {# takes were}} dropped{kept, plural, =0 {} other {, # of them kept}}',
  'build.said.kept': 'Kept take {n} for {section}. {next, select, none {Every section is kept} other {Next: {next}}}',
  'build.said.locked': '{a} and {b} are now locked for this build, so every section matches',
  'build.said.unkept': '{section} is not kept any more{free, select, true {. {a} and {b} are free again} other {}}',
  'build.said.discarded': 'Discarded take {n}',
  'build.said.relyric': 'The build follows the new lyrics',
  'build.said.rendered': 'Saved “{title}” to the Library{again, select, true { as a new song} other {}}{old, plural, =0 {} one {; # section is sung to the old words} other {; # sections are sung to the old words}}. The build stays here',
  'build.said.reset': 'Started a new build',
  'build.song.style': 'Built from {n, plural, one {# section} other {# sections}}: {names}',
  'build.song.title': 'Built song',
});

/* ---------------------------------------------------------------- the sections */

/** A section of the lyrics as the build knows it: its name, its id (the name and its ordinal among sections of that
 *  name, "Verse#2"), how the page names it ("Verse 2" when the name repeats), its words and how many lines. */
export interface BuildSection { name: string; id: string; label: string; words: string; n: number }
/** The sections the lyrics hold (25-module-build.js buildSecs): a tag starts one; words before any tag are a Verse. */
export function buildSections(text: string): BuildSection[] {
  const out: { name: string; id: string; lines: string[] }[] = [], seen: { [name: string]: number } = {};
  let cur: (typeof out)[number] | null = null;
  for (const l of String(text).split('\n')) {
    const m = /^\[([^\]]+)\]/.exec(l.trim());
    if (m) { const name = m[1]!.trim(); seen[name] = (seen[name] ?? 0) + 1; cur = { name, id: `${name}#${seen[name]}`, lines: [] }; out.push(cur); }
    else if (l.trim()) {
      if (!cur) { seen.Verse = (seen.Verse ?? 0) + 1; cur = { name: 'Verse', id: `Verse#${seen.Verse}`, lines: [] }; out.push(cur); }
      cur.lines.push(l.trim());
    }
  }
  return out.map(s => ({ name: s.name, id: s.id, words: s.lines.join('\n'), n: s.lines.length, label: (seen[s.name] ?? 0) > 1 ? `${s.name} ${s.id.split('#')[1]}` : s.name }));
}

/** A section's colour (25-module-lyrics.js SECTION_COL and secColor): by its name, without a number. */
const SECTION_COLOUR: { [name: string]: string } = { verse: '#3f5ef0', chorus: '#e0225a', 'pre-chorus': '#9440f5', bridge: '#0d7f62', intro: '#6b7280', outro: '#6b7280', hook: '#a85a00' };
export const sectionColour = (name: string): string => SECTION_COLOUR[name.toLowerCase().replace(/\s*\d+$/, '')] ?? '#8b8496';

/* ---------------------------------------------------------------- the record and the session */

/** The user's decisions about a build (the record state): what Undo covers. `songs` are every song this desk's builds
 *  rendered, kept across a re-plan and a new build, so undoing a render is the one way one leaves the Library. */
export interface BuildRecord {
  session: string | null; seed: number; plan: string[]; kept: { [section: string]: string }; dropped: string[];
  rendered: { title: string; n: number } | null; songs: string[];
}
/** A take, as the build session tells it. */
export interface Take { id: string; ready: boolean; hue: number; dur: number; w: number; words: string; note: string }
/** What the build service tells the page (desk.sessions.build): each session's takes by section and the section
 *  looked at, and whether a plan is being made. */
export interface BuildSessions { planning: boolean; sessions: { [id: string]: { takes: { [section: string]: Take[] }; cur?: string } } }

const str = (x: unknown, n: number) => (typeof x === 'string' ? x.slice(0, n) : '');
const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
/** A record as it was stored or set, cleaned (the concept's deskState('build')): keeps point at takes not discarded. */
export function cleanBuild(raw: Json | undefined): BuildRecord {
  const b = (raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {}) as { [k: string]: unknown };
  const kept: BuildRecord['kept'] = {};
  if (b.kept && typeof b.kept === 'object') for (const [k, v] of Object.entries(b.kept as object)) if (typeof v === 'string') kept[k.slice(0, 80)] = v.slice(0, 40);
  const r = b.rendered as { title?: unknown; n?: unknown } | null | undefined;
  return {
    session: typeof b.session === 'string' && b.session ? b.session.slice(0, 80) : null,
    seed: isNum(b.seed) ? b.seed : 1,
    plan: Array.isArray(b.plan) ? b.plan.filter((x): x is string => typeof x === 'string').slice(0, 64) : [],
    kept, dropped: Array.isArray(b.dropped) ? b.dropped.filter((x): x is string => typeof x === 'string').slice(0, 256) : [],
    rendered: r && typeof r.title === 'string' ? { title: r.title.slice(0, 40), n: isNum(r.n) ? r.n : 1 } : null,
    songs: Array.isArray(b.songs) ? b.songs.filter((x): x is string => typeof x === 'string').slice(-64) : [],
  };
}
/** The sessions as the service told them, cleaned. */
export function cleanSessions(raw: Json | undefined): BuildSessions {
  const o = (raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {}) as { planning?: unknown; sessions?: unknown };
  const sessions: BuildSessions['sessions'] = {};
  for (const [id, s] of Object.entries(o.sessions && typeof o.sessions === 'object' ? o.sessions as object : {})) {
    const x = s as { takes?: unknown; cur?: unknown }, takes: { [k: string]: Take[] } = {};
    for (const [sec, ts] of Object.entries(x?.takes && typeof x.takes === 'object' ? x.takes as object : {})) {
      if (!Array.isArray(ts)) continue;
      takes[sec] = ts.filter(t => t && typeof t.id === 'string').map(t => ({ id: t.id, ready: !!t.ready, hue: isNum(t.hue) ? t.hue : 200, dur: isNum(t.dur) ? t.dur : 20, w: isNum(t.w) ? t.w : 1, words: str(t.words, 4000), note: str(t.note, 60) }));
    }
    sessions[id] = { takes, ...(typeof x?.cur === 'string' ? { cur: x.cur } : {}) };
  }
  return { planning: o.planning === true, sessions };
}

/** Where the board's node keeps its record, and what it reads (by place in `reads`): the lyrics, the planner's
 *  Chain of Thought, takes per section, section mode, Follow score, Plan strength, the prompt and the song's title. */
const READS = ['lyrics', 'cot', 'takes', 'mode', 'cond', 'bar', 'prompt', 'title'] as const;
type BuildRead = typeof READS[number];
const keysOf = (node: Node): { [r in BuildRead]?: string } => Object.fromEntries(READS.map((r, i) => [r, node.reads?.[i]]).filter(([, k]) => !!k));
const recordKey = (ix: Index, node: Node) => node.state?.find(k => ix.states.get(k)?.s.kind === 'record');
const read = (ix: Index, desk: Desk, k: string | undefined) => (k && ix.params.has(k) && has(ix, k, desk.engine) ? valueOf(ix, k, desk) : undefined);

/** The build as the board sees it on this desk: the record, the session's takes (those not discarded), the sections. */
export function buildAt(ix: Index, node: Node, desk: Desk) {
  const rk = recordKey(ix, node), raw = rk ? stateOf(ix, rk, desk) : undefined, rec = raw ? cleanBuild(raw) : null;
  const told = cleanSessions(desk.sessions?.build), k = keysOf(node);
  const secs = buildSections(String(read(ix, desk, k.lyrics) ?? ''));
  const ses = rec?.session ? told.sessions[rec.session] : undefined, gone = new Set(rec?.dropped ?? []);
  const takes: { [section: string]: Take[] } = {};
  for (const [s, ts] of Object.entries(ses?.takes ?? {})) takes[s] = ts.filter(t => !gone.has(t.id));
  /* a keep points at a take that is there */
  const kept: { [section: string]: string } = {};
  for (const [s, id] of Object.entries(rec?.kept ?? {})) if ((takes[s] ?? []).some(t => t.id === id && t.ready)) kept[s] = id;
  const planned = !!rec?.session;
  const cur = planned ? (secs.find(s => s.id === ses?.cur) ?? secs.find(s => kept[s.id] == null) ?? secs[0]) : undefined;
  return { rk, rec, told, secs, takes, kept, planned, cur, k };
}
type Built = ReturnType<typeof buildAt>;
const keptTake = (b: Built, sec: string) => (b.takes[sec] ?? []).find(t => t.id === b.kept[sec]);
/** A keep sung to words its section no longer has (E-26). */
const staleKeep = (b: Built, s: BuildSection) => { const t = keptTake(b, s.id); return !!t && t.words !== s.words; };
/** How the lyrics compare with the sections the score was planned for. */
function driftOf(b: Built) {
  const ids = new Set(b.secs.map(s => s.id)), plan = b.rec?.plan ?? [];
  return {
    changed: plan.length !== b.secs.length || plan.some((id, i) => id !== b.secs[i]?.id),
    lost: Object.keys(b.kept).filter(id => !ids.has(id)).length, added: b.secs.filter(s => !plan.includes(s.id)).length,
  };
}
/** What Re-plan costs: the build's takes that have landed, and its keeps (E-04). */
const costOf = (b: Built) => ({ takes: Object.values(b.takes).flat().filter(t => t.ready).length, kept: Object.keys(b.kept).length });

/* ---------------------------------------------------------------- what the board draws */

export interface BuildModel {
  /** Planned, or not yet (the empty board offering Plan the score); a plan being made. */
  planned: boolean; planning: boolean;
  /** The record's session, for the part's requests to the build service, and the section the session was last told
   *  the page looks at (the board tells it the one it shows when they differ). */
  session: string | null; looked: string | null;
  /** What Plan the score asks for: the sections as they stand. */
  plan: { sections: { id: string; words: string }[] };
  sections: { id: string; label: string; ab: string; colour: string; grow: number; kept: boolean; stale: boolean; cur: boolean; fresh: boolean; aria: Slot; tip: Slot }[];
  /** The banner while the lyrics differ from the plan or a keep is stale, and whether it offers Keep building. */
  drift: { text: Slot; goon: boolean } | null;
  replanTip: Slot;
  /** The section being built: its words, what Generate asks for, its takes. */
  cur: { id: string; label: string; colour: string; says: Slot; make: { label: Slot; primary: boolean; ask: { section: string; words: string; n: number; ownChords: boolean } }; next: { id: string; label: Slot } | null } | null;
  takes: { id: string; name: Slot; keep: boolean; ready: boolean; hue: number; w: number; dur: number; meta: Slot; tip: Slot; play: Slot; stop: Slot; discard: Slot }[];
  /** The keeps in section order, for Play kept: each take's id, its section and length. */
  keptOrder: { take: string; section: string; dur: number }[];
  /** Follow score and Plan strength while a keep locks them: their read-outs, in the order the board's children hold them. */
  /** Whether any of the board's settings (its children) is drawn at this tier. */
  ctl: boolean;
  locked: { at: number; label: Slot; text: string; aria: Slot; tip: Slot }[] | null;
  foot: { count: Slot; render: Slot; canRender: boolean; again: boolean; reset: boolean };
  /** What Render asks the service for: the song's title and style line, its length, and the kept takes in order. */
  render: { title: string; style: Slot; dur: number; takes: string[] } | null;
}

/** The song's title as Generate names one (E-40): the title, else the prompt's first phrase, capitalised. */
function titleOf(ix: Index, desk: Desk, b: Built): string {
  const t = String(read(ix, desk, b.k.title) ?? '').trim() || String(read(ix, desk, b.k.prompt) ?? '').split(/[,.\n]/)[0]!.trim();
  const s = t.slice(0, 40);
  return s ? s[0]!.toUpperCase() + s.slice(1) : '';
}
const fmt = (ix: Index, desk: Desk, k: string) => formatValue(resolve(ix.params.get(k)!.p, desk.engine), valueOf(ix, k, desk)!, desk.lists, desk.engine);

export function buildModel(ix: Index, node: Node, desk: Desk): BuildModel {
  const b = buildAt(ix, node, desk), d = driftOf(b), cost = costOf(b), seed = b.rec?.seed ?? 0;
  const kept = (s: BuildSection) => b.kept[s.id] != null;
  const sections = b.planned ? b.secs.map(s => {
    const st = kept(s) && staleKeep(b, s);
    return {
      id: s.id, label: s.label, ab: s.name.slice(0, 1), colour: sectionColour(s.name), grow: Math.max(1, s.n), kept: kept(s), stale: st, cur: s === b.cur, fresh: !(b.rec?.plan ?? []).includes(s.id),
      aria: st ? msg('build.sec.stale', { label: s.label }) : kept(s) ? msg('build.sec.kept', { label: s.label }) : s.label,
      tip: st ? msg('build.sec.tip.stale', { label: s.label }) : kept(s) ? msg('build.sec.tip.kept', { label: s.label }) : s.label,
    };
  }) : [];
  const nStale = sections.filter(s => s.stale).length;
  const n = Number(read(ix, desk, b.k.takes) ?? 1), mode = String(read(ix, desk, b.k.mode) ?? '');
  const cur = b.cur, takes = cur ? b.takes[cur.id] ?? [] : [], next = cur ? b.secs.find(s => !kept(s) && s !== cur) : undefined;
  const ks = b.secs.filter(kept).length, all = ks === b.secs.length;
  /* the section mode at which each take gets its own chords and melody (the node's ownChords); at any other every
     take follows the score */
  const ownChords = mode !== '' && mode === String((node.opts as { ownChords?: string } | undefined)?.ownChords ?? '');
  const ordered = b.secs.filter(kept).map(s => keptTake(b, s.id)!);
  /* the board's children that hold Follow score and Plan strength, where they are drawn: a keep shows them as read-outs */
  const lockKeys = b.planned && Object.keys(b.kept).length ? (node.children ?? []).map((c, at) => ({ at, k: [c.param, ...(c.children ?? []).map(g => g.param)].find(k => !!k && (k === b.k.cond || k === b.k.bar) && has(ix, k, desk.engine)) })).filter((x): x is { at: number; k: string } => !!x.k && nodeShown(ix, node.children![x.at]!, desk)) : [];
  const title = titleOf(ix, desk, b) || '';
  return {
    planned: b.planned, planning: b.told.planning, session: b.rec?.session ?? null, looked: (b.rec?.session ? b.told.sessions[b.rec.session]?.cur : undefined) ?? null,
    plan: { sections: b.secs.map(s => ({ id: s.id, words: s.words })) },
    sections,
    drift: b.planned && (d.changed || nStale) ? { text: msg('build.drift', { lost: d.lost, stale: nStale, added: d.added }), goon: d.changed } : null,
    replanTip: msg('build.replan.tip', { seed, takes: cost.takes }),
    cur: b.planned && cur ? {
      id: cur.id, label: cur.label, colour: sectionColour(cur.name), says: msg(ownChords ? 'build.mode.new' : 'build.mode.follow'),
      make: { label: takes.length ? msg('build.more') : msg('build.make', { n }), primary: !takes.length, ask: { section: cur.id, words: cur.words, n, ownChords } },
      next: kept(cur) && next ? { id: next.id, label: msg('build.next', { label: next.label }) } : null,
    } : null,
    takes: cur ? takes.map((t, i) => {
      const nm = msg('build.take', { n: i + 1 }), old = t.ready && t.words !== cur.words, dur = mmss(t.dur);
      return {
        id: t.id, name: nm, keep: b.kept[cur.id] === t.id, ready: t.ready, hue: t.hue, w: t.w, dur: t.dur,
        meta: t.ready ? (old ? msg('build.take.old', { dur }) : msg('build.take.meta', { dur, note: t.note || 'none' })) : msg('build.take.busy'),
        tip: t.ready ? msg('build.take.tip', { dur, note: t.note || 'none', old: String(old) }) : msg('build.take.busy.tip'),
        play: msg('build.take.play', { take: nm, section: cur.label }), stop: msg('build.take.stop', { take: nm, section: cur.label }), discard: msg('build.take.discard', { take: nm }),
      };
    }) : [],
    keptOrder: b.secs.filter(kept).map(s => ({ take: b.kept[s.id]!, section: s.id, dur: keptTake(b, s.id)!.dur })),
    ctl: (node.children ?? []).some(c => nodeShown(ix, c, desk)),
    locked: lockKeys.length ? lockKeys.map(({ at, k }) => { const label = paramWords(ix, k, desk).label, text = fmt(ix, desk, k); return { at, label, text, aria: msg('build.locked.aria', { label, value: text }), tip: msg('build.locked.tip') }; }) : null,
    foot: {
      count: msg('build.count', { rendered: b.rec?.rendered?.title ?? 'none', k: ks, n: b.secs.length, seed: desk.tier === 'lab' ? seed : 'none' }),
      render: b.rec?.rendered ? msg('build.render.again') : all || !ks ? msg('build.render') : msg('build.render.part', { k: ks, n: b.secs.length }),
      canRender: ks > 0, again: !!b.rec?.rendered, reset: !!b.rec?.rendered,
    },
    render: ks ? { title: title || 'Built song', style: msg('build.song.style', { n: ks, names: list(b.secs.filter(kept).map(s => s.label), 'dot') }), dur: ordered.reduce((a, t) => a + t.dur, 0), takes: ordered.map(t => t.id) } : null,
  };
}

/* ---------------------------------------------------------------- what a hand on the board does */

/** The two settings a keep locks, by their names (the said words). */
const lockNames = (ix: Index, desk: Desk, b: Built): [Slot, Slot] => [b.k.cond, b.k.bar].map(k => (k && ix.params.has(k) ? paramWords(ix, k, desk).label : '')) as [Slot, Slot];

defineInstrument('buildBoard', (ix, node, desk) => ({ build: buildModel(ix, node, desk) }), {
  /* the score landed (`arg`: the session and its seed, from the build service): the plan is the sections as they stand
     now; a build that was there is replaced whole, said with what it dropped (E-04) */
  planned: x => {
    const b = buildAt(x.ix, x.node, x.desk), a = x.arg as { session?: string; seed?: number }, st = b.rk ? x.v.state[b.rk] : undefined;
    if (!st || typeof a?.session !== 'string') return false;
    const had = b.planned ? costOf(b) : { takes: 0, kept: 0 };
    const next: BuildRecord = { session: a.session, seed: isNum(a.seed) ? a.seed : 1, plan: b.secs.map(s => s.id), kept: {}, dropped: [], rendered: null, songs: b.rec?.songs ?? [] };
    st.apply(b.planned ? 'replan' : 'plan', next as unknown as Json);
    if (had.takes || had.kept) x.tell(msg('build.replanned', { n: had.takes, kept: had.kept }), true); else x.tell(msg('build.planned'), 'say');
  },
  /* Keep (or let go of the keep): `arg` is the take's id in the section being built */
  keep: x => {
    const b = buildAt(x.ix, x.node, x.desk), cur = b.cur, st = b.rk ? x.v.state[b.rk] : undefined, id = String(x.arg);
    if (!b.rec || !cur || !st) return false;
    const n = (b.takes[cur.id] ?? []).findIndex(t => t.id === id) + 1; if (!n) return false;
    const [a, bb] = lockNames(x.ix, x.desk, b);
    if (b.kept[cur.id] === id) {
      const kept = { ...b.kept }; delete kept[cur.id];
      st.apply('unkeep', { ...b.rec, kept } as unknown as Json, undefined, { n, section: cur.label });
      x.tell(msg('build.said.unkept', { section: cur.label, free: String(!Object.keys(kept).length), a, b: bb }), 'say');
      return;
    }
    const first = !Object.keys(b.kept).length, kept = { ...b.kept, [cur.id]: id }, next = b.secs.find(s => kept[s.id] == null);
    st.apply('keep', { ...b.rec, kept } as unknown as Json, undefined, { n, section: cur.label });
    x.tell(msg('build.said.kept', { n, section: cur.label, next: next?.label ?? 'none' }), 'say');
    if (first) x.tell(msg('build.said.locked', { a, b: bb }));
  },
  /* Discard a take of the section being built: it leaves the build, and its keep with it (which frees the locks when it
     was the last: E-24) */
  discard: x => {
    const b = buildAt(x.ix, x.node, x.desk), cur = b.cur, st = b.rk ? x.v.state[b.rk] : undefined, id = String(x.arg);
    if (!b.rec || !cur || !st) return false;
    const n = (b.takes[cur.id] ?? []).findIndex(t => t.id === id) + 1; if (!n) return false;
    const kept = { ...b.kept }; if (kept[cur.id] === id) delete kept[cur.id];
    st.apply('discard', { ...b.rec, kept, dropped: [...b.rec.dropped, id] } as unknown as Json, undefined, { n });
    x.tell(msg('build.said.discarded', { n }), true);
  },
  /* Keep building: the lyrics as they are now become the plan; keeps and takes whose section is gone are let go */
  relyric: x => {
    const b = buildAt(x.ix, x.node, x.desk), st = b.rk ? x.v.state[b.rk] : undefined;
    if (!b.rec || !st) return false;
    const ids = new Set(b.secs.map(s => s.id)), kept = Object.fromEntries(Object.entries(b.kept).filter(([s]) => ids.has(s)));
    const dropped = [...b.rec.dropped, ...Object.entries(b.takes).filter(([s]) => !ids.has(s)).flatMap(([, ts]) => ts.map(t => t.id))];
    st.apply('relyric', { ...b.rec, kept, dropped, plan: b.secs.map(s => s.id) } as unknown as Json);
    x.tell(msg('build.said.relyric'), true);
  },
  /* the song rendered from the keeps (`arg`: its id and title, from the build service): the build stays (B-06) */
  rendered: x => {
    const b = buildAt(x.ix, x.node, x.desk), st = b.rk ? x.v.state[b.rk] : undefined, a = x.arg as { song?: string; title?: string };
    if (!b.rec || !st || typeof a?.song !== 'string') return false;
    const secs = b.secs.filter(s => b.kept[s.id] != null), title = String(a.title ?? '').slice(0, 40), again = !!b.rec.rendered;
    st.apply('render', { ...b.rec, rendered: { title, n: secs.length }, songs: [...b.rec.songs, a.song] } as unknown as Json, undefined, { title });
    x.tell(msg('build.said.rendered', { title, again: String(again), old: secs.filter(s => staleKeep(b, s)).length }), true);
  },
  /* Start a new build: the board is empty again; the songs it rendered stay in the Library */
  reset: x => {
    const b = buildAt(x.ix, x.node, x.desk), st = b.rk ? x.v.state[b.rk] : undefined;
    if (!b.rec || !st) return false;
    st.apply('reset', { ...cleanBuild(undefined), songs: b.rec.songs } as unknown as Json);
    x.tell(msg('build.said.reset'), true);
  },
});
