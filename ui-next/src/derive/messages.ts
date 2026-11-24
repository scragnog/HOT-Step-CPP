// What the core says, as messages: an id and its slots. The page looks an id up in the locale's table and fills the
// slots; English is the table below, and the application's other locales (ui/src/i18n/locales) are tables of the same
// ids. A definition's own words are messages too: their id is the translation key stringsOf() gives them, and `en` is
// the English the definition holds, which is also their template ("{value:samp.guide} has no settings of its own").
// So nothing the core returns is a finished sentence, and no grammar is decided outside a table: plurals, lists and
// the case of a word are written in the template (docs/05, "Strings and translation").

/** A slot's value: text, a number, another message, or a list joined as the locale joins one. */
export type Slot = string | number | Msg | SlotList;
export interface SlotList { list: Slot[]; join: 'and' | 'or' | 'comma' | 'dot' | 'semicolon' | 'sentences' }
export interface Msg {
  id: string;
  /** The English template, for a definition's own words (core messages have theirs in EN). */
  en?: string;
  slots?: { [name: string]: Slot };
}

export const msg = (id: string, slots?: Msg['slots']): Msg => (slots ? { id, slots } : { id });
export const list = (items: Slot[], join: SlotList['join'] = 'and'): SlotList => ({ list: items, join });

/** The core's English. Templates: {name} a slot, {name|lower} with its first letter lower case (a label inside a
 *  sentence), {n, plural, one {...} other {...}} with # for the number, {x, select, value {...} other {...}}. */
export const EN: { [id: string]: string } = {
  'value.on': 'on',
  'value.off': 'off',
  'value.about': '≈ {v}',
  'name.owned': '{owner} {label|lower}',
  'name.the': 'the {name|lower}',
  'step.set': '{name} {from} → {to}',
  'step.set.undone': '{name} back to {from}',
  'step.set.redone': '{name} {to}',
  'step.set.put': 'Puts {name} back to {from}',
  'step.row': '{row}: {from} → {to}',
  'step.row.undone': '{row} back to {from}',
  'step.row.redone': '{row}: {to}',
  'step.row.put': 'Puts {row} back to {from}',
  'step.edit': 'Edit {name}',
  'step.edit.put': 'Puts back {name} as {many, select, true {they were} other {it was}}',
  'step.seed': 'Seed {from} → {to}',
  'step.seed.undone': 'Seed back to {from}',
  'step.seed.redone': 'Seed {to}',
  'step.seed.put': 'Puts the seed back to {from}',
  'step.pair': '{a}, {b}',
  'step.pair.undone': '{names} put back',
  'step.reset': 'Reset {names}',
  'step.put': 'Puts back {what}',
  'step.put.desk': 'Puts back {what}. Anything else you did since stays as it is',
  'path.mods': 'the modules',
  'path.pedals': 'the pedals',
  'path.engine': 'the engine',
  'step.change': 'Change {what}',
  'step.settings': '{n} settings',
  'switch.toast': 'On {engine}.',
  'switch.toast.said': 'On {engine}. {said}.',
  'switch.capped': '{capped}; your {waits}',
  'switch.waits': '{values} {many, select, true {wait} other {waits}} on {engine}',
  'switch.keeps': '{engine} keeps {parts}',
  'switch.models': 'its {n, plural, one {model} other {models}} ({names})',
  'switch.adapters': '{n, plural, one {# adapter} other {# adapters}}',
  'switch.settings': '{n, plural, one {# setting} other {# settings}}',
  'switch.back': "{engine}'s own settings are back",
  'switch.removed': '{name} was removed: it needs {engines}',
  'switch.new': '{toast} New here: {names}, in Modules.',
  'switch.new.more': '{first} and {n} more',
  /* the engine panel and the VRAM meter (vram.ts, package 5) */
  'engine.rate': '{khz} kHz output',
  'engine.rate.short': '{khz}k',
  'engine.takes': '×{n}',
  'engine.takes.tip': '{n, plural, one {One song per Generate} other {Up to # songs per Generate}}',
  'engine.testing': 'testing',
  'models.head': '{engine} models',
  'models.sub': 'Each one shows what it costs in VRAM',
  'models.sub.quick': 'Studio shows every component',
  'models.testing': 'in integration testing',
  'models.sep': ' · ',
  'models.title': '{role} model',
  'models.aria': '{role} model ({label}): {value}',
  'models.vram': 'VRAM on a {card} GB card',
  'vram.title': 'VRAM. Solid: loaded now. Outline: what the next Generate needs, with its peak. Full bar: the {card} GB card. Click for the engine log.',
  'vram.aria': 'VRAM: {loaded} GB loaded, about {need} GB for the next Generate, on a {card} GB card',
  'vram.of': '/ {card} ',
  'vram.adapters': 'Adapters',
  'also.set': 'Also set: {items}.',
  'also.off': '{name} off',
  'also.value': '{name} {value}',
  'also.link': 'Edit in {tier}',
  'tier.quick': 'Quick',
  'tier.studio': 'Studio',
  'tier.lab': 'Lab',
  'set.said': '{name}: {lo} to {hi}. Set to {v}.',
  'load.told': '{n, plural, one {# setting} other {# settings}} could not be used',
  'load.unreadable': "That revision couldn't be read",
  'load.engine': 'This revision is for {engine}; switch to it to load it',
  'plate.with': 'with {groups}',
  'plate.group': '{label}: {values}',
  'plate.switch': '{label} {value}',
  'desk.hidden': 'Hidden at this tier, still applied: {names}',
  'desk.removed': 'Removed {name}{n, plural, =0 {} one { and its # pedal} other { and its # pedals}}',
  'desk.tookOff': 'Took {name} off the board{last, select, true {. That was the last one, so the Pedalboard left the desk too} other {}}',
  'desk.parked': "Parked: {engine} doesn't use {n, plural, one {this. It returns} other {these. Each returns}} on an engine that does.",
  'step.add': 'Add {name}',
  'step.remove': 'Remove {name}',
  'step.move': 'Move {name}',
  'step.engine': 'Switch to {engine}',
  'part.tip': '{tip} {how}',
  'knob.type': '{label}: type a value from {min} to {max}',
  'knob.typeNumber': 'Type a number for {label}',
  'lcd.prefixed': '{prefix}{text}',
  'meter.vu': 'MATCH',
  'pedal.tip': '{a}. {b}',
  'pedal.on': 'Turn on {name}',
  'pedal.off': 'Turn off {name}',
  'unit.on': 'Turn on',
  'unit.off': 'Turn off',
  'tile.unsup': "{engine} doesn't use this.",
  'tile.group.lab': 'Lab · {parent}',
  'parked.unit': 'side unit',
  'parked.pedal': 'pedal',
  'parked.worksOn': 'Works on {engines}',
  'parked.engines': '{n} engines',
  'card.lab': 'Lab tier only',
  'card.on': 'Already on the desk',
  'card.notUsed': 'Not used by {engine}. Works on {engines}.',
  'card.notUsedLab': 'Not used by {engine}. Works on {engines}. A Lab module.',
  'card.engines': '{n} engines',
  'card.label': '{name}. {tip}',
  'qempty': 'Add {first}, a seed or pedals from Modules',
  'row.sets': '{label}: {value}',
  'step.engine.put': 'Puts the engine back',
  'board.count': '{parts}',
  'board.none': 'No pedals in the chain yet',
  'board.pedals': '{n, plural, one {# pedal} other {# pedals}}',
  'board.on': '{n} on',
  'board.idle': '{n} {n, plural, one {needs} other {need}} setting up',
  'board.bypassed': 'chain bypassed: no pedal runs',
  'board.units': '{n, plural, one {# side unit} other {# side units}}, {on} on',
  'board.parked': '{n} parked under the desk',
  /* the shared controls (controls.ts) */
  'box.about': '≈{v}',
  'box.auto': 'AUTO',
  'box.seconds': '≈ {s} s',
  'box.typeNumberOrAuto': 'Type a number for {label}, or leave it empty for auto',
  'ruler.upTo': 'up to {v}',
  'ruler.io': '{n} bars of clean beat',
  'ruler.io.at120': '{n} bars of clean beat, drawn at 120 BPM: the tempo comes from {box}',
  'ruler.io.short': '{n} bars each way needs about {need} at {bpm} BPM',
  'lookback.farSeconds': '{s} s ago',
  'lookback.far': '{n} {unit} ago',
  'lookback.rate': 'the planner writes {n} {unit} a second',
  'lookback.now': 'now',
  'lookback.tip.rate': 'Drag to set how far back {who} checks for repeats. The planner writes {n} {unit} a second. Left reaches further back.',
  'lookback.tip': 'Drag to set how far back {who} checks for repeats. Left reaches further back.',
  'power.off': '{label} · off',
  'power.tip.on': 'Switch it off',
  'power.tip.off': 'Switch it on: its settings stay adjustable while it is off',
  /* a file's own steps (steps.ts fileStep) */
  'step.file.load': 'Load {file} as {noun}',
  'step.file.replace': '{noun}: {from} → {to}',
  'step.file.clear': 'Clear {noun}',
  'step.file.back': 'Put {file} back',
  'step.file.cleared': 'Cleared {file}',
  'step.file.putBack': 'Puts {file} back',
  'step.file.out': 'Took {file} out',
  'step.file.loaded': 'Loaded {file}',
  'step.file.takeOut': 'Takes {file} out',
  'file.notAudio': "{name} isn't an audio file. Use WAV, FLAC, MP3, OGG or M4A.",
  'file.outside': 'Drop the file on one of the drop zones',
  /* the read-outs (readouts.ts) */
  'dice.tip': 'Fills in as the planner tries during a run',
  'flow.compare': "windows per 3:00 · {pct}% {more, select, fewer {fewer} other {more}} than the checkpoint's {ref}",
  'flow.same': "windows per 3:00 · as many as the checkpoint's own hop",
  'flow.seam': '{s} s shared at each seam',
  'flow.seam.listen': '{s} s shared at each seam · listen at the joins',
  'curve.strength': 'plan strength {a} → {b}',
  'curve.cfg': 'plan CFG {v}',
  'curve.cads': 'CADS {v}',
  'plan.caption': 'caption',
  'plan.tempo': 'tempo',
  'plan.key': 'key',
  'plan.form': 'form',
  'plan.score': 'score',
  'plan.lyrics': 'lyrics',
  'plan.length': 'length',
  'plan.none': '(none yet)',
  'plan.cot': 'plan --cot',
  'plan.plain': 'plan',
  'plan.model': 'plan --model 8B',
  'plan.off': 'plan --off',
  'plan.off.note': 'The planner is off. The DiT renders from your caption and fields.',
  'plan.compose': 'compose --cot "{mode}"',
  'plan.tags': 'tags "{cap}"',
  'plan.tags.none': 'tags',
  'plan.bpm.yours': '{v} BPM (yours)',
  'plan.bpm.pick': "{v} BPM (planner's pick)",
  'plan.key.yours': '{v} (yours)',
  'plan.key.pick': "{v} (planner's pick)",
  'plan.prose': 'from the caption prose',
  'plan.form.planner': 'planner decides',
  'plan.form.composer': 'composer decides',
  'plan.score.0': 'melody + chords, ABC',
  'plan.score.1': 'melody line, no chords',
  'plan.score.2': 'no lead sheet (guided, slowest)',
  'plan.instrumental': 'none, instrumental',
  'plan.upTo': 'up to {v}',
  'plan.modelEnds': 'model ends it, up to {v}',
  'lrc.line': '[{t}] {name|lower}',
  'lrc.chorus': 'chorus',
  'field.kept': 'Still using {what}.',
  'field.keptNothing': 'Nothing is applied until it is valid.',
  'field.refused': '{err} Kept “{v}”.',
  'field.refused.empty': '{err} Kept it empty.',
};

const JOIN_EN: { [j in SlotList['join']]: (a: string[]) => string } = {
  and: a => (a.length > 1 ? a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1] : a[0] ?? ''),
  or: a => (a.length > 1 ? a.slice(0, -1).join(', ') + ' or ' + a[a.length - 1] : a[0] ?? ''),
  comma: a => a.join(', '),
  dot: a => a.join(' · '),
  semicolon: a => a.join('; '),
  sentences: a => a.join('. '),
};

/** The text a slot says in a locale: `table` is that locale's messages (EN by default, which is also where a missing
 *  id is looked up), `lang` its tag for plurals and lists. */
export function say(s: Slot, table: { [id: string]: string } = EN, lang = 'en'): string {
  if (typeof s === 'string') return s;
  if (typeof s === 'number') return String(s);
  if ('list' in s) {
    const parts = s.list.map(x => say(x, table, lang));
    if (lang === 'en' || s.join === 'comma' || s.join === 'dot' || s.join === 'semicolon' || s.join === 'sentences') return JOIN_EN[s.join](parts);
    return new Intl.ListFormat(lang, { type: s.join === 'or' ? 'disjunction' : 'conjunction' }).format(parts);
  }
  const template = table[s.id] ?? EN[s.id] ?? s.en ?? s.id;
  return render(template, s.slots ?? {}, table, lang);
}

/** Fills a template: a top-level {...} is a slot, a plural or a select; text outside braces is kept. */
function render(t: string, slots: { [name: string]: Slot }, table: { [id: string]: string }, lang: string): string {
  let out = '';
  for (let i = 0; i < t.length; i++) {
    if (t[i] !== '{') { out += t[i]; continue; }
    let depth = 1, j = i + 1;
    for (; j < t.length && depth; j++) { if (t[j] === '{') depth++; else if (t[j] === '}') depth--; }
    const inner = t.slice(i + 1, j - 1);
    out += placeholder(inner, slots, table, lang);
    i = j - 1;
  }
  return out;
}
function placeholder(inner: string, slots: { [name: string]: Slot }, table: { [id: string]: string }, lang: string): string {
  const m = /^\s*([^,]+?)\s*(?:,\s*(plural|select)\s*,([\s\S]*))?$/.exec(inner);
  if (!m) return `{${inner}}`;
  /* a slot's name may hold "|" ({file:slots.0|Adapter 1}); only "|lower" at its end is a modifier */
  const [, whole, kind, rest] = m, mod = whole!.endsWith('|lower') ? 'lower' : '', name = mod ? whole!.slice(0, -6) : whole;
  if (!(name! in slots)) return `{${inner}}`;
  const v = slots[name!]!;
  if (kind) {
    const branches = new Map<string, string>();
    for (const b of rest!.matchAll(/\s*([^\s{]+)\s*\{((?:[^{}]|\{[^{}]*\})*)\}/g)) branches.set(b[1]!, b[2]!);
    const key = kind === 'plural' ? new Intl.PluralRules(lang).select(Number(say(v, table, lang))) : say(v, table, lang);
    const branch = branches.get(kind === 'plural' && branches.has(`=${say(v, table, lang)}`) ? `=${say(v, table, lang)}` : key) ?? branches.get('other') ?? '';
    return render(branch.replace(/#/g, say(v, table, lang)), slots, table, lang);
  }
  const text = say(v, table, lang);
  return mod === 'lower' ? lowFirst(text) : text;
}
/** The first letter lower case, unless the word is an acronym or a name with capitals inside (DCW, StableStep,
 *  pP-VAE stays as it is): one rule, used wherever a label stands inside a sentence. */
export const lowFirst = (s: string): string => (/^[A-Z][a-z]*(?=[\s(]|$)/.test(s) ? s[0]!.toLowerCase() + s.slice(1) : s);
