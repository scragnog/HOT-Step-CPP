// The existing Create page's bar, run in Node: its panels (GlobalParamBar and what it opens for each engine) are
// rendered as functions over the real global params store (react.mjs, hooks.mjs), so every control the page draws is
// found with what it shows (its label, its range, its options) and the handler the page calls when it is used. The
// request comparison drives the page this way: to set a value it finds the control the user would use, by its label,
// and calls that control's own onChange, as a click or a drag does; the store's setters, the page's own handlers and
// getGlobalParams() then do what they do. Nothing here knows a store field from a definition's wire.
//
// What the page draws depends on its state (a schedule's shape sliders appear once the schedule is Beta), so the bar
// is drawn again after every change.

import fs from 'node:fs';
import path from 'node:path';
import type { EngineId } from '../../src/definitions/engines.ts';
import type { Json } from '../../src/definitions/types.ts';
import { APP, FIX } from './real.ts';

/* ---------------------------------------------------------------- the stand-ins the bar is drawn with */

const EN = JSON.parse(fs.readFileSync(path.join(APP, 'ui/src/i18n/locales/en.json'), 'utf8')).translation as Record<string, string>;
const t = (k: string, o?: string | { defaultValue?: string; [k: string]: unknown }) => {
  if (typeof o === 'string') o = { defaultValue: o };
  let s = EN[k] ?? o?.defaultValue ?? k;
  for (const [n, v] of Object.entries(o || {})) s = s.replace(new RegExp(`\\{\\{\\s*${n}\\s*\\}\\}`, 'g'), String(v));
  return s;
};
let engine: EngineId = 'ace';
const plugin = (name: string) => ({ name, description: '', params: [], kind: 'plugin' });
const REGISTRY = { solvers: (FIX.plugins.solver as string[]).map(plugin), schedulers: (FIX.plugins.scheduler as string[]).map(plugin), guidance: (FIX.plugins.guidance as string[]).map(plugin), postprocess: (FIX.plugins.postprocess as string[]).map(plugin) };
/* a list the server fills at run time reads as empty here (make-fixtures.ts writes it "<server list>") */
const listed = (x: unknown): unknown => (Array.isArray(x) ? x.map(listed) : x && typeof x === 'object' ? Object.fromEntries(Object.entries(x).map(([k, v]) => [k, k === 'options' && typeof v === 'string' ? [] : listed(v)])) : x);
const capabilities = () => ({ ...(FIX.core[engine] ?? {}), id: engine, extensions: listed(FIX.extensions[engine] ?? []), core: { batch: { max: FIX.core[engine]?.batchMax ?? 1 }, duration: { max: FIX.core[engine]?.durationMax ?? 600 } }, features: FIX.features[engine] ?? {}, backend: engine });

/** What any other import of a stand-in gives: something that can be called and read without end (a query's
 *  options, a client's methods), never a value a component would draw. */
const DEEP: unknown = new Proxy(function deep() {}, {
  get: (_, k) => (k === Symbol.toPrimitive ? () => '' : k === 'then' || typeof k === 'symbol' ? undefined : DEEP),
  apply: () => DEEP,
});
/** A component the bar draws from a file that does not run here: an element of that name, whose props are read. */
const marker = (name: string): unknown => new Proxy(Object.assign(function StandIn() { return null; }, { $standIn: name }), {
  /* an upper-case import may be a class (a preset collection): constructing one gives the endless stand-in */
  construct: () => DEEP as object,
  get: (target, k) => (k in target ? (target as unknown as Record<string | symbol, unknown>)[k] : DEEP),
});
/* What the server answers the queries that decide whether a panel's switch can be used: every optional model pack is
   installed, so every switch the page has is there to drive. Any other query has no data. */
const ANSWERS: Record<string, unknown> = {
  ppVaeOptions: { available: true },
  stableStepOptions: { available: true, tokenizer: true },
  registryOptions: { packs: [{ id: 'mm3-latent-refiner', fileIds: ['refiner'] }], files: [{ id: 'refiner', installed: true, outdated: false }] },
};
/* HeartMuLa's model catalogue: the selection, and the selected adapter listed with an MLP and another home model, so
   all of its dials are shown */
ANSWERS.backendModelsOptions = () => {
  const d = ((globalThis as { __realApp?: { backend: { models?: { heartmula?: { defaults?: Record<string, string> } } } } }).__realApp?.backend.models?.heartmula?.defaults) ?? {};
  return { defaults: d, selectable: true, lmAdapters: d.lmAdapter ? [d.lmAdapter] : [], lmAdapterMeta: d.lmAdapter ? { [d.lmAdapter]: { mlp: true, variant: 'other' } } : {} };
};
const QUERIES = Object.fromEntries(Object.keys(ANSWERS).map(n => [n, (...args: unknown[]) => ({ queryKey: [n, ...args] })]));
const DISCLOSURE = /Accordion|Open$|open$|-fold-/;
const React = await import('./react.mjs');
const OVERRIDES: Record<string, Record<string, unknown>> = {
  'queries/global-models/options': QUERIES,
  '@tanstack/react-query': {
    useQuery: (o?: { queryKey?: unknown[] }) => ({ data: typeof ANSWERS[String(o?.queryKey?.[0])] === 'function' ? (ANSWERS[String(o?.queryKey?.[0])] as () => unknown)() : ANSWERS[String(o?.queryKey?.[0])], isLoading: false, isPending: false, isError: false, isSuccess: true, isFetching: false, error: null, refetch() {} }),
    useQueries: () => [],
    useMutation: () => ({ mutate() {}, mutateAsync: async () => undefined, isPending: false, reset() {} }),
    useQueryClient: () => ({ invalidateQueries() {}, setQueryData() {}, getQueryData() {}, removeQueries() {}, cancelQueries() {} }),
  },
  'react-i18next': { useTranslation: () => ({ t, i18n: { language: 'en', changeLanguage() {} } }) },
  'hooks/useCapabilities': { useCapabilities: () => ({ capabilities: capabilities(), loading: false }) },
  'hooks/usePluginRegistry': {
    usePluginRegistry: () => ({ registry: REGISTRY, loading: false, error: null, reload() {},
      findSolver: (n: string) => REGISTRY.solvers.find(x => x.name === n), findScheduler: (n: string) => REGISTRY.schedulers.find(x => x.name === n), findGuidance: (n: string) => REGISTRY.guidance.find(x => x.name === n) }),
  },
  /* a remembered value, kept like any other state; a remembered open-or-closed section (hs-genAccordion-perf, an
     Accordion's persistKey) reads as opened */
  'hooks/usePersistedState': {
    usePersistedState: (k: string, d: unknown) => (React.useState as (i: () => unknown) => [unknown, unknown])(() => {
      const s = (globalThis as { localStorage: { getItem(k: string): string | null } }).localStorage.getItem(k);
      return s !== null ? JSON.parse(s) : d === false && DISCLOSURE.test(k) ? true : d;
    }),
  },
  'context/AuthContext': { useAuth: () => ({ user: null, token: 'token', isAuthenticated: true }) },
  'components/lyric-studio/playlistStore': { usePlaylist: () => ({ items: [] }) },
};
/** What a hook of a stand-in gives: a query result with no data yet and nothing pending, any other field read on. */
const EMPTY: unknown = new Proxy({}, {
  get: (_, k) => (k === Symbol.iterator ? [][Symbol.iterator] : k === 'length' ? 0 : k === 'valueOf' || k === 'toString' || k === Symbol.toPrimitive ? () => ''
    : typeof k === 'symbol' || k === 'then' || k === 'data' ? undefined
    : /^(is|has)[A-Z]|^(loading|error|pending|fetching|status)$/.test(k) ? false : EMPTY),
});

(globalThis as Record<string, unknown>).__uiStub = (key: string) => {
  const own = OVERRIDES[key.replace(/\.(ts|tsx)$/, '')] ?? {};
  return new Proxy({}, {
    get(_, name) {
      if (typeof name !== 'string') return undefined;
      if (name in own) return own[name];
      if (name === 'default' || /^[A-Z]/.test(name)) return marker(name === 'default' ? key.split('/').pop()! : name);
      if (/^use[A-Z]/.test(name)) return () => EMPTY;
      return DEEP;
    },
  });
};

const fileUrl = (p: string) => 'file:///' + path.join(APP, p).split(path.sep).join('/');
const { GlobalParamBar } = await import(fileUrl('ui/src/components/global-bar/GlobalParamBar.tsx'));
const { CreatePanel } = await import(fileUrl('ui/src/components/create/CreatePanel.tsx'));
const { within, forget, flushEffects } = React;
/** What CreatePanel hands App when Generate is pressed: the legacy params and the intent (CreatePanel.tsx handleGenerate). */
let generated: { params: Record<string, unknown>; intent: Record<string, unknown> } | null = null;
const PANEL_PROPS = { onGenerate: (params: Record<string, unknown>, intent: Record<string, unknown>) => { generated = { params, intent }; }, activeJobCount: 0, reuseData: null, paramsRef: { current: null } };

/* ---------------------------------------------------------------- drawing the bar, and its controls */

interface El { $el: true; type: unknown; props: Record<string, unknown> }
const isEl = (x: unknown): x is El => !!x && typeof x === 'object' && (x as El).$el === true;
const nameOf = (type: unknown): string => (typeof type === 'string' ? type : typeof type === 'function' ? ((type as { $standIn?: string }).$standIn ?? (type as { name: string }).name) : 'unknown');

/** A control as the page draws it: where it is (the panels and sections around it), what it says, and its handler. */
export interface Control {
  /** "Panel > Section > Label": what a user reads to find it. */
  path: string;
  label: string;
  part: string;
  props: Record<string, unknown>;
  /** The handler the page calls when the control is used. */
  handler: (v: unknown) => void;
}
const OPENED = new Set(['BarSection', 'Accordion', 'AccordionSection']);
/* components of the bar read as the control they are, not drawn: their props are what the page wires up */
const READ = new Set(['ToggleSwitch', 'SeedInput']);
/* of those, the ones named by their part, not by the words before them (the seed's field sits after "Random") */
const BY_PART = new Set(['SeedInput']);
const HANDLERS = ['onChange', 'onValueChange', 'onToggle', 'onSelect', 'onCheckedChange'];
const words = (x: unknown): string => (typeof x === 'string' ? x : typeof x === 'number' ? String(x) : Array.isArray(x) ? x.map(words).join('') : isEl(x) ? words(x.props.children) : '');

/** The buttons drawn in the last render, by their words. */
let buttons: Control[] = [];

/** Draws the bar for the engine in use and returns every control it shows. */
export function controls(e: EngineId): { controls: Control[]; errors: string[] } {
  engine = e;
  const out: Control[] = [], errors: string[] = [];
  buttons = [];
  /* the last words read before a control, in the page's order: a control with no label of its own is named by them
     (Shift's slider stands under the ParamLabel "Shift") */
  let lastWords = '';
  /* the same words twice in one place (the stack's Strength per adapter) are told apart by their order: " #2" */
  const seenPaths = new Map<string, number>();
  const unique = (p: string) => { const n = (seenPaths.get(p) ?? 0) + 1; seenPaths.set(p, n); return n > 1 ? `${p} #${n}` : p; };
  /* a control with no label of its own beside a label in the same row is named by it: the label before it, else the
     one after it (a switch, then its ParamLabel) */
  const hints = new WeakMap<object, string>(), after = new WeakSet<object>(), headed = new WeakSet<Control>();
  const wordsAt = (x: unknown): string => {
    if (!isEl(x)) return '';
    if (nameOf(x.type) === 'ParamLabel' && typeof x.props.label === 'string') return x.props.label;
    return /^[a-z0-9]+$/.test(nameOf(x.type)) && nameOf(x.type) !== 'button' && typeof x.props.children === 'string' && x.props.children.trim().length <= 48 ? x.props.children.trim() : '';
  };
  const unnamed = (x: unknown) => isEl(x) && HANDLERS.some(k => typeof x.props[k] === 'function') && !(typeof x.props.label === 'string' && x.props.label);
  const nameSiblings = (list: unknown[]) => {
    const flatList = list.flat(4).filter(x => x != null && typeof x !== 'boolean');
    flatList.forEach((x, i) => {
      if (!unnamed(x)) return;
      const w = wordsAt(flatList[i - 1]) || wordsAt(flatList[i + 1]);
      if (w) hints.set(x as object, w);
      if (w && !wordsAt(flatList[i - 1])) after.add(x as object);
    });
  };
  const walk = (node: unknown, where: string[], at = ''): void => {
    if (node == null || typeof node === 'boolean' || typeof node === 'string' || typeof node === 'number') return;
    if (Array.isArray(node)) {
      nameSiblings(node);
      /* a block that opens with a row holding only its switch (the Denoiser's, LSS's) names what follows it in the
         block: "Generation > Denoiser > Strength" */
      let group = '';
      node.forEach((c, i) => {
        const start = out.length;
        walk(c, group ? [...where, group] : where, `${at}.${i}`);
        const added = out.slice(start);
        if (!group && i === 0 && added.length === 1 && isSwitch(added[0]!) && headed.has(added[0]!)) group = added[0]!.label;
      });
      return;
    }
    if (!isEl(node)) return;
    const type = node.type, props = node.props;
    /* a section that opens and closes is read as open: its contents are what the page shows once it is opened */
    if (typeof type === 'function' && !(type as { $standIn?: string }).$standIn && !OPENED.has(nameOf(type)) && !READ.has(nameOf(type))) {
      let r: unknown;
      const key = `${at}/${nameOf(type)}${(node as { key?: unknown }).key ?? ''}`;
      try { r = within(key, () => (type as (p: unknown) => unknown)(props)); } catch (err) { errors.push(`${nameOf(type)}: ${(err as Error).message} ${((err as Error).stack ?? '').split(String.fromCharCode(10)).slice(1, 3).join(' ')}`); return; }
      walk(r, where, key);
      return;
    }
    const part = nameOf(type);
    const label = typeof props.label === 'string' ? props.label : typeof props.title === 'string' && part !== 'button' ? props.title : '';
    const h = HANDLERS.find(k => typeof props[k] === 'function');
    if (part === 'ParamLabel' && label) { lastWords = label; return; }
    /* a short line of words (a caps label beside a switch) names the control after it */
    const text = /^[a-z0-9]+$/.test(part) && part !== 'button' && !h && (typeof props.children === 'string' || typeof props.children === 'number') ? String(props.children).trim() : '';
    if (text && text.length <= 48) { lastWords = text; return; }
    /* a button with no words of its own (a switch drawn as a button) is named by the words before it */
    if (part === 'button' && typeof props.onClick === 'function') { const w = words(props.children).trim() || lastWords; buttons.push({ path: unique([...where, w].join(' > ')), label: w, part, props, handler: props.onClick as (v: unknown) => void }); }
    if (h) {
      const own = label || (BY_PART.has(part) ? '' : hints.get(node) || lastWords) || (typeof props['aria-label'] === 'string' ? props['aria-label'] as string : '') || (typeof props.placeholder === 'string' ? props.placeholder as string : '');
      const name = (own || `@${part}`).split(' — ')[0]!.trim();
      const c = { path: unique([...where, name].join(' > ')), label: name, part, props, handler: props[h] as (v: unknown) => void };
      out.push(c); if (!label && after.has(node)) headed.add(c);
      lastWords = '';
    }
    /* a section's own switch (an Accordion's toggle) */
    const toggle = props.toggle as { checked?: unknown; onChange?: unknown } | undefined;
    if (OPENED.has(part) && label && toggle && typeof toggle.onChange === 'function') out.push({ path: [...where, label].join(' > '), label, part: `${part} switch`, props: { ...props, checked: toggle.checked }, handler: toggle.onChange as (v: unknown) => void });
    const section = OPENED.has(part) && label ? [...where, label] : where;
    /* a section's switch in its header is named by the section */
    for (const [k, v] of Object.entries(props)) if (k !== 'children' && (isEl(v) || Array.isArray(v))) { if (k === 'headerToggle' && label) lastWords = label; walk(v, section); }
    walk(props.children, section);
  };
  walk({ $el: true, type: GlobalParamBar, props: {} }, []);
  lastWords = '';
  walk({ $el: true, type: CreatePanel, props: PANEL_PROPS }, ['Create']);
  flushEffects();
  return { controls: out, errors };
}

/** How a control's handler is called with a value: an <input>'s with an event, a component's with the value. */
export function use(c: Control, v: unknown): void {
  if (/^(input|textarea|select)$/.test(c.part)) {
    /* typed into the field, then the field left: a field that commits on blur (the refiner's window) gets both */
    const target = { value: typeof v === 'boolean' ? String(v) : String(v ?? ''), valueAsNumber: Number(v), checked: v };
    const event = { target, currentTarget: target, key: 'Enter', preventDefault() {}, stopPropagation() {} };
    c.handler(event);
    if (typeof c.props.onBlur === 'function') (c.props.onBlur as (e: unknown) => void)(event);
  } else c.handler(v);
}
const isSwitch = (c: Control) => /switch|Toggle/i.test(c.part) || typeof c.props.checked === 'boolean';

/** Every control the bar can show on an engine: drawn, then drawn again with each switch it shows turned on, until
 *  nothing new appears (a panel's settings show once its switch is on). The store is put back afterwards. */
export function explore(e: EngineId, reset: () => void): { controls: Control[]; errors: string[] } {
  const seen = new Map<string, Control>(), errors = new Set<string>(), turned = new Set<string>();
  fresh(reset);
  try {
    for (let round = 0; round < 6; round++) {
      const r = controls(e); r.errors.forEach(x => errors.add(x));
      let more = false;
      for (const c of r.controls) {
        if (!seen.has(c.path)) { seen.set(c.path, c); more = true; }
        if (isSwitch(c) && !turned.has(c.path)) { turned.add(c.path); try { use(c, true); more = true; } catch (err) { errors.add(`${c.path}: ${(err as Error).message}`); } }
      }
      if (!more) break;
    }
    /* then each option of each choice, one at a time, for what a choice reveals (Beta's two sliders, DRY's settings) */
    for (const c of [...seen.values()].filter(x => Array.isArray(x.props.options))) {
      for (const o of c.props.options as { value?: unknown }[]) {
        const now = controls(e).controls.find(x => x.path === c.path);
        if (!now) break;
        try { use(now, o?.value ?? o); } catch { continue; }
        for (const x of controls(e).controls) if (!seen.has(x.path)) seen.set(x.path, x);
      }
      const back = controls(e).controls.find(x => x.path === c.path);
      if (back) try { use(back, c.props.value); } catch { /* left as it is */ }
    }
  } finally { fresh(reset); }
  return { controls: [...seen.values()], errors: [...errors] };
}

/** The value that means "press this button" in drive(). */
export const PRESS = '@press';
/** The control at a path: exactly, or "<place> > *<Part>" for the first control of that part there (a picker whose
 *  words are its value). */
const find = (list: Control[], p: string) => {
  const star = / > \*(\w+)$/.exec(p);
  return star ? list.find(x => x.part === star[1] && x.path.startsWith(p.slice(0, star.index) + ' > ')) : list.find(x => x.path === p);
};
/** Uses the controls at these paths with these values, as a user would, in order, the bar drawn again before each so
 *  that a control a switch reveals is there; a button is pressed (PRESS). A control the page does not show in that
 *  state is left, and named. */
export function drive(e: EngineId, sets: [path: string, value: unknown, when?: 'last'][]): { missing: string[] } {
  const missing: string[] = [];
  /* what is marked `last` (a switch put where the desk has it, after everything behind it is set) goes after the
     rest, the one drawn last on the page first, so a switch never hides a control before it is used */
  const first = sets.filter(s => s[2] !== 'last'), last = sets.filter(s => s[2] === 'last');
  const early = run(e, first);
  const order = controls(e).controls.map(c => c.path);
  last.sort((a, b) => order.indexOf(b[0]) - order.indexOf(a[0]));
  missing.push(...run(e, last));
  /* what a switch hid until it was put back (lyrics behind Instrumental) is tried once more */
  missing.push(...run(e, first.filter(s => early.includes(s[0]))));
  return { missing };
}
function run(e: EngineId, sets: [path: string, value: unknown, when?: 'last'][]): string[] {
  const missing: string[] = [];
  let left = sets.map(s => [s[0], s[1]] as [string, unknown]);
  for (let pass = 0; pass < 3 && left.length; pass++) {
    const next: typeof left = [];
    for (const [p, v] of left) {
      const drawn = controls(e).controls;
      if (v === PRESS) { const b = find(buttons, p); if (b) b.handler({ preventDefault() {}, stopPropagation() {} }); else next.push([p, v]); continue; }
      /* a control the page shows disabled cannot be used (the seed's field while Random is on) */
      const c = find(drawn, p);
      if (c && !c.props.disabled) use(c, v); else next.push([p, v]);
    }
    left = next;
  }
  for (const [p] of left) missing.push(p);
  return missing;
}

/** A fresh page on an engine: the store as a fresh install has it (`reset`), and no component state. */
export function fresh(reset: () => void): void { reset(); forget(); generated = null; }

/** Presses Generate on CreatePanel and returns what it hands App (CreatePanel.tsx handleGenerate), or why not. */
export const lastButtons = () => buttons;
export function pressGenerate(e: EngineId): { params: Record<string, unknown>; intent: Record<string, unknown> } | string {
  generated = null;
  controls(e);
  const go = buttons.find(b => b.path.startsWith('Create >') && /^Generate( |$)/.test(b.label) && !/with AI/.test(b.label));
  if (!go) return 'no Generate button';
  if ((go.props as { disabled?: unknown }).disabled) return 'Generate is disabled';
  go.handler({ preventDefault() {}, stopPropagation() {} });
  return generated ?? 'Generate did not hand anything over';
}

export const wordsOf = words;
export type { Json };
