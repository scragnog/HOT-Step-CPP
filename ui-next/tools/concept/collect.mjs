// Collects what the concept page actually does, as fixtures the derivation tests (tools/test-core.ts) compare the
// core with. It opens the built page (hotstep-tiers.html) in Chromium and runs the page's own functions on desks it
// sets up: the undo step names (vsWords), each tier's "Also set" line as the page paints it (paintAlso), engine switches
// (enterEngine, engineNews), what a revision stores (revSnap) and what cleaning keeps (cleanVals, cleanSlots, the
// seed's cleaner). The cases come from a fixed seed, so a second run writes the same file.
//
//   PLAYWRIGHT_MODULE=<...>/node_modules/playwright node tools/concept/collect.mjs
//
// HOTSTEP_CONCEPT names the concept's folder (default: HOT-Step-UI beside the repository). It writes
// tools/fixtures/concept.json. Run it when the concept changes; nothing else needs a browser.

import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const CONCEPT = process.env.HOTSTEP_CONCEPT ?? path.resolve(ROOT, '..', '..', 'HOT-Step-UI');
if (!process.env.PLAYWRIGHT_MODULE) { console.log('Set PLAYWRIGHT_MODULE to the folder of the playwright package.'); process.exit(1); }
const { chromium } = createRequire(import.meta.url)(process.env.PLAYWRIGHT_MODULE);
/* the two-value controls of the definitions (XY pads), handed to the page so it names their steps */
const PAIRS = [['lm.temp', 'lm.topp'], ['yp.temp', 'yp.topp'], ['apg.mom', 'apg.norm'], ['perf.cfgcut', 'perf.cache'], ['lat.shift', 'lat.rescale'], ['samp.cs', 'samp.ce']];

const browser = await chromium.launch();
const page = await (await browser.newContext({ viewport: { width: 1920, height: 1080 } })).newPage();
const errors = [];
page.on('pageerror', e => errors.push(e.message));
await page.goto('file:///' + path.join(CONCEPT, 'hotstep-tiers.html').replace(/\\/g, '/'));
await page.waitForTimeout(600);

const out = await page.evaluate(PAIRS => {
  let seed = 0x5eed;
  const rnd = () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const pick = a => a[Math.floor(rnd() * a.length)];
  const clone = x => JSON.parse(JSON.stringify(x));
  const ENGS = Object.keys(BK);
  const on = (k, bk) => !C[k].only || C[k].only.includes(bk);
  /* a control's range on an engine, as its knob draws it: its engine's line, else its own min and max */
  const range = (k, bk) => { const c = C[k], r = c.rng && (c.rng[bk] || (c.min === undefined ? Object.values(c.rng)[0] : null)); return r ? [r[0], r[1], r[2] ?? c.step ?? 1] : [c.min, c.max, c.step ?? 1]; };
  /* the values a control is tried at: its ends and a step off its default for a number, the other position of a switch,
     each option of a choice (six at most), a word for text */
  function tries(k, bk) {
    const c = C[k], d = defOf(k);
    if (c.type === 'toggle') return [!d];
    if (typeof d === 'number' || c.type === 'slider' || c.type === 'knob') { const [lo, hi, st] = range(k, bk); const off = +(d + st <= hi ? d + st : d - st).toFixed(4); return [...new Set([lo, hi, off])].filter(v => v !== d); }
    const o = optsOf(k); if (Array.isArray(o)) return o.filter(v => v !== d).slice(0, 6);
    return ['x'];
  }
  const base = bk => { S.backend = bk; S.vals = {}; S.slots = []; S.seed = null; S.mods = []; S.pedals = []; S.eng = { stash: {}, hold: {} }; heldSrc = {}; return snapState(); };
  const res = { steps: [], also: [], switches: [], revisions: [], cleaning: [], controls: {} };
  /* ---- the controls, for mapping the page's values (option names) to the definitions' */
  res.plugins = { solver: PLUG.solver, narSolver: PLUG.narSolver, sched: PLUG.sched, guide: PLUG.guide };
  for (const bk of ENGS) { S.backend = bk; res.controls[bk] = Object.fromEntries(Object.keys(C).filter(k => on(k, bk)).map(k => [k, { type: C[k].type, opts: Array.isArray(optsOf(k)) ? optsOf(k) : null, names: C[k].names || null, def: defOf(k), auto: !!C[k].auto, engOwn: isEngOwn(k), content: CONTENT_KEYS.includes(k) }])); }
  /* ---- undo step names */
  for (const bk of ENGS) {
    const b = base(bk);
    for (const k of Object.keys(C).filter(x => on(x, bk))) {
      for (const v of tries(k, bk)) { const a = clone(b); a.vals[k] = v; if (C[k].auto) a.vals[k + '.auto'] = false; res.steps.push({ engine: bk, before: b.vals, after: a.vals, words: vsWords(b, a) }); }
      if (C[k].auto) { const a = clone(b); a.vals[k + '.auto'] = false; res.steps.push({ engine: bk, before: b.vals, after: a.vals, words: vsWords(b, a) }); }
    }
    for (const [x, y] of PAIRS) {
      if (!C[x] || !C[y] || !on(x, bk) || !on(y, bk)) continue;
      const a = clone(b); a.vals[x] = tries(x, bk)[0]; a.vals[y] = tries(y, bk)[0];
      res.steps.push({ engine: bk, before: b.vals, after: a.vals, words: vsWords(b, a) });
      res.steps.push({ engine: bk, before: a.vals, after: b.vals, words: vsWords(a, b) });
    }
    const keys = Object.keys(C).filter(x => on(x, bk) && C[x].type !== 'text' && C[x].type !== 'textarea');
    for (let i = 0; i < 12; i++) {
      const a = clone(b), n = 2 + Math.floor(rnd() * 3), ks = new Set(); while (ks.size < n) ks.add(pick(keys));
      for (const k of ks) { const t = tries(k, bk); if (t.length) a.vals[k] = pick(t); }
      res.steps.push({ engine: bk, before: b.vals, after: a.vals, words: vsWords(b, a) });
    }
    /* the seed, and the description */
    const s1 = clone(b); s1.seed = 1234; s1.vals['seed.lock'] = true;
    res.steps.push({ engine: bk, before: b.vals, after: s1.vals, seedBefore: null, seedAfter: 1234, words: vsWords(b, s1) });
    const s2 = clone(s1); s2.vals['seed.lock'] = false;
    res.steps.push({ engine: bk, before: s1.vals, after: s2.vals, seedBefore: 1234, seedAfter: 1234, words: vsWords(s1, s2) });
    const pr = clone(b); pr.prompt = 'a new description';
    res.steps.push({ engine: bk, before: b.vals, after: b.vals, prompt: true, words: vsWords(b, pr) });
  }
  /* ---- "Also set" at Quick and Studio, as the page paints it */
  const val = (k, bk) => { const t = tries(k, bk); return t.length ? pick(t) : undefined; };
  for (const bk of ENGS) for (const m of Object.keys(MODS).filter(m => (!MODS[m].only || MODS[m].only.includes(bk)) && !MODS[m].lab && !MODS[m].source)) {
    for (const tier of ['quick', 'studio']) for (let i = 0; i < 6; i++) {
      base(bk); S.tier = tier; S.mods = [m];
      if (m === 'post') S.pedals = PEDALS.filter(p => (!p.only || p.only.includes(bk)) && rnd() < 0.5).map(p => p.id);
      const keys = (m === 'post' ? ['pp.on', ...S.pedals.flatMap(id => PED[id].keys || [])] : MODS[m].keys || []).filter(k => C[k] && on(k, bk));
      for (const k of keys) if (rnd() < 0.4) { const v = val(k, bk); if (v !== undefined) { S.vals[k] = v; if (C[k].auto) S.vals[k + '.auto'] = false; } }
      render(); if (typeof paint === 'function') paint(); paintAlso();
      const el = document.querySelector(`#bento > .tile[data-m="${m}"] > .q-also`);
      res.also.push({ engine: bk, tier, module: m, pedals: [...S.pedals], vals: clone(S.vals), text: el ? el.querySelector('span').textContent : null, link: el ? el.querySelector('[data-tier]').dataset.tier : null, keys: el ? el.querySelector('[data-ring]').dataset.ring.split(' ') : [] });
    }
  }
  /* ---- what each tile lets you change at each tier: the keys its controls carry (data-ctl, data-bind), on an empty
     desk and on one with two adapters and tagged lyrics */
  res.tiers = [];
  for (const bk of ENGS) for (const m of Object.keys(MODS).filter(m => (!MODS[m].only || MODS[m].only.includes(bk)) && !MODS[m].source)) {
    for (const loaded of [false, true]) for (const tier of MODS[m].lab ? ['lab'] : ['quick', 'studio', 'lab']) {
      base(bk); S.tier = tier; S.mods = MODS[m].lab ? [MODS[m].parentId || Object.keys(MODS).find(x => MODS[x].name === MODS[m].parent), m].filter(Boolean) : [m];
      if (loaded) { if (MODS.adapters.only.includes(bk)) S.slots = [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'grit', type: 'LoRA' }]; S.vals['lyrics.text'] = '[Verse]\na line\n[Chorus]\nanother'; }
      render(); if (typeof paint === 'function') paint();
      const t = document.querySelector(`#bento > .tile[data-m="${m}"]`);
      const keys = t ? [...new Set([...t.querySelectorAll('[data-ctl], [data-bind]')].map(e => e.dataset.ctl || e.dataset.bind))].filter(k => C[k]).sort() : null;
      res.tiers.push({ engine: bk, module: m, tier, loaded, strip: !!t?.classList.contains('is-strip'), keys, vals: clone(S.vals), slots: clone(S.slots) });
    }
  }
  /* ---- engine switches: a desk on one engine, then two switches */
  const allKeys = bk => Object.keys(C).filter(k => on(k, bk));
  for (let i = 0; i < 160; i++) {
    const from = pick(ENGS); base(from);
    for (const k of allKeys(from)) if (rnd() < 0.08) { const v = val(k, from); if (v !== undefined) { S.vals[k] = v; if (C[k].auto) S.vals[k + '.auto'] = false; } }
    if (rnd() < 0.5) { S.vals['tempo.dur'] = pick([30, 200, 300, 500]); S.vals['tempo.dur.auto'] = false; }
    if (rnd() < 0.5) S.vals['takes.n'] = pick([1, 3, 6, 9]);
    if (MODS.adapters.only.includes(from) && rnd() < 0.4) S.slots = [{ file: 'tape-lofi-v2', type: 'LoRA' }];
    const desk0 = { engine: from, vals: clone(S.vals), slots: clone(S.slots) };
    const hops = [];
    for (const to of [pick(ENGS.filter(x => x !== from)), pick(ENGS)]) {
      if (to === S.backend) continue;
      const r = enterEngine(to);
      hops.push({ to, vals: clone(S.vals), slots: clone(S.slots), hold: clone(S.eng.hold), stash: clone(S.eng.stash), news: engineNews(r) });
    }
    res.switches.push({ desk: desk0, hops });
  }
  /* ---- what a revision stores */
  for (let i = 0; i < 60; i++) {
    const bk = pick(ENGS); base(bk);
    for (const k of allKeys(bk)) if (rnd() < 0.1) { const v = val(k, bk); if (v !== undefined) S.vals[k] = v; }
    prompt.value = 'a desk';
    const withContent = rnd() < 0.5;
    res.revisions.push({ engine: bk, vals: clone(S.vals), rev: revSnap('r', withContent), withContent });
  }
  /* ---- what cleaning keeps */
  const bad = [
    { 'samp.steps': 12.5, 'tempo.dur': 9999, 'pp.ss.str': 99, 'tempo.key': 'H major', 'nope.key': 3 },
    { 'samp.steps': '30', 'pp.on': 'yes', 'lyrics.text': 42, 'tempo.bpm.auto': 'no', 'samp.cs': 0.9, 'samp.ce': 0.2 },
    { 'samp.rseed': 'forty-two', 'pp.ss.seed': '12x', 'lat.ts': '1,0.5,0.7', 'samp.cfg': -5, 'tempo.bpm': 1e9 },
    { 'samp.steps': Infinity, 'ad.adv': 'Stacked', 'dcw.mode': 'Low-Frequency', 'lyrics.text': 'x'.repeat(20010) },
  ];
  for (const v of bad) res.cleaning.push({ input: v, vals: cleanVals(v) });
  for (let i = 0; i < 40; i++) {
    const bk = pick(ENGS), v = {};
    for (const k of allKeys(bk)) if (rnd() < 0.1) { const c = C[k]; const t = tries(k, bk); v[k] = rnd() < 0.3 ? pick(['x', -1e6, 1e6, true, null, 3.3]) : t.length ? pick(t) : 1; if (c.auto && rnd() < 0.5) v[k + '.auto'] = pick([true, false, 'yes']); }
    res.cleaning.push({ input: v, vals: cleanVals(v) });
  }
  res.slots = [[{ file: 'a', type: 'LoRA', ph: [10, 90] }, { file: 2 }, { file: 'b', type: 'LoKr', ph: [90, 10] }, { file: 'c', type: 'LoRA' }, { file: 'd', type: 'LoRA' }]].map(x => ({ input: x, out: cleanSlots(x) }));
  return res;
}, PAIRS);

out.errors = errors;
const file = path.join(ROOT, 'tools', 'fixtures', 'concept.json');
fs.writeFileSync(file, JSON.stringify({ note: 'Collected from the concept page by tools/concept/collect.mjs. Do not edit by hand.', ...out }, null, 0).replace(/\n/g, '\r\n') + '\r\n');
console.log(`wrote ${path.relative(process.cwd(), file)}: ${out.tiers.length} tiles, ${out.steps.length} steps, ${out.also.length} tile lines, ${out.switches.length} switch walks, ${out.revisions.length} revisions, ${out.cleaning.length} cleanings; page errors: ${errors.length}`);
await browser.close();
