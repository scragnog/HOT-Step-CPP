// Every part on every surface (SL-04): the test build's parts gallery (src/app/testing/gallery.ts: every built part in
// a tile's body, a row, a rack, a well, an LCD, a tile's foot, a labelled pairing, on a pedal and on a side unit) drawn
// in both themes at every tier, its values changed. For each part where it stands, the check asks:
//   drawn      it has a box, inside its tile, pedal or unit;
//   legible    its label (and a button's own words) reads at 3:1 or better against what it stands on;
//   styled     its label is the page's mono caps label, not a bare browser default;
//   marked     changed, it carries the page's changed mark (a 5px dot);
//   concept    where the concept draws the same part on the same kind of surface, its label's type and colours and its
//              face's colours are the concept's (the concept page, same theme, the modules that have it).
// Screenshots of each theme and tier go to tools/visual-baseline/out/gallery/ for the eye.
//
//   kept       every part's look on every surface is what it was when it was last looked at and recorded
//              (tools/fixtures/gallery.json; --record writes it, after the screenshots have been looked at);
//   semantics  a segmented choice says which one is pressed, a choice row which one is checked;
//   new part   a part the concept never drew (markup alone: data-ctl, .changed, a .kl label) gets the page's label
//              and changed mark from the shared rule (styles/rebuild/35-part-states.css), on every surface.
//
//   node tools/visual-baseline/gallery.mjs [--page dist-test/index.html] [--concept <hotstep-tiers.html>] [--record]

import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url)), repo = path.resolve(here, '..', '..');
const require = createRequire(import.meta.url);
const pw = (() => { for (const m of [process.env.PLAYWRIGHT_MODULE, 'playwright'].filter(Boolean)) { try { return require(m); } catch { /* next */ } } throw new Error('Playwright not found: set PLAYWRIGHT_MODULE'); })();
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; };
const PAGE = path.resolve(opt('page', path.join(repo, 'dist-test', 'index.html')));
const CONCEPT = opt('concept', 'C:/Users/sneth/misc/HOT-Step-UI/hotstep-tiers.html');
const OUT = path.join(here, 'out', 'gallery'); fs.mkdirSync(OUT, { recursive: true });

const serve = file => new Promise(res => { const body = fs.readFileSync(file); const s = http.createServer((q, r) => { r.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }); r.end(body); }); s.listen(0, '127.0.0.1', () => res({ url: `http://127.0.0.1:${s.address().port}/p.html`, close: () => s.close() })); });

/* a part's root on the page, by its markup */
const ROOTS = { newPart: '.zz-new', knob: '.kn', ledButton: '.led-btn', segmented: '.seg', pick: '.pk-lcdwrap', choiceRow: '.qchoice', textField: 'input[data-bind]', meter: '.vu', note: '.lbl:not([data-seam]), .pnote', lcdValue: '.bigval', button: '.pbtn', powerButton: '.master', textArea: 'textarea', xyPad: '.xy', labelled: '.foot-ctl, .lcdsel' };
const SURFACES = { 'zz-g-tile': 'tile', 'zz-g-row': 'row', 'zz-g-rack': 'rack', 'zz-g-well': 'well', 'zz-g-lcd': 'lcd', 'zz-g-foot': 'foot', 'zz-g-labelled': 'labelled' };

/* in the page: every part's facts */
const MEASURE = `(() => {
  const ROOTS = ${JSON.stringify(ROOTS)};
  const rgb = s => { const m = /rgba?\\(([^)]+)\\)/.exec(s); if (!m) return null; const [r, g, b, a = 1] = m[1].split(/[ ,/]+/).filter(Boolean).map(Number); return { r, g, b, a }; };
  const lum = c => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b); };
  /* what an element's words stand on: the first ancestor with a background colour (a gradient face counts as its
     middle colour where its colours can be read), composited over the page */
  const under = el => { for (let e = el; e; e = e.parentElement) { const cs = getComputedStyle(e), c = rgb(cs.backgroundColor); if (c && c.a > 0.5) return c;
      const g = cs.backgroundImage.match(/rgba?\\([^)]+\\)/g); if (g && g.length) { const cc = g.map(rgb).filter(x => x && x.a > 0.5); if (cc.length) return cc[Math.floor(cc.length / 2)]; } }
    return rgb(getComputedStyle(document.body).backgroundColor) || { r: 255, g: 255, b: 255, a: 1 }; };
  const contrast = el => { const f = rgb(getComputedStyle(el).color), b = under(el); if (!f) return null; const L1 = lum(f), L2 = lum(b); return +((Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05)).toFixed(2); };
  const sig = el => { if (!el) return null; const cs = getComputedStyle(el); return { size: cs.fontSize, weight: cs.fontWeight, transform: cs.textTransform, family: cs.fontFamily.split(',')[0].trim(), color: cs.color }; };
  const face = el => { const cs = getComputedStyle(el); return { bg: cs.backgroundColor, img: cs.backgroundImage === 'none' ? 'none' : 'image', radius: cs.borderRadius }; };
  const marked = el => [el, ...el.querySelectorAll('*')].some(x => { const a = getComputedStyle(x, '::after'); return a.content !== 'none' && a.width === '5px' && a.height === '5px' && a.backgroundColor !== 'rgba(0, 0, 0, 0)'; });
  const out = [];
  /* a new part, by its markup alone, on each surface */
  for (const c of document.querySelectorAll('[data-m^="zz-g"] :is(.tile-body, .knobs, .rackrow, .samp-strip, .lcd, .tile-foot), .pedal[data-pid^="zz-g"], .pedal[data-xid^="zz-g"] .u-ctls')) { if (c.querySelector(':scope > .zz-new')) continue; const n = document.createElement('div'); n.className = 'zz-new changed'; n.dataset.ctl = 'zz.new'; n.innerHTML = '<span class="kl">New</span>'; c.appendChild(n); }
  const hosts = [...document.querySelectorAll('#bento > .tile[data-m^="zz-g"]')].map(t => [t.dataset.m, t]).concat([...document.querySelectorAll('.pedal[data-pid^="zz-g"], .pedal[data-xid^="zz-g"]')].map(p => [p.dataset.pid ? 'pedal' : 'unit', p]));
  const seen = new Set();
  /* every part the map below names, then any other control (a part a package adds: its root carries data-ctl) */
  const roots = box => [...Object.entries(ROOTS).flatMap(([part, sel]) => [...box.querySelectorAll(sel)].map(el => [part, el])), ...[...box.querySelectorAll('[data-ctl]:not(.zz-new)')].map(el => ['control', el])];
  for (const [host, box] of hosts) for (const [part, el] of roots(box)) {
    if (part === 'control' && ([...seen].some(x => x === el || x.contains(el) || el.contains(x)))) continue;
    if (el.closest('.tile, .pedal') !== box.closest('.tile, .pedal') && host !== 'pedal' && host !== 'unit') continue;
    seen.add(el);
    const r = el.getBoundingClientRect(), hb = box.getBoundingClientRect(), label = el.querySelector(':scope > .kl') || (el.classList.contains('kl') ? el : null);
    const words = part === 'ledButton' || part === 'button' || part === 'note' || part === 'lcdValue' ? el : label;
    const c = el.matches('[data-ctl]') ? el : el.querySelector('[data-ctl]') || el.closest('[data-ctl]'), ctl = c ? c.dataset.ctl + ':' + (c.classList[0] || c.tagName.toLowerCase()) : '';
    out.push({ host, part, ctl, drawn: r.width > 0 && r.height > 0, inside: r.left >= hb.left - 1 && r.right <= hb.right + 1 && r.top >= hb.top - 1 && r.bottom <= hb.bottom + 1,
      pressed: el.matches('.seg') ? el.querySelectorAll('[aria-pressed="true"]').length : el.matches('.qchoice') ? el.querySelectorAll('[aria-checked="true"]').length : null,
      contrast: words && words.textContent.trim() ? contrast(words) : null, textArea: !!el.querySelector('textarea'), label: sig(label), words: sig(words), face: face(el), changed: el.classList.contains('changed') || !!el.querySelector('.changed'), marked: marked(el.closest('.foot-ctl, .lcdsel') || el.closest('[data-ctl]') || el) });
  }
  return out;
})()`;

/* the concept's own: the parts it draws, on the surface it draws them, for comparison */
const REFS = [
  { part: 'knob', surface: 'row', scene: 'mm3', sel: '.tile[data-m="guidance"] .kn' },
  { part: 'knob', surface: 'pedal', scene: 'ace', sel: '.pedal[data-pid="ss"] .kn' },
  { part: 'ledButton', surface: 'rack', scene: 'mm3', sel: '.tile[data-m="flow"] .rackrow .led-btn' },
  { part: 'ledButton', surface: 'pedal', scene: 'ace', sel: '.pedal[data-pid="ss"] .psw .led-btn' },
  { part: 'segmented', surface: 'foot', scene: 'mm3', sel: '.tile[data-m="flow"] .tile-foot .seg' },
  { part: 'labelled', surface: 'foot', scene: 'mm3', sel: '.tile[data-m="flow"] .tile-foot .foot-ctl' },
  { part: 'pick', surface: 'pedal', scene: 'ace', sel: '.pedal[data-pid="ss"] .pk-lcdwrap' },
  { part: 'meter', surface: 'pedal', scene: 'ace', sel: '.pedal[data-pid="master"] .vu' },
  { part: 'note', surface: 'tile', scene: 'sa3', sel: '.tile[data-m="guidance"] .tile-body > .lbl' },
  { part: 'textArea', surface: 'tile', scene: 'ace', sel: '.tile[data-m="avoid"] textarea' },
  { part: 'powerButton', surface: 'tile', scene: 'ace', sel: '.master' },
  { part: 'xyPad', surface: 'tile', scene: 'ace', sel: '.tile[data-m="guidance"] .xy' },
];
const SCENE = { ace: ['ace', ['avoid', 'guidance', 'sampler', 'post'], ['ss', 'master', 'wh']], mm3: ['mm3', ['guidance', 'flow', 'plans', 'sampler', 'post'], ['ss', 'wh']], sa3: ['sa3', ['guidance', 'sampler', 'post'], ['tile', 'wh']] };
const SURFACE_OF = { tile: ['tile', 'labelled'], row: ['row'], rack: ['rack'], foot: ['foot', 'labelled'], pedal: ['pedal'] };

async function run() {
  const browser = await pw.chromium.launch(), page = await serve(PAGE), concept = fs.existsSync(CONCEPT) ? await serve(CONCEPT) : null;
  const rows = [], refs = {};
  for (const scheme of ['light', 'dark']) {
    /* the concept's references, at Lab */
    if (concept) for (const [sc, [bk, mods, pedals]] of Object.entries(SCENE)) {
      const ctx = await browser.newContext({ viewport: { width: 1600, height: 1080 }, colorScheme: scheme }); await ctx.route(u => !u.href.startsWith(concept.url.replace(/p\.html$/, '')), r => r.abort());
      const p = await ctx.newPage(); await p.goto(concept.url); await p.waitForTimeout(300);
      await p.evaluate(([bk, mods, pedals]) => { if (bk !== S.backend) enterEngine(bk); S.tier = 'lab'; S.mods = mods; S.pedals = pedals; render(); }, [bk, mods, pedals]); await p.waitForTimeout(500);
      for (const r of REFS.filter(x => x.scene === sc)) refs[`${scheme}|${r.part}|${r.surface}`] = await p.evaluate(([sel, m]) => { const el = document.querySelector(sel); if (!el) return null; return (0, eval)(m.replace('const out = [];', 'const out = []; return (() => { const el0 = document.querySelector(' + JSON.stringify(sel) + '); const label = el0.querySelector(":scope > .kl"); return { label: sig(label), words: sig(label || el0), face: face(el0) }; })();')); }, [r.sel, MEASURE]);
      await ctx.close();
    }
    for (const tier of ['quick', 'studio', 'lab']) {
      const ctx = await browser.newContext({ viewport: { width: 1600, height: 1080 }, colorScheme: scheme }); await ctx.route(u => !u.href.startsWith(page.url.replace(/p\.html$/, '')), r => r.abort());
      const p = await ctx.newPage(); const errs = []; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error' && !/Failed to load/.test(m.text())) errs.push(m.text()); });
      await p.goto(page.url); await p.waitForTimeout(300);
      await p.evaluate(([tier, mods]) => {
        S.backend = 'ace'; S.tier = tier; S.mods = [...mods, 'post']; S.pedals = ['zz-g-pedal', 'zz-g-unit'];
        for (const x of ['tile', 'row', 'rack', 'well', 'lcd', 'foot', 'labelled', 'pedal', 'unit']) Object.assign(S.vals, { [`zzg.${x}.n1`]: 7, [`zzg.${x}.c`]: 'hard', [`zzg.${x}.b`]: true, [`zzg.${x}.t`]: 'x' });
        render();
      }, [tier, Object.keys(SURFACES)]);
      await p.waitForTimeout(700);
      await p.screenshot({ path: path.join(OUT, `${scheme}-${tier}.png`), fullPage: true });
      const got = await p.evaluate(MEASURE);
      for (const g of got) rows.push({ scheme, tier, ...g, surface: SURFACES[g.host] ?? g.host });
      if (errs.length) rows.push({ scheme, tier, error: errs.join(' | ') });
      await ctx.close();
    }
  }
  await browser.close(); page.close(); concept?.close();
  /* the verdicts */
  const fails = [], seen = new Map();
  /* every part's look, kept: by place (theme, tier, host, part, the parameter it edits, its index among those), so a
     part added to the gallery moves no other part's record */
  const FIX = path.join(repo, 'tools', 'fixtures', 'gallery.json'), look = {}, nth = new Map();
  for (const r of rows.filter(x => x.part)) { const k = `${r.scheme}|${r.tier}|${r.host}|${r.part}|${r.ctl}`, i = (nth.get(k) ?? -1) + 1; nth.set(k, i); look[`${k}|${i}`] = JSON.stringify([r.label, r.words, r.face, r.marked, r.pressed]); }
  if (args.includes('--record')) { fs.writeFileSync(FIX, JSON.stringify(look, null, 1)); console.log(`recorded ${Object.keys(look).length} looks into ${path.relative(repo, FIX)}`); }
  else if (fs.existsSync(FIX)) { const kept = JSON.parse(fs.readFileSync(FIX, 'utf8')); for (const [k, v] of Object.entries(kept)) if (look[k] !== v) fails.push(`${k.replace(/\|/g, ' ')}: its look is not what was recorded (${look[k] ? 'changed' : 'gone'}): ${(look[k] ?? '').slice(0, 160)}`); for (const k of Object.keys(look)) if (!(k in kept)) fails.push(`${k}: new since the last record`); }
  for (const r of rows) {
    if (r.error) { fails.push(`${r.scheme} ${r.tier}: page error ${r.error}`); continue; }
    const at = `${r.scheme} ${r.tier} ${r.part} on ${r.surface}`;
    seen.set(`${r.part}|${r.surface}`, (seen.get(`${r.part}|${r.surface}`) || 0) + 1);
    if (!r.drawn) fails.push(`${at}: not drawn`);
    else if (!r.inside) fails.push(`${at}: stands outside its ${r.surface === 'pedal' || r.surface === 'unit' ? r.surface : 'tile'}`);
    if (r.contrast !== null && r.contrast < 3) fails.push(`${at}: its words read at ${r.contrast}:1 (want 3:1)`);
    if (r.label && (r.label.transform !== 'uppercase' || !/mono|Mono/.test(r.label.family) && !/var|Geist/.test(r.label.family))) { if (r.label.transform !== 'uppercase' && r.part !== 'knob') fails.push(`${at}: its label is not the page's caps label (${JSON.stringify(r.label)})`); }
    /* a text area's words are its change (the concept marks Avoid's note no other way) */
    if (r.pressed !== null && r.pressed !== undefined && r.pressed !== 1) fails.push(`${at}: ${r.pressed} of its options say they are chosen (want 1)`);
    if (r.changed && !r.marked && !(r.part === 'textArea' || (r.part === 'labelled' && r.textArea))) fails.push(`${at}: changed, and no changed mark`);
    const surfaceKind = Object.entries(SURFACE_OF).find(([, ss]) => ss.includes(r.surface))?.[0] ?? r.surface;
    const ref = refs[`${r.scheme}|${r.part}|${surfaceKind}`];
    if (ref && r.tier === 'lab') {
      const want = { ...ref }, have = { label: r.label, words: r.words, face: r.face };
      for (const k of ['size', 'weight', 'transform', 'family']) for (const w of ['label', 'words']) if (want[w] && have[w] && want[w][k] !== have[w][k]) fails.push(`${at}: ${w} ${k} ${have[w][k]}, the concept's ${want[w][k]}`);
      if (surfaceKind !== 'pedal') { for (const w of ['label', 'words']) if (want[w] && have[w] && want[w].color !== have[w].color) fails.push(`${at}: ${w} colour ${have[w].color}, the concept's ${want[w].color}`);
        if (want.face && (want.face.bg !== have.face.bg || want.face.img !== have.face.img)) fails.push(`${at}: face ${JSON.stringify(have.face)}, the concept's ${JSON.stringify(want.face)}`); }
    }
  }
  const parts = [...new Set(rows.filter(r => r.part).map(r => r.part))], surfaces = [...new Set(rows.filter(r => r.surface).map(r => r.surface))];
  console.log(`gallery: ${rows.length} part instances, ${parts.length} parts on ${surfaces.length} surfaces (${surfaces.join(', ')}), 2 themes x 3 tiers; ${Object.keys(refs).filter(k => refs[k]).length} concept references compared at Lab`);
  const missing = []; for (const p of Object.keys(ROOTS)) for (const s of [...Object.values(SURFACES), 'pedal', 'unit']) if (!seen.has(`${p}|${s}`)) missing.push(`${p} on ${s}`);
  if (missing.length) console.log(`not drawn on (by the definition or the tier; checked where drawn): ${missing.join(', ')}`);
  console.log(`screenshots: ${path.relative(repo, OUT)}`);
  const uniq = [...new Set(fails)];
  if (uniq.length) { console.log(`\n${uniq.length} failed:`); uniq.slice(0, 80).forEach(f => console.log('  ' + f)); process.exitCode = 1; } else console.log('RESULT: every part reads right wherever it stands');
}
await run();
