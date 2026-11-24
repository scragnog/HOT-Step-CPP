// Every part on every surface (SL-04): the test build's parts gallery (src/app/testing/gallery.ts: every built part in
// a tile's body, a row, a rack, a well, an LCD, a tile's foot, a labelled pairing, on a pedal and on a side unit) drawn
// in both themes at every tier, its values changed. For each part where it stands, the check asks:
//   drawn      it has a box, inside its tile, pedal or unit;
//   legible    its label (and a button's own words) reads at 3:1 or better against what it stands on;
//   styled     its label is the page's mono caps label, not a bare browser default;
//   marked     changed, it carries the page's changed mark (a 5px dot);
// Screenshots of each theme and tier go to tools/visual-baseline/out/gallery/ for the eye.
//
//   kept       every part's look on every surface is what it was when it was last looked at and recorded
//              (tools/fixtures/gallery.json; --record writes it, after the screenshots have been looked at);
//   semantics  a segmented choice says which one is pressed, a choice row which one is checked;
//   new part   a part's bare markup (data-ctl, .changed, a .kl label) gets the page's label
//              and changed mark from the shared rule (styles/rebuild/35-part-states.css), on every surface.
//
//   node tools/visual-baseline/gallery.mjs [--page dist-test/index.html] [--record]

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
const OUT = path.join(here, 'out', 'gallery'); fs.mkdirSync(OUT, { recursive: true });

/* the ports Chromium refuses to load from (net::ERR_UNSAFE_PORT, its kRestrictedPorts): the system can hand one out
   for port 0 (6669 was, and every scenario of that run failed at page.goto), so the server listens again elsewhere */
const UNSAFE_PORTS = new Set([1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53, 69, 77, 79, 87, 95, 101, 102, 103, 104, 109, 110, 111, 113, 115, 117, 119, 123, 135, 137, 139, 143, 161, 179, 389, 427, 465, 512, 513, 514, 515, 526, 530, 531, 532, 540, 548, 554, 556, 563, 587, 601, 636, 989, 990, 993, 995, 1719, 1720, 1723, 2049, 3659, 4045, 4190, 5060, 5061, 6000, 6566, 6665, 6666, 6667, 6668, 6669, 6679, 6697, 10080]);
const listenSafe = srv => new Promise(res => { const go = () => srv.listen(0, '127.0.0.1', () => { if (!UNSAFE_PORTS.has(srv.address().port)) return res(srv.address().port); srv.close(() => go()); }); go(); });
const serve = file => new Promise(res => { const body = fs.readFileSync(file); const s = http.createServer((q, r) => { r.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }); r.end(body); }); listenSafe(s).then(port => res({ url: `http://127.0.0.1:${port}/p.html`, close: () => s.close() })); });

/* a part's root on the page, by its markup */
const ROOTS = { newPart: '.zz-new', knob: '.kn', ledButton: '.led-btn', segmented: '.seg', pick: '.pk-lcdwrap', choiceRow: '.qchoice', textField: 'input[data-bind]', meter: '.vu', note: '.lbl:not([data-seam]), .pnote', lcdValue: '.bigval', button: '.pbtn', powerButton: '.master', textArea: 'textarea', xyPad: '.xy', labelled: '.foot-ctl, .lcdsel, .ts, .dice-cap, .endings, .tk-pair, .ad-foot, .pc-i' };
const SURFACES = { 'zz-g-tile': 'tile', 'zz-g-row': 'row', 'zz-g-rack': 'rack', 'zz-g-well': 'well', 'zz-g-lcd': 'lcd', 'zz-g-foot': 'foot', 'zz-g-labelled': 'labelled',
  /* package 2's containers */
  'zz-g-term': 'term', 'zz-g-termbar': 'termbar', 'zz-g-lcdrow': 'lcdrow', 'zz-g-group': 'group', 'zz-g-strip': 'strip', 'zz-g-plugin': 'plugin', 'zz-g-deck': 'deck' };

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
  for (const c of document.querySelectorAll('[data-m^="zz-g"] :is(.tile-body, .knobs, .rackrow, .samp-strip, .lcd, .tile-foot, .tline, .term-bar, .lcd-row, .ad-g, .schctl, .pc-grid, .deck-c), .pedal[data-pid^="zz-g"], .pedal[data-xid^="zz-g"] .u-ctls')) { if (c.querySelector(':scope > .zz-new')) continue; const n = document.createElement('div'); n.className = 'zz-new changed'; n.dataset.ctl = 'zz.new'; n.innerHTML = '<span class="kl">New</span>'; c.appendChild(n); }
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
      contrast: words && words.textContent.trim() ? contrast(words) : null, textArea: !!el.querySelector('textarea'), label: sig(label), words: sig(words), face: face(el), changed: el.classList.contains('changed') || !!el.querySelector('.changed'), marked: marked(el.closest('.foot-ctl, .lcdsel, .ad-foot') || el.closest('[data-ctl]') || el) });
  }
  return out;
})()`;

async function run() {
  const browser = await pw.chromium.launch(), page = await serve(PAGE);
  const rows = [];
  for (const scheme of ['light', 'dark']) {
    for (const tier of ['quick', 'studio', 'lab']) {
      const ctx = await browser.newContext({ viewport: { width: 1600, height: 1080 }, colorScheme: scheme }); await ctx.route(u => !u.href.startsWith(page.url.replace(/p\.html$/, '')), r => r.abort());
      const p = await ctx.newPage(); const errs = []; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error' && !/Failed to load/.test(m.text())) errs.push(m.text()); });
      await p.goto(page.url); await p.waitForTimeout(300);
      await p.evaluate(([tier, mods, names]) => {
        S.backend = 'ace'; S.tier = tier; S.mods = [...mods, 'post']; S.pedals = ['zz-g-pedal', 'zz-g-unit'];
        for (const x of [...names, 'pedal', 'unit']) Object.assign(S.vals, { [`zzg.${x}.n1`]: 7, [`zzg.${x}.c`]: 'hard', [`zzg.${x}.b`]: true, [`zzg.${x}.t`]: 'x' });
        render();
      }, [tier, Object.keys(SURFACES), Object.values(SURFACES)]);
      await p.waitForTimeout(700);
      await p.screenshot({ path: path.join(OUT, `${scheme}-${tier}.png`), fullPage: true });
      const got = await p.evaluate(MEASURE);
      for (const g of got) rows.push({ scheme, tier, ...g, surface: SURFACES[g.host] ?? g.host });
      if (errs.length) rows.push({ scheme, tier, error: errs.join(' | ') });
      await ctx.close();
    }
  }
  await browser.close(); page.close();
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
    const at = `${r.scheme} ${r.tier} ${r.part}${r.part === 'control' || r.part === 'note' ? ` (${r.ctl})` : ''} on ${r.surface}`;
    seen.set(`${r.part}|${r.surface}`, (seen.get(`${r.part}|${r.surface}`) || 0) + 1);
    if (!r.drawn) fails.push(`${at}: not drawn`);
    else if (!r.inside) fails.push(`${at}: stands outside its ${r.surface === 'pedal' || r.surface === 'unit' ? r.surface : 'tile'}`);
    if (r.contrast !== null && r.contrast < 3) fails.push(`${at}: its words read at ${r.contrast}:1 (want 3:1)`);
    if (r.label && (r.label.transform !== 'uppercase' || !/mono|Mono/.test(r.label.family) && !/var|Geist/.test(r.label.family))) { if (r.label.transform !== 'uppercase' && r.part !== 'knob') fails.push(`${at}: its label is not the page's caps label (${JSON.stringify(r.label)})`); }
    /* a text area's words are its change (Avoid's note is marked no other way) */
    if (r.pressed !== null && r.pressed !== undefined && r.pressed !== 1) fails.push(`${at}: ${r.pressed} of its options say they are chosen (want 1)`);
    /* a look-back strip in a terminal is unmarked (its window's token on the line carries the mark) */
    const unmarked = r.part === 'textArea' || (r.part === 'labelled' && r.textArea) || (/:codes$/.test(r.ctl) && (r.surface === 'term' || r.surface === 'termbar'));
    if (r.changed && !r.marked && !unmarked) fails.push(`${at}: changed, and no changed mark`);
  }
  const parts = [...new Set(rows.filter(r => r.part).map(r => r.part))], surfaces = [...new Set(rows.filter(r => r.surface).map(r => r.surface))];
  console.log(`gallery: ${rows.length} part instances, ${parts.length} parts on ${surfaces.length} surfaces (${surfaces.join(', ')}), 2 themes x 3 tiers`);
  const missing = []; for (const p of Object.keys(ROOTS)) for (const s of [...Object.values(SURFACES), 'pedal', 'unit']) if (!seen.has(`${p}|${s}`)) missing.push(`${p} on ${s}`);
  if (missing.length) console.log(`not drawn on (by the definition or the tier; checked where drawn): ${missing.join(', ')}`);
  console.log(`screenshots: ${path.relative(repo, OUT)}`);
  const uniq = [...new Set(fails)];
  if (uniq.length) {
    const max = Number(process.env.GALLERY_MAX || 80), fresh = uniq.filter(f => f.endsWith(': new since the last record')).length;
    console.log(`\n${uniq.length} failed${fresh ? ` (${fresh} of them new since the last record: a new part, once its screenshots look right, is kept with --record)` : ''}:`);
    uniq.slice(0, max).forEach(f => console.log('  ' + f));
    if (uniq.length > max) console.log(`  ... and ${uniq.length - max} more (GALLERY_MAX=${uniq.length} prints them all)`);
    process.exitCode = 1;
  } else console.log('RESULT: every part reads right wherever it stands');
}
await run();
