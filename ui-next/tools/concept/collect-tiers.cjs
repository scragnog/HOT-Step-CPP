// What each tile lets you change at each tier, on the running concept page, at NON-default values:
// for every module on every engine, every switch and choice of the module (and of the module it sits under) is put on
// each other value in turn (six at most), with 0, 1 and 2 adapters where the engine has them; the Pedalboard with every
// pedal and unit on the board, each of its switches and choices likewise. Writes tiers-nondefault.json.
const { chromium } = require('C:/Users/sneth/AppData/Local/Temp/claude/eaea7d32363d0709/50064857-58df-438d-a589-216d56e57e6a/scratchpad/pw/node_modules/playwright');
const fs = require('fs'), path = require('path');
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1920, height: 1080 } })).newPage();
  const errs = []; page.on('pageerror', e => errs.push(e.message));
  await page.goto('file:///C:/Users/sneth/misc/HOT-Step-UI/hotstep-tiers.html');
  await page.waitForTimeout(600);
  const out = await page.evaluate(() => {
    const ENGS = Object.keys(BK), res = [];
    const on = (k, bk) => C[k] && (!C[k].only || C[k].only.includes(bk));
    const base = bk => { S.backend = bk; S.vals = {}; S.slots = []; S.seed = null; S.mods = []; S.pedals = []; S.eng = { stash: {}, hold: {} }; };
    const alts = k => { const c = C[k], d = defOf(k); if (c.type === 'toggle') return [!d]; const o = optsOf(k); return Array.isArray(o) ? o.filter(v => v !== d).slice(0, 6) : []; };
    const keysIn = el => [...new Set([...el.querySelectorAll('[data-ctl], [data-bind]')].map(e => e.dataset.ctl || e.dataset.bind))].filter(k => C[k]).sort();
    for (const bk of ENGS) {
      for (const m of Object.keys(MODS).filter(m => (!MODS[m].only || MODS[m].only.includes(bk)) && !MODS[m].source && m !== 'post')) {
        const parent = MODS[m].lab ? (MODS[m].parentId || Object.keys(MODS).find(x => MODS[x].name === MODS[m].parent)) : null;
        const ks = [...new Set([...(MODS[m].keys || []), ...(parent ? MODS[parent].keys || [] : [])])].filter(k => on(k, bk) && alts(k).length);
        const adapters = MODS.adapters.only.includes(bk) && (m === 'adapters' || m === 'plans' || m === 'prompt') ? [0, 1, 2] : [0];
        for (const tier of MODS[m].lab ? ['lab'] : ['quick', 'studio', 'lab']) for (const n of adapters) for (const k of ks) for (const v of alts(k)) {
          base(bk); S.tier = tier; S.mods = [parent, m].filter(Boolean);
          S.slots = [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'grit', type: 'LoRA' }].slice(0, n);
          S.vals[k] = v; if (C[k].auto) S.vals[k + '.auto'] = false;
          try { render(); if (typeof paint === 'function') paint(); } catch (e) { res.push({ engine: bk, kind: 'module', id: m, tier, error: String(e) }); continue; }
          const t = document.querySelector(`#bento > .tile[data-m="${m}"]`);
          res.push({ engine: bk, kind: 'module', id: m, tier, set: [k, v], slots: n, parent, strip: !!t?.classList.contains('is-strip'), keys: t ? keysIn(t) : null });
        }
      }
      /* the board: every pedal and unit this engine has, each of its switches and choices on each other value */
      const board = [...PEDALS, ...EXTRAS].filter(p => !p.only || p.only.includes(bk));
      for (const p of board) {
        const ks = (PED[p.id]?.keys || p.keys || []).filter(k => on(k, bk) && alts(k).length);
        for (const tier of ['quick', 'studio', 'lab']) for (const [k, v] of [[null, null], ...ks.flatMap(k => alts(k).map(v => [k, v]))]) {
          base(bk); S.tier = tier; S.mods = ['post']; S.pedals = board.map(x => x.id);
          if (k) { S.vals[k] = v; }
          try { render(); if (typeof paint === 'function') paint(); } catch (e) { res.push({ engine: bk, kind: 'board', id: p.id, tier, error: String(e) }); continue; }
          const el = document.querySelector(`#bento [data-pid="${p.id}"], #bento [data-xid="${p.id}"]`);
          res.push({ engine: bk, kind: 'board', id: p.id, tier, set: k ? [k, v] : null, keys: el ? keysIn(el) : null, found: !!el });
        }
      }
    }
    const controls = {}; for (const bk of ENGS) { S.backend = bk; controls[bk] = Object.fromEntries(Object.keys(C).filter(k => on(k, bk)).map(k => [k, { type: C[k].type, opts: Array.isArray(optsOf(k)) ? optsOf(k) : null, names: C[k].names || null }])); }
    const sample = document.querySelector('#bento .pedal, #bento [class*="pedal"]');
    return { res, controls, pedalAttrs: sample ? [...sample.attributes].map(a => a.name + '=' + a.value.slice(0, 40)) : null };
  });
  fs.writeFileSync(path.join(__dirname, '..', 'tiers-nondefault.json'), JSON.stringify(out));
  console.log('states:', out.res.length, 'errors in page:', errs.length, 'render errors:', out.res.filter(r => r.error).length, 'board found:', out.res.filter(r => r.kind === 'board' && r.found).length, '/', out.res.filter(r => r.kind === 'board').length, 'pedal attrs:', JSON.stringify(out.pedalAttrs));
  await browser.close();
})();
