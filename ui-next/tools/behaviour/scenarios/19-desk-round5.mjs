// Round 5, the desk at Quick and Studio, and tier changes (review Q, with L-04, L-07, L-08, L-10, L-12): read-outs that
// stay true while dragged, the tier change as one object per part, the reader's place kept, drops that do not scroll,
// every tier saying what it hides, and factory desks that compose.
const ACE = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes'];
const desk = (tier, mods, pedals = []) => t => t.do(`${tier} desk`, ([tier, mods, pedals]) => { S.tier = tier; S.mods = [...mods]; S.pedals = [...pedals]; pedals.forEach(id => { if (PED[id]?.on) S.vals[PED[id].on] = true; }); S.qAll = true; render(); }, [tier, mods, pedals], 400);
/* the empty part of the desk: in every column, the gaps between tiles and the foot under the last one, down to the
   desk's lowest edge, as a share of the desk's area */
const EMPTY = () => {
  const g = document.querySelector('#bento'), gr = g.getBoundingClientRect(), cols = getComputedStyle(g).gridTemplateColumns.split(' ').map(parseFloat);
  const ts = [...g.children].filter(t => t.offsetParent && getComputedStyle(t).position !== 'absolute').map(t => { const r = t.getBoundingClientRect(); return { l: r.left - gr.left, t: r.top - gr.top, w: r.width, h: r.height }; });
  const bottom = Math.max(0, ...ts.map(t => t.t + t.h)); let x = 0, area = 0;
  cols.forEach(cw => { const mid = x + cw / 2; let y = 0; ts.filter(t => t.l <= mid && mid <= t.l + t.w).sort((a, b) => a.t - b.t).forEach(t => { area += Math.max(0, t.t - y - (y ? 12 : 0)) * cw; y = t.t + t.h; }); area += Math.max(0, bottom - y) * cw; x += cw + 12; });
  return area / Math.max(1, gr.width * bottom);
};
/* every factory desk, on the engines whose modules are composed for it, at every tier, at one width: the share of the desk
   left empty stays under a limit. 8% on ACE-Step (the default engine: its factory desks measure 6% at worst, so a tile
   whose height changes by a row's worth fails here), 12% on MiniMax-Music3 and YuE2, 8% on Stable Audio 3 and HeartMuLa
   (since round 5's implS: Length takes the row there and a one-take Takes strip a row of its own; they measure 3.6% at worst,
   so the default engine's limit holds them). */
const composes = (w, h) => ({ id: `factory-desks-leave-no-empty-desk-${w}`, w, h, run: async t => {
  const bad = await t.js(src => { const empty = eval(src), out = [];
    for (const [eng, lim] of [['ace', .08], ['mm3', .12], ['yue', .12], ['sa3', .08], ['hm', .08]]) { if (S.backend !== eng) switchEngine(eng);
      for (const f of FACTORY.filter(f => f.mods.length)) for (const tier of ['quick', 'studio', 'lab']) {
        S.tier = tier; S.mods = [...f.mods]; S.pedals = [...f.pedals]; S.pedals.forEach(id => { if (PED[id]?.on) S.vals[PED[id].on] = true; }); S.qAll = true; render();
        const e = empty(); if (e > lim) out.push(`${eng} ${f.name} ${tier}: ${Math.round(e * 100)}% empty`); } }
    switchEngine('ace'); return out; }, '(' + EMPTY.toString() + ')');
  await t.eq(`every factory desk at ${w}: the desk is composed, with no more than 8% of it empty on ACE-Step (Q-23, L-08)`, b => b, [], bad);
} });
export default [
  { id: 'bpm-drag-shows-the-value', w: 1920, h: 1080, run: async t => {
    await desk('quick', ['tempo'])(t);
    await t.click('.tile[data-m="tempo"] [data-auto="tempo.bpm"]');
    const lcd = () => { const el = document.querySelector('input[data-num="tempo.bpm"]'); return (el.value === numVal('tempo.bpm') && !el.dataset.editing) || `shows ${el.value}, BPM ${get('tempo.bpm')}, editing ${el.dataset.editing}`; };
    for (let i = 0; i < 3; i++) {
      await t.mouseDrag('input[data-num="tempo.bpm"]', [[0, -10], [0, -10], [0, -10], [0, -10], [0, -10], [0, -10]]);
      await t.ok(`drag ${i + 1}: the BPM figure shows the BPM, and is not left in typing mode (Q-02)`, lcd);
    }
  } },
  { id: 'avoid-note-text-fills-the-sheet', w: 1920, h: 1080, run: async t => {
    await desk('studio', ['avoid', 'tempo'])(t);
    await t.ok('the text area fills the note under its tape (Q-03, L-04)', () => { const ta = document.querySelector('.note textarea'), n = ta.closest('.note'); return ta.clientHeight >= n.clientHeight - 40 || `text ${ta.clientHeight}px in a ${n.clientHeight}px note`; });
    await t.click('.note textarea');
    await t.type('jazz, piano, slow ballads, heavy autotune, distorted guitars, children choir, vinyl crackle, spoken word intro, long fade out, saxophone solo, brass section, out of tune bells, kazoo');
    await t.click('.tile[data-m="tempo"] .tile-head h3');
    await t.ok('left, the note reads from its first word (Q-03)', () => document.querySelector('.note textarea').scrollTop === 0);
    await t.ok('Avoid is never widened into a half-empty tile (Q-20)', () => !('grow' in document.querySelector('.tile[data-m="avoid"]').dataset));
  } },
  { id: 'tier-change-is-one-object-per-part', w: 1600, h: 900, run: async t => {
    await desk('quick', ['tempo', 'avoid', 'takes', 'seed'])(t);
    await t.do('listen to the transition', () => { window.__vtA = null; const o = document.startViewTransition.bind(document);
      document.startViewTransition = cb => { const v = o(cb); v.ready.then(() => { const as = document.getAnimations().filter(a => a.effect?.pseudoElement); window.__vtA = as.map(a => ({ p: a.effect.pseudoElement, delay: a.effect.getTiming().delay, end: a.effect.getComputedTiming().endTime })); }, () => {}); return v; }; });
    await t.click('#tiers [data-tier="studio"]');
    await t.ok('a tier change ran as a view transition', () => Array.isArray(window.__vtA) && window.__vtA.length > 0 || 'no animations seen');
    await t.ok('the page itself does not cross-fade: no fade of the old or new page (Q-13)', () => !window.__vtA.some(a => /-(old|new)\(root\)/.test(a.p)) || window.__vtA.filter(a => /root/.test(a.p)).map(a => a.p + ' ' + a.end));
    await t.ok('the tier switch does not take part: no group, no image animates (Q-13)', () => !window.__vtA.some(a => /vt-tiers\)/.test(a.p)));
    await t.ok('a tile\'s header does not fade (Q-13)', () => !window.__vtA.some(a => /^::view-transition-(old|new)\(vt-h-(tempo|takes|seed|avoid)\)/.test(a.p)));
    await t.ok('a tile\'s old and new faces are never shown together (Q-13)', () => { const bad = ['tempo', 'avoid', 'takes', 'seed'].filter(m => { const o = window.__vtA.find(a => a.p === `::view-transition-old(vt-${m})`), n = window.__vtA.find(a => a.p === `::view-transition-new(vt-${m})`); return !o || !n || o.end > n.delay + 1; }).map(m => JSON.stringify(window.__vtA.filter(a => a.p.includes(`(vt-${m})`)))); return !bad.length || bad; });
    await t.ok('the prompt card\'s words are one object: they do not fade at all (Q-13)', () => !window.__vtA.some(a => /^::view-transition-(old|new)\(vt-prompt\)/.test(a.p)));
    await t.eq('after it, nothing is left named', () => [document.documentElement.classList.contains('tier-morph'), [...document.querySelectorAll('[style*="view-transition-name"]')].length], [false, 0]);
  } },
  { id: 'rapid-tier-changes-raise-no-error', w: 1600, h: 900, run: async t => {
    await desk('studio', ACE)(t);
    await t.do('count unhandled rejections', () => { window.__rej = []; addEventListener('unhandledrejection', e => window.__rej.push(String(e.reason?.message || e.reason))); });
    await t.do('five tier changes, each before the last has started', () => { for (const k of ['studio', 'lab', 'quick', 'lab', 'studio']) setTier(k); });
    for (const k of ['Alt+3', 'Alt+1', 'Alt+2']) await t.press(k);
    await t.eq('no change left a rejection unhandled (Q-21)', () => window.__rej, []);
    await t.eq('the last tier wins, and no transition is left running (Q-21; an uncaught rejection fails this scenario)', () => [S.tier, document.documentElement.classList.contains('tier-morph')], ['studio', false]);
  } },
  ...['reading', 'pointer', 'focus'].map(how => ({ id: `tier-change-keeps-the-place-${how}`, w: 1920, h: 1080, run: async t => {
    await desk('studio', [...ACE, 'post'], ['norm', 'ss', 'lift'])(t);
    /* Sampler's header just under the bar */
    if (how !== 'pointer') await t.hover('#tiers');
    await t.do('scroll to the Sampler', () => { const line = document.querySelector('.deskbar').getBoundingClientRect().bottom, r = document.querySelector('.tile[data-m="sampler"]').getBoundingClientRect(); scrollBy(0, r.top - line - 2); });
    if (how === 'pointer') await t.hover('.tile[data-m="seed"] .tile-head h3');
    if (how === 'focus') await t.focus('.tile[data-m="takes"] .grip');
    let m = { reading: 'sampler', pointer: 'seed', focus: 'takes' }[how];
    for (const [k, n] of [['quick', 1], ['lab', 3], ['studio', 2]]) {
      /* reading: the tile whose top edge is nearest the bar (first in the order on a tie) */
      if (how === 'reading') m = await t.js(() => { const line = document.querySelector('.deskbar').getBoundingClientRect().bottom; let best = Infinity, el = null; document.querySelectorAll('#bento > .tile[data-m], .lyr-card').forEach(x => { const r = x.getBoundingClientRect(); if (r.bottom <= line + 8 || r.top >= innerHeight) return; const d = Math.abs(r.top - line); if (d < best) { best = d; el = x; } }); return el?.dataset.m || 'lyrics'; });
      if (how === 'pointer') await t.hover(`.tile[data-m="${m}"] .tile-head h3`);
      const top = await t.js(m =>document.querySelector(`.tile[data-m="${m}"]`).getBoundingClientRect().top, m);
      await t.press(`Alt+${n}`);
      await t.ok(`${how}: to ${k}, ${m} keeps its top edge where it was (Q-14)`, ([m, top]) => { const r = document.querySelector(`.tile[data-m="${m}"]`).getBoundingClientRect().top; return Math.abs(r - top) < 2 || `${Math.round(top)} → ${Math.round(r)}`; }, [m, top]);
    }
  } })),
  { id: 'a-drop-does-not-scroll', w: 1920, h: 1080, run: async t => {
    await desk('studio', ['tempo', 'avoid', 'timbre', 'sampler', 'seed', 'takes'])(t);
    await t.do('scroll 300', () => scrollTo(0, 300));
    await t.do('drag Takes over the top of Tempo', () => { const el = document.querySelector('.tile[data-m="takes"] [data-drag]'); window.__dt = new DataTransfer(); el.dispatchEvent(new DragEvent('dragstart', { bubbles: true, cancelable: true, dataTransfer: window.__dt })); }, null, 200);
    await t.do('over', () => { const r = document.querySelector('.tile[data-m="tempo"]').getBoundingClientRect(), x = r.left + r.width * .85, y = 150; const o = { bubbles: true, cancelable: true, dataTransfer: window.__dt, clientX: x, clientY: y }; const el = document.elementFromPoint(x, y); el.dispatchEvent(new DragEvent('dragenter', o)); el.dispatchEvent(new DragEvent('dragover', o)); window.__dpt = [el, o]; }, null, 300);
    const p = await t.js(() => { const g = document.querySelector('#bento > .ghost').getBoundingClientRect(); return { y: scrollY, gx: g.left, gy: g.top }; });
    await t.do('drop', () => { const [el, o] = window.__dpt; el.dispatchEvent(new DragEvent('drop', o)); }, null, 600);
    await t.ok('nothing scrolls, and the tile stands where its preview stood (Q-15)', p => { const r = document.querySelector('.tile[data-m="takes"]').getBoundingClientRect(); return (scrollY === p.y && Math.abs(r.left - p.gx) < 2 && Math.abs(r.top - p.gy) < 2) || `scroll ${p.y} → ${scrollY}, preview ${Math.round(p.gx)},${Math.round(p.gy)}, tile ${Math.round(r.left)},${Math.round(r.top)}`; }, p);
    await t.ok('the desk paints under the bars, whatever moves on it (Q-15)', () => getComputedStyle(document.querySelector('#deskwrap')).zIndex === '0');
  } },
  { id: 'every-tier-says-what-it-hides', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['planner', 'sampler', 'post'], ['ss', 'norm'])(t);
    await t.do('set Lab-only values', () => { setVal('samp.cs', 0.1); setVal('lm.topk', 40); setVal('pp.ss.steps', 12); setVal('pp.ss.str', 25); });
    await t.press('Alt+2');
    await t.ok('Studio: the Sampler names a value only Lab shows, and leads to Lab (Q-16)', () => /Also set: CFG interval start 0\.10\. Edit in Lab/.test(document.querySelector('.tile[data-m="sampler"] .q-also')?.textContent || ''));
    await t.ok('Studio: the Pedalboard names its pedals\' hidden settings, with the pedal (Q-16)', () => /StableStep steps 12/.test(document.querySelector('.tile[data-m="post"] .q-also')?.textContent || '') && !/refine/.test(document.querySelector('.tile[data-m="post"] .q-also').textContent));
    await t.press('Alt+1');
    await t.ok('Quick: the board also names the refine it hides there (Q-16)', () => /StableStep refine 25% · StableStep steps 12/.test(document.querySelector('.tile[data-m="post"] .q-also')?.textContent || ''));
    await t.press('Alt+3');
    await t.ok('Lab draws everything: no such line', () => !document.querySelector('.q-also'));
  } },
  { id: 'quick-adapters-say-it-once', w: 1920, h: 1080, run: async t => {
    await t.do('two adapters, stacked', () => { S.tier = 'quick'; S.mods = ['adapters', 'avoid', 'timbre', 'seed']; S.slots = [{ file: 'synthwave-80s', type: 'LoRA' }, { file: 'tape-lofi-v2', type: 'LoRA' }]; S.vals['ad.adv'] = 'Advanced'; render(); }, null, 400);
    await t.ok('one line, in the control\'s words, with the way to Studio (Q-18)', () => { const tl = document.querySelector('.tile[data-m="adapters"]'); return (/2 adapters, stacked\. Edit in Studio/.test(tl.textContent) && !tl.querySelector('.q-also') && !/Advanced|Open Studio/.test(tl.textContent)) || tl.textContent.replace(/\s+/g, ' '); });
    await t.ok('the name is not cut: every letter of it shows (Q-18)', () => { const n = document.querySelector('.qcart .lab-n'); return n.scrollHeight <= n.clientHeight + 1 && n.scrollWidth <= n.clientWidth + 1 && n.textContent.replace(/\s/g, '') === 'synthwave-80s'; });
  } },
  { id: 'side-unit-object-stays-inside-the-unit', w: 1920, h: 1080, run: async t => {
    await desk('studio', ['post'], ['norm', 'ss', 'vst', 'master', 'lrc', 'q'])(t);
    await t.do('a two-column board', () => { const b = document.querySelector('.tile[data-m="post"]'); b.style.width = '647px'; b.style.justifySelf = 'start'; }, null, 300);
    await t.ok('nothing on a side unit crosses its lip (Q-05)', () => { const bad = [...document.querySelectorAll('.pedal.xunit')].filter(u => { const r = u.getBoundingClientRect(), c = u.querySelector('.u-ctls')?.getBoundingClientRect(); return c && c.right > r.right - 28; }).map(u => u.dataset.xid); return !bad.length || bad; });
  } },
  composes(1920, 1080), composes(1600, 900), composes(1536, 864), composes(1440, 900), composes(1280, 720),
  { id: 'sampler-knobs-share-the-well', w: 1920, h: 1080, run: async t => {
    await desk('studio', ['sampler', 'seed'])(t);
    await t.ok('the four controls stand on four equal columns across the well (Q-25, L-10)', () => { const s = document.querySelector('.samp-strip'), r = s.getBoundingClientRect(), k = [...s.children].map(c => c.getBoundingClientRect()); const last = k[k.length - 1]; const cs = k.map(b => b.left + b.width / 2), d = cs.slice(1).map((c, i) => c - cs[i]); return (k.length === 4 && Math.max(...d) - Math.min(...d) < 3 && Math.abs((cs[0] - r.left) - (r.right - cs[3])) < 6) || `centres ${cs.map(c => Math.round(c - r.left))} in ${Math.round(r.width)}`; });
  } },
];
