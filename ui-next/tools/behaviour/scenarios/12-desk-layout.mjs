// Round 4, the desk's layout (Q-01, Q-02, Q-08, L-01): the user's order is the layout. A drop lands where its preview
// stood and the preview stands under the pointer; the arrow keys move a tile one visible step; the same desk gives the
// same layout after a reload, another page, a resize there and back, a tier there and back, and Undo/Redo; the factory
// and starter desks compose at every size; reading order, Tab order and visual order agree.
const ACE = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes'];
const desk = (tier, mods, pedals = []) => t => t.do(`${tier} desk`, ([tier, mods, pedals]) => { S.tier = tier; S.mods = [...mods]; S.pedals = [...pedals]; S.qAll = true; render(); }, [tier, mods, pedals], 400);
/* every tile's box on the page, as one string: the layout */
const layout = () => [...document.querySelectorAll('#bento > .tile[data-m], #bento > .desk-empty')].map(t => { const r = t.getBoundingClientRect(); return `${t.dataset.m || 'empty'}@${Math.round(r.left)},${Math.round(r.top + scrollY)},${Math.round(r.width)}x${Math.round(r.height)}`; }).join(' ');
/* empty slots taller than 160px (a short tile) between tiles in any column, above the last tile in that column: a
   column-sized hole, which the factory desks must never leave (Q-08) */
const holes = () => { const g = document.querySelector('#bento'), gr = g.getBoundingClientRect(), cols = getComputedStyle(g).gridTemplateColumns.split(' ').map(parseFloat);
  const ts = [...g.children].filter(t => t.getBoundingClientRect().height > 0).map(t => { const r = t.getBoundingClientRect(); return { l: r.left - gr.left, r: r.right - gr.left, t: r.top - gr.top, b: r.bottom - gr.top }; });
  const out = []; let x = 0; cols.forEach((w, ci) => { const mid = x + w / 2; x += w + 12; let y = 0; ts.filter(t => t.l < mid && t.r > mid).sort((a, b) => a.t - b.t).forEach(t => { if (t.t - y > 160) out.push(`c${ci + 1} ${Math.round(y)}-${Math.round(t.t)}`); y = Math.max(y, t.b + 12); }); });
  return out; };
/* the tiles in DOM (and Tab) order are in reading order on screen: each one lower than the one before, or level with it
   (tops within 8px) and to its right */
const readingOrder = () => { const ts = [...document.querySelectorAll('#bento > .tile[data-m]')].map(t => [t.dataset.m, t.getBoundingClientRect()]); const bad = [];
  for (let i = 1; i < ts.length; i++) { const [a, p] = ts[i - 1], [b, q] = ts[i]; if (q.top < p.top - 8 || (Math.abs(q.top - p.top) <= 8 && q.left < p.left)) bad.push(`${b} before ${a}`); } return bad; };
/* the factory desks in their own tier and at Quick, the empty desk's starters, and Everything */
const FACTORY_DESKS = () => FACTORY.filter(f => f.mods.length).flatMap(f => [[f.name, f.tier, f.mods, f.pedals], ...(f.tier !== 'quick' && f.name !== 'Everything' && f.name !== 'Deep sampler' ? [[f.name + ' at Quick', 'quick', f.mods, f.pedals]] : []), ...(f.name === 'Everything' ? [[f.name + ' at Studio', 'studio', f.mods, f.pedals]] : [])]);
const composes = (w, h) => ({ id: `factory-desks-compose-${w}`, w, h, run: async t => {
  const list = await t.js(FACTORY_DESKS);
  for (const [name, tier, mods, pedals] of [...list, ['starters', 'studio', ['tempo', 'sampler', 'seed'], []], ['starters', 'lab', ['tempo', 'sampler', 'seed'], []], ['empty', 'studio', [], []]]) {
    await desk(tier, mods, pedals)(t);
    await t.eq(`${name} (${tier}): no hole taller than 160px (Q-08)`, holes, []);
    await t.eq(`${name} (${tier}): reading order is the order (Q-01)`, readingOrder, []);
  }
} });
/* a drag by real drag events: start on the tile's header, move over a point, read the preview, drop */
const dragTo = async (t, m, target, fx, fy) => {
  await t.do(`drag ${m}`, m => { const el = document.querySelector(`.tile[data-m="${m}"] [data-drag]`); window.__dt = new DataTransfer(); window.__dsrc = el; el.dispatchEvent(new DragEvent('dragstart', { bubbles: true, cancelable: true, dataTransfer: window.__dt })); }, m, 200);
  await t.do(`over ${target}`, ([target, fx, fy]) => { const tg = document.querySelector(`.tile[data-m="${target}"]`); tg.scrollIntoView({ block: 'center', behavior: 'instant' }); const r = tg.getBoundingClientRect(), x = r.left + r.width * fx, y = r.top + r.height * fy;
    const o = { bubbles: true, cancelable: true, dataTransfer: window.__dt, clientX: x, clientY: y }; const el = document.elementFromPoint(x, y) || tg; el.dispatchEvent(new DragEvent('dragenter', o)); el.dispatchEvent(new DragEvent('dragover', o)); window.__dpt = [el, o, x + scrollX, y + scrollY]; }, [target, fx, fy], 300);
  /* the preview's box, the pointer, and how near the nearest place the tile could take is to it */
  const prev = await t.js(() => { const g = document.querySelector('#bento > .ghost'); if (!g) return null; const r = g.getBoundingClientRect(); const [, , px, py] = window.__dpt;
    const dist = s => Math.hypot(Math.max(s.x - px, 0, px - s.x - s.w), Math.max(s.y - py, 0, py - s.y - s.h));
    return { x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height, px, py, dmin: Math.min(...dragRects.map(dist)), n: dragRects.length }; });
  return prev;
};
const dragScenario = (w, h, tier) => ({ id: `drag-lands-on-its-preview-${tier}-${w}`, w, h, run: async t => {
  await desk(tier, ['tempo', 'avoid', 'seed', 'takes', 'timbre', 'sampler'])(t);
  for (const [m, target, fx, fy] of [['seed', 'tempo', .25, .4], ['takes', 'tempo', .8, .3], ['avoid', 'sampler', .5, .6], ['timbre', 'seed', .3, .5]]) {
    const before = await t.js(() => S.mods.join()), undo0 = await t.js(() => UNDO.length);
    const p = await dragTo(t, m, target, fx, fy);
    await t.ok(`${m} over ${target}: a preview is shown (Q-01)`, () => !!document.querySelector('#bento > .ghost'));
    /* the preview is under the pointer, or, where no place the tile can take in the order is under it, the nearest one */
    await t.ok(`${m} over ${target}: the preview is the place nearest the pointer`, p => { if (!p) return 'no preview'; const d = Math.hypot(Math.max(p.x - p.px, 0, p.px - p.x - p.w), Math.max(p.y - p.py, 0, p.py - p.y - p.h)); return (p.n > 1 && d <= p.dmin + 2) || `preview ${Math.round(d)}px from the pointer, nearest place ${Math.round(p.dmin)}px`; }, p);
    await t.do('drop', () => { const [el, o] = window.__dpt; el.dispatchEvent(new DragEvent('drop', o)); }, null, 600);
    await t.ok(`${m}: the drop lands exactly where the preview stood`, ([m, p]) => { const r = document.querySelector(`.tile[data-m="${m}"]`).getBoundingClientRect(); return (Math.abs(r.left + scrollX - p.x) < 2 && Math.abs(r.top + scrollY - p.y) < 2 && Math.abs(r.width - p.w) < 2) || `landed ${Math.round(r.left + scrollX)},${Math.round(r.top + scrollY)} ${Math.round(r.width)} / preview ${Math.round(p.x)},${Math.round(p.y)} ${Math.round(p.w)}`; }, [m, p]);
    await t.ok(`${m}: a drop that changed the order moved the tile, and only then is a step to undo`, ([before, undo0]) => (S.mods.join() === before) === (UNDO.length === undo0), [before, undo0]);
    await t.eq(`${m}: reading order is the order`, readingOrder, []);
  }
} });
export default [
  composes(1920, 1080), composes(1600, 900), composes(1440, 900), composes(1280, 720),
  dragScenario(1920, 1080, 'studio'), dragScenario(1440, 900, 'studio'), dragScenario(1280, 720, 'lab'), dragScenario(1920, 1080, 'quick'),
  { id: 'layout-is-a-function-of-the-order', w: 1440, h: 900, run: async t => {
    for (const [tier, mods] of [['studio', ACE], ['quick', ACE], ['lab', ['tempo', 'sampler', 'latent', 'seed']]]) {
      await desk(tier, mods)(t);
      const a = await t.js(layout);
      await t.click('[data-nav="Library"]'); await t.click('[data-nav="create"]');
      await t.eq(`${tier}: the same after another page and back (Q-02)`, layout, a);
      await t.do('let autosave run', () => {}, null, 600); await t.reload();
      await t.eq(`${tier}: the same after a reload`, layout, a);
      await t.resize(1920, 1080); await t.resize(1440, 900);
      await t.eq(`${tier}: the same after a resize there and back`, layout, a);
      await t.resize(1280, 720); await t.resize(1440, 900);
      await t.eq(`${tier}: the same after a smaller size and back`, layout, a);
      const other = tier === 'studio' ? 'lab' : 'studio';
      await t.click(`#tiers [data-tier="${other}"]`); await t.click(`#tiers [data-tier="${tier}"]`);
      await t.eq(`${tier}: the same after a tier there and back`, layout, a);
      await t.focus(`.tile[data-m="${mods[1]}"] .grip`); await t.press('ArrowRight');
      const moved = await t.js(layout);
      await t.ok(`${tier}: an arrow key changed it`, ([a, b]) => a !== b, [a, moved]);
      await t.do('undo', () => document.querySelector('#undoBtn').click());
      await t.eq(`${tier}: Undo gives back the same layout`, layout, a);
    }
  } },
  { id: 'arrow-keys-move-one-visible-step', w: 1920, h: 1080, run: async t => {
    await desk('studio', ACE)(t);
    const at = () => { const r = document.querySelector('.tile[data-m="avoid"]').getBoundingClientRect(); return { x: r.left, y: r.top + scrollY, n: UNDO.length, say: document.querySelector('#say').textContent }; };
    await t.focus('.tile[data-m="avoid"] .grip');
    for (const k of ['ArrowRight', 'ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp', 'ArrowUp', 'ArrowUp', 'ArrowLeft', 'ArrowDown']) {
      const a = await t.js(at); await t.press(k); const b = await t.js(at);
      const ok = k === 'ArrowUp' ? b.y < a.y - 4 : k === 'ArrowDown' ? b.y > a.y + 4 : k === 'ArrowLeft' ? (b.y === a.y ? b.x < a.x : b.y < a.y) : (b.y === a.y ? b.x > a.x : b.y > a.y);
      const refused = b.x === a.x && b.y === a.y && /can't move/.test(b.say);
      await t.ok(`${k} from ${Math.round(a.x)},${Math.round(a.y)}: the tile moves that way, or the key is refused and says so (Q-01)`, ([ok, refused, b]) => ok || refused || 'nothing visible happened: ' + JSON.stringify(b), [ok, refused, b]);
      await t.eq(`${k}: focus stays on the grip`, () => document.activeElement?.dataset.move, 'avoid');
      await t.eq(`${k}: reading order is the order`, readingOrder, []);
    }
    await t.ok('a refused key records no move', () => !/can't/.test(UNDO.at(-1)?.label || ''));
  } },
  ...[['studio', 1920, 1080], ['quick', 1920, 1080], ['lab', 1280, 720]].map(([tier, w, h]) => ({ id: `tab-order-is-reading-order-${tier}-${w}`, w, h, run: async t => {
    await desk(tier, tier === 'lab' ? [...ACE, 'latent', 'perf'] : ACE)(t);
    await t.do('focus the prompt', () => document.querySelector('#prompt').focus());
    const seen = [];
    for (let i = 0; i < 160 && seen.length < 8; i++) { await t.page.keyboard.press('Tab'); const m = await t.js(() => document.activeElement.closest('#bento > .tile[data-m]')?.dataset.m); if (m && m !== 'lyrics' && !seen.includes(m)) seen.push(m); }
    await t.eq('Tab visits the tiles in the order they read on screen (Q-01)', () => [...document.querySelectorAll('#bento > .tile[data-m]')].filter(t => t.dataset.m !== 'lyrics').map(t => [t.dataset.m, t.getBoundingClientRect()]).sort((a, b) => Math.abs(a[1].top - b[1].top) > 8 ? a[1].top - b[1].top : a[1].left - b[1].left).map(x => x[0]).slice(0, 8).join(), seen.join());
  } })),
];
