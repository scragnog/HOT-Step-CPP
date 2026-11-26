// Motion (package 9): each motion is set off as a user sets it off, and then, with the page's timeline held still (the
// harness holds it), every animation is moved by hand to a few moments of its run and the page is read there: where a
// box stands, what it shows, how opaque it is. So a motion that is missing, jumps, starts from the wrong place, stretches
// what it carries or moves under reduced motion fails here; tools/motion/frames.mjs
// compares every 16ms of each motion with its approved samples.

const ACE = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes'];
const desk = (tier, mods, pedals = []) => t => t.do(`${tier} desk`, ([tier, mods, pedals]) => { S.tier = tier; S.mods = [...mods]; S.pedals = [...pedals]; pedals.forEach(id => { if (PED[id]?.on) S.vals[PED[id].on] = true; }); S.qAll = true; render(); scrollTo(0, 0); }, [tier, mods, pedals], 400);
/* after an action that does not settle: two frames of the page's clock, and two of the browser's (a ResizeObserver) */
const tick = async (t, n = 2) => { for (let i = 0; i < n; i++) { await t.page.clock.runFor(17); await t.js(() => new Promise(r => { const ro = new ResizeObserver(() => { ro.disconnect(); r(); }); ro.observe(document.documentElement); })); } };
/* every finite animation moved to ms of its run */
const seek = (t, ms) => t.js(ms => { for (const a of document.getAnimations()) { try { const e = a.effect?.getComputedTiming(); if (e && Number.isFinite(e.endTime)) a.currentTime = ms; } catch (e) {} } }, ms);
/* boxes are read in page coordinates: a page that scrolls to what arrived does not move what stands still */
const box = (t, sel) => t.js(sel => { const el = document.querySelector(sel); if (!el) return null; const r = el.getBoundingClientRect(); return { x: Math.round(r.left + scrollX), y: Math.round(r.top + scrollY), w: Math.round(r.width), h: Math.round(r.height), o: Math.round(+getComputedStyle(el).opacity * 100) / 100 }; }, sel);
const boxes = (t, sel, key) => t.js(([sel, key]) => Object.fromEntries([...document.querySelectorAll(sel)].map(el => { const r = el.getBoundingClientRect(); return [el.getAttribute(key), { x: Math.round(r.left + scrollX), y: Math.round(r.top + scrollY), w: Math.round(r.width), h: Math.round(r.height) }]; })), [sel, key]);
/* a tier change's part (shell/morph.ts): its box (a frame's surface, else the element's) and its faces' opacity (the old
   faces are gone before the new tier is drawn: 0), at the moment the animations stand at */
const vt = (t, name) => t.js(n => { const p = window.__morph?.().find(p => p.name === n); if (!p) return null; const e = p.el.getBoundingClientRect(), b = getComputedStyle(p.el, '::before'), px = v => parseFloat(v) || 0, r = p.surface ? { x: e.left + p.el.clientLeft + px(b.left), y: e.top + p.el.clientTop + px(b.top), w: px(b.width), h: px(b.height) } : { x: e.left, y: e.top, w: e.width, h: e.height }; return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.w), h: Math.round(r.h), old: 0, new: Math.round(+getComputedStyle(p.faces[0] ?? p.el).opacity * 100) / 100 }; }, name);
const xy = (a, b) => !!a && !!b && Math.abs(a.x - b.x) <= 2 && Math.abs(a.y - b.y) <= 2;
const at = (a, b, d = 2) => !!a && !!b && ['x', 'y', 'w', 'h'].every(k => Math.abs(a[k] - b[k]) <= d);
const check = (t, label, ok, detail) => t.ok(label, ([ok, d]) => ok || d, [!!ok, JSON.stringify(detail)]);
/* the transitions the page starts, kept for the checks */
const listenVT = t => t.do('listen to the transition', () => { window.__vts = 0; const o = document.startViewTransition?.bind(document); if (o) document.startViewTransition = cb => { window.__vts++; return (window.__vt = o(cb)); }; });
/* what moves while the timeline holds: the properties every running animation's keyframes animate */
const moved = t => t.js(() => [...new Set(document.getAnimations().filter(a => !(a instanceof CSSTransition) && !(a instanceof CSSAnimation) && Number.isFinite(a.effect?.getComputedTiming().endTime) && a.playState !== 'finished').flatMap(a => a.effect.getKeyframes().flatMap(k => Object.keys(k).filter(p => !['offset', 'easing', 'composite', 'computedOffset'].includes(p)))))]);

export default [
  /* the tier change (Q-04, Q-13): each tile is one frame that grows or shrinks in place from its old box to its new one;
     its old face is gone before its new one comes in; its header never fades; the place read stays */
  { id: 'tier-change-frames-grow-in-place', w: 1600, h: 900, run: async t => {
    await desk('quick', ['tempo', 'avoid', 'takes', 'seed'])(t);
    await listenVT(t);
    const before = await box(t, '.tile[data-m="tempo"]');
    await t.js(() => document.querySelector('#tiers [data-tier="studio"]').click());
    await tick(t, 3);
    const steps = [];
    for (const ms of [0, 32, 48, 64, 96, 160, 270]) { await seek(t, ms); steps.push([ms, await vt(t, 'vt-tempo'), await vt(t, 'vt-h-tempo')]); }
    await check(t, 'the tier change moves Tempo as one of its parts', steps[0][1], steps);
    await check(t, 'at its start Tempo\'s frame stands where the tile stood', at(steps[0][1], before), [before, steps[0][1]]);
    const hs = steps.map(s => s[1]?.h ?? 0), grows = hs[hs.length - 1] > hs[0];
    await check(t, 'its frame grows step by step, never back (one object, not a cut)', hs.every((h, i) => !i || (grows ? h >= hs[i - 1] - 0.5 : h <= hs[i - 1] + 0.5)) && new Set(hs).size > 3, hs);
    await check(t, 'the old face is gone by 60ms and the new one has not come in before it (never both)', steps.every(([ms, g]) => !g || (ms >= 64 ? g.old === 0 : true) && (ms < 60 ? g.new === 0 : true)), steps.map(s => [s[0], s[1]?.old, s[1]?.new]));
    await check(t, 'the header is the same object in both tiers: it never fades', steps.every(([, , h]) => !h || (h.new === 1 && h.old === steps[0][2]?.old)), steps.map(s => [s[0], s[2]?.old, s[2]?.new]));
    await t.settle();
    const after = await box(t, '.tile[data-m="tempo"]');
    await check(t, 'at its end the frame stands where the tile now stands', at(steps[steps.length - 1][1], after), [steps[steps.length - 1][1], after]);
    await t.eq('after it, nothing is left drawn for it: no surface, no animation, no View Transition', () => [document.documentElement.classList.contains('tier-morph'), document.querySelectorAll('[data-morph]').length, window.__morph().length, window.__vts], [false, 0, 0, 0]);
  } },
  /* a tile that arrives scales in where it lands; the tiles it pushes slide from where they stood */
  { id: 'a-tile-arrives-and-pushes', run: async t => {
    await desk('studio', ['tempo', 'avoid', 'takes'])(t);
    const was = await boxes(t, '#bento > .tile[data-m]', 'data-m');
    await t.js(() => addItem('seed')); await tick(t);
    await seek(t, 0);
    const s0 = await box(t, '.tile[data-m="seed"]'), now0 = await boxes(t, '#bento > .tile[data-m]', 'data-m');
    await seek(t, 2000);
    const s1 = await box(t, '.tile[data-m="seed"]'), now1 = await boxes(t, '#bento > .tile[data-m]', 'data-m');
    await check(t, 'at the start the new tile is unseen and a little smaller (scale .96)', s0 && s0.o < 0.05 && Math.abs(s0.w - s1.w * 0.96) <= 2, [s0, s1]);
    const pushed = Object.keys(was).filter(m => !xy(was[m], now1[m]));
    await check(t, 'what it pushed starts where it stood, and ends where it now stands', pushed.length > 0 && pushed.every(m => xy(now0[m], was[m])), { pushed, was, now0, now1 });
    await t.settle();
  } },
  /* a tile removed fades where it stood; its neighbours close up by sliding; Undo brings it back the same way */
  { id: 'a-tile-leaves-and-comes-back', run: async t => {
    await desk('studio', ['tempo', 'avoid', 'sampler', 'takes', 'seed'])(t);
    const was = await boxes(t, '#bento > .tile[data-m]', 'data-m');
    await t.js(() => document.querySelector('.tile[data-m="avoid"] .tile-head .x').click()); await tick(t);
    await seek(t, 0);
    const g0 = await box(t, '.exit-layer > .tile[data-m="avoid"]'), now0 = await boxes(t, '#bento > .tile[data-m]', 'data-m');
    await seek(t, 130);
    const g1 = await box(t, '.exit-layer > .tile[data-m="avoid"]');
    await seek(t, 2000);
    const now1 = await boxes(t, '#bento > .tile[data-m]', 'data-m');
    await check(t, 'the tile that left is shown where it stood, then fades out', at(g0, was.avoid) && g0.o > 0.95 && g1 && g1.o < 0.2, [was.avoid, g0, g1]);
    const closed = Object.keys(now1).filter(m => !xy(was[m], now1[m]));
    await check(t, 'its neighbours close up from where they stood', closed.length > 0 && closed.every(m => xy(now0[m], was[m])), { closed, was, now0, now1 });
    await t.settle();
    await t.ok('once it has faded, nothing of it is left', () => !document.querySelector('.exit-layer'));
    await t.js(() => document.querySelector('#undoBtn').click()); await tick(t);
    await seek(t, 0);
    const u0 = await box(t, '.tile[data-m="avoid"]');
    await t.settle();
    const u1 = await box(t, '.tile[data-m="avoid"]');
    await check(t, 'Undo brings it back by its entrance, to where it stood', u0 && u0.o < 0.05 && at(u1, was.avoid), [u0, u1, was.avoid]);
    await t.ok('and focus is on its grip', () => document.activeElement?.matches('.tile[data-m="avoid"] .grip'));
  } },
  /* a tile moved by the keyboard slides to its new place */
  { id: 'a-tile-moved-by-key-slides', run: async t => {
    await desk('studio', ['tempo', 'avoid', 'sampler', 'takes', 'seed'])(t);
    await t.focus('[data-move="seed"]');
    const was = await box(t, '.tile[data-m="seed"]');
    await t.page.keyboard.press('ArrowUp'); await tick(t);
    await seek(t, 0); const m0 = await box(t, '.tile[data-m="seed"]');
    await seek(t, 140); const mid = await box(t, '.tile[data-m="seed"]');
    await seek(t, 2000); const m1 = await box(t, '.tile[data-m="seed"]');
    await check(t, 'it starts where it stood and slides to its new place through the places between', !xy(was, m1) && xy(m0, was) && !xy(mid, was) && !xy(mid, m1), [was, m0, mid, m1]);
    await t.settle();
  } },
  /* a drop lands from its preview's box, a touch larger, and settles into the same box (M-13, Q-15) */
  { id: 'a-drop-lands-from-its-preview', run: async t => {
    await desk('studio', ['tempo', 'avoid', 'timbre', 'sampler', 'seed', 'takes'])(t);
    await t.do('drag Takes over the top of Tempo', () => { const el = document.querySelector('.tile[data-m="takes"] [data-drag]'); window.__dt = new DataTransfer(); el.dispatchEvent(new DragEvent('dragstart', { bubbles: true, cancelable: true, dataTransfer: window.__dt })); }, null, 200);
    await t.do('over', () => { const r = document.querySelector('.tile[data-m="tempo"]').getBoundingClientRect(), x = r.left + r.width * .85, y = r.top + 20; const o = { bubbles: true, cancelable: true, dataTransfer: window.__dt, clientX: x, clientY: y }; const el = document.elementFromPoint(x, y); el.dispatchEvent(new DragEvent('dragenter', o)); el.dispatchEvent(new DragEvent('dragover', o)); window.__dpt = [el, o]; }, null, 300);
    const g = await box(t, '#bento > .ghost');
    await t.js(() => { const [el, o] = window.__dpt; el.dispatchEvent(new DragEvent('drop', o)); }); await tick(t);
    await seek(t, 0); const l0 = await box(t, '.tile[data-m="takes"]');
    await seek(t, 2000); const l1 = await box(t, '.tile[data-m="takes"]');
    await check(t, 'it starts on the preview\'s box, lifted 2% larger', g && l0 && Math.abs((l0.x + l0.w / 2) - (g.x + g.w / 2)) <= 2 && Math.abs((l0.y + l0.h / 2) - (g.y + g.h / 2)) <= 2 && Math.abs(l0.w - g.w * 1.02) <= 2, [g, l0]);
    await check(t, 'and settles into that same box', at(l1, g), [g, l1]);
    await t.settle();
  } },
  /* a pedal that joins drops onto the board, the cables draw out of their jacks, and its LED blooms once it lands */
  { id: 'a-pedal-drops-on-and-its-cables-draw', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['tempo', 'post'], ['norm', 'lift'])(t);
    const was = await boxes(t, '[data-pedals] > .pedal[data-pid]', 'data-pid');
    await t.js(() => addItem('pd:ss')); await tick(t);
    await seek(t, 0); const p0 = await box(t, '[data-pedals] > .pedal[data-pid="ss"]'), others0 = await boxes(t, '[data-pedals] > .pedal[data-pid]', 'data-pid');
    await check(t, 'the pedals already there start where they stood', ['norm', 'lift'].every(id => xy(others0[id], was[id])), [was, others0]);
    await t.ok('a new cable draws out of its jacks (its stroke runs in)', () => [...document.querySelectorAll('.board .cables g[data-k="in>ss"] .cb, .board .cables g[data-k="ss>lift"] .cb')].some(p => p.getAnimations().some(a => a.effect.getKeyframes().some(k => 'strokeDashoffset' in k))));
    await seek(t, 2000); const p1 = await box(t, '[data-pedals] > .pedal[data-pid="ss"]');
    await t.page.clock.runFor(17);
    await t.ok('the cables follow the pedals as they move: once it has landed, its plug is at its jack', () => { const p = document.querySelector('[data-pedals] > .pedal[data-pid="ss"]').getBoundingClientRect(), g = document.querySelector('.board .cables g[data-k="in>ss"] .plug')?.getBoundingClientRect(); return (!!g && Math.abs(g.right - p.left) <= 3 && g.top > p.top && g.bottom < p.bottom) || JSON.stringify([p.left, p.top, p.bottom, g?.right, g?.top]); });
    await check(t, 'it starts unseen, 10px above where it lands', p0 && p1 && p0.o < 0.05 && Math.abs((p0.y + p0.h / 2) - (p1.y + p1.h / 2) + 10) <= 2, [p0, p1]);
    await t.js(() => document.getAnimations().forEach(a => { try { if (Number.isFinite(a.effect.getComputedTiming().endTime)) a.finish(); } catch (e) {} }));
    await t.ok('once it has landed it is rung', () => document.querySelector('[data-pedals] > .pedal[data-pid="ss"]').classList.contains('flash'));
    await t.settle();
  } },
  /* reduced motion (M-08): nothing moves, but every change still shows: what arrives fades in and is rung with the still
     outline, what leaves fades out, and the tier changes with no transition */
  { id: 'reduced-motion-nothing-moves-every-change-shows', reduced: true, run: async t => {
    await desk('studio', ['tempo', 'avoid', 'takes'])(t);
    await listenVT(t);
    await t.js(() => addItem('seed')); await tick(t);
    await t.eq('a tile arrives by fading only', () => [...new Set(document.querySelector('.tile[data-m="seed"]').getAnimations().filter(a => !(a instanceof CSSTransition) && !(a instanceof CSSAnimation)).flatMap(a => a.effect.getKeyframes().flatMap(k => Object.keys(k).filter(p => !['offset', 'easing', 'composite', 'computedOffset'].includes(p)))))], ['opacity']);
    await check(t, 'nothing else moves', !(await moved(t)).some(p => /transform|translate|scale|height|width|clipPath/.test(p)), await moved(t));
    await t.ok('and it is rung with the still outline (S-27)', () => { const tl = document.querySelector('.tile[data-m="seed"]'); return (tl.classList.contains('flash') && getComputedStyle(tl).outlineStyle === 'solid') || `${tl.className} ${getComputedStyle(tl).outlineStyle}`; });
    await t.settle();
    await t.js(() => document.querySelector('.tile[data-m="avoid"] .tile-head .x').click()); await tick(t);
    await check(t, 'a tile leaves by fading only', !(await moved(t)).some(p => /transform|translate|scale|height|width|clipPath/.test(p)) && (await moved(t)).includes('opacity'), await moved(t));
    await t.settle();
    await t.press('Alt+3');
    await t.eq('the tier changes with no transition', () => [S.tier, window.__vts], ['lab', 0]);
  } },
  /* a re-pack that a tile's content sets off slides the tiles it moves (the concept's packDesk) */
  { id: 'a-tile-that-grows-moves-the-rest-smoothly', run: async t => {
    await desk('studio', ['tempo', 'seed', 'avoid', 'takes'])(t);
    await t.focus('[data-bind="lyrics.text"]');
    await t.page.keyboard.insertText(Array.from({ length: 14 }, (_, i) => `line ${i}`).join(String.fromCharCode(10))); await tick(t, 4);
    await t.ok('the tiles it moves slide over --t-base', () => [...document.querySelectorAll('#bento > .tile[data-m]')].some(x => x.getAnimations().some(a => a.effect.getTiming().duration === 200 && a.effect.getKeyframes().some(k => /translate/.test(k.transform || '')))) || 'no tile slides');
    await t.settle();
  } },
  /* what pushes the content under it moves it smoothly: the prompt box shrinking on Undo, a suggestion opening, Song
     info opening (F-09, F-28) */
  { id: 'what-the-prompt-card-opens-moves-what-follows-smoothly', run: async t => {
    await desk('quick', ['tempo', 'avoid', 'takes', 'seed'])(t);
    await t.focus('#prompt'); await t.type('a\nb\nc\nd\ne\nf');
    await t.do('leave the box', () => { document.activeElement.blur(); scrollTo(0, 0); });
    const grown = await box(t, '.pc-foot');
    await t.js(() => document.querySelector('#undoBtn').click()); await tick(t);
    await seek(t, 0); const u0 = await box(t, '.pc-foot');
    await seek(t, 2000); const u1 = await box(t, '.pc-foot');
    await check(t, 'Undo shrinks the box: its foot starts where it stood and rises to its place', at(u0, grown) && u1.y < grown.y - 40, [grown, u0, u1]);
    await t.settle();
    await t.focus('#prompt');
    const f0 = await box(t, '.pc-foot');
    await t.page.keyboard.insertText('lofi beat at 92 bpm'); await tick(t);
    await seek(t, 0); const s0 = await box(t, '.pc-foot');
    await seek(t, 2000); const s1 = await box(t, '.pc-foot');
    await check(t, 'a suggestion opens: what is under it starts where it stood and moves down', Math.abs(s0.y - f0.y) <= 12 && s1.y > f0.y + 20, [f0, s0, s1]);
    await t.settle();
    await t.page.keyboard.press('Control+A'); await t.page.keyboard.press('Delete'); await tick(t);
    await seek(t, 0); const c0 = await box(t, '.pc-foot');
    await seek(t, 2000); const c1 = await box(t, '.pc-foot');
    await check(t, 'the suggestion closes: what is under it starts where it stood and rises', Math.abs(c0.y - s1.y) <= 12 && c1.y < s1.y - 20, [s1, c0, c1]);
    await t.settle();
    await t.do('leave the box', () => { document.activeElement.blur(); scrollTo(0, 0); });
    const d0 = await box(t, '.tile[data-m="tempo"]');
    await t.js(() => document.querySelector('#songInfoBtn').click()); await tick(t);
    await seek(t, 0); const i0 = await box(t, '.tile[data-m="tempo"]');
    await seek(t, 2000); const i1 = await box(t, '.tile[data-m="tempo"]');
    await check(t, 'Song info opens: the desk under it starts where it stood and moves down', Math.abs(i0.y - d0.y) <= 14 && i1.y > d0.y + 40, [d0, i0, i1]);
    await t.settle();
  } },
  /* the small motions: an LED switch turned on kicks (and is not kicked by a redraw); a stamp chosen lands with a thunk;
     a side panel folded fades in its rail; Generate's label cross-fades when its words change */
  { id: 'the-small-motions-of-a-change', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['tempo', 'planner', 'lyrics'])(t);
    await t.do('the planner off', () => setVal('lm.on', false));
    await t.js(() => document.querySelector('.tile[data-m="planner"] [data-ledtog="lm.on"]').click()); await tick(t, 1);
    await t.ok('an LED switch turned on kicks its glow', () => document.querySelector('.tile[data-m="planner"] [data-ledtog="lm.on"]').classList.contains('kick'));
    await t.settle(600);
    await t.js(() => document.querySelector('#tiers [data-tier="studio"]').click()); await t.settle();
    await t.ok('a redraw kicks nothing', () => !document.querySelector('.kick'));
    await t.js(() => document.querySelector('.stamp-wrap [data-pick="lyrics.lang"]').click()); await tick(t, 1);
    await t.js(() => document.querySelector('#dd .opt-stamp:not([aria-selected="true"]):not([hidden])').click()); await tick(t, 2);
    await t.ok('a stamp chosen lands with a thunk', () => !!document.querySelector('.stamp-wrap.thunk'));
    await t.settle();
    await t.js(() => document.querySelector('[data-collapse="left"]').click()); await tick(t, 2);
    await t.ok('a side panel folded fades its rail in', () => [...document.querySelectorAll('#leftPanel > .rail-btn')].some(b => b.getAnimations().some(a => a.effect.getKeyframes().some(k => 'opacity' in k))) || 'no fade');
    await t.settle();
    await t.js(() => { S.source = { song: S.songs[0].id }; S.mods = [...S.mods, 'repaint']; render(); }); await tick(t, 2);
    await t.ok('Generate\'s label cross-fades to its new words', () => /Repaint/.test(document.querySelector('#goLabel').textContent) && document.querySelector('#goLabel').getAnimations().some(a => a.effect.getKeyframes().some(k => 'opacity' in k)));
    await t.settle();
  } },
  /* what the tier brings is rung when a link reveals it, and what is added is brought into view */
  { id: 'what-arrives-is-shown', w: 1600, h: 900, run: async t => {
    await desk('studio', ['tempo', 'sampler', 'guidance', 'seed'])(t);
    await t.js(() => document.querySelector('#hiddenNote [data-tier="lab"]').click()); await tick(t, 3);
    await t.ok('Show in Lab rings the module it reveals', () => document.querySelector('.tile[data-m="guidance"]').classList.contains('flash'));
    await desk('studio', ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'takes'])(t);
    await t.do('add Seed', () => addItem('seed'), null, 400);
    await t.ok('a tile added below the fold is brought into view', () => { const r = document.querySelector('.tile[data-m="seed"]').getBoundingClientRect(); return (r.top >= 0 && r.bottom <= innerHeight + 1) || `${Math.round(r.top)}..${Math.round(r.bottom)} of ${innerHeight}`; });
  } },
  /* Quick's fold opening and folding again: the tiles it shows arrive as an add does, those it folds leave where they
     stood, and what moves slides (the concept's qall click, transition(render)); the concept's two stand-ins make the
     desk long enough to fold, as its suite does */
  { id: 'the-fold-opens-and-folds-again-by-moving', w: 1600, h: 900, run: async t => {
    await t.do('stand-ins on a Quick desk', () => {
      ['tx1', 'tx2'].forEach((id, i) => { defineModule(id, { at: 300 + i, name: 'Stand-in ' + (i + 1), cat: 'Engine', w: 1, keys: [] }); defineFace(id, () => '<p>stand-in</p>'); });
      S.tier = 'quick'; S.backend = 'ace'; S.mods = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes', 'tx1', 'tx2', 'post']; S.pedals = ['lrc']; S.qAll = false; render(); scrollTo(0, 0);
    }, null, 400);
    const was = await boxes(t, '#bento > .tile[data-m]', 'data-m');
    await t.js(() => document.querySelector('#bento > [data-qall]').click()); await tick(t);
    await seek(t, 0);
    const s0 = await box(t, '.tile[data-m="seed"]'), now0 = await boxes(t, '#bento > .tile[data-m]', 'data-m');
    await seek(t, 2000);
    const s1 = await box(t, '.tile[data-m="seed"]'), now1 = await boxes(t, '#bento > .tile[data-m]', 'data-m');
    await check(t, 'opened: a tile it shows starts unseen and a little smaller, as an arrival does', s0 && s1 && s0.o < 0.05 && Math.abs(s0.w - s1.w * 0.96) <= 2, [s0, s1]);
    const moved = Object.keys(was).filter(m => now1[m] && !xy(was[m], now1[m]));
    await check(t, 'what it moves starts where it stood', moved.every(m => xy(now0[m], was[m])), { moved, was, now0, now1 });
    await t.settle();
    const open = await boxes(t, '#bento > .tile[data-m]', 'data-m');
    await t.js(() => document.querySelector('#bento > [data-qall]').click()); await tick(t);
    await seek(t, 0);
    const g0 = await box(t, '.exit-layer > .tile[data-m="seed"]');
    await check(t, 'folded again: a tile it hides is shown where it stood as it fades', at(g0, open.seed) && g0.o > 0.95, [open.seed, g0]);
    await t.settle();
    await t.ok('and nothing of it is left', () => !document.querySelector('.exit-layer') && !document.querySelector('#bento > .tile[data-m="seed"]'));
  } },
  /* the decorative loops rest when nothing has happened for a while, and pick up on any input */
  { id: 'the-loops-rest-when-nothing-happens', run: async t => {
    await desk('studio', ['takes', 'tempo'])(t);
    await t.page.clock.runFor(13000);
    await t.ok('after 12 seconds with nothing happening, the page is calm', () => document.body.classList.contains('calm'));
    await t.press('Shift');
    await t.ok('a key wakes it', () => !document.body.classList.contains('calm'));
  } },
  /* the player joins the go bar: what stands beside it slides over from where it stood */
  { id: 'the-player-slides-in-beside-the-line', run: async t => {
    await desk('studio', ['tempo'])(t);
    const m0 = await box(t, '#meter');
    await t.js(() => document.querySelector('#latest .song [data-play]').click()); await tick(t);
    await seek(t, 0); const a0 = await box(t, '#meter');
    await seek(t, 2000); const a1 = await box(t, '#meter');
    /* (the concept's player keeps its gap from the first frame) */
    await check(t, 'the go bar\'s line starts where it stood and moves over for the player', a1.x > m0.x + 40 && a0.x - m0.x < (a1.x - m0.x) / 4, [m0, a0, a1]);
    await t.settle();
  } },
];
