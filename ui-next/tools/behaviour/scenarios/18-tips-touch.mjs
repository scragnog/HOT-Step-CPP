// Round 5, tooltips and the changed mark: a tip says every word and digit of its source, the quiet after a press delays a tip rather than
// cancelling it, a held press is never thrown away, and the plugin menus can be read under a finger.
const touchAt = async (t, sel, holdMs) => {
  const r = await t.js(sel => { const el = document.querySelector(sel); el.scrollIntoView({ block: 'center', behavior: 'instant' }); const b = el.getBoundingClientRect(); return { x: b.left + b.width / 2, y: b.top + b.height / 2 }; }, sel);
  const cdp = await t.page.context().newCDPSession(t.page);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: r.x, y: r.y }] });
  await t.settle(holdMs);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await t.settle(300); await cdp.detach();
};
export default [
  { id: 'tips-keep-every-digit', run: async t => {
    await t.eq('the sentence splitter keeps decimals and versions (F-01, Q-01, E-04)', () => [tipSentences('Temperature 1.05'), tipSentences('Full bar: the 31.8 GB card. Click for the engine log.'), tipSentences('ACE-Step 1.5. Drag sideways.')],
      [['Temperature 1.05'], ['Full bar: the 31.8 GB card.', 'Click for the engine log.'], ['ACE-Step 1.5.', 'Drag sideways.']]);
    await t.hover('#engMeter', 900);
    await t.ok('the VRAM meter\'s tip says 31.8 GB', () => /the 31\.8 GB card/.test(document.querySelector('#tip')?.textContent || '') || document.querySelector('#tip')?.textContent);
    await t.do('every module on the desk, in Lab', () => { S.tier = 'lab'; S.mods = Object.keys(MODS).filter(m => !MODS[m].source && modOk(m)); S.pedals = PEDALS.filter(p => !p.only || p.only.includes(S.backend)).map(p => p.id); render(); });
    await t.ok('every title on the page splits into its sentences with nothing lost', () => { const bad = [];
      document.querySelectorAll('[title], [data-tip-held]').forEach(el => { const src = (el.title || el.dataset.tipHeld || '').replace(/\s+/g, ' ').trim(); if (src && tipSentences(src).join(' ') !== src) bad.push(src); });
      return bad.length ? bad.slice(0, 3) : true; });
  } },
  { id: 'tip-waits-out-the-quiet-after-a-press', w: 1440, run: async t => {
    await t.do('lab sampler', () => { S.tier = 'lab'; S.mods = ['sampler', 'takes']; render(); });
    await t.click('.tile[data-m="takes"] .tile-head');
    await t.hover('.tile[data-m="sampler"] [data-auto="samp.shift"]', 300);
    await t.ok('no tip yet, inside the quiet after the press', () => !document.querySelector('#tip') || document.querySelector('#tip').hidden);
    await t.do('rest', () => {}, null, 2000);
    await t.ok('the tip opens once the quiet is over, the pointer still resting (L-06)', () => !document.querySelector('#tip')?.hidden && document.querySelector('#tip').textContent.length > 3);
  } },
  { id: 'a-held-press-still-presses', w: 820, h: 1180, touch: true, run: async t => {
    await t.do('a description', () => { const p = document.querySelector('#prompt'); p.value = 'warm tape hiss'; p.dispatchEvent(new Event('input', { bubbles: true })); });
    await touchAt(t, '#go', 700);
    await t.ok('Generate held for 0.7 s under a finger queues a song (S-01)', () => S.songs.some(s => s.pending) || S.batches?.length > 0);
    await touchAt(t, '#tiers [data-tier="studio"]', 800);
    await t.eq('a tier held for 0.8 s is chosen', () => S.tier, 'studio');
  } },
  { id: 'short-plugin-menu-is-named-cards', engine: 'sa3', run: async t => {
    await t.do('sa3 sampler', () => { S.backend = 'sa3'; S.tier = 'studio'; S.mods = ['sampler']; render(); });
    await t.click('.tile[data-m="sampler"] [data-pick="sa3.sampler"]');
    await t.ok('three samplers are cards with their names, not bare symbols (S-02)', () => { const o = [...document.querySelectorAll('#dd .opt')]; return o.length === 3 && o.every(b => b.classList.contains('opt-el') && b.querySelector('b')?.textContent.trim().length > 2) || o.map(b => b.className); });
  } },
  { id: 'the-changed-mark-moves-nothing', run: async t => {
    for (const [bk, tier] of [['ace', 'quick'], ['ace', 'studio'], ['ace', 'lab'], ['mm3', 'lab'], ['yue', 'lab']]) {
      await t.ok(`marking every control changed moves nothing on ${bk} ${tier}`, ([bk, tier]) => {
        S.backend = bk; S.tier = tier; S.mods = Object.keys(MODS).filter(m => modOk(m) && !MODS[m].source); S.pedals = PEDALS.filter(p => !p.only || p.only.includes(bk)).map(p => p.id); render(); window.__bh.finish();
        const els = [...document.querySelectorAll('#bento *, .lyr-card *')], box = () => els.map(e => { const q = e.getBoundingClientRect(); return [q.left, q.top, q.width, q.height]; });
        const a = box(), ctl = [...document.querySelectorAll('[data-ctl]:not(.changed)')]; ctl.forEach(e => e.classList.add('changed')); const b = box(); ctl.forEach(e => e.classList.remove('changed'));
        const moved = els.filter((e, i) => a[i].some((v, j) => Math.abs(v - b[i][j]) > 0.6)).map(e => (e.closest('[data-ctl]')?.dataset.ctl || '?') + ' ' + e.className);
        return moved.length ? moved.slice(0, 4) : true; }, [bk, tier]);
    }
  } },
  ...[[1600, 'studio'], [1920, 'quick'], [1280, 'quick'], [1000, 'lab']].map(([w, tier]) => ({ id: `desk-bar-keeps-clear-${w}-${tier}`, w, run: async t => {
    await t.do('a desk with the planner', tier => { S.tier = tier; S.mods = ['takes', 'planner']; render(); }, tier);
    /* two steps with long names, the second undone: a long Undo label and a Redo */
    await t.do('a step', () => setVal('lm.temp', 1.85));
    await t.do('another', () => setVal('lm.topp', 0.95));
    await t.click('#undoBtn');
    await t.ok('Undo and Redo both stand in the bar', () => !document.querySelector('#undoBtn').hidden && !document.querySelector('#redoBtn').hidden);
    await t.ok('Saved, Undo and Redo stay 16 px clear of the tier switch (F-02)', () => { const t = document.querySelector('#tiers').getBoundingClientRect();
      const hit = ['#saveState', '#undoBtn', '#redoBtn'].map(s => document.querySelector(s)).filter(e => e && !e.hidden).map(e => [e.id, e.getBoundingClientRect()]).filter(([, r]) => r.top < t.bottom && r.bottom > t.top && r.left < t.right + 15.5 && r.right > t.left - 15.5);
      return hit.length ? hit.map(([id, r]) => `${id} ${Math.round(r.left)}-${Math.round(r.right)} vs tiers ${Math.round(t.left)}-${Math.round(t.right)}`) : true; });
    await t.ok('and the Undo arrow keeps its size', () => document.querySelector('#undoBtn svg').getBoundingClientRect().width >= 12.5);
  } })),
];
