// What survives a reload, how saved state is cleaned on the way back in, and the pedalboard's cables (every cable's
// ends on its pedals' jacks, none behind a pedal) through adds, removals, bypass, resizes, tiers and engines.
const cables = async (t, label) => t.ok('cables: ' + label, () => {
  const svg = document.querySelector('.board .cables'); if (!svg) return 'no board';
  const sr = svg.getBoundingClientRect(), P = {}; document.querySelectorAll('[data-pedals] > .pedal').forEach(p => { const r = p.getBoundingClientRect(); P[p.dataset.pid] = { l: r.left - sr.left, r: r.right - sr.left, t: r.top - sr.top, b: r.bottom - sr.top, j: r.top - sr.top + parseFloat(getComputedStyle(p, '::after').top) + 6 }; }); /* j: the jack as drawn (L-17: one height above the shelf) */
  const bad = [], ids = Object.keys(P); let n = 0;
  [...svg.querySelectorAll('g.lk:not(.byp)')].forEach(g => { if (g._gone) return; const [a, b] = g.dataset.k.split('>'), pc = g.querySelector('path.cb'), p0 = pc.getPointAtLength(0), p1 = pc.getPointAtLength(pc.getTotalLength()); n++;
    if (a !== 'in') { const p = P[a]; if (!p || Math.abs(p0.x - (p.r + 10)) > 1 || Math.abs(p0.y - p.j) > 1) bad.push(g.dataset.k + ' start'); }
    if (b !== 'out') { const p = P[b]; if (!p || Math.abs(p1.x - (p.l - 10)) > 1 || Math.abs(p1.y - p.j) > 1) bad.push(g.dataset.k + ' end'); }
    const L = pc.getTotalLength(); for (let s = 2; s < L - 2; s += 6) { const pt = pc.getPointAtLength(s); for (const id of ids) { const p = P[id]; if (pt.x > p.l + 1 && pt.x < p.r - 1 && pt.y > p.t + 1 && pt.y < p.b - 1) { bad.push(g.dataset.k + ' crosses ' + id); s = L; break; } } } });
  const want = ids.length ? ids.length + 1 : 0;
  return { ok: !bad.length && n === want, links: n, want, bad };
});
export default [
  { id: 'reload-keeps-the-desk', run: async t => {
    await t.do('a full desk', () => { S.tier = 'lab'; S.backend = 'ace'; S.mods = ['repaint', 'tempo', 'timbre', 'adapters', 'seed']; S.source = { song: 2 }; S.region = [0.2, 0.4]; S.pedals = ['norm', 'wh']; S.vals['samp.steps'] = 33; S.vals['tempo.bpm'] = 101; S.vals['tempo.bpm.auto'] = false; S.slots = [{ file: 'folk-voice', type: 'LoKr' }]; S.seed = 555; S.timbreFile = 'ref.wav'; S.masterRef = 'm.wav'; S.latent = 'a.latent'; S.qAll = true; S.songInfoOpen = true; S.stars.push({ v: 9, n: 'nine', d: '', tags: [], star: false }); S.refs.unshift('new.wav'); S.left = false; S.wfOpen.add('Lo-fi tape'); document.querySelector('#prompt').value = 'kept prompt'; render(); });
    await t.do('let autosave run', () => {}, null, 600);
    const before = await t.js(() => JSON.stringify([S.tier, S.backend, S.mods, S.source, S.region, S.pedals, S.vals, S.slots, S.seed, S.timbreFile, S.masterRef, S.latent, S.qAll, S.songInfoOpen, S.stars.length, S.refs[0], S.left, [...S.wfOpen], document.querySelector('#prompt').value]));
    await t.reload();
    await t.eq('every part of the desk comes back', () => JSON.stringify([S.tier, S.backend, S.mods, S.source, S.region, S.pedals, S.vals, S.slots, S.seed, S.timbreFile, S.masterRef, S.latent, S.qAll, S.songInfoOpen, S.stars.length, S.refs[0], S.left, [...S.wfOpen], document.querySelector('#prompt').value]), before);
    await t.ok('and is drawn', () => !!document.querySelector('#bento .tile[data-m="repaint"]') && document.querySelector('#goLabel').textContent === 'Repaint region');
  } },
  { id: 'reload-songs', run: async t => {
    await t.do('a run', () => { document.querySelector('#prompt').value = 'saved song'; generate(); }, null, 200);
    await t.do('pagehide mid-run', () => dispatchEvent(new Event('pagehide')));
    await t.reload();
    await t.ok('a song still rendering is not saved', () => !S.songs.some(s => s.pending) && S.songs.length === 6);
    await t.do('finish one', () => { document.querySelector('#prompt').value = 'saved song'; generate(); }, null, 4000);
    await t.do('hidden tab saves', () => { Object.defineProperty(document, 'visibilityState', { value: 'hidden', configurable: true }); document.dispatchEvent(new Event('visibilitychange')); });
    await t.reload();
    await t.ok('a finished one is', () => S.songs[0].title === 'Saved song' && S.songs.length === 7 && S.nextId > 100);
  } },
  { id: 'restore-cleans', storage: { 'hs3-desk': JSON.stringify({ v: 1, desk: { tier: 'nope', backend: 'xx', mods: ['tempo', 'tempo', 'ghost', 'repaint'], pedals: ['norm', 'zz'], vals: { 'samp.steps': 'many', 'tempo.bpm': 999, 'tempo.key': 'H major', 'seed.lock': 1, 'tempo.bpm.auto': false, 'nope.k': 3 }, slots: [{ file: 'a', type: 'LoRA' }, 'x'], seed: -4, source: { song: 999 }, region: [2, 3], workflow: 'Not a workflow', prompt: 'p' }, songs: [{ id: 1, title: 'ok', style: 's', dur: 100, hue: 1, bk: 'ace' }, { id: 'bad' }], stars: [{ v: 3, n: 'x' }, { v: -1 }], refs: ['a.wav', 5], left: false, wfOpen: ['Songwriter', 7] }) }, run: async t => {
    await t.eq('unknown values fall back, bad ones are dropped', () => [S.tier, S.backend, S.mods, S.pedals, S.vals, S.slots.length, S.seed, S.source, S.region, S.workflow, document.querySelector('#prompt').value],
      ['quick', 'ace', ['tempo', 'post'], ['norm'], { 'tempo.bpm': 240, 'tempo.bpm.auto': false }, 1, null, null, [0.35, 0.55], 'Start fresh', 'p']);
    await t.eq('songs, seeds, references and panels too', () => [S.songs.map(s => s.id), S.stars.map(s => s.v), S.refs, S.left, [...S.wfOpen]], [[1], [3], ['a.wav'], false, ['Songwriter']]);
  } },
  { id: 'old-shape-ignored', storage: { 'hs3-desk': JSON.stringify({ v: 0, desk: { tier: 'lab' } }) }, run: async t => {
    /* F-16: an older shape is a desk this page can't use, so it is kept aside and the page says so like any other */
    await t.eq('a desk saved in an older shape starts clean, kept aside, and says so', () => [S.tier, localStorage.getItem('hs3-desk-unreadable'), /couldn't be read.*A copy is kept/.test(document.querySelector('#toast .toast-msg')?.textContent || ''), document.querySelector('#toast .toast-act')?.textContent], ['quick', JSON.stringify({ v: 0, desk: { tier: 'lab' } }), true, 'Download the copy']);
  } },
  { id: 'cables', run: async t => {
    await t.do('lab ten pedals', () => { S.tier = 'lab'; S.mods = ['post']; S.pedals = ['tile', 'refiner', 'ss', 'ppvae', 'lift', 'nat', 'gain', 'vst', 'master', 'norm']; render(); }, null, 900);
    await cables(t, 'lab, ten pedals at 1920');
    await t.do('add side unit', () => addItem('ex:wh'), null, 900); await cables(t, 'after adding a side unit');
    await t.click('[data-pedalx="ss"]', 900); await cables(t, 'after removing StableStep');
    await t.do('add ss back', () => addItem('pd:ss'), null, 900); await cables(t, 'after adding it back');
    await t.click('[data-foot="pp.nat.on"]', 900); await cables(t, 'after a foot switch');
    await t.click('.master[data-ledtog="pp.on"]', 900); await cables(t, 'chain bypassed');
    await t.click('.master[data-ledtog="pp.on"]', 900); await cables(t, 'chain engaged');
    for (const w of [1600, 1536, 1440, 1280, 1024, 390]) { await t.resize(w, 900, 900); await cables(t, 'at ' + w); }
    await t.resize(1920, 1080, 900);
    for (const tier of ['studio', 'quick', 'lab']) { await t.click(`#tiers [data-tier="${tier}"]`, 900); await cables(t, 'tier ' + tier); }
    for (const bk of ['yue', 'mm3', 'sa3', 'hm', 'ace']) { await t.do('engine ' + bk, bk => { S.backend = bk; render(); }, bk, 900); await cables(t, 'engine ' + bk); }
    await t.do('one pedal', () => { S.pedals = ['norm']; render(); }, null, 900); await cables(t, 'one pedal');
    await t.do('left collapsed', () => { S.left = false; render(); }, null, 900); await cables(t, 'with the left panel collapsed');
  } },
];
