// Round 4, the Lab tier on ACE-Step (review L): a bypassed pedal stays bypassed under the pointer and with focus; one
// changed mark and one focus ring; a setting that makes another control irrelevant shows it at once; typed values; the
// tooltip never covers what the control reports; the adapters tile takes the room it needs.
const ALL = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'repetition', 'sampler', 'guidance', 'perf', 'latent', 'seed', 'takes'];
const desk = (tier, mods, pedals = [], vals = {}) => t => t.do(`${tier} desk`, ([tier, mods, pedals, vals]) => { S.tier = tier; S.mods = [...mods]; S.pedals = [...pedals]; Object.assign(S.vals, vals); render(); }, [tier, mods, pedals, vals]);
export default [
  { id: 'bypassed-pedal-stays-bypassed', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['post'], ['refiner', 'ss'], { 'pp.ss.on': true, 'pp.refiner.on': true })(t);
    await t.click('.pedal[data-pid="ss"] .foot');
    await t.ok('pressing the switch bypasses it while the pointer is still on the pedal (L-02)', () => { const p = document.querySelector('.pedal[data-pid="ss"]'); return p.classList.contains('off') && /saturate\(0\.25\) brightness\(0\.74\)/.test(getComputedStyle(p).filter) || getComputedStyle(p).filter; });
    await t.press('Shift+Tab');
    await t.ok('with focus inside, the face and the other controls stay bypassed (L-02)', () => { const p = document.querySelector('.pedal[data-pid="ss"]'); return /saturate/.test(getComputedStyle(p, '::before').backdropFilter) && [...p.querySelectorAll('.pknobs > .kn')].slice(1).every(k => /saturate/.test(getComputedStyle(k).filter)); });
    await t.ok('every pedal is counted, and the side units (L-34)', () => /^2 pedals · 1 on/.test(document.querySelector('[data-ppcount]').textContent) || document.querySelector('[data-ppcount]').textContent);
  } },
  { id: 'stablestep-knobs-hold-their-cells', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['post'], ['ss'], { 'pp.ss.on': true })(t);
    const x = () => [...document.querySelectorAll('.pedal[data-pid="ss"] .pknobs > .kn:not(.kn-empty)')].slice(0, 3).map(k => Math.round(k.getBoundingClientRect().left));
    const x0 = await t.js(x);
    await t.do('source blend: crossover', () => { setVal('pp.ss.blend', 'Crossover'); });
    await t.eq('stepping Source blend never moves Refine, Vocal and Steps (L-16)', x, x0);
    await t.do('fixed seed', () => { setVal('pp.ss.seedf', false); });
    await t.focus('.pedal[data-pid="ss"] [data-bind="pp.ss.seed"]');
    await t.do('type words', () => { const el = document.querySelector('.pedal[data-pid="ss"] [data-bind="pp.ss.seed"]'); el.value = 'not a seed 999999999999'; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); });
    await t.eq('the fixed seed takes only a seed (L-14)', () => get('pp.ss.seed'), '4242');
  } },
  { id: 'off-settings-show-at-once', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['planner', 'repetition', 'sampler', 'perf', 'timbre'], ['master'], { 'perf.cfgcut': 0.45 })(t);
    await t.ok('the Sampler says where CFG cutoff ends guidance (L-12)', () => /guidance until 0\.45 \(CFG cutoff\)/.test(document.querySelector('[data-scoperead]').textContent) || document.querySelector('[data-scoperead]').textContent);
    await t.click('.tile[data-m="planner"] [data-ledtog="lm.on"]');
    await t.ok('planner off: the Repetition guard says so where it says guard off (L-11)', () => { const s = document.querySelector('.tile[data-m="repetition"] .term-plan-off'); return !!s && getComputedStyle(s).display !== 'none'; });
    await t.ok('the blend with no track stands back and says why (L-19)', () => document.querySelector('.venn').classList.contains('is-off') && /Load a track/.test(document.querySelector('.venn-cap').textContent));
    await t.do('a mastering reference, lent as the timbre reference', () => { S.masterRef = 'master-ref.wav'; S.vals['pp.master.timbre'] = true; render(); });
    await t.ok('Timbre shows the Mastering track as its own, marked (L-15)', () => /master-ref\.wav/.test(document.querySelector('.tile[data-m="timbre"] .sl-txt').textContent) && /from Mastering/.test(document.querySelector('.tile[data-m="timbre"] .sl-txt').textContent) && !document.querySelector('.venn').classList.contains('is-off'));
    await t.do('without the Timbre module', () => { S.mods = S.mods.filter(m => m !== 'timbre'); render(); });
    await t.ok('the pedal says Timbre is not on the desk and offers it (L-15)', () => !!document.querySelector('.pedal[data-pid="master"] [data-additem="timbre"]'));
  } },
  { id: 'instrumental-stands-everything-back', w: 1920, h: 1080, run: async t => {
    await desk('lab', [], [], { 'lyrics.text': '[Verse]\na\nb' })(t);
    await t.click('.tile[data-m="lyrics"] [data-ledtog="lyrics.instr"]');
    /* round 5 (Q-27): Instrumental is said by its stamp and its lit pill, not a fourth time in the stat; the language stamp stands back too */
    await t.ok('the stat says nothing more, the tabs, Expand and the language stamp stand back and say so (L-22, Q-27)', () => document.querySelector('[data-lstat]').textContent === '' && document.querySelector('.paper-wrap .stamp-wrap button').disabled && /^Lyrics are ignored$/.test(document.querySelector('[data-strip]').textContent) && [...document.querySelectorAll('.paper-wrap .itabs button')].every(b => b.getAttribute('aria-disabled') === 'true') && document.querySelector('.paper-wrap .wild-x').disabled);
  } },
  { id: 'timesteps-keep-what-was-typed', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['latent', 'sampler'])(t);
    await t.click('.tile[data-m="latent"] [data-bind="lat.ts"]'); await t.type('0.5, 0.9, abc'); await t.press('Tab');
    await t.ok('a typo keeps the text, marked, with the reason beside it (L-13)', () => { const el = document.querySelector('[data-bind="lat.ts"]'); return el.value === '0.5, 0.9, abc' && el.getAttribute('aria-invalid') === 'true' && /numbers from 0 to 1/.test(document.querySelector('[data-err="lat.ts"]').textContent) && get('lat.ts') === ''; });
    await t.do('one timestep', () => { const el = document.querySelector('[data-bind="lat.ts"]'); el.value = '0.9'; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); });
    await t.ok('one value is not a schedule (L-13)', () => get('lat.ts') === '' && /at least two/.test(document.querySelector('[data-err="lat.ts"]').textContent));
  } },
  { id: 'takes-keyboard-and-reset', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['takes'])(t);
    await t.hover('.reel2[data-take="5"]');
    await t.focus('[data-takes]'); await t.press('End');
    await t.ok('a key ends the pointer preview: the LCD shows the value set (L-32)', () => document.querySelector('.tk-lcd b').textContent === '8' && document.querySelector('.tk-reels').classList.contains('kbd'));
    await t.dblclick('.reel2[data-take="3"]');
    await t.eq('a double-click on a reel is a click on it: the reels are the values, not a knob to reset (round 5 L-22, replaces L-32)', () => +get('takes.n'), 3);
  } },
  { id: 'knob-value-can-be-typed', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['sampler'])(t);
    await t.click('.tile[data-m="sampler"] .kn:has([data-knob="samp.cfg"]) > output');
    await t.ok('a click on the read-out opens a field (L-40)', () => document.activeElement?.classList.contains('kn-in'));
    await t.press('Control+a'); await t.type('7.3'); await t.press('Enter');
    await t.eq('Enter sets the typed value', () => +get('samp.cfg'), 7.3);
    await t.focus('.tile[data-m="sampler"] [data-knob="samp.cfg"]'); await t.press('Enter'); await t.type('99'); await t.press('Escape');
    await t.eq('Escape puts it back', () => +get('samp.cfg'), 7.3);
  } },
  { id: 'repaint-home-end', w: 1920, h: 1080, run: async t => {
    await t.do('lab desk with a repaint', () => { S.tier = 'lab'; S.source = S.songs[0]?.id ?? S.source; S.mods = ['repaint']; S.region = [0.3, 0.5]; render(); });
    await t.focus('[data-rpwave]'); await t.press('Home');
    await t.ok('Home moves the region to the start, its length kept (L-36)', () => Math.abs(S.region[0]) < 1e-9 && Math.abs(S.region[1] - 0.2) < 1e-6 || S.region);
    await t.press('End');
    await t.ok('End to the end', () => Math.abs(S.region[1] - 1) < 1e-9 && Math.abs(S.region[0] - 0.8) < 1e-6 || S.region);
  } },
  { id: 'tip-keeps-off-the-read-out', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['takes', 'sampler'])(t);
    await t.hover('.reel2[data-take="5"]', 900);
    await t.ok('the reels\' tip opens clear of the LCD and the cost line (L-06)', () => { const tip = document.querySelector('#tip'); if (!tip || tip.hidden) return 'no tip'; const a = tip.getBoundingClientRect(); return [...document.querySelectorAll('.tile[data-m="takes"] :is(.tk-lcd, .tk-cost, output, [data-out])')].every(e => { const b = e.getBoundingClientRect(); return a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom; }); });
    await t.hover('.tile[data-m="sampler"] [data-knob="samp.cfg"]', 900);
    await t.ok('a knob\'s tip never covers its value (L-06)', () => { const tip = document.querySelector('#tip'); if (!tip || tip.hidden) return 'no tip'; const a = tip.getBoundingClientRect(), b = document.querySelector('[data-out="samp.cfg"]').getBoundingClientRect(); return a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom; });
  } },
  { id: 'one-changed-mark', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['latent', 'seed', 'planner', 'tempo'], ['nat'], { 'lss.str': 0.9, 'lss.var': 0.4, 'den.str': 0.9, 'den.smooth': 0.9, 'lm.logp': true, 'lm.seedlock': false, 'seed.lock': true, 'tempo.beatio': true, 'pp.nat.vr': 5, 'pp.nat.vd': 0.5 })(t);
    await t.ok('every changed control carries one 5px dot, and it never touches the next label or control (L-08)', () => {
      const bad = []; document.querySelectorAll('.changed').forEach(el => { [el, ...el.querySelectorAll('*')].forEach(c => { const cs = getComputedStyle(c, '::after'); if (cs.content === 'none' || cs.width !== '5px') return;
        const r = c.getBoundingClientRect(); /* the dot sits 4px after its host's text or box: probe the 3px beyond it for another control's ink */
        const y = r.top + r.height / 2, xs = [r.right + 1, r.right + 4, r.right + 7, r.right + 10];
        for (const x of xs) { const h = document.elementFromPoint(x, y); if (h && h !== c && !c.contains(h) && !h.contains(c) && h.closest('.kl, output, .tok, .flag, .seg, .pk-lcdwrap, .lcdsel')) { bad.push((el.dataset.ctl || el.className) + ' → ' + h.className); break; } } }); });
      return !bad.length || bad.slice(0, 4).join(' | '); });
    await t.ok('the dot on a light surface is the dark amber (E-12)', () => getComputedStyle(document.querySelector('.tile[data-m="seed"] .lock.changed'), '::after').backgroundColor === 'rgb(154, 98, 0)' || getComputedStyle(document.querySelector('.tile[data-m="seed"] .lock.changed'), '::after').backgroundColor);
  } },
];
