// Round 5, the Lab tier on ACE-Step (review L, with what was left of review Q): section masks that stay in their
// cartridge, pedal and side-unit controls, and the smaller Lab and Studio faces.
const desk = (tier, mods, pedals = [], extra) => t => t.do(`${tier} desk`, ([tier, mods, pedals, extra]) => { S.tier = tier; S.mods = [...mods]; S.pedals = [...pedals]; pedals.forEach(id => { if (PED[id]?.on) S.vals[PED[id].on] = true; }); if (extra) { if (extra.vals) Object.assign(S.vals, extra.vals); if (extra.slots) S.slots = extra.slots; } render(); }, [tier, mods, pedals, extra || null], 400);
const SLOTS = [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'folk-voice', type: 'LoKr' }, { file: 'synthwave-80s', type: 'DoRA' }];
const LYR = n => Array.from({ length: n }, (_, i) => `[${['Verse', 'Chorus', 'Bridge'][i % 3]} ${Math.floor(i / 3) + 1}]\nla`).join('\n');
export default [
  ...[[1920, 1080], [1280, 720]].map(([w, h]) => ({ id: `section-masks-stay-in-their-cartridge-${w}`, w, h, run: async t => {
    for (const tier of ['lab', 'studio']) for (const k of [2, 3]) for (const n of [4, 9, 16, 30]) {
      await desk(tier, ['adapters'], [], { slots: SLOTS.slice(0, k), vals: { 'ad.adv': 'Advanced', 'lyrics.text': LYR(n) } })(t);
      await t.ok(`${tier}, ${k} adapters, ${n} sections: every chip of every strip lies inside its cartridge, wide enough for its tag (L-03)`, () => {
        const bad = []; document.querySelectorAll('.cart .secband').forEach((b, ci) => { const c = b.closest('.cart').getBoundingClientRect();
          if (b.children.length !== sections(get('lyrics.text')).length) bad.push(`#${ci + 1} has ${b.children.length} chips`);
          [...b.children].forEach(x => { const q = x.getBoundingClientRect(), r = document.createRange(); r.selectNodeContents(x); const tw = r.getBoundingClientRect().width;
            if (q.left < c.left + 1 || q.right > c.right - 1 || q.bottom > c.bottom - 1) bad.push(`#${ci + 1} ${x.textContent} outside`);
            if (tw > q.width) bad.push(`#${ci + 1} ${x.textContent} ${tw.toFixed(1)} in ${q.width.toFixed(1)}`); }); });
        return !bad.length || bad.slice(0, 6); });
    }
    await t.click('.cart:nth-child(1) .secband button:nth-child(16)');
    await t.ok('the sixteenth section of the first cartridge can be pressed (L-03)', () => sectionDirectives(get('lyrics.text'))[15]?.[0] === 0.5 || JSON.stringify(sectionDirectives(get('lyrics.text'))[15]));
  } })),
  { id: 'lab-pedals-are-as-tall-as-their-controls', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['post'], ['tile', 'refiner', 'ss', 'gain', 'vst', 'master', 'norm'])(t);
    await t.ok('pedals of a row stand on one line, and a short pedal is not stretched to the tallest (L-12)', () => { const ps = [...document.querySelectorAll('[data-pedals] > .pedal')], r = id => document.querySelector(`[data-pid="${id}"]`).getBoundingClientRect();
      return (Math.abs(r('tile').bottom - r('ss').bottom) < 1 && r('tile').height < r('ss').height - 100) || ps.map(p => p.dataset.pid + ' ' + Math.round(p.getBoundingClientRect().height) + '@' + Math.round(p.getBoundingClientRect().bottom)); });
    await t.ok('no pedal carries more than a step of bare face above its LED, and none less than 12px (L-12)', () => { const bad = [...document.querySelectorAll('[data-pedals] > .pedal')].map(p => { const led = p.querySelector('.led').getBoundingClientRect().top, last = Math.max(...[...p.children].filter(c => !c.matches('.led, .foot, .always, .px, b')).map(c => c.getBoundingClientRect().bottom)); return [p.dataset.pid, Math.round(led - last)]; }).filter(([, g]) => g < 12 || g > 44); return !bad.length || bad; });
    await t.ok('a kept knob cell is a plate that says what fills it (L-13)', () => { const e = document.querySelector('[data-pid="ss"] .kn-empty[data-blank]'); return !!e && /Crossover or Mix/.test(e.dataset.blank) && getComputedStyle(e, '::before').boxShadow !== 'none' || e?.outerHTML; });
    const h0 = await t.js(() => document.querySelector('[data-pid="ss"]').offsetHeight);
    for (const b of ['Crossover', 'Mix']) { await t.do(b, b => setVal('pp.ss.blend', b), b); await t.eq(`StableStep is as tall with Source blend ${b} (L-13)`, () => document.querySelector('[data-pid="ss"]').offsetHeight, h0); }
  } },
  { id: 'pedals-with-nothing-to-run', w: 1920, h: 1080, run: async t => {
    await desk('studio', [], [])(t);
    await t.do('add VST3 and Mastering', () => { addItem('pd:vst'); addItem('pd:master'); addItem('pd:norm'); });
    await t.eq('a pedal with nothing to run arrives bypassed (Q-10)', () => [get('pp.vst.on'), get('pp.master.on'), get('pp.norm.on')], [false, false, true]);
    await t.click('[data-pid="master"] .foot');
    await t.ok('engaged with nothing to run: the count says so, and the pedal its reason in two lines (Q-10)', () => { const s = document.querySelector('[data-pid="master"] .pmid small'); return /1 on · 1 needs setting up/.test(document.querySelector('[data-ppcount]').textContent) && s.getBoundingClientRect().height < 40 && /Needs a reference track/.test(document.querySelector('[data-pid="master"]').title || document.querySelector('[data-pid="master"]').dataset.tipHeld) || [document.querySelector('[data-ppcount]').textContent, s.getBoundingClientRect().height, document.querySelector('[data-pid="master"]').title]; });
    await t.click('[data-pid="master"] .pmid .linkbtn');
    await t.ok('"set in Lab" goes to Lab and to the reference slot (Q-10, Q-28)', () => S.tier === 'lab' && !!document.activeElement?.closest('[data-pid="master"] .pslot') || [S.tier, document.activeElement?.outerHTML.slice(0, 80)]);
    await t.ok('in Lab the reason does not send you to Lab', () => !/in Lab/.test(document.querySelector('[data-pid="master"]').title + (document.querySelector('[data-pid="master"]').dataset.tipHeld || '')));
  } },
  { id: 'side-units-say-their-actions', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['post'], ['norm', 'gain', 'lrc', 'q'])(t);
    await t.ok('power and remove are named for what they do (Q-11)', () => [...document.querySelectorAll('.xunit')].every(u => /^Turn (on|off)$/.test(u.querySelector('.u-pow').title) && u.querySelector('.u-x').title === 'Take off the board'));
    await t.ok('foot switches say Turn on / Turn off as the units do (L-20)', () => [...document.querySelectorAll('[data-pedals] .foot')].every(f => /^Turn (on|off) /.test(f.getAttribute('aria-label'))));
    await t.ok('every unit on a pedal rides small after its number (Q-11)', () => { const o = document.querySelector('[data-pid="gain"] output'); return o.dataset.unit === 'dB' && !/dB/.test(o.textContent); });
    await t.ok('the count says how many side units are on (Q-31)', () => /2 side units, 2 on/.test(document.querySelector('[data-ppcount]').textContent) || document.querySelector('[data-ppcount]').textContent);
    await t.ok('a tip that only repeats a printed line does not open (Q-11)', () => { const kn = document.querySelector('[data-pid="norm"] .kn[title]'); return sameAsLabel(kn, 'Ceiling') && !sameAsLabel(kn, 'Ceiling of the limiter'); });
  } },
  { id: 'lab-controls-round5', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['sampler', 'tempo', 'perf', 'adapters', 'latent', 'seed', 'takes', 'planner'], [], { slots: [SLOTS[0]] })(t);
    await t.do('Shift by hand, then auto again', () => { S.vals['samp.shift.auto'] = false; setVal('samp.shift', 4.9); });
    await t.click('.tile[data-m="sampler"] [data-auto="samp.shift"]');
    await t.ok('on auto the Shift knob stands at the value auto stands for, in the auto colour (L-11)', () => { const k = document.querySelector('[data-knob="samp.shift"]'), c = C['samp.shift']; return Math.abs(parseFloat(k.style.getPropertyValue('--p')) - (defOf('samp.shift') - c.min) / (c.max - c.min)) < 1e-6 && k.classList.contains('is-auto') || k.style.getPropertyValue('--p'); });
    await t.do('trim on, length 10:00', () => { setVal('tempo.trim', true); S.vals['tempo.dur.auto'] = false; setVal('tempo.dur', 600); }, null, 300);
    await t.ok('the trim tag stands clear of the time cap at the end (L-15)', () => { const c = document.querySelector('.tile[data-m="tempo"] [data-rcap]').getBoundingClientRect(), r = document.querySelector('.tile[data-m="tempo"] [data-trim]').getBoundingClientRect(); return c.left - r.right >= 6 || c.left - r.right; });
    await t.ok('every solver, schedule and guidance mode may take two lines before it is cut (L-16; the web fonts are blocked here, so the names themselves are judged in round5/implL/picks.js)', () => { const bad = []; for (const k of ['samp.solver', 'samp.sched', 'samp.guide']) for (const v of optsOf(k)) { setVal(k, v); const b = document.querySelector(`.tile[data-m="sampler"] [data-pick="${k}"] .pk-val`); if (getComputedStyle(b).webkitLineClamp !== '2' || (b.scrollHeight > b.clientHeight + 1 && b.clientHeight < 2 * parseFloat(getComputedStyle(b).lineHeight) - 2)) bad.push(v + ' ' + b.clientHeight + '/' + b.scrollHeight); } setVal('samp.sched', 'Linear'); setVal('samp.guide', 'APG'); setVal('samp.solver', 'Euler'); return !bad.length || bad; });
    await t.do('CFG window', () => { setVal('samp.cs', 0.1); setVal('samp.ce', 0.8); });
    await t.ok('the CFG handles\' names stand above the plot, never on it (L-17)', () => { const svg = document.querySelector('.tile[data-m="sampler"] [data-scope] svg').getBoundingClientRect(); return [...document.querySelectorAll('.tile[data-m="sampler"] .ivl-l')].every(l => l.getBoundingClientRect().bottom <= svg.top + 1); });
    const gw = await t.js(() => document.querySelector('.tile[data-m="perf"] .gauge').offsetWidth);
    await t.do('perf pad', () => { setVal('perf.cache', 0.6); setVal('perf.cfgcut', 0.2); });
    await t.eq('the speed gauge keeps one width (L-18)', () => document.querySelector('.tile[data-m="perf"] .gauge').offsetWidth, gw);
    await t.click('.tile[data-m="latent"] [data-example="lat.ts"]');
    await t.do('an invalid list', () => { const el = document.querySelector('[data-bind="lat.ts"]'); el.value = '0.9, banana, 0.2'; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); });
    await t.ok('an invalid list says what is still in force and offers it back (L-19)', () => /Still using the last valid list \(6 timesteps\)/.test(document.querySelector('[data-err="lat.ts"]').textContent) && !!document.querySelector('[data-putback="lat.ts"]'));
    await t.do('leave the field', () => document.activeElement.blur());
    await t.ok('invalid is the field\'s border, not a focus-like outer ring (L-19)', () => getComputedStyle(document.querySelector('[data-bind="lat.ts"]')).outlineStyle === 'none' || getComputedStyle(document.querySelector('[data-bind="lat.ts"]')).outline);
    await t.click('[data-putback="lat.ts"]');
    await t.ok('Put it back restores the list in force', () => document.querySelector('[data-bind="lat.ts"]').value === get('lat.ts') && !document.querySelector('[data-err="lat.ts"]').textContent);
    await t.focus('.tile[data-m="adapters"] [data-fader="s0"]'); await t.press('Enter'); await t.type('2.5'); await t.press('Enter');
    await t.eq('a fader takes a typed value, as a knob does (L-22)', () => [get('s0'), document.activeElement?.dataset.fader], [2.5, 's0']);
    await t.ok('with one adapter on Single, the way to a second stands beside it (L-22)', () => !!document.querySelector('.tile[data-m="adapters"] [data-cartadd2]'));
    await t.click('.tile[data-m="adapters"] [data-cartadd2]');
    await t.focus('#pop .cartpick'); await t.press('ArrowRight');
    await t.ok('the arrow keys move through the adapter list (L-22)', () => document.activeElement === document.querySelectorAll('#pop .cartpick')[1]);
    await t.press('Enter');
    await t.eq('loading the second stacks them, in one step (L-22)', () => [S.slots.length, get('ad.adv'), UNDO[UNDO.length - 1].label === `Load ${S.slots[1]?.file}`], [2, 'Advanced', true]);
    await t.ok('the second fader\'s step names its adapter (found here)', () => C.s1.undoName && /strength$/.test(C.s1.undoName({ slots: S.slots })));
    await t.do('three saved seeds, one loaded', () => { S.stars = [{ v: 11, n: 'one' }, { v: 22, n: 'two' }, { v: 33, n: 'three' }]; S.seed = 11; S.vals['seed.lock'] = true; render(); });
    for (let i = 0; i < 4; i++) { const was = await t.js(() => S.seed); await t.click('[data-smrand]'); await t.ok(`Random saved seed ${i + 1} never picks the one loaded (L-22)`, w => S.seed !== w && /^Loaded “.+” \(\d+\)$/.test(document.querySelector('#toast .toast-msg').textContent) || [S.seed, document.querySelector('#toast .toast-msg').textContent], was); }
    await t.ok('the seed lock\'s changed dot sits inside the lock (L-14)', () => { const l = document.querySelector('.seed-row .lock.changed'), a = getComputedStyle(l, '::after'); return !!l && parseFloat(a.right) >= 0 && a.left !== `${l.offsetWidth + 4}px` || a.left + ' ' + a.right; });
    await t.ok('a one-digit token is still three characters wide (L-25)', () => [...document.querySelectorAll('.tile[data-m="planner"] .tok')].every(x => x.getBoundingClientRect().width >= 3 * 7 - 1));
    await t.ok('the negative prompt, empty, says so between one pair of quotes (L-24)', () => document.querySelector('[data-bind="lm.neg"]').placeholder === 'empty');
    await t.ok('wrapped option lines hang under the first option (L-24)', () => [...document.querySelectorAll('.tile[data-m="planner"] .tline.hang')].length >= 2);
  } },
  { id: 'runs-of-presses-are-one-step', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['tempo'])(t);
    const u0 = await t.js(() => UNDO.length);
    await t.click('[data-tap]'); await t.click('[data-tap]'); await t.click('[data-tap]'); await t.click('[data-tap]');
    await t.eq('a run of taps is one step (L-22)', u0 => UNDO.length - u0, 1, u0);
    const u1 = await t.js(() => UNDO.length);
    await t.focus('.wheel [tabindex="0"]'); await t.press('ArrowRight', 150); await t.press('ArrowRight', 150); await t.press('ArrowRight', 150);
    await t.eq('a run of arrow presses on the key wheel is one step (L-25)', u1 => [UNDO.length - u1, get('tempo.key') !== 'Auto'], [1, true], u1);
  } },
  { id: 'timbre-one-column-round5', w: 1920, h: 1080, run: async t => {
    await desk('studio', ['timbre', 'seed', 'takes', 'avoid'])(t);
    await t.do('four recent, a long one loaded', () => { S.refs = ['reference-mix.wav', 'my-band-master.flac', 'live-room-take.wav', 'a-very-long-reference-file-name-for-testing-master-v12.wav']; S.timbreFile = S.refs[3]; render(); }, null, 300);
    await t.ok('recent: the label on its line, the two latest and "2 more…" on one row (L-21)', () => { const ch = [...document.querySelectorAll('.tile[data-m="timbre"] .refs > button')]; return ch.length === 3 && new Set(ch.map(b => Math.round(b.getBoundingClientRect().top))).size === 1 && /2 more/.test(ch[2].textContent) || ch.map(b => b.textContent); });
    await t.ok('the loaded name is cut once, keeping its end (L-21)', () => { const b = document.querySelector('[data-fitname]'); return b.scrollWidth <= b.clientWidth + 1 && /ster-v12\.wav$/.test(b.textContent) || b.textContent; });
    await t.click('.tile[data-m="timbre"] [data-refsmore]');
    await t.click('#pop [data-pref="live-room-take.wav"]');
    await t.eq('the rest are one click further (L-21)', () => [S.timbreFile, document.querySelector('#pop').hidden], ['live-room-take.wav', true]);
  } },
  { id: 'drop-preview-is-the-tile', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['sampler', 'latent', 'repetition', 'guidance', 'perf', 'planner'])(t);
    for (const m of ['perf', 'repetition']) {
      await t.do(`drag ${m}`, m => { const el = document.querySelector(`.tile[data-m="${m}"] [data-drag]`); window.__dt = new DataTransfer(); el.dispatchEvent(new DragEvent('dragstart', { bubbles: true, cancelable: true, dataTransfer: window.__dt })); }, m, 200);
      await t.do('over the guidance tile', () => { const tg = document.querySelector('.tile[data-m="guidance"]'); tg.scrollIntoView({ block: 'center', behavior: 'instant' }); const r = tg.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + 20, o = { bubbles: true, cancelable: true, dataTransfer: window.__dt, clientX: x, clientY: y }, el = document.elementFromPoint(x, y) || tg; el.dispatchEvent(new DragEvent('dragenter', o)); el.dispatchEvent(new DragEvent('dragover', o)); window.__dpt = [el, o]; }, null, 300);
      const g = await t.js(() => { const r = document.querySelector('#bento > .ghost').getBoundingClientRect(); return [Math.round(r.width), Math.round(r.height)]; });
      await t.do('drop', () => { const [el, o] = window.__dpt; el.dispatchEvent(new DragEvent('drop', o)); }, null, 600);
      await t.eq(`${m}: the preview is the tile's own face, the size it lands at (L-26)`, m => { const r = document.querySelector(`.tile[data-m="${m}"]`).getBoundingClientRect(); return [Math.round(r.width), Math.round(r.height)]; }, g, m);
      await t.eq(`${m}: after the drag, one tile carries its name`, m => document.querySelectorAll(`[data-m="${m}"], [data-mo]`).length, 1, m);
    }
  } },
  { id: 'light-screens-share-the-column-edges', w: 1920, h: 1080, scheme: 'light', run: async t => {
    await desk('lab', ['sampler'])(t);
    await t.ok('the scope\'s bezel is inside its box: no shadow spreads past it (L-23)', () => { const sh = getComputedStyle(document.querySelector('.tile[data-m="sampler"] .scope')).boxShadow.split(/,(?![^(]*\))/); return sh.every(p => /inset/.test(p) || !(parseFloat(p.trim().replace(/^rgba?\([^)]*\)\s*/, '').split(/\s+/)[3]) > 0)) || sh; });
  } },
  ...[['sampler', 'takes', 'seed', 'planner'], ['sampler', 'seed', 'takes', 'planner'], ['tempo', 'sampler', 'takes', 'seed'], ['sampler', 'avoid', 'takes', 'seed', 'planner'], ['planner', 'sampler', 'seed', 'takes'], ['sampler', 'timbre', 'seed', 'takes']].map((mods, n) => ({ id: `a-content-change-keeps-every-width-${n + 1}`, w: 1920, h: 1080, run: async t => {
    await desk('studio', mods)(t);
    await t.do('a solver set in Studio', () => setVal('samp.solver', 'DPM++ 3M'));
    await t.do('Quick, where it is an "Also set" line', () => { S.tier = 'quick'; render(); }, null, 1200);
    const cols = () => [...document.querySelectorAll('#bento > .tile[data-m]')].map(x => x.dataset.m + ' ' + x.style.gridColumn);
    const before = await t.js(cols);
    await t.ok('the Sampler says what it hides', () => !!document.querySelector('.tile[data-m="sampler"] .q-also'));
    await t.click('.tile[data-m="sampler"] .tile-head h3'); await t.press('Control+z', 1200);
    await t.eq('Undo that takes the line away moves no tile sideways and changes no width (Q-29)', cols, before);
    await t.press('Control+Shift+z', 1200);
    await t.eq('nor does Redo (Q-29)', cols, before);
  } })),
  { id: 'add-by-enter-focuses-what-arrived', w: 1920, h: 1080, run: async t => {
    await desk('studio', ['tempo'])(t);
    await t.focus('#rightPanel .bcard[data-additem="seed"]'); await t.press('Enter');
    await t.ok('Enter on a Modules card adds it and focus goes to it, as a click does (L-22, Q-30)', () => S.mods.includes('seed') && !!document.activeElement?.closest('.tile[data-m="seed"]') || document.activeElement?.outerHTML.slice(0, 80));
  } },
];
