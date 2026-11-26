// Stable Audio 3 and HeartMuLa, sizes below the desktop range, and input modes (review 4, S- findings, round 4).
const desk = (tier, mods, extra) => t => t.do(`${tier} desk`, ([tier, mods, extra]) => { S.tier = tier; S.mods = mods; S.pedals = []; if (extra?.vals) { Object.assign(S.vals, extra.vals); } Object.assign(S, { ...(extra || {}), vals: S.vals }); render(); }, [tier, mods, extra]);
export default [
  { id: 'every-value-change-is-an-undo-step', run: async t => {
    await t.click('#presetList .pre.wfrow[data-preset="Songwriter"]');
    await desk('lab', ['sampler', 'tempo'])(t);
    await t.focus('[data-knob="samp.cfg"]'); await t.press('End');
    await t.eq('a key press on a knob is a step, named for what it did (S-03)', (() => document.querySelector('#undoBtn').hidden ? '' : document.querySelector('#undoBtn span').textContent), 'Guidance 9.0 → 20.0');
    await t.do('hold the down arrow: one press, then nine repeats', async () => {}, null);
    await t.page.keyboard.down('ArrowDown'); for (let i = 0; i < 9; i++) await t.page.keyboard.down('ArrowDown'); await t.page.keyboard.up('ArrowDown'); await t.settle();
    await t.eq('a held key is one step', () => [UNDO.filter(e => e.value).length, document.querySelector('#undoBtn span').textContent], [2, 'Guidance 20.0 → 19.0']);
    await t.mouseDrag('[data-knob="samp.cfg"]', [[0, 10], [0, 10], [0, 10], [0, 10]]);
    await t.ok('a drag is one step, however many values it passes', () => UNDO.filter(e => e.value).length === 3 && /^Guidance 19\.0 → /.test(document.querySelector('#undoBtn span').textContent));
    await t.ok('the Undo button says what it will put back', () => { const b = document.querySelector('#undoBtn'); b.dispatchEvent(new PointerEvent('pointerenter')); return /^Undo Guidance 19\.0 → .* \(Ctrl\+Z\)\. Puts Guidance back to 19\.0\.$/.test(b.title) ? true : b.title; });
    await t.press('Control+z');
    await t.ok('Ctrl+Z takes back the drag first, and says what it put back', () => +get('samp.cfg') === 19 && (document.querySelector('#toast .toast-msg')?.textContent || '') === 'Undone: Guidance back to 19.0' ? true : [get('samp.cfg'), (document.querySelector('#toast .toast-msg')?.textContent || '')]);
    await t.press('Control+z'); await t.press('Control+z');
    await t.ok('then the held key, then the first press: Guidance is where it started', () => !('samp.cfg' in S.vals) || +S.vals['samp.cfg'] === 9);
    await t.eq('and the next Ctrl+Z reaches the older desk step', (() => document.querySelector('#undoBtn').hidden ? '' : document.querySelector('#undoBtn span').textContent), 'Open "Songwriter"');
    await t.press('Control+Shift+z');
    await t.ok('Redo says what it set', () => +get('samp.cfg') === 20 && (document.querySelector('#toast .toast-msg')?.textContent || '') === 'Redone: Guidance 20.0' ? true : [get('samp.cfg'), (document.querySelector('#toast .toast-msg')?.textContent || '')]);
    await t.click('[data-ledtog="tempo.beatio"]');
    await t.eq('a click is a step, and a new step clears Redo', () => [(() => document.querySelector('#undoBtn').hidden ? '' : document.querySelector('#undoBtn span').textContent)(), document.querySelector('#redoBtn').hidden], ['Beat intro and outro (DJ bars) off → on', true]);
    await t.click('[data-auto="samp.shift"]');
    await t.ok('auto is a value like any other', () => /^Shift auto → /.test((() => document.querySelector('#undoBtn').hidden ? '' : document.querySelector('#undoBtn span').textContent)()) ? true : (() => document.querySelector('#undoBtn').hidden ? '' : document.querySelector('#undoBtn span').textContent)());
    await t.press('Control+z');
    await t.ok('and is put back', () => isAuto('samp.shift'));
    /* one gesture each: a change in one task joins the step before it */
    for (let i = 0; i < 120; i++) await t.js(i => setVal('samp.steps', 10 + (i % 40)), i);
    await t.settle();
    await t.eq('the history keeps the last hundred', () => UNDO.length, 100);
  } },
  { id: 'text-and-prompt-chips-are-steps', run: async t => {
    await desk('studio', ['tempo'], { backend: 'sa3' })(t);
    await t.click('#prompt'); await t.type('warm pads, slow build at 90bpm');
    await t.click('.tile[data-m="tempo"] [data-capadd="90 BPM"]');
    await t.ok('a tempo inside a sentence is the user\'s words: the chip is added after them (S-04)', () => document.querySelector('#prompt').value === 'warm pads, slow build at 90bpm, 90 BPM' ? true : document.querySelector('#prompt').value);
    await t.ok('the pressed chip looks pressed (S-02)', () => { const b = document.querySelector('[data-capadd="90 BPM"]'), o = document.querySelector('[data-capadd="140 BPM"]'); return b.getAttribute('aria-pressed') === 'true' && getComputedStyle(b).backgroundColor !== getComputedStyle(o).backgroundColor; });
    await t.eq('the chip is one undo step (S-03)', (() => document.querySelector('#undoBtn').hidden ? '' : document.querySelector('#undoBtn span').textContent), 'Add 90 BPM to the prompt'); /* Stable Audio 3's word for the box (S-05) */
    await t.click('.tile[data-m="tempo"] [data-capadd="120 BPM"]');
    await t.eq('a chip of the same group replaces only the item, not the words', () => document.querySelector('#prompt').value, 'warm pads, slow build at 90bpm, 120 BPM');
    await t.click('.top .brand'); await t.press('Control+z'); await t.press('Control+z');
    await t.eq('Ctrl+Z takes back the chips, newest first', () => document.querySelector('#prompt').value, 'warm pads, slow build at 90bpm');
    await t.press('Control+z');
    await t.eq('then the typing, as one step', () => document.querySelector('#prompt').value, '');
  } },
  { id: 'touch-scroll-or-control', w: 390, h: 844, touch: true, run: async t => {
    await t.do('a Lab desk', () => { loadPreset(FACTORY.find(f => f.name === 'Everything'), false); S.tier = 'lab'; render(); });
    const cdp = await t.page.context().newCDPSession(t.page);
    const swipe = async (sel, dx, dy, hold) => { const r = await t.js(sel => { const el = document.querySelector(sel); el.scrollIntoView({ block: 'center', behavior: 'instant' }); const b = el.getBoundingClientRect(); return { x: b.left + b.width / 2, y: b.top + b.height / 2, sy: scrollY }; }, sel); await t.settle(100);
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: r.x, y: r.y, id: 1 }] }); if (hold) await t.settle(hold);
      for (let i = 1; i <= 10; i++) { await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: r.x + dx * i / 10, y: r.y + dy * i / 10, id: 1 }] }); await t.page.waitForTimeout(16); }
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] }); await t.settle(200); return r.sy; };
    const before = await t.js(() => JSON.stringify(S.vals));
    await swipe('[data-knob="samp.cfg"]', 0, -160); await swipe('[data-ruler="tempo.dur"]', 0, -160); await swipe('[data-num="samp.steps"]', 0, -60); await swipe('[data-scrub="lm.topk"]', 0, -120); await swipe('.xy[data-xy]', 0, -120);
    await t.ok('a swipe up or down that starts on a knob, ruler, number box, token or pad changes none of them (S-29)', b => JSON.stringify(S.vals) === b, before);
    await swipe('[data-knob="samp.cfg"]', 0, -60, 300);
    await t.ok('resting a finger on a knob first, then dragging, turns it', () => +get('samp.cfg') > 9);
    await swipe('[data-ruler="tempo.dur"]', -60, 4);
    await t.ok('a sideways drag on the ruler moves it', () => S.vals['tempo.dur.auto'] === false);
    await t.eq('each touch edit is an undo step of its own', () => UNDO.filter(e => e.value).length, 2);
  } },
  { id: 'length-tile-tells-the-truth-on-sa3-and-heartmula', run: async t => {
    await desk('studio', ['tempo', 'takes', 'lyrics'], { backend: 'ace', vals: { 'tempo.dur.auto': false, 'tempo.dur': 300, 'takes.n': 4 } })(t);
    await t.do('switch to HeartMuLa', () => switchEngine('hm'));
    await t.ok('the length held at 4:00 says so under its ruler (S-10)', () => /Capped at 4:00 here · your 5:00 returns on ACE-Step 1.5/.test(document.querySelector('[data-held="tempo.dur"]').textContent) && !document.querySelector('[data-held="tempo.dur"]').hidden);
    await t.ok('Takes is a strip that says the 4 takes wait (S-37, S-10)', () => { const x = document.querySelector('.tile[data-m="takes"]'); return x.classList.contains('is-strip') && !x.hasAttribute('data-grow') && /Your 4 takes return on ACE-Step 1.5/.test(x.textContent); });
    await t.ok('the lyric sheet says empty means instrumental (S-08)', () => /Leave this empty for an instrumental/.test(document.querySelector('[data-bind="lyrics.text"]').placeholder));
    await t.focus('[data-ruler="tempo.dur"]'); await t.press('ArrowLeft');
    await t.ok('moving the ruler ends the hold, and its line goes', () => document.querySelector('[data-held="tempo.dur"]').hidden);
    await t.do('a 0:30 cap and three runs', () => { S.vals['tempo.dur'] = 30; S.vals['tempo.dur.auto'] = false; document.querySelector('#prompt').value = 'folk'; S.vals['prompt.text'] = 'folk'; render(); generate(); generate({}); });
    await t.do('let them finish', () => {}, null, 30000);
    await t.ok('HeartMuLa never runs past the cap (S-07)', () => S.songs.filter(x => x.bk === 'hm').length > 0 && S.songs.filter(x => x.bk === 'hm').every(x => x.dur <= 30) ? true : S.songs.filter(x => x.bk === 'hm').map(x => x.dur));
    await t.do('auto on again, then off', () => { delete S.vals['tempo.dur']; S.vals['tempo.dur.auto'] = true; render(); });
    await t.click('.tile[data-m="tempo"] [data-auto="tempo.dur"]');
    await t.eq('turning auto off starts where auto was showing, 4:00 (S-17)', () => +get('tempo.dur'), 240);
    await t.do('Stable Audio 3, Beat I/O at 8 bars on a 0:11 song', () => { switchEngine('sa3'); });
    await t.do('settings', () => { S.vals['tempo.dur'] = 11; S.vals['tempo.dur.auto'] = false; S.vals['tempo.beatio'] = true; S.vals['tempo.beatbars'] = '8'; render(); });
    await t.ok('Studio draws the Beat I/O marks on the ruler', () => [...document.querySelectorAll('.tile[data-m="tempo"] [data-io]')].some(x => !x.hidden));
    await t.ok('and says 8 bars each way does not fit (S-39)', () => /8 bars each way needs about 0:32 at 120 BPM/.test(document.querySelector('[data-ionote]').textContent) && !document.querySelector('[data-ionote]').hidden);
  } },
  { id: 'f6-moves-between-modules', run: async t => {
    await t.do('a Lab desk', () => { loadPreset(FACTORY.find(f => f.name === 'Everything'), false); S.tier = 'lab'; render(); });
    await t.click('#prompt'); await t.press('F6');
    const first = await t.js(() => document.activeElement.closest('.tile')?.dataset.m);
    await t.ok('F6 from the prompt enters the desk at its first module (S-33)', () => !!document.activeElement.closest('#bento > .tile'));
    await t.press('F6'); await t.press('F6');
    await t.ok('each press is the next module in reading order', f => { const ts = [...document.querySelectorAll('#bento > .tile[data-m]')]; return ts.indexOf(document.activeElement.closest('.tile')) === ts.findIndex(x => x.dataset.m === f) + 2; }, first);
    await t.press('Shift+F6');
    await t.ok('Shift+F6 goes back one', f => { const ts = [...document.querySelectorAll('#bento > .tile[data-m]')]; return ts.indexOf(document.activeElement.closest('.tile')) === ts.findIndex(x => x.dataset.m === f) + 1; }, first);
  } },
];
