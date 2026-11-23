// Round 5, Stable Audio 3, HeartMuLa, sizes below desktop and input modes (review 4, S-01 to S-37): what the page says on
// these two engines, what stays with an engine, how their desks compose, and the narrow sizes. The pending scenarios at
// the end state behaviours the page is known to get wrong (README, "Pending scenarios"), whose fix is script the rebuild
// replaces.
const desk = (tier, mods, extra) => t => t.do(`${tier} desk`, ([tier, mods, extra]) => { if (extra?.backend && extra.backend !== S.backend) enterEngine(extra.backend); S.tier = tier; S.mods = mods; S.pedals = extra?.pedals || []; if (extra?.vals) Object.assign(S.vals, extra.vals); if (extra?.slots) S.slots = extra.slots; render(); }, [tier, mods, extra]);
const toastText = () => document.querySelector('#toast .toast-msg')?.textContent || '';

export default [
  /* S-05: one word for the main text box on each engine, from one place */
  { id: 'sa3-calls-its-box-a-prompt', run: async t => {
    await desk('studio', ['tempo'], { backend: 'sa3' })(t);
    await t.ok('the engine card and its note say prompt, not caption (S-05)', () => { const c = ENG.sa3.tag + ' ' + BK.sa3.note; return (/^Prompt only/.test(ENG.sa3.tag) && !/caption/i.test(c)) || c; });
    await t.ok('the Length tile says prompt', () => { const x = document.querySelector('.tile[data-m="tempo"]').textContent; return (/come from your prompt/.test(x) && /Add to the prompt/.test(x)) || x.slice(0, 200); });
    await t.click('#aiBtn');
    await t.ok('Write it for me: "Wrote a prompt (simulated)"', () => /^Wrote a prompt \(simulated\)/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) || (document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.ok('the box is named Prompt for assistive tech', () => document.querySelector('#prompt').getAttribute('aria-label') === 'Prompt' || document.querySelector('#prompt').getAttribute('aria-label'));
    await t.click('#draftsBtn');
    await t.ok('Drafts: "Prompt and song info" (no lyric sheet on this engine; S-33)', () => /^Prompt and song info/.test(document.querySelector('#pop .pop-p')?.textContent || '') || document.querySelector('#pop .pop-p')?.textContent);
  } },
  { id: 'heartmula-calls-its-box-tags', run: async t => {
    await desk('quick', ['sampler'], { backend: 'hm' })(t);
    await t.ok('the Quick Sampler row is "Follow the tags" (S-05)', () => /Follow the tags/.test(document.querySelector('.tile[data-m="sampler"]').textContent) || document.querySelector('.tile[data-m="sampler"]').textContent.slice(0, 160));
    await t.ok('the box is named Tags', () => document.querySelector('#prompt').getAttribute('aria-label') === 'Tags' || document.querySelector('#prompt').getAttribute('aria-label'));
    await t.click('#aiBtn');
    await t.ok('Write it for me writes tags and says so: "Wrote tags and lyrics (simulated)" (S-05, S-33)', () => (/^Wrote tags and lyrics \(simulated\)/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) && !/intimate room sound/.test(document.querySelector('#prompt').value)) || `${(document.querySelector('#toast .toast-msg')?.textContent || '')} | ${document.querySelector('#prompt').value}`);
  } },
  /* S-08: the sampler's settings stay with their engine, the experimental switch and the plugin's settings included */
  { id: 'lua-switch-stays-with-its-engine', run: async t => {
    await desk('lab', ['sampler', 'guidance'], { backend: 'hm' })(t);
    await t.click('.tile[data-m="sampler"] [data-ledtog="samp.plugins"]');
    await t.do('APG, momentum 0.76', () => { setVal('samp.guide', 'APG'); setVal('apg.mom', 0.76); });
    await t.click('#engBtn'); await t.click('#engPanel [data-backend="sa3"]');
    await t.ok('Stable Audio 3 arrives with Lua plugins off and its own momentum (S-08)', () => (get('samp.plugins') === false && +get('apg.mom') === 0.75) || JSON.stringify({ plugins: get('samp.plugins'), mom: get('apg.mom') }));
    await t.do('to ACE-Step', () => switchEngine('ace'));
    await t.ok('ACE-Step keeps its own momentum too', () => +get('apg.mom') === 0.75 || get('apg.mom'));
    await t.do('back to HeartMuLa', () => switchEngine('hm'));
    await t.ok('back on HeartMuLa, its switch and momentum are back', () => (get('samp.plugins') === true && +get('apg.mom') === 0.76 && get('samp.guide') === 'APG') || JSON.stringify({ plugins: get('samp.plugins'), mom: get('apg.mom'), guide: get('samp.guide') }));
  } },
  /* S-09: a revision is every setting for its engine; one that holds no adapters unloads them */
  { id: 'revision-without-adapters-unloads-them', run: async t => {
    await desk('studio', ['adapters', 'planner'], { backend: 'hm', slots: [{ file: 'heart-folk-lm', type: 'LoRA' }], vals: { s0: 1.4, 'hm.temp': 1.3 } })(t);
    await t.click('[data-wfopen="Start fresh"]');
    await t.do('Start fresh open in the list', () => { S.wfOpen.add('Start fresh'); S.wfFold = false; renderPresets(); });
    await t.click('.pre.rev[data-rev="Start fresh"][data-rn="Reference"] b');
    await t.ok('the adapter is unloaded, not left at strength 1.00 (S-09)', () => S.slots.length === 0 || JSON.stringify(S.slots));
    await t.ok('and the toast says so, with Undo', () => (/heart-folk-lm unloaded/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) && !!document.querySelector('#toast .toast-act')) || (document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.click('#toast .toast-act');
    await t.ok('Undo brings it back at 1.40', () => (S.slots.length === 1 && +get('s0') === 1.4) || JSON.stringify({ slots: S.slots, s0: get('s0') }));
  } },
  /* S-10: Stable Audio 3's Tiled decoder is idle until Lua plugins are on, says so, and offers the switch */
  { id: 'tiled-decoder-says-it-needs-lua', run: async t => {
    await desk('lab', ['sampler', 'post'], { backend: 'sa3', pedals: ['tile'], vals: { 'pp.tile.on': true } })(t);
    await t.ok('engaged without Lua plugins: the amber idle LED and one line "Needs Lua plugins · Turn on" (S-10)', () => { const p = document.querySelector('.pedal[data-pid="tile"]'); return (p.classList.contains('idle') && /Needs Lua plugins · Turn on/.test(p.textContent)) || `${p.className} | ${p.textContent}`; });
    await t.ok('the board does not count it as running', () => /need(s)? setting up/.test(document.querySelector('[data-ppcount]').textContent) || document.querySelector('[data-ppcount]').textContent);
    await t.click('.pedal[data-pid="tile"] [data-luaon]');
    await t.ok('Turn on flips the Sampler switch, and the pedal runs', () => { const p = document.querySelector('.pedal[data-pid="tile"]'); return (get('samp.plugins') === true && !p.classList.contains('idle') && !/Needs Lua/.test(p.textContent)) || `${get('samp.plugins')} ${p.className}`; });
    await t.ok('focus stays with the pedal', () => !!document.activeElement?.closest('.pedal[data-pid="tile"]') || document.activeElement?.outerHTML.slice(0, 80));
    await t.ok('one name for the switch: the undo step says Lua plugins', () => /^Lua plugins off → on$/.test(document.querySelector('#undoBtn span')?.textContent || '') || document.querySelector('#undoBtn span')?.textContent);
  } },
  /* S-04: a pedal that is on because it was added is not a change, and the link never claims hidden modules in Lab */
  { id: 'added-pedals-are-not-changes', run: async t => {
    await desk('lab', ['post'], { backend: 'ace' })(t);
    for (const pid of ['ss', 'ppvae', 'lift', 'gain', 'norm']) await t.do(`add ${pid}`, pid => addItem('pd:' + pid), pid);
    await t.ok('the go bar counts no changes for pedals just added (S-04)', () => /everything on defaults/.test(document.querySelector('#meter').textContent) || document.querySelector('#meter').textContent);
    await t.do('one real change: StableStep refine', () => setVal('pp.ss.str', 0.6));
    await t.ok('one change is counted', () => /1 change,/.test(document.querySelector('#meter').textContent) || document.querySelector('#meter').textContent);
    await t.click('#meter [data-showchanges]');
    await t.ok('the link never claims a module this tier hides in Lab', () => !/this tier hides/.test(document.querySelector('#toast .toast-msg')?.textContent || '') || document.querySelector('#toast .toast-msg').textContent);
  } },
  /* S-14: HeartMuLa takes one adapter, so its tile is never widened for it */
  { id: 'heartmula-adapters-hug-one-cartridge', w: 1920, h: 1080, run: async t => {
    await desk('studio', ['lyrics', 'adapters', 'planner', 'sampler'], { backend: 'hm', slots: [{ file: 'heart-folk-lm', type: 'LoRA' }] })(t);
    await t.ok('Studio: one column, the note under the cartridge (S-14)', () => { const tl = document.querySelector('.tile[data-m="adapters"]'), w = tl.getBoundingClientRect().width; return (w < 400 && !tl.classList.contains('widened')) || `${Math.round(w)}px ${tl.className}`; });
  } },
  /* S-27: under reduced motion an added tile carries the static ring */
  { id: 'reduced-motion-arrivals-are-ringed', reduced: true, run: async t => {
    await desk('studio', ['tempo'], { backend: 'ace' })(t);
    await t.do('add Seed', () => addItem('seed'), null, 300);
    await t.ok('the new tile is ringed (S-27)', () => { const tl = document.querySelector('.tile[data-m="seed"]'); return (tl.classList.contains('flash') && getComputedStyle(tl).outlineStyle === 'solid') || `${tl.className} ${getComputedStyle(tl).outlineStyle}`; });
  } },
  /* S-24, S-25: under a finger the page's 40px rule holds, and the picker speaks of taps */
  { id: 'touch-targets-and-words', w: 820, h: 1180, touch: true, run: async t => {
    await desk('lab', ['seed', 'post'], { backend: 'ace', pedals: ['gain'] })(t);
    await t.ok('top bar pages, Notes, desk bar, seed lock and search, Chain, pedal knob: 40px or more (S-24)', () => { const bad = []; for (const s of ['nav.main button', '#notesBtn', '.deskbar button', '.tile[data-m="seed"] .lock', '.tile[data-m="seed"] .sm-q', 'button.master', '.pedal .knob.sm']) for (const el of document.querySelectorAll(s)) { if (!el.offsetParent) continue; const r = el.getBoundingClientRect(); if (Math.min(r.width, r.height) < 39.5) bad.push(`${s} ${Math.round(r.width)}x${Math.round(r.height)}`); } return !bad.length || bad.join('; '); });
    await t.press('F9');
    await t.ok('the picker says Tap, not keys and drags (S-25)', () => { const x = document.querySelector('.fl-info')?.textContent || ''; return (/Tap to add/.test(x) && !/drag to place|walk the wall/.test(x)) || x; });
    await t.ok('its close and categories are 40px', () => { const bad = [...document.querySelectorAll('#pkClose, #pkBar > button')].filter(b => b.getBoundingClientRect().height < 39.5).map(b => b.textContent || b.id); return !bad.length || bad.join(', '); });
  } },

  /* ---- pending: behaviours the page gets wrong today; each fails now, for its finding's reason ---- */
  { id: 'guidance-pad-is-painted-when-it-appears', pending: { finding: 'S-03', right: 'A face rebuilt outside the main render is painted before it is shown: choosing APG in the Sampler shows the Guidance plugin\'s pad with its dot at the current momentum and threshold, and both read-outs filled, in the first frame.' }, run: async t => {
    await desk('lab', ['sampler', 'guidance'], { backend: 'hm', vals: { 'samp.plugins': true } })(t);
    await t.click('.tile[data-m="sampler"] [data-pick="samp.guide"]');
    await t.click('#dd [data-opt="APG"]');
    await t.ok('the pad carries its value and read-outs at once', () => { const x = document.querySelector('.tile[data-m="guidance"] .xy'); return (!!x && !!x.getAttribute('aria-valuetext') && /\d/.test(x.textContent)) || (x ? `valuetext ${x.getAttribute('aria-valuetext')}, text "${x.textContent.replace(/\s+/g, ' ').trim()}"` : 'no pad'); });
  } },
  { id: 'beat-io-note-follows-the-prompt-bpm', pending: { finding: 'S-06', right: 'The Beat I/O markers and the note under the ruler follow the tempo the song will have, wherever it is set: a BPM chip pressed on the Length tile, or a BPM typed in the prompt, changes them in the same frame.' }, run: async t => {
    await desk('studio', ['tempo'], { backend: 'sa3', vals: { 'tempo.beatio': true, 'tempo.beatbars': '8', 'tempo.dur': 20, 'tempo.dur.auto': false } })(t);
    await t.ok('at 120 BPM the note says so', () => /at 120 BPM/.test(document.querySelector('[data-ionote]').textContent) || document.querySelector('[data-ionote]').textContent);
    await t.click('.tile[data-m="tempo"] [data-capadd="90 BPM"]');
    await t.ok('after the 90 BPM chip the note reads 90 BPM', () => /at 90 BPM/.test(document.querySelector('[data-ionote]').textContent) || document.querySelector('[data-ionote]').textContent);
  } },
  { id: 'one-tempo-for-every-estimate', pending: { finding: 'S-07', right: 'One function gives the song\'s tempo (the BPM field, else a BPM written in the prompt or tags, else none) and every read-out that estimates by tempo uses it: with "96 BPM" in HeartMuLa\'s tags the lyric sheet says "at 96 BPM (from your tags)".' }, run: async t => {
    await desk('studio', ['tempo'], { backend: 'hm', vals: { 'lyrics.text': '[Verse]\nline one\nline two' } })(t);
    await t.focus('#prompt'); await t.type('folk, 96 BPM');
    await t.ok('the lyric sheet estimates at 96 BPM', () => { const s = [...document.querySelectorAll('.stat')].map(x => x.textContent).join(' | '); return /96 BPM/.test(s) || s; });
  } },
  { id: 'tab-leaves-the-song-menu', pending: { finding: 'S-16', right: 'Tab and Shift+Tab in a song card\'s actions menu close it and move to the control after (or before) its ⋯ button; focus is never left on the page body while a menu is open.' }, run: async t => {
    await t.focus('#latest .song[data-id="3"] [data-songmenu]'); await t.press('Enter');
    await t.press('Tab');
    await t.ok('the menu is closed and focus is on a control', () => (document.querySelector('#menu')?.hidden !== false && document.activeElement !== document.body) || `menu open ${!document.querySelector('#menu').hidden}, focus ${document.activeElement.tagName}`);
  } },
  { id: 'tab-stays-in-a-popover-over-a-toast', pending: { finding: 'S-17', right: 'A toast claims the next Tab only while focus is where it was when the toast appeared and no layer has opened since; inside a popover, menu or dialog, Tab stays in that layer.' }, run: async t => {
    await t.do('a switch with Undo', () => switchEngine('sa3'));
    await t.press('Control+z');
    await t.click('#draftsBtn');
    await t.focus('#draftName'); await t.press('Tab');
    await t.ok('Tab from the draft name goes to Save draft', () => document.activeElement?.matches('[data-draftsave]') || document.activeElement?.outerHTML.slice(0, 80));
  } },
  { id: 'delete-resets-a-value-control', pending: { finding: 'S-18', right: 'Delete (or Backspace) on a focused knob, token, fader, ruler, CFG handle or look-back strip returns it to its default, announced ("Guidance back to 9.0"); the control\'s tip names it beside the double-click.' }, run: async t => {
    await desk('studio', ['sampler'], { backend: 'ace' })(t);
    await t.focus('.tile[data-m="sampler"] [data-knob="samp.cfg"]'); await t.press('ArrowUp'); await t.press('ArrowUp');
    await t.press('Delete');
    await t.eq('Delete puts Guidance back to its default', () => +get('samp.cfg'), 9);
  } },
  { id: 'topk-scrub-scales-with-the-value', pending: { finding: 'S-28', right: 'A scrub over a wide range scales with the value (about 1 per pixel under 100, 10 under 1000, 100 above), and Shift+Arrow steps ×10, PageUp/PageDown ×100: HeartMuLa\'s top-K from 50, dragged 100px to the right, lands under 200.' }, run: async t => {
    await desk('lab', ['planner'], { backend: 'hm' })(t);
    await t.mouseDrag('.tile[data-m="planner"] [data-scrub="hm.topk"]', [[25, 0], [25, 0], [25, 0], [25, 0]]);
    await t.ok('100px right from 50 stays in the useful range', () => (+get('hm.topk') > 50 && +get('hm.topk') < 200) || get('hm.topk'));
  } },
  { id: 'length-drag-snaps-to-round-values', pending: { finding: 'S-31', right: 'Dragging a length ruler snaps to 5 s (15 s above 2:00) and to a labelled tick within 6px; the arrow keys keep 1 s steps.' }, run: async t => {
    await desk('studio', ['tempo'], { backend: 'sa3' })(t);
    await t.mouseDrag('.tile[data-m="tempo"] [data-ruler="tempo.dur"]', [[37, 0], [40, 0], [41, 0]]);
    await t.ok('the dragged length is a round value', () => { const v = +get('tempo.dur'); return v % (v > 120 ? 15 : 5) === 0 || v; });
  } },
  /* S-35 (third point): the tree pattern; found already right in this round's page, kept as a check */
  { id: 'right-opens-a-workflow-before-entering-it', run: async t => {
    await t.focus('.pre.wfrow[data-preset="Deep sampler"]'); await t.press('ArrowRight');
    await t.ok('the first Right opens it and stays on the row', () => (document.activeElement?.matches('.pre.wfrow[data-preset="Deep sampler"]') && !!document.querySelector('.pre.rev[data-rev="Deep sampler"]')) || document.activeElement?.outerHTML.slice(0, 100));
    await t.press('ArrowRight');
    await t.ok('the second moves to its first revision', () => document.activeElement?.matches('.pre.rev[data-rev="Deep sampler"]') || document.activeElement?.outerHTML.slice(0, 100));
  } },
  { id: 'a-swipe-on-a-knob-teaches-rest-then-drag', w: 390, h: 844, touch: true, pending: { finding: 'S-36', right: 'The first time a vertical swipe starts on a knob, fader, number box or XY pad under a finger, the page scrolls as it does now and one toast says "Rest a finger on a knob, then drag. A swipe scrolls."; the control lifts at 200 ms of rest, as it does under a mouse press.' }, run: async t => {
    await desk('studio', ['sampler'], { backend: 'ace' })(t);
    const r = await t.js(() => { const el = document.querySelector('.tile[data-m="sampler"] [data-knob="samp.cfg"]'); el.scrollIntoView({ block: 'center', behavior: 'instant' }); const b = el.getBoundingClientRect(); return { x: b.left + b.width / 2, y: b.top + b.height / 2 }; });
    const cdp = await t.page.context().newCDPSession(t.page);
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: r.x, y: r.y, id: 1 }] });
    for (let i = 1; i <= 10; i++) { await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: r.x, y: r.y - 12 * i, id: 1 }] }); await t.page.waitForTimeout(16); }
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] }); await t.settle(300); await cdp.detach();
    await t.ok('a toast teaches rest-then-drag', () => /Rest a finger/.test(document.querySelector('#toast .toast-msg')?.textContent || '') || (document.querySelector('#toast .toast-msg')?.textContent || 'no toast'));
  } },
  { id: 'switch-toast-says-when-length-changes-meaning', pending: { finding: 'S-37', right: 'When an engine switch carries a length whose meaning changes (an exact length becomes a ceiling, or back), the switch toast says so: "On HeartMuLa 3:20 is the longest the song may run, not its length."' }, run: async t => {
    await desk('studio', ['tempo'], { backend: 'sa3', vals: { 'tempo.dur': 200, 'tempo.dur.auto': false } })(t);
    await t.click('#engBtn'); await t.click('#engPanel [data-backend="hm"]');
    await t.ok('the toast says what 3:20 now means', () => /3:20 is the longest the song may run/.test(document.querySelector('#toast .toast-msg')?.textContent || '') || document.querySelector('#toast .toast-msg')?.textContent);
  } },
];
