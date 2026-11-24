// Package 6's own claims: Generate from press to finished card, the queue (several runs, cancelling one or all, with
// Undo), the go bar's line and Generate's words in every state, Latest's cards, the song menu, the player and the toast
// stack. Each holds on the concept page too (--page hotstep-tiers.html), except those marked pending (the concept's
// own pending findings, which the rebuild builds) and the two that need the rebuild's simulated server (mockCard).
// Checks run in the page, so they use no helper of this file. A scene writes the description both where the concept keeps it (#prompt) and where the definitions do (prompt.text).
const prompt = v => t => t.do('a description', v => { const p = document.querySelector('#prompt'); if (p) p.value = v; S.vals['prompt.text'] = v; render(); }, v);
const desk = (tier, mods, extra = {}) => t => t.do(`${tier} desk`, ([tier, mods, extra]) => { if (extra.backend && extra.backend !== S.backend) enterEngine(extra.backend); S.tier = tier; S.mods = mods; S.pedals = extra.pedals || []; Object.assign(S.vals, extra.vals || {}); render(); }, [tier, mods, extra]);

export default [
  { id: 'generate-queues-one-card-per-take', run: async t => {
    await desk('studio', ['takes'], { vals: { 'takes.n': 3 } })(t); await prompt('warm folk song')(t);
    await t.ok('at rest: Generate, no queue, Cancel hidden', () => document.querySelector('#goLabel').textContent === 'Generate' && document.querySelector('#goQ').hidden && document.querySelector('#cancelGo').hidden && document.querySelector('#go').title === 'Generate (Ctrl+Enter)');
    await t.click('#go', 200);
    await t.ok('one card per take, Cancel all with its count, the count in the title', () => S.songs.filter(s => s.pending).length === 3 && !document.querySelector('#cancelGo').hidden && document.querySelector('#cancelGo').getAttribute('aria-label') === 'Cancel all 3 songs' && document.querySelector('#qCount').textContent === '3' && /^\(3\) /.test(document.title) || `${S.songs.filter(s => s.pending).length} | ${document.title}`);
    await t.ok('the queue is a badge on Generate, and Generate keeps its label', () => /3$/.test(document.querySelector('#goQ').textContent) && !document.querySelector('#goQ').hidden && document.querySelector('#goLabel').textContent === 'Generate');
    await t.do('render a while', () => {}, null, 1200);
    await t.ok('its progress sweeps across Generate; the cards say their stage (C-26)', () => +getComputedStyle(document.querySelector('#go')).getPropertyValue('--gf') > 0.2 && /^Step \d+ of \d+$/.test(document.querySelector('#latest .song.pending .stage').textContent) && /3 running; another one queues after them\.$/.test(document.querySelector('#go').title) || document.querySelector('#go').title);
    await t.ok('a rendering card brings its colours into focus as it goes', () => { const c = document.querySelector('#latest .song.pending'); return +c.style.getPropertyValue('--f') > 0 && parseFloat(c.querySelector('.bar').style.width) > 0; });
    await t.focus('.brand'); await t.press('Control+Enter', 200);
    await t.ok('Ctrl+Enter queues another run, in line behind the first', () => S.songs.filter(s => s.pending).length === 6 && /In line · 1 ahead/.test(document.querySelector('#latest .song.pending .stage').textContent) && /3 running, 3 waiting/.test(document.querySelector('#go').title));
  } },
  { id: 'cancel-one-and-all-with-undo', run: async t => {
    await desk('studio', ['takes'], { vals: { 'takes.n': 3 } })(t); await prompt('warm folk song')(t);
    await t.click('#go', 400); await t.click('#go', 400);
    const id = await t.js(() => S.songs.filter(s => s.pending)[0].id);
    await t.click(`[data-cancelsong="${id}"]`, 200);
    await t.ok('× on a card cancels that song; the rest carry on', () => S.songs.filter(s => s.pending).length === 5 && /the rest carry on/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) && document.querySelector('#toast .toast-act')?.textContent === 'Undo');
    await t.do('the first run finishes', () => {}, null, 4200);
    await t.ok('the first run finishes, the second starts', () => S.songs.filter(s => s.pending).length === 2 && !/In line/.test(document.querySelector('#latest .song.pending .stage').textContent));
    await t.click('#cancelGo');
    await t.ok('Cancel all clears the queue, with Undo, and Cancel goes', () => S.songs.filter(s => s.pending).length === 0 && document.querySelector('#cancelGo').hidden && /Cancelled 2 songs/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) && !/^\(/.test(document.title));
    await t.click('#toast .toast-act');
    await t.ok('Undo puts them back as they were started, saying the one rendering starts again (F-13)', () => S.songs.filter(s => s.pending).length === 2 && S.batches.length === 1 && /starts again/.test((document.querySelector('#toast .toast-msg')?.textContent || '')));
    await t.click('#toast .toast-act');
    await t.ok('and Redo cancels them again', () => S.songs.filter(s => s.pending).length === 0 && document.querySelector('#cancelGo').hidden);
    await t.press('Control+z');
    await t.ok('Ctrl+Z undoes the cancel too', () => S.songs.filter(s => s.pending).length === 2);
  } },
  { id: 'a-run-ends-as-cards', run: async t => {
    await desk('studio', ['takes'], { vals: { 'takes.n': 3 } })(t); await prompt('warm folk song')(t);
    await t.click('#go', 200);
    await t.do('let it finish', () => {}, null, 4200);
    await t.eq('the songs are cards', () => S.songs.filter(s => s.of === 3 && !s.pending).length, 3);
    await t.ok('a take\'s number is a badge on its art, not the end of its title (C-11)', () => { const c = [...document.querySelectorAll('#latest .song')].find(x => x.querySelector('.take-no')); return !!c && /^\d\/3$/.test(c.querySelector('.take-no').textContent) && !/take/.test(c.querySelector('h3').textContent); });
    await t.ok('the new ones are marked until played (C-45)', () => document.querySelectorAll('#latest .song.unheard').length === 3);
    await t.ok('the card in sight is the message: no toast', () => (el => { if (!el) return false; const r = el.getBoundingClientRect(), top = document.querySelector('.top').getBoundingClientRect().bottom, g = document.querySelector('.gobar').getBoundingClientRect(), bot = g.bottom >= innerHeight - 2 ? g.top : innerHeight; return r.top >= top && r.bottom <= bot; })(document.querySelector('#latest .song.unheard')) ? !/new songs? from/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) : /3 new songs from ACE-Step 1\.5/.test((document.querySelector('#toast .toast-msg')?.textContent || '')));
    await t.ok('Generate is at rest again', () => document.querySelector('#cancelGo').hidden && document.querySelector('#goQ').hidden && document.querySelector('#go').title === 'Generate (Ctrl+Enter)');
    await t.ok('the count says what the page holds (F-27)', () => /^\d+ of your \d+ songs$|^\d+ songs$/.test(document.querySelector('#songCount').textContent));
  } },
  { id: 'a-finished-run-out-of-sight-is-said', w: 1440, h: 900, run: async t => {
    await desk('lab', ['tempo', 'sampler', 'seed', 'takes', 'adapters', 'timbre'], { vals: { 'takes.n': 1 } })(t); await prompt('late night drive')(t);
    await t.click('#go', 200);
    await t.do('the page scrolled back to its top', () => scrollTo(0, 0), null, 100);
    await t.do('let it finish', () => {}, null, 4200);
    await t.ok('the card out of sight is said in a toast', () => (el => { if (!el) return false; const r = el.getBoundingClientRect(), top = document.querySelector('.top').getBoundingClientRect().bottom, g = document.querySelector('.gobar').getBoundingClientRect(), bot = g.bottom >= innerHeight - 2 ? g.top : innerHeight; return r.top >= top && r.bottom <= bot; })(document.querySelector('#latest .song.unheard')) || /1 new song from ACE-Step 1\.5/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) || (document.querySelector('#toast .toast-msg')?.textContent || ''));
  } },
  { id: 'a-run-keeps-the-engine-it-started-on', run: async t => {
    await desk('studio', [], {})(t); await prompt('a quiet tune')(t);
    await t.click('#go', 400); await t.click('#go', 400);
    await t.do('switch to Stable Audio 3', () => switchEngine('sa3'), null, 200);
    await t.do('let both finish', () => {}, null, 8000);
    await t.ok('both songs are ACE-Step\'s', () => S.songs.slice(0, 2).every(s => s.bk === 'ace' && !s.pending) && [...document.querySelectorAll('#latest .song')].slice(0, 2).every(c => c.querySelector('.eng-mark')?.getAttribute('aria-label') === 'ACE-Step 1.5'));
  } },
  { id: 'an-empty-box-is-refused-with-an-example', run: async t => {
    await t.click('#go');
    await t.ok('ACE-Step: a description, some lyrics, or Instrumental, with an example', () => /needs a description, some lyrics, or Instrumental/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) && document.querySelector('#toast .toast-act').textContent === 'Use the example' && S.songs.filter(s => s.pending).length === 0);
    await t.click('#toast .toast-act');
    await t.ok('Use the example fills the box', () => !!(document.querySelector('#prompt')?.value || get('prompt.text')));
    await desk('studio', [], { backend: 'sa3' })(t); await prompt('')(t);
    await t.click('#go');
    await t.ok('Stable Audio 3 needs a prompt (S-05)', () => /Stable Audio 3 needs a prompt first/.test((document.querySelector('#toast .toast-msg')?.textContent || '')));
    await desk('studio', [], { backend: 'hm' })(t);
    await t.click('#go');
    await t.ok('HeartMuLa needs some tags', () => /HeartMuLa needs some tags first/.test((document.querySelector('#toast .toast-msg')?.textContent || '')));
  } },
  { id: 'the-line-says-the-desk', run: async t => {
    await desk('studio', ['tempo', 'sampler'])(t);
    await t.eq('the tier, the modules, and no changes', () => document.querySelector('#meter').textContent, 'Studio · 2 modules · everything on defaults');
    await t.do('two changes', () => { setVal('samp.cfg', 12); setVal('tempo.sig', '3/4'); });
    await t.ok('the changes are a link that points at them', () => document.querySelector('#meter [data-showchanges]')?.textContent === '2 changes' && /, the rest on defaults$/.test(document.querySelector('#meter').textContent));
    await t.ok('the line is its own title', () => document.querySelector('#meter').title === document.querySelector('#meter').textContent);
    await t.click('#meter [data-showchanges]');
    await t.ok('pressing it pulses the changed controls', () => document.querySelectorAll('#bento .ping').length >= 2);
    await desk('quick', ['sampler', 'latent', 'perf', 'repetition', 'guidance', 'planner'], { backend: 'mm3' })(t);
    await t.ok('hidden and parked are counted (E-13)', () => /2 hidden in Quick/.test(document.querySelector('#meter').textContent) && /2 parked/.test(document.querySelector('#meter').textContent) && !!document.querySelector('#meter .l'));
  } },
  { id: 'the-line-says-when-a-run-will-not-fit', run: async t => {
    await desk('studio', ['takes'], { backend: 'mm3', vals: { 'takes.n': 4, 'model.lm': '8B planner · F16' } })(t);
    await t.ok('a run that will be refused says so where Generate is, with the way out (G-B-08)', () => /^Won't fit: needs \d+\.\d of 31\.8 GB\. Use \d takes · Studio/.test(document.querySelector('#meter').textContent) && document.querySelector('#go').classList.contains('wontfit'));
    await t.click('#meter [data-takesfit]');
    await t.ok('Use N takes sets the takes that fit', () => +get('takes.n') < 4 && !document.querySelector('#go').classList.contains('wontfit') && !document.querySelector('#meter .m-warn'));
    await desk('studio', ['takes'], { backend: 'ace', vals: { 'takes.n': 8, 'model.lm': '4B · BF16', 'model.dit': 'XL Merge SFT+Turbo λ0.3 · Q8_0' } })(t);
    await t.ok('ACE-Step plans in passes, said under Generate', () => /8 takes, planned in \d passes/.test(document.querySelector('#meter').textContent) || document.querySelector('#meter').textContent);
  } },
  { id: 'a-run-certain-not-to-fit-is-not-started', pending: { finding: 'F-21', right: 'When the next run is certain not to fit, Generate does not start it: the "Use N takes" link pulses and the line is announced.' }, run: async t => {
    await desk('studio', ['takes'], { backend: 'mm3', vals: { 'takes.n': 4, 'model.lm': '8B planner · F16' } })(t); await prompt('big song')(t);
    await t.click('#go', 300); await t.click('#go', 300);
    await t.ok('no run started and no notice', () => S.songs.filter(s => s.pending).length === 0 && !document.querySelector('#latest .song.errored') || `${S.songs.filter(s => s.pending).length} pending, ${document.querySelectorAll('#latest .song.errored').length} notices`);
    await t.ok('the way out pulses and the line is said', () => document.querySelector('#meter [data-takesfit]').classList.contains('ping') && /Won't fit/.test(document.querySelector('#say').textContent));
  } },
  { id: 'a-refusal-the-server-makes', run: async t => {
    if (await t.js(() => typeof mockCard !== 'function')) return; /* the rebuild's simulated server only */
    await desk('studio', ['takes'], { backend: 'mm3', vals: { 'takes.n': 2, 'model.lm': '8B planner · F16', 'model.dit': 'Flow DiT · F16', 'model.x': 'Depth decoder · F16' } })(t); await prompt('big song')(t);
    await t.do('the server\'s card is smaller than the page thought', () => mockCard(30));
    await t.click('#go', 300);
    await t.ok('the run is a one-line notice across the top of Latest, with Retry and Dismiss (G-D-09)', () => { const n = document.querySelector('#latest .song.errored'); return !!n && /^Didn't run/.test(n.querySelector('.err-line').textContent) && /Needs about 30\.6 GB for 2 takes at once; the card has 30 GB\./.test(n.textContent) && n.querySelector('[data-retry]')?.textContent === 'Retry with 1 take' && !!n.querySelector('[data-dismiss]') || n?.textContent; });
    await t.ok('in sight, it is said, not toasted', () => /MiniMax-Music3 refused: 2 takes don't fit at once/.test(document.querySelector('#say').textContent) && !/refused/.test((document.querySelector('#toast .toast-msg')?.textContent || '')));
    await t.click('#latest .song.errored [data-retry]', 300);
    await t.ok('Retry runs with the takes that fit, and focus goes to Generate (F-13)', () => !document.querySelector('#latest .song.errored') && S.songs.filter(s => s.pending).length === 1 && +get('takes.n') === 1 && document.activeElement?.id === 'go');
    await t.do('finish, and refuse twice', () => { S.vals['takes.n'] = 2; render(); generate(); }, null, 4200);
    await t.do('again', () => generate(), null, 300);
    await t.eq('a refusal never stacks a second identical notice (F-21)', () => document.querySelectorAll('#latest .song.errored').length, 1);
    await t.click('#latest .song.errored [data-dismiss]');
    await t.ok('Dismiss removes it, and focus goes to Generate (F-13)', () => !document.querySelector('#latest .song.errored') && document.activeElement?.id === 'go');
    await t.do('the server\'s own card again', () => mockCard(undefined));
  } },
  { id: 'a-run-that-stops-part-way', run: async t => {
    if (await t.js(() => typeof mockFail !== 'function')) return; /* the rebuild's simulated server only */
    await desk('studio', ['takes'], { vals: { 'takes.n': 2 } })(t); await prompt('warm folk song')(t);
    await t.do('the next run will stop half-way', () => mockFail());
    await t.click('#go', 2400);
    await t.ok('it is a notice with Run it again and Dismiss, and nothing is left rendering', () => { const n = document.querySelector('#latest .song.errored'); return !!n && /ACE-Step 1\.5 stopped part-way through the run\./.test(n.textContent) && n.querySelector('[data-retry]')?.textContent === 'Run it again' && !!n.querySelector('[data-dismiss]') && S.songs.filter(s => s.pending).length === 0 || n?.textContent; });
    await t.click('#latest .song.errored [data-retry]', 300);
    await t.ok('Run it again queues it as it was started, focus on Generate', () => !document.querySelector('#latest .song.errored') && S.songs.filter(s => s.pending).length === 2 && document.activeElement?.id === 'go');
  } },
  { id: 'cancel-puts-focus-on-generate', pending: { finding: 'F-13', right: 'After Cancel on a queued song focus goes to Generate, never to the page or to the next song\'s Cancel.' }, run: async t => {
    await prompt('warm folk song')(t); await t.click('#go', 400); await t.click('#go', 400); await t.click('#go', 400);
    const id = await t.js(() => S.songs.filter(s => s.pending)[1].id);
    await t.click(`[data-cancelsong="${id}"]`);
    await t.ok('focus on Generate', () => document.activeElement?.id === 'go' || document.activeElement?.outerHTML.slice(0, 80));
  } },
  { id: 'separate-presses-each-queue', pending: { finding: 'F-22', right: 'Every separate press of Generate queues a run; only the second click of one double-click is ignored.' }, run: async t => {
    await prompt('warm folk song')(t);
    const r = await t.js(() => { const b = document.querySelector('#go').getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2]; });
    await t.page.mouse.click(r[0], r[1]); await t.page.clock.runFor(310); await t.page.mouse.click(r[0], r[1]); await t.settle(300);
    await t.eq('two presses 310ms apart queue two runs', () => S.batches.length, 2);
    await t.page.mouse.click(r[0], r[1], { clickCount: 2 }); await t.settle(300);
    await t.eq('a double-click queues one', () => S.batches.length, 3);
  } },
  { id: 'leaving-while-rendering-asks-first', pending: { finding: 'F-37', right: 'While a song is rendering or queued, leaving the page asks first (beforeunload).' }, run: async t => {
    await prompt('warm folk song')(t);
    await t.ok('nothing to ask at rest', () => { const e = new Event('beforeunload', { cancelable: true }); dispatchEvent(e); return !e.defaultPrevented; });
    await t.click('#go');
    await t.ok('beforeunload is refused while a run is live', () => { const e = new Event('beforeunload', { cancelable: true }); dispatchEvent(e); return e.defaultPrevented || 'not prevented'; });
  } },
  { id: 'a-reload-says-what-it-lost', pending: { finding: 'F-27', right: 'After a reload that dropped a song still rendering, one toast says so.' }, run: async t => {
    await prompt('warm folk song')(t); await t.click('#go'); await t.reload();
    await t.ok('one toast says a song was lost', () => /^1 song was still rendering when the page closed and wasn't kept\.$/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) || (document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.ok('and the songs the server kept are there', () => !S.songs.some(s => s.pending) && S.songs.length >= 6);
  } },
  { id: 'the-cards-in-every-state', w: 1920, h: 1080, run: async t => {
    await t.ok('a finished card: engine mark, length, Play, its menu button, its whole description on two lines (D-26, F-14)', () => { const c = document.querySelector('#latest .song[data-id="2"]'); return c.querySelector('.eng-mark').getAttribute('aria-label') === 'Stable Audio 3' && c.querySelector('.dur').textContent === '3:56' && c.querySelector('[data-play]').getAttribute('aria-label') === 'Play Evening texture' && c.querySelector('[data-songmenu]').getAttribute('aria-label') === 'Actions for Evening texture' && c.querySelector('.meta p').title === 'Ambient, warm pads, slow evolving textures · Stable Audio 3'; });
    await t.ok('whole rows only (D-26)', () => { const g = document.querySelector('#latest'), cols = getComputedStyle(g).gridTemplateColumns.split(' ').length, n = [...g.children].filter(x => !x.hidden).length; return n % cols === 0 || n < cols || `${n} shown, ${cols} columns`; });
    await desk('studio', ['takes'], { vals: { 'takes.n': 4 } })(t); await prompt('warm folk song')(t);
    await t.click('#go', 400); await t.click('#go', 400); await t.click('#go', 400);
    await t.ok('a long queue folds behind a card that says how many wait (C-47)', () => { const f = document.querySelector('#latest .qfold'); return !!f && /^\+\d+ waiting/.test(f.textContent) && f.getAttribute('aria-expanded') === 'false' && document.querySelectorAll('#latest .song.pending.folded').length > 0; });
    await t.click('#latest .qfold');
    await t.ok('and opens', () => document.querySelector('#latest .qfold').getAttribute('aria-expanded') === 'true' && /^Fold the queue/.test(document.querySelector('#latest .qfold').textContent) && !document.querySelector('#latest .song.pending.folded:not([hidden])') && [...document.querySelectorAll('#latest .song.pending')].every(x => !x.hidden));
  } },
  { id: 'play-pause-seek-and-stop', run: async t => {
    await t.click('#latest .song[data-id="2"] [data-play]');
    await t.ok('▶ plays the song in the player, in the go bar', () => S.playing === 2 && !document.querySelector('#player').hidden && !!document.querySelector('.gobar #player #ptrack') && document.querySelector('#player').getAttribute('aria-label') === 'Player' && document.querySelector('#latest .song[data-id="2"]').classList.contains('playing'));
    await t.do('time', () => {}, null, 3000);
    await t.ok('the position moves', () => /^0:0[2-4] \/ 3:56$/.test(document.querySelector('#ptime').textContent));
    await t.click('#ppause');
    await t.ok('pause', () => document.querySelector('#ppause').getAttribute('aria-label') === 'Play' && document.querySelector('#ppause').title === 'Play (Space)');
    await t.focus('#ptrack'); await t.press('End');
    await t.ok('End seeks to the end', () => /^3:55/.test(document.querySelector('#ptime').textContent) && /^3:55 of 3:56$/.test(document.querySelector('#ptrack').getAttribute('aria-valuetext')));
    await t.press('Home'); await t.press('ArrowRight'); await t.press('Shift+ArrowRight');
    await t.ok('arrows seek, Shift further', () => /^0:35/.test(document.querySelector('#ptime').textContent));
    await t.press(' ');
    await t.ok('Space plays and pauses', () => document.querySelector('#ppause').getAttribute('aria-label') === 'Pause');
    await t.do('focus leaves for the page', () => { document.activeElement.blur(); window._y0 = scrollY; });
    await t.press(' ');
    await t.ok('Space pauses from the page too (C-27)', () => document.querySelector('#ppause').getAttribute('aria-label') === 'Play' && scrollY === window._y0);
    await t.do('near the end', () => { PL.pos = PL.s.dur - 2; PL.paused = false; paintPlayer(); }, null, 3500);
    await t.ok('at the end it stops and rests at the start, paused (C-27)', () => PL.pos === 0 && PL.paused && document.querySelector('#ppause').getAttribute('aria-label') === 'Play');
    await t.click('#pstop');
    await t.ok('■ stops and closes it', () => S.playing === null && document.querySelector('#player').hidden);
  } },
  { id: 'the-song-menu', run: async t => {
    await t.click('#latest .song[data-id="1"] [data-songmenu]');
    await t.ok('it opens with focus on its first item, named for its song', () => !document.querySelector('#menu').hidden && document.activeElement.dataset.sa === 'repaint' && document.querySelector('#menu').getAttribute('aria-label') === 'Actions for Studio sketch 01' && document.querySelector('#latest .song[data-id="1"] [data-songmenu]').getAttribute('aria-expanded') === 'true');
    await t.eq('its items', () => [...document.querySelectorAll('#menu [role="menuitem"]')].map(b => b.dataset.sa), ['repaint', 'reuse', 'Cover Studio', 'Song Builder', 'Compare', 'Stem Separator', 'MIDI Studio']);
    await t.press('ArrowDown'); await t.eq('arrows move in the menu', () => document.activeElement.dataset.sa, 'reuse');
    await t.press('End'); await t.eq('End', () => document.activeElement.dataset.sa, 'MIDI Studio');
    await t.press('Home'); await t.eq('Home', () => document.activeElement.dataset.sa, 'repaint');
    await t.press('Escape');
    await t.ok('Escape closes it and refocuses its button', () => document.querySelector('#menu').hidden && document.activeElement.matches('[data-songmenu]'));
    await t.click('#latest .song[data-id="3"] [data-songmenu]'); await t.click('#latest .song[data-id="3"] [data-songmenu]');
    await t.ok('the ⋯ button toggles it', () => document.querySelector('#menu').hidden);
    await t.click('#latest .song[data-id="3"] [data-songmenu]'); await t.click('#menu [data-sa="Stem Separator"]');
    await t.ok('the other pages say where they go', () => /Stem Separator opens with "Untitled take"/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) && document.querySelector('#menu').hidden);
    await t.click('#latest .song[data-id="4"] [data-songmenu]'); await t.click('.top .brand');
    await t.ok('a click outside closes it', () => document.querySelector('#menu').hidden);
  } },
  { id: 'reuse-puts-the-words-back', run: async t => {
    await prompt('something else')(t);
    await t.click('#latest .song[data-id="3"] [data-songmenu]');
    await t.ok('the item names the box', () => /^Reuse its description/.test(document.querySelector('#menu [data-sa="reuse"]').textContent));
    await t.click('#menu [data-sa="reuse"]');
    await t.ok('the words go back in the box, one step, and the toast says what was replaced, with Undo', () => (document.querySelector('#prompt')?.value ?? get('prompt.text')) === 'Folk rock, acoustic guitar, brushed drums' && /Replaced the description/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) && document.querySelector('#toast .toast-act')?.textContent === 'Undo');
  } },
  { id: 'repaint-from-the-song-menu', run: async t => {
    await desk('studio', ['tempo'], { backend: 'mm3' })(t);
    await t.click('#latest .song[data-id="1"] [data-songmenu]');
    await t.ok('on MiniMax-Music3 Repaint says it switches (E-14)', () => /Switches to ACE-Step 1.5/.test(document.querySelector('#menu [data-sa="repaint"]').textContent));
    await t.click('#menu [data-sa="repaint"]');
    await t.ok('one press: ACE-Step with Repaint first on the desk and the song as its source, and Generate repaints', () => S.backend === 'ace' && S.mods[0] === 'repaint' && S.source?.song === 1 && /On ACE-Step 1\.5\..*Repaint is on the desk/.test(document.querySelector('#toast').textContent) && document.querySelector('#goLabel').textContent === 'Repaint region');
    await t.press('Control+z');
    await t.ok('and Undo puts back MiniMax-Music3 without Repaint', () => S.backend === 'mm3' && !S.mods.includes('repaint') && document.querySelector('#goLabel').textContent === 'Generate');
  } },
  { id: 'the-vram-meter-follows-the-run', run: async t => {
    await desk('studio', [], {})(t); await prompt('a quiet tune')(t);
    const rest = await t.js(() => document.querySelectorAll('#emBar i.on').length);
    await t.click('#go', 1200);
    await t.ok('while a run renders the needle swings toward what it needs, and the engine says it is live', r => document.querySelectorAll('#emBar i.on').length > r && document.querySelector('#engine').classList.contains('lg-live') || `${document.querySelectorAll('#emBar i.on').length} lit, ${r} at rest`, rest);
    await t.do('another engine selected while it renders', () => switchEngine('sa3'), null, 200);
    await t.ok('the meter says the engine that renders holds the card, and what the next Generate needs (B-34)', () => /^VRAM: ACE-Step 1\.5 is rendering with \d+\.\d GB loaded\. Next: Stable Audio 3, about \d+\.\d GB, on a 31\.8 GB card$/.test(document.querySelector('#engMeter').getAttribute('aria-label')) || document.querySelector('#engMeter').getAttribute('aria-label'));
    await t.do('let it finish', () => {}, null, 4200);
    await t.ok('then the meter is the selected engine\'s again, at rest', () => /^VRAM: \d+\.\d GB loaded/.test(document.querySelector('#engMeter').getAttribute('aria-label')) && !document.querySelector('#engine').classList.contains('lg-live'));
  } },
  { id: 'the-dice-and-the-cover-follow-the-runs', run: async t => {
    await desk('lab', ['replan', 'post'], { backend: 'yue', pedals: ['art'], vals: { 'yue.replan': true, 'yue.tries': 4, 'pp.art.on': true } })(t); await prompt('a long folk ballad')(t);
    await t.ok('no die is used before a run', () => document.querySelectorAll('[data-dice-box] .die').length === 4 && !document.querySelector('[data-dice-box] .die.used'));
    await t.click('#go', 1200);
    await t.ok('the dice light up as the run re-plans', () => document.querySelectorAll('[data-dice-box] .die.used').length >= 1);
    await t.do('let it finish', () => {}, null, 4200);
    await t.ok('the cover swatch shows the newest song\'s colour', () => +getComputedStyle(document.querySelector('.u-art')).getPropertyValue('--h') === S.songs[0].hue || `${getComputedStyle(document.querySelector('.u-art')).getPropertyValue('--h')} vs ${S.songs[0].hue}`);
  } },
  { id: 'the-toast-stack', run: async t => {
    await t.do('a message with Undo, then a plain one', () => { toast('First, with Undo', { label: 'Undo', fn: () => { window.__undo = 1; } }); toast('Second'); }, null, 100);
    await t.ok('the first moves up and keeps its Undo (F-12)', () => { const o = document.querySelector('.toast-old'); return !!o && /First, with Undo/.test(o.textContent) && !!o.querySelector('.toast-act') && (document.querySelector('#toast .toast-msg')?.textContent || '') === 'Second'; });
    await t.ok('it stands above the newest', () => document.querySelector('.toast-old').getBoundingClientRect().bottom <= document.querySelector('#toast').getBoundingClientRect().top);
    await t.click('.toast-old .toast-act');
    await t.ok('its Undo still works', () => window.__undo === 1);
    await t.do('the same plain message twice', () => { toast('Again'); toast('Again'); }, null, 100);
    await t.ok('is said once', () => ![...document.querySelectorAll('.toast-old')].some(o => /Again/.test(o.textContent)));
  } },
  { id: 'a-plain-message-never-pushes-out-an-undo', pending: { finding: 'F-18', right: 'A plain message never pushes out one that carries an action (Undo); and Escape on a toast\'s button dismisses that toast.' }, run: async t => {
    await t.do('one with Undo, then three plain ones', () => { toast('Removed it', { label: 'Undo', fn: () => {} }); toast('One'); toast('Two'); toast('Three'); }, null, 100);
    await t.ok('the Undo toast is still there', () => [...document.querySelectorAll('#toast, .toast-old')].some(el => /Undo/.test(el.querySelector('.toast-act')?.textContent || '')));
    await t.ok('three show at most', () => document.querySelectorAll('.toast-old').length <= 2);
    await t.do('one with an action', () => toast('Removed something', { label: 'Undo', fn: () => {} }));
    await t.focus('#toast .toast-act'); await t.press('Escape');
    await t.ok('Escape on its button dismisses it', () => document.querySelector('#toast').hidden);
  } },
  { id: 'tab-reaches-a-toasts-undo', run: async t => {
    await t.focus('#tiers [aria-checked="true"]');
    await t.do('a message with Undo', () => toast('Removed something', { label: 'Undo', fn: () => {} }));
    await t.press('Tab');
    await t.ok('the first Tab goes to its Undo (C-13)', () => document.activeElement?.classList.contains('toast-act'));
    await t.press('Shift+Tab');
    await t.ok('Shift+Tab goes back to where focus was', () => document.activeElement?.matches('#tiers [aria-checked="true"]'));
  } },
];
