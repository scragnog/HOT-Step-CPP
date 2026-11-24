// The prompt card and its tools, Generate and the run queue, the song cards, the song menu and the player.
const msg = () => document.querySelector('#toast .toast-msg')?.textContent || '';
export default [
  { id: 'prompt-suggestions', run: async t => {
    await t.click('#prompt'); await t.type('slow jam 92 bpm\n[Verse]\nla la');
    await t.ok('lyrics and a BPM in the prompt are noticed', () => !!document.querySelector('#suggest [data-sg="lyrics"]') && !!document.querySelector('#suggest [data-sg="bpm"][data-v="92"]'));
    await t.click('#suggest [data-sg="lyrics"]');
    await t.eq('Move them to the lyrics sheet', () => [get('lyrics.text'), document.querySelector('#prompt').value], ['[Verse]\nla la', 'slow jam 92 bpm']);
    await t.click('#toast .toast-act');
    await t.eq('Undo puts them back', () => document.querySelector('#prompt').value, 'slow jam 92 bpm\n[Verse]\nla la');
    await t.click('#suggest [data-sg="bpm"]');
    await t.ok('Set the tempo field adds Tempo with that BPM', () => S.mods.includes('tempo') && get('tempo.bpm') === 92 && !isAuto('tempo.bpm'));
    await t.ok('and the suggestion goes', () => !document.querySelector('#suggest [data-sg="bpm"]'));
    await t.do('placeholder', () => { document.querySelector('#prompt').value = ''; document.querySelector('#prompt').blur(); }, null, 3600);
    await t.ok('an empty prompt rotates its placeholder', () => document.querySelector('#prompt').placeholder !== 'dreamy indie folk, warm acoustic guitar, soft female vocals');
  } },
  { id: 'prompt-tools', run: async t => {
    await t.click('#prompt'); await t.type('folk {fast|slow} song');
    await t.ok('a wildcard shows Expand and Expand on Generate', () => !document.querySelector('#wildBtn').hidden && !document.querySelector('#wildAuto').hidden && /Wildcard/.test(document.querySelector('#suggest').textContent));
    await t.click('#wildAuto');
    await t.eq('Expand on Generate toggles', () => [get('wild.auto'), document.querySelector('#wildAuto').getAttribute('aria-pressed')], [true, 'true']);
    await t.click('#wildBtn');
    await t.ok('Expand picks a choice', () => /^folk (fast|slow) song$/.test(document.querySelector('#prompt').value) && document.querySelector('#wildBtn').hidden);
    await t.click('#ideasBtn');
    await t.ok('Presets & genres opens', () => !document.querySelector('#pop').hidden && document.querySelectorAll('#pop [data-preset2]').length === 6);
    await t.click('#pop [data-genre="shoegaze"]');
    await t.ok('a genre is appended', () => /, shoegaze$/.test(document.querySelector('#prompt').value));
    await t.click('#pop [data-preset2]');
    await t.ok('a style preset replaces the description, with Undo', () => /^Acoustic guitar, upright bass/.test(document.querySelector('#prompt').value) && /Replaced the description/.test(document.querySelector('#toast .toast-msg').textContent) && document.querySelector('#pop').hidden);
    await t.click('#ideasBtn'); await t.click('#ideasBtn');
    await t.ok('its button toggles the popover', () => document.querySelector('#pop').hidden);
    await t.click('#ideasBtn'); await t.click('.top .brand');
    await t.ok('a click outside closes it', () => document.querySelector('#pop').hidden);
    await t.do('empty prompt', () => { const p = document.querySelector('#prompt'); p.value = ''; p.dispatchEvent(new Event('input', { bubbles: true })); });
    await t.click('#aiBtn');
    await t.ok('Write it for me fills the caption and lyrics', () => /indie folk/.test(document.querySelector('#prompt').value) && /\[Verse\]/.test(get('lyrics.text')));
    /* F-17: with both filled it asks before replacing the lyrics, and asking adds nothing to Undo */
    await t.do('lyrics of my own', () => setVal('lyrics.text', '[Verse]\nrain on the window'));
    await t.js(() => { window.__u = UNDO.length; });
    await t.click('#aiBtn');
    await t.ok('a second time both are filled: it asks, changes nothing, adds no undo step', () => /indie folk/.test(document.querySelector('#prompt').value) && get('lyrics.text') === '[Verse]\nrain on the window' && /^Both are filled in/.test(document.querySelector('#toast .toast-msg').textContent) && UNDO.length === window.__u);
    await t.click('#toast .toast-act');
    await t.ok('Replace lyrics writes them, and says only what it did (C-29)', () => /indie folk/.test(document.querySelector('#prompt').value) && /Kettle on/.test(get('lyrics.text')) && /^Wrote new lyrics/.test(document.querySelector('#toast .toast-msg').textContent) && document.querySelector('#toast .toast-act')?.textContent === 'Undo');
    await t.click('#songInfoBtn');
    await t.ok('Song info opens with focus in it', () => !document.querySelector('#songInfo').hidden && document.activeElement.matches('#songInfo input'));
    await t.type('The Band');
    await t.eq('artist binds', () => get('song.artist'), 'The Band');
    await t.click('#songInfoBtn');
    await t.ok('and closes, marked as filled', () => document.querySelector('#songInfo').hidden && document.querySelector('#songInfoBtn').classList.contains('on'));
  } },
  /* F-22: a field's Ctrl+Z and the page's Undo share no state that can disagree; every text Ctrl+Z gives was on screen */
  { id: 'field-undo-and-page-undo', run: async t => {
    const p = () => document.querySelector('#prompt').value;
    await t.click('#prompt'); await t.type('my own words');
    await t.click('#ideasBtn'); await t.click('#pop [data-preset2]');
    await t.click('#undoBtn');
    await t.eq('the page\'s Undo puts the description back', p, 'my own words');
    await t.click('#prompt'); await t.press('End'); await t.type(', and more');
    await t.press('Control+z'); await t.press('Control+z'); await t.press('Control+z');
    await t.eq('the field\'s Ctrl+Z steps back through its own texts', p, 'my own words');
    await t.press('Control+z');
    await t.eq('and on into what the page put there', p, 'Acoustic guitar, upright bass and brushed drums, warm and intimate');
    await t.press('Control+z');
    await t.eq('and before it', p, 'my own words');
    await t.press('Control+y'); await t.press('Control+y'); await t.press('Control+Shift+z');
    await t.eq('Redo walks forward the same way', p, 'my own words,');
    await t.do('a direct write', () => { const el = document.querySelector('#prompt'); el.value = 'set from outside'; });
    await t.press('Control+z');
    await t.eq('a value set without telling is a step of its own', p, 'my own words,');
    await t.press('Control+y');
    await t.eq('and comes back with Redo', p, 'set from outside');
    await t.click('#brSearch'); await t.type('tem'); await t.press('Control+z'); await t.press('Control+z'); await t.press('Control+z');
    await t.ok('Ctrl+Z in another field never touches the description', () => document.querySelector('#prompt').value === 'set from outside' && document.querySelector('#brSearch').value === '');
  } },
  { id: 'generate', run: async t => {
    await t.click('#go');
    await t.ok('an empty prompt with no lyrics is refused with an example', () => /needs a description, some lyrics, or Instrumental/.test(document.querySelector('#toast .toast-msg').textContent) && document.querySelector('#toast .toast-act').textContent === 'Use the example');
    await t.click('#toast .toast-act');
    await t.ok('Use the example fills the prompt', () => document.querySelector('#prompt').value === document.querySelector('#prompt').placeholder);
    await t.do('takes', () => { S.tier = 'studio'; S.mods = ['takes']; S.vals['takes.n'] = 3; render(); });
    await t.click('#go', 200);
    await t.ok('Generate queues one card per take', () => S.songs.filter(s => s.pending).length === 3 && !document.querySelector('#cancelGo').hidden && document.title === '(3) HOT-Step');
    await t.focus('#prompt'); await t.press('Control+Enter', 200);
    await t.ok('Ctrl+Enter queues another run', () => S.songs.filter(s => s.pending).length === 6 && /In line · 1 ahead/.test(document.querySelector('.song.pending .stage').textContent));
    await t.click('#latest .song.pending [data-cancelsong]', 200);
    await t.ok('× on a card cancels that song', () => S.songs.filter(s => s.pending).length === 5 && /the rest carry on/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.do('run', () => {}, null, 4200);
    await t.ok('the first run finishes, the second starts', () => S.songs.filter(s => s.pending).length === 2 && !/In line/.test(document.querySelector('.song.pending .stage').textContent));
    await t.click('#cancelGo');
    await t.ok('Cancel all clears the queue', () => !S.songs.some(s => s.pending) && document.querySelector('#cancelGo').hidden && /Cancelled 2 songs/.test(document.querySelector('#toast .toast-msg').textContent));
    /* F-13: and its Undo puts the runs back, as they were started, saying the one rendering starts again */
    await t.click('#toast .toast-act');
    await t.ok('Undo of Cancel all queues the runs again', () => S.songs.filter(s => s.pending).length === 2 && S.batches.length === 1 && /starts again/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.click('#toast .toast-act');
    await t.ok('and Redo cancels them again', () => !S.songs.some(s => s.pending) && document.querySelector('#cancelGo').hidden);
    await t.do('run out', () => {}, null, 4000);
    await t.eq('finished songs are cards', () => S.songs.filter(s => s.of === 3 && !s.pending).length, 3);
    await t.ok('a take\'s number is a badge on its art, not the end of its title (C-11)', () => { const c = [...document.querySelectorAll('#latest .song')].find(x => x.querySelector('.take-no')); return !!c && /^\d\/3$/.test(c.querySelector('.take-no').textContent) && !/take/.test(c.querySelector('h3').textContent); });
    await t.ok('and the new ones are marked until played (C-45)', () => document.querySelectorAll('#latest .song.unheard').length === 3);
  } },
  { id: 'generate-engines', run: async t => {
    await t.do('mm3 too big', () => { S.backend = 'mm3'; S.tier = 'studio'; S.mods = ['takes']; S.vals['takes.n'] = 4; S.vals['model.lm'] = '8B planner · F16'; document.querySelector('#prompt').value = 'big song'; render(); });
    await t.click('#go', 300);
    /* a run certain not to fit is not started: the takes line says so (design decision P6-1; a refusal the server makes is 36-generate's a-refusal-the-server-makes) */
    await t.ok('MiniMax-Music3 does not start a run that cannot fit', () => !S.songs.some(s => s.pending) && !document.querySelector('#latest .song.errored'));
    await t.do('ace passes', () => { S.backend = 'ace'; S.vals['takes.n'] = 8; S.vals['model.lm'] = '4B · BF16'; S.vals['model.dit'] = 'XL Merge SFT+Turbo λ0.3 · Q8_0'; render(); generate(); }, null, 200);
    await t.ok('ACE-Step plans in passes, said under Generate', () => /planned in \d passes/.test(document.querySelector('#meter').textContent) || S.songs.filter(s => s.pending).length === 9);
    await t.do('cancel', () => cancelAll());
    await t.do('yue preview', () => { S.backend = 'yue'; S.mods = ['takes', 'planner']; S.vals['lm.preview'] = true; render(); generate(); });
    await t.ok('YuE2 with Preview opens the score first', () => !document.querySelector('#scoreModal').hidden);
    await t.press('Escape');
    await t.do('sa3', () => { S.backend = 'sa3'; document.querySelector('#prompt').value = ''; render(); generate(); });
    await t.ok('SA3 needs a prompt (S-05: one word per engine)', () => /Stable Audio 3 needs a prompt first/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.do('hm', () => { S.backend = 'hm'; render(); generate(); });
    await t.ok('HeartMuLa needs tags', () => /HeartMuLa needs some tags first/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.do('latent', () => { S.backend = 'ace'; S.latent = 'x.latent'; S.vals['takes.n'] = 1; document.querySelector('#prompt').value = 'go'; render(); generate(); }, null, 100);
    await t.ok('a latent run says it continues', () => /Continuing from x.latent/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.do('wild auto', () => { cancelAll(); S.vals['wild.auto'] = true; document.querySelector('#prompt').value = 'a {b|c} d'; render(); generate(); }, null, 100);
    await t.ok('wildcards expand at Generate', () => /^a (b|c) d$/.test(S.songs.find(s => s.pending).style));
  } },
  { id: 'songs-player', run: async t => {
    await t.click('#latest .song[data-id="2"] [data-play]');
    await t.ok('▶ plays the song in the player', () => S.playing === 2 && !document.querySelector('#player').hidden && !!document.querySelector('#player #ptrack'));
    await t.do('time', () => {}, null, 3000);
    await t.ok('the position moves', () => /^0:0[2-4] \//.test(document.querySelector('#ptime').textContent));
    await t.click('#ppause');
    await t.ok('pause', () => document.querySelector('#ppause').getAttribute('aria-label') === 'Play');
    await t.focus('#ptrack'); await t.press('End');
    await t.ok('End seeks to the end', () => /^3:55/.test(document.querySelector('#ptime').textContent));
    await t.press('Home'); await t.press('ArrowRight'); await t.press('Shift+ArrowRight');
    await t.ok('arrows seek, Shift further', () => /^0:35/.test(document.querySelector('#ptime').textContent));
    await t.press(' ');
    await t.ok('Space plays and pauses', () => document.querySelector('#ppause').getAttribute('aria-label') === 'Pause');
    await t.mouseDrag('#ptrack', [[20, 0]], [0.5, 0.5]);
    await t.ok('a drag on the waveform seeks', () => PL.pos > 118);
    await t.do('focus leaves for the page', () => { document.activeElement.blur(); window._y0 = scrollY; });
    await t.press(' ');
    await t.ok('Space pauses from the page too, as its tooltip says (C-27)', () => document.querySelector('#ppause').getAttribute('aria-label') === 'Play' && scrollY === window._y0);
    await t.do('near the end', () => { PL.pos = PL.s.dur - 2; PL.paused = false; paintPlayer(); }, null, 3500);
    await t.ok('at the end it stops and rests at the start, paused (C-27)', () => PL.pos === 0 && PL.paused && document.querySelector('#ppause').getAttribute('aria-label') === 'Play');
    await t.click('#pstop');
    await t.ok('■ stops', () => S.playing === null && document.querySelector('#player').hidden);
    await t.click('#latest .song[data-id="3"] [data-songmenu]');
    await t.click('#menu [data-sa="reuse"]');
    await t.eq('Reuse its prompt', () => document.querySelector('#prompt').value, 'Folk rock, acoustic guitar, brushed drums');
    await t.click('#latest .song[data-id="3"] [data-songmenu]'); await t.click('#latest .song[data-id="3"] [data-songmenu]');
    await t.ok('the ⋯ button toggles its menu', () => document.querySelector('#menu').hidden);
    await t.click('#latest .song[data-id="3"] [data-songmenu]'); await t.click('#menu [data-sa="Stem Separator"]');
    await t.ok('the other actions say where they go', () => /Stem Separator opens with "Untitled take"/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.click('#latest .song[data-id="3"] [data-songmenu]'); await t.press('Escape');
    await t.ok('Escape closes the menu and refocuses its button', () => document.querySelector('#menu').hidden && document.activeElement.matches('[data-songmenu]'));
    await t.click('#latest .song[data-id="4"] [data-songmenu]'); await t.click('.top .brand');
    await t.ok('so does a click outside', () => document.querySelector('#menu').hidden);
    await t.click('.latest .res-head [data-goto="Library"]');
    await t.ok('Open the Library opens the Library, as the top bar does (C-35)', () => document.querySelector('#app').hidden && /Library/.test(document.querySelector('#other').textContent) && document.querySelector('.top [data-nav="Library"]').getAttribute('aria-current') === 'page');
    await t.click('.top .brand');
    await t.ok('and the brand goes back to Create', () => !document.querySelector('#app').hidden && document.querySelector('.top [data-nav="create"]').getAttribute('aria-current') === 'page');
  } },
];
