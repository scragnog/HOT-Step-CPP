// Overlays and the keyboard: the picker, notes, engine log, drafts, toasts, Escape layering, Tab traps, the theme,
// the other pages, tooltips without a mouse, the focus ring, held keys.
export default [
  { id: 'picker', w: 1600, run: async t => {
    await t.do('studio', () => { S.tier = 'studio'; S.mods = ['tempo']; render(); });
    await t.click('[data-openpicker]');
    await t.ok('the Picker button opens it, focus in the filter', () => !document.querySelector('#pickerWrap').hidden && document.activeElement.id === 'pkSearch');
    await t.press('Escape');
    await t.ok('Escape closes it and returns focus', () => document.querySelector('#pickerWrap').hidden && !!document.activeElement.closest('[data-openpicker]'));
    await t.press('F9');
    await t.ok('F9 opens it', () => !document.querySelector('#pickerWrap').hidden);
    await t.type('see');
    await t.ok('typing filters and ranks', () => document.querySelector('#pkCount').textContent === '1 match' && !!document.querySelector('.fl-it[data-pkid="seed"]:not(.nomatch)'));
    await t.press('Enter');
    await t.ok('Enter adds the first match and keeps picking', () => S.mods.includes('seed') && /Added Seed/.test(document.querySelector('#pkInfo').textContent));
    await t.press('Escape');
    await t.ok('Escape with a filter clears the filter first', () => document.querySelector('#pkSearch').value === '' && !document.querySelector('#pickerWrap').hidden);
    await t.click('#pkBar [data-pkcat="Pedals"]');
    await t.ok('a category narrows the wall', () => S.pkCat === 'Pedals' && document.querySelectorAll('#pkLeft .fl-it.dim').length > 0);
    await t.click('#pkBar [data-pkcat="All"]');
    await t.focus('#pkSearch'); await t.press('ArrowDown');
    await t.ok('ArrowDown goes into the wall', () => document.activeElement.matches('.fl-it'));
    const first = await t.js(() => document.activeElement.dataset.pkid);
    await t.press('ArrowRight');
    await t.ok('arrows move between snapshots', id => document.activeElement.dataset.pkid !== id && document.activeElement.matches('.fl-it'), first);
    await t.press('End'); await t.press('Home');
    await t.ok('Home and End', id => document.activeElement.dataset.pkid === id, first);
    await t.press('Tab');
    await t.ok('Tab cycles inside the picker', () => document.querySelector('#pickerWrap').contains(document.activeElement));
    await t.focus('.fl-it[data-pkid="avoid"]'); await t.press(' ');
    await t.ok('Space adds the focused snapshot', () => S.mods.includes('avoid'));
    await t.focus('.fl-it[data-pkid="guidance"]'); await t.press('Enter');
    await t.ok('a Lab module offers Lab in the info bar', () => !!document.querySelector('#pkInfo [data-labadd="guidance"]') && document.activeElement.matches('#pkInfo [data-labadd]'));
    await t.press('Enter');
    await t.ok('which switches to Lab and adds it', () => S.tier === 'lab' && S.mods.includes('guidance'));
    await t.click('.fl-it[data-pkid="tempo"]');
    await t.ok('a click on one already on the desk closes the picker and shows it', () => document.querySelector('#pickerWrap').hidden);
    await t.press('F9'); await t.hover('.fl-it[data-pkid="timbre"]');
    await t.ok('pointing shows its description', () => /Timbre reference/.test(document.querySelector('#pkInfo').textContent));
    await t.press('Control+Enter');
    await t.ok('Ctrl+Enter does not generate while the picker is open', () => !S.songs.some(s => s.pending));
    await t.click('#pkClose');
    await t.ok('× closes it', () => document.querySelector('#pickerWrap').hidden);
    await t.press('F9'); await t.click('#pickerWrap .fl-col h2');
    await t.ok('a click on the backdrop closes it', () => document.querySelector('#pickerWrap').hidden);
    await t.press('F9'); await t.press('F9');
    await t.ok('F9 toggles it', () => document.querySelector('#pickerWrap').hidden);
  } },
  { id: 'notes-logs-escape', run: async t => {
    await t.click('#notesBtn');
    await t.ok('Concept notes open as a modal drawer', () => !document.querySelector('#notes').hidden && !document.querySelector('#notesScrim').hidden && document.activeElement.id === 'notesClose');
    await t.press('Shift+Tab');
    await t.ok('Shift+Tab wraps inside it', () => document.querySelector('#notes').contains(document.activeElement));
    await t.press('Escape');
    await t.ok('Escape closes it, focus back on its button', () => document.querySelector('#notes').hidden && document.activeElement.id === 'notesBtn');
    await t.click('#notesBtn'); await t.click('#notesScrim', 300, [0.1, 0.5]);
    await t.ok('the scrim closes it', () => document.querySelector('#notes').hidden);
    await t.click('#notesBtn'); await t.click('#notesClose');
    await t.ok('so does ×', () => document.querySelector('#notes').hidden);
    await t.click('#engMeter');
    await t.ok('the VRAM meter opens the engine log', () => !document.querySelector('#logs').hidden && document.querySelector('#engMeter').getAttribute('aria-expanded') === 'true' && /ace-server up/.test(document.querySelector('#logText').textContent));
    await t.click('#engMeter');
    await t.ok('and toggles it', () => document.querySelector('#logs').hidden);
    /* F-15: the log is a dock under the go bar; a click elsewhere leaves it, Ctrl+Enter still generates */
    await t.click('#engMeter'); await t.click('.top .brand');
    await t.ok('a click outside leaves it open, and the go bar stands on it', () => !document.querySelector('#logs').hidden && document.querySelector('.gobar').getBoundingClientRect().bottom <= document.querySelector('#logs').getBoundingClientRect().top + 1);
    await t.click('#prompt'); await t.type('dock test'); await t.press('Control+Enter', 200);
    await t.ok('Ctrl+Enter generates while it is open', () => S.songs.some(s => s.pending) && !document.querySelector('#logs').hidden);
    await t.click('#cancelGo');
    await t.click('#logsClose');
    await t.ok('Close closes it', () => document.querySelector('#logs').hidden && document.activeElement.id === 'engMeter');
    await t.click('#engMeter'); await t.press('Escape');
    await t.ok('and so does Escape', () => document.querySelector('#logs').hidden);
    await t.do('studio', () => { S.tier = 'studio'; S.mods = ['sampler']; render(); });
    await t.click('#engBtn'); await t.click('#engPanel [data-pick="model.dit"]');
    await t.press('Escape');
    await t.ok('Escape closes only the top layer: the dropdown', () => document.querySelector('#dd').hidden && !document.querySelector('#engPanel').hidden);
    await t.press('Escape');
    await t.ok('then the panel', () => document.querySelector('#engPanel').hidden);
    /* F-27: Escape somewhere else leaves the save row and what was typed in it; Escape in it closes it */
    await t.click('#saveBtn'); await t.type('half'); await t.click('#prompt'); await t.press('Escape');
    await t.ok('Escape outside the save row leaves it as it was', () => !document.querySelector('#saveRow').hidden && document.querySelector('#saveName').value === 'half' && document.activeElement.id === 'prompt');
    await t.click('[data-saveas="wf"]'); await t.press('Escape');
    await t.ok('then the save row, from inside it', () => document.querySelector('#saveRow').hidden && document.activeElement.id === 'saveBtn');
  } },
  { id: 'drafts', run: async t => {
    await t.click('#prompt'); await t.type('first draft');
    await t.click('#draftsBtn');
    await t.ok('Drafts opens above its button', () => !document.querySelector('#pop').hidden && document.querySelector('#pop').classList.contains('fixedpop') && /No drafts yet/.test(document.querySelector('#pop').textContent));
    await t.click('#pop [data-draftsave]');
    await t.ok('Save draft keeps the prompt', () => S.drafts.length === 1 && S.drafts[0].prompt === 'first draft' && JSON.parse(localStorage.getItem('hs3-drafts')).length === 1);
    await t.click('#draftsBtn');
    await t.ok('the button toggles it', () => document.querySelector('#pop').hidden);
    await t.do('change engine and prompt', () => { S.backend = 'yue'; document.querySelector('#prompt').value = 'other'; render(); });
    await t.click('#draftsBtn');
    await t.ok('on YuE2 a draft can be named', () => !!document.querySelector('#draftName'));
    await t.click('#draftName'); await t.type('named one'); await t.press('Enter');
    await t.eq('Enter in the name saves it', () => [S.drafts.length, S.drafts[0].name, S.drafts[0].backend], [2, 'named one', 'yue']);
    await t.eq('every row is one height, whatever its engine (C-06)', () => [...document.querySelectorAll('#pop .draft')].map(r => Math.round(r.getBoundingClientRect().height)), [44, 44]);
    await t.ok('another engine\'s draft says so on its second line, at full strength, with the same Load button', () => { const r = document.querySelectorAll('#pop .draft')[1]; return /ACE-Step 1\.5/.test(r.querySelector('small').textContent) && getComputedStyle(r).opacity === '1' && r.querySelector('[data-draftload]').textContent === 'Load'; });
    await t.ok('Save draft rests while nothing changed since the last one', () => document.querySelector('#pop [data-draftsave]').disabled);
    await t.click('#pop [data-draftload="1"]');
    await t.ok('Load on another engine switches engine, with Undo', () => S.backend === 'ace' && document.querySelector('#prompt').value === 'first draft' && /Loaded the draft on ACE-Step 1.5$/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.click('#toast .toast-act');
    await t.eq('Undo restores the engine and prompt', () => [S.backend, document.querySelector('#prompt').value], ['yue', 'other']);
    await t.click('#draftsBtn'); await t.click('#pop [data-draftdel="0"]');
    await t.ok('delete acts at once, with Undo (it is undoable, so it does not ask)', () => S.drafts.length === 1 && /Deleted the draft/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.ok('focus stays in the open list', () => !document.querySelector('#pop').hidden && !!document.activeElement.closest('#pop'));
    await t.click('#toast .toast-act');
    await t.eq('Undo restores it', () => S.drafts.length, 2);
    await t.ok('and focus goes back into the list, not to Generate (C-25)', () => !document.querySelector('#pop').hidden && !!document.activeElement.closest('#pop'));
    await t.do('fill to the limit', () => { for (let i = 0; i < 28; i++) S.drafts.push({ name: 'd' + i, backend: 'ace', at: Date.now() - 1000 - i, prompt: 'p', vals: {} }); saveDrafts(); document.querySelector('#prompt').value = 'third'; });
    await t.click('#toast .toast-x');
    await t.click('#draftsBtn');
    await t.ok('Drafts toggles shut (it stayed open through the toast\'s Undo)', () => document.querySelector('#pop').hidden);
    await t.click('#draftsBtn'); await t.click('#pop [data-draftsave]');
    await t.ok('past thirty the oldest makes room, with Undo', () => ({ ok: S.drafts.length === 30 && /made room/.test(document.querySelector('#toast .toast-msg').textContent), n: S.drafts.length, pop: document.querySelector('#pop').hidden, msg: document.querySelector('#toast .toast-msg')?.textContent }));
  } },
  { id: 'toasts', run: async t => {
    await t.do('a toast', () => toast('Hello there'));
    await t.ok('a toast shows', () => !document.querySelector('#toast').hidden && document.querySelector('#toast .toast-msg').textContent === 'Hello there');
    await t.hover('#toast');
    await t.do('wait while hovered', () => {}, null, 9000);
    await t.ok('it waits while hovered', () => !document.querySelector('#toast').hidden);
    await t.hover('.top .brand');
    await t.do('wait after', () => {}, null, 5000);
    await t.ok('and goes after', () => document.querySelector('#toast').hidden);
    await t.do('another', () => toast('With an action', { label: 'Do it', fn: () => { window.__did = 1; } }));
    await t.click('#toast .toast-act');
    await t.ok('its action runs and the toast closes', () => window.__did === 1 && document.querySelector('#toast').hidden);
    await t.do('third', () => toast('Dismiss me'));
    await t.click('#toast .toast-x');
    await t.ok('× dismisses it', () => document.querySelector('#toast').hidden);
    await t.do('fourth', () => { toast('One'); toast('Two, a little longer'); });
    await t.eq('a replaced message shows the new text', () => document.querySelector('#toast .toast-msg').textContent, 'Two, a little longer');
  } },
  { id: 'theme-nav', run: async t => {
    await t.click('#themeSeg [data-themeset="light"]');
    await t.eq('the Light segment sets Light (C-40: three segments, each one press away)', () => [document.documentElement.dataset.theme, localStorage.getItem('hs3-theme'), document.querySelector('#themeSeg [aria-checked="true"]').dataset.themeset], ['light', 'light', 'light']);
    await t.click('#themeSeg [data-themeset="dark"]');
    await t.eq('Dark is one press from Light', () => document.documentElement.dataset.theme, 'dark');
    await t.reload();
    await t.eq('the theme is kept', () => [document.documentElement.dataset.theme, document.querySelector('#themeSeg [aria-checked="true"]').dataset.themeset], ['dark', 'dark']);
    await t.focus('#themeSeg [data-themeset="dark"]'); await t.press('ArrowRight');
    await t.eq('arrow keys move round to System', () => [document.documentElement.dataset.theme, localStorage.getItem('hs3-theme'), document.activeElement.dataset.themeset], [undefined, null, 'system']);
    for (const nav of ['Library', 'Studios', 'Training', 'Settings']) {
      await t.click(`[data-nav="${nav}"]`);
      await t.ok(`${nav} shows its own page`, nav => document.querySelector('#app').hidden && !document.querySelector('#other').hidden && document.querySelector('#other h2').textContent === nav && document.querySelector(`[data-nav="${nav}"]`).getAttribute('aria-current') === 'page', nav);
    }
    await t.press('F9');
    await t.ok('F9 does nothing off the Create page', () => document.querySelector('#pickerWrap').hidden);
    await t.press('Control+Enter');
    await t.ok('nor does Ctrl+Enter', () => !S.songs.some(s => s.pending));
    await t.click('[data-nav="create"]');
    await t.ok('Create comes back', () => !document.querySelector('#app').hidden && document.querySelector('#other').hidden);
  } },
  { id: 'focus-tooltips-keys', run: async t => {
    await t.do('studio', () => { S.tier = 'studio'; S.mods = ['sampler', 'seed']; render(); });
    await t.focus('#prompt'); await t.press('Tab');
    await t.ok('Tab turns on the keyboard focus ring', () => document.documentElement.classList.contains('kbnav'));
    await t.focus('#draftsBtn'); await t.press('Tab');
    await t.do('wait', () => {}, null, 600);
    await t.ok('a control reached by keyboard shows its title as a tip', () => { const tip = document.querySelector('#tip'); return !!tip && !tip.hidden && tip.textContent.length > 0; });
    await t.press('Escape');
    await t.ok('Escape hides the tip', () => document.querySelector('#tip').hidden);
    await t.click('.top .brand');
    await t.ok('a pointer press turns the ring off', () => !document.documentElement.classList.contains('kbnav'));
    await t.click('.tile[data-m="sampler"] .knob[data-knob="samp.cfg"]');
    await t.ok('a pointer-focused knob does not show the ring', () => document.activeElement.matches('.knob') && !document.activeElement.matches(':focus-visible'));
    await t.press('ArrowUp');
    await t.ok('the first arrow key brings it up', () => document.activeElement.matches('.knob:focus-visible'));
    await t.focus('#presetList .pre.wfrow[data-preset="Seed hunter"]');
    await t.do('held Enter', () => document.activeElement.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', repeat: true, bubbles: true, cancelable: true })));
    await t.ok('a held Enter on a row-button does not repeat into a button press', () => true);
    await t.press('Control+Enter');
    await t.ok('Ctrl+Enter on a workflow row loads it rather than generating', () => S.workflow === 'Seed hunter' && !S.songs.some(s => s.pending));
  } },
  { id: 'touch-tips', w: 1440, touch: true, run: async t => {
    await t.do('studio', () => { S.tier = 'studio'; S.mods = ['seed']; render(); });
    await t.do('long press', () => { const b = document.querySelector('[data-seedrand]'), r = b.getBoundingClientRect(); b.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerType: 'touch', clientX: r.left + 4, clientY: r.top + 4 })); }, null, 700);
    await t.ok('pressing and holding shows the tip', () => !document.querySelector('#tip')?.hidden && /fresh seed/.test(document.querySelector('#tip').textContent));
    await t.do('later', () => {}, null, 3000);
    await t.ok('and it goes by itself', () => document.querySelector('#tip').hidden);
  } },
];
