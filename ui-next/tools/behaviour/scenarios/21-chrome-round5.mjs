// Round 5, the chrome around the desk and the flows through it (review 5, F-01 to F-37, and L-07): the go-bar player,
// the picker wall, names, stored data that can't be read, Latest, the stub pages, the Lab desk on a short window.
// The pending scenarios at the end state behaviours the page is known to get wrong (see README, "Pending scenarios").
const withPrompt = t => t.do('a description', () => { document.querySelector('#prompt').value = 'warm folk song'; });
/* every live tile of the picker is wholly inside its column (nothing below the fold) */
const pickerFits = () => { const out = []; for (const w of document.querySelectorAll('#pkLeft, #pkRight')) { const r = w.getBoundingClientRect();
  for (const it of w.querySelectorAll('.fl-it')) { if (!it.offsetParent) continue; const b = it.getBoundingClientRect(); if (b.bottom > r.bottom + 1 || b.top < r.top - 1) out.push(it.dataset.pkid); } }
  return out.length ? out.join(', ') + ' not wholly in view' : true; };
const pickerAt = (w, h) => ({ id: `picker-wall-fits-${w}`, w, h, run: async t => {
  await t.press('F9');
  await t.ok(`at ${w}x${h} the whole wall is on screen, no column scrolls (F-05)`, pickerFits);
  await t.ok('and nothing says "below"', () => !document.querySelector('.fl-more') || 'shows ' + document.querySelector('.fl-more').textContent);
} });

export default [
  /* F-03: the go-bar player follows the go bar's width, so Quick (a 900px bar at every size) always names its song */
  ...[[1920, 1080], [1600, 900], [1280, 720]].map(([w, h]) => ({ id: `quick-player-names-its-song-${w}`, w, h, run: async t => {
    await t.do('quick', () => { setTier('quick'); });
    await t.click('#latest .song[data-id="4"] [data-play]');
    await t.ok(`at ${w}, Quick: the player shows the title and a seek bar of at least 160px (F-03)`, () => { const b = document.querySelector('#player .pi > b'), tr = document.querySelector('#ptrack');
      return (b.getClientRects().length && b.getBoundingClientRect().width > 60 && tr.getBoundingClientRect().width >= 160) || `title ${b.getClientRects().length ? b.getBoundingClientRect().width : 'hidden'}, track ${tr.getBoundingClientRect().width}`; });
    await withPrompt(t); await t.click('#go'); await t.click('#go');
    await t.ok('with a run queued too (Cancel all showing), the title stays and the time goes first', () => { const b = document.querySelector('#player .pi > b'), tr = document.querySelector('#ptrack');
      return (b.getClientRects().length && tr.getBoundingClientRect().width >= 160) || `title ${b.getClientRects().length}, track ${tr.getBoundingClientRect().width}`; });
  } })),
  /* F-05, F-09: the wall fits the two larger sizes; elsewhere each column scrolls by itself and counts what is below it */
  pickerAt(1920, 1080), pickerAt(1600, 900), pickerAt(1536, 864), pickerAt(1440, 900),
  { id: 'picker-wall-scrolls-by-column-1280', w: 1280, h: 720, run: async t => {
    await t.press('F9');
    await t.ok('at 1280x720 every module is wholly in view (F-05)', () => { const w = document.querySelector('#pkLeft'), r = w.getBoundingClientRect(), out = [...w.querySelectorAll('.fl-it')].filter(it => it.offsetParent && it.getBoundingClientRect().bottom > r.bottom + 1).map(it => it.dataset.pkid); return !out.length || out.join(', '); });
    await t.ok('the Pedalboard column scrolls by itself, and its "+N below" stands at its own foot', () => { const r = document.querySelector('#pkRight'), b = document.querySelector('#pkMoreR'); if (!b) return 'no count';
      const rr = r.getBoundingClientRect(), br = b.getBoundingClientRect(); return (r.scrollHeight > r.clientHeight && br.left >= rr.left && br.right <= rr.right && Math.abs(br.bottom - rr.bottom) < 40 && /^\+\d+ below$/.test(b.textContent)) || JSON.stringify([rr.left, rr.right, rr.bottom, br.left, br.right, br.bottom, b.textContent]); });
    await t.click('#pkMoreR');
    await t.ok('the count scrolls that column only', () => document.querySelector('#pkRight').scrollTop > 0 && document.querySelector('#pkLeft').scrollTop === 0 && document.querySelector('#pkCols').scrollTop === 0);
  } },
  { id: 'picker-category-bar', w: 1920, h: 1080, run: async t => {
    await t.press('F9');
    /* the counts are the catalogue's (what the wall draws, PEDALS, EXTRAS), so adding or removing a module or a pedal
       changes what is expected, not the scenario (RA-03) */
    await t.ok('the count says what is shown (F-09)', () => document.querySelector('#pkCount').textContent === `${document.querySelectorAll('#pickerWrap .fl-it[data-pkid]').length} in all` || document.querySelector('#pkCount').textContent);
    await t.click('#pkBar [data-pkcat="Pedals"]');
    await t.ok('"N pedals" for the Pedals filter', () => document.querySelector('#pkCount').textContent === `${PEDALS.length} pedals` || document.querySelector('#pkCount').textContent);
    await t.ok('the pressed category is filled, All is the size of its neighbours', () => { const p = document.querySelector('#pkBar [data-pkcat="Pedals"]'), a = document.querySelector('#pkBar [data-pkcat="All"]'), c = document.querySelector('#pkBar [data-pkcat="Compose"]');
      const bg = getComputedStyle(p).backgroundColor, fa = getComputedStyle(a).fontSize, fc = getComputedStyle(c).fontSize; return (bg !== 'rgba(0, 0, 0, 0)' && getComputedStyle(a).backgroundColor === 'rgba(0, 0, 0, 0)' && fa === fc) || [bg, fa, fc].join(' / '); });
    await t.click('#pkBar [data-pkcat="Side units"]');
    await t.ok('"N side units"', () => document.querySelector('#pkCount').textContent === `${EXTRAS.length} side units` || document.querySelector('#pkCount').textContent);
  } },
  /* F-10: two names that look the same are the same name */
  { id: 'names-with-inner-spaces-clash', run: async t => {
    await t.do('a revision named "My polish"', () => { S.workflows = [{ name: 'Mine', tier: 'studio', mods: ['tempo'], pedals: [] }]; S.revs = Object.create(null); S.revs.Mine = [{ name: 'My polish', backend: 'ace', vals: {}, slots: [], user: true }]; S.workflow = 'Mine'; saveUser(); renderPresets(); });
    await t.click('#saveBtn'); await t.click('[data-saveas="rev"]'); await t.type('My  polish'); await t.click('#saveOk');
    await t.ok('"My  polish" (two spaces) offers Replace, as the same name (F-10)', () => (document.querySelector('#saveOk').textContent === 'Replace' && S.revs.Mine.length === 1) || document.querySelector('#saveOk').textContent + ' / ' + S.revs.Mine.map(r => r.name).join());
    await t.press('Escape');
    await t.ok('names compare normalised', () => sameName('My  polish ', 'my polish') && !sameName('My polish', 'My polis'));
  } },
  /* F-16: whatever stored can't be used is copied aside before anything writes over it, and said in one set of words */
  { id: 'unreadable-desk-kept-aside', storage: { 'hs3-desk': '{"v":1,"desk":null}' }, run: async t => {
    await t.eq('a desk that can\'t be used is copied aside (F-16)', () => localStorage.getItem('hs3-desk-unreadable'), '{"v":1,"desk":null}');
    await t.ok('and the page says so, with a download', () => { const m = document.querySelector('#toast .toast-msg')?.textContent || ''; return (/couldn't be read.*A copy is kept in this browser; nothing was deleted/.test(m) && document.querySelector('#toast .toast-act')?.textContent === 'Download the copy') || m; });
    await t.click('#toast .toast-act');
    await t.ok('which downloads the copies', async () => { const d = window.__bh.downloads.pop(); return (d && /unread/.test(d.name) && JSON.parse(await d.blob.text())['hs3-desk-unreadable'] === '{"v":1,"desk":null}') || JSON.stringify(d?.name); });
    await t.do('a change', () => { S.tier = 'studio'; render(); autosave(); }, null, 2000);
    await t.eq('the copy outlives the next autosave', () => localStorage.getItem('hs3-desk-unreadable'), '{"v":1,"desk":null}');
  } },
  { id: 'skipped-records-kept-aside', storage: { 'hs3-wf': JSON.stringify({ workflows: [{ name: 'Good', tier: 'studio', mods: ['tempo'], pedals: [] }, { name: 5 }], revs: {} }) }, run: async t => {
    await t.ok('a record that can\'t be read is skipped and the whole stored value is copied aside first (F-16)', () => /"name":5/.test(localStorage.getItem('hs3-wf-unreadable') || '') && S.workflows.length === 1 || localStorage.getItem('hs3-wf-unreadable'));
    await t.do('save the list', () => saveUser());
    await t.ok('after the next save the skipped record is still in the copy', () => /"name":5/.test(localStorage.getItem('hs3-wf-unreadable')) && !/"name":5/.test(localStorage.getItem('hs3-wf')));
  } },
  /* F-11, F-23, F-31: the Workflows list */
  { id: 'workflow-list-names', w: 1920, h: 1080, run: async t => {
    await t.do('a long name and a revision of a factory workflow', () => { S.workflows = [{ name: 'A very long workflow name th', tier: 'studio', mods: ['tempo'], pedals: [] }]; S.revs = Object.create(null); S.revs['Deep sampler'] = [{ name: 'Polishing mine', backend: 'ace', vals: {}, slots: [], user: true }]; S.wfOpen = new Set(['Deep sampler']); saveUser(); renderPresets(); });
    await t.ok('a 28-character name is shown whole (F-31)', () => { const b = document.querySelector('.pre.wfrow[data-preset="A very long workflow name th"] b'); return b.scrollHeight <= b.clientHeight + 1 || `${b.scrollHeight} > ${b.clientHeight}`; });
    await t.ok('a revision you saved under a factory workflow is marked yours at rest (F-23)', () => { const r = document.querySelector('.pre.rev[data-rn="Polishing mine"]'); return (!!r?.querySelector('.pmine') && /yours/.test(r.getAttribute('aria-label'))) || r?.outerHTML.slice(0, 200); });
    await t.do('no workflows of ours', () => { S.workflows = []; renderPresets(); });
    await t.ok('and the empty Yours group says where they are', () => /marked “yours”/.test(document.querySelector('#presetList .empty-note').textContent) || document.querySelector('#presetList .empty-note').textContent);
  } },
  /* F-12: a model change is named by its role */
  { id: 'model-steps-named-by-role', run: async t => {
    await t.click('#engBtn');
    await t.do('pick another sound model', () => { const b = document.querySelector('#engPanel [data-pick="model.dit"], #engPanel [data-dd="model.dit"]'); setVal('model.dit', 'Turbo · Q8_0'); });
    await t.ok('the step reads "Sound model …" (F-12)', () => /^Sound model XL Turbo · Q8_0 → Turbo · Q8_0$/.test(UNDO.at(-1)?.label || '') || UNDO.at(-1)?.label);
  } },
  /* F-14, F-27: Latest */
  { id: 'latest-cards-read-whole', w: 1920, h: 1080, run: async t => {
    await t.ok('every sample card shows its whole description, on two lines at most (F-14)', () => { const bad = [...document.querySelectorAll('#latest .song .meta p')].filter(p => p.offsetParent && p.scrollHeight > p.clientHeight + 1).map(p => p.textContent); return !bad.length || bad.join(' | '); });
    await t.ok('and the ⋯ button shares the title\'s line only', () => { const c = document.querySelector('#latest .song .meta'), p = c.querySelector('p'), m = c.querySelector('.more'); return p.getBoundingClientRect().width > c.getBoundingClientRect().width - 2 && m.getBoundingClientRect().bottom <= p.getBoundingClientRect().top + 4 || 'narrow'; });
    await t.ok('the count says what this page holds (F-27)', () => /^\d+ songs?$|^\d+ of your (newest 60|\d+) songs$/.test(document.querySelector('#songCount').textContent) || document.querySelector('#songCount').textContent);
  } },
  /* F-26, F-35: targets, and the way back from a stub page */
  { id: 'targets-and-stub-way-back', w: 1920, h: 1080, run: async t => {
    await t.ok('the workflow chevron and "Open the Library" are at least 24px (F-26)', () => { const s = [...document.querySelectorAll('.pchev:not(:disabled)')].slice(0, 1).concat(document.querySelector('.res-head .linkbtn')).map(b => b.getBoundingClientRect()); return s.every(r => r.width >= 24 && r.height >= 24) || JSON.stringify(s.map(r => [r.width, r.height])); });
    await t.click('.top [data-nav="Library"]');
    await t.click('#other [data-goto="create"]');
    await t.ok('a stub page has "Back to Create", which goes back with focus on Create (F-35)', () => !document.querySelector('#app').hidden && document.activeElement?.dataset.nav === 'create');
  } },
  /* L-07: on a short window an empty prompt card is one line tall until focused, and an empty Lab sheet four lines */
  { id: 'short-window-lab-shows-hardware', w: 1280, h: 720, run: async t => {
    await t.do('Radio master in Lab', () => { const p = findPreset('Radio master'); S.mods = p.mods.filter(m => MODS[m]); S.pedals = [...p.pedals]; S.workflow = p.name; S.tier = 'lab'; render(); });
    await t.ok('the empty prompt card stands one line tall (L-07)', () => document.querySelector('.prompt-card').getBoundingClientRect().height <= 64 || document.querySelector('.prompt-card').getBoundingClientRect().height);
    await t.ok('and the Pedalboard starts in the first screen, above the go bar', () => { const b = document.querySelector('#bento .tile[data-m="post"]').getBoundingClientRect().top, g = document.querySelector('.gobar').getBoundingClientRect().top; return b < g - 60 || `${b} vs ${g}`; });
    await t.click('#prompt');
    await t.ok('focused, the card opens to its full height', () => document.querySelector('.prompt-card').getBoundingClientRect().height > 100);
  } },

  /* ---- pending: behaviours the page gets wrong today; each fails now, for its finding's reason ---- */
  { id: 'ctrl-enter-generates-from-rows-and-cards', pending: { finding: 'F-07', right: 'Ctrl+Enter is Generate from every focus position on Create with no layer open, a workflow row, a revision row or a Modules card included; only a plain Enter opens, loads or adds.' }, run: async t => {
    await withPrompt(t);
    await t.focus('.pre.wfrow[data-preset="Songwriter"]'); await t.press('Control+Enter');
    await t.ok('on a workflow row it generates and opens nothing', () => (S.songs.some(s => s.pending) && S.workflow !== 'Songwriter') || `pending ${S.songs.filter(s => s.pending).length}, workflow ${S.workflow}`);
    await t.focus('.bcard[data-additem="seed"]'); await t.press('Control+Enter');
    await t.ok('on a Modules card it generates and adds nothing', () => (S.songs.filter(s => s.pending).length >= 2 && !S.mods.includes('seed')) || `pending ${S.songs.filter(s => s.pending).length}, seed on desk ${S.mods.includes('seed')}`);
  } },
  { id: 'tier-keys-wait-under-a-modal', pending: { finding: 'F-08', right: 'No page shortcut acts under a modal layer (the Concept notes, the score modal, a drawer): Alt+1/2/3 there leave the tier as it is.' }, run: async t => {
    await t.click('#notesBtn'); await t.press('Alt+3');
    await t.eq('Alt+3 under the Concept notes leaves the tier', () => S.tier, 'quick');
  } },
  { id: 'focus-after-cancel-retry-dismiss', pending: { finding: 'F-13', right: 'After Cancel on a queued song, Retry or Dismiss on a "Didn\'t run" notice, focus goes to Generate (Dismiss: to the next notice\'s first button if there is one), never to the page or to the next song\'s Cancel.' }, run: async t => {
    await withPrompt(t); await t.click('#go'); await t.click('#go'); await t.click('#go');
    const id = await t.js(() => S.songs.filter(s => s.pending)[1].id);
    await t.click(`[data-cancelsong="${id}"]`);
    await t.ok('Cancel on one queued song: focus on Generate', () => document.activeElement?.id === 'go' || document.activeElement?.outerHTML.slice(0, 80));
    await t.do('MiniMax-Music3, too big', () => { cancelSongs(new Set(S.songs.filter(s => s.pending).map(s => s.id))); S.backend = 'mm3'; S.tier = 'studio'; S.mods = ['takes']; S.vals['takes.n'] = 4; S.vals['model.lm'] = '8B planner · F16'; render(); }, null, 300);
    await t.click('#go', 300);
    await t.click('#latest .song.errored [data-retry]', 300);
    await t.ok('Retry: focus on Generate', () => document.activeElement?.id === 'go' || document.activeElement?.tagName);
    await t.do('finish, and refuse again', () => { S.vals['takes.n'] = 4; render(); generate(); }, null, 4200);
    await t.click('#latest .song.errored [data-dismiss]');
    await t.ok('Dismiss: focus on Generate', () => document.activeElement?.id === 'go' || document.activeElement?.tagName);
  } },
  { id: 'focus-after-moving-lyrics', pending: { finding: 'F-13', right: '"Move them to the lyrics sheet" puts focus in the lyrics field it filled.' }, run: async t => {
    await t.do('studio with lyrics', () => { S.tier = 'studio'; render(); });
    await t.click('#prompt'); await t.type('slow jam\n[Verse]\nla la');
    await t.click('#suggest [data-sg="lyrics"]');
    await t.ok('focus is in the lyrics', () => !!document.activeElement?.closest('.paper') || document.activeElement?.tagName);
  } },
  { id: 'rename-in-another-tab-is-followed', pending: { finding: 'F-17', right: 'A workflow renamed in another tab stays open here under its new name, and is never reported as deleted.' }, run: async t => {
    await t.do('an open workflow of ours', () => { S.workflows = [{ name: 'From A', tier: 'quick', mods: ['tempo'], pedals: [] }]; S.revs = Object.create(null); saveUser(); loadPreset(findPreset('From A'), false); });
    const p2 = await t.page.context().newPage(); await p2.goto(t.page.url()); await t.settle(1500);
    await p2.evaluate(() => { S.workflows[0].name = 'Renamed in A'; saveUser(); });
    await t.settle(300);
    await t.ok('this tab follows the rename', () => (S.workflow === 'Renamed in A' && !/deleted in another tab/.test((document.querySelector('#toast .toast-msg')?.textContent || ''))) || `workflow ${S.workflow}, toast "${(document.querySelector('#toast .toast-msg')?.textContent || '')}"`);
    await p2.close();
  } },
  { id: 'toasts-keep-their-undo', pending: { finding: 'F-18', right: 'A plain message never pushes out one that carries an action (Undo); and Escape on a toast\'s button dismisses that toast and puts focus back where it was.' }, run: async t => {
    await t.do('desk', () => { S.tier = 'studio'; S.mods = ['tempo', 'seed']; render(); });
    await t.click('.tile[data-m="seed"] [data-remove="seed"]');
    await t.do('three plain messages in one beat', () => { toast('One'); toast('Two'); toast('Three'); }, null, 100);
    await t.ok('the Undo toast is still there', () => [...document.querySelectorAll('#toast, .toast-old')].some(el => /Undo/.test(el.querySelector('.toast-act')?.textContent || '')) || [...document.querySelectorAll('#toast, .toast-old')].map(el => el.textContent).join(' | '));
    await t.do('one with an action', () => toast('Removed something', { label: 'Undo', fn: () => {} }));
    await t.focus('#toast .toast-act'); await t.press('Escape');
    await t.ok('Escape on its button dismisses the toast', () => document.querySelector('#toast').hidden || 'still showing');
  } },
  { id: 'toast-reachable-under-the-picker', pending: { finding: 'F-18', right: 'A toast raised while the picker or the Concept notes are open joins their Tab cycle while it shows, so its buttons can be reached by the keyboard.' }, run: async t => {
    await t.press('F9');
    await t.do('a toast with an action', () => toast('Something happened', { label: 'Undo', fn: () => {} }));
    let reached = false; for (let i = 0; i < 50 && !reached; i++) { await t.page.keyboard.press('Tab'); reached = await t.js(() => !!document.activeElement?.closest('#toast')); }
    await t.ok('Tab reaches the toast from inside the picker', r => r || 'never reached', reached);
  } },
  { id: 'keyboard-tip-gives-way-to-the-pointer', pending: { finding: 'F-19', right: 'A tooltip opened by keyboard focus closes on the next pointer move of more than 8px, on any key but a modifier, and after 6 seconds.' }, run: async t => {
    await t.focus('#engBtn'); await t.press('Tab', 1500);
    await t.ok('the keyboard opened the VRAM meter tip', () => (document.activeElement?.id === 'engMeter' && !!document.querySelector('#tip') && !document.querySelector('#tip').hidden) || `${document.activeElement?.id} ${document.querySelector('#tip')?.hidden}`);
    await t.hover('#ask', 300);
    await t.ok('a pointer move elsewhere closes it', () => !document.querySelector('#tip') || document.querySelector('#tip').hidden || document.querySelector('#tip').textContent);
  } },
  { id: 'rapid-tier-and-theme-switching-is-quiet', pending: { finding: 'F-20', right: 'Switching tier or theme several times in quick succession leaves no error in the console (a view transition that is skipped is expected, not thrown).' }, run: async t => {
    for (const k of ['Alt+2', 'Alt+3', 'Alt+1', 'Alt+3']) await t.page.keyboard.press(k);
    for (const k of ['light', 'dark', 'system', 'dark']) await t.page.evaluate(k => document.querySelector(`#themeSeg [data-themeset="${k}"]`)?.click(), k);
    await t.settle(1000);
    await t.eq('the tier ends where the last key put it', () => S.tier, 'lab');
  } },
  { id: 'generate-does-not-queue-a-certain-refusal', pending: { finding: 'F-21', right: 'When the next run is certain not to fit, Generate does not start it: the "Use N takes" link pulses and the line is announced; a refusal that does happen never stacks a second identical notice.' }, run: async t => {
    await t.do('MiniMax-Music3, too big', () => { S.backend = 'mm3'; S.tier = 'studio'; S.mods = ['takes']; S.vals['takes.n'] = 4; S.vals['model.lm'] = '8B planner · F16'; document.querySelector('#prompt').value = 'big song'; render(); });
    await t.click('#go', 300); await t.click('#go', 300);
    await t.ok('no run was started and no notice stacked', () => document.querySelectorAll('#latest .song.errored').length === 0 || `${document.querySelectorAll('#latest .song.errored').length} notices`);
  } },
  { id: 'separate-presses-each-queue', pending: { finding: 'F-22', right: 'Every separate press of Generate queues a run; only the second click of one double-click is ignored.' }, run: async t => {
    await withPrompt(t);
    const r = await t.js(() => { const b = document.querySelector('#go').getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2]; });
    await t.page.mouse.click(r[0], r[1]); await t.page.clock.runFor(310); await t.page.mouse.click(r[0], r[1]); await t.settle(300);
    await t.eq('two presses 310ms apart queue two runs', () => S.songs.filter(s => s.pending).length, 2);
  } },
  { id: 'adding-from-the-panel-answers-one-way', pending: { finding: 'F-25', right: 'Adding from the Modules panel, module or pedal, keeps focus on the card and says "Added <name>"; a module the engine does not use is parked with a toast that carries Undo.' }, run: async t => {
    await t.do('lab', () => { S.tier = 'lab'; S.mods = []; S.pedals = []; render(); });
    await t.focus('.bcard[data-additem="pd:ss"]'); await t.press('Enter');
    await t.ok('a pedal card keeps focus, as a module card does', () => document.activeElement?.matches('.bcard[data-additem="pd:ss"]') || document.activeElement?.outerHTML.slice(0, 80));
    await t.focus('.bcard[data-additem="score"]'); await t.press('Enter');
    await t.ok('the parked module\'s toast carries Undo', () => !!document.querySelector('#toast .toast-act') || (document.querySelector('#toast .toast-msg')?.textContent || ''));
  } },
  { id: 'song-menu-keeps-off-the-go-bar', w: 1280, h: 720, pending: { finding: 'F-28', right: 'A song card\'s menu opens above its ⋯ button, right edges together, inside the centre column, and never over the go bar; it opens below only when above lacks room.' }, run: async t => {
    await t.do('studio', () => { S.tier = 'studio'; render(); });
    const id = await t.js(() => [...document.querySelectorAll('#latest .song[data-id]')].filter(s => s.offsetParent)[3].dataset.id);
    await t.click(`#latest .song[data-id="${id}"] [data-songmenu]`);
    await t.ok('the menu stays clear of the go bar and inside the centre column', () => { const m = document.querySelector('#menu').getBoundingClientRect(), g = document.querySelector('.gobar').getBoundingClientRect(), c = document.querySelector('#centre').getBoundingClientRect();
      const over = m.left < g.right && m.right > g.left && m.top < g.bottom && m.bottom > g.top;
      return (!over && m.left >= c.left && m.right <= c.right) || JSON.stringify({ menu: [m.left, m.top, m.right, m.bottom], gobarTop: g.top, centre: [c.left, c.right] }); });
  } },
  { id: 'picker-f9-during-close-reopens', pending: { finding: 'F-29', right: 'F9 pressed while the picker is closing opens it again.' }, run: async t => {
    await t.press('F9');
    await t.page.keyboard.press('F9'); await t.page.clock.runFor(60); await t.page.keyboard.press('F9'); await t.settle(400);
    await t.ok('F9 during the close reopens the picker', () => !document.querySelector('#pickerWrap').hidden || 'closed');
  } },
  { id: 'picker-down-stays-in-its-column', w: 1920, h: 1080, pending: { finding: 'F-29', right: 'In the picker ArrowDown and ArrowUp stay in their own column (at its foot they stop); only ArrowLeft and ArrowRight cross to the other column.' }, run: async t => {
    await t.press('F9'); await t.focus('.fl-it[data-pkid="plans"]'); await t.press('ArrowDown');
    await t.ok('ArrowDown from the foot of Modules stays in Modules', () => !!document.activeElement?.closest('#pkLeft') || document.activeElement?.dataset.pkid);
  } },
  { id: 'picker-rings-the-enter-target', pending: { finding: 'F-29', right: 'With a filter typed in the picker, the tile Enter will add is ringed while focus stays in the filter.' }, run: async t => {
    await t.press('F9'); await t.type('rep');
    await t.ok('the tile Enter will add is ringed', () => { const el = document.querySelector('.fl-it[data-pkid="repetition"] .box'); const s = getComputedStyle(el); return s.outlineStyle !== 'none' || s.outlineStyle; });
  } },
  { id: 'list-keys', pending: { finding: 'F-36', right: 'In the Workflows list F2 renames the focused row of yours, Delete arms its Delete, and typing a letter moves to the next row starting with it.' }, run: async t => {
    await t.do('two of ours', () => { S.workflows = [{ name: 'Alpha', tier: 'quick', mods: ['tempo'], pedals: [] }, { name: 'Beta', tier: 'quick', mods: ['seed'], pedals: [] }]; S.revs = Object.create(null); saveUser(); renderPresets(); });
    await t.focus('.pre.wfrow[data-preset="Alpha"]'); await t.press('F2');
    await t.ok('F2 renames', () => !!document.querySelector('.pre.wfrow input.pren') || 'no field');
    await t.press('Escape');
    await t.focus('.pre.wfrow[data-preset="Alpha"]'); await t.press('b');
    await t.ok('typing "b" moves to Beta', () => document.activeElement?.dataset.preset === 'Beta' || document.activeElement?.dataset.preset);
    await t.focus('.pre.wfrow[data-preset="Alpha"]'); await t.press('Delete');
    await t.ok('Delete arms the row Delete and deletes nothing yet', () => (S.workflows.some(w => w.name === 'Alpha') && [...document.querySelectorAll('#pop button, .pre.wfrow[data-preset="Alpha"] button')].some(b => /Delete\?/.test(b.textContent))) || 'not armed');
  } },
  { id: 'leaving-while-rendering-warns', pending: { finding: 'F-37', right: 'While a song is rendering or queued, leaving the page asks first (beforeunload), since those songs are lost.' }, run: async t => {
    await withPrompt(t); await t.click('#go');
    await t.ok('beforeunload is refused while a run is live', () => { const e = new Event('beforeunload', { cancelable: true }); dispatchEvent(e); return e.defaultPrevented || 'not prevented'; });
  } },
  { id: 'reload-mid-run-says-what-was-lost', pending: { finding: 'F-27', right: 'After a reload that dropped a song still rendering, one toast says so: "1 song was still rendering when the page closed and wasn\'t kept."' }, run: async t => {
    await withPrompt(t); await t.click('#go'); await t.reload();
    await t.ok('the page says a song was lost', () => /still rendering when the page closed/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) || (document.querySelector('#toast .toast-msg')?.textContent || ''));
  } },
];
