// Undo that reverses only its own action, Redo, overwrites behind Undo, and stored saves that survive bad records and
// a second tab (round 3, review C: C-02, C-03, C-07, C-08, C-09, C-14).
const msg = () => document.querySelector('#toast .toast-msg')?.textContent || '';
const act = () => document.querySelector('#toast .toast-act')?.textContent || '';
export default [
  { id: 'undo-leaves-later-work', run: async t => {
    await t.click('#presetList .pre.wfrow[data-preset="Songwriter"]');
    await t.ok('opening a workflow is one Undo entry', () => /Open "Songwriter"/.test(document.querySelector('#undoBtn span').textContent));
    await t.click('#prompt'); await t.type('slow folk ballad');
    await t.do('lyrics and a tempo, typed after opening it', () => { setVal('lyrics.text', '[Verse]\nhello'); S.vals['tempo.bpm.auto'] = false; setVal('tempo.bpm', 97); });
    /* changed on purpose (S-03): what was typed since is a step of its own, newer than the opening, so the Undo button
       names it; the opening is still undone on its own from its toast, and leaves the later work */
    await t.ok('the Undo button names the newest step, the settings just changed', () => { const b = document.querySelector('#undoBtn'); b.dispatchEvent(new PointerEvent('pointerenter')); return /Change the lyrics and BPM/.test(b.querySelector('span').textContent) && /Puts back the lyrics and BPM/.test(b.title) ? true : b.title; });
    await t.ok('its toast still offers its Undo', () => /Opened Songwriter/.test(document.querySelector('#toast .toast-msg')?.textContent || '') && document.querySelector('#toast .toast-act')?.textContent === 'Undo');
    await t.click('#toast .toast-act'); await t.click('.top .brand');
    await t.eq('the opening, undone from its toast, reverses only itself: the desk goes back, what was typed since stays', () => [S.workflow, S.mods.join(), S.pedals.join(), document.querySelector('#prompt').value, get('lyrics.text'), S.vals['tempo.bpm']], ['Start fresh', '', '', 'slow folk ballad', '[Verse]\nhello', 97]);
    await t.ok('the toast offers Redo', () => /Undone: Open "Songwriter"/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) && (document.querySelector('#toast .toast-act')?.textContent || '') === 'Redo');
    await t.ok('a Redo button stands beside Undo', () => !document.querySelector('#redoBtn').hidden && /Redo Open "Songwriter"/.test(document.querySelector('#redoBtn').getAttribute('aria-label')));
    await t.press('Control+Shift+z');
    await t.eq('Ctrl+Shift+Z redoes it, still keeping what was typed', () => [S.workflow, S.mods.join(), document.querySelector('#prompt').value, S.vals['tempo.bpm']], ['Songwriter', 'tempo,avoid,takes,post', 'slow folk ballad', 97]);
    await t.ok('and offers Undo again', () => /Redone: Open "Songwriter"/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) && (document.querySelector('#toast .toast-act')?.textContent || '') === 'Undo' && document.querySelector('#redoBtn').hidden);
    await t.press('Control+z'); await t.press('Control+y');
    await t.eq('Ctrl+Y redoes too', () => S.workflow, 'Songwriter');
    await t.click('#undoBtn');
    await t.click('#redoBtn');
    await t.eq('and so does the Redo button', () => [S.workflow, S.mods.join()], ['Songwriter', 'tempo,avoid,takes,post']);
  } },
  { id: 'undo-keeps-a-later-edit-of-the-same-thing', run: async t => {
    await t.do('an empty form', () => { document.querySelector('#prompt').value = ''; setVal('lyrics.text', ''); });
    await t.click('#aiBtn');
    await t.ok('Write it for me fills the description and lyrics, with Undo', () => document.querySelector('#prompt').value.length > 10 && get('lyrics.text').length > 10 && (document.querySelector('#toast .toast-act')?.textContent || '') === 'Undo');
    await t.click('#prompt'); await t.press('End'); await t.type(', with cello');
    /* changed on purpose (S-03): the typing is the newest step, so Ctrl+Z takes it back first; Write it for me's own Undo,
       from its toast, then shows the rule this checks */
    await t.click('.top .brand');
    await t.ok('the Undo button names the typing', () => /Edit the description/.test(document.querySelector('#undoBtn span').textContent) ? true : document.querySelector('#undoBtn span').textContent);
    await t.click('#toast .toast-act');
    await t.ok('Undo puts back the lyrics, but keeps the description edited since, and says so', () => get('lyrics.text') === '' && /with cello$/.test(document.querySelector('#prompt').value) && /The description changed since, so it was left as it is/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) ? true : [get('lyrics.text'), document.querySelector('#prompt').value, (document.querySelector('#toast .toast-msg')?.textContent || '')]);
  } },
  { id: 'undo-not-under-overlays', run: async t => {
    await t.click('#presetList .pre.wfrow[data-preset="Songwriter"]');
    await t.click('#ideasBtn');
    await t.press('Control+z');
    await t.ok('Ctrl+Z does nothing while a popover is open', () => S.workflow === 'Songwriter' && !document.querySelector('#pop').hidden);
    await t.press('Escape');
    await t.click('#notesBtn');
    await t.press('Control+z');
    await t.ok('nor under the Concept notes', () => S.workflow === 'Songwriter');
  } },
  { id: 'revision-revert-keeps-the-desk', run: async t => {
    await t.click('#presetList [data-presettier="Lo-fi tape"]');
    await t.click('.pre.rev[data-rev="Lo-fi tape"][data-rn="Tape at 82 BPM"] b');
    await t.do('change a setting and remove a module', () => setVal('samp.steps', 31));
    await t.click('.tile[data-m="tempo"] [data-remove="tempo"]');
    await t.ok('both rows are modified', () => S.wfDirty && S.revDirty);
    await t.click('.pre.rev.mod [data-rrevert]');
    await t.eq('the revision\'s Revert puts back its settings and leaves the desk as arranged (C-03)', () => [S.vals['samp.steps'], S.mods.includes('tempo'), S.revDirty, S.wfDirty], [30, false, false, true]);
    await t.ok('and says so, with Undo', () => /Put back the settings of Tape at 82 BPM/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) && (document.querySelector('#toast .toast-act')?.textContent || '') === 'Undo');
  } },
  { id: 'replace-is-undoable', run: async t => {
    await t.click('#presetList .pre.wfrow[data-preset="Seed hunter"]');
    await t.do('a saved revision', () => { setVal('samp.steps', 12); });
    await t.click('#saveBtn'); await t.type('One'); await t.press('Enter');
    await t.do('change it', () => setVal('samp.steps', 44));
    await t.click('#saveBtn'); await t.type('one'); await t.press('Enter'); await t.press('Enter');
    await t.eq('Replace keeps the casing and stores the new settings', () => [S.revs['Seed hunter'].length, S.revs['Seed hunter'][0].name, S.revs['Seed hunter'][0].vals['samp.steps']], [1, 'One', 44]);
    await t.click('#toast .toast-act');
    await t.eq('Undo brings the old revision back (C-02)', () => S.revs['Seed hunter'][0].vals['samp.steps'], 12);
    await t.click('#toast .toast-act');
    await t.eq('and Redo replaces it again', () => S.revs['Seed hunter'][0].vals['samp.steps'], 44);
    await t.click('#saveBtn'); await t.type('Draft'); await t.press('Escape');
    await t.click('#saveBtn');
    await t.eq('a name typed and abandoned does not come back (C-22)', () => document.querySelector('#saveName').value, '');
  } },
  { id: 'corrupt-records', storage: {
      'hs3-wf': JSON.stringify({ workflows: [{ name: 'nopedals', mods: ['tempo'], tier: 'zzz' }, { name: 'Good', tier: 'lab', mods: ['seed', 'nope'], pedals: ['norm'] }, { mods: [] }, 7, null], revs: { Good: [{ name: 'r', backend: 'ace', vals: { 'samp.steps': 9 } }, { name: 'bad', backend: 'zzz', vals: {} }], Other: 'notalist' } }),
      'hs3-drafts': JSON.stringify([{ backend: 'zzz' }, { name: 'ok', backend: 'ace', at: 1, prompt: 'p', vals: {} }, 'x']) }, run: async t => {
    await t.ok('the page boots past bad records (C-08)', () => !!document.querySelector('#bento') && booted && document.querySelectorAll('#presetList .pre.wfrow').length > 7);
    await t.eq('what can be repaired is kept, repaired', () => S.workflows.map(w => [w.name, w.tier, w.mods.join(), w.pedals.join()]), [['nopedals', 'studio', 'tempo', ''], ['Good', 'lab', 'seed', 'norm']]);
    await t.eq('revisions are checked one by one', () => [Object.keys(S.revs), S.revs.Good.map(r => r.name)], [['Good'], ['r']]);
    await t.ok('and the page says, once, what it skipped', () => /5 saved workflows or revisions couldn't be read and were skipped/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) && /2 drafts couldn't be read/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) ? true : (document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.click('#draftsBtn');
    await t.ok('Drafts opens with the readable draft', () => !document.querySelector('#pop').hidden && document.querySelectorAll('#pop .draft').length === 1);
  } },
  { id: 'unreadable-key-kept-aside', storage: { 'hs3-wf': '{broken', 'hs3-drafts': '{"a":1}' }, run: async t => {
    await t.ok('a key that is not JSON is copied aside before anything overwrites it', () => localStorage.getItem('hs3-wf-unreadable') === '{broken' && /A copy is kept in this browser; nothing was deleted/.test((document.querySelector('#toast .toast-msg')?.textContent || '')));
    await t.click('#saveBtn'); await t.click('[data-saveas="wf"]'); await t.type('Fresh'); await t.press('Enter');
    await t.ok('saving still works, and the copy stays', () => S.workflows[0]?.name === 'Fresh' && localStorage.getItem('hs3-wf-unreadable') === '{broken');
  } },
  ...[[1440, 900, 'studio'], [1280, 720, 'studio'], [1280, 720, 'quick']].map(([w, h, tier]) => ({ id: `gobar-holds-${w}-${tier}`, w, h, run: async t => {
    await t.do('a desk, a song playing with a long title, and a run', tier => { S.tier = tier; if (tier !== 'quick') S.mods = ['tempo', 'sampler']; S.songs[0].title = 'An extraordinarily long song title that goes on and on and on'; render(); renderSongs(); togglePlay(S.songs[0].id); document.querySelector('#prompt').value = 'dreamy pop'; generate(); }, tier, 300);
    await t.ok('the page does not scroll sideways (C-05)', () => document.documentElement.scrollWidth === document.documentElement.clientWidth || document.documentElement.scrollWidth - document.documentElement.clientWidth);
    await t.ok('the go bar stays in its column', () => { const g = document.querySelector('.gobar').getBoundingClientRect(), c = document.querySelector('#centre').getBoundingClientRect(); return g.right <= c.right + 10.5 || [g.right, c.right]; });
    await t.ok('nothing in it overlaps, and nothing leaves the player', () => { const kids = [...document.querySelector('.gobar').children].filter(x => !x.hidden && x.getClientRects().length).map(x => x.getBoundingClientRect()); for (let i = 0; i < kids.length; i++) for (let j = i + 1; j < kids.length; j++) { const a = kids[i], b = kids[j]; if (a.left < b.right - 0.5 && b.left < a.right - 0.5) return 'overlap ' + i + '/' + j; } const p = document.querySelector('#player').getBoundingClientRect(); return [...document.querySelector('#player').children].every(x => { const q = x.getBoundingClientRect(); return !q.width || (q.left >= p.left - 0.5 && q.right <= p.right + 0.5); }); });
  } })),
  { id: 'hover-tooltip', run: async t => {
    await t.hover('#importBtn', 500);
    await t.ok('a resting pointer gets the page\'s own tip, and the browser\'s is held back (C-31)', () => { const tip = document.querySelector('#tip'), b = document.querySelector('#importBtn'); return !!tip && !tip.hidden && tip.textContent === 'Import a revision from a JSON file' && !b.hasAttribute('title') ? true : [tip?.hidden, tip?.textContent, b.getAttribute('title')]; });
    await t.hover('#ask', 50);
    await t.ok('it goes when the pointer leaves, and the title comes back', () => document.querySelector('#tip').hidden && document.querySelector('#importBtn').title === 'Import a revision from a JSON file');
  } },
  { id: 'two-tabs', run: async t => {
    const p2 = await t.page.context().newPage(); await p2.goto(t.page.url()); await t.settle(1500);
    await p2.evaluate(() => { document.querySelector('#saveBtn').click(); document.querySelector('[data-saveas="wf"]').click(); document.querySelector('#saveName').value = 'From tab 2'; document.querySelector('#saveOk').click(); });
    await t.settle(200);
    await t.ok('a workflow saved in another tab appears here (C-09)', () => S.workflows.some(w => w.name === 'From tab 2') && !!document.querySelector('.pre.wfrow[data-preset="From tab 2"]'));
    await t.click('#saveBtn'); await t.click('[data-saveas="wf"]'); await t.type('From tab 1'); await t.press('Enter');
    await t.eq('a save here keeps it', () => JSON.parse(localStorage.getItem('hs3-wf')).workflows.map(w => w.name).sort(), ['From tab 1', 'From tab 2']);
    await p2.evaluate(() => { const i = S.workflows.findIndex(w => w.name === 'From tab 2'); S.workflows.splice(i, 1); saveUser(); });
    await t.settle(200);
    await t.eq('a delete in the other tab reaches this one, and keeps ours', () => [S.workflows.map(w => w.name), JSON.parse(localStorage.getItem('hs3-wf')).workflows.map(w => w.name)], [['From tab 1'], ['From tab 1']]);
    await p2.close();
  } },
];
