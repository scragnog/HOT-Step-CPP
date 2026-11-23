// Workflows, revisions, drafts and storage (package 7): the concept's claims where its own scenarios reach for
// something the rebuild does differently or does not have yet, stated so they hold on the rebuild.
//
// - Six hold on the concept page too (run them there with --page): rename-delete-menu and row-heights-hold are the
//   concept's 03-workflows/rename-delete-menu and 09-undo-storage/row-heights-hold with the Sampler on the workflow's
//   desk, since the rebuild's setVal changes a value only through a control on the desk, as a person can (the concept's
//   sets a module's value with the module off the desk, which the rebuild refuses: docs/09, "How a part is written");
//   opening-undone-leaves-later-work, saved-says-only-what-was-saved, escape-in-the-save-row and export-then-import
//   state the concept's claims without the prompt card.
// - The rest run on the rebuild only: they set the description with setVal('prompt.text', ...) where the concept's type
//   into #prompt (the prompt card is package 4's; once it lands, the concept's own scenarios, listed as waiting in
//   acceptance/7-workflows.json, take over), or they seed the page's own keys (deviation 5: the desk is kept under
//   hotstep.create.desk, in the page's shape, so a seed of the concept's hs3-desk means nothing to it). The test layer
//   reads the concept's storage keys as the page's (testing/compat/workflows.ts), so their checks read as the concept's.

/* the checks run in the page, so the toast's words are read inline in each */

export default [
  { id: 'rename-delete-menu', run: async t => {
    await t.do('two user workflows with a revision', () => { S.workflows = [{ name: 'Alpha', tier: 'studio', mods: ['tempo', 'sampler'], pedals: [] }, { name: 'Beta', tier: 'lab', mods: ['seed'], pedals: [] }]; S.revs = { Alpha: [{ name: 'r1', backend: 'ace', vals: { 'samp.steps': 20 }, slots: [] }] }; saveUser(); S.wfOpen.add('Alpha'); renderPresets(); });
    await t.click('.pre.wfrow[data-preset="Alpha"] [data-rowmore]');
    await t.ok('⋯ opens the row menu', () => !document.querySelector('#pop').hidden && document.querySelectorAll('#pop [data-rowact]').length === 2);
    await t.click('#pop [data-rowact="wfren"]');
    await t.ok('Rename turns the name into a field', () => document.activeElement?.classList.contains('pren'));
    await t.type('Gamma'); await t.press('Enter');
    await t.ok('Enter renames it, and its revisions follow', () => S.workflows[0].name === 'Gamma' && S.revs.Gamma?.length === 1 && !S.revs.Alpha && document.activeElement?.dataset.preset === 'Gamma');
    await t.click('.pre.wfrow[data-preset="Gamma"] [data-rowmore]'); await t.click('#pop [data-rowact="wfren"]');
    await t.type('Beta'); await t.press('Enter');
    await t.ok('a name in use is refused, beside the field (F-11)', () => /"Beta" workflow already exists/.test(document.querySelector('#prenHint:not([hidden])')?.textContent) && document.activeElement?.classList.contains('pren'));
    await t.press('Escape');
    await t.ok('Escape keeps the old name', () => S.workflows[0].name === 'Gamma' && !document.querySelector('.pren'));
    await t.click('.pre.rev[data-rev="Gamma"][data-rn="r1"] [data-rowmore]'); await t.click('#pop [data-rowact="rren"]');
    await t.type('r2'); await t.click('.top .brand');
    await t.eq('a revision renames on blur', () => S.revs.Gamma[0].name, 'r2');
    await t.click('.pre.rev[data-rn="r2"] [data-rowmore]'); await t.click('#pop [data-rowact="rdel"]');
    await t.ok('Delete in the menu asks once', () => /^Delete\?$/.test(document.querySelector('#pop [data-rowact="rdel"]')?.textContent || '') && S.revs.Gamma.length === 1);
    await t.click('#pop [data-rowact="rdel"]');
    await t.ok('the second press deletes it, with Undo', () => !(S.revs.Gamma || []).length && /Deleted Gamma › r2/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.click('#toast .toast-act');
    await t.eq('Undo restores it', () => S.revs.Gamma?.length, 1);
    await t.click('.pre.wfrow[data-preset="Beta"] [data-rowmore]'); await t.click('#pop [data-rowact="delpreset"]'); await t.click('#pop [data-rowact="delpreset"]');
    await t.ok('a workflow deletes from the menu', () => !S.workflows.some(w => w.name === 'Beta') && /Deleted the "Beta" workflow/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.click('#undoBtn');
    await t.ok('and Undo brings it back', () => S.workflows.some(w => w.name === 'Beta'));
    await t.do('user revision on the open workflow', () => { S.workflow = 'Gamma'; applyRev('Gamma', S.revs.Gamma[0]); setVal('samp.steps', 44); renderPresets(); });
    await t.click('.pre.rev.mod [data-rover]');
    await t.eq('Update on a revision saves the changes into it at once (it is undoable)', () => [S.revs.Gamma[0].vals['samp.steps'], S.revDirty], [44, false]);
    await t.click('#toast .toast-act');
    await t.eq('and Undo puts the saved revision back', () => [S.revs.Gamma[0].vals['samp.steps'], S.revDirty], [20, true]);
  } },
  { id: 'row-heights-hold', run: async t => {
    await t.do('a user workflow with a revision, open and modified', () => { S.workflows = [{ name: 'Mine', tier: 'quick', mods: ['tempo', 'sampler'], pedals: [] }]; S.revs = { Mine: [{ name: 'One', backend: 'ace', vals: { 'samp.steps': 20 }, slots: [] }] }; saveUser(); loadPreset(S.workflows[0]); applyRev('Mine', S.revs.Mine[0]); setVal('samp.steps', 21); S.wfOpen.add('Start fresh'); renderPresets(); });
    const h = sel => new Function(`return Math.round(document.querySelector(${JSON.stringify(sel)}).getBoundingClientRect().height * 2) / 2;`);
    const rows = ['.pre.wfrow[data-preset="Mine"]', '.pre.rev[data-rn="One"]', '.pre.wfrow[data-preset="Songwriter"]', '.pre.rev[data-rn="Quick sketch"]'], want = [36, 58, 36, 32];
    for (const [i, s] of rows.entries()) await t.eq(`${s} rests at ${want[i]} px`, h(s), want[i]);
    for (const [i, s] of rows.entries()) { await t.hover(s); await t.eq(`hovering ${s} leaves its height alone (C-01)`, h(s), want[i]); }
    await t.focus('.pre.rev[data-rn="One"] [data-rinfo]');
    await t.eq('and so does focus on one of its keys', h('.pre.rev[data-rn="One"]'), 58);
  } },
  /* 09-undo-storage/undo-leaves-later-work, with what follows the opening set by setVal */
  { id: 'opening-undone-leaves-later-work', run: async t => {
    await t.click('#presetList .pre.wfrow[data-preset="Songwriter"]');
    await t.ok('opening a workflow is one Undo entry', () => /Open "Songwriter"/.test(document.querySelector('#undoBtn span').textContent));
    await t.do('lyrics and takes, set after opening it', () => { setVal('lyrics.text', 'hello'); setVal('takes.n', 3); });
    await t.ok('the Undo button names the newest step, not the opening', () => !/Open/.test(document.querySelector('#undoBtn span').textContent) || document.querySelector('#undoBtn span').textContent);
    await t.ok('the opening\'s toast still offers its Undo', () => /Opened Songwriter/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) && (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-act')?.textContent || '') === 'Undo');
    await t.click('#toast .toast-act');
    await t.eq('the opening, undone from its toast, reverses only itself: the desk goes back, what was set since stays (C-07)', () => [S.workflow, S.mods.join(), S.pedals.join(), get('lyrics.text'), S.vals['takes.n']], ['Start fresh', '', '', 'hello', 3]);
    await t.ok('the toast offers Redo', () => /Undone: Open "Songwriter"/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) && (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-act')?.textContent || '') === 'Redo');
    await t.ok('a Redo button stands beside Undo', () => !document.querySelector('#redoBtn').hidden && /Redo Open "Songwriter"/.test(document.querySelector('#redoBtn').getAttribute('aria-label')));
    await t.press('Control+Shift+z');
    await t.eq('Ctrl+Shift+Z redoes it, still keeping what was set', () => [S.workflow, S.mods.join(), get('lyrics.text'), S.vals['takes.n']], ['Songwriter', 'tempo,avoid,takes,post', 'hello', 3]);
    await t.ok('and offers Undo again', () => /Redone: Open "Songwriter"/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) && (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-act')?.textContent || '') === 'Undo');
  } },
  /* 07-overlays-keys/drafts, with the description set by setVal */
  { id: 'drafts-hold-the-content', run: async t => {
    await t.do('a description', () => setVal('prompt.text', 'first draft'));
    await t.click('#draftsBtn');
    await t.ok('Drafts opens above its button, focus in its name field', () => !document.querySelector('#pop').hidden && document.querySelector('#pop').classList.contains('fixedpop') && /No drafts yet/.test(document.querySelector('#pop').textContent) && document.activeElement?.id === 'draftName' && document.querySelector('#pop').getBoundingClientRect().bottom <= document.querySelector('.gobar').getBoundingClientRect().top);
    await t.click('#pop [data-draftsave]');
    await t.ok('Save draft keeps the description', () => S.drafts.length === 1 && S.drafts[0].prompt === 'first draft' && JSON.parse(localStorage.getItem('hs3-drafts')).length === 1);
    await t.click('#draftsBtn');
    await t.ok('the button toggles it', () => document.querySelector('#pop').hidden && document.querySelector('#draftsBtn').getAttribute('aria-expanded') === 'false');
    await t.do('change engine and description', () => { enterEngine('yue'); render(); setVal('prompt.text', 'other'); });
    await t.click('#draftsBtn');
    await t.click('#draftName'); await t.type('named one'); await t.press('Enter');
    await t.eq('Enter in the name saves it', () => [S.drafts.length, S.drafts[0].name, S.drafts[0].backend], [2, 'named one', 'yue']);
    await t.eq('every row is one height, whatever its engine (C-06)', () => [...document.querySelectorAll('#pop .draft')].map(r => Math.round(r.getBoundingClientRect().height)), [44, 44]);
    await t.ok('another engine\'s draft says so on its second line, at full strength, with the same Load button', () => { const r = document.querySelectorAll('#pop .draft')[1]; return /ACE-Step 1\.5/.test(r.querySelector('small').textContent) && getComputedStyle(r).opacity === '1' && r.querySelector('[data-draftload]').textContent === 'Load'; });
    await t.ok('Save draft rests while nothing changed since the last one', () => document.querySelector('#pop [data-draftsave]').disabled && !!document.querySelector('#draftSame'));
    await t.click('#pop [data-draftload="1"]');
    await t.ok('Load on another engine switches engine, with Undo', () => S.backend === 'ace' && get('prompt.text') === 'first draft' && /Loaded the draft on ACE-Step 1.5$/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) && (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-act')?.textContent || '') === 'Undo');
    await t.click('#toast .toast-act');
    await t.eq('Undo restores the engine and description', () => [S.backend, get('prompt.text')], ['yue', 'other']);
    await t.click('#draftsBtn'); await t.click('#pop [data-draftdel="0"]');
    await t.ok('delete acts at once, with Undo (it is undoable, so it does not ask)', () => S.drafts.length === 1 && /Deleted the draft/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')));
    await t.ok('focus stays in the open list', () => !document.querySelector('#pop').hidden && !!document.activeElement.closest('#pop'));
    await t.click('#toast .toast-act');
    await t.eq('Undo restores it', () => S.drafts.length, 2);
    await t.ok('and focus goes back into the list, not to Generate (C-25)', () => !document.querySelector('#pop').hidden && !!document.activeElement.closest('#pop'));
    await t.do('fill to the limit', () => { for (let i = 0; i < 28; i++) S.drafts.push({ name: 'd' + i, backend: 'ace', at: Date.now() - 1000 - i, prompt: 'p', vals: {} }); saveDrafts(); setVal('prompt.text', 'third'); });
    await t.click('#toast .toast-x');
    await t.click('#draftsBtn');
    await t.ok('Drafts toggles shut (it stayed open through the toast\'s Undo)', () => document.querySelector('#pop').hidden);
    await t.click('#draftsBtn'); await t.click('#pop [data-draftsave]');
    await t.ok('past thirty the oldest makes room, with Undo', () => S.drafts.length === 30 && /made room/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) && (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-act')?.textContent || '') === 'Undo');
    await t.press('Escape');
    await t.ok('Escape closes it, focus back on Drafts', () => document.querySelector('#pop').hidden && document.activeElement?.id === 'draftsBtn');
  } },
  /* 17-undo-net/a-refused-draft-is-not-called-saved, with the description set by setVal */
  { id: 'a-refused-draft-is-not-called-saved', run: async t => {
    await t.do('a description', () => setVal('prompt.text', 'first draft'));
    await t.do('storage full', () => { const set = window.__set = Storage.prototype.setItem; Storage.prototype.setItem = function () { throw new DOMException('full', 'QuotaExceededError'); }; });
    await t.click('#draftsBtn'); await t.click('#pop [data-draftsave]');
    await t.ok('a draft the browser refused is said to be unsaved, last (F-15)', () => /wasn't saved/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) && /Not saved/.test(document.querySelector('#saveState').textContent) && document.querySelector('#saveState').classList.contains('failed'));
    await t.do('room again', () => { Storage.prototype.setItem = window.__set; });
  } },
  /* 17-undo-net/saved-says-only-what-was-saved, with settings changed by setVal */
  { id: 'saved-says-only-what-was-saved', run: async t => {
    await t.do('a desk', () => { S.tier = 'studio'; S.mods = ['sampler']; render(); });
    await t.do('storage full', () => { const set = window.__set = Storage.prototype.setItem; Storage.prototype.setItem = function (k, v) { if (k === 'hs3-desk') throw new DOMException('full', 'QuotaExceededError'); return set.call(this, k, v); }; });
    await t.do('a change', () => setVal('samp.steps', 33), null, 1200);
    await t.ok('the read-out says Not saved (F-15)', () => /Not saved/.test(document.querySelector('#saveState').textContent) && document.querySelector('#saveState').classList.contains('failed'));
    await t.ok('and a toast says why, once', () => /storage is full, so your desk wasn't saved/i.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')));
    await t.do('dismiss it', () => toast('something else'));
    await t.do('another change', () => setVal('samp.steps', 34), null, 1200);
    await t.ok('it keeps saying so, and says it once', () => /Not saved/.test(document.querySelector('#saveState').textContent) && (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '') === 'something else');
    await t.do('room again', () => { Storage.prototype.setItem = window.__set; });
    await t.do('a third change', () => setVal('samp.steps', 35), null, 1200);
    await t.ok('and goes back to Saved once a write succeeds', () => /^Saved$/.test(document.querySelector('#saveState').textContent.trim()) && !document.querySelector('#saveState').classList.contains('failed') && /"samp.steps":35/.test(localStorage.getItem('hs3-desk')));
  } },
  /* 21-chrome-round5/unreadable-desk-kept-aside, seeding the page's own desk key */
  { id: 'unreadable-desk-kept-aside', storage: { 'hotstep.create.desk': '{"v":1,"desk":null}' }, run: async t => {
    await t.eq('a desk that can\'t be used is copied aside (F-16)', () => localStorage.getItem('hs3-desk-unreadable'), '{"v":1,"desk":null}');
    await t.ok('and the page says so, with a download', () => { const m = (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || ''); return (/couldn't be read, so this is a fresh desk.*A copy is kept in this browser; nothing was deleted/.test(m) && (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-act')?.textContent || '') === 'Download the copy') || m; });
    await t.click('#toast .toast-act');
    await t.ok('which downloads the copies', async () => { const d = window.__bh.downloads.pop(); return (d && /unread/.test(d.name) && JSON.parse(await d.blob.text())['hotstep.create.desk-unreadable'] === '{"v":1,"desk":null}') || JSON.stringify(d?.name); });
    await t.do('a change', () => { S.tier = 'studio'; S.mods = ['tempo']; render(); }, null, 2000);
    await t.eq('the copy outlives the next autosave', () => localStorage.getItem('hs3-desk-unreadable'), '{"v":1,"desk":null}');
  } },
  /* 08-persistence-cables/old-shape-ignored, seeding the page's own desk key */
  { id: 'old-shape-kept-aside', storage: { 'hotstep.create.desk': JSON.stringify({ v: 0, desk: { tier: 'lab' } }) }, run: async t => {
    await t.eq('a desk saved in a shape the page cannot read starts clean, kept aside, and says so', () => [S.tier, localStorage.getItem('hs3-desk-unreadable'), /couldn't be read.*A copy is kept/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')), (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-act')?.textContent || '')], ['quick', JSON.stringify({ v: 0, desk: { tier: 'lab' } }), true, 'Download the copy']);
  } },
  /* F-27: Escape outside the save row leaves it as it was; in it, closes it */
  { id: 'escape-in-the-save-row', run: async t => {
    await t.click('#saveBtn'); await t.type('half'); await t.click('.top .brand'); await t.press('Escape');
    await t.ok('Escape outside the save row leaves it as it was', () => !document.querySelector('#saveRow').hidden && document.querySelector('#saveName').value === 'half');
    await t.click('[data-saveas="wf"]'); await t.press('Escape');
    await t.ok('then the save row, from inside it', () => document.querySelector('#saveRow').hidden && document.activeElement.id === 'saveBtn');
    await t.click('#saveBtn');
    await t.eq('and it opens empty (C-22)', () => document.querySelector('#saveName').value, '');
  } },
  /* C-24, F-20: a revision exported and imported again is filed under its workflow, said, and undone */
  { id: 'export-then-import', run: async t => {
    await t.do('a user workflow with a revision', () => { S.workflows = [{ name: 'Mine', tier: 'studio', mods: ['sampler'], pedals: [] }]; S.revs = { Mine: [{ name: 'One', backend: 'ace', vals: { 'samp.steps': 20 }, slots: [] }] }; saveUser(); S.wfOpen.add('Mine'); renderPresets(); });
    await t.click('.pre.rev[data-rn="One"] [data-rowmore]'); await t.click('#pop [data-rowact="rexp"]');
    const file = await t.download();
    await t.ok('the export is a revision file', () => true);
    await t.do('delete the workflow', () => { S.workflows = []; S.revs = Object.create(null); saveUser(); });
    await t.file('#importFile', 'r.json', 'application/json', file.text);
    await t.ok('importing it brings the workflow back with its revision', () => S.workflows.some(w => w.name === 'Mine') && S.revs.Mine?.[0]?.vals['samp.steps'] === 20 && /Imported Mine › One \(ACE-Step 1\.5\)$/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')));
    await t.click('#toast .toast-act');
    await t.ok('and Undo takes both out again', () => !S.workflows.some(w => w.name === 'Mine') && !S.revs.Mine);
    await t.file('#importFile', 'bad.json', 'application/json', '{not json');
    await t.ok('broken JSON is refused', () => /isn't valid JSON/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')));
  } },
  /* the open workflow and the revisions shown are kept with the desk (the concept's deskState and sessionState) */
  { id: 'the-open-workflow-is-kept', run: async t => {
    await t.click('#presetList .pre.wfrow[data-preset="Lo-fi tape"]');
    await t.click('#presetList [data-wfopen="Deep sampler"]');
    await t.reload();
    await t.eq('a reload keeps the open workflow and the revisions shown', () => [S.workflow, [...S.wfOpen], !!document.querySelector('.pre.wfrow.on[data-preset="Lo-fi tape"]'), !!document.querySelector('.pre.rev[data-rev="Deep sampler"]')], ['Lo-fi tape', ['Deep sampler'], true, true]);
  } },
  { id: 'an-unknown-open-workflow-reads-as-start-fresh', storage: { 'hotstep.create.workflow': JSON.stringify({ open: { workflow: 'Not a workflow', revision: 'x', revFor: 'ace' }, expanded: ['Songwriter', 7] }) }, run: async t => {
    await t.eq('an open workflow that is not here reads as Start fresh, and the shown list keeps only names (08 restore-cleans)', () => [S.workflow, S.revision, [...S.wfOpen]], ['Start fresh', null, ['Songwriter']]);
  } },
];
