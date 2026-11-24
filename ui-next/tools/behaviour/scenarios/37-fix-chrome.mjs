// The fix round around the desk (review-r2 RC-01 to RC-18, review-r3 RA-04): each defect the round fixed, reached the
// way a user reaches it (real clicks, keys and files, a second tab opened in the same browser), on the rebuild only.
// They read the page's own storage keys (deviation 5); the second tab runs on the browser's clock, the first on the
// harness's, so a step in one is followed by a short real wait before the other is read.

const WF = 'hotstep.create.workflows', DESK = 'hotstep.create.desk';
const revRow = (wf, name) => `.pre.rev[data-rev="${wf}"][data-rn="${name}"]`;
/* the user's revisions under a workflow, as the list shows them (rows with a menu) and as storage holds them; it runs in
   the page, so it holds everything it uses */
const mine = wf => ({
  shown: [...document.querySelectorAll('.pre.rev')].filter(x => x.dataset.rev === wf && x.querySelector('[data-rowmore]')).map(x => x.dataset.rn).sort(),
  stored: (JSON.parse(localStorage.getItem('hotstep.create.workflows') || '{}').revisions || []).filter(r => r.workflow === wf).map(r => r.name).sort(),
});
const both = (...names) => ({ shown: names, stored: names });
const toastIs = re => new RegExp(re).test(document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '');

/** Saves the desk as a revision of the open workflow from the save row, as a person does. */
async function saveRev(t, name) { await t.click('#saveBtn'); await t.click('[data-saveas="rev"]'); await t.type(name); await t.press('Enter'); }
/** Runs an action of a revision's row menu (Delete is pressed twice: once to arm it). */
async function rowAct(t, wf, name, act) {
  await t.click(`${revRow(wf, name)} [data-rowmore]`);
  await t.click(`#pop [data-rowact="${act}"]`);
  if (act === 'rdel') await t.click('#pop [data-rowact="rdel"]');
}

/* a second tab of the same browser, on the same page; real time, so each step waits for the page to draw */
async function otherTab(t) {
  const p = await t.page.context().newPage();
  await p.goto(t.page.url()); await p.waitForTimeout(700);
  return { p, mine: () => p.evaluate(mine, 'Start fresh'), click: async sel => { if (sel.endsWith(' [data-rowmore]')) await p.locator(sel.slice(0, -15)).first().hover(); await p.locator(sel).first().click(); await p.waitForTimeout(250); }, type: async s => { await p.keyboard.type(s); await p.waitForTimeout(250); }, press: async k => { await p.keyboard.press(k); await p.waitForTimeout(250); }, eval: (fn, a) => p.evaluate(fn, a) };
}

export default [
  /* RC-02: a revision saved here, then deleted or renamed, stays deleted or renamed, after a reload too; Undo and Redo
     do what they say. The package's own scenario seeded its revision through storage, so it never saw a revision the
     page had made itself merged with its own copy read back. */
  { id: 'rc02-delete-and-rename-a-saved-revision-stick', run: async t => {
    const SF = 'Start fresh';
    await saveRev(t, 'My song');
    await t.eq('the revision is saved under Start fresh, and kept', mine, both('My song'), SF);
    await rowAct(t, SF, 'My song', 'rdel');
    await t.eq('Delete takes it out of the list and out of storage', mine, both(), SF);
    await t.ok('and says so, with Undo', toastIs, 'Deleted Start fresh › My song');
    await t.click('#toast .toast-act');
    await t.eq('Undo puts it back', mine, both('My song'), SF);
    await t.click('#toast .toast-act');
    await t.eq('Redo deletes it again', mine, both(), SF);
    await t.reload();
    await t.eq('and it stays deleted after a reload', mine, both(), SF);
    await saveRev(t, 'My song');
    await rowAct(t, SF, 'My song', 'rren');
    await t.type('Bee'); await t.press('Enter');
    await t.eq('Rename leaves one revision, under its new name, in the list and in storage', mine, both('Bee'), SF);
    await t.click('#toast .toast-act');
    await t.eq('Undo gives it its old name back', mine, both('My song'), SF);
    await t.ok('and does not say a name is taken', () => !/taken/.test(document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.click('#toast .toast-act');
    await t.eq('Redo renames it again', mine, both('Bee'), SF);
    await t.ok('and does not say a name is taken either', () => !/taken/.test(document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.reload();
    await t.eq('and after a reload it is one revision, Bee', mine, both('Bee'), SF);
  } },
  /* RC-02 with a file: a revision exported, deleted, imported back, renamed, then deleted stays deleted */
  { id: 'rc02-an-imported-revision-deletes', run: async t => {
    const SF = 'Start fresh';
    await saveRev(t, 'Filed');
    await rowAct(t, SF, 'Filed', 'rexp');
    const file = await t.download();
    await rowAct(t, SF, 'Filed', 'rdel');
    await t.file('#importFile', file.name, 'application/json', file.text);
    await t.eq('the file comes back in', mine, both('Filed'), SF);
    await rowAct(t, SF, 'Filed', 'rren');
    await t.press('End'); await t.type(' again'); await t.press('Enter');
    await t.eq('renamed once imported, it is one revision', mine, both('Filed again'), SF);
    await rowAct(t, SF, 'Filed again', 'rdel');
    await t.reload();
    await t.eq('deleted, it stays deleted after a reload', mine, both(), SF);
  } },
  /* RC-02 across two tabs, in both orders: each tab's delete and rename reach the other, and neither brings back what the
     other took out */
  { id: 'rc02-two-tabs-delete-and-rename', run: async t => {
    const SF = 'Start fresh';
    await saveRev(t, 'One'); await saveRev(t, 'Two');
    const b = await otherTab(t);
    await t.eq('the other tab opens with both', () => true, await b.mine().then(m => JSON.stringify(m) === JSON.stringify(both('One', 'Two')) || m));
    await b.click(`${revRow(SF, 'One')} [data-rowmore]`); await b.click('#pop [data-rowact="rdel"]'); await b.click('#pop [data-rowact="rdel"]');
    await t.settle(300);
    await t.eq('a delete in the other tab reaches this one', mine, both('Two'), SF);
    await rowAct(t, SF, 'Two', 'rren'); await t.press('End'); await t.type('2'); await t.press('Enter');
    await b.p.waitForTimeout(300);
    await t.eq('a rename here does not bring the deleted one back', mine, both('Two2'), SF);
    await t.eq('and the other tab shows the rename', () => true, await b.mine().then(m => JSON.stringify(m) === JSON.stringify(both('Two2')) || m));
    await b.click(`${revRow(SF, 'Two2')} [data-rowmore]`); await b.click('#pop [data-rowact="rren"]'); await b.press('End'); await b.type('b'); await b.press('Enter');
    await t.settle(300);
    await saveRev(t, 'Five');
    await t.eq('a rename there and a save here: both kept, nothing back from before', mine, both('Five', 'Two2b'), SF);
    await b.p.reload(); await b.p.waitForTimeout(800);
    await t.eq('the other tab reloads with the same two', () => true, await b.mine().then(m => JSON.stringify(m) === JSON.stringify(both('Five', 'Two2b')) || m));
    await b.p.close();
    await t.reload();
    await t.eq('and so does this one', mine, both('Five', 'Two2b'), SF);
  } },
  /* RC-08: the desk in two tabs (C-09). A tab keeps its own desk when another tab writes, says so, and offers the other
     one; a reload shows the tab's own desk and offers the newer one again; Use that desk is one step Undo takes back; a
     tab that only looks writes nothing; a new tab opens with the desk changed last. */
  { id: 'rc08-two-tabs-keep-their-own-desk', run: async t => {
    const desk = () => { const d = JSON.parse(localStorage.getItem('hotstep.create.desk') || '{}'); return { prompt: (d.vals || {})['prompt.text'] ?? '', tier: d.tier }; };
    const here = () => ({ prompt: document.querySelector('#prompt').value, tier: document.querySelector('#tiers [aria-checked="true"]')?.textContent.trim() });
    await t.click('#prompt'); await t.type('from tab A');
    await t.click('.top .brand');
    const markA = await t.js(() => JSON.parse(localStorage.getItem('hotstep.create.desk')).tab);
    const b = await otherTab(t);
    await t.eq('a tab opened later shows the desk changed last', () => true, await b.eval(() => document.querySelector('#prompt').value).then(v => v === 'from tab A' || v));
    await t.eq('and, only looking, writes nothing over it', () => JSON.parse(localStorage.getItem('hotstep.create.desk')).tab, markA);
    await b.click('#tiers [role="radio"]:last-child');
    await b.click('#prompt'); await b.p.keyboard.press('Control+a'); await b.p.keyboard.press('Delete'); await b.p.waitForTimeout(300);
    await t.page.bringToFront(); await t.settle(300);
    await t.eq('this tab still shows its own desk', here, { prompt: 'from tab A', tier: 'Quick' });
    await t.ok('and says the other tab has a newer one, offering it', () => /Another tab has a newer desk\. This one still shows its own\./.test(document.querySelector('#toast').innerText) && /Use that desk/.test(document.querySelector('#toast').innerText) || document.querySelector('#toast').innerText);
    await t.eq('storage holds the desk changed last', desk, { prompt: '', tier: 'lab' });
    await t.reload();
    await t.eq('reloaded, this tab shows its own desk, not the other tab\'s', here, { prompt: 'from tab A', tier: 'Quick' });
    await t.ok('and offers the newer one again', () => /Another tab has a newer desk/.test(document.querySelector('#toast').innerText) || document.querySelector('#toast').innerText);
    await t.eq('which is still in storage, not written over', desk, { prompt: '', tier: 'lab' });
    await t.click('#toast .toast-act');
    await t.eq('Use that desk puts the other tab\'s desk here', here, { prompt: '', tier: 'Lab' });
    await t.ok('and says so, with Undo', () => /Now showing the other tab's desk/.test(document.querySelector('#toast').innerText) || document.querySelector('#toast').innerText);
    await t.click('#toast .toast-act');
    await t.eq('Undo puts this tab\'s own desk back', here, { prompt: 'from tab A', tier: 'Quick' });
    await t.eq('which is now the desk changed last', desk, { prompt: 'from tab A', tier: 'quick' });
    await b.p.bringToFront(); await b.p.waitForTimeout(300);
    await t.eq('and the other tab, keeping its own, is offered it in turn', () => true, await b.eval(() => [document.querySelector('#prompt').value, /Another tab has a newer desk/.test(document.querySelector('#toast').innerText)]).then(v => (v[0] === '' && v[1]) || v));
    await b.p.close();
  } },
  /* RA-04 and RC-09: the Drafts list is the page's one popover, so a layer: Ctrl+Z says to close it first and changes
     nothing behind it, Ctrl+Enter starts no run, Shift+Tab out of it closes it, and nothing else is ever drawn into #pop
     beside it */
  { id: 'ra04-drafts-is-the-one-popover-and-a-layer', run: async t => {
    await t.click('#browser [data-additem="sampler"]');
    await t.click('#prompt'); await t.type('a description');
    await t.click('#draftsBtn');
    await t.ok('Drafts opens with focus in its name field', () => !document.querySelector('#pop').hidden && document.activeElement?.id === 'draftName' && document.querySelector('#draftsBtn').getAttribute('aria-expanded') === 'true');
    await t.press('Control+z');
    await t.ok('Ctrl+Z says to close it first', () => /Close this first to undo on the desk/.test(document.querySelector('#toast').innerText) || document.querySelector('#toast').innerText);
    await t.eq('and changes nothing behind it', () => document.querySelector('#prompt').value, 'a description');
    await t.focus('#draftName');
    await t.press('Control+Enter');
    await t.ok('Ctrl+Enter starts no run under it', () => !/^\(\d+\)/.test(document.title) || document.title);
    await t.press('Shift+Tab');
    await t.ok('Shift+Tab out of it closes it, focus back on Drafts', () => document.querySelector('#pop').hidden && document.activeElement?.id === 'draftsBtn' && document.querySelector('#draftsBtn').getAttribute('aria-expanded') === 'false');
    await t.click('#draftsBtn');
    await t.focus('#bento [data-move]'); await t.press('Enter');
    await t.ok('the moves of a tile opened while it is open take its place: #pop never holds two things', () => (!document.querySelector('#pop').hidden && !document.querySelector('#pop #draftName') && document.querySelector('#draftsBtn').getAttribute('aria-expanded') === 'false') || document.querySelector('#pop').innerText.slice(0, 120));
    await t.press('Escape');
    await t.click('#draftsBtn');
    await t.press('Escape');
    await t.ok('one Escape closes it and gives focus back', () => document.querySelector('#pop').hidden && document.activeElement?.id === 'draftsBtn' && !document.querySelector('#pop').children.length);
    await t.press('Control+Enter');
    await t.ok('with it closed, Ctrl+Enter is Generate again', () => /^\(\d+\)/.test(document.title) || document.title);
  } },
  /* RC-03: the row menu is the same popover: the arrows move one item, Tab leaves it closed, Escape closes it, not a toast */
  { id: 'rc03-row-menu-keys', run: async t => {
    await saveRev(t, 'Keys');
    await t.click(`${revRow('Start fresh', 'Keys')} [data-rowmore]`);
    const at = () => document.activeElement?.dataset.rowact ?? document.activeElement?.id ?? document.activeElement?.className;
    await t.eq('the menu opens on its first item', at, 'rren');
    await t.press('ArrowDown'); await t.eq('ArrowDown moves one item', at, 'rexp');
    await t.press('ArrowDown'); await t.eq('and one more', at, 'rdel');
    await t.press('ArrowDown'); await t.eq('and round to the first', at, 'rren');
    await t.press('ArrowUp'); await t.eq('ArrowUp moves one back, round to the last', at, 'rdel');
    await t.press('Tab');
    await t.ok('Tab past its last item closes it, focus not left in a toast', () => document.querySelector('#pop').hidden && !document.activeElement?.closest('#toast, #pop'));
    await t.click(`${revRow('Start fresh', 'Keys')} [data-rowmore]`);
    await t.press('Escape');
    await t.ok('Escape closes it, focus back on its ⋯', () => document.querySelector('#pop').hidden && document.activeElement?.matches('[data-rowmore]'));
  } },
  /* RC-09: Ctrl+Enter in the save row's name field saves, as Enter does there, and starts no run (the concept's field
     takes its Enter); Ctrl+Z there is the field's, then the desk's (the concept's save row is no layer) */
  { id: 'rc09-ctrl-enter-in-the-save-row-saves', run: async t => {
    await t.click('#prompt'); await t.type('words');
    await t.click('#saveBtn'); await t.click('[data-saveas="rev"]'); await t.type('By keys');
    await t.press('Control+Enter');
    await t.ok('Ctrl+Enter saves the revision', () => [...document.querySelectorAll('.pre.rev')].some(x => x.dataset.rn === 'By keys') && document.querySelector('#saveRow').hidden);
    await t.ok('and starts no run', () => !/^\(\d+\)/.test(document.title) || document.title);
  } },
  /* RC-05: the song menu closed by a click outside gives focus back to its ⋯, as Escape does */
  { id: 'rc05-song-menu-outside-click-focus', run: async t => {
    const id = await t.js(() => [...document.querySelectorAll('#latest .song[data-id]')].find(s => s.offsetParent && s.querySelector('[data-songmenu]'))?.dataset.id);
    await t.click(`#latest .song[data-id="${id}"] [data-songmenu]`);
    await t.ok('the menu opens', () => !document.querySelector('#menu').hidden);
    await t.click('#rightPanel .side-head h2');
    await t.ok('a click outside closes it and puts focus back on its button', id => (document.querySelector('#menu').hidden && document.activeElement?.matches(`#latest .song[data-id="${id}"] [data-songmenu]`)) || document.activeElement?.outerHTML.slice(0, 80), id);
  } },
  /* RC-06: the engine panel is a layer: F9 closes it as the picker opens, and Escape from the picker lands on Engine */
  { id: 'rc06-picker-closes-the-engine-panel', run: async t => {
    await t.click('#engBtn');
    await t.ok('the engine panel opens', () => !document.querySelector('#engPanel').hidden);
    await t.press('F9');
    await t.ok('F9 opens the picker and closes the panel under it', () => !document.querySelector('#pickerWrap').hidden && document.querySelector('#engPanel').hidden);
    await t.press('Escape', 400);
    await t.ok('Escape closes the picker, focus on the Engine button', () => (document.querySelector('#pickerWrap').hidden && document.querySelector('#engPanel').hidden && document.activeElement?.id === 'engBtn') || document.activeElement?.outerHTML.slice(0, 80));
    await t.click('#engBtn'); await t.press('Escape');
    await t.ok('Escape closes the panel itself, focus back on Engine', () => document.querySelector('#engPanel').hidden && document.activeElement?.id === 'engBtn');
  } },
  /* RC-07: Redo of an add puts focus on the tile that came back */
  { id: 'rc07-redo-of-an-add-focuses-the-tile', run: async t => {
    await t.click('#browser [data-additem="tempo"]');
    await t.click('.top .brand');
    await t.press('Control+z');
    await t.ok('Undo takes it off', () => !document.querySelector('#bento .tile[data-m="tempo"]'));
    await t.press('Control+Shift+z');
    await t.ok('Redo puts it back with focus on its grip', () => document.activeElement?.matches('.tile[data-m="tempo"] .grip') || document.activeElement?.outerHTML.slice(0, 80));
  } },
  /* RC-10: at 360 on touch, a module added from the Modules drawer is brought into view with focus on its grip */
  { id: 'rc10-added-from-a-drawer-comes-into-view', w: 360, h: 740, touch: true, run: async t => {
    await t.click('[data-open="right"]');
    await t.click('#browser [data-additem="seed"]');
    await t.ok('the drawer closes, focus is on the new tile\'s grip', () => (!document.querySelector('#rightPanel.open') && document.activeElement?.matches('.tile[data-m="seed"] .grip')) || document.activeElement?.outerHTML.slice(0, 80));
    await t.ok('and the tile is in view', () => { const r = document.querySelector('.tile[data-m="seed"]').getBoundingClientRect(); return (r.top < innerHeight && r.bottom > 0) || [r.top, r.bottom, innerHeight]; });
  } },
  /* RC-11: the tile being read stays where it was on screen through a tier change (Q-14); the motion is package 9's */
  { id: 'rc11-the-tile-being-read-stays-put', run: async t => {
    await t.click('.pre.wfrow[data-preset="Everything"]');
    await t.click('#tiers [data-tier="studio"]');
    await t.js(() => { const g = document.querySelector('.tile[data-m="sampler"]'); scrollTo(0, scrollY + g.getBoundingClientRect().top - 514); });
    await t.focus('.tile[data-m="sampler"] .grip');
    const y0 = await t.js(() => Math.round(document.querySelector('.tile[data-m="sampler"]').getBoundingClientRect().top));
    const same = y0 => { const y = Math.round(document.querySelector('.tile[data-m="sampler"]').getBoundingClientRect().top); return Math.abs(y - y0) <= 1 || `${y} where it was ${y0}`; };
    await t.press('Alt+3');
    await t.ok('at Lab the Sampler is where it was', same, y0);
    await t.press('Alt+1');
    await t.ok('and at Quick too', same, y0);
  } },
  /* RC-13: a Modules filter that matches nothing says so, with Clear the filter */
  { id: 'rc13-an-empty-filter-says-so', run: async t => {
    await t.click('#brSearch'); await t.type('zzz');
    await t.eq('the panel says nothing matches', () => document.querySelector('#brList .br-none')?.textContent, 'No modules match “zzz”. Clear the filter');
    await t.click('#brList [data-brclear]');
    await t.ok('Clear the filter empties it, the cards come back, focus in the filter', () => document.querySelector('#brSearch').value === '' && document.querySelectorAll('#brList .bcard').length > 10 && document.activeElement?.id === 'brSearch');
  } },
  /* RC-14: the hidden Lab modules are named in the desk's order */
  { id: 'rc14-hidden-modules-in-desk-order', run: async t => {
    await t.click('.pre.wfrow[data-preset="Deep sampler"]');
    await t.click('#tiers [data-tier="quick"]');
    await t.eq('the note names them as the desk orders them', () => document.querySelector('#hiddenNote span')?.textContent, 'Hidden at this tier, still applied: Guidance plugin, Performance trades, Latent & correction and Repetition guard');
  } },
  /* RC-15: a toast showing before a modal opens is not in the modal's Tab cycle (its Undo would act under it) */
  { id: 'rc15-an-earlier-toast-stays-out-of-a-modal', run: async t => {
    await t.click('#browser [data-additem="tempo"]');
    await t.click('.tile[data-m="tempo"] .x');
    await t.ok('a toast with Undo shows', () => !!document.querySelector('#toast:not([hidden]) .toast-act'));
    await t.click('#notesBtn');
    const seen = [];
    for (let i = 0; i < 4; i++) { await t.press('Tab', 50); seen.push(await t.js(() => (document.activeElement?.closest('#toast') ? 'toast' : document.activeElement?.closest('#notes') ? 'notes' : 'out'))); }
    await t.eq('Tab goes round the notes only', () => true, seen.every(x => x === 'notes') || seen);
  } },
  /* RC-16: a prompt tool's Undo says what it puts back, and that anything else stays (as every desk step does) */
  { id: 'rc16-a-tool-undo-says-the-rest-stays', run: async t => {
    await t.click('#ideasBtn');
    await t.click('#pop [data-preset2]');
    await t.ok('Style preset\'s Undo ends with what stays', () => { const b = document.querySelector('#undoBtn'); b.dispatchEvent(new PointerEvent('pointerenter')); return /Puts back the description\. Anything else you did since stays as it is\.?$/.test(b.title) || b.title; });
  } },
  /* RC-17: a file whose workflow cannot be read says where the revision went */
  { id: 'rc17-an-unreadable-workflow-is-said', run: async t => {
    const f = JSON.stringify({ format: 'hotstep-revision', v: 1, workflow: '<img src=x onerror=alert(1)>', revision: { name: '<b>bold</b>x', engine: 'ace', vals: { 'gd.cfg': 1e9 } } });
    await t.file('#importFile', 'odd.json', 'application/json', f);
    await t.ok('the toast says where it went and that its workflow could not be read', () => (/Start fresh › <b>bold<\/b>x/.test(document.querySelector('#toast .toast-msg').textContent) && /the workflow the file names could not be read/.test(document.querySelector('#toast .toast-msg').textContent)) || document.querySelector('#toast .toast-msg').textContent);
    await t.ok('and no markup ran', () => !document.querySelector('img[src="x"]'));
  } },
  /* RC-01: the navigation shows each other page's stand-in, moves aria-current to it, and comes back */
  { id: 'rc01-the-navigation-goes-somewhere', run: async t => {
    const where = () => ({ app: !document.getElementById('app').hidden, other: !document.getElementById('other').hidden, current: document.querySelector('.top [aria-current="page"]')?.dataset.nav, h: document.querySelector('#other h2')?.textContent ?? null });
    await t.click('.top [data-nav="Studios"]');
    await t.eq('Studios shows its stand-in and is the current page', where, { app: false, other: true, current: 'Studios', h: 'Studios' });
    await t.ok('which says the page keeps its own place, with the way back', () => /Studios keeps its own page/.test(document.querySelector('#other p').textContent) && /Back to Create/.test(document.querySelector('#other [data-goto="create"]').textContent));
    await t.press('Control+z');
    await t.ok('the desk\'s Undo does nothing there', () => document.querySelector('#toast').hidden || !/Nothing to undo|Undone/.test(document.querySelector('#toast').textContent));
    await t.click('#other [data-goto="create"]');
    await t.eq('Back to Create shows Create again', where, { app: true, other: false, current: 'create', h: null });
    await t.ok('focus goes to Create in the top bar, not to the hidden link', () => document.activeElement?.matches('.top [data-nav="create"]') || document.activeElement?.outerHTML.slice(0, 80));
    await t.click('.res-head [data-goto="Library"]');
    await t.eq('Latest\'s Open the Library opens the Library', where, { app: false, other: true, current: 'Library', h: 'Library' });
    await t.ok('with focus on Library in the top bar', () => document.activeElement?.matches('.top [data-nav="Library"]') || document.activeElement?.outerHTML.slice(0, 80));
    await t.click('.top .brand');
    await t.eq('the brand goes back to Create', where, { app: true, other: false, current: 'create', h: null });
  } },
  /* RC-04: with no room above or below and no page to move, the song menu stands beside its card (F-34), clear of the
     go bar (F-28), as tall as that room allows */
  { id: 'rc04-song-menu-beside-its-card', run: async t => {
    const id = await t.js(() => [...document.querySelectorAll('#latest .song[data-id]')].find(s => s.offsetParent && s.querySelector('[data-songmenu]'))?.dataset.id);
    await t.click(`#latest .song[data-id="${id}"] [data-songmenu]`);
    await t.ok('it covers neither its card nor the go bar, and has the room beside it', id => {
      const m = document.querySelector('#menu').getBoundingClientRect(), c = document.querySelector(`#latest .song[data-id="${id}"]`).getBoundingClientRect(), g = document.querySelector('.gobar').getBoundingClientRect();
      const over = (a, b) => a.left < b.right - 0.5 && a.right > b.left + 0.5 && a.top < b.bottom - 0.5 && a.bottom > b.top + 0.5;
      return (!over(m, c) && !over(m, g) && m.left >= c.right && m.height >= 280) || JSON.stringify({ menu: [m.left, m.top, m.width, m.height], card: [c.left, c.right], gobar: [g.top, g.bottom] });
    }, id);
  } },
];
