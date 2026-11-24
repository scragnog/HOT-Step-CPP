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
];
