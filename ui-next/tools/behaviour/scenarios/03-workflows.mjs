// Workflows and revisions: load, open in tier, revisions, save, replace, update, revert, rename, delete, export,
// import, the row menu, and the older storage shapes.
const toast = () => document.querySelector('#toast .toast-msg')?.textContent || '';
export default [
  { id: 'load-factory', run: async t => {
    await t.click('#presetList .pre.wfrow[data-preset="Lo-fi tape"]');
    await t.eq('a workflow row loads its modules and pedals, not its tier', () => [S.workflow, S.mods.join(), S.pedals.join(), S.tier], ['Lo-fi tape', 'adapters,tempo,sampler,post', 'nat,norm', 'quick']);
    await t.ok('with an Undo toast', () => /Opened Lo-fi tape/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.click('#presetList [data-presettier="Deep sampler"]');
    await t.eq('open in its own tier', () => [S.workflow, S.tier], ['Deep sampler', 'lab']);
    await t.focus('#presetList .pre.wfrow[data-preset="Seed hunter"]'); await t.press('Enter');
    await t.eq('Enter on a row loads it', () => S.workflow, 'Seed hunter');
    await t.focus('#presetList .pre.wfrow[data-preset="Songwriter"]'); await t.press(' ');
    await t.eq('Space on a row loads it', () => S.workflow, 'Songwriter');
    await t.click('#toast .toast-act');
    await t.eq('Undo goes back to the previous desk', () => S.workflow, 'Seed hunter');
  } },
  { id: 'revisions', run: async t => {
    await t.click('#presetList [data-wfopen="Deep sampler"]');
    await t.ok('the chevron shows revisions', () => !!document.querySelector('.pre.rev[data-rev="Deep sampler"][data-rn="Polishing"]'));
    await t.click('.pre.rev[data-rev="Deep sampler"][data-rn="Polishing"] b');
    await t.eq('a revision row loads its workflow and settings', () => [S.workflow, S.revision, S.backend, S.vals['samp.steps'], S.vals['samp.solver']], ['Deep sampler', 'Polishing', 'ace', 40, 'DPM++ 3M']);
    await t.click('.pre.rev[data-rn="Polishing"] [data-rinfo]');
    await t.ok('i shows what it holds', () => /Steps/.test(document.querySelector('.pdetail')?.textContent || '') && /DPM\+\+ 3M/.test(document.querySelector('.pdetail').textContent));
    await t.click('.pre.rev[data-rn="Polishing"] [data-rinfo]');
    await t.ok('and hides it again', () => !document.querySelector('.pdetail'));
    await t.focus('.pre.rev[data-rev="Deep sampler"][data-rn="Faithful render"]'); await t.press('Enter');
    await t.eq('Enter loads a revision, switching engine', () => [S.revision, S.backend], ['Faithful render', 'mm3']);
    await t.do('change a setting', () => setVal('samp.steps', 22));
    await t.ok('a changed revision gets Revert and Save as', () => !!document.querySelector('.pre.rev.mod [data-rrevert]') && !!document.querySelector('.pre.rev.mod [data-saveasnew]'));
    await t.click('.pre.rev.mod [data-rrevert]');
    await t.eq('Revert loads it again', () => [S.vals['samp.steps'], S.revDirty], [30, false]);
    await t.click('.pre.rev[data-rn="Faithful render"] [data-rexp]');
    await t.ok('export writes the revision as JSON', async () => { const d = window.__bh.downloads.at(-1); if (!d) return 'no download'; const j = JSON.parse(await d.blob.text()); return d.name === 'Deep_sampler-Faithful_render.mm3.json' && j.format === 'hotstep-revision' && j.revision.vals['samp.steps'] === 30 && j.workflow.name === 'Deep sampler'; });
    await t.ok('and says so', () => /Exported Deep sampler › Faithful render/.test(document.querySelector('#toast .toast-msg').textContent));
  } },
  { id: 'save-revision-and-workflow', run: async t => {
    await t.click('#presetList .pre.wfrow[data-preset="Seed hunter"]');
    await t.do('settings', () => { setVal('samp.steps', 31); setVal('takes.n', 2); });
    await t.click('#saveBtn');
    await t.ok('Save opens the row as a revision of the open workflow', () => !document.querySelector('#saveRow').hidden && document.querySelector('[data-saveas="rev"]').getAttribute('aria-pressed') === 'true' && document.activeElement.id === 'saveName');
    await t.press('Enter');
    await t.ok('no name, no save: the row says so beside the field, and focus stays in it', () => /Give it a name first/.test(document.querySelector('#saveHint').textContent) && document.querySelector('#saveHint').classList.contains('warn') && document.activeElement.id === 'saveName');
    await t.type('Night run'); await t.press('Enter');
    await t.ok('Enter saves the revision', () => S.revs['Seed hunter']?.[0]?.name === 'Night run' && S.revision === 'Night run' && document.querySelector('#saveRow').hidden);
    await t.ok('it is stored', () => JSON.parse(localStorage.getItem('hs3-wf')).revs['Seed hunter'][0].vals['samp.steps'] === 31);
    await t.click('.pre.rev[data-rn="Night run"] [data-rinfo]');
    await t.eq('its details list the settings in order', () => [...document.querySelectorAll('.pdetail .pd-g span')].map(s => s.textContent), ['Steps31', 'Takes per run2']);
    await t.click('#saveBtn'); await t.click('#saveName'); await t.type('night RUN'); await t.click('#saveOk');
    await t.ok('the same name asks to replace', () => document.querySelector('#saveOk').textContent === 'Replace' && /already exists/.test(document.querySelector('#saveHint').textContent));
    await t.click('#saveOk');
    await t.eq('Replace overwrites it, keeping the name as it was', () => S.revs['Seed hunter'].map(r => r.name), ['Night run']);
    await t.ok('and offers Undo', () => /Replaced Seed hunter › Night run/.test(document.querySelector('#toast .toast-msg').textContent) && document.querySelector('#toast .toast-act')?.textContent === 'Undo' && /Replace/.test(document.querySelector('#undoBtn span').textContent));
    await t.click('#saveBtn'); await t.click('[data-saveas="wf"]');
    await t.ok('New workflow switches the row', () => document.querySelector('[data-saveas="wf"]').getAttribute('aria-pressed') === 'true' && document.querySelector('#saveInc').hidden);
    await t.click('#saveName'); await t.type('Songwriter'); await t.press('Enter');
    await t.ok('a factory name is refused', () => /belongs to a factory workflow/.test(document.querySelector('#saveHint').textContent));
    await t.do('clear name', () => { document.querySelector('#saveName').value = ''; });
    await t.click('#saveName'); await t.type('My desk'); await t.press('Enter');
    await t.ok('a new workflow is saved and opened', () => S.workflows[0]?.name === 'My desk' && S.workflow === 'My desk' && !!document.querySelector('.pre.wfrow[data-preset="My desk"]'));
    await t.click('#toast .toast-act');
    await t.ok('its toast offers a revision too', () => !document.querySelector('#saveRow').hidden && document.querySelector('[data-saveas="rev"]').getAttribute('aria-pressed') === 'true');
    await t.press('Escape');
    await t.ok('Escape closes the save row', () => document.querySelector('#saveRow').hidden && document.activeElement.id === 'saveBtn');
    await t.click('#saveBtn'); await t.click('#saveBtn');
    await t.ok('the Save button toggles the row', () => document.querySelector('#saveRow').hidden);
  } },
  { id: 'update-revert-saveasnew', run: async t => {
    await t.do('a user workflow', () => { S.workflows = [{ name: 'Mine', tier: 'studio', mods: ['tempo', 'seed'], pedals: [] }]; saveUser(); loadPreset(S.workflows[0]); });
    await t.click('#browser .bcard[data-additem="takes"]');
    await t.ok('a changed desk marks its workflow', () => S.wfDirty && !!document.querySelector('.pre.wfrow.mod[data-preset="Mine"] [data-wfupd="Mine"]'));
    await t.click('[data-wfupd="Mine"]');
    await t.eq('Update saves the layout', () => [S.workflows[0].mods.join(), S.wfDirty], ['tempo,seed,takes', false]);
    await t.click('#toast .toast-act');
    await t.eq('and Undo takes it back', () => S.workflows[0].mods.join(), 'tempo,seed');
    await t.click('#browser .bcard[data-additem="avoid"]');
    await t.click('[data-revert="Mine"]');
    await t.eq('Revert puts the modules back', () => [S.mods.join(), S.wfDirty], ['tempo,seed', false]);
    await t.click('#presetList .pre.wfrow[data-preset="Radio master"]');
    await t.click('#browser .bcard[data-additem="seed"]');
    await t.click('[data-saveasnew="wf|Radio master"]');
    await t.ok('Save as… on a factory row opens the save row with a name', () => !document.querySelector('#saveRow').hidden && document.querySelector('#saveName').value === 'Radio master (mine)' && document.querySelector('[data-saveas="wf"]').getAttribute('aria-pressed') === 'true');
  } },
  { id: 'rename-delete-menu', run: async t => {
    await t.do('two user workflows with a revision', () => { S.workflows = [{ name: 'Alpha', tier: 'studio', mods: ['tempo'], pedals: [] }, { name: 'Beta', tier: 'lab', mods: ['seed'], pedals: [] }]; S.revs = { Alpha: [{ name: 'r1', backend: 'ace', vals: { 'samp.steps': 20 }, slots: [] }] }; saveUser(); S.wfOpen.add('Alpha'); renderPresets(); });
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
  /* F-02: one name rule (trimmed, any capitals) for rename, save and import; a rename never files two workflows under one
     key, so none is lost, and it can be undone */
  { id: 'one-name-rule', run: async t => {
    await t.do('two user workflows', () => { S.workflows = [{ name: 'abc', tier: 'quick', mods: ['tempo'], pedals: [] }, { name: 'xyz', tier: 'quick', mods: ['tempo', 'seed'], pedals: [] }]; S.revs = Object.create(null); saveUser(); renderPresets(); });
    const ren = async (from, to) => { await t.click(`.pre.wfrow[data-preset="${from}"] [data-rowmore]`); await t.click('#pop [data-rowact="wfren"]'); await t.press('Control+a'); await t.type(to); await t.press('Enter'); };
    await ren('xyz', 'ABC');
    /* F-11: the refusal is said beside the field, not in a toast */
    await t.ok('a name that differs only in capitals is refused beside the field, and both workflows are kept', () => /"abc" workflow already exists/.test(document.querySelector('#prenHint:not([hidden])')?.textContent) && document.querySelector('.pren').getAttribute('aria-describedby') === 'prenHint' && document.querySelector('#toast').hidden && document.querySelector('.pren')?.getAttribute('aria-invalid') === 'true' && JSON.parse(localStorage.getItem('hs3-wf')).workflows.length === 2);
    await t.press('Escape');
    await ren('xyz', ' songwriter ');
    await t.ok('a factory name in other capitals is refused with the save row\'s words', () => /belongs to a factory workflow/.test(document.querySelector('#prenHint:not([hidden])')?.textContent) && S.workflows.map(w => w.name).join() === 'abc,xyz');
    await t.press('Escape');
    await ren('abc', 'ABC');
    await t.ok('a workflow can change its own capitals', () => S.workflows.map(w => w.name).join() === 'ABC,xyz' && JSON.parse(localStorage.getItem('hs3-wf')).workflows.length === 2);
    await t.ok('the rename toasts with Undo', () => /Renamed "abc" to "ABC"/.test(document.querySelector('#toast .toast-msg').textContent) && !!document.querySelector('#toast .toast-act'));
    await t.click('#undoBtn');
    await t.eq('and Undo puts the old name back', () => S.workflows.map(w => w.name).join(), 'abc,xyz');
    await t.file('#importFile', 'r.json', 'application/json', JSON.stringify({ workflow: { name: 'SONGWRITER', tier: 'quick', mods: ['tempo'], pedals: [] }, revision: { name: 'From file', backend: 'ace', vals: {} } }));
    await t.ok('an import names an existing workflow whatever its capitals', () => S.revs.Songwriter?.[0]?.name === 'From file' && !S.workflows.some(w => /songwriter/i.test(w.name)));
    await t.file('#importFile', 'r.json', 'application/json', JSON.stringify({ workflow: { name: 'Songwriter' }, revision: { name: 'pop 3 TAKES', backend: 'ace', vals: {} } }));
    await t.ok('an import does not take a factory revision\'s name', () => S.revs.Songwriter.some(r => r.name === 'pop 3 TAKES (2)'));
    await t.file('#importFile', 'r.json', 'application/json', JSON.stringify({ workflow: { name: '__proto__', tier: 'quick', mods: ['tempo'], pedals: [] }, revision: { name: 'odd', backend: 'ace', vals: {} } }));
    await t.ok('any workflow name works, even __proto__', () => S.workflows.some(w => w.name === '__proto__') && Object.hasOwn(S.revs, '__proto__') && /Imported __proto__ › odd/.test(document.querySelector('#toast .toast-msg').textContent));
  } },
  /* F-03: after a row action focus is on that row (or the row that replaced it), never on the page */
  { id: 'row-actions-keep-focus', run: async t => {
    const at = () => { const a = document.activeElement; return a === document.body ? 'BODY' : a.matches('.pre.wfrow') ? 'wf:' + a.dataset.preset : a.matches('.pre.rev') ? 'rev:' + a.dataset.rn : a.matches('[data-rowmore]') ? 'more:' + a.closest('.pre').dataset.rn : a.tagName + '.' + a.className; };
    await t.hover('.pre.wfrow[data-preset="Songwriter"]'); await t.click('.pre.wfrow[data-preset="Songwriter"] b');
    await t.eq('a click on a workflow row (its tooltip held) leaves focus on it', at, 'wf:Songwriter');
    await t.focus('.pre.wfrow[data-preset="Lo-fi tape"]'); await t.press('Enter');
    await t.eq('Enter on a row too', at, 'wf:Lo-fi tape');
    await t.do('remove a module', () => removeMod('tempo'));
    await t.click('.pre.wfrow.on [data-revert]');
    await t.eq('the workflow\'s Revert leaves focus on its row', at, 'wf:Lo-fi tape');
    await t.click('.pre.rev[data-rn="Tape at 82 BPM"] b');
    await t.do('change a setting', () => setVal('samp.steps', 22));
    await t.click('.pre.rev.on [data-rrevert]');
    await t.eq('the revision\'s Revert leaves focus on its row', at, 'rev:Tape at 82 BPM');
    await t.do('a user revision', () => { S.revs['Lo-fi tape'] = [{ name: 'mine', backend: 'ace', vals: {}, slots: [] }]; renderPresets(); });
    await t.click('.pre.rev[data-rn="mine"] [data-rowmore]'); await t.click('#pop [data-rowact="rexp"]');
    await t.eq('Export from the row menu leaves focus on the row\'s ⋯', at, 'more:mine');
  } },
  { id: 'import', run: async t => {
    const good = JSON.stringify({ format: 'hotstep-revision', version: 1, workflow: { name: 'Imported wf', tier: 'lab', mods: ['sampler', 'nope'], pedals: ['norm', 'zz'] }, revision: { name: 'From file', backend: 'yue', vals: { 'samp.steps': 40, 'samp.cfg': 'x', 'bogus.key': 1 }, slots: [{ file: 'a', type: 'LoRA' }, 7] } });
    await t.file('#importFile', 'r.json', 'application/json', good);
    await t.ok('a revision file imports into its workflow, cleaned', () => { const w = S.workflows.find(x => x.name === 'Imported wf'), r = S.revs['Imported wf']?.[0]; return !!w && w.mods.join() === 'sampler' && w.pedals.join() === 'norm' && r?.name === 'From file' && JSON.stringify(r.vals) === '{"samp.steps":40}' && r.slots.length === 1; });
    await t.ok('the toast counts what was left out', () => /3 values didn't fit their controls/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.file('#importFile', 'r.json', 'application/json', good);
    await t.ok('the same again replaces it, with Undo', () => /Replaced Imported wf › From file/.test(document.querySelector('#toast .toast-msg').textContent) && document.querySelector('#toast .toast-act').textContent === 'Undo' && S.revs['Imported wf'].length === 1);
    await t.file('#importFile', 'bad.json', 'application/json', '{not json');
    await t.ok('broken JSON is refused', () => /isn't valid JSON/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.file('#importFile', 'other.json', 'application/json', '{"a":1}');
    await t.ok('JSON that is not a revision is refused', () => /isn't a HOT-Step revision/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.click('#importBtn', 300);
  } },
  { id: 'legacy-storage', storage: { 'hs3-layouts': JSON.stringify([{ name: 'Old layout', tier: 'studio', mods: ['tempo'], pedals: [] }]), 'hs3-presets': JSON.stringify([{ name: 'Old preset', tier: 'lab', mods: ['seed'], vals: { 'samp.steps': 9 }, backend: 'ace' }]), 'hs3-profiles': JSON.stringify([{ name: 'Old layout', backend: 'mm3', vals: { 'samp.steps': 12 } }, { name: 'Stray', backend: 'ace', vals: { 'takes.n': 2 } }]) }, run: async t => {
    await t.eq('older layouts and presets become workflows', () => S.workflows.map(w => w.name), ['Old layout', 'Old preset']);
    await t.eq('older profiles become revisions under their workflow or Start fresh', () => [Object.keys(S.revs).sort(), S.revs['Start fresh']?.[0]?.name], [['Old layout', 'Old preset', 'Start fresh'], 'Stray']);
  } },
  { id: 'corrupt-storage', storage: { 'hs3-wf': '{"workflows": 5}', 'hs3-desk': '{"v":1,"desk":"nope"}', 'hs3-drafts': '"x"' }, run: async t => {
    await t.ok('an unreadable desk starts fresh and says so', () => /couldn't be read/.test(document.querySelector('#toast .toast-msg')?.textContent || '') && S.tier === 'quick');
    await t.eq('nothing else breaks', () => [S.workflows.length, S.drafts.length], [0, 0]);
  } },
];
