// Drag and drop (cards, tiles, picker items, pedals, files) and the side panels as columns, overlays and drawers.
const studio = (mods = ['tempo', 'seed', 'takes', 'sampler']) => t => t.do('studio desk', mods => { S.tier = 'studio'; S.mods = mods; S.pedals = []; render(); }, mods);
export default [
  { id: 'drag-card-onto-desk', w: 1600, run: async t => {
    await studio()(t);
    await t.drag('#browser .bcard[data-drag="avoid"]', '.tile[data-m="seed"]', { at: [0.2, 0.3] });
    /* a card stands where it was dropped: at the place nearest the pointer (Q-01; the left/right-half rule is gone) */
    await t.ok('a card dropped on a tile is added where the pointer was', () => { const [, , x, y] = [0, 0, ...(window.__dpt ? [window.__dpt[1].clientX, window.__dpt[1].clientY] : [])]; const r = document.querySelector('.tile[data-m="avoid"]')?.getBoundingClientRect(); if (!r) return 'not added'; const d = Math.hypot(Math.max(r.left - x, 0, x - r.right), Math.max(r.top - y, 0, y - r.bottom)); return d < 200 || ('stands ' + Math.round(d) + 'px from the pointer'); });
    await t.ok('the drag classes are cleared', () => !document.body.classList.contains('dnd') && !document.querySelector('.ghost'));
    await t.drag('#browser .bcard[data-drag="timbre"]', '.tile[data-m="sampler"]', { at: [0.8, 0.3] });
    await t.ok('and another, over another tile', () => { const [, , x, y] = [0, 0, ...(window.__dpt ? [window.__dpt[1].clientX, window.__dpt[1].clientY] : [])]; const r = document.querySelector('.tile[data-m="timbre"]')?.getBoundingClientRect(); if (!r) return 'not added'; const d = Math.hypot(Math.max(r.left - x, 0, x - r.right), Math.max(r.top - y, 0, y - r.bottom)); return d < 200 || ('stands ' + Math.round(d) + 'px from the pointer'); });
    await t.drag('#browser .bcard[data-drag="adapters"]', '.tile[data-m="tempo"]', { drop: false });
    await t.ok('a cancelled drag adds nothing and cleans up', () => !S.mods.includes('adapters') && !document.body.classList.contains('dnd') && !document.querySelector('#bento .ghost'));
  } },
  { id: 'drag-move-tile', w: 1600, run: async t => {
    await studio()(t);
    await t.drag('.tile[data-m="tempo"] [data-drag="move:tempo"]', '.tile[data-m="sampler"]', { at: [0.9, 0.95] });
    /* the drop lands where its preview stood, and the preview stands at the place nearest the pointer (Q-01): below the last
       tile, that is the end */
    /* round 5: the Sampler may fold to one column beside the sheet, so where "the end" is depends on the width; the drop's
       place is checked by 12-desk-layout's drag-lands-on-its-preview */
    await t.ok('a tile dragged by its header moves', () => S.mods.indexOf('tempo') > 0 || S.mods.join());
    const moved = await t.js(() => S.mods.join());
    await t.ok('the layout is marked changed', () => S.wfDirty === true);
    await t.drag('.tile[data-m="takes"] [data-drag="move:takes"]', '.tile[data-m="seed"]', { drop: false });
    await t.eq('a cancelled move puts it back', () => [S.mods.join(), !!document.querySelector('.tile.moving')], [moved, false]);
  } },
  { id: 'drag-from-picker-and-pedal', w: 1600, run: async t => {
    await studio()(t);
    await t.press('F9');
    await t.drag('#pkLeft .fl-it[data-drag="avoid"]', '.tile[data-m="seed"]', { at: [0.2, 0.3] });
    await t.ok('a picker item dropped on the desk is added and the picker closes', () => S.mods.includes('avoid') && document.querySelector('#pickerWrap').hidden);
    await t.drag('#browser .bcard[data-drag="pd:norm"]', '.tile[data-m="seed"]');
    await t.ok('a pedal dropped on the desk brings the Pedalboard', () => S.pedals.includes('norm') && S.mods.includes('post') && S.vals['pp.norm.on'] === true);
    await t.drag('#browser .bcard[data-drag="pd:gain"]', '.tile[data-m="post"]');
    await t.eq('a second pedal joins in chain order', () => S.pedals.join(), 'gain,norm');
    await t.drag('#browser .bcard[data-drag="ex:wh"]', '.tile[data-m="post"]');
    await t.ok('a side unit too', () => S.pedals.includes('wh') && !!document.querySelector('.pedal.xunit[data-xid="wh"]'));
  } },
  { id: 'file-drops', run: async t => {
    await studio(['timbre', 'tempo'])(t);
    await t.dropFile('.tile[data-m="timbre"] label.drop', 'kick-loop.wav', 'audio/wav');
    await t.eq('an audio file dropped on the sleeve loads', () => S.timbreFile, 'kick-loop.wav');
    await t.dropFile('.tile[data-m="timbre"] label.drop', 'notes.txt', 'text/plain');
    await t.ok('a non-audio file is refused', () => S.timbreFile === 'kick-loop.wav' && /isn't an audio file/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.dropFile('.tile[data-m="tempo"]', 'x.wav', 'audio/wav');
    await t.ok('a drop outside a zone says where to drop', () => /Drop the file on one of the drop zones/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.file('.tile[data-m="timbre"] [data-file="timbre"]', 'take-2.flac', 'audio/flac');
    await t.eq('choosing a file works too', () => [S.timbreFile, S.refs[0]], ['take-2.flac', 'take-2.flac']);
    await t.click('.tile[data-m="timbre"] [data-fileclear="timbre"]');
    await t.ok('× clears it, with Undo', () => S.timbreFile === '' && document.querySelector('#toast .toast-act')?.textContent === 'Undo');
    await t.click('#toast .toast-act');
    await t.eq('Undo restores it', () => S.timbreFile, 'take-2.flac');
    /* the tile shows the two latest; the rest are one click further, under "N more…" (round 5 L-21) */
    if (await t.js(() => !document.querySelector('.tile[data-m="timbre"] [data-pref="reference-mix.wav"]'))) { await t.click('.tile[data-m="timbre"] [data-refsmore]'); await t.click('#pop [data-pref="reference-mix.wav"]'); }
    else await t.click('.tile[data-m="timbre"] [data-pref="reference-mix.wav"]');
    await t.eq('a recent reference is one click (or two, when it is not one of the latest two)', () => S.timbreFile, 'reference-mix.wav');
    await t.dropFile('#latBtn', 'song.latent', '');
    await t.ok('a .latent dropped on Import latent loads its metadata', () => S.latent === 'song.latent' && S.vals['tempo.bpm'] === 96);
    await t.click('#latChip [data-latclear]');
    await t.ok('the latent chip clears it', () => S.latent === null && /Back to a fresh render/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.file('#latBtn [data-file="latent"]', 'nope.wav', 'audio/wav');
    await t.ok('a wrong file for the latent is refused', () => /isn't a .latent or .hslat file/.test(document.querySelector('#toast .toast-msg').textContent));
  } },
  { id: 'panels-collapse', w: 1920, run: async t => {
    await t.click('[data-collapse="right"]');
    await t.ok('the Modules panel collapses to its rail, focus on the rail', () => document.querySelector('#rightPanel').classList.contains('collapsed') && S.right === false && document.activeElement.dataset.expand === 'right');
    await t.click('[data-expand="right"]');
    await t.ok('and expands', () => !document.querySelector('#rightPanel').classList.contains('collapsed') && document.activeElement.dataset.collapse === 'right');
    await t.click('[data-collapse="left"]');
    await t.eq('the Workflows panel too', () => [S.left, document.querySelector('#app').classList.contains('no-left')], [false, true]);
    await t.reload();
    await t.eq('the choice is kept', () => S.left, false);
    await t.click('[data-expand="left"]');
  } },
  { id: 'panels-overlay-1280', w: 1280, run: async t => {
    await t.ok('below 1400 the Modules panel starts on its rail', () => document.querySelector('#rightPanel').classList.contains('collapsed'));
    await t.click('[data-expand="right"]');
    await t.ok('and opens as an overlay', () => document.querySelector('#app').classList.contains('r-over'));
    await t.click('.top .brand');
    await t.ok('a press outside closes it', () => !document.querySelector('#app').classList.contains('r-over') && S.rOpen === false);
    /* F-27: Escape closes it too, and focus goes back to its rail button */
    await t.click('[data-expand="right"]'); await t.press('Escape');
    await t.ok('Escape closes it, focus back on its rail button', () => !document.querySelector('#app').classList.contains('r-over') && document.activeElement?.dataset.expand === 'right');
    await t.do('lab', () => { S.tier = 'lab'; render(); });
    await t.ok('in Lab below 1600 the Workflows panel starts on its rail', () => document.querySelector('#leftPanel').classList.contains('collapsed'));
    await t.resize(1700);
    await t.ok('wider, it comes back', () => !document.querySelector('#leftPanel').classList.contains('collapsed'));
  } },
  { id: 'drawers-1000', w: 1000, h: 900, run: async t => {
    await t.click('[data-open="right"]');
    await t.ok('the Modules drawer opens, focus goes in', () => document.querySelector('#rightPanel').classList.contains('open') && !document.querySelector('#drawerScrim').hidden && document.querySelector('[data-open="right"]').getAttribute('aria-expanded') === 'true' && document.querySelector('#rightPanel').contains(document.activeElement));
    await t.click('#drawerScrim', 300, [0.1, 0.5]);
    await t.ok('the scrim closes it', () => !document.querySelector('#rightPanel').classList.contains('open') && document.querySelector('#drawerScrim').hidden);
    await t.click('[data-open="left"]');
    await t.press('Escape');
    await t.ok('Escape closes the drawer and returns focus to its button', () => !document.querySelector('.side.open') && document.activeElement.dataset.open === 'left');
    await t.click('[data-open="left"]');
    await t.click('#leftPanel [data-collapse="left"]');
    await t.ok('in a drawer the collapse button closes the drawer', () => !document.querySelector('.side.open') && S.left !== false);
    await t.click('[data-open="right"]');
    await t.click('#browser .bcard[data-additem="tempo"]');
    await t.ok('adding from the drawer closes it', () => S.mods.includes('tempo') && !document.querySelector('.side.open'));
  } },
];
