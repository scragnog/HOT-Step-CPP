// The desk: tiers, engines, adding, removing and moving modules, undo, parked and hidden modules.
const studio = (mods = ['tempo', 'seed', 'takes', 'sampler'], extra = {}) => t => t.do('studio desk', ([mods, extra]) => { S.tier = 'studio'; S.mods = mods; S.pedals = []; Object.assign(S, extra); render(); }, [mods, extra]);
export default [
  { id: 'tiers', run: async t => {
    for (const tier of ['studio', 'lab', 'quick']) {
      await t.click(`#tiers [data-tier="${tier}"]`);
      await t.eq(`tier button sets ${tier}`, () => [S.tier, document.querySelector('#centre').className, document.querySelector('#tiers [aria-checked="true"]').dataset.tier], [tier, 'centre t-' + tier, tier]);
    }
    await t.eq('Quick shows the ask heading and the lyric card', () => [!document.querySelector('#ask').hidden, !!document.querySelector('#quickLyrics .lyr-card')], [true, true]);
    await t.click('#tiers [data-tier="studio"]');
    await t.ok('Studio puts the lyrics on the desk as a tile', () => !!document.querySelector('#bento .tile[data-m="lyrics"]') && document.querySelector('#ask').hidden);
  } },
  { id: 'engines-panel', run: async t => {
    await t.click('#engBtn');
    await t.ok('engine panel opens with five cards', () => !document.querySelector('#engPanel').hidden && document.querySelectorAll('#engPanel .ep-card').length === 5);
    for (const bk of ['mm3', 'yue', 'sa3', 'hm', 'ace']) {
      await t.click(`#engPanel [data-backend="${bk}"]`);
      await t.eq(`card selects ${bk}`, bk => [S.backend, document.querySelector('#engName').textContent === BK[bk].name, !document.querySelector('#engPanel').hidden], [bk, true, true], bk);
    }
    await t.focus('#engPanel [data-backend="ace"]');
    /* F-16: the arrows move focus only; Enter chooses (a switch rebuilds the desk, so it does not follow focus) */
    await t.js(() => { window.__u = UNDO.length; });
    await t.press('ArrowRight');
    await t.ok('arrow keys move between engines without switching', () => S.backend === 'ace' && document.activeElement.dataset.backend === 'mm3' && UNDO.length === window.__u);
    await t.press('ArrowLeft'); await t.press('ArrowLeft');
    await t.ok('and wrap around', () => S.backend === 'ace' && document.activeElement.dataset.backend === 'hm');
    await t.press('Enter');
    await t.eq('Enter chooses the one in focus', () => S.backend, 'hm');
    await t.press('Escape');
    await t.ok('Escape closes the panel and focus returns to its button', () => document.querySelector('#engPanel').hidden && document.activeElement.id === 'engBtn');
    await t.click('#engBtn'); await t.click('.top .brand');
    await t.ok('a click outside closes it', () => document.querySelector('#engPanel').hidden);
  } },
  { id: 'engine-switch-effects', run: async t => {
    await studio(['tempo', 'takes'])(t);
    await t.do('nine takes', () => { S.vals['takes.n'] = 9; render(); });
    await t.click('#engBtn'); await t.click('#engPanel [data-backend="sa3"]');
    await t.eq('takes read the engine limit', () => [S.vals['takes.n'], S.eng.hold['takes.n'].v], [1, 9]);
    await t.ok('and a toast says the 9 come back (B-11; worded so on purpose, S-12)', () => /Takes capped at 1; your 9 takes wait on ACE-Step 1\.5/.test(document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.press('Escape');
    await t.do('repaint source', () => { S.backend = 'ace'; S.source = { song: 1 }; S.mods = ['repaint', 'tempo']; render(); });
    await t.click('#engBtn'); await t.click('#engPanel [data-backend="mm3"]');
    await t.eq('Repaint leaves on another engine', () => [S.mods.includes('repaint'), S.source], [false, null]);
    await t.ok('with a toast', () => /Repaint was removed: it needs ACE-Step/.test(document.querySelector('#toast .toast-msg')?.textContent || ''));
  } },
  { id: 'add-remove-undo', run: async t => {
    await studio()(t);
    await t.click('#browser .bcard[data-additem="avoid"]');
    await t.ok('a card click adds the module at the end', () => S.mods[S.mods.length - 1] === 'avoid' && !!document.querySelector('.tile[data-m="avoid"]'));
    await t.click('#browser .bcard[data-additem="avoid"]');
    await t.eq('adding it again does not duplicate it', () => S.mods.filter(m => m === 'avoid').length, 1);
    await t.click('.tile[data-m="seed"] [data-remove="seed"]');
    await t.ok('× removes the tile', () => !S.mods.includes('seed') && !document.querySelector('.tile[data-m="seed"]'));
    await t.ok('the toast offers Undo', () => document.querySelector('#toast .toast-act')?.textContent === 'Undo');
    await t.ok('focus moves to the next tile', () => document.activeElement?.matches('.tile[data-m="takes"] .grip'));
    await t.click('#browser .bcard[data-additem="seed"]');
    await t.eq('re-adding puts it back where it was', () => S.mods.join(), 'tempo,seed,takes,sampler,avoid');
    await t.click('.tile[data-m="sampler"] [data-remove="sampler"]');
    await t.click('#toast .toast-act');
    await t.ok('the toast Undo brings it back', () => S.mods.includes('sampler') && /Undone: Remove Sampler/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.click('.tile[data-m="tempo"] [data-remove="tempo"]');
    await t.ok('the Undo button names the last action', () => !document.querySelector('#undoBtn').hidden && document.querySelector('#undoBtn span').textContent === 'Remove Tempo, key & length');
    await t.click('#undoBtn');
    await t.ok('the Undo button takes it back', () => S.mods.includes('tempo'));
    await t.click('.tile[data-m="avoid"] [data-remove="avoid"]');
    await t.click('.top .brand'); await t.press('Control+z');
    await t.ok('Ctrl+Z outside a field undoes', () => S.mods.includes('avoid'));
    await t.click('.tile[data-m="avoid"] [data-remove="avoid"]');
    await t.focus('#prompt'); await t.type('warm'); await t.press('Control+z');
    await t.ok('Ctrl+Z inside a text field is left to the field while it has typing to take back', () => !S.mods.includes('avoid') && document.querySelector('#prompt').value === '');
    /* changed on purpose (L-05, E-18): a field with nothing left of its own hands the keys on to the page's Undo, so
       pressing Ctrl+Z again always does something */
    await t.press('Control+z');
    await t.ok('then, with nothing left in the field, Ctrl+Z goes on to the page: the removal is undone', () => S.mods.includes('avoid'));
  } },
  { id: 'add-by-keyboard', run: async t => {
    await studio()(t);
    await t.focus('#browser .bcard[data-additem="timbre"]'); await t.press('Enter');
    await t.ok('Enter on a card adds it', () => S.mods.includes('timbre'));
    await t.focus('#browser .bcard[data-additem="avoid"]'); await t.press(' ');
    await t.ok('Space on a card adds it', () => S.mods.includes('avoid'));
    await t.focus('.tile[data-m="avoid"] .grip'); await t.press('ArrowLeft');
    await t.ok('arrow keys on the grip move the tile, focus stays on it', () => S.mods.indexOf('avoid') === S.mods.indexOf('timbre') - 1 && document.activeElement?.dataset.move === 'avoid');
    await t.ok('the move is spoken', () => /Moved Avoid left/.test(document.querySelector('#say').textContent));
    await t.press('ArrowDown'); await t.press('ArrowRight');
    await t.eq('and back to the end', () => S.mods[S.mods.length - 1], 'avoid');
    await t.press('ArrowRight');
    await t.eq('past the end nothing moves', () => S.mods[S.mods.length - 1], 'avoid');
  } },
  { id: 'lab-modules', run: async t => {
    await studio(['sampler'])(t);
    await t.click('#browser .bcard[data-additem="guidance"]');
    await t.ok('a Lab module in Studio offers Lab in a popover', () => !document.querySelector('#pop').hidden && !!document.querySelector('#pop [data-labadd="guidance"]'));
    await t.click('#pop [data-closepop]');
    await t.ok('Not now closes it', () => document.querySelector('#pop').hidden && !S.mods.includes('guidance'));
    await t.click('#browser .bcard[data-additem="guidance"]');
    await t.click('#pop [data-labadd="guidance"]');
    await t.eq('Switch to Lab and add it', () => [S.tier, S.mods.includes('guidance')], ['lab', true]);
    await t.click('#tiers [data-tier="studio"]');
    await t.ok('a hidden Lab module is announced above the desk', () => !document.querySelector('#hiddenNote').hidden && /Hidden at this tier, still applied: \w/.test(document.querySelector('#hiddenNote').textContent));
    await t.click('#hiddenNote [data-tier="lab"]');
    await t.eq('Open in Lab', () => S.tier, 'lab');
    await t.do('add by script while in Studio', () => { S.tier = 'studio'; render(); addItem('perf'); });
    await t.ok('addItem without an anchor toasts instead', () => /is a Lab module. Switch to Lab/.test(document.querySelector('#toast .toast-msg').textContent));
  } },
  { id: 'quick-more-parked-empty', run: async t => {
    /* A-28: two more than six are simply shown; three or more fold behind "Show n more" */
    await t.do('quick, eight modules', () => { S.tier = 'quick'; S.mods = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes']; render(); });
    await t.eq('eight show all eight, with no fold', () => [document.querySelectorAll('#bento > .tile[data-m]').length, !!document.querySelector('#bento [data-qall]')], [8, false]);
    /* Q-05: the Pedalboard is never folded, so eight modules and the board show all nine */
    await t.do('quick, nine modules', () => { S.mods = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes', 'post']; S.pedals = ['lrc']; render(); });
    await t.eq('eight modules and the Pedalboard all show (Q-05)', () => [document.querySelectorAll('#bento > .tile[data-m]').length, !!document.querySelector('#bento [data-qall]')], [9, false]);
    /* no engine has nine modules Quick shows, so two stand-ins are registered to exercise the fold */
    await t.do('quick, ten modules and the board', () => { ['tx1', 'tx2'].forEach((id, i) => { defineModule(id, { at: 300 + i, name: 'Stand-in ' + (i + 1), cat: 'Engine', w: 1, keys: [] }); defineFace(id, () => '<p>stand-in</p>'); }); S.mods = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes', 'tx1', 'tx2', 'post']; render(); });
    await t.eq('Quick shows six and the Pedalboard; the button names the rest (Q-05)', () => [document.querySelectorAll('#bento > .tile[data-m]').length, !!document.querySelector('#bento > .tile[data-m="post"]'), document.querySelector('#bento [data-qall]')?.textContent], [7, true, 'Show 4 more: Seed, Takes, Stand-in 1 and Stand-in 2']);
    await t.do('change a folded module', () => setVal('takes.n', 3));
    await t.do('render', () => render());
    await t.ok('a module holding a changed value is never folded (Q-05)', () => !!document.querySelector('#bento > .tile[data-m="takes"]'));
    await t.do('click Seed in the Modules panel while it is folded', () => document.querySelector('#browser .bcard[data-additem="seed"]').click(), null, 600);
    await t.ok('a ring aimed at a folded tile opens the fold first (Q-05)', () => S.qAll === true && !!document.querySelector('#bento > .tile[data-m="seed"]'));
    await t.click('#bento [data-qall]');
    await t.eq('and fewer again', () => S.qAll, false);
    await t.do('back to the real modules', () => { S.mods = S.mods.filter(m => !m.startsWith('tx')); ['tx1', 'tx2'].forEach(id => { delete MODS[id]; delete FACES[id]; }); catalogued(); delete S.vals['takes.n']; render(); });
    await t.do('no pedals', () => { S.pedals = []; render(); });
    await t.do('yue studio with ACE-only modules', () => { S.tier = 'studio'; S.backend = 'yue'; S.mods = ['avoid', 'timbre', 'seed']; render(); });
    await t.ok('modules the engine does not use are parked', () => document.querySelectorAll('.parked .pchip[data-pm]').length === 2 && !document.querySelector('#bento > .tile[data-m="avoid"]'));
    await t.click('.parked .pchip[data-pm="avoid"] [data-remove="avoid"]');
    await t.ok('a parked chip can be removed', () => !S.mods.includes('avoid'));
    await t.do('empty studio', () => { S.backend = 'ace'; S.mods = []; render(); });
    await t.ok('an empty desk offers starters', () => document.querySelectorAll('.desk-empty [data-additem]').length === 3);
    await t.click('.desk-empty [data-additem="tempo"]');
    await t.ok('a starter adds its module', () => S.mods.includes('tempo'));
    await t.do('empty again', () => { S.mods = []; render(); });
    await t.click('.desk-empty [data-openpicker]');
    await t.ok('Open the picker from the empty desk', () => !document.querySelector('#pickerWrap').hidden);
  } },
  { id: 'show-changes', run: async t => {
    await studio(['tempo', 'sampler'])(t);
    await t.ok('no changes at first', () => /everything on defaults/.test(document.querySelector('#meter').textContent));
    await t.do('two changes', () => { setVal('samp.cfg', 12); setVal('tempo.sig', '3/4'); });
    await t.ok('the meter counts them', () => /2 changes/.test(document.querySelector('#meter').textContent));
    await t.click('#meter [data-showchanges]');
    await t.ok('pressing it rings the changed controls', () => document.querySelectorAll('#bento .ping').length >= 2);
    await t.do('changes only in hidden', () => { S.mods = ['perf']; S.vals = { 'perf.cache': 0.3 }; render(); });
    await t.click('#meter [data-showchanges]');
    await t.ok('a change in a hidden module says so', () => / in [A-Z].*, which \w+ hides/.test(document.querySelector('#toast .toast-msg').textContent));
  } },
  { id: 'every-engine-every-tier', run: async t => {
    for (const bk of ['ace', 'mm3', 'yue', 'sa3', 'hm']) for (const tier of ['quick', 'studio', 'lab']) {
      await t.do(`${bk} ${tier} everything`, ([bk, tier]) => { S.backend = bk; S.tier = tier; S.qAll = true; S.mods = Object.keys(MODS).filter(m => !MODS[m].source && m !== 'post'); S.pedals = [...PEDALS, ...EXTRAS].map(p => p.id); S.mods.push('post'); render(); }, [bk, tier]);
      await t.ok(`${bk} ${tier}: every live module has a tile with a body`, () => { const bad = []; S.mods.filter(m => shown(m) && modOk(m)).forEach(m => { const t = document.querySelector(`#bento > .tile[data-m="${m}"]`); if (!t || t.getBoundingClientRect().height < 30) bad.push(m); }); return bad.length ? bad : true; });
      await t.ok(`${bk} ${tier}: no two tiles overlap`, () => { const ts = [...document.querySelectorAll('#bento > .tile')].map(t => [t.dataset.m, t.getBoundingClientRect()]), bad = [];
        for (let i = 0; i < ts.length; i++) for (let j = i + 1; j < ts.length; j++) { const [m1, a] = ts[i], [m2, b] = ts[j]; if (a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1) bad.push(m1 + '/' + m2); } return bad.length ? bad : true; });
    }
  } },
];
