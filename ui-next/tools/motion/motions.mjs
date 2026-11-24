// The motions tools/motion/frames.mjs compares with their approved samples, and tools/motion/cost.mjs measures: each
// sets a scene through the test build's compatibility layer (S, render, addItem), sets the motion off as a user does,
// and names the boxes that move. A key is a selector, `|attr` naming each match by an attribute; `vt:<name>` is a View
// Transition part.
const ACE = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes'];
const desk = (tier, mods, pedals = []) => new Function(`S.tier = ${JSON.stringify(tier)}; S.mods = ${JSON.stringify(mods)}; S.pedals = ${JSON.stringify(pedals)}; S.pedals.forEach(id => { if (PED[id]?.on) S.vals[PED[id].on] = true; }); S.qAll = true; render(); scrollTo(0, 0);`);
const TILES = '#bento > .tile[data-m]|data-m', EXIT = '.exit-layer > .tile|data-m', PEDALS = '[data-pedals] > .pedal[data-pid]|data-pid', PEXIT = '.pedal-exit > .pedal|data-pid';
const vt = mods => [...mods.flatMap(m => [`vt:vt-${m}`, `vt:vt-h-${m}`]), 'vt:vt-prompt', 'vt:vt-lyrics', 'vt:vt-h-lyrics', 'vt:vt-ask', 'vt:vt-hnote', 'vt:vt-gobar', 'vt:vt-bar', 'vt:vt-deskend'];
/* a Quick desk long enough to fold (the test build's two stand-ins, as 38-fold.mjs makes them) */
const fold = open => new Function(`['tx1', 'tx2'].forEach((id, i) => { defineModule(id, { at: 300 + i, name: 'Stand-in ' + (i + 1), cat: 'Engine', w: 1, keys: [] }); defineFace(id, () => '<span>stand-in</span>'); }); S.tier = 'quick'; S.backend = 'ace'; S.mods = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes', 'tx1', 'tx2', 'post']; S.pedals = ['lrc']; S.qAll = ${open}; render(); scrollTo(0, 0);`);
const click = sel => p => p.click(sel);
const key = (sel, k) => async p => { await p.focus(sel); await p.keyboard.press(k); };
const few = ['tempo', 'avoid', 'takes', 'seed'], LINES = 'a\nb\nc\nd\ne\nf';

export const MOTIONS = [
  { id: 'tier-quick-to-studio', what: 'the tier change, Quick to Studio, four modules (the reviewers\' n1)', scene: desk('quick', few), act: click('#tiers [data-tier="studio"]'), keys: vt(few) },
  { id: 'tier-studio-to-quick', what: 'Studio to Quick, four modules', scene: desk('studio', few), act: click('#tiers [data-tier="quick"]'), keys: vt(few) },
  { id: 'tier-studio-to-lab', what: 'Studio to Lab, every ACE-Step module and the board, 1920', w: 1920, h: 1080, scene: desk('studio', [...ACE, 'post'], ['norm', 'ss']), act: click('#tiers [data-tier="lab"]'), keys: vt([...ACE, 'post', 'guidance', 'latent', 'perf']) },
  { id: 'tier-lab-to-quick', what: 'Lab to Quick, a Lab module leaving', scene: desk('lab', ['tempo', 'avoid', 'guidance', 'seed']), act: click('#tiers [data-tier="quick"]'), keys: vt(['tempo', 'avoid', 'guidance', 'seed']) },
  { id: 'tier-dark', what: 'Quick to Studio in the dark theme', scheme: 'dark', scene: desk('quick', few), act: click('#tiers [data-tier="studio"]'), keys: vt(few) },
  { id: 'tile-arrives', what: 'a tile added scales in; what it pushes slides', w: 1920, h: 1080, scene: desk('studio', ['tempo', 'avoid', 'takes']), act: p => p.evaluate(() => addItem('seed')), keys: [TILES], opacity: true },
  { id: 'tile-leaves', what: 'a tile removed fades where it stood; its neighbours close up', w: 1920, h: 1080, scene: desk('studio', ['tempo', 'avoid', 'sampler', 'takes', 'seed']), act: click('.tile[data-m="avoid"] .tile-head .x'), keys: [TILES, EXIT], opacity: true },
  { id: 'undo-brings-it-back', what: 'Undo brings the removed tile back', w: 1920, h: 1080, scene: desk('studio', ['tempo', 'avoid', 'sampler', 'takes', 'seed']), before: async p => { await p.click('.tile[data-m="avoid"] .tile-head .x'); await p.waitForTimeout(600); }, act: click('#undoBtn'), keys: [TILES], opacity: true },
  { id: 'tile-moves-by-key', what: 'a tile moved by the keyboard slides', w: 1920, h: 1080, scene: desk('studio', ['tempo', 'avoid', 'sampler', 'takes', 'seed']), act: key('[data-move="takes"]', 'ArrowLeft'), keys: [TILES] },
  { id: 'pedal-joins', what: 'a pedal joins the board, the others slide, the cables draw out', w: 1920, h: 1080, scene: desk('lab', ['tempo', 'post'], ['norm', 'lift']), act: p => p.evaluate(() => addItem('pd:ss')), keys: [PEDALS, TILES], opacity: true },
  { id: 'pedal-leaves', what: 'a pedal leaves the board', w: 1920, h: 1080, scene: desk('lab', ['tempo', 'post'], ['ss', 'lift', 'norm']), act: click('[data-pedalx="lift"]'), keys: [PEDALS, PEXIT], opacity: true },
  { id: 'reduced-tile-arrives', what: 'under reduced motion a tile arrives by fading only', reduced: true, w: 1920, h: 1080, scene: desk('studio', ['tempo', 'avoid', 'takes']), act: p => p.evaluate(() => addItem('seed')), keys: [TILES], opacity: true },
  { id: 'reduced-tier-change', what: 'under reduced motion the tier changes with nothing moving', reduced: true, scene: desk('quick', few), act: click('#tiers [data-tier="studio"]'), keys: [TILES], opacity: true, until: 120 },
  { id: 'prompt-shrinks-on-undo', what: 'the prompt box shrinks by lines when Undo takes its text back; what is under it does not jump', w: 1920, h: 1080, scene: desk('quick', few), before: async p => { await p.focus('#prompt'); await p.keyboard.insertText(LINES); await p.evaluate(() => { document.activeElement.blur(); scrollTo(0, 0); }); await p.waitForTimeout(400); }, act: click('#undoBtn'), keys: ['.pc-foot'] },
  { id: 'prompt-grows', what: "the prompt box grows by lines, easing (P9-2); what is under it does not jump", w: 1920, h: 1080, scene: desk('quick', few), before: p => p.focus('#prompt'), act: p => p.keyboard.insertText('a\nb\nc\nd\ne\nf'), keys: ['.pc-foot', '#bento > .tile[data-m="tempo"]'] },
  { id: 'suggestion-opens', what: 'a suggestion opens under the box; what is under it does not jump', w: 1920, h: 1080, scene: desk('quick', few), before: p => p.focus('#prompt'), act: p => p.keyboard.insertText('lofi beat at 92 bpm'), keys: ['.pc-foot', '#bento > .tile[data-m="tempo"]', '#suggest'], opacity: true },
  { id: 'song-info-opens', what: 'Song info opens under the card\'s tools', w: 1920, h: 1080, scene: desk('quick', few), act: click('#songInfoBtn'), keys: ['#songInfo', '#bento > .tile[data-m="tempo"]'], opacity: true },
  { id: 'generate-starts', what: 'Generate pressed: Cancel slides out from under it, the cards slide along for the new ones', w: 1920, h: 1080, scene: desk('quick', few), before: async p => { await p.focus('#prompt'); await p.keyboard.insertText('warm folk song'); await p.evaluate(() => { document.activeElement.blur(); scrollTo(0, 0); }); }, act: click('#go'), keys: ['#cancelGo', '#go', '#meter'], opacity: true },
  { id: 'fold-opens', what: 'the fold of Quick opened: the tiles it shows arrive, what they push slides', w: 1600, h: 1400, scene: fold(false), act: p => p.evaluate(() => document.querySelector('#bento > [data-qall]').click()), keys: [TILES], opacity: true },
  { id: 'fold-folds', what: 'the fold of Quick folded again: the tiles it hides fade where they stood', w: 1600, h: 1400, scene: fold(true), act: p => p.evaluate(() => document.querySelector('#bento > [data-qall]').click()), keys: [TILES, EXIT], opacity: true },
];
