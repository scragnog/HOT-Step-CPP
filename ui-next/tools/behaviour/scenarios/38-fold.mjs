// Quick's fold (A-28, Q-05; the concept's renderDesk and flash, 30-desk.js): past six modules, Quick shows six and a
// button names the rest; it opens and folds again by a click, is no undo step, opens by itself for a module that arrives
// folded away, never hides a module holding a set value, is kept over a reload, and shows only at Quick. No engine has
// enough modules Quick shows for the fold to take three, so each scenario puts the test build's two stand-ins on the desk
// as its scene (defineModule; testing/standins.ts); everything after that is real input.

const W = { w: 1600, h: 900 };
const EIGHT = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes'];
const scene = mods => async t => {
  await t.do('stand-ins on a Quick desk', mods => {
    ['tx1', 'tx2'].forEach((id, i) => { defineModule(id, { at: 300 + i, name: 'Stand-in ' + (i + 1), cat: 'Engine', w: 1, keys: [] }); defineFace(id, () => '<p>stand-in</p>'); });
    S.tier = 'quick'; S.backend = 'ace'; S.mods = mods; S.pedals = ['lrc']; render();
  }, mods, 400);
  await t.js(() => { window.__f = {
    tiles: () => [...document.querySelectorAll('#bento > .tile[data-m]')].map(e => e.dataset.m),
    button: () => { const b = document.querySelector('#bento > [data-qall]'); return b ? [b.textContent, b.getAttribute('aria-expanded')] : null; },
  }; });
};
const TEN = [...EIGHT, 'tx1', 'tx2', 'post'];
const B = '#bento > [data-qall]';

export default [
  { id: 'the-button-shows-and-folds-again', ...W, run: async t => {
    await scene(TEN)(t);
    await t.eq('Quick shows six and the Pedalboard; the button names the rest', () => [__f.tiles(), __f.button()], [['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'post'], ['Show 4 more: Seed, Takes, Stand-in 1 and Stand-in 2', 'false']]);
    await t.ok('the button stands after the tiles, the width of the desk', () => { const b = document.querySelector('#bento > [data-qall]'), g = document.querySelector('#bento').getBoundingClientRect(), r = b.getBoundingClientRect(); return [...document.querySelectorAll('#bento > .tile')].every(x => x.getBoundingClientRect().bottom <= r.top + 1) && Math.abs(r.width - g.width) < 2; });
    const steps = await t.js(() => UNDO.length);
    await t.click(B);
    await t.eq('a click shows every module, and the button folds them again', () => [__f.tiles().length, __f.button()], [11, ['Show fewer modules', 'true']]);
    await t.eq('opening the fold is no step', () => UNDO.length, steps);
    await t.click(B);
    await t.eq('another click folds them', () => [__f.tiles().length, __f.button()], [7, ['Show 4 more: Seed, Takes, Stand-in 1 and Stand-in 2', 'false']]);
    await t.focus(B); await t.press('Enter');
    await t.eq('Enter on the button works it as a click does', () => [__f.tiles().length, __f.button()[1]], [11, 'true']);
  } },
  { id: 'a-module-that-arrives-folded-opens-it', ...W, run: async t => {
    await scene(['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'tx1', 'tx2', 'post'])(t);
    await t.eq('nine tiles, two of which would fold: no fold', () => [__f.tiles().length, __f.button()], [9, null]);
    await t.click('#browser .bcard[data-additem="seed"]');
    await t.ok('Seed arrives where the fold would hide it: the fold opens, so it is in sight', () => S.qAll === true && __f.tiles().includes('seed') && __f.button()?.[0] === 'Show fewer modules');
    await t.press('Control+z');
    await t.eq('Undo takes Seed away, and the fold with it', () => [__f.tiles().includes('seed'), __f.button()], [false, null]);
    await t.click('#browser .bcard[data-additem="takes"]');
    await t.click(B);
    await t.eq('folded again, it names what it hides', () => __f.button(), ['Show 3 more: Stand-in 1, Stand-in 2 and Takes', 'false']);
    await t.click('#browser .bcard[data-additem="takes"]', 100);
    await t.ok('adding what is already there, folded, opens the fold and rings it (Q-05)', () => S.qAll === true && document.querySelector('#bento > .tile[data-m="takes"]')?.classList.contains('flash'));
  } },
  { id: 'a-set-value-keeps-its-module-in-sight', ...W, run: async t => {
    await scene(TEN)(t);
    await t.click(B);
    await t.click('.tile[data-m="takes"] [data-take="3"]');
    await t.eq('three takes, set on the Takes tile', () => get('takes.n'), 3);
    await t.click(B);
    await t.eq('folded, Takes stays in sight, and the button names only what it hides', () => [__f.tiles().includes('takes'), __f.button()], [true, ['Show 3 more: Seed, Stand-in 1 and Stand-in 2', 'false']]);
    await t.press('Control+z');
    await t.eq('Undo puts Takes back on its default, and the fold takes it again', () => [__f.tiles().includes('takes'), __f.button()], [false, ['Show 4 more: Seed, Takes, Stand-in 1 and Stand-in 2', 'false']]);
  } },
  { id: 'the-fold-is-kept-over-a-reload', ...W, run: async t => {
    await scene(TEN)(t);
    await t.click(B, 700);
    await t.reload();
    await t.eq('open, it is open again after a reload', () => [S.qAll, document.querySelectorAll('#bento > .tile[data-m]').length], [true, 11]);
    await t.click('#bento > [data-qall]', 700);
    await t.reload();
    await t.eq('closed, it is closed again', () => [S.qAll, document.querySelectorAll('#bento > .tile[data-m]').length], [false, 7]);
  } },
  { id: 'only-quick-folds', ...W, run: async t => {
    await scene(TEN)(t);
    await t.press('Alt+2');
    await t.eq('Studio shows every module and the lyric sheet, with no button', () => [__f.tiles().length, __f.button()], [12, null]);
    await t.press('Alt+1');
    await t.eq('back at Quick, the fold is as it was', () => [__f.tiles().length, __f.button()?.[1]], [7, 'false']);
  } },
  { id: 'song-info-is-kept-over-a-reload', ...W, run: async t => {
    await t.click('#songInfoBtn', 700);
    await t.reload();
    await t.eq('Song info stands open after a reload, as it was (the concept\'s S.songInfoOpen)', () => [document.querySelector('#songInfoBtn').getAttribute('aria-expanded'), document.querySelector('#songInfo').hidden], ['true', false]);
    await t.click('#songInfoBtn', 700);
    await t.reload();
    await t.eq('and closed, closed', () => [document.querySelector('#songInfoBtn').getAttribute('aria-expanded'), document.querySelector('#songInfo').hidden], ['false', true]);
  } },
];
