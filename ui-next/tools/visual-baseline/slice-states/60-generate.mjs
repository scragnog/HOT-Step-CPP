// Package 6's states (one file of tools/visual-baseline/slice-states/, gathered by slice-states.mjs): the go bar at
// rest, with a queue, with a run that will not fit and with ACE-Step's passes; Latest at rest, with runs rendering,
// waiting and folded, with finished songs, and with a notice the server refused; the player in the go bar; the song
// menu; the toast stack. In both themes, at 1920, 1440 and 1280, and with reduced motion. The desk is not compared:
// the scope `generate` is the go bar, Latest, the song menu, the player and the toasts.
//
//   npm run visual:all -- --only "^generate$"            (and visual:approve, README.md)

const scene = (bk, tier, mods = [], vals = {}, text = 'warm folk song') => async h => {
  await h.js(([bk, tier, mods, vals, text]) => { if (bk !== S.backend) enterEngine(bk); S.tier = tier; S.mods = mods; S.pedals = []; Object.assign(S.vals, vals); const p = document.querySelector('#prompt'); if (p) p.value = text; S.vals['prompt.text'] = text; render(); }, [bk, tier, mods, vals, text]);
  await h.settle(600);
};
/* presses of Generate, each its own press (the concept takes a second press within 350ms as the same) */
const press = (n, after = 600) => async h => { for (let i = 0; i < n; i++) { await h.click('#go'); await h.settle(400); } await h.settle(after); };
const then = (...steps) => async h => { for (const s of steps) await s(h); };
const heightFor = w => (w <= 1000 ? 800 : w <= 1440 ? 900 : 1080);
const st = (id, run, o = {}) => ({ id, w: 1440, h: heightFor(o.w ?? 1440), scheme: 'light', reduced: false, pseudo: false, shot: false, run, ...o });
const takes = n => scene('ace', 'studio', ['takes'], { 'takes.n': n });

export const STATES = [
  /* at rest: the go bar and Latest, in both themes and at the widths */
  st('gen-rest', scene('ace', 'studio'), { pseudo: true, shot: '.latest' }),
  st('gen-rest-dark', scene('ace', 'studio'), { scheme: 'dark' }),
  st('gen-rest-quick', scene('ace', 'quick')),
  ...[1920, 1280].map(w => st(`gen-rest-w${w}`, scene('ace', 'studio'), { w })),
  /* runs rendering: one with three takes, then a second waiting; the sweep across Generate, Cancel all, the badge */
  st('gen-running', then(takes(3), press(1, 1200))),
  st('gen-queue', then(takes(3), press(2, 1200))),
  st('gen-queue-dark', then(takes(3), press(2, 1200)), { scheme: 'dark' }),
  st('gen-queue-reduced', then(takes(3), press(2, 1200)), { reduced: true }),
  /* a long queue, folded */
  st('gen-fold', then(takes(4), press(3, 800)), { w: 1920, h: 1080 }),
  /* finished: the new cards with their unheard marks and take badges */
  st('gen-done', then(takes(2), press(1, 4600))),
  st('gen-done-dark', then(takes(2), press(1, 4600)), { scheme: 'dark' }),
  /* the line: a run that will not fit (MiniMax-Music3) and ACE-Step's passes; hidden and parked */
  st('gen-wontfit', scene('mm3', 'studio', ['takes'], { 'takes.n': 4, 'model.lm': '8B planner · F16' })),
  st('gen-passes', scene('ace', 'studio', ['takes'], { 'takes.n': 8, 'model.lm': '4B · BF16', 'model.dit': 'XL Merge SFT+Turbo λ0.3 · Q8_0' })),
  st('gen-line-hidden-parked', scene('mm3', 'quick', ['sampler', 'latent', 'perf', 'repetition', 'guidance', 'planner'])),
  /* a run the engine refused: the concept refuses four takes at the card; the rebuild does not start a run certain
     not to fit (F-21), so its simulated server is given a smaller card and refuses two */
  st('gen-notice', then(async h => { await h.js(() => { const rb = typeof mockCard === 'function'; if (rb) mockCard(30); if (S.backend !== 'mm3') enterEngine('mm3'); S.tier = 'studio'; S.mods = ['takes']; Object.assign(S.vals, { 'takes.n': rb ? 2 : 4, 'model.lm': '8B planner · F16', 'model.dit': 'Flow DiT · F16', 'model.x': 'Depth decoder · F16' }); const p = document.querySelector('#prompt'); if (p) p.value = 'big song'; S.vals['prompt.text'] = 'big song'; render(); }); await h.settle(600); }, press(1, 600))),
  /* the player in the go bar, playing and paused */
  st('gen-player', then(scene('ace', 'studio'), async h => { await h.click('#latest .song[data-id="2"] [data-play]'); await h.settle(3200); })),
  st('gen-player-dark', then(scene('ace', 'studio'), async h => { await h.click('#latest .song[data-id="2"] [data-play]'); await h.settle(1200); await h.click('#ppause'); await h.settle(400); }), { scheme: 'dark', pseudo: true }),
  st('gen-player-w1280', then(scene('ace', 'studio'), async h => { await h.click('#latest .song[data-id="4"] [data-play]'); await h.settle(1200); }), { w: 1280 }),
  /* the song menu */
  st('gen-menu', then(scene('ace', 'studio'), async h => { await h.click('#latest .song[data-id="1"] [data-songmenu]'); await h.settle(500); }), { pseudo: true }),
  st('gen-menu-mm3-dark', then(scene('mm3', 'studio'), async h => { await h.click('#latest .song[data-id="3"] [data-songmenu]'); await h.settle(500); }), { scheme: 'dark' }),
  /* the toast stack: an Undo moved up by a plain message */
  st('gen-toasts', then(scene('ace', 'studio'), async h => { await h.js(() => { toast('Removed the Seed module', { label: 'Undo', fn: () => {} }); toast('Cancelled 2 songs'); }); await h.settle(300); })),
  st('gen-toasts-dark', then(scene('ace', 'studio'), async h => { await h.js(() => { toast('Removed the Seed module', { label: 'Undo', fn: () => {} }); toast('Cancelled 2 songs'); }); await h.settle(300); }), { scheme: 'dark' }),
];

/** The regions compared, by their keys: the go bar and what it holds (the player, the line, Drafts, Cancel,
 *  Generate), Latest and its grid, the song menu, the newest toast and the one above it. */
export const SCOPES = { generate: ['#centre>div:4', '#player', '#meter', '#draftsBtn', '#cancelGo', '#go', '#centre>section:5', '#songCount', '#latest', '#menu', '#toast', 'html>body:1>div:13'] };

/** This file's visual set, for npm run visual:all (visual-all.mjs): the states it compares (
 *  a regular expression on their ids) and the scope it compares them in. */
export const VISUAL_SET = { name: 'generate', only: '^gen-', scope: 'generate' };
