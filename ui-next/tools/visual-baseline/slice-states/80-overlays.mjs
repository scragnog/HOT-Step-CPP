// Package 8's states (one file of tools/visual-baseline/slice-states/, gathered by slice-states.mjs): the concept's
// overlay and drag states (states.mjs ov-picker, picker-filter, picker-cat, the popover, the keyboard tip, drag-card,
// drag-move, drag-picker-peek, drag-pedal) on a desk of the modules, pedals and side unit the rebuild draws whole (the
// slice's), with the lyric sheet in place (package 4). The scope `overlays` compares
// the picker, the popover, the tip and the desk under a drag.
//
//   node tools/visual-baseline/vb.mjs capture concept-overlays --states slice --scope overlays --only "^ov8-" --page <concept>/hotstep-tiers.html
//   node tools/visual-baseline/vb.mjs capture rebuild-overlays --states slice --scope overlays --only "^ov8-" --plan-from concept-overlays
//   node tools/visual-baseline/vb.mjs compare concept-overlays rebuild-overlays --scope overlays --accept tools/visual-baseline/overlays-accepted.json

const scene = (tier, mods, pedals = [], bk = 'ace') => async h => {
  await h.js(([bk, tier, mods, pedals]) => { if (bk !== S.backend) { if (typeof enterEngine === 'function') enterEngine(bk); else S.backend = bk; } S.tier = tier; S.mods = mods; S.pedals = pedals; render(); }, [bk, tier, mods, pedals]);
  await h.settle(900);
};
const desk = scene('studio', ['avoid', 'guidance', 'post'], ['gain', 'norm', 'wh']);
const picker = async h => { await h.press('F9'); await h.settle(900); await h.expect(() => !document.querySelector('#pickerWrap').hidden); };
/* a drag in progress: dragstart on the source, dragover on the target (states.mjs drag) */
const drag = (src, to) => async h => {
  await h.js(src => { const el = document.querySelector(src); window.__dt = new DataTransfer(); el.dispatchEvent(new DragEvent('dragstart', { bubbles: true, cancelable: true, dataTransfer: window.__dt })); }, src);
  await h.settle(200);
  await h.js(to => { const t = document.querySelector(to), r = t.getBoundingClientRect(); const o = { bubbles: true, cancelable: true, dataTransfer: window.__dt, clientX: r.left + r.width * 0.3, clientY: r.top + 30 }; t.dispatchEvent(new DragEvent('dragenter', o)); t.dispatchEvent(new DragEvent('dragover', o)); }, to);
  await h.settle(600);
};

const LIST = [];
const add = s => LIST.push({ w: 1440, h: 900, scheme: 'light', reduced: false, pseudo: false, shot: false, ...s });

/* the picker: open, in both themes and at the sizes its wall is laid out for, filtered, by category, a Lab module's
   offer in its info line, what was just added */
for (const scheme of ['light', 'dark']) add({ id: `ov8-picker-${scheme}`, scheme, pseudo: scheme === 'light', shot: scheme === 'light' ? '#pickerWrap' : false, run: async h => { await desk(h); await picker(h); } });
for (const [w, hh] of [[1920, 1080], [1280, 720], [1000, 800]]) add({ id: `ov8-picker-w${w}`, w, h: hh, run: async h => { await desk(h); await picker(h); } });
add({ id: 'ov8-picker-filter', run: async h => { await desk(h); await picker(h); await h.page.keyboard.type('se'); await h.settle(600); await h.press('ArrowDown'); await h.settle(); } });
add({ id: 'ov8-picker-filter-typed', run: async h => { await desk(h); await picker(h); await h.page.keyboard.type('rep'); await h.settle(600); } });
add({ id: 'ov8-picker-cat', run: async h => { await desk(h); await picker(h); await h.js(() => document.querySelectorAll('#pkBar button')[2]?.click()); await h.settle(); } });
add({ id: 'ov8-picker-cat-pedals-dark', scheme: 'dark', run: async h => { await desk(h); await picker(h); await h.js(() => document.querySelector('#pkBar [data-pkcat="Pedals"]').click()); await h.settle(); } });
add({ id: 'ov8-picker-lab-offer', run: async h => { await desk(h); await picker(h); await h.js(() => document.querySelector('.fl-it[data-pkid="perf"]').focus()); await h.press('Enter'); await h.settle(); } });
add({ id: 'ov8-picker-added', run: async h => { await desk(h); await picker(h); await h.js(() => document.querySelector('.fl-it[data-pkid="pd:ss"]').focus()); await h.press('Enter'); await h.settle(); } });

/* the popover: the Lab offer by its card, in both themes, and on an engine that would park the module; a tile's moves */
for (const scheme of ['light', 'dark']) add({ id: `ov8-pop-lab-${scheme}`, scheme, pseudo: scheme === 'light', run: async h => { await desk(h); await h.click('#browser .bcard[data-additem="perf"]'); await h.settle(); await h.expect(() => !document.querySelector('#pop').hidden); } });
add({ id: 'ov8-pop-lab-parked', run: async h => { await scene('studio', ['yuebatch'], [], 'yue')(h); await h.click('#browser .bcard[data-additem="guidance"]'); await h.settle(); } });
add({ id: 'ov8-pop-moves', run: async h => { await desk(h); await h.click('.tile[data-m="avoid"] .grip'); await h.settle(); await h.expect(() => !document.querySelector('#pop').hidden); } });

/* the tip: by the keyboard on the VRAM meter, by the pointer on a pedal's knob */
add({ id: 'ov8-tip-keyboard', run: async h => { await desk(h); await h.js(() => document.querySelector('#engBtn').focus()); await h.press('Tab'); await h.settle(900); } });
add({ id: 'ov8-tip-hover', w: 1920, h: 1080, run: async h => { await scene('lab', ['post'], ['gain'])(h); await h.page.hover('.pedal[data-pid="gain"] .knob'); await h.settle(900); } });

/* a drag in progress: a card onto the desk, a tile being moved, from the picker (it peeks), a pedal onto the board */
add({ id: 'ov8-drag-card', w: 1600, h: 1080, run: async h => { await scene('studio', ['post'], ['gain'])(h); await drag('#browser [data-drag="avoid"]', '.tile[data-m="post"]')(h); } });
add({ id: 'ov8-drag-move', w: 1600, h: 1080, run: async h => { await desk(h); await drag('.tile[data-m="avoid"] [data-drag^="move:"]', '.tile[data-m="post"]')(h); } });
add({ id: 'ov8-drag-move-dark', w: 1600, h: 1080, scheme: 'dark', run: async h => { await desk(h); await drag('.tile[data-m="avoid"] [data-drag^="move:"]', '.tile[data-m="post"]')(h); } });
add({ id: 'ov8-drag-picker-peek', w: 1600, h: 1080, run: async h => { await scene('studio', ['post'], ['gain'])(h); await picker(h); await drag('#pkLeft [data-drag="avoid"]', '.tile[data-m="post"]')(h); } });
add({ id: 'ov8-drag-pedal', w: 1600, h: 1080, run: async h => { await desk(h); await drag('#browser [data-drag="pd:ss"]', '.tile[data-m="post"]')(h); } });

export const STATES = LIST;
export const SCOPES = { overlays: ['#pickerWrap', '#pop', '#tip', '#bento', '#browser'] };

/** This file's visual set, for npm run visual:all (visual-all.mjs): the states it compares (
 *  a regular expression on their ids), the scope it compares them in, and its accepted differences. */
export const VISUAL_SET = { name: 'overlays', only: '^ov8-', scope: 'overlays', accept: 'overlays-accepted.json' };
