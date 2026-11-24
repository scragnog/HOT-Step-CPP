// The consolidation round's core share (review r3, RA-): each defect it fixed, driven as a user reaches it, with real
// clicks and keys from the page's first state (no scene sets what the user would have to do). Rebuild only: they read
// the rebuild's markup where the concept has none of the same.

/* what the go bar and the Takes tile say, installed in the page for the checks (which run there) */
const helpers = t => t.js(() => { window.__fc = {
  line: () => document.querySelector('#meter')?.textContent ?? '',
  cost: () => document.querySelector('.tile[data-m="takes"] .tk-gb')?.textContent ?? '',
  need: () => (/needs (\d+\.\d) of/.exec(document.querySelector('#meter')?.textContent ?? '') || [])[1],
}; });

export default [
  /* RA-01: one VRAM calculation. The Takes tile, the go bar and the meter say one figure, for the models chosen. */
  { id: 'takes-tile-and-go-bar-say-one-figure', w: 1920, h: 1080, run: async t => {
    await helpers(t);
    await t.click('#tiers [data-tier="studio"]');
    await t.click('#engBtn');
    await t.click('#engPanel [data-backend="mm3"]');
    await t.click('#engPanel [data-pick="model.lm"]');
    await t.click('#dd [data-opt="8B planner · F16"]');
    await t.press('Escape');
    await t.click('#browser .bcard[data-additem="takes"]');
    await t.click('.tile[data-m="takes"] [data-take="4"]');
    await t.ok('the go bar says the run will not fit, with its figure', () => /^Won't fit: needs \d+\.\d of 31\.8 GB/.test(__fc.line()) || __fc.line().slice(0, 120));
    await t.ok("the Takes tile says the same figure, and that it won't fit", () => {
      const need = __fc.need(), tile = __fc.cost(), over = document.querySelector('.tile[data-m="takes"] .tk-cost')?.classList.contains('over');
      return (!!need && tile === `${need} of 31.8 GB` && over) || [need, tile, over].join(' / ');
    });
    await t.ok('and the meter reads what the run needs', () => document.querySelector('#emUse')?.textContent === __fc.need() || [__fc.need(), document.querySelector('#emUse')?.textContent].join(' / '));
    await t.click('#meter [data-takesfit]');
    await t.ok('"Use N takes" sets the takes the tile then shows as fitting', () => { const c = document.querySelector('.tile[data-m="takes"] .tk-cost'); return (!/Won't fit/.test(__fc.line()) && !!c && !c.classList.contains('over')) || __fc.line().slice(0, 120); });
  } },
  /* RA-09: a toast's Undo takes back the step its action took, never an earlier one. */
  { id: 'reuse-with-the-same-words-offers-no-undo-of-your-typing', w: 1920, h: 1080, run: async t => {
    await t.click('#prompt');
    await t.type('Lo-fi hip hop, mellow electric piano');
    await t.click('#tiers [data-tier="studio"]');
    await t.click('#latest .song[data-id="1"] [data-songmenu]');
    await t.click('#menu [data-sa="reuse"]');
    await t.ok('the words were already there: no step, so no "Replaced" toast with an Undo', () => !/Replaced the description/.test(document.querySelector('#toast')?.textContent ?? '') || document.querySelector('#toast').textContent);
    await t.ok('the description is as typed', () => document.querySelector('#prompt').value === 'Lo-fi hip hop, mellow electric piano');
    await t.click('#prompt');
    await t.press('End');
    await t.type(', rain');
    await t.click('#tiers [data-tier="quick"]');
    await t.click('#latest .song[data-id="2"] [data-songmenu]');
    await t.click('#menu [data-sa="reuse"]');
    await t.ok('other words replace the box, and the toast offers Undo', () => document.querySelector('#prompt').value === 'Ambient, warm pads, slow evolving textures' && document.querySelector('#toast .toast-act')?.textContent === 'Undo');
    await t.click('#toast .toast-act');
    await t.ok('its Undo puts back what was typed, not less', () => document.querySelector('#prompt').value === 'Lo-fi hip hop, mellow electric piano, rain' || document.querySelector('#prompt').value);
  } },
  /* RA-08: every named action has a runner; the VST3 chain's "+ Add" says what the concept says. */
  { id: 'vst-chain-add-says-the-host-is-elsewhere', w: 1920, h: 1080, run: async t => {
    await t.click('#tiers [data-tier="lab"]');
    await t.click('#browser .bcard[data-additem="pd:vst"]');
    await t.click('.pedal[data-pid="vst"] button.pbtn');
    await t.ok('it says the host is not on this page', () => /The VST3 host isn't part of this concept/.test(document.querySelector('#toast')?.textContent ?? '') || document.querySelector('#toast')?.textContent);
    await t.ok('and no placeholder part is drawn anywhere', () => !document.querySelector('[data-seam]'));
  } },
];
