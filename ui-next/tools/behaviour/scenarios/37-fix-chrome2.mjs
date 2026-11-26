// The second fix round around the desk: what the core round's re-reading of the concept's scenarios found for this
// share (acceptance/fix-core.json, RA-06), each reached the way a user reaches it (real clicks, keys and taps from the
// page's first state), on the rebuild only: they read the rebuild's markup where the concept's differs (a choice's
// data-opt is its value, deviation 10).

/* what the toast says, installed in the page for the checks (which run there) */
const helpers = t => t.js(() => { window.__toast = () => (document.querySelector('#toast')?.hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent ?? ''); });

/* a finger's tap at an element's centre, held for holdMs (18-tips-touch.mjs touchAt) */
const touchAt = async (t, sel, holdMs) => {
  const r = await t.js(sel => { const el = document.querySelector(sel); el.scrollIntoView({ block: 'center', behavior: 'instant' }); const b = el.getBoundingClientRect(); return { x: b.left + b.width / 2, y: b.top + b.height / 2 }; }, sel);
  const cdp = await t.page.context().newCDPSession(t.page);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: r.x, y: r.y }] });
  await t.settle(holdMs);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await t.settle(300); await cdp.detach();
};

export default [
  /* RC-01's remainder (07-overlays-keys/theme-nav): the Create page's keys do nothing on another page (31-screens.js,
     33-picker.js, 31-shell.js, 35-player.js, 14-undo.js). F9 opened the picker over the Library, whose heading then
     covered the top bar's way back to Create. */
  { id: 'rc01-create-keys-do-nothing-on-another-page', run: async t => {
    await helpers(t);
    await t.click('#latest .song[data-id="2"] [data-play]');
    await t.ok('a song plays', () => document.querySelector('#ppause')?.getAttribute('aria-label') === 'Pause');
    for (const nav of ['Library', 'Studios', 'Training', 'Settings']) {
      await t.click(`.top [data-nav="${nav}"]`);
      await t.do('focus leaves for the page', () => document.activeElement?.blur());
      await t.press('F9');
      await t.ok(`F9 opens no picker on ${nav}`, () => document.querySelector('#pickerWrap').hidden && !document.querySelector('#other').hidden);
      await t.press('Alt+3');
      await t.eq(`Alt+3 leaves the tier on ${nav}`, () => document.querySelector('#tiers [aria-checked="true"], #tiers [aria-pressed="true"]')?.dataset.tier, 'quick');
      await t.press(' ');
      await t.ok(`Space leaves the player playing on ${nav}`, () => document.querySelector('#ppause')?.getAttribute('aria-label') === 'Pause');
    }
    await t.click('#notesBtn');
    await t.press('Control+z');
    await t.ok('Ctrl+Z under the notes on another page says nothing (it is no desk to close)', () => !/Close this first|Nothing to undo|Undone/.test(__toast()) || __toast());
    await t.press('Escape');
    await t.click('.top [data-nav="create"]');
    await t.ok('the top bar\'s Create brings Create back, nothing covering it', () => !document.getElementById('app').hidden && document.getElementById('other').hidden);
    await t.press('F9');
    await t.ok('where F9 opens the picker again', () => !document.querySelector('#pickerWrap').hidden);
  } },
  /* E-15 (15-other-engines/tempo-has-one-name): every list that describes a module says what its tile holds on the
     engine in use: on MiniMax-Music3 Tempo is "Tempo & key", with Beat I/O and no length */
  { id: 'e15-modules-describe-tempo-per-engine', run: async t => {
    await t.click('#tiers [data-tier="studio"]');
    await t.ok('on ACE-Step the card says Tempo holds a length', () => /^Tempo, key & length\. BPM with tap tempo, key on a circle of fifths, time signature and length\.$/.test(document.querySelector('#browser .bcard[data-additem="tempo"]').title) || document.querySelector('#browser .bcard[data-additem="tempo"]').title);
    await t.click('#engBtn');
    await t.click('#engPanel [data-backend="mm3"]');
    await t.press('Escape');
    await t.ok('the Modules panel\'s card names it as its tile does and says what it holds there', () => { const c = document.querySelector('#browser .bcard[data-additem="tempo"]'); return (/^Tempo & key\. BPM with tap tempo, key on a circle of fifths, and Beat I\/O\.$/.test(c.title) && c.getAttribute('aria-label') === c.title) || c.title; });
    await t.click('#brSearch'); await t.type('beat');
    await t.ok('the filter finds it by those words', () => [...document.querySelectorAll('#brList .bcard')].some(c => c.dataset.additem === 'tempo') || [...document.querySelectorAll('#brList .bcard')].map(c => c.dataset.additem));
    await t.press('Escape');
    await t.click('#browser .bcard[data-additem="tempo"]');
    await t.ok('its tile has that name', () => /^Tempo & key/.test(document.querySelector('.tile[data-m="tempo"] .tile-head h3')?.textContent ?? '') || document.querySelector('.tile[data-m="tempo"] .tile-head')?.textContent);
    await t.press('F9');
    await t.click('#pkSearch'); await t.type('tempo');
    await t.ok('the picker\'s info line describes it as on MiniMax-Music3 too', () => /Beat I\/O/.test(document.querySelector('#pkInfo')?.textContent ?? '') || document.querySelector('#pkInfo')?.textContent);
  } },
  /* S-02 (18-tips-touch/plugin-menu-reads-under-a-finger, which taps the option already chosen here: deviation 10): a
     first tap on a symbol of a plugin menu shows it in the header and says a second tap uses it; the second chooses */
  { id: 's02-plugin-symbol-first-tap-shows-it', w: 820, h: 1180, touch: true, run: async t => {
    await touchAt(t, '#tiers [data-tier="studio"]', 60);
    await touchAt(t, '[data-open="right"]', 60);
    await touchAt(t, '#browser .bcard[data-additem="sampler"]', 60);
    await touchAt(t, '.tile[data-m="sampler"] [data-pick="samp.solver"]', 60);
    await t.ok('the solver table is open', () => !document.querySelector('#dd').hidden && !!document.querySelector('#dd .opt-pt'));
    const other = await t.js(() => [...document.querySelectorAll('#dd .opt-pt')].find(b => b.getAttribute('aria-selected') !== 'true').dataset.opt);
    const was = await t.js(() => document.querySelector('#dd .opt-pt[aria-selected="true"]').dataset.opt);
    await touchAt(t, `#dd .opt-pt[data-opt="${other}"]`, 60);
    await t.eq('a first tap shows the symbol in the header, offers Use it and chooses nothing', o => [document.querySelector('#dd .opt-pt[aria-selected="true"]')?.dataset.opt, document.querySelector('#dd .opt-pt.pv')?.dataset.opt, !!document.querySelector('#ptBig [data-ptuse]')], [was, other, true], other);
    await t.ok('and says a second tap uses it', () => /^.+\. Tap again to use it$/.test(document.querySelector('#dd .pt-head [role="status"]')?.textContent ?? '') || document.querySelector('#dd .pt-head [role="status"]')?.textContent);
    await touchAt(t, `#dd .opt-pt[data-opt="${other}"]`, 60);
    await t.ok('a second tap chooses it and closes the menu', o => document.querySelector('#dd').hidden && document.querySelector('.tile[data-m="sampler"] [data-pick="samp.solver"]').textContent.trim().length > 0 && !!o, other);
    await touchAt(t, '.tile[data-m="sampler"] [data-pick="samp.solver"]', 60);
    await t.eq('it is the one chosen now', () => document.querySelector('#dd .opt-pt[aria-selected="true"]')?.dataset.opt, other);
  } },
  /* 09-undo-storage/undo-keeps-a-later-edit-of-the-same-thing, as a user reaches it: an Undo or Redo that leaves a
     later edit alone says so; one with nothing left to move says that instead of "Undone" (F-24) */
  { id: 'undo-says-what-it-left-alone', run: async t => {
    await helpers(t);
    await t.click('#tiers [data-tier="studio"]');
    await t.click('#aiBtn');
    await t.ok('Write it for me fills the description and the lyrics, with Undo', () => document.querySelector('#prompt').value.length > 10 && document.querySelector('#toast .toast-act')?.textContent === 'Undo');
    await t.click('#prompt'); await t.press('End'); await t.type(', with cello');
    await t.click('.top .brand');
    await t.click('#toast .toast-act');
    await t.ok('its Undo puts back the lyrics, keeps the description typed since, and says so', () => (/with cello$/.test(document.querySelector('#prompt').value) && __toast() === 'Undone: Write it for me. The description changed since, so it was left as it is') || [document.querySelector('#prompt').value, __toast()]);
    await t.click('#toast .toast-act');
    await t.ok('its Redo says the same', () => __toast() === 'Redone: Write it for me. The description changed since, so it was left as it is' || __toast());
    await t.click('.tile[data-m="lyrics"] [data-bind="lyrics.text"]'); await t.press('Control+End'); await t.type('!');
    await t.click('.top .brand');
    await t.click('#toast .toast-act');
    await t.ok('an Undo with nothing of its step left to put back says so, and why, not "Undone"', () => __toast() === 'Nothing to put back for Write it for me: the description and the lyrics changed since' || __toast());
    await t.ok('and leaves both as they are', () => (/with cello$/.test(document.querySelector('#prompt').value) && /!$/.test(document.querySelector('[data-bind="lyrics.text"]').value)) || [document.querySelector('#prompt').value, document.querySelector('[data-bind="lyrics.text"]').value]);
  } },
];
