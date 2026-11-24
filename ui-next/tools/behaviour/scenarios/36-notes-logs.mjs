// The Concept notes and the engine log (package 11), as scenarios that hold on the concept page too (run them there with
// --page), but for the one marked pending, which is the concept's F-18 for the notes. The notes: a modal drawer from the
// top bar (its scrim, focus in and back, Tab round it, Escape, the scrim and × closing it, its exit, the page's keys
// waiting under it, its sections, its × staying in reach as it scrolls, its edge fades, how it stacks over the log);
// the engine log: a dock opened from the VRAM meter (aria-expanded, focus to Close and back, the page's room at its
// foot, the go bar and the toast standing on it, a press elsewhere leaving it, Escape closing it last of all, its exit)
// and what it says on each engine, following a model choice. Each uses only what both pages share (S, render, get,
// toast, UNDO, the markup's ids and data-* hooks).

/* a desk, and the log's lines as a helper in the page (the checks run there) */
const desk = (tier, mods = [], pedals = []) => t => t.do(`${tier} desk`, ([tier, mods, pedals]) => {
  window.__lines = () => document.querySelector('#logText').textContent.split('\n');
  S.tier = tier; S.mods = mods; S.pedals = pedals; render();
}, [tier, mods, pedals], 900);
/* the card each engine has in the engine panel, by the concept's ids, which both pages write as data-backend */
const toEngine = async (t, bk) => { await t.click('#engBtn'); await t.click(`#engPanel .ep-card[data-backend="${bk}"]`, 900); await t.press('Escape'); };

export default [
  { id: 'notes-open-as-a-modal-drawer-and-give-focus-back', run: async t => {
    await desk('studio', ['avoid', 'guidance'])(t);
    await t.ok('closed, the drawer and its scrim are hidden', () => document.querySelector('#notes').hidden && document.querySelector('#notesScrim').hidden);
    await t.click('#notesBtn');
    await t.ok('the notes button opens the drawer over its scrim, focus on its ×', () => !document.querySelector('#notes').hidden && !document.querySelector('#notesScrim').hidden && document.activeElement.id === 'notesClose');
    await t.ok('a modal dialog named by its heading, the × named', () => { const n = document.querySelector('#notes'); return n.getAttribute('role') === 'dialog' && n.getAttribute('aria-modal') === 'true' && document.getElementById(n.getAttribute('aria-labelledby'))?.textContent === 'Three tiers, one desk' && document.querySelector('#notesClose').getAttribute('aria-label') === 'Close notes'; });
    await t.ok('it stands at the window\'s right edge, over the scrim, the scrim over the page', () => { const n = document.querySelector('#notes').getBoundingClientRect(), z = el => +getComputedStyle(document.querySelector(el)).zIndex; return Math.abs(n.right - document.documentElement.clientWidth) < 1 && n.top === 0 && z('#notes') > z('#notesScrim') && z('#notesScrim') > z('.top'); });
    await t.press('Tab');
    await t.ok('Tab stays inside it', () => document.querySelector('#notes').contains(document.activeElement));
    await t.press('Shift+Tab');
    await t.ok('so does Shift+Tab', () => document.querySelector('#notes').contains(document.activeElement));
    await t.press('Escape');
    await t.ok('Escape closes it and its scrim, focus back on the notes button', () => document.querySelector('#notes').hidden && document.querySelector('#notesScrim').hidden && document.activeElement.id === 'notesBtn');
    await t.press('Enter');
    await t.ok('Enter on the button opens it again, focus on its ×', () => !document.querySelector('#notes').hidden && document.activeElement.id === 'notesClose');
    await t.press('Enter');
    await t.ok('Enter on its × closes it, focus back on the button', () => document.querySelector('#notes').hidden && document.activeElement.id === 'notesBtn');
    await t.click('#notesBtn'); await t.click('#notesScrim', 300, [0.1, 0.5]);
    await t.ok('a press on the scrim closes it', () => document.querySelector('#notes').hidden && document.querySelector('#notesScrim').hidden);
    await t.click('#notesBtn'); await t.click('#notesClose');
    await t.ok('so does a click on its ×, focus back on the button', () => document.querySelector('#notes').hidden && document.activeElement.id === 'notesBtn');
  } },
  { id: 'notes-leave-as-they-came', run: async t => {
    await t.click('#notesBtn');
    await t.ok('it slides in from the right and its scrim fades in (the motion tokens)', () => getComputedStyle(document.querySelector('#notes')).animationName === 'ovDrawer' && getComputedStyle(document.querySelector('#notesScrim')).animationName === 'ovFade');
    const exit = await t.js(() => {
      document.querySelector('#notesClose').click();
      const g = [...document.querySelectorAll('.ov-ghost')];
      return document.querySelector('#notes').hidden && g.length === 2 && g.some(x => x.classList.contains('drawer')) && g.some(x => x.classList.contains('notes-scrim')) && g.every(x => x.inert && x.getAttribute('aria-hidden') === 'true' && !x.hasAttribute('role'));
    });
    await t.ok('closing hides it at once and leaves an inert copy sliding out, and one of the scrim fading', r => r, exit);
    await t.do('the exit plays', () => {}, null, 400);
    await t.ok('the copies are gone once their exit has played', () => !document.querySelector('.ov-ghost'));
  } },
  { id: 'the-page-waits-under-the-notes', run: async t => {
    await desk('studio', ['post'], ['ss'])(t);
    await t.focus('[data-knob="pp.ss.str"]'); await t.press('ArrowUp');
    const was = await t.js(() => get('pp.ss.str'));
    await t.click('#notesBtn');
    await t.press('Control+z');
    await t.ok('Ctrl+Z under the notes takes nothing back and says to close them first', was => get('pp.ss.str') === was && /Close this first/.test(document.querySelector('#toast .toast-msg')?.textContent || ''), was);
    await t.press('F9');
    await t.ok('F9 opens no picker over them', () => document.querySelector('#pickerWrap').hidden && !document.querySelector('#notes').hidden);
    await t.press('Escape');
    await t.ok('Escape closes the notes', () => document.querySelector('#notes').hidden);
  } },
  { id: 'notes-say-what-the-page-does', run: async t => {
    await t.click('#notesBtn');
    await t.eq('its heading, then the concept\'s five sections, in order', () => [document.querySelector('#notes h2').textContent, ...[...document.querySelectorAll('#notes h3')].map(h => h.textContent)],
      ['Three tiers, one desk', 'Tiers set how much each module shows', 'Adding modules', 'Workflows and revisions', 'Your work is kept', 'Grounding']);
    await t.eq('each section\'s points', () => [...document.querySelectorAll('#notes ul')].map(u => u.children.length), [8, 5, 6, 8, 1]);
    await t.eq('the words the concept sets in bold', () => [...document.querySelectorAll('#notes li b')].map(b => b.textContent),
      ['Quick', 'Studio', 'Lab', 'Changing tier', 'The desk reads like a page', 'One way to read every control', 'Off is one state', 'F9', 'workflow', 'revision', 'yours', 'Saved', 'Not saved', 'Touch', 'Undo', 'Redo', 'Drafts', 'Switching engine', 'Build by section']);
    await t.ok('the top bar\'s point names the engine log', () => /click it for the engine log, which docks under the go bar until you close it/.test(document.querySelector('#notes').textContent));
  } },
  { id: 'notes-scroll-with-their-close-in-reach', w: 1440, h: 700, run: async t => {
    await t.click('#notesBtn');
    await t.ok('they scroll without a scroll bar, their foot fading', () => { const n = document.querySelector('#notes'); return n.scrollHeight > n.clientHeight && n.classList.contains('noscroll') && n.classList.contains('fade-own') && n.classList.contains('fade-b') && !n.classList.contains('fade-t'); });
    await t.do('scrolled to the foot', () => { const n = document.querySelector('#notes'); n.scrollTop = n.scrollHeight; n.dispatchEvent(new Event('scroll')); });
    await t.ok('at the foot the head fades, and the × still stands in view at the top (S-35)', () => { const n = document.querySelector('#notes'), x = document.querySelector('#notesClose').getBoundingClientRect(); return n.classList.contains('fade-t') && !n.classList.contains('fade-b') && x.top >= 0 && x.bottom <= 60 && !!document.elementFromPoint(x.left + x.width / 2, x.top + x.height / 2)?.closest('#notesClose'); });
    await t.click('#notesClose');
    await t.ok('and closes them', () => document.querySelector('#notes').hidden);
  } },
  { id: 'the-meter-opens-the-engine-log-as-a-dock', run: async t => {
    await desk('studio', ['avoid', 'guidance'])(t);
    await t.ok('closed, the log is hidden and the meter says so', () => document.querySelector('#logs').hidden && document.querySelector('#engMeter').getAttribute('aria-expanded') === 'false' && !document.documentElement.classList.contains('logs-open'));
    await t.click('#engMeter');
    await t.ok('the VRAM meter opens it, says it is open, focus on its Close', () => !document.querySelector('#logs').hidden && document.querySelector('#engMeter').getAttribute('aria-expanded') === 'true' && document.activeElement.id === 'logsClose');
    await t.ok('a region named Engine log, along the window\'s foot, rising from it', () => { const l = document.querySelector('#logs'), r = l.getBoundingClientRect(); return l.getAttribute('role') === 'region' && l.getAttribute('aria-label') === 'Engine log' && Math.abs(r.bottom - innerHeight) < 1 && r.left === 0 && getComputedStyle(l).animationName === 'ovSheet'; });
    await t.ok('the page has its height of room at its foot, and the go bar stands on it (F-15)', () => { const h = Math.round(document.querySelector('#logs').getBoundingClientRect().height); return document.documentElement.classList.contains('logs-open') && document.documentElement.style.getPropertyValue('--logs-h') === h + 'px' && getComputedStyle(document.body).paddingBottom === h + 'px' && document.querySelector('.gobar').getBoundingClientRect().bottom <= document.querySelector('#logs').getBoundingClientRect().top + 1; });
    await t.do('a toast', () => toast('Something happened'));
    await t.ok('a toast stands above it', () => document.querySelector('#toast').getBoundingClientRect().bottom <= document.querySelector('#logs').getBoundingClientRect().top);
    await t.click('.top .brand');
    await t.ok('a press elsewhere leaves it open', () => !document.querySelector('#logs').hidden);
    await t.click('#engMeter');
    await t.ok('a second press on the meter closes it, says so, and gives the room back', () => document.querySelector('#logs').hidden && document.querySelector('#engMeter').getAttribute('aria-expanded') === 'false' && !document.documentElement.classList.contains('logs-open') && getComputedStyle(document.body).paddingBottom === '0px');
    await t.click('#engMeter'); await t.click('#logsClose');
    await t.ok('its Close closes it, focus back on the meter', () => document.querySelector('#logs').hidden && document.activeElement.id === 'engMeter');
    await t.press('Enter');
    await t.ok('Enter on the meter opens it', () => !document.querySelector('#logs').hidden && document.activeElement.id === 'logsClose');
    await t.press('Escape');
    await t.ok('Escape closes it, focus back on the meter', () => document.querySelector('#logs').hidden && document.activeElement.id === 'engMeter');
  } },
  { id: 'escape-closes-the-log-last', run: async t => {
    await desk('studio', ['avoid', 'guidance'])(t);
    await t.click('#engMeter'); await t.click('#notesBtn');
    await t.ok('the notes open over the log', () => !document.querySelector('#notes').hidden && !document.querySelector('#logs').hidden && +getComputedStyle(document.querySelector('#notesScrim')).zIndex > +getComputedStyle(document.querySelector('#logs')).zIndex);
    await t.press('Escape');
    await t.ok('Escape closes the notes only', () => document.querySelector('#notes').hidden && !document.querySelector('#logs').hidden);
    await t.click('#engBtn');
    await t.press('Escape');
    await t.ok('then the engine panel only', () => document.querySelector('#engPanel').hidden && !document.querySelector('#logs').hidden);
    await t.focus('.top .brand'); await t.press('Escape');
    await t.ok('then the log, focus left where it was', () => document.querySelector('#logs').hidden && document.activeElement.matches('.top .brand'));
    await t.click('#engMeter');
    await t.ok('closing it plays its exit: an inert copy sinking out', () => { document.querySelector('#logsClose').click(); const g = document.querySelector('.ov-ghost'); return document.querySelector('#logs').hidden && !!g && g.classList.contains('logs') && g.inert; });
  } },
  { id: 'the-log-says-what-the-engine-holds', run: async t => {
    await desk('studio', ['avoid', 'guidance'])(t);
    await t.eq('ACE-Step: the engine and the models it renders with', () => document.querySelector('#logEngine').textContent, 'ACE-Step 1.5 · XL Turbo Q8_0, 4B Q8_0');
    await t.eq('the server, the models by role, the plugins, the VRAM and the queue', () => __lines(), ['[engine]  ace-server up, 5 backends registered (HeartMuLa in integration testing)', '[ace]     sound XL Turbo Q8_0, planner 4B Q8_0, decoder ScragVAE',
      '[plugins] 46 solvers · 17 schedules · 18 guidance modes', '[vram]    13.3 / 31.8 GB loaded · the next Generate needs about 15.6 GB', '[queue]   idle']);
    await t.ok('its VRAM is the meter\'s', () => __lines()[3].startsWith(`[vram]    ${document.querySelector('#emUse').textContent} / 31.8 GB`));
    await t.click('#engBtn'); await t.click('#engPanel [data-pick="model.dit"]'); await t.click('#dd [data-opt="Turbo · Q4_K_M"]'); await t.press('Escape');
    await t.ok('a model chosen in the engine panel is named whole, at once, the VRAM with it', () => document.querySelector('#logEngine').textContent === 'ACE-Step 1.5 · Turbo Q4_K_M, 4B Q8_0' && /sound Turbo Q4_K_M,/.test(__lines()[1]) && __lines()[3].startsWith(`[vram]    ${document.querySelector('#emUse').textContent} /`) || [document.querySelector('#logEngine').textContent, __lines()]);
    await toEngine(t, 'mm3');
    await t.eq('MiniMax-Music3: its own models, its own roles', () => [document.querySelector('#logEngine').textContent, __lines()[1], __lines()[3]],
      ['MiniMax-Music3 · Flow DiT Q8_0, 8B planner Q8_0', '[mm3]     sound Flow DiT Q8_0, planner 8B planner Q8_0, detail Depth decoder Q8_0', '[vram]    16.2 / 31.8 GB loaded · the next Generate needs about 16.2 GB']);
    await toEngine(t, 'hm');
    await t.eq('HeartMuLa: its variant first', () => [document.querySelector('#logEngine').textContent, __lines()[1]],
      ['HeartMuLa · HeartMuLa 3B, Global Q8_0, Local Q8_0', '[hm]      variant HeartMuLa 3B, language Global Q8_0, decoder Local Q8_0, codec Codec flow Q8_0']);
    await toEngine(t, 'sa3');
    await t.eq('Stable Audio 3: one model', () => [document.querySelector('#logEngine').textContent, __lines()[1], __lines()[3]], ['Stable Audio 3 · SA3 DiT BF16', '[sa3]     sound SA3 DiT BF16', '[vram]    5.7 / 31.8 GB loaded · the next Generate needs about 5.7 GB']);
  } },
  { id: 'the-log-on-a-phone', w: 390, h: 844, run: async t => {
    await desk('studio', ['avoid', 'guidance'])(t);
    await t.click('#engMeter');
    await t.ok('it takes at most a third of the screen, and its head is one row: Close beside Engine (S-23)', () => { const l = document.querySelector('#logs').getBoundingClientRect(), b = document.querySelector('.logs-head > b').getBoundingClientRect(), c = document.querySelector('#logsClose').getBoundingClientRect(); return l.height <= innerHeight / 3 + 1 && c.top < b.bottom && c.bottom > b.top && c.right <= innerWidth; });
    await t.click('#notesBtn');
    await t.ok('the notes take the whole width', () => { const n = document.querySelector('#notes').getBoundingClientRect(); return n.left === 0 && Math.abs(n.width - document.documentElement.clientWidth) < 1; });
  } },
  { id: 'a-toast-joins-the-notes-tab-cycle', pending: { finding: 'F-18', right: 'A toast raised while the picker or the Concept notes are open joins their Tab cycle while it shows, so its buttons can be reached by the keyboard.' }, run: async t => {
    await t.click('#notesBtn');
    await t.do('a toast with an action', () => toast('Something happened', { label: 'Undo', fn: () => {} }));
    let reached = false; for (let i = 0; i < 6 && !reached; i++) { await t.page.keyboard.press('Tab'); reached = await t.js(() => !!document.activeElement?.closest('#toast')); }
    await t.ok('Tab reaches the toast from inside the notes', r => r || 'never reached', reached);
    let back = false; for (let i = 0; i < 4 && !back; i++) { await t.page.keyboard.press('Tab'); back = await t.js(() => document.querySelector('#notes').contains(document.activeElement)); }
    await t.ok('and goes on round to the notes', r => r || 'never came back', back);
  } },
];
