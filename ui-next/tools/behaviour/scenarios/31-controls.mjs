// The shared controls (package 1), each in a module the page draws today, as scenarios that hold on the concept page
// too (run them there with --page): the number box, the length ruler with its auto switch and markers, the plugin
// picker's table, grid and filter, a Quick row's value between its choices, scrub tokens, a pill and the How
// adventurous row, a checked text field that keeps or refuses what was typed, and a side unit's chip selector. Each uses
// only what both pages share (S, render, get, toast, the markup's data-* hooks; an option is found by its words, since
// the concept's data-opt is its label and the rebuild's its value, deviation 10). The pending ones state what the
// concept gets wrong (S-18) and pass here.

const desk = (tier, mods, vals = {}, backend = 'ace', pedals = []) => async t => {
  /* the selectors and the toast's words, for the checks (which run in the page) */
  await t.js(() => { window.__c = { RULER: '.tile[data-m="tempo"] [data-ruler="tempo.dur"]', CAP: '.tile[data-m="tempo"] [data-rcap]', NUM: '.tile[data-m="sampler"] [data-num="samp.steps"]',
    ROW: '.tile[data-m="sampler"] .qchoice[data-q="samp.steps"]', PILL: '.tile[data-m="planner"] .pill-tog[data-ledtog="lm.on"]', MOOD: '.tile[data-m="planner"] .mood',
    TS: '.tile[data-m="latent"] [data-bind="lat.ts"]', RS: '.tile[data-m="sampler"] [data-bind="samp.rseed"]', CH: '.pedal[data-xid="q"] .chipsw[data-ctl="pp.q.eval"]', TAKES: '.tile[data-m="build"] [data-scrub="yue.btakes"]',
    toast: () => (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '') }; });
  await t.do(`${tier} desk`, ([tier, mods, vals, backend, pedals]) => {
    if (backend !== S.backend) { if (typeof enterEngine === 'function') enterEngine(backend); else S.backend = backend; }
    S.backend = backend; S.tier = tier; S.mods = mods; S.pedals = pedals; Object.assign(S.vals, vals); render();
  }, [tier, mods, vals, backend, pedals]);
};
/* marks the option of the open list whose words are these, so a click can find it on either page */
const option = (t, words) => t.do(`find ${words}`, w => { const b = [...document.querySelectorAll('#dd .opt')].find(x => (x.querySelector('b')?.textContent ?? x.textContent).trim() === w || x.getAttribute('aria-label')?.startsWith(w + ',')); b?.setAttribute('data-bh', 'opt'); }, words);
const RULER = '.tile[data-m="tempo"] [data-ruler="tempo.dur"]', NUM = '.tile[data-m="sampler"] [data-num="samp.steps"]';

export default [
  { id: 'number-box-types-steps-and-scrubs', run: async t => {
    await desk('studio', ['sampler'])(t);
    await t.click(NUM); await t.type('20'); await t.press('Enter');
    await t.eq('typing a number and Enter sets it', () => [get('samp.steps'), document.querySelector(__c.NUM).value], [20, '20']);
    await t.press('ArrowUp'); await t.eq('ArrowUp steps', () => get('samp.steps'), 21);
    await t.press('PageUp'); await t.eq('PageUp steps ten', () => get('samp.steps'), 31);
    await t.press('Shift+ArrowDown'); await t.eq('Shift steps ten', () => get('samp.steps'), 21);
    await t.press('Control+a'); await t.type('fast'); await t.press('Enter');
    await t.ok('a word is refused, said and put back', () => (/Type a number for Steps/.test(__c.toast()) && document.querySelector(__c.NUM).value === '21' && get('samp.steps') === 21) || [__c.toast(), document.querySelector(__c.NUM).value]);
    await t.press('Control+a'); await t.type('500'); await t.press('Enter');
    await t.ok('out of range is clamped, and said', () => (get('samp.steps') === 300 && /Steps: 1 to 300\. Set to 300\./.test(__c.toast())) || [get('samp.steps'), __c.toast()]);
    await t.press('Control+a'); await t.type('77'); await t.press('Escape');
    await t.eq('Escape puts what is in force back', () => [document.querySelector(__c.NUM).value, get('samp.steps')], ['300', 300]);
    await t.mouseDrag(NUM, [[0, 10], [0, 20]]);
    await t.ok('dragging the box down scrubs it, and the box shows the value', () => (+get('samp.steps') < 300 && document.querySelector(__c.NUM).value === String(get('samp.steps'))) || [get('samp.steps'), document.querySelector(__c.NUM).value]);
    await t.ok('the seconds under it follow the steps', () => /^≈ \d+\.\d s$/.test(document.querySelector('.tile[data-m="sampler"] [data-stepsec]').textContent) || document.querySelector('.tile[data-m="sampler"] [data-stepsec]').textContent);
  } },
  { id: 'ruler-drags-keys-and-leaves-auto', run: async t => {
    await desk('studio', ['tempo'])(t);
    await t.ok('on auto the ruler shows the estimate', () => (document.querySelector(__c.CAP).textContent === '≈ 3:00' && document.querySelector(__c.RULER).classList.contains('auto') && document.querySelector(__c.RULER).getAttribute('aria-valuetext') === 'automatic') || document.querySelector(__c.CAP).textContent);
    await t.click('.tile[data-m="tempo"] [data-auto="tempo.dur"]');
    await t.ok('auto off: the length is the estimate, set', () => (get('tempo.dur') === 180 && document.querySelector(__c.CAP).textContent === '3:00' && !document.querySelector(__c.RULER).classList.contains('auto')) || [get('tempo.dur'), document.querySelector(__c.CAP).textContent]);
    await t.mouseDrag(RULER, [[40, 0]], [0.5, 0.5]);
    await t.ok('a press travels there and the cap follows the hand', () => (+get('tempo.dur') > 300 && /^\d+:\d\d$/.test(document.querySelector(__c.CAP).textContent)) || [get('tempo.dur'), document.querySelector(__c.CAP).textContent]);
    await t.focus(RULER);
    await t.press('End'); await t.eq('End', () => get('tempo.dur'), 600);
    await t.ok('at the end the cap stands flush with it', () => document.querySelector(__c.CAP).classList.contains('at-end'));
    await t.press('Home'); await t.eq('Home', () => get('tempo.dur'), 10);
    await t.press('ArrowUp'); await t.eq('an arrow steps', () => get('tempo.dur'), 15);
    await t.dblclick(RULER);
    await t.ok('a double-click puts it back on auto', () => document.querySelector(__c.RULER).classList.contains('auto') && document.querySelector(__c.CAP).textContent === '≈ 3:00');
  } },
  { id: 'ruler-says-what-auto-means-on-each-engine', run: async t => {
    await desk('studio', ['tempo'], {}, 'hm')(t);
    await t.eq('where the model ends the song, auto is "up to" its longest', () => document.querySelector(__c.CAP).textContent, 'up to 4:00');
    await t.click('.tile[data-m="tempo"] [data-auto="tempo.dur"]');
    await t.eq('auto off starts where auto was showing (S-17)', () => [get('tempo.dur'), document.querySelector(__c.CAP).textContent], [240, '4:00']);
    await desk('studio', ['tempo'], { 'tempo.dur.auto': true }, 'sa3')(t);
    await t.eq('where the length is exact, auto says it without an estimate', () => document.querySelector(__c.CAP).textContent, '0:30');
  } },
  { id: 'ruler-draws-beat-io-and-trim', run: async t => {
    await desk('lab', ['tempo'], { 'tempo.beatio': true, 'tempo.beatbars': '8', 'tempo.dur': 20, 'tempo.dur.auto': false, 'tempo.trim': true })(t);
    await t.eq('Beat I/O draws both ends', () => [...document.querySelectorAll('.tile[data-m="tempo"] [data-io]')].filter(e => !e.hidden).length, 2);
    await t.ok('a Beat I/O longer than the song says what it needs', () => /8 bars each way needs about 0:32 at 120 BPM/.test(document.querySelector('.tile[data-m="tempo"] [data-ionote]')?.textContent || '') || document.querySelector('[data-ionote]')?.textContent);
    await t.ok('Auto-trim draws its tag', () => { const el = document.querySelector('.tile[data-m="tempo"] [data-trim]'); return !!el && !el.hidden; });
  } },
  { id: 'plugin-picker-table-grid-and-filter', run: async t => {
    await desk('lab', ['sampler'])(t);
    await t.click('.tile[data-m="sampler"] [data-pick="samp.solver"]');
    await t.ok('a long plugin list opens as a periodic table, on the chosen one', () => !document.querySelector('#dd').hidden && document.querySelector('#dd').classList.contains('pt') && document.activeElement.matches('#dd [aria-selected="true"]'));
    await t.press('ArrowRight');
    await t.ok('the arrows move, and the header shows the option', () => (/^Heun/.test(document.activeElement.getAttribute('aria-label') || '') && /Heun/.test(document.querySelector('#ptBig').textContent)) || document.activeElement.outerHTML.slice(0, 120));
    await t.press('d');
    await t.ok('a letter types into the filter', () => document.activeElement.id === 'ddq' && document.querySelector('#ddq').value === 'd');
    await t.type('pm++ 3m'); await t.press('Enter');
    await t.eq('Enter in the filter takes the first match', () => [document.querySelector('.tile[data-m="sampler"] [data-pick="samp.solver"] .pk-val').textContent, document.querySelector('#dd').hidden], ['DPM++ 3M', true]);
    await t.ok('focus goes back to its button', () => document.activeElement.dataset.pick === 'samp.solver');
    await t.press('ArrowDown');
    await t.ok('ArrowDown on the button opens it', () => !document.querySelector('#dd').hidden);
    await t.press('Escape');
    await t.ok('Escape closes it and refocuses', () => document.querySelector('#dd').hidden && document.activeElement.dataset.pick === 'samp.solver');
    await t.click('.tile[data-m="sampler"] [data-pick="samp.sched"]');
    await t.ok('the schedules are a grid of curves', () => document.querySelector('#dd').classList.contains('schedw') && document.querySelectorAll('#dd .opt').length === 17 && document.querySelectorAll('#dd .opt .spark').length === 17);
    await t.click('#ddq'); await t.type('zzz');
    await t.ok('a filter with no match says so', () => !!document.querySelector('#dd .dd-none') && !document.querySelector('#dd .dd-none').hidden);
    await t.press('Escape');
    await t.click('.tile[data-m="sampler"] [data-pick="samp.sched"]');
    await t.press('End'); await t.press('Home');
    await t.ok('End and Home', () => document.activeElement.querySelector('b')?.textContent === 'Linear');
    await t.press('Tab');
    await t.ok('Tab closes it and carries on from its button', () => document.querySelector('#dd').hidden && !document.querySelector('#dd').contains(document.activeElement) && document.activeElement !== document.body && !document.activeElement.matches('[data-pick="samp.sched"]'));
    await t.click('.tile[data-m="sampler"] [data-pick="samp.sched"]');
    await option(t, 'Power'); await t.click('#dd [data-bh="opt"]');
    await t.ok('a click chooses, and the face shows it', () => document.querySelector('.tile[data-m="sampler"] [data-pick="samp.sched"] .pk-val').textContent === 'Power' || document.querySelector('.tile[data-m="sampler"] [data-pick="samp.sched"] .pk-val').textContent);
  } },
  { id: 'quick-row-between-its-choices', run: async t => {
    await desk('quick', ['sampler'], { 'samp.steps': 20 })(t);
    const ROW = '.tile[data-m="sampler"] .qchoice[data-q="samp.steps"]';
    await t.ok('a value between the choices checks none, marks its place and says it', () => { const r = document.querySelector(__c.ROW), m = r.querySelector('.q-mark'); return (!r.querySelector('[aria-checked="true"]') && !!m && !m.hidden && m.offsetLeft > 0 && /20 steps/.test(r.querySelector('.kl').textContent)) || r.outerHTML.slice(0, 200); });
    await t.eq('the Tab stop is the nearest choice', () => document.querySelector(`${__c.ROW} button[tabindex="0"]`).textContent.trim(), 'Standard');
    await t.focus(`${ROW} button[tabindex="0"]`); await t.press('ArrowRight');
    await t.eq('an arrow takes the next choice that way from the value', () => get('samp.steps'), 30);
    await t.do('back between', () => { S.vals['samp.steps'] = 20; render(); });
    await t.focus(`${ROW} button[tabindex="0"]`); await t.press('ArrowLeft');
    await t.eq('and the other way never back past it', () => get('samp.steps'), 12);
  } },
  { id: 'scrub-tokens-step-drag-and-type', run: async t => {
    await desk('lab', ['build'], { 'lyrics.text': '[Verse]\na\n[Chorus]\nb' }, 'yue')(t);
    const TAKES = '.tile[data-m="build"] [data-scrub="yue.btakes"]';
    await t.focus(TAKES); await t.press('ArrowRight');
    await t.eq('a token steps with the arrows', () => get('yue.btakes'), 3);
    await t.mouseDrag('.tile[data-m="build"] [data-scrub="yue.bcond"]', [[30, 0]]);
    await t.ok('and drags sideways', () => +get('yue.bcond') > 1);
    await t.focus(TAKES); await t.press('Enter');
    await t.ok('Enter turns it into a field with its number', () => document.activeElement.classList.contains('tok-type') && document.activeElement.value === '3');
    await t.press('Control+a'); await t.type('1'); await t.press('Enter');
    await t.ok('typed and Enter sets it, and focus is back on the token', () => (get('yue.btakes') === 1 && document.activeElement.matches(__c.TAKES)) || [get('yue.btakes'), document.activeElement.outerHTML.slice(0, 80)]);
    await t.press('Enter'); await t.press('Control+a'); await t.type('3'); await t.press('Escape');
    await t.eq('Escape leaves it', () => get('yue.btakes'), 1);
    await t.dblclick(TAKES);
    await t.eq('a double-click resets it', () => get('yue.btakes'), 2);
  } },
  { id: 'pill-and-how-adventurous', run: async t => {
    await desk('quick', ['planner'])(t);
    const PILL = '.tile[data-m="planner"] .pill-tog[data-ledtog="lm.on"]';
    await t.click(PILL);
    await t.ok('a pill flips its switch and says so', () => get('lm.on') === false && document.querySelector(__c.PILL).getAttribute('aria-pressed') === 'false' && !document.querySelector(__c.PILL).classList.contains('on'));
    await t.click(PILL);
    await t.ok('and back', () => get('lm.on') === true && document.querySelector(__c.PILL).classList.contains('on'));
    const MOOD = '.tile[data-m="planner"] .mood';
    await t.ok('How adventurous checks the mood that is the temperature', () => document.querySelector(`${__c.MOOD} [aria-checked="true"]`)?.textContent.trim() === 'Even');
    await t.focus(`${MOOD} [aria-checked="true"]`); await t.press('ArrowRight');
    await t.ok('an arrow moves and chooses', () => (get('lm.temp') === 1.05 && document.querySelector(`${__c.MOOD} [aria-checked="true"]`)?.textContent.trim() === 'Bold' && document.activeElement.textContent.trim() === 'Bold') || get('lm.temp'));
  } },
  { id: 'checked-text-keeps-or-refuses', run: async t => {
    await desk('lab', ['latent', 'sampler'], { 'samp.retake': 0.5 })(t);
    const TS = '.tile[data-m="latent"] [data-bind="lat.ts"]';
    await t.click(TS); await t.type('x');
    await t.ok('while it fails its check it is marked, and nothing is set', () => { const el = document.querySelector(__c.TS); return el.classList.contains('bad') && el.getAttribute('aria-invalid') === 'true' && get('lat.ts') === ''; });
    await t.press('Tab');
    await t.ok('a field that keeps what was typed keeps it, and says nothing is applied', () => { const el = document.querySelector(__c.TS), m = document.querySelector('[data-err="lat.ts"]'); return (el.value === 'x' && /Nothing is applied until it is valid/.test(m?.textContent || '') && get('lat.ts') === '') || [el.value, m?.textContent]; });
    await t.click(TS); await t.press('Control+a'); await t.type('0.9, 0.5, 0'); await t.press('Enter');
    await t.eq('a valid list is set on Enter', () => get('lat.ts'), '0.9, 0.5, 0');
    const RS = '.tile[data-m="sampler"] [data-bind="samp.rseed"]';
    await t.click(RS); await t.press('Control+a'); await t.type('x'); await t.press('Tab');
    await t.ok('a field that does not keep refuses it, puts back what is in force and says so', () => (document.querySelector(__c.RS).value === '-1' && get('samp.rseed') === '-1' && /Kept “-1”/.test(__c.toast())) || [document.querySelector(__c.RS).value, __c.toast()]);
  } },
  { id: 'side-unit-chip-selector', run: async t => {
    await desk('lab', ['post'], { 'pp.q.on': true }, 'ace', ['q'])(t);
    const CH = '.pedal[data-xid="q"] .chipsw[data-ctl="pp.q.eval"]';
    await t.ok('a short choice on a side unit is lit chips under its caps label, in short words', () => { const c = document.querySelector(__c.CH); return (!!c && c.querySelector('.kl').textContent === 'Evaluate' && [...c.querySelectorAll('button')].map(b => b.textContent).join() === 'Raw,Master,Both') || c?.outerHTML.slice(0, 200); });
    await t.click(`${CH} button[title="Mastered"]`);
    await t.ok('a chip chooses', () => { const on = [...document.querySelectorAll('.pedal[data-xid="q"] [data-seg="pp.q.eval"]')].filter(b => b.getAttribute('aria-pressed') === 'true'); return (get('pp.q.eval') === 'Mastered' && on.length === 1 && on[0].textContent === 'Master' && on[0].closest('[data-ctl]').classList.contains('changed')) || on.map(b => b.outerHTML); });
  } },
  { id: 'delete-resets-tokens-and-rulers', pending: { finding: 'S-18', right: 'Delete (or Backspace) on a focused knob, token, fader, ruler, CFG handle or look-back strip returns it to its default, announced ("Guidance back to 9.0"); the control\'s tip names it beside the double-click.' }, run: async t => {
    await desk('lab', ['build', 'tempo'], { 'lyrics.text': '[Verse]\na', 'tempo.dur': 300, 'tempo.dur.auto': false }, 'ace')(t);
    await t.focus(RULER); await t.press('Delete');
    await t.ok('Delete puts a ruler back on auto', () => document.querySelector(__c.RULER).classList.contains('auto'));
    await desk('lab', ['build'], { 'lyrics.text': '[Verse]\na', 'yue.btakes': 6 }, 'yue')(t);
    await t.focus('.tile[data-m="build"] [data-scrub="yue.btakes"]'); await t.press('Backspace');
    await t.eq('Backspace puts a token back to its default', () => get('yue.btakes'), 2);
  } },
];
