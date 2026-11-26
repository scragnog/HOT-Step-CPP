// Each module's own interactions, on the engines that use it.
const desk = (tier, mods, extra) => t => t.do(`${tier} desk`, ([tier, mods, extra]) => { S.tier = tier; S.mods = mods; S.pedals = []; if (extra?.vals) { Object.assign(S.vals, extra.vals); } Object.assign(S, { ...(extra || {}), vals: S.vals }); render(); }, [tier, mods, extra]);
export default [
  { id: 'tempo', run: async t => {
    await desk('lab', ['tempo'])(t);
    await t.click('.tile[data-m="tempo"] [data-tap]'); await t.do('wait', () => {}, null, 500); await t.click('.tile[data-m="tempo"] [data-tap]');
    await t.ok('two taps set the BPM from their interval', () => !isAuto('tempo.bpm') && +get('tempo.bpm') > 40);
    await t.click('.tile[data-m="tempo"] .wheel path[data-key="E minor"]');
    await t.eq('a wheel segment sets the key', () => [get('tempo.key'), document.querySelector('[data-keylcd]').textContent], ['E minor', 'E MINOR']);
    await t.focus('.tile[data-m="tempo"] .wheel [aria-checked="true"]');
    await t.press('ArrowRight'); await t.eq('arrows go round the ring', () => get('tempo.key'), 'B minor');
    await t.press('ArrowUp'); await t.eq('up goes to the major ring', () => get('tempo.key'), 'D major');
    await t.press('ArrowDown'); await t.press('ArrowDown'); await t.eq('down twice reaches Auto', () => get('tempo.key'), 'Auto');
    await t.press('ArrowLeft'); await t.eq('from Auto, left is F major', () => get('tempo.key'), 'F major');
    await t.ok('the focus ring follows on the wheel', () => document.querySelector('.wheel .wring').getAttribute('d') !== '');
    await t.press('Home'); await t.eq('Home is Auto', () => get('tempo.key'), 'Auto');
    await t.click('.tile[data-m="tempo"] [data-ledtog="tempo.beatio"]');
    await t.click('.tile[data-m="tempo"] [data-seg="tempo.beatbars"][data-v="8"]');
    await t.ok('beat I/O and its bars', () => get('tempo.beatio') === true && get('tempo.beatbars') === '8');
    const keyRow0 = await t.js(() => document.querySelector('.tile[data-m="tempo"] [data-keylcd]')?.getBoundingClientRect().top);
    await t.click('.tile[data-m="tempo"] [data-ledtog="tempo.trim"]');
    await t.ok('auto-trim brings its buffer and fade', () => !!document.querySelector('.lcd-trim:not(.held) [data-scrub="tempo.trimbuf"]'));
    await t.ok('and moves nothing on the LCD: the row was kept for it (L-27)', y => Math.abs(document.querySelector('.tile[data-m="tempo"] [data-keylcd]')?.getBoundingClientRect().top - y) < 1 || 'key row moved', keyRow0);
    await t.do('stable audio', () => { S.backend = 'sa3'; render(); });
    await t.click('.tile[data-m="tempo"] [data-capadd="120 BPM"]');
    await t.eq('on SA3 a snippet goes into the prompt', () => document.querySelector('#prompt').value, '120 BPM');
    await t.click('.tile[data-m="tempo"] [data-capadd="A minor"]');
    await t.eq('and the next one after a comma', () => document.querySelector('#prompt').value, '120 BPM, A minor');
  } },
  { id: 'seed-manager', run: async t => {
    await desk('studio', ['seed'])(t);
    await t.click('.tile[data-m="seed"] [data-dice]');
    await t.ok('Roll once sets and keeps a seed', () => Number.isInteger(S.seed) && get('seed.lock') === true);
    await t.click('.tile[data-m="seed"] [data-seedrand]');
    await t.eq('Random each time', () => [S.seed, get('seed.lock')], [null, false]);
    await t.click('.tile[data-m="seed"] [data-seededit]');
    await t.ok('the split-flap becomes a field', () => document.activeElement.matches('[data-seedin]'));
    await t.type('123abc'); await t.press('Enter');
    await t.ok('a bad seed is refused and the field stays', () => /A seed is a whole number/.test(document.querySelector('#toast .toast-msg').textContent) && document.activeElement.matches('[data-seedin]'));
    await t.do('clear field', () => { document.querySelector('[data-seedin]').value = ''; });
    await t.type('4242'); await t.press('Enter');
    await t.eq('Enter keeps a typed seed', () => [S.seed, get('seed.lock'), document.activeElement.matches('[data-seednum]')], [4242, true, true]);
    await t.click('.tile[data-m="seed"] [data-seededit]'); await t.type('99'); await t.press('Escape');
    await t.eq('Escape leaves it', () => S.seed, 4242);
    await t.click('.tile[data-m="seed"] [data-seededit]'); await t.type('77'); await t.click('.top .brand');
    await t.eq('blur commits (typing replaces the selected seed)', () => S.seed, 77);
    await t.click('.tile[data-m="seed"] [data-ledtog="seed.lock"]');
    await t.eq('the lock toggles keeping it', () => get('seed.lock'), false);
    await t.click('.tile[data-m="seed"] [data-star]');
    await t.ok('Save this seed opens the form', () => !!document.querySelector('.sm-save') && document.activeElement.matches('.sm-save input'));
    await t.press('Escape');
    await t.ok('Escape in the form cancels it', () => !document.querySelector('.sm-save'));
    await t.click('.tile[data-m="seed"] [data-star]');
    await t.do('fill', () => { document.querySelector('[data-smf="n"]').value = 'warm one'; document.querySelector('[data-smf="t"]').value = 'pads, keeper'; });
    await t.focus('.sm-save [data-smf="d"]'); await t.press('Enter');
    await t.ok('Enter in the form saves the seed', () => S.stars[0].v === 77 && S.stars[0].n === 'warm one' && S.stars[0].tags.join() === 'pads,keeper');
    await t.click('.tile[data-m="seed"] [data-star]'); await t.click('.sm-save [data-smsave]');
    await t.ok('saving it again updates it, with Undo', () => /Updated saved seed 77/.test(document.querySelector('#toast .toast-msg').textContent) && S.stars.filter(s => s.v === 77).length === 1);
    await t.click('.tile[data-m="seed"] [data-star]'); await t.click('.sm-save [data-smcancel]');
    await t.ok('Cancel closes the form', () => !document.querySelector('.sm-save'));
    await t.click('.tile[data-m="seed"] [data-smstar="77104"]');
    await t.eq('a star marks a saved seed', () => S.stars.find(s => s.v === 77104).star, true);
    await t.click('.tile[data-m="seed"] [data-loadseed="1834920213"]');
    await t.eq('a saved seed loads', () => [S.seed, get('seed.lock')], [1834920213, true]);
    await t.click('.tile[data-m="seed"] [data-smq]'); await t.type('airy');
    await t.eq('search filters the list', () => document.querySelectorAll('[data-stars] .sm-row').length, 1);
    await t.do('clear search', () => { const q = document.querySelector('[data-smq]'); q.value = ''; q.dispatchEvent(new Event('input', { bubbles: true })); });
    /* by keyboard: at the centre of the × a mouse lands on the row's seed number, which overflows onto it */
    await t.focus('.tile[data-m="seed"] [data-smdel="77104"]'); await t.press('Enter');
    await t.ok('delete asks first', () => S.stars.some(s => s.v === 77104) && document.querySelector('[data-smdel="77104"]').classList.contains('armed'));
    await t.press('Enter');
    await t.ok('then deletes, with Undo', () => !S.stars.some(s => s.v === 77104) && document.querySelector('#toast .toast-act')?.textContent === 'Undo');
    await t.click('#toast .toast-act');
    await t.ok('Undo restores it', () => S.stars.some(s => s.v === 77104));
    await t.click('.tile[data-m="seed"] [data-smrand]');
    await t.ok('From saved loads one of them', () => S.stars.some(s => s.v === S.seed) && /^Loaded (“.+” \(\d+\)|saved seed \d+)$/.test(document.querySelector('#toast .toast-msg').textContent));
    /* LM seed follows stands on the Seed tile while the planner is on the desk (design decision 31) */
    await t.do('lab, with the planner', () => { S.tier = 'lab'; S.mods = ['seed', 'planner']; render(); });
    await t.click('.tile[data-m="seed"] [data-ledtog="lm.seedlock"]');
    await t.ok('in Lab the LM seed link toggles', () => get('lm.seedlock') === false && /LM seed: random/.test(document.querySelector('[data-lmseed]').textContent));
  } },
  { id: 'takes', run: async t => {
    await desk('studio', ['takes'])(t);
    await t.click('.tile[data-m="takes"] [data-take="4"]');
    await t.eq('a reel sets the takes', () => [get('takes.n'), document.querySelector('.tk-lcd b').textContent], [4, '4']);
    await t.hover('.tile[data-m="takes"] [data-take="7"]');
    await t.ok('pointing at a reel previews its count', () => document.querySelector('.tk-lcd b').textContent === '7' && document.querySelector('.tk-lcd').classList.contains('pv'));
    await t.hover('.top .brand');
    await t.eq('and leaving puts it back', () => document.querySelector('.tk-lcd b').textContent, '4');
    await t.do('mm3 with too many', () => { S.backend = 'mm3'; S.vals['model.lm'] = '8B planner · F16'; S.vals['takes.n'] = 4; render(); });
    await t.ok('a run that will not fit is said under Generate', () => !!document.querySelector('#meter [data-takesfit]') && document.querySelector('#go').classList.contains('wontfit'));
    await t.click('#meter [data-takesfit]');
    await t.ok('Use N takes fixes it', () => +get('takes.n') < 4 && !document.querySelector('#go').classList.contains('wontfit'));
  } },
  { id: 'continue-and-repaint', run: async t => {
    await desk('lab', ['cont'], { backend: 'yue' })(t);
    await t.file('.tile[data-m="cont"] [data-file="cont"]', 'demo.wav', 'audio/wav');
    await t.ok('a recording loads with a selection', () => S.contFile === 'demo.wav' && !!document.querySelector('[data-contwave]'));
    await t.focus('[data-contwave]'); await t.press('ArrowRight'); await t.press('Shift+ArrowRight');
    await t.eq('arrows move the selection, Shift resizes it', () => S.contSel, [1, 22]);
    await t.mouseDrag('[data-contwave]', [[30, 0]], [0.6, 0.5]);
    await t.ok('a drag outside the selection draws a new one', () => S.contSel[0] > 22 || S.contSel[1] > 22);
    await t.do('ace with songs', () => { S.backend = 'ace'; S.mods = []; S.tier = 'studio'; render(); });
    await t.click('#latest .song[data-id="1"] [data-songmenu]');
    await t.ok('a song menu opens, focus on its first item', () => !document.querySelector('#menu').hidden && document.activeElement.dataset.sa === 'repaint');
    await t.press('ArrowDown'); await t.eq('arrows move in the menu', () => document.activeElement.dataset.sa, 'reuse');
    await t.press('End'); await t.eq('End', () => document.activeElement.dataset.sa, 'MIDI Studio');
    await t.press('Home'); await t.press('Enter');
    await t.ok('Repaint a region puts Repaint on the desk with that song', () => S.mods[0] === 'repaint' && S.source?.song === 1 && document.querySelector('#goLabel').textContent === 'Repaint region');
    await t.focus('[data-rpwave]'); await t.press('ArrowRight'); await t.press('Shift+ArrowRight');
    await t.eq('arrows move the region, Shift moves its end', () => S.region.map(x => +x.toFixed(2)), [0.36, 0.57]);
    await t.mouseDrag('[data-rpwave]', [[50, 0]], [0.1, 0.5]);
    await t.ok('dragging draws a new region', () => S.region[0] < 0.2);
    await t.click('.tile[data-m="repaint"] [data-remove="repaint"]');
    await t.eq('removing Repaint drops the source', () => [S.source, document.querySelector('#goLabel').textContent], [null, 'Generate']);
  } },
  { id: 'pedalboard', run: async t => {
    await t.do('studio board', () => { S.tier = 'studio'; S.mods = []; S.pedals = []; render(); addItem('pd:ss'); addItem('pd:norm'); addItem('ex:wh'); addItem('ex:art'); });
    await t.ok('pedals arrive switched on', () => S.mods.includes('post') && get('pp.ss.on') && get('pp.norm.on') && get('pp.wh.on'));
    await t.click('.pedal[data-pid="ss"] [data-foot="pp.ss.on"]');
    await t.ok('a foot switch bypasses its pedal', () => get('pp.ss.on') === false && document.querySelector('.pedal[data-pid="ss"]').classList.contains('byp'));
    await t.click('.tile[data-m="post"] .master[data-ledtog="pp.on"]');
    await t.ok('the chain switch bypasses the board, side units after it wait', () => get('pp.on') === false && /chain bypassed/i.test(document.querySelector('[data-ppcount]').textContent) && !!document.querySelector('.xunit[data-xid="art"] .u-wait'));
    await t.click('.tile[data-m="post"] .master[data-ledtog="pp.on"]');
    await t.click('.xunit[data-xid="wh"] .u-pow');
    await t.eq('a side unit powers off', () => get('pp.wh.on'), false);
    await t.do('lab', () => { S.tier = 'lab'; render(); });
    await t.focus('.pedal[data-pid="norm"] .knob[data-knob="pp.norm.ceil"]'); await t.press('ArrowDown');
    await t.eq('a pedal knob turns', () => get('pp.norm.ceil'), -1.1);
    await t.do('re-add', () => addItem('pd:ss'), null, 50);
    await t.ok('adding a pedal already there rings it', () => document.querySelector('.pedal[data-pid="ss"]').classList.contains('here'));
    await t.click('.pedal[data-pid="ss"] [data-pedalx="ss"]');
    await t.ok('× takes a pedal off, with Undo', () => !S.pedals.includes('ss') && /Took StableStep off the board/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.click('#toast .toast-act');
    await t.ok('Undo puts it back', () => S.pedals.includes('ss'));
    await t.do('one pedal', () => { S.pedals = ['norm']; render(); });
    await t.click('.pedal[data-pid="norm"] [data-pedalx="norm"]');
    await t.ok('the last pedal takes the board with it', () => !S.mods.includes('post') && /the Pedalboard left the desk too/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.do('art', () => { S.pedals = ['art']; S.vals['pp.art.on'] = true; render(); });
    await t.click('#songInfoBtn');
    await t.ok('Cover art adds its subject to Song info', () => !document.querySelector('#songInfo .si-art').hidden);
  } },
];
