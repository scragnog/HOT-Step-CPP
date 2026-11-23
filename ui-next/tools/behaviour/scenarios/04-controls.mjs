// The shared controls, each by pointer and by keyboard: knobs, LED and pill toggles, segments, auto, number boxes,
// rulers, faders, XY pads, scrub tokens, interval handles, the codes window, every dropdown, cycle buttons, typed checks.
const desk = (tier, mods, extra) => t => t.do(`${tier} desk`, ([tier, mods, extra]) => { S.tier = tier; S.mods = mods; S.pedals = []; Object.assign(S, extra || {}); render(); }, [tier, mods, extra]);
export default [
  { id: 'knobs', run: async t => {
    await desk('studio', ['sampler'])(t);
    await t.mouseDrag('.tile[data-m="sampler"] .knob[data-knob="samp.cfg"]', [[0, -20], [0, -20]]);
    await t.ok('dragging a knob up raises it', () => +get('samp.cfg') > 9 && document.querySelector('[data-knob="samp.cfg"]').closest('.kn').classList.contains('changed'));
    await t.focus('.tile[data-m="sampler"] .knob[data-knob="samp.cfg"]');
    await t.press('Home'); await t.eq('Home goes to the minimum', () => get('samp.cfg'), 0);
    await t.press('End'); await t.eq('End to the maximum', () => get('samp.cfg'), 20);
    await t.press('ArrowDown'); await t.eq('an arrow steps', () => get('samp.cfg'), 19.9);
    await t.press('Shift+ArrowDown'); await t.eq('Shift steps ten', () => get('samp.cfg'), 18.9);
    await t.press('PageDown'); await t.eq('PageDown steps ten', () => get('samp.cfg'), 17.9);
    await t.ok('the read-out follows', () => document.querySelector('[data-out="samp.cfg"]').textContent === '17.9');
    await t.dblclick('.tile[data-m="sampler"] .knob[data-knob="samp.cfg"]');
    await t.ok('double-click resets it', () => !('samp.cfg' in S.vals) && +get('samp.cfg') === 9);
    await t.focus('.tile[data-m="sampler"] .knob[data-knob="samp.shift"]'); await t.press('ArrowUp');
    await t.eq('a knob with auto leaves auto when turned', () => [S.vals['samp.shift.auto'], S.vals['samp.shift']], [false, 3.1]);
  } },
  { id: 'toggles-segments-auto', run: async t => {
    await desk('lab', ['tempo', 'planner', 'sampler', 'seed'])(t);
    await t.click('.tile[data-m="planner"] [data-ledtog="lm.on"]');
    await t.eq('an LED toggle flips its setting', () => get('lm.on'), false);
    await t.ok('the planner terminal says so', () => /planner is off/.test(document.querySelector('[data-term]').textContent));
    await t.click('.tile[data-m="planner"] [data-ledtog="lm.on"]');
    await t.click('.tile[data-m="planner"] [data-seg="lm.codesmode"][data-v="Steps"]');
    await t.ok('a segment sets its value and re-faces the tile', () => get('lm.codesmode') === 'Steps' && !!document.querySelector('.tile[data-m="planner"] [data-scrub="lm.codesteps"]'));
    await t.click('.tile[data-m="tempo"] [data-auto="tempo.bpm"]');
    await t.eq('AUTO turns auto off', () => [isAuto('tempo.bpm'), document.querySelector('[data-num="tempo.bpm"]').value], [false, '120']);
    await t.click('.tile[data-m="tempo"] [data-auto="tempo.bpm"]');
    await t.eq('and back on', () => [isAuto('tempo.bpm'), document.querySelector('[data-num="tempo.bpm"]').value], [true, '']);
    await t.click('.tile[data-m="tempo"] [data-auto="tempo.dur"]');
    await t.ok('the length ruler leaves auto', () => !isAuto('tempo.dur') && !document.querySelector('[data-ruler="tempo.dur"]').classList.contains('auto'));
    await t.do('a range slider and a checkbox', () => { document.body.insertAdjacentHTML('beforeend', `<div id="tmpctl">${ctlHTML('lm.temp')}${ctlHTML('lm.cot')}${ctlHTML('lm.neg')}${ctlHTML('lm.repmode')}${ctlHTML('avoid.text')}</div>`); });
    await t.do('slider input', () => { const r = document.querySelector('#tmpctl input[type=range]'); r.value = 1.25; r.dispatchEvent(new Event('input', { bubbles: true })); });
    await t.eq('a range input sets its key and its read-out', () => [get('lm.temp'), document.querySelector('#tmpctl [data-out="lm.temp"]').textContent], [1.25, '1.25']);
    await t.click('#tmpctl [data-bind="lm.cot"]');
    await t.eq('a checkbox toggle', () => get('lm.cot'), false);
    await t.click('#tmpctl [data-seg="lm.repmode"][data-v="DRY"]');
    await t.eq('a segment from ctlHTML', () => get('lm.repmode'), 'DRY');
    await t.do('remove', () => document.querySelector('#tmpctl').remove());
  } },
  { id: 'number-boxes', run: async t => {
    await desk('studio', ['tempo', 'sampler'])(t);
    await t.click('.tile[data-m="tempo"] [data-num="tempo.bpm"]');
    await t.type('96'); await t.press('Enter');
    await t.eq('typing a BPM and Enter sets it', () => [get('tempo.bpm'), isAuto('tempo.bpm')], [96, false]);
    await t.focus('.tile[data-m="tempo"] [data-num="tempo.bpm"]'); await t.press('ArrowUp');
    await t.eq('ArrowUp steps', () => get('tempo.bpm'), 97);
    await t.press('PageUp'); await t.eq('PageUp steps ten', () => get('tempo.bpm'), 107);
    await t.press('Shift+ArrowDown'); await t.eq('Shift steps ten', () => get('tempo.bpm'), 97);
    await t.do('type a word', () => { const el = document.querySelector('[data-num="tempo.bpm"]'); el.focus(); el.value = 'fast'; });
    await t.press('Enter');
    await t.ok('a word is refused and put back', () => /Type a number for BPM/.test(document.querySelector('#toast .toast-msg').textContent) && document.querySelector('[data-num="tempo.bpm"]').value === '97');
    await t.do('type too much', () => { const el = document.querySelector('[data-num="tempo.bpm"]'); el.focus(); el.value = '500'; });
    await t.press('Enter');
    await t.ok('out of range is clamped, with a toast', () => get('tempo.bpm') === 240 && /BPM: 1 to 240\. Set to 240\./.test(document.querySelector('#toast .toast-msg').textContent));
    await t.do('type nothing', () => { const el = document.querySelector('[data-num="tempo.bpm"]'); el.focus(); el.value = ''; });
    await t.press('Tab');
    await t.eq('empty on blur is auto again', () => isAuto('tempo.bpm'), true);
    await t.do('type and Escape', () => { const el = document.querySelector('[data-num="samp.steps"]'); el.focus(); el.value = '77'; });
    await t.press('Escape');
    await t.eq('Escape puts the old value back', () => [document.querySelector('[data-num="samp.steps"]').value, get('samp.steps')], ['12', 12]);
    await t.mouseDrag('.tile[data-m="sampler"] [data-num="samp.steps"]', [[0, -10], [0, -20]]);
    await t.ok('dragging a number box up scrubs it', () => +get('samp.steps') > 12 && !document.body.classList.contains('scrubbing'));
  } },
  { id: 'ruler-fader-scrub', run: async t => {
    await desk('lab', ['tempo', 'adapters', 'build', 'timbre'], { backend: 'ace', slots: [{ file: 'tape-lofi-v2', type: 'LoRA' }] })(t);
    await t.do('duration manual', () => { S.vals['tempo.dur.auto'] = false; render(); });
    await t.mouseDrag('.tile[data-m="tempo"] [data-ruler="tempo.dur"]', [[40, 0]], [0.5, 0.5]);
    await t.ok('the ruler follows the pointer', () => +get('tempo.dur') > 300 && /\d:\d\d/.test(document.querySelector('[data-rcap]').textContent));
    await t.focus('.tile[data-m="tempo"] [data-ruler="tempo.dur"]'); await t.press('End');
    await t.eq('End on the ruler', () => get('tempo.dur'), 600);
    await t.mouseDrag('.tile[data-m="adapters"] .cart [data-fader="s0"]', [[0, 30]], [0.5, 0.3]);
    await t.ok('a fader follows the pointer', () => +get('s0') !== 1 && +get('s0') >= 0 && 's0' in S.vals);
    await t.focus('.tile[data-m="adapters"] .cart [data-fader="s0"]'); await t.press('Home');
    await t.ok('Home on a fader; the cartridge art stills', () => get('s0') === 0 && document.querySelector('[data-adart="s0"]').classList.contains('still'));
    await t.mouseDrag('.tile[data-m="timbre"] [data-scrub="timbre.str"]', [[-60, 0]]);
    await t.ok('the reference blend scrubs sideways', () => +get('timbre.str') < 1 && /% reference/.test(document.querySelector('[data-venn]').getAttribute('aria-valuetext')));
    await t.do('yue build', () => { S.backend = 'yue'; S.vals['lyrics.text'] = '[Verse]\na\n[Chorus]\nb'; render(); });
    await t.focus('.tile[data-m="build"] [data-scrub="yue.btakes"]'); await t.press('ArrowRight');
    await t.eq('a scrub token steps with the arrows', () => get('yue.btakes'), 3);
    await t.mouseDrag('.tile[data-m="build"] [data-scrub="yue.bcond"]', [[30, 0]]);
    await t.ok('and drags', () => +get('yue.bcond') > 1);
  } },
  { id: 'xy-ivl-codes', run: async t => {
    await desk('lab', ['planner', 'sampler', 'repetition', 'perf', 'latent'])(t);
    await t.mouseDrag('.tile[data-m="planner"] .xy', [[30, -20]], [0.3, 0.6]);
    await t.ok('dragging the XY pad sets both settings', () => 'lm.temp' in S.vals && 'lm.topp' in S.vals);
    await t.focus('.tile[data-m="planner"] .xy'); await t.do('reset first', () => { delete S.vals['lm.temp']; delete S.vals['lm.topp']; sync('lm.temp'); });
    await t.press('ArrowRight'); await t.press('Shift+ArrowUp');
    await t.eq('arrows nudge X and Y', () => [get('lm.temp'), get('lm.topp')], [0.85, 1]);
    await t.dblclick('.tile[data-m="planner"] .xy');
    await t.eq('double-click resets both', () => ['lm.temp' in S.vals, 'lm.topp' in S.vals], [false, false]);
    await t.mouseDrag('.tile[data-m="sampler"] [data-ivl="samp.cs"]', [[60, 0]]);
    await t.ok('the CFG interval start handle moves', () => +get('samp.cs') > 0 && /guidance from/.test(document.querySelector('[data-scoperead]').textContent));
    await t.focus('.tile[data-m="sampler"] [data-ivl="samp.ce"]'); await t.press('Home');
    /* A-31: the handles keep one step apart, so both stay grabbable, and an empty window says guidance is off */
    await t.ok('the end handle cannot pass the start, and stays a step after it', () => Math.abs(+get('samp.ce') - +get('samp.cs') - 0.01) < 1e-9);
    await t.do('an empty window', () => { S.vals['samp.cs'] = 0.5; S.vals['samp.ce'] = 0.51; sync('samp.ce'); });
    await t.ok('a window that holds no step reads "guidance off"', () => /guidance off/.test(document.querySelector('[data-scoperead]').textContent));
    await t.mouseDrag('.tile[data-m="repetition"] [data-codes]', [[-40, 0]], [0.7, 0.5]);
    await t.ok('the look-back window drags', () => +get('lm.win') !== 64);
    await t.focus('.tile[data-m="repetition"] [data-codes]'); await t.press('End');
    await t.eq('End on the window', () => get('lm.win'), 256);
    await t.focus('.tile[data-m="perf"] .xy'); await t.press('ArrowLeft');
    await t.ok('the performance pad changes the speed gauge', () => document.querySelector('[data-speed]').textContent !== '±0%');
    await t.mouseDrag('.tile[data-m="latent"] .xy', [[20, 10]], [0.5, 0.5]);
    await t.ok('the latent pad redraws its curve', () => 'lat.shift' in S.vals && document.querySelector('[data-dist] polyline:last-of-type') !== null);
  } },
  { id: 'dropdowns', run: async t => {
    await desk('lab', ['sampler', 'tempo'])(t);
    await t.click('.tile[data-m="sampler"] [data-pick="samp.solver"]');
    await t.ok('a plugin picker opens with its periodic table', () => !document.querySelector('#dd').hidden && document.querySelector('#dd').classList.contains('pt') && document.activeElement.matches('#dd [aria-selected="true"]'));
    await t.press('ArrowRight');
    await t.ok('arrows move and show the option', () => document.activeElement.dataset.opt === 'Heun' && /Heun/.test(document.querySelector('#ptBig').textContent));
    await t.press('d');
    await t.ok('typing filters, into the filter field', () => document.activeElement.id === 'ddq' && document.querySelector('#ddq').value === 'd');
    await t.type('pm++ 3m'); await t.press('Enter');
    await t.eq('Enter in the filter takes the first match', () => [get('samp.solver'), document.querySelector('#dd').hidden], ['DPM++ 3M', true]);
    await t.ok('focus goes back to its button', () => document.activeElement.dataset.pick === 'samp.solver');
    await t.press('ArrowDown');
    await t.ok('ArrowDown on a picker button opens it', () => !document.querySelector('#dd').hidden);
    await t.press('Escape');
    await t.ok('Escape closes it and refocuses', () => document.querySelector('#dd').hidden && document.activeElement.dataset.pick === 'samp.solver');
    await t.click('.tile[data-m="sampler"] [data-pick="samp.sched"]');
    await t.ok('the schedule picker is a grid of curves', () => document.querySelector('#dd').classList.contains('schedw') && document.querySelectorAll('#dd .opt').length === 17);
    await t.click('#ddq'); await t.type('zzz');
    await t.ok('a filter with no match says so', () => !document.querySelector('#dd .dd-none').hidden);
    await t.press('Escape');
    await t.click('.tile[data-m="sampler"] [data-pick="samp.sched"]');
    await t.press('End'); await t.press('Home');
    await t.ok('Home and End', () => document.activeElement.dataset.opt === 'Linear');
    await t.press('Tab');
    await t.ok('Tab closes it and carries on from its button', () => document.querySelector('#dd').hidden && !document.querySelector('#dd').contains(document.activeElement) && document.activeElement !== document.body);
    await t.click('.tile[data-m="sampler"] [data-pick="samp.sched"]');
    await t.click('#dd [data-opt="Power"]');
    await t.ok('a click chooses; Power brings its exponent knob', () => get('samp.sched') === 'Power' && !!document.querySelector('[data-knob="sch.exp"]'));
    await t.click('.tile[data-m="sampler"] [data-pick="samp.sched"]'); await t.click('#dd [data-opt="Beta (Custom)"]');
    await t.ok('Beta brings α and β', () => !!document.querySelector('[data-knob="sch.alpha"]') && !!document.querySelector('[data-knob="sch.beta"]'));
    await t.click('.tile[data-m="sampler"] [data-pick="samp.sched"]'); await t.click('#dd [data-opt="Composite (2-Stage)"]');
    await t.ok('Composite brings its stages', () => !!document.querySelector('[data-pick="sch.a"]') && !!document.querySelector('[data-knob="sch.split"]'));
    await t.click('.tile[data-m="sampler"] [data-pick="sch.a"]');
    await t.ok('an LCD picker', () => document.querySelector('#dd').classList.contains('dd-lcd'));
    await t.click('.tile[data-m="sampler"] [data-pick="sch.a"]');
    await t.ok('clicking its button again closes it', () => document.querySelector('#dd').hidden);
    await t.click('.tile[data-m="tempo"] [data-cycle="tempo.sig|1"]');
    await t.eq('the ▸ beside an LCD picker cycles forward', () => get('tempo.sig'), '4/4');
    await t.click('.tile[data-m="tempo"] [data-cycle="tempo.sig|-1"]'); await t.click('.tile[data-m="tempo"] [data-cycle="tempo.sig|-1"]');
    await t.eq('and ◂ back, wrapping', () => get('tempo.sig'), '2/4');
    await t.click('#engBtn'); await t.click('#engPanel [data-pick="model.dit"]');
    await t.ok('a model picker groups quantisations as chips', () => document.querySelector('#dd').classList.contains('dd-model') && document.querySelectorAll('#dd .opt-mrow').length > 0);
    await t.click('#dd [data-opt="Turbo · Q4_K_M"]');
    await t.ok('choosing a model updates the VRAM read-out', () => get('model.dit') === 'Turbo · Q4_K_M' && /Turbo/.test(document.querySelector('#logEngine').textContent));
    await t.click('.top .brand');
  } },
  { id: 'language-stamp', run: async t => {
    await desk('studio', [])(t);
    await t.click('#bento .tile[data-m="lyrics"] [data-pick="lyrics.lang"]');
    await t.ok('the language stamp opens its list', () => document.querySelector('#dd').classList.contains('dd-stamp') && !document.querySelector('.stamps-rec'));
    await t.click('#dd [data-opt="French"]');
    await t.eq('a choice is remembered', () => [get('lyrics.lang'), JSON.parse(localStorage.getItem('hs3-lang-recent'))], ['French', ['French']]);
    await t.click('#bento .tile[data-m="lyrics"] [data-pick="lyrics.lang"]');
    await t.click('#dd [data-opt="German"]');
    await t.click('#bento .tile[data-m="lyrics"] [data-pick="lyrics.lang"]');
    await t.eq('recent languages come first', () => [...document.querySelectorAll('#dd .stamps-rec [data-opt]')].map(b => b.dataset.opt), ['French']);
    await t.press('Escape');
  } },
  { id: 'typed-checks-examples', run: async t => {
    await desk('lab', ['planner', 'latent', 'sampler'], { backend: 'mm3' })(t);
    await t.click('.tile[data-m="planner"] [data-bind="mm.seed"]'); await t.type('12'); await t.press('Tab');
    await t.eq('a valid seed commits when you leave the field', () => get('mm.seed'), '12');
    await t.click('.tile[data-m="planner"] [data-bind="mm.seed"]'); await t.press('End'); await t.type('x');
    /* A-08: while it is wrong it is marked, and nothing is committed */
    await t.ok('a wrong planner seed is marked as you type, and not committed', () => { const el = document.querySelector('[data-bind="mm.seed"]'); return el.classList.contains('bad') && el.getAttribute('aria-invalid') === 'true' && get('mm.seed') === '12'; });
    await t.press('Tab');
    /* changed on purpose (E-33): the planner seed is a kept field: what was typed stays, marked, and the line under it says what is in force */
    await t.ok('on change what was typed stays, marked, and the seed in force is said with a way back', () => { const el = document.querySelector('[data-bind="mm.seed"]'), m = document.querySelector('[data-err="mm.seed"]'); return (el.value === '12x' && el.classList.contains('bad') && get('mm.seed') === '12' && /Still using seed 12/.test(m.textContent) && !!m.querySelector('[data-putback]')) || `${el.value} | ${m?.textContent}`; });
    await t.do('ace', () => { S.backend = 'ace'; render(); });
    await t.click('.tile[data-m="latent"] [data-example="lat.ts"]');
    await t.eq('Insert example fills custom timesteps', () => get('lat.ts'), '0.97, 0.76, 0.5, 0.28, 0.085, 0');
    await t.ok('and the scope draws them', () => document.querySelectorAll('[data-scope] .sdot, .ivl-wrap .sdot').length === 6);
    await t.click('.tile[data-m="latent"] [data-seg="dcw.mode"][data-v="Low-Frequency"]');
    await t.ok('DCW mode relabels the low knob', () => /Scaler|Low/.test(document.querySelector('[data-knob="dcw.low"]').closest('.kn').textContent) && !document.querySelector('[data-knob="dcw.high"]'));
    await t.click('.tile[data-m="latent"] [data-ledtog="den.on"]');
    await t.ok('a group power toggle lights its group', () => get('den.on') === true && !document.querySelector('[data-ledtog="den.on"]').closest('.lat-g').classList.contains('off'));
  } },
];
