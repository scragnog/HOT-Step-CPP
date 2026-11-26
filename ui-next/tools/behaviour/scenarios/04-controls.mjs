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
];
