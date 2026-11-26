// Round 3, the other four engines, switching, sizes and input modes (review B): what an engine keeps and what the song
// keeps across a switch, ranges and markers that follow the engine, hit areas that never take a neighbour's click, and
// the smaller fixes that changed behaviour.
const desk = (bk, tier, mods, extra) => t => t.do(`${bk} ${tier} desk`, ([bk, tier, mods, extra]) => { S.backend = bk; S.tier = tier; S.mods = mods; S.pedals = extra?.pedals || []; if (extra?.vals) Object.assign(S.vals, extra.vals); if (extra?.slots) S.slots = extra.slots; render(); }, [bk, tier, mods, extra]);
const sw = async (t, bk) => { await t.click('#engBtn'); await t.click(`#engPanel [data-backend="${bk}"]`); await t.press('Escape'); };
export default [
  { id: 'engine-switch-there-and-back-loses-nothing', run: async t => {
    await desk('ace', 'lab', ['sampler', 'tempo', 'adapters', 'takes'], { vals: { 'samp.steps': 300, 'samp.cfg': 20, 'tempo.dur': 600, 'tempo.dur.auto': false, 'takes.n': 8, 'model.dit': 'XL Turbo · MXFP4', 'ad.adv': 'Advanced', 'lyrics.text': '[Verse]\nhello' }, slots: [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'folk-voice', type: 'LoKr' }] })(t);
    await t.do('a description', () => { document.querySelector('#prompt').value = 'my song'; });
    await sw(t, 'mm3');
    await t.eq('values out of the new engine are its own, in range (B-01)', () => [+get('samp.steps') < 300, +get('samp.cfg') < 20, get('takes.n')], [true, true, 4]);
    await t.eq('the other engine\'s adapters are not loaded here (B-02)', () => S.slots.length, 0);
    await t.ok('the toast says what stayed behind, with Undo (B-48)', () => /ACE-Step 1\.5 keeps its models? \(.*\), 2 adapters and \d+ settings?\./.test(document.querySelector('#toast .toast-msg')?.textContent || '') && document.querySelector('#toast .toast-act')?.textContent === 'Undo');
    await t.eq('the song travels: description and lyrics', () => [document.querySelector('#prompt').value, get('lyrics.text')], ['my song', '[Verse]\nhello']);
    await t.do('a model choice on this engine', () => setVal('model.dit', 'Flow DiT · F16'));
    await sw(t, 'hm');
    await t.eq('a length past the engine reads at its limit (B-01)', () => [get('tempo.dur'), S.eng.hold['tempo.dur'].v], [240, 600]);
    await sw(t, 'mm3');
    await t.eq('each engine keeps its own model choice (B-19)', () => get('model.dit'), 'Flow DiT · F16');
    await sw(t, 'ace');
    await t.eq('back on the first engine, its desk is as it was', () => [get('samp.steps'), get('samp.cfg'), get('tempo.dur'), get('takes.n'), get('model.dit'), S.slots.map(s => s.file).join()], [300, 20, 600, 8, 'XL Turbo · MXFP4', 'tape-lofi-v2,folk-voice']);
    await t.do('let autosave run', () => {}, null, 600);
    await t.reload();
    await t.eq('and after a reload', () => [S.backend, get('samp.steps'), S.slots.length, S.eng.stash.mm3?.vals['model.dit']], ['ace', 300, 2, 'Flow DiT · F16']);
    await sw(t, 'sa3');
    await t.press('Control+z');
    await t.eq('a switch is one undo step (B-48)', () => [S.backend, get('samp.steps'), get('takes.n'), S.slots.length], ['ace', 300, 8, 2]);
  } },
  { id: 'ruler-markers-follow-the-engine', run: async t => {
    await desk('sa3', 'lab', ['tempo'], { vals: { 'tempo.beatio': false } })(t);
    await t.ok('no Beat I/O marker while it is off (B-04)', () => [...document.querySelectorAll('.tile[data-m="tempo"] [data-io]')].every(x => x.hidden));
    await t.click('.tile[data-m="tempo"] [data-ledtog="tempo.beatio"]');
    await t.ok('on Stable Audio 3, the out marker ends at the 30 s auto length', () => { const r = document.querySelector('[data-io="r"]'); return !r.hidden && Math.abs(parseFloat(r.style.left) + parseFloat(r.style.width) - 5) < 0.01; });
    await desk('hm', 'lab', ['tempo'])(t);
    await t.eq('HeartMuLa has no Beat I/O or trim markers', () => document.querySelectorAll('.tile[data-m="tempo"] [data-io], .tile[data-m="tempo"] [data-trim]').length, 0);
    await t.ok('its auto length spans the whole ruler, "up to 4:00" (B-32)', () => document.querySelector('[data-rcap]').textContent === 'up to 4:00' && document.querySelector('.tile[data-m="tempo"] h3, .tile[data-m="tempo"] .tile-head').textContent.includes('Length'));
  } },
  { id: 'hit-areas-go-to-the-nearer-control', run: async t => {
    await desk('ace', 'lab', ['tempo'])(t);
    /* a point in the 1px gap, nearer the picker than the arrow, that only the arrow's pad covers */
    await t.click('.tile[data-m="tempo"] [data-cycle="tempo.sig|1"]', null, [-0.04, 0.5]);
    await t.ok('a click in the arrow\'s pad but nearer the picker opens the picker (left over from A-12)', () => !document.querySelector('#dd').hidden && get('tempo.sig') === 'Auto');
    await t.press('Escape');
    await t.click('.tile[data-m="tempo"] [data-cycle="tempo.sig|1"]');
    await t.ok('a click on the arrow itself steps it', () => get('tempo.sig') !== 'Auto' && document.querySelector('#dd').hidden);
  } },
  { id: 'number-box-tab-changes-nothing', run: async t => {
    await desk('mm3', 'lab', ['sampler'])(t);
    await t.do('a loaded revision, unmodified', () => { S.revision = 'x'; S.revDirty = false; });
    await t.focus('.tile[data-m="sampler"] [data-num="samp.steps"]');
    await t.press('Tab');
    await t.eq('passing through the box stores nothing and marks nothing (B-27)', () => ['samp.steps' in S.vals, S.revDirty], [false, false]);
  } },
  { id: 'prompt-chips-and-ranges', run: async t => {
    await desk('sa3', 'studio', ['tempo'])(t);
    await t.do('a prompt', () => { document.querySelector('#prompt').value = 'ambient pads'; });
    await t.click('[data-capadd="120 BPM"]'); await t.click('[data-capadd="90 BPM"]'); await t.click('[data-capadd="D major"]');
    await t.eq('a tempo chip replaces the other tempo (B-13)', () => document.querySelector('#prompt').value, 'ambient pads, 90 BPM, D major');
    await t.ok('and shows pressed', () => document.querySelector('[data-capadd="90 BPM"]').getAttribute('aria-pressed') === 'true' && document.querySelector('[data-capadd="120 BPM"]').getAttribute('aria-pressed') === 'false');
    await t.click('[data-capadd="D major"]');
    await t.eq('pressing it again takes it out', () => document.querySelector('#prompt').value, 'ambient pads, 90 BPM');
    await desk('yue', 'lab', ['adhere'])(t);
    await t.do('drag the ends past each other', () => { setVal('yue.pcfg1', 0.9); setVal('yue.pcfg2', 0.2); setVal('yue.cads1', 1); });
    await t.eq('a range never inverts (B-15)', () => [get('yue.pcfg1') < get('yue.pcfg2'), get('yue.cads1') < get('yue.cads2')], [true, true]);
  } },
  { id: 'grip-menu-moves-a-tile', run: async t => {
    await desk('ace', 'studio', ['tempo', 'avoid', 'seed'])(t);
    await t.click('.tile[data-m="seed"] .grip');
    await t.ok('the grip offers the moves (B-23)', () => !document.querySelector('#pop').hidden && document.querySelectorAll('#pop [data-mto]').length === 4);
    await t.click('#pop [data-mto="top"]');
    await t.eq('Move to the top', () => S.mods.join(), 'seed,tempo,avoid');
    await t.press('Control+z');
    await t.eq('and it is undoable', () => S.mods.join(), 'tempo,avoid,seed');
  } },
  { id: 'parked-pedals-and-units', run: async t => {
    await desk('hm', 'studio', ['post'], { pedals: ['tile', 'norm', 'lrc'] })(t);
    await t.ok('a pedal and a side unit the engine skips are parked with the modules (B-12)', () => { const p = document.querySelector('.parked'); return !!p && !!p.querySelector('[data-pp="tile"]') && !!p.querySelector('[data-pp="lrc"]'); });
    await t.ok('the board and the go bar count them', () => /2 parked under the desk/.test(document.querySelector('[data-ppcount]').textContent) && /2 parked/.test(document.querySelector('#meter').textContent));
  } },
  { id: 'lyric-timing-follows-composer-cfg', run: async t => {
    await desk('yue', 'lab', ['score', 'planner'], { vals: { 'yue.timing': 'Hard' } })(t);
    await t.do('composer CFG off auto', () => { S.vals['yue.cfg.auto'] = false; setVal('yue.cfg', 2.1); });
    /* the lyric-timing line warns; the planner's token is not painted as clashing (design decision P3c-9) */
    await t.ok('the note warns at once and offers the fix (B-09)', () => /skips it/.test(document.querySelector('.tile[data-m="score"] .ctl-row .lbl').textContent));
    await t.click('.tile[data-m="score"] [data-auto="yue.cfg"]');
    await t.ok('Set CFG to auto clears it', () => isAuto('yue.cfg') && !/skips it/.test(document.querySelector('.tile[data-m="score"] .ctl-row .lbl').textContent));
  } },
  { id: 'continue-selection-keys', run: async t => {
    await desk('yue', 'lab', ['cont'])(t);
    await t.do('a loaded file', () => { S.contFile = 'take.wav'; S.contDur = 231; S.contSel = [10, 30]; render(); });
    await t.focus('[data-contwave]');
    await t.press('End'); await t.eq('End takes the selection to the end (B-28)', () => S.contSel, [211, 231]);
    await t.press('Home'); await t.eq('Home to the start', () => S.contSel, [0, 20]);
    await t.press('PageUp'); await t.eq('Page Up moves it by its length', () => S.contSel, [20, 40]);
    await t.ok('the ruler uses round steps and ends on the file\'s length', () => { const s = [...document.querySelectorAll('.cw-ruler > span')].map(x => x.textContent); return s[1] === '1:00' && s[s.length - 1] === '3:51'; });
  } },
];
