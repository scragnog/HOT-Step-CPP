// The containers and read-outs (package 2), each in the module that draws it:
// the terminal and its lines (flags, the lines Lab adds, the plan the
// planner starts from, the planner off), a repetition guard that is off, Tempo's LCD rows (the beat lamps, the trim
// held), the rack groups (a power chip, a pill, token pairs, the plan strength curve), the plans deck, the re-planning
// dice, the flow windows, the speed gauge, the side units' objects, the drop zones, and the fader bank.

const desk = (tier, mods, vals = {}, backend = 'ace', pedals = [], more = {}) => async t => {
  await t.js(() => { window.__c = { toast: () => (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || ''),
    undo: () => (document.querySelector('#undoBtn').hidden ? '' : document.querySelector('#undoBtn span').textContent), text: s => document.querySelector(s)?.textContent ?? null,
    shown: s => { const e = document.querySelector(s); return !!e && getComputedStyle(e).display !== 'none' && getComputedStyle(e).visibility !== 'hidden'; } }; });
  await t.do(`${backend} ${tier} desk`, ([tier, mods, vals, backend, pedals, more]) => {
    if (typeof lyricsBlock === 'function') window.lyricsBlock = () => '';
    if (backend !== S.backend) { if (typeof enterEngine === 'function') enterEngine(backend); else S.backend = backend; }
    S.backend = backend; S.tier = tier; S.mods = mods; S.pedals = pedals; Object.assign(S.vals, vals); Object.assign(S, more); render();
  }, [tier, mods, vals, backend, pedals, more]);
};
const PL = '.tile[data-m="planner"]', REP = '.tile[data-m="repetition"]', TEMPO = '.tile[data-m="tempo"]';

export default [
  { id: 'terminal-prints-flags-and-gathers-lab-lines', run: async t => {
    await desk('lab', ['planner'])(t);
    await t.ok('a command line: "$ sample", then each value after its flag', () => { const l = document.querySelector('.tile[data-m="planner"] .term > .tline.hang'), o = l.querySelector('[data-scrub="lm.temp"]').closest('.opt');
      return (/^\$ sample --temp/.test(l.textContent.replace(/\s+/g, ' ').trim()) && o.querySelector('.k').textContent === '--temp' && parseFloat(getComputedStyle(l).textIndent) < 0) || l.textContent; });
    await t.ok('the lines Lab adds stand together under the rule', () => { const p = document.querySelector('.tile[data-m="planner"] .term > .tparams'); return !!p && !!p.querySelector('[data-scrub="lm.topk"]') && !p.querySelector('[data-scrub="lm.temp"]') && !!p.querySelector('.flag[data-ledtog="lm.logp"]'); });
    await t.ok('codes strength stands on the line, codes steps does not', () => !!document.querySelector('[data-scrub="lm.codes"]') && !document.querySelector('[data-scrub="lm.codesteps"]'));
    await t.click(`${PL} .tchips[data-ctl="lm.codesmode"] button:nth-child(2)`);
    await t.ok('a terminal chip chooses, and the line takes the token it brings', () => get('lm.codesmode') === 'Steps' && !!document.querySelector('[data-scrub="lm.codesteps"]') && !document.querySelector('[data-scrub="lm.codes"]'));
    await t.click(`${PL} .flag[data-ledtog="lm.logp"]`);
    await t.eq('a flag switches, and says so', () => [get('lm.logp'), document.querySelector('.flag[data-ledtog="lm.logp"]').getAttribute('aria-pressed')], [true, 'true']);
    await t.do('studio', () => { S.tier = 'studio'; render(); });
    await t.ok('at Studio the Lab lines are gone', () => !document.querySelector('.tile[data-m="planner"] .tparams') && !!document.querySelector('[data-scrub="lm.temp"]'));
  } },
  { id: 'planner-off-stands-the-terminal-back', run: async t => {
    await desk('lab', ['planner'])(t);
    await t.click(`${PL} [data-ledtog="lm.on"]`);
    await t.ok('the plan says the planner is off', () => /plan --off/.test(__c.text('[data-term]')) && /The planner is off/.test(__c.text('[data-term]')));
    await t.ok('its lines stand back at 45%, its power switch does not', () => { const l = document.querySelector('.tile[data-m="planner"] .term > .tline'), p = document.querySelector('.tile[data-m="planner"] .term-bar [data-ledtog="lm.on"]');
      return (getComputedStyle(l).opacity === '0.45' && getComputedStyle(p).opacity === '1') || [getComputedStyle(l).opacity, getComputedStyle(p).opacity]; });
    await t.focus(`${PL} [data-scrub="lm.temp"]`); await t.press('ArrowUp');
    await t.eq('and stays editable', () => get('lm.temp'), 0.85);
  } },
  { id: 'plan-text-follows-what-it-reads', run: async t => {
    await desk('lab', ['planner', 'tempo'])(t);
    await t.ok('with nothing set, the planner picks', () => /tempo\s+90 BPM \(planner's pick\)/.test(__c.text('[data-term]')) && /key\s+D major \(planner's pick\)/.test(__c.text('[data-term]')) && /form\s+planner decides/.test(__c.text('[data-term]')) || __c.text('[data-term]'));
    await t.do('a tempo, a key and sections', () => { S.vals['tempo.bpm.auto'] = false; S.vals['tempo.bpm'] = 128; S.vals['tempo.key'] = 'C# major'; S.vals['lyrics.text'] = '[Verse 1]\na\n[Chorus]\nb'; render(); });
    await t.ok('what you set is yours, spelled as a musician does', () => { const v = [...document.querySelectorAll('[data-term] .v')].map(e => e.textContent); return (v.includes('128 BPM (yours)') && v.includes('D♭ major (yours)') && v.includes('Verse 1 · Chorus')) || v; });
    await desk('lab', ['planner'], {}, 'yue')(t);
    await t.ok('YuE2 composes: the score its chain of thought writes', () => /compose --cot "Full lead sheet"/.test(__c.text('[data-term]')) && /score\s+melody \+ chords, ABC/.test(__c.text('[data-term]')) || __c.text('[data-term]'));
    await desk('studio', ['planner'], {}, 'mm3')(t);
    await t.ok('MiniMax-Music3 takes tempo and key from the caption', () => /plan --model 8B/.test(__c.text('[data-term]')) && /tempo\s+from the caption prose/.test(__c.text('[data-term]')) || __c.text('[data-term]'));
    await desk('studio', ['planner'], { 'lyrics.text': '' }, 'hm')(t);
    await t.ok('HeartMuLa takes tags, and the model ends the song', () => /\$ tags \(none yet\)/.test(__c.text('[data-term]')) && /lyrics\s+none, instrumental/.test(__c.text('[data-term]')) && /length\s+model ends it, up to 4:00/.test(__c.text('[data-term]')) || __c.text('[data-term]'));
  } },
  { id: 'guard-says-when-it-is-off', run: async t => {
    await desk('lab', ['planner', 'repetition'])(t);
    await t.ok('on, the guard says nothing of being off', () => !document.querySelector('.tile[data-m="repetition"] .term').classList.contains('guard-off') && !__c.shown('.tile[data-m="repetition"] .term-off'));
    await t.focus(`${REP} [data-scrub="lm.rep"]`); await t.press('Home');
    await t.ok('a penalty of 1.00 is a guard switched off, said on its line', () => document.querySelector('.tile[data-m="repetition"] .term').classList.contains('guard-off') && __c.shown('.tile[data-m="repetition"] .term-off') && __c.text('.tile[data-m="repetition"] .term-off') === 'guard off');
    await t.press('ArrowUp');
    await t.ok('and on again as soon as it penalises', () => !document.querySelector('.tile[data-m="repetition"] .term').classList.contains('guard-off'));
    await t.ok('DRY settings wait for DRY', () => !document.querySelector('[data-scrub="lm.drybase"]'));
    await t.click(`${REP} .tchips[data-ctl="lm.repmode"] button:nth-child(3)`);
    await t.ok('and come with it', () => get('lm.repmode') === 'DRY' && !!document.querySelector('[data-scrub="lm.drybase"]') && !!document.querySelector('[data-scrub="lm.drymin"]'));
    await t.click(`${PL} [data-ledtog="lm.on"]`);
    await t.ok('with the planner off the guard says so where it says guard off (L-11)', () => __c.shown('.tile[data-m="repetition"] .term-plan-off') && __c.text('.tile[data-m="repetition"] .term-plan-off') === 'planner off' && !__c.shown('.tile[data-m="repetition"] .term-off'));
    await desk('lab', ['repetition'], {}, 'yue')(t);
    await t.ok('YuE2: a line and a strip per guard', () => document.querySelectorAll('.tile[data-m="repetition"] .gline').length === 2 && document.querySelectorAll('.tile[data-m="repetition"] .gline [data-codes]').length === 2);
    await t.focus(`${REP} [data-scrub="yp.rep"]`); await t.press('Home');
    await t.ok('each guard is off on its own', () => document.querySelector('[data-guard="yp.rep"]').classList.contains('guard-off') && !document.querySelector('[data-guard="yc.rep"]').classList.contains('guard-off'));
  } },
  { id: 'tempo-lcd-rows', run: async t => {
    await desk('lab', ['tempo'])(t);
    await t.ok('the top row: the number, its word with the beat lamps under it, AUTO', () => { const top = document.querySelector('.tile[data-m="tempo"] .lcd > .lcd-top');
      return !!top && !!top.querySelector('[data-num="tempo.bpm"]') && top.querySelectorAll('.lcd-unit .lcd-pulse i').length === 4 && !!top.querySelector('.lcd-btns [data-auto="tempo.bpm"]'); });
    await t.ok('on auto the lamps rest', () => [...document.querySelectorAll('.tile[data-m="tempo"] .lcd-pulse i')].every(i => i.classList.contains('idle')));
    await t.click(`${TEMPO} .lcd-top [data-auto="tempo.bpm"]`);
    await t.ok('AUTO beside the number turns auto off at the tempo auto stood for, and the lamps beat', () => !isAuto('tempo.bpm') && get('tempo.bpm') === 120 && document.querySelector('.tile[data-m="tempo"] [data-auto="tempo.bpm"]').getAttribute('aria-pressed') === 'false' && ![...document.querySelectorAll('.tile[data-m="tempo"] .lcd-pulse i')].some(i => i.classList.contains('idle')));
    await t.do('120 BPM in 3/4', () => { S.vals['tempo.bpm.auto'] = false; S.vals['tempo.bpm'] = 120; S.vals['tempo.sig'] = '3/4'; render(); });
    await t.ok('three lamps, a beat half a second apart', () => { const i = [...document.querySelectorAll('.tile[data-m="tempo"] .lcd-pulse i')]; return (i.length === 3 && !i[0].classList.contains('idle') && i[0].style.getPropertyValue('--beat') === '0.500s') || i.map(x => x.getAttribute('style')); });
    await t.ok('the key row says the key in capitals', () => /^AUTO$|^[A-G]/.test(__c.text('.tile[data-m="tempo"] [data-keylcd]')));
    await t.ok('the trim settings are held out of sight while Auto-trim is off', () => !__c.shown('.tile[data-m="tempo"] .lcd-trim') && document.querySelector('.tile[data-m="tempo"] .lcd-trim').classList.contains('held'));
    await t.click(`${TEMPO} [data-ledtog="tempo.trim"]`);
    await t.ok('and stand at the row\'s end once it is on', () => __c.shown('.tile[data-m="tempo"] .lcd-trim') && !!document.querySelector('.tile[data-m="tempo"] .lcd-ends .lcd-trim [data-scrub="tempo.trimbuf"]'));
  } },
  { id: 'rack-groups-switch-and-stand-back', run: async t => {
    await desk('lab', ['sampler', 'adhere'], {}, 'yue')(t);
    const G = '.tile[data-m="adhere"] .ad-groups > .ad-g';
    await t.ok('three racks in a row; Plan CFG is off: hatched, its chip says so', () => document.querySelectorAll('.tile[data-m="adhere"] .ad-groups > .ad-g').length === 3 && document.querySelectorAll('.tile[data-m="adhere"] .ad-groups > .ad-g')[1].classList.contains('dim') && /Plan CFG · off/.test(document.querySelectorAll('.tile[data-m="adhere"] .ad-g')[1].querySelector('.ad-gh .pill-tog').textContent));
    await t.ok('its tokens stand in pairs, the range with its arrow', () => { const g = document.querySelectorAll('.tile[data-m="adhere"] .ad-g')[1], p = g.querySelectorAll('.toks > .tk-pair'); return p.length === 2 && !!p[0].querySelector('.tk-arrow') && p[0].querySelectorAll('.tok').length === 2; });
    await t.click(`${G}:nth-child(2) .ad-gh .pill-tog`);
    await t.ok('the chip switches it on, and the curve shows its band', () => get('yue.pcfg') === 1.5 && !document.querySelectorAll('.tile[data-m="adhere"] .ad-g')[1].classList.contains('dim') && /plan CFG 1\.5/.test(__c.text('.tile[data-m="adhere"] .ad-leg')) && !!document.querySelector('[data-adhere] svg rect'));
    await t.focus('.tile[data-m="adhere"] [data-knob="yue.pws"]'); await t.press('End');
    await t.ok('the curve follows the plan strength', () => /plan strength 2\.00 → 1\.00/.test(__c.text('.tile[data-m="adhere"] .ad-leg')) && document.querySelector('[data-adhere] polyline').getAttribute('points').startsWith('8.0,8.0'));
    await desk('lab', ['sampler', 'latent'])(t);
    await t.ok('a rack switched by a pill: off, hatched', () => document.querySelector('.tile[data-m="latent"] .lat-groups > .lat-g').classList.contains('off'));
    await t.click('.tile[data-m="latent"] [data-ledtog="dcw.on"]');
    await t.ok('the pill turns it on; its compact choice says each option short', () => !document.querySelector('.tile[data-m="latent"] .lat-g').classList.contains('off') && document.querySelector('.tile[data-m="latent"] [data-ctl="dcw.mode"] button').textContent === 'Both' && document.querySelector('.tile[data-m="latent"] [data-ctl="dcw.mode"] button').title === 'Both (Low + High)');
  } },
  { id: 'plans-deck-rows', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['planner', 'plans'], {}, 'mm3')(t);
    await t.ok('two columns: labels and controls, a recorder\'s replay indented under it', () => { const l = [...document.querySelectorAll('.tile[data-m="plans"] .deck > .deck-l')]; return (l.map(x => x.textContent).join('|') === 'Last plan|Plank|Replay|Full plan|Replay' && l.filter(x => x.classList.contains('deck-sub')).length === 2) || l.map(x => x.textContent); });
    await t.click('.tile[data-m="plans"] [data-ledtog="mm.reuse"]');
    await t.ok('the planner says the plan cache is off (E-15)', () => /8B planner · plan cache off/.test(__c.text('.tile[data-m="planner"] .term-bar')) || __c.text('.tile[data-m="planner"] .term-bar'));
    await t.ok('a changed row carries its mark after its label (E-36)', () => { const l = [...document.querySelectorAll('.tile[data-m="plans"] .deck-l')].find(x => x.textContent === 'Last plan'); return getComputedStyle(l, '::after').width === '5px'; });
    await t.click('.tile[data-m="plans"] [data-cycle="mm.hidpath|1"]');
    await t.click('.tile[data-m="plans"] [data-cycle="mm.plank|1"]');
    await t.ok('and names the Plank it replays', () => /8B planner · replaying 11-19 22\.41/.test(__c.text('.tile[data-m="planner"] .term-bar')) || __c.text('.tile[data-m="planner"] .term-bar'));
    await t.ok('a recorder that replays stands back; the other replay says it is not used (E-34)', () => document.querySelector('[data-ledtog="mm.savear"]').classList.contains('is-off') && !!document.querySelector('.tile[data-m="plans"] .deck-lcd.is-off') && __c.text('.tile[data-m="plans"] .deck-why') === 'Not used: the Plank replay above wins');
  } },
  { id: 'dice-flow-and-gauge-read-their-values', run: async t => {
    await desk('lab', ['planner', 'replan'], {}, 'yue')(t);
    await t.eq('a die per allowed attempt, under its caption', () => [document.querySelectorAll('.tile[data-m="replan"] .dice-cap .dicebox .die').length, __c.text('.tile[data-m="replan"] .dice-cap > .kl')], [3, 'Plan attempts']);
    await t.do('five tries', () => setVal('yue.tries', 5));
    await t.eq('the dice follow the tries', () => document.querySelectorAll('.tile[data-m="replan"] .die').length, 5);
    await t.click('.tile[data-m="replan"] [data-ledtog="yue.replan"]');
    await t.ok('re-planning off: the dice are off', () => [...document.querySelectorAll('.tile[data-m="replan"] .die')].every(d => d.classList.contains('off')));
    await desk('lab', ['sampler', 'flow'], {}, 'mm3')(t);
    await t.eq('seven windows at a 6 s hop; 30 a song', () => [document.querySelectorAll('.tile[data-m="flow"] .flowviz i').length, __c.text('.tile[data-m="flow"] [data-flowcap] b')], [7, '30']);
    await t.do('the checkpoint\'s hop', () => setVal('mm.hop', 100));
    await t.eq('the windows and the count follow the hop', () => [document.querySelectorAll('.tile[data-m="flow"] .flowviz i').length, __c.text('.tile[data-m="flow"] [data-flowcap]')], [11, '44 windows per 3:00 · as many as the checkpoint\'s own hop4.0 s shared at each seam']);
    await t.do('the longest hop', () => setVal('mm.hop', 160));
    await t.ok('past 6 s the seam says to listen at the joins', () => /listen at the joins/.test(__c.text('.tile[data-m="flow"] [data-flowcap]')));
    await desk('lab', ['sampler', 'perf'])(t);
    await t.eq('nothing traded, no faster', () => __c.text('.tile[data-m="perf"] [data-speed]'), '0%');
    await t.do('CFG cut at half', () => setVal('perf.cfgcut', 0.5));
    await t.eq('the gauge reads the trade', () => __c.text('.tile[data-m="perf"] [data-speed]'), '+20%');
  } },
  { id: 'side-unit-objects', run: async t => {
    await desk('lab', ['post'], {}, 'ace', ['lrc', 'art', 'q'])(t);
    await t.eq('the lyric timestamps name the first section, verse by default', () => __c.text('.pedal[data-xid="lrc"] .u-lcd span'), '[00:12.40] verse');
    await t.do('lyrics that start with a chorus', () => { S.vals['lyrics.text'] = '[Chorus]\nla la'; render(); });
    await t.eq('and follow the lyrics', () => __c.text('.pedal[data-xid="lrc"] .u-lcd span'), '[00:12.40] chorus');
    await t.ok('the quality evaluator shows its badge, the cover art its swatch', () => __c.text('.pedal[data-xid="q"] .u-badge') === '8.4' && /^\d+$/.test(document.querySelector('.pedal[data-xid="art"] .u-art').style.getPropertyValue('--h')));
    await desk('lab', ['post'], {}, 'yue', ['fa'])(t);
    await t.ok('forced alignment says what it needs, its tip says more', () => __c.text('.pedal[data-xid="fa"] .u-status') === 'After the render · needs MMS' && /MMS aligner/.test(document.querySelector('.pedal[data-xid="fa"] .u-status').title));
  } },
  { id: 'drop-zones-take-audio-and-clear', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['post', 'timbre', 'tempo'], {}, 'ace', ['master'])(t);
    await t.file('.pedal[data-pid="master"] [data-file="master"]', 'ref.wav', 'audio/wav');
    await t.ok('a chosen file loads into the pedal\'s slot, named for the file', () => S.masterRef === 'ref.wav' && __c.text('.pedal[data-pid="master"] .pslot .ps-name') === 'ref.wav' && document.querySelector('.pedal[data-pid="master"] .pslot').classList.contains('loaded') && __c.undo() === 'Load ref.wav as the mastering reference');
    await t.click('.pedal[data-pid="master"] [data-fileclear="master"]');
    await t.eq('× clears it', () => S.masterRef, '');
    await t.dropFile('.tile[data-m="timbre"] label.drop', 'notes.txt', 'text/plain');
    await t.ok('a file that is no audio is refused, and said', () => S.timbreFile === '' && /isn't an audio file/.test(__c.toast()));
    await t.dropFile('.tile[data-m="timbre"] label.drop', 'kick.wav', 'audio/wav');
    await t.ok('an audio file dropped on the sleeve loads', () => S.timbreFile === 'kick.wav' && document.querySelector('.tile[data-m="timbre"] .drop.sleeve').classList.contains('loaded') && /kick\.wav/.test(__c.text('.tile[data-m="timbre"] .sl-txt')));
    await t.dropFile('.tile[data-m="tempo"]', 'x.wav', 'audio/wav');
    await t.ok('a drop outside a zone says where to drop', () => /Drop the file on one of the drop zones/.test(__c.toast()));
    await t.do('the mastering reference lent to Timbre', () => { S.timbreFile = ''; S.masterRef = 'master-ref.wav'; S.vals['pp.master.timbre'] = true; render(); });
    await t.ok('Timbre shows the lent track as its own, marked, and takes none itself (L-15)', () => /master-ref\.wav/.test(__c.text('.tile[data-m="timbre"] .sl-txt')) && /from Mastering/.test(__c.text('.tile[data-m="timbre"] .sl-txt')) && !document.querySelector('.tile[data-m="timbre"] [data-file]'));
  } },
  { id: 'fader-bank-scale-and-tapes', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['adapters'], {}, 'ace', [], { slots: [{ file: 'tape-lofi-v2', type: 'LoRA' }] })(t);
    await t.ok('the bank: its heading with its note as the tip, the 2-1-0 scale, a fader a group on its tape', () => { const b = document.querySelector('.tile[data-m="adapters"] .rig-master');
      return !!b && /every DiT adapter/.test(b.querySelector('.rig-mh span').title) && [...b.querySelectorAll('.fdr-scale span')].map(s => s.textContent).join('') === '210' && b.querySelectorAll('.faders .fdr').length === 6 && b.querySelector('.faders .fdr .fdr-l').textContent === 'Self'; });
    await t.focus('.tile[data-m="adapters"] .rig-master [data-fader="ag.self"]'); await t.press('ArrowDown');
    await t.ok('a fader in the bank works as every fader does', () => get('ag.self') < 1);
  } },
];
