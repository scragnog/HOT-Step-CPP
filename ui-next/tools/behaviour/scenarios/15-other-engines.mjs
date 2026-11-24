// MiniMax-Music3 and YuE2: their own modules, and what the page says about them (review 3, E- findings).
const desk = (tier, mods, extra) => t => t.do(`${tier} desk`, ([tier, mods, extra]) => { S.tier = tier; S.mods = mods; S.pedals = []; if (extra?.vals) { Object.assign(S.vals, extra.vals); } Object.assign(S, { ...(extra || {}), vals: S.vals }); render(); }, [tier, mods, extra]);
export default [
  { id: 'build-replan-and-locks', run: async t => {
    await desk('lab', ['build'], { backend: 'yue' })(t);
    await t.do('two sections', () => setVal('lyrics.text', '[Verse]\nline\n[Chorus]\nline'));
    await t.click('.tile[data-m="build"] [data-bplan]'); await t.do('wait', () => {}, null, 1000);
    await t.ok('once planned, focus is on making takes (E-09)', () => document.activeElement.matches('.tile[data-m="build"] [data-btake]'));
    await t.click('.tile[data-m="build"] [data-btake]'); await t.do('wait', () => {}, null, 2000);
    await t.ok('Re-plan names what it will drop before the press (E-04)', () => /drops this build's 2 takes/.test(document.querySelector('.b-foot [data-breplan]').title));
    const undo0 = await t.js(() => document.querySelector('#undoBtn')?.textContent || '');
    await t.click('.b-foot [data-breplan]'); await t.do('wait', () => {}, null, 1000);
    await t.ok('Re-plan with takes and no keep says what it dropped (E-04)', () => !Object.values(S.build.takes).flat().length && /2 takes were dropped/.test(document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.press('Control+z');
    await t.ok('and Undo brings the takes back', () => S.build.takes['Verse#1']?.length === 2);
    const tid = await t.js(() => S.build.takes['Verse#1'][0].id);
    await t.click(`.tile[data-m="build"] [data-bkeep="${tid}"]`);
    await t.ok('a keep locks Follow score, and the locked value can be reached by keyboard (E-25)', () => S.build.locked && document.querySelector('.tok-ro').tabIndex === 0);
    await t.click(`.tile[data-m="build"] [data-bkeep="${tid}"]`);
    await t.ok('removing the last keep unlocks (E-25)', () => !S.build.locked && !document.querySelector('.tok-ro') && !document.querySelector('.b-locknote'));
    await t.click(`.tile[data-m="build"] [data-bkeep="${tid}"]`);
    await t.do('the verse gets new words', () => setVal('lyrics.text', '[Verse]\nother line\n[Chorus]\nline'));
    await t.ok('a keep sung to old words is flagged on its tab and in the banner, with one Re-plan (E-26)', () => !!document.querySelector('.b-src [data-bsec="Verse#1"].stale .b-stale') && /1 kept take was sung to the old words/.test(document.querySelector('.b-drift').textContent) && document.querySelectorAll('.tile[data-m="build"] [data-breplan]').length === 1);
  } },
  { id: 'build-tiers', run: async t => {
    await desk('quick', ['build'], { backend: 'yue', build: { planned: true, seed: 5, cur: null, plan: ['Verse#1'], takes: {}, kept: {}, locked: false } })(t);
    await t.do('lyrics', () => setVal('lyrics.text', '[Verse]\nline'));
    await t.ok('Quick shows sections, takes and Render, no controls (E-19)', () => !!document.querySelector('.tile[data-m="build"] [data-btake]') && !document.querySelector('.tile[data-m="build"] .b-ctl'));
    await t.do('studio', () => { S.tier = 'studio'; render(); });
    await t.ok('Studio adds section mode and takes, not Follow score', () => !!document.querySelector('.tile[data-m="build"] [data-seg="yue.bmode"]') && !document.querySelector('.tile[data-m="build"] [data-ctl="yue.bcond"]'));
    await t.do('lab', () => { S.tier = 'lab'; render(); });
    await t.ok('Lab adds Follow score, Plan strength and the seed', () => !!document.querySelector('.tile[data-m="build"] [data-ctl="yue.bcond"]') && /seed 5/.test(document.querySelector('.b-count').textContent));
  } },
  { id: 'plans-follow-the-fader', run: async t => {
    await desk('lab', ['plans', 'adapters'], { backend: 'mm3', slots: [{ file: 'my-band-ep', type: 'LoRA' }] })(t);
    /* the cache chip stands on the Last plan row (E-17) */ const n = () => document.querySelectorAll('.tile[data-m="plans"] .deck-row > *').length;
    const n0 = await t.js(n);
    await t.focus('[data-fader="ad.plan"]'); await t.press('ArrowDown');
    await t.ok('the Plan fader below 1.00 brings Compact planner cache at once (E-07)', k => document.querySelectorAll('.tile[data-m="plans"] .deck-row > *').length === k + 1 && !!document.querySelector('[data-ledtog="mm.kvq8"]') && document.activeElement.matches('[data-fader="ad.plan"]'), n0);
    await t.press('End');
    await t.ok('and back at 1.00 it goes', k => document.querySelectorAll('.tile[data-m="plans"] .deck-row > *').length === k && !document.querySelector('[data-ledtog="mm.kvq8"]'), n0);
    await t.do('both replays', () => { setVal('mm.plank', 'plank 11-19 22.41'); setVal('mm.hidpath', 'chorus-idea'); });
    await t.ok('two replays: the tile says which one is used, and the replaying row has nothing to save (E-34)', () => /Plank replay above wins/.test(document.querySelector('.tile[data-m="plans"]').textContent) && document.querySelector('[data-ledtog="mm.savear"]').classList.contains('is-off'));
  } },
  { id: 'prompt-truth-yue-mm3', run: async t => {
    await desk('studio', [], { backend: 'yue' })(t);
    await t.click('#prompt'); await t.type('{dreamy|gritty} indie folk');
    await t.ok('on YuE2 the wildcard line says what Generate will do (E-05)', () => /Sent as written unless you expand them/.test(document.querySelector('#suggest').textContent) && !document.querySelector('#wildAuto').hidden);
    await t.click('#wildAuto');
    await t.ok('and Expand on Generate is offered there', () => get('wild.auto') === true && /Each song picks one option/.test(document.querySelector('#suggest').textContent));
    await t.click('#go', 200);
    await t.ok('so the song gets one option, not the braces', () => S.songs[0].pending && !/[{}|]/.test(S.songs[0].style));
    await t.do('a structured caption', () => { setFieldText(document.querySelector('#prompt'), 'Global Metadata: smoky jazz trio, warm\nVocal Details: male lead\nArrangement: [Intro] piano'); renderSuggest(); });
    await t.ok('on YuE2 a Structured Caption is offered as plain words (carried item)', () => !!document.querySelector('#suggest [data-sg="plain"]'));
    await t.click('#suggest [data-sg="plain"]');
    await t.eq('which keeps the style and the vocals', () => document.querySelector('#prompt').value, 'smoky jazz trio, warm, male lead');
    await t.do('mm3', () => { S.backend = 'mm3'; render(); setFieldText(document.querySelector('#prompt'), 'Global Metadata: smoky jazz trio, warm\nVocal Details: male lead'); });
    await t.click('#go', 200);
    await t.ok('a MiniMax-Music3 song shows its Global Metadata as its description (E-35)', () => S.songs[0].style === 'smoky jazz trio, warm');
  } },
  { id: 'one-count-of-hidden-and-parked', run: async t => {
    await desk('quick', ['sampler', 'latent', 'perf', 'repetition', 'guidance', 'planner'], { backend: 'mm3' })(t);
    await t.ok('Quick on MiniMax-Music3: the Parked row lists the Lab modules it does not use (E-13)', () => !!document.querySelector('.parked [data-pm="latent"]') && !!document.querySelector('.parked [data-pm="perf"]'));
    await t.ok('and the go bar counts the same: 2 hidden (used), 2 parked', () => /2 hidden in Quick/.test(document.querySelector('#meter').textContent) && /2 parked/.test(document.querySelector('#meter').textContent) && /Repetition guard and Guidance plugin/.test(document.querySelector('#hiddenNote').textContent));
  } },
  { id: 'tempo-has-one-name', run: async t => {
    await desk('studio', [], { backend: 'mm3' })(t);
    await t.ok('the Modules panel names Tempo as its tile does on MiniMax-Music3, and describes what it holds there (E-15)', () => { const c = document.querySelector('.bcard[data-additem="tempo"]'); return /^Tempo & key\. .*Beat I\/O/.test(c.title) && !/Tempo, key & length/.test(c.textContent); });
  } },
  { id: 'lyric-length-uses-the-tempo', run: async t => {
    await desk('studio', ['tempo'], { backend: 'yue' })(t);
    await t.do('four lines', () => setVal('lyrics.text', '[Verse]\na\nb\nc\nd'));
    await t.ok('on auto the estimate says it guesses (E-16)', () => /at about 120 BPM/.test(document.querySelector('[data-lstat]').textContent));
    await t.click('.tile[data-m="tempo"] [data-num="tempo.bpm"]'); await t.type('90'); await t.press('Enter');
    await t.ok('a BPM typed on YuE2 is the one the lyric sheet uses', () => /≈ 0:21 at 90 BPM/.test(document.querySelector('[data-lstat]').textContent) || document.querySelector('[data-lstat]').textContent);
  } },
  { id: 'model-list-as-wide-as-its-row', run: async t => {
    await desk('lab', [], { backend: 'yue' })(t);
    await t.click('#engBtn'); await t.click('#engPanel [data-pick="model.dit"]');
    await t.ok('the model list is as wide as its row and YuE2\'s five quant chips stand on one line (E-32)', () => { const b = document.querySelector('#engPanel [data-pick="model.dit"]').getBoundingClientRect(), d = document.querySelector('#dd').getBoundingClientRect(), q = [...document.querySelectorAll('#dd .opt-mrow')[0].querySelectorAll('.opt-q')].map(e => Math.round(e.getBoundingClientRect().top)); return (Math.abs(b.width - d.width) < 1 && new Set(q).size === 1 && q.length === 5) || { b: b.width, d: d.width, q }; });
  } },
  { id: 'song-menu-repaint-from-another-engine', run: async t => {
    await desk('studio', ['tempo'], { backend: 'mm3' })(t);
    await t.click('#latest .song[data-id="1"] [data-songmenu]');
    await t.ok('on MiniMax-Music3 Repaint is live and says it switches (E-14)', () => { const b = document.querySelector('#menu [data-sa="repaint"]'); return !b.disabled && /Switches to ACE-Step 1.5/.test(b.textContent); });
    await t.click('#menu [data-sa="repaint"]');
    await t.ok('one press: ACE-Step with Repaint on the desk, one undo step, the switch toast', () => S.backend === 'ace' && S.mods[0] === 'repaint' && S.source?.song === 1 && /On ACE-Step 1\.5\..*Repaint is on the desk/.test(document.querySelector('#toast').textContent));
    await t.press('Control+z');
    await t.ok('and Undo puts back MiniMax-Music3 without Repaint', () => S.backend === 'mm3' && !S.mods.includes('repaint'));
  } },
  { id: 'switch-toast-names-models-and-counts-settings', run: async t => {
    await desk('lab', ['flow', 'sampler'], { backend: 'mm3' })(t);
    await t.do('two models and three settings of its own', () => { setVal('model.lm', '8B planner · F16'); setVal('model.dit', 'Flow DiT · F16'); setVal('mm.hop', 120); setVal('samp.steps', 30); setVal('mm.dcache', 2); });
    await t.click('#engBtn'); await t.click('#engPanel [data-backend="yue"]'); await t.press('Escape');
    await t.ok('the toast lists the models in brackets and counts every setting MiniMax-Music3 keeps, Hop included (E-33)', () => /MiniMax-Music3 keeps its models \((8B planner F16, Flow DiT F16|Flow DiT F16, 8B planner F16)\) and 3 settings\./.test(document.querySelector('#toast .toast-msg').textContent) || document.querySelector('#toast .toast-msg').textContent);
  } },
  { id: 'held-takes-leave-a-trace', run: async t => {
    await desk('studio', ['takes'], { backend: 'ace' })(t);
    await t.focus('[data-takes]'); await t.press('End');
    await t.click('#engBtn'); await t.click('#engPanel [data-backend="mm3"]'); await t.press('Escape');
    await t.do('the toast goes', () => {}, null, 12000);
    await t.ok('MiniMax-Music3 reads 4 and the tile still says the 8 are held for ACE-Step 1.5 (E-39; wording changed on purpose, S-10; eight at most, design decision 27)', () => get('takes.n') === 4 && /Your 8 takes return on ACE-Step 1\.5/.test(document.querySelector('.tk-held')?.textContent || ''));
    await t.focus('[data-takes]'); await t.press('ArrowLeft');
    await t.ok('a change here clears it', () => !document.querySelector('.tk-held'));
  } },
  { id: 'play-while-rendering', run: async t => {
    await desk('lab', ['flow'], { backend: 'mm3', vals: { 'mm.stream': true } })(t);
    await t.do('a description', () => { document.querySelector('#prompt').value = 'city pop'; });
    await t.click('#go', 300);
    await t.ok('the rendering card has no Play before the first window (E-40)', () => document.querySelector('.song.pending.streams [data-play]')?.hidden === true);
    await t.do('render on', () => {}, null, 1500);
    await t.ok('then it can be played, and the playable part is a second tone behind the render', () => { const c = document.querySelector('.song.pending.streams'); return !c.querySelector('[data-play]').hidden && parseFloat(c.querySelector('.bar.ready').style.width) > 0 && parseFloat(c.querySelector('.bar.ready').style.width) < parseFloat(c.querySelector('.bar:not(.ready)').style.width); });
    await t.click('.song.pending.streams [data-play]');
    await t.do('listen', () => {}, null, 1000);
    await t.ok('the player plays no further than is rendered', () => S.playing != null && PL.pos <= Math.floor(PL.s.dur * PL.s.ready));
  } },
];