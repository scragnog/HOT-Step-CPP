// The instruments of package 3c (YuE2's Build by section, Continue from recording and Lead sheet, MiniMax-Music3's
// Flow renderer), as scenarios: the build's whole life
// from empty to planned, takes made, kept, discarded and rendered, every user decision a step and every arrival none;
// a build following new lyrics; the strip's ways out; the recording's selection under the hand; the lead sheet's foot,
// its preview and its lyric-timing line.

const desk = (tier, mods, vals = {}, backend = 'yue', more = {}) => async t => {
  await t.js(() => { window.B = '.tile[data-m="build"]'; window.__b = {
    toast: () => (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || ''),
    undo: () => (document.querySelector('#undoBtn').hidden ? '' : document.querySelector('#undoBtn span').textContent),
    text: s => document.querySelector(s)?.textContent.replace(/\u00a0/g, ' ').replace(/\s+/g, ' ').trim() ?? null,
    n: s => document.querySelectorAll(s).length,
  }; });
  await t.do(`${backend} ${tier} desk`, ([tier, mods, vals, backend, more]) => {
    if (backend !== S.backend) enterEngine(backend);
    S.tier = tier; S.mods = mods; S.pedals = []; Object.assign(S.vals, vals); Object.assign(S, more); render();
  }, [tier, mods, vals, backend, more]);
};
const B = '.tile[data-m="build"]';
const TWO = '[Verse]\nline\n[Chorus]\nline';
const plan = async t => { await t.do('two sections', l => setVal('lyrics.text', l), TWO); await t.click(`${B} [data-bplan]`); await t.do('the score lands', () => {}, null, 1000); };
const takes = async t => { await t.click(`${B} [data-btake]`); await t.do('the takes land', () => {}, null, 2000); };
const tid = (t, i) => t.js(i => document.querySelectorAll('.tile[data-m="build"] .b-take')[i].dataset.take, i);

export default [
  { id: 'build-plans-then-makes-takes', run: async t => {
    await desk('lab', ['build'])(t);
    await t.do('two sections', l => setVal('lyrics.text', l), TWO);
    await t.ok('empty, the board offers to plan the score', () => /Build the song one section at a time/.test(__b.text(`${B} .b-empty`)) && !!document.querySelector(`${B} [data-bplan]`) && !document.querySelector(`${B} .b-src`));
    const steps = await t.js(() => UNDO.length);
    await t.click(`${B} [data-bplan]`, 200);
    await t.ok('while the planner writes, the button says so', () => document.querySelector(`${B} [data-bplan]`)?.getAttribute('aria-busy') === 'true' && /Planning/.test(__b.text(`${B} [data-bplan]`)));
    await t.do('the score lands', () => {}, null, 1000);
    await t.eq('then the sections, in their colours, the first being built', () => [...document.querySelectorAll(`${B} .b-src [data-bsec]`)].map(b => [b.dataset.bsec, b.classList.contains('cur')]), [['Verse#1', true], ['Chorus#1', false]]);
    await t.ok('focus goes on to making takes (E-09)', () => document.activeElement?.matches(`${B} [data-btake]`));
    await t.eq('the button names the takes it makes', () => __b.text(`${B} [data-btake]`), 'Generate 2 takes');
    await takes(t);
    await t.eq('the takes land, each with its waveform, length, play, discard and Keep', () => [...document.querySelectorAll(`${B} .b-take`)].map(c => [!c.classList.contains('busy'), !!c.querySelector('[data-bplay]'), !!c.querySelector('[data-bdrop]'), c.querySelector('[data-bkeep]')?.textContent, /^\d:\d\d/.test(c.querySelector('small').textContent)]), [[true, true, true, 'Keep', true], [true, true, true, 'Keep', true]]);
    await t.eq('once there are takes, the button makes more', () => __b.text(`${B} [data-btake]`), 'More takes');
    await t.eq('planning was one step at most; the takes arriving were none', n0 => UNDO.length - n0 <= 1, true, steps);
    const n = await t.js(() => UNDO.length);
    await t.click(`${B} .b-src [data-bsec="Chorus#1"]`);
    await t.ok('picking a section shows its own takes, and is no step', () => document.querySelector(`${B} .b-src [data-bsec="Chorus#1"]`).classList.contains('cur') && !document.querySelector(`${B} .b-take`) && /^Chorus/.test(__b.text(`${B} .b-cur`)));
    await t.eq('(no step)', () => UNDO.length, n);
    await t.click(`${B} .b-src [data-bsec="Verse#1"]`);
    await t.eq('back on the verse, its takes are there', () => __b.n(`${B} .b-take`), 2);
    await t.click(`${B} .b-take:nth-child(2) [data-bplay]`, 300);
    await t.ok('a take can be heard before it is kept: its card plays, its button says Stop (B-07)', () => { const c = document.querySelector(`${B} .b-take.playing`); return !!c && c === document.querySelectorAll(`${B} .b-take`)[1] && c.querySelector('[data-bplay]').getAttribute('aria-pressed') === 'true' && /^Stop/.test(c.querySelector('[data-bplay]').getAttribute('aria-label')); });
    await t.click(`${B} .b-take:nth-child(1) [data-bplay]`, 300);
    await t.ok('playing another stops the first', () => document.querySelectorAll(`${B} .b-take.playing`).length === 1 && document.querySelector(`${B} .b-take.playing`) === document.querySelectorAll(`${B} .b-take`)[0]);
  } },
  { id: 'build-keeps-and-discards-are-steps', run: async t => {
    await desk('lab', ['build'])(t);
    await plan(t); await takes(t);
    const t2 = await tid(t, 1);
    await t.click(`${B} [data-bkeep="${t2}"]`);
    await t.eq('Keep is a step named for the take and its section', () => __b.undo(), 'Keep take 2 for Verse');
    await t.ok('the keep stays on its card and checks its section, and offers the next one (B-37)', t2 => document.querySelector(`${B} [data-take="${t2}"]`).classList.contains('keep') && document.querySelector(`${B} .b-src [data-bsec="Verse#1"]`).classList.contains('kept') && /Next: Chorus/.test(__b.text(`${B} .b-cur`)), t2);
    await t.ok('Follow score and Plan strength are locked: read-outs with a padlock, reachable by keyboard (E-25, E-31)', () => document.querySelectorAll(`${B} .foot-ctl.locked .tok-ro`).length === 2 && [...document.querySelectorAll(`${B} .tok-ro`)].every(r => r.tabIndex === 0 && r.querySelector('svg')) && /Locked for this build/.test(__b.text(`${B} .b-locknote`)));
    await t.ok('the first keep says the settings are locked', () => /now locked for this build/.test(__b.toast()));
    const t1 = await tid(t, 0);
    await t.click(`${B} [data-bdrop="${t1}"]`);
    await t.eq('Discard is a step', () => __b.undo(), 'Discard take 1');
    await t.ok('said in a toast that offers Undo', () => /Discarded take 1/.test(__b.toast()) && document.querySelector('#toast .toast-act')?.textContent === 'Undo');
    await t.eq('the take is gone', () => __b.n(`${B} .b-take`), 1);
    await t.click('#toast .toast-act');
    await t.eq('and Undo brings it back', () => __b.n(`${B} .b-take`), 2);
    await t.click(`${B} [data-bkeep="${t2}"]`);
    await t.eq('letting go of the keep is a step too', () => __b.undo(), 'Stop keeping take 2 for Verse');
    await t.ok('and frees the locks (E-25)', () => !document.querySelector(`${B} .tok-ro`) && !document.querySelector(`${B} .b-locknote`) && !!document.querySelector(`${B} [data-scrub="yue.bcond"]`));
  } },
  { id: 'build-render-keeps-the-build', run: async t => {
    await desk('lab', ['build'])(t);
    await plan(t); await takes(t);
    const t2 = await tid(t, 1);
    await t.click(`${B} [data-bkeep="${t2}"]`);
    await t.eq('with one of two sections kept, Render says how much it renders', () => __b.text(`${B} [data-brender]`), 'Render 1 of 2 sections');
    await t.click(`${B} [data-brender]`);
    await t.ok('Render saves the song and says the build stays (B-06)', () => /^Saved “Built song” to the Library\. The build stays here/.test(__b.toast()) && document.querySelector('#toast .toast-act')?.textContent === 'Undo');
    await t.ok('the build is still there, rendered, and can be rendered again or started anew', t2 => /^Rendered as “Built song” · 1 of 2 kept/.test(__b.text(`${B} .b-count`)) && __b.text(`${B} [data-brender]`) === 'Render again' && !!document.querySelector(`${B} [data-bnew]`) && document.querySelector(`${B} [data-take="${t2}"]`).classList.contains('keep'), t2);
    await t.eq('Render is a step', () => __b.undo(), 'Render "Built song"');
    await t.ok('the song is first in the Library, built from its sections', () => S.songs[0]?.bk === 'yue' && S.songs[0].title === 'Built song' && /^Built from 1 section: Verse$/.test(S.songs[0].style));
    await t.press('Control+z');
    await t.ok('and Undo takes it out of the Library', () => !S.songs.some(s => s.title === 'Built song'));
    await t.ok('Undo takes the render back and leaves the keep', t2 => !/Rendered/.test(__b.text(`${B} .b-count`)) && !document.querySelector(`${B} [data-bnew]`) && document.querySelector(`${B} [data-take="${t2}"]`).classList.contains('keep'), t2);
    await t.click(`${B} [data-brender]`);
    await t.click(`${B} [data-bnew]`);
    await t.ok('Start a new build empties the board, with Undo', () => !!document.querySelector(`${B} [data-bplan]`) && /Started a new build/.test(__b.toast()) && __b.undo() === 'Start a new build');
    await t.press('Control+z');
    await t.ok('and Undo brings the build back whole', t2 => !!document.querySelector(`${B} [data-take="${t2}"].keep`), t2);
  } },
  { id: 'build-follows-new-lyrics', run: async t => {
    await desk('studio', ['build'])(t);
    await plan(t); await takes(t);
    const t1 = await tid(t, 0);
    await t.click(`${B} [data-bkeep="${t1}"]`);
    await t.do('an intro above, and the verse rewritten', () => setVal('lyrics.text', '[Intro]\nhi\n[Verse]\nnew words\n[Chorus]\nline'));
    await t.ok('the keep stays with its section and is flagged as sung to the old words (B-05, E-26)', () => { const v = document.querySelector(`${B} .b-src [data-bsec="Verse#1"]`); return v.classList.contains('kept') && v.classList.contains('stale') && !!v.querySelector('.b-stale') && document.querySelector(`${B} .b-src [data-bsec="Intro#1"]`).classList.contains('new'); });
    await t.eq('the banner says what changed, with one Re-plan and Keep building', () => [__b.text(`${B} .b-drift > span`), __b.n(`${B} [data-breplan]`), __b.n(`${B} .b-drift [data-bgoon]`)], ['The lyrics changed since this score was planned. 1 kept take was sung to the old words. 1 new section is not built yet.', 1, 1]);
    await t.ok('Re-plan names what it would drop before the press (E-04)', () => /and drops this build's 2 takes$/.test(document.querySelector(`${B} [data-breplan]`).title));
    await t.click(`${B} .b-drift [data-bgoon]`);
    await t.eq('Keep building is a step', () => __b.undo(), 'Keep building on the new lyrics');
    await t.ok('the plan follows the lyrics: the banner says only that a keep is sung to old words', () => !document.querySelector(`${B} .b-drift [data-bgoon]`) && /1 kept take was sung to the old words/.test(__b.text(`${B} .b-drift`)) && !document.querySelector(`${B} .b-src .new`));
    await t.press('Control+z');
    await t.ok('Undo puts the old plan back', () => !!document.querySelector(`${B} .b-src [data-bsec="Intro#1"].new`));
    await t.do('let the autosave run', () => {}, null, 800);
    await t.reload();
    await t.ok('the build, its takes and its keep come back after a reload (B-06)', () => !!document.querySelector(`.tile[data-m="build"] .b-src [data-bsec="Verse#1"].kept`) && document.querySelectorAll('.tile[data-m="build"] .b-take').length === 2);
  } },
  { id: 'build-strip-offers-the-way-on', run: async t => {
    await desk('lab', ['build', 'planner'], { 'lyrics.text': '' })(t);
    await t.ok('with nothing to build, the tile is a strip that offers section tags', () => document.querySelector(B).classList.contains('is-strip') && /Tag the lyrics by section/.test(__b.text(B)) && /\+ \[Verse\]/.test(__b.text(`${B} .sbtn`)));
    await t.click(`${B} .sbtn`);
    await t.ok('a tag goes into the lyrics and the board opens', () => /^\[Verse\]/.test(get('lyrics.text')) && !!document.querySelector(`${B} [data-bplan]`));
    await t.do('Chain of Thought off', () => setVal('lm.cotmode', 'Off'));
    await t.ok('the board folds to a strip that says why, as wide as the tile (E-19)', () => document.querySelector(B).classList.contains('is-strip') && document.querySelector(B).classList.contains('w4') && /Needs a planned score, and Chain of Thought is off/.test(__b.text(B)));
    await t.click(`${B} .sbtn`);
    await t.ok('its way out turns Chain of Thought on, and the board is back', () => get('lm.cotmode') === 'Full lead sheet' && !document.querySelector(B).classList.contains('is-strip') && !!document.querySelector(`${B} [data-bplan]`));
  } },
  { id: 'recording-selection-under-the-hand', run: async t => {
    await desk('lab', ['cont'], {}, 'yue', { contFile: 'take.wav', contDur: 200, contSel: [20, 40] })(t);
    await t.ok('the sleeve says the recording’s length, and the tile says it is experimental', () => /3:20 long/.test(__b.text('.tile[data-m="cont"] .sleeve')) && __b.text('.tile[data-m="cont"] .exp') === 'Experimental' && !/Build by section/.test(__b.text('.tile[data-m="cont"]')));
    const at = s => t.js(s => { const el = document.querySelector('[data-contwave]'), w = el.getBoundingClientRect().width; return (10 + s / 200 * (w - 20)) / w; }, s);
    await t.mouseDrag('[data-contwave]', [[40, 0]], [await at(30), 0.5]);
    await t.ok('a drag inside carries the part, its length kept (G-B-01)', () => S.contSel[0] > 20 && S.contSel[1] - S.contSel[0] === 20);
    await t.eq('one step, named in the recording\'s times', () => /^Part to continue from 0:20–0:40 → 0:\d\d–\d:\d\d$/.test(__b.undo()), true);
    const [a] = await t.js(() => S.contSel);
    await t.mouseDrag('[data-contwave]', [[-30, 0]], [await at(a + 20), 0.5]);
    await t.ok('a drag at an edge moves that edge only', a => S.contSel[0] === a && S.contSel[1] < a + 20, a);
    await t.mouseDrag('[data-contwave]', [[2, 0]], [await at(150), 0.5]);
    await t.ok('a click outside that barely moves changes nothing', a => S.contSel[0] === a, a);
    await t.mouseDrag('[data-contwave]', [[120, 0]], [await at(120), 0.5]);
    await t.ok('a drag outside draws a new part, never longer than 30 seconds', () => S.contSel[0] >= 119 && S.contSel[1] - S.contSel[0] <= 30 && S.contSel[1] - S.contSel[0] > 1);
    await t.ok('the caption follows the part', () => { const [a, z] = S.contSel; const m = s => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; return __b.text('.tile[data-m="cont"] .cap').startsWith(`${m(a)} – ${m(z)} · ${z - a} s of 30 s max`); });
    await t.dblclick('[data-contwave]');
    await t.eq('a double-click puts back the first 20 seconds', () => S.contSel, [0, 20]);
    await t.ok('the caption and the slider say the part in the recording\'s times', () => /^0:00 – 0:20 · 20 s of 30 s max/.test(__b.text('.tile[data-m="cont"] .cap')) && document.querySelector('[data-contwave]').getAttribute('aria-valuetext') === '0:00 to 0:20');
  } },
  { id: 'lead-sheet-foot-and-preview', run: async t => {
    await desk('studio', ['score', 'planner'])(t);
    await t.eq('empty, the foot offers an example and a planned score', () => [...document.querySelectorAll('[data-abcfoot] button')].map(b => b.textContent.trim()), ['Insert example', 'Plan and preview a score']);
    await t.click('[data-abcfoot] [data-scoreprev]');
    await t.ok('the preview opens on a planned score, with its seed and verdict, focus on its way on', () => !document.querySelector('#scoreModal').hidden && /^seed \d+ · /.test(__b.text('.sc-meta')) && !!document.querySelector('.sc-verdict') && document.activeElement?.matches('[data-scorego], [data-scoreretry]'));
    for (let i = 0; i < 8; i++) await t.press('Tab', 50);
    await t.ok('Tab goes round the dialog', () => document.querySelector('#scoreModal').contains(document.activeElement));
    await t.press('Escape');
    await t.ok('Escape closes it, and focus goes back to the button that opened it', () => document.querySelector('#scoreModal').hidden && document.activeElement?.matches('[data-abcfoot] [data-scoreprev]'));
    await t.do('a runaway plan', () => openScore(12345), null, 600);
    await t.ok('a plan that never closes is drawn as it plays: a repeat sign and "and again", no final bar, Retry first (E-23, B-47)', () => document.querySelector('.sc-verdict').textContent.trim().toLowerCase() === 'runaway' && /and again/.test(__b.text('.staff .st-more')) && __b.n('.staff .st-line.open') === 5 && document.activeElement?.matches('[data-scoreretry]') && /then loops/.test(__b.text('.sc-meta')));
    await t.click('#scoreModal .sc-order [data-scoresec="2"]');
    await t.ok('a section in the trail plays from its first bar, lit in the trail and on the staff (E-01)', () => { const i = SC.meas.findIndex(m => m.si === 2), x = +document.querySelector('.staff .st-head').getAttribute('x'), R = scoreLayout(SC).rows.flatMap(r => r.bars)[i]; return document.querySelector('.sc-order [data-scoresec="2"]').classList.contains('cur') && !!document.querySelector('.staff .sc-sec.cur[data-si="2"]') && x >= R.notes[0] - 5.5 && x <= R.bx + R.bw && __b.text('[data-scoreplay]') === 'Stop'; });
    await t.ok('the playhead stands on its system, as tall as its five lines', () => { const h = document.querySelector('.staff .st-head'); return +h.getAttribute('height') < 45; });
    await t.click('#scoreModal [data-scoreplay]');
    await t.eq('Stop stops it', () => __b.text('[data-scoreplay]'), 'Play');
    await t.click('#scoreModal .sc-acts [data-scoreclose]');
    await t.click('[data-abcfoot] [data-example="yue.abc"]');
    await t.ok('Insert example fills the sheet, and the foot counts its lines', () => /^X:1/.test(get('yue.abc')) && /^\d+ lines\. The composer follows this instead of planning one\./.test(__b.text('[data-abcfoot] .lbl')) && __b.text('[data-abcfoot] [data-scoreprev]') === 'Preview this score');
    await t.click('[data-abcfoot] [data-scoreprev]');
    await t.ok('with ABC on the sheet, the preview shows it as written', () => /pasted on the lead sheet$/.test(__b.text('.sc-meta')) && __b.text('.sc-head h3') === 'Stay a little longer' && /^X:1/.test(__b.text('.sc-abc')));
    await t.click('#scoreModal [data-scoredlabc]');
    await t.ok('and downloads it under its title', () => window.__bh.downloads.at(-1)?.name === 'Stay_a_little_longer.abc.txt');
    await t.press('Escape');
    await t.do('words that are not ABC', () => { document.activeElement?.blur(); setVal('yue.abc', 'just some words'); });
    await t.ok('the foot says they do not read as ABC (E-26)', () => document.querySelector('[data-abcfoot] .lbl').classList.contains('warn') && /doesn't read as ABC/.test(__b.text('[data-abcfoot] .lbl')));
  } },
  { id: 'lead-sheet-without-a-planned-score', run: async t => {
    await desk('studio', ['score', 'planner'], { 'lm.cotmode': 'Off' })(t);
    await t.ok('with Chain of Thought off, the foot says the Planner writes nothing down and offers the way back (E-12)', () => /The Planner is set to write nothing down\./.test(__b.text('[data-abcfoot]')) && !document.querySelector('[data-abcfoot] [data-scoreprev]') && !!document.querySelector('[data-abcfoot] [data-bcot]'));
    await t.ok('the sheet stays, saying it waits for a planner that writes', () => /Planner is set to write nothing down/.test(document.querySelector('.tile[data-m="score"] [data-bind="yue.abc"]')?.placeholder ?? ''));
    await t.click('[data-abcfoot] [data-bcot]');
    await t.ok('Write a full score turns it back on and hands focus to the preview', () => get('lm.cotmode') === 'Full lead sheet' && document.activeElement?.matches('[data-abcfoot] [data-scoreprev]'));
  } },
  { id: 'lyric-timing-says-what-it-needs', run: async t => {
    await desk('lab', ['score', 'planner'], { 'yue.timing': 'Hard' })(t);
    await t.eq('Hard says what it does and that it needs composer CFG 1', () => __b.text('.tile[data-m="score"] .ctl-row .lbl'), 'Hides a section\'s lyrics until its first vocal note. Needs composer CFG 1 (auto).');
    await t.do('composer CFG off auto', () => { S.vals['yue.cfg.auto'] = false; render(); setVal('yue.cfg', 2.1); });
    await t.ok('the line warns at once, in the warning\'s ink, with the fix (B-09)', () => { const l = document.querySelector('.tile[data-m="score"] .ctl-row .lbl'); return l.classList.contains('warn') && /It needs composer CFG 1, so your 2\.10 skips it\./.test(l.textContent) && !!l.querySelector('[data-auto="yue.cfg"]'); });
    await t.click('.tile[data-m="score"] [data-auto="yue.cfg"]');
    await t.ok('Set CFG to auto clears it', () => isAuto('yue.cfg') && !document.querySelector('.tile[data-m="score"] .ctl-row .lbl').classList.contains('warn'));
    /* 05-modules/lead-sheet's last check, with Soft set by its words (that scenario finds it by data-v, deviation 10) */
    await t.do('lyric timing soft', () => setVal('yue.timing', 'Soft'));
    await t.ok('Soft says what it does', () => /^Nudges each lyric block toward its section/.test(__b.text('.tile[data-m="score"] .ctl-row .lbl')));
    await t.do('lyric timing off', () => setVal('yue.timing', 'Off'));
    await t.eq('Off says nothing', () => __b.text('.tile[data-m="score"] .ctl-row .lbl'), '');
  } },
  { id: 'flow-knobs-and-switches-share-a-row', run: async t => {
    await desk('lab', ['flow'], {}, 'mm3')(t);
    await t.ok('the knobs and the switches stand in one row (D-11)', () => { const r = document.querySelector('.tile[data-m="flow"] .flow-row'); return !!r && r.children.length === 2 && r.children[0].matches('.knobs') && r.children[1].matches('.rackrow') && r.children[1].querySelectorAll('.led-btn').length === 3; });
  } },
  /* E-46 and E-47 (pending in the concept: it has neither, so these two do not hold there) */
  { id: 'build-plays-the-kept-takes-in-order', run: async t => {
    await desk('lab', ['build'])(t);
    await plan(t); await takes(t);
    await t.click(`${B} [data-bkeep="${await tid(t, 0)}"]`);
    await t.click(`${B} .b-src [data-bsec="Chorus#1"]`); await takes(t);
    await t.click(`${B} [data-bkeep="${await tid(t, 1)}"]`);
    const d = await t.js(() => S.build.takes['Verse#1'].find(x => x.id === S.build.kept['Verse#1']).dur);
    await t.click(`${B} [data-bplaykept]`, 300);
    await t.ok('Play kept plays the first section’s keep, its section lit, and becomes Stop', () => document.querySelector(`${B} .b-src .playing`)?.dataset.bsec === 'Verse#1' && document.querySelector(`${B} [data-bplaykept]`).getAttribute('aria-pressed') === 'true' && /Stop/.test(__b.text(`${B} [data-bplaykept]`)));
    await t.do('the verse plays out', () => {}, null, d * 1000);
    await t.ok('then the next section’s, its card playing where it is shown', () => document.querySelector(`${B} .b-src .playing`)?.dataset.bsec === 'Chorus#1' && document.querySelector(`${B} .b-take.playing`)?.dataset.take === S.build.kept['Chorus#1']);
    await t.click(`${B} [data-bplaykept]`);
    await t.ok('a second press stops', () => !document.querySelector(`${B} .b-src .playing`) && !document.querySelector(`${B} .b-take.playing`) && /Play kept/.test(__b.text(`${B} [data-bplaykept]`)));
  } },
  { id: 'recording-plays-its-chosen-part', run: async t => {
    await desk('lab', ['cont'], {}, 'yue', { contFile: 'take.wav', contDur: 200, contSel: [20, 40] })(t);
    await t.click('.tile[data-m="cont"] [data-contplay]', 300);
    await t.ok('the play button plays the chosen part: the playhead stands in it and moves on', () => { const h = document.querySelector('[data-contwave] .cw-head'); return !!h && +h.style.getPropertyValue('--p') >= 10 && +h.style.getPropertyValue('--p') <= 20 && document.querySelector('[data-contplay]').getAttribute('aria-pressed') === 'true'; });
    const p0 = await t.js(() => +document.querySelector('[data-contwave] .cw-head').style.getPropertyValue('--p'));
    await t.do('a few seconds', () => {}, null, 5000);
    await t.ok('the playhead has moved across the part', p0 => +document.querySelector('[data-contwave] .cw-head').style.getPropertyValue('--p') > p0, p0);
    await t.do('the part plays out', () => {}, null, 16000);
    await t.ok('and stops at its end', () => !document.querySelector('[data-contwave] .cw-head') && document.querySelector('[data-contplay]').getAttribute('aria-pressed') === 'false');
  } },
];
