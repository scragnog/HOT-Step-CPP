// Round 5, MiniMax-Music3 and YuE2 (review 3, E-01 to E-47): what the page says on these two engines and how their own
// modules look. The pending scenarios at the end state behaviours the page is known to get wrong (README, "Pending
// scenarios"): stale read-outs and focus, whose fix is script the rebuild replaces.
const desk = (tier, mods, extra) => t => t.do(`${tier} desk`, ([tier, mods, extra]) => { if (extra?.backend && extra.backend !== S.backend) enterEngine(extra.backend); S.tier = tier; S.mods = mods; S.pedals = extra?.pedals || []; if (extra?.vals) Object.assign(S.vals, extra.vals); if (extra?.slots) S.slots = extra.slots; render(); }, [tier, mods, extra]);
const planned = async t => { await t.do('two sections', () => setVal('lyrics.text', '[Verse]\nline\n[Chorus]\nline')); await t.click('.tile[data-m="build"] [data-bplan]'); await t.do('wait', () => {}, null, 1000); };

export default [
  /* E-02: on a two-column board each side unit takes the row, and nothing in it is cut or runs under its lip */
  ...['mm3', 'yue'].map(bk => ({ id: `side-units-fit-a-narrow-board-${bk}`, w: 1920, h: 1080, run: async t => {
    await desk('studio', ['post'], { backend: bk, pedals: ['ss', 'norm', 'q', bk === 'yue' ? 'fa' : 'lrc', 'wh', 'art'] })(t);
    await t.ok(`${bk}: the board is two columns wide`, () => { const w = document.querySelector('.tile[data-m="post"]').getBoundingClientRect().width; return w < 700 || w; });
    await t.ok('every side unit is a row of its own, its read-out whole and clear of its lip (E-02)', () => { const bad = [];
      for (const u of document.querySelectorAll('.tile[data-m="post"] .xunit')) { const b = u.getBoundingClientRect(), lip = u.querySelector('.u-lip')?.getBoundingClientRect(), board = u.parentElement.getBoundingClientRect();
        if (b.width < board.width - 80) bad.push(`${u.dataset.xid} ${Math.round(b.width)}px in ${Math.round(board.width)}px`);
        for (const c of u.querySelectorAll('.u-lcd, .u-status, .u-lcd > *')) { const r = c.getBoundingClientRect(); if (c.scrollWidth > c.clientWidth + 1) bad.push(`${u.dataset.xid} cut "${c.textContent.trim()}"`); if (lip && r.right > lip.left + 1) bad.push(`${u.dataset.xid} under the lip`); }
        if (b.height > 80) bad.push(`${u.dataset.xid} ${Math.round(b.height)}px tall`); }
      return !bad.length || bad.join('; '); });
  } })),
  /* E-05: one model of YuE2's adapter strength: Quick's Strength row is the AR cartridge's Strength fader */
  { id: 'yue-adapter-strength-is-one-control', run: async t => {
    await desk('quick', ['adapters'], { backend: 'yue', slots: [{ file: 'folk-voice-ar', type: 'LoRA' }] })(t);
    await t.click('.tile[data-m="adapters"] [data-setq="s0|1.5"]');
    await t.do('studio', () => { S.tier = 'studio'; render(); });
    await t.ok('in Studio the AR cartridge shows that Strength beside Plan and Compose (E-05)', () => { const f = [...document.querySelectorAll('.tile[data-m="adapters"] .fdr-pair .fdr')].map(x => x.querySelector('.fdr-l')?.textContent + '=' + x.querySelector('output')?.textContent); return (f.length === 3 && /^Strength=1\.50$/.test(f[0])) || f.join(', '); });
  } },
  /* E-10, E-27 */
  { id: 'yue-adapters-say-what-yue-does', run: async t => {
    await desk('studio', ['adapters'], { backend: 'yue', slots: [{ file: 'folk-voice-ar', type: 'LoRA' }] })(t);
    await t.ok('one AR adapter: the line names the NAR slot, not blending or lyric sections (E-10)', () => { const x = document.querySelector('.tile[data-m="adapters"]').textContent; return (/second slot takes a NAR renderer adapter/.test(x) && !/blend them|lyric section/.test(x)) || x; });
    await t.do('lab, two adapters', () => { S.tier = 'lab'; S.slots.push({ file: 'tape-warmth-nar', type: 'LoRA' }); render(); });
    await t.ok('the bank says both adapters, and the foot stands under the bank, not beside it (E-27)', () => { const tile = document.querySelector('.tile[data-m="adapters"]'), mh = tile.querySelector('.rig-mh small')?.textContent, bank = tile.querySelector('.rig-master').getBoundingClientRect(), foot = tile.querySelector('[data-adfoot]').getBoundingClientRect();
      return (mh === 'both adapters' && foot.top >= bank.bottom - 1 && !tile.querySelector('.rig-wrap [data-adfoot]')) || JSON.stringify({ mh, bankBottom: bank.bottom, footTop: foot.top }); });
  } },
  /* E-07, E-29: the Tempo tile on MiniMax-Music3 at Quick says how a BPM reaches the song, and is named for what it shows */
  { id: 'mm3-quick-tempo-says-where-tempo-goes', run: async t => {
    await desk('quick', ['tempo'], { backend: 'mm3' })(t);
    await t.ok('the tile is named Tempo and says the caption carries it (E-07, E-29)', () => { const tl = document.querySelector('.tile[data-m="tempo"]'); return (/^Tempo$/.test(tl.querySelector('.tile-head h3, h3')?.textContent.trim()) && /reads tempo from the caption/.test(tl.textContent)) || tl.textContent.slice(0, 160); });
    await t.do('a style', () => { document.querySelector('#prompt').value = 'dream pop'; });
    await t.focus('.tile[data-m="tempo"] [data-num="tempo.bpm"]'); await t.type('96'); await t.press('Enter');
    await t.click('.tile[data-m="tempo"] [data-composecap]');
    await t.ok('Compose caption from the tile writes the BPM in', () => /^Global Metadata: dream pop, 96 BPM/.test(document.querySelector('#prompt').value) || document.querySelector('#prompt').value);
  } },
  /* E-14: the lyrics language reaches MiniMax-Music3 through Compose caption, and its stamp says so */
  { id: 'mm3-lyrics-language-is-composed-in', run: async t => {
    await desk('studio', [], { backend: 'mm3', vals: { 'lyrics.lang': 'Japanese' } })(t);
    await t.ok('the stamp says what the language does here (E-14)', () => /Compose caption writes the language/.test(document.querySelector('.paper-wrap .stamp-wrap button')?.title || '') || document.querySelector('.paper-wrap .stamp-wrap button')?.title);
    await t.do('a style', () => { document.querySelector('#prompt').value = 'city pop'; });
    await t.click('#structBtn');
    await t.ok('Compose caption writes it into Vocal Details', () => /Vocal Details: .*sung in Japanese/.test(document.querySelector('#prompt').value) || document.querySelector('#prompt').value);
    await t.ok('and its toast writes the button\'s name in sentence case (E-21)', () => /^Compose caption built/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) || (document.querySelector('#toast .toast-msg')?.textContent || ''));
  } },
  /* E-08, E-22: the dataset is chosen on both engines, and a dataset change that orphans the track says so, for good */
  { id: 'caption-dataset-change-is-said', run: async t => {
    await desk('studio', [], { backend: 'mm3', vals: { 'cap.src': 'Track 04 · Paper boats' } })(t);
    await t.ok('MiniMax-Music3 has the "From" chip, and no sentence about an adapter\'s dataset (E-08)', () => (!!document.querySelector('#capSrc [data-pick="cap.ds"]') && !/adapter learned/.test(document.querySelector('#capSrc').innerHTML + document.querySelector('#capLock').innerHTML)) || document.querySelector('#capSrc').innerHTML.slice(0, 200));
    await t.click('#capSrc [data-pick="cap.ds"]'); await t.click('#dd [data-v="folk-sessions"], #dd [data-opt="folk-sessions"]');
    await t.eq('the track is not in that dataset: the source is Custom, stored (E-22)', () => S.vals['cap.src'], 'Custom');
    await t.ok('and the page says so', () => /Paper boats isn't in folk-sessions/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) || (document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.click('#capSrc [data-pick="cap.ds"]'); await t.click('#dd [data-v="my-band-ep"], #dd [data-opt="my-band-ep"]');
    await t.ok('back on the first dataset nothing re-locks', () => !document.querySelector('#prompt').readOnly);
  } },
  { id: 'quick-hides-the-unused-dataset-chip', run: async t => {
    await desk('quick', [], { backend: 'yue' })(t);
    await t.ok('at Quick with the source on Custom there is no "From" chip (E-22)', () => !document.querySelector('#capSrc [data-pick="cap.ds"]'));
  } },
  /* E-11, E-19 */
  { id: 'engine-switch-toast-is-short', run: async t => {
    await t.do('a loaded desk', () => { S.tier = 'studio'; S.mods = ['tempo', 'takes', 'adapters', 'score']; S.vals['takes.n'] = 9; S.slots = [{ file: 'tape-lofi-v2', type: 'LoRA' }]; render(); });
    await t.do('to YuE2', () => switchEngine('yue'), null, 300);
    await t.ok('the toast is a few short sentences, and names no module already on the desk (E-11)', () => { const m = (document.querySelector('#toast .toast-msg')?.textContent || ''); return (m.split(/\s+/).length <= 32 && /New here: .* more, in Modules/.test(m) && !/Lead sheet/.test(m)) || m; });
    await t.do('back to ACE-Step', () => switchEngine('ace'), null, 300);
    await t.ok('coming back to the engine the page started on is not news (E-19)', () => !/New here/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) || (document.querySelector('#toast .toast-msg')?.textContent || ''));
  } },
  /* E-12, E-26, E-40: the lead sheet says what the Planner is set to, chords are drawn for a full lead sheet, titles start with a capital */
  { id: 'lead-sheet-follows-the-planner', run: async t => {
    await desk('studio', ['score', 'planner'], { backend: 'yue', vals: { 'lm.cotmode': 'Off' } })(t);
    await t.ok('with Chain of Thought off the tile says the Planner writes nothing down, and offers no plan to preview (E-12)', () => { const f = document.querySelector('[data-abcfoot]'); return (/write nothing down/.test(f.textContent) && !f.querySelector('[data-scoreprev]') && !!f.querySelector('[data-bcot]')) || f.textContent; });
    await t.click('[data-abcfoot] [data-bcot]');
    await t.ok('Write a full score turns it back on and focuses the preview', () => (get('lm.cotmode') === 'Full lead sheet' && document.activeElement?.matches('[data-abcfoot] [data-scoreprev]')) || get('lm.cotmode') + ' ' + document.activeElement?.outerHTML.slice(0, 60));
    await t.do('a style', () => { document.querySelector('#prompt').value = 'folk rock, banjo'; });
    await t.click('[data-abcfoot] [data-scoreprev]', 600);
    await t.ok('a full lead sheet carries chord symbols, and its title starts with a capital (E-12, E-40)', () => (document.querySelectorAll('.sc-card .st-chord').length > 0 && document.querySelector('.sc-card h3').textContent === 'Folk rock') || `${document.querySelectorAll('.sc-card .st-chord').length} chords, "${document.querySelector('.sc-card h3')?.textContent}"`);
    await t.press('Escape');
    await t.do('melody only', () => setVal('lm.cotmode', 'Melody only'));
    await t.click('[data-abcfoot] [data-scoreprev]', 600);
    await t.eq('Melody only draws no chords', () => document.querySelectorAll('.sc-card .st-chord').length, 0);
    await t.press('Escape');
    await t.click('.tile[data-m="score"] [data-bind="yue.abc"]'); await t.type('<b>not abc</b>');
    await t.ok('text that is not ABC is said to be so (E-26)', () => /doesn't read as ABC/.test(document.querySelector('[data-abcfoot]').textContent) || document.querySelector('[data-abcfoot]').textContent);
  } },
  /* E-13: composer CFG on auto reads one way wherever it shows, and its token explains itself */
  { id: 'composer-cfg-reads-auto-one-way', run: async t => {
    await desk('lab', ['planner'], { backend: 'yue' })(t);
    const tokText = () => document.querySelector('[data-scrub="yue.cfg"]').textContent;
    await t.eq('after a render it reads "auto (1.00)" (E-13)', tokText, 'auto (1.00)');
    await t.focus('[data-scrub="yue.cfg"]'); await t.press('ArrowUp'); await t.dblclick('[data-scrub="yue.cfg"]');
    await t.eq('after a reset by double-click, the same', tokText, 'auto (1.00)');
    await t.ok('and its tip is the setting\'s explanation', () => /checkpoint's own 1\.00/.test(document.querySelector('[data-scrub="yue.cfg"]').title) || document.querySelector('[data-scrub="yue.cfg"]').title);
  } },
  /* E-15, E-16, E-17, E-36: Plans & endings */
  { id: 'plans-say-one-thing', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['planner', 'plans'], { backend: 'mm3' })(t);
    await t.click('.tile[data-m="plans"] [data-ledtog="mm.reuse"]');
    await t.eq('the terminal says the cache is off (E-15)', () => document.querySelector('[data-mmcache]').textContent, '8B planner · plan cache off');
    await t.click('.tile[data-m="plans"] [data-cycle="mm.plank|1"]');
    await t.eq('and names the Plank it replays', () => document.querySelector('[data-mmcache]').textContent, '8B planner · replaying 11-19 22.41');
    await t.ok('the replay LCD shows the whole name (E-16)', () => { const b = document.querySelector('.tile[data-m="plans"] [data-pick="mm.plank"] b'); return (b.scrollWidth <= b.clientWidth + 1 && b.textContent === '11-19 22.41') || `${b.textContent} ${b.scrollWidth}/${b.clientWidth}`; });
    await t.ok('every Endings chip explains itself (E-17)', () => [...document.querySelectorAll('.tile[data-m="plans"] .endings .led-btn')].every(b => b.title.length > 20));
    await t.ok('changed rows carry their mark after the row label, and controls carry none (E-36)', () => { const lab = [...document.querySelectorAll('.tile[data-m="plans"] .deck-l')].filter(l => getComputedStyle(l, '::after').content !== 'none' && getComputedStyle(l, '::after').width === '5px').map(l => l.textContent);
      const ctl = [...document.querySelectorAll('.tile[data-m="plans"] .deck-c .changed')].filter(c => { const s = getComputedStyle(c, '::after'); return s.content !== 'none' && s.visibility !== 'hidden'; }).length; return (lab.includes('Last plan') && lab.includes('Replay') && !ctl) || JSON.stringify({ lab, ctl }); });
  } },
  /* E-31: values in force on Build by section are at full strength */
  { id: 'build-locked-values-read', run: async t => {
    await desk('lab', ['build'], { backend: 'yue' })(t);
    await planned(t);
    await t.click('.tile[data-m="build"] [data-btake]'); await t.do('wait', () => {}, null, 2000);
    const tid = await t.js(() => S.build.takes['Verse#1'][0].id);
    await t.click(`.tile[data-m="build"] [data-bkeep="${tid}"]`);
    await t.ok('a locked value and its label are not faded (E-31)', () => { const f = document.querySelector('.tile[data-m="build"] .foot-ctl.locked'); return (f && getComputedStyle(f).opacity === '1') || (f ? getComputedStyle(f).opacity : 'no locked control'); });
  } },
  /* E-33 */
  { id: 'planner-seed-keeps-what-was-typed', run: async t => {
    await desk('lab', ['planner'], { backend: 'mm3' })(t);
    await t.click('[data-bind="mm.seed"]'); await t.type('abc-12'); await t.press('Enter');
    await t.ok('a refused seed stays in the field, marked (E-33)', () => { const f = document.querySelector('[data-bind="mm.seed"]'); return (f.value === 'abc-12' && f.classList.contains('bad')) || `${f.value} ${f.className}`; });
  } },
  /* E-38 */
  { id: 'yue-offers-the-tempo-field', run: async t => {
    await desk('studio', ['tempo'], { backend: 'yue' })(t);
    await t.click('#prompt'); await t.type('slow folk, 92 BPM');
    await t.click('#suggest [data-sg="bpm"]');
    await t.eq('YuE2 offers to set the tempo field, as ACE-Step does (E-38)', () => +get('tempo.bpm'), 92);
  } },

  /* E-20: Reuse puts back the text the song was made from, never its card line with the artist in it */
  { id: 'reuse-gives-back-the-source-text', run: async t => {
    await desk('studio', [], { backend: 'mm3', vals: { 'song.artist': 'The Kettles' } })(t);
    await t.do('a caption', () => { document.querySelector('#prompt').value = 'Global Metadata: city pop, bright\nVocal Details: female lead'; });
    await t.click('#go'); await t.do('let it finish', () => {}, null, 12000);
    const id = await t.js(() => S.songs.find(s => s.bk === 'mm3' && !s.pending).id);
    await t.do('another caption', () => { document.querySelector('#prompt').value = 'something else'; });
    await t.click(`#latest .song[data-id="${id}"] [data-songmenu]`);
    await t.ok('the menu names the caption (E-20)', () => /Reuse its caption/.test(document.querySelector('#menu [data-sa="reuse"]').textContent));
    await t.click('#menu [data-sa="reuse"]');
    await t.eq('and Reuse puts back the whole Structured Caption, no artist folded in', () => document.querySelector('#prompt').value, 'Global Metadata: city pop, bright\nVocal Details: female lead');
  } },

  /* E-25: Build by section at Quick names the tier its settings are in, strip or not */
  { id: 'build-at-quick-names-studio', run: async t => {
    await desk('quick', ['build'], { backend: 'yue', vals: { 'yue.btakes': 3 } })(t);
    await t.ok('as a strip (no tags) it says Edit in Studio for Takes per section (E-25)', () => /Takes per section 3\. Edit in Studio/.test(document.querySelector('.tile[data-m="build"] .q-also')?.textContent || '') || document.querySelector('.tile[data-m="build"] .q-also')?.textContent);
  } },
  /* E-32: the off rail of MiniMax-Music3's top-p is on the wide edge */
  { id: 'mm3-topp-off-is-on-the-wide-edge', run: async t => {
    await desk('lab', ['planner'], { backend: 'mm3' })(t);
    await t.ok('off is a rail along the top, under CALM & WIDE (E-32)', () => { const x = document.querySelector('.xy[data-yoff]'), r = x.querySelector('.xy-rail').getBoundingClientRect(), p = x.getBoundingClientRect(), d = x.querySelector('.xy-dot').getBoundingClientRect();
      return (r.top - p.top < 14 && d.top >= p.top - 1 && Math.abs((d.top + d.bottom) / 2 - (r.top + r.bottom) / 2) < 4) || JSON.stringify([r.top - p.top, d.top - p.top]); });
  } },

  /* ---- pending: behaviours the page gets wrong today; each fails now, for its finding's reason ---- */
  { id: 'takes-tile-follows-retry', pending: { finding: 'E-03', right: 'When Takes changes from anywhere (Retry with N takes, the go bar\'s Use N takes, Undo, a workflow), the Takes tile shows the new count and drops a warning that no longer holds, at once.' }, run: async t => {
    await t.do('MiniMax-Music3, too big', () => { enterEngine('mm3'); S.tier = 'studio'; S.mods = ['takes']; S.vals['takes.n'] = 4; S.vals['model.lm'] = '8B planner · F16'; S.vals['model.dit'] = 'Flow DiT · F16'; document.querySelector('#prompt').value = 'big song'; render(); });
    await t.click('#go', 300);
    await t.click('#latest .song.errored [data-retry]', 300);
    await t.ok('after Retry with fewer takes the tile shows the new count and no "Won\'t fit"', () => { const tl = document.querySelector('.tile[data-m="takes"]'), n = get('takes.n'); return (!/Won't fit/.test(tl.textContent) && tl.querySelector('.tk-lcd, .lcd, output')?.textContent.includes(String(n))) || `takes.n ${n}; tile "${tl.textContent.replace(/\s+/g, ' ').trim().slice(0, 120)}"`; });
  } },
  { id: 'automatic-source-follows-the-tempo', pending: { finding: 'E-06', right: 'With the caption or description source on Automatic, a change of BPM (or of its auto switch) re-resolves the dataset track at once: the lock line, the text and the chip\'s tip name the track nearest the new tempo.' }, run: async t => {
    await desk('studio', ['tempo'], { backend: 'yue', vals: { 'cap.src': 'Automatic' } })(t);
    await t.ok('at the default tempo Automatic resolves to a track', () => !!document.querySelector('#capLock:not([hidden])'));
    await t.focus('.tile[data-m="tempo"] [data-num="tempo.bpm"]'); await t.type('118'); await t.press('Enter');
    await t.ok('at 118 BPM the lock line names Track 04 (118 BPM)', () => /Track 04/.test(document.querySelector('#capLock').textContent) || document.querySelector('#capLock').textContent);
  } },
  { id: 'build-button-follows-takes', pending: { finding: 'E-09', right: 'Build by section\'s primary button names the number of takes it will make, following the Takes value on every step.' }, run: async t => {
    await desk('lab', ['build'], { backend: 'yue' })(t);
    await planned(t);
    await t.focus('.tile[data-m="build"] [data-scrub="yue.btakes"]'); await t.press('ArrowRight'); await t.press('ArrowRight');
    await t.ok('the button says the takes it will make', () => { const n = get('yue.btakes'), b = document.querySelector('.tile[data-m="build"] [data-btake]').textContent; return b.includes(`${n} takes`) || `takes ${n}, button "${b}"`; });
  } },
  { id: 'discarding-the-only-keep-frees-the-locks', pending: { finding: 'E-24', right: 'Discarding the last kept take frees Follow score and Plan strength, exactly as letting go of the keep does.' }, run: async t => {
    await desk('lab', ['build'], { backend: 'yue' })(t);
    await planned(t);
    await t.click('.tile[data-m="build"] [data-btake]'); await t.do('wait', () => {}, null, 2000);
    const tid = await t.js(() => S.build.takes['Verse#1'][0].id);
    await t.click(`.tile[data-m="build"] [data-bkeep="${tid}"]`);
    await t.do('back to the verse', () => { S.build.cur = 'Verse#1'; redrawBuild(); });
    await t.click(`.tile[data-m="build"] [data-bdrop="${tid}"]`);
    await t.ok('nothing is kept, so nothing is locked', () => (!Object.keys(S.build.kept).length && !S.build.locked && !document.querySelector('.tile[data-m="build"] .foot-ctl.locked')) || `kept ${Object.keys(S.build.kept).length}, locked ${S.build.locked}`);
  } },
  { id: 'focus-returns-to-the-recording', pending: { finding: 'E-23', right: 'After a recording is chosen for Continue from recording (or replaced), focus is on its waveform, and a toast says what was loaded with Undo.' }, run: async t => {
    await desk('lab', ['cont'], { backend: 'yue' })(t);
    await t.file('.tile[data-m="cont"] input[type=file]', 'second.flac', 'audio/flac', 'x');
    await t.ok('focus is on the waveform', () => document.activeElement?.matches('[data-contwave]') || document.activeElement?.tagName);
    await t.ok('and a toast names the file, with Undo', () => (/second\.flac/.test((document.querySelector('#toast .toast-msg')?.textContent || '')) && !!document.querySelector('#toast .toast-act')) || (document.querySelector('#toast .toast-msg')?.textContent || ''));
  } },
  /* additions the review asked for: stated as behaviour, so the rebuild has them */
  { id: 'build-plays-the-kept-takes-in-order', pending: { finding: 'E-46', right: 'Build by section has "Play kept" beside Render: it plays every kept take in section order, the section strip lighting the one playing, and a second press stops.' }, run: async t => {
    await desk('lab', ['build'], { backend: 'yue' })(t);
    await planned(t);
    await t.click('.tile[data-m="build"] [data-btake]'); await t.do('wait', () => {}, null, 2000);
    const tid = await t.js(() => S.build.takes['Verse#1'][0].id);
    await t.click(`.tile[data-m="build"] [data-bkeep="${tid}"]`);
    await t.ok('a Play kept button stands beside Render', () => !!document.querySelector('.tile[data-m="build"] [data-bplaykept]') || 'none');
  } },
  { id: 'continue-plays-the-chosen-part', pending: { finding: 'E-47', right: 'Continue from recording has a play button that plays the selected part of the recording, its playhead moving across the selection.' }, run: async t => {
    await desk('lab', ['cont'], { backend: 'yue' })(t);
    await t.file('.tile[data-m="cont"] input[type=file]', 'first.flac', 'audio/flac', 'x');
    await t.ok('a play button for the chosen part', () => !!document.querySelector('.tile[data-m="cont"] [data-contplay]') || 'none');
  } },
];
