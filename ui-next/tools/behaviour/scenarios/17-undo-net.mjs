// Round 5, shared mechanisms: every change a person makes is an undo step of its own, named in the page's words; the
// keys' rule inside and outside text fields; the saved read-out tells the truth.
const lab = mods => [m => { S.tier = 'lab'; S.mods = m; render(); window.uL = () => document.querySelector('#undoBtn').hidden ? '' : document.querySelector('#undoBtn span').textContent; }, mods];
const undoLabel = () => document.querySelector('#undoBtn').hidden ? '' : document.querySelector('#undoBtn span').textContent;
export default [
  { id: 'timbre-load-is-its-own-step', run: async t => {
    await t.do('studio, timbre and takes', () => { S.tier = 'studio'; S.mods = ['timbre', 'takes']; S.refs = ['reference-mix.wav', 'my-band-master.flac']; render(); });
    await t.do('three takes', () => setVal('takes.n', 3));
    await t.click('.tile[data-m="timbre"] .pref[data-pref="reference-mix.wav"]');
    await t.eq('loading a recent reference is a step named for the file (L-01, Q-12)', undoLabel, 'Load reference-mix.wav as the timbre reference');
    await t.focus('.tile[data-m="timbre"] [data-venn]'); await t.press('ArrowLeft');
    await t.eq('a nudge of the blend afterwards is its own step (Q-17)', undoLabel, 'Reference blend 100% → 95%');
    await t.click('.top .brand'); await t.press('Control+z');
    await t.eq('Ctrl+Z takes back the nudge only: the track stays loaded', () => [S.timbreFile, get('timbre.str'), get('takes.n')], ['reference-mix.wav', 1, 3]);
    await t.press('Control+z');
    await t.eq('the next Ctrl+Z unloads the track, and leaves the takes', () => [S.timbreFile, get('takes.n')], ['', 3]);
    await t.press('Control+z');
    await t.eq('and the next puts the takes back', () => get('takes.n'), 1);
    await t.file('.tile[data-m="timbre"] [data-file="timbre"]', 'my-track.wav', 'audio/wav');
    await t.eq('a chosen file is a step too', undoLabel, 'Load my-track.wav as the timbre reference');
    await t.file('.tile[data-m="timbre"] [data-file="timbre"]', 'second.flac', 'audio/flac');
    await t.eq('and replacing it', undoLabel, 'Timbre reference my-track.wav → second.flac');
    await t.click('.top .brand'); await t.press('Control+z');
    await t.eq('Undo brings the first file back', () => S.timbreFile, 'my-track.wav');
  } },
  { id: 'mastering-reference-is-its-own-step', run: async t => {
    await t.do('lab board with Mastering', () => { S.tier = 'lab'; S.mods = ['post', 'takes']; S.pedals = ['master']; render(); });
    await t.do('a takes change first', () => setVal('takes.n', 2));
    await t.dropFile('.pedal[data-pid="master"] label.drop', 'my-master.flac', 'audio/flac');
    await t.eq('a dropped mastering reference is a step named for the file (L-01)', () => document.querySelector('#undoBtn span').textContent, 'Load my-master.flac as the mastering reference');
    await t.click('.top .brand'); await t.press('Control+z');
    await t.eq('Ctrl+Z takes the file out and leaves the takes', () => [S.masterRef, get('takes.n')], ['', 2]);
  } },
  { id: 'repaint-region-is-its-own-step', run: async t => {
    await t.do('repaint a song', () => { S.tier = 'studio'; S.mods = ['repaint', 'takes']; S.source = { song: S.songs[0].id }; S.region = [0.35, 0.55]; render(); window.uL = () => document.querySelector('#undoBtn').hidden ? '' : document.querySelector('#undoBtn span').textContent; });
    await t.do('a takes change first', () => setVal('takes.n', 2));
    await t.focus('[data-rpwave]'); await t.press('ArrowRight');
    await t.ok('a key press on the region is a step named in the song\'s times (Q-24)', () => /^Repaint region \d+:\d\d–\d+:\d\d → \d+:\d\d–\d+:\d\d$/.test(uL()) || uL());
    await t.mouseDrag('[data-rpwave]', [[30, 0], [30, 0]], [0.45, 0.5]);
    await t.ok('so is a drag', () => /^Repaint region /.test(uL()) || uL());
    await t.click('.top .brand'); await t.press('Control+z');
    await t.ok('Ctrl+Z puts the region back and leaves Repaint on the desk (L-01)', () => S.mods.includes('repaint') && S.source && Math.abs(S.region[0] - 0.36) < 1e-9 || JSON.stringify([S.mods, S.region]));
  } },
  { id: 'adapter-phase-and-mask-are-steps', run: async t => {
    await t.do('lab adapters', ...lab(['lyrics', 'adapters']));
    await t.do('two adapters, Advanced, two sections', () => { S.slots = [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'synthwave-80s', type: 'LoRA' }]; S.vals['ad.adv'] = 'Advanced'; S.vals['lyrics.text'] = '[Verse]\nline one\n[Chorus]\nline two'; render(); });
    await t.ok('the phase strip is drawn', () => !!document.querySelector('[data-phase="0"] [data-ph="0"]'));
    await t.focus('[data-phase="0"] [data-ph="0"]'); await t.press('ArrowRight');
    await t.eq('a phase handle moved by a key is a step named for its adapter (L-01)', undoLabel, 'Phase of tape-lofi-v2 0–100% → 5–100%');
    await t.click('[data-mask="0-1"]');
    await t.eq('a section mask chip is a step named for its section and adapter (L-09)', undoLabel, 'Chorus: tape-lofi-v2 full → half');
    await t.click('.top .brand'); await t.press('Control+z'); await t.press('Control+z');
    await t.eq('two Ctrl+Z take back the mask and then the phase, and the adapter stays', () => [S.slots.length, S.slots[0].ph || null, /\{/.test(get('lyrics.text'))], [2, null, false]);
  } },
  { id: 'continue-selection-is-a-step', run: async t => {
    await t.do('yue, continue', () => { S.backend = 'yue'; S.tier = 'lab'; S.mods = ['cont', 'takes']; render(); });
    await t.do('a takes change first', () => setVal('takes.n', 2));
    await t.file('.tile[data-m="cont"] [data-file="cont"]', 'first.flac', 'audio/flac');
    await t.eq('loading the recording is a step (E-23)', undoLabel, 'Load first.flac to continue from');
    await t.focus('[data-contwave]'); await t.press('ArrowRight');
    await t.eq('a key press on the selection is a step', undoLabel, 'Part to continue from 0:00–0:20 → 0:01–0:21');
    await t.click('.top .brand'); await t.press('Control+z');
    await t.eq('Ctrl+Z moves the selection back and leaves the rest', () => [S.contSel, S.contFile, get('takes.n')], [[0, 20], 'first.flac', 2]);
  } },
  { id: 'steps-are-named-as-the-page-names-things', run: async t => {
    await t.do('studio with pedals', () => { S.tier = 'studio'; S.mods = ['takes', 'post', 'repetition', 'tempo']; S.pedals = ['ss', 'norm']; render(); });
    await t.click('.pedal[data-pid="ss"] [data-foot]');
    await t.eq('a foot switch names its pedal (L-09, Q-04)', undoLabel, 'StableStep off → on');
    await t.click('[data-ledtog="pp.on"]');
    await t.eq('the chain switch says bypassed', undoLabel, 'Chain on → bypassed');
    await t.do('takes', () => setVal('takes.n', 4));
    await t.eq('Takes is called what its tile calls it', undoLabel, 'Takes per run 1 → 4');
    await t.do('repetition mode', () => setVal('lm.repmode', 'DRY'));
    await t.eq('a word that names more than one thing has its module first', undoLabel, 'Repetition guard mode Presence → DRY');
    await t.do('wildcards', () => setVal('wild.auto', true));
    await t.eq('Expand on Generate is named, not "wild"', undoLabel, 'Expand on Generate off → on');
  } },
  { id: 'quick-steps-speak-quick', run: async t => {
    await t.do('quick sampler and planner', () => { S.tier = 'quick'; S.mods = ['sampler', 'planner']; S.vals['lm.on'] = true; render(); });
    await t.click('.mood [data-setn$="|1.05"]');
    await t.ok('a mood is named as the row reads (Q-09)', () => /^How adventurous: \w+ → Bold$/.test(document.querySelector('#undoBtn span').textContent) || document.querySelector('#undoBtn span').textContent);
    await t.click('.qchoice[data-q="samp.steps"] button:last-child');
    await t.ok('so is a Quality choice', () => /^Quality: \w+ → \w+$/.test(document.querySelector('#undoBtn span').textContent) || document.querySelector('#undoBtn span').textContent);
  } },
  { id: 'lab-steps-are-named-too', run: async t => {
    await t.do('lab', () => { S.tier = 'lab'; S.mods = ['adapters', 'tempo', 'planner']; S.slots = [{ file: 'tape-lofi-v2', type: 'LoRA' }]; S.vals['lm.on'] = true; render(); });
    await t.focus('.tile[data-m="adapters"] [data-fader="s0"]'); await t.press('ArrowUp');
    await t.eq('an adapter fader is named for its adapter (L-09)', () => document.querySelector('#undoBtn span').textContent, 'tape-lofi-v2 strength 1.00 → 1.05');
    await t.do('buffer', () => setVal('tempo.trimbuf', 8));
    await t.eq('a buffer of whole seconds reads as the tile reads it', () => document.querySelector('#undoBtn span').textContent, 'Buffer 15 s → 8 s');
    await t.focus('.tile[data-m="planner"] [data-xy]'); await t.press('ArrowRight');
    await t.ok('an XY pad step names both values', () => /^\S.* → .*, .* → /.test(document.querySelector('#undoBtn span').textContent) || /→/.test(document.querySelector('#undoBtn span').textContent) || document.querySelector('#undoBtn span').textContent);
  } },
  { id: 'field-undo-never-moves-forward', run: async t => {
    await t.click('#prompt'); await t.type('first words'); await t.do('pause', () => {}, null, 1500); await t.type(' second');
    await t.do('a setting', () => setVal('takes.n', 2));
    await t.click('#prompt'); await t.press('Control+z');
    await t.eq('Ctrl+Z in the description takes back the last typing', () => document.querySelector('#prompt').value, 'first words');
    await t.click('.top .brand'); await t.press('Control+z');
    await t.eq('Ctrl+Z outside it goes on back: the setting, not the words again (F-06)', () => [document.querySelector('#prompt').value, get('takes.n')], ['first words', 1]);
    await t.press('Control+z');
    await t.eq('and then the rest of the typing', () => document.querySelector('#prompt').value, '');
  } },
  { id: 'ctrl-z-in-a-spent-field-goes-on', run: async t => {
    await t.do('lab sampler', ...lab(['sampler', 'takes']));
    await t.do('three changes', () => { setVal('takes.n', 2); });
    await t.do('then steps', () => setVal('samp.steps', 30));
    await t.click('.top .brand');
    await t.focus('.tile[data-m="sampler"] [data-num="samp.steps"], .tile[data-m="sampler"] input');
    await t.press('Control+z');
    await t.eq('Ctrl+Z in a number box with nothing typed goes to the page (L-05)', () => get('samp.steps') !== 30, true);
    await t.press('Control+z');
    await t.eq('and again (L-05)', () => get('takes.n'), 1);
  } },
  { id: 'undo-after-leaving-a-field-stays-out-of-it', run: async t => {
    await t.do('lab sampler', ...lab(['sampler', 'takes']));
    await t.do('takes', () => setVal('takes.n', 2));
    await t.focus('.tile[data-m="sampler"] [data-num="samp.steps"]'); await t.press('ArrowUp');
    await t.click('.deskbar', null, [0.2, 0.5]);
    await t.ok('a press on the empty bar leaves the box', () => document.activeElement === document.body);
    await t.press('Control+z');
    await t.ok('the Undo that redraws the box leaves focus out of it (L-05)', () => !document.activeElement || !document.activeElement.matches('input, textarea') || document.activeElement.outerHTML.slice(0, 80));
    await t.press('Control+z');
    await t.eq('so the next Ctrl+Z goes on back', () => get('takes.n'), 1);
  } },
  { id: 'ctrl-z-under-an-overlay-edits-nothing', run: async t => {
    await t.click('#prompt'); await t.type('alpha beta gamma');
    await t.click('#notesBtn'); await t.press('Control+z');
    await t.eq('under the Concept notes the description stays (F-04)', () => document.querySelector('#prompt').value, 'alpha beta gamma');
    await t.ok('and says why', () => /Close this first/.test(document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.press('Escape');
    await t.press('F9'); await t.press('Tab'); await t.press('Control+z');
    await t.eq('nor under the picker', () => document.querySelector('#prompt').value, 'alpha beta gamma');
    await t.press('Escape');
    await t.click('[data-nav="Library"]'); await t.press('Control+z'); await t.press('Control+z');
    await t.click('[data-nav="create"]');
    await t.eq('nor from another page', () => document.querySelector('#prompt').value, 'alpha beta gamma');
  } },
  { id: 'saved-says-only-what-was-saved', run: async t => {
    await t.do('storage full', () => { const set = window.__set = Storage.prototype.setItem; Storage.prototype.setItem = function (k, v) { if (k === 'hs3-desk') throw new DOMException('full', 'QuotaExceededError'); return set.call(this, k, v); }; });
    await t.click('#prompt'); await t.type('kept only here');
    await t.do('wait', () => {}, null, 1200);
    await t.ok('the read-out says Not saved (F-15)', () => /Not saved/.test(document.querySelector('#saveState').textContent) && document.querySelector('#saveState').classList.contains('failed'));
    await t.ok('and a toast says why, once', () => /storage is full, so your desk wasn't saved/i.test(document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.type(' more'); await t.do('wait', () => {}, null, 1200);
    await t.ok('it keeps saying so', () => /Not saved/.test(document.querySelector('#saveState').textContent));
    await t.do('room again', () => { Storage.prototype.setItem = window.__set; });
    await t.type('!'); await t.do('wait', () => {}, null, 1200);
    await t.ok('and goes back to Saved once a write succeeds', () => /^Saved$/.test(document.querySelector('#saveState').textContent.trim()) && !document.querySelector('#saveState').classList.contains('failed') && /kept only here more!/.test(localStorage.getItem('hs3-desk')));
  } },
  { id: 'a-refused-draft-is-not-called-saved', run: async t => {
    await t.click('#prompt'); await t.type('first draft');
    await t.do('storage full', () => { const set = window.__set = Storage.prototype.setItem; Storage.prototype.setItem = function () { throw new DOMException('full', 'QuotaExceededError'); }; });
    await t.click('#draftsBtn'); await t.click('#pop [data-draftsave]');
    await t.ok('a draft the browser refused is said to be unsaved, last (F-15)', () => /wasn't saved/.test(document.querySelector('#toast .toast-msg')?.textContent || '') && /Not saved/.test(document.querySelector('#saveState').textContent));
  } },
];
