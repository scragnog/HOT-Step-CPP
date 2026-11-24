// The prompt card and the lyric sheet (package 4), as scenarios that hold on the concept page too (run them there with
// --page): each text field's own Ctrl+Z and the page's Undo together, Ctrl+Z in a field that has nothing of its own
// left, the field drawn again keeping its history, the lyric sheet's tabs, strip, stamp and Instrumental, how it grows
// and scrolls, the caption a dataset supplies, and the tools that need no popover. Several are the concept's own
// scenarios with the one step they share with a later package (the popover's presets) replaced by a step of this
// package's (Expand on the box), or with an option found by its value as well as its words (deviation 10: the concept's
// data-opt and data-v are labels, the rebuild's values). Each uses only what both pages share.

const desk = (tier, mods, vals = {}, backend = 'ace') => t => t.do(`${backend} ${tier} desk`, ([tier, mods, vals, backend]) => {
  if (backend !== S.backend) { if (typeof enterEngine === 'function') enterEngine(backend); else S.backend = backend; }
  S.backend = backend; S.tier = tier; S.mods = mods; S.pedals = []; Object.assign(S.vals, vals); render();
}, [tier, mods, vals, backend]);
const LYR = '[data-bind="lyrics.text"]';

export default [
  /* 06-prompt-generate/field-undo-and-page-undo, with Expand on the box as the page's step */
  { id: 'field-undo-and-page-undo', run: async t => {
    const p = () => document.querySelector('#prompt').value;
    await t.click('#prompt'); await t.type('my {own|own} words');
    await t.click('#wildBtn');
    await t.eq('Expand writes into the box, as a step of the page\'s', p, 'my own words');
    await t.click('#undoBtn');
    await t.eq('the page\'s Undo puts the description back', p, 'my {own|own} words');
    await t.click('#prompt'); await t.press('End'); await t.type(', and more');
    await t.press('Control+z'); await t.press('Control+z'); await t.press('Control+z');
    await t.eq('the field\'s Ctrl+Z steps back through its own texts', p, 'my {own|own} words');
    await t.press('Control+z');
    await t.eq('and on into what the page put there', p, 'my own words');
    await t.press('Control+z');
    await t.eq('and before it', p, 'my {own|own} words');
    await t.press('Control+y'); await t.press('Control+y'); await t.press('Control+Shift+z');
    await t.eq('Redo walks forward the same way', p, 'my {own|own} words,');
    await t.do('a direct write', () => { document.querySelector('#prompt').value = 'set from outside'; });
    await t.press('Control+z');
    await t.eq('a value set without telling is a step of its own', p, 'my {own|own} words,');
    await t.press('Control+y');
    await t.eq('and comes back with Redo', p, 'set from outside');
    await t.click('#brSearch'); await t.type('tem'); await t.press('Control+z'); await t.press('Control+z'); await t.press('Control+z');
    await t.ok('Ctrl+Z in another field never touches the description', () => document.querySelector('#prompt').value === 'set from outside' && document.querySelector('#brSearch').value === '');
  } },
  /* 17-undo-net/field-undo-never-moves-forward, with its setting on a module that is on the desk */
  { id: 'field-undo-never-moves-forward', run: async t => {
    await desk('quick', ['takes'])(t);
    await t.click('#prompt'); await t.type('first words'); await t.do('pause', () => {}, null, 1500); await t.type(' second');
    await t.do('a setting', () => setVal('takes.n', 2));
    await t.click('#prompt'); await t.press('Control+z');
    await t.eq('Ctrl+Z in the description takes back the last typing', () => document.querySelector('#prompt').value, 'first words');
    await t.click('.top .brand'); await t.press('Control+z');
    await t.eq('Ctrl+Z outside it goes on back: the setting, not the words again (F-06)', () => [document.querySelector('#prompt').value, get('takes.n')], ['first words', 1]);
    await t.press('Control+z');
    await t.eq('and then the rest of the typing', () => document.querySelector('#prompt').value, '');
  } },
  { id: 'lyrics-keep-their-history-on-another-tier', run: async t => {
    await desk('studio', [])(t);
    await t.click(`.tile[data-m="lyrics"] ${LYR}`); await t.type('[Verse]'); await t.press('Enter'); await t.type('oneline');
    await t.ok('typing in the sheet is one step, named for the lyrics', () => /Edit the lyrics/.test(document.querySelector('#undoBtn span').textContent) && UNDO.length === 1 || [document.querySelector('#undoBtn span').textContent, UNDO.length]);
    await t.press('Alt+1');
    await t.click(`#quickLyrics ${LYR}`); await t.press('Control+z');
    await t.eq('the sheet drawn again at Quick keeps its own history', () => document.querySelector('#quickLyrics [data-bind="lyrics.text"]').value, '[Verse]\n');
    await t.press('Control+z'); await t.press('Control+z');
    await t.eq('back to empty', () => [document.querySelector('#quickLyrics [data-bind="lyrics.text"]').value, get('lyrics.text')], ['', '']);
    await t.press('Control+z');
    await t.eq('and Ctrl+Z after that goes on to the page, never putting the words back (F-06)', () => [document.querySelector('#quickLyrics [data-bind="lyrics.text"]').value, get('lyrics.text')], ['', '']);
  } },
  /* a caption a dataset supplies is not written over by a song's Reuse either; Edit a copy first (E-31) */
  { id: 'reuse-leaves-a-supplied-caption', run: async t => {
    await desk('studio', [], { 'cap.src': 'Track 04 · Paper boats' }, 'mm3')(t);
    const shown = await t.js(() => document.querySelector('#prompt').value);
    await t.click('#latest .song [data-songmenu]');
    await t.click('#menu [data-sa="reuse"]');
    await t.ok('Reuse leaves the supplied caption and says why, in the source\'s own words', s => document.querySelector('#prompt').value === s && document.querySelector('#prompt').readOnly && /comes from the dataset \(Caption source: Track 04 · Paper boats\)\. Edit a copy to reuse this one/.test(document.querySelector('#toast .toast-msg')?.textContent || '') || (document.querySelector('#toast .toast-msg')?.textContent || ''), shown);
  } },
  /* 17-undo-net/ctrl-z-under-an-overlay-edits-nothing, with the picker as the layer (no package draws the Concept notes) */
  { id: 'ctrl-z-under-the-picker-edits-nothing', run: async t => {
    await t.click('#prompt'); await t.type('alpha beta gamma');
    await t.press('F9'); await t.press('Control+z');
    await t.eq('under the picker the description stays (F-04)', () => document.querySelector('#prompt').value, 'alpha beta gamma');
    await t.ok('and says why', () => /Close this first/.test(document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.press('Escape');
    await t.click('#prompt'); await t.press('Control+z');
    await t.ok('back in the box, its own Ctrl+Z takes back its typing', () => document.querySelector('#prompt').value !== 'alpha beta gamma');
  } },
  { id: 'what-the-page-wrote-is-a-step-of-the-field', run: async t => {
    await desk('studio', [])(t);
    await t.click('#aiBtn');
    await t.ok('Write it for me fills the lyrics it was not in', () => /Kettle on/.test(document.querySelector('.tile[data-m="lyrics"] [data-bind="lyrics.text"]').value));
    await t.click('.tile[data-m="lyrics"] [data-bind="lyrics.text"]'); await t.press('Control+End'); await t.type('!');
    await t.press('Control+z');
    await t.ok('the field\'s Ctrl+Z takes back the typing, to what the page wrote', () => /Stay a little longer, stay$/.test(document.querySelector('[data-bind="lyrics.text"]').value));
    await t.press('Control+z');
    await t.eq('and before it, the empty sheet: what the page wrote was a step of the field\'s own (F-22)', () => [document.querySelector('[data-bind="lyrics.text"]').value, get('lyrics.text')], ['', '']);
    await t.ok('the field\'s own step, not the page\'s Undo: the description Write it for me wrote stays', () => /room sound/.test(document.querySelector('#prompt').value) || document.querySelector('#prompt').value);
  } },
  /* 05-modules/lyrics, with the vocal gender found by its value as well as its words */
  { id: 'lyrics-tags-strip-instrumental-wildcards', run: async t => {
    await desk('lab', [])(t);
    await t.click('#bento .tile[data-m="lyrics"] [data-tag="[Verse]"]');
    await t.eq('a tag tab inserts the tag', () => get('lyrics.text'), '[Verse]\n');
    await t.type('hello there');
    await t.click('#bento .tile[data-m="lyrics"] [data-tag="[Chorus]"]');
    await t.eq('the next tag goes on a fresh line', () => get('lyrics.text'), '[Verse]\nhello there\n\n[Chorus]\n');
    await t.ok('the strip shows the sections', () => document.querySelectorAll('[data-strip] [data-secjump]').length === 2);
    await t.click('[data-strip] [data-secjump="0"]');
    await t.ok('a strip segment puts the caret in its section', () => document.activeElement.matches('[data-bind="lyrics.text"]') && document.activeElement.selectionStart === 8);
    await t.click('#bento .tile[data-m="lyrics"] [data-ledtog="lyrics.instr"]');
    await t.ok('Instrumental keeps the lyrics read-only', () => get('lyrics.instr') === true && document.querySelector('[data-bind="lyrics.text"]').readOnly);
    await t.click('#bento .tile[data-m="lyrics"] [data-tag="[Bridge]"]');
    await t.ok('tags are refused while Instrumental is on', () => /Instrumental is on/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')));
    await t.click('#bento .tile[data-m="lyrics"] [data-ledtog="lyrics.instr"]');
    await t.do('wildcards', () => setVal('lyrics.text', 'go {fast|slow} now'));
    await t.click('#bento .tile[data-m="lyrics"] [data-wildx="lyrics"]');
    await t.ok('Expand choices picks one, with Undo', () => /^go (fast|slow) now$/.test(get('lyrics.text')) && document.querySelector('#toast .toast-act')?.textContent === 'Undo');
    await t.click('#bento .tile[data-m="lyrics"] [data-wildx="lyrics"]');
    await t.ok('with none left it says so', () => /No \{a\|b\|c\} wildcards in the lyrics/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')));
    await t.click('#bento .tile[data-m="lyrics"] [data-ledtog="wild.auto"]');
    await t.eq('on Generate', () => get('wild.auto'), true);
    await t.do('mm3', () => { S.backend = 'mm3'; render(); });
    await t.click('#bento .tile[data-m="lyrics"] [data-seg="lyrics.gender"]:is([data-v="Duet"], [data-v="duet"])');
    await t.eq('MiniMax has a vocal gender', () => get('lyrics.gender'), 'Duet');
    await t.do('sa3', () => { S.backend = 'sa3'; render(); });
    await t.ok('Stable Audio has no lyrics, and says the ones written are kept', () => /has no lyrics\. Describe everything, vocals included, in the prompt\. Your lyrics are kept/.test(document.querySelector('#bento').textContent) || document.querySelector('#bento').textContent.slice(0, 200));
  } },
  { id: 'lyric-sheet-grows-then-scrolls-with-a-marker', w: 1600, h: 900, run: async t => {
    await desk('studio', [])(t);
    const h0 = await t.js(sel => document.querySelector(sel).offsetHeight, LYR);
    await t.do('ten lines', () => { const el = document.querySelector('[data-bind="lyrics.text"]'); setFieldText(el, Array.from({ length: 10 }, (_, i) => `line ${i + 1}`).join('\n'), 0, el.value.length); });
    await t.ok('the sheet grows with its lines, a ruled line each (A-28)', h0 => { const el = document.querySelector('[data-bind="lyrics.text"]'); return el.offsetHeight > h0 && !el.classList.contains('capped') && !document.querySelector('.paper > .smark') || [h0, el.offsetHeight, el.className]; }, h0);
    await t.do('sixty lines', () => { const el = document.querySelector('[data-bind="lyrics.text"]'); setFieldText(el, Array.from({ length: 60 }, (_, i) => `line ${i + 1}`).join('\n'), 0, el.value.length); });
    await t.ok('past its cap it stops and scrolls, with a marker on the margin and a fade at its foot (Q-28)', () => { const el = document.querySelector('[data-bind="lyrics.text"]'), cap = Math.max(420, Math.round(innerHeight * 0.62)); return el.classList.contains('capped') && el.classList.contains('more-down') && !el.classList.contains('more-up') && Math.abs(el.offsetHeight - cap) <= 1 && !!document.querySelector('.paper > .smark') || [el.className, el.offsetHeight, cap]; });
    await t.do('scroll to the foot', () => { const el = document.querySelector('[data-bind="lyrics.text"]'); el.scrollTop = el.scrollHeight; el.dispatchEvent(new Event('scroll')); });
    await t.ok('scrolled, the marker rides down and the fade moves to the top', () => { const el = document.querySelector('[data-bind="lyrics.text"]'), m = document.querySelector('.paper > .smark'); return el.classList.contains('more-up') && !el.classList.contains('more-down') && m && m.offsetTop + m.offsetHeight >= el.offsetTop + el.clientHeight - 2 || [el.className, m?.style.top]; });
  } },
  /* 18-tips-touch/language-stamp-stays-put, with the option found by its value as well as its words */
  { id: 'language-stamp-stays-put', run: async t => {
    await t.do('studio lyrics', () => { S.tier = 'studio'; S.mods = ['takes']; render(); window.__st = (() => { const r = document.querySelector('.stamp-wrap').getBoundingClientRect(); return [r.right, r.top]; })(); });
    await t.click('.stamp-wrap [data-pick="lyrics.lang"]');
    await t.click('#dd .opt-stamp:is([data-opt="French"], [data-opt="fr"])');
    await t.ok('choosing a language leaves the stamp where it was (L-02, E-01)', () => { const s = document.querySelector('.stamp-wrap'), r = s.getBoundingClientRect(), a = window.__st; return s.classList.contains('changed') && getComputedStyle(s).position === 'absolute' && Math.abs(r.right - a[0]) < 1 && Math.abs(r.top - a[1]) < 1 || [getComputedStyle(s).position, r.right, r.top, a]; });
    await t.eq('and the sheet is in French', () => get('lyrics.lang'), 'French');
    await t.press('Alt+1');
    await t.ok('at Quick the stamp shows while the language is not the default (Q-07)', () => !!document.querySelector('#quickLyrics .stamp-wrap'));
  } },
  /* 06-prompt-generate/prompt-tools without the popover (package 8's) */
  { id: 'prompt-tools-without-the-popover', run: async t => {
    await t.click('#prompt'); await t.type('folk {fast|slow} song');
    await t.ok('a wildcard shows Expand and Expand on Generate', () => !document.querySelector('#wildBtn').hidden && !document.querySelector('#wildAuto').hidden && /Wildcard/.test(document.querySelector('#suggest').textContent));
    await t.click('#wildAuto');
    await t.eq('Expand on Generate toggles', () => [get('wild.auto'), document.querySelector('#wildAuto').getAttribute('aria-pressed')], [true, 'true']);
    await t.ok('and the wildcard line says what Generate will do', () => /Each song picks one option/.test(document.querySelector('#suggest').textContent));
    await t.click('#wildBtn');
    await t.ok('Expand picks a choice, and focus stays in the box', () => /^folk (fast|slow) song$/.test(document.querySelector('#prompt').value) && document.querySelector('#wildBtn').hidden && document.activeElement === document.querySelector('#prompt'));
    await t.do('empty prompt', () => { const p = document.querySelector('#prompt'); p.value = ''; p.dispatchEvent(new Event('input', { bubbles: true })); });
    await t.click('#aiBtn');
    await t.ok('Write it for me fills the description and lyrics, with Undo', () => /indie folk/.test(document.querySelector('#prompt').value) && /\[Verse\]/.test(get('lyrics.text')) && /^Wrote a description and lyrics \(simulated\)/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) && document.querySelector('#toast .toast-act')?.textContent === 'Undo' || (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.do('lyrics of my own', () => setVal('lyrics.text', '[Verse]\nrain on the window'));
    await t.js(() => { window.__u = UNDO.length; });
    await t.click('#aiBtn');
    await t.ok('a second time both are filled: it asks, changes nothing, adds no undo step (F-17)', () => /indie folk/.test(document.querySelector('#prompt').value) && get('lyrics.text') === '[Verse]\nrain on the window' && /^Both are filled in/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) && UNDO.length === window.__u);
    await t.click('#toast .toast-act');
    await t.ok('Replace lyrics writes them, and says only what it did (C-29)', () => /Kettle on/.test(get('lyrics.text')) && /^Wrote new lyrics/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) && document.querySelector('#toast .toast-act')?.textContent === 'Undo');
    await t.click('#toast .toast-act');
    await t.eq('its Undo puts the lyrics back', () => get('lyrics.text'), '[Verse]\nrain on the window');
    await t.click('#songInfoBtn');
    await t.ok('Song info opens with focus in it', () => !document.querySelector('#songInfo').hidden && document.activeElement.matches('#songInfo input'));
    await t.type('The Band');
    await t.eq('artist binds', () => get('song.artist'), 'The Band');
    await t.click('#songInfoBtn');
    await t.ok('and closes, marked as filled', () => document.querySelector('#songInfo').hidden && document.querySelector('#songInfoBtn').classList.contains('on'));
  } },
  /* 06-prompt-generate/minimax-caption, with the caption source's options found by their values as well as their words */
  { id: 'minimax-caption-and-the-caption-source', run: async t => {
    await t.do('mm3', () => { if (typeof enterEngine === 'function') enterEngine('mm3'); S.backend = 'mm3'; S.tier = 'studio'; S.mods = ['tempo']; S.vals['tempo.bpm'] = 100; S.vals['tempo.bpm.auto'] = false; S.vals['tempo.key'] = 'A minor'; render(); });
    await t.click('#structBtn');
    await t.ok('Compose caption needs a description', () => /Write a few words of style first/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')));
    await t.click('#prompt'); await t.type('dream pop, hazy');
    await t.click('#structBtn');
    await t.ok('then builds a Structured Caption with BPM and key', () => /^Global Metadata: dream pop, 100 BPM, A minor/.test(document.querySelector('#prompt').value) && /with your BPM and key/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) || document.querySelector('#prompt').value);
    await t.click('#structBtn');
    await t.ok('and not twice', () => /already a Structured Caption/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')));
    await t.click('#capSrc [data-pick="cap.src"]');
    await t.ok('the caption source is a chip list', () => document.querySelector('#dd').classList.contains('dd-chip') && !!document.querySelector('#dd [data-opt="Track 01 · Night drive"]'));
    await t.click('#dd :is([data-opt="Automatic"], [data-opt="auto"])');
    await t.ok('Automatic takes a dataset caption and locks the prompt', () => document.querySelector('#prompt').readOnly && !document.querySelector('#capLock').hidden && /Global Metadata/.test(document.querySelector('#prompt').value));
    await t.click('#prompt');
    await t.ok('a click on the locked prompt points at Edit a copy', () => document.querySelector('#capLock [data-capcustom]').classList.contains('ping'));
    await t.press('x');
    await t.ok('typing says why it is locked, in the control\'s own noun (E-31)', () => /comes from the dataset \(Caption source: Automatic\)\. Edit a copy/.test(document.querySelector('#say').textContent));
    const shown = await t.js(() => document.querySelector('#prompt').value);
    await t.click('#capLock [data-capcustom]');
    await t.ok('Edit a copy makes the caption you were looking at editable (E-06)', s => !document.querySelector('#prompt').readOnly && document.querySelector('#prompt').value === s && get('cap.src') === 'Custom', shown);
    await t.click('#toast .toast-act');
    await t.ok('and Undo puts the dataset caption back, locked', s => document.querySelector('#prompt').readOnly && document.querySelector('#prompt').value === s && get('cap.src') === 'Automatic', shown);
    await t.do('yue studio', () => { if (typeof enterEngine === 'function') enterEngine('yue'); S.backend = 'yue'; render(); });
    const yshown = await t.js(() => document.querySelector('#prompt').value);
    await t.click('#dsTags [data-dstag="gated drums"]');
    await t.ok('a tag on a supplied caption edits a copy of it (E-06)', s => !document.querySelector('#prompt').readOnly && document.querySelector('#prompt').value === s + ', gated drums', yshown);
    await t.do('a plain description', () => { setVal('cap.src', 'Custom'); setFieldText(document.querySelector('#prompt'), 'synthwave-ish, warm'); paintPromptTools(); });
    await t.ok('a tag is pressed only when it is one of the parts (E-31)', () => !document.querySelector('#dsTags [data-dstag="synthwave"]').classList.contains('on'));
    await t.click('#dsTags [data-dstag="synthwave"]');
    await t.ok('a dataset tag on YuE2 adds to the caption', () => /synthwave/.test(document.querySelector('#prompt').value) && document.querySelector('#dsTags [data-dstag="synthwave"]').classList.contains('on'));
    await t.click('#dsTags [data-dstag="synthwave"]');
    await t.ok('and removes it', () => document.querySelector('#prompt').value === 'synthwave-ish, warm');
  } },
];
