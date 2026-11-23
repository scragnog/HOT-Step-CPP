// Package 4's states (one file of tools/visual-baseline/slice-states/, gathered by slice-states.mjs): the prompt card
// and the lyric sheet, compared whole (the `fields` scope: the column the card stands in with the ask and the Quick
// sheet, the lyric tile at the head of the desk, and the row the desk keeps for an engine with no lyrics), on every
// engine and tier, in both themes, at the narrow widths, and in the states they take: a sheet with sections, at Lab
// with its tabs and wildcard tools, Instrumental, the language stamp, a sheet past its cap with its marker, the box's
// suggestions, a caption a dataset supplies (locked) and its tags, Song info open, a latent.
//
//   node tools/visual-baseline/vb.mjs capture concept-fields --states slice --only "^fields-" --scope fields --page <concept>/hotstep-tiers.html
//   node tools/visual-baseline/vb.mjs capture rebuild-fields --states slice --only "^fields-" --scope fields --plan-from concept-fields
//   node tools/visual-baseline/vb.mjs compare concept-fields rebuild-fields --scope fields --accept tools/visual-baseline/fields-accepted.json

const LYRICS = '[Verse]\nKettle on, the window fogged\nYour coat still hanging by the door\n\n[Pre-Chorus]\nStay a little\n\n[Chorus]\nStay a little longer, stay\nLet the morning wait';
const scene = (bk, tier, mods, vals = {}, extra = {}) => async h => {
  await h.js(([bk, tier, mods, vals, extra]) => { if (bk !== S.backend) enterEngine(bk); S.tier = tier; S.mods = mods; S.pedals = []; Object.assign(S.vals, vals); Object.assign(S, extra); render(); }, [bk, tier, mods, vals, extra]);
  await h.settle(900);
};
/* text put into the box as the page puts it (an input the box takes as typing), then the box left */
const box = text => async h => { await h.js(t => { setFieldText(document.querySelector('#prompt'), t); document.querySelector('#prompt').blur(); }, text); await h.settle(900); };
const then = (...steps) => async h => { for (const s of steps) await s(h); };
/* the lyric tile and the row kept for the note are recorded as parts (keyed from their selector, wherever the desk puts
   them); the card's column is a region of the scope */
const partsOf = id => { const [, bk, tier] = /-(ace|mm3|yue|sa3|hm)-(quick|studio|lab)/.exec(id) ?? []; return tier === 'quick' ? [] : bk === 'sa3' ? ['#bento > .deskrow'] : ['#bento > .tile[data-m="lyrics"]']; };
const st = (id, run, o = {}) => ({ id, w: 1600, h: 1000, scheme: 'light', reduced: false, pseudo: false, shot: false, run, parts: partsOf(id), ...o });

export const STATES = [
  /* every engine at every tier, a sheet with sections */
  ...['ace', 'mm3', 'yue', 'sa3', 'hm'].flatMap(bk => ['quick', 'studio', 'lab'].map(tier =>
    st(`fields-${bk}-${tier}`, scene(bk, tier, ['avoid'], { 'lyrics.text': LYRICS }), { pseudo: bk === 'ace', shot: bk === 'ace' || (bk === 'mm3' && tier === 'lab') ? '#quickcol' : false }))),
  st('fields-ace-studio-dark', scene('ace', 'studio', ['avoid'], { 'lyrics.text': LYRICS }), { scheme: 'dark', shot: '#bento' }),
  st('fields-ace-lab-dark', scene('ace', 'lab', [], { 'lyrics.text': LYRICS }), { scheme: 'dark' }),
  st('fields-ace-quick-dark', scene('ace', 'quick', [], { 'lyrics.text': LYRICS }), { scheme: 'dark', shot: '#quickcol' }),
  /* the narrow desktop widths: the stamp takes a row of its own on a narrow desk (S-20) */
  ...[1440, 1280, 1000].flatMap(w => ['studio', 'lab'].map(tier => st(`fields-w${w}-ace-${tier}`, scene('ace', tier, ['avoid'], { 'lyrics.text': LYRICS }), { w, h: w <= 1000 ? 800 : 900 }))),
  /* Instrumental, a chosen language and the vocal gender, Expand on Generate on (MiniMax-Music3, Lab) */
  st('fields-mm3-lab-instrumental', scene('mm3', 'lab', [], { 'lyrics.text': LYRICS + ' {x|y}', 'lyrics.instr': true, 'lyrics.lang': 'Japanese', 'lyrics.gender': 'Duet', 'wild.auto': true }), { shot: '#bento' }),
  st('fields-mm3-studio-instrumental-dark', scene('mm3', 'studio', [], { 'lyrics.text': LYRICS, 'lyrics.instr': true, 'lyrics.lang': 'French' }), { scheme: 'dark' }),
  /* the stamp at Quick while the language is not the default (Q-07) */
  st('fields-ace-quick-french', scene('ace', 'quick', [], { 'lyrics.text': LYRICS, 'lyrics.lang': 'French' })),
  /* a sheet past its cap: it scrolls, with its marker on the margin (Q-28) */
  st('fields-ace-studio-capped', scene('ace', 'studio', [], { 'lyrics.text': Array.from({ length: 60 }, (_, i) => `line ${i + 1}`).join('\n') }), { h: 900 }),
  /* the box's suggestions (lyrics, a BPM, a wildcard), a latent, Song info open */
  st('fields-ace-studio-suggestions', then(scene('ace', 'studio', [], {}, { latent: 'night-drive.latent' }), box('slow {soft|warm} folk, 92 bpm\n[Verse]\nla la')), { pseudo: true }),
  st('fields-ace-studio-song-info', then(scene('ace', 'studio', [], { 'song.title': 'Kettle', 'song.artist': 'The Band' }), async h => { await h.click('#songInfoBtn'); await h.settle(900); }), { shot: '#quickcol' }),
  st('fields-yue-lab-structured', then(scene('yue', 'lab', []), box('Global Metadata: smoky jazz trio, warm\nVocal Details: male lead'))),
  st('fields-yue-studio-wildcards-dark', then(scene('yue', 'studio', [], { 'wild.auto': true }), box('{dreamy|gritty} indie folk')), { scheme: 'dark' }),
  /* a caption a dataset supplies: locked, where it comes from, Edit a copy; YuE2's tags (E-06, E-08) */
  st('fields-mm3-studio-caption-track', scene('mm3', 'studio', [], { 'cap.src': 'Track 04 · Paper boats' }), { shot: '#quickcol' }),
  st('fields-yue-studio-automatic', scene('yue', 'studio', [], { 'cap.src': 'Automatic' })),
  st('fields-yue-studio-automatic-dark', scene('yue', 'studio', [], { 'cap.src': 'Automatic' }), { scheme: 'dark' }),
  st('fields-yue-quick-custom', scene('yue', 'quick', [])),
  /* an engine with no lyrics, with lyrics kept for engines that sing them (S-11) */
  st('fields-sa3-studio-kept', scene('sa3', 'studio', ['avoid'], { 'lyrics.text': LYRICS })),
  st('fields-sa3-quick-dark', scene('sa3', 'quick', []), { scheme: 'dark' }),
];

/** The prompt card's column: the ask, the card, the Quick sheet. */
export const SCOPES = {
  fields: ['#quickcol'],
};
