// The roles: what the page needs to find among the definitions by what it is, not by its key. Where the core or a part
// needs a particular parameter or state (the main box, the takes count, the song a repaint works on), it asks for the
// role (`roleKey(ix, 'takes')`), and the parameter or state that plays it says so in its definition (`role: 'takes'`).
// So a key renamed the documented way (docs/06, 8b: the new key and `was`) keeps its role and the page keeps working,
// and code never names a key, a state or a module id: the lint and tools/test-view.ts refuse one (docs/06, "Roles").
//
// The validator checks every role here against the definitions (tools/validate-definitions.ts): what plays it is of
// the kind and type it says, no two parameters play one role on one engine (a `many` role may have several), and a
// role nothing plays is refused when `needed`, else named in a note with what the page does without it. A role no
// code asks for is refused too: take it out of this list.
//
// Each entry: `of` (a parameter or a state), `type` (a parameter's value type, or a state's kind), whether `many` may
// play it, whether it is `needed`, and `doc`: what the page does with it, and without it.

export interface RoleSpec {
  of: 'param' | 'state';
  type: 'number' | 'boolean' | 'choice' | 'text' | 'file' | 'seed' | 'span' | 'slots' | 'record' | 'list';
  many?: boolean;
  needed?: boolean;
  doc: string;
}

export const ROLES = {
  /* the prompt card and the lyric sheet */
  prompt: { of: 'param', type: 'text', needed: true, doc: 'The main box: what a run is made from (Generate refuses an empty one), what Reuse puts back, what names a draft and a revision\'s content.' },
  lyrics: { of: 'param', type: 'text', doc: 'The lyric sheet\'s text: where suggestions move lyrics to, what Write it for me fills, what a draft and a revision say they hold. Without it the page has no lyrics anywhere.' },
  instrumental: { of: 'param', type: 'boolean', doc: 'Instrumental: the lyrics stand back, and Generate on ACE-Step needs no lyrics. Without it, never instrumental.' },
  songArtist: { of: 'param', type: 'text', doc: 'Song info\'s artist: the card\'s line names it. Without it Song info has no artist field.' },
  songTitle: { of: 'param', type: 'text', doc: 'Song info\'s title: a song\'s title when it is set. Without it every title comes from the words.' },
  songSubject: { of: 'param', type: 'text', doc: 'Song info\'s subject field. Without it Song info has no subject field.' },
  coverSubject: { of: 'param', type: 'text', doc: 'The cover art\'s subject, shown in Song info while its unit is on. Without it Song info shows no cover field.' },
  wildcards: { of: 'param', type: 'boolean', doc: 'Expand wildcards on each run: Generate picks one of each {a|b} group. Without it wildcards are sent as typed.' },
  trigger: { of: 'param', type: 'text', doc: 'The adapter\'s trigger word, its chip on the prompt card. Without it the card has no trigger chip.' },
  captionSource: { of: 'param', type: 'choice', doc: 'Where the caption comes from (Custom, Automatic, a dataset\'s track): the box locks on a supplied caption. Without it the box is always the user\'s.' },
  captionDataset: { of: 'param', type: 'choice', doc: 'The caption dataset whose tracks and tags the card offers. Without it the card offers none.' },
  bpm: { of: 'param', type: 'number', doc: 'The song\'s tempo field: "Set the BPM" from the words, a composed caption\'s BPM, the lyric sheet\'s estimate. Without it the words\' BPM is offered nowhere.' },
  key: { of: 'param', type: 'choice', doc: 'The song\'s key, written into a composed caption. Without it the caption names no key.' },
  vocalGender: { of: 'param', type: 'choice', doc: 'The vocal\'s gender, written into a composed caption. Without it the caption names none.' },
  language: { of: 'param', type: 'choice', doc: 'The lyrics\' language, written into a composed caption when changed. Without it the caption names none.' },
  /* a run */
  takes: { of: 'param', type: 'number', doc: 'How many songs one Generate makes, and what they cost the card (the meter, the go bar, the Takes tile). Without it every run makes one song.' },
  steps: { of: 'param', type: 'number', doc: 'The sampler\'s step count, which the run\'s stage line counts. Without it the line counts 12.' },
  length: { of: 'param', type: 'number', doc: 'The song\'s exact length, where the engine takes one. Without it the engine ends each song itself.' },
  longest: { of: 'param', type: 'number', doc: 'The longest a song may run where the engine ends it itself. Without it, as long as the engine runs.' },
  planner: { of: 'param', type: 'boolean', doc: 'The planner is on: what each take costs the card while it plans (the VRAM figures\' `take.when`). Without it a take\'s cost always counts.' },
  scorePreview: { of: 'param', type: 'boolean', doc: 'Preview the score first: Generate opens the score preview before a run while the lead sheet is empty. Without it, never.' },
  leadSheet: { of: 'param', type: 'text', doc: 'The lead sheet\'s text: a pasted score needs no preview. Without it every preview plans.' },
  stream: { of: 'param', type: 'boolean', doc: 'Play while rendering: a take can be played as it renders. Without it, only once it is done.' },
  replan: { of: 'param', type: 'boolean', doc: 'Re-plan runaway scores: the dice light up as a run tries again. Without it a run never re-plans.' },
  replanTries: { of: 'param', type: 'number', doc: 'How many tries re-planning has. Without it, one.' },
  /* the engine panel and the log */
  variant: { of: 'param', type: 'choice', doc: 'The engine\'s model variant: the first model the log\'s head names. Without it the head starts at the sound model.' },
  soundModel: { of: 'param', type: 'choice', doc: 'The sound model, which the log\'s head names. Without it the head names the models line only.' },
  plannerModel: { of: 'param', type: 'choice', doc: 'The planner\'s model, which the log\'s head names after the sound model.' },
  /* what modules keep */
  repaintSource: { of: 'state', type: 'record', doc: 'The song a repaint works on, which the song menu\'s "Repaint a region" writes: Generate repaints it, as one song. Its module is the one that repaints; without it the song menu offers no repaint.' },
  repaintRegion: { of: 'state', type: 'span', doc: 'The region of the song a repaint works on, written with its song.' },
  latent: { of: 'state', type: 'file', doc: 'The latent a run continues from, which the start toast names. Without it Import latent has nowhere to go.' },
  recording: { of: 'state', type: 'file', doc: 'The recording a run continues from: the start toast names it, a build cannot run beside it, its length is told. Its module is the one that continues.' },
  recordingPart: { of: 'state', type: 'span', doc: 'The part of the recording continued from, named in the start toast.' },
  build: { of: 'state', type: 'record', doc: 'A build by section: Generate refuses to run it beside a recording, its rendered songs join Latest. Its module is the one that builds.' },
  adapters: { of: 'state', type: 'slots', doc: 'The adapters loaded: counted on the card\'s VRAM and in the go bar\'s changes. Without it no adapter is ever loaded.' },
  reference: { of: 'state', type: 'file', many: true, doc: 'A reference track a drop zone takes: remembered among the recent references. Without any, nothing is remembered.' },
} as const satisfies { [role: string]: RoleSpec };

export type Role = keyof typeof ROLES;
