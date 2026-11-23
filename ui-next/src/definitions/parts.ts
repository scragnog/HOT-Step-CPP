// The parts inventory: every part a definition may name in a face, and what each one takes.
//
// A part is one of four kinds:
//   control    edits a parameter (or two): its value contract, changed marker, tip and undo come from the parameter
//   container  holds other nodes and gives them the design language's housing (a rack, a well, an LCD, a terminal)
//   readout    shows something, edits nothing
//   bespoke    a part written for one module because what it draws or does cannot be said as data; it still binds
//              to declared parameters, so defaults, undo, saving and the changed marker keep working
//
// `binds` lists the value types it can edit or show ('pair' takes two numbers in `params`, 'state' a module's
// StateDef, 'none' binds nothing). `opts` lists the options a node may pass, each with what it does; the validator
// refuses an option not listed. `css` names the concept's stylesheets (C:\Users\sneth\misc\HOT-Step-UI\src\css at the
// tag vanilla-reference) that hold its look; the first is its own file. Every part also takes 35-changed.css (the
// changed dot), 35-states.css (focus ring, hover, press) and 35-labels.css (the caps label) where it has a label.
//
// To add a part: add an entry here, write its React component under the name, and give it its stylesheet. Every
// definition can then name it. docs/06-how-to.md has the steps.

import type { ValueType } from './types.ts';

export type PartKind = 'control' | 'container' | 'readout' | 'bespoke';
export type Binds = ValueType | 'pair' | 'state' | 'none';

export interface PartSpec {
  kind: PartKind;
  binds: readonly Binds[];
  /** What it is for, in one or two sentences. */
  doc: string;
  /** The states it draws beyond the shared ones (changed, focus, hover, dim, hidden). */
  states?: readonly string[];
  opts?: { readonly [name: string]: string };
  css: readonly string[];
  /** For a bespoke part: exactly what it is responsible for, and what it leaves to the generic machinery. */
  responsible?: string;
}

export const PARTS = {
  // ---------------------------------------------------------------- controls
  knob: {
    kind: 'control', binds: ['number'], css: ['15-knob.css', '27-pedals.css'],
    doc: 'A rotary knob with a detent tick at the default, its label under it and a read-out that can be clicked to type a value. The rack face of every number in Lab.',
    states: ['turning', 'resetting', 'auto (drawn at the auto value in the auto colour, with an auto pill)', 'typing'],
    opts: { small: 'The small knob (pedals, rack rows).', unitBelow: 'On a pedal, the unit on its own line under the number (dBFS, LUFS).' },
  },
  fader: {
    kind: 'control', binds: ['number'], css: ['15-fader.css', '25-module-adapters.css'],
    doc: 'A vertical fader with a cap, tick marks and a unity line at 1.00. Used in fader banks (adapter group scales, depth thirds) and cartridges.',
    states: ['sliding', 'stops (travel beyond max shown as a stop)'],
    opts: { travel: 'The value the top of the slot stands for, when the slot runs past the maximum (Plan: max 1, travel 2).' },
  },
  ruler: {
    kind: 'control', binds: ['number'], css: ['15-ruler.css'],
    doc: 'A length on a minutes timeline, dragged along; labels on round steps. The length of a song.',
    states: ['auto (shows the estimate, or "up to" where the model ends the song)', 'sliding', 'at-start/at-end caption'],
    opts: { scaleMax: 'The seconds the timeline spans. Default: the parameter maximum.', markers: 'Draw Beat I/O and trim markers from the Tempo parameters. Never drawn at Quick.' },
  },
  xyPad: {
    kind: 'control', binds: ['pair'], css: ['15-xy-pad.css'],
    doc: 'Two settings that belong together on one plane: a puck with a trail, a grid, axis read-outs that each carry their own changed marker, corner words outside the pad.',
    states: ['grab', 'release', 'nudge', 'y-off (the Y minimum drawn as an off rail)'],
    opts: { xl: 'The X axis words.', yl: 'The Y axis words.', corners: 'Four corner words: top left, top right, bottom left, bottom right.', yOff: 'Words on the rail when the Y minimum means off.', look: 'apg | heat | overlay | term: the plot colouring.' },
  },
  numberBox: {
    kind: 'control', binds: ['number'], css: ['15-numbox.css'],
    doc: 'A typed number box that can also be dragged up and down. Steps in the Sampler well.',
    states: ['editing', 'overridden (a value another module replaces)', 'empty means auto'],
    opts: { seconds: 'Show an estimate of seconds under it (Steps: "≈ 1.3 s").' },
  },
  lcdNumber: {
    kind: 'control', binds: ['number'], css: ['25-module-tempo.css', '15-lcd.css'],
    doc: 'A big seven-segment number on an LCD that can be typed or dragged, with its unit beside it. BPM.',
    opts: { tap: 'Offer TAP (tap tempo, bespoke behaviour of the Tempo tile) and AUTO beside it.' },
  },
  pick: {
    kind: 'control', binds: ['choice'], css: ['15-pick.css', '30-dropdown.css'],
    doc: 'The button that shows a choice and opens its dropdown. The variant sets where it lives.',
    states: ['open', 'overridden'],
    opts: { variant: 'plugin: a periodic-table tile and the kind above the value (solver, schedule, guidance); lcd: an LCD with previous and next arrows; chip: a pill with its label; model: name, quant badge, size and a memory bar; stamp: an ink stamp (the lyric language).' },
  },
  chips: {
    kind: 'control', binds: ['choice'], css: ['15-controls.css'],
    doc: 'A chip selector: up to four options as lit chips under a caps label. Lab rack face of a short choice.',
  },
  segmented: {
    kind: 'control', binds: ['choice'], css: ['15-controls.css'],
    doc: 'A segmented choice: a row of joined buttons, one pressed.',
    opts: { size: 'sm: the compact row inside a rack group.' },
  },
  termChips: {
    kind: 'control', binds: ['choice'], css: ['25-module-planner.css'],
    doc: 'A choice written as terminal chips after a flag (--mode Presence Frequency DRY).',
  },
  lcdChips: {
    kind: 'control', binds: ['choice'], css: ['25-module-tempo.css'],
    doc: 'Numbered chips on an LCD with a word after them (1 2 4 8 BARS).',
    opts: { word: 'The word printed after the chips.' },
  },
  ledButton: {
    kind: 'control', binds: ['boolean'], css: ['15-controls.css'],
    doc: 'An LED switch: a rack button with a lamp that lights when on. Lab rack face of a boolean.',
  },
  pill: {
    kind: 'control', binds: ['boolean'], css: ['15-controls.css'],
    doc: 'A pill toggle with a lamp, used inside a module look (Instrumental, Level match, a rack group power switch).',
  },
  flag: {
    kind: 'control', binds: ['boolean'], css: ['25-module-planner.css'],
    doc: 'A terminal flag with a check box: [x] --cot.',
  },
  lcdButton: {
    kind: 'control', binds: ['boolean'], css: ['25-module-tempo.css', '15-lcd.css'],
    doc: 'A lit word on an LCD that switches (BEAT I/O, AUTO-TRIM).',
  },
  recButton: {
    kind: 'control', binds: ['boolean'], css: ['25-module-plans.css'],
    doc: 'A record button (● SAVE NEXT) that arms the next run to save something.',
    states: ['is-off (stands back: nothing new to save)'],
  },
  powerButton: {
    kind: 'control', binds: ['boolean'], css: ['25-module-planner.css'],
    doc: 'A terminal power switch with its word (● planner) and the scope switch (● Lua plugins): the switch a whole instrument hangs on.',
    opts: { look: 'term: the terminal bar; scope: on a scope read-out.' },
  },
  lockButton: {
    kind: 'control', binds: ['boolean'], css: ['15-controls.css', '25-module-seed.css'],
    doc: 'A padlock button that keeps a value (Keep this seed).',
  },
  powerChip: {
    kind: 'control', binds: ['number'], css: ['15-controls.css', '25-module-adhere.css'],
    doc: 'A rack group power chip for a group switched by a number: off is one value, on sets a moderate one. Says "· off" on itself when off.',
    opts: { off: 'The value that means off.', on: 'The value pressing it on sets.' },
  },
  token: {
    kind: 'control', binds: ['number'], css: ['15-numbox.css', '25-module-planner.css'],
    doc: 'A scrub token: a number in running text or a terminal line, dragged sideways, stepped with the arrows, typed on Enter.',
    states: ['clash (another setting makes it skip)'],
  },
  textField: {
    kind: 'control', binds: ['text'], css: ['15-controls.css', '15-lcd.css'],
    doc: 'A one-line text field. Checked fields show their error beside them and commit on Enter or leaving.',
    states: ['bad (fails its check)', 'kept (invalid text kept, what is in force said)'],
    opts: { look: 'plain | lcd: an LCD input | term: an inline terminal input', size: 'Width in characters.' },
  },
  textArea: {
    kind: 'control', binds: ['text'], css: ['15-controls.css'],
    doc: 'A text area. The look gives it its material.',
    opts: { look: 'plain | note: the taped sticky note (Avoid) | staff: the ruled ABC sheet (Lead sheet) | paper: the lyric sheet | term: a quoted one-line terminal string that wraps (--neg)' },
  },
  choiceRow: {
    kind: 'control', binds: ['number', 'choice'], css: ['15-controls.css'],
    doc: 'Quick\'s plain choice row: a radio group of words that each set a detailed value. A value between the choices checks none: a marker stands at its place and the label says the value.',
    opts: { look: 'plain | mood: the coloured How adventurous row' },
  },
  lookback: {
    kind: 'control', binds: ['number'], css: ['25-module-repetition.css'],
    doc: 'A look-back window drawn as a strip of codes that grows leftward into the past, with a time scale under it.',
    opts: { unit: 'The word for what it counts (codes, frames, tokens).', perSecond: 'How many a second, for the scale and the tip.', who: 'Who checks for repeats, for the tip (the planner).' },
  },
  dropZone: {
    kind: 'control', binds: ['state'], css: ['15-controls.css', '25-module-timbre.css', '27-pedals.css'],
    doc: 'A file drop zone: a record sleeve (modules) or a slot (pedals) that takes a dropped or chosen audio file, shows its name cut to fit, and clears with ×.',
    opts: { state: 'The StateDef key of the file it holds.', look: 'sleeve | slot', empty: 'Its words while empty.', hint: 'Its second line while empty.' },
  },
  // ---------------------------------------------------------------- containers
  row: {
    kind: 'container', binds: ['none'], css: ['15-knob.css'],
    doc: 'A row of knobs or controls with even spacing (the .knobs row).',
  },
  rack: {
    kind: 'container', binds: ['none'], css: ['15-controls.css'],
    doc: 'A rack row: LED switches and chip selectors side by side on a rack strip.',
  },
  well: {
    kind: 'container', binds: ['none'], css: ['25-module-sampler.css'],
    doc: 'A recessed well that holds a module\'s main controls under its instrument (the Sampler strip).',
  },
  lcd: {
    kind: 'container', binds: ['none'], css: ['15-lcd.css', '25-module-tempo.css'],
    doc: 'An LCD panel. Its children are rows (lcdRow) or LCD parts.',
  },
  lcdRow: {
    kind: 'container', binds: ['none'], css: ['25-module-tempo.css'],
    doc: 'One row of an LCD. Ends-aligned when it holds switches at both ends.',
    opts: { ends: 'Spread to both ends.' },
  },
  terminal: {
    kind: 'container', binds: ['none'], css: ['25-module-planner.css'],
    doc: 'A terminal: dark, monospaced, a command bar on top, lines below. The planner and the repetition guard.',
    opts: { title: 'The tip on the whole terminal.' },
  },
  termBar: {
    kind: 'container', binds: ['none'], css: ['25-module-planner.css'],
    doc: 'The terminal\'s top bar: its power switch and flags.',
  },
  termLine: {
    kind: 'container', binds: ['none'], css: ['25-module-planner.css'],
    doc: 'A command line: "$ <command>" then flags, each followed by the control for its parameter in `params`, drawn by type: a number as a token, a choice as terminal chips, a boolean as a flag, text as an inline field. Children follow the flags. Wrapped lines hang under the first flag.',
    opts: { command: 'The command word after $.', flags: 'The flag printed before each of `params`, in order (--temp, --cfg).' },
  },
  group: {
    kind: 'container', binds: ['none'], css: ['25-module-adhere.css', '25-module-latent.css'],
    doc: 'A rack group: a boxed set of controls with a header that may carry its power switch; a group switched off stands back.',
    opts: { title: 'Its caps heading when it has no power switch.' },
  },
  foot: {
    kind: 'container', binds: ['none'], css: ['20-tile.css'],
    doc: 'A tile\'s foot row: small labelled controls and a note, kept to the bottom of the tile.',
  },
  labelled: {
    kind: 'container', binds: ['none'], css: ['15-controls.css'],
    doc: 'A caps label beside one small control (the foot-ctl and lcdsel pairing).',
  },
  deck: {
    kind: 'container', binds: ['none'], css: ['25-module-plans.css'],
    doc: 'A two-column grid of labels and controls, so a choice never moves the rows under it (Plans & endings).',
  },
  pair: {
    kind: 'container', binds: ['none'], css: ['25-module-adhere.css'],
    doc: 'Two tokens as a range: from → to, under one label.',
  },
  bank: {
    kind: 'container', binds: ['none'], css: ['15-fader.css', '25-module-adapters.css'],
    doc: 'A fader bank with a 2-1-0 scale and a heading, each fader labelled on tape.',
    opts: { title: 'Its heading.', note: 'Its heading\'s tip.' },
  },
  // ---------------------------------------------------------------- read-outs
  note: {
    kind: 'readout', binds: ['none'], css: ['01-base.css', '15-controls.css'],
    doc: 'A quiet line of words: a hint, why something is off, what another module changes.',
    opts: { look: 'plain | warn | cap: the caption note under an instrument | hint: Quick\'s hint' },
  },
  link: {
    kind: 'readout', binds: ['none'], css: ['14-buttons.css'],
    doc: 'A text link that runs an action (Insert example, Show in Lab, Turn on Lua plugins).',
  },
  button: {
    kind: 'readout', binds: ['none'], css: ['14-buttons.css'],
    doc: 'A small button that runs an action (+ Load adapter, Roll once).',
  },
  lcdValue: {
    kind: 'readout', binds: ['choice', 'number', 'text'], css: ['25-module-tempo.css'],
    doc: 'A value printed on an LCD under a caps word (KEY D MAJOR).',
  },
  unitLcd: {
    kind: 'readout', binds: ['none', 'choice'], css: ['28-side-units.css'],
    doc: 'A side unit\'s two-line LCD. Each line is a parameter\'s LCD text, or fixed words.',
    opts: { lines: 'Up to two lines: a parameter key (printed as its option\'s lcd text) or words in double quotes.', prefix: 'Words before the first line (WHISPER ).', notAtLab: 'Not drawn at Lab, where the unit\'s own controls show the same values.' },
  },
  unitStatus: {
    kind: 'readout', binds: ['none'], css: ['28-side-units.css'],
    doc: 'A side unit\'s status words in small caps (After the render · needs MMS).',
  },
  unitBadge: {
    kind: 'readout', binds: ['none'], css: ['28-side-units.css'],
    doc: 'A side unit\'s round score badge.',
  },
  dice: {
    kind: 'readout', binds: ['number'], css: ['25-module-replan.css'],
    doc: 'A row of numbered dice, one per allowed attempt, filled as attempts are used.',
    opts: { offWhen: 'A boolean parameter that, when false, draws the dice off.' },
  },
  // ---------------------------------------------------------------- bespoke
  samplerScope: {
    kind: 'bespoke', binds: ['none'], css: ['25-module-sampler.css'],
    doc: 'The Sampler\'s scope: the noise schedule drawn from the steps, shift, schedule and shape settings, with the CFG interval\'s two handles and its read-out.',
    opts: { interval: 'Draw the CFG interval\'s two handles (Lab).' },
    responsible: 'Drawing sigma(t) from samp.steps, samp.shift, samp.sched and the sch.* shape parameters, or from lat.ts when Latent & correction sets custom timesteps; the guidance band from samp.cs/samp.ce and the hatched CFG cutoff from perf.cfgcut; the two interval handles (they edit samp.cs and samp.ce through the value contract, a step apart); the one-line read-out (native sampling, guidance window, custom timesteps with Show). It does not own any value: defaults, undo, changed markers and saving of those parameters are generic.',
  },
  keyWheel: {
    kind: 'bespoke', binds: ['choice'], css: ['25-module-tempo.css'],
    doc: 'The circle of fifths: majors outside, minors inside, Auto in the middle, a radio group driven by the arrows.',
    responsible: 'Drawing the two rings from tempo.key\'s options, spelling keys as a musician does (B♭, F♯), the arrow-key walk around and between rings, and the selection nudge. tempo.key stays an ordinary choice: undo, changed marker, "Also set" and saving are generic.',
  },
  tapTempo: {
    kind: 'bespoke', binds: ['number'], css: ['25-module-tempo.css'],
    doc: 'TAP: two taps or more within 2.5 s set the BPM from their spacing.',
    responsible: 'Timing the taps and calling set on tempo.bpm (clamped to its range); the run of taps is one undo step through the parameter\'s `run`.',
  },
  beatPulse: {
    kind: 'readout', binds: ['none'], css: ['25-module-tempo.css'],
    doc: 'Lamps that pulse on the beat at the set BPM, one per beat of the bar.',
  },
  capChips: {
    kind: 'bespoke', binds: ['none'], css: ['25-module-tempo.css'],
    doc: 'On engines where tempo and key go in the prompt: chips that add or take out "120 BPM" or "D major" as items of the prompt, two exclusive groups.',
    responsible: 'Reading which chips are pressed from the prompt text, replacing an item of the same group, never touching words inside a sentence, one undo step per press, and offering the user\'s own tempo and key first.',
  },
  takesReels: {
    kind: 'bespoke', binds: ['number'], css: ['25-module-takes.css'],
    doc: 'The takes count as an LCD and a set of reels, one per take the engine allows; the reels are one slider. Under it, the VRAM the count costs.',
    responsible: 'Drawing as many reels as the engine\'s takes maximum, the hover preview, the reel spin, and the VRAM cost line from the loaded models. takes.n stays a number with a per-engine range; its label per engine and its held value are generic.',
  },
  splitFlap: {
    kind: 'bespoke', binds: ['state'], css: ['25-module-seed.css'],
    doc: 'The seed as ten split-flap cells that flip to a new number; pressing it types one.',
    opts: { state: 'The StateDef key of the seed.' },
    responsible: 'The flap animation, the ten-cell layout, typing or pasting a seed (checked 0 to 4294967295), Roll once and Random each time. The seed is the module\'s state `seed`; seed.lock is an ordinary boolean.',
  },
  savedSeeds: {
    kind: 'bespoke', binds: ['none'], css: ['25-module-seed.css'],
    doc: 'The saved seeds list: search, star, name, describe, tag, load, delete, random from saved.',
    responsible: 'The list and its own store (the session\'s saved seeds); every list action is its own undo step with its own name.',
  },
  timbreBlend: {
    kind: 'bespoke', binds: ['number'], css: ['25-module-timbre.css'],
    doc: 'Two overlapping circles, the reference dragged into your sound: the reference blend.',
    responsible: 'Drawing the circles at the blend and dragging sideways. timbre.str is an ordinary number (its percent format, undo and changed marker are generic).',
  },
  recentRefs: {
    kind: 'bespoke', binds: ['none'], css: ['25-module-timbre.css'],
    doc: 'The two latest references as chips, and the rest in a popover.',
    responsible: 'The session list of recent reference files and loading one into the module\'s file state.',
  },
  flowWindows: {
    kind: 'readout', binds: ['none'], css: ['25-module-flow.css'],
    doc: 'MiniMax-Music3\'s flow windows drawn at the hop, and the count of windows a 3:00 song needs.',
  },
  speedGauge: {
    kind: 'readout', binds: ['none'], css: ['25-module-perf.css'],
    doc: 'An LCD that estimates how much faster the trades make a run.',
  },
  latentCurve: {
    kind: 'bespoke', binds: ['none'], css: ['25-module-latent.css'],
    doc: 'The latent distribution before (grey) and after (green) shift and rescale, with the XY pad that sets them laid over it as its child.',
    responsible: 'Drawing the two curves from lat.shift and lat.rescale and hosting its child XY pad over them. The pad is the generic xyPad part.',
  },
  adhereCurve: {
    kind: 'readout', binds: ['none'], css: ['25-module-adhere.css'],
    doc: 'Plan strength across the run, from noise to clean, with the Plan CFG band and the CADS ramp.',
  },
  planText: {
    kind: 'readout', binds: ['none'], css: ['25-module-planner.css'],
    doc: 'The plan the planner will start from, as terminal text: caption, tempo, key, form, and whose each value is. On MiniMax-Music3 its first line also says whether a saved plan is replayed or the plan cache is on (from mm.reuse, mm.plank, mm.hidpath).',
  },
  vuMeter: {
    kind: 'readout', binds: ['none'], css: ['27-pedals.css'],
    doc: 'The Mastering pedal\'s printed VU meter, -20 to +3, red from 0.',
  },
  lrcPreview: {
    kind: 'readout', binds: ['none'], css: ['28-side-units.css'],
    doc: 'Two timed lyric lines, the first named after the lyrics\' first section.',
  },
  coverSwatch: {
    kind: 'readout', binds: ['none'], css: ['28-side-units.css'],
    doc: 'A swatch in the newest song\'s colour, standing for its cover art.',
  },
  abcFoot: {
    kind: 'bespoke', binds: ['none'], css: ['25-module-score.css'],
    doc: 'The lead sheet\'s foot: Insert example, Preview this score or Plan and preview a score, and whether the text reads as ABC.',
    responsible: 'Recognising ABC (a K: line and bars), counting lines, opening the score preview (the score modal), and the way back when the planner is set to write nothing.',
  },
  cartridges: {
    kind: 'bespoke', binds: ['state'], css: ['25-module-adapters.css'],
    opts: { flat: 'Quick\'s flat cartridge: the first adapter lying flat, its art beside its name.' },
    doc: 'Adapter cartridges: each loaded adapter as a cassette with label art, its strength fader, its kind\'s housing, and in a stack its effective share, active phase and per-section mask.',
    responsible: 'The slots state (which files are loaded, per engine), adding and removing cartridges, the label art, phase windows and section masks (saved in the lyrics), and naming those steps for undo. Every dial on a cartridge is a declared parameter (s0..s2, ad.arp, ad.arc) drawn with the fader part.',
  },
  buildBoard: {
    kind: 'bespoke', binds: ['state'], css: ['25-module-build.css'],
    doc: 'Build by section: the planned sections as columns, takes under each, keep one, render the song.',
    responsible: 'The build record (planned score, takes per section, keeps, the render), its drift against the lyrics, its own undo steps (Plan, Keep, Discard, Render, Start a new build), and talking to the build session API. Its four settings are declared parameters.',
  },
  recordingSpan: {
    kind: 'bespoke', binds: ['state'], css: ['25-module-cont.css'],
    doc: 'A recording\'s waveform with up to 30 seconds selected: drag inside to move, an edge to resize, outside to draw a new one.',
    responsible: 'The span state and its keyboard contract (arrows move, Shift resizes, Page moves by its length), the ruler under it, and the step names for it.',
  },
  repaintRegion: {
    kind: 'bespoke', binds: ['state'], css: ['25-module-repaint.css'],
    doc: 'The source song\'s waveform with the region to regenerate.',
    responsible: 'The source song and region state, drawing the waveform, the region\'s handles and keys, and naming its steps in the song\'s times.',
  },
  lyricSheet: {
    kind: 'bespoke', binds: ['text'], css: ['25-module-lyrics.css'],
    doc: 'The ruled lyric paper with its section tabs, the language stamp, the structure strip under it and the line count.',
    responsible: 'Inserting section tags at line boundaries, the structure strip (sections as coloured segments that jump to their line), the line count and its time estimate, wildcard expansion. lyrics.text and lyrics.instr are ordinary parameters.',
  },
  promptBox: {
    kind: 'bespoke', binds: ['text'], css: ['22-prompt.css'],
    doc: 'The main text box (description, caption, prompt or tags) with its tools and the song info.',
    responsible: 'The box, its rotating placeholder, Write it for me and Compose caption, the caption source; its field undo is the generic text-field contract.',
  },
  modelPicker: {
    kind: 'bespoke', binds: ['choice'], css: ['15-pick.css', '30-engine-panel.css'],
    doc: 'The engine panel\'s model rows, with the VRAM meter they add up to.',
    responsible: 'Listing installed models from the server, the VRAM figures, and naming model steps by role (Sound model, Planner model). Each model choice is an ordinary choice parameter.',
  },
} as const satisfies Record<string, PartSpec>;

export type PartName = keyof typeof PARTS;

/** The text checks a parameter may name (types.ts Check). */
export const VALIDATORS = {
  seed: 'A whole number from 0 to 4294967295.',
  seedOrMinusOne: 'A whole number from 0 to 4294967295, or -1.',
  seedOrEmpty: 'A whole number from 0 to 4294967295, or empty.',
  timesteps: 'Numbers from 0 to 1, comma-separated, at least two, each smaller than the last.',
} as const;

/** Desk state a condition may name besides the definitions' own StateDef keys: facts the page works out. */
export const DESK_STATES = {
  adapters: 'How many adapters are loaded on this engine (atLeast).',
  runtimeForced: 'Adapters are forced to runtime mode: timestep windows, or section directives with a stack of two or more.',
  sectionDirectives: 'The lyrics carry per-section adapter directives.',
  customTimesteps: 'Latent & correction is on the desk with two or more valid custom timesteps, which replace Steps and Schedule.',
  vstPlugins: 'How many VST3 plugins are in the chain (atLeast).',
} as const;

/** The containers every definition gets without naming them: drawn by the shell from the definition itself. */
export const SHELLS = {
  tile: { css: ['20-tile.css', '21-desk.css'], doc: 'The tile: header with grip, group and name, its × , the body, the "Also set" foot, the unsupported note, and the one-line strip a tile becomes when it has nothing to show.' },
  pedal: { css: ['27-pedals.css', '27-pedalboard.css'], doc: 'The pedal: face in its hue, name, LED, foot switch (or "Always on"), remove tab, jacks and cables; 1U or 2U at Lab (2U from three knobs), one size at Studio (132 x 196) and at Quick (112 wide). Its Lab body is laid out from its parameters: knobs in 64 px cells, then selectors, then the definition\'s lab nodes, then switches with what they reveal.' },
  sideUnit: { css: ['28-side-units.css'], doc: 'The side unit: a rack strip with a name column, its object, its controls at Lab, a power button and a remove button.' },
  board: { css: ['27-pedalboard.css'], doc: 'The Pedalboard tile: chain switch, count, the board with cables, the side units row.' },
} as const;
