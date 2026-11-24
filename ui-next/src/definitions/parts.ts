// The parts inventory: every part a definition may name in a face, what each one edits, reads and takes, and the
// other lists a definition draws from (units, text checks, option sources, desk facts, named actions).
//
// A part is one of four kinds:
//   control    edits a parameter (or two): its value contract, changed marker, tip and undo come from the parameter
//   container  holds other nodes and gives them the design language's housing (a rack, a well, an LCD, a terminal)
//   readout    shows something, edits nothing
//   bespoke    a part written for one module because what it draws or does cannot be said as data. It edits only
//              what its node binds (param, params, state), reads only what its node lists in `reads`, and names no
//              undo step of its own: PartProps below is all it is given.
//
// `edits` lists the value types a node's `param`/`params` may have ('pair' is two numbers in `params`), `state` the
// kinds of state its node's `state` may name, `reads` whether it takes a `reads` list, `drive` how the generated
// tests operate it (docs/05, "Automated tests"). `opts` lists the options a node may pass, each with its type and
// what it does: the type check refuses an option a part does not list or of the wrong type. `css` names the
// concept's stylesheets (C:\Users\sneth\misc\HOT-Step-UI\src\css) that hold its look; the first is its own file.
// Every part also takes 35-changed.css, 35-states.css and 35-labels.css where it has a label.
//
// To add a part: add an entry here, write its React component under the name (it receives PartProps), and give it
// its stylesheet. Every definition can then name it. docs/06-how-to.md has the steps.

import type { StateDef, ValueType, Tier, EngineId, Json, Value, Node, ParamFace } from './types.ts';
import type { Slot } from '../derive/messages.ts';

export type PartKind = 'control' | 'container' | 'readout' | 'bespoke';
export type Drive = 'keys' | 'click' | 'choose' | 'type' | 'file';

/** An option a node may pass to its part: its type, and what it does. */
export interface OptSpec {
  /** boolean, number, text, texts (a list of words), param (a parameter key), or a list of the words it may be. */
  type: 'boolean' | 'number' | 'text' | 'texts' | 'param' | readonly string[];
  doc: string;
}

export interface PartSpec {
  kind: PartKind;
  /** What its node's `param`/`params` may edit. */
  edits?: readonly (ValueType | 'pair')[];
  /** The kinds of state its node's `state` may name. */
  state?: readonly StateDef['kind'][];
  /** It shows values it does not edit, listed in its node's `reads`: any (true), or exactly these first, in order, for a
   *  part whose drawing or formula uses each (a parameter of that type, or 'state'). */
  reads?: true | readonly (ValueType | 'state')[];
  /** How the generated tests operate it. Every part that edits has one. */
  drive?: Drive;
  /** What it is for, in one or two sentences. */
  doc: string;
  /** The states it draws beyond the shared ones (changed, focus, hover, dim, hidden). */
  states?: readonly string[];
  opts?: { readonly [name: string]: OptSpec };
  css: readonly string[];
  /** For a bespoke part: exactly what it is responsible for, and what it leaves to the generic machinery. */
  responsible?: string;
  /** How a hand or a key works it, said at the end of its tip ("Drag up or down ..."). */
  howTo?: string;
  /** For a bespoke part: the named steps it takes, which its node's state or first parameter must declare. A file's
   *  `load` and `clear` are declared without saying. */
  steps?: readonly string[];
}

export const PARTS = {
  // ---------------------------------------------------------------- controls
  knob: {
    kind: 'control', edits: ['number'], drive: 'keys', css: ['15-knob.css', '27-pedals.css'],
    howTo: 'Drag up or down (hold Shift for fine steps), scroll, or use the arrow keys. Double-click to reset.',
    doc: 'A rotary knob with a detent tick at the default, its label under it and a read-out that can be clicked to type a value. The rack face of every number in Lab.',
    states: ['turning', 'resetting', 'auto (drawn at the auto value in the auto colour, with an auto pill)', 'typing'],
    opts: { small: { type: 'boolean', doc: 'The small knob (rack rows).' } },
  },
  fader: {
    kind: 'control', edits: ['number'], drive: 'keys', css: ['15-fader.css', '25-module-adapters.css'],
    doc: 'A vertical fader with a cap, tick marks and a unity line at 1.00. Used in fader banks (adapter group scales, depth thirds) and cartridges.',
    states: ['sliding', 'stops (travel beyond max shown as a stop)'],
    opts: { travel: { type: 'number', doc: 'The value the top of the slot stands for, when the slot runs past the maximum (Plan: max 1, travel 2).' } },
  },
  ruler: {
    kind: 'control', edits: ['number'], drive: 'keys', css: ['15-ruler.css'], reads: true,
    doc: 'A length on a minutes timeline, dragged along; labels on round steps. The length of a song. With markers it reads Beat I/O, its bars, the BPM, Auto-trim and the prompt, in that order.',
    states: ['auto (shows the estimate, or "up to" where the model ends the song)', 'sliding', 'at-start/at-end caption'],
    opts: {
      scaleMax: { type: 'number', doc: 'The seconds the timeline spans. Default: the parameter maximum.' },
      markers: { type: 'boolean', doc: 'Draw Beat I/O and trim markers from what it reads (Beat I/O, its bars, the BPM, Auto-trim, the prompt). Never drawn at Quick.' },
    },
  },
  xyPad: {
    kind: 'control', edits: ['pair'], drive: 'keys', css: ['15-xy-pad.css'],
    doc: 'Two settings that belong together on one plane: a puck with a trail, a grid, axis read-outs that each carry their own changed marker, corner words outside the pad. An axis whose parameter is absent on the engine leaves a one-axis slider.',
    states: ['grab', 'release', 'nudge', 'y-off (the Y minimum drawn as an off rail)'],
    opts: {
      xl: { type: 'text', doc: 'The X axis words.' },
      yl: { type: 'text', doc: 'The Y axis words.' },
      corners: { type: 'texts', doc: 'Four corner words: top left, top right, bottom left, bottom right.' },
      yOff: { type: 'text', doc: 'Words on the rail when the Y minimum means off.' },
      look: { type: ['apg', 'heat', 'overlay', 'term'], doc: 'The plot colouring.' },
    },
  },
  numberBox: {
    kind: 'control', edits: ['number'], drive: 'keys', css: ['15-numbox.css'],
    doc: 'A typed number box that can also be dragged up and down. Steps in the Sampler well.',
    states: ['editing', 'overridden (a value another module replaces)', 'empty means auto'],
    opts: { seconds: { type: 'boolean', doc: 'Show an estimate of seconds under it (Steps: "≈ 1.3 s").' } },
  },
  lcdNumber: {
    kind: 'control', edits: ['number'], drive: 'keys', css: ['25-module-tempo.css', '15-lcd.css'],
    doc: 'A big seven-segment number on an LCD that can be typed or dragged, with its unit beside it, and AUTO beside it when the parameter has auto. BPM.',
  },
  pick: {
    kind: 'control', edits: ['choice'], drive: 'choose', css: ['15-pick.css', '30-dropdown.css'],
    doc: 'The button that shows a choice and opens its dropdown. The variant sets where it lives.',
    states: ['open', 'overridden'],
    opts: {
      variant: { type: ['plugin', 'lcd', 'chip', 'model', 'stamp'], doc: 'plugin: a periodic-table tile and the kind above the value (solver, schedule, guidance); lcd: an LCD with previous and next arrows; chip: a pill with its label; model: name, quant badge, size and a memory bar; stamp: an ink stamp (the lyric language).' },
      title: { type: 'text', doc: 'The heading of its list (plugin: Solvers; stamp: Sung in…). Default: the label of its parameter.' },
    },
  },
  chips: {
    kind: 'control', edits: ['choice'], drive: 'choose', css: ['15-controls.css'],
    doc: 'A chip selector: up to four options as lit chips under a caps label. A side unit\'s short choice at Lab.',
  },
  segmented: {
    kind: 'control', edits: ['choice'], drive: 'choose', css: ['15-controls.css'],
    doc: 'A segmented choice: a row of joined buttons, one pressed.',
    opts: { size: { type: ['sm'], doc: 'sm: the compact row inside a rack group.' } },
  },
  termChips: {
    kind: 'control', edits: ['choice'], drive: 'choose', css: ['25-module-planner.css'],
    doc: 'A choice written as terminal chips after its flag (--mode Presence Frequency DRY).',
  },
  lcdChips: {
    kind: 'control', edits: ['choice'], drive: 'choose', css: ['25-module-tempo.css'],
    doc: 'Numbered chips on an LCD with a word after them (1 2 4 8 BARS).',
    opts: { word: { type: 'text', doc: 'The word printed after the chips.' } },
  },
  ledButton: {
    kind: 'control', edits: ['boolean'], drive: 'click', css: ['15-controls.css'],
    doc: 'An LED switch: a rack button with a lamp that lights when on. Lab rack face of a boolean.',
  },
  pill: {
    kind: 'control', edits: ['boolean'], drive: 'click', css: ['15-controls.css'],
    doc: 'A pill toggle with a lamp, used inside a module look (Instrumental, Level match, a rack group power switch).',
  },
  flag: {
    kind: 'control', edits: ['boolean'], drive: 'click', css: ['25-module-planner.css'],
    doc: 'A terminal flag with a check box: [x] --cot.',
  },
  lcdButton: {
    kind: 'control', edits: ['boolean'], drive: 'click', css: ['25-module-tempo.css', '15-lcd.css'],
    doc: 'A lit word on an LCD that switches (BEAT I/O, AUTO-TRIM).',
  },
  recButton: {
    kind: 'control', edits: ['boolean'], drive: 'click', css: ['25-module-plans.css'],
    doc: 'A record button (● SAVE NEXT) that arms the next run to save something.',
    states: ['is-off (stands back: nothing new to save)'],
  },
  powerButton: {
    kind: 'control', edits: ['boolean'], drive: 'click', css: ['25-module-planner.css'],
    doc: 'A terminal power switch with its word (● planner) and the scope switch (● Lua plugins): the switch a whole instrument hangs on.',
    opts: {
      look: { type: ['term', 'scope'], doc: 'term: the terminal bar; scope: on a scope read-out.' },
      name: { type: 'text', doc: 'Its accessible name, where its printed word alone does not say what it switches ("Post-processing chain" for "Chain").' },
    },
  },
  lockButton: {
    kind: 'control', edits: ['boolean'], drive: 'click', css: ['15-controls.css', '25-module-seed.css'],
    doc: 'A padlock button that keeps a value (Keep this seed).',
  },
  powerChip: {
    kind: 'control', edits: ['number'], drive: 'click', css: ['15-controls.css', '25-module-adhere.css'],
    doc: 'A rack group power chip for a group switched by a number: off is one value, on sets a moderate one. Says "· off" on itself when off.',
    opts: {
      off: { type: 'number', doc: 'The value that means off.' },
      on: { type: 'number', doc: 'The value pressing it on sets.' },
    },
  },
  token: {
    kind: 'control', edits: ['number'], drive: 'keys', css: ['15-numbox.css', '25-module-planner.css'],
    howTo: 'Drag sideways, use the arrow keys, or press Enter to type',
    doc: 'A scrub token: a number in running text or a terminal line, dragged sideways, stepped with the arrows, typed on Enter. In a terminal line its label is its flag.',
    states: ['clash (another setting makes it skip)'],
  },
  textField: {
    kind: 'control', edits: ['text'], drive: 'type', css: ['15-controls.css', '15-lcd.css'],
    doc: 'A one-line text field. Checked fields show their error beside them and commit on Enter or leaving.',
    states: ['bad (fails its check)', 'kept (invalid text kept, what is in force said)'],
    opts: {
      look: { type: ['plain', 'lcd', 'term'], doc: 'plain; lcd: an LCD input; term: an inline terminal input.' },
      size: { type: 'number', doc: 'Width in characters.' },
    },
  },
  textArea: {
    kind: 'control', edits: ['text'], drive: 'type', css: ['15-controls.css'],
    doc: 'A text area. The look gives it its material.',
    opts: { look: { type: ['plain', 'note', 'staff', 'paper', 'term'], doc: 'note: the taped sticky note (Avoid); staff: the ruled ABC sheet (Lead sheet); paper: the lyric sheet; term: a quoted one-line terminal string that wraps (--neg).' } },
  },
  choiceRow: {
    kind: 'control', edits: ['number', 'choice'], drive: 'choose', css: ['15-controls.css'],
    doc: 'Quick\'s plain choice row: a radio group of words that each set a detailed value, or several. A value between the choices checks none: a marker stands at its place and the label says the value.',
    opts: { look: { type: ['plain', 'mood'], doc: 'mood: the coloured How adventurous row.' } },
  },
  lookback: {
    kind: 'control', edits: ['number'], drive: 'keys', css: ['25-module-repetition.css'],
    doc: 'A look-back window drawn as a strip of codes that grows leftward into the past, with a time scale under it.',
    opts: {
      unit: { type: 'text', doc: 'The word for what it counts (codes, frames, tokens).' },
      perSecond: { type: 'number', doc: 'How many a second, for the scale and the tip.' },
      who: { type: 'text', doc: 'Who checks for repeats, for the tip (the planner).' },
    },
  },
  dropZone: {
    kind: 'control', state: ['file'], drive: 'file', css: ['15-controls.css', '25-module-timbre.css', '27-pedals.css'],
    doc: 'A file drop zone: a record sleeve (modules) or a slot (pedals) that takes a dropped or chosen audio file, shows its name cut to fit, and clears with ×. Its node\'s `state` is the file it holds.',
    opts: {
      look: { type: ['sleeve', 'slot'], doc: 'A record sleeve or a pedal slot.' },
      empty: { type: 'text', doc: 'Its words while empty.' },
      hint: { type: 'text', doc: 'Its second line while empty.' },
      from: { type: 'text', doc: 'Shows a file another part lends it (its node\'s `state` is that part\'s file): the name, then these words, and it takes no file of its own.' },
    },
  },
  // ---------------------------------------------------------------- containers
  row: {
    kind: 'container', css: ['15-knob.css'],
    doc: 'A row of knobs or controls with even spacing (the .knobs row).',
  },
  rack: {
    kind: 'container', css: ['15-controls.css'],
    doc: 'A rack row: LED switches and chip selectors side by side on a rack strip.',
  },
  well: {
    kind: 'container', css: ['25-module-sampler.css'],
    doc: 'A recessed well that holds a module\'s main controls under its instrument (the Sampler strip).',
  },
  lcd: {
    kind: 'container', css: ['15-lcd.css', '25-module-tempo.css'],
    doc: 'An LCD panel. Its children are rows (lcdRow) or LCD parts.',
  },
  lcdRow: {
    kind: 'container', css: ['25-module-tempo.css'],
    doc: 'One row of an LCD. Ends-aligned when it holds switches at both ends.',
    opts: { ends: { type: 'boolean', doc: 'Spread to both ends.' } },
  },
  terminal: {
    kind: 'container', css: ['25-module-planner.css'],
    doc: 'A terminal: dark, monospaced, a command bar on top, lines below. The planner and the repetition guard.',
    opts: { title: { type: 'text', doc: 'The tip on the whole terminal.' } },
  },
  termBar: {
    kind: 'container', css: ['25-module-planner.css'],
    doc: 'The terminal\'s top bar: its power switch and flags.',
  },
  termLine: {
    kind: 'container', css: ['25-module-planner.css'],
    doc: 'A command line: "$ <command>" then its children, each control after its label printed as a flag (--temp). Wrapped lines hang under the first flag.',
    opts: { command: { type: 'text', doc: 'The command word after $.' } },
  },
  group: {
    kind: 'container', css: ['25-module-adhere.css', '25-module-latent.css'],
    doc: 'A rack group: a boxed set of controls with a header that may carry its power switch; a group switched off stands back.',
    opts: {
      title: { type: 'text', doc: 'Its caps heading when it has no power switch.' },
      look: { type: ['rack', 'strip', 'plugin'], doc: 'rack (default): a boxed group, its switch or heading on top; strip: an inset strip with its caps heading (the Sampler\'s schedule controls); plugin: a plugin\'s controls under its name, "experimental" and its Reset action.' },
    },
  },
  foot: {
    kind: 'container', css: ['20-tile.css'],
    doc: 'A tile\'s foot row: small labelled controls and a note, kept to the bottom of the tile.',
  },
  labelled: {
    kind: 'container', css: ['15-controls.css'],
    doc: 'A caps label beside one small control (the foot-ctl and lcdsel pairing). Where it stands and what it holds give it the pairing the concept draws there: above a text field, a strip of switches or dice; in a rack group the pair of a token; in a deck a row of its two columns.',
    opts: { look: { type: ['strip'], doc: 'strip: an inset strip, its caption first and what it holds after it on one line (Retake seed in the Sampler).' } },
  },
  deck: {
    kind: 'container', css: ['25-module-plans.css'],
    doc: 'A two-column grid of labels and controls, so a choice never moves the rows under it (Plans & endings).',
  },
  pair: {
    kind: 'container', css: ['25-module-adhere.css'],
    doc: 'Two tokens as a range: from → to, under one label.',
  },
  bank: {
    kind: 'container', css: ['15-fader.css', '25-module-adapters.css'],
    doc: 'A fader bank with a 2-1-0 scale and a heading, each fader labelled on tape.',
    opts: {
      title: { type: 'text', doc: 'Its heading.' },
      note: { type: 'text', doc: 'Its heading\'s tip.' },
    },
  },
  // ---------------------------------------------------------------- read-outs
  note: {
    kind: 'readout', css: ['01-base.css', '15-controls.css'],
    doc: 'A quiet line of words: a hint, why something is off, what another module changes. Its actions follow as links.',
    opts: { after: { type: 'text', doc: 'badge: the words after the badge, on its line.' }, look: { type: ['plain', 'warn', 'cap', 'hint', 'badge'], doc: 'warn; cap: the caption note under an instrument; hint: Quick\'s hint.' } },
  },
  link: {
    kind: 'readout', css: ['14-buttons.css'],
    doc: 'A text link that runs an action (Insert example, Show in Lab, Turn on Lua plugins).',
  },
  button: {
    kind: 'readout', css: ['14-buttons.css'],
    doc: 'A small button that runs an action (+ Load adapter, Roll once).',
  },
  lcdValue: {
    kind: 'readout', reads: true, css: ['25-module-tempo.css'],
    doc: 'A value printed on an LCD under a caps word (KEY D MAJOR): the first of `reads` written as everywhere else, or its `text`.',
  },
  unitLcd: {
    kind: 'readout', reads: true, css: ['28-side-units.css'],
    doc: 'A side unit\'s two-line LCD: each of `reads` on its own line, a choice as its option\'s LCD words, or its `text` as one line.',
    opts: {
      prefix: { type: 'text', doc: 'Words before the first line (WHISPER ).' },
      notAtLab: { type: 'boolean', doc: 'Not drawn at Lab, where the unit\'s own controls show the same values.' },
    },
  },
  unitStatus: {
    kind: 'readout', css: ['28-side-units.css'],
    doc: 'A side unit\'s status words in small caps (After the render · needs MMS).',
  },
  unitBadge: {
    kind: 'readout', css: ['28-side-units.css'],
    doc: 'A side unit\'s round score badge.',
  },
  dice: {
    kind: 'readout', reads: ['number'], css: ['25-module-replan.css'],
    doc: 'A row of numbered dice, one per allowed attempt (the first of `reads`), filled as attempts are used.',
    opts: { offWhen: { type: 'param', doc: 'A switch that, while off, draws the dice off.' } },
  },
  meter: {
    kind: 'readout', reads: true, css: ['27-pedals.css'],
    doc: 'A printed meter with a needle: the first of `reads` on its scale, or a level the engine reports when it reads nothing (the Mastering pedal\'s VU meter).',
    opts: {
      min: { type: 'number', doc: 'The scale\'s left end.' },
      max: { type: 'number', doc: 'Its right end.' },
      red: { type: 'number', doc: 'Where the scale turns red.' },
    },
  },
  beatPulse: {
    kind: 'readout', reads: ['number'], css: ['25-module-tempo.css'],
    doc: 'Lamps that pulse on the beat at the BPM it reads, one per beat of the bar (the time signature it reads next; four without one).',
  },
  flowWindows: {
    kind: 'readout', reads: ['number'], css: ['25-module-flow.css'],
    doc: 'MiniMax-Music3\'s flow windows drawn at the hop it reads, and the count of windows a 3:00 song needs.',
  },
  speedGauge: {
    kind: 'readout', reads: ['number', 'number', 'number'], css: ['25-module-perf.css'],
    doc: 'An LCD that estimates how much faster the trades it reads make a run.',
  },
  adhereCurve: {
    kind: 'readout', reads: ['number', 'number', 'choice', 'number', 'number', 'number', 'number', 'number', 'number'], css: ['25-module-adhere.css'],
    doc: 'Plan strength across the run, from noise to clean, with the Plan CFG band and the CADS ramp, from the values it reads.',
  },
  planText: {
    kind: 'readout', reads: true, css: ['25-module-planner.css'],
    doc: 'The plan the planner will start from, as terminal text: caption, tempo, key, form, and whose each value is.',
    opts: { look: { type: ['plan', 'prose', 'compose', 'tags'], doc: 'plan: caption, tempo, key and form, reading the caption, the lyrics, the BPM, the key, the planner switch and its chain of thought; prose: tempo and key come from the caption (the caption, the lyrics); compose: the score the chain of thought writes (the caption, the lyrics, its mode); tags: the tags, the sections of the lyrics and the length (the caption, the lyrics, the length).' } },
  },
  lrcPreview: {
    kind: 'readout', reads: ['text'], css: ['28-side-units.css'],
    doc: 'Two timed lyric lines, the first named after the first section of the lyrics it reads.',
  },
  coverSwatch: {
    kind: 'readout', css: ['28-side-units.css'],
    doc: 'A swatch in the newest song\'s colour, standing for its cover art.',
  },
  // ---------------------------------------------------------------- bespoke
  samplerScope: {
    kind: 'bespoke', edits: ['number'], reads: ['number', 'number', 'choice'], drive: 'keys', css: ['25-module-sampler.css'],
    doc: 'The Sampler\'s scope: the noise schedule drawn from the values it reads, the guidance band, and at Lab the CFG interval\'s two handles.',
    opts: { interval: { type: 'boolean', doc: 'Draw the CFG interval\'s two handles; they edit the node\'s two params.' } },
    responsible: 'Drawing sigma(t) from its reads (steps, shift, schedule and the shape parameters, or the custom timesteps when Latent & correction sets them); the guidance band from the CFG interval and the hatched CFG cutoff; the two interval handles, which set its two params a step apart; the one-line read-out. Defaults, undo, changed markers and saving of every value are generic.',
  },
  keyWheel: {
    kind: 'bespoke', edits: ['choice'], drive: 'choose', css: ['25-module-tempo.css'],
    doc: 'The circle of fifths: majors outside, minors inside, Auto in the middle, a radio group driven by the arrows. It stands beside the LCD it holds as its child (.tempo); with nothing to edit (Quick, an engine without a key) the LCD stands alone.',
    responsible: 'Drawing the two rings from its param\'s options, spelling keys as a musician does (B♭, F♯), the arrow-key walk around and between rings, and the selection nudge.',
  },
  tapTempo: {
    kind: 'bespoke', edits: ['number'], drive: 'click', css: ['25-module-tempo.css'],
    doc: 'TAP: two taps or more within 2.5 s set the BPM from their spacing.',
    responsible: 'Timing the taps and setting its param (clamped to its range); the run of taps is one undo step through the parameter\'s `run`.',
  },
  capChips: {
    kind: 'bespoke', edits: ['text'], reads: ['number', 'choice'], steps: ['addChip', 'removeChip'], drive: 'click', css: ['25-module-tempo.css'],
    doc: 'On engines where tempo and key go in the prompt: chips that add or take out "120 BPM" or "D major" as items of the prompt text it edits, two exclusive groups.',
    responsible: 'Reading which chips are pressed from the text, replacing an item of the same group, never touching words inside a sentence, and offering the tempo and key it reads first. Each press is one step of the text it edits.',
  },
  takesReels: {
    kind: 'bespoke', edits: ['number'], drive: 'keys', css: ['25-module-takes.css'],
    doc: 'The takes count as an LCD and a set of reels, one per take the engine allows; the reels are one slider. Under it, the VRAM the count costs.',
    responsible: 'Drawing as many reels as its param\'s maximum, the hover preview, the reel spin, and the VRAM cost line from the loaded models. The LCD reads the value as its format writes it, then the node\'s label.',
  },
  splitFlap: {
    kind: 'bespoke', edits: ['boolean'], state: ['seed'], reads: ['boolean'], steps: ['set'], drive: 'click', css: ['25-module-seed.css'],
    doc: 'The seed as ten split-flap cells that flip to a new number; pressing it types one. Roll once and Random each time.',
    responsible: 'The flap animation, the ten-cell layout, typing or pasting a seed (checked 0 to 2^64-1, kept as its decimal text past 2^53), Roll once and Random each time. Each sets the seed and its lock (its param) in one step, the seed state\'s `set` step.',
  },
  savedSeeds: {
    kind: 'bespoke', edits: ['boolean'], state: ['list', 'seed'], steps: ['save', 'update', 'delete', 'star', 'unstar', 'set'], drive: 'click', css: ['25-module-seed.css'],
    doc: 'The saved seeds list: search, star, name, describe, tag, load, delete, random from saved.',
    responsible: 'The list (its `list` state) and its search; loading a saved seed sets the seed and its lock in the seed state\'s `set` step; every list action is the list state\'s step of that name.',
  },
  timbreBlend: {
    kind: 'bespoke', edits: ['number'], drive: 'keys', css: ['25-module-timbre.css'],
    doc: 'Two overlapping circles, the reference dragged into your sound: the reference blend.',
    responsible: 'Drawing the circles at its param and dragging sideways.',
  },
  recentRefs: {
    kind: 'bespoke', state: ['list', 'file'], steps: ['load'], drive: 'click', css: ['25-module-timbre.css'],
    doc: 'The two latest references as chips, and the rest in a popover.',
    responsible: 'Showing its `list` state and loading one into its `file` state (that state\'s `load` step).',
  },
  latentCurve: {
    kind: 'bespoke', reads: ['number', 'number'], css: ['25-module-latent.css'],
    doc: 'The latent distribution before (grey) and after (green) the shift and rescale it reads, with the XY pad that sets them laid over it as its child.',
    responsible: 'Drawing the two curves and hosting its child. It edits nothing; the pad is the generic xyPad part.',
  },
  abcFoot: {
    kind: 'bespoke', reads: ['text', 'choice'], css: ['25-module-score.css'],
    opts: { chords: { type: 'text', doc: 'The plan mode (its second read) at which the planner writes chord symbols over the melody.' } },
    doc: 'The lead sheet\'s foot: whether the text it reads is ABC, how many lines, and its actions as they apply (Insert example, Preview this score or Plan and preview a score, the way back when the planner writes nothing).',
    responsible: 'Recognising ABC (a K: line and bars) and counting lines, and choosing which of its node\'s actions apply. It edits nothing itself.',
  },
  cartridges: {
    kind: 'bespoke', edits: ['number', 'choice', 'text'], state: ['slots'], steps: ['load', 'remove'], drive: 'keys', css: ['25-module-adapters.css'],
    opts: { flat: { type: 'boolean', doc: 'Quick\'s flat cartridge: the first adapter lying flat, its art beside its name.' } },
    doc: 'Adapter cartridges: each loaded adapter as a cassette with label art, its strength fader, its kind\'s housing, and in a stack its effective share, active phase and per-section mask.',
    responsible: 'Showing its slots state and adding and removing cartridges (the state\'s `load` and `remove` steps: loading a second adapter while Stack is Single sets Stack, removing one moves the later strengths up and renumbers the lyric directives), each slot\'s active phase (kept in its slot), the label art, and drawing its params with the fader part. Section masks are written into the lyrics text it edits.',
  },
  buildBoard: {
    kind: 'bespoke', state: ['record'], reads: ['text', 'choice'], steps: ['plan', 'replan', 'relyric', 'keep', 'unkeep', 'discard', 'render', 'reset'], css: ['25-module-build.css'],
    opts: { ownChords: { type: 'text', doc: 'The section mode (its fourth read) at which each take gets chords and a melody of its own; at any other, every take follows the score.' } },
    doc: 'Build by section: the planned sections as columns, takes under each, keep one, render the song.',
    responsible: 'Showing its record state (planned score, takes per section, keeps, the render) and its drift against the lyrics it reads, and talking to the build session API. Every change is the record state\'s step of that name.',
  },
  recordingSpan: {
    kind: 'bespoke', state: ['span'], reads: ['state'], steps: ['move'], css: ['25-module-cont.css'],
    doc: 'A recording\'s waveform with up to 30 seconds selected: drag inside to move, an edge to resize, outside to draw a new one.',
    responsible: 'Its span state and its keyboard contract (arrows move, Shift resizes, Page moves by its length), and the ruler under it. A move is the span\'s step.',
  },
  repaintRegion: {
    kind: 'bespoke', state: ['record', 'span'], steps: ['pick', 'move'], css: ['25-module-repaint.css'],
    doc: 'The source song\'s waveform with the region to regenerate.',
    responsible: 'Its source record and region span, drawing the waveform, and the region\'s handles and keys. A move is the span\'s step, in the song\'s times.',
  },
  lyricSheet: {
    kind: 'bespoke', edits: ['text', 'choice', 'boolean'], reads: ['boolean', 'number'], drive: 'type', css: ['25-module-lyrics.css'],
    doc: 'The ruled lyric paper with its section tabs, the language stamp, the structure strip under it and the line count.',
    responsible: 'Inserting section tags at line boundaries, the structure strip (sections as coloured segments that jump to their line), the line count and its time estimate, and wildcard expansion. Every value it edits is one of its params.',
  },
  promptBox: {
    kind: 'bespoke', edits: ['text', 'choice', 'boolean'], state: ['list', 'file'], steps: ['write', 'compose', 'copy', 'addTag', 'removeTag', 'expand', 'toLyrics', 'tempo', 'plain', 'preset', 'load', 'clear'], drive: 'type', css: ['22-prompt.css'],
    doc: 'The main text box (description, caption, prompt or tags) with its tools and the song info.',
    responsible: 'The box, its rotating placeholder, Write it for me and Compose caption, and the caption source. Every value it edits is one of its params.',
  },
  modelPicker: {
    kind: 'bespoke', edits: ['choice'], drive: 'choose', css: ['15-pick.css', '30-engine-panel.css'],
    doc: 'The engine panel\'s model rows, with the VRAM meter they add up to.',
    responsible: 'Listing installed models from the server and the VRAM figures. Each model choice is one of its params; its undo name is the parameter\'s.',
  },
} as const satisfies Record<string, PartSpec>;

export type PartName = keyof typeof PARTS;

/* ---- a node's options, typed from the part's OptSpec, so a wrong option fails where it is typed ---- */
type OptValue<T> = T extends 'boolean' ? boolean : T extends 'number' ? number : T extends 'text' | 'param' ? string
  : T extends 'texts' ? string[] : T extends readonly (infer W)[] ? W : never;
type SpecOf<P extends PartName> = (typeof PARTS)[P];
export type OptsOf<P extends PartName> = SpecOf<P> extends { opts: infer O }
  ? { [K in keyof O]?: O[K] extends { type: infer T } ? OptValue<T> : never } : Record<string, never>;
export type NodeOf<P extends PartName> = { part: P; opts?: OptsOf<P> };

/* ---------------------------------------------------------------- what a part is given
   A part's component receives PartProps and nothing else (src/derive/parts.ts bindPart builds them). The only way it
   changes the desk is a setter here, and there is one for exactly what its node binds: `param` and `params`, the
   state in its node's `state`, and the parameters a named step it takes lists in `changes`. It never names an undo
   step: a setter's change is named by the core from the definition (src/derive/steps.ts), or the part asks for one of
   its definition's named steps by its name. Several setters inside one `batch` are one step (an XY pad's drag). */

/** A parameter as the part sees it on the engine in use: its resolved face, its value, and how to change it. */
export interface BoundParam {
  key: string;
  face: ParamFace;
  value: Value;
  /** The value written as everywhere else, as a message (src/derive/messages.ts say). */
  text: Slot;
  changed: boolean;
  /** On auto, for a parameter with auto. */
  auto: boolean;
  /** Its `when` holds. */
  applies: boolean;
  /** Sets it (cleaned, on the engine's step grid, a pair's end kept from the other), one undo step. With `step`, the
   *  step is the parameter's named step of that name, its slots filled from `slots`, and `also` may set the parameters
   *  that step lists in `changes`. */
  set(value: Value, step?: string, slots?: { [slot: string]: string | number }, also?: { [param: string]: Value }): void;
  setAuto(on: boolean): void;
  reset(): void;
  /** The value contract's keys (arrows, Shift, Page Up and Down, Home, End) on a number: true when the key was one of
   *  them and the value moved by it (src/derive/models.ts keyStep, the contract's one statement). */
  press(key: string, shift: boolean): boolean;
  /** A number typed in: read as the page writes numbers (parseTyped); outside its range it is put at the end and the
   *  page says so, and what is no number is said and changes nothing. */
  typed(text: string): void;
  /** A hand's place along a number's range, 0 to 1. */
  toFraction(f: number): void;
  /** A hand dragging a scrub token sideways: `px` pixels on from the value it started at (src/derive/controls.ts
   *  scrubValue: a wide range moves with the value). */
  scrub(from: number, px: number, shift: boolean): void;
  /** A hand on a ruler: its place `f` along a scale spanning `span` from 0, on a track `px` pixels wide; a length is
   *  snapped to round values (src/derive/controls.ts rulerSnap). */
  along(f: number, span: number, px: number): void;
}

/** A state the node binds. `apply` takes the name of one of its StateDef's steps; the step's words come from the
 *  definition. `also` may set the node's own parameters and those the step lists in `changes` (Roll once sets the
 *  seed and its lock; loading a second adapter stacks them). A state another definition lends (a drop zone's
 *  `from`) can be read and not applied. */
export interface BoundState {
  key: string;
  kind: StateDef['kind'];
  value: Json;
  apply(step: string, next: Json, also?: { [param: string]: Value }, slots?: { [slot: string]: string | number }): void;
}

/** What a node reads: a parameter (its value, text, auto switch, whether it applies and whether its definition is on
 *  the desk: the lyric sheet's time estimate at the tempo in force), a state's value, or a desk fact. */
export interface ReadValue {
  key: string;
  value: Value | Json;
  text: Slot;
  auto: boolean;
  applies: boolean;
  onDesk: boolean;
}

/** What a part's component receives. */
export interface PartProps {
  node: Node;
  engine: EngineId;
  tier: Tier;
  /** The parameters its node edits (`param`, `params`), and nothing else. */
  params: Readonly<Record<string, BoundParam>>;
  /** The state its node names. */
  state: Readonly<Record<string, BoundState>>;
  /** What its node reads, with no setter. */
  reads: Readonly<Record<string, ReadValue>>;
  /** Runs `fn`, whose sets are one undo step. */
  batch(fn: () => void): void;
  /** Its node's actions, ready to run. */
  run(action: number): void;
  /** Its children, drawn. */
  children: readonly unknown[];
}

/* ---------------------------------------------------------------- the other lists */

/** Units a number is shown in. Adding one here is all a new unit needs. */
export const UNITS = {
  s: 'seconds; reads as m:ss from a minute (docs/05, "How a number is written")',
  ms: 'milliseconds',
  Hz: 'hertz',
  dB: 'decibels',
  dBFS: 'decibels full scale; on a pedal knob, on its own line under the number',
  LUFS: 'loudness units; on a pedal knob, on its own line under the number',
  '%': 'percent, written right after the number',
} as const;
export type UnitName = keyof typeof UNITS;

/** The text checks a parameter may name: what each accepts, in words and as the test the page, the cleaner and the
 *  validator all run. */
const seedText = (s: string) => /^\d{1,10}$/.test(s.trim()) && Number(s.trim()) <= 4294967295;
export const CHECKS = {
  seed: { doc: 'A whole number from 0 to 4294967295.', test: seedText },
  seedOrMinusOne: { doc: 'A whole number from 0 to 4294967295, or -1.', test: (s: string) => s.trim() === '-1' || seedText(s) },
  seedOrEmpty: { doc: 'A whole number from 0 to 4294967295, or empty.', test: (s: string) => s.trim() === '' || seedText(s) },
  timesteps: {
    doc: 'Numbers from 0 to 1, comma-separated, at least two, each smaller than the last.',
    test: (s: string) => { const v = s.split(',').map(x => x.trim()).filter(Boolean).map(Number); return v.length >= 2 && v.every((x, i) => Number.isFinite(x) && x >= 0 && x <= 1 && (i === 0 || x < v[i - 1]!)); },
  },
} as const satisfies Record<string, { doc: string; test(text: string): boolean }>;
export type CheckName = keyof typeof CHECKS;

/** Where a server-listed choice's options come from, and which fixture list the validator checks its values
 *  against (tools/fixtures/real-app.json). */
export const OPTION_SOURCES = {
  'plugins.solver': { doc: 'GET /api/plugins solvers', fixture: 'plugins.solver' },
  'plugins.scheduler': { doc: 'GET /api/plugins schedulers', fixture: 'plugins.scheduler' },
  'plugins.guidance': { doc: 'GET /api/plugins guidance modes', fixture: 'plugins.guidance' },
  'plugins.postprocess': { doc: 'GET /api/plugins post-process plugins', fixture: 'plugins.postprocess' },
  models: { doc: 'GET /api/backends/models, one bucket', fixture: null },
  languages: { doc: 'the vocal languages the app knows', fixture: null },
  'adapters.lm': { doc: 'the planner LM adapters installed', fixture: null },
  'mm3.planks': { doc: 'MiniMax-Music3\'s saved Planks', fixture: null },
  'mm3.hiddens': { doc: 'MiniMax-Music3\'s saved full plans', fixture: null },
  'caption.sources': { doc: 'the caption source\'s tracks', fixture: null },
  'caption.datasets': { doc: 'the caption datasets', fixture: null },
} as const;
export type OptionSourceName = keyof typeof OPTION_SOURCES;

/** What a desk fact is worked out from: the values and the state in force (a definition off the desk reads as its
 *  defaults and holds no state: src/derive/core.ts, "the value in force"), and the engine. */
export interface FactInput {
  value(key: string): Value | undefined;
  state: { readonly [key: string]: Json | undefined };
  engine: EngineId;
}
const lines = (t: unknown) => String(t ?? '').split('\n');
const DIRECTIVE = /^\s*\[[^\]]+\]\s*\{[^}]*#\d+\s*=/;
const slotsOf = (f: FactInput) => (Array.isArray(f.state.slots) ? f.state.slots : []) as { ph?: number[] }[];
const stackOn = (f: FactInput) => f.engine === 'ace' && f.value('ad.adv') === 'advanced' && slotsOf(f).length >= 2;
const directives = (f: FactInput) => stackOn(f) && lines(f.value('lyrics.text')).some(l => DIRECTIVE.test(l));

/** Desk facts a condition may name besides the definitions' own state keys: facts the page works out, each with the
 *  rule it is worked out by (`of`), or none when the page is told it (the plugin host, another engine's stash). */
export const DESK_FACTS = {
  adapters: { doc: 'How many adapters are loaded on this engine (atLeast).', of: (f: FactInput) => slotsOf(f).length },
  adaptersAway: { doc: 'The other engines that keep adapters for when you switch back, as words (" · ACE-Step 1.5 keeps its adapter for when you switch back"), or nothing. From the engines\' stashes.' },
  runtimeForced: {
    doc: 'Adapters are forced to runtime mode: section directives or a phase window, with a stack of two or more.',
    of: (f: FactInput) => directives(f) || (stackOn(f) && slotsOf(f).some(s => !!s.ph && (s.ph[0]! > 0 || s.ph[1]! < 100))),
  },
  sectionDirectives: { doc: 'The lyrics carry per-section adapter directives, with a stack of two or more.', of: directives },
  sectionTags: { doc: 'How many section tags ([Verse], [Chorus]) the lyrics hold (atLeast).', of: (f: FactInput) => lines(f.value('lyrics.text')).filter(l => /^\s*\[[^\]]+\]/.test(l)).length },
  customTimesteps: {
    doc: 'Latent & correction holds two or more valid custom timesteps, which replace Steps and Schedule.',
    of: (f: FactInput) => CHECKS.timesteps.test(String(f.value('lat.ts') ?? '')),
  },
  vstPlugins: { doc: 'How many VST3 plugins are in the chain (atLeast). Told by the plugin host.' },
} as const satisfies Record<string, { doc: string; of?: (f: FactInput) => number | boolean }>;
export type DeskFactName = keyof typeof DESK_FACTS;

/** Named actions a node may run that are not changes of a value: each opens something or talks to the server. */
export const ACTIONS = {
  vstHost: 'Opens the VST3 host to add a plugin to the chain.',
  loadAdapter: 'Opens the adapter list to load one into the next slot.',
  loadPlannerAdapter: 'Opens the planner LM adapter list (ACE-Step).',
  composeCaption: 'Runs Compose caption, which writes BPM, key and vocal details into the caption.',
  scorePreview: 'Opens the score preview for the lead sheet, or plans one first.',
} as const;
export type ActionName = keyof typeof ACTIONS;

/** The containers every definition gets without naming them: drawn by the shell from the definition itself. */
export const SHELLS = {
  tile: { css: ['20-tile.css', '21-desk.css'], doc: 'The tile: header with grip, group and name, its ×, the body, the "Also set" foot, the unsupported note, and the one-line strip a tile becomes when it has nothing to show.' },
  pedal: { css: ['27-pedals.css', '27-pedalboard.css'], doc: 'The pedal: face in its hue (the neutral face when it has no switch), name, LED, foot switch (or "Always on"), remove tab, jacks and cables; 1U or 2U at Lab (2U from three numbers), one size at Studio (132 x 196) and at Quick (112 wide). Its Lab body is laid out from its parameters by layOut below.' },
  sideUnit: { css: ['28-side-units.css'], doc: 'The side unit: a rack strip with a name column, its object, its controls at Lab (laid out by layOut below), a power button and a remove button.' },
  board: { css: ['27-pedalboard.css'], doc: 'The Pedalboard tile: chain switch, count, the board with cables, the side units row.' },
} as const;

/** The part a pedal's or a side unit's parameter is laid out with at Lab (docs/01, "How a pedal and a side unit are
 *  laid out"). The shell and the validator both call this; it is the only statement of the rule. */
export function layOut(kind: 'pedal' | 'unit', type: ValueType, optionCount: number | null): PartName {
  if (type === 'number') return 'knob';
  if (type === 'boolean') return 'ledButton';
  if (type === 'text') return 'textField';
  return kind === 'unit' && optionCount !== null && optionCount <= 4 ? 'chips' : 'pick';
}

/** A pedal is 2U at Lab, and in its thumbnail, from three numbers. */
export const pedalUnits = (numbers: number): 1 | 2 => (numbers >= 3 ? 2 : 1);

