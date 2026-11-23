# The parts inventory

A definition builds a tile out of named parts. This page lists every part, what each one is for, what it edits, reads
and takes, and where its look lives. The list a definition may name is `src/definitions/parts.ts`; this page is that
file with the reasons beside it. The table at the end is generated from it by `npm run docs`, and the validator fails
when the table is out of date.

The concept is `C:\Users\sneth\misc\HOT-Step-UI`. Paths below like `15-knob.css` are in its `src/css`, and `21-knob.js`
in its `src/js`.

## Four kinds of part

- **Controls** edit a parameter. A control never decides its own range, default, label, tip, changed marker, undo name
  or how its value is written: all of that comes from the parameter it is bound to
  ([05](05-what-the-definition-drives.md)). What a control owns is how it looks and how a hand or a key moves it. Every
  number control keeps the concept's one value contract (`20-controls.js`): arrows step, Shift steps ten, Page Up and
  Page Down step ten, Home and End go to the ends, double-click resets (auto included), and a typed value out of range
  is clamped and said once ("Steps: 1 to 64. Set to 64.").
- **Containers** hold other nodes and give them their housing: a rack strip, a recessed well, an LCD, a terminal, a rack
  group with its power switch, a tile foot. A container none of whose children is shown on the engine and tier is not
  drawn at all.
- **Read-outs** show something and edit nothing: a note, a value on an LCD, a meter, the plan text.
- **Bespoke parts** are written for one module because what they draw or do cannot be said as data: the key wheel, the
  split-flap seed, the Sampler's scope, the adapter cartridges. Each edits only what its node binds, and its row says
  exactly what it is responsible for.

## What a node binds

A node names its part and binds it with four fields. Which of them a part takes is in its row:

| Field | Means | Which parts take it |
| --- | --- | --- |
| `param`, `params` | The parameters it **edits**. These are what "Also set", the changed marker's tests and the check that every parameter has a control read | A part with `edits`: the value types it can edit (`pair` is two numbers, an XY pad) |
| `reads` | What it **shows** without editing: a read-out's value, a bespoke drawing's inputs | A part with `reads` |
| `state` | The module state it edits: a drop zone's file, the seed, the adapter slots | A part with `state`: the kinds it holds |

A control bound to a parameter the engine in use does not have is not drawn there; a node whose bindings no engine it
is shown on has is refused.

## What a bespoke part is given

A part's component receives `PartProps` (`parts.ts`) and nothing else:

- `params`: the parameters its node edits, each with its face on the engine in use, its value and its words, and a `set`
  that cleans the value and makes the undo step named from the parameter. There is no setter for anything else, so a
  part cannot write a value it is not bound to.
- `state`: the state its node names, each with `apply(step, next, also?)`, where `step` is the name of one of the state's
  steps in its definition (`steps` in [02](02-definition-format.md), "State"). The words of the undo step come from the
  definition. `also` changes some of the node's own parameters in the same step (Roll once sets the seed and its lock).
- `reads`: what its node reads, with values and words and no setter.
- the node itself, resolved for the engine and tier, its actions ready to run, and its children drawn.

So a bespoke part cannot keep its own copy of a value, decide a default, name an undo step, or write state the definition
does not declare: none of these is in what it is given. The validator checks what can be checked without running it:
that the node's `params` are of the types the part edits, its `state` of the kinds it holds, and its state its own
definition's (a drop zone may show another's file with `opts.from`, read-only).

## What every part gets without asking

Three systems of the concept apply to every part and are not repeated in each row:

- the changed marker: an amber dot 4 px after the control's words, or after its box when it has none (`35-changed.css`);
- focus, hover and press feedback: one 2 px ring, pink, or green on an LCD or terminal (`35-states.css`);
- the one caps label style (`35-labels.css`).

A node can also stand back: `dim` draws it at 45% with its reason beside it, still operable (`20-tile.css`, the concept's
"Off is one state" rule).

## Shells: drawn from the definition, never named in a face

| Shell | What it is | Styles |
| --- | --- | --- |
| Tile | Header with grip, group and name, its ×, the body, the "Also set" foot line, the unsupported note, and the one-line strip a tile becomes when it has nothing to show. Its width class comes from `size`. | `20-tile.css`, `21-desk.css` |
| Pedal | Face in its hue (`hue`), or the neutral face when it has no switch (`on: null`); name, LED, foot switch (or "Always on"), remove tab, jacks and cables. 1U or 2U at Lab (2U from three numbers: 1U = 164 px, 2U = 368 px), one size at Studio (132 x 196), one at Quick (112 wide). Its Lab body is laid out from its parameters (below). | `27-pedals.css`, `27-pedalboard.css` |
| Side unit | A rack strip with a name column, its object (the definition's `face`), its controls at Lab, a power button and a remove button; `wide` takes the full row. | `28-side-units.css` |
| Pedalboard | The chain switch and count, the board with cables, the side-units row. The Pedalboard module's only face node is its chain switch; the rest is this shell. | `27-pedalboard.css` |
| Thumbnail frame | The 176 x 112 module canvas, the pedal body (1U 80 x 110, 2U 184 x 110, the same three-number rule) and the side-unit strip, into which a definition's thumbnail file is drawn. | `30-thumbnails.css`, `24-thumbnails.js` |

### How a pedal and a side unit are laid out from their parameters

The concept builds every pedal's body in one function (`pedalHTML`, `27-pedalboard.js`) and every unit's in another
(`paintBoard`). The rebuild keeps that, so a pedal definition has no Lab face of its own beyond what is special. The part
each parameter gets is `layOut()` in `parts.ts`, which the shell and the validator both call:

1. Numbers become small knobs in 64 px cells, five to a row, then wrapping into two rows. A unit in dBFS or LUFS is
   written on its own line under the number. A number whose `when` can hide it keeps its cell, drawn as a blanking
   plate, so switching never moves the other knobs. The plate's words are worked out from the hidden knobs' conditions:
   for knobs hidden by `{ param, is }`, "with <that parameter's label>: <the option labels that show them, joined by
   or>" ("with Source blend: Crossover or Mix").
2. Choices become LCD pickers with previous and next arrows; text becomes an LCD input.
3. Then the definition's own Lab nodes (`face` nodes tiered `lab`): the meter, the reference slot.
4. Then switches as LED buttons, each followed by what stands `beside` it, its room kept while hidden.
5. At Studio a pedal shows its `main` knob, or its Studio nodes when it has none. At Quick it shows its name and switch.

A side unit shows its `face` at every tier, then at Lab its parameters: numbers as knobs, a choice of up to four options
as chips, longer choices as LCD pickers, switches as LED buttons, text as LCD inputs.

The validator counts these laid-out parts as used, marked "(laid out)" in the table.

## The other lists in parts.ts

Each list is the only place its entries are stated; the format's types are derived from it, so an entry that is not
there fails the type check where it is typed, and adding one is one line.

| List | What it holds |
| --- | --- |
| `UNITS` | The units a number may be shown in, and how each is written |
| `CHECKS` | The text checks a parameter may name, with what each accepts (`seed`, `seedOrMinusOne`, `seedOrEmpty`, `timesteps`) |
| `OPTION_SOURCES` | Where a server-listed choice's options come from, and which fixture list the validator checks its values against |
| `DESK_FACTS` | Facts the page works out that a condition or words may name: `adapters`, `adaptersAway`, `runtimeForced`, `sectionDirectives`, `sectionTags`, `customTimesteps`, `vstPlugins` |
| `ACTIONS` | Named actions that open something or talk to the server (`vstHost`, `loadAdapter`, `loadPlannerAdapter`, `composeCaption`, `scorePreview`) |

The transforms a wire may name are in `transforms.ts` ([05](05-what-the-definition-drives.md), "What is sent").

## What the concept has that is not a part

- The concept's plain switch (`.tog`, drawn by `ctlHTML` in `20-controls.js`) is used by no face: every switch on the
  desk is an LED button, a pill, a flag, an LCD word, a record button, a power button or a padlock.
- The concept's generic `rackCtl` and `ctlHTML` renderers pick a control from the value type. In the rebuild that choice
  is written in the face, or made by `layOut()`; there is no "draw whatever fits" part.
- Overlays (dropdowns, the picker wall, the score modal, the engine panel's lists) are opened by parts but are not parts:
  the `pick` part opens the shared dropdown (`30-dropdown.css`), the `scorePreview` action opens the score modal.

## The inventory

The "Used by" column is filled in from the definitions.

<!-- generated: inventory (npm run docs) -->
| Part | Kind | Edits | State | Reads | What it is for | Options | States | Styles (concept src/css) | Used by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `knob` | control | number |  |  | A rotary knob with a detent tick at the default, its label under it and a read-out that can be clicked to type a value. The rack face of every number in Lab. | `small` (boolean): The small knob (rack rows). | turning; resetting; auto (drawn at the auto value in the auto colour, with an auto pill); typing | 15-knob.css, 27-pedals.css | repaint, sampler, guidance, perf, latent, flow, replan, adhere, yuebatch, refiner (laid out), ss (laid out), ppvae (laid out), lift (laid out), nat (laid out), gain (laid out), norm (laid out), wh (laid out) |
| `fader` | control | number |  |  | A vertical fader with a cap, tick marks and a unity line at 1.00. Used in fader banks (adapter group scales, depth thirds) and cartridges. | `travel` (number): The value the top of the slot stands for, when the slot runs past the maximum (Plan: max 1, travel 2). | sliding; stops (travel beyond max shown as a stop) | 15-fader.css, 25-module-adapters.css | adapters |
| `ruler` | control | number |  |  | A length on a minutes timeline, dragged along; labels on round steps. The length of a song. | `scaleMax` (number): The seconds the timeline spans. Default: the parameter maximum.<br>`markers` (boolean): Draw Beat I/O and trim markers from the Tempo parameters. Never drawn at Quick. | auto (shows the estimate, or "up to" where the model ends the song); sliding; at-start/at-end caption | 15-ruler.css | tempo |
| `xyPad` | control | pair |  |  | Two settings that belong together on one plane: a puck with a trail, a grid, axis read-outs that each carry their own changed marker, corner words outside the pad. An axis whose parameter is absent on the engine leaves a one-axis slider. | `xl` (text): The X axis words.<br>`yl` (text): The Y axis words.<br>`corners` (texts): Four corner words: top left, top right, bottom left, bottom right.<br>`yOff` (text): Words on the rail when the Y minimum means off.<br>`look` (apg \\| heat \\| overlay \\| term): The plot colouring. | grab; release; nudge; y-off (the Y minimum drawn as an off rail) | 15-xy-pad.css | planner, guidance, perf, latent |
| `numberBox` | control | number |  |  | A typed number box that can also be dragged up and down. Steps in the Sampler well. | `seconds` (boolean): Show an estimate of seconds under it (Steps: "≈ 1.3 s"). | editing; overridden (a value another module replaces); empty means auto | 15-numbox.css | sampler |
| `lcdNumber` | control | number |  |  | A big seven-segment number on an LCD that can be typed or dragged, with its unit beside it, and AUTO beside it when the parameter has auto. BPM. |  |  | 25-module-tempo.css, 15-lcd.css | tempo |
| `pick` | control | choice |  |  | The button that shows a choice and opens its dropdown. The variant sets where it lives. | `variant` (plugin \\| lcd \\| chip \\| model \\| stamp): plugin: a periodic-table tile and the kind above the value (solver, schedule, guidance); lcd: an LCD with previous and next arrows; chip: a pill with its label; model: name, quant badge, size and a memory bar; stamp: an ink stamp (the lyric language). | open; overridden | 15-pick.css, 30-dropdown.css | tempo, adapters, sampler, plans, tile (laid out), ss (laid out), norm (laid out), wh (laid out) |
| `chips` | control | choice |  |  | A chip selector: up to four options as lit chips under a caps label. A side unit's short choice at Lab. |  |  | 15-controls.css | q (laid out) |
| `segmented` | control | choice |  |  | A segmented choice: a row of joined buttons, one pressed. | `size` (sm): sm: the compact row inside a rack group. |  | 15-controls.css | lyrics, repaint, adapters, latent, score, build, flow, adhere |
| `termChips` | control | choice |  |  | A choice written as terminal chips after its flag (--mode Presence Frequency DRY). |  |  | 25-module-planner.css | planner, repetition |
| `lcdChips` | control | choice |  |  | Numbered chips on an LCD with a word after them (1 2 4 8 BARS). | `word` (text): The word printed after the chips. |  | 25-module-tempo.css | tempo |
| `ledButton` | control | boolean |  |  | An LED switch: a rack button with a lamp that lights when on. Lab rack face of a boolean. |  |  | 15-controls.css | adapters, flow, plans, replan, yuebatch, ss (laid out), master (laid out), wh (laid out) |
| `pill` | control | boolean |  |  | A pill toggle with a lamp, used inside a module look (Instrumental, Level match, a rack group power switch). |  |  | 15-controls.css | lyrics, planner, sampler, latent, adhere, seed |
| `flag` | control | boolean |  |  | A terminal flag with a check box: [x] --cot. |  |  | 25-module-planner.css | planner |
| `lcdButton` | control | boolean |  |  | A lit word on an LCD that switches (BEAT I/O, AUTO-TRIM). |  |  | 25-module-tempo.css, 15-lcd.css | tempo |
| `recButton` | control | boolean |  |  | A record button (● SAVE NEXT) that arms the next run to save something. |  | is-off (stands back: nothing new to save) | 25-module-plans.css | plans |
| `powerButton` | control | boolean |  |  | A terminal power switch with its word (● planner) and the scope switch (● Lua plugins): the switch a whole instrument hangs on. | `look` (term \\| scope): term: the terminal bar; scope: on a scope read-out. |  | 25-module-planner.css | planner, sampler, post |
| `lockButton` | control | boolean |  |  | A padlock button that keeps a value (Keep this seed). |  |  | 15-controls.css, 25-module-seed.css | seed |
| `powerChip` | control | number |  |  | A rack group power chip for a group switched by a number: off is one value, on sets a moderate one. Says "· off" on itself when off. | `off` (number): The value that means off.<br>`on` (number): The value pressing it on sets. |  | 15-controls.css, 25-module-adhere.css | adhere |
| `token` | control | number |  |  | A scrub token: a number in running text or a terminal line, dragged sideways, stepped with the arrows, typed on Enter. In a terminal line its label is its flag. |  | clash (another setting makes it skip) | 15-numbox.css, 25-module-planner.css | tempo, adapters, planner, repetition, sampler, build, adhere |
| `textField` | control | text |  |  | A one-line text field. Checked fields show their error beside them and commit on Enter or leaving. | `look` (plain \\| lcd \\| term): plain; lcd: an LCD input; term: an inline terminal input.<br>`size` (number): Width in characters. | bad (fails its check); kept (invalid text kept, what is in force said) | 15-controls.css, 15-lcd.css | adapters, planner, sampler, latent, plans, ss (laid out), art (laid out) |
| `textArea` | control | text |  |  | A text area. The look gives it its material. | `look` (plain \\| note \\| staff \\| paper \\| term): note: the taped sticky note (Avoid); staff: the ruled ABC sheet (Lead sheet); paper: the lyric sheet; term: a quoted one-line terminal string that wraps (--neg). |  | 15-controls.css | repaint, avoid, planner, score |
| `choiceRow` | control | number, choice |  |  | Quick's plain choice row: a radio group of words that each set a detailed value, or several. A value between the choices checks none: a marker stands at its place and the label says the value. | `look` (plain \\| mood): mood: the coloured How adventurous row. |  | 15-controls.css | adapters, planner, sampler |
| `lookback` | control | number |  |  | A look-back window drawn as a strip of codes that grows leftward into the past, with a time scale under it. | `unit` (text): The word for what it counts (codes, frames, tokens).<br>`perSecond` (number): How many a second, for the scale and the tip.<br>`who` (text): Who checks for repeats, for the tip (the planner). |  | 25-module-repetition.css | repetition |
| `dropZone` | control |  | file |  | A file drop zone: a record sleeve (modules) or a slot (pedals) that takes a dropped or chosen audio file, shows its name cut to fit, and clears with ×. Its node's `state` is the file it holds. | `look` (sleeve \\| slot): A record sleeve or a pedal slot.<br>`empty` (text): Its words while empty.<br>`hint` (text): Its second line while empty.<br>`from` (text): Shows a file another part lends it (its node's `state` is that part's file): the name, then these words, and it takes no file of its own. |  | 15-controls.css, 25-module-timbre.css, 27-pedals.css | timbre, cont, master |
| `row` | container |  |  |  | A row of knobs or controls with even spacing (the .knobs row). |  |  | 15-knob.css | repaint, sampler, guidance, latent, flow, replan, adhere, yuebatch |
| `rack` | container |  |  |  | A rack row: LED switches and chip selectors side by side on a rack strip. |  |  | 15-controls.css | flow, plans, replan, yuebatch |
| `well` | container |  |  |  | A recessed well that holds a module's main controls under its instrument (the Sampler strip). |  |  | 25-module-sampler.css | sampler |
| `lcd` | container |  |  |  | An LCD panel. Its children are rows (lcdRow) or LCD parts. |  |  | 15-lcd.css, 25-module-tempo.css | tempo |
| `lcdRow` | container |  |  |  | One row of an LCD. Ends-aligned when it holds switches at both ends. | `ends` (boolean): Spread to both ends. |  | 25-module-tempo.css | tempo |
| `terminal` | container |  |  |  | A terminal: dark, monospaced, a command bar on top, lines below. The planner and the repetition guard. | `title` (text): The tip on the whole terminal. |  | 25-module-planner.css | planner, repetition |
| `termBar` | container |  |  |  | The terminal's top bar: its power switch and flags. |  |  | 25-module-planner.css | planner |
| `termLine` | container |  |  |  | A command line: "$ <command>" then its children, each control after its label printed as a flag (--temp). Wrapped lines hang under the first flag. | `command` (text): The command word after $. |  | 25-module-planner.css | planner, repetition |
| `group` | container |  |  |  | A rack group: a boxed set of controls with a header that may carry its power switch; a group switched off stands back. | `title` (text): Its caps heading when it has no power switch. |  | 25-module-adhere.css, 25-module-latent.css | sampler, latent, adhere |
| `foot` | container |  |  |  | A tile's foot row: small labelled controls and a note, kept to the bottom of the tile. |  |  | 20-tile.css | lyrics, adapters, sampler, build, flow |
| `labelled` | container |  |  |  | A caps label beside one small control (the foot-ctl and lcdsel pairing). |  |  | 15-controls.css | lyrics, tempo, adapters, sampler, latent, score, build, flow, plans, replan, adhere, vst |
| `deck` | container |  |  |  | A two-column grid of labels and controls, so a choice never moves the rows under it (Plans & endings). |  |  | 25-module-plans.css | plans |
| `pair` | container |  |  |  | Two tokens as a range: from → to, under one label. |  |  | 25-module-adhere.css | adhere |
| `bank` | container |  |  |  | A fader bank with a 2-1-0 scale and a heading, each fader labelled on tape. | `title` (text): Its heading.<br>`note` (text): Its heading's tip. |  | 15-fader.css, 25-module-adapters.css | adapters |
| `note` | readout |  |  |  | A quiet line of words: a hint, why something is off, what another module changes. Its actions follow as links. | `look` (plain \\| warn \\| cap \\| hint): warn; cap: the caption note under an instrument; hint: Quick's hint. |  | 01-base.css, 15-controls.css | repaint, tempo, planner, sampler, guidance, score, cont, flow, master |
| `link` | readout |  |  |  | A text link that runs an action (Insert example, Show in Lab, Turn on Lua plugins). |  |  | 14-buttons.css | latent |
| `button` | readout |  |  |  | A small button that runs an action (+ Load adapter, Roll once). |  |  | 14-buttons.css | planner, vst |
| `lcdValue` | readout |  |  | yes | A value printed on an LCD under a caps word (KEY D MAJOR): the first of `reads` written as everywhere else, or its `text`. |  |  | 25-module-tempo.css | tempo, tile, vst, master |
| `unitLcd` | readout |  |  | yes | A side unit's two-line LCD: each of `reads` on its own line, a choice as its option's LCD words, or its `text` as one line. | `prefix` (text): Words before the first line (WHISPER ).<br>`notAtLab` (boolean): Not drawn at Lab, where the unit's own controls show the same values. |  | 28-side-units.css | vst, wh |
| `unitStatus` | readout |  |  |  | A side unit's status words in small caps (After the render · needs MMS). |  |  | 28-side-units.css | fa |
| `unitBadge` | readout |  |  |  | A side unit's round score badge. |  |  | 28-side-units.css | q |
| `dice` | readout |  |  | yes | A row of numbered dice, one per allowed attempt (the first of `reads`), filled as attempts are used. | `offWhen` (param): A switch that, while off, draws the dice off. |  | 25-module-replan.css | replan |
| `meter` | readout |  |  | yes | A printed meter with a needle: the first of `reads` on its scale, or a level the engine reports when it reads nothing (the Mastering pedal's VU meter). | `min` (number): The scale's left end.<br>`max` (number): Its right end.<br>`red` (number): Where the scale turns red. |  | 27-pedals.css | master |
| `beatPulse` | readout |  |  | yes | Lamps that pulse on the beat at the BPM it reads, one per beat of the bar. |  |  | 25-module-tempo.css | tempo |
| `flowWindows` | readout |  |  | yes | MiniMax-Music3's flow windows drawn at the hop it reads, and the count of windows a 3:00 song needs. |  |  | 25-module-flow.css | flow |
| `speedGauge` | readout |  |  | yes | An LCD that estimates how much faster the trades it reads make a run. |  |  | 25-module-perf.css | perf |
| `adhereCurve` | readout |  |  | yes | Plan strength across the run, from noise to clean, with the Plan CFG band and the CADS ramp, from the values it reads. |  |  | 25-module-adhere.css | adhere |
| `planText` | readout |  |  | yes | The plan the planner will start from, as terminal text: caption, tempo, key, form, and whose each value is. On MiniMax-Music3 its first line also says whether a saved plan is replayed or the plan cache is on. |  |  | 25-module-planner.css | planner |
| `lrcPreview` | readout |  |  | yes | Two timed lyric lines, the first named after the first section of the lyrics it reads. |  |  | 28-side-units.css | lrc |
| `coverSwatch` | readout |  |  |  | A swatch in the newest song's colour, standing for its cover art. |  |  | 28-side-units.css | art |
| `samplerScope` | bespoke | number |  | yes | The Sampler's scope: the noise schedule drawn from the values it reads, the guidance band, and at Lab the CFG interval's two handles. **Responsible for:** Drawing sigma(t) from its reads (steps, shift, schedule and the shape parameters, or the custom timesteps when Latent & correction sets them); the guidance band from the CFG interval and the hatched CFG cutoff; the two interval handles, which set its two params a step apart; the one-line read-out. Defaults, undo, changed markers and saving of every value are generic. | `interval` (boolean): Draw the CFG interval's two handles; they edit the node's two params. |  | 25-module-sampler.css | sampler |
| `keyWheel` | bespoke | choice |  |  | The circle of fifths: majors outside, minors inside, Auto in the middle, a radio group driven by the arrows. **Responsible for:** Drawing the two rings from its param's options, spelling keys as a musician does (B♭, F♯), the arrow-key walk around and between rings, and the selection nudge. |  |  | 25-module-tempo.css | tempo |
| `tapTempo` | bespoke | number |  |  | TAP: two taps or more within 2.5 s set the BPM from their spacing. **Responsible for:** Timing the taps and setting its param (clamped to its range); the run of taps is one undo step through the parameter's `run`. |  |  | 25-module-tempo.css | tempo |
| `capChips` | bespoke | text |  | yes | On engines where tempo and key go in the prompt: chips that add or take out "120 BPM" or "D major" as items of the prompt text it edits, two exclusive groups. **Responsible for:** Reading which chips are pressed from the text, replacing an item of the same group, never touching words inside a sentence, and offering the tempo and key it reads first. Each press is one step of the text it edits. |  |  | 25-module-tempo.css | tempo |
| `takesReels` | bespoke | number |  |  | The takes count as an LCD and a set of reels, one per take the engine allows; the reels are one slider. Under it, the VRAM the count costs. **Responsible for:** Drawing as many reels as its param's maximum, the hover preview, the reel spin, and the VRAM cost line from the loaded models. The LCD reads the value as its format writes it, then the node's label. |  |  | 25-module-takes.css | takes |
| `splitFlap` | bespoke | boolean | seed |  | The seed as ten split-flap cells that flip to a new number; pressing it types one. Roll once and Random each time. **Responsible for:** The flap animation, the ten-cell layout, typing or pasting a seed (checked 0 to 4294967295), Roll once and Random each time. Each sets the seed and its lock (its param) in one step, the seed state's `set` step. |  |  | 25-module-seed.css | seed |
| `savedSeeds` | bespoke | boolean | list, seed |  | The saved seeds list: search, star, name, describe, tag, load, delete, random from saved. **Responsible for:** The list (its `list` state) and its search; loading a saved seed sets the seed and its lock in the seed state's `set` step; every list action is the list state's step of that name. |  |  | 25-module-seed.css | seed |
| `timbreBlend` | bespoke | number |  |  | Two overlapping circles, the reference dragged into your sound: the reference blend. **Responsible for:** Drawing the circles at its param and dragging sideways. |  |  | 25-module-timbre.css | timbre |
| `recentRefs` | bespoke |  | list, file |  | The two latest references as chips, and the rest in a popover. **Responsible for:** Showing its `list` state and loading one into its `file` state (that state's `load` step). |  |  | 25-module-timbre.css | timbre |
| `latentCurve` | bespoke |  |  | yes | The latent distribution before (grey) and after (green) the shift and rescale it reads, with the XY pad that sets them laid over it as its child. **Responsible for:** Drawing the two curves and hosting its child. It edits nothing; the pad is the generic xyPad part. |  |  | 25-module-latent.css | latent |
| `abcFoot` | bespoke |  |  | yes | The lead sheet's foot: whether the text it reads is ABC, how many lines, and its actions as they apply (Insert example, Preview this score or Plan and preview a score, the way back when the planner writes nothing). **Responsible for:** Recognising ABC (a K: line and bars) and counting lines, and choosing which of its node's actions apply. It edits nothing itself. |  |  | 25-module-score.css | score |
| `cartridges` | bespoke | number, choice, text | slots |  | Adapter cartridges: each loaded adapter as a cassette with label art, its strength fader, its kind's housing, and in a stack its effective share, active phase and per-section mask. **Responsible for:** Showing its slots state and adding and removing cartridges (the state's `load` and `remove` steps), the label art, and drawing its params with the fader part. Phase windows and section masks are written into the lyrics text it edits. | `flat` (boolean): Quick's flat cartridge: the first adapter lying flat, its art beside its name. |  | 25-module-adapters.css | adapters |
| `buildBoard` | bespoke |  | record | yes | Build by section: the planned sections as columns, takes under each, keep one, render the song. **Responsible for:** Showing its record state (planned score, takes per section, keeps, the render) and its drift against the lyrics it reads, and talking to the build session API. Every change is the record state's step of that name. |  |  | 25-module-build.css | build |
| `recordingSpan` | bespoke |  | span | yes | A recording's waveform with up to 30 seconds selected: drag inside to move, an edge to resize, outside to draw a new one. **Responsible for:** Its span state and its keyboard contract (arrows move, Shift resizes, Page moves by its length), and the ruler under it. A move is the span's step. |  |  | 25-module-cont.css | cont |
| `repaintRegion` | bespoke |  | record, span |  | The source song's waveform with the region to regenerate. **Responsible for:** Its source record and region span, drawing the waveform, and the region's handles and keys. A move is the span's step, in the song's times. |  |  | 25-module-repaint.css | repaint |
| `lyricSheet` | bespoke | text, choice, boolean |  | yes | The ruled lyric paper with its section tabs, the language stamp, the structure strip under it and the line count. **Responsible for:** Inserting section tags at line boundaries, the structure strip (sections as coloured segments that jump to their line), the line count and its time estimate, and wildcard expansion. Every value it edits is one of its params. |  |  | 25-module-lyrics.css | lyrics |
| `promptBox` | bespoke | text, choice, boolean |  |  | The main text box (description, caption, prompt or tags) with its tools and the song info. **Responsible for:** The box, its rotating placeholder, Write it for me and Compose caption, and the caption source. Every value it edits is one of its params. |  |  | 22-prompt.css | prompt |
| `modelPicker` | bespoke | choice |  |  | The engine panel's model rows, with the VRAM meter they add up to. **Responsible for:** Listing installed models from the server and the VRAM figures. Each model choice is one of its params; its undo name is the parameter's. |  |  | 15-pick.css, 30-engine-panel.css | models |
<!-- end generated -->
