# The parts inventory

A definition builds a tile out of named parts. This page lists every part the concept has, what each one is for, what
it takes, and where its look lives. The list a definition may name is `src/definitions/parts.ts`; this page is that file
with the reasons beside it. The table at the end is printed from it:

```
node tools/validate-definitions.ts --inventory
```

so after adding a part, print it again and paste it here rather than editing rows by hand.

The concept is `C:\Users\sneth\misc\HOT-Step-UI` at the tag `vanilla-reference`. Paths below like `15-knob.css` are in
its `src/css`, and `21-knob.js` in its `src/js`.

## Four kinds of part

- **Controls** edit a parameter. A control never decides its own range, default, label, tip, changed marker, undo name
  or how its value is written: all of that comes from the parameter it is bound to (see
  [05-what-the-definition-drives.md](05-what-the-definition-drives.md)). What a control owns is how it looks and how a
  hand or a key moves it. Every number control keeps the concept's one value contract (`20-controls.js`): arrows step,
  Shift steps ten, Page Up and Page Down step ten, Home and End go to the ends, double-click resets (auto included), and
  a typed value out of range is clamped and said once ("Steps: 1 to 64. Set to 64.").
- **Containers** hold other nodes and give them their housing: a rack strip, a recessed well, an LCD, a terminal, a rack
  group with its power switch, a tile foot.
- **Read-outs** show something and edit nothing: a note, a value on an LCD, the VU meter, the plan text.
- **Bespoke parts** are written for one module because what they draw or do cannot be said as data: the key wheel, the
  split-flap seed, the Sampler's scope, the adapter cartridges. Each still binds to parameters the definition declares,
  so its values get defaults, clamping, undo, saving and the changed marker like any other. Its row in the table says
  exactly what it is responsible for, and therefore what it must not do itself.

`binds` says which value types a part can be bound to: `number`, `boolean`, `choice`, `text`, `pair` (two numbers, the
XY pad), `state` (one of the module's StateDef entries, such as a file) or `none`.

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
| Pedal | Face in its hue (`hue`), name, LED, foot switch (or "Always on" when `on` is null), remove tab, jacks and cables. 1U or 2U at Lab (2U from three knobs: 1U = 164 px, 2U = 368 px), one size at Studio (132 x 196), one at Quick (112 wide). Its Lab body is laid out from its parameters (below). | `27-pedals.css`, `27-pedalboard.css` |
| Side unit | A rack strip with a name column, its object (the definition's `face`), its controls at Lab, a power button and a remove button; `wide` takes the full row. | `28-side-units.css` |
| Pedalboard | The chain switch and count, the board with cables, the side-units row. The Pedalboard module's only face node is its chain switch; the rest is this shell. | `27-pedalboard.css` |
| Thumbnail frame | The 176 x 112 module canvas, the pedal body (1U 80 x 110, 2U 184 x 110) and the side-unit strip, into which a definition's thumbnail file is drawn. | `30-thumbnails.css`, `24-thumbnails.js` |

### How a pedal and a side unit are laid out from their parameters

The concept builds every pedal's body in one function (`pedalHTML`, `27-pedalboard.js`) and every unit's in another
(`paintBoard`). The rebuild keeps that, so a pedal definition has no Lab face of its own beyond what is special:

1. Numbers become small knobs in 64 px cells, five to a row, then wrapping into two rows. A pedal with three numbers or
   more is 2U at Lab. A number whose `when` can hide it keeps its cell, drawn as a blanking plate, so switching never
   moves the other knobs; the plate says what fills it (`blank`).
2. Choices become LCD pickers with previous and next arrows; text becomes an LCD input.
3. Then the definition's own Lab nodes (`face` nodes tiered `lab`): the VU meter, the reference slot.
4. Then switches as LED buttons, each followed by what stands `beside` it, its room kept while hidden.
5. At Studio a pedal shows its `main` knob, or its Studio nodes when it has none. At Quick it shows its name and switch.

A side unit shows its `face` at every tier, then at Lab its parameters: numbers as knobs, a choice of up to four options
as chips, longer choices as LCD pickers, switches as LED buttons, text as LCD inputs.

The validator counts these laid-out parts as used, marked "(laid out)" in the table.

## Desk state a condition may name

Besides each definition's own `state` keys (`timbreFile`, `masterRef`, `slots`, `seed`, ...), a condition may name a
few facts the page works out. They are listed in `DESK_STATES` in `parts.ts`:

| State | Meaning |
| --- | --- |
| `adapters` | How many adapters are loaded on this engine (used with `atLeast`). |
| `runtimeForced` | Adapters are forced to runtime mode: timestep windows, or section directives with a stack of two or more. |
| `sectionDirectives` | The lyrics carry per-section adapter directives. |
| `customTimesteps` | Latent & correction is on the desk with two or more valid custom timesteps, which replace Steps and Schedule. |
| `vstPlugins` | How many VST3 plugins are in the chain. |

## Text checks

A text parameter may name one check (`VALIDATORS` in `parts.ts`), with its own message: `seed` (0 to 4294967295),
`seedOrMinusOne`, `seedOrEmpty`, `timesteps` (numbers from 0 to 1, comma-separated, at least two, falling).

## What the concept has that is not a part

- The concept's plain switch (`.tog`, drawn by `ctlHTML` in `20-controls.js`) is used by no face: every switch on the
  desk is an LED button, a pill, a flag, an LCD word, a record button, a power button or a padlock. It is left out,
  following the rule that the inventory holds only parts a definition uses. Add it back the day a module needs it.
- The concept's generic `rackCtl` and `ctlHTML` renderers pick a control from the value type. In the rebuild that choice
  is written in the face, or made by the pedal and side-unit layout above; there is no "draw whatever fits" part.
- Overlays (dropdowns, the picker wall, the score modal, the engine panel's lists) are opened by parts but are not parts:
  the `pick` part opens the shared dropdown (`30-dropdown.css`), the `abcFoot` part opens the score modal.

## The inventory

The "Used by" column is filled in by the validator from the definitions.

| Part | Kind | Binds | What it is for | Options | States | Styles (concept src/css) | Used by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `knob` | control | number | A rotary knob with a detent tick at the default, its label under it and a read-out that can be clicked to type a value. The rack face of every number in Lab. | `small`: The small knob (pedals, rack rows).<br>`unitBelow`: On a pedal, the unit on its own line under the number (dBFS, LUFS). | turning; resetting; auto (drawn at the auto value in the auto colour, with an auto pill); typing | 15-knob.css, 27-pedals.css | repaint, sampler, guidance, perf, latent, flow, replan, adhere, yuebatch, refiner (laid out), ss (laid out), ppvae (laid out), lift (laid out), nat (laid out), gain (laid out), norm (laid out), wh (laid out) |
| `fader` | control | number | A vertical fader with a cap, tick marks and a unity line at 1.00. Used in fader banks (adapter group scales, depth thirds) and cartridges. | `travel`: The value the top of the slot stands for, when the slot runs past the maximum (Plan: max 1, travel 2). | sliding; stops (travel beyond max shown as a stop) | 15-fader.css, 25-module-adapters.css | adapters |
| `ruler` | control | number | A length on a minutes timeline, dragged along; labels on round steps. The length of a song. | `scaleMax`: The seconds the timeline spans. Default: the parameter maximum.<br>`markers`: Draw Beat I/O and trim markers from the Tempo parameters. Never drawn at Quick. | auto (shows the estimate, or "up to" where the model ends the song); sliding; at-start/at-end caption | 15-ruler.css | tempo |
| `xyPad` | control | pair | Two settings that belong together on one plane: a puck with a trail, a grid, axis read-outs that each carry their own changed marker, corner words outside the pad. | `xl`: The X axis words.<br>`yl`: The Y axis words.<br>`corners`: Four corner words: top left, top right, bottom left, bottom right.<br>`yOff`: Words on the rail when the Y minimum means off.<br>`look`: apg \| heat \| overlay \| term: the plot colouring. | grab; release; nudge; y-off (the Y minimum drawn as an off rail) | 15-xy-pad.css | planner, guidance, perf, latent |
| `numberBox` | control | number | A typed number box that can also be dragged up and down. Steps in the Sampler well. | `seconds`: Show an estimate of seconds under it (Steps: "≈ 1.3 s"). | editing; overridden (a value another module replaces); empty means auto | 15-numbox.css | sampler |
| `lcdNumber` | control | number | A big seven-segment number on an LCD that can be typed or dragged, with its unit beside it. BPM. | `tap`: Offer TAP (tap tempo, bespoke behaviour of the Tempo tile) and AUTO beside it. |  | 25-module-tempo.css, 15-lcd.css | tempo |
| `pick` | control | choice | The button that shows a choice and opens its dropdown. The variant sets where it lives. | `variant`: plugin: a periodic-table tile and the kind above the value (solver, schedule, guidance); lcd: an LCD with previous and next arrows; chip: a pill with its label; model: name, quant badge, size and a memory bar; stamp: an ink stamp (the lyric language). | open; overridden | 15-pick.css, 30-dropdown.css | tempo, adapters, sampler, plans, tile (laid out), ss (laid out), norm (laid out), wh (laid out) |
| `chips` | control | choice | A chip selector: up to four options as lit chips under a caps label. Lab rack face of a short choice. |  |  | 15-controls.css | q (laid out) |
| `segmented` | control | choice | A segmented choice: a row of joined buttons, one pressed. | `size`: sm: the compact row inside a rack group. |  | 15-controls.css | lyrics, repaint, adapters, latent, score, build, flow, adhere |
| `termChips` | control | choice | A choice written as terminal chips after a flag (--mode Presence Frequency DRY). |  |  | 25-module-planner.css | planner, repetition |
| `lcdChips` | control | choice | Numbered chips on an LCD with a word after them (1 2 4 8 BARS). | `word`: The word printed after the chips. |  | 25-module-tempo.css | tempo |
| `ledButton` | control | boolean | An LED switch: a rack button with a lamp that lights when on. Lab rack face of a boolean. |  |  | 15-controls.css | adapters, flow, plans, replan, yuebatch, ss (laid out), master (laid out), wh (laid out) |
| `pill` | control | boolean | A pill toggle with a lamp, used inside a module look (Instrumental, Level match, a rack group power switch). |  |  | 15-controls.css | lyrics, planner, sampler, latent, adhere, seed |
| `flag` | control | boolean | A terminal flag with a check box: [x] --cot. |  |  | 25-module-planner.css | planner |
| `lcdButton` | control | boolean | A lit word on an LCD that switches (BEAT I/O, AUTO-TRIM). |  |  | 25-module-tempo.css, 15-lcd.css | tempo |
| `recButton` | control | boolean | A record button (● SAVE NEXT) that arms the next run to save something. |  | is-off (stands back: nothing new to save) | 25-module-plans.css | plans |
| `powerButton` | control | boolean | A terminal power switch with its word (● planner) and the scope switch (● Lua plugins): the switch a whole instrument hangs on. | `look`: term: the terminal bar; scope: on a scope read-out. |  | 25-module-planner.css | planner, sampler, post |
| `lockButton` | control | boolean | A padlock button that keeps a value (Keep this seed). |  |  | 15-controls.css, 25-module-seed.css | seed |
| `powerChip` | control | number | A rack group power chip for a group switched by a number: off is one value, on sets a moderate one. Says "· off" on itself when off. | `off`: The value that means off.<br>`on`: The value pressing it on sets. |  | 15-controls.css, 25-module-adhere.css | adhere |
| `token` | control | number | A scrub token: a number in running text or a terminal line, dragged sideways, stepped with the arrows, typed on Enter. |  | clash (another setting makes it skip) | 15-numbox.css, 25-module-planner.css | tempo, adapters, repetition, sampler, build, adhere |
| `textField` | control | text | A one-line text field. Checked fields show their error beside them and commit on Enter or leaving. | `look`: plain \| lcd: an LCD input \| term: an inline terminal input<br>`size`: Width in characters. | bad (fails its check); kept (invalid text kept, what is in force said) | 15-controls.css, 15-lcd.css | adapters, planner, sampler, latent, plans, ss (laid out), art (laid out) |
| `textArea` | control | text | A text area. The look gives it its material. | `look`: plain \| note: the taped sticky note (Avoid) \| staff: the ruled ABC sheet (Lead sheet) \| paper: the lyric sheet \| term: a quoted one-line terminal string that wraps (--neg) |  | 15-controls.css | repaint, avoid, planner, score |
| `choiceRow` | control | number, choice | Quick's plain choice row: a radio group of words that each set a detailed value. A value between the choices checks none: a marker stands at its place and the label says the value. | `look`: plain \| mood: the coloured How adventurous row |  | 15-controls.css | adapters, planner, sampler |
| `lookback` | control | number | A look-back window drawn as a strip of codes that grows leftward into the past, with a time scale under it. | `unit`: The word for what it counts (codes, frames, tokens).<br>`perSecond`: How many a second, for the scale and the tip.<br>`who`: Who checks for repeats, for the tip (the planner). |  | 25-module-repetition.css | repetition |
| `dropZone` | control | state | A file drop zone: a record sleeve (modules) or a slot (pedals) that takes a dropped or chosen audio file, shows its name cut to fit, and clears with ×. | `state`: The StateDef key of the file it holds.<br>`look`: sleeve \| slot<br>`empty`: Its words while empty.<br>`hint`: Its second line while empty. |  | 15-controls.css, 25-module-timbre.css, 27-pedals.css | timbre, cont, master |
| `row` | container | none | A row of knobs or controls with even spacing (the .knobs row). |  |  | 15-knob.css | repaint, sampler, guidance, latent, flow, replan, adhere, yuebatch |
| `rack` | container | none | A rack row: LED switches and chip selectors side by side on a rack strip. |  |  | 15-controls.css | flow, plans, replan, yuebatch |
| `well` | container | none | A recessed well that holds a module's main controls under its instrument (the Sampler strip). |  |  | 25-module-sampler.css | sampler |
| `lcd` | container | none | An LCD panel. Its children are rows (lcdRow) or LCD parts. |  |  | 15-lcd.css, 25-module-tempo.css | tempo |
| `lcdRow` | container | none | One row of an LCD. Ends-aligned when it holds switches at both ends. | `ends`: Spread to both ends. |  | 25-module-tempo.css | tempo |
| `terminal` | container | none | A terminal: dark, monospaced, a command bar on top, lines below. The planner and the repetition guard. | `title`: The tip on the whole terminal. |  | 25-module-planner.css | planner, repetition |
| `termBar` | container | none | The terminal's top bar: its power switch and flags. |  |  | 25-module-planner.css | planner |
| `termLine` | container | none | A command line: "$ <command>" then flags, each followed by the control for its parameter in `params`, drawn by type: a number as a token, a choice as terminal chips, a boolean as a flag, text as an inline field. Children follow the flags. Wrapped lines hang under the first flag. | `command`: The command word after $.<br>`flags`: The flag printed before each of `params`, in order (--temp, --cfg). |  | 25-module-planner.css | planner, repetition |
| `group` | container | none | A rack group: a boxed set of controls with a header that may carry its power switch; a group switched off stands back. | `title`: Its caps heading when it has no power switch. |  | 25-module-adhere.css, 25-module-latent.css | sampler, latent, adhere |
| `foot` | container | none | A tile's foot row: small labelled controls and a note, kept to the bottom of the tile. |  |  | 20-tile.css | lyrics, adapters, sampler, build, flow |
| `labelled` | container | none | A caps label beside one small control (the foot-ctl and lcdsel pairing). |  |  | 15-controls.css | lyrics, tempo, adapters, sampler, latent, score, build, flow, plans, replan, adhere, vst |
| `deck` | container | none | A two-column grid of labels and controls, so a choice never moves the rows under it (Plans & endings). |  |  | 25-module-plans.css | plans |
| `pair` | container | none | Two tokens as a range: from → to, under one label. |  |  | 25-module-adhere.css | adhere |
| `bank` | container | none | A fader bank with a 2-1-0 scale and a heading, each fader labelled on tape. | `title`: Its heading.<br>`note`: Its heading's tip. |  | 15-fader.css, 25-module-adapters.css | adapters |
| `note` | readout | none | A quiet line of words: a hint, why something is off, what another module changes. | `look`: plain \| warn \| cap: the caption note under an instrument \| hint: Quick's hint |  | 01-base.css, 15-controls.css | repaint, tempo, planner, sampler, guidance, score, cont, flow, master |
| `link` | readout | none | A text link that runs an action (Insert example, Show in Lab, Turn on Lua plugins). |  |  | 14-buttons.css | latent |
| `button` | readout | none | A small button that runs an action (+ Load adapter, Roll once). |  |  | 14-buttons.css | planner, vst |
| `lcdValue` | readout | choice, number, text | A value printed on an LCD under a caps word (KEY D MAJOR). |  |  | 25-module-tempo.css | tempo, tile, vst, master |
| `unitLcd` | readout | none, choice | A side unit's two-line LCD. Each line is a parameter's LCD text, or fixed words. | `lines`: Up to two lines: a parameter key (printed as its option's lcd text) or words in double quotes.<br>`prefix`: Words before the first line (WHISPER ).<br>`notAtLab`: Not drawn at Lab, where the unit's own controls show the same values. |  | 28-side-units.css | vst, wh |
| `unitStatus` | readout | none | A side unit's status words in small caps (After the render · needs MMS). |  |  | 28-side-units.css | fa |
| `unitBadge` | readout | none | A side unit's round score badge. |  |  | 28-side-units.css | q |
| `dice` | readout | number | A row of numbered dice, one per allowed attempt, filled as attempts are used. | `offWhen`: A boolean parameter that, when false, draws the dice off. |  | 25-module-replan.css | replan |
| `samplerScope` | bespoke | none | The Sampler's scope: the noise schedule drawn from the steps, shift, schedule and shape settings, with the CFG interval's two handles and its read-out. **Responsible for:** Drawing sigma(t) from samp.steps, samp.shift, samp.sched and the sch.* shape parameters, or from lat.ts when Latent & correction sets custom timesteps; the guidance band from samp.cs/samp.ce and the hatched CFG cutoff from perf.cfgcut; the two interval handles (they edit samp.cs and samp.ce through the value contract, a step apart); the one-line read-out (native sampling, guidance window, custom timesteps with Show). It does not own any value: defaults, undo, changed markers and saving of those parameters are generic. | `interval`: Draw the CFG interval's two handles (Lab). |  | 25-module-sampler.css | sampler |
| `keyWheel` | bespoke | choice | The circle of fifths: majors outside, minors inside, Auto in the middle, a radio group driven by the arrows. **Responsible for:** Drawing the two rings from tempo.key's options, spelling keys as a musician does (B♭, F♯), the arrow-key walk around and between rings, and the selection nudge. tempo.key stays an ordinary choice: undo, changed marker, "Also set" and saving are generic. |  |  | 25-module-tempo.css | tempo |
| `tapTempo` | bespoke | number | TAP: two taps or more within 2.5 s set the BPM from their spacing. **Responsible for:** Timing the taps and calling set on tempo.bpm (clamped to its range); the run of taps is one undo step through the parameter's `run`. |  |  | 25-module-tempo.css | tempo |
| `beatPulse` | readout | none | Lamps that pulse on the beat at the set BPM, one per beat of the bar. |  |  | 25-module-tempo.css | tempo |
| `capChips` | bespoke | none | On engines where tempo and key go in the prompt: chips that add or take out "120 BPM" or "D major" as items of the prompt, two exclusive groups. **Responsible for:** Reading which chips are pressed from the prompt text, replacing an item of the same group, never touching words inside a sentence, one undo step per press, and offering the user's own tempo and key first. |  |  | 25-module-tempo.css | tempo |
| `takesReels` | bespoke | number | The takes count as an LCD and a set of reels, one per take the engine allows; the reels are one slider. Under it, the VRAM the count costs. **Responsible for:** Drawing as many reels as the engine's takes maximum, the hover preview, the reel spin, and the VRAM cost line from the loaded models. takes.n stays a number with a per-engine range; its label per engine and its held value are generic. |  |  | 25-module-takes.css | takes |
| `splitFlap` | bespoke | state | The seed as ten split-flap cells that flip to a new number; pressing it types one. **Responsible for:** The flap animation, the ten-cell layout, typing or pasting a seed (checked 0 to 4294967295), Roll once and Random each time. The seed is the module's state `seed`; seed.lock is an ordinary boolean. | `state`: The StateDef key of the seed. |  | 25-module-seed.css | seed |
| `savedSeeds` | bespoke | none | The saved seeds list: search, star, name, describe, tag, load, delete, random from saved. **Responsible for:** The list and its own store (the session's saved seeds); every list action is its own undo step with its own name. |  |  | 25-module-seed.css | seed |
| `timbreBlend` | bespoke | number | Two overlapping circles, the reference dragged into your sound: the reference blend. **Responsible for:** Drawing the circles at the blend and dragging sideways. timbre.str is an ordinary number (its percent format, undo and changed marker are generic). |  |  | 25-module-timbre.css | timbre |
| `recentRefs` | bespoke | none | The two latest references as chips, and the rest in a popover. **Responsible for:** The session list of recent reference files and loading one into the module's file state. |  |  | 25-module-timbre.css | timbre |
| `flowWindows` | readout | none | MiniMax-Music3's flow windows drawn at the hop, and the count of windows a 3:00 song needs. |  |  | 25-module-flow.css | flow |
| `speedGauge` | readout | none | An LCD that estimates how much faster the trades make a run. |  |  | 25-module-perf.css | perf |
| `latentCurve` | bespoke | none | The latent distribution before (grey) and after (green) shift and rescale, with the XY pad that sets them laid over it as its child. **Responsible for:** Drawing the two curves from lat.shift and lat.rescale and hosting its child XY pad over them. The pad is the generic xyPad part. |  |  | 25-module-latent.css | latent |
| `adhereCurve` | readout | none | Plan strength across the run, from noise to clean, with the Plan CFG band and the CADS ramp. |  |  | 25-module-adhere.css | adhere |
| `planText` | readout | none | The plan the planner will start from, as terminal text: caption, tempo, key, form, and whose each value is. On MiniMax-Music3 its first line also says whether a saved plan is replayed or the plan cache is on (from mm.reuse, mm.plank, mm.hidpath). |  |  | 25-module-planner.css | planner |
| `vuMeter` | readout | none | The Mastering pedal's printed VU meter, -20 to +3, red from 0. |  |  | 27-pedals.css | master |
| `lrcPreview` | readout | none | Two timed lyric lines, the first named after the lyrics' first section. |  |  | 28-side-units.css | lrc |
| `coverSwatch` | readout | none | A swatch in the newest song's colour, standing for its cover art. |  |  | 28-side-units.css | art |
| `abcFoot` | bespoke | none | The lead sheet's foot: Insert example, Preview this score or Plan and preview a score, and whether the text reads as ABC. **Responsible for:** Recognising ABC (a K: line and bars), counting lines, opening the score preview (the score modal), and the way back when the planner is set to write nothing. |  |  | 25-module-score.css | score |
| `cartridges` | bespoke | state | Adapter cartridges: each loaded adapter as a cassette with label art, its strength fader, its kind's housing, and in a stack its effective share, active phase and per-section mask. **Responsible for:** The slots state (which files are loaded, per engine), adding and removing cartridges, the label art, phase windows and section masks (saved in the lyrics), and naming those steps for undo. Every dial on a cartridge is a declared parameter (s0..s2, ad.arp, ad.arc) drawn with the fader part. | `flat`: Quick's flat cartridge: the first adapter lying flat, its art beside its name. |  | 25-module-adapters.css | adapters |
| `buildBoard` | bespoke | state | Build by section: the planned sections as columns, takes under each, keep one, render the song. **Responsible for:** The build record (planned score, takes per section, keeps, the render), its drift against the lyrics, its own undo steps (Plan, Keep, Discard, Render, Start a new build), and talking to the build session API. Its four settings are declared parameters. |  |  | 25-module-build.css | build |
| `recordingSpan` | bespoke | state | A recording's waveform with up to 30 seconds selected: drag inside to move, an edge to resize, outside to draw a new one. **Responsible for:** The span state and its keyboard contract (arrows move, Shift resizes, Page moves by its length), the ruler under it, and the step names for it. |  |  | 25-module-cont.css | cont |
| `repaintRegion` | bespoke | state | The source song's waveform with the region to regenerate. **Responsible for:** The source song and region state, drawing the waveform, the region's handles and keys, and naming its steps in the song's times. |  |  | 25-module-repaint.css | repaint |
| `lyricSheet` | bespoke | text | The ruled lyric paper with its section tabs, the language stamp, the structure strip under it and the line count. **Responsible for:** Inserting section tags at line boundaries, the structure strip (sections as coloured segments that jump to their line), the line count and its time estimate, wildcard expansion. lyrics.text and lyrics.instr are ordinary parameters. |  |  | 25-module-lyrics.css | lyrics |
| `promptBox` | bespoke | text | The main text box (description, caption, prompt or tags) with its tools and the song info. **Responsible for:** The box, its rotating placeholder, Write it for me and Compose caption, the caption source; its field undo is the generic text-field contract. |  |  | 22-prompt.css | prompt |
| `modelPicker` | bespoke | choice | The engine panel's model rows, with the VRAM meter they add up to. **Responsible for:** Listing installed models from the server, the VRAM figures, and naming model steps by role (Sound model, Planner model). Each model choice is an ordinary choice parameter. |  |  | 15-pick.css, 30-engine-panel.css | models |
