# Every module, pedal and side unit, in the format

All of them are written, for all five engines, in `src/definitions/`:

- `modules/` (25): the 22 desk modules of the concept, plus the three built-in parts of the Create page that own
  parameters: the prompt card (`prompt`), the lyric sheet (`lyrics`) and the engine panel's models (`models`).
- `pedals/` (10) and `units/` (5).
- `index.ts` lists them; `thumbs/` holds their drawings.

Every one of the concept's 283 controls (`Object.keys(C)` on the built page, the YuE2 plugin controls included) is a
parameter in exactly one definition; the prompt's text, which the concept kept outside its controls, is the 284th
(`prompt.text`). The files were first generated from a dump of the concept's registries on every engine (ranges,
defaults, labels and options as the page computes them per engine), so the numbers are the concept's, and the faces were
then written by hand from its `defineFace` code. They are ordinary source files now: edit them directly.

## How to read the table

- **Pure data**: the definition names only shared parts. Nothing about it needs code of its own.
- **Data plus a named part**: the definition is data, and one or more of its parts is written for it: a bespoke part, or a
  read-out whose drawing or formula is that module's own. The parameters are all declared, so everything generic works.
- **Mostly bespoke**: the face is essentially one bespoke part doing the module's work. Its parameters and state are still
  declared, so undo, saving, ownership and the changed marker are still derived; what the part draws and the actions it
  offers are its own.

| Item | Kind | Engines | Params | Class | Why |
| --- | --- | --- | --- | --- | --- |
| Prompt (`prompt`) | built in | all | 7 | Mostly bespoke | `promptBox`: the main box, its placeholder, Write it for me, Compose caption and the caption source are behaviour, not layout. Its parameters (the text, song info, caption source, wildcards) are declared. |
| Lyrics (`lyrics`) | built in | ace, mm3, yue2, hm | 4 | Mostly bespoke | `lyricSheet`: tag insertion at line boundaries, the structure strip, the line count and wildcard expansion. The foot (Instrumental, vocal gender) is data. |
| Models (`models`) | built in | all | 5 | Mostly bespoke | `modelPicker`: server-listed files with VRAM figures. Each model choice is a declared parameter with a per-engine role name for undo. |
| Repaint | module | ace | 3 | Mostly bespoke | `repaintRegion`: the source song, its waveform and the region are state with their own keys and step names. Mode, crossfade and lyrics are data. |
| Tempo, key & length (`tempo`) | module | all | 11 | Data plus named parts | The LCD, rulers, notes and every per-engine name are data. Bespoke: `keyWheel` (the circle of fifths and its keyboard walk), `tapTempo`, `beatPulse`, `capChips` (prompt chips on Stable Audio 3 and HeartMuLa). |
| Avoid | module | ace | 1 | Pure data | A text area with the note look. |
| Timbre reference (`timbre`) | module | ace | 1 | Data plus named parts | The drop zone is shared. `timbreBlend` (the two circles) and `recentRefs` are its own. |
| Adapters | module | ace, mm3, yue2, hm | 32 | Mostly bespoke | `cartridges`: loading files per engine, label art, slots per engine, phase windows and section masks saved in the lyrics. The fader banks, rebase row, stack bar and option foot around it are data, and all 32 dials are declared. |
| Planner | module | ace, mm3, yue2, hm | 32 | Data plus named parts | The terminal, its command lines, flags, tokens, XY pads and Quick's mood row are data, per engine. `planText` (the plan the planner starts from) is its own read-out. |
| Repetition guard (`repetition`) | Lab | ace, mm3, yue2 | 12 | Pure data | Terminal lines and the shared look-back strip, per engine. |
| Sampler | module | all | 56 | Data plus named parts | The well, the pickers, the schedule shape groups and all YuE2 plugin control groups are data. `samplerScope` (the schedule drawing and the CFG interval handles) is its own. |
| Guidance plugin (`guidance`) | Lab | ace, mm3, sa3, hm | 2 | Pure data | An XY pad, a knob and three strip conditions. |
| Performance trades (`perf`) | Lab | ace | 3 | Data plus named parts | XY pad and knob; `speedGauge` is its own estimate. |
| Latent & correction (`latent`) | Lab | ace | 15 | Data plus named parts | Three rack groups with power pills, the timesteps field and its check are data. `latentCurve` draws the distribution under the XY pad. |
| Lead sheet (`score`) | module | yue2 | 2 | Data plus named parts | The staff text area and lyric timing are data, the timing warning a condition. `abcFoot` recognises ABC and opens the score preview. |
| Build by section (`build`) | module | yue2 | 4 | Mostly bespoke | `buildBoard`: sections, takes, keeps and the render talk to a build session. Its four settings are data. |
| Continue from recording (`cont`) | Lab | yue2 | 0 | Mostly bespoke | `recordingSpan`: the 30-second selection on a waveform is state with its own keyboard contract. The drop zone is shared. |
| Flow renderer (`flow`) | Lab | mm3 | 7 | Data plus named parts | Knobs, LED switches and segmented choices are data; `flowWindows` is its own drawing. |
| Plans & endings (`plans`) | Lab | mm3 | 10 | Pure data | The deck, record buttons, LCD pickers and LED switches, with every "not used" state as a condition. |
| Re-planning (`replan`) | Lab | yue2 | 4 | Pure data | Dice, LED switches and knobs. |
| Plan adherence (`adhere`) | Lab | yue2 | 12 | Data plus named parts | Three rack groups with power chips, tokens and pills are data; `adhereCurve` is its own drawing. |
| Throughput (`yuebatch`) | Lab | yue2 | 4 | Pure data | An LED switch and three knobs. |
| Seed | module | all | 1 | Mostly bespoke | `splitFlap` and `savedSeeds`: the seed itself is state, typed into the flaps; the saved list is the session's. The lock is a declared parameter. |
| Takes | module | all | 1 | Mostly bespoke | `takesReels`: reels per engine maximum and the VRAM line. The count's range, label and held value per engine are data. |
| Pedalboard (`post`) | module | all | 1 | Pure data | The chain switch; the board itself is a shell. |
| Tiled decoder (`tile`) | pedal | ace, sa3 | 2 | Pure data | Laid out from its parameters; idle reason and its Turn on action are data. |
| MM3 latent refiner (`refiner`) | pedal | all | 5 | Pure data | Laid out from its parameters. |
| StableStep (`ss`) | pedal | ace, mm3, yue2, hm | 12 | Pure data | Laid out; the crossover and mix knobs keep their room behind a blanking plate whose words are data; the fixed seed stands beside its switch. |
| PP-VAE (`ppvae`) | pedal | ace | 2 | Pure data | Laid out. |
| Spectral lifter (`lift`) | pedal | ace | 6 | Pure data | Laid out. |
| Vocal naturalizer (`nat`) | pedal | ace, mm3, yue2 | 8 | Pure data | Laid out. |
| Pre-VST gain (`gain`) | pedal | all | 1 | Pure data | Laid out; always on. |
| VST3 chain (`vst`) | pedal | all | 1 | Data plus named parts | Its LCD and idle reason are data; + Add runs the bespoke action `vstHost` (the plugin host is outside the concept). |
| Mastering (`master`) | pedal | all | 2 | Data plus named parts | The reference slot and its three notes are data; `vuMeter` is its own drawing. |
| Final normalizer (`norm`) | pedal | all | 4 | Pure data | Laid out; the custom target's plate words are data. |
| Lyric timestamps (`lrc`) | unit | ace, mm3 | 1 | Data plus named parts | `lrcPreview` reads the first section's name from the lyrics. |
| Forced alignment (`fa`) | unit | yue2 | 1 | Pure data | A status line. |
| Whisper lyrics (`wh`) | unit | ace, mm3, yue2 | 5 | Pure data | A two-line unit LCD from its own parameters; its controls laid out at Lab. |
| Cover art (`art`) | unit | ace, mm3, yue2 | 2 | Data plus named parts | `coverSwatch` takes the newest song's colour. |
| Quality evaluator (`q`) | unit | ace, mm3, yue2, sa3 | 2 | Pure data | A badge; its choice laid out as chips. |

**Counts: 18 pure data, 13 data plus a named part, 9 mostly bespoke (40 in all).** Every pedal and side unit is pure
data or data plus one read-out. The nine mostly bespoke items are the three built-in parts of the page and six modules
whose instrument is the module (the cartridges, the build board, the split-flap, the reels, the two waveform selections).

## Things the concept does that the definitions handle differently

- **Cover art subject** (`pp.art.subject`) is declared in the concept and saved as song content, but no face draws it.
  Here it is a Lab control of the Cover art unit, laid out like the unit's other settings.
- **YuE2's NAR plugin controls** were generated at load from a table (`YUEPLUG` in `25-module-sampler.js`). They are
  ordinary parameters here (`yn.<plugin>.<n>`), each with `when` on the solver or scheduler that brings it, and the real
  names are the server's `yue2Wass*`, `yue2Eq*`, `yue2Lie*`, `yue2Od*` and `yue2Ht*` extensions.
- **The Sampler's schedule shapes** (Beta, Power, Composite) are separate parameters in the concept and here; the real
  app packs them into the scheduler string (`'beta:0.50:0.70'`), recorded in their wire notes.
- **The Tempo tile's time signature and key** have "Auto" as an option, as the concept does, stored as `''` because that
  is what the real app sends for auto.
- **The Adapters bank** is two banks here, one per engine family (ACE-Step's group scales; the others' master section),
  where the concept chose the keys at render time. The result on screen is the same.
