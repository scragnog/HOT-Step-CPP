# Every module, pedal and side unit, in the format

All of them are written, for all five engines, in `src/definitions/`:

- `modules/` (25): the 22 desk modules of the concept, plus the three built-in parts of the Create page that own
  parameters: the prompt card (`prompt`), the lyric sheet (`lyrics`) and the engine panel's models (`models`).
- `pedals/` (10) and `units/` (5).
- `index.ts` lists them; `thumbs/` holds their drawings; `factory.ts` holds the concept's factory workflows and revisions.

Every one of the concept's 283 controls (`Object.keys(C)` on the built page, the YuE2 plugin controls included) is a
parameter in exactly one definition; the prompt's text, which the concept kept outside its controls, is the 284th
(`prompt.text`). The files were first generated from a dump of the concept's registries on every engine, so the numbers
are the concept's, and the faces were written by hand from its `defineFace` code. They are ordinary source files: edit
them directly.

## How to read the table

- **Shared parts only**: every part the definition names is used by other definitions too. Nothing about it needs code
  of its own.
- **Data plus parts of its own**: the definition is data, and one or more of its parts is used by it alone: a read-out
  whose drawing or formula is that module's, a container of its look, or a bespoke instrument inside a face that is
  otherwise data. The parameters are all declared, so everything generic works.
- **Mostly bespoke**: the face is essentially one bespoke part doing the module's work. Its parameters and state are
  still declared, so undo, saving, ownership, the changed marker and the request are still derived; what the part draws
  and the actions it offers are its own.

| Item | Kind | Engines | Params | Class | Its own parts, and why |
| --- | --- | --- | --- | --- | --- |
| Prompt (`prompt`) | built in | all | 8 | Mostly bespoke | `promptBox`: the main box, its placeholder, Write it for me, Compose caption, the caption source, the trigger word and the imported latent are behaviour, not layout; its named steps are the definition's. |
| Lyrics (`lyrics`) | built in | ace, mm3, yue2, hm | 4 | Mostly bespoke | `lyricSheet`: tag insertion at line boundaries, the structure strip, the line count and wildcard expansion. The foot (Instrumental, vocal gender) is data. |
| Models (`models`) | built in | all | 5 | Mostly bespoke | `modelPicker`: server-listed files with VRAM figures. Each model choice is a declared parameter. |
| Repaint | module | ace | 3 | Mostly bespoke | `repaintRegion`: the source song, its waveform and the region are state with their own steps. |
| Tempo, key & length (`tempo`) | module | all | 11 | Data plus parts of its own | The LCD and its rows, the rulers and every per-engine name are data. Bespoke: `keyWheel`, `tapTempo`, `capChips` (prompt chips on Stable Audio 3 and HeartMuLa). |
| Avoid | module | ace | 1 | Shared parts only | A text area with the note look. |
| Timbre reference (`timbre`) | module | ace | 1 | Data plus parts of its own | The drop zone is shared, and shows Mastering's reference when that pedal lends it. `timbreBlend` and `recentRefs` are its own. |
| Adapters | module | ace, mm3, yue2, hm | 31 | Mostly bespoke | `cartridges`: the slots per engine, label art, each slot's phase window, and section masks written into the lyrics. The fader banks, rebase row, stack foot and option foot are data. |
| Planner | module | ace, mm3, yue2, hm | 32 | Data plus parts of its own | The terminal, its lines, flags and tokens, the XY pad and Quick's mood row are data, per engine. `planText` is its own read-out. |
| Repetition guard (`repetition`) | Lab | ace, mm3, yue2 | 12 | Data plus parts of its own | Terminal lines; `lookback` is its own. |
| Sampler | module | all | 56 | Data plus parts of its own | The well, the pickers, the schedule shape groups and all YuE2 plugin control groups are data. `samplerScope` is its own. |
| Guidance plugin (`guidance`) | Lab | ace, mm3, sa3, hm | 2 | Shared parts only | An XY pad, a knob and three strips. |
| Performance trades (`perf`) | Lab | ace | 3 | Data plus parts of its own | XY pad and knob; `speedGauge` is its own estimate. |
| Latent & correction (`latent`) | Lab | ace | 15 | Data plus parts of its own | Three rack groups with power pills, the timesteps field and its check are data. `latentCurve` draws under the XY pad. |
| Lead sheet (`score`) | module | yue2 | 2 | Data plus parts of its own | The staff text area and lyric timing are data. `abcFoot` recognises ABC and offers its actions. |
| Build by section (`build`) | module | yue2 | 4 | Mostly bespoke | `buildBoard`: sections, takes, keeps and the render talk to a build session. Its strips and four settings are data. |
| Continue from recording (`cont`) | Lab | yue2 | 0 | Mostly bespoke | `recordingSpan`: the 30-second selection is state with its own keyboard contract. The drop zone is shared. |
| Flow renderer (`flow`) | Lab | mm3 | 7 | Data plus parts of its own | Knobs, LED switches and segmented choices; `flowWindows` is its own drawing. |
| Plans & endings (`plans`) | Lab | mm3 | 10 | Data plus parts of its own | LCD pickers and LED switches with every "not used" state as a condition; `deck` and `recButton` are its own. |
| Re-planning (`replan`) | Lab | yue2 | 4 | Data plus parts of its own | LED switches and knobs; `dice` is its own. |
| Plan adherence (`adhere`) | Lab | yue2 | 12 | Data plus parts of its own | Rack groups, tokens and pills; `adhereCurve`, `powerChip` and `pair` are its own. |
| Throughput (`yuebatch`) | Lab | yue2 | 4 | Shared parts only | An LED switch and three knobs. |
| Seed | module | all | 1 | Mostly bespoke | `splitFlap` and `savedSeeds`: the seed and the saved list are state with their own steps; the lock is a declared parameter. |
| Takes | module | all | 1 | Mostly bespoke | `takesReels`: reels per engine maximum and the VRAM line. The count's range, words and held value per engine are data. |
| Pedalboard (`post`) | module | all | 1 | Shared parts only | The chain switch; the board itself is a shell. |
| Tiled decoder (`tile`) | pedal | ace, sa3 | 2 | Shared parts only | Laid out; its idle reason and Turn on action are data. |
| MM3 latent refiner (`refiner`) | pedal | all | 5 | Shared parts only | Laid out. |
| StableStep (`ss`) | pedal | ace, mm3, yue2, hm | 12 | Shared parts only | Laid out; the crossover and mix knobs keep their room behind a blanking plate; the fixed seed stands beside its switch. |
| PP-VAE (`ppvae`) | pedal | ace | 2 | Shared parts only | Laid out. |
| Spectral lifter (`lift`) | pedal | ace | 6 | Shared parts only | Laid out. |
| Vocal naturalizer (`nat`) | pedal | ace, mm3, yue2 | 8 | Shared parts only | Laid out. |
| Pre-VST gain (`gain`) | pedal | all | 1 | Shared parts only | Laid out; always on. |
| VST3 chain (`vst`) | pedal | all | 1 | Shared parts only | Its LCD, idle reason and plugin count are data; + Add runs the named action `vstHost`. |
| Mastering (`master`) | pedal | all | 2 | Data plus parts of its own | The reference slot and its three notes are data; `meter` is used by it alone so far. |
| Final normalizer (`norm`) | pedal | all | 4 | Shared parts only | Laid out; the presets' targets are in its wire. |
| Lyric timestamps (`lrc`) | unit | ace, mm3 | 1 | Data plus parts of its own | `lrcPreview` reads the first section's name from the lyrics. |
| Forced alignment (`fa`) | unit | yue2 | 1 | Data plus parts of its own | `unitStatus`. |
| Whisper lyrics (`wh`) | unit | ace, mm3, yue2 | 5 | Shared parts only | A two-line unit LCD reading its own parameters; its controls laid out at Lab. |
| Cover art (`art`) | unit | ace, mm3, yue2 | 2 | Data plus parts of its own | `coverSwatch` takes the newest song's colour. |
| Quality evaluator (`q`) | unit | ace, mm3, yue2, sa3 | 2 | Data plus parts of its own | `unitBadge`; its choice laid out as chips. |

**Counts: 14 built from shared parts only, 17 data plus parts of their own, 9 mostly bespoke (40 in all).** Every pedal
and side unit is data, with at most one read-out of its own. The nine mostly bespoke items are the three built-in parts
of the page and six modules whose instrument is the module (the cartridges, the build board, the split-flap, the reels,
the two waveform selections).

## Things the definitions handle differently from the concept

Each is a decision, recorded here so that a comparison with the concept can tell it from a mistake.

- **Cover art subject** (`pp.art.subject`) is drawn in the song info on the prompt card, as the concept has it, and also
  laid out at Lab with the Cover art unit's other settings, because a unit's parameters are laid out by rule.
- **YuE2's NAR plugin controls** were generated at load from a table (`YUEPLUG` in `25-module-sampler.js`). They are
  ordinary parameters here (`yn.<plugin>.<n>`), each with `when` on the solver or scheduler that brings it.
- **The Sampler's schedule shapes** (Beta, Power, Composite) are separate parameters in the concept and here; the request
  packs them into the scheduler string through the `scheduler` transform.
- **Conditions the concept wrote in faces** are a parameter's `when` here where they decide whether it applies (APG's two
  settings, the DRY settings, the codes strength and steps, the normalizer's Custom knob, the retake seed, the lead
  sheet's text): the concept hides the same controls in its face, and here the condition also keeps the value out of the
  request.
- **Owners** follow the concept's rule (`isEngOwn`): the Planner and the Lab modules under it (Repetition guard, Plans
  and endings, Re-planning, Throughput) and Build by section travel with the song. Two groups the concept's rule misses
  are kept per engine as its own comment says adapter and sampler settings are: the adapter group scales (`ag.*`) and
  YuE2's plugin controls (`yn.*`). `tools/fixtures/concept-differences.json` records it.
- **Takes** on Stable Audio 3 and HeartMuLa read as one take (range 1 to 1), as the concept's engine limit has it, and
  send nothing; ACE-Step allows 8, the server's `batch.max`, where the concept allows 9 and the existing page sends 8
  for 9 ([08](08-open-questions.md) question 3).
- **Ranges and steps** follow the concept where it differs from the existing UI's sliders, and each such wire's note says
  so ([04](04-parameter-names.md), "Different step or range").
- **APG on Stable Audio 3 and HeartMuLa**: the concept offers the APG pad there; no code those engines run reads the two
  values, which their wires' notes say ([08](08-open-questions.md) question 3).
- **The denoiser and LSS switches** have no field in the real app; here each switch gates its strength's wire, which
  sends 0 while it is off.
- **The planner's tier summaries** and Tempo's say what each tier adds on each engine; the concept's say ACE-Step's on
  every engine, and Tempo's put Beat I/O at Lab where its face shows it at Studio.
- **MiniMax-Music3's renderer** offers Auto, the server's own choice when nothing is set (TensorRT when the machine has
  it, else GGML), and starts there; the concept offers GGML and TensorRT and starts on GGML, which the server does not do.
  The precision applies on Auto and TensorRT.
- **The trigger word** is sent on ACE-Step and MiniMax-Music3; on YuE2 and HeartMuLa the engine adds its adapter's own
  trigger and the existing page sends no typed one, so the definitions send none there either.
- **Lua plugins on, the solver, schedule and guidance left alone** send Native (the engine's own loop), as the concept
  starts them; the existing page sends ACE-Step's picks there ([08](08-open-questions.md) question 5).
- **One format everywhere**: the concept's undo steps write a length as m:ss and the reference blend as a percent,
  where its read-outs write 10 s and 1.00. The definitions write a value one way everywhere.
- **The denoiser and LSS strengths** have undo names of their own ("Denoiser strength", "LSS strength"); the concept
  names both "Latent & correction strength".
