# Generation

The bar across the top of the window holds the settings that apply to every song you
generate, whichever studio you start it from. A studio supplies the content (caption,
lyrics, source audio, a region to repaint). The bar supplies everything else: which
models run, how the sampler steps, what the language model does, and what happens to the
audio after it is rendered. This page covers the bar, the post-processing chain, the job
queue and the task modes each studio uses.

Settings in the bar are saved in your browser's local storage and come back the next
time you open the app. A different browser, or one with its site data cleared, starts from
the defaults listed below. To keep a set of settings you can return to, save a
[profile](#profiles).

## The bar at a glance

| Section | What it holds |
|---|---|
| Models | The DiT, LM, VAE decoder and text encoder in use, plus a link to the Model Manager. |
| Backend | The active generation backend. Only shown when more than one backend is installed. See [Backends](backends.md). |
| Adapters | LoRA and LoKr adapters for the DiT and the planner LM. See [Adapters](adapters.md). |
| Generation | Steps, guidance, shift, solver, schedule, guidance mode, seed, batch size and the engine-side audio correction stages. |
| LM / Thinking | The language model stage. The switch in the section header turns the LM on or off. |
| Post-Processing | Everything that runs on the finished audio. The switch in the header is the master switch for the chain. |
| Profiles | Save, apply, import and export named snapshots of every setting. |
| VRAM | GPU memory in use. Hover it to see which models are loaded and unload them. |

Each section shows a one-line summary of its current state. The Generation summary, for
example, reads like `12s · Euler · Linear · APG 9.0 · σ3.0 · Seed Rnd`: steps, solver,
schedule, guidance mode and scale, shift, and whether the seed is random or fixed.

Hovering a section opens it and moving away closes it. Clicking the section header pins it
open, so you can reach a file picker or another window without it collapsing. A pinned
section closes when you click its header again, click outside it, or press Esc. Most
controls have a small help card: hover the label to read what the control does.

The bar changes shape with the active backend. MiniMax-Music3 and YuE2 show their own
model pickers, planner controls and adapter panels, and a section the backend has no use
for says so instead of showing ACE-Step controls that would do nothing. Everything below
describes the ACE-Step 1.5 layout. For the others, see [Backends](backends.md).

## Models

| Control | What it does |
|---|---|
| Format | Filter chips (GGUF, SafeTensors, ONNX), shown when models in more than one format are installed. They only filter the lists; a selected model stays selected when its format is hidden. |
| DiT Model | The diffusion transformer that turns the plan into audio latents. The main quality and VRAM decision. |
| LM Model | The 5 Hz language model that writes metadata, the rewritten caption and the audio codes. |
| VAE Decoder | Turns latents into audio. Only shown when more than one VAE is installed. |
| Text Encoder | Only shown when more than one text encoder is installed. |
| Get More Models | Opens the Model Manager. |

Any empty slot is filled with the first installed model of that type. On first launch, if
no models are found after the engine has had a few seconds to start, the Model Manager
opens by itself. When the active backend reports its own weights missing, a banner under
the bar links to the Model Manager; nothing downloads until you ask.

Which quant to pick is covered in [Getting higher quality output](quality.md) and the
download list in [Models](models.md).

## Generation

| Control | Range | Default | What it does |
|---|---|---|---|
| Inference Steps | 1 to 300 | 12 | Denoising steps. More steps take longer; past roughly 40 on ACE-Step they stop helping. |
| Guidance Scale | 0 to 20 | 9.0 | How hard the DiT is pushed toward the caption and lyrics (classifier-free guidance strength). |
| Performance | | all off | Three speed-for-quality trades. See [Performance](#performance). |
| Shift | Auto, or 0 to 10 | 3.0 | Timestep shift. Biases the noise schedule toward the noisy end, trading fine detail for structure. Auto derives it from the duration and the step count. Switching Auto off returns to 3.0. |
| Solver | plugin list | Euler | How each step moves from noise toward audio. |
| Schedule | plugin list | Linear | How the steps are spaced between full noise and clean audio. |
| Guidance | plugin list | APG | How the conditional and unconditional predictions are combined each step. |
| Timbre Reference | audio file | none | See [Timbre reference](#timbre-reference). |
| DCW Correction | toggle | off | See [DCW correction](#dcw-correction). |
| Auto-Trim Endings | toggle | off | See [Auto-trim and the duration buffer](#auto-trim-and-the-duration-buffer). |
| Latent Post-Processing | | no change | See [Latent post-processing](#latent-post-processing). |
| Denoiser | toggle | off | See [Denoiser and LSS](#denoiser-and-lss). |
| LSS | toggle | off | See [Denoiser and LSS](#denoiser-and-lss). |
| Generation Seed | 0 to 2147483647 | Random | See [Seed and the Seed Manager](#seed-and-the-seed-manager). |
| Batch Size | 1 to 9 | 1 | See [Batch size](#batch-size). |

### Solver, schedule and guidance

These three pickers are Lua plugins, so the lists grow when a plugin is added and each
entry carries its own description in its help card.

- The solver decides how one step is taken. Solvers are grouped by how many network
  evaluations (NFE) each step costs. A 2-NFE solver at 20 steps costs about what a 1-NFE
  solver costs at 40.
- The schedule decides where the steps fall. Three extra entries are built into the picker
  rather than coming from plugins. Beta (Custom) has Alpha and Beta sliders (0.1 to 2.0,
  defaults 0.5 and 0.7). Power has an Exponent slider (0.25 to 4.0, default 2.0), where
  above 1 favours structure and below 1 favours detail. Composite (2-Stage) switches from
  Stage A to Stage B partway through, set by Crossover and Split (0.1 to 0.9, defaults 0.5).
- The guidance mode decides how the Guidance Scale is applied. APG, the default, has two
  extra sliders: Momentum (0 to 1, default 0.75) smooths guidance across steps, and Norm
  Threshold (0 to 10, default 2.5) clips its magnitude.

A plugin that declares its own settings shows them in a collapsible panel under its
picker, with a Reset button. The full lists with descriptions are in
[Plugins](plugins.md).

### Performance

Collapsed by default. Its header turns amber and lists what is active whenever any of the
three is changed from its default.

| Control | Range | Default | What it does |
|---|---|---|---|
| CFG Cutoff | 0 to 1 | 1 | Fraction of DiT steps that use full guidance. 0.5 is about 20% faster but can loosen prompt adherence. |
| LM CFG Cutoff | 0.3 to 1 | 1 | Fraction of LM audio-code tokens that use guidance. 0.7 is about 15% faster on the LM stage. |
| Step Cache | 0 to 0.7 | 0 | Reuses the previous step's velocity to skip forward passes. The help text suggests 0.3 to 0.5. |

### Timbre reference

A reference track whose tone and texture guide the render. It is VAE-encoded and fed to
the DiT during synthesis. Pick an uploaded file from the list or use Upload Reference.
Uploads here and in Mastering share one list of reference tracks.

With no timbre reference set, the mastering reference is used instead when Mastering is on
and its "Also use as timbre reference" switch is on. A dedicated timbre reference always
wins over the mastering one.

If a saved profile points at a reference file that has since been deleted, the generation
is refused at submit with a message telling you to re-upload or clear it.

### DCW correction

A wavelet-domain correction applied during sampling, with its strength varied by timestep.

| Correction Mode | What it corrects |
|---|---|
| Low-Frequency | Low bands: bass, kick and rhythm. |
| High-Frequency | High bands: hi-hats, vocals and presence. |
| Both (Low + High) | Both, with separate Low-Freq and High-Freq scalers. The default mode. |
| Pixel-Space (No Wavelets) | Corrects the latent directly. More uniform, less targeted. |

Scalers run 0 to 1 and default to 0.2. Reset restores Both at 0.2 and 0.2.

### Auto-trim and the duration buffer

When on, the engine is asked for Duration Buffer seconds more than the song length you
set (5 to 30, default 15). After rendering, the audio is scanned from the end for a
natural ending, a gap of silence of at least two seconds inside the buffer. If one is
found the track is cut there with no fade. If not, it is cut at your requested length and
faded out over Fade-Out seconds (0.5 to 5, default 2). The trim happens before
post-processing, so every later stage works on the trimmed audio.

Auto-trim only acts when you set a duration. With duration on Auto the LM picks the length
and no buffer is added. MiniMax-Music3 ignores the buffer entirely.

### Latent post-processing

Applied to the DiT's output latents just before the VAE decodes them. Reset returns all
three to no change.

| Control | Range | Default | What it does |
|---|---|---|---|
| Latent Shift | -2 to 2 | 0 | Added to every latent value. |
| Latent Rescale | 0.1 to 3 | 1 | Every latent value is multiplied by this before the shift is added. |
| Custom Timesteps | text | empty | A comma-separated list of descending values, for example `0.97,0.76,0.5,0.28,0.085,0`. When set, it replaces both the schedule and the step count. |

### Denoiser and LSS

Both run inside the engine as part of the render, so they are not affected by the
Post-Processing master switch.

The Denoiser is a spectral gate that removes VAE fuzz after decoding. Switching it on sets
Strength to 0.5 (range 0.01 to 1). Smoothing (0 to 1, default 0.7) and Mix (0 to 1,
default 0.25) shape how it is applied.

LSS (Latent Spectral Suppressor) works before decoding: latent channels whose variance is
below Var Threshold (0.01 to 0.5, default 0.15) are pulled down toward 1 minus Strength.
Switching it on sets Strength to 0.65. DC Remove, on by default, also removes each
channel's DC offset.

### Seed and the Seed Manager

With Random on (the default) every generation draws a new seed. Turn Random off to type a
seed or step it with the minus and plus buttons; the same seed, settings and content
reproduce the same render. The seed actually used is stored with the song even when it was
random, so a take you like can be reproduced later. This seed drives the DiT. The LM has
its own seed, described under [LM / Thinking](#lm--thinking).

The save icon beside Generation Seed opens the Seed Manager:

- Save the current seed under a name (default `seed_<number>`) with an optional description.
- Click a saved seed to load it. Loading a seed also switches Random off.
- Star a seed to keep it at the top of the list; search filters by name, description and tags.
- The "random" button loads a random seed from your saved list.
- Delete takes two clicks.

Saved seeds are stored by the server, so they are the same in every browser.
<!-- TODO(verify): SeedManagerDrawer's header comment says it imports existing ComfyUI SeedSaver files; confirm where the server reads them from before documenting it. -->

### Batch size

Asks for several tracks from one request (1 to 9). The LM plans that many variations, then
the DiT renders them one after another, each as its own song in the Library. With the LM
switched off, and for cover and repaint tasks, the extra tracks reuse the same plan with a
different random DiT seed each. The generation timeout in Settings applies per track, not
per batch.

## LM / Thinking

The switch in the section header turns the LM stage on or off. With it off, the DiT works
from your caption and lyrics alone, and any metadata left on Auto is filled with fixed
values: 120 BPM, 120 seconds, C major, 4/4.

| Control | Range | Default | What it does |
|---|---|---|---|
| Chain-of-Thought Caption | toggle | on | When on, the LM rewrites your caption into a fuller style description before generation. When off, your caption is used as typed. |
| Temperature | 0 to 2 | 0.8 | Higher is more varied, lower is more predictable. |
| CFG Scale | 0 to 10 | 2.2 | Guidance strength for the LM itself. |
| Top-K | 0 to 200 | 0 | Limits sampling to the K most likely tokens. 0 turns it off. |
| Top-P | 0 to 1 | 0.92 | Nucleus sampling threshold. |
| Repetition Penalty | 1.0 to 1.5 | 1.1 | Penalises recently used audio codes to break stuck loops. 1.0 turns it off. |
| Rep. Mode | Presence, Frequency, DRY | Presence | Shown when the penalty is above 1.0. See below. |
| Rep. Window (codes) | 8 to 256 | 64 | How far back the penalty looks. The LM emits 5 codes per second, so 64 codes is 12.8 seconds; the label shows the seconds. |
| DRY Base, DRY Min Match | 1.05 to 4, 2 to 32 | 1.75, 3 | DRY mode only. How fast the penalty grows with match length, and the shortest verbatim repeat it acts on. |
| Negative Prompt | text | `NO USER INPUT` | Negative conditioning text for the LM. |
| LM Codes | Strength 0 to 1, or Step Count | Strength 1.0 | How many DiT steps the LM's audio codes condition. Strength is a fraction of the step budget. With Step Count on, you set an absolute number of steps (default 6) that stays fixed when you change Inference Steps. |
| LM Seed | toggle and number | Use DiT Seed on | With Use DiT Seed on, the LM seed follows the Generation Seed: a fixed Generation Seed makes both deterministic, a random one makes both random. Turn it off to set an independent LM seed, which has its own Seed Manager. |

The three repetition modes trade loop-breaking against keeping deliberate repetition:

- Presence penalises each distinct code in the window once, however often it recurred. It
  cannot tell a stuck loop from a chorus coming back, so a strength that breaks loops also
  flattens structure.
- Frequency scales the penalty with how often a code recurred (capped at 8), so a loop is
  hit much harder than the music around it. Use a lower penalty than with Presence.
- DRY only penalises codes that would extend a verbatim recent repeat, growing with the
  length of the match. Raise DRY Min Match if sustained textures get chewed up.

Repeat generations with the same seed and LM settings can skip the LM entirely: Settings has
a "Cache LM audio codes" option that reuses the previous codes. See
[Settings](studios/settings.md).

## Post-Processing

Everything in this section runs on the audio after the engine has rendered it. The switch
in the section header is a master switch: with it off, none of the stages below run,
whatever their own switches say. The exceptions are Lyric Timestamps (LRC) and Whisper
Lyrics, which work with the master switch off.

Post-processing never overwrites the render. It works on a copy, and the song keeps both
the original and the processed file; the Library lets you play either. If the master
switch was off when a song was made, you can run the chain on it later from the track's
menu. A song that has already been processed is refused, so a chain cannot be stacked
twice. See [Library](studios/library.md).
<!-- TODO(verify): exact label of the "run post-processing" entry in the Library track menu. -->

### Order of the chain

The stages always run in this order, per track:

1. Quality Evaluator on the unprocessed audio (if its target includes Unmastered).
2. Vocal separation, when StableStep or Whisper's "Isolate vocals first" needs it. Runs once
   and is shared.
3. StableStep.
4. PP-VAE Re-encode.
5. Spectral Lifter.
6. Vocal Naturalizer (skipped on instrumentals).
7. Pre-VST Gain Offset.
8. VST Chain.
9. Mastering.
10. Final Normalizer.
11. A peak guard that limits the file to just under full scale if a stage that was meant
    to control peaks failed.
12. Quality Evaluator on the processed audio (if its target includes Mastered).

A stage that fails is logged and skipped; the song is still saved. Cover art runs after
the chain, or alongside it when Parallel Cover Art is on in Settings.

### Stages

| Stage | Default | What it does |
|---|---|---|
| Lyric Timestamps (LRC) | on | Synchronised lyric timing for karaoke-style playback, taken from the DiT's lyric attention during the render. Hidden on backends that cannot provide it. |
| Whisper Lyrics | off | Transcribes the sung lyrics with word-level timestamps, using your lyrics as a spelling guide. Model (Auto-detect, Large v3 Turbo, Large v3, Medium, Base), Language, Beam Size (1 to 10, default 5) and "Isolate vocals first". Needs a Whisper model from the Model Manager. |
| Tiled Decoder | off | Replaces the built-in VAE decode with a tiled decode plugin (overlap crossfading, optional dual-pass merge, channel suppression and a small DSP chain). Only shown when a postprocess plugin is installed; its settings come from the plugin. |
| PP-VAE Re-encode | off | Runs the decoded audio through a second, higher-fidelity autoencoder to clean up fizz and high-frequency noise. Adds roughly 1 to 2 seconds. Original Blend (0 to 1, default 0) mixes the unprocessed audio back in. The ONNX (ORT/TRT) switch, on by default, uses ONNX Runtime with TensorRT and falls back to GGUF when the ONNX files are missing. Only shown when the PP-VAE model is installed. |
| StableStep | off | Re-renders the instrumental through Stable Audio 3 to replace VAE fizz with real detail. See [StableStep](#stablestep). |
| Spectral Lifter | off | Removes AI shimmer, reduces spectral noise and can extend the high end. Denoise Strength (default 0.3), Noise Floor (0.1), HF Extension (0), Transient Boost (0), Shimmer Reduction (6 dB, applied to the 10 to 14 kHz band). |
| Vocal Naturalizer | off | Experimental. Five DSP stages that pull back robotic and auto-tune artefacts: vibrato, formant variation, metallic cut (6 to 10 kHz), quantization masking and transition smoothing, with a master Amount (default 0.5). It processes the whole mix, so A/B it against the same render with it off. |
| Pre-VST Gain Offset | 0 dB | A fixed gain (-10 to +10 dB) before the VST chain and mastering. Negative values give a VST chain headroom. When something runs after it, peaks above full scale are passed on for that stage to handle; when nothing does, it limits. |
| VST Chain | empty | Your own VST3 plugins. See [VST chain](#vst-chain). |
| Mastering | off | Matches the level, frequency balance and dynamics of a reference track you upload. "Also use as timbre reference" feeds the same track to the DiT as a [timbre reference](#timbre-reference). |
| Final Normalizer | off | Always the last stage that changes the audio. Measures integrated loudness (ITU-R BS.1770-4) and sets the gain to hit a target: Spotify / YouTube Music (-14 LUFS, the default), Apple Music / Tidal (-16), EBU R128 (-23), Club / DJ Playback (-8) or Custom (-30 to -5). A look-ahead limiter holds peaks at the Peak Ceiling (-6 to 0 dBFS, default -1). The resulting true peak is written to the generation log. |
| Cover Art | off | See [Cover art](#cover-art). |
| Quality Evaluator | off | Scores each track from spectral analysis and shows the score as a badge in the Library. Evaluate: Unmastered (default), Mastered or Both. |

When the active backend is not ACE-Step, the Tiled Decoder, PP-VAE and Spectral Lifter
panels are hidden because they depend on ACE-Step's VAE. The VST chain, Mastering,
StableStep and the Final Normalizer work with any backend.

### StableStep

StableStep splits a vocal track into vocals and instrumental, refines only the
instrumental through Stable Audio 3, then mixes the vocal back in at the balance the
original had. An instrumental track is refined whole. It needs its own models from the
Model Manager's StableStep tab (GGML about 5.8 GB, or ONNX about 12 GB); until they are
installed the panel says "not installed" and explains what is missing.

| Control | Default | What it does |
|---|---|---|
| Refine strength | 30% | 10% to 60%. How much of the instrumental is re-rendered. Higher re-interprets the instrumentation more. |
| Backend | Auto | Auto, ONNX (TensorRT) or GGML. The ONNX backend builds a TensorRT engine the first time each song-length bucket is used, which is slow once and then cached. |
| Adapters | none | StableStep adapters from `models/sa3-adapters/`, each with its own strength (0 to 200%). Any active adapter switches the refine to the GGML backend. |
| Preserve source dynamics | on | The refined audio follows the original's loudness envelope, so adapters trained on heavily mastered material cannot brickwall the result. |
| Re-encode vocals through PP-VAE | off | Smooths fizzy vocals at the cost of about 5 dB of air above 10 kHz. Off leaves the vocal stem untouched. |
| Vocal level | 0 dB | -6 to +6 dB trim of the vocal against the refined instrumental. 0 keeps the original balance. |
| Source blend | Off | Crossover keeps the original below a frequency (default 250 Hz, width 200 Hz) and the refine above it. Mix blends the two full-band (SA3 amount, default 100%). |
| Follow generation seed | on | The refine uses the song's seed, so the same seed reproduces the same refine. Untick to set a fixed seed. |
| Sampler (advanced) | Native | Refine steps (1 to 64, default 8) and separate solver, schedule and guidance pickers for the refine. These do not change the Generation section's picks. |

### VST chain

The VST Chain panel inside Post-Processing runs VST3 plugins you already own on every
render.

1. Click Add Plugin. The first time, the app scans for installed VST3 plugins. Search by
   name, vendor or category, and click a plugin to add it. The list stays open so you can
   add several; click Done when finished. Rescan picks up newly installed plugins.
2. Use the arrows to reorder the chain. Plugins run top to bottom.
3. Use the link icon to open a plugin's own window and set it up. Its settings are saved
   when you close that window. A green dot marks a plugin whose window is open.
4. The power icon bypasses a plugin without removing it; the bin removes it.

With at least one plugin enabled, "Monitor with VST Chain" plays the track currently loaded
in the player through the chain, pausing the normal player so you do not hear both. While
the monitor runs, a strip replaces the Profiles and VRAM controls at the right of the bar,
with the track name, a seek bar, the time, pause and stop. Changes made in a plugin window
while monitoring are only heard after you restart the monitor; a banner offers the restart.

Presets save the current chain under a name so you can switch between chains. The chain
itself is stored by the server; presets are stored in the browser.
<!-- TODO(verify): VST3 scanning and the monitor on Linux and macOS builds. The route and host binary exist, but only the Windows path has been confirmed here. -->

### Cover art

When on, a 1024 x 1024 cover is generated with FLUX.2-klein-4B after each song finishes,
using the song's subject or lyrics as the brief. The model and its runtime are a one-click
download of about 5.9 GB from this panel; progress and a cancel button show while it
downloads. Cover art only runs while the Post-Processing master switch is on. Auto-Gen can
override the subject for a single song.

## Adapters

The Adapters section loads LoRA and LoKr adapters onto the DiT, sets their strength, and
selects a planner adapter for the LM. It has a Simple mode for one adapter and an Advanced
mode for stacking several, with per-section influence, loading mode, VRAM options and
per-layer group scales. Everything about it is in [Adapters](adapters.md).

## Profiles

Profiles, at the right of the bar, saves every setting in the bar under a name. Profiles
are stored by the server.

- Type a name and click Save. "Include caption and lyrics" (on by default) also stores the
  Custom-Gen content fields: caption, lyrics, negative prompt, instrumental, BPM, duration,
  key, time signature, vocal language and trigger word.
- Apply loads a profile immediately, content included, without reloading the page.
- Click a profile's name to expand a grouped list of every value it holds.
- The row icons rename, overwrite with the current settings, export as JSON and delete.
  Overwrite and delete ask first.
- The import icon in the header loads a profile from a JSON file. A name that already
  exists asks before overwriting.

A profile does not include the VST chain, the LM repetition penalty settings, the planner
adapter, or the knobs other backends add to the bar. Applying a profile while another
backend is active writes the solver, schedule and guidance picks into that backend's
settings, since those are kept per backend.

## VRAM indicator

The figure at the right of the bar is GPU memory used and total, in GB, with a bar that
turns yellow above 70% and red above 90%. It refreshes every five seconds. Hover it to list
the models the engine currently holds in VRAM, each with its size and an unload button.
A model in use cannot be unloaded, and an unloaded model reloads the next time it is
needed. Whatever is not listed is working memory and cannot be unloaded.

## The job queue

Every generation goes into one queue, shown in the Queue section of the activity column on
the right of the window. The engine uses one GPU, so the server runs one job at a time.
Queue as many songs as you like, from any studio; each waits its turn. Post-processing
re-runs from the Library wait in the same line.

- A job captures the settings in force when you queued it. Changing the bar afterwards
  affects only jobs you queue next.
- A job that fails is retried once automatically with a new random seed, unless you
  cancelled it.
- A job that stops reporting progress is treated as stalled and failed: after two minutes
  while sampling steps are counting, or 15 minutes in stages that report nothing until
  they finish, such as model loads and VAE decode. A job that runs
  longer than the Generation Timeout in Settings (default 30 minutes, counted per track from
  when the engine picks it up, not while it waits) is also failed.
- While the engine is starting or paused for training, queued jobs wait for it rather than
  failing.
- YuE2 jobs waiting with matching settings render together as one engine batch when Batch
  Queued Songs is on. See [Backends](backends.md#yue2). The queue boxes those jobs together
  and shows the time the whole batch took, the audio it produced and the ratio of the two;
  each song inside the box shows only its own length. The job that takes the engine lane holds
  it for 5 seconds before rendering, so the first Generate pressed in Lyric Studio and the
  ones queued right after it go into one call; the engine takes at most four songs per batch
  (a compiled limit, not a setting).
- Every finished row shows a ratio next to its length: audio seconds per second of
  generation, so 3.0 means a minute of music took twenty seconds.
- The queue survives a page reload: running jobs reconnect. After a full restart of the app
  the server has forgotten them, so unstarted and interrupted jobs are held and a banner
  offers Resume or Discard. Nothing restarts on its own.
- The Queue section has Retry for failed items, Clear Done for finished ones, and Reset,
  which cancels everything active and pending on the server.

## Task modes

The engine has one generation pipeline and several task modes. The studio picks the mode;
the bar's settings apply to all of them.

| Task mode | What it does | Used by |
|---|---|---|
| text2music | A new song from a caption and lyrics. | [Custom-Gen](studios/create.md), [Auto-Gen](studios/insta-gen.md), [Lyric Studio](studios/lyric-studio.md), [STORM](studios/storm.md), the first section in [Song Builder](studios/song-builder.md) |
| cover | Re-renders source audio through the quantised plan, a free reinterpretation. | [Cover Studio](studios/cover-studio.md) |
| cover-nofsq | Cover from the clean source latents, closer to a faithful remix. | [Cover Studio](studios/cover-studio.md) |
| repaint | Regenerates a time region of existing audio, keeping the rest. | [Repaint](studios/repaint-studio.md) |
| extend | Not a separate engine mode. Song Builder extends or prepends a song with a repaint whose region runs past the start or end. | [Song Builder](studios/song-builder.md) |
| lego | Generates a new instrument track layered over an existing backing track. | [Stem Builder](studios/stem-builder.md) |
| extract | Isolates one instrument track from a mix. | [Stem Separator](studios/stem-studio.md) |
| complete | Fills in a mix around a single stem. | No studio uses it yet. |

HOT-Step skips the LM stage for cover, cover-nofsq, repaint, lego and extract; the caption
and lyrics go to the DiT as written. lego, extract and complete need a base or SFT DiT, not
a turbo model.
<!-- TODO(verify): whether Stem Separator's extract jobs use any Generation-bar settings. They are submitted by the Stem Separator's own route, not the shared generation queue path. -->

## Related

- [Custom-Gen](studios/create.md)
- [Library](studios/library.md)
- [Settings](studios/settings.md)
- [Backends](backends.md)
- [Plugins](plugins.md)
- [Adapters](adapters.md)
- [Getting higher quality output](quality.md)
- [Models](models.md)
