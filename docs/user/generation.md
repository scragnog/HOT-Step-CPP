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

## HeartMuLa controls

Choose HeartMuLa in the engine picker. In Custom-Gen, Tags describes the genre,
mood, instruments and vocal style; Lyrics keeps the song's words. Duration is
an upper limit of 240 seconds. Auto lets the model end the song within that limit.
A duration carried over from another backend is capped at 240 seconds.

Models selects the variant, global language model, local decoder, codec flow and
waveform decoder independently. Adapters selects a compatible HeartMuLa GGUF LoRA;
None uses the base model. Adapter strength ranges from 0 to 2, with 1 as the
default and 0 using base logits. Generation exposes the seed and the backend's
sampling and codec controls. Model selections and sampling settings are remembered
for this backend. Queued requests capture the component choices, adapter and strength.

## The bar at a glance

| Section | What it holds |
|---|---|
| Models | The DiT, LM, VAE decoder and text encoder in use, plus a link to the Model Manager. |
| Adapters | LoRA and LoKr adapters for the DiT and the planner LM. See [Adapters](adapters.md). |
| Generation | Steps, guidance, shift, solver, schedule, guidance mode, seed, batch size and the engine-side audio correction stages. |
| LM / Thinking | The language model stage. The switch in the section header turns the LM on or off. |
| Post-Processing | Everything that runs on the finished audio. The switch in the header is the master switch for the chain. |
| Profiles | A button at the right of the bar. Save, apply, import and export named snapshots of every setting. |

The engine picker is not part of the bar. It sits at the top of the left sidebar, under
the logo, and only appears when more than one backend is installed; see
[Backends](backends.md). GPU memory is shown in the Terminal box, described under
[VRAM indicator](#vram-indicator).

In a window narrower than 1024 pixels, or one taller than it is wide, the bar shrinks to a
menu button, the logo and a Settings button. Settings opens the same sections as a list;
tap a section's header to expand it. Profiles and the VST monitor strip are only
in the full-width bar.

Each section shows a one-line summary of its current state. The Generation summary, for
example, reads like `12s · Euler · Linear · APG 9.0 · σ3.0 · Seed Rnd`: steps, solver,
schedule, guidance mode and scale, shift, and whether the seed is random or fixed.

Hovering a section opens it and moving away closes it. Clicking the section header pins it
open, so you can reach a file picker or another window without it collapsing. A pinned
section closes when you click its header again, click outside it, or press Esc. Most
controls have a small help card: hover the label to read what the control does.

The bar changes shape with the active backend. MiniMax-Music3, YuE2 and Stable Audio 3
show their own model pickers and generation controls, MiniMax-Music3 and YuE2 add their own
planner controls and adapter panels, and a section the backend has no use for says so
instead of showing ACE-Step controls that would do nothing. Everything below
describes the ACE-Step 1.5 layout. For the others, see [Backends](backends.md).

In Create with YuE2 selected, the Caption source picker can use tracks from a linked
training dataset. The tags below it come from those captions; clicking one switches
to Custom and adds or removes that phrase from your style description.

YuE2 Cover Studio uses the same generation queue but starts from an approved ABC
lead sheet. SheetSage2 can transcribe the source recording, or you can supply the
score yourself. The recording is not fed to the audio renderer. The score,
source identity and AR/NAR adapter pair are
captured with the job. See [Cover Studio](studios/cover-studio.md).

## Models

With Stable Audio 3 active, Models lists the installed generation DiTs from the
models folder, including quant names ending in `-imat`. BF16 is the default.
Training files named `sa3-dit-base-*` are excluded. The selection is saved on the
server and captured when a job is queued. If a selected file is removed, the
engine falls back to BF16 and the generation log records the filename it loaded.

| Control | What it does |
|---|---|
| Format | Filter chips (GGUF, SafeTensors, ONNX), shown when models in more than one format are installed. They only filter the lists; a selected model stays selected when its format is hidden. |
| DiT Model | The diffusion transformer that turns the plan into audio latents. The main quality and VRAM decision. |
| Quantization | Shown under DiT Model, LM Model and Text Encoder once a model is installed in more than one quant. The list above picks the model; this one picks its quant. Switching model keeps the same quant when the new model has it. Other backends split a component list the same way when it holds more than one model. |
| LM Model | The 5 Hz language model that writes metadata, the rewritten caption and the audio codes. |
| VAE Decoder | Turns latents into audio. Only shown when more than one VAE is installed. |
| Text Encoder | Only shown when more than one text encoder is installed. |
| Get More Models | Opens the Model Manager. |

YuE2 keeps its standard and legacy VAE choices visible. Missing files or failed metadata
probes disable the affected choice and show the reason beneath it. Download the model
in Model Manager or choose an installed VAE. Generation checks the captured model
selection before planning starts, including files removed while a job was queued.

Any empty slot, or one whose model is no longer installed, is filled with the first
installed model of that type. On first launch, if
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

A small reset arrow appears beside a label whenever its value has moved from the default above; click
it to snap that one field back. Each visible fixed-seed input (Generation Seed, LM Seed, StableStep
refine seed) resets on its own without changing the Random, Use DiT Seed or Follow generation seed
switch beside it; while that switch hides the input, its arrow is hidden too. On a MiniMax-Music3 LM adapter,
the strength dials reset to the selected adapter's own recommended scales rather than the house
defaults, since picking a different adapter can change what "default" means.

### Solver, schedule and guidance

These three pickers are Lua plugins, so the lists grow when a plugin is added and each
entry carries its own description in its help card.

- The solver decides how one step is taken. Solvers are listed by how many network
  evaluations (NFE) each step costs: one-evaluation solvers first, then multi-evaluation
  solvers with their NFE in the name, then adaptive ones that choose their own count. A
  2-NFE solver at 20 steps costs about what a 1-NFE solver costs at 40.
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

Saved seeds are stored by the server, so they are the same in every browser. Each one is
a JSON file in the `seeds/` folder of the data folder's `library/`. The file format is the
one the SeedSaver node from MD_Nodes uses in ComfyUI, so seed files saved there can be
copied into that folder and appear in the list.

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
| Top-K | 0 to 200 | 0 | Limits sampling to the K most likely tokens. 0 turns it off. (YuE2's Composer Top-K is separate: 0 is unlimited there, and unlimited with a temperature above 1.0 is clamped to 1.0.) |
| Top-P | 0 to 1 | 0.92 | Nucleus sampling threshold. |
| Min-P | 0 to 0.5 | 0 | After Top-P, drops every audio code under this fraction of the most likely code's probability. Cuts the unlikely tail without narrowing close choices. 0 turns it off. |
| CFG Plausibility | 0 to 0.5 | 0 | Applies CFG only to codes the conditioned pass gives at least this fraction of its top probability; the rest keep their unguided value. 0 turns it off. |
| CFG Rescale | 0 to 1 | 0 | Pulls the guided distribution's spread back toward the conditioned pass's, so a high CFG Scale changes which codes win without sharpening or flattening everything. 0 turns it off. |
| CFG in Log-Probabilities | toggle | off | Blends the two CFG passes after normalising each to log-probabilities instead of on raw logits. |
| Repetition Penalty | 1.0 to 1.5 | 1.1 | Penalises recently used audio codes to break stuck loops. 1.0 turns it off. |
| Rep. Mode | Presence, Frequency, DRY | Presence | Shown when the penalty is above 1.0. See below. |
| Rep. Window (codes) | 8 to 256 | 64 | How far back the penalty looks. The LM emits 5 codes per second, so 64 codes is 12.8 seconds; the label shows the seconds. |
| DRY Base, DRY Min Match | 1.05 to 4, 2 to 32 | 1.75, 3 | DRY mode only. How fast the penalty grows with match length, and the shortest verbatim repeat it acts on. |
| Negative Prompt | text | `NO USER INPUT` | Negative conditioning text for the LM. |
| LM Codes | Strength 0 to 1, or Step Count | Strength 1.0 | How many DiT steps the LM's audio codes condition. Strength is a fraction of the step budget. With Step Count on, you set an absolute number of steps (default 6) that stays fixed when you change Inference Steps. |
| LM Seed | toggle and number | Use DiT Seed on | With Use DiT Seed on, the LM seed follows the Generation Seed: a fixed Generation Seed makes both deterministic, a random one makes both random. Turn it off to set an independent LM seed, which has its own Seed Manager. |

Min-P and the three CFG refinements are ignored, with a warning in the engine log,
when a draft model is loaded for speculative decoding at batch size 1, the same as
the repetition penalty.

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
whatever their own switches say. The exceptions are Lyric Timestamps (LRC), YuE2's Lyric
Timestamps (Forced Alignment) and Whisper Lyrics, which work with the master switch off.

Post-processing never overwrites the render. It works on a copy, and the song keeps both
the original and the processed file; the Library lets you play either. If the master
switch was off when a song was made, Run Post-Processing in the track's menu runs the
chain on it later, with your current settings and even with the master switch off. It
needs the raw WAV render. A song that has already been processed is refused, so a chain
cannot be stacked twice; Remove Post-Processed Version in the same menu deletes the
processed copy. See [Library](studios/library.md).

### Order of the chain

The stages always run in this order, per track:

1. Quality Evaluator on the unprocessed audio (if its target includes Unmastered).
2. MM3 Latent Refiner, when enabled and its dependencies are installed.
3. Vocal separation, when StableStep or Whisper's "Isolate vocals first" needs it. Runs once
   and is shared.
4. StableStep.
5. PP-VAE Re-encode.
6. Spectral Lifter.
7. Vocal Naturalizer (skipped on instrumentals).
8. Pre-VST Gain Offset.
9. VST Chain.
10. Mastering.
11. Final Normalizer.
12. A peak guard that limits the file to just under full scale if a stage that was meant
    to control peaks failed.
13. Quality Evaluator on the processed audio (if its target includes Mastered).

A stage that fails is logged and skipped; the song is still saved. Cover art runs after
the chain, or alongside it when Parallel Cover Art is on in Settings.

Songs queued from Lyric Studio with Master to Album Track on run with Mastering on,
mastered against a random track from that album's training dataset, and skip the VST
chain. Your own post-processing settings are left as they are. See
[Lyric Studio](studios/lyric-studio.md).

### Stages

| Stage | Default | What it does |
|---|---|---|
| Lyric Timestamps (LRC) | on | Synchronised lyric timing for karaoke-style playback, taken from the model's lyric attention during the render: the DiT on ACE-Step, the planner LM on MiniMax-Music3. Hidden on backends that cannot provide it. |
| Lyric Timestamps (Forced Alignment) | off | YuE2 only. Aligns the finished render against your lyrics for karaoke playback and section markers. Adds a pass per track and needs the YuE2 Lyric Aligner from the Model Manager. |
| Whisper Lyrics | off | Transcribes the sung lyrics with word-level timestamps, using your lyrics as a spelling guide. Model (Auto-detect, Large v3 Turbo, Large v3, Medium, Base), Language, Beam Size (1 to 10, default 5) and "Isolate vocals first". Needs a Whisper model from the Model Manager and the `whisper-cli` executable ([Whisper lyrics finds no executable](troubleshooting.md#whisper-lyrics-finds-no-executable)). |
| Tiled Decoder | off | Replaces the built-in VAE decode with a tiled decode plugin (overlap crossfading, optional dual-pass merge, channel suppression and a small DSP chain). Only shown when a postprocess plugin is installed; its settings come from the plugin. With none picked yet, it picks MD Audio Tiled Decoder V2.1 when that is installed. |
| PP-VAE Re-encode | off | Runs the decoded audio through a second, higher-fidelity autoencoder to clean up fizz and high-frequency noise. Adds roughly 1 to 2 seconds. Original Blend (0 to 1, default 0) mixes the unprocessed audio back in. It runs on GGML and is shown when the PP-VAE model is installed. |
| StableStep | off | Re-renders the instrumental through Stable Audio 3 to replace VAE fizz with real detail. See [StableStep](#stablestep). |
| MM3 Latent Refiner | off | Optional deterministic whole-mix refinement before separation and the other audio stages. Its model pack is currently unlisted. See [MM3 Latent Refiner](#mm3-latent-refiner). |
| Spectral Lifter | off | Removes AI shimmer, reduces spectral noise and can extend the high end. Denoise Strength (default 0.3), Noise Floor (0.1), HF Extension (0), Transient Boost (0), Shimmer Reduction (6 dB, applied to the 10 to 14 kHz band). |
| Vocal Naturalizer | off | Experimental. Five DSP stages that pull back robotic and auto-tune artefacts: vibrato, formant variation, metallic cut (6 to 10 kHz), quantization masking and transition smoothing, with a master Amount (default 0.5). It processes the whole mix, so A/B it against the same render with it off. |
| Pre-VST Gain Offset | 0 dB | A fixed gain (-10 to +10 dB) before the VST chain and mastering. Negative values give a VST chain headroom. When something runs after it, peaks above full scale are passed on for that stage to handle; when nothing does, it limits. |
| VST Chain | empty | Your own VST3 plugins. See [VST chain](#vst-chain). |
| Mastering | off | Matches the level, frequency balance and dynamics of a reference track you upload. "Also use as timbre reference" feeds the same track to the DiT as a [timbre reference](#timbre-reference); ACE-Step only, so the switch is hidden on the other backends. |
| Final Normalizer | off | Always the last stage that changes the audio. Measures integrated loudness (ITU-R BS.1770-4) and sets the gain to hit a target: Spotify / YouTube Music at -14 LUFS (the default), Apple Music / Tidal at -16, EBU R128 (Broadcast) at -23, Club / DJ Playback at -8, or Custom from -30 to -5. A look-ahead limiter holds peaks at the Peak Ceiling (-6 to 0 dBFS, default -1). The resulting true peak is written to the generation log. |
| Cover Art | off | See [Cover art](#cover-art). |
| Quality Evaluator | off | Scores each track from spectral analysis and shows the score as a badge in the Library. Evaluate: Unmastered (default), Mastered or Both. |

When the active backend is not ACE-Step, the Tiled Decoder, PP-VAE and Spectral Lifter
panels are hidden because they depend on ACE-Step's VAE. The VST chain, Mastering and the
Final Normalizer work with any backend. StableStep works with every backend except Stable
Audio 3, whose output is already Stable Audio 3.

### MM3 Latent Refiner

The refiner processes the whole raw render before vocal separation, StableStep, mastering
and VST. It can accept any backend's audio, resampling to 44.1 kHz internally and back to
the source rate. Quality outside MiniMax-Music3 is untested. It is deterministic and has
no seed control.

The panel enables the switch only when the registry reports the complete model pack
installed. The pack is currently unlisted pending the MERT-v1-95M non-commercial licence
decision, conditioning and model publication. Until then, the panel says models are not
installed, and Model Manager offers no public download. See [Model files](models.md).

| Control | Default | What it does |
|---|---|---|
| Enable | off | Runs on new renders when the Post-Processing master switch is on. Run Post-Processing on an existing raw WAV uses the current settings even with the master switch off. |
| Steps | 32 | 1 to 64 deterministic refinement steps. More steps take longer. |
| Window seconds | 30 | Processes overlapping windows. Set 0 for direct mode, one dense sequence for the whole clip. Direct-mode memory grows quadratically with clip length and the engine rejects clips beyond its limit. |
| Overlap seconds | 2 | Nonnegative crossfade duration, strictly shorter than a nonzero window. Reducing the window also reduces overlap when needed. Ignored in direct mode. |
| Refined mix | 100% | Audio-domain dry/wet blend after decode and resampling. 0% is an exact bypass that loads no refiner models; 100% is the full refined output. |

Settings persist locally and round-trip through parameter profiles and JSON presets.
Older settings without refiner keys use the defaults. With the refiner off, no refiner
parameters are sent. Missing models or a failed stage produce a warning in the generation
log and keep the raw audio for the remaining stages.

### StableStep

StableStep splits a vocal track into vocals and instrumental, refines only the
instrumental through Stable Audio 3, then mixes the vocal back in at the balance the
original had. An instrumental track is refined whole. It needs its own models from the
Model Manager's StableStep tab (the GGML set, about 5.8 GB); until they are
installed the panel says "not installed" and explains what is missing.

| Control | Default | What it does |
|---|---|---|
| Refine strength | 30% | 10% to 60%. How much of the instrumental is re-rendered. Higher re-interprets the instrumentation more. |
| Backend | Auto | Auto and GGML both run the GGML backend. Old saved backend choices load as Auto. |
| Adapters | none | StableStep adapters from `models/sa3-adapters/`, each with its own strength (0 to 200%). |
| Preserve source dynamics | on | The refined audio follows the original's loudness envelope, so adapters trained on heavily mastered material cannot brickwall the result. |
| Re-encode vocals through PP-VAE | off | Smooths fizzy vocals at the cost of about 5 dB of air above 10 kHz. Off leaves the vocal stem untouched. |
| Vocal level | 0 dB | -6 to +6 dB trim of the vocal against the refined instrumental. 0 keeps the original balance. |
| Source blend | Off | Crossover keeps the original below a frequency (default 250 Hz, width 200 Hz) and the refine above it. Mix blends the two full-band (SA3 amount, default 100%). |
| Follow generation seed | on | The refine uses the song's seed, so the same seed reproduces the same refine. Turn it off to set a fixed seed. |
| Sampler (advanced) | Native | Refine steps (1 to 64, default 8) and separate solver, schedule and guidance pickers for the refine, with a guidance scale (1 to 10) once a guidance mode is picked. The SA3 Clean button sets a tested combination: Wasserstein Flow v1.2.1, Psychoacoustic, Phase Inversion and 24 steps. These do not change the Generation section's picks. |

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
The monitor plays at the track's own rate (44.1 kHz for some backends, 48 kHz for
ACE-Step). If your sound device cannot switch to that rate, the monitor converts to the
device's rate itself, at the same pitch and speed. When the next track has a different rate,
the plugins are set up again for it. Chains of any length run their plugins in the order
shown, including three or more. If the monitor misbehaves, `vst_host.log` in the newest
`logs/<session>` folder records each start, stop, seek and rate change.

Presets save the current chain under a name so you can switch between chains. The chain
and its presets are both stored by the server, so they are the same in every browser.
Presets an older version saved in this browser move to the server the first time the
panel loads them; one whose name matches a preset already on the server waits for you to
pick Keep both or Replace saved.
<!-- TODO(verify): VST3 scanning and the monitor on Linux and macOS builds. The route and host binary exist, but only the Windows path has been confirmed here. -->

### Cover art

When on, a 1024 x 1024 cover is generated with FLUX.2-klein-4B after each song finishes,
using the song's subject or lyrics as the brief. The model and its runtime are a one-click
download of about 5.9 GB from this panel; progress and a cancel button show while it
downloads. Cover art only runs while the Post-Processing master switch is on. Auto-Gen can
override the subject for a single song.

When both the prompt and subject are blank, the selected Caption LLM writes one short
visual scene for the image. If it is unavailable or slow, cover art uses the existing
lyric keywords instead.

## Adapters

The Adapters section loads LoRA and LoKr adapters onto the DiT, sets their strength, and
selects a planner adapter for the LM. It has a Simple mode for one adapter and an Advanced
mode for stacking several, with per-section influence, loading mode, VRAM options and
per-layer group scales. Everything about it is in [Adapters](adapters.md).

The ACE-Step Adapters section has a **Runtime DoRA magnitudes** switch. It defaults
to off for compatibility with older runtime renders and is shared with MiniMax-Music3
requests. See [DoRA loading](adapters.md#file-formats) for its effect on runtime memory.

## Profiles

Profiles, at the right of the bar, saves every setting in the bar under a name. Profiles
are stored by the server, separately for each engine: the list, save, rename and delete
work on the active engine's profiles only, and the header names that engine. A profile
saved on MiniMax-Music3, YuE2 or Stable Audio 3 leaves out ACE-Step's models, adapters, DiT
sampling and LM settings. Profiles saved before this change belong to ACE-Step.

- Type a name and click Save. "Include caption and lyrics" (on by default) also stores the
  Custom-Gen content fields: caption, lyrics, negative prompt, instrumental, BPM, duration,
  key, time signature, vocal language, trigger word, and the Beat I/O switch with its bar
  count.
- Apply loads a profile immediately, content included, without reloading the page.
  Older profiles that name removed ONNX VAE or StableStep choices load the current defaults.
- Click a profile's name to expand a grouped list of every value it holds.
- The row icons rename, overwrite with the current settings, export as JSON and delete.
  Overwrite and delete ask first.
- The import icon in the header loads a profile from a JSON file. A name that already
  exists asks before overwriting.

A profile does not include the VST chain, the LM repetition penalty settings or the planner
adapter. Knobs a backend adds to the bar (`backendParams`) are saved too and merged
over the live ones on apply. Importable YuE2 presets live in
[presets/yue2-plan-play](presets/yue2-plan-play/README.md). Applying a profile while another
backend is active writes the solver, schedule and guidance picks into that backend's
settings, since those are kept per backend.

## VRAM indicator

The VRAM badge at the top of the Terminal box, in the right-hand column, is GPU memory used
and total, in GB, with a bar that turns yellow above 70% and red above 90%. It refreshes about every 1.5 seconds. Hover it to list
the models the engine currently holds in VRAM, each with its size and an unload button.
A model in use cannot be unloaded, and an unloaded model reloads the next time it is
needed. Whatever is not listed is working memory and cannot be unloaded.

## The job queue

Every generation goes into one queue, shown in the Queue section of the activity column on
the right of the window. The engine uses one GPU, so the server runs one job at a time. (An experimental Multi-GPU
setting in Settings starts one engine per card; with two or more cards ticked, jobs run in parallel, one per card.)
Queue as many songs as you like, from any studio; each waits its turn. Post-processing
re-runs from the Library wait in the same line.

- A job captures the settings in force when you queued it. Changing the bar afterwards
  affects only jobs you queue next.
- A job that fails is retried once automatically with a new random seed, unless you
  cancelled it.
- A job that stops reporting progress is treated as stalled and failed. Stages that report
  nothing until they finish, such as model loads and VAE decode, get 15 minutes. While
  sampling steps are counting, two minutes of silence makes the server ask the engine; if
  the engine says the job is still running, it gets the 15-minute window too. A YuE2 NAR
  chunk gets longer in proportion to its length, and YuE2's decode and output stages are
  only held to the Generation Timeout. A job that runs longer than the Generation Timeout
  in Settings (default 30 minutes, counted per track from when the engine picks it up, not
  while it waits) is also failed. A render the engine has already finished is kept, even
  when it lands at the limit.
- While the engine is starting or paused for training, queued jobs wait for it rather than
  failing. A song queued for one engine waits, and says so in its row, while another
  engine is active; the queue never switches engines by itself.
- YuE2 jobs waiting with matching settings render together as one engine batch when Batch
  Queued Songs is on. See [Backends](backends.md#yue2). The queue boxes those jobs together
  and shows the time the whole batch took, the audio it produced and the ratio of the two;
  each song inside the box shows only its own length. The queue waits 5 seconds after your
  last Generate click before sending an album's songs, restarted by each further click, so
  the first track you queue in Lyric Studio goes up with the ones queued right after it
  rather than starting alone; the engine takes at most four songs per batch (a compiled
  limit, not a setting), and songs that have been waiting longer than that are sent at once.
- Every finished row shows a ratio next to its length: audio seconds per second of
  generation, so 3.0 means a minute of music took twenty seconds.
- The queue survives a page reload and a full restart of the app, because the server keeps
  it in its database. Songs that had not been sent to the engine yet carry on once it is
  ready. A song that was rendering when the app stopped is marked interrupted, with a note
  to check the Library, and only Retry renders it again. A browser-owned queue (see
  below) reconnects running jobs after a page reload instead; after a full restart it
  holds unstarted and interrupted songs and a banner offers Resume or Discard.
- The Queue section has Retry for failed items, Clear Done for finished ones, and Reset,
  which cancels everything active and pending on the server. Send To Playlist (off by
  default) adds each finished song to the playlist.
- The queue itself lives either in your browser or on the server; a fresh install is
  server-owned from the start. An older browser-owned queue with pending work moves to
  the server automatically the next time you load the app, keeping each item's state
  (queued work keeps running, nothing needs a manual Resume); finished history stays
  where it is. A versioned backup is kept regardless, and nothing is deleted. You only
  see a banner about this if the move gets stuck, with Retry and Export and roll back
  buttons. Once a move has started, the browser queue stays paused — it does not pick
  back up on its own — until one of those two resolves it, so nothing can run twice.
  The same banner appears if a roll back fails, for example because the server was
  still submitting an item: the server queue stays paused and the browser queue stays
  paused until you retry the roll back or move the queue to the server again.

An external agent (Claude Code, Codex) can also submit and poll generations through the
`hotstep` MCP server, on ACE-Step 1.5, MiniMax-Music3 and YuE2. Stable Audio 3 is not
covered. See [Driving HOT-Step with an agent](mcp.md).

### Progress badge and done notifications

While jobs run, the Create entry in the sidebar (the menu drawer on a narrow screen) shows a badge with the running count and the active job's percent, and the tab title reads like `(2) 43% · HOT-Step`. The first time you click Generate the browser asks whether HOT-Step may show notifications; if you allow it, a notification appears when a job finishes while the tab is in the background. Clicking it brings the tab forward and opens the song.

## Task modes

The engine has one generation pipeline and several task modes. The studio picks the mode;
the bar's settings apply to all of them except extract, described below the table.

| Task mode | What it does | Used by |
|---|---|---|
| text2music | A new song from a caption and lyrics. | [Custom-Gen](studios/create.md), [Auto-Gen](studios/insta-gen.md), [Lyric Studio](studios/lyric-studio.md), [STORM](studios/storm.md), the first section in [Song Builder](studios/song-builder.md) |
| cover | ACE-Step re-renders source audio through the quantised plan. | [Cover Studio](studios/cover-studio.md) |
| cover-nofsq | ACE-Step covers from clean source latents. | [Cover Studio](studios/cover-studio.md) |
| repaint | Regenerates a time region of existing audio, keeping the rest. | [Repaint](studios/repaint-studio.md) |
| extend | Not a separate engine mode. Song Builder extends or prepends a song with a repaint whose region runs past the start or end. | [Song Builder](studios/song-builder.md) |
| lego | Generates a new instrument track layered over an existing backing track. | [Stem Builder](studios/stem-builder.md) |
| extract | Isolates one instrument track from a mix. | [Stem Separator](studios/stem-studio.md) |
| complete | Fills in a mix around a single stem. | No studio uses it yet. |

These task names describe ACE-Step engine modes. YuE2 Cover Studio submits an ordinary
YuE2 generation with `yue2Cover` source identity and an approved `yue2Abc` score;
the AR composer follows that score instead of planning a new one. It does not run
ACE-Step's `cover` or `cover-nofsq` mode. ACE-Step skips the LM stage for cover,
cover-nofsq, repaint, lego and extract; the caption and lyrics go to the DiT as
written. lego, extract and complete need a base or SFT DiT, not a turbo model.
Stem Separator's extract jobs run outside the shared queue, through the Stem Separator's
own route. They use the studio's own model picker with adapters off and a random seed;
the bar's steps, solver, schedule and guidance settings do not reach them, so the engine
uses its defaults.

Returning to a studio does not submit another generation. The queue keeps its
existing order and ownership rules while status views share current results.
Leaving a status view stops that view's observation; use the job's explicit
cancel control to stop server work. A completed server-queued render reaches
the Library once, including when both the queue and its submitting view follow it.

## Related

- [Custom-Gen](studios/create.md)
- [Library](studios/library.md)
- [Settings](studios/settings.md)
- [Backends](backends.md)
- [Plugins](plugins.md)
- [Adapters](adapters.md)
- [Getting higher quality output](quality.md)
- [Models](models.md)
- [Driving HOT-Step with an agent (MCP)](mcp.md)

## Release data paths

Release audio and references live under the shared data folder's `library/`. The `/audio` and `/references` URLs keep their existing meanings after a move or repoint. See [Data folder and upgrades](data-folder.md).
