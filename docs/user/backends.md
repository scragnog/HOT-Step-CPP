# Generation backends

HOT-Step CPP registers five music models, which the app calls backends: ACE-Step 1.5,
MiniMax-Music3, YuE2, Stable Audio 3 and HeartMuLa. HeartMuLa is registered for
integration testing; its native engine routes are still pending. Backends run inside the same C++/GGML engine
process, so switching between them changes which model's weights are loaded, not which
program is running. Each one turns a text description into stereo audio, but they
build it differently, accept different inputs, and work with different parts of the app.
ACE-Step 1.5, MiniMax-Music3 and YuE2 sing the lyrics you give them; Stable Audio 3
renders from the caption alone.
Their capability manifests also declare which Create and Lyric Studio caption source
picker applies (`none`, MM3 tracks, or YuE2 dataset).
ACE-Step 1.5 is the default and the only backend that supports every studio.
YuE2 also supports Cover Studio, through a reviewed lead sheet rather than audio
conditioning. MiniMax-Music3 and Stable Audio 3 are text-to-music only.

For which quant and settings give the best output on each backend, see
[Getting higher quality output](quality.md), which has a per-backend cheat sheet. This
page covers what each backend is and what it can do.

## Switching backends

The engine picker is at the top of the left sidebar, directly under the logo: a pink box
headed "Generation engine" with the active backend's name. Click it and choose another
backend from the list. The picker appears whenever two or more backends are registered with
the server, and is hidden when there is only one.

- The choice is stored on the server, so it survives a restart. A new install starts on
  ACE-Step 1.5.
- Switching tells the outgoing backend to release its models from VRAM, in the
  background, so two model families do not sit on the GPU together. Every generation
  waits until the other backends confirm their models are released, then loads its own.
- With the experimental Multi-GPU setting on, each card runs its own engine and every model must fit
  in one card. With two or more cards ticked, jobs run in parallel, one per card; training, post-processing and previews still take every card.
- The global bar reshapes to the active backend. The Models, Adapters, Generation and
  LM / Thinking clusters show that backend's own controls, and controls that do nothing
  on it are hidden.
- The sidebar lists only the studios the active backend can use. Cover Studio is listed
  under ACE-Step and YuE2; Repaint, Stem Builder and Song Builder under ACE-Step
  only; STORM under ACE-Step and YuE2. The rest, Training included, are always listed.
  If the open studio is not listed for the backend you switch to, the app moves to
  Custom-Gen.
- If the active backend's model files are not installed, the bar says so and shows a
  Get models button that opens the [Model Manager](studios/model-manager.md).

On YuE2, hovering the picker shows the model's licence notice (see [YuE2](#yue2)).

## ACE-Step 1.5

The original model the app was built around, and the one HOT-Step's engine started as a
port of. A generation runs in three stages:

1. A planner language model (0.6B, 1.7B or 4B) fills in whatever you left blank (BPM,
   key, time signature, duration, language, an enriched caption) and writes a plan of
   audio codes. You can switch this stage off from the LM / Thinking cluster.
2. A diffusion transformer (DiT) renders the audio latents from the plan, the caption
   and the lyrics. It comes in Standard and XL sizes, each in several variants.
3. A VAE decoder turns the latents into 48 kHz stereo audio.

A small text encoder model conditions the DiT on the caption.

What sets it apart from the others:

- It is the only backend with source-audio-conditioned covers, repaint
  (Repaint) and stem generation (Stem Builder). It also supports continuous streaming in STORM.
- BPM, key, time signature and duration are real input fields. Duration is a target the
  planner aims for, from 10 to 600 seconds.
- It runs the full set of Lua [plugins](plugins.md): solvers, schedulers, guidance modes
  and the postprocess plugins that replace the VAE decode.
- It takes DiT adapters (LoRA and LoKr) in the Adapters cluster, with merge and runtime
  modes, per-group strengths and per-section masking, plus a planner LM adapter in the
  same cluster. See [Adapters](adapters.md).
- **Batch Size** renders up to 9 songs per request, one after another.
- Song Builder extends a song section by section, by repainting past either end.
- Lyric timestamps (LRC) come from the DiT's own attention during the render.

Models: a DiT, a planner LM, the text encoder and a VAE. The Model Manager's ACE-Step
packs bundle these (the packs also include the CUDA runtime files):

| Pack | Contents | Download |
|---|---|---|
| Quick Start | Standard Turbo DiT Q8_0, 4B LM Q8_0 | ~9.7 GB |
| Minimal Setup | Standard Turbo DiT Q4_K_M, 1.7B LM Q8_0 | ~5.8 GB |
| XL Quality | XL Turbo DiT Q8_0, 4B LM Q8_0 (needs about 12 GB VRAM) | ~12.4 GB |
| Blackwell Optimized | XL Turbo DiT MXFP4 for RTX 50-series cards, 4B LM Q8_0 | ~9.8 GB |

Every pack adds the text encoder, the standard VAE, ScragVAE and the PP-VAE. The full
list of files is on the [Models](models.md) page.

## MiniMax-Music3

A native C++/GGML port of MiniMax-Music3. Its pipeline has five model components, run in three steps:

1. An 8B planner language model writes the song frame by frame (25 frames per second)
   and decides where it ends by emitting a stop token.
2. A depth decoder adds the acoustic detail for each planned frame.
3. A condition encoder, a flow-matching DiT and a vocoder render the plan to 44.1 kHz
   stereo in overlapping windows, which are stitched together.

What sets it apart:

- The caption is a Structured Caption with three sections in order: Global Metadata
  (genre, tempo, key, mood, production), Vocal Details (or the lead instrument for an
  instrumental) and Arrangement (a section-by-section timeline). Tempo and key live in
  that prose; there are no BPM or key fields. Lyrics stay out of the caption, and the
  caption and lyrics share a 5,000-token budget. The Compose Caption button under the
  caption box turns a plain-English description into a Structured Caption with no AI
  model involved.
- The song decides its own length, up to 300 seconds, and there is no duration
  control.
- Play While Rendering (Generation cluster, off by default) starts playback a few
  seconds in, while later windows are still being rendered. The full file is saved as
  usual.
- Variations Per Render (1 to 4) renders several different songs from one prompt in a
  single pass, sharing the planner. Each request is otherwise one take.
- Flow Steps (default 30, range 2 to 60) is the main speed control. Below 30, Low-Step
  Compensation reshapes the schedule so short renders keep their low end and stereo
  image.
- The Lua solver, scheduler and guidance plugins can drive its flow DiT, behind the
  Sampler Plugins toggle, which is off by default and marked experimental. Postprocess
  plugins do not apply.
- Planner LM adapters (LoRA or LoKr) load from the LM Adapter picker in the Adapters
  cluster, with Strength, Attention and MLP dials. They are applied per generation, as
  live deltas in Runtime mode or merged into the weights in Merge mode. **Add trigger
  to caption** (on by default) puts the adapter's trigger word in the caption for you.
  Flow-DiT adapters have no control in the app.
- Lyric timestamps (LRC) come from the planner's attention, at line level.
- The planner's output can be saved and replayed, so you can re-render a plan with
  different flow settings without planning again. By default, with a fixed seed,
  changing only flow-stage settings reuses the last plan.
- Renderer (in the Models cluster, under the Flow DiT picker) switches the DiT between
  GGML and TensorRT. TensorRT needs the MM3 TensorRT DiT pack plus one builder DLL for
  your GPU, and the first render builds an engine once, which takes a few minutes.

Models: five GGUFs (planner LM, depth decoder, condition encoder, flow DiT, vocoder), each
selectable at its own quantisation in the Models cluster. Picking a quant loads it into
the engine and the choice is remembered across restarts. The packs:

| Pack | Precision | Download |
|---|---|---|
| MiniMax-Music3 (Q4_K_M) | LM and DiT Q4_K_M, depth Q8_0 | ~7.6 GB |
| MiniMax-Music3 (NVFP4) | LM and DiT NVFP4, for RTX 50-series cards | ~7.7 GB |
| MiniMax-Music3 (Q6_K) | LM and DiT Q6_K, depth Q8_0 | ~9.9 GB |
| MiniMax-Music3 (Balanced mix) | LM Q8_0, DiT Q4_K_M, depth Q8_0 | ~12.2 GB |
| MiniMax-Music3 (Q8_0) | Q8_0 throughout, recommended | ~13.4 GB |
| MiniMax-Music3 (F16) | Full precision | ~23.6 GB |

Training an adapter also needs the MiniMax-Music3 Training pack (~2.8 GB) on top of a
generation pack. See [Training MiniMax-Music3 adapters](training/minimax-music3.md).

The model's licence requires the name "MiniMax-Music3" to be shown as-is, which is why
the picker never shortens it.

## YuE2

A native C++/GGML port of YuE2. Its language model has two halves that share no weights:

1. The AR (autoregressive) half composes. It first writes a lead sheet in ABC notation
   (structure, chords, melody), then the song itself as codec frames, 25 per second,
   and decides where the song ends.
2. The NAR half renders those frames into audio latents with a flow-matching solver.

A VAE decoder then produces 48 kHz stereo. Every GGUF generation also applies a small
companion decoder adapter to the NAR half, which ships in each YuE2 pack.

What sets it apart:

- The prompt is freeform: a plain style description plus lyrics. BPM and key are folded
  into the style sentence rather than sent as fields. The language comes from the
  lyrics.
- Include BPM/key in caption (on by default) controls that tail. Turn it off for an
  adapter trained without BPM and key: Song Info BPM and key are left out, and a
  "<bpm> BPM, key of <key>." tail already on the caption (from a dataset caption) is
  removed. Training captions are unchanged.
- The song decides its own length, up to six minutes, and there is no duration control.
- Chain of Thought (Generation cluster) sets how much the planner writes before audio:
  Full (the default), Melody only, or Off. Off skips the lead sheet but runs guidance,
  which makes it the slowest mode, not the fastest.
- Preview the score first (off by default) plans the lead sheet only, shows it with
  playback, and lets you continue, re-plan with a new seed, or cancel before any audio
  is rendered. You can also paste your own ABC lead sheet to render from.
- Lyric Timing (Lead sheet section) ties each sung lyric block to its score section's
  first Vocal note, once a score and lyrics are supplied. Off is the default. Soft
  nudges the composer toward that timing. Hard hides a section's lyrics, and optionally
  its score lines, from the composer until then. It needs CFG 1 — the engine's own
  default for Full and Melody only — so an explicit CFG elsewhere skips it with a note
  and renders at the requested CFG instead; a score it can't build a schedule from
  (unlabelled sections, no tempo, no matching timed lyric block) also renders without
  it, and the generation log says why.
- Cover Studio takes an uploaded or Library recording and uses SheetSage2 to
  transcribe its melody and chords into ABC. Review and approve the full score before rendering.
  You can supply ABC yourself without installing SheetSage2. The YuE2 **Models**
  dropdown picks the transcriber: SheetSage2, or SheetSage2-Refined for odd meters,
  slash and tension chords and, optionally, modal keys. When the vocal has
  no melody SheetSage2 can hear (rap, screamed vocals), Cover Studio separates
  the vocal and writes its rhythm on the Vocal line as one repeated note, keeping
  the chords and the instrumental line. Edit the notes if you want a tune. The
  rhythm follows the source's density and sections, not every syllable exactly.
  The recording does
  not condition the audio renderer: YuE2 composes from the approved score,
  caption and lyrics. By default, rendering keeps only the vocal melody,
  removes chord symbols and the score tempo, and keeps the source key. You can
  retain the instrumental line or chords, keep or set tempo, and transpose the
  score within six semitones. Advanced **Condition strength** tightens style,
  lyrics and score together. Cover jobs store the source identity, reviewed
  full score, all five render choices and ABC actually rendered.
  Choose the base model or an explicit AR/NAR adapter pair for that job.
- Instrumental follows the upstream YuE2 instrumental workflow. Whenever the app holds
  the score before rendering (a cover, a previewed or pasted lead sheet, or the plan
  picked by re-planning), it moves the Vocal melody onto the Ins voice. Where both
  voices sound at once, the melody wins and Ins keeps the rest of its notes; chord
  symbols stay on Vocal. On the base model the style also gains "Instrumental" and
  "no vocals, no singing, no choir, no spoken words"; an adapter's style is left as
  trained. With **Re-plan runaway scores** off and no score supplied, an instrumental
  still plans its score once before rendering, so the melody can move. That one plan
  renders whatever it is, with no redraw. With Chain of Thought off there is no score
  to move, so the melody may still come out as wordless vocals.
- Without the preview, the app plans first anyway and redraws the seed when the lead sheet
  is a runaway, has no vocal line, or hit its cap, up to **Re-plan tries** (3 by default). A
  single unnamed ABC voice, such as `V:1`, counts as the vocal melody; in a multi-voice
  score, an unnamed voice with `w:` lyric lines counts as vocal too. With
  **Re-plan flagged scores** (on by default) it also redraws a plan that passes but is
  illegible: one chord for the whole song, a riff looped for most of it, a vocal melody on
  one or two pitches, or far fewer sections than the lyric. If no attempt is clean, the
  least-flagged plan renders and the song's Plan facts and the generation log say so (a
  sign the adapter is not writing good plans); if every attempt is broken outright, the last
  one renders. With a joint-trained adapter picked, plans are judged against the adapter's own
  training sheets: a check that at least half of them fail is the style, not a fault, and is
  skipped. A rap album's sheets carry almost no vocal melody (the transcriber writes sung
  melody), so "no vocal line" is not held against its plans. The run keeps the result in
  `style-norms.json` in its folder; checkpoint previews use the same file. Every song of a
  batch plans in the same pass. The first pass draws one plan
  per song; if any song needs another try, later passes draw up to 8 plans at once, shared
  between the songs still looking, in roughly the time of two single plans. The first clean
  attempt wins, even when later ones in the same pass are clean as well. Each redraw's seed
  follows from the song's seed, so the same seed and settings replay the same attempts and
  render the same song, whichever VAE decodes it. Generate songs one at a time for that: songs
  sent together render as one batch, which can change them slightly. The song keeps the
  winning attempt's seed, so to repeat a song exactly, reuse the seed you typed. A song whose plan
  is settled shows "Plan ready, waiting for N other songs"; one that has finished composing
  while the rest of its batch is still going shows "Composed, waiting for N other songs".
- The re-plan and recompose knobs live in the collapsed **Re-planning** section of the
  Generation panel.
- Recompose runaway songs (2 by default) handles a good lead sheet composed into a
  runaway. A song counts as one once its composition passes 1.5 times the lead sheet's own
  length plus 20 seconds, or six minutes. The engine composes it again with new seeds, up to
  4 at a time, and keeps the first attempt that ends properly. All of it happens before any
  audio is rendered. Songs of a batch that have finished composing stop costing GPU time
  while the others carry on.
- Batch Size renders up to 4 songs in one pass, and Noise Variations renders the same
  composed song from up to 9 different noise seeds.
- Batch Queued Songs (on by default) does the same across the queue: when several YuE2
  jobs are waiting with the same settings and adapters, the one that reaches the engine
  takes the others with it and composes them together, up to that same limit. An album
  queued from Lyric Studio is the typical case. Songs per minute go up; each song finishes
  when its batch does rather than as soon as its own turn would have. A job that needs a
  different adapter, guidance or sampler setting waits for its own turn. A batch that
  fails hands its passengers back to the queue to render alone. Removing one song of a
  running batch from the queue (its X) takes only that song out: the engine stops planning
  or composing it, never recomposes it, leaves it out of the render, and the rest of the
  batch carries on. That holds for the song whose job started the batch too. Batched decoding rounds
  differently from solo decoding, so a seed rendered in a batch can differ from the same
  seed rendered alone; the Batch Size setting has always had the same property.
- With Keep Models in VRAM on (Settings, Environment tab), a composed song renders on a
  second engine lane while the next queued YuE2 song starts composing. The output is
  identical to a solo render. The two lanes share one GPU, so composing slows while a
  render runs and the measured gain on a pair of songs is a few percent, not a doubling.
  It needs about 4 GB of free VRAM at the handoff; below that, or with ConvRot or a Lua
  NAR plugin, the song renders inline as before.
  After the render, songs whose post-processing is CPU only (a VST chain, mastering, the
  normaliser) finish at once and side by side; only StableStep, Whisper, the aligner and
  cover art wait for the GPU.
- It has its own NAR solver and scheduler choices (Midpoint, Wasserstein Flow, Ancestral
  Euler, Latent Channel EQ or Timestep Lie; Uniform, HT V3, Plan-Aware, Plan-Loose or Plan
  Overdrive). Their tuning knobs sit in a collapsed "Controls" section under the picker.
  Apart from Midpoint and Uniform, each choice runs as a Lua plugin: HT V3 is the shared
  scheduler of that name, and the rest are YuE2-only. The shared plugin pickers are not
  offered.
- Adapters load into two slots in the Adapters cluster, one for the AR half and one for
  the NAR half, each with its own strengths. An adapter is merged into the loaded model:
  a pick saves at once, a dial saves when you let go of it, and the next generation
  reloads the model with the change. The app writes the adapter's trigger word into the
  style prompt.
  Single-file adapters from ComfyUI or ai-toolkit load through **Import adapter…**, which
  splits them into the two halves.
- An adapter trained on another YuE2 variant (a base-model adapter on the YuE2-RS
  fine-tune, say) is re-based onto the loaded LM, as ACE-Step's Basin Re-base does: the
  picker tags it with the variant it was trained on, and **Re-base strength (β)**,
  default 0.75, sets how far the layers it targets move toward that model before it
  merges. 1 runs them as that model, 0 applies the adapter to the loaded LM unchanged.
  That variant's LM has to be installed, in any quant; the loaded quant is used when
  present. Without it the adapter merges unchanged. An adapter that records no training
  LM counts as trained on the base model.
- Lyric timestamps come from a forced aligner run after the render, switched on per
  render in the Post-Processing cluster. It needs the `mms-fa-f32.gguf` model.

### What changes what on YuE2

The composer supplies YuE2's song plan. Composer seed, sampling settings, composer CFG, style words and the ABC sketch can change that plan. The NAR renderer turns it into audio. Solver, scheduler, guidance and noise settings affect the rendering. With the experimental plan controls from #269, Plan Strength, Plan CFG, CADS and plan-aware scheduler plugins can also change how closely the rendering follows the plan; their effects are not limited to clarity or loudness.

Dials that change the song plan:

- An ABC melody sketch (pasted lead sheet).
- Style words in the prompt.
- Seed, Composer Temperature, Top-K, Top-P, Min-P and repetition penalty.
- Composer CFG, and its refinements in the Composer section: CFG Plausibility (guide only plausible frames), CFG in Log-Probabilities and CFG Rescale. All three only act when CFG is not 1, which means Chain of Thought off or a set CFG Scale.
- In Cover Studio, Follow source: the source recording's own semantic codes steer the composer ([Cover Studio](studios/cover-studio.md)).

Dials that change how closely the rendering follows the plan (experimental):

- Plan Strength.
- Plan CFG.
- CADS.
- Plan-aware schedulers.

Listening observations (not guarantees):

- Top-K 0 (unlimited) with Composer Temperature above 1.0 breaks the render, so the server clamps the temperature to 1.0 in that case and adds a warning to the job notes. The Composer Top-K slider now reaches 0 for unlimited.
- Temperature works in both directions: around 0.7 and around 1.3 are both usable.
- Higher CFG follows the style more closely but costs some high-end air.
- Words like "choir" or "reverb" in the style add heavy reverb.

Models: a language model GGUF and a VAE GGUF, selected separately in the Models cluster.
The packs:

| Pack | LM precision | Download |
|---|---|---|
| YuE2 Tiny (imatrix) | Q2_K-imat, audibly degraded | ~2.4 GB |
| YuE2 Compact (imatrix) | Q4_K_M-imat, for limited VRAM | ~3.1 GB |
| YuE2 Recommended | Q8_0 | ~4.6 GB |
| YuE2 Full Precision | BF16 | ~7.9 GB |
| YuE2 INT8 ConvRot (CUDA) | ConvRot checkpoint plus the Q8_0 GGUF; needs an NVIDIA GPU with BF16 support (Ampere or newer) | ~8.5 GB |

Training packs (Artist (AR) Training, Joint Training, Minted Regulariser) are separate.
See [Training YuE2 adapters](training/yue2.md).

YuE2's weights are licensed CC BY-NC 4.0. The upstream authors have clarified that
individual creators, musicians and researchers may use the model and its outputs
freely, including commercially, and that only companies need a commercial licence
(contact gezhang@umich.edu). Adapters trained on YuE2 carry the same terms.

## Stable Audio 3

Stable Audio 3 Medium, the model behind the StableStep refine, as a generation backend
of its own. A T5Gemma text encoder reads the caption, the DiT denoises from pure noise
in one pass, and the SAME decoder turns the result into 44.1 kHz stereo.

What sets it apart:

- The caption is the whole prompt, cut at 256 tokens. Stable Audio 3 ignores the
  lyrics, BPM, key and time signature fields.
- Duration is the exact length of the output, up to 600 seconds. Left on Auto, a
  render is 30 seconds long. Each request renders one track.
- **Steps** (default 8, range 1 to 64) and **Sampler** sit in the Generation cluster.
  The default sampler is MD Pingpong Blend, a Lua solver that runs without the
  Sampler Plugins toggle. You can select native PingPong or Euler instead, or turn on
  **Sampler Plugins** (off by default) and pick a solver, which replaces the default.
  Full-loop solvers are not offered, and guidance plugins do little because the model
  has no unconditional prediction.
- **Loudness Normalize** (off by default) brings the output to **LUFS Target**
  (default -14, range -30 to -6) with a -1 dBFS peak ceiling, before post-processing.
  Without it the output level can swing by about 18 dB between captions.
- The VST chain, mastering and Whisper run on its output. StableStep, PP-VAE and
  Spectral Lifter do not, and there are no lyric timestamps.
- The generation bar has no adapter picker for it. To render with an adapter, use the
  audition in Training Studio's SA3 card.

Seed breeding, set through the API, only runs when a seed B is set. If you set seed B
and leave the blend empty, the blend defaults to 0.5 (an even mix of the two seeds' noise).
A blend of 0 turns breeding off; 1 uses seed B's noise throughout sampling.
Both the native Pingpong and MD Pingpong Blend paths mix the second noise
stream at every re-noise step. StableStep refinement does not use seed breeding.
This differs from ACE-Step, where a shared
seed B with no blend sends no blend at all.

STORM has Stable Audio 3 slot and Directed Drift code, but the Stable Audio 3 backend
reports no streaming support, so with it active the sidebar hides STORM and opening
`/storm` moves the app to Custom-Gen. See [STORM](studios/storm.md).

Models: the StableStep files in the Model Manager, which are a generation DiT
(`sa3-dit-*.gguf`), the T5Gemma text encoder, the SAME encoder and decoder, and the
tokenizer files. The **Flow DiT** picker in the Models cluster lists the installed
generation DiTs, with BF16 as the default; see [Generating a song](generation.md#models).
The Stable Audio 3 Training pack (~9.3 GB) installs all of them along with the
training base.

Training Studio trains Stable Audio 3 Medium DoRA-rows adapters. With Stable Audio 3
active, the studio trains it directly. With another backend active, choose
**Stable Audio 3 Medium · DoRA-rows** in the **Training target** picker above the phase
tabs. The choice drives every tab and is locked while a job runs. SA3 shows only
Dataset and Train tabs. Preparation strips vocals with SuperSep level 5 and keeps the
originals. Training supports CUDA and CPU; the card lists missing model files. See
[Stable Audio 3 training](training/sa3.md) for the recipe, cancellation, saved-run
resume and fixed-seed adapter auditions.

## Capability matrix

"Yes" and "No" come from each backend's capability manifest in the server and the
controls the UI shows for it. "?" means not yet confirmed.

| | ACE-Step 1.5 | MiniMax-Music3 | YuE2 | Stable Audio 3 |
|---|---|---|---|---|
| Text to music | Yes | Yes | Yes | Yes |
| Lyrics | Yes | Yes | Yes | No |
| Caption format | Free text, plus BPM, key and time signature fields | Structured Caption | Free text; BPM and key folded into the style | Free text, up to 256 tokens |
| Duration | Set by you, 10 to 600 s | Model-ended, ceiling 300 s | Model-ended, up to 360 s, no control | Set by you, up to 600 s; Auto gives 30 s |
| Songs per request | Up to 9 | 1, or up to 4 with Variations Per Render | Up to 4 with Batch Size | 1 |
| Cover Studio | Yes, source-audio conditioning | No | Yes, reviewed ABC composition conditioning | No |
| Repaint | Yes (Repaint) | No | No | No |
| Extend | Yes, in Song Builder | No | No | No |
| Adapters at generation | DiT LoRA and LoKr, planner LM LoRA and LoKr | Planner LM LoRA and LoKr, runtime or merge, applied per render | AR and NAR LoRA and LoKr, merged at load; ComfyUI joint files via import | DoRA-rows, in Training Studio auditions only |
| Training in Training Studio | Yes | Yes | Yes | Yes |
| Streaming playback | Continuous streaming in STORM; no per-render preview | Play While Rendering | Continuous streaming in STORM | STORM slots, with no sidebar entry |
| STORM Directed Drift | Caption, seed, guidance, LSS, steps, BPM and duration | No | Caption, seed, CFG scale, ODE steps and NAR cache ratio | Caption, seed, guidance, steps and duration |
| Lua plugins (solvers, schedulers, guidance) | Yes, plus postprocess plugins | Yes, opt-in and experimental; no postprocess plugins | No shared pickers; its own NAR solver and scheduler choices | Yes, opt-in; MD Pingpong Blend runs by default |
| Lyric timestamps (LRC) | Yes, from the DiT | Yes, from the planner | Yes, forced alignment after the render | No |
| Post-processing (VST chain, mastering, StableStep, Whisper) | Yes | Yes | Yes | Yes, except StableStep |
| PP-VAE and Spectral Lifter | Yes | No | No | No |
| Output | 48 kHz stereo | 44.1 kHz stereo | 48 kHz stereo | 44.1 kHz stereo |
| Typical VRAM | ? | ? | ? | ? |
| Typical time per song | ? | ? | ? | ? |

<!-- TODO(verify): typical VRAM per backend at the recommended pack. The only figure in the code is "XL Turbo needs about 12 GB" (model-registry.json, XL Quality pack). -->
<!-- TODO(verify): typical wall-clock time per song per backend, on a named GPU, at default settings. -->

## HeartMuLa

The HeartMuLa server backend is registered for integration testing. Its engine
routes and generation UI are still being implemented; a backend entry does not
mean the native renderer is available yet.

HeartMuLa accepts style tags and lyrics and produces 48 kHz stereo audio. Duration
is an upper bound from 1 to 240 seconds, default 120 when omitted. Create's Auto
duration uses the 240-second ceiling. Audio EOS can end a song earlier.
Each request produces one song. A seed can be a safe non-negative integer
or an exact uint64 decimal string.

Global LM, local decoder, codec flow and scalar decoder quants can be selected
independently, along with the LM variant. Tokenizer and RVQ files are also required;
codec encoder files are not needed for generation. The engine checks component
compatibility and available VRAM before loading weights. A refusal preserves the
requested settings and reports the required and available memory.

The **LM** tab holds the language model's sampling settings: temperature, top-k and CFG
scale, defaulting to the reference values (1.0, 50 and 1.5). The **Generation** tab holds
the codec settings (10 steps, guidance 1.25) and the **Sampler Plugins** switch. With it on,
the shared Lua solvers, schedulers and guidance modes drive the codec's flow decode instead
of its own Euler loop. Plugins that take over the whole loop are refused, and guidance
plugins only run while codec guidance is above 1.

One GGUF LM LoRA can be selected in the **Adapters** tab. Its dials follow YuE2's layout:
Strength (0 to 2), Attention and MLP, and Early, Mid and Late for the three depth thirds of
the 28 backbone layers. The MLP dial only appears for adapters trained with **Train MLP
layers**. An adapter trained on another variant (a 3B adapter on happy-new-year, say)
is re-based onto the loaded one, as ACE-Step's Basin Re-base does: the picker tags it
with its home variant, and **Re-base strength (β)**, default 0.75, sets how far the
layers it trained move toward that home model. 1 runs them as the home model, 0
applies the adapter to the loaded model unchanged. The home variant's global model
has to be installed, in any quant. An adapter's trigger word (the dataset tag
its training samples' tags led with) is shown in the picker and added to the front of
the tags at render time, unless the tags already contain it. Adapters trained before
the word was stored in the file take it from the dataset of the run that made them.
The generation log and the engine log both show the exact tags and lyrics sent. If the
selected file is deleted, the picker shows it as not found and renders are
refused until another adapter, or None, is picked. Put converted adapters
under `<adapters>/heartmula-lm-adapters/`, optionally in run subfolders. Training Studio
trains and publishes GGUF adapters there directly, see [HeartMuLa training](training/heartmula.md);
Safetensors training exports from other tools must be converted before runtime use. Queued jobs retain their
component and adapter choices when the picker changes or another backend is selected.

| Capability | HeartMuLa |
|---|---|
| Text to music and lyrics | Yes, once native engine routes land |
| Duration | Model-ended, requested ceiling up to 240 s |
| Songs per request | 1 |
| Independent component quants | Global, local, flow, scalar decoder |
| Generation adapters | One GGUF LM LoRA: strength, attention, MLP and depth-third dials, 0 to 2 |
| Post-processing | Shared chain, including VST, mastering and StableStep |
| Shared Lua sampler plugins | Solvers, schedulers and guidance on the codec decode, opt-in |
| PP-VAE, Spectral Lifter | No |
| Covers, repaint, streaming, training, lyric alignment | No |
| Output | 48 kHz stereo |

The server maps `heartmulaTags` or the shared caption to `tags`. Optional request
fields are `heartmulaTemperature`, `heartmulaTopK`, `heartmulaCfgScale`,
`heartmulaCodecSteps`, `heartmulaCodecGuidance`, `heartmulaNoiseSeed`,
`heartmulaAdapterStrength`, `heartmulaAdapterScaleAttn`, `heartmulaAdapterScaleMlp`,
`heartmulaAdapterScaleEarly`, `heartmulaAdapterScaleMid`, `heartmulaAdapterScaleLate`,
and, with `samplerPluginsEnabled`, the shared `inferMethod`, `scheduler`, `guidanceMode`,
`pluginParams` and `apgNormThreshold` plus `heartmulaFlowShift`. Per-request model overrides are `heartmulaVariant`,
`heartmulaGlobalModel`, `heartmulaLocalModel`, `heartmulaFlowModel`,
`heartmulaScalarDecoderModel` and `heartmulaAdapter`. Engine request fields and
validation bounds are defined in the [engine contract](../../engine/docs/HEARTMULA_CONTRACT.md).

## Related

- [Generating a song](generation.md)
- [Getting higher quality output](quality.md)
- [Models](models.md)
- [Model Manager](studios/model-manager.md)
- [Adapters](adapters.md)
- [Plugins](plugins.md)
- [Training Studio](studios/training-studio.md)
- [Training ACE-Step adapters](training/ace-step.md)
- [Training MiniMax-Music3 adapters](training/minimax-music3.md)
- [Training YuE2 adapters](training/yue2.md)
- [Stable Audio 3 training](training/sa3.md)
