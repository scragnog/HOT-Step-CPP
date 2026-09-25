# Stem Builder

Stem Builder generates a new instrument stem over an existing backing track,
using the DiT engine's source-conditioned ("lego") task instead of separating
one out. Upload a track, pick an instrument, and it renders a stem that fits
the audio you gave it. Reach for it to add a missing part to a track, try an
alternative take of one that is already there, or build an arrangement up one
generated layer at a time.

![Stem Builder](../../images/studio-stem-builder.webp)
<!-- screenshot: needed -->

## Where to find it

Sidebar entry "Stem Builder", directly below Stem Studio.

Stem Builder needs:

- The active backend to support it. The studio shows a "not supported"
  message instead of its controls on a backend without this capability;
  today that means the ACE backend.
- At least one plain Base DiT checkpoint installed
  (`acestep-v15-base-*` or `acestep-v15-xl-base-*`). SFT, merge and turbo
  checkpoints are not accepted for this task, and the studio shows a warning
  with a link to Get More Models when none is found.
- A source backing track: upload a file, or send a track in from another
  studio's output via "use as source".

## Workflow

1. Drop an audio file (wav, mp3, flac or ogg) onto the upload zone, or click
   it to browse.
2. If more than one base model is installed, pick one from the Model
   dropdown.
3. Pick the instrument to generate from the Target Track grid. Only one
   track can be selected at a time.
4. Optionally open "Style Hint" and describe the tone you want, for example
   "tight house drums, warm vintage tone".
5. Click Build. The button label follows your track choice, for example
   "Build drums".
6. The progress bar and stage text track the render; Cancel stops it.
7. When it finishes, the new stem opens in the preview player next to the
   source and is added to the Layer Stack.
8. To add another layer on top, use the Source button on the newest Layer
   Stack entry (or on an entry in Recent Builds) to make that output the new
   source, then repeat from step 3.

## Controls

| Control | What it does |
|---|---|
| Source Audio | Upload zone for the backing track. Once a file is loaded, shows the filename and a clear (X) button to remove it and reset the session. |
| Model | Base DiT model to render with. Only appears once at least one pure Base checkpoint is installed; auto-selects the first one found. |
| Target Track | Grid of the 12 instrument tracks the engine recognises (vocals, backing vocals, drums, bass, guitar, keyboard, percussion, strings, synth, fx, brass, woodwinds). Single-select. |
| Style Hint (optional) | Free-text description of the stem's character, passed to the engine as the generation caption. |
| Build / Cancel | Starts or stops the render. Disabled until a source, track and model are all set. |
| Preview player | Plays the source and the generated stem together, synced. Independent volume slider and mute per side, plus a shared seek bar. Opens automatically on a finished render. |
| Layer Stack | Every stem built this session, most recent first. Each has a Play button; only the most recent has a Source button, which loads its output as the next source. |
| Recent Builds | Past Stem Builder renders pulled from the song library (up to 20), each with Play, Use as source and Delete. Delete removes the song and its audio file. |
| Queue | Count of pending or in-flight Stem Builder jobs, alongside the shared generation queue. |

## Tips and limits

Duration is not a control here: the rendered stem always matches the length
of the source track.

A Stem Builder render skips mastering, the post-processing effects chain and
any loaded adapter, so what comes out is the raw stem, not the fully
processed output you would get from Create or Cover Studio.

The Layer Stack only lets you carry forward the *most recent* layer as the
next source, so a session builds one chain of layers, not a branching tree.
Loading a new file, or clearing the source, resets the stack.

There is no download or mixdown control inside Stem Builder itself. Every
generated layer is saved as its own song, filed under "Stem Build" in the
Library, where it can be played or downloaded like any other track.

<!-- TODO(verify): confirm whether the active DiT model's max duration/VRAM limits cap how long a source track can be, and whether an over-length source is rejected or truncated. -->

## Related

- [Stem Studio](stem-studio.md)
- [Library](library.md)
- [Create](create.md)
- [Generation settings](../generation.md)
