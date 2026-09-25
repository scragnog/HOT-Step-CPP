# Song Builder

Build a song one section at a time instead of generating it in a single pass. Generate a
handful of variants for a section, audition them, pick a favourite, then generate the next
section as a continuation of that pick. Reach for it when you want control over structure
(an intro that sets up a specific chorus, a bridge that follows a particular verse) that a
single text2music generation cannot target directly.

![Song Builder](../../images/studio-song-builder.webp)
<!-- screenshot: needed -->

## Where to find it

Sidebar entry "Song Builder" (`/song-builder`). Song Builder generates through the normal
job queue (see [Generation](../generation.md)), so it needs the same models loaded as any
other generation. It requires the ACE-Step backend: the MiniMax-Music3 backend does not
support the repaint task every section after the first depends on (see
[Backends](../backends.md)).

## Workflow

1. From the project list, start a new song: title, style/caption, and optionally BPM, key
   and time signature (leave them "Auto" to let the model choose).
2. With no section built yet, the only option is the first section: pick a label (preset
   chips for Intro, Verse, Pre-Chorus, Chorus, Bridge, Outro, Solo, Instrumental, or type
   your own), write lyrics or leave them empty for an instrumental section, set a length,
   and generate. This runs as a normal text2music job.
3. Variants stream in one at a time (one job per variant, so the GPU worker finishes them
   in sequence rather than all at once). Play each candidate in the global play bar and
   click Use on the one you want. You can also stop early and pick from whatever finished.
4. Once a section is chosen, the composer switches to Append (continue after the chosen
   section) or Prepend (add before it, e.g. an intro in front of an already-built verse).
   Each new section generates as a repaint extension of the whole song built so far, not a
   separate clip. The model sees the real audio, not just a text description of it.
5. Repeat: label, lyrics, length, generate, audition, pick. The timeline strip shows every
   committed section in order; the most recently chosen one is the "head" that the next
   section extends.
6. There is no separate "finish" step. The head song after your last chosen section is the
   finished track, and it lands in the [Library](library.md) like any other generation
   (filter by source "Song Builder"). If you built without "Apply mastering" checked, run
   post-processing on it from the Library afterward to get the full mastering chain.

## Controls

| Control | What it does |
|---|---|
| Style / caption | Shared style prompt for the whole song, set at project creation and editable afterward in Song settings. |
| BPM, Key, Time sig | Shared musical parameters. "Auto" lets the model decide. Once the first section is chosen, an unset BPM/key backfills from that section's audio so later sections and bar-based lengths line up. |
| Section label | Preset chips (Intro, Verse, Pre-Chorus, Chorus, Bridge, Outro, Solo, Instrumental) or free text. Stored as a `[Label]` tag in the lyric sheet fed to the model. |
| Section lyrics | Lyrics for this section only. Leave empty for an instrumental section. Earlier sections' lyrics (as recorded, not as re-sung) are prepended or appended to build the cumulative sheet each generation sees. A preview is available under "Preview cumulative lyrics sent to the model". |
| Length (Bars / Seconds) | Bars mode only appears once the project has a real BPM, and converts bars to seconds at the project's BPM and time signature. Seconds mode is a 5-90s slider. "≈ from lyrics" estimates a bar count from the number of sung lines (about 2 bars per line) as a starting point. |
| Append / Prepend | Direction for the next section relative to the current head. Prepend is only available once at least one section is chosen. |
| Transition blend | 0-10s. How many seconds of the existing audio at the seam get overwritten and regenerated as a transition, instead of preserved verbatim. 0 is a hard seam. |
| Extend from / Connect intro at (clip point) | Where the new section attaches to the song so far. Defaults to the very end (append) or very start (prepend). Set by loading "Song so far" in the play bar, scrubbing to a spot, then clicking "Set to playhead". |
| Match a section's feel | Optional. Biases the new section toward an earlier chosen section's harmonic shape via its latent, then diverges once your lyrics take over. Strength slider 0.1-0.9 (default 0.4). Marked experimental in the UI; only sections with a stored latent are selectable. |
| Apply mastering/post-processing to these variants | Off by default. When off, these variants skip mastering, PP-VAE re-encode, StableStep, the spectral lifter and LUFS normalization for speed. Cover art, Whisper lyric transcription, quality evaluation, auto-trim and LRC generation are always skipped mid-build regardless of this checkbox. |
| Edit (pencil icon, timeline) | Corrects a committed section's stored label/lyrics without regenerating audio. Use it when the model sang something different from what you typed, so later sections build on what was actually sung rather than the original text. |
| Discard / Stop & keep what's done | Discard cancels the in-flight variant jobs and drops the section. Stop & keep what's done cancels the jobs but keeps whatever variants already finished, so you can still pick from them. |

## Tips and limits

Every section after the first is a repaint over the entire song built so far, so the model
always sees real audio context, not a text summary of it. Each project keeps a fixed
variant count of 4 candidates per section; there is no control to change it after the
project is created.

Song Builder tunes generation for speed and VRAM automatically: it respects your global
"Keep Models in VRAM" setting rather than forcing it, frees the LM after the first section
(repaint sections never need it), and uses smaller VAE decode tiles. None of this is
user-configurable from this studio.

Because sections chain off the previous pick's latent, deleting or discarding a chosen
section partway through a build does not retroactively fix later sections generated from
it. Deleting a project removes its sections but leaves the candidate songs themselves in
the Library.

<!-- TODO(verify): the maximum sensible number of chained sections/total song length is not
enforced or documented in the code read for this page. -->

## Related

- [Create](create.md)
- [Repaint Studio](repaint-studio.md)
- [Library](library.md)
- [Lyric Studio](lyric-studio.md)
- [Generation](../generation.md)
- [Backends](../backends.md)
