# Repaint

Repaint regenerates a chosen time region of an existing track while leaving
the rest untouched. Load a source song, drag out the region on the waveform,
adjust the lyrics for that stretch if needed, and render. Reach for it when
one part of a track needs another pass (a flat chorus, a rough transition)
rather than a full re-generation.

![Repaint](../../images/studio-repaint-studio.webp)
<!-- screenshot: needed -->

## Where to find it

Sidebar: Repaint, between Cover Studio and Stem Separator.

Repaint only runs on the ACE-Step 1.5 backend. If MiniMax-Music3 or YuE2 is
active in the global bar, the studio shows a not-supported message instead of
its controls; switch back to ACE-Step 1.5 (see [Backends](../backends.md)).

A source track is required before the waveform and lyrics panels appear:
either a song from the [Library](library.md) or an uploaded audio file.

## Workflow

1. Load a source track: drag an audio file onto the drop zone or click it to
   browse, or click Pick from Library and choose a song.
   The studio's fields also save to a server draft about a second after each
   change. This browser's copy is still the one you edit. The draft belongs
   to this browser: another browser or tool that edits the same draft never
   overwrites your fields. If that happens, your next change is saved as a
   new draft and a short note under the header says so. If a save fails, a
   note says the draft wasn't saved; your edits stay in this browser, and the
   next change tries again. To go back to a saved draft, open **Load a saved
   draft** under the header and pick one; if the form has edits not yet saved
   to a draft, you are asked first. A draft whose uploaded source has gone
   missing is reported as unavailable, so you can reupload it. Loading a
   draft never starts a job.
2. Set the region to regenerate by dragging the waveform's two handles, the
   range slider below it, or the Start/End fields directly. The region
   defaults to the middle third of the track when a source first loads.
3. Edit the lyrics for the region. If a synced lyrics file exists next to the
   source audio, lines inside the region are editable one at a time and lines
   outside are shown dimmed for context; otherwise a plain textarea covers
   the whole lyrics.
4. Optionally enter a Style Description. Left blank, the source track's own
   style is reused.
5. Choose a Repaint Mode and Boundary Crossfade, then click Repaint Region.
6. Track the queued and generating stages in the settings panel; Cancel stops
   the workflow and its audio request. The finished render lands in the queue
   and the Library like any other generation. A server restart marks an active
   workflow interrupted rather than starting a second render.

## Controls

| Control | What it does |
|---|---|
| Source Track | Load the audio to repaint: drag-and-drop or click to upload a file, or Pick from Library to choose an existing song. Clearing the source resets the region. |
| Waveform region | Play/Pause and Play Region buttons, a dual-handle range slider, and Start/End numeric fields (seconds) all edit the same region. The area outside it is dimmed on the waveform. Start cannot go below 0 in this studio. |
| Region Lyrics / Lyrics | Line-by-line editor synced to the region when a `.lrc` file exists next to the source audio (out-of-region lines are read-only context); a plain textarea otherwise. Either way, the edited text becomes the lyrics sent for the whole track, not only the region. |
| Style Description | Caption used for the render. Left empty, the source track's own style/caption is reused. |
| Repaint Mode | Conservative, Balanced (default) or Aggressive. The selected mode is captured as a repaint injection ratio in the generation request; its audible effect still needs listening verification. |
| Boundary Crossfade | 0 to 30 frames at the region edges (25 fps engine rate; 10 frames / 0.4s by default). The selected frame count is captured in the generation request; its audible effect still needs listening verification. |

## Tips and limits

This studio carries a dismissable "Work in progress" banner in the UI; it is
under active development and may not behave as expected.

If a source saved by an older version cannot be verified, select the Library
song again or re-upload the audio. Node checks source ownership and the region
against the current audio duration before adding the render to the queue.

The region controls cap Start at 0, so extending audio before or after the
track (outpainting) is not reachable from this studio, even though the
repaint task itself supports it. Building new instrument layers, isolating a
stem, or completing a mix from a single stem are separate engine task types
that this studio does not expose either; that workflow lives in Stem
Builder, elsewhere in the sidebar.

## Related

- [Create](create.md)
- [Cover Studio](cover-studio.md)
- [Library](library.md)
- [Generation](../generation.md)
- [Backends](../backends.md)
