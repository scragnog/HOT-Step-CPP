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
6. Track progress in the settings panel; Cancel stops the job. The finished
   render lands in the queue and the Library like any other generation.

## Controls

| Control | What it does |
|---|---|
| Source Track | Load the audio to repaint: drag-and-drop or click to upload a file, or Pick from Library to choose an existing song. Clearing the source resets the region. |
| Waveform region | Play/Pause and Play Region buttons, a dual-handle range slider, and Start/End numeric fields (seconds) all edit the same region. The area outside it is dimmed on the waveform. Start cannot go below 0 in this studio. |
| Region Lyrics / Lyrics | Line-by-line editor synced to the region when a `.lrc` file exists next to the source audio (out-of-region lines are read-only context); a plain textarea otherwise. Either way, the edited text becomes the lyrics sent for the whole track, not only the region. |
| Style Description | Caption used for the render. Left empty, the source track's own style/caption is reused. |
| Repaint Mode | Conservative, Balanced (default) or Aggressive, intended to trade how much of the original region is preserved against how much freedom the regenerated region gets. Verified against the current server code: the selected mode is not forwarded to the generation request, so switching between the three currently makes no difference to the render. |
| Boundary Crossfade | 0 to 30 frames at the region edges (25 fps engine rate; 10 frames / 0.4s by default), shown in both frames and seconds. Same as Repaint Mode: verified not forwarded to the request, so this slider currently has no effect either. The engine applies its own fixed blend at the region boundary regardless of this setting. |

## Tips and limits

This studio carries a dismissable "Work in progress" banner in the UI; it is
under active development and may not behave as expected.

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
