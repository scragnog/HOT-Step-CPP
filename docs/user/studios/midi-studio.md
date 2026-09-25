# MIDI Studio

Turns a track into multi-instrument MIDI. Pick a song from your library, or
upload a WAV or MP3, and the native transcription engine writes a `.mid` with
separate parts for drums, bass, guitar, keys, and the rest of the mix. Reach
for it when you want the notes behind a track, not just the audio, for
example to see what a bassline is doing or to pull a MIDI file into a DAW.

![MIDI Studio](../../images/studio-midi-studio.webp)
<!-- screenshot: needed -->

## Where to find it

MIDI Studio in the sidebar. It works with any backend, since transcription
runs on its own engine binary (`ace-midi`), separate from music generation.

Before the first transcription:

- **Model weights.** The transcription model (MuScriptor, by Kyutai and
  Mirelo) is gated on Hugging Face. You need a free Hugging Face account,
  accept the model's access conditions on its page, then create and save a
  read token in MIDI Studio. Weights then download in-app per model size.
  This is separate from [Model Manager](model-manager.md), which does not
  list these weights.
- **Engine binary.** `ace-midi` ships next to the main engine. If MIDI Studio
  shows "Transcription engine not found," reinstall the app.

## Workflow

1. If you have not already, open the Model Access card, follow the link to
   request access on the model's Hugging Face page, create a read token
   there, and paste it in.
2. Under Transcription model, pick a size (small, medium, large) and click
   "Download weights" if it is not already marked Ready.
3. Pick a source track: "Choose from Library" and search, or "Upload from
   PC" (drag-and-drop onto the panel also works).
4. Click "Transcribe to MIDI".
5. Watch the piano roll fill in as chunks complete, or check back later.
   The job keeps running and appears under Transcriptions either way.
6. Once done, expand the job to preview it, or download the `.mid`.

## Controls

| Control | What it does |
|---|---|
| Choose from Library | Search your [library](library.md) and pick a track with audio as the transcription source. |
| Upload from PC | Upload a local `.wav` or `.mp3` as the source (same as dragging a file onto the panel). |
| Transcription model | `small` (103M, fastest, runs on CPU), `medium` (307M, balanced, GPU recommended), or `large` (1.4B, best quality, GPU required). |
| Download weights | Downloads the selected model's gated weights from Hugging Face using your saved token. Shows live progress; needs a saved token first. |
| Transcribe to MIDI | Queues a transcription job for the selected source and model. Disabled until a source is picked and that model's weights are downloaded. |
| Transcriptions list | Every job, newest first: queued/transcribing status with chunk progress, or a done/failed result. Jobs persist to disk and survive an app restart. |
| Preview (piano roll icon) | Expands the finished job into the piano-roll player. |
| Download .mid | Downloads the transcribed MIDI file. |
| Cancel / delete | Stops a running job, or removes a finished one and its files. |
| Play / pause (in preview) | Plays the original audio and the MIDI transcription together, in sync. |
| Original ↔ MIDI slider | Crossfades between the two audio sources so you can hear either or a blend. |
| Instrument legend, mute (M) / solo (S) | One entry per instrument family found in the track. Mute or solo affects only the MIDI side of playback, not the original audio. |
| Piano roll | Click anywhere on the roll to seek. Notes are colored by instrument family; an unfinished region is shaded while a live job is still transcribing. |

## Tips and limits

Transcription is one-way: audio in, a `.mid` file out. There is no path from
a transcribed (or any other) MIDI file back into music generation.

The model weights are CC BY-NC 4.0 (non-commercial), and MIDI files you
create here inherit that restriction.

Mute and solo in the preview only silence the synthesized MIDI playback; the
original audio track is unaffected and keeps playing under the crossfade.

A transcription job that runs past an hour is stopped automatically.

## Related

- [Library](library.md)
- [Stem Studio](stem-studio.md)
- [Model Manager](model-manager.md)
