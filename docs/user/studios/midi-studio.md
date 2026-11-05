# MIDI Studio

Turns a track into multi-instrument MIDI. Pick a song from your library, or
upload a WAV, MP3 or FLAC, and the native transcription engine writes a `.mid` with
separate parts for drums, bass, guitar, keys, and the rest of the mix. Reach
for it when you want the notes behind a track, not just the audio, for
example to see what a bassline is doing or to pull a MIDI file into a DAW.
MuScriptor also feeds the shared lead-sheet source picker in
[Cover Studio](cover-studio.md) for YuE2 and HeartMuLa, where its chords are shown
as estimates.

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
- **Tempo & bars (optional).** A second, ungated 80 MB model, Beat This!
  (by CPJKU), finds the beats and downbeats. Click its Download link under the
  model sizes. With it, the MIDI gets the song's tempo, a time signature and
  bar lines; without it, the MIDI is written at a fixed 120 BPM and does not
  line up with the bars in a DAW.
- **Engine binary.** `ace-midi` ships next to the main engine. If MIDI Studio
  shows "Transcription engine not found," reinstall the app.

## Workflow

1. If you have not already, open the Model Access card, follow the link to
   request access on the model's Hugging Face page, create a read token
   there, and paste it in. Save token refuses anything that is not a plain
   `hf_` token. Once saved, the card shrinks to a "Token saved" row with
   **Change token**.
2. Under Transcription model, pick a size (small, medium, large) and click
   "Download weights" if it is not already marked Ready.
3. Pick a source track: "Choose from Library" and search, or "Upload from
   PC" (drag-and-drop onto the panel also works).
4. Click "Transcribe to MIDI".
5. Watch the piano roll fill in as chunks complete, or check back later.
   The job keeps running and appears under Transcriptions either way. If a
   generation or post-processing pass is running or queued, the job waits as
   queued until the GPU is free, then starts.
6. Once done, expand the job to preview it, or download the `.mid`.
7. With the tempo model, the expanded job also shows a Tempo & bars panel.
   Correct the tempo, meter or beat there, or quantize for notation, and
   click "Rewrite MIDI". The download then gets the new file; nothing is
   transcribed again.

If the source audio has a `.lrc` file beside it with the same base name,
the downloaded MIDI includes its timestamped lines as lyric events. Section
labels such as `[Chorus]` are skipped. The piano-roll preview shows notes only.

## Controls

| Control | What it does |
|---|---|
| Choose from Library | Search your [library](library.md) and pick a track with audio as the transcription source. |
| Upload from PC | Upload a local `.wav`, `.mp3` or `.flac` as the source (same as dragging a file onto the panel). FLAC is converted to WAV before transcription. |
| Transcription model | `small` (103M, fastest, runs on CPU), `medium` (307M, balanced, GPU recommended), or `large` (1.4B, best quality, GPU required). |
| Download weights | Downloads the selected model's gated weights and its `config.json` from Hugging Face using your saved token. Shows live progress; needs a saved token first. It is the same download as Model Manager > Shared > MIDI Transcription, so starting it in one place shows progress in both. |
| Transcribe to MIDI | Queues a transcription job for the selected source and model. Disabled until a source is picked and that model's weights are downloaded. |
| Transcriptions list | Every job, newest first: queued/transcribing status with chunk progress, or a done/failed result. Jobs persist to disk and survive an app restart. |
| Preview piano roll (chevron) | Expands the finished job into the piano-roll player. A failed job that kept a partial `.mid` has it too. |
| Download .mid | Downloads the transcribed MIDI file. |
| Cancel / delete | Stops a running job, or removes a finished one and its files. |
| Play / pause (in preview) | Plays the original audio and the MIDI transcription together, in sync. |
| Original ↔ MIDI slider | Crossfades between the two audio sources so you can hear either or a blend. |
| Instrument legend, mute (M) / solo (S) | One entry per instrument family found in the track. Mute or solo affects only the MIDI side of playback, not the original audio. |
| Speculative decoding | Shown when `large` is selected; on by default. The small model guesses a few tokens ahead and the large model checks them in one pass, keeping only what it would have written itself, so the MIDI is unchanged. A 4-minute song took 79 s instead of 122 s on an RTX 5090. Needs the small model's weights (about 0.5 GB more GPU memory); without them the large model runs alone. |
| Tempo & bars (model row) | Downloads Beat This!, the beat tracker that gives new transcriptions a tempo, time signature and bar lines. |
| Tempo & bars (job panel) | Shows the detected tempo and meter, and whether the MIDI follows the performance with a tempo map. BPM and beats per bar override them (blank keeps the detected value). Tempo picks automatic, follow the performance, or one tempo. Beat halves or doubles a tempo detected at the wrong level. Quantize snaps notes to a grid (detected, 8ths, 16ths, 32nds, or 8th or 16th triplets) at 25 to 100% strength. Fill short rests holds notes over gaps of half a beat or less. Trim leading bars drops empty bars before the first note. "Rewrite MIDI" applies them; "As detected" goes back. |
| YuE2 lead sheet | Writes the transcription as the two-voice ABC score that YuE2 covers use, on the job's current bars: the sung notes as the Vocal line, the top notes of one instrument as the Ins line (Ins line picks which; Automatic takes the busiest one that is not voice, bass or drums), an estimated key, and, with Chords on, a chord symbol every half bar estimated from the accompaniment. "Copy ABC" puts it on the clipboard for Cover Studio's ABC box; "Save .abc" downloads it. |
| Piano roll | Click anywhere on the roll to seek. Notes are colored by instrument family; with tempo and bars, the background shows the MIDI's beats and numbered bar lines instead of 10-second marks; an unfinished region is shaded while a live job is still transcribing. |

## Tips and limits

The transcription list refreshes about every 1.5 seconds while jobs are queued
or transcribing and stops polling when they finish. A slow refresh does not
start overlapping cycles. Deleting a job waits for server confirmation; an
older pending refresh cannot put a successfully deleted row back. Returning
to MIDI Studio shows existing jobs without submitting another transcription.

How the tempo is written: the beats are fitted with one tempo when they are steady (within 5% of a beat), and with a tempo map that changes only at bar lines when they drift, so bar lines stay on the beat. The time signature is written only when 90% of bars agree on it, always as n/4; otherwise set Beats per bar yourself. The music is delayed by whole bars so bar 1 of the MIDI is a real bar line; the preview moves it back so it stays in sync with the audio. Transcriptions made before the tempo model was downloaded keep their 120 BPM file; transcribe them again to get bars.

The lead sheet is only as good as the transcription's vocal line. On a dense mix the transcriber often hears little singing, and the panel says so when the Vocal line comes out sparse. Its chords are a simple estimate from the instruments' notes, not a trained chord model.

Transcription is one-way: audio in, a `.mid` file out. There is no path from
a transcribed (or any other) MIDI file back into music generation.

The model weights are CC BY-NC 4.0 (non-commercial), and MIDI files you
create here inherit that restriction.

Mute and solo in the preview only silence the synthesized MIDI playback; the
original audio track is unaffected and keeps playing under the crossfade.

Transcription shares the GPU queue with generation and post-processing, one
job at a time, so it never runs beside a render and slows both down.

A transcription job is stopped automatically after 20 minutes of silence from
the engine. Each line of output resets that timer, so a long track can keep
running while it makes progress. A six-hour hard cap applies from when
transcription starts, excluding time spent queued. A job that fails or is
stopped after it has transcribed some notes keeps them: the
server saves those notes as a partial `.mid` and the job stays marked failed,
with the error, after a restart too. Notes still sounding at the moment it
stopped are left out.

## Related

- [Library](library.md)
- [Stem Studio](stem-studio.md)
- [Model Manager](model-manager.md)

## Release data paths

Release MuScriptor tokens and preserved optional-install files live under `<data folder>/library/muscriptor`, with weights under `models/muscriptor` (shared with the other models) unless overridden. Weights downloaded by an earlier release under `library/models/muscriptor` keep working there. The shipped `ace-midi` stays in the app. Upgrading reuses downloaded weights; preserved optional Python environments may need rebuilding after relocation. See [Data folder and upgrades](../data-folder.md).
