# Library

Every song you generate or import lands here. Browse it as a cover art grid, a
compact list, or a data-dense table, filter by which studio made a track,
select several at once, and open a track for its full parameter breakdown,
lyrics, and metadata. The player, the playlist queue, and the song details
panel all live alongside it and are covered on this page too.

![Library](../../images/hot-step-library.webp)

## Where to find it

Sidebar entry: Library. No model or backend has to be loaded first, it works
on whatever songs are already in the database, including ones imported from
disk.

## Workflow

1. Open Library from the sidebar.
2. Pick a view (grid, list, or table) and, optionally, a source filter tab to
   narrow the list to one studio's output.
3. Click a track to open its details in the right sidebar, or hover it for
   quick play/download/menu controls.
4. Use the track's action menu (or the right sidebar) for edit, download,
   send-to-Cover-Studio, metadata, or delete.
5. To bring in audio from outside the app, click Upload into Library and drop
   the files in.

## Controls

| Control | What it does |
|---|---|
| View toggle (grid / list / table) | Switches how the song list renders. Remembered per browser. |
| Source filter tabs | All, Custom-Gen, Auto-Gen, Lyric Studio, Cover Studio, Repaint, Stem Build, Song Builder, Imported. Each tab shows a count and filters client-side; only shown on the Library page, not on the Recent Songs list elsewhere in the app. |
| Select | Enters multi-select mode: check tracks, then Download or Delete the selection. Select all / deselect all toggles from the checkbox at the left of the bulk action bar. |
| Per page | 20, 40, 60, 80, 100, or All. Remembered per browser. |
| Columns (table view only) | Choose which columns show. Title and Actions are always on; everything else, including generation details like seed, CFG, solver, and adapter, is opt-in and off by default. Column widths are drag-resizable and persist. |
| Rename (pencil icon) | Inline title edit, on the card, row, or table cell. |
| Track menu (⋯) | Run Post-Processing, Edit (loads the track's parameters back into Create), Add to Playlist, Download, Send to Cover Studio, Edit Metadata, Export Params, Retranscribe Lyrics, Generate/Regenerate Cover Art, Set as Track A / Track B, Delete. Same menu on every song list in the app: grid card, list row, table row, playlist, and the right sidebar. |
| Set as Track A / Track B | Pins two tracks for the A/B comparison bar above the list, with a Play A/B button and a Compare button that opens a parameter diff modal. |
| Upload into Library | Opens the import dialog (see below). Only on the Library page. |

A track still being rendered by MiniMax-Music3's streaming mode appears in
the list before its file exists, with a live percentage badge and progress
bar instead of the normal controls. It is already playable, just not
downloadable yet.

## Player

The transport bar at the bottom of the window is global: it stays visible
across every view, not just Library.

| Control | What it does |
|---|---|
| Shuffle / previous / stop / play-pause / next / repeat | Standard transport. Repeat cycles off → all → one. |
| Spectrum toggle | Shows/hides the frequency analyzer above the waveform. |
| Disco mode | Cosmetic pulse visualization synced to the track. |
| Playback speed | Cycles 0.5x / 0.75x / 1x / 1.25x / 1.5x / 2x. |
| 48k / 44.1k | Previews how the 48 kHz render sounds clocked out at 44.1 kHz, pitched down about 1.47 semitones with tempo unchanged. A quick way to A/B a resample artifact without actually resampling. |
| Variant switch (No Adapter / Unmastered / Mastered) | Only shown when a track has that variant. Each segment plays it and carries its own small download button for that exact render. |
| Trim/crop | Toggles trim mode; the waveform below becomes clickable to set in/out points. |
| Volume | Slider plus mute toggle. |
| Edit / Download / Delete | Quick actions on the currently loaded track. |

Above the transport, a collapsible area (only expands while something is
playing, or trim mode is on) holds section markers, the waveform, a
synced lyrics bar, and, when both slots are filled, the A/B mini-bar.

## Playlists

The playlist is a play queue, not a saved collection: it lives in the
browser's local storage, not the server database, so it does not sync
between devices and is lost if site data is cleared. There is one queue, not
multiple named playlists.

Find it in the right-hand Activity column (Playlist tab, alongside Recent
Songs). Add a track with the list-plus icon next to it anywhere in the app,
or "Add to Playlist" from its action menu. From the playlist itself:
reorder with the up/down arrows, remove a track with the x, or use the
footer's Play All, Download All, and Clear.

## Song details

Click a track (outside of selection mode) to open the right sidebar.

- Cover art, inline-editable title, a backend chip (ACE-Step 1.5,
  MiniMax-Music3, or YuE2), the artist name if set, and the subject or
  description text.
- Play/pause, Edit, and the same track action menu as everywhere else.
- Chips for length, tempo, key, and time signature, shown only when the
  backend that made the track actually measures them. ACE-Step reports all
  three; MiniMax-Music3 and YuE2 do not use fixed tempo/key inputs, so those
  chips are omitted rather than showing a meaningless value.
- Collapsible sections: Prompt (the style/caption text), How it was made
  (the parameters that render's own backend used, grouped as Models,
  Sampling, Planner, Adapters, Reproduction for ACE-Step; Render, Adapter,
  Reproduction for MiniMax-Music3; Plan, Render, Outcome, Adapters,
  Reproduction for YuE2), and Lyrics.

Drag the handle on its left edge to resize the panel.

### Metadata editor

Edit Metadata, from the track menu, opens a small form: title, artist,
album, year, genre/style, BPM, key, comment, lyrics, and a cover image
upload. Title, genre, BPM, key, lyrics, and cover replace the song's own
record and show up elsewhere in the library immediately. Artist, album,
year, and comment are embed-only overrides: they get written into the
audio file's tags on download without changing what the library displays
for the track.

## Download and export

A track's Download action uses the format, bitrate, and preferred version
(original / mastered / both) set on the Settings page. There is no
per-download format prompt. The player's variant switch has its own
inline download buttons that bypass that preference and grab an exact
version instead. Selecting tracks and choosing Download in the bulk action
bar downloads them one at a time, staggered so the browser doesn't block
them, not as a single archive.

Export Params, on the track menu, downloads that render's parameters as a
JSON preset file, separate from the audio and not affected by the
Settings page format.

Downloaded audio has metadata (title, artist, album, year, genre, comment,
cover art) embedded into the file server-side; WAV downloads with no
metadata to add are streamed straight from disk rather than re-encoded.

## Importing your own audio

Upload into Library (header button, and in the empty-state) accepts
`.wav`, `.mp3`, `.flac`, `.m4a`, `.mp4`, `.aac`, `.ogg`, `.opus`, `.webm`,
`.aiff`, and `.aif`. No generation is involved: an imported track becomes
an ordinary library row and gets everything a render has: post-processing,
downloads, playlists, metadata editing.

The optional description field is used as the file's genre tag when it has
none of its own, and as the target caption if you also opt into "Run
post-processing after import". Post-processing runs the chain configured in
the Post-Processing menu, one imported track at a time; without a
description, the StableStep stage in that chain refuses to run for a file
with no genre tag (every other stage still runs).

## Cover art

Generate/Regenerate Cover Art, on the track menu, opens a prompt editor
pre-filled with the same prompt the app would build automatically from the
track's title, style, and lyrics. Edit it freely before generating; Reset
puts the auto-built prompt back. Cover art generation needs the
FLUX.2-klein-4B model downloaded once, from the Cover Art section of the
Post-Processing menu; if it isn't installed yet, the error names the exact
files still missing.

To set a cover without generating one, use the image upload in the metadata
editor instead.

## Tips and limits

- Deleting one track from its menu happens immediately, with no
  confirmation. Bulk delete (via Select) does ask first.
- The playlist queue is per-browser. It is not the place to keep a track
  safe. Use the library itself, or download the file, for that.
- "How it was made" only ever shows the fields the track's own backend
  used. A YuE2 or MiniMax-Music3 song will not show ACE-Step's step count or
  CFG scale, because those fields were never read for that render.
- View mode, column choices, and page size are saved per browser, not per
  account. They don't follow you to another machine.
- Settings has an "include latent file" download toggle, for tracks that
  have one. It adds a second download of the raw latent alongside the audio
  and is not a Library control.

## Related

- [Auto-Gen](insta-gen.md)
- [Create](create.md)
- [Cover Studio](cover-studio.md)
- [Repaint Studio](repaint-studio.md)
- [Stem Studio](stem-studio.md)
- [Generation](../generation.md)
- [Troubleshooting](../troubleshooting.md)
