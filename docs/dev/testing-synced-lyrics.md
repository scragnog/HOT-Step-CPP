# Testing synced lyrics export

A track's lyric timings live in two sidecars beside the render: `<stem>.lyrics.json`
holds word-level timings, `<stem>.lrc` holds line-level ones. This page is the check
list for getting those timings out of the app: the **Download .LRC** track menu
action, the two `/api/download/:id/lrc` endpoints, the sidecar written beside a saved
audio file, and the operations that have to keep the two sidecars in step
(retranscribe, crop, delete).

Run it after touching `server/src/services/lrcExport.ts`, `server/src/routes/download.ts`,
`server/src/routes/songs.ts` (retranscribe, crop and the delete routes),
`server/src/services/audioCrop.ts`, `ui/src/utils/downloadTrack.ts`,
`ui/src/utils/lrcUtils.ts` or `ui/src/components/shared/SongActionsMenu.tsx`.

## What this covers

- The menu item appears only on tracks that have timings, and downloads a `.lrc`.
- The export served by the server: status, headers, filename, body.
- Rebuilding an `.lrc` from word timings when no `.lrc` exists.
- Save-to-folder mode: the `.lrc` alone, and the one written beside saved audio.
- Retranscribe, crop and delete keeping both sidecars correct.

## Before you start

You need a running dev stack:

```powershell
pwsh -NoProfile -File tools/detach.ps1 dev.bat
```

Use `http://localhost:3000` for the UI and every API check below.

The pure editor timing tests use the project's Node test runner, without an
additional test dependency. From `ui/`, run:

```powershell
node --import ../server/node_modules/tsx/dist/loader.mjs --test src/components/player/lyric-editor/timingModel.test.ts
```

Paths below use `<data root>`, the folder described in
[Data folder and upgrades](../user/data-folder.md). Sidecars are in its `audio`
subfolder, next to the renders they belong to.

Timings arrive one of three ways, and every test below works with any of them:

1. The engine's own alignment (ACE-Step DiT, MiniMax planner, YuE2 forced alignment)
   writes `<stem>.lrc` during generation.
2. Whisper lyrics, on during the render, writes `<stem>.lyrics.json`.
3. Retranscribe Lyrics, from the track menu, writes both.

Pick a scratch song rather than a favourite: two tests below crop and delete it.
Generate one with lyrics and Whisper lyrics on, or use an existing track and run
**Retranscribe Lyrics** from its menu first.

Set up the shell. The `songs` routes want a bearer token, so fetch one from the local
auto-login (it is reissued on every server start, so refetch after a restart):

```
export API=http://localhost:3000
export AUDIO="<data root>/audio"
export TOKEN=$(curl -sS "$API/api/auth/auto" \
  | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>console.log(JSON.parse(s).token))')
export SONG=$(curl -sS -H "Authorization: Bearer $TOKEN" "$API/api/songs" \
  | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>console.log(JSON.parse(s)[0].id))')
echo "$SONG"
```

`/api/download/:id/lrc` itself takes no token, like the audio download route.

## Test 1: the track menu

1. Open Library and find the scratch track.
2. Open its `...` menu.
3. **Download .LRC** should be listed just after **Retranscribe Lyrics**.
4. Click it. With no Download folder set, the browser saves one file named
   `prepend - artist - title.lrc`; with a folder set, a toast names the saved path.
5. Open a track with no timings (an imported audio file with no sidecars). The item
   should be absent, not greyed out.

The item is gated on a successful read of either sidecar, so an absent item on a
track that does have timings points at the sidecar fetch, not the menu. Check the
network panel for `<stem>.lrc` and `<stem>.lyrics.json` under `/audio/`.

## Test 2: the export endpoint

```
curl -sS -D /tmp/lrc-headers.txt -o /tmp/track.lrc "$API/api/download/$SONG/lrc"
cat /tmp/lrc-headers.txt
head -8 /tmp/track.lrc
```

Expect a `200`, `Content-Type: text/plain; charset=utf-8` and a
`Content-Disposition` naming a `.lrc`. In the body, optional `[ti:...]` and `[ar:...]`
lines come first, then one `[mm:ss.xx]text` line per lyric line. When the track has an
engine-written `.lrc`, the body is that file byte for byte, headers included, so the
tags are whatever the engine wrote.

Sanity checks on the body:

```
grep -c '^\[[0-9]' /tmp/track.lrc          # line count, should match the lyric lines
tail -1 /tmp/track.lrc                     # last timestamp must be inside the song
grep -n '[0-9][0-9][0-9]\]' /tmp/track.lrc # must print nothing: centiseconds are 2 digits
```

A song with no timings must answer `404`, with a message pointing at Retranscribe
Lyrics.

## Test 3: rebuilding from word timings

This is the case that made the feature necessary: Whisper and the YuE2 aligner write
only `.lyrics.json`.

```
STEM=$(ls "$AUDIO" | grep '\.lyrics\.json$' | head -1 | sed 's/\.lyrics\.json$//')
mv "$AUDIO/$STEM.lrc" "$AUDIO/$STEM.lrc.bak"
curl -sS "$API/api/download/$SONG/lrc" | head -8
mv "$AUDIO/$STEM.lrc.bak" "$AUDIO/$STEM.lrc"
```

The rebuilt file should still have `[ti:]`/`[ar:]` (they come from the song row, not
the sidecar), one timestamp per line taken from the line's `start`, and section labels
as `[mm:ss.xx] [Verse 1]` where `.lyrics.json` carries a `section`.

## Test 3b: enhanced export (word stamps)

```
curl -sS "$API/api/download/$SONG/lrc?enhanced=1" | head -4
```

Each lyric line should keep its `[mm:ss.xx]` stamp and carry `<mm:ss.xx>` before
every word, taken from the word starts in `.lyrics.json`:

```
[00:12.34]<00:12.34>Hello <00:12.61>world
```

An enhanced export is rebuilt from the word timings whenever the track has any, so
it can differ from the plain `.lrc` on disk even when one exists. A track with
word timings on only some lines enhances those and writes the rest plain, and a
track with no word timings at all still exports, serving the plain file unchanged.

Turning on **Enhanced LRC (word timings)** in Settings makes the menu action and
the folder-mode sidecar send the same `enhanced=1`; with it off, neither does.

## Test 4: save into the download folder

Leave Settings, Download folder empty and confirm the refusal:

```
curl -sS -o /dev/null -w '%{http_code}\n' -X POST "$API/api/download/$SONG/lrc/save"
```

Expect `409`. Now set the folder in Settings (`DOWNLOAD_DIR`) and repeat:

```
curl -sS -X POST "$API/api/download/$SONG/lrc/save"
ls -1 "<download folder>"
```

Expect `{"path":"...","filename":"..."}` and the file on disk, named exactly like the
audio download would be. The unmastered variant carries the same suffix the audio
gets, so the pair matches:

```
curl -sS -X POST "$API/api/download/$SONG/lrc/save?version=original"
```

Now save the audio itself and check the sidecar lands beside it:

```
curl -sS -X POST "$API/api/download/$SONG/save?format=mp3"
```

There should be both `....mp3` and `....lrc` with the same stem. A track without
timings must still save its audio, with no `.lrc` and no error. Saving the same
track twice must produce `... (2).lrc` rather than overwriting.

## Test 5: retranscribe rewrites both sidecars

Record the current state, retranscribe, then compare:

```
md5sum "$AUDIO/$STEM.lrc" "$AUDIO/$STEM.lyrics.json"
curl -sS -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{}' "$API/api/songs/$SONG/retranscribe"
md5sum "$AUDIO/$STEM.lrc" "$AUDIO/$STEM.lyrics.json"
```

Both checksums should change. The response carries `lineCount` and `wordCount`; the
line count should equal the number of `[mm:ss.xx]` lines in the new `.lrc`, and the
export from Test 2 should now match what the lyrics bar highlights.

In the UI, the menu item turns into **Transcribing lyrics...** with a spinner and stays
disabled while the request runs, and a toast stays up until it is replaced by
`Lyrics transcribed with <model>: N lines, M words`, or by the server's reason on
failure. Closing the menu does not lose the run: the spinner is per track and the toast
survives. The toast names the model from `GET /api/songs/retranscribe/model`, and its
**Cancel** button aborts the request; the server sees the response close, kills
`whisper-cli`, and writes no sidecars. A cancelled run is not a failure and leaves the
existing `.lrc` and `.lyrics.json` untouched.

Retranscription is not in the generation queue, which is the GPU lane. A second request
for the same track while one runs answers `409`. Before the fix,
the `.lyrics.json` moved and the `.lrc` stayed on the old alignment.

## Test 6: crop keeps both sidecars on the new timeline

```
node -e "const j=require('$AUDIO/$STEM.lyrics.json');console.log(j.lines[0].start, j.lines.at(-1).end)"
curl -sS -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"inPoint":5,"outPoint":25}' "$API/api/songs/$SONG/crop"
node -e "const j=require('$AUDIO/$STEM.lyrics.json');console.log(j.lines[0].start, j.lines.at(-1).end)"
head -4 "$AUDIO/$STEM.lrc"
```

After the crop, both files should be shifted by the in point: no line before `0`, no
line past `20`, and the first lyric line's timestamp in the `.lrc` should agree with
`lines[0].start` in the `.json`. Then export again and confirm the exported file
matches the cropped audio rather than the original.

## Test 6b: the editor is gated, and its transport works

Collapse the **Lyrics** strip: the pencil must disappear with it. Expand it on a track with
`.lyrics.json` and the pencil appears; on a track with only an engine `.lrc` it must not, because
there are no word timings to place. Click **Edit**: a modal opens over the app with the track's
title (or none) and an **Unsaved** badge only after the first change.

Play from the modal, seek with the 1 s and 0.1 s buttons, change speed to 0.5x, drag the volume,
and toggle **Follow**: the player bar underneath reflects the same speed, volume and position,
because it is the same audio element. With Follow off, let it play: the view stays where you left
it, and closing and reopening the editor keeps it off. Drag the timeline with Follow on: the view
must pan, Follow must switch itself off, and one toast must say Follow has been disabled. With
Follow already off, dragging pans and shows no toast. Click the dimmed background: the modal must
stay open. Press Escape on a clean editor: it closes without asking.

## Test 6c: place and save word timings

Click a word chip. It highlights, its row shows `m:ss.d → m:ss.d`, and the step box controls the
nudge. Drag the chip: it moves with the pointer and its width (the word's span) does not change.
Use the frame-the-word button beside the zoom slider: the view fills with that word and the ruler
switches to 1 s ticks. **Play from word** starts just before it, and **Loop word** now seeks there
as it arms. **Start at
playhead** moves the onset to the playhead. Then **Save**.

```
node -e "const j=require('$AUDIO/$STEM.lyrics.json');console.log(j.lines[0])"
head -4 "$AUDIO/$STEM.lrc"
```

The saved line's `start` must equal its first word's `start`, its `end` its last word's `end`,
its `text` the words joined back together, and the `.lrc` must carry those times with the title
and artist headers intact. The strip and the waveform ticks move to the new times without a
reload.

Export: click **Export lyrics** at the right of the transport row. A `.lrc` arrives named like the
audio download, with any pending change written first; with Enhanced LRC on in Settings it carries
`<mm:ss.xx>` word stamps. With the file changed on disk underneath, the click must refuse and say
to reload instead of exporting the old timeline.

Autosave: change something and wait about a second — the badge goes **Saving…** then **Saved**
with no toast, and the file on disk already has the change. With **Auto save** on, no Save button is
shown. Switch it off in the footer: the green **Save** appears, a change no longer reaches the file
after a pause, and **Save** writes it once. Close with unsaved edits: it asks first. Reopen the
editor: the switch stays off. Undo (`Ctrl+Z`) and redo
(`Ctrl+Shift+Z`), and the header arrows: each action steps one at a time, a whole drag counts as
one, and the badge goes back to **Unsaved** because the undone state has not been written yet
(the next autosave writes it). Drag the empty timeline sideways: the view pans and the playhead
does not move. Click the timeline without dragging: it seeks.

## Test 6d: split a line and shift the whole track

Select the first word of the second phrase in a line that covers both, then **Split line**. The
row becomes two, both halves keep their word times, and the new line starts at that word — with the
playhead parked somewhere else entirely, which is the point: split follows the selection, not the
playhead. With no word selected, the same button splits at the playhead instead. **Whole track
+step** moves every chip right by the step, which is the uniform-offset fix; **From selected +step**
moves only the selected word and those after it, and the chips before it stay put. **Merge next**
folds two lines back into one with the words in order. Play across the boundary: the strip switches
where you split.

Overlap: drag a word left onto its neighbour. It stops at the neighbour's end and the readout
never shows an earlier start. Drag the last word of a line past the next line's first word: it
stops there too. Post an overlapping payload to the endpoint and the stored words come back
non-overlapping. Select the last line and scroll: the extra space below the last row lets it come
up off the bottom edge.

Width: select a word and use **Len +/-**. With **Auto shift** on (the default), lengthening the
word moves every word after it later by the overhang, the readout's duration grows, and no chip
overlaps; with it off, the word stops at the next word's start and the duration stops growing.
Then drag the white handle on the word's right edge: the chip grows with the pointer and the
pill above it shows `start → end`. Drag the left handle: the start moves but not past the previous
word's end, and nothing after it moves. Undo once puts the whole resize back.

**Snap to prev** puts the selected word's start on the previous word's end, in the same line or the
line above; the readout jumps to that time and the chip sits flush against its neighbour. On the
very first word it says there is nothing before it.

Without the UI:

```
curl -sS -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"audioUrl":"<audio_url>","lines":[{"start":0,"end":0,"words":[{"word":"Hello","start":5.1,"end":5.6}]}]}' \
  "$API/api/songs/lyrics/timings"
```

The response carries `lines` and `words`, the `.lyrics.json` is replaced (sorted, blank words
dropped, each line's bounds taken from its words) and the `.lrc` is rebuilt. An empty `lines`
array answers `400`, and a track with no `.lyrics.json` answers `404`, leaving the files alone.
A body whose `expectedRevision` is not the file's current one answers `409` and writes nothing.

## Test 6e: keys, zoom, preview and the overwrite guards

Open the editor and press `Space` with a word selected: the track starts and stops, and the pane
must not scroll. Press it again with nothing focused on the timeline: same result, no double
toggle from the app's global shortcut. `→` and `←` move the selected word by the step and `Shift`
with either changes its length instead; undo steps each back one. With 200 ms in the step box,
`Shift+→` with Auto shift on pushes the following words along by the overhang.

`Ctrl+=` and `Ctrl+-` zoom in and out. Hover each button in the editor: every one opens the house
hover card, the same style as the Auto save switch, and the cards name their keys. The zoom slider
is bottom-left in the footer beside the go-to-playhead and frame-selection icons, and the
transport row no longer carries one.

Page with the round arrows at the left and right edges of the timeline. Back must scroll the view
left by about one window, so the first word on screen ends up last and the new first word is
earlier; it must never scroll right. Forward scrolls right by about a window, so the last word on
screen becomes the first. Both turn Follow off, and forward disables itself once the last word is
at the left edge.

Play a track: the karaoke preview under the timeline highlights the word being sung, and it
follows a chip being dragged before the strip below updates.

Place a word by hand, let it save, then run **Retranscribe Lyrics** from the track menu. The first
click must refuse and say the timings were placed by hand; a second click within a minute
overwrites. The marker is in the file:

```
node -e "const j=require('$AUDIO/$STEM.lyrics.json');console.log(j.editedByHand, j.revision)"
```

With the editor still open, crop the same track from another tab. The next save must be refused:
the footer shows the red **Changed on disk** state with **Reload** instead of **Save**, and no
further request is sent. **Reload** replaces the rows with the cropped file. Stop the server
mid-edit: the badge reads **Not saved** with one toast and no repeats; start the server and press
**Save**, which writes once.

## Test 7: delete removes the sidecars

```
curl -sS -X DELETE -H "Authorization: Bearer $TOKEN" "$API/api/songs/$SONG"
ls "$AUDIO" | grep "$STEM"        # must print nothing
```

Repeat for the bulk path, `POST /api/songs/bulk-delete` with `{"ids":["<id>"]}`, and
for `DELETE /api/songs` (delete all) on a database you can afford to clear. Both
sidecars must go with the audio.

## Test 7b: an LRC with repeated timestamps

A hand-made LRC may put several timestamps on one line. Put one next to a render
and open the track:

```
printf '[ti:Test]\n[00:05.00][01:05.00]Same chorus line\n[00:10.00]Second line\n' > "$AUDIO/$STEM.lrc"
```

The lyrics bar should show `Same chorus line` at 5 s and again at 65 s, with no
`[01:05.00]` text in either. Cropping that track should shift both times. MIDI
Studio reads the same file, so exporting a MIDI from it should place two lyric
events rather than one with a bracket in it.

## Test 8: an external player

Copy an audio file and its `.lrc` into a folder, keeping both names identical, and
open the audio in a player with lyrics support: VLC (Lyrics panel), foobar2000,
Poweramp on Android, or a lyric app that reads a sidecar. The lines should advance in
time with the song and start on the sung line, not a word early or late.

Expect one cosmetic difference from a hand-made LRC: section labels are exported as
timed lines (`[00:12.00] [Chorus]`), so most players show `[Chorus]` as a lyric line
at that moment. The app hides those from the lyrics bar; other players do not.

## Format conformance

The writer emits a subset of the LRC format. What is standard, and what is not:

| What we write | Status | Effect outside the app |
|---|---|---|
| `[mm:ss.xx]` line timestamps, centiseconds | Standard | Every LRC player reads it |
| `[ti:]` and `[ar:]` before the first timestamp | Standard | Title and artist show in players that use them |
| One timestamp per line, chronological | Standard | A repeated chorus is written out twice, at its two times |
| `[mm:ss.xx] [Chorus]` section markers | Not part of the format | The label is shown as lyrics by most players |
| `<mm:ss.xx>` word stamps when Enhanced LRC is on | The enhanced LRC (A2) extension | Words highlight individually in players that support it; the rest ignore the stamps |
| No `[offset:]`, `[al:]`, `[by:]`, `[length:]` | Optional tags | Players fall back to the file and the audio length |
| UTF-8 with no BOM | Encoding is unspecified by the format | Some older Windows players need a BOM to detect UTF-8 for CJK lyrics |
| Metadata tags in the audio file are untouched | Not in scope | `UNSYNCEDLYRICS`/`USLT` still hold the plain lyric text |

A line may carry several timestamps (`[00:12.00][01:20.00]line`), which is how hand-made
files spell a chorus that returns. The app's parsers expand that into one entry per time
and drop enhanced word stamps from the text, and the crop path shifts every timestamp it
finds. See `parseLrcEntries` in `ui/src/utils/lrcUtils.ts` and `splitLrcLine` in
`server/src/services/lrcExport.ts`.

## Pass and fail

Pass when all of these hold:

- The menu item appears exactly on tracks with timings, and downloads a file.
- The export is `200` with an attachment disposition, or `404` with the retranscribe
  hint when there are no timings.
- Save mode writes the `.lrc` alone, and beside saved audio, with matching stems and
  no overwrites.
- Retranscribe and crop change the `.json` and the `.lrc` together, in the same
  timeline, and the app's highlight matches the exported file.
- Delete removes both sidecars on the single, bulk and delete-all routes.
- A player advances the lines with the audio.

Fail, and worth an issue, when the `.lrc` and the `.json` disagree about a line's
start, when the export 404s on a track whose lyrics bar shows timings, or when an
exported file's lines drift from the audio by more than a few hundred milliseconds.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `404` from the export | No `.lrc` and no `.lyrics.json` beside the base render | Run Retranscribe Lyrics, or generate with lyric alignment |
| Menu item missing on a track with a lyrics bar | The fetch was served HTML by a built UI (SPA fallback), so it counts as "no lyrics" | Load through the dev server, or check the fetch's content type |
| `.lrc` and `.lyrics.json` disagree | A writer updated one and not the other | Check the retranscribe and crop paths |
| Timestamps all `[00:00.00]` | The alignment produced no usable times | Check the job log for the lyrics step |
| Filename is `track.lrc` | The title had no ASCII letters or digits, so the sanitiser emptied it | Expected: the audio download falls back the same way, so the pair still matches |
| Lines end with a stray block | The engine wrote CRLF | The app strips it; players tolerate it |
| `403`/`401` from `songs` routes | The token is missing or the server restarted | Refetch `/api/auth/auto` |

## Related

- [Library](../user/studios/library.md): the track menu and download behaviour
- [HTTP API index](api.md): the generated route table
- [Media for frontend clients](frontend-media.md): import and export contracts
- [Writing and maintaining the docs](docs-contributing.md): the doc checker
