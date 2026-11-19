# Lyric editor workflow audit

The lyric editor (`ui/src/components/player/LyricEditorModal.tsx`) was built from
a sequence of requests, so this page walks it the way a person uses it and
records what is still wrong. Already fixed and removed from the lists below: the
failed-save retry storm, hand-placed timings being overwritten by Retranscribe
without a confirm, the crop and second-window write race, wordless lines being
erased, no karaoke preview while editing, dragging stopping at the viewport
edge, Follow switching off with a notice when a drag starts, no way back to the playhead, the
Undo/Revert naming, auto shift past the end of the track, words drawn overlapping
before a save repaired them, out-of-order words being deleted instead of sorted,
the zoom not being remembered, the zoom row taking its own strip, missing
keyboard bindings, and the step box clamping on every keystroke.

The first workflow pass (November 2026) is written up below: the word-actions row
was eighteen buttons in one undifferentiated run, a misheard word could only be
retyped from that toolbar, ticking a line left you to find the next one by hand,
nothing pointed at the words the aligner was unsure of, and both footer shift
groups had buttons reading the same thing.

P0 means work is lost or the task is blocked. P1 costs minutes each session.
P2 is friction regular users stop noticing.

The editor is split by concern under `ui/src/components/player/lyric-editor/`:
`timingModel.ts` holds the pure clamps and transforms (and has the tests),
`useEditHistory.ts` the undo stack, `useLyricSaver.ts` the write and its
revision rules, `useTimelineView.ts` zoom and Follow, `useVisiblePeaks.ts` the
waveform.

## The workflow

1. Get word timings, from the render's aligner or Retranscribe Lyrics, or open
   a track that only has a line-level `.lrc` — its lines are split into word
   chips inside the time the file already gives each one.
2. Open it from the Lyrics strip's pencil, or from a song's **…** menu →
   **Edit Word Placement** on any surface that lists the track. Either door cues
   the track into the player, paused, so the audio is there whether or not
   anything was loaded before; the menu item refuses only when the track has no
   audio file to place words against.
3. Orient with the zoom slider in the footer's bottom-left corner, or `Ctrl+-`
   and `Ctrl+=`.
4. Find where the machine is least sure: **Check doubtful word (n)** selects the
   next under-confidence word, seeks just before it and scrolls to it, and every
   one of them is underlined on the timeline. Ad-libs — words the recogniser
   heard but the sheet never asked for — are underlined apart (dashed, violet)
   and stay out of the queue: they are usually the singer being right.
5. Listen: Space plays, **Play from word** starts a quarter second before the
   syllable, **Loop** repeats it (or the whole selection) at 0.5x.
6. Fix what you hear: drag the chip, drag an edge, `←`/`→` nudge by the step,
   `Shift+←`/`Shift+→` lengthen, Snap to prev closes a gap, Start at playhead puts
   it where you stopped. Move between words with `Ctrl+←`/`Ctrl+→` so your hand
   stays on the keys.
7. Fix what you read: double-click a chip (or press F2 on it) and type. The word
   keeps its times; the word field in the toolbar does the same for a clean
   single edit.
8. When a line is right, tick it. The tick asks whether you listened, records the
   line as ground truth for the fixtures, and selects the next line's first word
   — seeking there when the transport is paused, so a check made mid-playback
   does not skip the music.
9. Re-time a span with the two AI actions on the right of the word row, or take a
   preview's repairs one row at a time. Nothing is written until Apply.
10. Save, by autosave or the green Save button, against the file's revision.
11. Come back later, or export.

## What the workflow pass changed

- **The word row is three clusters**, divided by rules: the line's own tools
  (section label, merges, split), then the selected word (text, step, move,
  stretch, snap, start, play, loop, add, delete), then the two AI actions. Before
  this, merges and split sat on the far right past the AI buttons, so "make this
  two lines" and "re-time this span" were neighbours.
- **Tick and advance.** Ticking a line on selects the next line's first word and
  seeks just before it. A tick used to end the check with no next step, so every
  line cost a tick click, a word click and a Space.
- **Double-click (or F2) to retype a word** where it is, with Enter to commit and
  Escape to cancel. The toolbar field remains for keyboard-only use and for a
  word whose chip is too narrow to read.
- **Check doubtful word (n)** walks the aligner's low-confidence words, in
  reading order and wrapping, using the same jump the repair rows use. Dotted
  underlines mark them on the timeline. A word with no reported probability is
  not counted: many Whisper models report none, and treating that as 0.0 would
  underline the entire track. Ad-libs get a dashed violet underline and no queue
  slot — a queue that stops on them mostly teaches people to delete true words.
- **An applied diff is one undo step.** Applying used to reset the history, so a
  diff that turned out wrong cost a fresh alignment. Apply and Reload now go
  through `replace()`, which records the array on screen first; autosave is held
  for 2.5 s after an undo so the restored state is visible before the file
  changes to match it.
- **A re-align no longer asks about hand-placed timings.** The run is a preview
  and the pipeline carries ticked lines back, so the question arrived before the
  evidence. The diff reports how many hand-placed lines were kept and lost, and
  Apply is the confirmation. Retranscribe keeps its 409, because it applies
  itself and rewrites the line text the carry matches by.
- **Word placement is reachable from the song menu.** Every surface that lists a
  song — Library, Recent Songs, the playlist, Song Details, the queue — mounts
  `SongActionsMenu`, so one item appears everywhere: **Edit Word Placement**. It
  is shown even when it cannot open, and the one refusal is a toast that writes
  nothing: a track with no audio file has nothing to place words against. Needing
  the track loaded is no longer a refusal: the item cues the track into the
  player, paused, through `playbackStore.cue()` — a track restored into the bar
  after a reload was `currentTrack` without its file in the element, so asking
  for a play first left the editor with nothing to play. `cue()` also queues a
  seek for a click that lands before the file has reported its duration, which
  is what the first word click after opening does. This closes B14: an
  engine-timed `.lrc` track has no word sidecar, and both doors now open it
  anyway — `lrcSeed.ts` splits each line into word chips inside the time the
  `.lrc` gives that line, the first save creates the `.lyrics.json`, and the
  header badge reads **LRC lines** so nobody mistakes a seeded chip for one the
  aligner placed. A track with no timing at all — lyrics and nothing else — no
  longer refuses either: the item starts the retranscription itself and opens
  the editor when the sidecar lands. Retranscription runs the same aligner with
  the track's own lyrics as its sheet, so it supplies *when* and never rewrites
  *what*, which is the only honest way to place words that have no time yet;
  stacking every line at 0:00 is not a starting point.

  Once open, the editor makes the same judgement from the timings themselves:
  `timingsLookRough` (`timingModel.ts`) is true for a sheet with no lengths, one
  where most words share an instant, and one whose words stop inside the first
  fifth of a track over a minute long. The header then offers **Align my
  lyrics**, which is the existing whole-track preview run — Whisper for *when*,
  the sheet for *what*, nothing written until Apply — rather than asking the
  person to select a word first so the transport's Re-align has something to act
  on. Placed tracks are left alone, since their remaining work is the doubtful
  queue.
- **Merging across a section boundary says so.** The merged line keeps the
  earlier label and a toast names both, instead of one label vanishing quietly.
- **Distinct footer labels.** "Whole track" and "From selected" each had two
  buttons whose cards carried the same heading; the cards now name the direction,
  and the groups have dividers.
- **The review strip stops asking after Apply.** Its second button says Close
  review and explains that nothing is being discarded.
- **Loop's label follows the selection** (Loop word / Loop 3 words), so it stops
  promising one syllable while playing a line.
- **`Ctrl+←`/`Ctrl+→` walks words** without moving them. Arrows alone nudge, which
  is deliberate, but there was no way to move the selection without the pointer.

What this pass deliberately did not do: word-level split and merge (B9 below),
reduction of the four toolbar rows (C5), and any browser verification.

## P1: minutes lost every session

**B2. No overview, so zooming loses the song.** At 300 px/s a four-minute track
is 72,000 px wide. There is no minimap and no viewport marker. Feature: a
whole-track strip with the visible window drawn on it, click to jump.

**B6. Undo does not survive closing the editor.** History is component state,
capped at 60 entries, and autosave has usually written the result before the
modal closes. Feature: history in a store keyed by track.

**B9. A word is the smallest unit that can be timed.** One word sung across two
syllables can only be stretched. Feature: split and merge words, keeping text
and times.

**B10 (remaining half). A line's text cannot be edited as a line.** A word is
retypeable (toolbar field, or double-click on its chip), and the line text is
rebuilt from the words on save, so a wrong line can be fixed word by word. What
is still missing is replacing a whole line's words at once, which matters when
the aligner spliced two lines together or dropped half a bar.

**B15. Whisper setup failures dead-end the workflow.** On Linux and WSL a
missing `whisper-cli` answers Retranscribe with a build instruction, and the
editor is unreachable. Feature: a setup assistant and a link to troubleshooting.

**C1. Touch targets.** Handles are 6 px, the pan threshold is 3 px, and pinch
zoom does not exist. Fix: larger hit areas, a touch threshold, two-finger pan.

**C3. Screen readers.** Chips now carry `aria-pressed` and a name with their
times, but selection is still colour and the timeline has no list role. Fix: a
listbox-like timeline and a live region for the word under the playhead.

**C4. Colour-only states.** Selection, the playhead and the active word are all
pink. Fix: different shapes or outlines.

**C5. Stacked toolbars on a short screen.** Header, selected word, transport and
footer still take about 150 px before the timeline's minimum. Fix: collapse the
rare rows into a menu.

**C7. Pan is drag only for the pointer.** PageUp and PageDown now page the view
and Shift+wheel scrolls horizontally in the browser, but a mouse without a
keyboard still has only the grab handle. Fix: wheel mapping without Shift.

**C10. Every chip re-renders on every playhead tick.** The modal takes
`currentTime` as a prop and reconciles a button per word at about 10 Hz. Fine at
a few hundred words, not forever. Fix: memoize rows and drive the playhead from
the frame channel the waveform uses.

## Themes

The remaining P1s are two families. The editor still hides its own context
(minimap, sections, engine-timed tracks), and the word is still the wrong unit
for the accuracy left (split, merge, retype). History is the third: it exists
only inside the modal.

## Order

1. B14, the cheap orientation fix for engine-timed tracks.
2. B9 and B10, word split, merge and text.
3. B2, the minimap.
4. B6, history that survives a reload.
5. The accessibility and touch items.
6. B15, the setup assistant.

## Limits

No browser was run. Touch behaviour, the re-render cost and the crop race are
reasoned from the code. The two `.lyrics.json` write paths were read directly.
