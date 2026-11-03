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

P0 means work is lost or the task is blocked. P1 costs minutes each session.
P2 is friction regular users stop noticing.

The editor is split by concern under `ui/src/components/player/lyric-editor/`:
`timingModel.ts` holds the pure clamps and transforms (and has the tests),
`useEditHistory.ts` the undo stack, `useLyricSaver.ts` the write and its
revision rules, `useTimelineView.ts` zoom and Follow, `useVisiblePeaks.ts` the
waveform.

## The workflow

1. Get word timings, from the render's aligner or Retranscribe Lyrics.
2. Open the editor from the Lyrics strip's pencil.
3. Orient with the zoom slider in the footer's bottom-left corner, or `Ctrl+-`
   and `Ctrl+=`.
4. Place words by dragging, resizing either edge, nudging, snapping, rippling,
   splitting and merging lines.
5. Check by ear: play from the word, loop it, slow to 0.5x.
6. Save, by autosave or the green Save button, against the file's revision.
7. Come back later, or export.

## P1: minutes lost every session

**B2. No overview, so zooming loses the song.** At 300 px/s a four-minute track
is 72,000 px wide. There is no minimap and no viewport marker. Feature: a
whole-track strip with the visible window drawn on it, click to jump.

**B6. Undo does not survive closing the editor.** History is component state,
capped at 60 entries, and autosave has usually written the result before the
modal closes. Feature: history in a store keyed by track.

**B8. Undo becomes the saved state a second later.** Undo marks the file dirty,
then autosave writes it, and the badge goes `Unsaved` then `Saved`. The undo is
kept, but the labels imply it was committed. Fix: hold autosave after an undo
until the next edit.

**B9. A word is the smallest unit that can be timed.** One word sung across two
syllables can only be stretched. Feature: split and merge words, keeping text
and times.

**B10. A misheard word cannot be corrected.** Words come from reconciling
Whisper against the prompt, and the editor will not change a letter. Feature:
edit word and line text, with the line text rebuilt from the words on save.

**B13. Section labels do not appear in the editor.** Lines carry `section` and
the strip shows it, so "the second chorus" has to be found by counting rows.
Fix: show the label where the section changes.

**B14. Engine-timed tracks have no editor and no explanation.** A track with
only an engine `.lrc` has no pencil, which is correct, but nothing says why or
what to do. Fix: an Edit control that explains it, with a Retranscribe action.

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

**C7. Pan is drag only.** No horizontal wheel mapping and no keyboard pan. Fix:
Shift+wheel and page keys.

**C9. Merge keeps the earlier section.** Merging across a chorus boundary
relabels the second half. Fix: keep both, or state the rule.

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

1. B13 and B14, the cheap orientation fixes.
2. B9 and B10, word split, merge and text.
3. B2, the minimap.
4. B6 and B8, history that survives and reads correctly.
5. The accessibility and touch items.
6. B15, the setup assistant.

## Limits

No browser was run. Touch behaviour, the re-render cost and the crop race are
reasoned from the code. The two `.lyrics.json` write paths were read directly.
