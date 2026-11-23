# Alignment decisions

Status: D1–D7 are built. D8–D16 were settled and built in the workflow pass:
they are the questions the editor raised once someone used it, answered the same
way, with the same five alternatives. D2's engine corridor is decided and still
waits on a fixture. Where a pick depends on a measurement that does not exist
yet, that is said rather than hidden.

The settled questions (licence, non-Latin script, preview-then-apply, vocal
stems by default) are in the [scope document](README.md#decisions-taken).

---

## D1. Who owns the vocal-stem cache?

Alignment used to look for a reusable stem and almost never find one: it was
built with its own root (`tmp/lyric-vocals`) while Cover Studio wrote to
`tmp/cover-vocals`. The cache is content-addressed, so the two directories held
identical, interchangeable files under different roofs. `vocalStemCache.ts` is
now the one construction both services import.

| Option | How | Cost / risk |
|---|---|---|
| A. Point the lyric cache at the cover root | one changed `root` string | fixes today's miss; the two constructors can drift apart again |
| B. One shared cache instance both services import | new module owning the root, level and separator | small refactor of two call sites; single place to change the version |
| C. Search both roots in `find()` | try each directory, newest valid file wins | two sources of truth, and a deleted stem resurrects from the other root |
| D. Always separate into an alignment-owned cache | `isolateVocals` defaults to true | a GPU job per track for an unmeasured benefit |
| E. Drop stems from alignment until measured | align the mix, delete the cache | simplest, but throws away a free improvement when a stem exists |

**Pick: B, with the shared root at `tmp/cover-vocals`.** The key is already
`sha256(audio) + level + version`, so the two callers cannot disagree about what
a stem *is* — only about where it lives, which is the bug. Sharing the existing
root also means stems already on disk start hitting immediately, and there is
one place to bump `COVER_STEM_VERSION` when separation changes.

Built: `vocalStemCache.ts` owns the one instance; `yue2Cover` and
`lyricsAlignment` both import it; `find()` stays as the cheap probe. Nothing
else in either pipeline moved.

What would reverse it: if cover stems ever need a different level or version
than alignment (for example alignment tuned to a narrower band), the shared
instance is wrong and per-service roots are right.

---

## D2. Is the CTC corridor implemented, or only described?

`yue2Align(audio, lyrics)` posts the whole track and the exact text; the engine
runs one forward and a Viterbi with no bounds (`yue2-server.h:808`). Exact text
already forces a monotone path, so timing works — but nothing used Whisper's
words to constrain the path, and the module header claimed the CTC pass was
"checked against" them while nothing compared the two.

| Option | How | Cost / risk |
|---|---|---|
| A. Leave it unconstrained and fix the comment | delete the claim, document why exact text is enough | honest and free; drift inside one run stays undetected |
| B. Post-hoc anchor audit in TypeScript | compare CTC onsets with reconciled Whisper anchors, flag or clamp outliers | no engine work; flags after the fact, cannot fix a wrong path |
| C. Anchor corridor in the engine | send anchor pairs, bound each character's frame range in the Viterbi | best accuracy; engine change, API change, needs a fixture to prove it |
| D. Crop-align each anchor segment and stitch | reuse the range path per segment | each crop is a different normalisation input; N model loads; stitches produce seams |
| E. Drop the aligner, use stats-guided Whisper | one timing source, simpler pipeline | abandons the only aligner scored against the actual lyrics |

**Pick: B now, C only when a fixture shows the drift.** B costs a pass over data
the pipeline already has, makes the disputed comment true, and gives the
diagnostics block the number that decides C: how far CTC strays from the
anchors, and where. It cannot repair a path, but a corridor built before knowing
the drift exists is tuning by anecdote. D is rejected on the engine's own terms —
MMS_FA layer-normalises over its whole input, so a crop is a different model
input and a stitched path is not one path. E is a regression on the repo's only
text-aware aligner.

**Built as:** `anchorDrift()` in `lyricsAlignment.ts`, run on the aligner's own
output after reconciliation and before the statistics pass. CTC words are paired
with anchors by text (`textPairing.ts`, four-word resync), not by nearest time,
and the audit is written as `diagnostics.anchorDrift: { medianMs, p90Ms,
outliers, worst, sections }` on a full-track run. Outlier words are named, not
moved. There is no separate stage in `diagnostics.stages`: the block is computed
between `reconcile` and `polish`. Polish carries it over, a selection run keeps
the track's last whole-track audit rather than erasing it, and the module header
now says the anchors are what the CTC path is audited against.

**Where the evidence accumulates.** A favourable number on one track proves
nothing, so the audit is written into the `.lyrics.json` of a full-track run and
outlives the run. Three rules keep the pool meaning one thing: a polish pass
carries the block over (it describes the aligner's output and polish has no
anchors of its own), a range run records none of its own — a crop's line indices
are its own, and pooling it with a track would average two measurements — but
keeps the last whole-track audit rather than erasing it, and
`sections[]` scores each run of lines separately, because a chorus drifting
300 ms late and a verse 40 ms early average into one reassuring median. Then
`npm run lyrics:drift` reads every sidecar on disk and prints the pooled numbers
with one row per section occurrence — the comparison C turns on.

What would reverse it: a fixture showing CTC median error already under 100 ms
with no outliers would make C unnecessary, not just unproven; pooled drift whose
repeat rows look like their first-occurrence rows says the same thing without a
fixture.

---

## D3. What does a fixture contain?

`save` used to copy the whole sidecar, so a file where 30 words were checked by
ear and 400 were left as the machine produced them was all "ground truth".
Every later score was then mostly agreement with the previous run. It now saves
only the lines ticked in the editor.

| Option | How | Cost / risk |
|---|---|---|
| A. Whole sidecar (as it was) | copy the file, score everything | trivial; flatters every change; a machine error becomes the reference |
| B. Only the hand-checked lines | a line-level `checked` flag, fixture stores those lines | the editor already selects whole lines; needs one sidecar field or an editor-only set |
| C. Only individually confirmed words | a per-word toggle | accurate, but confirming 30 words is 30 clicks, and the timeline has no room for a second control |
| D. Both: whole-file trend plus labelled subset | two fixtures, two numbers | the trend number is the one that lies, and two stores to keep in sync |
| E. No stored fixture; score against the previous run | diff two alignments | measures change, never accuracy; a consistently wrong run scores itself as right |

**Pick: B.** A fixture is only worth the name if every time in it was placed by a
person, and lines are the editor's existing unit of intent — the `≡` chip
already selects a whole line. Storing only those lines also keeps the fixture
small enough to read in a diff, which makes it reviewable the way a test
expectation is.

What changes: save takes a line selection (the editor's current selection, or
all lines flagged `checked`); `LyricsFixture` gains `lines` for checked lines
only and keeps `words` counts; `score` reports `missing` relative to the checked
set; the UI needs a way to mark a line checked — a flag on the line's sidecar
entry is the smallest version, and it survives an apply because the pipeline
carries `section` the same way.

What would reverse it: if the `checked` flag is considered too much schema for a
sidecar, C (per-word) is the honest fallback, not A.

---

## D4. Does the user accept a whole diff or individual repairs?

Preview apply is all-or-nothing today: one `repairs[]` list, Apply writes all of
it, Discard writes none.

| Option | How | Cost / risk |
|---|---|---|
| A. Whole diff (current) | one Apply | simple; a bad repair class has to be discarded wholesale |
| B. Per-repair checkboxes | server applies only checked repairs | partial sets break the invariant that repairs are computed together — a snap accepted without its cut-back can overlap its neighbour |
| C. Apply all, then review in the editor | repairs marked on the chips, click to jump, undo per gesture | no partial-write hazard; review happens on a file that has already changed |
| D. Shadow sidecar with A/B toggle | write to `.lyrics.preview.json`, listen to both, keep one | best comparison; two files, a swap step, and every reader must know which is live |
| E. Auto-apply safe classes, preview the rest | monotonic repairs write silently, cut-backs and snaps wait | fewer decisions; "safe" is an assumption the fixture has not yet tested |

**Pick: C.** The repairs are not independent — that is the whole reason the
cut-back runs before the monotonic repair — so B lets a user assemble an
inconsistent file. C keeps the write atomic, uses machinery the editor already
has (per-gesture undo), and turns review into "jump to each moved word and
listen", which is the check the feature exists for. D is the right answer only
if A/B listening becomes routine, which is a fixture question.

What changes: after apply, chips whose `line:word` appears in
`diagnostics.repairs` get a marker; the preview panel's rows become buttons that
select that word at its new time; the header badge tooltip already names the
moved count. No change to the apply route.

What would reverse it: a fixture showing one repair class (probably cut-backs)
is reliably wrong on sustained vocals would make E the better default, with that
class demoted to preview-only.

---

## D5. How does the statistics pass get run on its own?

`mode: 'polish'` existed on the server with `dryRun`, and for a while nothing in
the UI called it: the only way to run the pass was a full alignment, which
re-runs Whisper and the aligner.

| Option | How | Cost / risk |
|---|---|---|
| A. Nothing (current) | full run or nothing | the cheap pass is unreachable exactly when tuning it |
| B. A "Repair timings" button, previewed | `mode: 'polish'`, same review panel | one button and a call; duplicates the preview plumbing by nothing |
| C. Polish automatically when the editor opens | run on sidecar load | rewrites placements without being asked, and burns work on every open |
| D. Polish on every save | hook the save path | the editor's own placement would be "repaired" against the user, the thing `editedByHand` exists to prevent |
| E. CLI only | `npm run lyrics:polish` | fine for the developer tuning thresholds; invisible to everyone else |

**Pick: B.** It is the pass's natural home: seconds, no model, no GPU, and the
same diff panel, so the person tuning cut-back thresholds can apply, listen,
undo, and change a constant without paying for a Whisper run each time. C and D
contradict the decision that nothing writes without a preview.

Built: `LyricEditorWordActions` has **Repair timings** beside Re-align;
`useLyricAlign` gained `runPolish()` that starts `mode: 'polish'`; the review
panel renders whatever `repairs` the job returns.

What would reverse it: nothing foreseeable; if the pass ever needed a model or
seconds became minutes, it stops being a button and becomes part of a run.

---

## D6. What happens when two tracks align at once?

The per-track lock stops a second run on the same song, but two songs can start
together. The engine serialises them on `g_yue2_align_mutex` (`yue2-server.h:780`),
so the second waits inside a 900 s client timeout with no sign of why.

| Option | How | Cost / risk |
|---|---|---|
| A. Nothing | engine queue absorbs it | a long first track can time out the second with a misleading error |
| B. A server-side FIFO for aligner work | one promise chain in `lyricsAlignJobs`, position in `stage` | ~30 lines; the waiting run shows "waiting for the aligner (2nd)" |
| C. Refuse the second run with 409 | check a module-level busy flag | honest, hostile: the user just queues it themselves, later |
| D. Raise the timeout and log waits | 900 s → 1800 s | hides the problem; a stuck run now takes twice as long to fail |
| E. Worker processes with a pool | one process per concurrent align | parallelism the engine cannot give (one mutex, 5 GB buffer); large change |

**Pick: B.** It is the same shape as `gpuLane.ts` without the GPU, matches how the
editor already reports stages, and bounds memory at one aligner instance. A
position in the stage is the difference between "slow" and "broken" from the
user's side.

What changes: a module-level chain in `lyricsAlignJobs` gates the
`runSongAlignment` call for full and range runs; `view.stage` becomes
`queued-aligner` with a depth; the UI's `STAGE_LABEL` gains that string. Polish
runs do not queue — they use no model.

What would reverse it: if alignment moves to a resident model (the engine's own
`ponytail:` note), concurrency becomes a VRAM question and a pool may be worth
it.

---

## D7. How do section labels survive a selection re-align?

A range run sends only the selected lines' text, so `[Chorus]` markers are not in
it; labels are carried back from the nearest line by start time
(`lyricsAlignment.ts`, the `range` branch). A run that returns a different number
of lines can attach a label to the wrong one.

| Option | How | Cost / risk |
|---|---|---|
| A. Nearest start time (current) | already written | wrong whenever the new line lands closer to a different old line |
| B. By selected index | the nth returned line takes the nth old label | breaks when the run drops or adds a line |
| C. Text match, then index | normalised text identifies the line; index as fallback | robust to a dropped line, needs a normaliser and a tie rule |
| D. Do not carry labels at all | new lines have none | loses chorus markers on every partial re-align |
| E. Proportional position inside the crop | map by fraction of the crop's duration | a guess that is wrong in exactly the case that matters (the block moved) |

**Pick: C.** The text is the one thing a re-align does not change — the words are
the same words, retimed — so matching on it cannot be fooled by the times the
run was asked to recompute. Index is a reasonable fallback because a run usually
returns the same lines in the same order, and the current nearest-time rule is
kept only as the last resort.

What changes: a `carrySections(rangeLines, newLines)` helper in
`lyricsAlignment.ts`; comparison normalises case, punctuation and whitespace;
`diagnostics.range.sectionsMatched` records `text | index | time` so a fallback
is visible.

What would reverse it: a run that rewrites words (a lyric correction applied
before re-aligning) would defeat text matching, and index would then be better.

---

## D8. Does a re-align still refuse a hand-timed file?

The guard was the only thing standing between a machine run and hand-placed
words. Now every run is a preview and `carryChecked()` puts the ticked lines back
before the statistics pass touches the rest, so the guard asks a question before
the answer exists.

| Option | How | Cost / risk |
|---|---|---|
| A. Keep the 409 on every run | unchanged | one extra confirmation per run, after a preview that shows the same thing |
| B. Drop it everywhere, retranscribe included | delete the guard | retranscribe applies itself and rewrites the text the carry matches on, so a hand line can vanish with nothing to review |
| C. Refuse only when a ticked line cannot be carried | compare before starting | whether it can be carried is only known after the run |
| D. Run, report `handPlacedKept`/`handPlacedLost` in the diff, keep the 409 on retranscribe | two counts and one guard | a lost line is read in the preview instead of refused up front |
| E. A dialog before starting that passes `force` | editor-side confirmation | the same question as the 409, one layer further from the evidence |

**Pick: D.** The diff is the review; Apply is the refusal. Retranscribe keeps the
guard because it is its own confirmation and changes the line text that the
carry matches by.

What changes: the guard leaves `startLyricsAlign`; `AlignJobResult` gains the two
counts; `refusesOverwrite` stays exported for the retranscribe route.

What would reverse it: a diff that stops naming the moved words, or the removal
of `carryChecked()` — either makes the preview unable to show what was lost.

---

## D9. What does ticking a line mean?

A tick is "these words are right" and, through `handCheckedLines`, also the
fixture's ground truth. The question is what the review loop does next.

| Option | How | Cost / risk |
|---|---|---|
| A. Only a flag | nothing moves | the next line has to be found by hand every time |
| B. Select the next line and seek there, playing or not | as first built | mid-playback it skips the music being judged |
| C. Select the next line always, seek only while paused | one look at `engine.playing` | while playing, the view scrolls and the audio carries the next line in |
| D. Auto-play the next line | `seek` then `toggle` | the person loses the pause they were using to think |
| E. Advance without selecting | scroll only | the word buttons still act on the line just ticked |

**Pick: C.** Ticking ends a check, so the next line should be ready — but a seek
during playback is a skip, not a convenience.

What changes: `toggleChecked` gates the `seek` on `!engine.playing`; selection and
scroll are unchanged. Ticking off still moves nothing.

What would reverse it: someone ticking lines with the transport stopped and
expecting the playhead to stay put — then A with a separate Next-line command.

---

## D10. What joins the "check this" queue?

| Option | How | Cost / risk |
|---|---|---|
| A. Low confidence and ad-libs | as first built | an ad-lib the singer really sang becomes a stop in a queue that says "check this" |
| B. Low confidence only; ad-libs marked apart | two ref lists, two marks | the queue misses invented words, which stay visible on the timeline |
| C. Nothing marked; walk the N least confident | sort by confidence | invents a second ordering and hides the per-word flag |
| D. A per-model threshold in settings | configurable | a knob with no measured default is worse than a constant |
| E. Two queues | doubtful, then ad-lib | two buttons for one job |

**Pick: B.** A doubt is "the machine may be wrong"; an ad-lib is "the machine
heard something you did not write", which is usually the singer being right. It is
marked with a dashed violet underline and its tooltip says so, but the queue —
which exists to spend a minute wisely — does not stop on it.

What changes: `doubtfulRefs` loses its ad-lib branch, `adlibRefs` is new, the
timeline takes an `adlib` set beside `doubtful`, and a low-confidence ad-lib (both
things at once) stays in the queue because the confidence says so.

What would reverse it: real tracks where invented words are mostly misheard
rather than performed — then ad-libs belong back in the queue (A).

---

## D11. Is hand placement per line or per word?

| Option | How | Cost / risk |
|---|---|---|
| A. Per line | the tick the editor has | coarse: one bad word inside a good line still makes the whole line an anchor |
| B. Per word | a second control on every chip | accurate, and a control the timeline has no room for |
| C. Both, word flag overriding | union of the two | two ways to say the same thing, so the fixture needs a rule for each |
| D. Per word, no line flag | drop the line tick | loses "this whole line is right", which is what checking by ear produces |
| E. No flag; the fixture is its own file | save from the CLI | the editor cannot say what was checked |

**Pick: A**, as D3 already chose. `checked` stays on the word type because the
server honours it when present, but the UI offers one granularity, so a fixture
and an anchor mean the same thing.

What would reverse it: a line that commonly contains one wrong word — a
misheard name — where carrying the line blocks fixing it. Then C, with the word
flag winning inside a ticked line.

---

## D12. How long does an undo last, and is Apply undoable?

| Option | How | Cost / risk |
|---|---|---|
| A. Session only; Apply clears the stack | `reset()` after a run | a mis-applied diff costs a fresh alignment |
| B. Session only; Apply and Reload are one undo step | `replace()` keeps the stack | after a reload the stack can hold lines that are no longer on disk |
| C. Persist per track in the browser | a store keyed by audioUrl | two windows can fight over one file's history |
| D. Persist server-side with the revision | history as sidecar rows | the most work for the least-used undo |
| E. No undo; Revert to saved only | drop the stack | throws away the per-gesture granularity the editor already has |

**Pick: B.** An Apply is the biggest change the editor makes and the one most
worth taking back, so it becomes a single history entry. History stays in the
component: a reload of the page is a fresh read of the file, which is the honest
baseline.

What changes: `useEditHistory.replace()` records the current array before
swapping; `refreshAfterAlign` and `reloadFromDisk` use it instead of `reset()`.
Autosave is held for 2.5 s after an undo so the state it restored is visible
before the file changes to match it.

What would reverse it: editors left open long enough that people expect their
history back after a reload (C), or a second window editing the same track.

---

## D13. Merging two lines with different section labels

| Option | How | Cost / risk |
|---|---|---|
| A. Keep the earlier label, say nothing | current | a label disappears without a word |
| B. Keep both, as `A → B` | join the strings | invents a label no `.lrc` or preset understands |
| C. Keep the earlier label and name both | a toast after the merge | one message per real boundary |
| D. Take the later label | swap the rule | wrong for the first half of the line, which is where most of it is |
| E. Refuse the merge | disable the button | a genuine case (a line that straddles a chorus) becomes unrepresentable |

**Pick: C.** The merge is real and sometimes necessary; the loss is what needs
saying, and the section field is right there to overrule it.

What changes: the modal wraps `edits.mergeLine` and, when the two labels differ,
shows the pair in a toast.

What would reverse it: structured labels (`Chorus 2`) where a join produces
nonsense — then E inside the engine-side exporter, not the editor.

---

## D14. How is the fixture score reached?

| Option | How | Cost / risk |
|---|---|---|
| A. CLI only | `npm run lyrics:fixtures -- score` | the loop exists but is buried in a script name |
| B. Named scripts plus the documented loop | `lyrics:score` / `lyrics:fixtures:list` | still a terminal, but discoverable and short |
| C. A score panel in the editor | a route and a panel | a UI for a file that usually does not exist yet |
| D. A settings-page panel | read the fixture store | the same data, further from the track it measures |
| E. A CI job on a bundled fixture | run on every push | needs audio in the repo, which it must not have |

**Pick: B**, until a fixture exists for a track someone is actively tuning.
Scoring is a measurement, not a daily action, and the numbers are only useful
alongside the code that would change them.

What changes: two script aliases in `server/package.json`.

What would reverse it: someone other than the maintainer needing the number —
then C, with the panel reading the same fixture store.

---

## D15. What does a ticked line do to the statistics pass?

| Option | How | Cost / risk |
|---|---|---|
| A. Anchors every word on it | the passes skip it | a systematic drift inside that line cannot be repaired |
| B. Anchors only its first and last word | the rest is free | a hand-placed middle can be overwritten |
| C. Weights it instead of fixing it | a factor in the snap pass | "strong weight" is not a number anyone has measured |
| D. Ignored by polish; only a fixture | no effect | ticking stops protecting anything |
| E. Anchors the line's span, not its words | cut-back respects it | the words inside drift with the neighbours |

**Pick: A.** A tick is a person saying these words are right, so the pass that
exists to second-guess the machine has no business moving them. The cost — an
unrepairable drift inside a ticked line — is the correct trade: ticking is how
you tell the pipeline to leave that line alone.

What changes: nothing new; `polishLines` skips checked lines and `carryChecked`
restores them across a re-align, both already built.

What would reverse it: a track where a ticked line needs repair around a single
fixed word — then B, with the ends as anchors.

---

## D16. Reusing a repeated line's timing

| Option | How | Cost / risk |
|---|---|---|
| A. Copy the earlier occurrence's offsets verbatim | pair by text, copy | a chorus sung slightly faster lands progressively late |
| B. Copy, scaled to the repeat's own span, landing on an energy rise | scale ≤ 15%, snap | needs a scale bound and a snap target |
| C. Re-run Whisper on the repeat | a second pass | minutes per repeat for a line the run already timed once |
| D. Copy only when confidence is high | a threshold | the repeats that need help are exactly the low-confidence ones |
| E. Copy only within the same section | pair by section and text | a chorus repeated in a different section is not reused |

**Pick: B.** A repeat is the same words at a different time, and usually a
slightly different pace; the scale bound keeps a mis-pairing from stretching a
line across the track, and the snap hides the residual. A ticked occurrence is
preferred as the source.

What changes: `reuseRepeats()` before the monotonic, cut-back and snap passes, so
both a full run and a polish run get it; repairs are reported with reason
`repeat`.

What would reverse it: repeats sung with a different *structure* — two words
changed in one occurrence, or a line that only shares its opening — which the
shared prefix/suffix match refuses — or a fixture showing the copy is worse than
the machine's own timing, in which case D's threshold returns as a guard.

---

## How the built pieces actually landed

- **D3.** `LyricsLine.checked` rides through `POST /lyrics/timings`; the editor
  ticks a line with the small check box left of its `≡`; `handCheckedLines()`
  is the only source `lyrics:fixtures save` copies, and it refuses when nothing
  is ticked. `score` therefore reports `missing` against checked lines only.
- **D4.** The preview rows are buttons: each selects its word, seeks just before
  it and scrolls the timeline to it. After Apply the strip stays as a receipt
  and every moved word keeps an amber ring whose tooltip names the repair class;
  the marks disappear on the first edit, because the repair positions would no
  longer describe the array on screen.
- **Hand-placed lines are anchors.** Every word of a ticked line is fixed: the
  cut-back, snapping and monotonic passes skip it, and an overlap is solved by
  pushing the next unchecked word. The wall is geometric as well as protected: a
  tail before an anchor is cut back to it even when the tail is too short to be
  flagged, and a pushed word keeps its duration where the next onset leaves room
  rather than collapsing to the 50 ms floor. A selection re-align is held to the
  same rule at its edges — `polishLines` takes the neighbour lines' bounds, and
  `clampToNeighbours()` checks the block again before the splice, recording
  `range.seamFixes` when it had to move anything.
- **Overlapping voices are separate streams.** `reconcileLyrics()` marks an
  unscripted word that starts more than 150 ms inside the word before it as
  `layer: 'backing'` and starts it in its own line, so a harmony sung over the
  lead's tail no longer stretches the lead line's bounds. `polishLines()` keeps
  one monotonic cursor per layer: a backing word is neither pushed by a lead
  word it overlaps nor allowed to cut the lead's held note back to its onset. A
  word's `layer` wins over its line's, and a word or line with no layer is the
  lead — which is what every sidecar written before the field means. A re-align rebuilds every word from the
  audio, so `carryChecked()` puts the ticked lines back first, matched by text;
  the statistics pass then repairs the rest around them. The file keeps
  `editedByHand` while any carried line remains, so the next run still asks
  before overwriting one.
- **Repeats reuse the first occurrence.** `reuseRepeats()` pairs a line with its
  earlier occurrence by exact text and word count — or, failing that, by the
  leading and trailing words it shares, so a chorus that comes back with one word
  changed ("I used to" against "I still") still reuses the words either side of
  the change — and copies the shared words' offsets (scaled at most 15% for a
  different pace), moving a word only when the current run disagrees by more than
  200 ms and landing the result on the nearest energy rise. The word that differs
  keeps the aligner's own timing, and a source word the aligner itself doubted
  (confidence under `SNAP_CONFIDENCE`) is never used as a prediction. A ticked
  occurrence is preferred as the source. A span whose words are not singable
  lengths is not a pace on either side: a swallowed source is skipped rather
  than copied, so the later occurrence keeps the aligner's placement instead of
  bunching at its first word. The repair is reported with reason `repeat`, so
  the editor shows it like any other move.
- **Retranscribe** (`POST /songs/:id/retranscribe`) is the one run that applies
  itself. It is the action itself that is the confirmation: hand-placed timings
  answer 409 (`code: 'manual-timings'`) and only `force` gets past it. It now
  shares both pieces that matter — `refusesOverwrite()` for the guard and
  `writeTimingsFile()` for the write (revision bump, `editedByHand` clear,
  `.lrc` rebuild) — so "what it means to commit timings" has one implementation
  and two doors. Its in-flight lock is still per song id rather than per
  sidecar, but both doors take the aligner queue, so they serialise; the
  revision check is what stops one stale preview landing on the other's file.

## Summary

| # | Decision | Pick |
|---|---|---|
| D1 | Vocal-stem cache ownership | One shared cache instance at `tmp/cover-vocals` |
| D2 | CTC corridor | Anchor audit in TypeScript now; engine corridor once a fixture justifies it |
| D3 | Fixture scope | Only hand-checked lines, not the whole sidecar |
| D4 | Applying a diff | Apply all, mark repaired words, jump and undo in the editor |
| D5 | Running the stats pass alone | A previewed "Repair timings" action |
| D6 | Two tracks at once | A FIFO for aligner work with a position in the stage |
| D7 | Sections across a range re-align | Text match, index fallback, time last |
| D8 | Re-align and the hand-placed guard | Run and report kept/lost in the diff; retranscribe keeps the 409 |
| D9 | What a tick does next | Select the next line; seek only while paused |
| D10 | The doubt queue's members | Low confidence only; ad-libs marked apart |
| D11 | Hand-placement granularity | One line at a time, as D3 chose |
| D12 | Undo lifetime | Session only; Apply and Reload are one undo step |
| D13 | Merging across a section | Keep the earlier label, name both in a toast |
| D14 | Reaching the fixture score | Named scripts plus the documented loop |
| D15 | A ticked line in the statistics pass | Every word on it is an anchor |
| D16 | Reusing a repeat's timing | Copy scaled, land on an energy rise |
