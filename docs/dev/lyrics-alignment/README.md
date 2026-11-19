# Synced lyrics alignment: scope

Status: built (P0 to P3). This page still states the problem, the requirements,
three ways to meet each one, and the path this branch took, so the code has
something to be measured against. The implementation notes at the end say where
it landed and what is still unmeasured.

## The problem

A render has two possible sets of timings. The backend's own alignment writes a
`.lrc` during generation. Whisper, run afterwards, writes `<stem>.lyrics.json`
with word times and rebuilds the `.lrc` from them. The second is the one the
player highlights and the one the editor edits, so its accuracy is the accuracy
of the feature.

It is not good enough on sung audio, for reasons that are structural rather than
tuning:

1. Whisper is a speech recogniser. Sung vowels are longer, consonants are
   quieter, and the accompaniment masks both.
2. Token times come from decoder timestamp tokens, on a roughly 1 s grid, then
   get interpolated. `server/src/services/whisperTranscribe.ts` already carries
   a comment saying exactly this: even token times put lines early or late on
   sung audio.
3. Whisper transcribed the whole file in one CLI run with a 120 s timeout
   (`transcribeWithWhisper`), so a long track was both slow and cut off. P0
   replaced that with activity-gated windows and a configurable budget.
4. Silence gets filled with invented words. That is why the `--prompt` flag was
   removed and why `reconcileLyrics` throws away words with no source match.
5. The word sequence Whisper produces is not the lyric text, so timing requires
   a reconciliation step (`needlemanWunsch`), and every mismatch is a word with
   no honest time.

The result is a file that is right on the chorus and drifts at section
boundaries, which is the complaint the lyric editor and `lyricsOffset` exist to
absorb by hand.

## What exists today

| Piece | Where | What it gives |
|---|---|---|
| Engine alignment | ACE DiT cross-attention, MiniMax planner, YuE2 MMS-FA | `.lrc` written at generation time |
| Whisper transcribe | `server/src/services/whisperTranscribe.ts` | word-level `WhisperResult` |
| Reconciliation | `server/src/services/lyricsReconcile.ts` | NW word alignment, phonetic hash, fuzzy match |
| Retranscribe route | `POST /api/songs/:id/retranscribe` in `server/src/routes/songs.ts` | runs whisper inline, rewrites both sidecars |
| Manual offset | `server/src/services/lyricsOffset.ts` | one constant shift per track |
| Hand placement | `server/src/services/lyricTimings.ts` | editor writes, revision guard, `editedByHand` |
| Editor | `ui/src/components/player/lyric-editor/` | drag, resize, split, merge, undo |
| MMS-FA forced aligner | `engine/src/yue2/yue2-mmsfa.h`, `yue2-ctc-align.h`, `POST /yue2/align` | CTC spans for exact lyrics, codepoint offsets |
| MMS-FA client | `yue2Align()` in `server/src/services/backends/yue2/client.ts` | already sends audio plus lyrics, 900 s timeout |
| Vocal separation | `POST /api/supersep/separate` | stems from the engine |

Two details change the shape of the problem.

`normaliseWhisperJson` already reads a `t_dtw` field from each token and prefers
it over the decoder offset. The flag that would populate it is never passed.

The engine already ships a CTC forced aligner (Meta's MMS_FA, ported to GGML and
gated against torchaudio's own output). It takes audio plus the exact lyrics and
returns per-word spans, one word per sung word, with codepoint offsets into the
lyrics. `POST /yue2/align` is not YuE2-specific in any way except where it looks
for the model file: it decodes WAV or MP3 at any rate, and it is the only aligner
in the repo that scores words against the text the song was actually given.

## The thesis, and where it holds

The thesis is: Whisper does the first push, then statistics and a sliding window
clean it up, with the DTW search path confined to a narrow corridor.

The first-push half is right, and it is what the repo lacks now. Whisper's value
is not its precision, it is that it produces a monotone map from audio time to
lyric index when the text is not known in advance, and it covers 99 languages
without extra models.

The corridor half is right in principle and wrong in one specific way. A
Sakoe-Chiba band or an Itakura parallelogram is drawn around the straight
diagonal, which encodes "the words are spread evenly across the file". Songs
break that assumption harder than speech does: 15 s intros, 20 s outros,
instrumental breaks between verses, a chorus repeated three times with different
amounts of ad-libbing. A band tight enough to help is too narrow to contain the
true path, and the aligner then fails in the middle of a chorus instead of
drifting gently. The band has to be drawn around a path that is already roughly
known, which is what the Whisper push is for.

The other correction is that MFCC/energy DTW is not a text-to-audio matcher. It
aligns two audio signals. To use it against lyrics the text first has to be
turned into audio or into a forced alignment, and the moment a forced alignment
exists it is already the answer. In this repo MFCC DTW already has a job, in
`server/src/services/backends/yue2/coverDrift.ts`, where it compares two
renders of the same song. That is the role it should keep: measuring and
refining between anchors, not establishing them.

So the recommended shape is: Whisper for the push, the in-repo CTC aligner for
the timing, and statistics plus a windowed energy profile to police the result.

## Requirements

Each requirement lists three realistic options and picks one. The picks are what
the first implementation should do, not laws.

### R1. Source of truth for timing

| Option | How | Cost |
|---|---|---|
| A. Whisper plus NW reconcile | current pipeline | timing only as good as token interpolation |
| B. CTC forced alignment on the exact lyrics | MMS_FA via `POST /yue2/align` | needs the 1.26 GB model, Latin-ish scripts |
| C. Engine alignment head | `.lrc` already on disk from the render | free, but only when the backend has a head |

Pick: B when the model is installed, falling back to A. C is not a substitute,
because a backend's own alignment is the thing being checked; when it exists it
should be compared, not trusted.

B costs one call. `yue2Align()` is already written, already sends the two parts
multipart, and already documents its own budget: one wav2vec2-large forward over
the whole track plus a Viterbi, about 100 s of CPU per four minutes of audio,
hence its 900 s timeout. It is not on the GPU lane and it is already safe to run
from a background job.

Whisper keeps three jobs under this choice: producing anchors, verifying that
the render actually sang the supplied lyrics (the YuE2 cover path already has
`whisperTranscriptPlausible` for this), and serving every case where B is
unavailable.

### R2. Whisper invocation

| Option | How | Effect |
|---|---|---|
| A. Current flags | `-ojf --split-on-word -ml 1 --beam-size 5` | decoder offsets, 1 s grid |
| B. Add `--dtw <preset>` | whisper.cpp token DTW over alignment heads | real token times, same binary |
| C. Replace with Python aligners | stable-ts, whisperX, ctc-forced-aligner | new runtime, new install story |

Pick: B. The parser already handles `t_dtw`, so this is a flag plus a preset
lookup and a regeneration of the model priority list. C is rejected because this
app ships a single Node/C++ stack with no Python runtime, and because
whisperX's improvement comes from a wav2vec2 forced aligner that this repo
already has natively. A stays as the fallback when the installed build has no
preset for the chosen model.

### R3. Constraining the search path

| Option | Constraint | Failure mode on songs |
|---|---|---|
| A. Unconstrained | none | path finds a better-scoring wrong route through a repeated chorus |
| B. Sakoe-Chiba band | `abs(i/T - j/N) <= r` | intro and outro fall outside the band |
| C. Itakura parallelogram | slope limited to 1/2..2 | forbids held notes and melisma, which are exactly song behaviour |
| D. Anchor-seeded corridor | piecewise band between anchor points | needs anchors, which R2 provides |

Pick: D, with B applied inside each anchor segment. Anchors are the Whisper
words that survive reconciliation with high confidence. Between two consecutive
anchors the true path cannot leave the rectangle they define, so a band in that
rectangle is safe even when it is narrow. Where anchor density is low (a long
instrumental), widen the band and let the lattice decide rather than forcing a
line.

C is rejected on its merits, not on cost: a slope constraint that caps how long
one lyric symbol may occupy the audio is wrong for sustained vowels.

### R4. Windowing

| Option | Window | Note |
|---|---|---|
| A. Whole track | 1 pass | Whisper cannot: 30 s encoder, and one CLI call was capped at 120 s |
| B. Fixed windows with overlap | 30 s, 5 s overlap | boundaries cut words in half |
| C. Activity-gated windows | windows start and end on vocal energy | needs the energy profile anyway |

Pick: C for Whisper, A for the CTC aligner. The second half is not a preference:
`yue2-mmsfa.h` layer-normalises the waveform once over the whole utterance, so
the engine documents that a track fed in pieces is a different model input in
every piece. Windowing is a Whisper-side technique here, and it is also the
right shape for the statistics layer, which is where "sliding window" belongs:
a 2 s context median over a 250 ms hop is what makes an energy threshold work in
both a quiet verse and a loud chorus.

### R5. Vocals isolation

| Option | How | Cost |
|---|---|---|
| A. None | align the mix | accompaniment masks consonants |
| B. Existing SuperSep stem | `POST /api/supersep/separate`, cache the stem | one GPU-lane job per track |
| C. External Demucs or MDX | new runtime | another install story |

Pick: B. The engine and the route already exist, and the stem is reusable for
covers and stem work. The 2025 literature on lyrics alignment is consistent that
separation helps; the open question for this repo is whether it helps MMS_FA as
much as it helps Whisper, which is a measurement, not an argument.

### R6. Statistical layer

| Option | What it does | Risk |
|---|---|---|
| A. None | accept aligner output | single bad span drags a whole line |
| B. Priors and outlier rejection | syllable-rate and duration checks, monotonic repair | none if only flagged |
| C. Energy-onset snapping | move an onset onto the nearest energy rise within +/-120 ms | can move a correct word if the gate is wrong |

Pick: B then C, in that order, and C only inside a tolerance that cannot cross
a neighbour. Concretely: median inter-onset interval from the anchors; flag any
word whose implied duration is over 3x or under 0.5x the local median; re-time
flagged words onto the nearest energy rise; leave everything else alone. Every
adjustment is recorded, so the editor can show what was moved.

Two more rules sit around those three. A line that repeats gets a fourth: its
words take the earlier occurrence's offset from the line start (scaled only
inside 15% for a different pace), and a word the repeat disagrees with by more
than 200 ms is landed on the nearest energy rise. And a line ticked as placed by
ear is an anchor, not a measurement — none of those rules move its words, and the
words around it are what gets pushed instead.

### R7. When the sung words differ from the lyric sheet

| Option | Approach | Handles |
|---|---|---|
| A. Global NW on words | current | substitutions and skipped words |
| B. Phonetic hash | already in `matchScore` | misheard spellings |
| C. Anchor DP with repeats | monotone, allows skip and repeat | a chorus sung twice with ad-libs |

Pick: A and B as they are, plus a repeat rule. The specific failure this fixes
is a chorus that appears three times in the audio and once in the lyrics: global
NW has to place it once, and every later word inherits the error. Detecting a
repeated line and allowing the path to re-enter it is the cheap fix, and it is
worth doing before any tuning.

Built at the timing level instead of inside the path: `reuseRepeats` in
`lyricsAlignment.ts` runs after both stages have produced words, pairs a line with
its earlier occurrence — by exact text and word count first, and failing that by
the prefix and suffix the two lines share, so a chorus that comes back with one
word changed still reuses the words either side of the change — and copies the
earlier word offsets onto it where this run disagrees by more than 200 ms. Only
the shared words move, the pace is measured across the shared span alone, and an
earlier word the aligner was unsure of (confidence under `SNAP_CONFIDENCE`) is
never used as a prediction, so one doubtful onset is not spread to every later
occurrence of the line. Teaching the needleman-Wunsch pass to re-enter a line
would change which transcript the path prefers; copying offsets changes only the
timings, which is the failure that was observed.

### R8. Granularity

| Option | Unit | Trade-off |
|---|---|---|
| A. Line | one stamp per line | cannot highlight karaoke |
| B. Word | one span per word | matches `.lyrics.json` and the editor |
| C. Syllable or phoneme | sub-word spans | needs a lexicon, changes the contract |

Pick: B for the stored contract. C stays internal: the CTC aligner works on
characters and merges them into words, which is where the precision comes from.
Adding syllable spans to the sidecar is a separate decision with an editor cost.

### R9. Provenance and persistence

| Option | Shape | Consequence |
|---|---|---|
| A. Overwrite as now | `lines[]` with `method` | no way to tell a good run from a bad one |
| B. Add evidence fields | per-word `confidence`, `source`, plus an alignment `diagnostics` block | editor and support can see why |
| C. Separate raw sidecar | `<stem>.align.json` raw, `.lyrics.json` derived | more files, easier debugging |

Pick: B. Keep `revision` and `editedByHand` semantics exactly as they are; a
hand-placed file must still refuse a machine overwrite without an explicit
force. Record `vocalsIsolated` truthfully — before this it was hardcoded `false`
at every call site, so a run on a separated stem claimed it had aligned the mix.

### R10. Where it runs

| Option | Placement | Failure mode |
|---|---|---|
| A. Inline in the route | current | a long track pins the request; 409 on refresh |
| B. GPU lane job | serialised with generations | blocks user generations for minutes |
| C. Background job with id and progress | own worker, cancellable | more code |

Pick: C. Retranscription is already minutes long, already abortable, and
already refuses concurrency per track. It needs a job id so closing the tab does
not lose the run, and it must not sit on the GPU lane, which the route comment
already says out loud.

### R11. Evaluation

| Option | Basis | Weakness |
|---|---|---|
| A. Ear test | listen and judge | not repeatable |
| B. Hand-labelled fixture | mark 30 words per track, measure error | costs an hour to build |
| C. Cross-aligner disagreement | run two aligners, compare | both can be wrong the same way |

Pick: B, built with the editor itself, which already seeks at 0.1 s and loops a
single word, so labelling is the workflow the app was built for. Use C as the
cheap regression signal on every track, and A to decide whether a number that
improved actually sounds better.

## Recommended path

1. Vocal stem. Run SuperSep once per track, cache the vocal file next to the
   render, reuse it. Everything downstream reads the stem, not the mix.
2. Whisper push on the stem, windowed and activity-gated, with `--dtw`. Reconcile
   against the lyrics as now, but keep every matched word with its confidence and
   index. That set is the anchor list.
3. CTC forced alignment of the exact lyrics on the stem, with the lattice
   restricted to the corridor between anchors (R3). Where anchors are missing,
   leave the lattice open.
4. Statistics pass (R6): outlier flags, energy re-timing inside tolerance,
   monotonic repair, line bounds recomputed from words.
5. Write `.lyrics.json` with per-word `confidence` and `source`, a `diagnostics`
   block naming the stages that ran, the model, the window count, and how many
   words were moved. Rebuild `.lrc`. Clear `editedByHand`, bump `revision`.
6. Editor unchanged in its contract. It gains one action and one badge: run the
   alignment again, and see which stage produced the current timings.

Step 2 alone is worth shipping early. It is a flag and a timeout, and it is the
difference between decoder-grid times and token times.

## What would have to be measured first

The only number that already exists is the aligner's cost, measured by its own
client: roughly 100 s of CPU per four minutes, one model load per call. Nothing
else has been measured, so these are the numbers the plan commits to producing
before any of stages 3 to 5 is called done.

- Word onset error against the hand-labelled fixture: median, P90, share within
  100 ms and within 250 ms, per stage.
- Monotonic violations per track, which should be zero.
- Second-chorus offset, the single number that exposes a global-alignment
  failure.
- Anchor drift per track, written to `diagnostics.anchorDrift` by a full-track
  run and pooled by `npm run lyrics:drift`: median, P90, outliers, and the same
  numbers per section occurrence. The repeat rows are what decide whether the
  engine corridor (D2-C) has anything to fix.
- Wall time per track and peak VRAM for each stage, on this machine, with the
  GPU idle.
- Whether separation changes the answer for MMS_FA. If it does not, stage 1 is
  optional and the cost drops by a GPU job.

## Phasing

P0. Whisper `--dtw`, timeout raised and made configurable, windowed invocation,
truthful `vocalsIsolated`. No new model, no new contract fields.

P1. Call `yue2Align()` for any backend, on the vocal stem, with provenance
fields. Fall back to P0 whenever the model is absent.

P2. Corridor and statistics pass, with the diagnostics block.

P3. Editor action and badge, plus the fixture-based regression check in docs.

## Decisions taken

The questions that were still open after the first implementation — stem-cache
ownership, the corridor, fixture scope, per-repair accept, the polish action,
aligner concurrency and section labels across a range — are in
[decisions.md](decisions.md), each with five alternatives and one pick.

| Question | Decision |
|---|---|
| Is the CTC aligner allowed to be the default? | Yes. The weights are CC BY-NC-4.0 and the editor names that beside Re-align. HOT-Step is open source and non-commercial, and a timing is a time: the restriction is on the weights, not on the `.lyrics.json` a run produces. |
| What happens to non-Latin lyrics? | Measured, not guessed. Above 15% of the lyric letters outside the aligner's romanised alphabet (`MAX_NON_ROMANISED` in `lyricsAlignment.ts`), the CTC stage is skipped and Whisper times the track. `diagnostics.fallback` says so and the editor badge shows it, so the run is honest rather than confidently wrong. |
| Does the statistics pass happen behind the user's back? | No. Every run finishes as a preview: the repairs, their before/after times and their reasons are listed, and `POST /api/songs/lyrics/align/:jobId/apply` is the only thing that writes. Discard writes nothing. The preview is held for an hour, and applying over a newer file needs an explicit force. Retranscribe is the one run that applies itself: it answers 409 on hand-placed timings and takes `force` as the confirmation, then writes through the same function an apply uses. |
| Is vocal separation required? | No. A stem already on disk from cover or stem work is used, because it is free; otherwise the mix is aligned. `isolateVocals: true` asks for a separation now, `false` forbids a stem. This is the opportunistic option until a measurement says separation changes the answer for MMS_FA. |

## Fixtures

A hand-labelled fixture is the only measurement this feature trusts. The editor
is the labelling tool: place thirty words on a track by ear, then

```
npm run lyrics:fixtures -- save "<audioUrl>"     # the checked lines become ground truth
npm run lyrics:fixtures -- score "<audioUrl>"    # score the sidecar as it is now
npm run lyrics:fixtures -- list
```

`save` copies the lines ticked in the editor — the check box left of a line's
`≡` — to `<data root>/fixtures/lyrics/<track>.json`, so a re-alignment cannot
destroy the reference. It keeps nothing else: a line the machine placed and
nobody listened to is not ground truth, and scoring against it would mostly
measure agreement with the previous run. With no line ticked, `save` refuses
rather than recording the whole sidecar. `score` prints the numbers the
phasing table asks for: median, P90 and max onset error, the share inside 100 ms
and 250 ms, monotonic violations, the second-chorus offset, and a per-section
median. Words are paired by text with a four-word resync, not by
Needleman-Wunsch, because both sides are the same lyrics and a general alignment
would hide a whole line landing somewhere else.

## Risks and open questions

The MMS_FA weights are CC BY-NC-4.0 upstream, and `server/src/data/model-registry.json`
records that. That question is answered above: the aligner stays available, the
editor names the licence, and what a run writes is a timing.

The port's CTC labels are a 28-symbol romanised set. Latin-script lyrics are
fine; a word with no mappable character inside an otherwise Latin sheet is
dropped rather than guessed. Lyrics in another script are timed by Whisper, and
say so: `nonRomanisedRatio()` measures the share of lyric letters outside the
Latin script and a track over `MAX_NON_ROMANISED` skips the CTC pass with a
fallback reason naming the percentage. Accented Latin counts as Latin, and
digits are not counted at all, so `café` and `2024` do not trigger it.

Whisper's `--dtw` takes a preset name that must match the installed model
family. The preset for `large-v3-turbo` was added after the flag itself, so the
bundled whisper.cpp build and the model priority list have to be checked
together. A mismatch must fall back cleanly rather than pass a bad preset.

The engine aligner loads 1.26 GB per request and wants about 5 GB of compute
buffer. It frees both afterwards. Parallel alignment is therefore not free, and
the per-track mutex in the engine is the thing that makes a background job safe.

## What is built

| Phase | Where it lives |
|---|---|
| P0 | `whisperTranscribe.ts` passes `--dtw <preset>` (`dtwPresetFor`), retries once without it when a build rejects the name, reads its budget from `WHISPER_TIMEOUT_MS`, and runs a track in activity-gated windows (`audioActivity.ts`, `transcribeTrack`). Every writer now records the real `vocalsIsolated`. |
| P1 | `lyricsAlignment.ts` calls `yue2Align()` on the exact lyrics, turns spans into sidecar lines with `yue2LyricsJson`, and falls back to Whisper when the model is absent or the text is outside the aligner's romanised alphabet. Vocals come from the existing SuperSep stem through the cache Cover Studio shares (`vocalStemCache.ts`). |
| P2 | `polishLines` repairs monotonic order, cuts back a word that swallowed a gap (its tail collides with the next onset or sits over silence, never a held note that keeps sounding), and moves only flagged onsets onto a nearby energy rise within 120 ms. Baselines are median plus MAD per section and per vocal layer (same-layer lines nearby first, then the layer-blind fallback, then the track), the duration band is syllable-aware, and the energy threshold is a one-second local level rather than the file's peak. `diagnostics` records the baseline, per-section bands, and up to 500 `repairs[]` with before/after and reason. `mode: 'polish'` re-runs only this pass; every mode finishes as a preview, so `dryRun` no longer changes whether anything is written. |
| P3 | The lyric editor shows a provenance badge, and `POST /api/songs/lyrics/align` runs a full or per-selection alignment as a pollable job (`lyricsAlignJobs.ts`). A selection is a word span: clicking a line's chip selects every word on it, Shift-click extends by words, and Re-align replaces only the lines the selection touches. The word row also offers **Repair timings**, which runs only the statistics pass as the same kind of preview. The timeline labels a ticked line, a backing or ad-lib layer, and the track's anchor-drift badge. |

Two limits stated above still stand: the MMS_FA weights are CC BY-NC-4.0, and
the numbers in "What would have to be measured first" have not been produced
yet. No accuracy claim is made for any stage.

### Pronunciation targets

The aligner scores letters against audio, and the engine keeps only `[a-z']`
from a token when it builds the CTC targets. For an ordinary lyric that is the
right thing to do. For the words a song makes hard it is the wrong text:

| Written | Engine target before | Now posted as |
|---|---|---|
| `AGI` | `agi` | `ay gee eye` |
| `NVDA` | `nvda` | `en vee dee ay` |
| `2024` | nothing — the word is dropped | `twenty twenty four` |

Nobody sings "agi", and nobody sings a year as a cardinal, so the target was
the one part of the alignment that did not describe the performance.
`server/src/services/lyricsPronunciation.ts` builds the string the aligner is
given, in the same move the pdoom-video project makes in its `analysis/pron.py`:
an explicit table for the tokens an artist spells out, then digits, then a
fallback for an all-caps token with no vowel to hold (`XML` is spelled, `NASA`
and `FOOM` are not). Because the engine's contract is that the returned
offsets index the string it was sent, the module also returns the map from each
unit back to the lyric word it spells, and `yue2LyricsJson` folds the spans onto
that word — one word in the sidecar, one span, the first unit's onset to the
last unit's end, the mean of their confidences.

Two consequences worth stating. A word with no letters at all (a dash, a note
symbol) has no target and no span — unchanged, and still honest. And a digit-only
word that used to disappear from its line is now timed, which is the fix this
came in for. `R8` is untouched: units exist inside the aligner call, and the
stored contract stays word-level.

### Unlisted singing: the star lattice

The aligned text is the lyric sheet, so every frame of the render has to be
explained by a character of it or by `blank`. `blank` is what the acoustic model
calls *silence*, which makes it the wrong filler for a passage where somebody is
plainly singing something the sheet does not contain: an ad-lib, a backing "ah"
pad, an outro chant. There the path pays less by holding a lyric character over
the passage, so the word grows to two or three times its sung length — exactly the
tail `polishLines` then cuts back.

`yue2_ctc_forced_align_star` / `yue2_cursor_align_star`
(`engine/src/yue2/yue2-ctc-align.h`) add one escape symbol whose frame score is
the frame's best letter minus `YUE2_ALIGN_STAR_MARGIN` (1.5 nats, i.e. about a
fifth of the best letter's probability). A star is placed at each end of the
lyrics and between consecutive lines; the path parks garbage wherever the
character it needs scores worse than that ratio, and lyric words still win
wherever they really match. This is the technique pdoom-video's alignment uses
(`analysis/ctcalign.py`), and its notes credit it with the hard spots — the final
chorus buried under a sustained pad in particular.

`POST /yue2/align` uses the star lattice. **`ace-train yue2-align` does not**: it
still calls `yue2_cursor_align`, because the rows that command writes are the AR
trainer's `cursor_words` and the thing this file's header says is gated against
twelve torchaudio fixtures. The two are separate functions for that reason, and
`tools/yue2-ctc-align/test_yue2_ctc_align.cpp` pins the trainer path's rows
against values recorded before the star existed — it runs with g++ alone, no
engine build and no model:

```
g++ -O2 -std=c++17 -o /tmp/test-yue2-ctc tools/yue2-ctc-align/test_yue2_ctc_align.cpp
/tmp/test-yue2-ctc
```

What this does not do is measure the improvement on real audio. The fixture
proves the two lattices behave as designed (a synthetic ad-lib passage moves word
2 from 0.16 s to 0.48 s, and a lyric-only passage is aligned identically); whether
a real chorus with a backing pad gets better is the same ear-and-fixture question
as every other number on this page.

## Non-goals

- Real-time or streaming alignment.
- Per-syllable karaoke spans in the stored contract.
- Training or fine-tuning an aligner.
- Replacing the editor's interaction model. It gains one action and one badge.
- Fixing the backend's own generation-time alignment. That is compared, not
  repaired, here.
