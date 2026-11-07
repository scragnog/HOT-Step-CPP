---
name: ear-test-scoresheet
description: The standard way to run a listening test in HOT-Step - a local HTML score sheet next to the renders where the user plays each track, scores it 1-5 on named criteria, and the page charts the two score groups by rung so the point where likeness and quality cross is visible. Use whenever renders need judging by ear - checkpoint ladders, recipe A/B tests, sampler or quant comparisons - and whenever you are about to ask the user to "listen to these files and tell me".
---

# Ear-test score sheet

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

The user's verdict (2026-09-23): "the best way so far we've had to score tests like
this". Use it instead of asking for free-text impressions of a folder of WAVs.

What it gives you:
- Every render plays inside the page, straight from disk (WAV is fine).
- Scores 1-5 on named criteria, a Keep / Borderline / Over the line verdict, a
  notes box per render, and a pick per group (the "last good rung").
- A chart per group that averages the criteria into two lines, one for what
  you're pushing for (likeness) and one for what breaks (quality), plotted by rung.
- Rows for renders that don't exist yet say "Rendering…" and fill in on their
  own when the file appears, so the user can start while the rest render.
- Scores save to `scores.json` beside the page, which is how you read them.

Files in this folder:
- `template.html`: the page (local file; the default).
- `make-scoresheet.mjs`: writes `index.html` from a `study.json`, splicing in the
  shared save logic from `tools/listening-save.mjs`.
- `template-artifact.html`: the claude.ai-hosted variant, only for when the user
  can't open files on this machine (see the end).

## Running a test

### 1. Lay out the study folder

Put renders under `_experiments/_LISTENING/<date>-<study>/` (the listening-hub
convention), e.g. `round1/<group>/NN-<label>.wav`. The page lives at the top
of that folder and refers to tracks by relative path. Keep the folder name to
letters, digits, `.`, `_` and `-`: the app serves only names of that shape.

### 2. Write study.json and generate the page

```json
{
  "id": "yue2-decoder-long-2026-09-23",
  "title": "YuE2 Decoder Ladder",
  "intro": "Play each render, score it 1–5 on each criterion. Scores save as you go.",
  "criteria": [
    {"key": "voice", "name": "Voice", "desc": "the singer sounds like the artist", "series": "likeness"},
    {"key": "writing", "name": "Songwriting", "desc": "melodies, hooks and structure feel like theirs", "series": "likeness"},
    {"key": "sound", "name": "Sound", "desc": "guitar, drum and production tone match the album", "series": "likeness"},
    {"key": "diction", "name": "Diction", "desc": "every word intelligible, nothing garbled", "series": "quality"},
    {"key": "audio", "name": "Audio", "desc": "clean: no hiss, phasing, crackle or clipping", "series": "quality"},
    {"key": "coherence", "name": "Coherence", "desc": "holds together, no loops, ends properly", "series": "quality"}
  ],
  "series": {"likeness": {"name": "Likeness", "color": "var(--accent)"},
             "quality":  {"name": "Quality",  "color": "var(--warn)"}},
  "roundNames": {"1": "Round 1 · decoder sweep"},
  "axis": {"1": "Decoder step"},
  "pickLabel": "Last good rung",
  "tracks": [
    {"id": "r1-a-01", "round": 1, "group": "album_a", "order": 1,
     "label": "Base", "sublabel": "no adapter", "reference": true, "file": "round1/album_a/01-base.wav"},
    {"id": "r1-a-02", "round": 1, "group": "album_a", "order": 2,
     "label": "Decoder 300", "sublabel": "planner frozen at 240", "x": 300, "file": "round1/album_a/02-nar300.wav"}
  ]
}
```

```
node .claude/skills/ear-test-scoresheet/make-scoresheet.mjs <study-folder>/study.json
```

- `group` becomes a tab; `x` is the chart position; `order` sorts rows.
- The six criteria above (three per series) are the proven set: about 40 s of
  scoring per render. Keep two series, "pushing for" and "what breaks".
- `reference: true` marks a control (base model, no adapter): scorable, but
  off the chart and out of the picker. Always include one; it anchors the scale.
- Repeated takes of one rung are separate tracks with the same `x`; the chart
  averages them into one point.
- List every planned render up front, rendered or not. To add a round later,
  append tracks and re-run the generator. Scores are keyed by track id, so
  they survive.

### 3. Hand it to the user

The user opens `index.html` from the folder (`file://`); hand over the
folder path, not a URL. A browser page can't write a file on its own, so:

- HOT-Step running: the page finds the app's copy at
  `http://localhost:3001/listening/<folder>/` and moves itself there, carrying
  any scores already kept in the browser. The app serves study folders from the
  checkout's `_experiments/_LISTENING/` or its parent's, and every change POSTs
  to `/listening/<folder>/scores`, which writes `scores.json` beside the page:
  no dialog, no download. The route answers loopback requests only and refuses
  foreign origins (`server/src/routes/listening.ts`).
- HOT-Step not running: on first use the user clicks **Save scores to a
  file…** and saves `scores.json` next to the page. After that every change
  writes to it; the browser remembers the file and asks once per session to
  reconnect. Browsers without file saving (Firefox) get **Export scores**
  instead: the user exports and drops `scores.json` in the folder.

Scores are also kept in the browser either way.

### 4. Read the results

`scores.json`: `{scores: {<track id>: {<criterion>: 1-5, verdict?, note?}},
picks: {"r<round>-<group>": {lastGood, note}}}`.

**Scores alone are enough.** On the first study the user skipped verdicts and picks
and the criteria still answered the question. Average each series per rung,
lay the groups side by side, and read the trend, not single rungs.

## What the first study taught (read before interpreting)

- **Renders are not reproducible from the seed.** The same checkpoints and
  seed through `/api/generate` gave different songs of different lengths.
  Every rung is a fresh take, so plan two renders per rung when a decision
  rests on it.
- **The user's scoring noise is about ±1 per criterion.** Measured by putting the
  same combination in two rounds without saying so. Treat a difference under
  ~0.5 on a series average as noise. A single low rung between two good ones
  is noise, not a line.
- **Keep automated metrics off the page.** They bias the ear. Compare them
  afterwards; the forced-aligner diction score correlated only 0.37 with the
  user's diction scores.
- **Note boxes carry the surprises** ("there are no vocals in this track").
  Read every note.

## Rendering ladders for YuE2

`server/scripts/yue2-ladder.mjs <config.json>` renders planner (AR) and decoder
(NAR) checkpoints from different steps through the app's own generate path,
scores diction with `/yue2/align`, writes numbered WAVs plus `results.jsonl`,
and restores the user's adapter picks at the end. It skips WAVs that already
exist, so re-running after a failure only fills the gaps. Config:
`{runDir, outDir, caption, lyrics, seed, pairs: [[arStep, narStep], ...], takes?}`;
`[0, 0]` is the base-model reference. `takes` above 1 renders each pair that
many times (seed, seed+1, ...) as `-t1`, `-t2` files. It needs the app running
with YuE2 as the active backend, and talks to `HOTSTEP_URL` (default
`http://localhost:3001`) and the engine at `ACE_URL` (default
`http://127.0.0.1:8085`).

## Traps

- `/api/generate` needs a bearer token: `GET /api/auth/auto` returns one.
  `/api/backends/models` does not, so a script can change the picks and then
  fail to render. The ladder script handles both.
- Editing `server/src` restarts the dev server and kills any running training
  batch. Put tools in `server/scripts/` (tracked, not watched); new files under
  `tools/` are gitignored unless `.gitignore` whitelists them.
- Watch free space on D: before a long ladder. A full disk cut one render off
  mid-write (a 0-byte WAV); delete it and re-run the ladder.
- Prompts: use Lyric Studio lyrics the adapter never trained on, with their
  own caption (`generations` table in `<data root>/library/hotstep.db`: `lyrics`
  plus the caption for the backend under test, `caption` for ACE-Step,
  `caption_yue2` for YuE2, `caption_mm3` for MM3). The same prompt and seed for
  every rung of a group.

## Hosted variant (only when needed)

`template-artifact.html` is the same page published as a claude.ai artifact,
for scoring away from this machine. It keeps rows and scores in the artifact's
database (`capabilities: {"db": {}, "assets": {}}`; config in a `CONFIG` block
at the top of its script; rows seeded with `ArtifactData` into `tracks`).
Audio must be uploaded: the asset store refuses `.wav`/`.mp3` but takes an
audio-only AAC `.mp4` (`ffmpeg -i in.wav -vn -codec:a aac -b:a 256k -movflags
+faststart out.mp4`), then each track gets `{url: "/_blob/<id>"}` via a pinned
(`if_version`) `ArtifactData` batch update. Much more work than the local page.
