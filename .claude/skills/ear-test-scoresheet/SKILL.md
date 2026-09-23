---
name: ear-test-scoresheet
description: The standard way to run a listening test in HOT-Step - a published score-sheet page (template.html) where Rob plays each render in the browser, scores it 1-5 on named criteria, and the page charts the two score groups by rung so the point where likeness and quality cross is visible. Use whenever renders need judging by ear - checkpoint ladders, recipe A/B tests, sampler or quant comparisons - and whenever you are about to ask Rob to "listen to these files and tell me".
---

# Ear-test score sheet

Rob's verdict (2026-09-23): "the best way so far we've had to score tests like
this". Use it instead of asking for free-text impressions of a folder of WAVs.

What it gives you:
- Every render plays inside the page. Rob does not open a file browser.
- Scores 1-5 on named criteria, a Keep / Borderline / Over the line verdict, a
  notes box per render, and a pick per group (the "last good rung").
- A chart per group that averages the criteria into two lines, one for what
  you're pushing for (likeness) and one for what breaks (quality), plotted by rung.
- Everything saves to the page's own database, which you read back with
  `ArtifactData`. Rows appear live as you attach audio, so Rob can start
  scoring while the rest are still rendering.

`template.html` in this folder is the page. It was used for the YuE2 two-stop
calibration (planner and decoder sweeps on three albums) and republished from
this exact file.

## Running a test

### 1. Make the page

1. Copy `template.html` to a new file in your scratchpad, one file per study
   (a new file path = a new artifact and a clean database).
2. Edit `<title>` and the `CONFIG` block at the top of the script. Nothing else
   should need changing:
   - `criteria`: `{key, name, desc, series}`. Keep two series, "pushing for" and
     "what breaks". Six criteria (three each) scored in ~40 s per render for
     Rob; more gets tiring.
   - `roundNames` / `axis`: labels per round number; `pickLabel` for the picker.
3. Publish with the Artifact tool, `capabilities: {"db": {}, "assets": {}}`.
   Load the `artifact-capabilities` skill first if this session has not.

### 2. Seed one row per render

`ArtifactData` batch of `set` writes to collection `tracks`, doc id like
`r1-<group>-NN`:

```json
{"round": 1, "group": "greenday_dookie", "order": 2,
 "label": "Planner 50", "sublabel": "decoder 100", "x": 50,
 "file": "02-ar050-nar100.wav", "folder": "D:\\...\\round1-planner\\greenday_dookie"}
```

- `group` becomes a tab; `x` is the chart position; `order` sorts rows.
- Add `"reference": true` for a control render (base model, no adapter). It is
  listed and scorable but stays off the chart and out of the picker. Always
  include one: it anchors the scale (the base scored 1 on likeness and 5 on
  quality every time).
- Seed rows before the audio exists: they show "Rendering…" and fill in live.

### 3. Attach audio as it renders

The Artifact asset upload refuses `.wav` and `.mp3`. It takes `.mp4`, and an
audio-only AAC MP4 plays in `<audio>` in every browser:

```
ffmpeg -v error -y -i in.wav -vn -codec:a aac -b:a 256k -movflags +faststart out.mp4
```

A 3-4 minute song comes out at 5-12 MB (the limit is 20 MiB; a float WAV is ~80 MB).

Then, per batch of finished files:
1. `Artifact` publish with `url`, `asset: true`, `file_paths` (up to 25 per call).
2. `ArtifactData` batch of `update` writes adding `{"url": "/_blob/<id>", "asset": "<id>"}`
   to each track. **Pin every entry with `if_version`**: an unpinned write to a
   document that already exists refuses the whole batch. Freshly seeded rows
   are version 1.

A background loop that converts each WAV once its size stops changing keeps
the MP4s ready. Upload in batches as they appear.

### 4. Read the results

`ArtifactData` `list` (or `query` on `track >= "r2-"`) on collection `scores`,
limit 100. Each doc is `{<criterion>: 1-5, verdict?, note?, track}`. Picks live
in `picks` (`r<round>-<group>`: `{lastGood, note}`).

**Scores alone are enough.** Rob skipped verdicts and picks on the first study
and the criteria still answered the question. Average each series per rung,
lay the groups side by side, and read the trend, not single rungs.

## What the first study taught (read before interpreting)

- **Renders are not reproducible from the seed.** The same checkpoints and
  seed through `/api/generate` gave different songs of different lengths.
  Every rung is a fresh take, so plan two renders per rung when a decision
  rests on it.
- **Rob's scoring noise is about ±1 per criterion.** Measured by putting the
  same combination in both rounds without saying so. Treat a difference under
  ~0.5 on a series average as noise. A single low rung between two good ones
  is noise, not a line.
- **Keep automated metrics off the page.** They bias the ear. Compare them
  afterwards; the forced-aligner diction score correlated only 0.37 with Rob's
  diction scores.
- **Note boxes carry the surprises** ("there are no vocals in this track").
  Read every note.

## Rendering ladders for YuE2

`server/scripts/yue2-ladder.mjs <config.json>` renders planner (AR) and decoder
(NAR) checkpoints from different steps through the app's own generate path,
scores diction with `/yue2/align`, writes numbered WAVs plus `results.jsonl`,
and restores the user's adapter picks at the end. It skips WAVs that already
exist, so re-running after a failure only fills the gaps. Config:
`{runDir, outDir, caption, lyrics, seed, pairs: [[arStep, narStep], ...]}`;
`[0, 0]` is the base-model reference.

Stage outputs under `_experiments/_LISTENING/<date>-<study>/<round>/<group>/`
(see the listening-hub convention).

## Traps

- `/api/generate` needs a bearer token: `GET /api/auth/auto` returns one.
  `/api/backends/models` does not, so a script can change the picks and then
  fail to render. The ladder script handles both.
- Editing `server/src` restarts the dev server and kills any running training
  batch. Put tools in `server/scripts/` (tracked, not watched); `tools/yue2-*`
  is gitignored.
- Watch free space on D: before a long ladder. A full disk cut one render off
  mid-write (a 0-byte WAV); delete it and re-run the ladder.
- Prompts: use Lyric Studio lyrics the adapter never trained on, with their
  own caption (`generations` table in `server/data/hotstep.db`: `caption`,
  `lyrics`). The same prompt and seed for every rung of a group.
