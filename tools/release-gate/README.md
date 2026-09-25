# Release gate

Go or no-go for a HOT-Step build, before the tag. It drives the app through its
HTTP API only, the way the UI does, and never judges audio quality: every render
it makes is staged for the ear test at the end.

```
node tools/release-gate/run.mjs --zip release/out/HOT-Step-CPP-vX.Y.Z-win-x64-cuda.zip   # the real pre-tag run
node tools/release-gate/run.mjs                                                             # attach to the dev app on :3001
node tools/release-gate/run.mjs --tiers 0-3                                                 # the half-hour subset
node tools/release-gate/run.mjs --update-goldens --tiers 7                                  # rewrite the tier 7 references
```

Exit 0 is GO, exit 1 is NO-GO, exit 2 means it could not run (app busy, boot
failed). The summary goes to `logs/release-gate/<stamp>/summary.md`, one log
and one TAP file per tier sit beside it, and the renders land in
`_experiments/_LISTENING/<stamp>-release-gate/` with a score sheet.

## Modes

| Flag | App under test |
|---|---|
| none | Attaches to `--url` (default `http://localhost:3001`). Refuses to start if a generation or training job is running. Preprocess and training will stop and restart the engine, as they do from the UI. |
| `--zip <file>` or `--dir <folder>` | Extracts or uses a release build, boots its bundled Node on `--port` 3199 with the engine on `--engine-port` 18085, and points it at this checkout's `models/` and `adapters/` (`--models`, `--adapters` to change). The data directory is the extracted folder's own, so it starts as a fresh install. Shut down at the end unless `--keep`. |
| `--dev` | Spawns `tsx src/index.ts` from this tree on :3199 with an empty data directory under the run folder. |

Testing the extracted zip is the point: a feature can work here because this
machine holds a file the archive never got (#137, #139). Only the packaged
build with the shipped model catalogue proves a user can run it.

## Tiers

Tiers run in order. Tiers 0, 1 and 2 are gates: if one fails the rest are not
run. Everything else runs to completion so one report shows every failure.

| Tier | What it proves | Blocks |
|---|---|---|
| 0 | `tsc` on server and ui, `verify-hooks.ps1`, the `server/src` unit tests, `check-release-prereqs.mjs` | yes |
| 1 | Health with `engine.ready`; ace, minimax-m3 and yue2 registered; DiT, VAE and LM catalogued; every `.lua` under `engine/plugins/` appears in the registry; thirty read-only endpoints answer 200 | yes |
| 2 | ACE text2music with the LM and lyrics, the skip-LM path, then cover, repaint, extract, lego and complete from that render; the generation log ends with `GENERATION COMPLETED` | yes |
| 3 | Same seed twice reproduces; a non-default solver, scheduler and guidance mode each change the output; a DiT adapter changes the output; an LM adapter loads | yes |
| 4 | MM3 text2music, MM3 with a planner adapter, YuE2 text2music. Each skips when its models are not installed. | yes |
| 5 | analyze, supersep, stem-studio extract, mastering, a post-process re-run, MIDI transcription on a render. Missing binaries and models skip. | yes |
| 6 | A dataset from this run's renders; build, preprocess, one epoch of DiT LoRA then a render with it, one epoch of 0.6B LM LoRA then a render with it; MM3 codes plus five planner steps; YuE2 joint train five steps. Trainers whose models are absent skip. | yes |
| 7 | Fixed-seed renders per backend compared to stored spectral fingerprints in `goldens/` | warn only (`--strict-golden` to block) |

Tier 3 is the one the others cannot replace. The hook in
`pipeline-synth-ops.cpp` that routes through `hot-step-sampler.h` fails silently:
every solver still "works" and every render is the upstream default. Rendering
the same seed with and without the knob and asserting the audio differs is the
cheapest detector, and the same trick proves adapters merge.

## What it does not cover

- Audio quality. The score sheet is the last step of every run, and it is the
  one that matters.
- UI visuals. Type-checking and a served page are all that is mechanical.
- Anything behind an external LLM (Lyric Studio, assistant, cover art). Tier 1
  reports which providers are configured and stops there.
- Multi-GPU, Vulkan and CPU builds. Run the zip for each variant you ship.

## Goldens

`goldens/<case>.json` holds a 48-band log spectrogram at 10 frames a second for
a fixed request and seed, plus the engine defaults at the time. They are
specific to the models installed on the machine that wrote them. Write them once
with `--update-goldens` after a listen, and rewrite them when a model or a
sampler default changes on purpose. A drift below 0.97 similarity is a warning
that names the render to listen to.

## Adding a probe

Each tier is a `node:test` file. `lib.mjs` has the HTTP helpers, `generate()`
(submit, poll, download, remember the render), `assertAudible()` (parses, long
enough, not silent), training-job polling, WAV parsing and the correlation and
fingerprint maths. State shared between tiers is `state.get/set` on the run's
`state.json`. A probe that finds its feature absent calls `t.skip(reason)` and
returns; a probe that finds it broken throws with the server's error text.
