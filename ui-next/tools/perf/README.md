# Measuring the page's speed

The performance pass's measurements, kept so they can be taken again: every action a user does, on three saved desks
(empty, typical, full), with real input, timed as the browser times an interaction (interaction to next paint), at
normal and at four times slower CPU; the load from disk; the CPU at rest and during a run; memory over a long session;
and what each motion costs. [09](../../docs/09-architecture.md), "Rendering", says what the page does to be fast and
what the numbers were.

Targets: every interaction paints its response within 100 ms on a full desk; a drag holds 60 fps with no long task; the
page is usable within 1 s of opening with a full saved desk; no measurable CPU at rest, little during a run, no growth
of memory.

```
npm run perf                                   builds this checkout and measures everything (about 45 min)
npm run perf -- --label after --build <dir>    measures a build kept elsewhere (a folder holding dist-file/ and dist-test/)
npm run perf -- --quick                        one run of each (about 15 min)
npm run perf -- --skip throttle,memory         leave some out (perf, throttle, load, idle, memory, motion; --trace adds trace)
node tools/perf/summarize.cjs --compare tools/perf/results/<a> tools/perf/results/<b> [out.md]   one table, a against b
```

Results go to `tools/perf/results/<label>/` (ignored by git): `perf.json`, `perf-throttle4.json`, `load.json`,
`load-throttle4.json`, `idle.json`, `memory.json`, `memory-nogen.json`, `motion.json`, `machine-load.jsonl` and
`summary.md`. Timings run at normal priority and change with what else the machine runs: compare two builds measured
one after the other (keep the "before" build's `dist-file/` and `dist-test/` in a folder and pass it with `--build`).
Playwright is found as the other tools find it (`npm install`, or `PLAYWRIGHT_MODULE`); Chromium runs on the GPU
(`PERF_SOFT=1` for software compositing).

## The scripts

| Script | What it measures |
| --- | --- |
| `perf.cjs` | every action of `actions.cjs`, each in a fresh page on a saved desk: the interaction's duration (Event Timing), input to next frame, main-thread time, layouts, long tasks, frames during a drag. `--throttle 4`, `--runs`, `--desks`, `--only` |
| `load.cjs` | from navigation to usable (the end of the last long task before a second without one), first paint, main-thread time |
| `idle.cjs` | main-thread time and renderer CPU at rest, after a click, and during a run of the simulated engine |
| `memory.cjs` | the heap, nodes and listeners over a mixed session of every action (`--no-generate` leaves out the songs, which the page keeps on purpose) |
| `trace.cjs`, `layout-detail.cjs` | what an action laid out and restyled, and which script forced it |
| `profile.cjs`, `load-profile.cjs` | where an action's (or the load's) time goes, by source line, with the page's counters and React's renders, on the instrumented build: `npx vite build --mode file --config tools/perf/vite.instr.mjs`, then `PERF_BUILD=tools/perf/results/instr` |
| `solve-bench.ts` | the desk's layout solver on the recorded desks (`node tools/perf/solve-bench.ts`) |
| `make-desks.cjs` | writes `desks/*.json`, the saved desks the others open (from the test build) |
| `../motion/cost.mjs` | what each motion costs the main thread (package 9's) |

## The guards

`tools/behaviour/scenarios/39-perf.mjs` (in the acceptance, `acceptance/perf.json`) does the actions whose cost the
pass removed, as a user does them on a full desk, and holds what the page counted to a limit (`src/app/tally.ts`,
counted in the test build only): the models asked again for a knob's change, an Undo's notifications of the store, the
tiles measured for a removal, a drag and a move by key, the layouts solved for a move by key, the shell drawn again.
`switches/perf.json` brings each cause back in turn, and each fails a guard (`npm run switches -- --package "^perf$"`).
