# Frontend reference client

A headless client that drives the real Node routes over HTTP, proving the contracts
documented in [`docs/dev/build-a-frontend.md`](../../../docs/dev/build-a-frontend.md) are
enough to build a replacement frontend without importing `ui/src` or any execution service.

`client.ts` imports shared contracts only (`server/src/contracts/`), talks HTTP, and has no
import from `ui/src` or a backend/execution service, including transitively. `fakeServer.ts`
mounts the actual production route factories against an isolated temp database and data
directory, with a fake engine or fake runner standing in for whatever would otherwise spawn a
real process, download a model, or call a remote worker. Node's own route/service code is
never faked — only what it would talk to outside the process.

## Running it

```
cd server
npx tsx --test --test-force-exit scripts/frontend-reference/*.test.ts   # everything
npx tsx --test --test-force-exit scripts/frontend-reference/client.test.ts   # one file
npx tsc --noEmit -p scripts/tsconfig.json   # typecheck the scripts package
```

`--test-force-exit` is required: an open handle somewhere in this harness or its production
dependencies keeps the Node test runner's process alive after every test has already passed.
It was timeboxed once and not found; `--test-force-exit` is the workaround, not a fix. If you
narrow it down, say so in your commit and remove the flag.

Run the new/changed file alone first, then combined with `client.test.ts` and
`safetyGuards.test.ts` (shared-file changes can affect either), then the full list above, then
the server's own full suite — adding a reference test does not excuse running only the new
file.

## Files

| File | Is |
|---|---|
| `client.ts` | The reference client. Shared contracts only, HTTP, no `ui/src` or service imports |
| `fakeServer.ts` | Mounts the real route factories on an isolated temp DB/data dir; starts the fake engine; installs the safety guards before any production import |
| `safetyGuards.ts` | Fail-closed subprocess/network guards. A blocked call still records a violation even if the caller catches the thrown error |
| `fixtures.ts` | Disposable inputs: a valid fixture WAV, generic upload helpers |
| `trainingFixtures.ts` | Files the training start routes check for on disk (trainer binary, base model files), never executed or loaded as a model |
| `fakeEngine.ts` | A fake `ace-server`: just enough of its wire protocol for Create, Insta-Gen, Cover, Repaint, Lego and stem separation |
| `fakeMm3Engine.ts` | The MiniMax-Music3 half of the fake engine: `/mm3/*`, including the live window stream |
| `fakeTrainingRunner.ts` | Stand-ins for the trainer runners, installed with `labelingQueue.ts`'s `overrideTrainingRunner`; everything around them (routes, queue, SSE, job records) is real |
| `restartChild.ts` | One life of the restart-interruption harness; run as a real child process by `client.restart.test.ts`, not imported |
| `client.test.ts` | Create (direct submit, and resolve then the durable audio queue), Insta-Gen, Cover (open only), Repaint, Lego, stem separation |
| `client.library.test.ts` | Song Builder, Library, Playlists, studio drafts, presets, import/export, settings/backends |
| `client.training.test.ts` | ACE, MM3 and YuE2 training: dataset, revision, recipe resolve, start |
| `client.trainingRuns.test.ts` | A training start that runs to a terminal state, per family, plus cancel and the one-job-per-dataset refusal |
| `client.cover.test.ts` | Cover Studio's full chain: open, caption, transcribe, approve, render |
| `client.lyric.test.ts` | Lyric Studio batch refusals: auth failure, rejected input, per-item preflight errors, an idempotency conflict, cancelling a finished or unknown job |
| `client.lyric.chain.test.ts` | Lyric Studio's full chain (fetch, profile, generate, refine, render through the audio queue), with Genius and the LLMs as fixtures through `lyricWorkflow.ts`'s `overrideLyricWorkflowDeps` |
| `client.streaming.test.ts` | STORM streaming/recording: start, control, record, export, disconnect; stream-session and MM3 stream refusals |
| `client.mm3stream.test.ts` | The live MM3 stream specifically (makes the MM3 backend active) |
| `client.restart.test.ts` | A training job, a workflow job and a YuE2 preparation pipeline still running when the harness is killed, read back by a second process on the same data dir |
| `safetyGuards.test.ts` | Proves the guard mechanism itself: a violation is recorded even when production code catches the thrown error |

## Safety

Nothing mounted here may spawn a real process or reach a real network endpoint. The guards in
`safetyGuards.ts` are installed before any production module is imported, so every route's own
`execFile`/`fetch` call resolves to the guarded version. A blocked call is recorded in
`FakeServer.violations` before it throws, so it stays visible even if production code catches
the resulting error and degrades gracefully — the exact shape `essentiaClient.ts` and
`workerUpdate.ts` use. Every `client.*.test.ts` file asserts that no violation was recorded:
its last test checks `server.violations`, and `client.restart.test.ts` checks each harness
life's report.

Read-only Git `rev-parse` and `status` calls receive fixed fixture output through
both `execFileSync` and callback-based `execFile`, including promisified callers.
They never launch Git. Other subprocess calls remain blocked. The fake engine
confirms `/yue2/unload` with `unloaded: true` and zero freed memory, allowing the
real residency arbiter to switch families without allocating GPU memory.
LM results include fixture audio codes for the real planner validation. Data-root
and install-root overrides point into fixture directories, and worker GPU
selection is empty. The exact GPU-discovery and training-capacity subprocess calls
receive empty fixture results, so preflight sees no NVIDIA cards without probing hardware.
ACE's `/models/loaded` likewise returns an empty resident list. Trainer fixtures
include the PiSSA initialization file checked before MM3 training starts. The
harness calls production training-job recovery explicitly at startup, as `index.ts`
does, so restart tests reconcile interrupted training records.

A route that cannot be isolated without a production change is reported as blocked, with the
exact call site and a proposed dependency seam, in the test file's own header comment — never
worked around with a fake response standing in for the route being qualified. The currently
blocked seams are listed in `RESEARCH/FRONTEND_DECOUPLING_BATCH7_CLOSURE.md` (workspace, not
this repo).

## Adding coverage

- One `fakeServer` per test **file**, in `test.before`/`test.after` — module-level singletons
  in route code (in-memory job maps, queue pumps) are shared across every request a server
  receives, exactly as in the real app, so splitting one workflow across two servers would
  hide state a real client sees.
- A new fake adapter (another engine behaviour, another runner) goes in this directory, named
  for what it fakes, and documents in its own header which production call sites it stands in
  for and which it leaves real.
- `client.ts` is shared by every test file: get sign-off before editing it if more than one
  slice is touching it, and add new methods as one block rather than interleaving with
  existing ones.
- Exercise the actual documented HTTP sequence for the workflow (see the matching
  `docs/dev/frontend-*.md` page), including its stale-revision, rejected-input, cancellation,
  per-item-failure, auth-failure and restart-interruption variants where they apply — not just
  the happy path.
