# Windows portable updater acceptance

Builds real portable packages in a disposable clone, then starts the shipped
protocol-1 `HOT-Step.bat` on isolated ports and drives the packaged UI/API.
This changes no production source. It deliberately kills its own process trees
and corrupts disposable state. Never point it at an existing user installation.

Run from PowerShell at the repository root:

```powershell
pwsh -NoProfile -File tools/update-gate/build-fixtures.ps1 -Reuse
node tools/update-gate/suites.mjs
node tools/update-gate/run.mjs all
node tools/update-gate/report.mjs
```

The default disposable root is `D:\Ace-Step-Latest\hotstep-1148-fixture`.
For a different location, pass `-FixtureRoot` to the builder and set
`UPDATE_GATE_ROOT` for the Node scripts. Reusing a clone requires `-Reuse` and
fast-forwards it to `-SourceCommit`; it never resets a checkout. Each acceptance
case refuses an existing case directory. The current batch defaults to `round2`
and clone-only tags `v92.0.0`, `v92.0.1` and `v92.0.2`. Set builder `-Batch` and
`-VersionMajor`, and matching `UPDATE_GATE_BATCH` and `UPDATE_GATE_MAJOR`, for a
new batch. `UPDATE_GATE_CASE_SUFFIX` isolates a corrected harness retry. The third package uses the default
protocol. No tags are pushed or created in the live checkout.

Requires installed Playwright and Chromium. Set `UPDATE_GATE_PLAYWRIGHT` and
`UPDATE_GATE_CHROMIUM`, or the helper finds cached Windows installations.
The server uses port 13948 and the engine 18948; both must be free. Metadata-only
GGUF files allow the real engine to scan and answer health without loading
weights. The CPU package reuses native binaries with `-SkipEngine`.

Select cases with a comma-separated argument instead of `all`: `clean`,
`failed-health`, `no-models`, `both-no-models`, `before-pointer`, `after-pointer`, `mid-restore`, `after-commit`, `committed-missing`,
`busy-lock`, `protocol-zero`, `recovery-no-server`. Run suites once per batch.
Logs, row snapshots, receipts, screenshots and summaries go to ignored
`evidence/<batch>/`. A nonzero exit is a failed acceptance case, including a reproduced
product defect. Retain that evidence when handing findings to an author.

`preload.mjs` replaces only the fixture release URLs, supplies a slow response
for one model-download job, and freezes filesystem operations at the selected
crash boundary. The controller records the exact process tree and checks the
Multica daemon PID before killing it. Its database comparisons exclude only
the changing `server_instances` heartbeat table. The full rows remain in logs.

The failed-health case requires a missing candidate engine to roll back promptly.
Both empty-model variants must install. Committed recovery must preserve data
and serve the app even if its engine is now absent. Clean and rollback cases
verify the dialog survives a real disconnect, ignores a stale receipt, and waits
for manual Reload. The no-server case waits five real minutes for launcher
guidance after a damaged snapshot. CUDA packages and subjective UI approval are
outside this CPU-only batch.

For the focused third run, build with `-SourceCommit 1ced828d7273b01c9269b8f74f5334c6fc55845c
-VersionMajor 93 -Batch round3 -Reuse`. Set `UPDATE_GATE_BATCH=round3`,
`UPDATE_GATE_MAJOR=93`, and `UPDATE_GATE_RECOVERY_SHORT=1`, then run
`node tools/update-gate/run.mjs committed-missing,recovery-no-server`.
The short recovery case verifies the closing text, saves `updater/` aside,
changes only the pending transaction state to committed, and records the result
and remaining leases. This is a maintainer experiment on a disposable fixture,
not a user recovery instruction. It does not repeat the proven five-minute wait.

`build-owner-kit.ps1` packages the CUDA 13.1 pair using existing native binaries
and prepares an isolated owner kit without starting it. Run after fixture builds
and before suites, because packaging temporarily rebuilds the clone's SQLite
addon. `verify-owner-kit.mjs` checks both archive identities and the feed seam
without launching the kit. The kit's README gives the manual steps. Keep the
fixture and owner kit until the owner check is complete.
