# Windows portable updater acceptance

Builds real portable packages in a disposable clone, then starts the shipped
protocol-1 `HOT-Step.bat` on isolated ports and drives the packaged UI/API.
This changes no production source. It deliberately kills its own process trees
and corrupts disposable state. Never point it at an existing user installation.

Run from PowerShell at the repository root:

```powershell
pwsh -NoProfile -File tools/update-gate/build-fixtures.ps1
node tools/update-gate/suites.mjs
node tools/update-gate/run.mjs all
node tools/update-gate/report.mjs
```

The default disposable root is `D:\Ace-Step-Latest\hotstep-1148-fixture`.
For a different location, pass `-FixtureRoot` to the builder and set
`UPDATE_GATE_ROOT` for the Node scripts. The builder refuses an existing clone;
each acceptance case refuses an existing case directory. Clone-only fixture
tags are `v91.0.0`, `v91.0.1` and `v91.0.2`. The third package uses the default
protocol. No tags are pushed or created in the live checkout.

Requires installed Playwright and Chromium. Set `UPDATE_GATE_PLAYWRIGHT` and
`UPDATE_GATE_CHROMIUM`, or the helper finds cached Windows installations.
The server uses port 13948 and the engine 18948; both must be free. Metadata-only
GGUF files allow the real engine to scan and answer health without loading
weights. The CPU package reuses native binaries with `-SkipEngine`.

Select cases with a comma-separated argument instead of `all`: `clean`,
`failed-health`, `no-models`, `before-pointer`, `after-pointer`, `mid-restore`, `after-commit`,
`busy-lock`, `protocol-zero`, `recovery-no-server`. Run suites once per batch.
Logs, row snapshots, receipts, screenshots and summaries go to ignored
`evidence/`. A nonzero exit is a failed acceptance case, including a reproduced
product defect. Retain that evidence when handing findings to an author.

`preload.mjs` replaces only the fixture release URLs, supplies a slow response
for one model-download job, and freezes filesystem operations at the selected
crash boundary. The controller records the exact process tree and checks the
Multica daemon PID before killing it. Its database comparisons exclude only
the changing `server_instances` heartbeat table. The full rows remain in logs.

The failed-health case requires a missing candidate engine to roll back. The
no-server case requires the updater overlay to survive the global disconnect
screen. Those assertions caught defects in the original acceptance run.
