# Wrapper Resilience — Progress Log (HOT-Step-CPP)

Branch: `feat/wrapper-resilience-hotstep` (off `origin/master`)
Companion branch: `orinmi-New feat/wrapper-resilience-runpy` (off `origin/fresh_build`)

This repo is third-party (github.com/scragnog/HOT-Step-CPP). We patch locally
on this dev box. Commits stay on the local feature branch. We do NOT push to
`scragnog/HOT-Step-CPP`. Whether to submit a courtesy PR upstream is the
user's call after review.

## Failure context (Jun 6 production incident)

Cold-start LoKr precompute in the C++ engine takes ~17s. During that window
the httplib-backed ace-server (single-threaded ThreadPool, port :8085) stalls
and does not respond to status polls. The node wrapper's `pollUntilDone`
fetches the status exactly once with no retry, sees `fetch failed`, marks the
wrapper job failed, and returns an error to its caller.

The caller is the orinmi-New worker code path:

```
Studio /api/music/render
  -> Supabase queue
    -> orinmi-New worker (apps/ose/worker-supabase.py)
      -> apps/ose/acestep/run.py
        -> HOT-Step-CPP node wrapper :3001
          -> ace-server (C++) :8085
```

`run.py:249` raises on the wrapper error. The Supabase render row transitions
to `failed`. The C++ engine, meanwhile, keeps running cleanly, finishes the
job, and writes audio that nobody collects.

## 3-component plan (user-chosen scope)

- **C1 — apps/ose/acestep/run.py resilience (companion repo, production unblock):**
  Classified retry on transient wrapper errors, plus engine-direct reconcile
  via :8085 so we don't lose work the C++ engine actually produced.

- **C2 — HOT-Step-CPP engine C++ patch (THIS REPO):**
  - Cancellable LoKr precompute.
  - `JobPhase` enum.
  - New `/warm`, `/jobs` endpoints.
  - Honor `?keep_loaded=1` and `--keep-loaded` engine flag forwarded by the
    wrapper.
  - Eliminates the 17s cold-start cost so the stall stops happening at all.
  - **DO NOT run a full cmake build in a workflow agent** (5+ minutes, blocks
    the workflow). Configure-only (`cmake -S engine -B engine/build`) is fine;
    actual compile is the user's call.

- **C3 — HOT-Step-CPP node wrapper patch (THIS REPO, tiny):**
  - Expose `ace_job_id` and `ace_phase` in `GET /api/generate/status/:id`.
  - ~3 lines of TS. Lets companion `run.py` correlate wrapper jobs to engine
    jobs cleanly when reconciling via :8085.

## User decisions

- **No upstream push.** Local-only feature branch. Courtesy PR is the user's
  decision after review.
- **No daemon restart.** Do NOT restart `hotstep-server.service` or
  `acestep-server.service` during this work — would destabilize the running
  smoke-test environment.
- **Ultracode mode.** Subagents must branch from the correct base, never
  commit to integration branches directly, and never use `--no-verify` /
  `--no-gpg-sign`.
- **Commit hygiene.** No "Co-Authored-By: Claude" / no Anthropic mention in
  commit messages. Plain subjects, author defaults to repo config.
- **Pre-existing dirty files in this repo** (`server/package-lock.json`,
  `ui/package-lock.json`) are NOT to be staged or committed by this work.
  Same for the untracked `AUDIT_FIX_PROGRESS.md`.

## Decisions log

<!-- Subsequent components append dated entries here as work proceeds. -->
