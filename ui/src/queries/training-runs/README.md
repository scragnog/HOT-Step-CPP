# Training run server state

`queries.ts` owns the run, review, sheet and status snapshots. Worker reads
capture the selected worker in their canonical key and pass that worker and
Query's abort signal to the transport. Local run indexes, review records,
previews and worker administration retain their local routes. They omit the
worker scope even when Train on points to a remote machine. SA3 checkpoint
previews live in their own folder, `../sa3-previews/`, but key into this area
as `training-runs/sa3-previews` with the worker captured.

GPU reads use `sharedKeys.gpus()`. Preparation detail and current job use
`sharedKeys.preparation(id)` and `sharedKeys.preparationJob(id)`, sharing the
training-data executor's local snapshots. The dataset-filtered preparation
list keeps its area key. Successful commands, terminal events and reconnects
also invalidate shared preparation snapshots.

`hooks.ts` retains only job/pipeline IDs and reload signals as client state.
Job adoption is guarded by a selection generation, including a switch away
and back to the same worker or dataset. Query owns the job object and polling.
`useActiveTrainingJob()` reads the selected job from that one cache entry; the
store keeps only `activeJobId`, the worker, and the sample/log/preview/metric
reducers. Read cancellation only cancels the HTTP read; it never cancels a
server job.

`events.ts` shares the existing job EventSource with chart/log reducers.
Job, progress and status frames merge through `jobFrame.ts`, the job cache's one
rule; the store keeps no job copy. Terminal events invalidate snapshots once per job and
captured worker, including replayed events. Reconnect invalidates authoritative snapshots. Teardown drops
late frames. Metrics, preview rows received over SSE, form values, selections,
score-note drafts, playback and wall-clock timers remain client state.

Commands run from user actions or the existing explicit workflow executor as
component `useMutation` hooks whose success callback calls `trainingWritePlan`.
A command executes once and invalidates the captured scope after success; it
never retries a submission. `trainingWritePlan` accepts explicit keys for
resources whose canonical key carries no worker scope (the unscoped workers
list, per-dataset rows). Training-store actions retain their published scoped
signatures and their own cache invalidation. Recipe resolution, preset
resolution and preparation context capture remain command-time operations.
Caption-before-training remains an explicit sequential workflow; its job reads
use the canonical training-data job cache rather than a separate transport.

## Consumer disposition

| Consumers | Snapshot owner |
|---|---|
| `useMm3Status`, `useYue2Status`, `useYue2ArStatus` | Worker-scoped Query status, invalidated by job events |
| `TrainPanel` | Phase host with no reads of its own: hands the section to the target's plug-in (`trainingPlugins.tsx`) and, for non-ACE targets, keeps the job stream and run-event bridge |
| `plugins/ace.tsx` | Canonical training-data LM/DiT status queries; cached recipes seed untouched form fields; owns the ACE job stream and run events |
| `Sa3TrainCard` | Worker-scoped `training-data/sa3-status` (built in the component, see tanstack-query.md F4#17) and `training-runs/sa3-previews` (`../sa3-previews/`); local `global-models/sa3-adapters`; audition renders follow `shell-queue/generation-status`; start, cancel, render and audition are `useMutation`s |
| `TrainingPreviewPanel` | Shared checkpoint-preview view for the MM3, YuE2 and SA3 cards and the YuE2 ladder review; renders rows its caller has already read |
| `MonitorPanel`, `Yue2BatchPanel` | Canonical training-data pipeline/batch/job queries; scoped store actions submit commands |
| `Mm3RunsPanel`, `Yue2RunsList`, `Yue2ArRunsList` | Local Query run indexes, refreshed after job transitions |
| `Yue2AitkTrainCard` | Query jobs, preparation defaults, local runs/previews, pipeline/job snapshots, GPU and worker lists |
| `Yue2SheetPreview` | Worker-scoped Query source list/detail; selection and score/audio rendering stay local |
| `Yue2TrainStages` | Local Query preparation records, including the server-owned chain started through the store |
| `RefinePanel`, `Yue2LadderReview`, `ReviewPanel` | Local Query run/preview/review/score/album indexes; captured mutation refresh |
| `Yue2OptimisePanel` | Worker-scoped Query optimise snapshot and job |
| `WorkersPanel` | Canonical local worker list and Query update status per worker |
| `TrainLmForm`, `TrainDitForm`, `Mm3TrainCard`, `Mm3CodesCard`, `Yue2TrainCard`, `Yue2StageCard`, `Yue2ArTrainCard`, `Yue2AitkBatchWizard` | Forms, pure estimates, cached status hooks and scoped training-store actions |
| `TrainingRunStats`, `TrainingChart`, `Mm3PreviewStrip` | Active job via `useActiveTrainingJob` (Query cache); charts, elapsed-time and audio UI stay client-owned |

## Endpoints and cadence for runtime acceptance

Worker-scoped training routes below use `/api/training` on this PC or
`/api/workers/<worker>/api/training` for a captured remote target.
Other routes explicitly remain local.

| Route | Cadence |
|---|---|
| Worker `datasets/:id/mm3`, `yue2`, `yue2-ar`, `train-lm`, `train-dit`, `yue2-optimise`, `yue2-joint-prepare` | 30 s freshness; events, commands and explicit refresh invalidate; adapter input debounce 400 ms |
| Worker `jobs/:id` | 1.5 s while queued/running; stop on done/failed/cancelled |
| Worker `pipelines` | 2 s while running/paused; stop when terminal |
| Worker `yue2-batch` | 3 s while running/paused; stop when terminal |
| Worker `datasets/:id/yue2-sheet` and source detail | List every 5 s while open and live; detail on selection |
| Worker `datasets/:id/sa3-train` | Always fresh (`staleTime` 0); 1.5 s while its job is queued or running |
| Worker `datasets/:id/sa3-previews?run=...` | 1.5 s while a cell is queued or rendering, or the training job is active; a render or retry invalidates that dataset's previews |
| Local `datasets/:id/mm3-runs`, `yue2-runs`, `yue2-ar-runs` | 30 s freshness; job transitions invalidate |
| Local `datasets/:id/yue2-joint-runs` | 5 s while a local run is live or the displayed job is active |
| Local `datasets/:id/yue2-joint-previews?run=...` | 5 s while live; joint training retains the existing 30-minute drain window for parallel previews |
| Local `/api/training/ops/preparation` list/detail/job | 1.5 s while a preparation chain is active; stop when terminal |
| Local `yue2-review` | 30 s while mounted |
| Local `yue2-batch` for Review's finish queue | 10 s while mounted |
| Local rung/album scores and `/api/training/ops/review/ladder/:dataset/:run` | 30 s freshness; successful scoring/render/cleanup invalidates; selection revalidates the ladder |
| Local `/api/workers`, worker `mirror` | Worker list 5 s in Workers; mirror 15 s in Review |
| Local worker update status | 5 s while updating; stop when terminal |
| Local `/api/settings/gpus`, `/api/training/ops/recipes/:family?worker=...` | 30 s freshness |

Polling uses the foundation's capped error backoff, up to 30 s. The terminal
predicates stop polling independently of errors. There are no component-owned
network interval timers in this slice. Runtime counts still require Tester's
baseline comparison; unit tests do not establish a reduction in the live app.

For the final test on port 3000:

1. Open Train with MM3 or YuE2 consumers together. Check one in-flight status
   request per key, then switch away and back within 30 s without another read.
2. Delay a worker A status/job reply, switch to B, then return to A. Check that
   B never shows A's job or defaults and that an old callback cannot adopt a job.
3. Start one job, observe the active cadence, then finish/cancel it. Check that
   job polling stops and the run list and status refresh. Reconnect/replay the
   stream and confirm no repeated Start, Render or audition submission.
4. Select a remote worker, then open Review/Refine and a mirrored run. Verify
   local run, preview, ladder and score URLs. Score a rung and check that the
   review list refreshes; render once and check for one render POST.
5. Start/pause/resume a batch or preparation chain. Check one cache owner for
   each snapshot, fresh rows after commands, terminal stop and stable Follow.
6. Change sheet selection during a delayed read. Check that the old sheet is
   absent while the new one loads. Leave the panel and check stream teardown.

No app or GPU work is authorized for this implementation slice. Worker, render
and download cases remain for the consolidated runtime test. Integration owns
the final updates to the public training pages and the foundation manifest.
