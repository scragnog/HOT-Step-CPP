# Training for frontend clients

This page lists every training operation a client calls and the order it calls them in, from reading a dataset to starting a run and reviewing its checkpoints. The types are in the shared contracts: [`trainingOperation.ts`](../../server/src/contracts/trainingOperation.ts) (the command envelope), [`trainingRecipes.ts`](../../server/src/contracts/trainingRecipes.ts), [`trainingPreparation.ts`](../../server/src/contracts/trainingPreparation.ts), [`trainingReview.ts`](../../server/src/contracts/trainingReview.ts), [`trainingStarts.ts`](../../server/src/contracts/trainingStarts.ts) (start routes, request builders and response shapes) and [`training.ts`](../../server/src/contracts/training.ts) (training targets and the SA3 recipe, asset and checkpoint-preview shapes). Every path is listed in the [route index](api.md); the operation routes are in its `/api/training/ops` section. How the trainers themselves work is in [training-internals.md](training-internals.md).

Training routes need no auth token, except the two audition-draft routes, which answer 400 without one. Reads change nothing; the start, preparation and review commands change state.

Training Studio trains for one of four targets, the backend ids in `TRAINING_TARGET_IDS`: `ace`, `yue2`, `minimax-m3` and `sa3`. The target is the client's choice; Node stores none and never reads one. ACE trains with the `ace-lm` and `ace-dit` recipe families, MM3 with `mm3-lm`, YuE2 with `yue2-nar`, `yue2-ar` and `yue2-joint`, and SA3 through its own start route ([below](#starting-sa3-training)). The bundled UI holds its picker on the target of a queued or running job, read from the job's `kind`: `sa3-train` is SA3; `preprocess`, `train-lm` and `train-dit` are ACE; kinds starting `mm3-` are MM3 and `yue2-` YuE2.

## Two kinds of route

**Operations** live under `/api/training/ops/<domain>`, where the domain is `recipes`, `preparation` or `review`. Each command is an envelope:

```json
{
  "version": 1,
  "operation": { "kind": "recipe:ace-dit", "idempotencyKey": "6f1c…" },
  "worker": { "kind": "local" },
  "dataset": { "id": "<dataset id>", "revision": "<as read>" },
  "sources": [],
  "payload": { }
}
```

`worker` is `{ "kind": "local" }` or `{ "kind": "remote", "name": "<worker>" }`. Once a command is accepted, it keeps that worker whatever the user selects later. `dataset` and each `sources` entry carry the revision the client read. Revisions are opaque: compare them, never parse them. Operations always go to this machine, even when the worker is remote; Node talks to the worker itself.

Every operation refusal has the body `{ error, reason?, currentRevision?, issues? }`:

| Status | `reason` | Meaning |
|---|---|---|
| 400 | (none) | The envelope or payload failed its schema; `issues` lists `{ path, message }`. |
| 409 | `unknown-worker` | No worker of that name is configured any more. |
| 503 | `worker-offline` | The worker did not answer its status call. |
| 409 | `worker-version` | The worker runs another app version. Update it first. |
| 409 | `unsupported-capability` | That kind of operation never runs on a worker. Review is local-only; prepare, train and preview may run remote. |
| 404 | `missing-dataset` | The dataset no longer exists. |
| 409 | `stale-dataset` | The dataset changed since it was read. `currentRevision` holds the new one: reload and resubmit. |

**Start routes** are the trainers' existing routes under `/api/training/datasets/:id/...`. They take a plain JSON body, read the fields they know and validate on their own terms (below). Most ignore unknown fields; the SA3 route refuses them. For a remote worker, a client calls them through the worker proxy, `/api/workers/<name>/api/training/...` ([`trainingStartBase`](../../server/src/contracts/trainingStarts.ts)). The proxy forwards training routes only and refuses `/ops`.

## Reading a dataset and its revision

`GET /api/training/datasets/:id` returns the dataset detail. Its `updatedAt` is the revision to put in `dataset.revision`, and the read itself may move it, so use the value it returns. Any edit to the dataset (labels, counters, status) moves the revision, so read it just before submitting. For YuE2 preparation, `GET /api/training/ops/preparation/context/:datasetId` returns both revisions the command needs: `{ dataset: { id, revision }, source: { kind: 'dataset-sources', id, revision } }`.

`GET /api/training/capabilities` reports which trainers, models and binaries this machine has. `GET /api/workers` lists configured workers with `online`, `version` and `versionMatch`.

## Recipes

| Call | Returns |
|---|---|
| `GET /api/training/ops/recipes` | `{ version, families }`: `ace-lm`, `ace-dit`, `mm3-lm`, `yue2-nar`, `yue2-ar`, `yue2-joint`. |
| `GET /api/training/ops/recipes/:family?preset=&worker=` | The family's recipe with no overrides, read from the worker when `worker` names one. 400 for an unknown family. |
| `POST /api/training/ops/recipes/resolve` | The recipe for the submitted form ([`RecipeResolveResponse`](../../server/src/contracts/trainingStarts.ts)). |

The resolve payload is `{ family, recipeVersion: 1, overrides, preset? }`; build it with `recipeResolveRequest`. `overrides` is the form, as field-to-value pairs. Node lays it over the user's stored defaults, which lie over the built-in defaults, then checks the form's text fields ("Invalid recipe form" with `issues` when one is not a string). Only `ace-lm` and `ace-dit` have stored defaults. With a remote worker, the built-in and stored layers come from the worker.

`preset` changes the built-in layer. For `mm3-lm` and `yue2-nar` it is `fast`, `balanced` or `thorough`. For `ace-dit` it is the adapter type: `lora` gives the LoRA defaults, anything else the LoKr ones. `ace-lm`, `yue2-ar` and `yue2-joint` ignore it here. A few override keys are switches for the start body rather than form fields: `variantKey` (ACE), `activePreset`, `trainLaunder` and `flashSupported` (`mm3-lm`), `activePreset` (`yue2-nar`), `overtrain` and `mintedMissing` (`yue2-ar`).

The response holds:

- `resolved`: the complete form after layering;
- `execution`: the complete start body for the family's start route;
- `provenance`: where each value came from, `builtin`, `stored` or `override`;
- `builtin` and `stored`, the two lower layers;
- `effective`, for a `yue2-joint` form whose `method` is `base-matched` or `fixed-recipe`: the form with that method's fixed values applied;
- `worker` and `operation`, echoed from the command.

Resolving writes nothing and starts nothing. Before starting, check that the worker the response names is still the one you mean to train on; the bundled UI refuses to start if the user switched workers after resolving.

## Starting ACE, MM3 and YuE2 NAR/AR

1. Read the dataset (above) and make sure the prerequisites below hold.
2. `POST /api/training/ops/recipes/resolve` with the form.
3. `POST` the response's `execution`, unchanged, to the family's start route, under the base for the chosen worker.
4. Follow the job (below).

| Family | Start route | Success | Prerequisites the route checks |
|---|---|---|---|
| `ace-lm` | `POST /datasets/:id/train-lm` | 202 `{ jobId }` | The dataset is built (`POST /datasets/:id/build`) and preprocessed (`POST /datasets/:id/preprocess`, which writes the tensors); an LM base model is installed. |
| `ace-dit` | `POST /datasets/:id/train-dit` | 202 `{ jobId }` | Built and preprocessed, the requested preprocess variant exists, the DiT base is installed. |
| `mm3-lm` | `POST /datasets/:id/mm3-train-lm` | 200 `{ jobId, kind, runName, outDir, attnBackend }` | The dataset is built, MM3 models installed, MM3 codes written (`POST /datasets/:id/mm3-codes`), and at least one track with a `.mm3.txt` caption beside its audio. The trainer skips tracks without one; with none at all the route answers 400. A shared caption file is refused. |
| `yue2-nar` | `POST /datasets/:id/yue2-train` | 200 `{ jobId, kind, runName, outDir, clips, … }` | YuE2 models installed, the latent cache built (`POST /datasets/:id/yue2-preprocess`), a trigger word unless `allowNoTrigger`. |
| `yue2-ar` | `POST /datasets/:id/yue2-ar-train` | 200 `{ jobId, kind, runName, outDir, warnings, … }` | As NAR, plus the minted pack unless `allowNoMinted`, cursor spans when `cursorWeight` is set, and `steps` within the overtrain limit unless `allowOvertrain`. |

The exact success fields are in [`TrainingStartResponses`](../../server/src/contracts/trainingStarts.ts). Each of these routes also answers:

- 404 `{ error: 'Dataset not found' }`;
- 409 `{ error: 'A job is already running for this dataset' }` (one job per dataset);
- 503 when the trainer binary is missing;
- 400 `{ error }` for a field it refuses, such as an unknown enum value, an out-of-range number or a bad adapter name.

The ACE routes refuse out-of-range numbers. The MM3 and YuE2 routes mostly fall back to the default instead, so a body from `resolve` is the reliable way to get exactly the values shown to the user. The YuE2 NAR and AR routes train LoRA only: `adapterType: 'dora'` or `dora: true` answers 400, since DoRA is a joint-training option.

There is no snapshot guard between resolve and start: the start route trusts its body. A client that wants the accepted values must send `execution` as returned and must not mix in fields from a later form.

An MM3 run with saved optimizer state can train further. `GET /datasets/:id/mm3-runs` returns `{ runs, busy }`, and `POST /datasets/:id/mm3-resume-lm` with `{ runName, addSteps? }` (default 250 more) or `{ runName, steps }` (the new total) answers 200 `{ jobId, kind, runName, outDir, from, steps, optionsSource }`. The recipe comes from the run itself; the body may change only `saveEvery`, the stopping fields (`stopMode`, `targetLoss`, `targetLossMetric`, `targetLossEpochs`), `preview`, `attnBackend` and `prefixN`. An unknown run answers 404. A run without saved state or a step cap that is not above the saved step answers 400.

## Starting YuE2 joint training

Start a new joint run through the preparation operation. It resolves the recipe and checks the worker, the revisions, the prerequisites and other active work in one accepted command, then posts to `/datasets/:id/yue2-joint-train` itself. That route also takes a plain body directly (`trainingMethod: 'aitk'` is required), which the bundled Refine panel uses to continue a run from a checkpoint with `resumeRunId` and `resumeStep`. A direct post gets only the route's own checks: one job per dataset, the request fields, and the manifest of a run that is not auto-prepared. A body carrying an `admission` value is a continuation Node admitted itself; a value that was never issued, or was already used, answers 409.

1. `GET /api/training/ops/preparation/context/:datasetId` for the dataset and source revisions.
2. `POST /api/training/ops/preparation` with [`jointStartRequest`](../../server/src/contracts/trainingStarts.ts). Its payload is `{ recipes: { joint: { version: 1, overrides, preset? } }, mode: 'train-after-preparation', stages: ['joint'], trigger, lyricTiming }` and its operation kind must be `yue2-preparation`.
3. The response is `{ pipeline }` ([`Yue2PreparationSummary`](../../server/src/contracts/trainingPreparation.ts)). Follow it with `GET /api/training/ops/preparation/:id`, and read the latest stage's training job with `GET /api/training/ops/preparation/:id/job` (`{ job }`, `null` before any stage has one).

`stages: ['joint']` alone means the inputs are already prepared. If any are missing, the command answers 409 with an error naming them. To prepare first, list the stages to run from `latents`, `codes`, `sheet`, `stems` and `align`. The stage list is refused with 400 when:

- a stage appears twice;
- `prepare-only` names a trainer, or `train-after-preparation` names none;
- `joint` is combined with `nar` or `ar`, or with any other stage but no `latents`;
- a training stage has no form in `recipes`.

`mode: 'prepare-only'` prepares without training and needs only the `prepare` capability. The same command can also train NAR and AR, with `stages` including `nar` or `ar` and their forms in `recipes`.

What the command checks, in order:

1. The schema.
2. The idempotency key. A repeat of the same body returns the existing pipeline. The same key with a different body answers 409 `Idempotency key belongs to another command`.
3. The worker: configured, online, same version.
4. The dataset revision. It is checked again after any remote recipe fetch, and the source revision is checked at start.
5. That no other preparation, batch, pipeline or dataset job is active (409).

The joint recipe is resolved inside the command from `overrides` and `preset`, exactly as `recipes/resolve` would. The resolved body goes to the joint route on this machine, or to the worker. On a worker, Node first pushes the dataset to it.

Control: `POST /api/training/ops/preparation/:id/pause`, `/resume`, `/retry` and `/cancel` each return `{ pipeline }`, or 404 for an unknown id.

- Pause holds the pipeline at the next stage boundary; a running stage finishes first.
- Retry is allowed only from `failed` or `interrupted`. It answers 409 in any other state, while other work is active for the dataset, while a recorded stage job still runs, or when the dataset or its sources changed since the command was accepted.
- Pause, resume and cancel on a pipeline in a state they do not apply to return it unchanged.
- `GET /api/training/ops/preparation?datasetId=` lists pipelines.

## Starting SA3 training

SA3 has no recipe family, so `recipes/resolve` does not apply. Its start route checks the whole body itself and refuses what it cannot use; nothing falls back to a default. Call both routes under the base for the chosen worker.

1. `GET /datasets/:id/sa3-train` returns `{ available, backend, missing, defaults, runs, activeJob }`.
   - `available` is true only on a CUDA or CPU engine build with `ace-train` and every SA3 training model installed; `missing` lists the model files that are not.
   - `defaults` is the recipe with nothing set.
   - `runs` lists the dataset's runs in descending order of run id (adapter name, then time stamp), as `{ run, adapterName, recipe, resumable, published, baseSha256?, inferenceSha256? }`. `run` is the opaque `<adapterName>/<stamp>` id the other SA3 routes take.
   - `activeJob` is the dataset's queued or running job, or `null`.
2. `POST /datasets/:id/sa3-train` answers 202 `{ jobId, job }`. Follow the job as below; its `kind` is `sa3-train`.

| Key | Default | Accepted |
|---|---|---|
| `steps` | `1000` | integer, 1 to 1,000,000 |
| `rank` | `16` | integer, 1 to 256 |
| `alpha` | `16` | 0.001 to 1024 |
| `learningRate` | `0.0001` | 1e-8 to 0.1 |
| `duration` | `380` | longest training crop in seconds, 1 to 384 |
| `seed` | `42` | integer, 0 to 2147483647 |
| `checkpointEvery` | `100` | integer, 1 to 1,000,000 |
| `exportEvery` | `0` | integer, 0 to 1,000,000 |

The body may also carry `adapterName` (default: the dataset slug; 1 to 64 letters, digits, `_` or `-`, starting with a letter or digit), `preview` ([checkpoint previews](#checkpoint-previews), new runs only) and `resumeRun`. Any other key answers 400 `Unsupported SA3 option: <key>`, and a value outside its range 400 `Invalid SA3 <key>`.

`{ resumeRun: <run>, steps? }` continues a run whose `resumable` is true. `steps` may raise the total and never lower it. Every other recipe key, and `adapterName`, must be left out or match the saved run (400 `SA3 resume cannot change <key>`), and `preview` is refused: a resumed run keeps the preview settings it started with.

The route also answers 404 for an unknown dataset; 409 while the dataset has a job, or while any batch pipeline or preparation pipeline is active; 400 when the trainer is not available (the message names the backend and the missing files) or the body is not an object.

The job's `phase` runs `prepare`, `engine-stop`, `loading-models`, `encode`, `train`, `export` and `engine-restart`. With checkpoint previews on, training runs in segments with `engine-restart`, `preview` and `engine-stop` between them, and the last step's previews render after the export. Its `step` metrics carry `loss`, `totalSteps` and `stepMs`. A finished run publishes its final export as an SA3 adapter, named in the run's `published`.

## Following a job

| Call | Returns |
|---|---|
| `GET /api/training/jobs?datasetId=` | `{ jobs: TrainingJobStatus[] }` |
| `GET /api/training/jobs/:jobId` | [`TrainingJobStatus`](../../server/src/contracts/trainingStarts.ts): `status` is `queued`, `running`, `done`, `failed` or `cancelled`, with `phase`, `done`/`total` and `error`; 404. |
| `GET /api/training/jobs/:jobId/stream` | Server-sent events, `data: { type, … }` for `job`, `progress`, `sample` (labelling), `log`, `metric`, `preview` (MM3) and `status` events; the server closes the stream after `status`. It replays recent events on connect, so a reconnect catches up. A job no longer held in memory (a cancelled one, or any job after a restart) answers 404 here; `GET /jobs/:jobId` still reads it from disk. |
| `DELETE /api/training/jobs/:jobId` | `{ ok: true }`; stops the trainer process tree. An SA3 trainer is asked to stop first and killed after 30 seconds if it has not. 404 for an unknown job. |
| Per-family run list | `runs` in [`TRAINING_STARTS`](../../server/src/contracts/trainingStarts.ts), for example `GET /datasets/:id/yue2-joint-runs`. SA3 lists its runs in `GET /datasets/:id/sa3-train`. |

Training jobs cannot be paused; only preparation pipelines and batches can. Through a worker, job routes go through the same proxy base as the start.

## Checkpoint previews

YuE2 joint, MM3 and SA3 runs can render audio at saved checkpoints while they train. Each keeps its own records and audio route. ACE has no checkpoint previews; its auditions are a separate feature.

| Target | Turned on by | Records | Audio |
|---|---|---|---|
| YuE2 joint | `preview` in the joint form | `GET /datasets/:id/yue2-joint-previews?run=<jobId>` returns `{ run, output, previews }` | the record's `audioUrl`; add `&format=mp3` for the cached MP3 |
| MM3 | `preview` in the start or resume body, with `everySteps` or `everyMinutes` above 0 | `preview` events on the job stream, `{ type: 'preview', run, preview }`; there is no listing route | `GET /api/training/mm3/preview?run=<run>&file=<preview.file>` |
| SA3 | `preview` in a new run's start body | `GET /datasets/:id/sa3-previews?run=` returns `{ run, settings, previews }`, the newest run when `run` is left out | the record's `audioUrl` |

SA3 `preview` is `{ everySteps, prompts, seconds, seeds }`. `everySteps` 0 turns previews off; otherwise they need 1 to 8 prompts of up to 2000 characters. `seconds` is 1 to 384 (default 30) and `seeds` 1 to 4 integers (default `[424242]`). There is one record per step, prompt and seed; its `status` is `queued`, `rendering`, `done`, `failed` or `skipped`, and `audioUrl` is set only when it is `done`. An unknown SA3 `run` answers 404 on the list, render and audio routes. YuE2 joint records have `status` `rendering`, `done` or `failed`.

Both listing targets also render on demand:

- `POST /datasets/:id/yue2-joint-previews/render` with `{ run, step }`, plus optional `seconds` (8 to 360, default 180), `seed`, `takes` (1 to 4) and sampler overrides, renders one checkpoint and answers `{ run, step, previews }`. 400 when the step has no complete checkpoint, 503 with the engine down.
- `POST /datasets/:id/sa3-previews/render` with `{ run, step, settings? }` renders, or retries, the step's records that have no done render, using the run's saved settings under any given, and answers the refreshed list. 400 when the step has no captured snapshot or the settings are invalid, 409 while the dataset has a job, 503 with the engine down.

YuE2 joint runs and their previews are read from this machine's index; runs trained on a worker are mirrored into it. SA3 runs and previews stay on the machine that trained them: read them through the same base as the start. The worker proxy rewrites `/api/training/` URLs in JSON responses, so an SA3 record's `audioUrl` already points through it.

## Restarts

Training is never resumed or resubmitted by itself after a Node restart:

- **Training jobs** run in memory, and each is also recorded on disk. At startup, every job recorded as queued or running is rewritten as `failed` with error `Server restarted`. Its trainer process was stopped on shutdown.
- **YuE2 joint runs** on this machine that were running become `interrupted`. Runs mirrored from a worker are left alone; the worker owns them. A joint run can be continued from a checkpoint by a new start with `resumeRunId` and `resumeStep` in the joint form.
- **SA3 and MM3 runs** keep their run folders. Continue one with a new start: `resumeRun` on `POST /datasets/:id/sa3-train` when the run is `resumable`, or `POST /datasets/:id/mm3-resume-lm`. A resumed SA3 job first renders any checkpoint previews the cut-off run still owed.
- **Preparation pipelines** that were active become `interrupted`, with an error saying the server restarted. `POST …/:id/retry` continues them explicitly.
- **Batches and pipelines** come back `paused`.

## Review

Review is local-only. It reads and writes this machine's run index, including runs mirrored from workers. Every command carries the ladder revisions the client read.

| Call | Effect | Response |
|---|---|---|
| `GET /api/training/yue2-review` | Read only | `{ rows }`, every YuE2 joint run with complete rung checkpoints, across datasets, newest first. A row's `datasetId` and `refineRun` name its ladder. |
| `GET /api/training/ops/review/ladder/:datasetId/:runId` | Read only | [`ReviewLadderResponse`](../../server/src/contracts/trainingReview.ts): the rungs with scores (`facts`), the best rung, and the revisions to echo. |
| `POST /api/training/ops/review/select` | Links the rung's checkpoints as the dataset's adapter (a side effect), unless already linked | `{ linked: true, alreadyLinked, plan }`. `plan` previews what cleanup would delete; nothing is deleted. |
| `POST /api/training/ops/review/cleanup` | Deletes what `choice` selects (caches, other checkpoints, other runs, resume state, other previews) | `{ freedBytes, done, finishError? }` |
| `POST /api/training/ops/review/decision` | With `narFurther`, may start decoder training | `{ outcome: 'skip' }`, `{ outcome: 'reject', worker? }` for a mirrored worker run, or `{ outcome: 'chain', jobId }` with the new job. |
| `POST /api/training/ops/review/finish-batch` | Queues a finish batch for one or more scored ladders | `{ batch, selections }` |
| `POST /api/training/ops/review/draft` | Creates a Create draft that mirrors one audition render. Needs a token. | `{ draftId, revision }`. Repeating the idempotency key returns the same draft. |
| `GET /api/training/ops/review/draft/:id` | Read only. Needs a token. | `{ draft }`, a workflow document; 404 when not found. |

The pick commands (`select`, `cleanup`, `decision`) take a payload of `{ datasetId, runId, runRevision, reviewRevision, step, checkpointDir, blindLabel? }`. `cleanup` adds `choice`, and `decision` adds `narFurther` and `nar` ([`trainingReview.ts`](../../server/src/contracts/trainingReview.ts)). The command needs:

- `dataset` in the envelope, matching `datasetId`;
- a `sources` entry `{ kind: 'yue2-run', id: runId, revision: String(runRevision) }`.

`finish-batch` takes `entries` of picks, each with its `datasetRevision`, plus `knee`. Each entry needs a matching `yue2-run` source. `draft` takes `{ datasetId, previewId, slot, cell }` and a source `{ kind: 'audition-preview', id: previewId, revision: <preview createdAt> }`.

Every review command answers 409 with an `error` saying what to reload when any of these has changed since the ladder was read:

- the run, the review revision, the dataset or the linked checkpoint;
- the rung is gone, or no longer best when finishing needs a scored best;
- a job, pipeline or other review command is active for the dataset.

A run that is still running or already finished cannot be selected. 404 means the dataset, run or audition preview no longer exists.
