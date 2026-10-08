# Training for frontend clients

This page lists every training operation a client calls and the order it calls them in, from reading a dataset to starting a run and reviewing its checkpoints. The types are in the shared contracts: [`trainingOperation.ts`](../../server/src/contracts/trainingOperation.ts) (the command envelope), [`trainingRecipes.ts`](../../server/src/contracts/trainingRecipes.ts), [`trainingPreparation.ts`](../../server/src/contracts/trainingPreparation.ts), [`trainingReview.ts`](../../server/src/contracts/trainingReview.ts) and [`trainingStarts.ts`](../../server/src/contracts/trainingStarts.ts) (start routes, request builders and response shapes). Every path is listed in the [route index](api.md); the operation routes are in its `/api/training/ops` section. How the trainers themselves work is in [training-internals.md](training-internals.md).

Training routes need no auth token, except the two audition-draft routes, which answer 400 without one. Reads change nothing; the start, preparation and review commands change state.

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

**Start routes** are the trainers' existing routes under `/api/training/datasets/:id/...`. They take a plain JSON body, read the fields they know, ignore unknown fields and validate on their own terms (below). For a remote worker, a client calls them through the worker proxy, `/api/workers/<name>/api/training/...` ([`trainingStartBase`](../../server/src/contracts/trainingStarts.ts)). The proxy forwards training routes only and refuses `/ops`.

## Reading a dataset and its revision

`GET /api/training/datasets/:id` returns the dataset detail. Its `updatedAt` is the revision to put in `dataset.revision`, and the read itself may move it, so use the value it returns. Any edit to the dataset (labels, counters, status) moves the revision, so read it just before submitting. For YuE2 preparation, `GET /api/training/ops/preparation/context/:datasetId` returns both revisions the command needs: `{ dataset: { id, revision }, source: { kind: 'dataset-sources', id, revision } }`.

`GET /api/training/capabilities` reports which trainers, models and binaries this machine has. `GET /api/workers` lists configured workers with `online`, `version` and `versionMatch`.

## Recipes

| Call | Returns |
|---|---|
| `GET /api/training/ops/recipes` | `{ version, families }`: `ace-lm`, `ace-dit`, `mm3-lm`, `yue2-nar`, `yue2-ar`, `yue2-joint`. |
| `GET /api/training/ops/recipes/:family?preset=&worker=` | The family's recipe with no overrides, read from the worker when `worker` names one. 400 for an unknown family. |
| `POST /api/training/ops/recipes/resolve` | The recipe for the submitted form ([`RecipeResolveResponse`](../../server/src/contracts/trainingStarts.ts)). |

The resolve payload is `{ family, recipeVersion: 1, overrides, preset? }`; build it with `recipeResolveRequest`. `overrides` is the form, as field-to-value pairs. Node layers it over the built-in defaults, the user's stored defaults and the preset, then validates the result ("Invalid recipe form" with `issues` when it fails). With a remote worker, the defaults come from the worker. The response holds:

- `resolved`: the complete form after layering;
- `execution`: the complete start body for the family's start route;
- `provenance`: where each value came from (built-in, stored, preset, override);
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
| `mm3-lm` | `POST /datasets/:id/mm3-train-lm` | 200 `{ jobId, kind, runName, outDir, attnBackend }` | MM3 models installed, MM3 codes written (`POST /datasets/:id/mm3-codes`), a `.mm3.txt` caption for each sample. A shared caption file is refused. |
| `yue2-nar` | `POST /datasets/:id/yue2-train` | 200 `{ jobId, kind, runName, outDir, clips, … }` | YuE2 models installed, the latent cache built (`POST /datasets/:id/yue2-preprocess`), a trigger word unless `allowNoTrigger`. |
| `yue2-ar` | `POST /datasets/:id/yue2-ar-train` | 200 `{ jobId, kind, runName, outDir, warnings, … }` | As NAR, plus the minted pack unless `allowNoMinted`, cursor spans when `cursorWeight` is set, and `steps` within the overtrain limit unless `allowOvertrain`. |

The exact success fields are in [`TrainingStartResponses`](../../server/src/contracts/trainingStarts.ts). Every start route also answers:

- 404 `{ error: 'Dataset not found' }`;
- 409 `{ error: 'A job is already running for this dataset' }` (one job per dataset);
- 503 when the trainer binary is missing;
- 400 `{ error }` for a field it refuses, such as an unknown enum value, an out-of-range number or a bad adapter name.

The ACE routes refuse out-of-range numbers. The MM3 and YuE2 routes mostly fall back to the default instead, so a body from `resolve` is the reliable way to get exactly the values shown to the user.

There is no snapshot guard between resolve and start: the start route trusts its body. A client that wants the accepted values must send `execution` as returned and must not mix in fields from a later form.

## Starting YuE2 joint training

Start a new joint run through the preparation operation. It resolves the recipe and checks the worker, the revisions, the prerequisites and other active work in one accepted command, then posts to `/datasets/:id/yue2-joint-train` itself. That route also takes a plain body directly (`trainingMethod: 'aitk'` is required), which the bundled Refine panel uses to continue a run from a checkpoint with `resumeRunId` and `resumeStep`. A direct post gets only the route's own checks: one job per dataset, the request fields, and the manifest of a run that is not auto-prepared. A body carrying an `admission` value is a continuation Node admitted itself; a value that was never issued, or was already used, answers 409.

1. `GET /api/training/ops/preparation/context/:datasetId` for the dataset and source revisions.
2. `POST /api/training/ops/preparation` with [`jointStartRequest`](../../server/src/contracts/trainingStarts.ts). Its payload is `{ recipes: { joint: { version: 1, overrides, preset? } }, mode: 'train-after-preparation', stages: ['joint'], trigger, lyricTiming }` and its operation kind must be `yue2-preparation`.
3. The response is `{ pipeline }` ([`Yue2PreparationSummary`](../../server/src/contracts/trainingPreparation.ts)). Follow it with `GET /api/training/ops/preparation/:id`, and read the running stage's training job with `GET /api/training/ops/preparation/:id/job`.

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

Control: `POST /api/training/ops/preparation/:id/pause`, `/resume`, `/retry` and `/cancel` each return `{ pipeline }`, with 404 for an unknown id and 409 when the pipeline's state forbids the action.

- Pause holds the pipeline at the next stage boundary; a running stage finishes first.
- Retry is allowed only from `failed` or `interrupted`.
- `GET /api/training/ops/preparation?datasetId=` lists pipelines.

## Following a job

| Call | Returns |
|---|---|
| `GET /api/training/jobs?datasetId=` | `{ jobs: TrainingJobStatus[] }` |
| `GET /api/training/jobs/:jobId` | [`TrainingJobStatus`](../../server/src/contracts/trainingStarts.ts): `status` is `queued`, `running`, `done`, `failed` or `cancelled`, with `phase`, `done`/`total` and `error`; 404. |
| `GET /api/training/jobs/:jobId/stream` | Server-sent events, `data: { type, … }` for job, progress, log, metric and status events. It replays recent events on connect, so a reconnect catches up. |
| `DELETE /api/training/jobs/:jobId` | `{ ok: true }`; stops the trainer process tree. 404 for an unknown job. |
| Per-family run list | `runs` in [`TRAINING_STARTS`](../../server/src/contracts/trainingStarts.ts), for example `GET /datasets/:id/yue2-joint-runs`. |

Training jobs cannot be paused; only preparation pipelines and batches can. Through a worker, job routes go through the same proxy base as the start.

## Restarts

Training is never resumed or resubmitted by itself after a Node restart:

- **Training jobs** live in memory. At startup, every job recorded as queued or running is rewritten as `failed` with error `Server restarted`. Its trainer process was stopped on shutdown.
- **YuE2 joint runs** on this machine that were running become `interrupted`. Runs mirrored from a worker are left alone; the worker owns them. A joint run can be continued from a checkpoint by a new start with `resumeRunId` and `resumeStep` in the joint form.
- **Preparation pipelines** that were active become `interrupted`, with an error saying the server restarted. `POST …/:id/retry` continues them explicitly.
- **Batches and pipelines** come back `paused`.

## Review

Review is local-only. It reads and writes this machine's run index, including runs mirrored from workers. Every command carries the ladder revisions the client read.

| Call | Effect | Response |
|---|---|---|
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
