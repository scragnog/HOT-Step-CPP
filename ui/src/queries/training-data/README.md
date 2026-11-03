# Training data server state

The training-data area covers Training Studio's datasets, samples, labelling,
preprocessing, preparation and the worker ("Train on") selection: everything
under `components/training-studio/` except the training-runs files,
`stores/trainingStore.ts`, and the `trainingPreparationApi`,
`trainingOperations` and `trainingRecipesApi` clients. It uses the client,
keys and policy in `../client.ts`, `../keys.ts` and `../contract.ts`.
Selections, pending sample edits, stream metrics and preview buffers stay
client state.

## Worker scope

Every key in `queries.ts` carries `worker` (null = this PC) unless the
resource lives only on this PC. A `queryFn` reads its target back from its own
key, so a read scheduled for worker A lands in A's entry whatever "Train on"
says by the time it answers. Callers capture the worker when they schedule a
read, never after an await.

`trainingStore` fetches through these options and projects the result into
the store only while (worker, dataset) still match the selection. A "Train on"
switch cancels the old worker's reads (`cancelWorkerReads`). The projection is
the compatibility surface for training-runs; nothing else fetches these
snapshots.

## Reads

| Resource | Key | Notes |
|---|---|---|
| Datasets, dataset detail, capabilities | `training-data/datasets`, `dataset`, `capabilities` | Worker-scoped store projections |
| Jobs and per-stage status | `training-data/job`, `preprocess-status`, `train-lm-status`, `train-dit-status` | Job status is always live (`staleTime` 0); stream terminal events invalidate the worker scope once |
| Auditions, LS generations, prepared data, pipelines, YuE2 batches | `training-data/<name>` | Worker-scoped |
| Workers, dispatch lines, LM adapters | `training-data/workers`, `worker-dispatch`, `lm-adapters` | Local |
| Preparation chain and its job | shared `yue2-preparation`, `yue2-preparation-job` | Local. The store's follow loop polls these every 1.5 s; a Monitor panel on the same chain joins its reads |
| Folder listing | shared `browse` | Folder picker and batch import root |
| Scan preview | `training-data/scan-preview` | Worker-scoped; New dataset and each batch import row |
| MM3 and YuE2 caption files | `training-data/sample-mm3`, `sample-yue2` | Seed the drawer's editors |
| Lyric Studio export preview | `training-data/lyric-studio-preview` | Seeds the artist/album fields |
| YuE2 caption source | `training-data/yue2-caption-source` | Fresh per handle (`dataset`, `lyricsSet` or `adapter`). Called by `utils/yue2CaptionSource.ts`; the util keeps the error surface and its local cache |
| SA3 training status | `training-data/sa3-status` | Worker-scoped; polls while a job is active. Built inline in `components/training-studio/Sa3TrainCard.tsx`, not in `queries.ts` ([tanstack-query.md](../../../../docs/dev/tanstack-query.md) F4#17) |
| HeartMuLa status and checkpoint takes | `training-data/heartmula-status`, `heartmula-previews` | Worker-scoped. Status is live (`staleTime` 0) and polls while a job or batch runs; takes poll per run while one is rendering. Read by `useHeartmulaController` and the review plugin; the preview retry is a mutation that invalidates both |

The folder listing, scan preview, caption file and export preview rows are
one-shot reads with `staleTime` 0: each open asks the server, and concurrent
reads share a request. They seed forms, so no background refetch may write
over the user's edits:

- Folder picker and New dataset cancel the read they replace, unless the new
  read has the same key, in which case it joins.
- The sample drawer observes its caption files with `useQuery`
  (`components/training-studio/useCaptionFile.ts`), so Query cancels the read
  when the drawer moves to another sample or worker. The editor is seeded only
  by the first answer fetched on each visit to a key, never by a cached copy
  or a later refetch; returning to a sample is a new visit and seeds again. A
  confirmed save writes the server's answer into the entry.
- The audition handoff reads the dataset's revision through the local
  `training-data/dataset` entry (`worker` null), whatever "Train on" says,
  because audition previews and drafts are local.

## Commands

Creating, renaming and deleting datasets, sample edits, labelling,
preprocessing, preparation start/pause/resume/cancel, pipeline and batch
starts, exports and draft handoffs are explicit commands with the worker
captured before they start. They never retry. A successful command updates or
invalidates the entries of the scope it was made in; `trainingStore` runs its
commands through `writeAndInvalidate` (`../writes.ts`) with keys built from the
captured worker. The preparation context
read at submission and recipe/preset resolution are part of building a
submission, so they run once as explicit operations, not cached reads.

## Consumer dispositions

The foundation manifest in
[ui-server-state.md](../../../../docs/dev/ui-server-state.md#consumer-manifest)
lists every training-data file and its disposition. Files with reads:

| Consumer | Disposition |
|---|---|
| `stores/trainingStore.ts` | query projection and commands; preparation follow reads the shared keys |
| `BatchImportWizard.tsx` | query (browse, scan preview); pipeline start is a command |
| `FolderPicker.tsx` | query (browse) |
| `NewDatasetWizard.tsx` | query (scan preview); create is a command |
| `SampleDrawer.tsx`, `useCaptionFile.ts` | query (caption files); edits are client state, saves are commands |
| `SendToLyricStudio.tsx` | query (export preview); export is a command |
| `sendAuditionToCustomGen.ts` | query (local dataset revision); draft creation is a command |
| `PipelineStageProgress.tsx` | adopts the stage job into the studio's one projection; job refresh through Query |
| `useHeartmulaController.ts`, `plugins/heartmula.tsx` | query (status, takes); prepare/train/batch are commands, the preview retry is a mutation |
