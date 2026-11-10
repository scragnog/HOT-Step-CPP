# Training System — Architecture & Continuation Guide

HeartMuLa's `/api/training/heartmula-batch` starts the same durable runner as
YuE2, with `family: heartmula` and Prepare/Train stages. Its `recipe` body is
the ordinary HeartMuLa train request. Existing `/api/training/yue2-batch`
list, pause, resume and cancel routes serve both families, and older snapshots
without a family remain YuE2. Snapshots retain HeartMuLa run IDs to resume
optimizer checkpoints after interruption. The prepared-data routes accept
`target=heartmula` on GET and `target: heartmula` on DELETE to limit both
inventory and deletion to `<dataset>/heartmula/prepare`.

Resolve runtime paths first using the [data folder configuration](config.md#dev-checkouts).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

*The complete map of HOT-Step's native training system (Training Studio). Written as the handoff/continuation doc: everything an agent or contributor needs to keep working on this subsystem in a fresh session. Built 2026-07-27/28; measurements are from an RTX 5090 (32 GB, sm_120) unless a passage names another machine.*

## Stable Audio 3 Medium training

`trainSa3Runner.ts` owns preparation, `ace-train sa3-train`, engine restoration
and publication. `GET /api/training/datasets/:id/sa3-train` returns readiness,
defaults and saved runs; POST starts a new run or resumes one by `resumeRun`.
Preparation calls real SuperSep level 5, preserves sources and writes a derived
44.1 kHz stereo snapshot. `sa3Dataset.ts` maps parsed caption, genre, BPM, key
and signature to plain prompt text. Lyrics never enter that prompt.

The trainer freezes the Medium base DiT, SAME and T5Gemma. CUDA keeps the DiT
base in BF16; adapters and optimizer state are F32. CPU upcasts adapter-site
weights to F32 because its BF16 OUT_PROD is unsupported. The segmented backward
graphs run on one backend, with an operation-support audit before execution;
unsupported operations fail instead of falling back to another device.
AdamW runs on host adapter masters. Do not describe the whole training loop as
GPU-only. Defaults are rank/alpha 16, LR 1e-4, batch 1 and 1,000 steps.
The seconds embedder is frozen in the server workflow.

Runs live under `<adapters>/sa3-medium/<dataset>/<adapter>/<timestamp>/`.
`run.json` records the recipe and model provenance; `train-manifest.json` and
`dataset/` preserve the prepared snapshot. `state.gguf` contains resume state;
`adapter-raw.gguf` contains raw training parameters. Neither is an inference
adapter. `adapter.gguf` stores `dora-rows-v1` row scales against the inference
DiT and is validated before atomic publication to `<models>/sa3-adapters/`.
The local training base is distinct from the distilled inference DiT.

Resume may increase total steps but keeps the other recipe fields. It reuses
the snapshot and the engine checks model, conditioning and timing fingerprints.
Cancellation writes `STOP`; startup/encoding cancellation has a 30-second kill
fallback and may leave no checkpoint. The runner restores the engine in its
cleanup path, including after failed training.

For integration checks on an idle dev stack, create a dedicated test dataset
and run `node tools/sa3-training-check.mjs --dataset <id> --out <new directory>`.
The checker trains three steps with a 12-second crop, resumes to five, checks
picker publication and renders two held-out captions at strengths 0 and 1 with
seed 42. It keeps a JSON receipt and WAVs, checks finite/non-silent audio and
requires the paired files to differ. This is a mechanical round-trip check,
not a listening verdict or a 1,000-step quality pilot. Run its `--self-test`
without models. GPU memory, operation placement and steady timings need
separate receipts, on the same HEAD and recipe for each machine.

### SA3 checkpoint previews and the segmented trainer

`runSa3TrainJob` has two modes. Without `preview` settings it runs one trainer
process to `run.steps` and publishes, as above. With them (`everySteps > 0` and
at least one prompt; the `preview` option is accepted on a new run only and is
saved in `run.json`) it follows YuE2's serial lifecycle:

1. Stop the engine. `ace-train sa3-train` runs to the next preview step as an
   absolute `--steps` bound with `--export-every 0`, saves `state.gguf` and its
   inference export, and exits. `sa3NextBound` picks the bound; `run.goalSteps`
   keeps progress and the chart at the logical goal. Total steps is not in the
   resume fingerprint or the AdamW update, so only the process arguments change
   between segments.
2. `captureSa3Snapshot` (`sa3Snapshots.ts`) validates the export and keeps it
   once as `<run>/snapshots/step-N.gguf`. A capture is never replaced.
3. Restart the engine and render every prompt and seed (`renderSa3Previews`,
   `sa3Preview.ts`), stop the engine, then resume with `--resume` in the same
   run directory. A failed render is logged and training continues once the
   engine is stopped again; a failed engine stop or restart aborts the run.
4. Only the last step's export is published to `<models>/sa3-adapters/`. Its
   snapshot is captured too, and its previews render after the engine is back,
   never twice for a step.

A resumed job first renders any captured step that still owes previews
(`sa3PreviewPending` reads stored statuses, so a cut-off round counts as owed),
and re-exports a preview step whose capture was lost by running `--steps` at the
saved step. `tools/sa3-resume-proof.mjs` compares 12 uninterrupted steps with
4+4+4 segments: restart states bitwise, draws exact, loss and final tensors
within `abs 1e-6 + 1e-5*|ref|`. EMA is display state and resets, so it is not
part of that evidence.

Records live in `<run>/previews/index.json`, one per step, prompt and seed;
audio is written before its record is committed. Types are in
`server/src/contracts/training.ts`. Routes are scoped by the dataset and an
opaque run id (`<adapterName>/<stamp>`, resolved by `sa3RunById` under the
dataset's own runs, never a browser path):

| Route | Purpose |
|---|---|
| `GET /datasets/:id/sa3-previews?run=` | `{ run, settings, previews }`; the newest run when `run` is omitted. A done record carries an `audioUrl`. |
| `POST /datasets/:id/sa3-previews/render` | `{ run, step, settings? }`: renders or retries the step's cells that have no done render. 400 if the step has no captured snapshot, 409 with a job on the dataset, 503 with the engine down. Runs on the GPU lane. |
| `GET /datasets/:id/sa3-previews/audio?run=&file=` | One WAV, only for a done record's own file name inside the run's `previews` folder (links resolved; it must stay inside the run). |

Renders go to `/sa3-synth` with `snapshot_run=<dataset>/<adapter>/<stamp>` and
`snapshot_step=N` (see [engine](engine.md#stable-audio-3-sampling)), so no
filesystem path crosses the wire.

### Training Studio shell and plug-ins

Training Studio resolves one **effective target** above the tabs
(`useTrainingTarget.ts`, rules in `trainingTargets.ts`) from the active backend.
The former `hs-trainingTarget` browser override is no longer read. A queued or
running target-bound job pins the target to its own (`targetForJobKind`). The
target feeds the phase stepper, dataset chips, Preprocess gate, Train page and
previews, so changing a view cannot retarget a submitted job. An unregistered backend trains as ACE.

Each target is one plug-in (`plugins/{ace,yue2,mm3,sa3}.tsx`, registry
`trainingPlugins.tsx`). The static half is `TrainingTargetSpec` in
`trainingTargets.ts` (`phases`, `reachablePhases`, `phaseLabels`,
`prepareStages`, `trainStages`, `recipeFields`, `previewSource`). The rendering
half is `TrainingPlugin` in `trainingPluginTypes.ts` (`chips`, `readiness`,
`renderPrepare`, `renderOptimise`, `renderTrain`, `performAll`). Chip rules are
pure functions in `plugins/chips.ts`; gate order is `resolveTrainGate`
(`trainGates.ts`). A phase the target cannot stand on falls back to Train
(`resolvePhase`). SA3 has `phases: ['dataset', 'train']`, so no Preprocess. The
shell owns layout and selection; each plug-in owns its parameters and
serialization, and server validation stays authoritative. This is not a generic
form engine.

The preview view is shared. `TrainingPreviewPanel.tsx` holds the settings
block, field, checkpoint card and take rows (presentation only, no server
state). YuE2's ladder, MM3's ladder and the SA3 card all draw it and keep their
own scoring. YuE2's and MM3's ladders are one component,
`CheckpointLadder.tsx` (the rung loop, Render, Use this rung or checkpoint);
YuE2's blind labels, scores, album score and cleanup ride in its slots. MM3's ladder
(`Mm3CheckpointLadder.tsx`) reads `GET /datasets/:id/mm3-previews?run=`, the
WAVs in `<run>/previews` with a `<file>.json` record beside each. Render more
uses the preview prompt the run recorded in `hotstep-run.json`
(`previewPlan`), so a caption edit or a cleared codes cache does not change
or break it; a run from before that resolves it once and records it. "Use this
checkpoint" is the `mm3/select` operation (`services/training/mm3Ladder.ts`),
which records the dataset slug's chosen reference in `<training>/mm3-linked.json`
using YuE2's atomic JSON pattern, updates any existing album presets
(`pointMm3PresetsAt`, the same function the local runner and worker mirror use
at run end), and sets the global LM adapter. Like YuE2, a linked album without
a preset row gets a durable record but no new preset row. The record drives
`inUse`, cleanup eligibility and run deletion; presets are also checked
independently so cleanup never removes a checkpoint they use. Older datasets
fall back to their existing preset reference until their next pick. `previewSource`
records where each target's previews come from: `yue2-ladder`, `mm3-ladder`,
`sa3-previews` or `none`. ACE is
`none`: it has no checkpoint exports or renderer, so it keeps Audition. SA3
previews are read through `ui/src/queries/sa3-previews/`.

The SA3 asset block (`Sa3DatasetAssets`, `datasetAssets.ts`) is read fresh like
the other backends' blocks, from the preparation and export receipts: prepared
audio and clip count, the encoding cache only when its fingerprint validates,
captured snapshots across the dataset's runs, and the newest published adapter.
Readiness is never inferred from a finished run alone.

## HeartMuLa backbone LoRA training

Native backbone LoRA for HeartMuLa 3B. The engine side (spec, CLI, measured numbers)
is [HeartMuLa training](engine.md#heartmula-training); this section is the server and
UI side. The shared shapes are `server/src/contracts/heartmulaTraining.ts`.

**Files.** `services/training/heartmulaPrepare.ts` (sample list, stale check),
`heartmulaTrain.ts` (recipe and preview validation, model discovery, argv),
`heartmulaRuns.ts` (run folders, status, checkpoints, publication), `heartmulaRunner.ts`
(the two jobs), `heartmulaPreview.ts` (preview catalogue and render),
`heartmulaStatus.ts` (readiness and the status route). Routes are `heartmula-*` in
`routes/training.ts`. The UI is the HeartMuLa plug-in (`plugins/heartmula.tsx`),
`useHeartmulaController.ts` and the three `Heartmula*Card.tsx` leaves; `TrainPanel`
hosts the plug-in's Review page.

**Runs.** A run lives at `<adapters>/heartmula-runs/<slug>/<adapter name>/<stamp>/`
with `run.json` (the recipe and resolved preview settings), `status.json`, the
trainer's `hm_train_state.bin` and `adapters/step-NNNNNN.gguf`, and `previews/`.
Checkpoints stay outside `heartmula-lm-adapters`, so only a finished run reaches the
generation picker. A stored live state with no job behind it reads as failed and
stays resumable. A resume keeps the saved recipe and preview settings.

New HeartMuLa runs use dropout 0.05 and accumulation 4. With no explicit `steps`,
the budget is `min(480, 30 * ceil(preparedSamples / gradAccum))`, matching the
trainer's short final update per epoch. `heartmulaStatus` reports dataset-sized
defaults, and `resolveHeartmulaRun` resolves the budget from the prepared manifest
at admission. Batch validation and the UI retain omitted steps until then, so
each prepared dataset gets its own budget. Explicit steps and saved recipes are
unchanged; `HEARTMULA_RECIPE_LEGACY.loraDropout` remains 0 for old resume checks.
These defaults mitigate possible overfitting; listening tests must establish quality.

Checkpoint preview requests read the trigger from their adapter and call
`withHeartmulaTrigger`, as normal generation does, before checking the 4096-byte
tag limit. The saved prompt stays intact; base takes use it without adapter metadata.

**Readiness.** CUDA build, `ace-train` present, the four base files, and a selected
GPU of at least `HEARTMULA_MIN_VRAM_MB`. That floor is the trainer's measured
smallest mode (10.41 GiB allocated, every block recomputed) plus 1.5 GiB for the CUDA
context. The default mode needs 15.36 GiB; `--keep-layers auto` bridges the gap.
Prepare also needs the encoder pack; there is no fallback to ACE preprocessing.

**Segmented runs and previews.** `runHeartmulaTrainJob` follows YuE2's lifecycle. With
previews on (`everySteps > 0`):

1. Before the first segment, the base model is rendered (the engine is up).
2. Each segment runs `ace-train heartmula-train --stop-after N --resume`, where N is
   the next multiple of `everySteps` below the run's total. The trainer checkpoints
   with reason `stop` and exits 3; `runYue2AceTrain` restarts the engine.
3. `renderHeartmulaPreview` renders the saved `step-N.gguf` through `heartmulaClient`
   and unloads the model. The next segment stops the engine again and resumes.
4. After exit 0 the adapter is published, then the final take is rendered, even when
   the last step is off the cadence. A step that already has a done take is not
   rendered twice.

A checkpoint with reason `now` (the Save and pause button) ends the run as paused, not
as a preview pause. On resume, a cadence step with no take is rendered first. A failed
render is recorded and training goes on. A render that cannot leave the engine clean
(the model did not unload) throws `HeartmulaPreviewCleanupError` and fails the run, with
the checkpoints kept. Cancel marks the take `skipped` and cancels the engine job. A run
never trains while it previews. A manual preview takes the trainers' GPU chain in
`labelingQueue.ts` (`reserveGpuForPreview`, `runReservedPreview`): it is refused while any
GPU job is queued or running, for any dataset, and while it holds the GPU the HeartMuLa
starts are refused and other GPU jobs queue behind it.

**Provenance.** Each take records its run, kind (`base`, `step`, `final`), step, the
resolved fixture (tags, lyrics, seed, sampler), the adapter file relative to the run
and its SHA-256, the render time, and the pause it cost (trainer `stop` checkpoint to
the next trainer `start`: engine restart, render, engine stop, model reload). Takes
are stored in `previews/index.json`, audio first and the record after, and served only
by the file name a done record holds.

**Previews render on the trainer's base.** An adapter is bound to its base model by
fingerprint, so a take always renders on 3B F16 (`HEARTMULA_TRAIN_BASE`), whatever the
user generates with. A render on another variant or quantization is refused by the
engine with `base_fingerprint differs`; this was found in the first real run and is why
the pick is fixed.

**Measured in the app** (RTX 5090, nothing else on the GPU, `--attn auto --keep-layers
auto`: flash-bf16, 28 blocks kept): four 94-second samples, 4 micro-batches, median
1001 ms per step and 15.78 GiB allocated, which scales to the engine contract's
908 ms at its 84.7-second clip. A 30-second preview took 13 s to render and paused
training for 18 to 20 s. Cancelling during a preview left the checkpoint, marked the
take skipped, returned the GPU to its idle level and left the engine ready to generate.

## YuE2 joint training

The opt-in `fixed-recipe` method (the joint card's HOT-Approved button; `applyFixedRecipe` in `yue2JointTrainRunner.ts`) defaults to the bf16 LM base (the request's `base` is kept when set, and the standalone prepare route takes the same `base`), rank-64 LoRA (alpha 64) at the native four fused sites, 1,500 steps saved every 200 at seed 42, one song per update, whole-song AR, a 1,500-frame NAR crop, CE weight 0.5 and KL weight 0.2. It uses AdamW through LmOptim at LR 1e-4 with epsilon 1e-6, constant LR, an AR LR multiplier of 0.5, caption and lead-sheet dropout of 0.5 each, and no text or lyric dropout or lyric timing. Like every method it trains on a -14 LUFS cache, but nothing refuses another level. The server forces these values, apart from the base, even if a saved form contains other values. Its runner omits `--nar-drift` and retains the installed v9 companion decoder.

The method's preparation mode removes lyric cues, keeps unsupported section headings in Title case, then applies its caption parser and NFC normalization. The style starts with the bare trigger word, or the dataset slug when no trigger is set, without duplicating an existing prefix. Native trigger injection stays off to avoid its "in the style of" wrapper. Preparation writes an isolated prepared-dataset cache index and marks the output manifest with `caption_grammar: fixed-recipe-v2`; the start path rejects a manifest without it. Older prepared datasets are automatically re-prepared on the next automatic preparation. Caption cleanup uses a temporary copy of the legacy manifest, leaving the old method cache index alone.

The joint run index reconciles with the configured `yue2-joint-adapters` folder at startup and when its folder set changes. Missing finished runs leave the index. Copied folders enter from `run.json`, or from a unique dataset trigger in the folder name for older runs. Every index write also saves the current record in the run folder; an unavailable adapters root leaves the index alone.

The native CUDA joint trainer follows the YuE2 recipe. Each step trains
both AR and NAR adapters, refreshes detached AR conditioning, clips the combined
gradients and makes one AdamW8bit update. With a whole-song decoder (`--nar-crop-frames 0`)
the NAR conditions on the same token sequence the AR loss runs on, so the AR forward
captures the conditioning prefix itself and the refresh pass is skipped: bit-identical
adapters, about 10-15% less time per update. A cropped decoder or a frozen planner
still runs the refresh. Existing separate YuE2 trainers remain
available as Legacy.

```text
ace-train yue2-joint-train --checkpoint base.safetensors --dataset prepared/dataset.json --output new-run --steps 150 --save-every 50 --seed 42 --device CUDA0
```

The checkpoint is the raw YuE2 ConvRot safetensors model or any `yue2-lm-*.gguf`
(chosen by extension). A GGUF base loads through `Yue2AitkModel::load_gguf`: every
linear is a plain `mul_mat` over the GGUF weight (fused q|k|v and gate|up stay
separate parts when their quant types differ, joined with `join_rows`), the head
loss backs through a BF16 transpose of `output.weight` built at load, and the run
sets `GGML_BACKWARD_MM=1` so no activation gradient is an `OUT_PROD` of a BF16 or
quantized weight. In bf16 precision the LoRA factors are BF16 too, so on a backend
that refuses a BF16 `OUT_PROD` (Metal and CPU) the trainer probes the op before the
first backward and sets `GGML_BACKWARD_MM=1` for ConvRot runs as well (#240); CUDA and
strict-f32 graphs are unchanged. The app still offers the ConvRot base only on CUDA
builds (`yue2JointTrainRunner.ts`). Same seed, first step on CUDA: bf16 GGUF ar_ce 3.6378 / nar_mse
1.18468 against ConvRot's 3.6373 / 1.18457. Preparation records a base only as
provenance, so one prepared dataset serves every base; a mismatch is a warning.

Training GEMMs run BF16 × BF16 on tensor cores with FP32 accumulation (2026-10-13), for
every recipe: GGUF base and head weights in their own BF16, trainable adapter factors
(LoRA A/B, LoKr w1/w2) through a BF16 cast at each use, which is how the PyTorch
reference trains. The inputs are already BF16-rounded, so only the summation order and
the factor rounding change. Three things keep the F32 path: quantized GGUF bases,
`CONVROT8` base linears (not `mul_mat`), and the frozen companion decoder. Its stored
F32 factors are not BF16-representable, and casting them moved the decoder's gradients
about 4%, so it stays exact; that costs no measurable time. On the reference workload a
step went from 4.24 to 3.31 s on an RTX 5090 (4.30 to 3.38 s with the companion).
`YUE2_AITK_PRECISION=strict-f32` restores the previous dispatch (every training GEMM
`GGML_PREC_F32`, attention as before) bit for bit; any other value than `bf16` or
`strict-f32` refuses to start. The run log names the precision at start-up, and each
`joint` line in `train.jsonl` records `gemm` plus how many GEMMs took each path
(`gemm_bf16`, `gemm_f32`). The casts need the HOT-ggml `bf16-mul-mat-grad` change.
`yue2-train-gemm-precision-test` checks both paths against each other at YuE2 shapes.
Since 2026-11-08 bf16 mode no longer sets `NVIDIA_TF32_OVERRIDE=0`, so the F32 GEMMs
that remain (quantized bases, the companion decoder) run on TF32 tensor cores on sm_80+:
2008 to 1880 ms a step on an RTX 5090 over 20 steps, loss within 0.025 %. strict-f32
still sets it, and an inherited `NVIDIA_TF32_OVERRIDE=0` turns TF32 off for an A/B. The
start-up log says which is in force (`[yue2-aitk] F32 GEMMs: ...`).

`YUE2_AITK_HOST_PROFILE=1` adds one `{"stage":"host_profile","step":N,"ms":{...}}` line per
step on stderr: host wall time per trainer phase (block executor phases, constants, gradient
transfer, clip, optimizer, head loss). A phase that ends in a blocking read includes the
wait for the GPU. LmOptim logs once per run where its scheduler placed each optimizer
graph (`[lm-optim] ... graph placement: CUDA0=N`, plus any ops that fell back to the CPU).

The AR head's CE + KL stays on the device by default (`Yue2AitkArHeadRows` in
`yue2-aitk-endpoints.h`, device mode of `yue2-aitk-head-loss.h`): the final-norm graph is
built once per expert pass, prediction rows are gathered with `get_rows`, each head chunk's
F32 gradient lands in a device tensor, and a scatter graph applies the BF16 rounding, the
`--ar-loss-weight` scale and `set_rows` into the norm's dy. Only the norm's dx comes back
to the host. Setting `YUE2_AITK_HEAD_HOST` (any value) restores the previous host
round-trip bit for bit; the cursor loss reads host rows, so `--cursor-weight` above 0 with
non-empty cursor frames takes the host path too. `yue2-head-loss-device-test` checks the two modes against each other on
the CPU backend.

`--device` takes any ggml device (`CUDA0`, `Vulkan0`, `MTL0`, `CPU`). Off CUDA the
same trainer runs from ggml ops: `--optimizer adamw` (the CUDA 8-bit kernel) maps
to LmOptim's AdamW, the head's CE + KL gradient is built from `soft_max`/`log`/
`get_rows`/`set_rows` instead of `joint_loss_cuda.cu`, and attention uses the
fused `FLASH_ATTN_TRAIN` ops where the backend has them (CUDA, CPU, Metal, and Vulkan
through the fork's `zzzz-vulkan-train-ops` change; the runner probes at start-up) and exact
attention elsewhere
(`YUE2_AITK_MATH_ATTENTION=1` forces it, for diagnosis). Exact attention keeps
`[keys, S, heads]` probabilities per layer, which rules out whole songs, and on
Vulkan it produced non-finite values in the AR forward (not root-caused; the
fused path does not). On Metal the backward recompute copies each layer's
attention O+LSE kept from the adapted forward instead of running the attention
forward again (`keep_attention` in `yue2-aitk-stack.h`; about 63 MB a layer at
S = 7,668). It is kept only if the device still has
`YUE2_AITK_ATTENTION_RESERVE_MB` (default 16384, above the largest measured
step footprint) free on top, otherwise the
trainer warns and recomputes, and it is freed as that backward returns.
`YUE2_AITK_ATTENTION_RECOMPUTE=1` restores the recompute; CUDA, CPU and Vulkan
always recompute. Two more Metal and non-CUDA savings are on by default, and each has a rollback
switch. On Metal the NAR's attention, which sees every key, is built with no mask and no
causal hint, so the attention backward can take its dS path; `YUE2_AITK_NAR_NOMASK=0`
restores the masked graph. Other backends always keep the mask (CUDA's pipelined kernels
need the hint). With `--kl-weight 0` and no teacher, the non-CUDA head loss skips the base
logits and the KL term and reports `ar_kl` as 0; `YUE2_AITK_HEAD_SKIPKL=0` restores the full
graph. This covers CPU, Vulkan and Metal, but has only been measured on Metal; CUDA uses
`joint_loss_cuda.cu` and is unaffected. On an M2 Max both left the losses and gradient
norm unchanged and cut about 1 s from a 13 s step. ConvRot needs CUDA: `CONVROT8` has no other kernel. The dataset must
use the joint trainer's schema-1 manifest, including matching model and source
hashes. Existing Legacy caches cannot be passed directly. Output must be a new
directory. Each `checkpoint-stepN` contains the combined
`adapter.safetensors` in the compatible layout, native inference exports `<trigger>_yue2_joint_ar_stepN.safetensors` and
`<trigger>_yue2_joint_nar_stepN.safetensors` (older runs wrote `native-ar.safetensors`
and `native-nar.safetensors`; both are still found), and `optimizer.resume`. Use both native exports for
generation. Resume into another new directory with `--resume` pointing
to that record; `--steps` is the total desired step count. Use the same dataset,
model, seed and CUDA ordinal.

`optimizer.resume` stores the LmOptim state (`adamw-lm`, `prodigy`, `muon`) as
F32, so a resumed run matches an uninterrupted one bit for bit. Records are
about twice their old size: about 1.4 GB for `adamw-lm` at rank 64 and about
2.4 GB for Prodigy. Records written before this
format (version 2) kept the moments 8-bit, which zeroed small second moments.
They still load, but the trainer prints a note that the run will drift from an
uninterrupted one. The CUDA `adamw` optimizer keeps 8-bit state in memory and
saves it as held.

The CLI emits JSON progress and records losses in `train.jsonl`. Ctrl+C requests
a stop at the next completed step and saves that state. The app's Windows Stop
action currently terminates the process, so only already published checkpoints
are guaranteed to survive it.

`--adam-eps` sets the AdamW epsilon of the LmOptim rule (`--optimizer adamw-lm`, or
Muon's AdamW-ruled parameters); without it the rule keeps 1e-8 and runs are
bit-identical to before (10 steps, old and new binary: same losses, same adapter and
`optimizer.resume` sha256). The 8-bit `adamw` kernel and Prodigy refuse the flag. The
value is written to the resume record and a resume with a different one is refused.
On CPU, `--adam-eps` with `--optimizer adamw` is remapped to `adamw-lm`.
At real LoRA gradient sizes the choice is not cosmetic: on a one-step fixture,
1e-8 against a reference run at 1e-6 moved the updated B factors by 25-82% (rel-L2).

`--fixture-replay <dir>` runs one pinned joint step against a reference fixture and
exits without training or saving: fixed window, stored flow noise and time, the
Direct-mode prefix and an integer-hash LoRA A with B = 0 (formula in
`engine/src/train/yue2-aitk-fixture-replay.h`). It writes step-1 losses, raw
B gradients, post-update B and step-2 losses to `<dir>/replay-native.json` plus raw
F32 dumps, for a parity check against another trainer's numbers.

In Training Studio, Joint Training is the only YuE2 training workflow: the card
always sends `trainingMethod: 'aitk'`, and the joint route refuses anything else.
Complete the latent, semantic-token and ABC cache stages, then prepare the joint
dataset and train; Start training runs any preparation that is missing. The
separate Legacy AR and NAR runners and their routes remain for API callers. The
joint checkpoint picker applies both native adapters for generation.

### Two training methods (2026-09-27)

The trainer had two recipes on this date; a third, `fixed-recipe`, is described at the top of
this section. Since the ear test the same evening, base-matched is the card's default: its
Fast, Balanced and Thorough presets (100, 200 and 300 updates of 4, 4 and 8 songs, a dim-128
LoKr, 60 s decoder crops, ten checkpoints each; `YUE2_JOINT_PRESETS` in
`server/src/contracts/trainingRecipes.ts`) all run it. The card's Legacy preset (2026-09-29)
runs the tuned recipe and HOT-Approved runs `fixed-recipe`. The Legacy seven-stage picker is
gone from the UI (its runners and routes remain).
**Tuned** is every default above and below this section: Prodigy, the KL anchor and KL
stop, lyric timing, the planner at a lower rate, the 60 s decoder crop; it stays in the server
and engine so old runs resume and refine. **Base-matched**
trains the way the YuE2 technical report (Appendix C, eq. 5-7, Table 14) says the base
was trained, as far as an adapter fine-tune can. The server applies it
(`applyBaseMatchedRecipe` in `yue2JointTrainRunner.ts`): fixed parts replace the request,
knobs fill in where blank. Every engine flag it uses defaults to the old behaviour, and a
run that never passes them is bit-identical to before (gate: 10 tuned steps and 5 native
AdamW steps, old vs new binary, identical losses and adapter sha256s, 2026-09-27).

| Report | Flag | Base-matched value | Tuned |
|---|---|---|---|
| `L = 0.25 L_AR + L_FM` | `--ar-loss-weight` | 0.25 (scales the CE gradient before the shared clip; Adam cancels a plain loss scale, so the clip is where it acts) | 1.0 |
| Adam β = (0.9, 0.95), wd 0.1 | `--beta1 --beta2 --weight-decay`, `--optimizer adamw-lm` | 0.9 / 0.95 / 0.1 | 0.9 / 0.999 / 1e-4, Prodigy |
| CE over generated tokens and closing markers (eq. 5) | `--ar-targets base` | MUSIC_START never a target; ABC_END only when the sheet is kept | every suffix token |
| Batch 256 | `--grad-accum` | 4 songs per update (Thorough preset 8; a "step" stays one update; each micro-step draws its own song from the same RNG stream) | 1 |
| Text and lyrics dropped separately or together | `--text-dropout --lyric-dropout --both-dropout` | 0.1 / 0.1 / 0.1 (rates are not published) | caption dropout 0.5 instead |
| Whole songs, no temporal split | `--nar-crop-frames` | 1500 (60 s crop), a deliberate departure since 2026-09-29: a blind test scored it level with whole songs at about 40% less time. `0` trains whole songs (about 12 GB VRAM, 11-15 s a step on a 4-minute song) | 1500 |
| Whole songs for the planner | `--ar-crop-frames` | 0 (whole song). Opt-in speed lever: the planner trains on the first N frames only (the reference trainer's `ar_max_tokens`, a prefix because the causal forward needs every earlier frame anyway). It never sees `MUSIC_END`, so it trains no endings. Ignored when the decoder trains whole songs (`--nar-crop-frames 0`) or the song is shorter than the decoder crop; refuses `--cursor-weight` above 0. Unheard | 0 |
| No KL anchor, no stops | `--kl-weight 0`, fixed steps | `--nar-drift` is passed anyway (since 2026-10-06), so the frozen teacher still runs for the planner's KL reading and the checkpoint meters (recon, drift): about 7% of every update, read-only | KL 0.2 + target |
| Warmup + cosine | `--warmup`, `--lr-schedule cosine-floor --lr-floor 0.1` | 3% of steps, floor 0.1 | wsd |

Condition dropout needs prefix variants the importer now always emits (`prefix_{full,off}_{nolyrics,notext,uncond}_ids`);
`yue2AutoPrepare` re-prepares a dataset that lacks them once, and both methods then share it.
Their texts: lyric-dropped `[Lyrics]\n\n` (the official instrumental tooling's lyric-free
request), text-dropped `[Tags]\n\n` (by symmetry, unpublished), unconditional = the bare
instruction with no headers (`protocol.py` `negative_prefix`; its off form has no ABC bracket,
so the suffix opens with MUSIC_START). `ace-train yue2-batch-check` is a CPU self-check of these
layouts and of `--ar-targets base`. Loudness: `yue2-preprocess --loudness-lufs` (0 = off) is now
passed by the server (`loudnessLufs` on the cache route, `YUE2_CACHE_LUFS` = -14 when the body omits it).
Every method trains at -14 since 2026-10-16. The batch chain's cache stage, and the single-dataset
preparation pipeline when it carries a joint stage (`cacheLufs` in `yue2PreparationPipeline.ts`),
re-cut a dataset at any other level (the target is part of the latent key, so the manifest is
rewritten and codes/sheet/align run again). No start path checks the cache's loudness; a direct Start trains on the cache as cut. The manifest always carries `loudness_target_lufs` now, 0 included, so an off cache
is not mistaken for a pre-normalisation one. Overs are peak-scaled, not clamped: a clip whose
sample peak passes 1.0 is scaled as a whole to -0.3 dBFS (`yue2_peak_scale` in `yue2-loudness.h`),
at preprocess before encoding (the gain is folded into the recorded `loudness_gain_db`, which is now
written for off caches too) and after the VAE decode, which overshoots loud material by about 4 dB.
Clips at or under 1.0 are untouched. Not matched: the report's two no-semantic-token
tasks (no published layout; our inference never runs them) and its flow-time distribution
(unstated).

Ladder previews (2026-09-27): for a base-matched run the relay fires `state.onRung` on every
`checkpoint` event (there are no `kl_mark` rungs), the runner's parallel preview chain renders one
take per checkpoint while training continues, and `listYue2AitkRuns` flags every complete
checkpoint of such a run as a rung so the Review route, `bestScoredRung` and the ladder cards
treat it as one. The route clears `preview.everySteps` for a parallel base-matched preview so the
run is not segmented. Parallel previews need a card of 28 GB or more (`PARALLEL_PREVIEW_MIN_VRAM_MB`, read from nvidia-smi for the selected GPU): training plus a render measured ~21 GB before the preview's 4-way recomposition, and on a 24 GB WDDM card the overflow spills into system RAM and both processes stall until the hang watchdog kills the run (2026-10-01). On a smaller or unknown card the runner turns `parallel` off at job start, so previews pause training (`--pause-on-kl-mark`, or `everySteps = saveEvery` for base-matched). The ladder UI (`Yue2LadderReview.tsx`: cards, scores, scoreboard, "Use this
rung", cleanup modal) was extracted from `RefinePanel.tsx` and is shared by the training card and
the hidden Refine tab; `finishScoredLadders` runs a base-matched ladder through `finish` only.

Legacy "Save every: AR KL" (2026-10-06): a fresh tuned KL-stop run may carry `klCheckpointEvery` (the route clamps it to 0.01-1), which passes `--kl-checkpoint-every` to the main run. Rungs preview on `kl_mark`, the freeze checkpoint previews on `planner_frozen` (rungs stop below the target), and once `state.frozenAt` is set every step-cadence checkpoint previews. `rungsOf` flags the frozen checkpoints of such a run as rungs; joint-phase step saves stay routine.

Stops and tails. Under `--lr-schedule wsd` with `--target-kl`, the tail starts when the KL trend line's slope says `reading + 0.5 × slope × lr_decay_steps` reaches the target, so a linear or cosine decay ends on the target; a reading at the target with no warning starts the tail as before. `--kl-overshoot-margin` (default 0.1, 0 = off) acts on the stop at once if the reading passes the target by that much during the tail. During decoder-only training `--recon-stop F` stops when a least-squares line through the last `--recon-stop-window` `nar_recon` readings (at least 3; the app passes 10) gains less than F across them; it was a two-point comparison until 2026-09-26, which the meter's 0.2-0.5% per-checkpoint jitter tripped on runs still improving 0.5-0.9% per 50 steps. The server option `reconKeepDelta` deletes a routine checkpoint whose `nar_recon` did not improve on the best kept one by that amount once a newer checkpoint lands (the newest is never deleted; rung checkpoints are exempt). Cleanup no longer moves the run directory (the rename failed on Windows file locks); it writes a `finished` marker in the run folder (`yue2RunFinished`), which drops the run from Finish scored (`finished` on `GET /yue2-review`) and from later otherRuns sweeps. Runs under the legacy `refined/` subfolder count as finished, and the index still relocates a run found there. Preview records carry `plan.attempts` (planner re-plans) and `composerReplans`, recovered from the seed the engine echoes back, which advances by 1000003 per recompose. The caption-source list is served by `GET /api/training/yue2-dataset-captions` keyed by dataset id, lyrics set or adapter path, composed from the dataset's sidecars (a `.yue2.txt` caption is the whole trained sentence; an ACE caption gets the genre/BPM/key tail), and matches the prepared manifest's style strings minus the trigger opener.

Lyric-timing supervision is optional. Training Studio has left it off for new
joint runs since 2026-09-28 (`Yue2TrainStages.tsx`); the route still turns it on
for a request that omits `lyricTiming`, at cursor weight 0.08. It reuses the
vocal-stem and forced-alignment stages; turning the option off skips those
optional stages and the cursor objective. Base-matched forces the cursor weight to
0 and `fixed-recipe` forces lyric timing off, so only the tuned recipe trains the
cursor. The timing weight is configurable and recorded with the run.

Checkpoint previews select the AR adapter at `preview.arScale`, default 0.6, and the NAR adapter at 1.0. The joint form and batch ladder defaults carry the same AR scale. Manual renders accept `arScale`, falling back to the run's saved preview scale and then 0.6. Baseline previews select no adapters.

Checkpoint previews are optional. The route leaves them off when a request omits
the `preview` block; the joint card sends the parallel ladder block above (one
300 s take per checkpoint, 12 decoder steps). Without a picked `previewSongId` they
use the prepared dataset's first sung track, with a fixed default seed of 424242.
Preview duration is bounded to 8 to 360 seconds (default 90), and
`previewMaxFrames` (default 2250, at most 9000) caps the semantic frames. Reaching
that cap is a preview limit, not a natural ending. Preview audio is stored with the run
and remains playable after reopening it. Native AR and NAR exports from the
same checkpoint should be selected together; their generation strengths remain
independent and default to 1/1.

The "Train multiple" control starts a server-owned batch (`yue2BatchRunner.ts`,
`POST /api/training/yue2-batch`). It applies one recipe to every selected dataset
and runs each dataset's stages in order: captions (when the recipe asks for
auto-captioning), cache, codes, sheet, stems and align (with lyric timing), train,
then the planner refinement for a tuned recipe (base-matched and `fixed-recipe` skip
it, as does `autoRefine: false`). A stage is skipped when the dataset's own status route already
reports it done, and every run gets its own output directory. The batch record
lives in `<training dir>/yue2-batches/<id>.json` and is saved at every
transition; after a server restart it comes back `paused` and resumes from the
record with finished work kept. `GET /yue2-batch` and `GET`/`DELETE
/yue2-batch/:id` read and cancel batches; `POST /yue2-batch/:id/items`, `/pause`
and `/resume` change a running one, and `POST /yue2-batch/finish` finishes scored
ladders.

The native cache preparation command reuses complete full-song caches:

```text
ace-train yue2-prepare-aitk --legacy-manifest yue2_preprocess.json --checkpoint base.safetensors --tokenizer yue2-lm-q8_0.gguf --model vae=vae.gguf --model semantic=tokenizer.gguf --model sheetsage=sheetsage.gguf --output new-prepared-directory
```

`--tokenizer` is the text tokenizer in an LM GGUF or a directory containing
`vocab.json` and `merges.txt`. `--model semantic` is the separate audio tokenizer.
Preparation validates geometry, payloads and captions, records model hashes,
and publishes a new manifest without changing the source cache. For caches
prepared by other YuE2 training tools, use `yue2-import-aitk-cache --help`.

`yue2-prepare-aitk --caption-dropout-trigger <text>` sets each item's dropout
style to exactly `<text>`, with no "in the style of" wrapper. It accepts one
line of at most 128 bytes and rejects CR, LF and NUL. Without this flag,
dropout uses the bare `--trigger` value, or an empty style if there is no trigger.
The Legacy `--trigger` injection still wraps non-empty full captions as before;
the dropout style does not use that wrapper. Lyrics and sheet conditioning stay
the same when caption dropout fires.

Set this flag during preparation, where the trigger-only prefixes are tokenized.
`yue2-joint-train --caption-dropout <p>` selects those prepared prefixes with
probability `p`; the trainer does not accept `--caption-dropout-trigger`.
HOT-Approved, the Fixed recipe preset, passes the trigger or, when none is set,
the dataset slug. Re-prepare datasets made before this change to populate those
prefixes correctly. The server invalidates the old Fixed recipe preparation
cache and re-prepares it automatically.

The prepared manifest's `trigger` and `style_template` (`upstream` or `bare`)
pass through to every native AR/NAR export, along with a trigger-prefixed file
name. Fixed recipe captions already open with the trigger, so after preparation
the server stamps `trigger` and `style_template: bare` into the manifest. A
Fixed recipe cache without that stamp counts as stale and is re-prepared.
Exports from older Fixed recipe runs carry neither field; the adapter catalogue
recovers the trigger from the run's prepared styles instead.

The synthetic native resume check reproduces uninterrupted training exactly.
Real full-song updates and native adapter merging have passed. Native and Torch
calculations have documented rounding differences; audio quality still needs
listening qualification. The ConvRot base requires a CUDA build and BF16-capable
NVIDIA hardware (Ampere or newer); a GGUF base runs on Vulkan too (whole-song
steps verified finite on an RTX 4090 over Vulkan, 47-95 s per song; not yet
ear-tested, and not yet run on AMD). Model Manager's **YuE2 Joint Training Pack**
contains the required weights; Legacy remains available on its supported backends.

### Live interval edits (GH #307, 2026-10-23)

`PATCH /datasets/:id/yue2-joint-runs/:jobId/params` with `{ saveEvery?, previewEvery? }`
(whole numbers from 1 to the step cap) calls `updateYue2JointParams`. It changes the job's
options, so later segments start with the new `--save-every` and preview cadence, and writes
`<outDir>/.params.json` (`{ rev, save_every, pause_at? }`) for the trainer running now. Every
segment gets `--params-file <outDir>/.params.json --params-rev <n>`, the edit revision its
arguments already carry. `poll_params` in `yue2-aitk-runtime.cpp`
runs once per step, before the save check: one `last_write_time`, and a read plus yyjson parse
only when the mtime has moved. Measured on the RTX 5090 workstation: 1.7-2.0 µs for the stat
and 17 µs for stat, read and parse, against steps of 0.3-5 s, so it polls every step.

Only a revision above `--params-rev` applies. Older ones are already in the segment's
arguments, and a `pause_at` left by an earlier segment would otherwise pause the next one at
once; an edit made after the server built the arguments (while it waits for the engine to
stop) still lands. A file that does not parse is retried next step (the server may be mid-write). An
out-of-range value is logged and skipped. Unknown keys are ignored. `pause_at` is written only
while the segment's pause comes from the preview cadence (not from a live plan check), as
segment start + previewEvery, clamped to the step cap. The checkpoint validator
(`yue2JointExpectedStep`) trusts only the engine: its `paused` event, its early stop, else the
step cap. An edit can land after the trainer's last read, so the requested pause is not proof. The route answers 409 when the run is not training
on this machine; the UI sends it with the training target, so a worker's run is reached
through the worker. `klCheckpointEvery` is not editable live.

### YuE2 preparation operations

`stages: ['joint']` with `mode: 'train-after-preparation'` is direct Start-training.
`Yue2PreparationPipeline.start` reads the artifacts before admission and returns 409 if
latents, codes or sheets are missing, plus stems and align when `lyricTiming` is set. No
preparation job is queued. Joint combined with other stages still requires `latents`.

### Optimise phase and base loss (2026-09-30)

YuE2's phase 2 is **Prepare** (the `preprocess` phase id, relabelled; `Yue2TrainStages section="prepare"`), then **Optimise** (`optimise`, `Yue2OptimisePanel.tsx`), then Train. The joint run-all chain takes a null training callback for prepare-only runs.

Optimise results live in `<source dir>/_hotstep-optimisation.json` (`yue2Optimise.ts`: `readOptimisation`, `writeOptimisationSection`), one top-level section per measurement, never in the database. `GET /datasets/:id/yue2-optimise` returns the file and whether the cache, codes and sheets exist; `POST /datasets/:id/yue2-optimise/base-loss` starts a `yue2-base-loss` job.

The job runs `ensureYue2PreparedDataset` (the same prepared dataset training uses, lyric timing off), then `ace-train yue2-joint-train --eval-base-loss` with `--cursor-weight 0 --ar-targets base --nar-crop-frames 1500 --seed 42` and the companion decoder. The engine loads the base, creates no adapter or optimizer, and for every item in dataset order prints `{"stage":"eval","item":i,...,"ar_ce","nar_mse","nar_mse_t":[...]}` and finally `{"stage":"eval_summary",...}`. Planner CE: `yue2_aitk_joint::base_ar_ce` (the frozen-teacher forward plus the head kernel at KL weight 0, full prompt, lead sheet kept). Decoder MSE: `nar_probe_prefix` + `nar_probe_predict(nar = nullptr)` against the flow target at t = 250 / 500 / 750, on a crop drawn by a per-item sampler seeded from `--seed` and the item index, so two runs agree bit for bit (checked on a 20-song album). It keeps the engine running (about 72 s for 20 songs on a 5090 beside a loaded engine). The bare `{"stage":"eval"}` event marks the start and carries no `item`.

`--eval-adapted` (with `--resume`, so the run's own `--steps` and flags) scores an existing LoRA/DoRA/LoKr checkpoint the same way, on the same per-item samples, once with the base and once with the restored adapters (no dropout, no update, no optimizer step; the optimizer is only built because the resume path restores through it). The adapted decoder is conditioned by the adapted planner prefix, as at generation. Output: `<output>/adapted-loss.jsonl` with `ar_ce_base/adapted`, `nar_mse_base/adapted` per item and an `eval_adapted_summary` line.

`--holdout-items 0,2` excludes those zero-based prepared-dataset indices from every training shuffle. Indices must be distinct, in range, and leave at least one training item. Without `--eval-dataset`, adapted evaluation scores only this split and reports `"held_out":true`. A separate prepared dataset passed as `--eval-dataset` takes precedence and scores every item in that dataset. With neither option, evaluation scores the training dataset and reports `"held_out":false`.

New resume metadata saves `holdout_items`, including an empty array for unsplit runs. Every resume, including evaluation, must supply the same split as a set; list order may differ. The saved training order must contain exactly the nonempty complement, with no duplicates, and the saved cursor must be strictly below that order's length. A checkpoint without split metadata is treated as an unsplit legacy run: it can resume with no hold-out, but adding a hold-out is rejected because those items may already have trained the adapter. Changing or removing a saved split also requires a new run. Resume never filters or resets the saved order or cursor.

`--eval-nar-window N` overrides the adapted-evaluation crop in frames, with `0` selecting the whole song; without it, evaluation uses `--nar-crop-frames`. `--eval-seed N` overrides the per-item evaluation draws, defaulting to `--seed`. Both affect evaluation only. They do not stop a run by themselves; the planner gate below is the only hold-out stop, and it is off unless asked for.

### Held-out planner gate

`--planner-gate-every N` (GH #316; 0 means off) scores the live planner on the `--holdout-items` every N updates: token-weighted planner CE through the `--eval-adapted` draws, no dropout, on samplers of its own, so the training RNG and optimizer are untouched. Once `--planner-gate-min-epochs` (default 5) training epochs have run, counted as song draws (steps times grad-accum) over the training songs only, `--planner-gate-patience` (default 3) readings in a row not below `best - --planner-gate-epsilon` (default 0.001, absolute CE) freeze the planner where it is, with no rollback, through the existing freeze. The decoder then trains `--nar-extra-steps` more, which the gate requires and which no longer needs `--target-kl`. The gate refuses an empty hold-out, `--eval-dataset`, `--lr-schedule wsd`, and the scoring modes; `--eval-adapted` and `--meter-only` score a gated checkpoint without the gate flags. `--eval-seed` seeds the readings (the engine falls back to `--seed`).

The gate's settings, best, best step and stale count ride in the resume record. Every resumed segment, frozen or not, must pass the same gate flags and `--eval-seed`, or the engine refuses with "resume planner gate mismatch". Stdout carries `{"stage":"planner_gate","step","epoch","ce","best","best_step","stale","patience","armed","items","tokens","ms"}` per reading (also in `train.jsonl`) and `{"stage":"planner_frozen","step","reason"}` with reason `kl`, `planner_ce` or `manual`.

The server side is in `yue2JointTrainRunner.ts`. A request carries `plannerGate: { every, minEpochs, patience, epsilon, evalSeed, holdoutIds }` plus `narExtraSteps` above 0; the recipes that zero the decoder budget (base-matched, fixed recipe) keep the gate's value. `holdoutIds` are prepared-manifest item ids, which are the names preprocessing gives the source songs (`yue2StagedNames` over the dataset's included audio files, minus the old `.engine.wav` conversions). `GET /datasets/:id/yue2-joint-holdout?seed=` computes them from that current inventory and never from a cache, so the card shows the split before any stage has run and a changed song list changes the answer. `planPlannerGateHoldout` picks the default split (20% rounded up, at least 3, a seeded shuffle of the sorted ids, one training song left); Start uses the same pick. The runner resolves the ids to indices in the manifest it trains on, logs the count and ids, and stores both in the run record. A resume inherits the saved gate and indices, keeps the saved `narExtraSteps`, and fails if the ids no longer sit at the saved indices. A song that preprocessing drops (for example one that fails to encode) is missing from the manifest, and the run then fails at start with the missing ids named. The runner emits each reading as a `metric: 'eval'` event with a `plannerGate` field and leaves `job.phase` alone; `planner_frozen` logs by reason. Planner refinement (`refinePlanner`) and `autoRefine` are refused on a gated run, since they unfreeze the planner with a zero decoder budget. The defaults are starting settings, not validated ones.

### Dataset profiles (Dataset-Calibrated Training, in progress)

The aim is a recipe sized to the album instead of one recipe for every dataset.
It is opt-in: the route only acts when a fresh base-matched request sets
`calibrated: true`, and without it no code path changes.

With it on, `POST /datasets/:id/yue2-joint-train` calls `ensureDatasetProfile`
(reuses the saved profile when it covers the same audio files) and
`calibrateYue2Length(minutes, steps, saveEvery)`: both are scaled by
`minutes / 45`, clamped to 0.6-2, so the rung count stays the same. This runs
before `applyBaseMatchedRecipe`, so warmup follows the new step count. The result
is stored as `calibration` in the run options, is logged at job start, and goes
into each rung score's settings snapshot (which now also records `method` and
`gradAccum`). The handler is async for this.

Evidence kept for fitting: `yue2_album_scores` holds one 1-5 verdict per run, plus `instruments` and `vocals` directions (`under` / `right` / `over`, which the report scores -1 / 0 / +1)
(`GET/PUT /datasets/:id/yue2-album-score`, set above the ladder).
`deleteYue2AitkRun` and the rung-pick cleanup copy each segment's `train.jsonl`
to `<training dir>/datasets/<slug>/train-logs/<jobId>-<segment>.jsonl`, and
`clearPreparedCaches` saves the lead sheets to `yue2-sheets.json` first, which
the profile reads once the cache is gone. Since 2026-09-30 every clear (rung-pick cleanup, the batch's clear-cache
option, the card) keeps `yue2-latents` (`YUE2_CORE_CACHE`: latents, codes, sheets, alignment, prepared sets) unless
`includeYue2Core` is passed, which only the card's opt-in toggle does; stale data is rebuilt by the stages' own checks. The report takes the album score as a
target and reads archived loss logs.

`server/src/services/training/datasetProfile.ts` measures a dataset and saves
`<training dir>/datasets/<slug>/dataset-profile.json`. It reads only what
survives a cache cleanup: the audio (one ffmpeg pass per song for EBU R128
loudness and range, spectral centroid, flux, flatness and frame RMS) and the
sidecars (BPM folded into one octave, key, genre, caption words and how much the
album's captions agree, lyric words per second). When the YuE2 cache is still on
disk it adds lead-sheet measures from `classifyYue2Score`. A 15-song album takes
about 2 s.

`server/scripts/dataset-profile-report.ts` profiles every album with scored
YuE2 rungs, joins the profiles with `yue2_rung_scores`, and ranks each measure
against the ear targets (peak likeness and overall, best step, first step at
likeness 4) within one recipe. `--calc <calc.py>` also runs a third-party
settings calculator for comparison. The report goes to
`docs/plans/dataset-calibration/` (local-only).

First run (2026-09-29, 37 albums): under the tuned Prodigy recipe, albums with
more minutes of audio took longer to reach likeness 4 (Spearman 0.55, n=23), and
albums with denser lyrics got there sooner (-0.52). The third-party calculator's
step count tracked it at 0.23. Peak scores sit at the ceiling (22 of 27 runs
peaked at 5), so the scores can show how fast an album trains but not how well.
Base-matched has 4 full ladders, too few to rank. The lead-sheet measures are
confounded: only albums whose cache was never cleaned up have them.

### Training workers and the mirror

`ui/src/services/trainingApi.ts` splits every training call into two bases:
`request()` (job control — start/cancel/status, via the switchable `API_BASE`)
and `localRequest()`
(run records, ladder review, finished-run previews, rung/album scores, cleanup
and finish, always `/api/training` on this machine). A worker only trains;
reviewing and scoring a run — even one a worker trained — always happens here,
so a worker with no `blindLabels` set yet can never make the Review page list a
blind ladder in step order (`Yue2LadderReview.tsx`'s `labelsPending` guard
refuses to render a blind ladder at all until the run's own labels are in).

`pullLinked` (`POST /api/workers/:name/pull`, the Train on bar's "Fetch finished
adapters") copies each AR/NAR pair the worker itself has linked, plus its train.jsonl
file(s) and a `<jobId>.json` sidecar noting the kept step, through the worker's
`adapter-file` route into `trainLogArchiveDir` here. It only sees the worker's
*currently linked* pairs.

`server/src/services/training/yue2Mirror.ts` copies every run a worker holds into this
machine's own `yue2-joint-adapters/<folder>` as its files land, so a worker run becomes an
ordinary local run tagged `origin: { worker, remoteJobId }`. Review, scoring, Use this
rung, cleanup, Finish scored and linking never touch a worker. `startYue2Mirror()` runs
one loop per configured worker (every 15 s while a run is training, has previews
rendering or has files left to copy, otherwise every 120 s; an offline worker is logged
once per outage). `POST /api/workers/:name/mirror` runs a pass now, or joins the one
running, and returns its `MirrorStatus`; `GET` returns the last one.

The worker side is three routes in `routes/workers.ts`'s `workerRouter`, backed by
`trainingWorkers.ts`:

- `GET /api/training/worker/yue2-mirror` (`workerMirrorManifest`): every unfinished run
  in the worker's index with its status, options, blind labels, preview catalogue, a
  `previewsBusy` flag, and `files` (forward-slash path, size, mtime; no hashing). The
  files are the run's `train.jsonl` logs, `style-norms.json`, each complete checkpoint's
  AR and NAR weights (either naming) and `meters.json`, and each done
  preview's `.wav` and `.score.abc`. Never `optimizer.resume` or `adapter.safetensors`:
  nothing here reads them. ace-train writes a checkpoint into `.checkpoint-stepN.tmp` and
  renames it into place (`yue2-aitk-runtime.cpp`, `save_checkpoint`), so a checkpoint
  folder holding both weight files is complete. A run finished on the worker (a
  `finished` marker, or a NAR-further run under `refined/`) is left out: landed here it
  would look like an ordinary unfinished ladder, which the next cleanup sweeps away.
  Previews render inside the run's training job or on the GPU lane; a `rendering` record
  with neither alive was cut off by a worker restart and is reported `failed`, so it
  neither holds the run nor shows as rendering forever.
- `GET /api/training/worker/yue2-mirror-file?jobId=&rel=` (`workerMirrorFile`): `rel` is
  matched against that run's manifest entries, never joined onto a path. The sha256 goes
  in `x-sha256`, hashed by streaming and cached per path, size and mtime, so each file is
  hashed once per worker process; only the first `size` bytes are hashed and served, so a
  log still growing stays consistent. `x-mtime` is the mtime served. `HEAD` returns the
  same headers with no body.
- `DELETE /api/training/worker/yue2-mirror/:jobId` (`deleteWorkerMirrorRun`): drops the
  folder and index entry. 409 while the run is training, while a batch is active or the
  run ended under two minutes ago (a batch's refine stage and the runner's automatic
  refinement resume from its `optimizer.resume`, queued a moment after it ends), while
  any job for its dataset is queued or running, or while the GPU lane is busy (a manual preview render bypasses the
  labeling queue); 400 for an `output` that is not a strict descendant of the ladder root.

A pass takes each manifest run under `withYue2RunLock` of its local jobId, so cleanup
(which holds the same lock throughout) and deletes never interleave with it. A run with no
dataset here is skipped and logged once. The local record is found by `origin`, else
created at `freeYue2RunDirectory(yue2-joint-adapters/<worker folder>)` under the worker's
own jobId, and recorded in the index before any file lands so the disk reconcile never
imports the folder as a run of its own. Every manifest `rel` must match a fixed pattern
(`MIRROR_REL`). A file is fetched when it is missing, its size differs, or its mtime differs
from `<output>/.mirror.json`, the ledger of `{ size, mtimeMs, sha256 }` per file; files
already recorded are never re-hashed. A file already here at the worker's size but not in
the ledger (a migrated run, or a pass cut off between rename and ledger write) is hashed
locally once and compared with the worker's hash from a `HEAD`; a match is recorded
instead of downloaded. The first pass over each run folder in a process removes the
`.part-<uuid>` and `.mirror.json.<uuid>.tmp` files a restart left behind. Each download goes to a unique `.part`, is hashed
while it streams, compared with `x-sha256`, and renamed into place, with one retry on a
mismatch. Small files go first, so meters land before weights and previews before
checkpoints. The preview catalogue is rewritten from the worker's, with a done preview
whose audio has not landed shown as `rendering`; previews rendered on this machine keep
their own records. Checkpoints come from the local disk scan, as for any run. Once the
worker status is `done`, `failed` or `cancelled`, no preview is rendering, and every file
is copied and verified, the pass asks the worker to delete its copy; a 409 is retried next
pass. `interrupted` runs are kept on the worker so they can resume there.

A run marked finished here is never fetched again (cleanup may have deleted its other
checkpoints and previews); the pass only asks the worker to drop it. Discarding a mirrored
run here (`DELETE /datasets/:id/yue2-joint-runs/:jobId`, or cleanup's other-runs sweep,
both through `deleteYue2Run`) appends `{ worker, remoteJobId }` to
`<training dir>/yue2-mirror-tombstones.json` first, so no pass recreates it; passes keep
asking the worker to delete a tombstoned run until it agrees.

NAR further training stays refused for a mirrored run (`yue2BatchRunner.ts`'s `nar`
stage): it resumes from the rung's `optimizer.resume` and the run's prepared dataset
(`yue2-joint-train`'s `resumeRunId` branch), and neither is copied.
`reconcileYue2AitkRunsAtStartup` skips any run with `origin`: its `running` status is the
worker's, which this machine restarting does not change. The `live` flag on the run list
and Review rows follows that mirrored status.

`migrateYue2OriginRuns()` converts runs copied by the older ladder pull, which kept a
`remote:<worker>:<id>` jobId (rung scores are keyed by it, so it stays) and staged
unfinished runs under `yue2-joint-adapters/_remote/<worker>/`. It moves each staged folder
to `yue2-joint-adapters/<name>` (`-N` on a clash) with `moveYue2AitkRun`, repoints album
presets, `yue2_rung_scores.checkpoint_dir` and `yue2-linked.json` with
`rebaseYue2JointLinks` (folder and records go back on failure), notes the new folder on the
archived loss log, rewrites a moved record, or one still carrying the pull's
per-checkpoint fields, from the disk scan (dropping them), and removes the empty `_remote`
tree. It scans the catalogue once and leaves clean records alone, since it runs on every
boot before the server listens. It is idempotent.
`startYue2Mirror()` awaits it before the first pass, so no pass writes into a folder about
to move. The first mirror pass then fills in the rung weights the worker still holds.

#### MM3 on a worker

`POST /api/workers/:name/mm3-dispatch` (body `{ datasetIds, clearCache, recipe }`, the
`mm3-batch` body) runs the same dispatch as `yue2-dispatch` with `family: 'mm3'`: no
captioning, a check that the dataset is built with its manifest at
`<sourceDir>/dataset.json` (the only path `upsertPushedDataset` adopts on the worker) and
has at least one `.mm3.txt`, then the push, then `POST /api/training/mm3-batch` on the
worker. A dispatch or a worker batch of the other family is never joined. The pushed
manifest keeps this machine's absolute `audio_path`s; `pm_resolve_audio` in
`preprocess-io.h` falls back to the file beside the manifest.

MM3 runs need no index: a run is its folder, attributed by `hotstep-run.json`. The worker
offers `GET /api/training/worker/mm3-mirror` (`workerMm3MirrorManifest`: runs of datasets
whose `sourceDir` is under `worker-datasets/`, with `live` while a job for the dataset or
the GPU lane is busy) and `mm3-mirror-file?run=&rel=` (same headers as
`yue2-mirror-file`). Files offered: `hotstep-run.json`, `train-log.jsonl`,
`train-console.log`, every file of a `ckpt-N/` once its weights' `.json` sidecar exists
(the engine writes the folder in place and the sidecar last) and `previews/*.wav|json`.
Never `resume-state.*`.

`mirrorMm3Runs` in `yue2Mirror.ts` runs inside each pass and lands a run at
`mm3-lm-adapters/<run>` with the same `.mirror.json` ledger. A file is fetched again when
the worker's size or mtime moves past the ledger's; a file in the ledger but gone from
disk was cleaned up here and stays gone. A fetched `hotstep-run.json` keeps the local
`blindLabels`. `<training dir>/mm3-mirror.json` records each `{ worker, run }`: a recorded
run whose folder is gone is marked `discarded` and never copied again, and a run name that
already exists here without a record is refused. Once a run seen `live` by a pass is no
longer live and fully copied, `linkMirroredMm3Run` points the dataset's album presets at
its last checkpoint (`pointMm3PresetsAt`, as `refreshLinkedMm3Presets` does locally),
once. Nothing is deleted on the worker.

## YuE2 NAR audio-domain loss

`yue2-nar-train` gained two flags on 2026-10-17. `--companion` merges the
tokenizer's companion decoder adapter under the NAR before training, as every
render does, and the export records it; `--manifest` may repeat, so several
caches train as one set.

`ace-train yue2-nar-train --audio-loss <w>` adds `w * aux` to the flow MSE on steps
whose `t <= --audio-tmax` (default 0.4). The clean-latent estimate
`x0_hat = x_t - t * v_pred` is decoded through the frozen VAE over `--audio-frames`
(150) frames plus `--audio-margin` (25) each side, at a random place in the clip,
and the scored span is compared with the source recording. `aux` is the
`spec_loss` of the joint_v6 trainer that produced the companion adapter: L1 of
log-mel (128 HTK mels over |STFT 2048|), plus 0.5 times the multi-resolution STFT
(spectral convergence and log-magnitude L1 at 512, 1024 and 2048, hop N/4), plus 0.5
times the difference in side-to-mid energy, all on mono except the width term.
Frames are not centre-padded and the mel hop is 512, not 480.

The reference is the source decoded the way `yue2-preprocess` decoded it (WAV/MP3
natively, anything else through `--ffmpeg`) and scaled by the source's
`loudness_gain_db`, which is the audio the latents were encoded from. A manifest
without that field is refused. Whole tracks are cached LRU. Once per run the trainer
decodes the first scored clip's clean latents and reports the lag of best
correlation against the reference; it should be 0.

The inference decoder (`yue2-vae-graph.h`) cannot be backpropagated on CUDA:
`IM2COL`'s backward has no CUDA kernel and `PAD` has no backward. So
`train/yue2-nar-audio-loss.h` rebuilds the decoder in `[C, T]` layout from ops that
do: a k-tap conv is k matmuls over shifted contiguous views, padding and the
transposed conv's overlap-add are `ACC` onto a persistent zero canvas. STFT frames at
hop N/4 split into four non-overlapping sets, each a contiguous view, so one frame
set is one matmul against a windowed DFT basis.

`--fd-check --audio-loss 1` checks the rebuilt decoder against `yue2_vae_decode`
(1e-6 relative), self-tests the decoder and the spectral loss separately, probes
`d(aux)/d(x0_hat)`, then runs the usual LoRA probes on the audio term alone. Pass
`--manifest` so it uses a real clip and its audio: on N(0,1) latents the decoder is
so rough that a central difference only turns linear at steps near 1e-6 of the
input. Even on real latents the decoder needs steps under ~1e-5 of the input, so the
audio gradient is correct but very local.

Cost at the defaults: about 5.5 GB more VRAM and 0.22 s/it against 0.18 for the
flow-only pilot (rank 128, 10 s clips). Near-silent references (hidden tracks,
gaps) push `aux` to 12-18 against ~1 elsewhere, through the log terms; the global
clip bounds the step.

Pilot result (HOTSTEP-353, 2026-10-18): the broad pilot rerun with `--audio-loss 2`,
same data, seed and draws, benched on 20 held-out tracks. Against the flow-only
pilot's final checkpoint it held the vocal envelope (+0.027, 20/20 tracks) and the
top two bands (+2.3 dB, closer to the source), but narrowed the bass less (+1.4 dB
side level, 16/20). It did not beat the flow-only step-3000 checkpoint, and its
vocal envelope stayed at the companion-only baseline. Its checkpoints varied less
from one to the next than the flow-only ones did. On the bench alone it does not earn
a place in the full broad run at this weight; the listening test decides.

## MiniMax-Music3 (MM3) LM adapters

MM3 planner-LM adapter training is a separate pipeline from everything below
(ace-train `mm3-codes` / `mm3-lm-train`, Training Studio's MM3 card). Its
recipe, defaults, and the institutional knowledge behind them — the acoustic
loss that keeps vocal timbre intact, structured/tiled crops, the fd gradient
tripwire, render dials — live in
[.claude/skills/mm3-lm-adapter-training/SKILL.md](../../.claude/skills/mm3-lm-adapter-training/SKILL.md).
Backend/runtime facts (endpoints, sampling knobs, replay) are in
[.claude/skills/mm3-backend/SKILL.md](../../.claude/skills/mm3-backend/SKILL.md).

### `--attn exact|flash` on `mm3-lm-train` (2026-09-05)

The same fused op as the DiT and AS1.5 LM trainers
(`GGML_OP_FLASH_ATTN_TRAIN`/`_BACK`, closed 2026-09-01), now wired into the
MM3 planner-LM trainer. The CLI default is `exact`; the in-app recipe has used
`flash` since 2026-09-06 (`MM3_LM_DEFAULTS.attnBackend`). On its first day the
flag was refused together with `--prefix-frames > 0` or `--prefix-n > 0`,
because a KV prefix makes the mask rectangular. Since 7070238e (2026-09-06)
both LM trainers probe the fused op at `S_kv = n_pfx + S` and compose flash with
either prefix. The `mm3-train-lm` and `mm3-resume-lm` routes now coerce
`attnBackend` to `exact` only when the engine's GPU backend is Vulkan or Metal
(`engineSupportsFlashAttnTraining()` in `aceTrain.ts`), and echo the resolved
value in the response.

Measured on an RTX 5090 (32 GB), `mm3-lm-f16` and `mm3-lm-q8_0` bases,
a 12-track album (longest ~11,178 frames), rank 256/alpha 256,
Muon, checkpointed. Peak VRAM is whole-device `used` including the ~10.3 GB
resident base:

| crop (frames) | exact peak VRAM | flash peak VRAM |
|---|---|---|
| 750 | 18,626 MB | 18,634 MB |
| 1,500 | 19,098 MB | 19,084 MB |
| 2,500 | 21,124 MB | 19,706 MB |
| 3,500 | 25,939 MB | 20,332 MB |
| 4,300 | 28,631 MB (starting to spill) | — |
| 5,000 | 30,293 MB (spilling) | 21,272 MB |
| 6,000 | OOM | — |
| 9,000 | — | 23,832 MB |
| 11,200 (longest track) | — | 27,244 MB |
| 19,000 | — | 31,473 MB (spilling) |
| 21,000 | — | OOM |

Flash is within noise of exact up to ~1,500 frames — under gradient
checkpointing the small softmax already disappears into allocator slack — then
saves VRAM growing to ~9 GB by 5,000 frames. Practical spill threshold on this
card is ~29 GB used regardless of mode; past it step time balloons 2–8× before
`cudaMalloc` actually fails, so "still fits" is not the same question as
"still fast". **Usable crop ceiling: ~4,300 frames at `--attn exact`, at least
11,178 frames (this corpus's longest track, no OOM reached) at `--attn
flash`** — roughly 2.6× on this measurement. `--attn flash` resolves to tf32
on this card; `--attn flash-f32` pins f32. Paired 20-step run at the shipped
recipe's crop (750, same seed and crop sequence both arms): 2118 ms/step mean
flash vs 2215 ms exact, maximum loss drift 1.9e-4 over 20 steps. On this date
nothing had been heard and the recipe trained at crop 750, where flash measures
no benefit. On 2026-09-06 flash tied exact in a blind test (68 vs 72 of 90) and
joined the stacked recipe below; the whole-song presets since 2026-09-11 depend
on it. The whole-graph (non-checkpointed) finite-difference gate could not be run in any
mode on this 32 GB card — both the exact and flash naive arms need ~27.5 GB on
top of the 16 GB f16 base — so flash is proven correct only on an F32-isolated
2-layer slice, not the full model. Full per-frame numbers and fitting caveats:
`.claude/skills/flash-attn-training/SKILL.md` §3/§7.

### Attention mask upload and the step profile (2026-11-06)

`lm_ckpt_upload_mask` and the MM3 loop now build the causal mask with
`lm_mask_set_causal` (`train/lm-common.h`): row fills written straight in the
buffer's type (F16 under flash, F32 under exact), with no F32 staging vector
or per-element conversion. The same bytes as before, checked by
`engine/tests/test-lm-causal-mask.cpp`. The MM3 loop also marks the
checkpointed path's last-S cache, because both wrote the same `t_msk` and each
rebuilt it when S changed. Whole-song crops change S on almost every step, so
that was two ~100 ms host builds per step at S 7,178. The ACE trainer's naive
path (`upload_mask` in `train/lm-train-run.h`) uses the same call since later
that day; it already kept one last-S cache, so there it only drops the F32
staging. `HOTSTEP_LM_PHASE_TIMER`
now starts its clock before the mask and token uploads. The measured step
breakdown (FP32 SIMT GEMMs are 58 % of kernel time, flash attention 24 %) is in
[.claude/skills/mm3-backend/reference/training.md](../../.claude/skills/mm3-backend/reference/training.md).

### BF16 and TF32 levers on `mm3-lm-train` (2026-11-06)

Three opt-in flags, all off by default, so a default run is unchanged (a
flags-off 20-step run reproduced the earlier binary's losses exactly):

- **`--weights bf16` on any base.** Lever A used to need a BF16-native base.
  `lm_linear` now casts a non-BF16 projection to BF16 in-graph (a quant goes
  through F32, since ggml-cuda has no quant-to-BF16 copy) and keys the
  out_prod rewrite by the cast, so the shipped q8_0 base reaches
  `cublasGemmEx(CUDA_R_16BF)` in both directions. 1.65x per step, VRAM level.
- **`--attn flash-bf16`.** `lm_attn_flash` casts Q/K/V to BF16 with
  `GGML_PREC_BF16` and sets causal hint 0 when there is no KV prefix, as
  `yue2-aitk-graph.h` does. 1.19x. `dit_flash_probe` (`flash-prec.h`) takes
  the precision so the probe asks about the BF16 kernels.
- **`--tf32 on`.** The trainer sets `NVIDIA_TF32_OVERRIDE=0` before the CUDA
  backend starts, which is why every F32 GEMM ran `cutlass_80_simt_sgemm`.
  The flag skips it. 1.39x alone, +1.4 % on top of the other two.

`--weights bf16 --attn flash-bf16` measured 2.32x (6.64 to 2.86 s per
micro-step) on a whole-song recipe, max per-step loss delta 0.57 % over 20
steps. `mm3-lm-probe` accepts `--weights` and `--attn` and passes on f16 for
every arm. The `--fd-check` finite differences fail for `flash-bf16` on
attention-side probes while the route check passes; the full table, the
kernel split after the change and the gate numbers are in
[.claude/skills/mm3-backend/reference/training.md](../../.claude/skills/mm3-backend/reference/training.md).

The server exposes the first two as one request field, `trainPrecision`
(`fastest` default since 2026-11-06, `fast` = `--weights bf16`, `fastest` =
`--weights bf16` plus `--attn flash-bf16` when the resolved attention is
flash, `full` = FP32). The recipe projection always sends the form's value, so
an explicit Full is not lost to the server default; a run manifest without
the field resumes as `full`. `recommendMm3Config` adds the BF16 weight stage
(`MM3_VRAM_MODEL.bf16StageMb`) for the default precision on a non-BF16 base. The mapping is
in `buildMm3TrainLmArgs`; the run manifest records the field, so a preview
relaunch and the resume route keep it, and the resume route accepts it as an
override. Since 2026-11-08 `fast` and `fastest` also pass `--tf32 on` (their
big GEMMs already round to BF16, so TF32 on the leftover F32 GEMMs is the
smaller rounding; 1.09x on a 4090) and `full` keeps strict FP32.

### Step cost after the BF16 levers (2026-11-06)

- **The PiSSA frozen branch adds a negated term instead of subtracting.**
  `lm_linear` scales `A0 x` by `-s` and adds `B0` times it, where it used to
  scale by `s` and subtract. The bits are the same (negation is exact in
  IEEE arithmetic and a GEMM of a negated input is the negated GEMM), but the
  backward of SUB is a NEG over `[out, S]` at every site, about 1.2 % of an
  MM3 step's kernel time under `--weights bf16 --attn flash-bf16`, and ADD's
  is nothing. 20-step losses, gradient norms and depth losses are identical to
  the previous binary with and without the BF16 flags.
- **The acoustic-loss head runs its weight GEMMs in 2D.** `mm3_depth_train_head`
  fed the frozen depth decoder `[in, 8, K]` activations, which ggml-cuda runs
  as K batched products of 8 columns, re-reading each F32 weight per frame.
  Reshaped to one `[in, 8K]` GEMM, the head's compute at K = 128 fell from
  161 to 65 ms per micro-step. The products are the same, but the kernel sums
  them in another order, so results are not bit-identical: over 20 steps,
  loss within 0.18 % (mean 0.02 %), depth loss within 0.06 %, step 1's loss
  unchanged; the head's finite-difference tripwire passes as before.
  Whole-song recipe: 2.85 to 2.70 s per micro-step with
  `--weights bf16 --attn flash-bf16` (1.05x), 6.69 to 6.54 s with the
  default flags.
- **`--weights bf16` on a q8_0 base converts each layer's weights once per
  graph, outside it.** The in-graph cast went q8_0 to F32 to BF16, writing and
  re-reading a full F32 copy of every projection (12.5 % of kernel time). On a
  CUDA build `lm_ckpt_alloc` now keeps a one-layer BF16 stage (368 MB on MM3)
  and `lm_ckpt_layer_w` fills it from layer l with `q8_bf16_cuda.cu` before
  each layer graph, which then reads it as a BF16 base. The kernel does what
  the two ggml copies do (F16 scale to float, times the int8, round to BF16
  nearest-even), so the 20-step run reproduces the previous commit's losses,
  gradient norms and depth losses exactly. 2.70 to 2.48 s per micro-step
  (1.09x), peak VRAM 19,087 to 19,465 MB (the stage; the compute buffer's
  peak is elsewhere in the graph, so the dropped F32 copies do not lower it).
  Other bases and other backends keep the in-graph cast.
- **`--pissa-fold on` (opt-in, `mm3-lm-train`).** With the stage in place, the
  frozen PiSSA/HOT-PiZZA term can go into the weights: the stage kernel writes
  `bf16(W - s B0 A0)` and `lm_linear` skips the frozen branch
  (`LmLayerOpts::pissa_folded`, set only in `lm_ckpt_micro_step`, never for an
  adapter-off pass and never for the KV-prefix builders, which read the
  unstaged base). That removes the `A0 x` and `B0 t0` GEMMs, an `[out, S]` add
  per site and their backward: 2.47 to 2.25 s per micro-step (1.10x) under
  `--weights bf16 --attn flash-bf16`. It is the same function, but
  `W - s B0 A0` is rounded to BF16 instead of `W`, so it is not bit-identical.
  Against the f32-window run over 20 steps it is no further off than the
  unfolded BF16 path (loss max 0.34 % / mean 0.06 %, unfolded 0.65 % / 0.07 %).
  The fd gate cannot test it (the F32-isolated layers never take the BF16
  path) and it has not been heard, so it stays off; the server does not pass
  it. Refused for DoRA, HRA, HiRA, LoHa, LoKr and plain PiSSA under rank
  dropout, where the frozen term is not constant.
- **`HOTSTEP_LM_PHASE_TIMER` now sees that head.** It ran outside the timer,
  so its GPU time was reported as `build(host)`: of the 274 ms "host" per
  micro-step, 161 ms was this head on the GPU. True host time is about 85 ms
  (3 % under the BF16 levers), and graph construction in the 36 backward
  segments is about 2.5 ms of it. Building graph i+1 while graph i ran was
  tried and measured 0.3 % (noise), so it was not kept; graph reuse and
  Lever B have nothing left to amortise here.

### HOT-PiZZA — the MM3 default method (2026-09-06)

HOT-PiZZA is PiSSA with principal-subspace dropout. It was called HOT-PiSSA on
its first day; the engine still accepts `--hot-pissa` and reads the
`hot-pissa` identity from that day's files.

`--hot-pizza` (both LM trainers; implies `--pissa`) is PiSSA with the
rank-dropout mask applied to the principal component itself rather than to
the delta. The training forward is `y = W x + s (M B A) x - s (B0 A0) x`: every
micro-step a random `--rank-dropout` share of the base's own top-r singular
subspace is deleted at every site and the rest scaled `1/keep`, while B A
(initialised to that subspace) fits the album. Plain `--pissa` masks both
branches so the mask only touches the delta. It was found as a masking bug
on 2026-09-05 and kept on purpose: in the blind MM3 method test on
albumA (nine methods, three songs, everything else locked) its adapter
scored 68/90 and, rated again against the corrected PiSSA and LoRA, 66.5,
top both times with no plan failures, while corrected PiSSA produced drone
plans in half its renders (39.5) and LoRA sat at 60–64. One album so far; a
second artist is the pending validation. The export is either the DELTA
form (rank r, adapter-only, 0.7 GB at r128 F16) when a residual file sits
beside the base, or an ordinary rank-2r LoRA (F16, 1.4 GB) when none does or
`--pissa-standalone` is passed; see "PiSSA residual" below.
Training loss reads high under it by construction (the trailing mean sat at
2.7–5 where LoRA reads 1.8–3), so stop on steps: the in-app recipes stop on
steps (`stopMode: 'steps'`), and the checkpoint rated by ear that day was the
step-500 one. Resumes the same way `--pissa` does (resume format v5).

#### PiSSA residual: the frozen half shipped once (2026-09-09)

Half of every PiSSA/HOT-PiZZA export is the frozen pair A0/B0, the base
weight's own top-r singular directions, identical for every adapter on the
same base. `engine/src/pissa-residual.h` defines a file that carries it once:
`<base stem>.pissa-r<rank>.safetensors` beside the base GGUF (F16, canonical
scale-1 factors, `pissa.meta` with rank / SVD parameters / the base's byte
size). The shipped one, `mm3-lm-q8_0.pissa-r128.safetensors` (0.7 GB), is in
the model registry (`mm3-lm-q8_0-pissa-r128`) and the MM3 packs.

The trainer's PiSSA init reads it when rank, oversample, iterations, layer
range and base size match (else SVD cache, else the SVD, and then it writes
the file). Checkpoints then export in the delta form: `lora_A = A - A0`,
`lora_B = s(B - B0)` at rank r, `hot_step.param_method = 4`, a
`hot_step.pissa.meta` tensor and `hot_step_pissa_residual` in
adapter_config.json. `minimax/mm3-lm-adapter.h` rebuilds the same rank-2r
pair the older exports carried from the two files, so the runtime graph,
merge mode and group scales are untouched. The delta file is not a plain
LoRA (an external PEFT loader would apply (B-B0)(A-A0)); the AS1.5 LM loader
refuses the marker. `engine/tools/pissa_cache_to_residual.py` converts an
SVD cache into the file; it made the shipped one. `mm3-lm-train
--pissa-standalone` (CLI only, 2026-09-24) keeps the residual out of the run:
nothing is written beside the base, and every checkpoint is an ordinary rank-2r
PEFT LoRA on the original base for loaders that merge plain PEFT files.

The shipped MM3 recipe since 2026-09-06 stacks four cost levers on top of
HOT-PiZZA, each first tied against the default in a blind set and then heard
together: `--prefix-frames 2048` (was 4096), `--attn flash` (composes with the
prefix since 7070238e), `--optimizer adamw --lr 8e-5` (was Prodigy) and
`--pissa-frozen-f16`. On albumA that is 4.7 s/step, 39 min for 500
steps and a 25.9 GB peak against 5.9 s, 49 min and 30.4 GB.

Since 2026-09-07 three more levers ride on top, each tied blind on its own
and then as a combination: `--prefix-frames 1024` (was 2048), `--max-frames
500` (was 750) and `--prefix-chunk 1024` (was 256; changes nothing but the
prefill speed). Together 3.1 s/step and 26 min per 500 steps at a 26.4 GB
peak. Raising the learning rate to shorten the run was tried and rejected:
every 2x-LR arm scored a few points under, and one such run planned a song
with no vocals. Turning the acoustic loss off was the lowest-scored arm.

The Training Studio offers three recipes (Recipe row on the MM3 card;
`preset` on the API). Since 2026-09-11 all three train **every track as one
whole sequence** (`--max-frames 9000`, the engine's 360 s ceiling, no history
prefix, flash attention) and differ only in depth: **Fast** (300 steps),
**Balanced** (default, 600 steps) and **Thorough** (900 steps), all at
`--lr 8e-5`. Tracks longer than the window are left out of the run by default
(`longTracks: "exclude"`, the engine's `--drop-over-frames`; the log names
them) and the route refuses a request that exclusion would gut; `"crop"`
trains them in window-sized crops instead. A third value, `"excise"`, is meant
to cut one repeated section out of each over-length track (`ace-train
mm3-retarget`) and train the result whole, but the server half is unwired:
`retargetDataset()` in `mm3Retarget.ts` has no caller and no UI offers it, so
today `"excise"` trains like `"exclude"` without the gutting check. A
`maxFrames` outside 64 to 9000 is a 400, not a silent clamp (cef52ded). The
engine clamps its sequence
buffers to the album's longest track, so an album of three-minute songs
reserves memory for three-minute songs, not for the ceiling.

Why: the earlier 30-second-crop recipes (crop 750, 82 s of history) produced
adapters that lost the song's shape — one sung passage surrounded by minutes
of looped instrumental, and no ending. On the 2026-09-10 overnight those
adapters ended 12 of 72 renders across six albums while the base model ended
36 of 36 on the same prompts, regardless of album length or genre.
Whole-song training on the same albums ended about half of its renders and
restored most of the vocal share. Cost on an RTX 5090 at rank 128: 6 to 12 s
per step and a 29 GB peak (measured: 25.4 GB at 2000 frames, 26.8 at 4000,
29.1 at 9000, linear in the sequence length; these three anchors are the
server's flash VRAM fit, `MM3_VRAM_MODEL.flash` in `mm3Train.ts`), so it is a
32 GB recipe as shipped; rank 64 or the acoustic loss off would be the levers for a 24 GB
card and neither has been heard under this regime.

From 2026-09-07 to 2026-09-09 the default recipe also ran **prior preservation
against base-model endings**: every 3rd step scored a 20 s excerpt of a base-model
plan (its opening or its natural ending) against the frozen base, from a corpus
built by `tools/mm3-reg-corpus` and shipped as bundled data
(`server/src/data/mm3-reg-corpus/base-endings-k500`). Without it, adapters on
that recipe ran to the 300 s ceiling in 6 of 6 renders; with it 3 of 6 ended
naturally, and the step count grew by half so the artist still got 500 updates.
It was a workaround for the dataset-wide caption, which broke endings; once that
caption was removed the prior only cost likeness. Prior preservation is now off
unless the request names a corpus: `regularisation: { datasetId, every, topK }`
for another dataset with codes, or `{ corpusDir, every, topK }` for a folder
holding `dataset.json`, `.mm3.txt` captions and `codes/` (the shipped corpus).

Two cost knobs for any PiSSA-family run (2026-09-06): `--pissa-cache-dir <dir>`
stores the SVD init factors once per base file / rank / oversample / iters /
layer range and uploads the identical bytes on the next run (the init is a
pure function of those, so a hit is bit-exact; the in-app trainer keeps the
cache under `<adapters>/mm3-lm-adapters/_pissa-init-cache/`); the init now
logs its wall time either way. `--pissa-frozen-f16` holds the frozen A0/B0
pair in F16, halving the ~1.3 GB it costs at r128, at the price of an init
that cancels to f16 precision rather than exactly. It is off by default at the
CLI and on in the in-app recipe since 2026-09-06, where it was heard as part of
the stacked recipe.

### Six adapter parameterizations, both LM trainers (2026-09-04/05)

`train-lm` and `mm3-lm-train` share one `LmLora` implementation
(`engine/src/train/lm-graph.h`), so `--dora`, `--rslora`, `--hira`, `--loha`,
`--pissa` (+ `--pissa-oversample`, `--pissa-iters`) and `--hra` behave
identically on both trainers, mutually exclusive under the same rules as the
DiT trainer documented below (HiRA excludes DoRA; LoHa excludes both; PiSSA
excludes DoRA/HiRA/LoHa; HRA excludes everything including rsLoRA and needs
an even `--rank`); `--rslora` and `--lora-plus-ratio` stack with any of them.
The delta each one computes is the same as the DiT table further down — the
LM's projection sites (`q/k/v/o_proj`, `gate/up/down_proj`) are the only
difference.

**The DiT trainer's blind listening test found that no parameterization beat
plain LoRA** (`docs/plans/adapter-parameterizations-roadmap.md`, local).
Nothing on the LM side has been rendered at all — every gate that ran here is
a finite-difference probe or an export/load round trip through the runtime
loader, never audio. Do not read "passed its gate" as "sounds better".

HiRA used to fail its finite-difference gate at the CLI's default epsilon
(worst relative error 3.3e-2 against a 2e-2 bar, passing only at `--fd-eps
0.05`). That was the ESTIMATOR, not the gradient, and it is fixed: `--fd-eps`
is now a floor on the step rather than the step itself, and any probe whose
loss change `2*eps*||g||` would land under 256 F32 ULPs of the loss gets a
bigger one — the rule `train-dit`'s own T4 gate has always used. HiRA's delta
is `W (.) (s B A)`, so its factor gradients come out 16x to 68x smaller than a
plain LoRA's on the same sites, and the forward's fixed quantisation floor
(~3.5e-7 on the loss, measured) showed up in its relative error 50x magnified.
Sweeping eps from 0.005 to 2.0 on the failing probe moved it 0.059 -> 0.0001:
an error that shrinks as the step grows is arithmetic, not a wrong backward,
and the checkpointed-vs-naive route comparison passed at 4.2e-7 throughout.
All seven arms now pass `--fd-check 8 --f32-layers 2` at the default epsilon
(worst 0.0070, PiSSA); LoRA, rsLoRA, DoRA and PiSSA are unchanged to the last
printed digit, because the floor only raises a step that was under it.

### HRA scales the graph with `--rank`, and rank 64 does not fit

A full-depth `--hra` run at the default rank (64) used to die before step 1 on
a ggml `cgraph->n_nodes < cgraph->size` assert. Two separate things were wrong
and both are fixed (2026-09-05).

The crash was graph SIZING. `lm_hra_reflect` emits 13 ggml nodes per
reflection, so the node count is proportional to `--rank`: 5,824 forward nodes
per layer at rank 64, where a plain LoRA emits about 35. Every builder on the
LM training path allocated its `cgraph` from a constant chosen against a LoRA,
and the 8,192-node segment probe overran first. The budgets now come from the
bank (`lm_site_extra_nodes` and friends in `engine/src/train/lm-graph.h`), and
so does the host arena behind them. Measured segment size: 221 nodes for a
plain rank-64 LoRA, 2,132 / 8,180 / 16,244 for HRA at rank 8 / 32 / 64 — a
slope of exactly 2.769 backward-graph nodes per forward reflection node, which
is what the 5x budget multiplier is sized against. With no rank-proportional
arm in play every graph is emitted at exactly the size it always had; the
`train-lm --self-test` T22 anchor (naive trunk 396 nodes, P7 segment 221, both
hashes) is unmoved, and a plain-LoRA 10-step MM3 run is loss-identical to six
decimals before and after.

What was left after that was arithmetic, not a bug: rank 64 genuinely does not
fit. Each reflection retains its running `x`, its broadcast `v` and its
correction, all `[in, S]` F32 and all backward inputs, so the activation cost
is linear in rank AND in the crop with no allocator trick to remove it. On a
32 GB card at crop 750 (`mm3-lm-q8_0` + `mm3-depth-f16`, a 12-track album,
36 layers, checkpointed, AdamW, acoustic loss on, seed 42, mean over 10 steps):

| `--hra --rank` | trainable params | worst segment | peak VRAM | ms/step |
|---|---|---|---|---|
| 8  | 10.6M | 2,132 nodes  | 18,062 MiB | ~6,800 (6.6-6.8 s) |
| 32 | 42.5M | 8,180 nodes  | ~30,700 MiB (30,551-30,730) | ~27,000 (25.7-28.9 s) |
| 64 | 84.9M | 16,244 nodes | refused: needs >= 33.6 GiB | — |
| plain LoRA, rank 64 (reference) | 174.6M | 221 nodes | 18,626 MiB | ~2,400 (2.3-2.4 s) |

Ranges are three runs of the same command; peak VRAM is device-wide, so it
carries a little noise. Rank 32 fits with under 2 GB to spare and costs 11x a
plain LoRA's step time for a quarter of its parameter count; rank
64 asks the allocator for a 29.4 GiB compute buffer on its own. `mm3-lm-train`
now estimates that term before allocating and refuses with a message naming the
rank, the crop and the arithmetic, rather than letting `cudaMalloc` fail and
`ggml_gallocr` segfault. The estimate deliberately leaves out the baseline
compute buffer, so it under-reads and only ever refuses what could not have
started. **`train-lm` (AS1.5) got the graph sizing but not the refusal** — its
VRAM auto-fit does not know about HRA either, and no high-rank AS1.5 HRA run
has been measured.

MM3's binary resume format is now v4 and refuses a checkpoint whose
parameterization or rsLoRA bit disagrees with the run being continued
(`mm3-lm-resume.h`). Format v5 lets `mm3-lm-train --pissa` resume: the frozen
A0/B0 are rebuilt at init, and the state carries an FNV-1a hash of them (as F32,
so `--pissa-frozen-f16` does not change it) that the load compares. A v4 PiSSA
state has no hash and is trusted. `train-dit --pissa` still refuses `--resume`. `hra`+`rslora` and `--prefix-n` combined with prior preservation
(`--reg-*`) are both refused with a 400 from the training routes rather than
failing after the model loads.

### Artist token and KV prefix now reach MM3 generation (2026-09-05)

The soft-prompt machinery documented below for AS1.5 (`--artist-token`,
`--prefix-n`) was already trainable on `mm3-lm-train`; until this date nothing
at MM3 generation time applied what it trained. `mm3-lm-adapter.h` now reads
the same `hot_step.artist_token.{vec,meta}` / `hot_step.prefix.L<l>.{k,v}` /
`hot_step.prefix.meta` tensors out of `adapter_model.safetensors` that the
AS1.5 runtime reads (site 2 = mm3_lm in the `.meta` tensor — **one file
format for both LM trainers**), installs the token at the prompt's
placeholder span, and seeds the prefix into the KV cache before the prefill.
Wired into both the runtime and merge generation paths; the merge path is
compile-checked only, not executed, because exercising it needs a resident MM3
LM inside a running `ace-server`. The prefix is seeded into every batch row,
including the CFG unconditional one — MM3 has one KV cache and a mask with no
batch axis, so seeding the conditional rows only (what the AS1.5 runtime does)
is not expressible there.

Debug surface: `POST /mm3/lm-plan` accepts `"lm_soft_off": true` to run an
adapter's LoRA half alone — same weights, same seed, soft prompt off — which
is the A/B this feature is provable by. Proven on logits and generated codes
(the off arm reproduces the LoRA-only run byte for byte, and clearing the
adapter afterward reproduces the original no-adapter run byte for byte).
**Nothing trained with a token or a prefix has been rendered to audio.**

### The MSVC 2 GiB `stat()` trap

MSVC's `stat()` is 32-bit on file size: a model, adapter, or codes file at or
past 2 GiB reads back as "missing" instead of erroring, because the size field
wraps. Every such check in `engine/src/minimax/` now routes through
`hs_stat`/`HS_STAT_T` (`engine/src/hot-step-fsutf8.h`) instead of the bare
libc call — `mm3-adapter.h`, `mm3-lm-adapter.h`, and `mm3-job.h` were the
three call sites. Grep for a bare `stat(` in `engine/src/minimax/` before
adding a new one; the pattern to follow is commit `ae64b19c`.

## What it is

End-to-end adapter training **entirely in C++/GGML — no Python anywhere**:

```
Dataset creation → Tensor preprocessing → LM LoRA training (0.6B/1.7B/4B)
                                        → DiT LoRA/LoKR training
                                        → Pure-LM audition (A/B, no DiT influence)
```

That is the ACE-Step pipeline, which most of the sections below describe. The
MM3, YuE2 and SA3 trainers above are further `ace-train` subcommands with their
own cards in the same Training Studio.

Side-Step (`D:\Ace-Step-Latest\Side-Step`, local) is the Python reference implementation this system reaches parity with; its dev sanctioned porting. Dataset/sidecar formats are byte-compatible with Side-Step; tensor caches deliberately are not (safetensors, not pickle).

## Component map

| Piece | Engine | Server | UI |
|---|---|---|---|
| Dataset creation | — (uses ace-server `/understand` optionally, legacy) | `routes/training.ts` + `services/training/{sidecarIO,datasetScan,labelStore,essentiaClient,enhanceService,captionPrompt,datasetBuilder,labelingQueue}.ts` | `training-studio/{DatasetList,NewDatasetWizard,SampleGrid,SampleDrawer,LabelPanel,EnhancePanel,BuildPanel}.tsx` |
| Preprocess | `ace-train preprocess` (`engine/src/train/preprocess-*.h`, `st-write.h`) | `preprocessRunner.ts`, `preprocessStatus.ts`, `aceTrain.ts` | `Preprocess{Panel,OptionsForm,VariantCard}.tsx` |
| LM training | `ace-train train-lm` (`lm-*.h`: graph/optim/ckpt/bf16/vram/data/extract/export/selftest/train-run) | `trainLmRunner.ts`, `trainLmStatus.ts` | `TrainPanel.tsx`, `TrainLmForm.tsx`, `TrainingChart.tsx`, `TrainingRunStats.tsx` |
| DiT training | `ace-train train-dit` (`dit-*.h`: same family) | `trainDitRunner.ts`, `trainDitStatus.ts` | `TrainDitForm.tsx` (+ shared chart/stats) |
| Audition | `POST /codes-decode` on **ace-server** (`hot-step-server.cpp`, ~174 lines) | `audition{Service,Runner,Store}.ts` | `AuditionCard.tsx`, `AuditionPlayer.tsx`, `LmAdapterPicker.tsx` |
| Engine lifecycle | — | `services/aceEngineProcess.ts` (stop/restart with epoch-guarded respawn cancellation) | engine-paused banners |

### Training review and audition drafts

The review endpoint returns Node-computed rung scores, eligibility, blind labels, the earliest-step best rung and a revision. Finish and manual selection send an explicit run and checkpoint with captured dataset and review revisions; stale or incomplete selections are refused. Node decides whether Further NAR is skipped, rejected or chained, then rechecks the captured pick at job admission before starting a continuation. Selection and cleanup share a dataset lock. Finished runs cannot be linked or cleaned up twice.

Audition-to-Create creates a user-owned server draft from a stored preview. Create retrieves it with authentication and mirrors the recipe only when the user presses Apply. Creating the draft does not generate audio or change the open Create form.

### Training recipes

`GET /api/training/ops/recipes/` lists the ACE LM/DiT, MM3 LM, YuE2 NAR/AR and YuE2 joint families. `GET /api/training/ops/recipes/:family` returns a family's built-in defaults, stored partial defaults, resolved values, execution projection, provenance and engine-deferred fields. A `worker` query selects that worker's recipe; a `preset` query selects a supported preset. `POST /api/training/ops/recipes/resolve` accepts the training operation snapshot with `family`, optional `preset` and form `overrides`, and returns the captured worker and resolved request. The form keeps unsaved edits locally and sends overrides for resolution immediately before the existing trainer start route.

YuE2 chains use `POST /api/training/ops/preparation/`. The command carries the dataset row revision, a signature of source file names, sizes and modification times, the worker, selected stages and versioned recipe forms. Node resolves the forms once, pins the worker URL and saves the operation under the training directory's `yue2-preparation/` folder. It checks both revisions before acceptance and at each stage boundary. Each skip uses the worker's current cache and run status. Preparation-only rejects training stages.

`GET /api/training/ops/preparation/context/:datasetId` supplies the two revisions. `GET /api/training/ops/preparation/?datasetId=…`, `GET /:id` and `GET /:id/job` expose saved stages and job IDs after reconnect. `POST /:id/pause` holds after the current stage. `POST /:id/cancel` requests job cancellation; poll `GET /:id` for the acknowledged terminal state. `POST /:id/retry` checks the recorded job before retrying a failed or interrupted stage. A restart marks active work interrupted and never submits another stage on its own. If a saved record cannot be read, new starts fail until that record is reconciled.

For ACE, stored defaults are a separate partial layer used by the batch pipeline. The interactive request projects the resolved form once; it must not spread stored pipeline values underneath fields omitted by the interactive projection. Explicit zero and false overrides remain values, not missing fields. Engine-selected model, device and other deferred values are reported as deferred rather than guessed by the browser.

Frozen contracts (types, routes, JSONL event schemas, CLI) are duplicated verbatim between `server/src/services/training/types.ts` and `ui/src/services/trainingApi.ts` — **kept in sync by hand, deliberately**. Newer shared shapes are imported by the UI instead of copied: `server/src/contracts/training.ts` (target ids, the SA3 asset block, recipe and preview shapes) and `server/src/contracts/trainingRecipes.ts` (the recipe seed values).

Job model: two promise-chain queues in `labelingQueue.ts` (`laneFor`), a GPU lane for trainers, preprocess and other engine work and a network/CPU lane for labeling, each serial and both running at once (one global chain until 2026-07-30). SSE streams with replayable capped buffers, `_meta.json` persistence, `TrainingMetricEvent` for training numbers. GPU jobs stop the ace-server child first (`stopEngine`, default on) and restart it in a `finally`.

### Cover-laundered codes (the dense-mix gate, 2026-08-31)

`ace-train mm3-launder` is an alternative codes export for dense-mix artists:
real audio -> rec7 state encoder (`mm3-rec7-*.gguf`) -> the released depth
chain -> the flow DiT's own stitched window latents -> champion RVQ encode.
The champion under-encodes buried vocals on real dense mixes; rendering the
track back through the model first puts the vocal where the codes can carry
it. Ear-validated A/B (a dense-mix album, 2026-08-31): the laundered-codes
adapter won on every axis at identical training curves.

Fully native (no python at runtime; the rec7 GGUF carries the LM's two 16k-row
semantic table slices so the 8B never runs — its weights load only for the
depth chain's token embedding). ~1.4x realtime per track, 359 s cap (the
pipeline's 9000-frame ceiling). Gated everywhere: the codes card has a
"Cover-launder the codes" toggle writing to `mm3-codes-laundered/` (a sibling cache,
never a replacement), and the train form only offers "Train on cover-laundered
codes" when that cache exists. Both off = byte-identical to the pipeline
before the gate existed — the launder is a new subcommand plus additive
options; no shared code path moved.

Parity gates: `ace-train rec7-selftest` (states vs the python reference,
cosine 0.999999 measured) and the phase-0 latents-direct measurement (stitched
flow latents vs DAV re-encode: cosine 0.9936 on owned spans; raw window tails
are 0.57 — the stitch uses `mm3_window_crop` at latent granularity for exactly
that reason). Convert encoders with
`engine/tools/convert-rvq-encoder.py --head <head.pt> --m3 <checkpoint dir>` (adds
the state head + LM slices to the standard V4 GGUF).

## Key design facts (the ones that bite)

**Data & labeling**
- Sidecars (`<stem>.txt`) live **next to the audio**; `dataset.json` in the source folder — Side-Step-compatible on purpose. Studio-private state (audio_codes, provenance, raw analyzer results) lives in `<data root>/library/training/datasets/<slug>/labels/`.
- Labeling runs its lanes side by side from the first file: the SheetSage2 key pass, Pass A (BPM/key), Genius lookups (started for every file at job start, paced by their limiter) and captions (MOSS as one batch from job start). Tempo defaults to beat_this (`ace-midi --track`, on the GPU lane; `beatThisClient.ts`) and key to SheetSage2: `transcribeLabelSheets` runs one `ace-train yue2-sheet` over the files with no `.abc` sidecar (engine kept up) and saves each score beside its audio the moment that file is done; Pass A waits only for its own file's score and reads `K:`. Essentia runs only for a file whose tempo or key the other two could not give; with beat_this as the tempo source it also rewrites the sidecar's `Q:` to the measured tempo at the score's own beat level (a 157 BPM measurement of a score written at 80 becomes 79), keeping a `.bak` of an older score. `tempoSource`/`keySource: 'essentia'` opt out; Essentia results are cached by size+mtime. Then Genius lyrics ∥ Genius lyrics (relaxed collab matching — a two-artist collaboration credit broke exact primary-artist equality) → Gemini captions **with the audio attached** (96 kbps MP3 inline; the local caption is *omitted* from the prompt to avoid anchoring), or local MOSS captions, which also hear the audio (other chat providers rewrite text only). The same caption pass writes the YuE2 sentence to `<stem>.yue2.txt` and, under MOSS, the MM3 Structured Caption to `<stem>.mm3.txt` (`CaptionOptions.wantYue2` / `wantMm3`). `ace-understand` is out of the default flow (weak captions, hallucinated lyrics, J-pop prior); still reachable via API `useUnderstand:true`. Music Flamingo (`caption.provider: 'flamingo'`) is the second local captioner and runs through the same `ace-caption` batch as MOSS (`mossCaption.ts`, `LocalCaptioner`). `LabelOptions.localLyrics` adds Pass B step 2b (phase `lyrics`) right after Genius: a track with no lyrics from Genius or its sidecar and not marked instrumental takes its Music Flamingo transcription (`takeLocalLyrics`) from a batch started at job start (`prepareLocalLyrics`, or the caption batch itself when Flamingo is the captioner, one encode for both), with `sources.lyrics = 'flamingo'`. Local captioners get a per-track hint, the file's embedded genre tag (`genreHint`, the `ace-caption --src-list` third column), and write the YuE2 caption themselves (`--mode yue2`: six short fields assembled into the sentence by `assembleYue2FromFields`, BPM from local analysis); a chat provider is used for it only when `yue2Provider` names one. The label route defaults `useGenius` off and `localLyrics` on when Flamingo is installed; under `overwrite_lyrics`/`overwrite_all` step 2b re-transcribes existing lyrics. `takeLocalLyrics` decodes with no repetition penalty and, when `looksLooped` fires, redoes the track once at 1.05 (17-track test: median/mean WER 20.0%/22.5% against 19.6%/29.4% for one pass; q8_0 vs f16, temperature 0.4/0.7 and penalty 1.02 did not help). Every lyrics write (Genius, Flamingo, Understand) goes through `normaliseLyricTags` (`lyricTags.ts`, replacing the Genius-only `sanitizeHeaders`); the `normalise-lyrics` job (`POST /datasets/:id/enhance/normalise-lyrics`, `{ retag }`) applies it to existing sidecars, and with `retag` gives lyrics with no sung-section tags the section starts of a Flamingo transcript via `transferTags` (word-level LCS, skipped under 40% word overlap). On a 17-track album test Flamingo scored median WER 24% against Whisper large-v3-turbo's 59% and MOSS's 59%, with section tags on every track.
- Optional `essentiaTempo.minTempo/maxTempo` values become `rhythm` keys in the extractor's YAML profile; no bounds means no profile and Essentia's defaults. The label cache also keys on these bounds. An embedded BPM tag replaces Essentia's estimate only when it is a half or double tempo match (within 2 BPM); without such a match, the Essentia value is kept. The Review grid's ½ and 2× buttons save a manual BPM correction before dataset build.
- Dataset language is **declared, not detected** (`default_language`, forced-write on every label pass).
- Lyrics *absence* never writes `is_instrumental` — only lyrics presence writes `false`.

**FSQ (critical correctness area)**
- `engine/src/fsq-quant.h` is the single source of truth. The reference path is **ResidualFSQ `preserve_symmetry`**: soft clamp `c=1+1/(L−1)` → `tanh(z/c)·c` → hard clamp → `floor((L−1)(w+1)/2+0.5)`. NOT `FSQ.bound` (that branch never executes in vqp). Encode verified 13046/13046 vs the checkpoint's own tokenizer.
- `verify-hooks.ps1` Hook 5 guards both FSQ files (`fsq-detok.h` must include `fsq-quant.h`, and neither it nor `fsq-tok.h` may carry upstream's own quantizer copy) — upstream syncs must not revert this.
- Side-Step's stored `lm_codes.jsonl` are a **bf16 ceiling** (~85% identity); engine-F32 vs fp32-reference is the honest gate (99.0%).

**LM trainer**
- Hand-rolled loop (`lm-optim.h`): ggml-opt can't grad-clip and its grad accessor crashes with dynamic graphs. Persistent grad accumulators indexed by forward-node order, **in-graph** global-norm clip at 1.0, AdamW nodes, Side-Step-exact LR schedule (`lr(0)=0`, 5% warmup, cosine→0.1).
- ggml gap catalogue: `out_prod` F32-only upstream (the central constraint; the HOT-ggml fork's `bf16-out-prod` change, Hook 7, lets ggml-cuda's `OUT_PROD` take a BF16 `src0`), no upstream backward for flash-attn (closed 2026-09-01 by HOT-Step's own `GGML_OP_FLASH_ATTN_TRAIN`/`_BACK` ops — CPU + CUDA, GQA, additive masks, bitwise-deterministic; see `docs/dev/ggml-fork.md`) / `SET_ROWS` (KV cache) / fused swiglu; CE labels are dense one-hot; `CONCAT` has no backward (use `ACC`); rms_norm backward is wrong on **non-contiguous (permuted) inputs** — norm before permute.
- 4B fits ~12 GB via per-layer checkpoint segments + a transient per-layer F32 window + chunked-CE head (BF16 embedding). Naive path preserved byte-identical for 0.6B/1.7B.
- `--weights bf16` (experimental): backward surgery rewrites `out_prod(W_f32,gT)` → `mul_mat(cont(transpose(W_bf16)),g)`. **1.256× at 4B** measured; gradient cos ~0.9992 vs F32 (BF16 rounding compounds sub-linearly with depth; partly ggml-cuda's dst-BF16 rounding, only fixable by a vendor patch). When this was measured `engine/ggml` was stock upstream; it is now the HOT-ggml fork, whose changes are listed in [ggml-fork.md](ggml-fork.md).
- `--batch` intentionally **not built** — measured amortisable overhead below the bar where it matters (4B: 9.3% < 10%); flag exits 2 citing the numbers.
- CLI defaults (Side-Step parity, from real runs): target loss **4.0** (not 0.4!), GA 2, epochs 75, milestone step 1.0, lr 1e-4, LoRA r16/α32, AdamW. The app's own defaults (the form seed `ACE_LM_FORM_DEFAULTS` in `server/src/contracts/trainingRecipes.ts`, and the `train-lm` route fallbacks the batch pipeline's empty option bag lands on) keep target 4.0, GA 2 and lr 1e-4 but run up to 150 epochs with milestones off; the rest of the app recipe is in the next block.
- **Staged target chain** (2026-08-29, route default): `targetLoss` is the final target, and the route prepends every rung of the 2.0 → 1.5 ladder that sits above it, one `ace-train` leg per rung chained with `--init-adapter`, so each rung restarts the optimizer state and LR schedule. At the default target 4.0 no rung applies and the run is one leg. `targetLossStages` pins the list; an explicit `initAdapter` (a resume) collapses it to the final target. The form sends `initAdapter: 'latest'` by default: the newest run of that adapter name with weights (never a `-calibrated` bake), or a fresh run when there is none.
- Self-test: `ace-train train-lm --self-test` — T1–T13 (+T14–T17 bf16, report-not-gate; T18–T21 cover `--attn flash`, T22 anchors the graph sizes). T5/T4 finite differences run **TF32-off in a child process** (cuBLAS TF32 noise defeats central differences; gradients were always right). Known pre-existing 4B rough edges: T11e/T12 slightly over bars (BF16-dtype-resolution class).
- **`--attn exact|flash` (2026-09-02).** Same fused flash-attention forward/backward as the DiT (the HOT-ggml `flash-attn-train` change), with the causal mask as one triangular −INF fill — the kernel skips all-−INF tiles, so causal training gets roughly half its attention compute free. **The CLI default is `exact`.** The Training Studio and the route have defaulted to `flash` since 2026-09-03, but the route turns it back to `exact` whenever the soft-prompt KV prefix is on, and that prefix has been on by default since 2026-09-04, so a default run trains exact attention (the engine has accepted flash with a prefix since 7070238e; see the next block). What it buys, measured on this box (4B, low-VRAM path, `albumF` substituted for the absent album I fixture, paired/interleaved): against the shipped arm (`exact --attn-head-block 8`) flash is **5.5% faster per micro-step** and **3.8% lower peak VRAM**; against an equal-shape reference (`exact --attn-head-block 0`) it is only **1.2% faster** — almost all of the win is the head-block copies going away, not the fused kernel, which is the opposite split from the DiT (there the fused kernel is the win and equal-shape flash is 8.5% *slower*). Attention-only bound from `fattn-train-test --bench-lm`: 0.74×/0.79×/0.80× of the blocked arm at S=1024/2113/3500. On the naive path (0.6B/1.7B, no per-layer checkpointing) flash's linear-in-S attention state roughly **doubles the auto-fit `maxLen`** on 0.6B (3136→6208 tokens) and clears a 600 s dataset's `skippedLong` count from 2 to 0; on 1.7B the same dataset only gets a 1.27× lift and still skips one long track. 50-epoch same-seed drift (flash vs exact) is the same class as the already-shipped `--weights bf16` deviation — smaller on two of three measures, ~20% larger on final CE from a single seed, no error bar yet. Not yet ear-validated. Full numbers, VRAM-model refit, and the pre-existing bugs it surfaced (an MM3 export assert, an exact-mode auto-fit OOM crash) are in `.claude/skills/flash-attn-training/SKILL.md` §3/§7.

**LM trainer — caption dropout and prior preservation** (2026-09-02, both DEFAULT OFF)

The 2026-09-02 attribute study (`docs/plans/lm-attr-probe/`) found that everything an ACE
planner-LM adapter measurably does is done by CE ~2.0, and that training deeper past it mostly
buys looping plans. These are the two levers the MM3 LM trainer already had and this one did not.
Both are opt-in at the CLI; with both off the trainer emits the same graphs, consumes the same RNG and
produces the same loss trajectory it did before they existed (verified against a purpose-built
pre-change binary: same 4B run, `--order fixed --epochs 1 --limit 4`, per-step `loss`/`gradNorm`/
`clipScale` identical to the last printed digit). **The Training Studio and the batch pipeline turned
both ON on 2026-09-03** (`captionDropout 0.3`, `regEvery 3`, cached teacher): in the 3-artist
render-scored experiment (`docs/plans/lm-attr-probe/RESULTS.md` §6–7) that combination measured best on
loudness and timbre at the control's loop rate; caption dropout alone hurt, and the live teacher was
worse than the cached flat-tail prior. The same day's other default changes for `train-lm` were LoRA
rank 128 / alpha 256, **Prodigy** as the optimizer (`--optimizer prodigy`, ported into train-lm from
mm3-lm-train: the step size is estimated online, `--lr` becomes a schedule multiplier forced to 1.0,
`--prodigy-d0` seeds the estimate at 1e-6), **`--weights bf16`** (the DiT's bf16-mirror counterpart,
1.256× at 4B) and **`--attn flash`** on.

**Since 2026-09-04 the default is the joint soft-prompt recipe** (arm 05 of that day's listening test,
best by ear and on the eval): LoRA rank 16 / alpha 32 with AdamW at lr 1e-4, an artist token named
after the adapter (k 32, token LR 5e-3) plus an 8-column KV prefix, caption dropout 0.3, prior
preservation **off** (`regEvery 0`: arm 05 trained without it, and the run that combined it with the
soft prompt was the one that memorised), target loss 4.0, `--weights bf16`. With the soft prompt on,
r128 + Prodigy overfit in a handful of epochs. `attnBackend` still defaults to `flash`,
but the prefix makes the route run it as `exact` (above). The form seed is `ACE_LM_FORM_DEFAULTS` and
the route fallbacks match it, so a batch-pipeline run trains the same recipe. A default caption dropout
on a variant with no trigger word is dropped with a warning; an explicit one is a 400, as is an explicit
`regEvery` when no other 600 s corpus exists. Neither lever is ear-validated yet. Explicit values still
win, and resumes adopt the source run's identity. The only flags-off output change is an additive
`"captionDropout":0` on the `start` event and a `caption_dropout` key in `lm_train_log.json`.

- **`--caption-dropout <p>`** (0..1, default 0). Each style micro-step draws from a per-epoch
  seed-derived stream and, with probability p, trains the row with its caption replaced by **the
  trigger word alone**. The point is to make the trigger carry the style by itself, which is the
  only way a bare-trigger prompt works at inference. The variant is built by the same
  `lm_build_sequence()` the real sample uses, so the whole prompt — **including the CoT YAML,
  whose `caption:` line IS the caption** — is what the engine’s own prompt code produces for a
  bare trigger, and `n_masked`/`s_tr` differ from the full-caption row accordingly. That is the
  one place this deviates from MM3, whose prompt has no CoT. Requires a trigger (`--trigger`, else
  the variant’s `custom_tag`); `p > 0` with none resolvable is **exit 2**, deliberately, rather
  than dropping to an empty caption that is far out of distribution for this prompt format. Do not
  set 1.0 — the descriptor path would go untrained and the adapter could summon the album but
  never be steered. Per-epoch usage count lands in stderr and a `caption_dropout` JSONL event.
  Treated as NON-identity on resume (like `--bwd`): a `--init-adapter` run neither adopts nor
  refuses on it.

- **Prior preservation** (`--reg-codes`, `--reg-songs` 24, `--reg-every` 0=off, `--reg-topk` 64,
  `--reg-prior-dir` `<out>/prior`). Every `reg_every`th micro-step trains on **another artist’s**
  song against the frozen base’s own cached top-K next-token distribution instead of that song’s
  ground-truth codes, so the adapter is punished for changing its mind about material that has
  nothing to do with this artist. `--reg-every 3` is bghira’s ratio (one prior step per two style
  steps). The distributions are captured **once, before the first optimizer step**, while the
  LoRA/LoKR delta is exactly zero and a forward pass therefore IS the base, and cached to
  `<reg-prior-dir>/<artist>_<path hash>_<row id>.<lm stem>.k<K>.prior` (`train/lm-prior.h`, the
  MM3 layout with an `ACEPRIOR` magic). A `--init-adapter` resume **cannot** regenerate them and
  refuses on a missing or stale cache. Reg rows always use their own full captions — never caption
  dropout.
  - **A reg step is INSERTED into the epoch, not swapped in for a song.** An ACE epoch is one
    teacher-forced pass over the album and both target-loss and best-epoch selection read its mean,
    so an epoch that quietly saw two-thirds of the songs would corrupt the number the run is
    steered by. (MM3 is step-driven and has no such contract, which is why it can simply replace.)
    The cost is visible: `--reg-every 3` runs 1.5× the micro-steps per epoch. Reg steps count
    toward the step counter and are excluded from the reported style loss; their own CE is the
    `reg_epoch` event’s `regCe`.
  - **Low-VRAM/checkpointed path only.** The capture reuses the chunked head’s top-K readback
    (`lm_ckpt_capture_topk`), which the naive path does not have. `--reg-every` on the naive path
    exits 2 with that message; `--low-vram off` is refused at parse time. In practice 4B works out
    of the box and 0.6B/1.7B need `--low-vram on`.
  - **Label rows must sum to 1, and how they are brought there matters.** ggml’s cross-entropy
    backward is hard-wired to `(softmax(z) − labels)·d/nr`, which is the true gradient of the
    forward loss only when the label row sums to 1. Hand it a row summing to `S` and the gradient
    it computes is the true one **plus** `(1−S)·softmax_j` — the gradient of `(1−S)·logsumexp(z)`,
    a pull on the raw logit scale that changes no probability at all and is unbounded below. At
    MM3’s `S ≈ 0.99` that is a 1% perturbation, which is why the code survived the port; at ACE’s
    `S ≈ 0.2` it would have carried ~80% of the gradient. The ACE trainer brings each row to 1 by
    adding a **uniform tail**: the captured top-K keep their true probabilities and the missing
    mass `1−S` is spread flat over the 65,535 audio-code tokens (`LmSample::soft_tail`,
    `lm_soft_labels_write`). It deliberately does **not** rescale the top-K to sum to 1: at 18–24%
    coverage that would hold the adapter to a target 4–5× sharper than the base, and sharpening is
    the loop attractor the term exists to resist. The `.prior` cache and the reported coverage stay
    **raw**, and MM3’s shared capture code (`lm_ckpt_capture_topk`) is untouched.
  - **READ THE COVERAGE NUMBER — ACE is not MM3 here.** MM3 scores 16,385 classes and its top-64
    routinely covers >99% of the base’s probability mass. ACE scores **217,204** and its base sits
    at CE ~9 on another artist’s codes, i.e. thousands of effective classes: measured on one
    artist's run regularised on a second artist's songs, top-64 covers **18.5%** and top-256 only **24.0%**. With the uniform tail
    the target is then mostly floor: it says "stay flat" much more than "stay the base". Whether
    that still buys anything is **unmeasured** past a smoke test. The trainer logs a `warn` below
    50%. `--reg-topk` (max 256) sharpens the picture a little; the real fix is `--reg-teacher live`
    below.

- **`--reg-teacher cached|live`** (2026-09-03, default `cached` = everything above, byte-identical).
  `live` replaces the cached top-K with the frozen base's **own distribution, recomputed on every
  reg micro-step**: no cache, no K, no coverage warning, 100% coverage by construction.
  - **How.** A live reg step is two passes. First a forward-only pass of the same row with the
    adapter delta switched off (`LmLayerOpts::adapter_off` makes `lm_linear` skip both the LoRA and
    the LoKr term) that stops after P3 and stashes the final hidden states into an `[H, s_max]`
    buffer (20.8 MB at 4B). Then the ordinary training micro-step, whose chunked head re-derives
    that chunk's teacher logits from the stash with one extra head GEMM, softmaxes them **on the
    GPU** and writes the dense `[V, chunk]` block straight into the label buffer. Nothing crosses
    PCIe: reading the teacher's probabilities back to the host would be ~2.6 GB per reg step and a
    host `exp()` over 229 M values.
  - **The mask is PER POSITION, and that is not a detail.** "Softmax over the audio-code range
    only" is right at a code position and wrong everywhere else: with `--loss-on-cot` (the default)
    a reg row's supervised span starts at `<think>`, so ~13% of its positions are CoT YAML text
    (measured on that regularisation corpus at 4B). A code-only target there renormalises 65,535 logits the base has
    driven to near zero — noise — and demands total code probability 1.0 at a position where the
    base emits text, i.e. it trains the model to emit codes instead of a plan. So the range mask is
    applied only where the row's own ground-truth token IS an audio code; elsewhere the teacher is
    the base's full-width distribution. Before this fix the step-0 check read reg CE 11.93 against a
    teacher entropy of 9.46 (a 2.47-nat gap that is exactly this bug); after it, 8.6515 vs 8.6515.
  - **The step-0 check.** At init the delta is exactly zero, so the student IS the base and the reg
    CE must equal what the base's own logits say. The run computes both (the second on the host in
    double precision) and **refuses to start** if they differ by more than 1e-3. Measured 1.3e-06
    at 4B. Skipped with a log line on an `--init-adapter` resume, where the premise no longer holds.
  - **`--reg-topk` and `--reg-prior-dir` are ignored** in live mode — nothing is captured and
    nothing is written — so unlike the cached teacher a **resume needs no cache** and can never hit
    the stale-cache refusal.
  - **Cost.** One extra forward per reg step: measured 1177 → 1326 ms per reg micro-step at 4B
    (+13%), epoch 4411 → 4561 ms at `--reg-every 3` (+3.4%), VRAM 12090 → 12112 MB. The VRAM model
    does **not** yet carry the 21.6 MB stash+mask, so `estMb` under-predicts by that much in live
    mode.
  - **Not yet evidence.** Whether the live teacher trains a better adapter than the cached one is
    unmeasured — the arms have not been run.

Usage (4B, low-VRAM by default):

```
ace-train train-lm --stages train,export ^
  --codes  <data root>/library\training\tensors\<dataset>\<variant>\lm_codes.jsonl ^
  --out    <run dir> --models models --lm acestep-5Hz-lm-4B-BF16.gguf ^
  --caption-dropout 0.25 ^
  --reg-every 3 --reg-codes <other artist>\lm_codes.jsonl,<another>\lm_codes.jsonl ^
  --reg-songs 24 --reg-topk 64
```

Same again with the live teacher (drop `--reg-topk`, it does nothing here):

```
  --reg-every 3 --reg-codes <other artist>\lm_codes.jsonl --reg-songs 24 --reg-teacher live
```

`ace-train train-lm --help` carries the same flags with their defaults.

**DiT trainer**
- Trainable graph is **bit-identical** to the inference forward (17 debug-tensor diff = 0.0). Unfused load via `g_dit_load_no_fuse` (mirrors `g_qwen3_load_no_fuse`; the zero-stub-adapter trick is dead).
- F32 mirror streamed from the GGUF via CPU-backend load (a GPU-copy mirror transiently costs +5–8 GB and OOMs cards the steady state fits).
- Crop-window training, one window per step. Since 2026-08-29 the app draws song-anchored structured crops (`--crop-anchor song`, start and end shares 0.2 each, the rest random; `dit_sample_crop_structured` in `dit-data.h`), and since 2026-09-03 `crop_max` defaults to 800 frames in every attention mode: long auto-fit crops lost three ear A/Bs to short ones, so the cap is a quality guard and flash mode no longer lifts it. An explicit `--crop-max` still wins both ways. **Two attention formulations** (`--attn`, 2026-09-01): `exact` is the original `mul_mat → soft_max_ext → mul_mat` graph, whose retained O(S²·Nh) softmax made full-song full-depth ≈ 68 GB and capped crops at ~1000–1250 frames at 24 GB; `flash` (the Training Studio default) routes both self- and cross-attention through HOT-Step's own fused flash-attention forward+backward (the HOT-ggml `flash-attn-train` change, TF32 tensor cores on sm_80+, f32 everywhere else), whose attention memory is linear in S — on a 32 GB card the auto-fit then picked full-song crops at full 32-layer depth until that cap (measured: crop 3846 on a ~5600-frame album, 3.08× the exact cap, 28.1 GB), and per attention site it is *faster* than the cuBLAS chain at every crop (0.94×/0.64×/0.49× at S=625/1250/3000). `flash-f32` pins the strict scalar kernels. Exact mode's graph is byte-identical to pre-2026-09 runs; the `--layers` top-K ladder still describes exact mode and is due a flash-mode re-derivation (plan-doc B1). **The flat "<16 GB card refused" rule is gone (2026-09-02).** The trainer now refuses only when the cheapest configuration the auto-fit can express — K=2 (or your pinned depth), `crop_min`, the widest checkpoint split, batch 1, at this run's adapter/mirror/attention settings — does not fit free VRAM after the reserve; the fatal quotes that floor, the configuration it is, how much of it is the frozen-weight mirror, and the priced settings that would lower it most. Measured on `dit-xl-thirds` (`--attn flash --mirror bf16`, LoRA r16): a card with **12 GB free trains at full 32-layer depth** at crop 410 with 4 checkpoint segments. The real wall below that is the frozen base itself — its bf16 mirror is 8133 MB, so ~8.3 GB is the floor for this base at any settings, and 8 GB and 6 GB are correctly refused. `capabilities.trainDit.minVramMb` (8192) is a UI banner hint only; nothing refuses on it.
- Product loss = flow_snr (window-normalized — batch-1 mean normalization is silently a no-op, the trap) + channel_balance from `channel_stats.json`; lr default **5e-4** (1e-4 measurably does not train).
- Defaults (form seeds `ACE_DIT_FORM_DEFAULTS` / `ACE_DIT_LOKR_FORM_DEFAULTS` in `server/src/contracts/trainingRecipes.ts`; the `train-dit` route fallbacks match, and an omitted `adapterType` is LoKR): the card opens on **LoKR** (dim512/α512/factor 6, lr **2e-3**, weight decay 0.001, loss-weighting `none`); **LoRA** is r128/α256, lr 5e-4, loss-weighting `flow_snr`. Both use GA 4, genre ratio 30%, target-MLP **on**, Prodigy (since 2026-09-03; it forces `--lr` to 1.0, so the LR values apply to AdamW and Muon), `--mirror bf16-f32`, `--bwd mm` and flash attention. The quality dial sets Fast / Balanced / Thorough to 150 / 500 / 900 epochs at target loss 0.3 / 0.3 / 0.2; the route default is 500 epochs at 0.3. The next three bullets argue from the 2026-07-30 values (LoRA 400 epochs, LoKR 250, target 0.6). The CLI's own target loss is 0.4, which needs Side-Step-length runs; the epoch cap is the practical stop.
- **LoKR lr and grad-accum move together** (2026-07-30 five-run A/B on one album dataset). The old 1e-2 @ GA 20 reproduced Side-Step's effective batch of 20 by accumulation; 2e-3 @ GA 4 is the same effective LR per sample under linear scaling and measured *identical* epochs-to-target (227 vs 228) with better-behaved gradients (median grad-norm 0.062 vs 0.031 — the √5 a 5× smaller batch predicts — and no warmup spike, where GA 20 peaked at 13.5 on epoch 1). Setting one without the other is an untested config.
- **The 250-epoch LoKR horizon is a schedule fix, not a shortcut.** Every 400-horizon run stopped with the cosine only halfway down (LR still ~50% of peak), so it never decayed *into* the target. 250 cut epochs-to-0.6 from 228 → 203 and audio-seconds-to-target by 14% — the only one of five knobs tested that reduced the work required rather than rearranging it. Don't go much lower: that run used 203 of its 250, and a horizon under the epochs actually needed strands the LR at its 10% floor with the tail unfinished.
- **Time-to-target is set by audio-seconds seen, not epochs or update count.** ~85,000 audio-seconds to reach 0.6, stable to ±5% across a 5× change in update count and a 1.8× change in window length. Changing grad-accum or crop length only re-slices it; only the schedule shape (above) and engine throughput move it.
- Adapter output: PEFT dir, loads through **both** `adapter-merge.h` and runtime paths (round-trip reproduces trainer loss; that check is a permanent gate). Exported **BF16** since 2026-07-30 (LyCORIS/Side-Step parity; halves a dim-512 census from 872 → 436 MB). LK5 gates the dtype — it compares alpha as a *bit pattern* against the writer's own rounding, so an alpha that is not BF16-exact still fails honestly.

**LoKR `apply()` — never let the token axis reach `ne2` of a mul_mat** (2026-07-30, `docs/plans/2026-07-30-dit-trainer-step-profile.md`)
- **The single largest perf defect found in this trainer: 9.7×.** LoKR training ran at 1902 ms/step where the same run with a LoRA took 116 ms — 16.4× — and the whole gap was `DitLokrAdapter::apply()`.
- **Mechanism.** Both kron contractions have a 2-D trainable factor (`ne2 == 1`) as src0. If src1 carries the token count in `ne2`, ggml emits the weight gradient as `out_prod(src1, grad)` with `dst->ne[2] == S`, and ggml-cuda's `out_prod` takes its `dps2 > 1` **fallback**: one `cublasSgemm` per token (`out-prod.cu:96-108`). One `OUT_PROD [512,5,344,1]` node = 344 launches of a 512×5 GEMM; ×189 such nodes ≈ 65,000 micro-GEMMs per micro-step from that shape alone, plus a `repeat_back` to reduce the per-token gradient slabs. Op histogram: 80% `OUT_PROD` + 8% `REPEAT_BACK`.
- **Fix (free).** `[in_n, in_m, S]` and `[in_n, in_m·S]` are the SAME BYTES, so folding the token axis into the column count is a pure `ggml_reshape_2d` — it leaves `ne2 == 1`, takes the strided-batched fast path in one call, and removes the `repeat_back` entirely. Restore 3-D only around the `permute`s, which genuinely need it.
- **Generalise this.** Any adapter or side-path whose trainable factor is 2-D must keep tokens out of `ne2`, or it silently buys the per-token fallback. A LoRA never hit it because its A/B matmuls are already 2-D.
- **Contraction ORDER is the second win: −10% end-to-end** (2026-09-02, `docs/plans/2026-09-01-flash-attn-backward.md` "Stream A"). The two transposes above cannot be deleted — `mul_mat` writes the fresh index back to `ne0`, so `delta` coming out `[out_k, out_l·S]` forces the last contraction's src0 to be the w2 factor — but they can be **moved**. Both copies are `(in_m + out_l)·C` wide, where C is the channel count at the leg switch, so the cheapest chain is simply the one cutting at `min(out_k, in_n)` (monolithic) or `min(out_k, dim, in_n)` (factorized), ties to lower flops. `dit_lokr_pick_order()` decides per site at init and prints the census: on dit-xl-thirds at dim 512/factor 6 that is **224 w2-first / 128 w1-first**, flipping every H-input site whose output factorizes with `out_l < in_m` (self/cross `q_proj`, `gate`, `up`). Copy elements/token/layer 80,640 → 36,864 (−54%). Measured against a purpose-built pre-change binary (albumJ, production LoKR, crop 1250, 20 epochs): **−10.7% exact / −10.2% flash**, peak VRAM −820 MB / −1572 MB, auto-crop +9.1% at an equal estimate, at 3.8e-5 of summation-order loss drift. Riding along, both bit-identical: the `scale == 1.0f` node is skipped (alpha == dim on every production site, so it was 352 fwd + 352 bwd kernels per micro-step multiplying by exactly one) and the swap uses `ggml_cont_2d`.
- **The exact-mode arena term deliberately still prices the OLD order.** `dit_lokr_apply_arena_bytes()` keeps the all-W2 arithmetic on purpose: its over-count is covering for exact mode's own 13–18% under-prediction, so cutting it ~40% against a true arena that fell ~15% would spend a margin holding up a different, unrefitted fit. Flash mode does price the reorder (`dit_vram_lokr_apply_bytes_flash()`), but `DIT_FLASH_LOKR_RETENTION` was fitted pre-reorder and is now too conservative — the flash estimate over-predicts +16.5% where +7.5% was intended, i.e. about one crop step of headroom unspent.
- Gated by **LK3** (kron-matvec vs materialized-kron reference, 2.86e-08) and **LK4** (finite differences). A loss-level A/B is the *wrong* check here and misled once: the VRAM auto-fit picked crop 692 vs 694 between the two runs, so the losses differed for reasons unrelated to the change. Pin `--crop` **and** `--layers` if you ever want one.
- **`--profile-step <n>` / `--profile-ops`** are the tools that found it, and they stay: per-micro-step buckets (assemble/upload/build/backward/alloc/compute/readback/free) plus graph node and scheduler split counts, and a per-node op/shape histogram for one warmed-up step. `dit_train_log.json` now carries `runtime.backend` / `graph_nodes` / `sched_splits` / `profile_ms`, so a finished run explains itself — before this, "was that run even on CUDA?" was unanswerable after the fact.

**Muon optimizer** (2026-07-30, `--optimizer adamw|muon`, shared by both trainers via `lm-optim.h`)
- **The DiT default from the 2026-07-30 ear test until 2026-09-03**, when Prodigy replaced it; Muon and AdamW stay selectable. Measured on one album dataset (LoKR dim512+MLP, 32 layers, crop/depth pinned): **161 epochs to ma5 0.6 vs AdamW's 227** — 1.41× more sample-efficient — and ~1.23× on wall-clock once bucketed. A real run reached 0.6 in ~5 min and the adapter was judged perfect by ear.
- **Per PARAMETER, not global.** 2-D parameters with a short side ≥ `--muon-min-dim` (16) get orthogonalised-momentum (Newton-Schulz) updates; everything else stays on AdamW. A LoKR `w1` is `[4,5]` — orthogonalising a 4-dimensional subspace is vacuous — so on a DiT LoKR run the split is **448 Muon / 352 AdamW**. Every published Muon uses this hybrid.
- **The LR does NOT mean AdamW's.** Muon's update is normalised by construction, so per element it is ~20× smaller; `--muon-lr-scale` multiplies the shared schedule for Muon parameters only. Sweep on the DiT: 5 undershoots, **20 matches**, 50 overshoots. `--muon-lr-scale` and the base `--lr` are not independent.
- **Free VRAM:** one momentum buffer instead of two — 16315 → 15443 MB trainer-owned at LoKR dim512 (**872 MB**). `dit-vram.h` still charges for both, so the auto-fit is conservative, not wrong.
- **Bucketing is what makes it affordable, and its knob is non-monotone.** Same-shape parameters are batched into one Newton-Schulz. Optimizer step, LoKR dim512+MLP: **298.6 ms** (bucket 1) → 136.1 (8) → **138.0 (16, default)** → 187.3 (64) → 201.2 (128), against AdamW's **19.2 ms**. Larger buckets are WORSE because the gather is a left-folded `ggml_concat`, quadratic in bucket size. ~55 ms of what remains is irreducible NS FLOPs (~2.8 TFLOP/step); killing the concat (allocate accumulators/momenta as slabs of one per-bucket tensor) is the remaining win.
- **LM trainer: wired and functional, NOT yet defaulted.** All 392 LoRA parameters qualify at rank 16 (**the short side of a LoRA `A` is the rank** — a rank-8 adapter would fall entirely to AdamW; the startup line reports the split). 8-epoch smoke: loss 7.193 vs AdamW's 7.581 at +14% step time. That window is far too short to conclude anything — the DiT looked like parity at 10 epochs and was 1.41× ahead at 200. A run to target 4.0 is the honest test.
- Gated by **MU1**: update vs a double-precision host reference (CPU 5.7e-05, bar 1e-4; CUDA reported not gated — cuBLAS TF32 floors five chained iterations at ~4e-3), semi-orthogonality of **the graph's own** O, and a bucket case with grads spread ×8/×1/×⅛ that catches a global-instead-of-per-slab Frobenius norm.

**DiT micro-batching + gradient checkpointing** (2026-07-29, `docs/plans/2026-07-29-dit-batching-checkpointing.md`)
- `--batch <n>` (default **1** = OFF, range 1-16): crops per micro-batch, drawn from that many DIFFERENT songs (Side-Step DataLoader parity — B same-song crops was never on the table). Clamped down to the dataset's own song count with a `warn` when the variant has fewer songs than requested; further clamped by ggml's CUDA `REPEAT_BACK` cap on `Nkv·max(S,enc_S)·B` (GQA head-expansion backward — **both** attentions expand, and cross-attention's token axis is the dataset's padded `enc_S`, not the crop), and by the C4 auto-fit's own VRAM order (crop shrinks first, then checkpoint segments rise, then B itself gives with a warn — **before** depth is ever touched, since depth is the quality axis). `--grad-accum` counts **micro-batches**, not samples — effective samples/optimizer-step is `batch × grad-accum`.
- **Why the batch default is 1** (changed 2026-07-29 from 5, on the measurement below): batching is ~2.5× SLOWER at full depth on a 32 GB card and ~2.4× faster on shallow / partial-depth runs, so it ships off and is opt-in for the shallow case. The same default is mirrored in `DitTrainArgs`, the `ace-train` usage text, the `POST /api/training/datasets/:id/train-dit` route fallback, `TrainDitOptions` (both mirror files) and `ACE_DIT_FORM_DEFAULTS` (which `ACE_DIT_LOKR_FORM_DEFAULTS` spreads, so LoKR inherits it; `TrainDitForm.tsx` re-exports them as `TRAIN_DIT_DEFAULTS` / `TRAIN_DIT_LOKR_DEFAULTS`). `--ckpt` stays at 1 (auto), and auto resolves to a single unsegmented run whenever full depth fits — `seg_candidates()` tries 1 first — so the default pair is the monolithic graph.
- **Mixed-length batches mask the pad in ATTENTION, not just in the loss.** When any element of a micro-batch is padded (a song shorter than the crop), `dit_batch_assemble` switches the self-attention mask from the shared `[S,S]` broadcast to a per-element `[S,S,1,B]` pair: `sa` (sliding window + padded KV columns) for `layer_type 0` and `sa_pad` (padded KV columns only, no window) for `layer_type 1`, which otherwise takes no self-attention mask at all. Without this the pad reaches every valid query as an attention KEY and from there the adapter gradients — the design-B4 loss mask cannot stop it, since it only zeroes the padded frames' own loss contribution. Padded QUERY rows are deliberately left unmasked (a fully-masked softmax row is NaN); they are inert anyway, because a zero output gradient contributes exactly zero to dQ/dK/dV. Self-test rungs SB3/SC2 gate it by dirtying the padded INPUT and requiring loss and every adapter gradient to move by EXACTLY 0. The unpadded case keeps the `[S,S]` fast path, and B=1 can never pad.
- **Grad-accum weighting is an element share.** A micro-batch's `t_lossgrad` is `elems_in_micro_batch / elems_in_window`, not `1/n_micro_batches`: the in-graph `t_lw` already normalises within a micro-batch, so `sum_mb (nb/wlen) * loss_mb` telescopes to Side-Step's global mean-of-per-sample-means over the whole optimizer window. The two agree while every micro-batch is full; a short tail of `nb < B` elements used to be over-weighted by `B/nb`. At B=1 the two forms are algebraically identical, so the `--batch 1 --ckpt 0` anchor is untouched.
- `--ckpt <n>` (default **1** = auto): hand-rolled segmented-recompute checkpointing — `0` disables it (byte-identical to the pre-batching monolithic graph), `1` lets the VRAM fit pick a segment count, `2-32` pins it. Mechanism: one no-grad forward over the whole trained stack to capture segment-boundary activations, then segments rebuilt LAST→FIRST with grads, each feeding the next via a `spike-s3`-style surrogate loss (`sum(seg_out ⊙ G)`) seeded from the boundary gradient above it; adapter grads accumulate in-place across every segment's graph compute (`lm-optim.h` untouched — it never needed to change, since PARAM-flagged boundary tensors get their own grad slot through the existing `ggml_build_backward_expand` machinery). Bit-exact vs `--ckpt 0` at the same seed/batch (self-test SC1-SC3).
- JSONL: `step.micro` counts MICRO-BATCHES (not samples — samples is `micro × batch`, modulo a short tail). `start` cannot know the resolved batch/segment count (it is emitted before the model loads), so it carries `batchRequested` / `ckptRequested`; the RESOLVED pair is `vram.batch` / `vram.ckptSegments` (+ `ckptSource`), emitted after the fit.
- Server/UI: `TrainDitOptions.batch` / `.ckptSegments` (route validation mirrors the CLI ranges; `ckptSegments` is the CLI's `--ckpt` value verbatim — 0/1/2-32). TrainDitForm's Advanced drawer has "Batch size" and a "Checkpointing" select (Auto/Off/2/4/8/16).
- VRAM model (`dit-vram.h`): the arena/KV-expand terms scale with a straight `×B`, validated (not re-derived — the pre-existing arena polynomial was already the fitted part) against a measured {B 1,5} × {ckpt 0,4} × {crop 375,750} grid at `--layers 8`: every cell landed within **+0.1% to +6.9%** of the trainer-owned high-water peak, always over- never under-predicting. The boundary-buffer term is exact arithmetic (matched every grid cell's reported `boundaryMb` to the byte). See the dated comment block in `dit-vram.h` for the full 8-cell table.
- Throughput (RTX 5090, 32 GB, LoKR dim512+MLP, full available depth = 32 layers, `--mirror bf16`, 10 epochs, same seed, tiny 5-song smoke tensors): `--batch 1 --ckpt 0` (yesterday's behavior) auto-fit a 674-frame crop and ran ~13.5 audio-seconds of training content per wall-clock second; `--batch 5 --ckpt auto` was forced to 4 checkpoint segments to fit batch 5 at full depth at all (batch 5 does **not** fit full-depth on this card with `--ckpt 0` — the auto-fit correctly reduces B to 1 rather than touch depth, confirming the C4 order), auto-fit a smaller 524-frame crop, and ran ~5.3 audio-seconds/wall-second — **~2.5× SLOWER**, not the ≥3× hoped for. Layers=8 (shallow, LoRA rank-16, no MLP) tells the opposite story: the grid's own probe timings show batching there is ~2.4× **faster** per crop. Read together: batching's ROI is depth/shape-dependent — a already-compute-bound full-depth 32-layer graph on a fast card gets little from wider batches while paying checkpointing's ~30-40% extra-forward tax and a forced-smaller crop; a shallow/small graph that under-uses the GPU at batch 1 gets a real win. Neither Side-Step's 0.223 s/(60 s crop) anchor nor the ≥3× target is met at full depth on this card/config — reported honestly per the design's own instruction to do so; smaller-depth or larger-VRAM configurations are the likelier place batching pays off and are untested here.
- bf16-mirror A/B (same-seed loss-curve diff, `--mirror f32` vs `--mirror bf16`) repeated at `--batch 5` (LoRA rank-16, `--layers 8`, `--ckpt 0`, 10 epochs): max abs loss delta **7.9e-5**, max relative delta **5.8e-5** — essentially unchanged from the original batch-1 measurement (7.6e-5), so amendment A2's CUDA GEMM-route change (batched-cublas at B>1) does not measurably widen the f32/bf16 drift.
- **`--mirror bf16-f32` (2026-09-02) is the mode to reach for, not `bf16`.** `bf16` does not merely store the frozen weight in BF16 — it rounds activations and gradients to BF16 at every trainable-layer GEMM, and the adapters that come out of it sound coarse and "bitty" by ear. `bf16-f32` keeps the BF16 store and casts each weight to F32 at its `mul_mat` site, so the arithmetic is the f32 mirror's while residency is the bf16 mirror's.
- Measured on one album dataset (32-layer XL base, LoKR dim 512 + MLP, `--attn flash --bwd mm --optimizer muon`, 12 same-seed epochs, crop pinned 552): `bf16-f32` is **bit-identical to `--mirror f32`** on `loss`/`ma5`/`raw`/`gnorm` at every epoch — to the last digit of the full-precision run log, and the two exported `lokr_weights.safetensors` share an MD5 — where `bf16` diverges from epoch 1 and reaches 7.8e-3 by epoch 12 (and 2.0e-2 at epoch 6, where its `gnorm` reads 20.7 against 13.0). Mirror 8133 MB vs f32's 15893 MB; trainer-owned peak 14370 MB vs bf16's 14190 MB and f32's 21997 MB — the transient F32 window costs 180 MB, not 7.8 GB. Unpinned, the flash auto-fit picked crop **1542** against f32's 610 and bf16's 1632 (before the 800-frame default cap). It is not free: ~3.21 s/epoch against f32's 2.56 and bf16's 2.17 (+25% over f32 at equal crop), which the 2.5× crop repays.
- Loss curves at the throughput bench's Side-Step-parity LR (0.01) do **not** cleanly descend over just 10 toy epochs on a 5-song dataset — the cosine schedule (tuned for hundreds of production epochs) compresses its whole warmup+decay into 10-20 steps, producing a noisy rise-then-plateau in both the batch-1 and batch-5 runs. A same-config re-run at LR 0.001 descends cleanly in both (ma5 1.27→1.04 at batch 1, 1.34→1.09 at batch 5) confirming the mechanism itself is sound — the non-descent at LR 0.01 is a toy-bench artifact (LR schedule too aggressive for a 10-epoch/5-song smoke run), not a regression.

**`--bwd <outprod|mm>` — the mul_mat-backward reformulation** (2026-07-29, the HOT-ggml `mm-backward` change)
- **The constraint it removes.** ggml emits the MUL_MAT *activation* gradient as `out_prod(W, transpose(grad))`, and ggml-cuda's `OUT_PROD` is F32-only (`cublasSgemm`). That forces the frozen weight to be F32, which in turn drags the **forward** `mul_mat` into `ggml_cuda_op_mul_mat_cublas`'s F32 branch → TF32 tensor cores. Upstream `ggml.c` has always carried the alternative, commented out: `mul_mat(cont(transpose(W)), grad)`. It is dtype-agnostic, so a BF16 weight rides real BF16 tensor cores in **both** directions with no dequant and no F32 window. `--bwd mm` turns it on.
- **The two arms are provably the same op**, not an approximation: `out_prod(W[n,m,q1,r1], transpose(grad)[p,m,qq,rr]) -> [n,p,qq,rr]` and `mul_mat(cont(transpose(W))[m,n,q1,r1], grad[m,p,qq,rr]) -> [n,p,qq,rr]`, and `ggml_can_out_prod` / `ggml_can_mul_mat` reduce to the identical `b->ne[2] % a->ne[2] == 0, b->ne[3] % a->ne[3] == 0` broadcast pair. This is **not** a quality trade like `--weights bf16`, which genuinely changes the quantity computed — `--bwd` only changes how the same quantity is reached.
- **Per-layer GEMM measurement** (`ace-train spike gemmbench`, RTX 5090, Qwen3-4B projection shapes, reproduced 2026-07-29): `bf16_pre` beats the shipped `cur_lowvr` mix by **1.67–1.78×** per layer per step (1.41–1.64× vs `cur_naive`). Gradient parity vs the TF32 `out_prod` reference: **cosine 0.999996, max relative ~3–4e-3**. `bf16_cont` ≈ `bf16_pre`, i.e. the per-use `cont(transpose(W))` is free at these shapes — so the implementation needs **no pre-transposed weight cache**, which is why it lives entirely in graph-construction code and stays backend-agnostic.
- **End-to-end it is much smaller than that — 1.10–1.17×, and you must interleave to see it at all.** Same-seed `train-dit` A/B (LoKR dim512+MLP, `--mirror bf16 --layers 8 --batch 1 --ckpt 0 --order fixed`, 3 interleaved rounds per arm, min/median over epochs 2-6): crop 1250 → outprod median 4179 ms/epoch vs mm 3785 ms = **1.10×**; crop 375 → 1435 vs 1229 ms = **1.17×**. The gap vs the per-layer figure is Amdahl, not a broken lever: at `--layers 8` only 8 of 32 layers have a backward at all, and the step is dominated by the retained-softmax O(S²·Nh) attention (exact mode; measured before the flash-attention backward existed) plus LoKR kron reconstruction — which is exactly why the smaller crop, with less attention per projection, scores higher. **Do not trust a single non-interleaved A/B here**: run-to-run contention on a working machine is ±10%, the same size as the effect, and the first two one-shot runs gave 1.47× and 0.82× purely on which arm caught the interference.
- **Loss drift is negligible and deterministic.** Over 10 same-seed epochs: max absolute delta **7.4e-5**, max relative **6.7e-5** (6-epoch runs: 4.3e-5 at crop 1250, 2.7e-5 at crop 375); epoch 1 is bit-identical to 9 decimals, and each arm reproduces its own curve exactly across 3 repeats. Same order as the `--mirror f32`/`bf16` A/B (7.9e-5) — this is GEMM-reassociation rounding, not a different training run.
- **Self-test:** `ace-train train-dit --self-test` gives the identical **21/22** rung-for-rung with the env unset and with `GGML_BACKWARD_MM=1`, T9 (the known `E[v^2]` fingerprint drift) the only failure in both. The FD gate T4 reports the same max rel 2.3142e-04 on the same loss 1.659584 either way; SC1's graph grows 4386→4505 nodes under mm (the extra `cont` per converted site) while its loss and its segmented-vs-monolithic gradient delta stay bit-identical.
- **Contiguity guard (found by the self-test, not by reading).** `grad` becomes `mul_mat`'s src1, and the CUDA mul_mat kernels require it row-contiguous — `ggml-cuda/mmf.cu:28` asserts `nb10 == ts_src1` and **aborts** otherwise, where `out_prod` happily takes an arbitrary transposed view. The patch therefore takes the mm arm only when `ggml_is_contiguous(grad)` and otherwise falls back to out_prod. Without the guard the LoKR rung SC3 hard-aborts. `ggml_cont` on grad is not the fix — grad is activation-sized, so copying it costs more than the GEMM saves.
- **How it is wired.** It is a commit on the HOT-ggml fork that `engine/ggml` is pinned to (originally a vendored patch beside `bf16-out-prod`; [ggml-fork.md](ggml-fork.md)), touching `ggml.c` only, and guarded by **Hook 8** in `engine/verify-hooks.ps1`. The change is **env-gated**: with `GGML_BACKWARD_MM` unset the emitted graph is byte-identical to upstream, so a stock build regresses nothing. `ace-train`'s `--bwd mm` sets the variable in `cmd_train_lm`/`cmd_train_dit` **before** `ggml_time_init()` and any model load — the patch latches it into a static on the first backward it builds, so it must be set before graph construction, and setting it there covers `--self-test` too. `--bwd outprod` deliberately does *not* clear an externally-set variable, so exporting `GGML_BACKWARD_MM=1` still A/Bs the whole self-test battery.
- **`--weights bf16` (LM) and `--bwd mm` COLLIDE — refused, not coerced.** Lever A reaches the same mul_mat backward by a different route: it lets ggml build the out_prod form and then rewrites those nodes in place, asserting **exactly 7 rewrites / 0 skipped / 0 residual** per segment (its S18 tripwire). Under `--bwd mm` ggml emits mul_mat directly, the surgery finds nothing, and the tripwire `GGML_ABORT`s — correct behaviour, but mid-run, after the model load. `ace-train train-lm` now exits 2 on the pair and the route answers 400. There is no version of that pair worth building: on the `f32-window` path `--bwd mm` gains nothing either, because the weight it transposes is the **F32 window**, so the GEMM stays TF32 and the extra `cont` is pure cost. **The LM already solved this problem its own way; `--bwd mm` is a train-dit lever.**
- **Defaults are deliberately split three ways.** The engine default is `outprod` (a bare `ace-train` invocation is unchanged). The **server** defaults **train-dit to `mm`** and **train-lm to `outprod`** — the LM's `weights` already defaults to `bf16`, so an LM default of `mm` would brick the *default* LM job on the collision above. `buildTrainLmArgs`/`buildTrainDitArgs` always emit `--bwd` so an older `ace-train.exe` rejects it loudly instead of silently running the slow path. Surfaced as "Backward GEMM" in both Advanced drawers; recorded in the `start` JSONL event and in `dit_train_log.json` / `lm_train_log.json`'s `config.bwd`.

**Trigger words (embedded in the adapter)**
- The tag was always trained in — `pp_caption_text` in `preprocess-run.h` bakes `custom_tag` into the caption before text encoding, and `lm-extract.h` re-applies it — but export used to drop it, leaving inference to guess the trigger from the filename (wrong for our adapters: the dir is `<name>-<size>`, not the tag).
- Both trainers now write `hot_step_trigger`, `hot_step_trigger_position` and `modelspec.trigger_phrase` into the adapter's safetensors `__metadata__`. Tensors and `adapter_config.json` are untouched, so ComfyUI/PEFT/Side-Step load it exactly as before. **Not** `adapter_config.json` — PEFT does `LoraConfig(**json)` and unknown keys are a version-dependent TypeError.
- Source of the value: `--trigger`/`--trigger-position` flags, else the variant's `preprocess_meta.json` (`custom_tag`/`tag_position`). The **variant meta, not the dataset row**, is authoritative — a dataset's tag can be edited after preprocessing, in which case the tensors (and the adapter) carry the old one.
- **All three positions are real, and `replace` means the caption IS the trigger (2026-08-12).** Before this, `replace` applied no tag at all during preprocessing (verbatim Side-Step), so the Build panel's "Instead of the caption" option silently trained a *captionless, triggerless* adapter and export refused to embed a trigger for it. Now `pp_caption_text` collapses the caption to the tag, `lm_apply_tag` matches, and `lm_trigger_normalize` embeds it. Under `replace` the sidecar's descriptive caption **and** the genre ratio are both discarded — only the `# Metas` block (bpm/timesignature/keyscale/duration) and the lyric prompt still vary per sample.
- **Changing `custom_tag`/`tag_position` re-encodes.** The trigger pair is part of the caption the text encoder baked into each tensor file, so both are compared in the P20 resume check (`PCacheSettings`). They were absent from that comparison until 2026-08-12, which meant flipping `tag_position` on an already-preprocessed dataset skipped every file and trained on the old captions while `preprocess_meta.json` advertised the new setting. Unlike the other keys, an **empty/absent** cached value still counts as a difference (empty normalises to `prepend`, and "no tag → tag" is a genuine caption change).
- Generation-side resolution lives in `services/generation/triggerWords.ts` and runs **server-side** for every caller: per adapter, manual override → embedded → filename fallback → none. A stack can mix prepend and append. An embedded `replace` is passed through verbatim — never downgraded to `prepend`, since an adapter trained that way only ever saw the bare trigger and prepending it to prose is exactly the off-distribution case the embedded position exists to prevent. The gate widened from "a DiT adapter is loaded" to "any adapter", so planner-LM adapters now contribute their trigger too.
- Legacy corpus: `server/scripts/stamp-adapter-triggers.mjs`, dry-run by default. Verified on a copy — 132 MB payload SHA-256 identical, header stays 8-byte aligned, `.bak` kept.

**Adapter layout (per-base + per-run, 2026-07-28)**
- `<adapters>/lm-06b|lm-17b|lm-4b/<artist>/<run>/` and `<adapters>/dit-<shorthand>/<artist>/<run>/`, where `<run>` = `YYYY-MM-DD_HH-MM-SS` (logs/ convention) — retraining an artist never overwrites an earlier adapter. Artist names carry **no** `-<size>` suffix; the parent folder says the base.
- Single source of truth: `server/src/services/training/adapterLayout.ts` (size slugs, the confirmed DiT shorthand map `xl-thirds`/`xl-base-turbo`/`xl-sft-turbo`/…, run stamps, latest-run resolution). `migrate-adapter-layout.mjs` moves an old corpus; its shorthand map must stay in sync.
- Writes go through `lmRunDirFor`/`ditRunDirFor` (fresh stamped dir); reads through `adapterDirFor`/`adapterDitDirFor` (newest run → unversioned artist dir → legacy flat `lm/<name>-4B` / root DiT dir). Two legacy forms are read-everywhere, written-never.
- Scanners: `GET /api/adapters/lm` walks all lm-* roots + legacy `lm/` (entries carry `lmSize`, `run`, `trigger`) — this is what the global-bar planner list, Lyric Studio's preset picker and the Training Studio picker all consume. `POST /api/adapters/scan` descends into `dit-*` folders (and their run subdirs) only.
- The audition's base-LM pinning now derives the size from the `lm-<size>` parent folder (suffix kept as legacy fallback) — do not reintroduce suffix-only derivation.

**Audition (pure-LM preview)**
- `POST /codes-decode`: codes → `detok_ggml_decode` → tail-call the existing VAE decode worker (layouts are byte-identical; zero changes to `/vae`). Deterministic; full song ≈ 3.1 s warm.
- A/B = two `/lm` calls, same explicit `lm_seed`, base side **must** byte-match an adapterless run (V5 hard gate). Base LM auto-derived from the adapter's `lm-<size>` parent folder, with the `-<size>` name suffix as the legacy fallback (`lmSizeOfAdapterPath`, then `pickLmFor`) — never let the engine's `resolve_name` fallback pick (sticky 0.6B met a 4B adapter: "36 layers but model has 28").
- LM-echo sideband trap: never forward the `/lm` reply into another request — build requests fresh (see `generation-request-flow` skill).

**Adapter parameterizations beyond LoRA/LoKr (2026-09-04)**
All are `train-dit --adapter-type lora` variants (rsLoRA and LoRA+ also in `train-lm`); DoRA, HiRA and LoHa exclude each other, rsLoRA and LoRA+ stack with any. None is ear-validated yet — every one passed the gates below and nothing more. Plan and results: `docs/plans/adapter-parameterizations-roadmap.md` (local).

| flag | delta | export / load | gate that passed |
|---|---|---|---|
| `--dora` | LoRA + learned per-output magnitude `m`, detached column norm refreshed once per optimizer window (`DitAdapter::preWindow`, one graph PER LAYER — a whole-adapter graph overflows the scheduler hash set) | PEFT `lora_magnitude_vector`; merge path already read it | `HOTSTEP_DIT_ST_DORA=1 --self-test` T4, 24/24 |
| `--rslora` | scale `alpha/sqrt(r)` in-graph | `use_rslora: true`; honoured by merge, runtime and the LM loader — trainer and loaders MUST agree or every adapter merges at the wrong strength silently | config + merge log |
| `--lora-plus-ratio f` | B tensors at `f x` A's LR (paper 16). AdamW-rule tensors only; Muon ignores `lr_mul` | none (training-only) | per-tensor group log |
| `--hira` | `W + W (.) (BA)` — a full `[in,out]` delta per site, materialised in-graph | `peft_type: HIRA`; merge = Hadamard with the base; runtime low-rank path REFUSES | `HOTSTEP_DIT_ST_HIRA=1` T4; merge 352 pairs |
| `--hra` | HRA (Householder Reflection Adaptation): `y = W (R x)`, `R = H_{r-1}..H_0`, `H = I - 2vv^T/||v||^2`, r = `--rank` (even; pairs start equal so R = I). Orthogonal fine-tuning without the Cayley inverse OFT/BOFT need, which ggml cannot express. Each reflection is a matvec + broadcast divide + K=1 outer-product matmul on the site's input (not `ggml_repeat`: REPEAT's shape template may not need gradients) | `R - I` has rank <= r, so `W(R-I) = (W U) Q^T` (Q = basis of span(v), U = (R-I)Q) is an EXACT rank-r plain LoRA (alpha = r); `W U` runs on the GPU through the adapter's scheduler handle; raw vectors in `hra_vectors.safetensors` for the resume. Step time ~10x a LoRA's at r=8 (the reflections run on every site's activations): 25 s vs 2.5 s per epoch on a one-album smoke | `HOTSTEP_DIT_ST_HRA=1` T4 (16 probes, max rel 5.6e-4); 2-epoch smoke; resume (352 tensors); merge 352 pairs |
| `--pissa` | init only: A/B start on W's top-r singular directions (GPU randomized SVD: q = r+8 columns, 2 power iterations, host QR + Jacobi on the small side) and the mirror's base keeps the residual `W - s B0 A0`, so the first forward is exactly W. Rank 16 captures ~50% of the weight energy on this DiT (random subspace: ~0.6%) | plain PEFT LoRA of rank 2r on the ORIGINAL base: `s(BA - B0A0) = s[B,-B0][A;A0]`, `r` and `lora_alpha` doubled; A0/B0 live in `pissa_init.L<l>.safetensors` beside the run. Merge and runtime need nothing new. NOT resumable (refused) | init energy diagnostic; 2-epoch smoke; merge |
| `--loha` | `W + (B1A1) (.) (B2A2)` (LyCORIS LoHa), rank = `--rank`, effective rank up to r^2 | LyCORIS `lycoris_layers_<l>_<site>.hada_w{1,2}_{a,b}` + `.alpha` inside `adapter_model.safetensors`, `peft_type: LOHA`; `adapter_merge_loha` mirrors the LoKr merge; runtime REFUSES | `HOTSTEP_DIT_ST_LOHA=1` T4; merge 352 modules |

- **A trained cond leaf changes the checkpoint plan.** The plan normally segments only the trained window and runs the frozen layers forward-only — exact for adapters, WRONG for anything the frozen layers consume: the V1 artist token feeds every layer's cross-attention, and the window plan dropped ~96% of its gradient (SC1 with `HOTSTEP_DIT_ST_ARTIST=4`: rel 1.01 vs monolithic, same at 1 and 2 segments; whole-stack plan: projection 1.0000, rel 1.7e-4). With a token the plan now spans all layers at the same layers-per-segment (`vm.full_stack_ckpt`), `--ckpt 0` is forced on, and the backward is correspondingly longer. Any future trained leaf below `lora_lo` needs the same.
- HiRA and LoHa hold weight-sized F32 tensors per site for backward: `dit_hira_apply_arena_bytes(..., n_full)` (2 for HiRA, 3 for LoHa) is in the VRAM model and the auto-fit.
- **FD-gate trap (hit twice):** an init that pins one factor to exactly zero (LoRA B, LoHa B2) makes the other factors' gradients exactly 0, and T4's `rel = |fd-g| / max(|fd|,|g|,1e-6)` then scores f32 noise against zero and fails. The rung sets `cfg.b_sigma`; every new parameterization's init must honour it. The self-test switches live in the FD child's cfg block (dit-selftest.h ~3870), not the first `b_sigma` block — the wrong block validates nothing.
- Resume: `--init-adapter` restores DoRA's `m` (absent = keeps the norm init, i.e. a plain-LoRA source resumed as DoRA) and LoHa's four factors through their LyCORIS keys; HiRA shares LoRA's A/B names. Round-tripped 2026-09-04 (1056 and 1408 tensors, epoch 1 at the source loss).

**Soft prompts on the AS1.5 LM (artist tokens + KV prefix, 2026-09-03/04)**
- `train-lm --artist-token <name> [--artist-token-k 32] [--artist-token-init <caption word>] [--artist-token-lr 5e-3] [--prefix-n 8]`. The token is k learned embedding rows behind a placeholder spliced into every caption (`prompt.h` `lm_append_user_span`, shared with inference so the k==0 prompt is byte-identical); the prefix is n trainable K/V columns per layer riding the frozen-KV-prefix machinery (no RoPE on prefix K; every row sees it; forces head-blocked attention OFF — the 4B's blocked path asserts on a prefix mask).
- **One file:** everything lands in `adapter_model.safetensors` — `hot_step.artist_token.{vec,meta}`, `hot_step.prefix.L<l>.{k,v}`, `hot_step.prefix.meta` (site 1 = as15_lm, 2 = mm3_lm). The LM loader installs them when the adapter is selected; the engine applies the token at the prompt offset and seeds the prefix into the Phase-2 KV cache. `HOTSTEP_ARTIST_TOKEN` is only a standalone override.
- Soft-prompt tensors get their own LR group (`--artist-token-lr`, always AdamW, even under Muon/Prodigy). They learn ~50x faster than the LoRA, so `--target-loss 1.5` fires early on a joint run (17 epochs, and the eval then showed a 6 s verbatim replay of training codes). The app's joint recipe stops at 4.0 instead: the arm picked by ear stopped at 3.87 (epoch 42) with no replay.
- **Resume:** `--init-adapter` adopts the trained token/prefix from the file (lm-train-run.h, after the prefix init). The server therefore emits the soft-prompt flags on EVERY leg of a staged chain; a k/n that differs from the source run is refused as a shape mismatch.
- Measured (one album, 4B, `lm-adapter-eval.ts`, one run each): joint LoRA+token+prefix and LoRA-only both TOWARD; k=8 edged LoRA-only on transitions only; **k=32 won every metric** (unigram JS 0.739 vs 0.771; transitions 0.0121 vs 0.0177, under the album's own 0.0178 floor). The 2026-09-04 listening test then picked the joint k=32 arm (arm 05) by ear, and it became the app default (see the caption dropout block above). Both trainers' Training Studio controls exist; the DiT-side contextualised token (V4) still needs the preprocess-stage encoders differentiated.
- Gradient sanity for anything touching the embedding stage: pin the token with `--lr 1e-12` and compare naive vs checkpointed loss/gnorm (9.354661/9.354632 agreed). `LmCkptRun::embed_build` alone STOPS uploading token ids — a get_rows hook needs `embed_uses_tok` or it reads a stale buffer (loss 24.5 vs 9.35, no error).

## Build & verification rules

- `ace-train`-only changes: `cmake --build engine/build --config Release --target ace-train` — safe with the app running **if no ace-train.exe process is live** (training runs spawn it). New `train/*.h` headers need **no CMake change**.
- `hot-step-server.cpp` (ace-server) changes: full `dev-rebuild.bat` cycle (app goes down; restart it with `dev.bat`). Never `build.cmd` directly, never `--clean-first`.
- Self-tests are the regression net: `ace-train train-lm --self-test`, `ace-train train-dit --self-test`, `ace-train spike …` (gemmbench, bf16layer, dit2 — the measurement harnesses that justified every design call; keep them).
- GPU discipline for agents: `nvidia-smi` before runs, stay under ~29 GB total, never kill the user's python/node/ace-server, bounded runs while the machine is in use.
- TypeScript: `server` `npx tsc --noEmit` (covers `server/src/` only; `server/scripts/` needs `npx tsc --noEmit -p scripts/tsconfig.json`); `ui` `npx tsc -p tsconfig.app.json --noEmit` (a bare `npx tsc --noEmit` in `ui/` checks nothing, see AGENTS.md).

## Measured reference numbers

| Thing | Number |
|---|---|
| LM 0.6B full E2E (17 songs, 16 epochs) | ~70 s wall |
| LM 4B, checkpointed | ~12.4 GB trainer-owned, ~1.06–1.34 s/micro-step; a real run auto-stops ~epoch 29 at target 4.0 ≈ 12 min (bf16: ~9.5 min) |
| DiT full-depth, LoRA r16 attn-only, crop 750 frames (**S=375**) | ~147 ms/step, ~23 GB; crop ceiling ~1000–1250 frames @24 GB. **Frames, not tokens: `S = frames / patch_size` and patch is 2, so "crop 750" is 750 frames = 30 s = S 375.** |
| DiT full-depth **LoKR dim512 + MLP**, crop ~690 frames (S≈344), RTX 5090 | **~195 ms/step, ~2.7 s/epoch at 14 songs** (was 1902 ms/step before the 2026-07-30 `apply()` fix). A 14-song adapter reaches ma5 0.6 in **~9 min / ~196 epochs**, vs ~90 min before |
| Preprocess | ~6 GB VRAM (vae-chunk 384; 1024 costs ~17 GB), ~3 s/song |
| Codes decode (audition) | ~200 ms warm 30 s clip; ~3.1 s full song |
| TF32 vs BF16 GEMM (5090) | ~95 vs ~165 TFLOP/s; end-to-end lever 1.256× at 4B |

## Pending / open decisions

1. ~~One `dev-rebuild` owed~~ — **done**: the 2026-07-28 17:05 rebuild (during the SuperSep work) postdates every one of those source edits; `ace-train train-dit --help` shows `--no-target-mlp` and `ace-server.exe` carries the audition fixes.
1b. **Trigger stamper awaits approval**: `node server/scripts/stamp-adapter-triggers.mjs --datasets D:\Ace-Step-Latest\Datasets-LoRA-LoKR` prints the proposed table for 179 adapters (28 from real dataset.json ground truth, 151 from the filename). Nothing is written without `--apply`. Also outstanding: the per-adapter **manual trigger override** UI — the server already accepts `triggerSpecs` entries with `source:'override'` and a `path`, but no control emits them yet.
2. **bf16 listen test**: train twin LM adapters (f32-window vs `--weights bf16`), A/B via audition; ship-call by ear.
3. ~~**LoKR for DiT**~~: done. `train-dit --adapter-type lokr` (`dit-adapter-lokr.h`), and the ACE card and `train-dit` route default to it.
4. **ConvRot bases**: `train-dit` still refuses a DiT GGUF that carries `acestep.convrot_map` (`convrot-unsupported`, `dit-mirror.h`) pending a convrot spike (`…-convrot-*` GGUFs exist). The YuE2 joint trainer has its own CUDA `CONVROT8` path for its ConvRot base.
5. ~~**BF16 `out_prod` vendor patch**~~: settled. `engine/ggml` is now the HOT-ggml fork, and its `bf16-out-prod` change (Hook 7) gives ggml-cuda's `OUT_PROD` a BF16 `src0`. Small cards were unlocked another way: flash attention plus the bf16 mirror train at full depth on 12 GB free (DiT trainer above).
6. **Top-K DiT adapter quality** on small cards: runs, but musical usefulness unmeasured.
7. Micro-batching: closed with measurements; revisit only if the graph-build overhead picture changes.
8. UI `minVram` hint string may be optimistic at r128+MLP defaults.

## Where the deep documents live

`docs/plans/` is **gitignored (local-only)** — on the maintainer's machine the full design/implementation plans exist: `2026-07-27-training-studio-design.md` (master design), `-dataset-studio-implementation.md`, `-preprocess-implementation.md`, `-lm-trainer-implementation.md`, `2026-07-28-lm-4b-training.md`, `-dit-trainer-implementation.md`, `-lm-speed-levers.md`, `-codes-preview.md`. Each contains the frozen contracts and verification ladders. The commit history (`git log --oneline --grep=training`) carries the measured numbers per milestone.

## Release data paths

Release defaults place training under `<data folder>/library/training`. The server supplies absolute child paths; moves rewrite persisted dataset and checkpoint paths inside transferred folders after a database backup. External paths stay unchanged. See [Data folder and upgrades](../user/data-folder.md).
