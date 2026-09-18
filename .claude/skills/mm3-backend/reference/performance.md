# Performance, caches and streaming

> Reference for the `mm3-backend` skill. Read only when the task needs it.

## Alternative composer LMs (depth-pruned + guidance-distilled) — 2026-08-22

The composer LM is **independently swappable**: the 5-way split means a variant replaces
`mm3-lm-*.gguf` only, and depth/cond/dit/voc stay as they are. Proven with
[Mothersuperior/minimax-music3-composer-5.7b-distilled](https://huggingface.co/Mothersuperior/minimax-music3-composer-5.7b-distilled)
(36 → 21 blocks, 5.69B, repair-distilled against the teacher's **CFG-guided** distributions;
two LR arms). Everything but the block count is bit-for-bit stock — the depth decoder
consumes a 4096-wide hidden and the 200 k audio vocabulary is what makes it a music model,
so those cannot move.

```
python engine/tools/convert-mm3.py --src <arm-dir> --out models/mm3 \
  --components lm --quant q8_0 --lm-layers 21 --ar-cfg-scale 1.0 \
  --suffix=-d21-lr6e5 --tokenizer <official>/tokenizer/tokenizer.json
```

- `--suffix` (needs `=`, else argparse eats the leading `-`) makes the whole trailing token the
  variant name, so tagged files appear as extra entries in the LM dropdown next to the stock
  quants. That is the A/B mechanism — no file juggling.
- `--lm-layers` is guarded by the **leftover-tensor diff**, not by trust: a wrong count leaves
  whole `model.layers.N.*` groups unconsumed and the run dies.
- `source layout: unknown` on a bare `Qwen3ForCausalLM` dir is expected and harmless.

**CFG 1.0 means single-row, and the engine acts on it.** `mm3_cfg_rows()` (mm3-model.h) returns
1 when `mm3.ar.cfg_scale == 1.0`, because `u + (c-u)*1.0` is identically `c` — the
unconditional row would be computed, read back and cancelled. The LM graph, its KV cache and
the depth decoder all build single-row; the AR loop mirrors row 0 into row 1 so every consumer
downstream stays unconditional. Keyed on the arithmetic, never a model name.

Measured (RTX 5090, matched caption/seed/duration, only the LM swapped):

| | teacher 36L / 2 rows | distilled 21L / 1 row |
|---|---|---|
| LM decode | 8.0 ms/step | **3.8** |
| depth decode | 9.4 ms/frame | **7.9** |
| AR stage | 5317 ms | **3534** |
| end to end (12 s clip) | 9.5 s | **7.7 s** |
| LM KV cache | 288 kB/pos | **84 kB/pos** |
| Q8_0 file | 9.13 GB | **6.05 GB** |

**The depth decoder is the clean control for the row change alone** — identical weights in both
runs, so its 1.19× is bought purely by dropping the row. Design note A ("2 rows are ~free
because decode is bandwidth-bound") is therefore only *mostly* right: at these tiny per-row
matmuls the second row costs ~20 %, not ~0 %. The rest of the LM's 2.1× is the 36→21 prune.

**Casualty: LRC alignment.** `MM3_ALIGN_HEADS` (mm3-align.h) pins layers 12/19/**24**, found
empirically on the teacher. On 21 layers, 24 does not exist — the replay loop clamps
(`mm3-lm-graph.h`, `i < m.lm.blk.size()`) so nothing crashes, but the heads are
teacher-specific and the timestamps are not to be trusted. Re-discovery
(`MM3_ALIGN_DUMP=1`) would be needed per variant. Audio is unaffected.

**Not yet judged by ear.** Renders staged in
`_experiments/_LISTENING/2026-08-22_mm3-distilled-lm/`. Remember trap #10 before drawing any
conclusion from them: this is a **planner** swap, so 02/03 are different takes, not degraded
copies of 01, and a single seed proves nothing.

## Low step counts go THIN, not dull — and why (root-caused + fixed 2026-08-21)

Dropping `steps` below the checkpoint's 30 degrades in a specific, non-obvious way.
Measured on a matched 10-vs-30-step pair (same seed/caption/structure):

| | 10-step vs 30-step |
|---|---|
| L/R correlation | **−0.07** vs +0.77 (anti-phase mids, 160 Hz–2.6 kHz) |
| side/mid ratio | **+0.5 dB** vs −9.5 dB |
| Mid spectrum | **−8 dB @ 60 Hz**, tapering to **0 dB above 2 kHz** |

So it is thin and phasey ("tinny", "cheap radio"), **not** dull — HF is already at the
correct *absolute* level. Do not reach for a treble fix; the tilt is the illusion.

**Root cause, three facts that only bite together:**
1. The vocoder decodes latent channels **0–63 as LEFT and 64–127 as RIGHT in two
   INDEPENDENT passes** (`mm3-vocoder-graph.h:596`). Zero cross-channel coupling.
2. Initial noise is **i.i.d. across all 128 channels** (`mm3-pipeline.h:462`), so the two
   halves start completely uncorrelated.
3. The schedule is **uniform, shift=1** (`mm3-dit-graph.h:744`), faithful to upstream —
   and the GGUF declares `mm3.flow.steps = 30`. **There is no low-step compensation in
   the reference at all.**

⇒ Every bit of stereo coherence must be manufactured by the DiT along the trajectory.
Coarse Euler steps leave that work unfinished, and the same starved high-noise phase
costs the low end. It is NOT residual noise: the excess side energy tracks the music
envelope at +0.93 and *drops* in quiet passages — that measurement is what rules the
noise hypothesis out, so run it before assuming otherwise.

**The fix (shipped, server-side, no rebuild):** `shift = 29 / (steps − 1)`, derived by
setting the shifted grid's first step `1/(shift·(steps−1)+1)` equal to the 30-step
native `1/30`. Returns exactly 1.0 at 30 steps, so the curve is continuous and can
never perturb a default render. Lives in `mm3LowStepShift()`
(`server/src/services/backends/minimax/generate.ts`), applied by the `mm3AutoLowStep`
extension (default ON). It forces **scheduler + shift only** — forwarding the shared
`inferMethod`/`guidanceMode` pickers would silently swap MM3 onto ACE's APG default.

**EAR-VALIDATED at 10 steps (shift 3.2).** 8–29 is interpolation on a curve anchored at
both ends. **Below 8 is extrapolation** — the first-step match is bought with an
ever-larger final leap to clean (0.54 @ 6 steps, 0.86 @ 2), which must break down
somewhere. Symptom to tune against: muddy/smeared = shift too high for the budget;
thin/wide again = too low. Slider min is now 2 steps.

**DSP fallback** (built, measured, NOT shipped): a linear-phase M/S correction —
+8 dB mid low-shelf, −3 dB side with a −8 dB bell at 1.4 kHz — recovers the 30-step
balance to 0.63 dB (mid) / 0.83 dB (side) RMS error. It cannot restore HF *coherence*
(only ~0.2 correlated with the 30-step above 2.5 kHz), so fixing the trajectory beats
correcting after the vocoder. Analysis scripts + A/B renders:
`D:/Ace-Step-Latest/_experiments/_LISTENING/2026-08-21_lowstep-dsp/`.

## MM3 Plank: the AR code cache (built 2026-08-17 — compiled, NOT YET RUN)

Capture the AR stage's output codes on one render, replay them on later ones.
Opt in per request: `get_ar_codes: true` captures; `forced_semantic` +
`forced_acoustic` replay. The blob comes back from `GET /mm3/job?id=<id>&ar=1`
as little-endian i32 — `[n_sem][sem…][n_ac][ac…]` — and is stored as
`<uuid>.mm3plank` beside a `.mm3plank.json` sidecar in `<data>/mm3-planks/`.

**Read this before building on it: replay is NOT a speedup.** The AR loop still
executes every per-frame forward pass, so the planning progress bar still runs
to 100 % on a replay job. Forcing fixes *which tokens come out*, not how much
compute runs. Confirmed independently by MDMAchine on a live run, and the same
root cause closes the other obvious AR lever: `MM3_LM_CFG_ROWS 2`
(`mm3-lm-graph.h`) is compile-time, and AR decode is **weight-streaming-bound**
(17.2 GB of f16 streams regardless of batch), so skipping the uncond row buys
nothing either. Anything that tries to skip AR *compute* hits bandwidth, not
arithmetic.

What it IS for: a fixed semantic bed, so an A/B of two flow-stage
solver/scheduler/guidance settings compares those settings instead of two
different AR samplings. For the speedup, see the AR cache below — that is the
`frame_hiddens` path this section used to describe as unbuilt.

**Saved plans do the bed job better, and the plank still earns its keep.** A
plan (`.mm3hiddens`, below) pins the same bed *and* skips the compute, so prefer
it for A/B work. What the plank has that a plan does not: it is ~4096x smaller,
and it is portable across a model change — forced codes replay through whatever
LM is loaded, where a plan saved under a different quant or adapter is refused.
Archive and share planks; pin plans.

Frame arithmetic: entry 0 is the un-emitted iteration, so `I` codes render
`I-1` frames. `mm3-request.h` derives `max_frames`/`duration` from the array and
`mm3-ar-loop.h` clamps again — do not also set them client-side.

Layers: `mm3-request.h` (parse) → `mm3-job.h` (capture + `?ar=1`) →
`Job::result_ar_codes` (hot-step-server.cpp) → `minimax/plank.ts` (dir resolve,
containment check, blob decode) → `minimax/generate.ts` (save/replay) →
`mm3SaveArCodes` / `mm3PlankPath` capability extensions. The plank reference
reaches the server from the browser, so it goes through
`resolveMm3PlankPath()` — never join it onto a path directly.

## Streaming player: listen while it renders (SHIPPED 2026-08-22 — measured, not yet heard)

Opt in per request with `"stream": true`; UI toggle **Play While Rendering**
(`mm3Stream`, manifest extension, default OFF). A window's PCM is FINAL the
moment it is vocoded and cropped, so it is pushed to a per-job queue and served
as a chunked body of concatenated self-contained WAVs on
`GET /mm3/stream?id=<job>` while the render continues.

**Two levels, and the second is the one that matters.**

*Streaming* moves the vocoder inside the window loop, so a window is emitted as
soon as it is denoised instead of in a second pass at the end. Always available.

*Interleaving* (`stream_interleave`) dispatches windows **while the AR planner
is still planning**, via `MM3ArOptions::on_frame_ready`. Without it the first
window cannot exist until the planner is done — which on a fresh plan is most
of the render, so streaming alone buys almost nothing there. This is the part
that needs the LM and the flow stack **co-resident**, i.e. no staged handover,
which is why it is a per-run VRAM decision made in `mm3-job.h`.

**Measured (60 s clip, 30 steps, RTX 5090, all-q8_0 except cond/voc):**

| | first audio | total | sustained |
|---|---|---|---|
| fresh plan, interleaved | **10.1 s** | 56.8 s | 1.06x realtime, buffer grows +5.0 → +13.5 s |
| AR cache hit (no planner at all) | **3.3 s** | 30.4 s | 1.97x realtime |
| fresh plan, serial fallback | ~AR stage (23–51 s) | same | n/a |

**THROUGHPUT IS THE REAL CONSTRAINT, NOT LATENCY.** Interleaving reorders work;
it does not create any. Playback only survives if audio is produced faster than
it is consumed, and that is a per-configuration fact:

| | fresh plan | verdict |
|---|---|---|
| q8_0 | 1.06x realtime | sustains, buffer grows |
| f16 | **0.74x** | the player IS caught, mid-song, and stalls |

So the quant ladder is not just a speed preference here — it decides whether a
fresh-plan stream plays through. Measure before claiming a config works.

**The `+1` frame rule, and why it is not an off-by-one.** While the planner runs,
F is unknown. Window k needs a RIGHT crop iff a window k+1 exists, iff
`F > cs + win`. Dispatch is gated on `frames >= cs + win + 1`, which makes that
true outright because the planner has not stopped. Gating on `cs + win` instead
is wrong in exactly one case, and it is a case that happens: the planner hits
EOS on that very frame, `F == cs + win`, window k turns out to be the LAST one,
and it has already been emitted with 258 latents (3.0 s) cropped off its end.
One extra frame — 40 ms of audio — buys a crop that can never be wrong. After
the planner returns, the settled plan is re-derived and every already-dispatched
window is checked against it rather than trusted.

**The transport is STORM's, the scheduler is NOT.** The byte stream is the same
shape `POST /api/generate/storm/stream` already emits, so `extractWav` is shared
(moved to `ui/src/utils/wavStream.ts`). But `useStreamAudio` CROSSFADES between
independent generations; MM3 windows are consecutive spans of ONE signal and
must HARD-SPLICE, so `useMm3StreamAudio.ts` is a separate hook. Three things it
does that a copy of the STORM path would get wrong: the AudioContext is built at
the **rate read from the first WAV header** (MM3 is 44.1 kHz, `useStreamAudio`
hardcodes 48 k, and a resampling context destroys the exact frame counts the
splice depends on); scheduling accumulates **frames, not `ab.duration`**; and on
underrun it **stalls rather than drops** — the whole timeline shifts forward so
every later window plays in full, because a pause is better than a hole in a
track someone is deciding whether to keep.

**Proven bit-identical, twice, and that is the bar.** Re-runnable against a live
app, and worth running after ANY change to the window loop, the crop arithmetic
or the WAV encoder:

```
node server/scripts/check-mm3-stream.mjs           # engine: the two byte-identity claims + the 409s
node server/scripts/check-mm3-stream-node.mjs      # Node tier: proxy, /status flag, cancel mid-stream
node server/scripts/check-mm3-stream-latency.mjs   # first audio + sustained rate, fresh vs cached
```

`stream:false` and `stream:true` at the same seed produce a **byte-identical
WAV** — and since the streamed run is the INTERLEAVED one, that also proves the
reordering (condition and denoise window k while the planner works on later
frames) is numerically neutral, which is the only way to ship a control-flow
change this invasive. The concatenated PCM of every streamed chunk is likewise
**byte-identical to the saved WAV's PCM**. A seam would be an offset bug, and it
is measurable, so it is measured rather than listened for.

`process_window()` in mm3-pipeline.h is the ONLY copy of the per-window body;
the planner hook and the post-planner sweep both call it, so they cannot drift.
`mm3_window_crop_lr()` / `mm3_window_crop()` are one piece of arithmetic in two
spellings (the dispatcher knows "is there another window after this one?" before
it knows how many there are), and `mm3_clamp_sample()` likewise — those
identities are what make the streamed bytes the saved bytes.

**Traps.**
- **Installing a sink CHANGES WHEN THE VOCODER RUNS** — inline per window
  instead of one pass after every window is denoised. That is why it is opt-in:
  a render with no sink keeps today's exact stage order and VRAM profile.
- **Interleaving and `after_ar` are mutually exclusive.** That hook means "the
  LM is done, swap residency", and interleaving has already run stage 2 against
  a live LM by the time the AR returns. `mm3-job.h` only asks for interleaving
  when it is not staging; mm3-pipeline.h also disables interleaving if a hook is
  somehow set, so the conflict fails SAFE (serial).
- **An AR cache hit never interleaves**, and does not need to: stage 1 does not
  run, so the serial sweep starts emitting immediately (this is the plan's
  trap 1 — window dispatch must not assume a live AR loop).
- **`frame_hiddens` is read with a fresh `.data()` every window.** The planner is
  still appending to it between dispatches. `mm3-ar-loop.h` reserves the whole
  `max_frames` block up front so it does not in fact reallocate — but a pointer
  captured before the planner ran would be a dangling read the day that reserve
  changes, and the failure mode is silent garbage audio.
- **`ar_ms` had to stop double-counting.** Interleaving nests stage 2 inside the
  AR call, so its wall time is no longer the planner's cost. Before the fix a
  60.2 s render reported ar 50.3 s + flow 27.4 s. `dispatch_ms` is measured
  around each nested window and subtracted.
- **Progress needed its own stage.** With the planner and the flow stage taking
  turns several times a second, the `ar` (10–35) and `flow` (40–85) bands made
  the bar oscillate for the whole render. Interleaved runs report a single
  `"stream"` stage carrying windows-out and frames-planned; `minimaxStageText`
  maps it to one monotonic 10–90 axis.
- **The engine's `streaming` echo is the authority, not the request**, and
  `stream_interleaved` on `GET /mm3/job` is the authority on which of the two
  levels you got. Declining co-residency is NOT a failure — the render still
  streams, just later. The UI says why rather than looking broken.
- **`GET /mm3/stream` must never take `g_mm3_mutex`** — the generation it is
  streaming holds that for its whole run (the same reason `/mm3/props` blocks
  and `/mm3/job` does not). Its queue has its own mutex, and the counters
  `/mm3/job` reports are **atomics**, because that handler reads them while
  already holding `MM3JobState::mtx` — the lock the worker takes on every Euler
  step.
- **One reader, consumed on read.** Chunks are popped as they are written, so a
  second reader would silently steal half the song; a reconnect cannot resume.
  409, not a partial stream.
- **`DataSink::write(d, 0)` means end-of-stream** in cpp-httplib. A window that
  crops to zero samples is skipped rather than emitted as a header-only WAV.
- **The queue is capped at 128 MB unread** (`MM3_STREAM_MAX_UNREAD_BYTES`),
  sized to clear a 360 s 16-bit render so "submit, then press Listen at the
  end" still works — verified: a render nobody ever attached to is
  byte-identical to a non-streamed one, and a reader attaching AFTER it
  finished still gets every window. 32-bit float output is 4x and CAN hit the
  cap; the stream is then dropped with a log line and the render continues
  untouched. (A finished job 409s only once its chunks have been drained —
  "already finished" is about an empty queue, not about the job's status.)

Layers: `mm3-ar-loop.h` (`on_frame_ready`) -> `mm3-pipeline.h` (`MM3ChunkCb`,
`process_window`, the dispatch predicate, `mm3_window_crop_lr`) ->
`mm3-request.h` (`stream`) -> `mm3-job.h` (`MM3StreamQueue`, the co-residency
check, `GET /mm3/stream`) -> `minimax/client.ts`
(`stream`/`streaming`/`stream_interleaved`/`mm3StreamUrl`) ->
`minimax/generate.ts` (mapping, `job.mm3Streaming`/`mm3Interleaved`, the
`stream` stage text) -> `minimax/index.ts` (manifest extension) ->
`routes/generate.ts` (`GET /api/generate/mm3/stream/:id` pipe + `mm3_streaming`
/ `mm3_interleaved` on `/status`) -> `useMm3StreamAudio.ts` +
`Mm3StreamPlayer.tsx` -> `CreatePanel.tsx`.

## MM3 AR cache: the speedup the Plank is not (built + measured 2026-08-21)

`engine/src/minimax/mm3-ar-cache.h`. One slot holding the previous render's
`frame_hiddens`, so a render that changes only FLOW-side settings skips stage 1
outright. Opt in per request with `reuse_ar: true`; UI toggle **Reuse Planner
Output** (`mm3ReuseAr`, default ON).

**Why this works where the plank does not:** the flow DiT never sees the codes.
Its real input is the `[F, 8, 4096]` hidden block the condition encoder eats
(`mm3-pipeline.h`), so pinning codes still re-runs every AR forward pass to
regenerate them. Caching the hiddens skips the work.

**Measured on a 12 s clip, q8_0, RTX 5090:**

| | AR | flow | total | warm |
|---|---|---|---|---|
| miss (12 steps) | 5364 ms | 1844 ms | 8745 ms | 4140 ms |
| hit (24 steps, cfg 2.2) | **0 ms** | 3428 ms | 3782 ms | **1159 ms** |

The hit did *twice* the flow work in under half the time. `warm_ms` drops
because a hit never loads the LM at all — `mm3_load_parts(lm=false,
depth=false, rest=true)`, and `staged` goes false so there is no mid-run
handover either. **Proven bit-identical:** same request twice (miss then hit)
-> byte-identical WAV *and* byte-identical plank blob.

**The key is the whole correctness argument** (`mm3_ar_cache_key`, mm3-job.h) —
same discipline as ACE's `computeLmCacheKey`, same failure mode if something is
missing (a knob that looks dead). In it: prompt (caption+lyrics+instrumental),
`max_frames`, resolved AR seed, effective `get_lrc`, LM adapter path+mtime+mode
+all six dials, forced-code digests, and the LM/depth file path+bytes+file_type.
Deliberately NOT in it: steps, cfg_flow, sampler plugins and their params,
flow_shift, forced_noise, wav_bits, cond/dit/voc model choices — the knobs the
cache exists to let you tweak. Bump the `v=` field if the AR path changes shape.

**`ar_seed` (new wire field, and the reason a seed change need not re-plan).**
MM3 natively drives both the plan and the flow noise from one seed, so changing
the seed always re-plans. `ar_seed` splits them the way ACE's `lm_seed` does:
set it (UI: **Planner Seed**, blank = tied) to pin the plan while the main seed
rerolls the flow noise. Verified: seed 9999 + ar_seed 4242 hit the slot filled
by seed 4242 and produced different audio.

**Traps.**
- **A random seed can never hit.** `randomSeed` draws fresh every render, so the
  plan is fresh every render. generate.ts pushes a note saying exactly that
  rather than letting the toggle look dead.
- **RAM, not VRAM**: 128 KB per frame = ~3 MB per second of audio, so ~600 MB
  for a 200 s song, held in the ENGINE's host memory. Reported by `/mm3/props`
  -> `ar_cache: {present, frames, mb, hits}`. Dropped by `POST /mm3/unload`, by a
  model-role change, and by the next miss (before the new run, not after).
- **Deliberately NOT hooked into `mm3_unload()`** — that fires after every
  generation when keep-loaded is off, which would drop the slot before it could
  ever hit.
- **The manifest default is mirrored as `!== false` in generate.ts.** The UI only
  writes a backend-declared param once the user touches it, so for a
  `default: true` toggle an absent value must resolve to ON or the control lies.
  Applies to any future default-true extension param.
- Stage-1 byproducts (codes, LRC, EOS flag) are cached alongside the hiddens and
  restored onto the result, so plank capture and `x-lrc-text` behave identically
  on a hit — verified.

Layers: `mm3-ar-cache.h` (slot) -> `mm3-job.h` (key + lookup + fill) ->
`mm3-pipeline.h` (`cached_hiddens` borrowed, `HID` pointer) -> `client.ts`
(`reuse_ar`/`ar_seed`/`ar_cached`) -> `generate.ts` (mapping + notes) ->
`index.ts` (manifest params).

## Saved plans: the AR cache on disk (built 2026-08-27 — compiles, NOT YET RUN)

The slot above, serialised to a `.mm3hiddens` file, so a pinned plan survives a
restart. Adapted from an MDMAchine drop; the engine side is his idea, the
identity checking and the priming route are not. Opt in per request with
`save_frame_hiddens` + `frame_hiddens_save_path` to write, or
`forced_frame_hiddens_file` to replay. UI: **Save Plan To Disk** /
**Plan Name** / **Replay Saved Plan** (`mm3SaveHiddens`, `mm3HiddensName`,
`mm3HiddensPath`), all in the LM cluster beside the plank controls.

**It loads INTO the slot, not around it.** This is the whole design and the one
thing to preserve. Every decision a hit makes — `mm3_load_parts(lm=!ar_hit,
depth=!ar_hit)`, `staged`, `stage2_only`, the adapter block, the byproduct
restore — reads the local `ar_hit`, not `req.gen.cached_hiddens`. A replay that
set the pointer directly (the obvious implementation, and the one in the
original drop) still streams 17.2 GB of LM weights it never calls, still does
the staged handover, and still hands back an empty LRC and no plank codes. So
the file primes `g_mm3_ar_cache` and `ar_hit` is then computed as
`(req.reuse_ar || ar_from_file) && …`. Nothing else needed changing.

**Refusal vs warning, and why the file carries the key.** A blob made under a
different LM quant has *exactly the right shape* and is numerically meaningless,
so a dimension check waves it through and the render is silently wrong. The file
therefore stores two keys (`mm3-hiddens-file.h`):

| key | holds | mismatch |
|---|---|---|
| `mm3_ar_model_key` | LM + depth path/bytes/quant, adapter identity + all six dials, `[cb, emb]` | **REFUSED** — plans fresh |
| `mm3_ar_cache_key` | the above + prompt, `max_frames`, `ar_seed`, LRC flag | **NOTE** — replays anyway |

The soft half is deliberately soft: the caption only reaches the flow DiT
*through* these hiddens, so replaying against an edited caption is legitimate —
it just means the caption and duration controls no longer describe the output.
`mm3_ar_cache_key` is `v=2` for the split; the model fields moved into
`mm3_ar_key_add_models()` and now trail the forced codes.

**Traps.**
- **The save reads the SLOT, not `r.ar`** — the block was `std::move`d out of
  `r.ar` by the fill. So `save_frame_hiddens` also has to *trigger* the fill
  (`req.reuse_ar || req.save_frame_hiddens`), or the save either writes nothing
  or writes whatever an earlier render left in the slot, under the name the user
  just typed. That second one is the failure mode worth remembering.
- **Written atomically** (`.tmp` + rename), because an interrupted 600 MB write
  otherwise leaves a file that looks fine until the shape check.
- **Passed to the writer by pointer.** Copying the slot to hand it to a function
  is another 600 MB of host RAM for a 200 s song.
- **Ensembles never save or replay** — the slot holds one plan, so `takes > 1`
  gets a warning rather than a silently dead picker.
- The blob never crosses the Node/engine wire in either direction; only paths
  do. Server-side references go through `resolveMm3HiddensPath()`, same
  containment rule as the plank.

**Size is the whole reason the plank survives.** 128 KB/frame vs 32 B/frame is
4096x: ~600 MB per 200 s song against ~150 KB. A plan is for pinning a bed you
are about to iterate flow settings against; a plank is for anything you want to
keep, share, or inspect, and it is portable across a quant or adapter change
(forced codes replay *through* whatever LM is loaded) where a plan is refused.
A "library of past songs as plans" does not fit on a disk — 100 songs is 60 GB.

Layers: `mm3-hiddens-file.h` (format + I/O) -> `mm3-job.h` (prime + save) ->
`mm3-request.h` (3 fields) -> `hiddens.ts` (dir, containment, list, delete) ->
`generate.ts` (mapping + sidecar) -> `index.ts` (manifest) ->
`routes/backends.ts` (`/api/mm3/plans`, `/plan-meta`, DELETE) ->
`BackendExtensionControls.tsx` (`visible_when`).

## Performance budget (RTX 5090, f16, 12 s clip ≈ 12.4 s wall ≈ 1.0× realtime)

AR 25.5 ms/frame (LM step 15.3 — bandwidth-bound; depth 9.2, 37 % of AR for 7 % of params) ·
flow 2.2 s/window · vocoder 85 ms/window. Speed levers in order: **q8_0 LM** (~2× LM step,
measured 16.6→8.8 ms/step; re-validate by ear — quant can flip borderline codes), **NVFP4
depth** (9.4→4.9 ms/frame, below), TRT much later. Known quality morsel: our synth on
identical codes measures ~18 % lower spectral flatness than the reference (unresolved, minor).

**Current q8_0 steady state (2026-08-21, post head-slice):** LM 8.3 ms/step · depth q8_0
6.4–6.9 ms/frame · flow q8_0 31–33 ms/forward (quant bought only ~5 % — the flow DiT is
compute-bound, ~94 TFLOPS effective). The LM head now computes only the contiguous
EOS+semantic row span (mm3_lm_head_slice_span; opt-in per graph — mm3-lm-probe keeps the full
head), proven bit-identical, −8 %/step. LM streams ~8.5 GB in 8.3 ms ≈ 57 % of a 5090's peak —
what remains is kernel-level (mmvq at 2 columns) or KV-cache quantization (~5 % late-song);
the AR stage is close to its architectural floor. Remaining flow levers, in value order:
fewer steps via the sampler plugins (linear; multistep deterministic solvers first, listen at
window seams), a native CFG-interval knob (skip the uncond forward outside a mid-sigma band),
TeaCache-style velocity reuse, TRT much later.

**CUDA graphs: already active — do not build a capture project (measured 2026-08-21).**
The vendored ggml-cuda has per-graph keyed capture (keyed on the split cgraph's nodes[0],
2-call warmup, 10 s idle eviction) and it engages for every MM3 graph unprompted. A/B vs
`GGML_CUDA_DISABLE_GRAPHS=1`: LM decode 9.1 vs 12.9 ms/step (−29 % with graphs), depth 9.4 vs
11.1 ms/frame (−14 %). The old "depth is launch-bound → kernel fusion / CUDA graphs" diagnosis
is therefore STALE: with graphs on, depth is **matvec-efficiency-bound** (~1 GB f16 streamed
per codebook step at ~half of peak on 4–16-column matmuls), so the lever is the quant ladder:
depth f16 9.4 → q8_0 6.4 → **NVFP4 4.9 ms/frame** (Blackwell-native kernels; Q4_K_M is NO
better than q8_0 — K-quant dequant cost eats the bandwidth win in mmv). Zero code, per-role
picker. Acoustic codes = timbre: ear-check on multiple seeds before adopting. Diagnostics that
found all this and stay available: `MM3_DEPTH_PROF=1` (phase timing per frame, mm3-depth-graph.h)
and `GGML_CUDA_GRAPH_LOG=1` (per-compute capture decisions, engine/patches/cudagraph-log.patch).

**select-model trap: a role OMITTED from the body means auto (= best-first = f16), not "keep".**
Raw-API partial bodies like `{"depth":"q8_0"}` silently reset the LM to f16 — measured as a
mystery 2× LM slowdown that looked exactly like CUDA-graph thrash until the load line
(16,374 MB) gave it away. The Node layer now merges missing roles from persisted settings
(index.ts selectModel) and the UI always sends every role; when poking the ENGINE directly,
always send the full selection.

**CLOSED NEGATIVE (2026-08-21): batching the flow DiT's cond+uncond CFG passes.** Full batch-2
graph built and A/B'd (bcdcab0): 81 ms/step vs 66 two-pass at L=689 — **22 % SLOWER**, corr
0.999991. The forward is COMPUTE-bound (~145 GB/s effective, nowhere near bandwidth), so the
ceiling was ~3 ms of weight re-streaming and ggml's batched matmul/flash dispatch costs more.
The code stays in mm3-dit-graph.h behind `MM3_DIT_CFG_BATCH=1` — re-measure on a new ggml
before believing it, don't rebuild it from scratch. Real LM-side numbers from the same day
(254 s render, f16): LM 23.4 ms/step *with* LRC capture on, 16.6 without (the +41 % is the
all-manual-attention cost, live in every get_lrc render); runtime LM adapter r256 = +6.6
ms/step (+28 %, not the hoped +9 %: 252 modules ≈ +1000 nodes on a 2545-node decode graph —
launch overhead, not just the +8 % streaming).
