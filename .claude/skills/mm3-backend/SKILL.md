---
name: mm3-backend
description: Maps HOT-Step's native MiniMax-Music3 backend — engine port modules, endpoints, server/UI integration, parity/fixture infrastructure, and the hard-won trap list. Use when working on anything MM3 — engine/src/minimax/, backends/minimax/, /mm3/* endpoints, the backend toggle/capability gating, MM3 model files or Model Manager entries, debugging MM3 generations, MM3 performance work, or extending MM3 features (covers, training, Lyric Studio).
---

# MiniMax-Music3 backend

Native C++/GGML port of [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3),
built 2026-08-13 (release day) as HOT-Step's second generation backend behind an N-backend
abstraction. **Status: rudimentary text2music only** (caption + lyrics + duration + seed);
no covers/repaint/stems/adapters/training. Output = raw 44.1 kHz stereo WAV (app norm is 48 k —
post-chain steps that hardcode 48 k are skipped for MM3).

Deep docs (local, gitignored): `docs/plans/multi-backend-architecture.md` (architecture plan,
day-0 findings, op inventory) and `docs/plans/mm3-gguf-layout.md` (GGUF contract + loader
addendum). Caption format: the **mm3-captioning** skill.

## Model + pipeline (25 fps frames; every module parity-proven vs the diffusers reference)

```
caption+lyrics → Qwen2 BPE → Global LM 8.59B (Qwen3 arch, semantic codes @ ids 151675–168058,
  EOS 151670, AR CFG 1.5 as persistent 2-row batch) → per frame: RVQ depth decoder 0.6B
  (7 acoustic codebooks) → frame_hiddens [F,8,4096] → per 200-frame window (hop 100):
  condition encoder 25M (×3.4453125 nearest resample) → flow DiT 2.4B (30 Euler steps,
  CFG 1.7, zeros-cond uncond as separate pass) → vocoder 54M (DAC-style, ×512 → 44.1 kHz)
  → overlap-crop stitch
```

## File map

| Piece | Where |
|---|---|
| Engine modules | `engine/src/minimax/` — `mm3-model.h` (loader/residency), `mm3-tokenizer.h`, `mm3-lm-graph.h`, `mm3-ar-loop.h`, `mm3-sample.h`, `mm3-depth-graph.h`, `mm3-cond-graph.h`, `mm3-dit-graph.h`, `mm3-vocoder-graph.h`, `mm3-pipeline.h` (e2e + chunking), `mm3-request.h` (prompt assembly/hygiene), `mm3-job.h` (job queue + VRAM arbitration), `mm3-server.h` (endpoints) |
| Hooks | one include in `engine/tools/hot-step-server.cpp` (+ `mm3_register_routes`/`mm3_register_job_routes` call sites); checked by `engine/verify-hooks.ps1` (hooks 4/4b/4c) |
| Server backend | `server/src/services/backends/` — `types.ts` (EngineBackend + capability manifest), `registry.ts`, `ace/`, `minimax/{client,index,generate}.ts`; routes `server/src/routes/backends.ts`; generation branch at top of `runGeneration` in `routes/generate.ts` |
| UI | `stores/backendStore.ts`, `hooks/useCapabilities.ts`, `global-bar/BackendToggle.tsx` (hidden until ≥2 backends), `shared/BackendCapabilityGate.tsx` (studio guards), gating in `GlobalParamBar.tsx` |
| Models | **5-way split since 2026-08-14** (ported from ServeurpersoCom/minimaxmusic.cpp): `models/mm3/mm3-{lm,depth,cond,dit,voc}-<quant>.gguf` (archs `qwen3` / `mm3-{depth,cond,dit,voc}`), legacy `mm3-synth-*` bundles still load (fill any role; split file wins per quant token). Per-role quant mixing (LM Q8_0 + DiT Q4_K_M is the headline combo); a DiT/adapter swap reloads only cond+dit+voc — LM stays warm. cond/voc are **never quantised** (f16 only). Hosted `scragnog/MiniMax-Music3-GGUF`; registry role `mm3`, packs rebuilt on split components in `server/src/data/model-registry.json` |
| Converter | `engine/tools/convert-mm3.py` (safetensors→GGUF bundle; folds weight-norm; refuses pruned/int8_convrot) then `engine/tools/split-mm3.py` (byte-exact bundle→5-way split; idempotent; cond/voc only from native bundles) |
| Fixtures / parity | `D:\Ace-Step-Latest\mm3-weights\fixtures\` (manifest.json + raw f32 dumps + reference WAVs), seed-spread study in `..\seed-spread-2026-08-13\`; venvs: `.venv-convert` (numpy/gguf), `.venv-ref` (patched diffusers @ dafe3733 — `patch_venv.py --restore`; `capture_fixtures.py --replay` rebuilds dumps without rerunning the model) |

## Engine endpoints (:8085 via app, standalone tests on :8086)

`GET /mm3/props` (files/config/loaded/limits — **blocks while an MM3 generation runs**; always
call with ~2.5 s timeout and keep last-known-good), `POST /mm3/warm` / `POST /mm3/unload`
(idempotent; unload frees weights+KV), `POST /mm3/synth` (production, rides the same FIFO GPU
worker as ACE `/synth`; standard `/job?id=` progress/cancel/result; request contract documented
in `mm3-request.h`/`mm3-job.h`), `GET /mm3/job?id=` (MM3-vocabulary progress, never blocks;
`&ar=1` returns the Plank code blob — see below),
`POST /mm3/tokenize-check` (cold-capable; 5000-token limit), plus deprecated bring-up endpoints
(`/mm3/voc-decode`, `/mm3/dit-forward`, `/mm3/flow-sample`, `/mm3/depth-frame`,
`/mm3/cond-encode`, `/mm3/lm-plan`, `/mm3/synth-e2e`) kept for parity work — they run GPU work
on httplib threads; never build production paths on them.

**Standalone launch gotcha:** ace-server exits `0xC0000135` with zero output unless
`engine/trtllm-libs` + `engine/deps/tensorrt_libs` are prepended to PATH (aceEngineProcess.ts
does this; `engine/server.cmd` does not).

**Caption echo (added 2026-08-21).** `POST /mm3/synth` prints the caption to stderr at job
creation, so it reaches the terminal, `ace_engine.log` and the in-app Terminal — the MM3
analogue of ACE's `[LM-Phase2] CoT[0]` dump, which MM3 had no equivalent of:

```
[MM3-Job] <id> created - 63 prompt tokens, ...
[MM3-Job] <id> caption (149 bytes in, 143 cleaned), lyrics 46 bytes:
<the cleaned caption>
```

It prints the **cleaned** caption (post `mm3_clean_caption`), not the raw body, because the two
differ exactly where a markdown-emitting tool pasted `**bold**` headings or `- ` bullets in —
the drift you would otherwise only hear. `MM3_LOG_PROMPT=1` swaps it for the whole assembled
template (`<|im_start|><|caption_start|>…<|lyrics_start|>[start]…<|audio_start|>`). The
Node-side `[Generate] … caption=N chars` line is the send-side half; a mismatch between the two
counts localises a drop to the wire rather than the UI.

## The trap list (each cost real debugging — do not relearn)

1. **ComfyUI's wrapper NEGATES the DiT output; the diffusers reference (and our port) does not.**
   `mm3.dit.output_negated` in the GGUF records Comfy's behavior. Do not "fix" the sign.
2. **`tokenizer.ggml.pre = qwen2` is misleading** — the reference uses the *slow* Qwen2Tokenizer
   (single-digit regex = classic GPT-2 pre-tokenization, which `bpe.h` implements). Matching the
   KV's llama.cpp meaning ({1,3} digit grouping) breaks token parity.
3. **Scheduler sigmas must replicate float32 `linspace(1, 1/30, 30)` rounding** — deriving
   `i/steps` is wrong in the 7th digit and it matters.
4. **AR iteration 0 is fed back but never emitted** (emitted frame j = iteration j+1). A
   one-frame indexing slip degrades conditioning parity 49×.
5. **The semantic code embeds via the LM's `token_embd`, not `depth.audio_embd`.**
6. Caption hygiene: **`splitlines()` for caption, `split("\n")` for lyrics** — mixing them leaks
   a trailing `\n` into the template. Empty lyrics → we substitute `[instrumental]` (the
   reference *rejects* empty; this substitution is a HOT-Step decision).
7. Condition resample is **plain `nearest`, not `nearest-exact`** (differs on 199/689 positions).
8. **Never use `std::normal_distribution`** for reproducible noise (stdlib-dependent bytes) —
   `mm3_fill_noise` uses splitmix64 + Box-Muller.
9. GGUFs live in the **`models/mm3/` subdir** deliberately: the ACE registry scan globs only the
   models root (unknown-arch warnings + 17 GB header reparse per boot if placed there).
10. **Single-seed spectral/genre judgments are meaningless** — the reference's own 11-seed spread
    spans 272× in flatness and wanders off-genre with minimal captions. Structured 3-section
    captions (mm3-captioning skill) are the adherence lever. Compare distributions, not takes.
11. VRAM: f16 stack ≈ 22.5 GB + KV (288 kB/position) + ~3 GB compute headroom. Engine-side
    arbitration evicts idle ACE modules before MM3 warm; Node-side `releaseVram()` handles the
    reverse on backend switch and before ACE gens. ~600 MB stays in the CUDA pool after unload
    (returns on process exit — not a leak).
12. The LM GGUF is **not interchangeable with stock Qwen3-8B GGUFs** (extended 200 k vocab,
    untied head) and llama.cpp alone cannot run music generation.
13. **`read_wav_buf` returns INTERLEAVED `[T,2]`; the DAV encoder wants PLANAR `[L:T][R:T]`.**
    Use `audio_io_read_wav_buf` (audio-io.h), which de-interleaves — never the raw reader.
    `mm3-preprocess` sliced the raw reader's output as `{p, p+T}` and made "left" the FIRST HALF
    of the song with L/R alternating. Since L≈R, that duplicates every sample: an exact **2×
    time stretch, one octave down**. Every cached target was the song in slow motion and five
    LoRA runs learned to generate slow motion (2026-08-15, fixed 82b2852).
14. **VERIFY PREPROCESSING BY DECODING A TARGET AND LISTENING — metadata cannot catch this
    class of bug.** #13 survived a full day because `T` is PER-CHANNEL frames, so
    `latent_frames / duration` stayed at exactly 86.1328 Hz and *every* arithmetic check on the
    manifest passed. The DAV parity gate passed too (it is fed `encode_ref.py`'s planar dump).
    The manifest was written by the same buggy code being checked, so it corroborated itself.
    One listen to a decoded target found it. `POST /mm3/voc-decode?frames=N` with raw f32
    `[128,N]` returns a WAV — there is no excuse not to.
    Objective version of the same gate: encode a **440 Hz sine** and measure what comes back
    (was 220.0 Hz, i.e. ratio 0.5000; correct is 440.0 Hz / 1.0000). A pure tone cannot be
    argued with, and it brackets which stage is at fault.
15. **Rob's ear beat every metric, twice.** He called "slow motion, too deep" on the first
    adapter and again on the third; both times it was explained away as regression-to-the-mean
    (which produces a genuinely similar description) and five runs of hyperparameter tuning
    followed on corrupt data. When the user reports a *physical* symptom — speed, pitch,
    duration — treat it as literal and test it literally before reaching for a statistical
    explanation.
16. **Gate every trained adapter on `||delta||/||W||` BEFORE any ear test.** Healthy LoRA
    merges move weights 1–5% Frobenius; at lr 5e-4 × 8k steps ours hit median 17% (max 34%)
    and at scale 1.0 that is a damaged model, not a strong style — jumbled inside a single
    689-latent window, invariant to rank/crop/CFG (AdamW makes total movement ≈ lr×steps
    regardless of rank, which is why every knob "did nothing"). Measure against the ComfyUI
    f16 checkpoint (`mm3-weights/comfy/diffusion_models/`), whose keys match the export
    directly; target median ≤5%. SimpleTuner's reference recipe is lr 5e-5.
17. **Training crops must not straddle conditioning-rollout seams.** `mm3-condition` builds
    the cache from independent 60 s segments; a crop across a seam pairs continuous audio
    with conditioning that jumps to an unrelated rollout mid-window — teaching "conditioning
    lies, smooth over it" (mean-collapse pressure). The seams were parsed and never consulted
    for a week (~13% of crops at 689, 27% at 1378); fixed 5117281 with reject-and-retry.
18. **Filter groups at EXPORT, never constrain them at TRAINING.** Measured (runs 09/10,
    matched ~2.6% delta, same groups): full-set training + MLPV surgery = coherent with
    clear lyrics; `--target mlpv` trained-from-scratch = intrusions and jumble, with LESS
    style at matched delta. Gradient denied its natural pathway (q,k routing) emulates it
    destructively through the remaining groups, so "safe-group" deltas from a constrained
    run carry structure-entangled content the base attention cannot support. The winning
    recipe: train ALL sites at modest delta, then zero q,k rows + proj heads in the export
    (`groupfilter.py` pattern — q,k rows are B[0:4096] of the fused qkv; ablation-proven:
    q,k = structure poison, proj_in/out = seed-dependent fuzz, MLP+V+out = timbre).
18. **An MM3-only install must not kill the server at boot.** The startup gates in
    `hot-step-server.cpp` (`registry_scan` empty → exit 1; partial ACE synth without LM →
    exit 1) predate MM3 and knew nothing about it: a user with only `models/mm3/*.gguf` got a
    dead engine → empty model dropdowns for BOTH backends + the MM3 "weights missing" CTA,
    while the Model Manager (Node disk scan, checks subdirs) said everything was installed
    (GitHub issue #118). Both gates now fall through when `mm3_weights_present()`
    (mm3-model.h — filename-only probe of `<models>` + `<models>/mm3`) is true; ACE handlers
    already degrade per-request with an empty registry. Any future boot-time hard-exit must
    ask "can MM3 still serve?" first.

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

## Sampler plugins: shared with ACE (built 2026-08-16 — NOT YET COMPILED OR HEARD)

The same Lua solver / scheduler / guidance plugins that drive the ACE DiT now
drive MM3's flow DiT. No plugin was modified and no plugin API was widened —
the plugin layer never had an ACE dependency (every `lua_call_*` entry point
takes raw `float *` + counts + a param map); what was ACE-specific was the
sampler, not the plugins. Bridge: **`engine/src/minimax/mm3-plugins.h`**.

Two conventions differ, and both mappings are exact:

1. **Time runs the other way.** ACE `t` descends 1→0 with `xt -= vt*dt`; MM3
   `sigma` ascends 0→1 with `x += dsigma*v`. Substituting `sigma = 1-t` and
   `v_ace = -v_mm3` makes them the same expression — *including the terminal
   step*, where MM3's last increment `(1 - sigma[steps-1])*v` is character-for-
   character ACE's engine-owned `x0 = xt - t_curr*vt`. MM3's steps+1 sigma array
   IS ACE's "N timesteps + engine-owned final step".
2. **The latents are transposed.** ACE is time-major `[T][Oc]`, MM3 is
   channel-major `[C=128][L]`. This is NOT cosmetic: `apg_forward` normalises
   per channel over time and indexes `[t*Oc+c]` to do it, so a channel-major
   buffer would be grouped along neither axis. The bridge transposes into the
   ACE view before any plugin sees a buffer. 4 transposes/step of 88k floats
   against two 2.4B forwards — free.

**Opt-in, and that is load-bearing.** Empty selection == the native arithmetic,
expression for expression, so the parity fixtures still cover the default path.
This holds even for `infer_method=euler` (Lua computes in double and rounds on
store; the native loop is float throughout — they can differ in the last ulp).
Server-side the picks only travel when `params.samplerPluginsEnabled` is on,
because ACE defaults solver/guidance to euler + apg — forwarding blindly would
move every MM3 render off the native loop silently, and `guidance_mode: "apg"`
is a genuinely different algorithm from MM3's plain CFG.

**The picks are per backend as of 07b2a5c** (they were one shared global before,
so changing the solver in MM3 mode rewrote ACE's). `BACKEND_SCOPED_FIELDS` in
`ui/src/stores/globalParamsStore.ts` lists them: `inferMethod`, `scheduler`,
`guidanceMode`, `pluginParams`, `backendParams`. Storage key is the bare `hs-*`
for `ace` and `hs-*@<backendId>` for anything else; a backend's first visit
seeds from the bare key, and `useBackendStore.subscribe` re-hydrates the store
on every switch. Anything writing one of those keys **outside the setters**
(`utils/paramProfiles.ts`, `sendAuditionToCustomGen.ts`) must route through the
exported `scopedKey()` — a bare-key write lands where the live store will never
read it back, and the next backend switch silently undoes it.

Not supported: **owns_loop solvers** (11 of them, mostly MDMAchine's — they'd
bypass the per-step overlap blend and break every window seam; engine warns and
falls back, and `SamplerPluginControls.tsx` filters them out of the dropdown),
**postprocess plugins** (they replace ACE's tiled VAE decode), **`composite:`
schedulers** (built by `sampler-schedule.h` from ACE's own globals), and the
**legacy sideband params** (`stork_substeps` etc., read from `g_hotstep_params`).

State resets **per window**, not per song: one 200-frame window == one ACE
"generation", so `MM3PluginRun::begin_window()` resets APG momentum and solver
history. Momentum leaking across a seam would smear one window's guidance
history onto the next one's first step, right where the overlap machinery is
trying to hide the join. Stochastic solvers are the open risk — they inject
noise that fights the overlap blend; test the seams by ear.

Wire fields (mm3-request.h, all optional): `infer_method`, `scheduler`,
`guidance_mode`, `flow_shift` (default 1.0 = MM3's own hardcoded shift; only
consulted with a scheduler plugin), `apg_norm_threshold`, `plugin_params`.
`POST /mm3/synth` echoes `sampler_plugins` back **only when something was
selected** — that is how you tell "I picked a solver" from "a solver ran".
Capability: `features.samplerPlugins` (distinct from `features.plugins`, which
selects WHICH Generation dropdown renders — MM3 needs the generic one for its
steps/cfg knobs, so it is `plugins: false, samplerPlugins: true`).

**Status: written, TypeScript type-checks clean on both tiers, C++ NOT compiled
and no audio generated.** Validate in this order: (1) `dev-rebuild.bat`;
(2) a generation with no plugins selected, confirming it is bit-identical to a
pre-change render on the same seed — that is the parity guarantee, and it is
the only claim here that can be checked without ears; (3) `linear` scheduler
alone, which should be near-identical to native (MM3's schedule IS shift=1
linear, differing only in float32 linspace rounding); (4) a solver, listening
specifically at window seams on a >8 s clip.

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
different AR samplings. It is also PORTABLE and survives a restart, which the
AR cache below does not. For the speedup, see the AR cache — that is the
`frame_hiddens` path this section used to describe as unbuilt.

Frame arithmetic: entry 0 is the un-emitted iteration, so `I` codes render
`I-1` frames. `mm3-request.h` derives `max_frames`/`duration` from the array and
`mm3-ar-loop.h` clamps again — do not also set them client-side.

Layers: `mm3-request.h` (parse) → `mm3-job.h` (capture + `?ar=1`) →
`Job::result_ar_codes` (hot-step-server.cpp) → `minimax/plank.ts` (dir resolve,
containment check, blob decode) → `minimax/generate.ts` (save/replay) →
`mm3SaveArCodes` / `mm3PlankPath` capability extensions. The plank reference
reaches the server from the browser, so it goes through
`resolveMm3PlankPath()` — never join it onto a path directly.

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

## Validation bar for MM3 changes

Forced-replay parity against the fixtures (never sampled-path comparisons — RNG can't match
torch). Established floors: per-module ≥ 0.999 corr vs the bf16 dumps (the dumps' own floor,
~1.6e-2 relRMSE) or ≥ 0.9999 vs an fp32 CPU rerun of the reference module. Full-clip replay:
0.9988. If a change should be bit-neutral, prove it with the deterministic seeds.

## Lyric timestamps: use Whisper, not attention (measured 2026-08-14)

ACE derives LRC from its **DiT's lyric cross-attention**. MM3 has no analogue:
its flow DiT has no cross-attention and never sees lyrics — conditioning is
channel concatenation from the condition encoder. The only place lyric tokens
and audio frames coexist is the **LM decode loop**, so that was probed
(`MM3_ALIGN_DUMP=1`, `MM3_ALIGN_FILE=<path>`; forces the manual F32 attention
path because flash fuses the softmax away).

Result: **viable at LINE level, which is the granularity that matters.**
`lrc_align()` emits lines ("consensus → DTW → sentence grouping → LRC text"),
not words, so line onset is the bar. Three heads track the lyric across every
test clip — **L12/H27, L19/H7, L24/H29** — and a naive DTW over their consensus
gives median line-onset error **0.83 s (indie)** and **0.71 s (synth)**, all
lines within 2 s. Folk was inconclusive (only 2 lines matched, and Whisper
itself renders that clip as a single 20 s segment).

Errors skew consistently NEGATIVE — attention leads the audio by ~0.6–0.8 s,
which is expected (the LM attends to a token as it begins generating that
content) and is a constant offset worth calibrating out, not noise.

Do NOT judge this at word level. An earlier pass did, called it unusable, and
was wrong twice over: it compared "fraction through BPE tokens" against
"fraction through words" — curves that differ even for a PERFECT alignment,
because tokens-per-word varies — and it used a naive DTW over 3 heads rather
than `lrc_align()`'s consensus denoising with ACE's 7-head-scale config.

Traps, each of which produced a wrong answer first:
1. **55 % of all 1152 heads sit permanently on one structural token.** Any head
   ranking must reward MOVEMENT — scoring "monotonic" as `delta >= 0` counts a
   pinned head as perfectly monotonic (it scored 0.98 and ranked first).
2. **Single-clip results do not generalise.** The best head on one clip (L16/H13)
   did not make the top 14 across three. Rank by the WORST clip, never the mean.
3. **Whisper `base` is not adequate ground truth for sung vocals** — it returned
   11 word timings for a folk clip `large-v3-turbo` transcribes as one 10-word
   line, which made attention look 7 s wrong. Validate against `large-v3-turbo`.
4. **Capture requires the manual attention path on EVERY layer, not just the
   three that are read.** This is the expensive, counter-intuitive one. Leaving
   the other 33 layers on flash is cheaper (9.8 vs 11.2 ms/step) and produces
   identical audio — but WRONG alignment: same seed, selective capture stamped
   the lines at 0.9/3.6/7.5/10.3 s where the vocal sits at 0.1/10.2/15.1/19.4,
   while all-manual gave 2.2/9.9/16.5/19.5. The captured layers' attention
   depends on whether the layers feeding them ran flash or manual by far more
   than that difference is supposed to matter. Not understood; do not "optimise"
   it back without re-validating against Whisper.
   Net cost WAS ~+49 % on the LM step — **superseded 2026-08-21 by the
   post-hoc replay pass** (`mm3_lm_lrc_replay`, mm3-lm-graph.h): decode runs
   pure flash (audio bit-identical to a no-LRC render, verified) and the
   alignment attention is recomputed afterwards from the sampled codes —
   teacher-forced 256-query chunks, all-manual, blocks 0..24 only, single CFG
   row via ne3=1 views onto KV row 0 — at ~0.1-0.5 s per song. Causality makes
   prefill attention identical to decode attention over the same tokens, and
   all-manual keeps it out of the mixed-graph trap above; LRC verified
   character-identical to the live path on the same forced codes.
   `MM3_LRC_LIVE=1` restores live capture (validation/fallback).
5. The dump is **MM3ALIGN2**: an ASCII header line, then `tokens` × int32 lyric
   token ids, then f32 in `[frame][layer][head][token]` order. v1 omitted the
   ids, which is what forced the bogus token-progress-vs-word-progress
   comparison. Resolve ids to text via `tokenizer.ggml.tokens` in the LM GGUF.

**SHIPPED** (`features.lyricTimestamps: true`, request field `get_lrc`). The
engine emits LRC on `Job::result_lrc` → the existing `x-lrc-text` header → the
server writes `<uuid>.lrc`, same contract ACE uses. Measured against Whisper:
median line error **0.39 s** over the matched lines of two clips (synth
+0.20/+0.52/+0.62/−0.02, indie +2.12/−0.26/+1.36/+0.14).

`lrc_align()` is NOT used, and that was measured rather than assumed: fed the
same captured attention it stamps every line in the first half of the song, and
sweeping violence_level 2.0 → 0.0 changes nothing. A plain monotonic DTW over
the head-averaged matrix, grouped on newline tokens, is what works — see
`mm3-align.h`. `lrc_align()` stays untouched for ACE.

## Runtime LM adapters (SHIPPED 2026-08-20 — engine + server + UI)

`engine/src/minimax/mm3-lm-adapter.h` loads PEFT LM LoRAs (SimpleTuner
`language_model.`-prefixed checkpoints, q/k/v/o + gate/up/down × 36 layers)
and applies them as RUNTIME low-rank deltas in the AR stage — base weights
untouched, so per-group scales are live per generation. Wire fields on
`POST /mm3/synth`: `lm_adapter` (abs path), `lm_adapter_scale{,_attn,_mlp,
_early,_mid,_late}` (all default 1.0; range ±4). Scales bake into cached
graphs as constants — `mm3_lm_set_adapter` invalidates the slots on change
(once per generation, cheap). Single-slot cache keyed (path, mtime), guarded
by g_mm3_mutex; dropped on /mm3/unload, transient release, and the staged
after_ar handover (VRAM). Load failure FAILS THE JOB — never silently base.
Server: `backends/minimax/lmAdapter.ts` (containment-checked refs under
`<adapters root>/mm3-lm-adapters/`, sidecar metadata, `params.mm3LmAdapter*`),
defaults = the ablation-validated attention 1.0 / MLP 0.5. Catalogue route
`GET /api/mm3/lm-adapters` (adapters + sidecars + default scales). UI:
`ui/src/components/global-bar/Mm3LmAdapterDropdown.tsx` renders in the Adapters
cluster whenever `capabilities().features.lmAdapters` is true and
`features.adapters` is false — its own flag, because `adapters: true` gates
ACE's whole DiT-stack UI (merge/runtime modes, per-section masking, trigger
embedding). Picker prefills the scale dials from the picked adapter's sidecar
`recommendedScales`; depth thirds sit behind an advanced disclosure. Values
ride `backendParams` → `getGlobalParams()`, so a new dial needs no store field.
**Two application modes since 2026-08-21** (`lm_adapter_mode`, UI toggle,
`params.mm3LmAdapterMode`): `runtime` (default; live dials, measured +28 %
LM step on f16 and **+51 % on q8_0** — the fixed overhead looms larger over
halved streaming) and `merge` (mm3-lm-merge.h folds scale·B·A into the
resident weights: merged step == base step, 15 s once per (adapter, dials)
on q8_0 via dequant+delta+requant — IQ types refused, need imatrix).
`MM3Model::lm_merge_tag` tracks what is baked in; mismatch forces a pristine
LM reload (verified live under keep-loaded); staged mode reloads per gen
anyway. Merge failure part-way drops LM residency — never leaves mixed
weights serving.
**Writing a checkpoint + a JSON sidecar into `<adapters>/mm3-lm-adapters/<run>/`
is the entire publish step** — that is the contract the future native trainer
targets (docs/plans/2026-08-20-mm3-training-server-design.md §5). r256 ≈ 1.4 GB
resident, ≈ +9 % AR cost (unmeasurable at short lengths). Smoke-validated:
252 modules load, base + adapter renders complete same wall time.

## Native LM LoRA training: `ace-train mm3-lm-train` (SHIPPED 2026-08-20)

MM3's LM is Qwen3-8B, and `train/lm-graph.h` was already a trainable
cache-free unfused Qwen3 forward — so the trainer is a RETARGET, not a build.
`train/mm3-lm-load.h` does the load side (llama.cpp tensor names, `qwen3.*` KV,
UNTIED head); `train/mm3-lm-train-run.h` does the data path, the frame
embedding, the output slice and the loop. AdamW, **Muon**, PEFT export,
checkpoint/resume all come from the existing ACE machinery.

Validation ladder, all runnable:
- `ace-train mm3-lm-probe` — the trainer's forward vs the engine's own prefill
  (cos 0.999999951, argmax identical).
- `ace-train mm3-lm-loss` — teacher-forced CE with falsification diagnostics
  (`--target-shift`, `--no-prompt`).
- `ace-train mm3-lm-train --fd-check N --f32-layers 2` — **the gradient gate,
  and the one to run after touching anything in the backward.** Exit code 0/1,
  two independent bars: checkpointed-vs-naive gradients < 2e-3 (measures
  3.78e-07) and finite differences < 2e-2 (measures 0.0023). Both verified by
  injected faults, not just by passing — see the header of
  `engine/src/train/mm3-lm-train-run.h`.
  **`--f32-layers` is not optional if you want an answer.** Without it, f16
  rounding across 36 layers is larger than the defect being looked for: an
  injected off-by-one moved the number from 7.70e-02 to 2.14e+00 and the
  command still exited 0. It truncates to 2 layers and mirrors them plus the
  scored head slice to F32 (~1.7 GB; a full 8.6B F32 mirror would be ~34 GB).
  Run it against **f16 even when training on q8_0** — isolating a quantized
  base would measure the quantizer, and `mm3_f32_isolate()` refuses it.

### Training bases: any installed quant, and the VRAM ladder

Since `quant-cpy-kquant.patch` the trainable base is **whatever is installed**,
not just f16/q8_0 — every K-quant, MXFP4, NVFP4 and IQ type. The mechanism is
QLoRA-style dequantize-per-matmul: `qwen3_f32()` emits `ggml_cast` on the frozen
weight, so the backward only ever sees the cast's F32 output.

Peak VRAM, MM3 8.6B, rank 256 / 1500 frames, **all within ~5 % of the same step
time** (~3.8 s/step on a 5090):

| base | size | peak | 1st-step loss vs f16 |
|---|---|---|---|
| f16 | 16.0 GB | 31.4 GB | reference |
| q8_0 | 8.5 GB | 22.6 GB | +0.02 % |
| Q6_K | 6.6 GB | 20.7 GB | +0.3 % |
| Q4_K_M | 5.1 GB | 19.2 GB | +0.8 % |
| MXFP4 | 5.1 GB | 19.1 GB | +2.7 % |
| Q2_K | 3.4 GB | 17.5 GB | **+14.3 % — do not train against this** |

**Rank is the bigger lever below 20 GB**, at 31.2 MB per unit: Q4_K_M peaks
19.2 GB at r256/1500, 13.2 at r64/1500, 11.1 at r32/750, 10.2 at r16/500.
~10 GB is the practical floor for MM3 LM training.

`estimateMm3PeakMb()` in `services/training/mm3Train.ts` predicts all of this to
<0.3 %, and `recommendMm3Config()` picks base + rank for the detected card
(read from the engine's `/vram`). **f16 is never recommended**: it measures
fidelity-equivalent to q8_0 at twice the VRAM, and at 1.5 GB free it pages over
WDDM — measured at 12–14 s/step against q8_0's 3.75 on the same 12 steps.

### Four things that cost real time here

1. **THE PROMPT DOMINATES THE SEQUENCE.** An MM3 prompt is ~1,125 tokens, so
   `--max-frames 128` still gives S=1253. Shrinking the crop is NOT the VRAM
   dial you expect, and per-layer gradient checkpointing is load-bearing rather
   than an optimisation. Naive fwd+bwd retains ~18 GB of activations on top of
   a 16 GB f16 base and spills into WDDM shared memory: 38 s/step where the
   same forward alone takes 644 ms.
2. **AT r256 ON A 32 GB CARD, MUON FITS AND ADAMW DOES NOT.** AdamW keeps two
   momentum buffers where Muon keeps one — +2.66 GB at rank 256 on a config
   already peaking at 31.7 GB. Lowering `--max-frames` does NOT rescue it (the
   crop only moves the ~0.6 GB of checkpoint buffers). Muon here is the
   optimizer that runs, not just the one that might train better.
3. **Grad clipping is nearly a no-op for Muon params.** Newton-Schulz
   Frobenius-normalises its input, so clipping only rescales a direction that
   is renormalised anyway. Do not read the clip figures as a tuning signal;
   `--muon-lr-scale` is the knob, and Muon's LR does not mean what AdamW's
   means.
4. **lm-ckpt.h was EXTENDED, not forked**, with two hooks that are inert by
   default (`LmCkptCfg::{head_w,head_row0,head_v}` for an untied scored head;
   `LmCkptRun::embed_build` for the frame-embedding entry). An ACE run emits
   byte-identical graphs. Keep it that way — that file is what makes ACE 4B
   training fit.

## Native codes export: `ace-train mm3-codes` (SHIPPED 2026-08-20)

Audio -> RVQ codes, natively. `engine/src/minimax/mm3-rvq-encode.h` ports
PurpleOrc's V4Encoder (169M: conv stack + 3 dilated ResBlocks + frame pooling +
8 pre-LN transformer layers + causal depth decoder). The graph is
**weights-agnostic** — the whole community encoder lineage shares this
architecture, so a new checkpoint is a `engine/tools/convert-rvq-encoder.py`
run (arch `mm3rvq`, verbatim PyTorch tensor names), never a code change.
Adopted checkpoint: `models/mm3/mm3-rvq-53kpooled-f32.gguf`.

```
ace-train mm3-codes --dataset <ds.json> --rvq <mm3-rvq-*.gguf> --enc <mm3-enc-*.gguf> --out <dir>
ace-train mm3-codes --rvq <mm3-rvq-*.gguf> --fixture <f.fix>     # standing gate, no audio needed
```

Four traps, all pinned in the header because each is SILENT if wrong:
1. **GELU must be `ggml_gelu_erf`**, not `ggml_gelu` — torch's default is
   erf-exact, ggml_gelu is the tanh approximation, ~1e-3 apart, against a
   semantic argmax margin whose p05 is 0.06.
2. **`GroupNorm(1, C)` reduces over channels AND positions jointly**, not per
   position — a [T,1,C,1] reshape into `ggml_group_norm(n_groups=1)`.
3. **The encoder stack is pre-LN with NO final norm**; `norm_out` is applied by
   V4Encoder outside `nn.TransformerEncoder`.
4. **Windowing is the model's own `frame_latent_starts`**, integer for integer,
   constants carried in the GGUF; windows are FIRST-WINS.

**Parity, and the lesson in how to gate an argmax port**: the fixture rung is
exact (feats/logits cos=1.000000000, 0/128 semantic codes differ). End-to-end
against the python exporter over 13 tracks: row counts identical, semantic
99.910% identical, acoustic 99.412%. The residual is f32 GEMM ordering, MEASURED
not assumed — the reference encoder run on OUR latents shows the same ~0.05%
(so it is not the DAV port), and every flipped frame sits at a reference argmax
margin of 0.0002-0.0028 against an all-frame median of 1.05. **"Bit-exact codes"
is the wrong unit gate for any argmax port**; split flips by margin, which the
fixture mode does.

## Training: DiT yes, LM ~~never~~ — SUPERSEDED (see below), assessed 2026-08-14

Full teardown + staged plan: `docs/plans/2026-08-14-mm3-training-feasibility.md` (local).

**MM3 ships decode-side only for the audio-token path — there is no audio → code encoder.**
Verified from the HF tensor inventory, not inferred: `qwen_7B/` (18.5 GB) is the **LM +
RVQ depth decoder in MiniMax's original `AbabForCausalLM` format** (`model.layers.*` +
`model.audio_decoder.*` with 7 audio heads, `embed_tokens [200000,4096]`), matching
`language_model/` + `rvq_depth_decoder/` to ~5 MB. It is **NOT a music tokenizer** — an earlier
note here said it was, which wrongly made LM training look possible. Consequences:

1. **LM / planner adapters are BLOCKED.** ← **SUPERSEDED 2026-08-17+**: the community
   audio→codes encoders (SimpleTuner v4 lineage → PurpleOrc 53k → Mothersuperior
   53k-pooled, best-by-ear 2026-08-20) provide the codes; LM code-SFT WORKS (our lm_sft
   line + SimpleTuner LM mode; Skiba timbre reached 2026-08-20 with r256 attn+MLP).
   The native-GGML LM trainer is the open build (docs/plans/2026-08-20-mm3-training-studio.md).
2. **DiT conditioning cannot be teacher-forced.** The only way to get `frame_hiddens` for a
   training clip is to AR-sample the LM from its caption/lyrics — so condition and target come
   from *different music*, matching only in caption/lyrics/duration. A timbre/production adapter
   can still work (low sigma is well-posed); the risk is the DiT learning to ignore
   `encoder_hidden_states`, i.e. prompt/lyric adherence. Measure it with the LRC probe above.

The encoder that IS needed and IS released: **`dav.pth` (492 MB) — the DAV encoder**, audio →
128-ch latents @ 86.13 Hz, L/R through shared weights, **mean only** (`logs_proj` unused). Our
local `comfy/vae/minimax_music3_dav.safetensors` is **decoder-only**, so it is a new download.

SimpleTuner ([#3074](https://github.com/bghira/SimpleTuner/pull/3074)/#3075, merged 2026-08-14)
trains the 2.4B flow DiT only, targets `to_q/to_k/to_v/to_out.0/ff_in/ff_out/proj_in/proj_out`,
plain rectified-flow loss, and defines the de-facto **ComfyUI MM3 LoRA format**. No published
evidence anyone has trained a *good* MM3 LoRA yet — treat it as a reference implementation, not
as validation. Two traps for loading those LoRAs: we **fuse q,k,v** into `dit.blk.N.attn_qkv`,
and our `mm3.dit.glu_order = value_gate` must be reconciled with their `swiglu_gate_first`
metadata. `convert-mm3.py:build_dit` already resolves both ComfyUI and diffusers names.

Recommended first step is **load, don't train**: add an `mm3` target to the adapter system,
train one LoRA in SimpleTuner, and hear whether it works before building a native trainer.
