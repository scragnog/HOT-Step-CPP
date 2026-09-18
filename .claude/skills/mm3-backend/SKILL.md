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

## LM sampling knobs (2026-08-25)

The AR stage's semantic draw takes the full knob set (engine fields on
`MM3GenRequest`, wire names `lm_*`, UI via the minimax backend param registry
keys `mm3Lm*`): `lm_temperature`, `lm_top_k` (0 = the checkpoint's 50),
`lm_top_p` (nucleus over the top-k survivors), and `lm_rep_penalty` with the
ACE LM's three modes ported (`dry` default / `frequency` / `presence`).

- **Time constants are 25fps-rescaled**: window 320 (~12.8 s), DRY min-match
  15 frames (0.6 s). Never copy ACE's 5 Hz numbers (64 / 3-6) literally.
- **DRY** punishes only codes that would extend a verbatim recent cycle — the
  memorising-adapter loop failure — and leaves musical restatement alone.
  Useful range 1.05-1.15.
- **Parity is proven**: at default knobs the sampler takes the exact pre-knob
  code path; fixed-seed renders hash bit-identical across the change
  (da869838…, old and new builds, identical launch). Knobs at defaults are
  omitted from the wire so the engine recipe stays authoritative.
- The **depth decoder's sampler is untouched** on purpose: loops do not live
  in the per-frame acoustic codes, and perturbing its input distribution
  re-opens the timbre question the acoustic loss closed (training skill).

### Where the knobs render: the `group` field (2026-08-27)

Declared knobs (`capabilities().extensions`) carry an optional
`group: 'generation' | 'lm'`, and each generic top-bar dropdown renders its own
group — `BackendGenerationDropdown` and `BackendLmDropdown`, both on the shared
schema renderer in `BackendExtensionControls.tsx`. An untagged knob is a
Generation knob, which is where every one of them lived before groups existed,
so an older manifest still renders exactly as it did.

MM3's `group: 'lm'` set is the six `mm3Lm*` sampling knobs plus the four that
decide what happens to the planner's output: `mm3ArSeed`, `mm3ReuseAr`,
`mm3SaveArCodes`, `mm3PlankPath`.

Two things to know before touching the LM cluster:

- **`features.lm` does not mean "has an LM."** It means "has ACE's CoT metadata
  LM" — a stage that is genuinely optional. MM3 reports `lm: false` and still
  has an LM; it is just an autoregressive planner that always runs. The bar
  shows the LM tab on `features.lm || any knob tagged group:'lm'`.
- **No global on/off in MM3 mode.** `GlobalParamBar` hangs the section's
  `headerToggle` (`skipLm`) only when `features.lm` is true. There is no MM3
  render without the planner, so a switch there would be a lie.

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
`GET /mm3/stream?id=` (live audio of a running job — chunked WAVs, one reader, never takes the
MM3 mutex; see "Streaming player" below), 
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

## Natural-ending candidates (SHIPPED 2026-09-09, 00e3e7af)

`POST /mm3/synth` accepts `require_eos: true` and `eos_rounds: N` (1..16) with
`takes: K`. The planner runs K takes in one batched pass (seed+t); any take that
reaches max_frames without EOS is DROPPED before the flow stage; if none ended
the plan repeats at seed + K (round r plans seed + r*K + t) up to `eos_rounds`
times, then fails with "no candidate ended naturally". The job JSON's `takes`
is the number RENDERED; `takes_planned`, `takes_dropped`, `eos_rounds_used`,
`require_eos` and a per-take `round` are added, and `take_detail` is emitted
whenever candidates were in play. Ignored on an interleaved stream (logged).
Server: `mm3RequireEnding` (default on, Generation dropdown) sends takes 3,
eos_rounds 4 and reads the surviving count/seeds from the completion detail.
Duration on MM3 is ALWAYS auto (a requested length was a hard cap that cut
endings off); the Create panel hides the control in MM3 mode. Batched take 0
is a different song from the same seed by design (check-mm3-ensemble.mjs).
Knock-ons: Save Plan To Disk is dead while the toggle is on; each ended
candidate costs its own flow pass.

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
    untied head) and llama.cpp alone cannot run music generation. It IS interchangeable with a
    **depth-pruned distilled composer** — see "Alternative composer LMs" below.
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

## Validation bar for MM3 changes

Forced-replay parity against the fixtures (never sampled-path comparisons — RNG can't match
torch). Established floors: per-module ≥ 0.999 corr vs the bf16 dumps (the dumps' own floor,
~1.6e-2 relRMSE) or ≥ 0.9999 vs an fp32 CPU rerun of the reference module. Full-clip replay:
0.9988. If a change should be bit-neutral, prove it with the deterministic seeds.

## Deeper reference (read on demand)

Everything below is out of this file to keep it cheap to load. Open the one you need:

- [`reference/features.md`](reference/features.md) — **Feature subsystems**: Sampler plugins: shared with ACE; Caption composer: plain English -> Structured Caption, no LLM; Lyric timestamps: use Whisper, not attention; Analysis tool: where the vocals sit
- [`reference/performance.md`](reference/performance.md) — **Performance, caches and streaming**: Alternative composer LMs; Low step counts go THIN, not dull; MM3 Plank: the AR code cache; Streaming player: listen while it renders; MM3 AR cache: the speedup the Plank is not; Saved plans: the AR cache on disk; Performance budget
- [`reference/training.md`](reference/training.md) — **Training and runtime adapters**: Runtime LM adapters; Native LM LoRA training: `ace-train mm3-lm-train`; Native codes export: `ace-train mm3-codes`; Training: DiT yes, LM ~~never~~
