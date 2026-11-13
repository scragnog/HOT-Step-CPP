---
name: mm3-backend
description: Maps HOT-Step's native MiniMax-Music3 backend — engine port modules, endpoints, server/UI integration, parity/fixture infrastructure, and the hard-won trap list. Use when working on anything MM3 — engine/src/minimax/, backends/minimax/, /mm3/* endpoints, the backend toggle/capability gating, MM3 model files or Model Manager entries, debugging MM3 generations, MM3 performance work, or extending MM3 features (covers, training, Lyric Studio).
---

# MiniMax-Music3 backend

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Native C++/GGML port of [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3),
built 2026-08-13 (release day) as HOT-Step's second generation backend behind an N-backend
abstraction. **Status: text2music only**: no cover, repaint, lego, extract or stems
(`features` in `server/src/services/backends/minimax/index.ts`). Around that it has grown runtime
LM adapters, the shared Lua sampler plugins, LRC, Whisper, StableStep and the VST/mastering
post chain, play-while-rendering streaming, natural-ending candidates and LM LoRA training in
the Training Studio. Flow-DiT LoRAs merge only through the `MM3_ADAPTER` env var
(`mm3_apply_adapters`, `engine/src/minimax/mm3-model.h`), and the native DiT trainer
(`ace-train mm3-train-dit`) is CLI-only. Output = raw 44.1 kHz stereo WAV (app norm is 48 k;
PP-VAE and Spectral Lifter, which assume ACE's VAE, are skipped for MM3).
Cover Studio automatically prepares each source: dataset BPM/key/lyrics and
`<stem>.mm3.txt` first, ordinary source analysis and queued Music-Flamingo lyrics
otherwise. The source caption can be edited; an empty caption offers a
Music-Flamingo button, with Model Manager guidance when unavailable.
`POST /api/mm3/cover/source-prep` queues lyrics/caption on the labeling GPU lane
alongside encoding and returns them in the job summary's `result`.
The panel always sends `launder: true` to `POST /api/mm3/cover/encode`.
API callers can still use plain `mm3-codes`. Laundering needs rec7 and the encoder,
takes roughly real time and about 14.5 GB of VRAM. Advanced holds `launderSteps`
(default 30) and `launderSeed` (42; -1 records the random seed actually used).
Source SHA-256, byte size and requested launder settings identify reusable planks;
`force: true` on encode bypasses reuse. The sidecar and song facts record launder
settings and model files. LM quant stays Q8_0.
Target artist/album/track captions use the source BPM, key and scale while keeping
the genre tail. Selecting an album applies its saved `mm3_adapter_path` through
the global params store. Match 100 omits forceBias for semantic hard pinning;
below 100 it sends `round(10^(3 * match/100))`, shown as bias in song facts.
Trim and planner seed remain; patterns and audio2audio remain API-only.
Source guide (`lm_guide_*`, API `mm3Guide*`; `--guide-k` on mm3-codes/mm3-launder)
ports YuE2's Follow-source mix to the semantic draw: the LM picks among the
encoder's top-K codes. Guide frame j is emitted frame j and iteration 0 reads
frame 0. Scale 1 with k 1 is byte-identical to the hard pin; keep that gate when
touching the loop. Round 8 of the cover probe (one real song, K 32): agreement
with the source tracks the scale almost linearly, and even at 0.9 the LM puts
only about 0.2 of its mass on the candidates. The guide lowers the cost of
following, but it does not make the LM agree. A vocal-forward source (backing
-12 dB) raised adapter word recall from 0.43 to 0.83. Vocal only comes back
spectrally off (LTAS about 20 dB), and a second pass loses words.
Round 9: the flow DiT renders mostly from stream 0, the LM's hidden row. Swapping
streams 1 to 7 of a cover for the clean launder's moved the audio 0.15 dB; swapping
stream 0 alone turned it into the launder. Nudging stream 0 until the drawn code
is in its top 10 (`lm_state_fix`) needs only about 3% of its norm and changes the
audio by 0.3 dB. A greedy depth decoder (`lm_depth_greedy`) derails the plan.
Blending the whole block between cover and launder (`--keep-hiddens`, then
`forced_frame_hiddens_file` under model key `external`) is a smooth dial.

Deep docs (local, gitignored): `docs/plans/multi-backend-architecture.md` (architecture plan,
day-0 findings, op inventory) and `docs/plans/mm3-gguf-layout.md` (GGUF contract + loader
addendum). Caption format: the **mm3-captioning** skill.

## Model + pipeline (25 fps frames; every module parity-proven vs the diffusers reference)

```
caption+lyrics → Qwen2 BPE → Global LM 8.59B (Qwen3 arch, semantic codes @ ids 151675–168058,
  EOS 151670, AR CFG 1.5 as persistent 2-row batch) → per frame: RVQ depth decoder 0.6B
  (7 acoustic codebooks) → frame_hiddens [F,8,4096] → per 200-frame window (hop 100;
  app default 150): condition encoder 25M (×3.4453125 nearest resample) → flow DiT 2.4B
  (30 Euler steps, app default 20;
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

MM3's `group: 'lm'` set is the six `mm3Lm*` sampling knobs plus the seven that
decide what happens to the planner's output: `mm3ArSeed`, `mm3ReuseAr`,
`mm3SaveArCodes`, `mm3PlankPath`, and the saved-plan trio `mm3SaveHiddens`,
`mm3HiddensName`, `mm3HiddensPath`.

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
| Engine modules | `engine/src/minimax/`: `mm3-model.h` (loader/residency, `MM3_ADAPTER` DiT merge), `mm3-tokenizer.h`, `mm3-lm-graph.h`, `mm3-ar-loop.h`, `mm3-sample.h`, `mm3-depth-graph.h`, `mm3-cond-graph.h`, `mm3-dit-graph.h`, `mm3-dit-trt.h` (TensorRT DiT renderer), `mm3-vocoder-graph.h`, `mm3-pipeline.h` (e2e + chunking), `mm3-request.h` (prompt assembly/hygiene), `mm3-job.h` (job queue + VRAM arbitration), `mm3-server.h` (endpoints), `mm3-plugins.h` (Lua sampler bridge), `mm3-align.h` (LRC), `mm3-ar-cache.h` + `mm3-hiddens-file.h` (AR cache, saved plans), `mm3-lm-adapter.h` / `mm3-lm-merge.h` / `mm3-lm-dora.h` (LM LoRAs), `mm3-adapter.h` (flow-DiT LoRA merge), `mm3-imatrix.h` (LM imatrix collection); training-side `mm3-dav-encode.h`, `mm3-rvq-encode.h`, `mm3-rec7.h` |
| Hooks | include/call pairs in `engine/tools/hot-step-server.cpp`: `minimax/mm3-server.h` + `mm3_register_routes`, and the mid-file `minimax/mm3-job.h` + `mm3_register_job_routes`; checked by `engine/verify-hooks.ps1` ("Hook 4-family" rows, plus "Hook 4-gate" for the boot gates) |
| Server backend | `server/src/services/backends/`: `types.ts` (EngineBackend + capability manifest), `registry.ts`, `ace/`, `minimax/{client,index,generate,hiddens,lmAdapter,plank,trigger}.ts`, plus `yue2/` and `sa3/`; routes `server/src/routes/backends.ts` (backends, capabilities, models, MM3 planks/plans/LM adapters); `runGeneration` in `routes/generate.ts` dispatches to the envelope's `backend.generate()`, which for MM3 is `runMinimaxGeneration` |
| UI | `stores/backendStore.ts`, `hooks/useCapabilities.ts`, `global-bar/BackendToggle.tsx` (the engine picker, rendered at the top of the sidebar by `sidebar/Sidebar.tsx`; null until ≥2 backends), `shared/BackendCapabilityGate.tsx` (studio guards), gating in `GlobalParamBar.tsx`, `global-bar/Backend{Generation,Lm,Models}Dropdown.tsx` + `BackendExtensionControls.tsx`, `SamplerPluginControls.tsx`, `Mm3LmAdapterDropdown.tsx`, streaming in `stores/mm3StreamStore.ts` |
| Models | **5-way split since 2026-08-14** (ported from ServeurpersoCom/minimaxmusic.cpp): `<data root>/models/mm3/mm3-{lm,depth,cond,dit,voc}-<quant>.gguf` (archs `qwen3` / `mm3-{depth,cond,dit,voc}`), legacy `mm3-synth-*` bundles still load (fill any role; split file wins per quant token). Per-role quant mixing (LM Q8_0 + DiT Q4_K_M is the headline combo); a DiT/adapter swap reloads only cond+dit+voc — LM stays warm. cond/voc are **never quantised** (f16 only). Hosted `scragnog/MiniMax-Music3-GGUF`; registry role `mm3`, packs rebuilt on split components in `server/src/data/model-registry.json` |
| Converter | `engine/tools/convert-mm3.py` (safetensors→GGUF bundle; folds weight-norm; refuses pruned/int8_convrot) then `engine/tools/split-mm3.py` (byte-exact bundle→5-way split; idempotent; cond/voc only from native bundles) |
| Fixtures / parity | `D:\Ace-Step-Latest\mm3-weights\fixtures\` (manifest.json + raw f32 dumps + reference WAVs), seed-spread study in `..\seed-spread-2026-08-13\`; venvs: `.venv-convert` (numpy/gguf), `.venv-ref` (patched diffusers @ dafe3733 — `patch_venv.py --restore`; `capture_fixtures.py --replay` rebuilds dumps without rerunning the model) <!-- TODO(verify): D:\Ace-Step-Latest\mm3-weights\ was absent on the dev machine on 2026-10-23; the engine headers still cite mm3-weights/fixtures/. Confirm where the fixtures live now. --> |

## Engine endpoints (:8085 via app, standalone tests on :8086)

`GET /mm3/props` (files/config/loaded/limits — **blocks while an MM3 generation runs**; the Node
client calls it with a 2.5 s timeout and keeps last-known-good, `minimax/client.ts` `mm3Props`),
`POST /mm3/warm` / `POST /mm3/unload` (idempotent; unload frees weights+KV),
`POST /mm3/select-model` (per-role quant selection; send every role, see performance reference),
`POST /mm3/synth` (production, rides the same FIFO GPU
worker as ACE `/synth`; standard `/job?id=` progress/cancel/result; request contract documented
in `mm3-request.h`/`mm3-job.h`), `GET /mm3/job?id=` (MM3-vocabulary progress, never blocks;
`&ar=1` returns the Plank code blob — see below), `GET /mm3/take?id=&take=N` (one ensemble take's
audio, `&lrc=1` / `&ar=1` for its LRC or codes; never takes the MM3 mutex),
`GET /mm3/stream?id=` (live audio of a running job — chunked WAVs, one reader, never takes the
MM3 mutex; see "Streaming player" below),
`POST /mm3/tokenize-check` (cold-capable; 5000-token limit), `POST /mm3/imatrix`
(`start`/`stop`/`status`/`reset`/`save` an LM activation-importance collection, driven by
`/mm3/lm-plan`, for the 2-3 bit LM quants), plus deprecated bring-up endpoints
(`/mm3/voc-decode`, `/mm3/dit-forward`, `/mm3/flow-sample`, `/mm3/depth-frame`,
`/mm3/cond-encode`, `/mm3/lm-plan`, `/mm3/synth-e2e`) kept for parity work — they run GPU work
on httplib threads; never build production paths on them. Routes register in
`mm3-server.h:2258-2270` and `mm3-job.h:1971-1974`.

**Standalone launch gotcha:** ace-server exits `0xC0000135` with zero output unless
`engine/trtllm-libs` + `engine/deps/tensorrt_libs` are prepended to PATH (aceEngineProcess.ts
does this; `engine/server.cmd` does not).
<!-- TODO(verify): since 70c0ce60 (2026-09-11) nvinfer_10.dll and nvonnxparser_10.dll are
delay-loaded (engine/CMakeLists.txt:359-373) and TRT-LLM is off by default, so a current build
may start without them on PATH. Re-test standalone launch before relying on this note. -->

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
times. Since 242645c4 a run where nothing ever ends renders the longest capped
plan anyway and reports `eos_fallback: true`, which the server relays as a
warning rather than a normal success. `min_frames` adds a floor: a candidate
that ends shorter is dropped and re-planned like one that never ended
(`takes_short`). The job JSON's `takes` is the number RENDERED; `takes_planned`,
`takes_dropped`, `takes_capped`, `takes_short`, `eos_rounds_used`, `require_eos`,
`eos_fallback` and a per-take `round` are added, and `take_detail` is emitted
whenever candidates were in play. A streamed render with `require_eos` stays
serial: the candidates need the whole plan before anything renders (logged).
Server (`minimax/generate.ts`): `mm3RequireEnding` (default on, Generation
dropdown) sends at least 3 takes (`mm3Takes` raises it for variations), eos_rounds 4,
and `stop_after_first_eos` when one song was asked for; `mm3MinLength` (default
on) derives `min_frames` from the lyrics at 4 words/s with a 15 s floor. It reads
the surviving count/seeds from the completion detail.
Duration on MM3 is ALWAYS auto (a requested length was a hard cap that cut
endings off): the wire always carries the 300 s ceiling and the Create panel
hides the control in MM3 mode. Batched take 0
is a different song from the same seed by design (check-mm3-ensemble.mjs).
Knock-ons: Save Plan To Disk is dead while the toggle is on; each ended
candidate costs its own flow pass.

**Adapter fade (2026-11-08, 28c79e42, opt-in, no UI dial yet).** Adapters
suppress the base's ending instinct, so `lm_adapter_fade_*` (engine
ARCHITECTURE.md) can step the LM adapter down to the base for the tail
(`MM3LmAdapterSchedule`; runtime mode only, the server forces it). Plan-only
count, one adapter, one prompt, 12 takes, 300 s cap: control 2/12, base 12/12,
**ramp from 210 s to the base at 270 s 9/12**, hard handoff at 210 s 5/12, ramp
or handoff at 150 s 2/12 and 0/12. A fade during the lyrics loses: the base
then has to finish the adapter's song and runs to the cap. A fade after the
last lyric works. On that prompt the base sang the sheet by ~183 s and the
adapter by ~260 s; the 210 s start was base pace + 30 s. Runtime mode costs
about +44% LM step on q8_0 against merge.

**Plan dial (2026-11-09, `lm_adapter_plan`, UI "Plan", default 1).** Splits
the adapter's two jobs at the semantic/acoustic seam: the LM's semantic draw
is the plan, the depth decoder reading the LM hidden is the voice. Below 1
every take gets a second CFG pair (`MM3LmGraph::base_rows`, rows
`[K*P, 2*K*P)`) on the same prompt and the same fed-back codes, with the
adapter delta masked off by a resident `[1, 1, B]` tensor in the KV buffer
(`mm3_lm_mm`: plain LoRA masks `A x`; DoRA and LoKr mask the whole
contribution). The loop mixes each row as `base + plan * (adapter - base)`
before CFG and the conditional top-k, draws once, feeds the code to all four
rows, and the depth decoder keeps reading the adapter rows. **Parity rule:**
at 1, or with no runtime adapter, there are no base rows: same graph, same
draws. Proven by WAV hash: plan 1.0 at seed 20261208, 3 takes reproduced the
fade study's three control WAVs byte for byte; `check-mm3-ensemble.mjs
--adapter` section 4 holds it. The base rows match a pure-base batch at the
same row positions exactly. Costs (5090, q8_0, PiSSA r128 adapter, 2 takes,
300 s): LM step 15.0 -> 18.2 ms, KV 4.8 -> 9.6 GB, peak 23.4 -> 29.4 GB
on the card with ~4.5 GB of other apps. The rows double, so takes clamp to 2
(`mm3_max_takes(c, plan_rows)`): 3 takes wanted a 14.3 GB KV, filled the 32
GB card and ran ~10x slower. A graph input for the mask (uploaded per step)
cost +3.8 ms/step at one take; the resident tensor costs +0.85. Merge mode
has no base weights, so the server forces runtime; an adapter with an artist
token or KV prefix ignores the dial (it would need a per-row attention mask).
At plan 0 the mix returns the base logit outright, so an adapter -inf cannot
veto a base code; the fatal non-finite threshold stays one dead row per take
per step (the base rows only widen the reported percentage).
**4090 (24 GB):** 1 take, 300 s, plan-only: 16.2 ms/step at 1, 18.3 at 0.5,
peak 20.5 GB. 2 takes below 1 spills under `/mm3/lm-plan` (DiT and vocoder
stay resident there) but fits the app's staged `/mm3/synth`: 21.8 ms/step,
peak 23.5 of 24.5 GB. Do not add VRAM to that path without re-checking it.
**The base's ending instinct did not come back:** natural endings were 1/6 at
1, 0/6 at 0.5, 0/6 at 0 on that prompt with 2 takes on the 5090 (the base
alone: 12/12), and 1/12 below 1 at 1 take across both cards. The base
plans over the adapter's acoustic codes, which is not the base's own song.

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
9. GGUFs live in the **`<data root>/models/mm3/` subdir** deliberately: the ACE registry scan globs only the
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
15. **The user's ear beat every metric, twice.** They called "slow motion, too deep" on the first
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
19. **An MM3-only install must not kill the server at boot.** The startup gates in
    `hot-step-server.cpp` (`registry_scan` empty → exit 1; partial ACE synth without LM →
    exit 1) predate MM3 and knew nothing about it: a user with only `<data root>/models/mm3/*.gguf` got a
    dead engine → empty model dropdowns for BOTH backends + the MM3 "weights missing" CTA,
    while the Model Manager (Node disk scan, checks subdirs) said everything was installed
    (GitHub issue #118). Both gates now ask `hot_step_any_family_weights_present()`
    (`engine/src/hot-step-families.h`), a table whose MM3 row is `mm3_weights_present()`
    (mm3-model.h, a filename-only probe of `<models>` + `<models>/mm3`) beside the YuE2 and SA3
    probes; `verify-hooks.ps1` "Hook 4-gate" fails the build if an upstream sync restores the
    single-name gate. ACE handlers already degrade per-request with an empty registry. Any
    future boot-time hard-exit must ask "can any family still serve?" first.

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
