# Release notes draft — v1.1.4 → HEAD

> Paste into the CI-drafted release body. Two flagship features this cycle —
> consider calling it **v1.2.0** rather than v1.1.5.

---

The StableStep release. Two flagship features: **StableStep**, a Stable Audio 3-powered
instrumental refiner that finally kills VAE fizz at the source, and **MIDI Studio**,
audio-to-MIDI transcription running on our own native GGML port of MuScriptor. Plus
timestep-gated adapters, parameter profiles, two new MDMAchine plugins, and a stack of fixes.

## ✨ StableStep — Stable Audio 3 instrumental refining

Generated tracks carry a characteristic high-frequency "fizz" from the ACE-Step
autoencoder. StableStep re-renders the instrumental through **Stable Audio 3**
(SDEdit-style partial re-noising, 8-step distilled rectified flow) so the fizz band is
*regenerated* with real spectral detail instead of filtered. Vocals are never touched:
BS-RoFormer splits them out (lead **and** backing), PP-VAE polishes them, and they're
remixed over the refined instrumental — the exact complement split guarantees nothing
is lost, and lyrics stay bit-identical.

- **One toggle + a strength slider** (0.10–0.60) in Post-Processing. The refine prompt is
  derived from each track's own caption automatically (vocal descriptors stripped).
- **Runs natively in the C++ engine** — no Python. Two backends, selectable in-app:
  - **GGML** (4 GGUF files, ~5.8 GB): CUDA, Vulkan, and CPU. The fastest option on
    NVIDIA in our testing — ~2 s of compute for a 30 s clip on an RTX 5090.
  - **ONNX Runtime / TensorRT** (~12 GB): NVIDIA alternative path.
- **Numerically validated end to end**: every ported component matches the reference
  implementation at cosine > 0.9998; the two backends agree with each other at 0.9999.
- **Model Manager → StableStep tab** downloads either backend set from
  [scragnog/HOT-Step-CPP-StableStep](https://huggingface.co/scragnog/HOT-Step-CPP-StableStep),
  with license acceptance built in (Stability AI Community License — free for individuals
  and orgs under $1M revenue). *Powered by Stability AI.*
- New engine surface for tinkerers: `POST /sa3-refine` (strength/steps/sampler/backend)
  and SuperSep `level=4` — a dedicated vocals+instrumental split.

## 🎼 MIDI Studio — audio-to-MIDI on the native engine (#80)

Transcribe any library track or uploaded WAV/MP3 into multi-track MIDI (34 instrument
groups + drums) using **MuScriptor** (Kyutai & Mirelo) — ported phase by phase to our own
C++/GGML `ace-midi` engine and validated **byte-exact** against the reference
implementation. A 3.5-minute track transcribes in ~50 s on an RTX 5090; no Python anywhere.

- **Live piano roll** fills in while transcription runs, with per-channel instrument
  coloring; crossfade playback slider between the original audio and the MIDI rendition,
  plus per-instrument mute/solo.
- **Three model sizes** (small 103M / medium 307M / large 1.4B) with in-app gated-weight
  download flow (weights are CC BY-NC 4.0 — non-commercial).
- Engine work along the way: exact-parity mel frontend + KV-cache greedy decode, F16 KV
  cache to fix CUDA decode corruption, byte-exact event decode + MIDI writer.
- `ace-midi` ships in every platform bundle.

## 🎛 Adapter system

- **Timestep-dependent adapter gating** (interval experts / MoE) — restrict any stacked
  adapter to a step window, e.g. one adapter shapes structure early, another handles
  detail late. UI windows are evaluated **per step** (not raw t), and they now compose
  correctly with Adapter VRAM quant (previously a silent 32 GB blowup).
- **PEFT DoRA support** + per-module `alpha_pattern`.
- **`runtime_lowrank` mode** — factor-apply without materialized deltas: the lowest-VRAM
  way to run big adapter stacks.
- **Merge (low VRAM)** — opt-in requant of merged weights back to the base's native
  quant type (~¼ merged-DiT VRAM on a Q8 base).
- LoKr Kronecker-apply self-test (`HOTSTEP_KRON_TEST`); `gain_domain` parse fix.

## 📋 Parameter profiles

Save, apply, and delete named full-state snapshots of your generation parameters —
in-app, with JSON import/export, inline rename, and click-to-inspect.

## 🔌 Plugins (MDMAchine)

- **MD HT Scheduler V3** and **MD Trajectory Anchor V5**, with user manuals.
- Max inference steps raised 200 → 300 (UI + engine clamp).

## 🔧 Fixes & polish

- **Storm streaming**: stream died instantly (premature `close` on body consumption);
  "Keep DiT & VAE loaded" is now honored between stream slots.
- **In-app restart** was dead on Windows (ping-as-sleep hang meant taskkill never fired).
- **Queue**: deleted songs pruned from the persisted queue on load; Nuke All Generations
  clears the queue store and recent-songs cache.
- **Logs**: GGML debug spam dropped at source with digit-insensitive dedup; terminal
  stick-to-bottom pin survives layout shifts.
- MCP: generating model name appended to song titles.

---

**Models:** StableStep sets are on
[Hugging Face](https://huggingface.co/scragnog/HOT-Step-CPP-StableStep) or one click away
in the Model Manager. MuScriptor weights are gated (free) — request access via the links
inside MIDI Studio.
