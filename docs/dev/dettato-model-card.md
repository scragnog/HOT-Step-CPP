---
license: apache-2.0
pipeline_tag: audio-to-audio
tags:
  - music
  - midi
  - transcription
  - safetensors
base_model:
  - kobimusic/dettato
  - kobimusic/dettato-evidence
  - kobimusic/dettato-grouper
  - kobimusic/dettato-drums
---

# dettato model mirror for HOT-Step

An unmodified copy of the four dettato 2.0.0 model repositories by KobiMusic,
for the native C++/GGML port in HOT-Step's MIDI Studio and Cover Studio. These
files are published in [scragnog/dettato](https://huggingface.co/scragnog/dettato).

## Why a mirror

The model registry pins each file's sha256 but has no revision field. If
upstream ever pushes new weights under the same names, every download from
upstream would fail its check until we ship a release. A copy we control cannot
move under the app. Each file is byte-identical to upstream's, so a weight
update is a copy and the hashes below stay comparable.

## Sources and licence

dettato is by KobiMusic ([code](https://github.com/kobimusic/dettato),
[models](https://huggingface.co/kobimusic)). Code and all four weight sets are
Apache 2.0. `LICENSE` and upstream's `NOTICE` sit at the repository root, as
upstream ships them. HOT-Step's C++ is a port written from upstream's Python
source (`engine/src/dettato/`), pinned to upstream commit
`51ade0f09f2272f4674166026582185d26baaeaf`.

**Training data caveat.** The grouper and drum model cards say part of their
training data (MUSDB18 and IRMAS) is licensed for non-commercial research only,
and tell users to check whether that matters for their use. It does not change
the Apache 2.0 licence of the weights. It is a question for you to answer before
commercial use.

## Contents

| Folder | Role | Size | Upstream revision |
|---|---|---|---|
| `dettato/` | Transcriber: audio to notes | 61,227,952 B | `a0d1e354d282c4ed458b83b287065587979b053e` |
| `dettato-evidence/` | Evidence repairer: drops stray notes, tightens note times | 19,731,388 B | `5633f28ee330a48bc31bdd3f9f98bb1a71206141` |
| `dettato-grouper/` | Grouper: notes to named instrument tracks | 11,603,972 B | `5a94cea2329f53228d7bcf1fd6a9f8d9778cc75b` |
| `dettato-drums/` | Drum hit transcriber | 20,195,244 B | `97801f2328e32e4c87b12586aa81121f62cacacb` |

Each folder holds `model.safetensors` and `config.json`, unmodified.

| File | sha256 |
|---|---|
| `dettato/model.safetensors` | `696d22c5341c6e3df1bdb5670cb7945b65aa36f06811dfa5dfe73dc06cd53be5` |
| `dettato-evidence/model.safetensors` | `25f06767a6ad72a9d409701b11ae500bd54622b3d7ddde8dd51152a148a3833a` |
| `dettato-grouper/model.safetensors` | `4d46f18dc59201d4392b5ee31a3e0640ea89268cf1fa7b8b087dcd910193e07c` |
| `dettato-drums/model.safetensors` | `fbdf6243f721dc2e445270671a9361104028d331b6b9d4641ce24102cdef9613` |

## Installation

Install them in `models/muscriptor/` under the active HOT-Step data folder, one
folder per model, as named above. The Model Manager (Shared > MIDI Transcription)
and MIDI Studio's "Download weights" do this, and the registry checks the hashes.
`ace-midi --dettato-model <dir>` takes the folder that holds the four.

## What the port does and does not match

The port runs all four stages: transcriber, repairer, grouper and drums. It is
validated against upstream 2.0.0 on `--device cpu` in float32, with the
evidence repairer on and bypassed; the gates and measured spread are in
`tools/dettato-reference/`. Differences to know about:

- Pitched-note velocity is upstream's `20 + 107 * confidence`: how sure the
  model was of the note, not how hard it was played. Only drum hits have a
  learned velocity.
- Instrument names map to HOT-Step's names; dettato's string-ensemble program
  (44) is written as 48.
- A grouper run on the GPU can place a few notes in a different track than the
  CPU run does, as upstream's own GPU run does. The gate is the measured
  upstream spread.
- Only CUDA and the CPU were measured; Vulkan runs on NVIDIA cards. AMD has not
  been tested on real hardware.

## Credit

Please cite and credit KobiMusic's dettato when you use these models.
