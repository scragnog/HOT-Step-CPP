---
license: apache-2.0
pipeline_tag: text-to-audio
tags:
  - gguf
  - heartmula
  - music
base_model:
  - HeartMuLa/HeartMuLa-oss-3B
  - HeartMuLa/HeartMuLa-oss-3B-happy-new-year
  - HeartMuLa/HeartMuLa-RL-oss-3B-20260123
  - HeartMuLa/HeartCodec-oss-20260123
---

# HeartMuLa GGUF components for HOT-Step

Split GGUF conversions of the HeartMuLa Team's language models and HeartCodec,
for native C++/GGML inference in HOT-Step. Python is used for conversion only.
These files are published in [scragnog/HeartMuLa-GGUF](https://huggingface.co/scragnog/HeartMuLa-GGUF).

## Sources and license

The original authors are the HeartMuLa Team. The source model repositories
declare Apache 2.0:

- [Base 3B](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B)
- [Happy-new-year 3B](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B-happy-new-year)
- [RL 3B](https://huggingface.co/HeartMuLa/HeartMuLa-RL-oss-3B-20260123)
- [HeartCodec](https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123)
- [Tokenizer and generation configuration](https://huggingface.co/HeartMuLa/HeartMuLaGen)
- [Original implementation](https://github.com/HeartMuLa/heartlib)

The GGUF conversion changes tensor layout, splits components and reduces storage
precision. K/IQ/FP4 LM weights use a separate importance matrix for each variant.
Keep the upstream license and attribution with redistributed files.

## Components and installation

All repository weights live at the repo root. Install them in
`models/heartmula/` under the active HOT-Step data folder. The Model Manager
handles that destination automatically.

| Component | Files | Needed for |
|---|---|---|
| Global LM | `heartmula-3b[-happy-new-year\|-rl]-global-<quant>.gguf` | Generation, one variant |
| Local LM | Matching `-local-<quant>.gguf` | Generation, same variant as global |
| Codec flow | `heartcodec-flow-F16.gguf` or `heartcodec-flow-Q8_0.gguf` | Generation |
| Waveform decoder | `heartcodec-scalar-decoder-F16.gguf` | Generation |
| RVQ | `heartcodec-rvq-F32.gguf` | Shared by generation and encoding |
| Tokenizer | `heartmula-tokenizer.gguf` | Shared by all LM variants and quant levels |
| Optional encoder pack | `heartcodec-{muencoder,wavlm,whisper,query}-F16.gguf` | Audio encoding, never loaded for text generation |

Each global/local part has F16 and 18 quant levels: Q8_0, Q6_K, Q5_K_M,
Q5_K_S, Q4_K_M, Q4_K_S, Q3_K_L, Q3_K_M, Q3_K_S, Q2_K,
IQ4_XS, IQ3_XXS, IQ2_XS, IQ2_XXS, IQ1_M, IQ1_S, MXFP4 and NVFP4.
Global and local precision can differ; their variant must match.

The scalar decoder has no quantizable 2-D matrices, so a Q8_0 copy would
duplicate F16 without reducing size. RVQ stays F32. The installed encoder
pack is F16; encoder Q8_0 files and a separate scalar encoder are absent
from this staging inventory. They need conversion and verification before
they can be added, with measured sizes and SHA-256 values.

## Use and limits

Switch to HeartMuLa in HOT-Step, choose matching global/local weights and
enter concise musical tags and lyrics. Supported duration is up to 240 seconds.
Start with the Q8_0 generation pack for the desired variant, or use F16 as a
reference if memory allows. No audio-quality ranking between variants is claimed.

The smallest LM rungs trade precision for memory. IQ1_S produced clipping in
one fixed-seed validation render, so it is an experimental memory option.
Long prose in the tags field can produce poor output in the original model too.
Native inference does not need the optional encoder pack for text generation.

The publication manifest contains exact paths, byte sizes and SHA-256 values.
Download these components through the HeartMuLa tab in the HOT-Step Model Manager.

## MuLaCover

[MuLaCover by MuLa Labs](https://github.com/HeartMuLa/MuLaCover) is a music
cover generation model. The two `heartmula-mulacover-*-BF16.gguf` files are
conversions of the [source weights](https://huggingface.co/HeartMuLa/MuLaCover).
The safetensors weights were converted to GGUF, split into global and local
parts, and stored as BF16.

These two files are not covered by the Apache 2.0 licence used for the rest of
this repository. The MuLaCover weights are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
The [MuLaCover model terms](MODEL_LICENSE-mulacover) also restrict generated
outputs to non-commercial use. Commercial use requires a licence from MuLa Labs.
