# MM3 latent refiner conversion and parity contract

This directory freezes the torch reference for the optional audio refiner before
the MERT, CLAP and DiT engine ports begin. It does not enable an app feature.
The three engine probe targets are deliberately empty entry points and return 0.
Their successful build proves target wiring only.

## Revisions and licences

`fixtures.json` is the authoritative machine-readable contract. It pins dependency
versions, checkpoint/config/processor revisions, local source SHA-256 hashes, the
12 input recipes, windowing, bridge steps and all numerical gates. The reference
package is Apache-2.0 code at revision
`452bf5e0e1b7225e919c1b4bef0026f13283396c` of
[terminusresearch/minimax-music3-latent-refiner-v0.10](https://huggingface.co/terminusresearch/minimax-music3-latent-refiner-v0.10).
The source is read locally without modifications.

| Component | Revision | Weight licence | Distribution |
|---|---|---|---|
| Refiner and normalization | `452bf5e0e1b7225e919c1b4bef0026f13283396c` | MiniMax-Music3 Community License | Local conversion only |
| [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) including remote Python code and feature extractor | `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5` | CC-BY-NC-4.0 | Public pack policy still requires a decision |
| [larger_clap_music](https://huggingface.co/laion/larger_clap_music) checkpoint, config and processor | `a0b4534a14f58e20944452dff00a22a06ce629d1` | Apache-2.0 | Retained; audio tower and projection converted to F32 |
| [DAV encoder/decoder](https://huggingface.co/SimpleTuner/MiniMax-Music-3-Encoder), `audio_vae` | `fce0d00b1ae42ee47874babb8c06fb859eb01443` | MiniMax-Music3 Community License | Reference only; existing engine GGUFs are reused |

The old experiment cache also contains an automatically converted CLAP safetensors
checkpoint at `4189a948e5e2aaa5df3e69da506e54513e977191`. This contract uses the
original `pytorch_model.bin` at the config revision above instead. Both conversion
and capture load the same explicitly hashed checkpoint, never a moving cache ref.

The validated environment is Python 3.12, torch/torchaudio `2.11.0+cu128`,
transformers `4.57.6`, diffusers `0.39.0`, huggingface-hub `0.36.2`, numpy `2.5.4`,
safetensors `0.8.0`, soundfile `0.14.0`, pyloudnorm `0.2.0` and gguf `0.17.1`.
This preserves the existing reference environment, including its diffusers/hub
dependency-version warning. Transformers 5 is not interchangeable with the
pinned MERT remote code. `nnAudio` is not needed because this MERT config disables
the CQT feature extractor.

## GGUF layout and tensor accounting

The initial parity GGUFs are lossless F32. Quantization is outside this contract.
Original torch names and shapes are preserved without squeezing or transposing.
A torch `[out,in]` matrix has GGML dimensions `[in,out]`; convolutions follow the
same dimension reversal. The converter reads every GGUF tensor back and checks
its bytes against the original F32 values. No MERT weight-normalization folding
or derived positional embeddings are baked into the file.

Metadata includes architecture `mm3-refiner-{refiner,mert,clap}`, source revision,
licence, original config, processor config where applicable, layout and the
SHA-256 of this contract. Full inventories in `artifacts/gguf/*-inventory.json`
list every tensor's name, shape, byte count and hash, and every deliberate CLAP
omission. The MERT and refiner conversions contain no omitted source tensors; the CLAP one omits exactly the 207 text-tower and logit-scale tensors listed below.

| Refiner group | Tensors | F32 payload bytes |
|---|---:|---:|
| `blocks.0..11.*`, nine tensors each | 108 | 538,324,992 |
| `proj_in.*` | 2 | 2,755,584 |
| `proj_out.*` | 2 | 393,728 |
| `time_embed.{0,2}.*` | 4 | 3,151,872 |
| `style_proj.{0,2}.*` | 4 | 3,938,304 |
| `task_embed.weight` | 1 | 12,288 |
| `context_embed` | 1 | 3,072 |
| `degraded_in_proj.weight` | 1 | 393,216 |
| `degraded_null`, `style_null`, `mert_null` | 3 | 42,496 |
| `normalization.{latent_mean,latent_std,layer_mean,layer_std}` | 4 | 80,896 |
| Total, 126 checkpoint tensors plus four statistics | 130 | 549,096,448 |

The refiner file `mm3-refiner-refiner-f32.gguf` is **549,104,864 bytes** including
GGUF metadata and alignment. Nonpersistent RoPE frequencies are computed by the
runtime. The trained nulls are retained even though the inference bridge uses
real conditioning. No slice-0 constant is substituted while that decision is
pending. If frozen style is approved, the converter can embed `constants.clap_mean`
from the hashed ablation NPZ after `style_mode` is changed to `frozen_mean` and
the contract revision is updated; the runtime must then explicitly use it.

| MERT group | Tensors | F32 payload bytes |
|---|---:|---:|
| `feature_extractor.conv_layers.*`, seven convolution weights plus first-layer group norm | 9 | 16,801,792 |
| `feature_projection.layer_norm.*` | 2 | 4,096 |
| `feature_projection.projection.*` | 2 | 1,575,936 |
| `encoder.pos_conv_embed.conv.{bias,weight_g,weight_v}` | 3 | 18,877,952 |
| `encoder.layer_norm.*` | 2 | 6,144 |
| `encoder.layers.0..11.*`, sixteen tensors each | 192 | 340,217,856 |
| `masked_spec_embed`, retained training-only parameter | 1 | 3,072 |
| Total | 211 | 377,486,848 |

`mm3-refiner-mert-f32.gguf` is **377,505,664 bytes**. The positional convolution's
`weight_g` has torch shape `[1,1,128]`, `weight_v` is `[768,48,128]`. Preserve their
weight-normalization dimension when reconstructing the convolution.

CLAP accounting is 555 source tensors: 348 `audio_model.*` and
`audio_projection.*` tensors, 274,742,884 F32 payload bytes, and 207 explicitly
excluded text-tower/projection or training logit-scale tensors. The contract
records `decision: retained` (the frozen-mean style constant was rejected), so
`convert.py --components clap` emits `mm3-refiner-clap-f32.gguf`, architecture
`mm3-refiner-clap`: **274,780,768 bytes**, 348 tensors, 274,742,884 payload
bytes, byte-exact readback verified, the 207 omissions listed in
`artifacts/gguf/clap-inventory.json`. The converter still refuses `--components
clap` while the decision is not `retained`.

## Capturing and comparing

Run commands from the checkout root using the existing local reference venv:

```powershell
$python = '_experiments/mm3-whats-new-2026-11-05/refiner/.venv/Scripts/python.exe'
& $python tools/mm3-refiner/convert.py
& $python tools/mm3-refiner/capture-reference.py --out tools/mm3-refiner/artifacts/fixtures-v1
& $python tools/mm3-refiner/capture-reference.py --out tools/mm3-refiner/artifacts/fixtures-v1 --reload
```

All artifacts stay in ignored `tools/mm3-refiner/artifacts/`. The first short-clip
trial in `artifacts/fixtures/` predates the finalized odd-clip recipe; use
`fixtures-v1`, whose manifests bind to the final contract hash. `--clips short
odd` selects a subset. Existing finalized clips are hash-verified before being
skipped; partial clips have no manifest and are recaptured. `--reload` checks
every tensor's shape, dtype, finiteness and hash and every WAV's hash, rate,
stereo channels and original sample count. Capture checks all pinned source
hashes and environment versions before loading models from local directories.

The corpus is the ten native genre/seed renders named in `fixtures.json`, plus
two deterministic prefixes of the first render. `short` has 22,050 samples,
0.5 s. `odd` has 1,324,033 samples and crosses the rounded 30 s window boundary;
after padding it also has an odd latent frame count. All inputs are padded to
the 512-sample DAV hop before the stock pipeline, then trimmed to the original
length after decode, following the existing runner's workaround for the
upstream final-window coverage bug.

Each clip manifest records its source WAV hash and binds to `fixtures.json`.
For each of 1, 8 and 32 bridge steps, each window contains:

- DAV latents `[1,128,L]`, input waveform and the F32 linspace schedule.
- MERT processor input audio at 24 kHz and normalized `input_values`, all 13
  interpolated hidden states `[1,13,L,768]`, and all 13 states after per-layer
  statistics normalization.
- CLAP processor input audio at 48 kHz, all processor tensors, and the pooled
  style vector `[1,512]`. The stock path uses the center crop up to 10 s and
  passes only `input_features` into `get_audio_features`.
- Normalized DAV latents `[1,L,128]`, one DiT forward at t=1, every velocity,
  initial bridge state and every subsequent Euler state through t=0.
- Each window's overlap weights, the aggregate weights, sample window starts,
  stitched output latents, raw decoded audio and final gain-matched PCM_24 WAV.

Captured tensors are little-endian NPY in torch C order. Intermediate bridge
states come from a forward hook on the unmodified stock bridge; each is the
actual input minus its F32 schedule delta times the actual velocity. The final
windowed latents are independently reconstructed from the captured windows and
weights and required to equal the stock stitched latents bit for bit.
The final audio uses the runner's integrated-LUFS gain match, clips to [-1,1],
and trims the zero pad. There is no automatic alignment, transposition, gain
matching or shape inference in the parity checker.

```powershell
& $python tools/mm3-refiner/check-parity.py engine.npy tools/mm3-refiner/artifacts/fixtures-v1/short/steps-8/window-00/mert_normalized.npy --stage mert
# Raw little-endian F32 also works, with explicit reference dimension order:
& $python tools/mm3-refiner/check-parity.py engine.f32 reference.npy --stage dit --shape 1 44 128
```

The checker requires max-absolute error, relative RMSE and cosine to pass
simultaneously. Relative RMSE is `norm(engine-reference)/max(norm(reference),1e-30)`.
Matching all-zero tensors have cosine 1, and a one-sided zero has cosine 0.
It rejects empty/non-finite arrays and mismatched shapes. Exit 0 means pass,
exit 1 means numerical failure; malformed inputs raise an error and exit nonzero.
Bounds in `fixtures.json` were fixed before engine porting. They are proposed
engineering gates, not measured engine accuracy, and need reviewer agreement
and a contract revision if changed. Apply them to every tensor/window/clip;
whole-corpus averages do not replace a failing boundary.

## Probe build

The issue authorizes one configure and build of only these stubs, with the
existing shared build configuration. It does not rebuild or relaunch the app:

```powershell
cmake -S engine -B engine/build
cmake --build engine/build --config Release --target mm3-refiner-mert-probe mm3-refiner-clap-probe mm3-refiner-dit-probe -j 4
```

Each target links `yyjson` and uses `link_ggml_backends`, following
`heartcodec-probe`. Later port slices replace only their own source file.
An exit-0 stub is never a numerical parity result.

Local validation captured and reloaded 4,182 tensors and 36 final WAVs across all
12 clips. NPY payloads occupy 25,597,383,150 bytes including headers. The odd
case has 2,587 stitched latent frames. Both GGUF inventories passed byte-exact
readback. All three stubs configured, linked and exited 0. The checker passed
identity/raw-F32 comparisons and rejected drift, independent-limit failures,
non-finite tensors and bad shapes. Python compilation and the docs check passed;
the docs check reported 12 existing missing-screenshot warnings. Engine smoke
passed its round, odd and HeartMuLa cases with the GPU idle.

## CLAP port check

`engine/src/minimax/mm3-refiner-clap.h` ports the invoked path only: centre crop of
at most 10 s, torchaudio-style 48 kHz resample, the stock processor (repeatpad,
slaney log-mel), HTSAT, pooling, projection and L2 normalisation; no feature fusion
and no text tower. `check-clap.ps1` runs `mm3-refiner-clap-probe` over the
fixtures (steps-1 windows, `-All` for every copy) and gates four things, each on
the bound frozen in `fixtures.json`: the 48 kHz audio and the log-mel computed from
the reference's own 48 kHz audio on `clap_processor`, the log-mel of the engine's
own resampled audio on `clap_full_chain_mel`, and the pooled vector on `clap`.
The full-chain mel has its own, wider bound (contract revision 2, agreed by the
reviewer): torchaudio resamples in float32, and its 1-2 ulp noise moves the
near-silent top mel bins of band-limited renders by up to 3.9e-3 dB, above the
1e-4 dB processor bound, while the vector moves by 1e-7. Measured worst over 90
windows: 3.937e-3 max_abs, 3.5e-6 relative RMSE, cosine 1. The probe builds
alone, with no `build.cmd`:

```powershell
cmake -S engine -B engine/build/clap-probe -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BACKEND_DL=OFF
cmake --build engine/build/clap-probe --config Release --target mm3-refiner-clap-probe
pwsh -NoProfile -File tools/mm3-refiner/check-clap.ps1
```

Changing `decision` in `fixtures.json` changes the contract hash. The slice-1
fixture manifests and the refiner and MERT GGUF metadata still carry the earlier
hash, so `capture-reference.py --reload` needs a recapture (or a manifest rebind)
before it runs again; the engine probes do not read that hash.
