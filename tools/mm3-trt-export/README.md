# MM3 TensorRT DiT exporter

Dev-only Python that produces the two files a user downloads to run the native
MiniMax-Music3 DiT on TensorRT:

| File | Size | Shipped | What it is |
|---|---|---|---|
| `mm3-dit-trt.onnx` | ~2 MB | yes | The DiT forward graph. Structure only — no real weights. |
| `mm3-dit-trt.engine.json` | ~123 KB | yes | GGUF tensor name → ONNX/TensorRT weight name, for all 368 DiT parameters. |
| `mm3-dit-trt.onnx.data` | 4,863,811,584 B | **no** | The external-data file the graph names. Synthesized locally, then deleted. |

They install to `models/mm3/`. The user's machine builds the actual TensorRT
engine from the ONNX on first use (`engine/src/minimax/mm3-dit-trt.h`), because
a serialized engine is locked to one GPU architecture and one TensorRT version
and can never be a download.

## Why the ONNX is weightless

`mm3-dit-trt.h` refits **all 368** DiT parameters out of the selected GGUF
every time it loads an engine. It never reads a value from the ONNX. The graph
is needed for its node structure, its initializer names, shapes and dtypes, and
its five inputs / one output — nothing else.

So shipping real weights would mean asking every user to download 4.86 GB of
BF16 numbers that are overwritten microseconds after the engine deserializes,
on top of the GGUF they already have. Instead the module is built with
throwaway parameters, and the values are dropped at export time. The graph
keeps an ordinary external-data layout — it still names a 4.86 GB
`mm3-dit-trt.onnx.data` — but that file is **generated on the machine that
needs it and deleted afterwards**, never downloaded.

### The external-data layout (the C++ contract)

`build()` in `mm3-dit-trt.h` has to write this file before it hands the ONNX to
the parser, and remove it after. The layout:

- **Which tensors.** Every initializer of type BFLOAT16 with rank ≥ 1: 368 of
  the graph's 381. The other 13 (11 INT64 shape constants and two rank-0 BF16
  scalars, `val_8` and `val_137`) stay inline in the `.onnx` and must not be
  touched — their values are real.
- **Order.** `graph.initializer` order, i.e. the order they appear in the
  protobuf.
- **Offsets.** Running offset starting at 0; after each tensor it advances by
  `length` rounded **up to a 4096-byte boundary**. So every tensor starts
  4096-aligned, ranges are disjoint, and small alignment gaps are never read.
- **Length.** `count * 2` bytes (BF16), where `count` is the product of the
  initializer's dims.
- **Total.** `4863811584` bytes for the current export. Derive it rather than
  hardcoding it: it is `max(offset + length)` over the external initializers,
  and both `export_mm3_dit_onnx.py` and `make_placeholder_data.py` print it
  (`external_data_required_bytes`). Writing more bytes than that is harmless;
  writing fewer is not.

**The generator does not need any of that.** The placeholder value is a pure
function of its own byte offset, so C++ only needs the total size and this
loop — element index `k = byte_offset / 2`, little-endian `uint16` out:

```cpp
static uint16_t placeholder_bf16(uint32_t k) {
    uint32_t h = k;                 // lowbias32 avalanche
    h ^= h >> 16;  h *= 0x7FEB352Du;
    h ^= h >> 15;  h *= 0x846CA68Bu;
    h ^= h >> 16;
    uint16_t bits = uint16_t(0x3C00u + ((h >> 7) & 0x1FFu));
    return uint16_t(bits | uint16_t(((h >> 31) & 1u) << 15));
}
```

`make_placeholder_data.py` is the same function in numpy, chunked; it writes
the 4.86 GB file in about 26 s. It exists for tooling and verification — the
runtime path is the C++ one.

### Why the placeholder cannot be zeros, or constant, or repetitive

Three separate traps, each of which bit us:

1. **Zero-initialised *parameters* break the export.** The ONNX optimizer folds
   an all-zero `Gemm` bias away, so a zeros export comes out with 379
   initializers instead of 381: `time_embed.0.bias` and `time_embed.2.bias`
   vanish and the refit has nowhere to put them. Hence `--init randn`; the
   values are discarded anyway, only the structure survives.
2. **A shared blob breaks TensorRT.** Pointing every initializer at offset 0 of
   one 64 MiB zero blob is legal ONNX — nothing requires external-data ranges
   to be disjoint, and `onnx.checker` and `onnx.load` both accept it — but the
   TensorRT parser keys weights by data pointer and value and rejects the
   network:

   ```
   INetworkDefinition::setWeightsName: Error Code 3: API Usage Error
   (condition: wtsPtr->count() == w.count. Weights of same values but of
   different counts are used in the network.)
   ```

   Hence disjoint per-tensor ranges.
3. **A weakly-mixed pattern still collides.** A plain multiplicative hash
   (`k * 2654435761`) leaves the low 16 output bits a function of the low 16
   input bits, so the byte pattern repeats every 256 KB; against 4096-aligned
   offsets that produced only 61 distinct byte sequences across the 368
   tensors. The avalanche mix above gives all 368 distinct content, no zeros,
   and BF16 magnitudes in [0.0078125, 0.1245] — nothing denormal, NaN or Inf.

The `--compare` graph diff is what catches trap 1, and the GPU verify script
catches traps 2 and 3. Run both.

## The interpreter

There is no requirements file for this; it runs against the environment the
original export used:

- `D:/Ace-Step-Latest/mm3-weights/.venv-ref/Scripts/python.exe` — Python 3.12.10,
  torch 2.11.0+cu128
- `onnx` 1.22.0 and `onnxscript` 0.7.2 come from
  `_experiments/2026-09-11-mm3-speed/native-renderer-probe/export-deps` via
  `PYTHONPATH` (they are deliberately not installed into the venv).

Set `PYTHONIOENCODING=utf-8` on Windows or the exporter's progress output dies
on a legacy code page. The scripts force `CUDA_VISIBLE_DEVICES=""` before torch
loads: exporting is CPU work and must never take a slice of a busy GPU. Peak
RSS is around 11 GB for the weightless export.

## Regenerating

```powershell
$py   = 'D:/Ace-Step-Latest/mm3-weights/.venv-ref/Scripts/python.exe'
$deps = 'D:/Ace-Step-Latest/hot-step-cpp/_experiments/2026-09-11-mm3-speed/native-renderer-probe/export-deps'
$out  = 'D:/Ace-Step-Latest/hot-step-cpp/_experiments/2026-09-11-mm3-speed/weightless-onnx'
$env:PYTHONPATH = "$deps;D:/Ace-Step-Latest/hot-step-cpp/tools/mm3-trt-export"
$env:PYTHONIOENCODING = 'utf-8'

& $py tools/mm3-trt-export/export_mm3_dit_onnx.py `
    --weights none --out-dir $out `
    --compare $out/../native-renderer-probe/mm3-full-bf16.onnx

& $py tools/mm3-trt-export/make_manifest.py `
    --onnx $out/mm3-dit-trt.onnx `
    --gguf models/mm3/mm3-dit-f16.gguf `
    --output $out/mm3-dit-trt.engine.json `
    --compare models/mm3/mm3-dit-trt.engine.json
```

The exporter writes the placeholder `.onnx.data` as it goes so its own checks
can read it; pass `--no-data` to skip that, and regenerate on demand with

```powershell
& $py tools/mm3-trt-export/make_placeholder_data.py --onnx $out/mm3-dit-trt.onnx
```

Publish `mm3-dit-trt.onnx` and `mm3-dit-trt.engine.json`. Never publish
`mm3-dit-trt.onnx.data`.

`--weights <path-to-flowmatching_vae.pth>` reproduces the original full BF16
export instead — same layout, real values, and then the `.onnx.data` *is* the
weights, for A/B work only.

**The manifest belongs to one specific ONNX.** The dynamo exporter folds
`Linear` weights into pre-transposed constants named `val_<n>` by graph
position, so those names are an artefact of that export. Regenerate the two
files together, and publish them together.

## Checks the tools run

`export_mm3_dit_onnx.py`

- `onnx.checker.check_model` (structural). `full_check=True` is *not* used: ONNX
  shape inference rejects BF16 `Conv` at opset 18, and the full-weight export
  fails the same check. TensorRT's parser accepts both.
- `onnx.load(load_external_data=True)` — all 381 initializers materialise, which
  proves the offsets in the graph match the file that was written.
- `--compare` — node count, per-op-type node counts, graph inputs/outputs, opset
  and IR version, and every initializer's name, dtype and shape.

`make_manifest.py`

- Resolves each module parameter to its ONNX initializer: by name when the
  exporter kept it, or through the owning MatMul's
  `pkg.torch.onnx.name_scopes` metadata when it was folded into a `val_<n>`
  transpose.
- Reads the GGUF header directly (no `gguf` package) and checks every tensor
  exists with the expected `ne` — ggml stores dimensions in the reverse of the
  torch order, which is exactly the mistake this check exists to catch.
- Asserts 368 entries and that they cover every `dit.*` tensor in the GGUF
  except `dit.rope_inv_freq` and `dit.time_fourier.weight`, which the renderer
  computes itself. That is the same assertion `mm3-dit-trt.h` makes at load
  time, made early.
- `--compare` reports semantic equality against an existing manifest. Entry
  order will differ: the manifest at `models/mm3/mm3-dit-trt.engine.json` was
  written in TensorRT refitter enumeration order, which needs a built engine.
  Order carries no meaning — the refit loop looks entries up by name — and
  `byte_identical_after_reorder` confirms the content is otherwise the same
  file.

## Verifying on a GPU

`verify_weightless_build.ps1` builds an engine from the weightless ONNX with
the probe's builder, refits it from the F16 DiT GGUF, replays the 273-frame
saved plan, and compares the audio byte-for-byte against the prepared-engine
baseline. It is the only GPU step; read the header comment before running it.

## FP8 lab (`fp8_lab.py`)

Dev-only companion for the opt-in FP8 MLP engine (`MM3_DIT_TRT_FP8=1`). It
parses the same weightless ONNX with the TensorRT Python wheel that matches the
engine (10.16.1.11, plus numpy and cuda-python; no torch or onnx), refits from
a GGUF through the manifest, and builds the same variants the engine builds.
Commands: `placeholder`, `inspect`, `build`, `time` (with a per-layer
profile), `calib` and `compare`; the docstring lists the flags.

The FP8 calibration file comes from BF16 renders with
`MM3_DIT_TRT_DUMP=<dir>` set on the engine (one dump directory per render; the
engine numbers calls across renders, so move them out between jobs): `calib
--dump <dir> [<dir>...] --gguf <dit> --out mm3-dit-trt.fp8.json` takes the max
over all dumps, reduced on the GPU (~3 min for six 60 s renders), and keeps each
dump's amax under `per_dump`. The shipped file is
`server/src/data/mm3-dit-trt.fp8.json`; the engine falls back to
`<models>/mm3/mm3-dit-trt.fp8.json` when a request carries no `dit_fp8_calib`. `compare` on two
dumps of the same seed gives the per-window latent parity. Results are in the
`mm3-backend` skill, `reference/performance.md`.
