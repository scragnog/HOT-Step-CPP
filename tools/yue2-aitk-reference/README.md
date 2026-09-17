# YuE2 AI Toolkit reference tools

These tools support the native YuE2 training port. They are development tools,
not a new selectable trainer. The engine, Legacy workflow and training default
are unchanged. Python is required for reference comparisons, not for the native
C++/CUDA probes themselves.

## Native ConvRot8 computation

`convrot_cpu.h` and `convrot_probe.cpp` provide a scalar arithmetic oracle.
`convrot_cuda.cu` implements rotation, activation quantization, actual int8
cuBLAS multiplication, output scaling and straight-through input gradients.
It is a standalone prototype; it has not been integrated into GGML autograd.

Build from a Visual Studio x64 Developer Command Prompt at the repository root:

```bat
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
tools\yue2-aitk-reference\build-probes.cmd
```

The default target is `sm_120` for the test machine's RTX 5090. Set
`AITK_CUDA_ARCH` for another supported device. GPU tests must use the project's
Work reservation guard. This command uses the local AI Toolkit environment:

```powershell
& tools/mcp-lyricstudio/work-run.ps1 --name AITK-convrot-check --resource gpu `
  --reason 'ConvRot reference comparison' --cwd D:\Ace-Step-Latest\hot-step-cpp -- `
  D:\Ace-Step-Latest\ai-toolkit\venv\Scripts\python.exe `
  tools/yue2-aitk-reference/convrot_fixture.py `
  --toolkit D:\Ace-Step-Latest\ai-toolkit `
  --probe _experiments/aitk-port/bin/convrot_cuda.exe `
  --device cuda --production-shapes --output _experiments/aitk-port/new-convrot-check
```

Choose a new output directory for each run. CPU mode is the default and does
not need a GPU reservation. CUDA mode calls the pinned Toolkit's real Triton
quantizer and custom linear/autograd operator, rather than a PyTorch fallback.

The binary fixture uses little-endian uint32 header fields: magic `0x314B5441`,
rows, input width, output width, rotation group and BF16 flag. Payload arrays
are row-major: FP32 input, int8 rotated weights, FP32 weight scales, FP32 output
gradient and FP32 bias. BF16 values are widened losslessly to FP32 for transport.
The output contains FP32 rotated input, int8 activation codes, FP32 activation
scales, FP32 output and FP32 input gradient, in that order.

Verified on 2026-09-17: 11 cases pass against Torch 2.9.1+cu128 with
triton-windows 3.5.1.post24 on the 5090. BF16 checks use exact equality; FP32
checks use a tensor-scale absolute bound of `2e-6 + 2e-6 * max(abs(reference))`.
Actual projection cases include 2048→4096, 2048→12288 and 6144→2048, plus
non-aligned row counts, zero rows and halfway rounding values. This establishes
these isolated operations, not a whole-model or speed result.

Three details mattered at full dimensions:

- Match Triton's emitted `div.full.f32` quantization and fused epilogue arithmetic.
- The backward casts stored scales to BF16 before multiplying int8 codes, and
  the GEMM writes BF16 directly. Writing FP32 and rounding later selected a
  different cuBLAS accumulation path.
- BF16 inverse rotation uses GEMM, as in the reference. A mathematically equal
  radix-4 sum differed at a cancellation tie. The scalar oracle is diagnostic,
  not the authority for reference GPU rounding.

## Stream-ordered CUDA API

The reusable CUDA operation is in `convrot_cuda_api.h/.cu`. It accepts
caller-owned device buffers, workspace, stream and cuBLAS handle; it neither
allocates device memory nor synchronizes. `convrot_api_probe.cpp` checks the
same reference fixtures on a non-default stream and verifies restoration of
the caller's cuBLAS stream, pointer mode and math mode. Pass `--no-bias` to
`convrot_fixture.py` with this probe to test YuE2's bias-free projections.
Both bias modes pass the 11 reference cases, including production dimensions.
This API is not yet registered as a GGML training operation.

## Adapter interchange

`test_fused_lora.cpp` tests the native header
`engine/src/train/yue2-aitk-lora.h`: shared input factors for fused projections,
distinct AR/NAR tensor names, and actual GGML backward accumulation checked
against both separate branches and analytic gradients. Existing Legacy
adapter creation is unchanged.

`adapter_layout.py CHECKPOINT` validates both expert namespaces and fused LoRA
factor shapes. Its fused/split APIs preserve shared input factors and reject
unequal factors when fusing. `--export-split OUTPUT` writes independent storage
with metadata recording the shared-factor requirement. This split namespace is
a development interchange format, not a claim that the current native adapter
loader can consume it. Tests round-trip the real Dookie checkpoint exactly.

## Optimizer fixture

`optimizer_fixture.py --inspect` reports the resolved contract without loading
GPU libraries. `--execute --steps 5 --export NEW_DIRECTORY --toolkit-source PATH`
runs the actual bitsandbytes optimizer; run it under the same GPU guard.
It saves parameters, changing gradients, every state tensor/codebook/block
scale, step counters and source/environment identities. The 4095/4096-element
boundary exercises FP32 versus uint8 optimizer state.

Trainable LoRA parameters default to FP32, matching `network.force_to` in the
reference trainer. BF16 checkpoint export does not change that contract.
`--bf16-diagnostic` is explicitly a separate diagnostic. No native AdamW8bit
implementation is claimed by exporting these reference trajectories.

## Reference manifest

`manifest.py` writes a deterministic JSON inventory for the pinned AI Toolkit
checkout and a local Dookie experiment. It is intentionally read-only and uses
only the Python standard library. It does not import PyTorch or safetensors,
touch CUDA, download models, or inspect checkpoint tensor payloads. Safetensors
inspection reads and validates the header, recording tensor names, dtypes,
shapes, offsets, metadata, size, and SHA-256.

The Git section records the pinned commit/tree, current checkout, upstream delta,
working-tree patch, untracked paths, and patch hashes. This keeps local source
patches distinct from the pinned upstream reference. Package metadata is parsed
from the supplied environment files without importing packages. Config files
are hashed by content.

Example (the paths are inputs; they are not embedded in the exporter):

```powershell
python tools/yue2-aitk-reference/manifest.py `
  --toolkit-root D:\Ace-Step-Latest\ai-toolkit `
  --experiment-root D:\Ace-Step-Latest\ai-toolkit\experiments\dookie-reference-20260916 `
  --checkpoint D:\Ace-Step-Latest\ai-toolkit\experiments\dookie-reference-20260916\output\dookie_yue2_default_300\dookie_yue2_default_300_000000050.safetensors `
  --output _experiments\aitk-port\reference-manifest.json
```

Run the focused malformed-header tests with:

```powershell
python -m unittest discover -s tools/yue2-aitk-reference -p "test_*.py"
```
