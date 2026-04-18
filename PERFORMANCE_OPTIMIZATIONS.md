# Performance Optimizations: VRAM → Speed Trade-offs

**Date:** 2026-04-18  
**Status:** Options 1 & 2 implemented, 3–6 pending

The C++ engine uses roughly half the VRAM of the Python hot-step-9000 pipeline. This document catalogues opportunities to trade surplus VRAM for speed.

---

## Option 1: Eliminate Redundant GPU Uploads ✅ Implemented

**Impact:** HIGH · **VRAM cost:** NONE · **Complexity:** LOW

### Problem

Every `evaluate_velocity` call re-uploads 6 constant tensors to the GPU via PCIe:

| Tensor | Size (typical 60s gen) | Uploads per run (RK4 ×12) |
|--------|----------------------|---------------------------|
| `enc_buf` | H_enc × enc_S × N_graph × 4B | 48 |
| `pos_data` | S × N_graph × 4B | 48 |
| `sa_mask_sw` | S² × N_graph × 2B | 48 |
| `sa_mask_pad` | S² × N_graph × 2B | 48 |
| `ca_mask` | enc_S × S × N_graph × 2B | 48 |

This is defensive — the scheduler may alias input buffers as scratch. But these tensors are **constant** within a generation run (only cover-mode context switch modifies `enc_buf`/`ca_mask` once, mid-run).

### Fix

Track a `constants_dirty` flag. Only re-upload when the flag is set (initial upload, or after cover-mode switch). All other calls skip the PCIe transfers entirely.

### Expected Gain

- Euler 12-step: eliminates ~11 × 5 uploads = 55 unnecessary PCIe transfers
- RK4 12-step: eliminates ~47 × 5 uploads = 235 unnecessary PCIe transfers
- Estimated time savings: 5–30ms per run depending on tensor sizes and PCIe bandwidth

---

## Option 2: Pre-Allocate RK4 Temporaries ✅ Implemented

**Impact:** MEDIUM · **VRAM cost:** NONE · **Complexity:** LOW

### Problem

Inside the RK4 per-step loop (lines 584–612), four `std::vector<float>` temporaries are allocated and freed every step:

```cpp
std::vector<float> k1(vt.begin(), vt.end());      // copy 1
std::vector<float> xt_tmp(n_total);                 // alloc
std::vector<float> k2(vt.begin(), vt.end());       // copy 2
std::vector<float> k3(vt.begin(), vt.end());       // copy 3
```

With 12 steps: 48 heap allocations + copies + deallocations.

### Fix

Pre-allocate `k1`, `k2`, `k3`, `xt_tmp` once before the loop. Use `memcpy` to fill them instead of constructing new vectors.

### Expected Gain

- Eliminates heap fragmentation and allocation overhead
- Marginal wall-clock improvement (~1–5ms) but cleaner memory behaviour

---

## Option 3: Co-Resident DiT + VAE (Not Yet Implemented)

**Impact:** MODERATE · **VRAM cost:** ~0.5–1.0 GB · **Complexity:** MEDIUM

### Problem

The pipeline loads DiT → generates → unloads DiT → loads VAE → decodes → unloads VAE. Each load involves reading GGUF from disk and allocating GPU buffers. This adds 1–3 seconds of I/O per generation.

### Proposed Fix

Add a `keep_resident` mode where both DiT and VAE stay in VRAM across jobs. The pipeline would:
1. Load both at startup
2. Skip load/unload between phases
3. Only free on shutdown or when switching models

### Trade-offs

| Pro | Con |
|-----|-----|
| Eliminates 1–3s load time per job | ~1GB more VRAM permanently consumed |
| Faster back-to-back generations | Cannot use the freed VRAM for larger VAE tiles |
| Simpler job lifecycle | Model swap requires full reload |

### Prerequisites

- Expose a `--keep-resident` flag or auto-detect based on available VRAM
- Update `ace-server.cpp` job dispatch to skip load/unload when resident

---

## Option 4: Larger VAE Tile Size (Not Yet Implemented)

**Impact:** MODERATE (for long audio) · **VRAM cost:** ~100–400 MB · **Complexity:** LOW

### Problem

Current defaults: `vae_chunk=256, vae_overlap=64`. Each chunk processes ~3.2s of audio. For a 60s generation, the VAE runs ~20 tiled passes. Each pass has GPU kernel launch overhead and inter-tile blending artifacts.

### Proposed Fix

Expose chunk size as a tunable parameter (already exists as `AceSynthParams::vae_chunk`). Larger chunks reduce the number of passes:

| `vae_chunk` | Audio per chunk | Passes for 60s | VRAM delta |
|-------------|----------------|-----------------|------------|
| 256 (default) | ~3.2s | ~20 | baseline |
| 512 | ~6.4s | ~10 | +~150 MB |
| 1024 | ~12.8s | ~5 | +~400 MB |
| Full (no tiling) | entire track | 1 | +~1 GB+ |

### Trade-offs

- Larger tiles eliminate blending artifacts at tile boundaries
- Diminishing returns past 1024 — kernel launch overhead is amortized
- Very long tracks (120s+) may OOM with full-tile mode

### Prerequisites

- Auto-detect optimal chunk size based on available VRAM after DiT unload
- Add `--vae-chunk` CLI flag to `ace-server`

---

## Option 5: CUDA Graph Capture (Not Yet Implemented)

**Impact:** POTENTIALLY HIGH (10–30%) · **VRAM cost:** ~200–500 MB · **Complexity:** HIGH

### Problem

The DiT forward pass consists of 24 transformer layers, each launching dozens of CUDA kernels (matmul, softmax, layernorm, etc.). Each kernel has ~5–10µs launch overhead. With ~100 kernels per forward: ~0.5–1.0ms of pure launch overhead per NFE.

### Proposed Fix

GGML has experimental CUDA graph support. The DiT graph topology is **identical** across all steps (same shapes, same ops) — a perfect candidate for graph capture:

1. Run one "warm-up" step normally
2. Capture the graph into a CUDA graph
3. Replay the captured graph for all remaining steps

### Trade-offs

| Pro | Con |
|-----|-----|
| 10–30% speedup from eliminated launch overhead | CUDA graphs consume VRAM for the captured kernel state |
| Especially impactful for small batch sizes | Breaks if tensor shapes change (they don't for DiT) |
| Battle-tested in inference engines (TensorRT, vLLM) | GGML's graph capture may have edge cases on newer architectures |

### Prerequisites

- Verify ggml CUDA graph capture works with flash attention and custom ops
- Test on target architecture (Blackwell RTX 5090)
- Measure actual kernel launch overhead to quantify expected gain
- May need `ggml_backend_sched` API changes to support replay mode

---

## Option 6: Full BF16 Weights (Not Yet Implemented)

**Impact:** MINOR · **VRAM cost:** 2–4× weight size · **Complexity:** LOW

### Problem

If using quantized GGUF files (Q4_K, Q8_0), every matmul requires dequantizing weights to BF16/FP16 before the CUDA kernel can multiply. This adds compute overhead proportional to model size.

### Proposed Fix

Provide BF16 GGUF files alongside quantized ones. Users with sufficient VRAM would select the full-precision variant.

### Trade-offs

| Format | DiT VRAM | Dequant overhead | Quality |
|--------|----------|------------------|---------|
| Q4_K_M | ~1.5 GB | Moderate | Slight loss |
| Q8_0 | ~2.5 GB | Minor | Near-lossless |
| BF16 | ~5 GB | None | Lossless |

### Prerequisites

- Generate BF16 GGUF files using existing `quantize` tool
- On modern GPUs (Ampere+), dequant is highly optimized — actual speed gain may be <5%
- Profile before committing: the gain may not justify the VRAM cost

---

## Summary Matrix

| # | Optimization | Impact | VRAM | Status |
|---|---|---|---|---|
| 1 | Skip constant re-uploads | 🟢 HIGH | None | ✅ Done |
| 2 | Pre-allocate RK4 temps | 🟡 MEDIUM | None | ✅ Done |
| 3 | Co-resident DiT + VAE | 🟡 MODERATE | ~1 GB | ⬜ Planned |
| 4 | Larger VAE tiles | 🟡 MODERATE | ~150–400 MB | ⬜ Planned |
| 5 | CUDA graph capture | 🟢 HIGH | ~200–500 MB | ⬜ Research |
| 6 | BF16 weights | 🔵 MINOR | 2–4× | ⬜ Optional |
