# YourMT3+ GGML spike

A feasibility spike, not the port. It runs the YourMT3+ Perceiver-TF encoder
and the 13-stream T5 decoder natively on GGML and checks them against the CPU
reference captures made by `tools/lead-sheet-reference/`.

- `convert.py` writes the encoder, projection, embedding, decoder and LM head
  tensors of the checkpoint to an unquantised f32 GGUF. The spectrogram and
  convolutional pre-encoder are left out: the spike feeds the encoder from the
  captured pre-encoder output.
- `spike.cpp` builds the encoder graph, the multi-channel projection, the
  cross-attention K/V, and a cached decoder step. It runs prefill, the cached
  step and the full greedy decode on one segment batch. It prints one `PARITY`
  row per capture point, the backend placement of each graph, an optional
  per-op profile and timings.

The checkpoint and the converted GGUF are not redistributable. Keep both under
`_experiments/`, never in git.

## Build and run

The tool links the HOT-ggml libraries that the engine build already produced,
so it compiles in seconds and does not rebuild ggml or CUDA. Run it from
PowerShell; Git Bash rewrites MSVC's `/EHsc` flag into a path.

```powershell
$exp = "_experiments/lead-sheet-port-spike"
python tools/lead-sheet-port-spike/convert.py _experiments/lead-sheet-reference/weights/yourmt3/last.ckpt "$exp/yourmt3-f32.gguf"
cmake -S tools/lead-sheet-port-spike -B "$exp/build" -G "Visual Studio 17 2022" -A x64
cmake --build "$exp/build" --config Release
$env:PATH = "$PWD\engine\build\Release;$env:PATH"
$env:SPIKE_BACKEND_DIR = "$PWD\engine\build\Release"
& "$exp/build/Release/yourmt3-spike.exe" "$exp/yourmt3-f32.gguf" _experiments/lead-sheet-reference/captures/v2-vocal --reps 3 --profile
```

Flags: `--cpu` (CPU backend; set `CUDA_VISIBLE_DEVICES=-1` to keep CUDA out),
`--tf32` (drop the strict-F32 matmul hint), `--dec-from-capture` (feed the
decoder the captured pre-decoder states, which isolates it from the encoder),
`--no-eos-stop` (decode all 256 tokens, for worst-case timing),
`--reps N`, `--profile`. `SPIKE_TIES=1` lists every MoE router decision whose
2nd and 3rd expert probabilities are closer than 1e-5.

## Things the full port must keep

- Every matmul carries `GGML_HINT_STRICT_F32`. Without it ggml-cuda runs F32
  GEMMs on TF32 and the encoder error grows to about 1 (one decoded stream
  of 104 diverged on the main capture).
- The T5 feed-forward uses tanh GELU. ggml-cpu's `ggml_gelu` reads an fp16
  table and misses the logit gate by about 8e-2. CUDA's is exact. The spike
  builds the formula from basic ops on the CPU backend.
- Partial RoPE rotates only the first 4 of each head's 16 dimensions and only
  the queries. The rotation width comes from the head count, not the head
  size. Positions are `linspace(-1, 1)` over the latents ("pixel") and
  `0..T-1` over time ("lang"); the frequencies come from the checkpoint.
- LayerNorm is two-pass: `rms_norm(x - mean(x))`, then weight and bias.
  ggml-cuda's `ggml_norm` takes the variance as E[x^2] - mean^2 in one pass;
  the extra noise flipped top-2 router choices whose 2nd and 3rd
  probabilities are a few 1e-6 apart (49 such decisions on the main capture),
  and the encoder missed the gate on four of five captures.
- `max_abs` rejects any non-finite value; `spike --self-test` checks that.
