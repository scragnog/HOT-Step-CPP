# Runs mm3-refiner-dit-probe over the slice-1 fixtures (steps 1/8/32, every
# window) and over two extra torch references the fixtures lack: an odd frame
# count (43, from the short clip) and a full-size odd window (2583 frames, from
# the odd clip), both at 64 steps. Every bridge runs twice and must be
# bit-identical. The extra references are captured once with the stock
# bridge_sample into the ignored artifacts/dit-probe/extra/ and reused.
#
#   -Mode f32   CUDA (or the best GPU) with cuBLAS TF32 disabled through
#               NVIDIA_TF32_OVERRIDE=0, the precision the fixtures were captured at.
#   -Mode tf32  the backend's default; ggml-cuda enables TF32 on every cuBLAS handle.
#   -Mode cpu   GGML_BACKEND=CPU, short clip and the 43-frame case only.
#
# -ChainMert also runs every steps-32 window end to end on engine conditioning:
# mm3-refiner-mert-probe writes its MERT states (same mode) into the ignored
# artifacts/dit-probe/mert-<mode>/, and the bridge reruns on them against the
# torch final latents. The MERT probe's own gate result is printed, not counted.
#
# Exit code: 0 when every DiT probe run passes, else the first failing one's (1 numerical, 2 error).
param(
    [ValidateSet('f32', 'tf32', 'cpu')] [string] $Mode = 'f32',
    [switch] $ChainMert,
    [string] $Fixtures = 'tools/mm3-refiner/artifacts/fixtures-v1',
    [string] $Gguf = 'tools/mm3-refiner/artifacts/gguf',
    [string] $Probe = 'engine/build/Release/mm3-refiner-dit-probe.exe',
    [string] $Python = '_experiments/mm3-whats-new-2026-11-05/refiner/.venv/Scripts/python.exe'
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path "$PSScriptRoot/../..")
$refiner = "$Gguf/mm3-refiner-refiner-f32.gguf"
$extra = 'tools/mm3-refiner/artifacts/dit-probe/extra'
foreach ($path in $Probe, $Fixtures, $refiner) {
    if (-not (Test-Path $path)) { Write-Error "missing $path" }
}

if (-not (Test-Path "$extra/L2583-steps64/window-00/refined_latents.npy")) {
    if (-not (Test-Path $Python)) { Write-Error "missing $Python (needed once for the extra references)" }
    @'
import json, sys
from pathlib import Path
import numpy as np
import torch
from minimax_music3_latent_refiner.model import MiniMaxMusic3LatentRefiner, bridge_sample
from safetensors.torch import load_file

fixtures, out = Path(sys.argv[1]), Path(sys.argv[2])
spec = json.loads(Path("tools/mm3-refiner/fixtures.json").read_text())["models"]["refiner"]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = "cuda"
model = MiniMaxMusic3LatentRefiner.from_pretrained(str(Path(spec["weights"]).parent), local_files_only=True).to(dev).eval()
norm = load_file(spec["normalization"], device=dev)
mean, std = norm["latent_mean"][None, None], norm["latent_std"][None, None]
seen = []
model.register_forward_hook(lambda m, args, output: seen.append((args[0].clone(), output.clone())))
for name, src, frames, steps in [("L43-steps64", "short/steps-1/window-00", 43, 64),
                                 ("L2583-steps64", "odd/steps-1/window-01", 2583, 64)]:
    d = out / name / "window-00"
    d.mkdir(parents=True, exist_ok=True)
    put = lambda n, t: np.save(d / f"{n}.npy", np.ascontiguousarray(t.detach().cpu().numpy() if torch.is_tensor(t) else t, dtype=np.float32))
    dav = np.load(fixtures / src / "dav_latents.npy")[:, :, :frames]
    mert = np.load(fixtures / src / "mert_normalized.npy")[:, :, :frames]
    clap = np.load(fixtures / src / "clap_vector.npy")
    degraded = (torch.from_numpy(np.ascontiguousarray(dav)).to(dev).transpose(1, 2) - mean) / std
    layers = torch.from_numpy(np.ascontiguousarray(mert)).to(dev)
    seen.clear()
    final = bridge_sample(model, layers[:, 7], layers, degraded, torch.from_numpy(clap).to(dev), steps)
    assert len(seen) == steps
    for n, t in [("dav_latents", dav), ("mert_normalized", mert), ("clap_vector", clap), ("normalized_latents", degraded),
                 ("schedule", torch.linspace(1.0, 0.0, steps + 1, device=dev)), ("dit_forward", seen[0][1]),
                 ("refined_latents", (final * std + mean).transpose(1, 2))]:
        put(n, t)
    for k, (state, velocity) in enumerate(seen):
        put(f"step-{k:02d}", state)
        put(f"velocity-{k:02d}", velocity)
    put(f"step-{steps:02d}", final)
    print(f"captured {name}: {frames} frames, {steps} steps")
'@ | & $Python - $Fixtures $extra
    if ($LASTEXITCODE -ne 0) { Write-Error "extra reference capture failed" }
}

$saved = @{ NVIDIA_TF32_OVERRIDE = $env:NVIDIA_TF32_OVERRIDE; GGML_BACKEND = $env:GGML_BACKEND }
$code = 0
try {
    $env:NVIDIA_TF32_OVERRIDE = if ($Mode -eq 'f32') { '0' } else { $null }
    $env:GGML_BACKEND = if ($Mode -eq 'cpu') { 'CPU' } else { $null }
    $sets = if ($Mode -eq 'cpu') { "$Fixtures/short", "$extra/L43-steps64" } else { $Fixtures, $extra }
    foreach ($set in $sets) {
        & $Probe --refiner $refiner --contract tools/mm3-refiner/fixtures.json --fixtures $set --repeat 1
        if ($LASTEXITCODE -ne 0 -and $code -eq 0) { $code = $LASTEXITCODE }
    }
    if ($ChainMert -and $Mode -ne 'cpu') {
        $mertProbe = Join-Path (Split-Path $Probe) 'mm3-refiner-mert-probe.exe'
        # Windows depend on the audio only, so one MERT pass serves steps 1, 8 and 32.
        foreach ($clip in Get-ChildItem $Fixtures -Directory) {
            $states = "tools/mm3-refiner/artifacts/dit-probe/mert-$Mode/$($clip.Name)"
            # A failed MERT run must not leave an earlier run's states for the bridge to reuse.
            Remove-Item -Recurse -Force $states -ErrorAction SilentlyContinue
            & $mertProbe --mert "$Gguf/mm3-refiner-mert-f32.gguf" --refiner $refiner --contract tools/mm3-refiner/fixtures.json `
                --fixtures "$Fixtures/$($clip.Name)/steps-32" --out $states | Select-String -Pattern 'mert parity'
            # Exit 1 is a parity miss, which still feeds the chain; 2 is an execution error.
            if ($LASTEXITCODE -ge 2) { Write-Output "MERT probe error on $($clip.Name)"; $code = 2; break }
            foreach ($steps in 1, 8, 32) {
                Write-Output "chain $($clip.Name) steps-$steps, engine MERT ($Mode) -> bridge:"
                & $Probe --refiner $refiner --contract tools/mm3-refiner/fixtures.json `
                    --fixtures "$Fixtures/$($clip.Name)/steps-$steps" --mert-dir $states
                if ($LASTEXITCODE -ne 0 -and $code -eq 0) { $code = $LASTEXITCODE }
            }
        }
    }
} finally {
    $env:NVIDIA_TF32_OVERRIDE = $saved.NVIDIA_TF32_OVERRIDE
    $env:GGML_BACKEND = $saved.GGML_BACKEND
}
exit $code
