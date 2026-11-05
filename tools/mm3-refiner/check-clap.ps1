# Runs mm3-refiner-clap-probe over the slice-1 fixtures and gates the 48 kHz
# processor audio, the log-mel input_features and the pooled CLAP vector per
# window on tolerances.clap_processor and tolerances.clap in fixtures.json.
#
#   -Mode cpu   GGML_BACKEND=CPU (the slice gate; the probe may be a CPU-only build).
#   -Mode f32   best backend with cuBLAS TF32 disabled through NVIDIA_TF32_OVERRIDE=0.
#   -Mode tf32  the backend's default; every CLAP GEMM is marked strict F32, so this
#               mode must pass the gate too.
#   -All        also run the steps-8 and steps-32 copies of each window (the CLAP
#               input does not depend on the bridge step count).
#
# Build the probe on its own, no build.cmd:
#   cmake -S engine -B engine/build/clap-probe -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BACKEND_DL=OFF
#   cmake --build engine/build/clap-probe --config Release --target mm3-refiner-clap-probe
#
# Exit code is the probe's: 0 pass, 1 numerical fail, 2 error.
param(
    [ValidateSet('cpu', 'f32', 'tf32')] [string] $Mode = 'cpu',
    [switch] $All,
    [string] $Fixtures = 'tools/mm3-refiner/artifacts/fixtures-v1',
    [string] $Gguf = 'tools/mm3-refiner/artifacts/gguf',
    [string] $Probe = 'engine/build/clap-probe/Release/mm3-refiner-clap-probe.exe'
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path "$PSScriptRoot/../..")
foreach ($path in $Probe, $Fixtures, "$Gguf/mm3-refiner-clap-f32.gguf") {
    if (-not (Test-Path $path)) { Write-Error "missing $path" }
}
$saved = @{ NVIDIA_TF32_OVERRIDE = $env:NVIDIA_TF32_OVERRIDE; GGML_BACKEND = $env:GGML_BACKEND }
try {
    $env:NVIDIA_TF32_OVERRIDE = if ($Mode -eq 'f32') { '0' } else { $null }
    $env:GGML_BACKEND = if ($Mode -eq 'cpu') { 'CPU' } else { $null }
    $extra = @()
    if ($All) { $extra += '--all' }
    & $Probe --clap "$Gguf/mm3-refiner-clap-f32.gguf" --contract tools/mm3-refiner/fixtures.json `
        --fixtures $Fixtures @extra
    $code = $LASTEXITCODE
} finally {
    $env:NVIDIA_TF32_OVERRIDE = $saved.NVIDIA_TF32_OVERRIDE
    $env:GGML_BACKEND = $saved.GGML_BACKEND
}
exit $code
