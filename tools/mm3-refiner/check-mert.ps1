# Runs mm3-refiner-mert-probe over the slice-1 fixtures and gates all 13
# normalised MERT states per window on tolerances.mert in fixtures.json.
#
#   -Mode f32   CUDA (or the best GPU) with cuBLAS TF32 disabled through
#               NVIDIA_TF32_OVERRIDE=0, the precision the fixtures were captured at.
#   -Mode tf32  the backend's default, no override. ggml-cuda runs F32 GEMMs on
#               TF32, but mm3-refiner-mert.h marks every MERT GEMM strict F32
#               (GGML_HINT_STRICT_F32), so this mode must pass the gate too.
#   -Mode cpu   GGML_BACKEND=CPU.
#
# Exit code is the probe's: 0 pass, 1 numerical fail, 2 error.
param(
    [ValidateSet('f32', 'tf32', 'cpu')] [string] $Mode = 'f32',
    [string] $Fixtures = 'tools/mm3-refiner/artifacts/fixtures-v1',
    [string] $Gguf = 'tools/mm3-refiner/artifacts/gguf',
    [string] $Probe = 'engine/build/Release/mm3-refiner-mert-probe.exe'
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path "$PSScriptRoot/../..")
foreach ($path in $Probe, $Fixtures, "$Gguf/mm3-refiner-mert-f32.gguf", "$Gguf/mm3-refiner-refiner-f32.gguf") {
    if (-not (Test-Path $path)) { Write-Error "missing $path" }
}
$saved = @{ NVIDIA_TF32_OVERRIDE = $env:NVIDIA_TF32_OVERRIDE; GGML_BACKEND = $env:GGML_BACKEND }
try {
    $env:NVIDIA_TF32_OVERRIDE = if ($Mode -eq 'f32') { '0' } else { $null }
    $env:GGML_BACKEND = if ($Mode -eq 'cpu') { 'CPU' } else { $null }
    & $Probe --mert "$Gguf/mm3-refiner-mert-f32.gguf" --refiner "$Gguf/mm3-refiner-refiner-f32.gguf" `
        --contract tools/mm3-refiner/fixtures.json --fixtures $Fixtures
    $code = $LASTEXITCODE
} finally {
    $env:NVIDIA_TF32_OVERRIDE = $saved.NVIDIA_TF32_OVERRIDE
    $env:GGML_BACKEND = $saved.GGML_BACKEND
}
exit $code
