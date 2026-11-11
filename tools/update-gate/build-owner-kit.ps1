param(
    [string]$FixtureRoot = 'D:\Ace-Step-Latest\hotstep-1148-fixture',
    [string]$SourceCommit = '1ced828d7273b01c9269b8f74f5334c6fc55845c'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$clone = Join-Path $FixtureRoot 'source'
$logs = Join-Path $PSScriptRoot 'evidence/round3'
if ((git -C $clone rev-parse HEAD) -ne $SourceCommit) { throw 'Wrong fixture commit' }
if (-not (Select-String -LiteralPath "$repo\engine\build\CMakeCache.txt" -Pattern 'CUDAToolkit_BIN_DIR:PATH=.*CUDA/v13.1/bin')) { throw 'CUDA asset identity not verified' }
Copy-Item -LiteralPath "$repo\engine\build\Release\ggml-cuda.dll" -Destination "$clone\engine\build\Release\ggml-cuda.dll"
foreach ($version in @('94.0.0','94.0.1')) {
    git -C $clone tag "v$version" $SourceCommit
    if ($LASTEXITCODE) { throw 'Fixture tag failed' }
    $argsList = @('-NoProfile','-File',"$clone\release\build-release.ps1",'-Version',$version,'-Variant','cuda','-CudaAsset','cuda13.1','-UpdaterProtocol','1','-SkipEngine','-SkipUI')
    $p = Start-Process pwsh -ArgumentList $argsList -PassThru -WindowStyle Hidden -RedirectStandardOutput "$logs\build-$version.stdout.log" -RedirectStandardError "$logs\build-$version.stderr.log"
    if (-not $p.WaitForExit(900000)) {
        $daemon = & "$env:MULTICA_CLI_PATH" daemon status --output json | ConvertFrom-Json
        if ($LASTEXITCODE -ne 0 -or $p.Id -eq $daemon.pid) { throw 'Cannot safely stop timed-out build' }
        taskkill /PID $p.Id /T /F | Out-Null
        throw "Build timeout $version, stopped owned tree $($p.Id)"
    }
    if ($p.ExitCode -ne 0) { throw "Build $version failed: $($p.ExitCode)" }
    Write-Output "BUILD PASSED $version cuda13.1 (engine not started)"
}
$env:UPDATE_GATE_ROOT = $FixtureRoot
node "$PSScriptRoot\prepare-owner-kit.mjs"
if ($LASTEXITCODE) { throw 'Owner kit preparation failed' }
