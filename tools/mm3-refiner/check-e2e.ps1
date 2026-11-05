# Consolidated local refiner checks. Run only with the GPU idle.
# Outputs and captured WAVs live under artifacts/integration (gitignored).
param([switch]$SelfTest, [switch]$SkipProbe, [switch]$DamagedControlsOnly,
      [int]$Port = 8185, [string]$EngineUrl = 'http://localhost:8085')
$ErrorActionPreference = 'Stop'
if ($SelfTest -or $DamagedControlsOnly) {
    # Exercise the actual comparator without starting an engine or loading models.
    $source = Get-Content "$PSScriptRoot/check-integration.ps1" -Raw
    $definition = [regex]::Match($source, "(?s)Add-Type -TypeDefinition @'\r?\n(.*?)\r?\n'@").Groups[1].Value
    if (-not $definition) { throw 'WAV comparator definition not found' }
    Add-Type -TypeDefinition $definition
}
if ($SelfTest) {
    function Wave([int]$rate, [int]$frames, [int]$channels = 2) {
        $w = [WavData]::new(); $w.Rate = $rate; $w.N = $frames; $w.Ch = $channels
        $w.Data = [float[][]]::new($channels)
        for ($c = 0; $c -lt $channels; $c++) { $w.Data[$c] = [float[]]::new($frames) }
        return $w
    }
    $a = Wave 44100 8
    $same = [WavData]::Compare($a, (Wave 44100 8), 8)
    if ($same[0] -ne 0 -or $same[1] -ne 0) { throw 'Equal WAVs did not match' }
    $cases = @(
        @{ wav = (Wave 48000 8); n = 8; name = 'sample rate' },
        @{ wav = (Wave 44100 7); n = 7; name = 'truncated frames' },
        @{ wav = (Wave 44100 9); n = 8; name = 'extra frames' },
        @{ wav = (Wave 44100 8 1); n = 8; name = 'channels' },
        @{ wav = (Wave 44100 8); n = 7; name = 'partial comparison' }
    )
    foreach ($case in $cases) {
        $rejected = $false
        try { $null = [WavData]::Compare($a, $case.wav, $case.n) }
        catch { if ($_.Exception.InnerException -is [ArgumentException]) { $rejected = $true } else { throw } }
        if (-not $rejected) { throw "Accepted mismatched $($case.name)" }
        Write-Host "PASS rejects $($case.name)"
    }
    Write-Host 'PASS comparator self-test: equal audio and five invalid comparisons'
    exit 0
}
if (-not $DamagedControlsOnly) {
    & "$PSScriptRoot/check-integration.ps1" -SkipProbe:$SkipProbe -Port $Port -ParitySteps 1,8,32
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# The native fixtures do not cover the two damaged inputs used in the ablation.
# Their reference WAVs use the same pinned torch package and default 32 steps.
$repo = (Resolve-Path "$PSScriptRoot/../..").Path
$controls = "$repo/_experiments/mm3-whats-new-2026-11-05/refiner/control"
$out = "$PSScriptRoot/artifacts/e2e"
New-Item -ItemType Directory -Force $out | Out-Null
$jobs = @(Invoke-RestMethod "$EngineUrl/jobs" -TimeoutSec 10)
if ($jobs | Where-Object { $_.status -in @('running', 'queued') }) { throw 'Engine has active jobs; retry when idle' }
$failed = 0
foreach ($name in 'degraded', 'degraded_heavy') {
    $body = [IO.File]::ReadAllBytes("$controls/$name.wav")
    $submitted = Invoke-RestMethod "$EngineUrl/mm3/refine?steps=32" -Method Post -ContentType audio/wav -Body $body -TimeoutSec 30
    $deadline = (Get-Date).AddSeconds(180)
    do {
        $job = Invoke-RestMethod "$EngineUrl/job?id=$($submitted.id)" -TimeoutSec 20
        if ($job.status -ne 'running') { break }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)
    if ($job.status -ne 'done') { throw "$name did not finish: $($job | ConvertTo-Json -Compress)" }
    $bytes = (Invoke-WebRequest "$EngineUrl/job?id=$($submitted.id)&result=1" -TimeoutSec 30).Content
    [IO.File]::WriteAllBytes("$out/$name.wav", [byte[]]$bytes)
    $actual = [WavData]::Parse([byte[]]$bytes)
    $reference = [WavData]::Parse([IO.File]::ReadAllBytes("$controls/out/$name.wav"))
    $metrics = [WavData]::Compare($actual, $reference, $actual.N)
    $ok = $metrics[0] -le 0.02 -and $metrics[1] -le 0.01
    if (-not $ok) { $failed++ }
    Write-Host ("{0} damaged control {1}: max_abs={2:N6} rel_rmse={3:N6} (gates 0.02 / 0.01)" -f $(if ($ok) { 'PASS' } else { 'FAIL' }), $name, $metrics[0], $metrics[1])
}
if ($failed) { exit 1 }
Write-Host 'PASS consolidated refiner checks'
exit 0
