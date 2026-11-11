param([string]$FixtureRoot = 'D:\Ace-Step-Latest\hotstep-1148-fixture')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$clone = Join-Path $FixtureRoot 'source'
$logs = Join-Path $PSScriptRoot 'evidence'
New-Item -ItemType Directory -Force $logs | Out-Null
if (Test-Path $clone) { throw "Refusing to overwrite existing clone: $clone" }
New-Item -ItemType Directory -Force $FixtureRoot | Out-Null
git clone --no-hardlinks --single-branch --branch master $repo $clone
if ($LASTEXITCODE) { throw 'Clone failed' }
function CopyTree($source, $target) {
    robocopy $source $target /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copy failed: $source" }
}
CopyTree "$repo\server\node_modules" "$clone\server\node_modules"
CopyTree "$repo\ui\node_modules" "$clone\ui\node_modules"
CopyTree "$repo\release\node_modules" "$clone\release\node_modules"
CopyTree "$repo\release\.node-cache\node-v24.18.0-win-x64" "$clone\release\.node-cache\node-v24.18.0-win-x64"
$engineOut = "$clone\engine\build\Release"
New-Item -ItemType Directory -Force $engineOut | Out-Null
foreach ($name in @('ace-server.exe','ace-train.exe','ace-midi.exe','ace-caption.exe','mastering.exe','mp3-codec.exe','neural-codec.exe','vst-host.exe','quantize.exe','ggml.dll','ggml-base.dll','ggml-cpu.dll')) {
    Copy-Item -LiteralPath "$repo\engine\build\Release\$name" -Destination $engineOut
}
Get-ChildItem "$repo\engine\build\Release\ggml-cpu-*.dll" | Copy-Item -Destination $engineOut
foreach ($version in @('91.0.0','91.0.1','91.0.2')) {
    git -C $clone tag "v$version"
    if ($LASTEXITCODE) { throw 'Fixture tag failed' }
    $argsList = @('-NoProfile','-File',"$clone\release\build-release.ps1",'-Version',$version,'-Variant','cpu','-SkipEngine')
    if ($version -ne '91.0.2') { $argsList += @('-UpdaterProtocol','1') }
    if ($version -ne '91.0.0') { $argsList += '-SkipUI' }
    $p = Start-Process pwsh -ArgumentList $argsList -PassThru -WindowStyle Hidden -RedirectStandardOutput "$logs\build-$version.stdout.log" -RedirectStandardError "$logs\build-$version.stderr.log"
    if (-not $p.WaitForExit(900000)) { Stop-Process -Id $p.Id; throw "Build timeout $version, PID $($p.Id)" }
    if ($p.ExitCode -ne 0) { throw "Build $version failed with $($p.ExitCode), see evidence logs" }
    Write-Output "BUILD PASSED $version"
}
