# run_experiment_live.ps1 -- live-teacher / flash-attn twin arms, following run_experiment.ps1's
# structure exactly but with a different arm set. See docs/plans/lm-attr-probe/OVERNIGHT.md
# (Live teacher spec) and RESULTS.md #6.
#
# Arms:
#   flash    = { attnBackend: "flash" }                                    (flash-attn twin of ctrl)
#   pplive   = { regEvery: 3, regTeacher: "live" }                         (live-teacher prior preservation)
#   bothlive = { regEvery: 3, regTeacher: "live", captionDropout: 0.3 }    (pplive + caption dropout)
param(
  [string]$Slugs = 'kinks_somethingelse,inxs_kick,nas_illmatic',
  [int]$Samples = 8,
  [int]$MaxDuration = 150,
  [int]$Steps = 8,
  [string]$SynthModel = 'acestep-v15-merge-base-sft-turbo-xl-thirds-BF16.gguf',
  [string]$Inventory = "$PSScriptRoot\..\..\docs\plans\lm-attr-probe\inventory.json",
  [string]$Ledger = "$PSScriptRoot\..\..\docs\plans\lm-attr-probe\train-arm-ledger.json",
  [string]$LogFile = "$PSScriptRoot\..\..\docs\plans\lm-attr-probe\run_experiment_live.log",
  [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$repo = Resolve-Path "$PSScriptRoot\..\.."

$Arms = [ordered]@{
  flash    = '{"attnBackend":"flash"}'
  pplive   = '{"regEvery":3,"regTeacher":"live"}'
  bothlive = '{"regEvery":3,"regTeacher":"live","captionDropout":0.3}'
}

function Write-Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
  Write-Host $line
  if (-not $DryRun) {
    New-Item -ItemType Directory -Force (Split-Path $LogFile) | Out-Null
    Add-Content -Path $LogFile -Value $line
  }
}

function Invoke-Logged([string]$exe, [string[]]$cmdArgs, [string]$what) {
  Write-Log "RUN: $exe $($cmdArgs -join ' ')"
  if ($DryRun) { return }
  & $exe @cmdArgs 2>&1 | ForEach-Object {
    Write-Host $_
    Add-Content -Path $LogFile -Value $_
  }
  if ($LASTEXITCODE -ne 0) {
    Write-Log "FAILED ($what): exit $LASTEXITCODE -- stopping the sweep, re-run this script to resume"
    exit $LASTEXITCODE
  }
}

if (-not (Test-Path $Inventory)) {
  if ($DryRun) { Write-Log "note: $Inventory does not exist yet -- would be built by inventory.py first" }
  else { python "$PSScriptRoot\inventory.py" --json $Inventory | Out-Null }
}
$inv = @{}
if (Test-Path $Inventory) {
  foreach ($a in (Get-Content $Inventory -Raw | ConvertFrom-Json)) { $inv[$a.slug] = $a }
}

function Get-Variant([string]$slug) {
  if ($inv.ContainsKey($slug)) { return $inv[$slug].variant }
  $tensorsDir = Join-Path $repo "server\data\training\tensors\$slug"
  $dirs = Get-ChildItem $tensorsDir -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName 'lm_codes.jsonl') } |
    Sort-Object Name
  if (-not $dirs) { throw "no preprocessed tensor variant with lm_codes.jsonl under $tensorsDir" }
  return $dirs[-1].Name
}

function Get-LedgerRunDir([string]$slug, [string]$arm) {
  if (-not (Test-Path $Ledger)) { return '' }
  $l = Get-Content $Ledger -Raw | ConvertFrom-Json
  $key = "$slug|$arm"
  $entry = $l.PSObject.Properties[$key]
  if ($entry -and $entry.Value.runDir) { return $entry.Value.runDir }
  return ''
}

# Wait for any ace-train.exe to be gone before starting a build/GPU run (the stray
# nirvana-exact job on kinks must not be cancelled -- just wait it out).
function Wait-ForNoAceTrain {
  $waited = $false
  while (Get-Process -Name 'ace-train' -ErrorAction SilentlyContinue) {
    if (-not $waited) { Write-Log "waiting for ace-train.exe to exit before starting..."; $waited = $true }
    Start-Sleep -Seconds 15
  }
  if ($waited) { Write-Log "ace-train.exe gone, proceeding" }
}

$slugList = ($Slugs -split ',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$t0 = Get-Date
Write-Log "== live-teacher/flash experiment start: $($slugList -join ', ') x $($Arms.Keys -join ', ')$(if ($DryRun) { ' [DRY RUN]' })"

foreach ($slug in $slugList) {
  $variant = Get-Variant $slug
  $lmAttrDir = Join-Path $repo "server\data\training\tensors\$slug\$variant\lm-attr"
  $rendersDir = Join-Path $lmAttrDir 'renders'

  foreach ($arm in $Arms.Keys) {
    $opts = $Arms[$arm]
    $t1 = Get-Date
    Write-Log "== $slug / $arm  opts=$opts"

    if (-not $DryRun) { Wait-ForNoAceTrain }

    # -- 1. train ------------------------------------------------------------
    $trainArgs = @('tsx', 'scripts/lm-train-arm.ts', '--dataset', $slug, '--arm', $arm,
                   '--opts-b64', [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($opts)),
                   '--variant', $variant, '--ledger', $Ledger)
    if ($DryRun) { $trainArgs += '--dry-run' }
    Push-Location "$repo\server"
    try { Invoke-Logged 'npx' $trainArgs "train $slug/$arm" }
    finally { Pop-Location }

    $runDir = Get-LedgerRunDir $slug $arm
    if (-not $runDir) {
      if ($DryRun) { $runDir = "<run dir for $slug/$arm, known only after training>" }
      else {
        Write-Log "FAILED: $slug/$arm trained but the ledger has no run dir for it -- check $Ledger"
        exit 1
      }
    }

    if (-not $DryRun) { Wait-ForNoAceTrain }

    # -- 2. plans --------------------------------------------------------------
    $plansDir = Join-Path $lmAttrDir "plans_$arm"
    Push-Location "$repo\server"
    try {
      Invoke-Logged 'npx' @('tsx', 'scripts/lm-adapter-eval.ts', 'generate',
        '--dataset', $slug, '--variant', $variant, '--adapter', $runDir,
        '--seeds', '1', '--samples', "$Samples", '--max-duration', "$MaxDuration", '--out', $plansDir
      ) "plans $slug/$arm"
    } finally { Pop-Location }

    # -- 3. render (through the base DiT, no DiT adapter) --------
    Push-Location "$repo\server"
    try {
      Invoke-Logged 'npx' @('tsx', 'scripts/lm-plan-render.ts',
        '--runs', $plansDir, '--out', $rendersDir, '--steps', "$Steps",
        '--sides', 'adapter', '--label', $arm, '--synth-model', $SynthModel
      ) "render $slug/$arm"
    } finally { Pop-Location }

    # -- 4. attrs --------------------------------------------------------------
    Invoke-Logged 'python' @("$PSScriptRoot\extract_attrs.py", '--renders', '--slug', $slug, '--workers', '6') "attrs $slug/$arm"

    Write-Log ("   {0:n0}s for {1}/{2}  ({3:n0}s total)" -f ((Get-Date) - $t1).TotalSeconds, $slug, $arm, ((Get-Date) - $t0).TotalSeconds)
  }
}

Write-Log "== scoring"
Invoke-Logged 'python' @("$PSScriptRoot\score_renders.py") "score_renders"

Write-Log "== per-arm summary"
Invoke-Logged 'python' @("$PSScriptRoot\summarize_arms.py") "summarize_arms"

Write-Log ("== live-teacher/flash experiment done ({0:n0}s total)" -f ((Get-Date) - $t0).TotalSeconds)
