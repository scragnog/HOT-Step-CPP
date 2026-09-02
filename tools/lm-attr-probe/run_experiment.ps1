# run_experiment.ps1 -- the caption-dropout / prior-preservation arm sweep
# (docs/plans/lm-attr-probe/HANDOFF.md, RESULTS.md #4): 3 artists x 4 arms, each arm training a
# fresh planner-LM adapter through the Node server's train-lm job queue, then scored the same way
# as run_a4.ps1 / run_stages.ps1 (rendered through the BASE DiT, no DiT adapter).
#
# Arms (train-lm POST body options on top of the server's own defaults, targetLoss 1.5):
#   ctrl = {}                                   (baseline -- same recipe as every other shipped adapter)
#   pp   = { regEvery: 3 }                      (prior preservation)
#   cd   = { captionDropout: 0.3 }              (caption dropout)
#   both = { regEvery: 3, captionDropout: 0.3 }
#
# Per (artist, arm), sequentially (one GPU, one job at a time):
#   1. train  : lm-train-arm.ts              (skips if the ledger already has a run dir with weights)
#   2. plans  : lm-adapter-eval.ts generate  --adapter <trained run dir>  -> plans_<arm>/
#   3. render : lm-plan-render.ts  --sides adapter --label <arm>          -> renders/ (shared per artist)
#   4. attrs  : extract_attrs.py --renders --slug <slug>
# Then once, over everything: score_renders.py, then summarize_arms.py for the per-arm corpus table.
#
# RESUMABLE: re-running skips any arm already in the ledger (server/scripts/lm-train-arm.ts's own
# ledger) and any plan/render extract_attrs.py already wrote (lm-adapter-eval.ts's runs.json resume
# + lm-plan-render.ts's existing-wav skip + extract_attrs.py's own skip-existing-npz). Safe to kill
# and re-run at any point.
#
# Needs the app up (dev.bat) for steps 1-3 -- the engine is stopped by the SERVER during training and
# by lm-adapter-eval.ts/lm-plan-render.ts's own request lifecycle for generation/render, and comes
# back up after each; this script never touches ace-server directly.
#
#   powershell -File tools/lm-attr-probe/run_experiment.ps1 [-Slugs "a,b,c"] [-Samples 8]
#       [-MaxDuration 150] [-Steps 8] [-SynthModel <name>] [-DryRun]
param(
  [string]$Slugs = 'kinks_somethingelse,inxs_kick,nas_illmatic',
  [int]$Samples = 8,
  [int]$MaxDuration = 150,
  [int]$Steps = 8,
  [string]$SynthModel = 'acestep-v15-merge-base-sft-turbo-xl-thirds-BF16.gguf',
  [string]$Inventory = "$PSScriptRoot\..\..\docs\plans\lm-attr-probe\inventory.json",
  [string]$Ledger = "$PSScriptRoot\..\..\docs\plans\lm-attr-probe\train-arm-ledger.json",
  [string]$LogFile = "$PSScriptRoot\..\..\docs\plans\lm-attr-probe\run_experiment.log",
  [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$repo = Resolve-Path "$PSScriptRoot\..\.."

# Arm -> train-lm options JSON, in the order they run. An ordered hashtable so
# ctrl (the cheap sanity check) always runs before the two-lever combination.
$Arms = [ordered]@{
  ctrl = '{}'
  pp   = '{"regEvery":3}'
  cd   = '{"captionDropout":0.3}'
  both = '{"regEvery":3,"captionDropout":0.3}'
}

function Write-Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
  Write-Host $line
  if (-not $DryRun) {
    New-Item -ItemType Directory -Force (Split-Path $LogFile) | Out-Null
    Add-Content -Path $LogFile -Value $line
  }
}

# Run an external command, tee its output to the log, and stop the whole sweep on a non-zero exit --
# a silently-skipped failure here (a bad --opts, an engine that never came back) would corrupt every
# later resume check that trusts the ledger/plans/renders it was supposed to have written.
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
  # Fallback for a slug not in the (possibly stale) inventory: newest tensor
  # variant dir that actually has lm_codes.jsonl, matching newestVariantKey().
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
  # ConvertFrom-Json turns the ledger's "slug|arm" object keys into
  # NoteProperty names -- PSObject member lookup, not a dictionary.
  $entry = $l.PSObject.Properties[$key]
  if ($entry -and $entry.Value.runDir) { return $entry.Value.runDir }
  return ''
}

$slugList = ($Slugs -split ',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$t0 = Get-Date
Write-Log "== experiment start: $($slugList -join ', ') x $($Arms.Keys -join ', ')$(if ($DryRun) { ' [DRY RUN]' })"

foreach ($slug in $slugList) {
  $variant = Get-Variant $slug
  $lmAttrDir = Join-Path $repo "server\data\training\tensors\$slug\$variant\lm-attr"
  $rendersDir = Join-Path $lmAttrDir 'renders'

  foreach ($arm in $Arms.Keys) {
    $opts = $Arms[$arm]
    $t1 = Get-Date
    Write-Log "== $slug / $arm  opts=$opts"

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

    # -- 2. plans --------------------------------------------------------------
    $plansDir = Join-Path $lmAttrDir "plans_$arm"
    Push-Location "$repo\server"
    try {
      Invoke-Logged 'npx' @('tsx', 'scripts/lm-adapter-eval.ts', 'generate',
        '--dataset', $slug, '--variant', $variant, '--adapter', $runDir,
        '--seeds', '1', '--samples', "$Samples", '--max-duration', "$MaxDuration", '--out', $plansDir
      ) "plans $slug/$arm"
    } finally { Pop-Location }

    # -- 3. render (through the base DiT, no DiT adapter -- same decoder as gt/base) --------
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

Write-Log ("== experiment done ({0:n0}s total)" -f ((Get-Date) - $t0).TotalSeconds)
