# Chain-stage ladder for artists that already have run_a4 plans + renders: generate plans with the
# stage1 / stage2 adapters (same captions, seeds), render them through the BASE DiT into the same
# renders dir under side labels 'stage1' / 'stage2', extract attributes, re-score.
# Run AFTER run_a4.ps1 for the same artists. Needs the app up.
#
#   powershell -File tools/lm-attr-probe/run_stages.ps1 -Slugs "a,b,c" [-Samples 8] [-MaxDuration 150] [-Steps 8] [-SynthModel <name>]
param(
  [string]$Slugs = '',
  [int]$Samples = 8,
  [int]$MaxDuration = 150,
  [int]$Steps = 8,
  [string]$SynthModel = '',
  [string]$Inventory = "$PSScriptRoot\..\..\docs\plans\lm-attr-probe\inventory.json"
)
$ErrorActionPreference = 'Continue'
$repo = Resolve-Path "$PSScriptRoot\..\.."
$inv = Get-Content $Inventory -Raw | ConvertFrom-Json
$want = @{}
foreach ($s in ($Slugs -split ',')) { if ($s.Trim()) { $want[$s.Trim()] = $true } }
$t0 = Get-Date
foreach ($a in $inv) {
  if ($want.Count -gt 0 -and -not $want.ContainsKey($a.slug)) { continue }
  if (-not $a.preset_adapter) { continue }
  $base = Join-Path $repo "server\data\training\tensors\$($a.slug)\$($a.variant)\lm-attr"
  $renders = Join-Path $base 'renders'
  if (-not (Test-Path (Join-Path $renders 'renders.json'))) { Write-Host "skip $($a.slug): no run_a4 renders yet"; continue }
  foreach ($stage in @('stage1', 'stage2')) {
    $adapter = Join-Path $a.preset_adapter $stage
    if (-not (Test-Path (Join-Path $adapter 'lokr_weights.safetensors')) -and -not (Test-Path (Join-Path $adapter 'adapter_model.safetensors'))) {
      Write-Host "skip $($a.slug) ${stage}: no weights"; continue
    }
    $plans = Join-Path $base "plans_$stage"
    New-Item -ItemType Directory -Force $plans | Out-Null
    $t1 = Get-Date
    Write-Host "== $($a.slug) $stage  adapter=$adapter"
    Push-Location "$repo\server"
    try {
      npx tsx scripts/lm-adapter-eval.ts generate --dataset $a.slug --variant $a.variant `
        --adapter "$adapter" --seeds 1 --samples $Samples --max-duration $MaxDuration --out "$plans"
      $sm = @(); if ($SynthModel) { $sm = @('--synth-model', $SynthModel) }
      npx tsx scripts/lm-plan-render.ts --runs "$plans" --out "$renders" --steps $Steps --sides adapter --label $stage @sm
    } finally { Pop-Location }
    Write-Host ("   {0:n0}s" -f ((Get-Date) - $t1).TotalSeconds)
  }
  python "$PSScriptRoot\extract_attrs.py" --renders --slug $a.slug --workers 6
}
Write-Host ("== scoring ({0:n0}s total)" -f ((Get-Date) - $t0).TotalSeconds)
python "$PSScriptRoot\score_renders.py"
