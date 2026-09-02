# A-4 end to end, per artist, resumable. Needs the app up (dev.bat).
#   plans   : lm-adapter-eval.ts generate  (base + adapter codes, same captions/seeds)
#   renders : lm-plan-render.ts            (gt / base / adapter codes through the BASE DiT, no DiT adapter)
#   attrs   : extract_attrs.py --renders
#   scores  : score_renders.py (+ score_plans.py render-free secondary) at the end
#
#   powershell -File tools/lm-attr-probe/run_a4.ps1 [-Samples 8] [-MaxDuration 150] [-Steps 8] [-Only slug] [-Limit N] [-SkipRender]
param(
  [int]$Samples = 8,
  [int]$MaxDuration = 150,
  [int]$Steps = 8,
  [string]$Only = '',
  [int]$Limit = 0,
  [switch]$SkipRender,
  [string]$Inventory = "$PSScriptRoot\..\..\docs\plans\lm-attr-probe\inventory.json"
)
$ErrorActionPreference = 'Continue'
$repo = Resolve-Path "$PSScriptRoot\..\.."
if (-not (Test-Path $Inventory)) { python "$PSScriptRoot\inventory.py" --json $Inventory | Out-Null }
$inv = Get-Content $Inventory -Raw | ConvertFrom-Json
$n = 0
$t0 = Get-Date
foreach ($a in $inv) {
  if ($Only -and $a.slug -ne $Only) { continue }
  if (-not $a.preset_adapter) { Write-Host "skip $($a.slug): no shipped adapter"; continue }
  $plans = Join-Path $repo "server\data\training\tensors\$($a.slug)\$($a.variant)\lm-attr\plans"
  $renders = Join-Path $repo "server\data\training\tensors\$($a.slug)\$($a.variant)\lm-attr\renders"
  New-Item -ItemType Directory -Force $plans | Out-Null
  $t1 = Get-Date
  Write-Host "== $($a.slug)  adapter=$($a.preset_adapter)"
  Push-Location "$repo\server"
  try {
    npx tsx scripts/lm-adapter-eval.ts generate --dataset $a.slug --variant $a.variant `
      --adapter "$($a.preset_adapter)" --seeds 1 --samples $Samples --max-duration $MaxDuration --out "$plans"
    if (-not $SkipRender) {
      npx tsx scripts/lm-plan-render.ts --runs "$plans" --out "$renders" --steps $Steps
    }
  } finally { Pop-Location }
  if (-not $SkipRender) {
    python "$PSScriptRoot\extract_attrs.py" --renders --slug $a.slug --workers 6
  }
  Write-Host ("   {0:n0}s for {1}  ({2:n0}s total)" -f ((Get-Date) - $t1).TotalSeconds, $a.slug, ((Get-Date) - $t0).TotalSeconds)
  $n++
  if ($Limit -gt 0 -and $n -ge $Limit) { break }
}
Write-Host "== scoring"
python "$PSScriptRoot\score_plans.py" --probe ridge --rep detok64-ctx10 --min-r2 0.2
if (-not $SkipRender) { python "$PSScriptRoot\score_renders.py" }
