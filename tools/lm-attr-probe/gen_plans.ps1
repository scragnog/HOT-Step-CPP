# A-4 driver: base-vs-adapter plans for every 600 s artist, via the existing eval generator.
# Needs the app up (dev.bat). Idempotent: lm-adapter-eval.ts resumes from runs.json.
#
#   powershell -File tools/lm-attr-probe/gen_plans.ps1 [-Seeds 1] [-MaxDuration 180] [-Only slug] [-Limit N]
param(
  [int]$Seeds = 1,
  [int]$MaxDuration = 180,
  [string]$Only = '',
  [int]$Limit = 0,
  [string]$Inventory = "$PSScriptRoot\..\..\docs\plans\lm-attr-probe\inventory.json"
)
$ErrorActionPreference = 'Stop'
$repo = Resolve-Path "$PSScriptRoot\..\.."
if (-not (Test-Path $Inventory)) { python "$PSScriptRoot\inventory.py" --json $Inventory | Out-Null }
$inv = Get-Content $Inventory -Raw | ConvertFrom-Json
$n = 0
foreach ($a in $inv) {
  if ($Only -and $a.slug -ne $Only) { continue }
  if (-not $a.preset_adapter) { Write-Host "skip $($a.slug): no shipped adapter"; continue }
  $out = Join-Path $repo "server\data\training\tensors\$($a.slug)\$($a.variant)\lm-attr\plans"
  New-Item -ItemType Directory -Force $out | Out-Null
  $t0 = Get-Date
  Write-Host "== $($a.slug)  adapter=$($a.preset_adapter)"
  Push-Location "$repo\server"
  try {
    npx tsx scripts/lm-adapter-eval.ts generate --dataset $a.slug --variant $a.variant `
      --adapter "$($a.preset_adapter)" --seeds $Seeds --max-duration $MaxDuration --out "$out"
  } finally { Pop-Location }
  Write-Host ("   {0:n0}s" -f ((Get-Date) - $t0).TotalSeconds)
  $n++
  if ($Limit -gt 0 -and $n -ge $Limit) { break }
}
