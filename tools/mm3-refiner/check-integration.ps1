# MM3 latent refiner, engine integration gate (slice 5).
#
#   probe   mm3-refiner-probe: end-to-end parity against the torch fixtures (12 clips x steps 1, 8, 32)
#           and the behaviour checks (bypass, bounds, seams, errors, cancel, silence, odd tail, mono,
#           48 kHz, mix, residency).
#   route   POST /mm3/refine on throwaway ace-server instances (default port 8185, models in
#           artifacts/models/mm3): 400/413/501 answers, mix=0 bypass, phases, torch parity over HTTP,
#           cancel, FIFO with competing jobs, 48 kHz and mono input, a failed job's error, and
#           VRAM back at its baseline after the jobs (leak check) with the peak recorded.
#
# Needs the GPU idle (Unreal paused for timings), a built engine (dev-rebuild.bat) and the ignored
# artifacts: fixtures-v1/ and gguf/ from convert.py / capture-reference.py, plus
# artifacts/models/mm3/ holding mm3-refiner-refiner-f32, mm3-refiner-mert-f32, mm3-refiner-clap-f32,
# mm3-enc-f16 and mm3-voc-f16 (hardlink or copy; see README). The route half runs the real CLAP style
# source; HOTSTEP_MM3_REFINE_TEST_STYLE is used only for the override and corrupt-file checks. Exit 0 pass, 1 fail.
#
#   pwsh -NoProfile -File tools\mm3-refiner\check-integration.ps1 [-SkipProbe] [-SkipRoute] [-Port 8185]
param([switch]$SkipProbe, [switch]$SkipRoute, [int]$Port = 8185,
      [ValidateSet(1, 8, 32)][int[]]$ParitySteps = @(8))
$ErrorActionPreference = 'Stop'
$repo      = (Resolve-Path "$PSScriptRoot\..\..").Path
$art       = "$repo\tools\mm3-refiner\artifacts"
$out       = "$art\integration"
$probe     = "$repo\engine\build\Release\mm3-refiner-probe.exe"
$server    = "$repo\engine\build\Release\ace-server.exe"
$fixtures  = "$art\fixtures-v1"
$models    = "$art\models"
$native    = "$repo\_experiments\mm3-whats-new-2026-11-05\native"
New-Item -ItemType Directory -Force $out | Out-Null
$script:pass = 0; $script:fail = 0
function Check([bool]$ok, [string]$what) {
    if ($ok) { $script:pass++; Write-Host "  PASS $what" } else { $script:fail++; Write-Host "  FAIL $what" -ForegroundColor Red }
}

# ── probe ───────────────────────────────────────────────────────────────────
if (-not $SkipProbe) {
    foreach ($steps in 1, 8, 32) {
        $log = "$out\parity-steps$steps.log"
        & $probe parity --contract "$repo\tools\mm3-refiner\fixtures.json" --fixtures $fixtures --models "$models\mm3" --clips all --steps $steps > $log 2>&1
        $code = $LASTEXITCODE
        Write-Host "parity steps=$steps exit=$code"
        Select-String -Path $log -Pattern '^worst|^  (overlap|stitched|raw|final|gain)|^PASS|^FAIL' | Select-Object -Last 8 | ForEach-Object { Write-Host "  $($_.Line.Trim())" }
        Check ($code -eq 0) "end-to-end parity on 12 clips at $steps steps"
    }
    & $probe behave --fixtures $fixtures --models "$models\mm3" --scratch $out > "$out\behave.log" 2>&1
    $code = $LASTEXITCODE
    Select-String -Path "$out\behave.log" -Pattern '^PASS|^FAIL|^  FAIL' | ForEach-Object { Write-Host "  $($_.Line.Trim())" }
    Check ($code -eq 0) "behaviour checks"
}

# ── helpers for the route half ──────────────────────────────────────────────
function New-Wav([string]$path, [double]$seconds, [int]$rate, [int]$channels) {
    $n = [int]($seconds * $rate); $ms = New-Object IO.MemoryStream; $bw = New-Object IO.BinaryWriter($ms)
    $bw.Write([Text.Encoding]::ASCII.GetBytes('RIFF')); $bw.Write([int](36 + $n * $channels * 2)); $bw.Write([Text.Encoding]::ASCII.GetBytes('WAVEfmt '))
    $bw.Write([int]16); $bw.Write([int16]1); $bw.Write([int16]$channels); $bw.Write([int]$rate); $bw.Write([int]($rate * $channels * 2)); $bw.Write([int16]($channels * 2)); $bw.Write([int16]16)
    $bw.Write([Text.Encoding]::ASCII.GetBytes('data')); $bw.Write([int]($n * $channels * 2))
    $rng = New-Object Random 7
    for ($i = 0; $i -lt $n; $i++) {
        $t = $i / $rate
        $v = 0.25 * [Math]::Sin(2 * [Math]::PI * 220 * $t) + 0.1 * [Math]::Sin(2 * [Math]::PI * 1760 * $t) + 0.05 * ($rng.NextDouble() - 0.5)
        for ($c = 0; $c -lt $channels; $c++) { $bw.Write([int16]($v * 32767)) }
    }
    $bw.Flush(); [IO.File]::WriteAllBytes($path, $ms.ToArray())
}
Add-Type -TypeDefinition @'
using System;
using System.Text;
public class WavData {
    public int Rate, Ch, N; public float[][] Data;
    public static WavData Parse(byte[] b) {
        int pos = 12, fmt = 0, ch = 0, rate = 0, bits = 0, off = 0, len = 0;
        while (pos + 8 <= b.Length) {
            string id = Encoding.ASCII.GetString(b, pos, 4); int l = BitConverter.ToInt32(b, pos + 4);
            if (id == "fmt ") { fmt = BitConverter.ToInt16(b, pos + 8); ch = BitConverter.ToInt16(b, pos + 10); rate = BitConverter.ToInt32(b, pos + 12); bits = BitConverter.ToInt16(b, pos + 22); }
            if (id == "data") { off = pos + 8; len = Math.Min(l, b.Length - off); break; }
            pos += 8 + l + (l % 2);
        }
        int bytes = bits / 8, n = len / (bytes * ch); var d = new float[ch][];
        for (int c = 0; c < ch; c++) d[c] = new float[n];
        for (int i = 0; i < n; i++) for (int c = 0; c < ch; c++) {
            int o = off + (i * ch + c) * bytes;
            d[c][i] = fmt == 3 ? BitConverter.ToSingle(b, o) : bits == 16 ? BitConverter.ToInt16(b, o) / 32768f : ((b[o] | (b[o + 1] << 8) | (b[o + 2] << 16)) << 8 >> 8) / 8388608f;
        }
        return new WavData { Rate = rate, Ch = ch, N = n, Data = d };
    }
    // max |a-b| and the relative RMSE of a against b, over the first n frames of both channels
    public static double[] Compare(WavData a, WavData b, int n) {
        if (a.Rate != b.Rate || a.N != b.N || a.Ch != b.Ch || n != a.N)
            throw new ArgumentException("WAV comparison requires matching sample rates, frame counts and channels, over the complete clip");
        double max = 0, d2 = 0, r2 = 0;
        for (int c = 0; c < 2; c++) for (int i = 0; i < n; i++) {
            double d = a.Data[c][i] - b.Data[c][i]; if (Math.Abs(d) > max) max = Math.Abs(d); d2 += d * d; r2 += (double)b.Data[c][i] * b.Data[c][i];
        }
        return new double[] { max, Math.Sqrt(d2 / Math.Max(r2, 1e-30)) };
    }
    public static double MaxChannelDiff(WavData a) { double m = 0; for (int i = 0; i < a.N; i++) m = Math.Max(m, Math.Abs(a.Data[0][i] - a.Data[1][i])); return m; }
}
'@
function Read-Wav([byte[]]$b) { [WavData]::Parse($b) }
function Get-VramMB { [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1) }
function Start-Engine([string]$modelsDir, [string]$styleFile, [string]$tag) {
    if ($styleFile) { $env:HOTSTEP_MM3_REFINE_TEST_STYLE = $styleFile } else { Remove-Item Env:HOTSTEP_MM3_REFINE_TEST_STYLE -ErrorAction SilentlyContinue }
    $p = Start-Process -FilePath $server -ArgumentList @('--models', $modelsDir, '--host', '127.0.0.1', '--port', "$Port") -RedirectStandardOutput "$out\engine-$tag.out.log" -RedirectStandardError "$out\engine-$tag.err.log" -PassThru -WindowStyle Hidden
    Remove-Item Env:HOTSTEP_MM3_REFINE_TEST_STYLE -ErrorAction SilentlyContinue
    for ($i = 0; $i -lt 60; $i++) { try { $null = Invoke-WebRequest "http://127.0.0.1:$Port/mm3/props" -TimeoutSec 2 -SkipHttpErrorCheck; return $p } catch { Start-Sleep -Milliseconds 500 } }
    Stop-Process -Id $p.Id -Force; throw "engine '$tag' did not come up; see $out\engine-$tag.err.log"
}
function Submit([byte[]]$body, [string]$query) {
    Invoke-WebRequest "http://127.0.0.1:$Port/mm3/refine$query" -Method Post -Body $body -ContentType audio/wav -SkipHttpErrorCheck
}
# polls until the job leaves "running"; returns the last status and the set of phases seen
function Wait-Job([string]$id, [int]$timeoutSec = 300, [scriptblock]$each = $null) {
    $phases = [Collections.Generic.List[string]]::new(); $maxStep = 0; $maxTotal = 0; $peak = 0
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        $s = Invoke-RestMethod "http://127.0.0.1:$Port/job?id=$id"
        if (-not $phases.Contains($s.phase)) { $phases.Add($s.phase) }
        if ($s.phase -eq 'refine_bridge') { $maxStep = [Math]::Max($maxStep, $s.phase_step); $maxTotal = [Math]::Max($maxTotal, $s.phase_total) }
        $peak = [Math]::Max($peak, (Get-VramMB))
        if ($each) { & $each $s }
        if ($s.status -ne 'running') { return @{ status = $s; phases = $phases; bridge_step = $maxStep; bridge_total = $maxTotal; peak_mb = $peak } }
        Start-Sleep -Milliseconds 150
    }
    throw "job $id still running after $timeoutSec s"
}
function Get-Result([string]$id) { (Invoke-WebRequest "http://127.0.0.1:$Port/job?id=$id&result=1").Content }
function Err([object]$r) { try { ($r.Content | ConvertFrom-Json).error } catch { '' } }

# ── route ───────────────────────────────────────────────────────────────────
if (-not $SkipRoute) {
    $clipBytes = [IO.File]::ReadAllBytes("$native\boombap-s1001.wav")
    $wav8      = "$out\tone-8s.wav";  New-Wav $wav8 8 44100 2
    $tone      = [IO.File]::ReadAllBytes($wav8)
    $long      = "$out\tone-61s.wav"; New-Wav $long 61 44100 2
    $style     = "$out\style.f32"
    $b = [IO.File]::ReadAllBytes("$fixtures\boombap-s1001\steps-8\window-00\clap_vector.npy"); [IO.File]::WriteAllBytes($style, $b[($b.Length - 2048)..($b.Length - 1)])

    # a pack-less models dir (encoder and decoder only): 400s, 413, bypass and the 501 for the missing pack
    $nopack = "$out\models-nopack"; New-Item -ItemType Directory -Force "$nopack\mm3" | Out-Null
    foreach ($f in 'mm3-enc-f16.gguf', 'mm3-voc-f16.gguf') { if (-not (Test-Path "$nopack\mm3\$f")) { New-Item -ItemType HardLink -Path "$nopack\mm3\$f" -Target "$models\mm3\$f" | Out-Null } }
    Write-Host "route: no pack"
    $e = Start-Engine $nopack '' 'nopack'
    try {
        $r = Submit $tone '?steps=8';                         Check ($r.StatusCode -eq 501 -and (Err $r) -like '*mm3-refiner-refiner-*.gguf*') "missing pack: 501 naming the file ($(Err $r))"
        $r = Submit ([byte[]]@()) '';                         Check ($r.StatusCode -eq 400) "empty body: 400"
        $r = Submit ([byte[]](1..200)) '';                    Check ($r.StatusCode -eq 400) "undecodable body: 400"
        foreach ($q in '?steps=0', '?steps=65', '?steps=2.5', '?steps=abc', '?window=-1', '?overlap=-1', '?window=5&overlap=5', '?mix=1.5', '?mix=-0.1') {
            $r = Submit $tone $q;                             Check ($r.StatusCode -eq 400) "$q : 400 ($(Err $r))"
        }
        $r = Submit ([IO.File]::ReadAllBytes($long)) '?window=0'
        Check ($r.StatusCode -eq 413) "direct mode on 61 s: 413 ($(Err $r))"
        $r = Submit ([IO.File]::ReadAllBytes($long)) '?window=0&mix=0'
        Check ($r.StatusCode -eq 413) "direct mode limit also guards mix=0 requests: 413"
        foreach ($q in '?window=120', '?window=120&mix=0', '?window=61', '?window=1e300', '?window=1e308&mix=0') {
            $r = Submit ([IO.File]::ReadAllBytes($long)) $q
            Check ($r.StatusCode -eq 413) "a large window is bounded like direct mode ($q on 61 s): 413 ($(Err $r))"
        }
        $r = Submit $tone '?window=120&mix=0'
        Check ($r.StatusCode -eq 200) "window=120 on an 8 s clip is fine (the sequence is the clip): 200"
        $r = Submit $tone '?window=1e300&overlap=1e300'
        Check ($r.StatusCode -eq 400) "window and overlap of 1e300: 400, no overflow"
        $r = Submit $tone '?mix=0'
        Check ($r.StatusCode -eq 200) "mix=0 needs no pack: 200"
        $id = ($r.Content | ConvertFrom-Json).id; $j = Wait-Job $id 30
        $o = Read-Wav (Get-Result $id); $i = Read-Wav $tone
        $same = $o.Rate -eq $i.Rate -and $o.N -eq $i.N -and $o.Ch -eq 2; $dmax = if ($same) { [WavData]::Compare($o, $i, $i.N)[0] } else { 1 }
        Check ($j.status.status -eq 'done' -and $same -and $dmax -lt 1e-6) "mix=0 returns the input (max diff $dmax), same rate and length"
    } finally { Stop-Process -Id $e.Id -Force }

    Write-Host "route: pack without the CLAP file"
    $noclap = "$out\models-noclap"; New-Item -ItemType Directory -Force "$noclap\mm3" | Out-Null
    foreach ($f in 'mm3-refiner-refiner-f32.gguf', 'mm3-refiner-mert-f32.gguf', 'mm3-enc-f16.gguf', 'mm3-voc-f16.gguf') { if (-not (Test-Path "$noclap\mm3\$f")) { New-Item -ItemType HardLink -Path "$noclap\mm3\$f" -Target "$models\mm3\$f" | Out-Null } }
    $e = Start-Engine $noclap '' 'noclap'
    try {
        $r = Submit $tone '?steps=8'
        Check ($r.StatusCode -eq 501 -and (Err $r) -like '*mm3-refiner-clap-*.gguf*') "missing CLAP file: 501 naming it, no silent fallback ($(Err $r))"
        $r = Submit $tone '?mix=0'
        Check ($r.StatusCode -eq 200) "mix=0 still needs no pack: 200"
    } finally { Stop-Process -Id $e.Id -Force }
    $e = Start-Engine $noclap $style 'override'
    try {
        $r = Submit $tone '?steps=2'; $id = ($r.Content | ConvertFrom-Json).id; $j = Wait-Job $id 120
        Check ($r.StatusCode -eq 200 -and $j.status.status -eq 'done') "HOTSTEP_MM3_REFINE_TEST_STYLE overrides the CLAP tower (no CLAP file needed)"
    } finally { Stop-Process -Id $e.Id -Force }

    Write-Host "route: corrupt refiner file"
    $bad = "$out\models-corrupt"; New-Item -ItemType Directory -Force "$bad\mm3" | Out-Null
    foreach ($f in 'mm3-refiner-mert-f32.gguf', 'mm3-enc-f16.gguf', 'mm3-voc-f16.gguf') { if (-not (Test-Path "$bad\mm3\$f")) { New-Item -ItemType HardLink -Path "$bad\mm3\$f" -Target "$models\mm3\$f" | Out-Null } }
    Copy-Item "$out\corrupt-refiner.gguf" "$bad\mm3\mm3-refiner-refiner-f32.gguf" -Force
    $e = Start-Engine $bad $style 'corrupt'
    try {
        $r = Submit $tone '?steps=2'; $id = ($r.Content | ConvertFrom-Json).id; $j = Wait-Job $id 60
        Check ($r.StatusCode -eq 200 -and $j.status.status -eq 'failed' -and $j.status.error -like 'refiner:*') "unreadable refiner: job fails and the status JSON says why ($($j.status.error))"
    } finally { Stop-Process -Id $e.Id -Force }

    Write-Host "route: full run"
    $e = Start-Engine $models '' 'full'
    try {
        # torch parity over HTTP with the real CLAP style source: boombap-s1001, defaults but 8 steps, against the fixture's final.wav
        $vram0 = Get-VramMB
        $r = Submit $clipBytes '?steps=8'; $id = ($r.Content | ConvertFrom-Json).id
        $j = Wait-Job $id 300
        Check ($j.status.status -eq 'done') "boombap-s1001 at 8 steps completes"
        Check ($j.phases.Contains('loading_refiner') -and $j.phases.Contains('refine_condition') -and $j.phases.Contains('refine_bridge')) "phases seen: $($j.phases -join ', ')"
        Check ($j.bridge_total -eq 16 -and $j.bridge_step -ge 8) "bridge progress counts forwards over windows x steps (max $($j.bridge_step)/$($j.bridge_total))"
        $res = Read-Wav (Get-Result $id); $ref = Read-Wav ([IO.File]::ReadAllBytes("$fixtures\boombap-s1001\steps-8\final.wav")); $inp = Read-Wav $clipBytes
        $cmp = [WavData]::Compare($res, $ref, $res.N); $dmax = $cmp[0]; $rel = $cmp[1]
        Check ($res.Rate -eq 44100 -and $res.Ch -eq 2 -and $res.N -eq $inp.N) "result is 44.1 kHz stereo, $($res.N) frames = the input's"
        Check ($dmax -le 0.02 -and $rel -le 0.01) ("HTTP result vs torch final.wav: max_abs=$([Math]::Round($dmax,6)) rel_rmse=$([Math]::Round($rel,6)) (gates 0.02 / 0.01)")
        Start-Sleep -Seconds 1; $vramWarm = Get-VramMB   # after the first job: the CUDA context and pools now exist
        Write-Host "  peak VRAM during the job: $($j.peak_mb) MB; idle before the first job $vram0 MB, after it $vramWarm MB"

        # all ten native clips, real CLAP style, requested steps, against each fixture's final.wav (waveform gates 0.02 / 0.01)
        $nativeClips = @(Get-ChildItem "$native\*.wav")
        if ($nativeClips.Count -ne 10) { throw "Expected exactly ten native clips, found $($nativeClips.Count)" }
        foreach ($parityStep in $ParitySteps) {
            $worst = @{ name = ''; dmax = 0.0; rel = 0.0 }; $allOk = $true; $peakClips = 0
            foreach ($d in $nativeClips) {
                $name = $d.BaseName; $bytes = [IO.File]::ReadAllBytes($d.FullName)
                $r = Submit $bytes "?steps=$parityStep"; $id = ($r.Content | ConvertFrom-Json).id; $jc = Wait-Job $id 300
                $resultBytes = Get-Result $id
                [IO.File]::WriteAllBytes("$out\$name-steps-$parityStep.wav", [byte[]]$resultBytes)
                $res = Read-Wav $resultBytes; $ref = Read-Wav ([IO.File]::ReadAllBytes("$fixtures\$name\steps-$parityStep\final.wav"))
                $cmp = [WavData]::Compare($res, $ref, $res.N)
                $ok = $jc.status.status -eq 'done' -and $cmp[0] -le 0.02 -and $cmp[1] -le 0.01; $allOk = $allOk -and $ok
                $peakClips = [Math]::Max($peakClips, $jc.peak_mb)
                if ($cmp[1] -gt $worst.rel) { $worst = @{ name = $name; dmax = $cmp[0]; rel = $cmp[1] } }
                Write-Host ("    {0}: max_abs={1:N6} rel_rmse={2:N6} {3}" -f $name, $cmp[0], $cmp[1], $(if ($ok) { 'ok' } else { 'FAIL' }))
            }
            Check $allOk ("end-to-end parity on the ten native clips, real CLAP style, steps=$parityStep (worst by rel_rmse: {0} max_abs={1:N6} rel_rmse={2:N6})" -f $worst.name, $worst.dmax, $worst.rel)
            Write-Host "  peak VRAM across the ten clips: $peakClips MB"
        }

        # cancel in the bridge
        $r = Submit $clipBytes '?steps=64'; $id = ($r.Content | ConvertFrom-Json).id
        $t0 = Get-Date; $sent = $false
        $j = Wait-Job $id 120 { param($s) if (-not $sent -and $s.phase -eq 'refine_bridge' -and $s.phase_step -ge 2) { $null = Invoke-RestMethod "http://127.0.0.1:$Port/job?id=$id&cancel=1" -Method Post; Set-Variable sent $true -Scope 2 } }
        Check ($j.status.status -eq 'cancelled' -and $j.bridge_step -lt 128) "cancel in the bridge: status cancelled at forward $($j.bridge_step) of $($j.bridge_total)"
        $r = Submit $tone '?steps=2'; $id = ($r.Content | ConvertFrom-Json).id; $j = Wait-Job $id 120
        Check ($j.status.status -eq 'done') "a job after the cancel completes"

        # competing jobs: three queued back to back finish in order with identical output
        $ids = @(); 1..3 | ForEach-Object { $r = Submit $tone '?steps=4&window=6&overlap=1'; $ids += ($r.Content | ConvertFrom-Json).id }
        $doneAt = @{}; $deadline = (Get-Date).AddSeconds(240)
        while ($doneAt.Count -lt 3 -and (Get-Date) -lt $deadline) {
            foreach ($id in $ids) { if (-not $doneAt.ContainsKey($id)) { $s = Invoke-RestMethod "http://127.0.0.1:$Port/job?id=$id"; if ($s.status -ne 'running') { $doneAt[$id] = @{ at = Get-Date; status = $s.status } } } }
            Start-Sleep -Milliseconds 100
        }
        $inOrder = $doneAt.Count -eq 3 -and $doneAt[$ids[0]].at -le $doneAt[$ids[1]].at -and $doneAt[$ids[1]].at -le $doneAt[$ids[2]].at
        Check ($doneAt.Count -eq 3 -and ($doneAt.Values | Where-Object { $_.status -ne 'done' }).Count -eq 0 -and $inOrder) "three queued jobs all finish, FIFO"
        $h = $ids | ForEach-Object { (Get-FileHash -InputStream ([IO.MemoryStream]::new([byte[]](Get-Result $_))) -Algorithm SHA256).Hash }
        Check (($h | Select-Object -Unique).Count -eq 1) "the three results are byte-identical (deterministic bridge)"

        # input formats
        $w48 = "$out\tone-48k.wav"; New-Wav $w48 5 48000 2
        $r = Submit ([IO.File]::ReadAllBytes($w48)) '?steps=2'; $id = ($r.Content | ConvertFrom-Json).id; $j = Wait-Job $id 120
        $o = Read-Wav (Get-Result $id)
        Check ($j.status.status -eq 'done' -and $o.Rate -eq 48000 -and $o.N -eq 240000) "48 kHz input: result is 48 kHz and 240000 frames"
        $wm = "$out\tone-mono.wav"; New-Wav $wm 5 44100 1
        $r = Submit ([IO.File]::ReadAllBytes($wm)) '?steps=2'; $id = ($r.Content | ConvertFrom-Json).id; $j = Wait-Job $id 120
        $o = Read-Wav (Get-Result $id); $lr = [WavData]::MaxChannelDiff($o)
        Check ($j.status.status -eq 'done' -and $o.Ch -eq 2 -and $o.N -eq 220500 -and $lr -eq 0) "mono input: stereo result, dual mono, 220500 frames"
        $r = Submit $tone '?steps=4&mix=0.5'; $id = ($r.Content | ConvertFrom-Json).id; $j = Wait-Job $id 120
        Check ($j.status.status -eq 'done') "mix=0.5 completes"
        $r = Submit ([IO.File]::ReadAllBytes($long)) '?window=0'
        Check ($r.StatusCode -eq 413) "direct mode on 61 s: 413 with the pack installed"

        # leak: the engine is idle, STRICT residency unloads after each job, so VRAM sits at its baseline
        Start-Sleep -Seconds 1; $vram1 = Get-VramMB
        # 300 MB: the desktop's own processes move this reading by +-250 MB (a 30-job loop on the same engine wandered 5944-6332 MB with no trend)
        Check ([Math]::Abs($vram1 - $vramWarm) -lt 300) "no VRAM growth over the jobs after the first ($vramWarm MB -> $vram1 MB; STRICT residency unloads each job)"
    } finally { Stop-Process -Id $e.Id -Force }
}

Write-Host ("`n{0} passed, {1} failed" -f $script:pass, $script:fail)
if ($script:fail) { Write-Host 'FAIL mm3 refiner integration' -ForegroundColor Red; exit 1 }
Write-Host 'PASS mm3 refiner integration'; exit 0
