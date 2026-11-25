# dettato reference captures

CPU reference for the dettato port (`ace-midi --dettato-model`). Upstream is
[kobimusic/dettato](https://github.com/kobimusic/dettato), Apache-2.0, version
2.0.0, pinned at commit `51ade0f09f2272f4674166026582185d26baaeaf`. These
scripts are developer tools, not an application runtime.

The reference run is `--device cpu`: float32, deterministic algorithms, four
intra-op threads, one inter-op thread, with the evidence repairer replaced by a
stub that returns the transcriber's own notes (the repairer is not ported). A
second run of the same fixtures on CUDA, where upstream uses bfloat16 autocast,
gives the **upstream spread**, the gap a GPU backend of the port is allowed.
Repeating a CPU capture gives identical tensor hashes (checked on `clip-3s`).

Weights, upstream clone, venvs, fixture audio and tensor dumps live under
`_experiments/dettato-reference/` (outside git). Files here:

| File | Purpose |
|---|---|
| `tolerances.json` | Gates, declared before any port code exists |
| `spread.json` | Measured upstream spread, per fixture, plus `_worst` per gate |
| `manifest.json` | Pins, fixture hashes, wall times, capture sizes |
| `prepare.py` | Copies the four upstream model repos unmodified, records sha256 |
| `fixtures.py` | Builds the 11 fixture WAVs from songs our own engine rendered |
| `capture.py` | One fixture, one device: runs upstream, records every stage |
| `compare.py` | Gate metrics between two captures (spread now, port later) |
| `verify.py` | Hashes, finite values, shapes, grouping coverage |

## Reproduce

PowerShell, from the repository root. Foreground only; the 260 s fixture takes
about six minutes on the CPU.

```powershell
$root = (Get-Location).Path
$exp = "$root/_experiments/dettato-reference"
git clone --no-checkout https://github.com/kobimusic/dettato.git "$exp/upstream"
git -C "$exp/upstream" checkout --detach 51ade0f09f2272f4674166026582185d26baaeaf
uv venv --python 3.14 "$exp/venv-cpu"
uv venv --python 3.14 "$exp/venv-cuda"
uv pip install --python "$exp/venv-cpu/Scripts/python.exe" torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$exp/venv-cuda/Scripts/python.exe" torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
foreach ($v in "venv-cpu", "venv-cuda") { uv pip install --python "$exp/$v/Scripts/python.exe" -r tools/dettato-reference/requirements.txt }
Set-Location tools/dettato-reference
$py = "$exp/venv-cpu/Scripts/python.exe"
# huggingface_hub must already hold the four kobimusic/dettato* snapshots (the first upstream run, or `hf download`, fetches them)
& $py prepare.py --out "$exp/weights"
& $py fixtures.py --songs <dir with the six engine songs> --out "$exp/fixtures"
& $py capture.py --upstream "$exp/upstream" --weights "$exp/weights" --audio "$exp/fixtures/t3-solo-piano.wav" --out "$exp/captures/t3-solo-piano/cpu" --device cpu
& "$exp/venv-cuda/Scripts/python.exe" capture.py --upstream "$exp/upstream" --weights "$exp/weights" --audio "$exp/fixtures/t3-solo-piano.wav" --out "$exp/captures/t3-solo-piano/cuda" --device cuda
& $py compare.py --captures "$exp/captures" --out spread.json
& $py verify.py --captures "$exp/captures" --weights "$exp/weights" --fixtures "$exp/fixtures"
```

## Fixtures

Six songs rendered by our own engine, plus five derived from them, so no
third-party audio is involved: `silence-10s`, `clip-3s` (piano, 20 s in),
`long-260s` (three songs joined, the host-buffer path), `src-32k` (the funk
song resampled to 32 kHz, the drum model must skip) and `detune-30c` (the piano
song played 30 cents fast; upstream estimates 31.6 cents and retunes).

## Capture contract

Per fixture and device: `result.json` (notes, groups, drum decode of the chosen
and of both branches, tracks, tuning, timings) and raw `.bin` tensors listed in
`tensors.json` with shape, dtype, size and sha256. float16 and int8 arrays keep
upstream's rounding, which is part of the function.

- Transcriber: six all-window maps (`maps.*`), the six maps of the every-4 s
  sub-run (`sub.maps.*`), the sub-run `sub.tap.hidden` (T x 88 x 368) and
  `sub.tap.stem` (T x 88 x 96) in float16, decoded notes (`transcriber_notes`,
  `sub_notes`), the 16 kHz input (`x16`, and `x16_retuned` after a retune).
- Grouper: `grouper.meta` (N x 12 float32), `grouper.feat` (N x 3 x 368 int8),
  `grouper.fsc` (float16 scales), `grouper.probs`, `grouper.emb`, `grouper.h`,
  per-window `grouper.win.*`, and `groups` (note ids per instrument index).
- Drums: `x44`, `drums.features` (float16), router, kit and orchestral
  probabilities, the hits, and the decode of both branches (`drum_branches`).

## Limits

- The all-window `hidden` tap the repairer reads is not captured; the repairer
  is out of scope.
- Upstream's grouper leaves a note in no track when the decode has too little
  to work with (`clip-3s`: 4 of 10 notes). The port has to do the same.
- The CUDA runs are one run each. CUDA is not bitwise repeatable, so the spread
  is a sample, not a bound. Timings were taken with the GPU shared and mean
  nothing.
- Hotshot's earlier `norepair.py` CUDA render has 287 piano notes on the piano song (286 here) and 496 piano notes on the trio song (493 here): the size of CUDA's run-to-run variation, not a different pipeline.
