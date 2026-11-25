# dettato reference captures

CPU reference for the dettato port (`ace-midi --dettato-model`). Upstream is
[kobimusic/dettato](https://github.com/kobimusic/dettato), Apache-2.0, version
2.0.0, pinned at commit `51ade0f09f2272f4674166026582185d26baaeaf`. These
scripts are developer tools, not an application runtime.

The reference run is `--device cpu`: float32, deterministic algorithms, four
intra-op threads, one inter-op thread, with the evidence repairer replaced by a
stub that returns the transcriber's own notes (the port's `--dettato-no-repair`). A
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
| `replay_grouper.py` | Grouper spread on identical inputs: CUDA grouper fed the CPU capture's notes and features |
| `replay_repairer.py` | Repairer spread on identical inputs: CUDA repairer fed a `cpu-repair` capture's notes and evidence |
| `exact_grouper.py` | Upstream's CPU grouper with PCEN in float64, fed the CPU capture's features: the reference for the port's grouper network (`cpu-exact/`) |
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
& "$exp/venv-cuda/Scripts/python.exe" replay_grouper.py --upstream "$exp/upstream" --weights "$exp/weights" --captures "$exp/captures"
& $py compare.py --captures "$exp/captures" --out spread.json
& $py verify.py --captures "$exp/captures" --weights "$exp/weights" --fixtures "$exp/fixtures"
& $py exact_grouper.py --upstream "$exp/upstream" --weights "$exp/weights" --captures "$exp/captures"
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

## Repairer run

`capture.py --repairer` runs the real evidence repairer instead of the stub.
Two captures exist, `clip-3s/cpu-repair` and `t4-jazz-trio-sax/cpu-repair`
(CPU, fp32). They add: `repair.hidden` (the all-window `hidden` tap, float16
T x 88 x 368), `repair.ev` (N x 769 evidence rows, aligned with the
transcriber's notes in `result.repair.input_notes`), `repair.k.*` (the sorted
input notes), `repair.pk` / `repair.on_r` / `repair.off_r` with
`repair.window_ran` (the model's edits; a window that passes through unchanged
has `window_ran` false), `result.repair.insertions` (every proposal) and
`result.repair.output` (the repaired notes). The grouper and drums run on the
repaired notes, so these captures also hold the full pipeline's final tracks.
They are outside the spread and the other gates in `tolerances.json`. The
repairer's own gates are in its `repairer` block, checked by the engine's
`dettato-repairer-test`. The trio song: 837 transcribed, 61 dropped, 3 inserted.

`cuda-repair` holds the same two fixtures run on CUDA. The repairer has no
autocast upstream, so it runs in float32 there. `replay_repairer.py` feeds the
CPU capture's notes and evidence to the CUDA repairer and writes
`repairer-replay.json`. On the trio song, P(keep) differs by at most 7e-7 and
one onset moves one 5 ms bin.

## Grouper reference

The grouper's PCEN smooths every channel with a 200-tap filter through a float32 FFT.
Wherever the power sits far below the row's peak, its output is that FFT's rounding noise,
and on the CPU the FFT is MKL's. Fed the CPU capture's own features, upstream with any
other FFT regroups notes. With numpy's float32 pocketfft, MKL at length 8192 or an exact
filter, t1 keeps 79% of notes in the same track. t2 keeps 99%, 74% or 73%. The window
probabilities move by up to 0.1 to 0.5. Upstream's own CUDA run on identical inputs
(`grouper-replay.json`) keeps 70% on t1.

So the oracle's exact grouping cannot be a target. `exact_grouper.py` runs upstream with
the filter in float64 (FFT error around 1e-12) and writes `<fixture>/cpu-exact/`, a run
the port reproduces deterministically in its exact-PCEN mode
(`dettato-grouper-test`). `<fixture>/grouper-exact.json` records how far that run sits
from the oracle. The port ships with a float32 FFT filter, as upstream does: numpy's
float32 FFT sits 2 to 4 times closer to the oracle's probabilities on average than the
exact filter, because the network was trained on that noise.

## Limits

- Upstream's grouper leaves a note in no track when the decode has too little
  to work with (`clip-3s`: 4 of 10 notes). The port has to do the same.
- The CUDA runs are one run each. CUDA is not bitwise repeatable, so the spread
  is a sample, not a bound. Timings were taken with the GPU shared and mean
  nothing.
- Hotshot's earlier `norepair.py` CUDA render has 287 piano notes on the piano song (286 here) and 496 piano notes on the trio song (493 here): the size of CUDA's run-to-run variation, not a different pipeline.
