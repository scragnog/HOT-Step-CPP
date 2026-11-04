# Lead-sheet CPU references

Reference captures for YourMT3+ and ChordNet at MuLaCover commit
`e27fa7fa0da7c9817879cc6210cc5693f9327946`. These scripts are developer tools,
not an application runtime. They require CPU-only PyTorch and refuse a CUDA
build. They neither start the app nor use its GPU.

`manifest.json` lists the completed captures, checkpoint hashes, CPU times,
coverage, skips and artifact hashes. `inventory.json` contains the serialized
state tensor names, shapes and dtypes, model state inventories and parameter
counts. YourMT3 has 47,078,022 parameters; each ChordNet fold has 500,772.
The serialized YourMT3 checkpoint also contains training data; file size is
not a parameter-count estimate. The upstream restricted loader discards
`pitchshift.*` entries. Both serialized and inference-state inventories are kept.

Weights, source clone, venv, private source mapping, audio and tensor dumps live
under `_experiments/lead-sheet-reference/`. The primary clips are in `clips-v2/`;
`clips/` and `captures/library-vocal/` preserve the rejected first candidate.
No listening judgement is implied by an alias or a model prediction.

## Reproduce

Run from the repository root in PowerShell. Use a fresh experiment directory
to preserve previous captures. All commands below are foreground commands.

```powershell
$root = (Get-Location).Path
$exp = "$root/_experiments/lead-sheet-reference"
git clone --no-checkout https://github.com/HeartMuLa/MuLaCover.git "$exp/upstream"
git -C "$exp/upstream" checkout --detach e27fa7fa0da7c9817879cc6210cc5693f9327946
uv venv --python 3.11 "$exp/venv"
$py = "$exp/venv/Scripts/python.exe"
uv pip install --python $py torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python $py -r "$root/tools/lead-sheet-reference/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cpu
Set-Location "$exp/upstream"
& $py "$root/tools/lead-sheet-reference/prepare.py" --out ../weights
& $py "$root/tools/lead-sheet-reference/inventory.py" --upstream . --weights ../weights --out ../captures
Set-Location $root
& $py tools/lead-sheet-reference/run_clip.py --upstream "$exp/upstream" --weights "$exp/weights" --audio "$exp/clips-v2/library-vocal.wav" --out "$exp/captures/repeat-vocal"
& $py tools/lead-sheet-reference/verify.py "$exp/captures/repeat-vocal"
Set-Location "$exp/upstream"
& $py "$root/tools/lead-sheet-reference/check_baseline.py" --upstream . --weights ../weights --audio ../clips-v2/library-vocal.wav --capture ../captures/repeat-vocal
```

The source audio is private test material and is not a dependency downloaded
by these scripts. `clips-v2/clips.json` records source hashes, source IDs or
local paths, offsets and the silence edit. Preserve these exact WAVs for the
HOTSTEP-968 ABC-versus-native comparison. That issue had no chosen clip list
when these captures began. `select_clips.py` can prepare new candidates from
a read-only library database, but a later library may select different songs.
It fails when it cannot find both candidates. It does not certify their content.

`run_clip.py` gives each entire capture a 1,800-second deadline and kills and
waits for its own Python child on expiry. This is stricter than a 30-minute
YourMT3-only deadline. It saves the command, exit status, elapsed time and log.
Do not detach it. A timeout is a stop condition, never permission to use CUDA.

## Capture contract

All raw numeric arrays are little-endian contiguous f32 with shape, original
dtype, byte count and SHA-256 in each capture's `tensors.json`. Complex CQT
arrays have separate real and imaginary dumps. Token IDs and events are JSON
integers. Every prediction batch and all 13 channels are retained. Batch axes
follow upstream: up to eight nonoverlapping segments of 32,767 samples.
The exact boundary is 32,767 / 16,000 = 2.0479375 seconds, not 2.048 exactly.

YourMT3 captures padded resampled waveforms, spectrograms, pre-encoder,
encoder and pre-decoder outputs, prefill and first cached-step logits,
decoded tokens, the complete event vocabulary, channel mapping, merged notes,
program-100 vocal selection and drums. Cached-step inputs are the argmax of
the prefill logits. BOS is 0, EOS is 1 and PAD is 0 in the pinned helper.

ChordNet captures its mono 22,050 Hz input, actual tuning estimate, full
hybrid CQT, magnitude and 18:270 crop, CNN and LSTM outputs for all five
folds, six probabilities per fold, their average, Viterbi frame labels and
segments in seconds. The librosa BPM comes from the source sample rate.
Chord rounding uses NumPy's nearest-even rounding to whole beats upstream;
melody rounding uses floor(x + 0.5) to sixteenths. Do not substitute one rule
for the other.

`triplets-sixteenths.json`, four roll/mask tensors and three exported MIDI
files describe the upstream single-BPM condition. `seconds-variant.json`
retains the same notes and chord boundaries before snapping. This second
representation is events in seconds, not an invented dense seconds grid.
The baseline check also writes `without-drums/`, a valid condition with
empty drum triplets and zero drum roll. It does not claim source separation.

An empty program-100 selection is recorded as upstream rejection; no fallback
instrument or fabricated melody is inserted. Likewise a zero estimated BPM
is retained as a rejection. Such captures do not have valid rolls or MIDI.

## Validation and limits

`tolerances.json` declares strict maximum absolute error below 1e-3 for fp32
arrays, with no relative tolerance and no relaxed stages. Discrete outputs
must match exactly. Seconds use absolute error below 1e-6. No port existed
when these tolerances were declared.

`verify.py` checks hashes, shapes, finite arrays, all five folds, the 13-channel
logits, crop equality, exact fold averaging, normalized probabilities, vocal
selection, binary rolls and readable MIDI. It reports repeated pitches,
boundary-crossing notes and inversion labels as model-inferred coverage.
`check_baseline.py` reruns without observation hooks and compares merged notes,
rounded chords and rolls exactly. This passed for the main vocal excerpt.

The transcriber implementation files are unmodified. `common.py` installs a
package namespace for `mulacover` to avoid the generator-only root imports
of torchtune, torchao and codec dependencies. It then imports the original
transcriber modules normally. This is recorded to make the import difference
explicit; it changes no neural forward function or transcription algorithm.

The instrumental test fixture produced one program-100 note. The existing
vocal-stem fixture produced no program-100 notes and no drum notes. These are
oracle outputs, not assertions about audible voice or residual separation
bleed. Listening and confirmation of source content remain human decisions.

No product, engine, server or UI files changed; no app suite, engine rebuild
or GPU smoke was run. The repository's `tools/*` ignore rule currently hides
this directory. Its owner must allow `tools/lead-sheet-reference/` before a
normal explicit-path commit; do not force-add or edit another owner's file.
