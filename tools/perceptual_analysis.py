#!/usr/bin/env python3
"""
perceptual_analysis.py — Post-hoc perceptual metrics for HOT-Step benchmark WAVs
==================================================================================

Runs human-perceptible audio quality metrics across all generated benchmark WAVs.
Outputs a single CSV that can be merged with existing benchmark_results.csv files.

Metrics computed:
  - spectral_contrast   : Clarity/definition (higher = clearer, less muddy)
  - onset_strength       : Transient punch/crispness (higher = punchier)
  - onset_density        : Detected onsets per second (musical detail/busyness)
  - stereo_width         : Spatial spread (0=mono, 1=full stereo, >1=out-of-phase)
  - bass_ratio           : Energy in bass band (<250Hz) as fraction of total
  - mid_ratio            : Energy in mid band (250-4000Hz) as fraction of total
  - treble_ratio         : Energy in treble band (>4000Hz) as fraction of total
  - balance_deviation    : How far from "ideal" 0.2/0.5/0.3 balance (lower = more balanced)
  - beat_confidence      : Beat tracking confidence (higher = more consistent rhythm)
  - flux_stability       : Inverse of spectral flux variance (higher = more stable texture)
  - hnr_vocal            : Harmonic-to-noise ratio in vocal band 300-3000Hz (higher = cleaner vocals)
  - rms_variance         : Variance of RMS energy over time (higher = more dynamic, musical)
  - mfcc_variance        : Variance of timbral features (higher = more varied, musical)
  - is_degenerate        : 1.0 if output appears to be noise/buzz/static, 0.0 otherwise
  - perceptual_score     : Weighted composite of all metrics (higher = better)

Usage:
  python perceptual_analysis.py E:\\benchmarks\\run_2026-05-27_22-04
  python perceptual_analysis.py E:\\benchmarks\\run_2026-05-27_22-04 --combo euler__linear__apg
  python perceptual_analysis.py E:\\benchmarks\\run_2026-05-27_22-04 --resume

Requires: pip install librosa numpy soundfile
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Force UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    print("ERROR: librosa is required. Install with: pip install librosa")
    sys.exit(1)

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False


# ── Metric computation ──

def load_audio_stereo(path: str, sr: int = 44100):
    """Load audio file, return (mono, stereo_channels, sr)."""
    if HAS_SF:
        data, file_sr = sf.read(path, dtype='float32')
        if data.ndim == 1:
            stereo = np.stack([data, data])  # mono → fake stereo
        else:
            stereo = data.T  # (samples, channels) → (channels, samples)
        mono = librosa.to_mono(stereo)
        if file_sr != sr:
            mono = librosa.resample(mono, orig_sr=file_sr, target_sr=sr)
            stereo = np.stack([
                librosa.resample(stereo[0], orig_sr=file_sr, target_sr=sr),
                librosa.resample(stereo[1], orig_sr=file_sr, target_sr=sr),
            ])
        return mono, stereo, sr
    else:
        mono, file_sr = librosa.load(path, sr=sr, mono=True)
        return mono, np.stack([mono, mono]), sr


def compute_spectral_contrast(y, sr):
    """Mean spectral contrast across bands — measures clarity/definition."""
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    return float(np.mean(contrast))


def compute_onset_strength(y, sr):
    """Mean onset strength — measures transient punch."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    return float(np.mean(onset_env))


def compute_onset_density(y, sr):
    """Onsets per second — measures musical detail/busyness."""
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    duration = len(y) / sr
    if duration > 0:
        return float(len(onset_frames) / duration)
    return 0.0


def compute_stereo_width(stereo):
    """Stereo width from L/R correlation. 0=mono, 1=uncorrelated, >1=out-of-phase."""
    if stereo.shape[0] < 2:
        return 0.0
    left, right = stereo[0], stereo[1]
    # Pearson correlation
    if np.std(left) < 1e-10 or np.std(right) < 1e-10:
        return 0.0
    corr = np.corrcoef(left, right)[0, 1]
    # Width: 0 for perfect correlation (mono), 1 for uncorrelated, 2 for anti-correlated
    return float(1.0 - corr)


def compute_subband_balance(y, sr):
    """Energy ratios in bass (<250Hz), mid (250-4000Hz), treble (>4000Hz)."""
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    bass_mask = freqs < 250
    mid_mask = (freqs >= 250) & (freqs < 4000)
    treble_mask = freqs >= 4000

    bass_energy = np.sum(S[bass_mask, :] ** 2)
    mid_energy = np.sum(S[mid_mask, :] ** 2)
    treble_energy = np.sum(S[treble_mask, :] ** 2)
    total = bass_energy + mid_energy + treble_energy

    if total < 1e-10:
        return 0.0, 0.0, 0.0, 0.0

    bass_r = float(bass_energy / total)
    mid_r = float(mid_energy / total)
    treble_r = float(treble_energy / total)

    # Deviation from "ideal" balance (roughly 20% bass, 50% mid, 30% treble)
    ideal = np.array([0.20, 0.50, 0.30])
    actual = np.array([bass_r, mid_r, treble_r])
    deviation = float(np.sqrt(np.sum((actual - ideal) ** 2)))

    return bass_r, mid_r, treble_r, deviation


def compute_beat_confidence(y, sr):
    """Beat tracking confidence — how consistently a beat is detected."""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    if len(beats) < 3:
        return 0.0
    # Compute inter-beat intervals
    ibis = np.diff(beats)
    if len(ibis) < 2:
        return 0.0
    # Confidence = inverse of coefficient of variation (lower CV = more consistent)
    mean_ibi = np.mean(ibis)
    if mean_ibi < 1e-10:
        return 0.0
    cv = np.std(ibis) / mean_ibi
    # Convert to 0-1 score (CV of 0 = perfect, CV of 1+ = bad)
    confidence = float(max(0.0, 1.0 - cv))
    return confidence


def compute_flux_stability(y, sr):
    """Spectral flux stability — inverse variance of frame-to-frame spectral change."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if len(onset_env) < 2:
        return 0.0
    flux_var = np.var(onset_env)
    # Invert: higher = more stable (less erratic spectral change)
    if flux_var < 1e-10:
        return 100.0
    return float(1.0 / flux_var)


def compute_hnr_vocal(y, sr):
    """Harmonic-to-noise ratio in vocal frequency band (300-3000Hz)."""
    # Bandpass to vocal range
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    vocal_mask = (freqs >= 300) & (freqs <= 3000)
    S_vocal = S[vocal_mask, :]

    if S_vocal.size == 0:
        return 0.0

    # Compute harmonic and percussive components in vocal band
    H, P = librosa.decompose.hpss(S_vocal)

    harmonic_energy = np.sum(H ** 2)
    noise_energy = np.sum(P ** 2)

    if noise_energy < 1e-10:
        return 60.0  # effectively pure tone

    hnr_db = float(10 * np.log10(harmonic_energy / noise_energy))
    return hnr_db


def compute_rms_variance(y, sr):
    """Variance of RMS energy over time — music is dynamic, buzz/noise is flat."""
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    if len(rms) < 2:
        return 0.0
    return float(np.var(rms))


def compute_mfcc_variance(y, sr):
    """Mean variance across MFCC coefficients — measures timbral diversity.
    Music has changing timbre (instruments, vocals, sections);
    buzz/noise has near-constant timbral signature."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Variance of each coefficient over time, then average
    per_coeff_var = np.var(mfccs, axis=1)
    return float(np.mean(per_coeff_var))


def detect_degeneracy(metrics: dict) -> float:
    """Detect if audio is degenerate (buzz, noise, static, silence).
    Returns 1.0 for degenerate, 0.0 for normal music.
    
    Heuristics based on empirical observation:
    - Very low RMS variance = flat/static signal
    - Very low MFCC variance = no timbral change
    - Abnormally high flux stability = monotonous
    - Very low treble ratio with high mid = typical of buzz/hum
    """
    flags = 0
    
    # Extremely low RMS variance — signal has no dynamics
    if metrics['rms_variance'] < 0.0001:
        flags += 1
    
    # Very low MFCC variance — no timbral change
    if metrics['mfcc_variance'] < 50.0:
        flags += 1
    
    # Abnormally high flux stability — too stable to be music
    if metrics['flux_stability'] > 5.0:
        flags += 1
    
    # Collapsed treble (< 6%) — buzz/hum signature
    if metrics['treble_ratio'] < 0.065:
        flags += 1
    
    # Need at least 2 flags to call it degenerate (avoid false positives)
    return 1.0 if flags >= 2 else 0.0


def compute_perceptual_score(metrics: dict) -> float:
    """Weighted composite perceptual score. Higher = better.
    
    Includes degeneracy detection — degenerate outputs (buzz, noise, static)
    receive a heavy penalty to prevent them from scoring above real music."""
    score = 0.0

    # Spectral contrast: higher = clearer (weight: 3)
    score += metrics['spectral_contrast'] * 3.0

    # Onset strength: higher = punchier (weight: 2)
    score += metrics['onset_strength'] * 2.0

    # Stereo width: 0.3-0.8 is ideal, penalize extremes (weight: 1)
    width = metrics['stereo_width']
    width_score = 1.0 - abs(width - 0.5) * 2.0  # peaks at 0.5
    score += max(0, width_score) * 5.0

    # Balance: lower deviation = better (weight: -3, penalty)
    score -= metrics['balance_deviation'] * 10.0

    # Beat confidence: higher = better (weight: 2)
    score += metrics['beat_confidence'] * 10.0

    # Flux stability: cap at 2.0 (anything higher is suspicious)
    score += min(metrics['flux_stability'], 2.0) * 2.0

    # HNR: higher = cleaner vocals (weight: 1)
    score += max(0, metrics['hnr_vocal']) * 0.5

    # RMS variance: reward dynamic range (music breathes, buzz doesn't)
    score += min(metrics['rms_variance'] * 500.0, 5.0)

    # MFCC variance: reward timbral diversity
    score += min(metrics['mfcc_variance'] * 0.02, 5.0)

    # Degeneracy penalty: -30 points for detected buzz/noise/static
    if metrics['is_degenerate'] > 0.5:
        score -= 30.0

    return score


def analyse_file(wav_path: str, sr: int = 44100) -> dict:
    """Compute all perceptual metrics for a single WAV file."""
    mono, stereo, sr = load_audio_stereo(wav_path, sr=sr)

    metrics = {}
    metrics['spectral_contrast'] = compute_spectral_contrast(mono, sr)
    metrics['onset_strength'] = compute_onset_strength(mono, sr)
    metrics['onset_density'] = compute_onset_density(mono, sr)
    metrics['stereo_width'] = compute_stereo_width(stereo)
    bass_r, mid_r, treble_r, deviation = compute_subband_balance(mono, sr)
    metrics['bass_ratio'] = bass_r
    metrics['mid_ratio'] = mid_r
    metrics['treble_ratio'] = treble_r
    metrics['balance_deviation'] = deviation
    metrics['beat_confidence'] = compute_beat_confidence(mono, sr)
    metrics['flux_stability'] = compute_flux_stability(mono, sr)
    metrics['hnr_vocal'] = compute_hnr_vocal(mono, sr)
    metrics['rms_variance'] = compute_rms_variance(mono, sr)
    metrics['mfcc_variance'] = compute_mfcc_variance(mono, sr)
    metrics['is_degenerate'] = detect_degeneracy(metrics)
    metrics['perceptual_score'] = compute_perceptual_score(metrics)

    return metrics


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc perceptual metrics for HOT-Step benchmark WAVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", help="Path to benchmark run directory")
    parser.add_argument("--combo", default=None,
                        help="Analyse only a specific combo (e.g. euler__linear__apg)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip combos that already have perceptual_results.csv")
    parser.add_argument("--sr", type=int, default=44100,
                        help="Sample rate for analysis (default: 44100)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    # Find combo directories
    if args.combo:
        combo_dirs = [run_dir / args.combo]
        if not combo_dirs[0].exists():
            print(f"ERROR: Combo directory not found: {combo_dirs[0]}")
            sys.exit(1)
    else:
        combo_dirs = sorted([
            d for d in run_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
        ])

    # Filter to only dirs with WAV files
    combo_dirs = [d for d in combo_dirs if list(d.glob("*.wav"))]

    total_combos = len(combo_dirs)
    total_wavs = sum(len(list(d.glob("*.wav"))) for d in combo_dirs)

    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║       PERCEPTUAL ANALYSIS — Post-hoc metrics                 ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print(f"║  Run dir:    {str(run_dir):<49}║")
    print(f"║  Combos:     {total_combos:<49}║")
    print(f"║  WAV files:  {total_wavs:<49}║")
    print(f"║  Est. time:  ~{total_wavs * 8 // 60} min ({total_wavs * 8}s at ~8s/file){' ':<25}║")
    print(f"╚════════════════════════════════════════════════════════════════╝")
    print()

    # CSV header
    fieldnames = [
        'combo', 'filename', 'steps', 'seed',
        'spectral_contrast', 'onset_strength', 'onset_density',
        'stereo_width', 'bass_ratio', 'mid_ratio', 'treble_ratio',
        'balance_deviation', 'beat_confidence', 'flux_stability',
        'hnr_vocal', 'rms_variance', 'mfcc_variance',
        'is_degenerate', 'perceptual_score',
    ]

    # Combined output CSV
    combined_csv_path = run_dir / "perceptual_results.csv"
    all_rows = []

    start_time = time.time()
    processed_combos = 0
    processed_files = 0
    skipped_combos = 0

    for combo_idx, combo_dir in enumerate(combo_dirs, 1):
        combo_name = combo_dir.name

        # Resume support
        per_combo_csv = combo_dir / "perceptual_results.csv"
        if args.resume and per_combo_csv.exists():
            # Load existing results
            with open(per_combo_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row)
            skipped_combos += 1
            print(f"  [{combo_idx}/{total_combos}] SKIP (exists): {combo_name}")
            continue

        wav_files = sorted(combo_dir.glob("*.wav"))
        if not wav_files:
            continue

        elapsed = time.time() - start_time
        if processed_combos > 0:
            avg_per_combo = elapsed / processed_combos
            remaining = (total_combos - combo_idx) * avg_per_combo
            eta_str = f" | ETA: {remaining/60:.0f}m"
        else:
            eta_str = ""

        print(f"  [{combo_idx}/{total_combos}] {combo_name} ({len(wav_files)} files){eta_str}")

        combo_rows = []
        for wav_path in wav_files:
            fname = wav_path.name
            # Parse seed and steps from filename: seed42_steps200.wav
            parts = fname.replace(".wav", "").split("_")
            seed = ""
            steps = ""
            for p in parts:
                if p.startswith("seed"):
                    seed = p[4:]
                elif p.startswith("steps"):
                    steps = p[5:]

            try:
                metrics = analyse_file(str(wav_path), sr=args.sr)
                row = {
                    'combo': combo_name,
                    'filename': fname,
                    'steps': steps,
                    'seed': seed,
                    **{k: f"{v:.6f}" for k, v in metrics.items()},
                }
                combo_rows.append(row)
                processed_files += 1
                print(f"    ✓ {fname}  contrast={metrics['spectral_contrast']:.2f}  "
                      f"onset={metrics['onset_strength']:.2f}  "
                      f"width={metrics['stereo_width']:.3f}  "
                      f"beat={metrics['beat_confidence']:.3f}  "
                      f"pscore={metrics['perceptual_score']:.1f}")
            except Exception as e:
                print(f"    ✗ {fname}  ERROR: {e}")

        # Write per-combo CSV
        if combo_rows:
            with open(per_combo_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(combo_rows)
            all_rows.extend(combo_rows)

        processed_combos += 1

    # Write combined CSV
    if all_rows:
        with open(combined_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    elapsed = time.time() - start_time
    print()
    print(f"{'='*60}")
    print(f"  COMPLETE — {processed_files} files analysed in {elapsed/60:.1f} min")
    print(f"  Skipped: {skipped_combos} combos (resume)")
    print(f"  Combined CSV: {combined_csv_path}")
    print(f"  Per-combo CSVs: perceptual_results.csv in each combo dir")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
