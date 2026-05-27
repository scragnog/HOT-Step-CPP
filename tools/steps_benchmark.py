#!/usr/bin/env python3
"""
steps_benchmark.py — HOT-Step inference steps quality benchmark
================================================================

Generates audio at multiple step counts using the SAME seed & params,
then compares output quality to find the point of diminishing returns.

Usage:
  python steps_benchmark.py <params.json> [--steps 8,25,50,75,100,150,200]
                                           [--seeds 42,123,999]
                                           [--output ./benchmark_results]
                                           [--engine http://127.0.0.1:8085]
                                           [--format wav16]
                                           [--skip-generation]

The params.json is a HOT-Step generation_params JSON (the format stored
in the DB / exported from the UI). The script translates it to AceRequest
format for the engine, overriding inference_steps per run.

Requires: pip install librosa numpy soundfile requests
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

# Optional — graceful degradation if librosa isn't installed
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# ── Param translation (mirrors server/src/services/generation/translateParams.ts) ──

def translate_params(params: dict) -> dict:
    """Translate frontend GenerationParams JSON to AceRequest format."""
    req: dict[str, Any] = {
        "caption": params.get("prompt") or params.get("songDescription")
                   or params.get("caption") or params.get("style") or "",
    }

    # Lyrics
    if params.get("instrumental"):
        req["lyrics"] = "[Instrumental]"
    elif params.get("lyrics"):
        req["lyrics"] = params["lyrics"]

    # Metadata
    if params.get("bpm"):
        req["bpm"] = params["bpm"]
    if params.get("duration"):
        req["duration"] = params["duration"]
    if params.get("keyScale"):
        req["keyscale"] = params["keyScale"]
    if params.get("timeSignature"):
        ts = str(params["timeSignature"])
        req["timesignature"] = ts.split("/")[0] if "/" in ts else ts
    if params.get("vocalLanguage"):
        req["vocal_language"] = params["vocalLanguage"]

    # Seed — always fixed for benchmarks (never random)
    if params.get("seed") is not None:
        req["seed"] = params["seed"]
    else:
        req["seed"] = 42

    # DiT params
    if params.get("inferenceSteps"):
        req["inference_steps"] = params["inferenceSteps"]
    if params.get("guidanceScale") is not None:
        req["guidance_scale"] = params["guidanceScale"]
    if params.get("shift") is not None:
        req["shift"] = params["shift"]
    if params.get("inferMethod"):
        req["infer_method"] = params["inferMethod"]
    if params.get("scheduler"):
        req["scheduler"] = params["scheduler"]
    if params.get("guidanceMode"):
        req["guidance_mode"] = params["guidanceMode"]

    # Model routing
    if params.get("ditModel"):
        req["synth_model"] = params["ditModel"]
    if params.get("lmModel"):
        req["lm_model"] = params["lmModel"]
    if params.get("vaeModel"):
        req["vae_model"] = params["vaeModel"]
    if params.get("embeddingModel"):
        req["emb_model"] = params["embeddingModel"]
    if params.get("loraPath"):
        req["adapter"] = params["loraPath"]
    if params.get("loraScale") is not None:
        req["adapter_scale"] = params["loraScale"]
    if params.get("adapterGroupScales"):
        req["adapter_group_scales"] = params["adapterGroupScales"]
    if params.get("adapterMode"):
        req["adapter_mode"] = params["adapterMode"]

    # Solver sub-params
    if params.get("storkSubsteps") is not None:
        req["stork_substeps"] = params["storkSubsteps"]
    if params.get("beatStability") is not None:
        req["beat_stability"] = params["beatStability"]
    if params.get("frequencyDamping") is not None:
        req["frequency_damping"] = params["frequencyDamping"]
    if params.get("temporalSmoothing") is not None:
        req["temporal_smoothing"] = params["temporalSmoothing"]

    # Guidance sub-params
    if params.get("apgMomentum") is not None:
        req["apg_momentum"] = params["apgMomentum"]
    if params.get("apgNormThreshold") is not None:
        req["apg_norm_threshold"] = params["apgNormThreshold"]

    # DCW
    if params.get("dcwEnabled") is not None:
        req["dcw_enabled"] = params["dcwEnabled"]
    if params.get("dcwMode"):
        req["dcw_mode"] = params["dcwMode"]
    if params.get("dcwScaler") is not None:
        req["dcw_scaler"] = params["dcwScaler"]
    if params.get("dcwHighScaler") is not None:
        req["dcw_high_scaler"] = params["dcwHighScaler"]

    # Latent post-processing
    if params.get("latentShift") is not None:
        req["latent_shift"] = params["latentShift"]
    if params.get("latentRescale") is not None:
        req["latent_rescale"] = params["latentRescale"]
    if params.get("customTimesteps"):
        req["custom_timesteps"] = params["customTimesteps"]

    # Post-VAE denoiser
    if params.get("denoiseStrength") is not None:
        req["denoise_strength"] = params["denoiseStrength"]
    if params.get("denoiseSmoothing") is not None:
        req["denoise_smoothing"] = params["denoiseSmoothing"]
    if params.get("denoiseMix") is not None:
        req["denoise_mix"] = params["denoiseMix"]

    # Lua plugin params
    if params.get("pluginParams") and len(params["pluginParams"]) > 0:
        req["plugin_params"] = params["pluginParams"]

    # CoT caption
    if params.get("useCotCaption") is not None:
        req["use_cot_caption"] = params["useCotCaption"]

    # Negative prompt
    if params.get("negative_prompt"):
        req["negative_prompt"] = params["negative_prompt"]

    # Postprocess plugin (e.g. md_audio_tiled tiled decoder)
    if params.get("postprocessPlugin"):
        req["postprocess_plugin"] = params["postprocessPlugin"]

    return req


# ── Engine client (talks directly to ace-server :8085) ──

class AceEngine:
    def __init__(self, base_url: str, fmt: str = "wav16", keep_loaded: bool = False):
        self.base = base_url.rstrip("/")
        self.fmt = fmt
        self.keep_loaded = keep_loaded
        self.session = requests.Session()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base}/health", timeout=5)
            return r.ok
        except Exception:
            return False

    def submit_synth(self, ace_req: dict) -> str:
        """Submit a synth job, return job ID."""
        params = {}
        if self.fmt != "mp3":
            params["format"] = self.fmt
        if self.keep_loaded:
            params["keep_loaded"] = "1"
        r = self.session.post(
            f"{self.base}/synth",
            json=ace_req,
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        return r.json()["id"]

    def poll_until_done(self, job_id: str, timeout: float = 1800) -> str:
        """Poll job until done/failed. Returns status."""
        start = time.time()
        while time.time() - start < timeout:
            r = self.session.get(
                f"{self.base}/job",
                params={"id": job_id},
                timeout=120,
            )
            r.raise_for_status()
            status = r.json()["status"]
            if status in ("done", "failed", "cancelled"):
                return status
            time.sleep(1)
        raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

    def get_result(self, job_id: str) -> bytes:
        """Fetch audio result bytes."""
        r = self.session.get(
            f"{self.base}/job",
            params={"id": job_id, "result": 1},
            timeout=300,
        )
        r.raise_for_status()
        return r.content


# ── Audio quality analysis ──

def analyze_audio(filepath: str, sr: int = 44100) -> dict:
    """
    Analyse audio quality metrics. Returns dict of metric values.
    Adapted from CompareMP3s35.py with additions.
    """
    if not HAS_LIBROSA:
        raise ImportError("librosa is required for audio analysis. Install with: pip install librosa")

    y, sr = librosa.load(filepath, sr=sr)
    stft = np.abs(librosa.stft(y))
    frequencies = librosa.fft_frequencies(sr=sr)

    # 1. Sibilance: energy in 4–10 kHz relative to total (lower = better)
    sib_band = (frequencies >= 4000) & (frequencies <= 10000)
    sibilance = float(np.mean(
        np.sum(stft[sib_band, :], axis=0) / (np.sum(stft, axis=0) + 1e-6)
    ))

    # 2. Spectral Flatness: how tone-like vs noise-like (higher = cleaner)
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # 3. Peakiness: variance of mid-band energy (lower = smoother, less metallic)
    mid_band = (frequencies >= 1000) & (frequencies <= 8000)
    peakiness = float(np.var(np.mean(stft[mid_band, :], axis=1)))

    # 4. Crest Factor: peak / RMS — dynamic range indicator (higher = punchier)
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-6)
    crest = float(np.max(np.abs(y)) / rms)

    # 5. Noise Floor: mean of quietest 10% of samples (lower = cleaner)
    noise_floor = float(np.mean(np.sort(np.abs(y))[:int(len(y) * 0.1)]))

    # 6. Spectral Centroid: "brightness" (reference-dependent)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # 7. THD proxy: high harmonic energy / fundamental energy
    thd = float(np.sum(stft[100:, :]) / (np.sum(stft[0:100, :]) + 1e-6))

    # 8. Spectral Bandwidth: frequency spread (wider = richer harmonics)
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

    # 9. RMS Energy: overall loudness
    rms_energy = float(np.sqrt(np.mean(y ** 2)))

    return {
        "sibilance": sibilance,
        "flatness": flatness,
        "peakiness": peakiness,
        "crest": crest,
        "noise_floor": noise_floor,
        "centroid": centroid,
        "thd": thd,
        "bandwidth": bandwidth,
        "rms": rms_energy,
    }


def spectral_distance(file_a: str, file_b: str, sr: int = 44100) -> float:
    """
    Compute normalised spectral distance between two audio files.
    Uses mean squared difference of magnitude spectrograms, normalised
    by the average energy. Lower = more similar.
    """
    if not HAS_LIBROSA:
        return float("nan")

    y_a, _ = librosa.load(file_a, sr=sr)
    y_b, _ = librosa.load(file_b, sr=sr)

    # Align lengths
    min_len = min(len(y_a), len(y_b))
    y_a = y_a[:min_len]
    y_b = y_b[:min_len]

    stft_a = np.abs(librosa.stft(y_a))
    stft_b = np.abs(librosa.stft(y_b))

    # Align time frames
    min_frames = min(stft_a.shape[1], stft_b.shape[1])
    stft_a = stft_a[:, :min_frames]
    stft_b = stft_b[:, :min_frames]

    msd = np.mean((stft_a - stft_b) ** 2)
    avg_energy = (np.mean(stft_a ** 2) + np.mean(stft_b ** 2)) / 2 + 1e-10
    return float(msd / avg_energy)


# ── Ranking ──

def compute_weighted_score(metrics: dict) -> float:
    """
    Compute a single quality score from metrics.
    Lower = better. Normalised per-metric then weighted.
    """
    # For each metric: direction (lower_is_better or higher_is_better) and weight
    weights = {
        "sibilance":   (2.5, "lower"),    # Piercing S/T sounds
        "flatness":    (1.5, "higher"),   # Clarity/balance
        "peakiness":   (3.0, "lower"),    # Metallic ringing
        "crest":       (0.8, "higher"),   # Dynamic punch
        "noise_floor": (1.0, "lower"),    # Background static
        "thd":         (2.0, "lower"),    # Digital crunch
        "bandwidth":   (0.5, "higher"),   # Harmonic richness
    }

    # Convert to "lower is better" for all metrics
    score = 0.0
    total_weight = 0.0
    for key, (weight, direction) in weights.items():
        val = metrics.get(key, 0)
        if direction == "higher":
            val = -val  # invert so lower = better
        score += val * weight
        total_weight += weight

    return score / total_weight if total_weight > 0 else 0.0


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="HOT-Step inference steps quality benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sweep with default steps
  python steps_benchmark.py my_params.json

  # Custom step counts
  python steps_benchmark.py my_params.json --steps 8,12,25,50,100,200

  # Multiple seeds for statistical significance
  python steps_benchmark.py my_params.json --seeds 42,123,999

  # Skip generation, just re-analyse existing files
  python steps_benchmark.py my_params.json --skip-generation
        """,
    )
    parser.add_argument("params_json", help="Path to HOT-Step generation_params JSON file")
    parser.add_argument("--steps", default="8,25,50,75,100,150,200",
                        help="Comma-separated step counts to benchmark (default: 8,25,50,75,100,150,200)")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated seeds for reproducibility (default: 42)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: ./benchmark_results_YYYY-MM-DD_HH-MM)")
    parser.add_argument("--engine", default="http://127.0.0.1:8085",
                        help="ace-server URL (default: http://127.0.0.1:8085)")
    parser.add_argument("--format", default="wav16", choices=["wav16", "wav24", "wav32", "mp3"],
                        help="Audio format from engine (default: wav16)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation, only run analysis on existing files")
    parser.add_argument("--skip-lm", action="store_true",
                        help="Skip LM phase — requires audio_codes in the params JSON")
    parser.add_argument("--keep-loaded", action="store_true", default=True,
                        help="Keep models in VRAM between runs (default: on)")
    parser.add_argument("--no-keep-loaded", action="store_true",
                        help="Don't keep models in VRAM (unload after each run)")
    parser.add_argument("--no-pp", action="store_true", default=True,
                        help="Strip post-processing flags (PP-VAE, Spectral Lifter, etc.) for clean benchmarks (default: on)")

    args = parser.parse_args()
    if args.no_keep_loaded:
        args.keep_loaded = False

    # Parse step counts and seeds
    step_counts = [int(s.strip()) for s in args.steps.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Load params JSON
    params_path = Path(args.params_json)
    if not params_path.exists():
        print(f"ERROR: Params file not found: {params_path}")
        sys.exit(1)

    with open(params_path, "r", encoding="utf-8") as f:
        raw_params = json.load(f)

    # Determine if this is already an AceRequest (snake_case) or GenerationParams (camelCase)
    # by checking for camelCase keys
    is_frontend_format = any(k in raw_params for k in [
        "inferenceSteps", "guidanceScale", "inferMethod", "ditModel",
    ])

    if args.output:
        out_dir = Path(args.output)
    else:
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        out_dir = Path(f"./benchmark_results_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = "wav" if args.format.startswith("wav") else "mp3"

    # ── Generation phase ──
    if not args.skip_generation:
        engine = AceEngine(args.engine, args.format, keep_loaded=args.keep_loaded)

        # Health check
        if not engine.health():
            print(f"ERROR: ace-server not reachable at {args.engine}")
            print("Make sure the app is running (LAUNCH.bat or dev.bat)")
            sys.exit(1)

        kl_str = "ON" if args.keep_loaded else "OFF"
        pp_str = "stripped" if args.no_pp else "as-is"
        print(f"╔══════════════════════════════════════════════════════╗")
        print(f"║  HOT-Step Inference Steps Benchmark                 ║")
        print(f"╠══════════════════════════════════════════════════════╣")
        print(f"║  Steps:       {args.steps:<38}║")
        print(f"║  Seeds:       {args.seeds:<38}║")
        print(f"║  Engine:      {args.engine:<38}║")
        print(f"║  Format:      {args.format:<38}║")
        print(f"║  Keep loaded: {kl_str:<38}║")
        print(f"║  Post-proc:   {pp_str:<38}║")
        print(f"║  Output:      {str(out_dir):<38}║")
        print(f"╚══════════════════════════════════════════════════════╝")
        print()

        total_jobs = len(step_counts) * len(seeds)
        job_num = 0

        for seed in seeds:
            for steps in step_counts:
                job_num += 1
                filename = f"seed{seed}_steps{steps}.{ext}"
                filepath = out_dir / filename

                if filepath.exists():
                    print(f"  [{job_num}/{total_jobs}] SKIP (exists): {filename}")
                    continue

                # Build AceRequest
                if is_frontend_format:
                    params_copy = dict(raw_params)
                    params_copy["inferenceSteps"] = steps
                    params_copy["seed"] = seed
                    params_copy["randomSeed"] = False
                    params_copy["batchSize"] = 1
                    # Strip post-processing for clean benchmarks
                    if args.no_pp:
                        params_copy["ppVaeReencode"] = False
                        params_copy["spectralLifterEnabled"] = False
                        params_copy["masteringEnabled"] = False
                        params_copy["vocalNaturalizerEnabled"] = False
                        params_copy["denoiseStrength"] = 0
                        # Keep postprocessPlugin (tiled decoder) — it IS the VAE decode path
                    ace_req = translate_params(params_copy)
                else:
                    ace_req = dict(raw_params)
                    if args.no_pp:
                        ace_req.pop("denoise_strength", None)
                        ace_req.pop("denoise_smoothing", None)
                        ace_req.pop("denoise_mix", None)
                    ace_req["inference_steps"] = steps
                    ace_req["seed"] = seed

                # Force skip LM (benchmark sends direct to /synth)
                # User must provide audio_codes in params OR use skipLm mode
                if args.skip_lm or raw_params.get("skipLm"):
                    # Ensure required metadata exists for synth-only
                    if not ace_req.get("bpm"):
                        ace_req["bpm"] = 120
                    if not ace_req.get("duration") or ace_req["duration"] <= 0:
                        ace_req["duration"] = 30
                    if not ace_req.get("keyscale"):
                        ace_req["keyscale"] = "C major"
                    if not ace_req.get("timesignature"):
                        ace_req["timesignature"] = "4"

                print(f"  [{job_num}/{total_jobs}] Generating: seed={seed}, steps={steps} ...", end=" ", flush=True)

                t0 = time.time()
                try:
                    job_id = engine.submit_synth(ace_req)
                    status = engine.poll_until_done(job_id)

                    if status != "done":
                        print(f"FAILED ({status})")
                        continue

                    audio_data = engine.get_result(job_id)
                    with open(filepath, "wb") as f_out:
                        f_out.write(audio_data)

                    elapsed = time.time() - t0
                    size_kb = len(audio_data) / 1024
                    print(f"OK ({elapsed:.1f}s, {size_kb:.0f} KB)")

                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"ERROR ({elapsed:.1f}s): {e}")
                    continue

        print()

    # ── Analysis phase ──
    if not HAS_LIBROSA:
        print("WARNING: librosa not installed. Skipping audio analysis.")
        print("Install with: pip install librosa soundfile")
        sys.exit(0)

    print("═══════════════════════════════════════════════════════")
    print("  ANALYSIS")
    print("═══════════════════════════════════════════════════════")
    print()

    # Collect all generated files
    results = []
    for seed in seeds:
        for steps in step_counts:
            filename = f"seed{seed}_steps{steps}.{ext}"
            filepath = out_dir / filename
            if not filepath.exists():
                print(f"  WARNING: Missing file: {filename}")
                continue

            print(f"  Analysing: {filename} ...", end=" ", flush=True)
            try:
                metrics = analyze_audio(str(filepath))
                metrics["seed"] = seed
                metrics["steps"] = steps
                metrics["filename"] = filename
                metrics["filepath"] = str(filepath)
                metrics["quality_score"] = compute_weighted_score(metrics)
                results.append(metrics)
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")

    if not results:
        print("\nNo files to analyse.")
        sys.exit(1)

    # ── Spectral convergence ──
    print()
    print("  Computing spectral convergence...")
    for seed in seeds:
        seed_results = sorted(
            [r for r in results if r["seed"] == seed],
            key=lambda r: r["steps"],
        )
        for i, r in enumerate(seed_results):
            if i == 0:
                r["spectral_delta"] = float("nan")
            else:
                prev = seed_results[i - 1]
                r["spectral_delta"] = spectral_distance(
                    prev["filepath"], r["filepath"]
                )

    # ── Results table ──
    print()
    print("═══════════════════════════════════════════════════════════════════════════════════════════════════════════════")
    print("  RESULTS (sorted by quality score — lower = better)")
    print("═══════════════════════════════════════════════════════════════════════════════════════════════════════════════")
    print()

    # Header
    hdr = (
        f"{'FILE':<28} │ {'STEPS':>5} │ {'SEED':>6} │ "
        f"{'SIB↓':>7} │ {'FLAT↑':>7} │ {'PEAK↓':>8} │ "
        f"{'CREST↑':>7} │ {'NOISE↓':>9} │ {'THD↓':>7} │ "
        f"{'BW↑':>7} │ {'Δ-SPEC':>7} │ {'SCORE':>7} │ {'RANK':>4}"
    )
    print(hdr)
    print("─" * len(hdr))

    # Sort by quality score
    sorted_results = sorted(results, key=lambda r: r["quality_score"])
    for rank, r in enumerate(sorted_results, 1):
        delta_str = f"{r['spectral_delta']:.4f}" if not np.isnan(r["spectral_delta"]) else "  ---"
        print(
            f"{r['filename']:<28} │ {r['steps']:>5} │ {r['seed']:>6} │ "
            f"{r['sibilance']:>7.4f} │ {r['flatness']:>7.4f} │ {r['peakiness']:>8.1f} │ "
            f"{r['crest']:>7.1f} │ {r['noise_floor']:>9.6f} │ {r['thd']:>7.3f} │ "
            f"{r['bandwidth']:>7.0f} │ {delta_str:>7} │ {r['quality_score']:>7.4f} │ {rank:>4}"
        )

    print("─" * len(hdr))

    # ── Per-step average (across seeds) ──
    if len(seeds) > 1:
        print()
        print("  PER-STEP AVERAGES (across seeds)")
        print()
        avg_hdr = f"{'STEPS':>5} │ {'SIB↓':>7} │ {'FLAT↑':>7} │ {'PEAK↓':>8} │ {'CREST↑':>7} │ {'THD↓':>7} │ {'Δ-SPEC':>7} │ {'SCORE':>7}"
        print(avg_hdr)
        print("─" * len(avg_hdr))

        for steps in step_counts:
            step_results = [r for r in results if r["steps"] == steps]
            if not step_results:
                continue
            n = len(step_results)
            avg = {
                "sibilance": sum(r["sibilance"] for r in step_results) / n,
                "flatness": sum(r["flatness"] for r in step_results) / n,
                "peakiness": sum(r["peakiness"] for r in step_results) / n,
                "crest": sum(r["crest"] for r in step_results) / n,
                "thd": sum(r["thd"] for r in step_results) / n,
                "spectral_delta": np.nanmean([r["spectral_delta"] for r in step_results]),
                "quality_score": sum(r["quality_score"] for r in step_results) / n,
            }
            delta_str = f"{avg['spectral_delta']:.4f}" if not np.isnan(avg["spectral_delta"]) else "  ---"
            print(
                f"{steps:>5} │ {avg['sibilance']:>7.4f} │ {avg['flatness']:>7.4f} │ "
                f"{avg['peakiness']:>8.1f} │ {avg['crest']:>7.1f} │ {avg['thd']:>7.3f} │ "
                f"{delta_str:>7} │ {avg['quality_score']:>7.4f}"
            )
        print("─" * len(avg_hdr))

    # ── Convergence analysis ──
    print()
    print("  CONVERGENCE ANALYSIS (spectral delta between consecutive step counts)")
    print("  Lower Δ-SPEC = less change from previous step count = diminishing returns")
    print()
    for seed in seeds:
        seed_results = sorted(
            [r for r in results if r["seed"] == seed],
            key=lambda r: r["steps"],
        )
        if len(seeds) > 1:
            print(f"  Seed {seed}:")
        for r in seed_results:
            delta = r.get("spectral_delta", float("nan"))
            if np.isnan(delta):
                bar = "  (baseline)"
            else:
                bar_len = int(delta * 500)  # scale for visual
                bar = "█" * min(bar_len, 60) + f"  {delta:.6f}"
            print(f"    {r['steps']:>5} steps: {bar}")
        print()

    # ── Save CSV ──
    csv_path = out_dir / "benchmark_results.csv"
    fieldnames = [
        "filename", "steps", "seed", "sibilance", "flatness", "peakiness",
        "crest", "noise_floor", "centroid", "thd", "bandwidth", "rms",
        "spectral_delta", "quality_score",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted_results:
            writer.writerow(r)

    print(f"  Results saved to: {csv_path}")

    # ── Save config ──
    config_path = out_dir / "benchmark_config.json"
    config = {
        "params_json": str(params_path.resolve()),
        "step_counts": step_counts,
        "seeds": seeds,
        "engine": args.engine,
        "format": args.format,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generation_params": raw_params,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"  Config saved to:  {config_path}")
    print()
    print("  Done! 🎵")


if __name__ == "__main__":
    main()
