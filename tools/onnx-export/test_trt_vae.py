#!/usr/bin/env python3
"""Validate and benchmark VAE ONNX model with TensorRT Execution Provider.

Compares CUDA EP (baseline) against TensorRT EP for latency and numerical
accuracy. Caches TRT engines for subsequent runs.

Usage:
    python test_trt_vae.py --onnx models/onnx/vae_decoder.onnx
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def check_providers():
    """Check which ONNX Runtime execution providers are available."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is not installed.")
        print("Install with: pip install onnxruntime-gpu")
        sys.exit(1)

    available = ort.get_available_providers()
    print(f"onnxruntime version: {ort.__version__}")
    print(f"Available providers: {available}")

    has_cuda = "CUDAExecutionProvider" in available
    has_trt = "TensorrtExecutionProvider" in available

    if not has_cuda:
        print("\n[WARN] CUDAExecutionProvider is NOT available.")
        print("  You likely have `onnxruntime` (CPU-only) instead of `onnxruntime-gpu`.")
        print("  Install with: pip install onnxruntime-gpu")
        print("  (You may need to uninstall onnxruntime first)")

        # Check which package is installed
        try:
            import importlib.metadata
            try:
                ver = importlib.metadata.version("onnxruntime-gpu")
                print(f"  onnxruntime-gpu version: {ver}")
            except importlib.metadata.PackageNotFoundError:
                print("  onnxruntime-gpu: NOT installed")
            try:
                ver = importlib.metadata.version("onnxruntime")
                print(f"  onnxruntime (CPU): {ver}")
            except importlib.metadata.PackageNotFoundError:
                pass
        except ImportError:
            pass

    return has_cuda, has_trt


def benchmark_session(sess, input_data, warmup=3, iterations=20):
    """Benchmark an ONNX Runtime session.

    Args:
        sess: ONNX Runtime InferenceSession.
        input_data: Dict of input name → numpy array.
        warmup: Number of warmup iterations.
        iterations: Number of timed iterations.

    Returns:
        Tuple of (output_array, mean_latency_ms, std_latency_ms).
    """
    # Warmup
    for _ in range(warmup):
        output = sess.run(None, input_data)

    # Timed runs
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        output = sess.run(None, input_data)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    return output[0], latencies.mean(), latencies.std()


def run_benchmark(onnx_path: str, trt_cache_dir: str):
    """Run the full benchmark comparing CUDA EP vs TRT EP.

    Args:
        onnx_path: Path to the ONNX model file.
        trt_cache_dir: Directory to cache TRT engines.
    """
    import onnxruntime as ort

    has_cuda, has_trt = check_providers()

    if not has_cuda:
        print("\nCannot run GPU benchmarks without CUDAExecutionProvider.")
        print("Falling back to CPU-only test...")
        run_cpu_test(onnx_path)
        return

    # Test input: 10 seconds of audio (250 latent frames)
    test_latents = np.random.randn(1, 64, 250).astype(np.float32)
    input_data = {"latents": test_latents}
    print(f"\nTest input shape: {test_latents.shape}")
    print(f"Expected output: [1, 2, {250 * 1920}] samples")

    # ── CUDA EP benchmark ──
    print("\n" + "=" * 60)
    print("CUDA EP Benchmark")
    print("=" * 60)

    cuda_opts = ort.SessionOptions()
    cuda_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        cuda_sess = ort.InferenceSession(
            onnx_path,
            sess_options=cuda_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    except Exception as e:
        print(f"Failed to create CUDA session: {e}")
        print("Falling back to CPU-only test...")
        run_cpu_test(onnx_path)
        return

    cuda_output, cuda_mean, cuda_std = benchmark_session(cuda_sess, input_data)
    print(f"Output shape: {cuda_output.shape}")
    print(f"Latency: {cuda_mean:.2f} ± {cuda_std:.2f} ms")

    # ── TensorRT EP benchmark ──
    if not has_trt:
        print("\n" + "=" * 60)
        print("TensorRT EP: NOT AVAILABLE")
        print("=" * 60)
        print("TensorrtExecutionProvider is not available in this onnxruntime build.")
        print("To enable TRT:")
        print("  1. Install onnxruntime-gpu with TRT support")
        print("  2. Ensure TensorRT libraries are on PATH")
        print("\nSkipping TRT benchmark. CUDA EP results above are the baseline.")
        return

    print("\n" + "=" * 60)
    print("TensorRT EP Benchmark")
    print("=" * 60)

    os.makedirs(trt_cache_dir, exist_ok=True)

    trt_opts = ort.SessionOptions()
    trt_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    trt_provider_options = {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": trt_cache_dir,
        "trt_max_workspace_size": str(8 * 1024 * 1024 * 1024),  # 8GB
    }

    print("Building TRT engine (first run may take minutes)...")
    t0 = time.time()
    try:
        trt_sess = ort.InferenceSession(
            onnx_path,
            sess_options=trt_opts,
            providers=[
                ("TensorrtExecutionProvider", trt_provider_options),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
    except Exception as e:
        print(f"Failed to create TRT session: {e}")
        print("TRT EP may not be properly configured. Skipping TRT benchmark.")
        return

    engine_time = time.time() - t0
    print(f"TRT engine ready in {engine_time:.1f}s")

    trt_output, trt_mean, trt_std = benchmark_session(trt_sess, input_data)
    print(f"Output shape: {trt_output.shape}")
    print(f"Latency: {trt_mean:.2f} ± {trt_std:.2f} ms")

    # ── Comparison ──
    print("\n" + "=" * 60)
    print("Comparison: CUDA EP vs TensorRT EP")
    print("=" * 60)

    abs_diff = np.abs(cuda_output - trt_output)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    print(f"CUDA EP latency:  {cuda_mean:.2f} ± {cuda_std:.2f} ms")
    print(f"TRT EP latency:   {trt_mean:.2f} ± {trt_std:.2f} ms")
    print(f"Speedup:          {cuda_mean / trt_mean:.2f}x")
    print(f"Max abs diff:     {max_diff:.6e}")
    print(f"Mean abs diff:    {mean_diff:.6e}")

    if max_diff < 0.05:
        print("\n[PASS] Numerical accuracy: GOOD (fp16 rounding is expected)")
    elif max_diff < 0.5:
        print("\n[WARN] Numerical accuracy: ACCEPTABLE (fp16 precision loss)")
    else:
        print(f"\n[FAIL] Numerical accuracy: POOR (max diff = {max_diff:.4f})")
        print("  This may indicate a TRT conversion issue.")


def run_cpu_test(onnx_path: str):
    """Fallback: run a basic CPU test to verify the ONNX model loads."""
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("CPU-only Test (fallback)")
    print("=" * 60)

    test_latents = np.random.randn(1, 64, 50).astype(np.float32)
    input_data = {"latents": test_latents}

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    output = sess.run(None, input_data)

    print(f"Input shape:  {test_latents.shape}")
    print(f"Output shape: {output[0].shape}")
    print(f"Output range: [{output[0].min():.4f}, {output[0].max():.4f}]")
    print("[PASS] Model loads and runs on CPU successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and benchmark VAE ONNX model with TensorRT EP"
    )
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Path to the exported ONNX file",
    )
    parser.add_argument(
        "--trt-cache",
        type=str,
        default=None,
        help="Directory for TRT engine cache (default: alongside ONNX file)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of benchmark iterations (default: 20)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.onnx):
        print(f"ERROR: ONNX file not found: {args.onnx}")
        sys.exit(1)

    trt_cache = args.trt_cache
    if trt_cache is None:
        trt_cache = os.path.join(os.path.dirname(args.onnx), "trt_cache")

    run_benchmark(args.onnx, trt_cache)
    print("\nDone!")


if __name__ == "__main__":
    main()
