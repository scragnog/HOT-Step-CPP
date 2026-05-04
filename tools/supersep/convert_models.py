"""Convert SuperSep .ckpt models to ONNX format for C++ inference.

Uses ZFTurbo's Music-Source-Separation-Training model architecture classes.

Usage:
    $env:PYTHONPATH = "D:\\Ace-Step-Latest\\Music-Source-Separation-Training"
    py convert_models.py --model-dir "D:\\Ace-Step-Latest\\SuperSep\\models" --output-dir "D:\\Ace-Step-Latest\\hot-step-cpp\\models\\supersep"

This is a BUILD-TIME tool only. No Python is needed at runtime.
"""

import argparse
import os
import sys

import torch
import yaml

# Force UTF-8 output on Windows to avoid codec errors from emoji in torch.onnx
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── Model definitions ────────────────────────────────────────────────────

MODELS = {
    "bs_roformer": {
        "ckpt": "BS-Roformer-SW.ckpt",
        "config": "BS-Roformer-SW.yaml",
        "output": "bs_roformer_sw.onnx",
        "description": "Stage 1: Primary 6-stem split",
    },
    "mel_band_roformer": {
        "ckpt": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "config": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956_config.yaml",
        "output": "mel_band_roformer_karaoke.onnx",
        "description": "Stage 2: Vocal sub-separation (lead/backing)",
    },
    "mdx23c": {
        "ckpt": "MDX23C-DrumSep-aufr33-jarredou.ckpt",
        "config": "config_drumsep_mdx23c.yaml",
        "output": "mdx23c_drumsep.onnx",
        "description": "Stage 3: Drum sub-separation",
    },
    "htdemucs": {
        "ckpt": None,  # Downloaded via torch.hub
        "config": "htdemucs_6s.yaml",
        "output": "htdemucs_6s.onnx",
        "description": "Stage 4: 'Other' refinement via HTDemucs v4",
    },
}


def load_config(model_dir, config_name):
    """Load YAML config, checking model_dir first, then script's directory."""
    for base in [model_dir, os.path.join(os.path.dirname(__file__), "configs")]:
        p = os.path.join(base, config_name)
        if os.path.isfile(p):
            with open(p) as f:
                return yaml.full_load(f)
    return None


def onnx_export_legacy(model, dummy_input, output_path, opset,
                       input_names, output_names, dynamic_axes=None):
    """Export using the legacy TorchScript-based ONNX exporter.

    PyTorch 2.x defaults to the dynamo-based exporter which chokes on
    data-dependent control flow in BSRoformer/MelBandRoformer. Force
    the legacy path by setting dynamo=False.
    """
    kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )
    if dynamic_axes:
        kwargs["dynamic_axes"] = dynamic_axes

    # PyTorch >= 2.6 supports dynamo=False to force legacy exporter
    try:
        sig = torch.onnx.export.__code__.co_varnames
        if "dynamo" in sig:
            kwargs["dynamo"] = False
    except Exception:
        pass

    with torch.no_grad():
        torch.onnx.export(model, dummy_input, output_path, **kwargs)


def export_bs_roformer(model_dir, output_path, opset):
    """Export BS-Roformer to ONNX."""
    from models.bs_roformer.bs_roformer import BSRoformer

    info = MODELS["bs_roformer"]
    config = load_config(model_dir, info["config"])
    if not config:
        print(f"  ERROR: Config {info['config']} not found")
        return False

    model_cfg = config.get("model", config)
    print(f"  Config keys: {list(model_cfg.keys())}")

    model = BSRoformer(**model_cfg)

    ckpt_path = os.path.join(model_dir, info["ckpt"])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    # BS-Roformer does internal STFT + chunking. Export with fixed chunk size
    # matching its internal segment_size. The C++ engine handles chunking.
    sr = config.get("audio", {}).get("sample_rate", 44100)
    # Use the model's segment_size if available, else 10s
    segment = getattr(model, 'segment_size', None)
    if segment:
        n_samples = int(segment * sr)
        print(f"  Using model segment_size: {segment}s = {n_samples} samples")
    else:
        n_samples = sr * 10
        print(f"  Using default 10s = {n_samples} samples")

    dummy = torch.randn(1, 2, n_samples)
    print(f"  Input shape: {dummy.shape} @ {sr}Hz")

    onnx_export_legacy(
        model, dummy, output_path, opset,
        input_names=["waveform"],
        output_names=["sources"],
        # Fixed input shape — C++ engine handles chunking & overlap-add
    )
    print(f"  Saved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")
    return True


def export_mel_band_roformer(model_dir, output_path, opset):
    """Export Mel-Band RoFormer to ONNX."""
    from models.bs_roformer.mel_band_roformer import MelBandRoformer

    info = MODELS["mel_band_roformer"]
    config = load_config(model_dir, info["config"])
    if not config:
        print(f"  ERROR: Config {info['config']} not found")
        return False

    model_cfg = config.get("model", config)
    print(f"  Config keys: {list(model_cfg.keys())}")

    model = MelBandRoformer(**model_cfg)

    ckpt_path = os.path.join(model_dir, info["ckpt"])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    sr = config.get("audio", {}).get("sample_rate", 44100)
    segment = getattr(model, 'segment_size', None)
    if segment:
        n_samples = int(segment * sr)
        print(f"  Using model segment_size: {segment}s = {n_samples} samples")
    else:
        n_samples = sr * 10
        print(f"  Using default 10s = {n_samples} samples")

    dummy = torch.randn(1, 2, n_samples)
    print(f"  Input shape: {dummy.shape} @ {sr}Hz")

    onnx_export_legacy(
        model, dummy, output_path, opset,
        input_names=["waveform"],
        output_names=["sources"],
    )
    print(f"  Saved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")
    return True


def export_mdx23c(model_dir, output_path, opset):
    """Export MDX23C to ONNX."""
    from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net

    info = MODELS["mdx23c"]
    config = load_config(model_dir, info["config"])
    if not config:
        print(f"  ERROR: Config {info['config']} not found")
        return False

    model_cfg = config.get("model", config)
    print(f"  Config keys: {list(model_cfg.keys())}")

    model = TFC_TDF_net(model_cfg)

    ckpt_path = os.path.join(model_dir, info["ckpt"])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    sr = config.get("audio", {}).get("sample_rate", 44100)
    chunk = config.get("audio", {}).get("chunk_size", 261120)
    dummy = torch.randn(1, 2, chunk)
    print(f"  Input shape: {dummy.shape} (chunk_size={chunk} @ {sr}Hz)")

    onnx_export_legacy(
        model, dummy, output_path, opset,
        input_names=["waveform"],
        output_names=["sources"],
    )
    print(f"  Saved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")
    return True


def export_htdemucs(model_dir, output_path, opset):
    """Export HTDemucs to ONNX."""
    try:
        import demucs.pretrained
    except ImportError:
        print("  ERROR: 'demucs' package not installed. Run: py -m pip install demucs")
        return False

    model = demucs.pretrained.get_model("htdemucs_6s")
    model.eval()

    sr = model.samplerate
    duration = 10
    n_samples = sr * duration
    dummy = torch.randn(1, 2, n_samples)
    print(f"  Input shape: {dummy.shape} ({duration}s @ {sr}Hz)")

    onnx_export_legacy(
        model, dummy, output_path, opset,
        input_names=["waveform"],
        output_names=["sources"],
    )
    print(f"  Saved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")
    return True


EXPORTERS = {
    "bs_roformer": export_bs_roformer,
    "mel_band_roformer": export_mel_band_roformer,
    "mdx23c": export_mdx23c,
    "htdemucs": export_htdemucs,
}


def main():
    parser = argparse.ArgumentParser(description="Convert SuperSep models to ONNX")
    parser.add_argument("--model-dir", required=True, help="Directory containing .ckpt model files")
    parser.add_argument("--output-dir", default="./onnx", help="Output directory for .onnx files")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--models", nargs="*", choices=list(MODELS.keys()),
                        help="Specific models to convert (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    to_convert = args.models or list(MODELS.keys())

    print("SuperSep ONNX Converter")
    print(f"  Model dir:  {args.model_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Opset:      {args.opset}")
    print()

    results = {}
    for name in to_convert:
        info = MODELS[name]
        out_path = os.path.join(args.output_dir, info["output"])
        print(f"[{name}] {info['description']}")

        try:
            ok = EXPORTERS[name](args.model_dir, out_path, args.opset)
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok = False

        results[name] = ok
        print()

    print("=" * 50)
    print("Summary:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
