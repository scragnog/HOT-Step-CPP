"""Convert SuperSep .ckpt models to ONNX format for C++ inference.

Uses ZFTurbo's MSS_ONNX_TensorRT export_to_onnx approach adapted
for the specific models used in the HOT-Step SuperSep pipeline.

Requirements:
    pip install -r requirements.txt

Usage:
    python convert_models.py --model-dir D:/models/supersep --output-dir ./onnx

This is a BUILD-TIME tool only. The resulting .onnx files ship with the app.
No Python is needed at runtime.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml


# ── Model architectures ─────────────────────────────────────────────────

MODELS = {
    "bs_roformer": {
        "ckpt": "BS-Roformer-SW.ckpt",
        "config": "bs_roformer_sw.yaml",
        "type": "bs_roformer",
        "output": "bs_roformer_sw.onnx",
        "description": "Stage 1: Primary 6-stem split (vocals/bass/drums/guitar/piano/other)",
    },
    "mel_band_roformer": {
        "ckpt": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "config": "mel_band_roformer_karaoke.yaml",
        "type": "mel_band_roformer",
        "output": "mel_band_roformer_karaoke.onnx",
        "description": "Stage 2: Vocal sub-separation (lead/backing)",
    },
    "mdx23c": {
        "ckpt": "MDX23C-DrumSep-aufr33-jarredou.ckpt",
        "config": "mdx23c_drumsep.yaml",
        "type": "segm_models",
        "output": "mdx23c_drumsep.onnx",
        "description": "Stage 3: Drum sub-separation (kick/snare/toms/hihat/ride/crash)",
    },
    "htdemucs": {
        "ckpt": None,  # Uses yaml config to fetch from torch.hub
        "config": "htdemucs_6s.yaml",
        "type": "htdemucs",
        "output": "htdemucs_6s.onnx",
        "description": "Stage 4: 'Other' refinement via Demucs v4",
    },
}


def find_config(model_dir: str, config_name: str) -> str:
    """Find config YAML, checking model_dir and this script's configs/ dir."""
    candidates = [
        os.path.join(model_dir, config_name),
        os.path.join(os.path.dirname(__file__), "configs", config_name),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return ""


def export_roformer(model_dir: str, model_info: dict, output_path: str, opset: int = 17):
    """Export BS-Roformer or Mel-Band-Roformer to ONNX."""
    from models.bs_roformer import BSRoformer, MelBandRoformer

    config_path = find_config(model_dir, model_info["config"])
    if not config_path:
        print(f"  ERROR: Config {model_info['config']} not found")
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", config)

    if model_info["type"] == "mel_band_roformer":
        model = MelBandRoformer(**model_config)
    else:
        model = BSRoformer(**model_config)

    ckpt_path = os.path.join(model_dir, model_info["ckpt"])
    if not os.path.isfile(ckpt_path):
        print(f"  ERROR: Checkpoint {ckpt_path} not found")
        return False

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    # For ONNX export, we need the inner model (without STFT wrapper)
    # STFT/iSTFT will be handled in C++
    n_fft = model_config.get("n_fft", 2048)
    hop_length = model_config.get("hop_length", 441)

    # Create dummy spectrogram input (post-STFT)
    batch_size = 1
    n_channels = 2  # stereo
    freq_bins = n_fft // 2 + 1
    time_frames = 256  # variable

    # The model expects complex spectrogram as (batch, channels, freq, time, 2)
    dummy_input = torch.randn(batch_size, n_channels, freq_bins, time_frames, 2)

    print(f"  Exporting with input shape: {dummy_input.shape}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["spectrogram"],
        output_names=["mask"],
        dynamic_axes={
            "spectrogram": {0: "batch", 3: "time_frames"},
            "mask": {0: "batch", 3: "time_frames"},
        },
    )
    print(f"  Saved: {output_path}")
    return True


def export_mdx23c(model_dir: str, model_info: dict, output_path: str, opset: int = 17):
    """Export MDX23C to ONNX."""
    from models.mdx23c import TFC_TDF_net

    config_path = find_config(model_dir, model_info["config"])
    if not config_path:
        print(f"  ERROR: Config {model_info['config']} not found")
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", config)
    model = TFC_TDF_net(model_config)

    ckpt_path = os.path.join(model_dir, model_info["ckpt"])
    if not os.path.isfile(ckpt_path):
        print(f"  ERROR: Checkpoint {ckpt_path} not found")
        return False

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    n_fft = model_config.get("n_fft", 4096)
    freq_bins = n_fft // 2 + 1
    time_frames = 256

    dummy_input = torch.randn(1, 2, freq_bins, time_frames, 2)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["spectrogram"],
        output_names=["mask"],
        dynamic_axes={
            "spectrogram": {0: "batch", 3: "time_frames"},
            "mask": {0: "batch", 3: "time_frames"},
        },
    )
    print(f"  Saved: {output_path}")
    return True


def export_htdemucs(model_dir: str, model_info: dict, output_path: str, opset: int = 17):
    """Export HTDemucs to ONNX."""
    import demucs.pretrained

    model = demucs.pretrained.get_model("htdemucs_6s")
    model.eval()

    # HTDemucs operates on raw waveform
    sr = model.samplerate
    duration_seconds = 10
    n_samples = sr * duration_seconds

    dummy_input = torch.randn(1, 2, n_samples)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["waveform"],
        output_names=["sources"],
        dynamic_axes={
            "waveform": {0: "batch", 2: "samples"},
            "sources": {0: "batch", 3: "samples"},
        },
    )
    print(f"  Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert SuperSep models to ONNX")
    parser.add_argument("--model-dir", required=True, help="Directory containing .ckpt model files")
    parser.add_argument("--output-dir", default="./onnx", help="Output directory for .onnx files")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--models", nargs="*", choices=list(MODELS.keys()), help="Specific models to convert (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    models_to_convert = args.models or list(MODELS.keys())

    print(f"SuperSep ONNX Converter")
    print(f"  Model dir: {args.model_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Opset: {args.opset}")
    print()

    results = {}
    for name in models_to_convert:
        info = MODELS[name]
        output_path = os.path.join(args.output_dir, info["output"])
        print(f"[{name}] {info['description']}")

        try:
            if info["type"] in ("bs_roformer", "mel_band_roformer"):
                ok = export_roformer(args.model_dir, info, output_path, args.opset)
            elif info["type"] == "segm_models":
                ok = export_mdx23c(args.model_dir, info, output_path, args.opset)
            elif info["type"] == "htdemucs":
                ok = export_htdemucs(args.model_dir, info, output_path, args.opset)
            else:
                print(f"  SKIP: Unknown model type '{info['type']}'")
                ok = False
        except Exception as e:
            print(f"  FAILED: {e}")
            ok = False

        results[name] = ok
        print()

    print("Summary:")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
