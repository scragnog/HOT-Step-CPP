#!/usr/bin/env python3
"""
export_dit.py — Export AceStep DiT forward pass to ONNX for TensorRT acceleration.

Exports the SINGLE FORWARD PASS (one diffusion timestep) of the DiT model,
wrapping the full 32-layer transformer + attention mask computation + RoPE
into a single ONNX graph with 4 simplified inputs.

Usage:
    python export_dit.py --model-dir <path-to-safetensors-model> --output <output.onnx>

The diffusion loop, guidance (APG/CFG), and solvers stay in C++.
TRT compiles the ONNX graph once; LoRA adapters use IRefitter weight swapping.
"""

import argparse
import sys
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# We need the model's own code
# The model dir contains modeling_acestep_v15_xl_base.py


class DiTForwardWrapper(nn.Module):
    """
    Wrapper around AceStepDiTModel.forward() that simplifies the interface
    for ONNX export.
    
    ONNX inputs (4 total):
        input_latents:  [B, T, 192]  fp16  — pre-concatenated [context_latents, xt]
        enc_hidden:     [B, S, 2048] fp16  — encoder hidden states
        t:              [B]          fp32  — current timestep
        t_r:            [B]          fp32  — reference timestep
    
    ONNX output:
        velocity:       [B, T, 64]   fp16  — predicted flow velocity
    
    Masks and position IDs are computed internally from T and S.
    """
    
    def __init__(self, dit_model):
        super().__init__()
        self.dit = dit_model
        self.config = dit_model.config
    
    def forward(self, input_latents, enc_hidden, t, t_r):
        """
        Args:
            input_latents: [B, T, 192] — concatenated context + noise latents
            enc_hidden:    [B, S, 2048] — encoder hidden states  
            t:             [B] — timestep
            t_r:           [B] — reference timestep
        Returns:
            velocity:      [B, T, 64] — predicted velocity
        """
        B = input_latents.shape[0]
        T = input_latents.shape[1]
        
        # Split input_latents into context (128 dim) and noise (64 dim)
        context_latents = input_latents[:, :, :128]
        hidden_states = input_latents[:, :, 128:]
        
        # Call the DiT forward pass
        # We pass None for masks — the model computes them internally
        # We disable cache, output_attentions, etc.
        outputs = self.dit(
            hidden_states=hidden_states,
            timestep=t,
            timestep_r=t_r,
            attention_mask=None,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=None,
            context_latents=context_latents,
            use_cache=False,
            past_key_values=None,
            output_attentions=False,
        )
        
        # outputs[0] is the velocity prediction [B, T, 64]
        velocity = outputs[0]
        return velocity


def load_dit_model(model_dir: str, device: str = "cuda", dtype=torch.float16):
    """Load the AceStepDiTModel from a safetensors checkpoint."""
    model_dir = Path(model_dir)
    
    # Add model dir to sys.path so we can import the model code
    sys.path.insert(0, str(model_dir))
    
    # Import the model class
    from modeling_acestep_v15_xl_base import AceStepConditionGenerationModel
    from configuration_acestep_v15 import AceStepConfig
    
    # Load config
    import json
    with open(model_dir / "config.json") as f:
        config_dict = json.load(f)
    
    config = AceStepConfig(**config_dict)
    # Force SDPA for ONNX export (no flash attention)
    config._attn_implementation = "sdpa"
    
    print(f"[export_dit] Loading model from {model_dir}...")
    t0 = time.time()
    
    # Load the full model, then extract just the decoder (DiT)
    from safetensors.torch import load_file
    state_dict = load_file(str(model_dir / "model.safetensors"))
    
    # Create the full model to get proper initialization
    full_model = AceStepConditionGenerationModel(config)
    full_model.load_state_dict(state_dict, strict=False)
    
    # Extract just the decoder (DiT model)
    dit_model = full_model.decoder
    dit_model = dit_model.to(device=device, dtype=dtype)
    dit_model.eval()
    
    t1 = time.time()
    print(f"[export_dit] Model loaded in {t1-t0:.1f}s")
    print(f"[export_dit] DiT: {sum(p.numel() for p in dit_model.parameters())/1e9:.2f}B params")
    
    return dit_model, config


def export_onnx(dit_model, config, output_path: str, opset: int = 18):
    """Export the DiT forward pass to ONNX."""
    device = next(dit_model.parameters()).device
    dtype = next(dit_model.parameters()).dtype
    
    wrapper = DiTForwardWrapper(dit_model)
    wrapper.eval()
    
    # Create dummy inputs for tracing
    B = 1
    T = 512   # typical sequence length (divisible by patch_size=2)
    S = 256   # typical encoder sequence length
    
    dummy_input_latents = torch.randn(B, T, 192, device=device, dtype=dtype)
    dummy_enc_hidden = torch.randn(B, S, 2048, device=device, dtype=dtype)
    dummy_t = torch.tensor([0.5], device=device, dtype=torch.float32)
    dummy_t_r = torch.tensor([0.5], device=device, dtype=torch.float32)
    
    print(f"[export_dit] Tracing with shapes: input_latents={list(dummy_input_latents.shape)}, "
          f"enc_hidden={list(dummy_enc_hidden.shape)}, t={list(dummy_t.shape)}")
    
    # Test forward pass first
    print("[export_dit] Testing forward pass...")
    with torch.no_grad():
        test_out = wrapper(dummy_input_latents, dummy_enc_hidden, dummy_t, dummy_t_r)
    print(f"[export_dit] Output shape: {list(test_out.shape)} (expected [{B}, {T}, 64])")
    
    # Export to ONNX
    print(f"[export_dit] Exporting to ONNX (opset {opset})...")
    t0 = time.time()
    
    torch.onnx.export(
        wrapper,
        (dummy_input_latents, dummy_enc_hidden, dummy_t, dummy_t_r),
        output_path,
        opset_version=opset,
        input_names=["input_latents", "enc_hidden", "t", "t_r"],
        output_names=["velocity"],
        dynamic_axes={
            "input_latents": {0: "batch", 1: "seq_len"},
            "enc_hidden":    {0: "batch", 1: "enc_seq_len"},
            "t":             {0: "batch"},
            "t_r":           {0: "batch"},
            "velocity":      {0: "batch", 1: "seq_len"},
        },
        do_constant_folding=True,
        export_params=True,
    )
    
    t1 = time.time()
    file_size = os.path.getsize(output_path)
    print(f"[export_dit] Exported to {output_path}")
    print(f"[export_dit] File size: {file_size/1e9:.2f} GB")
    print(f"[export_dit] Export time: {t1-t0:.1f}s")
    
    return output_path


def verify_onnx(onnx_path: str, dit_model, config):
    """Verify the ONNX model produces matching output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[export_dit] onnxruntime not installed, skipping verification")
        return
    
    device = next(dit_model.parameters()).device
    dtype = next(dit_model.parameters()).dtype
    
    wrapper = DiTForwardWrapper(dit_model)
    wrapper.eval()
    
    # Create test inputs
    B, T, S = 1, 256, 128
    input_latents = torch.randn(B, T, 192, device=device, dtype=dtype)
    enc_hidden = torch.randn(B, S, 2048, device=device, dtype=dtype)
    t = torch.tensor([0.3], device=device, dtype=torch.float32)
    t_r = torch.tensor([0.3], device=device, dtype=torch.float32)
    
    # PyTorch reference
    with torch.no_grad():
        ref_out = wrapper(input_latents, enc_hidden, t, t_r)
    
    # ONNX inference
    sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    ort_out = sess.run(None, {
        "input_latents": input_latents.cpu().float().numpy(),
        "enc_hidden": enc_hidden.cpu().float().numpy(),
        "t": t.cpu().numpy(),
        "t_r": t_r.cpu().numpy(),
    })
    
    # Compare
    import numpy as np
    ref_np = ref_out.cpu().float().numpy()
    ort_np = ort_out[0]
    
    max_diff = np.max(np.abs(ref_np - ort_np))
    mean_diff = np.mean(np.abs(ref_np - ort_np))
    print(f"[export_dit] Verification: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    if max_diff < 0.01:
        print("[export_dit] ✅ ONNX output matches PyTorch (within FP16 tolerance)")
    else:
        print("[export_dit] ⚠️ Large difference detected — may need investigation")


def main():
    parser = argparse.ArgumentParser(description="Export AceStep DiT to ONNX")
    parser.add_argument("--model-dir", required=True,
                        help="Path to the model directory (containing model.safetensors + config.json)")
    parser.add_argument("--output", default=None,
                        help="Output ONNX file path (default: models/onnx/dit_<model_name>.onnx)")
    parser.add_argument("--opset", type=int, default=18,
                        help="ONNX opset version (default: 18)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX output matches PyTorch")
    parser.add_argument("--device", default="cuda",
                        help="Device for model loading (default: cuda)")
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        model_name = Path(args.model_dir).name
        onnx_dir = Path(args.model_dir).parent.parent / "models" / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(onnx_dir / f"dit_{model_name}.onnx")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load model
    dit_model, config = load_dit_model(args.model_dir, device=args.device)
    
    # Export
    export_onnx(dit_model, config, args.output, opset=args.opset)
    
    # Verify
    if args.verify:
        verify_onnx(args.output, dit_model, config)
    
    print("[export_dit] Done!")


if __name__ == "__main__":
    main()
