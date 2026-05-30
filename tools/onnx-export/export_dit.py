#!/usr/bin/env python3
"""
export_dit.py — Export AceStep DiT forward pass to ONNX for TensorRT acceleration.

Exports the SINGLE FORWARD PASS (one diffusion timestep) of the DiT model,
wrapping the full 32-layer transformer + attention mask computation + RoPE
into a single ONNX graph with 4 simplified inputs.

Precision recipes (--precision):
  fp32       — Full FP32. Correct but slow. Baseline for validation.
  bf16_mixed — (default for XL) bf16 bulk + fp32 ConvTranspose1d island.
               Used with TRT STRONGLY_TYPED mode. Demon-proven recipe.
               bf16 has same exponent range as fp32 — no activation overflow.

Usage:
    python export_dit.py --model-dir <path-to-safetensors-model> --output <output.onnx>
    python export_dit.py --model-dir <path> --output <path> --precision bf16_mixed

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


class _Fp32CastWrapper(nn.Module):
    """Run an inner module in fp32, casting around it.

    Used when TRT has no kernel for a specific op shape in bf16.
    The wrapper casts input to fp32, runs the inner module, then casts
    output back to the caller's dtype.
    """
    def __init__(self, inner: nn.Module):
        super().__init__()
        inner.float()  # force inner weights to fp32
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        # Disable autocast — without this, the outer autocast(bf16) overrides
        # our explicit fp32 computation and TRT sees bf16 weights.
        with torch.amp.autocast('cuda', enabled=False):
            return self.inner(x.float()).to(out_dtype)


class PatchEmbedLinear(nn.Module):
    """Replace Conv1d(C_in, C_out, K, stride=K) with reshape + Linear.
    
    TRT 10.16 has NO kernels for 1D convolutions with patch_size shapes
    in any precision mode (fp16, bf16, or fp32). This is mathematically equivalent:
      Conv1d: input[B, C_in, T] → output[B, C_out, T//K]
      Linear: input[B, C_in, T] → unfold[B, T//K, C_in*K] → Linear → [B, C_out, T//K]
    """
    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        C_out, C_in, K = conv.weight.shape
        self.kernel_size = K
        self.linear = nn.Linear(C_in * K, C_out, bias=conv.bias is not None)
        # Conv weight [C_out, C_in, K] → Linear weight [C_out, C_in*K]
        self.linear.weight.data = conv.weight.data.reshape(C_out, -1).clone()
        if conv.bias is not None:
            self.linear.bias.data = conv.bias.data.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T] (from Lambda transpose in proj_in)
        B, C, T = x.shape
        K = self.kernel_size
        # Unfold patches: [B, C, T] → [B, T//K, C*K]
        x = x.reshape(B, C, T // K, K)       # [B, C, T//K, K]
        x = x.permute(0, 2, 1, 3)             # [B, T//K, C, K]
        x = x.reshape(B, T // K, C * K)       # [B, T//K, C*K]
        out = self.linear(x)                   # [B, T//K, C_out]
        return out.transpose(1, 2)             # [B, C_out, T//K]


class UnPatchLinear(nn.Module):
    """Replace ConvTranspose1d(C_in, C_out, K, stride=K) with Linear + reshape.
    
    TRT 10.16 has NO kernels for 1D transposed convolutions with patch_size shapes.
    This is mathematically equivalent:
      ConvTranspose1d: input[B, C_in, T//K] → output[B, C_out, T]
      Linear: input[B, T//K, C_in] → Linear → [B, T//K, C_out*K] → fold → [B, C_out, T]
    """
    def __init__(self, deconv: nn.ConvTranspose1d):
        super().__init__()
        C_in, C_out, K = deconv.weight.shape
        self.kernel_size = K
        self.C_out = C_out
        self.linear = nn.Linear(C_in, C_out * K, bias=deconv.bias is not None)
        # ConvTranspose1d weight [C_in, C_out, K] → Linear weight [C_out*K, C_in]
        self.linear.weight.data = deconv.weight.data.permute(1, 2, 0).reshape(C_out * K, C_in).clone()
        if deconv.bias is not None:
            # ConvTranspose1d bias [C_out] → Linear bias [C_out*K] (repeat per patch)
            self.linear.bias.data = deconv.bias.data.repeat_interleave(K).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T//K] (from Lambda transpose in proj_out)
        B, C, T_small = x.shape
        K = self.kernel_size
        x = x.transpose(1, 2)                              # [B, T//K, C_in]
        x = self.linear(x)                                  # [B, T//K, C_out*K]
        x = x.reshape(B, T_small, self.C_out, K)            # [B, T//K, C_out, K]
        x = x.permute(0, 2, 1, 3)                           # [B, C_out, T//K, K]
        x = x.reshape(B, self.C_out, T_small * K)           # [B, C_out, T]
        return x


class DiTForwardWrapper(nn.Module):
    """
    Wrapper around AceStepDiTModel.forward() that simplifies the interface
    for ONNX export.
    
    ONNX inputs (4 total):
        input_latents:  [B, T, 192]  — pre-concatenated [context_latents, xt]
        enc_hidden:     [B, S, 2048] — encoder hidden states
        t:              [B]          fp32 — current timestep
        t_r:            [B]          fp32 — reference timestep
    
    ONNX output:
        velocity:       [B, T, 64]   — predicted flow velocity
    
    Masks and position IDs are computed internally from T and S.
    """
    
    def __init__(self, dit_model, precision="bf16_mixed"):
        super().__init__()
        self.dit = dit_model
        self.config = dit_model.config
        self.precision = precision
    
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
        
        # bf16 autocast: the dynamo exporter decomposes complex ops
        # (view_as_complex → rotate_half) into real-number equivalents,
        # so no Cast(to=COMPLEX128) appears in the ONNX graph.
        if self.precision == "bf16_mixed":
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float32
        with torch.amp.autocast('cuda', dtype=autocast_dtype):
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


def apply_bf16_mixed(dit_model):
    """Apply the bf16_mixed precision recipe (XL models).
    
    bf16 bulk + fp32 island for proj_out ConvTranspose1d.
    
    bf16 has the SAME exponent range as fp32 (8 bits vs fp16's 5 bits),
    so intermediate activations never overflow. This is the key difference
    from fp16_mixed which NaN'd because the XL residual stream accumulated
    values exceeding fp16's ±65504 range over 32 layers.
    
    The entire model runs in bf16 EXCEPT:
      - proj_out ConvTranspose1d → wrapped in _Fp32CastWrapper because
        TRT 10.16 has no bf16 deconv kernel for this shape.
    
    Uses STRONGLY_TYPED mode so TRT honors the bf16/fp32 split from the
    ONNX graph. TRT's bf16 tensor cores provide the same throughput as fp16.
    """
    dit_model.to(torch.bfloat16)
    print("[export_dit] Applied bf16 bulk conversion")
    
    # FP32 island: proj_out ConvTranspose1d (TRT has no bf16 deconv kernel)
    # NOTE: This gets replaced by UnPatchLinear AFTER this function runs
    # (replace_conv_with_linear handles it). But we still wrap it in
    # _Fp32CastWrapper in case the Conv→Linear replacement changes.
    if hasattr(dit_model, 'proj_out') and isinstance(dit_model.proj_out, nn.Sequential):
        for i, mod in enumerate(dit_model.proj_out):
            if isinstance(mod, nn.ConvTranspose1d):
                dit_model.proj_out[i] = _Fp32CastWrapper(mod)
                print(f"[export_dit] FP32 island: proj_out[{i}] ConvTranspose1d → _Fp32CastWrapper")
                break
    
    return dit_model


def replace_conv_with_linear(dit_model):
    """Replace Conv1d/ConvTranspose1d with equivalent Linear ops.
    
    TRT 10.16 has NO kernels for 1D convolutions with patch_size=2 in ANY
    precision mode (fp16, bf16, fp32, or mixed). PatchEmbedLinear/UnPatchLinear
    reformulate these as reshape+matmul which TRT handles perfectly.
    
    Must be called for ALL precision recipes, not just mixed precision.
    
    Handles _Fp32CastWrapper: if a ConvTranspose1d is already wrapped in
    _Fp32CastWrapper (from bf16_mixed recipe), we unwrap it, convert to
    UnPatchLinear, and re-wrap in _Fp32CastWrapper.
    """
    if hasattr(dit_model, 'proj_in') and isinstance(dit_model.proj_in, nn.Sequential):
        for i, mod in enumerate(dit_model.proj_in):
            if isinstance(mod, nn.Conv1d):
                dit_model.proj_in[i] = PatchEmbedLinear(mod)
                print(f"[export_dit] Conv→Linear: proj_in[{i}] Conv1d → PatchEmbedLinear")
    
    if hasattr(dit_model, 'proj_out') and isinstance(dit_model.proj_out, nn.Sequential):
        for i, mod in enumerate(dit_model.proj_out):
            if isinstance(mod, nn.ConvTranspose1d):
                dit_model.proj_out[i] = UnPatchLinear(mod)
                print(f"[export_dit] Conv→Linear: proj_out[{i}] ConvTranspose1d → UnPatchLinear")
            elif isinstance(mod, _Fp32CastWrapper) and isinstance(mod.inner, nn.ConvTranspose1d):
                # Unwrap, convert, re-wrap
                linear_mod = UnPatchLinear(mod.inner)
                dit_model.proj_out[i] = _Fp32CastWrapper(linear_mod)
                print(f"[export_dit] Conv→Linear: proj_out[{i}] Fp32Cast(ConvTranspose1d) → Fp32Cast(UnPatchLinear)")
    
    return dit_model


def load_dit_model(model_dir: str, device: str = "cuda", precision: str = "bf16_mixed"):
    """Load the AceStepDiTModel from a safetensors checkpoint."""
    model_dir = Path(model_dir)
    
    # Fix Windows encoding issues with transformers emoji output
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    # Monkey-patch transformers auto_docstring to avoid lookup failure
    # for custom model types not registered in HF model registry
    try:
        import transformers.utils.auto_docstring as _ad
        _orig = _ad.auto_docstring
        _ad.auto_docstring = lambda *a, **kw: (lambda cls: cls)  # no-op decorator
    except Exception:
        pass
    
    # Add model dir to sys.path so we can import the model code.
    # Also add the Demon app root — model config files are re-export stubs
    # that import from the acestep package (from Demon).
    sys.path.insert(0, str(model_dir))
    demon_root = Path(model_dir).parent.parent.parent / "Demon"
    if demon_root.exists():
        sys.path.insert(0, str(demon_root))
        print(f"[export_dit] Added {demon_root} to sys.path for acestep package")
        # The model config stubs reference acestep.models.common but the
        # actual module is acestep.models. Create a shim alias.
        try:
            import acestep.models as _am
            sys.modules["acestep.models.common"] = _am
            # Also create the subpackage entry so Python's import system is happy
            import types
            if not hasattr(_am, "common"):
                _am.common = _am
        except ImportError:
            print("[export_dit] WARNING: Could not import acestep.models")
    
    # Auto-detect the modeling module — different model variants use different
    # filenames (modeling_acestep_v15_xl_base.py, xl_turbo.py, etc.)
    import glob
    modeling_files = glob.glob(str(model_dir / "modeling_acestep_v15*.py"))
    if not modeling_files:
        print(f"[export_dit] ERROR: No modeling_acestep_v15*.py found in {model_dir}")
        sys.exit(1)
    modeling_module = Path(modeling_files[0]).stem
    print(f"[export_dit] Using modeling module: {modeling_module}")
    
    import importlib
    mod = importlib.import_module(modeling_module)
    AceStepDiTModel = mod.AceStepDiTModel
    from configuration_acestep_v15 import AceStepConfig
    
    # Load config
    import json
    with open(model_dir / "config.json") as f:
        config_dict = json.load(f)
    
    config = AceStepConfig(**config_dict)
    # Force SDPA for ONNX export (no flash attention)
    config._attn_implementation = "sdpa"
    
    print(f"[export_dit] Loading model from {model_dir}...")
    print(f"[export_dit] Precision recipe: {precision}")
    t0 = time.time()
    
    # Create just the DiT model (decoder) — no need for full model
    dit_model = AceStepDiTModel(config)
    
    # Load weights — handle both single-file and sharded safetensors
    from safetensors.torch import load_file
    
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    
    if index_path.exists():
        # Sharded: load index to find all shard files
        import json as _json
        with open(index_path) as f:
            index = _json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        print(f"[export_dit] Loading {len(shard_files)} shards...")
        state_dict = {}
        for shard in shard_files:
            shard_path = model_dir / shard
            print(f"[export_dit]   Loading {shard}...")
            state_dict.update(load_file(str(shard_path)))
    elif single_path.exists():
        state_dict = load_file(str(single_path))
    else:
        print(f"[export_dit] ERROR: No model.safetensors found in {model_dir}")
        sys.exit(1)
    
    # Filter and remap: "decoder.X" -> "X" for the DiT model
    dit_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("decoder."):
            dit_state_dict[k[len("decoder."):]] = v
    
    missing, unexpected = dit_model.load_state_dict(dit_state_dict, strict=False)
    if missing:
        print(f"[export_dit] Warning: {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"[export_dit] Warning: {len(unexpected)} unexpected keys")
    
    # Apply precision recipe AFTER loading weights (so weights are converted correctly)
    if precision == "bf16_mixed":
        dit_model = dit_model.to(device=device)  # move to GPU first
        dit_model = apply_bf16_mixed(dit_model)
    elif precision == "fp32":
        dit_model = dit_model.to(device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown precision: {precision}. Use 'bf16_mixed' or 'fp32'.")
    
    # Replace Conv1d/ConvTranspose1d with Linear equivalents for ALL precision modes.
    # TRT 10.16 has no kernels for 1D convolutions with patch_size=2.
    dit_model = replace_conv_with_linear(dit_model)
    
    dit_model.eval()
    
    t1 = time.time()
    print(f"[export_dit] Model loaded in {t1-t0:.1f}s")
    print(f"[export_dit] DiT: {sum(p.numel() for p in dit_model.parameters())/1e9:.2f}B params")
    
    # Log dtype distribution
    dtypes = {}
    for p in dit_model.parameters():
        dt = str(p.dtype)
        dtypes[dt] = dtypes.get(dt, 0) + p.numel()
    for dt, count in sorted(dtypes.items()):
        print(f"[export_dit]   {dt}: {count/1e6:.1f}M params")
    
    return dit_model, config


def export_onnx(dit_model, config, output_path: str, opset: int = 18, precision: str = "bf16_mixed"):
    """Export the DiT forward pass to ONNX."""
    device = next(dit_model.parameters()).device
    
    # Dummy inputs match precision recipe
    if precision == "bf16_mixed":
        tensor_dtype = torch.bfloat16
    else:
        tensor_dtype = torch.float32
    
    wrapper = DiTForwardWrapper(dit_model, precision=precision)
    wrapper.eval()
    
    # Create dummy inputs for tracing
    B = 1
    T = 512   # typical sequence length (divisible by patch_size=2)
    S = 256   # typical encoder sequence length
    
    dummy_input_latents = torch.randn(B, T, 192, device=device, dtype=tensor_dtype)
    dummy_enc_hidden = torch.randn(B, S, 2048, device=device, dtype=tensor_dtype)
    dummy_t = torch.tensor([0.5], device=device, dtype=torch.float32)  # always fp32
    dummy_t_r = torch.tensor([0.5], device=device, dtype=torch.float32)  # always fp32
    
    print(f"[export_dit] Tracing with shapes: input_latents={list(dummy_input_latents.shape)}, "
          f"enc_hidden={list(dummy_enc_hidden.shape)}, t={list(dummy_t.shape)}")
    print(f"[export_dit] Input dtype: {tensor_dtype}, t/t_r dtype: fp32")
    
    # Test forward pass first
    print("[export_dit] Testing forward pass...")
    with torch.no_grad():
        test_out = wrapper(dummy_input_latents, dummy_enc_hidden, dummy_t, dummy_t_r)
    print(f"[export_dit] Output shape: {list(test_out.shape)} (expected [{B}, {T}, 64])")
    print(f"[export_dit] Output dtype: {test_out.dtype}")
    
    # Check for NaN
    if torch.isnan(test_out).any():
        print("[export_dit] ERROR: Output contains NaN! Aborting export.")
        sys.exit(1)
    
    # Export to ONNX
    print(f"[export_dit] Exporting to ONNX (opset {opset})...")
    t0 = time.time()
    
    # Dynamo requires dynamic_shapes (not dynamic_axes)
    # Each input gets a dict mapping dim index → Dim object
    batch = torch.export.Dim("batch", min=1, max=4)
    seq_len = torch.export.Dim("seq_len", min=64, max=8192)
    enc_seq_len = torch.export.Dim("enc_seq_len", min=64, max=2048)
    
    dynamic_shapes = {
        "input_latents": {0: batch, 1: seq_len},
        "enc_hidden":    {0: batch, 1: enc_seq_len},
        "t":             {0: batch},
        "t_r":           {0: batch},
    }
    
    onnx_program = torch.onnx.export(
        wrapper,
        (dummy_input_latents, dummy_enc_hidden, dummy_t, dummy_t_r),
        output_path,
        opset_version=opset,
        input_names=["input_latents", "enc_hidden", "t", "t_r"],
        output_names=["velocity"],
        dynamic_shapes=dynamic_shapes,
        export_params=True,
        external_data=True,
        dynamo=True,
    )
    
    # ── Post-process: rename val_N initializers to original parameter FQNs ──
    # Ported from Demon's rename_val_initializers_to_fqn (export.py:636-879).
    #
    # The dynamo exporter replaces parameter names with opaque val_0, val_1, ...
    # TRT refit addresses weights by ONNX name, so we must restore FQNs.
    #
    # Strategy: SHA-256 byte hash of full tensor data, tried in both
    # orientations (torch [out,in] and ONNX MatMul [in,out]). Dynamo
    # transposes Linear weights for MatMul but preserves the raw bytes,
    # so exact-hash matching is reliable.
    #
    # Proto-only save: we never re-encode the external data file (onnx's
    # writer has been observed to silently convert bf16→fp16 on re-save).
    
    print("[export_dit] Renaming val_N initializers to parameter FQNs...")
    import hashlib, json
    import onnx
    from onnx import TensorProto
    import numpy as np
    
    model_proto = onnx.load(output_path, load_external_data=False)
    base_dir = os.path.dirname(output_path)
    
    def _sha(b: bytes) -> bytes:
        return hashlib.sha256(b).digest()
    
    def _bytes_for(p: torch.Tensor):
        """Raw bytes of a torch tensor in its native dtype."""
        p_cpu = p.detach().cpu().contiguous()
        if p_cpu.dtype == torch.bfloat16:
            return p_cpu.view(torch.uint16).numpy().tobytes()
        if p_cpu.dtype in (torch.float16, torch.float32):
            return p_cpu.numpy().tobytes()
        return None
    
    _TORCH_TO_ONNX_DT = {
        torch.float32: TensorProto.FLOAT,
        torch.float16: TensorProto.FLOAT16,
        torch.bfloat16: TensorProto.BFLOAT16,
    }
    
    # Build torch-side hash index: (onnx_dtype, shape, sha256) → (fqn, transposed)
    # Hash each 2D param in both orientations.
    torch_hash_index = {}
    for name, p in wrapper.named_parameters():
        if p.dim() != 2:
            continue
        canon = "dit." + name if not name.startswith("dit.") else name
        onnx_dt = _TORCH_TO_ONNX_DT.get(p.dtype)
        if onnx_dt is None:
            continue
        
        # Original orientation [out, in]
        b_orig = _bytes_for(p)
        if b_orig is None:
            continue
        shape_orig = tuple(p.shape)
        torch_hash_index.setdefault(
            (onnx_dt, shape_orig, _sha(b_orig)), (canon, False)
        )
        
        # Transposed orientation [in, out] — how ONNX MatMul stores it
        p_t = p.transpose(0, 1)
        b_trans = _bytes_for(p_t)
        if b_trans is not None:
            shape_trans = (shape_orig[1], shape_orig[0])
            torch_hash_index.setdefault(
                (onnx_dt, shape_trans, _sha(b_trans)), (canon, True)
            )
    
    print(f"[export_dit] Built hash index: {len(torch_hash_index)} entries "
          f"from {sum(1 for _,p in wrapper.named_parameters() if p.dim()==2)} 2D params")
    
    def _read_external_bytes(init):
        """Read raw bytes for one initializer from its external data file."""
        loc = None
        offset = 0
        length = None
        for ed in init.external_data:
            if ed.key == "location":
                loc = ed.value
            elif ed.key == "offset":
                offset = int(ed.value)
            elif ed.key == "length":
                length = int(ed.value)
        if loc is None:
            return None
        ext_path = os.path.join(base_dir, loc)
        with open(ext_path, "rb") as f:
            f.seek(offset)
            return f.read(length) if length is not None else f.read()
    
    # Match val_N initializers to torch parameters by SHA-256
    used_names = {init.name for init in model_proto.graph.initializer}
    val_inits_changed = {}  # old_name → new_name
    transposed_fqns = []
    claimed_torch = set()
    float_dtypes = (TensorProto.BFLOAT16, TensorProto.FLOAT16, TensorProto.FLOAT)
    renamed = 0
    skipped = 0
    
    for init in model_proto.graph.initializer:
        if not init.name.startswith("val_"):
            continue
        dims = tuple(init.dims)
        if len(dims) != 2:
            continue
        nelem = int(np.prod(dims))
        if nelem < 16:
            continue
        if init.data_type not in float_dtypes:
            continue
        
        raw = _read_external_bytes(init)
        if raw is None:
            raw = bytes(init.raw_data) if init.raw_data else None
        if raw is None:
            continue
        
        expected_bytes = nelem * (4 if init.data_type == TensorProto.FLOAT else 2)
        if len(raw) != expected_bytes:
            skipped += 1
            continue
        
        key = (init.data_type, dims, _sha(raw))
        result = torch_hash_index.get(key)
        if result is None:
            skipped += 1
            continue
        canon, is_transposed = result
        if canon in claimed_torch or canon in used_names:
            skipped += 1
            continue
        
        val_inits_changed[init.name] = canon
        claimed_torch.add(canon)
        used_names.add(canon)
        if is_transposed:
            transposed_fqns.append(canon)
        renamed += 1
    
    # Apply renames to proto (initializers + node inputs + graph inputs/value_info)
    if val_inits_changed:
        for init in model_proto.graph.initializer:
            if init.name in val_inits_changed:
                init.name = val_inits_changed[init.name]
        for node in model_proto.graph.node:
            for i, ref in enumerate(node.input):
                if ref in val_inits_changed:
                    node.input[i] = val_inits_changed[ref]
        for vi in list(model_proto.graph.input) + list(model_proto.graph.value_info):
            if vi.name in val_inits_changed:
                vi.name = val_inits_changed[vi.name]
        
        # Proto-only save — external data files keep original bytes
        onnx.save(model_proto, output_path)
        
        print(f"[export_dit] Renamed {renamed} val_N initializers to FQNs "
              f"({len(transposed_fqns)} transposed, {skipped} skipped)")
    else:
        print("[export_dit] WARNING: No val_N initializers matched any parameter")
    
    # Emit refit manifest sidecar
    manifest = {
        "version": 1,
        "onnx_path": os.path.basename(output_path),
        "weights_transposed": sorted(transposed_fqns),
        "weights_renamed": renamed,
    }
    manifest_path = output_path + ".refit_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[export_dit] Refit manifest saved to {manifest_path}")
    
    t1 = time.time()
    print(f"[export_dit] ONNX trace completed in {t1-t0:.1f}s")
    
    # Verify files exist
    data_path = output_path + ".data"
    onnx_size = os.path.getsize(output_path)
    data_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
    
    if data_size == 0:
        # Dynamo didn't write external data — re-save manually
        print("[export_dit] External data missing, re-saving with onnx library...")
        import onnx
        from onnx.external_data_helper import convert_model_to_external_data
        
        model_proto = onnx.load(output_path, load_external_data=False)
        data_filename = os.path.basename(output_path) + ".data"
        convert_model_to_external_data(
            model_proto,
            all_tensors_to_one_file=True,
            location=data_filename,
            size_threshold=1024,
            convert_attribute=False,
        )
        onnx.save(model_proto, output_path)
        onnx_size = os.path.getsize(output_path)
        data_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
    
    print(f"[export_dit] Exported to {output_path}")
    print(f"[export_dit] ONNX graph: {onnx_size/1e6:.1f} MB")
    print(f"[export_dit] Weight data: {data_size/1e9:.2f} GB")
    print(f"[export_dit] Total export time: {time.time()-t0:.1f}s")
    
    return output_path


def verify_onnx(onnx_path: str, dit_model, config, precision: str = "bf16_mixed"):
    """Verify the ONNX model produces matching output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[export_dit] onnxruntime not installed, skipping verification")
        return
    
    device = next(dit_model.parameters()).device
    
    if precision == "bf16_mixed":
        tensor_dtype = torch.bfloat16
    else:
        tensor_dtype = torch.float32
    
    wrapper = DiTForwardWrapper(dit_model, precision=precision)
    wrapper.eval()
    
    # Create test inputs
    B, T, S = 1, 256, 128
    input_latents = torch.randn(B, T, 192, device=device, dtype=tensor_dtype)
    enc_hidden = torch.randn(B, S, 2048, device=device, dtype=tensor_dtype)
    t = torch.tensor([0.3], device=device, dtype=torch.float32)
    t_r = torch.tensor([0.3], device=device, dtype=torch.float32)
    
    # PyTorch reference
    with torch.no_grad():
        ref_out = wrapper(input_latents, enc_hidden, t, t_r)
    
    # ONNX inference — feed fp32 (ORT doesn't support bf16 on most providers)
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
    
    if max_diff < 0.05:  # bf16 has slightly larger tolerance than fp16
        print("[export_dit] PASS: ONNX output matches PyTorch (within bf16 tolerance)")
    else:
        print("[export_dit] WARNING: Large difference detected — may need investigation")


def main():
    parser = argparse.ArgumentParser(description="Export AceStep DiT to ONNX")
    parser.add_argument("--model-dir", required=True,
                        help="Path to the model directory (containing model.safetensors + config.json)")
    parser.add_argument("--output", default=None,
                        help="Output ONNX file path (default: models/onnx/dit_<model_name>.onnx)")
    parser.add_argument("--opset", type=int, default=18,
                        help="ONNX opset version (default: 18)")
    parser.add_argument("--precision", default="bf16_mixed",
                        choices=["bf16_mixed", "fp32"],
                        help="Precision recipe (default: bf16_mixed)")
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
    dit_model, config = load_dit_model(args.model_dir, device=args.device, precision=args.precision)
    
    # Export
    export_onnx(dit_model, config, args.output, opset=args.opset, precision=args.precision)
    
    # Verify
    if args.verify:
        verify_onnx(args.output, dit_model, config, precision=args.precision)
    
    print("[export_dit] Done!")


if __name__ == "__main__":
    main()
