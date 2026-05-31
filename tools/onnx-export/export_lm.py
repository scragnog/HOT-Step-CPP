#!/usr/bin/env python3
"""
Export Qwen3ForCausalLM to ONNX for TensorRT inference.

Produces two ONNX models:
  1. lm_full.onnx  — Full-vocab (Phase 1): logits over entire 217K vocabulary
  2. lm_audio.onnx — Partial-vocab (Phase 2): logits over audio codes only (~65K tokens)

Both models take explicit KV cache tensors as input/output for
autoregressive generation with TensorRT.

Usage:
    python export_lm.py --model-dir models/acestep-5Hz-lm-4B \\
                        --output models/onnx/lm-4B/ \\
                        --device cuda

Requirements:
    pip install torch transformers onnx
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix Windows encoding for torch.onnx unicode diagnostics
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

# ── Constants matching C++ engine (prompt.h) ─────────────────────────────────
TOKEN_IM_END = 151645        # <|im_end|> — EOS token
AUDIO_CODE_BASE = 151669     # First audio code token
AUDIO_CODE_COUNT = 65535     # Number of audio code tokens
LM_PARTIAL_OFFSET = TOKEN_IM_END  # Phase 2 partial head starts here


# ══════════════════════════════════════════════════════════════════════════════
#  Export Wrappers
# ══════════════════════════════════════════════════════════════════════════════

class Qwen3LMFullWrapper(nn.Module):
    """
    Full-vocab wrapper for ONNX export.
    Calls the Qwen3Model transformer directly, applies tied lm_head.
    KV cache flows as explicit flat tensors via DynamicCache conversion.
    """

    def __init__(self, model):
        super().__init__()
        self.transformer = model.model  # Qwen3Model
        self.lm_head_weight = model.model.embed_tokens.weight  # Tied
        self.n_layers = model.config.num_hidden_layers
        self.out_vocab = model.config.vocab_size
        print(f"[Export] Full LM head: {self.out_vocab} tokens")

    def forward(self, input_ids, position_ids, attention_mask, *past_kvs):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for i in range(self.n_layers):
            cache.update(past_kvs[2 * i], past_kvs[2 * i + 1], i)

        outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )

        logits = F.linear(outputs.last_hidden_state, self.lm_head_weight).float()

        pkv = outputs.past_key_values
        result = [logits]
        for layer in pkv.layers:
            result.append(layer.keys)
            result.append(layer.values)
        return tuple(result)


class Qwen3LMPartialWrapper(nn.Module):
    """
    Partial-vocab wrapper for ONNX export (Phase 2 — audio codes only).
    Same transformer, but lm_head projects to tokens [offset..vocab_size).
    """

    def __init__(self, model, partial_vocab_offset: int):
        super().__init__()
        self.transformer = model.model
        self.n_layers = model.config.num_hidden_layers

        n_partial = model.config.vocab_size - partial_vocab_offset
        embed_weight = model.model.embed_tokens.weight.data
        self.partial_lm_head = nn.Parameter(
            embed_weight[partial_vocab_offset:].clone().contiguous(),
            requires_grad=False
        )
        self.out_vocab = n_partial
        print(f"[Export] Partial LM head: {n_partial} tokens "
              f"(offset={partial_vocab_offset})")

    def forward(self, input_ids, position_ids, attention_mask, *past_kvs):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for i in range(self.n_layers):
            cache.update(past_kvs[2 * i], past_kvs[2 * i + 1], i)

        outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )

        logits = F.linear(outputs.last_hidden_state, self.partial_lm_head).float()

        # Output only the NEW KV tokens (same as full wrapper)
        seq_len = input_ids.shape[1]
        pkv = outputs.past_key_values
        result = [logits]
        for layer in pkv.layers:
            result.append(layer.keys[:, :, -seq_len:, :].contiguous())
            result.append(layer.values[:, :, -seq_len:, :].contiguous())
        return tuple(result)


# ══════════════════════════════════════════════════════════════════════════════
#  ONNX Export Helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_dummy_inputs(config, batch=1, seq_len=5, past_seq_len=3,
                       device="cpu", dtype=torch.bfloat16):
    """Create dummy inputs for ONNX tracing."""
    n_kv = config.num_key_value_heads
    d = config.head_dim
    n_layers = config.num_hidden_layers

    inputs = (
        torch.randint(0, config.vocab_size, (batch, seq_len),
                      dtype=torch.long, device=device),
        torch.arange(past_seq_len, past_seq_len + seq_len,
                      dtype=torch.long, device=device).unsqueeze(0).expand(batch, -1),
        torch.ones(batch, past_seq_len + seq_len,
                   dtype=torch.long, device=device),
    )
    for _ in range(n_layers):
        inputs += (
            torch.randn(batch, n_kv, past_seq_len, d, dtype=dtype, device=device),
            torch.randn(batch, n_kv, past_seq_len, d, dtype=dtype, device=device),
        )
    return inputs


def build_io_names(n_layers):
    """Build input/output name lists."""
    input_names = ["input_ids", "position_ids", "attention_mask"]
    output_names = ["logits"]
    for i in range(n_layers):
        input_names += [f"past_key_{i}", f"past_value_{i}"]
        output_names += [f"present_key_{i}", f"present_value_{i}"]
    return input_names, output_names


def build_dynamic_shapes(n_layers):
    """Build dynamic_shapes for dynamo export."""
    batch = torch.export.Dim("batch", min=1, max=4)
    seq_len = torch.export.Dim("seq_len", min=1, max=1024)
    past_seq_len = torch.export.Dim("past_seq_len", min=1, max=8192)
    total_len = torch.export.Dim("total_len", min=2, max=9216)

    return {
        "input_ids": {0: batch, 1: seq_len},
        "position_ids": {0: batch, 1: seq_len},
        "attention_mask": {0: batch, 1: total_len},
        # *args must be a TUPLE (not list) to match the pytree structure
        "past_kvs": tuple(
            {0: batch, 2: past_seq_len} for _ in range(n_layers * 2)
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SHA-256 Weight Renaming (for dynamo-exported ONNX)
# ══════════════════════════════════════════════════════════════════════════════
#
# torch.onnx.export with dynamo=True renames all parameters to val_N.
# We rename them back to their original FQNs using SHA-256 digest matching.
# This is critical for adapter refit — the C++ runtime needs to map
# safetensors weight names to ONNX initializer names.
#

def _sha(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _bytes_for(p: torch.Tensor) -> bytes:
    """Get raw bytes from a parameter, handling bf16 via uint16 view."""
    if p.dtype == torch.bfloat16:
        return p.detach().cpu().view(torch.uint16).numpy().tobytes()
    return p.detach().cpu().numpy().tobytes()


_TORCH_TO_ONNX_DT = {
    torch.float32: 1,   # FLOAT
    torch.float16: 10,  # FLOAT16
    torch.bfloat16: 16, # BFLOAT16
    torch.int64: 7,     # INT64
    torch.int32: 6,     # INT32
}


def _read_external_bytes(init, onnx_dir):
    """Read raw bytes for an ONNX initializer from external data file."""
    for ext in init.external_data:
        if ext.key == "location":
            fpath = os.path.join(onnx_dir, ext.value)
        elif ext.key == "offset":
            offset = int(ext.value)
        elif ext.key == "length":
            length = int(ext.value)
    with open(fpath, "rb") as f:
        f.seek(offset)
        return f.read(length)


def rename_weights(onnx_path, torch_model):
    """
    Rename val_N ONNX initializers back to their PyTorch FQN using
    SHA-256 digest matching. Returns (renamed_count, transposed_fqns).
    """
    import onnx

    print(f"[Rename] Renaming weights in {os.path.basename(onnx_path)}...")
    model = onnx.load(onnx_path, load_external_data=False)
    onnx_dir = os.path.dirname(onnx_path)

    # Build torch-side hash index: (dtype, shape, sha256) → (fqn, is_transposed)
    hash_idx = {}
    for fqn, param in torch_model.named_parameters():
        if param.dim() < 2 or param.numel() < 16:
            continue
        dt = _TORCH_TO_ONNX_DT.get(param.dtype)
        if dt is None:
            continue

        raw = _bytes_for(param)
        shape = tuple(param.shape)
        key = (dt, shape, _sha(raw))
        hash_idx[key] = (fqn, False)

        # Also try transposed
        pt = param.t().contiguous()
        raw_t = _bytes_for(pt)
        shape_t = tuple(pt.shape)
        key_t = (dt, shape_t, _sha(raw_t))
        hash_idx[key_t] = (fqn, True)

    # Match ONNX initializers
    renamed = {}
    transposed_fqns = set()
    for init in model.graph.initializer:
        if not init.name.startswith("val_"):
            continue
        if init.dims is None or len(init.dims) < 2:
            continue
        if init.data_type not in (1, 10, 16):  # FLOAT, FLOAT16, BFLOAT16
            continue
        if sum(init.dims) < 16:
            continue

        raw = _read_external_bytes(init, onnx_dir)
        shape = tuple(init.dims)
        key = (init.data_type, shape, _sha(raw))

        if key in hash_idx:
            fqn, is_t = hash_idx[key]
            old = init.name
            renamed[old] = fqn
            if is_t:
                transposed_fqns.add(fqn)

    # Apply renames
    for old, new in renamed.items():
        # Rename initializer
        for init in model.graph.initializer:
            if init.name == old:
                init.name = new
                break
        # Rename all node inputs referencing old name
        for node in model.graph.node:
            for j, inp in enumerate(node.input):
                if inp == old:
                    node.input[j] = new
        # Rename graph inputs
        for gi in model.graph.input:
            if gi.name == old:
                gi.name = new

    # Save (proto-only, don't re-encode external data)
    onnx.save_model(
        model, onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(onnx_path) + ".data",
    )

    print(f"[Rename] Renamed {len(renamed)} weights, "
          f"{len(transposed_fqns)} transposed")
    return renamed, sorted(transposed_fqns)


# ══════════════════════════════════════════════════════════════════════════════
#  Export Core
# ══════════════════════════════════════════════════════════════════════════════

def export_onnx(wrapper, config, output_path, device, opset=18,
                do_rename=True, torch_model=None):
    """Export a wrapper to ONNX with explicit KV cache I/O using dynamo."""
    n_layers = config.num_hidden_layers
    dtype = torch.bfloat16

    print(f"\n[Export] Exporting to {output_path}")
    print(f"  Layers: {n_layers}, Vocab out: {wrapper.out_vocab}")
    print(f"  Export dtype: BF16 (dynamo), Device: {device}")

    dummy = build_dummy_inputs(config, batch=1, seq_len=3, past_seq_len=3,
                               device=device, dtype=dtype)
    input_names, output_names = build_io_names(n_layers)
    dynamic_shapes = build_dynamic_shapes(n_layers)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wrapper.eval()

    # Test forward
    print("[Export] Test forward...")
    with torch.no_grad():
        test_out = wrapper(*dummy)
    print(f"  Logits: {test_out[0].shape} ({test_out[0].dtype})")
    print(f"  Present K[0]: {test_out[1].shape} ({test_out[1].dtype})")

    # Export with dynamo
    print("[Export] Dynamo export...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            output_path,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            export_params=True,
            external_data=True,
            dynamo=True,
        )
    elapsed = time.time() - t0
    print(f"[Export] ONNX written in {elapsed:.1f}s")

    # Validate
    import onnx
    model = onnx.load(output_path, load_external_data=False)
    print(f"[Export] Graph: {len(model.graph.node)} nodes, "
          f"{len(model.graph.input)} inputs, {len(model.graph.output)} outputs")

    # SHA-256 weight renaming
    renamed, transposed = {}, []
    if do_rename and torch_model is not None:
        renamed, transposed = rename_weights(output_path, torch_model)

    return output_path, renamed, transposed


# ══════════════════════════════════════════════════════════════════════════════
#  Verification
# ══════════════════════════════════════════════════════════════════════════════

def verify_onnx(wrapper, config, onnx_path, device):
    """Compare ONNX outputs against PyTorch reference."""
    import onnxruntime as ort

    print(f"\n[Verify] Comparing ONNX vs PyTorch for {os.path.basename(onnx_path)}")

    dummy = build_dummy_inputs(config, batch=1, seq_len=5, past_seq_len=3,
                               device=device, dtype=torch.bfloat16)

    wrapper.eval()
    with torch.no_grad():
        ref_out = wrapper(*dummy)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    ort_inputs = {}
    for inp, tensor in zip(sess.get_inputs(), dummy):
        arr = tensor.cpu()
        if arr.dtype == torch.bfloat16:
            arr = arr.float()
        ort_inputs[inp.name] = arr.numpy()

    ort_out = sess.run(None, ort_inputs)

    ref_logits = ref_out[0].cpu().float().numpy()
    ort_logits = ort_out[0]
    max_diff = np.max(np.abs(ref_logits - ort_logits))
    mean_diff = np.mean(np.abs(ref_logits - ort_logits))
    print(f"  Logits max diff:  {max_diff:.6f}")
    print(f"  Logits mean diff: {mean_diff:.6f}")

    max_kv_diff = 0
    n_layers = config.num_hidden_layers
    for i in range(n_layers * 2):
        ref_kv = ref_out[1 + i].cpu().float().numpy()
        ort_kv = ort_out[1 + i]
        d = np.max(np.abs(ref_kv - ort_kv))
        if d > max_kv_diff:
            max_kv_diff = d
    print(f"  KV max diff:      {max_kv_diff:.6f}")

    threshold = 0.05  # BF16 rounding
    ok = max_diff < threshold and max_kv_diff < threshold
    print(f"  Status: {'PASS' if ok else 'FAIL'} (threshold={threshold})")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  Config / Manifest
# ══════════════════════════════════════════════════════════════════════════════

def write_config(config, output_dir, out_vocab, label):
    """Write model config JSON for C++ runtime."""
    cfg = {
        "model_type": "qwen3_lm",
        "label": label,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "num_hidden_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
        "out_vocab_size": out_vocab,
        "rope_theta": getattr(config, 'rope_parameters', {}).get('rope_theta', 1000000),
        "rms_norm_eps": config.rms_norm_eps,
        "tie_word_embeddings": config.tie_word_embeddings,
        "max_position_embeddings": config.max_position_embeddings,
    }
    if label == "audio":
        cfg["partial_vocab_offset"] = LM_PARTIAL_OFFSET
    path = os.path.join(output_dir, f"config_{label}.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[Config] Written to {path}")


def write_refit_manifest(output_dir, label, onnx_basename, renamed, transposed):
    """Write refit manifest for C++ adapter refit."""
    manifest = {
        "version": 1,
        "label": label,
        "onnx_path": onnx_basename,
        "weights_transposed": transposed,
        "weights_renamed": renamed,
    }
    path = os.path.join(output_dir, f"{onnx_basename}.refit_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Manifest] Written to {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen3ForCausalLM to ONNX for TensorRT")
    parser.add_argument("--model-dir", required=True,
                        help="Path to HF model directory (safetensors)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: models/onnx/lm-<name>/)")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", default="cuda",
                        help="Device for export (cuda or cpu)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX output against PyTorch")
    parser.add_argument("--no-rename", action="store_true",
                        help="Skip SHA-256 weight renaming")
    parser.add_argument("--full-only", action="store_true",
                        help="Only export full-vocab model")
    parser.add_argument("--partial-only", action="store_true",
                        help="Only export partial-vocab model")
    args = parser.parse_args()

    model_name = os.path.basename(os.path.normpath(args.model_dir))
    if args.output is None:
        args.output = os.path.join("models", "onnx", model_name)
    os.makedirs(args.output, exist_ok=True)

    # Load model in BF16
    print(f"[Load] Loading {args.model_dir} in BF16...")
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(args.model_dir)
    config._attn_implementation = "sdpa"  # Required for ONNX (no flash attention)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=args.device if args.device != "cpu" else None,
    )
    model.eval()

    print(f"[Load] Qwen3ForCausalLM: {config.num_hidden_layers}L, "
          f"H={config.hidden_size}, V={config.vocab_size}, "
          f"Nkv={config.num_key_value_heads}")

    do_rename = not args.no_rename

    # ── Export full-vocab model (Phase 1) ────────────────────────────────────
    if not args.partial_only:
        print("\n" + "=" * 70)
        print("  FULL-VOCAB MODEL (Phase 1 — Text + Audio)")
        print("=" * 70)

        wrapper_full = Qwen3LMFullWrapper(model).to(args.device).eval()
        full_path = os.path.join(args.output, "lm_full.onnx")
        _, renamed, transposed = export_onnx(
            wrapper_full, config, full_path, args.device,
            opset=args.opset, do_rename=do_rename, torch_model=wrapper_full)
        write_config(config, args.output, config.vocab_size, "full")
        if do_rename:
            write_refit_manifest(args.output, "full", "lm_full.onnx",
                                 renamed, transposed)

        if args.verify:
            verify_onnx(wrapper_full, config, full_path, args.device)

    # ── Export partial-vocab model (Phase 2) ──────────────────────────────────
    if not args.full_only:
        print("\n" + "=" * 70)
        print("  PARTIAL-VOCAB MODEL (Phase 2 — Audio Codes)")
        print("=" * 70)

        wrapper_partial = Qwen3LMPartialWrapper(
            model, LM_PARTIAL_OFFSET).to(args.device).eval()
        partial_path = os.path.join(args.output, "lm_audio.onnx")
        _, renamed, transposed = export_onnx(
            wrapper_partial, config, partial_path, args.device,
            opset=args.opset, do_rename=do_rename, torch_model=wrapper_partial)
        write_config(config, args.output, wrapper_partial.out_vocab, "audio")
        if do_rename:
            write_refit_manifest(args.output, "audio", "lm_audio.onnx",
                                 renamed, transposed)

        if args.verify:
            verify_onnx(wrapper_partial, config, partial_path, args.device)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"[Done] All exports written to {args.output}/")
    for f in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, f)
        if os.path.isfile(fpath):
            sz = os.path.getsize(fpath)
            if sz > 1024 * 1024:
                print(f"  {f:45s} {sz / 1024**3:.2f} GB")
            else:
                print(f"  {f:45s} {sz / 1024:.1f} KB")


if __name__ == "__main__":
    main()
