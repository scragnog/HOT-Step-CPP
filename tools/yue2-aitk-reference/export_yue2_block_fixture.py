"""Export a small, source-faithful YuE2 fused-LoRA autograd fixture.

The exporter extracts the YuE2 layer definitions and Toolkit LoRA wrapper from
the pinned AI Toolkit sources, then quantizes the fused qkv/gate_up linears with
the actual ConvRot backend. It writes raw little-endian tensors and a schema;
it does not export a safetensors checkpoint or alter source models.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_nodes(path: Path, names: set[str], namespace: dict) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    missing = names - {node.name for node in nodes}
    if missing:
        raise RuntimeError(f"{path}: missing definitions: {sorted(missing)}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def load_yue2_layer(model_path: Path):
    namespace = {"torch": torch, "nn": nn, "F": F, "math": math,
                 "Optional": Optional, "Tuple": Tuple, "List": List,
                 "CONTEXT": 24576,
                 "ckpt": types.SimpleNamespace(checkpoint=None),
                 "OstrisModelMixin": object}
    extract_nodes(model_path, {"YuE2Config", "RMSNorm", "rope_cos_sin", "apply_rope",
                               "YuE2Attention", "YuE2MLP", "YuE2Layer"}, namespace)
    return namespace


def load_toolkit_lora(network_mixins_path: Path, lora_path: Path):
    # Extract the real mixin/wrapper classes to avoid importing the full trainer
    # dependency graph (diffusers, transformers, and model registries).
    namespace = {"torch": torch, "nn": nn, "math": math, "weakref": __import__("weakref"),
                 "Module": object, "Network": object, "QTensor": type(None),
                 "Optional": Optional, "Union": Union, "List": List,
                 "Dict": Dict, "Type": Type, "Any": Any,
                 "LINEAR_MODULES": ["Linear", "OstrisLinear"], "CONV_MODULES": ["Conv2d"],
                 "ExtractMode": object}
    extract_nodes(network_mixins_path, {"broadcast_and_multiply", "ExtractableModuleMixin",
                                        "ToolkitModuleMixin"}, namespace)
    namespace["ExtractableModuleMixin"] = namespace["ExtractableModuleMixin"]
    lora_ns = dict(namespace)
    lora_ns.update({"ToolkitModuleMixin": namespace["ToolkitModuleMixin"],
                    "ExtractableModuleMixin": namespace["ExtractableModuleMixin"]})
    extract_nodes(lora_path, {"LoRAModule"}, lora_ns)
    return lora_ns["LoRAModule"]


class NetworkStub:
    network_type = "lora"
    is_active = True
    is_merged_in = False
    is_lorm = False
    _multiplier = 1.0
    torch_multiplier = torch.tensor(1.0)


def write_tensor(output: Path, name: str, tensor: torch.Tensor, entries: dict) -> None:
    value = tensor.detach().to(device="cpu").contiguous()
    if value.dtype == torch.int8:
        array = value.numpy().astype("i1", copy=False)
        dtype = "i8"
    elif value.dtype == torch.bfloat16:
        array = value.view(torch.uint16).numpy().astype("<u2", copy=False)
        dtype = "bf16"
    else:
        array = value.float().numpy().astype("<f4", copy=False)
        dtype = "f32"
    filename = name + ".bin"
    path = output / filename
    path.write_bytes(array.tobytes(order="C"))
    entries[name] = {"file": filename, "dtype": dtype, "shape": list(value.shape),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--toolkit", type=Path, default=Path(r"D:\Ace-Step-Latest\ai-toolkit"))
    parser.add_argument("--intermediates", action="store_true",
                        help="export hooked norm/ConvRot inputs, outputs, and gradients")
    parser.add_argument("--prefix-tokens", type=int, default=0,
                        help="NAR-style detached BF16 KV prefix length; zero keeps the AR fixture")
    args = parser.parse_args()
    if args.prefix_tokens < 0:
        parser.error("--prefix-tokens must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    torch.manual_seed(20260917)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260917)

    model_path = args.toolkit / "extensions_built_in/audio_models/yue2/src/model.py"
    mixins_path = args.toolkit / "toolkit/network_mixins.py"
    lora_path = args.toolkit / "toolkit/lora_special.py"
    convrot_path = args.toolkit / "toolkit/util/convrot_quant.py"
    ostris_path = args.toolkit / "toolkit/util/ostris_quant.py"
    yue = load_yue2_layer(model_path)
    LoRAModule = load_toolkit_lora(mixins_path, lora_path)

    # This is the actual Toolkit quantizer/converter, including its ConvRot
    # custom operator path on CUDA and dequantized CPU fallback.
    sys.path.insert(0, str(args.toolkit.resolve()))
    from toolkit.util.convrot_quant import get_convrot_quantizer
    from toolkit.util.ostris_quant import convert_linear_to_ostris

    cfg = yue["YuE2Config"]()
    cfg.hidden_size, cfg.intermediate_size = 256, 512
    cfg.num_hidden_layers, cfg.num_attention_heads = 1, 4
    cfg.num_key_value_heads, cfg.head_dim = 2, 64
    cfg.rms_norm_eps, cfg.rope_theta = 1e-6, 1_000_000.0
    layer = yue["YuE2Layer"](cfg).to(device=device, dtype=torch.bfloat16)
    quantized_sites = {
        "qkv": layer.self_attn.qkv_proj,
        "output": layer.self_attn.o_proj,
        "gate_up": layer.mlp.gate_up_proj,
        "down": layer.mlp.down_proj,
    }
    for name, module in quantized_sites.items():
        quantizer = get_convrot_quantizer("convrot8")
        if quantizer is None or not convert_linear_to_ostris(module, quantizer):
            raise RuntimeError(f"failed to install ConvRot on fused {name}")

    network = NetworkStub()
    # ToolkitModuleMixin.forward indexes multiplier.size(0), so preserve the
    # batch-shaped multiplier contract even for this single-batch fixture.
    network.torch_multiplier = torch.ones(1, device=device)
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    wrappers = {}
    for name, module in quantized_sites.items():
        wrapper = LoRAModule(f"fixture.{name}", module, lora_dim=4, alpha=4, network=network)
        wrapper.to(device=device, dtype=torch.float32)
        wrapper.apply_to()
        wrapper.lora_down.weight.requires_grad_(True)
        wrapper.lora_up.weight.requires_grad_(True)
        wrappers[name] = wrapper
        with torch.no_grad():
            wrapper.lora_down.weight.copy_(torch.randn_like(wrapper.lora_down.weight) * 0.03)
            wrapper.lora_up.weight.copy_(torch.randn_like(wrapper.lora_up.weight) * 0.03)

    # LoRAModule.apply_to replaces each original module's forward method. A
    # hook on the original module still surrounds that redirected forward,
    # which captures the actual wrapper boundary rather than only the frozen
    # org_forward call. Retain references until backward has populated grads.
    intermediate_refs = {}
    intermediate_handles = []
    if args.intermediates:
        intermediate_sites = {
            "input_layernorm": layer.input_layernorm,
            "post_attention_layernorm": layer.post_attention_layernorm,
            "q_norm": layer.self_attn.q_norm,
            "k_norm": layer.self_attn.k_norm,
            **quantized_sites,
        }

        def capture(site, module, inputs, result):
            inp = inputs[0] if inputs else None
            out = result[0] if isinstance(result, (tuple, list)) else result
            if not isinstance(inp, torch.Tensor) or not isinstance(out, torch.Tensor):
                raise RuntimeError(f"intermediate hook {site} did not see tensor input/output")
            if inp.requires_grad:
                inp.retain_grad()
            if out.requires_grad:
                out.retain_grad()
            intermediate_refs[site] = {"input": inp, "output": out}

        for site, module in intermediate_sites.items():
            intermediate_handles.append(
                module.register_forward_hook(lambda m, i, o, site=site: capture(site, m, i, o)))

    x = torch.randn(1, 3, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    positions = torch.arange(args.prefix_tokens, args.prefix_tokens + 3,
                             device=device, dtype=torch.long).unsqueeze(0)
    cos, sin = yue["rope_cos_sin"](positions, cfg.head_dim, cfg.rope_theta)
    upstream = torch.randn(1, 3, 256, device=device, dtype=torch.float32).to(torch.bfloat16)
    prefix_k = prefix_v = None
    if args.prefix_tokens:
        prefix_shape = (1, cfg.num_key_value_heads, args.prefix_tokens, cfg.head_dim)
        prefix_k = torch.randn(prefix_shape, device=device, dtype=torch.float32).to(torch.bfloat16).detach()
        prefix_v = torch.randn(prefix_shape, device=device, dtype=torch.float32).to(torch.bfloat16).detach()
        output, _ = layer(x, cos, sin, prefix_kv=(prefix_k, prefix_v),
                          causal=False, return_kv=False)
    else:
        output, _ = layer(x, cos, sin, causal=True, return_kv=False)
    (output.float() * upstream.float()).sum().backward()

    for handle in intermediate_handles:
        handle.remove()

    entries = {}
    write_tensor(args.output, "x", x, entries)
    write_tensor(args.output, "upstream_dy", upstream, entries)
    write_tensor(args.output, "cos", cos, entries)
    write_tensor(args.output, "sin", sin, entries)
    if prefix_k is not None:
        write_tensor(args.output, "prefix_k", prefix_k, entries)
        write_tensor(args.output, "prefix_v", prefix_v, entries)
    write_tensor(args.output, "input_layernorm_weight", layer.input_layernorm.weight, entries)
    write_tensor(args.output, "post_attention_layernorm_weight", layer.post_attention_layernorm.weight, entries)
    write_tensor(args.output, "q_norm_weight", layer.self_attn.q_norm.weight, entries)
    write_tensor(args.output, "k_norm_weight", layer.self_attn.k_norm.weight, entries)
    for prefix, module in quantized_sites.items():
        write_tensor(args.output, prefix + "_qdata", module.cr8_qdata, entries)
        write_tensor(args.output, prefix + "_scales", module.cr8_scales.view(torch.float32), entries)
    for prefix, wrapper in wrappers.items():
        write_tensor(args.output, prefix + "_lora_A", wrapper.lora_down.weight, entries)
        write_tensor(args.output, prefix + "_lora_B", wrapper.lora_up.weight, entries)
        write_tensor(args.output, prefix + "_dA", wrapper.lora_down.weight.grad, entries)
        write_tensor(args.output, prefix + "_dB", wrapper.lora_up.weight.grad, entries)
    write_tensor(args.output, "forward", output, entries)
    write_tensor(args.output, "dx", x.grad, entries)
    if args.intermediates:
        for site, refs in intermediate_refs.items():
            write_tensor(args.output, f"stage_{site}_input", refs["input"], entries)
            write_tensor(args.output, f"stage_{site}_output", refs["output"], entries)
            if refs["input"].grad is None or refs["output"].grad is None:
                raise RuntimeError(f"missing retained gradient for intermediate {site}")
            write_tensor(args.output, f"stage_{site}_dx", refs["input"].grad, entries)
            write_tensor(args.output, f"stage_{site}_dy", refs["output"].grad, entries)

    schema = {
        "schema": 1,
        "scope": "single actual Toolkit YuE2Layer fused ConvRot + trainable LoRA block",
        "cpu_scope": "CPU uses Toolkit ConvRot dequant fallback; CUDA uses the custom ConvRot path when available",
        "device": str(device),
        "seed": 20260917,
        "config": {"hidden_size": 256, "intermediate_size": 512, "num_heads": 4,
                    "num_kv_heads": 2, "head_dim": 64, "tokens": 3, "rank": 4,
                    "alpha": 4, "rotation": 256, "prefix_tokens": args.prefix_tokens},
        "loss": "sum(forward * upstream_dy)",
        "attention": {"causal": args.prefix_tokens == 0,
                       "prefix_kv": args.prefix_tokens > 0,
                       "prefix_shape": ([1, cfg.num_key_value_heads, args.prefix_tokens, cfg.head_dim]
                                         if args.prefix_tokens else None),
                       "rope_positions": [args.prefix_tokens, args.prefix_tokens + 1,
                                          args.prefix_tokens + 2]},
        "intermediates": args.intermediates,
        "sites": {name: {"input": module.in_features, "output": module.out_features}
                  for name, module in quantized_sites.items()},
        "source_sha256": {str(path.relative_to(args.toolkit)): source_hash(path)
                          for path in (model_path, mixins_path, lora_path, convrot_path, ostris_path)},
        "tensors": entries,
    }
    (args.output / "schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tensor_count": len(entries),
                      "device": str(device), "schema": str(args.output / "schema.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
