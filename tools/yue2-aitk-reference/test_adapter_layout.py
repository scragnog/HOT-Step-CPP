import os
import tempfile
import unittest
from pathlib import Path

from adapter_layout import LayoutError, fused_to_split, read_inventory, split_to_fused, validate

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:  # pragma: no cover
    torch = None


class LayoutTests(unittest.TestCase):
    def test_real_checkpoint_has_both_experts_and_fused_sites(self):
        root = Path(r"D:\Ace-Step-Latest\ai-toolkit\experiments\dookie-reference-20260916\output\dookie_yue2_default_300")
        files = list(root.glob("*250.safetensors"))
        if not files:
            self.skipTest("reference checkpoint is not installed")
        inv = validate(files[0])
        self.assertEqual(inv.errors, [])
        self.assertEqual(inv.experts, {"diffusion_model": 112, "text_encoders": 112})
        self.assertEqual(sum(1 for p in inv.pairs if p.site == "qkv_proj"), 56)
        self.assertEqual(sum(1 for p in inv.pairs if p.site == "gate_up_proj"), 56)

    @unittest.skipUnless(torch is not None, "torch and safetensors are required for file roundtrip")
    def test_real_checkpoint_export_reload_fuse_roundtrip(self):
        root = Path(r"D:\Ace-Step-Latest\ai-toolkit\experiments\dookie-reference-20260916\output\dookie_yue2_default_300")
        files = list(root.glob("*250.safetensors"))
        if not files:
            self.skipTest("reference checkpoint is not installed")
        source = {k: v.cpu() for k, v in load_file(str(files[0]), device="cpu").items()}
        inv = validate(files[0])
        split, metadata = fused_to_split(source, metadata=inv.metadata)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "split.safetensors"
            # safetensors requires unique storage ranges, so export contiguous
            # clones while retaining the shared-A contract in metadata.
            serial = {k: v.clone().contiguous() for k, v in split.items()}
            save_file(serial, str(path), metadata={k: str(v) for k, v in metadata.items()})
            reloaded = load_file(str(path), device="cpu")
            self.assertEqual(read_inventory(path).metadata["yue2_adapter_layout"], "split_shared_a_v1")
            fused, _ = split_to_fused(reloaded, metadata=read_inventory(path).metadata)
        self.assertEqual(set(fused), set(source))
        for key in source:
            self.assertTrue(torch.equal(fused[key], source[key]), key)

    @unittest.skipUnless(torch is not None, "torch is required for tensor interchange tests")
    def test_round_trip_preserves_shared_a_and_alpha_metadata(self):
        a = torch.arange(4096, dtype=torch.float32).reshape(2, 2048)
        b = torch.arange(8192, dtype=torch.float32).reshape(4096, 2)
        source = {"diffusion_model.model.layers.0.self_attn.qkv_proj.lora_A.weight": a,
                  "diffusion_model.model.layers.0.self_attn.qkv_proj.lora_B.weight": b}
        split, md = fused_to_split(source, metadata={"alpha": "32"})
        self.assertEqual(md["alpha"], "32")
        self.assertIs(split["diffusion_model.model.layers.0.self_attn.qkv_proj.q.lora_A.weight"], a)
        with self.assertRaises(LayoutError):
            split_to_fused({**split, "diffusion_model.model.layers.0.self_attn.qkv_proj.k.lora_A.weight": a + 1})
        fused, md2 = split_to_fused(split, metadata=md)
        self.assertTrue(torch.equal(fused[source.keys().__iter__().__next__()], a))
        self.assertTrue(torch.equal(fused["diffusion_model.model.layers.0.self_attn.qkv_proj.lora_B.weight"], b))
        self.assertEqual(md2, md)

    def test_missing_expert_and_mismatched_pair_are_reported(self):
        if torch is None:
            self.skipTest("torch is required to create a temporary safetensors fixture")
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.safetensors"
            save_file({"diffusion_model.model.layers.0.self_attn.qkv_proj.lora_A.weight": torch.zeros(2, 3),
                       "diffusion_model.model.layers.0.self_attn.qkv_proj.lora_B.weight": torch.zeros(5, 2)}, str(path))
            errors = validate(path).errors
            self.assertTrue(any("missing expert: text_encoders" in e for e in errors))
            self.assertTrue(any("fused factors" in e for e in errors))

    @unittest.skipUnless(torch is not None, "torch is required for malformed split test")
    def test_missing_split_factor_is_rejected(self):
        with self.assertRaises(LayoutError):
            split_to_fused({"diffusion_model.model.layers.0.mlp.gate_up_proj.q.lora_A.weight": torch.zeros(2, 3)})


if __name__ == "__main__":
    unittest.main()
