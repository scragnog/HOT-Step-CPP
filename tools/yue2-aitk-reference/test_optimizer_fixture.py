import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("optimizer_fixture", HERE / "optimizer_fixture.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OptimizerFixtureContractTests(unittest.TestCase):
    def test_contract_captures_toolkit_overrides_and_state_boundary(self):
        contract = MODULE.contract_metadata()
        construction = contract["construction"]
        self.assertEqual(construction["eps"], 1e-6)
        self.assertEqual(construction["effective_state_bits"], 8)
        self.assertEqual(construction["optim_bits_argument"], 32)
        self.assertEqual(construction["min_8bit_size"], 4096)
        self.assertEqual(construction["block_size"], 256)
        self.assertEqual(construction["trainable_parameter_dtype"], "float32")
        self.assertIn("state1:uint8", contract["state"]["large_tensor"])
        self.assertIn("state1:float32", contract["state"]["small_tensor"])

    def test_inspect_runs_without_reference_python_environment(self):
        result = subprocess.run(
            [sys.executable, str(HERE / "optimizer_fixture.py"), "--inspect"],
            check=True, capture_output=True, text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["schema"], "yue2-aitk-adamw8bit-fixture-v1")
        self.assertEqual(payload["construction"]["block_wise"], True)

    def test_default_mode_is_inspection_only(self):
        result = subprocess.run(
            [sys.executable, str(HERE / "optimizer_fixture.py")],
            check=True, capture_output=True, text=True,
        )
        self.assertNotIn('"optimizer_executed"', result.stdout)

    def test_reference_sources_still_contain_the_contract_we_export(self):
        toolkit = Path(r"D:\Ace-Step-Latest\ai-toolkit\toolkit\optimizer.py").read_text(encoding="utf-8")
        bnb = Path(r"D:\Ace-Step-Latest\ai-toolkit\venv\Lib\site-packages\bitsandbytes\optim\optimizer.py").read_text(encoding="utf-8")
        adamw = Path(r"D:\Ace-Step-Latest\ai-toolkit\venv\Lib\site-packages\bitsandbytes\optim\adamw.py").read_text(encoding="utf-8")
        self.assertIn("bitsandbytes.optim.AdamW8bit", toolkit)
        self.assertIn("eps=1e-6", toolkit)
        self.assertIn("min_8bit_size=4096", bnb)
        self.assertIn("blocksize = 256", bnb)
        self.assertIn('"adam"', adamw)
        self.assertIn("8,  # Hardcoded to 8 bits", adamw)
        trainer = Path(r"D:\Ace-Step-Latest\ai-toolkit\jobs\process\BaseSDTrainProcess.py")
        if trainer.is_file():
            text = trainer.read_text(encoding="utf-8")
            self.assertIn("network.force_to(self.device_torch,dtype=torch.float32)", text.replace(" ", ""))

    def test_state_serializer_keeps_arrays_and_step_scalar(self):
        import torch
        state = torch.arange(256, dtype=torch.uint8)
        arrays = {}
        manifest = MODULE._state_arrays(
            "step_1_large", {"step": 1, "state1": state}, arrays
        )
        self.assertEqual(manifest["step"], {"scalar": 1})
        self.assertEqual(manifest["state1"]["array"], "step_1_large_state1")
        self.assertEqual(arrays["step_1_large_state1"].tolist(), list(range(256)))
        state.zero_()
        self.assertEqual(arrays["step_1_large_state1"].tolist(), list(range(256)))


if __name__ == "__main__":
    unittest.main()
