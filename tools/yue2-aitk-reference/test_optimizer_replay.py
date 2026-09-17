import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import optimizer_replay as replay


ROOT = Path(__file__).parents[2] / "_experiments" / "aitk-port" / "optimizer-fp32-cuda-02"


class OptimizerReplayTests(unittest.TestCase):
    def test_preserved_fixture_contains_all_five_state_trajectories(self):
        result = replay.validate(ROOT)
        self.assertEqual(result["snapshots"], 5)
        self.assertEqual(result["device"], "cuda")
        self.assertGreater(result["arrays"], 20)

    def test_binary_archive_is_portable_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fixture.aitkopt"
            replay.write_binary(ROOT, output)
            self.assertTrue(output.read_bytes().startswith(b"AITKOPT1"))
            with self.assertRaises(FileExistsError): replay.write_binary(ROOT, output)

    def test_compare_detects_every_array_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            metadata, arrays = replay.load_fixture(ROOT)
            arrays["parameter_large_initial"][0] += 1.0
            np.savez(target / "optimizer_tensors.npz", **arrays)
            (target / "optimizer_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError): replay.compare(ROOT, target, 0.0, 0.0)

    def test_cpu_formula_replay_stays_close_but_is_diagnostic(self):
        replayed, note = replay.replay_cpu(ROOT)
        _, expected = replay.load_fixture(ROOT)
        self.assertIn("diagnostic", note["warning"])
        worst = max(float(np.max(np.abs(expected[name].astype(np.float64) - value.astype(np.float64))))
                    for name, value in replayed.items())
        self.assertLess(worst, 1e-5)


if __name__ == "__main__":
    unittest.main()
