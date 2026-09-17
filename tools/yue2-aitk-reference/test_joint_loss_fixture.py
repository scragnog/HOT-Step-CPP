import tempfile
import unittest
from pathlib import Path
import struct
import subprocess
import os

import numpy as np
import torch
import torch.nn.functional as F

from joint_loss_fixture import export_fixtures, export_native_loss_case, load_reference_bodies, make_case, make_conditioning_case


class JointLossFixtureTests(unittest.TestCase):
    def test_actual_reference_bodies_are_loaded(self):
        bodies, digest = load_reference_bodies()
        self.assertTrue({"_train_prefix", "_ar_inputs", "_ar_losses", "get_noise_prediction", "parse_caption"}.issubset(bodies))
        self.assertEqual(len(digest), 64)

    def test_full_and_dropped_abc_target_boundaries(self):
        full = make_case("full", abc=True, start=2, end=5)
        dropped = make_case("drop", abc=False, start=2, end=5)
        self.assertEqual(full["ar_targets"][:4].tolist(), [101, 102, 151848, 151851])
        self.assertEqual(dropped["ar_targets"][:2].tolist(), [151848, 151851])
        self.assertEqual(full["ar_targets"][-1].item(), 151852)
        self.assertEqual(full["conditioning_ids"][-1].item(), 152011)  # codec token, no synthetic MUSIC_END
        self.assertNotEqual(full["ar_targets"].numel(), full["conditioning_ids"].numel())

    def test_real_end_and_windowed_ar_targets(self):
        case = make_case("song", abc=True, start=0, end=6)
        self.assertEqual(case["ar_targets"][-1].item(), 151852)
        self.assertEqual(case["ar_targets"].numel(), 2 + 2 + 6 + 1)
        self.assertEqual(case["conditioning_ids"][-1].item(), 152012)

    def test_kl_direction_and_target_reduction(self):
        case = make_case("kl", abc=True, start=0, end=6, chunk=512)
        self.assertTrue(case["ce_requires_grad"])
        self.assertTrue(case["kl_requires_grad"])
        # The actual body computes KL(base || adapted): input adapted log-probs,
        # target detached base log-probs. Reverse KL is intentionally different.
        adapted = torch.tensor([[2.0, 0.5, -1.0]], requires_grad=True)
        base = torch.tensor([[0.2, 1.4, -0.3]])
        forward = F.kl_div(F.log_softmax(adapted, -1), F.log_softmax(base, -1), log_target=True, reduction="sum")
        reverse = F.kl_div(F.log_softmax(base, -1), F.log_softmax(adapted, -1), log_target=True, reduction="sum")
        self.assertNotAlmostEqual(forward.item(), reverse.item(), places=6)
        self.assertTrue(torch.isfinite(case["kl"]))

    def test_chunk_128_matches_512(self):
        a = make_case("c128", abc=True, start=0, end=6, chunk=128)
        b = make_case("c512", abc=True, start=0, end=6, chunk=512)
        self.assertTrue(torch.allclose(a["ce"], b["ce"], atol=1e-7, rtol=0))
        self.assertTrue(torch.allclose(a["kl"], b["kl"], atol=1e-7, rtol=0))

    def test_adapter_off_base_prefill_is_recorded_and_restored(self):
        case = make_case("flags", abc=True, start=0, end=6)
        self.assertEqual(case["prefill_active"], [True, False])
        self.assertEqual(case["prefill_requires_grad"], [True, False])

    def test_export_is_portable_and_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = export_fixtures(td)
            self.assertEqual(manifest["schema"], "yue2_joint_loss_fixture_v1")
            for case in manifest["cases"]:
                with np.load(Path(td) / f"{case['name']}.npz") as data:
                    self.assertTrue("ar_targets" in data.files or "nar_cache_0_k" in data.files)
            self.assertTrue((Path(td) / "manifest.json").is_file())

    def test_actual_noise_prediction_dropout_crop_and_detached_cache(self):
        case = make_conditioning_case()
        self.assertEqual(case["window"], [2, 5])
        self.assertTrue(torch.equal(case["nar_latent"], torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)))
        self.assertEqual(case["ar_prefill_active"], [True, False, True])
        self.assertEqual(case["ar_prefill_requires_grad"], [True, False, False])
        self.assertEqual(case["abc_mode_used"], "off")
        self.assertTrue(all(not key.requires_grad and not value.requires_grad for key, value in case["nar_cache"]))
        self.assertEqual(case["nar_prompt_len"], 3 + 2 + 1 + 3)

    def test_native_cpu_probe_matches_torch_for_128_and_512_chunks(self):
        repo = Path(__file__).resolve().parents[2]
        source = repo / "tools" / "yue2-aitk-reference" / "joint_loss_probe.cpp"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exe = root / "joint_loss_probe.exe"
            vcvars = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")
            if vcvars.is_file():
                build_script = root / "build-probe.cmd"
                build_script.write_text(f'@echo off\ncall "{vcvars}"\ncl /nologo /std:c++17 /EHsc /O2 "{source}" /Fe:"{exe}"\n', encoding="utf-8")
                build = subprocess.run(["cmd.exe", "/d", "/c", str(build_script)], cwd=repo, capture_output=True, text=True)
            else:
                build = subprocess.run(["cl", "/nologo", "/std:c++17", "/EHsc", "/O2", str(source), f"/Fe:{exe}"],
                                       cwd=repo, capture_output=True, text=True)
            if build.returncode != 0:
                self.fail(f"Native loss probe failed to compile: {build.stdout}\n{build.stderr}")
            expected = export_native_loss_case(root / "c128", chunk=128)
            export_native_loss_case(root / "c512", chunk=512)
            short_expected = export_native_loss_case(root / "short", chunk=128, positions=7)
            outputs = []
            expected_by_label = {"c128": expected, "c512": expected, "short": short_expected}
            for label in ("c128", "c512", "short"):
                inp = root / label / "native_loss.bin"
                out = root / label / "native_loss.out"
                run = subprocess.run([str(exe), str(inp), str(out)], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                raw = out.read_bytes()
                ce, kl = struct.unpack_from("<dd", raw)
                grad = np.frombuffer(raw, dtype=np.float32, offset=16).copy()
                outputs.append((label, ce, kl, grad))
            for label, ce, kl, grad in outputs:
                expected = expected_by_label[label]
                self.assertAlmostEqual(ce, expected["ce"].item(), places=6)
                self.assertAlmostEqual(kl, expected["kl"].item(), places=6)
                np.testing.assert_allclose(grad, expected["gradient"].numpy().reshape(-1), atol=3e-6, rtol=3e-6)

    def test_native_cuda_probe_matches_large_vocab_when_guarded(self):
        probe = os.environ.get("YUE2_JOINT_LOSS_CUDA_PROBE")
        if not probe:
            self.skipTest("set YUE2_JOINT_LOSS_CUDA_PROBE when running under the GPU Work guard")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for label, positions, vocab, chunk, offset in (
                ("vocab", 7, 184704, 3, 0.0),
                ("final_chunk", 130, 17, 128, 0.0),
                ("shifted", 7, 184704, 3, 1000.0),
            ):
                with self.subTest(case=label):
                    case_dir = root / label
                    expected = export_native_loss_case(case_dir, chunk=chunk, positions=positions,
                                                       vocab=vocab, logit_offset=offset)
                    output = case_dir / "cuda.out"
                    run = subprocess.run([probe, str(case_dir / "native_loss.bin"), str(output)], capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    result = __import__("joint_loss_fixture").compare_native_output(output, expected)
                    self.assertLessEqual(result["gradient_max_abs"], 3e-5)


if __name__ == "__main__":
    unittest.main()
