import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


PROBE = Path(os.environ.get("AITK_CONVROT_PROBE", "_experiments/aitk-port/bin/convrot_probe.exe")).resolve()


@unittest.skipUnless(PROBE.is_file(), "Build the native ConvRot probe first")
class ConvRotProbeTests(unittest.TestCase):
    def reject(self, content):
        with tempfile.TemporaryDirectory() as directory:
            input_path, output = Path(directory) / "input.bin", Path(directory) / "output.bin"
            input_path.write_bytes(content)
            result = subprocess.run([str(PROBE), str(input_path), str(output)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_rejects_truncated_fixture_before_writing(self):
        self.reject(struct.pack("<6I", 0x314B5441, 2, 16, 8, 16, 1))

    def test_rejects_invalid_dimensions_before_allocating(self):
        self.reject(struct.pack("<6I", 0x314B5441, 2, 0xFFFFFFFF, 8, 16, 1))

    def test_refuses_input_as_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            content = b"preserve fixture"
            path.write_bytes(content)
            result = subprocess.run([str(PROBE), str(path), str(path)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
