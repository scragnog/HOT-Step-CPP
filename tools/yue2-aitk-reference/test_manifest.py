import json
import struct
import tempfile
import unittest
from pathlib import Path

from manifest import ManifestError, safetensors_header


def write_header(path: Path, header: object, payload: bytes = b"") -> None:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class SafetensorsHeaderTests(unittest.TestCase):
    def test_inventory_reads_metadata_and_tensor_shapes_without_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.safetensors"
            write_header(path, {
                "__metadata__": {"format": "pt", "expert": "nar"},
                "diffusion_model.layer.weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
            }, b"0123456789ab")
            record = safetensors_header(path)
            self.assertEqual(record["tensor_count"], 1)
            self.assertEqual(record["metadata"]["expert"], "nar")
            self.assertEqual(record["tensors"][0]["payload_bytes"], 12)

    def test_rejects_truncated_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.safetensors"
            path.write_bytes(struct.pack("<Q", 20) + b"{}")
            with self.assertRaises(ManifestError):
                safetensors_header(path)

    def test_rejects_malformed_tensor_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.safetensors"
            write_header(path, {"x": {"dtype": "F32", "shape": [1], "data_offsets": [8]}})
            with self.assertRaises(ManifestError):
                safetensors_header(path)


if __name__ == "__main__":
    unittest.main()
