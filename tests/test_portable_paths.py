"""Manifest portability must preserve data hashes and reject path escapes."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import _run_support as support
import run_zhou2025_repro as runner
from thz_dma.paths import relative_path
from zhou2025_repro.data import cache_directory


class PortablePathsTest(unittest.TestCase):
    def test_relative_windows_paths_and_escapes(self):
        self.assertEqual(relative_path(r"parameters\train_000000.json").parts,
                         ("parameters", "train_000000.json"))
        for name in ("", ".", "/outside", r"C:\outside", r"C:outside",
                     "../outside", r"data\..\outside"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                relative_path(name)
        with self.assertRaises(ValueError):
            cache_directory(ROOT / "runs/wrapper", {"source_run_id": "nested/run"})

    def test_new_manifests_and_job_paths_use_forward_slashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "src/check.py"
            source.parent.mkdir()
            source.write_text("pass\n", encoding="utf-8")
            manifest = root / "manifest.json"
            with patch.object(support, "PROJECT_ROOT", root):
                support.write_code_manifest(manifest, [source])
            self.assertEqual(json.loads(manifest.read_text())[0]["path"], "src/check.py")
        args = SimpleNamespace(config=ROOT / "configs/zhou2025_parametric_mimo.toml",
                               smoke=True, seed=None, device="cpu")
        config = runner.settings_from_args(args)
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["job"]["config_path"], "configs/zhou2025_parametric_mimo.toml")

    def test_relocated_windows_cache_keeps_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, wrapper = root / "runs/original", root / "runs/wrapper"
            (source / "data").mkdir(parents=True)
            wrapper.mkdir()
            channel = np.ones((4, 2, 3), dtype=np.complex64)
            record = {"scene_id": "11:0"}
            np.savez(source / "data/train_00000.npz", channel=channel, record=json.dumps(record))
            metadata = {
                "kind": "clean", "scenario": "dma", "complex_scale": 1.,
                "channel_shape": [4, 2, 3], "counts": {"train": 1},
                "storage": "existing_npz", "index": {"train": ["train_00000.npz"]},
                "source_directory": r"C:\old_project\runs\original",
                "file_sha256": {r"data\train_00000.npz": support.hash_file(source / "data/train_00000.npz")},
            }
            (wrapper / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")
            (wrapper / "status.json").write_text('{"status":"complete"}', encoding="utf-8")
            before = support.hash_file(wrapper / "dataset.json")
            with patch.object(runner, "PROJECT_ROOT", root):
                data = runner.load_data("wrapper")
                np.testing.assert_array_equal(data.physical("train", [0])[0], channel)
                self.assertEqual(data.record("train", 0), record)
                data.close()
                self.assertEqual(support.hash_file(wrapper / "dataset.json"), before)
                (source / "data/train_00000.npz").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "data changed"):
                    runner.load_data("wrapper")


if __name__ == "__main__":
    unittest.main()
