"""Small stdlib checks for the shared run-archive writes."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _run_support import (  # noqa: E402
    RunArchive,
    hash_file,
    read_toml,
    require_unit_pilot_energy,
)


class RunSupportTest(unittest.TestCase):
    def test_reference_archive_without_atmosphere(self):
        import _run_support as support
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'pyproject.toml').write_text('[project]\nname="reference"\n')
            source=root/'run.py';source.write_text('pass\n')
            with patch.object(support,'PROJECT_ROOT',root),patch.object(support,'__file__',str(source)):
                archive=support.create_run_archive('reference','reference',config_path=source,
                    config={'n_antennas':64},seeds={},script_path=str(source),environment={},checkpoint_note='')
            saved=json.loads((archive.directory/'config_resolved.json').read_text())
            self.assertIsNone(saved['absorption_table_sha256'])
            self.assertTrue((archive.directory/'code_manifest.json').exists())

    def test_resume_rejects_changed_config_and_sources(self):
        from run_direction1_stage2a import validate_resume, PROJECT_ROOT
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = {"seed": 7}
            source = "src/thz_dma/stage2a.py"
            manifest = [{"path": source, "sha256": hash_file(PROJECT_ROOT / source)}]
            (directory / "config_resolved.json").write_text(json.dumps({"config": config}))
            (directory / "code_manifest.json").write_text(json.dumps(manifest))
            validate_resume(directory, config)
            with self.assertRaises(ValueError):
                validate_resume(directory, {"seed": 8})
            manifest[0]["sha256"] = "changed"
            (directory / "code_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                validate_resume(directory, config)

    def test_archive_writes_valid_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = RunArchive("smoke", directory)
            archive.write_status("running", run_id="smoke")
            archive.write_metrics([{"scene": 1, "value": 2.5}])

            status = json.loads(archive.status_path.read_text(encoding="utf-8"))
            with (directory / "metrics_raw.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(status["status"], "running")
            self.assertEqual(rows, [{"scene": "1", "value": "2.5"}])
            self.assertEqual(len(hash_file(directory / "metrics_raw.csv")), 64)

            config_path = directory / "config.toml"
            config_path.write_text("[experiment]\nseed = 7\n", encoding="utf-8")
            self.assertEqual(read_toml(config_path)["experiment"]["seed"], 7)

    def test_unit_pilot_energy_guard(self) -> None:
        require_unit_pilot_energy(1.0, "test")
        with self.assertRaises(ValueError):
            require_unit_pilot_energy(0.5, "test")


if __name__ == "__main__":
    unittest.main()
