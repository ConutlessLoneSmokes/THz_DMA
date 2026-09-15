"""Small stdlib checks for the shared run-archive writes."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
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
