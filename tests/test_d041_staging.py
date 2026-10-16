"""First-round ordering and conservative interpretation of frozen controls."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from analyze_d041_diagnostic import noiseless_diagnosis


def fixture_metrics(*, smoke=False, failed_band="wide"):
    rows = []
    for band in ("narrow", "wide"):
        for operator in ("digital", "dma_J192", "dma_J8"):
            for method in ("clean_ddpm", "pinv_ls", "oracle_geometry_ls"):
                error = .2 if method == "clean_ddpm" and operator == "dma_J192" and band == failed_band else 1e-4
                for scene in ("scene0", "scene1"):
                    for condition in ("noiseless_v1e-08", "noiseless_v1e-10"):
                        rows.append({"band": band, "operator": operator, "estimator": method,
                            "stage": "reduced-noiseless" if operator == "dma_J8" else "full-noiseless",
                            "split": "test", "scene_id": scene, "condition": condition,
                            "nmse_db": 10*math.log10(error), "smoke_only": smoke,
                            "prediction_energy_ratio": 1., "normalized_correlation": 1-error/2,
                            "row_space_nmse_linear": error, "null_space_nmse_linear": 0.})
    return pd.DataFrame(rows)


class InterpretationTest(unittest.TestCase):
    def test_joint_pattern_requires_narrow_dma_and_wide_digital(self):
        _, diagnosis = noiseless_diagnosis(fixture_metrics(), -10., .1)
        codes = {row["code"] for row in diagnosis["candidate_findings"]}
        self.assertIn("wideband_dma_joint_failure", codes)
        self.assertFalse(diagnosis["unique_causal_attribution"])
        _, narrow_failed = noiseless_diagnosis(fixture_metrics(failed_band="narrow"), -10., .1)
        codes = {row["code"] for row in narrow_failed["candidate_findings"]}
        self.assertIn("full_dma_recovery_failure", codes)
        self.assertNotIn("wideband_dma_joint_failure", codes)

    def test_smoke_never_attributes_formal_learning_failure(self):
        _, diagnosis = noiseless_diagnosis(fixture_metrics(smoke=True), -10., .1)
        self.assertEqual([r["code"] for r in diagnosis["candidate_findings"]], ["smoke_only"])

    def test_numerical_variance_blocks_structural_interpretation(self):
        frame = fixture_metrics()
        frame.loc[(frame.band == "wide") & (frame.operator == "dma_J192") &
                  (frame.estimator == "clean_ddpm") & (frame.condition == "noiseless_v1e-10"), "nmse_db"] = -40.
        controls, diagnosis = noiseless_diagnosis(frame, -10., .1)
        self.assertFalse(diagnosis["numerical_variance_stable"])
        self.assertEqual([r["code"] for r in diagnosis["candidate_findings"]], ["numerical_variance_sensitive"])
        control = next(r for r in controls if (r["band"], r["operator"], r["estimator"]) ==
                       ("wide", "dma_J192", "clean_ddpm"))
        self.assertEqual(control["passed_fraction_at_both_variances"], 0.)


class PhaseOrderTest(unittest.TestCase):
    def trace_phase(self, phase):
        with tempfile.TemporaryDirectory(prefix="d041_phase_") as temporary:
            directory = Path(temporary)
            log = directory/"commands.jsonl"
            fake_python = directory/"python"
            fake_python.write_text("#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['D041_COMMAND_LOG'], 'a') as output:\n"
                "    output.write(json.dumps(sys.argv[1:])+'\\n')\n")
            fake_python.chmod(0o755)
            env = dict(os.environ, PATH=str(directory)+os.pathsep+os.environ["PATH"],
                D041_COMMAND_LOG=str(log), D041_PREFIX=directory.name, D041_DEVICE="cpu", D041_SMOKE="1", D041_RESUME="0")
            result = subprocess.run(["bash", str(ROOT/"scripts/run_d041_linux.sh"), phase],
                cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return [json.loads(line) for line in log.read_text().splitlines()]

    def test_all_completes_both_noiseless_bands_before_noise(self):
        commands = self.trace_phase("all")
        runners = [command for command in commands if command[0].endswith("run_d041_diagnostic.py")]
        self.assertEqual(len(runners), 18)
        noisy_positions = [index for index, command in enumerate(runners)
            if "--stage" in command and command[command.index("--stage")+1] == "noisy"]
        self.assertEqual(noisy_positions, [14, 15, 16, 17])
        first_band = [command[command.index("--band")+1] for command in runners[:7]]
        second_band = [command[command.index("--band")+1] for command in runners[7:14]]
        self.assertEqual(first_band, ["narrow"]*7)
        self.assertEqual(second_band, ["wide"]*7)
        scopes = [command[command.index("--scope")+1] for command in commands if "--scope" in command]
        self.assertEqual(scopes, ["noiseless", "all"])

    def test_narrow_alias_only_runs_first_noiseless_round(self):
        commands = self.trace_phase("narrow")
        runners = [command for command in commands if command[0].endswith("run_d041_diagnostic.py")]
        self.assertEqual(len(runners), 7)
        self.assertFalse(any("noisy" in command for command in runners))
        self.assertEqual(commands[-1][-1], "narrow-noiseless")

    def test_noisy_phase_rejects_missing_noiseless_controls(self):
        with tempfile.TemporaryDirectory(prefix="d041_missing_") as temporary:
            prefix = Path(temporary).name
            env = dict(os.environ, D041_PREFIX=prefix, D041_DEVICE="cpu", D041_SMOKE="1", D041_RESUME="0",
                MPLCONFIGDIR="/tmp/d041_matplotlib")
            result = subprocess.run(["bash", str(ROOT/"scripts/run_d041_linux.sh"), "noisy"],
                cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Complete both noiseless controls first", result.stderr)
            self.assertFalse(any((ROOT/"runs").glob(prefix+"*")))


if __name__ == "__main__":
    unittest.main()
