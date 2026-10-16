"""Read-only verification of this delivery's CPU smoke chain and preserved inputs."""
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from _run_support import hash_file, read_toml, write_json
from analyze_d041_budget_sweep import verify_report
from run_d041_budget_sweep import plans

PREFIX = "d041_budget_smoke_20261015_v1"
FIELDS = ["nmse_linear", "nmse_db", "observation_residual_relative", "prediction_energy_ratio",
    "normalized_correlation", "row_space_nmse_linear", "null_space_nmse_linear"]


def read(path):
    return json.loads(path.read_text())


def main():
    protected = read(HERE/"protected_files.json")
    for name, digest in protected.items():
        assert hash_file(ROOT/name) == digest, name
    totals = {"metric_rows": 0, "trajectory_rows": 0, "cases": 0, "snapshot_members": 0}
    runs = []
    maximum_gap = 0.
    for split in ["validation", "test"]:
        for budget in [0, 128, 512, 1024, 2048, 4096, 8192]:
            tag = "references" if budget == 0 else f"re{budget}"
            directory = ROOT/"runs"/f"{PREFIX}_{tag}_{split}"
            assert read(directory/"status.json")["status"] == "complete"
            config = read(directory/"config_resolved.json")["config"]
            assert config["job"]["smoke"] and config["job"]["split"] == split
            assert config["channel"]["include_absorption"] is False
            operators = read(directory/"operators.json")
            assert [op["label"] for op in operators] == [p[0] for p in plans(config)]
            for op in operators:
                if budget:
                    assert op["pilot_re"] == budget == op["total_pilot_energy"]
                    assert op["complex_observations"] == budget*8
                    assert op["complex_unknowns"] == 128*8*160
                if op["state_mode"] == "repeat":
                    assert op["switches"] == 0 and op["unique_dma_configurations"] == 1
            with zipfile.ZipFile(directory/"source_snapshot.zip") as snapshot:
                manifest = read(directory/"code_manifest.json")
                assert len(snapshot.namelist()) == len(manifest)
                for item in manifest:
                    assert hash_file(ROOT/item["path"]) == item["sha256"]
                    assert snapshot.read(item["path"]) == (ROOT/item["path"]).read_bytes()
                totals["snapshot_members"] += len(manifest)
            with (directory/"metrics_raw.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            with (directory/"trajectories.csv").open() as handle:
                trace = list(csv.DictReader(handle))
            expected_cases = len(operators)*2
            assert len(rows) == expected_cases*3
            assert len(trace) == expected_cases*(config["diffusion"]["steps"]+1)
            indexed = {(row["operator"], row["condition"], row["scene_id"], row["estimator"]): row for row in rows}
            assert len(indexed) == len(rows)
            cases = sorted((directory/"checkpoint").glob("case_*.json"))
            assert len(cases) == expected_cases
            tuning = read(directory/"tuning.json")
            for path in cases:
                case = read(path)
                assert [step["t"] for step in case["trajectory"]] == list(range(config["diffusion"]["steps"], -1, -1))
                for row in case["rows"]:
                    saved = indexed[tuple(row[k] for k in ["operator", "condition", "scene_id", "estimator"])]
                    assert row["physical_element_noise_complex_variance"] == 0
                    assert row["guidance"] == tuning[f"{row['operator']}:{row['condition']}"]["selected"]
                    for field in FIELDS:
                        gap = abs(float(saved[field])-row[field])
                        maximum_gap = max(maximum_gap, gap)
                        np.testing.assert_allclose(float(saved[field]), row[field], rtol=1e-12, atol=1e-15)
                    if row["estimator"] == "clean_ddpm":
                        for field in FIELDS:
                            np.testing.assert_allclose(case["trajectory"][-1][field], row[field], rtol=1e-12, atol=1e-15)
            if split == "test":
                prior = ROOT/"runs"/config["job"]["tuning_run_id"]
                assert read(directory/"validation_selection.json") == read(prior/"selection.json")
                assert read(directory/"tuning_source.json")["sha256"] == hash_file(prior/"tuning.json")
                assert tuning == read(prior/"tuning.json")
            totals["metric_rows"] += len(rows)
            totals["trajectory_rows"] += len(trace)
            totals["cases"] += len(cases)
            runs.append({"run_id": directory.name, "operators": len(operators), "metric_rows": len(rows), "trajectory_rows": len(trace)})
    validation = ROOT/"results/d041_budget_sweep"/f"{PREFIX}_validation"
    test = ROOT/"results/d041_budget_sweep"/f"{PREFIX}_test"
    for directory in [validation, test]:
        verify_report(directory)
    selection = read(validation/"selection.json")
    assert selection["workpoint"] is None and selection["smoke_only"]
    frozen_at = datetime.fromisoformat(selection["created_utc"])
    for row in runs:
        if row["run_id"].endswith("_test"):
            environment = (ROOT/"runs"/row["run_id"]/"environment.txt").read_text()
            started = next(line.split(": ", 1)[1] for line in environment.splitlines() if line.startswith("timestamp_utc: "))
            assert datetime.fromisoformat(started) > frozen_at
    formal = copy.deepcopy(config)
    formal["job"]["smoke"] = False
    formal["configuration_sweep"] = {**read_toml(ROOT/"configs/d041_budget_sweep.toml"), "experiment": "budget"}
    operators = main_groups = 0
    for budget in [0, 128, 512, 1024, 2048, 4096, 8192]:
        formal["configuration_sweep"]["budget_re"] = budget
        generated = list(plans(formal))
        operators += len(generated)
        main_groups += sum(p[3]["is_main"] for p in generated)
    assert (operators, main_groups) == (80, 24)
    assert (totals["metric_rows"], totals["trajectory_rows"], totals["cases"]) == (420, 1260, 140)
    payload = {"passed": True, "scope": "CPU smoke connectivity/archive/freeze checks; not formal performance",
        "protected_preexisting_files": len(protected), "formal_operators_per_split": operators,
        "formal_main_settings": main_groups, "runs": runs, **totals,
        "maximum_metric_csv_case_absolute_gap": maximum_gap, "validation_frozen_before_test": True}
    write_json(HERE/"verification.json", payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "runs"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
