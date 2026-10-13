"""One-off byte-preserving D-040 evidence snapshot; never edits source runs."""
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

ROOT = Path.cwd().resolve()
TARGET = ROOT / "research_route/experiment_results/2026-10-13"
assert (ROOT / "pyproject.toml").is_file() and (ROOT / ".git").is_dir()
SELECTED = sorted(p for p in (ROOT / "runs").iterdir() if p.is_dir() and (
    p.name.startswith("zhou_mimo_formal_20261009")
    or p.name == "zhou_mimo_independent_test_20261010_v1"
    or p.name.startswith("zhou_mimo_extended_20261011")
    or p.name.startswith("zhou_carrier_") and "smoke" not in p.name
    or p.name.startswith("zhou_wideband_formal_20261011")
))
assert len(SELECTED) == 41
EXTRA = ["diffusion_data_20261002_v1", "zhou_mimo_probe_eval_20261009_v1",
         "zhou_mimo_extended_smoke_20261011_v1"]
SELECTED += [ROOT / "runs" / name for name in EXTRA]
TARGET.mkdir(parents=True, exist_ok=False)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


files, inventory, external, source_members = [], [], {}, 0
extensions = {".json", ".csv", ".md", ".txt", ".py", ".png", ".pdf", ".patch", ".log"}
for run in SELECTED:
    status = read(run / "status.json")
    assert status["status"] == "complete", run.name
    inputs = [p for p in run.iterdir() if p.is_file() and
              (p.suffix in extensions or p.name == "source_snapshot.zip")]
    if (run / "analysis").exists():
        inputs += [p for p in (run / "analysis").rglob("*")
                   if p.is_file() and p.suffix in extensions]
    for source in sorted(inputs):
        destination = TARGET / run.name / source.relative_to(run)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        shutil.copyfile(source, destination)
        assert sha(destination) == digest and sha(source) == digest
        files.append({"source": source.relative_to(ROOT).as_posix(),
                      "path": destination.relative_to(TARGET).as_posix(),
                      "size_bytes": source.stat().st_size, "sha256": digest})
    count = 0
    if (run / "metrics_raw.csv").is_file():
        with (run / "metrics_raw.csv").open(encoding="utf-8", newline="") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
    inventory.append({"run_id": run.name, "status": status["status"], "rows": count,
                      "role": "supplemental_analysis" if run.name in EXTRA[1:] else "formal_or_source"})
    if (run / "source_snapshot.zip").is_file():
        with zipfile.ZipFile(run / "source_snapshot.zip") as archive:
            for entry in read(run / "code_manifest.json"):
                member = entry["path"].replace("\\", "/")
                assert hashlib.sha256(archive.read(member)).hexdigest() == entry["sha256"]
                source_members += 1
    if (run / "checkpoint/training.pt").is_file() and run.name not in EXTRA[1:]:
        checkpoint = run / "checkpoint/training.pt"
        external[checkpoint] = sha(checkpoint)
    if (run / "dataset.json").is_file() and run.name not in EXTRA[1:]:
        metadata = read(run / "dataset.json")
        if metadata.get("storage") == "existing_npz":
            base = run.parent / PureWindowsPath(metadata["source_directory"]).name
        elif "storage" in metadata:
            base = run
        else:
            base = run / "data"
        for name, digest in metadata["file_sha256"].items():
            path = base / Path(name.replace("\\", "/"))
            relative = path.relative_to(ROOT / "runs")
            if not (TARGET / relative).is_file():
                assert path.is_file(), path
                if path in external:
                    assert external[path] == digest
                external[path] = digest

assert len(inventory) == 44
assert sum(row["rows"] for row in inventory if row["role"] == "formal_or_source") == 194400
assert not any(Path(row["path"]).suffix in {".pt", ".npy", ".npz"} for row in files)
write(TARGET / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
      "original_bytes_preserved": True, "original_source_members_checked": source_members,
      "files": files})
transfer = [{"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
             "sha256": digest} for path, digest in sorted(external.items())]
write(TARGET / "large_artifacts.json", {"purpose": "Separate original runtime inputs; includes parameter JSON",
      "do_not_edit_saved_manifests": True, "total_bytes": sum(row["size_bytes"] for row in transfer),
      "files": transfer})
with (TARGET / "run_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["run_id", "status", "rows", "role"])
    writer.writeheader()
    writer.writerows(inventory)
shutil.copyfile(Path(__file__), TARGET / "archive_results.py")
summary = {"records": len(inventory), "copied_files": len(files),
           "snapshot_MiB": sum(row["size_bytes"] for row in files) / 2**20,
           "external_files": len(transfer), "external_GiB": sum(row["size_bytes"] for row in transfer) / 2**30,
           "formal_metric_rows": 194400, "source_members_checked": source_members}
write(ROOT / "runs/linux_repository_prep_20261013_v1/snapshot_checks.json", summary)
print(json.dumps(summary, ensure_ascii=False))
