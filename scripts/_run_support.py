"""Shared command-line and run-archive support for simulation scripts."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thz_dma import __version__  # noqa: E402


def parse_run_arguments(
    default_config: str, description: str | None = None
) -> tuple[Path, dict, str | None]:
    """Parse the common runner arguments and load the selected TOML file."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / default_config
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    return config_path, read_toml(config_path), args.run_id


def read_toml(path: Path) -> dict:
    """Load a TOML document with the Python 3.10 compatibility fallback."""

    with path.open("rb") as handle:
        return tomllib.load(handle)


def require_unit_pilot_energy(value: object, stage: str) -> None:
    """Keep resource comparisons on the declared one-unit-per-RE convention."""

    if not math.isclose(float(value), 1.0, rel_tol=1.0e-5, abs_tol=1.0e-8):
        raise ValueError(f"{stage} currently requires pilot_energy_per_re = 1.0")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def git_state(project_root: Path = PROJECT_ROOT) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable: project directory is not a Git worktree\n"
    return f"commit: {commit}\ndirty: {bool(status.strip())}\n"


def package_version(name: str) -> str:
    """Read installed metadata without importing the package or its dependencies."""

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def write_code_manifest(path: Path, files: Iterable[Path]) -> None:
    records = []
    for source in sorted({item.resolve() for item in files}):
        records.append(
            {
                "path": str(source.relative_to(PROJECT_ROOT)),
                "size_bytes": source.stat().st_size,
                "sha256": hash_file(source),
            }
        )
    write_json(path, records)


def write_raw_metrics(
    path: Path, rows: list[dict], fieldnames: list[str] | None = None
) -> None:
    if not rows:
        raise ValueError("simulation produced no metric rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _environment(extra: dict[str, object]) -> dict[str, object]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix is None and (Path(sys.prefix) / "conda-meta").is_dir():
        conda_prefix = sys.prefix
    conda_name = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_name is None and conda_prefix is not None:
        conda_name = Path(conda_prefix).name
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "conda_default_env": conda_name or "not detected",
        "conda_prefix": conda_prefix or "not detected",
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "pandas": package_version("pandas"),
        "matplotlib": package_version("matplotlib"),
        "package_version": __version__,
        "cwd": os.getcwd(),
        **extra,
    }


@dataclass(frozen=True)
class RunArchive:
    """Paths and small writes shared by every immutable experiment run."""

    run_id: str
    directory: Path

    @property
    def status_path(self) -> Path:
        return self.directory / "status.json"

    def write_json(self, name: str, payload: object) -> None:
        write_json(self.directory / name, payload)

    def write_status(self, state: str, **extra: object) -> None:
        write_json(
            self.status_path,
            {
                "status": state,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **extra,
            },
        )

    def write_metrics(
        self, rows: list[dict], fieldnames: list[str] | None = None
    ) -> None:
        write_raw_metrics(self.directory / "metrics_raw.csv", rows, fieldnames)

    def write_result(self, group: str, suffix: str, payload: object) -> Path:
        directory = PROJECT_ROOT / "results" / group
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}_{suffix}.json"
        write_json(path, payload)
        return path


def create_run_archive(
    prefix: str,
    requested_run_id: str | None,
    *,
    config_path: Path,
    config: dict,
    seeds: dict,
    script_path: str,
    environment: dict[str, object],
    checkpoint_note: str,
    extra_manifest_files: Iterable[Path] = (),
    figures_note: str | None = None,
) -> RunArchive:
    """Create the standard immutable run skeleton before computation starts."""

    run_id = requested_run_id or datetime.now(timezone.utc).strftime(
        f"{prefix}_%Y%m%dT%H%M%SZ"
    )
    archive = RunArchive(run_id, PROJECT_ROOT / "runs" / run_id)
    archive.directory.mkdir(parents=True, exist_ok=False)
    archive.write_status("running", run_id=run_id)

    absorption_path = PROJECT_ROOT / config["atmosphere"]["table_path"]
    archive.write_json(
        "config_resolved.json",
        {
            "config": config,
            "config_path": str(config_path),
            "config_sha256": hash_file(config_path),
            "absorption_table_sha256": hash_file(absorption_path),
        },
    )
    archive.write_json("seeds.json", seeds)
    (archive.directory / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in _environment(environment).items())
        + "\n",
        encoding="utf-8",
    )
    (archive.directory / "git_commit.txt").write_text(
        git_state(), encoding="utf-8"
    )

    # Hash the runner, shared archive code, and model sources at launch time.
    manifest_files = [
        PROJECT_ROOT / "pyproject.toml",
        config_path,
        absorption_path,
        Path(script_path).resolve(),
        Path(__file__).resolve(),
        *extra_manifest_files,
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
        *sorted((PROJECT_ROOT / "tests").glob("test_*.py")),
    ]
    write_code_manifest(archive.directory / "code_manifest.json", manifest_files)

    checkpoint_dir = archive.directory / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "README.txt").write_text(checkpoint_note, encoding="utf-8")
    if figures_note is not None:
        figures_dir = archive.directory / "figures"
        figures_dir.mkdir()
        (figures_dir / "README.txt").write_text(figures_note, encoding="utf-8")
    return archive
