"""Compact diagnostics and figures for the Stage 2 unified platform."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


COLORS = {"compatibility": "#4C78A8", "thz_physical": "#E45756"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_raw(frame: pd.DataFrame, summary: dict) -> dict:
    required = {
        "scene_id",
        "physics_level",
        "schedule",
        "schedule_family",
        "pilot_resource_elements",
        "num_configurations",
        "num_switches",
        "effective_output_snr_db",
        "combiner_row_energy_mean",
        "max_mode_rank_residual_fraction",
        "all_finite",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage 2 raw metrics lack columns: {missing}")
    expected = int(summary["experiment"]["expected_raw_metric_rows"])
    declared_actual = int(summary["experiment"]["actual_raw_metric_rows"])
    combinations = {
        (item["physics_level"], item["schedule"])
        for item in summary["condition_summary"]
    }
    grouped = frame.groupby("scene_id", sort=False)
    paired = all(
        {
            (str(row.physics_level), str(row.schedule))
            for row in group.itertuples()
        }
        == combinations
        for _, group in grouped
    )
    duplicate_count = int(
        frame.duplicated(["scene_id", "physics_level", "schedule"]).sum()
    )
    all_finite = bool((frame["all_finite"].astype(int) == 1).all())
    complete = bool(
        len(frame) == expected == declared_actual
        and paired
        and duplicate_count == 0
        and all_finite
    )
    return {
        "complete": complete,
        "expected_rows": expected,
        "actual_rows": int(len(frame)),
        "independent_scenes": int(frame["scene_id"].nunique()),
        "condition_count_per_scene": len(combinations),
        "all_scenes_fully_paired": paired,
        "duplicate_condition_rows": duplicate_count,
        "all_outputs_finite": all_finite,
    }


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (level, schedule, family), group in frame.groupby(
        ["physics_level", "schedule", "schedule_family"], sort=False
    ):
        residual = group["max_mode_rank_residual_fraction"].to_numpy(float)
        output_snr = group["effective_output_snr_db"].to_numpy(float)
        records.append(
            {
                "physics_level": str(level),
                "schedule": str(schedule),
                "schedule_family": str(family),
                "num_independent_scenes": int(len(group)),
                "pilot_resource_elements": int(
                    group["pilot_resource_elements"].iloc[0]
                ),
                "num_configurations": int(group["num_configurations"].iloc[0]),
                "num_switches": int(group["num_switches"].iloc[0]),
                "mean_effective_output_snr_db": float(np.mean(output_snr)),
                "median_effective_output_snr_db": float(np.median(output_snr)),
                "output_snr_q025_db": float(np.quantile(output_snr, 0.025)),
                "output_snr_q975_db": float(np.quantile(output_snr, 0.975)),
                "mean_rank_residual_fraction": float(np.mean(residual)),
                "median_rank_residual_fraction": float(np.median(residual)),
                "rank_residual_q975_fraction": float(np.quantile(residual, 0.975)),
                "mean_combiner_row_energy": float(
                    group["combiner_row_energy_mean"].mean()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _paired_tensor_gap(frame: pd.DataFrame, tensor_family: str) -> dict:
    subset = frame[frame["schedule_family"] == tensor_family]
    pivot = subset.pivot(
        index="scene_id",
        columns="physics_level",
        values="max_mode_rank_residual_fraction",
    )
    required = {"compatibility", "thz_physical"}
    if not required.issubset(pivot.columns):
        raise ValueError("tensor residual comparison lacks one physics level")
    difference = (
        pivot["thz_physical"] - pivot["compatibility"]
    ).to_numpy(float)
    return {
        "num_paired_scenes": int(difference.size),
        "mean_physical_minus_compatibility": float(np.mean(difference)),
        "median_physical_minus_compatibility": float(np.median(difference)),
        "q025_q975": [
            float(np.quantile(difference, 0.025)),
            float(np.quantile(difference, 0.975)),
        ],
        "physical_residual_higher_fraction": float(np.mean(difference > 0.0)),
        "interpretation": (
            "diagnostic only: it measures violation of the compatibility rank-P "
            "assumption, not estimator superiority"
        ),
    }


def _figure_rank_residual(aggregate: pd.DataFrame, output: Path) -> None:
    schedules = list(dict.fromkeys(aggregate["schedule"].tolist()))
    levels = ["compatibility", "thz_physical"]
    x = np.arange(len(schedules), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for level_index, level in enumerate(levels):
        values = []
        for schedule in schedules:
            row = aggregate[
                (aggregate["physics_level"] == level)
                & (aggregate["schedule"] == schedule)
            ]
            values.append(float(row["median_rank_residual_fraction"].iloc[0]))
        ax.bar(
            x + (level_index - 0.5) * width,
            np.maximum(values, 1.0e-16),
            width,
            label=level,
            color=COLORS[level],
        )
    ax.set_yscale("log")
    ax.set_ylabel("Median rank-P unfolding residual")
    ax.set_xticks(x, schedules, rotation=12, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _figure_output_snr(frame: pd.DataFrame, output: Path) -> None:
    labels = []
    values = []
    colors = []
    for (level, schedule), group in frame.groupby(
        ["physics_level", "schedule"], sort=False
    ):
        labels.append(f"{level}\n{schedule}")
        values.append(group["effective_output_snr_db"].to_numpy(float))
        colors.append(COLORS[str(level)])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    plot = ax.boxplot(values, patch_artist=True, showmeans=True)
    for patch, color in zip(plot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_ylabel("Effective post-DMA SNR (dB)")
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _figure_resources(frame: pd.DataFrame, output: Path) -> None:
    schedule_rows = frame.drop_duplicates("schedule").sort_values("schedule")
    labels = schedule_rows["schedule"].tolist()
    x = np.arange(len(labels), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    for axis, column, ylabel in zip(
        axes,
        ["pilot_resource_elements", "num_configurations", "num_switches"],
        ["Pilot RE", "Configurations / symbols", "Switches"],
    ):
        axis.bar(x, schedule_rows[column].to_numpy(float), color="#72B7B2")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_report(path: Path, compact: dict) -> None:
    gate = compact["platform_gate"]
    geometry = compact["geometry_and_ofdm"]
    gap = compact["tensor_structure_gap"]
    lines = [
        "# Stage 2 统一二维平台自动报告",
        "",
        f"- 平台状态：`{gate['status']}`",
        f"- 数据完整：`{compact['data_integrity']['complete']}`",
        f"- 阵元数：{geometry['num_microstrips']} × {geometry['elements_per_microstrip']} = {geometry['num_elements']}",
        f"- 孔径对角线：{1e3 * geometry['aperture_diagonal_m']:.3f} mm",
        f"- Rayleigh 距离：{geometry['rayleigh_distance_m']:.4f} m",
        f"- CP 裕量：{1e9 * geometry['cyclic_prefix_margin_s']:.4f} ns",
        "",
        "## 验收项",
        "",
    ]
    for name, passed in gate["checks"].items():
        lines.append(f"- `{name}`：`{passed}`")
    lines.extend(
        [
            "",
            "## 结构诊断",
            "",
            "- compatibility 层用于验证采集张量在设定 rank-P 模型下可恢复，不代表目标信道。",
            "- thz_physical 层保留球面传播、分子吸收、频变 Lorentzian 响应与有损色散波导。",
            f"- 张量采集下物理层减兼容层残差中位差：{gap['median_physical_minus_compatibility']:.6e}。",
            "- 该差值只检查低秩假设受到何种物理扰动，不比较算法优劣。",
            "",
            "## 下一接口",
            "",
            "平台通过后，方法适配器按同一真值、硬件和能量口径接入；同一采集协议内必须复用完全相同的观测。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_stage2(run_dir: str | Path) -> dict:
    """Validate one Stage 2 run and write compact machine-readable outputs."""

    directory = Path(run_dir).resolve()
    raw_path = directory / "metrics_raw.csv"
    summary = json.loads(
        (directory / "metrics_summary.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (directory / "config_resolved.json").read_text(encoding="utf-8")
    )["config"]
    frame = pd.read_csv(raw_path)
    integrity = _validate_raw(frame, summary)
    aggregate = _aggregate(frame)
    tensor_gap = _paired_tensor_gap(
        frame, str(config["analysis"]["tensor_schedule_family"])
    )
    checks = {str(key): bool(value) for key, value in summary["platform_checks"].items()}
    status = (
        "platform_ready_for_method_adapters"
        if integrity["complete"] and all(checks.values())
        else "platform_validation_failed"
    )
    compact = {
        "analysis_version": "stage2-platform-v1",
        "raw_metrics_sha256": _sha256(raw_path),
        "data_integrity": integrity,
        "platform_gate": {
            "status": status,
            "ready": status == "platform_ready_for_method_adapters",
            "checks": checks,
            "scope": "platform/observation validation only; no estimator ranking yet",
        },
        "geometry_and_ofdm": summary["model_metadata"],
        "resource_fairness": summary["resource_fairness"],
        "schedules": summary["schedules"],
        "tensor_structure_gap": tensor_gap,
        "condition_summary": aggregate.to_dict(orient="records"),
        "next_method_adapter_contract": summary["method_adapter_contract"],
    }

    analysis_dir = directory / "analysis"
    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(analysis_dir / "platform_condition_summary.csv", index=False)
    figure_paths = [
        figures_dir / "01_rank_structure_by_physics.png",
        figures_dir / "02_effective_output_snr.png",
        figures_dir / "03_acquisition_resources.png",
    ]
    _figure_rank_residual(aggregate, figure_paths[0])
    _figure_output_snr(frame, figure_paths[1])
    _figure_resources(frame, figure_paths[2])
    (analysis_dir / "analysis_compact.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_report(analysis_dir / "analysis_report.md", compact)
    manifest = {
        "raw_metrics_sha256": compact["raw_metrics_sha256"],
        "figures": [
            {
                "path": str(path.relative_to(directory)),
                "sha256": _sha256(path),
            }
            for path in figure_paths
        ],
        "plot_contract": "fixed physics levels and acquisition-resource definitions",
    }
    (analysis_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return compact
