"""Paired analysis for the Stage 2-A three-dimensional classical baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thz_dma.analysis.stage2a_domains import (
    DOMAINS, bounded_diagnostics, diagnostic_report_lines, domain_summary, validate_domains,
)


COLORS = {
    "oracle_geometry_ls": "#009E73",
    "grid_omp_3d": "#0072B2",
    "yang_ogols_3d": "#D55E00",
}

SHORT_LABELS = {
    "oracle_geometry_ls": "Oracle",
    "grid_omp_3d": "3-D OMP",
    "yang_ogols_3d": "3-D OG-OLS",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _db(value: float) -> float:
    return float(10.0 * np.log10(max(float(value), np.finfo(float).tiny)))


def _bootstrap_interval(
    values,
    statistic: Callable[[np.ndarray], float],
    *,
    rng: np.random.Generator,
    resamples: int,
) -> list[float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("bootstrap input has no finite values")
    if data.size == 1:
        value = float(statistic(data))
        return [value, value]
    indices = rng.integers(0, data.size, size=(resamples, data.size))
    estimates = np.asarray(
        [statistic(data[index]) for index in indices], dtype=float
    )
    return np.quantile(estimates, [0.025, 0.975]).astype(float).tolist()


def _validate_raw(frame: pd.DataFrame, summary: dict) -> dict:
    required = {
        "scene_id",
        "replicate_seed",
        "sample_index",
        "physics_level",
        "schedule",
        "schedule_family",
        "array_reference_snr_db",
        "effective_output_snr_db",
        "estimator",
        "channel_nmse_linear",
        "channel_nmse_db",
        "all_paths_success",
        "feasible_beam_loss_db",
        "net_rate_bps_hz",
        "estimator_runtime_seconds",
        "refinement_nonincreasing",
        "support_condition_number",
        "pilot_resource_elements",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage 2-A raw metrics lack columns: {missing}")
    key = ["scene_id", "schedule", "array_reference_snr_db", "estimator"]
    duplicates = int(frame.duplicated(key).sum())
    expected = int(summary["experiment"]["expected_raw_metric_rows"])
    schedules = [str(item["name"]) for item in summary["schedules"]]
    snr_values = [float(item) for item in summary["experiment"]["array_reference_snr_db"]]
    estimators = list(summary["method_provenance"])
    expected_conditions = {
        (schedule, snr, estimator)
        for schedule in schedules
        for snr in snr_values
        for estimator in estimators
    }
    paired = all(
        {
            (str(row.schedule), float(row.array_reference_snr_db), str(row.estimator))
            for row in subset.itertuples()
        }
        == expected_conditions
        for _, subset in frame.groupby("scene_id", sort=False)
    )
    numeric_required = [
        "effective_output_snr_db",
        "channel_nmse_linear",
        "channel_nmse_db",
        "all_paths_success",
        "feasible_beam_loss_db",
        "net_rate_bps_hz",
        "estimator_runtime_seconds",
        "support_condition_number",
    ]
    nonfinite = {
        column: int((~np.isfinite(frame[column].to_numpy(float))).sum())
        for column in numeric_required
    }
    observation_spread = (
        frame.groupby(
            ["scene_id", "schedule", "array_reference_snr_db"], sort=False
        )["effective_output_snr_db"]
        .agg(lambda values: float(np.ptp(values.to_numpy(float))))
        .to_numpy(float)
    )
    same_observation = bool(
        observation_spread.size > 0 and np.max(observation_spread) <= 1.0e-12
    )
    complete = bool(
        len(frame) == expected
        and duplicates == 0
        and paired
        and not any(nonfinite.values())
        and same_observation
    )
    return {
        "complete": complete,
        "expected_rows": expected,
        "actual_rows": int(len(frame)),
        "duplicate_key_rows": duplicates,
        "independent_scenes": int(frame["scene_id"].nunique()),
        "all_scenes_fully_paired": bool(paired),
        "same_observation_within_paired_method_block": same_observation,
        "maximum_methodwise_output_snr_spread_db": float(
            np.max(observation_spread) if observation_spread.size else np.inf
        ),
        "nonfinite_required_values": nonfinite,
        "independent_unit": "one generated three-path scene",
        "pairing_key": ["scene_id"],
    }


def _aggregate(
    frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> pd.DataFrame:
    records = []
    columns = [
        "schedule",
        "schedule_family",
        "array_reference_snr_db",
        "estimator",
        "method_source_key",
        "uses_extra_information",
        "training_symbols",
        "pilot_resource_elements",
        "num_switches",
        "switch_guard_symbol_equivalents",
    ]
    for keys, subset in frame.groupby(columns, dropna=False, sort=False):
        record = dict(zip(columns, keys))
        nmse = subset["channel_nmse_linear"].to_numpy(float)
        nmse_ci = _bootstrap_interval(nmse, np.mean, rng=rng, resamples=resamples)
        success = subset["all_paths_success"].to_numpy(float)
        success_ci = _bootstrap_interval(
            success, np.mean, rng=rng, resamples=resamples
        )
        beam = subset["feasible_beam_loss_db"].to_numpy(float)
        beam_ci = _bootstrap_interval(
            beam, np.median, rng=rng, resamples=resamples
        )
        net = subset["net_rate_bps_hz"].to_numpy(float)
        net_ci = _bootstrap_interval(net, np.mean, rng=rng, resamples=resamples)
        runtime = subset["estimator_runtime_seconds"].to_numpy(float)
        runtime_ci = _bootstrap_interval(
            runtime, np.median, rng=rng, resamples=resamples
        )
        record.update(
            {
                "num_independent_scenes": int(len(subset)),
                "mean_channel_nmse_db": _db(np.mean(nmse)),
                "mean_channel_nmse_ci95_low_db": _db(nmse_ci[0]),
                "mean_channel_nmse_ci95_high_db": _db(nmse_ci[1]),
                "median_channel_nmse_db": float(
                    np.median(subset["channel_nmse_db"])
                ),
                "mean_observation_nmse_db": _db(
                    np.mean(subset["observation_nmse_linear"])
                ),
                "all_paths_success_rate": float(np.mean(success)),
                "all_paths_success_ci95_low": float(success_ci[0]),
                "all_paths_success_ci95_high": float(success_ci[1]),
                "median_range_rmse_m": float(np.median(subset["range_rmse_m"])),
                "median_azimuth_rmse_deg": float(
                    np.median(subset["azimuth_rmse_deg"])
                ),
                "median_elevation_rmse_deg": float(
                    np.median(subset["elevation_rmse_deg"])
                ),
                "median_feasible_beam_loss_db": float(np.median(beam)),
                "median_feasible_beam_loss_ci95_low_db": float(beam_ci[0]),
                "median_feasible_beam_loss_ci95_high_db": float(beam_ci[1]),
                "mean_net_rate_bps_hz": float(np.mean(net)),
                "mean_net_rate_ci95_low_bps_hz": float(net_ci[0]),
                "mean_net_rate_ci95_high_bps_hz": float(net_ci[1]),
                "mean_payload_fraction": float(np.mean(subset["payload_fraction"])),
                "mean_data_state_selection_accuracy": float(
                    np.mean(subset["data_state_selection_correct"])
                ),
                "median_oracle_feasible_fraction_of_digital_snr": float(
                    np.median(subset["oracle_feasible_fraction_of_digital_snr"])
                ),
                "median_estimator_runtime_seconds": float(np.median(runtime)),
                "median_estimator_runtime_ci95_low_seconds": float(runtime_ci[0]),
                "median_estimator_runtime_ci95_high_seconds": float(runtime_ci[1]),
                "mean_effective_output_snr_db": float(
                    np.mean(subset["effective_output_snr_db"])
                ),
                "refinement_nonincreasing_rate": float(
                    np.mean(subset["refinement_nonincreasing"])
                ),
                "algorithm_convergence_rate": float(
                    np.mean(subset["algorithm_converged"])
                ),
                "search_boundary_hit_rate": float(
                    np.mean(subset["search_boundary_hit"])
                ),
            }
        )
        for domain in DOMAINS:
            if f"{domain}_linear" not in subset:
                continue
            values = subset[f"{domain}_linear"].dropna().to_numpy(float)
            ci = _bootstrap_interval(values, np.mean, rng=rng, resamples=resamples) if values.size else [None, None]
            record[f"mean_{domain}_db"] = _db(np.mean(values)) if values.size else None
            record[f"mean_{domain}_ci95_low_db"] = _db(ci[0]) if values.size else None
            record[f"mean_{domain}_ci95_high_db"] = _db(ci[1]) if values.size else None
        records.append(record)
    return pd.DataFrame.from_records(records)


def _paired_difference(
    subset: pd.DataFrame, estimator: str, reference: str, column: str
) -> np.ndarray:
    pivot = subset.pivot(index="scene_id", columns="estimator", values=column)
    if estimator not in pivot or reference not in pivot:
        raise ValueError(f"paired comparison lacks {estimator} or {reference}")
    values = (pivot[estimator] - pivot[reference]).dropna().to_numpy(float)
    if values.size == 0:
        raise ValueError("paired comparison has no complete pairs")
    return values


def _effect(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict:
    ci = _bootstrap_interval(values, np.mean, rng=rng, resamples=resamples)
    return {
        "num_paired_scenes": int(values.size),
        "mean": float(np.mean(values)),
        "ci95": [float(ci[0]), float(ci[1])],
        "median": float(np.median(values)),
        "fraction_below_zero": float(np.mean(values < 0.0)),
    }


def _method_contrasts(
    frame: pd.DataFrame,
    config: dict,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> list[dict]:
    reference = str(config["analysis"]["reference_estimator"])
    records = []
    for (schedule, family, snr), subset in frame.groupby(
        ["schedule", "schedule_family", "array_reference_snr_db"], sort=False
    ):
        for estimator in sorted(set(subset["estimator"]) - {reference}):
            runtime_frame = subset.assign(
                runtime_log10=np.log10(
                    np.maximum(
                        subset["estimator_runtime_seconds"].to_numpy(float),
                        np.finfo(float).tiny,
                    )
                )
            )
            runtime = _paired_difference(
                runtime_frame, estimator, reference, "runtime_log10"
            )
            records.append(
                {
                    "schedule": str(schedule),
                    "schedule_family": str(family),
                    "array_reference_snr_db": float(snr),
                    "estimator": str(estimator),
                    "reference_estimator": reference,
                    "comparison_role": (
                        "genie_reference"
                        if estimator == str(config["analysis"]["oracle_estimator"])
                        else "peer_estimator"
                    ),
                    "nmse_delta_db": _effect(
                        _paired_difference(
                            subset, estimator, reference, "channel_nmse_db"
                        ),
                        rng=rng,
                        resamples=resamples,
                    ),
                    "beam_loss_delta_db": _effect(
                        _paired_difference(
                            subset, estimator, reference, "feasible_beam_loss_db"
                        ),
                        rng=rng,
                        resamples=resamples,
                    ),
                    "net_rate_delta_bps_hz": _effect(
                        _paired_difference(
                            subset, estimator, reference, "net_rate_bps_hz"
                        ),
                        rng=rng,
                        resamples=resamples,
                    ),
                    "all_paths_success_delta": _effect(
                        _paired_difference(
                            subset, estimator, reference, "all_paths_success"
                        ),
                        rng=rng,
                        resamples=resamples,
                    ),
                    "runtime_log10_ratio": _effect(
                        runtime, rng=rng, resamples=resamples
                    ),
                    "median_runtime_ratio": float(10.0 ** np.median(runtime)),
                }
            )
    return records


def _primary_screen(
    aggregate: pd.DataFrame,
    contrasts: list[dict],
    frame: pd.DataFrame,
    config: dict,
    summary: dict,
    integrity: dict,
) -> dict:
    analysis = config["analysis"]
    primary_snr = float(analysis["primary_array_reference_snr_db"])
    reference = str(analysis["reference_estimator"])
    strong = str(analysis["strong_estimator"])
    oracle = str(analysis["oracle_estimator"])
    primary = aggregate[np.isclose(aggregate["array_reference_snr_db"], primary_snr)]
    if primary.empty:
        raise ValueError("primary Stage 2-A SNR block is empty")
    rankings = []
    for schedule, subset in primary.groupby("schedule", sort=False):
        regular = subset[subset["uses_extra_information"].astype(int) == 0].sort_values(
            "mean_channel_nmse_db"
        )
        oracle_row = subset[subset["estimator"] == oracle]
        rankings.append(
            {
                "schedule": str(schedule),
                "regular_estimators_by_mean_linear_nmse": [
                    {
                        "estimator": str(row.estimator),
                        "mean_channel_nmse_db": float(row.mean_channel_nmse_db),
                        "all_paths_success_rate": float(row.all_paths_success_rate),
                        "median_feasible_beam_loss_db": float(
                            row.median_feasible_beam_loss_db
                        ),
                        "mean_net_rate_bps_hz": float(row.mean_net_rate_bps_hz),
                        "median_runtime_seconds": float(
                            row.median_estimator_runtime_seconds
                        ),
                    }
                    for row in regular.itertuples()
                ],
                "oracle_reference": (
                    {
                        "mean_channel_nmse_db": float(
                            oracle_row.iloc[0]["mean_channel_nmse_db"]
                        ),
                        "median_feasible_beam_loss_db": float(
                            oracle_row.iloc[0]["median_feasible_beam_loss_db"]
                        ),
                    }
                    if len(oracle_row) == 1
                    else None
                ),
            }
        )
    strong_effects = [
        item
        for item in contrasts
        if item["estimator"] == strong
        and np.isclose(item["array_reference_snr_db"], primary_snr)
    ]
    diagnostic_only = bool(config["experiment"].get("diagnostic_only", False))
    if not bool(config["experiment"]["research_decision_enabled"]):
        performance_status = (
            "not_interpreted_diagnostic"
            if diagnostic_only
            else "not_interpreted_smoke"
        )
    elif strong_effects and all(
        item["nmse_delta_db"]["ci95"][1] < 0.0 for item in strong_effects
    ):
        performance_status = "strong_baseline_clear_gain_in_both_schedules"
    elif strong_effects and any(
        item["nmse_delta_db"]["mean"] < 0.0 for item in strong_effects
    ):
        performance_status = "strong_baseline_partial_or_inconclusive_gain"
    else:
        performance_status = "grid_omp_not_beaten_at_primary_snr"

    oracle_rows = frame[frame["estimator"] == oracle]
    strong_rows = frame[frame["estimator"] == strong]
    pilot_counts = {
        int(item["pilot_resource_elements"]) for item in summary["schedules"]
    }
    checks = {
        "raw_data_complete_and_paired": bool(integrity["complete"]),
        "same_observation_across_estimators": bool(
            integrity["same_observation_within_paired_method_block"]
        ),
        "physical_truth_only": set(frame["physics_level"].astype(str))
        == {"thz_physical"},
        "equal_128_pilot_re_across_schedules": pilot_counts == {128},
        "oracle_geometry_matching_exact": bool(
            len(oracle_rows) > 0 and (oracle_rows["all_paths_success"] == 1).all()
        ),
        "off_grid_refinement_never_increases_residual": bool(
            len(strong_rows) > 0
            and (strong_rows["refinement_nonincreasing"] == 1).all()
        ),
    }
    implementation_ready = all(checks.values())
    if not bool(config["experiment"]["research_decision_enabled"]):
        gate_status = "diagnostic_only" if diagnostic_only else "smoke_only"
    elif implementation_ready:
        gate_status = "baselines_ready_for_literature_adapters"
    else:
        gate_status = "baseline_validation_failed"
    return {
        "baseline_gate": {
            "status": gate_status,
            "implementation_ready": implementation_ready,
            "checks": checks,
            "performance_is_not_a_gate": True,
        },
        "performance_screen": {
            "status": performance_status,
            "scope": "Yang-style OG-OLS versus grid OMP at the primary SNR only",
            "primary_array_reference_snr_db": primary_snr,
            "reference_estimator": reference,
            "strong_estimator": strong,
            "paired_effects": strong_effects,
        },
        "schedule_rankings": rankings,
    }


def _figure_nmse(
    aggregate: pd.DataFrame, labels: dict[str, str], output: Path, metric_label="Legacy CSI NMSE"
) -> None:
    schedules = list(dict.fromkeys(aggregate["schedule"].astype(str)))
    fig, axes = plt.subplots(
        1, len(schedules), figsize=(6.4 * len(schedules), 4.8), squeeze=False
    )
    for axis, schedule in zip(axes[0], schedules):
        subset = aggregate[aggregate["schedule"] == schedule]
        for method in labels:
            method_rows = subset[subset["estimator"] == method].sort_values(
                "array_reference_snr_db"
            )
            x = method_rows["array_reference_snr_db"].to_numpy(float)
            y = method_rows["mean_channel_nmse_db"].to_numpy(float)
            low = y - method_rows["mean_channel_nmse_ci95_low_db"].to_numpy(float)
            high = method_rows["mean_channel_nmse_ci95_high_db"].to_numpy(float) - y
            axis.errorbar(
                x,
                y,
                yerr=np.vstack((low, high)),
                marker="o",
                capsize=3,
                color=COLORS[method],
                linestyle="--" if method == "oracle_geometry_ls" else "-",
                label=SHORT_LABELS.get(method, labels[method]),
            )
        axis.set_title(schedule)
        axis.set_xlabel("Ideal-array reference SNR (dB)")
        axis.set_ylabel(f"Mean {metric_label} (dB; lower is better)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _figure_primary_metrics(
    aggregate: pd.DataFrame, config: dict, labels: dict[str, str], output: Path
) -> None:
    primary_snr = float(config["analysis"]["primary_array_reference_snr_db"])
    subset = aggregate[np.isclose(aggregate["array_reference_snr_db"], primary_snr)]
    schedules = list(dict.fromkeys(subset["schedule"].astype(str)))
    methods = list(labels)
    x = np.arange(len(methods), dtype=float)
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7))
    specs = [
        ("all_paths_success_rate", "All-path success (%)", 100.0),
        ("median_feasible_beam_loss_db", "Median feasible beam loss (dB)", 1.0),
        ("mean_net_rate_bps_hz", "Mean net rate (bit/s/Hz)", 1.0),
    ]
    for schedule_index, schedule in enumerate(schedules):
        schedule_rows = subset[subset["schedule"] == schedule].set_index("estimator")
        offset = (schedule_index - (len(schedules) - 1) / 2.0) * width
        for axis, (column, ylabel, scale) in zip(axes, specs):
            values = [scale * float(schedule_rows.loc[method, column]) for method in methods]
            axis.bar(
                x + offset,
                values,
                width=width,
                alpha=0.82,
                label=schedule,
            )
            axis.set_ylabel(ylabel)
    for axis in axes:
        axis.set_xticks(
            x, [SHORT_LABELS.get(item, labels[item]) for item in methods], rotation=12
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle(f"Stage 2-A primary block: {primary_snr:g} dB")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _figure_runtime_tradeoff(
    aggregate: pd.DataFrame, config: dict, labels: dict[str, str], output: Path, metric_label="Legacy CSI NMSE"
) -> None:
    primary_snr = float(config["analysis"]["primary_array_reference_snr_db"])
    subset = aggregate[np.isclose(aggregate["array_reference_snr_db"], primary_snr)]
    fig, axis = plt.subplots(figsize=(7.5, 5.2))
    markers = ["o", "s", "^"]
    for schedule_index, (schedule, schedule_rows) in enumerate(
        subset.groupby("schedule", sort=False)
    ):
        for row in schedule_rows.itertuples():
            axis.scatter(
                float(row.median_estimator_runtime_seconds),
                float(row.mean_channel_nmse_db),
                marker=markers[schedule_index % len(markers)],
                color=COLORS[str(row.estimator)],
                s=72,
            )
    axis.set_xscale("log")
    axis.set_xlabel("Median batch-1 runtime (s; lower is better)")
    axis.set_ylabel(f"Mean {metric_label} (dB; lower is better)")
    axis.grid(alpha=0.25)
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS[method],
            markeredgecolor=COLORS[method],
            label=SHORT_LABELS.get(method, labels[method]),
        )
        for method in labels
    ]
    schedule_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[index % len(markers)],
            linestyle="none",
            color="black",
            label=schedule,
        )
        for index, schedule in enumerate(
            dict.fromkeys(subset["schedule"].astype(str))
        )
    ]
    axis.legend(handles=method_handles + schedule_handles, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_report(path: Path, compact: dict) -> None:
    screen = compact["primary_screen"]
    gate = screen["baseline_gate"]
    performance = screen["performance_screen"]
    lines = [
        "# Stage 2-A 三维传统参照自动报告",
        "",
        f"- 数据完整：`{compact['data_integrity']['complete']}`",
        f"- 实现门槛：`{gate['status']}`",
        f"- 主 SNR 方法筛查：`{performance['status']}`",
        "- Oracle 使用真实路径几何，只作 genie-aided 结构参照，不参与可部署方法排名，也不宣称是所有估计器的严格统计上界。",
        "- F1 与 F2 分开报告；二者共享 128 导频 RE，但训练符号和切换开销不同。",
        "",
        "## 实现检查",
        "",
    ]
    for name, passed in gate["checks"].items():
        lines.append(f"- `{name}`：`{passed}`")
    lines.extend(diagnostic_report_lines(compact))
    adaptive = compact.get("adaptive_search_diagnostics", {})
    if adaptive.get("status") == "complete":
        lines.extend(
            [
                "## 自适应局部网格诊断",
                "",
                "每条路径的随机粗中心在观测前生成；连续真值在对应中心的半功率邻域内随机偏移，没有人为半格放置。该信息条件只用于条件化实现诊断。",
                f"候选数范围为 {adaptive['candidate_count_min']}–{adaptive['candidate_count_max']}，中位数 {adaptive['candidate_count_median']:.1f}；全部真值位于关联局部窗：`{adaptive['all_truth_paths_inside_associated_windows']}`。",
                f"局部相关宽度在全局搜索边界截断的半宽数为 {adaptive['truncated_profile_halfwidth_count']}；涉及场景数为 {adaptive['scenes_with_truncated_profiles']}。",
                "",
            ]
        )
        for schedule, validation in adaptive.get(
            "dictionary_compute_validation", {}
        ).items():
            lines.append(
                f"- `{schedule}` CUDA—NumPy 模板等价性："
                f"`{validation['status']}`，相对误差 "
                f"{validation['relative_frobenius_error']:.3e}，"
                f"最小 $\\rho^2$={validation['minimum_row_rho2']:.9f}。"
            )
        for row in adaptive["condition_records"]:
            lines.append(
                f"- {row['schedule']} / {row['array_reference_snr_db']:g} dB / "
                f"`{row['estimator']}`：全带线性均值转 dB "
                f"{row['mean_fullband_128_nmse_db']:.3f}，逐样本中位数 "
                f"{row['median_fullband_128_nmse_db']:.3f}，≤−10 dB 比例 "
                f"{100.0 * row['fullband_nmse_le_minus10db_rate']:.1f}%，"
                f"三径成功率 {100.0 * row['all_paths_success_rate']:.1f}%。"
            )
        lines.append("")
    lines.extend(["", "## 主 SNR 兼容字段汇总（旧 16 点口径，不用于宽带恢复结论）", ""])
    for block in screen["schedule_rankings"]:
        lines.append(f"### {block['schedule']}")
        lines.append("")
        for index, row in enumerate(
            block["regular_estimators_by_mean_linear_nmse"], start=1
        ):
            lines.append(
                f"{index}. `{row['estimator']}`：NMSE {row['mean_channel_nmse_db']:.3f} dB，"
                f"全路径成功率 {100.0 * row['all_paths_success_rate']:.1f}%，"
                f"波束损失 {row['median_feasible_beam_loss_db']:.3f} dB，"
                f"净谱效率 {row['mean_net_rate_bps_hz']:.3f} bit/s/Hz。"
            )
        oracle = block["oracle_reference"]
        if oracle is not None:
            lines.append(
                f"- Oracle 参考：NMSE {oracle['mean_channel_nmse_db']:.3f} dB，"
                f"波束损失 {oracle['median_feasible_beam_loss_db']:.3f} dB。"
            )
        lines.append("")
    boundary = (
        "本轮只检验已知逐径粗中心后的局部候选采样与捕获域。即使条件诊断通过，也不能据此启动正式比较；未知全局几何下的粗获取仍需单独验证。"
        if adaptive.get("status") == "complete"
        else "实现门槛只检查代码链路。固定扇区仍需验证基本有效的几何与宽带恢复；不能凭实现门槛通过启动正式比较或新文献方法。"
    )
    lines.extend(["## 判读边界", "", boundary, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _adaptive_search_diagnostics(frame: pd.DataFrame, summary: dict) -> dict:
    search = summary.get("search", {})
    if search.get("mode") != "beam_center_adaptive":
        return {"status": "not_applicable"}
    required = {
        "adaptive_candidate_count",
        "adaptive_truth_window_coverage",
        "adaptive_estimate_outside_local_union",
        "fullband_128_nmse_linear",
        "fullband_128_nmse_db",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"status": "invalid", "missing_columns": missing}
    scene_counts = frame.groupby("scene_id", sort=False)[
        "adaptive_candidate_count"
    ].nunique()
    truncated_by_scene = []
    for scene in search.get("scene_records", []):
        count = sum(
            int(flag)
            for path_profiles in scene.get("schedule_profiles", {}).values()
            for path_profile in path_profiles
            for dimension_sides in path_profile.get("crossing_truncated", [])
            for flag in dimension_sides
        )
        truncated_by_scene.append(count)
    records = []
    for (schedule, snr, estimator), subset in frame.groupby(
        ["schedule", "array_reference_snr_db", "estimator"], sort=False
    ):
        fullband = subset["fullband_128_nmse_linear"].to_numpy(float)
        records.append(
            {
                "schedule": str(schedule),
                "array_reference_snr_db": float(snr),
                "estimator": str(estimator),
                "scenes": int(len(subset)),
                "mean_fullband_128_nmse_db": _db(float(np.mean(fullband))),
                "median_fullband_128_nmse_db": float(
                    np.median(subset["fullband_128_nmse_db"])
                ),
                "fullband_nmse_le_minus10db_rate": float(
                    np.mean(subset["fullband_128_nmse_db"] <= -10.0)
                ),
                "all_paths_success_rate": float(
                    np.mean(subset["all_paths_success"])
                ),
                "estimate_outside_local_union_rate": float(
                    np.mean(subset["adaptive_estimate_outside_local_union"])
                ),
            }
        )
    return {
        "status": "complete",
        "information_condition": (
            "one known pre-observation coarse center per path; conditional diagnostic, "
            "not a deployable coarse-center acquisition result"
        ),
        "artificial_half_grid_offset_used": False,
        "truth_offsets": "continuous random offsets around coarse centers",
        "all_truth_paths_inside_associated_windows": bool(
            (frame["adaptive_truth_window_coverage"] == 1).all()
            and search.get("all_truth_paths_inside_associated_windows", False)
        ),
        "candidate_count_constant_within_scene": bool((scene_counts == 1).all()),
        "candidate_count_min": int(frame["adaptive_candidate_count"].min()),
        "candidate_count_median": float(
            frame.groupby("scene_id", sort=False)["adaptive_candidate_count"]
            .first()
            .median()
        ),
        "candidate_count_max": int(frame["adaptive_candidate_count"].max()),
        "truncated_profile_halfwidth_count": int(sum(truncated_by_scene)),
        "scenes_with_truncated_profiles": int(
            sum(count > 0 for count in truncated_by_scene)
        ),
        "dictionary_compute": search.get("dictionary_compute", {}),
        "dictionary_compute_validation": search.get(
            "dictionary_compute_validation", {}
        ),
        "descriptive_fullband_screen_db": -10.0,
        "condition_records": records,
    }


def analyze_stage2a(run_dir: str | Path) -> dict:
    """Validate, summarize, bootstrap, and plot one Stage 2-A run."""

    directory = Path(run_dir).resolve()
    raw_path = directory / "metrics_raw.csv"
    frame = pd.read_csv(raw_path)
    summary = json.loads(
        (directory / "metrics_summary.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (directory / "config_resolved.json").read_text(encoding="utf-8")
    )["config"]
    integrity = _validate_raw(frame, summary)
    domains_available = validate_domains(frame, summary)
    rng = np.random.default_rng(int(config["experiment"]["bootstrap_seed"]))
    resamples = int(config["experiment"]["bootstrap_resamples"])
    aggregate = _aggregate(frame, rng=rng, resamples=resamples)
    contrasts = _method_contrasts(
        frame, config, rng=rng, resamples=resamples
    )
    primary = _primary_screen(
        aggregate, contrasts, frame, config, summary, integrity
    )
    compact = {
        "analysis_version": "stage2a-domains-snr-v2",
        "domain_metrics_available": domains_available,
        "domain_summary": domain_summary(frame),
        "bounded_diagnostics": bounded_diagnostics(frame, config, directory, integrity),
        "adaptive_search_diagnostics": _adaptive_search_diagnostics(frame, summary),
        "raw_metrics_sha256": _sha256(raw_path),
        "data_integrity": integrity,
        "comparison_contract": {
            "truth": "physical spherical THz three-path channel only",
            "primary_axis": "estimator under identical schedule observations",
            "schedules_compared_separately": True,
            "primary_metric": "fullband_128_nmse; pilot and heldout domains reported separately",
            "legacy_sections": "primary_screen, all_method_contrasts and original summary columns retain legacy channel_nmse semantics; not broadband conclusions",
            "secondary_metrics": [
                "three-path localization success",
                "feasible DMA codebook beam loss",
                "net spectral efficiency with schedule overhead",
                "batch-1 CPU runtime",
            ],
            "oracle_is_genie_reference_not_competitor": True,
            "paper_exact_reproduction": False,
        },
        "method_provenance": summary["method_provenance"],
        "primary_screen": primary,
        "all_method_contrasts": contrasts,
        "bootstrap": {
            "seed": int(config["experiment"]["bootstrap_seed"]),
            "resamples": resamples,
            "interval": "percentile 95%",
            "resampling_unit": "independent scene within schedule/SNR block",
        },
        "search": summary["search"],
        "resource_fairness": summary["resource_fairness"],
    }

    output = directory / "analysis"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output / "summary_table.csv", index=False)
    pd.DataFrame.from_records(compact["domain_summary"]).to_csv(output / "domain_summary_table.csv", index=False)
    domain_effects = []
    if domains_available:
        for domain in DOMAINS:
            valid = frame.dropna(subset=[f"{domain}_db"])
            for (schedule, snr), block in valid.groupby(["schedule", "array_reference_snr_db"], sort=False):
                for method in sorted(set(block.estimator) - {config["analysis"]["reference_estimator"]}):
                    effect = _effect(_paired_difference(block, method, config["analysis"]["reference_estimator"], f"{domain}_db"), rng=rng, resamples=resamples)
                    domain_effects.append({"domain": domain, "schedule": str(schedule), "snr_db": float(snr), "estimator": method, "reference_estimator": config["analysis"]["reference_estimator"], **effect})
    compact["domain_paired_method_effects"] = domain_effects
    pd.DataFrame.from_records(domain_effects).to_csv(output / "domain_paired_method_effects.csv", index=False)
    flattened = []
    for item in contrasts:
        flattened.append(
            {
                "schedule": item["schedule"],
                "array_reference_snr_db": item["array_reference_snr_db"],
                "estimator": item["estimator"],
                "reference_estimator": item["reference_estimator"],
                "comparison_role": item["comparison_role"],
                "nmse_delta_db_mean": item["nmse_delta_db"]["mean"],
                "nmse_delta_db_ci95_low": item["nmse_delta_db"]["ci95"][0],
                "nmse_delta_db_ci95_high": item["nmse_delta_db"]["ci95"][1],
                "beam_loss_delta_db_mean": item["beam_loss_delta_db"]["mean"],
                "net_rate_delta_bps_hz_mean": item["net_rate_delta_bps_hz"]["mean"],
                "all_paths_success_delta": item["all_paths_success_delta"]["mean"],
                "median_runtime_ratio": item["median_runtime_ratio"],
            }
        )
    pd.DataFrame.from_records(flattened).to_csv(
        output / "paired_method_effects.csv", index=False
    )
    labels = {
        method: summary["method_provenance"][method]["label"]
        for method in config["experiment"]["estimators"]
    }
    figure_paths = [
        figures / "01_nmse_by_snr_and_schedule.png",
        figures / "02_primary_communication_metrics.png",
        figures / "03_nmse_runtime_tradeoff.png",
    ]
    # Existing plot functions receive an explicit fullband view; raw legacy fields stay unchanged.
    plot_aggregate = aggregate
    if domains_available:
        plot_frame = frame.assign(channel_nmse_linear=frame.fullband_128_nmse_linear, channel_nmse_db=frame.fullband_128_nmse_db)
        plot_aggregate = _aggregate(plot_frame, rng=np.random.default_rng(int(config["experiment"]["bootstrap_seed"])), resamples=resamples)
    metric_label = "fullband 128-tone CSI NMSE" if domains_available else "legacy 16-tone CSI NMSE"
    _figure_nmse(plot_aggregate, labels, figure_paths[0], metric_label)
    _figure_primary_metrics(aggregate, config, labels, figure_paths[1])
    _figure_runtime_tradeoff(plot_aggregate, config, labels, figure_paths[2], metric_label)
    compact["nmse_figure_domain"] = "fullband_128_nmse" if domains_available else "legacy 16-tone channel_nmse"
    (output / "analysis_compact.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_report(output / "analysis_report.md", compact)
    manifest = {
        "raw_metrics_sha256": compact["raw_metrics_sha256"],
        "figures": [
            {"path": str(path.relative_to(directory)), "sha256": _sha256(path)}
            for path in figure_paths
        ],
        "plot_contract": "fixed Stage 2-A schedule, estimator and primary-SNR definitions",
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return compact
