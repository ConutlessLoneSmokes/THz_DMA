"""Compact statistical analysis and fixed plots for Stage 1-B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


COLORS = ["#0072B2", "#D55E00", "#009E73"]
MARKERS = ["o", "s", "^"]
LINESTYLES = ["-", "--", "-."]


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
        raise ValueError("bootstrap input has no finite observations")
    if data.size == 1:
        value = float(statistic(data))
        return [value, value]
    indices = rng.integers(0, data.size, size=(resamples, data.size))
    estimates = np.asarray(
        [statistic(data[index]) for index in indices], dtype=float
    )
    return np.quantile(estimates, [0.025, 0.975]).astype(float).tolist()


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _median(values: np.ndarray) -> float:
    return float(np.median(values))


def _method_inventory(frame: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    inventory = (
        frame[["design", "design_family", "num_configurations"]]
        .drop_duplicates()
        .sort_values(["num_configurations", "design_family"])
    )
    selectors = {
        "physical_j1": ("physical_frequency_response", 1),
        "physical_j2": ("physical_time_scan", 2),
        "flat_j4": ("flat_timescan", 4),
    }
    names: dict[str, str] = {}
    for key, (family, count) in selectors.items():
        subset = inventory[
            (inventory["design_family"] == family)
            & (inventory["num_configurations"] == count)
        ]
        if len(subset) != 1:
            raise ValueError(f"cannot resolve unique Stage 1-B method {key}")
        names[key] = str(subset.iloc[0]["design"])
    order = [names["physical_j1"], names["physical_j2"], names["flat_j4"]]
    labels = {
        names["physical_j1"]: "Physical frequency response, J=1",
        names["physical_j2"]: "Physical time scan, J=2",
        names["flat_j4"]: "Flat time scan, J=4",
    }
    return order, labels


def _validate_raw(frame: pd.DataFrame, summary: dict) -> dict:
    required = {
        "scenario",
        "replicate_seed",
        "sample_index",
        "snr_condition",
        "noise_mode",
        "target_snr_db",
        "design",
        "design_family",
        "channel_nmse_linear",
        "channel_nmse_db",
        "both_paths_success",
        "feasible_beam_loss_db",
        "feasible_gross_rate_bps_hz",
        "training_symbols",
        "switch_guard_symbol_equivalents",
        "num_configurations",
        "search_boundary_hit",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"metrics_raw.csv is missing columns: {missing}")
    key = [
        "scenario",
        "replicate_seed",
        "sample_index",
        "snr_condition",
        "design",
    ]
    duplicates = int(frame.duplicated(key).sum())
    expected = int(summary["experiment"]["expected_raw_metric_rows"])
    numeric_required = [
        "channel_nmse_linear",
        "channel_nmse_db",
        "both_paths_success",
        "feasible_beam_loss_db",
        "feasible_gross_rate_bps_hz",
        "training_symbols",
        "switch_guard_symbol_equivalents",
    ]
    nonfinite = {
        column: int((~np.isfinite(frame[column].to_numpy(dtype=float))).sum())
        for column in numeric_required
    }
    complete = len(frame) == expected and duplicates == 0 and not any(
        nonfinite.values()
    )
    return {
        "complete": bool(complete),
        "expected_rows": expected,
        "actual_rows": int(len(frame)),
        "duplicate_key_rows": duplicates,
        "nonfinite_required_values": nonfinite,
        "independent_unit": "one generated two-path scene",
    }


def _aggregate(
    frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> pd.DataFrame:
    records = []
    group_columns = [
        "scenario",
        "snr_condition",
        "noise_mode",
        "target_snr_db",
        "design",
        "design_family",
        "num_configurations",
        "training_symbols",
        "pilot_resource_elements",
        "num_switches",
        "switch_guard_symbol_equivalents",
    ]
    for keys, subset in frame.groupby(group_columns, dropna=False, sort=False):
        record = dict(zip(group_columns, keys))
        nmse = subset["channel_nmse_linear"].to_numpy(dtype=float)
        success = subset["both_paths_success"].to_numpy(dtype=float)
        beam_loss = subset["feasible_beam_loss_db"].to_numpy(dtype=float)
        gross_rate = subset["feasible_gross_rate_bps_hz"].to_numpy(dtype=float)
        nmse_ci = _bootstrap_interval(
            nmse, _mean, rng=rng, resamples=resamples
        )
        success_ci = _bootstrap_interval(
            success, _mean, rng=rng, resamples=resamples
        )
        beam_ci = _bootstrap_interval(
            beam_loss, _median, rng=rng, resamples=resamples
        )
        rate_ci = _bootstrap_interval(
            gross_rate, _mean, rng=rng, resamples=resamples
        )
        record.update(
            {
                "num_independent_scenes": int(len(subset)),
                "mean_channel_nmse_db": _db(np.mean(nmse)),
                "mean_channel_nmse_ci95_low_db": _db(nmse_ci[0]),
                "mean_channel_nmse_ci95_high_db": _db(nmse_ci[1]),
                "both_paths_success_rate": float(np.mean(success)),
                "both_paths_success_ci95_low": float(success_ci[0]),
                "both_paths_success_ci95_high": float(success_ci[1]),
                "median_feasible_beam_loss_db": float(np.median(beam_loss)),
                "median_feasible_beam_loss_ci95_low_db": float(beam_ci[0]),
                "median_feasible_beam_loss_ci95_high_db": float(beam_ci[1]),
                "mean_feasible_gross_rate_bps_hz": float(np.mean(gross_rate)),
                "mean_feasible_gross_rate_ci95_low_bps_hz": float(rate_ci[0]),
                "mean_feasible_gross_rate_ci95_high_bps_hz": float(rate_ci[1]),
                "search_boundary_hit_rate": float(
                    subset["search_boundary_hit"].mean()
                ),
                "mean_effective_output_snr_db": float(
                    subset["effective_output_snr_db"].mean()
                )
                if subset["effective_output_snr_db"].notna().any()
                else np.nan,
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _paired_values(
    subset: pd.DataFrame,
    first: str,
    second: str,
    value_column: str,
) -> np.ndarray:
    pivot = subset.pivot(
        index=["replicate_seed", "sample_index"],
        columns="design",
        values=value_column,
    )
    if first not in pivot or second not in pivot:
        raise ValueError(f"paired comparison lacks {first} or {second}")
    values = (pivot[first] - pivot[second]).dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("paired comparison has no complete scene pairs")
    return values


def _contrast(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict:
    interval = _bootstrap_interval(
        values, _mean, rng=rng, resamples=resamples
    )
    return {
        "num_paired_scenes": int(values.size),
        "mean": float(np.mean(values)),
        "ci95": [float(interval[0]), float(interval[1])],
        "median": float(np.median(values)),
        "fraction_below_zero": float(np.mean(values < 0.0)),
    }


def _net_rate(frame: pd.DataFrame, coherence_symbols: float) -> np.ndarray:
    overhead = (
        frame["training_symbols"].to_numpy(dtype=float)
        + frame["switch_guard_symbol_equivalents"].to_numpy(dtype=float)
    )
    data_fraction = np.clip(1.0 - overhead / coherence_symbols, 0.0, 1.0)
    return frame["feasible_gross_rate_bps_hz"].to_numpy(dtype=float) * data_fraction


def _primary_gate(
    frame: pd.DataFrame,
    config: dict,
    method_names: dict[str, str],
    *,
    rng: np.random.Generator,
    resamples: int,
) -> tuple[dict, dict]:
    decision = config["decision"]
    scenario = str(decision["primary_scenario"])
    condition = str(decision["primary_noise_condition"])
    primary = frame[
        (frame["scenario"] == scenario)
        & (frame["snr_condition"] == condition)
    ].copy()
    if primary.empty:
        raise ValueError("the preregistered primary block is absent")

    j1 = method_names["physical_j1"]
    j2 = method_names["physical_j2"]
    flat = method_names["flat_j4"]
    nmse_j1_j2 = _paired_values(primary, j1, j2, "channel_nmse_db")
    nmse_j1_flat = _paired_values(primary, j1, flat, "channel_nmse_db")
    coherence = float(decision["coherence_symbols_for_gate"])
    primary["net_rate_bps_hz"] = _net_rate(primary, coherence)
    rate_j1_j2 = _paired_values(primary, j1, j2, "net_rate_bps_hz")
    rate_j1_flat = _paired_values(primary, j1, flat, "net_rate_bps_hz")

    j1_rows = primary[primary["design"] == j1]
    success_values = j1_rows["both_paths_success"].to_numpy(dtype=float)
    beam_values = j1_rows["feasible_beam_loss_db"].to_numpy(dtype=float)
    success_ci = _bootstrap_interval(
        success_values, _mean, rng=rng, resamples=resamples
    )
    beam_ci = _bootstrap_interval(
        beam_values, _median, rng=rng, resamples=resamples
    )
    nmse_contrast = _contrast(
        nmse_j1_j2, rng=rng, resamples=resamples
    )
    rate_contrast = _contrast(
        rate_j1_j2, rng=rng, resamples=resamples
    )
    margin = float(decision["nmse_noninferiority_margin_db"])
    minimum_success = float(decision["minimum_both_path_success_rate"])
    maximum_beam_loss = float(decision["maximum_median_beam_loss_db"])

    point_criteria = {
        "nmse_noninferior_to_physical_j2": bool(
            nmse_contrast["mean"] <= margin
        ),
        "both_path_success_adequate": bool(
            np.mean(success_values) >= minimum_success
        ),
        "feasible_beam_loss_adequate": bool(
            np.median(beam_values) <= maximum_beam_loss
        ),
        "net_rate_not_lower_than_physical_j2": bool(
            rate_contrast["mean"] >= 0.0
        ),
    }
    confidence_criteria = {
        "nmse_noninferior_to_physical_j2": bool(
            nmse_contrast["ci95"][1] <= margin
        ),
        "both_path_success_adequate": bool(success_ci[0] >= minimum_success),
        "feasible_beam_loss_adequate": bool(beam_ci[1] <= maximum_beam_loss),
        "net_rate_not_lower_than_physical_j2": bool(
            rate_contrast["ci95"][0] >= 0.0
        ),
    }
    if all(confidence_criteria.values()):
        status = "confirmed_pass"
    elif all(point_criteria.values()):
        status = "provisional_pass"
    else:
        status = "stop_or_redesign"

    gate = {
        "status": status,
        "interpretation": {
            "confirmed_pass": (
                "all four continuation criteria pass with bootstrap uncertainty"
            ),
            "provisional_pass": (
                "all point criteria pass, but the current sample does not close "
                "every uncertainty bound"
            ),
            "stop_or_redesign": (
                "at least one preregistered point criterion fails; do not add a "
                "learning model before diagnosing that failure"
            ),
        }[status],
        "primary_scenario": scenario,
        "primary_noise_condition": condition,
        "coherence_symbols_for_gate": coherence,
        "thresholds": {
            "nmse_noninferiority_margin_db": margin,
            "minimum_both_path_success_rate": minimum_success,
            "maximum_median_beam_loss_db": maximum_beam_loss,
        },
        "physical_j1": {
            "both_paths_success_rate": float(np.mean(success_values)),
            "both_paths_success_ci95": [
                float(success_ci[0]),
                float(success_ci[1]),
            ],
            "median_feasible_beam_loss_db": float(np.median(beam_values)),
            "median_feasible_beam_loss_ci95_db": [
                float(beam_ci[0]),
                float(beam_ci[1]),
            ],
        },
        "physical_j1_minus_physical_j2": {
            "paired_channel_nmse_db": nmse_contrast,
            "paired_net_rate_bps_hz": rate_contrast,
        },
        "point_criteria": point_criteria,
        "confidence_criteria": confidence_criteria,
    }
    all_contrasts = {
        "physical_j1_minus_physical_j2_channel_nmse_db": nmse_contrast,
        "physical_j1_minus_flat_j4_channel_nmse_db": _contrast(
            nmse_j1_flat, rng=rng, resamples=resamples
        ),
        "physical_j1_minus_physical_j2_net_rate_bps_hz": rate_contrast,
        "physical_j1_minus_flat_j4_net_rate_bps_hz": _contrast(
            rate_j1_flat, rng=rng, resamples=resamples
        ),
    }
    return gate, all_contrasts


def _figure_nmse(
    aggregate: pd.DataFrame,
    scenarios: list[str],
    methods: list[str],
    labels: dict[str, str],
    path: Path,
) -> list[dict]:
    subset = aggregate[aggregate["noise_mode"] == "matched_output"]
    if subset.empty:
        return []
    figure, axes = plt.subplots(
        1, len(scenarios), figsize=(4.8 * len(scenarios), 3.4),
        sharex=True, sharey=True, layout="constrained"
    )
    figure.get_layout_engine().set(wspace=0.08)
    axes = np.atleast_1d(axes)
    plot_records = []
    for panel, (axis, scenario) in enumerate(zip(axes, scenarios), start=1):
        panel_data = subset[subset["scenario"] == scenario]
        for index, method in enumerate(methods):
            values = panel_data[panel_data["design"] == method].sort_values(
                "target_snr_db"
            )
            if values.empty:
                continue
            x = values["target_snr_db"].to_numpy(dtype=float)
            estimate = values["mean_channel_nmse_db"].to_numpy(dtype=float)
            low = values["mean_channel_nmse_ci95_low_db"].to_numpy(dtype=float)
            high = values["mean_channel_nmse_ci95_high_db"].to_numpy(dtype=float)
            axis.errorbar(
                x,
                estimate,
                yerr=np.maximum(
                    0.0, np.vstack((estimate - low, high - estimate))
                ),
                color=COLORS[index],
                marker=MARKERS[index],
                linestyle=LINESTYLES[index],
                linewidth=1.7,
                markersize=5,
                capsize=3,
                label=labels[method],
            )
            for x_value, value, lo, hi in zip(x, estimate, low, high):
                plot_records.append(
                    {
                        "figure": "01_nmse_vs_output_snr",
                        "panel": panel,
                        "scenario": scenario,
                        "method": method,
                        "x": float(x_value),
                        "estimate": float(value),
                        "ci95_low": float(lo),
                        "ci95_high": float(hi),
                    }
                )
        axis.set_title(scenario.replace("_", " ").title())
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axes[0].set_ylabel("Mean CSI NMSE (dB; lower is better)")
    figure.supxlabel("Matched output SNR (dB)")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    return plot_records


def _figure_path_success(
    aggregate: pd.DataFrame,
    scenarios: list[str],
    methods: list[str],
    labels: dict[str, str],
    path: Path,
) -> list[dict]:
    subset = aggregate[aggregate["noise_mode"] == "matched_output"]
    if subset.empty:
        return []
    figure, axes = plt.subplots(
        1, len(scenarios), figsize=(4.8 * len(scenarios), 3.4),
        sharex=True, sharey=True, layout="constrained"
    )
    figure.get_layout_engine().set(wspace=0.08)
    axes = np.atleast_1d(axes)
    plot_records = []
    for panel, (axis, scenario) in enumerate(zip(axes, scenarios), start=1):
        panel_data = subset[subset["scenario"] == scenario]
        for index, method in enumerate(methods):
            values = panel_data[panel_data["design"] == method].sort_values(
                "target_snr_db"
            )
            if values.empty:
                continue
            x = values["target_snr_db"].to_numpy(dtype=float)
            estimate = values["both_paths_success_rate"].to_numpy(dtype=float)
            low = values["both_paths_success_ci95_low"].to_numpy(dtype=float)
            high = values["both_paths_success_ci95_high"].to_numpy(dtype=float)
            axis.errorbar(
                x,
                100.0 * estimate,
                yerr=100.0
                * np.maximum(
                    0.0, np.vstack((estimate - low, high - estimate))
                ),
                color=COLORS[index],
                marker=MARKERS[index],
                linestyle=LINESTYLES[index],
                linewidth=1.7,
                markersize=5,
                capsize=3,
                label=labels[method],
            )
            for x_value, value, lo, hi in zip(x, estimate, low, high):
                plot_records.append(
                    {
                        "figure": "02_two_path_success_vs_output_snr",
                        "panel": panel,
                        "scenario": scenario,
                        "method": method,
                        "x": float(x_value),
                        "estimate": float(100.0 * value),
                        "ci95_low": float(100.0 * lo),
                        "ci95_high": float(100.0 * hi),
                    }
                )
        axis.set_title(scenario.replace("_", " ").title())
        axis.set_ylim(-3.0, 103.0)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axes[0].set_ylabel("Both-path recovery rate (%; higher is better)")
    figure.supxlabel("Matched output SNR (dB)")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    return plot_records


def _figure_net_rate(
    frame: pd.DataFrame,
    config: dict,
    scenarios: list[str],
    methods: list[str],
    labels: dict[str, str],
    *,
    rng: np.random.Generator,
    resamples: int,
    path: Path,
) -> list[dict]:
    condition = str(config["decision"]["primary_noise_condition"])
    subset = frame[frame["snr_condition"] == condition]
    coherence_values = [
        float(value)
        for value in config["data_evaluation"]["coherence_symbols_for_plot"]
    ]
    figure, axes = plt.subplots(
        1, len(scenarios), figsize=(4.8 * len(scenarios), 3.4),
        sharex=True, sharey=True, layout="constrained"
    )
    figure.get_layout_engine().set(wspace=0.08)
    axes = np.atleast_1d(axes)
    plot_records = []
    for panel, (axis, scenario) in enumerate(zip(axes, scenarios), start=1):
        panel_data = subset[subset["scenario"] == scenario]
        for index, method in enumerate(methods):
            method_data = panel_data[panel_data["design"] == method]
            estimates = []
            lows = []
            highs = []
            for coherence in coherence_values:
                net = _net_rate(method_data, coherence)
                interval = _bootstrap_interval(
                    net, _mean, rng=rng, resamples=resamples
                )
                estimates.append(float(np.mean(net)))
                lows.append(float(interval[0]))
                highs.append(float(interval[1]))
                plot_records.append(
                    {
                        "figure": "03_net_rate_vs_coherence",
                        "panel": panel,
                        "scenario": scenario,
                        "method": method,
                        "x": coherence,
                        "estimate": estimates[-1],
                        "ci95_low": lows[-1],
                        "ci95_high": highs[-1],
                    }
                )
            estimate = np.asarray(estimates)
            low = np.asarray(lows)
            high = np.asarray(highs)
            axis.errorbar(
                coherence_values,
                estimate,
                yerr=np.maximum(
                    0.0, np.vstack((estimate - low, high - estimate))
                ),
                color=COLORS[index],
                marker=MARKERS[index],
                linestyle=LINESTYLES[index],
                linewidth=1.7,
                markersize=5,
                capsize=3,
                label=labels[method],
            )
        axis.set_title(scenario.replace("_", " ").title())
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axes[0].set_ylabel("Net spectral efficiency (bit/s/Hz)")
    figure.supxlabel("Coherence block (OFDM symbols)")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    return plot_records


def _write_report(path: Path, compact: dict) -> None:
    gate = compact["primary_gate"]
    physical = gate["physical_j1"]
    contrast = gate["physical_j1_minus_physical_j2"]
    point = gate["point_criteria"]
    confidence = gate["confidence_criteria"]
    labels = {
        "confirmed_pass": "通过",
        "provisional_pass": "暂时通过，需更多样本闭合置信区间",
        "stop_or_redesign": "不通过，应先诊断或重设计",
        "smoke_only": "仅程序冒烟，不作研究判决",
        "invalid_incomplete_data": "数据不完整，判决无效",
    }
    lines = [
        "# Stage 1-B 自动判决报告",
        "",
        f"- 数据完整性：{'通过' if compact['data_integrity']['complete'] else '失败'}",
        f"- 主判决：**{labels[gate['status']]}**",
        f"- 主场景：`{gate['primary_scenario']}`",
        f"- 主噪声条件：`{gate['primary_noise_condition']}`",
        "",
        "## 主结果",
        "",
        (
            "- 物理频响 J=1 的双路径恢复率："
            f"{100.0 * physical['both_paths_success_rate']:.1f}% "
            f"(95% bootstrap CI "
            f"{100.0 * physical['both_paths_success_ci95'][0]:.1f}%–"
            f"{100.0 * physical['both_paths_success_ci95'][1]:.1f}%)。"
        ),
        (
            "- 物理频响 J=1 的可行数据波束损失中位数："
            f"{physical['median_feasible_beam_loss_db']:.3f} dB "
            f"(95% bootstrap CI "
            f"{physical['median_feasible_beam_loss_ci95_db'][0]:.3f}–"
            f"{physical['median_feasible_beam_loss_ci95_db'][1]:.3f} dB)。"
        ),
        (
            "- J=1 相对物理时间扫描 J=2 的配对 CSI NMSE 差："
            f"{contrast['paired_channel_nmse_db']['mean']:+.3f} dB "
            f"(95% bootstrap CI "
            f"{contrast['paired_channel_nmse_db']['ci95'][0]:+.3f}–"
            f"{contrast['paired_channel_nmse_db']['ci95'][1]:+.3f} dB；"
            "负值有利于 J=1)。"
        ),
        (
            "- 扣除训练后，J=1 相对 J=2 的配对净谱效率差："
            f"{contrast['paired_net_rate_bps_hz']['mean']:+.4f} bit/s/Hz "
            f"(95% bootstrap CI "
            f"{contrast['paired_net_rate_bps_hz']['ci95'][0]:+.4f}–"
            f"{contrast['paired_net_rate_bps_hz']['ci95'][1]:+.4f})。"
        ),
        "",
        "## 门槛检查",
        "",
    ]
    for name, passed in point.items():
        lines.append(f"- 点估计 `{name}`：{'通过' if passed else '失败'}")
    for name, passed in confidence.items():
        lines.append(f"- 置信区间 `{name}`：{'通过' if passed else '未闭合'}")
    lines.extend(
        [
            "",
            "## 适用边界",
            "",
            "困难双路径场景只用于定位失效边界，不参与主通过判决。当前链路指标是载频处可行 DMA 状态的净谱效率代理，还不是完整宽带 OFDM 数据传输率。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_stage1b(run_dir: str | Path) -> dict:
    """Analyze one completed Stage 1-B run without modifying raw metrics."""

    run_path = Path(run_dir).resolve()
    raw_path = run_path / "metrics_raw.csv"
    summary_path = run_path / "metrics_summary.json"
    resolved_path = run_path / "config_resolved.json"
    if not raw_path.is_file() or not summary_path.is_file() or not resolved_path.is_file():
        raise FileNotFoundError(
            "run directory must contain metrics_raw.csv, metrics_summary.json, "
            "and config_resolved.json"
        )
    frame = pd.read_csv(raw_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    config = resolved["config"]
    resamples = int(config["experiment"]["bootstrap_resamples"])
    bootstrap_seed = int(config["experiment"]["bootstrap_seed"])
    rng = np.random.default_rng(bootstrap_seed)

    data_integrity = _validate_raw(frame, summary)
    methods, labels = _method_inventory(frame)
    method_names = {
        "physical_j1": methods[0],
        "physical_j2": methods[1],
        "flat_j4": methods[2],
    }
    aggregate = _aggregate(frame, rng=rng, resamples=resamples)
    gate, paired_contrasts = _primary_gate(
        frame,
        config,
        method_names,
        rng=rng,
        resamples=resamples,
    )
    if not bool(config["experiment"].get("research_decision_enabled", True)):
        gate["status"] = "smoke_only"
        gate["interpretation"] = (
            "program-path validation only; this configuration cannot issue a "
            "research continuation decision"
        )
    if not data_integrity["complete"]:
        gate["status"] = "invalid_incomplete_data"
        gate["interpretation"] = (
            "raw data integrity failed; no continuation decision is valid"
        )

    analysis_dir = run_path / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    aggregate.to_csv(analysis_dir / "summary_table.csv", index=False)
    scenarios = [str(item["name"]) for item in config["scenarios"]]
    plot_records = []
    plot_records.extend(
        _figure_nmse(
            aggregate,
            scenarios,
            methods,
            labels,
            figures_dir / "01_nmse_vs_output_snr.png",
        )
    )
    plot_records.extend(
        _figure_path_success(
            aggregate,
            scenarios,
            methods,
            labels,
            figures_dir / "02_two_path_success_vs_output_snr.png",
        )
    )
    plot_records.extend(
        _figure_net_rate(
            frame,
            config,
            scenarios,
            methods,
            labels,
            rng=rng,
            resamples=resamples,
            path=figures_dir / "03_net_rate_vs_coherence.png",
        )
    )
    pd.DataFrame.from_records(plot_records).to_csv(
        analysis_dir / "plot_data.csv", index=False
    )

    compact = {
        "analysis_schema": "direction1-stage1b-compact-v1",
        "source": {
            "run_dir": str(run_path),
            "metrics_raw_sha256": _sha256(raw_path),
            "config_sha256": resolved.get("config_sha256", "unavailable"),
            "analysis_code_sha256": _sha256(Path(__file__).resolve()),
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_resamples": resamples,
        },
        "data_integrity": data_integrity,
        "methods": method_names,
        "primary_gate": gate,
        "paired_contrasts": paired_contrasts,
        "interpretation_limits": [
            "exploratory two-path screen, not a publication-complete baseline suite",
            "hard_overlap is a boundary diagnostic and is excluded from the main gate",
            "data rate is a center-frequency feasible-state proxy, not full wideband OFDM rate",
            "no learning model or hardware mismatch is included",
        ],
    }
    (analysis_dir / "analysis_compact.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_report(analysis_dir / "analysis_report.md", compact)
    figure_manifest = {
        "raw_source": str(raw_path),
        "raw_source_sha256": compact["source"]["metrics_raw_sha256"],
        "transformation_code": str(Path(__file__).resolve()),
        "transformation_code_sha256": compact["source"][
            "analysis_code_sha256"
        ],
        "uncertainty": (
            "95% percentile bootstrap over independent generated scenes; "
            f"seed={bootstrap_seed}, resamples={resamples}"
        ),
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "palette": "Okabe-Ito dark subset plus marker and line-style redundancy",
        "export": {
            "format": "PNG",
            "dpi": 180,
            "background": "white",
            "figure_size_inches": [4.8 * len(scenarios), 3.4],
        },
        "figures": [
            {
                "path": "figures/01_nmse_vs_output_snr.png",
                "alt_text": (
                    "Two panels compare mean CSI NMSE versus matched output SNR "
                    "for three measurement schedules in nominal and difficult "
                    "two-path scenes; error bars are 95% bootstrap intervals."
                ),
            },
            {
                "path": "figures/02_two_path_success_vs_output_snr.png",
                "alt_text": (
                    "Two panels compare the percentage of scenes in which both "
                    "paths meet range and angle tolerances versus matched output SNR."
                ),
            },
            {
                "path": "figures/03_net_rate_vs_coherence.png",
                "alt_text": (
                    "Two panels compare center-frequency net spectral efficiency "
                    "after training overhead across coherence-block lengths."
                ),
            },
        ],
    }
    (analysis_dir / "figure_manifest.json").write_text(
        json.dumps(
            figure_manifest, indent=2, ensure_ascii=False, allow_nan=False
        ),
        encoding="utf-8",
    )
    return compact
