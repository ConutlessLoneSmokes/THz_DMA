"""Compact paired analysis for the Stage 1-C estimator transfer screen."""

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


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


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


def _validate_raw(frame: pd.DataFrame, summary: dict) -> dict:
    required = {
        "scenario",
        "replicate_seed",
        "sample_index",
        "snr_condition",
        "design",
        "design_family",
        "estimator",
        "channel_nmse_linear",
        "channel_nmse_db",
        "both_paths_success",
        "feasible_beam_loss_db",
        "feasible_gross_rate_bps_hz",
        "estimator_runtime_seconds",
        "training_symbols",
        "switch_guard_symbol_equivalents",
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
        "estimator",
    ]
    duplicates = int(frame.duplicated(key).sum())
    expected = int(summary["experiment"]["expected_raw_metric_rows"])
    numeric_required = [
        "channel_nmse_linear",
        "channel_nmse_db",
        "both_paths_success",
        "feasible_beam_loss_db",
        "feasible_gross_rate_bps_hz",
        "estimator_runtime_seconds",
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
        "pairing_key": ["replicate_seed", "sample_index"],
    }


def _net_rate(frame: pd.DataFrame, coherence_symbols: float) -> np.ndarray:
    overhead = (
        frame["training_symbols"].to_numpy(dtype=float)
        + frame["switch_guard_symbol_equivalents"].to_numpy(dtype=float)
    )
    fraction = np.clip(1.0 - overhead / coherence_symbols, 0.0, 1.0)
    return frame["feasible_gross_rate_bps_hz"].to_numpy(dtype=float) * fraction


def _aggregate(
    frame: pd.DataFrame,
    *,
    coherence_symbols: float,
    rng: np.random.Generator,
    resamples: int,
) -> pd.DataFrame:
    records = []
    columns = [
        "scenario",
        "snr_condition",
        "noise_mode",
        "target_snr_db",
        "design",
        "design_family",
        "estimator",
        "method_source_key",
        "num_configurations",
        "training_symbols",
        "pilot_resource_elements",
        "num_switches",
        "switch_guard_symbol_equivalents",
    ]
    for keys, subset in frame.groupby(columns, dropna=False, sort=False):
        record = dict(zip(columns, keys))
        nmse = subset["channel_nmse_linear"].to_numpy(dtype=float)
        success = subset["both_paths_success"].to_numpy(dtype=float)
        beam = subset["feasible_beam_loss_db"].to_numpy(dtype=float)
        rate = subset["feasible_gross_rate_bps_hz"].to_numpy(dtype=float)
        runtime = subset["estimator_runtime_seconds"].to_numpy(dtype=float)
        net = _net_rate(subset, coherence_symbols)
        nmse_ci = _bootstrap_interval(
            nmse, np.mean, rng=rng, resamples=resamples
        )
        success_ci = _bootstrap_interval(
            success, np.mean, rng=rng, resamples=resamples
        )
        beam_ci = _bootstrap_interval(
            beam, np.median, rng=rng, resamples=resamples
        )
        net_ci = _bootstrap_interval(
            net, np.mean, rng=rng, resamples=resamples
        )
        runtime_ci = _bootstrap_interval(
            runtime, np.median, rng=rng, resamples=resamples
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
                "median_feasible_beam_loss_db": float(np.median(beam)),
                "median_feasible_beam_loss_ci95_low_db": float(beam_ci[0]),
                "median_feasible_beam_loss_ci95_high_db": float(beam_ci[1]),
                "mean_feasible_gross_rate_bps_hz": float(np.mean(rate)),
                "mean_net_rate_bps_hz": float(np.mean(net)),
                "mean_net_rate_ci95_low_bps_hz": float(net_ci[0]),
                "mean_net_rate_ci95_high_bps_hz": float(net_ci[1]),
                "median_batch1_runtime_seconds": float(np.median(runtime)),
                "median_batch1_runtime_ci95_low_seconds": float(runtime_ci[0]),
                "median_batch1_runtime_ci95_high_seconds": float(runtime_ci[1]),
                "algorithm_convergence_rate": float(
                    subset["algorithm_converged"].mean()
                ),
                "median_algorithm_iterations": float(
                    subset["algorithm_iterations"].median()
                ),
                "median_residual_energy_fraction": float(
                    subset["residual_energy_fraction"].median()
                ),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _paired_difference(
    subset: pd.DataFrame,
    estimator: str,
    reference: str,
    column: str,
) -> np.ndarray:
    pivot = subset.pivot(
        index=["replicate_seed", "sample_index"],
        columns="estimator",
        values=column,
    )
    if estimator not in pivot or reference not in pivot:
        raise ValueError(f"paired comparison lacks {estimator} or {reference}")
    values = (pivot[estimator] - pivot[reference]).dropna().to_numpy(dtype=float)
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


def _estimator_contrasts(
    frame: pd.DataFrame,
    config: dict,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> list[dict]:
    reference = str(config["analysis"]["reference_estimator"])
    coherence = float(config["analysis"]["coherence_symbols"])
    records = []
    group_columns = ["scenario", "snr_condition", "design", "design_family"]
    for keys, subset in frame.groupby(group_columns, sort=False):
        working = subset.copy()
        working["net_rate_bps_hz"] = _net_rate(working, coherence)
        estimators = sorted(set(working["estimator"]) - {reference})
        for estimator in estimators:
            nmse = _paired_difference(
                working, estimator, reference, "channel_nmse_db"
            )
            beam = _paired_difference(
                working, estimator, reference, "feasible_beam_loss_db"
            )
            net = _paired_difference(
                working, estimator, reference, "net_rate_bps_hz"
            )
            runtime_log10 = _paired_difference(
                working.assign(
                    runtime_log10=np.log10(
                        np.maximum(
                            working["estimator_runtime_seconds"].to_numpy(float),
                            np.finfo(float).tiny,
                        )
                    )
                ),
                estimator,
                reference,
                "runtime_log10",
            )
            record = dict(zip(group_columns, keys))
            record.update(
                {
                    "estimator": estimator,
                    "reference_estimator": reference,
                    "nmse_delta_db": _effect(
                        nmse, rng=rng, resamples=resamples
                    ),
                    "beam_loss_delta_db": _effect(
                        beam, rng=rng, resamples=resamples
                    ),
                    "net_rate_delta_bps_hz": _effect(
                        net, rng=rng, resamples=resamples
                    ),
                    "runtime_log10_ratio": _effect(
                        runtime_log10, rng=rng, resamples=resamples
                    ),
                    "median_runtime_ratio": float(
                        10.0 ** np.median(runtime_log10)
                    ),
                }
            )
            records.append(record)
    return records


def _resource_contrasts(
    frame: pd.DataFrame,
    config: dict,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> list[dict]:
    primary_family = str(config["analysis"]["primary_design_family"])
    reference_families = [
        str(value)
        for value in config["analysis"].get(
            "resource_reference_design_families", []
        )
    ]
    coherence = float(config["analysis"]["coherence_symbols"])
    working = frame.copy()
    working["net_rate_bps_hz"] = _net_rate(working, coherence)
    records = []
    for keys, subset in working.groupby(
        ["scenario", "snr_condition", "estimator"], sort=False
    ):
        designs = (
            subset[["design", "design_family"]]
            .drop_duplicates()
            .set_index("design_family")["design"]
            .to_dict()
        )
        primary = designs.get(primary_family)
        if primary is None:
            continue
        for family in reference_families:
            reference = designs.get(family)
            if reference is None:
                continue
            pivot_index = ["replicate_seed", "sample_index"]
            record = {
                "scenario": keys[0],
                "snr_condition": keys[1],
                "estimator": keys[2],
                "primary_design": primary,
                "reference_design": reference,
            }
            for column, label in [
                ("channel_nmse_db", "nmse_delta_db"),
                ("net_rate_bps_hz", "net_rate_delta_bps_hz"),
            ]:
                pivot = subset.pivot(
                    index=pivot_index, columns="design", values=column
                )
                values = (pivot[primary] - pivot[reference]).dropna().to_numpy(float)
                record[label] = _effect(values, rng=rng, resamples=resamples)
            records.append(record)
    return records


def _primary_result(
    aggregate: pd.DataFrame,
    contrasts: list[dict],
    config: dict,
    provenance: dict,
) -> dict:
    analysis = config["analysis"]
    subset = aggregate[
        (aggregate["scenario"] == str(analysis["primary_scenario"]))
        & (
            aggregate["snr_condition"]
            == str(analysis["primary_noise_condition"])
        )
        & (
            aggregate["design_family"]
            == str(analysis["primary_design_family"])
        )
    ].copy()
    if subset.empty:
        raise ValueError("primary Stage 1-C analysis block is empty")
    subset = subset.sort_values("mean_channel_nmse_db")
    ranking = []
    for _, row in subset.iterrows():
        estimator = str(row["estimator"])
        ranking.append(
            {
                "estimator": estimator,
                "label": provenance[estimator]["label"],
                "mean_channel_nmse_db": float(row["mean_channel_nmse_db"]),
                "both_paths_success_rate": float(
                    row["both_paths_success_rate"]
                ),
                "median_feasible_beam_loss_db": float(
                    row["median_feasible_beam_loss_db"]
                ),
                "mean_net_rate_bps_hz": float(row["mean_net_rate_bps_hz"]),
                "median_batch1_runtime_seconds": float(
                    row["median_batch1_runtime_seconds"]
                ),
            }
        )
    relevant = [
        item
        for item in contrasts
        if item["scenario"] == str(analysis["primary_scenario"])
        and item["snr_condition"] == str(analysis["primary_noise_condition"])
        and item["design_family"] == str(analysis["primary_design_family"])
    ]
    clear_gains = [
        item["estimator"]
        for item in relevant
        if item["nmse_delta_db"]["ci95"][1] < 0.0
    ]
    point_gains = [
        item["estimator"]
        for item in relevant
        if item["nmse_delta_db"]["mean"] < 0.0
    ]
    if not bool(config["experiment"]["research_decision_enabled"]):
        status = "smoke_only"
    elif clear_gains:
        status = "clear_external_method_gain"
    elif point_gains:
        status = "inconclusive_external_method_gain"
    else:
        status = "grid_omp_not_beaten"
    return {
        "status": status,
        "status_scope": (
            "method-screening outcome only; not a publication or direction gate"
        ),
        "scenario": str(analysis["primary_scenario"]),
        "noise_condition": str(analysis["primary_noise_condition"]),
        "design_family": str(analysis["primary_design_family"]),
        "reference_estimator": str(analysis["reference_estimator"]),
        "ranking_by_mean_linear_nmse": ranking,
        "clear_nmse_gain_estimators": clear_gains,
        "point_nmse_gain_estimators": point_gains,
        "paired_contrasts": relevant,
    }


def _figure_nmse(
    aggregate: pd.DataFrame,
    labels: dict[str, str],
    output: Path,
) -> None:
    conditions = list(dict.fromkeys(aggregate["snr_condition"].astype(str)))
    designs = list(dict.fromkeys(aggregate["design"].astype(str)))
    estimators = list(labels)
    fig, axes = plt.subplots(
        1, len(conditions), figsize=(6.2 * len(conditions), 4.6), squeeze=False
    )
    for axis, condition in zip(axes[0], conditions):
        subset = aggregate[aggregate["snr_condition"] == condition]
        x = np.arange(len(estimators), dtype=float)
        width = 0.34 if len(designs) == 2 else 0.8 / max(len(designs), 1)
        for index, design in enumerate(designs):
            values = []
            low = []
            high = []
            for estimator in estimators:
                row = subset[
                    (subset["design"] == design)
                    & (subset["estimator"] == estimator)
                ].iloc[0]
                value = float(row["mean_channel_nmse_db"])
                values.append(value)
                low.append(value - float(row["mean_channel_nmse_ci95_low_db"]))
                high.append(float(row["mean_channel_nmse_ci95_high_db"]) - value)
            offset = (index - (len(designs) - 1) / 2.0) * width
            axis.bar(
                x + offset,
                values,
                width=width,
                yerr=np.asarray([low, high]),
                capsize=3,
                color=COLORS[index % len(COLORS)],
                alpha=0.85,
                label=design,
            )
        axis.set_xticks(x, [labels[item] for item in estimators], rotation=18)
        axis.set_ylabel("Channel NMSE (dB; lower is better)")
        axis.set_title(condition)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _figure_primary_metrics(
    aggregate: pd.DataFrame,
    config: dict,
    labels: dict[str, str],
    output: Path,
) -> None:
    analysis = config["analysis"]
    subset = aggregate[
        (aggregate["scenario"] == str(analysis["primary_scenario"]))
        & (
            aggregate["snr_condition"]
            == str(analysis["primary_noise_condition"])
        )
        & (
            aggregate["design_family"]
            == str(analysis["primary_design_family"])
        )
    ].set_index("estimator")
    estimators = list(labels)
    x = np.arange(len(estimators))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    axes[0].bar(
        x,
        [100.0 * float(subset.loc[item, "both_paths_success_rate"]) for item in estimators],
        color=COLORS[: len(estimators)],
    )
    axes[0].set_ylabel("Both-path success (%)")
    axes[0].set_ylim(0.0, 100.0)
    axes[1].bar(
        x,
        [float(subset.loc[item, "median_feasible_beam_loss_db"]) for item in estimators],
        color=COLORS[: len(estimators)],
    )
    axes[1].set_ylabel("Median feasible beam loss (dB)")
    axes[2].bar(
        x,
        [float(subset.loc[item, "mean_net_rate_bps_hz"]) for item in estimators],
        color=COLORS[: len(estimators)],
    )
    axes[2].set_ylabel("Mean net rate (bit/s/Hz)")
    for axis in axes:
        axis.set_xticks(x, [labels[item] for item in estimators], rotation=18)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Primary J=1 estimator comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _figure_runtime_tradeoff(
    aggregate: pd.DataFrame,
    config: dict,
    labels: dict[str, str],
    output: Path,
) -> None:
    analysis = config["analysis"]
    subset = aggregate[
        (aggregate["scenario"] == str(analysis["primary_scenario"]))
        & (
            aggregate["snr_condition"]
            == str(analysis["primary_noise_condition"])
        )
    ]
    fig, axis = plt.subplots(figsize=(7.0, 5.0))
    designs = list(dict.fromkeys(subset["design"].astype(str)))
    for index, estimator in enumerate(labels):
        method = subset[subset["estimator"] == estimator]
        for design_index, design in enumerate(designs):
            row = method[method["design"] == design].iloc[0]
            marker = "o" if design_index == 0 else "s"
            axis.scatter(
                float(row["median_batch1_runtime_seconds"]),
                float(row["mean_channel_nmse_db"]),
                s=70,
                marker=marker,
                color=COLORS[index % len(COLORS)],
            )
            axis.annotate(
                f"{labels[estimator]}\n{design}",
                (
                    float(row["median_batch1_runtime_seconds"]),
                    float(row["mean_channel_nmse_db"]),
                ),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
            )
    axis.set_xscale("log")
    axis.set_xlabel("Median batch-1 estimator runtime (s; lower is better)")
    axis.set_ylabel("Channel NMSE (dB; lower is better)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_report(path: Path, compact: dict) -> None:
    primary = compact["primary_method_screen"]
    lines = [
        "# Stage 1-C 最近邻方法迁移自动报告",
        "",
        f"- 数据完整：`{compact['data_integrity']['complete']}`",
        f"- 方法筛查状态：`{primary['status']}`",
        "- 判决范围：该状态仅比较固定观测下的估计器，不决定研究方向是否继续。",
        "- 复现口径：当前两项外部方法均为同场景适配，不是论文原始数值的精确复现。",
        "",
        "## 主场景 J=1 排名",
        "",
        "| Estimator | NMSE (dB) | Both paths | Beam loss (dB) | Net rate | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary["ranking_by_mean_linear_nmse"]:
        lines.append(
            "| {label} | {nmse:.4f} | {success:.1f}% | {beam:.4f} | "
            "{rate:.4f} | {runtime:.6f} |".format(
                label=row["label"],
                nmse=row["mean_channel_nmse_db"],
                success=100.0 * row["both_paths_success_rate"],
                beam=row["median_feasible_beam_loss_db"],
                rate=row["mean_net_rate_bps_hz"],
                runtime=row["median_batch1_runtime_seconds"],
            )
        )
    lines.extend(
        [
            "",
            "## 配对效应相对 Grid OMP",
            "",
        ]
    )
    for item in primary["paired_contrasts"]:
        nmse = item["nmse_delta_db"]
        net = item["net_rate_delta_bps_hz"]
        lines.append(
            f"- `{item['estimator']}`：NMSE 差 {nmse['mean']:.4f} dB "
            f"(95% CI {nmse['ci95'][0]:.4f}, {nmse['ci95'][1]:.4f})；"
            f"净谱效率差 {net['mean']:.4f} bit/s/Hz "
            f"(95% CI {net['ci95'][0]:.4f}, {net['ci95'][1]:.4f})。"
        )
    lines.extend(
        [
            "",
            "## 解释规则",
            "",
            "- `clear_external_method_gain`：至少一种适配方法相对 OMP 的配对 NMSE 95% 区间完全低于 0。",
            "- `inconclusive_external_method_gain`：点估计改善，但区间仍跨 0。",
            "- `grid_omp_not_beaten`：当前适配方法没有降低主场景平均配对 NMSE。",
            "- 路径成功率只作诊断；主输出同时查看 CSI、可行数据波束、净谱效率和在线时间。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_stage1c(run_dir: str | Path) -> dict:
    """Validate, summarize, bootstrap, and plot one Stage 1-C run."""

    directory = Path(run_dir).resolve()
    raw_path = directory / "metrics_raw.csv"
    summary_path = directory / "metrics_summary.json"
    config_path = directory / "config_resolved.json"
    frame = pd.read_csv(raw_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))["config"]
    integrity = _validate_raw(frame, summary)
    bootstrap_seed = int(config["experiment"]["bootstrap_seed"])
    resamples = int(config["experiment"]["bootstrap_resamples"])
    rng = np.random.default_rng(bootstrap_seed)
    coherence = float(config["analysis"]["coherence_symbols"])
    aggregate = _aggregate(
        frame,
        coherence_symbols=coherence,
        rng=rng,
        resamples=resamples,
    )
    contrasts = _estimator_contrasts(
        frame, config, rng=rng, resamples=resamples
    )
    resource = _resource_contrasts(
        frame, config, rng=rng, resamples=resamples
    )
    provenance = summary["method_provenance"]
    primary = _primary_result(aggregate, contrasts, config, provenance)

    output = directory / "analysis"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output / "summary_table.csv", index=False)
    pd.DataFrame.from_records(
        [
            {
                "scenario": item["scenario"],
                "snr_condition": item["snr_condition"],
                "design": item["design"],
                "design_family": item["design_family"],
                "estimator": item["estimator"],
                "reference_estimator": item["reference_estimator"],
                "nmse_delta_db_mean": item["nmse_delta_db"]["mean"],
                "nmse_delta_db_ci95_low": item["nmse_delta_db"]["ci95"][0],
                "nmse_delta_db_ci95_high": item["nmse_delta_db"]["ci95"][1],
                "beam_loss_delta_db_mean": item["beam_loss_delta_db"]["mean"],
                "net_rate_delta_mean": item["net_rate_delta_bps_hz"]["mean"],
                "median_runtime_ratio": item["median_runtime_ratio"],
            }
            for item in contrasts
        ]
    ).to_csv(output / "paired_method_effects.csv", index=False)

    labels = {
        method: provenance[method]["label"]
        for method in config["experiment"]["estimators"]
    }
    figure_paths = [
        figures / "01_nmse_by_estimator.png",
        figures / "02_primary_communication_metrics.png",
        figures / "03_nmse_runtime_tradeoff.png",
    ]
    _figure_nmse(aggregate, labels, figure_paths[0])
    _figure_primary_metrics(aggregate, config, labels, figure_paths[1])
    _figure_runtime_tradeoff(aggregate, config, labels, figure_paths[2])

    compact = {
        "analysis_version": "stage1c-v1",
        "raw_metrics_sha256": _sha256(raw_path),
        "data_integrity": integrity,
        "comparison_contract": {
            "primary_axis": "estimator under identical observations",
            "resource_axis": "J=1 versus time-scan reference, reported separately",
            "primary_metric": "paired channel NMSE difference in dB",
            "secondary_metrics": [
                "both-path success",
                "feasible DMA beam loss",
                f"net rate at coherence {coherence:g} symbols",
                "batch-1 CPU runtime",
            ],
            "path_success_is_diagnostic": True,
            "paper_exact_reproduction": False,
        },
        "method_provenance": provenance,
        "primary_method_screen": primary,
        "all_estimator_contrasts": contrasts,
        "resource_reference_contrasts": resource,
        "bootstrap": {
            "seed": bootstrap_seed,
            "resamples": resamples,
            "interval": "percentile 95%",
            "resampling_unit": "independent scene within each paired block",
        },
    }
    (output / "analysis_compact.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_report(output / "analysis_report.md", compact)
    manifest = {
        "raw_metrics_sha256": compact["raw_metrics_sha256"],
        "figures": [
            {
                "path": str(path.relative_to(directory)),
                "sha256": _sha256(path),
            }
            for path in figure_paths
        ],
        "plot_contract": (
            "fixed Stage 1-C estimator ordering and paired primary block"
        ),
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return compact
