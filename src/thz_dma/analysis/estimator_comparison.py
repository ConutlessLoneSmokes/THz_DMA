"""Shared analysis for paired channel-estimator comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thz_dma.metrics.statistics import (
    bootstrap_interval as _bootstrap_interval,
    power_db as _db,
)


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def validate_estimator_rows(frame: pd.DataFrame, summary: dict) -> dict:
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


def net_rate(frame: pd.DataFrame, coherence_symbols: float) -> np.ndarray:
    overhead = (
        frame["training_symbols"].to_numpy(dtype=float)
        + frame["switch_guard_symbol_equivalents"].to_numpy(dtype=float)
    )
    fraction = np.clip(1.0 - overhead / coherence_symbols, 0.0, 1.0)
    return frame["feasible_gross_rate_bps_hz"].to_numpy(dtype=float) * fraction


def aggregate_estimator_metrics(
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
        net = net_rate(subset, coherence_symbols)
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


def paired_effect(
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


def estimator_contrasts(
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
        working["net_rate_bps_hz"] = net_rate(working, coherence)
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
                    "nmse_delta_db": paired_effect(
                        nmse, rng=rng, resamples=resamples
                    ),
                    "beam_loss_delta_db": paired_effect(
                        beam, rng=rng, resamples=resamples
                    ),
                    "net_rate_delta_bps_hz": paired_effect(
                        net, rng=rng, resamples=resamples
                    ),
                    "runtime_log10_ratio": paired_effect(
                        runtime_log10, rng=rng, resamples=resamples
                    ),
                    "median_runtime_ratio": float(
                        10.0 ** np.median(runtime_log10)
                    ),
                }
            )
            records.append(record)
    return records


def resource_contrasts(
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
    working["net_rate_bps_hz"] = net_rate(working, coherence)
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
                record[label] = paired_effect(values, rng=rng, resamples=resamples)
            records.append(record)
    return records


def figure_nmse(
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


def figure_primary_metrics(
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


def figure_runtime_tradeoff(
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
