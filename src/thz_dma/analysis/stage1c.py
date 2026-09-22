"""Compact paired analysis for the Stage 1-C estimator transfer screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thz_dma.analysis.estimator_comparison import (
    aggregate_estimator_metrics as _aggregate,
    estimator_contrasts as _estimator_contrasts,
    figure_nmse as _figure_nmse,
    figure_primary_metrics as _figure_primary_metrics,
    figure_runtime_tradeoff as _figure_runtime_tradeoff,
    resource_contrasts as _resource_contrasts,
    validate_estimator_rows as _validate_raw,
)
from thz_dma.analysis.io import sha256 as _sha256











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
