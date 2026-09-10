"""Compact paired analysis for the Stage 1-D tensor-transfer screen."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thz_dma.analysis.stage1c import (  # noqa: E402
    COLORS,
    _aggregate,
    _bootstrap_interval,
    _estimator_contrasts,
    _figure_nmse,
    _figure_primary_metrics,
    _figure_runtime_tradeoff,
    _resource_contrasts,
    _sha256,
    _validate_raw,
)


def _tensor_diagnostics(
    frame: pd.DataFrame,
    config: dict,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> list[dict]:
    tensor_estimator = str(config["analysis"]["tensor_estimator"])
    subset = frame[frame["estimator"] == tensor_estimator]
    required = [
        "tensor_rank_residual_fraction",
        "tensor_denoise_change_fraction",
        "tensor_factorization_order",
        "tensor_frequency_grid_aligned",
    ]
    missing = sorted(set(required) - set(subset.columns))
    if missing:
        raise ValueError(f"Stage 1-D metrics lack tensor diagnostics: {missing}")
    records = []
    for keys, group in subset.groupby(
        ["scenario", "snr_condition", "design", "design_family"],
        sort=False,
    ):
        rank_residual = group["tensor_rank_residual_fraction"].to_numpy(float)
        rank_ci = _bootstrap_interval(
            rank_residual, np.median, rng=rng, resamples=resamples
        )
        records.append(
            {
                "scenario": keys[0],
                "snr_condition": keys[1],
                "design": keys[2],
                "design_family": keys[3],
                "num_independent_scenes": int(len(group)),
                "mean_tensor_rank_residual_fraction": float(
                    np.mean(rank_residual)
                ),
                "median_tensor_rank_residual_fraction": float(
                    np.median(rank_residual)
                ),
                "median_tensor_rank_residual_ci95": [
                    float(rank_ci[0]),
                    float(rank_ci[1]),
                ],
                "median_tensor_denoise_change_fraction": float(
                    group["tensor_denoise_change_fraction"].median()
                ),
                "tensor_factorization_order": int(
                    group["tensor_factorization_order"].mode().iloc[0]
                ),
                "tensor_convergence_rate": float(
                    group["algorithm_converged"].mean()
                ),
                "median_tensor_iterations": float(
                    group["algorithm_iterations"].median()
                ),
                "frequency_grid_alignment_rate": float(
                    group["tensor_frequency_grid_aligned"].mean()
                ),
            }
        )
    return records


def _find_contrast(
    contrasts: list[dict],
    config: dict,
    design_family: str,
) -> dict:
    analysis = config["analysis"]
    matches = [
        item
        for item in contrasts
        if item["scenario"] == str(analysis["primary_scenario"])
        and item["snr_condition"]
        == str(analysis["primary_noise_condition"])
        and item["design_family"] == design_family
        and item["estimator"] == str(analysis["tensor_estimator"])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one tensor contrast for design family {design_family}"
        )
    return matches[0]


def _primary_result(
    aggregate: pd.DataFrame,
    contrasts: list[dict],
    diagnostics: list[dict],
    config: dict,
    provenance: dict,
) -> dict:
    analysis = config["analysis"]
    primary_family = str(analysis["primary_design_family"])
    native_family = str(analysis["tensor_native_design_family"])
    subset = aggregate[
        (aggregate["scenario"] == str(analysis["primary_scenario"]))
        & (
            aggregate["snr_condition"]
            == str(analysis["primary_noise_condition"])
        )
        & (aggregate["design_family"] == primary_family)
    ].sort_values("mean_channel_nmse_db")
    if subset.empty:
        raise ValueError("primary Stage 1-D analysis block is empty")
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
    same_observation = _find_contrast(contrasts, config, primary_family)
    native = _find_contrast(contrasts, config, native_family)
    if not bool(config["experiment"]["research_decision_enabled"]):
        status = "smoke_only"
    elif same_observation["nmse_delta_db"]["ci95"][1] < 0.0:
        status = "clear_same_observation_tensor_gain"
    elif native["nmse_delta_db"]["ci95"][1] < 0.0:
        status = "tensor_gain_requires_native_acquisition"
    elif (
        same_observation["nmse_delta_db"]["mean"] < 0.0
        or native["nmse_delta_db"]["mean"] < 0.0
    ):
        status = "inconclusive_tensor_gain"
    else:
        status = "tensor_not_better_than_ogols"

    warning = float(analysis["tensor_rank_residual_warning_fraction"])
    structural = []
    for item in diagnostics:
        if (
            item["scenario"] == str(analysis["primary_scenario"])
            and item["snr_condition"] == "noiseless"
        ):
            structural.append(
                {
                    **item,
                    "rank_p_model_warning": bool(
                        item["median_tensor_rank_residual_fraction"] > warning
                    ),
                }
            )
    return {
        "status": status,
        "status_scope": (
            "bounded tensor-mechanism screen only; not a paper-exact "
            "reproduction or publication gate"
        ),
        "scenario": str(analysis["primary_scenario"]),
        "noise_condition": str(analysis["primary_noise_condition"]),
        "primary_design_family": primary_family,
        "tensor_native_design_family": native_family,
        "reference_estimator": str(analysis["reference_estimator"]),
        "tensor_estimator": str(analysis["tensor_estimator"]),
        "ranking_by_mean_linear_nmse": ranking,
        "same_observation_tensor_effect": same_observation,
        "native_acquisition_tensor_effect": native,
        "noiseless_rank_p_structure": structural,
        "rank_residual_warning_fraction": warning,
    }


def _figure_tensor_structure(
    diagnostics: list[dict], config: dict, output: Path
) -> None:
    analysis = config["analysis"]
    scenario = str(analysis["primary_scenario"])
    warning = float(analysis["tensor_rank_residual_warning_fraction"])
    noiseless = [
        item
        for item in diagnostics
        if item["scenario"] == scenario and item["snr_condition"] == "noiseless"
    ]
    noisy = [
        item
        for item in diagnostics
        if item["scenario"] == scenario
        and item["snr_condition"] == str(analysis["primary_noise_condition"])
    ]
    designs = [item["design"] for item in noiseless]
    x = np.arange(len(designs))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].bar(
        x,
        [100.0 * item["median_tensor_rank_residual_fraction"] for item in noiseless],
        color=COLORS[: len(designs)],
    )
    axes[0].axhline(100.0 * warning, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Noiseless rank-P residual (%)")
    axes[0].set_xticks(x, designs, rotation=15)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        x,
        [100.0 * item["tensor_convergence_rate"] for item in noisy],
        color=COLORS[: len(designs)],
    )
    axes[1].set_ylabel("Tensor factorization convergence (%)")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].set_xticks(x, [item["design"] for item in noisy], rotation=15)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Transferred rank-P tensor assumption diagnostics")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_report(path: Path, compact: dict) -> None:
    primary = compact["primary_tensor_screen"]
    lines = [
        "# Stage 1-D 张量结构迁移自动报告",
        "",
        f"- 数据完整：`{compact['data_integrity']['complete']}`",
        f"- 张量筛查状态：`{primary['status']}`",
        "- 参照方法：Yang-style OG-OLS；OMP 只保留为低复杂度基线。",
        "- 复现边界：当前实现只迁移低秩频率张量机制，不是 Zhang 2025 原算法复现。",
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
    lines.extend(["", "## 张量方法相对 Yang 的配对效应", ""])
    for label, item in [
        ("J1 同观测", primary["same_observation_tensor_effect"]),
        ("张量原生采集", primary["native_acquisition_tensor_effect"]),
    ]:
        nmse = item["nmse_delta_db"]
        net = item["net_rate_delta_bps_hz"]
        lines.append(
            f"- {label}：NMSE 差 {nmse['mean']:.4f} dB "
            f"(95% CI {nmse['ci95'][0]:.4f}, {nmse['ci95'][1]:.4f})；"
            f"净谱效率差 {net['mean']:.4f} bit/s/Hz。"
        )
    lines.extend(["", "## 无噪声 rank-P 结构诊断", ""])
    for item in primary["noiseless_rank_p_structure"]:
        lines.append(
            f"- `{item['design']}`：中位低秩残差 "
            f"{100.0 * item['median_tensor_rank_residual_fraction']:.3f}%；"
            f"告警 `{item['rank_p_model_warning']}`。"
        )
    lines.extend(
        [
            "",
            "## 解释规则",
            "",
            "- `clear_same_observation_tensor_gain`：J1 相同观测下张量迁移相对 Yang 的 NMSE 区间完全低于 0。",
            "- `tensor_gain_requires_native_acquisition`：只有增加张量所需配置维后形成明确收益。",
            "- `inconclusive_tensor_gain`：点估计改善但区间跨 0。",
            "- `tensor_not_better_than_ogols`：两个采集块中均未超过 Yang。",
            "- 无噪声低秩残差用于检查迁移假设，不直接作为通信性能判决。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_stage1d(run_dir: str | Path) -> dict:
    """Validate, summarize, bootstrap, and plot one Stage 1-D run."""

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
    resources = _resource_contrasts(
        frame, config, rng=rng, resamples=resamples
    )
    diagnostics = _tensor_diagnostics(
        frame, config, rng=rng, resamples=resamples
    )
    provenance = summary["method_provenance"]
    primary = _primary_result(
        aggregate, contrasts, diagnostics, config, provenance
    )

    output = directory / "analysis"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output / "summary_table.csv", index=False)
    pd.DataFrame.from_records(diagnostics).to_csv(
        output / "tensor_structure_diagnostics.csv", index=False
    )
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
        figures / "04_tensor_structure_diagnostic.png",
    ]
    _figure_nmse(aggregate, labels, figure_paths[0])
    _figure_primary_metrics(aggregate, config, labels, figure_paths[1])
    _figure_runtime_tradeoff(aggregate, config, labels, figure_paths[2])
    _figure_tensor_structure(diagnostics, config, figure_paths[3])

    compact = {
        "analysis_version": "stage1d-v1",
        "raw_metrics_sha256": _sha256(raw_path),
        "data_integrity": integrity,
        "comparison_contract": {
            "primary_axis": (
                "tensor preprocessing versus Yang OG-OLS under identical J1 observations"
            ),
            "resource_axis": (
                "one-configuration J1 versus four-configuration tensor-native scan"
            ),
            "reference_estimator": str(config["analysis"]["reference_estimator"]),
            "primary_metric": "paired channel NMSE difference in dB",
            "secondary_metrics": [
                "feasible DMA beam loss",
                f"net rate at coherence {coherence:g} symbols",
                "batch-1 CPU runtime",
                "noiseless rank-P tensor residual",
            ],
            "path_success_is_diagnostic": True,
            "paper_exact_reproduction": False,
        },
        "method_provenance": provenance,
        "primary_tensor_screen": primary,
        "all_estimator_contrasts": contrasts,
        "resource_reference_contrasts": resources,
        "tensor_structure_diagnostics": diagnostics,
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
                "path": str(item.relative_to(directory)),
                "sha256": _sha256(item),
            }
            for item in figure_paths
        ],
        "plot_contract": "fixed Stage 1-D method ordering and paired blocks",
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return compact
