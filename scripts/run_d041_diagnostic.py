"""Fresh paired D-041 data, algebra checks, Clean DDPM and frozen diagnostics."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
import torch

from _run_support import PROJECT_ROOT, RunArchive, create_run_archive, hash_file, read_toml, relative_path
from run_direction1_diffusion import atomic_json
from run_zhou2025_repro import (
    load_data, load_model, read_json, run_directory, save_data_manifest,
    synchronize, train, verify_physics, write_arrays,
)
from zhou2025_repro.data import decode, encode, signal_scale
from zhou2025_repro.ddpm import recover
from zhou2025_repro.d041 import (
    DiagnosticDigitalOperator, SpectralDmaOperator, diagnostics, geometry_ls,
    geometry_templates, hardware_weights, paired_array, paired_channel, state_bank,
)


def resolved_settings(args):
    config = read_toml(args.config)
    base_path = PROJECT_ROOT/config["base_config"]
    base = read_toml(base_path)
    for key in ("system", "channel", "atmosphere", "dma", "noise"):
        config[key] = copy.deepcopy(base[key])
    config["base_config_sha256"] = hash_file(base_path)
    config["band"] = args.band
    config["device"] = args.device or config["device"]
    config["channel"]["include_absorption"] = False
    dma = config["dma"]
    dma["element_spacing_m"] = config["diagnostic"]["element_spacing_m"]
    dma["microstrip_spacing_m"] = config["diagnostic"]["microstrip_spacing_m"]
    if (dma["num_microstrips"], dma["elements_per_microstrip"]) != (8, 160):
        raise ValueError("D-041 fixes the single-transmitter 8 x 160 array")
    if args.band == "narrow":
        config["system"].update(bandwidth_hz=0., num_subcarriers=1, nfft=1, cyclic_prefix_samples=0)
    config["acquisition"] = {"pilot_energy_per_re": 1., "full_frequency_coverage": True,
        "num_rf_outputs": 8, "digital_receive_channels": 1280,
        "state_design": "fixed_uniform_resonance_bank_prefix", "switch_guard_symbol_equivalents": .25}
    counts = config["diagnostic"]["configuration_counts"]
    if (len(set(counts)) != len(counts) or any(not isinstance(n, int) or n < 1 for n in counts)
            or counts != sorted(counts, reverse=True) or counts[0] < 160):
        raise ValueError("configuration counts must decrease from at least 160")
    floors = config["diagnostic"]["noiseless_likelihood_variances"]
    if len(set(floors)) < 2 or any(not math.isfinite(x) or x <= 0 for x in floors):
        raise ValueError("at least two distinct positive numerical variances required")
    if args.smoke:
        config["data"].update(train_count=8, validation_count=4, test_count=4)
        config["network"]["width"] = 16
        config["diffusion"].update(steps=8, beta_end=.3)
        config["training"].update(batch_size=4, epochs=2, validation_every=2, validation_scenes=2)
        config["evaluation_run"].update(validation_scenes=2, test_scenes=2, snr_db=[0., 20.], guidance=[.5, 1.])
    config["job"] = {key: value for key, value in vars(args).items() if key != "config"}
    config["job"]["config_path"] = args.config.resolve().relative_to(PROJECT_ROOT).as_posix()
    config["job"]["sure_variant"] = "corrected"  # shared Clean training archive schema
    return config


def signature(config):
    return {key: config[key] for key in ("band", "scenario", "system", "channel", "atmosphere",
        "dma", "noise", "acquisition", "diagnostic", "diffusion", "evaluation_run", "device")}


def prepare(archive, config):
    array = paired_array(config)
    counts = {split: config["data"][split+"_count"] for split in ("train", "validation", "test")}
    seeds = {split: config["data"][split+"_seed"] for split in counts}
    if len(set(seeds.values())) != 3:
        raise ValueError("split seeds must be distinct")
    shape = (len(array.frequencies), *array.positions.shape[:2])
    def producer(split, indices):
        pairs = [paired_channel(config, array, seeds[split], index) for index in indices]
        return np.stack([pair[0] for pair in pairs]).astype(np.complex64), [pair[1] for pair in pairs]
    progress = write_arrays(archive, counts, shape, producer, config["data"]["batch_size"])
    np.savez(archive.directory/"array.npz", frequencies_hz=array.frequencies,
        element_positions_m=array.positions, feed_distances_m=array.feed_distances)
    metadata = {"kind": "clean", "scenario": "dma", "storage": "npy", "band": config["band"],
        "counts": counts, "split_seeds": seeds, "channel_shape": list(shape),
        "known_noise_real_variance_phys": 0., "include_absorption": False,
        "complex_scale": signal_scale(progress["train_energy"], progress["train_entries"], 0.),
        "normalization_source": "all training samples; global complex RMS; never per scene",
        "files": {split: f"channels_{split}.npy" for split in counts},
        "physics": {key: config[key] for key in ("system", "channel", "atmosphere", "dma", "noise", "acquisition", "diagnostic")},
        "generator": "sample_stage2_paths + exact spherical_planar_path_channel; no coarse-center offsets",
        "pairing": "same split seed and scene index give byte-identical path records across bands",
        "diagnostic_only": True}
    save_data_manifest(archive, metadata)
    metadata["file_sha256"]["array.npz"] = hash_file(archive.directory/"array.npz")
    atomic_json(archive.directory/"dataset.json", metadata)


def operator_groups(config, stage, *, dtype=torch.complex64):
    array, bank = paired_array(config), state_bank(config)
    weights = hardware_weights(config, array, bank)
    shape = (len(array.frequencies), *array.positions.shape[:2])
    if stage != "reduced-noiseless":
        yield "digital", DiagnosticDigitalOperator(shape, device=config["device"]), None
    counts = config["diagnostic"]["configuration_counts"]
    selected = counts[:1] if stage == "full-noiseless" else counts[1:] if stage == "reduced-noiseless" else counts
    for count in selected:
        op = SpectralDmaOperator(weights[:count], shape[1],
            rf_noise_fraction=config["noise"]["rf_noise_variance_fraction_of_element_noise"],
            rtol=config["diagnostic"]["svd_relative_tolerance"], device=config["device"], dtype=dtype)
        yield f"dma_J{count}", op, weights[:count]


def operator_record(archive, label, operator, weights):
    tones, strips, elements = operator.channel_shape
    if weights is None:
        return {"label": label, "frontend": "digital", "configurations": 1,
            "complex_observations": tones*strips*elements, "complex_unknowns": tones*strips*elements,
            "rank": tones*strips*elements, "condition_max": 1., "pilot_re": tones,
            "total_pilot_energy": tones, "training_symbols": 1, "switches": 0}
    s, ranks = operator.singular_values.numpy(), operator.ranks.numpy()
    mask = operator.keep.numpy()
    conditions = [float(row[0]/row[keep][-1]) for row, keep in zip(s, mask)]
    count = weights.shape[0]
    np.savez(archive.directory/f"spectrum_{label}.npz", singular_values=s, ranks=ranks,
        relative_noise_variance=operator.relative_variance.cpu().numpy(),
        raw_row_energy=np.sum(abs(weights)**2, axis=-1))
    return {"label": label, "frontend": "dma", "configurations": count,
        "complex_observations": tones*strips*count, "complex_unknowns": tones*strips*elements,
        "rank": int(ranks.sum())*strips, "rank_per_frequency_min": int(ranks.min()),
        "rank_per_frequency_max": int(ranks.max()), "condition_max": max(conditions),
        "singular_value_min": float(s[mask].min()), "singular_value_max": float(s.max()),
        "pilot_re": tones*count, "total_pilot_energy": tones*count,
        "training_symbols": count, "switches": count-1,
        "weights_sha256": hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest(),
        "spectrum_file": f"spectrum_{label}.npz", "svd_precision": "complex128 CPU",
        "same_weights_on_all_strips": True, "raw_weights_not_normalized": True}


def save_hardware(archive, config):
    array = paired_array(config)
    np.savez(archive.directory/"hardware.npz", resonances_hz=state_bank(config),
        frequencies_hz=array.frequencies, element_positions_m=array.positions, feed_distances_m=array.feed_distances)


def matched_data(config):
    data = load_data(config["job"]["data_run_id"])
    verify_physics(config, data)
    if data.metadata.get("band") != config["band"] or data.metadata.get("include_absorption") is not False:
        raise ValueError("requires freshly generated, matching D-041 absorption-free data")
    return data


@torch.no_grad()
def check(archive, config):
    data = matched_data(config)
    save_hardware(archive, config)
    rows, checks, metadata = [], [], []
    settings = config["diagnostic"]
    array = paired_array(config)
    for label, op, weights in operator_groups(config, "noisy", dtype=torch.complex128):
        record = operator_record(archive, label, op, weights)
        metadata.append(record)
        truth = torch.as_tensor(data.physical("validation", [0]), device=config["device"], dtype=torch.complex128)/data.scale
        image = encode(truth, "dma")
        clean = op.forward(image)
        generator = torch.Generator(device=config["device"]).manual_seed(config["evaluation_run"]["noise_seed"])
        dual = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        lhs, rhs = (clean.conj()*dual).sum().real, (image*op.adjoint(dual)).sum()
        adjoint_gap = float(abs(lhs-rhs))/max(float(abs(lhs)), float(abs(rhs)), 1e-30)
        if weights is None:
            direct = truth[:, None]
        else:
            raw = np.einsum("jke,bkse->bsjk", weights.conj(), truth.cpu().numpy())
            direct = torch.as_tensor(raw, device=clean.device)/op.relative_variance.sqrt()[None]
        forward_gap = float((clean-direct).abs().square().sum()/direct.abs().square().sum())
        estimate = op.least_squares(clean)
        linear = diagnostics(op, truth, estimate, clean, clean)
        full = label == "digital" or record["configurations"] == max(settings["configuration_counts"])
        passed = (adjoint_gap <= settings["adjoint_relative_tolerance"] and forward_gap < 1e-20
            and linear["error_decomposition_relative_gap"] < 1e-10)
        if full:
            passed = passed and (record["rank"] == int(np.prod(data.shape)) and
                record["condition_max"] <= settings["maximum_full_condition"] and
                linear["nmse_linear"] <= settings["linear_nmse_tolerance"] and
                linear["observation_residual_relative"] <= settings["linear_residual_tolerance"])
        # Physical element noise is independent in different configurations.
        whitening = None
        if weights is not None:
            rng = np.random.default_rng(config["evaluation_run"]["noise_seed"])
            w = weights[:min(2, len(weights)), len(array.frequencies)//2]
            n = (rng.standard_normal((8192, *w.shape))+1j*rng.standard_normal((8192, *w.shape)))/math.sqrt(2)
            raw = np.einsum("dje,je->dj", n, w.conj())
            fraction = config["noise"]["rf_noise_variance_fraction_of_element_noise"]
            raw += math.sqrt(fraction)*(rng.standard_normal(raw.shape)+1j*rng.standard_normal(raw.shape))/math.sqrt(2)
            white = raw/np.sqrt(np.sum(abs(w)**2, axis=-1)+fraction)
            power = np.mean(abs(white)**2, axis=0)
            correlation = float(abs(np.mean(white[:, 0].conj()*white[:, 1]))) if len(w)>1 else 0.
            whitening = {"draws": 8192, "complex_power": power.tolist(),
                "real_component_power": np.mean(white.real**2, axis=0).tolist(), "cross_configuration_covariance_abs": correlation}
            passed = passed and max(abs(power-1)) < settings["whitening_power_tolerance"] and correlation < .06
        checks.append({"operator": label, "passed": bool(passed), "full_information": full,
            "adjoint_relative_gap": adjoint_gap, "forward_relative_squared_gap": forward_gap,
            "whitening": whitening, "linear": linear})
        for method, prediction in (("pinv_ls", estimate), ("oracle_geometry_ls", geometry_ls(op, clean,
                torch.as_tensor(geometry_templates(array, data.record("validation", 0)), device=clean.device)/data.scale))):
            rows.append({"operator": label, "estimator": method, "scene_id": data.record("validation", 0)["scene_id"],
                **diagnostics(op, truth, prediction, clean, clean)})
        print(f"checked {label}: rank={record['rank']}, condition={record['condition_max']:.3g}, passed={passed}", flush=True)
    archive.write_json("operators.json", metadata)
    archive.write_metrics(rows)
    archive.write_json("checks.json", {"passed": all(c["passed"] for c in checks), "checks": checks,
        "scope": "physics/algebra/noise checks; no trained-DDPM performance claim", "smoke_only": config["job"]["smoke"]})
    if not all(c["passed"] for c in checks):
        raise ArithmeticError("D-041 algebra/whitening checks failed; inspect checks.json before training")


def conditions(config):
    if config["job"]["stage"] == "noisy":
        return [(f"snr_{snr:g}", float(snr), None) for snr in config["evaluation_run"]["snr_db"]]
    return [(f"noiseless_v{variance:g}", None, float(variance))
            for variance in config["diagnostic"]["noiseless_likelihood_variances"]]


@torch.no_grad()
def training_reference(data, op, config):
    total = 0.
    for start in range(0, data.counts["train"], config["data"]["batch_size"]):
        indices = range(start, min(start+config["data"]["batch_size"], data.counts["train"]))
        y = op.forward(data.images("train", indices, config["device"]))
        total += float(y.abs().square().double().mean())*len(indices)
    return total/data.counts["train"]


def noise_like(clean, config, label, index, split):
    frontend = 0 if label == "digital" else 1
    seed = int(np.random.SeedSequence([config["evaluation_run"]["noise_seed"], frontend, index,
        0 if split == "validation" else 1]).generate_state(1)[0])
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    shape = list(clean.shape)
    if frontend:
        shape[2] = max(config["diagnostic"]["configuration_counts"])
    standard = torch.randn(shape, device=clean.device, dtype=clean.dtype, generator=generator)
    # Common configurations receive the same standardized noise; frontend-specific
    # training calibration still sets the scale, as required by the SNR control.
    return standard[:, :, :clean.shape[2]] if frontend else standard


def sample_observation(data, op, config, label, index, split, variance, noiseless):
    truth = torch.as_tensor(data.physical(split, [index]), device=config["device"])/data.scale
    clean = op.forward(encode(truth, "dma"))
    observed = clean if noiseless else clean+math.sqrt(variance)*noise_like(clean, config, label, index, split)
    return truth, clean, observed


def posterior_seed(config, label, index, split):
    # Image shape is identical within a band, so pair initial states across all
    # frontends/counts as well as across guidance/noise-variance candidates.
    return int(np.random.SeedSequence([config["evaluation_run"]["posterior_seed"], index,
        0 if split == "validation" else 1]).generate_state(1)[0])


def evaluation_dependencies(config, data):
    return {"data_run_id": data.directory.name, "dataset_sha256": hash_file(data.directory/"dataset.json"),
        "train_run_id": config["job"]["train_run_id"],
        "checkpoint_sha256": hash_file(run_directory(config["job"]["train_run_id"])/"checkpoint/training.pt")}


@torch.no_grad()
def evaluate(archive, config):
    data = matched_data(config)
    job, settings = config["job"], config["evaluation_run"]
    model, schedule, checkpoint, source = load_model(job["train_run_id"], config["device"])
    if (checkpoint["kind"] != "dm" or source.metadata["kind"] != "clean" or
            source.directory != data.directory or checkpoint["diffusion"] != config["diffusion"]):
        raise ValueError("digital/DMA require the same matched Clean DDPM and dataset")
    verify_physics(config, source)
    dependencies = evaluation_dependencies(config, data)
    dependency_path = archive.directory/"dependencies.json"
    if dependency_path.exists() and read_json(dependency_path) != dependencies:
        raise ValueError("evaluation dependencies changed across resume")
    atomic_json(dependency_path, dependencies)
    frozen = None
    if job["split"] == "test":
        prior = run_directory(job["tuning_run_id"])
        previous = read_json(prior/"config_resolved.json")["config"]
        if (read_json(prior/"status.json")["status"] != "complete" or previous["job"]["split"] != "validation"
                or previous["job"]["stage"] != job["stage"] or signature(previous) != signature(config)
                or read_json(prior/"dependencies.json") != dependencies):
            raise ValueError("test requires completed validation with identical stage/settings/dependencies")
        frozen = read_json(prior/"tuning.json")
        freeze_record = {"run_id": job["tuning_run_id"], "sha256": hash_file(prior/"tuning.json")}
        if (archive.directory/"tuning_source.json").exists() and read_json(archive.directory/"tuning_source.json") != freeze_record:
            raise ValueError("frozen tuning changed across test resume")
        archive.write_json("tuning_source.json", freeze_record)
    tuning_path = archive.directory/"tuning.json"
    tuning = read_json(tuning_path) if tuning_path.exists() else {}
    count = min(settings[job["split"]+"_scenes"], data.counts[job["split"]])
    tune_count = min(settings["validation_scenes"], data.counts["validation"])
    rows, records, references = [], [], {}
    save_hardware(archive, config)
    array = paired_array(config)
    for label, op, weights in operator_groups(config, job["stage"]):
        record = operator_record(archive, label, op, weights)
        records.append(record)
        reference = training_reference(data, op, config)
        if not reference > 0 or not math.isfinite(reference):
            raise ValueError("nonzero training-reference whitened observation power required")
        references[label] = {"training_scenes": data.counts["train"], "normalized_whitened_power": reference,
            "physical_whitened_power": reference*data.scale**2}
        for condition, snr, floor in conditions(config):
            variance = floor if snr is None else reference/10**(snr/10)
            key = f"{label}:{condition}"
            if frozen is not None:
                tuning[key] = frozen[key]
            elif key not in tuning:
                losses = []
                for choice in settings["guidance"]:
                    errors = []
                    for index in range(tune_count):
                        truth, _, observed = sample_observation(data, op, config, label, index, "validation", variance, snr is None)
                        prediction, _ = recover(model, schedule, op, observed, variance, scale=choice,
                            seed=posterior_seed(config, label, index, "validation"), refinement=settings["refinement"])
                        errors.append(float((decode(prediction, "dma")-truth).abs().square().sum()/truth.abs().square().sum()))
                    losses.append(float(np.mean(errors)))
                tuning[key] = {"selected": settings["guidance"][int(np.argmin(losses))],
                    "candidates": settings["guidance"], "validation_nmse_linear": losses,
                    "validation_scenes": tune_count, "select_final_endpoint_only": True}
            atomic_json(tuning_path, tuning)  # freeze before saving any corresponding metric block
            for index in range(count):
                case_path = archive.directory/"checkpoint"/f"case_{label}_{condition}_{index:05d}.json"
                if case_path.exists():
                    rows.extend(read_json(case_path)["rows"])
                    continue
                truth, clean, observed = sample_observation(data, op, config, label, index, job["split"], variance, snr is None)
                selected = tuning[key]["selected"]
                scene = data.record(job["split"], index)
                obs_count = int(clean.numel())
                complex_noise_phys = 0. if snr is None else variance*data.scale**2
                noise_power = float((observed-clean).abs().square().mean())
                relative = op.relative_variance
                rf_power = float((clean.abs().square()*relative).mean())
                common = {"band": config["band"], "operator": label, "condition": condition,
                    "split": job["split"], "scene_id": scene["scene_id"], "scene_index": index,
                    "stage": job["stage"], "target_whitened_snr_db": snr,
                    "actual_whitened_snr_db": None if snr is None else 10*math.log10(float(clean.abs().square().mean())/variance),
                    "realized_whitened_snr_db": None if snr is None else 10*math.log10(float(clean.abs().square().mean())/noise_power),
                    "actual_rf_output_snr_db": None if snr is None else 10*math.log10(rf_power/(variance*float(relative.mean()))),
                    "input_training_reference_snr_db": None if snr is None else -10*math.log10(variance),
                    "observation_noise_variance_normalized": 0. if snr is None else variance,
                    "physical_element_noise_complex_variance": complex_noise_phys,
                    "physical_rf_added_noise_complex_variance": complex_noise_phys*config["noise"]["rf_noise_variance_fraction_of_element_noise"],
                    "likelihood_variance_normalized": variance, "numerical_likelihood_variance_only": snr is None,
                    "complex_scale": data.scale, "complex_observations": obs_count,
                    "rank": record["rank"], "condition_max": record["condition_max"],
                    "pilot_re": record["pilot_re"], "total_pilot_energy": record["total_pilot_energy"],
                    "training_symbols": record["training_symbols"], "switches": record["switches"],
                    "guidance": selected, "model_run_id": job["train_run_id"], "smoke_only": job["smoke"]}
                trajectory = []
                def observer(state, info):
                    # Label the VP noise level; only the t=0 endpoint is the final CSI.
                    estimate = decode(state, "dma")/math.sqrt(info["state_alpha_bar"])
                    trajectory.append({**common, **info, "state_interpretation": "VP state / sqrt(current alpha_bar)",
                        "real_rms": float(state.square().mean().sqrt()),
                        **diagnostics(op, truth, estimate, observed, clean)})
                synchronize(config["device"])
                started = time.perf_counter()
                image, trace = recover(model, schedule, op, observed, variance, scale=selected,
                    seed=posterior_seed(config, label, index, job["split"]), refinement=settings["refinement"], trace_observer=observer)
                synchronize(config["device"])
                elapsed = time.perf_counter()-started
                estimates = [("clean_ddpm", decode(image, "dma"), elapsed, trace[-1]["network_calls"])]
                started = time.perf_counter()
                prediction = op.least_squares(observed)
                synchronize(config["device"])
                estimates.append(("pinv_ls", prediction, time.perf_counter()-started, 0))
                templates = torch.as_tensor(geometry_templates(array, scene), device=observed.device)/data.scale
                started = time.perf_counter()
                prediction = geometry_ls(op, observed, templates)
                synchronize(config["device"])
                estimates.append(("oracle_geometry_ls", prediction, time.perf_counter()-started, 0))
                case_rows = [{**common, "estimator": method, "online_seconds_with_diagnostics": seconds,
                    "network_calls": calls, "additional_true_geometry": method == "oracle_geometry_ls",
                    **diagnostics(op, truth, estimate, observed, clean)} for method, estimate, seconds, calls in estimates]
                atomic_json(case_path, {"rows": case_rows, "trajectory": trajectory})
                rows.extend(case_rows)
            print(f"saved {config['band']} {label} {condition}: {count} scenes; guidance={tuning[key]['selected']}", flush=True)
    archive.write_json("operators.json", records)
    archive.write_json("observation_reference.json", {"reference": references, "source": "all clean training scenes only",
        "noise_convention": "complex E|n|^2; real component variance = half",
        "dma_standard_noise_paired_over_configuration_prefixes": True,
        "posterior_initialization_paired_within_band": True,
        "independent_matched_frontend_noise_is_diagnostic_only": True})
    expected = len(records)*len(conditions(config))*count*3
    if len(rows) != expected:
        raise ValueError(f"incomplete evaluation: {len(rows)} != {expected}")
    frame = pd.DataFrame(rows)
    if frame.duplicated(["operator", "condition", "scene_id", "estimator"]).any():
        raise ValueError("duplicate evaluation rows")
    archive.write_metrics(rows)
    summary = []
    for (label, condition, method), block in frame.groupby(["operator", "condition", "estimator"]):
        summary.append({"operator": label, "condition": condition, "estimator": method, "scenes": len(block),
            "nmse_db": float(10*np.log10(max(block.nmse_linear.mean(), 1e-30))),
            **{key: float(block[key].mean()) for key in ("observation_residual_relative", "prediction_energy_ratio",
                "normalized_correlation", "row_space_nmse_linear", "null_space_nmse_linear")}})
    sensitivity = []
    if job["stage"] != "noisy":
        for label, block in frame.query("estimator == 'clean_ddpm'").groupby("operator"):
            pivot = block.pivot(index="scene_id", columns="condition", values="nmse_db")
            spread = pivot.max(axis=1)-pivot.min(axis=1)
            gates = pivot <= config["diagnostic"]["recovery_nmse_gate_db"]
            sensitivity.append({"operator": label, "max_per_scene_spread_db": float(spread.max()),
                "stable_within_declared_tolerance": bool((spread <= config["diagnostic"]["variance_sensitivity_db"]).all()),
                "same_pass_fail_at_all_variances": bool((gates.nunique(axis=1) == 1).all())})
    archive.write_json("metrics_summary.json", {"rows": len(rows), "expected_rows": expected,
        "band": config["band"], "stage": job["stage"], "split": job["split"], "summary": summary,
        "numerical_variance_sensitivity": sensitivity, "smoke_only": job["smoke"], "diagnostic_only": True,
        "trajectory_endpoint": "all stages saved; final t=0 selected by protocol, never by truth",
        "online_seconds_include_trace_metrics": True})
    pd.DataFrame(summary).to_csv(archive.directory/"comparison.csv", index=False)
    temporary = archive.directory/"trajectories.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for path in sorted((archive.directory/"checkpoint").glob("case_*.json")):
            trajectory = read_json(path)["trajectory"]
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(trajectory[0]))
                writer.writeheader()
            writer.writerows(trajectory)
    temporary.replace(archive.directory/"trajectories.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check", "train", "evaluate"])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT/"configs/zhou2025_d041.toml")
    parser.add_argument("--band", choices=["narrow", "wide"], required=True)
    parser.add_argument("--device")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-run-id")
    parser.add_argument("--train-run-id")
    parser.add_argument("--stage", choices=["full-noiseless", "reduced-noiseless", "noisy"], default="full-noiseless")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--tuning-run-id")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command != "prepare" and not args.data_run_id:
        parser.error("requires --data-run-id")
    if args.command == "evaluate" and not args.train_run_id:
        parser.error("evaluate requires one matched --train-run-id")
    if args.command == "evaluate" and args.split == "test" and not args.tuning_run_id:
        parser.error("test requires --tuning-run-id; no test-time parameter selection")
    config = resolved_settings(args)
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; repair/check driver or use --device cpu for smoke checks")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    directory = run_directory(args.run_id)
    if args.resume:
        archive = RunArchive(args.run_id, directory)
        previous = read_json(directory/"config_resolved.json")["config"]
        previous["job"]["resume"] = True
        if previous != config:
            raise ValueError("resume configuration differs")
        for item in read_json(directory/"code_manifest.json"):
            if hash_file(PROJECT_ROOT/relative_path(item["path"])) != item["sha256"]:
                raise ValueError(f"resume source changed: {item['path']}")
        if read_json(archive.status_path)["status"] == "complete":
            raise ValueError("completed runs are immutable; use a new run ID")
        archive.write_status("running", resumed=True)
    else:
        archive = create_run_archive("d041", args.run_id, config_path=args.config.resolve(), config=config,
            seeds={"data": config["data"], "training": config["training"]["seed"],
                "state": config["diagnostic"]["state_seed"], "noise": config["evaluation_run"]["noise_seed"],
                "posterior": config["evaluation_run"]["posterior_seed"]}, script_path=__file__,
            environment={"torch": str(torch.__version__), "cuda": str(torch.version.cuda), "device": config["device"],
                "gpu": torch.cuda.get_device_name(config["device"]) if config["device"].startswith("cuda") else "CPU"},
            checkpoint_note="New D-041 data/models/cases only; resume requires unchanged sources/config/dependencies.\n",
            extra_manifest_files=[PROJECT_ROOT/"scripts/run_zhou2025_repro.py",
                PROJECT_ROOT/"scripts/run_direction1_diffusion.py", PROJECT_ROOT/config["base_config"],
                PROJECT_ROOT/"scripts/analyze_d041_diagnostic.py", PROJECT_ROOT/"scripts/run_d041_linux.sh"])
        with zipfile.ZipFile(directory/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
            for item in read_json(directory/"code_manifest.json"):
                snapshot.write(PROJECT_ROOT/relative_path(item["path"]), item["path"])
    try:
        {"prepare": prepare, "check": check, "train": train, "evaluate": evaluate}[args.command](archive, config)
        archive.write_status("complete", command=args.command, band=args.band, smoke_only=args.smoke)
    except BaseException as error:
        archive.write_status("failed", command=args.command, error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
