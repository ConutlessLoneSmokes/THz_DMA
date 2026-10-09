"""Prepare/train/evaluate the D-032 conditional score prototype; each stage is resumable."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from _run_support import (PROJECT_ROOT, RunArchive, create_run_archive, hash_file,
                          read_toml, write_json)
from run_direction1_stage2a import validate_resume
from run_direction1_stage2a_linear import project_basis
from thz_dma.channels.planar import (PlanarPath, sample_stage2_paths,
                                    channel_from_planar_estimate)
from thz_dma.estimators.adaptive_grid import (_measure_adaptive_profiles,
    _sample_truth_around_centers, _adaptive_candidate_union, adaptive_settings)
from thz_dma.estimators.planar_baselines import (build_stage2a_operators,
                                                estimate_planar_baseline)
from thz_dma.estimators.planar_sparse import (PlanarTemplateBatchEvaluator,
    build_planar_template_dictionary_from_candidates)
from thz_dma.estimators.linear_channel import conditional_planar_covariance_basis, lmmse_coefficients
from thz_dma.estimators.diffusion import (ConditionalScoreNet, TorchDmaOperator,
    pack_channel, center_condition, posterior_samples, build_score_model, LowRankChannelPrior)
from thz_dma.metrics.diagnostics import channel_domain_metrics
from thz_dma.metrics.statistics import bootstrap_interval
from thz_dma.observations.multistrip import (array_input_noise_variance,
    observe_multistrip_schedule, effective_tensor_output_snr_db)
from thz_dma.evaluation.beamforming import build_planar_data_codebook, feasible_planar_data_metrics


def atomic_json(path, payload):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, payload)
    temporary.replace(path)


def atomic_torch(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def record_channel(record, platform, config):
    return channel_from_planar_estimate(platform.frequencies_hz, platform,
        np.asarray(record["coordinates"]), np.asarray(record["gain_real"])
        + 1j*np.asarray(record["gain_imag"]), config)


def read_example(directory, name):
    with np.load(directory / "data" / name, allow_pickle=False) as data:
        return data["channel"], json.loads(str(data["record"]))


def make_record(config, operators, evaluators, seed, sample):
    centers = sample_stage2_paths(config, np.random.default_rng(
        np.random.SeedSequence([seed, sample, 2026091101])))
    points = np.array([[p.range_m, p.azimuth_rad, p.elevation_rad] for p in centers])
    profile = _measure_adaptive_profiles(operators, points, config, evaluators)
    paths, truth = _sample_truth_around_centers(centers, profile, config,
        np.random.default_rng(np.random.SeedSequence([seed, sample, 2026091102])))
    return {"scene_id": f"{seed}:{sample}", "centers_m_rad_rad": points.tolist(),
            "schedule_profiles": profile["schedule_profiles"], "truth": truth,
            "coordinates": [[p.range_m, p.azimuth_rad, p.elevation_rad] for p in paths],
            "gain_real": [p.complex_gain.real for p in paths],
            "gain_imag": [p.complex_gain.imag for p in paths]}


def prepare(archive, config):
    settings = config["diffusion"]
    platform, _, operators = build_stage2a_operators(config, PROJECT_ROOT)
    evaluators = {name: PlanarTemplateBatchEvaluator(op, compute_backend="torch_cuda",
        compute_device=settings["device"]) for name, op in operators.items()}
    directory = archive.directory / "data"
    directory.mkdir(exist_ok=True)
    history_dir = PROJECT_ROOT / "runs" / settings["history_run"]
    history_config = json.loads((history_dir / "config_resolved.json").read_text())["config"]
    if config["job"]["smoke"]:
        history_config["noise"]["array_reference_snr_db"] = config["noise"]["array_reference_snr_db"]
    for key in config:
        if key not in ("diffusion", "job") and config[key] != history_config[key]:
            raise ValueError(f"history configuration mismatch: {key}")
    if json.loads((history_dir / "status.json").read_text())["status"] != "complete":
        raise ValueError("history reference is incomplete")
    history = json.loads((history_dir / "metrics_summary.json").read_text())["search"]["scene_records"]
    if config["job"]["smoke"]:
        history = history[:1]
    seeds = [settings["data"][s+"_seed"] for s in ("train", "validation", "test")]
    if len(set(seeds)) != 3 or set(seeds) & set(history_config["experiment"]["replicate_seeds"]):
        raise ValueError("data split seed collision")
    index, file_hashes = {}, {}
    train_energy, train_components = 0.0, 0
    for split in ("train", "validation", "test", "history"):
        count = len(history) if split == "history" else settings["data"][split+"_count"]
        index[split] = []
        for sample in range(count):
            name = f"{split}_{sample:05d}.npz"
            path = directory / name
            if path.exists():
                channel, record = read_example(archive.directory, name)
            else:
                if split == "history":
                    record = copy.deepcopy(history[sample])
                    seed, old_sample = map(int, record["scene_id"].split(":"))
                    centers = sample_stage2_paths(config, np.random.default_rng(
                        np.random.SeedSequence([seed, old_sample, 2026091101])))
                    record["coordinates"] = (np.asarray(record["centers_m_rad_rad"])
                        + record["truth"]["offsets_m_rad_rad"]).tolist()
                    record["gain_real"] = [p.complex_gain.real for p in centers]
                    record["gain_imag"] = [p.complex_gain.imag for p in centers]
                else:
                    record = make_record(config, operators, evaluators, settings["data"][split+"_seed"], sample)
                channel = record_channel(record, platform, config).astype(np.complex64)
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, channel=channel, record=json.dumps(record))
                temporary.replace(path)
            if not np.isfinite(channel).all():
                raise ValueError(f"invalid channel in {name}")
            if split == "train":
                train_energy += float(np.sum(np.abs(channel.astype(np.complex128))**2))
                train_components += 2 * channel.size
            index[split].append(name)
            file_hashes[name] = hash_file(path)
            print(f"saved {split} {sample+1}/{count}", flush=True)
    scale = math.sqrt(train_energy / train_components)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid training-only normalization")
    atomic_json(archive.directory / "dataset.json", {
        "index": index, "file_sha256": file_hashes, "real_component_scale": scale,
        "normalization_source": "training channels only; global real-component RMS; no per-test normalization",
        "history_metrics_sha256": hash_file(history_dir / "metrics_raw.csv")})


def require_dataset(data_id, config):
    directory = PROJECT_ROOT / "runs" / data_id
    if json.loads((directory / "status.json").read_text())["status"] != "complete":
        raise ValueError("data preparation must finish first")
    archived = json.loads((directory / "config_resolved.json").read_text())["config"]
    if (any(archived[k] != config[k] for k in config if k not in ("job", "diffusion"))
            or any(archived["diffusion"][k] != config["diffusion"][k] for k in ("data", "history_run"))):
        raise ValueError("dataset config differs; use a new matching experiment")
    dataset = json.loads((directory / "dataset.json").read_text())
    # Verify once at stage start; training does not reread checksums at every step.
    for name, digest in dataset["file_sha256"].items():
        if hash_file(directory / "data" / name) != digest:
            raise ValueError(f"dataset modified: {name}")
    return directory, dataset


def load_batch(directory, names, scale, config, device):
    channels, conditions = [], []
    for name in names:
        h, record = read_example(directory, name)
        channels.append(h / scale)
        conditions.append(center_condition(record["centers_m_rad_rad"], config))
    return (pack_channel(torch.as_tensor(np.stack(channels), device=device)),
            torch.as_tensor(np.stack(conditions), device=device))


@torch.no_grad()
def validate_model(model, directory, dataset, config):
    model.eval()
    settings = config["diffusion"]
    device = settings["device"]
    generator = torch.Generator(device=device).manual_seed(settings["training"]["seed"]+1)
    names = dataset["index"]["validation"][:settings["training"]["validation_scenes"]]
    levels = np.geomspace(settings["posterior"]["sigma_min"], settings["posterior"]["sigma_max"], 5)
    losses = []
    for name in names:
        clean, centers = load_batch(directory, [name], dataset["real_component_scale"], config, device)
        for sigma in levels:
            noise = torch.randn(clean.shape, device=device, generator=generator)
            prediction = model(clean+sigma*noise, clean.new_full((1,), sigma), centers)
            weight = 1+sigma**2 if settings["network"].get("kind") == "edm_fourier" else 1.
            losses.append(float(weight*F.mse_loss(prediction, noise)))
    return float(np.mean(losses))


def train(archive, config):
    settings, job = config["diffusion"], config["job"]
    directory, dataset = require_dataset(job["data_run_id"], config)
    dataset_hash = hash_file(directory / "dataset.json")
    device, training = settings["device"], settings["training"]
    torch.manual_seed(training["seed"])
    torch.cuda.manual_seed_all(training["seed"])
    model = build_score_model(settings["network"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    checkpoint_path = archive.directory / "checkpoint/training.pt"
    start, history, best, best_step = 0, [], float("inf"), 0
    best_model = copy.deepcopy(model.state_dict())
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint["dataset_sha256"] != dataset_hash:
            raise ValueError("training dataset changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, history = checkpoint["step"], checkpoint["history"]
        best, best_step, best_model = checkpoint["best_loss"], checkpoint["best_step"], checkpoint["best_model"]
        torch.set_rng_state(checkpoint["cpu_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    names = dataset["index"]["train"]
    posterior = settings["posterior"]
    started = time.perf_counter()
    for step in range(start+1, training["steps"]+1):
        model.train()
        indices = torch.randint(len(names), (training["batch_size"],)).tolist()
        clean, centers = load_batch(directory, [names[i] for i in indices],
                                    dataset["real_component_scale"], config, device)
        sigma = torch.exp(torch.empty(len(indices), device=device).uniform_(
            math.log(posterior["sigma_min"]), math.log(posterior["sigma_max"])))
        noise = torch.randn_like(clean)
        noisy = clean + sigma[:, None, None, None] * noise
        errors = (model(noisy, sigma, centers)-noise).square().mean(dim=(1,2,3))
        if settings["network"].get("kind") == "edm_fourier":
            errors = errors*(1+sigma.square())
        loss = errors.mean()
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        if step % training["validation_every"] == 0 or step == training["steps"]:
            score = validate_model(model, directory, dataset, config)
            if not math.isfinite(score):
                raise ArithmeticError("nonfinite validation loss")
            if score < best:
                best, best_step, best_model = score, step, copy.deepcopy(model.state_dict())
            metric = "validation_edm_loss" if settings["network"].get("kind") == "edm_fourier" else "validation_noise_mse"
            history.append({"step": step, "training_loss": float(loss.detach()), metric: score})
            atomic_torch(checkpoint_path, {"model": model.state_dict(), "best_model": best_model,
                "optimizer": optimizer.state_dict(), "step": step, "best_loss": best,
                "best_step": best_step, "history": history, "dataset_sha256": dataset_hash,
                "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()})
            atomic_json(archive.directory / "training_history.json", history)
            print(f"step {step}/{training['steps']}, {metric}={score:.6f}", flush=True)
    archive.write_json("training_summary.json", {"parameters": sum(p.numel() for p in model.parameters()),
        "best_validation_loss": best,
        "validation_loss_type": "weighted_edm" if settings["network"].get("kind") == "edm_fourier" else "noise_mse",
        "best_step": best_step,
        "session_seconds": time.perf_counter()-started, "dataset_sha256": dataset_hash,
        "trained_steps": training["steps"], "not_a_recovery_performance_result": True})


def observation_signal_reference(directory, dataset, operators):
    """Training-set mean whitened pilot power; independent of evaluation truth."""
    scale = dataset["real_component_scale"]
    powers = {name: 0. for name in operators}
    with torch.no_grad():
        for name in dataset["index"]["train"]:
            channel, _ = read_example(directory, name)
            op = next(iter(operators.values()))
            x = pack_channel(torch.as_tensor(channel/scale, device=op.matrix.device)[None])
            for label, operator in operators.items():
                powers[label] += float(operator.forward(x).abs().square().double().mean())*scale**2
    return {name: value/len(dataset["index"]["train"]) for name,value in powers.items()}


def evaluate(archive, config):
    settings, job = config["diffusion"], config["job"]
    observation_axis = "observation_snr_db" in job
    snrs = job.get("observation_snr_db", job.get("snr_db", config["noise"]["array_reference_snr_db"]))
    snr_column = "observation_snr_target_db" if observation_axis else "array_reference_snr_db"
    directory, dataset = require_dataset(job["data_run_id"], config)
    training_dir = PROJECT_ROOT / "runs" / job["train_run_id"]
    trained_config = json.loads((training_dir / "config_resolved.json").read_text())["config"]
    if (any(trained_config["diffusion"][k] != settings[k] for k in ("network", "training"))
            or any(trained_config["diffusion"]["posterior"][k] != settings["posterior"][k]
                   for k in ("sigma_min", "sigma_max"))):
        raise ValueError("training configuration mismatch")
    if json.loads((training_dir / "status.json").read_text())["status"] != "complete":
        raise ValueError("training run must be complete")
    model_path = training_dir / "checkpoint/training.pt"
    model_hash = hash_file(model_path)
    checkpoint = torch.load(model_path, map_location=settings["device"], weights_only=True)
    if checkpoint["dataset_sha256"] != hash_file(directory / "dataset.json"):
        raise ValueError("model trained on a different dataset")
    model = build_score_model(settings["network"]).to(settings["device"])
    model.load_state_dict(checkpoint["best_model"])
    model.eval()
    del checkpoint
    dependency = {"model_sha256": model_hash, "dataset_sha256": hash_file(directory / "dataset.json")}
    dependency_path = archive.directory / "dependencies.json"
    if dependency_path.exists() and json.loads(dependency_path.read_text()) != dependency:
        raise ValueError("evaluation dependency changed since previous session")
    archive.write_json("dependencies.json", dependency)
    platform, weights, operators = build_stage2a_operators(config, PROJECT_ROOT)
    tensor_operators = {s.name: TorchDmaOperator(operators[s.name], s.pilot_indices,
        device=settings["device"]) for s in platform.schedules}
    reference = None
    if observation_axis:
        reference_path = archive.directory/"observation_snr_calibration.json"
        if reference_path.exists():
            reference = json.loads(reference_path.read_text())["training_reference_power"]
        else:
            reference = observation_signal_reference(directory, dataset, tensor_operators)
            archive.write_json(reference_path.name, {"training_reference_power": reference,
                "training_scenes": len(dataset["index"]["train"]), "dataset_sha256": dependency["dataset_sha256"],
                "definition": "mean whitened clean pilot power over training scenes / complex white noise variance",
                "matched_observation_snr_diagnostic_only": True})
    common = set.intersection(*(set(s.pilot_indices) for s in platform.schedules))
    beam_index = min(common, key=lambda i: abs(platform.frequencies_hz[i]-config["system"]["carrier_frequency_hz"]))
    dma = config["dma"]
    codebook, _ = build_planar_data_codebook(carrier_frequency_hz=float(platform.frequencies_hz[beam_index]),
        feed_distances_m=platform.feed_distances_m,
        **{k: dma[k] for k in ("num_microstrips", "elements_per_microstrip", "tuning_bandwidth_hz",
            "quality_factor", "waveguide_effective_index", "waveguide_end_power_fraction")},
        codebook_size=config["data_evaluation"]["codebook_size"],
        seed_start=config["data_evaluation"]["codebook_seed_start"])
    if job["scene_limit"] > len(dataset["index"][job["split"]]):
        raise ValueError("scene limit exceeds the selected split")
    names = dataset["index"][job["split"]][:job["scene_limit"]]
    scale = dataset["real_component_scale"]
    with torch.no_grad():
        warmup = torch.zeros(settings["posterior"]["samples"], 16, 160, 128, device=settings["device"])
        model(warmup, warmup.new_ones(len(warmup)), warmup.new_zeros((len(warmup), 9)))
        torch.cuda.synchronize()
        del warmup
    rows = []
    methods = ["oracle_geometry_ls", "grid_omp_3d", "sbl_jointfreq_3d", "lmmse", "diffusion_mean"]
    structured = settings["posterior"].get("covariance", "isotropic") == "mc_subspace"
    if structured:
        methods.append("gaussian_diffusion_mean")
    for number, name in enumerate(names):
        saved_path = archive.directory / "checkpoint" / (Path(name).stem+".json")
        if saved_path.exists():
            saved_rows = json.loads(saved_path.read_text())
            if len(saved_rows) != 2*len(snrs)*len(methods):
                raise ValueError("incomplete evaluation checkpoint")
            rows.extend(saved_rows)
            print(f"restored {number+1}/{len(names)}", flush=True)
            continue
        _, record = read_example(directory, name)
        seed, sample = map(int, record["scene_id"].split(":"))
        truth = record_channel(record, platform, config)
        condition = torch.as_tensor(center_condition(record["centers_m_rad_rad"], config),
                                    device=settings["device"])[None]
        profile = {"settings": adaptive_settings(config), "schedule_profiles": record["schedule_profiles"]}
        candidates, _ = _adaptive_candidate_union(np.asarray(record["centers_m_rad_rad"]),
            profile, np.asarray(record["coordinates"]), config)
        paths = tuple(PlanarPath(*p, complex(r, i)) for p, r, i in
            zip(record["coordinates"], record["gain_real"], record["gain_imag"]))
        prep_started = time.perf_counter()
        basis = conditional_planar_covariance_basis(platform, config, record,
            draws=settings["posterior"]["prior_draws"],
            rng=np.random.default_rng(np.random.SeedSequence([settings["posterior"]["prior_seed"], seed, sample])))
        covariance_seconds = time.perf_counter()-prep_started
        prior, prior_seconds = None, 0.
        if structured:
            prep_started = time.perf_counter()
            prior = LowRankChannelPrior(basis/scale, device=settings["device"],
                relative_cutoff=settings["posterior"]["covariance_relative_cutoff"])
            torch.cuda.synchronize()
            prior_seconds = time.perf_counter()-prep_started
        scene_rows = []
        for schedule_index, schedule in enumerate(platform.schedules):
            operator = operators[schedule.name]
            prep_started = time.perf_counter()
            dictionary = build_planar_template_dictionary_from_candidates(operator, *candidates,
                chunk_size=config["search"]["dictionary_chunk_size"], storage_dtype="complex64",
                compute_backend="torch_cuda", compute_device=settings["device"])
            dictionary_seconds = time.perf_counter()-prep_started
            prep_started = time.perf_counter()
            observed_basis = project_basis(basis, weights[schedule.name], operator, schedule.pilot_indices)
            projection_seconds = time.perf_counter()-prep_started
            for snr in snrs:
                snr_index = snrs.index(snr) if observation_axis else config["noise"]["array_reference_snr_db"].index(snr)
                variance = reference[schedule.name]/10**(snr/10) if observation_axis else array_input_noise_variance(truth, snr)
                rf_variance = variance*config["noise"]["rf_noise_variance_fraction_of_element_noise"]
                observation = observe_multistrip_schedule(weights[schedule.name], truth[schedule.pilot_indices],
                    pilot_energy_per_re=1.0, element_noise_variance=variance, rf_noise_variance=rf_variance,
                    rng=np.random.default_rng(np.random.SeedSequence([seed, sample, schedule_index, 2026090921])))
                white = operator.whiten(observation.observed)
                expected_y = torch.as_tensor((white/scale).reshape(operator.observation_shape),
                    dtype=torch.complex64, device=settings["device"])[None]
                order = np.random.default_rng(np.random.SeedSequence([2026100293, seed, sample,
                    schedule_index, snr_index])).permutation(methods)
                for method in order:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    extras = {}
                    if method in ("diffusion_mean", "gaussian_diffusion_mean"):
                        posterior_seed = int(np.random.SeedSequence([settings["posterior"]["seed"], seed,
                            sample, schedule_index, snr_index]).generate_state(1)[0])
                        core = model if method == "diffusion_mean" else None
                        samples, trace = posterior_samples(core, tensor_operators[schedule.name], expected_y,
                            condition, variance/scale**2, settings["posterior"], seed=posterior_seed, gaussian_prior=prior)
                        estimate = samples.mean(0).cpu().numpy()*scale
                        extras = {"posterior_seed": posterior_seed,
                            "posterior_spread_normalized": float((samples-samples.mean(0)).abs().square().mean()),
                            "sampler_trace": json.dumps(trace), "posterior_variance_calibrated": False}
                        del samples
                    elif method == "lmmse":
                        coefficients = lmmse_coefficients(observed_basis, white, variance)
                        estimate = (basis.reshape(len(basis), -1).T @ coefficients.astype(np.complex64)).reshape(truth.shape)
                    else:
                        fit = estimate_planar_baseline(method, dictionary, operator, white, paths, config,
                                                       noise_variance=variance)
                        coordinates = fit.get("reconstruction_coordinates", np.column_stack(
                            (fit["range_m"], fit["azimuth_rad"], fit["elevation_rad"])))
                        gains = fit.get("reconstruction_gains", fit["complex_gain"])
                        estimate = channel_from_planar_estimate(platform.frequencies_hz, platform, coordinates, gains, config)
                        extras = {"algorithm_converged": int(fit["converged"]),
                                  "algorithm_iterations": int(fit["iterations"])}
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-started
                    link = feasible_planar_data_metrics(truth[beam_index], estimate[beam_index], codebook,
                        element_noise_variance=variance, rf_noise_variance=rf_variance,
                        coherence_symbols=config["data_evaluation"]["coherence_symbols"],
                        training_symbols=schedule.training_symbols,
                        switch_guard_symbol_equivalents=schedule.num_switches*config["acquisition"]["switch_guard_symbol_equivalents"])
                    scene_rows.append({"scene_id": record["scene_id"], "schedule": schedule.name,
                        "array_reference_snr_db": float(10*np.log10(np.mean(np.sum(abs(truth)**2,axis=(1,2)))/variance)) if observation_axis else snr,
                        **({"observation_snr_target_db": snr,
                            "whitened_observation_snr_db": float(10*np.log10(np.mean(abs(operator.whiten(observation.clean))**2)/variance))} if observation_axis else {}),
                        "estimator": str(method),
                        "effective_output_snr_db": effective_tensor_output_snr_db(observation.clean, observation.output_noise_variance),
                        **channel_domain_metrics(truth, estimate, schedule.pilot_indices, np.arange(128)),
                        "pilot_beam_loss_db": link["feasible_beam_loss_db"], "pilot_net_rate": link["net_rate_bps_hz"],
                        "estimation_and_reconstruction_seconds": elapsed,
                        "dictionary_preparation_seconds": dictionary_seconds if method in ("grid_omp_3d", "sbl_jointfreq_3d") else 0.0,
                        "covariance_preparation_seconds": covariance_seconds+projection_seconds if method == "lmmse" else 0.0,
                        "diffusion_prior_preparation_seconds": covariance_seconds+prior_seconds if structured and method in ("diffusion_mean", "gaussian_diffusion_mean") else 0.,
                        "covariance_rank": prior.rank if structured and method in ("diffusion_mean", "gaussian_diffusion_mean") else None,
                        "covariance_retained_energy": prior.retained_energy if structured and method in ("diffusion_mean", "gaussian_diffusion_mean") else None,
                        "prior_draws": settings["posterior"]["prior_draws"] if method == "lmmse" or (structured and method in ("diffusion_mean", "gaussian_diffusion_mean")) else 0,
                        "algorithm_converged": None, "algorithm_iterations": None,
                        "posterior_seed": None, "posterior_spread_normalized": None,
                        "sampler_trace": "", "posterior_variance_calibrated": False, **extras})
            del dictionary
        del basis
        del prior
        atomic_json(saved_path, scene_rows)
        rows.extend(scene_rows)
        print(f"saved scene {number+1}/{len(names)}, rows={len(rows)}", flush=True)
    frame = pd.DataFrame(rows)
    assert len(frame) == len(names)*2*len(snrs)*len(methods)
    assert not frame.duplicated(["scene_id", "schedule", snr_column, "estimator"]).any()
    archive.write_metrics(rows)
    analysis = []
    rng = np.random.default_rng(2026100294)
    for (schedule, snr), block in frame.groupby(["schedule", snr_column]):
        pivot = block.pivot(index="scene_id", columns="estimator", values="fullband_128_nmse_db")
        for method in methods:
            g = block[block.estimator == method]
            delta = (pivot.diffusion_mean - pivot[method]).to_numpy()
            analysis.append({"schedule": schedule, "snr_db": snr, "estimator": method,
                "mean_linear_nmse_db": float(10*np.log10(g.fullband_128_nmse_linear.mean())),
                "median_nmse_db": float(g.fullband_128_nmse_db.median()),
                "fraction_nmse_below_minus10": float((g.fullband_128_nmse_db <= -10).mean()),
                "mean_pilot_net_rate": float(g.pilot_net_rate.mean()),
                "mean_seconds": float(g.estimation_and_reconstruction_seconds.mean()),
                "paired_diffusion_minus_method_db": float(delta.mean()),
                "paired_ci95": bootstrap_interval(delta, np.mean, rng=rng, resamples=1000)})
    archive.write_json("metrics_summary.json", {"rows": len(rows), "scenes": len(names),
        "split": job["split"], "smoke_only": job["smoke"], "conditional_diagnostic_only": True,
        "diffusion_guidance": settings['posterior'].get('guidance', 'denoiser_vjp'),
        "beam_subcarrier_index": int(beam_index), "beam_frequency_hz": float(platform.frequencies_hz[beam_index]),
        "rate_excludes_estimator_latency": True, "summary": analysis})
    pd.DataFrame(analysis).to_csv(archive.directory / "comparison.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    summary = pd.DataFrame(analysis)
    for ax, (schedule, block) in zip(axes, summary.groupby("schedule")):
        for method, group in block.groupby("estimator"):
            ax.plot(group.snr_db, group.mean_linear_nmse_db, "o-", label=method)
        ax.set(title=schedule, xlabel="Training-reference whitened observation SNR (dB)" if observation_axis else "Input reference SNR (dB)", ylabel="Fullband NMSE (dB)")
        ax.grid(alpha=.3)
    axes[-1].legend(fontsize=7)
    figure.savefig(archive.directory / "comparison.png", dpi=160)
    plt.close(figure)
    lines = ["# Conditional diffusion comparison", "",
        f"Split: {job['split']}; scenes: {len(names)}; smoke only: {job['smoke']}.", "",
        f"Guidance: {settings['posterior'].get('guidance', 'denoiser_vjp')}; sampler: {settings['posterior'].get('sampler', 'ald')}.", "",
        "Negative paired delta favors diffusion. Intervals resample paired scenes; a one-scene smoke interval has no statistical meaning.",
        "Rates use the common pilot frequency and exclude computation latency. Dictionary/covariance preparation is recorded separately in metrics_raw.csv.", "",
        "| Schedule | SNR | Method | Mean linear NMSE, dB | Diffusion minus method, dB [95% CI] |",
        "|---|---:|---|---:|---:|"]
    for row in analysis:
        lo, hi = row["paired_ci95"]
        lines.append(f"| {row['schedule']} | {row['snr_db']:g} | {row['estimator']} | {row['mean_linear_nmse_db']:.3f} | {row['paired_diffusion_minus_method_db']:.3f} [{lo:.3f}, {hi:.3f}] |")
    (archive.directory / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if observation_axis:
        archive.write_json("snr_axis.json", {"column": snr_column,
            "definition": "training-reference whitened pilot observation SNR",
            "protocol_noise_variances_differ": True, "physical_equal_input_noise_comparison": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/direction1_diffusion.toml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-run-id")
    parser.add_argument("--train-run-id")
    parser.add_argument("--split", choices=["validation", "test", "history"], default="validation")
    parser.add_argument("--scene-limit", type=int)
    parser.add_argument("--snr-db", type=float, nargs="+", help="evaluation subset of the frozen SNR grid")
    parser.add_argument("--observation-snr-db", type=float, nargs="+", help="diagnostic grid calibrated from training-set whitened pilot power")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for identifier in (args.run_id, args.data_run_id, args.train_run_id):
        if identifier is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            parser.error("run IDs must contain only letters, digits, underscores or hyphens")
    if args.stage in ("train", "evaluate") and not args.data_run_id:
        parser.error("this stage requires --data-run-id")
    if args.stage == "evaluate" and not args.train_run_id:
        parser.error("evaluation requires --train-run-id")
    if args.snr_db is not None and args.stage != "evaluate":
        parser.error("--snr-db applies only to evaluation")
    if args.observation_snr_db is not None:
        if args.stage != "evaluate" or args.snr_db is not None or args.smoke:
            parser.error("--observation-snr-db requires evaluation and excludes --snr-db/--smoke")
        if len(set(args.observation_snr_db)) != len(args.observation_snr_db) or any(not math.isfinite(s) for s in args.observation_snr_db):
            parser.error("observation SNR grid must be finite and unique")
    settings = read_toml(args.config.resolve())
    config = read_toml(PROJECT_ROOT / settings["base_config"])
    if args.smoke:
        settings["data"].update(train_count=4, validation_count=2, test_count=2)
        settings["training"].update(steps=2, batch_size=1, validation_every=1, validation_scenes=1)
        settings["posterior"].update(levels=2, steps_per_level=1, samples=2, prior_draws=4, validation_scenes=1)
        config["noise"]["array_reference_snr_db"] = [55.0]
    if not torch.cuda.is_available() or not settings["device"].startswith("cuda"):
        raise RuntimeError("this prototype requires the configured CUDA device")
    if config["channel"]["num_paths"] != 3 or config["system"]["num_subcarriers"] != 128 or config["dma"]["num_microstrips"] != 8:
        raise ValueError("prototype is frozen to three paths, 128 tones, eight strips")
    if min(settings["data"][s+"_count"] for s in ("train", "validation", "test")) < 1:
        raise ValueError("all data splits must be nonempty")
    for field in ("steps", "batch_size", "validation_every", "validation_scenes"):
        if settings["training"][field] < 1:
            raise ValueError(f"positive training {field} required")
    if any(not math.isfinite(settings["training"][k]) or settings["training"][k] <= 0
           for k in ("learning_rate", "gradient_clip_norm")):
        raise ValueError("positive learning rate and gradient norm required")
    if settings["training"]["validation_scenes"] > settings["data"]["validation_count"]:
        raise ValueError("validation subset exceeds the validation split")
    posterior = settings["posterior"]
    if not (math.isfinite(posterior["sigma_max"]) and 0 < posterior["sigma_min"] < posterior["sigma_max"]):
        raise ValueError("invalid training noise range")
    config["diffusion"] = settings
    defaults = {"validation": settings["posterior"]["validation_scenes"],
                "test": settings["data"]["test_count"], "history": 1 if args.smoke else 100}
    limit = defaults[args.split] if args.scene_limit is None else args.scene_limit
    if limit < 1:
        raise ValueError("positive scene limit required")
    config["job"] = {"stage": args.stage, "data_run_id": args.data_run_id,
        "train_run_id": args.train_run_id, "split": args.split, "scene_limit": limit, "smoke": args.smoke}
    if args.snr_db is not None:
        if len(set(args.snr_db)) != len(args.snr_db) or any(s not in config["noise"]["array_reference_snr_db"] for s in args.snr_db):
            raise ValueError("--snr-db must be a unique subset of the frozen SNR grid")
        config["job"]["snr_db"] = args.snr_db
    if args.observation_snr_db is not None:
        config["job"]["observation_snr_db"] = args.observation_snr_db
    if args.resume:
        archive = RunArchive(args.run_id, PROJECT_ROOT / "runs" / args.run_id)
        validate_resume(archive.directory, config)
        if json.loads(archive.status_path.read_text())["status"] == "complete":
            raise ValueError("completed runs are immutable")
        archive.write_status("running", run_id=args.run_id, resumed=True)
    else:
        archive = create_run_archive("diffusion", args.run_id, config_path=args.config.resolve(),
            config=config, seeds={"data": settings["data"], "training": settings["training"]["seed"],
                "posterior": settings["posterior"]["seed"]}, script_path=__file__,
            environment={"torch": str(torch.__version__), "cuda": str(torch.version.cuda),
                "device": torch.cuda.get_device_name(settings["device"]), "dtype": "float32/complex64"},
            checkpoint_note="Atomic stage checkpoints; same command plus --resume; smoke runs are not scientific results.\n",
            extra_manifest_files=[PROJECT_ROOT / settings["base_config"],
                PROJECT_ROOT / "scripts/run_direction1_stage2a.py", PROJECT_ROOT / "scripts/run_direction1_stage2a_linear.py"])
        manifest = json.loads((archive.directory / "code_manifest.json").read_text())
        with zipfile.ZipFile(archive.directory / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
            for entry in manifest:
                bundle.write(PROJECT_ROOT / entry["path"], entry["path"])
        patch = subprocess.run(["git", "diff", "--binary"], cwd=PROJECT_ROOT, capture_output=True, check=True)
        (archive.directory / "code_diff.patch").write_bytes(patch.stdout)
    print(f"{args.stage}: {archive.directory}", flush=True)
    try:
        {"prepare": prepare, "train": train, "evaluate": evaluate}[args.stage](archive, config)
        archive.write_status("complete", run_id=args.run_id, smoke_only=args.smoke)
    except BaseException as error:
        archive.write_status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                             run_id=args.run_id, error=str(error))
        raise


if __name__ == "__main__":
    main()
