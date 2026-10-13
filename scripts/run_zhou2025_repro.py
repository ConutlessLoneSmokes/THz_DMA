"""Prepare/train/compare published DDPM and SURE-DM on local parametric CSI."""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _run_support import PROJECT_ROOT, RunArchive, create_run_archive, hash_file, read_toml, relative_path
from run_direction1_diffusion import atomic_json, atomic_torch
from thz_dma.channels.atmosphere import AbsorptionTable
from thz_dma.observations.pilots import complex_standard_normal, full_band_pilot_indices
from thz_dma.estimators.planar_baselines import build_stage2a_operators
from thz_dma.metrics.statistics import bootstrap_interval
from thz_dma.metrics.diagnostics import channel_domain_metrics
from thz_dma.evaluation.beamforming import build_planar_data_codebook, feasible_planar_data_metrics
from zhou2025_repro.data import ChannelData, cache_directory, encode, decode, signal_scale
from zhou2025_repro.mimo import parametric_mimo, MimoOperator, qpsk_pilots
from zhou2025_repro.dma import DmaOperator, CoarseReferences
from zhou2025_repro.ddpm import DenoisingCNN, VPSchedule, noise_loss, recover
from zhou2025_repro.sure import risk, denoise as sure_denoise, matched_time
from zhou2025_repro.baselines import EmpiricalLMMSE, real_observation, lasso
from zhou2025_repro.parametric_baselines import angular_omp, mimo_path_templates, oracle_true_geometry
from zhou2025_repro.wideband import DigitalArrayOperator


def run_directory(identifier):
    if not identifier or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", identifier):
        raise ValueError("invalid run identifier")
    return PROJECT_ROOT / "runs" / identifier


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def settings_from_args(args):
    config = read_toml(args.config)
    if getattr(args, "device", None) is not None:
        config["device"] = str(torch.device(args.device))
    if "base_config" in config:
        base = read_toml(PROJECT_ROOT / config["base_config"])
        for key in ("system", "channel", "atmosphere", "dma", "acquisition", "noise",
                    "search", "adaptive_search", "estimator", "evaluation", "data_evaluation"):
            config[key] = base[key]
        config["base_config_sha256"] = hash_file(PROJECT_ROOT / config["base_config"])
    if args.smoke:
        config["data"].update(train_count=8, validation_count=4, test_count=4)
        config["network"]["width"] = max(16, config["network"]["width"] // 4)
        config["diffusion"].update(steps=8, beta_end=.3)
        config["training"].update(batch_size=4, epochs=2, validation_every=2)
        config["sure"].update(epochs=2)
        config["evaluation_run"].update(scene_limit=2, baseline_count=8,
                                        snr_db=[0., 20.], guidance=[.5, 1.],
                                        lasso_factors=[.03, .3], lasso_steps=100)
        if config["scenario"] == "dma" and config["evaluation_run"].get("frontend", "dma") == "dma":
            config["evaluation_run"]["conditional_references"] = ["oracle_geometry_ls"]
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    config["job"] = {key: value for key, value in vars(args).items() if key != "config"}
    config["job"]["config_path"] = args.config.resolve().relative_to(PROJECT_ROOT).as_posix()
    return config


def load_data(identifier):
    directory = run_directory(identifier)
    if read_json(directory / "status.json")["status"] != "complete":
        raise ValueError("data run must be complete")
    metadata = read_json(directory / "dataset.json")
    atmosphere = metadata.get("physics", {}).get("atmosphere")
    if atmosphere:
        saved = read_json(directory/"config_resolved.json")["absorption_table_sha256"]
        if hash_file(PROJECT_ROOT/atmosphere["table_path"]) != saved:
            raise ValueError("dataset absorption table changed")
    if "parent_run_id" in metadata:
        if hash_file(run_directory(metadata["parent_run_id"])/"dataset.json") != metadata["parent_dataset_sha256"]:
            raise ValueError("parent dataset manifest changed")
    if metadata.get("kind") == "denoised":
        if hash_file(run_directory(metadata["sure_train_run_id"])/"checkpoint/training.pt") != metadata["sure_checkpoint_sha256"]:
            raise ValueError("SURE denoiser checkpoint changed")
    source = directory if metadata["storage"] == "npy" else cache_directory(directory, metadata)
    for name, digest in metadata["file_sha256"].items():
        path = source / relative_path(name)
        if hash_file(path) != digest:
            raise ValueError(f"data changed: {name}")
    return ChannelData(directory, metadata)


def load_model(identifier, device):
    directory = run_directory(identifier)
    if read_json(directory / "status.json")["status"] != "complete":
        raise ValueError("trained run must be complete")
    checkpoint = torch.load(directory / "checkpoint/training.pt", map_location=device, weights_only=True)
    model = DenoisingCNN(**checkpoint["network"]).to(device=device,
        dtype=torch.float64 if checkpoint["precision"] == "float64" else torch.float32)
    model.load_state_dict(checkpoint["best_model"])
    model.eval()
    schedule = VPSchedule(**checkpoint["diffusion"]).to(device)
    data = load_data(checkpoint["data_run_id"])
    if hash_file(data.directory / "dataset.json") != checkpoint["dataset_sha256"]:
        raise ValueError("model's dataset manifest changed")
    return model, schedule, checkpoint, data


def verify_physics(config, data):
    if data.scenario != config["scenario"]:
        raise ValueError("data/model scenario mismatch")
    for key, saved in data.metadata["physics"].items():
        if config[key] != saved:
            raise ValueError(f"data physical configuration differs: {key}")


def write_arrays(archive, counts, shape, producer, batch_size):
    """Atomic progress follows flushed new arrays, replaying an unfinished batch."""
    checkpoint = archive.directory / "checkpoint/data_progress.json"
    progress = read_json(checkpoint) if checkpoint.exists() else {
        "completed": {split: 0 for split in counts}, "train_energy": 0., "train_entries": 0}
    records = archive.directory / "parameters"
    records.mkdir(exist_ok=True)
    for split, count in counts.items():
        path = archive.directory / f"channels_{split}.npy"
        array = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+",
            dtype=np.complex64, shape=(count, *shape))
        if array.shape != (count, *shape) or array.dtype != np.complex64:
            raise ValueError("partial channel array has a different shape/dtype")
        for start in range(progress["completed"][split], count, batch_size):
            indices = list(range(start, min(start + batch_size, count)))
            values, parameters = producer(split, indices)
            if values.shape != (len(indices), *shape) or not np.isfinite(values).all():
                raise ValueError("producer returned invalid physical CSI")
            array[start:start+len(indices)] = values
            array.flush()
            if parameters is not None:
                atomic_json(records / f"{split}_{start:06d}.json", parameters)
            if split == "train":
                progress["train_energy"] += float(np.sum(abs(values.astype(np.complex128))**2))
                progress["train_entries"] += values.size
            progress["completed"][split] = start + len(indices)
            atomic_json(checkpoint, progress)
            print(f"saved {split} {start+len(indices)}/{count}", flush=True)
        del array
    return progress


def save_data_manifest(archive, metadata):
    metadata["schema_version"] = 1
    if metadata["storage"] == "npy":
        files = [*metadata["files"].values(),
                 *[p.relative_to(archive.directory).as_posix() for p in sorted((archive.directory / "parameters").glob("*.json"))]]
        metadata["file_sha256"] = {name: hash_file(archive.directory / name) for name in files}
    atomic_json(archive.directory / "dataset.json", metadata)


def prepare(archive, config):
    counts = {split: config["data"][split+"_count"] for split in ("train", "validation", "test")}
    seeds = {split: config["data"][split+"_seed"] for split in counts}
    if len(set(seeds.values())) != 3:
        raise ValueError("split seeds must be distinct")
    metadata = {"kind": "clean", "scenario": config["scenario"], "counts": counts,
                "split_seeds": seeds, "known_noise_real_variance_phys": 0.}
    if config["scenario"] == "mimo":
        table = AbsorptionTable.from_txt(PROJECT_ROOT / config["atmosphere"]["table_path"],
            max_frequency_hz=config["atmosphere"]["max_model_frequency_hz"])
        shape = (config["mimo"]["receive_antennas"], config["mimo"]["transmit_antennas"])
        def producer(split, indices):
            channels, records = [], []
            for index in indices:
                h, paths = parametric_mimo(config, np.random.default_rng(
                    np.random.SeedSequence([seeds[split], index])), table)
                channels.append(h)
                records.append({"scene_id": f"{seeds[split]}:{index}", "paths": paths})
            return np.stack(channels), records
        progress = write_arrays(archive, counts, shape, producer, config["data"]["batch_size"])
        metadata.update(storage="npy", channel_shape=shape,
            files={split: f"channels_{split}.npy" for split in counts},
            complex_scale=signal_scale(progress["train_energy"], progress["train_entries"], 0.),
            physics={key: config[key] for key in ("system", "channel", "atmosphere", "mimo")})
    else:
        source = run_directory(config["data"]["source_run_id"])
        if read_json(source / "status.json")["status"] != "complete":
            raise ValueError("existing DMA cache is incomplete")
        cached = read_json(source / "dataset.json")
        source_config = read_json(source / "config_resolved.json")
        physical_config = source_config["config"]
        if hash_file(PROJECT_ROOT/config["atmosphere"]["table_path"]) != source_config["absorption_table_sha256"]:
            raise ValueError("original DMA cache absorption table changed")
        for key in ("system", "channel", "atmosphere", "dma", "acquisition"):
            if config[key] != physical_config[key]:
                raise ValueError(f"DMA cache physical mismatch: {key}")
        indices = {split: cached["index"][split][:count] for split, count in counts.items()}
        if any(len(indices[split]) != counts[split] for split in counts):
            raise ValueError("requested DMA split exceeds existing cache")
        all_names = [name for names in indices.values() for name in names]
        digests = {name: hash_file(source/name) for name in ("dataset.json", "config_resolved.json")}
        for name in all_names:
            if hash_file(source / "data" / name) != cached["file_sha256"][name]:
                raise ValueError("original DMA channel cache changed")
            digests["data/"+name] = cached["file_sha256"][name]
        metadata.update(storage="existing_npz", source_directory=source.name, source_run_id=source.name,
            index=indices, file_sha256=digests,
            split_seeds={split: physical_config["diffusion"]["data"][split+"_seed"] for split in counts},
            channel_shape=[config["system"]["num_subcarriers"],
                           config["dma"]["num_microstrips"], config["dma"]["elements_per_microstrip"]],
            complex_scale=float(cached["real_component_scale"])*math.sqrt(2),
            normalization_source="entire original training split; real RMS converted to complex RMS",
            physics={key: config[key] for key in ("system", "channel", "atmosphere", "dma", "acquisition")})
    metadata.setdefault("normalization_source", "training complex RMS only")
    save_data_manifest(archive, metadata)


def corrupt(archive, config):
    source = load_data(config["job"]["data_run_id"])
    verify_physics(config, source)
    if source.metadata["kind"] != "clean":
        raise ValueError("corruption requires the clean simulation reference")
    variance = config["job"]["noise_variance"]
    if variance is None or not np.isfinite(variance) or variance <= 0:
        raise ValueError("corrupt requires positive --noise-variance per real component")
    physical_variance = variance * source.scale**2
    seed = config["data"]["noise_seed"]
    splits = {"train": 0, "validation": 1, "test": 2}
    def producer(split, indices):
        values = source.physical(split, indices)
        for row, index in enumerate(indices):
            rng = np.random.default_rng(np.random.SeedSequence([seed, splits[split], index]))
            values[row] += np.sqrt(2*physical_variance)*complex_standard_normal(rng, source.shape)
        return values, None
    progress = write_arrays(archive, source.counts, source.shape, producer, config["data"]["batch_size"])
    metadata = {key: source.metadata[key] for key in ("scenario", "counts", "split_seeds", "channel_shape", "physics")}
    metadata.update(kind="noisy", storage="npy", parent_run_id=source.directory.name,
        parent_dataset_sha256=hash_file(source.directory / "dataset.json"),
        files={split: f"channels_{split}.npy" for split in source.counts},
        known_noise_real_variance_phys=physical_variance,
        requested_canonical_real_variance=variance,
        complex_scale=signal_scale(progress["train_energy"], progress["train_entries"], physical_variance),
        normalization_source="noisy training second moment minus known complex noise power",
        noise_seed=seed, full_csi_awgn_not_dma_lmmse_error=True)
    save_data_manifest(archive, metadata)


@torch.no_grad()
def validation_loss(model, schedule, data, config, kind, variance, time_index):
    settings = config["training"]
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    rng = torch.Generator(device=device).manual_seed(settings["validation_seed"])
    count = min(settings["validation_scenes"], data.counts["validation"])
    total = 0.
    model.eval()
    for start in range(0, count, settings["batch_size"]):
        indices = list(range(start, min(start+settings["batch_size"], count)))
        images = data.images("validation", indices, device, dtype=dtype)
        if kind == "dm":
            loss = noise_loss(model, schedule, images, generator=rng)
        else:
            probe = torch.randn(images.shape, device=device, dtype=dtype, generator=rng)
            loss = risk(lambda q: sure_denoise(model, q, variance, time_index),
                images, variance, variant=config["job"]["sure_variant"],
                epsilon=config["sure"]["epsilon"], probe=probe).mean()
        total += float(loss)*len(indices)
    return total/count


@torch.no_grad()
def validation_profile(model, schedule, data, config):
    settings = config["training"]
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    count = min(settings["validation_scenes"], data.counts["validation"])
    times = np.unique(np.linspace(0, len(schedule.beta)-1, min(5, len(schedule.beta)), dtype=int))
    rows = []
    model.eval()
    for index in times:
        generator = torch.Generator(device=device).manual_seed(settings["validation_seed"])
        noise_error, channel_error = 0., 0.
        bar = float(schedule.alpha_bar[index])
        for start in range(0, count, settings["batch_size"]):
            indices = list(range(start, min(start+settings["batch_size"], count)))
            clean = data.images("validation", indices, device, dtype=dtype)
            noise = torch.randn(clean.shape, dtype=dtype, device=device, generator=generator)
            times_batch = torch.full((len(clean),), int(index), device=device)
            noisy = schedule.perturb(clean, times_batch, noise)
            error = model(noisy, times_batch+1)-noise
            noise_error += float(error.square().mean())*len(clean)
            ratio = (1-bar)/bar*error.square().flatten(1).sum(1)/clean.square().flatten(1).sum(1)
            channel_error += float(ratio.sum())
        rows.append({"t": int(index)+1, "alpha_bar": bar, "samples": count,
                     "noise_prediction_mse": noise_error/count,
                     "denoising_nmse_linear": channel_error/count,
                     "target_distribution": data.metadata["kind"]})
    if not np.isfinite([[r["noise_prediction_mse"], r["denoising_nmse_linear"]] for r in rows]).all():
        raise ArithmeticError("nonfinite validation denoising profile")
    return rows


def train(archive, config):
    data_id = config["job"]["data_run_id"]
    data = load_data(data_id)
    verify_physics(config, data)
    kind = "sure" if config["job"]["command"] == "train-sure" else "dm"
    if kind == "sure" and data.metadata["kind"] != "noisy":
        raise ValueError("SURE training requires the separately saved noisy dataset")
    settings, device = config["training"], config["device"]
    seed = settings["seed"]
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    channels = 2 if data.scenario == "mimo" else 2*data.shape[1]
    network = {"channels": channels, "width": config["network"]["width"],
               "time_dim": config["network"]["time_dim"]}
    precision = config["sure"]["precision"] if kind == "sure" else "float32"
    if precision not in ("float32", "float64"):
        raise ValueError("unknown training precision")
    dtype = torch.float64 if precision == "float64" else torch.float32
    model = DenoisingCNN(**network).to(device=device, dtype=dtype)
    schedule = VPSchedule(**config["diffusion"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    dataset_hash = hash_file(data.directory / "dataset.json")
    variance = data.metadata["known_noise_real_variance_phys"]/data.scale**2
    time_index, target, delta = matched_time(schedule, variance) if kind == "sure" else (None, None, None)
    epochs = config["sure"]["epochs"] if kind == "sure" else settings["epochs"]
    batches = math.ceil(data.counts["train"]/settings["batch_size"])
    total_steps = epochs*batches
    checkpoint_path = archive.directory / "checkpoint/training.pt"
    start, best, best_step, history = 0, float("inf"), 0, []
    permutation = torch.empty(0, dtype=torch.long)
    best_model = copy.deepcopy(model.state_dict())
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if saved["dataset_sha256"] != dataset_hash or saved["kind"] != kind:
            raise ValueError("training input/kind changed across resume")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best, best_step = saved["step"], saved["best_loss"], saved["best_step"]
        history, best_model, permutation = saved["history"], saved["best_model"], saved["permutation"].cpu()
        torch.set_rng_state(saved["cpu_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
    started = time.perf_counter()
    for step in range(start, total_steps):
        position = step % batches
        if position == 0:
            permutation = torch.randperm(data.counts["train"])
        indices = permutation[position*settings["batch_size"]:(position+1)*settings["batch_size"]].tolist()
        images = data.images("train", indices, device, dtype=dtype)
        model.train()
        if kind == "dm":
            loss = noise_loss(model, schedule, images)
        else:
            loss = risk(lambda q: sure_denoise(model, q, variance, time_index),
                images, variance, variant=config["job"]["sure_variant"],
                epsilon=config["sure"]["epsilon"]).mean()
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        finished = step+1
        if finished % settings["validation_every"] == 0 or finished == total_steps:
            value = validation_loss(model, schedule, data, config, kind, variance, time_index)
            if not math.isfinite(value):
                raise ArithmeticError("nonfinite validation loss")
            if value < best:
                best, best_step, best_model = value, finished, copy.deepcopy(model.state_dict())
            history.append({"step": finished, "epoch": (step//batches)+1,
                            "training_loss": float(loss.detach()),
                            "validation_loss": value, "gradient_norm": float(gradient_norm)})
            atomic_torch(checkpoint_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "best_model": best_model, "best_loss": best, "best_step": best_step, "step": finished,
                "history": history, "permutation": permutation, "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
                "network": network, "diffusion": config["diffusion"], "kind": kind, "precision": precision,
                "scenario": data.scenario, "data_kind": data.metadata["kind"], "data_run_id": data_id,
                "dataset_sha256": dataset_hash, "complex_scale": data.scale,
                "sure_variant": config["job"]["sure_variant"] if kind == "sure" else None,
                "real_noise_variance": variance, "noise_time_index": time_index})
            atomic_json(archive.directory / "training_history.json", history)
            print(f"{kind} step {finished}/{total_steps}: validation_loss={value:.6g}", flush=True)
    synchronize(device)
    archive.write_json("training_summary.json", {"kind": kind, "data_kind": data.metadata["kind"],
        "trained_steps": total_steps, "epochs": epochs, "parameters": sum(p.numel() for p in model.parameters()),
        "best_step": best_step, "best_validation_loss": best, "session_seconds": time.perf_counter()-started,
        "precision": precision, "dataset_sha256": dataset_hash,
        "noise_variance_per_real_component": variance, "target_alpha_bar": target,
        "discrete_noise_schedule_delta": delta, "no_recovery_performance_claim": True,
        "validation_uses_clean_labels": kind == "dm" and data.metadata["kind"] == "clean"})
    if kind == "dm":
        model.load_state_dict(best_model)
        archive.write_json("denoising_profile.json", validation_profile(model, schedule, data, config))


def denoise_dataset(archive, config):
    source = load_data(config["job"]["data_run_id"])
    verify_physics(config, source)
    model, _, checkpoint, learned_data = load_model(config["job"]["train_run_id"][0], config["device"])
    if checkpoint["kind"] != "sure" or learned_data.directory != source.directory:
        raise ValueError("denoising must use the SURE model trained on this noisy dataset")
    variance, time_index = checkpoint["real_noise_variance"], checkpoint["noise_time_index"]
    dtype = next(model.parameters()).dtype
    @torch.no_grad()
    def producer(split, indices):
        noisy = source.images(split, indices, config["device"], dtype=dtype)
        image = sure_denoise(model, noisy/math.sqrt(1+variance), variance, time_index)
        physical = decode(image, source.scenario)*source.scale
        return physical.to(torch.complex64).cpu().numpy(), None
    progress = write_arrays(archive, source.counts, source.shape, producer, config["data"]["batch_size"])
    metadata = {key: source.metadata[key] for key in ("scenario", "counts", "split_seeds", "channel_shape", "physics")}
    metadata.update(kind="denoised", storage="npy", parent_run_id=source.directory.name,
        parent_dataset_sha256=hash_file(source.directory / "dataset.json"),
        files={split: f"channels_{split}.npy" for split in source.counts}, complex_scale=source.scale,
        normalization_source="inherited noise-corrected training scale; denoised outputs not renormalized",
        denoised_training_complex_rms=float(math.sqrt(progress["train_energy"]/progress["train_entries"])),
        sure_variant=checkpoint["sure_variant"], sure_train_run_id=config["job"]["train_run_id"][0],
        sure_checkpoint_sha256=hash_file(run_directory(config["job"]["train_run_id"][0]) / "checkpoint/training.pt"),
        known_noise_real_variance_phys=0.)
    save_data_manifest(archive, metadata)


def build_operators(config, data):
    if data.scenario == "mimo":
        result = []
        for count in config["evaluation_run"]["pilot_counts"]:
            pilots = qpsk_pilots(data.shape[-1], count, np.random.default_rng(
                np.random.SeedSequence([config["evaluation_run"]["pilot_seed"], count])),
                device=config["device"])
            result.append((f"QPSK_Np{count}", MimoOperator(pilots, data.shape[0]), None))
        return result, None
    if config["acquisition"]["pilot_energy_per_re"] != 1.:
        raise ValueError("shared DMA operator requires unit pilot energy")
    platform, _, physical = build_stage2a_operators(config, PROJECT_ROOT)
    frontend = config["evaluation_run"].get("frontend", "dma")
    if frontend == "digital":
        if config["evaluation_run"].get("conditional_references"):
            raise ValueError("digital stage does not use DMA conditional references")
        return [("digital_"+schedule.name,
                 DigitalArrayOperator(schedule.pilot_indices, data.shape,
                     repetitions=schedule.num_configurations, device=config["device"]), schedule)
                for schedule in platform.schedules], None
    if frontend != "dma":
        raise ValueError("wideband frontend must be digital or dma")
    result = [(schedule.name, DmaOperator(physical[schedule.name], schedule.pilot_indices,
               tones=data.shape[0], device=config["device"]), schedule)
              for schedule in platform.schedules]
    return result, platform


@torch.no_grad()
def fit_lmmse(data, operator, canonical_scale, config):
    count = min(config["evaluation_run"]["baseline_count"], data.counts["train"])
    chunks, reference = [], 0.
    for start in range(0, count, config["data"]["batch_size"]):
        indices = list(range(start, min(start+config["data"]["batch_size"], count)))
        images = data.images("train", indices, config["device"])*data.scale/canonical_scale
        observed = operator.forward(images)
        reference += float(observed.abs().square().double().mean())*len(indices)
        chunks.append(real_observation(observed).double()/math.sqrt(count))
    synchronize(config["device"])
    fitted = EmpiricalLMMSE(torch.cat(chunks))
    return fitted, count, reference/count


@torch.no_grad()
def lmmse_estimate(fitted, basis_data, count, operator, received, variance, canonical_scale, config):
    coefficients = fitted.coefficients(received, variance)
    image = coefficients.new_zeros((len(received), *operator.image_shape))
    for start in range(0, count, config["data"]["batch_size"]):
        indices = list(range(start, min(start+config["data"]["batch_size"], count)))
        images = basis_data.images("train", indices, config["device"]).to(coefficients)
        values = coefficients[:, start:start+len(indices)] @ images.flatten(1)
        image += values.reshape_as(image)*basis_data.scale/canonical_scale/math.sqrt(count)
    return decode(image, basis_data.scenario)*canonical_scale


def nmse(truth, estimate):
    energy = np.sum(abs(truth.astype(np.complex128))**2)
    if not np.isfinite(energy) or energy <= 0 or not np.isfinite(estimate).all():
        raise ValueError("nonzero truth energy and finite estimate required")
    return float(np.sum(abs(estimate-truth)**2)/energy)


def link_context(config, platform):
    if platform is None:
        return None
    common = set.intersection(*(set(s.pilot_indices) for s in platform.schedules))
    index = min(common, key=lambda i: abs(platform.frequencies_hz[i]-config["system"]["carrier_frequency_hz"]))
    dma = config["dma"]
    codebook, _ = build_planar_data_codebook(
        carrier_frequency_hz=float(platform.frequencies_hz[index]), feed_distances_m=platform.feed_distances_m,
        **{key: dma[key] for key in ("num_microstrips", "elements_per_microstrip",
          "tuning_bandwidth_hz", "quality_factor", "waveguide_effective_index", "waveguide_end_power_fraction")},
        codebook_size=config["data_evaluation"]["codebook_size"], seed_start=config["data_evaluation"]["codebook_seed_start"])
    return index, codebook


def metric_row(data, truth, estimate, scene_index, split, label, method, snr, variance, clean, schedule,
               config, link, elapsed, **extras):
    error = nmse(truth, estimate)
    row = {"scene_id": data.record(split, scene_index)["scene_id"], "schedule": label,
           "snr_db": snr, "estimator": method, "nmse_linear": error,
           "nmse_db": float(10*np.log10(max(error, np.finfo(float).tiny))),
           "complex_noise_variance_phys": variance*data.scale**2,
           "actual_whitened_snr_db": float(10*np.log10(float(clean.abs().square().mean())/variance)),
           "inference_batch_size": 1, "online_seconds": elapsed,
           "information_condition": "global_training_prior_no_coarse_centers",
           "split": split, "smoke_only": config["job"]["smoke"], **extras}
    if data.scenario == "dma":
        legacy_indices = full_band_pilot_indices(data.shape[0], config["evaluation"]["channel_nmse_subcarrier_count"])
        row.update(channel_domain_metrics(truth, estimate, schedule.pilot_indices, legacy_indices))
        row.update(pilot_resource_elements=schedule.pilot_resource_elements,
                   training_symbols=schedule.training_symbols, num_switches=0 if link is None else schedule.num_switches,
                   receiver_frontend=config["evaluation_run"].get("frontend", "dma"),
                   receiver_rf_outputs=(data.shape[1]*data.shape[2] if link is None else data.shape[1]))
        if link is None:
            return row
        beam_index, codebook = link
        if np.sum(abs(estimate[beam_index])**2) > 0:
            row.update(feasible_planar_data_metrics(truth[beam_index], estimate[beam_index], codebook,
                element_noise_variance=variance*data.scale**2,
                rf_noise_variance=variance*data.scale**2*config["noise"]["rf_noise_variance_fraction_of_element_noise"],
                coherence_symbols=config["data_evaluation"]["coherence_symbols"],
                training_symbols=schedule.training_symbols,
                switch_guard_symbol_equivalents=schedule.num_switches*config["acquisition"]["switch_guard_symbol_equivalents"]))
            row["data_combiner_failed"] = False
        else:
            row.update(net_rate_bps_hz=0., data_combiner_failed=True)
        row["pilot_resource_elements"] = schedule.pilot_resource_elements
        row["rate_excludes_online_solver_time"] = True
    return row


def observation_variance(data, settings, reference, snr, noise_scale=None):
    """Keep physical input noise fixed when datasets use different training scales."""
    convention = settings.get("snr_reference", "operator_output")
    if convention not in ("operator_output", "training_element"):
        raise ValueError("SNR reference must be operator_output or training_element")
    power = data.shape[-1] if data.scenario == "mimo" else (1. if convention == "training_element" else reference)
    variance = power / 10**(snr/10)
    if noise_scale is not None:
        variance *= (noise_scale/data.scale)**2
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("positive finite observation variance required")
    return variance


def dependency_record(data, models, baseline_data):
    result = {"target": {"run_id": data.directory.name, "dataset_sha256": hash_file(data.directory/"dataset.json")}}
    result["models"] = {name: hash_file(run_directory(name)/"checkpoint/training.pt") for name in models}
    result["covariance_data"] = {name: hash_file(item.directory/"dataset.json") for name, item in baseline_data.items()}
    return result


def standard_noise(values, settings, op_index, split):
    salt = 0 if split == "validation" else 1
    samples = [complex_standard_normal(np.random.default_rng(np.random.SeedSequence(
        [settings["noise_seed"], op_index, salt, index])), tuple(values.shape[1:]))
        for index in range(len(values))]
    return torch.as_tensor(np.stack(samples), device=values.device, dtype=values.dtype)


def evaluate_coarse_references(archive, config, data, operator, schedule, platform, op_index,
                              label, clean, observed_noise, reference, link):
    settings, job = config["evaluation_run"], config["job"]
    methods = settings.get("conditional_references", [])
    if not set(methods) <= {"oracle_geometry_ls", "grid_omp_3d", "sbl_jointfreq_3d", "conditional_lmmse"}:
        raise ValueError("unknown additional-information reference")
    rows = []
    for index in range(len(clean)):
        saved = archive.directory/"checkpoint"/f"coarse_{op_index}_{index}.json"
        if saved.exists():
            rows.extend(read_json(saved))
            continue
        record = data.record(job["split"], index)
        controls = CoarseReferences(operator, schedule, platform, record, config, methods,
            device=config["device"], draws=settings["conditional_prior_draws"],
            seed=np.random.SeedSequence([settings["conditional_prior_seed"], op_index, index]))
        truth = data.physical(job["split"], [index])[0]
        block = []
        for snr in settings["snr_db"]:
            variance = observation_variance(data, settings, reference, snr)
            observed = (clean[index]+math.sqrt(variance)*observed_noise[index]).cpu().numpy()*data.scale
            for method in methods:
                started = time.perf_counter()
                estimate, extras = controls.estimate(method, observed, variance*data.scale**2)
                elapsed = time.perf_counter()-started
                row = metric_row(data, truth, estimate, index, job["split"], label, "conditional_"+method,
                    snr, variance, clean[index:index+1], schedule, config, link, elapsed, **extras)
                row["information_condition"] = ("true_path_geometry_genie" if method=="oracle_geometry_ls"
                                                else "known_pre_observation_path_centers")
                block.append(row)
        atomic_json(saved, block)
        rows.extend(block)
        print(f"saved extra-information references: {label}, scene {index+1}/{len(clean)}", flush=True)
    return rows


def evaluate(archive, config):
    job, settings = config["job"], config["evaluation_run"]
    data = load_data(job["data_run_id"])
    verify_physics(config, data)
    if data.metadata["kind"] != "clean":
        raise ValueError("online performance is evaluated against a clean reference dataset")
    mimo_references = settings.get("mimo_references", [])
    if (len(set(mimo_references)) != len(mimo_references)
            or not set(mimo_references) <= {"angular_omp", "oracle_geometry_ls"}
            or (mimo_references and data.scenario != "mimo")):
        raise ValueError("unknown or duplicated MIMO reference")
    table, templates = None, {}
    if "oracle_geometry_ls" in mimo_references:
        table = AbsorptionTable.from_txt(PROJECT_ROOT / config["atmosphere"]["table_path"],
            max_frequency_hz=config["atmosphere"]["max_model_frequency_hz"])
    models, covariance_data = {}, {data.directory.name: data}
    for name in job["train_run_id"]:
        loaded = load_model(name, config["device"])
        model, schedule, checkpoint, source = loaded
        verify_physics(config, source)
        if checkpoint["kind"] != "dm" or checkpoint["diffusion"] != config["diffusion"]:
            raise ValueError("evaluate requires DDPM weights with the selected schedule")
        models[name] = loaded
        if source.metadata["kind"] == "noisy":
            covariance_data[source.directory.name] = source
        elif source.metadata["kind"] == "denoised":
            noisy = load_data(source.metadata["parent_run_id"])
            covariance_data[noisy.directory.name] = noisy
    dependencies = dependency_record(data, models, covariance_data)
    noise_scale = None
    if settings.get("noise_scale_run_id"):
        if data.scenario != "mimo":
            raise ValueError("cross-dataset noise scale is only used for the MIMO carrier control")
        noise_data = load_data(settings["noise_scale_run_id"])
        if noise_data.scenario != "mimo" or noise_data.metadata["kind"] != "clean":
            raise ValueError("noise scale reference requires clean MIMO training data")
        noise_scale = noise_data.scale
        dependencies["noise_scale_reference"] = {"run_id": noise_data.directory.name,
            "dataset_sha256": hash_file(noise_data.directory/"dataset.json"), "complex_scale": noise_scale}
        noise_data.close()
    dependency_path = archive.directory / "dependencies.json"
    if dependency_path.exists() and read_json(dependency_path) != dependencies:
        raise ValueError("evaluation dependencies changed across resume")
    atomic_json(dependency_path, dependencies)
    frozen = None
    if job["split"] == "test":
        if not job["tuning_run_id"]:
            raise ValueError("independent test requires --tuning-run-id from validation")
        tuning_directory = run_directory(job["tuning_run_id"])
        prior = read_json(tuning_directory/"config_resolved.json")["config"]
        if (read_json(tuning_directory/"status.json")["status"] != "complete"
                or prior["job"]["split"] != "validation"
                or read_json(tuning_directory/"dependencies.json") != dependencies
                or prior["evaluation_run"] != settings):
            raise ValueError("test requires completed validation with identical frozen dependencies/settings")
        frozen = read_json(tuning_directory/"tuning.json")
        tuning_source = {"run_id": job["tuning_run_id"], "sha256": hash_file(tuning_directory/"tuning.json")}
        source_path = archive.directory/"tuning_source.json"
        if source_path.exists() and read_json(source_path) != tuning_source:
            raise ValueError("frozen validation tuning changed across test resume")
        archive.write_json("tuning_source.json", tuning_source)
    tuning_path = archive.directory/"tuning.json"
    tuning = read_json(tuning_path) if tuning_path.exists() else {}
    count = min(settings["scene_limit"], data.counts[job["split"]])
    validation_count = min(settings["validation_scenes"], data.counts["validation"])
    if "oracle_geometry_ls" in mimo_references:
        for index in range(count):
            record = data.record(job["split"], index)
            if "paths" not in record:
                raise ValueError("MIMO geometry reference requires saved clean path parameters")
            templates[index] = mimo_path_templates(config, record["paths"], table)/data.scale
    op_list, platform = build_operators(config, data)
    link = link_context(config, platform)
    rows, references = [], {}
    for op_index, (label, operator, acquisition) in enumerate(op_list):
        fitted = {}
        for basis_name, basis_data in covariance_data.items():
            started = time.perf_counter()
            covariance, basis_count, reference = fit_lmmse(basis_data, operator, data.scale, config)
            fitted[basis_name] = covariance, basis_count, time.perf_counter()-started
            if basis_name == data.directory.name:
                references[label] = reference
            print(f"prepared global LMMSE: {label}, {basis_name}, training scenes={basis_count}", flush=True)
        archive.write_json("observation_reference.json", {"training_power": references,
            "scale_source": "clean training complex RMS",
            "nominal_snr": ("Nt*reference_training_scale^2/physical complex variance" if noise_scale is not None else
                "Nt/complex variance") if data.scenario=="mimo" else
                ("training mean per-element power/physical complex variance" if
                 settings.get("snr_reference") == "training_element" else
                 "training-reference whitened output power/complex variance"),
            "snr_reference": settings.get("snr_reference", "operator_output"),
            "noise_scale_reference": dependencies.get("noise_scale_reference"),
            "matched_protocol_noise_is_diagnostic_only": data.scenario=="dma" and
                settings.get("snr_reference", "operator_output") != "training_element"})
        with torch.no_grad():
            val_images = data.images("validation", range(validation_count), config["device"])
            validation_clean = operator.forward(val_images)
            eval_images = data.images(job["split"], range(count), config["device"])
            eval_clean = operator.forward(eval_images)
        validation_noise = standard_noise(validation_clean, settings, op_index, "validation")
        evaluation_noise = standard_noise(eval_clean, settings, op_index, job["split"])
        for snr_index, snr in enumerate(settings["snr_db"]):
            variance = observation_variance(data, settings, references[label], snr, noise_scale)
            observed = eval_clean+math.sqrt(variance)*evaluation_noise
            validation_y = validation_clean+math.sqrt(variance)*validation_noise
            methods = [*(f"lmmse_{name}" for name in covariance_data), *models]
            if data.scenario == "mimo":
                methods += ["ls", "lasso", *mimo_references]
            for method in methods:
                saved_path = archive.directory/"checkpoint"/f"eval_{op_index}_{snr_index}_{method}.json"
                if saved_path.exists():
                    rows.extend(read_json(saved_path))
                    continue
                tuning_key = f"{label}:{snr:g}:{method}"
                selected, model_run, convergence = None, None, None
                if method in models:
                    model, schedule, checkpoint, source = models[method]
                    ratio = data.scale/source.scale
                    def estimate(y, choice, index, split):
                        seed = int(np.random.SeedSequence([settings["posterior_seed"], op_index, index,
                            0 if split=="validation" else 1]).generate_state(1)[0])
                        image, trace = recover(model, schedule, operator, y*ratio, variance*ratio**2,
                            scale=choice, seed=seed, refinement=settings["refinement"])
                        return (decode(image, data.scenario)*source.scale).cpu().numpy(), trace
                    if frozen is None:
                        losses = []
                        truths = data.physical("validation", range(validation_count))
                        for choice in settings["guidance"]:
                            errors = [nmse(truths[i], estimate(validation_y[i:i+1], choice, i, "validation")[0][0])
                                      for i in range(validation_count)]
                            losses.append(float(np.mean(errors)))
                        selected = settings["guidance"][int(np.argmin(losses))]
                        tuning[tuning_key] = {"selected": selected, "candidates": settings["guidance"],
                                             "validation_nmse": losses}
                    else:
                        selected = frozen[tuning_key]["selected"]
                        tuning[tuning_key] = frozen[tuning_key]
                    model_run = method
                elif method == "lasso":
                    penalties = (np.asarray(settings["lasso_factors"])*math.sqrt(variance)
                        *float(torch.linalg.vector_norm(operator.b, dim=1).max())*math.sqrt(math.log(np.prod(data.shape))))
                    if frozen is None:
                        losses = []
                        for penalty in penalties:
                            angle, _, _ = lasso(validation_y, operator.b, float(penalty), steps=settings["lasso_steps"])
                            image = torch.stack((angle.real, angle.imag), dim=1)
                            estimates = (decode(image, "mimo")*data.scale).cpu().numpy()
                            truths = data.physical("validation", range(validation_count))
                            losses.append(float(np.mean([nmse(h, e) for h, e in zip(truths, estimates)])))
                        selected = float(penalties[int(np.argmin(losses))])
                        tuning[tuning_key] = {"selected": selected, "candidates": penalties.tolist(), "validation_nmse": losses}
                    else:
                        selected = frozen[tuning_key]["selected"]
                        tuning[tuning_key] = frozen[tuning_key]
                scene_rows = []
                for index in range(count):
                    synchronize(config["device"])
                    started = time.perf_counter()
                    trace, covariance_seconds, diagnostics = [], 0., {}
                    with torch.no_grad():
                        if method in models:
                            estimates, trace = estimate(observed[index:index+1], selected, index, job["split"])
                            prediction = estimates[0]
                        elif method.startswith("lmmse_"):
                            name = method[len("lmmse_"):]
                            covariance, basis_count, covariance_seconds = fitted[name]
                            prediction = lmmse_estimate(covariance, covariance_data[name], basis_count, operator,
                                observed[index:index+1], variance, data.scale, config)[0].cpu().numpy()
                        else:
                            if method == "ls":
                                angle = observed[index:index+1]@torch.linalg.pinv(operator.b)
                            elif method == "angular_omp":
                                angle, fit = angular_omp(observed[index:index+1], operator.b,
                                    max_atoms=settings["omp_max_atoms"])
                                diagnostics = {"information_condition": "declared_DFT_bin_sparsity_no_true_geometry",
                                    "algorithm_iterations": fit["max_atoms"],
                                    "support_condition_number": float(np.max(fit["support_condition_number"]))}
                            elif method == "oracle_geometry_ls":
                                angle, fit = oracle_true_geometry(observed[index:index+1], operator.b, templates[index])
                                diagnostics = {"information_condition": "true_path_geometry_genie_unknown_gains",
                                    "algorithm_iterations": 1,
                                    "support_condition_number": float(fit["support_condition_number"][0])}
                            else:
                                angle, iterations, convergence = lasso(observed[index:index+1], operator.b, selected,
                                                                       steps=settings["lasso_steps"])
                            image = torch.stack((angle.real, angle.imag), dim=1)
                            prediction = (decode(image, "mimo")*data.scale)[0].cpu().numpy()
                    synchronize(config["device"])
                    elapsed = time.perf_counter()-started
                    truth = data.physical(job["split"], [index])[0]
                    scene_rows.append(metric_row(data, truth, prediction, index, job["split"], label, method,
                        snr, variance, eval_clean[index:index+1], acquisition, config, link, elapsed,
                        model_run_id=model_run, guidance=selected if method in models else None,
                        lasso_penalty=selected if method=="lasso" else None,
                        lasso_converged=bool(convergence) if method=="lasso" else None,
                        covariance_preparation_seconds=covariance_seconds,
                        network_calls=trace[-1]["network_calls"] if trace else 0,
                        **diagnostics,
                        actual_rf_output_snr_db=float(10*np.log10(
                            float((eval_clean[index:index+1].abs().square()*operator.relative_variance).sum())
                            /(variance*float(operator.relative_variance.sum())))) if acquisition is not None else None))
                atomic_json(tuning_path, tuning)
                atomic_json(saved_path, scene_rows)
                rows.extend(scene_rows)
                archive.write_metrics(rows, sorted({key for row in rows for key in row}))
                print(f"saved {label}, {snr:g} dB, {method}: NMSE={10*np.log10(np.mean([r['nmse_linear'] for r in scene_rows])):.3f} dB",
                      flush=True)
        if data.scenario == "dma" and settings.get("conditional_references"):
            rows.extend(evaluate_coarse_references(archive, config, data, operator, acquisition, platform,
                op_index, label, eval_clean, evaluation_noise, references[label], link))
    extra_count = 2+len(mimo_references) if data.scenario=="mimo" else len(settings.get("conditional_references", []))
    expected = count*len(op_list)*len(settings["snr_db"])*(len(covariance_data)+len(models)+extra_count)
    if len(rows) != expected:
        raise ValueError(f"incomplete evaluation rows: {len(rows)} != {expected}")
    summarize(archive, rows, config)


def summarize(archive, rows, config):
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(["scene_id", "schedule", "snr_db", "estimator"]).any()
    summary, paired = [], []
    for (schedule, snr, method), block in frame.groupby(["schedule", "snr_db", "estimator"]):
        summary.append({"schedule": schedule, "snr_db": float(snr), "estimator": method,
            "nmse_db": float(10*np.log10(block.nmse_linear.mean())),
            "mean_batch1_seconds": float(block.online_seconds.mean()), "scenes": len(block)})
    baseline = "lmmse_"+config["job"]["data_run_id"]
    rng = np.random.default_rng(config["evaluation_run"]["bootstrap_seed"])
    for (schedule, snr), block in frame.groupby(["schedule", "snr_db"]):
        pivot = block.pivot(index="scene_id", columns="estimator", values="nmse_db")
        for method in config["job"]["train_run_id"]:
            differences = (pivot[method]-pivot[baseline]).to_numpy()
            paired.append({"schedule": schedule, "snr_db": float(snr), "model": method,
                "mean_difference_db": float(differences.mean()),
                "ci95": bootstrap_interval(differences, np.mean, rng=rng, resamples=1000)})
    archive.write_metrics(rows, sorted({key for row in rows for key in row}))
    archive.write_json("metrics_summary.json", {"rows": len(rows), "smoke_only": config["job"]["smoke"],
        "split": config["job"]["split"], "confidence_unit": "paired channel scenes; training seeds are separate models",
        "summary": summary, "paired_vs_clean_global_lmmse": paired})
    pd.DataFrame(summary).to_csv(archive.directory/"comparison.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = frame.schedule.unique()
    figure, axes = plt.subplots(1, len(labels), figsize=(6*len(labels), 4), squeeze=False, constrained_layout=True)
    for ax, label in zip(axes[0], labels):
        for method, block in pd.DataFrame(summary).query("schedule == @label").groupby("estimator"):
            ax.plot(block.snr_db, block.nmse_db, "o-", label=method)
        ax.set(title=label, xlabel="Declared observation SNR (dB)", ylabel="NMSE (mean sample ratio, dB)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=6)
    figure.savefig(archive.directory/"comparison.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "corrupt", "train", "train-sure", "denoise", "evaluate"])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT/"configs/zhou2025_parametric_mimo.toml")
    parser.add_argument("--device", help="override the configured device, e.g. cpu or cuda:0")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-run-id")
    parser.add_argument("--train-run-id", nargs="+", default=[])
    parser.add_argument("--noise-variance", type=float)
    parser.add_argument("--sure-variant", choices=["paper", "corrected"], default="corrected")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--tuning-run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command != "prepare" and not args.data_run_id:
        parser.error("this command requires --data-run-id")
    if args.command == "denoise" and not args.train_run_id:
        parser.error("this command requires --train-run-id")
    if args.command == "denoise" and len(args.train_run_id) != 1:
        parser.error("denoise requires exactly one SURE training run")
    config = settings_from_args(args)
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("configured CUDA device is unavailable")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    directory = run_directory(args.run_id)
    if args.resume:
        archive = RunArchive(args.run_id, directory)
        saved = read_json(directory/"config_resolved.json")["config"]
        saved["job"]["resume"] = True
        if saved != config:
            raise ValueError("resume configuration differs from the archived job")
        for item in read_json(directory/"code_manifest.json"):
            if hash_file(PROJECT_ROOT/relative_path(item["path"])) != item["sha256"]:
                raise ValueError(f"resume source changed: {item['path']}")
        if read_json(archive.status_path)["status"] == "complete":
            raise ValueError("completed runs are immutable; use a new run ID")
        archive.write_status("running", resumed=True)
    else:
        archive = create_run_archive("zhou2025", args.run_id, config_path=args.config.resolve(),
            config=config, seeds={"data": config["data"], "training": config["training"]["seed"],
            "noise": config["evaluation_run"]["noise_seed"], "posterior": config["evaluation_run"]["posterior_seed"]},
            script_path=__file__, environment={"torch": str(torch.__version__), "cuda": str(torch.version.cuda),
            "device": config["device"], "gpu": torch.cuda.get_device_name(config["device"])
            if config["device"].startswith("cuda") else "CPU"},
            checkpoint_note="Atomic data/training/evaluation checkpoints; resume only identical code/config/dependencies.\n",
            extra_manifest_files=(PROJECT_ROOT/"scripts/run_direction1_diffusion.py",
                                  PROJECT_ROOT/"scripts/run_direction1_stage2a.py"))
        with zipfile.ZipFile(directory/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
            for item in read_json(directory/"code_manifest.json"):
                snapshot.write(PROJECT_ROOT/relative_path(item["path"]), relative_path(item["path"]).as_posix())
    handlers = {"prepare": prepare, "corrupt": corrupt, "train": train, "train-sure": train,
                "denoise": denoise_dataset, "evaluate": evaluate}
    try:
        handlers[args.command](archive, config)
        archive.write_status("complete", command=args.command, smoke_only=args.smoke)
    except BaseException as error:
        archive.write_status("failed", command=args.command, error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
