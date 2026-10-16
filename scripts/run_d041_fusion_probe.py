"""Validation-only mean-prior diagnostics, guidance boundary checks and scalar fusion."""
import argparse
import copy
import hashlib
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
import torch

from _run_support import PROJECT_ROOT, RunArchive, create_run_archive, hash_file, read_toml
import run_d041_budget_sweep as budget
import run_d041_configuration_sweep as sweep
import run_d041_diagnostic as diagnostic
from run_direction1_diffusion import atomic_json
from run_zhou2025_repro import load_model, read_json, run_directory, synchronize, verify_physics
from zhou2025_repro.data import decode, encode
from zhou2025_repro.ddpm import recover
from zhou2025_repro.d041 import diagnostics, paired_array, state_bank
from zhou2025_repro.moment_recovery import clean_prediction, gaussian_control, recover_moments


def resolved_settings(args):
    config = copy.deepcopy(read_json(run_directory(args.data_run_id)/"config_resolved.json")["config"])
    settings = read_toml(args.config)
    if args.smoke:
        settings.update(calibration_train_scenes=2, prior_validation_scenes=1, tuning_scenes=1,
                        screening_start=1, screening_scenes=1, guidance=[4., 16.])
    if (config["band"] != "wide" or config["channel"]["include_absorption"] or
            settings["screening_start"] < settings["tuning_scenes"] or
            settings["prior_validation_scenes"] > settings["tuning_scenes"] or
            any(type(settings[k]) is not int or settings[k] < 1 for k in
                ("calibration_train_scenes", "prior_validation_scenes", "tuning_scenes", "screening_scenes")) or
            not settings["guidance"] or any(not math.isfinite(v) or v <= 0 for v in settings["guidance"]) or
            len(set(settings["guidance"])) != len(settings["guidance"]) or
            not math.isfinite(settings["variance_floor_real"]) or settings["variance_floor_real"] <= 0 or
            settings["noiseless_variances_complex"] != [1e-8, 1e-10] or settings["noisy_snr_db"] != 20. or
            [(r["budget_re"], r["label"], r["noisy"]) for r in settings["operators"]] != [
                (8192, "re8192_J64_K128_distinct", True), (4096, "re4096_J64_K64_distinct", True),
                (4096, "re4096_J128_K32_distinct", False), (4096, "re4096_J128_K32_interleaved", False),
                (128, "re128_J16_K8_distinct", False)]):
        raise ValueError("predeclared wide/noabs validation protocol, positive settings and disjoint scenes required")
    config["fusion_probe"], config["device"] = settings, args.device
    config["job"] = {**vars(args), "config": args.config.relative_to(PROJECT_ROOT).as_posix()}
    for name in (args.run_id, args.data_run_id, args.train_run_id):
        run_directory(name)
    return config


def signature(config):
    return {k: config[k] for k in ("band", "system", "channel", "atmosphere", "dma", "noise",
                                   "acquisition", "diagnostic", "diffusion", "data", "fusion_probe")}


def verify_run(folder):
    if read_json(folder/"status.json")["status"] != "complete":
        raise ValueError(f"completed parent required: {folder.name}")
    for name, digest in read_json(folder/"manifest.json").items():
        if hash_file(folder/name) != digest:
            raise ValueError(f"saved output changed: {folder.name}/{name}")
    for item in read_json(folder/"code_manifest.json"):
        if hash_file(PROJECT_ROOT/item["path"]) != item["sha256"]:
            raise ValueError(f"source changed since parent: {item['path']}")


def operators(config):
    for re in dict.fromkeys(r["budget_re"] for r in config["fusion_probe"]["operators"]):
        block = copy.deepcopy(config)
        block["configuration_sweep"] = {**read_toml(PROJECT_ROOT/"configs/d041_budget_sweep.toml"),
                                       "experiment": "budget", "budget_re": re}
        block["job"].update(smoke=False, frozen_operator_labels=[r["label"]
            for r in config["fusion_probe"]["operators"] if r["budget_re"] == re])
        yield from budget.operator_groups(block, "full-noiseless")


def metrics(op, truth, estimate, observed, clean):
    result = diagnostics(op, truth, estimate, observed, clean)
    for name, selected in [("observed_frequency_nmse", op.active.any(0)),
                           ("unobserved_frequency_nmse", ~op.active.any(0))]:
        mask = torch.as_tensor(selected, device=truth.device)
        if mask.any():
            result[name] = float((estimate[:, mask]-truth[:, mask]).abs().square().sum()/
                                 truth[:, mask].abs().square().sum())
        else:
            result[name] = None
    if any(not math.isfinite(v) for v in result.values() if v is not None):
        raise ArithmeticError("nonfinite recovery diagnostic")
    return result


def common_noise(op, config, index):
    # Draw the original full RE grid, then gather; the old full-tone noise helper
    # slices configuration axis 2 and cannot be applied to compact RE axis 2.
    seed = int(np.random.SeedSequence([config["evaluation_run"]["noise_seed"], 1, index, 0]).generate_state(1)[0])
    standard = torch.randn((1, op.channel_shape[1], 192, op.channel_shape[0]),
        dtype=op.matrix.dtype, device=op.matrix.device,
        generator=torch.Generator(device=op.matrix.device).manual_seed(seed))
    return standard[:, :, :op.active.shape[0]].flatten(2).index_select(2, op.indices)


def conditions(settings, record):
    result = [(f"noiseless_v{v:g}", None, v) for v in settings["noiseless_variances_complex"]]
    if record["noisy"]:
        result.append(("snr_20", 20., None))
    return result


@torch.no_grad()
def calibrate(archive, config, model, schedule, data):
    settings, device = config["fusion_probe"], config["device"]
    archive.write_json("gaussian_checks.json", gaussian_control())
    groups = list(operators(config))
    batch_size = config["data"]["batch_size"]
    selected_times = set(settings["trace_times"]+[len(schedule.beta)])
    for t in range(1, len(schedule.beta)+1):
        path = archive.directory/"checkpoint"/f"calibration_t{t:03d}.json"
        if path.exists():
            continue
        rows, decomposed, training_mse = [], [], []
        bar = float(schedule.alpha_bar[t-1])
        for split, count in [("train", settings["calibration_train_scenes"]),
                             ("validation", settings["prior_validation_scenes"])]:
            for start in range(0, count, batch_size):
                indices = list(range(start, min(count, start+batch_size)))
                images = data.images(split, indices, device)
                noise = torch.stack([torch.randn(images.shape[1:], dtype=images.dtype, device=device,
                    generator=torch.Generator(device=device).manual_seed(int(np.random.SeedSequence([
                        settings["calibration_seed"], 0 if split == "train" else 1, t, i]).generate_state(1)[0]))) for i in indices])
                state = math.sqrt(bar)*images+math.sqrt(1-bar)*noise
                predicted = clean_prediction(state, model(state, torch.full((len(indices),), t, device=device)), bar)
                error = (predicted-images).square().flatten(1).double().mean(1)
                if not torch.isfinite(error).all():
                    raise ArithmeticError("nonfinite offline denoising calibration")
                if split == "train":
                    training_mse.extend(error.tolist())
                for j, index in enumerate(indices):
                    truth, estimate = decode(images[j:j+1], "dma"), decode(predicted[j:j+1], "dma")
                    energy = float(truth.abs().square().sum())
                    row = {"split": split, "scene_index": index, "t": t,
                        "real_coordinate_mse": float(error[j]),
                        "nmse_linear": float((estimate-truth).abs().square().sum())/energy,
                        "prediction_energy_ratio": float(estimate.abs().square().sum())/energy,
                        "normalized_correlation": float((truth.conj()*estimate).sum().abs()/
                            (truth.abs().square().sum()*estimate.abs().square().sum()).clamp_min(1e-30).sqrt()),
                        "source": "complete CSI with artificial VP noise; not a DMA estimator"}
                    rows.append(row)
                    if split == "validation" and t in selected_times:
                        for label, op, _ in groups:
                            clean = op.forward(images[j:j+1])
                            decomposed.append({**row, "operator": label,
                                **metrics(op, truth, estimate, clean, clean)})
        value = max(float(np.mean(training_mse)), settings["variance_floor_real"])
        atomic_json(path, {"t": t, "variance_per_real_component": value,
                           "train_scenes": len(training_mse), "prior_rows": rows, "decomposed": decomposed})
        print(f"calibrated VP step {t}/{len(schedule.beta)}", flush=True)
    blocks = [read_json(archive.directory/"checkpoint"/f"calibration_t{t:03d}.json")
              for t in range(1, len(schedule.beta)+1)]
    archive.write_json("calibration_profile.json", {"variances_real": [r["variance_per_real_component"] for r in blocks],
        "training_scenes": settings["calibration_train_scenes"], "representation": "training RMS + unitary spatial DFT",
        "target": "training-average squared error around frozen denoiser; not exact conditional covariance",
        "validation_not_used_for_variance": True, "variance_floor_real": settings["variance_floor_real"]})
    pd.DataFrame([r for b in blocks for r in b["prior_rows"]]).to_csv(archive.directory/"prior_denoising.csv", index=False)
    pd.DataFrame([r for b in blocks for r in b["decomposed"]]).to_csv(archive.directory/"prior_decomposition.csv", index=False)


class ObservedDenoiser(torch.nn.Module):
    def __init__(self, model, schedule, callback, times):
        super().__init__()
        self.model, self.schedule, self.callback, self.times = model, schedule, callback, times
        self.calls = 0

    def forward(self, state, time_index):
        epsilon = self.model(state, time_index)
        self.calls += 1
        t = int(time_index[0])
        if t in self.times:
            mean = clean_prediction(state, epsilon, float(self.schedule.alpha_bar[t-1]))
            self.callback(mean, {"t": t, "network_calls": self.calls})
        return epsilon


def select_guidance(rows, candidates):
    table = pd.DataFrame(rows)
    choices = []
    for choice in candidates:
        block = table.loc[(table.method == "zhou") & (table.guidance == choice)]
        valid = len(block) > 0 and (block.status == "success").all()
        choices.append({"guidance": choice, "valid": bool(valid),
                        "mean_nmse_linear": float(block.nmse_linear.mean()) if valid else None})
    valid = [r for r in choices if r["valid"]]
    if not valid:
        raise ArithmeticError("no finite guidance candidate; retain failed cases and stop")
    best = min(valid, key=lambda r: r["mean_nmse_linear"])
    return {"selected": best["guidance"], "candidates": choices,
            "at_upper_boundary": best["guidance"] == max(candidates)}


def worst_variance_summary(table, tuning):
    rows = []
    for (label, method), block in table.loc[table.condition.str.startswith("noiseless_")].groupby(["operator", "method"]):
        if method == "zhou":
            block = block.loc[[r.guidance == tuning[f"{label}:{r.condition}"]["selected"]
                               for r in block.itertuples()]]
        failures, values, gaps = 0, [], []
        for _, scene in block.groupby("scene_index"):
            if len(scene) != 2:
                raise ValueError("two numerical variances per paired scene required")
            if (scene.status != "success").any():
                failures += 1
                continue
            values.append(float(scene.nmse_linear.max()))
            gaps.append(float(10*np.log10(scene.nmse_linear.clip(lower=1e-30)).max()-
                              10*np.log10(scene.nmse_linear.clip(lower=1e-30)).min()))
        count = len(values)+failures
        rows.append({"operator": label, "method": method, "scenes": count, "failed_scenes": failures,
            "worst_variance_mean_nmse_db": None if failures else 10*math.log10(max(float(np.mean(values)), 1e-30)),
            "fraction_below_minus10db": sum(v <= .1 for v in values)/count,
            "maximum_numerical_variance_gap_db": max(gaps) if gaps else None,
            "numerical_variance_stable": not failures and max(gaps) <= .1})
    return rows


@torch.no_grad()
def evaluate(archive, config, model, schedule, data):
    job, settings = config["job"], config["fusion_probe"]
    calibration = read_json(run_directory(job["calibration_run_id"])/"calibration_profile.json")
    frozen = read_json(run_directory(job["tuning_run_id"])/"tuning.json") if job["stage"] == "screen" else None
    indices = (range(settings["tuning_scenes"]) if frozen is None else
               range(settings["screening_start"], settings["screening_start"]+settings["screening_scenes"]))
    times = set(settings["trace_times"]+[len(schedule.beta)])
    records, references, all_cases, tuning = [], {}, [], {}
    hardware = {"element_positions_m": paired_array(config).positions, "resonances_hz": state_bank(config)}
    by_label = {r["label"]: r for r in settings["operators"]}
    for label, op, weights in operators(config):
        records.append(sweep.operator_record(archive, label, op, weights))
        hardware[label+"_active_re_mask"] = op.active
        hardware[label+"_state_indices"] = np.asarray(op.sweep_metadata["state_indices"])
        power = diagnostic.training_reference(data, op, config)
        references[label] = {"training_scenes": data.counts["train"], "normalized_whitened_power": power,
                             "physical_whitened_power": power*data.scale**2}
        for condition, snr, floor in conditions(settings, by_label[label]):
            variance = floor if snr is None else power/10**(snr/10)
            key, case_rows = f"{label}:{condition}", []
            choices = settings["guidance"] if frozen is None else [frozen[key]["selected"]]
            methods = [("zhou", q) for q in choices]+[("scalar_fusion", 1.), ("pinv_ls", 0.)]
            for index in indices:
                truth = torch.as_tensor(data.physical("validation", [index]), device=config["device"])/data.scale
                clean = op.forward(encode(truth, "dma"))
                observed = clean if snr is None else clean+math.sqrt(variance)*common_noise(op, config, index)
                observation_hash = hashlib.sha256(observed.cpu().numpy().tobytes()).hexdigest()
                for method, strength in methods:
                    path = archive.directory/"checkpoint"/f"case_{label}_{condition}_{index:04d}_{method}_{strength:g}.json"
                    if path.exists():
                        case = read_json(path)
                        if case["row"]["observation_sha256"] != observation_hash:
                            raise ValueError("resumed observation differs")
                    else:
                        seed = diagnostic.posterior_seed(config, label, index, "validation")
                        common = {"operator": label, "condition": condition, "phase": job["stage"],
                            "data_split": "validation", "scene_index": index,
                            "scene_id": data.record("validation", index)["scene_id"], "method": method,
                            "guidance": strength if method == "zhou" else None,
                            "actual_whitened_snr_db": None if snr is None else 10*math.log10(float(clean.abs().square().mean())/variance),
                            "realized_whitened_snr_db": None if snr is None else 10*math.log10(float(clean.abs().square().sum()/(observed-clean).abs().square().sum())),
                            "target_whitened_snr_db": snr, "likelihood_complex_variance": variance,
                            "observation_complex_variance": 0. if snr is None else variance,
                            "physical_element_noise_complex_variance": 0. if snr is None else variance*data.scale**2,
                            "observation_sha256": observation_hash, "posterior_seed": seed,
                            "pilot_re": op.active.sum().item(), "network_calls": 0}
                        trace, denoiser = [], []
                        def observer(state, info):
                            estimate = decode(state, "dma")/math.sqrt(info["state_alpha_bar"])
                            trace.append({**info, **metrics(op, truth, estimate, observed, clean)})
                        def mean_observer(mean, info):
                            denoiser.append({**info, **metrics(op, truth, decode(mean, "dma"), observed, clean)})
                        wrapper = ObservedDenoiser(model, schedule, mean_observer, times)
                        synchronize(config["device"])
                        started = time.perf_counter()
                        try:
                            if method == "zhou":
                                image, _ = recover(wrapper, schedule, op, observed, variance, scale=strength,
                                                    seed=seed, refinement=1, trace_observer=observer)
                                estimate = decode(image, "dma")
                            elif method == "scalar_fusion":
                                image = recover_moments(wrapper, schedule, op, observed, variance,
                                    variances_real=calibration["variances_real"], seed=seed, trace_observer=observer)
                                estimate = decode(image, "dma")
                            else:
                                estimate = op.least_squares(observed)
                            checked = metrics(op, truth, estimate, observed, clean)
                            row = {**common, **checked, "status": "success"}
                        except ArithmeticError as error:
                            row = {**common, "status": "nonfinite", "failure": str(error), "nmse_linear": None}
                        synchronize(config["device"])
                        row.update(network_calls=wrapper.calls, online_seconds_with_diagnostics=time.perf_counter()-started)
                        case = {"row": row, "trajectory": trace, "denoiser": denoiser}
                        atomic_json(path, case)
                    all_cases.append(case)
                    case_rows.append(case["row"])
            if frozen is None:
                tuning[key] = select_guidance(case_rows, settings["guidance"])
            else:
                tuning[key] = frozen[key]
            archive.write_json("tuning.json", tuning)
            print(f"evaluated {job['stage']} {key}", flush=True)
    np.savez_compressed(archive.directory/"hardware.npz", **hardware)
    archive.write_json("operators.json", records)
    archive.write_json("training_reference_power.json", references)
    archive.write_metrics([c["row"] for c in all_cases], fieldnames=list(dict.fromkeys(k for c in all_cases for k in c["row"])))
    for name, field in [("trajectories.csv", "trajectory"), ("denoiser_predictions.csv", "denoiser")]:
        pd.DataFrame([{**{k: c["row"][k] for k in ["operator", "condition", "scene_id", "scene_index", "method", "guidance"]}, **r}
                      for c in all_cases for r in c[field]]).to_csv(archive.directory/name, index=False)
    table = pd.DataFrame([c["row"] for c in all_cases])
    summaries = []
    for (label, condition, method), group in table.groupby(["operator", "condition", "method"]):
        if method == "zhou":
            group = group.loc[group.guidance == tuning[f"{label}:{condition}"]["selected"]]
        ok = group.status == "success"
        summaries.append({"operator": label, "condition": condition, "method": method, "scenes": len(group),
            "failed_scenes": int((~ok).sum()), "mean_nmse_db": 10*math.log10(max(float(group.nmse_linear.mean()), 1e-30)) if ok.all() else None,
            "fraction_below_minus10db": float((ok & (group.nmse_linear <= .1)).sum()/len(group))})
    pd.DataFrame(summaries).to_csv(archive.directory/"summary.csv", index=False)
    pd.DataFrame(worst_variance_summary(table, tuning)).to_csv(archive.directory/"worst_variance_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["calibrate", "tune", "screen"])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT/"configs/d041_fusion_probe.toml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-run-id", default="d041_noabs_20261013_v2_wb_data")
    parser.add_argument("--train-run-id", default="d041_noabs_20261013_v2_wb_clean")
    parser.add_argument("--calibration-run-id")
    parser.add_argument("--tuning-run-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="small counts; no training or model-step changes")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.config = args.config.resolve()
    config = resolved_settings(args)
    if args.stage != "calibrate" and not args.calibration_run_id:
        parser.error("evaluation requires --calibration-run-id")
    if args.stage == "screen" and not args.tuning_run_id:
        parser.error("screen requires --tuning-run-id")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model, schedule, checkpoint, data = load_model(args.train_run_id, args.device)
    if (checkpoint["kind"] != "dm" or data.metadata["kind"] != "clean" or
            data.directory != run_directory(args.data_run_id) or checkpoint["diffusion"] != config["diffusion"]):
        raise ValueError("frozen matched Clean DDPM and original data required")
    verify_physics(config, data)
    settings = config["fusion_probe"]
    if (settings["calibration_train_scenes"] > data.counts["train"] or
            settings["screening_start"]+settings["screening_scenes"] > data.counts["validation"]):
        raise ValueError("declared scene counts exceed source data")
    dependencies = diagnostic.evaluation_dependencies(config, data)
    parents = {}
    for identifier, stage in [(args.calibration_run_id, "calibrate"), (args.tuning_run_id, "tune")]:
        if identifier:
            folder = run_directory(identifier)
            verify_run(folder)
            previous = read_json(folder/"config_resolved.json")["config"]
            if (previous["job"]["stage"] != stage or signature(previous) != signature(config) or
                    read_json(folder/"dependencies.json") != dependencies):
                raise ValueError("parent protocol or data/model dependencies differ")
            parents[stage] = {"run_id": identifier, "manifest_sha256": hash_file(folder/"manifest.json")}
    if args.stage == "screen":
        if read_json(run_directory(args.tuning_run_id)/"parents.json")["calibrate"] != parents["calibrate"]:
            raise ValueError("screen must use the calibration frozen by tuning")
    directory = run_directory(args.run_id)
    if args.resume and directory.exists():
        archive = RunArchive(args.run_id, directory)
        previous = read_json(directory/"config_resolved.json")["config"]
        previous["job"]["resume"] = True
        if previous != config or read_json(directory/"dependencies.json") != dependencies or read_json(directory/"parents.json") != parents:
            raise ValueError("resume inputs differ")
        for item in read_json(directory/"code_manifest.json"):
            if hash_file(PROJECT_ROOT/item["path"]) != item["sha256"]:
                raise ValueError("resume source differs")
        if read_json(archive.status_path)["status"] == "complete":
            verify_run(directory)
            print(f"Completed fusion probe retained: {args.run_id}", flush=True)
            return
        archive.write_status("running", resumed=True)
    else:
        extra = [PROJECT_ROOT/"scripts"/name for name in ["run_d041_budget_sweep.py", "run_d041_configuration_sweep.py",
            "run_d041_diagnostic.py", "run_zhou2025_repro.py", "run_direction1_diffusion.py", "run_d041_fusion_linux.sh"]]
        extra += [PROJECT_ROOT/"configs/d041_budget_sweep.toml"]
        archive = create_run_archive("d041_fusion", args.run_id, config_path=args.config, config=config,
            script_path=__file__, seeds={"offline": settings["calibration_seed"], "online": config["evaluation_run"]},
            environment={"torch": str(torch.__version__), "cuda": str(torch.version.cuda), "device": args.device},
            checkpoint_note="Validation-only probes; parent data/models and all old runs stay read-only.\n", extra_manifest_files=extra)
        archive.write_json("dependencies.json", dependencies)
        archive.write_json("parents.json", parents)
        with zipfile.ZipFile(directory/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
            for item in read_json(directory/"code_manifest.json"):
                snapshot.write(PROJECT_ROOT/item["path"], item["path"])
    try:
        if args.stage == "calibrate":
            calibrate(archive, config, model, schedule, data)
        else:
            evaluate(archive, config, model, schedule, data)
        archive.write_status("complete", stage=args.stage, smoke_only=args.smoke, data_split="train/validation only")
        archive.write_json("manifest.json", {p.relative_to(directory).as_posix(): hash_file(p)
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "manifest.json"})
    except BaseException as error:
        archive.write_status("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
