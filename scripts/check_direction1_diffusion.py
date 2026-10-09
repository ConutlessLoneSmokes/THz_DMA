"""Audit D-032 archives and diagnose two validation scenes without changing the model."""
import argparse
import json
from pathlib import Path
import zipfile

import run_direction1_diffusion as r
import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="diffusion_val_20261002_v1")
    args = parser.parse_args()
    root = r.PROJECT_ROOT / "runs"
    directory = root / args.run_id
    config = json.loads((directory / "config_resolved.json").read_text())["config"]
    assert config["job"]["split"] == "validation" and not config["job"]["smoke"]
    data_dir, train_dir = (root / config["job"][k] for k in ("data_run_id", "train_run_id"))
    output = directory / "analysis"
    output.mkdir(exist_ok=True)
    checks = {}
    for folder in (directory, data_dir, train_dir):
        assert json.loads((folder / "status.json").read_text())["status"] == "complete"
        manifest = json.loads((folder / "code_manifest.json").read_text())
        with zipfile.ZipFile(folder / "source_snapshot.zip") as snapshot:
            import hashlib
            for entry in manifest:
                assert r.hash_file(r.PROJECT_ROOT / entry["path"]) == entry["sha256"], entry["path"]
                assert hashlib.sha256(snapshot.read(entry["path"].replace("\\", "/"))).hexdigest() == entry["sha256"]
        checks[folder.name+"_manifest_files"] = len(manifest)
    data_dir, data = r.require_dataset(data_dir.name, config)
    dependencies = json.loads((directory / "dependencies.json").read_text())
    assert dependencies["dataset_sha256"] == r.hash_file(data_dir / "dataset.json")
    assert dependencies["model_sha256"] == r.hash_file(train_dir / "checkpoint/training.pt")
    ids, train_energy, components = {}, 0., 0
    for split, names in data["index"].items():
        ids[split] = set()
        for name in names:
            channel, record = r.read_example(data_dir, name)
            assert channel.shape == (128, 8, 160) and np.isfinite(channel).all()
            assert record["scene_id"] not in ids[split]
            ids[split].add(record["scene_id"])
            if split == "train":
                train_energy += float(np.sum(abs(channel.astype(np.complex128))**2))
                components += 2*channel.size
    splits = list(ids)
    assert all(not ids[a] & ids[b] for i, a in enumerate(splits) for b in splits[i+1:])
    np.testing.assert_allclose(np.sqrt(train_energy/components), data["real_component_scale"], rtol=1e-12)
    checks["split_counts"] = {k: len(v) for k, v in ids.items()}
    checks["train_only_scale"] = data["real_component_scale"]
    key = ["scene_id", "schedule", "array_reference_snr_db", "estimator"]
    frame = pd.read_csv(directory / "metrics_raw.csv")
    methods = ["oracle_geometry_ls", "grid_omp_3d", "sbl_jointfreq_3d", "lmmse", "diffusion_mean"]
    expected = {(s["name"], snr, method) for s in config["acquisition"]["schedules"]
        for snr in config["noise"]["array_reference_snr_db"] for method in methods}
    assert len(frame) == config["job"]["scene_limit"]*len(expected) == 960
    assert frame.scene_id.nunique() == 16 and not frame.duplicated(key).any()
    assert set(frame.scene_id) <= ids["validation"]
    assert all(set(g[key[1:]].itertuples(index=False, name=None)) == expected for _,g in frame.groupby("scene_id"))
    saved = [row for p in sorted((directory / "checkpoint").glob("*.json")) for row in json.loads(p.read_text())]
    pd.testing.assert_frame_equal(frame.sort_values(key).reset_index(drop=True),
        pd.DataFrame(saved).sort_values(key).reset_index(drop=True).replace("", np.nan),
        check_dtype=False, atol=1e-12, rtol=1e-12)
    assert frame.groupby(key[:3]).effective_output_snr_db.nunique().eq(1).all()
    for col in ("fullband_128_nmse_linear", "pilot_nmse_linear", "pilot_beam_loss_db", "pilot_net_rate",
                "estimation_and_reconstruction_seconds"):
        assert np.isfinite(frame[col]).all()
    f2 = frame.schedule.str.contains("J8")
    assert frame.loc[~f2, "heldout_112_nmse_linear"].isna().all()
    assert np.isfinite(frame.loc[f2, "heldout_112_nmse_linear"]).all()
    checks.update(rows=len(frame), sbl_converged=int(frame.loc[frame.estimator=="sbl_jointfreq_3d", "algorithm_converged"].sum()))
    history = pd.read_json(train_dir / "training_history.json")
    checkpoint = torch.load(train_dir / "checkpoint/training.pt", map_location=config["diffusion"]["device"], weights_only=True)
    assert checkpoint["step"] == 10000 and len(history) == 40
    assert checkpoint["best_step"] == int(history.loc[history.validation_noise_mse.idxmin(), "step"])
    checks.update(best_step=checkpoint["best_step"], best_noise_mse=checkpoint["best_loss"])
    records = []
    for (schedule,snr,method), group in frame.groupby(key[1:]):
        records.append(dict(schedule=schedule, snr_db=snr, estimator=method,
            nmse_db=10*np.log10(group.fullband_128_nmse_linear.mean()),
            success_count=int((group.fullband_128_nmse_db <= -10).sum()),
            positive_nmse_count=int((group.fullband_128_nmse_db > 0).sum()),
            worst_nmse_db=float(group.fullband_128_nmse_db.max()),
            heldout_db=float(10*np.log10(group.heldout_112_nmse_linear.mean())) if "J8" in schedule else None,
            beam_loss_db=group.pilot_beam_loss_db.mean(), net_rate=group.pilot_net_rate.mean(),
            seconds=group.estimation_and_reconstruction_seconds.mean(),
            preparation_seconds=(group.dictionary_preparation_seconds+group.covariance_preparation_seconds).mean()))
    pd.DataFrame(records).to_csv(output / "audit_table.csv", index=False)
    r.write_json(output / "integrity.json", checks)
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    # Use only training/validation examples; test CSI is read above only for integrity.
    settings = config["diffusion"]
    model = r.ConditionalScoreNet(width=settings["network"]["width"]).to(settings["device"])
    model.load_state_dict(checkpoint["best_model"])
    model.eval()
    denoising = []
    with torch.no_grad():
        for split in ("train", "validation"):
            for name in data["index"][split][:4]:
                clean, centers = r.load_batch(data_dir, [name], data["real_component_scale"], config, settings["device"])
                generator = torch.Generator(device=settings["device"]).manual_seed(2026100401)
                for sigma in np.geomspace(.01, 3., 5):
                    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
                    noisy = clean+sigma*noise
                    prediction = model(noisy, clean.new_full((1,), sigma), centers)
                    denoised = noisy-sigma*prediction
                    shuffled = model(noisy, clean.new_full((1,), sigma), -centers)
                    denoising.append(dict(split=split, name=name, sigma=sigma,
                        noise_mse=float((prediction-noise).square().mean()),
                        noisy_nmse=float((noisy-clean).square().sum()/clean.square().sum()),
                        denoised_nmse=float((denoised-clean).square().sum()/clean.square().sum()),
                        changed_condition_noise_mse=float((shuffled-noise).square().mean())))
    pd.DataFrame(denoising).to_csv(output / "denoising_probe.csv", index=False)
    print("Per-scale denoising probe saved", flush=True)
    platform, weights, operators = r.build_stage2a_operators(config, r.PROJECT_ROOT)
    probes = []
    for name in data["index"]["validation"][:2]:
        _, record = r.read_example(data_dir, name)
        seed, sample = map(int,record["scene_id"].split(":"))
        truth = r.record_channel(record, platform, config)
        scale = data["real_component_scale"]
        condition = torch.tensor(r.center_condition(record["centers_m_rad_rad"], config), device=settings["device"])[None]
        for si, schedule in enumerate(platform.schedules):
            operator = operators[schedule.name]
            op = r.TorchDmaOperator(operator, schedule.pilot_indices, device=settings["device"])
            variance = r.array_input_noise_variance(truth,55)
            nu = variance/scale**2
            obs = r.observe_multistrip_schedule(weights[schedule.name], truth[schedule.pilot_indices],
                pilot_energy_per_re=1., element_noise_variance=variance,
                rf_noise_variance=variance*config["noise"]["rf_noise_variance_fraction_of_element_noise"],
                rng=np.random.default_rng(np.random.SeedSequence([seed,sample,si,2026090921])))
            y = torch.tensor((operator.whiten(obs.observed)/scale).reshape(operator.observation_shape),device=settings["device"],dtype=torch.complex64)[None]
            snri = config["noise"]["array_reference_snr_db"].index(55.)
            ps = int(np.random.SeedSequence([settings["posterior"]["seed"],seed,sample,si,snri]).generate_state(1)[0])
            for label, nsteps, yy, cc in (("replay",5,y,condition),("steps20",20,y,condition),
                    ("zero_observation",5,torch.zeros_like(y),condition),("changed_condition",5,y,-condition)):
                cfg = {**settings["posterior"], "steps_per_level": nsteps}
                samples, trace = r.posterior_samples(model,op,yy,cc,nu,cfg,seed=ps)
                mean = samples.mean(0)
                estimate = mean.cpu().numpy()*scale
                nmse = np.sum(abs(estimate-truth)**2)/np.sum(abs(truth)**2)
                residual = float((op.forward(r.pack_channel(mean[None]))-y).abs().square().mean())
                row = dict(scene_id=record["scene_id"],schedule=schedule.name,variant=label,nmse_db=10*np.log10(nmse),
                    estimate_energy_over_truth=float(np.sum(abs(estimate)**2)/np.sum(abs(truth)**2)),
                    correlation=float(abs(np.vdot(estimate,truth))/np.linalg.norm(estimate)/np.linalg.norm(truth)),
                    mean_residual_over_noise=residual/nu,
                    chain_final_residual_over_noise=trace[-1]["residual_mean_squared"]/nu)
                if label == "replay":
                    baseline = estimate.copy()
                    old = frame[(frame.scene_id==record["scene_id"]) & (frame.schedule==schedule.name)
                        & (frame.array_reference_snr_db==55) & (frame.estimator=="diffusion_mean")].iloc[0]
                    np.testing.assert_allclose(row["nmse_db"],old.fullband_128_nmse_db,atol=1e-4)
                row["relative_change_from_replay"] = float(np.linalg.norm(estimate-baseline)/np.linalg.norm(baseline))
                probes.append(row)
                pd.DataFrame(probes).to_csv(output / "posterior_probe.csv", index=False)
                print(record["scene_id"],schedule.name,label,round(row["nmse_db"],4),flush=True)
    r.write_json(output / "audit_provenance.json", {"script_sha256":r.hash_file(Path(__file__)),
        "dependencies":dependencies,"metric_sha256":r.hash_file(directory/"metrics_raw.csv"),
        "diagnostic_only":True,"denoising_scenes_per_split":4,"posterior_validation_scenes":2,
        "condition_control":"negative normalized centers; diagnostic perturbation, not a physical paired population"})


if __name__ == "__main__":
    main()
