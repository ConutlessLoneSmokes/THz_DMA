"""Twenty preselected GPU endpoints and six offline-denoising records; no tuning."""
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

from audit import ROOT, OUT, folder, read, METRICS
from _run_support import write_json
import run_d041_fusion_probe as probe
import run_d041_diagnostic as diagnostic
from run_zhou2025_repro import load_model, synchronize
from zhou2025_repro.data import decode, encode
from zhou2025_repro.ddpm import recover
from zhou2025_repro.moment_recovery import clean_prediction, recover_moments


@torch.no_grad()
def main():
    device = "cuda:0"
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    config = read(folder("screen")/"config_resolved.json")["config"]
    config["device"] = device
    model, schedule, checkpoint, data = load_model(config["job"]["train_run_id"], device)
    profile = read(folder("calibrate")/"calibration_profile.json")["variances_real"]
    screen = pd.read_csv(folder("screen")/"metrics_raw.csv")
    tuning = pd.read_csv(folder("tune")/"metrics_raw.csv")
    selected = screen.loc[((screen.scene_index == 32) & (screen.condition == "noiseless_v1e-08") |
                          (screen.scene_index.isin([32, 95]) & (screen.condition == "snr_20"))) &
                         screen.method.isin(["zhou", "scalar_fusion"])].copy()
    extra = tuning.loc[(tuning.operator == "re8192_J64_K128_distinct") & (tuning.condition == "noiseless_v1e-08") &
                       (tuning.scene_index == 0) & (tuning.method == "zhou") & tuning.guidance.isin([8., 16.])]
    selected = pd.concat([selected, extra], ignore_index=True)
    assert len(selected) == 20
    result = []
    for label, op, _ in probe.operators(config):
        for row in selected.loc[selected.operator == label].itertuples():
            truth = torch.as_tensor(data.physical("validation", [row.scene_index]), device=device)/data.scale
            clean = op.forward(encode(truth, "dma"))
            observed = clean if row.condition.startswith("noiseless_") else clean+math.sqrt(row.observation_complex_variance)*probe.common_noise(op, config, row.scene_index)
            assert hashlib.sha256(observed.cpu().numpy().tobytes()).hexdigest() == row.observation_sha256
            synchronize(device)
            started = time.perf_counter()
            if row.method == "zhou":
                image, _ = recover(model, schedule, op, observed, row.likelihood_complex_variance,
                    scale=row.guidance, seed=int(row.posterior_seed), refinement=1)
            else:
                image = recover_moments(model, schedule, op, observed, row.likelihood_complex_variance,
                    variances_real=profile, seed=int(row.posterior_seed))
            actual = probe.metrics(op, truth, decode(image, "dma"), observed, clean)
            np.testing.assert_allclose([actual[n] for n in METRICS], [getattr(row, n) for n in METRICS], rtol=2e-5, atol=1e-7)
            result.append({"operator": label, "condition": row.condition, "method": row.method, "scene_index": row.scene_index,
                "guidance": row.guidance if row.method == "zhou" else None,
                "absolute_nmse_difference": abs(actual["nmse_linear"]-row.nmse_linear), "seconds": time.perf_counter()-started})
            print(f"replayed {len(result)}/20 {label} {row.condition} {row.method}", flush=True)
    prior = pd.read_csv(folder("calibrate")/"prior_denoising.csv")
    prior_checks = []
    for split, index in [("train", 0), ("validation", 31)]:
        for t in (1, 50, 100):
            # Match the calibration batch shape as well as per-scene RNG streams.
            start = index//config["data"]["batch_size"]*config["data"]["batch_size"]
            indices = list(range(start, start+config["data"]["batch_size"]))
            images = data.images(split, indices, device)
            noise = torch.stack([torch.randn(images.shape[1:], dtype=images.dtype, device=device,
                generator=torch.Generator(device=device).manual_seed(int(np.random.SeedSequence([
                    config["fusion_probe"]["calibration_seed"], 0 if split == "train" else 1, t, i]).generate_state(1)[0]))) for i in indices])
            bar = float(schedule.alpha_bar[t-1])
            state = math.sqrt(bar)*images+math.sqrt(1-bar)*noise
            predicted = clean_prediction(state, model(state, torch.full((len(indices),), t, device=device)), bar)
            mse = float((predicted-images).square().flatten(1).double().mean(1)[index-start])
            saved = prior.loc[(prior.split == split) & (prior.scene_index == index) & (prior.t == t)].iloc[0]
            np.testing.assert_allclose(mse, saved.real_coordinate_mse, rtol=2e-5, atol=1e-9)
            prior_checks.append({"split": split, "index": index, "t": t, "relative_mse_difference": abs(mse-saved.real_coordinate_mse)/saved.real_coordinate_mse})
    write_json(OUT/"gpu_replays.json", {"passed": True, "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "endpoint_rtol": 2e-5, "endpoint_atol": 1e-7, "endpoints": result, "offline_denoising": prior_checks})
    data.close()


if __name__ == "__main__":
    main()
