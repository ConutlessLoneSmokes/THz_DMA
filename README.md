# AI Edge–Enabled THz DMA/RHS Research

Reproducible simulations for low-overhead wideband channel acquisition with
terahertz dynamic metasurface antennas. The repository contains the existing
Stage 0–2-A experiments and the local Zhou DDPM/SURE-DM reproduction.

## Current results

The completed D-039 migration has 20 formal stages and 144000 validation/test
rows. Original 300 GHz parametric MIMO gains persist. Full-tone digital DDPM
reaches −32.29 dB NMSE at declared 30 dB, compared with −29.90 dB direct LS.
Sparse-tone digital recovery remains at −1.48 dB, and full-CSI DMA recovery is
approximately 0 dB. The 40 GHz LMMSE baseline is stronger than DDPM.

These are matched local simulations with one training seed. The original
QuaDRiGa distribution, hardware measurements and multi-seed results are not
reproduced. Conditional DMA baselines have extra path-center information.
New carrier LASSO curves retain their iteration-budget limitation.

Read [START_HERE.md](START_HERE.md) for the current task and
[19i](research_route/19i_方向一传统参照与分步场景迁移.md) for the results,
digital/DMA observation difference and purposes of the proposed diagnostics.
On 2026-10-13 the user adopted D-041: disable molecular absorption and isolate
DMA combining, wideband structure and their interaction with a matched-array
narrowband/wideband × digital/DMA control. The matched-array narrowband DMA
case comes first. Begin without observation noise, then align received SNR.
The new control has not been implemented or run. Nonuniform pilot scans,
covariance-sample sweeps, SURE migration and new seeds are deferred.
Read [CODEX_START_PROMPT.md](CODEX_START_PROMPT.md) for the continuation prompt.

## Layout

| Path | Role |
|---|---|
| src/thz_dma/ | Shared physical channels, surfaces, observations, estimators and metrics; existing stage runners |
| src/zhou2025_repro/ | DDPM/SURE-DM, MIMO and wideband digital/DMA adapters; reuses shared physics |
| scripts/ | Prepare, train, evaluate, audit and diagnose; _run_support.py owns shared archival |
| configs/ | Original and diagnostic TOML configurations |
| data/atmosphere/ | Versioned physical table and provenance |
| tests/ | Operators, scaling, SURE, frozen evaluation, resume and portable manifest checks |
| research_route/ | Models, literature evidence, decisions and preserved research history |
| docs/LINUX_RUN.md | Linux environment, CPU smoke check, GPU placement and artifact transfer |

## Linux

Use Python 3.10 and the pinned direct dependencies in requirements-linux.txt.
Install the desired PyTorch wheel before the remaining project dependencies:

    python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -r requirements-linux.txt -e .
    MPLBACKEND=Agg python -m unittest discover -s tests -v

The [Linux guide](docs/LINUX_RUN.md) provides environment creation and four
self-contained CPU smoke commands, including a frozen independent test.
The [Linux checks workflow](.github/workflows/linux-checks.yml) performs those
checks on Ubuntu 26.04. Its status is visible in
[GitHub Actions](https://github.com/ConutlessLoneSmokes/THz_DMA/actions).

Each CUDA process uses one GPU. CUDA_VISIBLE_DEVICES=0 or =1 keeps its configured
device cuda:0; there is no DDP or automatic pooling of GPU memory.
The existing FP32 DDPM and FP64 SURE settings are preserved.

## Evidence and large artifacts

The [2026-10-13 snapshot](research_route/experiment_results/2026-10-13/README.md)
contains D-037–D-039 formal records, raw metrics, frozen choices, audits,
figures and original source snapshots. Byte hashes accompany the copied files.
The earlier [40-run snapshot](research_route/experiment_results/2026-10-09/README.md)
preserves VE/EDM diagnostics, including failed and interrupted runs.

Large CSI caches, model checkpoints, third-party datasets, PDFs and author
code remain local. large_artifacts.json in the new snapshot lists the inputs
to transfer separately into runs/, with their original paths and hashes.
The portable loader accepts Windows manifest paths while retaining the hash
checks. Completed inputs can feed a new evaluation ID; historical jobs cannot
resume across changed source versions. The Linux guide explains the transfer.

Every experiment creates a new immutable runs/ directory. Read
[AGENTS.md](AGENTS.md) and [the document map](research_route/README.md) before
changing a physical model, research boundary or experiment decision.
