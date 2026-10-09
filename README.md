# AI Edge–Enabled THz DMA/RHS Research

Research code and route documents for low-overhead wideband channel acquisition with terahertz dynamic metasurface antennas (DMA). The current implementation covers Direction 1 simulations from Stage 0 through the Stage 2-A diagnostics.

## Repository layout

- `src/thz_dma/stage*.py`: experiment orchestration and stable run entry points.
- `src/thz_dma/`: channel generation, acquisition schedules, estimators, evaluation,
  metrics, and analysis live in their matching subpackages; stage runners reuse them.
- `scripts/`: simulation, analysis, and diagnostic entry points; `_run_support.py`
  owns their shared archival and environment-recording logic.
- `configs/`: reproducible TOML configurations.
- `data/atmosphere/`: the absorption-coefficient table used by the simulations and its provenance notes.
- `research_route/`: research route, model, literature evidence, experiment records, and decision log.
- `tests/`: small standard-library checks for shared infrastructure.
- `START_HERE.md`: current project state and the next research task.

Run artifacts, paper PDFs, extracted full text, and presentation assets remain local because they are large or have separate redistribution terms.

Compact experiment evidence is versioned under `research_route/experiment_results/`.
The [2026-10-09 snapshot](research_route/experiment_results/2026-10-09/README.md)
contains 40 run records, available CSV results, configurations, source hashes,
audit reports, plots, and the current 19-test result. It preserves failed and
interrupted runs. Large channel caches, model weights, and third-party datasets
remain local; the snapshot documents what is needed to reproduce the runs.

The latest learned diffusion recovery improves on v4 but remains below conditional
LMMSE. Read [the diagnosis and proposed next experiment](research_route/19e_方向一学习先验差距与下一改进方案.md)
before starting a new training run.

## Setup

```bash
python -m venv .venv
python -m pip install -e .
```

Install the optional CUDA backend when needed:

```bash
python -m pip install -e ".[cuda]"
```

Run the lightweight checks with:

```bash
python -m unittest discover -s tests -v
```

## Run

Every run writes a new immutable directory under `runs/`. For example:

```bash
python scripts/run_direction1_stage2a.py \
  --config configs/direction1_stage2a_adaptive_300ghz.toml \
  --run-id stage2a_adaptive_example
```

Read `START_HERE.md` and `research_route/README.md` before extending the model or changing a research decision.
