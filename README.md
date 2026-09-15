# AI Edge–Enabled THz DMA/RHS Research

Research code and route documents for low-overhead wideband channel acquisition with terahertz dynamic metasurface antennas (DMA). The current implementation covers Direction 1 simulations from Stage 0 through the Stage 2-A diagnostics.

## Repository layout

- `src/thz_dma/`: channel, DMA, observation, estimator, evaluation, and analysis modules.
- `scripts/`: simulation, analysis, and diagnostic entry points; `_run_support.py`
  owns their shared archival and environment-recording logic.
- `configs/`: reproducible TOML configurations.
- `data/atmosphere/`: the absorption-coefficient table used by the simulations and its provenance notes.
- `research_route/`: research route, model, literature evidence, experiment records, and decision log.
- `tests/`: small standard-library checks for shared infrastructure.
- `START_HERE.md`: current project state and the next research task.

Run artifacts, paper PDFs, extracted full text, and presentation assets remain local because they are large or have separate redistribution terms.

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
