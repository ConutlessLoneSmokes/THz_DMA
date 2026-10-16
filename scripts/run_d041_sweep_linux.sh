#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SWEEP_PREFIX=${SWEEP_PREFIX:-d041_config_sweep_20261014_v1}
SWEEP_SOURCE=${SWEEP_SOURCE:-d041_noabs_20261013_v2}
SWEEP_DEVICE=${SWEEP_DEVICE:-cuda:0}
SWEEP_SMOKE=${SWEEP_SMOKE:-0}
SWEEP_RESUME=${SWEEP_RESUME:-0}
export MPLBACKEND=Agg
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/d041_matplotlib}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

evaluate() {
    local sweep_experiment="$1" sweep_band="$2" sweep_tag="$3" sweep_split="$4"
    local sweep_run="${SWEEP_PREFIX}_${sweep_tag}_${sweep_experiment}_${sweep_split}"
    local sweep_extra=()
    if [[ "$SWEEP_SMOKE" == 1 ]]; then sweep_extra+=(--smoke); fi
    if [[ "$sweep_split" == test ]]; then
        sweep_extra+=(--tuning-run-id "${SWEEP_PREFIX}_${sweep_tag}_${sweep_experiment}_validation")
    fi
    if [[ "$SWEEP_RESUME" == 1 && -d "runs/$sweep_run" ]]; then
        local sweep_status
        sweep_status=$(python - "$sweep_run" "$SWEEP_SMOKE" "$SWEEP_DEVICE" "$SWEEP_SOURCE" "$sweep_tag" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path.cwd()
directory = root/"runs"/sys.argv[1]
config = json.loads((directory/"config_resolved.json").read_text())["config"]
assert config["job"]["smoke"] == (sys.argv[2] == "1"), "smoke/formal differs"
assert config["device"] == sys.argv[3], "resume device differs"
assert config["job"]["data_run_id"] == f"{sys.argv[4]}_{sys.argv[5]}_data", "source differs"
for item in json.loads((directory/"code_manifest.json").read_text()):
    assert hashlib.sha256((root/item["path"]).read_bytes()).hexdigest() == item["sha256"], "source changed; use a new prefix"
print(json.loads((directory/"status.json").read_text())["status"])
PY
)
        if [[ "$sweep_status" == complete ]]; then
            echo "Completed sweep retained: $sweep_run"
            return
        fi
        sweep_extra+=(--resume)
    fi
    python scripts/run_d041_configuration_sweep.py --experiment "$sweep_experiment" --band "$sweep_band" \
        --run-id "$sweep_run" --data-run-id "${SWEEP_SOURCE}_${sweep_tag}_data" \
        --train-run-id "${SWEEP_SOURCE}_${sweep_tag}_clean" --device "$SWEEP_DEVICE" \
        --split "$sweep_split" "${sweep_extra[@]}"
}

pair() {
    evaluate "$1" "$2" "$3" validation
    evaluate "$1" "$2" "$3" test
}

case "${1:-all}" in
    curve-narrow) pair curve narrow nb ;;
    curve-wide) pair curve wide wb ;;
    re128) pair re128 wide wb ;;
    all) pair curve narrow nb; pair curve wide wb; pair re128 wide wb ;;
    *) echo "Usage: $0 {curve-narrow|curve-wide|re128|all}" >&2; exit 2 ;;
esac
