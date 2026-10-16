#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

BUDGET_PREFIX=${BUDGET_PREFIX:-d041_budget_sweep_20261015_v1}
BUDGET_SOURCE=${BUDGET_SOURCE:-d041_noabs_20261013_v2}
BUDGET_DEVICE=${BUDGET_DEVICE:-cuda:0}
BUDGET_SMOKE=${BUDGET_SMOKE:-0}
BUDGET_RESUME=${BUDGET_RESUME:-0}
export MPLBACKEND=Agg
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/d041_matplotlib}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

evaluate_budget() {
    local budget_re="$1" budget_split="$2" budget_tag
    budget_tag="re${budget_re}"
    if [[ "$budget_re" == 0 ]]; then budget_tag=references; fi
    local budget_extra=()
    if [[ "$BUDGET_SMOKE" == 1 ]]; then budget_extra+=(--smoke); fi
    if [[ "$BUDGET_RESUME" == 1 ]]; then budget_extra+=(--resume); fi
    if [[ "$budget_split" == test ]]; then
        budget_extra+=(--tuning-run-id "${BUDGET_PREFIX}_${budget_tag}_validation")
    fi
    python scripts/run_d041_budget_sweep.py --budget-re "$budget_re" \
        --run-id "${BUDGET_PREFIX}_${budget_tag}_${budget_split}" \
        --data-run-id "${BUDGET_SOURCE}_wb_data" --train-run-id "${BUDGET_SOURCE}_wb_clean" \
        --device "$BUDGET_DEVICE" --split "$budget_split" "${budget_extra[@]}"
}

run_split() {
    local budget_split="$1" budget_re
    if [[ "$budget_split" == test && ! -f "results/d041_budget_sweep/${BUDGET_PREFIX}_validation/manifest.json" ]]; then
        echo "Complete all validation budgets and freeze the validation report before test." >&2
        return 2
    fi
    if [[ "$budget_split" == test ]]; then
        python scripts/analyze_d041_budget_sweep.py --prefix "$BUDGET_PREFIX" --split validation --resume
    fi
    for budget_re in 0 128 512 1024 2048 4096 8192; do
        evaluate_budget "$budget_re" "$budget_split"
    done
    local report_extra=()
    if [[ "$BUDGET_RESUME" == 1 ]]; then report_extra+=(--resume); fi
    python scripts/analyze_d041_budget_sweep.py --prefix "$BUDGET_PREFIX" --split "$budget_split" "${report_extra[@]}"
}

case "${1:-all}" in
    validation) run_split validation ;;
    test) run_split test ;;
    all) run_split validation; run_split test ;;
    *) echo "Usage: $0 {validation|test|all}" >&2; exit 2 ;;
esac
