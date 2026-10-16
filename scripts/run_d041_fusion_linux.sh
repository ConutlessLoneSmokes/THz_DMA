#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
stage="${1:-all}"
case "$stage" in all|calibrate|tune|screen) ;; *) echo "usage: $0 [all|calibrate|tune|screen]" >&2; exit 2 ;; esac
prefix="${FUSION_PREFIX:-d041_fusion_probe_20261016_v1}"
source_prefix="${FUSION_SOURCE:-d041_noabs_20261013_v2}"
device="${FUSION_DEVICE:-cuda:0}"
flags=()
if [[ "${FUSION_SMOKE:-0}" == 1 ]]; then flags+=(--smoke); fi
if [[ "${FUSION_RESUME:-0}" == 1 ]]; then flags+=(--resume); fi
base=(--data-run-id "${source_prefix}_wb_data" --train-run-id "${source_prefix}_wb_clean" --device "$device")
if [[ "$stage" == all || "$stage" == calibrate ]]; then
    python scripts/run_d041_fusion_probe.py calibrate --run-id "${prefix}_calibrate" "${base[@]}" "${flags[@]}"
fi
if [[ "$stage" == all || "$stage" == tune ]]; then
    python scripts/run_d041_fusion_probe.py tune --run-id "${prefix}_tune" --calibration-run-id "${prefix}_calibrate" "${base[@]}" "${flags[@]}"
fi
if [[ "$stage" == all || "$stage" == screen ]]; then
    python scripts/run_d041_fusion_probe.py screen --run-id "${prefix}_screen" --calibration-run-id "${prefix}_calibrate" --tuning-run-id "${prefix}_tune" "${base[@]}" "${flags[@]}"
fi
