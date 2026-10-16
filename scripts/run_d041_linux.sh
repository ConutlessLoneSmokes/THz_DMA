#!/usr/bin/env bash
# User-started D-041 stages. No package installation, SSH or config-file edits.
set -euo pipefail
cd "$(dirname "$0")/.."
D041_PREFIX="${D041_PREFIX:-d041_noabs_20261013_v1}"
D041_DEVICE="${D041_DEVICE:-cuda:0}"
D041_SMOKE="${D041_SMOKE:-0}"
D041_RESUME="${D041_RESUME:-0}"
D041_PHASE="${1:-all}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/d041_matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
case "$D041_PHASE" in
    all|noiseless|noisy|narrow-noiseless|wide-noiseless|narrow-noisy|wide-noisy) ;;
    narrow) D041_PHASE=narrow-noiseless ;;
    wide) D041_PHASE=wide-noiseless ;;
    *) echo "Usage: bash scripts/run_d041_linux.sh [narrow-noiseless|wide-noiseless|noiseless|narrow-noisy|wide-noisy|noisy|all]" >&2; exit 2 ;;
esac
python - "$D041_DEVICE" <<'PY'
import sys, torch
if sys.argv[1].startswith("cuda"):
    assert torch.cuda.is_available(), "CUDA unavailable; verify driver and project environment first"
print("D-041 device:", sys.argv[1], "torch:", torch.__version__)
PY

run_stage() {
    local d041_run_id="$1"
    shift
    local d041_extra=()
    if [[ "$D041_SMOKE" == 1 ]]; then d041_extra+=(--smoke); fi
    if [[ -d "runs/$d041_run_id" ]]; then
        if [[ "$D041_RESUME" != 1 ]]; then
            echo "Run exists: $d041_run_id; use a new prefix or D041_RESUME=1 for unchanged runs." >&2
            exit 1
        fi
        local d041_status
        d041_status=$(python - "$d041_run_id" "$D041_SMOKE" "$D041_DEVICE" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path.cwd()
directory = root/"runs"/sys.argv[1]
config = json.loads((directory/"config_resolved.json").read_text())["config"]
assert config["job"]["smoke"] == (sys.argv[2] == "1"), "smoke/formal prefix mismatch"
assert config["device"] == sys.argv[3], "resume device differs"
for item in json.loads((directory/"code_manifest.json").read_text()):
    assert hashlib.sha256((root/pathlib.Path(item["path"])).read_bytes()).hexdigest() == item["sha256"], "source changed; use a new prefix"
print(json.loads((directory/"status.json").read_text())["status"])
PY
        )
        if [[ "$d041_status" == complete ]]; then
            echo "Verified completed run: $d041_run_id"
            return
        fi
        d041_extra+=(--resume)
    fi
    python scripts/run_d041_diagnostic.py "$@" --device "$D041_DEVICE" --run-id "$d041_run_id" "${d041_extra[@]}"
}

evaluate_band() {
    local d041_band="$1" d041_tag="$2"
    shift 2
    local d041_data="${D041_PREFIX}_${d041_tag}_data"
    local d041_model="${D041_PREFIX}_${d041_tag}_clean"
    for d041_stage in "$@"; do
        local d041_stage_tag="${d041_stage%%-*}"
        local d041_validation="${D041_PREFIX}_${d041_tag}_${d041_stage_tag}_val"
        run_stage "$d041_validation" evaluate --band "$d041_band" --stage "$d041_stage" \
            --data-run-id "$d041_data" --train-run-id "$d041_model" --split validation
        run_stage "${D041_PREFIX}_${d041_tag}_${d041_stage_tag}_test" evaluate --band "$d041_band" --stage "$d041_stage" \
            --data-run-id "$d041_data" --train-run-id "$d041_model" --split test --tuning-run-id "$d041_validation"
    done
}

noiseless_band() {
    local d041_band="$1" d041_tag="$2"
    local d041_data="${D041_PREFIX}_${d041_tag}_data"
    run_stage "$d041_data" prepare --band "$d041_band"
    run_stage "${D041_PREFIX}_${d041_tag}_check" check --band "$d041_band" --data-run-id "$d041_data"
    run_stage "${D041_PREFIX}_${d041_tag}_clean" train --band "$d041_band" --data-run-id "$d041_data"
    evaluate_band "$d041_band" "$d041_tag" full-noiseless reduced-noiseless
}

require_noiseless_controls() {
    # Both bandwidths must have frozen noise-free controls before noisy evaluation.
    local d041_tags=("$@")
    if [[ ${#d041_tags[@]} == 0 ]]; then d041_tags=(nb wb); fi
    python - "$D041_PREFIX" "$D041_SMOKE" "$D041_DEVICE" "${d041_tags[@]}" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path.cwd()
for tag in sys.argv[4:]:
    for suffix in ("data", "check", "clean", "full_val", "full_test", "reduced_val", "reduced_test"):
        directory = root/"runs"/f"{sys.argv[1]}_{tag}_{suffix}"
        if not (directory/"status.json").exists():
            raise SystemExit("Complete both noiseless controls first: missing " + directory.name)
        status = json.loads((directory/"status.json").read_text())
        if status["status"] != "complete":
            raise SystemExit("Complete both noiseless controls first: " + directory.name)
        config = json.loads((directory/"config_resolved.json").read_text())["config"]
        assert config["job"]["smoke"] == (sys.argv[2] == "1"), "smoke/formal prefix mismatch"
        assert config["device"] == sys.argv[3], "prerequisite device differs"
        for item in json.loads((directory/"code_manifest.json").read_text()):
            assert hashlib.sha256((root/pathlib.Path(item["path"])).read_bytes()).hexdigest() == item["sha256"], "source changed; use a new prefix"
PY
}

run_audit() {
    local d041_scope="$1" d041_output="$D041_PREFIX"
    if [[ "$d041_scope" != all ]]; then d041_output="${D041_PREFIX}_${d041_scope//-/_}"; fi
    if [[ "$D041_RESUME" == 1 && -f "results/$d041_output/audit.json" ]]; then
        echo "Existing audit retained: results/$d041_output/audit.json"
    else
        python scripts/analyze_d041_diagnostic.py --prefix "$D041_PREFIX" --scope "$d041_scope"
    fi
}

case "$D041_PHASE" in
    narrow-noiseless)
        noiseless_band narrow nb
        run_audit narrow-noiseless ;;
    wide-noiseless)
        require_noiseless_controls nb
        noiseless_band wide wb
        run_audit noiseless ;;
    noiseless|all)
        noiseless_band narrow nb
        noiseless_band wide wb
        run_audit noiseless
        if [[ "$D041_PHASE" == all ]]; then
            require_noiseless_controls
            evaluate_band narrow nb noisy
            evaluate_band wide wb noisy
            run_audit all
        fi ;;
    narrow-noisy|wide-noisy|noisy)
        require_noiseless_controls
        if [[ "$D041_PHASE" != wide-noisy ]]; then evaluate_band narrow nb noisy; fi
        if [[ "$D041_PHASE" != narrow-noisy ]]; then evaluate_band wide wb noisy; fi
        if [[ "$D041_PHASE" == noisy ]]; then run_audit all; fi ;;
esac
