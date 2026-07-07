#!/bin/bash
set -euo pipefail

WORK_DIR="work_dir_sweeps"
LOAD_FROM="weights/UCOD_DPL_dinov2.safetensors"
PSEUDO_INDEX="datasets/cache/pseudo_label_cache/TR-CAMO+TR-COD10K/index.json"

while getopts "w:m:" opt; do
  case "$opt" in
    w) WORK_DIR="$OPTARG" ;;
    m) LOAD_FROM="$OPTARG" ;;
    ?) echo "Usage: $0 [-w work_dir] [-m warm_start_checkpoint]"; exit 1 ;;
  esac
done

CONFIGS=(
  "configs/uscod/adf_sweeps/ADF_dinov2_balanced.py"
  "configs/uscod/adf_sweeps/ADF_dinov2_confident.py"
  "configs/uscod/adf_sweeps/ADF_dinov2_region.py"
  "configs/uscod/adf_sweeps/ADF_dinov2_temporal.py"
)

export PYTHONPATH=./
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3

if [ ! -f "$PSEUDO_INDEX" ]; then
  echo "Missing pseudo-label cache: $PSEUDO_INDEX"
  echo "Generate it first:"
  echo "  python generate_pseudo_label.py --dataset 'TR-CAMO+TR-COD10K'"
  exit 1
fi

LOAD_ARGS=()
if [ -n "$LOAD_FROM" ]; then
  if [ ! -f "$LOAD_FROM" ]; then
    echo "Warm-start checkpoint not found: $LOAD_FROM"
    echo "Continuing from scratch. Pass -m '' to silence this warning."
  else
    LOAD_ARGS=(--load_from "$LOAD_FROM")
  fi
fi

for CONFIG in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Running ADF sweep config: $CONFIG"
  echo "Work dir: $WORK_DIR"
  if [ "${#LOAD_ARGS[@]}" -gt 0 ]; then
    echo "Warm start: ${LOAD_ARGS[1]}"
  else
    echo "Warm start: none"
  fi
  echo "============================================================"
  python train.py --config "$CONFIG" --work_dir "$WORK_DIR" "${LOAD_ARGS[@]}"
done
