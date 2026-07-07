#!/bin/bash
# Script to evaluate BASELINE (no agents) model on all 4 test datasets

# Configuration — uses baseline config (no agentic AI)
CONFIG="configs/uscod/UCOD-DPL_dinov2.py"
CHECKPOINT="work_dir/uscod/UCOD-DPL_dinov2/UCOD-DPL_dinov2/ckp/epoch25.pth"

# Check if checkpoint exists
if [ ! -e "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Available checkpoints:"
    ls -lh work_dir/uscod/UCOD-DPL_dinov2/UCOD-DPL_dinov2/ckp/ 2>/dev/null || echo "No ckp directory found"
    exit 1
fi

echo "=========================================="
echo "Starting BASELINE evaluation (no agents)"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo ""

# Array of test datasets
DATASETS=("CHAMELEON" "TE-CAMO" "TE-COD10K" "NC4K")

# Run evaluation on each dataset
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating on: $DATASET"
    echo "=========================================="
    
    python eval.py \
        --config "$CONFIG" \
        --load_from "$CHECKPOINT" \
        --opts dataset_cfg.valset_cfg.DATASET "$DATASET"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Evaluation failed for $DATASET"
    else
        echo "SUCCESS: Evaluation completed for $DATASET"
    fi
done

echo ""
echo "=========================================="
echo "All baseline evaluations completed!"
echo "=========================================="
