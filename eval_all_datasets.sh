#!/bin/bash
# Script to evaluate ADF model on all 4 test datasets
# This will generate results for the full metrics table

# Configuration
CONFIG="configs/uscod/ADF_dinov2.py"
CHECKPOINT="work_dir/uscod/ADF_dinov2/UCOD-ADF/ckp/epoch25.pth"

# Check if checkpoint exists
if [ ! -e "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Available checkpoints:"
    ls -lh work_dir/uscod/ADF_dinov2/UCOD-ADF/ckp/
    exit 1
fi

echo "=========================================="
echo "Starting evaluation on all test datasets"
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
echo "All evaluations completed!"
echo "=========================================="
